//! Value log for WiscKey optimization.
//!
//! This module implements value separation, a key optimization in WiscKey-style LSM-trees.
//! Instead of storing values inline with keys in sorted tables, large values are stored
//! separately in a value log, and sorted tables only store references (batch ID, offset).
//!
//! ## Benefits
//!
//! - **Reduced Write Amplification**: During compaction, only keys and references are moved
//! - **Improved Scan Performance**: Sorted tables are smaller and more cache-friendly
//! - **Efficient Value Deletion**: Values can be garbage collected independently
//!
//! ## Architecture
//!
//! - **ValueLog**: Main interface for storing and retrieving values
//! - **ValueBatch**: Fixed-size batches of values stored on disk
//! - **ValueIndex**: Tracks which portions of batches contain live values for GC
//!
//! ## Value Storage
//!
//! Values are grouped into batches (files) and referenced by (batch_id, offset) pairs.
//! Each batch has a fixed size and is cached in memory for fast access.
//!
//! ## Garbage Collection
//!
//! When a batch's live data falls below `GARBAGE_COLLECT_THRESHOLD` (20%), it becomes
//! eligible for GC. Live values are copied to new batches and the old batch is deleted.

use std::num::NonZeroUsize;
use std::sync::Arc;

use tokio::sync::Mutex;

use crate::Error;

use lru::LruCache;

use crate::Params;
use crate::disk;
use crate::manifest::Manifest;

/// Byte offset within a value batch.
pub type ValueOffset = u32;

/// Unique identifier for a value batch file.
pub type ValueBatchId = u64;

/// Minimum valid value batch ID.
pub const MIN_VALUE_BATCH_ID: ValueBatchId = 1;

/// Complete value reference: (batch_id, offset).
pub type ValueId = (ValueBatchId, ValueOffset);

/// Number of shards for the value batch cache.
const NUM_SHARDS: NonZeroUsize = NonZeroUsize::new(16).unwrap();

/// Garbage collection threshold: when live data < 20%, batch is eligible for GC.
pub const GARBAGE_COLLECT_THRESHOLD: f64 = 0.2;

/// LRU cache for a shard of value batches.
type BatchShard = LruCache<ValueBatchId, Arc<ValueBatch>>;

#[cfg(test)]
mod tests;

mod index;
pub use index::{MIN_VALUE_INDEX_PAGE_ID, ValueIndex, ValueIndexPageId};

mod batch;
use batch::ValueBatch;
pub use batch::ValueBatchBuilder;

use crate::EntryList;
use crate::wal::{LogEntry, WriteAheadLog};

/// Manager for the value log in WiscKey mode.
///
/// Stores values separately from sorted tables and provides:
/// - Value storage and retrieval by (batch_id, offset)
/// - Garbage collection for deleted values
/// - Value index for tracking live data
/// - Sharded LRU cache for value batches
pub struct ValueLog {
    /// Write-ahead log for durable index updates.
    wal: Arc<WriteAheadLog>,

    /// Tracks which portions of value batches contain live values.
    index: ValueIndex,

    /// Sharded LRU caches for value batch files.
    batch_caches: Vec<Mutex<BatchShard>>,

    /// Database configuration parameters.
    params: Arc<Params>,

    /// Manifest for ID generation and metadata.
    manifest: Arc<Manifest>,
}

/// Reference to a value within a value batch.
///
/// Provides zero-copy access to value data without loading the entire batch.
pub struct ValueRef {
    /// The batch containing this value.
    batch: Arc<ValueBatch>,

    /// Byte offset within the batch where the value starts.
    offset: usize,

    /// Length of the value in bytes.
    length: usize,
}

impl ValueRef {
    /// Returns a slice of the value data.
    ///
    /// Provides zero-copy access to the value bytes without copying.
    pub fn get_value(&self) -> &[u8] {
        &self.batch.get_value_data()[self.offset..self.offset + self.length]
    }
}

impl ValueLog {
    /// Initializes the sharded LRU caches for value batches.
    ///
    /// Splits the available file handles across multiple shards to reduce
    /// lock contention during concurrent access.
    ///
    /// # Arguments
    ///
    /// * `params` - Database configuration parameters
    fn init_caches(params: &Params) -> Vec<Mutex<BatchShard>> {
        let max_value_files = NonZeroUsize::new(params.max_open_files / 2)
            .expect("Max open files needs to be greater than 2");

        let shard_size = NonZeroUsize::new(max_value_files.get() / NUM_SHARDS)
            .expect("Not enough open files to support the number of shards");

        (0..NUM_SHARDS.get())
            .map(|_| Mutex::new(BatchShard::new(shard_size)))
            .collect()
    }

    /// Creates a new value log with an empty index.
    ///
    /// # Arguments
    ///
    /// * `wal` - Write-ahead log for durable index updates
    /// * `params` - Database configuration parameters
    /// * `manifest` - Manifest for ID generation and metadata
    ///
    /// # Errors
    ///
    /// Returns an error if the initial index page cannot be created.
    pub async fn new(
        wal: Arc<WriteAheadLog>,
        params: Arc<Params>,
        manifest: Arc<Manifest>,
    ) -> Result<Self, Error> {
        let batch_caches = Self::init_caches(&params);
        let index = ValueIndex::new(params.clone(), manifest.clone()).await?;

        Ok(Self {
            wal,
            index,
            params,
            manifest,
            batch_caches,
        })
    }

    /// Opens an existing value log from disk.
    ///
    /// Loads the value index and cleans up any batches marked for deletion
    /// during recovery.
    ///
    /// # Arguments
    ///
    /// * `wal` - Write-ahead log for durable index updates
    /// * `params` - Database configuration parameters
    /// * `manifest` - Manifest containing value log metadata
    /// * `index` - Pre-loaded value index
    /// * `to_delete` - List of batch IDs to delete during recovery
    ///
    /// # Errors
    ///
    /// Returns an error if any batch file cannot be deleted.
    pub async fn open(
        wal: Arc<WriteAheadLog>,
        params: Arc<Params>,
        manifest: Arc<Manifest>,
        index: ValueIndex,
        to_delete: Vec<ValueBatchId>,
    ) -> Result<Self, Error> {
        let batch_caches = Self::init_caches(&params);
        let obj = Self {
            wal,
            index,
            params,
            manifest,
            batch_caches,
        };

        for batch_id in to_delete.into_iter() {
            obj.remove_batch_from_disk(batch_id).await?;
        }

        Ok(obj)
    }

    /// Marks a value as deleted and performs garbage collection if needed.
    ///
    /// Updates the value index to mark the value as deleted, then checks if
    /// the batch can be removed or should be compacted.
    ///
    /// # Arguments
    ///
    /// * `vid` - The value ID (batch_id, offset) to mark as deleted
    ///
    /// # Returns
    ///
    /// A list of key-value pairs to reinsert if the batch was compacted,
    /// or an empty list otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if the index update or WAL write fails.
    #[tracing::instrument(skip(self))]
    pub async fn mark_value_deleted(&self, vid: ValueId) -> Result<EntryList, Error> {
        let (page_id, page_offset) = self.index.mark_value_as_deleted(vid).await?;
        self.wal
            .store([LogEntry::DeleteValue(page_id, page_offset)].into_iter())
            .await?;

        let res = if self.try_to_remove(page_id).await? {
            vec![]
        } else {
            self.try_to_compact(page_id).await?.unwrap_or_else(Vec::new)
        };

        Ok(res)
    }

    /// Attempts to delete a batch if it contains no live values.
    ///
    /// Checks if all values in the batch have been deleted, and if so,
    /// removes the batch file from disk and updates the index.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The batch to potentially remove
    ///
    /// # Returns
    ///
    /// `true` if the batch was removed, `false` if it still contains live values.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be deleted or index update fails.
    #[tracing::instrument(skip(self))]
    async fn try_to_remove(&self, batch_id: ValueBatchId) -> Result<bool, Error> {
        log::trace!("Checking if value batch #{batch_id} can be removed");

        let num_active = self.index.count_active_entries(batch_id).await;

        // Can only remove if no values in this batch are active
        if num_active > 0 {
            return Ok(false);
        }

        log::trace!("Deleting empty batch #{batch_id}");
        self.index.mark_batch_as_deleted(batch_id).await?;

        // Hold lock so nobody else messes with the file while we do this
        self.remove_batch_from_disk(batch_id).await?;
        Ok(true)
    }

    /// Removes a batch file from disk and updates internal state.
    ///
    /// Deletes the batch file, removes it from the cache, and updates the
    /// minimum batch ID in the manifest if applicable.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The batch to remove from disk
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be deleted.
    async fn remove_batch_from_disk(&self, batch_id: ValueBatchId) -> Result<(), Error> {
        let shard_id = Self::batch_to_shard_id(batch_id);
        let mut cache = self.batch_caches[shard_id].lock().await;
        let fpath = self.get_batch_file_path(&batch_id);
        disk::remove_file(&fpath)
            .await
            .map_err(|err| Error::from_io_error("Failed to remove value log batch", err))?;
        cache.pop(&batch_id);

        // Can we remove entries entirely?
        let min_batch = self
            .manifest
            .get_minimum_value_batch()
            .await
            .max(MIN_VALUE_BATCH_ID);

        // We can only completely remove batches starting from the oldest one
        // Instead, we "empty" the batch, reducing its size to a single on-disk page
        if batch_id > min_batch {
            return Ok(());
        }

        let most_recent = self.manifest.get_most_recent_value_batch_id().await;
        let mut new_minimum = batch_id;

        while new_minimum < most_recent {
            if self.index.count_active_entries(batch_id).await > 0 {
                break;
            }
            new_minimum += 1;
        }

        log::debug!("Full removed {} value batches", new_minimum - batch_id + 1);
        self.manifest.set_minimum_value_batch_id(new_minimum).await;

        Ok(())
    }

    /// Checks if a batch should be compacted and returns entries to reinsert.
    ///
    /// If the batch's live data ratio falls below the garbage collection threshold,
    /// extracts the live entries for reinsertion and marks the batch as compacted.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The batch to check for compaction
    ///
    /// # Returns
    ///
    /// `Some(entries)` if the batch was compacted, `None` otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch cannot be loaded or index update fails.
    #[tracing::instrument(skip(self))]
    async fn try_to_compact(&self, batch_id: ValueBatchId) -> Result<Option<EntryList>, Error> {
        log::trace!("Checking if value batch #{batch_id} should be compacted (reinserted)");

        let batch = self.get_batch(batch_id).await?;
        let num_entries = batch.total_num_values() as usize;
        let num_active = self.index.count_active_entries(batch_id).await;
        let active_ratio = (num_active * 100) / (num_entries * 100);

        if active_ratio < 25 {
            log::trace!("Re-inserting sparse value batch #{batch_id}");
            let offsets = self.index.get_active_entries(batch_id).await;
            self.index.mark_batch_as_compacted(batch_id).await?;

            Ok(Some(batch.get_entries(&offsets)))
        } else {
            Ok(None)
        }
    }

    /// Maps a batch ID to its cache shard.
    ///
    /// Uses modulo to distribute batches across shards for reduced lock contention.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The batch ID to map
    #[inline]
    fn batch_to_shard_id(batch_id: ValueBatchId) -> usize {
        (batch_id as usize) % NUM_SHARDS
    }

    /// Returns the file path for a value batch.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The batch ID
    #[inline]
    fn get_batch_file_path(&self, batch_id: &ValueBatchId) -> std::path::PathBuf {
        self.params.db_path.join(format!("val{batch_id:08}.data"))
    }

    /// Creates a new value batch builder.
    ///
    /// Generates a unique batch ID and returns a builder for adding values.
    pub async fn make_batch(&self) -> ValueBatchBuilder<'_> {
        let identifier = self.manifest.generate_next_value_batch_id().await;
        ValueBatchBuilder::new(identifier, self)
    }

    /// Retrieves a value batch, loading from disk if not cached.
    ///
    /// Checks the appropriate shard cache first, then loads from disk if needed.
    ///
    /// # Arguments
    ///
    /// * `identifier` - The batch ID to retrieve
    ///
    /// # Errors
    ///
    /// Returns an error if the batch file cannot be read.
    #[tracing::instrument(skip(self))]
    async fn get_batch(&self, identifier: ValueBatchId) -> Result<Arc<ValueBatch>, Error> {
        let shard_id = Self::batch_to_shard_id(identifier);
        let mut cache = self.batch_caches[shard_id].lock().await;

        if let Some(batch) = cache.get(&identifier) {
            Ok(batch.clone())
        } else {
            log::trace!("Loading value batch #{identifier} from disk");

            let data = disk::read(&self.get_batch_file_path(&identifier), 0)
                .await
                .map_err(|err| Error::from_io_error("Failed to read value log batch", err))?;

            let obj = Arc::new(ValueBatch::from_existing(data));
            cache.put(identifier, obj.clone());

            Ok(obj)
        }
    }

    /// Returns a reference to a value by its ID.
    ///
    /// Loads the batch if needed and creates a zero-copy reference to the value.
    ///
    /// # Arguments
    ///
    /// * `value_ref` - The value ID (batch_id, offset)
    ///
    /// # Errors
    ///
    /// Returns an error if the batch cannot be loaded.
    pub async fn get_ref(&self, value_ref: ValueId) -> Result<ValueRef, Error> {
        log::trace!("Getting value at {value_ref:?}");

        let (id, offset) = value_ref;
        let batch = self.get_batch(id).await?;

        Ok(ValueBatch::get_ref(batch, offset))
    }

    /// Flushes all dirty index pages to disk.
    ///
    /// Ensures all pending index updates are persisted.
    ///
    /// # Errors
    ///
    /// Returns an error if any index page cannot be written.
    pub async fn flush(&self) -> Result<(), Error> {
        self.index.flush().await
    }
}
