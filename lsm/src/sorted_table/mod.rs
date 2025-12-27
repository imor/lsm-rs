//! Sorted tables (SSTables) for LSM-tree storage.
//!
//! This module implements sorted tables, which are immutable, on-disk data structures that
//! store key-value entries in sorted order. Sorted tables are the fundamental building blocks
//! of LSM-tree levels.
//!
//! ## Architecture
//!
//! Each sorted table consists of:
//! - **Data blocks**: Store the actual key-value entries with prefix compression
//! - **Index block**: Maps key ranges to data block IDs for efficient lookups
//! - **Metadata**: Min/max keys, size, and compaction state
//!
//! ## Level Organization
//!
//! - **Level 0**: Tables may overlap (created from memtable flushes)
//! - **Level 1+**: Tables are non-overlapping within each level
//!
//! ## Compaction
//!
//! Sorted tables support both size-based and seek-based compaction triggers:
//! - Size-based: Triggered when a level exceeds its size threshold
//! - Seek-based: Triggered when a table has been accessed too many times
//!
//! ## Immutability
//!
//! Once created, sorted tables are immutable. Updates create new tables during compaction.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering as AtomicOrdering};

use crate::data_blocks::{DataBlock, DataBlocks, DataEntry};
use crate::index_blocks::IndexBlock;
use crate::{Error, Params};

mod iterator;
pub use iterator::{InternalIterator, TableIterator};

mod builder;
pub use builder::TableBuilder;

#[cfg(test)]
mod tests;

/// Unique identifier for a sorted table.
pub type TableId = u64;

/// An immutable sorted table (SSTable) storing key-value entries on disk.
///
/// Sorted tables are the fundamental storage units in LSM-trees. Each table contains an
/// ordered collection of key-value entries organized into data blocks, with an index block
/// for efficient lookups.
///
/// ## Overlap Constraints
///
/// - **Level 0**: Tables may have overlapping key ranges (created from memtable flushes)
/// - **Level 1+**: Tables within a level have non-overlapping key ranges
///
/// ## Compaction
///
/// Tables track access counts to support seek-based compaction, which moves frequently
/// accessed tables to lower levels to improve read performance.
///
/// ## Thread Safety
///
/// The `being_compacted` flag uses atomic operations to ensure only one compaction task
/// processes this table at a time.
pub struct SortedTable {
    /// Unique identifier for this table.
    identifier: TableId,
    
    /// Index block containing metadata (min/max keys, size) and block index for lookups.
    index: IndexBlock,
    
    /// Manager for loading and caching this table's data blocks.
    data_blocks: Arc<DataBlocks>,
    
    /// Atomic flag indicating if this table is currently being compacted.
    being_compacted: AtomicBool,
    
    /// Remaining seeks before compaction is triggered. Decrements on each `get()` call.
    /// Only used when `seek_based_compaction` is enabled. When <= 0, compaction should
    /// be triggered to improve read performance.
    num_seeks_compaction_threshold: AtomicI32,
}

impl SortedTable {
    /// Loads an existing sorted table from disk.
    ///
    /// Reads the index block to retrieve table metadata and initializes the compaction
    /// threshold based on table size if seek-based compaction is enabled.
    ///
    /// # Arguments
    ///
    /// * `identifier` - The unique ID of the table to load
    /// * `data_blocks` - Manager for data block operations
    /// * `params` - Database configuration parameters
    ///
    /// # Errors
    ///
    /// Returns an error if the index block cannot be loaded from disk.
    pub async fn load(
        identifier: TableId,
        data_blocks: Arc<DataBlocks>,
        params: &Params,
    ) -> Result<Self, Error> {
        let index = IndexBlock::load(params, identifier).await?;

        let num_seeks_compaction_threshold = if let Some(count) = params.seek_based_compaction {
            ((index.get_size() / 1024).max(1) as i32) * (count as i32)
        } else {
            0
        };

        Ok(Self {
            identifier,
            index,
            data_blocks,
            num_seeks_compaction_threshold: AtomicI32::new(num_seeks_compaction_threshold),
            being_compacted: AtomicBool::new(false),
        })
    }

    /// Returns `true` if this table has reached its seek threshold and should be compacted.
    ///
    /// The threshold is based on table size and decrements with each `get()` call. When
    /// the count reaches zero or below, the table should be compacted to improve read
    /// performance. Only meaningful when `seek_based_compaction` is enabled.
    pub fn has_maximum_seeks(&self) -> bool {
        self.num_seeks_compaction_threshold
            .load(AtomicOrdering::SeqCst)
            <= 0
    }

    /// Attempts to mark this table as being compacted.
    ///
    /// Uses atomic compare-exchange to ensure only one task can compact this table at a time.
    ///
    /// # Returns
    ///
    /// * `true` - If the table was successfully marked for compaction
    /// * `false` - If another task is already compacting this table
    pub fn start_compaction(&self) -> bool {
        let order = AtomicOrdering::SeqCst;
        let result = self
            .being_compacted
            .compare_exchange(false, true, order, order);

        result.is_ok()
    }

    /// Clears the compaction flag after a failed compaction attempt.
    ///
    /// Should be called when compaction fails (e.g., due to lock contention) to allow
    /// future compaction attempts.
    ///
    /// # Panics
    ///
    /// Panics if the compaction flag was not set, indicating a programming error.
    pub fn stop_compaction(&self) {
        let prev = self.being_compacted.swap(false, AtomicOrdering::SeqCst);
        assert!(prev, "Compaction flag was not set!");
    }

    /// Returns the unique identifier of this table.
    pub fn get_id(&self) -> TableId {
        self.identifier
    }

    /// Returns the total size of this table in bytes.
    ///
    /// This includes all data blocks but not the index block overhead.
    pub fn get_size(&self) -> usize {
        self.index.get_size()
    }

    /// Returns the smallest key in this table.
    ///
    /// This is the minimum key across all entries in all data blocks.
    pub fn get_min(&self) -> &[u8] {
        self.index.get_min()
    }

    /// Returns the largest key in this table.
    ///
    /// This is the maximum key across all entries in all data blocks.
    pub fn get_max(&self) -> &[u8] {
        self.index.get_max()
    }

    /// Retrieves the entry for the specified key from this table.
    ///
    /// Uses the index block to identify the relevant data block via binary search,
    /// then searches within that block. Decrements the seek counter for compaction tracking.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to search for
    ///
    /// # Returns
    ///
    /// * `Some(DataEntry)` - If the key exists in this table
    /// * `None` - If the key is not found or is outside this table's key range
    #[tracing::instrument(skip(self, key))]
    pub async fn get(&self, key: &[u8]) -> Option<DataEntry> {
        self.num_seeks_compaction_threshold
            .fetch_sub(1, AtomicOrdering::Relaxed);

        let block_id = self.index.binary_search(key)?;
        let block = self.data_blocks.get_block(&block_id).await;

        DataBlock::get_by_key(&block, key)
    }

    /// Checks if this table's key range overlaps with the specified range.
    ///
    /// Used during compaction to determine which tables need to be merged. An overlap
    /// occurs if any part of the table's key range intersects with the query range.
    ///
    /// # Arguments
    ///
    /// * `min` - Minimum key of the query range (inclusive)
    /// * `max` - Maximum key of the query range (inclusive)
    ///
    /// # Returns
    ///
    /// `true` if this table contains any keys in the range `[min, max]`, `false` otherwise.
    #[inline(always)]
    pub fn overlaps(&self, min: &[u8], max: &[u8]) -> bool {
        self.get_max() >= min && self.get_min() <= max
    }
}
