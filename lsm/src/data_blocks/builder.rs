//! Data Block Builder for Creating Sorted Blocks
//!
//! This module provides the [`DataBlockBuilder`] type for incrementally constructing data
//! blocks during SSTable creation. The builder handles prefix compression, restart point
//! management, and optional bloom filter construction.
//!
//! # Overview
//!
//! The `DataBlockBuilder` is used during memtable flushes and compaction operations to
//! create immutable data blocks containing sorted key-value entries. It optimizes storage
//! through prefix compression while maintaining efficient lookup capabilities via restart
//! points.
//!
//! # Building Process
//!
//! Blocks are built incrementally by adding entries in sorted key order:
//!
//! 1. **Create builder**: Initialize with `DataBlockBuilder::new()`
//! 2. **Add entries**: Call `add_entry()` for each key-value pair in sorted order
//! 3. **Finalize**: Call `finish()` to complete the block, write to disk, and cache
//!
//! # Prefix Compression
//!
//! The builder automatically implements prefix compression to reduce storage overhead:
//!
//! - Compares each key with the previous key to find the common prefix
//! - Stores only the unique suffix along with the prefix length
//! - At restart intervals, stores full keys (prefix_len=0) for binary search
//!
//! **Example compression:**
//! ```text
//! Entry 0 (restart): "database:users:1001" → full key, 19 bytes
//! Entry 1:           "database:users:1002" → prefix=17, suffix="02", 2 bytes
//! Entry 2:           "database:users:1003" → prefix=17, suffix="03", 2 bytes
//! ```
//!
//! This can provide significant space savings for keys with common prefixes like
//! timestamps, user IDs, or hierarchical namespaces.
//!
//! # Restart Points
//!
//! At regular intervals (configured by `block_restart_interval`), the builder creates
//! restart points:
//!
//! - Stores a full, uncompressed key
//! - Records the byte offset in the restart list
//! - Enables binary search in the finished block
//!
//! **Trade-offs:**
//! - Lower interval (e.g., 8): Faster lookups, more space overhead
//! - Higher interval (e.g., 32): Better compression, slower lookups
//!
//! The default interval of 16 provides a good balance for most workloads.
//!
//! # Bloom Filters
//!
//! When the `bloom-filters` feature is enabled, the builder constructs a bloom filter
//! alongside the block data:
//!
//! - Each key is added to the bloom filter via `bloom_filter.set(key)`
//! - The filter is serialized into the block header
//! - Allows O(1) negative lookups ("definitely not present")
//! - Small false positive rate ("might be present")
//!
//! Bloom filters are especially valuable for blocks that are rarely accessed, as they
//! can avoid disk I/O entirely for non-existent keys.
//!
//! # Block Finalization
//!
//! The `finish()` method completes block construction:
//!
//! 1. **Generate ID**: Obtains a unique block identifier from the manifest
//! 2. **Write header**: Populates the DataBlockHeader with metadata
//! 3. **Append restart list**: Adds the array of restart point offsets
//! 4. **Create DataBlock**: Constructs the in-memory representation
//! 5. **Write to disk**: Persists the complete block data
//! 6. **Cache block**: Stores in the LRU cache for immediate access
//!
//! # Memory Management
//!
//! The builder accumulates data in memory until `finish()` is called. For large blocks:
//! - Monitor `current_size()` to track memory usage
//! - Finish blocks before they exceed target size limits
//! - The parent TableBuilder manages block size limits automatically
//!
//! # WiscKey Mode
//!
//! The builder supports two storage modes:
//!
//! **Vanilla Mode** (default):
//! - Values stored inline with keys in the block
//! - Entry format: header + key suffix + value data
//!
//! **WiscKey Mode** (`wisckey` feature):
//! - Values stored separately in value log
//! - Entry format: header + key suffix (no value data)
//! - Header contains value reference (batch ID, offset)
//! - Significantly reduces write amplification during compaction
//!
//! # Error Handling
//!
//! The builder's `finish()` method can fail if:
//! - Disk write operations fail (I/O errors)
//! - File system is out of space
//! - Permission issues prevent file creation
//!
//! On failure, the block is not added to the cache and the error is propagated to
//! the caller for appropriate handling (typically causing compaction/flush to retry).
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use lsm::data_blocks::{DataBlockBuilder, DataBlocks};
//!
//! // Create a builder
//! let data_blocks = Arc::new(DataBlocks::new(params, manifest));
//! let mut builder = DataBlockBuilder::new(data_blocks);
//!
//! // Add entries in sorted order
//! builder.add_entry(
//!     PrefixedKey::new(0, b"key1".to_vec()),
//!     b"key1",
//!     seq_number,
//!     PUT_OP,
//!     b"value1"
//! );
//!
//! builder.add_entry(
//!     PrefixedKey::new(3, b"2".to_vec()),  // shares "key" prefix
//!     b"key2",
//!     seq_number,
//!     PUT_OP,
//!     b"value2"
//! );
//!
//! // Finalize and write to disk
//! let block_id = builder.finish().await?;
//! ```

use cfg_if::cfg_if;

use std::sync::Arc;

use crate::manifest::SeqNumber;
use crate::{Error, disk};

use zerocopy::IntoBytes;

use super::block::{DataBlockHeader, EntryHeader};
use super::{DataBlock, DataBlockId, DataBlocks, PrefixedKey};

#[cfg(feature = "bloom-filters")]
use bloomfilter::Bloom;

#[cfg(feature = "bloom-filters")]
use super::block::{BLOOM_HEADER_SIZE, BLOOM_ITEM_COUNT, BLOOM_LENGTH};

#[cfg(feature = "wisckey")]
use crate::data_blocks::ValueId;

/// A builder for constructing data blocks with prefix-compressed keys.
///
/// `DataBlockBuilder` incrementally constructs a data block by adding entries one at a time.
/// It handles prefix compression, maintains a restart list for efficient lookups, and
/// optionally builds a bloom filter for fast key existence checks.
///
/// ## Building Process
///
/// 1. Create a new builder with `new()`
/// 2. Add entries sequentially with `add_entry()` (must be in sorted order)
/// 3. Call `finish()` to finalize the block, write it to disk, and cache it
///
/// ## Prefix Compression
///
/// Consecutive keys often share common prefixes. The builder stores only the unique suffix
/// for each key along with the length of the shared prefix from the previous key.
///
/// ## Restart Points
///
/// At regular intervals (controlled by `block_restart_interval`), the builder stores a full
/// key (no compression) and records its offset in the restart list. This enables binary
/// search during lookups.
pub struct DataBlockBuilder {
    /// Reference to the parent DataBlocks collection for accessing configuration and
    /// managing the block cache.
    data_blocks: Arc<DataBlocks>,

    /// In-memory buffer containing the block data being built. Includes the header
    /// (filled in at finish), entries, and eventually the restart list.
    data: Vec<u8>,

    /// The index of the next entry to be added. Also represents the current number of
    /// entries in this block builder.
    position: u32,

    /// List of byte offsets pointing to restart entries (entries with full keys, no prefix
    /// compression). Enables binary search in the finished block.
    restart_list: Vec<u32>,

    /// Probabilistic data structure for quick key existence checks. Allows fast negative
    /// lookups without scanning entries. Only available with the "bloom-filters" feature.
    #[cfg(feature = "bloom-filters")]
    bloom_filter: Bloom<[u8]>,
}

impl DataBlockBuilder {
    /// Creates a new empty `DataBlockBuilder`.
    ///
    /// Initializes the builder with space reserved for the block header, which will be
    /// written when `finish()` is called. Also creates an empty bloom filter if the
    /// "bloom-filters" feature is enabled.
    ///
    /// # Arguments
    ///
    /// * `data_blocks` - Reference to the parent DataBlocks collection
    ///
    /// # Panics
    ///
    /// Panics if the bloom filter cannot be created (only with "bloom-filters" feature).
    #[tracing::instrument(skip(data_blocks))]
    pub(super) fn new(data_blocks: Arc<DataBlocks>) -> Self {
        // Reserve space for the header
        let data = vec![0u8; std::mem::size_of::<DataBlockHeader>()];

        Self {
            data_blocks,
            data,
            position: 0,
            restart_list: vec![],
            #[cfg(feature = "bloom-filters")]
            bloom_filter: Bloom::new(BLOOM_LENGTH, BLOOM_ITEM_COUNT)
                .expect("Failed to create bloom filter"),
        }
    }

    /// Adds a key-value entry to the block being built.
    ///
    /// Entries must be added in sorted key order. The method handles prefix compression
    /// by storing only the unique suffix of the key. At restart intervals, it records
    /// the offset in the restart list for efficient binary search.
    ///
    /// # Arguments
    ///
    /// * `key` - Prefixed key containing the prefix length and unique suffix
    /// * `full_key` - The complete uncompressed key (used for bloom filter)
    /// * `seq_number` - Sequence number for versioning and MVCC
    /// * `entry_type` - Type of entry (insert, delete, etc.)
    /// * `entry_data` - The value data (non-WiscKey mode only)
    /// * `value_ref` - Reference to value location (batch ID and offset, WiscKey mode only)
    ///
    /// # Panics
    ///
    /// Panics if a restart point entry has a non-zero prefix length (should have full key).
    pub fn add_entry(
        &mut self,
        mut key: PrefixedKey,
        full_key: &[u8],
        seq_number: SeqNumber,
        entry_type: u8,
        #[cfg(not(feature = "wisckey"))] entry_data: &[u8],
        #[cfg(feature = "wisckey")] value_ref: ValueId,
    ) {
        if self
            .position
            .is_multiple_of(self.data_blocks.params.block_restart_interval)
        {
            assert!(key.prefix_len == 0);
            self.restart_list.push(self.data.len() as u32);
        }

        cfg_if! {
            if #[cfg(feature="bloom-filters")] {
                self.bloom_filter.set(full_key);
            } else {
                let _ = full_key;
            }
        }

        let header = EntryHeader {
            prefix_len: key.prefix_len,
            suffix_len: key.suffix.len() as u32,
            seq_number,
            entry_type,
            #[cfg(feature = "wisckey")]
            value_batch: value_ref.0,
            #[cfg(feature = "wisckey")]
            value_offset: value_ref.1,
            #[cfg(not(feature = "wisckey"))]
            value_length: entry_data.len() as u64,
        };

        self.data.extend_from_slice(header.as_bytes());

        self.data.append(&mut key.suffix);

        #[cfg(not(feature = "wisckey"))]
        self.data.extend_from_slice(entry_data);

        self.position += 1;
    }

    /// Finalizes the block, writes it to disk, and caches it.
    ///
    /// This method completes the block building process by:
    /// 1. Generating a unique block ID
    /// 2. Writing the block header with metadata (entry count, restart list position, bloom filter)
    /// 3. Appending the restart list to the block data
    /// 4. Creating a `DataBlock` instance
    /// 5. Writing the complete block to disk
    /// 6. Caching the block in memory for fast access
    ///
    /// # Returns
    ///
    /// * `Ok(Some(DataBlockId))` - If the block contains entries and was successfully written
    /// * `Ok(None)` - If the builder is empty (no entries were added)
    /// * `Err(Error)` - If writing to disk fails
    ///
    /// # Errors
    ///
    /// Returns an error if the block cannot be written to disk.
    #[tracing::instrument(skip(self))]
    pub async fn finish(mut self) -> Result<Option<DataBlockId>, Error> {
        if self.position == 0 {
            return Ok(None);
        }

        let identifier = self
            .data_blocks
            .manifest
            .generate_next_data_block_id()
            .await;

        #[cfg(feature = "bloom-filters")]
        let bloom_filter: &[u8; BLOOM_LENGTH + BLOOM_HEADER_SIZE] =
            self.bloom_filter.as_slice().try_into().unwrap();

        let header = DataBlockHeader {
            #[cfg(feature = "bloom-filters")]
            bloom_filter: *bloom_filter,
            number_of_entries: self.position,
            restart_list_start: self.data.len() as u32,
        };

        // Write header
        self.data[..std::mem::size_of::<DataBlockHeader>()].copy_from_slice(header.as_bytes());

        // Write restart list
        for restart_offset in self.restart_list.drain(..) {
            self.data.extend_from_slice(restart_offset.as_bytes());
        }

        let block = Arc::new(DataBlock {
            data: self.data,
            num_entries: header.number_of_entries,
            restart_interval: self.data_blocks.params.block_restart_interval,
            restart_list_start: header.restart_list_start as usize,
            #[cfg(feature = "bloom-filters")]
            bloom_filter: self.bloom_filter,
        });
        let shard_id = DataBlocks::block_to_shard_id(identifier);

        // Store on disk before grabbing the lock
        let block_data = &block.data;
        let fpath = self.data_blocks.get_file_path(&identifier);

        disk::write(&fpath, block_data).await.map_err(|err| {
            Error::from_io_error(format!("Failed to write data block at `{fpath:?}`"), err)
        })?;

        self.data_blocks.block_caches[shard_id]
            .lock()
            .put(identifier, block);

        Ok(Some(identifier))
    }

    /// Returns the current size of the block being built in bytes.
    ///
    /// This includes the header, all entries added so far, but not the restart list
    /// (which is only appended during `finish()`). Useful for determining when to
    /// start a new block to keep blocks within size limits.
    pub fn current_size(&self) -> usize {
        self.data.len()
    }
}
