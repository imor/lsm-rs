//! Data blocks for LSM-tree sorted tables (SSTables).
//!
//! This module provides the core data structures and functionality for managing data blocks,
//! which are the fundamental storage units in LSM-tree sorted tables. Data blocks store
//! sorted key-value entries using prefix compression to minimize storage overhead.
//!
//! ## Architecture
//!
//! - **DataBlock**: In-memory representation of a data block with entries and restart list
//! - **DataBlockBuilder**: Builder for constructing new data blocks incrementally
//! - **DataBlocks**: Manager for the block cache and disk I/O operations
//! - **DataEntry**: Handle to a specific entry within a block
//!
//! ## WiscKey Support
//!
//! When the "wisckey" feature is enabled, data blocks store only keys and value references
//! (batch ID and offset) rather than the full values. This separates large values from
//! the sorted table structure for better performance.
//!
//! ## Caching
//!
//! Data blocks are cached in memory using a sharded LRU cache for better concurrency.
//! Cache size is controlled by the `max_open_files` parameter.

use std::num::NonZeroUsize;
use std::sync::Arc;

use parking_lot::Mutex;

use lru::LruCache;

use zerocopy::FromBytes;

#[cfg(feature = "wisckey")]
use crate::EntryRef;
#[cfg(not(feature = "wisckey"))]
use crate::EntryRef;
use crate::Params;
use crate::manifest::Manifest;
use crate::{WriteOp, disk};

mod builder;
pub use builder::DataBlockBuilder;

mod block;
pub use block::DataBlock;

use block::EntryHeader;

#[cfg(feature = "wisckey")]
use crate::values::{ValueId, ValueLog};

/// The unique identifier of a data block
pub type DataBlockId = u64;

/// The minimum valid data block identifier
pub const MIN_DATA_BLOCK_ID: DataBlockId = 1;

/// A key with prefix compression metadata.
///
/// This structure represents a key that has been prefix-compressed relative to the previous
/// key in a sorted sequence. It stores only the unique suffix along with the length of the
/// shared prefix from the previous key.
///
/// ## Example
///
/// ```text
/// Previous key: "user_12345"
/// Current key:  "user_12346"
/// PrefixedKey:  prefix_len=9, suffix="6"
///
/// Previous key: "user_12346"
/// Current key:  "user_99999"
/// PrefixedKey:  prefix_len=5, suffix="99999"
/// ```
///
/// At restart points (periodic full keys), `prefix_len` is 0 and `suffix` contains the
/// complete key.
#[derive(Debug)]
pub struct PrefixedKey {
    /// Number of bytes to reuse from the previous key's prefix.
    prefix_len: u32,

    /// The unique suffix of this key that differs from the previous key.
    suffix: Vec<u8>,
}

impl PrefixedKey {
    /// Creates a new `PrefixedKey` with the specified prefix length and suffix.
    ///
    /// # Arguments
    ///
    /// * `prefix_len` - Number of bytes to reuse from the previous key
    /// * `suffix` - The unique suffix bytes for this key
    pub fn new(prefix_len: usize, suffix: Vec<u8>) -> Self {
        Self {
            prefix_len: prefix_len as u32,
            suffix,
        }
    }
}

/// The type alias for the block cache, which is an LRU cache mapping DataBlockId to an Arc of DataBlock
type BlockCache = LruCache<DataBlockId, Arc<DataBlock>>;

/// The type of operation represented by a data entry.
///
/// LSM-trees handle both insertions and deletions. Deletions are represented as
/// tombstone entries rather than immediately removing the key.
pub enum DataEntryType {
    /// An insert or update operation.
    Put,

    /// A delete operation (tombstone marker).
    Delete,
}

/// A handle to a specific key-value entry within a data block.
///
/// `DataEntry` provides access to an entry's metadata (sequence number, type) and value
/// without copying the data. It holds a reference to the parent block and tracks the
/// entry's offset and length within the block's buffer.
#[derive(Clone)]
pub struct DataEntry {
    /// The block containing this entry.
    block: Arc<DataBlock>,

    /// Byte offset of this entry's header within the block's data buffer.
    offset: usize,

    /// Total length of this entry in bytes (header + key suffix + value/value reference).
    len: u32,
}

/// Result of a binary search operation within a data block.
enum SearchResult {
    /// An exact match was found at a restart point.
    ExactMatch(DataEntry),

    /// The key might be in the range between these byte offsets, requiring sequential scan.
    Range(u32, u32),
}

impl DataEntry {
    /// Returns a reference to this entry's header.
    ///
    /// The header contains metadata like prefix length, suffix length, sequence number,
    /// entry type, and value information.
    fn get_header(&self) -> &EntryHeader {
        let header_len = std::mem::size_of::<EntryHeader>();
        let header_data = &self.block.data[self.offset..self.offset + header_len];
        EntryHeader::ref_from_bytes(header_data).expect("Failed to read entry header")
    }

    /// Returns the sequence number of this entry.
    ///
    /// The sequence number is used for versioning and MVCC. Higher values indicate
    /// more recent writes.
    pub fn get_sequence_number(&self) -> u64 {
        self.get_header().seq_number
    }

    /// Returns the byte offset immediately after this entry.
    ///
    /// This value can be used as the starting offset for reading the next entry.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Returns the type of this entry (Put or Delete).
    ///
    /// # Panics
    ///
    /// Panics if the entry type is not a recognized operation.
    pub fn get_type(&self) -> DataEntryType {
        let header = self.get_header();

        if header.entry_type == WriteOp::PUT_OP {
            DataEntryType::Put
        } else if header.entry_type == WriteOp::DELETE_OP {
            DataEntryType::Delete
        } else {
            panic!("Unknown data entry type");
        }
    }

    /// Returns the value data for this entry (non-WiscKey mode only).
    ///
    /// For Put entries, returns a slice containing the value bytes stored inline with the key.
    /// For Delete entries (tombstones), returns `None`.
    ///
    /// # Panics
    ///
    /// Panics if the entry type is not a recognized operation.
    #[cfg(not(feature = "wisckey"))]
    pub fn get_value(&self) -> Option<&[u8]> {
        let header = self.get_header();
        let value_offset =
            self.offset + std::mem::size_of::<EntryHeader>() + (header.suffix_len as usize);

        if header.entry_type == WriteOp::PUT_OP {
            let end = value_offset + (header.value_length as usize);
            Some(&self.block.data[value_offset..end])
        } else if header.entry_type == WriteOp::DELETE_OP {
            None
        } else {
            panic!("Unknown write op");
        }
    }

    /// Returns the value reference for this entry (WiscKey mode only).
    ///
    /// Returns a `ValueId` (batch ID and offset) pointing to where the actual value is stored
    /// in the value log. For Delete entries, returns `None`.
    ///
    /// # Panics
    ///
    /// Panics if the entry type is not a recognized operation.
    #[cfg(feature = "wisckey")]
    pub fn get_value_id(&self) -> Option<ValueId> {
        let header = self.get_header();

        if header.entry_type == WriteOp::PUT_OP {
            Some((header.value_batch, header.value_offset))
        } else if header.entry_type == WriteOp::DELETE_OP {
            None
        } else {
            panic!("Unknown write op");
        }
    }

    /// Converts this data entry into an `EntryRef` for non-WiscKey mode.
    ///
    /// For Put entries, wraps this entry in an `EntryRef::SortedTable` variant.
    /// For Delete entries, returns `None`.
    #[cfg(not(feature = "wisckey"))]
    pub fn get_entry_ref(self) -> Option<EntryRef> {
        match self.get_type() {
            DataEntryType::Put => {
                let entry = EntryRef::SortedTable { entry: self };
                Some(entry)
            }
            DataEntryType::Delete => None,
        }
    }

    /// Converts this data entry into an `EntryRef` for WiscKey mode.
    ///
    /// For Put entries, retrieves the value reference from the value log and wraps both
    /// this entry and the value reference in an `EntryRef::SortedTable` variant.
    /// For Delete entries, returns `None`.
    ///
    /// # Arguments
    ///
    /// * `value_log` - The value log to retrieve the value reference from
    #[cfg(feature = "wisckey")]
    pub async fn get_entry_ref(self, value_log: &ValueLog) -> Option<EntryRef> {
        match self.get_type() {
            DataEntryType::Put => {
                let value_ref = value_log
                    .get_ref(self.get_value_id().unwrap())
                    .await
                    .unwrap();
                let entry = EntryRef::SortedTable {
                    entry: self,
                    value_ref,
                };
                Some(entry)
            }
            DataEntryType::Delete => None,
        }
    }
}

/// Manager for data blocks with caching and disk I/O.
///
/// `DataBlocks` maintains a sharded LRU cache of data blocks in memory and handles loading
/// blocks from disk on cache misses. It also provides factory methods for creating new blocks
/// through builders.
///
/// ## Sharding
///
/// The block cache is split into multiple shards to reduce lock contention. Each shard
/// has its own LRU cache and lock, allowing concurrent access to different blocks.
///
/// ## Disk Layout
///
/// Data blocks are stored as individual files with names like `key00000001.data` in the
/// database directory.
pub struct DataBlocks {
    /// Database configuration parameters.
    params: Arc<Params>,

    /// Sharded LRU caches for storing recently accessed data blocks in memory.
    block_caches: Vec<Mutex<BlockCache>>,

    /// Manifest for generating unique block IDs and tracking metadata.
    manifest: Arc<Manifest>,
}

impl DataBlocks {
    /// The number of shards to split the block cache into for better concurrency.
    const BLOCK_CACHE_NUM_SHARDS: NonZeroUsize = NonZeroUsize::new(64).unwrap();

    /// Creates a new `DataBlocks` manager with sharded LRU caches.
    ///
    /// The total cache capacity is split evenly across shards. Each shard gets
    /// `(max_open_files / 2) / BLOCK_CACHE_NUM_SHARDS` entries.
    ///
    /// # Arguments
    ///
    /// * `params` - Database configuration parameters
    /// * `manifest` - Manifest for block ID generation and metadata
    ///
    /// # Panics
    ///
    /// Panics if `max_open_files` is too small to support the number of shards.
    pub fn new(params: Arc<Params>, manifest: Arc<Manifest>) -> Self {
        let max_data_files = NonZeroUsize::new(params.max_open_files / 2)
            .expect("Max open files needs to be greater than 2");

        let shard_size =
            NonZeroUsize::new(max_data_files.get() / Self::BLOCK_CACHE_NUM_SHARDS.get())
                .expect("Not enough open files to support the number of shards");

        let mut block_caches = Vec::with_capacity(Self::BLOCK_CACHE_NUM_SHARDS.get());
        for _ in 0..Self::BLOCK_CACHE_NUM_SHARDS.get() {
            block_caches.push(Mutex::new(BlockCache::new(shard_size)));
        }

        Self {
            params,
            block_caches,
            manifest,
        }
    }

    /// Maps a block ID to its corresponding cache shard index.
    ///
    /// Uses simple modulo hashing to distribute blocks across shards.
    #[inline]
    fn block_to_shard_id(block_id: DataBlockId) -> usize {
        (block_id as usize) % Self::BLOCK_CACHE_NUM_SHARDS
    }

    /// Returns the filesystem path where a block is stored.
    ///
    /// Blocks are stored as `keyXXXXXXXX.data` where X is the zero-padded block ID.
    #[inline]
    fn get_file_path(&self, block_id: &DataBlockId) -> std::path::PathBuf {
        self.params.db_path.join(format!("key{block_id:08}.data"))
    }

    /// Creates a new data block builder.
    ///
    /// The builder can be used to incrementally construct a data block by adding entries.
    /// Call `finish()` on the builder to write the block to disk and cache it.
    ///
    /// # Arguments
    ///
    /// * `self_ptr` - Arc reference to this DataBlocks manager
    #[tracing::instrument(skip(self_ptr))]
    pub fn build_block(self_ptr: Arc<DataBlocks>) -> DataBlockBuilder {
        DataBlockBuilder::new(self_ptr)
    }

    /// Retrieves a data block by its ID.
    ///
    /// First checks the appropriate cache shard for the block. On a cache miss, loads the
    /// block from disk, caches it, and returns it. The lock is not held during disk I/O
    /// for better concurrency, though this may result in loading the same block multiple
    /// times concurrently in rare cases.
    ///
    /// # Arguments
    ///
    /// * `id` - The unique identifier of the block to retrieve
    ///
    /// # Panics
    ///
    /// Panics if the block file cannot be read from disk.
    #[tracing::instrument(skip(self))]
    pub async fn get_block(&self, id: &DataBlockId) -> Arc<DataBlock> {
        let shard_id = Self::block_to_shard_id(*id);
        let cache = &self.block_caches[shard_id];

        if let Some(block) = cache.lock().get(id) {
            return block.clone();
        }

        // Do not hold the lock while loading from disk for better concurrency
        // Worst case this means we load the same block multiple times...
        let file_path = self.get_file_path(id);
        log::trace!("Loading data block from disk at {file_path:?}");
        let data = disk::read(&file_path, 0).await.unwrap_or_else(|err| {
            panic!("Failed to load data block from disk at {file_path:?}: {err}")
        });
        let block = Arc::new(DataBlock::new_from_data(
            data,
            self.params.block_restart_interval,
        ));

        cache.lock().put(*id, block.clone());
        log::trace!("Stored new block in cache");
        block
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    use tokio::test as async_test;

    #[cfg(feature = "wisckey")]
    #[async_test]
    async fn store_and_load() {
        let dir = tempdir().unwrap();
        let params = Arc::new(Params {
            db_path: dir.path().to_path_buf(),
            ..Default::default()
        });

        let manifest = Arc::new(Manifest::new(params.clone()).await);

        let data_blocks = Arc::new(DataBlocks::new(params.clone(), manifest));
        let mut builder = DataBlocks::build_block(data_blocks.clone());

        let key1 = PrefixedKey {
            prefix_len: 0,
            suffix: vec![5],
        };
        let seq1 = 14234524;
        let val1 = (4, 2);
        builder.add_entry(key1, &[5], seq1, WriteOp::PUT_OP, val1);

        let key2 = PrefixedKey {
            prefix_len: 1,
            suffix: vec![2],
        };
        let seq2 = 424234;
        let val2 = (4, 5);
        builder.add_entry(key2, &[5, 2], seq2, WriteOp::PUT_OP, val2);

        let id = builder.finish().await.unwrap().unwrap();
        let data_block1 = data_blocks.get_block(&id).await;
        let data_block2 = Arc::new(DataBlock::new_from_data(
            data_block1.data.clone(),
            params.block_restart_interval,
        ));

        let prev_key = vec![];
        let (key, entry) = DataBlock::get_entry_at_offset(data_block2.clone(), 0, &prev_key);

        assert_eq!(key, vec![5]);
        assert_eq!(entry.get_value_id(), Some(val1));

        let (key, entry) = DataBlock::get_entry_at_offset(data_block2.clone(), entry.len(), &key);

        assert_eq!(key, vec![5, 2]);
        assert_eq!(entry.get_value_id(), Some(val2));
        assert_eq!(entry.len(), data_block2.byte_len());
    }

    #[cfg(not(feature = "wisckey"))]
    #[async_test]
    async fn store_and_load() {
        let dir = tempdir().unwrap();
        let params = Arc::new(Params {
            db_path: dir.path().to_path_buf(),
            ..Default::default()
        });

        let manifest = Arc::new(Manifest::new(params.clone()).await);

        let data_blocks = Arc::new(DataBlocks::new(params.clone(), manifest));
        let mut builder = DataBlocks::build_block(data_blocks.clone());

        let key1 = PrefixedKey {
            prefix_len: 0,
            suffix: vec![5],
        };
        let seq1 = 14234524;
        let val1 = vec![4, 2];
        builder.add_entry(key1, &[5u8], seq1, WriteOp::PUT_OP, &val1);

        let key2 = PrefixedKey {
            prefix_len: 1,
            suffix: vec![2],
        };
        let seq2 = 424234;
        let val2 = vec![24, 50];
        builder.add_entry(key2, &[5u8, 2u8], seq2, WriteOp::PUT_OP, &val2);

        let id = builder.finish().await.unwrap().unwrap();
        let data_block1 = data_blocks.get_block(&id).await;
        let data_block2 = Arc::new(DataBlock::new_from_data(
            data_block1.data.clone(),
            params.block_restart_interval,
        ));

        let prev_key = vec![];
        let (key, entry) = DataBlock::get_entry_at_offset(data_block2.clone(), 0, &prev_key);

        assert_eq!(key, vec![5]);
        assert_eq!(entry.get_value(), Some(&val1[..]));

        let (key, entry) = DataBlock::get_entry_at_offset(data_block2.clone(), entry.len(), &key);

        assert_eq!(key, vec![5, 2]);
        assert_eq!(entry.get_value(), Some(&val2[..]));
        assert_eq!(entry.len(), data_block2.byte_len());
    }
}
