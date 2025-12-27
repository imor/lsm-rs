use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32};

use crate::data_blocks::{DataBlockBuilder, DataBlockId, DataBlocks, PrefixedKey};
use crate::index_blocks::IndexBlock;
use crate::manifest::SeqNumber;
use crate::{Error, Key, Params, WriteOp};

#[cfg(feature = "wisckey")]
use crate::values::ValueId;

use super::{SortedTable, TableId};

/// Builder for constructing sorted tables (SSTables) during compaction.
///
/// `TableBuilder` incrementally builds a sorted table by adding key-value entries in sorted
/// order. It manages the creation of multiple data blocks, tracks block metadata, and
/// constructs the index block upon completion.
///
/// ## Building Process
///
/// 1. Create a new builder with `new()`
/// 2. Add entries in sorted key order using `add_value()` or `add_deletion()`
/// 3. Call `finish()` to finalize all blocks and create the `SortedTable`
///
/// ## Data Block Management
///
/// When a data block reaches the maximum key count (`max_key_block_size`), the builder
/// automatically finalizes it, writes it to disk, and starts a new block.
///
/// ## Prefix Compression
///
/// The builder implements prefix compression within blocks using restart intervals to
/// balance space efficiency and lookup performance.
pub struct TableBuilder<'a> {
    /// Unique identifier for this table.
    identifier: TableId,

    /// Database configuration parameters.
    params: &'a Params,

    /// Manager for data block caching and disk I/O.
    data_blocks: Arc<DataBlocks>,

    /// The smallest key that will be stored in this table.
    min_key: Key,

    /// The largest key that will be stored in this table.
    max_key: Key,

    /// The current data block being built.
    data_block: DataBlockBuilder,

    /// Index mapping the first key of each block to its block ID.
    block_index: Vec<(Key, DataBlockId)>,

    /// The last key added to the current block (used for prefix compression).
    last_key: Key,

    /// Number of entries in the current block.
    block_entry_count: usize,

    /// Total size of all finalized blocks in bytes.
    size: u64,

    /// Counter for restart intervals (resets to 0 at restart points).
    restart_count: u32,

    /// The first key in the current block (used for block index).
    index_key: Option<Key>,
}

impl<'a> TableBuilder<'a> {
    /// Creates a new `TableBuilder` for constructing a sorted table.
    ///
    /// Initializes an empty builder with the first data block ready to receive entries.
    ///
    /// # Arguments
    ///
    /// * `identifier` - Unique ID for this table
    /// * `params` - Database configuration parameters
    /// * `data_blocks` - Manager for data block operations
    /// * `min_key` - The smallest key that will be stored in this table
    /// * `max_key` - The largest key that will be stored in this table
    #[tracing::instrument(skip(params, data_blocks, min_key, max_key))]
    pub fn new(
        identifier: TableId,
        params: &'a Params,
        data_blocks: Arc<DataBlocks>,
        min_key: Key,
        max_key: Key,
    ) -> TableBuilder<'a> {
        let block_index = vec![];
        let last_key = vec![];
        let block_entry_count = 0;
        let size = 0;
        let restart_count = 0;
        let index_key = None;
        let data_block = DataBlocks::build_block(data_blocks.clone());

        Self {
            identifier,
            params,
            data_blocks,
            block_index,
            data_block,
            last_key,
            block_entry_count,
            size,
            restart_count,
            index_key,
            min_key,
            max_key,
        }
    }

    /// Adds a Put entry to the table (WiscKey mode).
    ///
    /// Entries must be added in sorted key order. The value is stored as a reference
    /// (batch ID and offset) rather than inline.
    ///
    /// # Arguments
    ///
    /// * `key` - The key bytes (must be >= previous key)
    /// * `seq_number` - Sequence number for versioning
    /// * `value_ref` - Reference to the value in the value log
    ///
    /// # Errors
    ///
    /// Returns an error if writing a completed block to disk fails.
    #[cfg(feature = "wisckey")]
    #[tracing::instrument(skip(self, key, seq_number, value_ref))]
    pub async fn add_value(
        &mut self,
        key: &[u8],
        seq_number: SeqNumber,
        value_ref: ValueId,
    ) -> Result<(), Error> {
        self.add_entry(key, seq_number, WriteOp::PUT_OP, value_ref)
            .await
    }

    /// Adds a Delete entry (tombstone) to the table (WiscKey mode).
    ///
    /// Tombstones mark keys as deleted and must be added in sorted key order.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to delete (must be >= previous key)
    /// * `seq_number` - Sequence number for versioning
    ///
    /// # Errors
    ///
    /// Returns an error if writing a completed block to disk fails.
    #[cfg(feature = "wisckey")]
    #[tracing::instrument(skip(self, key, seq_number))]
    pub async fn add_deletion(&mut self, key: &[u8], seq_number: SeqNumber) -> Result<(), Error> {
        self.add_entry(key, seq_number, WriteOp::DELETE_OP, ValueId::default())
            .await
    }

    /// Adds a Put entry to the table (non-WiscKey mode).
    ///
    /// Entries must be added in sorted key order. The value is stored inline with the key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key bytes (must be >= previous key)
    /// * `seq_number` - Sequence number for versioning
    /// * `value` - The value bytes to store inline
    ///
    /// # Errors
    ///
    /// Returns an error if writing a completed block to disk fails.
    #[cfg(not(feature = "wisckey"))]
    #[tracing::instrument(skip(self, key, seq_number, value))]
    pub async fn add_value(
        &mut self,
        key: &[u8],
        seq_number: SeqNumber,
        value: &[u8],
    ) -> Result<(), Error> {
        self.add_entry(key, seq_number, WriteOp::PUT_OP, value)
            .await
    }

    /// Adds a Delete entry (tombstone) to the table (non-WiscKey mode).
    ///
    /// Tombstones mark keys as deleted and must be added in sorted key order.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to delete (must be >= previous key)
    /// * `seq_number` - Sequence number for versioning
    ///
    /// # Errors
    ///
    /// Returns an error if writing a completed block to disk fails.
    #[cfg(not(feature = "wisckey"))]
    #[tracing::instrument(skip(self, key, seq_number))]
    pub async fn add_deletion(&mut self, key: &[u8], seq_number: SeqNumber) -> Result<(), Error> {
        self.add_entry(key, seq_number, WriteOp::DELETE_OP, &[])
            .await
    }

    /// Internal method for adding entries with prefix compression.
    ///
    /// Calculates the prefix length relative to the previous key, manages restart intervals,
    /// and automatically finalizes blocks when they reach the size limit.
    ///
    /// # Arguments
    ///
    /// * `key` - The key bytes
    /// * `seq_number` - Sequence number for versioning
    /// * `op_type` - Operation type (PUT_OP or DELETE_OP)
    /// * `value` - Value reference (WiscKey) or value bytes (non-WiscKey)
    ///
    /// # Errors
    ///
    /// Returns an error if writing a completed block to disk fails.
    async fn add_entry(
        &mut self,
        key: &[u8],
        seq_number: SeqNumber,
        op_type: u8,
        #[cfg(feature = "wisckey")] value: ValueId,
        #[cfg(not(feature = "wisckey"))] value: &[u8],
    ) -> Result<(), Error> {
        if self.index_key.is_none() {
            self.index_key = Some(key.to_vec());
        }
        let mut prefix_len = 0;

        // After a certain interval we reset the prefixed keys
        // So that it is possible to binary search blocks
        if self.restart_count == self.params.block_restart_interval {
            self.restart_count = 0;
        } else {
            // Calculate key prefix length
            while prefix_len < key.len()
                && prefix_len < self.last_key.len()
                && key[prefix_len] == self.last_key[prefix_len]
            {
                prefix_len += 1;
            }
        }

        let suffix = key[prefix_len..].to_vec();

        self.block_entry_count += 1;
        self.restart_count += 1;

        let pkey = PrefixedKey::new(prefix_len, suffix);

        self.last_key = key.to_vec();

        self.data_block
            .add_entry(pkey, key, seq_number, op_type, value);

        if self.block_entry_count >= self.params.max_key_block_size {
            self.size += self.data_block.current_size() as u64;

            let mut next_block = DataBlocks::build_block(self.data_blocks.clone());
            std::mem::swap(&mut next_block, &mut self.data_block);

            let id = next_block.finish().await?.unwrap();
            self.block_index.push((self.index_key.take().unwrap(), id));
            self.block_entry_count = 0;
            self.restart_count = 0;
            self.last_key.clear();
        }

        Ok(())
    }

    /// Finalizes the table construction and creates a `SortedTable`.
    ///
    /// This method:
    /// 1. Finishes the current data block (if it contains entries)
    /// 2. Creates an index block with metadata and the block index
    /// 3. Calculates seek-based compaction threshold if enabled
    /// 4. Returns a complete `SortedTable` ready for use
    ///
    /// # Returns
    ///
    /// A `SortedTable` containing all the added entries organized into data blocks
    /// with an index for efficient lookups.
    ///
    /// # Errors
    ///
    /// Returns an error if writing the final block or index to disk fails.
    #[tracing::instrument(skip(self))]
    pub async fn finish(mut self) -> Result<SortedTable, Error> {
        let block_size = self.data_block.current_size();

        // Block will only be created if it contained entries
        if let Some(id) = self.data_block.finish().await? {
            self.size += block_size as u64;
            self.block_index.push((self.index_key.take().unwrap(), id));
        }

        log::debug!("Created new table with {} blocks", self.block_index.len());

        let index = IndexBlock::new(
            self.params,
            self.identifier,
            self.block_index,
            self.size,
            self.min_key,
            self.max_key,
        )
        .await?;

        let allowed_seeks = if let Some(count) = self.params.seek_based_compaction {
            ((index.get_size() / 1024).max(1) as i32) * (count as i32)
        } else {
            0
        };

        Ok(SortedTable {
            index,
            num_seeks_compaction_threshold: AtomicI32::new(allowed_seeks),
            identifier: self.identifier,
            being_compacted: AtomicBool::new(false),
            data_blocks: self.data_blocks,
        })
    }
}
