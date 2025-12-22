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

pub type TableId = u64;

/// Entries in each level are grouped into sorted tables
/// These tables contain an ordered set of key/value-pairs
///
/// Except for level 0, sorted tables do not overlap others on the same level
pub struct SortedTable {
    /// The unique identifier of this table
    identifier: TableId,
    /// The index of the table; it holds all relevant metadata
    index: IndexBlock,
    /// The data blocks of this table
    data_blocks: Arc<DataBlocks>,
    /// Is this table currently being compacted
    being_compacted: AtomicBool,
    /// The number of seek operations on this table before compaction is triggered
    /// This improves read performance for heavily queried keys
    num_seeks_compaction_threshold: AtomicI32,
}

impl SortedTable {
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

    /// Checks if seek-based compaction should be triggered for this table
    pub fn has_maximum_seeks(&self) -> bool {
        self.num_seeks_compaction_threshold
            .load(AtomicOrdering::SeqCst)
            <= 0
    }

    /// Tries to atomically mark this table as being compacted and
    /// returns false if another task is already compacting this table
    pub fn start_compaction(&self) -> bool {
        let order = AtomicOrdering::SeqCst;
        let result = self
            .being_compacted
            .compare_exchange(false, true, order, order);

        result.is_ok()
    }

    /// Compaction has failed, e.g., due to lock contention
    /// Remove the compaction flag
    pub fn stop_compaction(&self) {
        let prev = self.being_compacted.swap(false, AtomicOrdering::SeqCst);
        assert!(prev, "Compaction flag was not set!");
    }

    pub fn get_id(&self) -> TableId {
        self.identifier
    }

    /// Get the size of this table (in bytes)
    pub fn get_size(&self) -> usize {
        self.index.get_size()
    }

    /// Get the minimum key of this table
    pub fn get_min(&self) -> &[u8] {
        self.index.get_min()
    }

    /// Get the maximum key of this table
    pub fn get_max(&self) -> &[u8] {
        self.index.get_max()
    }

    /// Gets an entry for particular key in this table
    /// Returns None if no entry for the key exists
    #[tracing::instrument(skip(self, key))]
    pub async fn get(&self, key: &[u8]) -> Option<DataEntry> {
        self.num_seeks_compaction_threshold
            .fetch_sub(1, AtomicOrdering::Relaxed);

        let block_id = self.index.binary_search(key)?;
        let block = self.data_blocks.get_block(&block_id).await;

        DataBlock::get_by_key(&block, key)
    }

    /// Check if this table overlaps with the specified range
    ///
    /// min and max are both inclusive
    #[inline(always)]
    pub fn overlaps(&self, min: &[u8], max: &[u8]) -> bool {
        self.get_max() >= min && self.get_min() <= max
    }
}
