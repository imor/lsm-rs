//! LSM-tree level structure and table management.
//!
//! This module implements the `Level` abstraction, which organizes sorted tables (SSTables)
//! into hierarchical levels within the LSM-tree. Each level maintains:
//! - A collection of sorted tables with non-overlapping key ranges (except L0)
//! - Metadata for compaction scheduling and table selection
//! - Placeholders to coordinate concurrent compaction operations
//! - Capacity management based on level-specific size limits
//!
//! # Level Organization
//!
//! - **Level 0 (L0)**: Contains recently flushed memtables. Tables may have overlapping
//!   key ranges and are ordered by creation time (newest to oldest).
//! - **Level 1+**: Contain compacted tables with non-overlapping key ranges, sorted by
//!   minimum key for efficient lookups.
//!
//! # Compaction Strategy
//!
//! The level supports two compaction triggers:
//! - **Size-based**: Triggered when total level size exceeds `max_size()`
//! - **Seek-based**: Triggered when a table exceeds its maximum seek count
//!
//! Compaction involves selecting tables from this level and merging them with
//! overlapping tables in the next level down.

use crate::data_blocks::{DataBlocks, DataEntry};
use crate::manifest::{INVALID_TABLE_ID, LevelId, Manifest};
use crate::sorted_table::{SortedTable, TableBuilder, TableId};
use crate::{Error, Key, Params};

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::RwLock;

use parking_lot::Mutex as PMutex;

/// Minimum number of L0 tables before size-based compaction is triggered.
/// TODO: add slowdown writes trigger
const L0_COMPACTION_TRIGGER: usize = 4;

/// Vector of sorted tables wrapped in Arc for shared ownership.
pub type TableVec = Vec<Arc<SortedTable>>;

/// A placeholder representing a table being created during compaction.
///
/// Placeholders prevent race conditions by reserving key ranges during concurrent
/// compaction operations. They are inserted before compaction begins and removed
/// once the new table is finalized.
pub struct TablePlaceholder {
    /// Minimum key in the placeholder's range.
    min: Key,
    /// Maximum key in the placeholder's range.
    max: Key,
    /// Unique identifier for this placeholder/table.
    id: TableId,
}

impl TablePlaceholder {
    /// Checks if this placeholder overlaps with the given key range.
    ///
    /// # Arguments
    ///
    /// * `min` - Minimum key of the range to check
    /// * `max` - Maximum key of the range to check
    ///
    /// # Returns
    ///
    /// `true` if the ranges overlap, `false` otherwise
    fn overlaps(&self, min: &[u8], max: &[u8]) -> bool {
        self.max.as_slice() >= min && self.min.as_slice() <= max
    }
}

/// Represents a single level in the LSM-tree hierarchy.
///
/// Each level maintains a collection of sorted tables and provides methods for:
/// - Reading values by key
/// - Adding new tables (from memtable flushes or compaction)
/// - Selecting tables for compaction
/// - Detecting key range overlaps
///
/// # Concurrency
///
/// Level operations use fine-grained locking:
/// - Table list is protected by an async RwLock for concurrent reads
/// - Compaction offset uses a parking_lot Mutex for low-latency updates
/// - Atomic operations track seek-based compaction candidates
pub struct Level {
    /// Level index (0 for L0, 1 for L1, etc.).
    index: LevelId,
    /// Offset for round-robin table selection during size-based compaction.
    next_compaction_offset: PMutex<usize>,
    /// Whether seek-based compaction is enabled for this level.
    do_seek_based_compaction: bool,
    /// Table ID candidate for seek-based compaction (INVALID_TABLE_ID if none).
    seek_based_compaction: AtomicU64,
    /// Shared data block cache and I/O handler.
    data_blocks: Arc<DataBlocks>,
    /// Collection of sorted tables at this level.
    tables: RwLock<TableVec>,
    /// Database configuration parameters.
    params: Arc<Params>,
    /// Manifest for tracking table metadata and generating IDs.
    manifest: Arc<Manifest>,
    /// Tables in the process of being created during compaction.
    table_placeholders: RwLock<Vec<TablePlaceholder>>,
}

impl Level {
    /// Creates a new empty level.
    ///
    /// # Arguments
    ///
    /// * `index` - Level index (0 for L0, 1 for L1, etc.)
    /// * `data_blocks` - Shared data block cache and I/O handler
    /// * `params` - Database configuration parameters
    /// * `manifest` - Manifest for tracking table metadata
    ///
    /// # Returns
    ///
    /// A new `Level` instance with no tables
    pub fn new(
        index: LevelId,
        data_blocks: Arc<DataBlocks>,
        params: Arc<Params>,
        manifest: Arc<Manifest>,
    ) -> Self {
        Self {
            index,
            do_seek_based_compaction: params.seek_based_compaction.is_some(),
            params,
            manifest,
            data_blocks,
            seek_based_compaction: AtomicU64::new(INVALID_TABLE_ID),
            next_compaction_offset: PMutex::new(0),
            tables: RwLock::new(vec![]),
            table_placeholders: RwLock::new(vec![]),
        }
    }

    /// Sets the next compaction offset for round-robin table selection.
    ///
    /// # Arguments
    ///
    /// * `offset` - Index of the next table to consider for compaction
    ///
    /// # Note
    ///
    /// This method is primarily used for testing to control compaction behavior.
    #[allow(dead_code)]
    pub fn set_next_compaction_offset(&self, offset: usize) {
        *self.next_compaction_offset.lock() = offset;
    }

    /// Removes a table placeholder after compaction completes.
    ///
    /// # Arguments
    ///
    /// * `id` - Table ID of the placeholder to remove
    ///
    /// # Panics
    ///
    /// Panics if no placeholder with the given ID exists.
    pub async fn remove_table_placeholder(&self, id: TableId) {
        let mut placeholders = self.table_placeholders.write().await;
        for (pos, placeholder) in placeholders.iter().enumerate() {
            if placeholder.id == id {
                placeholders.remove(pos);
                return;
            }
        }

        panic!("no such placeholder");
    }

    /// Loads an existing sorted table from disk and adds it to this level.
    ///
    /// # Arguments
    ///
    /// * `id` - Table ID to load
    ///
    /// # Returns
    ///
    /// `Ok(())` on success
    ///
    /// # Errors
    ///
    /// Returns an error if the table cannot be loaded from disk.
    pub async fn load_table(&self, id: TableId) -> Result<(), Error> {
        let table = SortedTable::load(id, self.data_blocks.clone(), &self.params).await?;

        let mut tables = self.tables.write().await;
        tables.push(Arc::new(table));

        log::trace!("Loaded table {id} on level {}", self.index);
        Ok(())
    }

    /// Creates a table builder for constructing a new sorted table at this level.
    ///
    /// # Arguments
    ///
    /// * `identifier` - Unique table ID for the new table
    /// * `min_key` - Minimum key that will be in the table
    /// * `max_key` - Maximum key that will be in the table
    ///
    /// # Returns
    ///
    /// A `TableBuilder` configured for this level
    pub fn build_table(&self, identifier: TableId, min_key: Key, max_key: Key) -> TableBuilder<'_> {
        TableBuilder::new(
            identifier,
            &self.params,
            self.data_blocks.clone(),
            min_key,
            max_key,
        )
    }

    /// Returns the level index.
    ///
    /// # Returns
    ///
    /// Level index (0 for L0, 1 for L1, etc.)
    pub fn get_index(&self) -> u32 {
        self.index
    }

    /// Adds a newly flushed table to L0.
    ///
    /// # Arguments
    ///
    /// * `table` - The sorted table to add
    ///
    /// # Panics
    ///
    /// Panics if called on a level other than L0.
    pub async fn add_l0_table(&self, table: SortedTable) {
        assert_eq!(self.index, 0);
        let mut tables = self.tables.write().await;
        tables.push(Arc::new(table));
    }

    /// Retrieves a data entry for the given key from this level.
    ///
    /// Searches tables from newest to oldest (important for L0 which may have overlapping keys).
    /// If seek-based compaction is enabled, increments seek counters and may trigger compaction.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to search for
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - `bool`: `true` if seek-based compaction was triggered for this level
    /// - `Option<DataEntry>`: The data entry if found, `None` otherwise
    #[tracing::instrument(skip(self,key), fields(index=self.index))]
    pub async fn get(&self, key: &[u8]) -> (bool, Option<DataEntry>) {
        let tables = self.tables.read().await;
        let mut compaction_triggered = false;

        // Iterate from back to front (newest to oldest)
        // as L0 may have overlapping entries
        for table in tables.iter().rev() {
            let result = table.get(key).await;

            if self.do_seek_based_compaction
                && table.has_maximum_seeks()
                && self
                    .seek_based_compaction
                    .compare_exchange(
                        INVALID_TABLE_ID,
                        table.get_id(),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    )
                    .is_ok()
            {
                log::trace!(
                    "Seek-based compaction triggered for table #{}",
                    table.get_id()
                );
                compaction_triggered = true;
            }

            if result.is_some() {
                return (compaction_triggered, result);
            }
        }

        (compaction_triggered, None)
    }

    /// Calculates the maximum size (in bytes) for this level before compaction is triggered.
    ///
    /// The size limit grows exponentially with level depth:
    /// - L0 and L1: 1 MB
    /// - L2: 10 MB
    /// - L3: 100 MB
    /// - And so on (10x per level)
    ///
    /// # Returns
    ///
    /// Maximum size in bytes for this level
    ///
    /// # Note
    ///
    /// L0 compaction is triggered by table count, not size, so this value is not used for L0.
    pub fn max_size(&self) -> usize {
        // Note: the result for level zero is not really used since we set
        // the level-0 compaction threshold based on number of files.

        // Result for both level-0 and level-1
        // This doesn't include the size of the values (for now)
        let mut result: usize = 1048576;
        let mut level = self.index;
        while level > 1 {
            result *= 10;
            level -= 1;
        }

        result
    }

    /// Checks if compaction should start and selects tables to compact.
    ///
    /// Compaction is triggered by:
    /// - **L0**: Number of tables exceeds `L0_COMPACTION_TRIGGER`
    /// - **L1+**: Total size exceeds `max_size()`
    /// - **Any level**: A table exceeds its seek count (seek-based compaction)
    ///
    /// For L0, may select multiple overlapping tables. For L1+, selects a single table
    /// using round-robin scheduling.
    ///
    /// # Returns
    ///
    /// - `Ok(Some(tables))`: Compaction should proceed with the given tables
    /// - `Ok(None)`: No compaction needed
    /// - `Err(())`: Compaction was needed but failed due to lock contention
    #[tracing::instrument(skip(self))]
    pub async fn maybe_start_compaction(&self) -> Result<Option<Vec<Arc<SortedTable>>>, ()> {
        log::trace!("Checking if we should compact level");
        let all_tables = self.tables.read().await;

        let (table, offset) = 'choice: {
            let mut next_offset = self.next_compaction_offset.lock();

            let size_based_compaction = if self.index == 0 {
                all_tables.len() > L0_COMPACTION_TRIGGER
            } else {
                let mut total_size = 0;

                for t in all_tables.iter() {
                    total_size += t.get_size();
                }

                total_size > self.max_size()
            };

            // Prefer size-based compaction over seek-based compaction
            if size_based_compaction {
                if all_tables.is_empty() {
                    panic!("Cannot start compaction; level {} is empty", self.index);
                }

                if *next_offset >= all_tables.len() {
                    *next_offset = 0;
                }

                let offset = *next_offset;
                let table = all_tables[offset].clone();

                *next_offset += 1;

                (table, offset)
            } else {
                let table_id = self.seek_based_compaction.load(Ordering::SeqCst);

                if table_id != INVALID_TABLE_ID {
                    for (pos, table) in all_tables.iter().enumerate() {
                        if table.get_id() == table_id {
                            self.seek_based_compaction
                                .store(INVALID_TABLE_ID, Ordering::SeqCst);
                            break 'choice (table.clone(), pos);
                        }
                    }
                }

                return Ok(None);
            }
        };

        // Try to set the compaction flag
        // otherwise, we abort (due to concurrency)
        if !table.start_compaction() {
            return Err(());
        }

        let mut tables = vec![table];
        let mut offsets = vec![offset];

        // Level 0 might have overlapping tables
        if self.index == 0 {
            let mut min = tables[0].get_min().to_vec();
            let mut max = tables[0].get_max().to_vec();

            //TODO how greedy should this be?
            let mut change = true;
            while change {
                change = false;
                for (pos, table) in all_tables.iter().enumerate() {
                    let mut found = false;
                    for offset in offsets.iter() {
                        if pos == *offset {
                            found = true;
                            break;
                        }
                    }

                    if found {
                        continue;
                    }

                    if table.overlaps(&min, &max) {
                        if table.start_compaction() {
                            min = std::cmp::min(&min[..], table.get_min()).to_vec();
                            max = std::cmp::max(&max[..], table.get_max()).to_vec();

                            offsets.push(pos);
                            tables.push(table.clone());
                            change = true;
                            break;
                        } else {
                            // Lock contention!
                            for table in tables {
                                table.stop_compaction();
                            }
                            return Err(());
                        }
                    }
                }
            }
        }

        Ok(Some(tables))
    }

    /// Finds overlapping tables in this level and prepares for compaction.
    ///
    /// This method performs three critical operations:
    /// 1. Identifies all tables whose key ranges overlap with `[min, max]`
    /// 2. Sets compaction flags on overlapping tables (locks them for compaction)
    /// 3. Creates a placeholder to reserve the key range and prevent concurrent compactions
    ///
    /// # Arguments
    ///
    /// * `min` - Minimum key of the range to check for overlaps
    /// * `max` - Maximum key of the range to check for overlaps
    /// * `fast_path` - Optional table ID to use if no overlaps exist (for fast compaction)
    ///
    /// # Returns
    ///
    /// - `Some((table_id, overlapping_tables))`: Compaction can proceed
    ///   - `table_id`: ID for the placeholder (and new table to be created)
    ///   - `overlapping_tables`: Tables that must be compacted together
    /// - `None`: Compaction aborted due to lock contention or existing placeholder
    ///
    /// # Note
    ///
    /// If `fast_path` is provided and no overlaps exist, the provided ID is used.
    /// Otherwise, a new table ID is generated from the manifest.
    #[tracing::instrument(skip(self))]
    pub async fn get_overlaps(
        &self,
        min: &[u8],
        max: &[u8],
        fast_path: Option<TableId>,
    ) -> Option<(TableId, Vec<Arc<SortedTable>>)> {
        let mut tables_to_compact: Vec<Arc<SortedTable>> = Vec::new();
        let tables = self.tables.read().await;

        let mut min = min;
        let mut max = max;

        for table in tables.iter() {
            if table.overlaps(min, max) {
                if !table.start_compaction() {
                    // Abort
                    for table in tables_to_compact.into_iter() {
                        table.stop_compaction();
                    }
                    return None;
                }

                tables_to_compact.push(table.clone());
                min = table.get_min().min(min);
                max = table.get_max().max(max);
            }
        }

        // set placeholder to avoid race conditions
        // and abort if one exists
        let mut placeholders = self.table_placeholders.write().await;
        for placeholder in placeholders.iter() {
            if placeholder.overlaps(min, max) {
                for table in tables_to_compact {
                    table.stop_compaction();
                }
                return None;
            }
        }

        let table_id = if let Some(table_id) = fast_path
            && tables_to_compact.is_empty()
        {
            table_id
        } else {
            self.manifest.generate_next_table_id().await
        };

        placeholders.push(TablePlaceholder {
            id: table_id,
            min: min.to_vec(),
            max: max.to_vec(),
        });

        Some((table_id, tables_to_compact))
    }

    /// Acquires an exclusive write lock on the table collection.
    ///
    /// # Returns
    ///
    /// A write guard providing mutable access to the table vector
    #[inline]
    pub async fn get_tables_rw(&self) -> tokio::sync::RwLockWriteGuard<'_, TableVec> {
        self.tables.write().await
    }

    /// Acquires a shared read lock on the table collection.
    ///
    /// # Returns
    ///
    /// A read guard providing immutable access to the table vector
    #[inline]
    pub async fn get_tables_ro(&self) -> tokio::sync::RwLockReadGuard<'_, TableVec> {
        self.tables.read().await
    }
}
