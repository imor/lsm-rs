//! Core database logic and state management.
//!
//! This module implements `DbLogic`, the central coordinator for all LSM-tree operations.
//! It manages:
//! - **Memtables**: Active and immutable memtables with write coordination
//! - **Levels**: Hierarchical organization of sorted tables
//! - **Compaction**: Memtable flushes and level-to-level compaction
//! - **Reads**: Lock-free multi-version reads across memtables and levels
//! - **Writes**: Write-ahead logging and memtable updates with backpressure
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐
//! │  Memtable   │ ← Active writes
//! └─────────────┘
//!       ↓ (when full)
//! ┌─────────────┐
//! │ Imm Memtable│ ← Queue for flushing
//! └─────────────┘
//!       ↓ (flush to disk)
//! ┌─────────────┐
//! │   Level 0   │ ← May have overlapping keys
//! └─────────────┘
//!       ↓ (compaction)
//! ┌─────────────┐
//! │   Level 1   │ ← Non-overlapping keys
//! └─────────────┘
//!       ↓
//!      ...
//! ```
//!
//! # Concurrency Model
//!
//! - **Lock-free reads**: Readers acquire shared locks on memtables/levels without blocking
//! - **Write coordination**: Single-writer model for memtable with backpressure
//! - **Compaction**: Background tasks with fine-grained table-level locks
//!
//! # Write Path
//!
//! 1. Append to Write-Ahead Log (WAL)
//! 2. Insert into active memtable
//! 3. If memtable full → freeze to immutable memtable
//! 4. Background task flushes immutable memtable to L0
//!
//! # Read Path
//!
//! 1. Check active memtable
//! 2. Check immutable memtables (newest to oldest)
//! 3. Check levels L0 → L1 → L2... (newest to oldest)
//! 4. Return first match or None

use std::collections::VecDeque;
use std::sync::Arc;

use tokio::sync::{RwLock, RwLockWriteGuard};
use tokio_condvar::Condvar;

use cfg_if::cfg_if;

use crate::data_blocks::{DataBlocks, DataEntryType};
use crate::disk::{create_dir, remove_dir_all};
use crate::level::Level;
use crate::level_logger::LevelLogger;
use crate::manifest::{LevelId, Manifest};
use crate::memtable::{
    ImmMemtableRef, Memtable, MemtableEntry, MemtableEntryRef, MemtableIterator, MemtableRef,
};
use crate::sorted_table::{InternalIterator, TableId, TableIterator};
use crate::wal::{LogEntry, WriteAheadLog};
use crate::{Error, Key, Params, StartMode, WriteBatch, WriteOptions};

#[cfg(feature = "wisckey")]
use crate::values::{ValueIndex, ValueLog, ValueRef};

use crate::data_blocks::DataEntry;

/// Result of a compaction attempt.
#[derive(Debug, PartialEq, Eq)]
enum CompactResult {
    /// No compaction was needed.
    NothingToDo,
    /// Compaction completed successfully.
    DidWork,
    /// Compaction was needed but couldn't proceed due to lock contention.
    Locked,
}

/// A reference to a key-value entry without copying its data.
///
/// Provides zero-copy access to values stored in either memtables or sorted tables.
/// When WiscKey mode is enabled, sorted table entries store references to the value log.
pub enum EntryRef {
    /// Entry from a sorted table on disk.
    SortedTable {
        /// The data entry containing key metadata.
        entry: DataEntry,
        /// Reference to value in the value log (WiscKey mode only).
        #[cfg(feature = "wisckey")]
        value_ref: ValueRef,
    },
    /// Entry from an in-memory memtable.
    Memtable {
        /// Reference to the memtable entry.
        entry: MemtableEntryRef,
    },
}

impl EntryRef {
    /// Returns the value associated with this entry.
    ///
    /// # Returns
    ///
    /// Byte slice containing the value data
    pub fn get_value(&self) -> &[u8] {
        match self {
            #[cfg(feature = "wisckey")]
            Self::SortedTable { value_ref, .. } => value_ref.get_value(),
            #[cfg(not(feature = "wisckey"))]
            Self::SortedTable { entry } => entry.get_value().unwrap(),
            Self::Memtable { entry } => entry.get_value().unwrap(),
        }
    }
}

/// Core database state manager and operation coordinator.
///
/// `DbLogic` is the central component that manages all LSM-tree operations including
/// reads, writes, and compaction. It coordinates between the active memtable, immutable
/// memtables awaiting flush, and the hierarchical levels of sorted tables.
///
/// # Usage
///
/// This type is typically not used directly. Instead, use the `Database` wrapper which
/// provides a higher-level API. `DbLogic` is public to enable the synchronous API
/// implementation in the `lsm-sync` crate.
///
/// # Concurrency
///
/// - **Reads**: Lock-free with shared access to memtables and levels
/// - **Writes**: Serialized through memtable write lock with backpressure
/// - **Compaction**: Background tasks with table-level coordination
///
/// # Guarantees
///
/// - **Durability**: All writes are logged to WAL before returning
/// - **Consistency**: Monotonic sequence numbers ensure correct ordering
/// - **Isolation**: Readers see a consistent snapshot via MVCC
pub struct DbLogic {
    /// Manifest tracking table metadata and generating sequence numbers.
    manifest: Arc<Manifest>,
    /// Database configuration parameters.
    params: Arc<Params>,
    /// Currently active memtable receiving new writes.
    memtable: RwLock<MemtableRef>,
    /// Queue of immutable memtables waiting to be flushed to L0.
    /// Each entry contains (WAL offset, memtable) for recovery coordination.
    imm_memtables: RwLock<VecDeque<(usize, ImmMemtableRef)>>,
    /// Condition variable for backpressure when immutable queue is full.
    imm_cond: Condvar,
    /// Hierarchical levels of sorted tables (L0, L1, L2, ...).
    levels: Vec<Level>,
    /// Write-ahead log for durability.
    wal: Arc<WriteAheadLog>,
    /// Optional logger for tracking level statistics.
    level_logger: Option<LevelLogger>,

    /// Value log for storing large values separately (WiscKey mode).
    #[cfg(feature = "wisckey")]
    value_log: Arc<ValueLog>,
}

impl DbLogic {
    /// Creates or opens a database instance.
    ///
    /// # Arguments
    ///
    /// * `start_mode` - How to handle existing database:
    ///   - `CreateOrOpen`: Open existing or create new
    ///   - `Open`: Open existing (error if doesn't exist)
    ///   - `CreateOrOverride`: Delete existing and create new
    /// * `params` - Database configuration parameters
    ///
    /// # Returns
    ///
    /// `Ok(DbLogic)` on success
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParams`: Invalid configuration or database doesn't exist
    /// - I/O errors during directory creation, manifest loading, or WAL recovery
    pub async fn new(start_mode: StartMode, params: Params) -> Result<Self, Error> {
        params.validate()?;

        let create = match start_mode {
            StartMode::CreateOrOpen => !params.db_path.exists(),
            StartMode::Open => {
                if !params.db_path.exists() {
                    return Err(Error::InvalidParams("DB does not exist".to_string()));
                }

                false
            }
            StartMode::CreateOrOverride => {
                if params.db_path.exists() {
                    log::info!(
                        "Removing old data at \"{}\"",
                        params.db_path.to_str().unwrap()
                    );

                    remove_dir_all(&params.db_path).map_err(|e| {
                        Error::from_io_error(
                            format!("Failed to remove existing database: folder: {e}",),
                            e,
                        )
                    })?;
                }

                true
            }
        };

        let params = Arc::new(params);
        let manifest;
        let memtable;
        let wal;

        #[cfg(feature = "wisckey")]
        let value_log;

        if create {
            create_dir(&params.db_path).map_err(|e| {
                Error::from_io_error(format!("Failed to create DB folder: {e}",), e)
            })?;
            log::info!(
                "Created database folder at \"{}\"",
                params.db_path.to_str().unwrap()
            );

            manifest = Arc::new(Manifest::new(params.clone()).await);
            memtable = RwLock::new(MemtableRef::wrap(Memtable::new(1)));
            wal = Arc::new(WriteAheadLog::new(params.clone()).await?);

            #[cfg(feature = "wisckey")]
            {
                value_log =
                    Arc::new(ValueLog::new(wal.clone(), params.clone(), manifest.clone()).await?);
            }
        } else {
            log::info!(
                "Opening database folder at \"{}\"",
                params.db_path.to_str().unwrap()
            );

            manifest = Arc::new(Manifest::open(params.clone()).await?);

            let mut mtable = Memtable::new(manifest.get_seq_number_offset().await);

            cfg_if! {
                if #[cfg(feature="wisckey")] {
                    let mut value_index = ValueIndex::open(params.clone(), manifest.clone()).await?;
                    let (w, recovery_result) =
                        WriteAheadLog::open(params.clone(), manifest.get_log_offset().await, &mut mtable, &mut value_index)
                            .await?;
                    wal = Arc::new(w);
                    value_log = Arc::new(ValueLog::open(wal.clone(), params.clone(), manifest.clone(), value_index, recovery_result.value_batches_to_delete).await?);
                } else {
                    let (w, _) = WriteAheadLog::open(params.clone(), manifest.get_log_offset().await, &mut mtable).await?;

                    wal = Arc::new(w);
                }
            }

            memtable = RwLock::new(MemtableRef::wrap(mtable));
        }

        let data_blocks = Arc::new(DataBlocks::new(params.clone(), manifest.clone()));

        let mut levels = Vec::new();
        for index in 0..params.num_levels {
            let index = index as LevelId;
            let level = Level::new(index, data_blocks.clone(), params.clone(), manifest.clone());
            levels.push(level);
        }

        if !create {
            for (level_id, tables) in manifest.get_table_ids().await.iter().enumerate() {
                for table_id in tables {
                    levels[level_id].load_table(*table_id).await?;
                }
            }
        }

        let level_logger = params.create_level_logger();

        Ok(Self {
            manifest,
            params,
            memtable,
            imm_memtables: Default::default(),
            imm_cond: Default::default(),
            levels,
            wal,
            level_logger,
            #[cfg(feature = "wisckey")]
            value_log,
        })
    }

    /// Returns a reference to the value log (WiscKey mode only).
    ///
    /// # Returns
    ///
    /// Arc reference to the value log
    #[cfg(feature = "wisckey")]
    pub fn get_value_log(&self) -> Arc<ValueLog> {
        self.value_log.clone()
    }

    /// Prepares iterators for range iteration across memtables and levels.
    ///
    /// Creates iterators for all data sources (active memtable, immutable memtables,
    /// and sorted tables) that overlap with the specified key range.
    ///
    /// # Arguments
    ///
    /// * `min_key` - Optional minimum key (inclusive). `None` means start from beginning.
    /// * `max_key` - Optional maximum key (inclusive). `None` means iterate to end.
    /// * `reverse` - If `true`, set up for reverse iteration; otherwise forward.
    ///
    /// # Returns
    ///
    /// Tuple containing:
    /// - Vector of memtable iterators
    /// - Vector of sorted table iterators
    /// - Owned minimum key (if specified)
    /// - Owned maximum key (if specified)
    async fn prepare_iter_inner(
        &self,
        min_key: Option<&[u8]>,
        max_key: Option<&[u8]>,
        reverse: bool,
    ) -> (
        Vec<MemtableIterator>,
        Vec<TableIterator>,
        Option<Vec<u8>>,
        Option<Vec<u8>>,
    ) {
        let mut table_iters = Vec::new();
        let mut mem_iters = Vec::new();

        if let Some(min_key) = &min_key
            && let Some(max_key) = &max_key
        {
            assert!(min_key < max_key);
        }

        {
            let memtable = self.memtable.read().await;
            let imm_mems = self.imm_memtables.read().await;

            mem_iters.push(memtable.clone_immutable().into_iter(reverse).await);

            for (_, imm) in imm_mems.iter() {
                let iter = imm.clone().into_iter(reverse).await;
                mem_iters.push(iter);
            }
        }

        for level in self.levels.iter() {
            let tables = level.get_tables_ro().await;

            for table in tables.iter() {
                let mut skip = false;

                if let Some(min_key) = min_key
                    && table.get_max() < min_key
                {
                    skip = true;
                }

                if let Some(max_key) = max_key
                    && table.get_min() > max_key
                {
                    skip = true;
                }

                if !skip {
                    let iter = TableIterator::new(table.clone(), reverse).await;
                    table_iters.push(iter);
                }
            }
        }

        (
            mem_iters,
            table_iters,
            min_key.map(|k| k.to_vec()),
            max_key.map(|k| k.to_vec()),
        )
    }

    /// Prepares iterators for forward iteration over the specified key range.
    ///
    /// # Arguments
    ///
    /// * `min_key` - Optional minimum key (inclusive)
    /// * `max_key` - Optional maximum key (inclusive)
    ///
    /// # Returns
    ///
    /// Tuple of (memtable iterators, table iterators, min key, max key)
    pub async fn prepare_iter(
        &self,
        min_key: Option<&[u8]>,
        max_key: Option<&[u8]>,
    ) -> (
        Vec<MemtableIterator>,
        Vec<TableIterator>,
        Option<Vec<u8>>,
        Option<Vec<u8>>,
    ) {
        self.prepare_iter_inner(min_key, max_key, false).await
    }

    /// Prepares iterators for reverse iteration over the specified key range.
    ///
    /// # Arguments
    ///
    /// * `max_key` - Optional maximum key (inclusive)
    /// * `min_key` - Optional minimum key (inclusive)
    ///
    /// # Returns
    ///
    /// Tuple of (memtable iterators, table iterators, min key, max key)
    pub async fn prepare_reverse_iter(
        &self,
        max_key: Option<&[u8]>,
        min_key: Option<&[u8]>,
    ) -> (
        Vec<MemtableIterator>,
        Vec<TableIterator>,
        Option<Vec<u8>>,
        Option<Vec<u8>>,
    ) {
        self.prepare_iter_inner(min_key, max_key, true).await
    }

    /// Retrieves a value by key from the database (WiscKey mode).
    ///
    /// Searches in order: active memtable → immutable memtables → levels (L0 → Ln).
    /// Returns the first matching entry found.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up
    ///
    /// # Returns
    ///
    /// - `Ok((compaction_triggered, Some(entry)))`: Value found
    /// - `Ok((compaction_triggered, None))`: Value not found
    ///
    /// The boolean indicates whether seek-based compaction was triggered during this read.
    ///
    /// # Errors
    ///
    /// Returns an error if value log lookup fails.
    #[cfg(feature = "wisckey")]
    #[tracing::instrument(skip(self, key))]
    pub async fn get(&self, key: &[u8]) -> Result<(bool, Option<EntryRef>), Error> {
        // First look in memtable
        {
            let memtable = self.memtable.read().await;

            if let Some(entry) = memtable.get().get(key) {
                return Ok((false, entry.get_entry_ref()));
            }
        }

        // Next check in immutable memtables
        {
            let imm_mems = self.imm_memtables.read().await;

            for (_, imm) in imm_mems.iter().rev() {
                if let Some(entry) = imm.get().get(key) {
                    return Ok((false, entry.get_entry_ref()));
                }
            }
        }

        // Finally check in levels
        let mut compaction_triggered = false;

        for level in self.levels.iter() {
            let (level_compact_triggered, result) = level.get(key).await;
            if level_compact_triggered {
                compaction_triggered = true;
            }

            if let Some(entry) = result {
                return Ok((
                    compaction_triggered,
                    entry.get_entry_ref(&self.value_log).await,
                ));
            }
        }

        // Does not exist
        Ok((compaction_triggered, None))
    }

    /// Retrieves a value by key from the database.
    ///
    /// Searches in order: active memtable → immutable memtables → levels (L0 → Ln).
    /// Returns the first matching entry found.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up
    ///
    /// # Returns
    ///
    /// - `Ok((compaction_triggered, Some(entry)))`: Value found
    /// - `Ok((compaction_triggered, None))`: Value not found
    ///
    /// The boolean indicates whether seek-based compaction was triggered during this read.
    ///
    /// # Errors
    ///
    /// This version does not return errors.
    #[cfg(not(feature = "wisckey"))]
    #[tracing::instrument(skip(self, key))]
    pub async fn get(&self, key: &[u8]) -> Result<(bool, Option<EntryRef>), Error> {
        // First look in memtable
        {
            let memtable = self.memtable.read().await;

            if let Some(entry) = memtable.get().get(key) {
                return Ok((false, entry.get_entry_ref()));
            }
        }

        // Next check in immutable memtables
        {
            let imm_mems = self.imm_memtables.read().await;

            for (_, imm) in imm_mems.iter().rev() {
                if let Some(entry) = imm.get().get(key) {
                    return Ok((false, entry.get_entry_ref()));
                }
            }
        }

        // Finally check in levels
        let mut compaction_triggered = false;

        for level in self.levels.iter() {
            let (level_compact_triggered, result) = level.get(key).await;
            if level_compact_triggered {
                compaction_triggered = true;
            }

            if let Some(entry) = result {
                return Ok((compaction_triggered, entry.get_entry_ref()));
            }
        }

        Ok((compaction_triggered, None))
    }

    /// Flushes the write-ahead log to disk.
    ///
    /// Ensures all buffered writes are persisted to durable storage.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success
    ///
    /// # Errors
    ///
    /// Returns an error if the fsync operation fails.
    pub async fn flush(&self) -> Result<(), Error> {
        self.wal.flush().await?;
        Ok(())
    }

    /// Applies a batch of writes to the database.
    ///
    /// Writes are first appended to the WAL, then applied to the active memtable.
    /// If the memtable becomes full, it is frozen and queued for flushing to L0.
    ///
    /// # Arguments
    ///
    /// * `write_batch` - Batch of put/delete operations to apply
    /// * `opt` - Write options (e.g., whether to flush WAL)
    ///
    /// # Returns
    ///
    /// - `Ok(true)`: Write succeeded and memtable was frozen
    /// - `Ok(false)`: Write succeeded without freezing memtable
    ///
    /// # Errors
    ///
    /// Returns an error if WAL append fails.
    ///
    /// # Backpressure
    ///
    /// This method blocks if the immutable memtable queue is full, providing
    /// natural backpressure to prevent unbounded memory growth.
    #[tracing::instrument(skip(self, write_batch, opt))]
    pub async fn write_opts(
        &self,
        write_batch: WriteBatch,
        opt: &WriteOptions,
    ) -> Result<bool, Error> {
        let mut memtable = self.memtable.write().await;

        // Write the batch to the WAL first
        let wal_offset = self.write_batch_to_wal(&write_batch, opt).await?;

        // Now apply the writes to the memtable
        {
            let mem_inner = memtable.get_mut();
            for op in write_batch.writes {
                op.write_to_memtable(mem_inner);
            }
        }

        // If the current memtable is full, mark it as immutable, so it can be flushed to L0
        self.try_freeze_memtable(memtable, wal_offset).await
    }

    /// Appends a write batch to the write-ahead log.
    ///
    /// # Arguments
    ///
    /// * `write_batch` - Batch of operations to log
    /// * `opt` - Options controlling flush behavior
    ///
    /// # Returns
    ///
    /// WAL offset after the write
    ///
    /// # Errors
    ///
    /// Returns an error if WAL append or flush fails.
    async fn write_batch_to_wal(
        &self,
        write_batch: &WriteBatch,
        opt: &WriteOptions,
    ) -> Result<usize, Error> {
        let writes = write_batch.writes.iter().map(LogEntry::Write);

        let write_pos = self.wal.store(writes).await?;

        if opt.flush {
            self.wal.flush().await?;
        }

        Ok(write_pos)
    }

    /// Checks if the memtable is full and freezes it if necessary.
    ///
    /// When frozen, the memtable is moved to the immutable queue and a new
    /// active memtable is created. Implements backpressure by blocking until
    /// the immutable queue has space.
    ///
    /// # Arguments
    ///
    /// * `memtable` - Write guard for the active memtable
    /// * `wal_offset` - Current WAL offset for recovery coordination
    ///
    /// # Returns
    ///
    /// - `Ok(true)`: Memtable was frozen
    /// - `Ok(false)`: Memtable was not full
    ///
    /// # Errors
    ///
    /// This method does not currently return errors.
    async fn try_freeze_memtable(
        &self,
        mut memtable: RwLockWriteGuard<'_, MemtableRef>,
        wal_offset: usize,
    ) -> Result<bool, Error> {
        // If the current memtable is full, mark it as immutable, so it can be flushed to L0
        let mem_inner = memtable.get_mut();
        if mem_inner.is_full(&self.params) {
            let next_seq_num = mem_inner.get_next_seq_number();
            let imm = memtable.take(next_seq_num);
            let mut imm_mems = self.imm_memtables.write().await;

            // The following while loop implements backpressure to prevent unbounded memory growth.
            //
            // What it does:
            // The loop waits until the imm_memtables (immutable memtables) queue is empty before allowing the current memtable to become immutable.
            //
            // How it works:
            // 1. Check if queue is full: If imm_memtables already contains memtables waiting to be flushed to disk
            // 2. Wait for space: Calls rw_write_wait() which:
            // * Releases the write lock on imm_memtables
            // * Waits on the condition variable imm_cond
            // *Re-acquires the write lock when notified
            // 3.Retry: Loops until the queue is empty
            //
            // Why it's needed:
            // Prevents the database from accepting writes faster than it can flush them to disk
            // Without this, you could accumulate many immutable memtables in memory, leading to OOM
            // The condition variable is notified in flush_frozen_memtable() after a memtable is successfully flushed:
            //
            // self.imm_cond.notify_all();
            //
            // This creates a natural throttling mechanism where write operations block when the system can't keep up with compaction.
            while !imm_mems.is_empty() {
                imm_mems = self
                    .imm_cond
                    .rw_write_wait(&self.imm_memtables, imm_mems)
                    .await;
            }

            imm_mems.push_back((wal_offset, imm));

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Flushes an immutable memtable to a new L0 sorted table.
    ///
    /// Takes the oldest immutable memtable from the queue, writes it to disk as a
    /// sorted table, updates the manifest, prunes the WAL, and notifies waiting writers.
    ///
    /// # Returns
    ///
    /// - `Ok(true)`: Successfully flushed a memtable
    /// - `Ok(false)`: No memtable to flush
    ///
    /// # Errors
    ///
    /// Returns an error if table creation, manifest update, or WAL operations fail.
    #[tracing::instrument(skip(self))]
    pub async fn flush_frozen_memtable(&self) -> Result<bool, Error> {
        log::trace!("Flushing frozen memtable");

        // SAFETY
        // Only one task will flush frozen memtables, so it is
        // fine to not hold the lock the entire time

        let to_flush = self.imm_memtables.read().await.front().cloned();

        if let Some((log_offset, frozen_memtable)) = to_flush {
            log::trace!("Found memtable to flush");

            // First create table
            let (min_key, max_key) = frozen_memtable.get().get_min_max_key();
            let l0 = self.levels.first().unwrap();
            let table_id = self.manifest.generate_next_table_id().await;
            let mut table_builder =
                l0.create_table_builder(table_id, min_key.to_vec(), max_key.to_vec());

            let memtable_entries = frozen_memtable.get().get_entries();

            cfg_if! {
                if #[cfg(feature="wisckey")] {
                    let mut vbuilder = self.value_log.make_batch().await;

                    for (key, entry) in memtable_entries {
                        match entry {
                            MemtableEntry::Value{seq_number, value} => {
                                let value_ref = vbuilder.add_entry(&key, &value).await;
                                table_builder.add_value(&key, seq_number, value_ref).await?;
                            }
                            MemtableEntry::Deletion{seq_number} => {
                                table_builder.add_deletion(&key, seq_number).await?;
                            }
                        }
                    }

                    vbuilder.finish().await?;
                } else {
                    for (key, entry) in memtable_entries {
                        match entry {
                            MemtableEntry::Value{seq_number, value} => {
                                table_builder.add_value(&key, seq_number, &value).await?;
                            }
                            MemtableEntry::Deletion{seq_number} => {
                                table_builder.add_deletion(&key, seq_number).await?;
                            }
                        }
                    }
                }
            }

            let table = table_builder.finish().await?;
            l0.add_l0_table(table).await;

            if let Some(logger) = &self.level_logger {
                logger.l0_table_added();
            }

            // Flush all value index changes to disk
            #[cfg(feature = "wisckey")]
            self.value_log.flush().await?;

            // Then update manifest and flush WAL
            let seq_offset = frozen_memtable.get().get_next_seq_number();
            self.manifest.set_seq_number_offset(seq_offset).await;
            self.manifest
                .update_table_set(vec![(0, table_id)], vec![])
                .await;

            self.wal.prune_wal(log_offset).await;
            self.manifest.set_log_offset(log_offset).await;

            // Finally, remove immutable memtable
            {
                let mut imm_mems = self.imm_memtables.write().await;
                let entry = imm_mems.pop_front();
                assert!(entry.is_some());
            }
            log::debug!("Created new L0 table");
            self.imm_cond.notify_all();

            Ok(true)
        } else {
            log::trace!("Found no frozen memtable to flush");
            Ok(false)
        }
    }

    /// Attempts level-to-level compaction across all levels.
    ///
    /// Iterates through levels L0 to Ln-1, attempting to compact each level with
    /// the level below it. Stops at the first successful compaction.
    ///
    /// # Returns
    ///
    /// - `Ok(true)`: Compaction completed or retry recommended (lock contention)
    /// - `Ok(false)`: No compaction was needed on any level
    ///
    /// # Retry Logic
    ///
    /// Returns `true` in two cases:
    /// 1. Compaction succeeded → there might be more work
    /// 2. Lock contention occurred → retry may succeed
    ///
    /// # Errors
    ///
    /// Returns an error if compaction operations fail (I/O, manifest updates, etc.).
    #[tracing::instrument(skip(self))]
    pub async fn do_level_compaction(&self) -> Result<bool, Error> {
        let mut was_locked = false;
        log::trace!("Attempting level compaction");

        // level-to-level compaction
        for levels in self.levels.windows(2) {
            let (parent, child) = (&levels[0], &levels[1]);
            let level_id = parent.get_index();
            match self.compact_level(parent, child).await? {
                CompactResult::DidWork => {
                    log::trace!("Compacted level {level_id}");
                    return Ok(true);
                }
                CompactResult::Locked => {
                    log::trace!("Cannot compact level {level_id} right now; lock was held");
                    was_locked = true;
                }
                CompactResult::NothingToDo => {
                    log::trace!("Nothing to do for level {level_id}");
                }
            }
        }

        // We'll try again if it was locked
        Ok(was_locked)
    }

    /// Compacts selected tables from a parent level to the child level below.
    ///
    /// # Compaction Strategies
    ///
    /// 1. **Fast compaction**: Single table with no overlaps → simply move to next level
    /// 2. **Merge compaction**: Merge overlapping tables using k-way merge
    /// 3. **Abort**: Lock contention or concurrent compaction detected
    ///
    /// # Arguments
    ///
    /// * `parent_level` - Source level to compact from
    /// * `child_level` - Destination level (must be `parent_level.index + 1`)
    ///
    /// # Returns
    ///
    /// - `Ok(CompactResult::DidWork)`: Compaction completed
    /// - `Ok(CompactResult::NothingToDo)`: No compaction needed or aborted
    /// - `Ok(CompactResult::Locked)`: Aborted due to lock contention
    ///
    /// # Errors
    ///
    /// Returns an error if table building, I/O, or manifest updates fail.
    #[tracing::instrument(skip(self, parent_level, child_level))]
    async fn compact_level(
        &self,
        parent_level: &Level,
        child_level: &Level,
    ) -> Result<CompactResult, Error> {
        assert_eq!(parent_level.get_index() + 1, child_level.get_index());

        let parent_tables_to_compact = match parent_level.maybe_start_compaction().await {
            Ok(Some(result)) => result,
            Ok(None) => return Ok(CompactResult::NothingToDo),
            Err(()) => return Ok(CompactResult::Locked),
        };
        assert!(!parent_tables_to_compact.is_empty());

        log::trace!("Starting compaction on level {}", parent_level.get_index());

        let (min_key, max_key) = parent_tables_to_compact.iter().fold(
            (
                parent_tables_to_compact[0].get_min(),
                parent_tables_to_compact[0].get_max(),
            ),
            |(min, max), table| (min.min(table.get_min()), max.max(table.get_max())),
        );

        let overlap_result = if parent_tables_to_compact.len() == 1 {
            child_level
                .get_overlaps(min_key, max_key, Some(parent_tables_to_compact[0].get_id()))
                .await
        } else {
            child_level.get_overlaps(min_key, max_key, None).await
        };

        // Abort due to concurrency?
        let (table_id, child_tables_to_compact) = match overlap_result {
            Some(res) => res,
            None => {
                log::trace!("Aborting compaction due to concurrency");
                for parent_table in parent_tables_to_compact {
                    parent_table.stop_compaction();
                }
                return Ok(CompactResult::NothingToDo);
            }
        };

        // Fast path
        if parent_tables_to_compact.len() == 1 && child_tables_to_compact.is_empty() {
            assert_eq!(parent_tables_to_compact[0].get_id(), table_id);
            self.fast_compaction(parent_level, child_level, table_id)
                .await;
            return Ok(CompactResult::DidWork);
        }

        // At this point, the compaction flag/lock has been set on all affected tables
        // and a placeholder was created on the child level

        log::debug!(
            "Compacting {} table(s) in level {} with {} table(s) in level {} into table #{table_id}",
            parent_tables_to_compact.len(),
            parent_level.get_index(),
            child_tables_to_compact.len(),
            child_level.get_index(),
        );

        let (min_key, max_key) = child_tables_to_compact
            .iter()
            .fold((min_key, max_key), |(min, max), table| {
                (min.min(table.get_min()), max.max(table.get_max()))
            });

        // Table can potentially contain a single entry
        assert!(min_key <= max_key);

        let min_key = min_key.to_vec();
        let max_key = max_key.to_vec();

        let mut table_iters = Vec::new();
        for table in parent_tables_to_compact.iter() {
            table_iters.push(TableIterator::new(table.clone(), false).await);
        }

        for child in child_tables_to_compact.iter() {
            table_iters.push(TableIterator::new(child.clone(), false).await);
        }

        let mut last_key: Option<Key> = None;

        #[cfg(feature = "wisckey")]
        let mut deleted_values = vec![];

        let mut table_builder = child_level.create_table_builder(table_id, min_key, max_key);

        loop {
            log::trace!("Starting compaction for next key");
            let mut min_key: Option<Vec<u8>> = None;

            for table_iter in table_iters.iter_mut() {
                // Advance the iterator, if needed
                if let Some(last_key) = &last_key {
                    while !table_iter.at_end() && table_iter.get_key() <= last_key.as_slice() {
                        table_iter.step().await;
                    }
                }

                if !table_iter.at_end() {
                    if let Some(key) = &min_key {
                        if table_iter.get_key() < key.as_slice() {
                            min_key = Some(table_iter.get_key().to_vec());
                        }
                    } else {
                        min_key = Some(table_iter.get_key().to_vec());
                    }
                }
            }

            if min_key.is_none() {
                break;
            }

            let mut min_iter: Option<&TableIterator> = None;
            let min_key = min_key.unwrap().clone();

            for table_iter in table_iters.iter_mut() {
                if table_iter.at_end() {
                    continue;
                }

                // Figure out if this table's entry is more recent
                let key = table_iter.get_key();

                if key != min_key {
                    continue;
                }

                if let Some(other_iter) = min_iter {
                    if table_iter.get_seq_number() > other_iter.get_seq_number() {
                        log::trace!(
                            "Overriding key {key:?}: new seq #{}, old seq #{}",
                            table_iter.get_seq_number(),
                            other_iter.get_seq_number()
                        );

                        // Check whether we overwrote a key that is about to
                        // be garbage collected
                        #[cfg(feature = "wisckey")]
                        deleted_values.push(other_iter.get_value_id().unwrap());

                        min_iter = Some(table_iter);
                    }
                } else {
                    log::trace!("Found new key {key:?}");
                    min_iter = Some(table_iter);
                }
            }

            let min_iter = min_iter.unwrap();
            match min_iter.get_entry_type() {
                DataEntryType::Put => {
                    table_builder
                        .add_value(
                            &min_key,
                            min_iter.get_seq_number(),
                            #[cfg(feature = "wisckey")]
                            min_iter.get_value_id().unwrap(),
                            #[cfg(not(feature = "wisckey"))]
                            min_iter.get_entry().unwrap().get_value(),
                        )
                        .await?;
                }
                DataEntryType::Delete => {
                    table_builder
                        .add_deletion(&min_key, min_iter.get_seq_number())
                        .await?;
                }
            }

            last_key = Some(min_key.to_vec());
        }

        let new_table = table_builder.finish().await?;

        let add_set = vec![(child_level.get_index(), new_table.get_id())];
        let mut remove_set = vec![];

        // Update tables atomically
        let mut all_parent_tables = parent_level.get_tables_rw().await;
        let mut all_child_tables = child_level.get_tables_rw().await;

        // Remove all previous child tables
        for table in child_tables_to_compact.iter() {
            let mut found = false;
            for (pos, other_table) in all_child_tables.iter().enumerate() {
                if other_table.get_id() == table.get_id() {
                    remove_set.push((child_level.get_index(), table.get_id()));
                    all_child_tables.remove(pos);
                    found = true;
                    break;
                }
            }
            assert!(found);
        }

        // Find position for new child table
        let mut new_pos = all_child_tables.len();
        for (pos, other_table) in all_child_tables.iter().enumerate() {
            if other_table.get_min() > new_table.get_min() {
                new_pos = pos;
                break;
            }
        }

        // Add new table to child level
        all_child_tables.insert(new_pos, Arc::new(new_table));
        child_level.remove_table_placeholder(table_id).await;

        // Remove table entries from parent level
        for table in parent_tables_to_compact.iter() {
            let mut found = false;
            for (pos, other_table) in all_parent_tables.iter().enumerate() {
                if other_table.get_id() == table.get_id() {
                    remove_set.push((parent_level.get_index(), table.get_id()));
                    all_parent_tables.remove(pos);
                    found = true;
                    break;
                }
            }
            assert!(found);
        }

        #[cfg(feature = "wisckey")]
        {
            let mut reinsert = WriteBatch::new();
            for vid in deleted_values.into_iter() {
                let entries = self.value_log.mark_value_deleted(vid).await?;
                for (k, v) in entries.into_iter() {
                    reinsert.put(k, v);
                }
            }

            // Reinsert values, if needed to defragment
            // Old batches will eventually be removed as a result
            if !reinsert.writes.is_empty() {
                let opts = WriteOptions { flush: true };
                self.write_opts(reinsert, &opts).await?;
            }
        }

        if let Some(logger) = &self.level_logger {
            logger.compaction(parent_level.get_index(), add_set.len(), remove_set.len());
        }

        self.manifest.update_table_set(add_set, remove_set).await;

        log::trace!("Done compacting tables");
        Ok(CompactResult::DidWork)
    }

    /// Stops the database and flushes pending data.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success
    ///
    /// # Errors
    ///
    /// Returns an error if WAL shutdown fails.
    pub async fn stop(&self) -> Result<(), Error> {
        self.wal.stop().await
    }

    /// Performs fast compaction by moving a table to the next level without merging.
    ///
    /// Used when a single table has no overlaps in the child level. Instead of
    /// creating a new table, the existing table is simply moved.
    ///
    /// # Arguments
    ///
    /// * `parent_level` - Source level
    /// * `child_level` - Destination level
    /// * `table_id` - ID of the table to move
    async fn fast_compaction(&self, parent_level: &Level, child_level: &Level, table_id: TableId) {
        let mut all_parent_tables = parent_level.get_tables_rw().await;
        let mut all_child_tables = child_level.get_tables_rw().await;

        // Remove table entry from parent level
        let table = {
            if let Some(i) = all_parent_tables
                .iter()
                .position(|table| table.get_id() == table_id)
            {
                all_parent_tables.remove(i)
            } else {
                panic!("Entry for parent table not found");
            }
        };

        log::debug!(
            "Moving table #{} from level {} to level {}",
            table_id,
            parent_level.get_index(),
            child_level.get_index(),
        );

        // Figure out where to place the table on the child lavel
        let new_pos = all_child_tables
            .iter()
            .position(|child_table| child_table.get_min() > table.get_min())
            .unwrap_or_default();

        // Add table to child level
        all_child_tables.insert(new_pos, table.clone());
        child_level.remove_table_placeholder(table_id).await;

        if let Some(pos) = all_parent_tables
            .iter()
            .position(|parent_table| parent_table.get_id() == table.get_id())
        {
            all_parent_tables.remove(pos);
        }

        // Update manifest
        let add_set = vec![(child_level.get_index(), table.get_id())];
        let remove_set = vec![(parent_level.get_index(), table.get_id())];
        self.manifest.update_table_set(add_set, remove_set).await;

        if let Some(logger) = &self.level_logger {
            logger.compaction(parent_level.get_index(), 1, 1);
        }

        // Unlock table
        table.stop_compaction();

        log::trace!("Done moving table #{table_id}");
    }
}

#[cfg(all(test, not(feature = "wisckey")))]
mod tests {
    use std::sync::Arc;

    use tempfile::TempDir;

    use tokio::test as async_test;

    use crate::StartMode;
    use crate::params::Params;

    use super::{CompactResult, DbLogic};

    async fn test_init() -> (TempDir, DbLogic) {
        let _ = env_logger::builder().is_test(true).try_init();

        let tmpdir = tempfile::Builder::new()
            .prefix("lsm-logic-test-")
            .tempdir()
            .unwrap();

        let params = Params {
            db_path: tmpdir.path().to_path_buf(),
            ..Default::default()
        };

        let logic = DbLogic::new(StartMode::CreateOrOverride, params)
            .await
            .unwrap();

        (tmpdir, logic)
    }

    async fn test_cleanup(tmpdir: TempDir, logic: DbLogic) {
        logic.stop().await.unwrap();

        drop(logic);
        drop(tmpdir);
    }

    /// Here we create overlapping tables on both level 0 and 1
    /// This checks if compaction also works if there is a cross-level overlap   
    #[async_test]
    async fn compact_with_child_level() {
        let (tempdir, logic) = test_init().await;
        let num_tables = 6;

        // Create five tables with the exact same key entries
        for idx in 0..num_tables {
            let level = if idx < num_tables - 1 {
                &logic.levels[0]
            } else {
                &logic.levels[1]
            };

            let table_id = logic.manifest.generate_next_table_id().await;

            let min_key = "000".to_string().into_bytes();
            let max_key = "100".to_string().into_bytes();

            let mut table_builder = level.create_table_builder(table_id, min_key, max_key);
            let mut seq_offset = 1;

            for num in 0..=100 {
                let key = format!("{num:03}").into_bytes();
                let value = "somevalue".to_string().into_bytes();
                let seq_number = seq_offset;
                seq_offset += 1;

                table_builder
                    .add_value(&key, seq_number, &value)
                    .await
                    .unwrap();
            }

            let table = table_builder.finish().await.unwrap();
            let table_id = table.get_id();

            level.get_tables_rw().await.push(Arc::new(table));

            logic
                .manifest
                .update_table_set(vec![(level.get_index(), table_id)], vec![])
                .await;
        }

        assert_eq!(logic.levels[0].get_tables_ro().await.len(), num_tables - 1);
        assert_eq!(
            logic.manifest.get_table_ids().await[0].len(),
            num_tables - 1
        );
        assert_eq!(logic.levels[1].get_tables_ro().await.len(), 1);
        assert_eq!(logic.manifest.get_table_ids().await[1].len(), 1);

        let old_table_id = logic.levels[1].get_tables_ro().await[0].get_id();

        let did_work = logic.do_level_compaction().await.unwrap();
        assert!(did_work);

        assert!(logic.levels[0].get_tables_ro().await.is_empty());
        assert!(logic.manifest.get_table_ids().await[0].is_empty());
        assert_eq!(logic.levels[1].get_tables_ro().await.len(), 1);
        assert_eq!(logic.manifest.get_table_ids().await[1].len(), 1);

        // Ensure a new table was created
        let new_table_id = logic.levels[1].get_tables_ro().await[0].get_id();
        assert_ne!(old_table_id, new_table_id);

        test_cleanup(tempdir, logic).await;
    }

    /// This adds multiple overlapping tables to L0 and expects them to be
    /// merged into one table in L1
    #[async_test]
    async fn l0_compaction() {
        let (tempdir, logic) = test_init().await;

        let num_tables = 5;

        // Create five tables with the exact same key entries
        for _ in 0..num_tables {
            let l0 = logic.levels.first().unwrap();
            let table_id = logic.manifest.generate_next_table_id().await;

            let min_key = "000".to_string().into_bytes();
            let max_key = "100".to_string().into_bytes();

            let mut table_builder = l0.create_table_builder(table_id, min_key, max_key);
            let mut seq_offset = 1;

            for num in 0..=100 {
                let key = format!("{num:03}").into_bytes();
                let value = "somevalue".to_string().into_bytes();
                let seq_number = seq_offset;
                seq_offset += 1;

                table_builder
                    .add_value(&key, seq_number, &value)
                    .await
                    .unwrap();
            }

            let table = table_builder.finish().await.unwrap();
            let table_id = table.get_id();

            l0.add_l0_table(table).await;

            // Then update manifest and flush WAL
            logic
                .manifest
                .update_table_set(vec![(l0.get_index(), table_id)], vec![])
                .await;
        }

        assert_eq!(logic.levels[0].get_tables_ro().await.len(), num_tables);
        assert_eq!(logic.manifest.get_table_ids().await[0].len(), num_tables);
        assert!(logic.levels[1].get_tables_ro().await.is_empty());
        assert!(logic.manifest.get_table_ids().await[1].is_empty());

        let did_work = logic.do_level_compaction().await.unwrap();
        assert!(did_work);

        assert!(logic.levels[0].get_tables_ro().await.is_empty());
        assert!(logic.manifest.get_table_ids().await[0].is_empty());
        assert_eq!(logic.levels[1].get_tables_ro().await.len(), 1);
        assert_eq!(logic.manifest.get_table_ids().await[1].len(), 1);

        test_cleanup(tempdir, logic).await;
    }

    /// Test that fast compaction (simply moving a table down) works as expected
    ///
    /// Note: This test makes some assumptions about the inner workings of
    /// DbLogic and might need to be adjusted with future changes
    #[async_test]
    async fn fast_compaction() {
        let (tempdir, logic) = test_init().await;

        let num_tables = 10;

        // Create five tables with the exact same key entries
        for idx in 0..num_tables {
            let l0 = logic.levels.first().unwrap();
            let table_id = logic.manifest.generate_next_table_id().await;

            let pos = idx * 100;
            let next_pos = (idx + 1) * 100 - 1;

            let min_key = format!("{pos:04}").into_bytes();
            let max_key = format!("{next_pos:04}").into_bytes();

            let mut table_builder = l0.create_table_builder(table_id, min_key, max_key);
            let mut seq_offset = 1;

            for num in pos..next_pos {
                let key = format!("{num:04}").into_bytes();
                let value = "somevalue".to_string().into_bytes();
                let seq_number = seq_offset;
                seq_offset += 1;

                table_builder
                    .add_value(&key, seq_number, &value)
                    .await
                    .unwrap();
            }

            let table = table_builder.finish().await.unwrap();
            let table_id = table.get_id();
            l0.add_l0_table(table).await;

            // Then update manifest and flush WAL
            logic
                .manifest
                .update_table_set(vec![(0, table_id)], vec![])
                .await;
        }

        assert_eq!(logic.levels[0].get_tables_ro().await.len(), num_tables);
        assert_eq!(logic.manifest.get_table_ids().await[0].len(), num_tables);
        assert!(logic.manifest.get_table_ids().await[1].is_empty());

        let did_work = logic.do_level_compaction().await.unwrap();
        assert!(did_work);

        // One table should have moved down
        assert_eq!(logic.levels[0].get_tables_ro().await.len(), num_tables - 1);
        assert_eq!(
            logic.manifest.get_table_ids().await[0].len(),
            num_tables - 1
        );
        assert_eq!(logic.levels[1].get_tables_ro().await.len(), 1);
        assert_eq!(logic.manifest.get_table_ids().await[1].len(), 1);

        let did_work = logic.do_level_compaction().await.unwrap();
        assert!(did_work);

        assert_eq!(
            logic.manifest.get_table_ids().await[0].len(),
            num_tables - 2
        );
        assert_eq!(logic.levels[1].get_tables_ro().await.len(), 2);
        assert_eq!(logic.manifest.get_table_ids().await[1].len(), 2);

        // Ensure no tables exist on both levels
        for table0 in logic.levels[0].get_tables_ro().await.iter() {
            for table1 in logic.levels[1].get_tables_ro().await.iter() {
                assert_ne!(table0.get_id(), table1.get_id());
            }
        }

        test_cleanup(tempdir, logic).await;
    }

    /// Test that no compaction happens if tables are already marked with a compaction flag
    #[async_test]
    async fn compaction_flag() {
        let (tempdir, logic) = test_init().await;

        let num_tables = 5;

        // Create five tables with the exact same key entries
        for _ in 0..num_tables {
            let l0 = logic.levels.first().unwrap();
            let table_id = logic.manifest.generate_next_table_id().await;

            let min_key = "000".to_string().into_bytes();
            let max_key = "100".to_string().into_bytes();

            let mut table_builder = l0.create_table_builder(table_id, min_key, max_key);
            let mut seq_offset = 1;

            for num in 0..=100 {
                let key = format!("{num:03}").into_bytes();
                let value = "somevalue".to_string().into_bytes();
                let seq_number = seq_offset;
                seq_offset += 1;

                table_builder
                    .add_value(&key, seq_number, &value)
                    .await
                    .unwrap();
            }

            let table = table_builder.finish().await.unwrap();
            let table_id = table.get_id();

            let could_set_flag = table.start_compaction();
            assert!(could_set_flag);

            l0.add_l0_table(table).await;

            // Then update manifest and flush WAL
            logic
                .manifest
                .update_table_set(vec![(0, table_id)], vec![])
                .await;
        }

        assert_eq!(logic.levels[0].get_tables_ro().await.len(), num_tables);
        assert_eq!(logic.manifest.get_table_ids().await[0].len(), num_tables);
        assert!(logic.manifest.get_table_ids().await[1].is_empty());

        let result = logic
            .compact_level(&logic.levels[0], &logic.levels[1])
            .await
            .unwrap();
        assert_eq!(result, CompactResult::Locked);

        test_cleanup(tempdir, logic).await;
    }

    #[async_test]
    async fn fast_compaction_with_offset() {
        let (tempdir, logic) = test_init().await;

        let num_tables = 10;

        // Create five tables with the exact same key entries
        for idx in 0..num_tables {
            let l0 = logic.levels.first().unwrap();
            let table_id = logic.manifest.generate_next_table_id().await;

            let pos = idx * 100;
            let next_pos = (idx + 1) * 100 - 1;

            let min_key = format!("{pos:04}").into_bytes();
            let max_key = format!("{next_pos:04}").into_bytes();

            let mut table_builder = l0.create_table_builder(table_id, min_key, max_key);
            let mut seq_offset = 1;

            for num in pos..next_pos {
                let key = format!("{num:04}").into_bytes();
                let value = "somevalue".to_string().into_bytes();
                let seq_number = seq_offset;
                seq_offset += 1;

                table_builder
                    .add_value(&key, seq_number, &value)
                    .await
                    .unwrap();
            }

            let table = table_builder.finish().await.unwrap();
            let table_id = table.get_id();
            l0.add_l0_table(table).await;

            // Then update manifest and flush WAL
            logic
                .manifest
                .update_table_set(vec![(0, table_id)], vec![])
                .await;
        }

        // Check that compaction works fine if it is not the first table that gets pushed down
        logic.levels[0].set_next_compaction_offset(3);

        assert_eq!(logic.levels[0].get_tables_ro().await.len(), num_tables);
        assert_eq!(logic.manifest.get_table_ids().await[0].len(), num_tables);
        assert!(logic.manifest.get_table_ids().await[1].is_empty());

        let did_work = logic.do_level_compaction().await.unwrap();
        assert!(did_work);

        // One table should have moved down
        assert_eq!(logic.levels[0].get_tables_ro().await.len(), num_tables - 1);
        assert_eq!(
            logic.manifest.get_table_ids().await[0].len(),
            num_tables - 1
        );
        assert_eq!(logic.levels[1].get_tables_ro().await.len(), 1);
        assert_eq!(logic.manifest.get_table_ids().await[1].len(), 1);

        let did_work = logic.do_level_compaction().await.unwrap();
        assert!(did_work);

        assert_eq!(
            logic.manifest.get_table_ids().await[0].len(),
            num_tables - 2
        );
        assert_eq!(logic.levels[1].get_tables_ro().await.len(), 2);
        assert_eq!(logic.manifest.get_table_ids().await[1].len(), 2);

        // Ensure no tables exist on both levels
        for table0 in logic.levels[0].get_tables_ro().await.iter() {
            for table1 in logic.levels[1].get_tables_ro().await.iter() {
                assert_ne!(table0.get_id(), table1.get_id());
            }
        }

        test_cleanup(tempdir, logic).await;
    }
}
