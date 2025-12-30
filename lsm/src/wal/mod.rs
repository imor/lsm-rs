//! Write-Ahead Log (WAL) Implementation
//!
//! The Write-Ahead Log is a critical component for ensuring durability and crash recovery
//! in the LSM-tree database. Before any write operation is applied to the in-memory memtable,
//! it is first written to the WAL on disk. This ensures that even if the system crashes
//! before the memtable is flushed to disk, the data can be recovered by replaying the WAL.
//!
//! # Architecture
//!
//! The WAL consists of three main components:
//! - [`WriteAheadLog`]: The main interface for logging operations
//! - [`WalWriter`]: A background task that handles actual disk I/O
//! - [`WalReader`]: Reads WAL files during recovery
//!
//! # WAL Entry Format
//!
//! Each log entry consists of:
//! 1. **Entry Type** (1 byte): Identifies the type of operation
//!    - `0`: Write operation (Put or Delete)
//!    - `1`: Delete value (Wisckey feature)
//!    - `2`: Delete batch (Wisckey feature)
//! 2. **Operation Data**: The specific data for the operation
//!    - For Write: operation type (1 byte), key length (8 bytes), key data, value length (8 bytes, for Put), value data (for Put)
//!    - For Delete operations: page ID and offset
//!
//! # File Organization
//!
//! The WAL is organized into fixed-size pages (4KB each) stored as separate files:
//! - Files are named sequentially: `00000001.wal`, `00000002.wal`, etc.
//! - When a page fills up, a new file is created
//! - Old files are garbage collected after the data has been flushed to SSTable files
//!
//! # Crash Recovery
//!
//! During database startup:
//! 1. The WAL reader scans all WAL files from the last checkpoint
//! 2. Each entry is parsed and replayed into the memtable
//! 3. For Wisckey mode, value index updates are also replayed
//! 4. Once recovery is complete, normal operations resume
//!
//! # Durability Guarantees
//!
//! - Writes are buffered in memory for performance
//! - Explicit `sync()` calls ensure data is persisted to disk via `fsync`
//! - The WAL guarantees write ordering: entries are written in strict sequential order
//! - After a successful `sync()`, all preceding writes are guaranteed to survive a crash
//!
//! # Lifecycle
//!
//! 1. **Creation**: WAL is initialized when the database opens
//! 2. **Writing**: Operations are logged as they occur
//! 3. **Syncing**: Periodic or explicit syncs ensure durability
//! 4. **Pruning**: After memtable flush, old WAL entries are deleted
//! 5. **Shutdown**: Graceful shutdown ensures all pending writes complete

#![allow(clippy::await_holding_lock)]

use std::sync::Arc;

use parking_lot::{Mutex, RwLock};
use tokio::sync::{Notify, oneshot};

use zerocopy::IntoBytes;

#[cfg(feature = "wisckey")]
use crate::values::{ValueIndex, ValueIndexPageId};

use crate::memtable::Memtable;
use crate::{Error, Params, WriteOp};

mod writer;
use writer::WalWriter;

mod reader;
pub use reader::RecoveryResult;
use reader::WalReader;

#[cfg(test)]
mod tests;

/// Represents a single entry in the write-ahead log.
///
/// In the vanilla LSM-tree configuration, the log only stores write operations
/// (Put and Delete) to maintain durability of key-value pairs.
///
/// When the Wisckey feature is enabled, the WAL also stores metadata about
/// value index operations. This reduces write amplification by allowing the
/// value index to be updated asynchronously while maintaining crash recovery
/// guarantees.
///
/// # Entry Types
///
/// - [`LogEntry::Write`]: Records a Put or Delete operation on a key
/// - [`LogEntry::DeleteBatch`]: (Wisckey only) Marks an entire value batch for deletion
/// - [`LogEntry::DeleteValue`]: (Wisckey only) Marks a single value for deletion
pub enum LogEntry<'a> {
    /// A write operation (Put or Delete) that modifies the key-value store.
    ///
    /// This entry type stores the complete operation data including the key
    /// and value (for Put operations). During recovery, these operations are
    /// replayed into the memtable to restore the database state.
    Write(&'a WriteOp),
    /// Marks an entire batch of values for deletion in the value log.
    ///
    /// This is a Wisckey-specific operation that records the deletion of a value batch.
    /// The tuple contains:
    /// - `ValueIndexPageId`: The page identifier where the batch is located
    /// - `u16`: The offset within the page
    ///
    /// During recovery, this ensures that deleted batches remain marked as deleted.
    #[cfg(feature = "wisckey")]
    DeleteBatch(ValueIndexPageId, u16),
    /// Marks a single value for deletion in the value log.
    ///
    /// This is a Wisckey-specific operation that records the deletion of a single value.
    /// The tuple contains:
    /// - `ValueIndexPageId`: The page identifier where the value is located
    /// - `u16`: The offset within the page
    ///
    /// During recovery, this ensures that deleted values remain marked as deleted.
    #[cfg(feature = "wisckey")]
    DeleteValue(ValueIndexPageId, u16),
}

/// Internal representation of log entry types as single bytes.
///
/// This enum is serialized directly to disk as a single byte at the start of
/// each WAL entry. The u8 representation allows efficient storage and parsing:
/// - `0`: Write operation
/// - `1`: Delete value operation (Wisckey)
/// - `2`: Delete batch operation (Wisckey)
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
enum LogEntryType {
    Write,
    DeleteValue,
    DeleteBatch,
}

impl TryFrom<u8> for LogEntryType {
    type Error = ();

    fn try_from(val: u8) -> Result<LogEntryType, ()> {
        match val {
            0 => Ok(LogEntryType::Write),
            1 => Ok(LogEntryType::DeleteValue),
            2 => Ok(LogEntryType::DeleteBatch),
            _ => Err(()),
        }
    }
}

impl LogEntry<'_> {
    /// Returns the entry type for serialization.
    ///
    /// Used to determine which byte value to write at the start of the
    /// serialized entry in the WAL file.
    fn get_type(&self) -> LogEntryType {
        match self {
            Self::Write(_) => LogEntryType::Write,
            #[cfg(feature = "wisckey")]
            Self::DeleteValue(_, _) => LogEntryType::DeleteValue,
            #[cfg(feature = "wisckey")]
            Self::DeleteBatch(_, _) => LogEntryType::DeleteBatch,
        }
    }
}

/// The size of each WAL page (file) in bytes.
///
/// The log is split into individual files (pages) of 4KB each. This size:
/// - Allows for efficient I/O operations
/// - Enables granular garbage collection of old log data
/// - Matches common filesystem block sizes for optimal performance
///
/// When a page fills up, a new file is created. Old pages can be deleted
/// once the memtable containing their data has been flushed to SSTables.
const PAGE_SIZE: usize = 4 * 1024;

/// Internal state of the write-ahead log shared between the writer task and WAL object.
///
/// This structure tracks the progress of write operations through various stages:
/// from being queued, to being written to disk, to being synced, and finally pruned.
///
/// # Thread Safety
///
/// Protected by a `RwLock` in `LogInner` to allow concurrent reads while ensuring
/// exclusive writes. This is safe because:
/// - Multiple threads can check positions concurrently
/// - Only the writer task modifies write positions
/// - User threads only modify queue positions
///
/// # Invariants
///
/// The following invariants must always hold:
/// - `sync_pos <= write_pos <= queue_pos`: Synced data must be written, written data must be queued
/// - `prune_pos <= can_prune_pos`: We can only prune up to what's been marked for pruning
///
struct LogStatus {
    /// Absolute byte position of the last queued write operation.
    ///
    /// This represents the end position after all currently queued data
    /// has been written. Incremented when operations are added to the queue.
    queue_pos: usize,

    /// Absolute byte position of the last write operation fulfilled to disk.
    ///
    /// This represents how much data has actually been written to the WAL file,
    /// though not necessarily synced. Updated by the writer task after writes complete.
    write_pos: usize,

    /// Absolute byte position of the last fsync operation.
    ///
    /// Data up to this position is guaranteed to be persisted to disk and will
    /// survive a crash. Updated after successful fsync operations.
    sync_pos: usize,

    /// Queue of pending data buffers to be written.
    ///
    /// Each vector contains serialized log entry data. The writer task consumes
    /// this queue and writes the data to disk.
    queue: Vec<Vec<u8>>,

    /// Position up to which the WAL can be pruned.
    ///
    /// Set when a memtable is flushed to indicate that WAL entries before this
    /// position are no longer needed for recovery.
    can_prune_pos: usize,

    /// Position up to which the WAL has actually been pruned.
    ///
    /// Old WAL files (pages) before this position have been deleted from disk.
    /// Updated by the writer task after file deletion.
    prune_pos: usize,

    /// Flag indicating that an fsync has been requested.
    ///
    /// Set when `sync()` is called, cleared after the sync completes.
    sync_requested: bool,

    /// Flag indicating that the WAL should shut down gracefully.
    ///
    /// Set during database shutdown to signal the writer task to terminate
    /// after completing all pending operations.
    stop_requested: bool,
}

impl LogStatus {
    /// Creates a new log status initialized to the given position.
    ///
    /// # Arguments
    ///
    /// * `position` - The current position in the WAL (all operations are at this point)
    /// * `start_position` - The position where pruning starts (typically after recovery)
    fn new(position: usize, start_position: usize) -> Self {
        Self {
            queue_pos: position,
            write_pos: position,
            sync_pos: position,
            prune_pos: start_position,
            can_prune_pos: start_position,
            queue: vec![],
            sync_requested: false,
            stop_requested: false,
        }
    }
}

/// Internal shared state for coordinating between the WAL interface and writer task.
///
/// This structure contains the synchronization primitives needed for the producer-consumer
/// pattern between user threads (producers) and the background writer task (consumer).
struct LogInner {
    /// The current status of the WAL, protected by a read-write lock.
    ///
    /// Multiple readers can check status concurrently, but writes require exclusive access.
    status: RwLock<LogStatus>,

    /// Notification mechanism for the writer task.
    ///
    /// Notified when new data is queued, sync is requested, or shutdown is initiated.
    queue_cond: Notify,

    /// Notification mechanism for waiting operations.
    ///
    /// Notified when writes complete, syncs finish, or pruning occurs.
    write_cond: Notify,
}

impl LogInner {
    /// Creates a new inner state with the given status.
    ///
    /// Initializes the synchronization primitives for coordinating between
    /// the WAL interface and the background writer task.
    ///
    /// # Arguments
    ///
    /// * `status` - Initial status of the WAL
    fn new(status: LogStatus) -> Self {
        Self {
            status: RwLock::new(status),
            queue_cond: Default::default(),
            write_cond: Default::default(),
        }
    }
}

/// The primary write-ahead log interface for ensuring durability.
///
/// The `WriteAheadLog` provides the main API for logging write operations before they
/// are applied to the memtable. This ensures that data is not lost in the event of a
/// crash or power failure.
///
/// # Durability Model
///
/// The WAL provides configurable durability guarantees:
/// - **Async writes**: Operations are buffered and written asynchronously for performance
/// - **Explicit sync**: Call `sync()` to ensure all buffered data is persisted to disk
/// - **Ordered writes**: All operations are written in strict sequential order
///
/// # Background Writer
///
/// The WAL uses a single background task to perform all disk I/O. This design:
/// - Simplifies ordering guarantees (no need for complex locking)
/// - Allows batching of writes for better performance
/// - Enables async/await without blocking user operations
///
/// # Usage
///
/// ```rust,ignore
/// // Create a new WAL
/// let wal = WriteAheadLog::new(params).await?;
///
/// // Log operations
/// let entries = vec![LogEntry::Write(&write_op)];
/// let position = wal.store(entries.into_iter()).await?;
///
/// // Ensure durability
/// wal.sync().await?;
///
/// // After memtable flush, prune old entries
/// wal.prune_wal(position).await;
///
/// // Clean shutdown
/// wal.stop().await?;
/// ```
pub struct WriteAheadLog {
    inner: Arc<LogInner>,

    /// Allows waiting for the background write task to shut down
    finish_receiver: Mutex<Option<oneshot::Receiver<()>>>,
}

impl WriteAheadLog {
    /// Creates a new and empty write-ahead log.
    ///
    /// Initializes the WAL by creating the first WAL file and starting the background
    /// writer task. This is used when creating a new database from scratch.
    ///
    /// # Arguments
    ///
    /// * `params` - Database parameters including the database path where WAL files will be stored
    ///
    /// # Returns
    ///
    /// Returns the initialized `WriteAheadLog` instance, or an error if file creation fails.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The database directory cannot be accessed
    /// - The initial WAL file cannot be created
    pub async fn new(params: Arc<Params>) -> Result<Self, Error> {
        let status = LogStatus::new(0, 0);

        let inner = Arc::new(LogInner::new(status));

        let writer = WalWriter::new(params.db_path.clone());
        let finish_receiver = Self::start_writer(inner.clone(), writer);

        Ok(Self {
            inner,
            finish_receiver: Mutex::new(Some(finish_receiver)),
        })
    }

    /// Opens an existing WAL and replays entries for crash recovery.
    ///
    /// This method performs crash recovery by:
    /// 1. Reading all WAL files from the given start position
    /// 2. Parsing and replaying each entry into the memtable
    /// 3. For Wisckey mode, also updating the value index with deletion markers
    /// 4. Resuming normal operations from the recovered position
    ///
    /// # Arguments
    ///
    /// * `params` - Database parameters including the database path
    /// * `start_position` - Byte position to start recovery from (typically from manifest)
    /// * `memtable` - Mutable reference to the memtable to populate during recovery
    /// * `value_index` - Mutable reference to the value index (Wisckey mode)
    ///
    /// # Returns
    ///
    /// Returns a tuple containing:
    /// - The initialized `WriteAheadLog` instance ready for new operations
    /// - A `RecoveryResult` with statistics about the recovery process
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - WAL files cannot be read
    /// - WAL entries are corrupted or have invalid format
    /// - File I/O operations fail
    #[cfg(feature = "wisckey")]
    pub async fn open(
        params: Arc<Params>,
        start_position: usize,
        memtable: &mut Memtable,
        value_index: &mut ValueIndex,
    ) -> Result<(Self, RecoveryResult), Error> {
        // This reads the file(s) in the current thread
        // because we cannot send it between threads easily

        let result = match WalReader::new(params.db_path.clone(), start_position).await? {
            Some(mut reader) => {
                // WAL file exists, replay entries
                reader.run(memtable, value_index).await?
            }
            None => {
                // No WAL file exists - all data was already flushed and WAL was pruned
                // Start fresh from the current position
                RecoveryResult {
                    new_position: start_position,
                    entries_recovered: 0,
                    value_batches_to_delete: Vec::new(),
                }
            }
        };

        let status = LogStatus::new(result.new_position, start_position);
        let inner = Arc::new(LogInner::new(status));
        let writer = WalWriter::continue_from(result.new_position, params.db_path.clone());
        let finish_receiver = Self::start_writer(inner.clone(), writer);

        Ok((
            Self {
                inner,
                finish_receiver: Mutex::new(Some(finish_receiver)),
            },
            result,
        ))
    }

    /// Opens an existing WAL and replays entries for crash recovery (vanilla LSM mode).
    ///
    /// This method performs crash recovery by:
    /// 1. Reading all WAL files from the given start position
    /// 2. Parsing and replaying each entry into the memtable
    /// 3. Resuming normal operations from the recovered position
    ///
    /// This is the vanilla LSM-tree version that only handles write operations
    /// (Put and Delete), without Wisckey value separation features.
    ///
    /// # Arguments
    ///
    /// * `params` - Database parameters including the database path
    /// * `start_position` - Byte position to start recovery from (typically from manifest)
    /// * `memtable` - Mutable reference to the memtable to populate during recovery
    ///
    /// # Returns
    ///
    /// Returns a tuple containing:
    /// - The initialized `WriteAheadLog` instance ready for new operations
    /// - A `RecoveryResult` with statistics about the recovery process
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - WAL files cannot be read
    /// - WAL entries are corrupted or have invalid format
    /// - File I/O operations fail
    #[cfg(not(feature = "wisckey"))]
    pub async fn open(
        params: Arc<Params>,
        start_position: usize,
        memtable: &mut Memtable,
    ) -> Result<(Self, RecoveryResult), Error> {
        // This reads the file(s) in the current thread
        // because we cannot send stuff between threads easily

        let result = match WalReader::new(params.db_path.clone(), start_position).await? {
            Some(mut reader) => {
                // WAL file exists, replay entries
                reader.run(memtable).await?
            }
            None => {
                // No WAL file exists - all data was already flushed and WAL was pruned
                // Start fresh from the current position
                RecoveryResult {
                    new_position: start_position,
                    entries_recovered: 0,
                }
            }
        };

        let status = LogStatus::new(result.new_position, start_position);
        let inner = Arc::new(LogInner::new(status));
        let writer = WalWriter::continue_from(result.new_position, params.db_path.clone());
        let finish_receiver = Self::start_writer(inner.clone(), writer);

        Ok((
            Self {
                inner,
                finish_receiver: Mutex::new(Some(finish_receiver)),
            },
            result,
        ))
    }

    /// Spawns the background task that performs actual WAL disk writes.
    ///
    /// This creates a single background task responsible for all WAL I/O operations.
    /// Having a single writer task:
    /// - Guarantees sequential ordering of all writes
    /// - Eliminates the need for complex concurrent write synchronization
    /// - Allows efficient batching of multiple operations
    ///
    /// The task runs until a stop is requested, continuously:
    /// 1. Waiting for queued operations
    /// 2. Writing batches to disk
    /// 3. Performing fsync when requested
    /// 4. Deleting old WAL files when pruned
    ///
    /// # Arguments
    ///
    /// * `inner` - Shared state for coordination between the WAL and writer
    /// * `writer` - The writer instance that performs actual file I/O
    ///
    /// # Returns
    ///
    /// Returns a oneshot receiver that will be signaled when the writer task terminates.
    /// This is used during shutdown to ensure graceful termination.
    fn start_writer(inner: Arc<LogInner>, mut writer: WalWriter) -> oneshot::Receiver<()> {
        let (finish_sender, finish_receiver) = oneshot::channel();

        let run_writer = async move {
            let mut done = false;

            while !done {
                done = writer
                    .update_log(&inner)
                    .await
                    .expect("Write-ahead logging task failed");
            }
            let _ = finish_sender.send(());
        };

        tokio::spawn(run_writer);

        finish_receiver
    }

    /// Stores one or more operations to the write-ahead log.
    ///
    /// This is the primary method for logging write operations. Each entry is serialized
    /// according to the WAL entry format and added to the write queue. The method waits
    /// until the data has been written to the OS buffer (but not necessarily synced to disk).
    ///
    /// # Entry Serialization Format
    ///
    /// Each entry is serialized as:
    /// - Entry type (1 byte)
    /// - For Write operations:
    ///   - Operation type (1 byte): Put or Delete
    ///   - Key length (8 bytes, little-endian u64)
    ///   - Key data (variable)
    ///   - Value length (8 bytes, for Put only)
    ///   - Value data (variable, for Put only)
    /// - For Delete operations (Wisckey):
    ///   - Page ID (variable)
    ///   - Offset (2 bytes)
    ///
    /// # Arguments
    ///
    /// * `entries` - Iterator of log entries to store. Can be a single operation or a batch.
    ///
    /// # Returns
    ///
    /// Returns the absolute byte position in the WAL after all entries have been written.
    /// This position can be used to track progress and for pruning old entries.
    ///
    /// # Errors
    ///
    /// Currently does not return errors during normal operation, but may in the future
    /// if write operations fail.
    ///
    /// # Note
    ///
    /// This method only guarantees that data is written to the OS buffer. For durability
    /// guarantees that survive crashes, call `sync()` after this method.
    #[tracing::instrument(skip(self, entries))]
    pub async fn store(&self, entries: impl Iterator<Item = LogEntry<'_>>) -> Result<usize, Error> {
        let mut writes = vec![];

        for entry in entries {
            let mut data = vec![entry.get_type() as u8];

            match entry {
                LogEntry::Write(op) => {
                    let op_type = op.get_type();
                    let key = op.get_key();
                    let key_len = op.get_key_length();
                    let value_len = op.get_value_length();

                    data.extend_from_slice(op_type.as_bytes());
                    data.extend_from_slice(key_len.as_bytes());
                    data.extend_from_slice(key);

                    match op {
                        WriteOp::Put(_, value) => {
                            data.extend_from_slice(value_len.as_bytes());
                            data.extend_from_slice(value);
                        }
                        WriteOp::Delete(_) => {}
                    }

                    writes.push(data);
                }
                #[cfg(feature = "wisckey")]
                LogEntry::DeleteValue(page_id, offset) | LogEntry::DeleteBatch(page_id, offset) => {
                    data.extend_from_slice(page_id.as_bytes());
                    data.extend_from_slice(offset.as_bytes());

                    writes.push(data);
                }
            }
        }

        let end_pos = self.queue_write(writes).await;
        self.wait_for_write_position(end_pos).await;

        Ok(end_pos)
    }

    /// Queues serialized write data for the background writer.
    ///
    /// Adds the provided data buffers to the write queue and updates the queue position.
    /// Notifies the background writer task that new data is available.
    ///
    /// # Arguments
    ///
    /// * `writes` - Vector of serialized log entry data to queue
    ///
    /// # Returns
    ///
    /// The absolute byte position after all queued data has been written.
    async fn queue_write(&self, writes: Vec<Vec<u8>>) -> usize {
        let mut status = self.inner.status.write();
        let mut end_pos = status.queue_pos;

        for data in writes {
            let write_len = data.len();
            status.queue.push(data);
            status.queue_pos += write_len;
            end_pos += write_len;
        }

        self.inner.queue_cond.notify_waiters();
        end_pos
    }

    /// Waits until data has been written to the WAL up to the specified position.
    ///
    /// This ensures that queued data has been written to the OS buffer, though
    /// not necessarily synced to disk.
    ///
    /// # Arguments
    ///
    /// * `position` - The byte position to wait for
    async fn wait_for_write_position(&self, position: usize) {
        self.wait_for_condition(position, |status: &LogStatus, position: usize| -> bool {
            status.write_pos >= position
        })
        .await
    }

    /// Waits until data has been synced to disk up to the specified position.
    ///
    /// This ensures that data has been persisted via fsync and will survive
    /// a crash or power failure.
    ///
    /// # Arguments
    ///
    /// * `position` - The byte position to wait for
    async fn wait_for_sync_pos(&self, position: usize) {
        self.wait_for_condition(position, |status: &LogStatus, position: usize| -> bool {
            status.sync_pos > position
        })
        .await
    }

    /// Waits until the WAL has been pruned up to the specified position.
    ///
    /// This ensures that old WAL files have been deleted from disk.
    ///
    /// # Arguments
    ///
    /// * `position` - The byte position to wait for
    async fn wait_for_prune_pos(&self, position: usize) {
        self.wait_for_condition(position, |status: &LogStatus, position: usize| -> bool {
            status.prune_pos >= position
        })
        .await
    }

    /// Generic wait helper that blocks until a predicate is satisfied.
    ///
    /// This is the core waiting mechanism used by all wait_for_* methods.
    /// It efficiently waits using Tokio's Notify primitive while checking
    /// the condition under a read lock.
    ///
    /// # Arguments
    ///
    /// * `position` - The position value to pass to the predicate
    /// * `predicate` - Function that returns true when the wait condition is satisfied
    ///
    /// # Implementation Notes
    ///
    /// Uses a manual pinning workaround for Rust issue #63768 to ensure
    /// the notification future is properly registered before releasing the lock.
    async fn wait_for_condition<F: Fn(&LogStatus, usize) -> bool>(
        &self,
        position: usize,
        predicate: F,
    ) {
        loop {
            // This works around the following bug:
            // https://github.com/rust-lang/rust/issues/63768
            let fut = self.inner.write_cond.notified();
            tokio::pin!(fut);

            {
                let status = self.inner.status.read();
                if predicate(&status, position) {
                    return;
                }
                // if status.prune_pos >= position {
                //     return;
                // }

                // Wait for next write
                fut.as_mut().enable();
            }

            fut.await;
        }
    }

    /// Gracefully shuts down the write-ahead log.
    ///
    /// This method ensures that all pending writes are completed before the WAL is closed.
    /// It performs the following steps:
    /// 1. Sets the stop flag to signal the writer task
    /// 2. Notifies the writer task to wake up
    /// 3. Waits for the writer task to complete all pending operations and terminate
    ///
    /// # Usage
    ///
    /// This should only be called during database shutdown and must be called exactly once.
    /// Calling it multiple times will panic.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` when the WAL has been successfully shut down.
    ///
    /// # Errors
    ///
    /// Currently does not return errors, but the signature allows for future error cases.
    ///
    /// # Panics
    ///
    /// Panics if called more than once on the same WAL instance.
    pub async fn stop(&self) -> Result<(), Error> {
        log::trace!("Shutting down write-ahead log. Waiting for writer to terminate.");

        self.inner.status.write().stop_requested = true;
        self.inner.queue_cond.notify_waiters();

        self.finish_receiver
            .lock()
            .take()
            .expect("Already stopped?")
            .await
            .unwrap();

        log::debug!("Write-ahead log shut down");
        Ok(())
    }

    /// Forces all buffered WAL data to be persisted to disk.
    ///
    /// This method calls `fsync` on the underlying WAL file, ensuring that all data
    /// written up to this point is guaranteed to survive a system crash or power failure.
    ///
    /// # Durability Guarantee
    ///
    /// After this method returns successfully:
    /// - All writes queued before the sync are guaranteed to be on disk
    /// - The data will be available during recovery after a crash
    /// - The sync position is updated to reflect the persisted data
    ///
    /// # Performance Considerations
    ///
    /// Calling `fsync` is expensive as it requires waiting for the OS and disk hardware
    /// to complete the write. Consider:
    /// - Batching multiple writes before syncing
    /// - Only syncing when durability is critical
    /// - Using async writes for better throughput when some data loss is acceptable
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` when the sync operation completes successfully.
    ///
    /// # Errors
    ///
    /// Currently does not return errors, but the signature allows for future error cases
    /// such as I/O failures during fsync.
    #[tracing::instrument(skip(self))]
    pub async fn sync(&self) -> Result<(), Error> {
        let last_pos = {
            let mut status = self.inner.status.write();

            // Nothing to sync?
            if status.sync_pos == status.write_pos {
                return Ok(());
            }

            assert!(status.sync_pos < status.write_pos);

            status.sync_requested = true;
            self.inner.queue_cond.notify_waiters();

            status.sync_pos
        };

        self.wait_for_sync_pos(last_pos).await;

        Ok(())
    }

    /// Marks old WAL entries for deletion after memtable flush.
    ///
    /// Once a memtable is successfully flushed to an SSTable, the corresponding WAL entries
    /// are no longer needed for recovery and can be safely deleted to free disk space.
    ///
    /// # Garbage Collection Process
    ///
    /// 1. The caller specifies a position up to which WAL data is no longer needed
    /// 2. This method marks that position for pruning
    /// 3. The background writer task deletes old WAL files (pages) that are fully before this position
    /// 4. The method waits until the deletion is complete
    ///
    /// # Arguments
    ///
    /// * `prune_pos` - Absolute byte position up to which the WAL can be pruned.
    ///   This must be greater than any previous prune position.
    ///
    /// # Panics
    ///
    /// Panics if `prune_pos` is less than or equal to the current prune position.
    /// Pruning can only move forward, never backward.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // After flushing memtable at position 8192
    /// wal.prune_wal(8192).await;
    /// // WAL files containing only data before position 8192 are now deleted
    /// ```
    #[tracing::instrument(skip(self))]
    pub async fn prune_wal(&self, prune_pos: usize) {
        {
            let mut status = self.inner.status.write();

            if prune_pos <= status.can_prune_pos {
                panic!(
                    "Offset can only be increased! Requested {prune_pos}, but was {}",
                    status.can_prune_pos
                );
            }

            status.can_prune_pos = prune_pos;
            self.inner.queue_cond.notify_waiters();
        }

        self.wait_for_prune_pos(prune_pos).await;
    }
}
