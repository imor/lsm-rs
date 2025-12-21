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

/// In the vanilla configuration, the log only stores
/// write operations.
/// For Wisckey, it also stores changes to the value_index
/// to reduce write amplification.
pub enum LogEntry<'a> {
    Write(&'a WriteOp),
    #[cfg(feature = "wisckey")]
    DeleteBatch(ValueIndexPageId, u16),
    #[cfg(feature = "wisckey")]
    DeleteValue(ValueIndexPageId, u16),
}

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

/// The log is split into individual files (pages) that can be
/// garbage collected once the logged data is not needed anymore
const PAGE_SIZE: usize = 4 * 1024;

/// The state of the log (internal DS shared between writer
/// task and WAL object)
///
/// Invariants:
///  - sync_pos <= write_pos <= queue_pos
///  - prune_pos <= can_prune_pos
///
struct LogStatus {
    /// Absolute count of queued write operations
    queue_pos: usize,

    /// Absolute count of fulfilled write operations
    write_pos: usize,

    /// At what write count did we last invoke fsync?
    sync_pos: usize,

    /// Pending data to be written
    queue: Vec<Vec<u8>>,

    /// Where the current can prune position is
    /// (anything below this is not needed anymore)
    can_prune_pos: usize,

    /// How much has actually been cleaned up
    prune_pos: usize,

    /// Was a sync requested?
    sync_requested: bool,

    /// Indicates the log should shut down
    stop_requested: bool,
}

impl LogStatus {
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

struct LogInner {
    status: RwLock<LogStatus>,
    queue_cond: Notify,
    write_cond: Notify,
}

impl LogInner {
    fn new(status: LogStatus) -> Self {
        Self {
            status: RwLock::new(status),
            queue_cond: Default::default(),
            write_cond: Default::default(),
        }
    }
}

/// The write-ahead log keeps track of the most recent changes
/// It can be used to recover from crashes
pub struct WriteAheadLog {
    inner: Arc<LogInner>,

    /// Allows waiting for the background write task to shut down
    finish_receiver: Mutex<Option<oneshot::Receiver<()>>>,
}

impl WriteAheadLog {
    /// Creates a new and empty write-ahead log
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

    /// Open an existing log and insert entries into memtable
    ///
    /// This is similar to `new` but fetches state from disk first.
    #[cfg(feature = "wisckey")]
    pub async fn open(
        params: Arc<Params>,
        start_position: usize,
        memtable: &mut Memtable,
        value_index: &mut ValueIndex,
    ) -> Result<(Self, RecoveryResult), Error> {
        // This reads the file(s) in the current thread
        // because we cannot send it between threads easily

        let mut reader = WalReader::new(params.db_path.clone(), start_position).await?;

        let result = reader.run(memtable, value_index).await?;

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

    #[cfg(not(feature = "wisckey"))]
    pub async fn open(
        params: Arc<Params>,
        start_position: usize,
        memtable: &mut Memtable,
    ) -> Result<(Self, RecoveryResult), Error> {
        // This reads the file(s) in the current thread
        // because we cannot send stuff between threads easily

        let mut reader = WalReader::new(params.db_path.clone(), start_position).await?;

        let result = reader.run(memtable).await?;

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

    /// Spawns the background task that will actually write
    /// to the WAL.
    ///
    /// There is exactly one task that writes to the log
    /// so that we have to worry about ordering less.
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

    /// Stores an operation and returns the new position in
    /// the logfile
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

    async fn wait_for_write_position(&self, position: usize) {
        self.wait_for_condition(position, |status: &LogStatus, position: usize| -> bool {
            status.write_pos >= position
        })
        .await
    }

    async fn wait_for_sync_pos(&self, position: usize) {
        self.wait_for_condition(position, |status: &LogStatus, position: usize| -> bool {
            status.sync_pos > position
        })
        .await
    }

    async fn wait_for_prune_pos(&self, position: usize) {
        self.wait_for_condition(position, |status: &LogStatus, position: usize| -> bool {
            status.prune_pos >= position
        })
        .await
    }

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

    /// Gracefully stop the write-ahead log
    ///
    /// This is intended to only be called during shutdown
    /// and shall be called exactly once.
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

    /// Once the memtable has been flushed we can prune old log entries
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
