//! WAL Reader for Crash Recovery
//!
//! This module provides the [`WalReader`] type which is responsible for reading
//! and replaying write-ahead log entries during database recovery after a crash
//! or restart.
//!
//! # Recovery Process
//!
//! When the database starts up, the reader:
//! 1. Opens WAL files starting from the position recorded in the manifest
//! 2. Sequentially reads and parses each log entry
//! 3. Replays write operations into the memtable
//! 4. For Wisckey mode, also updates the value index with deletion markers
//! 5. Continues until all WAL files have been processed or an incomplete entry is found
//!
//! # Entry Format Parsing
//!
//! The reader understands the WAL entry format:
//! - Reads the entry type byte first to determine the operation
//! - Parses operation-specific data based on the entry type
//! - Handles variable-length keys and values
//! - Correctly handles entries that span multiple WAL pages
//!
//! # Error Handling
//!
//! The reader is designed to handle:
//! - Incomplete entries at the end of the last WAL file (normal during recovery)
//! - Missing WAL files (indicates recovery is complete)
//! - Corrupted entry type bytes (panics, as this indicates data corruption)
//!
//! # Thread Safety
//!
//! The reader is not thread-safe and should only be used during the single-threaded
//! recovery phase before normal database operations begin.

use std::path::PathBuf;

use zerocopy::FromBytes;

#[cfg(feature = "wisckey")]
use crate::values::{ValueBatchId, ValueIndex};

use crate::memtable::Memtable;
use crate::{Error, disk};

use super::{LogEntryType, PAGE_SIZE, WalWriter, WriteOp};

/// Reads and replays write-ahead log entries during crash recovery.
///
/// The `WalReader` sequentially processes WAL files to restore database state
/// after a crash or restart. It maintains the current position in the log and
/// handles reading data that may span multiple WAL page files.
///
/// # Position Tracking
///
/// The reader tracks an absolute byte position across all WAL files:
/// - Position 0-4095: First file (00000001.wal)
/// - Position 4096-8191: Second file (00000002.wal)
/// - And so on...
///
/// # Page Management
///
/// The reader loads one WAL page (file) at a time into memory. When reading crosses
/// a page boundary, it automatically loads the next page. This keeps memory usage
/// bounded regardless of total WAL size.
pub struct WalReader {
    position: usize,
    current_page: Vec<u8>,
    db_path: PathBuf,
}

/// Statistics and results from the WAL recovery process.
///
/// This structure contains information about what was recovered from the WAL,
/// which is useful for logging, debugging, and determining the starting state
/// after recovery completes.
#[derive(Default)]
pub struct RecoveryResult {
    /// The absolute byte position where recovery ended.
    ///
    /// This is the position immediately after the last successfully parsed WAL entry.
    /// New write operations will continue from this position.
    pub new_position: usize,

    /// The total number of log entries successfully recovered and replayed.
    ///
    /// This count includes all entry types (writes, deletes, value deletions, etc.).
    pub entries_recovered: usize,

    /// List of value batches marked for deletion during recovery (Wisckey only).
    ///
    /// These batches were marked for deletion in the WAL but may not have been
    /// physically deleted yet. They will be cleaned up during garbage collection.
    #[cfg(feature = "wisckey")]
    pub value_batches_to_delete: Vec<ValueBatchId>,
}

impl WalReader {
    /// Creates a new WAL reader starting from the specified position.
    ///
    /// Opens the WAL file containing the start position and prepares for reading.
    /// The start position is typically obtained from the database manifest.
    ///
    /// # Arguments
    ///
    /// * `db_path` - Path to the database directory containing WAL files
    /// * `start_position` - Absolute byte position to start reading from
    ///
    /// # Returns
    ///
    /// Returns the initialized reader, or an error if the WAL file cannot be opened.
    ///
    /// # Errors
    ///
    /// Returns `Ok(None)` if the WAL file doesn't exist, indicating that all WAL entries
    /// have been flushed to disk and the WAL was pruned. Returns an error only if
    /// the file exists but cannot be read.
    ///
    /// # Returns
    ///
    /// - `Ok(Some(reader))`: WAL file found, ready for recovery
    /// - `Ok(None)`: No WAL file exists (all data was flushed and pruned)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The WAL file exists but cannot be read
    /// - File I/O operations fail unexpectedly
    pub async fn new(db_path: PathBuf, start_position: usize) -> Result<Option<Self>, Error> {
        let position = start_position;
        let file_num = position / PAGE_SIZE;

        let file_path = WalWriter::get_file_path(&db_path, file_num);
        log::trace!("Opening next log file at {file_path:?}");

        let current_page = match disk::read_uncompressed(&file_path, 0).await {
            Ok(page) => page,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                // WAL file doesn't exist - this is normal when all data has been
                // flushed to sorted tables and the WAL was pruned
                log::debug!("No WAL file found at {file_path:?}, assuming all data was flushed");
                return Ok(None);
            }
            Err(err) => {
                return Err(Error::from_io_error("Failed to open WAL file", err));
            }
        };

        Ok(Some(Self {
            current_page,
            position,
            db_path,
        }))
    }

    /// Runs the recovery process, replaying all WAL entries into the memtable and value index.
    ///
    /// This is the main recovery method for Wisckey mode. It reads each entry from the WAL
    /// and applies it to the appropriate data structure:
    /// - Write operations are inserted into the memtable
    /// - Value deletions update the value index
    /// - Batch deletions mark entire value batches as deleted
    ///
    /// The method continues until it encounters:
    /// - An incomplete entry (normal at the end of the last WAL file)
    /// - A missing WAL file (indicates all entries have been recovered)
    ///
    /// # Arguments
    ///
    /// * `memtable` - Mutable reference to the memtable to populate with recovered writes
    /// * `value_index` - Mutable reference to the value index to update with deletion markers
    ///
    /// # Returns
    ///
    /// Returns a `RecoveryResult` containing:
    /// - The position where recovery ended (for continuing normal operations)
    /// - The number of entries recovered
    /// - List of value batches to delete
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A WAL entry has an invalid or corrupted format
    /// - File I/O operations fail unexpectedly
    ///
    /// # Panics
    ///
    /// Panics if an entry type byte has an unexpected value, indicating WAL corruption.
    #[cfg(feature = "wisckey")]
    pub async fn run(
        &mut self,
        memtable: &mut Memtable,
        value_index: &mut ValueIndex,
    ) -> Result<RecoveryResult, Error> {
        let mut result = RecoveryResult::default();

        // Re-insert ops into memtable
        loop {
            let mut log_type = [0u8; 1];
            let success = self.read_from_log(&mut log_type[..], true).await?;

            if !success {
                break;
            }

            if log_type[0] == LogEntryType::Write as u8 {
                self.parse_write_entry(memtable).await?
            } else if log_type[0] == LogEntryType::DeleteValue as u8 {
                self.parse_value_deletion_entry(value_index).await?
            } else if log_type[0] == LogEntryType::DeleteBatch as u8 {
                self.parse_batch_deletion_entry(value_index).await?
            } else {
                panic!("Unexpected log entry type! {}", log_type[0]);
            }

            result.entries_recovered += 1;
        }

        log::debug!(
            "Found {} entries in write-ahead log",
            result.entries_recovered
        );
        result.new_position = self.position;
        Ok(result)
    }

    #[cfg(not(feature = "wisckey"))]
    pub async fn run(&mut self, memtable: &mut Memtable) -> Result<RecoveryResult, Error> {
        let mut result = RecoveryResult::default();

        // Re-insert ops into memtable
        loop {
            let mut log_type = [0u8; 1];
            let success = self.read_from_log(&mut log_type[..], true).await?;

            if !success {
                break;
            }

            if log_type[0] == LogEntryType::Write as u8 {
                self.parse_write_entry(memtable).await?
            } else {
                panic!("Unexpected log entry type!");
            }

            result.entries_recovered += 1;
        }

        log::debug!(
            "Found {} entries in write-ahead log",
            result.entries_recovered
        );
        result.new_position = self.position;
        Ok(result)
    }

    /// Parses and applies a write operation entry to the memtable.
    ///
    /// Reads a complete write entry (Put or Delete) from the current position and
    /// applies it to the memtable, restoring the operation as if it had just been
    /// executed.
    ///
    /// # Entry Format
    ///
    /// - Operation type (1 byte): PUT_OP or DELETE_OP
    /// - Key length (8 bytes, little-endian u64)
    /// - Key data (variable length)
    /// - For PUT_OP only:
    ///   - Value length (8 bytes)
    ///   - Value data (variable length)
    ///
    /// # Arguments
    ///
    /// * `memtable` - The memtable to insert the recovered operation into
    ///
    /// # Errors
    ///
    /// Returns an error if the entry cannot be read completely or has invalid format.
    ///
    /// # Panics
    ///
    /// Panics if the operation type is neither PUT_OP nor DELETE_OP.
    async fn parse_write_entry(&mut self, memtable: &mut Memtable) -> Result<(), Error> {
        let op_type: u8 = self.read_value().await?;
        let key_len: u64 = self.read_value().await?;

        let mut key = vec![0; key_len as usize];
        self.read_from_log(&mut key, false).await?;

        if op_type == WriteOp::PUT_OP {
            let val_len: u64 = self.read_value().await?;
            let mut value = vec![0; val_len as usize];
            self.read_from_log(&mut value, false).await?;
            memtable.put(key, value);
        } else if op_type == WriteOp::DELETE_OP {
            memtable.delete(key);
        } else {
            panic!("Unexpected op type!");
        }

        Ok(())
    }

    /// Reads a typed value from the current WAL position.
    ///
    /// This is a generic helper method that reads a fixed-size value of type `T`
    /// from the WAL and deserializes it using the `FromBytes` trait. The position
    /// is automatically advanced by the size of `T`.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The type to read, must implement `FromBytes`. Common types include
    ///   `u8`, `u16`, `u64` for reading fixed-size integers.
    ///
    /// # Returns
    ///
    /// Returns the deserialized value of type `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Not enough data remains in the WAL
    /// - The next WAL page cannot be opened
    ///
    /// # Note
    ///
    /// This method may open the next WAL page file if reading crosses a page boundary.
    async fn read_value<T: Sized + FromBytes>(&mut self) -> Result<T, Error> {
        let mut data = vec![0u8; std::mem::size_of::<T>()];
        self.read_from_log(&mut data, false).await?;
        Ok(T::read_from_bytes(&data).unwrap())
    }

    /// Parses and applies a value deletion entry to the value index.
    ///
    /// Reads a value deletion record from the WAL and marks the corresponding
    /// value as deleted in the value index. This is used during recovery to
    /// restore the deletion state.
    ///
    /// # Entry Format
    ///
    /// - Page ID (variable size integer)
    /// - Offset within page (2 bytes, u16)
    ///
    /// # Arguments
    ///
    /// * `value_index` - The value index to update with the deletion marker
    ///
    /// # Errors
    ///
    /// Returns an error if the entry cannot be read completely.
    #[cfg(feature = "wisckey")]
    async fn parse_value_deletion_entry(
        &mut self,
        value_index: &mut ValueIndex,
    ) -> Result<(), Error> {
        let page_id = self.read_value().await?;
        let offset = self.read_value().await?;
        value_index.mark_value_as_deleted_at(page_id, offset).await;

        Ok(())
    }

    /// Parses and applies a batch deletion entry to the value index.
    ///
    /// Reads a batch deletion record from the WAL and marks the corresponding
    /// value batch as deleted in the value index. This is used during recovery
    /// to restore the deletion state of entire batches.
    ///
    /// # Entry Format
    ///
    /// - Page ID (variable size integer)
    /// - Batch index within page (2 bytes, u16)
    ///
    /// # Arguments
    ///
    /// * `value_index` - The value index to update with the batch deletion marker
    ///
    /// # Errors
    ///
    /// Returns an error if the entry cannot be read completely or the index
    /// update fails.
    #[cfg(feature = "wisckey")]
    async fn parse_batch_deletion_entry(
        &mut self,
        value_index: &mut ValueIndex,
    ) -> Result<(), Error> {
        let page_id = self.read_value().await?;
        let offset = self.read_value().await?;
        value_index
            .mark_batch_as_deleted_at(page_id, offset)
            .await?;
        Ok(())
    }

    /// Reads raw bytes from the WAL into the provided buffer.
    ///
    /// This is the low-level method for reading data from WAL files. It handles:
    /// - Reading data that may span multiple WAL page files
    /// - Automatically loading the next page when needed
    /// - Detecting the end of the WAL during recovery
    ///
    /// # Arguments
    ///
    /// * `out` - Buffer to read data into. Must be non-empty.
    /// * `maybe` - If true, missing next page or incomplete data returns `Ok(false)`
    ///   instead of an error. Used when we're not sure if more data exists.
    ///
    /// # Returns
    ///
    /// - `Ok(true)`: Successfully read all requested bytes
    /// - `Ok(false)`: End of WAL reached (only when `maybe` is true)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File I/O operations fail (unless `maybe` is true and the error is NotFound)
    /// - The buffer cannot be filled and `maybe` is false
    ///
    /// # Implementation Notes
    ///
    /// The method reads data in chunks, handling page boundaries:
    /// 1. Read as much as possible from the current page
    /// 2. If more data is needed, load the next page
    /// 3. If the current page is not full and not on a boundary, assume we've reached the end
    /// 4. Repeat until the buffer is full or the end is reached
    async fn read_from_log(&mut self, out: &mut [u8], maybe: bool) -> Result<bool, Error> {
        let buffer_len = out.len();
        let mut buffer_pos = 0;
        assert!(buffer_len > 0);

        while buffer_pos < buffer_len {
            let offset = self.position % PAGE_SIZE;
            let file_remaining = self
                .current_page
                .len()
                .checked_sub(offset)
                .expect("Invalid offset. Page too small?");
            let buffer_remaining = buffer_len - buffer_pos;

            let len = buffer_remaining.min(file_remaining);

            if len > 0 {
                out[buffer_pos..buffer_pos + len]
                    .copy_from_slice(&self.current_page[offset..offset + len]);
                buffer_pos += len;
                self.position += len;
            } else if !self.position.is_multiple_of(PAGE_SIZE) {
                log::trace!(
                    "WAL reader is done. Current file was not full; assuming it is the most recent."
                );
                assert!(self.current_page.len() < PAGE_SIZE);
                return Ok(false);
            }

            // Move to next file?
            if self.position.is_multiple_of(PAGE_SIZE) {
                let file_num = self.position / PAGE_SIZE;
                let file_path = WalWriter::get_file_path(&self.db_path, file_num);
                log::trace!("Opening next log file at {file_path:?}");

                self.current_page = match disk::read_uncompressed(&file_path, 0).await {
                    Ok(data) => data,
                    Err(err) => {
                        if maybe && err.kind() == std::io::ErrorKind::NotFound {
                            // At last file but it is still exactly
                            // one page
                            log::trace!("WAL reader is done. No next log file found");
                            return Ok(false);
                        } else {
                            return Err(Error::from_io_error("Failed to open WAL file", err));
                        }
                    }
                }
            }
        }

        Ok(true)
    }
}
