//! WAL Writer Background Task
//!
//! This module implements the background task responsible for writing all WAL data to disk.
//! The writer operates as a single-threaded consumer in a producer-consumer pattern, where
//! user threads are producers that queue write operations.
//!
//! # Design Rationale
//!
//! Using a single background writer task provides several benefits:
//! - **Sequential ordering**: All writes happen in strict order without complex locking
//! - **Batching**: Multiple queued operations can be written together for efficiency
//! - **Non-blocking**: User threads can continue working while I/O happens asynchronously
//! - **Simple error handling**: I/O errors are handled in one place
//!
//! # Write Process
//!
//! The writer task continuously:
//! 1. Waits for queued operations using a condition variable
//! 2. Drains the queue and writes all data to the current WAL file
//! 3. Creates new WAL files when a page fills up (every 4KB)
//! 4. Performs `fsync` when explicitly requested for durability
//! 5. Deletes old WAL files when they've been pruned
//!
//! # File Management
//!
//! The writer maintains:
//! - An open file handle to the current WAL page
//! - The current absolute position across all WAL files
//! - The database path for creating/deleting WAL files
//!
//! When a page fills up (reaches 4KB), the writer:
//! - Closes the current file (implicitly via drop)
//! - Creates a new file with the next sequential number
//! - Continues writing to the new file
//!
//! # Synchronization
//!
//! The writer coordinates with user threads through shared state:
//! - Reads queued data from `LogStatus`
//! - Updates write/sync/prune positions after completing operations
//! - Notifies waiting threads via condition variables

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::wal::{LogInner, PAGE_SIZE};
use crate::{Error, disk};

/// The background task responsible for writing WAL data to disk.
///
/// `WalWriter` handles all physical I/O operations for the write-ahead log.
/// It runs as a single background task to ensure sequential ordering of writes
/// and efficient batching of operations.
///
/// # File Descriptor Management
///
/// The writer maintains an open file handle to the current WAL page. When a page
/// fills up, it creates a new file and updates the handle. This avoids the overhead
/// of opening/closing files for every write.
///
/// # Position Tracking
///
/// The `position` field tracks the absolute byte position across all WAL files:
/// - Position 0: Start of first file (00000001.wal)
/// - Position 4096: Start of second file (00000002.wal)
/// - Position N: (N / 4096) files in, (N % 4096) bytes into current file
pub struct WalWriter {
    wal_file: File,
    position: usize,
    db_path: PathBuf,
}

impl WalWriter {
    /// Creates a new WAL writer starting at position 0.
    ///
    /// This is used when creating a new database. It creates the first WAL file
    /// (00000001.wal) and prepares for writing.
    ///
    /// # Arguments
    ///
    /// * `db_path` - Path to the database directory where WAL files will be stored
    ///
    /// # Returns
    ///
    /// Returns the initialized writer.
    ///
    /// # Panics
    ///
    /// Panics if the first WAL file cannot be created, as this indicates a fundamental
    /// I/O problem that prevents the database from functioning.
    pub fn new(db_path: PathBuf) -> Self {
        let wal_file = Self::create_file(&db_path, 0).unwrap_or_else(|err| {
            panic!("Failed to create WAL file in directory {db_path:?}: {err}",)
        });

        Self {
            wal_file,
            db_path,
            position: 0,
        }
    }

    /// Creates a writer that continues from a specific position after recovery.
    ///
    /// This is used when opening an existing database. The writer picks up where
    /// recovery left off, either:
    /// - Opening an existing incomplete WAL page, or
    /// - Creating a new WAL page if the position is at a page boundary
    ///
    /// # Arguments
    ///
    /// * `position` - Absolute byte position to start writing from (typically from recovery)
    /// * `db_path` - Path to the database directory containing WAL files
    ///
    /// # Returns
    ///
    /// Returns the initialized writer ready to append new operations.
    ///
    /// # Panics
    ///
    /// Panics if the WAL file at the specified position cannot be opened or created.
    pub fn continue_from(position: usize, db_path: PathBuf) -> Self {
        let file_num = position / PAGE_SIZE;

        let wal_file = if position.is_multiple_of(PAGE_SIZE) {
            // At the beginning of a new file
            Self::create_file(&db_path, file_num).unwrap_or_else(|err| {
                panic!("Failed to create WAL file in directory {db_path:?}: {err}",)
            })
        } else {
            Self::open_file(&db_path, file_num).unwrap_or_else(|err| {
                panic!("Failed to open WAL file in directory {db_path:?}: {err}",)
            })
        };

        Self {
            wal_file,
            db_path,
            position,
        }
    }

    /// Opens an existing WAL file for appending.
    ///
    /// Used when resuming writing to a partially-filled WAL page after recovery.
    /// The file is opened in read-write mode without truncation to preserve existing data.
    ///
    /// # Arguments
    ///
    /// * `db_path` - Path to the database directory
    /// * `file_num` - The WAL file number (0-based index)
    ///
    /// # Returns
    ///
    /// Returns an open file handle positioned at the end of the file.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file does not exist
    /// - File permissions prevent opening
    /// - Other I/O errors occur
    pub fn open_file(db_path: &Path, file_num: usize) -> Result<File, std::io::Error> {
        let file_path = Self::get_file_path(db_path, file_num);
        log::trace!("Opening file at {file_path:?}");

        let wal_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(false)
            .truncate(false)
            .open(file_path)?;

        Ok(wal_file)
    }

    /// Main loop iteration: processes queued operations and updates the WAL.
    ///
    /// This is the core method of the writer task, called repeatedly in a loop.
    /// Each iteration:
    /// 1. Waits for work to be available (queued writes, sync request, or prune request)
    /// 2. Drains the write queue and writes all data to disk
    /// 3. Performs fsync if requested
    /// 4. Deletes old WAL files if pruning was requested
    /// 5. Updates positions and notifies waiting threads
    ///
    /// # Arguments
    ///
    /// * `inner` - Shared state for coordination with the WAL interface
    ///
    /// # Returns
    ///
    /// - `Ok(true)`: Stop was requested, the writer task should terminate
    /// - `Ok(false)`: Continue processing, more work may arrive
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Writing to the WAL file fails
    /// - Deleting old WAL files fails
    ///
    /// # Synchronization
    ///
    /// The method holds the status lock only briefly to extract work, then releases it
    /// while performing I/O. This allows user threads to continue queuing operations
    /// while the writer is writing.
    pub async fn update_log(&mut self, inner: &LogInner) -> Result<bool, Error> {
        let (to_write, sync_requested, sync_pos, new_offset, stop_requested) = loop {
            // This works around the following bug:
            // https://github.com/rust-lang/rust/issues/63768
            let fut = inner.queue_cond.notified();
            tokio::pin!(fut);

            {
                let mut status = inner.status.write();
                let to_write = std::mem::take(&mut status.queue);
                let sync_requested = status.flush_requested;
                let sync_pos = status.flush_pos;
                let stop_requested = status.stop_requested;

                let new_offset = if status.can_prune_pos > status.prune_pos {
                    Some((status.can_prune_pos, status.prune_pos))
                } else {
                    assert_eq!(status.can_prune_pos, status.prune_pos);
                    None
                };

                // Check whether there is something to do
                if !to_write.is_empty() || new_offset.is_some() || sync_requested || stop_requested
                {
                    assert_eq!(self.position, status.write_pos);

                    status.flush_requested = false;
                    break (
                        to_write,
                        sync_requested,
                        sync_pos,
                        new_offset,
                        stop_requested,
                    );
                }

                // wait for change to queue and retry
                assert_eq!(status.write_pos, status.queue_pos);
                fut.as_mut().enable();
            }

            fut.await;
        };

        // Don't hold lock while write
        for buf in to_write {
            self.write_all(buf)
                .await
                .map_err(|err| Error::from_io_error("Failed to write to wal", err))?;
        }

        // Only sync if necessary
        // We do not need to hold the lock while syncing
        // because there is only one write-ahead writer
        if sync_requested && sync_pos < self.position {
            self.sync().await;
            inner.status.write().flush_pos = self.position;
        }

        if let Some((new_offset, old_offset)) = new_offset {
            self.set_offset(old_offset, new_offset).await?;
        }

        // Notify about finished write(s)
        {
            let mut status = inner.status.write();
            assert!(status.write_pos <= self.position);
            status.write_pos = self.position;

            if let Some((new_offset, _)) = new_offset {
                status.prune_pos = new_offset;
            }

            inner.write_cond.notify_waiters();
        }

        if stop_requested {
            log::debug!("WAL writer finished");
        }

        Ok(stop_requested)
    }

    /// Deletes old WAL files that have been pruned.
    ///
    /// When the WAL is pruned (after memtable flush), this method removes the corresponding
    /// WAL page files from disk to reclaim space.
    ///
    /// # Arguments
    ///
    /// * `old_offset` - The previous prune position
    /// * `new_offset` - The new prune position
    ///
    /// # File Deletion
    ///
    /// All complete WAL pages between `old_offset` and `new_offset` are deleted.
    /// For example, if old_offset=1000 and new_offset=9000:
    /// - File 0 (positions 0-4095): Deleted
    /// - File 1 (positions 4096-8191): Deleted  
    /// - File 2 (positions 8192-12287): Kept (new_offset is within this file)
    ///
    /// # Errors
    ///
    /// Returns an error if file deletion fails.
    async fn set_offset(&mut self, old_offset: usize, new_offset: usize) -> Result<(), Error> {
        let old_file_num = old_offset / PAGE_SIZE;
        let new_file_num = new_offset / PAGE_SIZE;

        for file_num in old_file_num..new_file_num {
            let file_path = Self::get_file_path(&self.db_path, file_num);
            log::trace!("Removing file {file_path:?}");

            disk::remove_file(&file_path).await.map_err(|err| {
                Error::from_io_error(format!("Failed to remove log file {file_path:?}"), err)
            })?;
        }

        Ok(())
    }

    /// Syncs the current WAL file to disk using fsync.
    ///
    /// This ensures that all data written to the file is persisted to physical storage
    /// and will survive a power failure or system crash.
    ///
    /// # Panics
    ///
    /// Panics if the fsync system call fails. This is treated as a fatal error because
    /// it indicates the durability guarantee cannot be met.
    async fn sync(&mut self) {
        self.wal_file.sync_data().expect("Data sync failed");
    }

    /// Writes data to the WAL, handling page boundaries automatically.
    ///
    /// This method writes the entire data buffer to the WAL, creating new page files
    /// as needed when writes cross page boundaries. It ensures that:
    /// - All data is written sequentially
    /// - New files are created exactly at page boundaries
    /// - The position is correctly maintained across files
    ///
    /// # Arguments
    ///
    /// * `data` - The data to write. Can be any length, even spanning multiple pages.
    ///
    /// # Page Boundary Handling
    ///
    /// When a write would exceed the current page size (4KB):
    /// 1. Write as much as fits in the current page
    /// 2. Create a new WAL file for the next page
    /// 3. Continue writing to the new file
    /// 4. Repeat until all data is written
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Writing to the file fails
    /// - Creating a new WAL file fails
    ///
    /// # Panics
    ///
    /// Panics if writing fails, as this indicates a fundamental I/O problem.
    async fn write_all(&mut self, data: Vec<u8>) -> Result<(), std::io::Error> {
        let mut buf_pos = 0;
        while buf_pos < data.len() {
            let mut file_offset = self.position % PAGE_SIZE;

            // Figure out how much we can fit into the current file
            assert!(file_offset < PAGE_SIZE);

            let page_remaining = PAGE_SIZE - file_offset;
            let buffer_remaining = data.len() - buf_pos;
            let write_len = (buffer_remaining).min(page_remaining);

            assert!(write_len > 0);

            let to_write = &data[buf_pos..buf_pos + write_len];
            self.wal_file
                .write_all(to_write)
                .expect("Failed to write log file");

            buf_pos += write_len;
            self.position += write_len;
            file_offset += write_len;

            assert!(file_offset <= PAGE_SIZE);

            // Create a new file?
            if file_offset == PAGE_SIZE {
                let file_num = self.position / PAGE_SIZE;
                self.wal_file = Self::create_file(&self.db_path, file_num)?;
            }
        }

        Ok(())
    }

    /// Creates a new WAL page file.
    ///
    /// Creates a new file in the database directory with the naming convention
    /// `NNNNNNNN.wal` where N is the zero-padded file number (1-based).
    ///
    /// # Arguments
    ///
    /// * `db_path` - Path to the database directory
    /// * `file_num` - The WAL file number (0-based index, but file name is 1-based)
    ///
    /// # Returns
    ///
    /// Returns an open file handle ready for writing.
    ///
    /// # Errors
    ///
    /// Returns an error if file creation fails due to permissions, disk space, or other I/O issues.
    pub fn create_file(db_path: &Path, file_num: usize) -> Result<File, std::io::Error> {
        let file_path = Self::get_file_path(db_path, file_num);
        log::trace!("Creating new wal file at {file_path:?}");

        File::create(file_path)
    }

    /// Computes the file path for a WAL page.
    ///
    /// Generates the path for a WAL file based on its file number.
    /// Files are named with 8-digit zero-padded numbers starting from 1.
    ///
    /// # Arguments
    ///
    /// * `db_path` - Path to the database directory
    /// * `file_num` - The WAL file number (0-based index)
    ///
    /// # Returns
    ///
    /// Returns the full path to the WAL file.
    ///
    /// # Examples
    ///
    /// - file_num=0 → `db_path/00000001.wal`
    /// - file_num=1 → `db_path/00000002.wal`
    /// - file_num=99 → `db_path/00000100.wal`
    pub fn get_file_path(db_path: &Path, file_num: usize) -> PathBuf {
        db_path.join(Path::new(&format!("{:08}.wal", file_num + 1)))
    }
}
