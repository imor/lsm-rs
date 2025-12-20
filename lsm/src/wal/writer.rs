use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::wal::{LogInner, PAGE_SIZE};
use crate::{Error, disk};

/// The task that actually writes the log to disk
pub struct WalWriter {
    wal_file: File,
    position: usize,
    db_path: PathBuf,
}

impl WalWriter {
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

    /// Start the writer at a specific position after opening a log
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

    /// Open an existing log file (used during recovery/restart)
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

    /// Returns true if the writer is done and the associated task should terminate
    pub async fn update_log(&mut self, inner: &LogInner) -> Result<bool, Error> {
        let (to_write, sync_requested, sync_pos, new_offset, stop_requested) = loop {
            // This works around the following bug:
            // https://github.com/rust-lang/rust/issues/63768
            let fut = inner.queue_cond.notified();
            tokio::pin!(fut);

            {
                let mut status = inner.status.write();
                let to_write = std::mem::take(&mut status.queue);
                let sync_requested = status.sync_requested;
                let sync_pos = status.sync_pos;
                let stop_requested = status.stop_requested;

                let new_offset = if status.offset_pos > status.flush_pos {
                    Some((status.offset_pos, status.flush_pos))
                } else {
                    assert_eq!(status.offset_pos, status.flush_pos);
                    None
                };

                // Check whether there is something to do
                if !to_write.is_empty() || new_offset.is_some() || sync_requested || stop_requested
                {
                    assert_eq!(self.position, status.write_pos);

                    status.sync_requested = false;
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
            inner.status.write().sync_pos = self.position;
        }

        if let Some((new_offset, old_offset)) = new_offset {
            self.set_offset(old_offset, new_offset).await?;
        }

        // Notify about finished write(s)
        {
            let mut lock = inner.status.write();
            assert!(lock.write_pos <= self.position);
            lock.write_pos = self.position;

            if let Some((new_offset, _)) = new_offset {
                lock.flush_pos = new_offset;
            }

            inner.write_cond.notify_waiters();
        }

        if stop_requested {
            log::debug!("WAL writer finished");
        }

        Ok(stop_requested)
    }

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

    async fn sync(&mut self) {
        self.wal_file.sync_data().expect("Data sync failed");
    }

    /// Writes the data to the appropriate wal file
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

    /// Create a new file that is part of the log
    pub fn create_file(db_path: &Path, file_num: usize) -> Result<File, std::io::Error> {
        let file_path = Self::get_file_path(db_path, file_num);
        log::trace!("Creating new wal file at {file_path:?}");

        File::create(file_path)
    }

    pub fn get_file_path(db_path: &Path, file_num: usize) -> PathBuf {
        db_path.join(Path::new(&format!("{:08}.wal", file_num + 1)))
    }
}
