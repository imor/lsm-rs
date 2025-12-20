use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::wal::{LogInner, PAGE_SIZE};
use crate::{Error, disk};

/// The task that actually writes the log to disk
pub struct WalWriter {
    log_file: File,
    position: usize,
    db_path: PathBuf,
}

impl WalWriter {
    pub fn new(db_path: PathBuf) -> Self {
        let log_file = Self::create_file(&db_path, 0).unwrap_or_else(|err| {
            panic!("Failed to create WAL file in directory {db_path:?}: {err}",)
        });

        Self {
            log_file,
            db_path,
            position: 0,
        }
    }

    /// Start the writer at a specific position after opening a log
    pub fn continue_from(position: usize, db_path: PathBuf) -> Self {
        let fpos = position / PAGE_SIZE;

        let log_file = if position.is_multiple_of(PAGE_SIZE) {
            // At the beginning of a new file
            Self::create_file(&db_path, fpos).unwrap_or_else(|err| {
                panic!("Failed to create WAL file in directory {db_path:?}: {err}",)
            })
        } else {
            Self::open_file(&db_path, fpos).unwrap_or_else(|err| {
                panic!("Failed to open WAL file in directory {db_path:?}: {err}",)
            })
        };

        Self {
            log_file,
            db_path,
            position,
        }
    }

    /// Open an existing log file (used during recovery/restart)
    pub fn open_file(db_path: &Path, fpos: usize) -> Result<File, std::io::Error> {
        let fpath = Self::get_file_path(db_path, fpos);
        log::trace!("Opening file at {fpath:?}");

        let log_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(false)
            .truncate(false)
            .open(fpath)?;

        Ok(log_file)
    }

    /// Returns true if the writer is done and the associated task should terminate
    pub async fn update_log(&mut self, inner: &LogInner) -> Result<bool, Error> {
        let (to_write, sync_flag, sync_pos, new_offset, stop_flag) = loop {
            // This works around the following bug:
            // https://github.com/rust-lang/rust/issues/63768
            let fut = inner.queue_cond.notified();
            tokio::pin!(fut);

            {
                let mut lock = inner.status.write();
                let to_write = std::mem::take(&mut lock.queue);
                let sync_flag = lock.sync_requested;
                let sync_pos = lock.sync_pos;
                let stop_flag = lock.stop_requested;

                let new_offset = if lock.offset_pos > lock.flush_pos {
                    Some((lock.offset_pos, lock.flush_pos))
                } else {
                    assert_eq!(lock.offset_pos, lock.flush_pos);
                    None
                };

                // Check whether there is something to do
                if !to_write.is_empty() || new_offset.is_some() || sync_flag || stop_flag {
                    assert_eq!(self.position, lock.write_pos);

                    lock.sync_requested = false;
                    break (to_write, sync_flag, sync_pos, new_offset, stop_flag);
                }

                // wait for change to queue and retry
                assert_eq!(lock.write_pos, lock.queue_pos);
                fut.as_mut().enable();
            }

            fut.await;
        };

        // Don't hold lock while write
        for buf in to_write.into_iter() {
            self.write_all(buf)
                .await
                .map_err(|err| Error::from_io_error("Failed to writ write-ahead log", err))?;
        }

        // Only sync if necessary
        // We do not need to hold the lock while syncing
        // because there is only one write-ahead writer
        if sync_flag && sync_pos < self.position {
            self.sync().await;
            inner.status.write().sync_pos = self.position;
        }

        if let Some((new_offset, old_offset)) = new_offset {
            self.set_offset(new_offset, old_offset).await?;
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

        if stop_flag {
            log::debug!("WAL writer finished");
        }

        Ok(stop_flag)
    }

    async fn set_offset(&mut self, new_offset: usize, old_offset: usize) -> Result<(), Error> {
        let old_file_pos = old_offset / PAGE_SIZE;
        let new_file_pos = new_offset / PAGE_SIZE;

        for fpos in old_file_pos..new_file_pos {
            let fpath = self
                .db_path
                .join(Path::new(&format!("log{:08}.data", fpos + 1)));
            log::trace!("Removing file {fpath:?}");

            disk::remove_file(&fpath).await.map_err(|err| {
                Error::from_io_error(format!("Failed to remove log file {fpath:?}"), err)
            })?;
        }

        Ok(())
    }

    async fn sync(&mut self) {
        self.log_file.sync_data().expect("Data sync failed");
    }

    #[allow(unused_mut)]
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
            self.log_file
                .write_all(to_write)
                .expect("Failed to write log file");

            buf_pos += write_len;
            self.position += write_len;
            file_offset += write_len;

            assert!(file_offset <= PAGE_SIZE);

            // Create a new file?
            if file_offset == PAGE_SIZE {
                let file_pos = self.position / PAGE_SIZE;
                self.log_file = Self::create_file(&self.db_path, file_pos)?;
            }
        }

        Ok(())
    }

    /// Create a new file that is part of the log
    pub fn create_file(db_path: &Path, file_pos: usize) -> Result<File, std::io::Error> {
        let fpath = Self::get_file_path(db_path, file_pos);
        log::trace!("Creating new log file at {fpath:?}");

        File::create(fpath)
    }

    pub fn get_file_path(db_path: &Path, fpos: usize) -> PathBuf {
        db_path.join(Path::new(&format!("log{:08}.data", fpos + 1)))
    }
}
