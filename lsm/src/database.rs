//! Database API - The main user-facing interface for LSM-based key-value storage.
//!
//! This module provides the [`Database`] struct, which is the primary entry point
//! for interacting with the LSM (Log-Structured Merge-tree) database. It supports
//! asynchronous CRUD operations, range queries, batch writes, and background
//! compaction tasks.
//!
//! # Features
//!
//! - **Concurrent access**: The [`Database`] can be cloned and shared across threads
//! - **Asynchronous I/O**: All operations use async/await for non-blocking performance
//! - **Batch writes**: Efficient multi-key updates via [`WriteBatch`]
//! - **Range queries**: Iterate over key ranges with forward and reverse iterators
//! - **Background compaction**: Automatic memtable and level compaction
//! - **Graceful shutdown**: Clean termination of background tasks
//!
//! # Example
//!
//! ```no_run
//! use lsm::{Database, StartMode};
//!
//! # async fn example() -> Result<(), lsm::Error> {
//! // Open or create a database
//! let db = Database::new(StartMode::CreateOrOpen).await?;
//!
//! // Put a key-value pair
//! db.put(b"key".to_vec(), b"value".to_vec()).await?;
//!
//! // Get the value back
//! if let Some(entry) = db.get(b"key").await? {
//!     println!("Value: {:?}", entry.get_value());
//! }
//!
//! // Delete a key
//! db.delete(b"key".to_vec()).await?;
//!
//! // Graceful shutdown
//! db.stop().await?;
//! # Ok(())
//! # }
//! ```

use crate::iterate::DbIterator;
use crate::logic::{DbLogic, EntryRef};
use crate::tasks::{TaskManager, TaskType};
use crate::{Error, Key, Params, StartMode, Value, WriteBatch, WriteOptions};

use std::sync::Arc;

/// The main database structure for LSM-based key-value storage.
///
/// `Database` is the primary interface for interacting with an LSM database.
/// It provides asynchronous methods for CRUD operations, range queries, and
/// batch writes. The struct is cheaply clonable (via `Arc`) and can be safely
/// shared across multiple tasks or threads.
///
/// # Concurrency
///
/// Multiple `Database` instances can access the same in-memory state concurrently,
/// but **you should never instantiate more than one `Database` for the same on-disk
/// files**, as this would lead to data corruption.
///
/// # Background Tasks
///
/// The database automatically manages background tasks for:
/// - Memtable compaction (flushing in-memory data to disk)
/// - Level compaction (merging sorted tables to maintain LSM structure)
///
/// These tasks are triggered automatically when thresholds are met.
///
/// # Shutdown
///
/// Call [`Database::stop`] to gracefully shut down all background tasks before
/// dropping the database. The `Drop` implementation will terminate tasks, but
/// calling `stop()` explicitly ensures proper cleanup.
///
/// # Example
///
/// ```no_run
/// use lsm::{Database, StartMode, WriteBatch};
/// use futures::StreamExt;
///
/// # async fn example() -> Result<(), lsm::Error> {
/// let db = Database::new(StartMode::CreateOrOpen).await?;
///
/// // Single write
/// db.put(b"key1".to_vec(), b"value1".to_vec()).await?;
///
/// // Batch write
/// let mut batch = WriteBatch::new();
/// batch.put(b"key2".to_vec(), b"value2".to_vec());
/// batch.delete(b"old_key".to_vec());
/// db.write(batch).await?;
///
/// // Range iteration
/// let mut iter = db.range_iter(b"key1", b"key3").await;
/// while let Some((key, entry)) = iter.next().await {
///     println!("{:?}: {:?}", key, entry.get_value());
/// }
///
/// db.stop().await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Database {
    inner: Arc<DbLogic>,
    tasks: Arc<TaskManager>,
}

impl Database {
    /// Creates a new database instance with default parameters.
    ///
    /// This is a convenience method that uses [`Params::default()`] for configuration.
    /// For custom parameters, use [`Database::new_with_params`].
    ///
    /// # Arguments
    ///
    /// * `mode` - Controls how the database is opened or created (see [`StartMode`])
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The database directory cannot be created or accessed
    /// - The manifest file is corrupted
    /// - The WAL (Write-Ahead Log) cannot be initialized
    ///
    /// # Example
    ///
    /// ```no_run
    /// use lsm::{Database, StartMode};
    ///
    /// # async fn example() -> Result<(), lsm::Error> {
    /// let db = Database::new(StartMode::CreateOrOpen).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new(mode: StartMode) -> Result<Self, Error> {
        let params = Params::default();
        Self::new_with_params(mode, params).await
    }

    /// Creates a new database instance with custom parameters.
    ///
    /// Use this method when you need fine-grained control over database behavior,
    /// such as memtable size, compaction concurrency, or bloom filter settings.
    ///
    /// # Arguments
    ///
    /// * `mode` - Controls how the database is opened or created
    /// * `params` - Database configuration parameters
    ///
    /// # Errors
    ///
    /// Returns an error if the parameters are invalid or if database initialization fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use lsm::{Database, StartMode, Params};
    ///
    /// # async fn example() -> Result<(), lsm::Error> {
    /// let mut params = Params::default();
    /// params.compaction_concurrency = 4;
    ///
    /// let db = Database::new_with_params(StartMode::CreateOrOpen, params).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new_with_params(mode: StartMode, params: Params) -> Result<Self, Error> {
        let compaction_concurrency = params.compaction_concurrency;

        let inner = Arc::new(DbLogic::new(mode, params).await?);
        let tasks = Arc::new(TaskManager::new(inner.clone(), compaction_concurrency).await);

        Ok(Self { inner, tasks })
    }

    /// Retrieves the value associated with a key.
    ///
    /// This method searches for the key in memtables first, then in sorted tables
    /// on disk. It returns a zero-copy reference to the entry if found.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up
    ///
    /// # Returns
    ///
    /// - `Ok(Some(EntryRef))` if the key exists and is not deleted
    /// - `Ok(None)` if the key doesn't exist or was deleted
    /// - `Err(Error)` if an I/O error occurs
    ///
    /// # Performance
    ///
    /// This method may trigger background compaction if internal thresholds are met.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lsm::Database;
    /// # async fn example(db: &Database) -> Result<(), lsm::Error> {
    /// match db.get(b"my_key").await? {
    ///     Some(entry) => println!("Found: {:?}", entry.get_value()),
    ///     None => println!("Key not found"),
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[tracing::instrument(skip(self, key))]
    pub async fn get(&self, key: &[u8]) -> Result<Option<EntryRef>, Error> {
        match self.inner.get(key).await {
            Ok((needs_compaction, data)) => {
                if needs_compaction {
                    self.tasks.wake_up(&TaskType::LevelCompaction);
                }

                Ok(data)
            }
            Err(err) => Err(err),
        }
    }

    /// Deletes a key from the database.
    ///
    /// This operation uses tombstone markers rather than immediate deletion.
    /// The key will be marked as deleted, and the actual data will be removed
    /// during compaction.
    ///
    /// **Note**: For efficiency, this method does not verify whether the key exists.
    /// It simply marks the most recent version as deleted.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to delete
    ///
    /// # Errors
    ///
    /// Returns an error if the delete operation cannot be written to the WAL.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lsm::Database;
    /// # async fn example(db: &Database) -> Result<(), lsm::Error> {
    /// db.delete(b"unwanted_key".to_vec()).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[tracing::instrument(skip(self, key))]
    pub async fn delete(&self, key: Key) -> Result<(), Error> {
        let mut batch = WriteBatch::new();
        batch.delete(key);

        self.write_opts(batch, &WriteOptions::default()).await
    }

    /// Ensures all pending writes are flushed to disk.
    ///
    /// This method is only necessary if you've performed writes with
    /// `sync=false` in [`WriteOptions`]. It forces a synchronous flush
    /// of all buffered data to persistent storage.
    ///
    /// # Durability
    ///
    /// After this call returns successfully, all previous writes are guaranteed
    /// to survive a system crash.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lsm::{Database, WriteOptions};
    /// # async fn example(db: &Database) -> Result<(), lsm::Error> {
    /// let opts = WriteOptions { sync: false };
    /// db.put_opts(b"key".to_vec(), b"value".to_vec(), &opts).await?;
    ///
    /// // Later, ensure durability
    /// db.synchronize().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn synchronize(&self) -> Result<(), Error> {
        self.inner.synchronize().await
    }

    /// Deletes a key with custom write options.
    ///
    /// This is the lower-level version of [`Database::delete`] that allows
    /// you to control write behavior (e.g., synchronous vs. asynchronous writes).
    ///
    /// # Arguments
    ///
    /// * `key` - The key to delete
    /// * `opts` - Write options (e.g., `sync` flag)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lsm::{Database, WriteOptions};
    /// # async fn example(db: &Database) -> Result<(), lsm::Error> {
    /// let opts = WriteOptions { sync: true };
    /// db.delete_opts(b"key".to_vec(), &opts).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn delete_opts(&self, key: Key, opts: &WriteOptions) -> Result<(), Error> {
        let mut batch = WriteBatch::new();
        batch.delete(key);
        self.write_opts(batch, opts).await
    }

    /// Inserts or updates a key-value pair.
    ///
    /// If the key already exists, its value will be updated. This operation
    /// uses default write options (synchronous write).
    ///
    /// # Arguments
    ///
    /// * `key` - The key to insert or update
    /// * `value` - The value to associate with the key
    ///
    /// # Errors
    ///
    /// Returns an error if the write cannot be persisted to the WAL.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lsm::Database;
    /// # async fn example(db: &Database) -> Result<(), lsm::Error> {
    /// db.put(b"user:1".to_vec(), b"Alice".to_vec()).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn put(&self, key: Key, value: Value) -> Result<(), Error> {
        const OPTS: WriteOptions = WriteOptions::new();
        self.put_opts(key, value, &OPTS).await
    }

    /// Inserts or updates a key-value pair with custom write options.
    ///
    /// This is the lower-level version of [`Database::put`] that allows you
    /// to control write behavior.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to insert or update
    /// * `value` - The value to associate with the key
    /// * `opts` - Write options (e.g., `sync` flag for durability control)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lsm::{Database, WriteOptions};
    /// # async fn example(db: &Database) -> Result<(), lsm::Error> {
    /// // Fast asynchronous write
    /// let opts = WriteOptions { sync: false };
    /// db.put_opts(b"key".to_vec(), b"value".to_vec(), &opts).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[tracing::instrument(skip(self))]
    pub async fn put_opts(&self, key: Key, value: Value, opts: &WriteOptions) -> Result<(), Error> {
        let mut batch = WriteBatch::new();
        batch.put(key, value);
        self.write_opts(batch, opts).await
    }

    /// Returns an iterator over all entries in the database.
    ///
    /// The iterator yields entries in ascending key order. It provides a
    /// consistent snapshot of the database at the time `iter()` is called.
    ///
    /// # Returns
    ///
    /// A [`DbIterator`] that can be used with `.next().await` to retrieve entries.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lsm::Database;
    /// # use futures::StreamExt;
    /// # async fn example(db: &Database) -> Result<(), lsm::Error> {
    /// let mut iter = db.iter().await;
    /// while let Some((key, entry)) = iter.next().await {
    ///     println!("{:?}: {:?}", key, entry.get_value());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn iter(&self) -> DbIterator {
        let (mem_iters, table_iters, min_key, max_key) = self.inner.prepare_iter(None, None).await;

        DbIterator::new(
            mem_iters,
            table_iters,
            min_key,
            max_key,
            false,
            #[cfg(feature = "wisckey")]
            self.inner.get_value_log(),
        )
    }

    /// Returns an iterator over a range of keys.
    ///
    /// The iterator yields entries with keys in the range `[min_key, max_key)`
    /// (inclusive start, exclusive end) in ascending order.
    ///
    /// # Arguments
    ///
    /// * `min_key` - The minimum key (inclusive)
    /// * `max_key` - The maximum key (exclusive)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lsm::Database;
    /// # use futures::StreamExt;
    /// # async fn example(db: &Database) -> Result<(), lsm::Error> {
    /// // Iterate over keys from "a" to "m" (not including "m")
    /// let mut iter = db.range_iter(b"a", b"m").await;
    /// while let Some((key, _entry)) = iter.next().await {
    ///     println!("{:?}", key);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn range_iter(&self, min_key: &[u8], max_key: &[u8]) -> DbIterator {
        let (mem_iters, table_iters, min_key, max_key) =
            self.inner.prepare_iter(Some(min_key), Some(max_key)).await;

        DbIterator::new(
            mem_iters,
            table_iters,
            min_key,
            max_key,
            false,
            #[cfg(feature = "wisckey")]
            self.inner.get_value_log(),
        )
    }

    /// Returns a reverse iterator over a range of keys.
    ///
    /// The iterator yields entries with keys in the range `(min_key, max_key]`
    /// (exclusive start, inclusive end) in descending order.
    ///
    /// # Arguments
    ///
    /// * `max_key` - The maximum key (inclusive)
    /// * `min_key` - The minimum key (exclusive)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lsm::Database;
    /// # use futures::StreamExt;
    /// # async fn example(db: &Database) -> Result<(), lsm::Error> {
    /// // Iterate backwards from "z" to "m" (not including "m")
    /// let mut iter = db.reverse_range_iter(b"z", b"m").await;
    /// while let Some((key, _entry)) = iter.next().await {
    ///     println!("{:?}", key);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn reverse_range_iter(&self, max_key: &[u8], min_key: &[u8]) -> DbIterator {
        let (mem_iters, table_iters, min_key, max_key) = self
            .inner
            .prepare_reverse_iter(Some(max_key), Some(min_key))
            .await;

        DbIterator::new(
            mem_iters,
            table_iters,
            min_key,
            max_key,
            true,
            #[cfg(feature = "wisckey")]
            self.inner.get_value_log(),
        )
    }

    /// Writes a batch of updates atomically to the database.
    ///
    /// This method is more efficient than multiple individual `put()` or `delete()`
    /// calls when you need to update multiple keys. All operations in the batch
    /// are applied atomically.
    ///
    /// **Note**: For single-key writes, use [`Database::put`] instead.
    ///
    /// # Arguments
    ///
    /// * `write_batch` - A batch of put and delete operations
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lsm::{Database, WriteBatch};
    /// # async fn example(db: &Database) -> Result<(), lsm::Error> {
    /// let mut batch = WriteBatch::new();
    /// batch.put(b"key1".to_vec(), b"value1".to_vec());
    /// batch.put(b"key2".to_vec(), b"value2".to_vec());
    /// batch.delete(b"old_key".to_vec());
    ///
    /// db.write(batch).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn write(&self, write_batch: WriteBatch) -> Result<(), Error> {
        const OPTS: WriteOptions = WriteOptions::new();
        self.write_opts(write_batch, &OPTS).await
    }

    /// Writes a batch of updates with custom write options.
    ///
    /// This is the lower-level version of [`Database::write`] that allows you
    /// to control write behavior (e.g., synchronous vs. asynchronous writes).
    ///
    /// # Arguments
    ///
    /// * `write_batch` - A batch of put and delete operations
    /// * `opts` - Write options (e.g., `sync` flag)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lsm::{Database, WriteBatch, WriteOptions};
    /// # async fn example(db: &Database) -> Result<(), lsm::Error> {
    /// let mut batch = WriteBatch::new();
    /// batch.put(b"key1".to_vec(), b"value1".to_vec());
    ///
    /// let opts = WriteOptions { sync: false };
    /// db.write_opts(batch, &opts).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[tracing::instrument(skip(self, write_batch, opts))]
    pub async fn write_opts(
        &self,
        write_batch: WriteBatch,
        opts: &WriteOptions,
    ) -> Result<(), Error> {
        let needs_compaction = self.inner.write_opts(write_batch, opts).await?;

        if needs_compaction {
            self.tasks.wake_up(&TaskType::MemtableCompaction);
        }

        Ok(())
    }

    /// Gracefully shuts down the database and all background tasks.
    ///
    /// This method:
    /// 1. Flushes any pending writes to disk
    /// 2. Stops all background compaction tasks
    /// 3. Ensures all data is persisted
    ///
    /// **Important**: Always call this method before dropping the database to
    /// ensure data integrity. While the `Drop` implementation will terminate tasks,
    /// calling `stop()` explicitly allows for proper error handling.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing data to disk fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lsm::{Database, StartMode};
    /// # async fn example() -> Result<(), lsm::Error> {
    /// let db = Database::new(StartMode::CreateOrOpen).await?;
    /// // ... use database ...
    /// db.stop().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn stop(&self) -> Result<(), Error> {
        self.inner.stop().await?;
        self.tasks.stop_all().await
    }
}

impl Drop for Database {
    /// Cleans up background tasks when the database is dropped.
    fn drop(&mut self) {
        self.tasks.terminate();
    }
}
