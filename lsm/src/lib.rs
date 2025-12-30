//! # lsm-rs: An asynchronous LSM-based key-value database
//!
//! `lsm-rs` is a high-performance, asynchronous implementation of a Log-Structured
//! Merge-tree (LSM) key-value database written in Rust. It provides efficient storage
//! and retrieval of key-value pairs with support for range queries, batch operations,
//! and automatic background compaction.
//!
//! ## Overview
//!
//! LSM databases optimize for write-heavy workloads by using a log-structured approach.
//! Writes are first added to an in-memory table (memtable) and a write-ahead log (WAL)
//! for durability. When the memtable fills up, it's flushed to disk as a sorted table.
//! Multiple sorted tables are periodically merged (compacted) to maintain read performance.
//!
//! ## Features
//!
//! - **Asynchronous I/O**: All operations use async/await for high concurrency
//! - **Write-ahead logging (WAL)**: Durability guarantees for crash recovery
//! - **Background compaction**: Automatic memtable and level compaction
//! - **Range queries**: Efficient iteration over key ranges
//! - **Batch writes**: Atomic multi-key updates via [`WriteBatch`]
//! - **Bloom filters** (optional): Reduce disk I/O for negative lookups
//! - **WiscKey separation** (optional): Store large values separately
//! - **Snappy compression** (optional): Reduce storage space
//!
//! ## Feature Flags
//!
//! This crate supports the following optional features:
//!
//! - **`wisckey`**: Enables WiscKey-style value separation for large values.
//!   Values are stored in a separate value log, reducing write amplification.
//!   
//! - **`bloom-filters`**: Adds Bloom filters to sorted tables, reducing disk
//!   reads for non-existent keys.
//!   
//! - **`snappy-compression`**: Compresses data blocks using Snappy compression,
//!   trading CPU for reduced storage and I/O.
//!
//! Enable features in `Cargo.toml`:
//! ```toml
//! [dependencies]
//! lsm = { version = "*", features = ["bloom-filters", "snappy-compression"] }
//! ```
//!
//! ## Quick Start
//!
//! ```no_run
//! use lsm::{Database, StartMode, WriteBatch};
//! use futures::StreamExt;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), lsm::Error> {
//!     // Open or create a database
//!     let db = Database::new(StartMode::CreateOrOpen).await?;
//!
//!     // Insert a key-value pair
//!     db.put(b"hello".to_vec(), b"world".to_vec()).await?;
//!
//!     // Retrieve the value
//!     if let Some(entry) = db.get(b"hello").await? {
//!         println!("Value: {:?}", entry.get_value());
//!     }
//!
//!     // Batch write multiple keys
//!     let mut batch = WriteBatch::new();
//!     batch.put(b"key1".to_vec(), b"value1".to_vec());
//!     batch.put(b"key2".to_vec(), b"value2".to_vec());
//!     db.write(batch).await?;
//!
//!     // Iterate over a range
//!     let mut iter = db.range_iter(b"key1", b"key3").await;
//!     while let Some((key, entry)) = iter.next().await {
//!         println!("{:?}: {:?}", key, entry.get_value());
//!     }
//!
//!     // Graceful shutdown
//!     db.stop().await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Core Types
//!
//! - [`Database`]: The main database interface for CRUD operations
//! - [`WriteBatch`]: Batch multiple writes for atomic updates
//! - [`StartMode`]: Controls how the database is opened or created
//! - [`Params`]: Configuration parameters for database behavior
//! - [`EntryRef`]: A reference to a database entry (key-value pair)
//! - [`Key`]: Type alias for `Vec<u8>` (database keys)
//! - [`Value`]: Type alias for `Vec<u8>` (database values)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────┐
//! │   Database  │  ← Main user interface
//! └──────┬──────┘
//!        │
//!        ├─→ Memtable (in-memory writes)
//!        ├─→ WAL (durability)
//!        ├─→ Level 0 (flushed memtables)
//!        └─→ Level 1..N (compacted sorted tables)
//! ```
//!
//! ## Performance Considerations
//!
//! - **Writes**: Very fast (memtable + WAL append)
//! - **Reads**: May require checking multiple levels (use bloom filters)
//! - **Range scans**: Efficient due to sorted table structure
//! - **Compaction**: Background tasks keep read performance stable
//!
//! ## Safety and Concurrency
//!
//! - The [`Database`] struct is `Clone` and can be shared across tasks/threads
//! - **Never** open the same database directory with multiple `Database` instances
//! - Always call [`Database::stop`] for graceful shutdown

// Temporary workaround for the io_uring code
#![allow(clippy::arc_with_non_send_sync)]

pub mod iterate;

#[cfg(feature = "wisckey")]
pub mod values;

/// Database configuration parameters.
///
/// Re-exported from the `params` module. See [`Params`] for details.
mod params;
pub use params::Params;

/// Batch write operations and write options.
///
/// Re-exported types:
/// - [`WriteBatch`]: Container for batching multiple write operations
/// - [`WriteOp`]: Individual write operation (put or delete)
/// - [`WriteOptions`]: Options controlling write behavior (e.g., sync flag)
mod write_batch;
pub use write_batch::{WriteBatch, WriteOp, WriteOptions};

/// Sorted table (SSTable) implementation.
///
/// This module contains the on-disk sorted table format, including
/// data blocks, index blocks, and bloom filters.
pub mod sorted_table;

mod level_logger;

/// In-memory table (memtable) implementation.
///
/// Memtables buffer writes in memory before being flushed to disk.
pub mod memtable;

/// Background task management for compaction.
///
/// This module manages asynchronous compaction tasks that merge
/// sorted tables to maintain LSM structure.
pub mod tasks;

/// Core database logic and state management.
pub mod logic;

/// A reference to a database entry (key-value pair).
///
/// Re-exported from the `logic` module. [`EntryRef`] provides zero-copy
/// access to entries retrieved from the database.
pub use logic::EntryRef;

/// Manifest file management for database metadata.
///
/// The manifest tracks the current state of sorted tables and levels.
pub mod manifest;

mod data_blocks;
mod database;
mod disk;
mod index_blocks;
mod level;
mod wal;

/// Type alias for database keys.
///
/// Keys are represented as byte vectors (`Vec<u8>`). They are stored
/// and compared in lexicographic order.
pub type Key = Vec<u8>;

/// Type alias for database values.
///
/// Values are represented as byte vectors (`Vec<u8>`). With the `wisckey`
/// feature enabled, large values may be stored separately in a value log.
pub type Value = Vec<u8>;

/// Shorthand for a list of key-value pairs
#[cfg(feature = "wisckey")]
type EntryList = Vec<(Key, Value)>;

/// The main database interface.
///
/// Re-exported from the `database` module. [`Database`] is the primary
/// entry point for all database operations. See the [`database`] module
/// documentation for detailed usage examples.
pub use database::Database;

/// How many bytes do we align by?
const WORD_SIZE: usize = 8;

/// Pads the given offset to be aligned to WORD_SIZE.
fn pad_offset(offset: usize) -> usize {
    offset + compute_padding(offset)
}

/// Computes the number of padding bytes needed to align to WORD_SIZE.
fn compute_padding(offset: usize) -> usize {
    let remainder = offset % WORD_SIZE;
    if remainder == 0 {
        0
    } else {
        WORD_SIZE - remainder
    }
}

/// Adds padding bytes to align data to WORD_SIZE.
fn add_padding(data: &mut Vec<u8>) {
    let padding = compute_padding(data.len());
    if padding > 0 {
        data.resize(data.len() + padding, 0u8);
    }
}

/// Error types returned by database operations.
///
/// All database methods return `Result<T, Error>` where `Error` represents
/// the possible failure modes.
#[derive(Clone, Debug)]
pub enum Error {
    /// An I/O error occurred during file operations.
    ///
    /// Contains contextual information about where the error occurred
    /// and the underlying system error message.
    Io { context: String, message: String },

    /// Invalid database parameters were provided.
    ///
    /// This error is returned when creating a database with invalid
    /// configuration (e.g., negative memtable size).
    InvalidParams(String),

    /// A serialization or deserialization error occurred.
    ///
    /// This typically indicates corrupted data files or incompatible
    /// database versions.
    Serialization(String),
}

impl std::fmt::Display for Error {
    /// Formats the error for user-friendly display.
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        match self {
            Self::Io { context, message } => {
                fmt.write_fmt(format_args!("{context}: {message}"))?;
            }
            Self::InvalidParams(msg) => {
                fmt.write_fmt(format_args!("Invalid Parameter: {msg}"))?;
            }
            Self::Serialization(msg) => {
                fmt.write_fmt(format_args!("Serialization Error: {msg}"))?;
            }
        }

        Ok(())
    }
}

impl Error {
    /// Creates an `Error::Io` variant from a context string and an `std::io::Error`.
    ///
    /// This helper method is used to wrap I/O errors with additional context
    /// about where the error occurred.
    ///
    /// # Arguments
    ///
    /// * `context` - A string describing the operation or location of the error
    /// * `inner` - The underlying I/O error
    fn from_io_error<S: ToString>(context: S, inner: std::io::Error) -> Self {
        Self::Io {
            context: context.to_string(),
            message: format!("{inner}"),
        }
    }
}

/// Controls how the database is opened or created.
///
/// This enum allows you to specify the behavior when calling [`Database::new`]
/// or [`Database::new_with_params`].
///
/// # Examples
///
/// ```no_run
/// use lsm::{Database, StartMode};
///
/// # async fn example() -> Result<(), lsm::Error> {
/// // Create or open (most common)
/// let db = Database::new(StartMode::CreateOrOpen).await?;
///
/// // Open existing only (fail if not found)
/// let db = Database::new(StartMode::Open).await?;
///
/// // Create new (overwrite existing)
/// let db = Database::new(StartMode::CreateOrOverride).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub enum StartMode {
    /// Reuse an existing database, or create a new one if it doesn't exist.
    ///
    /// This is the most common mode for typical database usage.
    CreateOrOpen,

    /// Open an existing database only.
    ///
    /// Returns an error if the database doesn't exist. Use this when you
    /// want to ensure you're not accidentally creating a new database.
    Open,

    /// Create a new database, overwriting any existing data.
    ///
    /// **Warning**: This will delete all existing data in the database directory.
    /// Use with caution!
    CreateOrOverride,
}
