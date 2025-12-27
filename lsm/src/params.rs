//! Configuration parameters for the LSM-tree database.
//!
//! This module defines the `Params` struct which controls all configurable aspects of the
//! database including storage paths, memory limits, compaction settings, and performance tuning.

use std::path::{Path, PathBuf};

use crate::{Error, level_logger::LevelLogger};

/// Configuration parameters for customizing database behavior.
///
/// `Params` controls all aspects of LSM-tree operation including:
/// - Storage location and file management
/// - Memory usage (memtable size, open file limits)
/// - Data block organization (key block size, restart intervals)
/// - Compaction behavior (concurrency, triggers)
/// - Performance monitoring (level statistics logging)
///
/// ## Memory Management
///
/// - `max_memtable_size`: Larger values reduce write amplification but increase memory usage
/// - `max_open_files`: Controls the cache size for data blocks and index blocks
///
/// ## Compaction Tuning
///
/// - `compaction_concurrency`: Higher values improve throughput but increase CPU/IO load
/// - `seek_based_compaction`: Triggers compaction on frequently accessed tables
///
/// ## Data Block Tuning
///
/// - `block_restart_interval`: Trade-off between compression ratio and lookup speed
/// - `max_key_block_size`: Affects memory usage and disk I/O patterns
#[derive(Debug, Clone)]
pub struct Params {
    /// Filesystem path where the database files are stored.
    pub db_path: PathBuf,

    /// Maximum size of a memtable in bytes before it's flushed to disk.
    /// Also indirectly defines the maximum value block size. Default: 5MB.
    pub max_memtable_size: usize,

    /// Number of levels in the LSM tree. More levels allow for larger databases
    /// but may increase read latency. Default: 5.
    pub num_levels: usize,

    /// Maximum number of files (data blocks + index blocks) to keep open simultaneously.
    /// Higher values improve read performance but increase memory usage. Default: 1,000,000.
    pub max_open_files: usize,

    /// Maximum number of key-value entries per data block. Affects block size and
    /// memory usage. Default: 512.
    pub max_key_block_size: usize,

    /// How many entries between restart points in a data block. Lower values improve
    /// lookup speed but reduce compression ratio. Default: 16.
    pub block_restart_interval: u32,

    /// Optional path to a CSV file for logging level statistics (sizes, counts).
    /// Set to `None` to disable logging.
    pub log_level_stats: Option<String>,

    /// Number of concurrent compaction tasks. Higher values improve compaction throughput
    /// but increase CPU and I/O load. Default: 4.
    pub compaction_concurrency: usize,

    /// Number of seeks per kilobyte before triggering compaction on a table.
    /// Set to `None` to disable seek-based compaction. Default: Some(10).
    pub seek_based_compaction: Option<u32>,
}

impl Params {
    /// Validates the configuration parameters.
    ///
    /// Checks that all required parameters are properly configured:
    /// - Database path is not empty
    /// - Database path is a directory (if it exists)
    /// - Number of levels is at least 1
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidParams` if any validation check fails.
    pub fn validate(&self) -> Result<(), Error> {
        if self.db_path.components().next().is_none() {
            return Err(Error::InvalidParams(
                "DB path must not be empty!".to_string(),
            ));
        }

        if self.db_path.exists() && !self.db_path.is_dir() {
            return Err(Error::InvalidParams(
                "DB path must be a folder!".to_string(),
            ));
        }

        if self.num_levels == 0 {
            return Err(Error::InvalidParams(
                "There must be at least one level!".to_string(),
            ));
        }

        Ok(())
    }

    /// Creates a level logger if statistics logging is enabled.
    ///
    /// Returns a `LevelLogger` instance if `log_level_stats` is set, otherwise returns `None`.
    pub(crate) fn create_level_logger(&self) -> Option<LevelLogger> {
        self.log_level_stats
            .as_ref()
            .map(|path| LevelLogger::new(path, self.num_levels))
    }
}

impl Default for Params {
    /// Creates a `Params` instance with sensible default values.
    ///
    /// Default configuration:
    /// - `db_path`: `./storage.lsm`
    /// - `max_memtable_size`: 5 MB
    /// - `num_levels`: 5
    /// - `max_open_files`: 1,000,000
    /// - `max_key_block_size`: 512 entries
    /// - `block_restart_interval`: 16 entries
    /// - `log_level_stats`: None (disabled)
    /// - `compaction_concurrency`: 4 tasks
    /// - `seek_based_compaction`: Some(10) seeks per KB
    fn default() -> Self {
        Self {
            db_path: Path::new("./storage.lsm").to_path_buf(),
            max_memtable_size: 5 * 1024 * 1024,
            num_levels: 5,
            max_open_files: 1_000_000,
            max_key_block_size: 512,
            block_restart_interval: 16,
            log_level_stats: None,
            compaction_concurrency: 4,
            seek_based_compaction: Some(10),
        }
    }
}
