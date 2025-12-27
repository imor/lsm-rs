//! CSV-based statistics logging for LSM-tree level metrics.
//!
//! This module provides a simple logger that tracks the number of SSTables in each
//! level over time and writes this data to a CSV file for analysis and visualization.
//!
//! # Purpose
//!
//! The level logger helps monitor and analyze LSM-tree behavior by recording:
//! - How the number of tables in each level changes over time
//! - The impact of compactions on level sizes
//! - Growth patterns and compaction effectiveness
//!
//! # CSV Format
//!
//! The output CSV has the following structure:
//! - Column 1: Timestamp in milliseconds since logger creation
//! - Columns 2-N: Number of tables in level 0, level 1, ..., level N-1
//!
//! Example:
//! ```csv
//! time,level0,level1,level2,level3
//! 0,0,0,0,0
//! 1523,1,0,0,0
//! 3045,2,0,0,0
//! 4721,1,5,0,0
//! ```
//!
//! # Usage
//!
//! The logger is created at database initialization and updated automatically
//! when memtables are flushed or compactions complete.

use std::fs::File;
use std::time::Instant;

use crate::manifest::LevelId;

use parking_lot::Mutex;

/// Internal state for the level logger.
///
/// Wrapped in a Mutex to allow concurrent updates from compaction tasks.
struct Inner {
    /// Start time for calculating elapsed milliseconds.
    start: Instant,
    /// CSV writer for the output file.
    outfile: csv::Writer<File>,
    /// Current count of tables in each level (indexed by level ID).
    num_tables: Vec<usize>,
}

/// Logs LSM-tree level statistics to a CSV file.
///
/// Tracks the number of SSTables in each level over time, writing updates to a CSV file
/// whenever tables are added (via memtable flush) or moved (via compaction).
///
/// The logger is thread-safe and can be updated concurrently from multiple compaction tasks.
pub(crate) struct LevelLogger {
    /// Mutex-protected internal state.
    inner: Mutex<Inner>,
}

impl LevelLogger {
    /// Creates a new level logger.
    ///
    /// Initializes a CSV file with headers and sets all level counts to zero.
    ///
    /// # Arguments
    /// * `path` - Path to the output CSV file
    /// * `num_levels` - Total number of levels in the LSM-tree
    ///
    /// # Panics
    /// Panics if the file cannot be created or the header cannot be written.
    pub fn new(path: &str, num_levels: usize) -> Self {
        let outfile = csv::Writer::from_path(path).expect("Failed to create log file");

        let inner = Inner::new(outfile, num_levels);

        Self {
            inner: Mutex::new(inner),
        }
    }

    /// Records that a new table was added to L0.
    ///
    /// Called when a memtable is flushed to disk, creating a new L0 SSTable.
    /// Increments the L0 table count and writes a new CSV row.
    pub fn l0_table_added(&self) {
        let mut inner = self.inner.lock();
        inner.num_tables[0] += 1;

        inner.write();
    }

    /// Records a compaction between levels.
    ///
    /// Updates the table counts for two consecutive levels when a compaction
    /// merges tables from one level into the next.
    ///
    /// # Arguments
    /// * `level` - The source level (tables removed from this level)
    /// * `added` - Number of tables created in level + 1
    /// * `removed` - Number of tables removed from level
    ///
    /// Writes a new CSV row with the updated counts.
    pub fn compaction(&self, level: LevelId, added: usize, removed: usize) {
        let mut inner = self.inner.lock();
        inner.num_tables[level as usize] -= removed;
        inner.num_tables[level as usize + 1] += added;

        inner.write();
    }
}

impl Inner {
    /// Creates new internal logger state.
    ///
    /// Initializes the table count array to all zeros and writes the CSV header row.
    ///
    /// # Arguments
    /// * `outfile` - CSV writer for the output file
    /// * `num_levels` - Number of levels in the LSM-tree
    ///
    /// # CSV Header
    /// Writes a header row like: `time,level0,level1,level2,...`
    fn new(mut outfile: csv::Writer<File>, num_levels: usize) -> Self {
        let num_tables = vec![0; num_levels];

        let mut header = vec![format!("time")];
        for idx in 0..num_levels {
            header.push(format!("level{idx}"));
        }

        outfile.write_record(&header).unwrap();

        Self {
            outfile,
            num_tables,
            start: Instant::now(),
        }
    }

    /// Writes the current state to the CSV file.
    ///
    /// Creates a CSV row with:
    /// - Current timestamp (milliseconds since logger creation)
    /// - Table count for each level
    ///
    /// # Panics
    /// Panics if writing to the CSV file fails.
    fn write(&mut self) {
        let mut record = vec![];
        record.push(format!("{}", self.start.elapsed().as_millis()));

        for count in self.num_tables.iter() {
            record.push(format!("{count}"));
        }

        self.outfile.write_record(&record).unwrap();
    }
}
