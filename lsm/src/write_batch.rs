//! Write batching for atomic multi-operation commits.
//!
//! This module provides `WriteBatch` for grouping multiple write operations (puts and deletes)
//! into a single atomic batch. Batching improves throughput by:
//! - Reducing write amplification
//! - Amortizing WAL write costs
//! - Enabling more efficient memtable operations
//!
//! ## Usage
//!
//! ```no_run
//! # use lsm::WriteBatch;
//! let mut batch = WriteBatch::new();
//! batch.put(b"key1".to_vec(), b"value1".to_vec());
//! batch.put(b"key2".to_vec(), b"value2".to_vec());
//! batch.delete(b"key3".to_vec());
//! // Pass batch to Database::write() to apply atomically
//! ```

use crate::{Key, Value, memtable::Memtable};

/// A single write operation (Put or Delete).
#[derive(Debug)]
pub enum WriteOp {
    /// Insert or update a key-value pair.
    Put(Key, Value),

    /// Delete a key (creates a tombstone marker).
    Delete(Key),
}

/// A batch of write operations for atomic, high-throughput commits.
///
/// `WriteBatch` groups multiple put and delete operations together. The batch is not applied
/// to the database until it's passed to `Database::write()`, ensuring atomicity.
///
/// ## Benefits
///
/// - **Atomicity**: All operations in a batch succeed or fail together
/// - **Throughput**: Reduces per-operation overhead by batching WAL writes
/// - **Write Amplification**: More efficient than individual writes
///
/// ## Example
///
/// ```no_run
/// # use lsm::WriteBatch;
/// let mut batch = WriteBatch::new();
/// batch.put(b"user:1".to_vec(), b"Alice".to_vec());
/// batch.put(b"user:2".to_vec(), b"Bob".to_vec());
/// batch.delete(b"user:3".to_vec());
/// // db.write(batch).await?;
/// ```
#[derive(Debug)]
pub struct WriteBatch {
    /// Ordered list of write operations in this batch.
    pub(crate) writes: Vec<WriteOp>,
}

impl WriteOp {
    /// Operation type constant for Put operations.
    pub(crate) const PUT_OP: u8 = 1;

    /// Operation type constant for Delete operations.
    pub(crate) const DELETE_OP: u8 = 2;

    /// Returns the key associated with this operation.
    pub fn get_key(&self) -> &[u8] {
        match self {
            WriteOp::Put(key, _) => key,
            WriteOp::Delete(key) => key,
        }
    }

    /// Returns the operation type code (PUT_OP or DELETE_OP).
    pub fn get_type(&self) -> u8 {
        match self {
            WriteOp::Put(_, _) => WriteOp::PUT_OP,
            WriteOp::Delete(_) => WriteOp::DELETE_OP,
        }
    }

    /// Returns the length of the key in bytes.
    pub(crate) fn get_key_length(&self) -> u64 {
        self.get_key().len() as u64
    }

    /// Returns the length of the value in bytes (0 for Delete operations).
    #[allow(dead_code)]
    pub(crate) fn get_value_length(&self) -> u64 {
        match self {
            WriteOp::Put(_, value) => value.len() as u64,
            WriteOp::Delete(_) => 0u64,
        }
    }

    /// Applies this operation to the given memtable.
    ///
    /// Called internally when a batch is committed to write operations to the active memtable.
    pub(crate) fn write_to_memtable(self, memtable: &mut Memtable) {
        match self {
            WriteOp::Put(key, value) => {
                memtable.put(key, value);
            }
            WriteOp::Delete(key) => {
                memtable.delete(key);
            }
        }
    }
}

impl WriteBatch {
    /// Creates a new empty write batch.
    pub fn new() -> Self {
        Self { writes: Vec::new() }
    }

    /// Adds a Put operation to the batch.
    ///
    /// The operation will not be applied to the database until this batch is passed to
    /// `Database::write()`.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to insert or update
    /// * `value` - The value to associate with the key
    pub fn put(&mut self, key: Key, value: Value) {
        self.writes.push(WriteOp::Put(key, value));
    }

    /// Adds a Delete operation to the batch.
    ///
    /// Creates a tombstone marker for the key. The deletion will not be applied to the
    /// database until this batch is passed to `Database::write()`.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to delete
    pub fn delete(&mut self, key: Key) {
        self.writes.push(WriteOp::Delete(key));
    }
}

impl Default for WriteBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Allows specifying details of a write
#[derive(Debug, Clone)]
pub struct WriteOptions {
    /// Should the call block until it is guaranteed to be written to disk?
    pub flush: bool,
}

/// Creates a new `WriteOptions` instance with default settings.
impl WriteOptions {
    /// Creates a new `WriteOptions` with default values.
    pub const fn new() -> Self {
        Self { flush: true }
    }
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self::new()
    }
}
