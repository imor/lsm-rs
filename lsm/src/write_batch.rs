use crate::{Key, Value, memtable::Memtable};

#[derive(Debug)]
pub enum WriteOp {
    Put(Key, Value),
    Delete(Key),
}

/// A WriteBatch allows to bundle multiple updates together for higher throughput
///
/// Note: The batch will not be applied to the database until it is passed to `Database::write`
#[derive(Debug)]
pub struct WriteBatch {
    pub(crate) writes: Vec<WriteOp>,
}

impl WriteOp {
    pub(crate) const PUT_OP: u8 = 1;
    pub(crate) const DELETE_OP: u8 = 2;

    pub fn get_key(&self) -> &[u8] {
        match self {
            WriteOp::Put(key, _) => key,
            WriteOp::Delete(key) => key,
        }
    }

    pub fn get_type(&self) -> u8 {
        match self {
            WriteOp::Put(_, _) => WriteOp::PUT_OP,
            WriteOp::Delete(_) => WriteOp::DELETE_OP,
        }
    }

    pub(crate) fn get_key_length(&self) -> u64 {
        self.get_key().len() as u64
    }

    #[allow(dead_code)]
    pub(crate) fn get_value_length(&self) -> u64 {
        match self {
            WriteOp::Put(_, value) => value.len() as u64,
            WriteOp::Delete(_) => 0u64,
        }
    }

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
    pub fn new() -> Self {
        Self { writes: Vec::new() }
    }

    /// Record a put operation in the write batch
    /// Will not be applied to the Database until the WriteBatch is written
    pub fn put(&mut self, key: Key, value: Value) {
        self.writes.push(WriteOp::Put(key, value));
    }

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
    pub sync: bool,
}

impl WriteOptions {
    pub const fn new() -> Self {
        Self { sync: true }
    }
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self::new()
    }
}
