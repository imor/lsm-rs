//! In-memory write buffer for the LSM-tree.
//!
//! The memtable is a sorted in-memory data structure that buffers writes before they are
//! flushed to disk as sorted tables. It provides:
//! - Fast writes (O(log n) insertion)
//! - Fast reads for recent data
//! - Sorted iteration over entries
//! - Sequence numbering for MVCC
//!
//! ## Lifecycle
//!
//! 1. **Active memtable**: Receives all new writes
//! 2. **Immutable memtable**: When full, becomes read-only and a new active memtable is created
//! 3. **Flushed**: Immutable memtable is written to Level 0 as a sorted table
//! 4. **Discarded**: After successful flush, the memtable is dropped
//!
//! ## Recovery
//!
//! Memtables can be reconstructed from the write-ahead log (WAL) after a crash.

use std::cmp::Ordering;
use std::sync::Arc;

use async_trait::async_trait;

use crate::data_blocks::DataEntryType;
use crate::manifest::SeqNumber;
use crate::sorted_table::InternalIterator;
use crate::{EntryRef, Key, Params};

#[cfg(feature = "wisckey")]
use crate::values::ValueLog;

/// A reference-counted, mutable reference to a memtable.
///
/// Allows multiple parts of the system to share ownership of the active memtable while
/// maintaining the ability to swap it for a new one when it becomes full.
#[derive(Debug, Clone)]
pub struct MemtableRef {
    inner: Arc<Memtable>,
}

/// A reference-counted, immutable reference to a memtable.
///
/// Created when an active memtable becomes full and needs to be flushed. Prevents
/// modifications while allowing reads and iteration during the flush process.
#[derive(Debug, Clone)]
pub struct ImmMemtableRef {
    inner: Arc<Memtable>,
}

/// An entry in the memtable (either a value or a deletion tombstone).
///
/// Each entry is versioned with a sequence number for MVCC support.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemtableEntry {
    /// A Put operation with its value and sequence number.
    Value { seq_number: u64, value: Vec<u8> },
    
    /// A Delete operation (tombstone) with its sequence number.
    Deletion { seq_number: u64 },
}

/// A reference to an entry in the memtable.
///
/// Provides access to the entry's type and value without exposing the full memtable structure.
/// TODO: Make this zero-copy somehow to avoid cloning.
pub struct MemtableEntryRef {
    entry: MemtableEntry,
}

impl MemtableEntryRef {
    /// Returns the type of this entry (Put or Delete).
    pub fn get_type(&self) -> DataEntryType {
        self.entry.get_type()
    }

    /// Returns the value if this is a Put entry, or `None` if it's a deletion.
    pub fn get_value(&self) -> Option<&[u8]> {
        self.entry.get_value()
    }

    /// Converts this entry reference into an `EntryRef` if it's a Put operation.
    ///
    /// Returns `None` for deletion tombstones.
    pub fn get_entry_ref(self) -> Option<EntryRef> {
        match self.get_type() {
            DataEntryType::Put => {
                let entry = EntryRef::Memtable { entry: self };
                Some(entry)
            }
            DataEntryType::Delete => None,
        }
    }
}

impl MemtableEntry {
    /// Returns the type of this entry (Put or Delete).
    pub fn get_type(&self) -> DataEntryType {
        match &self {
            MemtableEntry::Value { .. } => DataEntryType::Put,
            MemtableEntry::Deletion { .. } => DataEntryType::Delete,
        }
    }

    pub fn get_value(&self) -> Option<&[u8]> {
        match self {
            MemtableEntry::Value { value, .. } => Some(value),
            MemtableEntry::Deletion { .. } => None,
        }
    }
}

/// Iterator for traversing memtable entries in sorted order.
///
/// Supports both forward and reverse iteration over all entries in the memtable.
/// The iterator clones entries during iteration to avoid lifetime issues.
#[derive(Debug)]
pub struct MemtableIterator {
    /// Reference to the memtable being iterated.
    inner: Arc<Memtable>,
    
    /// Index of the next entry to return (-1 for reverse at end, len for forward at end).
    next_index: i64,
    
    /// The current entry's key.
    key: Option<Key>,
    
    /// The current entry's value/deletion.
    entry: Option<MemtableEntry>,
    
    /// If `true`, iterates in reverse; if `false`, iterates forward.
    reverse: bool,
}

impl MemtableIterator {
    /// Creates a new iterator for the memtable.
    ///
    /// Initializes at the first (forward) or last (reverse) entry and loads it.
    ///
    /// # Arguments
    ///
    /// * `inner` - The memtable to iterate over
    /// * `reverse` - If `true`, iterate in reverse; if `false`, iterate forward
    pub async fn new(inner: Arc<Memtable>, reverse: bool) -> Self {
        let next_index = if reverse {
            (inner.entries.len() as i64) - 1
        } else {
            0
        };

        let mut obj = Self {
            inner,
            reverse,
            key: None,
            entry: None,
            next_index,
        };

        obj.step().await;

        obj
    }
}

/// Implements the internal iterator trait for memtable iteration.
#[async_trait]
impl InternalIterator for MemtableIterator {
    #[tracing::instrument]
    /// Advances the iterator to the next entry.
    async fn step(&mut self) {
        let entries = &self.inner.entries;
        let num_entries = entries.len() as i64;

        if self.reverse {
            match self.next_index.cmp(&(-1)) {
                Ordering::Less => {
                    panic!("Cannot step(); already at end");
                }
                Ordering::Equal => {
                    self.next_index -= 1;
                }
                Ordering::Greater => {
                    let (key, entry) = entries[self.next_index as usize].clone();
                    self.key = Some(key);
                    self.entry = Some(entry);
                    self.next_index -= 1;
                }
            }
        } else {
            match self.next_index.cmp(&num_entries) {
                Ordering::Greater => {
                    panic!("Cannot step(); already at end");
                }
                Ordering::Equal => {
                    self.next_index += 1;
                }
                Ordering::Less => {
                    let (key, entry) = entries[self.next_index as usize].clone();
                    self.key = Some(key);
                    self.entry = Some(entry);
                    self.next_index += 1;
                }
            }
        }
    }

    /// Checks if the iterator has reached the end.
    fn at_end(&self) -> bool {
        if self.reverse {
            self.next_index < -1
        } else {
            let len = self.inner.entries.len() as i64;
            self.next_index > len
        }
    }

    /// Returns the current key the iterator is pointing to.
    fn get_key(&self) -> &[u8] {
        self.key.as_ref().expect("Not a valid iterator")
    }

    /// Returns a reference to the current entry the iterator is pointing to.
    #[cfg(feature = "wisckey")]
    async fn get_entry(&self, _value_log: &ValueLog) -> Option<EntryRef> {
        self.entry.clone().map(|entry| EntryRef::Memtable {
            entry: MemtableEntryRef { entry },
        })
    }

    /// Returns a reference to the current entry the iterator is pointing to.
    #[cfg(not(feature = "wisckey"))]
    fn get_entry(&self) -> Option<EntryRef> {
        self.entry.clone().map(|entry| EntryRef::Memtable {
            entry: MemtableEntryRef { entry },
        })
    }

    /// Returns the sequence number of the current entry.
    fn get_seq_number(&self) -> SeqNumber {
        match self.entry.as_ref().unwrap() {
            MemtableEntry::Value { seq_number, .. } | MemtableEntry::Deletion { seq_number } => {
                *seq_number
            }
        }
    }

    /// Returns the type of the current entry (Put or Delete).
    fn get_entry_type(&self) -> DataEntryType {
        self.entry.as_ref().unwrap().get_type()
    }
}

impl ImmMemtableRef {
    /// Returns a reference to the underlying immutable memtable.
    pub fn get(&self) -> &Memtable {
        &self.inner
    }

    /// Creates an iterator over this immutable memtable.
    ///
    /// # Arguments
    ///
    /// * `reverse` - If `true`, iterate in reverse; if `false`, iterate forward
    pub async fn into_iter(self, reverse: bool) -> MemtableIterator {
        MemtableIterator::new(self.inner, reverse).await
    }
}

impl MemtableRef {
    /// Wraps a memtable in a reference-counted reference.
    pub fn wrap(inner: Memtable) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Creates an immutable reference to this memtable.
    ///
    /// Allows multiple readers while the memtable is being flushed.
    pub fn clone_immutable(&self) -> ImmMemtableRef {
        ImmMemtableRef {
            inner: self.inner.clone(),
        }
    }

    /// Swaps the current memtable for a new empty one and returns the old one as immutable.
    ///
    /// Called when the active memtable becomes full and needs to be flushed.
    ///
    /// # Arguments
    ///
    /// * `next_seq_number` - The starting sequence number for the new memtable
    pub fn take(&mut self, next_seq_number: u64) -> ImmMemtableRef {
        let mut inner = Arc::new(Memtable::new(next_seq_number));
        std::mem::swap(&mut inner, &mut self.inner);

        ImmMemtableRef { inner }
    }

    /// Returns a reference to the underlying memtable.
    pub fn get(&self) -> &Memtable {
        &self.inner
    }

    /// Returns a mutable reference to the inner memtable.
    ///
    /// This is only safe to call from DbLogic while holding the memtable lock.
    ///
    /// # Panics
    ///
    /// Panics if there are multiple references to the memtable, which should not happen
    /// when called with proper locking.
    pub(crate) fn get_mut(&mut self) -> &mut Memtable {
        Arc::get_mut(&mut self.inner).expect("Multiple references to memtable exist - this should not happen when called with proper locking")
    }
}

/// In-memory sorted buffer for writes not yet flushed to disk.
///
/// The memtable stores key-value pairs in sorted order and can be reconstructed from the
/// write-ahead log (WAL) after a crash. It provides fast writes and reads for recent data.
///
/// ## Structure
///
/// - Entries are stored in a sorted vector for binary search lookups
/// - Each entry has a sequence number for MVCC
/// - Size tracks memory usage to determine when to flush
///
/// ## Sequence Numbers
///
/// Sequence numbers increment monotonically with each operation, providing a total order
/// over all writes and enabling snapshot isolation.
#[derive(Debug)]
pub struct Memtable {
    /// Sorted list of (key, entry) pairs.
    entries: Vec<(Vec<u8>, MemtableEntry)>,
    
    /// Total size in bytes (sum of key and value lengths).
    size: usize,
    
    /// The next sequence number to assign to a write operation.
    next_seq_number: SeqNumber,
}

impl Memtable {
    /// Creates a new empty memtable.
    ///
    /// # Arguments
    ///
    /// * `next_seq_number` - The starting sequence number for this memtable
    pub fn new(next_seq_number: SeqNumber) -> Self {
        let entries = Vec::new();
        let size = 0;

        Self {
            entries,
            size,
            next_seq_number,
        }
    }

    /// Returns the next sequence number that will be assigned.
    #[inline]
    pub fn get_next_seq_number(&self) -> u64 {
        self.next_seq_number
    }

    /// Returns the minimum and maximum keys in this memtable.
    ///
    /// # Panics
    ///
    /// Panics if the memtable is empty.
    pub fn get_min_max_key(&self) -> (&[u8], &[u8]) {
        let len = self.entries.len();

        if len == 0 {
            panic!("Memtable is empty");
        }

        (&self.entries[0].0, &self.entries[len - 1].0)
    }

    /// Looks up a key in the memtable.
    ///
    /// Uses binary search to efficiently find the key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to search for
    ///
    /// # Returns
    ///
    /// * `Some(MemtableEntryRef)` - If the key exists
    /// * `None` - If the key is not found
    #[tracing::instrument(skip(self, key))]
    pub fn get(&self, key: &[u8]) -> Option<MemtableEntryRef> {
        match self
            .entries
            .binary_search_by_key(&key, |(key, _entry)| key.as_slice())
        {
            Ok(pos) => Some(MemtableEntryRef {
                entry: self.entries[pos].1.clone(),
            }),
            Err(_) => None,
        }
    }

    /// Finds the position where a key should be inserted.
    ///
    /// If the key already exists, removes the old entry and returns its position.
    /// This ensures keys are unique and newer writes override older ones.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to find the position for
    ///
    /// # Returns
    ///
    /// The index where the key should be inserted.
    fn get_key_pos(&mut self, key: &[u8]) -> usize {
        match self
            .entries
            .binary_search_by_key(&key, |(key, _entry)| key.as_slice())
        {
            Ok(pos) => {
                // remove old entry
                let entry_len = {
                    let (_, entry) = self.entries.remove(pos);
                    match entry {
                        MemtableEntry::Value { value, .. } => key.len() + value.len(),
                        MemtableEntry::Deletion { .. } => key.len(),
                    }
                };

                self.size -= entry_len;
                pos
            }
            Err(pos) => pos,
        }
    }

    /// Inserts or updates a key-value pair in the memtable.
    ///
    /// If the key already exists, the old entry is replaced. The operation is assigned
    /// a new sequence number and the memtable size is updated.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to insert or update
    /// * `value` - The value to associate with the key
    #[tracing::instrument(skip(self, key, value))]
    pub fn put(&mut self, key: Vec<u8>, value: Vec<u8>) {
        let pos = self.get_key_pos(key.as_slice());
        let entry_len = key.len() + value.len();

        self.entries.insert(
            pos,
            (
                key,
                MemtableEntry::Value {
                    value,
                    seq_number: self.next_seq_number,
                },
            ),
        );

        self.size += entry_len;
        self.next_seq_number += 1;
    }

    /// Marks a key as deleted by inserting a tombstone.
    ///
    /// If the key already exists, the old entry is replaced with a deletion tombstone.
    /// The operation is assigned a new sequence number.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to delete
    #[tracing::instrument(skip(self, key))]
    pub fn delete(&mut self, key: Vec<u8>) {
        let pos = self.get_key_pos(key.as_slice());
        let entry_len = key.len();

        self.entries.insert(
            pos,
            (
                key,
                MemtableEntry::Deletion {
                    seq_number: self.next_seq_number,
                },
            ),
        );

        self.size += entry_len;
        self.next_seq_number += 1;
    }

    /// Checks if the memtable has reached its size limit.
    ///
    /// When `true`, the memtable should be made immutable and flushed to disk.
    ///
    /// # Arguments
    ///
    /// * `params` - Database parameters containing the size limit
    #[inline]
    pub fn is_full(&self, params: &Params) -> bool {
        self.size >= params.max_memtable_size
    }

    /// Returns a copy of all entries in the memtable.
    ///
    /// FIXME: Avoid this copy somehow without breaking seek consistency.
    pub fn get_entries(&self) -> Vec<(Key, MemtableEntry)> {
        self.entries.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_put() {
        let mut mem = Memtable::new(1);

        let key1 = vec![5, 2, 4];
        let key2 = vec![3, 8, 1];

        let val1 = vec![5, 1];
        let val2 = vec![1, 8];

        mem.put(key1.clone(), val1.clone());
        mem.put(key2.clone(), val2.clone());

        assert_eq!(mem.get(&key1).unwrap().get_value().unwrap(), &val1);
        assert_eq!(mem.get(&key2).unwrap().get_value().unwrap(), &val2);
    }

    #[test]
    fn delete() {
        let mut mem = Memtable::new(1);

        assert_eq!(mem.entries.len(), 0);

        let key = vec![5, 2, 4];
        let val = vec![5, 1];

        mem.put(key.clone(), val.clone());
        mem.delete(key.clone());

        assert_eq!(mem.entries.len(), 1);
        assert_eq!(mem.get(&key).unwrap().get_value(), None);
    }

    #[test]
    fn override_entry() {
        let mut mem = Memtable::new(1);

        let key1 = vec![5, 2, 4];

        let val1 = vec![5, 1];
        let val2 = vec![1, 8];

        mem.put(key1.clone(), val1.clone());
        mem.put(key1.clone(), val2.clone());

        assert_eq!(mem.get(&key1).unwrap().get_value().unwrap(), &val2);
    }
}
