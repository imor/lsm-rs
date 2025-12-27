//! Data Block Implementation for LSM-Tree SSTables
//!
//! This module provides the core data structure for storing sorted key-value entries in
//! LSM-tree sorted tables (SSTables). Data blocks are the fundamental storage units that
//! hold actual key-value data on disk with efficient compression and fast lookup capabilities.
//!
//! # Overview
//!
//! Data blocks are immutable, fixed-size units that store multiple key-value entries using
//! prefix compression to minimize storage overhead. Each block is self-contained with:
//! - A header containing metadata and optional bloom filter
//! - Variable-length entries with prefix-compressed keys
//! - A restart list enabling efficient binary search
//!
//! # Key Features
//!
//! ## Prefix Compression
//!
//! Consecutive sorted keys often share common prefixes. Data blocks exploit this by storing
//! only the unique suffix of each key along with the length of the shared prefix from the
//! previous key. This significantly reduces storage requirements for keys with common patterns.
//!
//! **Example:**
//! ```text
//! Key 1: "user:alice:email"    → stored as full key
//! Key 2: "user:alice:name"     → prefix_len=11, suffix="name"
//! Key 3: "user:bob:email"      → prefix_len=5, suffix="bob:email"
//! ```
//!
//! ## Restart List
//!
//! While prefix compression saves space, it requires sequential reading to reconstruct keys.
//! The restart list solves this by creating periodic "restart points" where full keys are
//! stored (no compression), enabling efficient binary search.
//!
//! **How it works:**
//! 1. Every N entries (restart_interval), a full key is stored
//! 2. The byte offset of each restart entry is recorded in the restart list
//! 3. Binary search on the restart list narrows down to a small range
//! 4. Sequential scan within that range finds the exact key
//!
//! This provides O(log R) + O(N) lookup time where R is the number of restart points
//! and N is the restart interval, which is much faster than scanning the entire block.
//!
//! ## Bloom Filters (Optional)
//!
//! When the `bloom-filters` feature is enabled, each block includes a probabilistic bloom
//! filter in its header. This allows extremely fast negative lookups - if the bloom filter
//! says a key is not present, we can skip searching the block entirely without any disk I/O.
//!
//! # On-Disk Layout
//!
//! ```text
//! +------------------+
//! | DataBlockHeader  |  (fixed size)
//! |  - restart_list  |  (offset to restart list)
//! |  - num_entries   |  (total entry count)
//! |  - bloom_filter  |  (optional, 1056 bytes)
//! +------------------+
//! | Entry 1          |  (variable size)
//! |  - EntryHeader   |
//! |  - key suffix    |
//! |  - value/ref     |
//! +------------------+
//! | Entry 2          |
//! | ...              |
//! +------------------+
//! | Restart List     |  (array of u32 offsets)
//! +------------------+
//! ```
//!
//! # Entry Format
//!
//! Each entry consists of an `EntryHeader` followed by variable-length data:
//!
//! **Header Fields:**
//! - `prefix_len` (4 bytes) - bytes to reuse from previous key
//! - `suffix_len` (4 bytes) - length of unique key suffix
//! - `entry_type` (1 byte) - operation type (Put/Delete)
//! - `seq_number` (8 bytes) - version number for MVCC
//! - Value location (WiscKey mode) or value length (vanilla mode)
//!
//! **Variable Data:**
//! - Key suffix (suffix_len bytes)
//! - Value data (vanilla mode only) or omitted (WiscKey mode)
//!
//! # Search Algorithm
//!
//! Block lookups use a three-phase approach:
//!
//! 1. **Bloom filter check** (if enabled): Quick negative lookup
//!    - If bloom filter says "not present", return None immediately
//!    - If it says "maybe present", continue to phase 2
//!
//! 2. **Binary search on restart list**: Narrow down to a range
//!    - Compare target key with keys at restart points only
//!    - Find the restart point just before where the key would be
//!    - Results in a range of at most restart_interval entries
//!
//! 3. **Sequential scan**: Find exact match
//!    - Start from the identified restart point
//!    - Reconstruct each key using prefix compression
//!    - Compare until match found or range exhausted
//!
//! # Thread Safety
//!
//! `DataBlock` instances are immutable after creation and safe to share across threads
//! via `Arc<DataBlock>`. Multiple readers can access the same block concurrently without
//! locking since all operations are read-only.
//!
//! # Usage
//!
//! Data blocks are typically not created directly but through the `DataBlockBuilder`:
//!
//! ```rust,ignore
//! // Reading an existing block
//! let block = Arc::new(DataBlock::new_from_data(data, restart_interval));
//!
//! // Looking up a key
//! if let Some(entry) = DataBlock::get_by_key(&block, b"user:alice") {
//!     let value = entry.get_value();
//! }
//!
//! // Iterating entries
//! for i in 0..block.get_num_entries() {
//!     let (key, entry) = DataBlock::get_entry_at_index(&block, i);
//!     // Process entry...
//! }
//! ```

use std::cmp::Ordering;
use std::sync::Arc;

use crate::Key;

use super::{DataEntry, SearchResult};

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

#[cfg(feature = "bloom-filters")]
use bloomfilter::Bloom;

#[cfg(feature = "wisckey")]
use crate::values::{ValueBatchId, ValueOffset};

#[cfg(feature = "bloom-filters")]
//TODO change the size of this depending on max_key_block_length
pub(super) const BLOOM_LENGTH: usize = 1024;

#[cfg(feature = "bloom-filters")]
pub(super) const BLOOM_ITEM_COUNT: usize = 1024;

#[cfg(feature = "bloom-filters")]
/// Taken from https://github.com/jedisct1/rust-bloom-filter/blob/6b93b922be474998514b696dc84333d6c04ed991/src/bitmap.rs#L5
pub(super) const BLOOM_HEADER_SIZE: usize = 1 + 8 + 4 + 32;

/// The on-disk header at the beginning of a data block.
///
/// This fixed-size header appears at the start of every data block and contains essential
/// metadata for reading and parsing the block's contents. The header is followed by the
/// variable-length entries and the restart list.
///
/// ## Layout of a Data Block on Disk
///
/// 1. 4 bytes marking where the restart list starts
/// 2. 4 bytes indicating the number of entries in this block
/// 3. 1024+32 bytes for the bloom filter (if enabled)
/// 4. Sequence of variable-length entries
/// 5. Variable length restart list (each entry is 4 bytes; so we don't need length information)
///
#[derive(IntoBytes, Immutable, KnownLayout, FromBytes)]
#[repr(C, packed)]
pub(super) struct DataBlockHeader {
    /// Byte offset from the start of the block where the restart list begins.
    pub(super) restart_list_start: u32,

    /// Total number of key-value entries stored in this block.
    pub(super) number_of_entries: u32,

    /// Serialized bloom filter for fast key existence checks. Only present when the
    /// "bloom-filters" feature is enabled.
    #[cfg(feature = "bloom-filters")]
    pub(super) bloom_filter: [u8; BLOOM_LENGTH + BLOOM_HEADER_SIZE],
}

/// The header of a single key-value entry within a data block.
///
/// Each entry in a data block consists of this fixed-size header followed by variable-length
/// data (key suffix and optionally value). The header contains metadata needed to reconstruct
/// the full key and locate the value using prefix compression.
///
/// ## Entry Format
///
/// Entries use prefix compression to save space. The `prefix_len` field indicates how many
/// bytes to reuse from the *previous entry's uncompressed key*, while `suffix_len` specifies
/// the length of the unique suffix stored in this entry. To reconstruct the full key:
/// `key = previous_key[0..prefix_len] + current_entry_suffix`.
///
/// ### With WiscKey Feature
///
/// **Header:**
/// - Key prefix len (4 bytes) - number of bytes to reuse from previous uncompressed key
/// - Key suffix len (4 bytes) - length of the key suffix unique to this entry
/// - Seq_number (8 bytes)
/// - Entry type (1 byte)
/// - Value reference (batch id and offset)
///
/// **Content (not part of the header):**
/// - Variable length key suffix
///
/// ### Without WiscKey Feature
///
/// An entry is variable length and contains the following:
///
/// **Header:**
/// - Key prefix len (4 bytes) - number of bytes to reuse from previous uncompressed key
/// - Key suffix len (4 bytes) - length of the key suffix unique to this entry
/// - Value length (8 bytes)
/// - Entry Type (1 byte)
/// - Sequence number (8 bytes)
///
/// **Content (not part of the header):**
/// - Variable length key suffix
/// - Variable length value
#[derive(IntoBytes, Immutable, FromBytes, KnownLayout)]
#[repr(C, packed)]
pub(super) struct EntryHeader {
    /// Number of bytes to reuse from the previous entry's uncompressed key.
    pub(super) prefix_len: u32,

    /// Length of the key suffix unique to this entry (stored after the header).
    pub(super) suffix_len: u32,

    /// Type of entry (e.g., insert, delete/tombstone).
    pub(super) entry_type: u8,

    /// Sequence number for versioning. Higher values indicate more recent writes. Used for
    /// MVCC and conflict resolution.
    pub(super) seq_number: u64,

    /// Batch ID where the actual value is stored.
    #[cfg(feature = "wisckey")]
    pub(super) value_batch: ValueBatchId,

    /// Byte offset within the value batch where the value begins.
    #[cfg(feature = "wisckey")]
    pub(super) value_offset: ValueOffset,

    /// Length of the value data stored after the key suffix.
    #[cfg(not(feature = "wisckey"))]
    pub(super) value_length: u64,
}

//TODO support data block layouts without prefixed keys

/// An in-memory representation of a data block from an LSM-tree sorted table (SSTable).
///
/// Data blocks store sorted key-value entries using prefix compression to minimize storage.
/// They support efficient lookups via a restart list that enables binary search within the block.
///
/// ## DataBlock Structure
///
/// The block contains:
/// - A header with metadata (restart list position, entry count, optional bloom filter)
/// - Variable-length entries with prefix-compressed keys
/// - A restart list at the end for efficient binary search
///
/// ## Restart List Explanation
///
/// A restart list is an optimization technique used in LSM-tree data blocks to speed up key lookups.
/// Data blocks use prefix compression to save space - each entry only stores the part of its key
/// that differs from the previous key. This is efficient for storage, but means you must read
/// entries sequentially from the beginning to reconstruct full keys.
///
/// The restart list solves this by creating periodic "restart points" where full keys are stored
/// (no prefix compression), allowing binary search within the block.
///
/// ## How It Works
///
/// **Structure:** The restart list is stored at the end of the data block as an array of byte
/// offsets (each 4 bytes).
///
/// **Restart Interval:** Every N entries (controlled by `restart_interval`), a full key is stored
/// instead of a prefix-compressed one. The offset to this entry is added to the restart list.
///
/// **Search Process:**
///
/// 1. **Binary Search Phase:**
///    - Search restart list to find the restart point just before the target key
///    - Narrows search to a small range
/// 2. **Sequential Scan Phase:**
///    - From the restart point, sequentially scan entries
///    - Reconstruct each key using prefix compression
///    - Compare until target is found or range is exhausted
///
/// **Example:** With `restart_interval = 16`:
///
/// - Entry 0, 16, 32, 48... have full keys and offsets in the restart list
/// - To find a key, binary search finds the closest restart point (e.g., offset for entry 32)
/// - Then sequentially scan entries 32-47 to find the exact match
///
/// This gives you O(log n) performance for the coarse-grained search plus a small
/// O(restart_interval) sequential scan, which is much faster than scanning the entire block.
///
pub struct DataBlock {
    /// Byte offset where the restart list begins within `data`. The restart list is an array
    /// of u32 offsets pointing to entries with full (non-compressed) keys.
    pub(super) restart_list_start: usize,

    /// Total number of key-value entries stored in this block. Used for iteration and bounds
    /// checking.
    pub(super) num_entries: u32,

    /// How many entries between restart points. For example, if set to 16, entries 0, 16, 32,
    /// etc. will have full keys and appear in the restart list. Lower values improve search
    /// speed but increase space overhead.
    pub(super) restart_interval: u32,

    /// The complete block data including header, entries, and restart list. This is the raw
    /// bytes read from disk or created during block building.
    pub(super) data: Vec<u8>,

    /// Optional probabilistic data structure for quick key existence checks. Allows fast
    /// negative lookups (key definitely not present) without scanning entries. Only available
    /// when the "bloom-filters" feature is enabled.
    #[cfg(feature = "bloom-filters")]
    pub(super) bloom_filter: Bloom<[u8]>,
}

impl DataBlock {
    /// Creates a new `DataBlock` from raw block data.
    ///
    /// Parses the header from the provided data to extract metadata like entry count and
    /// restart list position. If the "bloom-filters" feature is enabled, also deserializes
    /// the bloom filter from the header.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw bytes of the complete block including header, entries, and restart list
    /// * `restart_interval` - Number of entries between restart points
    ///
    /// # Panics
    ///
    /// Panics if the data is empty or if the bloom filter cannot be deserialized.
    pub fn new_from_data(data: Vec<u8>, restart_interval: u32) -> Self {
        assert!(!data.is_empty(), "No data?");

        let header = DataBlockHeader::ref_from_bytes(&data[..Self::header_length()]).unwrap();

        #[cfg(feature = "bloom-filters")]
        let bloom_filter = Bloom::from_bytes(header.bloom_filter.as_slice().to_vec())
            .expect("Failed to load bloom filter");

        log::trace!("Created new data block from existing data");

        Self {
            num_entries: header.number_of_entries,
            restart_list_start: header.restart_list_start as usize,
            data,
            restart_interval,
            #[cfg(feature = "bloom-filters")]
            bloom_filter,
        }
    }

    /// Returns the size of the data block header in bytes.
    ///
    /// This is a compile-time constant based on the size of `DataBlockHeader`.
    const fn header_length() -> usize {
        std::mem::size_of::<DataBlockHeader>()
    }

    /// Reads and reconstructs a key-value entry at the specified byte offset.
    ///
    /// This method parses the entry header at the given offset, reconstructs the full key
    /// using prefix compression with the provided `previous_key`, and returns both the key
    /// and a `DataEntry` handle.
    ///
    /// # Arguments
    ///
    /// * `self_ptr` - Arc reference to this DataBlock
    /// * `offset` - Byte offset from the start of entries (after header) where the entry begins
    /// * `previous_key` - The uncompressed key from the previous entry, used for prefix
    ///   compression. Should be empty for restart points.
    ///
    /// # Returns
    ///
    /// A tuple of `(Key, DataEntry)` where the Key is the reconstructed full key and DataEntry
    /// contains metadata about the entry's location and length.
    ///
    /// # Panics
    ///
    /// Panics if the offset is invalid (beyond the restart list start).
    #[tracing::instrument(skip(self_ptr, previous_key))]
    pub fn get_entry_at_offset(
        self_ptr: Arc<DataBlock>,
        offset: u32,
        previous_key: &[u8],
    ) -> (Key, DataEntry) {
        let mut offset = (offset as usize) + Self::header_length();

        let header_len = std::mem::size_of::<EntryHeader>();

        if offset + header_len > self_ptr.restart_list_start {
            panic!("Invalid offset {offset}");
        }

        let header = EntryHeader::ref_from_bytes(&self_ptr.data[offset..offset + header_len])
            .expect("Failed to read entry header");
        let entry_offset = offset;

        offset += std::mem::size_of::<EntryHeader>();

        let kdata = [
            &previous_key[..(header.prefix_len as usize)],
            &self_ptr.data[offset..offset + (header.suffix_len as usize)],
        ]
        .concat();
        offset += header.suffix_len as usize;

        // Move offset to after the entry
        #[cfg(not(feature = "wisckey"))]
        {
            offset += header.value_length as usize;
        }

        let next_offset = offset - Self::header_length();

        let entry = DataEntry {
            block: self_ptr,
            offset: entry_offset,
            len: next_offset as u32,
        };

        (kdata, entry)
    }

    /// Returns the total number of key-value entries stored in this data block.
    ///
    /// This value is read from the block header and includes all entries regardless of
    /// their type (inserts, deletes, etc.).
    pub fn get_num_entries(&self) -> u32 {
        self.num_entries
    }

    /// Retrieves the key-value entry at the specified entry index.
    ///
    /// Uses the restart list to jump to the nearest restart point before the target index,
    /// then sequentially scans forward to the exact entry. This provides efficient random
    /// access without reading the entire block.
    ///
    /// # Arguments
    ///
    /// * `self_ptr` - Arc reference to this DataBlock
    /// * `index` - Entry index (0-based, not a byte offset)
    ///
    /// # Returns
    ///
    /// A tuple of `(Key, DataEntry)` for the entry at the given index.
    #[tracing::instrument(skip(self_ptr))]
    pub fn get_entry_at_index(self_ptr: &Arc<Self>, index: u32) -> (Key, DataEntry) {
        // First, get the closest restart offset
        let restart_pos = index / self_ptr.restart_interval;

        let restart_offset = self_ptr.get_restart_offset(restart_pos);
        let (mut key, mut entry) = Self::get_entry_at_offset(self_ptr.clone(), restart_offset, &[]);

        let mut current_idx = restart_pos * self_ptr.restart_interval;

        while current_idx < index {
            (key, entry) = Self::get_entry_at_offset(self_ptr.clone(), entry.len(), &key);
            current_idx += 1;
        }

        (key, entry)
    }

    /// Returns the size of the entries section in bytes.
    ///
    /// This excludes both the header at the beginning and the restart list at the end,
    /// returning only the size of the variable-length entries themselves.
    pub fn byte_len(&self) -> u32 {
        // "Cut-off" the beginning and end
        let rl_len = self.data.len() - self.restart_list_start;
        (self.data.len() - Self::header_length() - rl_len) as u32
    }

    /// Returns the number of restart points in the restart list.
    ///
    /// Each restart point is a 4-byte offset, so this divides the restart list byte size
    /// by 4 to get the count of restart entries.
    #[inline(always)]
    fn restart_list_len(&self) -> usize {
        let offset_len = std::mem::size_of::<u32>();
        let rl_len = self.data.len() - self.restart_list_start;

        assert!(rl_len.is_multiple_of(offset_len));
        rl_len / offset_len
    }

    /// Returns the byte offset for a specific restart point.
    ///
    /// Reads the offset value from the restart list at the given position and adjusts it
    /// to be relative to the start of entries (excluding the header).
    ///
    /// # Arguments
    ///
    /// * `pos` - Index into the restart list (0-based)
    ///
    /// # Returns
    ///
    /// Byte offset from the start of entries where the restart entry begins.
    #[inline(always)]
    fn get_restart_offset(&self, pos: u32) -> u32 {
        let offset_len = std::mem::size_of::<u32>();
        let pos = self.restart_list_start + (pos as usize) * offset_len;

        u32::read_from_bytes(&self.data[pos..pos + offset_len]).unwrap()
            - Self::header_length() as u32
    }

    /// Performs binary search on the restart list to locate a key or narrow down its range.
    ///
    /// Uses the restart list to efficiently search for a key without reading all entries.
    /// Each comparison is done at a restart point (full key, no prefix compression).
    ///
    /// # Arguments
    ///
    /// * `self_ptr` - Arc reference to this DataBlock
    /// * `key` - The key to search for
    ///
    /// # Returns
    ///
    /// * `SearchResult::ExactMatch(entry)` - If the key is found at a restart point
    /// * `SearchResult::Range(start, end)` - Byte offset range where the key might be found,
    ///   requiring a sequential scan to confirm
    #[tracing::instrument(skip(self_ptr, key))]
    fn binary_search(self_ptr: &Arc<Self>, key: &[u8]) -> SearchResult {
        let rl_len = self_ptr.restart_list_len();

        let mut start: u32 = 0;
        let mut end = (rl_len as u32) - 1;

        // binary search
        while end - start > 1 {
            let mid = start + (end - start) / 2;

            // We always perform the search at the restart positions for efficiency
            let offset = self_ptr.get_restart_offset(mid);
            let (this_key, entry) = Self::get_entry_at_offset(self_ptr.clone(), offset, &[]);

            match this_key.as_slice().cmp(key) {
                Ordering::Equal => {
                    // Exact match
                    return SearchResult::ExactMatch(entry);
                }
                Ordering::Less => {
                    // continue with right half
                    start = mid;
                }
                Ordering::Greater => {
                    // continue with left half
                    end = mid;
                }
            }
        }

        // There is no reset at the very end so we need to include
        // that part in the sequential search
        let end = if end + 1 == rl_len as u32 {
            self_ptr.byte_len()
        } else {
            self_ptr.get_restart_offset(end)
        };

        SearchResult::Range(start, end)
    }

    /// Searches for and returns the entry with the specified key.
    ///
    /// Uses a two-phase search strategy:
    /// 1. **Bloom filter check** (if enabled): Quick negative lookup to avoid unnecessary searches
    /// 2. **Binary search**: Narrows down to a range using the restart list
    /// 3. **Sequential scan**: Scans entries in the range to find the exact key
    ///
    /// # Arguments
    ///
    /// * `self_ptr` - Arc reference to this DataBlock
    /// * `key` - The key to search for
    ///
    /// # Returns
    ///
    /// * `Some(DataEntry)` - If the key is found in this block
    /// * `None` - If the key does not exist in this block
    #[tracing::instrument(skip(self_ptr, key))]
    pub fn get_by_key(self_ptr: &Arc<Self>, key: &[u8]) -> Option<DataEntry> {
        #[cfg(feature = "bloom-filters")]
        if !self_ptr.bloom_filter.check(key) {
            return None;
        }

        let (start, end) = match Self::binary_search(self_ptr, key) {
            SearchResult::ExactMatch(entry) => {
                return Some(entry);
            }
            SearchResult::Range(start, end) => (start, end),
        };

        let mut pos = self_ptr.get_restart_offset(start);

        let mut last_key = vec![];
        while pos < end {
            let (this_key, entry) = Self::get_entry_at_offset(self_ptr.clone(), pos, &last_key);

            if key == this_key {
                return Some(entry);
            }

            pos = entry.len();
            last_key = this_key;
        }

        // Not found
        None
    }
}
