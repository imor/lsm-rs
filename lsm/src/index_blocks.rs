//! Index blocks for sorted table metadata and lookups.
//!
//! Each sorted table has exactly one index block that stores metadata and enables efficient
//! lookups. The index block maps key ranges to data block IDs, allowing binary search to
//! quickly identify which data block might contain a given key.
//!
//! ## Structure
//!
//! An index block consists of:
//! 1. **Header**: Metadata including table size, min/max key lengths, and number of data blocks
//! 2. **Min key bytes**: The smallest key in the entire table
//! 3. **Max key bytes**: The largest key in the entire table
//! 4. **Offset list**: Array of byte offsets to each index entry (padded for alignment)
//! 5. **Index entries**: Each entry contains a block ID and the first key in that block
//!
//! ## Binary Search
//!
//! The index enables efficient key lookups using binary search over the sorted index entries.
//! This identifies the data block that might contain the key in O(log n) time.
//!
//! ## Disk Format
//!
//! Index blocks are stored as `idxXXXXXXXX.data` files where X is the zero-padded table ID.

use std::cmp::Ordering;
use std::mem::size_of;
use std::path::Path;

use crate::data_blocks::DataBlockId;
use crate::sorted_table::TableId;
use crate::{Error, disk};
use crate::{Key, Params};

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Header at the beginning of an index block.
///
/// Contains essential metadata about the sorted table including size, key range information,
/// and the number of data blocks. This fixed-size structure is followed by the min/max keys
/// and the index entries.
#[derive(Debug, IntoBytes, KnownLayout, Immutable, FromBytes)]
#[repr(C, packed)]
struct IndexBlockHeader {
    /// Total size of the table in bytes (sum of all data block sizes).
    size: u64,
    
    /// Length of the minimum key in bytes.
    min_key_len: u32,
    
    /// Length of the maximum key in bytes.
    max_key_len: u32,
    
    /// Number of data blocks in this table.
    num_data_blocks: u32,
    
    /// Padding for alignment.
    _padding: u32,
}

/// Header for a single entry in the index block.
///
/// Each index entry maps a key (the first key in a data block) to the corresponding data
/// block ID. The entry consists of this fixed-size header followed by the variable-length key.
#[derive(IntoBytes, KnownLayout, Immutable, FromBytes)]
#[repr(C, packed)]
struct IndexEntryHeader {
    /// Unique identifier of the data block this entry points to.
    block_id: DataBlockId,
    
    /// Length of the key in bytes.
    key_len: u32,
    
    /// Padding for alignment.
    _padding: u32,
}

/// Index block for a sorted table.
///
/// An index block stores metadata and provides efficient lookup capabilities for a sorted table.
/// It maps key ranges to data block IDs, enabling binary search to quickly identify which data
/// block might contain a given key.
///
/// ## Layout
///
/// The index block is stored as a single byte buffer with the following structure:
/// 1. **Header** (fixed size): Metadata about the table
/// 2. **Min key** (variable): The smallest key in the table
/// 3. **Max key** (variable): The largest key in the table
/// 4. **Offset list** (padded): Array of u32 offsets pointing to each index entry
/// 5. **Index entries** (variable): Each entry contains a block ID and the first key in that block
///
/// ## Lookup Process
///
/// 1. Check if the key is within the table's min/max range
/// 2. Binary search the index entries to find the data block
/// 3. Return the block ID that might contain the key
///
/// ## Persistence
///
/// Index blocks are stored on disk as `idxXXXXXXXX.data` files and loaded on demand.
pub struct IndexBlock {
    /// Complete serialized index block data including header, keys, offsets, and entries.
    data: Vec<u8>,
}

impl IndexBlock {
    /// Creates a new index block and writes it to disk.
    ///
    /// Constructs the index block by serializing the header, min/max keys, offset list, and
    /// index entries into a single byte buffer, then writes it to disk.
    ///
    /// # Arguments
    ///
    /// * `params` - Database configuration parameters
    /// * `id` - Unique identifier for this table
    /// * `index` - Vector of (first_key, block_id) pairs for each data block
    /// * `size` - Total size of all data blocks in bytes
    /// * `min_key` - The smallest key in the entire table
    /// * `max_key` - The largest key in the entire table
    ///
    /// # Returns
    ///
    /// A new `IndexBlock` with the serialized data.
    ///
    /// # Errors
    ///
    /// Returns an error if the index block cannot be written to disk.
    pub async fn new(
        params: &Params,
        id: TableId,
        index: Vec<(Key, DataBlockId)>,
        size: u64,
        min_key: Key,
        max_key: Key,
    ) -> Result<Self, Error> {
        let header = IndexBlockHeader {
            size,
            min_key_len: min_key.len() as u32,
            max_key_len: max_key.len() as u32,
            num_data_blocks: index.len() as u32,
            _padding: 0,
        };

        let mut block_data = header.as_bytes().to_vec();
        block_data.extend_from_slice(&min_key);
        block_data.extend_from_slice(&max_key);

        crate::add_padding(&mut block_data);

        // Reserve space for offsets
        let offset_start = block_data.len();
        let offset_len = crate::pad_offset(index.len());
        block_data.append(&mut vec![0u8; offset_len * size_of::<u32>()]);

        for (pos, (key, block_id)) in index.into_iter().enumerate() {
            let header = IndexEntryHeader {
                block_id,
                key_len: key.len() as u32,
                _padding: 0,
            };

            let entry_offset = block_data.len() as u32;

            block_data[offset_start + pos * size_of::<u32>()
                ..offset_start + (pos + 1) * size_of::<u32>()]
                .copy_from_slice(entry_offset.as_bytes());

            block_data.extend_from_slice(header.as_bytes());
            block_data.extend_from_slice(&key);
        }

        // Store on disk before grabbing the lock
        let fpath = Self::get_file_path(params, &id);
        disk::write(&fpath, &block_data)
            .await
            .map_err(|err| Error::from_io_error("Failed to write index block", err))?;

        Ok(IndexBlock { data: block_data })
    }

    /// Loads an existing index block from disk.
    ///
    /// Reads the index block file for the specified table ID and deserializes it.
    ///
    /// # Arguments
    ///
    /// * `params` - Database configuration parameters
    /// * `id` - Unique identifier of the table to load
    ///
    /// # Returns
    ///
    /// An `IndexBlock` containing the loaded data.
    ///
    /// # Errors
    ///
    /// Returns an error if the index block file cannot be read from disk.
    pub async fn load(params: &Params, id: TableId) -> Result<Self, Error> {
        log::trace!("Loading index block from disk");
        let fpath = Self::get_file_path(params, &id);
        let data = disk::read(&fpath, 0)
            .await
            .map_err(|err| Error::from_io_error("Failed to read index block", err))?;

        Ok(IndexBlock { data })
    }

    /// Returns the filesystem path where an index block is stored.
    ///
    /// Index blocks are stored as `idxXXXXXXXX.data` where X is the zero-padded table ID.
    #[inline]
    fn get_file_path(params: &Params, block_id: &TableId) -> std::path::PathBuf {
        let fname = format!("idx{block_id:08}.data");
        params.db_path.join(Path::new(&fname))
    }

    /// Returns a reference to the index block header.
    ///
    /// The header is located at the beginning of the data buffer and contains metadata
    /// about the table.
    fn get_header(&self) -> &IndexBlockHeader {
        IndexBlockHeader::ref_from_prefix(&self.data[..]).unwrap().0
    }

    /// Returns the byte offset of an index entry within the data buffer.
    ///
    /// Reads the offset from the offset list and returns the position where the index entry
    /// (block ID and key) is located.
    ///
    /// # Arguments
    ///
    /// * `pos` - Index of the entry (0-based)
    ///
    /// # Panics
    ///
    /// Panics if `pos` is greater than or equal to the number of data blocks.
    fn get_entry_offset(&self, pos: usize) -> usize {
        let header = self.get_header();
        assert!((pos as u32) < header.num_data_blocks);

        let offset = size_of::<IndexBlockHeader>()
            + header.min_key_len as usize
            + header.max_key_len as usize;

        let offset_offset = crate::pad_offset(offset) + pos * size_of::<u32>();
        *u32::ref_from_prefix(&self.data[offset_offset..]).unwrap().0 as usize
    }

    /// Returns the data block ID at the specified index position.
    ///
    /// # Arguments
    ///
    /// * `pos` - Index of the data block (0-based)
    ///
    /// # Returns
    ///
    /// The unique identifier of the data block at this position.
    pub fn get_block_id(&self, pos: usize) -> DataBlockId {
        let offset = self.get_entry_offset(pos);

        let entry_header = IndexEntryHeader::ref_from_bytes(
            &self.data[offset..offset + size_of::<IndexEntryHeader>()],
        )
        .unwrap();

        entry_header.block_id
    }

    /// Returns the first key of the data block at the specified index position.
    ///
    /// This is the key used for binary search to identify which block might contain a
    /// given search key.
    ///
    /// # Arguments
    ///
    /// * `pos` - Index of the data block (0-based)
    ///
    /// # Returns
    ///
    /// A byte slice containing the first key in the data block.
    pub fn get_block_key(&self, pos: usize) -> &[u8] {
        let offset = self.get_entry_offset(pos);

        let (entry_header, _) = IndexEntryHeader::ref_from_prefix(&self.data[offset..]).unwrap();

        let key_start = offset + size_of::<IndexEntryHeader>();
        &self.data[key_start..key_start + (entry_header.key_len as usize)]
    }

    /// Returns the total number of data blocks in this table.
    pub fn num_data_blocks(&self) -> usize {
        self.get_header().num_data_blocks as usize
    }

    /// Returns the total size of this table in bytes.
    ///
    /// This is the sum of all data block sizes. For WiscKey mode, this counts only the
    /// key and value reference sizes, not the actual value sizes in the value log.
    pub fn get_size(&self) -> usize {
        self.get_header().size as usize
    }

    /// Returns the smallest key in this table.
    ///
    /// This key is stored in the index block header region and represents the minimum
    /// key across all data blocks.
    pub fn get_min(&self) -> &[u8] {
        let header = self.get_header();
        let key_offset = size_of::<IndexBlockHeader>();

        &self.data[key_offset..key_offset + (header.min_key_len as usize)]
    }

    /// Returns the largest key in this table.
    ///
    /// This key is stored in the index block header region and represents the maximum
    /// key across all data blocks.
    pub fn get_max(&self) -> &[u8] {
        let header = self.get_header();
        let key_offset = size_of::<IndexBlockHeader>() + (header.min_key_len as usize);

        &self.data[key_offset..key_offset + (header.max_key_len as usize)]
    }

    /// Searches for a key and returns the data block ID that might contain it.
    ///
    /// Uses binary search over the index entries to identify which data block could contain
    /// the specified key. First checks if the key is within the table's min/max range.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to search for
    ///
    /// # Returns
    ///
    /// * `Some(DataBlockId)` - The ID of the data block that might contain the key
    /// * `None` - If the key is outside this table's key range (< min or > max)
    ///
    /// # Note
    ///
    /// This method returns the block that *might* contain the key. The caller must still
    /// search within the returned data block to confirm whether the key exists.
    #[tracing::instrument(skip(self, key))]
    pub fn binary_search(&self, key: &[u8]) -> Option<DataBlockId> {
        if key < self.get_min() || key > self.get_max() {
            return None;
        }

        let header = self.get_header();

        let mut start = 0;
        let mut end = (header.num_data_blocks as usize) - 1;

        while end - start > 1 {
            let mid = (end - start) / 2 + start;
            let mid_key = self.get_block_key(mid);

            match mid_key.cmp(key) {
                Ordering::Equal => {
                    return Some(self.get_block_id(mid));
                }
                Ordering::Greater => {
                    end = mid;
                }
                Ordering::Less => {
                    start = mid;
                }
            }
        }

        assert!(key >= self.get_block_key(start));

        if key >= self.get_block_key(end) {
            Some(self.get_block_id(end))
        } else {
            Some(self.get_block_id(start))
        }
    }
}
