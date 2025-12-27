//! Value batch storage for WiscKey optimization.
//!
//! This module implements value batches - fixed-size files that store multiple values
//! together. Each batch has a header, offset table, and value entries.
//!
//! ## Layout
//!
//! ```text
//! +------------------+
//! | ValueBatchHeader |
//! +------------------+
//! | Offset Table     | (padded to word boundary)
//! +------------------+
//! | Value Entry 1    |
//! | Value Entry 2    |
//! | ...              |
//! +------------------+
//! ```
//!
//! Each value entry contains: header (key_length, value_length), key bytes, value bytes.

use std::sync::Arc;

use crate::Error;
use crate::disk;
use crate::values::{ValueBatchId, ValueId, ValueLog, ValueOffset, ValueRef};

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// An immutable batch of values stored on disk.
///
/// Contains multiple value entries with an offset table for quick access.
/// The layout is: header, offset table (padded), value entries.
#[derive(Debug)]
pub(super) struct ValueBatch {
    /// Complete serialized batch data (header + offsets + entries).
    data: Vec<u8>,
}

/// Builder for constructing value batches.
///
/// Accumulates values and creates the final batch file when finished.
pub struct ValueBatchBuilder<'a> {
    /// Reference to the value log for file management.
    vlog: &'a ValueLog,

    /// Unique identifier for this batch.
    identifier: ValueBatchId,

    /// Serialized offset table pointing to each value entry.
    offsets: Vec<u8>,

    /// Serialized value entry data (headers + keys + values).
    value_data: Vec<u8>,
}

/// Header at the beginning of a value batch file.
#[derive(Debug, KnownLayout, Immutable, IntoBytes, FromBytes)]
#[repr(C, packed)]
pub(super) struct ValueBatchHeader {
    /// Number of value entries in this batch.
    pub num_values: u32,
}

/// Size of the value batch header in bytes.
pub const BATCH_HEADER_LEN: usize = std::mem::size_of::<ValueBatchHeader>();

/// Header for a single value entry.
#[derive(Debug, KnownLayout, Immutable, IntoBytes, FromBytes)]
#[repr(C, packed)]
pub(super) struct ValueEntryHeader {
    /// Length of the key in bytes.
    pub key_length: u32,

    /// Length of the value in bytes.
    pub value_length: u32,
}

impl<'a> ValueBatchBuilder<'a> {
    /// Creates a new empty value batch builder.
    ///
    /// # Arguments
    ///
    /// * `identifier` - Unique ID for this batch
    /// * `vlog` - Reference to the value log
    pub fn new(identifier: ValueBatchId, vlog: &'a ValueLog) -> Self {
        Self {
            identifier,
            vlog,
            value_data: vec![],
            offsets: vec![],
        }
    }

    /// Adds a key-value pair to this batch.
    ///
    /// The entry is appended with proper padding and an offset is recorded.
    ///
    /// # Arguments
    ///
    /// * `key` - The key bytes (used for GC tracking)
    /// * `val` - The value bytes to store
    ///
    /// # Returns
    ///
    /// A `ValueId` (batch_id, offset) referencing this value.
    ///
    /// # Panics
    ///
    /// Panics if the key or value is too long to fit in a u32.
    pub async fn add_entry(&mut self, key: &[u8], val: &[u8]) -> ValueId {
        // Add padding (if needed)
        let offset = crate::pad_offset(self.value_data.len());
        assert!(offset >= self.value_data.len());
        self.value_data.resize(offset, 0u8);

        self.offsets.extend_from_slice((offset as u32).as_bytes());

        let entry_header = ValueEntryHeader {
            key_length: key.len().try_into().expect("Key is too long"),
            value_length: val.len().try_into().expect("Value is too long"),
        };

        self.value_data.extend_from_slice(entry_header.as_bytes());
        self.value_data.extend_from_slice(key);
        self.value_data.extend_from_slice(val);

        (self.identifier, offset as u32)
    }

    /// Finalizes the batch and writes it to disk.
    ///
    /// This serializes the header, offset table, and value entries into a single
    /// file, caches the batch in memory, and updates the value index.
    ///
    /// # Returns
    ///
    /// The unique identifier for this batch on success.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written or the index update fails.
    pub async fn finish(mut self) -> Result<ValueBatchId, Error> {
        let num_values = (self.offsets.len() / size_of::<u32>()) as u32;

        let header = ValueBatchHeader { num_values };

        crate::add_padding(&mut self.offsets);

        let mut data = header.as_bytes().to_vec();
        let file_path = self.vlog.get_batch_file_path(&self.identifier);

        data.extend_from_slice(&self.value_data);
        disk::write(&file_path, &data)
            .await
            .map_err(|err| Error::from_io_error("Failed to write value log batch", err))?;

        let batch = Arc::new(ValueBatch { data });

        // Store in the cache so we don't have to load immediately
        {
            let shard_id = ValueLog::batch_to_shard_id(self.identifier);
            let mut shard = self.vlog.batch_caches[shard_id].lock().await;
            shard.put(self.identifier, batch);
        }

        self.vlog
            .index
            .add_batch(self.identifier, num_values as usize)
            .await?;

        log::trace!("Created value batch #{}", self.identifier);
        Ok(self.identifier)
    }
}

impl ValueBatch {
    /// Creates a value batch from existing serialized data.
    ///
    /// Used when loading a batch from disk. The data should contain
    /// a valid header, offset table, and value entries.
    ///
    /// # Arguments
    ///
    /// * `data` - Complete serialized batch data
    pub fn from_existing(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Creates a reference to a value at the given offset.
    ///
    /// This provides zero-copy access to a value within the batch
    /// by creating a `ValueRef` that points to the value's location.
    ///
    /// # Arguments
    ///
    /// * `self_ptr` - Arc reference to the batch
    /// * `pos` - Byte offset within the batch where the value entry starts
    ///
    /// # Returns
    ///
    /// A `ValueRef` that can be used to access the value bytes without copying.
    pub fn get_ref(self_ptr: Arc<ValueBatch>, pos: ValueOffset) -> ValueRef {
        let mut offset = pos as usize;
        let data = &self_ptr.get_value_data()[offset..];

        let (vheader, _) = ValueEntryHeader::ref_from_prefix(data).unwrap();

        offset += size_of::<ValueEntryHeader>();
        offset += vheader.key_length as usize;

        ValueRef {
            length: vheader.value_length as usize,
            batch: self_ptr,
            offset,
        }
    }

    /// Retrieves multiple key-value pairs from this batch.
    ///
    /// Used during garbage collection to extract live entries that need
    /// to be moved to new batches.
    ///
    /// # Arguments
    ///
    /// * `offsets` - List of byte offsets pointing to value entries
    ///
    /// # Returns
    ///
    /// Vector of (key, value) pairs with data copied into owned buffers.
    pub fn get_entries(&self, offsets: &[ValueOffset]) -> Vec<(Vec<u8>, Vec<u8>)> {
        offsets
            .iter()
            .map(|offset| {
                let mut offset = *offset as usize;
                let data = &self.get_value_data()[offset..];

                let (vheader, _) = ValueEntryHeader::ref_from_prefix(data).unwrap();

                offset += size_of::<ValueEntryHeader>();

                let key = data[offset..(vheader.key_length as usize)].to_vec();
                offset += vheader.key_length as usize;

                let value = data[offset..(vheader.value_length as usize)].to_vec();

                (key, value)
            })
            .collect()
    }

    /// Returns the raw value entry data, excluding the header.
    ///
    /// The returned slice contains the offset table and all value entries
    /// (headers + keys + values), but excludes the batch header.
    #[inline]
    pub(super) fn get_value_data(&self) -> &[u8] {
        &self.data[BATCH_HEADER_LEN..]
    }

    /// Returns a reference to this batch's header.
    ///
    /// The header contains metadata like the number of values in the batch.
    #[inline]
    fn get_header(&self) -> &ValueBatchHeader {
        ValueBatchHeader::ref_from_prefix(&self.data[..]).unwrap().0
    }

    /// Returns the total number of values originally stored in this batch.
    ///
    /// This count includes values that may have been deleted or garbage collected.
    /// For the current number of live values, use the value index instead.
    #[allow(dead_code)]
    pub fn total_num_values(&self) -> u32 {
        self.get_header().num_values
    }
}
