//! Value index for tracking live data in value batches.
//!
//! The value index is a bitmap-based structure that tracks which values in the value log
//! are still live (not deleted or compacted away). This enables efficient garbage collection
//! by identifying batches with low live data ratios.
//!
//! ## Structure
//!
//! - **Index pages**: Fixed-size (4KB) chunks covering multiple batches
//! - **Bitmaps**: One bit per value entry indicating if it's live
//! - **Batch states**: Track if a batch is Active, Compacted, or Deleted
//!
//! ## Garbage Collection
//!
//! When a batch's live ratio falls below the threshold, it's eligible for GC:
//! 1. Live values are copied to new batches
//! 2. The old batch is marked as Deleted
//! 3. Index page is updated and persisted
//!
//! ## Concurrency
//!
//! Uses RwLock for concurrent reads (common) and exclusive writes (GC operations).

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

use bitvec::vec::BitVec;

use tokio::sync::{RwLock, RwLockReadGuard};

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::manifest::Manifest;
use crate::values::ValueId;
use crate::{Error, Params, disk};

use super::{MIN_VALUE_BATCH_ID, ValueBatchId};

/// Unique identifier for an index page.
pub type ValueIndexPageId = u64;

/// Minimum valid index page identifier.
pub const MIN_VALUE_INDEX_PAGE_ID: ValueIndexPageId = 1;

/// Header at the beginning of an index page.
#[derive(KnownLayout, Immutable, IntoBytes, FromBytes)]
#[repr(C, align(8))]
struct IndexPageHeader {
    /// Unique identifier for this page.
    identifier: ValueIndexPageId,

    /// ID of the first batch tracked by this page.
    start_batch: ValueBatchId,

    /// Number of batches tracked by this page.
    num_batches: u64,

    /// Total number of value entries across all batches.
    num_entries: u64,
}

/// State of a value batch for garbage collection tracking.
#[derive(KnownLayout, Immutable, IntoBytes, PartialEq, Eq, Copy, Clone)]
#[repr(u8)]
enum ValueBatchState {
    /// Batch is active and contains live data.
    Active = 0,

    /// Batch has been compacted (live values moved elsewhere).
    Compacted = 1,

    /// Batch has been deleted from disk.
    Deleted = 2,
}

impl TryFrom<u8> for ValueBatchState {
    type Error = ();

    fn try_from(val: u8) -> Result<Self, ()> {
        for option in [Self::Active, Self::Compacted, Self::Deleted] {
            if val == option as u8 {
                return Ok(option);
            }
        }

        Err(())
    }
}

/// A 4KB chunk of the value index tracking value liveness.
///
/// Each page covers multiple value batches and maintains:
/// - A bitmap indicating which values are live
/// - Offsets pointing to each batch's position in the bitmap
/// - State tracking (Active/Compacted/Deleted) for each batch
///
/// ## Invariants
///
/// - `offsets.len() == batches.len()`
/// - Once sealed, no new batches can be added
/// - Dirty flag indicates unsaved changes
struct IndexPage {
    /// Metadata header for this page.
    header: IndexPageHeader,

    /// True if changes haven't been written to disk yet.
    dirty: bool,

    /// Starting position in the bitmap for each batch.
    offsets: Vec<u16>,

    /// State of each batch (Active, Compacted, or Deleted).
    batches: Vec<ValueBatchState>,

    /// Bitmap tracking live values (one bit per value entry).
    entries: BitVec<u8>,

    /// Once sealed, no new batches can be added (not stored on disk).
    sealed: bool,
}

impl IndexPage {
    /// Creates a new empty index page.
    ///
    /// # Arguments
    ///
    /// * `identifier` - Unique ID for this page
    /// * `start_batch` - The first batch ID that this page will track
    pub fn new(identifier: ValueIndexPageId, start_batch: ValueBatchId) -> Self {
        log::trace!("Creating new value index page with id={identifier}");

        Self {
            header: IndexPageHeader {
                identifier,
                start_batch,
                num_batches: 0,
                num_entries: 0,
            },
            dirty: true,
            sealed: false,
            offsets: Default::default(),
            batches: Default::default(),
            entries: Default::default(),
        }
    }

    /// Loads an index page from disk.
    ///
    /// Deserializes the page header, offset table, batch states, and bitmap
    /// from the specified file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the index page file
    /// * `sealed` - Whether this page should be marked as sealed (no more batches can be added)
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or contains invalid data.
    pub async fn open(path: &Path, sealed: bool) -> Result<Self, Error> {
        let data = disk::read(path, 0).await.map_err(|err| {
            Error::from_io_error(
                format!("Failed to read value index page at `{path:?}`"),
                err,
            )
        })?;

        let (header, ref mut data) = IndexPageHeader::read_from_prefix(&data).unwrap();

        log::trace!(
            "Opening existing value index page with id={} path={path:?}",
            header.identifier
        );

        let mut offsets = Vec::with_capacity(header.num_batches as usize);
        for _ in 0..header.num_batches {
            let (entry, next) = u16::read_from_prefix(data).unwrap();
            offsets.push(entry);
            *data = next;
        }

        let len = header.num_batches as usize;
        let batches: Vec<ValueBatchState> = data[..len]
            .iter()
            .map(|s| (*s).try_into().expect("Invalid state"))
            .collect();
        let entries = BitVec::from_slice(&data[len..]);

        Ok(Self {
            header,
            offsets,
            entries,
            batches,
            sealed,
            dirty: false,
        })
    }

    /// Returns the unique identifier for this index page.
    pub fn get_identifier(&self) -> ValueIndexPageId {
        self.header.identifier
    }

    /// Adds a new value batch to this page.
    ///
    /// Attempts to allocate space in this page for tracking the specified number
    /// of value entries. If successful, initializes the bitmap with all entries
    /// marked as live.
    ///
    /// # Arguments
    ///
    /// * `num_entries` - Number of value entries in the batch to track
    ///
    /// # Returns
    ///
    /// `true` if there was enough space and the batch was added, `false` otherwise.
    /// When false is returned, the page is marked as sealed.
    ///
    /// # Note
    ///
    /// Changes are not persisted until `sync()` is called.
    pub fn expand(&mut self, num_entries: usize) -> bool {
        assert!(!self.sealed);

        const MAX_SIZE: usize = 4 * 1024;
        assert!(num_entries < MAX_SIZE);

        let current_entries = self.entries.len();
        let new_size =
            (self.offsets.len() + 1) * std::mem::size_of::<u16>() + (current_entries + num_entries);

        // Is there enough space?
        if new_size <= MAX_SIZE {
            log::trace!(
                "Adding {num_entries} entries to value index page #{}",
                self.header.identifier
            );

            self.header.num_batches += 1;

            self.offsets.push(current_entries as u16);
            self.batches.push(ValueBatchState::Active);

            // We assume all entries are in use for a new batch
            self.entries.resize(current_entries + num_entries, true);
            self.dirty = true;

            true
        } else {
            self.sealed = true;
            false
        }
    }

    /// Writes the current state of the page to disk.
    ///
    /// Serializes the header, offset table, batch states, and bitmap to the
    /// specified file. Only writes if the page has been modified (dirty flag).
    ///
    /// # Arguments
    ///
    /// * `path` - File path where the page should be written
    ///
    /// # Returns
    ///
    /// `true` if the page was written, `false` if it was already clean.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub async fn sync(&mut self, path: &Path) -> Result<bool, Error> {
        if !self.dirty {
            return Ok(false);
        }

        log::trace!("Writing with value index page to disk (path={path:?})");

        let mut data = self.header.as_bytes().to_vec();
        data.extend_from_slice(self.offsets.as_bytes());
        data.extend_from_slice(self.batches.as_bytes());
        data.extend_from_slice(self.entries.as_raw_slice());

        disk::write(path, &data).await.map_err(|err| {
            Error::from_io_error(
                format!("Failed to write value index page at `{path:?}`"),
                err,
            )
        })?;

        self.dirty = false;
        Ok(true)
    }

    /// Checks if any values tracked by this page are still live.
    ///
    /// Used during garbage collection to determine if an entire page
    /// can be deleted.
    ///
    /// # Returns
    ///
    /// `true` if at least one value is still marked as live, `false` otherwise.
    pub fn is_in_use(&self) -> bool {
        for val in self.entries.iter() {
            if *val {
                return true;
            }
        }
        false
    }

    /// Returns whether this page is sealed.
    ///
    /// A sealed page cannot accept new batches and is considered full.
    pub fn is_sealed(&self) -> bool {
        self.sealed
    }

    /// Counts the number of live entries in the specified batch.
    ///
    /// Used to determine if a batch is eligible for garbage collection.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The batch to count live entries for
    ///
    /// # Returns
    ///
    /// The number of entries still marked as live in the batch.
    pub fn count_active_entries(&self, batch_id: ValueBatchId) -> usize {
        let (start_pos, end_pos) = self.get_batch_range(batch_id);
        let mut count = 0;
        for pos in start_pos..end_pos {
            if self.entries[pos] {
                count += 1;
            }
        }

        count
    }

    /// Returns the offsets of all live entries in the specified batch.
    ///
    /// Used during garbage collection to extract which values need to be
    /// copied to new batches.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The batch to get live entry offsets for
    ///
    /// # Returns
    ///
    /// A vector of offsets (as u32) for each live entry in the batch.
    pub fn get_active_entries(&self, batch_id: ValueBatchId) -> Vec<u32> {
        let (start_pos, end_pos) = self.get_batch_range(batch_id);
        self.entries[start_pos..end_pos]
            .iter()
            .enumerate()
            .filter_map(|(offset, active)| {
                if *active {
                    let pos = offset + start_pos;
                    Some(pos as u32)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns the start and end positions in the bitmap for the specified batch.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The batch to get the range for
    ///
    /// # Returns
    ///
    /// A tuple (start_pos, end_pos) defining the slice of the bitmap for this batch.
    #[inline]
    fn get_batch_range(&self, batch_id: ValueBatchId) -> (usize, usize) {
        let start_idx = batch_id
            .checked_sub(self.header.start_batch)
            .expect("Incompatible batch id") as usize;

        let start_pos = self.offsets[start_idx] as usize;
        let end_pos = if let Some(offset) = self.offsets.get(start_idx + 1) {
            *offset as usize
        } else {
            self.entries.len()
        };

        (start_pos, end_pos)
    }

    /// Marks a value as deleted using its value ID.
    ///
    /// Clears the corresponding bit in the bitmap to indicate the value
    /// is no longer live.
    ///
    /// # Arguments
    ///
    /// * `vid` - The value ID (batch_id, offset) to mark as deleted
    ///
    /// # Returns
    ///
    /// The absolute offset within the page's bitmap that was changed.
    ///
    /// # Panics
    ///
    /// Panics if the entry is already marked as deleted or if the batch ID
    /// doesn't belong to this page.
    ///
    /// # Note
    ///
    /// Changes are not persisted until `sync()` is called.
    pub fn mark_value_as_deleted(&mut self, vid: ValueId) -> u16 {
        let (start_pos, end_pos) = self.get_batch_range(vid.0);

        let mut marker = self.entries[start_pos..end_pos]
            .get_mut(vid.1 as usize)
            .expect("Entry index out of range");

        if !*marker {
            panic!("Entry already marked as deleted");
        }

        *marker = false;
        self.dirty = true;

        (start_pos as u16) + (vid.1 as u16)
    }

    /// Marks a value as deleted using its bitmap offset.
    ///
    /// This is used during recovery from the write-ahead log when we know
    /// the exact bitmap offset but not the value ID.
    ///
    /// # Arguments
    ///
    /// * `offset` - The offset within the page's bitmap
    pub fn mark_value_as_deleted_at(&mut self, offset: u16) {
        let mut marker = self
            .entries
            .get_mut(offset as usize)
            .expect("Offset out of range");

        if *marker {
            *marker = false;
            self.dirty = true;
        }
    }

    /// Marks an entire batch as deleted.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The batch to mark as deleted
    ///
    /// # Returns
    ///
    /// The index of the batch within this page.
    ///
    /// # Panics
    ///
    /// Panics if the batch is already marked as deleted.
    pub fn mark_batch_as_deleted(&mut self, batch_id: ValueBatchId) -> u16 {
        let idx = (batch_id - self.header.start_batch) as u16;
        self.mark_batch_as_deleted_at(idx);
        idx
    }

    /// Marks a batch as deleted using its index within this page.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the batch within this page
    ///
    /// # Panics
    ///
    /// Panics if the batch is already marked as deleted.
    pub fn mark_batch_as_deleted_at(&mut self, index: u16) {
        let marker = self.batches.get_mut(index as usize).expect("Out of range?");

        if *marker == ValueBatchState::Deleted {
            panic!("Batch already deleted?");
        }

        *marker = ValueBatchState::Deleted;
    }

    /// Marks a batch as compacted.
    ///
    /// A compacted batch has had its live values moved to other batches
    /// during garbage collection.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The batch to mark as compacted
    ///
    /// # Returns
    ///
    /// The index of the batch within this page.
    ///
    /// # Panics
    ///
    /// Panics if the batch is already compacted or deleted.
    pub fn mark_batch_as_compacted(&mut self, batch_id: ValueBatchId) -> u16 {
        let idx = batch_id - self.header.start_batch;
        let marker = self.batches.get_mut(idx as usize).expect("Out of range?");

        if *marker == ValueBatchState::Compacted {
            panic!("Batch already compacted?");
        }

        if *marker == ValueBatchState::Deleted {
            panic!("Batch already deleted?");
        }

        *marker = ValueBatchState::Deleted;
        idx as u16
    }
}

/// Keeps track of which entries in the value log are still
/// in use.
///
/// This is kept in a separate file to reduce the amount of
/// write amplification caused by value deletion.
/// A single page in the value index can hold information
/// for up to about 30k values.
pub struct ValueIndex {
    params: Arc<Params>,
    manifest: Arc<Manifest>,

    // Assuming a resonable number of entries (<1million)
    // this should never exceed 10mb, so we simply keep
    // the entire value index in memory.
    // For larger database it is safe to assume better hardware
    // with more available memory.
    pages: RwLock<VecDeque<(ValueBatchId, IndexPage)>>,
}

impl ValueIndex {
    /// Creates a new value index with an initial empty page.
    ///
    /// # Arguments
    ///
    /// * `params` - Database configuration parameters
    /// * `manifest` - Manifest for ID generation and metadata
    ///
    /// # Errors
    ///
    /// Returns an error if the initial page cannot be written to disk.
    pub async fn new(params: Arc<Params>, manifest: Arc<Manifest>) -> Result<Self, Error> {
        let obj = Self {
            params,
            manifest,
            pages: Default::default(),
        };

        // Create initial page and write it to disk
        {
            let mut pages = obj.pages.write().await;
            obj.create_new_page(&mut pages, MIN_VALUE_BATCH_ID).await;

            let (_, page) = pages.back_mut().unwrap();
            let fpath = obj.get_page_file_path(&page.get_identifier());
            page.sync(&fpath).await?;
        }

        Ok(obj)
    }

    /// Opens an existing value index from disk.
    ///
    /// Loads all index pages from the manifest's tracked range.
    ///
    /// # Arguments
    ///
    /// * `params` - Database configuration parameters
    /// * `manifest` - Manifest containing index page metadata
    ///
    /// # Errors
    ///
    /// Returns an error if any index page file cannot be read.
    pub async fn open(params: Arc<Params>, manifest: Arc<Manifest>) -> Result<Self, Error> {
        let obj = Self {
            params,
            manifest,
            pages: Default::default(),
        };

        let mut pages = obj.pages.write().await;

        let min_id = obj.manifest.get_minimum_value_index_page_id().await;
        let max_id = obj.manifest.get_most_recent_value_index_page_id().await;

        for page_id in min_id..=max_id {
            let sealed = page_id < max_id;

            let path = obj.get_page_file_path(&page_id);
            let page = IndexPage::open(&path, sealed).await?;
            let min_batch = page.header.start_batch;

            pages.push_back((min_batch, page));
        }

        drop(pages);
        Ok(obj)
    }

    /// Syncs all dirty index pages to disk.
    ///
    /// Iterates through all pages and writes those with pending changes.
    ///
    /// # Errors
    ///
    /// Returns an error if any page cannot be written to disk.
    pub async fn sync(&self) -> Result<(), Error> {
        let mut count = 0;
        let mut pages = self.pages.write().await;

        for (_, page) in pages.iter_mut() {
            if !page.dirty {
                continue;
            }

            let path = self.get_page_file_path(&page.get_identifier());
            let updated = page.sync(&path).await?;
            assert!(updated);
            count += 1;
        }

        log::trace!("Flushed {count} value index pages to disk");
        Ok(())
    }

    /// Returns the number of index pages currently loaded.
    pub async fn num_pages(&self) -> usize {
        self.pages.read().await.len()
    }

    /// Returns the file path for an index page.
    ///
    /// # Arguments
    ///
    /// * `page_id` - The unique identifier for the page
    #[inline]
    fn get_page_file_path(&self, page_id: &ValueIndexPageId) -> std::path::PathBuf {
        self.params.db_path.join(format!("vindex{page_id:08}.data"))
    }

    /// Finds the index of the page containing the specified batch.
    ///
    /// Uses binary search on the sorted page list.
    ///
    /// # Arguments
    ///
    /// * `pages` - The list of pages to search
    /// * `batch_id` - The batch ID to find
    ///
    /// # Returns
    ///
    /// The index of the page, or None if the batch has been garbage collected.
    #[inline]
    fn find_page_idx(
        pages: &VecDeque<(ValueBatchId, IndexPage)>,
        batch_id: ValueBatchId,
    ) -> Option<usize> {
        // We only keep the minimum batch id for every page
        // So the search might not return an exact match
        match pages.binary_search_by_key(&batch_id, |(k, _)| *k) {
            Ok(i) => Some(i),
            Err(0) => None, // already garbage collected?
            Err(i) => Some(i - 1),
        }
    }

    /// Finds the page containing the specified batch.
    ///
    /// # Arguments
    ///
    /// * `pages` - Read guard on the page list
    /// * `batch_id` - The batch ID to find
    ///
    /// # Returns
    ///
    /// A reference to the page, or None if not found.
    #[inline]
    fn find_page_for_batch<'a>(
        pages: &'a RwLockReadGuard<'_, VecDeque<(ValueBatchId, IndexPage)>>,
        batch_id: ValueBatchId,
    ) -> Option<&'a IndexPage> {
        let idx = Self::find_page_idx(pages, batch_id)?;
        Some(&pages[idx].1)
    }

    /// Returns the number of live entries in the specified batch.
    ///
    /// Used to determine if a batch is eligible for garbage collection.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The batch to count live entries for
    ///
    /// # Returns
    ///
    /// The number of live entries, or 0 if the batch has been garbage collected.
    pub async fn count_active_entries(&self, batch_id: ValueBatchId) -> usize {
        let pages = self.pages.read().await;
        match Self::find_page_for_batch(&pages, batch_id) {
            Some(p) => p.count_active_entries(batch_id),
            None => 0,
        }
    }

    /// Returns the offsets of all live entries in the specified batch.
    ///
    /// Used during garbage collection to identify which values need to be
    /// copied to new batches.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The batch to get live entry offsets for
    ///
    /// # Returns
    ///
    /// A vector of offsets for live entries, or an empty vector if the batch
    /// has been garbage collected.
    pub async fn get_active_entries(&self, batch_id: ValueBatchId) -> Vec<u32> {
        let pages = self.pages.read().await;
        match Self::find_page_for_batch(&pages, batch_id) {
            Some(p) => p.get_active_entries(batch_id),
            None => vec![],
        }
    }

    /// Adds a new batch to the index.
    ///
    /// Attempts to add the batch to the current page, creating a new page if needed.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The unique identifier for the batch
    /// * `num_entries` - The number of value entries in the batch
    ///
    /// # Errors
    ///
    /// Returns an error if the page cannot be written to disk.
    pub async fn add_batch(&self, batch_id: ValueBatchId, num_entries: usize) -> Result<(), Error> {
        let mut pages = self.pages.write().await;

        let p = if let Some((_, p)) = pages.back_mut()
            && p.expand(num_entries)
        {
            p
        } else {
            self.create_new_page(&mut pages, batch_id).await;
            let page = &mut pages.back_mut().unwrap().1;
            let success = page.expand(num_entries);
            assert!(success, "Data did not fit in new page?");
            page
        };

        // Persist changes to disk
        let path = self.get_page_file_path(&p.get_identifier());
        p.sync(&path).await?;
        Ok(())
    }

    /// Creates a new index page and adds it to the page list.
    ///
    /// # Arguments
    ///
    /// * `pages` - Mutable reference to the page list
    /// * `min_batch` - The first batch ID this page will track
    async fn create_new_page(
        &self,
        pages: &mut VecDeque<(ValueBatchId, IndexPage)>,
        min_batch: ValueBatchId,
    ) {
        let page_id = self.manifest.generate_next_value_index_id().await;

        let page = IndexPage::new(page_id, min_batch);
        pages.push_back((min_batch, page));
    }

    /// Marks a batch as deleted in the index.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The batch to mark as deleted
    ///
    /// # Returns
    ///
    /// A tuple of (page_id, offset) indicating where the change was made,
    /// for WAL logging.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch is not found (already garbage collected).
    pub async fn mark_batch_as_deleted(
        &self,
        batch_id: ValueBatchId,
    ) -> Result<(ValueIndexPageId, u16), Error> {
        let mut pages = self.pages.write().await;
        let page_idx = Self::find_page_idx(&pages, batch_id).expect("Outdated batch?");

        let page = &mut pages[page_idx].1;
        let offset = page.mark_batch_as_deleted(batch_id);
        Ok((page.get_identifier(), offset))
    }

    /// Marks a batch as deleted using page ID and batch index.
    ///
    /// This is used during recovery from the write-ahead log when we have
    /// the exact page and index from a previous deletion operation.
    ///
    /// # Arguments
    ///
    /// * `page_id` - The ID of the page containing the batch
    /// * `index` - The index of the batch within that page
    ///
    /// # Panics
    ///
    /// Panics if the page ID is not found.
    pub async fn mark_batch_as_deleted_at(
        &self,
        page_id: ValueIndexPageId,
        index: u16,
    ) -> Result<(), Error> {
        let mut pages = self.pages.write().await;
        match pages.binary_search_by_key(&page_id, |(_, p)| p.get_identifier()) {
            Ok(idx) => pages[idx].1.mark_batch_as_deleted_at(index),
            Err(_) => panic!("No page with id={page_id}"),
        }
        Ok(())
    }

    /// Marks a batch as compacted in the index.
    ///
    /// A compacted batch has had its live values moved during garbage collection.
    ///
    /// # Arguments
    ///
    /// * `batch_id` - The batch to mark as compacted
    ///
    /// # Returns
    ///
    /// A tuple of (page_id, offset) indicating where the change was made.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch is not found (already garbage collected).
    pub async fn mark_batch_as_compacted(
        &self,
        batch_id: ValueBatchId,
    ) -> Result<(ValueIndexPageId, u16), Error> {
        let mut pages = self.pages.write().await;
        let page_idx = Self::find_page_idx(&pages, batch_id).expect("Outdated batch?");

        let page = &mut pages[page_idx].1;
        let offset = page.mark_batch_as_compacted(batch_id);
        Ok((page.get_identifier(), offset))
    }

    /// Marks a value as deleted in the index.
    ///
    /// This updates the bitmap to indicate the value is no longer live.
    /// May trigger cleanup of unused pages.
    ///
    /// # Arguments
    ///
    /// * `vid` - The value ID (batch_id, offset) to mark as deleted
    ///
    /// # Returns
    ///
    /// A tuple of (page_id, offset) indicating where the change was made,
    /// for WAL logging.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch is not found (already garbage collected).
    pub async fn mark_value_as_deleted(
        &self,
        vid: ValueId,
    ) -> Result<(ValueIndexPageId, u16), Error> {
        let mut pages = self.pages.write().await;
        let page_idx = Self::find_page_idx(&pages, vid.0).expect("Outdated batch?");

        let page = &mut pages[page_idx].1;
        let offset = page.mark_value_as_deleted(vid);
        let page_id = page.get_identifier();

        // Tries to clean up unused pages
        // This only removes the oldest page(s)
        // and ensures there is always at least one page left
        //
        // TODO allow gaps as well!
        if page_idx == 0 && !page.is_in_use() && page.is_sealed() {
            loop {
                let id = {
                    let (_, page) = pages.front().unwrap();
                    let id = page.get_identifier();

                    // There is always one unsealed page
                    if !page.is_sealed() || page.is_in_use() {
                        self.manifest.set_minimum_value_index_page_id(id).await;
                        break;
                    }

                    id
                };

                log::trace!("Removing index page with id={id}");
                let fpath = self.get_page_file_path(&id);
                disk::remove_file(&fpath).await.map_err(|err| {
                    Error::from_io_error("Failed to remove value index page", err)
                })?;

                pages.pop_front();
            }
        }

        Ok((page_id, offset))
    }

    /// Marks a value as deleted using page ID and bitmap offset.
    ///
    /// This is used during recovery from the write-ahead log.
    ///
    /// # Arguments
    ///
    /// * `page_id` - The ID of the page containing the value
    /// * `offset` - The offset within the page's bitmap
    ///
    /// # Panics
    ///
    /// Panics if the page ID is not found.
    pub async fn mark_value_as_deleted_at(&self, page_id: ValueIndexPageId, offset: u16) {
        let mut pages = self.pages.write().await;

        match pages.binary_search_by_key(&page_id, |(_, p)| p.get_identifier()) {
            Ok(idx) => pages[idx].1.mark_value_as_deleted_at(offset),
            Err(_) => panic!("No page with id={page_id}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::test as async_test;

    use tempfile::{Builder, TempDir};

    use super::ValueIndex;

    use crate::manifest::Manifest;
    use crate::params::Params;

    async fn test_init() -> (TempDir, ValueIndex) {
        let tmp_dir = Builder::new()
            .prefix("lsm-value-index-test-")
            .tempdir()
            .unwrap();
        let _ = env_logger::builder().is_test(true).try_init();

        let params = Params {
            db_path: tmp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let params = Arc::new(params);
        let manifest = Arc::new(Manifest::new(params.clone()).await);

        (tmp_dir, ValueIndex::new(params, manifest).await.unwrap())
    }

    #[async_test]
    async fn add_batch() {
        let (_tmp_dir, value_index) = test_init().await;

        let batch_id = 1;
        let num_entries = 100;

        value_index.add_batch(batch_id, num_entries).await.unwrap();
        value_index
            .add_batch(batch_id + 1, num_entries)
            .await
            .unwrap();

        assert_eq!(value_index.num_pages().await, 1);
        assert_eq!(
            value_index.count_active_entries(batch_id).await,
            num_entries
        );
    }

    #[async_test]
    async fn multiple_pages() {
        let (_tmp_dir, value_index) = test_init().await;

        let batch_id = 1;
        let num_entries = 4000;

        value_index.add_batch(batch_id, num_entries).await.unwrap();
        value_index
            .add_batch(batch_id + 1, num_entries)
            .await
            .unwrap();

        assert_eq!(value_index.num_pages().await, 2);
        assert_eq!(
            value_index.count_active_entries(batch_id).await,
            num_entries
        );
    }

    #[async_test]
    async fn delete_entry() {
        let (_tmp_dir, value_index) = test_init().await;

        let batch_id = 1;
        let num_entries = 100;

        value_index.add_batch(batch_id, num_entries).await.unwrap();
        value_index
            .mark_value_as_deleted((batch_id, 2))
            .await
            .unwrap();
        value_index
            .mark_value_as_deleted((batch_id, 32))
            .await
            .unwrap();
        value_index
            .mark_value_as_deleted((batch_id, 59))
            .await
            .unwrap();

        assert_eq!(value_index.num_pages().await, 1);
        assert_eq!(
            value_index.count_active_entries(batch_id).await,
            num_entries - 3
        );
    }

    #[async_test]
    async fn remove_page() {
        let (_tmp_dir, value_index) = test_init().await;

        let batch_id = 1;
        let num_entries = 4000;

        value_index.add_batch(batch_id, num_entries).await.unwrap();
        value_index
            .add_batch(batch_id + 1, num_entries)
            .await
            .unwrap();

        for idx in 0..num_entries {
            value_index
                .mark_value_as_deleted((batch_id, idx as u32))
                .await
                .unwrap();
        }

        assert_eq!(value_index.num_pages().await, 1);
        assert_eq!(value_index.count_active_entries(batch_id).await, 0);
        assert_eq!(
            value_index.count_active_entries(batch_id + 1).await,
            num_entries
        );
    }
}
