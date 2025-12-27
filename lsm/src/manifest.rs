//! Manifest subsystem for tracking LSM-tree metadata.
//!
//! The manifest is the source of truth for all persistent metadata in the LSM-tree database.
//! It tracks critical information including:
//!
//! - Table identifiers organized by level
//! - Sequence numbers for ensuring write ordering
//! - Write-ahead log (WAL) offsets for crash recovery
//! - Data block ID allocation
//! - Value log metadata (when using WiscKey separation)
//!
//! The manifest persists all metadata to disk using memory-mapped files for efficient access
//! and atomic updates. It maintains a main database metadata file and separate metadata files
//! for each level in the LSM-tree hierarchy.
//!
//! # File Structure
//!
//! - `database.meta`: Main database metadata including global counters and offsets
//! - `level{N}.meta`: Per-level metadata containing ordered table ID lists
//!
//! # Concurrency
//!
//! The manifest uses async RwLocks to allow concurrent reads while ensuring exclusive access
//! for writes. This enables multiple readers to query metadata simultaneously while maintaining
//! consistency during updates.

use std::collections::HashSet;
use std::fs::OpenOptions;
use std::mem::size_of;
use std::path::Path;
use std::sync::Arc;

use byte_slice_cast::{AsByteSlice, AsSliceOf};

use memmap2::MmapMut;

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use tokio::sync::RwLock;

use crate::data_blocks::{DataBlockId, MIN_DATA_BLOCK_ID};
use crate::sorted_table::TableId;
use crate::{Error, Params};

#[cfg(feature = "wisckey")]
use crate::values::{MIN_VALUE_BATCH_ID, MIN_VALUE_INDEX_PAGE_ID, ValueBatchId, ValueIndexPageId};

/// Sequence number type for ordering writes.
///
/// Sequence numbers monotonically increase with each write operation,
/// providing a total ordering of all writes to the database.
pub type SeqNumber = u64;

/// Level identifier type for LSM-tree levels.
///
/// Level 0 is the topmost level (closest to memtable), and level numbers
/// increase as data moves down the tree.
pub type LevelId = u32;

/// Sentinel value representing an invalid or uninitialized table ID.
pub const INVALID_TABLE_ID: TableId = 0;

/// The first valid table ID allocated to a sorted string table.
pub const FIRST_TABLE_ID: TableId = 1;

/// Core metadata for the entire database.
///
/// This structure is memory-mapped to the main manifest file and contains
/// all global database state that must persist across restarts. It uses
/// zerocopy traits for efficient serialization directly to/from bytes.
///
/// # Layout
///
/// The struct uses `#[repr(C, align(8))]` to ensure a stable binary layout
/// across platforms and proper alignment for atomic operations.
#[derive(Default, KnownLayout, Immutable, IntoBytes, FromBytes)]
#[repr(C, align(8))]
struct DatabaseMetadata {
    /// The next table ID to allocate.
    ///
    /// This counter ensures each sorted string table receives a unique identifier.
    /// It monotonically increases and is never reused, even after table deletion.
    next_table_id: TableId,

    /// Total number of levels in the LSM-tree.
    ///
    /// This value is fixed at database creation and must match the configuration
    /// when reopening the database.
    num_levels: u32,

    /// Padding to maintain 8-byte alignment.
    _padding: u32,

    /// Current sequence number offset for write operations.
    ///
    /// Used to assign sequence numbers to writes, ensuring a total ordering.
    /// This value persists across restarts to maintain ordering invariants.
    seq_number_offset: SeqNumber,

    /// Byte offset into the write-ahead log (WAL).
    ///
    /// Tracks how much of the WAL has been successfully flushed to sorted tables.
    /// Used during recovery to determine which WAL entries need to be replayed.
    log_offset: usize,

    /// Next data block ID to allocate.
    ///
    /// Data blocks are the fundamental storage units for key-value data.
    /// This counter ensures unique allocation across the entire database lifetime.
    next_data_block_id: DataBlockId,

    /// Metadata for the WiscKey value log (when feature is enabled).
    #[cfg(feature = "wisckey")]
    value_log: ValueLogMetadata,
}

#[cfg(feature = "wisckey")]
#[derive(Default, KnownLayout, Immutable, IntoBytes, FromBytes)]
#[repr(C, align(8))]
/// Metadata for the WiscKey value log.
///
/// WiscKey is an optimization where large values are stored separately from keys,
/// reducing write amplification. This structure tracks the allocation and garbage
/// collection state of the value log.
///
/// # Invariants
///
/// - `minimum_batch < next_batch`: There are always unreclaimed batches between these bounds
/// - `minimum_index_page <= next_index_page`: Valid index page range
///
/// These invariants ensure the value log has valid data and prevent use-after-free bugs
/// during garbage collection.
struct ValueLogMetadata {
    /// Next value batch ID to allocate.
    ///
    /// Value batches group related values together for efficient I/O.
    next_batch: ValueBatchId,

    /// Minimum (oldest) value batch ID still in use.
    ///
    /// Batches below this ID have been garbage collected and reclaimed.
    /// Updated after successful compaction that removes old value references.
    minimum_batch: ValueBatchId,

    /// Next value index page ID to allocate.
    ///
    /// Index pages map keys to their locations in the value log.
    next_index_page: ValueIndexPageId,

    /// Minimum (oldest) value index page ID still in use.
    ///
    /// Index pages below this ID have been garbage collected.
    minimum_index_page: ValueIndexPageId,
}

/// Header structure for per-level metadata files.
///
/// Each level in the LSM-tree has its own metadata file containing an ordered list
/// of table identifiers. This header precedes the table ID array in the file.
///
/// # File Layout
///
/// ```text
/// [LevelMetadataHeader][TableId][TableId]...[TableId]
/// ```
///
/// The `#[repr(C, packed)]` ensures predictable binary layout without padding.
#[derive(IntoBytes, Default, Immutable, FromBytes, KnownLayout)]
#[repr(C, packed)]
struct LevelMetadataHeader {
    /// Number of tables currently present in this level.
    ///
    /// This count is used to determine how many TableId entries follow the header.
    num_tables: u64,
}

/// In-memory representation of metadata for a single LSM-tree level.
///
/// This structure manages a memory-mapped file containing the ordered list of
/// table IDs for one level. It supports efficient insertion, removal, and querying
/// of tables while maintaining the sorted order required for LSM operations.
struct LevelMetadata {
    /// Database configuration parameters.
    params: Arc<Params>,

    /// The level number this metadata represents (0 = topmost level).
    identifier: LevelId,

    /// Memory-mapped file containing the header and table ID array.
    ///
    /// The mmap allows efficient persistence and atomic updates via the OS page cache.
    data: MmapMut,
}

impl LevelMetadata {
    /// Returns a slice of all table IDs in this level, in sorted order.
    ///
    /// The returned slice is backed by the memory-mapped file and remains valid
    /// until the next mutation operation on this level.
    ///
    /// # Returns
    ///
    /// A slice of `TableId` values sorted in ascending order.
    pub fn get_table_ids(&self) -> &[TableId] {
        let (header, _) = LevelMetadataHeader::ref_from_prefix(&self.data[..]).unwrap();

        let start = size_of::<LevelMetadataHeader>();
        let end = start + (header.num_tables as usize * size_of::<TableId>());

        self.data[start..end].as_slice_of::<TableId>().unwrap()
    }

    /// Inserts a table ID into this level's sorted list.
    ///
    /// The table ID is inserted in the correct position to maintain sorted order.
    /// If the file is too small to accommodate the new entry, it is automatically
    /// resized in PAGE_SIZE increments.
    ///
    /// # Arguments
    ///
    /// * `id` - The table identifier to insert
    ///
    /// # Returns
    ///
    /// `true` if the table ID was newly inserted, `false` if it already existed.
    ///
    /// # Panics
    ///
    /// Panics if file operations (reopen, resize) fail, as these indicate
    /// unrecoverable I/O errors.
    pub fn insert(&mut self, id: TableId) -> bool {
        let tables = self.get_table_ids();

        let pos = match tables.binary_search(&id) {
            Ok(_) => return false,
            Err(pos) => pos,
        };

        let mut tables = tables.to_vec();
        tables.insert(pos, id);

        let (header, _) = LevelMetadataHeader::mut_from_prefix(&mut self.data[..]).unwrap();
        header.num_tables = tables.len() as u64;

        let start = size_of::<LevelMetadataHeader>();
        let new_data: &[u8] = tables.as_byte_slice();

        // Resize file?
        if start + new_data.len() >= self.data.len() {
            let fname = self
                .params
                .db_path
                .join(format!("{LEVEL_PREFIX}{}{LEVEL_SUFFIX}", self.identifier));

            let file = OpenOptions::new()
                .create(false)
                .truncate(false)
                .read(true)
                .write(true)
                .open(fname)
                .expect("Failed to reopen file");

            let new_size = (self.data.len() + PAGE_SIZE) as u64;
            log::trace!(
                "Resizing metadata file for level #{} to {new_size}",
                self.identifier
            );
            file.set_len(new_size)
                .expect("Failed to resize level metadata file");
            self.data = unsafe { MmapMut::map_mut(&file) }.unwrap();
        }

        self.data[start..start + new_data.len()].copy_from_slice(new_data);
        true
    }

    /// Removes a table ID from this level's sorted list.
    ///
    /// The table ID is removed and subsequent entries are shifted left to
    /// maintain contiguous storage. The file size is not reduced.
    ///
    /// # Arguments
    ///
    /// * `id` - The table identifier to remove
    ///
    /// # Returns
    ///
    /// `true` if the table ID was found and removed, `false` if it didn't exist.
    pub fn remove(&mut self, id: &TableId) -> bool {
        let tables = self.get_table_ids();

        let pos = match tables.binary_search(id) {
            Ok(pos) => pos,
            Err(_) => return false,
        };

        let mut tables = tables.to_vec();
        tables.remove(pos);

        let (header, next) = LevelMetadataHeader::mut_from_prefix(&mut self.data[..]).unwrap();
        header.num_tables = tables.len() as u64;

        let new_data: &[u8] = tables.as_byte_slice();
        next[..new_data.len()].copy_from_slice(new_data);

        true
    }

    /// Flushes all pending changes to disk.
    ///
    /// Ensures that all modifications to the level metadata are persisted by
    /// calling msync on the underlying memory-mapped file.
    ///
    /// # Panics
    ///
    /// Panics if the flush operation fails, as this indicates a critical I/O error.
    pub fn flush(&mut self) {
        self.data.flush().unwrap();
    }
}

/// The manifest manages all persistent metadata for the LSM-tree database.
///
/// This is the central coordination point for tracking tables across levels,
/// managing ID allocation, and maintaining consistency between in-memory and
/// on-disk state. All metadata changes flow through the manifest to ensure
/// proper persistence and recovery.
///
/// # Thread Safety
///
/// The manifest uses async RwLocks internally, allowing:
/// - Multiple concurrent readers for queries
/// - Exclusive access for mutations
/// - Prevention of race conditions during level compaction
///
/// # Persistence
///
/// All manifest data is backed by memory-mapped files, providing:
/// - Fast access without deserialization overhead
/// - Automatic persistence via the OS page cache
/// - Crash recovery via the write-ahead log
///
/// # Example
///
/// ```no_run
/// use std::sync::Arc;
/// use lsm::params::Params;
/// use lsm::manifest::Manifest;
///
/// # async fn example() {
/// let params = Arc::new(Params::default());
///
/// // Create a new manifest for an empty database
/// let manifest = Manifest::new(params.clone()).await;
///
/// // Allocate a new table ID
/// let table_id = manifest.generate_next_table_id().await;
///
/// // Add the table to level 0
/// manifest.update_table_set(vec![(0, table_id)], vec![]).await;
/// # }
/// ```
pub struct Manifest {
    /// Database configuration parameters.
    params: Arc<Params>,

    /// Memory-mapped main database metadata file.
    ///
    /// Protected by RwLock to allow concurrent reads of metadata while
    /// ensuring exclusive access during updates.
    metadata: RwLock<MmapMut>,

    /// Per-level metadata structures.
    ///
    /// Each element corresponds to one level in the LSM-tree, indexed by level ID.
    /// Protected by RwLock to coordinate updates during compaction.
    levels: RwLock<Vec<LevelMetadata>>,
}

/// Constant filename for the main manifest metadata file.
const MANIFEST_NAME: &str = "database.meta";

/// Filename components for per-level metadata files.
const LEVEL_PREFIX: &str = "level";

/// Filename suffix for per-level metadata files.
const LEVEL_SUFFIX: &str = ".meta";

/// Page size constant for resizing metadata files.
const PAGE_SIZE: usize = 4 * 1024;

impl Manifest {
    /// Creates a new manifest for an empty database.
    ///
    /// Initializes all metadata files with default values and sets up the level
    /// hierarchy. This should only be called when creating a brand new database.
    ///
    /// # Arguments
    ///
    /// * `params` - Database configuration parameters including the database path
    ///   and number of levels
    ///
    /// # Returns
    ///
    /// A new `Manifest` instance with all metadata files created and initialized.
    ///
    /// # File Creation
    ///
    /// Creates the following files:
    /// - `database.meta`: Main metadata file
    /// - `level0.meta` through `level{N-1}.meta`: Per-level metadata files
    ///
    /// All files are created with O_CREAT | O_TRUNC flags, so existing data is lost.
    ///
    /// # Panics
    ///
    /// Panics if any file operations fail, as this indicates the database directory
    /// is not accessible or writable.
    pub async fn new(params: Arc<Params>) -> Self {
        let meta = DatabaseMetadata {
            next_table_id: FIRST_TABLE_ID,
            num_levels: params.num_levels as u32,
            _padding: 0,
            seq_number_offset: 1,
            log_offset: 0,
            next_data_block_id: MIN_DATA_BLOCK_ID,
            #[cfg(feature = "wisckey")]
            value_log: ValueLogMetadata {
                next_batch: MIN_VALUE_BATCH_ID,
                minimum_batch: 0,
                next_index_page: MIN_VALUE_INDEX_PAGE_ID,
                minimum_index_page: MIN_VALUE_INDEX_PAGE_ID,
            },
        };

        let mut meta_mmap = {
            let manifest_path = params.db_path.join(Path::new(MANIFEST_NAME));

            let file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .read(true)
                .write(true)
                .open(manifest_path)
                .unwrap();

            file.set_len(size_of::<DatabaseMetadata>() as u64).unwrap();
            unsafe { MmapMut::map_mut(&file) }.unwrap()
        };
        meta_mmap.copy_from_slice(meta.as_bytes());
        meta_mmap.flush().unwrap();

        let mut levels = Vec::new();

        for idx in 0..params.num_levels {
            let level_path = params
                .db_path
                .join(format!("{LEVEL_PREFIX}{idx}{LEVEL_SUFFIX}"));

            let mut level_mmap = {
                let file = OpenOptions::new()
                    .create(true)
                    .truncate(true)
                    .read(true)
                    .write(true)
                    .open(level_path)
                    .unwrap();
                file.set_len(PAGE_SIZE as u64).unwrap();
                unsafe { MmapMut::map_mut(&file) }.unwrap()
            };

            let level_header = LevelMetadataHeader { num_tables: 0 };
            level_mmap[..size_of::<LevelMetadataHeader>()].copy_from_slice(level_header.as_bytes());
            level_mmap.flush().unwrap();

            levels.push(LevelMetadata {
                params: params.clone(),
                identifier: idx as LevelId,
                data: level_mmap,
            });
        }

        Self {
            metadata: RwLock::new(meta_mmap),
            levels: RwLock::new(levels),
            params,
        }
    }

    /// Opens an existing manifest from disk.
    ///
    /// Loads all metadata files and validates that the database structure matches
    /// the provided configuration. This is called when reopening an existing database.
    ///
    /// # Arguments
    ///
    /// * `params` - Database configuration parameters
    ///
    /// # Returns
    ///
    /// - `Ok(Manifest)` if the manifest was successfully loaded
    /// - `Err(Error)` if the manifest files don't exist or cannot be opened
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The manifest files don't exist
    /// - File permissions prevent reading/writing
    /// - The filesystem is inaccessible
    ///
    /// # Panics
    ///
    /// Panics if the number of levels in the persisted metadata doesn't match
    /// `params.num_levels`, as this indicates an incompatible database format.
    pub async fn open(params: Arc<Params>) -> Result<Self, Error> {
        let manifest_path = params.db_path.join(Path::new(MANIFEST_NAME));

        let file = OpenOptions::new()
            .create(false)
            .truncate(false)
            .read(true)
            .write(true)
            .open(manifest_path)
            .map_err(|err| Error::from_io_error("Failed to open manifest", err))?;

        let data = unsafe { MmapMut::map_mut(&file) }.unwrap();

        let meta = DatabaseMetadata::ref_from_bytes(&data[..]).unwrap();
        if meta.num_levels != params.num_levels as u32 {
            panic!("Number of levels is incompatible");
        }

        let mut table_count = 0;
        let mut levels = vec![];

        for idx in 0..meta.num_levels {
            let fname = params
                .db_path
                .join(format!("{LEVEL_PREFIX}{idx}{LEVEL_SUFFIX}"));

            let file = OpenOptions::new()
                .create(false)
                .truncate(false)
                .read(true)
                .write(true)
                .open(fname)
                .map_err(|err| Error::from_io_error("Failed to open manifest", err))?;

            let data = unsafe { MmapMut::map_mut(&file) }.unwrap();

            let (header, _) = LevelMetadataHeader::ref_from_prefix(&data[..]).unwrap();
            table_count += header.num_tables;

            levels.push(LevelMetadata {
                identifier: idx as LevelId,
                params: params.clone(),
                data,
            });
        }

        log::debug!("Found {table_count} tables");

        Ok(Self {
            metadata: RwLock::new(data),
            levels: RwLock::new(levels),
            params,
        })
    }

    /// Generates and allocates the next data block ID.
    ///
    /// Atomically increments the data block counter and returns the newly allocated ID.
    /// The change is immediately flushed to disk to ensure crash consistency.
    ///
    /// # Returns
    ///
    /// A unique `DataBlockId` that has never been allocated before and will never be
    /// allocated again.
    ///
    /// # Panics
    ///
    /// Panics if the flush to disk fails, indicating a critical I/O error.
    pub async fn generate_next_data_block_id(&self) -> DataBlockId {
        let mut mmap = self.metadata.write().await;
        let meta = DatabaseMetadata::mut_from_bytes(&mut mmap[..]).unwrap();

        let id = meta.next_data_block_id;
        meta.next_data_block_id += 1;

        mmap.flush().unwrap();

        id
    }

    /// Generates and allocates the next table ID.
    ///
    /// Atomically increments the table ID counter and returns the newly allocated ID.
    /// Each sorted string table in the database has a unique, never-reused identifier.
    ///
    /// # Returns
    ///
    /// A unique `TableId` that will be used to identify a new sorted table.
    ///
    /// # Panics
    ///
    /// Panics if the flush to disk fails, indicating a critical I/O error.
    pub async fn generate_next_table_id(&self) -> TableId {
        let mut mmap = self.metadata.write().await;
        let meta = DatabaseMetadata::mut_from_bytes(&mut mmap[..]).unwrap();

        let id = meta.next_table_id;
        meta.next_table_id += 1;

        mmap.flush().unwrap();

        id
    }

    /// Retrieves the current write-ahead log (WAL) offset.
    ///
    /// The WAL offset indicates how many bytes of the WAL have been successfully
    /// flushed to sorted tables. During recovery, only entries after this offset
    /// need to be replayed.
    ///
    /// # Returns
    ///
    /// The byte offset into the WAL of the last flushed entry.
    pub async fn get_log_offset(&self) -> usize {
        let mmap = self.metadata.read().await;
        let meta = DatabaseMetadata::ref_from_bytes(&mmap[..]).unwrap();

        meta.log_offset
    }

    /// Updates the write-ahead log (WAL) offset.
    ///
    /// Called after successfully flushing memtable data to sorted tables, advancing
    /// the recovery point. The offset must be monotonically increasing.
    ///
    /// # Arguments
    ///
    /// * `offset` - The new WAL offset, must be greater than the current offset
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The new offset is not greater than the current offset (violates monotonicity)
    /// - The flush to disk fails
    pub async fn set_log_offset(&self, offset: usize) {
        let mut mmap = self.metadata.write().await;
        let meta = DatabaseMetadata::mut_from_bytes(&mut mmap[..]).unwrap();

        assert!(meta.log_offset < offset);
        meta.log_offset = offset;

        mmap.flush().unwrap();
    }

    /// Retrieves the minimum (oldest) value batch ID still in use.
    ///
    /// Value batches with IDs below this have been garbage collected and reclaimed.
    /// This is used by the WiscKey garbage collector to determine which batches are safe to delete.
    ///
    /// # Returns
    ///
    /// The ID of the oldest value batch that still contains live data.
    #[cfg(feature = "wisckey")]
    pub async fn get_minimum_value_batch(&self) -> ValueBatchId {
        let mmap = self.metadata.read().await;
        let meta = DatabaseMetadata::ref_from_bytes(&mmap[..]).unwrap();

        meta.value_log.minimum_batch
    }

    /// Updates the minimum value batch ID after garbage collection.
    ///
    /// Advances the garbage collection watermark, indicating that all batches below
    /// the new offset have been reclaimed.
    ///
    /// # Arguments
    ///
    /// * `offset` - The new minimum batch ID, must be greater than the current minimum
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The new offset is not greater than the current offset
    /// - The flush to disk fails
    #[cfg(feature = "wisckey")]
    pub async fn set_minimum_value_batch_id(&self, offset: ValueBatchId) {
        let mut mmap = self.metadata.write().await;
        let meta = DatabaseMetadata::mut_from_bytes(&mut mmap[..]).unwrap();

        assert!(meta.value_log.minimum_batch < offset);
        meta.value_log.minimum_batch = offset;

        mmap.flush().unwrap();
    }

    /// Updates the minimum value index page ID after garbage collection.
    ///
    /// Advances the garbage collection watermark for index pages.
    ///
    /// # Arguments
    ///
    /// * `offset` - The new minimum index page ID, must be greater than the current minimum
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The new offset is not greater than the current offset
    /// - The flush to disk fails
    #[cfg(feature = "wisckey")]
    pub async fn set_minimum_value_index_page_id(&self, offset: ValueIndexPageId) {
        let mut mmap = self.metadata.write().await;
        let meta = DatabaseMetadata::mut_from_bytes(&mut mmap[..]).unwrap();

        assert!(meta.value_log.minimum_index_page < offset);
        meta.value_log.minimum_batch = offset;

        mmap.flush().unwrap();
    }

    /// Retrieves the minimum value index page ID still in use.
    ///
    /// Index pages with IDs below this have been garbage collected.
    ///
    /// # Returns
    ///
    /// The ID of the oldest value index page that still contains live data.
    #[cfg(feature = "wisckey")]
    pub async fn get_minimum_value_index_page_id(&self) -> ValueIndexPageId {
        let mmap = self.metadata.read().await;
        let meta = DatabaseMetadata::ref_from_bytes(&mmap[..]).unwrap();
        meta.value_log.minimum_index_page
    }

    /// Retrieves the current sequence number offset.
    ///
    /// The sequence number is used to assign a total ordering to all write operations.
    /// Each write receives a monotonically increasing sequence number.
    ///
    /// # Returns
    ///
    /// The current sequence number that will be assigned to the next write.
    pub async fn get_seq_number_offset(&self) -> SeqNumber {
        let mmap = self.metadata.read().await;
        let meta = DatabaseMetadata::ref_from_bytes(&mmap[..]).unwrap();

        meta.seq_number_offset
    }

    /// Updates the sequence number offset.
    ///
    /// Advances the sequence number counter, typically after a batch of writes has been
    /// processed. The sequence number must strictly increase to maintain write ordering.
    ///
    /// # Arguments
    ///
    /// * `offset` - The new sequence number, must be strictly greater than the current value
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The new offset is not strictly greater than the current offset
    /// - The flush to disk fails
    pub async fn set_seq_number_offset(&self, offset: SeqNumber) {
        let mut mmap = self.metadata.write().await;
        let meta = DatabaseMetadata::mut_from_bytes(&mut mmap[..]).unwrap();

        if offset <= meta.seq_number_offset {
            panic!(
                "Sequence number must montonically increase. Old value ({}) was >= than new value {offset}",
                meta.seq_number_offset
            );
        }

        meta.seq_number_offset = offset;

        mmap.flush().unwrap();
    }

    /// Generates and allocates the next value index page ID.
    ///
    /// Atomically increments the value index page counter for WiscKey.
    ///
    /// # Returns
    ///
    /// A unique `ValueIndexPageId` for a new index page.
    ///
    /// # Panics
    ///
    /// Panics if the flush to disk fails.
    #[cfg(feature = "wisckey")]
    pub async fn generate_next_value_index_id(&self) -> ValueIndexPageId {
        let mut mmap = self.metadata.write().await;
        let meta = DatabaseMetadata::mut_from_bytes(&mut mmap[..]).unwrap();

        let id = meta.value_log.next_index_page;
        meta.value_log.next_index_page += 1;

        mmap.flush().unwrap();

        id
    }

    /// Generates and allocates the next value batch ID.
    ///
    /// Atomically increments the value batch counter for WiscKey.
    ///
    /// # Returns
    ///
    /// A unique `ValueBatchId` for a new batch of values.
    ///
    /// # Panics
    ///
    /// Panics if the flush to disk fails.
    #[cfg(feature = "wisckey")]
    pub async fn generate_next_value_batch_id(&self) -> ValueBatchId {
        let mut mmap = self.metadata.write().await;
        let meta = DatabaseMetadata::mut_from_bytes(&mut mmap[..]).unwrap();

        let id = meta.value_log.next_batch;
        meta.value_log.next_batch += 1;

        mmap.flush().unwrap();

        id
    }

    /// Retrieves the most recently allocated value index page ID.
    ///
    /// Returns the ID of the last index page that was allocated, which is one less
    /// than the next ID that will be allocated.
    ///
    /// # Returns
    ///
    /// The ID of the most recent value index page.
    #[cfg(feature = "wisckey")]
    pub async fn get_most_recent_value_index_page_id(&self) -> ValueBatchId {
        let mmap = self.metadata.read().await;
        let meta = DatabaseMetadata::ref_from_bytes(&mmap[..]).unwrap();

        meta.value_log.next_index_page - 1
    }

    /// Retrieves the most recently allocated value batch ID.
    ///
    /// Returns the ID of the last batch that was allocated, which is one less
    /// than the next ID that will be allocated.
    ///
    /// # Returns
    ///
    /// The ID of the most recent value batch.
    #[cfg(feature = "wisckey")]
    pub async fn get_most_recent_value_batch_id(&self) -> ValueBatchId {
        let mmap = self.metadata.read().await;
        let meta = DatabaseMetadata::ref_from_bytes(&mmap[..]).unwrap();

        meta.value_log.next_batch - 1
    }

    /// Retrieves all table IDs across all levels.
    ///
    /// Returns a nested vector where each inner vector contains the sorted table IDs
    /// for one level. This is primarily used during database recovery to rebuild the
    /// in-memory representation of the LSM-tree structure.
    ///
    /// # Returns
    ///
    /// A vector of vectors, where `result[i]` contains the sorted table IDs for level `i`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lsm::manifest::Manifest;
    /// # async fn example(manifest: &Manifest) {
    /// let all_tables = manifest.get_table_ids().await;
    /// for (level, tables) in all_tables.iter().enumerate() {
    ///     println!("Level {}: {} tables", level, tables.len());
    /// }
    /// # }
    /// ```
    pub async fn get_table_ids(&self) -> Vec<Vec<TableId>> {
        let mut result = Vec::with_capacity(self.params.num_levels);

        for level in self.levels.read().await.iter() {
            result.push(level.get_table_ids().to_vec());
        }

        result
    }

    /// Atomically updates the set of tables across multiple levels.
    ///
    /// This is the primary mechanism for modifying the LSM-tree structure during
    /// compaction operations. Tables can be added to and removed from different levels
    /// in a single atomic operation.
    ///
    /// # Arguments
    ///
    /// * `add` - List of (level_id, table_id) pairs to add to the respective levels
    /// * `remove` - List of (level_id, table_id) pairs to remove from the respective levels
    ///
    /// # Concurrency
    ///
    /// The caller **must** hold locks on all affected levels before calling this method
    /// to prevent race conditions during compaction. This typically means holding the
    /// level locks in the database's level management system.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - Attempting to add a table ID that already exists in the specified level
    /// - Attempting to remove a table ID that doesn't exist in the specified level
    /// - Specifying an invalid level ID
    ///
    /// These panics indicate programming errors in the compaction logic.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lsm::manifest::Manifest;
    /// # async fn compact(manifest: &Manifest) {
    /// // Compaction: merge tables 1 and 2 from level 0 into table 3 on level 1
    /// manifest.update_table_set(
    ///     vec![(1, 3)],           // Add table 3 to level 1
    ///     vec![(0, 1), (0, 2)]   // Remove tables 1 and 2 from level 0
    /// ).await;
    /// # }
    /// ```
    pub async fn update_table_set(
        &self,
        add: Vec<(LevelId, TableId)>,
        remove: Vec<(LevelId, TableId)>,
    ) {
        log::trace!("Updating table set: add={add:?} remove={remove:?}");

        let mut levels = self.levels.write().await;
        let mut affected = HashSet::new();

        for (level, id) in add.into_iter() {
            let was_new = levels
                .get_mut(level as usize)
                .expect("No such level")
                .insert(id);

            if !was_new {
                panic!("Table with id={id} already existed on level #{level}");
            }
            affected.insert(level);
        }

        for (level, id) in remove.into_iter() {
            let existed = levels
                .get_mut(level as usize)
                .expect("No such level")
                .remove(&id);

            if !existed {
                panic!("No table with id={id} existed on level #{level}");
            }
            affected.insert(level);
        }

        for level_id in affected.into_iter() {
            levels[level_id as usize].flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tempfile::tempdir;

    use tokio::test as async_test;

    use crate::params::Params;

    use super::Manifest;

    #[async_test]
    async fn update_table_set() {
        let dir = tempdir().unwrap();
        let params = Arc::new(Params {
            db_path: dir.path().to_path_buf(),
            ..Default::default()
        });

        let manifest = Manifest::new(params).await;

        assert!(manifest.get_table_ids().await[0].is_empty());
        assert!(manifest.get_table_ids().await[1].is_empty());

        manifest
            .update_table_set(vec![(0, 1), (0, 2)], vec![])
            .await;

        assert_eq!(manifest.get_table_ids().await[0], vec![1, 2]);
        assert!(manifest.get_table_ids().await[1].is_empty());

        manifest
            .update_table_set(vec![(1, 3)], vec![(0, 1), (0, 2)])
            .await;

        assert!(manifest.get_table_ids().await[0].is_empty());
        assert_eq!(manifest.get_table_ids().await[1], vec![3]);
    }
}
