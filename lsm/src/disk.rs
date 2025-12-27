//! Low-level disk I/O operations with optional compression.
//!
//! This module provides async wrappers around file system operations with support for
//! Snappy compression when the `snappy-compression` feature is enabled. All functions
//! handle file creation, reading, writing, and deletion.
//!
//! ## Compression
//!
//! When the `snappy-compression` feature is enabled:
//! - `write()` automatically compresses data before writing
//! - `read()` automatically decompresses data after reading
//! - `write_uncompressed()` and `read_uncompressed()` bypass compression
//!
//! ## Sync Guarantees
//!
//! All write operations call `sync_all()` to ensure data is flushed to disk before returning,
//! providing durability guarantees.

use std::fs;
use std::io::{Read, Seek, Write};
use std::path::Path;

use cfg_if::cfg_if;

/// Reads file contents from the given offset to the end without decompression.
///
/// This function bypasses decompression even if the `snappy-compression` feature is enabled.
/// Useful for reading files that were written without compression.
///
/// # Arguments
///
/// * `fpath` - Path to the file to read
/// * `offset` - Byte offset to start reading from (0 to read from the beginning)
///
/// # Returns
///
/// The raw (possibly compressed) file contents as a byte vector.
///
/// # Errors
///
/// Returns an I/O error if the file cannot be opened or read.
#[inline(always)]
#[tracing::instrument]
pub async fn read_uncompressed(fpath: &Path, offset: u64) -> Result<Vec<u8>, std::io::Error> {
    let mut file = fs::File::open(fpath)?;

    if offset > 0 {
        file.seek(std::io::SeekFrom::Start(offset))?;
    }

    let mut buf = vec![];
    file.read_to_end(&mut buf)?;

    Ok(buf)
}

/// Reads and decompresses file contents from the given offset to the end.
///
/// Automatically decompresses data using Snappy if the `snappy-compression` feature is enabled.
/// Otherwise, returns the raw file contents.
///
/// # Arguments
///
/// * `fpath` - Path to the file to read
/// * `offset` - Byte offset to start reading from (0 to read from the beginning)
///
/// # Returns
///
/// The decompressed file contents as a byte vector.
///
/// # Errors
///
/// Returns an I/O error if:
/// - The file cannot be opened or read
/// - Decompression fails (with `snappy-compression` feature)
#[inline(always)]
#[tracing::instrument]
pub async fn read(fpath: &Path, offset: u64) -> Result<Vec<u8>, std::io::Error> {
    let compressed = read_uncompressed(fpath, offset).await?;

    cfg_if! {
        if #[ cfg(feature="snappy-compression") ] {
            let mut decoder = snap::raw::Decoder::new();
            Ok(decoder.decompress_vec(&compressed)?)
        } else {
            Ok(compressed)
        }
    }
}

/// Compresses and writes data to the specified file path.
///
/// Creates the file if it doesn't exist, truncates it if it does. Automatically compresses
/// data using Snappy if the `snappy-compression` feature is enabled. Calls `sync_all()` to
/// ensure data is flushed to disk.
///
/// # Arguments
///
/// * `fpath` - Path where the file will be written
/// * `data` - Data to compress and write
///
/// # Errors
///
/// Returns an I/O error if:
/// - The file cannot be created or opened
/// - Compression fails (with `snappy-compression` feature)
/// - Writing or syncing fails
///
/// # Note
///
/// TODO: It might be worth investigating if encoding/decoding chunks is more efficient.
#[tracing::instrument(skip(data))]
#[inline(always)]
pub async fn write(fpath: &Path, data: &[u8]) -> Result<(), std::io::Error> {
    //TODO it might be worth investigating if encoding/decoding
    // chunks is more efficient

    cfg_if! {
        if #[cfg(feature="snappy-compression") ] {
            let mut encoder = snap::raw::Encoder::new();
            let compressed = encoder.compress_vec(data)
                .expect("Failed to compress data");
        } else {
            let mut compressed = vec![];
            compressed.extend_from_slice(data);
        }
    }

    write_uncompressed(fpath, compressed).await
}

/// Writes uncompressed data to the specified file path.
///
/// Creates the file if it doesn't exist, truncates it if it does. This function bypasses
/// compression even if the `snappy-compression` feature is enabled. Calls `sync_all()` to
/// ensure data is flushed to disk.
///
/// # Arguments
///
/// * `fpath` - Path where the file will be written
/// * `data` - Uncompressed data to write
///
/// # Errors
///
/// Returns an I/O error if the file cannot be created, written, or synced.
#[tracing::instrument(skip(data))]
#[inline(always)]
pub async fn write_uncompressed(fpath: &Path, data: Vec<u8>) -> Result<(), std::io::Error> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(fpath)?;

    file.write_all(&data)?;
    file.sync_all()?;

    Ok(())
}

/// Deletes a file from the filesystem.
///
/// # Arguments
///
/// * `fpath` - Path to the file to delete
///
/// # Errors
///
/// Returns an I/O error if the file cannot be deleted (e.g., doesn't exist or insufficient permissions).
pub async fn remove_file(fpath: &Path) -> Result<(), std::io::Error> {
    std::fs::remove_file(fpath)?;
    Ok(())
}

/// Recursively removes a directory and all its contents.
///
/// # Arguments
///
/// * `path` - Path to the directory to remove
///
/// # Errors
///
/// Returns an I/O error if the directory cannot be removed.
pub fn remove_dir_all(path: &Path) -> Result<(), std::io::Error> {
    fs::remove_dir_all(path)?;
    Ok(())
}

/// Creates a new directory.
///
/// Does not create parent directories - they must already exist.
///
/// # Arguments
///
/// * `path` - Path of the directory to create
///
/// # Errors
///
/// Returns an I/O error if:
/// - The directory already exists
/// - Parent directories don't exist
/// - Insufficient permissions
pub fn create_dir(path: &Path) -> Result<(), std::io::Error> {
    fs::create_dir(path)?;
    Ok(())
}
