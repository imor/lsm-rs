use std::fs;
use std::io::{Read, Seek, Write};
use std::path::Path;

use cfg_if::cfg_if;

/// Read from the offset to the end of the file
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

/// Read the contents of the file from the given offset to
/// its end.
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

/// Writes the data to the specified file path
///
/// This will create the file if it does not exist yet.
/// It will also compress the data, if enabled.
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

/// Writes the uncompressed (even if the feature is enabled)
/// to the specified file path
///
/// This will create the file if it does not exist yet.
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

pub async fn remove_file(fpath: &Path) -> Result<(), std::io::Error> {
    std::fs::remove_file(fpath)?;
    Ok(())
}

pub fn remove_dir_all(path: &Path) -> Result<(), std::io::Error> {
    fs::remove_dir_all(path)?;
    Ok(())
}

pub fn create_dir(path: &Path) -> Result<(), std::io::Error> {
    fs::create_dir(path)?;
    Ok(())
}
