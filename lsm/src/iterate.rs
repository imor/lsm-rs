//! Iterator implementation for scanning key-value pairs in the LSM-tree.
//!
//! This module provides efficient iteration over the entire database or a key range,
//! merging data from multiple sources (memtables and SSTables) using a k-way merge algorithm.
//!
//! # Overview
//!
//! The iterator handles the complexity of merging sorted data from multiple sources:
//! - Multiple in-memory memtables
//! - Multiple on-disk SSTables across different levels
//!
//! It ensures that:
//! - Keys are returned in sorted order (forward or reverse)
//! - For duplicate keys, the version with the highest sequence number is returned
//! - Deleted entries (tombstones) are filtered out
//! - Only keys within the specified range are returned
//!
//! # K-Way Merge Algorithm
//!
//! The iterator maintains references to all source iterators and performs a k-way merge:
//! 1. Each iteration examines all source iterators
//! 2. Selects the next key according to sort order (min for forward, max for reverse)
//! 3. If multiple iterators have the same key, selects the one with highest sequence number
//! 4. Advances past all versions of the selected key in all iterators
//! 5. Filters out deletion markers before returning to the user
//!
//! # Iteration Modes
//!
//! - **Forward iteration**: Returns keys in ascending order
//! - **Reverse iteration**: Returns keys in descending order
//! - **Range scans**: Supports optional min/max key bounds
//!
//! # Async Stream
//!
//! The iterator implements the `Stream` trait, allowing it to be used with async/await
//! and standard stream combinators.

use std::cmp::Ordering;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

#[cfg(feature = "wisckey")]
use std::sync::Arc;

#[cfg(feature = "wisckey")]
use crate::values::ValueLog;

use crate::logic::EntryRef;
use crate::memtable::MemtableIterator;
use crate::sorted_table::{InternalIterator, TableIterator};
use crate::{Error, Key};

use futures::stream::Stream;


/// Future type representing the async computation of the next key-value pair.
///
/// Returns the updated iterator state and optionally the next item.
type IterFuture =
    dyn Future<Output = Result<(DbIteratorInner, Option<(Key, EntryRef)>), Error>> + Send;

/// Asynchronous iterator over the database key-value pairs.
///
/// Implements the `Stream` trait to provide async iteration over database entries.
/// The iterator merges data from multiple sources (memtables and SSTables) and
/// returns entries in sorted order.
///
/// # Example
///
/// ```ignore
/// use futures::StreamExt;
///
/// let mut iter = db.scan(None, None, false).await?;
/// while let Some((key, value)) = iter.next().await {
///     println!("Key: {:?}, Value: {:?}", key, value);
/// }
/// ```
pub struct DbIterator {
    /// Current state of the iterator as a pinned future.
    /// `None` indicates the iterator has been exhausted.
    state: Option<Pin<Box<IterFuture>>>,
}

impl DbIterator {
    /// Creates a new database iterator.
    ///
    /// # Arguments
    /// * `mem_iters` - Iterators for in-memory memtables
    /// * `table_iters` - Iterators for on-disk SSTables
    /// * `min_key` - Optional minimum key (inclusive for forward, exclusive for reverse)
    /// * `max_key` - Optional maximum key (exclusive for forward, inclusive for reverse)
    /// * `reverse` - If true, iterate in reverse (descending) order
    /// * `value_log` - Value log reference (only with "wisckey" feature)
    ///
    /// The iterator is initialized and immediately begins computing the first item.
    pub(crate) fn new(
        mem_iters: Vec<MemtableIterator>,
        table_iters: Vec<TableIterator>,
        min_key: Option<Vec<u8>>,
        max_key: Option<Vec<u8>>,
        reverse: bool,
        #[cfg(feature = "wisckey")] value_log: Arc<ValueLog>,
    ) -> Self {
        let inner = DbIteratorInner::new(
            mem_iters,
            table_iters,
            min_key,
            max_key,
            reverse,
            #[cfg(feature = "wisckey")]
            value_log,
        );
        let state = Box::pin(DbIteratorInner::next(inner));

        Self { state: Some(state) }
    }
}

impl Stream for DbIterator {
    type Item = (Key, EntryRef);

    /// Polls for the next key-value pair.
    ///
    /// Returns `Poll::Ready(Some((key, entry)))` when an item is available,
    /// `Poll::Ready(None)` when the iterator is exhausted, or `Poll::Pending`
    /// if the next item is still being computed.
    ///
    /// Deletion markers (tombstones) are filtered out and not returned to the user.
    fn poll_next(mut self: Pin<&mut Self>, ctx: &mut Context) -> Poll<Option<Self::Item>> {
        let (inner, res) = if let Some(mut fut) = self.state.take() {
            match Future::poll(fut.as_mut(), ctx) {
                // return and keep waiting for result
                Poll::Pending => {
                    self.state = Some(fut);
                    return Poll::Pending;
                }
                // item computation complete
                Poll::Ready(result) => {
                    let (inner, res) = result.expect("iteration failed");
                    (inner, res)
                }
            }
        } else {
            // no items left
            return Poll::Ready(None);
        };

        // Prepare next state?
        if res.is_some() {
            self.state = Some(Box::pin(DbIteratorInner::next(inner)));
        } else {
            self.state = None;
        }

        // return item
        Poll::Ready(res)
    }
}

/// Internal iterator state for k-way merge algorithm.
///
/// Maintains all source iterators and tracks the current position in the merge.
struct DbIteratorInner {
    /// The last key that was returned, used to skip duplicate keys.
    last_key: Option<Vec<u8>>,
    /// All source iterators (memtables and SSTables).
    iterators: Vec<Box<dyn InternalIterator>>,

    /// Whether to iterate in reverse (descending) order.
    reverse: bool,

    /// Optional minimum key bound.
    min_key: Option<Vec<u8>>,
    /// Optional maximum key bound.
    max_key: Option<Vec<u8>>,

    /// Value log for fetching values (only with "wisckey" feature).
    #[cfg(feature = "wisckey")]
    value_log: Arc<ValueLog>,
}

/// Tracks the current candidate for the next key-value pair.
///
/// Contains the sequence number and iterator index of the current best candidate
/// during the k-way merge.
type NextKV = Option<(crate::manifest::SeqNumber, usize)>;

impl DbIteratorInner {
    /// Creates a new internal iterator state.
    ///
    /// Combines all memtable and SSTable iterators into a single collection
    /// for k-way merging.
    ///
    /// # Arguments
    /// * `mem_iters` - Iterators for in-memory memtables
    /// * `table_iters` - Iterators for on-disk SSTables
    /// * `min_key` - Optional minimum key bound
    /// * `max_key` - Optional maximum key bound
    /// * `reverse` - Whether to iterate in reverse order
    /// * `value_log` - Value log reference (only with "wisckey" feature)
    fn new(
        mem_iters: Vec<MemtableIterator>,
        table_iters: Vec<TableIterator>,
        min_key: Option<Vec<u8>>,
        max_key: Option<Vec<u8>>,
        reverse: bool,
        #[cfg(feature = "wisckey")] value_log: Arc<ValueLog>,
    ) -> Self {
        let mut iterators: Vec<Box<dyn InternalIterator>> = vec![];
        for iter in mem_iters.into_iter() {
            iterators.push(Box::new(iter));
        }
        for iter in table_iters.into_iter() {
            iterators.push(Box::new(iter));
        }

        Self {
            iterators,
            last_key: None,
            min_key,
            max_key,
            reverse,
            #[cfg(feature = "wisckey")]
            value_log,
        }
    }

    /// Examines an iterator and determines if it should be the next candidate.
    ///
    /// This is the core of the k-way merge algorithm. For each source iterator,
    /// this method:
    /// 1. Advances past keys that have already been returned
    /// 2. Checks that the key is within the specified bounds
    /// 3. Compares with the current best candidate
    /// 4. Selects the iterator with the appropriate key (min/max) and highest sequence number
    ///
    /// # Arguments
    /// * `pos` - Index of the iterator to examine
    /// * `next_kv` - Current best candidate (sequence number and iterator index)
    ///
    /// # Returns
    /// A tuple of (should_replace, new_candidate) where:
    /// - `should_replace`: true if this iterator has a better candidate
    /// - `new_candidate`: the updated candidate if should_replace is true
    async fn parse_iter(&mut self, pos: usize, next_kv: NextKV) -> (bool, NextKV) {
        // Split slices to make the borrow checker happy
        let (prev, cur) = self.iterators[..].split_at_mut(pos);
        let iter = &mut *cur[0];

        if self.reverse {
            // This iterator might be "behind" other iterators
            if let Some(last_key) = &self.last_key {
                while !iter.at_end() && iter.get_key() >= last_key.as_slice() {
                    iter.step().await;
                }
            }

            // Don't pick a key that is greater than the maximum
            if let Some(max_key) = &self.max_key {
                while !iter.at_end() && iter.get_key() > max_key.as_slice() {
                    iter.step().await;
                }

                // There might be no key in this iterator that is <=max_key
                if iter.at_end() || iter.get_key() > max_key.as_slice() {
                    return (false, next_kv);
                }
            }

            if iter.at_end() {
                return (false, next_kv);
            }

            let key = iter.get_key();

            // Don't pick a key that is less or equal to the minimum
            if let Some(min_key) = &self.min_key
                && iter.get_key() <= min_key.as_slice()
            {
                return (false, next_kv);
            }

            let seq_number = iter.get_seq_number();

            if let Some((max_seq_number, max_pos)) = next_kv {
                let max_iter = &*prev[max_pos];
                let max_key = max_iter.get_key();

                match key.cmp(max_key) {
                    Ordering::Greater => (true, Some((seq_number, pos))),
                    Ordering::Equal => {
                        if seq_number > max_seq_number {
                            (true, Some((seq_number, pos)))
                        } else {
                            (false, next_kv)
                        }
                    }
                    Ordering::Less => (false, next_kv),
                }
            } else {
                (true, Some((seq_number, pos)))
            }
        } else {
            // This iterator might be "behind" other iterators
            if let Some(last_key) = &self.last_key {
                while !iter.at_end() && iter.get_key() <= last_key.as_slice() {
                    iter.step().await;
                }
            }

            // Don't pick a key that is smaller than the minimum
            if let Some(min_key) = &self.min_key {
                while !iter.at_end() && iter.get_key() < min_key.as_slice() {
                    iter.step().await;
                }

                // There might be no key in this iterator that is >=min_key
                if iter.at_end() || iter.get_key() < min_key.as_slice() {
                    return (false, next_kv);
                }
            }

            if iter.at_end() {
                return (false, next_kv);
            }

            let key = iter.get_key();

            // Don't pick a key that is greater or equal to the maximum
            if let Some(max_key) = &self.max_key
                && iter.get_key() >= max_key.as_slice()
            {
                return (false, next_kv);
            }

            let seq_number = iter.get_seq_number();

            if let Some((min_seq_number, min_pos)) = next_kv {
                let min_iter = &*prev[min_pos];
                let min_key = min_iter.get_key();

                match key.cmp(min_key) {
                    Ordering::Less => (true, Some((seq_number, pos))),
                    Ordering::Equal => {
                        if seq_number > min_seq_number {
                            (true, Some((seq_number, pos)))
                        } else {
                            (false, next_kv)
                        }
                    }
                    Ordering::Greater => (false, next_kv),
                }
            } else {
                (true, Some((seq_number, pos)))
            }
        }
    }

    /// Computes the next key-value pair in the iteration.
    ///
    /// Performs the k-way merge by examining all source iterators and selecting
    /// the next key according to the sort order. Skips over deletion markers
    /// (tombstones) and continues until a valid entry is found or all iterators
    /// are exhausted.
    ///
    /// # Returns
    /// Returns the updated iterator state and optionally the next key-value pair.
    /// Returns `None` when the iteration is complete.
    async fn next(mut self) -> Result<(Self, Option<(Key, EntryRef)>), Error> {
        let mut result = None;

        while result.is_none() {
            let mut next_kv = None;
            let num_iterators = self.iterators.len();

            for pos in 0..num_iterators {
                let (change, kv) = self.parse_iter(pos, next_kv).await;

                if change {
                    next_kv = kv;
                }
            }

            if let Some((_, pos)) = next_kv.take() {
                let iter = &*self.iterators[pos];

                let res_key = iter.get_key();
                self.last_key = Some(iter.get_key().to_vec());

                #[cfg(feature = "wisckey")]
                let entry = iter.get_entry(&self.value_log).await;
                #[cfg(not(feature = "wisckey"))]
                let entry = iter.get_entry();

                if let Some(entry) = entry {
                    result = Some(Some((res_key.to_vec(), entry)));
                } else {
                    // this is a deletion... skip
                }
            } else {
                // at end
                result = Some(None);
            };
        }

        let (key, result) = match result.unwrap() {
            Some(inner) => inner,
            None => {
                return Ok((self, None));
            }
        };

        Ok((self, Some((key, result))))
    }
}
