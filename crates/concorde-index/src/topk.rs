//! Disjunctive document-at-a-time top-k with block-max skipping.
//!
//! The pruning structure follows RISE (MIT, github.com/AngeloSav/rise-rs @ 3b178df): lists kept
//! sorted by frontier docid, whole-total block-max skipping, and a bounded heap whose top is the
//! worst kept hit. This module computes no scores: each [`QueryList`] brings a constant weight,
//! a tf→part table and per-block upper bounds from the layer above, and contributions are summed
//! as given (saturating at `u32::MAX`).
//!
//! One round either evaluates the smallest frontier docid (every list sitting on it contributes,
//! consuming its posting in one [`PostingReader::next_step`]) or, when the heap is full and the
//! sum of every live list's frontier-block bound cannot beat its worst hit, moves every list to
//! the earliest next block start: no document before it can qualify, since each list contributes
//! at most its frontier block's bound to such a document. The loop allocates nothing.

// Every round runs these helpers once per list and posting; left to the inliner, the driver keeps
// some as calls and passes their `Result`s through memory.
#![expect(
    clippy::inline_always,
    reason = "measured: the driver's round must inline whole"
)]

use crate::block::{docid_block, next_block_start};
use crate::error::CorruptPostings;
use crate::posting::PostingReader;
use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::{Ordering, Reverse};
use core::fmt;

/// The contribution of a posting by its term frequency, built by the scoring layer. Lists that
/// store no frequencies report tf 1.
#[derive(Clone, PartialEq, Eq)]
pub struct TfParts([u32; 256]);

impl TfParts {
    /// A table whose entry `tf` is the part a posting of that frequency adds.
    #[must_use]
    pub const fn new(parts: [u32; 256]) -> Self {
        Self(parts)
    }

    /// The part a posting of frequency `tf` adds.
    #[must_use]
    pub fn part(&self, tf: u8) -> u32 {
        self.0.get(usize::from(tf)).copied().unwrap_or(0)
    }
}

impl fmt::Debug for TfParts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TfParts").finish_non_exhaustive()
    }
}

/// One term of a disjunctive query: its postings and what each contributes.
#[derive(Debug)]
pub struct QueryList<'a, R> {
    reader: R,
    weight: u32,
    block_bounds: &'a [u32],
    tf_parts: &'a TfParts,
    /// The reader's frontier, cached so sorting never calls the reader.
    frontier: Option<u32>,
}

impl<'a, R: PostingReader> QueryList<'a, R> {
    /// A list whose postings each contribute `weight` plus `tf_parts`' entry for their tf.
    ///
    /// `block_bounds[b]` bounds the contribution of any posting in docid block `b`
    /// ([`docid_block`](crate::docid_block())); a block past the table's end is unbounded, which
    /// only disables pruning there.
    #[must_use]
    pub const fn new(
        reader: R,
        weight: u32,
        block_bounds: &'a [u32],
        tf_parts: &'a TfParts,
    ) -> Self {
        Self {
            reader,
            weight,
            block_bounds,
            tf_parts,
            frontier: None,
        }
    }

    /// The list's reader, positioned wherever the query left it.
    pub const fn reader(&self) -> &R {
        &self.reader
    }

    /// The cached frontier docid; `None` once the list is consumed.
    #[inline(always)]
    const fn frontier(&self) -> Option<u32> {
        self.frontier
    }

    /// Postings in the whole list.
    fn postings(&self) -> usize {
        usize::try_from(self.reader.len()).unwrap_or(usize::MAX)
    }

    /// The first docid of the block after the frontier's.
    #[inline(always)]
    fn next_block(&self) -> Option<u32> {
        self.frontier.and_then(next_block_start)
    }

    #[inline(always)]
    fn refresh(&mut self) -> Result<(), CorruptPostings> {
        self.frontier = self.reader.peek()?;
        Ok(())
    }

    /// The bound on any contribution in `docid`'s block.
    #[inline(always)]
    fn bound_at(&self, docid: u32) -> u32 {
        self.block_bounds
            .get(docid_block(docid))
            .copied()
            .unwrap_or(u32::MAX)
    }

    /// Consumes the frontier posting and returns its contribution.
    #[inline(always)]
    fn take(&mut self) -> Result<u32, CorruptPostings> {
        let part = self
            .reader
            .next_step()?
            .map_or(0, |posting| self.tf_parts.part(posting.tf()));
        self.refresh()?;
        Ok(self.weight.saturating_add(part))
    }

    #[inline(always)]
    fn skip_to(&mut self, target: u32) -> Result<(), CorruptPostings> {
        self.frontier = self.reader.advance(target)?;
        Ok(())
    }
}

/// A scored document. Hits order by score, then by lower docid: the greater hit ranks first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Hit {
    score: u32,
    docid: u32,
}

impl Hit {
    /// The sum of the contributions of the lists that matched.
    #[must_use]
    pub const fn score(self) -> u32 {
        self.score
    }

    /// The document.
    #[must_use]
    pub const fn docid(self) -> u32 {
        self.docid
    }
}

impl Ord for Hit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .cmp(&other.score)
            .then_with(|| other.docid.cmp(&self.docid))
    }
}

impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The `k` best documents matching any of `lists`, best first (score descending, then docid
/// ascending). `lists` is left in an unspecified order, each reader wherever the query left it.
///
/// # Errors
///
/// [`CorruptPostings`] when a list cannot be decoded.
pub fn topk_disjunctive<R: PostingReader>(
    lists: &mut [QueryList<'_, R>],
    k: usize,
) -> Result<Vec<Hit>, CorruptPostings> {
    if k == 0 {
        return Ok(Vec::new());
    }
    // Sized once, by what the lists can produce rather than by `k` alone.
    let postings = lists
        .iter()
        .fold(0usize, |sum, list| sum.saturating_add(list.postings()));
    let mut heap: BinaryHeap<Reverse<Hit>> = BinaryHeap::with_capacity(k.min(postings));
    for list in lists.iter_mut() {
        list.refresh()?;
    }
    sort_by_frontier(lists);
    loop {
        let live = lists.partition_point(|list| list.frontier().is_none());
        let Some(active) = lists.get_mut(live..).filter(|active| !active.is_empty()) else {
            break;
        };
        match prune(active, threshold(&heap, k))? {
            Round::Done => break,
            Round::Skipped => sort_by_frontier(lists),
            Round::Evaluate => {
                let Some(hit) = evaluate(active)? else {
                    break;
                };
                sort_by_frontier(lists);
                admit(&mut heap, k, hit);
            }
        }
    }
    let mut hits: Vec<Hit> = heap.into_iter().map(|Reverse(hit)| hit).collect();
    hits.sort_unstable_by_key(|&hit| Reverse(hit));
    Ok(hits)
}

/// What a candidate must score to enter the heap.
#[derive(Clone, Copy, Debug)]
enum Threshold {
    /// The heap has room: any candidate enters.
    Open,
    /// The heap is full: a candidate must score at least this.
    Required(u32),
    /// The worst kept hit scores `u32::MAX`: nothing can beat it.
    Unbeatable,
}

#[inline(always)]
fn threshold(heap: &BinaryHeap<Reverse<Hit>>, k: usize) -> Threshold {
    match heap.peek() {
        // Candidates arrive in docid order, so a tie with the worst hit loses: a candidate must
        // score strictly more.
        Some(Reverse(worst)) if heap.len() >= k => worst
            .score
            .checked_add(1)
            .map_or(Threshold::Unbeatable, Threshold::Required),
        Some(_) | None => Threshold::Open,
    }
}

/// What one round of the driver does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Round {
    /// Score the smallest frontier docid.
    Evaluate,
    /// Every list moved past blocks that cannot qualify; re-sort and go again.
    Skipped,
    /// No remaining document can enter the heap.
    Done,
}

/// Skips every list of `active` to the earliest next block start when no document before it
/// can meet `threshold`.
#[inline(always)]
fn prune<R: PostingReader>(
    active: &mut [QueryList<'_, R>],
    threshold: Threshold,
) -> Result<Round, CorruptPostings> {
    let required = match threshold {
        Threshold::Open => return Ok(Round::Evaluate),
        Threshold::Unbeatable => return Ok(Round::Done),
        Threshold::Required(required) => required,
    };
    if can_reach(active, required) {
        return Ok(Round::Evaluate);
    }
    // Every list in the last block of the docid space leaves nothing to skip to.
    let Some(jump) = active.iter().filter_map(QueryList::next_block).min() else {
        return Ok(Round::Done);
    };
    for list in active.iter_mut() {
        list.skip_to(jump)?;
    }
    Ok(Round::Skipped)
}

/// Whether the frontier-block bounds of `active` can sum to `required`.
#[inline(always)]
fn can_reach<R: PostingReader>(active: &[QueryList<'_, R>], required: u32) -> bool {
    let mut total = 0u32;
    for list in active {
        if let Some(frontier) = list.frontier() {
            total = total.saturating_add(list.bound_at(frontier));
        }
        if total >= required {
            return true;
        }
    }
    false
}

/// Scores the smallest frontier docid of `active` (sorted, all live), consuming the postings of
/// every list sitting on it.
#[inline(always)]
fn evaluate<R: PostingReader>(
    active: &mut [QueryList<'_, R>],
) -> Result<Option<Hit>, CorruptPostings> {
    let Some(docid) = active.first().and_then(QueryList::frontier) else {
        return Ok(None);
    };
    let mut score = 0u32;
    for list in active.iter_mut() {
        if list.frontier() != Some(docid) {
            break;
        }
        score = score.saturating_add(list.take()?);
    }
    Ok(Some(Hit { score, docid }))
}

/// Keeps `hit` when the heap has room or it beats the worst kept hit.
#[inline(always)]
fn admit(heap: &mut BinaryHeap<Reverse<Hit>>, k: usize, hit: Hit) {
    if heap.len() < k {
        heap.push(Reverse(hit));
    } else if let Some(mut worst) = heap.peek_mut()
        && hit > worst.0
    {
        *worst = Reverse(hit);
    }
}

/// Restores ascending frontier order (exhausted lists first) with one stable insertion pass:
/// after every round the lists are nearly sorted, so the pass is close to linear.
#[inline(always)]
fn sort_by_frontier<R: PostingReader>(lists: &mut [QueryList<'_, R>]) {
    for i in 1..lists.len() {
        let mut j = i;
        while let Some(before) = j.checked_sub(1)
            && sort_key(lists, before) > sort_key(lists, j)
        {
            lists.swap(before, j);
            j = before;
        }
    }
}

/// The frontier of list `index` as a sort key: exhausted is 0, docid `d` is `d + 1`.
#[inline(always)]
fn sort_key<R: PostingReader>(lists: &[QueryList<'_, R>], index: usize) -> u64 {
    lists
        .get(index)
        .and_then(QueryList::frontier)
        .map_or(0, |docid| u64::from(docid).saturating_add(1))
}
