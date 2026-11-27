//! The bi-level skip trailer every encoded list ends with.
//!
//! Level 0 has one entry per [`LOGICAL_BLOCK_SIZE`]-posting block; level 1 has one per
//! [`STRIDE`] level-0 blocks. An entry holds the last docid of the span it covers and the
//! absolute byte offset where the span starts; with term frequencies it also holds the span's
//! posting count (written for the layout's sake, never needed to jump: blocks are positional, so
//! block `b` starts at posting `b × 128`). `advance` binary-searches level 1, then level 0, and
//! decodes linearly inside the landing block.
//!
//! ```text
//! level-0 entries:  block_count × entry
//! level-1 entries:  ceil(block_count / 32) × entry
//! block_count:      u32 LE   // = ceil(count / 128); the buffer's final four bytes
//! entry:            last_docid u32 LE, start_offset u32 LE [, posting_count u32 LE]
//! ```
//!
//! A level-1 entry carries the last docid of its last block, the start offset of its first, and
//! their summed count. The count trailing the buffer lets a reader size both levels from the end.

use super::Layout;
use crate::block::{LOGICAL_BLOCK_SIZE, block_start, blocks_for};
use crate::error::{CorruptPostings, EncodeError};
use alloc::vec::Vec;

/// Level-0 blocks per level-1 entry: `32 × 128 = 4096` postings.
const STRIDE: usize = 32;

/// One span of postings as the trailer records it.
#[derive(Clone, Copy, Debug)]
struct Span {
    last_docid: u32,
    start_offset: u32,
    postings: u32,
}

impl Span {
    fn write(self, out: &mut Vec<u8>, layout: Layout) {
        out.extend_from_slice(&self.last_docid.to_le_bytes());
        out.extend_from_slice(&self.start_offset.to_le_bytes());
        match layout {
            Layout::DocidsOnly => {}
            Layout::WithTf => out.extend_from_slice(&self.postings.to_le_bytes()),
        }
    }

    /// The spans of `blocks` taken `STRIDE` at a time.
    fn coalesce(blocks: &[Self]) -> impl Iterator<Item = Self> + '_ {
        blocks.chunks(STRIDE).filter_map(|stride| {
            let (first, last) = (stride.first()?, stride.last()?);
            Some(Self {
                last_docid: last.last_docid,
                start_offset: first.start_offset,
                postings: stride
                    .iter()
                    .fold(0, |sum, block| sum.saturating_add(block.postings)),
            })
        })
    }
}

/// Collects the level-0 blocks while a list is encoded.
#[derive(Debug, Default)]
pub(super) struct SkipBuilder {
    closed: Vec<Span>,
    open: Option<Span>,
}

impl SkipBuilder {
    /// Records the posting at `position` (0-based), which starts at byte `offset` of the output.
    pub(super) fn posting(
        &mut self,
        position: usize,
        offset: usize,
        docid: u32,
    ) -> Result<(), EncodeError> {
        let block_size = usize::try_from(LOGICAL_BLOCK_SIZE).map_err(|_| EncodeError::TooLarge)?;
        if position.is_multiple_of(block_size) {
            self.closed.extend(self.open.take());
            self.open = Some(Span {
                last_docid: docid,
                start_offset: u32::try_from(offset).map_err(|_| EncodeError::TooLarge)?,
                postings: 0,
            });
        }
        if let Some(open) = &mut self.open {
            open.last_docid = docid;
            open.postings = open.postings.saturating_add(1);
        }
        Ok(())
    }

    /// Appends the trailer to `out`, which holds the whole payload.
    pub(super) fn finish(mut self, out: &mut Vec<u8>, layout: Layout) -> Result<(), EncodeError> {
        self.closed.extend(self.open.take());
        let count = u32::try_from(self.closed.len()).map_err(|_| EncodeError::TooLarge)?;
        for block in &self.closed {
            block.write(out, layout);
        }
        for stride in Span::coalesce(&self.closed) {
            stride.write(out, layout);
        }
        out.extend_from_slice(&count.to_le_bytes());
        if u32::try_from(out.len()).is_err() {
            return Err(EncodeError::TooLarge);
        }
        Ok(())
    }
}

/// Where a jump lands: the start of the first block that can hold the target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Landing {
    offset: usize,
    position: u32,
    prev: u32,
}

impl Landing {
    /// Absolute byte offset of the block's first posting.
    pub(super) const fn offset(self) -> usize {
        self.offset
    }

    /// Postings before the block.
    pub(super) const fn position(self) -> u32 {
        self.position
    }

    /// The docid before the block: the base its first gap is added to.
    pub(super) const fn prev(self) -> u32 {
        self.prev
    }
}

/// A parsed view of one list's trailer.
#[derive(Clone, Copy, Debug)]
pub(super) struct SkipIndex {
    level0: usize,
    blocks: usize,
    strides: usize,
    entry_bytes: usize,
}

impl SkipIndex {
    /// Parses the trailer of `data`, a list of `count` postings, and returns it with the byte
    /// offset where the payload ends.
    pub(super) fn parse(
        data: &[u8],
        count: u32,
        layout: Layout,
    ) -> Result<(Self, usize), CorruptPostings> {
        let total = data.len();
        let truncated = CorruptPostings::Truncated { len: total };
        let tail = total.checked_sub(4).ok_or(truncated)?;
        let found = read_u32(data, tail).ok_or(truncated)?;
        let expected = blocks_for(count);
        if found != expected {
            return Err(CorruptPostings::BlockCount { found, expected });
        }
        let blocks = usize::try_from(found).map_err(|_| CorruptPostings::TrailerOverlap)?;
        let strides = blocks.div_ceil(STRIDE);
        let entry_bytes = layout.entry_bytes();
        let level0 = blocks
            .checked_add(strides)
            .and_then(|entries| entries.checked_mul(entry_bytes))
            .and_then(|bytes| tail.checked_sub(bytes))
            .filter(|&start| start >= 4)
            .ok_or(CorruptPostings::TrailerOverlap)?;
        Ok((
            Self {
                level0,
                blocks,
                strides,
                entry_bytes,
            },
            level0,
        ))
    }

    /// The landing for `target`: the first block whose last docid is at least `target`. `None`
    /// when every block ends below it (the list holds no such docid).
    pub(super) fn landing(
        &self,
        data: &[u8],
        target: u32,
    ) -> Result<Option<Landing>, CorruptPostings> {
        let stride = first_reaching(0, self.strides, target, |i| {
            self.last_docid(data, self.blocks.checked_add(i)?)
        })?;
        let first = stride.saturating_mul(STRIDE).min(self.blocks);
        let last = first.saturating_add(STRIDE).min(self.blocks);
        let block = first_reaching(first, last, target, |i| self.last_docid(data, i))?;
        if block >= self.blocks {
            return Ok(None);
        }
        let offset = self
            .entry(data, block, 4)
            .ok_or(CorruptPostings::TrailerOverlap)?;
        let prev = match block.checked_sub(1) {
            Some(before) => self
                .last_docid(data, before)
                .ok_or(CorruptPostings::TrailerOverlap)?,
            None => 0,
        };
        let position = u32::try_from(block)
            .ok()
            .and_then(block_start)
            .ok_or(CorruptPostings::TrailerOverlap)?;
        Ok(Some(Landing {
            offset: usize::try_from(offset).map_err(|_| CorruptPostings::TrailerOverlap)?,
            position,
            prev,
        }))
    }

    /// The last docid of entry `index` (level 0, then level 1, contiguous).
    fn last_docid(&self, data: &[u8], index: usize) -> Option<u32> {
        self.entry(data, index, 0)
    }

    /// The word at byte `field` of entry `index`.
    fn entry(&self, data: &[u8], index: usize, field: usize) -> Option<u32> {
        let at = index
            .checked_mul(self.entry_bytes)?
            .checked_add(self.level0)?
            .checked_add(field)?;
        read_u32(data, at)
    }
}

/// The first index in `lo..hi` whose entry's last docid reaches `target` (`hi` when none does),
/// given that entries ascend.
fn first_reaching(
    mut lo: usize,
    mut hi: usize,
    target: u32,
    last_docid: impl Fn(usize) -> Option<u32>,
) -> Result<usize, CorruptPostings> {
    while lo < hi {
        let mid = lo.midpoint(hi);
        if last_docid(mid).ok_or(CorruptPostings::TrailerOverlap)? < target {
            lo = mid.saturating_add(1);
        } else {
            hi = mid;
        }
    }
    Ok(lo)
}

fn read_u32(data: &[u8], at: usize) -> Option<u32> {
    let bytes = data.get(at..at.checked_add(4)?)?;
    let word: [u8; 4] = bytes.try_into().ok()?;
    Some(u32::from_le_bytes(word))
}

/// Proofs over every input (`cargo kani`).
#[cfg(kani)]
mod proofs {
    use super::*;

    /// Parsing any short buffer under any posting count is total, and a parsed trailer leaves
    /// the header and its own entries inside the buffer.
    #[kani::proof]
    fn parse_is_total() {
        let data: [u8; 32] = kani::any();
        let len: usize = kani::any();
        kani::assume(len <= data.len());
        let count: u32 = kani::any();
        let layout = if kani::any() {
            Layout::DocidsOnly
        } else {
            Layout::WithTf
        };
        if let Ok((index, payload_end)) = SkipIndex::parse(&data[..len], count, layout) {
            assert!(payload_end >= 4);
            let entries = (index.blocks + index.strides) * index.entry_bytes;
            assert!(payload_end + entries + 4 == len);
        }
    }
}
