//! Delta-varint posting lists, with or without term frequencies.
//!
//! ```text
//! count:        u32 LE                       // postings in the list
//! per posting:  gap as LEB128 u32            // the first posting's docid, then the difference
//!               [tf as one u8]               // frequency layout only; capped at 255
//! skip trailer: see `skip`                   // frequency entries add the block's posting count
//! ```
//!
//! Gaps are strictly positive, so a decoded list is strictly increasing. Sequential reads never
//! touch the trailer; [`PostingReader::advance`] uses it to jump to the block that can hold its
//! target. Every read is bounded by the payload's end, so trailer bytes are never decoded as
//! postings.

mod skip;

use crate::error::{CorruptPostings, EncodeError};
use crate::posting::{Posting, PostingReader};
use alloc::vec::Vec;
use skip::{SkipBuilder, SkipIndex};

/// Encodes strictly increasing docids as a [`VarintReader`] list.
///
/// # Errors
///
/// [`EncodeError`] when `docids` is empty, does not strictly increase, or would exceed the
/// layout's 32-bit offsets.
pub fn encode_varint(docids: &[u32]) -> Result<Vec<u8>, EncodeError> {
    encode(docids, Layout::DocidsOnly, |_| Ok(1))
}

/// Encodes strictly increasing docids with their term frequencies as a [`FreqVarintReader`] list.
/// Frequencies above 255 are stored as 255.
///
/// # Errors
///
/// [`EncodeError`] when `docids` is empty, does not strictly increase, does not pair one-to-one
/// with `tfs`, or would exceed the layout's 32-bit offsets.
pub fn encode_freq_varint(docids: &[u32], tfs: &[u32]) -> Result<Vec<u8>, EncodeError> {
    if docids.len() != tfs.len() {
        return Err(EncodeError::Misaligned {
            docids: docids.len(),
            frequencies: tfs.len(),
        });
    }
    encode(docids, Layout::WithTf, |index| {
        let tf = tfs.get(index).copied().ok_or(EncodeError::Misaligned {
            docids: docids.len(),
            frequencies: tfs.len(),
        })?;
        Ok(u8::try_from(tf).unwrap_or(u8::MAX))
    })
}

fn encode(
    docids: &[u32],
    layout: Layout,
    tf_at: impl Fn(usize) -> Result<u8, EncodeError>,
) -> Result<Vec<u8>, EncodeError> {
    let count = u32::try_from(docids.len()).map_err(|_| EncodeError::TooLarge)?;
    if count == 0 {
        return Err(EncodeError::Empty);
    }
    let mut out = Vec::with_capacity(docids.len().saturating_mul(6).saturating_add(4));
    out.extend_from_slice(&count.to_le_bytes());
    let mut skips = SkipBuilder::default();
    let mut prev = None;
    for (index, &docid) in docids.iter().enumerate() {
        let gap = match prev {
            None => docid,
            Some(prev) => docid
                .checked_sub(prev)
                .filter(|&gap| gap > 0)
                .ok_or(EncodeError::NotIncreasing { index })?,
        };
        skips.posting(index, out.len(), docid)?;
        write_leb128(&mut out, gap);
        match layout {
            Layout::DocidsOnly => {}
            Layout::WithTf => out.push(tf_at(index)?),
        }
        prev = Some(docid);
    }
    skips.finish(&mut out, layout)?;
    Ok(out)
}

fn write_leb128(out: &mut Vec<u8>, mut value: u32) {
    loop {
        let low = value.to_le_bytes().first().map_or(0, |&byte| byte & 0x7F);
        value >>= 7;
        if value == 0 {
            out.push(low);
            return;
        }
        out.push(low | 0x80);
    }
}

/// The byte at `offset` of `data`, which must lie below `end`.
fn byte_at(data: &[u8], offset: usize, end: usize) -> Result<u8, CorruptPostings> {
    data.get(offset)
        .copied()
        .filter(|_| offset < end)
        .ok_or(CorruptPostings::PayloadTruncated { offset })
}

/// The LEB128 `u32` at `offset` of `data`, read below `end`, and the offset after it.
fn read_leb128(data: &[u8], offset: usize, end: usize) -> Result<(u32, usize), CorruptPostings> {
    let mut value = 0u32;
    let mut at = offset;
    for shift in [0, 7, 14, 21, 28] {
        let byte = byte_at(data, at, end)?;
        at = at.saturating_add(1);
        let bits = u32::from(byte & 0x7F);
        // The fifth byte carries the top four bits and ends the varint.
        if shift == 28 && byte > 0x0F {
            break;
        }
        value |= bits << shift;
        if byte & 0x80 == 0 {
            return Ok((value, at));
        }
    }
    Err(CorruptPostings::VarintOverflow { offset })
}

/// Whether postings carry a term frequency byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    DocidsOnly,
    WithTf,
}

impl Layout {
    /// Bytes per skip entry.
    const fn entry_bytes(self) -> usize {
        match self {
            Self::DocidsOnly => 8,
            Self::WithTf => 12,
        }
    }
}

/// The decoding state both layouts share.
#[derive(Clone, Debug)]
struct Cursor<'a> {
    data: &'a [u8],
    layout: Layout,
    count: u32,
    /// The next byte to decode.
    offset: usize,
    payload_end: usize,
    /// Postings consumed.
    pos: u32,
    /// The docid decoded last: the base of the next gap.
    prev: u32,
    /// The frontier, once decoded.
    frontier: Option<Posting>,
    skip: SkipIndex,
}

impl<'a> Cursor<'a> {
    fn open(data: &'a [u8], layout: Layout) -> Result<Self, CorruptPostings> {
        let count = data
            .first_chunk::<4>()
            .map(|&header| u32::from_le_bytes(header))
            .ok_or(CorruptPostings::Truncated { len: data.len() })?;
        let (skip, payload_end) = SkipIndex::parse(data, count, layout)?;
        Ok(Self {
            data,
            layout,
            count,
            offset: 4,
            payload_end,
            pos: 0,
            prev: 0,
            frontier: None,
            skip,
        })
    }

    fn byte(&mut self) -> Result<u8, CorruptPostings> {
        let byte = byte_at(self.data, self.offset, self.payload_end)?;
        self.offset = self.offset.saturating_add(1);
        Ok(byte)
    }

    fn leb128(&mut self) -> Result<u32, CorruptPostings> {
        let (value, next) = read_leb128(self.data, self.offset, self.payload_end)?;
        self.offset = next;
        Ok(value)
    }

    /// The frontier, decoding it if needed; `None` once the list is consumed.
    fn frontier(&mut self) -> Result<Option<Posting>, CorruptPostings> {
        if self.frontier.is_none() && self.pos < self.count {
            let gap = self.leb128()?;
            let docid = if self.pos == 0 {
                Some(gap)
            } else {
                self.prev.checked_add(gap).filter(|_| gap > 0)
            };
            let docid = docid.ok_or(CorruptPostings::BadGap { position: self.pos })?;
            let tf = match self.layout {
                Layout::DocidsOnly => 1,
                Layout::WithTf => self.byte()?,
            };
            self.prev = docid;
            self.frontier = Some(Posting::new(docid, tf));
        }
        Ok(self.frontier)
    }

    fn next_step(&mut self) -> Result<Option<Posting>, CorruptPostings> {
        let posting = self.frontier()?;
        if posting.is_some() {
            self.frontier = None;
            self.pos = self.pos.saturating_add(1);
        }
        Ok(posting)
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, CorruptPostings> {
        match self.frontier()? {
            None => return Ok(None),
            Some(posting) if posting.docid() >= target => return Ok(Some(posting.docid())),
            Some(_) => self.jump(target)?,
        }
        while let Some(posting) = self.frontier()? {
            if posting.docid() >= target {
                return Ok(Some(posting.docid()));
            }
            let _skipped = self.next_step()?;
        }
        Ok(None)
    }

    /// Jumps to the block that can hold `target` when it lies ahead of the cursor.
    fn jump(&mut self, target: u32) -> Result<(), CorruptPostings> {
        if let Some(landing) = self.skip.landing(self.data, target)?
            && landing.offset() > self.offset
        {
            self.offset = landing.offset();
            self.pos = landing.position();
            self.prev = landing.prev();
            self.frontier = None;
        }
        Ok(())
    }
}

/// Implements [`PostingReader`] for a newtype over [`Cursor`].
macro_rules! posting_reader {
    ($reader:ident) => {
        impl PostingReader for $reader<'_> {
            fn len(&self) -> u32 {
                self.0.count
            }

            fn pos(&self) -> u32 {
                self.0.pos
            }

            fn peek(&mut self) -> Result<Option<u32>, CorruptPostings> {
                Ok(self.0.frontier()?.map(Posting::docid))
            }

            fn tf(&mut self) -> Result<Option<u8>, CorruptPostings> {
                Ok(self.0.frontier()?.map(Posting::tf))
            }

            fn next_step(&mut self) -> Result<Option<Posting>, CorruptPostings> {
                self.0.next_step()
            }

            fn advance(&mut self, target: u32) -> Result<Option<u32>, CorruptPostings> {
                self.0.advance(target)
            }
        }
    };
}

/// A cursor over a list from [`encode_varint`]; every posting reports term frequency 1.
#[derive(Clone, Debug)]
pub struct VarintReader<'a>(Cursor<'a>);

impl<'a> VarintReader<'a> {
    /// Opens `data`, reading its header and skip trailer; postings decode lazily.
    ///
    /// # Errors
    ///
    /// [`CorruptPostings`] when the header or trailer is malformed.
    pub fn new(data: &'a [u8]) -> Result<Self, CorruptPostings> {
        Cursor::open(data, Layout::DocidsOnly).map(Self)
    }
}

posting_reader!(VarintReader);

/// A cursor over a list from [`encode_freq_varint`].
#[derive(Clone, Debug)]
pub struct FreqVarintReader<'a>(Cursor<'a>);

impl<'a> FreqVarintReader<'a> {
    /// Opens `data`, reading its header and skip trailer; postings decode lazily.
    ///
    /// # Errors
    ///
    /// [`CorruptPostings`] when the header or trailer is malformed.
    pub fn new(data: &'a [u8]) -> Result<Self, CorruptPostings> {
        Cursor::open(data, Layout::WithTf).map(Self)
    }
}

posting_reader!(FreqVarintReader);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leb128_round_trips_at_every_width() {
        for value in [0, 1, 127, 128, 16_383, 16_384, 1 << 21, 1 << 28, u32::MAX] {
            let mut out = Vec::new();
            write_leb128(&mut out, value);
            assert_eq!(read_leb128(&out, 0, out.len()), Ok((value, out.len())));
            // One byte short of the end is a truncation.
            assert_eq!(
                read_leb128(&out, 0, out.len() - 1),
                Err(CorruptPostings::PayloadTruncated {
                    offset: out.len() - 1
                })
            );
        }
    }
}

/// Proofs over every input (`cargo kani`).
#[cfg(kani)]
mod proofs {
    use super::*;

    /// Every `u32` round-trips through LEB128 in at most five bytes.
    #[kani::proof]
    #[kani::unwind(7)]
    fn leb128_round_trips() {
        let value: u32 = kani::any();
        let mut out = Vec::new();
        write_leb128(&mut out, value);
        assert!(out.len() <= 5);
        assert!(read_leb128(&out, 0, out.len()) == Ok((value, out.len())));
    }

    /// Decoding any five bytes is total, and a success consumes what it reports.
    #[kani::proof]
    #[kani::unwind(7)]
    fn leb128_decoding_is_total() {
        let bytes: [u8; 5] = kani::any();
        let end: usize = kani::any();
        kani::assume(end <= 5);
        if let Ok((_, next)) = read_leb128(&bytes, 0, end) {
            assert!(next >= 1 && next <= end);
        }
    }
}
