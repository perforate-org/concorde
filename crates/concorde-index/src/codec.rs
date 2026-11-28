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

// The cursor's per-posting path is a chain of small functions under a generic reader; left to the
// inliner, it stops partway and every posting pays for calls and `Result`s passed through memory.
#![expect(
    clippy::inline_always,
    reason = "measured: the decode path must inline whole"
)]
mod skip;

use crate::error::{CorruptPostings, EncodeError};
use crate::posting::{Posting, PostingReader};
use alloc::vec::Vec;
use core::marker::PhantomData;
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
/// The varint at the front of `window` and its length in bytes; `None` when it is longer than
/// five bytes or exceeds 32 bits, cases [`read_leb128`] reports.
#[inline(always)]
fn leb128_in(window: &[u8; 6]) -> Option<(u32, usize)> {
    let mut value = 0u32;
    for (shift, (len, &byte)) in [0u32, 7, 14, 21, 28].into_iter().zip((1..).zip(window)) {
        // The fifth byte carries the top four bits and ends the varint.
        if shift == 28 && byte > 0x0F {
            return None;
        }
        value |= u32::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return Some((value, len));
        }
    }
    None
}

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

/// A layout fixed at compile time, so the decoder never branches on it per posting.
trait Fixed {
    const LAYOUT: Layout;
}

/// [`Layout::DocidsOnly`] as a type.
#[derive(Clone, Copy, Debug)]
struct DocidsOnly;

impl Fixed for DocidsOnly {
    const LAYOUT: Layout = Layout::DocidsOnly;
}

/// [`Layout::WithTf`] as a type.
#[derive(Clone, Copy, Debug)]
struct WithTf;

impl Fixed for WithTf {
    const LAYOUT: Layout = Layout::WithTf;
}

/// The decoding state both layouts share.
#[derive(Clone, Debug)]
struct Cursor<'a, L> {
    data: &'a [u8],
    layout: PhantomData<L>,
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

impl<'a, L: Fixed> Cursor<'a, L> {
    fn open(data: &'a [u8]) -> Result<Self, CorruptPostings> {
        let count = data
            .first_chunk::<4>()
            .map(|&header| u32::from_le_bytes(header))
            .ok_or(CorruptPostings::Truncated { len: data.len() })?;
        let (skip, payload_end) = SkipIndex::parse(data, count, L::LAYOUT)?;
        Ok(Self {
            data,
            layout: PhantomData,
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
    #[inline(always)]
    fn frontier(&mut self) -> Result<Option<Posting>, CorruptPostings> {
        if self.frontier.is_none() && self.pos < self.count {
            self.frontier = Some(self.decode()?);
        }
        Ok(self.frontier)
    }

    #[inline(always)]
    fn next_step(&mut self) -> Result<Option<Posting>, CorruptPostings> {
        let posting = match self.frontier.take() {
            Some(posting) => posting,
            None if self.pos < self.count => self.decode()?,
            None => return Ok(None),
        };
        self.pos = self.pos.saturating_add(1);
        Ok(Some(posting))
    }

    /// Decodes the posting at the cursor, which the caller has checked exists.
    #[inline(always)]
    fn decode(&mut self) -> Result<Posting, CorruptPostings> {
        self.decode_fast().map_or_else(|| self.decode_checked(), Ok)
    }

    /// Decodes a posting held whole, and well formed, in the payload's next six bytes: a gap of
    /// at most five bytes and a term frequency. `None` leaves every other case, errors included,
    /// to [`Self::decode_checked`], which then decodes the same posting or reports the fault; the
    /// cursor is unchanged until the posting is known good.
    #[inline(always)]
    fn decode_fast(&mut self) -> Option<Posting> {
        let window = self
            .data
            .get(self.offset..self.payload_end)?
            .first_chunk::<6>()?;
        let (gap, mut used) = leb128_in(window)?;
        let docid = match self.pos {
            0 => gap,
            _ if gap == 0 => return None,
            _ => self.prev.checked_add(gap)?,
        };
        let tf = match L::LAYOUT {
            Layout::DocidsOnly => 1,
            Layout::WithTf => {
                let tf = *window.get(used)?;
                used = used.checked_add(1)?;
                tf
            }
        };
        self.offset = self.offset.checked_add(used)?;
        self.prev = docid;
        Some(Posting::new(docid, tf))
    }

    /// Decodes the posting at the cursor with every read checked, reporting the first fault.
    #[cold]
    #[inline(never)]
    fn decode_checked(&mut self) -> Result<Posting, CorruptPostings> {
        let gap = self.leb128()?;
        let docid = if self.pos == 0 {
            Some(gap)
        } else {
            self.prev.checked_add(gap).filter(|_| gap > 0)
        };
        let docid = docid.ok_or(CorruptPostings::BadGap { position: self.pos })?;
        let tf = match L::LAYOUT {
            Layout::DocidsOnly => 1,
            Layout::WithTf => self.byte()?,
        };
        self.prev = docid;
        Ok(Posting::new(docid, tf))
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

/// Defines a reader newtype over [`Cursor`] opened with a list layout, and implements
/// [`PostingReader`] for it.
macro_rules! posting_reader {
    ($(#[$doc:meta])* $reader:ident, $layout:ty) => {
        $(#[$doc])*
        #[derive(Clone, Debug)]
        pub struct $reader<'a>(Cursor<'a, $layout>);

        impl<'a> $reader<'a> {
            /// Opens `data`, reading its header and skip trailer; postings decode lazily.
            ///
            /// # Errors
            ///
            /// [`CorruptPostings`] when the header or trailer is malformed.
            pub fn new(data: &'a [u8]) -> Result<Self, CorruptPostings> {
                Cursor::open(data).map(Self)
            }
        }

        impl PostingReader for $reader<'_> {
            #[inline(always)]
            fn len(&self) -> u32 {
                self.0.count
            }

            #[inline(always)]
            fn pos(&self) -> u32 {
                self.0.pos
            }

            #[inline(always)]
            fn peek(&mut self) -> Result<Option<u32>, CorruptPostings> {
                Ok(self.0.frontier()?.map(Posting::docid))
            }

            #[inline(always)]
            fn tf(&mut self) -> Result<Option<u8>, CorruptPostings> {
                Ok(self.0.frontier()?.map(Posting::tf))
            }

            #[inline(always)]
            fn next_step(&mut self) -> Result<Option<Posting>, CorruptPostings> {
                self.0.next_step()
            }

            #[inline(always)]
            fn advance(&mut self, target: u32) -> Result<Option<u32>, CorruptPostings> {
                self.0.advance(target)
            }
        }
    };
}

posting_reader!(
    /// A cursor over a list from [`encode_varint`]; every posting reports term frequency 1.
    VarintReader,
    DocidsOnly
);

posting_reader!(
    /// A cursor over a list from [`encode_freq_varint`].
    FreqVarintReader,
    WithTf
);

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Decodes `bytes` posting by posting through both paths, from identical cursors, and checks
    /// they agree on every result and on where they leave the cursor, up to the first fault.
    fn paths_agree<L: Fixed + Clone>(bytes: &[u8]) {
        let Ok(mut fast) = Cursor::<L>::open(bytes) else {
            return;
        };
        let mut checked = fast.clone();
        while fast.pos < fast.count {
            let result = fast.decode();
            assert_eq!(result, checked.decode_checked());
            assert_eq!((fast.offset, fast.prev), (checked.offset, checked.prev));
            if result.is_err() {
                return;
            }
            fast.pos = fast.pos.saturating_add(1);
            checked.pos = checked.pos.saturating_add(1);
        }
    }

    proptest! {
        #[test]
        fn the_varint_fast_path_agrees_with_the_checked_read(window in any::<[u8; 6]>()) {
            let checked = read_leb128(&window, 0, window.len()).ok().filter(|&(_, len)| len <= 5);
            prop_assert_eq!(leb128_in(&window), checked);
        }

        #[test]
        fn fast_and_checked_decoding_agree_on_damaged_lists(
            gaps in proptest::collection::vec(1u32..1 << 20, 1..400),
            tfs in proptest::collection::vec(any::<u8>(), 400),
            damage in proptest::collection::vec((any::<usize>(), any::<u8>()), 0..4),
        ) {
            let docids: Vec<u32> = gaps
                .iter()
                .scan(0u32, |prev, &gap| {
                    *prev = prev.checked_add(gap)?;
                    Some(*prev)
                })
                .collect();
            let tfs: Vec<u32> = tfs.iter().take(docids.len()).map(|&tf| u32::from(tf)).collect();
            for mut bytes in [
                encode_freq_varint(&docids, &tfs).unwrap(),
                encode_varint(&docids).unwrap(),
            ] {
                for &(at, byte) in &damage {
                    let payload = bytes.len().saturating_sub(4);
                    let at = at.checked_rem(payload).unwrap().saturating_add(4);
                    bytes[at] = byte;
                }
                paths_agree::<WithTf>(&bytes);
                paths_agree::<DocidsOnly>(&bytes);
            }
        }
    }

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
