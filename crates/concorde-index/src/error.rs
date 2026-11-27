//! Errors: what can be wrong with the input to an encoder, and with an encoded list.
//!
//! Encoded lists come only from this crate's encoders, so a decoding error means the bytes were
//! damaged or written by another layout. Readers report it at the first offending read instead of
//! yielding wrong postings.

use core::fmt;

/// Why a docid list cannot be encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeError {
    /// The list is empty; an encoded list holds at least one posting.
    Empty,
    /// The docid at `index` is not greater than the one before it.
    NotIncreasing {
        /// The offending position.
        index: usize,
    },
    /// The term frequencies do not pair one-to-one with the docids.
    Misaligned {
        /// Docids given.
        docids: usize,
        /// Term frequencies given.
        frequencies: usize,
    },
    /// The encoded list would not fit the 32-bit counts and offsets of its layout.
    TooLarge,
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("posting list is empty"),
            Self::NotIncreasing { index } => {
                write!(f, "docid at index {index} does not increase")
            }
            Self::Misaligned {
                docids,
                frequencies,
            } => write!(f, "{frequencies} term frequencies for {docids} docids"),
            Self::TooLarge => f.write_str("posting list exceeds 32-bit offsets"),
        }
    }
}

impl core::error::Error for EncodeError {}

/// A structural fault in an encoded posting list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CorruptPostings {
    /// Shorter than the count header and the smallest skip trailer.
    Truncated {
        /// The buffer's length in bytes.
        len: usize,
    },
    /// The skip trailer's block count is not the one the posting count implies.
    BlockCount {
        /// The count the trailer stores.
        found: u32,
        /// The count the header implies.
        expected: u32,
    },
    /// The skip trailer is larger than the buffer, or runs into the header.
    TrailerOverlap,
    /// A varint or term frequency runs past the end of the payload.
    PayloadTruncated {
        /// The payload offset of the read.
        offset: usize,
    },
    /// A varint encodes more than 32 bits.
    VarintOverflow {
        /// The payload offset of the varint.
        offset: usize,
    },
    /// A gap is zero or carries the docid past `u32::MAX`.
    BadGap {
        /// The position of the posting.
        position: u32,
    },
}

impl fmt::Display for CorruptPostings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { len } => write!(f, "posting list of {len} bytes is truncated"),
            Self::BlockCount { found, expected } => write!(
                f,
                "skip trailer holds {found} blocks; the header implies {expected}"
            ),
            Self::TrailerOverlap => f.write_str("skip trailer overlaps the payload"),
            Self::PayloadTruncated { offset } => {
                write!(f, "posting payload truncated at byte {offset}")
            }
            Self::VarintOverflow { offset } => {
                write!(f, "varint at byte {offset} exceeds 32 bits")
            }
            Self::BadGap { position } => {
                write!(f, "gap before posting {position} is zero or overflows")
            }
        }
    }
}

impl core::error::Error for CorruptPostings {}
