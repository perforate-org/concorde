//! The physical layer of an inverted index: posting lists and the top-k driver that reads them.
//!
//! A posting list is the strictly increasing docids of the documents a term occurs in, with the
//! term's frequency in each. [`encode_varint`] and [`encode_freq_varint`] write one as delta
//! varints followed by a bi-level skip trailer; [`VarintReader`] and [`FreqVarintReader`] read it
//! back through the [`PostingReader`] cursor, decoding lazily and jumping with the trailer.
//! [`topk_disjunctive`] ranks the documents matching any of several lists, skipping blocks whose
//! bounds cannot beat the current top k.
//!
//! ```
//! use concorde_index::{
//!     FreqVarintReader, QueryList, TfParts, docid_block, encode_freq_varint, topk_disjunctive,
//! };
//!
//! // Index time: one list per term.
//! let rust = encode_freq_varint(&[1, 4, 900], &[2, 1, 5])?;
//! let db = encode_freq_varint(&[4, 7], &[1, 3])?;
//!
//! // Bounds per docid block (`docid_block`): the largest contribution any posting there can make.
//! let rust_bounds = [3, 0, 0, 0, 0, 0, 0, 6];
//! let db_bounds = [4];
//!
//! // Query time: a weight per term and a tf→part table from the scoring layer.
//! let parts = TfParts::new(core::array::from_fn(|tf| tf as u32));
//! let mut lists = [
//!     QueryList::new(FreqVarintReader::new(&rust)?, 1, &rust_bounds, &parts),
//!     QueryList::new(FreqVarintReader::new(&db)?, 1, &db_bounds, &parts),
//! ];
//! let hits = topk_disjunctive(&mut lists, 2)?;
//! assert_eq!((hits[0].docid(), hits[0].score()), (900, 6));
//! assert_eq!((hits[1].docid(), hits[1].score()), (4, 4));
//! # Ok::<(), Box<dyn core::error::Error>>(())
//! ```
//!
//! The crate knows storage facts only. Docids and term frequencies are opaque integers; scores,
//! weights and block bounds arrive from the layer above, and analyzers live elsewhere.
//!
//! Library code never panics: an encoder's precondition is an [`EncodeError`], a damaged list a
//! [`CorruptPostings`] at the first read that meets it. The crate is `no_std` + `alloc`.

#![no_std]

extern crate alloc;

// Layers, each using only those before it: `error` and `block`, then `posting`, then `codec`
// and `topk`.
mod block;
mod codec;
mod error;
mod posting;
mod topk;

pub use block::{LOGICAL_BLOCK_SIZE, docid_block};
pub use codec::{FreqVarintReader, VarintReader, encode_freq_varint, encode_varint};
pub use error::{CorruptPostings, EncodeError};
pub use posting::{PlainReader, Posting, PostingReader};
pub use topk::{Hit, QueryList, TfParts, topk_disjunctive};
