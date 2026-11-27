# concorde-index

The physical layer of an inverted index: posting lists, and a top-k driver that ranks the
documents matching any of several lists. It keeps three promises:

1. **Compact lists, cheap jumps.** A list is delta varints with an optional term-frequency byte
   per posting, followed by a bi-level skip trailer (one entry per 128 postings, one per 4096).
   Sequential reads never touch the trailer; `advance` binary-searches it and decodes linearly
   inside one block.
2. **Pruned ranking.** `topk_disjunctive` evaluates documents in docid order and, when no
   document before the next block boundary can beat the current top k, moves every list past
   it at once (block-max skipping after RISE). The loop allocates nothing.
3. **No panics.** An encoder's precondition is an `EncodeError`; a damaged list is a
   `CorruptPostings` at the first read that meets it, never a wrong posting.

The crate knows storage facts only. Docids and term frequencies are opaque integers; weights,
the tf→part table and block bounds come from the scoring layer above, and contributions are
summed as given. Analyzers live elsewhere (for Japanese, see
[morph-dict](https://github.com/perforate-org/morph-dict)). The crate is `no_std` + `alloc`.

## Use

```rust
use concorde_index::{
    FreqVarintReader, QueryList, TfParts, docid_block, encode_freq_varint, topk_disjunctive,
};

// Index time: one list per term.
let rust = encode_freq_varint(&[1, 4, 900], &[2, 1, 5])?;
let db = encode_freq_varint(&[4, 7], &[1, 3])?;

// Bounds per docid block (`docid_block`): the largest contribution any posting there can make.
let rust_bounds = [3, 0, 0, 0, 0, 0, 0, 6];
let db_bounds = [4];

// Query time: a weight per term and a tf→part table from the scoring layer.
let parts = TfParts::new(core::array::from_fn(|tf| tf as u32));
let mut lists = [
    QueryList::new(FreqVarintReader::new(&rust)?, 1, &rust_bounds, &parts),
    QueryList::new(FreqVarintReader::new(&db)?, 1, &db_bounds, &parts),
];
let hits = topk_disjunctive(&mut lists, 2)?;
assert_eq!((hits[0].docid(), hits[0].score()), (900, 6));
assert_eq!((hits[1].docid(), hits[1].score()), (4, 4));
```

A reader that filters another (deleted documents, a candidate set) implements `PostingReader`
and forwards `advance` and `next_step` to the inner reader, so skipping stays the codec's.

## Where things are written down

The code is the one source of truth. Each module's documentation (`cargo doc
--document-private-items`) draws its byte layout and gives the reasons for its choices;
`crates/concorde-index/tests/public-api.txt` pins the public API, and `tests/postings.rs` pins the
wire layout byte for byte.

## Development

`just gate` runs fmt, clippy, tests, docs, deslop-please and qlty; `just kani` the proofs and
`just mutants` mutation testing.

## Origin and license

Extracted from [Gleaph](https://github.com/perforate-org/gleaph)'s `ic-stable-text-postings`;
the layout is unchanged, so lists written by either read with the other.

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT)
at your option.
