//! The top-k driver against exhaustive scoring: the same hits in the same order, whatever the
//! bounds, and fewer postings decoded when the bounds are tight.

#![expect(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "test helpers over small known inputs; a broken helper should fail the test"
)]

use concorde_index::{
    CorruptPostings, FreqVarintReader, Hit, Posting, PostingReader, QueryList, TfParts,
    docid_block, encode_freq_varint, topk_disjunctive,
};
use proptest::prelude::*;
extern crate alloc;

use alloc::collections::BTreeMap;

/// One query term: postings with frequencies, and its constant weight.
#[derive(Clone, Debug)]
struct Term {
    docids: Vec<u32>,
    tfs: Vec<u32>,
    weight: u32,
}

impl Term {
    fn contribution(&self, tf: u32, parts: &TfParts) -> u32 {
        self.weight
            .saturating_add(parts.part(u8::try_from(tf).unwrap_or(u8::MAX)))
    }

    /// Tight per-docid-block bounds: the largest contribution in each block.
    fn tight_bounds(&self, parts: &TfParts) -> Vec<u32> {
        let mut bounds = Vec::new();
        for (&docid, &tf) in self.docids.iter().zip(&self.tfs) {
            let block = docid_block(docid);
            if bounds.len() <= block {
                bounds.resize(block + 1, 0);
            }
            bounds[block] = bounds[block].max(self.contribution(tf, parts));
        }
        bounds
    }
}

/// Every matching document scored exhaustively, best first, cut to `k`.
fn oracle(terms: &[Term], parts: &TfParts, k: usize) -> Vec<(u32, u32)> {
    let mut scores: BTreeMap<u32, u32> = BTreeMap::new();
    for term in terms {
        for (&docid, &tf) in term.docids.iter().zip(&term.tfs) {
            let score = scores.entry(docid).or_default();
            *score = score.saturating_add(term.contribution(tf, parts));
        }
    }
    let mut ranked: Vec<(u32, u32)> = scores.into_iter().map(|(d, s)| (s, d)).collect();
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    ranked.truncate(k);
    ranked
}

/// Counts the postings a reader decodes on the driver's behalf.
struct Counting<R> {
    inner: R,
    consumed: usize,
}

impl<R: PostingReader> PostingReader for Counting<R> {
    fn len(&self) -> u32 {
        self.inner.len()
    }

    fn pos(&self) -> u32 {
        self.inner.pos()
    }

    fn peek(&mut self) -> Result<Option<u32>, CorruptPostings> {
        self.inner.peek()
    }

    fn tf(&mut self) -> Result<Option<u8>, CorruptPostings> {
        self.inner.tf()
    }

    fn next_step(&mut self) -> Result<Option<Posting>, CorruptPostings> {
        let step = self.inner.next_step()?;
        self.consumed += usize::from(step.is_some());
        Ok(step)
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, CorruptPostings> {
        self.inner.advance(target)
    }
}

/// How the lists' block bounds are supplied.
#[derive(Clone, Copy, Debug)]
enum Bounds {
    Tight,
    Loose,
    Missing,
}

/// Runs the driver and returns its hits and the postings it scored.
fn drive(terms: &[Term], parts: &TfParts, k: usize, bounds: Bounds) -> (Vec<(u32, u32)>, usize) {
    let encoded: Vec<Vec<u8>> = terms
        .iter()
        .map(|t| encode_freq_varint(&t.docids, &t.tfs).unwrap())
        .collect();
    let tables: Vec<Vec<u32>> = terms
        .iter()
        .map(|t| match bounds {
            Bounds::Tight => t.tight_bounds(parts),
            Bounds::Loose => vec![u32::MAX; docid_block(*t.docids.last().unwrap()) + 1],
            Bounds::Missing => Vec::new(),
        })
        .collect();
    let mut lists: Vec<_> = terms
        .iter()
        .zip(&encoded)
        .zip(&tables)
        .map(|((term, bytes), table)| {
            let reader = Counting {
                inner: FreqVarintReader::new(bytes).unwrap(),
                consumed: 0,
            };
            QueryList::new(reader, term.weight, table, parts)
        })
        .collect();
    let hits = topk_disjunctive(&mut lists, k).unwrap();
    let consumed = lists.iter().map(|l| l.reader().consumed).sum();
    (
        hits.iter().map(|h| (h.score(), h.docid())).collect(),
        consumed,
    )
}

fn identity() -> TfParts {
    TfParts::new(core::array::from_fn(|tf| u32::try_from(tf).unwrap()))
}

fn term(docids: Vec<u32>, tf: u32, weight: u32) -> Term {
    let tfs = vec![tf; docids.len()];
    Term {
        docids,
        tfs,
        weight,
    }
}

#[test]
fn scores_sum_across_lists_and_ties_prefer_low_docids() {
    let parts = identity();
    let terms = [
        term(vec![1, 2, 3, 500], 1, 10),
        term(vec![2, 3, 900], 2, 10),
        term(vec![3, 901], 3, 10),
    ];
    let (hits, _) = drive(&terms, &parts, 3, Bounds::Tight);
    assert_eq!(hits, [(36, 3), (23, 2), (13, 901)]);
    assert_eq!(
        drive(&terms, &parts, 10, Bounds::Tight).0,
        oracle(&terms, &parts, 10)
    );
}

#[test]
fn tight_bounds_skip_blocks_that_cannot_qualify() {
    let parts = identity();
    // A dense list whose heavy documents sit in five blocks, and a second list inside the first block. Once the
    // heap holds the heavy documents, every light block's bound falls short and is skipped.
    let dense: Vec<u32> = (0..50_000).collect();
    let tfs: Vec<u32> = dense
        .iter()
        .map(|&d| if d % 10_000 == 0 { 200 } else { 1 })
        .collect();
    let terms = [
        Term {
            docids: dense,
            tfs,
            weight: 0,
        },
        term(vec![1, 2, 3], 3, 0),
    ];
    let (tight, consumed) = drive(&terms, &parts, 5, Bounds::Tight);
    assert_eq!(tight, oracle(&terms, &parts, 5));
    let (loose, exhaustive) = drive(&terms, &parts, 5, Bounds::Loose);
    assert_eq!(loose, tight);
    assert_eq!(exhaustive, 50_003);
    assert!(
        consumed * 20 < exhaustive,
        "{consumed} of {exhaustive} decoded"
    );
}

#[test]
fn edge_cases() {
    let parts = identity();
    let terms = [term(vec![1, 2], 1, 1)];
    assert_eq!(drive(&terms, &parts, 0, Bounds::Tight).0, []);
    let mut none: Vec<QueryList<'_, FreqVarintReader<'_>>> = Vec::new();
    assert_eq!(topk_disjunctive(&mut none, 3).unwrap(), Vec::<Hit>::new());
    // Saturated scores: nothing beats a full heap of u32::MAX, and the scan stops early.
    let terms = [
        term(vec![1, 2, 3, 1000, 2000], 1, u32::MAX),
        term(vec![1, 3], 1, u32::MAX),
    ];
    assert_eq!(
        drive(&terms, &parts, 2, Bounds::Tight).0,
        [(u32::MAX, 1), (u32::MAX, 2)]
    );
    // Lists in the last block of the docid space.
    let terms = [
        term(vec![u32::MAX - 3, u32::MAX], 1, 1),
        term(vec![u32::MAX - 2], 1, 5),
    ];
    assert_eq!(
        drive(&terms, &parts, 1, Bounds::Tight).0,
        oracle(&terms, &parts, 1)
    );
}

#[test]
fn hits_order_by_score_then_lower_docid() {
    let parts = identity();
    let hits = topk_disjunctive(
        &mut [QueryList::new(
            FreqVarintReader::new(&encode_freq_varint(&[4, 9], &[3, 7]).unwrap()).unwrap(),
            0,
            &[],
            &parts,
        )],
        2,
    )
    .unwrap();
    assert!(hits[0] > hits[1]);
    assert_eq!((hits[0].docid(), hits[0].score()), (9, 7));
    assert_eq!(parts.part(255), 255);
    assert!(format!("{parts:?}").starts_with("TfParts"));
}

#[test]
fn damaged_lists_are_errors() {
    let parts = identity();
    let mut bytes = encode_freq_varint(&[3, 9], &[1, 1]).unwrap();
    bytes[6] = 0; // the second gap
    let mut lists = [QueryList::new(
        FreqVarintReader::new(&bytes).unwrap(),
        1,
        &[],
        &parts,
    )];
    assert_eq!(
        topk_disjunctive(&mut lists, 5),
        Err(CorruptPostings::BadGap { position: 1 })
    );
}

fn terms() -> impl Strategy<Value = Vec<Term>> {
    let term = (
        proptest::collection::vec((1u32..300, 0u32..300), 1..800),
        0u32..50,
    )
        .prop_map(|(steps, weight)| {
            let mut docid = 0u32;
            let (docids, tfs) = steps
                .iter()
                .map(|&(gap, tf)| {
                    docid += gap;
                    (docid, tf)
                })
                .unzip();
            Term {
                docids,
                tfs,
                weight,
            }
        });
    proptest::collection::vec(term, 1..6)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    #[test]
    fn driver_matches_exhaustive_scoring(
        terms in terms(),
        k in 1usize..40,
        parts in proptest::collection::vec(0u32..100, 256),
    ) {
        let parts = TfParts::new(parts.try_into().unwrap());
        let expected = oracle(&terms, &parts, k);
        for bounds in [Bounds::Tight, Bounds::Loose, Bounds::Missing] {
            prop_assert_eq!(&drive(&terms, &parts, k, bounds).0, &expected, "{:?}", bounds);
        }
    }
}
