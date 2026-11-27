//! The codecs against the unencoded model: round trips, cursor moves, the wire layout, the
//! encoder's preconditions and damaged input.

#![expect(
    clippy::arithmetic_side_effects,
    clippy::unwrap_used,
    reason = "test helpers over small known inputs; a broken helper should fail the test"
)]

use concorde_index::{
    CorruptPostings, EncodeError, FreqVarintReader, PlainReader, Posting, PostingReader,
    VarintReader, encode_freq_varint, encode_varint,
};
use proptest::prelude::*;

/// Shapes that stress the layout: one posting, block and stride boundaries, wide gaps and the
/// top of the docid space.
fn shapes() -> Vec<Vec<u32>> {
    vec![
        vec![7],
        vec![0],
        vec![3, 9],
        (100..=355).collect(),
        (0..128).collect(),
        (0..129).collect(),
        (0..4096).map(|i| i * 3).collect(),
        (0..4097).map(|i| i * 3).collect(),
        (0..10_000).map(|i| i * 7 + 1).collect(),
        vec![0, 5_000, 90_000, 4_000_000, 2_000_000_000],
        vec![u32::MAX - 2, u32::MAX - 1, u32::MAX],
        vec![0, u32::MAX],
    ]
}

fn tfs_for(docids: &[u32]) -> Vec<u32> {
    docids.iter().map(|&d| d % 300).collect()
}

fn drain(reader: &mut impl PostingReader) -> Result<Vec<Posting>, CorruptPostings> {
    let mut out = Vec::new();
    while let Some(posting) = reader.next_step()? {
        out.push(posting);
    }
    Ok(out)
}

fn capped(tf: u32) -> u8 {
    u8::try_from(tf).unwrap_or(u8::MAX)
}

#[test]
fn round_trips_every_shape() {
    for docids in shapes() {
        let plain = encode_varint(&docids).unwrap();
        let mut reader = VarintReader::new(&plain).unwrap();
        assert_eq!(reader.len(), u32::try_from(docids.len()).unwrap());
        let expected: Vec<Posting> = docids.iter().map(|&d| Posting::new(d, 1)).collect();
        assert_eq!(drain(&mut reader).unwrap(), expected);
        assert!(reader.is_exhausted());
        assert_eq!(reader.next().unwrap(), None);

        let tfs = tfs_for(&docids);
        let freq = encode_freq_varint(&docids, &tfs).unwrap();
        let mut reader = FreqVarintReader::new(&freq).unwrap();
        let expected: Vec<Posting> = docids
            .iter()
            .zip(&tfs)
            .map(|(&d, &tf)| Posting::new(d, capped(tf)))
            .collect();
        assert_eq!(drain(&mut reader).unwrap(), expected);
        assert_eq!(reader.tf().unwrap(), None);
    }
}

#[test]
fn wire_layout_is_pinned() {
    // count, gaps 3 and 6, one level-0 and one level-1 entry {last 9, offset 4}, block count.
    let plain: Vec<u8> = [
        &[2, 0, 0, 0, 3, 6][..],
        &[9, 0, 0, 0, 4, 0, 0, 0],
        &[9, 0, 0, 0, 4, 0, 0, 0],
        &[1, 0, 0, 0],
    ]
    .concat();
    assert_eq!(encode_varint(&[3, 9]).unwrap(), plain);
    // Each gap is followed by its tf (300 capped at 255); entries add the posting count.
    let freq: Vec<u8> = [
        &[2, 0, 0, 0, 3, 1, 6, 255][..],
        &[9, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0],
        &[9, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0],
        &[1, 0, 0, 0],
    ]
    .concat();
    assert_eq!(encode_freq_varint(&[3, 9], &[1, 300]).unwrap(), freq);
    // A multi-byte varint: 300 = 0b10_0101100.
    assert_eq!(encode_varint(&[300]).unwrap()[4..6], [0xAC, 0x02]);
}

#[test]
fn encoder_preconditions_are_errors() {
    assert_eq!(encode_varint(&[]), Err(EncodeError::Empty));
    assert_eq!(
        encode_varint(&[1, 2, 2]),
        Err(EncodeError::NotIncreasing { index: 2 })
    );
    assert_eq!(
        encode_varint(&[5, 4]),
        Err(EncodeError::NotIncreasing { index: 1 })
    );
    assert_eq!(
        encode_freq_varint(&[1, 2], &[1]),
        Err(EncodeError::Misaligned {
            docids: 2,
            frequencies: 1
        })
    );
    assert_eq!(encode_freq_varint(&[], &[]), Err(EncodeError::Empty));
    for error in [
        EncodeError::Empty,
        EncodeError::NotIncreasing { index: 1 },
        EncodeError::TooLarge,
    ] {
        assert!(!error.to_string().is_empty());
    }
}

#[test]
fn malformed_headers_and_trailers_fail_closed() {
    let good = encode_varint(&(0..300).collect::<Vec<u32>>()).unwrap();
    assert_eq!(
        VarintReader::new(&good[..3]).unwrap_err(),
        CorruptPostings::Truncated { len: 3 }
    );
    assert_eq!(
        VarintReader::new(&good[..6]).unwrap_err(),
        CorruptPostings::BlockCount {
            found: u32::from_le_bytes([good[2], good[3], good[4], good[5]]),
            expected: 3
        }
    );
    let mut wrong_count = good.clone();
    let tail = wrong_count.len() - 4;
    wrong_count[tail] = 2;
    assert_eq!(
        VarintReader::new(&wrong_count).unwrap_err(),
        CorruptPostings::BlockCount {
            found: 2,
            expected: 3
        }
    );
    // A header claiming more postings than the trailer leaves room for.
    let mut huge = good;
    huge[..4].copy_from_slice(&u32::MAX.to_le_bytes());
    let blocks = u32::MAX.div_ceil(128).to_le_bytes();
    huge[tail..].copy_from_slice(&blocks);
    assert_eq!(
        VarintReader::new(&huge).unwrap_err(),
        CorruptPostings::TrailerOverlap
    );
    // A trailer that is the plain layout read with the frequency layout's entry width.
    let plain = encode_varint(&(0..5000).collect::<Vec<u32>>()).unwrap();
    assert!(
        FreqVarintReader::new(&plain).is_err()
            || drain(&mut FreqVarintReader::new(&plain).unwrap()).is_err()
    );
}

#[test]
fn damaged_payloads_fail_at_the_first_bad_read() {
    // Gap 0 after the first posting.
    let mut zero_gap = encode_varint(&[3, 9]).unwrap();
    zero_gap[5] = 0;
    let mut reader = VarintReader::new(&zero_gap).unwrap();
    assert_eq!(reader.next().unwrap(), Some(3));
    assert_eq!(reader.next(), Err(CorruptPostings::BadGap { position: 1 }));
    // A gap carrying the docid past u32::MAX.
    let mut overflow = encode_varint(&[u32::MAX - 1, u32::MAX]).unwrap();
    overflow[9] = 2;
    let mut reader = VarintReader::new(&overflow).unwrap();
    assert_eq!(reader.next().unwrap(), Some(u32::MAX - 1));
    assert_eq!(reader.peek(), Err(CorruptPostings::BadGap { position: 1 }));
    // A varint longer than 32 bits: the fifth byte may carry four bits and no continuation.
    let mut long = encode_varint(&[u32::MAX]).unwrap();
    long[8] = 0x1F;
    assert_eq!(
        VarintReader::new(&long).unwrap().next(),
        Err(CorruptPostings::VarintOverflow { offset: 4 })
    );
    long[8] = 0x8F;
    assert_eq!(
        VarintReader::new(&long).unwrap().next(),
        Err(CorruptPostings::VarintOverflow { offset: 4 })
    );
    long[8] = 0x0F;
    assert_eq!(VarintReader::new(&long).unwrap().next(), Ok(Some(u32::MAX)));
    // A continuation bit on the last payload byte runs into the trailer.
    let mut runaway = encode_varint(&[3, 9]).unwrap();
    runaway[5] |= 0x80;
    let mut reader = VarintReader::new(&runaway).unwrap();
    assert_eq!(reader.next().unwrap(), Some(3));
    assert_eq!(
        reader.next(),
        Err(CorruptPostings::PayloadTruncated { offset: 6 })
    );
    // A term frequency byte missing at the payload's end.
    let freq = encode_freq_varint(&[3], &[1]).unwrap();
    let mut short = freq[..4].to_vec();
    short.push(3);
    short.extend_from_slice(&freq[6..]);
    let mut reader = FreqVarintReader::new(&short).unwrap();
    assert_eq!(
        reader.tf(),
        Err(CorruptPostings::PayloadTruncated { offset: 5 })
    );
    for error in [
        CorruptPostings::Truncated { len: 1 },
        CorruptPostings::TrailerOverlap,
        CorruptPostings::BadGap { position: 1 },
    ] {
        assert!(!error.to_string().is_empty());
    }
}

#[test]
fn advance_jumps_with_the_skip_index() {
    let docids: Vec<u32> = (0..20_000).map(|i| i * 5).collect();
    let bytes = encode_varint(&docids).unwrap();
    let mut reader = VarintReader::new(&bytes).unwrap();
    assert_eq!(reader.advance(0).unwrap(), Some(0));
    assert_eq!(reader.pos(), 0);
    assert_eq!(reader.advance(50_001).unwrap(), Some(50_005));
    assert_eq!(reader.pos(), 10_001);
    // A target at or below the frontier leaves it in place.
    assert_eq!(reader.advance(7).unwrap(), Some(50_005));
    assert_eq!(reader.pos(), 10_001);
    assert_eq!(reader.advance(99_995).unwrap(), Some(99_995));
    assert_eq!(reader.advance(99_996).unwrap(), None);
    assert!(reader.is_exhausted());
    assert_eq!(reader.advance(1).unwrap(), None);
}

#[test]
fn plain_reader_is_the_model() {
    let mut empty = PlainReader::new(&[]);
    assert!(empty.is_empty());
    assert!(empty.is_exhausted());
    assert_eq!(empty.peek().unwrap(), None);
    let docids = [2, 4, 8];
    let mut reader = PlainReader::new(&docids);
    assert!(!reader.is_empty());
    assert!(!reader.is_exhausted());
    assert_eq!(reader.pos(), 0);
    assert_eq!(reader.tf().unwrap(), Some(1));
    assert_eq!(reader.advance(3).unwrap(), Some(4));
    assert_eq!(reader.pos(), 1);
    assert_eq!(reader.next().unwrap(), Some(4));
    assert_eq!(reader.advance(9).unwrap(), None);
    assert_eq!(reader.tf().unwrap(), None);
    assert_eq!(reader.pos(), 3);
    assert!(reader.is_exhausted());
}

#[test]
fn advance_decodes_only_the_landing_block() {
    // Docids-only gaps of 3 are one byte each: posting `i` is byte `4 + i`, block `b` bytes
    // `4 + 128 b ..`. Blocks past the first are filled with overflowing varints except the
    // target's, so a jump that lands anywhere else, or a linear walk, reads damage.
    let docids: Vec<u32> = (1..=10_000).map(|i| i * 3).collect();
    let clean = encode_varint(&docids).unwrap();
    let blocks = docids.len().div_ceil(128);
    let payload = |block: usize| 4 + block * 128..(4 + (block + 1) * 128).min(4 + docids.len());
    for target_block in [1, 2, 31, 32, 33, 64, 77, blocks - 1] {
        let mut bytes = clean.clone();
        for block in (1..blocks).filter(|&b| b != target_block) {
            bytes[payload(block)].fill(0xFF);
        }
        let first = target_block * 128;
        let last = (first + 127).min(docids.len() - 1);
        for index in [first, first + 1, (first + 64).min(last), last] {
            for target in [docids[index] - 2, docids[index]] {
                let mut reader = VarintReader::new(&bytes).unwrap();
                assert_eq!(
                    reader.advance(target).unwrap(),
                    Some(docids[index]),
                    "block {target_block}, target {target}"
                );
                assert_eq!(reader.pos(), u32::try_from(index).unwrap());
            }
        }
    }
}

/// One cursor move.
#[derive(Clone, Debug)]
enum Move {
    Peek,
    Tf,
    Next,
    Advance(u32),
}

fn moves() -> impl Strategy<Value = Vec<Move>> {
    proptest::collection::vec(
        prop_oneof![
            Just(Move::Peek),
            Just(Move::Tf),
            Just(Move::Next),
            (0u32..2_000_000).prop_map(Move::Advance),
        ],
        0..64,
    )
}

/// Strictly increasing docids from random gaps.
fn docids() -> impl Strategy<Value = Vec<u32>> {
    proptest::collection::vec(1u32..400, 1..6000).prop_map(|gaps| {
        gaps.iter()
            .scan(0u32, |docid, &gap| {
                *docid += gap;
                Some(*docid)
            })
            .collect()
    })
}

/// The model's answer to `step`: a reader over `docids` with `tfs`, as a list of postings.
struct Model {
    postings: Vec<Posting>,
    pos: usize,
}

impl Model {
    fn apply(&mut self, step: &Move) -> (Option<u32>, u32) {
        let current = self.postings.get(self.pos).copied();
        let answer = match step {
            Move::Peek => current.map(Posting::docid),
            Move::Tf => current.map(|p| u32::from(p.tf())),
            Move::Next => {
                self.pos += usize::from(current.is_some());
                current.map(Posting::docid)
            }
            Move::Advance(target) => {
                while self
                    .postings
                    .get(self.pos)
                    .is_some_and(|p| p.docid() < *target)
                {
                    self.pos += 1;
                }
                self.postings.get(self.pos).map(|p| p.docid())
            }
        };
        (answer, u32::try_from(self.pos).unwrap())
    }
}

fn run(reader: &mut impl PostingReader, step: &Move) -> (Option<u32>, u32) {
    let answer = match step {
        Move::Peek => reader.peek().unwrap(),
        Move::Tf => reader.tf().unwrap().map(u32::from),
        Move::Next => reader.next().unwrap(),
        Move::Advance(target) => reader.advance(*target).unwrap(),
    };
    (answer, reader.pos())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn cursors_follow_the_model(docids in docids(), steps in moves()) {
        let tfs = tfs_for(&docids);
        let plain = encode_varint(&docids).unwrap();
        let freq = encode_freq_varint(&docids, &tfs).unwrap();
        let mut plain_reader = VarintReader::new(&plain).unwrap();
        let mut freq_reader = FreqVarintReader::new(&freq).unwrap();
        let mut plain_model = Model {
            postings: docids.iter().map(|&d| Posting::new(d, 1)).collect(),
            pos: 0,
        };
        let mut freq_model = Model {
            postings: docids.iter().zip(&tfs).map(|(&d, &tf)| Posting::new(d, capped(tf))).collect(),
            pos: 0,
        };
        for step in &steps {
            prop_assert_eq!(run(&mut plain_reader, step), plain_model.apply(step));
            prop_assert_eq!(run(&mut freq_reader, step), freq_model.apply(step));
        }
    }

    #[test]
    fn damaged_lists_never_panic(
        docids in docids(),
        flips in proptest::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..8),
        targets in proptest::collection::vec(0u32..2_000_000, 0..16),
    ) {
        let tfs = tfs_for(&docids);
        for mut bytes in [encode_varint(&docids).unwrap(), encode_freq_varint(&docids, &tfs).unwrap()] {
            for (index, byte) in &flips {
                let at = index.index(bytes.len());
                bytes[at] ^= byte;
            }
            // Whatever a damaged list yields before its first error still strictly increases.
            if let Ok(mut reader) = FreqVarintReader::new(&bytes) {
                check_increasing(&mut reader, &targets);
            }
            if let Ok(mut reader) = VarintReader::new(&bytes) {
                check_increasing(&mut reader, &targets);
            }
        }
    }

    #[test]
    fn arbitrary_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..256)) {
        if let Ok(mut reader) = VarintReader::new(&bytes) {
            check_increasing(&mut reader, &[]);
        }
        if let Ok(mut reader) = FreqVarintReader::new(&bytes) {
            check_increasing(&mut reader, &[5, 1000]);
        }
    }
}

fn check_increasing(reader: &mut impl PostingReader, targets: &[u32]) {
    let mut last = None;
    for &target in targets {
        match reader.advance(target) {
            Ok(Some(docid)) => {
                assert!(docid >= target);
                assert!(last.is_none_or(|l| docid >= l));
                last = Some(docid);
            }
            Ok(None) | Err(_) => return,
        }
    }
    while let Ok(Some(posting)) = reader.next_step() {
        assert!(last.is_none_or(|l| posting.docid() > l || Some(posting.docid()) == last));
        last = Some(posting.docid());
    }
}
