//! Logical blocks: the 128-wide granularity shared by the skip index and block-max bounds.
//!
//! Two block spaces use the same width. The skip index blocks a list by *position* (postings
//! `0..128`, `128..256`, ...); block-max bounds block the *docid space* (docids `0..128`, ...),
//! so a bound table is indexed by [`docid_block`] whatever the list's density.

/// Postings per skip block, and docids per block-max bound.
pub const LOGICAL_BLOCK_SIZE: u32 = 1 << BLOCK_SHIFT;

/// `log2(LOGICAL_BLOCK_SIZE)`: blocks are found by shifting, never by dividing.
const BLOCK_SHIFT: u32 = 7;

/// The docid block holding `docid`: its index in a block-max bound table.
#[must_use]
pub fn docid_block(docid: u32) -> usize {
    // Lossless wherever `usize` has 32 bits; a narrower target saturates past every table.
    usize::try_from(docid >> BLOCK_SHIFT).unwrap_or(usize::MAX)
}

/// The first docid of the block after `docid`'s, or `None` in the last block of the docid
/// space.
pub(crate) const fn next_block_start(docid: u32) -> Option<u32> {
    match (docid >> BLOCK_SHIFT).checked_add(1) {
        Some(next) => next.checked_mul(LOGICAL_BLOCK_SIZE),
        None => None,
    }
}

/// Skip blocks covering a list of `postings` postings.
pub(crate) const fn blocks_for(postings: u32) -> u32 {
    postings.div_ceil(LOGICAL_BLOCK_SIZE)
}

/// The position of the first posting of skip block `block`, or `None` past `u32::MAX`.
pub(crate) const fn block_start(block: u32) -> Option<u32> {
    block.checked_mul(LOGICAL_BLOCK_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_arithmetic() {
        assert_eq!(docid_block(0), 0);
        assert_eq!(docid_block(127), 0);
        assert_eq!(docid_block(128), 1);
        assert_eq!(docid_block(u32::MAX), 0x01FF_FFFF);
        assert_eq!(next_block_start(0), Some(128));
        assert_eq!(next_block_start(127), Some(128));
        assert_eq!(next_block_start(128), Some(256));
        assert_eq!(next_block_start(u32::MAX - 128), Some(u32::MAX - 127));
        assert_eq!(next_block_start(u32::MAX - 127), None);
        assert_eq!(blocks_for(0), 0);
        assert_eq!(blocks_for(1), 1);
        assert_eq!(blocks_for(128), 1);
        assert_eq!(blocks_for(129), 2);
        assert_eq!(block_start(2), Some(256));
        assert_eq!(block_start(u32::MAX), None);
    }
}

/// Proofs over every input (`cargo kani`).
#[cfg(kani)]
mod proofs {
    use super::*;

    /// The next block starts after `docid`, on a block boundary, one block later.
    #[kani::proof]
    fn next_block_start_is_the_following_boundary() {
        let docid: u32 = kani::any();
        match next_block_start(docid) {
            Some(start) => {
                assert!(start > docid);
                assert!(start % LOGICAL_BLOCK_SIZE == 0);
                assert!(start - docid <= LOGICAL_BLOCK_SIZE);
                assert!(docid_block(start) == docid_block(docid) + 1);
            }
            None => assert!(docid_block(docid) == docid_block(u32::MAX)),
        }
    }
}
