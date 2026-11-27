//! The cursor contract every posting list honours, and the unencoded list.

use crate::error::CorruptPostings;

/// One posting: a document and the term's frequency in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Posting {
    docid: u32,
    tf: u8,
}

impl Posting {
    /// A posting of `docid` with term frequency `tf`.
    #[must_use]
    pub const fn new(docid: u32, tf: u8) -> Self {
        Self { docid, tf }
    }

    /// The document.
    #[must_use]
    pub const fn docid(self) -> u32 {
        self.docid
    }

    /// The term's frequency in the document (stored capped at 255).
    #[must_use]
    pub const fn tf(self) -> u8 {
        self.tf
    }
}

/// A forward-only cursor over one strictly increasing posting list.
///
/// The cursor's *frontier* is the next unconsumed posting. Every read either reports it, consumes
/// it, or moves it forward; it never moves back. Once the list is consumed every read reports
/// `Ok(None)`. A damaged encoding is an error at the first read that meets it.
///
/// Implement it to filter or wrap another reader (deleted documents, a candidate set): forward
/// [`Self::advance`] and [`Self::next_step`] to the inner reader's, so skipping and decoding stay
/// the codec's.
pub trait PostingReader {
    /// Postings in the whole list, consumed or not; known without decoding.
    fn len(&self) -> u32;

    /// Whether the list holds no postings at all.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Postings consumed so far: the frontier's position in the list.
    fn pos(&self) -> u32;

    /// Whether every posting has been consumed.
    fn is_exhausted(&self) -> bool {
        self.pos() >= self.len()
    }

    /// The frontier's docid, without consuming it.
    ///
    /// # Errors
    ///
    /// [`CorruptPostings`] when the frontier cannot be decoded.
    fn peek(&mut self) -> Result<Option<u32>, CorruptPostings>;

    /// The frontier's term frequency, without consuming it. Lists that store none report 1.
    ///
    /// # Errors
    ///
    /// [`CorruptPostings`] when the frontier cannot be decoded.
    fn tf(&mut self) -> Result<Option<u8>, CorruptPostings> {
        Ok(self.peek()?.map(|_| 1))
    }

    /// Consumes the frontier and returns its docid.
    ///
    /// # Errors
    ///
    /// [`CorruptPostings`] when the frontier cannot be decoded.
    fn next(&mut self) -> Result<Option<u32>, CorruptPostings> {
        Ok(self.next_step()?.map(Posting::docid))
    }

    /// Consumes the frontier and returns it whole: the hot loop's one call per posting.
    ///
    /// # Errors
    ///
    /// [`CorruptPostings`] when the frontier cannot be decoded.
    fn next_step(&mut self) -> Result<Option<Posting>, CorruptPostings>;

    /// Moves the frontier to the first posting whose docid is at least `target`, consuming every
    /// posting before it, and returns that docid; `None` when no remaining posting qualifies. A
    /// target at or below the frontier leaves it where it is.
    ///
    /// # Errors
    ///
    /// [`CorruptPostings`] when a posting on the way cannot be decoded.
    fn advance(&mut self, target: u32) -> Result<Option<u32>, CorruptPostings> {
        while let Some(docid) = self.peek()? {
            if docid >= target {
                return Ok(Some(docid));
            }
            let _consumed = self.next_step()?;
        }
        Ok(None)
    }
}

/// A cursor over an unencoded slice: the reference reader. Unlike an encoded list it may be
/// empty, and it trusts its slice to be strictly increasing.
#[derive(Clone, Debug)]
pub struct PlainReader<'a> {
    docids: &'a [u32],
    pos: usize,
}

impl<'a> PlainReader<'a> {
    /// A cursor at the start of `docids`, which must be strictly increasing.
    #[must_use]
    pub const fn new(docids: &'a [u32]) -> Self {
        Self { docids, pos: 0 }
    }
}

impl PostingReader for PlainReader<'_> {
    fn len(&self) -> u32 {
        u32::try_from(self.docids.len()).unwrap_or(u32::MAX)
    }

    fn pos(&self) -> u32 {
        u32::try_from(self.pos).unwrap_or(u32::MAX)
    }

    fn peek(&mut self) -> Result<Option<u32>, CorruptPostings> {
        Ok(self.docids.get(self.pos).copied())
    }

    fn next_step(&mut self) -> Result<Option<Posting>, CorruptPostings> {
        let Some(&docid) = self.docids.get(self.pos) else {
            return Ok(None);
        };
        self.pos = self.pos.saturating_add(1);
        Ok(Some(Posting::new(docid, 1)))
    }
}
