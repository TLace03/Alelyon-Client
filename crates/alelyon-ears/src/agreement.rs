//! Which words have stopped changing: local agreement between successive hypotheses.
//!
//! While someone is talking the recogniser re-reads the audio so far every 0.6 s, and its last few words
//! change as more sound arrives ("the stew for" becomes "the stew for dinner"). A word is committed, shown
//! as settled and never taken back, once two successive readings agree on it and on every word before it.
//! This is the LocalAgreement-2 policy of streaming Whisper (Machácek, Dabre and Bojar, 2023), written here
//! from its description. Words before the last committed time are dropped from a new reading, and a few
//! words it repeats from the end of what was committed (the audio window still holds them) are trimmed,
//! so nothing is committed twice.

use crate::whisper::Word;

/// How far before the last committed word's end a new word may start and still be new (seconds).
const OVERLAP_S: f64 = 0.1;
/// Longest run of already-committed words a new reading may repeat at its start.
const MAX_REPEAT: usize = 5;
/// A repeat is only looked for when the new reading starts this close to the last committed word.
const REPEAT_WINDOW_S: f64 = 1.0;

#[derive(Default, Debug)]
pub struct Agreement {
    committed: Vec<Word>,
    /// The previous reading's words after the committed ones.
    pending: Vec<Word>,
}

impl Agreement {
    pub fn committed(&self) -> &[Word] {
        &self.committed
    }

    pub fn last_committed_end(&self) -> f64 {
        self.committed.last().map(|w| w.end).unwrap_or(0.0)
    }

    /// Take a new reading (word times in the utterance's seconds). Returns the words committed by it and
    /// the words still settling.
    pub fn insert(&mut self, reading: Vec<Word>) -> (Vec<Word>, Vec<Word>) {
        let fresh = self.fresh(reading);
        let mut newly = Vec::new();
        let mut agreed = 0;
        while agreed < fresh.len() && agreed < self.pending.len() && fresh[agreed].key() == self.pending[agreed].key() {
            agreed += 1;
        }
        // The new reading's copy of each agreed word carries the later, better timing.
        newly.extend(fresh[..agreed].iter().cloned());
        self.committed.extend(newly.iter().cloned());
        self.pending = fresh[agreed..].to_vec();
        (newly, self.pending.clone())
    }

    /// The utterance is over: everything still settling is committed as it last read.
    pub fn flush(&mut self) -> Vec<Word> {
        let rest = std::mem::take(&mut self.pending);
        self.committed.extend(rest.iter().cloned());
        rest
    }

    /// Commit a final reading outright: the words of `reading` after what is already committed.
    pub fn finish(&mut self, reading: Vec<Word>) -> Vec<Word> {
        let fresh = self.fresh(reading);
        self.pending.clear();
        self.committed.extend(fresh.iter().cloned());
        fresh
    }

    fn fresh(&self, reading: Vec<Word>) -> Vec<Word> {
        let last_end = self.last_committed_end();
        let mut fresh: Vec<Word> = if self.committed.is_empty() {
            reading
        } else {
            reading.into_iter().filter(|w| w.start > last_end - OVERLAP_S).collect()
        };
        // Drop words with nothing in them to compare ("-", "...").
        fresh.retain(|w| !w.key().is_empty());
        if let (Some(first), false) = (fresh.first(), self.committed.is_empty()) {
            if (first.start - last_end).abs() < REPEAT_WINDOW_S {
                let longest = MAX_REPEAT.min(fresh.len()).min(self.committed.len());
                for n in (1..=longest).rev() {
                    let tail = &self.committed[self.committed.len() - n..];
                    if tail.iter().zip(&fresh[..n]).all(|(a, b)| a.key() == b.key()) {
                        fresh.drain(..n);
                        break;
                    }
                }
            }
        }
        fresh
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(spec: &[(&str, f64)]) -> Vec<Word> {
        spec.iter()
            .map(|&(t, s)| Word { text: format!(" {t}"), start: s, end: s + 0.3, probability: 0.9 })
            .collect()
    }

    fn texts(ws: &[Word]) -> Vec<String> {
        ws.iter().map(|w| w.text.trim().to_string()).collect()
    }

    #[test]
    fn a_word_is_committed_when_two_readings_agree_on_it() {
        let mut a = Agreement::default();
        let (c, p) = a.insert(words(&[("he", 0.0), ("hoped", 0.5), ("there", 0.9)]));
        assert!(c.is_empty(), "nothing can agree with a first reading");
        assert_eq!(texts(&p), ["he", "hoped", "there"]);
        let (c, p) = a.insert(words(&[("he", 0.0), ("hoped", 0.5), ("there", 0.9), ("would", 1.2), ("be", 1.5)]));
        assert_eq!(texts(&c), ["he", "hoped", "there"]);
        assert_eq!(texts(&p), ["would", "be"]);
    }

    #[test]
    fn a_changed_word_stays_pending_and_so_does_everything_after_it() {
        let mut a = Agreement::default();
        a.insert(words(&[("the", 0.0), ("stew", 0.3), ("four", 0.7)]));
        let (c, p) = a.insert(words(&[("the", 0.0), ("stew", 0.3), ("for", 0.7), ("dinner", 1.0)]));
        assert_eq!(texts(&c), ["the", "stew"]);
        assert_eq!(texts(&p), ["for", "dinner"]);
        let (c, _) = a.insert(words(&[("the", 0.0), ("stew", 0.3), ("for", 0.7), ("dinner", 1.0), ("turnips", 1.5)]));
        assert_eq!(texts(&c), ["for", "dinner"], "committed words are not committed again");
        assert_eq!(texts(a.committed()), ["the", "stew", "for", "dinner"]);
    }

    #[test]
    fn punctuation_and_case_do_not_block_agreement() {
        let mut a = Agreement::default();
        a.insert(words(&[("Dinner,", 0.0), ("turnips", 0.5)]));
        let (c, _) = a.insert(words(&[("dinner", 0.0), ("Turnips.", 0.5)]));
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn a_reading_that_repeats_the_committed_tail_is_trimmed() {
        let mut a = Agreement::default();
        a.insert(words(&[("one", 0.0), ("two", 0.5)]));
        a.insert(words(&[("one", 0.0), ("two", 0.5), ("three", 1.0)]));
        assert_eq!(texts(a.committed()), ["one", "two"]);
        // after a trim of the audio, the reading restarts slightly before "two" ended and repeats it
        let (c, p) = a.insert(words(&[("two", 0.75), ("three", 1.0), ("four", 1.4)]));
        assert_eq!(texts(&c), ["three"]);
        assert_eq!(texts(&p), ["four"]);
        assert_eq!(texts(a.committed()), ["one", "two", "three"]);
    }

    #[test]
    fn flush_commits_what_is_left_and_finish_commits_a_final_reading() {
        let mut a = Agreement::default();
        a.insert(words(&[("hello", 0.0), ("there", 0.4)]));
        assert_eq!(texts(&a.flush()), ["hello", "there"]);
        let mut b = Agreement::default();
        b.insert(words(&[("good", 0.0)]));
        b.insert(words(&[("good", 0.0), ("morning", 0.4)]));
        let rest = b.finish(words(&[("good", 0.0), ("morning", 0.4), ("Sinai", 0.9)]));
        assert_eq!(texts(&rest), ["morning", "Sinai"]);
        assert_eq!(texts(b.committed()), ["good", "morning", "Sinai"]);
    }
}
