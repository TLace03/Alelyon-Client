//! A reasoning model's scratchpad, removed (the native chat's spec §2.6, §3.3.6;
//! row C5 of the chat core's spec). Ports of the web Lattice's
//! `engine._strip_think` and `answer.streaming.ThinkFilter`.
//!
//! - [`strip_think`], for a finished answer: every `<think>…</think>` span is
//!   removed (the shortest from each opening tag, left to right), then
//!   everything from an opening tag that is never closed, then the text is
//!   stripped with Python's `str.strip`.
//! - [`ThinkFilter`], its streaming twin: text before a tag is passed on as it
//!   arrives; a fragment boundary inside a tag holds back exactly the part
//!   that could still become one, so the tag is never shown in halves and the
//!   stream never stalls; an opening tag that is never closed swallows the
//!   rest. Its output does not depend on how the text was cut into fragments
//!   (`chat/think.json` checks every single cut and recorded random ones).
//!
//! - [`think_text`], the agent chat's own addition (the chat core's spec
//!   §22.8 RP3): what [`strip_think`] removes, kept: each span's inner text
//!   and an unclosed tag's rest, so the reasoning can be recorded and
//!   replayed while the answer stays free of it.
//!
//! The tags are matched exactly (`<think>`, `</think>`), case and all.

use crate::py;

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

/// `_strip_think(text)`.
pub fn strip_think(text: &str) -> String {
    // `re.sub(r"<think>.*?</think>", "", text, flags=re.S)`.
    let mut kept = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find(OPEN) {
        let after = &rest[open + OPEN.len()..];
        let Some(close) = after.find(CLOSE) else {
            break;
        };
        kept.push_str(&rest[..open]);
        rest = &after[close + CLOSE.len()..];
    }
    kept.push_str(rest);
    // `re.sub(r"<think>.*$", "", text, flags=re.S)`: from an unclosed tag to the end.
    if let Some(open) = kept.find(OPEN) {
        kept.truncate(open);
    }
    py::strip(&kept).to_owned()
}

/// The reasoning [`strip_think`] removes from `text`, in order: the inner
/// text of each closed span (the shortest from each opening tag, left to
/// right), then the rest after an opening tag that is never closed. Each
/// part is stripped with Python's `str.strip`; empty parts are left out.
pub fn think_text(text: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut kept = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find(OPEN) {
        let after = &rest[open + OPEN.len()..];
        let Some(close) = after.find(CLOSE) else {
            break;
        };
        kept.push_str(&rest[..open]);
        parts.push(after[..close].to_owned());
        rest = &after[close + CLOSE.len()..];
    }
    kept.push_str(rest);
    if let Some(open) = kept.find(OPEN) {
        parts.push(kept[open + OPEN.len()..].to_owned());
    }
    parts
        .iter()
        .map(|part| py::strip(part).to_owned())
        .filter(|part| !part.is_empty())
        .collect()
}

/// `_partial_tail(text, tag)`: the length of the longest proper prefix of
/// `tag` that ends `text`. The tags are ASCII, so bytes are characters here.
fn partial_tail(text: &str, tag: &str) -> usize {
    (1..tag.len().min(text.len() + 1))
        .rev()
        .find(|&size| text.ends_with(&tag[..size]))
        .unwrap_or(0)
}

/// `ThinkFilter`: feed fragments as they arrive, flush at the end.
#[derive(Clone, Debug, Default)]
pub struct ThinkFilter {
    buffer: String,
    inside: bool,
}

impl ThinkFilter {
    pub fn new() -> Self {
        Self::default()
    }

    /// The part of `fragment` (and of what was held back) that belongs in the
    /// transcript.
    pub fn feed(&mut self, fragment: &str) -> String {
        self.buffer.push_str(fragment);
        let mut out = String::new();
        loop {
            if self.inside {
                match self.buffer.find(CLOSE) {
                    None => {
                        let keep = partial_tail(&self.buffer, CLOSE);
                        self.buffer.drain(..self.buffer.len() - keep);
                        break;
                    }
                    Some(at) => {
                        self.buffer.drain(..at + CLOSE.len());
                        self.inside = false;
                    }
                }
                continue;
            }
            match self.buffer.find(OPEN) {
                None => {
                    let keep = partial_tail(&self.buffer, OPEN);
                    let split = self.buffer.len() - keep;
                    out.push_str(&self.buffer[..split]);
                    self.buffer.drain(..split);
                    break;
                }
                Some(at) => {
                    out.push_str(&self.buffer[..at]);
                    self.buffer.drain(..at + OPEN.len());
                    self.inside = true;
                }
            }
        }
        out
    }

    /// Whatever was held back for a tag that never arrived; nothing when the
    /// stream ended inside a scratchpad.
    pub fn flush(&mut self) -> String {
        let held = std::mem::take(&mut self.buffer);
        if self.inside { String::new() } else { held }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tag_cut_in_two_is_never_shown() {
        let mut filter = ThinkFilter::new();
        assert_eq!(filter.feed("answer <thi"), "answer ");
        assert_eq!(filter.feed("nk>secret</thi"), "");
        assert_eq!(filter.feed("nk> rest"), " rest");
        assert_eq!(filter.flush(), "");
        let mut unclosed = ThinkFilter::new();
        assert_eq!(unclosed.feed("a<think>never closed <"), "a");
        assert_eq!(unclosed.flush(), "");
        let mut held = ThinkFilter::new();
        assert_eq!(held.feed("ends with <th"), "ends with ");
        assert_eq!(held.flush(), "<th");
    }

    #[test]
    fn think_text_keeps_exactly_what_strip_think_removes() {
        assert_eq!(
            think_text("a<think> x </think>b<think>y</think>c"),
            ["x", "y"]
        );
        assert_eq!(
            think_text("<think>a<think>b</think>c</think>d"),
            ["a<think>b"]
        );
        assert_eq!(
            think_text(
                " pre<think>unclosed
more"
            ),
            ["unclosed
more"]
        );
        assert_eq!(
            think_text("x<think>done</think>y<think>open"),
            ["done", "open"]
        );
        assert!(think_text("<THINK>upper</THINK>").is_empty());
        assert!(think_text("<think> </think>plain").is_empty());
    }

    #[test]
    fn strip_think_takes_the_shortest_span_and_then_an_unclosed_rest() {
        assert_eq!(strip_think("a<think>x</think>b<think>y</think>c"), "abc");
        assert_eq!(
            strip_think("<think>a<think>b</think>c</think>d"),
            "c</think>d"
        );
        assert_eq!(strip_think(" pre<think>unclosed\nmore"), "pre");
        assert_eq!(strip_think("<THINK>upper</THINK>"), "<THINK>upper</THINK>");
    }
}
