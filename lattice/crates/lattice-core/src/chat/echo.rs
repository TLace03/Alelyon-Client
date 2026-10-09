//! The development echo model's text (the native chat's spec §3.3.7; row C5 of
//! the chat core's spec): a port of the web Lattice's `_dev_echo`
//! (the web Lattice's `chat.py`), which streams a fixed reply
//! quoting the question so the interface can be exercised without a model.
//!
//! [`echo_text`] is `_ECHO.format(question=" ".join(question.split())[:300])`:
//! the question with its whitespace collapsed (Python's `str.split`) and cut
//! to 300 code points, inside the fixed reply. [`echo_pieces`] cuts the reply
//! into pieces of [`PIECE_CHARS`] code points, the last one shorter; the web
//! sends one every [`PIECE_GAP_MS`] ms. Both are pinned by `chat/echo.json`.
//! Who offers the echo, and how it is streamed and stopped, is the plain
//! turn's (row C8).

/// The reply before the quoted question.
pub const HEAD: &str = "This is the **development echo model**. It is not a language model: it streams this fixed reply so the interface can be exercised without one.\n\nYou asked:\n\n> ";
/// The reply after the quoted question.
pub const TAIL: &str = "\n\nHere is a code block to exercise rendering:\n\n```python\ndef dedupe(items):\n    return list(dict.fromkeys(items))\n```\n\n- Lists render as lists.\n- `inline code` renders as code.\n";
/// How many code points of the question are quoted.
pub const QUOTED_CHARS: usize = 300;
/// How many code points each streamed piece holds.
pub const PIECE_CHARS: usize = 6;
/// The pause between pieces, in milliseconds.
pub const PIECE_GAP_MS: u64 = 12;

/// The echo's whole reply to `question`.
pub fn echo_text(question: &str) -> String {
    let collapsed = question
        .split(crate::py::is_space)
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let quoted: String = collapsed.chars().take(QUOTED_CHARS).collect();
    format!("{HEAD}{quoted}{TAIL}")
}

/// The reply in streamed pieces.
pub fn echo_pieces(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    chars
        .chunks(PIECE_CHARS)
        .map(|piece| piece.iter().collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_question_is_quoted_collapsed_and_cut() {
        let text = echo_text("  a \t b\n");
        assert!(text.contains("> a b\n"), "{text}");
        let long = echo_text(&"q".repeat(400));
        assert!(long.contains(&format!("> {}\n", "q".repeat(QUOTED_CHARS))));
        assert!(!long.contains(&"q".repeat(QUOTED_CHARS + 1)));
    }

    #[test]
    fn pieces_are_six_code_points_and_rejoin_whole() {
        let text = echo_text("\u{1f600}\u{1f600}\u{1f600}");
        let pieces = echo_pieces(&text);
        assert!(
            pieces
                .iter()
                .all(|piece| piece.chars().count() <= PIECE_CHARS)
        );
        assert_eq!(pieces.concat(), text);
        assert!(echo_pieces("").is_empty());
    }
}
