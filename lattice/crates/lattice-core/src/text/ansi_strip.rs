//! Terminal escapes taken out of command output before the model reads it
//! (the chat core's spec §7.6: "The core only strips escapes for the
//! model, in `text/ansi_strip.rs` (CSI, OSC and lone-ESC sequences), with no
//! new dependency"). Not a port. The interface draws colours with its own
//! parser; the model gets plain text.
//!
//! What [`strip`] removes (ECMA-48's shapes, 7-bit and 8-bit):
//! - **CSI**: `ESC [` or U+009B, then parameter bytes (0x30–0x3F),
//!   intermediate bytes (0x20–0x2F) and one final byte (0x40–0x7E): colours,
//!   cursor moves, erases;
//! - **OSC** and the other strings (`ESC ]`, `ESC P`, `ESC X`, `ESC ^`,
//!   `ESC _`, and U+009D, U+0090, U+0098, U+009E, U+009F): everything up to
//!   BEL, `ESC \` or U+009C (window titles, hyperlinks' targets; a
//!   hyperlink's visible text stays, since it is outside the string);
//! - **a lone ESC**: `ESC`, any intermediate bytes, and one final byte
//!   (`ESC ( B`, `ESC =`, `ESC 7`); an ESC at the very end;
//! - an unterminated CSI or string runs to the end of the text and is
//!   removed with it.
//!
//! Everything else is kept as it is, other control characters included.

const ESC: char = '\u{1b}';
const BEL: char = '\u{07}';
/// The 8-bit string terminator.
const ST: char = '\u{9c}';

fn is_string_opener(c: char) -> bool {
    matches!(c, ']' | 'P' | 'X' | '^' | '_')
}

fn is_c1_string(c: char) -> bool {
    matches!(c, '\u{9d}' | '\u{90}' | '\u{98}' | '\u{9e}' | '\u{9f}')
}

/// `text` without its terminal escape sequences.
pub fn strip(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            ESC => match chars.peek().copied() {
                None => {}
                Some('[') => {
                    chars.next();
                    skip_csi(&mut chars);
                }
                Some(opener) if is_string_opener(opener) => {
                    chars.next();
                    skip_string(&mut chars);
                }
                Some(_) => {
                    // A lone escape: intermediates, then one final byte.
                    while let Some(&next) = chars.peek() {
                        if ('\u{20}'..='\u{2f}').contains(&next) {
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    chars.next();
                }
            },
            '\u{9b}' => skip_csi(&mut chars),
            c if is_c1_string(c) => skip_string(&mut chars),
            c => out.push(c),
        }
    }
    out
}

/// After the CSI opener: parameters, intermediates, one final byte.
fn skip_csi(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) {
    while let Some(&c) = chars.peek() {
        if ('\u{40}'..='\u{7e}').contains(&c) {
            chars.next();
            return;
        }
        if !('\u{20}'..='\u{3f}').contains(&c) {
            // Not a CSI byte: the sequence was broken there; the character
            // that broke it is text and stays.
            return;
        }
        chars.next();
    }
}

/// After a string opener: up to BEL, `ESC \` or ST.
fn skip_string(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) {
    while let Some(c) = chars.next() {
        match c {
            BEL | ST => return,
            ESC => {
                if chars.peek() == Some(&'\\') {
                    chars.next();
                }
                return;
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::strip;

    #[test]
    fn colours_cursor_moves_and_erases_go() {
        assert_eq!(strip("\x1b[31mred\x1b[0m plain"), "red plain");
        assert_eq!(strip("\x1b[1;38;5;196mbold\x1b[m"), "bold");
        assert_eq!(strip("a\x1b[2K\x1b[1Gb"), "ab");
        assert_eq!(strip("\x1b[?25lhidden\x1b[?25h"), "hidden");
        assert_eq!(strip("8-bit \u{9b}32mgreen\u{9b}0m"), "8-bit green");
    }

    #[test]
    fn titles_and_hyperlinks_lose_their_strings_and_keep_their_text() {
        assert_eq!(strip("\x1b]0;window title\x07after"), "after");
        assert_eq!(strip("\x1b]0;title\x1b\\after"), "after");
        assert_eq!(
            strip("\x1b]8;;https://example.invalid/\x1b\\link text\x1b]8;;\x1b\\"),
            "link text"
        );
        assert_eq!(strip("\x1bPdevice control\x1b\\x"), "x");
        assert_eq!(strip("\u{9d}8-bit title\u{9c}x"), "x");
    }

    #[test]
    fn lone_escapes_and_unterminated_sequences() {
        assert_eq!(strip("\x1b(Bascii\x1b=\x1b7"), "ascii");
        assert_eq!(strip("end\x1b"), "end");
        assert_eq!(strip("open\x1b[12;3"), "open");
        assert_eq!(strip("title\x1b]0;never ends"), "title");
        assert_eq!(strip("broken\x1b[12\nnext"), "broken\nnext");
    }

    #[test]
    fn other_text_and_controls_are_kept() {
        let text = "line one\r\nline two\ttabbed \u{2013} dash [brackets] ]\u{0}";
        assert_eq!(strip(text), text);
        assert_eq!(strip(""), "");
    }
}
