//! Python's text rules, reproduced where a file that both runtimes read depends on them.
//!
//! The native Lattice reads the files the Python Lattice writes and reads
//! (`model_endpoints.json`, `FAMEnvironment.env`, `analyst_model.json`). Where
//! Python's own behaviour decides what such a file means, this module says the
//! same thing in Rust, and the parity goldens (`tests/parity`) check the two
//! against each other:
//!
//! - `str.strip()` and `str.splitlines()`: Python's notion of whitespace and of
//!   a line break is wider than Rust's (`\x1c`-`\x1f` are whitespace, and `\r`
//!   alone, `\x0b`, `\x0c` and the Unicode line separators end a line);
//! - `json.loads(bytes)`: the encoding is detected (a UTF-8 byte order mark,
//!   UTF-16 and UTF-32 are accepted, which is what Windows tools that save "as
//!   Unicode" produce), where `serde_json` would refuse the file.
//!
//! Deliberate limits: `json.loads` accepts `NaN`, `Infinity` and lone
//! surrogate escapes, which `serde_json` refuses; such a file is `corrupt`
//! here where Python would go on to judge its rows. Both report the registry as
//! incomplete.
//!
//! Invariant: nothing here panics on any input.

use serde_json::Value;

/// `str.isspace()` for one character.
pub(crate) fn is_space(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}

/// `str.strip()`.
pub(crate) fn strip(text: &str) -> &str {
    text.trim_matches(is_space)
}

/// `str.splitlines()`.
pub(crate) fn splitlines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((index, c)) = chars.next() {
        if !matches!(
            c,
            '\n' | '\r'
                | '\u{b}'
                | '\u{c}'
                | '\u{1c}'
                | '\u{1d}'
                | '\u{1e}'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        ) {
            continue;
        }
        lines.push(&text[start..index]);
        let mut end = index + c.len_utf8();
        if c == '\r'
            && let Some(&(next, '\n')) = chars.peek()
        {
            chars.next();
            end = next + 1;
        }
        start = end;
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

#[derive(Clone, Copy)]
enum Encoding {
    Utf8,
    Utf8Sig,
    Utf16Bom,
    Utf16Le,
    Utf16Be,
    Utf32Bom,
    Utf32Le,
    Utf32Be,
}

/// `json.detect_encoding`.
fn detect_encoding(b: &[u8]) -> Encoding {
    if b.starts_with(&[0, 0, 0xfe, 0xff]) || b.starts_with(&[0xff, 0xfe, 0, 0]) {
        return Encoding::Utf32Bom;
    }
    if b.starts_with(&[0xfe, 0xff]) || b.starts_with(&[0xff, 0xfe]) {
        return Encoding::Utf16Bom;
    }
    if b.starts_with(&[0xef, 0xbb, 0xbf]) {
        return Encoding::Utf8Sig;
    }
    if b.len() >= 4 {
        if b[0] == 0 {
            return if b[1] != 0 {
                Encoding::Utf16Be
            } else {
                Encoding::Utf32Be
            };
        }
        if b[1] == 0 {
            return if b[2] != 0 || b[3] != 0 {
                Encoding::Utf16Le
            } else {
                Encoding::Utf32Le
            };
        }
    } else if b.len() == 2 {
        if b[0] == 0 {
            return Encoding::Utf16Be;
        }
        if b[1] == 0 {
            return Encoding::Utf16Le;
        }
    }
    Encoding::Utf8
}

fn utf16(bytes: &[u8], big_endian: bool) -> Option<String> {
    if !bytes.len().is_multiple_of(2) {
        return None;
    }
    let units: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|pair| {
            if big_endian {
                u16::from_be_bytes([pair[0], pair[1]])
            } else {
                u16::from_le_bytes([pair[0], pair[1]])
            }
        })
        .collect();
    String::from_utf16(&units).ok()
}

fn utf32(bytes: &[u8], big_endian: bool) -> Option<String> {
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    bytes
        .chunks_exact(4)
        .map(|quad| {
            let value = if big_endian {
                u32::from_be_bytes([quad[0], quad[1], quad[2], quad[3]])
            } else {
                u32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]])
            };
            char::from_u32(value)
        })
        .collect()
}

/// The text of a JSON document the way `json.loads(bytes)` decodes it.
pub(crate) fn decode_json_bytes(bytes: &[u8]) -> Option<String> {
    match detect_encoding(bytes) {
        Encoding::Utf8 => std::str::from_utf8(bytes).ok().map(str::to_owned),
        Encoding::Utf8Sig => std::str::from_utf8(&bytes[3..]).ok().map(str::to_owned),
        Encoding::Utf16Bom => {
            let big = bytes.starts_with(&[0xfe, 0xff]);
            utf16(&bytes[2..], big)
        }
        Encoding::Utf16Le => utf16(bytes, false),
        Encoding::Utf16Be => utf16(bytes, true),
        Encoding::Utf32Bom => {
            let big = bytes.starts_with(&[0, 0, 0xfe, 0xff]);
            utf32(&bytes[4..], big)
        }
        Encoding::Utf32Le => utf32(bytes, false),
        Encoding::Utf32Be => utf32(bytes, true),
    }
}

/// `json.loads(bytes)`: the value, or `None` for anything that raises.
pub(crate) fn loads_bytes(bytes: &[u8]) -> Option<Value> {
    serde_json::from_str(&decode_json_bytes(bytes)?).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_uses_pythons_whitespace() {
        assert_eq!(strip(" \t a b \n"), "a b");
        assert_eq!(strip("\u{1c}\u{a0}x\u{2003}"), "x");
        assert_eq!(
            strip("\u{feff}x"),
            "\u{feff}x",
            "a byte order mark is not whitespace"
        );
        assert_eq!(strip("   "), "");
    }

    #[test]
    fn splitlines_breaks_where_python_breaks() {
        assert_eq!(splitlines("a\nb\r\nc\rd"), ["a", "b", "c", "d"]);
        assert_eq!(splitlines("a\n"), ["a"]);
        assert_eq!(splitlines("a\n\n"), ["a", ""]);
        assert_eq!(splitlines(""), Vec::<&str>::new());
        assert_eq!(
            splitlines("a\u{b}b\u{c}c\u{1c}d\u{85}e\u{2028}f\u{2029}g"),
            ["a", "b", "c", "d", "e", "f", "g"]
        );
        assert_eq!(splitlines("\r\n"), [""]);
        assert_eq!(splitlines("caf\u{e9}\nz"), ["caf\u{e9}", "z"]);
    }

    #[test]
    fn json_is_decoded_from_every_encoding_python_detects() {
        let doc = r#"{"a": "café é"}"#;
        let expected: Value = serde_json::from_str(doc).unwrap();
        assert_eq!(loads_bytes(doc.as_bytes()), Some(expected.clone()));

        let mut bom = vec![0xef, 0xbb, 0xbf];
        bom.extend_from_slice(doc.as_bytes());
        assert_eq!(
            loads_bytes(&bom),
            Some(expected.clone()),
            "UTF-8 with a byte order mark"
        );

        let units: Vec<u16> = doc.encode_utf16().collect();
        let mut le_bom = vec![0xff, 0xfe];
        le_bom.extend(units.iter().flat_map(|u| u.to_le_bytes()));
        assert_eq!(
            loads_bytes(&le_bom),
            Some(expected.clone()),
            "UTF-16 LE with a byte order mark"
        );
        let mut be_bom = vec![0xfe, 0xff];
        be_bom.extend(units.iter().flat_map(|u| u.to_be_bytes()));
        assert_eq!(
            loads_bytes(&be_bom),
            Some(expected.clone()),
            "UTF-16 BE with a byte order mark"
        );
        let le_plain: Vec<u8> = units.iter().flat_map(|u| u.to_le_bytes()).collect();
        assert_eq!(
            loads_bytes(&le_plain),
            Some(expected.clone()),
            "UTF-16 LE detected from its NULs"
        );
        let be_plain: Vec<u8> = units.iter().flat_map(|u| u.to_be_bytes()).collect();
        assert_eq!(
            loads_bytes(&be_plain),
            Some(expected.clone()),
            "UTF-16 BE detected from its NULs"
        );

        let mut utf32_bom = vec![0xff, 0xfe, 0, 0];
        utf32_bom.extend(doc.chars().flat_map(|c| (c as u32).to_le_bytes()));
        assert_eq!(
            loads_bytes(&utf32_bom),
            Some(expected.clone()),
            "UTF-32 LE with a byte order mark"
        );
        let utf32_be: Vec<u8> = doc.chars().flat_map(|c| (c as u32).to_be_bytes()).collect();
        assert_eq!(
            loads_bytes(&utf32_be),
            Some(expected),
            "UTF-32 BE detected from its NULs"
        );
    }

    #[test]
    fn undecodable_or_malformed_documents_are_refused() {
        assert_eq!(loads_bytes(b"\xff\xff\xff"), None);
        assert_eq!(loads_bytes(b"{"), None);
        assert_eq!(loads_bytes(b"{} x"), None, "trailing data");
        assert_eq!(loads_bytes(b""), None);
        assert_eq!(
            loads_bytes(&[0xff, 0xfe, 0x7b]),
            None,
            "an odd number of UTF-16 bytes"
        );
    }
}
