//! Python's JSON, both ways (the native chat's spec §3.3.1; the chat core's spec
//! row C1).
//!
//! The shared chat store (`globals/lattice_chat`) is written by Python's
//! `json.dumps` and read by `json.loads`, and a record native rewrites must
//! come out as Python would have written it. `serde_json` cannot do that here:
//! this workspace builds it without `preserve_order` (its maps sort their
//! keys) and without `arbitrary_precision` (an integer beyond 64 bits arrives
//! as a float). (It is built with `float_roundtrip` since row E11d, so its
//! floats are exact too; this module's always were.) So this module parses
//! JSON itself:
//!
//! - [`PyValue`] keeps an object's key order, as a Python `dict` does (a
//!   repeated key keeps its first position and takes its last value), and an
//!   integer's decimal digits, however many there are. Floats are parsed by
//!   Rust's correctly rounded `str::parse`, as Python's `float()` is.
//! - [`loads`] accepts exactly what `json.loads(str)` accepts, and says which
//!   of those documents this port cannot hold ([`JsonError::BeyondNative`]):
//!   `NaN`, `Infinity` and `-Infinity`, a number that overflows to infinity, a
//!   lone surrogate escape (`"\ud800"`; a Rust string cannot hold one), and
//!   nesting deeper than [`MAX_DEPTH`]. Anything Python refuses is
//!   [`JsonError::Invalid`], including an integer of more than 4,300 digits
//!   (Python 3.12's conversion limit) and a leading byte order mark. The
//!   difference matters to the store: a document Python cannot read is
//!   corrupt, one only native cannot read must be left alone (deviations D1,
//!   D7).
//! - [`dumps`] writes `json.dumps(v)`: separators `", "` and `": "`,
//!   `ensure_ascii`, floats as `repr`. [`dumps_indent1`] writes
//!   `json.dumps(v, indent=1)`, the index format. `ensure_ascii` is required:
//!   Python reads the store with `str.splitlines()`, which splits on U+2028,
//!   U+2029, U+0085 and more, so a raw one would cut a record in two.
//! - The coercions the store's reader applies (`str()`, `float()`, `int()`,
//!   truthiness, iterating a value as `for x in (v or [])`): [`py_str`],
//!   [`py_float`], [`py_int`], [`py_truthy`], [`py_iter_strs`],
//!   [`py_iter_objects`] and [`py_count`]. Each says when Python would raise
//!   ([`CoerceError::Python`]) and when Python succeeds with something this
//!   port cannot hold ([`CoerceError::Native`]: an integer beyond `i64`, or a
//!   numeral written in another script, which `float()` and `int()` read).
//!   Only decimal digits (Unicode category Nd) make such a numeral; a
//!   superscript, a fraction, a Roman or circled numeral, or an Nd digit
//!   beside anything Python refuses, is refused as Python refuses it
//!   (spec 22.6 P2).
//!
//! Deviation D2: `str()` of a list or an object is its JSON text here, where
//! Python writes its `repr`.
//!
//! Invariants: nothing here panics on any input, and the parser does not
//! recurse, so no input can exhaust the stack. Past [`MAX_DEPTH`] it keeps
//! one bit per open level, not a frame (spec 22.6 P1), so a document of
//! brackets costs about one bit each there.

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::calc::float_repr;

/// The deepest nesting of lists and objects this port holds. Python's own
/// limit is its recursion limit (about 1,000); a document between the two is
/// [`Beyond::Depth`].
pub const MAX_DEPTH: usize = 512;

/// Python 3.12 refuses to convert an integer string longer than this
/// (`sys.int_info.default_max_str_digits`).
pub const MAX_INT_DIGITS: usize = 4300;

/// A JSON value as Python's `json.loads` gives it.
#[derive(Clone, Debug, PartialEq)]
pub enum PyValue {
    Null,
    Bool(bool),
    /// An integer's decimal digits, with `-` first when negative; never `-0`.
    Int(String),
    Float(f64),
    Str(String),
    List(Vec<PyValue>),
    /// Key order as written; a repeated key keeps its first position.
    Object(Vec<(String, PyValue)>),
}

impl PyValue {
    /// An integer.
    pub fn int(value: i64) -> Self {
        Self::Int(value.to_string())
    }

    /// A string.
    pub fn str(value: impl Into<String>) -> Self {
        Self::Str(value.into())
    }

    /// `d.get(key)` of an object; `None` for a missing key or a non-object.
    pub fn get(&self, key: &str) -> Option<&PyValue> {
        match self {
            Self::Object(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The text of a string value.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(text) => Some(text),
            _ => None,
        }
    }

    /// True for an object (a Python `dict`).
    pub fn is_object(&self) -> bool {
        matches!(self, Self::Object(_))
    }
}

/// Why a text did not give a [`PyValue`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JsonError {
    /// Python's `json.loads` refuses it too.
    Invalid,
    /// Python reads it, and it holds something this port cannot.
    BeyondNative(Beyond),
}

/// What Python reads and this port cannot hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Beyond {
    /// `NaN`, `Infinity`, `-Infinity`, or a number that overflows to infinity.
    NonFinite,
    /// A `\uD800`–`\uDFFF` escape that is not half of a pair.
    LoneSurrogate,
    /// Nesting deeper than [`MAX_DEPTH`].
    Depth,
}

/// `json.loads(text)`.
pub fn loads(text: &str) -> Result<PyValue, JsonError> {
    // Python refuses a document that starts with a byte order mark
    // ("Unexpected UTF-8 BOM").
    if text.starts_with('\u{feff}') {
        return Err(JsonError::Invalid);
    }
    let mut parser = Parser {
        bytes: text.as_bytes(),
        at: 0,
        beyond: None,
    };
    parser.whitespace();
    let value = parser.document()?;
    parser.whitespace();
    if parser.at != parser.bytes.len() {
        return Err(JsonError::Invalid);
    }
    match parser.beyond {
        Some(beyond) => Err(JsonError::BeyondNative(beyond)),
        None => Ok(value),
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
    /// The first thing seen that Python reads and this port cannot hold. The
    /// parse goes on, so a document that is also invalid is reported invalid.
    beyond: Option<Beyond>,
}

/// A list or an object being read and kept. There are at most
/// [`MAX_DEPTH`] of them; deeper levels are checked but not kept, and are
/// only counted ([`Skimmed`]).
enum Frame {
    List(Vec<PyValue>),
    Object {
        pairs: Vec<(String, PyValue)>,
        positions: HashMap<String, usize>,
        key: String,
    },
}

/// The levels open past [`MAX_DEPTH`], one bit each (set: a list, clear: an
/// object), so a document of N brackets costs N bits there, not N frames
/// (spec 22.6 P1). Python stops such a document at its recursion limit.
#[derive(Default)]
struct Skimmed {
    bits: Vec<u64>,
    len: usize,
}

impl Skimmed {
    fn push(&mut self, list: bool) {
        if self.len.is_multiple_of(64) {
            self.bits.push(0);
        }
        if list {
            self.bits[self.len / 64] |= 1 << (self.len % 64);
        }
        self.len += 1;
    }

    /// The innermost skimmed level: `Some(true)` for a list.
    fn last(&self) -> Option<bool> {
        let top = self.len.checked_sub(1)?;
        Some(self.bits[top / 64] & (1 << (top % 64)) != 0)
    }

    fn pop(&mut self) -> Option<bool> {
        let list = self.last()?;
        self.len -= 1;
        let word = self.len / 64;
        self.bits[word] &= !(1 << (self.len % 64));
        if self.len.is_multiple_of(64) {
            self.bits.pop();
        }
        Some(list)
    }
}

/// P1's falsifier counters and switch (test builds only).
#[cfg(test)]
pub(crate) mod probe {
    use std::cell::Cell;

    thread_local! {
        static FRAMES: Cell<usize> = const { Cell::new(0) };
        static FRAME_PER_BRACKET: Cell<bool> = const { Cell::new(false) };
    }

    /// Frames pushed on this thread since the last [`reset`].
    pub(crate) fn frames() -> usize {
        FRAMES.with(Cell::get)
    }

    pub(crate) fn reset(frame_per_bracket: bool) {
        FRAMES.with(|cell| cell.set(0));
        FRAME_PER_BRACKET.with(|cell| cell.set(frame_per_bracket));
    }

    pub(super) fn pushed() {
        FRAMES.with(|cell| cell.set(cell.get() + 1));
    }

    /// The mutant: every bracket past the limit costs a frame again.
    pub(super) fn frame_per_bracket() -> bool {
        FRAME_PER_BRACKET.with(Cell::get)
    }
}

/// Open a kept level (and count it, in test builds).
fn push_frame(stack: &mut Vec<Frame>, frame: Frame) {
    #[cfg(test)]
    probe::pushed();
    stack.push(frame);
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn eat(&mut self, byte: u8) -> bool {
        if self.peek() == Some(byte) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn eat_word(&mut self, word: &str) -> bool {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            true
        } else {
            false
        }
    }

    /// Python's JSON whitespace: space, tab, LF and CR only.
    fn whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.at += 1;
        }
    }

    fn note(&mut self, beyond: Beyond) {
        self.beyond.get_or_insert(beyond);
    }

    /// One whole value, iteratively: a stack of open lists and objects
    /// instead of recursion, and past [`MAX_DEPTH`] one bit per open level.
    fn document(&mut self) -> Result<PyValue, JsonError> {
        let mut stack: Vec<Frame> = Vec::new();
        let mut skimmed = Skimmed::default();
        // P1's mutant only: a frame per skimmed bracket, as before P1.
        #[cfg(test)]
        let mut shadow: Vec<Frame> = Vec::new();
        loop {
            // The start of a value.
            let mut value = match self.peek() {
                Some(open @ (b'[' | b'{')) => {
                    self.at += 1;
                    let deep = stack.len() >= MAX_DEPTH;
                    if deep {
                        self.note(Beyond::Depth);
                    }
                    self.whitespace();
                    if open == b'[' {
                        if self.eat(b']') {
                            PyValue::List(Vec::new())
                        } else {
                            if deep {
                                skimmed.push(true);
                                #[cfg(test)]
                                if probe::frame_per_bracket() {
                                    push_frame(&mut shadow, Frame::List(Vec::new()));
                                }
                            } else {
                                push_frame(&mut stack, Frame::List(Vec::new()));
                            }
                            continue;
                        }
                    } else if self.eat(b'}') {
                        PyValue::Object(Vec::new())
                    } else {
                        let key = self.key()?;
                        if deep {
                            skimmed.push(false);
                        } else {
                            push_frame(
                                &mut stack,
                                Frame::Object {
                                    pairs: Vec::new(),
                                    positions: HashMap::new(),
                                    key,
                                },
                            );
                        }
                        continue;
                    }
                }
                _ => self.scalar()?,
            };
            // A complete value: hand it to the level it belongs to, closing
            // every level that ends after it. A skimmed level drops it.
            loop {
                let list = match skimmed.last() {
                    Some(list) => list,
                    None => match stack.last_mut() {
                        None => return Ok(value),
                        Some(Frame::List(items)) => {
                            items.push(value);
                            true
                        }
                        Some(Frame::Object {
                            pairs,
                            positions,
                            key,
                        }) => {
                            let key = std::mem::take(key);
                            // A dict: a repeated key keeps its place and takes
                            // the later value.
                            match positions.get(&key) {
                                Some(&index) => pairs[index].1 = value,
                                None => {
                                    positions.insert(key.clone(), pairs.len());
                                    pairs.push((key, value));
                                }
                            }
                            false
                        }
                    },
                };
                self.whitespace();
                if self.eat(b',') {
                    self.whitespace();
                    if !list {
                        let next = self.key()?;
                        if skimmed.last().is_none()
                            && let Some(Frame::Object { key, .. }) = stack.last_mut()
                        {
                            *key = next;
                        }
                    }
                    break;
                }
                if !self.eat(if list { b']' } else { b'}' }) {
                    return Err(JsonError::Invalid);
                }
                value = if skimmed.pop().is_some() {
                    // Not kept: the document is reported beyond native anyway.
                    PyValue::Null
                } else {
                    match stack.pop() {
                        Some(Frame::List(items)) => PyValue::List(items),
                        Some(Frame::Object { pairs, .. }) => PyValue::Object(pairs),
                        None => return Err(JsonError::Invalid),
                    }
                };
            }
        }
    }

    /// `"key"`, whitespace, `:` and whitespace.
    fn key(&mut self) -> Result<String, JsonError> {
        if self.peek() != Some(b'"') {
            return Err(JsonError::Invalid);
        }
        let key = self.string()?;
        self.whitespace();
        if !self.eat(b':') {
            return Err(JsonError::Invalid);
        }
        self.whitespace();
        Ok(key)
    }

    fn scalar(&mut self) -> Result<PyValue, JsonError> {
        match self.peek() {
            Some(b'"') => self.string().map(PyValue::Str),
            Some(b'n') if self.eat_word("null") => Ok(PyValue::Null),
            Some(b't') if self.eat_word("true") => Ok(PyValue::Bool(true)),
            Some(b'f') if self.eat_word("false") => Ok(PyValue::Bool(false)),
            Some(b'N') if self.eat_word("NaN") => {
                self.note(Beyond::NonFinite);
                Ok(PyValue::Float(f64::NAN))
            }
            Some(b'I') if self.eat_word("Infinity") => {
                self.note(Beyond::NonFinite);
                Ok(PyValue::Float(f64::INFINITY))
            }
            Some(b'-') if self.bytes[self.at + 1..].starts_with(b"Infinity") => {
                self.at += "-Infinity".len();
                self.note(Beyond::NonFinite);
                Ok(PyValue::Float(f64::NEG_INFINITY))
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(JsonError::Invalid),
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.at;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.at += 1;
        }
        self.at - start
    }

    /// `-?(0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?`. Python stops a number before
    /// a `.` or an `e` with no digits after it, and nothing can follow a
    /// number there, so such a text is invalid either way.
    fn number(&mut self) -> Result<PyValue, JsonError> {
        let start = self.at;
        let negative = self.eat(b'-');
        match self.peek() {
            Some(b'0') => self.at += 1,
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return Err(JsonError::Invalid),
        }
        let integer_end = self.at;
        let mut float = false;
        if self.peek() == Some(b'.') {
            self.at += 1;
            if self.digits() == 0 {
                return Err(JsonError::Invalid);
            }
            float = true;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.at += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.at += 1;
            }
            if self.digits() == 0 {
                return Err(JsonError::Invalid);
            }
            float = true;
        }
        let text =
            std::str::from_utf8(&self.bytes[start..self.at]).map_err(|_| JsonError::Invalid)?;
        if float {
            let value: f64 = text.parse().map_err(|_| JsonError::Invalid)?;
            if value.is_infinite() {
                self.note(Beyond::NonFinite);
            }
            return Ok(PyValue::Float(value));
        }
        let digits = &text[usize::from(negative)..integer_end - start];
        if digits.len() > MAX_INT_DIGITS {
            // Python raises ValueError: "Exceeds the limit (4300 digits)".
            return Err(JsonError::Invalid);
        }
        Ok(PyValue::Int(if negative && digits != "0" {
            format!("-{digits}")
        } else {
            digits.to_owned()
        }))
    }

    fn hex4(&mut self) -> Result<u32, JsonError> {
        let Some(chunk) = self.bytes.get(self.at..self.at + 4) else {
            return Err(JsonError::Invalid);
        };
        let mut value = 0u32;
        for byte in chunk {
            let digit = char::from(*byte).to_digit(16).ok_or(JsonError::Invalid)?;
            value = value * 16 + digit;
        }
        self.at += 4;
        Ok(value)
    }

    /// A string, from its opening quote. Control characters must be escaped
    /// (Python's strict mode); every other character may appear raw.
    fn string(&mut self) -> Result<String, JsonError> {
        self.at += 1; // the opening quote
        let mut out = String::new();
        loop {
            // A run of plain bytes: copied as they are (the input is UTF-8, and
            // a run never ends inside a character, since `"`, `\` and control
            // bytes are ASCII).
            let start = self.at;
            while let Some(byte) = self.peek() {
                if byte == b'"' || byte == b'\\' || byte < 0x20 {
                    break;
                }
                self.at += 1;
            }
            out.push_str(
                std::str::from_utf8(&self.bytes[start..self.at]).map_err(|_| JsonError::Invalid)?,
            );
            match self.peek() {
                Some(b'"') => {
                    self.at += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.at += 1;
                    let Some(escape) = self.peek() else {
                        return Err(JsonError::Invalid);
                    };
                    self.at += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let unit = self.hex4()?;
                            out.push(self.code_point(unit)?);
                        }
                        _ => return Err(JsonError::Invalid),
                    }
                }
                // A raw control character, or the end of the text.
                _ => return Err(JsonError::Invalid),
            }
        }
    }

    /// The character a `\uXXXX` escape names, joining a surrogate pair the way
    /// Python does: a high surrogate takes a following `\uXXXX` only when that
    /// one is a low surrogate.
    fn code_point(&mut self, unit: u32) -> Result<char, JsonError> {
        if (0xd800..0xdc00).contains(&unit) && self.bytes[self.at..].starts_with(b"\\u") {
            let save = self.at;
            self.at += 2;
            let low = self.hex4()?;
            if (0xdc00..0xe000).contains(&low) {
                let joined = 0x10000 + ((unit - 0xd800) << 10) + (low - 0xdc00);
                return char::from_u32(joined).ok_or(JsonError::Invalid);
            }
            self.at = save;
        }
        match char::from_u32(unit) {
            Some(c) => Ok(c),
            None => {
                self.note(Beyond::LoneSurrogate);
                Ok('\u{fffd}')
            }
        }
    }
}

// ----------------------------------------------------------------- writing

/// `json.dumps(text)` for one string: quotes included, `ensure_ascii`.
/// The same as `lattice-agents`' `python_json_string`, which is crate-private.
pub fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(ch),
            _ => {
                let mut units = [0u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
    out
}

/// A float as `json.dumps` writes it (`allow_nan=True`).
fn float_text(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_owned()
    } else if value.is_infinite() {
        if value < 0.0 { "-Infinity" } else { "Infinity" }.to_owned()
    } else {
        float_repr(value)
    }
}

/// `json.dumps(value)`.
pub fn dumps(value: &PyValue) -> String {
    let mut out = String::new();
    write_compact(value, &mut out);
    out
}

fn write_compact(value: &PyValue, out: &mut String) {
    match value {
        PyValue::Null => out.push_str("null"),
        PyValue::Bool(true) => out.push_str("true"),
        PyValue::Bool(false) => out.push_str("false"),
        PyValue::Int(digits) => out.push_str(digits),
        PyValue::Float(number) => out.push_str(&float_text(*number)),
        PyValue::Str(text) => out.push_str(&json_string(text)),
        PyValue::List(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_compact(item, out);
            }
            out.push(']');
        }
        PyValue::Object(pairs) => {
            out.push('{');
            for (index, (key, item)) in pairs.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                out.push_str(&json_string(key));
                out.push_str(": ");
                write_compact(item, out);
            }
            out.push('}');
        }
    }
}

/// `json.dumps(value, indent=1)`: one space per level, `","` between items and
/// `": "` after a key, as Python writes it when an indent is given. The line
/// ends are `\n`; a text-mode write on Windows makes them CRLF (rule S2).
pub fn dumps_indent1(value: &PyValue) -> String {
    let mut out = String::new();
    write_indented(value, 0, &mut out);
    out
}

fn newline(level: usize, out: &mut String) {
    out.push('\n');
    out.extend(std::iter::repeat_n(' ', level));
}

fn write_indented(value: &PyValue, level: usize, out: &mut String) {
    match value {
        PyValue::List(items) if !items.is_empty() => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                newline(level + 1, out);
                write_indented(item, level + 1, out);
            }
            newline(level, out);
            out.push(']');
        }
        PyValue::Object(pairs) if !pairs.is_empty() => {
            out.push('{');
            for (index, (key, item)) in pairs.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                newline(level + 1, out);
                out.push_str(&json_string(key));
                out.push_str(": ");
                write_indented(item, level + 1, out);
            }
            newline(level, out);
            out.push('}');
        }
        _ => write_compact(value, out),
    }
}

// --------------------------------------------------------------- coercions

/// Why a coercion gave no value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoerceError {
    /// Python raises here too.
    Python,
    /// Python succeeds with something this port cannot hold.
    Native,
}

/// `str(value)`. Deviation D2: a list or an object is its JSON text.
pub fn py_str(value: &PyValue) -> String {
    match value {
        PyValue::Null => "None".to_owned(),
        PyValue::Bool(true) => "True".to_owned(),
        PyValue::Bool(false) => "False".to_owned(),
        PyValue::Int(digits) => digits.clone(),
        PyValue::Float(number) => float_repr(*number),
        PyValue::Str(text) => text.clone(),
        PyValue::List(_) | PyValue::Object(_) => dumps(value),
    }
}

/// `bool(value)`.
pub fn py_truthy(value: &PyValue) -> bool {
    match value {
        PyValue::Null => false,
        PyValue::Bool(flag) => *flag,
        PyValue::Int(digits) => digits != "0",
        // NaN is true, as in Python.
        PyValue::Float(number) => *number != 0.0,
        PyValue::Str(text) => !text.is_empty(),
        PyValue::List(items) => !items.is_empty(),
        PyValue::Object(pairs) => !pairs.is_empty(),
    }
}

/// The whitespace `float()` and `int()` strip from a string: ASCII space,
/// tab, LF, VT, FF and CR, and the non-ASCII characters `str.isspace()`
/// accepts. Unlike `str.strip()`, not `\x1c`–`\x1f` (Python keeps ASCII
/// characters as they are and strips with `Py_ISSPACE`).
fn number_space(c: char) -> bool {
    c.is_whitespace()
}

/// The first code point of every run of ten decimal digits (Unicode general
/// category Nd, digit values 0 to 9 in order), as CPython 3.12 knows them
/// (Unicode 15.0.0). Generated with `[c for c in range(0x110000) if
/// unicodedata.category(chr(c)) == "Nd" and unicodedata.decimal(chr(c)) ==
/// 0]`, which also showed that every Nd character sits in one of these runs.
const DECIMAL_ZEROS: [u32; 68] = [
    0x30, 0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xCE6, 0xD66,
    0xDE6, 0xE50, 0xED0, 0xF20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80, 0x1A90,
    0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900, 0xA9D0, 0xA9F0, 0xAA50, 0xABF0, 0xFF10,
    0x104A0, 0x10D30, 0x11066, 0x110F0, 0x11136, 0x111D0, 0x112F0, 0x11450, 0x114D0, 0x11650,
    0x116C0, 0x11730, 0x118E0, 0x11950, 0x11C50, 0x11D50, 0x11DA0, 0x11F50, 0x16A60, 0x16AC0,
    0x16B50, 0x1D7CE, 0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140, 0x1E2F0, 0x1E4F0, 0x1E950,
    0x1FBF0,
];

/// A decimal digit's value in any script (category Nd), as CPython's
/// `Py_UNICODE_TODECIMAL` gives it; `None` for every other character.
fn decimal_digit(c: char) -> Option<u8> {
    let code = u32::from(c);
    let at = DECIMAL_ZEROS.partition_point(|&zero| zero <= code);
    let zero = DECIMAL_ZEROS.get(at.checked_sub(1)?)?;
    u8::try_from(code - zero).ok().filter(|&digit| digit < 10)
}

/// A numeric string's characters outside ASCII, read as `float()` and `int()`
/// read them: CPython turns each decimal digit of any script (category Nd)
/// into its ASCII digit and each other whitespace character into a space,
/// and refuses a string holding anything else (a superscript, a fraction, a
/// Roman or circled numeral). `None` for ASCII text. Otherwise `Python` when
/// Python refuses the string, and `Native` when it reads it (a numeral in
/// another script, which this port does not hold: deviation D7). `read` is
/// the ASCII reader that decides, given the string as CPython rewrites it.
fn foreign<T>(text: &str, read: fn(&str) -> Result<T, CoerceError>) -> Option<CoerceError> {
    if text.is_ascii() {
        return None;
    }
    let mut ascii = String::with_capacity(text.len());
    for c in text.chars() {
        if c.is_ascii() {
            ascii.push(c);
        } else if let Some(digit) = decimal_digit(c) {
            ascii.push(char::from(b'0' + digit));
        } else if number_space(c) {
            ascii.push(' ');
        } else {
            return Some(CoerceError::Python);
        }
    }
    Some(match read(&ascii) {
        Err(CoerceError::Python) => CoerceError::Python,
        Ok(_) | Err(CoerceError::Native) => CoerceError::Native,
    })
}

/// Python's underscores in a numeral: each one between two ASCII digits.
/// The numeral without them.
fn without_underscores(text: &str) -> Result<String, CoerceError> {
    let bytes = text.as_bytes();
    for (index, byte) in bytes.iter().enumerate() {
        if *byte == b'_' {
            let before = index > 0 && bytes[index - 1].is_ascii_digit();
            let after = bytes.get(index + 1).is_some_and(u8::is_ascii_digit);
            if !(before && after) {
                return Err(CoerceError::Python);
            }
        }
    }
    Ok(text.replace('_', ""))
}

/// `float(text)`.
fn float_from_str(text: &str) -> Result<f64, CoerceError> {
    let text = text.trim_matches(number_space);
    if text.is_empty() {
        return Err(CoerceError::Python);
    }
    if let Some(error) = foreign(text, float_from_str) {
        return Err(error);
    }
    let plain = without_underscores(text)?;
    // Rust's grammar is Python's here: an optional sign, then `inf`,
    // `infinity` or `nan` in any case, or digits with an optional point and
    // exponent; an overflow gives infinity, as `float()` does for a string.
    plain.parse::<f64>().map_err(|_| CoerceError::Python)
}

/// `float(value)`.
pub fn py_float(value: &PyValue) -> Result<f64, CoerceError> {
    match value {
        PyValue::Bool(flag) => Ok(if *flag { 1.0 } else { 0.0 }),
        PyValue::Float(number) => Ok(*number),
        PyValue::Int(digits) => {
            // int -> float rounds to nearest, ties to even, and raises
            // OverflowError past the largest double.
            let number: f64 = digits.parse().map_err(|_| CoerceError::Python)?;
            if number.is_finite() {
                Ok(number)
            } else {
                Err(CoerceError::Python)
            }
        }
        PyValue::Str(text) => float_from_str(text),
        PyValue::Null | PyValue::List(_) | PyValue::Object(_) => Err(CoerceError::Python),
    }
}

fn i64_from_digits(digits: &str) -> Result<i64, CoerceError> {
    // The digits are a valid integer here, so a failure is an overflow.
    digits.parse::<i64>().map_err(|_| CoerceError::Native)
}

/// `int(text)`, base 10.
fn int_from_str(text: &str) -> Result<i64, CoerceError> {
    let text = text.trim_matches(number_space);
    if let Some(error) = foreign(text, int_from_str) {
        return Err(error);
    }
    let (negative, body) = match text.as_bytes().first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    };
    if body.is_empty() || !body.bytes().all(|b| b.is_ascii_digit() || b == b'_') {
        return Err(CoerceError::Python);
    }
    let digits = without_underscores(body)?;
    if digits.len() > MAX_INT_DIGITS {
        return Err(CoerceError::Python);
    }
    let trimmed = digits.trim_start_matches('0');
    let magnitude = if trimmed.is_empty() { "0" } else { trimmed };
    if negative && magnitude != "0" {
        i64_from_digits(&format!("-{magnitude}"))
    } else {
        i64_from_digits(magnitude)
    }
}

/// `int(value)`. A value Python converts but `i64` cannot hold is
/// [`CoerceError::Native`].
pub fn py_int(value: &PyValue) -> Result<i64, CoerceError> {
    match value {
        PyValue::Bool(flag) => Ok(i64::from(*flag)),
        PyValue::Int(digits) => i64_from_digits(digits),
        PyValue::Float(number) => {
            if number.is_nan() || number.is_infinite() {
                // ValueError and OverflowError.
                return Err(CoerceError::Python);
            }
            let truncated = number.trunc();
            if (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&truncated) {
                Ok(truncated as i64)
            } else {
                Err(CoerceError::Native)
            }
        }
        PyValue::Str(text) => int_from_str(text),
        PyValue::Null | PyValue::List(_) | PyValue::Object(_) => Err(CoerceError::Python),
    }
}

/// `[str(x) for x in (value or [])]`: a list's items, a string's characters,
/// an object's keys; a true number or `True` is not iterable.
pub fn py_iter_strs(value: &PyValue) -> Result<Vec<String>, CoerceError> {
    if !py_truthy(value) {
        return Ok(Vec::new());
    }
    match value {
        PyValue::List(items) => Ok(items.iter().map(py_str).collect()),
        PyValue::Str(text) => Ok(text.chars().map(String::from).collect()),
        PyValue::Object(pairs) => Ok(pairs.iter().map(|(key, _)| key.clone()).collect()),
        _ => Err(CoerceError::Python),
    }
}

/// `[x for x in (value or []) if isinstance(x, dict)]`: a turn's facts.
pub fn py_iter_objects(value: &PyValue) -> Result<Vec<PyValue>, CoerceError> {
    if !py_truthy(value) {
        return Ok(Vec::new());
    }
    match value {
        PyValue::List(items) => Ok(items
            .iter()
            .filter(|item| item.is_object())
            .cloned()
            .collect()),
        // A string's characters and an object's keys are strings, not dicts.
        PyValue::Str(_) | PyValue::Object(_) => Ok(Vec::new()),
        _ => Err(CoerceError::Python),
    }
}

/// `history._count`: a token count the record stated, or `None`. A bool is
/// `None`; a number is `int(value)`; anything else is `None`.
pub fn py_count(value: &PyValue) -> Result<Option<i64>, CoerceError> {
    match value {
        PyValue::Int(_) | PyValue::Float(_) => py_int(value).map(Some),
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(pairs: &[(&str, PyValue)]) -> PyValue {
        PyValue::Object(
            pairs
                .iter()
                .map(|(key, value)| ((*key).to_owned(), value.clone()))
                .collect(),
        )
    }

    /// (input, `json.dumps(json.loads(input))`, `json.dumps(..., indent=1)`),
    /// as CPython 3.12.10 printed them.
    const PYTHON_ROUND_TRIPS: &[(&str, &str, &str)] = &[
        (
            r#"{"b": 1, "a": [true, false, null], "c": {"z": 1.5, "y": "x"}}"#,
            r#"{"b": 1, "a": [true, false, null], "c": {"z": 1.5, "y": "x"}}"#,
            "{\n \"b\": 1,\n \"a\": [\n  true,\n  false,\n  null\n ],\n \"c\": {\n  \"z\": 1.5,\n  \"y\": \"x\"\n }\n}",
        ),
        (
            "123456789012345678901234567890",
            "123456789012345678901234567890",
            "123456789012345678901234567890",
        ),
        ("-0", "0", "0"),
        ("-0.0", "-0.0", "-0.0"),
        ("1e16", "1e+16", "1e+16"),
        ("1E5", "100000.0", "100000.0"),
        ("0.1", "0.1", "0.1"),
        ("1e-7", "1e-07", "1e-07"),
        ("5e-324", "5e-324", "5e-324"),
        (
            "1.7976931348623157e308",
            "1.7976931348623157e+308",
            "1.7976931348623157e+308",
        ),
        (
            r#""\u2028\u2029\u0085\u0001\u007f\ud83d\ude00 caf\u00e9 \/ \b\f\n\r\t""#,
            r#""\u2028\u2029\u0085\u0001\u007f\ud83d\ude00 caf\u00e9 / \b\f\n\r\t""#,
            r#""\u2028\u2029\u0085\u0001\u007f\ud83d\ude00 caf\u00e9 / \b\f\n\r\t""#,
        ),
        (
            r#"{"a": 1, "b": 2, "a": 3}"#,
            r#"{"a": 3, "b": 2}"#,
            "{\n \"a\": 3,\n \"b\": 2\n}",
        ),
        ("[]", "[]", "[]"),
        ("{}", "{}", "{}"),
        ("[[]]", "[[]]", "[\n []\n]"),
        ("[{}]", "[{}]", "[\n {}\n]"),
        (
            "12345678901234567890",
            "12345678901234567890",
            "12345678901234567890",
        ),
        (
            "-9223372036854775809",
            "-9223372036854775809",
            "-9223372036854775809",
        ),
        ("1.0", "1.0", "1.0"),
        ("100.0", "100.0", "100.0"),
        ("1e22", "1e+22", "1e+22"),
        (
            "123456789.123456789",
            "123456789.12345679",
            "123456789.12345679",
        ),
        ("\t[1]\r\n", "[1]", "[\n 1\n]"),
        (
            "\"a\u{7f}\u{2028}\"",
            r#""a\u007f\u2028""#,
            r#""a\u007f\u2028""#,
        ),
        (
            r#""\uD83D\uDE00""#,
            r#""\ud83d\ude00""#,
            r#""\ud83d\ude00""#,
        ),
        (
            r#""\ud800\udc00""#,
            r#""\ud800\udc00""#,
            r#""\ud800\udc00""#,
        ),
        ("true ", "true", "true"),
    ];

    #[test]
    fn values_come_back_as_python_writes_them() {
        for (input, compact, indented) in PYTHON_ROUND_TRIPS {
            let value = loads(input).unwrap_or_else(|e| panic!("{input}: {e:?}"));
            assert_eq!(dumps(&value), *compact, "dumps of {input}");
            assert_eq!(dumps_indent1(&value), *indented, "indent=1 of {input}");
            // What dumps writes reads back as the same value.
            assert_eq!(loads(&dumps(&value)), Ok(value.clone()), "{input}");
            assert_eq!(loads(&dumps_indent1(&value)), Ok(value), "{input}");
        }
    }

    #[test]
    fn key_order_and_repeated_keys_follow_a_python_dict() {
        let value = loads(r#"{"z": 1, "a": 2, "m": {"y": 0, "b": 0}, "z": 9}"#).unwrap();
        assert_eq!(
            value,
            obj(&[
                ("z", PyValue::Int("9".into())),
                ("a", PyValue::Int("2".into())),
                ("m", obj(&[("y", PyValue::int(0)), ("b", PyValue::int(0))])),
            ])
        );
        assert_eq!(value.get("z"), Some(&PyValue::int(9)));
        assert_eq!(value.get("missing"), None);
        assert_eq!(PyValue::int(1).get("z"), None);
    }

    #[test]
    fn integers_keep_every_digit() {
        let big = "-170141183460469231731687303715884105728123456789";
        assert_eq!(loads(big), Ok(PyValue::Int(big.to_owned())));
        assert_eq!(dumps(&loads(big).unwrap()), big);
        let limit = "7".repeat(MAX_INT_DIGITS);
        assert_eq!(loads(&limit), Ok(PyValue::Int(limit.clone())));
        // Python 3.12 raises ValueError past 4,300 digits.
        assert_eq!(
            loads(&"7".repeat(MAX_INT_DIGITS + 1)),
            Err(JsonError::Invalid)
        );
    }

    /// What CPython 3.12.10's `json.loads` accepts and this port cannot hold,
    /// and what it refuses (printed by the scratch probe `c1_py.py`).
    #[test]
    fn python_s_refusals_and_native_limits_are_told_apart() {
        use Beyond::*;
        use JsonError::*;
        let cases: &[(&str, JsonError)] = &[
            ("NaN", BeyondNative(NonFinite)),
            ("Infinity", BeyondNative(NonFinite)),
            ("-Infinity", BeyondNative(NonFinite)),
            ("1e400", BeyondNative(NonFinite)),
            ("[1e400]", BeyondNative(NonFinite)),
            ("-1e400", BeyondNative(NonFinite)),
            (r#""\ud800""#, BeyondNative(LoneSurrogate)),
            (r#""\udc00x""#, BeyondNative(LoneSurrogate)),
            (r#""\ud800\u0041""#, BeyondNative(LoneSurrogate)),
            (
                r#"[{"title": "\ud800", "updated": NaN}]"#,
                BeyondNative(LoneSurrogate),
            ),
            // Python refuses these: invalid wins over a native limit.
            ("[NaN, }", Invalid),
            ("[1,]", Invalid),
            ("01", Invalid),
            ("1.", Invalid),
            ("\u{feff}[]", Invalid),
            ("\"a\u{1}\"", Invalid),
            ("Nan", Invalid),
            ("[1] x", Invalid),
            ("", Invalid),
            (" ", Invalid),
            (r#"{"a" 1}"#, Invalid),
            ("{1: 2}", Invalid),
            (r#""\x""#, Invalid),
            (r#""\u12""#, Invalid),
            ("-", Invalid),
            ("-Infinityx", Invalid),
            ("\u{c}[1]", Invalid),
            ("[1,\u{c}2]", Invalid),
            (r#"{"a":1,}"#, Invalid),
            ("nul", Invalid),
            ("1e", Invalid),
            ("1e+", Invalid),
            (".5", Invalid),
            ("+1", Invalid),
            ("[", Invalid),
            (r#"{"a": 1"#, Invalid),
            ("\"abc", Invalid),
            (r#""\ud800\u12""#, Invalid),
        ];
        for (input, expected) in cases {
            assert_eq!(loads(input), Err(*expected), "{input:?}");
        }
    }

    /// P2 (spec 22.6): only decimal digits (Nd) are digits, by Python 3.12's
    /// table; other numeric characters are refused as Python refuses them.
    #[test]
    fn p2_only_decimal_digits_make_a_numeral() {
        for (c, digit) in [
            ('0', Some(0)),
            ('9', Some(9)),
            ('\u{663}', Some(3)),
            ('\u{1D7CE}', Some(0)),
            ('\u{1D7D7}', Some(9)),
            ('\u{1D7D8}', Some(0)),
            ('\u{1FBF9}', Some(9)),
            ('\u{1FBFA}', None),
            ('\u{B2}', None),
            ('\u{BD}', None),
            ('\u{2167}', None),
            ('\u{2460}', None),
            ('a', None),
        ] {
            assert_eq!(decimal_digit(c), digit, "{c:?}");
        }
        let int = |text: &str| py_int(&PyValue::str(text));
        let float = |text: &str| py_float(&PyValue::str(text));
        for text in [
            "\u{B2}",
            "\u{BD}",
            "1\u{B2}",
            "\u{661}\u{E9}",
            "\u{661} \u{662}",
        ] {
            assert_eq!(int(text), Err(CoerceError::Python), "{text:?}");
            assert_eq!(float(text), Err(CoerceError::Python), "{text:?}");
        }
        assert_eq!(int("\u{661}.\u{665}"), Err(CoerceError::Python));
        assert_eq!(float("\u{661}.\u{665}"), Err(CoerceError::Native));
        assert_eq!(int(" \u{661}\u{662}\u{3000}"), Err(CoerceError::Native));
        assert_eq!(int("\u{661}_\u{662}"), Err(CoerceError::Native));
    }

    #[test]
    fn depth_past_the_limit_is_beyond_native_and_never_recurses() {
        let at_limit = format!("{}{}", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));
        assert!(loads(&at_limit).is_ok());
        let past = format!("{}{}", "[".repeat(MAX_DEPTH + 1), "]".repeat(MAX_DEPTH + 1));
        assert_eq!(loads(&past), Err(JsonError::BeyondNative(Beyond::Depth)));
        // Far deeper than any stack would allow a recursive parser: checked,
        // not kept, and still told from an invalid document.
        let deep = format!("{}1{}", r#"{"a": ["#.repeat(50_000), "]}".repeat(50_000));
        assert_eq!(loads(&deep), Err(JsonError::BeyondNative(Beyond::Depth)));
        let broken = format!("{}1{}", "[".repeat(50_000), "]".repeat(49_999));
        assert_eq!(loads(&broken), Err(JsonError::Invalid));
    }

    /// A balanced document of `brackets` openers and as many closers: the
    /// outcome, the frames it pushed and how long it took.
    fn nested(
        brackets: usize,
        frame_per_bracket: bool,
    ) -> (Result<PyValue, JsonError>, usize, std::time::Duration) {
        let text = format!("{}{}", "[".repeat(brackets), "]".repeat(brackets));
        probe::reset(frame_per_bracket);
        let started = std::time::Instant::now();
        let outcome = loads(&text);
        let elapsed = started.elapsed();
        let frames = probe::frames();
        probe::reset(false);
        (outcome, frames, elapsed)
    }

    /// P1 (spec 22.6): past MAX_DEPTH the parser keeps one bit per open
    /// level, not a frame, so a 10 MB document of brackets pushes at most
    /// MAX_DEPTH frames, is still refused as beyond native (not invalid),
    /// and is read in bounded time. The same check fails against the mutant
    /// that pushes a frame per skimmed bracket again (on 1 MB, to keep the
    /// mutant's memory small).
    #[test]
    fn p1_a_deep_document_costs_bits_past_the_limit_not_frames() {
        let (mutated, frames, _) = nested(512 * 1024, true);
        println!("P1 against the mutant (frame per bracket, 1 MB): {frames} frames");
        assert_eq!(mutated, Err(JsonError::BeyondNative(Beyond::Depth)));
        assert!(frames > MAX_DEPTH, "the check cannot tell the mutant");
        let (outcome, frames, elapsed) = nested(5 * 1024 * 1024, false);
        println!("P1 against the real code (10 MB): {frames} frames in {elapsed:?}");
        assert_eq!(outcome, Err(JsonError::BeyondNative(Beyond::Depth)));
        assert!(frames <= MAX_DEPTH, "{frames} frames");
        assert!(elapsed < std::time::Duration::from_secs(60), "{elapsed:?}");
        // A deep object, a mixed nesting and a broken one past the limit.
        let objects = format!("{}1{}", "{\"a\": ".repeat(200_000), "}".repeat(200_000));
        assert_eq!(loads(&objects), Err(JsonError::BeyondNative(Beyond::Depth)));
        let mixed = format!(
            "{}[1, {{}}]{}",
            "[{\"k\": [".repeat(1_000),
            "]}]".repeat(1_000)
        );
        assert_eq!(loads(&mixed), Err(JsonError::BeyondNative(Beyond::Depth)));
        let crossed = format!("{}{}", "[".repeat(600), "}".repeat(600));
        assert_eq!(loads(&crossed), Err(JsonError::Invalid));
        let unclosed = "[".repeat(600);
        assert_eq!(loads(&unclosed), Err(JsonError::Invalid));
    }

    #[test]
    fn strings_escape_what_python_escapes() {
        let text =
            "a\"b\\c/\u{8}\u{c}\n\r\t\u{1}\u{1f}\u{7f}\u{85}\u{2028}\u{2029}\u{e9}\u{1f600} ~";
        assert_eq!(
            json_string(text),
            r#""a\"b\\c/\b\f\n\r\t\u0001\u001f\u007f\u0085\u2028\u2029\u00e9\ud83d\ude00 ~""#
        );
        // No raw line separator Python's splitlines() would cut at survives dumps.
        let dumped = dumps(&PyValue::str(text));
        assert!(dumped.is_ascii(), "{dumped}");
        assert_eq!(loads(&dumped), Ok(PyValue::str(text)));
    }

    #[test]
    fn non_finite_floats_are_written_as_python_writes_them() {
        assert_eq!(dumps(&PyValue::Float(f64::NAN)), "NaN");
        assert_eq!(dumps(&PyValue::Float(f64::INFINITY)), "Infinity");
        assert_eq!(dumps(&PyValue::Float(f64::NEG_INFINITY)), "-Infinity");
    }

    #[test]
    fn str_follows_python_except_for_containers() {
        assert_eq!(py_str(&PyValue::Null), "None");
        assert_eq!(py_str(&PyValue::Bool(true)), "True");
        assert_eq!(py_str(&PyValue::Bool(false)), "False");
        assert_eq!(py_str(&PyValue::Float(1e16)), "1e+16");
        assert_eq!(py_str(&PyValue::Float(1.0)), "1.0");
        assert_eq!(py_str(&PyValue::Float(-0.0)), "-0.0");
        assert_eq!(py_str(&PyValue::Float(1e-5)), "1e-05");
        assert_eq!(py_str(&PyValue::Int("-12".into())), "-12");
        assert_eq!(py_str(&PyValue::str("x")), "x");
        // D2: JSON text, where Python writes its repr.
        assert_eq!(
            py_str(&loads(r#"[1, "a", {"k": null}]"#).unwrap()),
            r#"[1, "a", {"k": null}]"#
        );
    }

    /// `float(x)` for strings, as CPython 3.12.10 printed it; `None` where it
    /// raised ValueError.
    #[test]
    fn float_of_a_string_follows_python() {
        let cases: &[(&str, Option<f64>)] = &[
            ("1.5", Some(1.5)),
            (" 1.5 ", Some(1.5)),
            ("\u{2003}1.5\u{3000}", Some(1.5)),
            ("\u{85}2", Some(2.0)),
            ("1_000.5", Some(1000.5)),
            ("1__0", None),
            ("_1", None),
            ("1_", None),
            ("1_e5", None),
            ("1e_5", None),
            ("1._5", None),
            ("-inf", Some(f64::NEG_INFINITY)),
            ("Infinity", Some(f64::INFINITY)),
            ("inFINity", Some(f64::INFINITY)),
            ("1e400", Some(f64::INFINITY)),
            (".5", Some(0.5)),
            ("5.", Some(5.0)),
            (".", None),
            ("e5", None),
            ("1e", None),
            ("+1", Some(1.0)),
            ("1.5e+3", Some(1500.0)),
            ("1e1_0", Some(1e10)),
            ("", None),
            ("0x10", None),
            ("\u{1c}1\u{1f}", None),
            (" -0 ", Some(-0.0)),
        ];
        for (text, expected) in cases {
            let got = py_float(&PyValue::str(*text));
            match expected {
                Some(value) => {
                    let got = got.unwrap_or_else(|e| panic!("{text:?}: {e:?}"));
                    assert_eq!(got.to_bits(), value.to_bits(), "{text:?}");
                }
                None => assert_eq!(got, Err(CoerceError::Python), "{text:?}"),
            }
        }
        for nan in ["nan", "+nan", "NaN"] {
            assert!(py_float(&PyValue::str(nan)).unwrap().is_nan(), "{nan}");
        }
        // Python reads Arabic-Indic digits ("١٢" is 12.0); this port cannot.
        assert_eq!(
            py_float(&PyValue::str("\u{661}\u{662}")),
            Err(CoerceError::Native)
        );
        assert_eq!(py_float(&PyValue::str("1\u{e9}")), Err(CoerceError::Python));
    }

    #[test]
    fn float_of_other_values_follows_python() {
        assert_eq!(py_float(&PyValue::Bool(true)), Ok(1.0));
        assert_eq!(py_float(&PyValue::Int("3".into())), Ok(3.0));
        assert_eq!(py_float(&PyValue::Null), Err(CoerceError::Python));
        assert_eq!(py_float(&PyValue::List(vec![])), Err(CoerceError::Python));
        // 2**1024 - 2**970 rounds to 2**1024: OverflowError; one less does not.
        let halfway = "179769313486231580793728971405303415079934132710037826936173778980444968292764750946649017977587207096330286416692887910946555547851940402630657488671505820681908902000708383676273854845817711531764475730270069855571366959622842914819860834936475292719074168444365510704342711559699508093042880177904174497792";
        assert_eq!(
            py_float(&PyValue::Int(halfway.into())),
            Err(CoerceError::Python)
        );
        let below = "179769313486231580793728971405303415079934132710037826936173778980444968292764750946649017977587207096330286416692887910946555547851940402630657488671505820681908902000708383676273854845817711531764475730270069855571366959622842914819860834936475292719074168444365510704342711559699508093042880177904174497791";
        assert_eq!(py_float(&PyValue::Int(below.into())), Ok(f64::MAX));
    }

    /// `int(x)`, as CPython 3.12.10 printed it.
    #[test]
    fn int_follows_python() {
        let strings: &[(&str, Result<i64, CoerceError>)] = &[
            ("3", Ok(3)),
            (" 3 ", Ok(3)),
            ("007", Ok(7)),
            ("0_7", Ok(7)),
            ("+5", Ok(5)),
            ("-5", Ok(-5)),
            ("-0", Ok(0)),
            ("1_000", Ok(1000)),
            ("\u{2003}3\u{3000}", Ok(3)),
            ("9223372036854775807", Ok(i64::MAX)),
            ("-9223372036854775808", Ok(i64::MIN)),
            // Python holds it; i64 cannot.
            ("9223372036854775808", Err(CoerceError::Native)),
            ("3.5", Err(CoerceError::Python)),
            ("1__0", Err(CoerceError::Python)),
            ("", Err(CoerceError::Python)),
            ("\u{1c}3", Err(CoerceError::Python)),
            ("+_1", Err(CoerceError::Python)),
            ("_1", Err(CoerceError::Python)),
            ("1_", Err(CoerceError::Python)),
            ("- 1", Err(CoerceError::Python)),
            ("0x1", Err(CoerceError::Python)),
            ("\u{661}", Err(CoerceError::Native)),
        ];
        for (text, expected) in strings {
            assert_eq!(py_int(&PyValue::str(*text)), *expected, "{text:?}");
        }
        assert_eq!(
            py_int(&PyValue::str("1".repeat(MAX_INT_DIGITS + 1))),
            Err(CoerceError::Python)
        );
        assert_eq!(
            py_int(&PyValue::str(format!("{}1", "0_".repeat(MAX_INT_DIGITS)))),
            Err(CoerceError::Python),
            "underscores do not count, digits do"
        );
        let floats: &[(f64, Result<i64, CoerceError>)] = &[
            (3.9, Ok(3)),
            (-0.5, Ok(0)),
            (-3.9, Ok(-3)),
            (1e30, Err(CoerceError::Native)),
            (f64::NAN, Err(CoerceError::Python)),
            (f64::INFINITY, Err(CoerceError::Python)),
            (-9_223_372_036_854_775_808.0, Ok(i64::MIN)),
            (9_223_372_036_854_775_808.0, Err(CoerceError::Native)),
        ];
        for (number, expected) in floats {
            assert_eq!(py_int(&PyValue::Float(*number)), *expected, "{number}");
        }
        assert_eq!(py_int(&PyValue::Bool(true)), Ok(1));
        assert_eq!(py_int(&PyValue::Int("-42".into())), Ok(-42));
        assert_eq!(
            py_int(&PyValue::Int("123456789012345678901".into())),
            Err(CoerceError::Native)
        );
        assert_eq!(py_int(&PyValue::Null), Err(CoerceError::Python));
    }

    #[test]
    fn truthiness_follows_python() {
        let cases: &[(PyValue, bool)] = &[
            (PyValue::Null, false),
            (PyValue::Bool(true), true),
            (PyValue::Bool(false), false),
            (PyValue::int(0), false),
            (PyValue::int(1), true),
            (PyValue::Float(0.0), false),
            (PyValue::Float(-0.0), false),
            (PyValue::Float(f64::NAN), true),
            (PyValue::str(""), false),
            (PyValue::str("a"), true),
            (PyValue::List(vec![]), false),
            (PyValue::List(vec![PyValue::int(0)]), true),
            (PyValue::Object(vec![]), false),
            (obj(&[("a", PyValue::int(1))]), true),
        ];
        for (value, expected) in cases {
            assert_eq!(py_truthy(value), *expected, "{value:?}");
        }
    }

    #[test]
    fn iteration_follows_python() {
        let strs = |text: &str| py_iter_strs(&loads(text).unwrap());
        assert_eq!(
            strs(r#"["a", 1, null, 1.5, true]"#),
            Ok(vec![
                "a".into(),
                "1".into(),
                "None".into(),
                "1.5".into(),
                "True".into()
            ])
        );
        assert_eq!(strs(r#""ab""#), Ok(vec!["a".into(), "b".into()]));
        assert_eq!(
            strs(r#"{"k": 1, "j": 2}"#),
            Ok(vec!["k".into(), "j".into()])
        );
        for falsy in ["null", "false", "0", "0.0", r#""""#, "[]", "{}"] {
            assert_eq!(strs(falsy), Ok(vec![]), "{falsy}");
        }
        for not_iterable in ["1", "1.5", "true"] {
            assert_eq!(
                strs(not_iterable),
                Err(CoerceError::Python),
                "{not_iterable}"
            );
        }
        let objects = |text: &str| py_iter_objects(&loads(text).unwrap());
        assert_eq!(
            objects(r#"[{"a": 1}, 2, "x", {}]"#),
            Ok(vec![obj(&[("a", PyValue::int(1))]), obj(&[])])
        );
        assert_eq!(objects(r#""text""#), Ok(vec![]));
        assert_eq!(objects(r#"{"a": {}}"#), Ok(vec![]));
        assert_eq!(objects("3"), Err(CoerceError::Python));
        assert_eq!(objects("null"), Ok(vec![]));
    }

    #[test]
    fn counts_follow_history_count() {
        assert_eq!(py_count(&PyValue::Bool(true)), Ok(None));
        assert_eq!(py_count(&PyValue::str("5")), Ok(None));
        assert_eq!(py_count(&PyValue::Null), Ok(None));
        assert_eq!(py_count(&PyValue::int(7)), Ok(Some(7)));
        assert_eq!(py_count(&PyValue::Float(7.9)), Ok(Some(7)));
        assert_eq!(
            py_count(&PyValue::Float(f64::NAN)),
            Err(CoerceError::Python)
        );
        assert_eq!(py_count(&PyValue::Float(1e30)), Err(CoerceError::Native));
    }
}
