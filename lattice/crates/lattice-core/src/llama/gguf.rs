//! What a GGUF file says about itself, read from its header without loading
//! it (the chat core's spec §22 LR4; ADR-0041 decision 4). A port of
//! the Python runtime's `gguf_header.py`, held to
//! it by the parity golden `llama/gguf_header.json` (LF7), which runs that
//! reader on fixture
//! files.
//!
//! What is ported, rule for rule:
//! - the header: the `GGUF` magic, version 2 or 3, the tensor and key counts
//!   (each at most 2^24), every key and value, then each tensor's name, its
//!   dimensions (at most 8), its type and its data offset; nothing past the
//!   tensor table is read;
//! - the bounds: a string is at most 1 MiB; an array at most 2^24 items; an
//!   array longer than 64 items is counted, not kept (a fixed-width one is
//!   skipped with a seek, which may pass the end of the file, as Python's
//!   does: only a later read finds the end);
//! - text is UTF-8 with each invalid sequence replaced (`errors="replace"`);
//! - a key given twice keeps its first place and its last value;
//! - [`Header::describe`]: architecture, name, quantisation (llama.cpp's
//!   `general.file_type` names), context length, the parameter count summed
//!   from the tensor table (exact, however large: Python's integers do not
//!   overflow, so neither does this), the tensor count, three raw values and
//!   the version, each as Python's `json.dumps` would write it.
//!
//! Deviations, each named in the golden where a fixture reaches it:
//! - nested arrays deeper than [`MAX_NESTING`] are refused as
//!   [`GgufError::Nesting`] (Python recurses until its own recursion limit);
//! - `str()` of a list (an architecture or name that is an array) is its
//!   JSON text, as the chat store's deviation D2 renders it; `int()` of a
//!   string accepts ASCII digits only (Python also takes other scripts').
//!
//! Errors are named by kind, not by Python's sentence, which quotes
//! `repr()`s this port does not reproduce.

use std::collections::HashMap;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::calc::float_repr;
use crate::chat::pyjson::{self, PyValue};

pub const MAGIC: [u8; 4] = *b"GGUF";
/// Arrays longer than this are counted, not kept.
pub const KEEP_ARRAY_MAX: u64 = 64;
/// A string longer than this is not a header field.
pub const MAX_STRING: u64 = 1 << 20;
/// More items than this is not a header field.
pub const MAX_COUNT: u64 = 1 << 24;
/// A tensor with more dimensions than this is refused.
pub const MAX_DIMS: u32 = 8;
/// Arrays of arrays deeper than this are refused (a native bound).
pub const MAX_NESTING: usize = 64;

const UINT8: u32 = 0;
const INT8: u32 = 1;
const UINT16: u32 = 2;
const INT16: u32 = 3;
const UINT32: u32 = 4;
const INT32: u32 = 5;
const FLOAT32: u32 = 6;
const BOOL: u32 = 7;
const STRING: u32 = 8;
const ARRAY: u32 = 9;
const UINT64: u32 = 10;
const INT64: u32 = 11;
const FLOAT64: u32 = 12;

/// The byte width of a fixed-width value type.
fn scalar_width(kind: u32) -> Option<u64> {
    match kind {
        UINT8 | INT8 | BOOL => Some(1),
        UINT16 | INT16 => Some(2),
        UINT32 | INT32 | FLOAT32 => Some(4),
        UINT64 | INT64 | FLOAT64 => Some(8),
        _ => None,
    }
}

/// `general.file_type` (llama.cpp's `llama_ftype`), as llama.cpp names them.
pub fn file_type_name(ftype: &str) -> Option<&'static str> {
    Some(match ftype.parse::<i64>().ok()? {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4_MOE",
        _ => return None,
    })
}

/// Why a file is not a GGUF header this reader understands (`GGUFError`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GgufError {
    /// The file could not be opened or read.
    Io,
    /// The header ends early.
    EndsEarly,
    /// No `GGUF` magic.
    NotGguf,
    /// A version other than 2 or 3.
    Version,
    /// A tensor or key count over [`MAX_COUNT`].
    ImplausibleCount,
    /// A string longer than [`MAX_STRING`].
    StringTooLong,
    /// An array longer than [`MAX_COUNT`].
    ArrayTooLong,
    /// A value type this reader does not know.
    UnknownType,
    /// A tensor with more than [`MAX_DIMS`] dimensions.
    TooManyDims,
    /// Arrays nested deeper than [`MAX_NESTING`] (native).
    Nesting,
}

impl GgufError {
    /// The kind, as the parity golden names it.
    pub fn kind(self) -> &'static str {
        match self {
            Self::Io => "io",
            Self::EndsEarly => "ends_early",
            Self::NotGguf => "not_gguf",
            Self::Version => "version",
            Self::ImplausibleCount => "implausible_count",
            Self::StringTooLong => "string_too_long",
            Self::ArrayTooLong => "array_too_long",
            Self::UnknownType => "unknown_type",
            Self::TooManyDims => "too_many_dims",
            Self::Nesting => "nesting",
        }
    }
}

/// Why `describe` could not finish: Python's `int()` raised on a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DescribeError {
    /// `ValueError`: text that is not an integer, or a float that is NaN.
    Value,
    /// `TypeError`: a list.
    Type,
    /// `OverflowError`: an infinite float.
    Overflow,
}

impl DescribeError {
    pub fn kind(self) -> &'static str {
        match self {
            Self::Value => "ValueError",
            Self::Type => "TypeError",
            Self::Overflow => "OverflowError",
        }
    }
}

/// One tensor of the table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub ggml_type: u32,
}

/// The metadata and tensor table.
#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub version: u32,
    /// In first-seen order; a repeated key keeps its first place and its last
    /// value. Arrays longer than [`KEEP_ARRAY_MAX`] are absent.
    pub metadata: Vec<(String, PyValue)>,
    pub tensors: Vec<TensorInfo>,
    /// Array-valued keys that were counted, not kept: key and length.
    pub array_lengths: Vec<(String, u64)>,
}

/// Reads with a position and the file's length, so a read past the end is
/// known before it is attempted and a seek may pass the end as Python's does.
struct Reader<R> {
    inner: R,
    at: u64,
    len: u64,
}

impl<R: Read + Seek> Reader<R> {
    fn bytes(&mut self, count: u64) -> Result<Vec<u8>, GgufError> {
        if self.at.checked_add(count).is_none_or(|end| end > self.len) {
            return Err(GgufError::EndsEarly);
        }
        let size = usize::try_from(count).map_err(|_| GgufError::EndsEarly)?;
        let mut buffer = vec![0u8; size];
        self.inner
            .read_exact(&mut buffer)
            .map_err(|_| GgufError::EndsEarly)?;
        self.at += count;
        Ok(buffer)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], GgufError> {
        let bytes = self.bytes(N as u64)?;
        let mut out = [0u8; N];
        out.copy_from_slice(&bytes);
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    /// `handle.seek(count, 1)`: past the end is allowed.
    fn skip(&mut self, count: u64) -> Result<(), GgufError> {
        let to = self.at.checked_add(count).ok_or(GgufError::EndsEarly)?;
        self.inner
            .seek(SeekFrom::Start(to))
            .map_err(|_| GgufError::Io)?;
        self.at = to;
        Ok(())
    }

    fn string(&mut self) -> Result<String, GgufError> {
        let length = self.u64()?;
        if length > MAX_STRING {
            return Err(GgufError::StringTooLong);
        }
        Ok(String::from_utf8_lossy(&self.bytes(length)?).into_owned())
    }

    fn scalar(&mut self, kind: u32) -> Result<PyValue, GgufError> {
        Ok(match kind {
            UINT8 => PyValue::Int(self.array::<1>()?[0].to_string()),
            INT8 => PyValue::Int((self.array::<1>()?[0] as i8).to_string()),
            UINT16 => PyValue::Int(u16::from_le_bytes(self.array()?).to_string()),
            INT16 => PyValue::Int(i16::from_le_bytes(self.array()?).to_string()),
            UINT32 => PyValue::Int(self.u32()?.to_string()),
            INT32 => PyValue::Int(i32::from_le_bytes(self.array()?).to_string()),
            FLOAT32 => PyValue::Float(f64::from(f32::from_le_bytes(self.array()?))),
            // struct's `?`: any byte that is not zero is True.
            BOOL => PyValue::Bool(self.array::<1>()?[0] != 0),
            UINT64 => PyValue::Int(self.u64()?.to_string()),
            INT64 => PyValue::Int(i64::from_le_bytes(self.array()?).to_string()),
            FLOAT64 => PyValue::Float(f64::from_le_bytes(self.array()?)),
            _ => return Err(GgufError::UnknownType),
        })
    }

    /// `_value`: `None` stands for an array that was counted, not kept.
    fn value(
        &mut self,
        kind: u32,
        key: &str,
        lengths: &mut Dict<u64>,
        depth: usize,
    ) -> Result<Option<PyValue>, GgufError> {
        if scalar_width(kind).is_some() {
            return self.scalar(kind).map(Some);
        }
        if kind == STRING {
            return self.string().map(|text| Some(PyValue::Str(text)));
        }
        if kind != ARRAY {
            return Err(GgufError::UnknownType);
        }
        if depth >= MAX_NESTING {
            return Err(GgufError::Nesting);
        }
        let item_kind = self.u32()?;
        let count = self.u64()?;
        if count > MAX_COUNT {
            return Err(GgufError::ArrayTooLong);
        }
        let keep = count <= KEEP_ARRAY_MAX;
        let mut items = Vec::new();
        match scalar_width(item_kind) {
            Some(width) if !keep => self.skip(count * width)?,
            _ => {
                for _ in 0..count {
                    let item = self.value(item_kind, key, lengths, depth + 1)?;
                    if keep {
                        items.push(item.unwrap_or(PyValue::Null));
                    }
                }
            }
        }
        if !keep {
            lengths.set(key, count);
            return Ok(None);
        }
        Ok(Some(PyValue::List(items)))
    }
}

/// A Python dict under construction: entries in first-seen order, found
/// through a map, so N keys cost O(N), as Python's dict does (spec 22.6 P1;
/// a linear scan made a crafted header of N keys cost N^2/2 comparisons).
struct Dict<T> {
    entries: Vec<(String, T)>,
    positions: HashMap<String, usize>,
}

impl<T> Dict<T> {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            positions: HashMap::new(),
        }
    }

    /// A dict assignment: a new key goes last, a known one keeps its place.
    fn set(&mut self, key: &str, value: T) {
        match self.positions.get(key) {
            Some(&index) => self.entries[index].1 = value,
            None => {
                self.positions.insert(key.to_owned(), self.entries.len());
                self.entries.push((key.to_owned(), value));
            }
        }
    }
}

/// `read_header` over any seekable source of `len` bytes.
pub fn read_from<R: Read + Seek>(source: R, len: u64) -> Result<Header, GgufError> {
    let mut reader = Reader {
        inner: source,
        at: 0,
        len,
    };
    if reader.array::<4>()? != MAGIC {
        return Err(GgufError::NotGguf);
    }
    let version = reader.u32()?;
    if !matches!(version, 2 | 3) {
        return Err(GgufError::Version);
    }
    let tensor_count = reader.u64()?;
    let kv_count = reader.u64()?;
    if tensor_count > MAX_COUNT || kv_count > MAX_COUNT {
        return Err(GgufError::ImplausibleCount);
    }
    let mut metadata = Dict::new();
    let mut lengths = Dict::new();
    for _ in 0..kv_count {
        let key = reader.string()?;
        let kind = reader.u32()?;
        if let Some(value) = reader.value(kind, &key, &mut lengths, 0)? {
            metadata.set(&key, value);
        }
    }
    let mut tensors = Vec::new();
    for _ in 0..tensor_count {
        let name = reader.string()?;
        let dims = reader.u32()?;
        if dims > MAX_DIMS {
            return Err(GgufError::TooManyDims);
        }
        let mut shape = Vec::with_capacity(dims as usize);
        for _ in 0..dims {
            shape.push(reader.u64()?);
        }
        let ggml_type = reader.u32()?;
        reader.u64()?; // the data offset: not needed to describe
        tensors.push(TensorInfo {
            name,
            shape,
            ggml_type,
        });
    }
    Ok(Header {
        version,
        metadata: metadata.entries,
        tensors,
        array_lengths: lengths.entries,
    })
}

/// `read_header`: a GGUF file's metadata and tensor table. The file is
/// opened as every model is (LR3a, `files::local_file`): one component at a
/// time, a link followed only to a local path, and no reparse point but a
/// link or a storage-only one passed through. A relative path is made
/// absolute against the current folder first, as Python's `open` reads it.
pub fn read_header(path: &Path) -> Result<Header, GgufError> {
    let absolute = std::path::absolute(path).map_err(|_| GgufError::Io)?;
    let file = super::files::local_file(&absolute).ok_or(GgufError::Io)?;
    let len = file.metadata().map_err(|_| GgufError::Io)?.len();
    read_from(BufReader::new(file), len)
}

/// Python's `str()` of a header value. A list is its JSON text (D2).
fn py_str(value: &PyValue) -> String {
    match value {
        PyValue::Str(text) => text.clone(),
        PyValue::Bool(true) => "True".to_owned(),
        PyValue::Bool(false) => "False".to_owned(),
        PyValue::Int(digits) => digits.clone(),
        PyValue::Float(number) => float_repr(*number),
        PyValue::Null => "None".to_owned(),
        other => pyjson::dumps(other),
    }
}

/// Python's `str(int(value))` for a header value: exact, however large.
fn py_int(value: &PyValue) -> Result<String, DescribeError> {
    match value {
        PyValue::Bool(flag) => Ok(if *flag { "1" } else { "0" }.to_owned()),
        PyValue::Int(digits) => Ok(digits.clone()),
        PyValue::Float(number) if number.is_nan() => Err(DescribeError::Value),
        PyValue::Float(number) if number.is_infinite() => Err(DescribeError::Overflow),
        PyValue::Float(number) => Ok(float_to_integer(*number)),
        PyValue::Str(text) => {
            // `int(str)`: whitespace around, a sign, digits with single
            // underscores between them. ASCII digits only (a deviation).
            // CPython strips only whitespace here (not `str.strip()`'s
            // `\x1c`-`\x1f`), and refuses more than 4,300 digits, leading
            // zeros included (spec 22.6 P2).
            let text = text.trim_matches(char::is_whitespace);
            let (negative, digits) = match text.strip_prefix('-') {
                Some(rest) => (true, rest),
                None => (false, text.strip_prefix('+').unwrap_or(text)),
            };
            let valid = !digits.is_empty()
                && !digits.starts_with('_')
                && !digits.ends_with('_')
                && !digits.contains("__")
                && digits.bytes().all(|b| b.is_ascii_digit() || b == b'_');
            if !valid {
                return Err(DescribeError::Value);
            }
            let plain = digits.replace('_', "");
            if plain.len() > pyjson::MAX_INT_DIGITS {
                return Err(DescribeError::Value);
            }
            let plain = plain.trim_start_matches('0');
            Ok(match (plain.is_empty(), negative) {
                (true, _) => "0".to_owned(),
                (false, true) => format!("-{plain}"),
                (false, false) => plain.to_owned(),
            })
        }
        PyValue::List(_) | PyValue::Object(_) | PyValue::Null => Err(DescribeError::Type),
    }
}

/// `int(x)` of a finite float, toward zero, exact.
fn float_to_integer(number: f64) -> String {
    let truncated = number.trunc();
    if truncated.abs() < 9.0e15 {
        return (truncated as i64).to_string();
    }
    // Beyond 2^53 every float is an integer: mantissa times a power of two.
    let bits = truncated.abs().to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i64 - 1075;
    let mantissa = (bits & ((1u64 << 52) - 1)) | (1u64 << 52);
    let mut value = Natural::from_u64(mantissa);
    for _ in 0..exponent.max(0) {
        value.mul_u64(2);
    }
    let digits = value.decimal();
    if number < 0.0 {
        format!("-{digits}")
    } else {
        digits
    }
}

/// A non-negative integer of any size, for the parameter count.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Natural(Vec<u32>);

impl Natural {
    fn from_u64(value: u64) -> Self {
        Self(vec![value as u32, (value >> 32) as u32])
    }

    fn mul_u64(&mut self, factor: u64) {
        let (low, high) = (factor & 0xffff_ffff, factor >> 32);
        let a = self.mul_u32(low);
        let mut b = self.mul_u32(high);
        b.0.insert(0, 0);
        *self = a;
        self.add(&b);
    }

    fn mul_u32(&self, factor: u64) -> Self {
        let mut out = Vec::with_capacity(self.0.len() + 1);
        let mut carry = 0u64;
        for limb in &self.0 {
            let product = u64::from(*limb) * factor + carry;
            out.push(product as u32);
            carry = product >> 32;
        }
        out.push(carry as u32);
        Self(out)
    }

    fn add(&mut self, other: &Self) {
        let mut carry = 0u64;
        let len = self.0.len().max(other.0.len());
        self.0.resize(len, 0);
        for at in 0..len {
            let sum = u64::from(self.0[at]) + u64::from(*other.0.get(at).unwrap_or(&0)) + carry;
            self.0[at] = sum as u32;
            carry = sum >> 32;
        }
        if carry > 0 {
            self.0.push(carry as u32);
        }
    }

    fn decimal(&self) -> String {
        let mut limbs: Vec<u32> = self.0.clone();
        while limbs.last() == Some(&0) {
            limbs.pop();
        }
        if limbs.is_empty() {
            return "0".to_owned();
        }
        let mut chunks = Vec::new();
        while !limbs.is_empty() {
            let mut remainder = 0u64;
            for limb in limbs.iter_mut().rev() {
                let value = (remainder << 32) | u64::from(*limb);
                *limb = (value / 1_000_000_000) as u32;
                remainder = value % 1_000_000_000;
            }
            chunks.push(remainder);
            while limbs.last() == Some(&0) {
                limbs.pop();
            }
        }
        let mut text = chunks.pop().map(|c| c.to_string()).unwrap_or_default();
        for chunk in chunks.iter().rev() {
            text.push_str(&format!("{chunk:09}"));
        }
        text
    }
}

/// What [`Header::describe`] gives: each field as `json.dumps` writes it.
#[derive(Clone, Debug, PartialEq)]
pub struct Description {
    pub architecture: String,
    pub name: String,
    pub quantization: String,
    /// Python's `int()` of the value, as decimal digits.
    pub context_length: Option<String>,
    /// Decimal digits: the count is exact, however large.
    pub parameters: String,
    pub tensors: usize,
    pub block_count: PyValue,
    pub embedding_length: PyValue,
    pub expert_count: PyValue,
    pub gguf_version: u32,
}

impl Description {
    /// `describe()` as `{field: json.dumps(value)}`, in Python's key order.
    pub fn dumped(&self) -> Vec<(&'static str, String)> {
        vec![
            ("architecture", pyjson::json_string(&self.architecture)),
            ("name", pyjson::json_string(&self.name)),
            ("quantization", pyjson::json_string(&self.quantization)),
            (
                "context_length",
                self.context_length
                    .clone()
                    .unwrap_or_else(|| "null".to_owned()),
            ),
            ("parameters", self.parameters.clone()),
            ("tensors", self.tensors.to_string()),
            ("block_count", pyjson::dumps(&self.block_count)),
            ("embedding_length", pyjson::dumps(&self.embedding_length)),
            ("expert_count", pyjson::dumps(&self.expert_count)),
            ("gguf_version", self.gguf_version.to_string()),
        ]
    }

    /// The context length, when it is one a server can be given.
    pub fn context_tokens(&self) -> Option<u32> {
        self.context_length.as_deref().and_then(|n| n.parse().ok())
    }
}

impl Header {
    fn get(&self, key: &str) -> Option<&PyValue> {
        self.metadata
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }

    /// `architecture`: `str(general.architecture)`, or `""`.
    pub fn architecture(&self) -> String {
        self.get("general.architecture")
            .map(py_str)
            .unwrap_or_default()
    }

    /// `parameters`: the weights, counted from the tensor table.
    pub fn parameters(&self) -> String {
        let mut total = Natural::default();
        for tensor in &self.tensors {
            let mut count = Natural::from_u64(1);
            for dim in &tensor.shape {
                count.mul_u64(*dim);
            }
            total.add(&count);
        }
        total.decimal()
    }

    /// `describe()`.
    pub fn describe(&self) -> Result<Description, DescribeError> {
        let architecture = self.architecture();
        let quantization = match self.get("general.file_type") {
            None => String::new(),
            Some(value) => {
                let ftype = py_int(value)?;
                file_type_name(&ftype)
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("file_type {ftype}"))
            }
        };
        let context_length = match self.get(&format!("{architecture}.context_length")) {
            None => None,
            Some(value) => Some(py_int(value)?),
        };
        let raw = |suffix: &str| {
            self.get(&format!("{architecture}.{suffix}"))
                .cloned()
                .unwrap_or(PyValue::Null)
        };
        Ok(Description {
            name: self.get("general.name").map(py_str).unwrap_or_default(),
            quantization,
            context_length,
            parameters: self.parameters(),
            tensors: self.tensors.len(),
            block_count: raw("block_count"),
            embedding_length: raw("embedding_length"),
            expert_count: raw("expert_count"),
            gguf_version: self.version,
            architecture,
        })
    }
}

/// Read and describe in one step, for a caller that needs only the facts.
pub fn describe_file(path: &Path) -> Option<Description> {
    read_header(path).ok()?.describe().ok()
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    /// A header builder, for the tests below.
    pub(crate) struct Build(Vec<u8>);

    impl Build {
        pub(crate) fn new(version: u32, tensors: u64, kvs: u64) -> Self {
            let mut bytes = MAGIC.to_vec();
            bytes.extend_from_slice(&version.to_le_bytes());
            bytes.extend_from_slice(&tensors.to_le_bytes());
            bytes.extend_from_slice(&kvs.to_le_bytes());
            Self(bytes)
        }

        pub(crate) fn string(mut self, text: &str) -> Self {
            self.0.extend_from_slice(&(text.len() as u64).to_le_bytes());
            self.0.extend_from_slice(text.as_bytes());
            self
        }

        pub(crate) fn u32(mut self, value: u32) -> Self {
            self.0.extend_from_slice(&value.to_le_bytes());
            self
        }

        pub(crate) fn u64(mut self, value: u64) -> Self {
            self.0.extend_from_slice(&value.to_le_bytes());
            self
        }

        pub(crate) fn kv_string(self, key: &str, value: &str) -> Self {
            self.string(key).u32(STRING).string(value)
        }

        pub(crate) fn kv_u32(self, key: &str, value: u32) -> Self {
            self.string(key).u32(UINT32).u32(value)
        }

        pub(crate) fn tensor(mut self, name: &str, shape: &[u64]) -> Self {
            self = self.string(name).u32(shape.len() as u32);
            for dim in shape {
                self = self.u64(*dim);
            }
            self.u32(0).u64(0)
        }

        pub(crate) fn bytes(self) -> Vec<u8> {
            self.0
        }
    }

    fn read(bytes: Vec<u8>) -> Result<Header, GgufError> {
        let len = bytes.len() as u64;
        read_from(Cursor::new(bytes), len)
    }

    #[test]
    fn a_small_header_is_described() {
        let bytes = Build::new(3, 2, 4)
            .kv_string("general.architecture", "llama")
            .kv_string("general.name", "Tiny")
            .kv_u32("general.file_type", 15)
            .kv_u32("llama.context_length", 4096)
            .tensor("a", &[4, 8])
            .tensor("b", &[3])
            .bytes();
        let facts = read(bytes).unwrap().describe().unwrap();
        assert_eq!(facts.architecture, "llama");
        assert_eq!(facts.name, "Tiny");
        assert_eq!(facts.quantization, "Q4_K_M");
        assert_eq!(facts.context_length.as_deref(), Some("4096"));
        assert_eq!(facts.parameters, "35");
        assert_eq!(facts.tensors, 2);
        assert_eq!(facts.block_count, PyValue::Null);
        assert_eq!(facts.context_tokens(), Some(4096));
    }

    #[test]
    fn bounds_and_ends_fail_as_python_fails() {
        assert_eq!(read(Vec::new()), Err(GgufError::EndsEarly));
        assert_eq!(read(b"GGML\x03\0\0\0".to_vec()), Err(GgufError::NotGguf));
        assert_eq!(read(Build::new(1, 0, 0).bytes()), Err(GgufError::Version));
        assert_eq!(
            read(Build::new(3, MAX_COUNT + 1, 0).bytes()),
            Err(GgufError::ImplausibleCount)
        );
        let long = Build::new(3, 0, 1).u64(MAX_STRING + 1).bytes();
        assert_eq!(read(long), Err(GgufError::StringTooLong));
        let at_bound = Build::new(3, 0, 1).u64(MAX_STRING).bytes();
        assert_eq!(read(at_bound), Err(GgufError::EndsEarly));
        let dims = Build::new(3, 1, 0).string("t").u32(MAX_DIMS + 1).bytes();
        assert_eq!(read(dims), Err(GgufError::TooManyDims));
        let unknown = Build::new(3, 0, 1).string("k").u32(13).bytes();
        assert_eq!(read(unknown), Err(GgufError::UnknownType));
    }

    #[test]
    fn a_long_fixed_width_array_is_skipped_even_past_the_end() {
        // 2^24 bytes declared and none present: Python seeks past the end and
        // finds nothing more to read.
        let bytes = Build::new(3, 0, 1)
            .string("big")
            .u32(ARRAY)
            .u32(UINT8)
            .u64(MAX_COUNT)
            .bytes();
        let header = read(bytes).unwrap();
        assert!(header.metadata.is_empty());
        assert_eq!(header.array_lengths, [("big".to_owned(), MAX_COUNT)]);
        let then_more = Build::new(3, 0, 2)
            .string("big")
            .u32(ARRAY)
            .u32(UINT8)
            .u64(100)
            .kv_u32("after", 1)
            .bytes();
        assert_eq!(read(then_more), Err(GgufError::EndsEarly));
    }

    #[test]
    fn the_parameter_count_is_exact_beyond_any_machine_integer() {
        let huge = u64::MAX;
        let bytes = Build::new(3, 2, 0)
            .tensor("a", &[huge, huge, huge])
            .tensor("b", &[2])
            .bytes();
        let header = read(bytes).unwrap();
        // (2^64 - 1)^3 + 2
        assert_eq!(
            header.parameters(),
            "6277101735386680762814942322444851025767571854389858533377"
        );
        let mut n = Natural::from_u64(0);
        n.mul_u64(5);
        assert_eq!(n.decimal(), "0");
    }

    #[test]
    fn int_follows_python_exactly() {
        for (text, expected) in [
            (" 2048 ", Ok("2048")),
            ("+7", Ok("7")),
            ("-3", Ok("-3")),
            ("-0", Ok("0")),
            ("007", Ok("7")),
            ("1_000", Ok("1000")),
            (
                "123456789012345678901234567890",
                Ok("123456789012345678901234567890"),
            ),
            ("1__0", Err(DescribeError::Value)),
            ("_1", Err(DescribeError::Value)),
            ("abc", Err(DescribeError::Value)),
            ("", Err(DescribeError::Value)),
            ("1.5", Err(DescribeError::Value)),
        ] {
            assert_eq!(
                py_int(&PyValue::Str(text.into())),
                expected.map(str::to_owned),
                "{text:?}"
            );
        }
        let float = |x: f64| py_int(&PyValue::Float(x));
        assert_eq!(float(7.9).as_deref(), Ok("7"));
        assert_eq!(float(-7.9).as_deref(), Ok("-7"));
        assert_eq!(float(1e20).as_deref(), Ok("100000000000000000000"));
        assert_eq!(
            float(1e300).unwrap().len(),
            301,
            "int(1e300) has 301 digits in Python"
        );
        assert_eq!(float(f64::NAN), Err(DescribeError::Value));
        assert_eq!(float(f64::INFINITY), Err(DescribeError::Overflow));
        assert_eq!(py_int(&PyValue::List(vec![])), Err(DescribeError::Type));
        assert_eq!(py_int(&PyValue::Bool(true)).as_deref(), Ok("1"));
    }

    /// P1 (spec 22.6): a header of a million distinct keys reads in linear
    /// time, as Python's dict reads it (a linear scan per key would take
    /// about 5e11 comparisons). The read runs on its own thread and must end
    /// within 60 s; a repeated key still keeps its first place and its last
    /// value.
    #[test]
    fn p1_a_million_keys_read_in_linear_time() {
        const KEYS: u64 = 1_000_000;
        let mut bytes = Build::new(3, 0, KEYS + 1).bytes();
        for index in 0..KEYS {
            let key = format!("k{index:07}");
            bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
            bytes.extend_from_slice(key.as_bytes());
            bytes.extend_from_slice(&UINT8.to_le_bytes());
            bytes.push((index % 251) as u8);
        }
        // The first key again, with a new value.
        bytes.extend_from_slice(&8u64.to_le_bytes());
        bytes.extend_from_slice(b"k0000000");
        bytes.extend_from_slice(&UINT8.to_le_bytes());
        bytes.push(250);
        println!("header: {} bytes", bytes.len());
        let (sender, receiver) = std::sync::mpsc::channel();
        let started = std::time::Instant::now();
        std::thread::spawn(move || {
            let _ = sender.send(read(bytes));
        });
        let header = receiver
            .recv_timeout(std::time::Duration::from_secs(60))
            .expect("a million keys must read in linear time (60 s bound)")
            .unwrap();
        println!("read in {:?}", started.elapsed());
        assert_eq!(header.metadata.len(), KEYS as usize);
        assert_eq!(
            header.metadata[0],
            ("k0000000".to_owned(), PyValue::Int("250".into()))
        );
        assert_eq!(
            header.metadata[KEYS as usize - 1].0,
            format!("k{:07}", KEYS - 1)
        );
    }
}
