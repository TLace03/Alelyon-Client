//! XML element wrapper, attribute policy and attribute value parsing.
//!
//! Invariants:
//! - Every element the importer reads is wrapped in an [`E`] that knows its XML
//!   path and 1-based line, so every error is precise.
//! - [`Ctx::check_attrs`] is the one gate every attribute passes: an attribute is
//!   *supported* (read by the importer), *recorded* (known to MuJoCo, not modelled
//!   yet: appended to `Scene::unsupported`, or refused under strict import) or
//!   *refused* (anything else: `MjcfErrorKind::UnsupportedAttribute`). There is no
//!   fourth case where an attribute is read as nothing.
//! - Numbers follow MuJoCo's attribute reader (`ReadAttr`, xml_util.cc:853-873):
//!   whitespace-separated, at most `max` values, at least `min` (an `exact`
//!   attribute has `min == max`). NaN and infinity are refused, where MuJoCo warns.

use std::str::FromStr;

use roxmltree::{Document, Node};

use crate::error::{MjcfError, MjcfErrorKind};
use crate::scene::Unsupported;

/// Shared state of one import.
pub(crate) struct Ctx {
    pub strict: bool,
    pub recorded: Vec<Unsupported>,
    /// Byte offset of the start of each line of the source, for line numbers in
    /// O(log n): `Document::text_pos_at` rescans the text from its start, which
    /// would make a model with n elements cost O(n^2).
    line_starts: Vec<usize>,
}

/// An element with its path and line.
#[derive(Clone)]
pub(crate) struct E<'d, 'i> {
    pub node: Node<'d, 'i>,
    pub path: String,
    pub line: u32,
}

impl Ctx {
    pub fn new(doc: &Document<'_>, strict: bool) -> Self {
        let mut line_starts = vec![0usize];
        line_starts.extend(
            doc.input_text()
                .bytes()
                .enumerate()
                .filter(|(_, b)| *b == 10)
                .map(|(i, _)| i + 1),
        );
        Ctx {
            strict,
            recorded: Vec::new(),
            line_starts,
        }
    }

    /// The 1-based line containing byte `offset` of the source.
    pub fn line_at(&self, offset: usize) -> u32 {
        self.line_starts.partition_point(|&start| start <= offset) as u32
    }

    /// Wraps `node` as a child of the element at `parent_path`.
    pub fn elem<'d, 'i>(&self, node: Node<'d, 'i>, parent_path: &str) -> E<'d, 'i> {
        let tag = node.tag_name().name();
        let label = node
            .attribute("name")
            .or_else(|| {
                if tag == "default" {
                    node.attribute("class")
                } else {
                    None
                }
            })
            .map(|n| format!("{tag}[{n}]"))
            .unwrap_or_else(|| tag.to_string());
        let path = if parent_path.is_empty() {
            label
        } else {
            format!("{parent_path}/{label}")
        };
        let line = self.line_at(node.range().start);
        E { node, path, line }
    }

    /// Records an unsupported item, or refuses it under strict import.
    pub fn record(&mut self, e: &E, item: &str, reason: &str) -> Result<(), MjcfError> {
        if self.strict {
            return Err(e.err(
                MjcfErrorKind::Strict,
                format!("{item} is not modelled yet ({reason}); strict import refuses it"),
            ));
        }
        self.recorded.push(Unsupported {
            path: e.path.clone(),
            item: item.to_string(),
            line: e.line,
            reason: reason.to_string(),
        });
        Ok(())
    }

    /// Records a whole element (its attributes and children are not read).
    pub fn record_element(&mut self, e: &E, reason: &str) -> Result<(), MjcfError> {
        self.record(e, "element", reason)
    }

    /// The attribute gate: see the module note.
    pub fn check_attrs(
        &mut self,
        e: &E,
        supported: &[&str],
        recorded: &[(&str, &str)],
    ) -> Result<(), MjcfError> {
        for attr in e.node.attributes() {
            let name = attr.name();
            if supported.contains(&name) {
                continue;
            }
            if let Some((_, reason)) = recorded.iter().find(|(n, _)| *n == name) {
                self.record(e, &format!("@{name}"), reason)?;
                continue;
            }
            return Err(e.err(
                MjcfErrorKind::UnsupportedAttribute,
                format!(
                    "attribute '{name}' of <{}> is not supported by this importer",
                    e.node.tag_name().name()
                ),
            ));
        }
        Ok(())
    }
}

impl<'d, 'i> E<'d, 'i> {
    pub fn tag(&self) -> &'d str {
        self.node.tag_name().name()
    }

    pub fn err(&self, kind: MjcfErrorKind, message: impl Into<String>) -> MjcfError {
        MjcfError::new(kind, self.line, self.path.clone(), message)
    }

    pub fn bad(&self, message: impl Into<String>) -> MjcfError {
        self.err(MjcfErrorKind::BadValue, message)
    }

    /// The element's child elements.
    pub fn children(&self) -> impl Iterator<Item = Node<'d, 'i>> + use<'d, 'i> {
        self.node.children().filter(|n| n.is_element())
    }

    pub fn attr(&self, name: &str) -> Option<&'d str> {
        self.node.attribute(name)
    }

    pub fn has(&self, name: &str) -> bool {
        self.node.attribute(name).is_some()
    }

    pub fn require(&self, name: &str) -> Result<&'d str, MjcfError> {
        self.attr(name).ok_or_else(|| {
            self.err(
                MjcfErrorKind::MissingAttribute,
                format!("required attribute '{name}' is missing"),
            )
        })
    }

    /// `name` parsed as `min..=max` numbers (`MuJoCo`'s `ReadAttr`).
    pub fn nums(&self, name: &str, min: usize, max: usize) -> Result<Option<Vec<f64>>, MjcfError> {
        let Some(text) = self.attr(name) else {
            return Ok(None);
        };
        let mut out = Vec::new();
        for token in text.split_ascii_whitespace() {
            let v = f64::from_str(token).map_err(|_| {
                self.bad(format!(
                    "bad format in attribute '{name}': '{token}' is not a number"
                ))
            })?;
            if !v.is_finite() {
                return Err(self.bad(format!(
                    "attribute '{name}' holds a value that is not finite"
                )));
            }
            out.push(v);
        }
        if out.is_empty() {
            return Err(self.bad(format!("attribute '{name}' is empty")));
        }
        if out.len() < min {
            return Err(self.bad(format!("attribute '{name}' does not have enough data")));
        }
        if out.len() > max {
            return Err(self.bad(format!("attribute '{name}' has too much data")));
        }
        Ok(Some(out))
    }

    /// `name` as exactly `N` numbers.
    pub fn exact<const N: usize>(&self, name: &str) -> Result<Option<[f64; N]>, MjcfError> {
        Ok(self.nums(name, N, N)?.map(|v| {
            let mut a = [0.0; N];
            a.copy_from_slice(&v);
            a
        }))
    }

    /// `name` as exactly `N` single-precision numbers, for MuJoCo's `float`
    /// attributes. Each token is parsed straight to `f32` (correctly rounded), as
    /// MuJoCo's reader does with `std::istringstream >> float`; parsing to `f64` and
    /// narrowing would round twice. NaN and infinity are refused, as in [`Self::nums`].
    pub fn floats<const N: usize>(&self, name: &str) -> Result<Option<[f32; N]>, MjcfError> {
        let Some(text) = self.attr(name) else {
            return Ok(None);
        };
        let mut out = [0f32; N];
        let mut n = 0usize;
        for token in text.split_ascii_whitespace() {
            if n >= N {
                return Err(self.bad(format!("attribute '{name}' has too much data")));
            }
            let v = f32::from_str(token).map_err(|_| {
                self.bad(format!(
                    "bad format in attribute '{name}': '{token}' is not a number"
                ))
            })?;
            if !v.is_finite() {
                return Err(self.bad(format!(
                    "attribute '{name}' holds a value that is not finite"
                )));
            }
            out[n] = v;
            n += 1;
        }
        if n == 0 {
            return Err(self.bad(format!("attribute '{name}' is empty")));
        }
        if n < N {
            return Err(self.bad(format!("attribute '{name}' does not have enough data")));
        }
        Ok(Some(out))
    }

    /// `name` as one number.
    pub fn num(&self, name: &str) -> Result<Option<f64>, MjcfError> {
        Ok(self.nums(name, 1, 1)?.map(|v| v[0]))
    }

    /// `name` as one `i32`.
    pub fn int(&self, name: &str) -> Result<Option<i32>, MjcfError> {
        let Some(text) = self.attr(name) else {
            return Ok(None);
        };
        let mut tokens = text.split_ascii_whitespace();
        match (tokens.next(), tokens.next()) {
            (Some(t), None) => i32::from_str(t).map(Some).map_err(|_| {
                self.bad(format!(
                    "bad format in attribute '{name}': '{t}' is not an integer"
                ))
            }),
            (None, _) => Err(self.bad(format!("attribute '{name}' is empty"))),
            (Some(_), Some(_)) => Err(self.bad(format!("attribute '{name}' has too much data"))),
        }
    }

    /// `name` as exactly `N` integers.
    pub fn ints<const N: usize>(&self, name: &str) -> Result<Option<[i32; N]>, MjcfError> {
        let Some(text) = self.attr(name) else {
            return Ok(None);
        };
        let mut out = [0i32; N];
        let mut n = 0usize;
        for token in text.split_ascii_whitespace() {
            if n >= N {
                return Err(self.bad(format!("attribute '{name}' has too much data")));
            }
            out[n] = i32::from_str(token).map_err(|_| {
                self.bad(format!(
                    "bad format in attribute '{name}': '{token}' is not an integer"
                ))
            })?;
            n += 1;
        }
        if n < N {
            return Err(self.bad(format!("attribute '{name}' does not have enough data")));
        }
        Ok(Some(out))
    }

    /// `name` as `true` or `false` (MuJoCo's `bool_map`).
    pub fn boolean(&self, name: &str) -> Result<Option<bool>, MjcfError> {
        match self.attr(name) {
            None => Ok(None),
            Some("true") => Ok(Some(true)),
            Some("false") => Ok(Some(false)),
            Some(other) => {
                Err(self.bad(format!("invalid keyword '{other}' in attribute '{name}'")))
            }
        }
    }

    /// `name` as one of `keywords`; returns the matching value.
    pub fn keyword<T: Copy>(
        &self,
        name: &str,
        keywords: &[(&str, T)],
    ) -> Result<Option<T>, MjcfError> {
        match self.attr(name) {
            None => Ok(None),
            Some(text) => keywords
                .iter()
                .find(|(k, _)| *k == text)
                .map(|(_, v)| Some(*v))
                .ok_or_else(|| self.bad(format!("invalid keyword '{text}' in attribute '{name}'"))),
        }
    }
}
