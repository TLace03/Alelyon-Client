//! The errors of the scene description and of the MJCF importer.
//!
//! A refusal names where it happened and which rule it broke, so an author can
//! fix the scene and a test can match on the rule that fired.
//!
//! Invariants:
//! - [`SceneError::Invalid`] always carries a `path` into the scene
//!   (`bodies[3].inertial.mass_kg`) and a short reason; it never carries a
//!   value read from a file other than numbers.
//! - [`MjcfError`] always carries the kind of refusal, the XML path of the
//!   element and, when the XML parser could place it, the 1-based line.
//! - Nothing in this crate panics on a bad scene or a bad XML document; every
//!   failure is one of these values.

use std::fmt;

/// Result alias used across the crate.
pub type Result<T> = std::result::Result<T, SceneError>;

/// Why a scene was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum SceneError {
    /// A field broke one of the scene's invariants.
    Invalid {
        /// Where, as a path into the scene (`geoms[2].shape.r`).
        path: String,
        /// The rule, in a few words.
        reason: String,
    },
    /// The JSON could not be read as a scene (syntax, a missing field, an
    /// unknown field, a wrong type).
    Json {
        /// The parser's message.
        message: String,
    },
    /// The MJCF importer refused the document.
    Mjcf(MjcfError),
}

impl SceneError {
    pub(crate) fn invalid(path: impl Into<String>, reason: impl Into<String>) -> Self {
        SceneError::Invalid {
            path: path.into(),
            reason: reason.into(),
        }
    }
}

impl fmt::Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SceneError::Invalid { path, reason } => write!(f, "invalid scene at {path}: {reason}"),
            SceneError::Json { message } => write!(f, "scene JSON refused: {message}"),
            SceneError::Mjcf(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for SceneError {}

impl From<MjcfError> for SceneError {
    fn from(error: MjcfError) -> Self {
        SceneError::Mjcf(error)
    }
}

/// The rule an MJCF document broke.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MjcfErrorKind {
    /// The text is not well-formed XML (or has a DTD, which is refused).
    Xml,
    /// An element outside the importer's subset and outside its recorded list.
    UnsupportedElement,
    /// An attribute outside the importer's subset and outside its recorded list.
    UnsupportedAttribute,
    /// An unsupported item that the importer records in `Scene::unsupported`,
    /// refused because the caller asked for strict import.
    Strict,
    /// A required attribute is absent.
    MissingAttribute,
    /// An attribute value is malformed or out of range.
    BadValue,
    /// A name refers to something that does not exist (a class, a joint, a
    /// mesh, a material).
    UnknownReference,
    /// A name is used twice where it must be unique.
    Duplicate,
    /// Two attributes (or an attribute and the element's context) contradict
    /// each other, or a physical rule of the model is broken (a mass that is
    /// negative, an inertia that breaks the triangle inequality).
    Inconsistent,
    /// A mesh file could not be read or decoded.
    Asset,
}

/// An MJCF document the importer refused.
#[derive(Clone, Debug, PartialEq)]
pub struct MjcfError {
    /// The rule that fired.
    pub kind: MjcfErrorKind,
    /// 1-based line in the document, or 0 when the error is not tied to one.
    pub line: u32,
    /// XML path of the element (`worldbody/body[torso]/geom[torso]`).
    pub path: String,
    /// What is wrong, precisely.
    pub message: String,
}

impl MjcfError {
    pub(crate) fn new(
        kind: MjcfErrorKind,
        line: u32,
        path: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        MjcfError {
            kind,
            line,
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for MjcfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MJCF {:?}", self.kind)?;
        if self.line > 0 {
            write!(f, " at line {}", self.line)?;
        }
        if !self.path.is_empty() {
            write!(f, " ({})", self.path)?;
        }
        write!(f, ": {}", self.message)
    }
}

impl std::error::Error for MjcfError {}
