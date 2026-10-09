//! The error of the world-state crate.
//!
//! Invariants:
//! - Every failure is a [`WorldError`] value; nothing in this crate panics on a
//!   bad scene, a bad layout request or a bad schedule.
//! - A refusal states the rule it broke in a few words and the numbers involved,
//!   never a free-text value read from a scene.

use std::fmt;

/// Why a request was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorldError {
    /// A layout request that cannot be satisfied (no environments, a size that
    /// overflows).
    Layout {
        /// The rule, in a few words.
        reason: &'static str,
    },
    /// A schedule whose rates are not usable.
    Schedule {
        /// The rule, in a few words.
        reason: &'static str,
    },
    /// A scene and a world that do not describe the same thing.
    SceneMismatch {
        /// The rule, in a few words.
        reason: &'static str,
    },
    /// The scene failed its own validation.
    Scene {
        /// The scene validator's message.
        message: String,
    },
}

impl fmt::Display for WorldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorldError::Layout { reason } => write!(f, "world layout refused: {reason}"),
            WorldError::Schedule { reason } => write!(f, "schedule refused: {reason}"),
            WorldError::SceneMismatch { reason } => write!(f, "scene and world disagree: {reason}"),
            WorldError::Scene { message } => write!(f, "scene refused: {message}"),
        }
    }
}

impl std::error::Error for WorldError {}

impl From<sim_scene::SceneError> for WorldError {
    fn from(error: sim_scene::SceneError) -> Self {
        WorldError::Scene {
            message: error.to_string(),
        }
    }
}
