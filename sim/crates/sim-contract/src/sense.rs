//! The six senses a bundle carries.
//!
//! Invariants:
//! - The set is closed at contract v0: sight, sound, touch, smell, taste and
//!   proprioception. Adding a sense changes the schema, which is the Sinai
//!   session's call.
//! - Depth is not a sense of its own: it is an extra channel of sight and is
//!   sampled with the RGB frame. Segmentation is not a channel of any sense: it
//!   is evaluation-only ground truth (see [`crate::GroundTruth`]).

use serde::{Deserialize, Serialize};

/// One sense. Used by [`crate::Rates`] to say when a sense is sampled and by
/// [`crate::Perturbation`] to say which sense an evaluation perturbs.
///
/// JSON: the lowercase name (`"sight"`, `"sound"`, `"touch"`, `"smell"`,
/// `"taste"`, `"proprioception"`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sense {
    /// Camera frames: RGB, and optionally depth.
    Sight,
    /// The microphone's audio buffer.
    Sound,
    /// Tactile maps per sensor.
    Touch,
    /// A concentration vector at the nose.
    Smell,
    /// A concentration vector at the tongue contact.
    Taste,
    /// Joint positions and velocities, and the optional base pose.
    Proprioception,
}

impl Sense {
    /// Every sense, in the order the bundle lists them.
    pub const ALL: [Sense; 6] = [
        Sense::Sight,
        Sense::Sound,
        Sense::Touch,
        Sense::Smell,
        Sense::Taste,
        Sense::Proprioception,
    ];
}
