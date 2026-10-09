//! Perturbations: evaluation-time interference with one sense.
//!
//! **EVALUATION ONLY.** The registered cross-sense tests (cue conflict and
//! binding) need the senses to disagree on purpose: sight shifted in time
//! against sound, or one sense rendered from a slightly different world than the
//! others. A [`Perturbation`] says which sense and how. Training never perturbs.
//!
//! Applying a perturbation is the simulator's job. This crate only defines the
//! request and validates it, so the simulator and the evaluation harness agree
//! on what a valid request is. A perturbation changes what one sense reports; it
//! never changes the world state the other senses are read from.
//!
//! Invariants:
//! - A perturbation applies to exactly one [`Sense`]. Perturbing sight perturbs
//!   all its channels (RGB and depth) together. The true segmentation in
//!   [`crate::GroundTruth`] is never perturbed: it always describes the true
//!   state, and the segmentation of the frame actually shown under a sight
//!   perturbation is recorded beside it ([`crate::ShownSegFrame`]).
//! - `Offset` shifts the sense's capture time: `seconds` is signed (positive
//!   later, negative earlier), and the sense's capture stamp and content both
//!   move. `Delay` holds the sense back: `seconds` is not negative, and the sense
//!   reports what the world was `seconds` ago while the other senses report now.
//! - `RenderFromPerturbedState` names a registered perturbed copy of the world
//!   state (a state delta); the sense is rendered from that copy.
//! - Every number is finite. A state delta id is an identifier (see
//!   [`StateDeltaId`]).
//! - Unknown fields are refused on read.

use serde::{Deserialize, Serialize};

use crate::check::{finite_f64, identifier, parse};
use crate::{ContractError, Sense};

/// The identity of a registered perturbed copy of the world state.
///
/// Invariants (checked by [`StateDeltaId::validate`]): 1 to 128 characters from
/// `[a-z0-9_.-]`, starting with a letter or digit.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StateDeltaId(pub String);

impl StateDeltaId {
    /// The id as text.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Checks the identifier's shape.
    pub fn validate(&self) -> Result<(), ContractError> {
        identifier("state_delta_id", &self.0)
    }
}

/// One interference with one sense. **Evaluation only.**
///
/// JSON: an object with a `kind` of `"offset"` or `"delay"` (each with `sense`
/// and `seconds`) or `"render_from_perturbed_state"` (with `sense` and
/// `state_delta_id`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Perturbation {
    /// Shift the sense's capture time by `seconds` (signed).
    Offset {
        /// The sense.
        sense: Sense,
        /// Seconds, positive later and negative earlier; finite.
        seconds: f64,
    },
    /// Report what the world was `seconds` ago.
    Delay {
        /// The sense.
        sense: Sense,
        /// Seconds, finite and not negative.
        seconds: f64,
    },
    /// Render the sense from a perturbed copy of the world state.
    RenderFromPerturbedState {
        /// The sense.
        sense: Sense,
        /// The registered perturbed copy.
        state_delta_id: StateDeltaId,
    },
}

impl Perturbation {
    /// The sense this perturbation applies to.
    pub fn sense(&self) -> Sense {
        match self {
            Perturbation::Offset { sense, .. }
            | Perturbation::Delay { sense, .. }
            | Perturbation::RenderFromPerturbedState { sense, .. } => *sense,
        }
    }

    /// Checks the numbers and the state delta id.
    pub fn validate(&self) -> Result<(), ContractError> {
        match self {
            Perturbation::Offset { seconds, .. } => finite_f64("offset.seconds", *seconds),
            Perturbation::Delay { seconds, .. } => {
                finite_f64("delay.seconds", *seconds)?;
                if *seconds < 0.0 {
                    return Err(ContractError::OutOfRange {
                        field: "delay.seconds",
                        reason: "a delay cannot be negative",
                    });
                }
                Ok(())
            }
            Perturbation::RenderFromPerturbedState { state_delta_id, .. } => {
                state_delta_id.validate()
            }
        }
    }

    /// Reads a perturbation from JSON, refusing unknown fields, and validates it.
    pub fn from_json(text: &str) -> Result<Self, ContractError> {
        let perturbation: Perturbation = parse(text)?;
        perturbation.validate()?;
        Ok(perturbation)
    }
}
