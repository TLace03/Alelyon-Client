//! Who a bundle is for, and when each sense of it was captured.
//!
//! Invariants:
//! - Every bundle carries an [`Identity`], and every identity carries the schema
//!   version it was written under. A reader checks the version before it reads
//!   anything else.
//! - The senses sample at different rates, so each channel of a bundle carries
//!   its own capture time in a [`Capture`]; `Identity::sim_time_s` is the time of
//!   the tick the bundle belongs to.
//! - Simulated time is simulated: nothing here is a wall-clock reading.

use serde::{Deserialize, Serialize};

use crate::check::finite_f64;
use crate::{ContractError, SCHEMA_VERSION};

/// The identity every bundle (and every ground-truth record) carries.
///
/// Invariants (checked by [`Identity::validate`]):
/// - `schema_version` equals [`SCHEMA_VERSION`].
/// - `sim_time_s` is finite and not negative.
///
/// By convention, not checked here: `step` counts control ticks from 0 within an
/// episode, and `sim_time_s = step / control_hz`. `seed` is the seed the episode
/// was reset with (`Reset::seed`); the same seed on the same device and driver
/// reproduces the episode bit for bit.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Identity {
    /// The schema version this value was written under.
    pub schema_version: u32,
    /// The environment slot (one of the many that step together).
    pub env_id: u32,
    /// The episode within the environment slot; a reset starts a new one.
    pub episode_id: u64,
    /// Control ticks since the episode began.
    pub step: u64,
    /// Simulated time of this tick, seconds since the episode began.
    pub sim_time_s: f64,
    /// The episode's seed.
    pub seed: u64,
}

impl Identity {
    /// An identity under the current schema version.
    pub fn new(env_id: u32, episode_id: u64, step: u64, sim_time_s: f64, seed: u64) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            env_id,
            episode_id,
            step,
            sim_time_s,
            seed,
        }
    }

    /// Checks the schema version and the simulated time.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(ContractError::SchemaVersion {
                found: self.schema_version,
            });
        }
        finite_f64("identity.sim_time_s", self.sim_time_s)?;
        if self.sim_time_s < 0.0 {
            return Err(ContractError::OutOfRange {
                field: "identity.sim_time_s",
                reason: "simulated time cannot be negative",
            });
        }
        Ok(())
    }
}

/// One channel's value and the simulated time it was captured at.
///
/// Invariants:
/// - `captured_at_s` is finite. For an unperturbed bundle it is the time of the
///   sample the channel carries, which is at or before `Identity::sim_time_s`
///   and later than the previous tick: each sense carries its own capture time
///   within the tick. An evaluation [`crate::Perturbation`] may move it.
/// - Nothing else is promised about the value; each channel's own type states
///   its invariants.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Capture<T> {
    /// Simulated time of the sample, in seconds.
    pub captured_at_s: f64,
    /// The sample.
    pub value: T,
}

impl<T> Capture<T> {
    /// A sample captured at `captured_at_s`.
    pub fn new(captured_at_s: f64, value: T) -> Self {
        Self {
            captured_at_s,
            value,
        }
    }

    /// Checks that the capture time is a number.
    pub fn validate_time(&self) -> Result<(), ContractError> {
        finite_f64("captured_at_s", self.captured_at_s)
    }
}
