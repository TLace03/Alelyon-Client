//! The step scheduler: which things are due at which physics step.
//!
//! The simulator steps physics at `physics_hz`. The control loop runs at
//! `control_hz` (at most the physics rate), and each sense samples at its own
//! rate, delivered on control ticks. This module says, for a physics step index,
//! which of physics, control and each sense are due. It is integer arithmetic
//! only, so every machine and every process agrees exactly.
//!
//! Invariants:
//! - `ticks` is a pure function of `(schedule, step)`: no clock, no state.
//! - Physics is due at every step. Control tick `k` is due at the first physics
//!   step at or after time `k / control_hz`, which is physics step
//!   `ceil(k * physics_hz / control_hz)`. Equivalently, control is due at step
//!   `n` exactly when `floor(n * control_hz / physics_hz)` is greater than it was
//!   at step `n - 1`, and step 0 is always due. Over any long run control is due
//!   `control_hz / physics_hz` of the steps; at equal rates, at every step.
//! - A control tick has an index `k = floor(n * control_hz / physics_hz)`.
//! - A sense is due at physics step `n` exactly when control is due at `n` and
//!   `sim_contract::Rates::is_due(sense, k, control_hz)` holds for the control
//!   index `k`: the contract's rule for senses is the contract's, applied on the
//!   control clock, so the two crates cannot disagree about when a bundle carries
//!   a channel.
//! - An unusable schedule (a rate of 0, or control faster than physics) is not a
//!   running clock: `ticks` says nothing is due, and `validate` says why.

use sim_contract::{RATES_V0, Rates, Sense};

use crate::error::WorldError;

/// The clocks of one simulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Schedule {
    /// Control ticks per simulated second. Must be at least 1 and at most
    /// `physics_hz`.
    pub control_hz: u32,
    /// Physics steps per simulated second. Must be at least 1.
    pub physics_hz: u32,
}

/// What is due at one physics step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ticks {
    /// The physics step index this describes.
    pub step: u64,
    /// Physics is due (always, for a valid schedule).
    pub physics: bool,
    /// Control is due.
    pub control: bool,
    /// The control tick index, when `control` is due; 0 otherwise.
    pub control_step: u64,
    senses: [bool; 6],
}

impl Ticks {
    /// Whether `sense` is due at this step.
    pub fn sense(&self, sense: Sense) -> bool {
        self.senses[sense_index(sense)]
    }

    /// The senses due at this step, in `Sense::ALL` order.
    pub fn due_senses(&self) -> impl Iterator<Item = Sense> + '_ {
        Sense::ALL.into_iter().filter(|s| self.sense(*s))
    }
}

fn sense_index(sense: Sense) -> usize {
    match sense {
        Sense::Sight => 0,
        Sense::Sound => 1,
        Sense::Touch => 2,
        Sense::Smell => 3,
        Sense::Taste => 4,
        Sense::Proprioception => 5,
    }
}

impl Schedule {
    /// A validated schedule.
    pub fn new(control_hz: u32, physics_hz: u32) -> Result<Schedule, WorldError> {
        let s = Schedule {
            control_hz,
            physics_hz,
        };
        s.validate()?;
        Ok(s)
    }

    /// Checks the rates: both at least 1, control at most physics.
    pub fn validate(&self) -> Result<(), WorldError> {
        if self.physics_hz == 0 {
            return Err(WorldError::Schedule {
                reason: "physics_hz must be at least 1",
            });
        }
        if self.control_hz == 0 {
            return Err(WorldError::Schedule {
                reason: "control_hz must be at least 1",
            });
        }
        if self.control_hz > self.physics_hz {
            return Err(WorldError::Schedule {
                reason: "control_hz must not exceed physics_hz",
            });
        }
        Ok(())
    }

    /// What is due at physics step `step`, under contract v0's rates.
    pub fn ticks(&self, step: u64) -> Ticks {
        self.ticks_with(&RATES_V0, step)
    }

    /// [`Schedule::ticks`] under other sense rates.
    pub fn ticks_with(&self, rates: &Rates, step: u64) -> Ticks {
        let mut ticks = Ticks {
            step,
            physics: false,
            control: false,
            control_step: 0,
            senses: [false; 6],
        };
        if self.validate().is_err() {
            return ticks;
        }
        ticks.physics = true;
        let c = u128::from(self.control_hz);
        let p = u128::from(self.physics_hz);
        let through = |n: u64| u128::from(n) * c / p;
        let control_index = through(step);
        let control_due = match step.checked_sub(1) {
            None => true,
            Some(previous) => control_index > through(previous),
        };
        if !control_due {
            return ticks;
        }
        ticks.control = true;
        ticks.control_step = control_index as u64;
        for sense in Sense::ALL {
            ticks.senses[sense_index(sense)] =
                rates.is_due(sense, ticks.control_step, self.control_hz);
        }
        ticks
    }
}
