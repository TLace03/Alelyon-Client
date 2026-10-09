//! When each sense is sampled.
//!
//! The senses sample at different rates, and the control loop steps at a rate of
//! its own, so a bundle for a given step carries only the channels that are due.
//! The rule is integer arithmetic, so every machine and every process agrees on
//! it exactly.
//!
//! Invariants:
//! - Contract v0 rates ([`RATES_V0`]): RGB (and with it depth) 10 Hz, audio
//!   16 kHz continuous, touch 100 Hz, smell 10 Hz, taste 10 Hz. The control rate
//!   is a parameter, not part of the table.
//! - Audio and proprioception are due at every step: audio is continuous and
//!   proprioception runs at the control rate.
//! - For the other senses, sample `k` of a sense at `rate` Hz is taken at time
//!   `k / rate` seconds and is delivered with the first step at or after it.
//!   So a sense is due at step `n` exactly when `floor(n * rate / control_hz)` is
//!   greater than it was at step `n - 1`, and step 0 is always due. Over any
//!   long run the sense is due `rate / control_hz` of the steps; a sense whose
//!   rate is at least the control rate is due at every step (a bundle carries
//!   one touch map per tick, not every 100 Hz sample).
//! - `due` is a pure function of its arguments: no clock, no state.

use serde::{Deserialize, Serialize};

use crate::Sense;

/// The sampling rates of the senses, in hertz.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rates {
    /// RGB, and with it depth.
    pub rgb_hz: u32,
    /// Audio samples per second (continuous).
    pub audio_hz: u32,
    /// Touch maps.
    pub touch_hz: u32,
    /// Smell vectors.
    pub smell_hz: u32,
    /// Taste vectors.
    pub taste_hz: u32,
}

/// The rates of contract v0.
pub const RATES_V0: Rates = Rates {
    rgb_hz: 10,
    audio_hz: 16_000,
    touch_hz: 100,
    smell_hz: 10,
    taste_hz: 10,
};

impl Rates {
    /// Whether `sense` is sampled at `step`, under the v0 rates and a control
    /// loop running at `control_hz`. See the module note for the rule.
    ///
    /// A `control_hz` of 0 is not a running clock: nothing is due.
    pub fn due(sense: Sense, step: u64, control_hz: u32) -> bool {
        RATES_V0.is_due(sense, step, control_hz)
    }

    /// [`Rates::due`] for these rates instead of v0's.
    pub fn is_due(&self, sense: Sense, step: u64, control_hz: u32) -> bool {
        if control_hz == 0 {
            return false;
        }
        let rate = match sense {
            Sense::Sound | Sense::Proprioception => return true,
            Sense::Sight => self.rgb_hz,
            Sense::Touch => self.touch_hz,
            Sense::Smell => self.smell_hz,
            Sense::Taste => self.taste_hz,
        };
        if rate == 0 {
            return false;
        }
        let samples_through = |n: u64| u128::from(n) * u128::from(rate) / u128::from(control_hz);
        match step.checked_sub(1) {
            None => true,
            Some(previous) => samples_through(step) > samples_through(previous),
        }
    }

    /// The sampling rate of `sense`, hertz: audio's sample rate for sound and
    /// `control_hz` for proprioception.
    pub fn hz(&self, sense: Sense, control_hz: u32) -> u32 {
        match sense {
            Sense::Sight => self.rgb_hz,
            Sense::Sound => self.audio_hz,
            Sense::Touch => self.touch_hz,
            Sense::Smell => self.smell_hz,
            Sense::Taste => self.taste_hz,
            Sense::Proprioception => control_hz,
        }
    }
}
