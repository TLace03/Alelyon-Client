//! The bundle: everything one environment's senses report in one tick.
//!
//! A bundle is the only thing Sinai's model sees of the world. Every channel is
//! views of the same world state at the same tick; no sense runs a simulation of
//! its own.
//!
//! Invariants:
//! - **Every channel is always present.** A channel of `Option` type set to
//!   `None` means "not sampled this tick" (the senses sample at different
//!   rates, see [`crate::Rates`]); a channel that is present but empty (no
//!   audio samples yet, no joints, no sensors, an empty species vector) means
//!   "the channel exists and has no content yet". Phase 1 needs every channel
//!   in the schema from day one, so a producer never leaves one out: the
//!   JSON of a bundle has all eight keys, with `null` for `None`, and reading a
//!   bundle with a key missing is an error.
//! - Every sampled channel carries its own capture time ([`Capture`]).
//! - Depth is an extra sense channel of the RGB frame (`sight`): it is sent only
//!   together with it.
//! - **Segmentation is not a channel of the bundle.** Per-instance ids that
//!   persist through occlusion would hand the model the object identities that
//!   the object-permanence test measures, so segmentation is evaluation-only
//!   data and lives in [`crate::GroundTruth`] (`segmentation`). A bundle has no
//!   field that can hold a segmentation frame; the `evaluation_only` test fails
//!   to compile if one is added back.
//! - A bundle holds observations and nothing else. The hidden causes behind
//!   them are [`crate::GroundTruth`], a separate type that no field of a bundle
//!   can hold, directly or through any type a bundle holds. See the
//!   `evaluation_only` test, which fails to compile if that changes.

use serde::{Deserialize, Deserializer, Serialize};

use crate::{
    AudioChunk, Capture, ContractError, FrameRef, FrameSemantic, Identity, Proprio, SpeciesTable,
    SpeciesVector, TouchMaps,
};

/// Reads an `Option` field that must be present in the JSON (as a value or as
/// `null`). Serde would otherwise read a missing `Option` key as `None`, which
/// would let "not sampled" and "left out" look the same.
pub(crate) fn present<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

/// One environment's observation for one tick.
///
/// Invariants are those of the module; [`Bundle::validate`] checks them and the
/// invariants of every channel's value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bundle {
    /// Who and when.
    pub identity: Identity,
    /// The RGB frame (448 x 448 `u8` x 3, device-resident), or `None` if sight
    /// was not sampled this tick.
    #[serde(deserialize_with = "present")]
    pub rgb: Option<Capture<FrameRef>>,
    /// Depth in metres (`f16`), pixel-aligned with `rgb`; only with `rgb`.
    #[serde(deserialize_with = "present")]
    pub depth: Option<Capture<FrameRef>>,
    /// This tick's chunk of the continuous audio stream.
    pub audio: Capture<AudioChunk>,
    /// The body's joints and base.
    pub proprioception: Capture<Proprio>,
    /// Tactile maps, or `None` if touch was not sampled this tick.
    #[serde(deserialize_with = "present")]
    pub touch: Option<Capture<TouchMaps>>,
    /// Odorant concentrations at the nose, mol/m^3, or `None` if smell was not
    /// sampled this tick.
    #[serde(deserialize_with = "present")]
    pub smell: Option<Capture<SpeciesVector>>,
    /// Tastant concentrations at the tongue contact, mol/L, or `None` if taste
    /// was not sampled this tick.
    #[serde(deserialize_with = "present")]
    pub taste: Option<Capture<SpeciesVector>>,
}

impl Bundle {
    /// A bundle with every channel present and none with content: the optional
    /// channels are `None`, audio is an empty chunk at `audio_first_sample_index`
    /// and proprioception is empty, each captured at the tick's time.
    pub fn empty(identity: Identity, audio_first_sample_index: u64) -> Self {
        let at = identity.sim_time_s;
        Self {
            identity,
            rgb: None,
            depth: None,
            audio: Capture::new(at, AudioChunk::empty(audio_first_sample_index)),
            proprioception: Capture::new(at, Proprio::empty()),
            touch: None,
            smell: None,
            taste: None,
        }
    }

    /// Checks the identity, every channel's capture time and value, and the
    /// rule that depth comes only with RGB. Smell and taste vectors are checked
    /// against the species tables they are indexed by.
    pub fn validate(
        &self,
        smell_table: &SpeciesTable,
        taste_table: &SpeciesTable,
    ) -> Result<(), ContractError> {
        self.identity.validate()?;
        for (channel, capture, semantic) in [
            ("rgb", &self.rgb, FrameSemantic::Rgb),
            ("depth", &self.depth, FrameSemantic::DepthMetres),
        ] {
            if let Some(capture) = capture {
                capture
                    .validate_time()
                    .and_then(|()| capture.value.validate_v0(semantic))
                    .map_err(|e| e.in_channel(channel))?;
            }
        }
        if self.rgb.is_none() && self.depth.is_some() {
            return Err(ContractError::ExtraChannelWithoutRgb { channel: "depth" });
        }
        self.audio
            .validate_time()
            .and_then(|()| self.audio.value.validate())
            .map_err(|e| e.in_channel("audio"))?;
        self.proprioception
            .validate_time()
            .and_then(|()| self.proprioception.value.validate())
            .map_err(|e| e.in_channel("proprioception"))?;
        if let Some(touch) = &self.touch {
            touch
                .validate_time()
                .and_then(|()| touch.value.validate())
                .map_err(|e| e.in_channel("touch"))?;
        }
        for (channel, capture, table) in [
            ("smell", &self.smell, smell_table),
            ("taste", &self.taste, taste_table),
        ] {
            if let Some(capture) = capture {
                capture
                    .validate_time()
                    .and_then(|()| capture.value.validate(table))
                    .map_err(|e| e.in_channel(channel))?;
            }
        }
        Ok(())
    }
}
