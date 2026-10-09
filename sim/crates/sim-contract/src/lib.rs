//! The observation contract (v0) between Sinai's simulator and the Sinai side.
//!
//! Sinai is the general-purpose world model of Project Angel. Its simulator
//! (ADR-0042) keeps one world state and reads every sense from it at the same
//! tick. This crate is the versioned schema of what crosses between the two:
//!
//! - **Out of the simulator, once per environment per tick:** a [`Bundle`] of
//!   the senses (camera frames that stay on the device, audio, touch, smell,
//!   taste, proprioception), each channel with its own capture time, under an
//!   [`Identity`].
//! - **Back into the simulator:** an [`Action`] (joint targets and sensor
//!   actions) or a [`Reset`] (a new episode of a scene).
//! - **For evaluation only, never an input to Sinai:** [`GroundTruth`], the
//!   hidden causes behind a tick (including the segmentation frames, whose
//!   per-instance ids would hand the model the object identities that the
//!   object-permanence test measures), and [`Perturbation`], an interference
//!   with one sense used by the cross-sense tests.
//!
//! Like `lattice-protocol`, this crate is a contract and nothing else: plain
//! data, serde in both directions, and `validate` methods that say what a valid
//! value is. It depends on serde and serde_json only, so the simulator, the
//! evaluation harness and tests can all build against it, and a Python training
//! loop can read its JSON (committed under `fixtures/v0/`) with no Rust at all.
//! It opens no device and has no GPU code.
//!
//! The terms of v0 were set with the Sinai session (ADR-0042, "The contract with
//! the Sinai session"). Any change to them is that session's call.
//!
//! Invariants the whole crate keeps:
//! - **Units are SI**: seconds, metres, radians, pascals, kelvin, newtons.
//!   Concentrations are mol/m^3 (smell, at the nose) and mol/L (taste, at the
//!   tongue contact). Poses are in a right-handed world frame with +Z up, and a
//!   quaternion is `[x, y, z, w]` with length 1. Cameras follow the OpenCV
//!   convention (x right, y down, z forward; intrinsics in pixels with pixel
//!   centres at integer coordinates; depth is z in metres, `+inf` for no hit;
//!   RGB is sRGB-encoded `u8`), stated in full on [`FrameRef`].
//! - **Every channel is always present** in a [`Bundle`]. `None` means "not
//!   sampled this tick"; an empty value means "the channel exists, with no
//!   content yet".
//! - **Evaluation-only data never reaches a bundle**: [`GroundTruth`] (which
//!   holds the segmentation frames) and [`Perturbation`] are separate types, and
//!   nothing a bundle holds can contain them.
//! - **Percepts are indexed by PubChem CID, never by words**: smell and taste
//!   vectors are positional over a versioned [`SpeciesTable`].
//! - **No NaN and no infinity** in any value: JSON cannot carry them, so
//!   `validate` refuses them first. (Pixels on the device are not values of this
//!   crate: a depth pixel with no hit is `+inf`.)
//! - **Inputs are read strictly**: [`Action`], [`Reset`] and [`Perturbation`]
//!   (and the types nested in them) refuse unknown fields.
//! - Producers call `validate` before they hand a value over; consumers may
//!   call it on what they receive. Nothing in this crate panics on a bad value.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod action;
mod audio;
mod bundle;
mod check;
mod error;
mod frame;
mod ground_truth;
mod identity;
mod perturbation;
mod proprio;
mod rates;
mod sense;
mod species;
mod touch;

pub use action::{Action, CameraAction, Reset, SceneId, SensorActions, Sniff};
pub use audio::{AUDIO_SAMPLE_RATE_HZ_V0, AudioChunk, AudioContinuity, AudioSamples};
pub use bundle::Bundle;
pub use error::ContractError;
pub use frame::{
    DeviceBufferId, FrameDtype, FrameRef, FrameSemantic, V0_FRAME_HEIGHT, V0_FRAME_WIDTH,
};
pub use ground_truth::{
    ContactEvent, EntityId, GroundTruth, NamedValue, SegFrame, SegTable, ShownSegFrame,
};
pub use identity::{Capture, Identity};
pub use perturbation::{Perturbation, StateDeltaId};
pub use proprio::{BasePose, Proprio};
pub use rates::{RATES_V0, Rates};
pub use sense::Sense;
pub use species::{Species, SpeciesTable, SpeciesVector};
pub use touch::{TouchMap, TouchMaps};

/// The schema version this crate reads and writes. Every [`Identity`] carries
/// it; a value written under another version is refused.
pub const SCHEMA_VERSION: u32 = 0;

/// The longest identifier ([`SceneId`], [`StateDeltaId`]) the contract allows,
/// in bytes.
pub const MAX_IDENTIFIER_LEN: usize = 128;
