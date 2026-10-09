//! The world state of Sinai's simulator: layout, host mirror, render view,
//! counter-based randomness and the step scheduler (the CPU reference).
//!
//! ADR-0042 keeps one world state and reads every sense from it. This crate is
//! the CPU side of that state, increment 1a: the structure-of-arrays layout of the
//! state per environment slot ([`WorldLayout`]), a host mirror of it
//! ([`HostWorld`]) that resets deterministically from a scene's initial pose, the
//! render view the renderer reads ([`gather_render_view`], [`cameras_view`], an
//! interface agreed with Lane R), the only random number generator of the
//! simulator (Philox4x32-10, [`rng`]) and the scheduler that says which of physics,
//! control and each sense is due at a physics step ([`Schedule`]). GPU kernels
//! come later, in card windows; they are checked against what is here, bit for
//! bit where this crate says so.
//!
//! Invariants the whole crate keeps:
//! - **Units are SI; quaternions are `[x, y, z, w]`**, like `sim-scene` and
//!   `sim-contract`.
//! - **Determinism**: nothing here reads a clock, the OS, or an unordered map. A
//!   reset is a pure function of `(scene, seed, noise)`; the numbers come from
//!   Philox, addressed by `(seed, env, step, stream, lane)`.
//! - **The render view is an agreed byte layout** (see the `render` module); a
//!   change to it needs Lane R.
//! - No GPU, no kit dependency, no unsafe code; nothing panics on a bad scene.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod error;
mod host;
mod kinematics;
mod layout;
mod render;
pub mod rng;
mod schedule;

pub use error::WorldError;
pub use host::{HostWorld, ResetNoise};
pub use layout::{ENV_STRIDE_ALIGN_BYTES, Field, FieldId, WorldLayout};
pub use render::{
    CAMERA_RECORD_BYTES, CamerasView, POSE_BYTES, RenderView, cameras_view, gather_render_view,
};
pub use rng::{Rng, philox4x32_10};
pub use schedule::{Schedule, Ticks};
