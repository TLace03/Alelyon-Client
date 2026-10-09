//! The one scene description for Sinai's simulator.
//!
//! Physics, rendering and the senses are views of one world state (ADR-0042), and
//! all of them start from one description of the scene: this crate. A [`Scene`]
//! lists bodies, joints, geoms, meshes, materials (mechanical, thermal, acoustic
//! and optical in one record), renderable instances, cameras and actuators, in SI
//! units, with every invariant validated and serde JSON that round-trips exactly
//! and refuses unknown fields. [`mjcf::load`] imports a subset of MuJoCo's MJCF,
//! ported from MuJoCo's own compiler and held to MuJoCo's results by a parity
//! test.
//!
//! Invariants the whole crate keeps:
//! - **Units are SI**: metres, kilograms, seconds, radians, kelvin. World frame is
//!   right-handed with +Z up. A quaternion is `[x, y, z, w]` (scalar last,
//!   Hamilton, unit length), the convention of `sim-contract` v0.
//! - **The world is not a body.** A field that can name the world is an
//!   `Option<BodyId>` and `None` is the world.
//! - **Validation is total.** `Scene::validate` checks every id, every range and
//!   every physical constraint, and returns the first violation with a path into
//!   the scene; nothing in the crate panics on a bad scene.
//! - **Nothing is dropped silently.** What an imported document carries that the
//!   scene cannot represent is listed in `Scene::unsupported`, or refused.
//! - **Deterministic.** Importing the same document twice gives equal scenes; no
//!   clock, no randomness, no iteration over unordered maps decides an output.
//! - No GPU, no kit dependency, no unsafe code.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod body;
mod error;
mod ids;
pub mod material;
pub mod mjcf;
pub mod pose;
mod scene;
mod validate;

pub use body::{
    Body, DEFAULT_SOLIMP, DEFAULT_SOLREF, Geom, Inertial, Joint, JointKind, Mesh, Shape,
};
pub use error::{MjcfError, MjcfErrorKind, Result, SceneError};
pub use ids::{BodyId, GeomId, JointId, MaterialId, MeshId};
pub use material::{Acoustic, Checker, Material, Mechanical, Optical, Thermal, srgb_to_linear};
pub use scene::{
    Actuator, ActuatorKind, Camera, CameraMount, Cone, ContactExclude, DEFAULT_GRAVITY,
    DEFAULT_TIMESTEP_S, Instance, Integrator, SCENE_VERSION, Scene, ShapeRef, Solver,
    SolverOptions, Tendon, TendonJoint, Unsupported,
};
pub use validate::QUAT_TOLERANCE;
