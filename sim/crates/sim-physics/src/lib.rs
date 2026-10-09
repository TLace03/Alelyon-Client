//! The CPU reference of Sinai's physics core (ADR-0042, lane P, phases 1b, 1c-i and 1c-ii).
//!
//! MuJoCo's formulation: reduced coordinates (hinge, slide, ball and free joints
//! in a tree of bodies), the composite rigid body inertia matrix, Newton-Euler bias
//! forces, joint springs and dampers, motors and position servos, soft constraints
//! for joint and tendon limits, for dry friction loss of joints and fixed
//! tendons and for contacts, the Newton and conjugate-gradient primal solvers (with
//! pyramidal and elliptic friction cones), and Euler or RK4 integration. It is a port of
//! MuJoCo 3.14.0 (Apache-2.0; see `NOTICE`), held to MuJoCo's own numbers by golden
//! files that `tools/sim_physics_mujoco_golden.py`,
//! `tools/sim_constraints_mujoco_golden.py` and `tools/sim_contacts_mujoco_golden.py`
//! write with the MuJoCo oracle.
//!
//! Collision detection (phase 1c-ii) has eleven primitive colliders (plane-sphere,
//! plane-capsule, plane-cylinder, plane-box, sphere-sphere, sphere-capsule,
//! sphere-cylinder, sphere-box, capsule-capsule, capsule-box and box-box), MuJoCo's filters
//! (bitmask, weld, static, parent, exclude, bounding sphere) and MuJoCo's contact order,
//! built so that a later GPU port is mechanical: a static candidate list replaces the
//! broadphase, and the step is two passes of units that are independent of one another and may
//! run in any order (one per candidate each, a prefix sum of the counts between them; pass 2
//! is out of place). [`Model::compile`] returns a [`NotModelled`] entry for every
//! geom pair that would need MuJoCo's convex path (an ellipsoid or a mesh, capsule-cylinder,
//! cylinder-cylinder, cylinder-box), a fixed tendon's spring, damper and armature, a request
//! for the PGS solver, and every unsupported record of the scene, so nothing is ignored
//! silently. A later GPU port is held to this crate.
//!
//! The pieces:
//! - [`Real`] (`f32` or `f64`): the whole engine is generic over it. `f64` is held
//!   to MuJoCo; `f32` is what the GPU will run, measured here and gated loosely.
//! - [`Model::compile`] turns a [`sim_scene::Scene`] into flat arrays in MuJoCo's
//!   conventions (world is body 0, parents before children, geoms by body).
//! - [`Data`] is one environment: the state (`qpos`, `qvel`, `ctrl`, `time`,
//!   `qacc_warmstart`) and every intermediate, allocated once, the contacts included.
//! - The step: [`kinematics`] (the geom frames too), [`com_pos`], [`tendon`], [`crb`],
//!   [`collide`], the constraint rows (limits, friction loss, contacts), [`com_vel`],
//!   [`passive`], [`rne`], [`actuation`], the constraint solve (all in [`forward`]), then
//!   [`euler`] or [`rk4`]; [`step`] does all of it. [`contact_force`] reads a contact's force
//!   in its frame. [`energy_pos`] and [`energy_vel`] measure the energy.
//! - [`step_world`] steps a whole [`sim_world::HostWorld`] of environments in
//!   `f32` and writes the poses and velocities the render view and the senses read.
//!
//! Invariants the whole crate keeps:
//! - **No allocation inside a step.** Everything a step touches is preallocated in
//!   [`Data`] (the contacts to the model's per-pair maxima); locals are fixed-size
//!   arrays.
//! - **Plain multiply and add only.** No fused multiply-add anywhere: the
//!   operation order is the one of the MuJoCo source, and Rust never fuses on its
//!   own. A GPU port that does not fuse either matches it mechanically. The colliders
//!   also rely on strict IEEE comparisons with infinities and NaNs: no fast math.
//! - **Quaternions are `[w, x, y, z]` inside, `[x, y, z, w]` outside.** The scene and
//!   `HostWorld` use `[x, y, z, w]`; [`convert_qpos`] is the single place that
//!   reorders them.
//! - **Deterministic.** No clock, no randomness, no unordered iteration; the same
//!   state steps to the same bits, whatever else is in the batch.
//! - No GPU, no kit dependency, no unsafe code.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod candidates;
mod collide_box;
mod collide_primitive;
mod collision;
mod constraint;
mod contact;
mod data;
mod energy;
mod factor;
mod integrate;
mod jac;
mod linalg;
mod math;
mod model;
mod real;
mod smooth;
mod solver;
mod world;

pub use collide_box::{box_box, capsule_box, sphere_box};
pub use collide_primitive::{
    GeomPose, PreContact, capsule_capsule, plane_box, plane_capsule, plane_cylinder, plane_sphere,
    sphere_capsule, sphere_cylinder, sphere_sphere,
};
pub use collision::collide;
pub use constraint::{ConstraintState, ConstraintType};
pub use contact::contact_force;
pub use data::{Data, SolverStat, Workspace};
pub use energy::{energy, energy_pos, energy_vel};
pub use integrate::{euler, integrate_pos, rk4, step};
pub use model::{
    ActuatorType, Candidate, Collider, DisableFlags, GeomType, JointType, Model, NotModelled,
    Options, PhysicsError, PrimalSolver, SameFrame, Sparsity, internal_body_to_scene,
    scene_body_to_internal,
};
pub use real::Real;
pub use smooth::{
    actuation, body_velocity, com_pos, com_vel, crb, forward, kinematics, passive, rne, tendon,
    tendon_velocity,
};
pub use world::{QuatOrder, convert_qpos, reorder_quat, step_world};

pub use sim_scene::{Cone, Integrator};

/// Test-only fault injection (the negative controls of the test suite). Not part
/// of the supported API.
#[doc(hidden)]
pub mod faults {
    pub use crate::collision::collide_in_order;
    pub use crate::smooth::Faults;

    use crate::data::Data;
    use crate::model::{Model, NotModelled, PhysicsError};
    use crate::real::Real;

    /// [`crate::forward`] with `faults` applied.
    pub fn forward_faulted<R: Real>(m: &Model<R>, d: &mut Data<R>, faults: &Faults) {
        crate::smooth::forward_with(m, d, faults);
    }

    /// [`crate::collide`] with `faults` applied (`flip_contact_normal`, `drop_last_boxbox_contact`
    /// and `includemargin_minus_gap` act in the collision step).
    pub fn collide_faulted<R: Real>(m: &Model<R>, d: &mut Data<R>, faults: &Faults) {
        crate::collision::collision(m, d, faults);
    }

    /// [`crate::step`] with `faults` applied.
    pub fn step_faulted<R: Real>(m: &Model<R>, d: &mut Data<R>, faults: &Faults) {
        crate::integrate::step_with(m, d, faults);
    }

    /// [`Model::compile`] with the compile-time faults of `faults`
    /// (`friction_mix_min`, `nested_order_everywhere`) applied.
    pub fn compile_faulted<R: Real>(
        scene: &sim_scene::Scene,
        faults: &Faults,
    ) -> Result<(Model<R>, Vec<NotModelled>), PhysicsError> {
        Model::<R>::compile_with(scene, faults)
    }
}
