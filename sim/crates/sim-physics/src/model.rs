//! The compiled model: a [`sim_scene::Scene`] turned into the flat arrays the
//! step reads, and the list of everything in the scene this phase ignores.
//!
//! The arrays follow MuJoCo's `mjModel` (3.14.0), because the step is a port and
//! a GPU port wants fixed arrays in topological order, not a graph.
//!
//! Invariants:
//! - **Bodies.** Index 0 is the world (MuJoCo's convention); `scene.bodies[b]` is
//!   internal body `b + 1`, and [`scene_body_to_internal`] and
//!   [`internal_body_to_scene`] are the only places that say so. `sim-scene`
//!   lists a parent before its children and [`Model::compile`] checks it again,
//!   so every loop over bodies `1..nbody` in order is a topological pass and
//!   every loop in reverse is a leaves-first pass.
//! - **Joints and dofs** are in the scene's joint order, which is body order, so
//!   the joints of a body are the contiguous range
//!   `body_jntadr[b] .. body_jntadr[b] + body_jntnum[b]` and its dofs
//!   `body_dofadr[b] .. + body_dofnum[b]`. A free joint has 7 `qpos` and 6 dofs, a
//!   ball joint 4 and 3, a hinge or slide 1 and 1 (`sim_scene::JointKind`).
//!   `dof_parentid` is the previous dof of the same body, else the last dof of
//!   the nearest ancestor body that has one, else -1 (MuJoCo's tree of dofs).
//! - **Quaternions** are `[w, x, y, z]` everywhere inside this crate, as in
//!   MuJoCo, including `qpos0`, `qpos_spring`, `body_quat` and `body_iquat`. The
//!   scene and `HostWorld` are `[x, y, z, w]`; [`crate::convert_qpos`] converts, and
//!   it is the only place that does (the quaternions of the scene are reordered
//!   here once, at compile time).
//! - **Everything is computed in `f64` and rounded once** to `R`: normalised
//!   quaternions and axes, `body_subtreemass`, `qpos0`, and the quantities MuJoCo's
//!   `mj_setConst` derives at `qpos0` (`dof_invweight0`, `tendon_invweight0`,
//!   `body_invweight0`, `meaninertia`). A `Model<f32>` is the `Model<f64>` rounded,
//!   not a model recomputed in single precision.
//! - `qpos0` and `qpos_spring`: a free joint takes the body's pose (position and
//!   normalised quaternion), a ball joint the identity quaternion (MuJoCo's, and
//!   `sim-world`'s), a hinge or slide 0 (`sim-scene` has no `ref` or `springref`).
//! - A body with joints must have mass and principal inertia of at least
//!   `1e-15`, or a static (jointless) descendant that does (MuJoCo's
//!   `CheckBodyMassInertia`); otherwise [`Model::compile`] refuses the scene.
//! - **Constraints** (phase 1c-i). Joint limits (`jnt_limited`, `jnt_range`,
//!   `jnt_margin`, `jnt_solref`, `jnt_solimp`), dof friction loss (`dof_frictionloss`,
//!   `dof_solref`, `dof_solimp`: a joint's `solreffriction` and `solimpfriction`, for
//!   each of its dofs) and fixed tendons (`wrap_*`, `tendon_*`) are MuJoCo's arrays.
//!   `opt` holds the solver options and `disable` MuJoCo's `disableflags` the step
//!   reads. A fixed tendon that lists a joint twice is refused: MuJoCo accepts it
//!   but converts its sparse Jacobian row to a dense one by overwriting, which
//!   drops the first coefficient.
//! - **Contacts** (phase 1c-ii). The geom arrays are MuJoCo's (`geom_type` with its integer
//!   codes, `geom_bodyid`, `geom_pos`, `geom_quat`, `geom_size`, `geom_rbound`,
//!   `geom_sameframe`, `geom_contype`, `geom_conaffinity`, `geom_condim`, `geom_priority`,
//!   `geom_solmix`, `geom_solref`, `geom_solimp`, `geom_friction`, `geom_margin`, `geom_gap`)
//!   with `body_geomadr`, `body_geomnum`, `body_has_bvh` and `exclude_signature`; scene geom `i`
//!   is MuJoCo's geom `i` (the scene lists geoms by body, the world's first, and compile refuses a
//!   scene that does not). The static candidate list ([`Model::candidates`], built here by
//!   `candidates.rs`) holds every geom pair MuJoCo's filters let through, in MuJoCo's contact
//!   order, with the slot range of its contacts, the mixed contact parameters (computed once in
//!   `f64`) and the margin; its sizes give `ncon_max` and the contact part of `nefc_max`, which
//!   size [`Data`]'s contact and row arrays. [`Model::rebuild_contact_pairs`] rebuilds it after
//!   `disable.filterparent` or `disable.midphase` changes.
//! - Nothing in the scene is ignored silently: [`Model::compile`] returns a
//!   [`NotModelled`] entry for every type pair of geoms that could collide and has no
//!   collider here (an ellipsoid or mesh pair, capsule-cylinder, cylinder-cylinder,
//!   cylinder-box), a fixed tendon's spring, damper or armature, a PGS solver, and every
//!   `Scene::unsupported` record.

use std::fmt;

use sim_scene::{ActuatorKind, Cone, Integrator, JointKind, Scene, Shape, Solver};

use crate::candidates;
use crate::data::Data;
use crate::factor;
use crate::real::Real;
use crate::smooth::{self, Faults};

/// Why a scene could not be compiled, or a batch step could not run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhysicsError {
    /// The scene failed its own validation, its body order is not topological, or
    /// it holds something MuJoCo computes wrongly (a repeated tendon joint).
    Scene {
        /// The reason.
        message: String,
    },
    /// A body with joints has no usable mass or inertia (and no static
    /// descendant that has).
    DegenerateMass {
        /// The scene body index (`scene.bodies[body]`).
        body: usize,
    },
    /// The data, the model and the world do not describe the same thing.
    Mismatch {
        /// The rule, in a few words.
        reason: &'static str,
    },
}

impl fmt::Display for PhysicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhysicsError::Scene { message } => write!(f, "scene refused: {message}"),
            PhysicsError::DegenerateMass { body } => write!(
                f,
                "mass and inertia of moving bodies must be larger than 1e-15 (scene body {body})"
            ),
            PhysicsError::Mismatch { reason } => {
                write!(f, "model, data and world disagree: {reason}")
            }
        }
    }
}

impl std::error::Error for PhysicsError {}

/// The kind of a joint, with MuJoCo's coordinate counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointType {
    /// Six degrees of freedom: 7 `qpos` (position, quaternion `[w, x, y, z]`), 6 dofs
    /// (linear velocity in the world frame, angular velocity in the body frame).
    Free,
    /// Three rotational degrees of freedom: 4 `qpos` (quaternion `[w, x, y, z]`),
    /// 3 dofs (angular velocity in the body frame).
    Ball,
    /// One rotation about `jnt_axis`.
    Hinge,
    /// One translation along `jnt_axis`.
    Slide,
}

impl JointType {
    /// Entries in `qpos`.
    pub const fn nq(self) -> usize {
        match self {
            JointType::Free => 7,
            JointType::Ball => 4,
            JointType::Hinge | JointType::Slide => 1,
        }
    }

    /// Degrees of freedom.
    pub const fn nv(self) -> usize {
        match self {
            JointType::Free => 6,
            JointType::Ball => 3,
            JointType::Hinge | JointType::Slide => 1,
        }
    }
}

/// The kind of an actuator this phase models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActuatorType {
    /// `qfrc = gear * ctrl` (MuJoCo's motor: gain 1, no bias).
    Motor,
    /// `force = kp * ctrl - kp * q` (MuJoCo's position actuator with gear 1 and kv 0).
    Position,
}

/// The primal constraint solver the step runs (MuJoCo's `opt.solver`, without the
/// dual `PGS`, which this phase refuses).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimalSolver {
    /// MuJoCo's Newton solver: Newton steps with an exact line search on the
    /// Cholesky-factored Hessian `M + J' D J`, updated incrementally.
    Newton,
    /// MuJoCo's conjugate-gradient solver (Hager-Zhang), preconditioned with `M^-1`.
    Cg,
}

/// MuJoCo's solver options (`mjOption`) that the constraint pipeline reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options<R: Real> {
    /// The solver. Default Newton.
    pub solver: PrimalSolver,
    /// The most solver iterations per solve (`opt.iterations`).
    pub iterations: usize,
    /// The solver tolerance (`opt.tolerance`): the solver stops when the scaled cost
    /// improvement or the scaled gradient norm is below it.
    pub tolerance: R,
    /// The most line-search iterations per solver iteration (`opt.ls_iterations`).
    pub ls_iterations: usize,
    /// The line-search tolerance (`opt.ls_tolerance`), relative to `tolerance`.
    pub ls_tolerance: R,
    /// The friction-to-normal impedance ratio of contacts (`opt.impratio`): the
    /// regularisation of a frictional contact's friction rows is its normal row's divided by it.
    pub impratio: R,
    /// The friction cone of contacts (`opt.cone`): pyramidal rows or an elliptic cone.
    pub cone: Cone,
}

/// MuJoCo's `disableflags` (`mjtDisableBit`) that the constraint pipeline reads. All
/// default to `false` (nothing disabled); `sim-scene` has no `<flag>`, so a test or
/// a caller sets them on the compiled [`Model`].
///
/// `filterparent` and `midphase` decide the candidate list of the collision step
/// ([`Model::candidates`]): after changing either, call
/// [`Model::rebuild_contact_pairs`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DisableFlags {
    /// `mjDSBL_CONSTRAINT`: no constraints at all (`qacc` is `qacc_smooth`), and no
    /// collision detection (MuJoCo's `mj_collision` returns at once).
    pub constraint: bool,
    /// `mjDSBL_FRICTIONLOSS`: no joint or tendon friction-loss rows.
    pub frictionloss: bool,
    /// `mjDSBL_LIMIT`: no joint or tendon limit rows.
    pub limit: bool,
    /// `mjDSBL_WARMSTART`: the solver starts from `qacc_smooth`, not from the better
    /// of it and `qacc_warmstart`.
    pub warmstart: bool,
    /// `mjDSBL_REFSAFE`: do not impose `solref[0] >= 2 * timestep`.
    pub refsafe: bool,
    /// `mjDSBL_CONTACT`: no collision detection and no contact rows.
    pub contact: bool,
    /// `mjDSBL_FILTERPARENT`: a body and its parent may collide.
    pub filterparent: bool,
    /// `mjDSBL_MIDPHASE`: the geom pairs of a body pair are tested in nested order, not
    /// in the order of MuJoCo's midphase (sorted by geom ids). The two orders differ only
    /// where bodies hold several geoms of different types.
    pub midphase: bool,
}

/// The type of a geom, with MuJoCo's integer codes (`mjtGeom`). The importer cannot
/// produce height fields (1) or signed distance fields (8).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GeomType {
    /// An infinite or finite plane through the geom's origin, normal along its z axis (0).
    Plane = 0,
    /// A sphere (2).
    Sphere = 2,
    /// A capsule along its z axis (3).
    Capsule = 3,
    /// An ellipsoid (4).
    Ellipsoid = 4,
    /// A cylinder along its z axis (5).
    Cylinder = 5,
    /// A box (6).
    Box = 6,
    /// A triangle mesh (7).
    Mesh = 7,
}

impl GeomType {
    /// MuJoCo's integer code (`mjtGeom`).
    pub const fn code(self) -> i32 {
        self as i32
    }
}

/// What the compiler found out about a geom's frame against its body's (MuJoCo's
/// `mjtSameFrame`), which [`crate::kinematics`] uses to skip the composition of the
/// geom's pose with its body's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SameFrame {
    /// The frames differ (0).
    None = 0,
    /// The geom's frame is the body's (1).
    Body = 1,
    /// The geom's frame is the body's inertial frame (2).
    Inertia = 2,
    /// The geom's orientation is the body's (3).
    BodyRot = 3,
    /// The geom's orientation is the body's inertial frame's (4).
    InertiaRot = 4,
}

impl SameFrame {
    /// MuJoCo's integer code (`mjtSameFrame`).
    pub const fn code(self) -> i32 {
        self as i32
    }
}

/// The collider that tests one geom pair (the entries of MuJoCo's `mjCOLLISIONFUNC`
/// that this crate ports): named for the geom types of the pair, the lower type first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Collider {
    /// `mjc_PlaneSphere`.
    PlaneSphere,
    /// `mjc_PlaneCapsule`.
    PlaneCapsule,
    /// `mjc_PlaneCylinder`.
    PlaneCylinder,
    /// `mjc_PlaneBox`.
    PlaneBox,
    /// `mjc_SphereSphere`.
    SphereSphere,
    /// `mjc_SphereCapsule`.
    SphereCapsule,
    /// `mjc_SphereCylinder`.
    SphereCylinder,
    /// `mjc_SphereBox`.
    SphereBox,
    /// `mjc_CapsuleCapsule`.
    CapsuleCapsule,
    /// `mjc_CapsuleBox`.
    CapsuleBox,
    /// `mjc_BoxBox`.
    BoxBox,
}

impl Collider {
    /// The most contacts the collider can report (MuJoCo's `mj_maxContact` for the pair).
    pub const fn max_contacts(self) -> usize {
        match self {
            Collider::PlaneSphere
            | Collider::SphereSphere
            | Collider::SphereCapsule
            | Collider::SphereCylinder
            | Collider::SphereBox => 1,
            Collider::PlaneCapsule | Collider::CapsuleCapsule => 2,
            Collider::PlaneCylinder | Collider::PlaneBox | Collider::CapsuleBox => 4,
            Collider::BoxBox => 8,
        }
    }
}

/// One geom pair the collision step tests, with everything about it that does not
/// depend on the state (the static candidate list, [`Model::candidates`]): the geoms
/// (oriented so that `type(g1) <= type(g2)`), the collider, the pair's range of
/// pre-contact slots, the detection margin and the parameters of the contacts it makes
/// (MuJoCo's `mj_contactParam`, mixed once at compile time).
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate<R: Real> {
    /// The first geom (the lower geom type; the contact normal points from it).
    pub g1: usize,
    /// The second geom.
    pub g2: usize,
    /// The collider.
    pub collider: Collider,
    /// The first of this pair's `slot_count` slots in the contact arrays.
    pub slot_offset: usize,
    /// The most contacts the pair can make (`mj_maxContact`).
    pub slot_count: usize,
    /// `margin + gap`: the distance below which the collider reports a contact.
    pub margin_gap: R,
    /// The sum of the two geoms' margins: a contact is a constraint only below it
    /// (`includemargin`).
    pub includemargin: R,
    /// The sum of the two geoms' gaps.
    pub gap: R,
    /// The contact dimension (1, 3, 4 or 6).
    pub condim: usize,
    /// The mixed `solref`.
    pub solref: [R; 2],
    /// `solreffriction`: zero for a pair of geoms (MuJoCo's default).
    pub solreffriction: [R; 2],
    /// The mixed `solimp`.
    pub solimp: [R; 5],
    /// The five friction coefficients `[f0, f0, f1, f2, f2]`, each at least `mjMINMU`.
    pub friction: [R; 5],
}

/// Something in the scene that this phase of the physics does not model. It is
/// returned by [`Model::compile`] so that nothing is ignored silently.
#[derive(Clone, Debug, PartialEq)]
pub enum NotModelled {
    /// A fixed tendon's spring, damper or armature (`stiffness`, `damping`,
    /// `armature` above zero): the tendon's limit and friction loss act, these do not.
    TendonPassive {
        /// The scene tendon index.
        tendon: usize,
        /// The spring stiffness.
        stiffness: f64,
        /// The damping.
        damping: f64,
        /// The armature.
        armature: f64,
    },
    /// The scene asks for MuJoCo's PGS solver, which this phase refuses: the model
    /// runs Newton instead (the same convex problem, a different algorithm).
    PgsSolver,
    /// Geom pairs of one type pair that MuJoCo's collision filter would let collide
    /// (`contype`/`conaffinity`, different welded bodies, not both static, not parent and
    /// child, not excluded) and that have no collider in this crate: every pair with an
    /// ellipsoid or a mesh, and capsule-cylinder, cylinder-cylinder and cylinder-box
    /// (MuJoCo's convex path, GJK and EPA, which is deferred). Those geom pairs produce no
    /// contacts. One entry per type pair, in order of `(type1, type2)`.
    Collision {
        /// The lower geom type of the pair.
        type1: GeomType,
        /// The higher geom type (not lower than `type1`).
        type2: GeomType,
        /// How many geom pairs of these types could collide.
        pairs: usize,
        /// The first such pair as scene geom indices, `(g1, g2)` with `type(g1) <=
        /// type(g2)`, in the order of the candidate list.
        first_pair: (usize, usize),
    },
    /// An entry of `Scene::unsupported`, copied unchanged.
    Unsupported {
        /// XML path of the element that carried it.
        path: String,
        /// The element or attribute.
        item: String,
        /// 1-based line, or 0.
        line: u32,
        /// Why it is listed.
        reason: String,
    },
}

impl fmt::Display for NotModelled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NotModelled::TendonPassive {
                tendon,
                stiffness,
                damping,
                armature,
            } => write!(
                f,
                "tendon {tendon} has stiffness {stiffness}, damping {damping}, armature {armature}: its limit and friction loss act, its spring, damper and armature are ignored"
            ),
            NotModelled::PgsSolver => write!(
                f,
                "the scene asks for the PGS solver, which is refused in this phase: Newton runs instead"
            ),
            NotModelled::Collision {
                type1,
                type2,
                pairs,
                first_pair,
            } => write!(
                f,
                "{pairs} {type1:?}-{type2:?} geom pairs could collide (first: {} and {}) and have no collider yet (MuJoCo's convex path is deferred): they make no contacts",
                first_pair.0, first_pair.1
            ),
            NotModelled::Unsupported {
                path,
                item,
                line,
                reason,
            } => write!(f, "{path} {item} (line {line}) is not modelled: {reason}"),
        }
    }
}

/// The internal body index of `scene.bodies[b]` (the world is body 0).
pub const fn scene_body_to_internal(b: usize) -> usize {
    b + 1
}

/// The index in `scene.bodies` of internal body `b`, or `None` for the world.
pub const fn internal_body_to_scene(b: usize) -> Option<usize> {
    if b == 0 { None } else { Some(b - 1) }
}

/// The sparsity of the joint-space inertia matrix (MuJoCo's `M_rownnz`, `M_rowadr`,
/// `M_colind`): row `i` holds the entries of dof `i` and its ancestors in the tree of
/// dofs, columns ascending, the diagonal last; the row of a "simple" dof (a body whose
/// inertia matrix is constant, `dof_simplenum`) holds the diagonal only. The solves and
/// products with `M` visit exactly these entries, in this order, as MuJoCo does.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Sparsity {
    /// Per dof: the number of entries of its row (the diagonal included).
    pub rownnz: Vec<usize>,
    /// Per dof: where its row starts in `colind`.
    pub rowadr: Vec<usize>,
    /// The column of each entry.
    pub colind: Vec<usize>,
}

/// A scene compiled for the step: MuJoCo's `mjModel`, restricted to what the
/// dynamics and the constraints read. See the module note for the conventions.
#[derive(Clone, Debug)]
pub struct Model<R: Real> {
    /// Bodies including the world (body 0).
    pub nbody: usize,
    /// Joints.
    pub njnt: usize,
    /// Length of `qpos`.
    pub nq: usize,
    /// Length of `qvel` (degrees of freedom).
    pub nv: usize,
    /// Actuators (length of `ctrl`).
    pub nu: usize,
    /// Fixed tendons.
    pub ntendon: usize,
    /// The most constraint rows a step can instantiate (every dof and tendon with
    /// friction loss, both limits of every limited hinge, slide and tendon, the one
    /// limit of a limited ball joint, and every contact slot of the candidate list with
    /// its rows): the length of [`Data`]'s `efc_*` arrays.
    pub nefc_max: usize,
    /// Geoms (MuJoCo's `ngeom`); scene geom `i` is geom `i`.
    pub ngeom: usize,
    /// The most contacts a step can hold: the sum of the slot counts of
    /// [`Model::candidates`] (the length of [`Data`]'s contact arrays).
    pub ncon_max: usize,
    /// The timestep `h`, seconds.
    pub timestep: R,
    /// Gravity, world frame.
    pub gravity: [R; 3],
    /// The integrator `step` uses.
    pub integrator: Integrator,
    /// The solver options.
    pub opt: Options<R>,
    /// MuJoCo's `disableflags`.
    pub disable: DisableFlags,

    /// Per body: the parent (0 for the world and for a child of the world).
    pub body_parentid: Vec<usize>,
    /// Per body: the child of the world at the root of its tree (itself for such a child).
    pub body_rootid: Vec<usize>,
    /// Per body: the body it is rigidly welded to that has degrees of freedom, or
    /// 0 when it and all its ancestors have none (MuJoCo's `body_weldid`). MuJoCo makes a
    /// mocap body its own weld root; this array does not, which cannot matter: the importer
    /// refuses mocap bodies.
    pub body_weldid: Vec<usize>,
    /// Per body: the first joint.
    pub body_jntadr: Vec<usize>,
    /// Per body: the number of joints.
    pub body_jntnum: Vec<usize>,
    /// Per body: the first dof.
    pub body_dofadr: Vec<usize>,
    /// Per body: the number of dofs.
    pub body_dofnum: Vec<usize>,
    /// `3 * nbody`: position in the parent frame.
    pub body_pos: Vec<R>,
    /// `4 * nbody`: orientation in the parent frame, `[w, x, y, z]`, unit.
    pub body_quat: Vec<R>,
    /// Per body: mass.
    pub body_mass: Vec<R>,
    /// Per body: the mass of the body and all its descendants.
    pub body_subtreemass: Vec<R>,
    /// `3 * nbody`: the centre of mass in the body frame.
    pub body_ipos: Vec<R>,
    /// `4 * nbody`: the orientation of the principal axes in the body frame, `[w, x, y, z]`.
    pub body_iquat: Vec<R>,
    /// Per body: how its inertial frame relates to its body frame (MuJoCo's `body_sameframe`,
    /// `user_model.cc`): [`SameFrame::Body`] when the inertial pose is within `kFrameEps = 1e-6`
    /// of the body frame in position and orientation, [`SameFrame::BodyRot`] when only the
    /// orientation is, else [`SameFrame::None`]. [`crate::kinematics`] copies the body's pose
    /// (`mj_local2Global`) in the first two cases, so a near-null inertial frame is snapped
    /// onto the body frame exactly as MuJoCo does it.
    pub body_sameframe: Vec<SameFrame>,
    /// `3 * nbody`: the principal moments of inertia.
    pub body_inertia: Vec<R>,
    /// `2 * nbody`: the approximate inverse inertia of each body at `qpos0`, translation
    /// then rotation (MuJoCo's `body_invweight0`).
    pub body_invweight0: Vec<R>,
    /// Per body: the first of its geoms, or -1 for a body with none (MuJoCo's
    /// `body_geomadr`). The geoms of a body are contiguous, bodies in order, the world's
    /// first.
    pub body_geomadr: Vec<i32>,
    /// Per body: the number of its geoms (`body_geomnum`).
    pub body_geomnum: Vec<usize>,
    /// Per body: whether MuJoCo gives it a bounding-volume hierarchy
    /// (`body_bvhadr >= 0`), which decides whether its geom pairs go through the
    /// midphase. Inferred as "has at least one geom" and held to the oracle.
    pub body_has_bvh: Vec<bool>,
    /// The body pairs that never collide (MuJoCo's `exclude_signature`):
    /// `(min body << 16) + max body`, sorted ascending.
    pub exclude_signature: Vec<u32>,

    /// Per geom: its type.
    pub geom_type: Vec<GeomType>,
    /// Per geom: its body (0 for the world).
    pub geom_bodyid: Vec<usize>,
    /// `3 * ngeom`: position in the body frame.
    pub geom_pos: Vec<R>,
    /// `4 * ngeom`: orientation in the body frame, `[w, x, y, z]`, unit.
    pub geom_quat: Vec<R>,
    /// `3 * ngeom`: MuJoCo's size vector of the type: sphere `[r, 0, 0]`, capsule and
    /// cylinder `[r, half_length, 0]`, box and ellipsoid the half-sizes, plane
    /// `[half_x, half_y, spacing]`.
    pub geom_size: Vec<R>,
    /// Per geom: the radius of its bounding sphere about its origin (MuJoCo's
    /// `geom_rbound`; 0 for a plane).
    pub geom_rbound: Vec<R>,
    /// Per geom: whether its frame is its body's or its inertial frame (MuJoCo's
    /// `geom_sameframe`).
    pub geom_sameframe: Vec<SameFrame>,
    /// Per geom: the collision type bitmask (`geom_contype`).
    pub geom_contype: Vec<u32>,
    /// Per geom: the collision affinity bitmask (`geom_conaffinity`).
    pub geom_conaffinity: Vec<u32>,
    /// Per geom: the contact dimension (1, 3, 4 or 6).
    pub geom_condim: Vec<u32>,
    /// Per geom: the contact priority.
    pub geom_priority: Vec<i32>,
    /// Per geom: the weight in the mix of contact parameters.
    pub geom_solmix: Vec<R>,
    /// `2 * ngeom`: `solref` of the geom's contacts.
    pub geom_solref: Vec<R>,
    /// `5 * ngeom`: `solimp` of the geom's contacts.
    pub geom_solimp: Vec<R>,
    /// `3 * ngeom`: sliding, torsional and rolling friction.
    pub geom_friction: Vec<R>,
    /// Per geom: the margin.
    pub geom_margin: Vec<R>,
    /// Per geom: the gap.
    pub geom_gap: Vec<R>,
    /// The static candidate list of the collision step (see [`Candidate`] and the module
    /// note of `collision.rs`): every geom pair that MuJoCo's filters let through, in
    /// MuJoCo's contact order, built by [`Model::compile`] and
    /// [`Model::rebuild_contact_pairs`].
    pub candidates: Vec<Candidate<R>>,
    /// The `disable.filterparent` the candidate list was built with.
    pub candidates_built_filterparent: bool,
    /// The `disable.midphase` the candidate list was built with.
    pub candidates_built_midphase: bool,
    /// A hash of the candidate list's geoms, colliders and slot ranges: a
    /// [`Data`] built for another list is not [`Data::fits`] this model.
    pub candidates_fingerprint: u64,

    /// Per joint: its kind.
    pub jnt_type: Vec<JointType>,
    /// Per joint: where its coordinates start in `qpos`.
    pub jnt_qposadr: Vec<usize>,
    /// Per joint: where its dofs start in `qvel`.
    pub jnt_dofadr: Vec<usize>,
    /// Per joint: the body it moves.
    pub jnt_bodyid: Vec<usize>,
    /// `3 * njnt`: the anchor in the body frame.
    pub jnt_pos: Vec<R>,
    /// `3 * njnt`: the unit axis in the body frame (`(0, 0, 1)` for free and ball joints).
    pub jnt_axis: Vec<R>,
    /// Per joint: the spring stiffness.
    pub jnt_stiffness: Vec<R>,
    /// Per joint: whether it has a limit (never a free joint).
    pub jnt_limited: Vec<bool>,
    /// `2 * njnt`: the limit, radians for a hinge or ball joint (a ball joint's is
    /// `[0, max_angle]`), metres for a slide; `[0, 0]` when not limited.
    pub jnt_range: Vec<R>,
    /// Per joint: the distance from the limit below which the limit row is active.
    pub jnt_margin: Vec<R>,
    /// `2 * njnt`: `solref` of the limit.
    pub jnt_solref: Vec<R>,
    /// `5 * njnt`: `solimp` of the limit.
    pub jnt_solimp: Vec<R>,
    /// `nq`: the reference coordinates.
    pub qpos0: Vec<R>,
    /// `nq`: the spring rest coordinates (equal to `qpos0` in this phase).
    pub qpos_spring: Vec<R>,

    /// Per dof: the body it belongs to.
    pub dof_bodyid: Vec<usize>,
    /// Per dof: its joint.
    pub dof_jntid: Vec<usize>,
    /// Per dof: the parent dof in the tree of dofs, or -1.
    pub dof_parentid: Vec<i32>,
    /// Per dof: the rotor inertia added to the diagonal of `M`.
    pub dof_armature: Vec<R>,
    /// Per dof: the viscous damping.
    pub dof_damping: Vec<R>,
    /// Per dof: the dry friction magnitude (the joint's `frictionloss`).
    pub dof_frictionloss: Vec<R>,
    /// `2 * nv`: `solref` of the friction loss (the joint's `solreffriction`).
    pub dof_solref: Vec<R>,
    /// `5 * nv`: `solimp` of the friction loss (the joint's `solimpfriction`).
    pub dof_solimp: Vec<R>,
    /// Per dof: the approximate inverse inertia at `qpos0` (MuJoCo's `dof_invweight0`).
    pub dof_invweight0: Vec<R>,
    /// Per dof: the number of consecutive dofs from it on that belong to "simple"
    /// bodies (MuJoCo's `dof_simplenum`); non-zero means its inertia row is the
    /// diagonal only.
    pub dof_simplenum: Vec<usize>,
    /// Per dof: the diagonal inertia at `qpos0` with the armature (MuJoCo's `dof_M0`,
    /// `mj_setM0`), which `crb` writes as the whole row of a simple dof.
    pub dof_m0: Vec<R>,
    /// The sparsity of `M` (MuJoCo's `M_rownnz`, `M_rowadr`, `M_colind`).
    pub qm_sparsity: Sparsity,
    /// Whether any dof has damping above zero (the Euler step then solves with
    /// `M + h diag(damping)`).
    pub has_damping: bool,
    /// The mean of the diagonal of `M` at `qpos0` (MuJoCo's `stat.meaninertia`); the
    /// solvers scale their tolerances by `1 / (meaninertia * max(1, nv))`.
    pub meaninertia: R,

    /// Per tendon: its first entry in `wrap_objid` and `wrap_prm`.
    pub tendon_adr: Vec<usize>,
    /// Per tendon: the number of joints it couples.
    pub tendon_num: Vec<usize>,
    /// Per wrap entry: the joint (a hinge or slide).
    pub wrap_objid: Vec<usize>,
    /// Per wrap entry: the coefficient: the tendon length is `sum coef * qpos[adr]`.
    pub wrap_prm: Vec<R>,
    /// Per tendon: the sorted dofs of its Jacobian row (MuJoCo's `ten_J_colind`).
    pub ten_j_colind: Vec<usize>,
    /// Per tendon: the start of its row in `ten_j_colind`.
    pub ten_j_rowadr: Vec<usize>,
    /// Per tendon: the number of entries of its row in `ten_j_colind`.
    pub ten_j_rownnz: Vec<usize>,
    /// Per tendon: whether it has a length limit.
    pub tendon_limited: Vec<bool>,
    /// `2 * ntendon`: the length limit; `[0, 0]` when not limited.
    pub tendon_range: Vec<R>,
    /// Per tendon: the distance from the limit below which the limit row is active.
    pub tendon_margin: Vec<R>,
    /// Per tendon: the dry friction magnitude.
    pub tendon_frictionloss: Vec<R>,
    /// `2 * ntendon`: `solref` of the length limit.
    pub tendon_solref_lim: Vec<R>,
    /// `5 * ntendon`: `solimp` of the length limit.
    pub tendon_solimp_lim: Vec<R>,
    /// `2 * ntendon`: `solref` of the friction loss.
    pub tendon_solref_fri: Vec<R>,
    /// `5 * ntendon`: `solimp` of the friction loss.
    pub tendon_solimp_fri: Vec<R>,
    /// Per tendon: the approximate inverse inertia at `qpos0` (MuJoCo's
    /// `tendon_invweight0`).
    pub tendon_invweight0: Vec<R>,

    /// Per actuator: motor or position servo.
    pub actuator_type: Vec<ActuatorType>,
    /// Per actuator: the joint it drives (a hinge or slide).
    pub actuator_jnt: Vec<usize>,
    /// Per actuator: the `qpos` index of that joint.
    pub actuator_qposadr: Vec<usize>,
    /// Per actuator: the dof index of that joint.
    pub actuator_dofadr: Vec<usize>,
    /// Per actuator: the gear (motor), 1 for a position servo.
    pub actuator_gear: Vec<R>,
    /// Per actuator: the proportional gain (position servo), 0 for a motor.
    pub actuator_kp: Vec<R>,
    /// Per actuator: whether `ctrl` is clamped to its range (a range was given).
    pub actuator_ctrllimited: Vec<bool>,
    /// `2 * nu`: the control range (zeros when unlimited).
    pub actuator_ctrlrange: Vec<R>,
}

/// How far from unit length a quaternion of the scene may be and be used as it is.
/// MuJoCo's compiled inertial-frame quaternion (the eigenvectors of the inertia of a
/// body with several geoms) is not exactly unit (its norm is within 2e-15 of 1), and the
/// importer reproduces it bit for bit; a quaternion that far from unit must not be
/// "corrected", which would move it from MuJoCo's. A scene that is further off (the
/// scene validates to 1e-6) is normalised, as MuJoCo's compiler does.
const QUAT_AS_IS_TOLERANCE: f64 = 1e-9;

/// `[x, y, z, w]` to `[w, x, y, z]`, normalised in `f64` unless it is already unit to
/// [`QUAT_AS_IS_TOLERANCE`] (MuJoCo normalises at compile time).
fn quat_wxyz_normalised(q: [f64; 4]) -> [f64; 4] {
    let mut w = [q[3], q[0], q[1], q[2]];
    let norm = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2] + w[3] * w[3]).sqrt();
    if norm < 1e-15 {
        w = [1.0, 0.0, 0.0, 0.0];
    } else if (norm - 1.0).abs() > QUAT_AS_IS_TOLERANCE {
        let inv = 1.0 / norm;
        for c in &mut w {
            *c *= inv;
        }
    }
    w
}

/// A joint axis as MuJoCo's model holds it: unit length. An axis that is already unit
/// (within `mjMINVAL`, as `mju_normalize4` leaves a unit quaternion) is kept as it
/// is: the importer has normalised it the way MuJoCo's compiler does, and normalising
/// it again would move some components by one rounding step.
fn axis_normalised(a: [f64; 3]) -> [f64; 3] {
    let norm = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if norm < 1e-15 {
        [1.0, 0.0, 0.0]
    } else if (norm - 1.0).abs() > 1e-15 {
        let inv = 1.0 / norm;
        [a[0] * inv, a[1] * inv, a[2] * inv]
    } else {
        a
    }
}

/// `src` rounded to `S`.
fn conv<R: Real, S: Real>(v: &[R]) -> Vec<S> {
    v.iter().map(|&x| S::from_f64(x.to_f64())).collect()
}

/// Port of `mj_setM0` (`engine_setconst.c`): `dof_M0`, the armature plus the diagonal of the
/// composite rigid body inertia matrix at `qpos0`, from the `cinert` and `cdof` of `d` (which
/// [`crate::kinematics`] and [`crate::com_pos`] have filled for `qpos0`). `crb` writes it as the whole row of
/// a simple dof, as MuJoCo's `mj_crb` does.
fn set_m0(m: &Model<f64>, d: &Data<f64>) -> Vec<f64> {
    // copy cinert into crb, backward pass over bodies, accumulate composite inertias
    let mut crb = d.cinert.clone();
    for i in (1..m.nbody).rev() {
        let p = m.body_parentid[i];
        if p > 0 {
            for k in 0..10 {
                let s = crb[10 * p + k] + crb[10 * i + k];
                crb[10 * p + k] = s;
            }
        }
    }
    (0..m.nv)
        .map(|i| {
            // buf = crb_body_i * cdof_i; dof_M0(i) = armature + cdof_i * buf
            let buf = crate::math::mul_inert_vec(
                &crate::math::v10(&crb, m.dof_bodyid[i]),
                crate::math::v6(&d.cdof, i),
            );
            m.dof_armature[i] + crate::math::dot6(crate::math::v6(&d.cdof, i), buf)
        })
        .collect()
}

impl<R: Real> Model<R> {
    /// Compiles `scene`: validates it, checks the body order, builds the arrays
    /// (see the module note) and lists everything this phase ignores.
    ///
    /// The returned [`NotModelled`] list is empty only when the scene has no pair
    /// of geoms that could collide, no fixed tendon with a spring, damper or
    /// armature, no PGS solver and no `unsupported` record.
    pub fn compile(scene: &Scene) -> Result<(Model<R>, Vec<NotModelled>), PhysicsError> {
        Self::compile_with(scene, &Faults::NONE)
    }

    /// [`Model::compile`] with the compile-time test faults of `faults`
    /// ([`crate::faults::compile_faulted`]).
    pub(crate) fn compile_with(
        scene: &Scene,
        faults: &Faults,
    ) -> Result<(Model<R>, Vec<NotModelled>), PhysicsError> {
        let (m, list) = Model::<f64>::build(scene, faults)?;
        Ok((m.rounded_to::<R>(), list))
    }

    /// Rebuilds the static candidate list and the contact maxima (`ncon_max`, `nefc_max`)
    /// after a caller changed `disable.filterparent` or `disable.midphase`. The list
    /// depends on both; the collision step `debug_assert`s that they still match, and
    /// [`Data::fits`] rejects a `Data` sized for the old list.
    ///
    /// The mixed contact parameters are recomputed in `f64` from this model's geom arrays
    /// and rounded to `R`; on a `Model<f32>` that is the rounded geoms' mixture, not the
    /// rounding of the `f64` mixture. To keep "an `f32` model is the `f64` model rounded
    /// once", change the flags on the `f64` model, rebuild, then [`Model::rounded_to`].
    pub fn rebuild_contact_pairs(&mut self) {
        let (list, _) = candidates::build_candidates(self, &Faults::NONE);
        self.set_candidates(list);
    }

    /// Installs a candidate list: the flags it was built with, the fingerprint, and the
    /// contact maxima it implies.
    fn set_candidates(&mut self, list: Vec<Candidate<R>>) {
        self.ncon_max = list.iter().map(|c| c.slot_count).sum();
        // every slot may become a contact of its dimension: 1 row for a frictionless
        // contact, `2 (dim - 1)` for a pyramidal one (never fewer than the `dim` rows of an
        // elliptic one, so changing `opt.cone` on a compiled model stays within capacity)
        let contact_rows: usize = list
            .iter()
            .map(|c| c.slot_count * if c.condim == 1 { 1 } else { 2 * (c.condim - 1) })
            .sum();
        self.nefc_max = self.noncontact_rows_max() + contact_rows;
        self.candidates_fingerprint = candidates::fingerprint(&list);
        self.candidates_built_filterparent = self.disable.filterparent;
        self.candidates_built_midphase = self.disable.midphase;
        self.candidates = list;
    }

    /// The most friction-loss and limit rows a step can instantiate (every dof and tendon
    /// with friction loss, both limits of every limited hinge, slide and tendon, the one
    /// limit of a limited ball joint).
    fn noncontact_rows_max(&self) -> usize {
        let mut n = self
            .dof_frictionloss
            .iter()
            .filter(|&&f| f != R::ZERO)
            .count();
        n += self
            .tendon_frictionloss
            .iter()
            .filter(|&&f| f > R::ZERO)
            .count();
        for (j, &limited) in self.jnt_limited.iter().enumerate() {
            if limited {
                n += if self.jnt_type[j] == JointType::Ball {
                    1
                } else {
                    2
                };
            }
        }
        n + 2 * self.tendon_limited.iter().filter(|&&l| l).count()
    }

    /// This model with every number rounded to `S` (the `f64` model rounded once).
    pub fn rounded_to<S: Real>(&self) -> Model<S> {
        let c = |v: &Vec<R>| -> Vec<S> { conv(v) };
        let s = |x: R| S::from_f64(x.to_f64());
        Model {
            nbody: self.nbody,
            njnt: self.njnt,
            nq: self.nq,
            nv: self.nv,
            nu: self.nu,
            ntendon: self.ntendon,
            nefc_max: self.nefc_max,
            ngeom: self.ngeom,
            ncon_max: self.ncon_max,
            timestep: s(self.timestep),
            gravity: [s(self.gravity[0]), s(self.gravity[1]), s(self.gravity[2])],
            integrator: self.integrator,
            opt: Options {
                solver: self.opt.solver,
                iterations: self.opt.iterations,
                tolerance: s(self.opt.tolerance),
                ls_iterations: self.opt.ls_iterations,
                ls_tolerance: s(self.opt.ls_tolerance),
                impratio: s(self.opt.impratio),
                cone: self.opt.cone,
            },
            disable: self.disable,
            body_parentid: self.body_parentid.clone(),
            body_rootid: self.body_rootid.clone(),
            body_weldid: self.body_weldid.clone(),
            body_jntadr: self.body_jntadr.clone(),
            body_jntnum: self.body_jntnum.clone(),
            body_dofadr: self.body_dofadr.clone(),
            body_dofnum: self.body_dofnum.clone(),
            body_pos: c(&self.body_pos),
            body_quat: c(&self.body_quat),
            body_mass: c(&self.body_mass),
            body_subtreemass: c(&self.body_subtreemass),
            body_ipos: c(&self.body_ipos),
            body_iquat: c(&self.body_iquat),
            body_sameframe: self.body_sameframe.clone(),
            body_inertia: c(&self.body_inertia),
            body_invweight0: c(&self.body_invweight0),
            body_geomadr: self.body_geomadr.clone(),
            body_geomnum: self.body_geomnum.clone(),
            body_has_bvh: self.body_has_bvh.clone(),
            exclude_signature: self.exclude_signature.clone(),
            geom_type: self.geom_type.clone(),
            geom_bodyid: self.geom_bodyid.clone(),
            geom_pos: c(&self.geom_pos),
            geom_quat: c(&self.geom_quat),
            geom_size: c(&self.geom_size),
            geom_rbound: c(&self.geom_rbound),
            geom_sameframe: self.geom_sameframe.clone(),
            geom_contype: self.geom_contype.clone(),
            geom_conaffinity: self.geom_conaffinity.clone(),
            geom_condim: self.geom_condim.clone(),
            geom_priority: self.geom_priority.clone(),
            geom_solmix: c(&self.geom_solmix),
            geom_solref: c(&self.geom_solref),
            geom_solimp: c(&self.geom_solimp),
            geom_friction: c(&self.geom_friction),
            geom_margin: c(&self.geom_margin),
            geom_gap: c(&self.geom_gap),
            candidates: self
                .candidates
                .iter()
                .map(|k| Candidate {
                    g1: k.g1,
                    g2: k.g2,
                    collider: k.collider,
                    slot_offset: k.slot_offset,
                    slot_count: k.slot_count,
                    margin_gap: s(k.margin_gap),
                    includemargin: s(k.includemargin),
                    gap: s(k.gap),
                    condim: k.condim,
                    solref: k.solref.map(s),
                    solreffriction: k.solreffriction.map(s),
                    solimp: k.solimp.map(s),
                    friction: k.friction.map(s),
                })
                .collect(),
            candidates_built_filterparent: self.candidates_built_filterparent,
            candidates_built_midphase: self.candidates_built_midphase,
            candidates_fingerprint: self.candidates_fingerprint,
            jnt_type: self.jnt_type.clone(),
            jnt_qposadr: self.jnt_qposadr.clone(),
            jnt_dofadr: self.jnt_dofadr.clone(),
            jnt_bodyid: self.jnt_bodyid.clone(),
            jnt_pos: c(&self.jnt_pos),
            jnt_axis: c(&self.jnt_axis),
            jnt_stiffness: c(&self.jnt_stiffness),
            jnt_limited: self.jnt_limited.clone(),
            jnt_range: c(&self.jnt_range),
            jnt_margin: c(&self.jnt_margin),
            jnt_solref: c(&self.jnt_solref),
            jnt_solimp: c(&self.jnt_solimp),
            qpos0: c(&self.qpos0),
            qpos_spring: c(&self.qpos_spring),
            dof_bodyid: self.dof_bodyid.clone(),
            dof_jntid: self.dof_jntid.clone(),
            dof_parentid: self.dof_parentid.clone(),
            dof_armature: c(&self.dof_armature),
            dof_damping: c(&self.dof_damping),
            dof_frictionloss: c(&self.dof_frictionloss),
            dof_solref: c(&self.dof_solref),
            dof_solimp: c(&self.dof_solimp),
            dof_invweight0: c(&self.dof_invweight0),
            dof_simplenum: self.dof_simplenum.clone(),
            dof_m0: c(&self.dof_m0),
            qm_sparsity: self.qm_sparsity.clone(),
            has_damping: self.has_damping,
            meaninertia: s(self.meaninertia),
            tendon_adr: self.tendon_adr.clone(),
            tendon_num: self.tendon_num.clone(),
            wrap_objid: self.wrap_objid.clone(),
            wrap_prm: c(&self.wrap_prm),
            ten_j_colind: self.ten_j_colind.clone(),
            ten_j_rowadr: self.ten_j_rowadr.clone(),
            ten_j_rownnz: self.ten_j_rownnz.clone(),
            tendon_limited: self.tendon_limited.clone(),
            tendon_range: c(&self.tendon_range),
            tendon_margin: c(&self.tendon_margin),
            tendon_frictionloss: c(&self.tendon_frictionloss),
            tendon_solref_lim: c(&self.tendon_solref_lim),
            tendon_solimp_lim: c(&self.tendon_solimp_lim),
            tendon_solref_fri: c(&self.tendon_solref_fri),
            tendon_solimp_fri: c(&self.tendon_solimp_fri),
            tendon_invweight0: c(&self.tendon_invweight0),
            actuator_type: self.actuator_type.clone(),
            actuator_jnt: self.actuator_jnt.clone(),
            actuator_qposadr: self.actuator_qposadr.clone(),
            actuator_dofadr: self.actuator_dofadr.clone(),
            actuator_gear: c(&self.actuator_gear),
            actuator_kp: c(&self.actuator_kp),
            actuator_ctrllimited: self.actuator_ctrllimited.clone(),
            actuator_ctrlrange: c(&self.actuator_ctrlrange),
        }
    }
}

impl Model<f64> {
    /// The compile, in `f64` (every other precision is this model rounded).
    fn build(
        scene: &Scene,
        faults: &Faults,
    ) -> Result<(Model<f64>, Vec<NotModelled>), PhysicsError> {
        scene.validate().map_err(|e| PhysicsError::Scene {
            message: e.to_string(),
        })?;
        for (i, b) in scene.bodies.iter().enumerate() {
            if let Some(p) = b.parent
                && p.index() >= i
            {
                return Err(PhysicsError::Scene {
                    message: format!("body {i} is listed before its parent {}", p.index()),
                });
            }
        }
        for (t, tendon) in scene.tendons.iter().enumerate() {
            for (k, a) in tendon.joints.iter().enumerate() {
                if tendon.joints[..k].iter().any(|b| b.joint == a.joint) {
                    return Err(PhysicsError::Scene {
                        message: format!(
                            "tendon {t} lists joint {} twice: MuJoCo's dense conversion of its sparse Jacobian row drops the first coefficient, so the scene is refused",
                            a.joint.index()
                        ),
                    });
                }
            }
        }

        let nbody = scene.bodies.len() + 1;
        let njnt = scene.joints.len();
        let nq = scene.nq();
        let nv = scene.nv();
        let nu = scene.actuators.len();
        let ntendon = scene.tendons.len();

        // ---- bodies (all in f64)
        let mut body_parentid = vec![0usize; nbody];
        let mut body_rootid = vec![0usize; nbody];
        let mut body_pos = vec![0.0f64; 3 * nbody];
        let mut body_quat = vec![0.0f64; 4 * nbody];
        let mut body_mass = vec![0.0f64; nbody];
        let mut body_ipos = vec![0.0f64; 3 * nbody];
        let mut body_iquat = vec![0.0f64; 4 * nbody];
        let mut body_inertia = vec![0.0f64; 3 * nbody];
        body_quat[0] = 1.0;
        body_iquat[0] = 1.0;
        for (b, body) in scene.bodies.iter().enumerate() {
            let i = scene_body_to_internal(b);
            let parent = body.parent.map_or(0, |p| scene_body_to_internal(p.index()));
            body_parentid[i] = parent;
            body_rootid[i] = if parent == 0 { i } else { body_rootid[parent] };
            body_pos[3 * i..3 * i + 3].copy_from_slice(&body.pos);
            body_quat[4 * i..4 * i + 4].copy_from_slice(&quat_wxyz_normalised(body.quat));
            body_iquat[4 * i] = 1.0;
            if let Some(inertial) = &body.inertial {
                body_mass[i] = inertial.mass_kg;
                body_ipos[3 * i..3 * i + 3].copy_from_slice(&inertial.com);
                body_inertia[3 * i..3 * i + 3].copy_from_slice(&inertial.diag_inertia);
                body_iquat[4 * i..4 * i + 4]
                    .copy_from_slice(&quat_wxyz_normalised(inertial.inertia_quat));
            }
        }
        // the compiler's rule for body_sameframe (user_model.cc), from the compiled frames (the
        // world body's frame is null)
        let body_sameframe: Vec<SameFrame> = (0..nbody)
            .map(|i| body_same_frame(&body_ipos[3 * i..3 * i + 3], &body_iquat[4 * i..4 * i + 4]))
            .collect();
        // mass of the subtree, accumulated leaves first (engine_setconst.c setFixed)
        let mut body_subtreemass = body_mass.clone();
        for i in (1..nbody).rev() {
            let p = body_parentid[i];
            let m = body_subtreemass[i];
            body_subtreemass[p] += m;
        }

        // ---- joints and dofs
        let mut jnt_type = Vec::with_capacity(njnt);
        let mut jnt_qposadr = Vec::with_capacity(njnt);
        let mut jnt_dofadr = Vec::with_capacity(njnt);
        let mut jnt_bodyid = Vec::with_capacity(njnt);
        let mut jnt_pos = Vec::with_capacity(3 * njnt);
        let mut jnt_axis = Vec::with_capacity(3 * njnt);
        let mut jnt_stiffness = Vec::with_capacity(njnt);
        let mut jnt_limited = Vec::with_capacity(njnt);
        let mut jnt_range = Vec::with_capacity(2 * njnt);
        let mut jnt_margin = Vec::with_capacity(njnt);
        let mut jnt_solref = Vec::with_capacity(2 * njnt);
        let mut jnt_solimp = Vec::with_capacity(5 * njnt);
        let mut qpos0 = vec![0.0f64; nq];
        let mut dof_bodyid = vec![0usize; nv];
        let mut dof_jntid = vec![0usize; nv];
        let mut dof_parentid = vec![-1i32; nv];
        let mut dof_armature = vec![0.0f64; nv];
        let mut dof_damping = vec![0.0f64; nv];
        let mut dof_frictionloss = vec![0.0f64; nv];
        let mut dof_solref = vec![0.0f64; 2 * nv];
        let mut dof_solimp = vec![0.0f64; 5 * nv];
        let mut body_jntadr = vec![0usize; nbody];
        let mut body_jntnum = vec![0usize; nbody];
        let mut body_dofadr = vec![0usize; nbody];
        let mut body_dofnum = vec![0usize; nbody];
        // the last dof of each body's chain, -1 for none (the world has none)
        let mut body_lastdof = vec![-1i32; nbody];

        // first pass: which joints belong to which body (contiguous, validated)
        {
            let (mut jadr, mut dadr) = (0usize, 0usize);
            let mut next = 0usize;
            for b in 1..nbody {
                body_jntadr[b] = jadr;
                body_dofadr[b] = dadr;
                while next < njnt && scene_body_to_internal(scene.joints[next].body.index()) == b {
                    jadr += 1;
                    dadr += scene.joints[next].kind.nv();
                    next += 1;
                }
                body_jntnum[b] = jadr - body_jntadr[b];
                body_dofnum[b] = dadr - body_dofadr[b];
            }
        }

        let (mut qadr, mut dadr) = (0usize, 0usize);
        for (j, joint) in scene.joints.iter().enumerate() {
            let b = scene_body_to_internal(joint.body.index());
            let kind = joint.kind;
            let jtype = match kind {
                JointKind::Free => JointType::Free,
                JointKind::Ball => JointType::Ball,
                JointKind::Hinge { .. } => JointType::Hinge,
                JointKind::Slide { .. } => JointType::Slide,
            };
            jnt_type.push(jtype);
            jnt_qposadr.push(qadr);
            jnt_dofadr.push(dadr);
            jnt_bodyid.push(b);
            jnt_pos.extend_from_slice(&joint.pos);
            let axis = match kind {
                JointKind::Hinge { axis } | JointKind::Slide { axis } => axis_normalised(axis),
                JointKind::Free | JointKind::Ball => [0.0, 0.0, 1.0],
            };
            jnt_axis.extend_from_slice(&axis);
            jnt_stiffness.push(joint.stiffness);
            jnt_limited.push(joint.range.is_some());
            jnt_range.extend_from_slice(&joint.range.unwrap_or([0.0, 0.0]));
            jnt_margin.push(joint.margin);
            jnt_solref.extend_from_slice(&joint.solref_limit);
            jnt_solimp.extend_from_slice(&joint.solimp_limit);
            match kind {
                JointKind::Free => {
                    qpos0[qadr..qadr + 3].copy_from_slice(&body_pos[3 * b..3 * b + 3]);
                    qpos0[qadr + 3..qadr + 7].copy_from_slice(&body_quat[4 * b..4 * b + 4]);
                }
                JointKind::Ball => qpos0[qadr..qadr + 4].copy_from_slice(&[1.0, 0.0, 0.0, 0.0]),
                JointKind::Hinge { .. } | JointKind::Slide { .. } => qpos0[qadr] = 0.0,
            }
            for k in 0..kind.nv() {
                let dof = dadr + k;
                dof_bodyid[dof] = b;
                dof_jntid[dof] = j;
                dof_armature[dof] = joint.armature;
                dof_damping[dof] = joint.damping;
                dof_frictionloss[dof] = joint.frictionloss;
                dof_solref[2 * dof..2 * dof + 2].copy_from_slice(&joint.solref_friction);
                dof_solimp[5 * dof..5 * dof + 5].copy_from_slice(&joint.solimp_friction);
                // the first dof of a body hangs from the last dof of the nearest
                // ancestor chain; later dofs hang from the previous one
                let parent_dof = if body_lastdof[b] >= 0 {
                    body_lastdof[b]
                } else {
                    let mut p = body_parentid[b];
                    while p != 0 && body_lastdof[p] < 0 {
                        p = body_parentid[p];
                    }
                    if p == 0 { -1 } else { body_lastdof[p] }
                };
                dof_parentid[dof] = parent_dof;
                body_lastdof[b] = dof as i32;
            }
            qadr += kind.nq();
            dadr += kind.nv();
        }
        debug_assert_eq!((qadr, dadr), (nq, nv));

        // ---- the weld tree: a body is welded to its parent's weld body unless it has joints
        let mut body_weldid = vec![0usize; nbody];
        for b in 1..nbody {
            body_weldid[b] = if body_jntnum[b] > 0 {
                b
            } else {
                body_weldid[body_parentid[b]]
            };
        }

        // ---- the mass of every moving body (MuJoCo's CheckBodyMassInertia)
        fn valid(
            b: usize,
            mass: &[f64],
            inertia: &[f64],
            parent: &[usize],
            jntnum: &[usize],
        ) -> bool {
            let ok = mass[b] >= 1e-15
                && inertia[3 * b] >= 1e-15
                && inertia[3 * b + 1] >= 1e-15
                && inertia[3 * b + 2] >= 1e-15;
            if ok {
                return true;
            }
            (1..parent.len()).any(|c| {
                parent[c] == b && jntnum[c] == 0 && valid(c, mass, inertia, parent, jntnum)
            })
        }
        for b in 1..nbody {
            if body_jntnum[b] > 0
                && !valid(b, &body_mass, &body_inertia, &body_parentid, &body_jntnum)
            {
                return Err(PhysicsError::DegenerateMass { body: b - 1 });
            }
        }

        // ---- fixed tendons
        let mut tendon_adr = Vec::with_capacity(ntendon);
        let mut tendon_num = Vec::with_capacity(ntendon);
        let mut wrap_objid = Vec::new();
        let mut wrap_prm = Vec::new();
        let mut ten_j_colind = Vec::new();
        let mut ten_j_rowadr = Vec::with_capacity(ntendon);
        let mut ten_j_rownnz = Vec::with_capacity(ntendon);
        let mut tendon_limited = Vec::with_capacity(ntendon);
        let mut tendon_range = Vec::with_capacity(2 * ntendon);
        let mut tendon_margin = Vec::with_capacity(ntendon);
        let mut tendon_frictionloss = Vec::with_capacity(ntendon);
        let mut tendon_solref_lim = Vec::with_capacity(2 * ntendon);
        let mut tendon_solimp_lim = Vec::with_capacity(5 * ntendon);
        let mut tendon_solref_fri = Vec::with_capacity(2 * ntendon);
        let mut tendon_solimp_fri = Vec::with_capacity(5 * ntendon);
        for tendon in &scene.tendons {
            tendon_adr.push(wrap_objid.len());
            tendon_num.push(tendon.joints.len());
            let row = ten_j_colind.len();
            ten_j_rowadr.push(row);
            ten_j_rownnz.push(tendon.joints.len());
            for term in &tendon.joints {
                wrap_objid.push(term.joint.index());
                wrap_prm.push(term.coef);
                ten_j_colind.push(jnt_dofadr[term.joint.index()]);
            }
            // MuJoCo sorts the dofs of a tendon's Jacobian row ascending
            ten_j_colind[row..].sort_unstable();
            tendon_limited.push(tendon.range.is_some());
            tendon_range.extend_from_slice(&tendon.range.unwrap_or([0.0, 0.0]));
            tendon_margin.push(tendon.margin);
            tendon_frictionloss.push(tendon.frictionloss);
            tendon_solref_lim.extend_from_slice(&tendon.solref_limit);
            tendon_solimp_lim.extend_from_slice(&tendon.solimp_limit);
            tendon_solref_fri.extend_from_slice(&tendon.solref_friction);
            tendon_solimp_fri.extend_from_slice(&tendon.solimp_friction);
        }

        // ---- geoms (MuJoCo's order: sorted by body, the world first; sim-scene's importer
        // sorts them, and a hand-built scene is refused if it does not)
        let ngeom = scene.geoms.len();
        if nbody >= 1 << 16 {
            return Err(PhysicsError::Scene {
                message: format!(
                    "{nbody} bodies: MuJoCo's collision pairs use 16-bit body ids, so a model needs fewer than 65536"
                ),
            });
        }
        let mut geom_type = Vec::with_capacity(ngeom);
        let mut geom_bodyid = Vec::with_capacity(ngeom);
        let mut geom_pos = Vec::with_capacity(3 * ngeom);
        let mut geom_quat = Vec::with_capacity(4 * ngeom);
        let mut geom_size = Vec::with_capacity(3 * ngeom);
        let mut geom_rbound = Vec::with_capacity(ngeom);
        let mut geom_sameframe = Vec::with_capacity(ngeom);
        let mut geom_contype = Vec::with_capacity(ngeom);
        let mut geom_conaffinity = Vec::with_capacity(ngeom);
        let mut geom_condim = Vec::with_capacity(ngeom);
        let mut geom_priority = Vec::with_capacity(ngeom);
        let mut geom_solmix = Vec::with_capacity(ngeom);
        let mut geom_solref = Vec::with_capacity(2 * ngeom);
        let mut geom_solimp = Vec::with_capacity(5 * ngeom);
        let mut geom_friction = Vec::with_capacity(3 * ngeom);
        let mut geom_margin = Vec::with_capacity(ngeom);
        let mut geom_gap = Vec::with_capacity(ngeom);
        let mut body_geomadr = vec![-1i32; nbody];
        let mut body_geomnum = vec![0usize; nbody];
        let mut previous_body = 0usize;
        for (g, geom) in scene.geoms.iter().enumerate() {
            let b = geom.body.map_or(0, |b| scene_body_to_internal(b.index()));
            if b < previous_body {
                return Err(PhysicsError::Scene {
                    message: format!(
                        "geom {g} comes after a geom of a later body: geoms must be listed by body, the world's first, as MuJoCo numbers them"
                    ),
                });
            }
            previous_body = b;
            if body_geomnum[b] == 0 {
                body_geomadr[b] = g as i32;
            }
            body_geomnum[b] += 1;
            geom_bodyid.push(b);
            geom_pos.extend_from_slice(&geom.pos);
            geom_quat.extend_from_slice(&quat_wxyz_normalised(geom.quat));
            let (ty, size, rbound) = match geom.shape {
                Shape::Sphere { r } => (GeomType::Sphere, [r, 0.0, 0.0], r),
                Shape::Capsule { r, half_len } => {
                    (GeomType::Capsule, [r, half_len, 0.0], r + half_len)
                }
                Shape::Cylinder { r, half_len } => (
                    GeomType::Cylinder,
                    [r, half_len, 0.0],
                    (r * r + half_len * half_len).sqrt(),
                ),
                Shape::Box { half } => (
                    GeomType::Box,
                    half,
                    (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt(),
                ),
                Shape::Ellipsoid { radii } => (
                    GeomType::Ellipsoid,
                    radii,
                    radii[0].max(radii[1]).max(radii[2]),
                ),
                Shape::Plane { size } => (GeomType::Plane, size, 0.0),
                // a mesh has no collider here; its bounding radius is the farthest vertex
                // (MuJoCo's rule for a mesh is not in the reference tree)
                Shape::Mesh { mesh } => {
                    let r = scene.meshes[mesh.index()]
                        .vertices
                        .iter()
                        .map(|v| {
                            let (x, y, z) = (f64::from(v[0]), f64::from(v[1]), f64::from(v[2]));
                            (x * x + y * y + z * z).sqrt()
                        })
                        .fold(0.0f64, f64::max);
                    (GeomType::Mesh, [0.0; 3], r)
                }
            };
            geom_type.push(ty);
            geom_size.extend_from_slice(&size);
            geom_rbound.push(rbound);
            geom_contype.push(geom.contype);
            geom_conaffinity.push(geom.conaffinity);
            geom_condim.push(geom.condim);
            geom_priority.push(geom.priority);
            geom_solmix.push(geom.solmix);
            geom_solref.extend_from_slice(&geom.solref);
            geom_solimp.extend_from_slice(&geom.solimp);
            geom_friction.extend_from_slice(&geom.friction);
            geom_margin.push(geom.margin);
            geom_gap.push(geom.gap);
        }
        // the compiler's rule for geom_sameframe (user_model.cc), from the compiled frames
        for g in 0..ngeom {
            let b = geom_bodyid[g];
            geom_sameframe.push(geom_same_frame(
                &geom_pos[3 * g..3 * g + 3],
                &geom_quat[4 * g..4 * g + 4],
                &body_ipos[3 * b..3 * b + 3],
                &body_iquat[4 * b..4 * b + 4],
            ));
        }
        let body_has_bvh: Vec<bool> = body_geomnum.iter().map(|&n| n > 0).collect();

        // ---- the body pairs that never collide (<contact><exclude>)
        let mut exclude_signature: Vec<u32> = scene
            .contact_excludes
            .iter()
            .map(|x| {
                let internal = |b: Option<sim_scene::BodyId>| {
                    b.map_or(0usize, |b| scene_body_to_internal(b.index()))
                };
                let (a, b) = (internal(x.body1), internal(x.body2));
                ((a.min(b) << 16) + a.max(b)) as u32
            })
            .collect();
        exclude_signature.sort_unstable();

        // ---- actuators
        let mut actuator_type = Vec::with_capacity(nu);
        let mut actuator_jnt = Vec::with_capacity(nu);
        let mut actuator_qposadr = Vec::with_capacity(nu);
        let mut actuator_dofadr = Vec::with_capacity(nu);
        let mut actuator_gear = Vec::with_capacity(nu);
        let mut actuator_kp = Vec::with_capacity(nu);
        let mut actuator_ctrllimited = Vec::with_capacity(nu);
        let mut actuator_ctrlrange = Vec::with_capacity(2 * nu);
        for a in &scene.actuators {
            let j = a.joint.index();
            actuator_jnt.push(j);
            actuator_qposadr.push(jnt_qposadr[j]);
            actuator_dofadr.push(jnt_dofadr[j]);
            let (ty, gear, kp, range) = match a.kind {
                ActuatorKind::Motor { gear, ctrlrange } => {
                    (ActuatorType::Motor, gear, 0.0, ctrlrange)
                }
                ActuatorKind::Position {
                    kp,
                    gear,
                    ctrlrange,
                } => (ActuatorType::Position, gear, kp, ctrlrange),
            };
            actuator_type.push(ty);
            actuator_gear.push(gear);
            actuator_kp.push(kp);
            actuator_ctrllimited.push(range.is_some());
            let [lo, hi] = range.unwrap_or([0.0, 0.0]);
            actuator_ctrlrange.push(lo);
            actuator_ctrlrange.push(hi);
        }

        let has_damping = dof_damping.iter().any(|&d| d > 0.0);
        let o = &scene.options;
        let opt = Options {
            solver: match o.solver {
                Solver::Cg => PrimalSolver::Cg,
                // PGS is refused: Newton runs, and `list_not_modelled` says so
                Solver::Newton | Solver::Pgs => PrimalSolver::Newton,
            },
            iterations: o.iterations as usize,
            tolerance: o.tolerance,
            ls_iterations: o.ls_iterations as usize,
            ls_tolerance: o.ls_tolerance,
            impratio: o.impratio,
            cone: o.cone,
        };

        let mut model = Model {
            nbody,
            njnt,
            nq,
            nv,
            nu,
            ntendon,
            nefc_max: 0,
            ngeom,
            ncon_max: 0,
            timestep: scene.timestep_s,
            gravity: scene.gravity,
            integrator: scene.integrator,
            opt,
            disable: DisableFlags::default(),
            body_parentid,
            body_rootid,
            body_weldid,
            body_jntadr,
            body_jntnum,
            body_dofadr,
            body_dofnum,
            body_pos,
            body_quat,
            body_mass,
            body_subtreemass,
            body_ipos,
            body_iquat,
            body_sameframe,
            body_inertia,
            body_invweight0: vec![0.0; 2 * nbody],
            body_geomadr,
            body_geomnum,
            body_has_bvh,
            exclude_signature,
            geom_type,
            geom_bodyid,
            geom_pos,
            geom_quat,
            geom_size,
            geom_rbound,
            geom_sameframe,
            geom_contype,
            geom_conaffinity,
            geom_condim,
            geom_priority,
            geom_solmix,
            geom_solref,
            geom_solimp,
            geom_friction,
            geom_margin,
            geom_gap,
            candidates: Vec::new(),
            candidates_built_filterparent: false,
            candidates_built_midphase: false,
            candidates_fingerprint: 0,
            jnt_type,
            jnt_qposadr,
            jnt_dofadr,
            jnt_bodyid,
            jnt_pos,
            jnt_axis,
            jnt_stiffness,
            jnt_limited,
            jnt_range,
            jnt_margin,
            jnt_solref,
            jnt_solimp,
            qpos_spring: qpos0.clone(),
            qpos0,
            dof_bodyid,
            dof_jntid,
            dof_parentid,
            dof_armature,
            dof_damping,
            dof_frictionloss,
            dof_solref,
            dof_solimp,
            dof_invweight0: vec![0.0; nv],
            dof_simplenum: vec![0; nv],
            dof_m0: vec![0.0; nv],
            qm_sparsity: Sparsity::default(),
            has_damping,
            meaninertia: 1.0,
            tendon_adr,
            tendon_num,
            wrap_objid,
            wrap_prm,
            ten_j_colind,
            ten_j_rowadr,
            ten_j_rownnz,
            tendon_limited,
            tendon_range,
            tendon_margin,
            tendon_frictionloss,
            tendon_solref_lim,
            tendon_solimp_lim,
            tendon_solref_fri,
            tendon_solimp_fri,
            tendon_invweight0: vec![0.0; ntendon],
            actuator_type,
            actuator_jnt,
            actuator_qposadr,
            actuator_dofadr,
            actuator_gear,
            actuator_kp,
            actuator_ctrllimited,
            actuator_ctrlrange,
        };
        model.set_sparsity();
        model.set_const();

        // the static candidate list and the contact maxima (the geom pairs that have no
        // collider are listed, never ignored silently)
        let (list, counts) = candidates::build_candidates(&model, faults);
        model.set_candidates(list);
        let not_modelled = list_not_modelled(scene, &counts);
        Ok((model, not_modelled))
    }

    /// `dof_simplenum` and the sparsity of `M` (MuJoCo's `FinalizeSimple` and
    /// `ComputeSparseSizes` of `user_model.cc`): a dof of a simple body has the
    /// diagonal only, any other dof its ancestors in the tree of dofs and itself.
    fn set_sparsity(&mut self) {
        let nv = self.nv;
        let simple = self.body_simple();
        // dof_simplenum: the run of dofs of simple bodies that starts at each dof
        let mut count = 0usize;
        for i in (0..nv).rev() {
            count = if simple[self.dof_bodyid[i]] != 0 {
                count + 1
            } else {
                0
            };
            self.dof_simplenum[i] = count;
        }
        let mut s = Sparsity::default();
        for i in 0..nv {
            s.rowadr.push(s.colind.len());
            let start = s.colind.len();
            if self.dof_simplenum[i] == 0 {
                // the ancestors, ascending (a parent has a smaller index)
                let mut j = self.dof_parentid[i];
                while j >= 0 {
                    s.colind.push(j as usize);
                    j = self.dof_parentid[j as usize];
                }
                s.colind[start..].reverse();
            }
            s.colind.push(i);
            s.rownnz.push(s.colind.len() - start);
        }
        self.qm_sparsity = s;
    }

    /// Port of the `qpos0` part of `mj_setConst` (`set0`, `setStat` in
    /// `engine_setconst.c`): `body_invweight0`, `dof_invweight0`, `tendon_invweight0`
    /// and `meaninertia`, from the inertia matrix at `qpos0`.
    ///
    /// `body_simple` (MuJoCo's flag for a body whose inertia matrix is constant:
    /// inertial frame at the body frame, no rotation before or after, only slide
    /// joints on an axis through the body origin) is derived here from the compile
    /// rules of `user_model.cc`; a body with only slides (`body_simple == 2`) takes
    /// `1 / mass` as its inverse weight, as MuJoCo does (the armature is not in it).
    fn set_const(&mut self) {
        let nv = self.nv;
        let simple = self.body_simple();
        let sp = self.qm_sparsity.clone();

        let mut d = Data::new(self);
        d.qpos.copy_from_slice(&self.qpos0);
        smooth::kinematics(self, &mut d);
        smooth::com_pos(self, &mut d);
        // dof_M0 first (mj_setM0), because `crb` writes it as the row of a simple dof
        self.dof_m0 = set_m0(self, &d);
        smooth::crb(self, &mut d);
        smooth::factor_m(self, &mut d);
        smooth::tendon(self, &mut d);

        // meaninertia: the mean of the diagonal of M at qpos0
        if nv > 0 {
            let mut s = 0.0;
            for i in 0..nv {
                s += d.qm[i * nv + i];
            }
            self.meaninertia = s / nv as f64;
        }

        // A = J inv(M) J' for the rows of `jac` (n rows of nv), into `a` (n x n)
        let inv_weight = |d: &Data<f64>, jac: &[f64], n: usize, a: &mut [f64]| {
            let mut tmp = vec![0.0f64; n * nv];
            for r in 0..n {
                let row = &mut tmp[r * nv..(r + 1) * nv];
                row.copy_from_slice(&jac[r * nv..(r + 1) * nv]);
                factor::solve(row, &d.qld, &d.qld_diag_inv, nv, &sp);
            }
            for i in 0..n {
                for j in 0..n {
                    a[i * n + j] =
                        crate::linalg::dot(&jac[i * nv..(i + 1) * nv], &tmp[j * nv..(j + 1) * nv]);
                }
            }
        };

        // body_invweight0
        self.body_invweight0[0] = 0.0;
        self.body_invweight0[1] = 0.0;
        for (i, &simple_i) in simple.iter().enumerate().skip(1) {
            if self.body_dofnum[self.body_weldid[i]] == 0 {
                // bodies with no dofs: zero invweight0
                self.body_invweight0[2 * i] = 0.0;
                self.body_invweight0[2 * i + 1] = 0.0;
            } else if simple_i == 2 {
                self.body_invweight0[2 * i] = 1.0 / self.body_mass[i].max(1e-15);
                self.body_invweight0[2 * i + 1] = 0.0;
            } else {
                // inverse spatial inertia: A = J inv(M) J'
                let mut jac = vec![0.0f64; 6 * nv];
                self.jac_body_com(&d, i, &mut jac);
                let mut a = [0.0f64; 36];
                if nv > 0 {
                    inv_weight(&d, &jac, 6, &mut a);
                }
                self.body_invweight0[2 * i] = (a[0] + a[7] + a[14]) / 3.0;
                self.body_invweight0[2 * i + 1] = (a[21] + a[28] + a[35]) / 3.0;
            }
        }

        // dof_invweight0
        for j in 0..self.njnt {
            let id = self.jnt_dofadr[j];
            let bi = self.jnt_bodyid[j];
            if simple[bi] == 2 {
                // simple body with no rotations: no off-diagonal inertia
                self.dof_invweight0[id] = 1.0 / self.body_mass[bi].max(1e-15);
            } else {
                let dnum = self.jnt_type[j].nv();
                let mut a = [0.0f64; 36];
                if nv > 0 {
                    let mut jac = vec![0.0f64; dnum * nv];
                    for r in 0..dnum {
                        jac[r * (nv + 1) + id] = 1.0;
                    }
                    inv_weight(&d, &jac, dnum, &mut a);
                }
                match dnum {
                    6 => {
                        let t = (a[0] + a[7] + a[14]) / 3.0;
                        let r = (a[21] + a[28] + a[35]) / 3.0;
                        self.dof_invweight0[id..id + 3].fill(t);
                        self.dof_invweight0[id + 3..id + 6].fill(r);
                    }
                    3 => {
                        let v = (a[0] + a[4] + a[8]) / 3.0;
                        self.dof_invweight0[id..id + 3].fill(v);
                    }
                    _ => self.dof_invweight0[id] = a[0],
                }
            }
        }

        // tendon_invweight0: ten_J inv(M) ten_J'
        for t in 0..self.ntendon {
            let row = d.ten_j[t * nv..(t + 1) * nv].to_vec();
            let mut x = row.clone();
            factor::solve(&mut x, &d.qld, &d.qld_diag_inv, nv, &sp);
            self.tendon_invweight0[t] = crate::linalg::dot(&row, &x);
        }
    }

    /// MuJoCo's `mj_jacBodyCom`: the 6-by-`nv` Jacobian of a body's centre of mass,
    /// translation rows first, then rotation, from the kinematics in `d`.
    fn jac_body_com(&self, d: &Data<f64>, body: usize, jac: &mut [f64]) {
        let nv = self.nv;
        let (jacp, jacr) = jac.split_at_mut(3 * nv);
        crate::jac::jac(
            self,
            &d.cdof,
            &d.subtree_com,
            Some(jacp),
            Some(jacr),
            crate::math::v3(&d.xipos, body),
            body,
        );
    }

    /// MuJoCo's `body_simple` (0, 1 or 2) for every body, by the rules of
    /// `mjCModel::CompileFull` (`user_model.cc`).
    fn body_simple(&self) -> Vec<u8> {
        const EPS: f64 = 1e-14;
        let near_zero3 = |v: &[f64]| v.iter().all(|x| x.abs() < FRAME_EPS);
        let mut simple = vec![0u8; self.nbody];
        for i in 1..self.nbody {
            // the inertial frame is the body frame (to kFrameEps)
            let sameframe = self.body_sameframe[i] == SameFrame::Body;
            let parent = self.body_parentid[i];
            let is_simple = sameframe
                && (self.body_rootid[i] == i
                    || (self.body_parentid[parent] == 0 && self.body_dofnum[parent] == 0));
            simple[i] = u8::from(is_simple);
            // a parent body is never simple (unless world)
            if parent > 0 {
                simple[parent] = 0;
            }
            // loop over the joints of this body
            let mut rotfound = false;
            for j in self.body_jntadr[i]..self.body_jntadr[i] + self.body_jntnum[i] {
                let axis = &self.jnt_axis[3 * j..3 * j + 3];
                let aligned = axis.iter().filter(|x| x.abs() > EPS).count() == 1;
                let ty = self.jnt_type[j];
                if rotfound
                    || !near_zero3(&self.jnt_pos[3 * j..3 * j + 3])
                    || (matches!(ty, JointType::Hinge | JointType::Slide) && !aligned)
                {
                    simple[i] = 0;
                }
                if matches!(ty, JointType::Ball | JointType::Hinge) {
                    rotfound = true;
                }
            }
            // simple body with sliders and no rotational dofs: level 2
            if simple[i] != 0 && self.body_dofnum[i] > 0 {
                simple[i] = 2;
                for j in self.body_jntadr[i]..self.body_jntadr[i] + self.body_jntnum[i] {
                    if self.jnt_type[j] != JointType::Slide {
                        simple[i] = 1;
                        break;
                    }
                }
            }
        }
        simple
    }
}

/// `kFrameEps` of `user_model.cc`: the difference below which two frames are the same.
const FRAME_EPS: f64 = 1e-6;

/// `IsSameVec` (`user_model.cc`): two 3-vectors are element-wise less than `kFrameEps` apart.
fn same_vec(a: &[f64], b: &[f64]) -> bool {
    (0..3).all(|k| (a[k] - b[k]).abs() < FRAME_EPS)
}

/// `IsSameQuat` (`user_model.cc`): two quaternions are element-wise less than `kFrameEps`
/// apart, including the double cover (`q` and `-q`).
fn same_quat(a: &[f64], b: &[f64]) -> bool {
    let minus = (0..4).all(|k| (a[k] - b[k]).abs() < FRAME_EPS);
    let plus = (0..4).all(|k| (a[k] + b[k]).abs() < FRAME_EPS);
    minus || plus
}

/// The compiler's rule for `body_sameframe` (`mjCModel::CompileFull`, `user_model.cc`): the
/// inertial pose is null (`IsNullPose`, to `kFrameEps`) gives `Body`; a null orientation
/// alone (`IsNullPose` with no position) gives `BodyRot`; anything else `None`.
fn body_same_frame(ipos: &[f64], iquat: &[f64]) -> SameFrame {
    let unit = [1.0f64, 0.0, 0.0, 0.0];
    if same_vec(ipos, &[0.0; 3]) && same_quat(iquat, &unit) {
        SameFrame::Body
    } else if same_quat(iquat, &unit) {
        SameFrame::BodyRot
    } else {
        SameFrame::None
    }
}

/// The compiler's rule for `geom_sameframe` (`mjCModel::CompileFull`, `user_model.cc`):
/// the geom's pose equals its body's (`Body`), its orientation does (`BodyRot`), its pose
/// equals the body's inertial frame (`Inertia`), its orientation does (`InertiaRot`), or
/// none. Positions are compared element-wise to `kFrameEps = 1e-6`, quaternions the same
/// way including the double cover (`q` and `-q`).
fn geom_same_frame(pos: &[f64], quat: &[f64], ipos: &[f64], iquat: &[f64]) -> SameFrame {
    let zero = [0.0f64; 3];
    let unit = [1.0f64, 0.0, 0.0, 0.0];
    if same_vec(pos, &zero) && same_quat(quat, &unit) {
        SameFrame::Body
    } else if same_quat(quat, &unit) {
        SameFrame::BodyRot
    } else if same_vec(pos, ipos) && same_quat(quat, iquat) {
        SameFrame::Inertia
    } else if same_quat(quat, iquat) {
        SameFrame::InertiaRot
    } else {
        SameFrame::None
    }
}

/// Everything in the scene this phase ignores (see [`NotModelled`]); `counts` are the geom
/// pairs of the collision candidate list that have no collider.
fn list_not_modelled(scene: &Scene, counts: &[candidates::CollisionCount]) -> Vec<NotModelled> {
    let mut out = Vec::new();
    if scene.options.solver == Solver::Pgs {
        out.push(NotModelled::PgsSolver);
    }
    for (t, tendon) in scene.tendons.iter().enumerate() {
        if tendon.stiffness > 0.0 || tendon.damping > 0.0 || tendon.armature > 0.0 {
            out.push(NotModelled::TendonPassive {
                tendon: t,
                stiffness: tendon.stiffness,
                damping: tendon.damping,
                armature: tendon.armature,
            });
        }
    }

    // geom pairs that MuJoCo's filters let through and that have no collider here
    for c in counts {
        out.push(NotModelled::Collision {
            type1: c.type1,
            type2: c.type2,
            pairs: c.pairs,
            first_pair: c.first_pair,
        });
    }

    for u in &scene.unsupported {
        out.push(NotModelled::Unsupported {
            path: u.path.clone(),
            item: u.item.clone(),
            line: u.line,
            reason: u.reason.clone(),
        });
    }
    out
}
