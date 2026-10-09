//! The one scene description: what physics simulates, what the renderer draws
//! and what the senses read.
//!
//! Invariants (checked by [`Scene::validate`], which every producer calls and
//! `Scene::from_json` calls for you):
//! - SI units throughout; world frame Z up, right-handed; quaternions
//!   `[x, y, z, w]` Hamilton (the convention of `sim-contract` v0).
//! - Every id (`BodyId`, `JointId`, ...) indexes a list of the same scene.
//! - JSON round-trips exactly (`Scene::to_json` then `Scene::from_json` gives an
//!   equal scene) and refuses unknown fields. The scene's own `version` must be
//!   [`SCENE_VERSION`].
//! - Nothing is dropped silently: an element of a source document that the
//!   scene cannot represent yet is listed in [`Scene::unsupported`].
//! - Instances place renderable things (a mesh or a geom) in the world and give
//!   each a segmentation id. Id 0 is reserved for the background, so no
//!   instance carries it.
//! - Cameras use the OpenCV camera frame: x right, y down, z forward (the
//!   camera looks along +z). Intrinsics are in pixels with pixel centres at
//!   INTEGER coordinates, so a centred principal point of a W x H image is
//!   `((W - 1) / 2, (H - 1) / 2)`. (Agreed with Lane R and the contract owner.)

use serde::{Deserialize, Serialize};

use crate::body::{Body, Geom, Joint, Mesh};
use crate::body::{default_solimp, default_solref};
use crate::error::{Result, SceneError};
use crate::ids::{BodyId, GeomId, JointId, MaterialId, MeshId};
use crate::material::Material;

/// The scene schema version this crate reads and writes.
pub const SCENE_VERSION: u32 = 0;

/// Gravity when a scene does not say: 9.81 m/s^2 down the Z axis.
pub const DEFAULT_GRAVITY: [f64; 3] = [0.0, 0.0, -9.81];

/// The physics timestep when a scene does not say: 2 ms (MuJoCo's default).
pub const DEFAULT_TIMESTEP_S: f64 = 0.002;

fn default_gravity() -> [f64; 3] {
    DEFAULT_GRAVITY
}

fn default_timestep() -> f64 {
    DEFAULT_TIMESTEP_S
}

fn identity_quat() -> [f64; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

/// The time integrator of the physics step (MuJoCo's `<option integrator>`).
///
/// `implicit`, `implicitfast` and `discrete` are not modelled: the MJCF importer
/// records them in [`Scene::unsupported`] and the scene keeps [`Integrator::Euler`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Integrator {
    /// MuJoCo's semi-implicit Euler: the velocity is advanced first, the position
    /// is advanced with the new velocity, and joint damping is integrated
    /// implicitly. First order. This is the default, as in MuJoCo.
    #[default]
    Euler,
    /// MuJoCo's classical fourth-order Runge-Kutta.
    Rk4,
}

/// The constraint solver MuJoCo runs (`<option solver>`).
///
/// `Pgs` is imported and carried, and `sim-physics` refuses to run it (phase 1c-i
/// has the two primal solvers only).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Solver {
    /// MuJoCo's projected Gauss-Seidel dual solver (`PGS`).
    Pgs,
    /// MuJoCo's conjugate-gradient primal solver (`CG`).
    Cg,
    /// MuJoCo's Newton primal solver (`Newton`), the default.
    #[default]
    Newton,
}

/// The friction cone of contacts (`<option cone>`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cone {
    /// MuJoCo's default: a pyramid of friction directions.
    #[default]
    Pyramidal,
    /// The elliptic (round) cone.
    Elliptic,
}

fn default_iterations() -> u32 {
    100
}

fn default_tolerance() -> f64 {
    1e-8
}

fn default_ls_iterations() -> u32 {
    50
}

fn default_ls_tolerance() -> f64 {
    0.01
}

fn default_impratio() -> f64 {
    1.0
}

/// The constraint-solver options of a scene (MuJoCo's `<option>` solver fields),
/// with MuJoCo's defaults.
///
/// Every field has a documented default, so `{}` is the default set and JSON
/// round-trips exactly. The numbers are what `sim-physics` runs with: the primal
/// solvers stop when the scaled cost improvement or the scaled gradient is below
/// `tolerance`, or after `iterations`; the exact line search stops at
/// `ls_tolerance * tolerance` (relative slope) or after `ls_iterations`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverOptions {
    /// The solver. Default [`Solver::Newton`].
    #[serde(default)]
    pub solver: Solver,
    /// The most solver iterations per solve. Default 100.
    #[serde(default = "default_iterations")]
    pub iterations: u32,
    /// The solver tolerance, >= 0 (0 never stops early). Default 1e-8.
    #[serde(default = "default_tolerance")]
    pub tolerance: f64,
    /// The most line-search iterations per solver iteration. Default 50.
    #[serde(default = "default_ls_iterations")]
    pub ls_iterations: u32,
    /// The line-search tolerance, >= 0, relative to `tolerance`. Default 0.01.
    #[serde(default = "default_ls_tolerance")]
    pub ls_tolerance: f64,
    /// The friction cone of contacts. Default [`Cone::Pyramidal`].
    #[serde(default)]
    pub cone: Cone,
    /// The ratio of friction to normal contact impedance, > 0. Default 1.
    #[serde(default = "default_impratio")]
    pub impratio: f64,
}

impl Default for SolverOptions {
    fn default() -> Self {
        SolverOptions {
            solver: Solver::Newton,
            iterations: default_iterations(),
            tolerance: default_tolerance(),
            ls_iterations: default_ls_iterations(),
            ls_tolerance: default_ls_tolerance(),
            cone: Cone::Pyramidal,
            impratio: default_impratio(),
        }
    }
}

/// A scene.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scene {
    /// The scene schema version; must equal [`SCENE_VERSION`].
    pub version: u32,
    /// A name for the scene (the MJCF `model` attribute when imported).
    #[serde(default)]
    pub name: String,
    /// Gravity, m/s^2, world frame. Default [`DEFAULT_GRAVITY`].
    #[serde(default = "default_gravity")]
    pub gravity: [f64; 3],
    /// The physics timestep, seconds, > 0. Default [`DEFAULT_TIMESTEP_S`].
    #[serde(default = "default_timestep")]
    pub timestep_s: f64,
    /// The time integrator of the physics step. Default [`Integrator::Euler`].
    #[serde(default)]
    pub integrator: Integrator,
    /// The constraint-solver options. Default [`SolverOptions::default`].
    #[serde(default)]
    pub options: SolverOptions,
    /// The rigid bodies (the world is implicit and not listed).
    #[serde(default)]
    pub bodies: Vec<Body>,
    /// The joints.
    #[serde(default)]
    pub joints: Vec<Joint>,
    /// The geoms.
    #[serde(default)]
    pub geoms: Vec<Geom>,
    /// The triangle meshes.
    #[serde(default)]
    pub meshes: Vec<Mesh>,
    /// The materials.
    #[serde(default)]
    pub materials: Vec<Material>,
    /// The renderable instances.
    #[serde(default)]
    pub instances: Vec<Instance>,
    /// The cameras.
    #[serde(default)]
    pub cameras: Vec<Camera>,
    /// The actuators.
    #[serde(default)]
    pub actuators: Vec<Actuator>,
    /// Fixed tendons (their limits and friction loss are simulated; their spring,
    /// damper and armature are not yet).
    #[serde(default)]
    pub tendons: Vec<Tendon>,
    /// Pairs of bodies whose geoms never collide (MuJoCo's `<contact><exclude>`).
    #[serde(default)]
    pub contact_excludes: Vec<ContactExclude>,
    /// What the source document carried that this scene does not represent (or
    /// represents without physics using it yet). Empty for a scene built here.
    #[serde(default)]
    pub unsupported: Vec<Unsupported>,
}

impl Default for Scene {
    fn default() -> Self {
        Scene::new()
    }
}

impl Scene {
    /// An empty scene with the documented defaults.
    pub fn new() -> Scene {
        Scene {
            version: SCENE_VERSION,
            name: String::new(),
            gravity: DEFAULT_GRAVITY,
            timestep_s: DEFAULT_TIMESTEP_S,
            integrator: Integrator::Euler,
            options: SolverOptions::default(),
            bodies: Vec::new(),
            joints: Vec::new(),
            geoms: Vec::new(),
            meshes: Vec::new(),
            materials: Vec::new(),
            instances: Vec::new(),
            cameras: Vec::new(),
            actuators: Vec::new(),
            tendons: Vec::new(),
            contact_excludes: Vec::new(),
            unsupported: Vec::new(),
        }
    }

    /// Length of the joint position vector (MuJoCo's `nq`): the sum of each
    /// joint's `nq`.
    pub fn nq(&self) -> usize {
        self.joints.iter().map(|j| j.kind.nq()).sum()
    }

    /// Length of the joint velocity vector (MuJoCo's `nv`).
    pub fn nv(&self) -> usize {
        self.joints.iter().map(|j| j.kind.nv()).sum()
    }

    /// Writes the scene as JSON (compact). The scene is not validated.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| SceneError::Json {
            message: e.to_string(),
        })
    }

    /// Writes the scene as indented JSON.
    pub fn to_json_pretty(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| SceneError::Json {
            message: e.to_string(),
        })
    }

    /// Reads a scene from JSON and validates it. Unknown fields, a missing
    /// required field and a wrong type are refused with [`SceneError::Json`];
    /// a scene that parses but breaks an invariant is refused with
    /// [`SceneError::Invalid`].
    pub fn from_json(json: &str) -> Result<Scene> {
        let scene: Scene = serde_json::from_str(json).map_err(|e| SceneError::Json {
            message: e.to_string(),
        })?;
        scene.validate()?;
        Ok(scene)
    }
}

/// What an instance draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "id",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum ShapeRef {
    /// A mesh from `Scene::meshes`.
    Mesh(MeshId),
    /// A geom from `Scene::geoms`: the instance draws the geom's shape.
    Geom(GeomId),
}

/// A renderable placement of a mesh or a geom.
///
/// The instance's pose in its body frame is `local_pos`/`local_quat` for a mesh.
/// For a geom reference the geom's own pose in the body applies first and the
/// local pose is an extra offset in the geom's frame (the identity in the common
/// case); the instance must then be attached to the geom's own body.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Instance {
    /// The body it follows, or `None` for a static instance (placed in the
    /// world frame by its local pose).
    pub body: Option<BodyId>,
    /// What it draws.
    pub mesh_or_geom: ShapeRef,
    /// The material it is drawn with.
    pub material: MaterialId,
    /// Position, metres (see the type note).
    #[serde(default)]
    pub local_pos: [f64; 3],
    /// Orientation, unit `[x, y, z, w]` (see the type note).
    #[serde(default = "identity_quat")]
    pub local_quat: [f64; 4],
    /// The segmentation id this instance writes. 0 is the background and is
    /// refused here.
    pub seg_id: u16,
}

/// A pinhole camera.
///
/// Convention (agreed with Lane R and the contract owner): the OpenCV camera
/// frame, x right, y down, z forward; the camera looks along +z. `fx`, `fy`,
/// `cx`, `cy` are in pixels, with pixel CENTRES at INTEGER coordinates: pixel
/// `(0, 0)` is centred on `(0.0, 0.0)`, so the principal point of an image
/// `W` pixels wide and `H` high that is centred on its middle is
/// `((W - 1) / 2, (H - 1) / 2)`. A point `(x, y, z)` in the camera frame
/// projects to `u = fx * x / z + cx`, `v = fy * y / z + cy`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Camera {
    /// The camera's name. Unique within a scene when non-empty.
    pub name: String,
    /// Where the camera frame is.
    pub mount: CameraMount,
    /// Focal length along x, pixels, > 0.
    pub fx: f32,
    /// Focal length along y, pixels, > 0.
    pub fy: f32,
    /// Principal point x, pixels.
    pub cx: f32,
    /// Principal point y, pixels.
    pub cy: f32,
    /// Near clip distance, metres, > 0.
    pub near: f32,
    /// Far clip distance, metres, > `near`.
    pub far: f32,
    /// Image width, pixels, > 0.
    pub width: u32,
    /// Image height, pixels, > 0.
    pub height: u32,
}

/// Where a camera's frame is.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CameraMount {
    /// Fixed in the world frame.
    World {
        /// Position of the camera frame, metres.
        pos: [f64; 3],
        /// Orientation of the camera frame, unit `[x, y, z, w]`.
        quat: [f64; 4],
    },
    /// Fixed in a body's frame, so it follows the body.
    Body {
        /// The body.
        body: BodyId,
        /// Position of the camera frame in the body frame, metres.
        local_pos: [f64; 3],
        /// Orientation of the camera frame in the body frame, unit `[x, y, z, w]`.
        local_quat: [f64; 4],
    },
}

/// An actuator driving one joint.
///
/// Only hinge and slide joints can be driven in this version, so one actuator
/// is one control value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Actuator {
    /// The actuator's name. Unique within a scene when non-empty.
    pub name: String,
    /// The joint it drives (a hinge or a slide).
    pub joint: JointId,
    /// What kind of actuator it is.
    pub kind: ActuatorKind,
}

/// The kind of an actuator.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ActuatorKind {
    /// A direct-drive motor: the joint force (or torque) is `gear * ctrl`.
    Motor {
        /// Force or torque per unit control, finite.
        gear: f64,
        /// Limits on the control, or `None` for unlimited.
        ctrlrange: Option<[f64; 2]>,
    },
    /// A position servo (MuJoCo's `<position>` with `kv` 0): the actuator force is
    /// `kp * (ctrl - gear * q)` and the joint force is `gear` times that.
    Position {
        /// Proportional gain, >= 0.
        kp: f64,
        /// Transmission ratio, finite (MuJoCo's `gear[0]`; 1 when absent, which is
        /// also MuJoCo's default). An actuator default's `gear` applies here too.
        #[serde(default = "unit_gear")]
        gear: f64,
        /// Limits on the control, or `None` for unlimited.
        ctrlrange: Option<[f64; 2]>,
    },
}

fn unit_gear() -> f64 {
    1.0
}

/// A fixed tendon: a length that is a linear combination of joint coordinates.
///
/// Physics simulates its limit (`range`) and its dry friction (`frictionloss`)
/// as soft constraints; its `stiffness`, `damping` and `armature` are carried but
/// not simulated yet (the importer lists a tendon that has any in
/// `Scene::unsupported`, and `sim-physics` lists it as not modelled).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tendon {
    /// The tendon's name. Unique within a scene when non-empty.
    pub name: String,
    /// The joints it couples and the coefficient of each: the tendon length is
    /// the sum of `coef * q` over them (`q` in radians or metres, as the joint
    /// is). Not empty.
    pub joints: Vec<TendonJoint>,
    /// Limits on the length, or `None` for unlimited (no unit conversion).
    #[serde(default)]
    pub range: Option<[f64; 2]>,
    /// Spring stiffness, >= 0.
    #[serde(default)]
    pub stiffness: f64,
    /// Damping, >= 0.
    #[serde(default)]
    pub damping: f64,
    /// Armature, >= 0.
    #[serde(default)]
    pub armature: f64,
    /// Dry friction magnitude, >= 0.
    #[serde(default)]
    pub frictionloss: f64,
    /// Reference of the length-limit constraint (MuJoCo's `solreflimit`); see
    /// `Joint::solref_limit`. Default [`crate::DEFAULT_SOLREF`].
    #[serde(default = "default_solref")]
    pub solref_limit: [f64; 2],
    /// Impedance of the length-limit constraint (MuJoCo's `solimplimit`).
    /// Default [`crate::DEFAULT_SOLIMP`].
    #[serde(default = "default_solimp")]
    pub solimp_limit: [f64; 5],
    /// Reference of the friction-loss constraint (MuJoCo's `solreffriction`).
    /// Default [`crate::DEFAULT_SOLREF`].
    #[serde(default = "default_solref")]
    pub solref_friction: [f64; 2],
    /// Impedance of the friction-loss constraint (MuJoCo's `solimpfriction`).
    /// Default [`crate::DEFAULT_SOLIMP`].
    #[serde(default = "default_solimp")]
    pub solimp_friction: [f64; 5],
    /// The distance from a length limit (the tendon's own unit) below which the
    /// limit constraint is active (MuJoCo's `margin`), >= 0. Default 0.
    #[serde(default)]
    pub margin: f64,
}

/// One joint's term in a fixed tendon.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TendonJoint {
    /// The joint (a hinge or a slide).
    pub joint: JointId,
    /// Its coefficient, finite.
    pub coef: f64,
}

/// A pair of bodies whose geoms are never tested against each other (MuJoCo's
/// `<contact><exclude body1 body2>`). `None` is the world body (MuJoCo's name `world`).
/// The two sides must differ.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContactExclude {
    /// The exclusion's name (MuJoCo's `name`); empty when it has none.
    #[serde(default)]
    pub name: String,
    /// The first body, or `None` for the world.
    pub body1: Option<BodyId>,
    /// The second body, or `None` for the world.
    pub body2: Option<BodyId>,
}

/// Something in a source document that the scene does not represent, or
/// represents without physics using it yet. Never dropped silently.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Unsupported {
    /// XML path of the element that carried it (`default/default[body]/geom`).
    pub path: String,
    /// The element (`element`) or the attribute (`@name`) that is not modelled.
    pub item: String,
    /// 1-based line in the source document, or 0 when unknown.
    pub line: u32,
    /// Why it is listed, in a few words.
    pub reason: String,
}
