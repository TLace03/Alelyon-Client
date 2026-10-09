//! Bodies, joints, geoms and meshes: the articulated rigid-body part of a scene.
//!
//! Invariants (checked by `Scene::validate`):
//! - SI units: metres, kilograms, radians, seconds. Right-handed, Z up.
//!   Quaternions are `[x, y, z, w]` (scalar last), Hamilton, unit length.
//! - The world is not a body. A body whose `parent` is `None` is a child of the
//!   world. Bodies are listed so that a parent comes before its children.
//! - A body's pose (`pos`, `quat`) is relative to its parent's frame (or the
//!   world frame) with every joint at its reference value.
//! - Joints are listed in non-decreasing body order, and a body's joints in
//!   the order they are applied. That order fixes the joint-coordinate layout
//!   (`sim_world::WorldLayout`), and it is MuJoCo's.
//! - A joint's `pos` and axis are in the body's frame. An angular joint's range
//!   is in radians and a `Slide`'s in metres; `None` means unlimited.
//! - A joint carries the soft-constraint parameters of its limit and of its
//!   friction loss in MuJoCo's form (`solref`, `solimp`, `margin`), with MuJoCo's
//!   defaults; the physics reads them, nothing else does.
//! - Inertia is stored the way MuJoCo does: a mass, the centre of mass in the
//!   body frame, the principal moments about it and the rotation of the
//!   principal axes (the inertial frame) in the body frame.
//! - A geom with `body == None` belongs to the world (static). A plane geom may
//!   only belong to the world or to a body with no joint in its chain.
//! - A mesh's vertices are in the mesh's own frame (after the importer's
//!   scale), which is the frame of the geom (or instance) that places it: the
//!   body frame when that placement is the identity.

use serde::{Deserialize, Serialize};

use crate::ids::{BodyId, MaterialId, MeshId};

fn identity_quat() -> [f64; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

fn one() -> u32 {
    1
}

fn default_condim() -> u32 {
    3
}

fn default_friction() -> [f64; 3] {
    [1.0, 0.005, 0.0001]
}

fn default_density() -> f64 {
    1000.0
}

fn one_f64() -> f64 {
    1.0
}

/// A rigid body.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Body {
    /// The body's name. Unique within a scene when non-empty.
    pub name: String,
    /// The parent body, or `None` for a child of the world.
    pub parent: Option<BodyId>,
    /// Position in the parent frame, metres.
    #[serde(default)]
    pub pos: [f64; 3],
    /// Orientation in the parent frame, unit `[x, y, z, w]`.
    #[serde(default = "identity_quat")]
    pub quat: [f64; 4],
    /// Mass properties, or `None` for a massless body (a kinematic frame).
    #[serde(default)]
    pub inertial: Option<Inertial>,
}

/// The mass properties of a body.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Inertial {
    /// Mass, kilograms, >= 0.
    pub mass_kg: f64,
    /// Centre of mass in the body frame, metres.
    pub com: [f64; 3],
    /// Principal moments of inertia about the centre of mass, kg m^2, each
    /// >= 0 and satisfying the triangle inequality `a + b >= c`.
    pub diag_inertia: [f64; 3],
    /// Rotation of the principal axes in the body frame, unit `[x, y, z, w]`.
    pub inertia_quat: [f64; 4],
}

/// A joint between a body and its parent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Joint {
    /// The joint's name. Unique within a scene when non-empty.
    pub name: String,
    /// The body the joint moves.
    pub body: BodyId,
    /// The kind of motion.
    pub kind: JointKind,
    /// The anchor in the body frame, metres (zero for a free joint).
    #[serde(default)]
    pub pos: [f64; 3],
    /// Limits: radians for hinge and ball joints (a ball joint's range is
    /// `[0, max_angle]`), metres for a slide. `None` is unlimited.
    #[serde(default)]
    pub range: Option<[f64; 2]>,
    /// Spring stiffness toward the reference position, N/m or N m/rad, >= 0.
    #[serde(default)]
    pub stiffness: f64,
    /// Viscous damping, N s/m or N m s/rad, >= 0.
    #[serde(default)]
    pub damping: f64,
    /// Rotor inertia added to the joint's degree(s) of freedom, kg or kg m^2, >= 0.
    #[serde(default)]
    pub armature: f64,
    /// Dry friction force or torque magnitude, N or N m, >= 0.
    #[serde(default)]
    pub frictionloss: f64,
    /// Reference of the joint-limit constraint (MuJoCo's `solreflimit`): the
    /// time constant (s) and damping ratio of the soft limit ("standard"
    /// format, both positive) or negative stiffness and damping ("direct"
    /// format, both not positive); a mix is refused. Default [`DEFAULT_SOLREF`].
    #[serde(default = "default_solref")]
    pub solref_limit: [f64; 2],
    /// Impedance of the joint-limit constraint (MuJoCo's `solimplimit`):
    /// `[d_min, d_max, width, midpoint, power]`. Default [`DEFAULT_SOLIMP`].
    #[serde(default = "default_solimp")]
    pub solimp_limit: [f64; 5],
    /// Reference of the friction-loss constraint (MuJoCo's `solreffriction`).
    /// Default [`DEFAULT_SOLREF`].
    #[serde(default = "default_solref")]
    pub solref_friction: [f64; 2],
    /// Impedance of the friction-loss constraint (MuJoCo's `solimpfriction`).
    /// Default [`DEFAULT_SOLIMP`].
    #[serde(default = "default_solimp")]
    pub solimp_friction: [f64; 5],
    /// The distance from a limit (radians for an angular joint, metres for a
    /// slide; MuJoCo does not convert it from degrees) below which the limit
    /// constraint is active, >= 0. Default 0.
    #[serde(default)]
    pub margin: f64,
}

/// MuJoCo's default `solref` (`mj_defaultSolRefImp`): time constant 0.02 s,
/// damping ratio 1.
pub const DEFAULT_SOLREF: [f64; 2] = [0.02, 1.0];

/// MuJoCo's default `solimp` (`mj_defaultSolRefImp`): `[d_min, d_max, width,
/// midpoint, power] = [0.9, 0.95, 0.001, 0.5, 2]`.
pub const DEFAULT_SOLIMP: [f64; 5] = [0.9, 0.95, 0.001, 0.5, 2.0];

pub(crate) fn default_solref() -> [f64; 2] {
    DEFAULT_SOLREF
}

pub(crate) fn default_solimp() -> [f64; 5] {
    DEFAULT_SOLIMP
}

/// The kind of motion a joint allows.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum JointKind {
    /// Six degrees of freedom. Only on a child of the world, and alone.
    Free,
    /// Three rotational degrees of freedom about the anchor.
    Ball,
    /// One rotation about `axis` through the anchor.
    Hinge {
        /// Unit axis in the body frame.
        axis: [f64; 3],
    },
    /// One translation along `axis`.
    Slide {
        /// Unit axis in the body frame.
        axis: [f64; 3],
    },
}

impl JointKind {
    /// Entries in the position vector: 7 for free (position and quaternion),
    /// 4 for ball (quaternion), 1 for hinge and slide (MuJoCo's `nq`).
    pub fn nq(&self) -> usize {
        match self {
            JointKind::Free => 7,
            JointKind::Ball => 4,
            JointKind::Hinge { .. } | JointKind::Slide { .. } => 1,
        }
    }

    /// Degrees of freedom: 6, 3, 1, 1 (MuJoCo's `nv`).
    pub fn nv(&self) -> usize {
        match self {
            JointKind::Free => 6,
            JointKind::Ball => 3,
            JointKind::Hinge { .. } | JointKind::Slide { .. } => 1,
        }
    }
}

/// A collision and visual primitive attached to a body (or the world).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Geom {
    /// The geom's name. Unique within a scene when non-empty.
    pub name: String,
    /// The body it is attached to, or `None` for the world.
    pub body: Option<BodyId>,
    /// The shape and its size.
    pub shape: Shape,
    /// Position in the body frame, metres.
    #[serde(default)]
    pub pos: [f64; 3],
    /// Orientation in the body frame, unit `[x, y, z, w]`.
    #[serde(default = "identity_quat")]
    pub quat: [f64; 4],
    /// The material it is made of.
    pub material: MaterialId,
    /// Collision type bitmask (MuJoCo's `contype`): geom 1 may collide with
    /// geom 2 when `contype1 & conaffinity2 != 0` or the reverse. Default 1.
    #[serde(default = "one")]
    pub contype: u32,
    /// Collision affinity bitmask (MuJoCo's `conaffinity`). Default 1.
    #[serde(default = "one")]
    pub conaffinity: u32,
    /// Contact dimensionality: 1 frictionless, 3 regular, 4 torsional, 6 rolling.
    /// Default 3.
    #[serde(default = "default_condim")]
    pub condim: u32,
    /// Sliding, torsional and rolling friction coefficients, each >= 0.
    /// Default `[1, 0.005, 0.0001]`.
    #[serde(default = "default_friction")]
    pub friction: [f64; 3],
    /// Mass density, kg/m^3, >= 0. Default 1000. Physics reads this, not the
    /// material's reference density.
    #[serde(default = "default_density")]
    pub density: f64,
    /// Reference of the contact constraint (MuJoCo's geom `solref`): the time
    /// constant (s) and damping ratio of the soft contact ("standard" format, both
    /// positive) or negative stiffness and damping ("direct" format, both not
    /// positive); a mix is refused. The two geoms of a contact are combined by
    /// MuJoCo's rule (`solmix`, `priority`). Default [`DEFAULT_SOLREF`].
    #[serde(default = "default_solref")]
    pub solref: [f64; 2],
    /// Impedance of the contact constraint (MuJoCo's geom `solimp`): `[d_min, d_max,
    /// width, midpoint, power]`. Default [`DEFAULT_SOLIMP`].
    #[serde(default = "default_solimp")]
    pub solimp: [f64; 5],
    /// The weight of this geom in the mix of its contact parameters with the other
    /// geom's (MuJoCo's `solmix`), finite and >= 0. Default 1.
    #[serde(default = "one_f64")]
    pub solmix: f64,
    /// The geom with the higher priority supplies the whole contact parameter set
    /// (MuJoCo's `priority`); equal priorities are mixed. Any integer. Default 0.
    #[serde(default)]
    pub priority: i32,
    /// The distance below which a contact with this geom is generated and is
    /// active as a constraint (MuJoCo's `margin`), metres, finite and >= 0 (MuJoCo's
    /// acceptance of a negative margin is not measured, so the scene refuses it).
    /// Default 0.
    #[serde(default)]
    pub margin: f64,
    /// A contact closer than `margin + gap` is generated, but it is a constraint only
    /// once its distance is below `margin` (MuJoCo's `gap`), metres, finite and
    /// >= 0. Default 0.
    #[serde(default)]
    pub gap: f64,
}

/// A geom's shape and its size, in metres.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Shape {
    /// A sphere of radius `r`.
    Sphere {
        /// Radius.
        r: f64,
    },
    /// A capsule along its local Z axis: a cylinder of half-length `half_len`
    /// capped by two hemispheres of radius `r`.
    Capsule {
        /// Radius.
        r: f64,
        /// Half the length of the cylindrical part.
        half_len: f64,
    },
    /// A box with half-extents `half` along its local axes.
    Box {
        /// Half-extents x, y, z.
        half: [f64; 3],
    },
    /// An infinite or finite plane through the geom's origin, normal along its
    /// local Z axis. `size` is MuJoCo's: half-extents along x and y for
    /// rendering (0 means infinite) and the render grid spacing, which must be
    /// positive.
    Plane {
        /// `[half_x, half_y, grid_spacing]`.
        size: [f64; 3],
    },
    /// A cylinder along its local Z axis.
    Cylinder {
        /// Radius.
        r: f64,
        /// Half the cylinder's length.
        half_len: f64,
    },
    /// An ellipsoid with semi-axes `radii` along its local axes.
    Ellipsoid {
        /// Semi-axes x, y, z.
        radii: [f64; 3],
    },
    /// A triangle mesh from `Scene::meshes`, placed by the geom's pose.
    Mesh {
        /// The mesh.
        mesh: MeshId,
    },
}

/// A triangle mesh.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mesh {
    /// The mesh's name. Unique within a scene when non-empty.
    pub name: String,
    /// Vertex positions in the mesh's own frame, metres.
    pub vertices: Vec<[f32; 3]>,
    /// Triangles as vertex indices, counter-clockwise seen from outside.
    pub triangles: Vec<[u32; 3]>,
}
