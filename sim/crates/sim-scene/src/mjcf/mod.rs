//! MuJoCo MJCF import: a subset, ported from MuJoCo's own compiler.
//!
//! [`load`] reads an MJCF document into a [`Scene`]. The rules of the format
//! (defaults and classes, orientation alternatives, `fromto`, inertia from
//! geoms, limits, units) are ported from MuJoCo's source at commit a8373cc4e
//! (`src/user/`, `src/xml/`, `plugin/stl_decoder/`; Apache-2.0, (c) DeepMind
//! Technologies Limited); each module header cites the file and lines it ports, and
//! the crate's `NOTICE` lists them. The model statistic and the camera frustum port
//! MuJoCo 3.14.0's engine (`statistic.rs`, `compile.rs`). The parity tests compare
//! the compiled `model/humanoid/humanoid.xml` and our own fixtures with MuJoCo
//! 3.14.0's own compile of them.
//!
//! # What is read
//!
//! | Element | Read |
//! |---|---|
//! | `<mujoco model>` | the scene name |
//! | `<compiler>` | `angle`, `eulerseq`, `inertiafromgeom`, `autolimits`, `inertiagrouprange`, `meshdir`, `assetdir`, `coordinate="local"` |
//! | `<option>` | `timestep`, `gravity`, `integrator` (`Euler` or `RK4`; `implicit`, `implicitfast` and `discrete` are recorded and the scene keeps Euler), and the solver options `solver` (`PGS`, `CG`, `Newton`), `iterations`, `tolerance`, `ls_iterations`, `ls_tolerance`, `cone`, `impratio` into `Scene::options` |
//! | `<default>` | nested classes with MuJoCo's inheritance; `geom`, `joint`, `motor`, `position`, `tendon`, `mesh`, `material` |
//! | `<asset>` | `mesh` (binary STL: `file`, `name`, `scale`, `inertia` legacy or exact), `material` (`rgba`, `metallic`, `roughness`, `emission`) |
//! | `<worldbody>`, `<body>` | `name`, `pos`, `quat`/`axisangle`/`xyaxes`/`zaxis`/`euler`, `childclass`; `<inertial>` with `diaginertia` or `fullinertia` |
//! | `<joint>`, `<freejoint>` | `type`, `pos`, `axis`, `limited`, `range`, `stiffness`, `damping`, `armature`, `frictionloss`, `solreflimit`, `solimplimit`, `solreffriction`, `solimpfriction`, `margin` |
//! | `<geom>` | `type`, `size`, `fromto`, `pos`, orientation, `mass`, `density`, `friction`, `condim`, `contype`, `conaffinity`, `group`, `rgba`, `material`, `mesh`, and the contact parameters `solref`, `solimp`, `solmix`, `priority`, `margin`, `gap` |
//! | `<contact>` | `<exclude name body1 body2>` into `Scene::contact_excludes` (`world` is the world body); `<pair>` is recorded |
//! | `<tendon><fixed>` | `limited`, `range`, `stiffness`, `damping`, `armature`, `frictionloss`, `solreflimit`, `solimplimit`, `solreffriction`, `solimpfriction`, `margin`, `<joint joint coef>` |
//! | `<actuator>` | `motor` and `position`: `joint`, `gear`, `ctrlrange`, `ctrllimited`, `kp` |
//! | `<camera>` (fixed, perspective) | `pos`, orientation, `fovy`, `resolution`, `sensorsize`, `focal`, `focalpixel`, `principal`, `principalpixel`, `mode`, `projection`, `target`, `class`, into `Scene::cameras` (see below) |
//! | `<site>` | `type`, `size`, `fromto`, `pos`, orientation, `material`: only its position is used, for the model extent; the site itself is recorded |
//! | `<visual>` | `<map znear zfar>` (the clip planes, as fractions of the model extent) and `<global offwidth offheight>` (the image of a camera without a resolution) |
//! | `<statistic>` | `extent`, `meansize`, `center`: they override the computed model statistic, as in MuJoCo |
//!
//! # Cameras
//!
//! A fixed perspective camera becomes a `Scene::cameras` record that draws what
//! MuJoCo's renderer draws (the derivation is in `compile.rs`): the pose is MuJoCo's
//! turned by half a turn about the camera's x axis into the scene's OpenCV frame; the
//! image is `resolution` when it is larger than 1 x 1, else the offscreen buffer
//! (`offwidth x offheight`, default 640 x 480); `fx = fy = H / (2 tan(fovy / 2))` for a
//! `fovy` camera, or the renderer's projection of the sensor for a `sensorsize`
//! camera; `near = znear * extent` and `far = zfar * extent` in `f32`, with the model
//! extent `setStat` computes (`statistic.rs`; [`load_with_statistic`] returns it).
//! MuJoCo's camera checks are kept (`fovy` below 180, `focal` and `principal` need
//! `sensorsize`, which needs a positive `resolution`, `fovy` and `sensorsize` not on
//! one element). A camera MuJoCo compiles but no pinhole can draw (`sensorsize`
//! without a focal length, a zero `fovy`, a non-positive image or clip range) is
//! refused. A tracking camera (`mode` other than `fixed`) and an orthographic one are
//! recorded.
//!
//! # What is recorded, and what is refused
//!
//! Nothing is dropped silently. Each thing the document carries is in exactly one
//! of three classes:
//!
//! - **Read**: it becomes part of the scene (the table above).
//! - **Recorded**: a closed, named list of MuJoCo elements and attributes that
//!   the simulator does not model yet and that do not change which bodies,
//!   joints, geoms and masses exist: rendering parameters (`<visual>` settings
//!   other than the clip planes and the offscreen size, `<statistic meanmass>`,
//!   lights, tracking and orthographic cameras, a camera's `ipd`, `output` and
//!   `user`, textures), the solver parameters the physics
//!   does not use (`jacobian`, `noslip_*`, `ccd_*`, `sdf_*`, the `o_*` contact
//!   overrides, which need the refused `<flag>`), explicit contact pairs, the
//!   `implicit`, `implicitfast` and `discrete` integrators, sites,
//!   keyframes, visualisation groups, user data. Each
//!   occurrence is appended to `Scene::unsupported` with its XML path, line and
//!   reason. Under [`LoadOptions::strict`] each one is an error instead.
//!   A fixed tendon is listed there only when it has a `stiffness`, `damping` or
//!   `armature`, which the scene carries and the physics does not yet simulate
//!   (its limit and friction loss are read and simulated).
//! - **Refused**: everything else, with `MjcfErrorKind::UnsupportedElement` or
//!   `UnsupportedAttribute` and the element's path and line. That includes what
//!   would change the mechanism or its dynamics if it were ignored: equality
//!   constraints, sensors, flexes, `<frame>`, `<replicate>`,
//!   `<attach>`, `<composite>`, `<flexcomp>`, `mocap`, `gravcomp`, `ref`,
//!   `springref`, `springdamper`, actuator force and length limits, other
//!   actuator kinds, spatial tendons, height fields, SDF geoms, primitives
//!   fitted to meshes, mesh sites, `<statistic meaninertia>` (MuJoCo's solvers
//!   scale their tolerance by it), `shellinertia`, `balanceinertia`, `boundmass`,
//!   `boundinertia`, `settotalmass`, `fusestatic`, `<flag>`, wind, fluid
//!   density and viscosity, mesh `refpos`/`refquat`, and `<include>`.
//!
//! # Deviations from MuJoCo's compile
//!
//! - A mesh keeps its own frame: MuJoCo recentres and rotates the stored mesh
//!   into its principal inertial frame and folds the offset into the geom pose;
//!   here the mesh and the geom pose are kept as authored and the offset is used
//!   only when the body's inertia is summed. The geometry in the world is the same.
//! - The importer adds one `Instance` per geom with segmentation id `index + 1`
//!   (disable with [`LoadOptions::instances`]), and one `Material` per distinct
//!   colour, with MuJoCo's `rgba` read as sRGB and converted to linear light.
//! - A plane's size is MuJoCo's three numbers, kept as `Shape::Plane { size }`.

mod compile;
mod import;
mod inertia;
mod mesh;
mod mjmath;
mod specs;
mod statistic;
mod xmlutil;

use std::path::Path;

use crate::error::Result;
use crate::scene::Scene;

pub use statistic::Statistic;

/// How to import.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadOptions {
    /// Refuse (with `MjcfErrorKind::Strict`) every item the importer would only
    /// record in `Scene::unsupported`. Default `false`.
    pub strict: bool,
    /// Create one `Instance` per geom so the renderer has something to draw.
    /// Default `true`.
    pub instances: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            strict: false,
            instances: true,
        }
    }
}

/// Imports MJCF text into a scene, with default options.
///
/// `asset_dir` is the directory mesh files are read from (the MuJoCo model
/// file's directory, plus the compiler's `meshdir`); nothing outside it is read.
/// The scene is validated before it is returned.
pub fn load(xml: &str, asset_dir: impl AsRef<Path>) -> Result<Scene> {
    load_with(xml, asset_dir, &LoadOptions::default())
}

/// The stack the importer runs on.
///
/// Reading recurses once per element level, up to the importer's nesting limit
/// (256 levels). A debug build spends about 10 KB of stack per level, more than a
/// Windows main thread's 1 MiB holds. The import therefore runs on a thread of
/// its own with this stack, so its limits hold whatever thread calls it. Only the
/// pages it touches are committed.
const IMPORT_STACK_BYTES: usize = 32 * 1024 * 1024;

/// [`load`] with explicit options.
///
/// Runs on a thread of its own with a fixed stack (see `IMPORT_STACK_BYTES`), and
/// falls back to the caller's thread only if the operating system refuses a new
/// thread. A panic inside the import is resumed on the caller's thread.
pub fn load_with(xml: &str, asset_dir: impl AsRef<Path>, options: &LoadOptions) -> Result<Scene> {
    load_with_statistic(xml, asset_dir, options).map(|(scene, _)| scene)
}

/// [`load_with`], and the model statistic MuJoCo's compiler derives for the same
/// document: the centre of the model's bounding box in its reference pose, its extent
/// and its mean body size (`setStat`; see [`Statistic`]), with the document's
/// `<statistic extent meansize center>` applied over the computed values, as in
/// MuJoCo's compiled model. The extent is what scales the cameras' clip planes.
pub fn load_with_statistic(
    xml: &str,
    asset_dir: impl AsRef<Path>,
    options: &LoadOptions,
) -> Result<(Scene, Statistic)> {
    let asset_dir = asset_dir.as_ref();
    let run = || -> Result<(Scene, Statistic)> {
        let imported = import::read(xml, asset_dir, options)?;
        compile::compile(imported, options)
    };
    std::thread::scope(|scope| {
        match std::thread::Builder::new()
            .name("mjcf-import".into())
            .stack_size(IMPORT_STACK_BYTES)
            .spawn_scoped(scope, run)
        {
            Ok(worker) => worker
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
            Err(_) => run(),
        }
    })
}
