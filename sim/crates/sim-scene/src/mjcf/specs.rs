//! The attribute state of each MJCF element kind, the default classes that carry
//! it, and the readers that fill it from an element.
//!
//! This is MuJoCo's `mjs*` spec structs and its default-class mechanism
//! (`mjXReader::Default`, xml_native_reader.cc:1732-1831; `mjXReader::Body`
//! class resolution, 2259-2606), reduced to the attributes the importer models.
//!
//! Invariants:
//! - A class is a full copy of its parent at the moment it is created, then
//!   overridden by its own elements (the parent's elements are all read before
//!   any nested class is created, so a nested class sees its parent's final
//!   state). The top-level class is `main`; several top-level `<default>`
//!   sections accumulate into it.
//! - An element starts as a copy of its class's spec and then reads its own
//!   attributes over it. A value given to `quat` does not change the orientation
//!   *kind* of a spec (only the alternatives `axisangle`, `xyaxes`, `zaxis`,
//!   `euler` do), exactly as in MuJoCo's `ReadAlternative` (xml_base.cc:52-75);
//!   a class that set an alternative and an element that sets `quat` therefore
//!   gets the class's alternative, as in MuJoCo.
//! - The attribute policy of each element is one table: *supported*, *recorded*
//!   or refused (see `xmlutil.rs`). Attributes that MuJoCo does not allow in a
//!   `<default>` (`name`, `class`, the transmission target) are refused there.
//! - Joint `stiffness` and `damping` take MuJoCo's up to three polynomial
//!   coefficients, but only the constant term can be modelled: higher terms must
//!   be zero or the element is refused.
//! - The soft-constraint attributes (`solreflimit`, `solimplimit`, `solreffriction`,
//!   `solimpfriction`, `margin`) of a joint and a fixed tendon are read as MuJoCo's
//!   reader does (`ReadAttr` with a length and no minimum): fewer numbers than the
//!   attribute holds override the leading entries and keep the class's others, more
//!   numbers are refused. `margin` is not converted from degrees.
//! - The contact attributes of a geom (`solref`, `solimp`, `solmix`, `priority`,
//!   `margin`, `gap`) are read the same way (phase 1c-ii); `solref` and `solimp` go
//!   through [`read_partial`] with the class's values as the starting point.
//! - A camera's `focal`, `focalpixel`, `principal`, `principalpixel` and
//!   `sensorsize` are MuJoCo `float` attributes and are parsed straight to `f32`
//!   (`E::floats`), as MuJoCo's reader does; `fovy` is a `double`. The schema's
//!   `exclusive fovy sensorsize` (mjcf.schema:1111) is checked per element, as
//!   MuJoCo's schema validator does: a class may set `fovy` and an element
//!   `sensorsize`.
//! - A site is read for its frame only (`type`, `size`, `fromto`, `pos` and the
//!   orientation), because the model statistic MuJoCo derives the clip planes
//!   from includes every site's position (`setStat`, engine_setconst.c); the site
//!   itself is recorded, not modelled.

use crate::error::{MjcfError, MjcfErrorKind};

use super::mesh::MeshInertia;
use super::mjmath::{Alt, OrientKind};
use super::xmlutil::{Ctx, E};

/// MuJoCo's default geom colour (`mjs_defaultGeom`, schema `rgba`).
pub(crate) const DEFAULT_RGBA: [f32; 4] = [0.5, 0.5, 0.5, 1.0];

/// A geom type the importer supports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GeomType {
    Plane,
    Sphere,
    Capsule,
    Ellipsoid,
    Cylinder,
    Box,
    Mesh,
}

/// MuJoCo's tri-state `limited` (`FalseTrueAuto`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Limited {
    False,
    True,
    Auto,
}

const LIMITED_KEYWORDS: [(&str, Limited); 3] = [
    ("false", Limited::False),
    ("true", Limited::True),
    ("auto", Limited::Auto),
];

/// A geom's attribute state (`mjsGeom`).
#[derive(Clone, Debug)]
pub(crate) struct GeomSpec {
    pub ty: GeomType,
    pub contype: i32,
    pub conaffinity: i32,
    pub condim: i32,
    pub group: i32,
    pub size: [f64; 3],
    pub material: Option<String>,
    pub friction: [f64; 3],
    /// NaN when undefined (MuJoCo's `mjNAN`).
    pub mass: f64,
    pub density: f64,
    /// NaN in the first entry when undefined.
    pub fromto: [f64; 6],
    pub pos: [f64; 3],
    pub quat: [f64; 4],
    pub alt: Alt,
    pub mesh: Option<String>,
    pub rgba: [f32; 4],
    pub solref: [f64; 2],
    pub solimp: [f64; 5],
    pub solmix: f64,
    pub priority: i32,
    pub margin: f64,
    pub gap: f64,
}

impl Default for GeomSpec {
    /// `mjs_defaultGeom` and the schema defaults (mjcf.schema:1035-1068).
    fn default() -> Self {
        GeomSpec {
            ty: GeomType::Sphere,
            contype: 1,
            conaffinity: 1,
            condim: 3,
            group: 0,
            size: [0.0; 3],
            material: None,
            friction: [1.0, 0.005, 0.0001],
            mass: f64::NAN,
            density: 1000.0,
            fromto: [f64::NAN; 6],
            pos: [0.0; 3],
            quat: [1.0, 0.0, 0.0, 0.0],
            alt: Alt::default(),
            mesh: None,
            rgba: DEFAULT_RGBA,
            solref: crate::DEFAULT_SOLREF,
            solimp: crate::DEFAULT_SOLIMP,
            solmix: 1.0,
            priority: 0,
            margin: 0.0,
            gap: 0.0,
        }
    }
}

/// The joint types the importer supports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JointType {
    Free,
    Ball,
    Slide,
    Hinge,
}

/// A joint's attribute state (`mjsJoint`).
#[derive(Clone, Debug)]
pub(crate) struct JointSpec {
    pub ty: JointType,
    pub pos: [f64; 3],
    pub axis: [f64; 3],
    pub limited: Limited,
    pub range: [f64; 2],
    pub stiffness: f64,
    pub damping: f64,
    pub armature: f64,
    pub frictionloss: f64,
    pub solref_limit: [f64; 2],
    pub solimp_limit: [f64; 5],
    pub solref_friction: [f64; 2],
    pub solimp_friction: [f64; 5],
    pub margin: f64,
}

impl Default for JointSpec {
    /// `mjs_defaultJoint` and the schema defaults (mjcf.schema:1002-1027).
    fn default() -> Self {
        JointSpec {
            ty: JointType::Hinge,
            pos: [0.0; 3],
            axis: [0.0, 0.0, 1.0],
            limited: Limited::Auto,
            range: [0.0; 2],
            stiffness: 0.0,
            damping: 0.0,
            armature: 0.0,
            frictionloss: 0.0,
            solref_limit: crate::DEFAULT_SOLREF,
            solimp_limit: crate::DEFAULT_SOLIMP,
            solref_friction: crate::DEFAULT_SOLREF,
            solimp_friction: crate::DEFAULT_SOLIMP,
            margin: 0.0,
        }
    }
}

/// An actuator's attribute state (`mjsActuator`), shared by every actuator tag
/// in a class as in MuJoCo.
#[derive(Clone, Debug)]
pub(crate) struct ActuatorSpec {
    pub ctrlrange: [f64; 2],
    pub ctrllimited: Limited,
    pub gear: [f64; 6],
    /// `gainprm[0]` (`mjs_setToMotor` sets it to 1, `mjs_setToPosition` to `kp`).
    pub kp: f64,
}

impl Default for ActuatorSpec {
    fn default() -> Self {
        ActuatorSpec {
            ctrlrange: [0.0; 2],
            ctrllimited: Limited::Auto,
            gear: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            kp: 1.0,
        }
    }
}

/// A fixed tendon's attribute state (`mjsTendon`, the rows `fixed` reads).
#[derive(Clone, Debug)]
pub(crate) struct TendonSpec {
    pub limited: Limited,
    pub range: [f64; 2],
    pub stiffness: f64,
    pub damping: f64,
    pub armature: f64,
    pub frictionloss: f64,
    pub solref_limit: [f64; 2],
    pub solimp_limit: [f64; 5],
    pub solref_friction: [f64; 2],
    pub solimp_friction: [f64; 5],
    pub margin: f64,
}

impl Default for TendonSpec {
    fn default() -> Self {
        TendonSpec {
            limited: Limited::Auto,
            range: [0.0; 2],
            stiffness: 0.0,
            damping: 0.0,
            armature: 0.0,
            frictionloss: 0.0,
            solref_limit: crate::DEFAULT_SOLREF,
            solimp_limit: crate::DEFAULT_SOLIMP,
            solref_friction: crate::DEFAULT_SOLREF,
            solimp_friction: crate::DEFAULT_SOLIMP,
            margin: 0.0,
        }
    }
}

/// A mesh asset's attribute state (`mjsMesh`, the rows the importer reads).
#[derive(Clone, Debug)]
pub(crate) struct MeshSpec {
    pub scale: [f64; 3],
    pub inertia: MeshInertia,
}

impl Default for MeshSpec {
    fn default() -> Self {
        MeshSpec {
            scale: [1.0; 3],
            inertia: MeshInertia::Legacy,
        }
    }
}

/// A material asset's attribute state (`mjsMaterial`, the rows the importer
/// reads). `metallic` and `roughness` are -1 when unspecified (schema).
#[derive(Clone, Debug)]
pub(crate) struct MaterialSpec {
    pub rgba: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emission: f32,
}

impl Default for MaterialSpec {
    fn default() -> Self {
        MaterialSpec {
            rgba: [1.0; 4],
            metallic: -1.0,
            roughness: -1.0,
            emission: 0.0,
        }
    }
}

/// How a camera moves (MuJoCo's `mjtCamLight`, keyword map `camlight_map`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CameraMode {
    Fixed,
    Track,
    TrackCom,
    TargetBody,
    TargetBodyCom,
}

const CAMERA_MODE_KEYWORDS: [(&str, CameraMode); 5] = [
    ("fixed", CameraMode::Fixed),
    ("track", CameraMode::Track),
    ("trackcom", CameraMode::TrackCom),
    ("targetbody", CameraMode::TargetBody),
    ("targetbodycom", CameraMode::TargetBodyCom),
];

impl CameraMode {
    /// The MJCF keyword.
    pub fn keyword(self) -> &'static str {
        CAMERA_MODE_KEYWORDS
            .iter()
            .find(|(_, m)| *m == self)
            .map_or("fixed", |(k, _)| *k)
    }
}

/// A camera's projection (MuJoCo's `mjtProjection`, keyword map `projection_map`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Projection {
    Perspective,
    Orthographic,
}

const PROJECTION_KEYWORDS: [(&str, Projection); 2] = [
    ("perspective", Projection::Perspective),
    ("orthographic", Projection::Orthographic),
];

/// A camera's attribute state (`mjsCamera`, the rows the importer reads).
#[derive(Clone, Debug)]
pub(crate) struct CameraSpec {
    pub mode: CameraMode,
    pub projection: Projection,
    /// The `target` body (not allowed in a `<default>`).
    pub target: Option<String>,
    /// Vertical field of view, degrees (MuJoCo's `fovy` is always degrees).
    pub fovy: f64,
    pub resolution: [i32; 2],
    pub focal: [f32; 2],
    pub focalpixel: [f32; 2],
    pub principal: [f32; 2],
    pub principalpixel: [f32; 2],
    pub sensorsize: [f32; 2],
    pub pos: [f64; 3],
    pub quat: [f64; 4],
    pub alt: Alt,
}

impl Default for CameraSpec {
    /// `mjs_defaultCamera` (user_init.c:173-187): fixed, perspective, `fovy` 45,
    /// resolution 1 x 1, no intrinsics.
    fn default() -> Self {
        CameraSpec {
            mode: CameraMode::Fixed,
            projection: Projection::Perspective,
            target: None,
            fovy: 45.0,
            resolution: [1, 1],
            focal: [0.0; 2],
            focalpixel: [0.0; 2],
            principal: [0.0; 2],
            principalpixel: [0.0; 2],
            sensorsize: [0.0; 2],
            pos: [0.0; 3],
            quat: [1.0, 0.0, 0.0, 0.0],
            alt: Alt::default(),
        }
    }
}

/// A site's type: MuJoCo's geom types (keyword map `geomtype_map`); which of them a
/// site may have is decided when the site is compiled, as MuJoCo does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SiteType {
    Plane,
    Hfield,
    Sphere,
    Capsule,
    Ellipsoid,
    Cylinder,
    Box,
    Mesh,
    Sdf,
}

const SITE_TYPE_KEYWORDS: [(&str, SiteType); 9] = [
    ("plane", SiteType::Plane),
    ("hfield", SiteType::Hfield),
    ("sphere", SiteType::Sphere),
    ("capsule", SiteType::Capsule),
    ("ellipsoid", SiteType::Ellipsoid),
    ("cylinder", SiteType::Cylinder),
    ("box", SiteType::Box),
    ("mesh", SiteType::Mesh),
    ("sdf", SiteType::Sdf),
];

/// A site's attribute state (`mjsSite`, the rows that place it).
#[derive(Clone, Debug)]
pub(crate) struct SiteSpec {
    pub ty: SiteType,
    pub size: [f64; 3],
    /// NaN in the first entry when undefined.
    pub fromto: [f64; 6],
    pub pos: [f64; 3],
    pub quat: [f64; 4],
    pub alt: Alt,
    pub material: Option<String>,
}

impl Default for SiteSpec {
    /// `mjs_defaultSite` (user_init.c) and the schema defaults: a sphere of size
    /// 0.005 at the body origin.
    fn default() -> Self {
        SiteSpec {
            ty: SiteType::Sphere,
            size: [0.005; 3],
            fromto: [f64::NAN; 6],
            pos: [0.0; 3],
            quat: [1.0, 0.0, 0.0, 0.0],
            alt: Alt::default(),
            material: None,
        }
    }
}

/// A default class: one spec per element kind.
#[derive(Clone, Debug)]
pub(crate) struct Class {
    pub name: String,
    pub geom: GeomSpec,
    pub joint: JointSpec,
    pub actuator: ActuatorSpec,
    pub tendon: TendonSpec,
    pub mesh: MeshSpec,
    pub material: MaterialSpec,
    pub camera: CameraSpec,
    pub site: SiteSpec,
}

impl Class {
    pub fn new(name: &str) -> Self {
        Class {
            name: name.to_string(),
            geom: GeomSpec::default(),
            joint: JointSpec::default(),
            actuator: ActuatorSpec::default(),
            tendon: TendonSpec::default(),
            mesh: MeshSpec::default(),
            material: MaterialSpec::default(),
            camera: CameraSpec::default(),
            site: SiteSpec::default(),
        }
    }
}

/// Reads the orientation attributes of `e` over `quat` and `alt`
/// (`ReadQuat` + `ReadAlternative`).
pub(crate) fn read_orientation(e: &E, quat: &mut [f64; 4], alt: &mut Alt) -> Result<(), MjcfError> {
    let mut numspec = usize::from(e.has("quat"));
    if let Some(q) = e.exact::<4>("quat")? {
        if q == [0.0; 4] {
            return Err(e.bad("zero quaternion is not allowed"));
        }
        *quat = q;
    }
    if let Some(v) = e.exact::<4>("axisangle")? {
        numspec += 1;
        alt.axisangle = v;
        alt.kind = OrientKind::AxisAngle;
    }
    if let Some(v) = e.exact::<6>("xyaxes")? {
        numspec += 1;
        alt.xyaxes = v;
        alt.kind = OrientKind::XyAxes;
    }
    if let Some(v) = e.exact::<3>("zaxis")? {
        numspec += 1;
        alt.zaxis = v;
        alt.kind = OrientKind::ZAxis;
    }
    if let Some(v) = e.exact::<3>("euler")? {
        numspec += 1;
        alt.euler = v;
        alt.kind = OrientKind::Euler;
    }
    if numspec > 1 {
        return Err(e.bad("multiple orientation specifiers are not allowed"));
    }
    Ok(())
}

pub(crate) const ORIENTATION_ATTRS: [&str; 5] = ["quat", "axisangle", "xyaxes", "zaxis", "euler"];

/// A polynomial coefficient attribute (`stiffness`, `damping`): the constant term.
fn read_poly(e: &E, name: &str) -> Result<Option<f64>, MjcfError> {
    let Some(v) = e.nums(name, 1, 3)? else {
        return Ok(None);
    };
    if v[1..].iter().any(|&c| c != 0.0) {
        return Err(e.err(
            MjcfErrorKind::UnsupportedAttribute,
            format!("polynomial '{name}' (a non-zero higher-order coefficient) is not supported"),
        ));
    }
    Ok(Some(v[0]))
}

fn non_default(e: &E, in_default: bool, attrs: &[&'static str]) -> Result<(), MjcfError> {
    if in_default {
        for a in attrs {
            if e.has(a) {
                return Err(e.err(
                    MjcfErrorKind::UnsupportedAttribute,
                    format!(
                        "attribute '{a}' is not allowed in a <default> <{}>",
                        e.tag()
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Reads a `<geom>` (in a body, or in a `<default>`) over `spec`.
pub(crate) fn read_geom(
    ctx: &mut Ctx,
    e: &E,
    spec: &mut GeomSpec,
    in_default: bool,
) -> Result<(), MjcfError> {
    ctx.check_attrs(
        e,
        &[
            "name",
            "class",
            "type",
            "contype",
            "conaffinity",
            "condim",
            "group",
            "size",
            "material",
            "friction",
            "mass",
            "density",
            "fromto",
            "pos",
            "quat",
            "axisangle",
            "xyaxes",
            "zaxis",
            "euler",
            "mesh",
            "rgba",
            "priority",
            "solmix",
            "solref",
            "solimp",
            "margin",
            "gap",
        ],
        &[("user", "user data, not modelled")],
    )?;
    non_default(e, in_default, &["name", "class"])?;
    // Everything else MuJoCo allows on a geom (`shellinertia`, `surfacevel`,
    // `adhesion`, `hfield`, `fitscale`, `fluidshape`, `fluidcoef`) changes how the
    // geom is simulated or how its mass is derived, and was refused by
    // `check_attrs` above with the attribute's name.

    if let Some(text) = e.attr("type") {
        spec.ty = match text {
            "plane" => GeomType::Plane,
            "sphere" => GeomType::Sphere,
            "capsule" => GeomType::Capsule,
            "ellipsoid" => GeomType::Ellipsoid,
            "cylinder" => GeomType::Cylinder,
            "box" => GeomType::Box,
            "mesh" => GeomType::Mesh,
            "hfield" | "sdf" => {
                return Err(e.err(
                    MjcfErrorKind::UnsupportedAttribute,
                    format!("geom type '{text}' is not supported by this importer"),
                ));
            }
            other => return Err(e.bad(format!("invalid keyword '{other}' in attribute 'type'"))),
        };
    }
    if let Some(v) = e.int("contype")? {
        spec.contype = v;
    }
    if let Some(v) = e.int("conaffinity")? {
        spec.conaffinity = v;
    }
    if let Some(v) = e.int("condim")? {
        spec.condim = v;
    }
    if let Some(v) = e.int("group")? {
        spec.group = v;
        ctx.record(
            e,
            "@group",
            "visualisation group, not modelled; read here only to select geoms by inertiagrouprange",
        )?;
    }
    if let Some(v) = e.nums("size", 1, 3)? {
        spec.size[..v.len()].copy_from_slice(&v);
    }
    if let Some(m) = e.attr("material") {
        spec.material = Some(m.to_string());
    }
    if let Some(v) = e.nums("friction", 1, 3)? {
        spec.friction[..v.len()].copy_from_slice(&v);
    }
    if let Some(v) = e.num("mass")? {
        spec.mass = v;
    }
    if let Some(v) = e.num("density")? {
        spec.density = v;
    }
    if let Some(v) = e.exact::<6>("fromto")? {
        spec.fromto = v;
    }
    if let Some(v) = e.exact::<3>("pos")? {
        spec.pos = v;
    }
    if let Some(m) = e.attr("mesh") {
        spec.mesh = Some(m.to_string());
    }
    if let Some(v) = e.exact::<4>("rgba")? {
        spec.rgba = [v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32];
        if v[3] != 1.0 {
            ctx.record(
                e,
                "@rgba",
                "alpha (transparency) is not modelled; the optical material is opaque",
            )?;
        }
    }
    // the contact parameters, read as MuJoCo's reader does: `solref` and `solimp` of
    // fewer numbers than the attribute holds override the leading entries and keep the
    // class's others
    read_partial(e, "solref", &mut spec.solref)?;
    read_partial(e, "solimp", &mut spec.solimp)?;
    if let Some(v) = e.num("solmix")? {
        spec.solmix = v;
    }
    if let Some(v) = e.int("priority")? {
        spec.priority = v;
    }
    if let Some(v) = e.num("margin")? {
        spec.margin = v;
    }
    if let Some(v) = e.num("gap")? {
        spec.gap = v;
    }
    read_orientation(e, &mut spec.quat, &mut spec.alt)
}

/// Reads a `<joint>` over `spec`.
pub(crate) fn read_joint(
    ctx: &mut Ctx,
    e: &E,
    spec: &mut JointSpec,
    in_default: bool,
) -> Result<(), MjcfError> {
    ctx.check_attrs(
        e,
        &[
            "name",
            "class",
            "type",
            "pos",
            "axis",
            "limited",
            "range",
            "stiffness",
            "damping",
            "armature",
            "frictionloss",
            "solreflimit",
            "solimplimit",
            "solreffriction",
            "solimpfriction",
            "margin",
        ],
        &[
            ("group", "visualisation group, not modelled"),
            ("user", "user data, not modelled"),
        ],
    )?;
    non_default(e, in_default, &["name", "class"])?;
    if let Some(text) = e.attr("type") {
        spec.ty = match text {
            "free" => JointType::Free,
            "ball" => JointType::Ball,
            "slide" => JointType::Slide,
            "hinge" => JointType::Hinge,
            other => return Err(e.bad(format!("invalid keyword '{other}' in attribute 'type'"))),
        };
    }
    if let Some(v) = e.exact::<3>("pos")? {
        spec.pos = v;
    }
    if let Some(v) = e.exact::<3>("axis")? {
        spec.axis = v;
    }
    if let Some(v) = e.keyword("limited", &LIMITED_KEYWORDS)? {
        spec.limited = v;
    }
    if let Some(v) = e.exact::<2>("range")? {
        spec.range = v;
    }
    if let Some(v) = read_poly(e, "stiffness")? {
        spec.stiffness = v;
    }
    if let Some(v) = read_poly(e, "damping")? {
        spec.damping = v;
    }
    if let Some(v) = e.num("armature")? {
        spec.armature = v;
    }
    if let Some(v) = e.num("frictionloss")? {
        spec.frictionloss = v;
    }
    read_soft(
        e,
        &mut spec.solref_limit,
        &mut spec.solimp_limit,
        &mut spec.solref_friction,
        &mut spec.solimp_friction,
        &mut spec.margin,
    )
}

/// The soft-constraint attributes shared by a joint and a fixed tendon.
fn read_soft(
    e: &E,
    solref_limit: &mut [f64; 2],
    solimp_limit: &mut [f64; 5],
    solref_friction: &mut [f64; 2],
    solimp_friction: &mut [f64; 5],
    margin: &mut f64,
) -> Result<(), MjcfError> {
    read_partial(e, "solreflimit", solref_limit)?;
    read_partial(e, "solimplimit", solimp_limit)?;
    read_partial(e, "solreffriction", solref_friction)?;
    read_partial(e, "solimpfriction", solimp_friction)?;
    if let Some(v) = e.num("margin")? {
        *margin = v;
    }
    Ok(())
}

/// An attribute of up to `N` numbers that override the leading entries of `dst`
/// (MuJoCo's `ReadAttr` for `solref` and `solimp`: the entries that are not given
/// keep the value the class holds).
fn read_partial<const N: usize>(e: &E, name: &str, dst: &mut [f64; N]) -> Result<(), MjcfError> {
    if let Some(v) = e.nums(name, 1, N)? {
        dst[..v.len()].copy_from_slice(&v);
    }
    Ok(())
}

/// Which actuator tag is being read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ActuatorTag {
    Motor,
    Position,
}

/// Reads a `<motor>` or `<position>` over `spec`. Returns the `kp` of a
/// position actuator (the attribute, else the spec's).
pub(crate) fn read_actuator(
    ctx: &mut Ctx,
    e: &E,
    spec: &mut ActuatorSpec,
    tag: ActuatorTag,
    in_default: bool,
) -> Result<(), MjcfError> {
    let mut supported = vec!["name", "class", "joint", "gear", "ctrlrange", "ctrllimited"];
    if tag == ActuatorTag::Position {
        supported.push("kp");
    }
    ctx.check_attrs(
        e,
        &supported,
        &[
            ("group", "visualisation and disable group, not modelled"),
            ("user", "user data, not modelled"),
        ],
    )?;
    non_default(e, in_default, &["name", "class", "joint"])?;
    if let Some(v) = e.nums("gear", 1, 6)? {
        spec.gear[..v.len()].copy_from_slice(&v);
    }
    if let Some(v) = e.exact::<2>("ctrlrange")? {
        spec.ctrlrange = v;
    }
    if let Some(v) = e.keyword("ctrllimited", &LIMITED_KEYWORDS)? {
        spec.ctrllimited = v;
    }
    match tag {
        ActuatorTag::Motor => spec.kp = 1.0, // mjs_setToMotor: unit gain
        ActuatorTag::Position => {
            if let Some(v) = e.num("kp")? {
                spec.kp = v;
            }
        }
    }
    Ok(())
}

/// Reads a fixed `<tendon>` (`<fixed>`, or `<tendon>` in a `<default>`) over `spec`.
pub(crate) fn read_tendon(
    ctx: &mut Ctx,
    e: &E,
    spec: &mut TendonSpec,
    in_default: bool,
) -> Result<(), MjcfError> {
    ctx.check_attrs(
        e,
        &[
            "name",
            "class",
            "limited",
            "range",
            "stiffness",
            "damping",
            "armature",
            "frictionloss",
            "solreflimit",
            "solimplimit",
            "solreffriction",
            "solimpfriction",
            "margin",
        ],
        &[
            ("group", "visualisation group, not modelled"),
            ("user", "user data, not modelled"),
        ],
    )?;
    non_default(e, in_default, &["name", "class"])?;
    if let Some(v) = e.keyword("limited", &LIMITED_KEYWORDS)? {
        spec.limited = v;
    }
    if let Some(v) = e.exact::<2>("range")? {
        spec.range = v;
    }
    if let Some(v) = read_poly(e, "stiffness")? {
        spec.stiffness = v;
    }
    if let Some(v) = read_poly(e, "damping")? {
        spec.damping = v;
    }
    if let Some(v) = e.num("armature")? {
        spec.armature = v;
    }
    if let Some(v) = e.num("frictionloss")? {
        spec.frictionloss = v;
    }
    read_soft(
        e,
        &mut spec.solref_limit,
        &mut spec.solimp_limit,
        &mut spec.solref_friction,
        &mut spec.solimp_friction,
        &mut spec.margin,
    )
}

/// Reads the mechanical attributes of a `<mesh>` over `spec` (not `name`,
/// `class` or `file`, which belong to the asset and are checked by the caller).
pub(crate) fn read_mesh_spec(e: &E, spec: &mut MeshSpec) -> Result<(), MjcfError> {
    if let Some(v) = e.exact::<3>("scale")? {
        spec.scale = v;
    }
    if let Some(text) = e.attr("inertia") {
        spec.inertia = match text {
            "legacy" => MeshInertia::Legacy,
            "exact" => MeshInertia::Exact,
            "convex" | "shell" => {
                return Err(e.err(
                    MjcfErrorKind::UnsupportedAttribute,
                    format!("mesh inertia '{text}' is not supported (legacy and exact are)"),
                ));
            }
            other => return Err(e.bad(format!("invalid keyword '{other}' in attribute 'inertia'"))),
        };
    }
    Ok(())
}

/// Reads the attributes of a `<material>` over `spec`.
pub(crate) fn read_material_spec(e: &E, spec: &mut MaterialSpec) -> Result<(), MjcfError> {
    if let Some(v) = e.exact::<4>("rgba")? {
        spec.rgba = [v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32];
    }
    if let Some(v) = e.num("metallic")? {
        spec.metallic = v as f32;
    }
    if let Some(v) = e.num("roughness")? {
        spec.roughness = v as f32;
    }
    if let Some(v) = e.num("emission")? {
        spec.emission = v as f32;
    }
    Ok(())
}

/// Reads a `<camera>` (in a body, or in a `<default>`) over `spec`.
pub(crate) fn read_camera(
    ctx: &mut Ctx,
    e: &E,
    spec: &mut CameraSpec,
    in_default: bool,
) -> Result<(), MjcfError> {
    ctx.check_attrs(
        e,
        &[
            "name",
            "class",
            "mode",
            "target",
            "projection",
            "fovy",
            "resolution",
            "focal",
            "focalpixel",
            "principal",
            "principalpixel",
            "sensorsize",
            "pos",
            "quat",
            "axisangle",
            "xyaxes",
            "zaxis",
            "euler",
        ],
        &[
            (
                "ipd",
                "the eye separation of stereo rendering; a scene camera is monocular",
            ),
            (
                "output",
                "the buffers a MuJoCo camera sensor outputs; a scene camera serves every buffer the senses read",
            ),
            ("user", "user data, not modelled"),
        ],
    )?;
    non_default(e, in_default, &["name", "class", "target"])?;
    // the schema's `exclusive fovy sensorsize` (mjcf.schema:1111), per element
    if e.has("fovy") && e.has("sensorsize") {
        return Err(e.err(
            MjcfErrorKind::Inconsistent,
            "at most one of 'fovy', 'sensorsize' can be specified",
        ));
    }
    if let Some(m) = e.keyword("mode", &CAMERA_MODE_KEYWORDS)? {
        spec.mode = m;
    }
    if let Some(p) = e.keyword("projection", &PROJECTION_KEYWORDS)? {
        spec.projection = p;
    }
    if let Some(t) = e.attr("target") {
        spec.target = Some(t.to_string());
    }
    if let Some(v) = e.num("fovy")? {
        spec.fovy = v;
    }
    if let Some(v) = e.ints::<2>("resolution")? {
        spec.resolution = v;
    }
    if let Some(v) = e.floats::<2>("focal")? {
        spec.focal = v;
    }
    if let Some(v) = e.floats::<2>("focalpixel")? {
        spec.focalpixel = v;
    }
    if let Some(v) = e.floats::<2>("principal")? {
        spec.principal = v;
    }
    if let Some(v) = e.floats::<2>("principalpixel")? {
        spec.principalpixel = v;
    }
    if let Some(v) = e.floats::<2>("sensorsize")? {
        spec.sensorsize = v;
    }
    if let Some(v) = e.exact::<3>("pos")? {
        spec.pos = v;
    }
    read_orientation(e, &mut spec.quat, &mut spec.alt)
}

/// Reads a `<site>` (in a body, or in a `<default>`) over `spec`: the attributes that
/// place it, and its `material` (which must exist). Its `group` and `rgba` are checked
/// and not kept, `user` is not read (the site is recorded, see the module note);
/// `mesh` is refused, because a mesh site takes its size from the mesh.
pub(crate) fn read_site(
    ctx: &mut Ctx,
    e: &E,
    spec: &mut SiteSpec,
    in_default: bool,
) -> Result<(), MjcfError> {
    ctx.check_attrs(
        e,
        &[
            "name",
            "class",
            "type",
            "group",
            "pos",
            "quat",
            "axisangle",
            "xyaxes",
            "zaxis",
            "euler",
            "material",
            "size",
            "fromto",
            "rgba",
            "user",
        ],
        &[],
    )?;
    non_default(e, in_default, &["name", "class"])?;
    if let Some(t) = e.keyword("type", &SITE_TYPE_KEYWORDS)? {
        spec.ty = t;
    }
    // parsed so that a malformed value is refused as MuJoCo refuses it; a site's
    // group and colour have no effect on the scene (its material is resolved when it
    // is compiled)
    e.int("group")?;
    e.exact::<4>("rgba")?;
    if let Some(v) = e.nums("size", 1, 3)? {
        spec.size[..v.len()].copy_from_slice(&v);
    }
    if let Some(m) = e.attr("material") {
        spec.material = Some(m.to_string());
    }
    if let Some(v) = e.exact::<6>("fromto")? {
        spec.fromto = v;
    }
    if let Some(v) = e.exact::<3>("pos")? {
        spec.pos = v;
    }
    read_orientation(e, &mut spec.quat, &mut spec.alt)
}
