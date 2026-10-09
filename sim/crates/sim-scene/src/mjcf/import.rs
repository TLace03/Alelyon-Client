//! Reads an MJCF document into the importer's raw records.
//!
//! The document is read in MuJoCo's section order (`mjXReader::Parse`,
//! xml_native_reader.cc:195-310): compiler, option, visual, statistic, default,
//! asset, contact, keyframe, worldbody, then tendon and actuator (which refer to
//! joints by name, so they are read after the bodies; MuJoCo defers the lookup to
//! compile time, the result is the same).
//!
//! Invariants:
//! - Anything not named in this module's tables is refused with
//!   `UnsupportedElement` or `UnsupportedAttribute` and the element's path and
//!   line. Nothing is skipped without either an error or an entry in
//!   `Scene::unsupported`.
//! - Bodies are numbered in depth-first document order (MuJoCo's order); a body's
//!   geoms and joints keep document order within the body.
//! - Names are unique per kind (body, joint, geom, site, camera, mesh, material,
//!   actuator, tendon, default class); `world` is reserved for the world body.
//! - Of `<visual>`, only `<map znear zfar>` and `<global offwidth offheight>` are
//!   read (they place a camera's clip planes and size its image); every other
//!   visual setting is recorded. Of `<statistic>`, `extent`, `center` and
//!   `meansize` are read: they override the computed model statistic, as in MuJoCo,
//!   and the extent scales the cameras' clip planes. `meanmass` is recorded, and
//!   `meaninertia` is refused: MuJoCo's constraint solvers scale their tolerance by
//!   it, so ignoring an override would change the dynamics.
//! - Mesh files are read only from inside the asset directory: an absolute path,
//!   a `..` component, or a symlink that leaves the directory is refused, and a
//!   file larger than MuJoCo's STL decoder reads is refused before it is read.

use std::collections::HashMap;
use std::path::{Component, Path};

use roxmltree::Document;

use crate::error::{MjcfError, MjcfErrorKind};
use crate::scene::{Cone, Integrator, Solver, SolverOptions, Unsupported};

use super::LoadOptions;
use super::mesh::{self, MAX_STL_BYTES, ProcessedMesh};
use super::mjmath::{Alt, OrientKind};
use super::specs::{
    ActuatorSpec, ActuatorTag, CameraSpec, Class, GeomSpec, JointSpec, JointType, Limited,
    MaterialSpec, ORIENTATION_ATTRS, SiteSpec, TendonSpec, read_actuator, read_camera, read_geom,
    read_joint, read_material_spec, read_mesh_spec, read_orientation, read_site, read_tendon,
};
use super::xmlutil::{Ctx, E};

/// The compiler settings (`<compiler>`) the importer supports.
#[derive(Clone, Debug)]
pub(crate) struct Compiler {
    pub degree: bool,
    pub eulerseq: [u8; 3],
    pub inertiafromgeom: Limited,
    pub autolimits: bool,
    pub inertiagrouprange: [i32; 2],
    pub meshdir: String,
}

impl Default for Compiler {
    /// MuJoCo's defaults (mjcf.schema:535-558).
    fn default() -> Self {
        Compiler {
            degree: true,
            eulerseq: *b"xyz",
            inertiafromgeom: Limited::Auto,
            autolimits: true,
            inertiagrouprange: [0, 5],
            meshdir: String::new(),
        }
    }
}

/// An explicit `<inertial>`.
#[derive(Clone, Debug)]
pub(crate) struct RawInertial {
    pub ipos: [f64; 3],
    pub mass: f64,
    pub inertia: [f64; 3],
    pub fullinertia: Option<[f64; 6]>,
    pub iquat: [f64; 4],
    pub ialt: Alt,
    pub path: String,
    pub line: u32,
}

/// A `<body>` as read.
#[derive(Clone, Debug)]
pub(crate) struct RawBody {
    pub name: String,
    pub parent: Option<usize>,
    pub pos: [f64; 3],
    pub quat: [f64; 4],
    pub alt: Alt,
    pub explicit: Option<RawInertial>,
    pub path: String,
    pub line: u32,
}

/// A `<geom>` as read, with its class already merged in.
#[derive(Clone, Debug)]
pub(crate) struct RawGeom {
    pub name: String,
    pub body: Option<usize>,
    pub spec: GeomSpec,
    pub path: String,
    pub line: u32,
}

/// A `<camera>` as read, with its class already merged in.
#[derive(Clone, Debug)]
pub(crate) struct RawCamera {
    pub name: String,
    pub body: Option<usize>,
    pub spec: CameraSpec,
    pub path: String,
    pub line: u32,
}

/// A `<site>` as read, with its class already merged in.
#[derive(Clone, Debug)]
pub(crate) struct RawSite {
    pub name: String,
    pub body: Option<usize>,
    pub spec: SiteSpec,
    pub path: String,
    pub line: u32,
}

/// The `<visual>` settings a scene camera uses (MuJoCo's `mjVisual`, two of its rows).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Visual {
    /// `<map znear>`: the near clip plane as a fraction of the model extent.
    pub znear: f32,
    /// `<map zfar>`: the far clip plane as a fraction of the model extent.
    pub zfar: f32,
    /// `<global offwidth>`: the width of MuJoCo's offscreen buffer, pixels.
    pub offwidth: i32,
    /// `<global offheight>`: the height of MuJoCo's offscreen buffer, pixels.
    pub offheight: i32,
}

impl Default for Visual {
    /// `mj_defaultVisual` (engine_init.c:143-144, 176-177).
    fn default() -> Self {
        Visual {
            znear: 0.01,
            zfar: 50.0,
            offwidth: 640,
            offheight: 480,
        }
    }
}

/// The `<statistic>` values a document sets (MuJoCo's `mjSpec::stat`): each
/// overrides the computed one after the statistic is computed.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StatOverride {
    pub extent: Option<f64>,
    pub meansize: Option<f64>,
    pub center: Option<[f64; 3]>,
}

/// A `<joint>` or `<freejoint>` as read.
#[derive(Clone, Debug)]
pub(crate) struct RawJoint {
    pub name: String,
    pub body: usize,
    pub spec: JointSpec,
    pub path: String,
    pub line: u32,
}

/// An actuator as read.
#[derive(Clone, Debug)]
pub(crate) struct RawActuator {
    pub name: String,
    pub tag: ActuatorTag,
    pub joint: String,
    pub spec: ActuatorSpec,
    pub path: String,
    pub line: u32,
}

/// A fixed tendon as read.
#[derive(Clone, Debug)]
pub(crate) struct RawTendon {
    pub name: String,
    pub spec: TendonSpec,
    pub joints: Vec<(String, f64, u32)>,
    pub path: String,
    pub line: u32,
}

/// A `<contact><exclude>` as read: the body names are resolved in `compile`.
#[derive(Clone, Debug)]
pub(crate) struct RawExclude {
    pub name: String,
    pub body1: String,
    pub body2: String,
    pub path: String,
    pub line: u32,
}

/// A mesh asset after loading and processing.
pub(crate) struct MeshAsset {
    pub name: String,
    pub processed: ProcessedMesh,
}

/// A material asset as read.
#[derive(Clone, Debug)]
pub(crate) struct MaterialAsset {
    pub name: String,
    pub spec: MaterialSpec,
}

/// Everything the reader produced.
pub(crate) struct Imported {
    pub model_name: String,
    pub timestep: f64,
    pub gravity: [f64; 3],
    pub integrator: Integrator,
    pub options: SolverOptions,
    pub compiler: Compiler,
    pub bodies: Vec<RawBody>,
    pub geoms: Vec<RawGeom>,
    pub joints: Vec<RawJoint>,
    pub actuators: Vec<RawActuator>,
    pub tendons: Vec<RawTendon>,
    pub excludes: Vec<RawExclude>,
    pub cameras: Vec<RawCamera>,
    pub sites: Vec<RawSite>,
    pub visual: Visual,
    /// The `<statistic>` overrides the document sets.
    pub stat: StatOverride,
    pub meshes: Vec<MeshAsset>,
    pub materials: Vec<MaterialAsset>,
    pub recorded: Vec<Unsupported>,
}

/// The deepest nesting of bodies (and of default classes) the importer reads: a
/// real model is a few dozen levels deep, and the limit keeps a hostile document
/// from exhausting the stack.
const MAX_NESTING: usize = 256;

/// The longest document the importer reads, bytes. MuJoCo models are kilobytes to
/// a few megabytes; the limit keeps a hostile document from exhausting memory.
const MAX_XML_BYTES: usize = 64 * 1024 * 1024;

struct Importer<'a> {
    ctx: Ctx,
    depth: usize,
    asset_dir: &'a Path,
    compiler: Compiler,
    classes: Vec<Class>,
    class_index: HashMap<String, usize>,
    timestep: f64,
    gravity: [f64; 3],
    integrator: Integrator,
    options: SolverOptions,
    bodies: Vec<RawBody>,
    geoms: Vec<RawGeom>,
    joints: Vec<RawJoint>,
    actuators: Vec<RawActuator>,
    tendons: Vec<RawTendon>,
    excludes: Vec<RawExclude>,
    cameras: Vec<RawCamera>,
    sites: Vec<RawSite>,
    visual: Visual,
    stat: StatOverride,
    meshes: Vec<MeshAsset>,
    mesh_index: HashMap<String, usize>,
    materials: Vec<MaterialAsset>,
    material_index: HashMap<String, usize>,
    body_index: HashMap<String, usize>,
    joint_names: HashMap<String, usize>,
    geom_names: HashMap<String, usize>,
    camera_names: HashMap<String, usize>,
    site_names: HashMap<String, usize>,
}

/// Parses `xml` into raw records.
pub(crate) fn read(
    xml: &str,
    asset_dir: &Path,
    options: &LoadOptions,
) -> Result<Imported, MjcfError> {
    if xml.len() > MAX_XML_BYTES {
        return Err(MjcfError::new(
            MjcfErrorKind::Xml,
            0,
            String::new(),
            format!("the document is larger than {MAX_XML_BYTES} bytes"),
        ));
    }
    let doc = Document::parse(xml).map_err(|e| {
        MjcfError::new(
            MjcfErrorKind::Xml,
            e.pos().row,
            String::new(),
            format!("{e}"),
        )
    })?;
    let root = doc.root_element();
    let mut ctx = Ctx::new(&doc, options.strict);
    let root_e = ctx.elem(root, "");
    if root_e.tag() != "mujoco" {
        return Err(root_e.err(
            MjcfErrorKind::UnsupportedElement,
            format!("the document element is <{}>, not <mujoco>", root_e.tag()),
        ));
    }
    ctx.check_attrs(&root_e, &["model"], &[])?;
    let model_name = root_e.attr("model").unwrap_or("").to_string();

    let mut imp = Importer {
        ctx,
        depth: 0,
        asset_dir,
        compiler: Compiler::default(),
        classes: vec![Class::new("main")],
        class_index: HashMap::from([("main".to_string(), 0)]),
        timestep: crate::scene::DEFAULT_TIMESTEP_S,
        gravity: crate::scene::DEFAULT_GRAVITY,
        integrator: Integrator::Euler,
        options: SolverOptions::default(),
        bodies: Vec::new(),
        geoms: Vec::new(),
        joints: Vec::new(),
        actuators: Vec::new(),
        tendons: Vec::new(),
        excludes: Vec::new(),
        cameras: Vec::new(),
        sites: Vec::new(),
        visual: Visual::default(),
        stat: StatOverride::default(),
        meshes: Vec::new(),
        mesh_index: HashMap::new(),
        materials: Vec::new(),
        material_index: HashMap::new(),
        body_index: HashMap::new(),
        joint_names: HashMap::new(),
        geom_names: HashMap::new(),
        camera_names: HashMap::new(),
        site_names: HashMap::new(),
    };
    imp.read_document(&root_e)?;
    Ok(Imported {
        model_name,
        timestep: imp.timestep,
        gravity: imp.gravity,
        integrator: imp.integrator,
        options: imp.options,
        compiler: imp.compiler,
        bodies: imp.bodies,
        geoms: imp.geoms,
        joints: imp.joints,
        actuators: imp.actuators,
        tendons: imp.tendons,
        excludes: imp.excludes,
        cameras: imp.cameras,
        sites: imp.sites,
        visual: imp.visual,
        stat: imp.stat,
        meshes: imp.meshes,
        materials: imp.materials,
        recorded: imp.ctx.recorded,
    })
}

/// Removes and returns the elements of section `tag`.
fn take<'d, 'i>(sections: &mut HashMap<&str, Vec<E<'d, 'i>>>, tag: &str) -> Vec<E<'d, 'i>> {
    sections.remove(tag).unwrap_or_default()
}

/// Sections MuJoCo has that this importer refuses (anything else unknown is
/// refused too, with a different message).
const REFUSED_SECTIONS: [(&str, &str); 6] = [
    ("size", "memory sizes are not part of a scene"),
    ("extension", "plugins are not supported"),
    ("custom", "custom numeric and text data are not supported"),
    ("deformable", "flexes and skins are not supported yet"),
    (
        "equality",
        "equality constraints change the mechanism and are not supported yet",
    ),
    (
        "sensor",
        "MuJoCo sensors are not part of a scene; the senses are the simulator's own",
    ),
];

impl<'a> Importer<'a> {
    fn read_document(&mut self, root: &E<'_, '_>) -> Result<(), MjcfError> {
        let mut sections: HashMap<&str, Vec<E<'_, '_>>> = HashMap::new();
        for node in root.children() {
            let e = self.ctx.elem(node, "");
            let tag = node.tag_name().name();
            match tag {
                "compiler" | "option" | "visual" | "statistic" | "default" | "asset"
                | "contact" | "keyframe" | "worldbody" | "tendon" | "actuator" => {
                    sections.entry(tag).or_default().push(e);
                }
                _ => {
                    let why = REFUSED_SECTIONS
                        .iter()
                        .find(|(t, _)| *t == tag)
                        .map_or("unrecognised element", |(_, w)| *w);
                    return Err(e.err(
                        MjcfErrorKind::UnsupportedElement,
                        format!("<{tag}> is not supported by this importer ({why})"),
                    ));
                }
            }
        }

        for e in take(&mut sections, "compiler") {
            self.read_compiler(&e)?;
        }
        for e in take(&mut sections, "option") {
            self.read_option(&e)?;
        }
        for e in take(&mut sections, "visual") {
            self.read_visual(&e)?;
        }
        for e in take(&mut sections, "statistic") {
            self.read_statistic(&e)?;
        }
        for e in take(&mut sections, "default") {
            self.read_default(&e, None)?;
        }
        for e in take(&mut sections, "asset") {
            self.read_asset(&e)?;
        }
        for e in take(&mut sections, "contact") {
            self.read_contact(&e)?;
        }
        for e in take(&mut sections, "keyframe") {
            self.ctx.record_element(
                &e,
                "keyframes (named initial states) are not modelled; a scene resets from its reference pose",
            )?;
        }
        for e in take(&mut sections, "worldbody") {
            self.read_worldbody(&e)?;
        }
        for e in take(&mut sections, "tendon") {
            self.read_tendon_section(&e)?;
        }
        for e in take(&mut sections, "actuator") {
            self.read_actuator_section(&e)?;
        }
        Ok(())
    }

    /// `<visual>`: the clip planes and the offscreen size are read, the rest recorded
    /// (`mjXReader::Visual`, xml_native_reader.cc:2022; attribute tables
    /// `kGlobalAttrs` and `kMapAttrs` of mjcf_read_table.inc).
    fn read_visual(&mut self, e: &E<'_, '_>) -> Result<(), MjcfError> {
        self.ctx.check_attrs(e, &[], &[])?;
        const VIEWER: &str =
            "a setting of MuJoCo's interactive viewer and free camera, not part of a scene";
        const DECORATION: &str = "a visualisation setting of MuJoCo's viewer (decoration scales, fog, haze, shadows), not part of a scene";
        for child in e.children() {
            let ce = self.ctx.elem(child, &e.path);
            match ce.tag() {
                "global" => {
                    let recorded: Vec<(&str, &str)> = [
                        "cameraid",
                        "orthographic",
                        "fovy",
                        "ipd",
                        "azimuth",
                        "elevation",
                        "linewidth",
                        "glow",
                        "realtime",
                        "ellipsoidinertia",
                        "bvactive",
                    ]
                    .iter()
                    .map(|a| (*a, VIEWER))
                    .collect();
                    self.ctx
                        .check_attrs(&ce, &["offwidth", "offheight"], &recorded)?;
                    if let Some(v) = ce.int("offwidth")? {
                        self.visual.offwidth = v;
                    }
                    if let Some(v) = ce.int("offheight")? {
                        self.visual.offheight = v;
                    }
                }
                "map" => {
                    let recorded: Vec<(&str, &str)> = [
                        "stiffness",
                        "stiffnessrot",
                        "force",
                        "torque",
                        "alpha",
                        "fogstart",
                        "fogend",
                        "haze",
                        "shadowclip",
                        "shadowscale",
                        "actuatortendon",
                    ]
                    .iter()
                    .map(|a| (*a, DECORATION))
                    .collect();
                    self.ctx.check_attrs(&ce, &["znear", "zfar"], &recorded)?;
                    if let Some([v]) = ce.floats::<1>("znear")? {
                        self.visual.znear = v;
                    }
                    if let Some([v]) = ce.floats::<1>("zfar")? {
                        self.visual.zfar = v;
                    }
                }
                "quality" => self.ctx.record_element(
                    &ce,
                    "render quality (shadow map size, multisampling, tessellation) belongs to the renderer",
                )?,
                "headlight" => self.ctx.record_element(
                    &ce,
                    "the headlight of MuJoCo's viewer; a scene's lighting is the renderer's",
                )?,
                "rgba" => self
                    .ctx
                    .record_element(&ce, "the colours of MuJoCo's viewer decorations")?,
                "scale" => self
                    .ctx
                    .record_element(&ce, "the sizes of MuJoCo's viewer decorations")?,
                other => {
                    return Err(ce.err(
                        MjcfErrorKind::UnsupportedElement,
                        format!("<{other}> in <visual> is not supported by this importer"),
                    ));
                }
            }
        }
        Ok(())
    }

    /// `<statistic>` (`mjXReader::Statistic`, xml_native_reader.cc:561-570): `extent`,
    /// `meansize` and `center` override the computed model statistic (the extent
    /// scales the clip planes); `meanmass` is recorded, `meaninertia` refused.
    fn read_statistic(&mut self, e: &E<'_, '_>) -> Result<(), MjcfError> {
        if e.has("meaninertia") {
            return Err(e.err(
                MjcfErrorKind::UnsupportedAttribute,
                "attribute 'meaninertia' of <statistic> is not supported by this importer (MuJoCo's constraint solvers scale their tolerance by it; the physics derives it from the model)",
            ));
        }
        self.ctx.check_attrs(
            e,
            &["extent", "meansize", "center"],
            &[(
                "meanmass",
                "the mean body mass MuJoCo's viewer scales its perturbations by, not part of a scene",
            )],
        )?;
        // read so that a malformed value is refused, as MuJoCo refuses it
        e.num("meanmass")?;
        if let Some(v) = e.num("extent")? {
            if v <= 0.0 {
                return Err(e.bad("extent must be strictly positive"));
            }
            self.stat.extent = Some(v);
        }
        if let Some(v) = e.num("meansize")? {
            self.stat.meansize = Some(v);
        }
        if let Some(v) = e.exact::<3>("center")? {
            self.stat.center = Some(v);
        }
        Ok(())
    }

    fn read_compiler(&mut self, e: &E<'_, '_>) -> Result<(), MjcfError> {
        self.ctx.check_attrs(
            e,
            &[
                "angle",
                "eulerseq",
                "inertiafromgeom",
                "autolimits",
                "inertiagrouprange",
                "meshdir",
                "assetdir",
                "coordinate",
            ],
            &[
                ("texturedir", "textures are not modelled"),
                (
                    "usethread",
                    "a compile-time threading hint, no effect on the scene",
                ),
                ("saveinertial", "a writer option, no effect on the scene"),
                (
                    "fitaabb",
                    "applies only to primitives fitted to meshes, which are not supported",
                ),
            ],
        )?;
        if let Some(child) = e.children().next() {
            let ce = self.ctx.elem(child, &e.path);
            return Err(ce.err(
                MjcfErrorKind::UnsupportedElement,
                format!("<{}> in <compiler> is not supported", ce.tag()),
            ));
        }
        if let Some(text) = e.attr("coordinate")
            && text != "local"
        {
            return Err(e.bad(
                "global coordinates are not supported (MuJoCo removed them too); only coordinate=\"local\"",
            ));
        }
        if let Some(v) = e.keyword("angle", &[("degree", true), ("radian", false)])? {
            self.compiler.degree = v;
        }
        if let Some(text) = e.attr("eulerseq") {
            let b = text.as_bytes();
            if b.len() != 3
                || !b
                    .iter()
                    .all(|c| matches!(c, b'x' | b'y' | b'z' | b'X' | b'Y' | b'Z'))
            {
                return Err(e.bad("attribute 'eulerseq' must be 3 characters from xyzXYZ"));
            }
            self.compiler.eulerseq = [b[0], b[1], b[2]];
        }
        if let Some(v) = e.keyword(
            "inertiafromgeom",
            &[
                ("false", Limited::False),
                ("true", Limited::True),
                ("auto", Limited::Auto),
            ],
        )? {
            self.compiler.inertiafromgeom = v;
        }
        if let Some(v) = e.boolean("autolimits")? {
            self.compiler.autolimits = v;
        }
        if let Some(v) = e.ints::<2>("inertiagrouprange")? {
            self.compiler.inertiagrouprange = v;
        }
        // assetdir fans out to meshdir; an explicit meshdir overrides it
        // (xml_native_reader.cc:339-345)
        if let Some(d) = e.attr("assetdir") {
            self.compiler.meshdir = d.to_string();
        }
        if let Some(d) = e.attr("meshdir") {
            self.compiler.meshdir = d.to_string();
        }
        Ok(())
    }

    fn read_option(&mut self, e: &E<'_, '_>) -> Result<(), MjcfError> {
        self.ctx.check_attrs(
            e,
            &[
                "timestep",
                "gravity",
                "integrator",
                "solver",
                "iterations",
                "tolerance",
                "ls_iterations",
                "ls_tolerance",
                "cone",
                "impratio",
            ],
            &[
                ("noslip_tolerance", "solver parameter, not modelled yet"),
                ("ccd_tolerance", "solver parameter, not modelled yet"),
                ("sleep_tolerance", "solver parameter, not modelled yet"),
                (
                    "magnetic",
                    "only magnetometer sensors read it, and sensors are not part of a scene",
                ),
                (
                    "o_margin",
                    "contact override, only active with the override flag, not modelled yet",
                ),
                (
                    "o_solref",
                    "contact override, only active with the override flag, not modelled yet",
                ),
                (
                    "o_solimp",
                    "contact override, only active with the override flag, not modelled yet",
                ),
                (
                    "o_friction",
                    "contact override, only active with the override flag, not modelled yet",
                ),
                ("jacobian", "solver parameter, not modelled yet"),
                ("noslip_iterations", "solver parameter, not modelled yet"),
                ("ccd_iterations", "solver parameter, not modelled yet"),
                ("sdf_iterations", "solver parameter, not modelled yet"),
                ("sdf_initpoints", "solver parameter, not modelled yet"),
            ],
        )?;
        if let Some(child) = e.children().next() {
            let ce = self.ctx.elem(child, &e.path);
            return Err(ce.err(
                MjcfErrorKind::UnsupportedElement,
                format!(
                    "<{}> in <option> is not supported (flags switch parts of the physics on and off)",
                    ce.tag()
                ),
            ));
        }
        if let Some(v) = e.num("timestep")? {
            if v <= 0.0 {
                return Err(e.bad("attribute 'timestep' must be positive"));
            }
            self.timestep = v;
        }
        if let Some(v) = e.exact::<3>("gravity")? {
            self.gravity = v;
        }
        // the solver options (MuJoCo's keywords are PGS, CG and Newton; pyramidal and
        // elliptic). MuJoCo accepts a negative iteration count or tolerance and means
        // nothing by it; the importer refuses them.
        match e.attr("solver") {
            None => {}
            Some("PGS") => self.options.solver = Solver::Pgs,
            Some("CG") => self.options.solver = Solver::Cg,
            Some("Newton") => self.options.solver = Solver::Newton,
            Some(other) => {
                return Err(e.bad(format!("invalid keyword '{other}' in attribute 'solver'")));
            }
        }
        match e.attr("cone") {
            None => {}
            Some("pyramidal") => self.options.cone = Cone::Pyramidal,
            Some("elliptic") => self.options.cone = Cone::Elliptic,
            Some(other) => {
                return Err(e.bad(format!("invalid keyword '{other}' in attribute 'cone'")));
            }
        }
        if let Some(v) = e.int("iterations")? {
            self.options.iterations = u32::try_from(v)
                .map_err(|_| e.bad("attribute 'iterations' must not be negative"))?;
        }
        if let Some(v) = e.int("ls_iterations")? {
            self.options.ls_iterations = u32::try_from(v)
                .map_err(|_| e.bad("attribute 'ls_iterations' must not be negative"))?;
        }
        if let Some(v) = e.num("tolerance")? {
            if v < 0.0 {
                return Err(e.bad("attribute 'tolerance' must not be negative"));
            }
            self.options.tolerance = v;
        }
        if let Some(v) = e.num("ls_tolerance")? {
            if v < 0.0 {
                return Err(e.bad("attribute 'ls_tolerance' must not be negative"));
            }
            self.options.ls_tolerance = v;
        }
        if let Some(v) = e.num("impratio")? {
            if v <= 0.0 {
                return Err(e.bad("attribute 'impratio' must be positive"));
            }
            self.options.impratio = v;
        }
        // MuJoCo's keywords are Euler, RK4, implicit, implicitfast and discrete
        // (`integrator_map` of the XML reader). Euler and RK4 are modelled by
        // `sim-physics`; the others are recorded and the scene keeps Euler.
        match e.attr("integrator") {
            None => {}
            Some("Euler") => self.integrator = Integrator::Euler,
            Some("RK4") => self.integrator = Integrator::Rk4,
            Some(kw @ ("implicit" | "implicitfast" | "discrete")) => self.ctx.record(
                e,
                "@integrator",
                &format!(
                    "integrator '{kw}' is not modelled; the scene keeps the Euler integrator (physics models Euler and RK4)"
                ),
            )?,
            Some(other) => {
                return Err(e.bad(format!(
                    "invalid keyword '{other}' in attribute 'integrator'"
                )));
            }
        }
        Ok(())
    }

    /// `<contact>`: an `<exclude name body1 body2>` is read (the body names are resolved
    /// when the bodies exist, in `compile`); a `<pair>` is recorded, not modelled, because
    /// a pair's compiled defaults (condim, friction, solref, margin, signature) come from
    /// MuJoCo compiler code that is not in the reference tree.
    fn read_contact(&mut self, e: &E<'_, '_>) -> Result<(), MjcfError> {
        self.ctx.check_attrs(e, &[], &[])?;
        for child in e.children() {
            let ce = self.ctx.elem(child, &e.path);
            match ce.tag() {
                "exclude" => {
                    self.ctx.check_attrs(&ce, &["name", "body1", "body2"], &[])?;
                    let name = ce.attr("name").unwrap_or("").to_string();
                    if !name.is_empty() && self.excludes.iter().any(|x| x.name == name) {
                        return Err(ce.err(
                            MjcfErrorKind::Duplicate,
                            format!("repeated exclude name '{name}'"),
                        ));
                    }
                    let body1 = ce.require("body1")?.to_string();
                    let body2 = ce.require("body2")?.to_string();
                    self.excludes.push(RawExclude {
                        name,
                        body1,
                        body2,
                        path: ce.path.clone(),
                        line: ce.line,
                    });
                }
                "pair" => self.ctx.record_element(
                    &ce,
                    "explicit contact pairs are not modelled yet: their compiled defaults come from MuJoCo compiler code absent from the reference tree",
                )?,
                other => {
                    return Err(ce.err(
                        MjcfErrorKind::UnsupportedElement,
                        format!("<{other}> in <contact> is not supported"),
                    ));
                }
            }
        }
        Ok(())
    }

    // ---- default classes ------------------------------------------------

    fn read_default(&mut self, e: &E<'_, '_>, parent: Option<usize>) -> Result<(), MjcfError> {
        self.ctx.check_attrs(e, &["class"], &[])?;
        let class_attr = e.attr("class");
        let index = match parent {
            None => {
                if let Some(c) = class_attr
                    && !c.is_empty()
                    && c != "main"
                {
                    return Err(e.bad("top-level default class 'main' cannot be renamed"));
                }
                0
            }
            Some(p) => {
                let name = class_attr.unwrap_or("");
                if name.is_empty() {
                    return Err(e.bad("empty class name"));
                }
                if self.class_index.contains_key(name) {
                    return Err(e.err(MjcfErrorKind::Duplicate, "repeated default class name"));
                }
                self.new_class(p, name)
            }
        };

        // elements other than nested defaults first, then nested defaults
        for child in e.children() {
            let ce = self.ctx.elem(child, &e.path);
            self.read_default_element(&ce, index)?;
        }
        for child in e.children() {
            if child.tag_name().name() == "default" {
                let ce = self.ctx.elem(child, &e.path);
                self.nested(&ce, |imp| imp.read_default(&ce, Some(index)))?;
            }
        }
        Ok(())
    }

    /// A new class named `name`, a full copy of class `parent`. Kept out of
    /// `read_default`'s frame (`inline(never)`): a class holds a spec of every element
    /// kind, and nested default classes recurse, so the copy on that frame would cost
    /// stack at every nesting level.
    #[inline(never)]
    fn new_class(&mut self, parent: usize, name: &str) -> usize {
        let mut class = self.classes[parent].clone();
        class.name = name.to_string();
        self.classes.push(class);
        let index = self.classes.len() - 1;
        self.class_index.insert(name.to_string(), index);
        index
    }

    /// Reads one child element of a `<default>` section into class `index` (not a nested
    /// `<default>`, which the caller recurses into). Kept out of `read_default`'s frame
    /// for the same reason as [`Self::new_class`]: each arm holds a spec copy.
    #[inline(never)]
    fn read_default_element(&mut self, ce: &E<'_, '_>, index: usize) -> Result<(), MjcfError> {
        match ce.tag() {
            "default" => {}
            "geom" => {
                let mut spec = self.classes[index].geom.clone();
                read_geom(&mut self.ctx, ce, &mut spec, true)?;
                self.classes[index].geom = spec;
            }
            "joint" => {
                let mut spec = self.classes[index].joint.clone();
                read_joint(&mut self.ctx, ce, &mut spec, true)?;
                self.classes[index].joint = spec;
            }
            "motor" | "position" => {
                let tag = if ce.tag() == "motor" {
                    ActuatorTag::Motor
                } else {
                    ActuatorTag::Position
                };
                let mut spec = self.classes[index].actuator.clone();
                read_actuator(&mut self.ctx, ce, &mut spec, tag, true)?;
                self.classes[index].actuator = spec;
            }
            "tendon" => {
                let mut spec = self.classes[index].tendon.clone();
                read_tendon(&mut self.ctx, ce, &mut spec, true)?;
                self.classes[index].tendon = spec;
            }
            "mesh" => {
                self.ctx.check_attrs(ce, &["scale", "inertia"], &[])?;
                let mut spec = self.classes[index].mesh.clone();
                read_mesh_spec(ce, &mut spec)?;
                self.classes[index].mesh = spec;
            }
            "material" => {
                self.ctx.check_attrs(
                    ce,
                    &["rgba", "metallic", "roughness", "emission"],
                    &[
                        ("texture", "textures are not modelled"),
                        ("texrepeat", "textures are not modelled"),
                        ("texuniform", "textures are not modelled"),
                        (
                            "specular",
                            "specular is not modelled; the renderer is metallic-roughness",
                        ),
                        (
                            "shininess",
                            "shininess is not modelled; the renderer is metallic-roughness",
                        ),
                        ("reflectance", "reflectance is not modelled yet"),
                    ],
                )?;
                let mut spec = self.classes[index].material.clone();
                read_material_spec(ce, &mut spec)?;
                self.classes[index].material = spec;
            }
            "camera" => {
                let mut spec = self.classes[index].camera.clone();
                read_camera(&mut self.ctx, ce, &mut spec, true)?;
                self.classes[index].camera = spec;
            }
            "site" => {
                let mut spec = self.classes[index].site.clone();
                read_site(&mut self.ctx, ce, &mut spec, true)?;
                self.classes[index].site = spec;
            }
            "light" | "pair" | "equality" => {
                self.ctx
                    .record_element(ce, "defaults for an element kind that is not modelled yet")?;
            }
            other => {
                return Err(ce.err(
                    MjcfErrorKind::UnsupportedElement,
                    format!("<{other}> in <default> is not supported by this importer"),
                ));
            }
        }
        Ok(())
    }

    /// The class named by `e`'s `class` attribute, or `fallback`.
    fn class_for(&self, e: &E<'_, '_>, fallback: usize) -> Result<usize, MjcfError> {
        match e.attr("class") {
            None => Ok(fallback),
            Some(name) => self.find_class(e, name),
        }
    }

    fn find_class(&self, e: &E<'_, '_>, name: &str) -> Result<usize, MjcfError> {
        self.class_index.get(name).copied().ok_or_else(|| {
            e.err(
                MjcfErrorKind::UnknownReference,
                format!("unknown default class '{name}'"),
            )
        })
    }

    // ---- assets ---------------------------------------------------------

    fn read_asset(&mut self, e: &E<'_, '_>) -> Result<(), MjcfError> {
        self.ctx.check_attrs(e, &[], &[])?;
        for child in e.children() {
            let ce = self.ctx.elem(child, &e.path);
            match ce.tag() {
                "mesh" => self.read_mesh(&ce)?,
                "material" => self.read_material(&ce)?,
                "texture" => self.ctx.record_element(
                    &ce,
                    "textures are not modelled; the optical material has a procedural checker instead",
                )?,
                other => {
                    return Err(ce.err(
                        MjcfErrorKind::UnsupportedElement,
                        format!("<{other}> in <asset> is not supported by this importer"),
                    ));
                }
            }
        }
        Ok(())
    }

    fn read_material(&mut self, e: &E<'_, '_>) -> Result<(), MjcfError> {
        self.ctx.check_attrs(
            e,
            &["name", "class", "rgba", "metallic", "roughness", "emission"],
            &[
                ("texture", "textures are not modelled"),
                ("texrepeat", "textures are not modelled"),
                ("texuniform", "textures are not modelled"),
                (
                    "specular",
                    "specular is not modelled; the renderer is metallic-roughness",
                ),
                (
                    "shininess",
                    "shininess is not modelled; the renderer is metallic-roughness",
                ),
                ("reflectance", "reflectance is not modelled yet"),
            ],
        )?;
        for child in e.children() {
            let ce = self.ctx.elem(child, &e.path);
            match ce.tag() {
                "layer" => self
                    .ctx
                    .record_element(&ce, "texture layers are not modelled")?,
                other => {
                    return Err(ce.err(
                        MjcfErrorKind::UnsupportedElement,
                        format!("<{other}> in <material> is not supported"),
                    ));
                }
            }
        }
        let class = self.class_for(e, 0)?;
        let mut spec = self.classes[class].material.clone();
        read_material_spec(e, &mut spec)?;
        let name = e.attr("name").unwrap_or("").to_string();
        if !name.is_empty() && self.material_index.contains_key(&name) {
            return Err(e.err(
                MjcfErrorKind::Duplicate,
                format!("repeated material name '{name}'"),
            ));
        }
        if spec.rgba[3] != 1.0 {
            self.ctx.record(
                e,
                "@rgba",
                "alpha (transparency) is not modelled; the optical material is opaque",
            )?;
        }
        if !name.is_empty() {
            self.material_index
                .insert(name.clone(), self.materials.len());
        }
        self.materials.push(MaterialAsset { name, spec });
        Ok(())
    }

    fn read_mesh(&mut self, e: &E<'_, '_>) -> Result<(), MjcfError> {
        self.ctx.check_attrs(
            e,
            &["name", "class", "file", "scale", "inertia"],
            &[
                (
                    "content_type",
                    "the decoder is chosen by the .stl extension",
                ),
                ("smoothnormal", "normals are not modelled"),
                (
                    "material",
                    "a mesh's material is chosen by the geom that uses it",
                ),
            ],
        )?;
        if let Some(child) = e.children().next() {
            let ce = self.ctx.elem(child, &e.path);
            return Err(ce.err(
                MjcfErrorKind::UnsupportedElement,
                format!("<{}> in <mesh> is not supported", ce.tag()),
            ));
        }
        if let Some(ct) = e.attr("content_type")
            && ct != "model/stl"
        {
            return Err(e.bad(format!(
                "content_type '{ct}' is not supported; only binary STL (model/stl)"
            )));
        }
        let file = e.require("file")?;
        let class = self.class_for(e, 0)?;
        let mut spec = self.classes[class].mesh.clone();
        read_mesh_spec(e, &mut spec)?;

        let ext_ok = Path::new(file)
            .extension()
            .and_then(|x| x.to_str())
            .is_some_and(|x| x.eq_ignore_ascii_case("stl"));
        if !ext_ok {
            return Err(e.err(
                MjcfErrorKind::Asset,
                format!("mesh file '{file}' is not a .stl file; only binary STL is supported"),
            ));
        }
        let name = match e.attr("name") {
            Some(n) if !n.is_empty() => n.to_string(),
            _ => Path::new(file)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string(),
        };
        if self.mesh_index.contains_key(&name) {
            return Err(e.err(
                MjcfErrorKind::Duplicate,
                format!("repeated mesh name '{name}'"),
            ));
        }
        let bytes = self.read_mesh_file(e, file)?;
        let stl = mesh::read_stl(&bytes)
            .map_err(|m| e.err(MjcfErrorKind::Asset, format!("{file}: {m}")))?;
        let flat: Vec<f32> = stl.vertices.iter().flat_map(|v| *v).collect();
        let processed = mesh::process(&flat, &stl.faces, spec.scale, spec.inertia)
            .map_err(|m| e.err(MjcfErrorKind::Asset, format!("mesh '{name}': {m}")))?;
        self.mesh_index.insert(name.clone(), self.meshes.len());
        self.meshes.push(MeshAsset { name, processed });
        Ok(())
    }

    /// Reads a mesh file from inside the asset directory.
    fn read_mesh_file(&self, e: &E<'_, '_>, file: &str) -> Result<Vec<u8>, MjcfError> {
        let refuse = |why: &str| e.err(MjcfErrorKind::Asset, format!("mesh path '{file}': {why}"));
        let relative_ok = |p: &str| {
            Path::new(p)
                .components()
                .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
        };
        if file.is_empty() || !relative_ok(file) {
            return Err(refuse(
                "must be a relative path with no '..' and no drive or root",
            ));
        }
        if !self.compiler.meshdir.is_empty() && !relative_ok(&self.compiler.meshdir) {
            return Err(e.err(
                MjcfErrorKind::Asset,
                "compiler meshdir/assetdir must be a relative path with no '..' and no drive or root",
            ));
        }
        let base = self.asset_dir.canonicalize().map_err(|err| {
            e.err(
                MjcfErrorKind::Asset,
                format!("asset directory cannot be opened: {err}"),
            )
        })?;
        let full = base
            .join(&self.compiler.meshdir)
            .join(file)
            .canonicalize()
            .map_err(|err| refuse(&format!("cannot be opened: {err}")))?;
        if !full.starts_with(&base) {
            return Err(refuse("resolves outside the asset directory"));
        }
        let meta =
            std::fs::metadata(&full).map_err(|err| refuse(&format!("cannot be read: {err}")))?;
        if !meta.is_file() {
            return Err(refuse("is not a regular file"));
        }
        if meta.len() > MAX_STL_BYTES {
            return Err(refuse("is larger than 200000 triangles"));
        }
        std::fs::read(&full).map_err(|err| refuse(&format!("cannot be read: {err}")))
    }

    // ---- bodies ---------------------------------------------------------

    fn read_worldbody(&mut self, e: &E<'_, '_>) -> Result<(), MjcfError> {
        if e.node.attributes().next().is_some() {
            return Err(e.err(
                MjcfErrorKind::UnsupportedAttribute,
                "the world body cannot have attributes",
            ));
        }
        self.read_body_children(e, None, 0)
    }

    fn read_body(
        &mut self,
        e: &E<'_, '_>,
        parent: Option<usize>,
        parent_class: usize,
    ) -> Result<(), MjcfError> {
        let mut supported = vec!["name", "pos", "childclass"];
        supported.extend(ORIENTATION_ATTRS);
        self.ctx.check_attrs(
            e,
            &supported,
            &[
                ("sleep", "sleeping policy, not modelled"),
                ("simple", "a compiler hint, no effect on the scene"),
                ("user", "user data, not modelled"),
            ],
        )?;
        let class = match e.attr("childclass") {
            None => parent_class,
            Some(name) => self.find_class(e, name)?,
        };
        let name = e.attr("name").unwrap_or("").to_string();
        if name == "world" {
            return Err(e.err(
                MjcfErrorKind::Duplicate,
                "the name 'world' is reserved for the world body",
            ));
        }
        if !name.is_empty() && self.body_index.contains_key(&name) {
            return Err(e.err(
                MjcfErrorKind::Duplicate,
                format!("repeated body name '{name}'"),
            ));
        }
        let mut pos = [0.0; 3];
        if let Some(v) = e.exact::<3>("pos")? {
            pos = v;
        }
        let mut quat = [1.0, 0.0, 0.0, 0.0];
        let mut alt = Alt::default();
        read_orientation(e, &mut quat, &mut alt)?;
        let index = self.bodies.len();
        if !name.is_empty() {
            self.body_index.insert(name.clone(), index);
        }
        self.bodies.push(RawBody {
            name,
            parent,
            pos,
            quat,
            alt,
            explicit: None,
            path: e.path.clone(),
            line: e.line,
        });
        self.nested(e, |imp| imp.read_body_children(e, Some(index), class))
    }

    /// Runs `f` one level deeper, refusing a document nested beyond [`MAX_NESTING`].
    fn nested(
        &mut self,
        e: &E<'_, '_>,
        f: impl FnOnce(&mut Self) -> Result<(), MjcfError>,
    ) -> Result<(), MjcfError> {
        if self.depth >= MAX_NESTING {
            return Err(e.err(
                MjcfErrorKind::Inconsistent,
                format!("elements are nested more than {MAX_NESTING} levels deep"),
            ));
        }
        self.depth += 1;
        let result = f(self);
        self.depth -= 1;
        result
    }

    fn read_body_children(
        &mut self,
        e: &E<'_, '_>,
        body: Option<usize>,
        class: usize,
    ) -> Result<(), MjcfError> {
        for child in e.children() {
            let ce = self.ctx.elem(child, &e.path);
            match ce.tag() {
                "body" => self.read_body(&ce, body, class)?,
                "inertial" => self.read_inertial(&ce, body)?,
                "joint" => self.read_body_joint(&ce, body, class)?,
                "freejoint" => self.read_freejoint(&ce, body)?,
                "geom" => self.read_body_geom(&ce, body, class)?,
                "camera" => self.read_body_camera(&ce, body, class)?,
                "site" => self.read_body_site(&ce, body, class)?,
                "light" => self
                    .ctx
                    .record_element(&ce, "lights of an MJCF body are not modelled yet")?,
                other => {
                    return Err(ce.err(
                        MjcfErrorKind::UnsupportedElement,
                        format!("<{other}> in a body is not supported by this importer"),
                    ));
                }
            }
        }
        Ok(())
    }

    fn read_inertial(&mut self, e: &E<'_, '_>, body: Option<usize>) -> Result<(), MjcfError> {
        let Some(b) = body else {
            return Err(e.err(
                MjcfErrorKind::Inconsistent,
                "the world body cannot have an inertial",
            ));
        };
        let mut supported = vec!["pos", "mass", "diaginertia", "fullinertia"];
        supported.extend(ORIENTATION_ATTRS);
        self.ctx.check_attrs(e, &supported, &[])?;
        if self.bodies[b].explicit.is_some() {
            return Err(e.err(
                MjcfErrorKind::Duplicate,
                "a body can have only one <inertial>",
            ));
        }
        let ipos = e.exact::<3>("pos")?.ok_or_else(|| {
            e.err(
                MjcfErrorKind::MissingAttribute,
                "required attribute 'pos' is missing",
            )
        })?;
        let mass = e.num("mass")?.ok_or_else(|| {
            e.err(
                MjcfErrorKind::MissingAttribute,
                "required attribute 'mass' is missing",
            )
        })?;
        let inertia = e.exact::<3>("diaginertia")?.unwrap_or([0.0; 3]);
        let fullinertia = e.exact::<6>("fullinertia")?;
        let mut iquat = [1.0, 0.0, 0.0, 0.0];
        let mut ialt = Alt::default();
        read_orientation(e, &mut iquat, &mut ialt)?;
        if fullinertia.is_some() && ialt.kind != OrientKind::Quat {
            return Err(e.err(
                MjcfErrorKind::Inconsistent,
                "fullinertia and inertial orientation cannot both be specified",
            ));
        }
        if fullinertia.is_some() && inertia != [0.0; 3] {
            return Err(e.err(
                MjcfErrorKind::Inconsistent,
                "fullinertia and diagonal inertia cannot both be specified",
            ));
        }
        self.bodies[b].explicit = Some(RawInertial {
            ipos,
            mass,
            inertia,
            fullinertia,
            iquat,
            ialt,
            path: e.path.clone(),
            line: e.line,
        });
        Ok(())
    }

    fn read_body_joint(
        &mut self,
        e: &E<'_, '_>,
        body: Option<usize>,
        class: usize,
    ) -> Result<(), MjcfError> {
        let Some(b) = body else {
            return Err(e.err(
                MjcfErrorKind::Inconsistent,
                "the world body cannot have joints",
            ));
        };
        let c = self.class_for(e, class)?;
        let mut spec = self.classes[c].joint.clone();
        read_joint(&mut self.ctx, e, &mut spec, false)?;
        let name = e.attr("name").unwrap_or("").to_string();
        self.push_joint(e, name, b, spec)
    }

    fn read_freejoint(&mut self, e: &E<'_, '_>, body: Option<usize>) -> Result<(), MjcfError> {
        let Some(b) = body else {
            return Err(e.err(
                MjcfErrorKind::Inconsistent,
                "the world body cannot have joints",
            ));
        };
        self.ctx.check_attrs(
            e,
            &["name"],
            &[("group", "visualisation group, not modelled")],
        )?;
        // a free joint is created without class defaults (mjs_addFreeJoint)
        let spec = JointSpec {
            ty: JointType::Free,
            ..JointSpec::default()
        };
        let name = e.attr("name").unwrap_or("").to_string();
        self.push_joint(e, name, b, spec)
    }

    fn push_joint(
        &mut self,
        e: &E<'_, '_>,
        name: String,
        body: usize,
        spec: JointSpec,
    ) -> Result<(), MjcfError> {
        if !name.is_empty() && self.joint_names.contains_key(&name) {
            return Err(e.err(
                MjcfErrorKind::Duplicate,
                format!("repeated joint name '{name}'"),
            ));
        }
        if spec.ty == JointType::Free && self.bodies[body].parent.is_some() {
            return Err(e.err(
                MjcfErrorKind::Inconsistent,
                "free joint can only be used on top level bodies",
            ));
        }
        if !name.is_empty() {
            self.joint_names.insert(name.clone(), self.joints.len());
        }
        self.joints.push(RawJoint {
            name,
            body,
            spec,
            path: e.path.clone(),
            line: e.line,
        });
        Ok(())
    }

    fn read_body_geom(
        &mut self,
        e: &E<'_, '_>,
        body: Option<usize>,
        class: usize,
    ) -> Result<(), MjcfError> {
        let c = self.class_for(e, class)?;
        let mut spec = self.classes[c].geom.clone();
        read_geom(&mut self.ctx, e, &mut spec, false)?;
        let name = e.attr("name").unwrap_or("").to_string();
        if !name.is_empty() && self.geom_names.contains_key(&name) {
            return Err(e.err(
                MjcfErrorKind::Duplicate,
                format!("repeated geom name '{name}'"),
            ));
        }
        if !name.is_empty() {
            self.geom_names.insert(name.clone(), self.geoms.len());
        }
        self.geoms.push(RawGeom {
            name,
            body,
            spec,
            path: e.path.clone(),
            line: e.line,
        });
        Ok(())
    }

    fn read_body_camera(
        &mut self,
        e: &E<'_, '_>,
        body: Option<usize>,
        class: usize,
    ) -> Result<(), MjcfError> {
        let c = self.class_for(e, class)?;
        let mut spec = self.classes[c].camera.clone();
        read_camera(&mut self.ctx, e, &mut spec, false)?;
        let name = e.attr("name").unwrap_or("").to_string();
        if !name.is_empty() && self.camera_names.contains_key(&name) {
            return Err(e.err(
                MjcfErrorKind::Duplicate,
                format!("repeated camera name '{name}'"),
            ));
        }
        if !name.is_empty() {
            self.camera_names.insert(name.clone(), self.cameras.len());
        }
        self.cameras.push(RawCamera {
            name,
            body,
            spec,
            path: e.path.clone(),
            line: e.line,
        });
        Ok(())
    }

    /// A site is read for its position (see `specs::read_site`) and recorded: the
    /// scene does not model sites.
    fn read_body_site(
        &mut self,
        e: &E<'_, '_>,
        body: Option<usize>,
        class: usize,
    ) -> Result<(), MjcfError> {
        let c = self.class_for(e, class)?;
        let mut spec = self.classes[c].site.clone();
        read_site(&mut self.ctx, e, &mut spec, false)?;
        let name = e.attr("name").unwrap_or("").to_string();
        if !name.is_empty() && self.site_names.contains_key(&name) {
            return Err(e.err(
                MjcfErrorKind::Duplicate,
                format!("repeated site name '{name}'"),
            ));
        }
        if !name.is_empty() {
            self.site_names.insert(name.clone(), self.sites.len());
        }
        self.ctx.record_element(
            e,
            "sites are not modelled (their positions are read for the model extent, which places the cameras' clip planes)",
        )?;
        self.sites.push(RawSite {
            name,
            body,
            spec,
            path: e.path.clone(),
            line: e.line,
        });
        Ok(())
    }

    // ---- tendons and actuators -----------------------------------------

    fn read_tendon_section(&mut self, e: &E<'_, '_>) -> Result<(), MjcfError> {
        self.ctx.check_attrs(e, &[], &[])?;
        for child in e.children() {
            let ce = self.ctx.elem(child, &e.path);
            match ce.tag() {
                "fixed" => self.read_fixed_tendon(&ce)?,
                other => {
                    return Err(ce.err(
                        MjcfErrorKind::UnsupportedElement,
                        format!("<{other}> tendons are not supported (only <fixed>)"),
                    ));
                }
            }
        }
        Ok(())
    }

    fn read_fixed_tendon(&mut self, e: &E<'_, '_>) -> Result<(), MjcfError> {
        let class = self.class_for(e, 0)?;
        let mut spec = self.classes[class].tendon.clone();
        read_tendon(&mut self.ctx, e, &mut spec, false)?;
        let name = e.attr("name").unwrap_or("").to_string();
        let mut joints = Vec::new();
        for child in e.children() {
            let ce = self.ctx.elem(child, &e.path);
            match ce.tag() {
                "joint" => {
                    self.ctx.check_attrs(&ce, &["joint", "coef"], &[])?;
                    let joint = ce.require("joint")?.to_string();
                    let coef = ce.num("coef")?.ok_or_else(|| {
                        ce.err(
                            MjcfErrorKind::MissingAttribute,
                            "required attribute 'coef' is missing",
                        )
                    })?;
                    joints.push((joint, coef, ce.line));
                }
                other => {
                    return Err(ce.err(
                        MjcfErrorKind::UnsupportedElement,
                        format!("<{other}> in a fixed tendon is not supported"),
                    ));
                }
            }
        }
        if joints.is_empty() {
            return Err(e.err(
                MjcfErrorKind::Inconsistent,
                "a tendon's path cannot be empty",
            ));
        }
        self.tendons.push(RawTendon {
            name,
            spec,
            joints,
            path: e.path.clone(),
            line: e.line,
        });
        Ok(())
    }

    fn read_actuator_section(&mut self, e: &E<'_, '_>) -> Result<(), MjcfError> {
        self.ctx.check_attrs(e, &[], &[])?;
        for child in e.children() {
            let ce = self.ctx.elem(child, &e.path);
            let tag = match ce.tag() {
                "motor" => ActuatorTag::Motor,
                "position" => ActuatorTag::Position,
                other => {
                    return Err(ce.err(
                        MjcfErrorKind::UnsupportedElement,
                        format!("<{other}> actuators are not supported (motor and position are)"),
                    ));
                }
            };
            let class = self.class_for(&ce, 0)?;
            let mut spec = self.classes[class].actuator.clone();
            read_actuator(&mut self.ctx, &ce, &mut spec, tag, false)?;
            let joint = ce.require("joint")?.to_string();
            self.actuators.push(RawActuator {
                name: ce.attr("name").unwrap_or("").to_string(),
                tag,
                joint,
                spec,
                path: ce.path.clone(),
                line: ce.line,
            });
        }
        Ok(())
    }
}
