//! Compiles the raw MJCF records into a [`Scene`]: MuJoCo's compile step.
//!
//! Ports, from MuJoCo commit a8373cc4e `src/user/user_objects.cc`:
//! - `mjCBody::Compile` (2687-2929): orientation resolution, the explicit
//!   `<inertial>` with `fullinertia`, `inferinertia` per geom, the inertial frame
//!   from geoms, the mass and inertia checks;
//! - `mjCBody::InertiaFromGeom` (2413-2479): single geom copied, several geoms
//!   summed with the parallel-axis theorem and diagonalised by `mjuu_eig3`;
//! - `mjCGeom::Compile` (3947-4155): `fromto`, orientation alternatives, mesh
//!   frame, size check, mass from `mass` or `density`;
//! - `mjCJoint::Compile` (3175-3258): `limited` and `autolimits`, degrees to
//!   radians, axis normalisation;
//! - `mjCActuator::Compile` (7313-7330) and `mjCTendon::Compile` (6575-6598):
//!   control and length limits;
//! - `mjCSite::Compile` (4241-4324): a site's type, `fromto`, orientation and size
//!   checks (the site itself is recorded; its position enters the model extent);
//! - `mjCCamera::Compile` (4408-4485) and `mjCCamera::ResolveReferences`
//!   (4395-4405): orientation, the target body, the `fovy` limit, the intrinsics in
//!   `float` arithmetic (`focalpixel` and `principalpixel` through the pixel density
//!   `resolution / sensorsize`) and their checks.
//!
//! The scene camera is what MuJoCo's renderer draws (`setView`,
//! src/render/classic/render_gl3.c:781-830 of commit a8373cc4e, which builds the
//! OpenGL projection from `mjv_cameraFrustum`, engine_vis_visualize.c:529-589 of
//! 3.14.0). With an image of `W x H` pixels, and `near = znear * extent`:
//! - a `fovy` camera: the frustum's half-height at `near` is `near tan(fovy / 2)` and
//!   its half-width is the image aspect times that, so `fx = fy = H / (2 tan(fovy / 2))`
//!   and the principal point is the image centre, `((W - 1) / 2, (H - 1) / 2)` in the
//!   scene's integer-centre convention;
//! - a `sensorsize` camera (focal lengths `f`, sensor size `s`, principal offset `p`,
//!   all in length units): `getFrustum` gives the frustum edges `near / f (s / 2 -+ p)`
//!   and the renderer uses its own width (`frustum_width`) instead of the aspect, so
//!   `fx = W f_x / s_x`, `fy = H f_y / s_y`, `cx = W / 2 - W p_x / s_x - 1/2` and
//!   `cy = H / 2 - H p_y / s_y - 1/2`. A positive principal offset moves the image of
//!   the optical axis left and up, as MuJoCo renders it.
//!
//! MuJoCo has two other camera models, and both disagree with its renderer for a
//! `sensorsize` camera: mouse picking (`mjv_select`) takes the half-width from the
//! viewport aspect, and the engine's pixel helpers (`mju_camIntrinsics`, used by the
//! depth-image rangefinder, and `cam_project`, the camera-projection sensor) put the
//! principal point at `W p_x / s_x` from the image corner or ignore it. The import
//! follows the renderer, because a scene camera says what the renderer draws; for a
//! `fovy` camera all of them agree, and the parity test checks the import against the
//! renderer's frustum, `mjv_select` and the camera-projection sensor.
//!
//! The image is `resolution` when it is larger than 1 x 1 (the test MuJoCo's own
//! visualiser applies before it draws a camera's frustum,
//! engine_vis_visualize.c:2430-2431), else MuJoCo's offscreen buffer,
//! `<visual><global offwidth offheight>` (default 640 x 480). The intrinsics are
//! computed in `f64` from the compiled values and rounded to `f32` once. The camera
//! frame turns from MuJoCo's (x right, y up, looking down -z) to the scene's OpenCV
//! frame (x right, y down, looking down +z) by a half turn about x: `q_cv = q_mj *
//! (0, 1, 0, 0)` (`[w, x, y, z]`), a permutation with sign changes, exact.
//!
//! Invariants:
//! - Bodies come out in depth-first document order, geoms and joints sorted
//!   stably by body (MuJoCo lists them per body in body order), so ids match the
//!   compiled MuJoCo model's.
//! - Internally quaternions are MuJoCo's `[w, x, y, z]`; they are converted to
//!   the scene's `[x, y, z, w]` exactly once, when a scene record is built.
//! - A material is created per distinct (named material, colour) pair; MuJoCo's
//!   `rgba` is read as sRGB and converted with the standard EOTF.
//! - The result is validated with `Scene::validate` before it is returned.

use std::collections::HashMap;
use std::f64::consts::PI;

use crate::body::{Body, Geom, Inertial, Joint, JointKind, Mesh, Shape};
use crate::error::{MjcfError, MjcfErrorKind, Result, SceneError};
use crate::ids::{BodyId, GeomId, JointId, MaterialId, MeshId};
use crate::material::{Material, srgb_to_linear};
use crate::scene::{
    Actuator, ActuatorKind, Camera, CameraMount, ContactExclude, Instance, SCENE_VERSION, Scene,
    ShapeRef, Tendon, TendonJoint, Unsupported,
};

use super::LoadOptions;
use super::import::{
    Compiler, Imported, RawActuator, RawBody, RawCamera, RawGeom, RawJoint, RawSite, RawTendon,
    Visual,
};
use super::inertia::{diag_inertia, volume};
use super::mesh::ProcessedMesh;
use super::mjmath::{
    EPS, OrientKind, frameaccum, full_inertia, globalinertia, mul_rmrt, normvec, offcenter,
    quat2mat, resolve_orientation, z2quat,
};
use super::specs::{
    ActuatorTag, CameraMode, DEFAULT_RGBA, GeomType, JointType, Limited, Projection, SiteType,
};
use super::statistic::{
    StatBody, StatGeom, StatJoint, StatJointKind, StatSite, Statistic, statistic,
};

fn xyzw(q: [f64; 4]) -> [f64; 4] {
    [q[1], q[2], q[3], q[0]]
}

/// A geom after compile.
struct CompiledGeom {
    /// Pose in the body as authored (after `fromto` / orientation), wxyz.
    pos: [f64; 3],
    quat: [f64; 4],
    /// Inertial frame of the geom in the body (the mesh's centre of mass and
    /// principal axes for a mesh), wxyz.
    ipos: [f64; 3],
    iquat: [f64; 4],
    mass: f64,
    inertia: [f64; 3],
    density: f64,
    shape: Shape,
}

/// MuJoCo's `checklimited`: a `range` without `limited` is an error unless
/// `autolimits`.
fn check_limited(
    kind_line: (u32, &str),
    autolimits: bool,
    entity: &str,
    attr: &str,
    limited: Limited,
    hasrange: bool,
) -> std::result::Result<(), MjcfError> {
    if !autolimits && limited == Limited::Auto && hasrange {
        return Err(MjcfError::new(
            MjcfErrorKind::Inconsistent,
            kind_line.0,
            kind_line.1,
            format!(
                "{entity} has `{attr}range` but not `{attr}limited`; set the autolimits=\"true\" compiler option, specify `{attr}limited` explicitly (\"true\" or \"false\"), or remove the `{attr}range` attribute"
            ),
        ));
    }
    Ok(())
}

/// MuJoCo's `islimited`.
fn is_limited(limited: Limited, range: [f64; 2]) -> bool {
    limited == Limited::True || (limited == Limited::Auto && range[0] < range[1])
}

/// Compiles. See the module note. Returns the scene and the model statistic (the
/// computed one, with the document's `<statistic>` values applied over it, as MuJoCo's
/// `mjModel::stat` holds them).
pub(super) fn compile(imp: Imported, options: &LoadOptions) -> Result<(Scene, Statistic)> {
    let c = &imp.compiler;

    // ---- meshes and the mesh name table
    let mesh_ids: HashMap<&str, usize> = imp
        .meshes
        .iter()
        .enumerate()
        .map(|(i, m)| (m.name.as_str(), i))
        .collect();

    // ---- bodies and their geoms, in body order
    let mut body_quats: Vec<[f64; 4]> = Vec::with_capacity(imp.bodies.len());
    for b in &imp.bodies {
        let mut q = b.quat;
        normvec(&mut q);
        if b.alt.kind != OrientKind::Quat {
            resolve_orientation(&mut q, c.degree, &c.eulerseq, &b.alt).map_err(|err| {
                MjcfError::new(
                    MjcfErrorKind::BadValue,
                    b.line,
                    &b.path,
                    format!("error '{err}' in frame alternative"),
                )
            })?;
        }
        body_quats.push(q);
    }

    let mut compiled: Vec<CompiledGeom> = Vec::with_capacity(imp.geoms.len());
    for g in imp.geoms.iter() {
        let infer = match g.body {
            None => false,
            Some(b) => {
                let explicit = imp.bodies[b].explicit.is_some();
                (!explicit || c.inertiafromgeom == Limited::True)
                    && g.spec.group >= c.inertiagrouprange[0]
                    && g.spec.group <= c.inertiagrouprange[1]
            }
        };
        compiled.push(compile_geom(g, infer, &imp, &mesh_ids)?);
    }

    // ---- body records
    let mut geoms_of_body: Vec<Vec<usize>> = vec![Vec::new(); imp.bodies.len()];
    for (gi, g) in imp.geoms.iter().enumerate() {
        if let Some(b) = g.body {
            geoms_of_body[b].push(gi);
        }
    }
    let mut bodies: Vec<Body> = Vec::with_capacity(imp.bodies.len());
    for (bi, b) in imp.bodies.iter().enumerate() {
        let sel: Vec<(&RawGeom, &CompiledGeom)> = geoms_of_body[bi]
            .iter()
            .map(|&gi| (&imp.geoms[gi], &compiled[gi]))
            .collect();
        let inertial = compile_inertial(b, &sel, c)?;
        bodies.push(Body {
            name: b.name.clone(),
            parent: b.parent.map(|p| BodyId(p as u32)),
            pos: b.pos,
            quat: xyzw(body_quats[bi]),
            inertial,
        });
    }

    // ---- joints, sorted stably by body
    let mut joint_order: Vec<usize> = (0..imp.joints.len()).collect();
    joint_order.sort_by_key(|&i| imp.joints[i].body);
    let mut joints: Vec<Joint> = Vec::with_capacity(joint_order.len());
    let mut joint_by_name: HashMap<&str, usize> = HashMap::new();
    for &ji in &joint_order {
        let j = &imp.joints[ji];
        let joint = compile_joint(j, c.degree, c.autolimits)?;
        if !j.name.is_empty() {
            joint_by_name.insert(j.name.as_str(), joints.len());
        }
        joints.push(joint);
    }

    // ---- materials
    let mut materials: Vec<Material> = imp
        .materials
        .iter()
        .map(|m| asset_material(&m.name, &m.spec))
        .collect();
    let material_by_name: HashMap<&str, usize> = imp
        .materials
        .iter()
        .enumerate()
        .filter(|(_, m)| !m.name.is_empty())
        .map(|(i, m)| (m.name.as_str(), i))
        .collect();
    let mut interned: HashMap<(Option<usize>, [u32; 3]), usize> = HashMap::new();

    // ---- geoms, sorted stably by body (the world first)
    let mut geom_order: Vec<usize> = (0..imp.geoms.len()).collect();
    geom_order.sort_by_key(|&i| imp.geoms[i].body.map_or(0, |b| b + 1));
    let mut geoms: Vec<Geom> = Vec::with_capacity(geom_order.len());
    for &gi in &geom_order {
        let g = &imp.geoms[gi];
        let cg = &compiled[gi];
        if g.spec.contype < 0 || g.spec.conaffinity < 0 {
            return Err(MjcfError::new(
                MjcfErrorKind::BadValue,
                g.line,
                &g.path,
                "contype and conaffinity must not be negative",
            )
            .into());
        }
        let material = resolve_material(g, &material_by_name, &mut materials, &mut interned)?;
        geoms.push(Geom {
            name: g.name.clone(),
            body: g.body.map(|b| BodyId(b as u32)),
            shape: cg.shape,
            pos: cg.pos,
            quat: xyzw(cg.quat),
            material: MaterialId(material as u32),
            contype: g.spec.contype as u32,
            conaffinity: g.spec.conaffinity as u32,
            condim: g.spec.condim as u32,
            friction: g.spec.friction,
            density: cg.density,
            solref: g.spec.solref,
            solimp: g.spec.solimp,
            solmix: g.spec.solmix,
            priority: g.spec.priority,
            margin: g.spec.margin,
            gap: g.spec.gap,
        });
    }

    // ---- contact exclusions: the body names resolve now that the bodies exist; the name
    // `world` is the world body
    let body_by_name: HashMap<&str, usize> = imp
        .bodies
        .iter()
        .enumerate()
        .filter(|(_, b)| !b.name.is_empty())
        .map(|(i, b)| (b.name.as_str(), i))
        .collect();
    let mut contact_excludes: Vec<ContactExclude> = Vec::with_capacity(imp.excludes.len());
    for x in &imp.excludes {
        let resolve = |name: &str| -> std::result::Result<Option<BodyId>, MjcfError> {
            if name == "world" {
                return Ok(None);
            }
            body_by_name
                .get(name)
                .map(|&i| Some(BodyId(i as u32)))
                .ok_or_else(|| {
                    MjcfError::new(
                        MjcfErrorKind::UnknownReference,
                        x.line,
                        &x.path,
                        format!("unknown body '{name}' in an exclusion"),
                    )
                })
        };
        let (body1, body2) = (resolve(&x.body1)?, resolve(&x.body2)?);
        if body1 == body2 {
            return Err(MjcfError::new(
                MjcfErrorKind::Inconsistent,
                x.line,
                &x.path,
                "an exclusion needs two different bodies",
            )
            .into());
        }
        contact_excludes.push(ContactExclude {
            name: x.name.clone(),
            body1,
            body2,
        });
    }

    // ---- the model statistic (setStat), on the compiled frames at qpos0
    let mut sites: Vec<(usize, [f64; 3], [f64; 4])> = Vec::with_capacity(imp.sites.len());
    for site in &imp.sites {
        sites.push(compile_site(site, c, &material_by_name)?);
    }
    let mut stat = model_statistic(
        &imp,
        &bodies,
        &body_quats,
        &joints,
        &geom_order,
        &compiled,
        &sites,
    );
    // the document's <statistic> overrides the computed values (user_model.cc:5806-5811)
    if let Some(extent) = imp.stat.extent {
        stat.extent = extent;
    }
    if let Some(meansize) = imp.stat.meansize {
        stat.meansize = meansize;
    }
    if let Some(center) = imp.stat.center {
        stat.center = center;
    }

    // ---- cameras, listed by body (the world's first), as MuJoCo numbers them
    let mut camera_order: Vec<usize> = (0..imp.cameras.len()).collect();
    camera_order.sort_by_key(|&i| imp.cameras[i].body.map_or(0, |b| b + 1));
    let mut cameras = Vec::with_capacity(camera_order.len());
    let mut recorded_cameras: Vec<Unsupported> = Vec::new();
    for (id, &ci) in camera_order.iter().enumerate() {
        let cam = &imp.cameras[ci];
        match compile_camera(cam, id, c, &body_by_name, imp.visual, stat.extent)? {
            CompiledCamera::Camera(camera) => cameras.push(camera),
            CompiledCamera::Recorded(reason) => {
                if options.strict {
                    return Err(MjcfError::new(
                        MjcfErrorKind::Strict,
                        cam.line,
                        &cam.path,
                        format!("element is not modelled yet ({reason}); strict import refuses it"),
                    )
                    .into());
                }
                recorded_cameras.push(Unsupported {
                    path: cam.path.clone(),
                    item: "element".to_string(),
                    line: cam.line,
                    reason,
                });
            }
        }
    }

    // ---- meshes
    let meshes: Vec<Mesh> = imp
        .meshes
        .iter()
        .map(|m| mesh_record(&m.name, &m.processed))
        .collect();

    // ---- actuators
    let mut actuators = Vec::with_capacity(imp.actuators.len());
    for a in &imp.actuators {
        actuators.push(compile_actuator(a, &joints, &joint_by_name, c.autolimits)?);
    }

    // ---- tendons
    let mut tendons = Vec::with_capacity(imp.tendons.len());
    let mut unsupported: Vec<Unsupported> = imp.recorded;
    unsupported.extend(recorded_cameras);
    for t in &imp.tendons {
        let tendon = compile_tendon(t, &joints, &joint_by_name, c.autolimits)?;
        // a fixed tendon's limit and friction loss are simulated; its spring, damper
        // and armature are carried but not simulated yet, so each one that is set is
        // recorded (a tendon that has none is not listed)
        for (item, value) in [
            ("@stiffness", tendon.stiffness),
            ("@damping", tendon.damping),
            ("@armature", tendon.armature),
        ] {
            if value == 0.0 {
                continue;
            }
            if options.strict {
                return Err(MjcfError::new(
                    MjcfErrorKind::Strict,
                    t.line,
                    &t.path,
                    format!(
                        "{item} of a fixed tendon is carried by the scene but not simulated yet; strict import refuses it"
                    ),
                )
                .into());
            }
            unsupported.push(Unsupported {
                path: t.path.clone(),
                item: item.to_string(),
                line: t.line,
                reason: "a fixed tendon's spring, damper and armature are carried by the scene but not simulated yet"
                    .to_string(),
            });
        }
        tendons.push(tendon);
    }

    // ---- instances: one per geom, segmentation ids 1..
    let mut instances = Vec::new();
    if options.instances {
        if geoms.len() > usize::from(u16::MAX) {
            return Err(SceneError::invalid(
                "instances",
                "more than 65535 geoms: segmentation ids are 16 bits",
            ));
        }
        for (i, g) in geoms.iter().enumerate() {
            instances.push(Instance {
                body: g.body,
                mesh_or_geom: ShapeRef::Geom(GeomId(i as u32)),
                material: g.material,
                local_pos: [0.0; 3],
                local_quat: [0.0, 0.0, 0.0, 1.0],
                seg_id: (i + 1) as u16,
            });
        }
    }

    let scene = Scene {
        version: SCENE_VERSION,
        name: imp.model_name,
        gravity: imp.gravity,
        timestep_s: imp.timestep,
        integrator: imp.integrator,
        options: imp.options,
        bodies,
        joints,
        geoms,
        meshes,
        materials,
        instances,
        cameras,
        actuators,
        tendons,
        contact_excludes,
        unsupported,
    };
    scene.validate()?;
    Ok((scene, stat))
}

/// `[x, y, z, w]` to `[w, x, y, z]`, exactly (a scene quaternion the importer wrote).
fn wxyz(q: [f64; 4]) -> [f64; 4] {
    [q[3], q[0], q[1], q[2]]
}

/// `std::max(a, b)`: `a < b ? b : a`.
fn std_max(a: f64, b: f64) -> f64 {
    if a < b { b } else { a }
}

/// The model statistic (see `statistic.rs`) of the compiled model: the bodies with
/// their inertial frames, the joints and geoms in scene order (MuJoCo's), each geom in
/// MuJoCo's frame (a mesh geom's moved to the mesh's centre of mass and principal
/// axes) with `mjCGeom::GetRBound` (user_objects.cc:3693-3730), and the sites.
fn model_statistic(
    imp: &Imported,
    bodies: &[Body],
    body_quats: &[[f64; 4]],
    joints: &[Joint],
    geom_order: &[usize],
    compiled: &[CompiledGeom],
    sites: &[(usize, [f64; 3], [f64; 4])],
) -> Statistic {
    let mut sbodies = Vec::with_capacity(bodies.len() + 1);
    sbodies.push(StatBody {
        parent: 0,
        pos: [0.0; 3],
        quat: [1.0, 0.0, 0.0, 0.0],
        ipos: [0.0; 3],
        iquat: [1.0, 0.0, 0.0, 0.0],
    });
    for (b, q) in bodies.iter().zip(body_quats) {
        let (ipos, iquat) = match &b.inertial {
            Some(i) => (i.com, wxyz(i.inertia_quat)),
            None => ([0.0; 3], [1.0, 0.0, 0.0, 0.0]),
        };
        sbodies.push(StatBody {
            parent: b.parent.map_or(0, |p| p.index() + 1),
            pos: b.pos,
            quat: *q,
            ipos,
            iquat,
        });
    }
    let sjoints: Vec<StatJoint> = joints
        .iter()
        .map(|j| StatJoint {
            body: j.body.index() + 1,
            kind: match j.kind {
                JointKind::Free => StatJointKind::Free,
                JointKind::Ball => StatJointKind::Ball,
                JointKind::Hinge { .. } => StatJointKind::Hinge,
                JointKind::Slide { axis } => StatJointKind::Slide { axis },
            },
            pos: j.pos,
        })
        .collect();
    let sgeoms: Vec<StatGeom> = geom_order
        .iter()
        .map(|&gi| {
            let g = &imp.geoms[gi];
            let cg = &compiled[gi];
            let (pos, quat) = match cg.shape {
                Shape::Mesh { .. } => (cg.ipos, cg.iquat),
                _ => (cg.pos, cg.quat),
            };
            let (rbound, plane) = match cg.shape {
                Shape::Sphere { r } => (r, None),
                Shape::Capsule { r, half_len } => (r + half_len, None),
                Shape::Cylinder { r, half_len } => ((r * r + half_len * half_len).sqrt(), None),
                Shape::Ellipsoid { radii } => {
                    (std_max(std_max(radii[0], radii[1]), radii[2]), None)
                }
                Shape::Box { half } => (
                    (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt(),
                    None,
                ),
                Shape::Plane { size } => (0.0, Some([size[0], size[1]])),
                Shape::Mesh { mesh } => (imp.meshes[mesh.index()].processed.rbound, None),
            };
            StatGeom {
                body: g.body.map_or(0, |b| b + 1),
                pos,
                quat,
                rbound,
                plane,
            }
        })
        .collect();
    // sites listed by body, as MuJoCo numbers them
    let mut site_order: Vec<usize> = (0..sites.len()).collect();
    site_order.sort_by_key(|&i| sites[i].0);
    let ssites: Vec<StatSite> = site_order
        .iter()
        .map(|&i| StatSite {
            body: sites[i].0,
            pos: sites[i].1,
            quat: sites[i].2,
        })
        .collect();
    statistic(&sbodies, &sjoints, &sgeoms, &ssites)
}

fn site_err(s: &RawSite, message: impl Into<String>) -> MjcfError {
    MjcfError::new(MjcfErrorKind::Inconsistent, s.line, &s.path, message)
}

/// `mjCSite::Compile`: the site's body (0 is the world), position and orientation
/// (`[w, x, y, z]`), with MuJoCo's checks.
fn compile_site(
    site: &RawSite,
    c: &Compiler,
    material_by_name: &HashMap<&str, usize>,
) -> std::result::Result<(usize, [f64; 3], [f64; 4]), MjcfError> {
    let spec = &site.spec;
    match spec.ty {
        SiteType::Plane | SiteType::Hfield => {
            return Err(site_err(site, "hfields and planes not allowed in site"));
        }
        SiteType::Mesh => {
            return Err(site_err(
                site,
                format!(
                    "mesh site '{}' needs a mesh, and mesh sites are not supported by this importer",
                    site.name
                ),
            ));
        }
        _ => {}
    }
    if let Some(m) = &spec.material
        && !material_by_name.contains_key(m.as_str())
    {
        return Err(MjcfError::new(
            MjcfErrorKind::UnknownReference,
            site.line,
            &site.path,
            format!("unknown material '{m}'"),
        ));
    }
    let mut size = spec.size;
    let mut pos = spec.pos;
    let mut quat = spec.quat;
    if !spec.fromto[0].is_nan() {
        if !matches!(
            spec.ty,
            SiteType::Capsule | SiteType::Cylinder | SiteType::Ellipsoid | SiteType::Box
        ) {
            return Err(site_err(
                site,
                "fromto requires capsule, cylinder, box or ellipsoid in site",
            ));
        }
        if pos != [0.0; 3] {
            return Err(site_err(site, "both pos and fromto defined in site"));
        }
        let f = spec.fromto;
        let mut vec = [f[0] - f[3], f[1] - f[4], f[2] - f[5]];
        size[1] = normvec(&mut vec) / 2.0;
        if size[1] < EPS {
            return Err(site_err(site, "fromto points too close in site"));
        }
        if matches!(spec.ty, SiteType::Ellipsoid | SiteType::Box) {
            size[2] = size[1];
            size[1] = size[0];
        }
        pos = [
            (f[0] + f[3]) / 2.0,
            (f[1] + f[4]) / 2.0,
            (f[2] + f[5]) / 2.0,
        ];
        quat = z2quat(vec);
    } else {
        resolve_orientation(&mut quat, c.degree, &c.eulerseq, &spec.alt).map_err(|err| {
            MjcfError::new(
                MjcfErrorKind::BadValue,
                site.line,
                &site.path,
                format!("orientation specification error '{err}' in site"),
            )
        })?;
    }
    normvec(&mut quat);
    // checksize (user_objects.cc:186-199; mjGEOMINFO = sphere 1, capsule 2,
    // ellipsoid 3, cylinder 2, box 3, sdf 0)
    let n = match spec.ty {
        SiteType::Sphere => 1,
        SiteType::Capsule | SiteType::Cylinder => 2,
        SiteType::Ellipsoid | SiteType::Box => 3,
        _ => 0,
    };
    for (i, v) in size.iter().enumerate().take(n) {
        if *v <= 0.0 {
            return Err(site_err(site, format!("size {i} must be positive in site")));
        }
    }
    Ok((site.body.map_or(0, |b| b + 1), pos, quat))
}

/// What a `<camera>` compiles to.
enum CompiledCamera {
    Camera(Camera),
    /// Recorded (not modelled), with the reason.
    Recorded(String),
}

fn camera_err(cam: &RawCamera, kind: MjcfErrorKind, message: impl Into<String>) -> MjcfError {
    MjcfError::new(kind, cam.line, &cam.path, message)
}

/// `mjCCamera::Compile` and the renderer's projection: see the module note. `id` is
/// the camera's MuJoCo id, for MuJoCo's error texts.
fn compile_camera(
    cam: &RawCamera,
    id: usize,
    c: &Compiler,
    body_by_name: &HashMap<&str, usize>,
    visual: Visual,
    extent: f64,
) -> std::result::Result<CompiledCamera, MjcfError> {
    let spec = &cam.spec;
    let name = &cam.name;

    // orientation, then normalisation (mjCCamera::Compile)
    let mut quat = spec.quat;
    resolve_orientation(&mut quat, c.degree, &c.eulerseq, &spec.alt).map_err(|err| {
        camera_err(
            cam,
            MjcfErrorKind::BadValue,
            format!("orientation specification error '{err}' in camera {id}"),
        )
    })?;
    normvec(&mut quat);

    // the target body (ResolveReferences); the world body is named `world`
    if let Some(t) = &spec.target
        && t != "world"
        && !body_by_name.contains_key(t.as_str())
    {
        return Err(camera_err(
            cam,
            MjcfErrorKind::UnknownReference,
            format!("unknown target body '{t}' in camera"),
        ));
    }

    if spec.fovy >= 180.0 {
        return Err(camera_err(
            cam,
            MjcfErrorKind::BadValue,
            format!(
                "fovy too large in camera '{name}' (id = {id}, value = {}): it must be below 180 degrees",
                spec.fovy
            ),
        ));
    }

    // the intrinsics (float arithmetic, as MuJoCo's compiler)
    let has_intrinsic = spec.focal != [0.0; 2]
        || spec.focalpixel != [0.0; 2]
        || spec.principal != [0.0; 2]
        || spec.principalpixel != [0.0; 2];
    let has_sensorsize = spec.sensorsize[0] > 0.0 && spec.sensorsize[1] > 0.0;
    if has_intrinsic && !has_sensorsize {
        return Err(camera_err(
            cam,
            MjcfErrorKind::Inconsistent,
            format!("focal/principal require sensorsize in camera '{name}' (id = {id})"),
        ));
    }
    if has_sensorsize && (spec.resolution[0] <= 0 || spec.resolution[1] <= 0) {
        return Err(camera_err(
            cam,
            MjcfErrorKind::Inconsistent,
            format!("sensorsize requires positive resolution in camera '{name}' (id = {id})"),
        ));
    }
    let mut intrinsic = [0f32; 4];
    if has_sensorsize {
        let density = [
            spec.resolution[0] as f32 / spec.sensorsize[0],
            spec.resolution[1] as f32 / spec.sensorsize[1],
        ];
        let pick = |pixel: f32, length: f32, d: f32| if pixel != 0.0 { pixel / d } else { length };
        intrinsic = [
            pick(spec.focalpixel[0], spec.focal[0], density[0]),
            pick(spec.focalpixel[1], spec.focal[1], density[1]),
            pick(spec.principalpixel[0], spec.principal[0], density[0]),
            pick(spec.principalpixel[1], spec.principal[1], density[1]),
        ];
    }

    // what the scene does not model is recorded (after MuJoCo's own checks, so a
    // camera MuJoCo refuses is refused here too)
    if spec.mode != CameraMode::Fixed {
        return Ok(CompiledCamera::Recorded(format!(
            "a camera in mode '{}' moves on its own as the model moves; only fixed cameras are imported",
            spec.mode.keyword()
        )));
    }
    if spec.projection == Projection::Orthographic {
        return Ok(CompiledCamera::Recorded(
            "an orthographic camera; a scene camera is a pinhole (perspective) camera".to_string(),
        ));
    }

    // what MuJoCo compiles but cannot render as a pinhole is refused
    let degenerate = |message: String| Err(camera_err(cam, MjcfErrorKind::BadValue, message));
    if spec.sensorsize != [0.0; 2] && !has_sensorsize {
        return degenerate(format!(
            "camera '{name}': sensorsize {:?} must be positive in both directions",
            spec.sensorsize
        ));
    }
    if has_sensorsize && (intrinsic[0] <= 0.0 || intrinsic[1] <= 0.0) {
        return degenerate(format!(
            "camera '{name}': a sensorsize camera needs positive focal lengths (focal or focalpixel), not {:?}",
            [intrinsic[0], intrinsic[1]]
        ));
    }
    if !has_sensorsize && spec.fovy <= 0.0 {
        return degenerate(format!(
            "camera '{name}': fovy must be positive, not {}",
            spec.fovy
        ));
    }
    let (width, height) = if spec.resolution[0] > 1 || spec.resolution[1] > 1 {
        if spec.resolution[0] <= 0 || spec.resolution[1] <= 0 {
            return degenerate(format!(
                "camera '{name}': resolution {:?} must be positive",
                spec.resolution
            ));
        }
        (spec.resolution[0] as u32, spec.resolution[1] as u32)
    } else {
        if visual.offwidth <= 0 || visual.offheight <= 0 {
            return degenerate(format!(
                "camera '{name}' has no resolution and the offscreen buffer (visual/global offwidth, offheight) {} x {} is not positive",
                visual.offwidth, visual.offheight
            ));
        }
        (visual.offwidth as u32, visual.offheight as u32)
    };
    let near = (f64::from(visual.znear) * extent) as f32;
    let far = (f64::from(visual.zfar) * extent) as f32;
    if !(near > 0.0 && far > near && far.is_finite()) {
        return degenerate(format!(
            "camera '{name}': the clip planes (visual/map znear {} and zfar {} times the extent {extent}) give near {near} and far {far}; a camera needs 0 < near < far",
            visual.znear, visual.zfar
        ));
    }

    let (w, h) = (f64::from(width), f64::from(height));
    let (fx, fy, cx, cy) = if has_sensorsize {
        let [fl_x, fl_y, p_x, p_y] = intrinsic.map(f64::from);
        let [s_x, s_y] = spec.sensorsize.map(f64::from);
        (
            w * fl_x / s_x,
            h * fl_y / s_y,
            w / 2.0 - w * p_x / s_x - 0.5,
            h / 2.0 - h * p_y / s_y - 0.5,
        )
    } else {
        let f = h / (2.0 * (spec.fovy * PI / 360.0).tan());
        (f, f, (w - 1.0) / 2.0, (h - 1.0) / 2.0)
    };

    // MuJoCo's camera frame to the scene's OpenCV frame: q * (0, 1, 0, 0), then [x, y, z, w]
    let [qw, qx, qy, qz] = quat;
    let q_cv = [qw, qz, -qy, -qx];
    let mount = match cam.body {
        None => CameraMount::World {
            pos: spec.pos,
            quat: q_cv,
        },
        Some(b) => CameraMount::Body {
            body: BodyId(b as u32),
            local_pos: spec.pos,
            local_quat: q_cv,
        },
    };
    Ok(CompiledCamera::Camera(Camera {
        name: name.clone(),
        mount,
        fx: fx as f32,
        fy: fy as f32,
        cx: cx as f32,
        cy: cy as f32,
        near,
        far,
        width,
        height,
    }))
}

fn mesh_record(name: &str, p: &ProcessedMesh) -> Mesh {
    Mesh {
        name: name.to_string(),
        vertices: p.vertices.clone(),
        triangles: p.faces.clone(),
    }
}

fn geom_err(g: &RawGeom, message: impl Into<String>) -> MjcfError {
    MjcfError::new(MjcfErrorKind::Inconsistent, g.line, &g.path, message)
}

/// `mjCGeom::Compile`.
fn compile_geom(
    g: &RawGeom,
    infer: bool,
    imp: &Imported,
    mesh_ids: &HashMap<&str, usize>,
) -> std::result::Result<CompiledGeom, MjcfError> {
    let c = &imp.compiler;
    let spec = &g.spec;
    if !matches!(spec.condim, 1 | 3 | 4 | 6) {
        return Err(MjcfError::new(
            MjcfErrorKind::BadValue,
            g.line,
            &g.path,
            "invalid condim in geom",
        ));
    }

    // mesh reference
    let mesh: Option<(usize, &ProcessedMesh)> = match (&spec.mesh, spec.ty) {
        (Some(name), GeomType::Mesh) => {
            let idx = mesh_ids.get(name.as_str()).copied().ok_or_else(|| {
                MjcfError::new(
                    MjcfErrorKind::UnknownReference,
                    g.line,
                    &g.path,
                    format!("unknown mesh '{name}'"),
                )
            })?;
            Some((idx, &imp.meshes[idx].processed))
        }
        (None, GeomType::Mesh) => {
            return Err(MjcfError::new(
                MjcfErrorKind::MissingAttribute,
                g.line,
                &g.path,
                "mesh geom must have a mesh attribute",
            ));
        }
        (Some(_), _) => {
            return Err(MjcfError::new(
                MjcfErrorKind::UnsupportedAttribute,
                g.line,
                &g.path,
                "a primitive geom that references a mesh (fitting a primitive to a mesh) is not supported",
            ));
        }
        (None, _) => None,
    };

    let mut quat = spec.quat;
    normvec(&mut quat);
    let mut size = spec.size;
    let mut pos = spec.pos;

    if !spec.fromto[0].is_nan() {
        if !matches!(
            spec.ty,
            GeomType::Capsule | GeomType::Cylinder | GeomType::Ellipsoid | GeomType::Box
        ) {
            return Err(geom_err(
                g,
                "fromto requires capsule, cylinder, box or ellipsoid in geom",
            ));
        }
        if pos != [0.0; 3] {
            return Err(geom_err(g, "both pos and fromto defined in geom"));
        }
        let f = spec.fromto;
        let mut vec = [f[0] - f[3], f[1] - f[4], f[2] - f[5]];
        size[1] = normvec(&mut vec) / 2.0;
        if size[1] < EPS {
            return Err(geom_err(g, "fromto points too close in geom"));
        }
        if matches!(spec.ty, GeomType::Ellipsoid | GeomType::Box) {
            size[2] = size[1];
            size[1] = size[0];
        }
        pos = [
            (f[0] + f[3]) / 2.0,
            (f[1] + f[4]) / 2.0,
            (f[2] + f[5]) / 2.0,
        ];
        quat = z2quat(vec);
    } else {
        resolve_orientation(&mut quat, c.degree, &c.eulerseq, &spec.alt).map_err(|err| {
            MjcfError::new(
                MjcfErrorKind::BadValue,
                g.line,
                &g.path,
                format!("orientation specification error '{err}' in geom"),
            )
        })?;
    }

    // mesh: accumulate the mesh frame into the inertial frame (the authored
    // pose stays as written; see mesh.rs)
    let mut ipos = pos;
    let mut iquat = quat;
    if let Some((_, pm)) = mesh {
        if !spec.fromto[0].is_nan() {
            return Err(geom_err(g, "fromto cannot be used with mesh geom"));
        }
        frameaccum(&mut ipos, &mut iquat, pm.com, pm.quat);
    }

    // checksize (user_objects.cc:186-199; mjGEOMINFO = plane 3, sphere 1,
    // capsule 2, ellipsoid 3, cylinder 2, box 3, mesh 0)
    match spec.ty {
        GeomType::Plane => {
            if size[2] <= 0.0 {
                return Err(geom_err(g, "plane size(3) must be positive"));
            }
        }
        other => {
            let n = match other {
                GeomType::Sphere => 1,
                GeomType::Capsule | GeomType::Cylinder => 2,
                GeomType::Ellipsoid | GeomType::Box => 3,
                _ => 0,
            };
            for (i, s) in size.iter().enumerate().take(n) {
                if *s <= 0.0 {
                    return Err(geom_err(g, format!("size {i} must be positive in geom")));
                }
            }
        }
    }

    // mass and inertia
    let mut mass = 0.0;
    let mut inertia = [0.0; 3];
    let mut density = spec.density;
    if infer {
        let pm = mesh.map(|(_, pm)| pm);
        if !spec.mass.is_nan() {
            if spec.mass == 0.0 {
                mass = 0.0;
                density = 0.0;
            } else if volume(spec.ty, size, pm) > EPS {
                mass = spec.mass;
                density = spec.mass / volume(spec.ty, size, pm);
                inertia = diag_inertia(spec.ty, size, mass, pm);
            }
        } else if density == 0.0 {
            mass = 0.0;
        } else {
            mass = density * volume(spec.ty, size, pm);
            inertia = diag_inertia(spec.ty, size, mass, pm);
        }
        if mass < 0.0 || inertia.iter().any(|&i| i < 0.0) || density < 0.0 {
            return Err(geom_err(g, "mass, inertia or density are negative in geom"));
        }
    }

    let shape = match spec.ty {
        GeomType::Plane => Shape::Plane { size },
        GeomType::Sphere => Shape::Sphere { r: size[0] },
        GeomType::Capsule => Shape::Capsule {
            r: size[0],
            half_len: size[1],
        },
        GeomType::Cylinder => Shape::Cylinder {
            r: size[0],
            half_len: size[1],
        },
        GeomType::Ellipsoid => Shape::Ellipsoid { radii: size },
        GeomType::Box => Shape::Box { half: size },
        GeomType::Mesh => Shape::Mesh {
            mesh: MeshId(mesh.map_or(0, |(i, _)| i) as u32),
        },
    };
    Ok(CompiledGeom {
        pos,
        quat,
        ipos,
        iquat,
        mass,
        inertia,
        density,
        shape,
    })
}

/// The inertial of one body: `mjCBody::Compile`'s inertia handling.
fn compile_inertial(
    b: &RawBody,
    geoms: &[(&RawGeom, &CompiledGeom)],
    c: &super::import::Compiler,
) -> Result<Option<Inertial>> {
    let berr = |message: String| -> SceneError {
        MjcfError::new(MjcfErrorKind::Inconsistent, b.line, &b.path, message).into()
    };

    // explicit <inertial>
    let mut defined = false;
    let mut mass = 0.0f64;
    let mut ipos = [0.0f64; 3];
    let mut iquat = [1.0f64, 0.0, 0.0, 0.0];
    let mut inertia = [0.0f64; 3];
    if let Some(x) = &b.explicit {
        defined = true;
        mass = x.mass;
        ipos = x.ipos;
        inertia = x.inertia;
        iquat = x.iquat;
        normvec(&mut iquat);
        if let Some(full) = x.fullinertia {
            // rotate the tensor from the inertial frame to the body frame, then
            // decompose it into principal axes
            let mat = quat2mat(iquat);
            let m = [
                full[0], full[3], full[4], full[3], full[1], full[5], full[4], full[5], full[2],
            ];
            let body = mul_rmrt(&mat, &m);
            let body6 = [body[0], body[4], body[8], body[1], body[2], body[5]];
            let (q, i) = full_inertia(body6).map_err(|e| {
                SceneError::from(MjcfError::new(
                    MjcfErrorKind::Inconsistent,
                    x.line,
                    &x.path,
                    format!("error '{e}' in fullinertia"),
                ))
            })?;
            iquat = q;
            inertia = i;
        }
        if x.ialt.kind != OrientKind::Quat {
            resolve_orientation(&mut iquat, c.degree, &c.eulerseq, &x.ialt).map_err(|e| {
                SceneError::from(MjcfError::new(
                    MjcfErrorKind::BadValue,
                    x.line,
                    &x.path,
                    format!("error '{e}' in inertia alternative"),
                ))
            })?;
        }
    }

    // inertial frame from geoms
    let mut derived = false;
    if c.inertiafromgeom == Limited::True || (!defined && c.inertiafromgeom == Limited::Auto) {
        // select geoms by group and mass (InertiaFromGeom)
        let selected: Vec<&CompiledGeom> = geoms
            .iter()
            .filter(|(raw, g)| {
                raw.spec.group >= c.inertiagrouprange[0]
                    && raw.spec.group <= c.inertiagrouprange[1]
                    && g.mass > EPS
            })
            .map(|(_, g)| *g)
            .collect();
        match selected.len() {
            0 => {}
            1 => {
                let g = selected[0];
                ipos = g.ipos;
                iquat = g.iquat;
                mass = g.mass;
                inertia = g.inertia;
                derived = true;
            }
            _ => {
                let mut total = 0.0;
                let mut com = [0.0; 3];
                for g in &selected {
                    total += g.mass;
                    com[0] += g.mass * g.ipos[0];
                    com[1] += g.mass * g.ipos[1];
                    com[2] += g.mass * g.ipos[2];
                }
                if total < EPS {
                    return Err(berr(
                        "body mass is too small, cannot compute center of mass".into(),
                    ));
                }
                mass = total;
                ipos = [com[0] / mass, com[1] / mass, com[2] / mass];
                let mut toti = [0.0f64; 6];
                for g in &selected {
                    let dpos = [
                        g.ipos[0] - ipos[0],
                        g.ipos[1] - ipos[1],
                        g.ipos[2] - ipos[2],
                    ];
                    let inert0 = globalinertia(g.inertia, g.iquat);
                    let inert1 = offcenter(g.mass, dpos);
                    for j in 0..6 {
                        toti[j] = toti[j] + inert0[j] + inert1[j];
                    }
                }
                let (q, i) = full_inertia(toti)
                    .map_err(|e| berr(format!("error '{e}' in alternative for principal axes")))?;
                iquat = q;
                inertia = i;
                derived = true;
            }
        }
    }

    // check and correct mass and inertia (boundmass = boundinertia = 0)
    mass = mass.max(0.0);
    for i in inertia.iter_mut() {
        *i = i.max(0.0);
    }
    if inertia[0] + inertia[1] < inertia[2]
        || inertia[0] + inertia[2] < inertia[1]
        || inertia[1] + inertia[2] < inertia[0]
    {
        return Err(berr(
            "inertia must satisfy A + B >= C; balanceinertia is not supported".into(),
        ));
    }

    if defined || derived {
        Ok(Some(Inertial {
            mass_kg: mass,
            com: ipos,
            diag_inertia: inertia,
            inertia_quat: xyzw(iquat),
        }))
    } else {
        Ok(None)
    }
}

/// `mjCJoint::Compile`.
fn compile_joint(
    j: &RawJoint,
    degree: bool,
    autolimits: bool,
) -> std::result::Result<Joint, MjcfError> {
    let jerr =
        |message: &str| MjcfError::new(MjcfErrorKind::Inconsistent, j.line, &j.path, message);
    let spec = &j.spec;
    let mut limited = spec.limited;
    let mut range = spec.range;

    if spec.ty == JointType::Free {
        limited = Limited::False;
    } else if limited == Limited::Auto {
        let hasrange = !(range[0] == 0.0 && range[1] == 0.0);
        check_limited(
            (j.line, &j.path),
            autolimits,
            "joint",
            "",
            limited,
            hasrange,
        )?;
    }

    let limit = if is_limited(limited, range) {
        if range[0] >= range[1] && spec.ty != JointType::Ball {
            return Err(jerr("range[0] should be smaller than range[1] in joint"));
        }
        if range[0] != 0.0 && spec.ty == JointType::Ball {
            return Err(jerr("range[0] should be 0 in ball joint"));
        }
        if degree && matches!(spec.ty, JointType::Hinge | JointType::Ball) {
            if range[0] != 0.0 {
                range[0] *= PI / 180.0;
            }
            if range[1] != 0.0 {
                range[1] *= PI / 180.0;
            }
        }
        Some(range)
    } else {
        None
    };

    let mut axis = spec.axis;
    if !matches!(spec.ty, JointType::Free | JointType::Ball) && normvec(&mut axis) < EPS {
        return Err(jerr("axis too small in joint"));
    }

    let kind = match spec.ty {
        JointType::Free => JointKind::Free,
        JointType::Ball => JointKind::Ball,
        JointType::Hinge => JointKind::Hinge { axis },
        JointType::Slide => JointKind::Slide { axis },
    };
    Ok(Joint {
        name: j.name.clone(),
        body: BodyId(j.body as u32),
        kind,
        pos: if spec.ty == JointType::Free {
            [0.0; 3]
        } else {
            spec.pos
        },
        range: limit,
        stiffness: spec.stiffness,
        damping: spec.damping,
        armature: spec.armature,
        frictionloss: spec.frictionloss,
        solref_limit: spec.solref_limit,
        solimp_limit: spec.solimp_limit,
        solref_friction: spec.solref_friction,
        solimp_friction: spec.solimp_friction,
        margin: spec.margin,
    })
}

fn joint_ref(
    path: &str,
    line: u32,
    name: &str,
    joints: &[Joint],
    by_name: &HashMap<&str, usize>,
) -> std::result::Result<usize, MjcfError> {
    let idx = by_name.get(name).copied().ok_or_else(|| {
        MjcfError::new(
            MjcfErrorKind::UnknownReference,
            line,
            path,
            format!("unknown joint '{name}'"),
        )
    })?;
    if !matches!(
        joints[idx].kind,
        JointKind::Hinge { .. } | JointKind::Slide { .. }
    ) {
        return Err(MjcfError::new(
            MjcfErrorKind::Inconsistent,
            line,
            path,
            format!(
                "joint '{name}' is a free or ball joint; only hinge and slide joints are supported here"
            ),
        ));
    }
    Ok(idx)
}

/// `mjCActuator::Compile`, the limit logic.
fn compile_actuator(
    a: &RawActuator,
    joints: &[Joint],
    by_name: &HashMap<&str, usize>,
    autolimits: bool,
) -> std::result::Result<Actuator, MjcfError> {
    let joint = joint_ref(&a.path, a.line, &a.joint, joints, by_name)?;
    let spec = &a.spec;
    if spec.gear[1..].iter().any(|&g| g != 0.0) {
        return Err(MjcfError::new(
            MjcfErrorKind::Inconsistent,
            a.line,
            &a.path,
            "gear[1..] is non-zero, which only applies to ball and free joints; hinge and slide joints use gear[0]",
        ));
    }
    if spec.ctrllimited == Limited::Auto {
        let hasrange = !(spec.ctrlrange[0] == 0.0 && spec.ctrlrange[1] == 0.0);
        check_limited(
            (a.line, &a.path),
            autolimits,
            "actuator",
            "ctrl",
            spec.ctrllimited,
            hasrange,
        )?;
    }
    let limited = is_limited(spec.ctrllimited, spec.ctrlrange);
    if spec.ctrlrange[0] >= spec.ctrlrange[1] && limited {
        return Err(MjcfError::new(
            MjcfErrorKind::Inconsistent,
            a.line,
            &a.path,
            "invalid control range for actuator",
        ));
    }
    let ctrlrange = limited.then_some(spec.ctrlrange);
    let kind = match a.tag {
        ActuatorTag::Motor => ActuatorKind::Motor {
            gear: spec.gear[0],
            ctrlrange,
        },
        ActuatorTag::Position => ActuatorKind::Position {
            kp: spec.kp,
            gear: spec.gear[0],
            ctrlrange,
        },
    };
    Ok(Actuator {
        name: a.name.clone(),
        joint: JointId(joint as u32),
        kind,
    })
}

/// `mjCTendon::Compile`, the limit logic, for a fixed tendon.
fn compile_tendon(
    t: &RawTendon,
    joints: &[Joint],
    by_name: &HashMap<&str, usize>,
    autolimits: bool,
) -> std::result::Result<Tendon, MjcfError> {
    let spec = &t.spec;
    let mut terms = Vec::with_capacity(t.joints.len());
    for (name, coef, line) in &t.joints {
        let idx = joint_ref(&t.path, *line, name, joints, by_name)?;
        terms.push(TendonJoint {
            joint: JointId(idx as u32),
            coef: *coef,
        });
    }
    if spec.limited == Limited::Auto {
        let hasrange = !(spec.range[0] == 0.0 && spec.range[1] == 0.0);
        check_limited(
            (t.line, &t.path),
            autolimits,
            "tendon",
            "",
            spec.limited,
            hasrange,
        )?;
    }
    let limited = is_limited(spec.limited, spec.range);
    if spec.range[0] >= spec.range[1] && limited {
        return Err(MjcfError::new(
            MjcfErrorKind::Inconsistent,
            t.line,
            &t.path,
            "invalid limits in tendon",
        ));
    }
    Ok(Tendon {
        name: t.name.clone(),
        joints: terms,
        range: limited.then_some(spec.range),
        stiffness: spec.stiffness,
        damping: spec.damping,
        armature: spec.armature,
        frictionloss: spec.frictionloss,
        solref_limit: spec.solref_limit,
        solimp_limit: spec.solimp_limit,
        solref_friction: spec.solref_friction,
        solimp_friction: spec.solimp_friction,
        margin: spec.margin,
    })
}

/// An asset `<material>` as a scene material.
fn asset_material(name: &str, spec: &super::specs::MaterialSpec) -> Material {
    let mut m = Material::named(name);
    let rgb = [spec.rgba[0], spec.rgba[1], spec.rgba[2]];
    let linear = rgb.map(srgb_to_linear);
    m.optical.base_colour_linear = linear;
    if spec.metallic >= 0.0 {
        m.optical.metallic = spec.metallic;
    }
    if spec.roughness >= 0.0 {
        m.optical.roughness = spec.roughness;
    }
    m.optical.emission_linear = linear.map(|c| c * spec.emission);
    m
}

/// The material of a geom: its named material, or a synthesised one for an
/// `rgba` that overrides it (MuJoCo's rule: an `rgba` different from the
/// default takes precedence over the material's, the rest of the material
/// still applies).
fn resolve_material(
    g: &RawGeom,
    by_name: &HashMap<&str, usize>,
    materials: &mut Vec<Material>,
    interned: &mut HashMap<(Option<usize>, [u32; 3]), usize>,
) -> std::result::Result<usize, MjcfError> {
    let named = match &g.spec.material {
        None => None,
        Some(n) => Some(by_name.get(n.as_str()).copied().ok_or_else(|| {
            MjcfError::new(
                MjcfErrorKind::UnknownReference,
                g.line,
                &g.path,
                format!("unknown material '{n}'"),
            )
        })?),
    };
    let rgba = g.spec.rgba;
    let overridden = rgba != DEFAULT_RGBA;
    if let (Some(m), false) = (named, overridden) {
        return Ok(m);
    }
    let rgb = [rgba[0], rgba[1], rgba[2]];
    let key = (named, rgb.map(f32::to_bits));
    if let Some(&idx) = interned.get(&key) {
        return Ok(idx);
    }
    let mut material = match named {
        Some(m) => materials[m].clone(),
        None => Material::named(String::new()),
    };
    let base = match named {
        Some(m) => materials[m].name.clone(),
        None => String::new(),
    };
    let tag = format!("rgba({},{},{})", rgb[0], rgb[1], rgb[2]);
    material.name = if base.is_empty() {
        tag
    } else {
        format!("{base}+{tag}")
    };
    material.optical.base_colour_linear = rgb.map(srgb_to_linear);
    materials.push(material);
    let idx = materials.len() - 1;
    interned.insert(key, idx);
    Ok(idx)
}
