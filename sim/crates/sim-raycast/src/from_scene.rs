//! From the simulator's ONE scene description (`sim_scene::Scene`) to the renderer's
//! [`SceneDesc`].
//!
//! The simulator keeps one scene description for physics, rendering and the
//! senses. The renderer reads its bodies, geoms, meshes, the OPTICAL part of
//! its materials (glTF metallic-roughness semantics, owned by the renderer) and its
//! instances:
//! - an instance of a MESH draws `Scene::meshes[m]`, flat-shaded (meshes carry
//!   no normals), at its local pose in its body;
//! - an instance of a GEOM draws the geom's analytic shape, tessellated here,
//!   at the geom's pose in its body followed by the instance's local pose (the
//!   order `sim_scene::Instance` documents); a geom whose shape is a mesh
//!   reuses that mesh;
//! - a static instance (no body) is placed in the world by its local pose;
//!   body `b` is `Scene::bodies[b]`, and the world has no entry (the physics'
//!   convention), so `bodies_per_env` is `Scene::bodies.len()`.
//!
//! Optical values map one to one, except perceptual roughness, which is raised
//! to at least 0.045 because GGX's distribution is singular at 0 (a render-side
//! clamp). The scene has no lights yet, so the
//! lighting is the caller's.

use sim_scene::{Scene, Shape, ShapeRef};

use crate::SceneError;
use crate::math::{Pose, Vec3};
use crate::mesh::Mesh;
use crate::scene::{Instance, Lighting, Material, SceneDesc};

/// The smallest perceptual roughness the renderer shades.
pub const MIN_ROUGHNESS: f32 = 0.045;

/// How finely analytic shapes are tessellated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tessellation {
    /// Segments around a round shape's axis.
    pub segments: u32,
    /// Rings from pole to pole of a sphere or ellipsoid (half of them per
    /// capsule cap).
    pub rings: u32,
    /// Cells per side of a plane. One cell (two triangles) is the default: a
    /// plane's colour is procedural, and its tree is then a single leaf
    /// (1.28x faster for the tabletop's floor, 2026-10-01).
    pub plane_cells: u32,
    /// The half extent, metres, a plane MuJoCo calls infinite (size 0) is
    /// drawn with.
    pub infinite_plane_half: f32,
}

impl Default for Tessellation {
    fn default() -> Self {
        Self {
            segments: 32,
            rings: 16,
            plane_cells: 1,
            infinite_plane_half: 500.0,
        }
    }
}

fn narrow3(v: [f64; 3]) -> Vec3 {
    v.map(|x| x as f32)
}

fn quat_mul64(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[3] * b[0] + b[3] * a[0] + (a[1] * b[2] - a[2] * b[1]),
        a[3] * b[1] + b[3] * a[1] + (a[2] * b[0] - a[0] * b[2]),
        a[3] * b[2] + b[3] * a[2] + (a[0] * b[1] - a[1] * b[0]),
        a[3] * b[3] - (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]),
    ]
}

fn rotate64(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let u = [q[0], q[1], q[2]];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let t = cross(u, v).map(|x| 2.0 * x);
    let c = cross(u, t);
    [
        v[0] + q[3] * t[0] + c[0],
        v[1] + q[3] * t[1] + c[1],
        v[2] + q[3] * t[2] + c[2],
    ]
}

/// `outer * inner` in f64, narrowed to f32 at the end.
fn compose(outer: ([f64; 3], [f64; 4]), inner: ([f64; 3], [f64; 4])) -> Pose {
    let p = rotate64(outer.1, inner.0);
    let q = quat_mul64(outer.1, inner.1);
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    Pose::new(
        narrow3([p[0] + outer.0[0], p[1] + outer.0[1], p[2] + outer.0[2]]),
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n].map(|x| x as f32),
    )
}

fn tessellate(shape: &Shape, t: &Tessellation) -> Option<Mesh> {
    let f = |x: f64| x as f32;
    Some(match *shape {
        Shape::Sphere { r } => Mesh::sphere(f(r), t.segments, t.rings),
        Shape::Capsule { r, half_len } => {
            Mesh::capsule(f(r), f(half_len), t.segments, t.rings.div_ceil(2))
        }
        Shape::Box { half } => Mesh::cuboid(narrow3(half)),
        Shape::Plane { size } => {
            let half = |x: f64| if x > 0.0 { f(x) } else { t.infinite_plane_half };
            Mesh::plane(half(size[0]), half(size[1]), t.plane_cells, t.plane_cells)
        }
        Shape::Cylinder { r, half_len } => Mesh::cylinder(f(r), f(half_len), t.segments),
        Shape::Ellipsoid { radii } => Mesh::ellipsoid(narrow3(radii), t.segments, t.rings),
        Shape::Mesh { .. } => return None,
    })
}

/// What the conversion did, for the caller to report.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Conversion {
    /// Degenerate (zero-area) triangles dropped from imported meshes.
    pub degenerate_triangles_dropped: usize,
    /// Materials whose roughness was raised to [`MIN_ROUGHNESS`].
    pub roughness_clamped: usize,
}

/// Convert `scene` into the renderer's description. Refuses an instance that
/// names a mesh, geom or material the scene does not have.
pub fn scene_desc(
    scene: &Scene,
    lighting: Lighting,
    tess: &Tessellation,
) -> Result<(SceneDesc, Conversion), SceneError> {
    let bad = |why: String| SceneError(why);
    let mut report = Conversion::default();
    let mut meshes = Vec::new();
    for m in &scene.meshes {
        let (mesh, dropped) = Mesh::flat(&m.vertices, &m.triangles);
        report.degenerate_triangles_dropped += dropped;
        meshes.push(mesh);
    }
    let materials = scene
        .materials
        .iter()
        .map(|m| {
            let o = &m.optical;
            if o.roughness < MIN_ROUGHNESS {
                report.roughness_clamped += 1;
            }
            Material {
                base_colour: o.base_colour_linear,
                roughness: o.roughness.max(MIN_ROUGHNESS),
                metallic: o.metallic,
                checker_cells_per_metre: o.checker.map_or(0.0, |c| c.cells_per_m),
                checker_dark: o.checker.map_or(1.0, |c| c.dark_multiplier),
                emission: o.emission_linear,
            }
        })
        .collect();
    // geom index -> mesh index, made on first use
    let mut geom_mesh: Vec<Option<u32>> = vec![None; scene.geoms.len()];
    let mut instances = Vec::with_capacity(scene.instances.len());
    for (i, inst) in scene.instances.iter().enumerate() {
        let local = (inst.local_pos, inst.local_quat);
        let (mesh, offset) = match inst.mesh_or_geom {
            ShapeRef::Mesh(id) => {
                if id.index() >= meshes.len() {
                    return Err(bad(format!(
                        "instance {i} names mesh {}, of {}",
                        id.index(),
                        meshes.len()
                    )));
                }
                (
                    id.index() as u32,
                    compose(([0.0; 3], [0.0, 0.0, 0.0, 1.0]), local),
                )
            }
            ShapeRef::Geom(id) => {
                let g = scene.geoms.get(id.index()).ok_or_else(|| {
                    bad(format!(
                        "instance {i} names geom {}, of {}",
                        id.index(),
                        scene.geoms.len()
                    ))
                })?;
                let mesh = match (geom_mesh[id.index()], &g.shape) {
                    (Some(m), _) => m,
                    (None, Shape::Mesh { mesh }) => {
                        if mesh.index() >= scene.meshes.len() {
                            return Err(bad(format!(
                                "geom {} names mesh {}",
                                id.index(),
                                mesh.index()
                            )));
                        }
                        mesh.index() as u32
                    }
                    (None, shape) => {
                        meshes.push(tessellate(shape, tess).expect("an analytic shape"));
                        (meshes.len() - 1) as u32
                    }
                };
                geom_mesh[id.index()] = Some(mesh);
                (mesh, compose((g.pos, g.quat), local))
            }
        };
        if inst.material.index() >= scene.materials.len() {
            return Err(bad(format!(
                "instance {i} names material {}",
                inst.material.index()
            )));
        }
        instances.push(Instance {
            body: inst.body.map(|b| b.0),
            mesh,
            material: inst.material.0,
            offset,
            seg_id: inst.seg_id,
        });
    }
    Ok((
        SceneDesc {
            meshes,
            materials,
            instances,
            bodies_per_env: scene.bodies.len() as u32,
            lighting,
        },
        report,
    ))
}
