//! The scene description the renderer reads, and its packing into the device
//! buffers of `kernels/common.glsl`.
//!
//! A scene is static for an episode: meshes, materials, instances and lighting
//! are uploaded once, at reset. What moves is the world state: one pose per
//! body per environment, written by the physics into a device buffer the
//! renderer reads in place (the device camera of `sim-render`). An
//! instance is a mesh with a material, attached to a body (or static) at a
//! fixed local offset, carrying the segmentation id its pixels report.

use crate::bvh::{self, BvhNode};
use crate::SceneError;
use crate::layout::{
    INSTANCE_STATIC_BYTES, MATERIAL_BYTES, NODE_BYTES, STATIC_BODY, TRI_BYTES, TRI_NORMAL_BYTES,
    Words,
};
use crate::math::{Pose, Vec3, length, normalize, quat_normalize, sub};
use crate::mesh::{Analytic, Mesh};

/// The most instances one environment may hold (`MAX_INSTANCES` in
/// `kernels/common.glsl`).
pub const MAX_INSTANCES: usize = 64;

/// A physically based material.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    /// Base colour, linear RGB in [0, 1].
    pub base_colour: Vec3,
    /// Perceptual roughness in [0.045, 1].
    pub roughness: f32,
    /// Metallic in [0, 1].
    pub metallic: f32,
    /// Checker cells per metre in the mesh's local frame; 0 for none.
    pub checker_cells_per_metre: f32,
    /// The checker's dark cells are the base colour times this.
    pub checker_dark: f32,
    /// Emitted radiance, linear RGB (W sr^-1 m^-2 in the renderer's relative
    /// units), each channel >= 0: `sim-scene`'s `emission_linear`.
    pub emission: Vec3,
}

impl Material {
    /// A plain dielectric of the given colour and roughness.
    pub fn plain(base_colour: Vec3, roughness: f32) -> Self {
        Self {
            base_colour,
            roughness,
            metallic: 0.0,
            checker_cells_per_metre: 0.0,
            checker_dark: 1.0,
            emission: [0.0; 3],
        }
    }
}

/// One instance of a mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instance {
    /// The body (index within an environment) it moves with, or `None` for
    /// static geometry.
    pub body: Option<u32>,
    /// Index into [`SceneDesc::meshes`].
    pub mesh: u32,
    /// Index into [`SceneDesc::materials`].
    pub material: u32,
    /// Its pose relative to its body (or the world, when static).
    pub offset: Pose,
    /// The segmentation id its pixels carry. 0 is reserved for "nothing hit".
    pub seg_id: u16,
}

/// The lighting: one sun, a sky and a ground, and the camera's exposure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lighting {
    /// Unit vector towards the sun, world frame.
    pub sun_direction: Vec3,
    /// The sun's irradiance on a surface facing it, linear RGB.
    pub sun_irradiance: Vec3,
    /// The sky's radiance at the zenith, linear RGB.
    pub sky_zenith: Vec3,
    /// How much brighter the sky is at the horizon (`1 + boost`).
    pub horizon_boost: f32,
    /// The radiance from below the horizon, linear RGB.
    pub ground: Vec3,
    /// The multiplier applied before the tone curve.
    pub exposure: f32,
    /// Whether the sun casts shadows.
    pub shadows: bool,
}

/// Everything static about a scene.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneDesc {
    /// The meshes, in their own local frames.
    pub meshes: Vec<Mesh>,
    /// The materials.
    pub materials: Vec<Material>,
    /// The instances, the same in every environment, at most
    /// [`MAX_INSTANCES`].
    pub instances: Vec<Instance>,
    /// Bodies per environment in the pose buffer.
    pub bodies_per_env: u32,
    /// The lighting.
    pub lighting: Lighting,
}

/// A triangle as the kernels store it: its first vertex and two edges.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PackedTri {
    /// The first vertex.
    pub v0: Vec3,
    /// `v1 - v0`.
    pub e1: Vec3,
    /// `v2 - v0`.
    pub e2: Vec3,
}

/// A static instance as the kernels store it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PackedInstance {
    /// The local offset.
    pub offset: Pose,
    /// The body index, or [`STATIC_BODY`].
    pub body: u32,
    /// The mesh's root node in [`PackedScene::nodes`].
    pub root: u32,
    /// The mesh's local box.
    pub bmin: Vec3,
    /// The mesh's local box.
    pub bmax: Vec3,
    /// The material index.
    pub material: u16,
    /// The segmentation id.
    pub seg_id: u16,
}

/// A scene packed for the device, with typed copies of what was packed for the
/// host reference renderer.
#[derive(Clone, Debug)]
pub struct PackedScene {
    /// All meshes' BVH nodes, with child and triangle indices made global.
    pub nodes: Vec<BvhNode>,
    /// All meshes' triangles in leaf order.
    pub tris: Vec<PackedTri>,
    /// Each triangle's vertex normals.
    pub tri_normals: Vec<[Vec3; 3]>,
    /// The materials.
    pub materials: Vec<Material>,
    /// The instances.
    pub instances: Vec<PackedInstance>,
    /// Bodies per environment.
    pub bodies_per_env: u32,
    /// The lighting.
    pub lighting: Lighting,
    /// The deepest BVH among the meshes.
    pub max_depth: u32,
    /// Whether any mesh was packed as its exact surface (a node with
    /// [`ANALYTIC_BIT`]); only a kernel built with `ANALYTIC` renders it.
    pub analytic: bool,
}

/// Set in a root node's count word: the node is an exact surface, not a
/// hierarchy. The low bits are the surface's kind
/// ([`crate::mesh::Analytic::kind`]); the node's box is the surface's
/// `[-extent, extent]`, which carries its parameters.
pub const ANALYTIC_BIT: u32 = 0x8000_0000;

/// How [`SceneDesc::pack_with`] packs the meshes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackOptions {
    /// The most triangles in a BVH leaf.
    pub max_leaf: usize,
    /// Pack each generated mesh as its exact surface (one node) instead of
    /// its triangles. Imported meshes keep their triangles.
    pub analytic: bool,
}

impl Default for PackOptions {
    /// Exact surfaces, the default since 2026-10-01: the camera then sees the same sphere, capsule and box the
    /// physics collides with, and it is the faster build (design note 8.4).
    fn default() -> Self {
        PackOptions {
            max_leaf: bvh::MAX_LEAF,
            analytic: true,
        }
    }
}

fn finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

impl SceneDesc {
    /// Check the description and pack it with the default [`PackOptions`]:
    /// every generated mesh as its exact surface, imported meshes as
    /// triangles. Refuses, by name, anything the kernels cannot render as
    /// described.
    pub fn pack(&self) -> Result<PackedScene, SceneError> {
        self.pack_with(PackOptions::default())
    }

    /// [`SceneDesc::pack`] with every mesh as triangles, the generated ones
    /// tessellated (the packing before exact surfaces, kept for the triangle
    /// builds and their tests).
    pub fn pack_triangles(&self) -> Result<PackedScene, SceneError> {
        self.pack_with_leaf(bvh::MAX_LEAF)
    }

    /// [`SceneDesc::pack_triangles`] with BVH leaves of at most `max_leaf`
    /// triangles (a measurement lever; [`bvh::MAX_LEAF`] is the default).
    pub fn pack_with_leaf(&self, max_leaf: usize) -> Result<PackedScene, SceneError> {
        self.pack_with(PackOptions {
            max_leaf,
            analytic: false,
        })
    }

    /// [`SceneDesc::pack`] with explicit [`PackOptions`].
    pub fn pack_with(&self, opts: PackOptions) -> Result<PackedScene, SceneError> {
        let max_leaf = opts.max_leaf;
        let bad = |why: String| Err(SceneError(why));
        if self.instances.is_empty() || self.instances.len() > MAX_INSTANCES {
            return bad(format!(
                "a scene holds 1 to {MAX_INSTANCES} instances; this one has {}",
                self.instances.len()
            ));
        }
        if self.materials.is_empty() || self.materials.len() > usize::from(u16::MAX) {
            return bad(format!(
                "{} materials is out of range",
                self.materials.len()
            ));
        }
        for (i, m) in self.materials.iter().enumerate() {
            let in_unit = |x: f32| (0.0..=1.0).contains(&x);
            if !(m.base_colour.iter().all(|&c| in_unit(c))
                && (0.045..=1.0).contains(&m.roughness)
                && in_unit(m.metallic)
                && m.checker_cells_per_metre >= 0.0
                && m.checker_cells_per_metre.is_finite()
                && in_unit(m.checker_dark)
                && m.emission.iter().all(|&e| e >= 0.0 && e.is_finite()))
            {
                return bad(format!("material {i} is outside its ranges: {m:?}"));
            }
        }
        let l = &self.lighting;
        if !(finite(&l.sun_direction)
            && (length(l.sun_direction) - 1.0).abs() < 1e-4
            && finite(&l.sun_irradiance)
            && finite(&l.sky_zenith)
            && finite(&l.ground)
            && l.horizon_boost.is_finite()
            && l.exposure.is_finite()
            && l.exposure > 0.0)
        {
            return bad(format!(
                "the lighting is not finite or its sun is not a unit vector: {l:?}"
            ));
        }

        let mut nodes = Vec::new();
        let mut tris = Vec::new();
        let mut tri_normals = Vec::new();
        let mut roots = Vec::with_capacity(self.meshes.len());
        let mut max_depth = 0;
        let mut any_analytic = false;
        for (mi, mesh) in self.meshes.iter().enumerate() {
            if opts.analytic
                && let Some(a) = mesh.analytic
            {
                let e = a.extent();
                // a plane is a box with no thickness; every other surface has
                // positive extents
                let ok = finite(&e)
                    && match a {
                        Analytic::Box(h) => h[0] > 0.0 && h[1] > 0.0 && h[2] >= 0.0,
                        _ => e.iter().all(|&x| x > 0.0),
                    };
                if !ok {
                    return bad(format!("mesh {mi}'s exact surface is degenerate: {a:?}"));
                }
                roots.push(nodes.len() as u32);
                nodes.push(BvhNode {
                    bmin: e.map(|x| -x),
                    bmax: e,
                    left_or_first: 0,
                    count: ANALYTIC_BIT | a.kind(),
                });
                max_depth = max_depth.max(1);
                any_analytic = true;
                continue;
            }
            if mesh.positions.len() != mesh.normals.len()
                || mesh
                    .triangles
                    .iter()
                    .any(|t| t.iter().any(|&k| k as usize >= mesh.positions.len()))
                || !mesh.positions.iter().all(|p| finite(p))
                || !mesh
                    .normals
                    .iter()
                    .all(|n| finite(n) && (length(*n) - 1.0).abs() < 1e-3)
            {
                return bad(format!(
                    "mesh {mi} is malformed (indices, non-finite values or normals)"
                ));
            }
            let bvh = bvh::build_with_leaf(mesh, max_leaf)
                .map_err(|e| SceneError(format!("mesh {mi}: {e}")))?;
            max_depth = max_depth.max(bvh.depth);
            let node_base = nodes.len() as u32;
            let tri_base = tris.len() as u32;
            roots.push(node_base);
            for n in &bvh.nodes {
                let mut g = *n;
                if g.count > 0 {
                    g.left_or_first += tri_base;
                } else {
                    g.left_or_first += node_base;
                }
                nodes.push(g);
            }
            for &t in &bvh.order {
                let [a, b, c] = mesh.triangles[t as usize];
                let (p0, p1, p2) = (
                    mesh.positions[a as usize],
                    mesh.positions[b as usize],
                    mesh.positions[c as usize],
                );
                tris.push(PackedTri {
                    v0: p0,
                    e1: sub(p1, p0),
                    e2: sub(p2, p0),
                });
                tri_normals.push([
                    mesh.normals[a as usize],
                    mesh.normals[b as usize],
                    mesh.normals[c as usize],
                ]);
            }
        }

        let mut instances = Vec::with_capacity(self.instances.len());
        for (ii, inst) in self.instances.iter().enumerate() {
            if inst.mesh as usize >= self.meshes.len() {
                return bad(format!(
                    "instance {ii} names mesh {}, of {}",
                    inst.mesh,
                    self.meshes.len()
                ));
            }
            if inst.material as usize >= self.materials.len() {
                return bad(format!(
                    "instance {ii} names material {}, of {}",
                    inst.material,
                    self.materials.len()
                ));
            }
            if inst.seg_id == 0 {
                return bad(format!(
                    "instance {ii} has segmentation id 0, which means nothing was hit"
                ));
            }
            if let Some(b) = inst.body
                && b >= self.bodies_per_env
            {
                return bad(format!(
                    "instance {ii} is attached to body {b}, of {}",
                    self.bodies_per_env
                ));
            }
            let q = inst.offset.rotation;
            if !finite(&inst.offset.position) || !finite(&q) {
                return bad(format!("instance {ii}'s offset is not finite"));
            }
            let root = roots[inst.mesh as usize];
            let rn = nodes[root as usize];
            instances.push(PackedInstance {
                offset: Pose::new(inst.offset.position, quat_normalize(q)),
                body: inst.body.unwrap_or(STATIC_BODY),
                root,
                bmin: rn.bmin,
                bmax: rn.bmax,
                material: inst.material as u16,
                seg_id: inst.seg_id,
            });
        }
        Ok(PackedScene {
            nodes,
            tris,
            tri_normals,
            materials: self.materials.clone(),
            instances,
            bodies_per_env: self.bodies_per_env,
            lighting: Lighting {
                sun_direction: normalize(self.lighting.sun_direction),
                ..self.lighting
            },
            max_depth,
            analytic: any_analytic,
        })
    }
}

impl PackedScene {
    /// The node buffer's bytes.
    pub fn node_bytes(&self) -> Vec<u8> {
        let mut w = Words(Vec::with_capacity(self.nodes.len() * NODE_BYTES));
        for n in &self.nodes {
            w.v3_bits(n.bmin, n.left_or_first).v3_bits(n.bmax, n.count);
        }
        w.0
    }

    /// The triangle buffer's bytes.
    pub fn tri_bytes(&self) -> Vec<u8> {
        let mut w = Words(Vec::with_capacity(self.tris.len() * TRI_BYTES));
        for t in &self.tris {
            w.v3_f(t.v0, 0.0).v3_f(t.e1, 0.0).v3_f(t.e2, 0.0);
        }
        w.0
    }

    /// The vertex-normal buffer's bytes.
    pub fn tri_normal_bytes(&self) -> Vec<u8> {
        let mut w = Words(Vec::with_capacity(
            self.tri_normals.len() * TRI_NORMAL_BYTES,
        ));
        for ns in &self.tri_normals {
            for n in ns {
                w.v3_f(*n, 0.0);
            }
        }
        w.0
    }

    /// The material buffer's bytes.
    pub fn material_bytes(&self) -> Vec<u8> {
        let mut w = Words(Vec::with_capacity(self.materials.len() * MATERIAL_BYTES));
        for m in &self.materials {
            w.v3_f(m.base_colour, m.roughness)
                .f(m.metallic)
                .f(m.checker_cells_per_metre)
                .f(m.checker_dark)
                .f(0.0)
                .v3_f(m.emission, 0.0);
        }
        w.0
    }

    /// The static instance buffer's bytes.
    pub fn instance_bytes(&self) -> Vec<u8> {
        let mut w = Words(Vec::with_capacity(
            self.instances.len() * INSTANCE_STATIC_BYTES,
        ));
        for i in &self.instances {
            w.v3_bits(i.offset.position, i.body)
                .quat(i.offset.rotation)
                .v3_bits(i.bmin, i.root)
                .v3_bits(i.bmax, u32::from(i.material) | (u32::from(i.seg_id) << 16));
        }
        w.0
    }

    /// Triangles over all meshes (each counted once, however often instanced).
    pub fn triangle_count(&self) -> usize {
        self.tris.len()
    }

    /// Triangles one environment's instances hold, counting every instance.
    pub fn instanced_triangles(&self) -> u64 {
        // a mesh's triangles are the leaves under its root; count them by
        // walking its subtree
        self.instances
            .iter()
            .map(|i| self.subtree_triangles(i.root))
            .sum()
    }

    fn subtree_triangles(&self, root: u32) -> u64 {
        let mut stack = vec![root];
        let mut total = 0u64;
        while let Some(k) = stack.pop() {
            let n = &self.nodes[k as usize];
            if n.count & ANALYTIC_BIT != 0 {
                // an exact surface: no triangles
            } else if n.count > 0 {
                total += u64::from(n.count);
            } else {
                stack.push(n.left_or_first);
                stack.push(n.left_or_first + 1);
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{f32_at, u32_at};

    fn tiny() -> SceneDesc {
        SceneDesc {
            meshes: vec![Mesh::cuboid([0.5, 0.5, 0.5]), Mesh::plane(1.0, 1.0, 2, 2)],
            materials: vec![Material::plain([0.5, 0.5, 0.5], 0.5)],
            instances: vec![
                Instance {
                    body: Some(0),
                    mesh: 0,
                    material: 0,
                    offset: Pose::IDENTITY,
                    seg_id: 1,
                },
                Instance {
                    body: None,
                    mesh: 1,
                    material: 0,
                    offset: Pose::at([0.0, 0.0, -1.0]),
                    seg_id: 2,
                },
            ],
            bodies_per_env: 1,
            lighting: Lighting {
                sun_direction: [0.0, 0.0, 1.0],
                sun_irradiance: [3.0; 3],
                sky_zenith: [0.3; 3],
                horizon_boost: 1.0,
                ground: [0.1; 3],
                exposure: 1.0,
                shadows: true,
            },
        }
    }

    #[test]
    fn packing_makes_indices_global_and_bytes_match_the_records() {
        let s = tiny().pack_triangles().unwrap();
        // the plane's root follows every node of the cuboid's tree
        let cuboid_nodes = bvh::build(&Mesh::cuboid([0.5; 3])).unwrap().nodes.len();
        assert_eq!(s.instances[1].root as usize, cuboid_nodes);
        let nb = s.node_bytes();
        assert_eq!(nb.len(), s.nodes.len() * NODE_BYTES);
        for (k, n) in s.nodes.iter().enumerate() {
            let w = k * 8;
            assert_eq!(f32_at(&nb, w), n.bmin[0]);
            assert_eq!(u32_at(&nb, w + 3), n.left_or_first);
            assert_eq!(u32_at(&nb, w + 7), n.count);
        }
        let ib = s.instance_bytes();
        assert_eq!(ib.len(), 2 * INSTANCE_STATIC_BYTES);
        assert_eq!(u32_at(&ib, 3), 0);
        assert_eq!(u32_at(&ib, 16 + 3), STATIC_BODY);
        assert_eq!(u32_at(&ib, 16 + 15), 2 << 16);
        assert_eq!(s.instanced_triangles(), 12 + 8);
        assert_eq!(s.tri_bytes().len(), s.tris.len() * TRI_BYTES);
        assert_eq!(s.tri_normal_bytes().len(), s.tris.len() * TRI_NORMAL_BYTES);
        assert_eq!(s.material_bytes().len(), MATERIAL_BYTES);
    }

    #[test]
    fn a_scene_the_kernels_cannot_render_is_refused_by_name() {
        let mut s = tiny();
        s.instances[0].seg_id = 0;
        assert!(matches!(s.pack(), Err(SceneError(m)) if m.contains("segmentation id 0")));
        let mut s = tiny();
        s.instances[0].body = Some(1);
        assert!(s.pack().is_err());
        let mut s = tiny();
        s.instances = vec![s.instances[0]; MAX_INSTANCES + 1];
        assert!(s.pack().is_err());
        let mut s = tiny();
        s.materials[0].roughness = 0.0;
        assert!(s.pack().is_err());
        let mut s = tiny();
        s.lighting.sun_direction = [0.0, 0.0, 2.0];
        assert!(s.pack().is_err());
        let mut s = tiny();
        s.instances[0].offset.rotation = [f32::NAN, 0.0, 0.0, 1.0];
        assert!(s.pack().is_err());
    }

    /// Packed as exact surfaces (the default), every generated mesh is one
    /// flagged node with its extents as its box, and no triangles; an
    /// imported (flat) mesh keeps its triangles; `pack_triangles` flags
    /// nothing.
    #[test]
    fn exact_surfaces_pack_as_one_flagged_node_each() {
        let mut s = tiny();
        let (flat, _) = Mesh::flat(&[[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], &[[0, 1, 2]]);
        s.meshes.push(flat);
        let plain = s.pack_triangles().unwrap();
        assert!(!plain.analytic);
        assert!(s.pack().unwrap().analytic, "exact surfaces are the default");
        assert!(plain.nodes.iter().all(|n| n.count & ANALYTIC_BIT == 0));
        let exact = s
            .pack_with(PackOptions {
                analytic: true,
                ..PackOptions::default()
            })
            .unwrap();
        assert!(exact.analytic);
        assert_eq!(exact.tris.len(), 1, "only the imported triangle");
        assert_eq!(
            exact.instanced_triangles(),
            0,
            "the instances are exact surfaces"
        );
        for (mi, mesh) in s.meshes.iter().enumerate() {
            if let Some(a) = mesh.analytic {
                let n = exact
                    .nodes
                    .iter()
                    .find(|n| n.count == ANALYTIC_BIT | a.kind() && n.bmax == a.extent())
                    .unwrap_or_else(|| panic!("mesh {mi}: no node for {a:?}"));
                assert_eq!(n.bmin, a.extent().map(|x| -x));
            }
        }
        // a degenerate surface is refused by name
        let mut bad = tiny();
        bad.meshes[0].analytic = Some(Analytic::Ellipsoid([0.1, 0.0, 0.1]));
        assert!(
            bad.pack_with(PackOptions {
                analytic: true,
                ..PackOptions::default()
            })
            .is_err()
        );
    }
}
