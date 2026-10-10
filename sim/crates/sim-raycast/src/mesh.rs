//! Triangle meshes in a body's local frame, and generators for the shapes the
//! phase-1 scenes use (box, sphere, cylinder, plane).
//!
//! Each generator also records the exact surface it tessellates
//! ([`Mesh::analytic`]), so a scene can be packed for kernels that intersect
//! that surface in closed form instead of its triangles.
//!
//! Invariants every generator keeps (tests below check them):
//! - triangles are wound counter-clockwise seen from outside, so the geometric
//!   normal `(b - a) x (c - a)` points outwards;
//! - the vertex normals are unit length and point outwards;
//! - the closed shapes (box, sphere, cylinder) are closed: every edge is shared
//!   by exactly two triangles, in opposite directions.

use crate::math::{Vec3, cross, max3, min3, normalize, sub};

/// A triangle mesh with one normal per vertex.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    /// Vertex positions, metres, in the mesh's local frame.
    pub positions: Vec<Vec3>,
    /// One unit normal per vertex.
    pub normals: Vec<Vec3>,
    /// Triangles as indices into `positions`, counter-clockwise from outside.
    pub triangles: Vec<[u32; 3]>,
    /// The exact surface the triangles tessellate, if a generator made them
    /// (`None` for imported meshes).
    pub analytic: Option<Analytic>,
}

/// An exact surface in a mesh's local frame, centred on the origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Analytic {
    /// A box with these half extents. A plane is a box whose z half extent is
    /// 0.
    Box(Vec3),
    /// An ellipsoid with these semi-axes; a sphere has three equal ones.
    Ellipsoid(Vec3),
    /// A cylinder along z with flat caps: radius, half-length.
    Cylinder {
        /// The radius.
        r: f32,
        /// The half-length along z.
        hz: f32,
    },
    /// A capsule along z: a cylinder of half-length `half_len` with
    /// hemispherical caps of radius `r`.
    Capsule {
        /// The radius.
        r: f32,
        /// The half-length of the cylinder part along z.
        half_len: f32,
    },
}

impl Analytic {
    /// The surface's half extents along x, y and z: its box is
    /// `[-extent, extent]`.
    pub fn extent(&self) -> Vec3 {
        match *self {
            Analytic::Box(h) => h,
            Analytic::Ellipsoid(r) => r,
            Analytic::Cylinder { r, hz } => [r, r, hz],
            Analytic::Capsule { r, half_len } => [r, r, half_len + r],
        }
    }

    /// The kernels' kind code (`ANALYTIC_*` in `kernels/common.glsl`).
    pub fn kind(&self) -> u32 {
        match self {
            Analytic::Box(_) => 0,
            Analytic::Ellipsoid(_) => 1,
            Analytic::Cylinder { .. } => 2,
            Analytic::Capsule { .. } => 3,
        }
    }
}

impl Mesh {
    fn push_vertex(&mut self, p: Vec3, n: Vec3) -> u32 {
        self.positions.push(p);
        self.normals.push(normalize(n));
        (self.positions.len() - 1) as u32
    }

    /// The axis-aligned bounds of the vertices: `(min, max)`.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for &p in &self.positions {
            lo = min3(lo, p);
            hi = max3(hi, p);
        }
        (lo, hi)
    }

    /// The enclosed volume by the divergence theorem; positive for a closed,
    /// outward-wound mesh.
    pub fn signed_volume(&self) -> f64 {
        self.triangles
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| self.positions[i as usize].map(f64::from));
                let cr = [
                    b[1] * c[2] - b[2] * c[1],
                    b[2] * c[0] - b[0] * c[2],
                    b[0] * c[1] - b[1] * c[0],
                ];
                (a[0] * cr[0] + a[1] * cr[1] + a[2] * cr[2]) / 6.0
            })
            .sum()
    }

    /// The geometric (unnormalised) normal of triangle `i`.
    pub fn face_normal(&self, i: usize) -> Vec3 {
        let [a, b, c] = self.triangles[i].map(|k| self.positions[k as usize]);
        cross(sub(b, a), sub(c, a))
    }

    /// A box centred on the origin with half extents `h`, flat-shaded (each
    /// face has its own four vertices): 24 vertices, 12 triangles.
    pub fn cuboid(h: Vec3) -> Mesh {
        let mut m = Mesh::default();
        // (normal axis, sign); u and v span the face so that u x v = normal
        let faces: [(usize, f32); 6] = [
            (0, 1.0),
            (0, -1.0),
            (1, 1.0),
            (1, -1.0),
            (2, 1.0),
            (2, -1.0),
        ];
        for (axis, sign) in faces {
            let mut n = [0.0f32; 3];
            n[axis] = sign;
            let ua = (axis + 1) % 3;
            let va = (axis + 2) % 3;
            let mut u = [0.0f32; 3];
            let mut v = [0.0f32; 3];
            u[ua] = 1.0;
            v[va] = sign;
            let corner = |su: f32, sv: f32| -> Vec3 {
                let mut p = [0.0f32; 3];
                p[axis] = sign * h[axis];
                p[ua] = su * h[ua] * u[ua];
                p[va] = sv * h[va] * v[va];
                p
            };
            let a = m.push_vertex(corner(-1.0, -1.0), n);
            let b = m.push_vertex(corner(1.0, -1.0), n);
            let c = m.push_vertex(corner(1.0, 1.0), n);
            let d = m.push_vertex(corner(-1.0, 1.0), n);
            m.triangles.push([a, b, c]);
            m.triangles.push([a, c, d]);
        }
        m.analytic = Some(Analytic::Box(h));
        m
    }

    /// A UV sphere of radius `r` centred on the origin, smooth-shaded, with
    /// `segments` around the z axis and `rings` from pole to pole.
    pub fn sphere(r: f32, segments: u32, rings: u32) -> Mesh {
        assert!(segments >= 3 && rings >= 2);
        let mut m = Mesh::default();
        let top = m.push_vertex([0.0, 0.0, r], [0.0, 0.0, 1.0]);
        let mut ring_start = Vec::new();
        for i in 1..rings {
            let theta = std::f32::consts::PI * i as f32 / rings as f32;
            ring_start.push(m.positions.len() as u32);
            for j in 0..segments {
                let phi = std::f32::consts::TAU * j as f32 / segments as f32;
                let n = [
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                ];
                m.push_vertex([n[0] * r, n[1] * r, n[2] * r], n);
            }
        }
        let bottom = m.push_vertex([0.0, 0.0, -r], [0.0, 0.0, -1.0]);
        let at = |ring: usize, j: u32| ring_start[ring] + (j % segments);
        for j in 0..segments {
            m.triangles.push([top, at(0, j), at(0, j + 1)]);
        }
        for ring in 0..(rings as usize - 2) {
            for j in 0..segments {
                let a = at(ring, j);
                let b = at(ring + 1, j);
                let c = at(ring + 1, j + 1);
                let d = at(ring, j + 1);
                m.triangles.push([a, b, c]);
                m.triangles.push([a, c, d]);
            }
        }
        let last = rings as usize - 2;
        for j in 0..segments {
            m.triangles.push([bottom, at(last, j + 1), at(last, j)]);
        }
        m.analytic = Some(Analytic::Ellipsoid([r, r, r]));
        m
    }

    /// A cylinder of radius `r` along z from `-hz` to `+hz`, with smooth sides
    /// and flat caps.
    pub fn cylinder(r: f32, hz: f32, segments: u32) -> Mesh {
        assert!(segments >= 3);
        let mut m = Mesh::default();
        let ring = |m: &mut Mesh, z: f32, normal_z: Option<f32>| -> u32 {
            let start = m.positions.len() as u32;
            for j in 0..segments {
                let phi = std::f32::consts::TAU * j as f32 / segments as f32;
                let (s, c) = phi.sin_cos();
                let n = match normal_z {
                    Some(nz) => [0.0, 0.0, nz],
                    None => [c, s, 0.0],
                };
                m.push_vertex([r * c, r * s, z], n);
            }
            start
        };
        let side_lo = ring(&mut m, -hz, None);
        let side_hi = ring(&mut m, hz, None);
        for j in 0..segments {
            let a = side_lo + j;
            let b = side_lo + (j + 1) % segments;
            let c = side_hi + (j + 1) % segments;
            let d = side_hi + j;
            m.triangles.push([a, b, c]);
            m.triangles.push([a, c, d]);
        }
        let cap_hi = ring(&mut m, hz, Some(1.0));
        let centre_hi = m.push_vertex([0.0, 0.0, hz], [0.0, 0.0, 1.0]);
        let cap_lo = ring(&mut m, -hz, Some(-1.0));
        let centre_lo = m.push_vertex([0.0, 0.0, -hz], [0.0, 0.0, -1.0]);
        for j in 0..segments {
            let jn = (j + 1) % segments;
            m.triangles.push([centre_hi, cap_hi + j, cap_hi + jn]);
            m.triangles.push([centre_lo, cap_lo + jn, cap_lo + j]);
        }
        m.analytic = Some(Analytic::Cylinder { r, hz });
        m
    }

    /// A capsule along z: a cylinder of half-length `half_len` capped by two
    /// hemispheres of radius `r`, smooth-shaded, with `segments` around z and
    /// `rings_per_cap` rings from each pole to the equator.
    pub fn capsule(r: f32, half_len: f32, segments: u32, rings_per_cap: u32) -> Mesh {
        assert!(segments >= 3 && rings_per_cap >= 1);
        let mut m = Mesh::default();
        let top = m.push_vertex([0.0, 0.0, r + half_len], [0.0, 0.0, 1.0]);
        let mut ring_start = Vec::new();
        let k = rings_per_cap;
        // the upper cap's rings down to its equator at +half_len, then the
        // lower cap's from its equator at -half_len: the band between the two
        // equators is the cylinder
        let thetas: Vec<(f32, f32)> = (1..=k)
            .map(|i| (std::f32::consts::FRAC_PI_2 * i as f32 / k as f32, half_len))
            .chain((0..k).map(|i| {
                (
                    std::f32::consts::FRAC_PI_2 + std::f32::consts::FRAC_PI_2 * i as f32 / k as f32,
                    -half_len,
                )
            }))
            .collect();
        for &(theta, offset) in &thetas {
            ring_start.push(m.positions.len() as u32);
            for j in 0..segments {
                let phi = std::f32::consts::TAU * j as f32 / segments as f32;
                let n = [
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                ];
                m.push_vertex([n[0] * r, n[1] * r, n[2] * r + offset], n);
            }
        }
        let bottom = m.push_vertex([0.0, 0.0, -r - half_len], [0.0, 0.0, -1.0]);
        let at = |ring: usize, j: u32| ring_start[ring] + (j % segments);
        for j in 0..segments {
            m.triangles.push([top, at(0, j), at(0, j + 1)]);
        }
        for ring in 0..ring_start.len() - 1 {
            for j in 0..segments {
                let a = at(ring, j);
                let b = at(ring + 1, j);
                let c = at(ring + 1, j + 1);
                let d = at(ring, j + 1);
                m.triangles.push([a, b, c]);
                m.triangles.push([a, c, d]);
            }
        }
        let last = ring_start.len() - 1;
        for j in 0..segments {
            m.triangles.push([bottom, at(last, j + 1), at(last, j)]);
        }
        m.analytic = Some(Analytic::Capsule { r, half_len });
        m
    }

    /// An ellipsoid with semi-axes `radii`, smooth-shaded: the unit sphere
    /// scaled per axis, with normals transformed by the inverse transpose.
    pub fn ellipsoid(radii: Vec3, segments: u32, rings: u32) -> Mesh {
        let mut m = Mesh::sphere(1.0, segments, rings);
        for p in &mut m.positions {
            *p = [p[0] * radii[0], p[1] * radii[1], p[2] * radii[2]];
        }
        for n in &mut m.normals {
            *n = normalize([n[0] / radii[0], n[1] / radii[1], n[2] / radii[2]]);
        }
        m.analytic = Some(Analytic::Ellipsoid(radii));
        m
    }

    /// A flat-shaded mesh from vertex positions and counter-clockwise
    /// triangles: every triangle gets its own three vertices carrying its face
    /// normal, so creases stay sharp (imported meshes carry no normals).
    /// Degenerate triangles (zero area) are dropped; the count is returned.
    pub fn flat(vertices: &[Vec3], triangles: &[[u32; 3]]) -> (Mesh, usize) {
        let mut m = Mesh::default();
        let mut dropped = 0;
        for t in triangles {
            let [a, b, c] = t.map(|k| vertices[k as usize]);
            let n = cross(sub(b, a), sub(c, a));
            if crate::math::length(n) <= 0.0 || !n.iter().all(|x| x.is_finite()) {
                dropped += 1;
                continue;
            }
            let i0 = m.push_vertex(a, n);
            let i1 = m.push_vertex(b, n);
            let i2 = m.push_vertex(c, n);
            m.triangles.push([i0, i1, i2]);
        }
        (m, dropped)
    }

    /// A flat rectangle in the z = 0 plane, facing +z, with half extents `hx`
    /// and `hy`, cut into `nx` by `ny` cells (two triangles each). Not closed.
    pub fn plane(hx: f32, hy: f32, nx: u32, ny: u32) -> Mesh {
        assert!(nx >= 1 && ny >= 1);
        let mut m = Mesh::default();
        for iy in 0..=ny {
            for ix in 0..=nx {
                let x = -hx + 2.0 * hx * ix as f32 / nx as f32;
                let y = -hy + 2.0 * hy * iy as f32 / ny as f32;
                m.push_vertex([x, y, 0.0], [0.0, 0.0, 1.0]);
            }
        }
        let at = |ix: u32, iy: u32| iy * (nx + 1) + ix;
        for iy in 0..ny {
            for ix in 0..nx {
                let a = at(ix, iy);
                let b = at(ix + 1, iy);
                let c = at(ix + 1, iy + 1);
                let d = at(ix, iy + 1);
                m.triangles.push([a, b, c]);
                m.triangles.push([a, c, d]);
            }
        }
        m.analytic = Some(Analytic::Box([hx, hy, 0.0]));
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{dot, length};
    use std::collections::HashMap;

    /// Every directed edge appears once and its reverse once. Positions are
    /// compared, not indices, because flat faces and caps duplicate vertices.
    fn is_closed(m: &Mesh) -> bool {
        let key = |p: Vec3| p.map(|c| (c * 1.0e5).round() as i64);
        let mut edges: HashMap<([i64; 3], [i64; 3]), i32> = HashMap::new();
        for t in &m.triangles {
            for k in 0..3 {
                let a = key(m.positions[t[k] as usize]);
                let b = key(m.positions[t[(k + 1) % 3] as usize]);
                *edges.entry((a, b)).or_default() += 1;
            }
        }
        edges
            .iter()
            .all(|(&(a, b), &n)| n == 1 && edges.get(&(b, a)) == Some(&1))
    }

    fn outward(m: &Mesh) -> bool {
        (0..m.triangles.len()).all(|i| {
            let [a, b, c] = m.triangles[i].map(|k| m.positions[k as usize]);
            let centroid = [
                (a[0] + b[0] + c[0]) / 3.0,
                (a[1] + b[1] + c[1]) / 3.0,
                (a[2] + b[2] + c[2]) / 3.0,
            ];
            dot(m.face_normal(i), centroid) > 0.0
        })
    }

    #[test]
    fn the_cuboid_is_closed_outward_and_has_its_volume() {
        let m = Mesh::cuboid([0.5, 0.25, 0.1]);
        assert_eq!(m.triangles.len(), 12);
        assert!(is_closed(&m));
        assert!(outward(&m));
        assert!((m.signed_volume() - 0.1).abs() < 1e-6);
        assert_eq!(m.bounds(), ([-0.5, -0.25, -0.1], [0.5, 0.25, 0.1]));
    }

    #[test]
    fn the_sphere_is_closed_outward_and_near_its_volume() {
        let m = Mesh::sphere(0.3, 32, 16);
        assert!(is_closed(&m));
        assert!(outward(&m));
        let exact = 4.0 / 3.0 * std::f64::consts::PI * 0.027;
        let v = m.signed_volume();
        // an inscribed polyhedron: a little smaller, never larger
        assert!(v < exact && v > 0.97 * exact, "{v} vs {exact}");
        for n in &m.normals {
            assert!((length(*n) - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn the_cylinder_is_closed_outward_and_near_its_volume() {
        let m = Mesh::cylinder(0.05, 0.1, 48);
        assert!(is_closed(&m));
        assert!(outward(&m));
        let exact = std::f64::consts::PI * 0.0025 * 0.2;
        let v = m.signed_volume();
        assert!(v < exact && v > 0.99 * exact, "{v} vs {exact}");
    }

    #[test]
    fn the_capsule_is_closed_outward_and_near_its_volume() {
        let m = Mesh::capsule(0.05, 0.1, 32, 8);
        assert!(is_closed(&m));
        assert!(outward(&m));
        let (r, h) = (0.05f64, 0.1f64);
        let exact =
            std::f64::consts::PI * r * r * (2.0 * h) + 4.0 / 3.0 * std::f64::consts::PI * r * r * r;
        let v = m.signed_volume();
        assert!(v < exact && v > 0.97 * exact, "{v} vs {exact}");
        let (lo, hi) = m.bounds();
        for (got, want) in lo
            .iter()
            .chain(&hi)
            .zip([-0.05, -0.05, -0.15, 0.05, 0.05, 0.15])
        {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    #[test]
    fn the_ellipsoid_is_closed_outward_and_near_its_volume() {
        let m = Mesh::ellipsoid([0.3, 0.2, 0.1], 48, 24);
        assert!(is_closed(&m));
        assert!(outward(&m));
        let exact = 4.0 / 3.0 * std::f64::consts::PI * 0.3 * 0.2 * 0.1;
        let v = m.signed_volume();
        assert!(v < exact && v > 0.98 * exact, "{v} vs {exact}");
    }

    #[test]
    fn a_flat_mesh_keeps_creases_and_drops_degenerate_triangles() {
        let cube = Mesh::cuboid([0.5; 3]);
        let tris: Vec<[u32; 3]> = cube.triangles.iter().copied().chain([[0, 0, 1]]).collect();
        let (m, dropped) = Mesh::flat(&cube.positions, &tris);
        assert_eq!(dropped, 1);
        assert_eq!(m.triangles.len(), 12);
        assert!(is_closed(&m));
        assert!(outward(&m));
        for (i, t) in m.triangles.iter().enumerate() {
            let face = normalize(m.face_normal(i));
            for &k in t {
                assert!(dot(m.normals[k as usize], face) > 0.9999);
            }
        }
    }

    #[test]
    fn the_plane_faces_up() {
        let m = Mesh::plane(2.0, 1.0, 4, 3);
        assert_eq!(m.triangles.len(), 24);
        for i in 0..m.triangles.len() {
            assert!(m.face_normal(i)[2] > 0.0);
        }
    }
}
