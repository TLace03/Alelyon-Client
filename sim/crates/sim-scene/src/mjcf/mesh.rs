//! Binary STL meshes and MuJoCo's mesh mass properties.
//!
//! Ports:
//! - the STL reader from MuJoCo's own decoder, `plugin/stl_decoder/stl_decoder.cc`
//!   (lines 53-114 of commit a8373cc4e): header, face count, size check, the
//!   2^30 coordinate bound and the vertex de-duplication by bit pattern;
//! - the mesh compile from `src/user/user_mesh.cc`: `triangle` (150-182),
//!   `ComputeFaceCentroid` (1265-1291), `ComputeVolume` (1129-1165),
//!   `ComputeInertia` (1505-1574) and the tail of `Process` (1293-1500: winding
//!   correction for a mirrored scale, scaling, the centre of mass, the eigen
//!   decomposition and the equivalent inertia box).
//!
//! Invariants:
//! - Binary STL only. An ASCII STL, a file whose size disagrees with its
//!   triangle count, a count outside 1..=200000, or a coordinate beyond 2^30 is
//!   refused, as MuJoCo's decoder refuses it.
//! - The mesh is kept in its own frame, scaled but NOT re-centred or rotated
//!   into its principal frame (MuJoCo re-centres the stored vertices and folds the
//!   offset into the geom pose; here the geom keeps its authored pose and the
//!   offset is returned as `com` and `quat`, which the importer applies when it
//!   sums a body's inertia). The geometry in the world is identical.
//! - Inertia mode is MuJoCo's default `legacy` (every pyramid volume taken
//!   positive) or `exact` (signed volumes, inconsistent winding refused).
//!   `convex` and `shell` are refused by the importer.
//! - `refpos` and `refquat` are not supported (the importer refuses them).

use super::mjmath::{MINVAL, cross, dot3, eig3, mulvecmat, quat2mat};

/// MuJoCo's mesh inertia modes that this importer implements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MeshInertia {
    /// Every pyramid volume is taken positive (the default).
    Legacy,
    /// Signed volumes; faces with inconsistent winding are refused.
    Exact,
}

/// A decoded STL: vertices de-duplicated by bit pattern, in first-seen order.
pub(crate) struct Stl {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
}

/// The longest STL MuJoCo's decoder reads: 200000 triangles of 50 bytes plus
/// the 84-byte header.
pub(crate) const MAX_STL_BYTES: u64 = 84 + 200_000 * 50;

/// Decodes a binary STL (`stl_decoder.cc:53-114`).
pub(crate) fn read_stl(bytes: &[u8]) -> Result<Stl, String> {
    if bytes.is_empty() {
        return Err("STL file is empty".into());
    }
    if bytes.len() < 84 {
        return Err("invalid header in STL file".into());
    }
    let nfaces = i32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]);
    if !(1..=200_000).contains(&nfaces) {
        return Err(
            "number of faces should be between 1 and 200000; perhaps this is an ASCII file?".into(),
        );
    }
    let nfaces = nfaces as usize;
    if nfaces * 50 != bytes.len() - 84 {
        return Err("STL file has the wrong size; perhaps this is an ASCII file?".into());
    }
    let body = &bytes[84..];
    let mut vertices: Vec<[f32; 3]> = Vec::new();
    let mut faces: Vec<[u32; 3]> = Vec::with_capacity(nfaces);
    let mut seen: std::collections::HashMap<[u32; 3], u32> = std::collections::HashMap::new();
    let bound = 2f32.powi(30);
    for i in 0..nfaces {
        let mut face = [0u32; 3];
        for (j, slot) in face.iter_mut().enumerate() {
            let at = 50 * i + 12 * (j + 1);
            let mut v = [0f32; 3];
            for (k, c) in v.iter_mut().enumerate() {
                let o = at + 4 * k;
                *c = f32::from_le_bytes([body[o], body[o + 1], body[o + 2], body[o + 3]]);
            }
            if v.iter().any(|c| c.abs() > bound) {
                return Err("vertex in STL file exceeds maximum bounds".into());
            }
            if v.iter().any(|c| !c.is_finite()) {
                return Err("vertex coordinate in STL file is not finite".into());
            }
            let key = [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()];
            let next = vertices.len() as u32;
            let index = *seen.entry(key).or_insert_with(|| {
                vertices.push(v);
                next
            });
            *slot = index;
        }
        faces.push(face);
    }
    Ok(Stl { vertices, faces })
}

/// A mesh after MuJoCo's compile: scaled, with its mass properties.
pub(crate) struct ProcessedMesh {
    /// Vertices after `scale`, in the mesh's own frame, narrowed to `f32`.
    pub vertices: Vec<[f32; 3]>,
    /// Faces, with the winding of a mirrored scale corrected.
    pub faces: Vec<[u32; 3]>,
    /// Volume at unit density (`GetVolumeRef`).
    pub volume: f64,
    /// Centre of mass in the mesh frame (`pos_`).
    pub com: [f64; 3],
    /// Principal axes of the mesh's inertia, `[w, x, y, z]` (`quat_`).
    pub quat: [f64; 4],
    /// Sizes of the equivalent inertia box (`boxsz_`).
    pub boxsz: [f64; 3],
    /// The bounding radius MuJoCo gives a geom of this mesh (`geom_rbound`): the
    /// length of the half-diagonal of the axis-aligned box `aamm_` of the vertices
    /// in the mesh's principal frame, centred on the centre of mass
    /// (`mjCMesh::Rotate`, user_mesh.cc:1576-1590; `mjCGeom::GetRBound`,
    /// user_objects.cc:3719-3725).
    pub rbound: f64,
}

/// `triangle` (user_mesh.cc:150): area, unit normal and centre of a triangle;
/// area 0 (normal left unnormalised) below `MINVAL`.
fn triangle(v1: [f64; 3], v2: [f64; 3], v3: [f64; 3]) -> (f64, [f64; 3], [f64; 3]) {
    let center = [
        (v1[0] + v2[0] + v3[0]) / 3.0,
        (v1[1] + v2[1] + v3[1]) / 3.0,
        (v1[2] + v2[2] + v3[2]) / 3.0,
    ];
    let b = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]];
    let c = [v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]];
    let mut normal = cross(b, c);
    let len = dot3(normal, normal).sqrt();
    if len < MINVAL {
        return (0.0, normal, center);
    }
    normal[0] /= len;
    normal[1] /= len;
    normal[2] /= len;
    (0.5 * len, normal, center)
}

fn vertex(dvert: &[f64], index: u32) -> [f64; 3] {
    let i = 3 * index as usize;
    [dvert[i], dvert[i + 1], dvert[i + 2]]
}

/// Compiles a mesh the way `mjCMesh::Process` does, for the non-shell,
/// non-convex modes. `scale` is the `scale` attribute.
pub(crate) fn process(
    stl_vertices: &[f32],
    faces_in: &[[u32; 3]],
    scale: [f64; 3],
    mode: MeshInertia,
) -> Result<ProcessedMesh, String> {
    let nvert = stl_vertices.len() / 3;
    if nvert < 4 {
        return Err("at least 4 vertices required".into());
    }
    for (i, f) in faces_in.iter().enumerate() {
        if f.iter().any(|&v| v as usize >= nvert) {
            return Err(format!("in face {i}, a vertex index does not exist"));
        }
    }
    let mut faces: Vec<[u32; 3]> = faces_in.to_vec();
    let mut dvert: Vec<f64> = stl_vertices.iter().map(|&v| f64::from(v)).collect();

    // exact mode refuses inconsistent winding (user_mesh.cc:1318-1328)
    if mode == MeshInertia::Exact {
        let mut halfedges: Vec<(u32, u32)> = Vec::new();
        for f in &faces {
            let (area, _, _) = triangle(
                vertex(&dvert, f[0]),
                vertex(&dvert, f[1]),
                vertex(&dvert, f[2]),
            );
            if area > MINVAL.sqrt() {
                halfedges.push((f[0], f[1]));
                halfedges.push((f[1], f[2]));
                halfedges.push((f[2], f[0]));
            }
        }
        halfedges.sort();
        if let Some(w) = halfedges.windows(2).find(|w| w[0] == w[1]) {
            return Err(format!(
                "faces have inconsistent orientation; check the faces containing the vertices {} and {}",
                w[0].0 + 1,
                w[0].1 + 1
            ));
        }
    }

    // winding correction for a mirrored scale (user_mesh.cc:1363-1376)
    if scale[0] * scale[1] * scale[2] <= 0.0 {
        for f in faces.iter_mut() {
            f.swap(1, 2);
        }
    }

    // scale (ApplyTransformations, user_mesh.cc:1230-1243)
    if scale != [1.0, 1.0, 1.0] {
        for v in dvert.chunks_exact_mut(3) {
            v[0] *= scale[0];
            v[1] *= scale[1];
            v[2] *= scale[2];
        }
    }

    // centroid of faces (ComputeFaceCentroid)
    let mut facecen = [0.0f64; 3];
    let mut total_area = 0.0;
    for f in &faces {
        let (area, _, center) = triangle(
            vertex(&dvert, f[0]),
            vertex(&dvert, f[1]),
            vertex(&dvert, f[2]),
        );
        facecen[0] += area * center[0];
        facecen[1] += area * center[1];
        facecen[2] += area * center[2];
        total_area += area;
    }
    if total_area >= MINVAL {
        facecen[0] /= total_area;
        facecen[1] /= total_area;
        facecen[2] /= total_area;
    }
    if total_area < MINVAL {
        return Err("mesh surface area is too small".into());
    }

    // volume and centre of mass (ComputeVolume)
    let mut com = [0.0f64; 3];
    let mut volume = 0.0;
    for f in &faces {
        let (area, normal, center) = triangle(
            vertex(&dvert, f[0]),
            vertex(&dvert, f[1]),
            vertex(&dvert, f[2]),
        );
        let vec = [
            center[0] - facecen[0],
            center[1] - facecen[1],
            center[2] - facecen[2],
        ];
        let mut v = dot3(vec, normal) * area / 3.0;
        if mode == MeshInertia::Legacy {
            v = v.abs();
        }
        volume += v;
        com[0] += v * (center[0] * 3.0 / 4.0 + facecen[0] / 4.0);
        com[1] += v * (center[1] * 3.0 / 4.0 + facecen[1] / 4.0);
        com[2] += v * (center[2] * 3.0 / 4.0 + facecen[2] / 4.0);
    }
    if volume >= MINVAL {
        com[0] /= volume;
        com[1] /= volume;
        com[2] /= volume;
    } else if volume < 0.0 {
        return Err("mesh volume is negative (misoriented triangles)".into());
    } else {
        return Err("mesh volume is too small".into());
    }

    // products of inertia about the centre of mass (ComputeInertia)
    let centered: Vec<f64> = dvert
        .chunks_exact(3)
        .flat_map(|v| [v[0] - com[0], v[1] - com[1], v[2] - com[2]])
        .collect();
    const K: [[usize; 2]; 6] = [[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]];
    let mut p = [0.0f64; 6];
    let mut total_volume = 0.0;
    for f in &faces {
        let d = vertex(&centered, f[0]);
        let e = vertex(&centered, f[1]);
        let g = vertex(&centered, f[2]);
        let (area, normal, center) = triangle(d, e, g);
        let mut v = dot3(center, normal) * area / 3.0;
        if mode == MeshInertia::Legacy {
            v = v.abs();
        }
        total_volume += v;
        for (j, k) in K.iter().enumerate() {
            let (a, b) = (k[0], k[1]);
            p[j] += v / 20.0
                * (2.0 * (d[a] * d[b] + e[a] * e[b] + g[a] * g[b])
                    + d[a] * e[b]
                    + d[b] * e[a]
                    + d[a] * g[b]
                    + d[b] * g[a]
                    + e[a] * g[b]
                    + e[b] * g[a]);
        }
    }
    let volume = total_volume;
    let inert = [p[1] + p[2], p[0] + p[2], p[0] + p[1], -p[3], -p[4], -p[5]];

    // principal axes, accurate to float32 vertices (reltol 1e-7)
    let full = [
        inert[0], inert[3], inert[4], inert[3], inert[1], inert[5], inert[4], inert[5], inert[2],
    ];
    let e = eig3(full, 1e-7);
    let eigval = e.eigval;
    if eigval[2] <= 0.0 {
        return Err("eigenvalue of mesh inertia must be positive".into());
    }
    const ATOL: f64 = 1e-9;
    const RTOL: f64 = 1e-6;
    if eigval[0] + eigval[1] < eigval[2] * (1.0 - RTOL) - ATOL
        || eigval[0] + eigval[2] < eigval[1] * (1.0 - RTOL) - ATOL
        || eigval[1] + eigval[2] < eigval[0] * (1.0 - RTOL) - ATOL
    {
        return Err("eigenvalues of mesh inertia violate A + B >= C".into());
    }
    let boxsz = [
        0.5 * (6.0 * (eigval[1] + eigval[2] - eigval[0]) / volume).sqrt(),
        0.5 * (6.0 * (eigval[0] + eigval[2] - eigval[1]) / volume).sqrt(),
        0.5 * (6.0 * (eigval[0] + eigval[1] - eigval[2]) / volume).sqrt(),
    ];

    // the box of the vertices in the principal frame (`Rotate` after the centre of
    // mass is moved to the origin, user_mesh.cc:1451-1460 and 1576-1590): each centred
    // vertex times the matrix of the conjugate quaternion, then MuJoCo's running
    // `std::min` / `std::max` from +-1e10
    let q = e.quat;
    let mat = quat2mat([q[0], -q[1], -q[2], -q[3]]);
    let mut aamm = [1e10, 1e10, 1e10, -1e10, -1e10, -1e10];
    for v in centered.chunks_exact(3) {
        let r = mulvecmat([v[0], v[1], v[2]], &mat);
        for k in 0..3 {
            // std::min(a, b) is `b < a ? b : a`, std::max(a, b) is `a < b ? b : a`
            if r[k] < aamm[k] {
                aamm[k] = r[k];
            }
            if aamm[k + 3] < r[k] {
                aamm[k + 3] = r[k];
            }
        }
    }
    let half = |k: usize| {
        let (lo, hi) = (aamm[k].abs(), aamm[k + 3].abs());
        if lo < hi { hi } else { lo }
    };
    let haabb = [half(0), half(1), half(2)];
    let rbound = (haabb[0] * haabb[0] + haabb[1] * haabb[1] + haabb[2] * haabb[2]).sqrt();

    let vertices = dvert
        .chunks_exact(3)
        .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
        .collect();
    Ok(ProcessedMesh {
        vertices,
        faces,
        volume,
        com,
        quat: e.quat,
        boxsz,
        rbound,
    })
}
