//! The host reference renderer: the kernels' algorithm, expression for
//! expression, in Rust `f32`.
//!
//! It exists to check the device. It reads the same packed scene, poses and
//! cameras and writes frames in the same byte layout, so a device frame and a
//! reference frame can be compared pixel by pixel. They are NOT expected to be
//! equal bit for bit: the device may contract multiplies and adds into fused
//! operations, its `pow`, `sin` and reciprocal are not the host's, and its
//! binary16 rounding is the driver's. The device tests compare with tolerances
//! and count disagreeing pixels instead (tests/device.rs).

use crate::camera::Intrinsics;
use crate::f16::{F16_INFINITY, f32_to_f16};
use crate::layout::{FrameLayout, STATIC_BODY};
use crate::math::{
    Pose, Vec3, add, cross, dot, length, max3, min3, mul, normalize, quat_normalize, quat_rotate,
    quat_rows, scale, sub,
};
use crate::scene::{ANALYTIC_BIT, Lighting, PackedScene};

const T_MIN: f32 = 1.0e-4;
const BARY_EPS: f32 = 1.0e-6;
const DIR_EPS: f32 = 1.0e-12;
const NO_HIT: f32 = 3.0e38;

/// A posed instance: what `prep_instances` writes for one environment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PosedInstance {
    /// Object-from-world, three rows of `[x, y, z, t]`.
    pub rows: [[f32; 4]; 3],
    /// The world box.
    pub bmin: Vec3,
    /// The world box.
    pub bmax: Vec3,
    /// The mesh's root node.
    pub root: u32,
    /// Material index.
    pub material: u16,
    /// Segmentation id.
    pub seg_id: u16,
}

/// One frame in the device's byte layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameBytes {
    /// RGB, 3 bytes a pixel, rows packed.
    pub rgb: Vec<u8>,
    /// Depth, binary16 little-endian, rows packed.
    pub depth: Vec<u8>,
    /// Segmentation, uint16 little-endian, rows packed.
    pub seg: Vec<u8>,
}

/// `prep_instances` for one environment: `body_poses` are that environment's
/// poses, indexed by body.
pub fn pose_instances(scene: &PackedScene, body_poses: &[Pose]) -> Vec<PosedInstance> {
    scene
        .instances
        .iter()
        .map(|s| {
            let (q, p) = if s.body == STATIC_BODY {
                (s.offset.rotation, s.offset.position)
            } else {
                let bp = body_poses[s.body as usize];
                let qb = quat_normalize(bp.rotation);
                let world = Pose::new(bp.position, qb).compose(&s.offset);
                (world.rotation, world.position)
            };
            let q = quat_normalize(q);
            let [ra, rb, rc] = quat_rows(q);
            let c0 = [ra[0], rb[0], rc[0]];
            let c1 = [ra[1], rb[1], rc[1]];
            let c2 = [ra[2], rb[2], rc[2]];
            let rows = [
                [c0[0], c0[1], c0[2], -dot(c0, p)],
                [c1[0], c1[1], c1[2], -dot(c1, p)],
                [c2[0], c2[1], c2[2], -dot(c2, p)],
            ];
            let lc = scale(add(s.bmin, s.bmax), 0.5);
            let le = scale(sub(s.bmax, s.bmin), 0.5);
            let wc = add([dot(ra, lc), dot(rb, lc), dot(rc, lc)], p);
            let abs3 = |v: Vec3| v.map(f32::abs);
            let we = [dot(abs3(ra), le), dot(abs3(rb), le), dot(abs3(rc), le)];
            let we = add(we, scale(add(we, [1.0e-3; 3]), 1.0e-5));
            PosedInstance {
                rows,
                bmin: sub(wc, we),
                bmax: add(wc, we),
                root: s.root,
                material: s.material,
                seg_id: s.seg_id,
            }
        })
        .collect()
}

fn safe_inverse(d: Vec3) -> Vec3 {
    d.map(|c| {
        let s = if c < 0.0 { -1.0 } else { 1.0 };
        s / c.abs().max(DIR_EPS)
    })
}

fn slab(o: Vec3, inv_d: Vec3, bmin: Vec3, bmax: Vec3, tmin: f32, tmax: f32) -> f32 {
    let t0 = mul(sub(bmin, o), inv_d);
    let t1 = mul(sub(bmax, o), inv_d);
    let lo = min3(t0, t1);
    let hi = max3(t0, t1);
    let tn = lo[0].max(lo[1]).max(lo[2].max(tmin));
    let tf = hi[0].min(hi[1]).min(hi[2].min(tmax));
    if tn <= tf { tn } else { NO_HIT }
}

#[allow(clippy::too_many_arguments)]
fn tri_hit(
    o: Vec3,
    d: Vec3,
    v0: Vec3,
    e1: Vec3,
    e2: Vec3,
    tmin: f32,
    tmax: f32,
) -> Option<(f32, f32, f32)> {
    let p = cross(d, e2);
    let det = dot(e1, p);
    if det == 0.0 {
        return None;
    }
    let inv = 1.0 / det;
    let s = sub(o, v0);
    let u = dot(s, p) * inv;
    if !(-BARY_EPS..=1.0 + BARY_EPS).contains(&u) {
        return None;
    }
    let q = cross(s, e1);
    let v = dot(d, q) * inv;
    if v < -BARY_EPS || u + v > 1.0 + BARY_EPS {
        return None;
    }
    let t = dot(e2, q) * inv;
    (t > tmin && t < tmax).then_some((t, u, v))
}

// Exact surfaces (`scene::ANALYTIC_BIT` nodes): the ANALYTIC builds'
// `analytic_hit` and `analytic_normal` (kernels/trace_common.glsl), step for
// step. The kinds are `mesh::Analytic::kind`.
const KIND_BOX: u32 = 0;
const KIND_ELLIPSOID: u32 = 1;
const KIND_CYLINDER: u32 = 2;

/// The kind and half extents of node `root`, if it is an exact surface.
fn analytic_kind(scene: &PackedScene, root: u32) -> Option<(u32, Vec3)> {
    let n = &scene.nodes[root as usize];
    (n.count & ANALYTIC_BIT != 0).then_some((n.count & 0xFF, n.bmax))
}

/// The side of a cylinder or capsule (radius `r`, |z| <= `hz` along it):
/// the nearer crossing in (tmin, best), if any, lowers `best` and sets `code`
/// to 0.
#[allow(clippy::too_many_arguments)]
fn tube_hit(r: f32, hz: f32, o: Vec3, d: Vec3, tmin: f32, best: &mut f32, code: &mut f32) {
    let a = d[0] * d[0] + d[1] * d[1];
    if a > 0.0 {
        let tc = -(o[0] * d[0] + o[1] * d[1]) / a;
        let px = o[0] + tc * d[0];
        let py = o[1] + tc * d[1];
        let h2 = r * r - (px * px + py * py);
        if h2 >= 0.0 {
            let dt = (h2 / a).sqrt();
            for t in [tc - dt, tc + dt] {
                let z = o[2] + t * d[2];
                if t > tmin && t < *best && z.abs() <= hz {
                    *best = t;
                    *code = 0.0;
                }
            }
        }
    }
}

/// The nearest crossing of an exact surface with half extents `e` within
/// (tmin, tmax): its distance, and which part of the surface was crossed as
/// the kernel stores it in `Hit::u` (box: 2 x axis, plus 1 for a face whose
/// normal points down the axis; cylinder: 0 side, 1 top, 2 bottom; ellipsoid
/// and capsule: 0).
#[allow(clippy::too_many_arguments)]
fn analytic_hit(
    kind: u32,
    e: Vec3,
    o: Vec3,
    d: Vec3,
    inv_d: Vec3,
    tmin: f32,
    tmax: f32,
) -> Option<(f32, f32)> {
    match kind {
        KIND_BOX => {
            let t0 = mul(sub(scale(e, -1.0), o), inv_d);
            let t1 = mul(sub(e, o), inv_d);
            let lo = min3(t0, t1);
            let hi = max3(t0, t1);
            let tn = lo[0].max(lo[1]).max(lo[2]);
            let tf = hi[0].min(hi[1]).min(hi[2]);
            if tn > tf {
                return None;
            }
            // entering: the face the ray enters through; from inside (two-
            // sided), the face it leaves through
            let (t, axis, up) = if tn > tmin {
                let axis = if lo[0] == tn {
                    0
                } else if lo[1] == tn {
                    1
                } else {
                    2
                };
                (tn, axis, d[axis] < 0.0)
            } else {
                let axis = if hi[0] == tf {
                    0
                } else if hi[1] == tf {
                    1
                } else {
                    2
                };
                (tf, axis, d[axis] >= 0.0)
            };
            let code = (2 * axis + usize::from(!up)) as f32;
            (t > tmin && t < tmax).then_some((t, code))
        }
        KIND_ELLIPSOID => {
            // the unit sphere in coordinates scaled by the semi-axes; the
            // closest approach first, for precision far from a small sphere
            let os = [o[0] / e[0], o[1] / e[1], o[2] / e[2]];
            let ds = [d[0] / e[0], d[1] / e[1], d[2] / e[2]];
            let a = dot(ds, ds);
            let tc = -dot(os, ds) / a;
            let pc = add(os, scale(ds, tc));
            let h2 = 1.0 - dot(pc, pc);
            if h2 < 0.0 {
                return None;
            }
            let dt = (h2 / a).sqrt();
            let mut t = tc - dt;
            if t <= tmin {
                t = tc + dt;
            }
            (t > tmin && t < tmax).then_some((t, 0.0))
        }
        KIND_CYLINDER => {
            let (r, hz) = (e[0], e[2]);
            let mut best = tmax;
            let mut code = -1.0f32;
            tube_hit(r, hz, o, d, tmin, &mut best, &mut code);
            if d[2] != 0.0 {
                for (sign, c) in [(1.0f32, 1.0f32), (-1.0, 2.0)] {
                    let t = (sign * hz - o[2]) / d[2];
                    let x = o[0] + t * d[0];
                    let y = o[1] + t * d[1];
                    if t > tmin && t < best && x * x + y * y <= r * r {
                        best = t;
                        code = c;
                    }
                }
            }
            (code >= 0.0).then_some((best, code))
        }
        _ => {
            // capsule: the tube between the cap centres, then each cap's
            // sphere where it lies beyond its centre
            let (r, hl) = (e[0], e[2] - e[0]);
            let mut best = tmax;
            let mut code = -1.0f32;
            tube_hit(r, hl, o, d, tmin, &mut best, &mut code);
            let a = dot(d, d);
            for sign in [1.0f32, -1.0] {
                let oc = [o[0], o[1], o[2] - sign * hl];
                let tc = -dot(oc, d) / a;
                let pc = add(oc, scale(d, tc));
                let h2 = r * r - dot(pc, pc);
                if h2 >= 0.0 {
                    let dt = (h2 / a).sqrt();
                    for t in [tc - dt, tc + dt] {
                        let z = o[2] + t * d[2];
                        if t > tmin && t < best && sign * z >= hl {
                            best = t;
                            code = 0.0;
                        }
                    }
                }
            }
            (code >= 0.0).then_some((best, code))
        }
    }
}

/// The outward (unnormalised) object-space normal of an exact surface at
/// `po`, a point on it, crossed at part `code` (see [`analytic_hit`]).
fn analytic_normal(kind: u32, e: Vec3, code: f32, po: Vec3) -> Vec3 {
    match kind {
        KIND_BOX => {
            let c = code as usize;
            let mut n = [0.0; 3];
            n[c / 2] = if c.is_multiple_of(2) { 1.0 } else { -1.0 };
            n
        }
        KIND_ELLIPSOID => [
            po[0] / (e[0] * e[0]),
            po[1] / (e[1] * e[1]),
            po[2] / (e[2] * e[2]),
        ],
        KIND_CYLINDER => {
            if code == 0.0 {
                [po[0], po[1], 0.0]
            } else if code == 1.0 {
                [0.0, 0.0, 1.0]
            } else {
                [0.0, 0.0, -1.0]
            }
        }
        _ => {
            let hl = e[2] - e[0];
            [po[0], po[1], po[2] - po[2].clamp(-hl, hl)]
        }
    }
}

#[derive(Clone, Copy)]
struct Hit {
    t: f32,
    u: f32,
    v: f32,
    inst: u32,
    tri: u32,
}

// the kernel's signature, argument for argument
#[allow(clippy::too_many_arguments)]
fn trace_blas(
    scene: &PackedScene,
    o: Vec3,
    d: Vec3,
    root: u32,
    inst: u32,
    tmin: f32,
    tmax: f32,
    h: &mut Hit,
) {
    let inv_d = safe_inverse(d);
    if let Some((kind, e)) = analytic_kind(scene, root) {
        if let Some((t, code)) = analytic_hit(kind, e, o, d, inv_d, tmin, h.t.min(tmax))
            && t < h.t
        {
            *h = Hit {
                t,
                u: code,
                v: 0.0,
                inst,
                tri: root,
            };
        }
        return;
    }
    let mut stack: Vec<(u32, f32)> = Vec::with_capacity(32);
    let rn = &scene.nodes[root as usize];
    if slab(o, inv_d, rn.bmin, rn.bmax, tmin, h.t.min(tmax)) >= NO_HIT {
        return;
    }
    let mut node = root;
    loop {
        let nd = &scene.nodes[node as usize];
        let mut pop = false;
        if nd.count > 0 {
            for k in 0..nd.count {
                let ti = nd.left_or_first + k;
                let tr = &scene.tris[ti as usize];
                if let Some((t, u, v)) = tri_hit(o, d, tr.v0, tr.e1, tr.e2, tmin, h.t.min(tmax))
                    && t < h.t
                {
                    *h = Hit {
                        t,
                        u,
                        v,
                        inst,
                        tri: ti,
                    };
                }
            }
            pop = true;
        } else {
            let (l, r) = (nd.left_or_first, nd.left_or_first + 1);
            let limit = h.t.min(tmax);
            let nl = &scene.nodes[l as usize];
            let nr = &scene.nodes[r as usize];
            let tl = slab(o, inv_d, nl.bmin, nl.bmax, tmin, limit);
            let tr = slab(o, inv_d, nr.bmin, nr.bmax, tmin, limit);
            match (tl < NO_HIT, tr < NO_HIT) {
                (true, true) => {
                    let (near, far, far_t) = if tr < tl { (r, l, tl) } else { (l, r, tr) };
                    stack.push((far, far_t));
                    node = near;
                }
                (true, false) => node = l,
                (false, true) => node = r,
                (false, false) => pop = true,
            }
        }
        if pop {
            let mut found = false;
            while let Some((n, t)) = stack.pop() {
                if t <= h.t.min(tmax) {
                    node = n;
                    found = true;
                    break;
                }
            }
            if !found {
                return;
            }
        }
    }
}

fn occluded_blas(scene: &PackedScene, o: Vec3, d: Vec3, root: u32, tmax: f32) -> bool {
    let inv_d = safe_inverse(d);
    if let Some((kind, e)) = analytic_kind(scene, root) {
        return analytic_hit(kind, e, o, d, inv_d, T_MIN, tmax).is_some();
    }
    let rn = &scene.nodes[root as usize];
    if slab(o, inv_d, rn.bmin, rn.bmax, T_MIN, tmax) >= NO_HIT {
        return false;
    }
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let nd = &scene.nodes[node as usize];
        if nd.count > 0 {
            for k in 0..nd.count {
                let tr = &scene.tris[(nd.left_or_first + k) as usize];
                if tri_hit(o, d, tr.v0, tr.e1, tr.e2, T_MIN, tmax).is_some() {
                    return true;
                }
            }
        } else {
            for c in [nd.left_or_first + 1, nd.left_or_first] {
                let n = &scene.nodes[c as usize];
                if slab(o, inv_d, n.bmin, n.bmax, T_MIN, tmax) < NO_HIT {
                    stack.push(c);
                }
            }
        }
    }
    false
}

fn to_object(x: &PosedInstance, p: Vec3) -> Vec3 {
    let r = &x.rows;
    [
        dot([r[0][0], r[0][1], r[0][2]], p) + r[0][3],
        dot([r[1][0], r[1][1], r[1][2]], p) + r[1][3],
        dot([r[2][0], r[2][1], r[2][2]], p) + r[2][3],
    ]
}

fn dir_to_object(x: &PosedInstance, d: Vec3) -> Vec3 {
    let r = &x.rows;
    [
        dot([r[0][0], r[0][1], r[0][2]], d),
        dot([r[1][0], r[1][1], r[1][2]], d),
        dot([r[2][0], r[2][1], r[2][2]], d),
    ]
}

fn normal_to_world(x: &PosedInstance, n: Vec3) -> Vec3 {
    let r = &x.rows;
    add(
        add(
            scale([r[0][0], r[0][1], r[0][2]], n[0]),
            scale([r[1][0], r[1][1], r[1][2]], n[1]),
        ),
        scale([r[2][0], r[2][1], r[2][2]], n[2]),
    )
}

fn sun_occluded(scene: &PackedScene, posed: &[PosedInstance], p: Vec3, l: Vec3) -> bool {
    let inv_l = safe_inverse(l);
    posed.iter().any(|x| {
        slab(p, inv_l, x.bmin, x.bmax, T_MIN, NO_HIT) < NO_HIT
            && occluded_blas(scene, to_object(x, p), dir_to_object(x, l), x.root, NO_HIT)
    })
}

fn sky(l: &Lighting, d: Vec3) -> Vec3 {
    let z = d[2] / length(d);
    if z < 0.0 {
        return l.ground;
    }
    let horizon = (1.0 - z).powf(4.0);
    scale(l.sky_zenith, 1.0 + l.horizon_boost * horizon)
}

fn mix3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    add(scale(a, 1.0 - t), scale(b, t))
}

/// The checker's parity at `po` (object space, in cells) on a face whose
/// geometric normal is `ng`: a 2-D checker in the face's plane, dropping the
/// coordinate along the normal's dominant axis (ties to the lower axis), as
/// `trace_v0.comp` computes it.
pub fn checker_is_dark(po: Vec3, ng: Vec3) -> bool {
    let an = ng.map(f32::abs);
    let cell = |k: usize| po[k].floor() as i32;
    let c = if an[0] >= an[1] && an[0] >= an[2] {
        cell(1) + cell(2)
    } else if an[1] >= an[2] {
        cell(0) + cell(2)
    } else {
        cell(0) + cell(1)
    };
    c & 1 == 1
}

fn shade(scene: &PackedScene, posed: &[PosedInstance], h: &Hit, o: Vec3, d: Vec3) -> Vec3 {
    use std::f32::consts::PI;
    let x = &posed[h.inst as usize];
    let m = &scene.materials[x.material as usize];
    let (n_obj, ng_obj) = if let Some((kind, e)) = analytic_kind(scene, x.root) {
        // an exact surface: its normal is both the shading and the geometric one
        let n = analytic_normal(kind, e, h.u, to_object(x, add(o, scale(d, h.t))));
        (n, n)
    } else {
        let tr = &scene.tris[h.tri as usize];
        let tn = &scene.tri_normals[h.tri as usize];
        let w0 = 1.0 - h.u - h.v;
        (
            add(add(scale(tn[0], w0), scale(tn[1], h.u)), scale(tn[2], h.v)),
            cross(tr.e1, tr.e2),
        )
    };
    let mut n = normalize(normal_to_world(x, n_obj));
    let mut ng = normalize(normal_to_world(x, ng_obj));
    if dot(ng, d) > 0.0 {
        ng = scale(ng, -1.0);
    }
    if dot(n, ng) < 0.0 {
        n = scale(n, -1.0);
    }
    let p = add(o, scale(d, h.t));

    let mut base = m.base_colour;
    if m.checker_cells_per_metre > 0.0 {
        // the kernel's 2-D checker in the face's plane (see trace_v0.comp)
        let po = scale(to_object(x, p), m.checker_cells_per_metre);
        if checker_is_dark(po, ng_obj) {
            base = scale(base, m.checker_dark);
        }
    }
    let lt = &scene.lighting;
    let v = scale(normalize(d), -1.0);
    let l = lt.sun_direction;
    let nl = dot(n, l).max(0.0);
    let mut vis = 1.0;
    if lt.shadows && nl > 0.0 && dot(ng, l) > 0.0 {
        let origin = add(p, scale(ng, 1.0e-4 + 1.0e-5 * h.t));
        if sun_occluded(scene, posed, origin, l) {
            vis = 0.0;
        }
    }
    if dot(ng, l) <= 0.0 {
        vis = 0.0;
    }
    let hv = normalize(add(l, v));
    let nh = dot(n, hv).max(0.0);
    let nv = dot(n, v).max(1.0e-4);
    let vh = dot(v, hv).max(0.0);
    let a = m.roughness * m.roughness;
    let a2 = a * a;
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    let dist = a2 / (PI * dd * dd);
    let k = (m.roughness + 1.0) * (m.roughness + 1.0) * 0.125;
    let g = (nl / (nl * (1.0 - k) + k)) * (nv / (nv * (1.0 - k) + k));
    let f0 = mix3([0.04; 3], base, m.metallic);
    let fres = add(f0, scale(sub([1.0; 3], f0), (1.0 - vh).powf(5.0)));
    let spec = scale(fres, dist * g / (4.0 * nl * nv).max(1.0e-4));
    let kd = scale(sub([1.0; 3], fres), 1.0 - m.metallic);
    let mut radiance = scale(
        mul(add(scale(mul(kd, base), 1.0 / PI), spec), lt.sun_irradiance),
        nl * vis,
    );
    let up = 0.5 + 0.5 * n[2];
    let amb = mix3(lt.ground, lt.sky_zenith, up);
    radiance = add(radiance, mul(amb, mul(kd, base)));
    radiance = add(radiance, scale(mul(amb, f0), (1.0 - m.roughness) * 0.5));
    add(radiance, m.emission)
}

fn encode_srgb8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let s = if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// The tone curve of the kernels: exposure, the ACES fit, sRGB 8-bit.
pub fn tonemap(radiance: Vec3, exposure: f32) -> [u8; 3] {
    radiance.map(|r| {
        let c = r * exposure;
        let c = (c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14);
        encode_srgb8(c)
    })
}

/// One pixel's outputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PixelSample {
    /// RGB.
    pub rgb: [u8; 3],
    /// Depth in metres, `inf` for no hit.
    pub depth: f32,
    /// Segmentation id, 0 for no hit.
    pub seg: u16,
}

/// Render the pixel at (`column`, `row`).
pub fn render_pixel(
    scene: &PackedScene,
    posed: &[PosedInstance],
    cam: &Pose,
    k: &Intrinsics,
    column: u32,
    row: u32,
) -> PixelSample {
    let q = quat_normalize(cam.rotation);
    let o = cam.position;
    let fx = column as f32;
    let fy = row as f32;
    let dc = [(fx - k.cx) / k.fx, (fy - k.cy) / k.fy, 1.0];
    let d = quat_rotate(q, dc);
    let inv_d = safe_inverse(d);
    let mut h = Hit {
        t: NO_HIT,
        u: 0.0,
        v: 0.0,
        inst: u32::MAX,
        tri: 0,
    };
    for (i, x) in posed.iter().enumerate() {
        if slab(o, inv_d, x.bmin, x.bmax, k.near, h.t.min(k.far)) >= NO_HIT {
            continue;
        }
        trace_blas(
            scene,
            to_object(x, o),
            dir_to_object(x, d),
            x.root,
            i as u32,
            k.near,
            k.far,
            &mut h,
        );
    }
    let lt = &scene.lighting;
    if h.inst != u32::MAX && h.t >= k.near && h.t <= k.far {
        PixelSample {
            rgb: tonemap(shade(scene, posed, &h, o, d), lt.exposure),
            depth: h.t,
            seg: posed[h.inst as usize].seg_id,
        }
    } else {
        PixelSample {
            rgb: tonemap(sky(lt, d), lt.exposure),
            depth: f32::INFINITY,
            seg: 0,
        }
    }
}

/// Render a whole frame in the device's byte layout.
pub fn render_frame(
    scene: &PackedScene,
    posed: &[PosedInstance],
    cam: &Pose,
    k: &Intrinsics,
    layout: FrameLayout,
) -> FrameBytes {
    let px = layout.pixels() as usize;
    let mut out = FrameBytes {
        rgb: Vec::with_capacity(3 * px),
        depth: Vec::with_capacity(2 * px),
        seg: Vec::with_capacity(2 * px),
    };
    for row in 0..layout.height {
        for column in 0..layout.width {
            let s = render_pixel(scene, posed, cam, k, column, row);
            out.rgb.extend_from_slice(&s.rgb);
            let bits = if s.depth.is_finite() {
                f32_to_f16(s.depth)
            } else {
                F16_INFINITY
            };
            out.depth.extend_from_slice(&bits.to_le_bytes());
            out.seg.extend_from_slice(&s.seg.to_le_bytes());
        }
    }
    out
}

fn pcg(v: u32) -> u32 {
    let state = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// `animate_poses` (the benchmark's physics stand-in) for one environment:
/// the pose of every body at `tick`, as the kernel computes it.
pub fn animate_poses(
    motions: &[crate::scenes::Motion],
    env: u32,
    tick: u32,
    seed: u32,
    dt: f32,
) -> Vec<Pose> {
    let phase = (pcg(pcg(seed) ^ env) >> 8) as f32 * (std::f32::consts::TAU / 16_777_216.0);
    let t = tick as f32 * dt;
    motions
        .iter()
        .map(|m| {
            let theta = m.omega * t + phase;
            let (s, c) = theta.sin_cos();
            let rel = sub(m.rest.position, m.pivot);
            let wobble = (std::f32::consts::TAU * m.frequency * t + phase).sin();
            let p = add(
                add(
                    [c * rel[0] - s * rel[1], s * rel[0] + c * rel[1], rel[2]],
                    m.pivot,
                ),
                scale(m.amplitude, wobble),
            );
            let qz = [0.0, 0.0, (0.5 * theta).sin(), (0.5 * theta).cos()];
            Pose::new(p, crate::math::quat_mul(qz, m.rest.rotation))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh;
    use crate::scene::{Instance, Material, SceneDesc};

    fn lighting() -> Lighting {
        Lighting {
            sun_direction: normalize([0.3, 0.2, 0.9]),
            sun_irradiance: [3.0; 3],
            sky_zenith: [0.3, 0.4, 0.6],
            horizon_boost: 1.0,
            ground: [0.1; 3],
            exposure: 1.0,
            shadows: true,
        }
    }

    /// The BVH traversal finds exactly the closest triangle a brute-force loop
    /// over every triangle finds.
    #[test]
    fn the_traversal_agrees_with_brute_force() {
        let mesh = Mesh::sphere(0.5, 24, 12);
        let scene = SceneDesc {
            meshes: vec![mesh],
            materials: vec![Material::plain([0.5; 3], 0.5)],
            instances: vec![Instance {
                body: None,
                mesh: 0,
                material: 0,
                offset: Pose::IDENTITY,
                seg_id: 1,
            }],
            bodies_per_env: 0,
            lighting: lighting(),
        }
        .pack_triangles()
        .unwrap();
        let mut state = 0x9E37_79B9u32;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let mut hits = 0;
        for _ in 0..2000 {
            let o = [2.0 * rnd(), 2.0 * rnd(), 2.0 * rnd()];
            // within 0.35 m of the centre, inside the facets (at least 0.48 m out)
            let target = [0.2 * rnd(), 0.2 * rnd(), 0.2 * rnd()];
            let d = sub(target, o);
            let mut h = Hit {
                t: NO_HIT,
                u: 0.0,
                v: 0.0,
                inst: u32::MAX,
                tri: 0,
            };
            trace_blas(&scene, o, d, 0, 0, T_MIN, NO_HIT, &mut h);
            let mut best = NO_HIT;
            for tr in &scene.tris {
                if let Some((t, _, _)) = tri_hit(o, d, tr.v0, tr.e1, tr.e2, T_MIN, NO_HIT) {
                    best = best.min(t);
                }
            }
            assert_eq!(h.t, best);
            if best < NO_HIT {
                hits += 1;
            }
        }
        // every ray aims inside the polyhedron, so every ray hits it
        assert_eq!(hits, 2000);
    }

    /// The defect the device test found on 2026-10-01: a face lying on a
    /// checker cell boundary (a floor at z = 0) must not change colour with
    /// the rounding of the coordinate along its normal.
    #[test]
    fn the_checker_does_not_flip_on_a_face_that_lies_on_a_cell_boundary() {
        let up = [0.0, 0.0, 1.0];
        for (x, y) in [(0.3, 0.3), (1.7, 0.2), (-0.4, 2.6), (-1.2, -3.9)] {
            let below = checker_is_dark([x, y, -1.0e-7], up);
            let above = checker_is_dark([x, y, 1.0e-7], up);
            assert_eq!(below, above, "({x}, {y})");
        }
        // and it is still a checker: neighbouring cells differ
        assert_ne!(
            checker_is_dark([0.5, 0.5, 0.0], up),
            checker_is_dark([1.5, 0.5, 0.0], up)
        );
        // a wall facing x uses y and z
        assert_ne!(
            checker_is_dark([0.0, 0.5, 0.5], [1.0, 0.0, 0.0]),
            checker_is_dark([0.0, 0.5, 1.5], [1.0, 0.0, 0.0])
        );
    }

    /// A camera straight above an infinite-enough floor sees the floor at its
    /// height in every pixel: depth is distance along the axis, not the ray.
    #[test]
    fn a_floor_seen_from_above_has_the_camera_height_as_depth() {
        let scene = SceneDesc {
            meshes: vec![Mesh::plane(50.0, 50.0, 4, 4)],
            materials: vec![Material::plain([0.5; 3], 0.8)],
            instances: vec![Instance {
                body: None,
                mesh: 0,
                material: 0,
                offset: Pose::IDENTITY,
                seg_id: 7,
            }],
            bodies_per_env: 0,
            lighting: lighting(),
        }
        .pack()
        .unwrap();
        let posed = pose_instances(&scene, &[]);
        let cam =
            crate::camera::look_at([0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]).unwrap();
        let k = Intrinsics::from_hfov(32, 32, 1.0, 0.01, 100.0);
        for (c, r) in [(0, 0), (31, 0), (16, 16), (5, 27)] {
            let s = render_pixel(&scene, &posed, &cam, &k, c, r);
            assert_eq!(s.seg, 7);
            assert!((s.depth - 2.0).abs() < 1e-5, "{}", s.depth);
        }
    }

    fn one_mesh_scene(mesh: Mesh, analytic: bool) -> PackedScene {
        SceneDesc {
            meshes: vec![mesh],
            materials: vec![Material::plain([0.5; 3], 0.5)],
            instances: vec![Instance {
                body: None,
                mesh: 0,
                material: 0,
                offset: Pose::IDENTITY,
                seg_id: 1,
            }],
            bodies_per_env: 0,
            lighting: lighting(),
        }
        .pack_with(crate::scene::PackOptions {
            analytic,
            ..Default::default()
        })
        .unwrap()
    }

    fn xorshift(state: &mut u32) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        (*state as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    fn closest(scene: &PackedScene, o: Vec3, d: Vec3) -> Hit {
        let mut h = Hit {
            t: NO_HIT,
            u: 0.0,
            v: 0.0,
            inst: u32::MAX,
            tri: 0,
        };
        trace_blas(scene, o, d, 0, 0, T_MIN, NO_HIT, &mut h);
        h
    }

    /// An exact box is hit where its 12 triangles are, with the same face
    /// normal, for rays from every side towards points inside it.
    #[test]
    fn an_exact_box_agrees_with_its_triangles() {
        let half = [0.5, 0.2, 0.1];
        let tris = one_mesh_scene(Mesh::cuboid(half), false);
        let exact = one_mesh_scene(Mesh::cuboid(half), true);
        assert!(exact.analytic && exact.tris.is_empty() && !tris.analytic);
        let (kind, e) = analytic_kind(&exact, 0).unwrap();
        let mut state = 0x1234_5678u32;
        for _ in 0..2000 {
            let o = [
                3.0 * xorshift(&mut state),
                3.0 * xorshift(&mut state),
                3.0 * xorshift(&mut state),
            ];
            let target = [
                0.9 * half[0] * xorshift(&mut state),
                0.9 * half[1] * xorshift(&mut state),
                0.9 * half[2] * xorshift(&mut state),
            ];
            let d = sub(target, o);
            let (a, b) = (closest(&tris, o, d), closest(&exact, o, d));
            assert!(a.inst == 0 && b.inst == 0, "o {o:?} d {d:?}");
            assert!(
                (a.t - b.t).abs() <= 1e-5 * a.t.max(1.0),
                "t {} vs {}",
                a.t,
                b.t
            );
            let n = analytic_normal(kind, e, b.u, add(o, scale(d, b.t)));
            let tn = exact_normal_of(&tris, a.tri);
            assert!(dot(normalize(n), tn) > 0.9999, "{n:?} vs {tn:?}");
        }
    }

    fn exact_normal_of(scene: &PackedScene, tri: u32) -> Vec3 {
        let t = &scene.tris[tri as usize];
        normalize(cross(t.e1, t.e2))
    }

    /// The curved surfaces: every ray towards a point inside is hit ON the
    /// surface (its equation holds), with an outward normal, at the nearest
    /// crossing (just before it lies outside); rays from the centre hit the
    /// surface from inside (two-sided); rays that pass wide miss.
    #[test]
    fn exact_curved_surfaces_are_hit_on_their_equations() {
        let ellipsoid = |e: Vec3| {
            move |p: Vec3| -> f32 {
                ((p[0] / e[0]).powi(2) + (p[1] / e[1]).powi(2) + (p[2] / e[2]).powi(2)).sqrt() - 1.0
            }
        };
        let cylinder = |r: f32, hz: f32| {
            move |p: Vec3| -> f32 {
                ((p[0] * p[0] + p[1] * p[1]).sqrt() / r - 1.0).max(p[2].abs() / hz - 1.0)
            }
        };
        let capsule = |r: f32, hl: f32| {
            move |p: Vec3| -> f32 {
                let z = p[2].clamp(-hl, hl);
                length([p[0], p[1], p[2] - z]) / r - 1.0
            }
        };
        type Field = Box<dyn Fn(Vec3) -> f32>;
        let cases: Vec<(&str, Mesh, Field, f32)> = vec![
            (
                "sphere",
                Mesh::sphere(0.04, 8, 4),
                Box::new(ellipsoid([0.04; 3])),
                0.04,
            ),
            (
                "ellipsoid",
                Mesh::ellipsoid([0.3, 0.2, 0.1], 8, 4),
                Box::new(ellipsoid([0.3, 0.2, 0.1])),
                0.1,
            ),
            (
                "cylinder",
                Mesh::cylinder(0.05, 0.1, 8),
                Box::new(cylinder(0.05, 0.1)),
                0.05,
            ),
            (
                "capsule",
                Mesh::capsule(0.05, 0.1, 8, 2),
                Box::new(capsule(0.05, 0.1)),
                0.05,
            ),
        ];
        let mut state = 0x9E37_79B9u32;
        for (name, mesh, f, size) in cases {
            let scene = one_mesh_scene(mesh, true);
            let (kind, e) = analytic_kind(&scene, 0).unwrap();
            for _ in 0..1000 {
                let o = [
                    2.0 * xorshift(&mut state),
                    2.0 * xorshift(&mut state),
                    2.0 * xorshift(&mut state),
                ];
                // a point well inside: a fifth of the smallest extent from the centre
                let target = [
                    0.2 * size * xorshift(&mut state),
                    0.2 * size * xorshift(&mut state),
                    0.2 * size * xorshift(&mut state),
                ];
                let d = sub(target, o);
                let h = closest(&scene, o, d);
                assert_eq!(h.inst, 0, "{name}: missed, o {o:?}");
                let p = add(o, scale(d, h.t));
                assert!(f(p).abs() < 1e-4, "{name}: off the surface by {}", f(p));
                let before = add(o, scale(d, h.t * (1.0 - 1e-3)));
                assert!(f(before) > 0.0, "{name}: not the nearest crossing");
                let n = analytic_normal(kind, e, h.u, p);
                let outside = add(p, scale(normalize(n), 1e-3 * size));
                assert!(
                    f(outside) > 0.0,
                    "{name}: the normal points inwards at {p:?}"
                );
                // from the centre, the ray meets the surface from inside
                let hi = closest(&scene, [0.0; 3], d);
                assert_eq!(hi.inst, 0, "{name}: no hit from inside");
                assert!(
                    f(scale(d, hi.t)).abs() < 1e-4,
                    "{name}: inside hit off the surface"
                );
                // passing 3 sizes wide of the centre misses
                let side = normalize(cross(d, [0.3, 0.5, 0.7]));
                let wide = add(
                    o,
                    scale(side, 3.0 * e.iter().fold(0.0f32, |a, &b| a.max(b))),
                );
                let hw = closest(&scene, wide, d);
                assert_eq!(hw.inst, u32::MAX, "{name}: hit a ray that passes wide");
            }
        }
    }

    /// The tabletop packed as exact surfaces renders the same scene as its
    /// tessellation: the same object in all but a sliver of pixels (curved
    /// silhouettes move by the tessellation's error), the same depth where
    /// the boxes and floor are seen.
    #[test]
    fn the_exact_tabletop_matches_its_tessellation() {
        let (desc, motions) = crate::scenes::tabletop();
        let tris = desc.pack_triangles().unwrap();
        let exact = desc
            .pack_with(crate::scene::PackOptions {
                analytic: true,
                ..Default::default()
            })
            .unwrap();
        assert!(
            exact.tris.is_empty(),
            "every tabletop mesh has an exact surface"
        );
        let rest: Vec<Pose> = motions.iter().map(|m| m.rest).collect();
        let l = FrameLayout {
            width: 112,
            height: 112,
        };
        let (cam, k) = crate::scenes::tabletop_camera(0, l.width, l.height);
        let a = render_frame(&tris, &pose_instances(&tris, &rest), &cam, &k, l);
        let b = render_frame(&exact, &pose_instances(&exact, &rest), &cam, &k, l);
        let px = (l.width * l.height) as usize;
        let seg = |f: &FrameBytes, i: usize| u16::from_le_bytes([f.seg[2 * i], f.seg[2 * i + 1]]);
        let depth = |f: &FrameBytes, i: usize| {
            crate::f16::f16_to_f32(u16::from_le_bytes([f.depth[2 * i], f.depth[2 * i + 1]]))
        };
        let mut seg_off = 0;
        let mut depth_off = 0;
        for i in 0..px {
            if seg(&a, i) != seg(&b, i) {
                seg_off += 1;
            } else if seg(&a, i) != 0 && (depth(&a, i) - depth(&b, i)).abs() > 1e-3 * depth(&a, i) {
                depth_off += 1;
            }
        }
        println!(
            "{seg_off} of {px} pixels see another object; {depth_off} differ in depth by > 0.1%"
        );
        assert!(seg_off * 200 < px, "{seg_off} of {px}");
        assert!(depth_off * 100 < px, "{depth_off} of {px}");
    }

    /// A plane keeps MuJoCo's drawn size as an exact surface (as the physics
    /// collides with it): rays from above hit inside its half extents
    /// and miss just outside them.
    #[test]
    fn an_exact_plane_ends_at_its_drawn_half_extents() {
        let scene = one_mesh_scene(Mesh::plane(2.5, 1.0, 1, 1), true);
        let (_, e) = analytic_kind(&scene, 0).unwrap();
        assert_eq!(e, [2.5, 1.0, 0.0]);
        let down = [0.0, 0.0, -1.0];
        for (x, y, hit) in [
            (2.45, 0.0, true),
            (2.55, 0.0, false),
            (0.0, 0.95, true),
            (0.0, 1.05, false),
            (-2.45, -0.95, true),
            (-2.55, 0.0, false),
        ] {
            let h = closest(&scene, [x, y, 3.0], down);
            assert_eq!(h.inst == 0, hit, "({x}, {y})");
            if hit {
                assert!((h.t - 3.0).abs() < 1e-6, "{}", h.t);
            }
        }
    }
}
