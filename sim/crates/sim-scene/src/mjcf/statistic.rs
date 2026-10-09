//! The model statistic MuJoCo derives when it compiles a model: the bounding box of
//! the model in its reference pose, the model's extent and the mean body size.
//!
//! Ports, from MuJoCo 3.14.0 (Apache-2.0,
//! (c) DeepMind Technologies Limited):
//! - `setStat` (engine_setconst.c:1196-1322): the box over body positions, centres of
//!   mass, joint anchors, sites and geoms (each geom grown by its bounding radius; a
//!   plane by a tenth of its larger half-size, or 0.01 if it is infinite), the extent
//!   as the box's longest side, the mean body size, and the extent raised to twice the
//!   mean body size;
//! - the forward kinematics `setStat` runs on, at `qpos0` (`mj_kinematics`,
//!   engine_core_smooth.c, as ported by `sim-physics`, crates/sim-physics/src/smooth.rs,
//!   and `mj_local2Global`, engine_core_util.c:976-1016), with the frame shortcuts of
//!   `setSameframe` (engine_setconst.c:89-190);
//! - the vector and quaternion helpers `mju_normalize4`, `mji_mulMatVec3`,
//!   `mji_rotVecQuat`, `mji_mulQuat`, `mju_quat2Mat` (engine_util_blas.c,
//!   engine_inline.h, engine_util_spatial.c) and `mju_dist3`, `mju_max`, `mjMIN`,
//!   `mjMAX` (engine_util_blas.c:144-147, engine_util_misc.c:1736-1744, mjmacro.h).
//!
//! The scene uses only the extent: MuJoCo's renderer puts a camera's near clip plane
//! at `znear * extent` and its far plane at `zfar * extent` (`mjv_cameraFrustum`,
//! engine_vis_visualize.c:575-588), and `<statistic extent>` overrides the computed
//! extent after `setStat` has run (`mjCModel::TryCompile`, user_model.cc:5800-5811).
//!
//! Invariants:
//! - The arithmetic is MuJoCo's, operation for operation, in `f64`: the engine's
//!   helpers (`mji_*`, `mju_*`), not the compiler's (`mjuu_*` in `mjmath.rs`), because
//!   `setStat` runs on the compiled model with the engine. The parity test holds the
//!   extent and the mean size to MuJoCo's own.
//! - At `qpos0` a free joint takes its body's pose from `qpos0`, which is the body's
//!   own pose; a hinge or slide joint is at 0 and a ball joint at the identity, so
//!   each local motion is the identity. MuJoCo still recomposes the body position
//!   about each hinge and ball anchor (`xanchor - R jnt_pos`, which can round away
//!   from the body's own position) and adds `axis * 0` for a slide; so does this port.
//! - The inputs are the compiled frames in MuJoCo's order: bodies depth first (the
//!   world is body 0), joints, geoms and sites listed by body. A mesh geom's frame is
//!   MuJoCo's, moved to the mesh's centre of mass and principal axes, with MuJoCo's
//!   bounding radius for it (`ProcessedMesh::rbound`).

/// MuJoCo's `mjMINVAL`.
const MINVAL: f64 = 1e-15;

/// `kSameFrameEps` (engine_setconst.c:89): frames closer than this are the same.
const SAME_FRAME_EPS: f64 = 1e-6;

/// The kind of a joint, as the kinematics needs it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum StatJointKind {
    Free,
    Ball,
    Hinge,
    Slide { axis: [f64; 3] },
}

/// A body: `parent` is 0 for a child of the world; quaternions are `[w, x, y, z]`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StatBody {
    pub parent: usize,
    pub pos: [f64; 3],
    pub quat: [f64; 4],
    /// The inertial frame (zero and the identity for a massless body).
    pub ipos: [f64; 3],
    pub iquat: [f64; 4],
}

/// A joint of body `body` (1-based: the world has none).
#[derive(Clone, Copy, Debug)]
pub(crate) struct StatJoint {
    pub body: usize,
    pub kind: StatJointKind,
    pub pos: [f64; 3],
}

/// A geom of body `body` (0 is the world). `rbound` is MuJoCo's `geom_rbound` (0 for
/// a plane), `plane` a plane's two half-sizes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StatGeom {
    pub body: usize,
    pub pos: [f64; 3],
    pub quat: [f64; 4],
    pub rbound: f64,
    pub plane: Option<[f64; 2]>,
}

/// A site of body `body` (0 is the world).
#[derive(Clone, Copy, Debug)]
pub(crate) struct StatSite {
    pub body: usize,
    pub pos: [f64; 3],
    pub quat: [f64; 4],
}

/// The model statistic (MuJoCo's `mjStatistic`, the rows `setStat` computes from the
/// geometry). As returned by the importer, the document's `<statistic extent meansize
/// center>` replace the computed values, as they do in MuJoCo's compiled model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Statistic {
    /// The centre of the model's bounding box in the reference pose, metres.
    pub center: [f64; 3],
    /// The model's size: the longest side of the bounding box, raised to at least
    /// twice the mean body size, metres. MuJoCo's renderer scales its clip planes
    /// by it.
    pub extent: f64,
    /// The mean body size, metres.
    pub meansize: f64,
}

/// `mjSameFrame`: how a frame relates to its body's frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SameFrame {
    None,
    Body,
    Inertia,
    BodyRot,
    InertiaRot,
}

fn is_null_vec3(v: [f64; 3]) -> bool {
    v[0].abs() < SAME_FRAME_EPS && v[1].abs() < SAME_FRAME_EPS && v[2].abs() < SAME_FRAME_EPS
}

/// `isNullQuat`: near the identity, accounting for the double cover.
fn is_null_quat(q: [f64; 4]) -> bool {
    let plus = (q[0] - 1.0).abs() < SAME_FRAME_EPS
        && q[1].abs() < SAME_FRAME_EPS
        && q[2].abs() < SAME_FRAME_EPS
        && q[3].abs() < SAME_FRAME_EPS;
    let minus = (q[0] + 1.0).abs() < SAME_FRAME_EPS
        && q[1].abs() < SAME_FRAME_EPS
        && q[2].abs() < SAME_FRAME_EPS
        && q[3].abs() < SAME_FRAME_EPS;
    plus || minus
}

/// `isSameQuat`: near equal, accounting for the double cover.
fn is_same_quat(a: [f64; 4], b: [f64; 4]) -> bool {
    let plus = (0..4).all(|k| (a[k] - b[k]).abs() < SAME_FRAME_EPS);
    let minus = (0..4).all(|k| (a[k] + b[k]).abs() < SAME_FRAME_EPS);
    plus || minus
}

fn is_same_vec3(a: [f64; 3], b: [f64; 3]) -> bool {
    (0..3).all(|k| (a[k] - b[k]).abs() < SAME_FRAME_EPS)
}

/// `setSameframe`'s rule for a body's inertial frame.
fn body_sameframe(ipos: [f64; 3], iquat: [f64; 4]) -> SameFrame {
    if is_null_vec3(ipos) && is_null_quat(iquat) {
        SameFrame::Body
    } else if is_null_quat(iquat) {
        SameFrame::BodyRot
    } else {
        SameFrame::None
    }
}

/// `setSameframe`'s rule for a geom or site frame in a body with inertial frame
/// `(ipos, iquat)`.
fn frame_sameframe(pos: [f64; 3], quat: [f64; 4], ipos: [f64; 3], iquat: [f64; 4]) -> SameFrame {
    if is_null_vec3(pos) && is_null_quat(quat) {
        SameFrame::Body
    } else if is_null_quat(quat) {
        SameFrame::BodyRot
    } else if is_same_vec3(pos, ipos) && is_same_quat(quat, iquat) {
        SameFrame::Inertia
    } else if is_same_quat(quat, iquat) {
        SameFrame::InertiaRot
    } else {
        SameFrame::None
    }
}

/// `mju_normalize4`.
fn normalize4(v: &mut [f64; 4]) {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
    if norm < MINVAL {
        *v = [1.0, 0.0, 0.0, 0.0];
    } else if (norm - 1.0).abs() > MINVAL {
        let inv = 1.0 / norm;
        for c in v.iter_mut() {
            *c *= inv;
        }
    }
}

/// `mji_mulMatVec3`.
fn mul_mat_vec3(mat: &[f64; 9], vec: [f64; 3]) -> [f64; 3] {
    [
        mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2],
        mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2],
        mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2],
    ]
}

/// `mji_rotVecQuat`, the general path (it gives the null quaternion's result exactly).
fn rot_vec_quat(vec: [f64; 3], quat: [f64; 4]) -> [f64; 3] {
    let tmp = [
        quat[0] * vec[0] + quat[2] * vec[2] - quat[3] * vec[1],
        quat[0] * vec[1] + quat[3] * vec[0] - quat[1] * vec[2],
        quat[0] * vec[2] + quat[1] * vec[1] - quat[2] * vec[0],
    ];
    [
        vec[0] + 2.0 * (quat[2] * tmp[2] - quat[3] * tmp[1]),
        vec[1] + 2.0 * (quat[3] * tmp[0] - quat[1] * tmp[2]),
        vec[2] + 2.0 * (quat[1] * tmp[1] - quat[2] * tmp[0]),
    ]
}

/// `mji_mulQuat`.
fn mul_quat(qa: [f64; 4], qb: [f64; 4]) -> [f64; 4] {
    [
        qa[0] * qb[0] - qa[1] * qb[1] - qa[2] * qb[2] - qa[3] * qb[3],
        qa[0] * qb[1] + qa[1] * qb[0] + qa[2] * qb[3] - qa[3] * qb[2],
        qa[0] * qb[2] - qa[1] * qb[3] + qa[2] * qb[0] + qa[3] * qb[1],
        qa[0] * qb[3] + qa[1] * qb[2] - qa[2] * qb[1] + qa[3] * qb[0],
    ]
}

/// `mju_quat2Mat`, the general path.
fn quat_to_mat(q: [f64; 4]) -> [f64; 9] {
    let q00 = q[0] * q[0];
    let q01 = q[0] * q[1];
    let q02 = q[0] * q[2];
    let q03 = q[0] * q[3];
    let q11 = q[1] * q[1];
    let q12 = q[1] * q[2];
    let q13 = q[1] * q[3];
    let q22 = q[2] * q[2];
    let q23 = q[2] * q[3];
    let q33 = q[3] * q[3];
    [
        q00 + q11 - q22 - q33,
        2.0 * (q12 - q03),
        2.0 * (q13 + q02),
        2.0 * (q12 + q03),
        q00 - q11 + q22 - q33,
        2.0 * (q23 - q01),
        2.0 * (q13 - q02),
        2.0 * (q23 + q01),
        q00 - q11 - q22 + q33,
    ]
}

/// `mju_dist3`.
fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// `mju_max`: `a >= b ? a : b`.
fn mju_max(a: f64, b: f64) -> f64 {
    if a >= b { a } else { b }
}

/// `mjMIN`: `a < b ? a : b`.
fn mj_min(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

/// `mjMAX`: `a > b ? a : b`.
fn mj_max(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

/// `updateBox` (engine_setconst.c:1186-1191).
fn update_box(xmin: &mut [f64; 3], xmax: &mut [f64; 3], pos: [f64; 3], radius: f64) {
    for i in 0..3 {
        xmin[i] = mj_min(xmin[i], pos[i] - radius);
        xmax[i] = mj_max(xmax[i], pos[i] + radius);
    }
}

/// The world frames of the model at `qpos0`.
struct Frames {
    xpos: Vec<[f64; 3]>,
    xquat: Vec<[f64; 4]>,
    xmat: Vec<[f64; 9]>,
    xipos: Vec<[f64; 3]>,
    xanchor: Vec<[f64; 3]>,
}

impl Frames {
    /// `mj_local2Global` for a position only.
    fn local_to_global(&self, body: usize, pos: [f64; 3], sameframe: SameFrame) -> [f64; 3] {
        match sameframe {
            SameFrame::None | SameFrame::BodyRot | SameFrame::InertiaRot => {
                let p = mul_mat_vec3(&self.xmat[body], pos);
                let x = self.xpos[body];
                [p[0] + x[0], p[1] + x[1], p[2] + x[2]]
            }
            SameFrame::Body => self.xpos[body],
            SameFrame::Inertia => self.xipos[body],
        }
    }
}

/// `mj_kinematics` at `qpos0`. `bodies[0]` is the world; `joints` are listed by body.
fn kinematics(bodies: &[StatBody], joints: &[StatJoint]) -> Frames {
    let nbody = bodies.len();
    let identity9 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut f = Frames {
        xpos: vec![[0.0; 3]; nbody],
        xquat: vec![[1.0, 0.0, 0.0, 0.0]; nbody],
        xmat: vec![identity9; nbody],
        xipos: vec![[0.0; 3]; nbody],
        xanchor: vec![[0.0; 3]; joints.len()],
    };
    let mut first = 0usize;
    for (i, b) in bodies.iter().enumerate().skip(1) {
        // this body's joints: a contiguous run of the list
        while first < joints.len() && joints[first].body < i {
            first += 1;
        }
        let mut end = first;
        while end < joints.len() && joints[end].body == i {
            end += 1;
        }
        let mut xpos;
        let mut xquat;
        if end - first == 1 && joints[first].kind == StatJointKind::Free {
            // a free joint's qpos0 is its body's pose
            xpos = b.pos;
            xquat = b.quat;
            normalize4(&mut xquat);
            f.xanchor[first] = xpos;
        } else {
            let pid = b.parent;
            if pid != 0 {
                let p = mul_mat_vec3(&f.xmat[pid], b.pos);
                let x = f.xpos[pid];
                xpos = [p[0] + x[0], p[1] + x[1], p[2] + x[2]];
                xquat = mul_quat(f.xquat[pid], b.quat);
            } else {
                xpos = b.pos;
                xquat = b.quat;
            }
            for (jid, j) in joints.iter().enumerate().take(end).skip(first) {
                let r = rot_vec_quat(j.pos, xquat);
                let xanchor = [r[0] + xpos[0], r[1] + xpos[1], r[2] + xpos[2]];
                match j.kind {
                    StatJointKind::Slide { axis } => {
                        let xaxis = rot_vec_quat(axis, xquat);
                        // qpos - qpos0 is 0 at the reference pose
                        let s = 0.0;
                        xpos[0] += xaxis[0] * s;
                        xpos[1] += xaxis[1] * s;
                        xpos[2] += xaxis[2] * s;
                    }
                    StatJointKind::Ball | StatJointKind::Hinge => {
                        // the local rotation at qpos0: a ball joint's normalised identity,
                        // a hinge's mji_axisAngle2Quat(axis, 0), both the identity
                        let qloc = [1.0, 0.0, 0.0, 0.0];
                        xquat = mul_quat(xquat, qloc);
                        let v = rot_vec_quat(j.pos, xquat);
                        xpos = [xanchor[0] - v[0], xanchor[1] - v[1], xanchor[2] - v[2]];
                    }
                    // a free joint is alone on its body (validated), handled above
                    StatJointKind::Free => {}
                }
                f.xanchor[jid] = xanchor;
            }
        }
        normalize4(&mut xquat);
        f.xquat[i] = xquat;
        f.xpos[i] = xpos;
        f.xmat[i] = quat_to_mat(xquat);
        first = end;
    }
    // centres of mass (mj_local2Global with body_sameframe; only the positions are
    // needed, so the inertial orientations are not formed)
    for (i, b) in bodies.iter().enumerate().skip(1) {
        f.xipos[i] = f.local_to_global(i, b.ipos, body_sameframe(b.ipos, b.iquat));
    }
    f
}

/// `setStat`, on the frames at `qpos0`. `bodies[0]` is the world; `joints`, `geoms`
/// and `sites` are listed by body, as MuJoCo numbers them.
pub(crate) fn statistic(
    bodies: &[StatBody],
    joints: &[StatJoint],
    geoms: &[StatGeom],
    sites: &[StatSite],
) -> Statistic {
    let nbody = bodies.len();
    let f = kinematics(bodies, joints);
    let place = |body: usize, pos: [f64; 3], quat: [f64; 4]| {
        let b = &bodies[body];
        f.local_to_global(body, pos, frame_sameframe(pos, quat, b.ipos, b.iquat))
    };
    let geom_xpos: Vec<[f64; 3]> = geoms.iter().map(|g| place(g.body, g.pos, g.quat)).collect();

    // the bounding box of bodies, joint centres, sites and geoms
    let mut xmin = [1e10; 3];
    let mut xmax = [-1e10; 3];
    for i in 1..nbody {
        update_box(&mut xmin, &mut xmax, f.xpos[i], 0.0);
        update_box(&mut xmin, &mut xmax, f.xipos[i], 0.0);
    }
    for a in &f.xanchor {
        update_box(&mut xmin, &mut xmax, *a, 0.0);
    }
    for s in sites {
        update_box(&mut xmin, &mut xmax, place(s.body, s.pos, s.quat), 0.0);
    }
    for (g, xpos) in geoms.iter().zip(&geom_xpos) {
        let mut rbound = 0.0;
        if g.rbound > 0.0 {
            rbound = g.rbound;
        } else if let Some(size) = g.plane {
            if size[0] != 0.0 || size[1] != 0.0 {
                // finite in at least one direction
                rbound = mj_max(size[0], size[1]) * 0.1;
            } else {
                // infinite in both directions
                rbound = 0.01;
            }
        }
        update_box(&mut xmin, &mut xmax, *xpos, rbound);
    }

    let center = [
        (xmin[0] + xmax[0]) * 0.5,
        (xmin[1] + xmax[1]) * 0.5,
        (xmin[2] + xmax[2]) * 0.5,
    ];
    // mj_defaultStatistic (engine_init.c:112-118) when the box is empty
    let mut extent = 2.0;
    if xmax[0] > xmin[0] {
        extent = mju_max(
            1e-5,
            mju_max(
                xmax[0] - xmin[0],
                mju_max(xmax[1] - xmin[1], xmax[2] - xmin[2]),
            ),
        );
    }

    // body size: the largest centre-of-mass to joint-anchor distance, of the body's
    // own joints and its children's
    let mut size = vec![0.0f64; nbody];
    for (j, a) in joints.iter().zip(&f.xanchor) {
        let id = j.body;
        size[id] = mju_max(size[id], dist3(f.xipos[id], *a));
        let id = bodies[id].parent;
        size[id] = mju_max(size[id], dist3(f.xipos[id], *a));
    }
    size[0] = 0.0;
    // then the geoms' bounding spheres about the centre of mass
    for (g, xpos) in geoms.iter().zip(&geom_xpos) {
        let i = g.body;
        if i == 0 {
            continue;
        }
        if g.rbound > 0.0 {
            size[i] = mju_max(size[i], g.rbound + dist3(f.xipos[i], *xpos));
        }
    }
    let mut meansize = 0.2;
    if nbody > 1 {
        meansize = 0.0;
        for s in size.iter_mut().skip(1) {
            *s = mju_max(*s, 1e-5);
            meansize += *s / (nbody - 1) as f64;
        }
    }
    extent = mju_max(extent, 2.0 * meansize);

    Statistic {
        center,
        extent,
        meansize,
    }
}
