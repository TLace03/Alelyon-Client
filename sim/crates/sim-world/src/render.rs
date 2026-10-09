//! The render view: what the renderer reads of the world state.
//!
//! AGREED INTERFACE with Lane R (the renderer). Do not change it without Lane R.
//!
//! ```text
//! RenderView.poses : [env][body], 32 bytes each, little-endian
//!   offset  0: position x  f32   metres, world frame
//!   offset  4: position y  f32
//!   offset  8: position z  f32
//!   offset 12: pad         f32   always 0.0 (bits 0)
//!   offset 16: quat x      f32   unit quaternion [x, y, z, w], world-from-body
//!   offset 20: quat y      f32
//!   offset 24: quat z      f32
//!   offset 28: quat w      f32
//!
//! CamerasView.records : [env][camera], 64 bytes each, little-endian
//!   offset  0..32: the world-from-camera pose, laid out exactly like a body pose
//!   offset 32: fx f32   offset 36: fy f32   offset 40: cx f32
//!   offset 44: cy f32   offset 48: near f32   offset 52: far f32
//!   offset 56: pad f32  offset 60: pad f32    both always 0.0 (bits 0)
//! ```
//!
//! Invariants:
//! - `poses` is packed: environment `e`'s body `b` is at byte
//!   `(e * n_bodies + b) * 32`. Body `b` is `scene.bodies[b]`; the world has no
//!   entry. There is no padding between environments (the 256-byte alignment is
//!   a property of [`crate::WorldLayout`]'s state buffers, not of this view).
//! - [`gather_render_view`] is the CPU reference: a pure copy of the `f32` bits of
//!   `BodyPos` and `BodyQuat`, so a GPU gather kernel must match it bit for bit.
//! - A camera's pose is the world pose of its mount: a world mount is the scene's
//!   pose narrowed to `f32`; a body mount is the body's world pose composed with
//!   the camera's local pose. The composition is defined in `f32`, in this
//!   operation order, with no fused multiply-add: rotate the local position `v` by
//!   the body quaternion `q`: `t = 2 * (q.xyz x v)`, `v' = v + q.w * t + (q.xyz x t)`;
//!   add the body position; the orientation is the Hamilton product
//!   `q_body * q_local`, divided by its length `sqrt(x*x + y*y + z*z + w*w)`
//!   (summed in that order). A GPU kernel that fuses multiply-adds may differ in
//!   the last bit of the composed pose; that is not claimed to match.
//! - Camera convention: the OpenCV camera frame (x right, y down, z forward);
//!   intrinsics in pixels with pixel centres at integer coordinates, so a centred
//!   principal point is `((W - 1) / 2, (H - 1) / 2)`. The record carries the
//!   intrinsics as the scene states them; it does not carry the image size,
//!   which stays in `Scene::cameras`.

use sim_scene::{CameraMount, Scene};

use crate::error::WorldError;
use crate::host::HostWorld;

/// Bytes of one body pose in the render view.
pub const POSE_BYTES: usize = 32;
/// Bytes of one camera record in the cameras view: 56 of data padded to 64,
/// the stride of Lane R's std430 struct of four `vec4`.
pub const CAMERA_RECORD_BYTES: usize = 64;

/// The body poses of every environment, in the agreed layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderView {
    /// `[env][body]`, 32 bytes each; see the module note.
    pub poses: Vec<u8>,
}

impl RenderView {
    /// The 32 bytes of pose number `index` (`env * n_bodies + body`), if present.
    pub fn pose_bytes(&self, index: usize) -> Option<&[u8]> {
        self.poses.get(index * POSE_BYTES..(index + 1) * POSE_BYTES)
    }
}

/// The camera records of every environment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CamerasView {
    /// `[env][camera]`, 64 bytes each; see the module note.
    pub records: Vec<u8>,
}

fn put_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_pose(out: &mut Vec<u8>, pos: [f32; 3], quat: [f32; 4]) {
    for v in pos {
        put_f32(out, v);
    }
    put_f32(out, 0.0);
    for v in quat {
        put_f32(out, v);
    }
}

/// The CPU reference for the render view: gathers every body's world pose of
/// every environment into the agreed layout.
pub fn gather_render_view(world: &HostWorld) -> RenderView {
    let layout = world.layout();
    let n_bodies = layout.n_bodies as usize;
    let mut poses = Vec::with_capacity(layout.n_envs as usize * n_bodies * POSE_BYTES);
    for env in 0..layout.n_envs {
        for body in 0..n_bodies {
            let (pos, quat) = world.body_pose(env, body);
            put_pose(&mut poses, pos, quat);
        }
    }
    RenderView { poses }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let qv = [q[0], q[1], q[2]];
    let c = cross(qv, v);
    let t = [2.0 * c[0], 2.0 * c[1], 2.0 * c[2]];
    let ct = cross(qv, t);
    [
        v[0] + q[3] * t[0] + ct[0],
        v[1] + q[3] * t[1] + ct[1],
        v[2] + q[3] * t[2] + ct[2],
    ]
}

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

fn normalize(q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n > 0.0 && n.is_finite() {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

/// A unit quaternion given in `f64`, narrowed to `f32` (normalised first).
fn narrow_quat(q: [f64; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let u = if n > 0.0 {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };
    [u[0] as f32, u[1] as f32, u[2] as f32, u[3] as f32]
}

fn narrow_vec(v: [f64; 3]) -> [f32; 3] {
    [v[0] as f32, v[1] as f32, v[2] as f32]
}

/// The world-from-camera poses and intrinsics of every camera of every
/// environment, in the agreed layout (module note).
///
/// Refuses a world that was not built from `scene` (different number of
/// bodies) and a camera that mounts on a body the scene does not have.
pub fn cameras_view(scene: &Scene, world: &HostWorld) -> Result<CamerasView, WorldError> {
    let layout = world.layout();
    if layout.n_bodies as usize != scene.bodies.len() {
        return Err(WorldError::SceneMismatch {
            reason: "the world has a different number of bodies than the scene",
        });
    }
    for camera in &scene.cameras {
        if let CameraMount::Body { body, .. } = camera.mount
            && body.index() >= scene.bodies.len()
        {
            return Err(WorldError::SceneMismatch {
                reason: "a camera is mounted on a body the scene does not have",
            });
        }
    }
    let mut records =
        Vec::with_capacity(layout.n_envs as usize * scene.cameras.len() * CAMERA_RECORD_BYTES);
    for env in 0..layout.n_envs {
        for camera in &scene.cameras {
            let (pos, quat) = match camera.mount {
                CameraMount::World { pos, quat } => (narrow_vec(pos), narrow_quat(quat)),
                CameraMount::Body {
                    body,
                    local_pos,
                    local_quat,
                } => {
                    let (bp, bq) = world.body_pose(env, body.index());
                    let r = rotate(bq, narrow_vec(local_pos));
                    (
                        [bp[0] + r[0], bp[1] + r[1], bp[2] + r[2]],
                        normalize(quat_mul(bq, narrow_quat(local_quat))),
                    )
                }
            };
            put_pose(&mut records, pos, quat);
            for v in [
                camera.fx,
                camera.fy,
                camera.cx,
                camera.cy,
                camera.near,
                camera.far,
            ] {
                put_f32(&mut records, v);
            }
            put_f32(&mut records, 0.0);
            put_f32(&mut records, 0.0);
        }
    }
    Ok(CamerasView { records })
}
