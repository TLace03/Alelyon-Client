//! The simulator's camera model, v0: a pinhole.
//!
//! The camera frame is OpenCV's: x right, y down, z forward. Pixel centres are
//! at integer coordinates, so the centre of the pixel in `column`, `row` is
//! `(u, v) = (column, row)` and a centred principal point is
//! `((width - 1) / 2, (height - 1) / 2)`. A pixel's ray in the camera frame is
//! `((u - cx) / fx, (v - cy) / fy, 1)`, so the ray parameter of a hit is its
//! depth: the distance along the camera's viewing axis, which is what contract
//! v0's depth channel holds. Row 0 is the top of the image.
//!
//! The contract leaves the camera's own axis convention to the simulator
//! (sim-contract, `action.rs`); this is that convention, shared with the
//! contract and the physics.
//!
//! Sensor effects (exposure control, noise, motion blur, rolling shutter) are
//! not in v0: the design note lists them for fidelity mode and as an optional
//! training-mode stage.

use crate::layout::{CAMERA_BYTES, Words};
use crate::math::{Pose, Vec3, cross, normalize, quat_from_columns, sub};

/// Pinhole intrinsics and clip distances.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Intrinsics {
    /// Focal length in pixels, x.
    pub fx: f32,
    /// Focal length in pixels, y.
    pub fy: f32,
    /// Principal point, pixels.
    pub cx: f32,
    /// Principal point, pixels.
    pub cy: f32,
    /// Nearest depth rendered, metres.
    pub near: f32,
    /// Farthest depth rendered, metres; at most 65504 (binary16's largest
    /// finite value), so every rendered depth is finite.
    pub far: f32,
}

impl Intrinsics {
    /// Square pixels, the principal point at the centre, and the given
    /// horizontal field of view in radians, measured between the outer edges
    /// of the first and last columns (at `-0.5` and `width - 0.5`).
    pub fn from_hfov(width: u32, height: u32, hfov: f32, near: f32, far: f32) -> Self {
        let fx = 0.5 * width as f32 / (0.5 * hfov).tan();
        Self {
            fx,
            fy: fx,
            cx: 0.5 * (width as f32 - 1.0),
            cy: 0.5 * (height as f32 - 1.0),
            near,
            far,
        }
    }

    /// Whether the values are ones the kernels can use.
    pub fn is_valid(&self) -> bool {
        [self.fx, self.fy, self.cx, self.cy, self.near, self.far]
            .iter()
            .all(|x| x.is_finite())
            && self.fx > 0.0
            && self.fy > 0.0
            && self.near > 0.0
            && self.far > self.near
            && self.far <= 65504.0
    }
}

/// The camera-to-world pose of a camera at `eye` looking at `target`, with
/// `up` appearing up in the image. `None` when `up` is parallel to the view.
pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Option<Pose> {
    let f = normalize(sub(target, eye));
    let r = cross(f, up);
    if crate::math::length(r) < 1e-6 {
        return None;
    }
    let r = normalize(r);
    // OpenCV: x right, y down, z forward, and right x down = forward
    let d = cross(f, r);
    Some(Pose::new(eye, quat_from_columns(r, d, f)))
}

/// The 64-byte camera record of `kernels/common.glsl`.
pub fn camera_bytes(pose: &Pose, k: &Intrinsics) -> [u8; CAMERA_BYTES] {
    let mut w = Words::default();
    w.v3_f(pose.position, 0.0)
        .quat(pose.rotation)
        .f(k.fx)
        .f(k.fy)
        .f(k.cx)
        .f(k.cy)
        .f(k.near)
        .f(k.far)
        .f(0.0)
        .f(0.0);
    w.0.try_into().expect("a camera is 64 bytes")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{dot, quat_rotate};

    #[test]
    fn look_at_points_z_at_the_target_and_y_down() {
        let pose = look_at([2.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]).unwrap();
        let z = quat_rotate(pose.rotation, [0.0, 0.0, 1.0]);
        let y = quat_rotate(pose.rotation, [0.0, 1.0, 0.0]);
        let x = quat_rotate(pose.rotation, [1.0, 0.0, 0.0]);
        assert!(dot(z, [-1.0, 0.0, 0.0]) > 0.9999);
        assert!(dot(y, [0.0, 0.0, -1.0]) > 0.9999);
        // looking along -x (west) with z up, the image's right is +y (north)
        assert!(dot(x, [0.0, 1.0, 0.0]) > 0.9999);
    }

    #[test]
    fn a_parallel_up_is_refused() {
        assert!(look_at([0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).is_none());
    }

    #[test]
    fn hfov_intrinsics_put_the_edge_at_half_the_field() {
        let k = Intrinsics::from_hfov(448, 448, 60f32.to_radians(), 0.05, 50.0);
        assert!(k.is_valid());
        // the last column's outer edge is at 447.5
        let edge = (447.5 - k.cx) / k.fx;
        assert!((edge - 30f32.to_radians().tan()).abs() < 1e-6);
        assert_eq!(k.cx, 223.5);
    }
}
