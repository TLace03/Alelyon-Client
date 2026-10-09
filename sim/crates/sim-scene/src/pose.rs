//! Poses and quaternions in the scene's convention.
//!
//! Invariants:
//! - A quaternion is `[x, y, z, w]` (scalar last), Hamilton, the convention of
//!   `sim-contract` v0. A pose is a position in metres and a unit quaternion
//!   that rotates the child frame into the parent frame.
//! - `compose` is `parent.compose(&child)`: the child's pose given in the parent
//!   frame becomes a pose in the parent's own parent frame. The product
//!   quaternion is renormalised, so a long chain of compositions does not drift
//!   off the unit sphere.
//! - Everything is `f64` and deterministic: no fused multiply-add is requested,
//!   and the operation order is fixed by the source.
//! - A zero quaternion has no direction; `quat_normalize` returns the identity
//!   for it instead of dividing by zero.

/// A 3-vector.
pub type Vec3 = [f64; 3];
/// A quaternion `[x, y, z, w]`.
pub type Quat = [f64; 4];

/// The identity rotation.
pub const IDENTITY_QUAT: Quat = [0.0, 0.0, 0.0, 1.0];

/// Hamilton product `a * b`: the rotation `b` followed by the rotation `a`
/// (so `quat_rotate(quat_mul(a, b), v) == quat_rotate(a, quat_rotate(b, v))`).
pub fn quat_mul(a: Quat, b: Quat) -> Quat {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

/// The conjugate (the inverse for a unit quaternion).
pub fn quat_conj(q: Quat) -> Quat {
    [-q[0], -q[1], -q[2], q[3]]
}

/// The Euclidean length of the four components.
pub fn quat_norm(q: Quat) -> f64 {
    (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt()
}

/// `q` scaled to unit length; the identity for a zero or non-finite input.
pub fn quat_normalize(q: Quat) -> Quat {
    let n = quat_norm(q);
    if n > 0.0 && n.is_finite() {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    } else {
        IDENTITY_QUAT
    }
}

/// Rotates `v` by the unit quaternion `q`.
pub fn quat_rotate(q: Quat, v: Vec3) -> Vec3 {
    let [qx, qy, qz, qw] = q;
    // t = 2 * cross(q.xyz, v); v' = v + w * t + cross(q.xyz, t)
    let tx = 2.0 * (qy * v[2] - qz * v[1]);
    let ty = 2.0 * (qz * v[0] - qx * v[2]);
    let tz = 2.0 * (qx * v[1] - qy * v[0]);
    [
        v[0] + qw * tx + (qy * tz - qz * ty),
        v[1] + qw * ty + (qz * tx - qx * tz),
        v[2] + qw * tz + (qx * ty - qy * tx),
    ]
}

/// The unit quaternion for a rotation of `angle` radians about `axis` (which
/// need not be unit length; a zero axis gives the identity).
pub fn quat_from_axis_angle(axis: Vec3, angle: f64) -> Quat {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if n == 0.0 || !n.is_finite() {
        return IDENTITY_QUAT;
    }
    let half = 0.5 * angle;
    let s = half.sin() / n;
    [axis[0] * s, axis[1] * s, axis[2] * s, half.cos()]
}

/// A rigid transform: where a frame is, and how it is rotated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    /// Position of the frame's origin in the parent frame, metres.
    pub pos: Vec3,
    /// Orientation of the frame in the parent frame, unit `[x, y, z, w]`.
    pub quat: Quat,
}

impl Pose {
    /// The identity transform.
    pub const IDENTITY: Pose = Pose {
        pos: [0.0; 3],
        quat: IDENTITY_QUAT,
    };

    /// A pose from a position and a quaternion (renormalised).
    pub fn new(pos: Vec3, quat: Quat) -> Pose {
        Pose {
            pos,
            quat: quat_normalize(quat),
        }
    }

    /// `self` followed by `child`, `child` being given in `self`'s frame.
    pub fn compose(&self, child: &Pose) -> Pose {
        let rotated = quat_rotate(self.quat, child.pos);
        Pose {
            pos: [
                self.pos[0] + rotated[0],
                self.pos[1] + rotated[1],
                self.pos[2] + rotated[2],
            ],
            quat: quat_normalize(quat_mul(self.quat, child.quat)),
        }
    }

    /// The transform of a point given in this frame, into the parent frame.
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        let r = quat_rotate(self.quat, p);
        [self.pos[0] + r[0], self.pos[1] + r[1], self.pos[2] + r[2]]
    }

    /// The inverse transform.
    pub fn inverse(&self) -> Pose {
        let inv = quat_conj(self.quat);
        let r = quat_rotate(inv, self.pos);
        Pose {
            pos: [-r[0], -r[1], -r[2]],
            quat: inv,
        }
    }
}
