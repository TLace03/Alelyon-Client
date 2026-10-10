//! Small host-side vector, quaternion and pose arithmetic in `f32`.
//!
//! The conventions are the contract's and the kernels' (`kernels/common.glsl`):
//! metres, a right-handed world frame with +Z up, and quaternions `[x, y, z, w]`
//! (Hamilton) that rotate body coordinates into world coordinates. The
//! operations mirror the kernels' expressions term for term, so the host
//! reference ([`crate::reference`]) computes what the device computes up to
//! floating-point contraction and the precision of transcendental functions.

/// A 3-vector.
pub type Vec3 = [f32; 3];
/// A quaternion `[x, y, z, w]`.
pub type Quat = [f32; 4];

/// `a + b`.
pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// `a - b`.
pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// `a * s`.
pub fn scale(a: Vec3, s: f32) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Component-wise product.
pub fn mul(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

/// The dot product.
pub fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The cross product `a x b`.
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Euclidean length.
pub fn length(a: Vec3) -> f32 {
    dot(a, a).sqrt()
}

/// `a / |a|`. The zero vector stays zero.
pub fn normalize(a: Vec3) -> Vec3 {
    let l = length(a);
    if l > 0.0 { scale(a, 1.0 / l) } else { a }
}

/// Component-wise minimum.
pub fn min3(a: Vec3, b: Vec3) -> Vec3 {
    [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])]
}

/// Component-wise maximum.
pub fn max3(a: Vec3, b: Vec3) -> Vec3 {
    [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])]
}

/// The identity rotation.
pub const QUAT_IDENTITY: Quat = [0.0, 0.0, 0.0, 1.0];

/// The rotation by `angle` radians about the unit `axis`.
pub fn quat_axis_angle(axis: Vec3, angle: f32) -> Quat {
    let a = normalize(axis);
    let (s, c) = (0.5 * angle).sin_cos();
    [a[0] * s, a[1] * s, a[2] * s, c]
}

/// The Hamilton product `a * b`: rotate by `b`, then by `a`.
pub fn quat_mul(a: Quat, b: Quat) -> Quat {
    let av = [a[0], a[1], a[2]];
    let bv = [b[0], b[1], b[2]];
    let v = add(add(scale(bv, a[3]), scale(av, b[3])), cross(av, bv));
    [v[0], v[1], v[2], a[3] * b[3] - dot(av, bv)]
}

/// `q / |q|`.
pub fn quat_normalize(q: Quat) -> Quat {
    let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / l, q[1] / l, q[2] / l, q[3] / l]
}

/// Rotate `v` by the unit quaternion `q`, as the kernels do:
/// `v + w t + u x t` with `t = 2 u x v`.
pub fn quat_rotate(q: Quat, v: Vec3) -> Vec3 {
    let u = [q[0], q[1], q[2]];
    let t = scale(cross(u, v), 2.0);
    add(add(v, scale(t, q[3])), cross(u, t))
}

/// The rotation matrix of a unit quaternion, as three rows.
pub fn quat_rows(q: Quat) -> [Vec3; 3] {
    let [x, y, z, w] = q;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// The unit quaternion of a rotation matrix given by its three COLUMNS (the
/// images of the x, y and z axes). Shepperd's method: the largest of the four
/// candidate divisors is used, so it is stable for every rotation.
pub fn quat_from_columns(cx: Vec3, cy: Vec3, cz: Vec3) -> Quat {
    let (m00, m01, m02) = (cx[0], cy[0], cz[0]);
    let (m10, m11, m12) = (cx[1], cy[1], cz[1]);
    let (m20, m21, m22) = (cx[2], cy[2], cz[2]);
    let trace = m00 + m11 + m22;
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s]
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s]
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s]
    };
    quat_normalize(q)
}

/// A rigid pose: where a frame's origin is and how it is rotated, in its
/// parent's frame (the world, for a body).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    /// Position, metres.
    pub position: Vec3,
    /// Rotation `[x, y, z, w]`, child to parent.
    pub rotation: Quat,
}

impl Pose {
    /// The identity pose.
    pub const IDENTITY: Pose = Pose {
        position: [0.0; 3],
        rotation: QUAT_IDENTITY,
    };

    /// A pose from a position and a rotation.
    pub fn new(position: Vec3, rotation: Quat) -> Self {
        Self { position, rotation }
    }

    /// A translation with no rotation.
    pub fn at(position: Vec3) -> Self {
        Self::new(position, QUAT_IDENTITY)
    }

    /// `self * child`: the child's pose in this pose's parent frame.
    pub fn compose(&self, child: &Pose) -> Pose {
        Pose {
            position: add(quat_rotate(self.rotation, child.position), self.position),
            rotation: quat_mul(self.rotation, child.rotation),
        }
    }

    /// The point `p`, given in this frame, in the parent frame.
    pub fn apply(&self, p: Vec3) -> Vec3 {
        add(quat_rotate(self.rotation, p), self.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vec3, b: Vec3, tol: f32) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() <= tol)
    }

    #[test]
    fn rotate_agrees_with_the_matrix_rows() {
        let q = quat_normalize([0.3, -0.5, 0.2, 0.8]);
        let rows = quat_rows(q);
        let v = [0.7, -1.1, 2.3];
        let by_rows = [dot(rows[0], v), dot(rows[1], v), dot(rows[2], v)];
        assert!(close(quat_rotate(q, v), by_rows, 1e-5));
    }

    #[test]
    fn a_quarter_turn_about_z_takes_x_to_y() {
        let q = quat_axis_angle([0.0, 0.0, 1.0], std::f32::consts::FRAC_PI_2);
        assert!(close(
            quat_rotate(q, [1.0, 0.0, 0.0]),
            [0.0, 1.0, 0.0],
            1e-6
        ));
    }

    #[test]
    fn the_product_rotates_by_the_right_operand_first() {
        let a = quat_axis_angle([0.0, 0.0, 1.0], 0.9);
        let b = quat_axis_angle([1.0, 0.0, 0.0], -0.4);
        let v = [0.2, 0.5, -0.3];
        let composed = quat_rotate(quat_mul(a, b), v);
        let stepwise = quat_rotate(a, quat_rotate(b, v));
        assert!(close(composed, stepwise, 1e-6));
    }

    #[test]
    fn the_quaternion_of_a_matrix_rebuilds_it() {
        for q in [
            quat_normalize([0.3, -0.5, 0.2, 0.8]),
            quat_normalize([0.9, 0.1, 0.0, 0.05]),
            quat_normalize([0.0, 0.99, 0.1, -0.01]),
            quat_normalize([0.1, 0.0, 0.995, 0.0]),
        ] {
            let r = quat_rows(q);
            let cols = [
                [r[0][0], r[1][0], r[2][0]],
                [r[0][1], r[1][1], r[2][1]],
                [r[0][2], r[1][2], r[2][2]],
            ];
            let back = quat_from_columns(cols[0], cols[1], cols[2]);
            let v = [0.3, -0.8, 0.5];
            assert!(close(quat_rotate(back, v), quat_rotate(q, v), 1e-5));
        }
    }

    #[test]
    fn compose_applies_the_child_then_the_parent() {
        let parent = Pose::new([1.0, 2.0, 3.0], quat_axis_angle([0.0, 0.0, 1.0], 0.5));
        let child = Pose::new([0.5, 0.0, 0.0], quat_axis_angle([0.0, 1.0, 0.0], 0.25));
        let p = [0.1, 0.2, 0.3];
        assert!(close(
            parent.compose(&child).apply(p),
            parent.apply(child.apply(p)),
            1e-5
        ));
    }
}
