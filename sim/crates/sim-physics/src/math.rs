//! Vector, quaternion and spatial-algebra helpers, ported from MuJoCo.
//!
//! Ports `engine_inline.h` (the `mji_` functions), `engine_util_spatial.c` and
//! `engine_util_blas.c` (`mju_normalize3`, `mju_normalize4`), 3.14.0. Every
//! function writes its arithmetic in the order of the C source and uses plain
//! multiply, add and subtract only (no fused multiply-add, see [`crate::Real`]).
//!
//! Conventions (MuJoCo's, kept so the port is mechanical):
//! - A quaternion is `[w, x, y, z]`. The scene and `HostWorld` use `[x, y, z, w]`;
//!   the conversion is done in one place, [`crate::convert_qpos`].
//! - A 6-vector is `[angular(3), linear(3)]`. A motion vector is a velocity
//!   (`cvel`, `cdof`); a force vector is `[torque(3), force(3)]`.
//! - A 3x3 matrix is row-major in 9 numbers. A 10-vector inertia is `cinert`:
//!   `[I00, I11, I22, I01, I02, I12, mx, my, mz, m]`, the rotational inertia about
//!   the subtree centre of mass, the mass times the offset, and the mass.
//!
//! Deliberate differences from the C source (each changes no value):
//! - The `mji_rotVecQuat`, `mju_quat2Mat` and `mju_axisAngle2Quat` shortcuts for
//!   the null quaternion and the zero vector are not taken where the general path
//!   gives the same bits (null quaternion in `rot_vec_quat` and `quat_to_mat`: the
//!   general formulas produce `v` and the identity exactly), so the code has no
//!   data-dependent branch there. `axis_angle_to_quat` keeps its `angle == 0`
//!   branch because the general formula would give `sin(0) * axis`, a signed zero
//!   that MuJoCo does not produce.
//! - Results are returned by value in fixed-size arrays; nothing allocates.

use crate::real::Real;

/// MuJoCo's `mjMINVAL`: the minimum value in any denominator.
#[inline]
pub(crate) fn min_val<R: Real>() -> R {
    R::from_f64(1e-15)
}

/// `a` as an `R`, for the literals of the ported code.
#[inline]
pub(crate) fn lit<R: Real>(a: f64) -> R {
    R::from_f64(a)
}

/// Three numbers starting at `3 * i`.
#[inline]
pub(crate) fn v3<R: Real>(a: &[R], i: usize) -> [R; 3] {
    [a[3 * i], a[3 * i + 1], a[3 * i + 2]]
}

/// Four numbers starting at `4 * i`.
#[inline]
pub(crate) fn v4<R: Real>(a: &[R], i: usize) -> [R; 4] {
    [a[4 * i], a[4 * i + 1], a[4 * i + 2], a[4 * i + 3]]
}

/// Six numbers starting at `6 * i`.
#[inline]
pub(crate) fn v6<R: Real>(a: &[R], i: usize) -> [R; 6] {
    [
        a[6 * i],
        a[6 * i + 1],
        a[6 * i + 2],
        a[6 * i + 3],
        a[6 * i + 4],
        a[6 * i + 5],
    ]
}

/// Nine numbers starting at `9 * i`.
#[inline]
pub(crate) fn v9<R: Real>(a: &[R], i: usize) -> [R; 9] {
    let mut out = [R::ZERO; 9];
    out.copy_from_slice(&a[9 * i..9 * i + 9]);
    out
}

/// Ten numbers starting at `10 * i`.
#[inline]
pub(crate) fn v10<R: Real>(a: &[R], i: usize) -> [R; 10] {
    let mut out = [R::ZERO; 10];
    out.copy_from_slice(&a[10 * i..10 * i + 10]);
    out
}

/// `res = a + b`.
#[inline]
pub(crate) fn add3<R: Real>(a: [R; 3], b: [R; 3]) -> [R; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// `res = a - b`.
#[inline]
pub(crate) fn sub3<R: Real>(a: [R; 3], b: [R; 3]) -> [R; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// `res = a * s`.
#[inline]
pub(crate) fn scl3<R: Real>(a: [R; 3], s: R) -> [R; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Euclidean norm (`mju_norm3`).
#[inline]
pub(crate) fn norm3<R: Real>(a: [R; 3]) -> R {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// Dot product of 3-vectors (`mju_dot3`).
#[inline]
pub(crate) fn dot3<R: Real>(a: [R; 3], b: [R; 3]) -> R {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Port of `mju_normalize3` (engine_util_blas.c): normalises in place and
/// returns the length before normalisation; a vector shorter than `mjMINVAL`
/// becomes `(1, 0, 0)`.
#[inline]
pub(crate) fn normalize3<R: Real>(v: &mut [R; 3]) -> R {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm < min_val::<R>() {
        v[0] = R::ONE;
        v[1] = R::ZERO;
        v[2] = R::ZERO;
    } else {
        let inv = R::ONE / norm;
        v[0] *= inv;
        v[1] *= inv;
        v[2] *= inv;
    }
    norm
}

/// Port of `mju_normalize4` (engine_util_blas.c): normalises in place and
/// returns the length before normalisation. A quaternion shorter than `mjMINVAL`
/// becomes the identity, and one whose length is within `mjMINVAL` of 1 is left
/// alone.
#[inline]
pub(crate) fn normalize4<R: Real>(v: &mut [R; 4]) -> R {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
    if norm < min_val::<R>() {
        v[0] = R::ONE;
        v[1] = R::ZERO;
        v[2] = R::ZERO;
        v[3] = R::ZERO;
    } else if (norm - R::ONE).abs() > min_val::<R>() {
        let inv = R::ONE / norm;
        v[0] *= inv;
        v[1] *= inv;
        v[2] *= inv;
        v[3] *= inv;
    }
    norm
}

/// Port of `mji_mulMatVec3`: `mat * vec`.
#[inline]
pub(crate) fn mul_mat_vec3<R: Real>(mat: &[R; 9], vec: [R; 3]) -> [R; 3] {
    [
        mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2],
        mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2],
        mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2],
    ]
}

/// Port of `mji_rotVecQuat` (engine_inline.h): rotates `vec` by the unit
/// quaternion `quat` (see the module note on the null-quaternion shortcut).
#[inline]
pub(crate) fn rot_vec_quat<R: Real>(vec: [R; 3], quat: [R; 4]) -> [R; 3] {
    let two = lit::<R>(2.0);
    // tmp = q_w * v + cross(q_xyz, v)
    let tmp = [
        quat[0] * vec[0] + quat[2] * vec[2] - quat[3] * vec[1],
        quat[0] * vec[1] + quat[3] * vec[0] - quat[1] * vec[2],
        quat[0] * vec[2] + quat[1] * vec[1] - quat[2] * vec[0],
    ];
    // res = v + 2 * cross(q_xyz, tmp)
    [
        vec[0] + two * (quat[2] * tmp[2] - quat[3] * tmp[1]),
        vec[1] + two * (quat[3] * tmp[0] - quat[1] * tmp[2]),
        vec[2] + two * (quat[1] * tmp[1] - quat[2] * tmp[0]),
    ]
}

/// Port of `mji_mulQuat`: the Hamilton product `qa * qb`, `[w, x, y, z]`.
#[inline]
pub(crate) fn mul_quat<R: Real>(qa: [R; 4], qb: [R; 4]) -> [R; 4] {
    [
        qa[0] * qb[0] - qa[1] * qb[1] - qa[2] * qb[2] - qa[3] * qb[3],
        qa[0] * qb[1] + qa[1] * qb[0] + qa[2] * qb[3] - qa[3] * qb[2],
        qa[0] * qb[2] - qa[1] * qb[3] + qa[2] * qb[0] + qa[3] * qb[1],
        qa[0] * qb[3] + qa[1] * qb[2] - qa[2] * qb[1] + qa[3] * qb[0],
    ]
}

/// Port of `mji_negQuat`: the conjugate.
#[inline]
pub(crate) fn neg_quat<R: Real>(q: [R; 4]) -> [R; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

/// Port of `mji_axisAngle2Quat`: the rotation of `angle` about the unit `axis`.
#[inline]
pub(crate) fn axis_angle_to_quat<R: Real>(axis: [R; 3], angle: R) -> [R; 4] {
    if angle == R::ZERO {
        [R::ONE, R::ZERO, R::ZERO, R::ZERO]
    } else {
        let half = lit::<R>(0.5);
        let s = (angle * half).sin();
        [(angle * half).cos(), axis[0] * s, axis[1] * s, axis[2] * s]
    }
}

/// Port of `mju_quat2Mat` (engine_util_spatial.c): the rotation matrix of a unit
/// quaternion, row-major.
#[inline]
pub(crate) fn quat_to_mat<R: Real>(q: [R; 4]) -> [R; 9] {
    let two = lit::<R>(2.0);
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
        two * (q12 - q03),
        two * (q13 + q02),
        two * (q12 + q03),
        q00 - q11 + q22 - q33,
        two * (q23 - q01),
        two * (q13 - q02),
        two * (q23 + q01),
        q00 - q11 - q22 + q33,
    ]
}

/// Port of `mji_quat2Vel`: converts a quaternion (an orientation difference) to
/// the 3D velocity that produces it in time `dt`.
#[inline]
pub(crate) fn quat_to_vel<R: Real>(quat: [R; 4], dt: R) -> [R; 3] {
    let mut axis = [quat[1], quat[2], quat[3]];
    let sin_a_2 = normalize3(&mut axis);
    let mut speed = lit::<R>(2.0) * sin_a_2.atan2(quat[0]);
    // when the axis-angle is larger than pi, the rotation is the other way
    let pi = lit::<R>(std::f64::consts::PI);
    if speed > pi {
        speed -= lit::<R>(2.0) * pi;
    }
    speed /= dt;
    scl3(axis, speed)
}

/// Port of `mji_subQuat` / `mju_subQuat`: the 3D velocity `res` with
/// `qb * quat(res) = qa`.
#[inline]
pub(crate) fn sub_quat<R: Real>(qa: [R; 4], qb: [R; 4]) -> [R; 3] {
    let qneg = neg_quat(qb);
    let qdif = mul_quat(qneg, qa);
    quat_to_vel(qdif, R::ONE)
}

/// Port of `mju_quatIntegrate` (engine_util_spatial.c): rotates `quat` by the
/// body-frame angular velocity `vel` for `scale` seconds, `quat <- quat * qrot`.
#[inline]
pub(crate) fn quat_integrate<R: Real>(quat: [R; 4], vel: [R; 3], scale: R) -> [R; 4] {
    let mut tmp = vel;
    let angle = scale * normalize3(&mut tmp);
    let qrot = axis_angle_to_quat(tmp, angle);
    let mut q = quat;
    normalize4(&mut q);
    mul_quat(q, qrot)
}

/// Port of `mji_cross`: `a x b`.
#[inline]
pub(crate) fn cross<R: Real>(a: [R; 3], b: [R; 3]) -> [R; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Port of `mji_crossMotion`: `vel x v` for motion vectors.
#[inline]
pub(crate) fn cross_motion<R: Real>(vel: [R; 6], v: [R; 6]) -> [R; 6] {
    let mut res = [
        -vel[2] * v[1] + vel[1] * v[2],
        vel[2] * v[0] - vel[0] * v[2],
        -vel[1] * v[0] + vel[0] * v[1],
        -vel[2] * v[4] + vel[1] * v[5],
        vel[2] * v[3] - vel[0] * v[5],
        -vel[1] * v[3] + vel[0] * v[4],
    ];
    res[3] += -vel[5] * v[1] + vel[4] * v[2];
    res[4] += vel[5] * v[0] - vel[3] * v[2];
    res[5] += -vel[4] * v[0] + vel[3] * v[1];
    res
}

/// Port of `mji_crossForce`: `vel x* f`, a motion vector acting on a force vector.
#[inline]
pub(crate) fn cross_force<R: Real>(vel: [R; 6], f: [R; 6]) -> [R; 6] {
    let mut res = [
        -vel[2] * f[1] + vel[1] * f[2],
        vel[2] * f[0] - vel[0] * f[2],
        -vel[1] * f[0] + vel[0] * f[1],
        -vel[2] * f[4] + vel[1] * f[5],
        vel[2] * f[3] - vel[0] * f[5],
        -vel[1] * f[3] + vel[0] * f[4],
    ];
    res[0] += -vel[5] * f[4] + vel[4] * f[5];
    res[1] += vel[5] * f[3] - vel[3] * f[5];
    res[2] += -vel[4] * f[3] + vel[3] * f[4];
    res
}

/// Port of `mji_dot6`: the 6D dot product, in the order of `mju_dot`.
#[inline]
pub(crate) fn dot6<R: Real>(a: [R; 6], b: [R; 6]) -> R {
    ((a[0] * b[0] + a[2] * b[2]) + (a[1] * b[1] + a[3] * b[3])) + (a[4] * b[4] + a[5] * b[5])
}

/// Port of `mju_inertCom`: the inertia of a body (principal moments `inert`
/// in the frame `mat`, mass `mass`) expressed about a point `dif` away from its
/// centre of mass, as the 10-vector `cinert`.
#[inline]
pub(crate) fn inert_com<R: Real>(inert: [R; 3], mat: &[R; 9], dif: [R; 3], mass: R) -> [R; 10] {
    // tmp = diag(inert) * mat'  (mat is local-to-global rotation)
    let tmp = [
        mat[0] * inert[0],
        mat[3] * inert[0],
        mat[6] * inert[0],
        mat[1] * inert[1],
        mat[4] * inert[1],
        mat[7] * inert[1],
        mat[2] * inert[2],
        mat[5] * inert[2],
        mat[8] * inert[2],
    ];
    // res_rot = mat * diag(inert) * mat'
    let mut res = [R::ZERO; 10];
    res[0] = mat[0] * tmp[0] + mat[1] * tmp[3] + mat[2] * tmp[6];
    res[1] = mat[3] * tmp[1] + mat[4] * tmp[4] + mat[5] * tmp[7];
    res[2] = mat[6] * tmp[2] + mat[7] * tmp[5] + mat[8] * tmp[8];
    res[3] = mat[0] * tmp[1] + mat[1] * tmp[4] + mat[2] * tmp[7];
    res[4] = mat[0] * tmp[2] + mat[1] * tmp[5] + mat[2] * tmp[8];
    res[5] = mat[3] * tmp[2] + mat[4] * tmp[5] + mat[5] * tmp[8];

    // res_rot -= mass * dif_cross * dif_cross
    res[0] += mass * (dif[1] * dif[1] + dif[2] * dif[2]);
    res[1] += mass * (dif[0] * dif[0] + dif[2] * dif[2]);
    res[2] += mass * (dif[0] * dif[0] + dif[1] * dif[1]);
    res[3] -= mass * dif[0] * dif[1];
    res[4] -= mass * dif[0] * dif[2];
    res[5] -= mass * dif[1] * dif[2];

    // res_tran = mass * dif
    res[6] = mass * dif[0];
    res[7] = mass * dif[1];
    res[8] = mass * dif[2];

    // res_mass = mass
    res[9] = mass;
    res
}

/// Port of `mju_mulInertVec`: the 10-vector inertia `i` times the 6D motion
/// vector `v`, giving a 6D force vector.
#[inline]
pub(crate) fn mul_inert_vec<R: Real>(i: &[R; 10], v: [R; 6]) -> [R; 6] {
    [
        i[0] * v[0] + i[3] * v[1] + i[4] * v[2] - i[8] * v[4] + i[7] * v[5],
        i[3] * v[0] + i[1] * v[1] + i[5] * v[2] + i[8] * v[3] - i[6] * v[5],
        i[4] * v[0] + i[5] * v[1] + i[2] * v[2] - i[7] * v[3] + i[6] * v[4],
        i[8] * v[1] - i[7] * v[2] + i[9] * v[3],
        i[6] * v[2] - i[8] * v[0] + i[9] * v[4],
        i[7] * v[0] - i[6] * v[1] + i[9] * v[5],
    ]
}

/// Port of `mju_dofCom` with an offset (a hinge): the motion axis of a rotation
/// about `axis` through a point `offset` away from the reference point.
#[inline]
pub(crate) fn dof_com_hinge<R: Real>(axis: [R; 3], offset: [R; 3]) -> [R; 6] {
    let c = cross(axis, offset);
    [axis[0], axis[1], axis[2], c[0], c[1], c[2]]
}

/// Port of `mju_dofCom` without an offset (a slide): a pure translation along `axis`.
#[inline]
pub(crate) fn dof_com_slide<R: Real>(axis: [R; 3]) -> [R; 6] {
    [R::ZERO, R::ZERO, R::ZERO, axis[0], axis[1], axis[2]]
}

/// Port of `mju_transformSpatial` for a motion vector (`flg_force = 0`) with no
/// rotation (`rotnew2old = NULL`): moves the reference point from `oldpos` to
/// `newpos`, both in the same frame.
#[inline]
pub(crate) fn transform_motion<R: Real>(vec: [R; 6], newpos: [R; 3], oldpos: [R; 3]) -> [R; 6] {
    let dif = sub3(newpos, oldpos);
    let cros = cross(dif, [vec[0], vec[1], vec[2]]);
    [
        vec[0],
        vec[1],
        vec[2],
        vec[3] - cros[0],
        vec[4] - cros[1],
        vec[5] - cros[2],
    ]
}

/// Port of `mji_mulMatTVec3` / `mju_mulMatTVec3`: `mat' * vec` for a row-major 3-by-3.
#[inline]
pub(crate) fn mul_mat_t_vec3<R: Real>(mat: &[R; 9], vec: [R; 3]) -> [R; 3] {
    [
        mat[0] * vec[0] + mat[3] * vec[1] + mat[6] * vec[2],
        mat[1] * vec[0] + mat[4] * vec[1] + mat[7] * vec[2],
        mat[2] * vec[0] + mat[5] * vec[1] + mat[8] * vec[2],
    ]
}

/// Port of `mju_mulMatTMat3`: `mat1' * mat2` for row-major 3-by-3 matrices.
#[inline]
pub(crate) fn mul_mat_t_mat3<R: Real>(mat1: &[R; 9], mat2: &[R; 9]) -> [R; 9] {
    [
        mat1[0] * mat2[0] + mat1[3] * mat2[3] + mat1[6] * mat2[6],
        mat1[0] * mat2[1] + mat1[3] * mat2[4] + mat1[6] * mat2[7],
        mat1[0] * mat2[2] + mat1[3] * mat2[5] + mat1[6] * mat2[8],
        mat1[1] * mat2[0] + mat1[4] * mat2[3] + mat1[7] * mat2[6],
        mat1[1] * mat2[1] + mat1[4] * mat2[4] + mat1[7] * mat2[7],
        mat1[1] * mat2[2] + mat1[4] * mat2[5] + mat1[7] * mat2[8],
        mat1[2] * mat2[0] + mat1[5] * mat2[3] + mat1[8] * mat2[6],
        mat1[2] * mat2[1] + mat1[5] * mat2[4] + mat1[8] * mat2[7],
        mat1[2] * mat2[2] + mat1[5] * mat2[5] + mat1[8] * mat2[8],
    ]
}

/// Port of `mju_clip`, as its ternary: `min` if `x < min`, else `max` if `x > max`, else
/// `x` (a NaN passes through).
#[inline]
pub(crate) fn clip<R: Real>(x: R, min: R, max: R) -> R {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

/// `res += vec * scl` (`mji_addToScl3`).
#[inline]
pub(crate) fn add_to_scl3<R: Real>(res: &mut [R; 3], vec: [R; 3], scl: R) {
    res[0] += vec[0] * scl;
    res[1] += vec[1] * scl;
    res[2] += vec[2] * scl;
}

/// `res = vec1 + scl * vec2` (`mji_addScl3`).
#[inline]
pub(crate) fn add_scl3<R: Real>(vec1: [R; 3], vec2: [R; 3], scl: R) -> [R; 3] {
    [
        vec1[0] + scl * vec2[0],
        vec1[1] + scl * vec2[1],
        vec1[2] + scl * vec2[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quaternion_rotation_matches_the_matrix() {
        let mut q = [0.8f64, 0.1, -0.3, 0.5];
        normalize4(&mut q);
        let m = quat_to_mat(q);
        let v = [0.3, -1.2, 2.5];
        let a = rot_vec_quat(v, q);
        let b = mul_mat_vec3(&m, v);
        for k in 0..3 {
            assert!((a[k] - b[k]).abs() < 1e-14);
        }
    }

    #[test]
    fn null_quaternion_is_exactly_the_identity() {
        let q = [1.0f64, 0.0, 0.0, 0.0];
        assert_eq!(
            quat_to_mat(q),
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
        let v = [0.3, -1.2, 2.5];
        assert_eq!(rot_vec_quat(v, q), v);
    }

    #[test]
    fn sub_quat_recovers_the_rotation_vector() {
        let axis = [0.0, 0.6, 0.8];
        let angle = 0.7;
        let qb = [0.9f64, 0.1, 0.2, 0.3];
        let mut qb = qb;
        normalize4(&mut qb);
        let qa = mul_quat(qb, axis_angle_to_quat(axis, angle));
        let d = sub_quat(qa, qb);
        for k in 0..3 {
            assert!((d[k] - axis[k] * angle).abs() < 1e-14, "{d:?}");
        }
    }

    #[test]
    fn quat_integrate_is_a_body_frame_rotation() {
        let q = [1.0f64, 0.0, 0.0, 0.0];
        let r = quat_integrate(q, [0.0, 0.0, 2.0], 0.25);
        // half the angle in the quaternion: 0.5 rad about z
        assert!((r[0] - 0.25f64.cos()).abs() < 1e-15);
        assert!((r[3] - 0.25f64.sin()).abs() < 1e-15);
        // zero velocity leaves the quaternion unchanged
        assert_eq!(quat_integrate(q, [0.0; 3], 0.25), q);
    }

    #[test]
    fn spatial_cross_products_satisfy_their_identities() {
        let v = [0.1f64, -0.2, 0.3, 0.4, 0.5, -0.6];
        // a motion crossed with itself vanishes
        assert_eq!(cross_motion(v, v), [0.0; 6]);
        // power conservation: v . (w x* f) = -(w x v) . f for motion v, w and force f
        let w = [0.7f64, 0.2, -0.1, 0.05, 0.9, 0.3];
        let f = [0.3f64, 0.8, -0.4, 0.2, -0.5, 0.6];
        let lhs = dot6(v, cross_force(w, f));
        let rhs = -dot6(cross_motion(w, v), f);
        assert!((lhs - rhs).abs() < 1e-14, "{lhs} {rhs}");
    }
}
