//! MuJoCo's compiler arithmetic, ported operation for operation.
//!
//! Source: MuJoCo commit a8373cc4e,
//! `src/user/user_util.cc` (the `mjuu_*` functions, cited per function below)
//! and `src/user/user_objects.cc` (`ResolveOrientation`, lines 273-362).
//! Apache-2.0, (c) DeepMind Technologies Limited.
//!
//! Invariants:
//! - Quaternions here are MuJoCo's `[w, x, y, z]`, not the scene's `[x, y, z, w]`:
//!   a line-for-line port stays checkable against the C++. Conversion happens at
//!   the boundary, in `compile.rs`.
//! - Every function keeps MuJoCo's operation order and its tolerances (`EPS`,
//!   the "do not normalise within `EPS` of 1" rule), because the parity test
//!   compares results to MuJoCo's to 1e-9 and a reassociated sum would still pass
//!   it by luck on one model and fail on another.
//! - Matrices are row-major `[f64; 9]`.

use std::f64::consts::PI;

/// `mjEPS`: minimum value in various calculations (user_util.h:31).
pub(crate) const EPS: f64 = 1e-14;

/// `mjMINVAL`: minimum value in any denominator (mjtype.h:27).
pub(crate) const MINVAL: f64 = 1e-15;

/// Default relative tolerance of `mjuu_eig3` (user_util.h:172).
pub(crate) const EIG_RELTOL: f64 = 4e-15;

/// `mjuu_normvec` (user_util.cc:141): normalises in place and returns the
/// previous length; returns 0 and leaves the vector alone when the squared
/// length is below `EPS`; does not divide when the length is within `EPS` of 1.
pub(crate) fn normvec(v: &mut [f64]) -> f64 {
    let mut nrm = 0.0;
    for x in v.iter() {
        nrm += x * x;
    }
    if nrm < EPS {
        return 0.0;
    }
    nrm = nrm.sqrt();
    if (nrm - 1.0).abs() > EPS {
        for x in v.iter_mut() {
            *x /= nrm;
        }
    }
    nrm
}

/// `mjuu_quat2mat` (user_util.cc:181).
pub(crate) fn quat2mat(q: [f64; 4]) -> [f64; 9] {
    if q[0] == 1.0 && q[1] == 0.0 && q[2] == 0.0 && q[3] == 0.0 {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }
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

/// `mjuu_mulquat` (user_util.cc:222): the product, normalised by `normvec`.
pub(crate) fn mulquat(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let mut tmp = [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ];
    normvec(&mut tmp);
    tmp
}

/// `mjuu_mulvecmat` (user_util.cc:234): `mat * vec`.
pub(crate) fn mulvecmat(vec: [f64; 3], m: &[f64; 9]) -> [f64; 3] {
    [
        m[0] * vec[0] + m[1] * vec[1] + m[2] * vec[2],
        m[3] * vec[0] + m[4] * vec[1] + m[5] * vec[2],
        m[6] * vec[0] + m[7] * vec[1] + m[8] * vec[2],
    ]
}

/// `mjuu_mulmat` (user_util.cc:286): `A * B`.
pub(crate) fn mulmat(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    [
        a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
        a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
        a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
        a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
        a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
        a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
        a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
        a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
        a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
    ]
}

/// `mjuu_mulRMRT` (user_util.cc:258): `R * M * R'`.
pub(crate) fn mul_rmrt(r: &[f64; 9], m: &[f64; 9]) -> [f64; 9] {
    let tmp = [
        r[0] * m[0] + r[1] * m[3] + r[2] * m[6],
        r[0] * m[1] + r[1] * m[4] + r[2] * m[7],
        r[0] * m[2] + r[1] * m[5] + r[2] * m[8],
        r[3] * m[0] + r[4] * m[3] + r[5] * m[6],
        r[3] * m[1] + r[4] * m[4] + r[5] * m[7],
        r[3] * m[2] + r[4] * m[5] + r[5] * m[8],
        r[6] * m[0] + r[7] * m[3] + r[8] * m[6],
        r[6] * m[1] + r[7] * m[4] + r[8] * m[7],
        r[6] * m[2] + r[7] * m[5] + r[8] * m[8],
    ];
    [
        tmp[0] * r[0] + tmp[1] * r[1] + tmp[2] * r[2],
        tmp[0] * r[3] + tmp[1] * r[4] + tmp[2] * r[5],
        tmp[0] * r[6] + tmp[1] * r[7] + tmp[2] * r[8],
        tmp[3] * r[0] + tmp[4] * r[1] + tmp[5] * r[2],
        tmp[3] * r[3] + tmp[4] * r[4] + tmp[5] * r[5],
        tmp[3] * r[6] + tmp[4] * r[7] + tmp[5] * r[8],
        tmp[6] * r[0] + tmp[7] * r[1] + tmp[8] * r[2],
        tmp[6] * r[3] + tmp[7] * r[4] + tmp[8] * r[5],
        tmp[6] * r[6] + tmp[7] * r[7] + tmp[8] * r[8],
    ]
}

/// `mjuu_transposemat` (user_util.cc:304).
pub(crate) fn transposemat(m: &[f64; 9]) -> [f64; 9] {
    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

/// `mjuu_crossvec` (user_util.cc:335): `b x c`.
pub(crate) fn cross(b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    [
        b[1] * c[2] - b[2] * c[1],
        b[2] * c[0] - b[0] * c[2],
        b[0] * c[1] - b[1] * c[0],
    ]
}

/// `mjuu_dot3` (user_util.cc:118).
pub(crate) fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `mjuu_z2quat` (user_util.cc:376): the quaternion rotating +Z onto `vec`.
pub(crate) fn z2quat(vec: [f64; 3]) -> [f64; 4] {
    let z = [0.0, 0.0, 1.0];
    let mut axis = cross(z, vec);
    let s = normvec(&mut axis);
    if s < 1e-10 {
        axis = [1.0, 0.0, 0.0];
    }
    let ang = s.atan2(vec[2]);
    [
        (ang / 2.0).cos(),
        axis[0] * (ang / 2.0).sin(),
        axis[1] * (ang / 2.0).sin(),
        axis[2] * (ang / 2.0).sin(),
    ]
}

/// `mjuu_frame2quat` (user_util.cc:393): the quaternion of the frame whose
/// axes are `x`, `y`, `z` (the matrix columns).
pub(crate) fn frame2quat(x: [f64; 3], y: [f64; 3], z: [f64; 3]) -> [f64; 4] {
    let mat = [x, y, z]; // mat[c][r]
    let mut q = [0.0; 4];
    if mat[0][0] + mat[1][1] + mat[2][2] > 0.0 {
        q[0] = 0.5 * (1.0 + mat[0][0] + mat[1][1] + mat[2][2]).sqrt();
        q[1] = 0.25 * (mat[1][2] - mat[2][1]) / q[0];
        q[2] = 0.25 * (mat[2][0] - mat[0][2]) / q[0];
        q[3] = 0.25 * (mat[0][1] - mat[1][0]) / q[0];
    } else if mat[0][0] > mat[1][1] && mat[0][0] > mat[2][2] {
        q[1] = 0.5 * (1.0 + mat[0][0] - mat[1][1] - mat[2][2]).sqrt();
        q[0] = 0.25 * (mat[1][2] - mat[2][1]) / q[1];
        q[2] = 0.25 * (mat[1][0] + mat[0][1]) / q[1];
        q[3] = 0.25 * (mat[2][0] + mat[0][2]) / q[1];
    } else if mat[1][1] > mat[2][2] {
        q[2] = 0.5 * (1.0 - mat[0][0] + mat[1][1] - mat[2][2]).sqrt();
        q[0] = 0.25 * (mat[2][0] - mat[0][2]) / q[2];
        q[1] = 0.25 * (mat[1][0] + mat[0][1]) / q[2];
        q[3] = 0.25 * (mat[2][1] + mat[1][2]) / q[2];
    } else {
        q[3] = 0.5 * (1.0 - mat[0][0] - mat[1][1] + mat[2][2]).sqrt();
        q[0] = 0.25 * (mat[0][1] - mat[1][0]) / q[3];
        q[1] = 0.25 * (mat[2][0] + mat[0][2]) / q[3];
        q[2] = 0.25 * (mat[2][1] + mat[1][2]) / q[3];
    }
    normvec(&mut q);
    q
}

/// `mjuu_frameaccum` (user_util.cc:452): `(pos, quat) <- (pos, quat) * child`.
pub(crate) fn frameaccum(
    pos: &mut [f64; 3],
    quat: &mut [f64; 4],
    childpos: [f64; 3],
    childquat: [f64; 4],
) {
    let mat = quat2mat(*quat);
    let vec = mulvecmat(childpos, &mat);
    pos[0] += vec[0];
    pos[1] += vec[1];
    pos[2] += vec[2];
    *quat = mulquat(*quat, childquat);
}

/// `mjuu_globalinertia` (user_util.cc:499): the diagonal `local` inertia in the
/// frame `quat`, as the symmetric matrix `(xx, yy, zz, xy, xz, yz)`.
pub(crate) fn globalinertia(local: [f64; 3], quat: [f64; 4]) -> [f64; 6] {
    let mat = quat2mat(quat);
    let tmp = [
        mat[0] * local[0],
        mat[3] * local[0],
        mat[6] * local[0],
        mat[1] * local[1],
        mat[4] * local[1],
        mat[7] * local[1],
        mat[2] * local[2],
        mat[5] * local[2],
        mat[8] * local[2],
    ];
    [
        mat[0] * tmp[0] + mat[1] * tmp[3] + mat[2] * tmp[6],
        mat[3] * tmp[1] + mat[4] * tmp[4] + mat[5] * tmp[7],
        mat[6] * tmp[2] + mat[7] * tmp[5] + mat[8] * tmp[8],
        mat[0] * tmp[1] + mat[1] * tmp[4] + mat[2] * tmp[7],
        mat[0] * tmp[2] + mat[1] * tmp[5] + mat[2] * tmp[8],
        mat[3] * tmp[2] + mat[4] * tmp[5] + mat[5] * tmp[8],
    ]
}

/// `mjuu_offcenter` (user_util.cc:521): the parallel-axis correction
/// `mass * [y^2+z^2, x^2+z^2, x^2+y^2, -xy, -xz, -yz]`.
pub(crate) fn offcenter(mass: f64, v: [f64; 3]) -> [f64; 6] {
    [
        mass * (v[1] * v[1] + v[2] * v[2]),
        mass * (v[0] * v[0] + v[2] * v[2]),
        mass * (v[0] * v[0] + v[1] * v[1]),
        -mass * v[0] * v[1],
        -mass * v[0] * v[2],
        -mass * v[1] * v[2],
    ]
}

/// The result of [`eig3`].
pub(crate) struct Eig3 {
    /// Eigenvalues, in decreasing order (up to the swap threshold).
    pub eigval: [f64; 3],
    /// The principal axes as a quaternion (the frame in which the matrix is diagonal).
    pub quat: [f64; 4],
}

/// `mjuu_eig3` (user_util.cc:659): eigendecomposition of a symmetric 3x3 matrix
/// by Jacobi iteration on a quaternion, then a bubble sort of the eigenvalues.
pub(crate) fn eig3(mat: [f64; 9], reltol: f64) -> Eig3 {
    const EIG_EPS: f64 = 1e-12; // kEigEPS, eigenvalue swap threshold
    const EIG_TOL: f64 = 4e-15; // kEigTOL

    let mut scale = 0.0f64;
    for v in mat {
        scale = scale.max(v.abs());
    }
    let tol = scale * reltol.max(EIG_TOL);

    let mut quat = [1.0, 0.0, 0.0, 0.0];
    let mut eigval = [0.0; 3];

    for _ in 0..500 {
        let eigvec = quat2mat(quat);
        let tmp2 = transposemat(&eigvec);
        let tmp = mulmat(&tmp2, &mat);
        let d = mulmat(&tmp, &eigvec);

        eigval = [d[0], d[4], d[8]];

        let (rk, ck, rotk) = if d[1].abs() > d[2].abs() && d[1].abs() > d[5].abs() {
            (0, 1, 2)
        } else if d[2].abs() > d[5].abs() {
            (0, 2, 1)
        } else {
            (1, 2, 0)
        };

        if d[3 * rk + ck].abs() <= tol {
            break;
        }

        let tau = (d[4 * ck] - d[4 * rk]) / (2.0 * d[3 * rk + ck]);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };

        let h = t / (1.0 + (1.0 + t * t).sqrt());
        let mut rot = [0.0; 4];
        rot[0] = 1.0 / (1.0 + h * h).sqrt();
        rot[rotk + 1] = (if rotk == 1 { h } else { -h }) * rot[0];

        quat = mulquat(quat, rot);
        normvec(&mut quat);
    }

    let eps = scale * reltol.max(EIG_EPS);
    for j in 0..3 {
        let j1 = j % 2;
        if eigval[j1] + eps < eigval[j1 + 1] {
            eigval.swap(j1, j1 + 1);
            let mut rot = [0.0; 4];
            // MuJoCo's literal (cos(pi/4) to 15 digits), not FRAC_1_SQRT_2: the
            // two differ in the last bits and the port is exact.
            #[allow(clippy::approx_constant)]
            let cos_quarter_pi = 0.707106781186548;
            rot[0] = cos_quarter_pi;
            rot[(j1 + 2) % 3 + 1] = rot[0];
            quat = mulquat(quat, rot);
            normvec(&mut quat);
        }
    }

    Eig3 { eigval, quat }
}

/// `mjuu_fullInertia` (user_util.cc:851): principal axes and moments of a full
/// inertia `(xx, yy, zz, xy, xz, yz)`. Errors with MuJoCo's message when the
/// smallest eigenvalue is below `EPS`.
pub(crate) fn full_inertia(full: [f64; 6]) -> Result<([f64; 4], [f64; 3]), &'static str> {
    let m = [
        full[0], full[3], full[4], full[3], full[1], full[5], full[4], full[5], full[2],
    ];
    let e = eig3(m, EIG_RELTOL);
    if e.eigval[2] < EPS {
        return Err("inertia must have positive eigenvalues");
    }
    Ok((e.quat, e.eigval))
}

/// How a frame's orientation was written in the XML (MuJoCo's `mjsOrientation`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum OrientKind {
    /// The `quat` attribute (or nothing): nothing to resolve.
    Quat,
    /// `axisangle="x y z angle"`.
    AxisAngle,
    /// `xyaxes="x1 x2 x3 y1 y2 y3"`.
    XyAxes,
    /// `zaxis="x y z"`.
    ZAxis,
    /// `euler="a b c"`, in the compiler's `eulerseq`.
    Euler,
}

/// The orientation alternatives of one frame (`mjsOrientation`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Alt {
    pub kind: OrientKind,
    pub axisangle: [f64; 4],
    pub xyaxes: [f64; 6],
    pub zaxis: [f64; 3],
    pub euler: [f64; 3],
}

impl Default for Alt {
    fn default() -> Self {
        Alt {
            kind: OrientKind::Quat,
            axisangle: [0.0; 4],
            xyaxes: [0.0; 6],
            zaxis: [0.0; 3],
            euler: [0.0; 3],
        }
    }
}

/// `ResolveOrientation` (user_objects.cc:273): the quaternion of an orientation
/// alternative. Returns MuJoCo's error text on failure. `quat` is left alone for
/// [`OrientKind::Quat`].
pub(crate) fn resolve_orientation(
    quat: &mut [f64; 4],
    degree: bool,
    sequence: &[u8; 3],
    alt: &Alt,
) -> Result<(), &'static str> {
    let mut axisangle = alt.axisangle;
    let mut xyaxes = alt.xyaxes;
    let mut zaxis = alt.zaxis;
    let mut euler = alt.euler;

    if alt.kind == OrientKind::AxisAngle {
        if degree {
            axisangle[3] = axisangle[3] / 180.0 * PI;
        }
        if normvec(&mut axisangle[..3]) < EPS {
            return Err("axisangle too small");
        }
        let ang2 = axisangle[3] / 2.0;
        quat[0] = ang2.cos();
        quat[1] = ang2.sin() * axisangle[0];
        quat[2] = ang2.sin() * axisangle[1];
        quat[3] = ang2.sin() * axisangle[2];
    }

    if alt.kind == OrientKind::XyAxes {
        if normvec(&mut xyaxes[..3]) < EPS {
            return Err("xaxis too small");
        }
        let d = dot3(
            [xyaxes[0], xyaxes[1], xyaxes[2]],
            [xyaxes[3], xyaxes[4], xyaxes[5]],
        );
        xyaxes[3] -= xyaxes[0] * d;
        xyaxes[4] -= xyaxes[1] * d;
        xyaxes[5] -= xyaxes[2] * d;
        if normvec(&mut xyaxes[3..]) < EPS {
            return Err("yaxis too small");
        }
        let x = [xyaxes[0], xyaxes[1], xyaxes[2]];
        let y = [xyaxes[3], xyaxes[4], xyaxes[5]];
        let mut z = cross(x, y);
        if normvec(&mut z) < EPS {
            return Err("cross(xaxis, yaxis) too small");
        }
        *quat = frame2quat(x, y, z);
    }

    if alt.kind == OrientKind::ZAxis {
        if normvec(&mut zaxis) < EPS {
            return Err("zaxis too small");
        }
        *quat = z2quat(zaxis);
    }

    if alt.kind == OrientKind::Euler {
        if degree {
            for e in euler.iter_mut() {
                *e = *e / 180.0 * PI;
            }
        }
        *quat = [1.0, 0.0, 0.0, 0.0];
        for i in 0..3 {
            let mut qrot = [(euler[i] / 2.0).cos(), 0.0, 0.0, 0.0];
            let sa = (euler[i] / 2.0).sin();
            match sequence[i] {
                b'x' | b'X' => qrot[1] = sa,
                b'y' | b'Y' => qrot[2] = sa,
                b'z' | b'Z' => qrot[3] = sa,
                _ => return Err("euler sequence can only contain x, y, z, X, Y, Z"),
            }
            let tmp = if matches!(sequence[i], b'x' | b'y' | b'z') {
                mulquat(*quat, qrot) // moving axes: post-multiply
            } else {
                mulquat(qrot, *quat) // fixed axes: pre-multiply
            };
            *quat = tmp;
        }
        normvec(quat);
    }
    Ok(())
}
