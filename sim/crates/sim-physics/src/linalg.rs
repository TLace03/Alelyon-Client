//! Dense linear algebra of the constraint solver, ported from MuJoCo's BLAS-like and
//! Cholesky routines (3.14.0).
//!
//! Ports, from `engine_util_blas.c`: `mju_dot`, `mju_mulMatVec`, `mju_mulMatTVec`,
//! `mju_sqrMatTD_impl` (the lower triangle of `J' diag J`); from
//! `engine_util_solve.c`: `mju_cholFactor`, `mju_cholSolve`, `mju_cholUpdate`; from
//! `engine_util_sparse.c`: `mju_mulSymVecSparse` and `mju_addToSymSparse` (the
//! products with the joint-space inertia matrix).
//!
//! Invariants:
//! - **Same operation order as MuJoCo.** `mju_dot` sums in four interleaved partial
//!   sums, `(s0 + s2) + (s1 + s3)`, and then the up to three leftover products; the
//!   AVX build of MuJoCo adds in exactly that order, so this port matches it whatever
//!   the platform. The other routines keep the loop nest, the skips of exact zeros
//!   (which change no value for finite numbers) and the order of the sums of the C
//!   source. No fused multiply-add anywhere (see [`crate::Real`]).
//! - **Dense storage instead of MuJoCo's sparse inertia matrix.** MuJoCo stores `M`
//!   as the entries a kinematic tree can make non-zero (a dof and its ancestors).
//!   [`mul_sym_vec_sparse`] and [`add_sym_sparse`] walk exactly the entries of the
//!   [`crate::Sparsity`] of `M` in the dense `qm`, in MuJoCo's order (the diagonal, then
//!   the ancestors from the nearest), so the products are the ones MuJoCo computes.
//!   The only difference is that the dense `qm` of a "simple" body holds rounding-size
//!   values where MuJoCo stores no entry at all (those entries are never read; see
//!   `factor.rs`).
//! - No allocation: every routine writes to slices the caller owns.

use crate::model::Sparsity;
use crate::real::Real;

/// MuJoCo's `mju_max`.
#[inline]
pub(crate) fn max<R: Real>(a: R, b: R) -> R {
    if a >= b { a } else { b }
}

/// MuJoCo's `mju_min`.
#[inline]
pub(crate) fn min<R: Real>(a: R, b: R) -> R {
    if a <= b { a } else { b }
}

/// Port of `mju_dot`: the dot product of `a` and `b` (the length of `a`).
#[inline]
pub(crate) fn dot<R: Real>(a: &[R], b: &[R]) -> R {
    let n = a.len();
    debug_assert!(b.len() >= n);
    let (mut r0, mut r1, mut r2, mut r3) = (R::ZERO, R::ZERO, R::ZERO, R::ZERO);
    let mut i = 0;
    while i + 4 <= n {
        r0 += a[i] * b[i];
        r1 += a[i + 1] * b[i + 1];
        r2 += a[i + 2] * b[i + 2];
        r3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut res = (r0 + r2) + (r1 + r3);
    match n - i {
        3 => res += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2],
        2 => res += a[i] * b[i] + a[i + 1] * b[i + 1],
        1 => res += a[i] * b[i],
        _ => {}
    }
    res
}

/// Port of `mju_norm`: `sqrt(dot(v, v))`.
#[inline]
pub(crate) fn norm<R: Real>(v: &[R]) -> R {
    dot(v, v).sqrt()
}

/// Port of `mju_mulMatVec`: `res = mat * vec` for the `nr x nc` row-major `mat`.
pub(crate) fn mul_mat_vec<R: Real>(res: &mut [R], mat: &[R], vec: &[R], nr: usize, nc: usize) {
    for r in 0..nr {
        res[r] = dot(&mat[r * nc..r * nc + nc], &vec[..nc]);
    }
}

/// Port of `mju_mulMatTVec`: `res = mat' * vec` for the `nr x nc` row-major `mat`
/// (rows whose `vec` entry is exactly zero are skipped, as in MuJoCo).
pub(crate) fn mul_mat_t_vec<R: Real>(res: &mut [R], mat: &[R], vec: &[R], nr: usize, nc: usize) {
    res[..nc].fill(R::ZERO);
    for r in 0..nr {
        let tmp = vec[r];
        if tmp != R::ZERO {
            for c in 0..nc {
                res[c] += mat[r * nc + c] * tmp;
            }
        }
    }
}

/// Port of `mju_sqrMatTD_impl` with `flg_upper = 0`: the lower triangle (diagonal
/// included) of `mat' * diag * mat` for the `nr x nc` row-major `mat`, written to
/// the `nc x nc` row-major `res`; the strict upper triangle of `res` is zero.
pub(crate) fn sqr_mat_td_lower<R: Real>(
    res: &mut [R],
    mat: &[R],
    diag: &[R],
    nr: usize,
    nc: usize,
) {
    res[..nc * nc].fill(R::ZERO);
    for j in 0..nr {
        if diag[j] != R::ZERO {
            for i in 0..nc {
                let tmp = mat[j * nc + i];
                if tmp != R::ZERO {
                    let scl = tmp * diag[j];
                    // mju_addToScl(res + i * nc, mat + j * nc, scl, i + 1)
                    for k in 0..=i {
                        res[i * nc + k] += mat[j * nc + k] * scl;
                    }
                }
            }
        }
    }
}

/// Port of `mju_mulSymVecSparse` for the dense inertia matrix `qm`, visiting the
/// entries of the sparsity `sp` of `M`: `res = M * vec`, summed the way MuJoCo sums its
/// sparse `M` (each row: the diagonal, then the other entries from the last stored
/// column, i.e. the nearest ancestor, back to the first; each entry is used for the row
/// and, mirrored, for the column's row).
pub(crate) fn mul_sym_vec_sparse<R: Real>(
    res: &mut [R],
    qm: &[R],
    vec: &[R],
    n: usize,
    sp: &Sparsity,
) {
    res[..n].fill(R::ZERO);
    for i in 0..n {
        let adr = sp.rowadr[i];
        let diag = sp.rownnz[i] - 1;
        // diagonal
        res[i] = qm[i * n + i] * vec[i];
        // off-diagonals
        for k in (0..diag).rev() {
            let j = sp.colind[adr + k];
            let val = qm[i * n + j];
            res[i] += val * vec[j]; // strict lower
            res[j] += val * vec[i]; // strict upper
        }
    }
}

/// Port of `mju_addToSymSparse` (lower triangle and diagonal): adds the entries of
/// `qm` that the sparsity `sp` of `M` lists to the lower triangle of the dense `res`.
pub(crate) fn add_sym_sparse<R: Real>(res: &mut [R], qm: &[R], n: usize, sp: &Sparsity) {
    for i in 0..n {
        for adr in sp.rowadr[i]..sp.rowadr[i] + sp.rownnz[i] {
            let j = sp.colind[adr];
            res[i * n + j] += qm[i * n + j];
        }
    }
}

/// Port of `mju_cholFactor`: the in-place Cholesky factorisation `mat = L L'` of the
/// lower triangle of the `n x n` row-major `mat`; a pivot below `mindiag` is replaced
/// by `mindiag` and its column below the diagonal cleared. Returns the rank (`n`
/// when no pivot was deficient).
pub(crate) fn chol_factor<R: Real>(mat: &mut [R], n: usize, mindiag: R) -> usize {
    let mut rank = n;
    for j in 0..n {
        // compute the new diagonal
        let mut tmp = mat[j * (n + 1)];
        if j > 0 {
            let row = &mat[j * n..j * n + j];
            tmp -= dot(row, row);
        }

        // correct diagonal values below the threshold
        let deficient = tmp < mindiag;
        if deficient {
            tmp = mindiag;
            rank -= 1;
        }

        // save the diagonal
        mat[j * (n + 1)] = tmp.sqrt();

        // process the off-diagonal entries
        if deficient {
            for i in j + 1..n {
                mat[i * n + j] = R::ZERO;
            }
        } else {
            let inv = R::ONE / mat[j * (n + 1)];
            for i in j + 1..n {
                let d = dot(&mat[i * n..i * n + j], &mat[j * n..j * n + j]);
                mat[i * n + j] = (mat[i * n + j] - d) * inv;
            }
        }
    }
    rank
}

/// Port of `mju_cholSolve`: `res = (L L')^-1 vec` with the factor from
/// [`chol_factor`] (`res` and `vec` are different slices).
pub(crate) fn chol_solve<R: Real>(res: &mut [R], mat: &[R], vec: &[R], n: usize) {
    res[..n].copy_from_slice(&vec[..n]);

    // forward substitution: solve L * res = vec
    for i in 0..n {
        if i > 0 {
            let d = dot(&mat[i * n..i * n + i], &res[..i]);
            res[i] -= d;
        }
        res[i] /= mat[i * (n + 1)];
    }

    // backward substitution: solve L' * res = res
    for i in (0..n).rev() {
        if i + 1 < n {
            for j in i + 1..n {
                let t = mat[j * n + i] * res[j];
                res[i] -= t;
            }
        }
        res[i] /= mat[i * (n + 1)];
    }
}

/// Port of `mju_cholUpdate`: the rank-one update (`plus`) or downdate of the factor
/// `mat`, `L L' +/- x x'`; `x` is used as scratch. Returns the rank (less than `n`
/// when a pivot had to be clamped to `mjMINVAL`, which asks the caller to
/// refactorise).
pub(crate) fn chol_update<R: Real>(mat: &mut [R], x: &mut [R], n: usize, plus: bool) -> usize {
    let min_val = R::from_f64(1e-15);
    let mut rank = n;
    for k in 0..n {
        if x[k] != R::ZERO {
            // prepare constants
            let lkk = mat[k * (n + 1)];
            let mut tmp = lkk * lkk + if plus { x[k] * x[k] } else { -x[k] * x[k] };
            if tmp < min_val {
                tmp = min_val;
                rank -= 1;
            }
            let r = tmp.sqrt();
            let c = r / lkk;
            let cinv = R::ONE / c;
            let s = x[k] / lkk;

            // update the diagonal
            mat[k * (n + 1)] = r;

            // update mat
            if plus {
                for i in k + 1..n {
                    mat[i * n + k] = (mat[i * n + k] + s * x[i]) * cinv;
                }
            } else {
                for i in k + 1..n {
                    mat[i * n + k] = (mat[i * n + k] - s * x[i]) * cinv;
                }
            }

            // update x
            for i in k + 1..n {
                x[i] = c * x[i] - s * mat[i * n + k];
            }
        }
    }
    rank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_sums_in_four_interleaved_partials() {
        // a sum whose value depends on the order: one block of four, (1e16 + -1e16) +
        // (1 + 1) = 2, then the three leftover products 1 + 1 + 1 added as one group
        // (MuJoCo's `res += a*b + c*d + e*f`): 5. A left-to-right sum loses the first
        // 1 (1e16 + 1 rounds to 1e16) and gives 4.
        let a = [1e16, 1.0, -1e16, 1.0, 1.0, 1.0, 1.0];
        let b = [1.0; 7];
        assert_eq!(dot(&a, &b), 5.0);
        let sequential = a.iter().zip(&b).fold(0.0, |s, (x, y)| s + x * y);
        assert_eq!(sequential, 4.0);
        assert_eq!(dot::<f64>(&[], &[]), 0.0);
        assert_eq!(dot(&[2.0], &[3.0]), 6.0);
    }

    #[test]
    fn cholesky_factor_solve_and_update_agree_with_a_direct_solve() {
        let n = 4;
        // a symmetric positive definite matrix, lower triangle row-major
        let a: Vec<f64> = (0..n * n)
            .map(|k| {
                let (i, j) = (k / n, k % n);
                let v = 1.0 / (1.0 + (i as f64 - j as f64).abs());
                if i == j { v + 2.0 } else { v }
            })
            .collect();
        let b = [1.0, -2.0, 0.5, 3.0];
        let mut l = a.clone();
        assert_eq!(chol_factor(&mut l, n, 1e-15), n);
        let mut x = vec![0.0; n];
        chol_solve(&mut x, &l, &b, n);
        for i in 0..n {
            let ax: f64 = (0..n).map(|j| a[i * n + j] * x[j]).sum();
            assert!((ax - b[i]).abs() < 1e-13, "{i}: {ax}");
        }

        // a rank-one update followed by the same downdate returns the factor
        let l0 = l.clone();
        let u = [0.3, -0.2, 0.5, 0.1];
        let mut up = u;
        assert_eq!(chol_update(&mut l, &mut up, n, true), n);
        let mut down = u;
        assert_eq!(chol_update(&mut l, &mut down, n, false), n);
        for i in 0..n {
            for j in 0..=i {
                assert!(
                    (l[i * n + j] - l0[i * n + j]).abs() < 1e-13,
                    "({i},{j}) {} vs {}",
                    l[i * n + j],
                    l0[i * n + j]
                );
            }
        }
        // and the updated factor factors A + u u'
        let mut l1 = a.clone();
        for i in 0..n {
            for j in 0..n {
                l1[i * n + j] += u[i] * u[j];
            }
        }
        assert_eq!(chol_factor(&mut l1, n, 1e-15), n);
        let mut up = u;
        let mut l2 = l0.clone();
        chol_update(&mut l2, &mut up, n, true);
        for i in 0..n {
            for j in 0..=i {
                assert!((l1[i * n + j] - l2[i * n + j]).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn a_deficient_pivot_is_clamped_and_the_rank_drops() {
        let n = 2;
        let mut m = vec![1.0, 0.0, 1.0, 1.0]; // [[1, ?], [1, 1]] lower: rank 1
        assert_eq!(chol_factor(&mut m, n, 1e-15), 1);
        assert_eq!(m[3], 1e-15f64.sqrt());
    }

    #[test]
    fn sparse_products_equal_the_dense_product_of_a_tree_matrix() {
        // dofs 0 -> 1 -> 2 (a chain) and 3 (a root): M(3, *) = 0 off the diagonal
        let n = 4;
        // rows: dof 0 {0}, dof 1 {0, 1}, dof 2 {0, 1, 2}, dof 3 {3}
        let sp = Sparsity {
            rownnz: vec![1, 2, 3, 1],
            rowadr: vec![0, 1, 3, 6],
            colind: vec![0, 0, 1, 0, 1, 2, 3],
        };
        let mut m = vec![0.0; n * n];
        let vals = [
            (0, 0, 3.0),
            (1, 0, 0.5),
            (1, 1, 2.0),
            (2, 0, 0.2),
            (2, 1, 0.4),
            (2, 2, 1.5),
            (3, 3, 4.0),
        ];
        for &(i, j, v) in &vals {
            m[i * n + j] = v;
            m[j * n + i] = v;
        }
        let v = [1.0, -2.0, 0.5, 3.0];
        let mut res = vec![0.0; n];
        mul_sym_vec_sparse(&mut res, &m, &v, n, &sp);
        for i in 0..n {
            let e: f64 = (0..n).map(|j| m[i * n + j] * v[j]).sum();
            assert!((res[i] - e).abs() < 1e-14);
        }
        let mut h = vec![0.0; n * n];
        add_sym_sparse(&mut h, &m, n, &sp);
        for i in 0..n {
            for j in 0..=i {
                assert_eq!(h[i * n + j], m[i * n + j]);
            }
            for j in i + 1..n {
                assert_eq!(h[i * n + j], 0.0, "the upper triangle is not written");
            }
        }
    }

    #[test]
    fn sqr_mat_td_is_the_lower_triangle_of_j_t_d_j() {
        let (nr, nc) = (3, 3);
        let j = [1.0, 0.0, 2.0, 0.0, 1.0, -1.0, 0.5, 0.5, 0.0];
        let d = [2.0, 0.0, 3.0];
        let mut h = vec![9.0; nc * nc];
        sqr_mat_td_lower(&mut h, &j, &d, nr, nc);
        for a in 0..nc {
            for b in 0..nc {
                let e: f64 = (0..nr).map(|r| j[r * nc + a] * d[r] * j[r * nc + b]).sum();
                if b <= a {
                    assert!((h[a * nc + b] - e).abs() < 1e-14);
                } else {
                    assert_eq!(h[a * nc + b], 0.0);
                }
            }
        }
    }
}
