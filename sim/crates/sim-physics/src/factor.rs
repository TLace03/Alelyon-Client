//! The dense `L'DL` factorisation of the joint-space inertia matrix and its solve.
//!
//! Ports `mj_factorI` and `mj_solveLD` (engine_core_smooth.c, 3.14.0) from
//! MuJoCo's sparse, tree-structured storage to a dense row-major `n * n` array.
//!
//! Why it is dense and what differs from MuJoCo (the spec's "factorisation
//! differs" note):
//! - MuJoCo stores only the entries of `M` that a kinematic tree can make non-zero
//!   (a dof and its ancestors) and factors them backwards (`M = L' D L` with `L`
//!   unit lower triangular), which creates no fill-in. This file runs the same
//!   backward elimination, in the same order, on the dense array, where the
//!   structural zeros are explicit zeros. Every non-zero is computed with the same
//!   operations in the same order; an operation on a structural zero adds or
//!   subtracts an exact zero. So the factors equal MuJoCo's up to the sign of a
//!   zero. The solve ([`solve`]) visits the entries of the sparsity of `M` in
//!   MuJoCo's order and ends in MuJoCo's `mju_dotSparse` order of sums, so `qacc`
//!   follows MuJoCo's rounding (in phase 1b it summed a dense row, and only
//!   rounding differed; the constraint solvers iterate on this solve).
//! - No fused multiply-add: every product and sum is a separate operation, in the
//!   order written (see [`crate::Real`]).
//! - MuJoCo's "simple" dofs (`dof_simplenum`) have a stored matrix row that is the
//!   single constant diagonal `dof_M0`: `crb` copies it (`Model::dof_m0`, phase 1c-ii; the
//!   general composite-inertia path computes the same diagonal from the rotated `cinert`,
//!   which differs from it by a few units in the last place, and 1b missed that), so the
//!   dense `M` of a simple dof has that diagonal and exact zeros, as MuJoCo stores; the solve
//!   and the products with `M` skip the off-diagonals ([`crate::Sparsity`]).
//! - O(n^3) work and O(n^2) memory instead of O(n * depth): fine for the model sizes
//!   of this phase (`nv` of a few dozen), and the thing a GPU port replaces.
//!
//! Invariants:
//! - `mat` is the lower triangle (row `k`, columns `0..=k`) of a symmetric positive
//!   definite matrix; the strict upper triangle is not read or written.
//! - A pivot below `mjMINVAL` is clamped up to it (MuJoCo's near-singular guard), and
//!   the first clamped index is returned.

use crate::math::min_val;
use crate::model::Sparsity;
use crate::real::Real;

/// Port of `mj_factorI`: factors the dense lower triangle `mat` (`n * n`,
/// row-major) in place as `L'DL`, `L` unit lower triangular stored below the
/// diagonal and `D` on the diagonal, and writes `1 / D` to `diag_inv`.
///
/// Returns the index of the first pivot that had to be clamped to `mjMINVAL`, or
/// `None`.
pub(crate) fn factor<R: Real>(mat: &mut [R], diag_inv: &mut [R], n: usize) -> Option<usize> {
    let mut clamped = None;
    // backward loop over rows
    for k in (0..n).rev() {
        // clamp a small or non-positive pivot from below
        if mat[k * n + k] < min_val::<R>() {
            mat[k * n + k] = min_val::<R>();
            if clamped.is_none() {
                clamped = Some(k);
            }
        }
        let inv_d = R::ONE / mat[k * n + k];
        diag_inv[k] = inv_d;

        // update the triangle above row k: row i < k, columns 0..=i, gets
        // row k's entries times -L(k, i) / D(k)
        for i in (0..k).rev() {
            let scl = -mat[k * n + i] * inv_d;
            for c in 0..=i {
                let upd = mat[k * n + c] * scl;
                mat[i * n + c] += upd;
            }
        }

        // row k: L(k, :) /= D(k)
        for c in 0..k {
            mat[k * n + c] *= inv_d;
        }
    }
    clamped
}

/// Port of `mj_solveLD` for one vector: `x <- inv(L'DL) x`, in place, visiting the
/// entries of `L` that the sparsity `sp` of `M` lists, in MuJoCo's order (a row of `L`
/// holds a dof's ancestors, columns ascending). The last step is MuJoCo's
/// `mju_dotSparse`: four interleaved partial sums over blocks of four ancestors, then
/// the rest one at a time. A row with the diagonal only (a root dof, a simple dof) is
/// skipped.
pub(crate) fn solve<R: Real>(x: &mut [R], ld: &[R], diag_inv: &[R], n: usize, sp: &Sparsity) {
    // x <- L^-T x
    for i in (0..n).rev() {
        // skip diagonal rows
        if sp.rownnz[i] == 1 {
            continue;
        }
        let x_i = x[i];
        if x_i != R::ZERO {
            let start = sp.rowadr[i];
            let end = start + sp.rownnz[i] - 1;
            for adr in start..end {
                let c = sp.colind[adr];
                x[c] -= ld[i * n + c] * x_i;
            }
        }
    }
    // x <- D^-1 x
    for i in 0..n {
        x[i] *= diag_inv[i];
    }
    // x <- L^-1 x
    for i in 0..n {
        // skip diagonal rows
        if sp.rownnz[i] == 1 {
            continue;
        }
        let d = sp.rownnz[i] - 1;
        let ind = &sp.colind[sp.rowadr[i]..sp.rowadr[i] + d];
        // mju_dotSparse(qLD + adr, x, d, colind + adr)
        let (mut r0, mut r1, mut r2, mut r3) = (R::ZERO, R::ZERO, R::ZERO, R::ZERO);
        let mut k = 0;
        while k + 4 <= d {
            r0 += ld[i * n + ind[k]] * x[ind[k]];
            r1 += ld[i * n + ind[k + 1]] * x[ind[k + 1]];
            r2 += ld[i * n + ind[k + 2]] * x[ind[k + 2]];
            r3 += ld[i * n + ind[k + 3]] * x[ind[k + 3]];
            k += 4;
        }
        let mut dot = (r0 + r2) + (r1 + r3);
        while k < d {
            dot += ld[i * n + ind[k]] * x[ind[k]];
            k += 1;
        }
        x[i] -= dot;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sparsity of a dense lower triangle: row `i` holds the columns `0..=i`.
    fn dense_sparsity(n: usize) -> Sparsity {
        let mut s = Sparsity::default();
        for i in 0..n {
            s.rowadr.push(s.colind.len());
            s.colind.extend(0..=i);
            s.rownnz.push(i + 1);
        }
        s
    }

    /// A symmetric positive definite matrix and its product with a vector.
    fn spd(n: usize) -> Vec<f64> {
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = 1.0 / (1.0 + (i as f64 - j as f64).abs()) * 0.3;
            }
            a[i * n + i] += 1.0 + i as f64;
        }
        a
    }

    #[test]
    fn factor_and_solve_invert_a_spd_matrix() {
        let n = 6;
        let a = spd(n);
        let mut ld = a.clone();
        let mut inv = vec![0.0; n];
        assert_eq!(factor(&mut ld, &mut inv, n), None);
        let b: Vec<f64> = (0..n).map(|i| 0.5 + i as f64).collect();
        let mut x = b.clone();
        solve(&mut x, &ld, &inv, n, &dense_sparsity(n));
        for (i, bi) in b.iter().enumerate() {
            let ax: f64 = (0..n).map(|j| a[i * n + j] * x[j]).sum();
            assert!((ax - bi).abs() < 1e-13, "row {i}: {ax} vs {bi}");
        }
    }

    #[test]
    fn a_tree_structured_matrix_factors_without_fill() {
        // dof 0 is the parent of the siblings 1 and 2: (1,0) and (2,0) are
        // non-zero, (2,1) is a structural zero, and the factor keeps it zero
        let n = 3;
        let mut m = vec![0.0; n * n];
        m[0] = 2.0;
        m[n] = 0.5;
        m[n + 1] = 3.0;
        m[2 * n] = 0.25;
        m[2 * n + 2] = 4.0;
        let mut ld = m.clone();
        let mut inv = vec![0.0; n];
        factor(&mut ld, &mut inv, n);
        assert_eq!(ld[2 * n + 1], 0.0);
        let mut x = vec![1.0, 2.0, 3.0];
        let b = x.clone();
        // dof 1 and dof 2 are siblings below dof 0: their rows list dof 0 only
        let sp = Sparsity {
            rownnz: vec![1, 2, 2],
            rowadr: vec![0, 1, 3],
            colind: vec![0, 0, 1, 0, 2],
        };
        solve(&mut x, &ld, &inv, n, &sp);
        let full = |i: usize, j: usize| if i >= j { m[i * n + j] } else { m[j * n + i] };
        for (i, bi) in b.iter().enumerate() {
            let ax: f64 = (0..n).map(|j| full(i, j) * x[j]).sum();
            assert!((ax - bi).abs() < 1e-14);
        }
    }

    #[test]
    fn a_non_positive_pivot_is_clamped_and_reported() {
        let n = 2;
        let mut m = vec![1.0, 0.0, 0.0, 0.0];
        let mut inv = vec![0.0; n];
        assert_eq!(factor(&mut m, &mut inv, n), Some(1));
        assert_eq!(m[3], 1e-15);
        assert!((inv[1] - 1e15).abs() < 1.0, "{}", inv[1]);
    }
}
