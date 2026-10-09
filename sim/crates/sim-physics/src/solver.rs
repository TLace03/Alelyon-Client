//! MuJoCo's primal constraint solvers, Newton and conjugate gradient, with the exact
//! line search, monolithic (no islands), dense.
//!
//! Ports, from MuJoCo 3.14.0 `engine_solver.c`: `mj_solPrimal` (called with
//! `flg_Newton` 1 for `mj_solNewton` and 0 for `mj_solCG`) with `PrimalUpdateConstraint`,
//! `PrimalUpdateGrad`, `PrimalUpdateMgrad`, `PrimalPrepare`, `PrimalEval`,
//! `updateBracket`, `PrimalSearch`, `frictionCost`, `frictionCostDif`, `MakeHessian`,
//! `FactorizeHessian` and `HessianIncremental` (the dense branches), and from
//! `engine_util_solve.c` the dense Cholesky routines (in `linalg.rs`).
//!
//! The problem the solvers minimise over `qacc` is MuJoCo's primal form:
//!
//! `cost(qacc) = 0.5 (qacc - qacc_smooth)' M (qacc - qacc_smooth) + s(J qacc - aref)`
//!
//! with `s` the sum of the row costs of `constraint.rs` (quadratic for an active
//! limit or a friction row inside its bound, linear for a friction row beyond it,
//! zero for a satisfied limit). The cost is convex (strictly, `M` being positive
//! definite) and piecewise quadratic, so its minimum is unique.
//!
//! Invariants:
//! - **Same iterates as MuJoCo, within rounding.** The solve follows `mj_solPrimal`
//!   statement for statement: the convergence certificate before any Hessian is built
//!   (the cost gap `0.5 grad' M^-1 grad` scaled by `1 / (meaninertia max(1, nv))`,
//!   and for Newton also the scaled gradient norm), one Newton or conjugate direction
//!   per iteration, the three-point bracketing line search with Newton steps on the
//!   one-dimensional cost, the termination rules (`improvement` or `gradient` below
//!   `tolerance`, the Newton decrement below it, `iterations`), and Hager-Zhang
//!   conjugate directions for CG. The line search's own tolerance is
//!   `tolerance * ls_tolerance` (a slope, scaled), its length `ls_iterations`.
//! - **The Hessian is updated incrementally, as MuJoCo does**: `M + J' D J` is built
//!   and Cholesky-factored once (dense, lower triangle), and when a row changes zone
//!   its `J_i sqrt(D_i)` is added to or removed from the factor by a rank-one update
//!   (`mju_cholUpdate`); if an update loses rank the Hessian is rebuilt and
//!   refactored. It is not recomputed from scratch each iteration.
//! - **Elliptic cones** (phase 1c-ii). A contact with an elliptic cone is `dim` rows that share one
//!   zone (top: no force; bottom: every row quadratic; middle: the cone, state `Cone`). `PrimalPrepare`
//!   keeps 9 numbers per cone (`q0, q1, q2, U0, V0, UU, UV, VV, Dm`) in the quad slots of its first
//!   three rows, `PrimalEval` takes the cone's cost along the search line from `ellipticCostDif` and
//!   its derivatives from `N` and `T`, and the Newton solver adds each cone-zone contact's Hessian
//!   (`contact_h`, stored by `PrimalUpdateConstraint`) to the factor as `dim` rank-one updates of
//!   `Lcone` (`HessianConeUpdate`, dense); `PrimalUpdateMgrad` solves with `Lcone` while any row is
//!   in the cone zone. CG never builds a cone Hessian. Every elliptic contact has `dim >= 3`
//!   (`sim-scene` refuses `condim 2`), which the nine numbers rely on.
//! - **Deviations from MuJoCo.** The `M` products are the dense `qm` summed in the
//!   order of MuJoCo's sparse `M` (see `linalg.rs`); the CG preconditioner solve is
//!   the dense `L'DL` solve of `factor.rs`. MuJoCo's per-iteration statistics
//!   (`mjSolverStat`) are not kept (the last iteration's record, the largest line-search and
//!   update counts of the solve, the largest number of rank-one updates actually performed in an
//!   iteration and the number of rank-loss refactorisations are); the sparse `HessianConeFolded`, the island and the
//!   discrete-metric branches are not ported.
//! - No allocation: every array is in [`crate::data::Workspace`].

use sim_scene::Cone;

use crate::constraint::{ConstraintState, ConstraintType, RowParams, constraint_update_impl};
use crate::data::{Data, SolverStat, Workspace};
use crate::factor;
use crate::linalg::{
    add_sym_sparse, chol_factor, chol_solve, chol_update, dot, max, min, mul_mat_t_vec,
    mul_mat_vec, mul_sym_vec_sparse, norm, sqr_mat_td_lower,
};
use crate::math::{lit, min_val};
use crate::model::{Model, PrimalSolver, Sparsity};
use crate::real::Real;
use crate::smooth::Faults;

/// A point on the line search: the step, the cost relative to the start, and the
/// first and second derivatives of the cost along the search direction.
#[derive(Clone, Copy)]
struct Pnt<R: Real> {
    alpha: R,
    cost: R,
    deriv: [R; 2],
}

impl<R: Real> Pnt<R> {
    fn at(alpha: R) -> Pnt<R> {
        Pnt {
            alpha,
            cost: R::ZERO,
            deriv: [R::ZERO; 2],
        }
    }
}

/// MuJoCo's `mjPrimalContext` for the dense monolithic case: borrows of the data a
/// solve reads and writes.
struct Primal<'a, R: Real> {
    nv: usize,
    nf: usize,
    nefc: usize,
    /// `1 / (meaninertia * max(1, nv))`, the scale of the improvement and gradient.
    scale: R,
    tolerance: R,
    unit_step: bool,
    // inertia
    qm: &'a [R],
    sparsity: &'a Sparsity,
    qld: &'a [R],
    qld_diag_inv: &'a [R],
    // inputs
    qfrc_smooth: &'a [R],
    qacc_smooth: &'a [R],
    j: &'a [R],
    efc_d: &'a [R],
    efc_r: &'a [R],
    efc_frictionloss: &'a [R],
    efc_aref: &'a [R],
    efc_type: &'a [ConstraintType],
    efc_id: &'a [usize],
    contact_mu: &'a [R],
    contact_friction: &'a [R],
    contact_dim: &'a [usize],
    // outputs
    qfrc_constraint: &'a mut [R],
    qacc: &'a mut [R],
    efc_force: &'a mut [R],
    efc_state: &'a mut [ConstraintState],
    contact_h: &'a mut [R],
    ws: &'a mut Workspace<R>,
    // globals
    /// The test-only fault that leaves the cone Hessian out of the Newton factor.
    drop_cone_hessian: bool,
    /// The constraint plus Gauss cost at the current `qacc`.
    cost: R,
    /// The quadratic polynomial of the Gauss cost along the search direction.
    quad_gauss: [R; 3],
    nactive: usize,
    /// The number of rows in the cone zone (MuJoCo's `ncone` counts rows, as the state is
    /// replicated over a contact's rows).
    ncone: usize,
    nupdate: usize,
    /// The rank-one Cholesky updates actually performed by the last Hessian maintenance (the
    /// incremental ones and the cone ones, those of a rank-loss refactor included): unlike
    /// `nupdate` it is never overwritten by the refactor's row count.
    nrank1: usize,
    /// The rank-loss refactorisations of the solve (each one a full `J' D J` rebuild and
    /// Cholesky factorisation: `O(nefc nv^2 + nv^3)`).
    nrefactor: usize,
    lsiter: usize,
}

/// MuJoCo's cost of a friction-loss row at `x` (Huber): `PrimalEval`'s `frictionCost`.
fn friction_cost<R: Real>(x: R, f: R, rf: R, d: R) -> R {
    let half = lit::<R>(0.5);
    // -bound < x < bound: quadratic
    if -rf < x && x < rf {
        half * d * x * x
    }
    // x < -bound: linear negative
    else if x <= -rf {
        f * (-half * rf - x)
    }
    // bound < x: linear positive
    else {
        f * (-half * rf + x)
    }
}

/// `frictionCostDif`: `cost(x) - cost(start)` of a friction-loss row.
fn friction_cost_dif<R: Real>(start: R, x: R, f: R, rf: R, d: R) -> R {
    let zone = |v: R| -> i32 {
        if -rf < v && v < rf {
            0
        } else if v <= -rf {
            -1
        } else {
            1
        }
    };
    let (state_start, state_x) = (zone(start), zone(x));

    // both quadratic
    if state_start == 0 && state_x == 0 {
        return lit::<R>(0.5) * d * (x - start) * (x + start);
    }
    // both linear negative
    if state_start == -1 && state_x == -1 {
        return f * (start - x);
    }
    // both linear positive
    if state_start == 1 && state_x == 1 {
        return f * (x - start);
    }
    // otherwise different zones: compute absolute costs and subtract
    friction_cost(x, f, rf, d) - friction_cost(start, f, rf, d)
}

/// Port of `ellipticCostDif`: the cost of an elliptic cone at `alpha` relative to `alpha =
/// 0`. `quad` holds the contact's 9 numbers (`[q0, q1, q2, U0, V0, UU, UV, VV, Dm]`, see
/// `Primal::prepare`): the zones (1 top, 2 bottom, 3 middle) at `0` and at `alpha` are found
/// from `N = U0 + alpha V0` and `T^2 = UU + alpha (2 UV + alpha VV)`, and the cost difference is
/// formula by formula the one of MuJoCo (the middle zone in the rationalised form that avoids
/// cancellation).
fn elliptic_cost_dif<R: Real>(quad: &[R], alpha: R, mu: R, dm: R) -> R {
    let (u0, v0, uu) = (quad[3], quad[4], quad[5]);
    let (uv, vv) = (quad[6], quad[7]);
    let half = lit::<R>(0.5);
    let two = lit::<R>(2.0);

    // the zone and the cost at alpha = 0
    let zone0;
    let mut t0 = R::ZERO;
    if uu <= R::ZERO {
        zone0 = if u0 < R::ZERO { 2 } else { 1 };
    } else {
        t0 = uu.sqrt();
        if u0 >= mu * t0 {
            zone0 = 1; // the top zone
        } else if mu * u0 + t0 <= R::ZERO {
            zone0 = 2; // the bottom zone
        } else {
            zone0 = 3; // the middle zone
        }
    }

    // the zone at alpha
    let n = u0 + alpha * v0;
    let tsqr = uu + alpha * (two * uv + alpha * vv);
    let zone_alpha;
    let mut t = R::ZERO;
    if tsqr <= R::ZERO {
        zone_alpha = if n < R::ZERO { 2 } else { 1 }; // the bottom or the top zone
    } else {
        t = tsqr.sqrt();
        if n >= mu * t {
            zone_alpha = 1; // the top zone
        } else if mu * n + t <= R::ZERO {
            zone_alpha = 2; // the bottom zone
        } else {
            zone_alpha = 3; // the middle zone
        }
    }

    // both in the top zone
    if zone0 == 1 && zone_alpha == 1 {
        return R::ZERO;
    }

    // both in the bottom zone
    if zone0 == 2 && zone_alpha == 2 {
        return alpha * alpha * quad[2] + alpha * quad[1];
    }

    // both in the middle zone: the rationalised formula avoids cancellation
    if zone0 == 3 && zone_alpha == 3 {
        let tsqr_delta = alpha * (two * uv + alpha * vv);
        let t_delta = tsqr_delta / (t + t0);
        let r_delta = alpha * v0 - mu * t_delta;
        let r0 = u0 - mu * t0;
        return half * dm * r_delta * (two * r0 + r_delta);
    }

    // CONE -> QUADRATIC (3 -> 2)
    if zone0 == 3 && zone_alpha == 2 {
        let dq = alpha * (alpha * quad[2] + quad[1]);
        let boundary0 = mu * u0 + t0;
        let gap0 = half * dm * boundary0 * boundary0;
        return dq + gap0;
    }

    // QUADRATIC -> CONE (2 -> 3)
    if zone0 == 2 && zone_alpha == 3 {
        let dq = alpha * (alpha * quad[2] + quad[1]);
        let boundary = mu * n + t;
        let gap = half * dm * boundary * boundary;
        return dq - gap;
    }

    // SATISFIED -> QUADRATIC (1 -> 2)
    if zone0 == 1 && zone_alpha == 2 {
        return alpha * alpha * quad[2] + alpha * quad[1] + quad[0];
    }

    // SATISFIED -> CONE (1 -> 3)
    if zone0 == 1 && zone_alpha == 3 {
        let r = n - mu * t;
        return half * dm * r * r;
    }

    // CONE -> SATISFIED (3 -> 1)
    if zone0 == 3 && zone_alpha == 1 {
        let r0 = u0 - mu * t0;
        return -half * dm * r0 * r0;
    }

    // QUADRATIC -> SATISFIED (2 -> 1)
    if zone0 == 2 && zone_alpha == 1 {
        return -quad[0];
    }

    R::ZERO
}

impl<R: Real> Primal<'_, R> {
    /// `PrimalUpdateConstraint`: efc_force, qfrc_constraint, the cost (constraint plus
    /// Gauss), the active-row count and the cone-row count, from `Jaref`. With
    /// `cone_hessian` (Newton with an elliptic cone) the Hessian of every cone-zone contact
    /// is written to `contact_h`.
    fn update_constraint(&mut self, cone_hessian: bool) {
        let (nv, nefc) = (self.nv, self.nefc);

        // update the constraints
        let params = RowParams {
            nf: self.nf,
            nefc,
            d: self.efc_d,
            r: self.efc_r,
            frictionloss: self.efc_frictionloss,
            etype: self.efc_type,
            eid: self.efc_id,
            contact_mu: self.contact_mu,
            contact_friction: self.contact_friction,
            contact_dim: self.contact_dim,
        };
        let h = if cone_hessian {
            Some(&mut *self.contact_h)
        } else {
            None
        };
        let s = constraint_update_impl(&params, &self.ws.jaref, self.efc_state, self.efc_force, h);
        self.cost = s;

        // qfrc_constraint = J' force
        mul_mat_t_vec(self.qfrc_constraint, self.j, self.efc_force, nefc, nv);

        // count the active and the cone rows
        self.nactive = 0;
        self.ncone = 0;
        for i in 0..nefc {
            if self.efc_state[i] != ConstraintState::Satisfied {
                self.nactive += 1;
            }
            if self.efc_state[i] == ConstraintState::Cone {
                self.ncone += 1;
            }
        }

        // add the Gauss cost, set in quad_gauss[0]
        let half = lit::<R>(0.5);
        let mut gauss = R::ZERO;
        for i in 0..nv {
            gauss +=
                half * (self.ws.ma[i] - self.qfrc_smooth[i]) * (self.qacc[i] - self.qacc_smooth[i]);
        }
        self.quad_gauss[0] = gauss;
        self.cost += gauss;
    }

    /// `PrimalUpdateGrad`: `grad = M qacc - qfrc_smooth - qfrc_constraint`.
    fn update_grad(&mut self) {
        for i in 0..self.nv {
            self.ws.grad[i] = self.ws.ma[i] - self.qfrc_smooth[i] - self.qfrc_constraint[i];
        }
    }

    /// `PrimalUpdateMgrad`: Newton, `Mgrad = H \ grad`; CG, `Mgrad = M \ grad`.
    fn update_mgrad(&mut self, newton: bool) {
        let nv = self.nv;
        if newton {
            // with an active cone the factor is the one that includes the cones' Hessians
            let l = if self.ncone > 0 {
                &self.ws.lcone
            } else {
                &self.ws.l
            };
            chol_solve(&mut self.ws.mgrad, l, &self.ws.grad, nv);
        } else {
            self.ws.mgrad[..nv].copy_from_slice(&self.ws.grad[..nv]);
            factor::solve(
                &mut self.ws.mgrad,
                self.qld,
                self.qld_diag_inv,
                nv,
                self.sparsity,
            );
        }
    }

    /// `PrimalUpdateGradient`.
    fn update_gradient(&mut self, newton: bool) {
        self.update_grad();
        self.update_mgrad(newton);
    }

    /// `PrimalPrepare`: the quadratic polynomial of every row's cost, and of the Gauss
    /// cost, along `search`.
    fn prepare(&mut self) {
        let nefc = self.nefc;
        let half = lit::<R>(0.5);

        // Gauss: alpha^2 0.5 v'Mv + alpha v'(Ma - qfrc_smooth) + the constant
        self.quad_gauss[1] =
            dot(&self.ws.search, &self.ws.ma) - dot(self.qfrc_smooth, &self.ws.search);
        self.quad_gauss[2] = half * dot(&self.ws.search, &self.ws.mv);

        // process the constraints
        let mut i = 0usize;
        while i < nefc {
            let jaref = self.ws.jaref[i];
            let jv = self.ws.jv[i];
            let d = self.efc_d[i];

            // init with the scalar quadratic
            let dj0 = d * jaref;
            let mut q0 = jaref * dj0;
            let mut q1 = jv * dj0;
            let mut q2 = jv * d * jv;
            let first = i;

            // an elliptic cone: extra processing
            if self.efc_type[i] == ConstraintType::ContactElliptic {
                // the contact info
                let id = self.efc_id[i];
                let dim = self.contact_dim[id];
                let mu = self.contact_mu[id];
                let friction = &self.contact_friction[5 * id..5 * id + 5];
                // the 9 values below live in the quad slots of rows i..i+2, which exist
                // because an elliptic contact has dim >= 3 (sim-scene's condim rule)
                debug_assert!(dim >= 3);
                let mut u = [R::ZERO; 6];
                let mut v = [R::ZERO; 6];
                let (mut uu, mut uv, mut vv) = (R::ZERO, R::ZERO, R::ZERO);

                // complete the vector quadratic (for the bottom zone)
                for j in 1..dim {
                    let djj = self.efc_d[i + j] * self.ws.jaref[i + j];
                    q0 += self.ws.jaref[i + j] * djj;
                    q1 += self.ws.jv[i + j] * djj;
                    q2 += self.ws.jv[i + j] * self.efc_d[i + j] * self.ws.jv[i + j];
                }

                // rescale to make the primal cone circular
                u[0] = jaref * mu;
                v[0] = jv * mu;
                for j in 1..dim {
                    u[j] = self.ws.jaref[i + j] * friction[j - 1];
                    v[j] = self.ws.jv[i + j] * friction[j - 1];
                }

                // accumulate the sums of squares
                for j in 1..dim {
                    uu += u[j] * u[j];
                    uv += u[j] * v[j];
                    vv += v[j] * v[j];
                }

                // store in quad[3..9], using the fact that dim >= 3
                let quad = &mut self.ws.quad[3 * first..3 * first + 9];
                quad[3] = u[0];
                quad[4] = v[0];
                quad[5] = uu;
                quad[6] = uv;
                quad[7] = vv;
                quad[8] = self.efc_d[i] / ((mu * mu) * (R::ONE + (mu * mu)));

                // advance to the next constraint
                i += dim - 1;
            }

            // apply the scaling
            self.ws.quad[3 * first] = q0 * half;
            self.ws.quad[3 * first + 1] = q1;
            self.ws.quad[3 * first + 2] = q2 * half;
            i += 1;
        }
    }

    /// `PrimalEval`: the cost relative to alpha = 0 and its derivatives at `p.alpha`.
    fn eval(&mut self, p: &mut Pnt<R>) {
        let nf = self.nf;
        let nefc = self.nefc;

        // clear the result
        let mut cost = R::ZERO;
        let alpha = p.alpha;
        let two = lit::<R>(2.0);
        let mut deriv = [R::ZERO, R::ZERO];

        // init the quadratic with the Gauss part, shifted: drop quad_gauss[0]
        let mut quad_total = [R::ZERO, self.quad_gauss[1], self.quad_gauss[2]];

        // process the constraints
        let mut i = 0usize;
        while i < nefc {
            // friction: compute cost(alpha) - cost(0) directly
            if i < nf {
                // search point, friction loss, bound (Rf)
                let start = self.ws.jaref[i];
                let dir = self.ws.jv[i];
                let x = start + alpha * dir;
                let f = self.efc_frictionloss[i];
                let d = self.efc_d[i];
                let rf = self.efc_r[i] * f;

                // cost delta
                cost += friction_cost_dif(start, x, f, rf, d);

                // -bound < x < bound: quadratic
                if -rf < x && x < rf {
                    deriv[0] += d * x * dir;
                    deriv[1] += d * dir * dir;
                }
                // x < -bound: linear negative
                else if x <= -rf {
                    deriv[0] += -f * dir;
                }
                // bound < x: linear positive
                else {
                    deriv[0] += f * dir;
                }
                i += 1;
                continue;
            }

            // limit and contact
            if self.efc_type[i] == ConstraintType::ContactElliptic {
                // an elliptic cone: the contact info
                let id = self.efc_id[i];
                let dim = self.contact_dim[id];
                let mu = self.contact_mu[id];
                let quad = &self.ws.quad[3 * i..3 * i + 9];

                // unpack the quad
                let (u0, v0, uu, uv, vv, dm) =
                    (quad[3], quad[4], quad[5], quad[6], quad[7], quad[8]);

                // the shifted cost
                cost += elliptic_cost_dif(quad, alpha, mu, dm);

                // N and Tsqr for the derivatives
                let n = u0 + alpha * v0;
                let tsqr = uu + alpha * (two * uv + alpha * vv);

                // no tangential force: the top or the bottom zone
                if tsqr <= R::ZERO {
                    // the bottom zone: quadratic derivatives
                    if n < R::ZERO {
                        deriv[0] += two * alpha * quad[2] + quad[1];
                        deriv[1] += two * quad[2];
                    }
                    // the top zone: nothing to do
                }
                // otherwise regular processing
                else {
                    // the tangential force
                    let t = tsqr.sqrt();

                    // N >= mu T: the top zone
                    if n >= mu * t {
                        // nothing to do
                    }
                    // mu N + T <= 0: the bottom zone
                    else if mu * n + t <= R::ZERO {
                        deriv[0] += two * alpha * quad[2] + quad[1];
                        deriv[1] += two * quad[2];
                    }
                    // otherwise the middle zone
                    else {
                        // derivatives
                        let n1 = v0;
                        let t1 = (uv + alpha * vv) / t;
                        let t2 = vv / t - (uv + alpha * vv) * t1 / (t * t);
                        deriv[0] += dm * (n - mu * t) * (n1 - mu * t1);
                        deriv[1] +=
                            dm * ((n1 - mu * t1) * (n1 - mu * t1) + (n - mu * t) * (-mu * t2));
                    }
                }

                // advance to the next constraint
                i += dim;
                continue;
            }

            // an inequality
            let start = self.ws.jaref[i];
            let x = start + alpha * self.ws.jv[i];
            let cost0 = if start < R::ZERO {
                self.ws.quad[3 * i]
            } else {
                R::ZERO
            };

            // active
            if x < R::ZERO {
                // shifted quad: add quad[1], quad[2] and (quad[0] - cost0)
                quad_total[0] += self.ws.quad[3 * i] - cost0;
                quad_total[1] += self.ws.quad[3 * i + 1];
                quad_total[2] += self.ws.quad[3 * i + 2];
            } else {
                cost -= cost0;
            }
            i += 1;
        }

        // add the total quadratic (quad_total[0] contains only shifted residuals)
        cost += alpha * alpha * quad_total[2] + alpha * quad_total[1] + quad_total[0];
        deriv[0] += two * alpha * quad_total[2] + quad_total[1];
        deriv[1] += two * quad_total[2];

        // check for convexity; SHOULD NOT OCCUR (MuJoCo also warns)
        if deriv[1] <= R::ZERO {
            deriv[1] = min_val::<R>();
        }

        // assign and count
        p.cost = cost;
        p.deriv = deriv;
        self.lsiter += 1;
    }

    /// `updateBracket`: updates the bracket point `p` from three candidates and, if it
    /// moved, evaluates the next Newton point `pnext`.
    fn update_bracket(&mut self, p: &mut Pnt<R>, cands: &[Pnt<R>; 3], pnext: &mut Pnt<R>) -> i32 {
        let mut flag = 0;
        for c in cands {
            // negative deriv
            if p.deriv[0] < R::ZERO && c.deriv[0] < R::ZERO && p.deriv[0] < c.deriv[0] {
                *p = *c;
                flag = 1;
            }
            // positive deriv
            else if p.deriv[0] > R::ZERO && c.deriv[0] > R::ZERO && p.deriv[0] > c.deriv[0] {
                *p = *c;
                flag = 2;
            }
        }

        // compute the next point if updated
        if flag != 0 {
            pnext.alpha = p.alpha - p.deriv[0] / p.deriv[1];
            self.eval(pnext);
        }
        flag
    }

    /// `PrimalSearch`: the exact line search along `search`. Returns the step (0 for no
    /// improvement) and writes the cost improvement.
    fn search(&mut self, tolerance: R, ls_iterations: usize, improvement: &mut R) -> R {
        let (nv, nefc) = (self.nv, self.nefc);

        // clear the results
        self.lsiter = 0;
        *improvement = R::ZERO;

        // save the search vector length, check
        let snorm = norm(&self.ws.search[..nv]);
        if snorm < min_val::<R>() {
            return R::ZERO; // search vector too small
        }

        // scaled gradtol and slope scaling
        let gtol = tolerance * snorm / self.scale;

        // Mv = M search
        mul_sym_vec_sparse(&mut self.ws.mv, self.qm, &self.ws.search, nv, self.sparsity);

        // Jv = J search
        mul_mat_vec(&mut self.ws.jv, self.j, &self.ws.search, nefc, nv);

        // prepare the quadratics
        self.prepare();

        // the test-only fault: a unit step instead of the line search
        if self.unit_step {
            let mut p = Pnt::at(R::ONE);
            self.eval(&mut p);
            *improvement = -p.cost;
            return R::ONE;
        }

        // init at alpha = 0, save
        let mut p0 = Pnt::at(R::ZERO);
        self.eval(&mut p0);

        // always attempt one Newton step
        let mut p1 = Pnt::at(p0.alpha - p0.deriv[0] / p0.deriv[1]);
        self.eval(&mut p1);

        // check for initial convergence
        if p1.deriv[0].abs() < gtol && (p1.alpha == R::ZERO || p1.cost < R::ZERO) {
            *improvement = -p1.cost;
            return p1.alpha;
        }

        // save the direction
        let dir = if p1.deriv[0] < R::ZERO {
            R::ONE
        } else {
            -R::ONE
        };

        // one-sided search
        let mut p2 = p0;
        let mut p2update = true;
        while p1.deriv[0] * dir <= -gtol && self.lsiter < ls_iterations {
            // save the current point
            p2 = p1;
            p2update = true;

            // move to the Newton point with respect to the current one
            p1.alpha -= p1.deriv[0] / p1.deriv[1];
            self.eval(&mut p1);

            // check for convergence
            if p1.deriv[0].abs() < gtol && p1.cost < R::ZERO {
                *improvement = -p1.cost;
                return p1.alpha; // SUCCESS
            }
        }

        // check for failure to bracket
        if self.lsiter >= ls_iterations {
            *improvement = -p1.cost;
            return p1.alpha;
        }

        // check for a p2 update; SHOULD NOT OCCUR
        if !p2update {
            *improvement = -p1.cost;
            return p1.alpha;
        }

        // compute the next points for the bracket
        let mut p2next = p1;
        let mut p1next = Pnt::at(p1.alpha - p1.deriv[0] / p1.deriv[1]);
        self.eval(&mut p1next);

        // bracketed search
        while self.lsiter < ls_iterations {
            // evaluate at the midpoint
            let mut pmid = Pnt::at(lit::<R>(0.5) * (p1.alpha + p2.alpha));
            self.eval(&mut pmid);

            // make the list of candidates
            let cands = [p1next, p2next, pmid];

            // check the candidates for convergence
            let mut bestcost = R::ZERO;
            let mut bestind: Option<usize> = None;
            for (i, c) in cands.iter().enumerate() {
                if c.deriv[0].abs() < gtol && (bestind.is_none() || c.cost < bestcost) {
                    bestcost = c.cost;
                    bestind = Some(i);
                }
            }
            if let Some(b) = bestind {
                *improvement = -cands[b].cost;
                return cands[b].alpha; // SUCCESS
            }

            // update the brackets
            let b1 = self.update_bracket(&mut p1, &cands, &mut p1next);
            let b2 = self.update_bracket(&mut p2, &cands, &mut p2next);

            // no update possible: numerical accuracy reached, use the midpoint
            if b1 == 0 && b2 == 0 {
                *improvement = -pmid.cost;
                return pmid.alpha;
            }
        }

        // choose the bracket with the best cost
        if p1.cost <= p2.cost && p1.cost < R::ZERO {
            *improvement = -p1.cost;
            p1.alpha // improvement but no convergence
        } else if p2.cost <= p1.cost && p2.cost < R::ZERO {
            *improvement = -p2.cost;
            p2.alpha
        } else {
            R::ZERO // no improvement
        }
    }

    /// `MakeHessian` (dense): `D` of the quadratic rows and `H = M + J' D J`.
    fn make_hessian(&mut self) {
        let (nv, nefc) = (self.nv, self.nefc);

        // compute the constraint inertia
        for i in 0..nefc {
            self.ws.d[i] = if self.efc_state[i] == ConstraintState::Quadratic {
                self.efc_d[i]
            } else {
                R::ZERO
            };
        }

        // H = J' D J (lower triangle), then + M
        sqr_mat_td_lower(&mut self.ws.l, self.j, &self.ws.d, nefc, nv);
        add_sym_sparse(&mut self.ws.l, self.qm, nv, self.sparsity);
    }

    /// `FactorizeHessian` (dense): maybe recompute `H` from the states, then
    /// `L = chol(H)`.
    fn factorize_hessian(&mut self, recompute: bool) {
        let nv = self.nv;
        if recompute {
            self.make_hessian();
        }
        chol_factor(&mut self.ws.l, nv, min_val::<R>());

        // add the cones to the factor if present
        if self.ncone > 0 {
            self.hessian_cone();
        }

        // mark the full update
        self.nupdate = self.nefc;
    }

    /// `HessianCone` (dense, `HessianConeUpdate`): `Lcone = L`, then for every contact in
    /// the cone zone `dim` rank-one updates of the factor with the rows of `Lc' J`, `Lc` being
    /// the Cholesky factor of the contact's cone Hessian (`contact_h`). (`drop_cone_hessian`
    /// is the test-only fault that stops after the copy.)
    fn hessian_cone(&mut self) {
        let (nv, nefc) = (self.nv, self.nefc);

        // start with Hcone = H
        self.ws.lcone[..nv * nv].copy_from_slice(&self.ws.l[..nv * nv]);
        if self.drop_cone_hessian {
            return;
        }

        // add the contributions
        let mut i = 0usize;
        while i < nefc {
            if self.efc_state[i] != ConstraintState::Cone {
                i += 1;
                continue;
            }
            let id = self.efc_id[i];
            let dim = self.contact_dim[id];

            // the Cholesky factor of the local Hessian
            let mut local = [R::ZERO; 36];
            local[..dim * dim].copy_from_slice(&self.contact_h[36 * id..36 * id + dim * dim]);
            chol_factor(&mut local, dim, min_val::<R>());

            // LTJ = L' J for this contact's rows
            self.ws.ltj[..dim * nv].fill(R::ZERO);
            for r in 0..dim {
                for c in 0..=r {
                    // mju_addToScl(LTJ + c * nv, J + (i + r) * nv, local[r * dim + c], nv)
                    let scl = local[r * dim + c];
                    for k in 0..nv {
                        self.ws.ltj[c * nv + k] += self.j[(i + r) * nv + k] * scl;
                    }
                }
            }

            // update
            for r in 0..dim {
                chol_update(
                    &mut self.ws.lcone,
                    &mut self.ws.ltj[r * nv..(r + 1) * nv],
                    nv,
                    true,
                );
            }

            // count the updates
            self.nupdate += dim;
            self.nrank1 += dim;

            // advance to the next constraint
            i += dim;
        }
    }

    /// `HessianIncremental` (dense): updates the factor for the rows whose state moved
    /// into or out of the quadratic zone since `ws.oldstate`.
    fn hessian_incremental(&mut self) {
        let (nv, nefc) = (self.nv, self.nefc);

        // clear the update counters
        self.nupdate = 0;
        self.nrank1 = 0;

        // update the H factorisation
        for i in 0..nefc {
            let old = self.ws.oldstate[i];
            let new = self.efc_state[i];
            let flag_update =
                if old != ConstraintState::Quadratic && new == ConstraintState::Quadratic {
                    Some(true) // add quad
                } else if old == ConstraintState::Quadratic && new != ConstraintState::Quadratic {
                    Some(false) // subtract quad
                } else {
                    None
                };

            // perform the update if flagged: cholupd = J(i,:) sqrt(D[i])
            if let Some(plus) = flag_update {
                let scl = self.efc_d[i].sqrt();
                for k in 0..nv {
                    self.ws.cholupd[k] = self.j[i * nv + k] * scl;
                }
                let rank = chol_update(&mut self.ws.l, &mut self.ws.cholupd, nv, plus);
                self.nupdate += 1;
                self.nrank1 += 1;

                // recompute H directly if accuracy was lost
                if rank < nv {
                    self.nrefactor += 1;
                    self.factorize_hessian(true);
                    return;
                }
            }
        }

        // add the cones if present
        if self.ncone > 0 {
            self.hessian_cone();
        }
    }
}

/// Port of `mj_solPrimal` (`mj_solNewton` or `mj_solCG` by the model's solver):
/// improves `d.qacc` (set by the warmstart) to the minimum of the primal cost, and
/// writes `efc_force`, `efc_state`, `qfrc_constraint` and adds the iteration count
/// to `d.solver_niter`.
pub(crate) fn solve_primal<R: Real>(m: &Model<R>, d: &mut Data<R>, faults: &Faults) {
    let newton = m.opt.solver == PrimalSolver::Newton;
    let nv = m.nv;
    let maxiter = m.opt.iterations;
    let tolerance = m.opt.tolerance;

    let Data {
        nf,
        nefc,
        qm,
        qld,
        qld_diag_inv,
        qfrc_smooth,
        qacc_smooth,
        efc_j,
        efc_d,
        efc_r,
        efc_frictionloss,
        efc_aref,
        efc_type,
        efc_id,
        contact_mu,
        contact_friction,
        contact_dim,
        contact_h,
        qfrc_constraint,
        qacc,
        efc_force,
        efc_state,
        solver_niter,
        solver_stat,
        ws,
        ..
    } = d;
    let nefc = *nefc;
    let elliptic = m.opt.cone == Cone::Elliptic;
    let mut ctx = Primal {
        nv,
        nf: *nf,
        nefc,
        scale: R::ONE,
        tolerance,
        unit_step: faults.unit_step_line_search,
        qm,
        sparsity: &m.qm_sparsity,
        qld,
        qld_diag_inv,
        qfrc_smooth,
        qacc_smooth,
        j: efc_j,
        efc_d,
        efc_r,
        efc_frictionloss,
        efc_aref,
        efc_type,
        efc_id,
        contact_mu,
        contact_friction,
        contact_dim,
        qfrc_constraint,
        qacc,
        efc_force,
        efc_state,
        contact_h,
        ws,
        drop_cone_hessian: faults.drop_cone_hessian,
        cost: R::ZERO,
        quad_gauss: [R::ZERO; 3],
        nactive: 0,
        ncone: 0,
        nupdate: 0,
        nrank1: 0,
        nrefactor: 0,
        lsiter: 0,
    };

    // Ma = M qacc
    mul_sym_vec_sparse(&mut ctx.ws.ma, ctx.qm, ctx.qacc, nv, ctx.sparsity);

    // Jaref = J qacc - aref
    mul_mat_vec(&mut ctx.ws.jaref, ctx.j, ctx.qacc, nefc, nv);
    for i in 0..nefc {
        ctx.ws.jaref[i] -= ctx.efc_aref[i];
    }

    // first update
    ctx.update_constraint(newton && elliptic);
    ctx.update_grad();

    // compute and save the scaling factor
    let meaninertia_times_nv = m.meaninertia * lit::<R>(nv.max(1) as f64);
    ctx.scale = R::ONE / meaninertia_times_nv;
    let scale = ctx.scale;
    let half = lit::<R>(0.5);

    // Mgrad = M \ grad: the CG preconditioned gradient, also the convergence certificate
    ctx.update_mgrad(false);

    // convergence certificate: the cost is strongly convex in the M-norm, bounding the
    // suboptimality by the duality gap at the current constraint forces; if already
    // below tolerance (a good warmstart), skip the Hessian and the main loop
    let flg_gap = max(
        R::ZERO,
        half * scale * dot(&ctx.ws.grad[..nv], &ctx.ws.mgrad[..nv]),
    ) < tolerance;

    // the gap bounds the cost suboptimality; Newton solutions are force-accurate, so
    // its zero-iteration exit also requires the gradient criterion; CG exits on the
    // gap alone
    let flg_gradient = scale * norm(&ctx.ws.grad[..nv]) < tolerance;
    let flg_certificate = flg_gap && (!newton || flg_gradient);
    let mut flg_done = flg_certificate;

    // Newton: compute and factorise the Hessian, Mgrad = H \ grad
    if !flg_done && newton {
        ctx.make_hessian();
        ctx.factorize_hessian(false);
        ctx.update_mgrad(true);

        // the Newton decrement already below tolerance: converged, skip the first line
        // search (gradient-gated like the certificate)
        flg_done = flg_gradient
            && max(
                R::ZERO,
                half * scale * dot(&ctx.ws.grad[..nv], &ctx.ws.mgrad[..nv]),
            ) < tolerance;
    }

    // start both with the preconditioned gradient
    if !flg_done {
        for i in 0..nv {
            ctx.ws.search[i] = ctx.ws.mgrad[i] * -R::ONE;
        }
    }

    // main loop
    let mut iter = 0usize;
    let mut stat = SolverStat {
        cost: ctx.cost,
        nactive: ctx.nactive,
        nupdate_max: ctx.nupdate,
        nrank1_max: ctx.nrank1,
        ..SolverStat::default()
    };
    while !flg_done && iter < maxiter {
        // perform the line search
        let mut ls_improvement = R::ZERO;
        let ls_tolerance = ctx.tolerance * m.opt.ls_tolerance;
        let alpha = ctx.search(ls_tolerance, m.opt.ls_iterations, &mut ls_improvement);
        stat.neval_max = stat.neval_max.max(ctx.lsiter);

        // no improvement: done
        if alpha == R::ZERO {
            break;
        }

        // move to the new solution
        for i in 0..nv {
            ctx.qacc[i] += ctx.ws.search[i] * alpha;
        }
        for i in 0..nv {
            ctx.ws.ma[i] += ctx.ws.mv[i] * alpha;
        }
        for i in 0..nefc {
            ctx.ws.jaref[i] += ctx.ws.jv[i] * alpha;
        }

        // save the old
        if !newton {
            ctx.ws.gradold[..nv].copy_from_slice(&ctx.ws.grad[..nv]);
            ctx.ws.mgradold[..nv].copy_from_slice(&ctx.ws.mgrad[..nv]);
        }
        ctx.ws.oldstate[..nefc].copy_from_slice(&ctx.efc_state[..nefc]);

        // update
        ctx.update_constraint(newton && elliptic);
        if newton {
            ctx.hessian_incremental();
        }
        ctx.update_gradient(newton);

        // count the state changes
        let mut nchange = 0usize;
        for i in 0..nefc {
            if ctx.efc_state[i] != ctx.ws.oldstate[i] {
                nchange += 1;
            }
        }

        // scale the improvement and the gradient, save the statistics
        let improvement = scale * ls_improvement;
        let gradient = scale * norm(&ctx.ws.grad[..nv]);
        stat = SolverStat {
            cost: ctx.cost,
            improvement,
            gradient,
            nactive: ctx.nactive,
            nchange,
            neval: ctx.lsiter,
            nupdate: ctx.nupdate,
            neval_max: stat.neval_max,
            nupdate_max: stat.nupdate_max.max(ctx.nupdate),
            nrank1_max: stat.nrank1_max.max(ctx.nrank1),
            nrefactor: ctx.nrefactor,
        };

        // the Newton decrement: 0.5 grad' H^-1 grad, the model's predicted improvement
        // of the next step; clamped to 0 so that tolerance == 0 keeps early termination
        // disabled
        let decrement = if newton {
            max(
                R::ZERO,
                half * scale * dot(&ctx.ws.grad[..nv], &ctx.ws.mgrad[..nv]),
            )
        } else {
            R::ZERO
        };

        // increment the iteration count
        iter += 1;

        // termination
        if (improvement > R::ZERO && improvement < tolerance)
            || gradient < tolerance
            || (newton && decrement < tolerance)
        {
            break;
        }

        // update the direction
        if newton {
            for i in 0..nv {
                ctx.ws.search[i] = ctx.ws.mgrad[i] * -R::ONE;
            }
        } else {
            // Hager-Zhang conjugate direction update
            let eta = lit::<R>(0.01);
            let ws = &mut *ctx.ws;

            // graddif = grad - gradold, Mgraddif = Mgrad - Mgradold
            for i in 0..nv {
                ws.graddif[i] = ws.grad[i] - ws.gradold[i];
                ws.mgraddif[i] = ws.mgrad[i] - ws.mgradold[i];
            }

            // d'y; restart to steepest descent if conjugacy is lost
            let d_dot_y = dot(&ws.search[..nv], &ws.graddif[..nv]);
            let beta = if d_dot_y < min_val::<R>() {
                R::ZERO
            } else {
                // the remaining inner products of the HZ formula
                let y_dot_my = dot(&ws.graddif[..nv], &ws.mgraddif[..nv]);
                let y_dot_mgrad = dot(&ws.graddif[..nv], &ws.mgrad[..nv]);
                let d_dot_grad = dot(&ws.search[..nv], &ws.grad[..nv]);

                // the primary Hager-Zhang beta coefficient
                let beta_hz =
                    (y_dot_mgrad - lit::<R>(2.0) * (y_dot_my / d_dot_y) * d_dot_grad) / d_dot_y;

                // the dynamic truncation threshold, so that d is not orthogonal to grad
                let d_norm = norm(&ws.search[..nv]);
                let grad_norm = norm(&ws.grad[..nv]);
                let eta_k = -R::ONE / max(min_val::<R>(), d_norm * min(eta, grad_norm));

                // apply the lower bound
                max(eta_k, beta_hz)
            };

            // update
            for i in 0..nv {
                ws.search[i] = -ws.mgrad[i] + beta * ws.search[i];
            }
        }
    }

    // the iteration count and the statistics of this solve
    stat.cost = ctx.cost;
    *solver_niter += iter;
    *solver_stat = stat;
}
