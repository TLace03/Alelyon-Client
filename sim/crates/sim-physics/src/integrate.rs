//! Time integration: position update on the joint manifolds, the Euler step with
//! implicit joint damping, the classical Runge-Kutta step, and the full `step`.
//!
//! Ports, from MuJoCo 3.14.0: `mj_integratePos` (engine_support.c),
//! `mju_quatIntegrate` (engine_util_spatial.c), and from engine_forward.c
//! `mj_EulerSkip`, `mj_Euler`, `mj_RungeKutta`, `mj_advance` and `mj_step`.
//!
//! Invariants:
//! - **Euler** (`mj_Euler`): `qvel += h * qacc`, then `qpos` is integrated with
//!   the NEW `qvel` (semi-implicit). When any dof has damping above zero, `qacc`
//!   is replaced by the solution of `(M + h diag(damping)) qacc = qfrc_smooth +
//!   qfrc_constraint` (joint damping integrated implicitly; `qfrc_smooth` already
//!   contains the explicit damper force, and `qfrc_constraint` is the constraint
//!   force the solver found at the explicit state, as in `mj_EulerSkip`).
//! - **RK4** (`mj_RungeKutta`, `N = 4`): four stages evaluated by a full
//!   [`crate::forward`] (constraints included) at stage states built from the
//!   initial `qpos` with `mj_integratePos` and the stage velocities, then one update
//!   with the weights `[1/6, 1/3, 1/3, 1/6]`: `qvel += h * sum(B_j F_j)`, `qpos`
//!   integrated from its initial value with the velocity `sum(B_j V_j)`, where `V_j`
//!   is the velocity part of stage `j`. Damping is explicit in RK4. The stages all
//!   warm-start the solver from the `qacc_warmstart` of the previous step.
//! - **Warmstart** (`mj_advance`): after the update `qacc_warmstart` is set to the
//!   `qacc` of the forward pass (for RK4 the one of the last stage), whatever
//!   acceleration the integrator used.
//! - `mj_discreteGyro` (the last call of `fwdConstraint`) returns at once unless
//!   `integrator == discrete`, so Euler and RK4 never run its body; it is not ported.
//! - **Free joint** (`mj_integratePos`): the position advances with the world
//!   frame linear velocity `qvel[0..3]`, the quaternion with the body frame
//!   angular velocity `qvel[3..6]` (`quat <- quat * exp(h w / 2)`). A ball joint's
//!   `qvel` is a body frame angular velocity. A hinge or slide is `q += h v`.
//! - `time` advances by `h` once per step (`mj_advance`), after the update.
//! - A step allocates nothing: the RK4 stages live in [`Data`].
//! - Plain multiply and add only: no fused multiply-add, in the operation order of
//!   the C source (see [`crate::Real`]). `qvel += qacc * h` and `qpos += dt * qvel`
//!   are a product and then a sum, as in MuJoCo.
//! - No divergence guard: MuJoCo's `mj_step` resets the data when `qpos`, `qvel`
//!   or `qacc` is not finite or above 1e10 (`mj_checkPos`, `mj_checkVel`,
//!   `mj_checkAcc`); this port does not, and a diverged state stays diverged.

use crate::data::Data;
use crate::math::quat_integrate;
use crate::model::{JointType, Model};
use crate::real::Real;
use crate::smooth::{self, Faults};

/// MuJoCo's RK4 tableau `A` (3 x 3, row-major, `mj_RungeKutta`): `C = row sums`.
const RK4_A: [f64; 9] = [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0];
/// MuJoCo's RK4 weights `B`.
const RK4_B: [f64; 4] = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0];

/// Port of `mj_integratePos`: advances `qpos` by `dt` with the velocity `qvel`
/// on each joint's manifold (quaternions through `mju_quatIntegrate`).
pub fn integrate_pos<R: Real>(m: &Model<R>, qpos: &mut [R], qvel: &[R], dt: R) {
    for j in 0..m.njnt {
        let mut padr = m.jnt_qposadr[j];
        let mut vadr = m.jnt_dofadr[j];
        match m.jnt_type[j] {
            JointType::Free | JointType::Ball => {
                if m.jnt_type[j] == JointType::Free {
                    // position update
                    for i in 0..3 {
                        qpos[padr + i] += dt * qvel[vadr + i];
                    }
                    padr += 3;
                    vadr += 3;
                }
                // quaternion update
                let q = [qpos[padr], qpos[padr + 1], qpos[padr + 2], qpos[padr + 3]];
                let vel = [qvel[vadr], qvel[vadr + 1], qvel[vadr + 2]];
                let r = quat_integrate(q, vel, dt);
                qpos[padr..padr + 4].copy_from_slice(&r);
            }
            JointType::Hinge | JointType::Slide => {
                // scalar update: the same for rotation and translation
                qpos[padr] += dt * qvel[vadr];
            }
        }
    }
}

/// Port of `mj_Euler` / `mj_EulerSkip` (with the factorisation computed): the
/// semi-implicit Euler step from the `qacc`, `qfrc_smooth` and `qm` that
/// [`crate::forward`] just computed. See the module note.
pub fn euler<R: Real>(m: &Model<R>, d: &mut Data<R>) {
    euler_with(m, d, &Faults::NONE);
}

pub(crate) fn euler_with<R: Real>(m: &Model<R>, d: &mut Data<R>, faults: &Faults) {
    let nv = m.nv;
    let h = m.timestep;
    if !m.has_damping || faults.euler_without_implicit_damping {
        // no damping (or disabled): explicit velocity integration
        d.qacc_step.copy_from_slice(&d.qacc);
    } else {
        // damping: integrate implicitly. qH = M + h * diag(damping)
        d.qh.copy_from_slice(&d.qm);
        for i in 0..nv {
            d.qh[i * nv + i] += h * m.dof_damping[i];
        }
        // factorise in place (a clamped pivot is not reported, as in MuJoCo's warning)
        crate::factor::factor(&mut d.qh, &mut d.qh_diag_inv, nv);
        // solve for qacc = qH^-1 (qfrc_smooth + qfrc_constraint)
        for i in 0..nv {
            d.qacc_step[i] = d.qfrc_smooth[i] + d.qfrc_constraint[i];
        }
        crate::factor::solve(&mut d.qacc_step, &d.qh, &d.qh_diag_inv, nv, &m.qm_sparsity);
    }

    // advance the state and time (mj_advance): velocity first, then the position
    // with the new velocity
    for i in 0..nv {
        d.qvel[i] += d.qacc_step[i] * h;
    }
    integrate_pos(m, &mut d.qpos, &d.qvel, h);
    d.time += h;

    // save qacc for the next step's warmstart
    d.qacc_warmstart.copy_from_slice(&d.qacc);
}

/// Port of `mj_RungeKutta` with `N = 4`: the classical Runge-Kutta step, from the
/// `qacc` that [`crate::forward`] just computed at the current state.
pub fn rk4<R: Real>(m: &Model<R>, d: &mut Data<R>) {
    rk4_with(m, d, &Faults::NONE);
}

pub(crate) fn rk4_with<R: Real>(m: &Model<R>, d: &mut Data<R>, faults: &Faults) {
    const N: usize = 4;
    let (nq, nv) = (m.nq, m.nv);
    let stride = nq + nv;
    let h = m.timestep;
    let time = d.time;
    let a = |i: usize| R::from_f64(RK4_A[i]);
    let b = |j: usize| R::from_f64(RK4_B[j]);

    // C(i) = sum_j A(i, j); T(i) = time + C(i) h   (C, T have size N - 1)
    let mut t = [R::ZERO; N - 1];
    for i in 1..N {
        let mut c = R::ZERO;
        for j in 0..i {
            c += a((i - 1) * (N - 1) + j);
        }
        t[i - 1] = time + c * h;
    }

    // init X[0], F[0]; forward() was already called
    d.rk_x[0..nq].copy_from_slice(&d.qpos);
    d.rk_x[nq..stride].copy_from_slice(&d.qvel);
    d.rk_f[0..nv].copy_from_slice(&d.qacc);

    // compute the remaining X[i], F[i]
    for i in 1..N {
        // dX = sum_j A(i, j) [V_j, F_j]
        d.rk_dx.fill(R::ZERO);
        for j in 0..i {
            let aij = a((i - 1) * (N - 1) + j);
            for k in 0..nv {
                d.rk_dx[k] += d.rk_x[j * stride + nq + k] * aij;
            }
            for k in 0..nv {
                d.rk_dx[nv + k] += d.rk_f[j * nv + k] * aij;
            }
        }

        // X[i] = X[0] '+' dX: positions on the manifold, velocities by addition
        let (head, tail) = d.rk_x.split_at_mut(i * stride);
        tail[..stride].copy_from_slice(&head[..stride]);
        integrate_pos(m, &mut tail[..nq], &d.rk_dx[..nv], h);
        for k in 0..nv {
            tail[nq + k] += d.rk_dx[nv + k] * h;
        }

        // set X[i], T[i - 1] in the data
        d.qpos.copy_from_slice(&d.rk_x[i * stride..i * stride + nq]);
        d.qvel
            .copy_from_slice(&d.rk_x[i * stride + nq..(i + 1) * stride]);
        d.time = t[i - 1];

        // evaluate F[i]
        smooth::forward_with(m, d, faults);
        d.rk_f[i * nv..(i + 1) * nv].copy_from_slice(&d.qacc);
    }

    // dX for the final update, with B instead of A
    d.rk_dx.fill(R::ZERO);
    for j in 0..N {
        let bj = b(j);
        for k in 0..nv {
            d.rk_dx[k] += d.rk_x[j * stride + nq + k] * bj;
        }
        for k in 0..nv {
            d.rk_dx[nv + k] += d.rk_f[j * nv + k] * bj;
        }
    }

    // reset the state and time
    d.time = time;
    d.qpos.copy_from_slice(&d.rk_x[0..nq]);
    d.qvel.copy_from_slice(&d.rk_x[nq..stride]);

    // advance state and time (mj_advance(act_dot, qacc = dX + nv, qvel = dX))
    for k in 0..nv {
        d.qvel[k] += d.rk_dx[nv + k] * h;
    }
    integrate_pos(m, &mut d.qpos, &d.rk_dx[..nv], h);
    d.time += h;

    // save qacc (of the last stage's forward pass) for the next step's warmstart
    d.qacc_warmstart.copy_from_slice(&d.qacc);
}

/// Port of `mj_step`: [`crate::forward`], then the model's integrator
/// ([`sim_scene::Integrator::Euler`] or [`sim_scene::Integrator::Rk4`]).
pub fn step<R: Real>(m: &Model<R>, d: &mut Data<R>) {
    step_with(m, d, &Faults::NONE);
}

pub(crate) fn step_with<R: Real>(m: &Model<R>, d: &mut Data<R>, faults: &Faults) {
    smooth::forward_with(m, d, faults);
    match m.integrator {
        sim_scene::Integrator::Euler => euler_with(m, d, faults),
        sim_scene::Integrator::Rk4 => rk4_with(m, d, faults),
    }
}
