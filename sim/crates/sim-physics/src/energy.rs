//! The mechanical energy of a state.
//!
//! Ports `mj_energyPos` and `mj_energyVel` (engine_sensor.c, 3.14.0), restricted
//! to what this phase models: the gravity potential of every body, the potential
//! of joint springs, and the kinetic energy `1/2 qvel' M qvel`. Tendon and flex
//! springs do not exist here.
//!
//! Invariants:
//! - [`energy_pos`] needs [`crate::kinematics`] (it reads `xipos`), and
//!   [`energy_vel`] needs [`crate::crb`] (it reads `qm`), for the state they are
//!   evaluated at; [`crate::forward`] provides both.
//! - The zero of the gravity potential is the world origin: `-sum m_i g . x_i`.
//! - A ball or free joint's rotational spring energy reads the raw `qpos`
//!   quaternion, not a normalised copy: MuJoCo normalises a local copy and then
//!   passes the raw one to `mju_subQuat`, and this port keeps that.
//! - The polynomial stiffness terms of `mju_polyPotential` are zero for any scene
//!   from `sim-scene`, so the spring potential is `1/2 k x^2`.
//! - Plain multiply and add only: no fused multiply-add, in the operation order of
//!   the C source, except that the kinetic energy's dot product is summed in
//!   index order where MuJoCo's `mju_dot` uses four partial sums (rounding only).

use crate::data::Data;
use crate::math::{dot3, lit, norm3, sub_quat, sub3, v3};
use crate::model::{JointType, Model};
use crate::real::Real;

/// Port of `mj_energyPos`: gravity potential plus joint spring potential.
/// Stores it in `d.energy[0]` and returns it.
pub fn energy_pos<R: Real>(m: &Model<R>, d: &mut Data<R>) -> R {
    // gravity: -sum_i mass_i * dot(gravity, xipos_i)
    let mut e = R::ZERO;
    for i in 1..m.nbody {
        e -= m.body_mass[i] * dot3(m.gravity, v3(&d.xipos, i));
    }

    // joint-level springs
    for b in 1..m.nbody {
        for j in m.body_jntadr[b]..m.body_jntadr[b] + m.body_jntnum[b] {
            let stiffness = m.jnt_stiffness[j];
            if stiffness == R::ZERO {
                continue;
            }
            let mut padr = m.jnt_qposadr[j];
            let half = lit::<R>(0.5);
            let mut rotation = false;
            match m.jnt_type[j] {
                JointType::Free => {
                    let dif = sub3(
                        [d.qpos[padr], d.qpos[padr + 1], d.qpos[padr + 2]],
                        [
                            m.qpos_spring[padr],
                            m.qpos_spring[padr + 1],
                            m.qpos_spring[padr + 2],
                        ],
                    );
                    let x = norm3(dif);
                    e += half * stiffness * (x * x);
                    padr += 3;
                    rotation = true;
                }
                JointType::Ball => rotation = true,
                JointType::Slide | JointType::Hinge => {
                    let x = d.qpos[padr] - m.qpos_spring[padr];
                    e += half * stiffness * (x * x);
                }
            }
            if rotation {
                let dif = sub_quat(
                    [
                        d.qpos[padr],
                        d.qpos[padr + 1],
                        d.qpos[padr + 2],
                        d.qpos[padr + 3],
                    ],
                    [
                        m.qpos_spring[padr],
                        m.qpos_spring[padr + 1],
                        m.qpos_spring[padr + 2],
                        m.qpos_spring[padr + 3],
                    ],
                );
                let x = norm3(dif);
                e += half * stiffness * (x * x);
            }
        }
    }
    d.energy[0] = e;
    e
}

/// Port of `mj_energyVel`: the kinetic energy `1/2 qvel' M qvel`. Stores it in
/// `d.energy[1]` and returns it.
pub fn energy_vel<R: Real>(m: &Model<R>, d: &mut Data<R>) -> R {
    let nv = m.nv;
    // vec = M qvel, then 0.5 * dot(vec, qvel)
    let mut dot = R::ZERO;
    for i in 0..nv {
        let mut row = R::ZERO;
        for j in 0..nv {
            row += d.qm[i * nv + j] * d.qvel[j];
        }
        dot += row * d.qvel[i];
    }
    let e = lit::<R>(0.5) * dot;
    d.energy[1] = e;
    e
}

/// The total mechanical energy `energy_pos + energy_vel` of the state the last
/// [`crate::forward`] evaluated.
pub fn energy<R: Real>(m: &Model<R>, d: &mut Data<R>) -> R {
    energy_pos(m, d) + energy_vel(m, d)
}
