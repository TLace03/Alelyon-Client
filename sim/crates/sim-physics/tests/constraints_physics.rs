//! Physics checks of the soft constraints (phase 1c-i), each against a prediction
//! derived on paper from MuJoCo's constraint model, not from MuJoCo's output.
//!
//! **The soft-constraint model.** A row with Jacobian `J` (one dof, or a tendon), reference
//! acceleration `aref = -B v - K I (pos - margin)` and regularisation `R = (1 - I) A / I`
//! (`A` the approximate inverse inertia of the row, `D = 1 / R`) pulls the acceleration to
//! the minimum of `0.5 (a - a_s)' M (a - a_s) + s(J a - aref)`, where for an active limit
//! `s = 0.5 D (J a - aref)^2` and for friction loss `s` is quadratic while `|J a - aref| <
//! R f` and `f |J a - aref|` (minus a constant) beyond it. The standard `solref = (tc, z)`
//! gives `K = 1 / (d_max^2 tc^2 z^2)` and `B = 2 / (d_max tc)`, `d_max = solimp[1]`, and
//! the impedance `I` runs from `d_min` to `d_max` with the penetration `x = |pos - margin| /
//! width` (a power law with the midpoint and power of `solimp`).
//!
//! **A limit at rest.** At rest (`v = 0`, `a = 0`) the constraint force balances the
//! applied force `F` through the row's Jacobian `c` (`c = 1` for a joint, the tendon
//! coefficient for a tendon): `c f = F` with `f = D aref = D K I p`, `p` the penetration,
//! `D = I / ((1 - I) A)`. So the rest penetration solves
//!
//! `F / c = I(p)^2 K p / ((1 - I(p)) A)`,
//!
//! with `F` the force the dynamics press into the limit (for the inverted pendulum
//! `m g L sin(hi + p)`, which depends on `p`) and `A` the row's inverse inertia
//! (`1 / I_pivot` for a joint, `c^2 / m` for a tendon of a slide). The tests solve it by
//! bisection, run the simulation to rest and compare.
//!
//! **Friction loss.** A force `F` below the friction loss does not move the joint: in the
//! quadratic zone the row is a damper `D B v`, so the joint creeps at the steady
//! `v = F / (D B) = F R / B` (and `c`-times scaled for a tendon); above it, the row is in
//! its linear zone, the friction force is exactly `f`, and the acceleration is
//! `(F - f) / m` (`(F - c f) / m` for a tendon).

use sim_physics::{ConstraintState, Data, Model, PrimalSolver, step};
use sim_scene::mjcf;

/// A model from MJCF text.
fn model(xml: &str) -> Model<f64> {
    let scene = mjcf::load(xml, std::env::temp_dir()).expect("imports");
    Model::<f64>::compile(&scene).expect("compiles").0
}

/// MuJoCo's impedance as its documentation gives it (independent of the engine's code).
fn impedance(solimp: [f64; 5], pos_minus_margin: f64) -> f64 {
    let [d0, d1, width, mid, power] = solimp;
    if d0 == d1 {
        return 0.5 * (d0 + d1);
    }
    let x = (pos_minus_margin.abs() / width).min(1.0);
    if x >= 1.0 {
        return d1;
    }
    let y = if x <= mid {
        x.powf(power) / mid.powf(power - 1.0)
    } else {
        1.0 - (1.0 - x).powf(power) / (1.0 - mid).powf(power - 1.0)
    };
    d0 + y * (d1 - d0)
}

/// Solves `force_into_limit(p) / c = I(p)^2 K p / ((1 - I(p)) A)` for the penetration `p`
/// by bisection on `(0, hi]` (the right side minus the left grows with `p`).
fn rest_penetration(
    force_into_limit: impl Fn(f64) -> f64,
    c: f64,
    a: f64,
    k: f64,
    solimp: [f64; 5],
) -> f64 {
    let g = |p: f64| {
        let i = impedance(solimp, p);
        i * i * k * p / ((1.0 - i) * a) - force_into_limit(p) / c
    };
    let (mut lo, mut hi) = (0.0f64, 0.5f64);
    assert!(g(lo) < 0.0 && g(hi) > 0.0, "the bracket must hold the root");
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if g(mid) > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// The solver tolerance of these checks. They are about the soft-constraint model, not about
/// how well a solver converges: at MuJoCo's default 1e-8 the conjugate-gradient solver
/// leaves an acceleration error of about 1e-5, which would be what the rest-state checks
/// below measure; far below it both solvers reach the optimum of the one-row problem.
const SOLVE_TO: f64 = 1e-14;

const SOLIMP_FLAT: &str = "0.9 0.9 0.001 0.5 2";
const SOLIMP_POWER: &str = "0.5 0.99 0.01 0.5 2";

/// The inverted pendulum: a point-like mass `m = 1` at `L = 0.5` above a hinge about `y`,
/// rotational inertia about the CoM `0.02` (so `I_pivot = 0.02 + m L^2 = 0.27`), no
/// damping, upper limit 25 degrees, gravity 9.81. Gravity presses it into the limit with
/// the torque `m g L sin(theta)`.
fn pendulum(solref: &str, solimp: &str, timestep: f64) -> Model<f64> {
    model(&format!(
        r#"<mujoco><option timestep="{timestep}" integrator="Euler"/>
           <worldbody><body name="arm">
             <joint name="hinge" type="hinge" axis="0 1 0" range="-60 25"
                    solreflimit="{solref}" solimplimit="{solimp}"/>
             <inertial pos="0 0 0.5" mass="1" diaginertia="0.02 0.02 0.02"/>
           </body></worldbody></mujoco>"#
    ))
}

const M: f64 = 1.0;
const L: f64 = 0.5;
const I_COM: f64 = 0.02;
const G: f64 = 9.81;

fn run_pendulum(m: &Model<f64>, solver: PrimalSolver, seconds: f64) -> (Data<f64>, f64) {
    let mut mm = m.clone();
    mm.opt.solver = solver;
    mm.opt.tolerance = SOLVE_TO;
    let mut d = Data::new(&mm);
    d.qpos[0] = 0.05; // released from rest, almost upright
    let hi = 25.0f64.to_radians();
    let mut deepest = 0.0f64;
    let steps = (seconds / mm.timestep).round() as usize;
    for _ in 0..steps {
        step(&mm, &mut d);
        deepest = deepest.max(d.qpos[0] - hi);
    }
    (d, deepest)
}

#[test]
fn a_pendulum_dropped_onto_its_joint_limit_rests_at_the_predicted_penetration() {
    let hi = 25.0f64.to_radians();
    let a = 1.0 / (I_COM + M * L * L); // the pivot's inverse inertia
    for (label, solref, solimp_text, solimp) in [
        (
            "flat impedance",
            "0.02 1",
            SOLIMP_FLAT,
            [0.9, 0.9, 0.001, 0.5, 2.0],
        ),
        (
            "power-law impedance",
            "0.02 1",
            SOLIMP_POWER,
            [0.5, 0.99, 0.01, 0.5, 2.0],
        ),
        (
            "slower and underdamped",
            "0.04 0.7",
            SOLIMP_POWER,
            [0.5, 0.99, 0.01, 0.5, 2.0],
        ),
    ] {
        let (tc, zeta) = {
            let mut it = solref.split(' ').map(|x| x.parse::<f64>().unwrap());
            (it.next().unwrap(), it.next().unwrap())
        };
        let k = 1.0 / (solimp[1] * solimp[1] * tc * tc * zeta * zeta);
        let predicted = rest_penetration(|p| M * G * L * (hi + p).sin(), 1.0, a, k, solimp);
        let m = pendulum(solref, solimp_text, 0.002);
        // the model's own inverse inertia is the one derived here
        assert!(
            (m.dof_invweight0[0] - a).abs() < 1e-12 * a,
            "{label}: {} vs {a}",
            m.dof_invweight0[0]
        );
        for solver in [PrimalSolver::Newton, PrimalSolver::Cg] {
            let (d, deepest) = run_pendulum(&m, solver, 6.0);
            let penetration = d.qpos[0] - hi;
            println!(
                "MEASURED f64 pendulum on its limit ({label}, {solver:?}): predicted rest penetration {predicted:e} rad; measured {penetration:e} rad; deepest during the impact {deepest:e} rad; final |qvel| {:e}",
                d.qvel[0].abs()
            );
            // it stops at the limit and stays: at rest, at the predicted depth
            assert!(
                d.qvel[0].abs() < 1e-8,
                "{label}: still moving, {}",
                d.qvel[0]
            );
            assert!(penetration > 0.0, "{label}: it should rest past the limit");
            assert!(
                (penetration - predicted).abs() <= 1e-4 * predicted,
                "{label} {solver:?}: rest penetration {penetration:e}, predicted {predicted:e}"
            );
            // and not beyond it by more than the soft constraint allows during the impact:
            // the deepest penetration is bounded by a small multiple of the rest one
            assert!(
                deepest < 0.05,
                "{label}: the pendulum went {deepest} rad past its limit"
            );
            // the row ends in its quadratic zone, active
            assert_eq!(d.nefc, 1);
            assert_eq!(d.efc_state[0], ConstraintState::Quadratic);
        }
    }
}

#[test]
fn the_refsafe_rule_slows_a_limit_that_the_timestep_cannot_resolve() {
    // solref time constant 0.002 s with a 4 ms timestep: refsafe raises it to 2 h = 8 ms,
    // so the rest penetration is that of tc = 0.008, not 0.002; with the rule disabled
    // (mjDSBL_REFSAFE) it is that of tc = 0.002 (a much stiffer spring, which the explicit
    // step cannot resolve but which is still at rest at the same equilibrium of its own)
    let hi = 25.0f64.to_radians();
    let a = 1.0 / (I_COM + M * L * L);
    let solimp = [0.9, 0.9, 0.001, 0.5, 2.0];
    let h = 0.004;
    let m = pendulum("0.002 1", SOLIMP_FLAT, h);
    for (disabled, tc) in [(false, 2.0 * h), (true, 0.002)] {
        let k = 1.0 / (solimp[1] * solimp[1] * tc * tc);
        let predicted = rest_penetration(|p| M * G * L * (hi + p).sin(), 1.0, a, k, solimp);
        let mut mm = m.clone();
        mm.disable.refsafe = disabled;
        let mut d = Data::new(&mm);
        // start at rest right at the predicted equilibrium: a wrong stiffness (a wrong
        // time constant) would make the pendulum move away from it
        d.qpos[0] = hi + predicted;
        for _ in 0..50 {
            step(&mm, &mut d);
        }
        let penetration = d.qpos[0] - hi;
        println!(
            "MEASURED f64 refsafe {}: tc used {tc}, predicted penetration {predicted:e}, after 50 steps from it {penetration:e}, qvel {:e}",
            if disabled { "disabled" } else { "on" },
            d.qvel[0]
        );
        assert!((penetration - predicted).abs() <= 1e-6 * predicted);
        assert!(d.qvel[0].abs() < 1e-9);
    }
}

fn slider(extra_joint: &str, extra: &str) -> Model<f64> {
    model(&format!(
        r#"<mujoco><option timestep="0.002" gravity="0 0 0" integrator="Euler"/>
           <worldbody><body name="b">
             <joint name="s" type="slide" axis="1 0 0" {extra_joint}/>
             <inertial pos="0 0 0" mass="2" diaginertia="0.1 0.1 0.1"/>
           </body></worldbody>
           {extra}
           <actuator><motor name="m" joint="s" gear="1"/></actuator></mujoco>"#
    ))
}

fn run_force(m: &Model<f64>, force: f64, solver: PrimalSolver, seconds: f64) -> Data<f64> {
    let mut mm = m.clone();
    mm.opt.solver = solver;
    mm.opt.tolerance = SOLVE_TO;
    let mut d = Data::new(&mm);
    d.ctrl[0] = force;
    for _ in 0..((seconds / 0.002).round() as usize) {
        step(&mm, &mut d);
    }
    d
}

#[test]
fn friction_loss_holds_a_joint_below_the_friction_and_lets_it_slip_above() {
    let mass = 2.0;
    let f = 1.0;
    let tc = 0.02;
    let solimp = [0.9, 0.95, 0.001, 0.5, 2.0]; // MuJoCo's default impedance
    let m = slider(&format!(r#"frictionloss="{f}""#), "");
    // the row's impedance at pos = margin = 0 is d_min (x = 0), its A = 1 / mass
    let imp = impedance(solimp, 0.0);
    assert_eq!(imp, 0.9);
    let a = 1.0 / mass;
    let r = (1.0 - imp) * a / imp;
    let b = 2.0 / (solimp[1] * tc);
    for solver in [PrimalSolver::Newton, PrimalSolver::Cg] {
        // below the friction loss: held, creeping at F R / B
        for force in [0.8, -0.8, 0.3] {
            let d = run_force(&m, force, solver, 1.0);
            let creep = force * r / b;
            println!(
                "MEASURED f64 friction loss holds ({solver:?}): force {force} < {f}: qvel {:e} (predicted creep F R / B = {creep:e}); displacement after 1 s {:e}",
                d.qvel[0], d.qpos[0]
            );
            assert!(
                (d.qvel[0] - creep).abs() <= 1e-6 * creep.abs(),
                "{solver:?} {force}: {} vs {creep}",
                d.qvel[0]
            );
            assert!(d.qpos[0].abs() < 1e-3, "the joint moved {}", d.qpos[0]);
            assert_eq!(d.efc_state[0], ConstraintState::Quadratic);
        }
        // above it: slips, with the constant friction force f against the motion
        for (force, sign) in [(1.5, 1.0), (-1.5, -1.0)] {
            let d = run_force(&m, force, solver, 1.0);
            let accel = (force - sign * f) / mass;
            println!(
                "MEASURED f64 friction loss slips ({solver:?}): force {force} > {f}: qvel after 1 s {:e} (predicted (F - f) t / m = {accel:e}); efc_force {:e}",
                d.qvel[0], d.efc_force[0]
            );
            assert!(
                (d.qvel[0] - accel).abs() < 1e-9,
                "{solver:?} {force}: {} vs {accel}",
                d.qvel[0]
            );
            let zone = if sign > 0.0 {
                ConstraintState::LinearPos
            } else {
                ConstraintState::LinearNeg
            };
            // jar > 0 for motion in +x: the force is -f
            assert_eq!(d.efc_state[0], zone);
            assert!((d.efc_force[0] + sign * f).abs() < 1e-12);
        }
    }
}

#[test]
fn a_tendon_limit_rests_at_the_predicted_length_and_tendon_friction_behaves_like_a_joints() {
    // a slide (mass 2) under a fixed tendon of coefficient c = 2: length = 2 q
    let c = 2.0;
    let mass = 2.0;
    let solimp_text = SOLIMP_FLAT;
    let solimp = [0.9, 0.9, 0.001, 0.5, 2.0];
    let tendon = |attrs: &str| {
        slider(
            "",
            &format!(
                r#"<tendon><fixed name="t" {attrs}><joint joint="s" coef="{c}"/></fixed></tendon>"#
            ),
        )
    };

    // ---- the limit: pushed by F = 3 into the upper length limit 0.3
    let force = 3.0;
    let m = tendon(&format!(
        r#"limited="true" range="-0.3 0.3" solreflimit="0.02 1" solimplimit="{solimp_text}""#
    ));
    // the tendon's inverse inertia is J M^-1 J' = c^2 / m
    let a_t = c * c / mass;
    assert!((m.tendon_invweight0[0] - a_t).abs() < 1e-12);
    let k = 1.0 / (solimp[1] * solimp[1] * 0.02 * 0.02);
    // the dof force of the tendon's constraint force f is c f, so c f = F
    let p = rest_penetration(|_| force, c, a_t, k, solimp);
    for solver in [PrimalSolver::Newton, PrimalSolver::Cg] {
        let d = run_force(&m, force, solver, 3.0);
        let length = c * d.qpos[0];
        println!(
            "MEASURED f64 tendon limit ({solver:?}): predicted rest length {:e} (0.3 + {p:e}); measured {length:e}; qvel {:e}",
            0.3 + p,
            d.qvel[0]
        );
        assert!(d.qvel[0].abs() < 1e-9);
        assert!(
            (length - (0.3 + p)).abs() <= 1e-6 * p,
            "{solver:?}: {length}"
        );
        assert_eq!(d.nefc, 1);
        assert_eq!(d.efc_type[0], sim_physics::ConstraintType::LimitTendon);
        // the constraint force balances the applied force through the tendon
        assert!((c * d.efc_force[0] - force).abs() < 1e-6 * force);
    }

    // ---- the friction loss of the tendon: f_t = 0.5 holds a joint force below c f_t = 1.0
    let f_t = 0.5;
    let m = tendon(&format!(r#"frictionloss="{f_t}""#));
    for solver in [PrimalSolver::Newton, PrimalSolver::Cg] {
        for force in [0.8, -0.8] {
            let d = run_force(&m, force, solver, 1.0);
            assert!(
                d.qpos[0].abs() < 1e-3,
                "{solver:?} {force}: moved {}",
                d.qpos[0]
            );
            assert_eq!(d.efc_state[0], ConstraintState::Quadratic);
            assert_eq!(d.efc_type[0], sim_physics::ConstraintType::FrictionTendon);
        }
        for (force, sign) in [(1.5, 1.0), (-1.5, -1.0)] {
            let d = run_force(&m, force, solver, 1.0);
            let accel = (force - sign * c * f_t) / mass;
            println!(
                "MEASURED f64 tendon friction slips ({solver:?}): force {force}: qvel after 1 s {:e} (predicted (F - c f) t / m = {accel:e})",
                d.qvel[0]
            );
            assert!(
                (d.qvel[0] - accel).abs() < 1e-9,
                "{solver:?} {force}: {}",
                d.qvel[0]
            );
        }
    }
}

#[test]
fn a_ball_joint_limit_stops_a_rotation_at_its_angle() {
    // a ball joint with a 30 degree limit (a rotation angle of the quaternion), pressed
    // against it by gravity on an off-centre mass: the angle settles just beyond 30
    // degrees (the soft limit), where the limit torque balances the gravity torque
    let mut m = model(
        r#"<mujoco><option timestep="0.002" integrator="Euler"/>
           <worldbody><body name="b">
             <joint name="ball" type="ball" range="0 30" solreflimit="0.02 1" solimplimit="0.9 0.9 0.001 0.5 2"/>
             <inertial pos="0.3 0 0" mass="1" diaginertia="0.05 0.05 0.05"/>
           </body></worldbody></mujoco>"#,
    );
    m.opt.tolerance = SOLVE_TO;
    let limit = 30.0f64.to_radians();
    let mut d = Data::new(&m);
    // tip the body about y by 20 degrees: the CoM (at +x) drops, gravity turns it further
    let half = 0.5 * 20.0f64.to_radians();
    d.qpos[..4].copy_from_slice(&[half.cos(), 0.0, half.sin(), 0.0]);
    let mut max_angle = 0.0f64;
    for _ in 0..3000 {
        step(&m, &mut d);
        let q = &d.qpos[..4];
        let angle = 2.0 * (q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt().atan2(q[0]);
        max_angle = max_angle.max(angle);
    }
    let q = &d.qpos[..4];
    let angle = 2.0 * (q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt().atan2(q[0]);
    println!(
        "MEASURED f64 ball joint limit: rest angle {angle:e} rad vs limit {limit:e} (penetration {:e}); largest angle in flight {max_angle:e}; |qvel| {:e}",
        angle - limit,
        d.qvel.iter().fold(0.0f64, |a, b| a.max(b.abs()))
    );
    // gravity drives the CoM downward (the body hangs from +x down), so the angle grows
    // until the limit holds it: at rest, past the limit by a small soft penetration
    assert!(angle > limit, "it should rest past the limit");
    assert!(angle - limit < 0.02, "penetration {}", angle - limit);
    assert!(d.qvel.iter().all(|v| v.abs() < 1e-6), "{:?}", d.qvel);
    // The rest penetration, as for the hinge: the limit force balances the gravity torque about
    // the rotation axis, m g 0.3 cos(angle), through a row with Jacobian norm 1 and the
    // approximate inverse inertia A of a ball joint. MuJoCo takes A as the mean of the three
    // inverse inertias about the pivot at the reference pose (inertia 0.05, and 0.05 + 0.09
    // by the parallel-axis theorem about the other two axes): A = (1/0.05 + 2/0.14) / 3.
    let a = (1.0 / 0.05 + 2.0 / 0.14) / 3.0;
    let (tc, z, d1) = (0.02f64, 1.0f64, 0.9f64);
    let k = 1.0 / (d1 * d1 * tc * tc * z * z);
    let predicted = rest_penetration(
        |p| M * G * 0.3 * (limit + p).cos(),
        1.0,
        a,
        k,
        [0.9, 0.9, 0.001, 0.5, 2.0],
    );
    println!(
        "MEASURED f64 ball joint limit: predicted rest penetration {predicted:e} rad (A = {a:e}); measured {:e}",
        angle - limit
    );
    assert!(
        ((angle - limit) - predicted).abs() < 1e-6 * predicted,
        "predicted {predicted:e}, measured {:e}",
        angle - limit
    );
}
