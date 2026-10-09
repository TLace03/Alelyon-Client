//! Checks that need no oracle: identities the engine must satisfy by itself.
//!
//! - Inverse dynamics: `rne` with acceleration returns `(M - armature) qacc +
//!   qfrc_bias`, so the composite-rigid-body inertia matrix and the Newton-Euler pass
//!   (two different algorithms in `smooth.rs`) must agree.
//! - `M` is symmetric and positive definite, the factorisation reproduces it, and
//!   `qacc` solves `M qacc = qfrc_smooth`.
//! - `integrate_pos` moves a free joint's position with the world-frame linear
//!   velocity and its orientation with the body-frame angular velocity.
//! - `Data::reset` and `Data::new` agree, and a step does not depend on stale
//!   scratch.

mod common;

use common::*;
use sim_physics::{Data, Integrator, Real, actuation, forward, integrate_pos, rne, step};

fn check_inverse_dynamics(which: Which) {
    let golden = golden_of(which);
    let c = compile::<f64>(scene_of(which));
    let m = &c.model;
    let mut worst = 0.0f64;
    for state in golden["states"].as_array().unwrap() {
        let mut d = data_from_state(m, state);
        forward(m, &mut d);
        let bias = d.qfrc_bias.clone();
        // M qacc from the dense matrix, without the armature: the Newton-Euler pass
        // knows the rigid bodies only, and the rotor inertia (armature) sits on the
        // diagonal of M alone
        let nv = m.nv;
        let m_qacc: Vec<f64> = (0..nv)
            .map(|i| {
                let rigid: f64 = (0..nv).map(|j| d.qm[i * nv + j] * d.qacc[j]).sum();
                rigid - m.dof_armature[i] * d.qacc[i]
            })
            .collect();
        // inverse dynamics by the recursive Newton-Euler pass
        rne(m, &mut d, true);
        let inverse: Vec<f64> = d.qfrc_bias.clone();
        let expected: Vec<f64> = (0..nv).map(|i| m_qacc[i] + bias[i]).collect();
        let e = compare(&inverse, &expected);
        worst = worst.max(e.rel());
        assert!(
            e.within(1e-10, 1e-12),
            "{}: {:e} of {:e}",
            which.name(),
            e.abs,
            e.scale
        );
    }
    println!(
        "MEASURED f64 {} rne(acc) vs M qacc + bias: max rel {worst:e}",
        which.name()
    );
}

#[test]
fn rne_with_acceleration_equals_the_rigid_m_qacc_plus_bias() {
    for which in MODELS {
        check_inverse_dynamics(which);
    }
}

#[test]
fn m_is_symmetric_positive_definite_and_qacc_solves_it() {
    for which in MODELS {
        let golden = golden_of(which);
        let c = compile::<f64>(scene_of(which));
        let m = &c.model;
        let nv = m.nv;
        for state in golden["states"].as_array().unwrap() {
            let mut d = data_from_state(m, state);
            forward(m, &mut d);
            for i in 0..nv {
                for j in 0..nv {
                    assert_eq!(d.qm[i * nv + j], d.qm[j * nv + i], "M is symmetric");
                }
                assert!(d.qm[i * nv + i] > 0.0, "M has a positive diagonal");
            }
            // M qacc = qfrc_smooth
            for i in 0..nv {
                let lhs: f64 = (0..nv).map(|j| d.qm[i * nv + j] * d.qacc[j]).sum();
                let scale = d.qfrc_smooth.iter().fold(1.0f64, |a, b| a.max(b.abs()));
                assert!(
                    (lhs - d.qfrc_smooth[i]).abs() < 1e-9 * scale,
                    "{} row {i}",
                    which.name()
                );
            }
            // positive definite: x' M x > 0 for the velocity of the state and for qacc
            for x in [&d.qvel, &d.qacc] {
                let q: f64 = (0..nv)
                    .map(|i| x[i] * (0..nv).map(|j| d.qm[i * nv + j] * x[j]).sum::<f64>())
                    .sum();
                assert!(q > 0.0);
            }
        }
    }
}

#[test]
fn integrate_pos_moves_a_free_joint_in_the_world_and_the_body_frame() {
    // the zoo's joints: free, ball, hinge, slide, hinge, slide
    let c = compile::<f64>(scene_of(Which::Zoo));
    let m = &c.model;
    let mut qpos = m.qpos0.clone();
    qpos[3..7].copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
    qpos[7..11].copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
    let mut qvel = vec![0.0; m.nv];
    // free joint: linear velocity (1, 2, 3) in the world, angular (0, 0, 2) in the body
    qvel[0..6].copy_from_slice(&[1.0, 2.0, 3.0, 0.0, 0.0, 2.0]);
    // ball: angular velocity (0, 1, 0)
    qvel[6..9].copy_from_slice(&[0.0, 1.0, 0.0]);
    // hinge 0.5, slide -0.25, hinge 4, slide 8
    qvel[9..13].copy_from_slice(&[0.5, -0.25, 4.0, 8.0]);
    let start = qpos.clone();
    let dt = 0.1;
    integrate_pos(m, &mut qpos, &qvel, dt);

    // free: position by dt * v
    let p0 = [start[0], start[1], start[2]];
    for k in 0..3 {
        assert_eq!(qpos[k], p0[k] + dt * qvel[k]);
    }
    // free: a rotation of dt * 2 = 0.2 rad about the body z axis, from the identity
    let half = 0.1f64;
    let expect = [half.cos(), 0.0, 0.0, half.sin()];
    for k in 0..4 {
        assert!((qpos[3 + k] - expect[k]).abs() < 1e-15, "{:?}", &qpos[3..7]);
    }
    // ball: 0.1 rad about y
    let expect = [0.05f64.cos(), 0.0, 0.05f64.sin(), 0.0];
    for k in 0..4 {
        assert!(
            (qpos[7 + k] - expect[k]).abs() < 1e-15,
            "{:?}",
            &qpos[7..11]
        );
    }
    // hinges and slides: q += dt * v
    for (i, v) in [0.5, -0.25, 4.0, 8.0].iter().enumerate() {
        assert_eq!(qpos[11 + i], start[11 + i] + dt * v);
    }
}

fn step_from<R: Real>(c: &Compiled<R>, dirty: bool) -> Data<R> {
    let golden = golden_of(Which::Zoo);
    let state = &golden["states"][2];
    let mut d = data_from_state(&c.model, state);
    if dirty {
        // scribble on every piece of scratch: a step must overwrite it, not read it
        for v in [
            &mut d.xpos,
            &mut d.xquat,
            &mut d.xmat,
            &mut d.xipos,
            &mut d.ximat,
            &mut d.xanchor,
            &mut d.xaxis,
            &mut d.subtree_com,
            &mut d.cdof,
            &mut d.cinert,
            &mut d.crb,
            &mut d.cvel,
            &mut d.cdof_dot,
            &mut d.cacc,
            &mut d.cfrc_body,
            &mut d.qm,
            &mut d.qld,
            &mut d.qld_diag_inv,
            &mut d.qh,
            &mut d.qh_diag_inv,
            &mut d.qfrc_bias,
            &mut d.qfrc_spring,
            &mut d.qfrc_damper,
            &mut d.qfrc_passive,
            &mut d.qfrc_actuator,
            &mut d.qfrc_smooth,
            &mut d.qacc,
            &mut d.qacc_step,
            &mut d.rk_x,
            &mut d.rk_f,
            &mut d.rk_dx,
        ] {
            v.fill(R::from_f64(123.456));
        }
        d.energy = [R::from_f64(9.0); 2];
    }
    for _ in 0..5 {
        step(&c.model, &mut d);
    }
    d
}

#[test]
fn a_step_depends_only_on_the_state_not_on_stale_scratch() {
    for integrator in [Integrator::Rk4, Integrator::Euler] {
        let mut scene = scene_of(Which::Zoo);
        scene.integrator = integrator;
        let c = compile::<f64>(scene);
        let clean = step_from(&c, false);
        let dirty = step_from(&c, true);
        let bits = |a: &[f64]| a.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&clean.qpos), bits(&dirty.qpos), "{integrator:?}");
        assert_eq!(bits(&clean.qvel), bits(&dirty.qvel), "{integrator:?}");
    }
}

#[test]
fn data_reset_restores_the_reference_pose() {
    let c = compile::<f64>(scene_of(Which::Zoo));
    let mut d = Data::new(&c.model);
    let fresh = d.clone();
    d.qvel.fill(1.0);
    d.ctrl.fill(0.5);
    for _ in 0..3 {
        step(&c.model, &mut d);
    }
    assert!(d.time > 0.0);
    d.reset(&c.model);
    assert_eq!(d.qpos, fresh.qpos);
    assert_eq!(d.qvel, fresh.qvel);
    assert_eq!(d.ctrl, fresh.ctrl);
    assert_eq!(d.time, 0.0);
    assert!(d.fits(&c.model));
}

#[test]
fn the_zoo_at_rest_at_its_reference_pose_sags_under_gravity_toward_equilibrium() {
    // at qpos0 and qvel 0 the springs are relaxed (qpos_spring = qpos0), so the only
    // forces are gravity and the (zero) controls: the bias force is the whole story,
    // and the passive force is exactly zero
    let c = compile::<f64>(scene_of(Which::Zoo));
    let mut d = Data::new(&c.model);
    forward(&c.model, &mut d);
    assert!(
        d.qfrc_passive.iter().all(|x| x.abs() < 1e-15),
        "{:?}",
        d.qfrc_passive
    );
    assert!(d.qfrc_actuator.iter().all(|x| *x == 0.0));
    assert!(d.qfrc_bias.iter().any(|x| x.abs() > 1e-3), "gravity acts");
}

#[test]
fn actuators_clamp_to_their_range_and_use_their_gear_and_gain() {
    // zoo: a motor on the elbow (dof 9, gear 2.5, range -1..1) and a position servo
    // on the slider (dof 10, kp 30, gear 2, target range -0.2..0.2, q at qpos index 12)
    let c = compile::<f64>(scene_of(Which::Zoo));
    let m = &c.model;
    let (elbow, slider, slider_q) = (9usize, 10usize, 12usize);
    assert_eq!(m.actuator_dofadr, vec![elbow, slider]);
    assert_eq!(m.actuator_qposadr[1], slider_q);
    let mut d = Data::new(m);

    // inside the range: motor gear * ctrl; servo gear * (kp * ctrl - kp * (gear * q))
    d.ctrl = vec![0.4, 0.1];
    d.qpos[slider_q] = 0.05;
    actuation(m, &mut d);
    assert_eq!(d.qfrc_actuator[elbow], 2.5 * 0.4);
    assert_eq!(
        d.qfrc_actuator[slider],
        2.0 * (30.0 * 0.1 - 30.0 * (2.0 * 0.05))
    );
    // every other dof is free of actuator force
    for (i, f) in d.qfrc_actuator.iter().enumerate() {
        if i != elbow && i != slider {
            assert_eq!(*f, 0.0, "dof {i}");
        }
    }

    // outside the range: the control is clamped first (MuJoCo's ctrllimited, which a
    // ctrlrange switches on)
    d.ctrl = vec![5.0, -3.0];
    actuation(m, &mut d);
    assert_eq!(d.qfrc_actuator[elbow], 2.5 * 1.0);
    assert_eq!(
        d.qfrc_actuator[slider],
        2.0 * (30.0 * -0.2 - 30.0 * (2.0 * 0.05))
    );
    d.ctrl = vec![-5.0, 3.0];
    actuation(m, &mut d);
    assert_eq!(d.qfrc_actuator[elbow], -2.5);
    assert_eq!(
        d.qfrc_actuator[slider],
        2.0 * (30.0 * 0.2 - 30.0 * (2.0 * 0.05))
    );

    // a servo pulls toward its target: zero target, displaced slider, force against q
    d.ctrl = vec![0.0, 0.0];
    actuation(m, &mut d);
    assert_eq!(d.qfrc_actuator[slider], -6.0);

    // a control that is not a number makes MuJoCo read every control as zero
    d.ctrl = vec![f64::NAN, 0.1];
    actuation(m, &mut d);
    assert_eq!(d.qfrc_actuator[elbow], 0.0);
    assert_eq!(d.qfrc_actuator[slider], -6.0);
}
