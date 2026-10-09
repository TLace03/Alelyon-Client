//! Steps and trajectories with contacts against MuJoCo 3.14.0 (`f64`, phase 1c-ii), both cones.
//!
//! **Steps.** From each golden state, with its `qacc_warmstart`, for the Newton and the CG solver
//! (the model's own other options): one Euler step and one RK4 step (`qpos`, `qvel`, `time`)
//! within 1e-10, and a 100-step trajectory with the model's own integrator, compared at steps
//! 1, 10 and 100 within 1e-9, 1e-8 and 1e-6, the warmstart carried from step to step as MuJoCo
//! carries it; each relative to the array's largest absolute value (floor 1e-12). RK4 runs the
//! collision step at every stage, as MuJoCo does.
//!
//! **Settle runs.** Each scene's own initial pose run from rest (the sphere sliding at 2 m/s, the
//! box at 1.2 m/s, the stack, the humanoid falling from `qpos0` at h = 0.005, the pile for 2,400
//! steps): the poses MuJoCo recorded at listed steps against ours. Contact dynamics are chaotic,
//! so only the frames up to step 100 are gated (at the trajectory tolerances above); every later
//! frame is reported (`MEASURED`), with the contact and row counts of the step.

mod common;

use common::cons::*;
use common::contacts::*;
use common::*;
use sim_physics::{Integrator, Model, step};

fn traj_tolerance(step: usize) -> Option<f64> {
    match step {
        0..=1 => Some(1e-9),
        2..=10 => Some(1e-8),
        11..=100 => Some(1e-6),
        _ => None,
    }
}

fn check(which: TWhich, v: Variant, solver: &str, report: &mut Report) {
    let c = tcompile::<f64>(which, v);
    let g = tgolden(which, v);
    let own = integrator_of(g["integrator"].as_str().unwrap());
    let kind = tsolver_of(solver);
    let tol = &TOL_F64;
    for s in tstates(&g) {
        let r = &s[solver];
        for (label, integ) in [
            ("euler_step", Integrator::Euler),
            ("rk4_step", Integrator::Rk4),
        ] {
            let mut mm = c.model.clone();
            mm.opt.solver = kind;
            mm.integrator = integ;
            let mut d = tdata(&mm, s);
            step(&mm, &mut d);
            assert_eq!(d.warning_collision_overflow, 0);
            let want = &r[label];
            rec(
                report,
                &format!("{label}.qpos"),
                safe_cmp(&qpos_of(&mm, &d), &farr(&want["qpos"])),
                tol.step.unwrap(),
                tol.floor,
            );
            rec(
                report,
                &format!("{label}.qvel"),
                safe_cmp(&qvel_of(&d), &farr(&want["qvel"])),
                tol.step.unwrap(),
                tol.floor,
            );
            rec(
                report,
                &format!("{label}.time"),
                safe_cmp(&[d.time], &[f(&want["time"])]),
                tol.step.unwrap(),
                tol.floor,
            );
        }
        // the 100-step trajectory with the model's own integrator
        let traj = &r["trajectory"];
        let steps: Vec<usize> = traj["steps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let mut mm = c.model.clone();
        mm.opt.solver = kind;
        mm.integrator = own;
        let mut d = tdata(&mm, s);
        let mut done = 0usize;
        for (k, &n) in steps.iter().enumerate() {
            while done < n {
                step(&mm, &mut d);
                assert_eq!(d.warning_collision_overflow, 0);
                done += 1;
            }
            rec(
                report,
                &format!("trajectory[{n}].qpos"),
                safe_cmp(&qpos_of(&mm, &d), &farr(&traj["qpos"][k])),
                tol.traj[k].unwrap(),
                tol.floor,
            );
            rec(
                report,
                &format!("trajectory[{n}].qvel"),
                safe_cmp(&qvel_of(&d), &farr(&traj["qvel"][k])),
                tol.traj[k].unwrap(),
                tol.floor,
            );
        }
    }
}

fn gate(which: TWhich, v: Variant, solver: &str) {
    let mut report = Report::default();
    check(which, v, solver, &mut report);
    report.print("f64", &format!("{} {} {solver}", which.name(), v.name()));
    assert!(
        report.all_within(),
        "{} {} {solver}: out of tolerance: {:?}",
        which.name(),
        v.name(),
        report.failures()
    );
}

macro_rules! step_tests {
    ($($name:ident: $which:expr, $variant:expr, $solver:expr;)*) => {
        $(
            #[test]
            fn $name() {
                gate($which, $variant, $solver);
            }
        )*
    };
}

step_tests! {
    sphere_pyramidal_newton_steps_match_mujoco: TWhich::Sphere, Variant::Pyramidal, "newton";
    sphere_pyramidal_cg_steps_match_mujoco: TWhich::Sphere, Variant::Pyramidal, "cg";
    sphere_elliptic_newton_steps_match_mujoco: TWhich::Sphere, Variant::Elliptic, "newton";
    sphere_elliptic_cg_steps_match_mujoco: TWhich::Sphere, Variant::Elliptic, "cg";
    box_pyramidal_newton_steps_match_mujoco: TWhich::Box, Variant::Pyramidal, "newton";
    box_pyramidal_cg_steps_match_mujoco: TWhich::Box, Variant::Pyramidal, "cg";
    box_elliptic_newton_steps_match_mujoco: TWhich::Box, Variant::Elliptic, "newton";
    box_elliptic_cg_steps_match_mujoco: TWhich::Box, Variant::Elliptic, "cg";
    stack_pyramidal_newton_steps_match_mujoco: TWhich::Stack, Variant::Pyramidal, "newton";
    stack_pyramidal_cg_steps_match_mujoco: TWhich::Stack, Variant::Pyramidal, "cg";
    stack_elliptic_newton_steps_match_mujoco: TWhich::Stack, Variant::Elliptic, "newton";
    stack_elliptic_cg_steps_match_mujoco: TWhich::Stack, Variant::Elliptic, "cg";
    capsules_pyramidal_newton_steps_match_mujoco: TWhich::Capsules, Variant::Pyramidal, "newton";
    capsules_pyramidal_cg_steps_match_mujoco: TWhich::Capsules, Variant::Pyramidal, "cg";
    capsules_elliptic_newton_steps_match_mujoco: TWhich::Capsules, Variant::Elliptic, "newton";
    capsules_elliptic_cg_steps_match_mujoco: TWhich::Capsules, Variant::Elliptic, "cg";
    pile_pyramidal_newton_steps_match_mujoco: TWhich::Pile, Variant::Pyramidal, "newton";
    pile_pyramidal_cg_steps_match_mujoco: TWhich::Pile, Variant::Pyramidal, "cg";
    pile_elliptic_newton_steps_match_mujoco: TWhich::Pile, Variant::Elliptic, "newton";
    pile_elliptic_cg_steps_match_mujoco: TWhich::Pile, Variant::Elliptic, "cg";
    humanoid_pyramidal_newton_steps_match_mujoco: TWhich::Humanoid, Variant::Pyramidal, "newton";
    humanoid_pyramidal_cg_steps_match_mujoco: TWhich::Humanoid, Variant::Pyramidal, "cg";
    humanoid_elliptic_newton_steps_match_mujoco: TWhich::Humanoid, Variant::Elliptic, "newton";
    humanoid_elliptic_cg_steps_match_mujoco: TWhich::Humanoid, Variant::Elliptic, "cg";
    zoo_pyramidal_newton_steps_match_mujoco: TWhich::Zoo, Variant::Pyramidal, "newton";
    zoo_pyramidal_cg_steps_match_mujoco: TWhich::Zoo, Variant::Pyramidal, "cg";
    zoo_elliptic_newton_steps_match_mujoco: TWhich::Zoo, Variant::Elliptic, "newton";
    zoo_elliptic_cg_steps_match_mujoco: TWhich::Zoo, Variant::Elliptic, "cg";
}

/// The settle runs against MuJoCo's recorded frames: gated up to step 100, reported after.
fn settle_gate(which: TWhich, v: Variant) {
    let g = tgolden(which, v);
    let s = &g["settle"];
    let ours = tsettle::<f64>(which, v, &g);
    let c = tcompile::<f64>(which, v);
    let frames = s["frames"].as_array().unwrap();
    assert_eq!(ours.len(), frames.len());
    let mut failures = Vec::new();
    for ((n, d), fr) in ours.iter().zip(frames) {
        assert_eq!(*n as u64, fr["step"].as_u64().unwrap());
        let eq = safe_cmp(&qpos_of(&c.model, d), &farr(&fr["qpos"]));
        let ev = safe_cmp(&qvel_of(d), &farr(&fr["qvel"]));
        let tol = traj_tolerance(*n);
        println!(
            "MEASURED f64 {} {} settle step {n}: qpos rel {:e}, qvel rel {:e}; ncon ours={} mujoco={}, nefc ours={} mujoco={} {}",
            which.name(),
            v.name(),
            eq.rel(),
            ev.rel(),
            d.ncon,
            fr["ncon"],
            d.nefc,
            fr["nefc"],
            if tol.is_some() {
                "(gated)"
            } else {
                "(reported)"
            }
        );
        if let Some(t) = tol
            && (!eq.within(t, 1e-12) || !ev.within(t, 1e-12))
        {
            failures.push(*n);
        }
    }
    assert!(
        failures.is_empty(),
        "{} {}: frames out of tolerance: {failures:?}",
        which.name(),
        v.name()
    );
}

macro_rules! settle_tests {
    ($($name:ident: $which:expr, $variant:expr;)*) => {
        $(
            #[test]
            fn $name() {
                settle_gate($which, $variant);
            }
        )*
    };
}

settle_tests! {
    sphere_pyramidal_settle_run_follows_mujoco: TWhich::Sphere, Variant::Pyramidal;
    sphere_elliptic_settle_run_follows_mujoco: TWhich::Sphere, Variant::Elliptic;
    box_pyramidal_settle_run_follows_mujoco: TWhich::Box, Variant::Pyramidal;
    box_elliptic_settle_run_follows_mujoco: TWhich::Box, Variant::Elliptic;
    stack_pyramidal_settle_run_follows_mujoco: TWhich::Stack, Variant::Pyramidal;
    stack_elliptic_settle_run_follows_mujoco: TWhich::Stack, Variant::Elliptic;
    humanoid_pyramidal_fall_follows_mujoco: TWhich::Humanoid, Variant::Pyramidal;
    humanoid_elliptic_fall_follows_mujoco: TWhich::Humanoid, Variant::Elliptic;
    pile_pyramidal_settle_run_follows_mujoco: TWhich::Pile, Variant::Pyramidal;
    pile_elliptic_settle_run_follows_mujoco: TWhich::Pile, Variant::Elliptic;
}

/// A step leaves `qacc` as the next warmstart, with contacts as without.
#[test]
fn a_step_with_contacts_leaves_qacc_as_the_next_warmstart() {
    let c = tcompile::<f64>(TWhich::Stack, Variant::Pyramidal);
    let g = tgolden(TWhich::Stack, Variant::Pyramidal);
    let s = &tstates(&g)[3];
    let mut d = tdata(&c.model, s);
    assert!(d.qacc_warmstart.iter().any(|&x| x != 0.0));
    step(&c.model, &mut d);
    assert_eq!(d.qacc_warmstart, d.qacc);
    assert!(d.solver_niter > 0 && d.ncon > 0);
    let _: &Model<f64> = &c.model;
}
