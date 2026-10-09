//! Steps and trajectories with the constraints on, against MuJoCo 3.14.0 (`f64`).
//!
//! From each golden state, with its `qacc_warmstart`, for the Newton and the CG solver
//! (the model's own other options): one Euler step and one RK4 step (`qpos`, `qvel`,
//! `time`), and a 100-step trajectory with the model's own integrator, compared at
//! steps 1, 10 and 100, the warmstart carried from step to step as MuJoCo carries it.
//! The tolerances are phase 1b's: 1e-10 for one step, 1e-9 / 1e-8 / 1e-6 at steps
//! 1 / 10 / 100, each relative to the array's largest absolute value (floor 1e-12).
//! Every maximum is printed (`MEASURED`).
//!
//! The zoo has no constraint: its steps must equal phase 1b's, so the same states and
//! tolerances apply and `nefc` is 0 throughout.

mod common;

use common::cons::*;
use common::*;
use sim_physics::Integrator;

fn check(which: CWhich, solver: &str, report: &mut Report) {
    let c = ccompile::<f64>(which);
    let g = cgolden(which);
    let own = integrator_of(g["integrator"].as_str().unwrap());
    let kind = solver_of(solver);
    let tol = &TOL_F64;
    for s in states(&g) {
        let r = &s[solver];
        // one Euler step and one RK4 step
        for (label, integ) in [
            ("euler_step", Integrator::Euler),
            ("rk4_step", Integrator::Rk4),
        ] {
            let d = run_steps(&c.model, s, kind, integ, 1);
            let want = &r[label];
            let mut mm = c.model.clone();
            mm.integrator = integ;
            rec(
                report,
                &format!("{label}.qpos"),
                cmp(&qpos_of(&mm, &d), &farr(&want["qpos"])),
                tol.step.unwrap(),
                tol.floor,
            );
            rec(
                report,
                &format!("{label}.qvel"),
                cmp(&qvel_of(&d), &farr(&want["qvel"])),
                tol.step.unwrap(),
                tol.floor,
            );
            rec(
                report,
                &format!("{label}.time"),
                cmp(&[d.time], &[f(&want["time"])]),
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
        let mut d = cdata(&mm, s);
        let mut done = 0usize;
        for (k, &n) in steps.iter().enumerate() {
            while done < n {
                sim_physics::step(&mm, &mut d);
                done += 1;
            }
            rec(
                report,
                &format!("trajectory[{n}].qpos"),
                cmp(&qpos_of(&mm, &d), &farr(&traj["qpos"][k])),
                tol.traj[k].unwrap(),
                tol.floor,
            );
            rec(
                report,
                &format!("trajectory[{n}].qvel"),
                cmp(&qvel_of(&d), &farr(&traj["qvel"][k])),
                tol.traj[k].unwrap(),
                tol.floor,
            );
        }
    }
}

fn gate(which: CWhich, solver: &str) {
    let mut report = Report::default();
    check(which, solver, &mut report);
    report.print("f64", &format!("{} {solver}", which.name()));
    assert!(
        report.all_within(),
        "{} {solver}: out of tolerance: {:?}",
        which.name(),
        report.failures()
    );
}

#[test]
fn humanoid_newton_steps_and_trajectories_match_mujoco() {
    gate(CWhich::Humanoid, "newton");
}

#[test]
fn humanoid_cg_steps_and_trajectories_match_mujoco() {
    gate(CWhich::Humanoid, "cg");
}

#[test]
fn constrained_newton_steps_and_trajectories_match_mujoco() {
    gate(CWhich::Constrained, "newton");
}

#[test]
fn constrained_cg_steps_and_trajectories_match_mujoco() {
    gate(CWhich::Constrained, "cg");
}

#[test]
fn zoo_steps_and_trajectories_are_phase_1bs() {
    gate(CWhich::Zoo, "newton");
    gate(CWhich::Zoo, "cg");
    // and equal to the phase-1b golden (constraints disabled there, none exist here):
    // MuJoCo's two runs of the zoo agree to the last bit when nefc = 0
    let g1b = golden_of(Which::Zoo);
    let g = cgolden(CWhich::Zoo);
    assert_eq!(g1b["integrator"], g["integrator"]);
    // (with contacts off, `ccompile`'s setting, no step instantiates a row: the zoo has no
    // limit or friction loss; its `nefc_max` now counts the contact slots of phase 1c-ii)
    let mm = ccompile::<f64>(CWhich::Zoo).model;
    assert!(mm.disable.contact);
    let mut d = sim_physics::Data::new(&mm);
    sim_physics::forward(&mm, &mut d);
    assert_eq!((d.nefc, d.ncon), (0, 0));
}

/// The warmstart is part of the state MuJoCo carries: the trajectory from the same
/// `qpos`, `qvel` and `ctrl` with a different `qacc_warmstart` is a different run of
/// the solver (it can end at a slightly different iterate), and `Data` keeps it.
#[test]
fn a_step_leaves_qacc_as_the_next_warmstart() {
    let c = ccompile::<f64>(CWhich::Constrained);
    let g = cgolden(CWhich::Constrained);
    let s = &states(&g)[3];
    let mut d = cdata(&c.model, s);
    assert!(
        d.qacc_warmstart.iter().any(|&x| x != 0.0),
        "state 3 starts from a non-zero warmstart"
    );
    sim_physics::step(&c.model, &mut d);
    assert_eq!(d.qacc_warmstart, d.qacc);
    assert!(d.solver_niter > 0);
    // RK4: the last stage's qacc
    let mut m = c.model.clone();
    m.integrator = Integrator::Rk4;
    let mut d = cdata(&m, s);
    sim_physics::step(&m, &mut d);
    assert_eq!(d.qacc_warmstart, d.qacc);
    // a reset zeroes it
    d.reset(&m);
    assert!(d.qacc_warmstart.iter().all(|&x| x == 0.0));
}
