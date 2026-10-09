//! Bit for bit: the constraint pipeline and the solvers are MuJoCo's, to the last bit.
//!
//! The solvers of phase 1c-i are ported statement for statement and keep MuJoCo's order
//! of every sum (the dot products, the products with the sparse `M`, the solves with
//! the factor of `M`, the Cholesky updates), so given the same model they follow the same
//! bits. The tolerances of `constraints_structure.rs`, `constraints_solve.rs` and
//! `constraints_steps.rs` are the spec's; this file records that on the two models with
//! constraints the error is exactly 0.0 in every comparison, on the platform the goldens
//! were generated on (Windows, MuJoCo 3.14.0's Python wheel): the constraint structure,
//! `qacc_smooth`, the Newton and CG solutions with their forces and iteration counts,
//! one Euler and one RK4 step, and the 100-step trajectories.
//!
//! The humanoid is bit for bit MuJoCo's only because two things are done as MuJoCo does
//! them, which the first test guards: a joint axis and an inertial-frame quaternion that
//! are already unit to rounding are used as they are, not normalised again (MuJoCo's
//! compiled inertial frame of a body with several geoms has a norm of `1 + 2e-15`, and
//! normalising it again moved `ximat`, `M` and the inverse inertias by a rounding step,
//! which the conjugate-gradient solver, being not self-correcting and having a termination
//! test that a rounding step can flip, amplified into 1e-8 .. 1e-5 differences in forces,
//! steps and trajectories).
//!
//! Exactness is a property of this platform's floating-point library (`sin`, `cos`,
//! `atan2` are not claimed bit identical across platforms, see `real.rs`); the gates in
//! the other files are the portable ones.
//!
//! The goldens were generated, and exactness observed, on Windows with the MSVC
//! toolchain only, so this file compiles there only; elsewhere the portable gates
//! still run.
#![cfg(all(target_os = "windows", target_env = "msvc"))]

mod common;

use common::cons::*;
use common::*;
use sim_physics::Integrator;

#[test]
fn the_compiled_humanoid_arrays_are_mujocos_bit_for_bit() {
    let m = ccompile::<f64>(CWhich::Humanoid).model;
    let g = golden_of(Which::Humanoid);
    let a = &g["arrays"];
    // the golden file holds quaternions as [x, y, z, w]
    let wxyz = |v: &serde_json::Value| -> Vec<f64> {
        farr(v)
            .chunks(4)
            .flat_map(|q| [q[3], q[0], q[1], q[2]])
            .collect()
    };
    assert_eq!(m.body_iquat, wxyz(&a["body_iquat"]), "body_iquat");
    assert_eq!(m.body_quat, wxyz(&a["body_quat"]), "body_quat");
    assert_eq!(m.jnt_axis, farr(&a["jnt_axis"]), "jnt_axis");
    assert_eq!(m.jnt_pos, farr(&a["jnt_pos"]), "jnt_pos");
    assert_eq!(m.body_pos, farr(&a["body_pos"]), "body_pos");
    assert_eq!(m.body_ipos, farr(&a["body_ipos"]), "body_ipos");
    assert_eq!(m.body_inertia, farr(&a["body_inertia"]), "body_inertia");
    assert_eq!(m.body_mass, farr(&a["body_mass"]), "body_mass");
    // and the inverse inertias computed from them at compile time
    let g = cgolden(CWhich::Humanoid);
    let c = &g["arrays"];
    assert_eq!(
        m.dof_invweight0,
        farr(&c["dof_invweight0"]),
        "dof_invweight0"
    );
    assert_eq!(
        m.body_invweight0,
        farr(&c["body_invweight0"]),
        "body_invweight0"
    );
    assert_eq!(
        m.tendon_invweight0,
        farr(&c["tendon_invweight0"]),
        "tendon_invweight0"
    );
    assert_eq!(m.meaninertia, f(&c["meaninertia"]), "meaninertia");
}

/// The largest absolute error over every comparison of one model, the comparison it is
/// at, how many there were, and the largest difference of a `solver_niter`.
fn worst_error(which: CWhich) -> (usize, f64, String, usize) {
    let c = ccompile::<f64>(which);
    let m = &c.model;
    let g = cgolden(which);
    let own = integrator_of(g["integrator"].as_str().unwrap());
    let mut errs: Vec<(String, f64)> = Vec::new();
    let mut niter_diff = 0usize;
    for (k, s) in states(&g).iter().enumerate() {
        // the structure and the smooth acceleration
        let mut d = cdata(m, s);
        sim_physics::forward(m, &mut d);
        let n = d.nefc;
        let mut put = |name: &str, e: f64| errs.push((format!("state {k} {name}"), e));
        put(
            "ten_J",
            cmp(&widen(&d.ten_j), &golden_rows(&s["ten_J"])).abs,
        );
        put(
            "qacc_smooth",
            cmp(&widen(&d.qacc_smooth), &farr(&s["qacc_smooth"])).abs,
        );
        put(
            "efc_J",
            cmp(&head(&d.efc_j, n * m.nv), &golden_rows(&s["efc_J"])).abs,
        );
        put("efc_R", cmp(&head(&d.efc_r, n), &farr(&s["efc_R"])).abs);
        put("efc_D", cmp(&head(&d.efc_d, n), &farr(&s["efc_D"])).abs);
        put(
            "efc_KBIP",
            cmp(&head(&d.efc_kbip, 4 * n), &golden_rows(&s["efc_KBIP"])).abs,
        );
        put(
            "efc_aref",
            cmp(&head(&d.efc_aref, n), &farr(&s["efc_aref"])).abs,
        );
        put(
            "efc_diagApprox",
            cmp(&head(&d.efc_diag_approx, n), &farr(&s["efc_diagApprox"])).abs,
        );
        for solver in SOLVERS {
            let kind = solver_of(solver);
            let r = &s[solver];
            let d = run_forward(m, s, kind);
            let mut put = |name: &str, e: f64| errs.push((format!("state {k} {solver} {name}"), e));
            put("qacc", cmp(&widen(&d.qacc), &farr(&r["qacc"])).abs);
            put(
                "efc_force",
                cmp(&head(&d.efc_force, d.nefc), &farr(&r["efc_force"])).abs,
            );
            put(
                "qfrc_constraint",
                cmp(&widen(&d.qfrc_constraint), &farr(&r["qfrc_constraint"])).abs,
            );
            niter_diff = niter_diff.max(
                d.solver_niter
                    .abs_diff(r["solver_niter"].as_u64().unwrap() as usize),
            );
            // one Euler and one RK4 step
            for (label, integ) in [
                ("euler_step", Integrator::Euler),
                ("rk4_step", Integrator::Rk4),
            ] {
                let d = run_steps(m, s, kind, integ, 1);
                let mut mm = m.clone();
                mm.integrator = integ;
                put(
                    &format!("{label}.qpos"),
                    cmp(&qpos_of(&mm, &d), &farr(&r[label]["qpos"])).abs,
                );
                put(
                    &format!("{label}.qvel"),
                    cmp(&qvel_of(&d), &farr(&r[label]["qvel"])).abs,
                );
            }
            // the 100-step trajectory
            let mut mm = m.clone();
            mm.opt.solver = kind;
            mm.integrator = own;
            let mut d = cdata(&mm, s);
            let mut done = 0usize;
            for (i, n) in [1usize, 10, 100].iter().enumerate() {
                while done < *n {
                    sim_physics::step(&mm, &mut d);
                    done += 1;
                }
                let tr = &r["trajectory"];
                put(
                    &format!("trajectory[{n}].qpos"),
                    cmp(&qpos_of(&mm, &d), &farr(&tr["qpos"][i])).abs,
                );
                put(
                    &format!("trajectory[{n}].qvel"),
                    cmp(&qvel_of(&d), &farr(&tr["qvel"][i])).abs,
                );
            }
        }
    }
    let worst = errs.iter().cloned().fold(
        (String::new(), 0.0f64),
        |a, b| if b.1 > a.1 { b } else { a },
    );
    (errs.len(), worst.1, worst.0, niter_diff)
}

fn exact(which: CWhich) {
    let (n, worst, at, niter_diff) = worst_error(which);
    println!(
        "MEASURED f64 {} bit for bit: worst absolute error over {n} comparisons (structure, Newton and CG solves, steps, trajectories) = {worst:e}{}; largest solver_niter difference {niter_diff}",
        which.name(),
        if worst > 0.0 {
            format!(" (at {at})")
        } else {
            String::new()
        }
    );
    assert_eq!(niter_diff, 0);
    assert_eq!(
        worst,
        0.0,
        "{}: not bit for bit MuJoCo's, worst at {at}",
        which.name()
    );
}

#[test]
fn the_humanoid_is_bit_for_bit_mujocos() {
    exact(CWhich::Humanoid);
}

#[test]
fn the_constrained_model_is_bit_for_bit_mujocos() {
    exact(CWhich::Constrained);
}
