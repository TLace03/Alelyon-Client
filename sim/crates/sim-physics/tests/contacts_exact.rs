//! Bit for bit: the collision step, the contact rows and the solvers are MuJoCo's, to the last bit.
//!
//! The colliders, the contact frames, the constraint rows and the primal solvers of phase 1c-ii
//! are ported statement for statement and keep MuJoCo's order of every sum (no fused multiply-add
//! anywhere: `Real` has none, and MuJoCo's `src/engine` has no `fmadd` intrinsic), so given the same
//! model they follow the same bits. The tolerances of the other contact tests are the spec's; this
//! file records that on every contact model and both cones the error is exactly 0.0 in every
//! comparison, on the platform the goldens were generated on (Windows, MuJoCo 3.14.0's Python
//! wheel): the compiled contact arrays, the contact list with its order, the constraint rows,
//! `qacc_smooth`, the Newton and CG solutions with their forces, zones, cone Hessians, contact forces
//! and iteration counts, one Euler and one RK4 step, the 100-step trajectories, the converged
//! optimum, the CG path iterate by iterate, the settle runs (the pile's 2,400 steps among them) and
//! the collider sweep (eleven colliders, 64 poses each, with and without margin and gap).
//!
//! **The settle runs are asserted here and nowhere else past step 100** (the phase-1c-ii review
//! found this header claiming it while the file ran none): every recorded frame of every model's
//! settle run, the pile's ten frames of its last second among them, `qpos` and `qvel` equal to
//! the bit with equal `ncon` and `nefc` (in `worst_error`, so in every `exact` test), and the
//! pile's last-second figures (the deepest contact `-dist`, the separating-axis penetrations,
//! the speed at 10 s) in `the_piles_last_second_figures_are_mujocos`. The inertial frames
//! (`body_sameframe`) are in `the_inertial_frames_are_bit_for_bit_mujocos`.
//!
//! One thing made it true that phase 1b missed: a "simple" dof (a free body whose inertial frame is
//! its body frame, `dof_simplenum`) has a stored row of `M` that is the constant diagonal `dof_M0`,
//! which `mj_crb` copies, whereas the general composite-inertia path computes the same diagonal
//! from the rotated `cinert` and differs by a few units in the last place. Before `crb` copied
//! `dof_M0`, every contact model with a free box or sphere had a `qacc_smooth` that differed from
//! MuJoCo's by up to 3.6e-15 (4 ulp), and the solutions inherited it.
//!
//! Exactness is a property of this platform's floating-point library (`sin`, `cos`, `atan2` are not
//! claimed bit identical across platforms, see `real.rs`); the gates in the other files are the
//! portable ones. The goldens were generated, and exactness observed, on Windows with the MSVC
//! toolchain only, so this file compiles there only.
#![cfg(all(target_os = "windows", target_env = "msvc"))]

mod common;

use common::cons::*;
use common::contacts::*;
use common::*;
use sim_physics::{Integrator, PrimalSolver, contact_force};

/// The largest absolute error over every comparison of one model and cone, how many comparisons
/// there were, where the worst was, and the largest `solver_niter` difference.
fn worst_error(which: TWhich, v: Variant) -> (usize, f64, String, usize) {
    let c = tcompile::<f64>(which, v);
    let m = &c.model;
    let g = tgolden(which, v);
    let own = integrator_of(g["integrator"].as_str().unwrap());
    let mut errs: Vec<(String, f64)> = Vec::new();
    let mut niter_diff = 0usize;
    let mut zones_differ = 0usize;

    // the compiled arrays
    {
        let a = &g["arrays"];
        let xyzw = |w: &[f64]| -> Vec<f64> {
            w.chunks(4).flat_map(|q| [q[1], q[2], q[3], q[0]]).collect()
        };
        let mut put = |name: &str, ours: Vec<f64>, theirs: Vec<f64>| {
            errs.push((format!("compiled {name}"), safe_cmp(&ours, &theirs).abs));
        };
        put("geom_pos", widen(&m.geom_pos), farr(&a["geom_pos"]));
        put(
            "geom_quat",
            xyzw(&widen(&m.geom_quat)),
            farr(&a["geom_quat"]),
        );
        put("geom_size", widen(&m.geom_size), farr(&a["geom_size"]));
        put(
            "geom_rbound",
            widen(&m.geom_rbound),
            farr(&a["geom_rbound"]),
        );
        put(
            "geom_solmix",
            widen(&m.geom_solmix),
            farr(&a["geom_solmix"]),
        );
        put(
            "geom_solref",
            widen(&m.geom_solref),
            farr(&a["geom_solref"]),
        );
        put(
            "geom_solimp",
            widen(&m.geom_solimp),
            farr(&a["geom_solimp"]),
        );
        put(
            "geom_friction",
            widen(&m.geom_friction),
            farr(&a["geom_friction"]),
        );
        put(
            "geom_margin",
            widen(&m.geom_margin),
            farr(&a["geom_margin"]),
        );
        put("geom_gap", widen(&m.geom_gap), farr(&a["geom_gap"]));
        put(
            "body_invweight0",
            widen(&m.body_invweight0),
            farr(&a["body_invweight0"]),
        );
        put(
            "meaninertia",
            vec![m.meaninertia],
            vec![f(&a["meaninertia"])],
        );
    }

    for (k, s) in tstates(&g).iter().enumerate() {
        // the contact list and the constraint structure
        let d = tforward(m, s, PrimalSolver::Newton);
        let mut report = Report::default();
        let notes = record_structure(m, &d, s, &mut report);
        assert!(
            notes.is_empty(),
            "{} {} state {k}: {notes:?}",
            which.name(),
            v.name()
        );
        errs.push((format!("state {k} structure"), report.max_abs()));

        for solver in TSOLVERS {
            let kind = tsolver_of(solver);
            let r = &s[solver];
            let d = tforward(m, s, kind);
            let mut report = Report::default();
            if !record_solution(m, &d, s, solver, &mut report) {
                zones_differ += 1;
            }
            errs.push((format!("state {k} {solver} solution"), report.max_abs()));
            // the contact force is mj_contactForce's
            let forces: Vec<f64> = (0..d.ncon).flat_map(|i| contact_force(m, &d, i)).collect();
            errs.push((
                format!("state {k} {solver} contact_force"),
                safe_cmp(&forces, &trows(&r["contact_force"])).abs,
            ));
            niter_diff = niter_diff.max(
                d.solver_niter
                    .abs_diff(r["solver_niter"].as_u64().unwrap() as usize),
            );
            // one Euler and one RK4 step
            let mut put = |name: &str, e: f64| errs.push((format!("state {k} {solver} {name}"), e));
            for (label, integ) in [
                ("euler_step", Integrator::Euler),
                ("rk4_step", Integrator::Rk4),
            ] {
                let mut mm = m.clone();
                mm.opt.solver = kind;
                mm.integrator = integ;
                let mut d = tdata(&mm, s);
                sim_physics::step(&mm, &mut d);
                assert_eq!(d.warning_collision_overflow, 0, "box-box overflow");
                put(
                    &format!("{label}.qpos"),
                    safe_cmp(&qpos_of(&mm, &d), &farr(&r[label]["qpos"])).abs,
                );
                put(
                    &format!("{label}.qvel"),
                    safe_cmp(&qvel_of(&d), &farr(&r[label]["qvel"])).abs,
                );
            }
            // the 100-step trajectory
            let mut mm = m.clone();
            mm.opt.solver = kind;
            mm.integrator = own;
            let mut d = tdata(&mm, s);
            let mut done = 0usize;
            for (i, n) in [1usize, 10, 100].iter().enumerate() {
                while done < *n {
                    sim_physics::step(&mm, &mut d);
                    assert_eq!(d.warning_collision_overflow, 0, "box-box overflow");
                    done += 1;
                }
                let tr = &r["trajectory"];
                put(
                    &format!("trajectory[{n}].qpos"),
                    safe_cmp(&qpos_of(&mm, &d), &farr(&tr["qpos"][i])).abs,
                );
                put(
                    &format!("trajectory[{n}].qvel"),
                    safe_cmp(&qvel_of(&d), &farr(&tr["qvel"][i])).abs,
                );
            }
        }

        // the converged optimum
        let mut mc = m.clone();
        let st = &g["converged_settings"];
        mc.opt.tolerance = f(&st["tolerance"]);
        mc.opt.iterations = st["iterations"].as_u64().unwrap() as usize;
        mc.opt.ls_iterations = st["ls_iterations"].as_u64().unwrap() as usize;
        let d = tforward(&mc, s, PrimalSolver::Newton);
        errs.push((
            format!("state {k} converged qacc"),
            safe_cmp(&widen(&d.qacc), &farr(&s["converged"]["qacc"])).abs,
        ));
        errs.push((
            format!("state {k} converged efc_force"),
            safe_cmp(
                &thead(&d.efc_force, d.nefc),
                &farr(&s["converged"]["efc_force"]),
            )
            .abs,
        ));
    }

    // the CG path, iterate by iterate (zoo state 0)
    if which == TWhich::Zoo {
        let s = &tstates(&g)[0];
        for (k, mj) in s["cg_iterates"].as_array().unwrap().iter().enumerate() {
            let mut mm = m.clone();
            mm.opt.solver = PrimalSolver::Cg;
            mm.opt.iterations = k;
            let mut d = tdata(&mm, s);
            sim_physics::forward(&mm, &mut d);
            assert_eq!(d.warning_collision_overflow, 0, "box-box overflow");
            errs.push((
                format!("cg iterate {k}"),
                safe_cmp(&widen(&d.qacc), &farr(mj)).abs,
            ));
        }
    }
    assert_eq!(
        zones_differ,
        0,
        "{} {}: a row ended in another zone",
        which.name(),
        v.name()
    );

    // the settle run: every recorded frame (the pile's last second, the humanoid's fall to step
    // 400 among them), `qpos` and `qvel` to the bit and the same contact and row counts
    if let Some(settle) = g.get("settle") {
        let frames = settle["frames"].as_array().unwrap();
        let ours = tsettle::<f64>(which, v, &g);
        assert_eq!(ours.len(), frames.len(), "{} {}", which.name(), v.name());
        for ((n, d), fr) in ours.iter().zip(frames) {
            assert_eq!(*n as u64, fr["step"].as_u64().unwrap());
            errs.push((
                format!("settle step {n} qpos"),
                safe_cmp(&qpos_of(m, d), &farr(&fr["qpos"])).abs,
            ));
            errs.push((
                format!("settle step {n} qvel"),
                safe_cmp(&qvel_of(d), &farr(&fr["qvel"])).abs,
            ));
            errs.push((
                format!("settle step {n} ncon"),
                (d.ncon as f64 - fr["ncon"].as_f64().unwrap()).abs(),
            ));
            errs.push((
                format!("settle step {n} nefc"),
                (d.nefc as f64 - fr["nefc"].as_f64().unwrap()).abs(),
            ));
        }
    }

    let worst = errs.iter().cloned().fold(
        (String::new(), 0.0f64),
        |a, b| if b.1 > a.1 { b } else { a },
    );
    (errs.len(), worst.1, worst.0, niter_diff)
}

fn exact(which: TWhich, v: Variant) {
    let (n, worst, at, niter_diff) = worst_error(which, v);
    println!(
        "MEASURED f64 {} {} bit for bit: worst absolute error over {n} comparisons (compiled arrays, structure, Newton and CG solutions, steps, trajectories, converged optimum, settle-run frames) = {worst:e}{}; largest solver_niter difference {niter_diff}",
        which.name(),
        v.name(),
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
        "{} {}: not bit for bit MuJoCo's, worst at {at}",
        which.name(),
        v.name()
    );
}

macro_rules! exact_tests {
    ($($name:ident: $which:expr;)*) => {
        $(
            #[test]
            fn $name() {
                for v in CONES {
                    exact($which, v);
                }
            }
        )*
    };
}

exact_tests! {
    the_sphere_scene_is_bit_for_bit_mujocos: TWhich::Sphere;
    the_box_scene_is_bit_for_bit_mujocos: TWhich::Box;
    the_stack_is_bit_for_bit_mujocos: TWhich::Stack;
    the_capsules_are_bit_for_bit_mujocos: TWhich::Capsules;
    the_pile_is_bit_for_bit_mujocos: TWhich::Pile;
    the_humanoid_with_contacts_is_bit_for_bit_mujocos: TWhich::Humanoid;
    the_zoo_is_bit_for_bit_mujocos: TWhich::Zoo;
}

/// The pile's last second (steps 2161 to 2400 of its 2,400): the deepest contact over every step,
/// the separating-axis penetrations of the ten recorded frames (the engine comparison's metric,
/// ours computed on our poses by the Rust copy, MuJoCo's by the generator's numpy copy: equal to
/// 1e-12, the two being different implementations of one formula, not engines) and the speed at
/// 10 s. The engine figures (the deepest contact, from the contacts of every step) are exact.
#[test]
fn the_piles_last_second_figures_are_mujocos() {
    for v in CONES {
        let g = tgolden(TWhich::Pile, v);
        let settle = &g["settle"];
        let total = settle["steps"].as_u64().unwrap() as usize;
        let frames = settle["frames"].as_array().unwrap();
        let frame_steps: Vec<usize> = frames
            .iter()
            .map(|fr| fr["step"].as_u64().unwrap() as usize)
            .collect();
        assert_eq!(frames.len(), 10);
        let mut c = tcompile::<f64>(TWhich::Pile, v);
        c.model.timestep = f(&settle["timestep"]);
        let m = &c.model;
        let mut d = sim_physics::Data::new(m);
        let (mut depth, mut sat_bb, mut sat_floor) = (0.0f64, 0.0f64, 0.0f64);
        let mut sat_errs = Vec::new();
        for n in 1..=total {
            sim_physics::step(m, &mut d);
            assert_eq!(d.warning_collision_overflow, 0);
            if n > total - 240 {
                for i in 0..d.ncon {
                    depth = depth.max(-d.contact_dist[i]);
                }
            }
            if let Some(k) = frame_steps.iter().position(|&s| s == n) {
                let qpos = scene_qpos(m, &d.qpos);
                let (pos, quat) = pile_poses_scene(&qpos);
                let (bb, fl, _, _) = box_pile_penetration(&pos, &quat, 0.1);
                sat_bb = sat_bb.max(bb);
                sat_floor = sat_floor.max(fl);
                sat_errs.push((bb - f(&frames[k]["sat"]["box_box_max"])).abs());
                sat_errs.push((fl - f(&frames[k]["sat"]["box_floor_max"])).abs());
            }
        }
        let speed = (0..m.nbody - 1)
            .map(|b| {
                let q = &d.qvel[6 * b..6 * b + 3];
                (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt()
            })
            .fold(0.0f64, f64::max);
        let (mj_depth, mj_speed) = (
            f(&settle["last_second_max_neg_dist"]),
            f(&settle["end_max_speed"]),
        );
        let sat_err = sat_errs.iter().cloned().fold(0.0f64, f64::max);
        println!(
            "MEASURED f64 pile {} last second: deepest contact {:e} m (MuJoCo {:e}), SAT box-box {:e} m (MuJoCo {:e}), SAT box-floor {:e} m (MuJoCo {:e}), speed at 10 s {:e} m/s (MuJoCo {:e}); worst SAT difference over the ten frames {sat_err:e}",
            v.name(),
            depth,
            mj_depth,
            sat_bb,
            f(&settle["sat_max_box_box_last_second"]),
            sat_floor,
            f(&settle["sat_max_box_floor_last_second"]),
            speed,
            mj_speed
        );
        assert_eq!(depth, mj_depth, "pile {}: the deepest contact", v.name());
        assert!(
            sat_err <= 1e-12,
            "pile {}: SAT metric {sat_err:e}",
            v.name()
        );
        assert!(
            (speed - mj_speed).abs() <= 1e-14 * mj_speed.max(1e-300).max(1e-12),
            "pile {}: speed {speed:e} vs {mj_speed:e}",
            v.name()
        );
    }
}

/// The inertial frames (`body_sameframe`, `contacts_inertial_frames.rs`): every array of the
/// frames, `M`, `qfrc_bias`, `qacc` and one Euler and one RK4 step equal MuJoCo's to the bit.
#[test]
fn the_inertial_frames_are_bit_for_bit_mujocos() {
    let c = inertial_model::<f64>();
    let g = inertial_golden();
    let report = inertial_report(&c.model, &g, 0.0, 0.0);
    report.print("f64", "sameframe (exact)");
    assert_eq!(
        report.max_abs(),
        0.0,
        "not bit for bit: {:?}",
        report.failures()
    );
}

/// The structure-only variants of the zoo (midphase off, parent filter off).
#[test]
fn the_zoos_flag_variants_are_bit_for_bit_mujocos() {
    for v in [Variant::Midphase, Variant::Filterparent] {
        let c = tcompile::<f64>(TWhich::Zoo, v);
        let g = tgolden(TWhich::Zoo, v);
        let (report, notes) = run_structure(&c, &g, None);
        assert!(notes.is_empty(), "{notes:?}");
        println!(
            "MEASURED f64 zoo {} bit for bit: worst absolute error of the structure over {} states = {:e}",
            v.name(),
            tstates(&g).len(),
            report.max_abs()
        );
        assert_eq!(report.max_abs(), 0.0, "zoo {}", v.name());
    }
}

/// The collider sweep: every contact of every pose, every field.
#[test]
fn the_collider_sweep_is_bit_for_bit_mujocos() {
    let g = read_json(&fixtures().join("contact_pairs_golden.json"));
    let (mut cases, mut contacts) = (0usize, 0usize);
    for (name, entry) in g["colliders"].as_object().unwrap() {
        for variant in ["plain", "mg"] {
            let o = run_sweep_variant(&entry["variants"][variant], None);
            assert!(
                o.integer_failures.is_empty(),
                "{name} {variant}: {:?}",
                o.integer_failures
            );
            assert_eq!(o.report.max_abs(), 0.0, "{name} {variant}");
            cases += o.cases;
            contacts += o.contacts;
        }
    }
    println!(
        "MEASURED f64 collider sweep bit for bit: worst absolute error 0e0 over {cases} poses and {contacts} contacts of 11 colliders"
    );
}
