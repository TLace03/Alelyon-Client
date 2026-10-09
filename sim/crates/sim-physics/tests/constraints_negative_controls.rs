//! Negative controls of the constraint comparison: it must FAIL when the engine is wrong.
//!
//! As in phase 1b, each control first runs the very comparison the parity tests use on the
//! unfaulted engine (the positive control, which must pass: through the public API and
//! through the fault machinery with no fault set) and then on an engine with one piece
//! broken by test-only fault injection (`sim_physics::faults::Faults`, hidden from the
//! public API). The comparison is the constraint structure (`nefc`, `efc_type`, `efc_id`
//! exactly; every array to 1e-12), the solution of the Newton and CG solvers at the model's
//! tolerance (1e-9) and the tendon kinematics. Each fault must fail it, and must fail where
//! the fault is, not everywhere:
//!
//! - the sign of `efc_aref` flipped: the reference acceleration and what depends on it (the
//!   solution), but not the Jacobian, the positions, `diagApprox`, `R`, `D` or `KBIP`;
//! - the friction-loss rows dropped: which rows exist (and so every per-row array and the
//!   solution), on the model that has friction loss; the humanoid has none, so the same
//!   fault must NOT be seen there (a control on the control);
//! - `diagApprox` replaced by 1: `diagApprox`, and `R` and `D` that follow from it, and the
//!   solution, but not the Jacobian, `aref` or `KBIP`;
//! - the exact line search replaced by a unit step: whether the converged-optimum test and the
//!   comparison at the model's tolerance catch it is MEASURED and reported (the converged
//!   optimum does not need the line search to be reached, so it may well not).

mod common;

use common::cons::*;
use common::*;
use sim_physics::faults::{Faults, forward_faulted};
use sim_physics::{Data, Model, PrimalSolver, Real};

const RTOL: f64 = 1e-12;
const FLOOR: f64 = 1e-14;
const SOLVE_RTOL: f64 = 1e-9;
const SOLVE_FLOOR: f64 = 1e-12;

/// Records `ours` against `reference`; arrays of different length are the worst error.
fn put(
    report: &mut Report,
    name: &str,
    ours: Vec<f64>,
    reference: Vec<f64>,
    rtol: f64,
    floor: f64,
) {
    let e = if ours.len() == reference.len() {
        cmp(&ours, &reference)
    } else {
        ErrStat {
            abs: f64::INFINITY,
            scale: 1.0,
        }
    };
    report.record(name, e, rtol, floor);
}

fn run_faulted<R: Real>(
    m: &Model<R>,
    s: &serde_json::Value,
    solver: PrimalSolver,
    faults: &Faults,
) -> Data<R> {
    let mut mm = m.clone();
    mm.opt.solver = solver;
    let mut d = cdata(&mm, s);
    forward_faulted(&mm, &mut d, faults);
    d
}

/// The structure and the solution of every state of `which`, with `faults`.
fn check(which: CWhich, solver: &str, faults: &Faults) -> Report {
    let c = ccompile::<f64>(which);
    let g = cgolden(which);
    let mut report = Report::default();
    for s in states(&g) {
        let d = run_faulted(&c.model, s, solver_of(solver), faults);
        let n = d.nefc;
        let nv = c.model.nv;
        // which rows exist, exactly
        put(
            &mut report,
            "nefc",
            vec![n as f64],
            vec![s["nefc"].as_f64().unwrap()],
            0.0,
            0.0,
        );
        let types: Vec<f64> = d.efc_type[..n]
            .iter()
            .map(|&t| type_code(t) as f64)
            .collect();
        put(
            &mut report,
            "efc_type",
            types,
            ints(&s["efc_type"]).iter().map(|&x| x as f64).collect(),
            0.0,
            0.0,
        );
        let ids: Vec<f64> = d.efc_id[..n].iter().map(|&i| i as f64).collect();
        put(
            &mut report,
            "efc_id",
            ids,
            ints(&s["efc_id"]).iter().map(|&x| x as f64).collect(),
            0.0,
            0.0,
        );
        let mut arr = |name: &str, ours: Vec<f64>, reference: Vec<f64>| {
            put(&mut report, name, ours, reference, RTOL, FLOOR);
        };
        arr("ten_length", widen(&d.ten_length), farr(&s["ten_length"]));
        arr("ten_J", widen(&d.ten_j), golden_rows(&s["ten_J"]));
        arr(
            "ten_velocity",
            widen(&d.ten_velocity),
            farr(&s["ten_velocity"]),
        );
        arr("efc_J", head(&d.efc_j, n * nv), golden_rows(&s["efc_J"]));
        arr("efc_pos", head(&d.efc_pos, n), farr(&s["efc_pos"]));
        arr("efc_margin", head(&d.efc_margin, n), farr(&s["efc_margin"]));
        arr(
            "efc_frictionloss",
            head(&d.efc_frictionloss, n),
            farr(&s["efc_frictionloss"]),
        );
        arr(
            "efc_diagApprox",
            head(&d.efc_diag_approx, n),
            farr(&s["efc_diagApprox"]),
        );
        arr("efc_R", head(&d.efc_r, n), farr(&s["efc_R"]));
        arr("efc_D", head(&d.efc_d, n), farr(&s["efc_D"]));
        arr(
            "efc_KBIP",
            head(&d.efc_kbip, 4 * n),
            golden_rows(&s["efc_KBIP"]),
        );
        arr("efc_aref", head(&d.efc_aref, n), farr(&s["efc_aref"]));
        arr("efc_vel", head(&d.efc_vel, n), farr(&s["efc_vel"]));
        arr(
            "qacc_smooth",
            widen(&d.qacc_smooth),
            farr(&s["qacc_smooth"]),
        );
        // the solution at the model's tolerance
        let r = &s[solver];
        put(
            &mut report,
            "qacc",
            widen(&d.qacc),
            farr(&r["qacc"]),
            SOLVE_RTOL,
            SOLVE_FLOOR,
        );
        put(
            &mut report,
            "efc_force",
            head(&d.efc_force, n),
            farr(&r["efc_force"]),
            SOLVE_RTOL,
            SOLVE_FLOOR,
        );
        put(
            &mut report,
            "qfrc_constraint",
            widen(&d.qfrc_constraint),
            farr(&r["qfrc_constraint"]),
            SOLVE_RTOL,
            SOLVE_FLOOR,
        );
    }
    report
}

/// A short name for the one fault that `faults` sets.
fn fault_name(faults: &Faults) -> &'static str {
    if faults.flip_aref_sign {
        "efc_aref sign flipped"
    } else if faults.drop_friction_rows {
        "friction-loss rows dropped"
    } else if faults.diag_approx_one {
        "diagApprox replaced by 1"
    } else if faults.unit_step_line_search {
        "unit-step line search"
    } else {
        "no fault"
    }
}

fn failures_of(which: CWhich, solver: &str, faults: Faults) -> Vec<String> {
    let r = check(which, solver, &faults);
    let name = fault_name(&faults);
    r.print("f64-fault", &format!("{} {solver} [{name}]", which.name()));
    let failed = r.failures();
    println!(
        "MEASURED fault [{name}] on {} with {solver}: {} arrays out of tolerance: {}",
        which.name(),
        failed.len(),
        if failed.is_empty() {
            "none (the fault is invisible here)".to_string()
        } else {
            failed.join(", ")
        }
    );
    failed
}

fn positive_control(which: CWhich, solver: &str) {
    let r = check(which, solver, &Faults::NONE);
    assert!(
        r.all_within(),
        "positive control ({} {solver}) failed: {:?}",
        which.name(),
        r.failures()
    );
    // and through the public API: the same numbers
    let c = ccompile::<f64>(which);
    let g = cgolden(which);
    for s in states(&g) {
        let d = run_forward(&c.model, s, solver_of(solver));
        let f = run_faulted(&c.model, s, solver_of(solver), &Faults::NONE);
        assert_eq!(
            d.qacc, f.qacc,
            "the fault machinery with no fault must not change a bit"
        );
        assert_eq!(d.efc_force[..d.nefc], f.efc_force[..f.nefc]);
    }
}

fn has(list: &[String], name: &str) -> bool {
    list.iter().any(|n| n == name)
}

#[test]
fn a_flipped_aref_sign_is_caught_where_aref_is() {
    for solver in SOLVERS {
        positive_control(CWhich::Constrained, solver);
        positive_control(CWhich::Humanoid, solver);
        for which in [CWhich::Constrained, CWhich::Humanoid] {
            let failed = failures_of(
                which,
                solver,
                Faults {
                    flip_aref_sign: true,
                    ..Faults::NONE
                },
            );
            assert!(
                has(&failed, "efc_aref"),
                "{} {solver}: {failed:?}",
                which.name()
            );
            assert!(
                has(&failed, "qacc"),
                "{} {solver}: {failed:?}",
                which.name()
            );
            assert!(
                has(&failed, "efc_force"),
                "{} {solver}: {failed:?}",
                which.name()
            );
            // nothing that does not depend on the reference acceleration is wrong
            for ok in [
                "nefc",
                "efc_type",
                "efc_id",
                "ten_length",
                "ten_J",
                "ten_velocity",
                "efc_J",
                "efc_pos",
                "efc_margin",
                "efc_frictionloss",
                "efc_diagApprox",
                "efc_R",
                "efc_D",
                "efc_KBIP",
                "efc_vel",
                "qacc_smooth",
            ] {
                assert!(
                    !has(&failed, ok),
                    "{} {solver}: {ok} must not depend on the fault: {failed:?}",
                    which.name()
                );
            }
        }
    }
}

#[test]
fn dropping_the_friction_loss_rows_is_caught_where_there_is_friction_loss() {
    for solver in SOLVERS {
        positive_control(CWhich::Constrained, solver);
        let failed = failures_of(
            CWhich::Constrained,
            solver,
            Faults {
                drop_friction_rows: true,
                ..Faults::NONE
            },
        );
        // which rows exist, and everything per row and the solution that depends on them
        for name in [
            "nefc",
            "efc_type",
            "efc_id",
            "efc_J",
            "efc_aref",
            "qacc",
            "efc_force",
        ] {
            assert!(has(&failed, name), "{solver}: {name} must fail: {failed:?}");
        }
        // what does not depend on the rows is untouched
        for ok in ["ten_length", "ten_J", "ten_velocity", "qacc_smooth"] {
            assert!(
                !has(&failed, ok),
                "{solver}: {ok} must not depend on the fault: {failed:?}"
            );
        }
        // the humanoid has no friction loss: the same fault is invisible there
        let failed = failures_of(
            CWhich::Humanoid,
            solver,
            Faults {
                drop_friction_rows: true,
                ..Faults::NONE
            },
        );
        assert!(
            failed.is_empty(),
            "the humanoid has no friction loss to drop: {failed:?}"
        );
    }
}

#[test]
fn replacing_diag_approx_by_one_is_caught_in_diag_approx_r_d_and_the_solution() {
    for solver in SOLVERS {
        for which in [CWhich::Constrained, CWhich::Humanoid] {
            positive_control(which, solver);
            let failed = failures_of(
                which,
                solver,
                Faults {
                    diag_approx_one: true,
                    ..Faults::NONE
                },
            );
            for name in ["efc_diagApprox", "efc_R", "efc_D", "qacc"] {
                assert!(
                    has(&failed, name),
                    "{} {solver}: {name} must fail: {failed:?}",
                    which.name()
                );
            }
            // the Jacobian, the references and the spring and damper do not use diagApprox
            for ok in [
                "nefc",
                "efc_type",
                "efc_id",
                "efc_J",
                "efc_pos",
                "efc_margin",
                "efc_frictionloss",
                "efc_KBIP",
                "efc_aref",
                "efc_vel",
                "qacc_smooth",
            ] {
                assert!(
                    !has(&failed, ok),
                    "{} {solver}: {ok} must not depend on the fault: {failed:?}",
                    which.name()
                );
            }
        }
    }
}

/// Worst relative error of `qacc` against MuJoCo's, with the line search replaced by a
/// unit step, at the model's tolerance (`reference` is the solver's golden block) or at the
/// converged settings (against the converged optimum). Returns the worst error over the
/// states and the largest solver-iteration difference from MuJoCo's.
fn unit_step_errors(which: CWhich, solver: &str, converged: bool) -> (f64, usize) {
    let mut c = ccompile::<f64>(which);
    let g = cgolden(which);
    if converged {
        let st = &g["converged_settings"];
        c.model.opt.tolerance = f(&st["tolerance"]);
        c.model.opt.iterations = st["iterations"].as_u64().unwrap() as usize;
        c.model.opt.ls_iterations = st["ls_iterations"].as_u64().unwrap() as usize;
    }
    let faults = Faults {
        unit_step_line_search: true,
        ..Faults::NONE
    };
    let mut worst = 0.0f64;
    let mut niter_diff = 0usize;
    for s in states(&g) {
        let d = run_faulted(&c.model, s, solver_of(solver), &faults);
        let reference = if converged {
            &s["converged"]
        } else {
            &s[solver]
        };
        worst = worst.max(cmp(&widen(&d.qacc), &farr(&reference["qacc"])).rel());
        if !converged {
            niter_diff = niter_diff.max(
                d.solver_niter
                    .abs_diff(reference["solver_niter"].as_u64().unwrap() as usize),
            );
        }
    }
    (worst, niter_diff)
}

#[test]
fn a_unit_step_instead_of_the_line_search_is_caught_at_the_models_tolerance_and_maybe_not_at_the_optimum()
 {
    // positive controls first: the unfaulted engine passes both comparisons
    for solver in SOLVERS {
        positive_control(CWhich::Constrained, solver);
    }
    let mut caught_at_tolerance = false;
    for which in [CWhich::Constrained, CWhich::Humanoid] {
        for solver in SOLVERS {
            let (at_tol, niter_diff) = unit_step_errors(which, solver, false);
            let (at_opt, _) = unit_step_errors(which, solver, true);
            let tol_caught = at_tol > SOLVE_RTOL || niter_diff > 0;
            let opt_caught = at_opt > 1e-10;
            println!(
                "MEASURED f64 unit-step line search, {} {solver}: at the model's tolerance qacc error {at_tol:e} and solver_niter differs by up to {niter_diff} -> {}; at the converged optimum qacc error {at_opt:e} -> {}",
                which.name(),
                if tol_caught { "CAUGHT" } else { "not caught" },
                if opt_caught { "CAUGHT" } else { "NOT caught" },
            );
            caught_at_tolerance |= tol_caught;
        }
    }
    // the comparison of the solution at the model's tolerance (qacc, efc_force and the
    // iteration count) sees it, on at least one model and solver
    assert!(
        caught_at_tolerance,
        "the unit step was not caught by any comparison"
    );
}
