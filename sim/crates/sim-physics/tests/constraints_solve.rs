//! The Newton and CG solvers against MuJoCo 3.14.0 (phase 1c-i).
//!
//! Three comparisons, per state of each model:
//! - **At the model's own tolerance** (`f64`): `qacc` and `efc_force` of each solver
//!   within 1e-9 relative to the array's largest absolute value (floor 1e-12) of
//!   MuJoCo's run from the same warmstart. A faithful port follows the same
//!   iterates, so `solver_niter` is also compared and reported, and so are the zones
//!   of the rows (`efc_state`) and `qfrc_constraint`.
//! - **The converged optimum** (`f64`): our Newton solve with `tolerance = 1e-15`,
//!   `iterations = 1000`, `ls_iterations = 100` within 1e-10 relative of MuJoCo's
//!   converged `qacc` and `efc_force` (the cost is strictly convex, so the optimum is
//!   unique).
//! - **`f32`, measured**: `qacc` against MuJoCo's converged optimum (gate: relative
//!   error at most 1e-3), and Newton's iteration count in `f32` against `f64`. These
//!   are the GPU port's starting expectation.
//!
//! Every error is printed as a `MEASURED` line.

mod common;

use common::cons::*;
use common::*;

const SOLVE_RTOL: f64 = 1e-9;
const SOLVE_FLOOR: f64 = 1e-12;
const CONVERGED_RTOL: f64 = 1e-10;

struct Row {
    state: usize,
    niter_ours: usize,
    niter_mujoco: usize,
    zones_equal: bool,
    /// The relative error of `qacc` and of `efc_force` in this state.
    qacc_rel: f64,
    force_rel: f64,
}

/// One solver on every state of one model: the report of `qacc`, `efc_force` and
/// `qfrc_constraint`, and the per-state iteration counts.
fn check_solver(which: CWhich, solver: &str) -> (Report, Vec<Row>) {
    let c = ccompile::<f64>(which);
    let g = cgolden(which);
    let mut report = Report::default();
    let mut rows = Vec::new();
    for (k, s) in states(&g).iter().enumerate() {
        let d = run_forward(&c.model, s, solver_of(solver));
        let r = &s[solver];
        let n = d.nefc;
        rec(
            &mut report,
            "qacc",
            cmp(&widen(&d.qacc), &farr(&r["qacc"])),
            SOLVE_RTOL,
            SOLVE_FLOOR,
        );
        rec(
            &mut report,
            "efc_force",
            cmp(&head(&d.efc_force, n), &farr(&r["efc_force"])),
            SOLVE_RTOL,
            SOLVE_FLOOR,
        );
        rec(
            &mut report,
            "qfrc_constraint",
            cmp(&widen(&d.qfrc_constraint), &farr(&r["qfrc_constraint"])),
            SOLVE_RTOL,
            SOLVE_FLOOR,
        );
        let qacc_rel = cmp(&widen(&d.qacc), &farr(&r["qacc"])).rel();
        let force_rel = cmp(&head(&d.efc_force, n), &farr(&r["efc_force"])).rel();
        let zones: Vec<i64> = d.efc_state[..n].iter().map(|&z| state_code(z)).collect();
        rows.push(Row {
            state: k,
            niter_ours: d.solver_niter,
            niter_mujoco: r["solver_niter"].as_u64().unwrap() as usize,
            zones_equal: zones == ints(&r["efc_state"]),
            qacc_rel,
            force_rel,
        });
    }
    (report, rows)
}

fn gate(which: CWhich, solver: &str) {
    let (report, rows) = check_solver(which, solver);
    let label = format!("{} {solver}", which.name());
    report.print("f64", &label);
    let equal = rows
        .iter()
        .filter(|r| r.niter_ours == r.niter_mujoco)
        .count();
    for r in &rows {
        println!(
            "MEASURED f64 {label} state {}: solver_niter ours={} mujoco={} {} efc_state {}; qacc rel {:e}, efc_force rel {:e}",
            r.state,
            r.niter_ours,
            r.niter_mujoco,
            if r.niter_ours == r.niter_mujoco {
                "equal"
            } else {
                "DIFFERENT"
            },
            if r.zones_equal { "equal" } else { "DIFFERENT" },
            r.qacc_rel,
            r.force_rel,
        );
    }
    println!(
        "MEASURED f64 {label}: solver_niter equal in {equal} of {} states",
        rows.len()
    );
    assert!(
        report.all_within(),
        "{label}: out of tolerance: {:?}",
        report.failures()
    );
    // a faithful port ends in the same zones
    assert!(
        rows.iter().all(|r| r.zones_equal),
        "{label}: a row ended in another zone"
    );
}

#[test]
fn humanoid_newton_matches_mujoco() {
    gate(CWhich::Humanoid, "newton");
}

#[test]
fn humanoid_cg_matches_mujoco() {
    gate(CWhich::Humanoid, "cg");
}

#[test]
fn constrained_newton_matches_mujoco() {
    gate(CWhich::Constrained, "newton");
}

#[test]
fn constrained_cg_matches_mujoco() {
    gate(CWhich::Constrained, "cg");
}

#[test]
fn zoo_has_nothing_to_solve() {
    for solver in SOLVERS {
        let (report, rows) = check_solver(CWhich::Zoo, solver);
        report.print("f64", &format!("zoo {solver}"));
        assert!(report.all_within());
        assert!(
            rows.iter()
                .all(|r| r.niter_ours == 0 && r.niter_mujoco == 0)
        );
    }
}

/// Our Newton solve at the tightest settings against MuJoCo's: the unique optimum.
fn check_converged(which: CWhich) -> Report {
    let mut c = ccompile::<f64>(which);
    let g = cgolden(which);
    let settings = &g["converged_settings"];
    c.model.opt.tolerance = f(&settings["tolerance"]);
    c.model.opt.iterations = settings["iterations"].as_u64().unwrap() as usize;
    c.model.opt.ls_iterations = settings["ls_iterations"].as_u64().unwrap() as usize;
    let mut report = Report::default();
    for s in states(&g) {
        let d = run_forward(&c.model, s, sim_physics::PrimalSolver::Newton);
        let r = &s["converged"];
        rec(
            &mut report,
            "qacc",
            cmp(&widen(&d.qacc), &farr(&r["qacc"])),
            CONVERGED_RTOL,
            SOLVE_FLOOR,
        );
        rec(
            &mut report,
            "efc_force",
            cmp(&head(&d.efc_force, d.nefc), &farr(&r["efc_force"])),
            CONVERGED_RTOL,
            SOLVE_FLOOR,
        );
    }
    report
}

#[test]
fn the_converged_solve_reaches_mujocos_optimum() {
    for which in CMODELS {
        let report = check_converged(which);
        report.print("f64", &format!("{} converged optimum", which.name()));
        assert!(
            report.all_within(),
            "{}: out of tolerance: {:?}",
            which.name(),
            report.failures()
        );
    }
}

/// The optimum is where the gradient of the primal cost vanishes: at our converged
/// solution `M (qacc - qacc_smooth) = qfrc_constraint`, and `qfrc_constraint` is
/// `J' efc_force`, to the precision of the solve. An identity of our own solution,
/// independent of MuJoCo.
#[test]
fn the_converged_solution_satisfies_the_optimality_condition() {
    for which in [CWhich::Humanoid, CWhich::Constrained] {
        let mut c = ccompile::<f64>(which);
        let g = cgolden(which);
        c.model.opt.tolerance = 1e-15;
        c.model.opt.iterations = 1000;
        c.model.opt.ls_iterations = 100;
        let nv = c.model.nv;
        let mut worst = 0.0f64;
        for s in states(&g) {
            let d = run_forward(&c.model, s, sim_physics::PrimalSolver::Newton);
            // M (qacc - qacc_smooth) - J' force
            let mut scale = 0.0f64;
            let mut err = 0.0f64;
            for i in 0..nv {
                let mut ma = 0.0;
                for j in 0..nv {
                    ma += d.qm[i * nv + j] * (d.qacc[j] - d.qacc_smooth[j]);
                }
                let jf: f64 = (0..d.nefc)
                    .map(|r| d.efc_j[r * nv + i] * d.efc_force[r])
                    .sum();
                err = err.max((ma - jf).abs());
                scale = scale.max(ma.abs()).max(jf.abs());
            }
            worst = worst.max(err / scale.max(1e-12));
        }
        println!(
            "MEASURED f64 {} optimality: max |M (qacc - qacc_smooth) - J' f| / max = {worst:e}",
            which.name()
        );
        assert!(worst < 1e-9, "{}: {worst:e}", which.name());
    }
}

fn f32_run(which: CWhich) {
    let c64 = ccompile::<f64>(which);
    let c32 = ccompile::<f32>(which);
    let g = cgolden(which);
    let mut worst = 0.0f64;
    let mut worst_force = 0.0f64;
    for (k, s) in states(&g).iter().enumerate() {
        let d32 = run_forward(&c32.model, s, sim_physics::PrimalSolver::Newton);
        let d64 = run_forward(&c64.model, s, sim_physics::PrimalSolver::Newton);
        let r = &s["converged"];
        let e = cmp(&widen(&d32.qacc), &farr(&r["qacc"]));
        let ef = cmp(&head(&d32.efc_force, d32.nefc), &farr(&r["efc_force"]));
        worst = worst.max(e.rel());
        worst_force = worst_force.max(ef.rel());
        println!(
            "MEASURED f32 {} newton state {k}: qacc rel error vs MuJoCo's optimum {:e}, efc_force {:e}; solver_niter f32={} f64={}; final scaled gradient f32={:e} f64={:e}",
            which.name(),
            e.rel(),
            ef.rel(),
            d32.solver_niter,
            d64.solver_niter,
            d32.solver_stat.gradient,
            d64.solver_stat.gradient
        );
        assert!(
            e.rel() <= 1e-3,
            "{} state {k}: f32 qacc relative error {:e} exceeds 1e-3",
            which.name(),
            e.rel()
        );
    }
    println!(
        "MEASURED f32 {} newton: worst qacc relative error {worst:e}, worst efc_force relative error {worst_force:e}",
        which.name()
    );
    // and CG, in f32 and in f64 (measured, not gated: the conjugate-gradient solver stops at
    // the model's tolerance, which leaves it further from the optimum than Newton, whatever
    // the precision; the f64 figure, which is MuJoCo's own CG bit for bit, says how much of
    // the f32 figure is the termination and how much the precision)
    let mut worst_cg = 0.0f64;
    let mut worst_cg64 = 0.0f64;
    for s in states(&g) {
        let reference = farr(&s["converged"]["qacc"]);
        let d = run_forward(&c32.model, s, sim_physics::PrimalSolver::Cg);
        worst_cg = worst_cg.max(cmp(&widen(&d.qacc), &reference).rel());
        let d = run_forward(&c64.model, s, sim_physics::PrimalSolver::Cg);
        worst_cg64 = worst_cg64.max(cmp(&widen(&d.qacc), &reference).rel());
    }
    println!(
        "MEASURED f32 {} cg: worst qacc relative error vs MuJoCo's optimum {worst_cg:e} (f64 cg at the same tolerance: {worst_cg64:e})",
        which.name()
    );
}

#[test]
fn humanoid_f32_is_measured_against_the_optimum() {
    f32_run(CWhich::Humanoid);
}

#[test]
fn constrained_f32_is_measured_against_the_optimum() {
    f32_run(CWhich::Constrained);
}

/// The CG path, iterate by iterate, from the first state of each model: MuJoCo's `qacc`
/// after 0, 1, 2, ... iterations (the golden file's `cg_iterates`) against ours. CG is
/// not self-correcting, so a difference between our first iterate and MuJoCo's would grow
/// along the path (the humanoid's state 0 runs to the iteration cap of 100); this test
/// follows the whole path and gates every iterate at 1e-12 relative: the port does not
/// merely end where MuJoCo ends, it goes where MuJoCo goes.
#[test]
fn cg_follows_mujocos_path_iterate_by_iterate() {
    for which in [CWhich::Humanoid, CWhich::Constrained] {
        let c = ccompile::<f64>(which);
        let g = cgolden(which);
        let s = &states(&g)[0];
        let iterates = s["cg_iterates"].as_array().expect("cg_iterates of state 0");
        let smooth = {
            let d = run_forward(&c.model, s, sim_physics::PrimalSolver::Cg);
            cmp(&widen(&d.qacc_smooth), &farr(&s["qacc_smooth"])).rel()
        };
        let mut worst = 0.0f64;
        let mut at = 0usize;
        let mut first = 0.0f64;
        for (k, mj) in iterates.iter().enumerate() {
            let mut m = c.model.clone();
            m.opt.solver = sim_physics::PrimalSolver::Cg;
            m.opt.iterations = k;
            let mut d = cdata(&m, s);
            sim_physics::forward(&m, &mut d);
            let e = cmp(&widen(&d.qacc), &farr(mj)).rel();
            if k == 1 {
                first = e;
            }
            if e > worst {
                worst = e;
                at = k;
            }
            if [0, 1, 2, 5, 10, 20, 40, 60, 80, 100].contains(&k) {
                println!(
                    "MEASURED f64 {} cg iterate {k}: qacc relative error vs MuJoCo's iterate {e:e}",
                    which.name()
                );
            }
        }
        println!(
            "MEASURED f64 {} cg path: qacc_smooth relative error {smooth:e}; iterate 1 {first:e}; worst over {} iterates {worst:e} (at iteration {at})",
            which.name(),
            iterates.len()
        );
        assert!(smooth < 1e-12, "{} qacc_smooth {smooth:e}", which.name());
        assert!(first < 1e-12, "{} iterate 1 {first:e}", which.name());
        assert!(worst < 1e-12, "{} worst iterate {worst:e}", which.name());
    }
}
