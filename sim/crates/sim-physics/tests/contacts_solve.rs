//! The Newton and CG solvers with contacts against MuJoCo 3.14.0 (phase 1c-ii), both cones.
//!
//! Per state of each model, in `f64`:
//! - **At the model's own tolerance**: `qacc`, `efc_force` and `qfrc_constraint` of each solver
//!   within 1e-9 relative to the array's largest absolute value (floor 1e-12) of MuJoCo's run
//!   from the same warmstart; the contacts' `mu`, the cone Hessians `H` (MuJoCo fills `H` for
//!   the contacts in the cone zone of the Newton solver; a CG solve builds none) and
//!   `mj_contactForce` of every contact within the same tolerance; the zones of the rows
//!   (`efc_state`) exactly. A faithful port follows the same iterates, so `solver_niter` is also
//!   compared and reported.
//! - **The converged optimum**: our Newton solve with `tolerance = 1e-28`, `iterations = 1000`,
//!   `ls_iterations = 100` within 1e-10 relative of MuJoCo's converged `qacc` and `efc_force`.
//!   The cost is strictly convex, so the optimum is unique, and the golden reference IS the
//!   optimum: its stationarity residual `max |M (qacc - qacc_smooth) - J' f| / max` is at most
//!   1.2e-14 (recorded in the golden file, which the generator refuses above 1e-12, and
//!   re-measured here from our own `M` and `J`). The phase-1c-ii spec asked for `tolerance =
//!   1e-15`, which is not a converged optimum: MuJoCo 3.14.0's Newton also stops when the Newton
//!   decrement (a predicted COST improvement) is below `tolerance`, so a solve at 1e-15 leaves a
//!   residual of up to 1.8e-8 on an elliptic problem (the stack), and agreeing with it to 1e-10
//!   was agreeing with MuJoCo's iterates, not with an optimum (found in the review; the golden
//!   file still records that old solve's residual as `residual_at_1e-15`). The comparison of
//!   iterates (the same path) is now `check_converged`'s, bit for bit in `contacts_exact.rs`; the
//!   comparison that does not depend on the path is
//!   `the_converged_optimum_does_not_depend_on_the_path`, which starts from another warmstart
//!   with another line-search tolerance and must reach the same point.
//! - **The CG path, iterate by iterate**, from zoo state 0, both cones.
//!
//! Every error is printed as a `MEASURED` line.

mod common;

use common::cons::*;
use common::contacts::*;
use common::*;
use sim_physics::{PrimalSolver, contact_force};

const SOLVE_RTOL: f64 = 1e-9;
const SOLVE_FLOOR: f64 = 1e-12;
const CONVERGED_RTOL: f64 = 1e-10;

struct Row {
    state: usize,
    niter_ours: usize,
    niter_mujoco: usize,
    zones_equal: bool,
}

/// One solver on every state of one model and cone: the report, and the per-state counts.
fn check_solver(which: TWhich, v: Variant, solver: &str) -> (Report, Vec<Row>) {
    let c = tcompile::<f64>(which, v);
    let g = tgolden(which, v);
    let mut report = Report::default();
    let mut rows = Vec::new();
    for (k, s) in tstates(&g).iter().enumerate() {
        let d = tforward(&c.model, s, tsolver_of(solver));
        let r = &s[solver];
        let n = d.nefc;
        let ncon = d.ncon;
        let put = |report: &mut Report, name: &str, ours: Vec<f64>, reference: Vec<f64>| {
            rec(
                report,
                name,
                safe_cmp(&ours, &reference),
                SOLVE_RTOL,
                SOLVE_FLOOR,
            );
        };
        put(&mut report, "qacc", widen(&d.qacc), farr(&r["qacc"]));
        put(
            &mut report,
            "efc_force",
            thead(&d.efc_force, n),
            farr(&r["efc_force"]),
        );
        put(
            &mut report,
            "qfrc_constraint",
            widen(&d.qfrc_constraint),
            farr(&r["qfrc_constraint"]),
        );
        put(
            &mut report,
            "contact_mu",
            thead(&d.contact_mu, ncon),
            farr(&r["contact_mu"]),
        );
        put(
            &mut report,
            "contact_H",
            thead(&d.contact_h, 36 * ncon),
            trows(&r["contact_H"]),
        );
        let forces: Vec<f64> = (0..ncon)
            .flat_map(|i| contact_force(&c.model, &d, i).map(|x| x))
            .collect();
        put(
            &mut report,
            "contact_force",
            forces,
            trows(&r["contact_force"]),
        );

        let zones: Vec<i64> = d.efc_state[..n].iter().map(|&z| state_code(z)).collect();
        rows.push(Row {
            state: k,
            niter_ours: d.solver_niter,
            niter_mujoco: r["solver_niter"].as_u64().unwrap() as usize,
            zones_equal: zones == ints(&r["efc_state"]),
        });
    }
    (report, rows)
}

fn gate(which: TWhich, v: Variant, solver: &str) {
    let (report, rows) = check_solver(which, v, solver);
    let label = format!("{} {} {solver}", which.name(), v.name());
    report.print("f64", &label);
    let equal = rows
        .iter()
        .filter(|r| r.niter_ours == r.niter_mujoco)
        .count();
    let list: Vec<String> = rows
        .iter()
        .map(|r| {
            if r.niter_ours == r.niter_mujoco {
                format!("{}", r.niter_ours)
            } else {
                format!("{}!={}", r.niter_ours, r.niter_mujoco)
            }
        })
        .collect();
    println!(
        "MEASURED f64 {label}: solver_niter ours = MuJoCo's in {equal} of {} states ({})",
        rows.len(),
        list.join(", ")
    );
    assert!(
        report.all_within(),
        "{label}: out of tolerance: {:?}",
        report.failures()
    );
    // a faithful port ends in the same zones
    for r in &rows {
        assert!(
            r.zones_equal,
            "{label} state {}: a row ended in another zone",
            r.state
        );
    }
}

macro_rules! solve_tests {
    ($($name:ident: $which:expr, $variant:expr, $solver:expr;)*) => {
        $(
            #[test]
            fn $name() {
                gate($which, $variant, $solver);
            }
        )*
    };
}

solve_tests! {
    sphere_pyramidal_newton_matches_mujoco: TWhich::Sphere, Variant::Pyramidal, "newton";
    sphere_pyramidal_cg_matches_mujoco: TWhich::Sphere, Variant::Pyramidal, "cg";
    sphere_elliptic_newton_matches_mujoco: TWhich::Sphere, Variant::Elliptic, "newton";
    sphere_elliptic_cg_matches_mujoco: TWhich::Sphere, Variant::Elliptic, "cg";
    box_pyramidal_newton_matches_mujoco: TWhich::Box, Variant::Pyramidal, "newton";
    box_pyramidal_cg_matches_mujoco: TWhich::Box, Variant::Pyramidal, "cg";
    box_elliptic_newton_matches_mujoco: TWhich::Box, Variant::Elliptic, "newton";
    box_elliptic_cg_matches_mujoco: TWhich::Box, Variant::Elliptic, "cg";
    stack_pyramidal_newton_matches_mujoco: TWhich::Stack, Variant::Pyramidal, "newton";
    stack_pyramidal_cg_matches_mujoco: TWhich::Stack, Variant::Pyramidal, "cg";
    stack_elliptic_newton_matches_mujoco: TWhich::Stack, Variant::Elliptic, "newton";
    stack_elliptic_cg_matches_mujoco: TWhich::Stack, Variant::Elliptic, "cg";
    capsules_pyramidal_newton_matches_mujoco: TWhich::Capsules, Variant::Pyramidal, "newton";
    capsules_pyramidal_cg_matches_mujoco: TWhich::Capsules, Variant::Pyramidal, "cg";
    capsules_elliptic_newton_matches_mujoco: TWhich::Capsules, Variant::Elliptic, "newton";
    capsules_elliptic_cg_matches_mujoco: TWhich::Capsules, Variant::Elliptic, "cg";
    pile_pyramidal_newton_matches_mujoco: TWhich::Pile, Variant::Pyramidal, "newton";
    pile_pyramidal_cg_matches_mujoco: TWhich::Pile, Variant::Pyramidal, "cg";
    pile_elliptic_newton_matches_mujoco: TWhich::Pile, Variant::Elliptic, "newton";
    pile_elliptic_cg_matches_mujoco: TWhich::Pile, Variant::Elliptic, "cg";
    humanoid_pyramidal_newton_matches_mujoco: TWhich::Humanoid, Variant::Pyramidal, "newton";
    humanoid_pyramidal_cg_matches_mujoco: TWhich::Humanoid, Variant::Pyramidal, "cg";
    humanoid_elliptic_newton_matches_mujoco: TWhich::Humanoid, Variant::Elliptic, "newton";
    humanoid_elliptic_cg_matches_mujoco: TWhich::Humanoid, Variant::Elliptic, "cg";
    zoo_pyramidal_newton_matches_mujoco: TWhich::Zoo, Variant::Pyramidal, "newton";
    zoo_pyramidal_cg_matches_mujoco: TWhich::Zoo, Variant::Pyramidal, "cg";
    zoo_elliptic_newton_matches_mujoco: TWhich::Zoo, Variant::Elliptic, "newton";
    zoo_elliptic_cg_matches_mujoco: TWhich::Zoo, Variant::Elliptic, "cg";
}

/// Our Newton solve at the converged settings and the golden state's own warmstart against
/// MuJoCo's: the same path to the optimum (see `the_converged_optimum_does_not_depend_on_the_path`
/// for the comparison that is about the optimum alone).
fn check_converged(which: TWhich, v: Variant) -> Report {
    let mut c = tcompile::<f64>(which, v);
    let g = tgolden(which, v);
    let settings = &g["converged_settings"];
    c.model.opt.tolerance = f(&settings["tolerance"]);
    c.model.opt.iterations = settings["iterations"].as_u64().unwrap() as usize;
    c.model.opt.ls_iterations = settings["ls_iterations"].as_u64().unwrap() as usize;
    let mut report = Report::default();
    for s in tstates(&g) {
        let d = tforward(&c.model, s, PrimalSolver::Newton);
        let r = &s["converged"];
        rec(
            &mut report,
            "qacc",
            safe_cmp(&widen(&d.qacc), &farr(&r["qacc"])),
            CONVERGED_RTOL,
            SOLVE_FLOOR,
        );
        rec(
            &mut report,
            "efc_force",
            safe_cmp(&thead(&d.efc_force, d.nefc), &farr(&r["efc_force"])),
            CONVERGED_RTOL,
            SOLVE_FLOOR,
        );
    }
    report
}

#[test]
fn the_converged_solve_reaches_mujocos_optimum() {
    for which in TMODELS {
        for v in CONES {
            let report = check_converged(which, v);
            report.print(
                "f64",
                &format!("{} {} converged optimum", which.name(), v.name()),
            );
            assert!(
                report.all_within(),
                "{} {}: out of tolerance: {:?}",
                which.name(),
                v.name(),
                report.failures()
            );
        }
    }
}

/// The residual `max |M (qacc - qacc_smooth) - J' f| / max` of a solution `(qacc, f)` of the
/// state `d` (whose `M`, `J` and `qacc_smooth` are the ones the solution belongs to).
fn optimality_residual(d: &sim_physics::Data<f64>, nv: usize, qacc: &[f64], force: &[f64]) -> f64 {
    let mut scale = 0.0f64;
    let mut err = 0.0f64;
    for i in 0..nv {
        let ma: f64 = qacc
            .iter()
            .zip(&d.qacc_smooth)
            .enumerate()
            .map(|(j, (a, s))| d.qm[i * nv + j] * (a - s))
            .sum();
        let jf: f64 = (0..d.nefc).map(|r| d.efc_j[r * nv + i] * force[r]).sum();
        err = err.max((ma - jf).abs());
        scale = scale.max(ma.abs()).max(jf.abs());
    }
    err / scale.max(1e-12)
}

/// The converged settings of a golden file on `c`'s model.
fn with_converged_settings(c: &mut Compiled<f64>, g: &serde_json::Value) {
    let settings = &g["converged_settings"];
    c.model.opt.tolerance = f(&settings["tolerance"]);
    c.model.opt.iterations = settings["iterations"].as_u64().unwrap() as usize;
    c.model.opt.ls_iterations = settings["ls_iterations"].as_u64().unwrap() as usize;
}

/// The largest stationarity residual allowed of a converged solution, relative to the larger of
/// the two terms: the generator refuses a reference above it, and ours must meet it too.
const RESIDUAL_MAX: f64 = 1e-12;

/// The optimum is where the gradient of the primal cost vanishes: at the converged solution
/// `M (qacc - qacc_smooth) = J' efc_force`. Both solutions are held to it on every model and
/// cone, and to nothing weaker than `RESIDUAL_MAX`: ours (our `M`, `J` and solve), MuJoCo's
/// (evaluated with our `M` and `J`, which equal MuJoCo's, and as the golden file's own
/// recorded residual from MuJoCo's `M` and `J`; the two must agree, which checks this
/// function against the generator's). The residual the OLD reference (`tolerance 1e-15`) left,
/// recorded in the golden file as `residual_at_1e-15`, is printed beside them: it is why that
/// reference was not an optimum (the review's finding).
#[test]
fn the_converged_solution_satisfies_the_optimality_condition() {
    for which in TMODELS {
        for v in CONES {
            let mut c = tcompile::<f64>(which, v);
            let g = tgolden(which, v);
            with_converged_settings(&mut c, &g);
            let nv = c.model.nv;
            let (mut ours, mut theirs, mut recorded, mut old) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for s in tstates(&g) {
                let d = tforward(&c.model, s, PrimalSolver::Newton);
                ours = ours.max(optimality_residual(&d, nv, &d.qacc, &d.efc_force));
                let r = &s["converged"];
                let mine = optimality_residual(&d, nv, &farr(&r["qacc"]), &farr(&r["efc_force"]));
                let golden = f(&r["residual"]);
                // the two computations of MuJoCo's residual (here and in the generator) agree
                assert!(
                    (mine - golden).abs() <= RESIDUAL_MAX,
                    "{} {}: residual of MuJoCo's solution {mine:e} here, {golden:e} in the generator",
                    which.name(),
                    v.name()
                );
                theirs = theirs.max(mine);
                recorded = recorded.max(golden);
                old = old.max(f(&r["residual_at_1e-15"]));
            }
            println!(
                "MEASURED f64 {} {} optimality: max |M (qacc - qacc_smooth) - J' f| / max = {ours:e} for our converged solution, {theirs:e} for MuJoCo's (our M and J; the generator recorded {recorded:e}); the old reference at tolerance 1e-15 left {old:e}",
                which.name(),
                v.name()
            );
            assert!(
                ours <= RESIDUAL_MAX && theirs <= RESIDUAL_MAX,
                "{} {}: {ours:e} (ours), {theirs:e} (MuJoCo's)",
                which.name(),
                v.name()
            );
        }
    }
}

/// The optimum is where it is whatever the path to it: our Newton solve of every state from the
/// ZERO warmstart (not the golden state's) with a line-search tolerance of 0.1 (not 0.01), a
/// different sequence of iterates in every state, reaches MuJoCo's converged `qacc` and
/// `efc_force` to the same 1e-10, which is a statement about the optimum and not about the
/// iterates (the old reference, at tolerance 1e-15, could not have passed it on the elliptic
/// models: it was 1e-8 from the optimum).
#[test]
fn the_converged_optimum_does_not_depend_on_the_path() {
    for which in TMODELS {
        for v in CONES {
            let mut c = tcompile::<f64>(which, v);
            let g = tgolden(which, v);
            with_converged_settings(&mut c, &g);
            c.model.opt.ls_tolerance = 0.1;
            let mut report = Report::default();
            let (mut moved, mut total) = (0usize, 0usize);
            for s in tstates(&g) {
                let mut d = tdata(&c.model, s);
                d.qacc_warmstart.fill(0.0);
                sim_physics::forward(&c.model, &mut d);
                assert_eq!(d.warning_collision_overflow, 0, "box-box overflow");
                let r = &s["converged"];
                rec(
                    &mut report,
                    "qacc",
                    safe_cmp(&widen(&d.qacc), &farr(&r["qacc"])),
                    CONVERGED_RTOL,
                    SOLVE_FLOOR,
                );
                rec(
                    &mut report,
                    "efc_force",
                    safe_cmp(&thead(&d.efc_force, d.nefc), &farr(&r["efc_force"])),
                    CONVERGED_RTOL,
                    SOLVE_FLOOR,
                );
                // the path was another one (a different iteration count) in most states
                total += 1;
                moved +=
                    usize::from(d.solver_niter != r["solver_niter"].as_u64().unwrap() as usize);
            }
            report.print(
                "f64",
                &format!(
                    "{} {} converged optimum from the zero warmstart, ls_tolerance 0.1 ({moved} of {total} states took another number of iterations than MuJoCo's)",
                    which.name(),
                    v.name()
                ),
            );
            assert!(
                report.all_within(),
                "{} {}: out of tolerance: {:?}",
                which.name(),
                v.name(),
                report.failures()
            );
        }
    }
}

/// The CG path, iterate by iterate, from zoo state 0 with each cone: MuJoCo's `qacc` after
/// 0, 1, 2, ... iterations (the golden file's `cg_iterates`) against ours at 1e-12 relative.
#[test]
fn cg_follows_mujocos_path_iterate_by_iterate_with_contacts() {
    for v in CONES {
        let c = tcompile::<f64>(TWhich::Zoo, v);
        let g = tgolden(TWhich::Zoo, v);
        let s = &tstates(&g)[0];
        let iterates = s["cg_iterates"].as_array().expect("cg_iterates of state 0");
        let (mut worst, mut at) = (0.0f64, 0usize);
        for (k, mj) in iterates.iter().enumerate() {
            let mut m = c.model.clone();
            m.opt.solver = PrimalSolver::Cg;
            m.opt.iterations = k;
            let mut d = tdata(&m, s);
            sim_physics::forward(&m, &mut d);
            assert_eq!(d.warning_collision_overflow, 0, "box-box overflow");
            let e = safe_cmp(&widen(&d.qacc), &farr(mj)).rel();
            if e > worst {
                worst = e;
                at = k;
            }
            if [0, 1, 2, 5, 10, 20, 40, 60, 80, 100].contains(&k) {
                println!(
                    "MEASURED f64 zoo {} cg iterate {k}: qacc relative error vs MuJoCo's iterate {e:e}",
                    v.name()
                );
            }
        }
        println!(
            "MEASURED f64 zoo {} cg path: worst over {} iterates {worst:e} (at iteration {at})",
            v.name(),
            iterates.len()
        );
        assert!(worst < 1e-12, "zoo {} worst iterate {worst:e}", v.name());
    }
}
