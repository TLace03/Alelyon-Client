//! The bounds a GPU port sizes its dispatches and its watchdog (TDR) budget from, against the
//! worst counts measured on every contact fixture (phase 1c-ii), in `f64` AND `f32` (the review
//! found the table measured in `f64` only, though the GPU runs `f32`, where the iteration counts
//! differ: the pile's CG runs to its iteration cap).
//!
//! Every loop of the step is bounded by a model constant or an option, and none depends on the
//! data otherwise:
//!
//! | quantity | bound |
//! |---|---|
//! | contacts per step | `ncon_max`, the sum of the per-pair maxima of the candidate list |
//! | constraint rows | `nefc_max` (limits and friction loss, plus `2 (condim - 1)` or `condim` rows per contact slot) |
//! | solver iterations of one solve | `opt.iterations` |
//! | line-search evaluations of one iteration | `opt.ls_iterations + 2` |
//! | rank-one Cholesky updates PERFORMED in one iteration | `nefc_max` plus the sum over slots of `condim` (a row changing zone, and a cone contact's `dim` updates) |
//! | MuJoCo's `nupdate` of one iteration (a refactor sets it to `nefc` plus the cone updates) | the same |
//! | rank-loss refactorisations of one solve | `opt.iterations` (at most one per iteration) |
//!
//! A rank-loss refactorisation is the expensive iteration: a full `J' D J` rebuild
//! (`O(nefc nv^2)`) and Cholesky factorisation (`O(nv^3)`) in place of the incremental updates
//! (`O(nv^2)` each). `SolverStat::nupdate_max` cannot show it, because MuJoCo's semantics
//! overwrite `nupdate` with the row count on a refactor (so it equals the `nefc` peak in nearly
//! every Newton row); `nrank1_max` counts the rank-one updates actually done and `nrefactor` the
//! refactorisations, so the worst iteration of a solve is bounded by `nrank1_max` updates, plus
//! one refactorisation when `nrefactor > 0`.
//!
//! The test runs every golden state with both solvers and 1,000 steps from the model's own initial
//! pose with both cones and both solvers, in both precisions, records the worst counts, prints them
//! next to the bounds (`MEASURED bounds ...`) and checks that none exceeds its bound. `README.md`
//! carries the table.

mod common;

use common::contacts::*;
use sim_physics::{Data, Model, PrimalSolver, Real, step};

/// The worst counts seen so far.
#[derive(Default, Clone, Copy)]
struct Worst {
    niter: usize,
    neval: usize,
    nupdate: usize,
    nrank1: usize,
    nrefactor: usize,
    ncon: usize,
    nefc: usize,
}

impl Worst {
    fn see<R: Real>(&mut self, d: &Data<R>) {
        self.niter = self.niter.max(d.solver_niter);
        self.neval = self.neval.max(d.solver_stat.neval_max);
        self.nupdate = self.nupdate.max(d.solver_stat.nupdate_max);
        self.nrank1 = self.nrank1.max(d.solver_stat.nrank1_max);
        self.nrefactor = self.nrefactor.max(d.solver_stat.nrefactor);
        self.ncon = self.ncon.max(d.ncon);
        self.nefc = self.nefc.max(d.nefc);
    }
}

/// The bounds of a model.
fn bounds<R: Real>(m: &Model<R>) -> Worst {
    let dims: usize = m.candidates.iter().map(|c| c.slot_count * c.condim).sum();
    Worst {
        niter: m.opt.iterations,
        neval: m.opt.ls_iterations + 2,
        nupdate: m.nefc_max + dims,
        nrank1: m.nefc_max + dims,
        nrefactor: m.opt.iterations,
        ncon: m.ncon_max,
        nefc: m.nefc_max,
    }
}

fn measure<R: Real>(which: TWhich, v: Variant, solver: PrimalSolver) -> (Worst, Worst) {
    let c = tcompile::<R>(which, v);
    let g = tgolden(which, v);
    let mut model = c.model.clone();
    model.opt.solver = solver;
    let mut worst = Worst::default();
    // every golden state, forward
    for s in tstates(&g) {
        let mut d = tdata(&model, s);
        sim_physics::forward(&model, &mut d);
        assert_eq!(d.warning_collision_overflow, 0, "box-box overflow");
        worst.see(&d);
    }
    // and 1,000 steps from the initial pose
    let mut d = Data::new(&model);
    for _ in 0..1000 {
        step(&model, &mut d);
        assert_eq!(d.warning_collision_overflow, 0, "box-box overflow");
        worst.see(&d);
    }
    (worst, bounds(&model))
}

fn gate_in<R: Real>(which: TWhich, precision: &str) {
    for v in CONES {
        for solver in [PrimalSolver::Newton, PrimalSolver::Cg] {
            let (w, b) = measure::<R>(which, v, solver);
            println!(
                "MEASURED bounds {precision} {} {} {solver:?}: solver_niter {} (bound {}), line-search evaluations {} (bound {}), nupdate {} (bound {}), rank-one updates performed {} (bound {}), rank-loss refactorisations {} (bound {}), ncon {} (bound {}), nefc {} (bound {})",
                which.name(),
                v.name(),
                w.niter,
                b.niter,
                w.neval,
                b.neval,
                w.nupdate,
                b.nupdate,
                w.nrank1,
                b.nrank1,
                w.nrefactor,
                b.nrefactor,
                w.ncon,
                b.ncon,
                w.nefc,
                b.nefc
            );
            let at = format!("{precision} {} {} {solver:?}", which.name(), v.name());
            assert!(w.niter <= b.niter, "{at}: iterations");
            assert!(w.neval <= b.neval, "{at}: line-search evaluations");
            assert!(w.nupdate <= b.nupdate, "{at}: nupdate");
            assert!(w.nrank1 <= b.nrank1, "{at}: rank-one updates");
            assert!(w.nrefactor <= b.nrefactor, "{at}: refactorisations");
            assert!(w.ncon <= b.ncon && w.nefc <= b.nefc, "{at}: sizes");
            if solver == PrimalSolver::Newton {
                assert!(w.nupdate > 0, "{at}: Newton factors the Hessian");
            } else {
                assert_eq!(w.nupdate, 0, "{at}: CG has no Hessian");
                assert_eq!((w.nrank1, w.nrefactor), (0, 0), "{at}: CG has no Hessian");
            }
        }
    }
}

fn gate(which: TWhich) {
    gate_in::<f64>(which, "f64");
    gate_in::<f32>(which, "f32");
}

macro_rules! bound_tests {
    ($($name:ident: $which:expr;)*) => {
        $(
            #[test]
            fn $name() {
                gate($which);
            }
        )*
    };
}

bound_tests! {
    the_sphere_scene_stays_within_its_bounds: TWhich::Sphere;
    the_box_scene_stays_within_its_bounds: TWhich::Box;
    the_stack_stays_within_its_bounds: TWhich::Stack;
    the_capsules_stay_within_their_bounds: TWhich::Capsules;
    the_pile_stays_within_its_bounds: TWhich::Pile;
    the_humanoid_stays_within_its_bounds: TWhich::Humanoid;
    the_zoo_stays_within_its_bounds: TWhich::Zoo;
}
