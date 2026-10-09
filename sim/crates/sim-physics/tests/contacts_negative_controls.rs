//! Negative controls of the contact comparison (phase 1c-ii): it must FAIL when the engine is
//! wrong, and only where the fault acts.
//!
//! As in phases 1b and 1c-i, each control first runs the very comparison the parity tests use on
//! the unfaulted engine (the positive control, which must pass: through the public API and
//! through the fault machinery with no fault set) and then on an engine with one piece broken by
//! test-only fault injection (`sim_physics::faults::Faults`, hidden from the public API). The
//! comparison is the contact structure (the contact list, `nefc`, `efc_type`, `efc_id` exactly;
//! every array to 1e-12) and the solution of the Newton solver at the model's tolerance (1e-9),
//! on every state of the model. A pseudo-array `integers.contact_list` (and `integers.rows`)
//! counts the differences of the integer fields, so a failing integer comparison is named like
//! a failing float array. The faults:
//!
//! - **flipped contact normal**: the frame, `efc_J` and the solution fail on every model with
//!   contacts, and a resting sphere falls through the floor; what does not depend on the normal
//!   (`dist`, `includemargin`, the friction, `solref`, `solimp`, `efc_pos`, `efc_margin`,
//!   `diagApprox`, `R`, `D`, `KBIP`, `qacc_smooth`, the contact list's integers) stays right;
//! - **friction mixed by the minimum**: fails on the humanoid and the zoo, whose geoms have
//!   unequal friction; the sphere, box, stack, capsule and pile scenes, with equal friction,
//!   pass;
//! - **last box-box contact dropped**: fails on the scenes with box-box pairs (the stack, the pile,
//!   the zoo) and on the box-box sweep only;
//! - **impratio ignored**: fails on the zoo (impratio 5) only, in `efc_R`, `mu` and the solution;
//! - **`includemargin = margin - gap`**: fails on the zoo (margin and gap) only, and CONTACT BY
//!   CONTACT: a contact's `includemargin` differs from MuJoCo's exactly where its pair has a gap,
//!   its exclude flag can change only there, and every contact of a pair without a gap is
//!   exactly MuJoCo's (a per-state check is nearly vacuous: 39 of the zoo's 43 states have a
//!   contact with a margin);
//! - **nested order everywhere**: fails on the zoo, and only in the state with the contact-order
//!   case; the faulted engine reproduces the golden file MuJoCo wrote with its midphase off;
//! - **cone Hessian dropped**: on all seven models, after the positive controls of BOTH
//!   comparisons (the model's tolerance with `solver_niter`, and the converged settings): reported,
//!   whether the converged-optimum comparison catches it, and what does (the solution at the
//!   model's tolerance, or `solver_niter`), as 1c-i did for the unit step; the pyramidal cone,
//!   which has no cone Hessian, is bit for bit unchanged.

mod common;

use common::contacts::*;
use common::*;
use serde_json::Value;
use sim_physics::faults::{Faults, forward_faulted};
use sim_physics::{Data, PrimalSolver, Real};

/// What one comparison of a model with the oracle found.
struct Outcome {
    /// The worst error of every array.
    report: Report,
    /// The states whose contact list or rows (integers) differ.
    integer_states: Vec<usize>,
    /// Whether every constraint row ended in MuJoCo's zone, in every state.
    zones_equal: bool,
}

fn run_state<R: Real>(
    c: &Compiled<R>,
    s: &Value,
    solver: &str,
    faults: &Faults,
    report: &mut Report,
) -> (Data<R>, bool, bool) {
    let mut mm = c.model.clone();
    mm.opt.solver = tsolver_of(solver);
    let mut d = tdata(&mm, s);
    forward_faulted(&mm, &mut d, faults);
    assert_eq!(d.warning_collision_overflow, 0, "box-box overflow");
    let notes = record_structure(&mm, &d, s, report);
    let zones = record_solution(&mm, &d, s, solver, report);
    (d, notes.is_empty(), zones)
}

/// Every state of `which`/`v` against `golden` (the golden of `which`/`v` unless given).
fn check_against(c: &Compiled<f64>, golden: &Value, solver: &str, faults: &Faults) -> Outcome {
    let mut report = Report::default();
    let mut integer_states = Vec::new();
    let mut zones_equal = true;
    for (k, s) in tstates(golden).iter().enumerate() {
        let (_, integers_ok, zones) = run_state(c, s, solver, faults, &mut report);
        if !integers_ok {
            integer_states.push(k);
        }
        zones_equal &= zones;
    }
    Outcome {
        report,
        integer_states,
        zones_equal,
    }
}

fn check(which: TWhich, v: Variant, solver: &str, faults: &Faults) -> Outcome {
    let c = tcompile_faulted::<f64>(which, v, faults);
    let g = tgolden(which, v);
    check_against(&c, &g, solver, faults)
}

fn has(list: &[String], name: &str) -> bool {
    list.iter().any(|n| n == name)
}

/// The positive control: the unfaulted engine passes every comparison, through the public API
/// and through the fault machinery with no fault.
fn positive_control(which: TWhich, v: Variant, solver: &str) {
    let o = check(which, v, solver, &Faults::NONE);
    assert!(
        o.report.all_within() && o.integer_states.is_empty() && o.zones_equal,
        "positive control ({} {} {solver}) failed: {:?}",
        which.name(),
        v.name(),
        o.report.failures()
    );
    let c = tcompile::<f64>(which, v);
    let g = tgolden(which, v);
    for s in tstates(&g) {
        let a = tforward(&c.model, s, tsolver_of(solver));
        let mut mm = c.model.clone();
        mm.opt.solver = tsolver_of(solver);
        let mut b = tdata(&mm, s);
        forward_faulted(&mm, &mut b, &Faults::NONE);
        assert_eq!(
            a.qacc, b.qacc,
            "the fault machinery with no fault must not change a bit"
        );
        assert_eq!(a.ncon, b.ncon);
        assert_eq!(a.efc_force[..a.nefc], b.efc_force[..b.nefc]);
    }
}

fn describe(which: TWhich, v: Variant, solver: &str, name: &str, o: &Outcome) -> Vec<String> {
    o.report.print(
        "f64-fault",
        &format!("{} {} {solver} [{name}]", which.name(), v.name()),
    );
    let mut failed = o.report.failures();
    if !o.zones_equal {
        failed.push("efc_state".to_string());
    }
    println!(
        "MEASURED fault [{name}] on {} {} with {solver}: {} arrays out of tolerance, {} states with a different contact list or rows: {}",
        which.name(),
        v.name(),
        failed.len(),
        o.integer_states.len(),
        if failed.is_empty() {
            "none (the fault is invisible here)".to_string()
        } else {
            failed.join(", ")
        }
    );
    failed
}

/// The models of the contact golden files.
const ALL: [TWhich; 7] = TMODELS;

/// What a flipped normal does NOT change.
const NORMAL_FREE: [&str; 14] = [
    "integers.contact_list",
    "integers.rows",
    "contact.dist",
    "contact.pos",
    "contact.includemargin",
    "contact.friction",
    "contact.solref",
    "contact.solimp",
    "efc_pos",
    "efc_margin",
    "efc_diagApprox",
    "efc_R",
    "efc_KBIP",
    "qacc_smooth",
];

#[test]
fn a_flipped_contact_normal_is_caught_everywhere_there_are_contacts() {
    for which in ALL {
        for v in CONES {
            positive_control(which, v, "newton");
            let faults = Faults {
                flip_contact_normal: true,
                ..Faults::NONE
            };
            let o = check(which, v, "newton", &faults);
            let failed = describe(which, v, "newton", "contact normal flipped", &o);
            for must in ["contact.frame", "efc_J", "qacc"] {
                assert!(
                    has(&failed, must),
                    "{} {}: {must} must fail: {failed:?}",
                    which.name(),
                    v.name()
                );
            }
            for ok in NORMAL_FREE {
                assert!(
                    !has(&failed, ok),
                    "{} {}: {ok} must not depend on the normal's sign: {failed:?}",
                    which.name(),
                    v.name()
                );
            }
        }
    }
}

/// With the normal flipped the contact pushes the wrong way: the resting sphere is pulled into
/// the floor and falls through it.
#[test]
fn with_the_normal_flipped_a_resting_sphere_falls_through_the_floor() {
    let c = tcompile::<f64>(TWhich::Sphere, Variant::Pyramidal);
    let run = |faults: &Faults| -> f64 {
        let mut d = Data::new(&c.model);
        // the sphere "rest" rests at the XML's pose: touching, 1 mm into the floor so that the
        // contact is a constraint from the first step
        d.qpos[2] = 0.099;
        for _ in 0..1000 {
            sim_physics::faults::step_faulted(&c.model, &mut d, faults);
            assert_eq!(d.warning_collision_overflow, 0, "box-box overflow");
        }
        d.qpos[2]
    };
    let z_ok = run(&Faults::NONE);
    let z_flipped = run(&Faults {
        flip_contact_normal: true,
        ..Faults::NONE
    });
    println!(
        "MEASURED fault [contact normal flipped]: after 1,000 steps the resting sphere's centre is at z = {z_ok:.6} m (radius 0.1) with the normals right and at z = {z_flipped:.3} m with them flipped"
    );
    assert!((z_ok - 0.0996).abs() < 1e-3, "{z_ok}");
    assert!(
        z_flipped < 0.0,
        "the sphere should fall through the floor: {z_flipped}"
    );
}

#[test]
fn friction_mixed_by_the_minimum_is_caught_where_geoms_have_unequal_friction() {
    let faults = Faults {
        friction_mix_min: true,
        ..Faults::NONE
    };
    for v in CONES {
        // unequal friction: the humanoid (floor 1, body 0.7) and the zoo
        for which in [TWhich::Humanoid, TWhich::Zoo] {
            positive_control(which, v, "newton");
            let o = check(which, v, "newton", &faults);
            let failed = describe(which, v, "newton", "friction mixed by min", &o);
            for must in ["contact.friction", "qacc"] {
                assert!(
                    has(&failed, must),
                    "{} {}: {must} must fail: {failed:?}",
                    which.name(),
                    v.name()
                );
            }
            // nothing about the geometry or the order is wrong
            for ok in [
                "integers.contact_list",
                "integers.rows",
                "contact.dist",
                "contact.pos",
                "contact.frame",
                "efc_pos",
                "qacc_smooth",
            ] {
                assert!(
                    !has(&failed, ok),
                    "{} {}: {ok}: {failed:?}",
                    which.name(),
                    v.name()
                );
            }
        }
        // equal friction everywhere: the minimum is the maximum
        for which in [
            TWhich::Sphere,
            TWhich::Box,
            TWhich::Stack,
            TWhich::Capsules,
            TWhich::Pile,
        ] {
            let o = check(which, v, "newton", &faults);
            let failed = describe(which, v, "newton", "friction mixed by min", &o);
            assert!(
                failed.is_empty(),
                "{} {}: {failed:?}",
                which.name(),
                v.name()
            );
            assert!(o.integer_states.is_empty());
        }
    }
}

#[test]
fn a_dropped_last_box_box_contact_is_caught_on_box_box_pairs_only() {
    let faults = Faults {
        drop_last_boxbox_contact: true,
        ..Faults::NONE
    };
    for v in CONES {
        // the scenes with box-box pairs
        for which in [TWhich::Stack, TWhich::Pile, TWhich::Zoo] {
            positive_control(which, v, "newton");
            let o = check(which, v, "newton", &faults);
            let failed = describe(which, v, "newton", "last box-box contact dropped", &o);
            assert!(
                has(&failed, "integers.contact_list"),
                "{} {}: {failed:?}",
                which.name(),
                v.name()
            );
            assert!(
                has(&failed, "qacc"),
                "{} {}: {failed:?}",
                which.name(),
                v.name()
            );
            assert!(!o.integer_states.is_empty());
        }
        // the scenes without
        for which in [
            TWhich::Sphere,
            TWhich::Box,
            TWhich::Capsules,
            TWhich::Humanoid,
        ] {
            let o = check(which, v, "newton", &faults);
            let failed = describe(which, v, "newton", "last box-box contact dropped", &o);
            assert!(
                failed.is_empty(),
                "{} {}: {failed:?}",
                which.name(),
                v.name()
            );
        }
    }
    // the sweep: the box-box collider fails, the other ten pass
    let g = read_json(&fixtures().join("contact_pairs_golden.json"));
    for (name, entry) in g["colliders"].as_object().unwrap() {
        for variant in ["plain", "mg"] {
            let ok = run_sweep_variant(&entry["variants"][variant], None);
            assert!(
                ok.integer_failures.is_empty() && ok.report.all_within(),
                "positive control {name}"
            );
            let o = run_sweep_variant(&entry["variants"][variant], Some(&faults));
            let failed = !o.integer_failures.is_empty() || !o.report.all_within();
            println!(
                "MEASURED fault [last box-box contact dropped] on the sweep {name} {variant}: {} ({} cases differ)",
                if failed { "CAUGHT" } else { "not seen" },
                o.integer_failures.len()
            );
            assert_eq!(failed, name == "box_box", "{name} {variant}");
        }
    }
}

#[test]
fn ignoring_impratio_is_caught_where_impratio_is_not_one() {
    let faults = Faults {
        ignore_impratio: true,
        ..Faults::NONE
    };
    for v in CONES {
        // the zoo has impratio 5
        positive_control(TWhich::Zoo, v, "newton");
        let o = check(TWhich::Zoo, v, "newton", &faults);
        let failed = describe(TWhich::Zoo, v, "newton", "impratio ignored", &o);
        for must in ["efc_R", "contact_mu", "qacc"] {
            assert!(
                has(&failed, must),
                "zoo {}: {must} must fail: {failed:?}",
                v.name()
            );
        }
        // the geometry, the Jacobians and the references do not use impratio (`efc_diagApprox` is
        // MuJoCo's, readjusted at the end of `mj_makeImpedance` from the final `R`, so it does)
        for ok in [
            "integers.contact_list",
            "integers.rows",
            "contact.dist",
            "contact.frame",
            "efc_J",
            "efc_pos",
            "efc_margin",
            "efc_aref",
            "efc_vel",
            "qacc_smooth",
        ] {
            assert!(!has(&failed, ok), "zoo {}: {ok}: {failed:?}", v.name());
        }
        // impratio 1 everywhere else: ignoring it changes nothing
        for which in [
            TWhich::Sphere,
            TWhich::Box,
            TWhich::Stack,
            TWhich::Capsules,
            TWhich::Pile,
            TWhich::Humanoid,
        ] {
            let o = check(which, v, "newton", &faults);
            let failed = describe(which, v, "newton", "impratio ignored", &o);
            assert!(
                failed.is_empty(),
                "{} {}: {failed:?}",
                which.name(),
                v.name()
            );
        }
    }
}

#[test]
fn includemargin_minus_gap_is_caught_on_the_margin_and_gap_cases_only() {
    let faults = Faults {
        includemargin_minus_gap: true,
        ..Faults::NONE
    };
    for v in CONES {
        positive_control(TWhich::Zoo, v, "newton");
        let o = check(TWhich::Zoo, v, "newton", &faults);
        let failed = describe(TWhich::Zoo, v, "newton", "includemargin = margin - gap", &o);
        for must in ["contact.includemargin", "integers.contact_list", "qacc"] {
            assert!(
                has(&failed, must),
                "zoo {}: {must} must fail: {failed:?}",
                v.name()
            );
        }
        // CONTACT BY CONTACT (the review: a per-state check is nearly vacuous, since 39 of the
        // zoo's 43 states have a contact with a margin): a contact's `includemargin` differs
        // from MuJoCo's exactly where its pair has a gap (`margin - gap` is `margin` otherwise),
        // its `exclude` flag can differ only there, and every contact of a pair with no gap is
        // exactly MuJoCo's, in every state
        let g = tgolden(TWhich::Zoo, v);
        let c = tcompile::<f64>(TWhich::Zoo, v);
        let (mut with_gap, mut gapless, mut exclude_flips) = (0usize, 0usize, 0usize);
        for (k, s) in tstates(&g).iter().enumerate() {
            let mut mm = c.model.clone();
            mm.opt.solver = PrimalSolver::Newton;
            let mut d = tdata(&mm, s);
            forward_faulted(&mm, &mut d, &faults);
            let gc = s["contacts"].as_array().unwrap();
            assert_eq!(d.ncon, gc.len(), "state {k}: the fault changes no contact");
            for (i, cg) in gc.iter().enumerate() {
                let geoms = tints(&cg["geom"]);
                let cand = c
                    .model
                    .candidates
                    .iter()
                    .find(|q| q.g1 == geoms[0] as usize && q.g2 == geoms[1] as usize)
                    .expect("a candidate");
                let includemargin_differs = d.contact_includemargin[i] != f(&cg["includemargin"]);
                let exclude_differs =
                    i64::from(d.contact_exclude[i]) != cg["exclude"].as_i64().unwrap();
                assert_eq!(
                    includemargin_differs,
                    cand.gap > 0.0,
                    "zoo {} state {k} contact {i}: includemargin differs from MuJoCo's iff its pair has a gap (gap {:e})",
                    v.name(),
                    cand.gap
                );
                assert!(
                    !exclude_differs || cand.gap > 0.0,
                    "zoo {} state {k} contact {i}: an exclude flag changed on a pair with no gap",
                    v.name()
                );
                // nothing about the contact's place or frame changes
                assert_eq!(d.contact_dist[i], f(&cg["dist"]));
                if cand.gap > 0.0 {
                    with_gap += 1;
                    exclude_flips += usize::from(exclude_differs);
                } else {
                    gapless += 1;
                }
            }
        }
        println!(
            "MEASURED fault [includemargin = margin - gap] on the zoo {}, contact by contact: includemargin differs on all {with_gap} contacts of pairs with a gap and on none of the {gapless} contacts of pairs without one; {exclude_flips} of those contacts also change their exclude flag",
            v.name()
        );
        assert!(
            with_gap > 0 && gapless > 0 && exclude_flips > 0,
            "the zoo must have gap, gapless and flipping contacts"
        );
        // no margin and no gap anywhere else: the fault is invisible
        for which in [
            TWhich::Sphere,
            TWhich::Box,
            TWhich::Stack,
            TWhich::Capsules,
            TWhich::Pile,
            TWhich::Humanoid,
        ] {
            let o = check(which, v, "newton", &faults);
            let failed = describe(which, v, "newton", "includemargin = margin - gap", &o);
            assert!(
                failed.is_empty(),
                "{} {}: {failed:?}",
                which.name(),
                v.name()
            );
        }
    }
}

#[test]
fn the_nested_order_everywhere_is_caught_in_the_contact_order_case_only() {
    let faults = Faults {
        nested_order_everywhere: true,
        ..Faults::NONE
    };
    let v = Variant::Pyramidal;
    positive_control(TWhich::Zoo, v, "newton");
    let o = check(TWhich::Zoo, v, "newton", &faults);
    let failed = describe(TWhich::Zoo, v, "newton", "nested order everywhere", &o);
    assert!(has(&failed, "integers.contact_list"), "{failed:?}");

    // the states where MuJoCo's own two orders (midphase on and off) differ
    let sorted = tgolden(TWhich::Zoo, Variant::Pyramidal);
    let nested = tgolden(TWhich::Zoo, Variant::Midphase);
    let differing: Vec<usize> = tstates(&sorted)
        .iter()
        .zip(tstates(&nested))
        .enumerate()
        .filter(|(_, (a, b))| a["contacts"] != b["contacts"])
        .map(|(k, _)| k)
        .collect();
    assert_eq!(
        o.integer_states, differing,
        "the fault must fail exactly in the states where MuJoCo's two orders differ"
    );
    println!(
        "MEASURED fault [nested order everywhere] on the zoo: the contact list differs in states {:?}, which are exactly the states where MuJoCo's midphase order and its nested order differ",
        o.integer_states
    );

    // and the faulted engine IS MuJoCo's nested order: it reproduces the midphase-off golden
    let c = tcompile_faulted::<f64>(TWhich::Zoo, v, &faults);
    // (the structure of the midphase-off file; it has no solution blocks)
    let (report, notes) = run_structure(&c, &nested, None);
    assert!(
        notes.is_empty() && report.all_within(),
        "{notes:?} {:?}",
        report.failures()
    );

    // every other model: the orders coincide (single-geom bodies, or capsules only)
    for which in [
        TWhich::Sphere,
        TWhich::Box,
        TWhich::Stack,
        TWhich::Capsules,
        TWhich::Pile,
        TWhich::Humanoid,
    ] {
        for v in CONES {
            let o = check(which, v, "newton", &faults);
            assert!(o.integer_states.is_empty(), "{} {}", which.name(), v.name());
            let failed = describe(which, v, "newton", "nested order everywhere", &o);
            assert!(
                failed.is_empty(),
                "{} {}: {failed:?}",
                which.name(),
                v.name()
            );
        }
    }
}

/// Our Newton solve of every state of `v` with `faults` (the cone Hessian left out, or none): at the
/// model's tolerance (against MuJoCo's Newton block) or at the converged settings (against the
/// converged optimum). Returns the worst relative `qacc` error and the largest iteration-count
/// difference.
fn cone_hessian_errors(
    which: TWhich,
    v: Variant,
    converged: bool,
    faults: &Faults,
) -> (f64, usize) {
    let mut c = tcompile::<f64>(which, v);
    let g = tgolden(which, v);
    if converged {
        let st = &g["converged_settings"];
        c.model.opt.tolerance = f(&st["tolerance"]);
        c.model.opt.iterations = st["iterations"].as_u64().unwrap() as usize;
        c.model.opt.ls_iterations = st["ls_iterations"].as_u64().unwrap() as usize;
    }
    let (mut worst, mut niter_diff) = (0.0f64, 0usize);
    for s in tstates(&g) {
        let mut mm = c.model.clone();
        mm.opt.solver = PrimalSolver::Newton;
        let mut d = tdata(&mm, s);
        forward_faulted(&mm, &mut d, faults);
        assert_eq!(d.warning_collision_overflow, 0, "box-box overflow");
        let reference = if converged {
            &s["converged"]
        } else {
            &s["newton"]
        };
        worst = worst.max(safe_cmp(&widen(&d.qacc), &farr(&reference["qacc"])).rel());
        if !converged {
            niter_diff = niter_diff.max(
                d.solver_niter
                    .abs_diff(reference["solver_niter"].as_u64().unwrap() as usize),
            );
        }
    }
    (worst, niter_diff)
}

/// The bits of a float array.
fn bits(a: &[f64]) -> Vec<u64> {
    a.iter().map(|x| x.to_bits()).collect()
}

/// With the pyramidal cone there is no cone Hessian: the engine with the fault is the engine,
/// to the bit (`qacc`, `efc_force`, `qfrc_constraint`, the zones and the iteration count), in
/// every state.
fn pyramidal_is_bit_for_bit_unchanged(which: TWhich) {
    let c = tcompile::<f64>(which, Variant::Pyramidal);
    let g = tgolden(which, Variant::Pyramidal);
    let faults = Faults {
        drop_cone_hessian: true,
        ..Faults::NONE
    };
    for (k, s) in tstates(&g).iter().enumerate() {
        let a = tforward(&c.model, s, PrimalSolver::Newton);
        let mut mm = c.model.clone();
        mm.opt.solver = PrimalSolver::Newton;
        let mut b = tdata(&mm, s);
        forward_faulted(&mm, &mut b, &faults);
        let what = format!("{} pyramidal state {k}", which.name());
        assert_eq!(bits(&a.qacc), bits(&b.qacc), "{what}: qacc");
        assert_eq!(
            bits(&a.efc_force[..a.nefc]),
            bits(&b.efc_force[..b.nefc]),
            "{what}: efc_force"
        );
        assert_eq!(
            bits(&a.qfrc_constraint),
            bits(&b.qfrc_constraint),
            "{what}: qfrc_constraint"
        );
        assert_eq!(
            a.efc_state[..a.nefc],
            b.efc_state[..b.nefc],
            "{what}: zones"
        );
        assert_eq!(a.solver_niter, b.solver_niter, "{what}: solver_niter");
    }
}

/// The dropped cone Hessian, on all seven models. The positive controls come first and cover BOTH
/// comparisons on every model (the review found three of seven run, and none at the converged
/// settings): the unfaulted engine passes the comparison at the model's tolerance (the structure,
/// the solution, the zones) with `solver_niter` equal, and the comparison at the converged settings.
///
/// The verdicts: at the model's tolerance (the solution and the iteration count) the fault is
/// seen on at least one model; at the converged optimum it is REPORTED. The converged reference
/// is now a true optimum (`contacts_solve.rs`: residual at most 1.2e-14), so a difference from it
/// is a distance from the optimum and not a difference of paths. A Newton solve whose Hessian is
/// wrong in the cone rows is still a descent method on a strictly convex cost, and was measured
/// to reach the optimum on two of the seven models (sphere 1.8e-15, humanoid 3.5e-14: NOT caught
/// there) and to stop short of it on the other five (box 3.4e-8, stack 8.4e-6, capsules 1.5e-9,
/// pile 1.3e-10, zoo 1.2e-3: caught, the pile barely above the 1e-10 gate). (The earlier "caught
/// at the converged optimum" verdicts, against the old reference at tolerance 1e-15, mixed two
/// effects: the faulted solve's own early stop by the Newton-decrement test at 1e-15, computed
/// with the wrong Hessian, and the old reference's distance from the optimum, up to 1.8e-8. The
/// sphere's 2.3e-9 and the humanoid's 2.5e-9 were early stops: run to 1e-28 both reach the
/// optimum.)
#[test]
fn a_dropped_cone_hessian_is_reported_at_the_models_tolerance_and_at_the_optimum() {
    let faults = Faults {
        drop_cone_hessian: true,
        ..Faults::NONE
    };
    let mut caught_at_tolerance = false;
    for which in ALL {
        // positive controls: the unfaulted engine passes both comparisons
        positive_control(which, Variant::Elliptic, "newton");
        let (ok_opt, _) = cone_hessian_errors(which, Variant::Elliptic, true, &Faults::NONE);
        let (ok_tol, ok_niter) =
            cone_hessian_errors(which, Variant::Elliptic, false, &Faults::NONE);
        assert!(
            ok_opt <= 1e-10,
            "{}: positive control at the converged optimum: {ok_opt:e}",
            which.name()
        );
        assert!(
            ok_tol <= SOLVE_RTOL && ok_niter == 0,
            "{}: positive control at the model's tolerance: {ok_tol:e}, solver_niter differs by {ok_niter}",
            which.name()
        );

        let (at_tol, niter_diff) = cone_hessian_errors(which, Variant::Elliptic, false, &faults);
        let (at_opt, _) = cone_hessian_errors(which, Variant::Elliptic, true, &faults);
        let tol_caught = at_tol > SOLVE_RTOL || niter_diff > 0;
        let opt_caught = at_opt > 1e-10;
        println!(
            "MEASURED f64 cone Hessian dropped, {} elliptic newton: at the model's tolerance qacc error {at_tol:e} and solver_niter differs by up to {niter_diff} -> {}; at the converged optimum qacc error {at_opt:e} -> {} (positive controls: {ok_tol:e} and {ok_opt:e})",
            which.name(),
            if tol_caught { "CAUGHT" } else { "not caught" },
            if opt_caught {
                "CAUGHT"
            } else {
                "NOT caught (the same optimum)"
            },
        );
        caught_at_tolerance |= tol_caught;

        // the pyramidal cone has no cone Hessian: the fault is invisible there, bit for bit
        pyramidal_is_bit_for_bit_unchanged(which);
    }
    // the comparison of the solution at the model's tolerance (qacc and the iteration count) sees
    // it on at least one model
    assert!(
        caught_at_tolerance,
        "the dropped cone Hessian was not caught by any comparison"
    );
}
