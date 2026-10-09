//! The constraint structure of the forward pass against MuJoCo 3.14.0, state by state
//! (`K = 8` states of each of three models): the fixed-tendon kinematics, which rows
//! exist (`nefc`, `efc_type`, `efc_id`, exactly), and every array MuJoCo computes
//! before it solves (`efc_J`, `efc_pos`, `efc_margin`, `efc_frictionloss`,
//! `efc_diagApprox`, `efc_R`, `efc_D`, `efc_KBIP`, `efc_aref`, `efc_vel`) plus
//! `qacc_smooth`, to 1e-12 relative to the array's largest absolute value (floor 1e-14).
//!
//! The states put joints past their limits, inside their margins and well inside, and
//! tendons past a bound, and `qvel` puts friction-loss rows in each zone, so the rows
//! that exist differ from state to state; the guard test at the end checks that the
//! golden files really cover that.

mod common;

use common::cons::*;
use common::*;
use sim_physics::{ConstraintState, Data, Real, forward};

const RTOL: f64 = 1e-12;
const FLOOR: f64 = 1e-14;

/// Compares the structure of `d` (after `forward`) with golden state `s`.
fn check_structure<R: Real>(
    m: &sim_physics::Model<R>,
    d: &Data<R>,
    s: &serde_json::Value,
    report: &mut Report,
) {
    // which rows exist, exactly
    assert_eq!(d.nefc as u64, s["nefc"].as_u64().unwrap(), "nefc");
    assert_eq!(d.ne as u64, s["ne"].as_u64().unwrap(), "ne");
    assert_eq!(d.nf as u64, s["nf"].as_u64().unwrap(), "nf");
    assert_eq!(d.nl as u64, s["nl"].as_u64().unwrap(), "nl");
    let n = d.nefc;
    let types: Vec<i64> = d.efc_type[..n].iter().map(|&t| type_code(t)).collect();
    assert_eq!(types, ints(&s["efc_type"]), "efc_type");
    let ids: Vec<i64> = d.efc_id[..n].iter().map(|&i| i as i64).collect();
    assert_eq!(ids, ints(&s["efc_id"]), "efc_id");

    let nv = m.nv;
    let g = |key: &str| farr(&s[key]);
    let mut put = |name: &str, ours: Vec<f64>, reference: Vec<f64>| {
        rec(report, name, cmp(&ours, &reference), RTOL, FLOOR);
    };
    put("ten_length", widen(&d.ten_length), g("ten_length"));
    put("ten_J", widen(&d.ten_j), golden_rows(&s["ten_J"]));
    put("ten_velocity", widen(&d.ten_velocity), g("ten_velocity"));
    put("efc_J", head(&d.efc_j, n * nv), golden_rows(&s["efc_J"]));
    put("efc_pos", head(&d.efc_pos, n), g("efc_pos"));
    put("efc_margin", head(&d.efc_margin, n), g("efc_margin"));
    put(
        "efc_frictionloss",
        head(&d.efc_frictionloss, n),
        g("efc_frictionloss"),
    );
    put(
        "efc_diagApprox",
        head(&d.efc_diag_approx, n),
        g("efc_diagApprox"),
    );
    put("efc_R", head(&d.efc_r, n), g("efc_R"));
    put("efc_D", head(&d.efc_d, n), g("efc_D"));
    put(
        "efc_KBIP",
        head(&d.efc_kbip, 4 * n),
        golden_rows(&s["efc_KBIP"]),
    );
    put("efc_aref", head(&d.efc_aref, n), g("efc_aref"));
    put("efc_vel", head(&d.efc_vel, n), g("efc_vel"));
    put("qacc_smooth", widen(&d.qacc_smooth), g("qacc_smooth"));
}

fn run<R: Real>(which: CWhich) -> Report {
    let c = ccompile::<R>(which);
    let g = cgolden(which);
    let mut report = Report::default();
    for s in states(&g) {
        let mut d = cdata(&c.model, s);
        forward(&c.model, &mut d);
        check_structure(&c.model, &d, s, &mut report);
    }
    report
}

fn gate(which: CWhich) {
    let report = run::<f64>(which);
    report.print("f64", &format!("{} constraint structure", which.name()));
    assert!(
        report.all_within(),
        "{}: out of tolerance: {:?}",
        which.name(),
        report.failures()
    );
}

#[test]
fn humanoid_constraint_structure_matches_mujoco() {
    gate(CWhich::Humanoid);
}

#[test]
fn constrained_constraint_structure_matches_mujoco() {
    gate(CWhich::Constrained);
}

#[test]
fn zoo_has_no_constraint_rows_and_matches_mujoco() {
    // the zoo has no limit, friction loss or tendon: nefc = 0, and everything else
    // (the tendon arrays are empty) is the phase-1b zoo
    gate(CWhich::Zoo);
    let g = cgolden(CWhich::Zoo);
    for s in states(&g) {
        assert_eq!(s["nefc"].as_u64(), Some(0));
    }
}

/// The golden states are not all alike: the comparison would pass vacuously on files
/// whose rows were all of one kind or whose states were all the same.
#[test]
fn the_golden_states_cover_every_kind_of_row_and_every_zone() {
    // humanoid: joint limits and tendon limits, in different numbers per state
    let g = cgolden(CWhich::Humanoid);
    let counts: Vec<u64> = states(&g)
        .iter()
        .map(|s| s["nefc"].as_u64().unwrap())
        .collect();
    assert!(counts.iter().min() != counts.iter().max(), "{counts:?}");
    let all_types: Vec<i64> = states(&g)
        .iter()
        .flat_map(|s| ints(&s["efc_type"]))
        .collect();
    assert!(all_types.contains(&3), "no joint limit row");
    assert!(all_types.contains(&4), "no tendon limit row");
    assert!(
        !all_types.contains(&1) && !all_types.contains(&2),
        "the humanoid has no friction loss"
    );
    // the limit rows are in both zones (satisfied and quadratic) somewhere
    let all_states: Vec<i64> = states(&g)
        .iter()
        .flat_map(|s| ints(&s["newton"]["efc_state"]))
        .collect();
    assert!(all_states.contains(&0) && all_states.contains(&1));

    // constrained: every one of the four kinds of row, and every zone of the cost
    let g = cgolden(CWhich::Constrained);
    let all_types: Vec<i64> = states(&g)
        .iter()
        .flat_map(|s| ints(&s["efc_type"]))
        .collect();
    for t in 1..=4 {
        assert!(all_types.contains(&t), "no row of type {t}");
    }
    let all_states: Vec<i64> = states(&g)
        .iter()
        .flat_map(|s| ints(&s["newton"]["efc_state"]))
        .collect();
    for z in [
        ConstraintState::Satisfied,
        ConstraintState::Quadratic,
        ConstraintState::LinearNeg,
        ConstraintState::LinearPos,
    ] {
        assert!(
            all_states.contains(&state_code(z)),
            "no row ends in the zone {z:?}"
        );
    }
    // the friction rows alone reach all three of their zones
    let mut friction_zones = Vec::new();
    for s in states(&g) {
        let types = ints(&s["efc_type"]);
        let zones = ints(&s["newton"]["efc_state"]);
        for (t, z) in types.iter().zip(&zones) {
            if *t == 1 || *t == 2 {
                friction_zones.push(*z);
            }
        }
    }
    for z in [1, 2, 3] {
        assert!(friction_zones.contains(&z), "no friction row in zone {z}");
    }
    // the number of rows varies, and the warmstart is zero in half the states
    let counts: Vec<u64> = states(&g)
        .iter()
        .map(|s| s["nefc"].as_u64().unwrap())
        .collect();
    assert!(counts.iter().min() != counts.iter().max(), "{counts:?}");
    let zero_ws = states(&g)
        .iter()
        .filter(|s| farr(&s["qacc_warmstart"]).iter().all(|&x| x == 0.0))
        .count();
    assert_eq!(zero_ws, 4);
}
