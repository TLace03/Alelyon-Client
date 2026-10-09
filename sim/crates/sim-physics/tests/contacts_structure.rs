//! The contact list and the constraint structure of the forward pass against MuJoCo 3.14.0,
//! state by state (phase 1c-ii): the models of `tools/sim_contacts_mujoco_golden.py`, each with
//! the pyramidal and the elliptic cone, and the zoo with `mjDSBL_MIDPHASE` and
//! `mjDSBL_FILTERPARENT` set.
//!
//! - **The contact list**, contact by contact, in MuJoCo's order: `ncon`, the geom ids, `dim`,
//!   `exclude` and `efc_address` exactly; `dist`, `pos`, `frame`, `includemargin`, `friction`,
//!   `solref`, `solreffriction` and `solimp` to 1e-12 relative to the array's largest absolute
//!   value (floor 1e-14). Any difference in order or count is a failure, and the message names
//!   the first contact that differs.
//! - **The rows**: `nefc`, `ne`, `nf`, `nl`, `efc_type` and `efc_id` exactly; `efc_J`, `efc_pos`,
//!   `efc_margin`, `efc_frictionloss`, `efc_diagApprox`, `efc_R`, `efc_D`, `efc_KBIP`, `efc_aref`,
//!   `efc_vel` and `qacc_smooth` to the same tolerance.
//!
//! The guard test at the end checks that the golden files cover what the comparison is for:
//! every collider, condim 1, 3, 4 and 6, the three row types of a contact, every zone of an
//! elliptic cone, a contact in the gap (generated and counted, but excluded), the contact order
//! of a body with two boxes against a body with two spheres (where MuJoCo's midphase order is not
//! the nested one), a body pair that is excluded and a parent-child pair that is filtered.

mod common;

use common::cons::*;
use common::contacts::*;
use common::*;
use sim_physics::Real;

/// One model and variant, every state.
fn run<R: Real>(which: TWhich, v: Variant) -> (Report, Vec<String>) {
    let c = tcompile::<R>(which, v);
    let g = tgolden(which, v);
    run_structure(&c, &g, None)
}

fn gate(which: TWhich, v: Variant) {
    let (report, notes) = run::<f64>(which, v);
    report.print(
        "f64",
        &format!("{} {} contact structure", which.name(), v.name()),
    );
    // any difference in order or count is a failure, reported with the first contact that differs
    assert!(
        notes.is_empty(),
        "{} {}: {:?}",
        which.name(),
        v.name(),
        notes
    );
    assert!(
        report.all_within(),
        "{} {}: out of tolerance: {:?}",
        which.name(),
        v.name(),
        report.failures()
    );
}

macro_rules! structure_test {
    ($name:ident, $which:expr, $variant:expr) => {
        #[test]
        fn $name() {
            gate($which, $variant);
        }
    };
}

structure_test!(
    sphere_pyramidal_structure_matches_mujoco,
    TWhich::Sphere,
    Variant::Pyramidal
);
structure_test!(
    sphere_elliptic_structure_matches_mujoco,
    TWhich::Sphere,
    Variant::Elliptic
);
structure_test!(
    box_pyramidal_structure_matches_mujoco,
    TWhich::Box,
    Variant::Pyramidal
);
structure_test!(
    box_elliptic_structure_matches_mujoco,
    TWhich::Box,
    Variant::Elliptic
);
structure_test!(
    stack_pyramidal_structure_matches_mujoco,
    TWhich::Stack,
    Variant::Pyramidal
);
structure_test!(
    stack_elliptic_structure_matches_mujoco,
    TWhich::Stack,
    Variant::Elliptic
);
structure_test!(
    capsules_pyramidal_structure_matches_mujoco,
    TWhich::Capsules,
    Variant::Pyramidal
);
structure_test!(
    capsules_elliptic_structure_matches_mujoco,
    TWhich::Capsules,
    Variant::Elliptic
);
structure_test!(
    pile_pyramidal_structure_matches_mujoco,
    TWhich::Pile,
    Variant::Pyramidal
);
structure_test!(
    pile_elliptic_structure_matches_mujoco,
    TWhich::Pile,
    Variant::Elliptic
);
structure_test!(
    humanoid_pyramidal_structure_matches_mujoco,
    TWhich::Humanoid,
    Variant::Pyramidal
);
structure_test!(
    humanoid_elliptic_structure_matches_mujoco,
    TWhich::Humanoid,
    Variant::Elliptic
);
structure_test!(
    zoo_pyramidal_structure_matches_mujoco,
    TWhich::Zoo,
    Variant::Pyramidal
);
structure_test!(
    zoo_elliptic_structure_matches_mujoco,
    TWhich::Zoo,
    Variant::Elliptic
);
structure_test!(
    zoo_without_midphase_structure_matches_mujoco,
    TWhich::Zoo,
    Variant::Midphase
);
structure_test!(
    zoo_without_parent_filter_structure_matches_mujoco,
    TWhich::Zoo,
    Variant::Filterparent
);

// ---------------------------------------------------------------------------
// coverage: the comparison would pass vacuously on files that lack these cases
// ---------------------------------------------------------------------------

/// MuJoCo's collision-table name of a pair of geom types (lower type first).
fn collider_name(t1: i64, t2: i64) -> &'static str {
    match (t1.min(t2), t1.max(t2)) {
        (0, 2) => "plane-sphere",
        (0, 3) => "plane-capsule",
        (0, 5) => "plane-cylinder",
        (0, 6) => "plane-box",
        (2, 2) => "sphere-sphere",
        (2, 3) => "sphere-capsule",
        (2, 5) => "sphere-cylinder",
        (2, 6) => "sphere-box",
        (3, 3) => "capsule-capsule",
        (3, 6) => "capsule-box",
        (6, 6) => "box-box",
        _ => "other",
    }
}

#[test]
fn the_golden_states_cover_every_collider_condim_row_type_and_zone() {
    use std::collections::BTreeSet;
    let mut colliders = BTreeSet::new();
    let mut condims = BTreeSet::new();
    let mut row_types = BTreeSet::new();
    let (mut gap_excluded, mut touching_excluded) = (0usize, 0usize);
    let mut zones_elliptic = BTreeSet::new();
    for which in TMODELS {
        for v in CONES {
            let g = tgolden(which, v);
            let geom_types = tints(&g["arrays"]["geom_type"]);
            let geom_gap = farr(&g["arrays"]["geom_gap"]);
            for s in tstates(&g) {
                for c in s["contacts"].as_array().unwrap() {
                    let geoms = tints(&c["geom"]);
                    colliders.insert(collider_name(
                        geom_types[geoms[0] as usize],
                        geom_types[geoms[1] as usize],
                    ));
                    if c["exclude"].as_i64() == Some(0) {
                        condims.insert(c["dim"].as_i64().unwrap());
                    }
                    if c["exclude"].as_i64() == Some(1) {
                        assert_eq!(c["efc_address"].as_i64(), Some(-1));
                        // a contact in the GAP: its pair has a gap (the sum of the geoms'),
                        // and it lies beyond includemargin (strictly) and within
                        // includemargin + gap. A contact that merely touches (dist ==
                        // includemargin == 0) is excluded too (`dist >= includemargin`) but is
                        // not in any gap, and does not count here.
                        let gap = geom_gap[geoms[0] as usize] + geom_gap[geoms[1] as usize];
                        let (dist, inc) = (f(&c["dist"]), f(&c["includemargin"]));
                        if gap > 0.0 && dist > inc && dist <= inc + gap {
                            gap_excluded += 1;
                        } else {
                            touching_excluded += 1;
                        }
                    }
                }
                row_types.extend(ints(&s["efc_type"]).into_iter().filter(|&t| t >= 5));
                if v == Variant::Elliptic {
                    let types = ints(&s["efc_type"]);
                    let zones = ints(&s["newton"]["efc_state"]);
                    for (t, z) in types.iter().zip(&zones) {
                        if *t == 7 {
                            zones_elliptic.insert(*z);
                        }
                    }
                }
            }
        }
    }
    for c in [
        "plane-sphere",
        "plane-capsule",
        "plane-cylinder",
        "plane-box",
        "sphere-sphere",
        "sphere-capsule",
        "sphere-cylinder",
        "sphere-box",
        "capsule-capsule",
        "capsule-box",
        "box-box",
    ] {
        assert!(colliders.contains(c), "no {c} contact in the golden states");
    }
    assert!(!colliders.contains("other"), "{colliders:?}");
    for dim in [1, 3, 4, 6] {
        assert!(condims.contains(&dim), "no contact of condim {dim}");
    }
    for t in [5, 6, 7] {
        assert!(row_types.contains(&t), "no contact row of type {t}");
    }
    assert!(
        gap_excluded > 0,
        "no contact in the gap (excluded, counted)"
    );
    // every zone of the elliptic cone: satisfied (top), quadratic (bottom), cone (middle)
    for z in [0, 1, 4] {
        assert!(
            zones_elliptic.contains(&z),
            "no elliptic contact ends in zone {z}: {zones_elliptic:?}"
        );
    }
    println!(
        "MEASURED coverage: colliders {colliders:?}, condims {condims:?}, contact row types {row_types:?}, {gap_excluded} contacts in the gap (dist strictly beyond includemargin, within includemargin + gap, a pair with a gap) and {touching_excluded} excluded contacts that merely touch (dist >= includemargin with no gap), elliptic zones {zones_elliptic:?}"
    );
}

/// The zoo's body with two boxes against the body with two spheres: MuJoCo's contact order
/// (the midphase sorts by geom ids) is not the nested one, and the `mjDSBL_MIDPHASE` variant
/// has the nested one. The two golden files must differ in the order of these contacts, and
/// ours must reproduce each.
#[test]
fn the_golden_files_contain_the_contact_order_case() {
    let g = tgolden(TWhich::Zoo, Variant::Pyramidal);
    let h = tgolden(TWhich::Zoo, Variant::Midphase);
    let mut differing_states = 0;
    for (a, b) in tstates(&g).iter().zip(tstates(&h)) {
        let order = |s: &serde_json::Value| -> Vec<Vec<i64>> {
            s["contacts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| tints(&c["geom"]))
                .collect()
        };
        let (oa, ob) = (order(a), order(b));
        let mut sa = oa.clone();
        let mut sb = ob.clone();
        sa.sort();
        sb.sort();
        // the same contacts in a different order
        if oa != ob {
            assert_eq!(
                sa, sb,
                "state {}: the two orders hold different contacts",
                a["name"]
            );
            differing_states += 1;
        }
    }
    println!(
        "MEASURED coverage: {differing_states} zoo states where MuJoCo's midphase order differs from the nested order"
    );
    assert!(
        differing_states > 0,
        "no state exercises the contact-order case"
    );
}

/// The parent-child filter: with `mjDSBL_FILTERPARENT` set, MuJoCo makes contacts between a
/// body and its parent, and without it, none; both are in the golden files and ours matches
/// each (the structure tests above). Here: the two files differ in their contact count.
#[test]
fn the_parent_filter_changes_the_contacts() {
    let g = tgolden(TWhich::Zoo, Variant::Pyramidal);
    let h = tgolden(TWhich::Zoo, Variant::Filterparent);
    let counts = |g: &serde_json::Value| -> Vec<u64> {
        tstates(g)
            .iter()
            .map(|s| s["ncon"].as_u64().unwrap())
            .collect()
    };
    let (a, b) = (counts(&g), counts(&h));
    assert!(a.iter().zip(&b).any(|(x, y)| y > x), "{a:?} vs {b:?}");
    assert!(a.iter().zip(&b).all(|(x, y)| y >= x));
}
