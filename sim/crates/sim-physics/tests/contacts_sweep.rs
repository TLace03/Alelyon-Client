//! The collider sweep against MuJoCo 3.14.0 (phase 1c-ii): for each of the eleven primitive
//! colliders, 64 seeded relative poses of two geoms (the first eight are exact degenerate cases:
//! coincident centres, an exact 90-degree and 45-degree rotation, a tilt of 1e-4 degrees, a
//! twist of exactly 45 degrees; the rest are random, penetrating, touching, inside the margin,
//! inside the gap and separated), without and with a margin of 0.02 and a gap of 0.03 on both
//! geoms. `contact_pairs_golden.json` holds `mj_collision`'s contact list of each pose, and
//! `contact_sweep_<collider>[_mg].xml` (written by the generator, byte-checked by `--check`)
//! the two-geom models.
//!
//! Gate, per collider: `ncon`, the geom ids, `exclude` and the ORDER of the contacts exactly;
//! `dist`, `pos`, `frame` and `includemargin` within 1e-12 relative to the field's largest
//! absolute value over the collider's contacts (floor 1e-14). The maximum per collider is
//! printed. The models are built through the importer and `Model::compile`, so the candidate
//! list, the bounding-sphere filter and the colliders are the ones the engine runs.

mod common;

use common::contacts::*;
use common::*;

const COLLIDERS: [&str; 11] = [
    "plane_sphere",
    "plane_capsule",
    "plane_cylinder",
    "plane_box",
    "sphere_sphere",
    "sphere_capsule",
    "sphere_cylinder",
    "sphere_box",
    "capsule_capsule",
    "capsule_box",
    "box_box",
];

fn gate(collider: &str) {
    let g = read_json(&fixtures().join("contact_pairs_golden.json"));
    assert_eq!(g["mujoco_version"], "3.14.0");
    let entry = &g["colliders"][collider];
    assert!(entry.is_object(), "no sweep for {collider}");
    for (label, variant) in [("plain", "plain"), ("margin+gap", "mg")] {
        let out = run_sweep_variant(&entry["variants"][variant], None);
        out.report
            .print("f64", &format!("sweep {collider} {label}"));
        let worst = SWEEP_FIELDS
            .iter()
            .map(|n| out.report.get(n).unwrap().abs)
            .fold(0.0f64, f64::max);
        println!(
            "MEASURED f64 sweep {collider} {label}: {} cases, {} with contacts, {} contacts ({} excluded in the gap); integer fields {}; worst float error {worst:e}",
            out.cases,
            out.cases_with_contacts,
            out.contacts,
            out.excluded,
            if out.integer_failures.is_empty() {
                "all equal"
            } else {
                "DIFFER"
            },
        );
        assert!(
            out.integer_failures.is_empty(),
            "{collider} {label}: the first difference: {}; {} in all",
            out.integer_failures[0],
            out.integer_failures.len()
        );
        assert!(
            out.report.all_within(),
            "{collider} {label}: out of tolerance: {:?}",
            out.report.failures()
        );
        // the sweep must have something to compare ...
        assert!(
            out.cases_with_contacts >= 8,
            "{collider} {label}: too few cases touch"
        );
        // ... and poses beyond the detection range (margin + gap), which MuJoCo and we agree
        // make no contact (the review found two colliders whose sweep had none)
        assert!(
            out.cases_with_contacts < out.cases,
            "{collider} {label}: every pose makes a contact, none is separated"
        );
        if label == "margin+gap" {
            // some contacts lie in the gap: generated, counted, excluded
            assert!(out.excluded > 0, "{collider}: no contact in the gap");
        }
    }
}

macro_rules! sweep_test {
    ($name:ident, $collider:expr) => {
        #[test]
        fn $name() {
            gate($collider);
        }
    };
}

sweep_test!(plane_sphere_sweep_matches_mujoco, "plane_sphere");
sweep_test!(plane_capsule_sweep_matches_mujoco, "plane_capsule");
sweep_test!(plane_cylinder_sweep_matches_mujoco, "plane_cylinder");
sweep_test!(plane_box_sweep_matches_mujoco, "plane_box");
sweep_test!(sphere_sphere_sweep_matches_mujoco, "sphere_sphere");
sweep_test!(sphere_capsule_sweep_matches_mujoco, "sphere_capsule");
sweep_test!(sphere_cylinder_sweep_matches_mujoco, "sphere_cylinder");
sweep_test!(sphere_box_sweep_matches_mujoco, "sphere_box");
sweep_test!(capsule_capsule_sweep_matches_mujoco, "capsule_capsule");
sweep_test!(capsule_box_sweep_matches_mujoco, "capsule_box");
sweep_test!(box_box_sweep_matches_mujoco, "box_box");

#[test]
fn the_sweep_file_names_all_eleven_colliders() {
    let g = read_json(&fixtures().join("contact_pairs_golden.json"));
    let names: Vec<&str> = g["colliders"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    let mut want = COLLIDERS.to_vec();
    want.sort_unstable();
    let mut have = names.clone();
    have.sort_unstable();
    assert_eq!(have, want);
}
