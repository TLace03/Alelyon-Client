//! The compiled contact arrays of `Model::compile` against what MuJoCo 3.14.0 compiled the
//! same XML to (phase 1c-ii), for every model of the contact golden files:
//!
//! - integers exactly: `geom_type`, `geom_bodyid`, `geom_condim`, `geom_priority`,
//!   `geom_contype`, `geom_conaffinity`, `geom_sameframe`, `body_geomadr`, `body_geomnum`,
//!   `exclude_signature`, `body_weldid`, and our `body_has_bvh` against `body_bvhadr >= 0`;
//! - floats exactly (the first gate), else the measured difference is printed and held to
//!   1e-14 relative to the array's largest absolute value: `geom_pos`, `geom_quat`,
//!   `geom_size`, `geom_rbound`, `geom_solmix`, `geom_solref`, `geom_solimp`, `geom_friction`,
//!   `geom_margin`, `geom_gap`, `body_invweight0`.
//!
//! `geom_rbound` is the one whose rule is inferred (MuJoCo's `mjCGeom::GetRBound` is not in
//! the reference tree): the comparison with the oracle's values is what holds it.
//!
//! The test also prints the static contact maxima (`ncon_max`, `nefc_max`) of every model
//! and the peaks MuJoCo measured over the golden states, and checks that the maxima bound
//! them (and that a model's `ncon_max` is MuJoCo's `mj_maxContact` summed over the
//! candidate list).

mod common;

use common::contacts::*;
use common::*;
use sim_physics::{Cone, Model, Real};

/// Compares our array with MuJoCo's: the exact result, else the error.
fn floats(name: &str, model: &str, ours: Vec<f64>, reference: Vec<f64>) {
    let e = compare(&ours, &reference);
    println!(
        "MEASURED f64 {model} compiled {name}: max_abs_err={:e} array_max_abs={:e} {}",
        e.abs,
        e.scale,
        if e.abs == 0.0 { "exact" } else { "NOT EXACT" }
    );
    assert!(
        e.within(1e-14, 1e-15),
        "{model} {name}: error {:e} (array max {:e})",
        e.abs,
        e.scale
    );
}

fn integers(name: &str, model: &str, ours: Vec<i64>, reference: Vec<i64>) {
    assert_eq!(ours, reference, "{model} {name}");
}

fn xyzw(wxyz: &[f64]) -> Vec<f64> {
    wxyz.chunks(4)
        .flat_map(|q| [q[1], q[2], q[3], q[0]])
        .collect()
}

fn model_case<R: Real>(which: TWhich) -> Model<R> {
    let g = tgolden(which, Variant::Pyramidal);
    let c = tcompile::<f64>(which, Variant::Pyramidal);
    let m = &c.model;
    let a = &g["arrays"];
    let name = which.name();
    assert_eq!(m.ngeom, a["geom_type"].as_array().unwrap().len(), "ngeom");

    // ---- integers
    integers(
        "geom_type",
        name,
        m.geom_type.iter().map(|t| i64::from(t.code())).collect(),
        tints(&a["geom_type"]),
    );
    integers(
        "geom_bodyid",
        name,
        m.geom_bodyid.iter().map(|&b| b as i64).collect(),
        tints(&a["geom_bodyid"]),
    );
    integers(
        "geom_condim",
        name,
        m.geom_condim.iter().map(|&c| i64::from(c)).collect(),
        tints(&a["geom_condim"]),
    );
    integers(
        "geom_priority",
        name,
        m.geom_priority.iter().map(|&c| i64::from(c)).collect(),
        tints(&a["geom_priority"]),
    );
    integers(
        "geom_contype",
        name,
        m.geom_contype.iter().map(|&c| i64::from(c)).collect(),
        tints(&a["geom_contype"]),
    );
    integers(
        "geom_conaffinity",
        name,
        m.geom_conaffinity.iter().map(|&c| i64::from(c)).collect(),
        tints(&a["geom_conaffinity"]),
    );
    integers(
        "geom_sameframe",
        name,
        m.geom_sameframe
            .iter()
            .map(|s| i64::from(s.code()))
            .collect(),
        tints(&a["geom_sameframe"]),
    );
    integers(
        "body_geomadr",
        name,
        m.body_geomadr.iter().map(|&b| i64::from(b)).collect(),
        tints(&a["body_geomadr"]),
    );
    integers(
        "body_geomnum",
        name,
        m.body_geomnum.iter().map(|&b| b as i64).collect(),
        tints(&a["body_geomnum"]),
    );
    integers(
        "body_weldid",
        name,
        m.body_weldid.iter().map(|&b| b as i64).collect(),
        tints(&a["body_weldid"]),
    );
    // `has_bvh` is inferred as "has a geom": held to MuJoCo's body_bvhadr >= 0
    integers(
        "has_bvh",
        name,
        m.body_has_bvh.iter().map(|&b| i64::from(b)).collect(),
        tints(&a["body_bvhadr"])
            .iter()
            .map(|&adr| i64::from(adr >= 0))
            .collect(),
    );
    integers(
        "exclude_signature",
        name,
        m.exclude_signature.iter().map(|&s| i64::from(s)).collect(),
        tints(&a["exclude_signature"]),
    );

    // ---- floats
    floats("geom_pos", name, widen(&m.geom_pos), farr(&a["geom_pos"]));
    floats(
        "geom_quat",
        name,
        xyzw(&widen(&m.geom_quat)),
        farr(&a["geom_quat"]),
    );
    floats(
        "geom_size",
        name,
        widen(&m.geom_size),
        farr(&a["geom_size"]),
    );
    floats(
        "geom_rbound",
        name,
        widen(&m.geom_rbound),
        farr(&a["geom_rbound"]),
    );
    floats(
        "geom_solmix",
        name,
        widen(&m.geom_solmix),
        farr(&a["geom_solmix"]),
    );
    floats(
        "geom_solref",
        name,
        widen(&m.geom_solref),
        farr(&a["geom_solref"]),
    );
    floats(
        "geom_solimp",
        name,
        widen(&m.geom_solimp),
        farr(&a["geom_solimp"]),
    );
    floats(
        "geom_friction",
        name,
        widen(&m.geom_friction),
        farr(&a["geom_friction"]),
    );
    floats(
        "geom_margin",
        name,
        widen(&m.geom_margin),
        farr(&a["geom_margin"]),
    );
    floats("geom_gap", name, widen(&m.geom_gap), farr(&a["geom_gap"]));
    floats(
        "body_invweight0",
        name,
        widen(&m.body_invweight0),
        farr(&a["body_invweight0"]),
    );
    floats(
        "meaninertia",
        name,
        vec![m.meaninertia],
        vec![f(&a["meaninertia"])],
    );

    // ---- the static maxima against the peaks MuJoCo measured over the golden states
    let mut peak_ncon = 0usize;
    let mut peak_nefc = [0usize; 2];
    for (k, v) in [Variant::Pyramidal, Variant::Elliptic]
        .into_iter()
        .enumerate()
    {
        let gv = tgolden(which, v);
        let cv = tcompile::<f64>(which, v);
        for s in tstates(&gv) {
            let ncon = s["ncon"].as_u64().unwrap() as usize;
            let nefc = s["nefc"].as_u64().unwrap() as usize;
            peak_ncon = peak_ncon.max(ncon);
            peak_nefc[k] = peak_nefc[k].max(nefc);
            assert!(ncon <= cv.model.ncon_max, "{name}: ncon exceeds ncon_max");
            assert!(nefc <= cv.model.nefc_max, "{name}: nefc exceeds nefc_max");
        }
        println!(
            "MEASURED f64 {name} {}: ncon_max={} nefc_max={} (peak over MuJoCo's states: ncon {peak_ncon}, nefc {})",
            v.name(),
            cv.model.ncon_max,
            cv.model.nefc_max,
            peak_nefc[k]
        );
    }

    // ncon_max is the sum of the per-pair maxima (mj_maxContact), and each pair's slots are
    // the prefix sum in list order
    let mut next = 0usize;
    for cand in &m.candidates {
        assert_eq!(cand.slot_offset, next, "{name}: slots are a prefix sum");
        assert_eq!(cand.slot_count, cand.collider.max_contacts());
        next += cand.slot_count;
    }
    assert_eq!(m.ncon_max, next);

    // nefc_max for the pyramidal cone bounds the elliptic one (so changing opt.cone on a
    // compiled model stays inside the allocation)
    let pyramidal = tcompile::<f64>(which, Variant::Pyramidal).model;
    let elliptic = tcompile::<f64>(which, Variant::Elliptic).model;
    assert_eq!(pyramidal.opt.cone, Cone::Pyramidal);
    assert_eq!(elliptic.opt.cone, Cone::Elliptic);
    assert_eq!(pyramidal.nefc_max, elliptic.nefc_max);
    let elliptic_rows: usize = pyramidal
        .candidates
        .iter()
        .map(|c| c.slot_count * c.condim)
        .sum();
    let pyramidal_rows: usize = pyramidal
        .candidates
        .iter()
        .map(|c| c.slot_count * if c.condim == 1 { 1 } else { 2 * (c.condim - 1) })
        .sum();
    assert!(pyramidal_rows >= elliptic_rows, "{name}");
    c.model.rounded_to::<R>()
}

macro_rules! model_test {
    ($name:ident, $which:expr) => {
        #[test]
        fn $name() {
            let _ = model_case::<f64>($which);
        }
    };
}

model_test!(sphere_model_arrays_match_mujoco, TWhich::Sphere);
model_test!(box_model_arrays_match_mujoco, TWhich::Box);
model_test!(stack_model_arrays_match_mujoco, TWhich::Stack);
model_test!(capsules_model_arrays_match_mujoco, TWhich::Capsules);
model_test!(pile_model_arrays_match_mujoco, TWhich::Pile);
model_test!(humanoid_model_arrays_match_mujoco, TWhich::Humanoid);
model_test!(zoo_model_arrays_match_mujoco, TWhich::Zoo);

/// A `Model<f32>` is the `Model<f64>` rounded once: every contact array, including the
/// candidates' mixed parameters.
#[test]
fn the_f32_model_is_the_f64_model_rounded_once() {
    for which in TMODELS {
        let c64 = tcompile::<f64>(which, Variant::Pyramidal);
        let c32 = tcompile::<f32>(which, Variant::Pyramidal);
        let (a, b) = (&c64.model, &c32.model);
        let round = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
        assert_eq!(round(&a.geom_pos), b.geom_pos, "{}", which.name());
        assert_eq!(round(&a.geom_quat), b.geom_quat);
        assert_eq!(round(&a.geom_size), b.geom_size);
        assert_eq!(round(&a.geom_rbound), b.geom_rbound);
        assert_eq!(round(&a.geom_solmix), b.geom_solmix);
        assert_eq!(round(&a.geom_solref), b.geom_solref);
        assert_eq!(round(&a.geom_solimp), b.geom_solimp);
        assert_eq!(round(&a.geom_friction), b.geom_friction);
        assert_eq!(round(&a.geom_margin), b.geom_margin);
        assert_eq!(round(&a.geom_gap), b.geom_gap);
        assert_eq!(a.geom_sameframe, b.geom_sameframe);
        assert_eq!(a.candidates.len(), b.candidates.len());
        assert_eq!(a.ncon_max, b.ncon_max);
        assert_eq!(a.nefc_max, b.nefc_max);
        assert_eq!(a.candidates_fingerprint, b.candidates_fingerprint);
        for (x, y) in a.candidates.iter().zip(&b.candidates) {
            assert_eq!((x.g1, x.g2, x.collider), (y.g1, y.g2, y.collider));
            assert_eq!(x.margin_gap as f32, y.margin_gap);
            assert_eq!(x.includemargin as f32, y.includemargin);
            assert_eq!(x.gap as f32, y.gap);
            assert_eq!(x.condim, y.condim);
            assert_eq!(round(&x.solref), y.solref.to_vec());
            assert_eq!(round(&x.solimp), y.solimp.to_vec());
            assert_eq!(round(&x.friction), y.friction.to_vec());
        }
    }
}
