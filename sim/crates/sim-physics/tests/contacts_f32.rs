//! `f32` with contacts, measured against MuJoCo 3.14.0's `f64` (phase 1c-ii): what a GPU port
//! starts from.
//!
//! - **Structure.** Per state, whether the contact structure of the `f32` engine equals MuJoCo's
//!   (`ncon`, the geom ids and order, `dim`, `exclude`, `efc_address`, and the rows' `nefc`,
//!   `efc_type`, `efc_id`). A state where it differs is reported with the differing contact and
//!   its `f64` distance to the threshold that decided it (`margin + gap` for whether the contact
//!   exists, `includemargin` for whether it is excluded).
//!   **Gated** (the phase-1c-ii review found the earlier test could pass with every state
//!   reclassified as "differs"): the number of states with equal structure is at least the
//!   measured floor (`FLOORS`), and the states that differ are a subset of the measured list
//!   (`UNEQUAL`, each with its cause), so a state that newly differs fails the test.
//! - **Solution.** On the states whose structure is equal: `qacc` of Newton against MuJoCo's
//!   converged optimum, gate 1e-3 relative to the larger of the optimum's largest acceleration
//!   and a tenth of the smooth acceleration's (`GATE_AT_REST`: a state at rest, whose optimum is
//!   tiny, is held to 1e-4 of the smooth acceleration, which is what an engine that rests must
//!   meet, and which its measured f32 error of about 1.5e-5 of it meets with a factor of six: no
//!   state is exempt); Newton's `solver_niter` in `f32` against `f64`; and the CG figures
//!   (measured, not gated).
//! - **Degenerate branches in `f32`.** MuJoCo's colliders test degenerate branches against the
//!   absolute `mjMINVAL = 1e-15`, which rounding residue in `f32` is far above: an upright cylinder
//!   on a plane and parallel capsules took the general branch on noise. `Real` carries scale-aware
//!   `f32` thresholds; the regression tests below hold an upright cylinder at every yaw, parallel
//!   capsules and a capsule along a box edge to the `f64` contact set.
//! - **The pile.** 2,400 steps in `f32` from its initial pose: the maximum box-box separating-axis
//!   penetration over the last second, the box-floor penetration and the speed at the end, against
//!   the `f64` figures (which are MuJoCo's: the `f64` run equals it to the last bit).

mod common;

use common::cons::*;
use common::contacts::*;
use common::*;
use serde_json::Value;
use sim_physics::{Data, Model, PrimalSolver, Real};

/// `|det|` of the two capsules' axes (`mjraw_CapsuleCapsule`'s test for parallel axes, against
/// `mjMINVAL = 1e-15` in `f64` and `1e-15 + 1e-6 ma mc` in `f32`), for the capsule-capsule
/// candidate of `g1` and `g2`.
fn capsule_det<R: Real>(m: &Model<R>, d: &Data<R>, g1: usize, g2: usize) -> f64 {
    let axis = |g: usize| {
        let (x, s) = (&d.geom_xmat[9 * g..9 * g + 9], m.geom_size[3 * g + 1]);
        [x[2] * s, x[5] * s, x[8] * s]
    };
    let dot = |a: [R; 3], b: [R; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let (a1, a2) = (axis(g1), axis(g2));
    let (ma, mb, mc) = (dot(a1, a1), -dot(a1, a2), dot(a2, a2));
    (ma * mc - mb * mb).abs().to_f64()
}

/// The squared length of `mjc_PlaneCylinder`'s `vec` (the cylinder axis, turned towards the plane,
/// scaled by its projection on the normal, minus the normal), whose comparison with
/// `Real::AXIS_RESIDUAL_SQR` (`mjMINVAL^2 = 1e-30` in `f64`, the machine epsilon in `f32`) decides
/// between the general configuration and the disk parallel to the plane: exactly 0 for an upright
/// cylinder in `f64`, a rounding residue of about 1e-14 in `f32`.
fn plane_cylinder_len_sqr<R: Real>(d: &Data<R>, plane: usize, cyl: usize) -> f64 {
    let (mp, mc) = (
        &d.geom_xmat[9 * plane..9 * plane + 9],
        &d.geom_xmat[9 * cyl..9 * cyl + 9],
    );
    let normal = [mp[2], mp[5], mp[8]];
    let mut axis = [mc[2], mc[5], mc[8]];
    let mut prj = normal[0] * axis[0] + normal[1] * axis[1] + normal[2] * axis[2];
    if prj > R::ZERO {
        axis = [-axis[0], -axis[1], -axis[2]];
        prj = -prj;
    }
    let v = [
        axis[0] * prj - normal[0],
        axis[1] * prj - normal[1],
        axis[2] * prj - normal[2],
    ];
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).to_f64()
}

/// Whether an `f32` number equals MuJoCo's `f64` one to the relative tolerance `tol` (against
/// `1 + |theirs|`): a micrometre and 1e-5 for a distance and a position, 1e-4 for a frame.
fn floats_close(ours: f64, theirs: f64, tol: f64) -> bool {
    (ours - theirs).abs() <= tol * (1.0 + theirs.abs())
}

/// The first difference of an `f32` contact structure from MuJoCo's, described with the `f64`
/// numbers of the contact that decided it (`None`: the structures are equal): the integers
/// (`ncon`, geoms and order, `dim`, `exclude`, `efc_address`, the rows' types), then every
/// contact's `dist`, `pos` and `frame` (a contact in another place is another structure, which a
/// collider that took another branch on a degenerate pose produces).
fn structure_difference<R: Real>(
    m: &Model<R>,
    d: &Data<R>,
    d64: &Data<f64>,
    s: &Value,
) -> Option<String> {
    let gc = s["contacts"].as_array().unwrap();
    let n = d.ncon.max(gc.len());
    let describe = |c: &Value| -> String {
        let g = tints(&c["geom"]);
        let (g1, g2) = (g[0] as usize, g[1] as usize);
        let cand = m
            .candidates
            .iter()
            .find(|k| k.g1 == g1 && k.g2 == g2)
            .expect("a candidate");
        let dist = f(&c["dist"]);
        let (inc, mg) = (f(&c["includemargin"]), cand.margin_gap.to_f64());
        let mut text = format!(
            "MuJoCo's contact between geoms {g1} and {g2} ({:?}): f64 dist {dist:e}, includemargin {inc:e}, margin+gap {mg:e}; distance to the exclude threshold {:e}, to the existence threshold {:e}",
            cand.collider,
            dist - inc,
            dist - mg
        );
        if cand.collider == sim_physics::Collider::CapsuleCapsule {
            text += &format!(
                "; |det| of the axes: f64 {:e}, f32 {:e}, against 1e-15 (f64) or 1e-15 + 1e-6 ma mc (f32): the parallel-axes branch needs it below",
                capsule_det(&m.rounded_to::<f64>(), d64, g1, g2),
                capsule_det(m, d, g1, g2)
            );
        }
        if cand.collider == sim_physics::Collider::PlaneCylinder {
            text += &format!(
                "; the squared length of the axis residual that decides the upright-cylinder branch: f64 {:e}, f32 {:e}, against 1e-30 (f64) or 1.2e-7 (f32)",
                plane_cylinder_len_sqr(d64, g1, g2),
                plane_cylinder_len_sqr(d, g1, g2)
            );
        }
        text
    };
    for i in 0..n {
        let ours = (i < d.ncon).then(|| {
            (
                [d.contact_geom[2 * i], d.contact_geom[2 * i + 1]],
                i64::from(d.contact_exclude[i]),
                d.contact_dim[i],
                i64::from(d.contact_efc_address[i]),
            )
        });
        let theirs = (i < gc.len()).then(|| {
            let g = tints(&gc[i]["geom"]);
            (
                [g[0] as usize, g[1] as usize],
                gc[i]["exclude"].as_i64().unwrap(),
                gc[i]["dim"].as_u64().unwrap() as usize,
                gc[i]["efc_address"].as_i64().unwrap(),
            )
        });
        if ours != theirs {
            let detail = if i < gc.len() {
                describe(&gc[i])
            } else {
                format!(
                    "an extra contact of ours between geoms {:?} (f32 dist {:e})",
                    ours.map(|o| o.0),
                    d.contact_dist[i].to_f64()
                )
            };
            return Some(format!(
                "contact {i}: ours {ours:?}, MuJoCo's {theirs:?}; {detail}; ncon ours {} vs {}; {}",
                d.ncon,
                gc.len(),
                set_note(d, gc)
            ));
        }
    }
    let n = d.nefc;
    let types: Vec<i64> = d.efc_type[..n].iter().map(|&t| type_code(t)).collect();
    if types != ints(&s["efc_type"]) {
        return Some("efc_type differs".into());
    }
    // the same contacts, in the same places
    for (i, c) in gc.iter().enumerate() {
        let (dist, pos, frame) = (f(&c["dist"]), farr(&c["pos"]), farr(&c["frame"]));
        let mut bad = Vec::new();
        if !floats_close(d.contact_dist[i].to_f64(), dist, 1e-5) {
            bad.push(format!("dist {:e} vs {dist:e}", d.contact_dist[i].to_f64()));
        }
        for (k, &theirs) in pos.iter().enumerate() {
            let ours = d.contact_pos[3 * i + k].to_f64();
            if !floats_close(ours, theirs, 1e-5) {
                bad.push(format!("pos[{k}] {ours:e} vs {theirs:e}"));
            }
        }
        for (k, &theirs) in frame.iter().enumerate() {
            let ours = d.contact_frame[9 * i + k].to_f64();
            if !floats_close(ours, theirs, 1e-4) {
                bad.push(format!("frame[{k}] {ours:e} vs {theirs:e}"));
            }
        }
        if !bad.is_empty() {
            return Some(format!(
                "contact {i} is in another place ({}); {}; {}",
                bad.join(", "),
                describe(c),
                set_note(d, gc)
            ));
        }
    }
    None
}

/// Whether `d`'s contacts are MuJoCo's `gc` as a SET (the same count, and each of MuJoCo's contacts
/// matched by one of ours with the same geoms and `dist` and `pos` within 1e-5), whatever their
/// order: told apart from a contact set that really differs.
fn same_contacts_in_another_order<R: Real>(d: &Data<R>, gc: &[Value]) -> bool {
    if d.ncon != gc.len() {
        return false;
    }
    let mut used = vec![false; d.ncon];
    for c in gc {
        let g = tints(&c["geom"]);
        let (dist, pos) = (f(&c["dist"]), farr(&c["pos"]));
        let found = (0..d.ncon).find(|&i| {
            !used[i]
                && d.contact_geom[2 * i] == g[0] as usize
                && d.contact_geom[2 * i + 1] == g[1] as usize
                && floats_close(d.contact_dist[i].to_f64(), dist, 1e-5)
                && (0..3).all(|k| floats_close(d.contact_pos[3 * i + k].to_f64(), pos[k], 1e-5))
        });
        match found {
            Some(i) => used[i] = true,
            None => return false,
        }
    }
    true
}

/// The note on a differing state: the same contacts in another order, or another set.
fn set_note<R: Real>(d: &Data<R>, gc: &[Value]) -> &'static str {
    if same_contacts_in_another_order(d, gc) {
        "as a SET the f32 contacts are MuJoCo's (dist and pos within 1e-5): only their ORDER differs"
    } else {
        "as a SET the f32 contacts differ from MuJoCo's"
    }
}

/// The states whose `f32` contact structure differs from MuJoCo's `f64` one: the measured list, and
/// what makes each differ (this file's classification of it, printed as a `MEASURED` line, shows
/// the same). A state that newly differs fails the test; a state that no longer differs does not
/// (better news than the list says), and the floor below keeps the equal count honest. Every
/// model not listed has equal structure in every state, both cones.
///
/// - stack 6 and zoo 37 (both box-box): the contacts are MuJoCo's AS A SET (`dist` and `pos`
///   within 1e-5), only the order differs: `f32` starts the clipped polygon at another vertex (a
///   tie between nearly equal candidates of an aligned pair that rounding decides); the cause
///   below that is UNMEASURED;
/// - zoo 25 (box-box): two boxes exactly touching, at a distance of 0 m, which is the
///   `septol` tie of the edge acceptance (`margin + gap + 1e-13 sum(half-sizes)` in `f64`, 1e-6 in
///   `f32`): `f32` generates 3 contacts in the gap (excluded, no rows) that `f64` does not.
const UNEQUAL: [(TWhich, &[usize], &str); 2] = [
    (
        TWhich::Stack,
        &[6],
        "box-box: the same contacts as a set, another order",
    ),
    (
        TWhich::Zoo,
        &[25, 37],
        "box-box: 25 three extra contacts in the gap at the septol tie; 37 the same contacts as a set, another order",
    ),
];

/// The listed states of `which`.
fn expected_unequal(which: TWhich) -> &'static [usize] {
    UNEQUAL
        .iter()
        .find(|(w, _, _)| *w == which)
        .map_or(&[], |(_, states, _)| states)
}

/// The gate on the `f32` solution: `qacc` within 1e-3 of the larger of the optimum's largest
/// acceleration and `GATE_AT_REST` times the smooth acceleration's. For a state in motion the
/// optimum is the larger and the gate is the plain 1e-3 relative; for a state at rest, where the
/// contact forces cancel the smooth acceleration (gravity) and the optimum is tiny, it is
/// 1e-3 * 0.1 = 1e-4 of the smooth acceleration, an absolute requirement an engine that rests
/// must meet (1e-3 m/s^2 for gravity: 2e-6 m/s after a step of 2 ms). No state is exempt.
const GATE_AT_REST: f64 = 0.1;

fn f32_model_run(which: TWhich, v: Variant) {
    let c32 = tcompile::<f32>(which, v);
    let c64 = tcompile::<f64>(which, v);
    let g = tgolden(which, v);
    let (mut equal, mut worst_newton, mut worst_cg, mut worst_cg64) =
        (0usize, 0.0f64, 0.0f64, 0.0f64);
    let (mut niter_equal, mut compared, mut worst_gate_ratio) = (0usize, 0usize, 0.0f64);
    let mut unequal = Vec::new();
    let mut at_rest = Vec::new();
    let nstates = tstates(&g).len();
    for (k, s) in tstates(&g).iter().enumerate() {
        let d32 = tforward(&c32.model, s, PrimalSolver::Newton);
        let d64 = tforward(&c64.model, s, PrimalSolver::Newton);
        match structure_difference(&c32.model, &d32, &d64, s) {
            None => {
                equal += 1;
                let reference = farr(&s["converged"]["qacc"]);
                let e = safe_cmp(&widen(&d32.qacc), &reference);
                let e64 = safe_cmp(&widen(&d64.qacc), &reference);
                // the gate's scale (see GATE_AT_REST)
                let smooth_max = d64.qacc_smooth.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                let scale = e.scale.max(GATE_AT_REST * smooth_max);
                let ratio = e.abs / (1e-3 * scale);
                let rest = e.scale < GATE_AT_REST * smooth_max;
                if rest {
                    at_rest.push(k);
                }
                worst_gate_ratio = worst_gate_ratio.max(ratio);
                worst_newton = worst_newton.max(e.rel());
                compared += 1;
                niter_equal += usize::from(d32.solver_niter == d64.solver_niter);
                println!(
                    "MEASURED f32 {} {} newton state {k}: contact structure equal to MuJoCo's; qacc absolute error vs MuJoCo's optimum {:e} = {ratio:.3} of the gate (1e-3 of {scale:e}; largest |qacc| of the optimum {:e}, of qacc_smooth {smooth_max:e}); relative to the optimum {:e} (f64 at the model's tolerance: {:e}){}; solver_niter f32={} f64={}",
                    which.name(),
                    v.name(),
                    e.abs,
                    e.scale,
                    e.rel(),
                    e64.rel(),
                    if rest {
                        " [at rest: held to 1e-4 of the smooth acceleration]"
                    } else {
                        ""
                    },
                    d32.solver_niter,
                    d64.solver_niter
                );
                assert!(
                    e.abs <= 1e-3 * scale,
                    "{} {} state {k}: f32 qacc error {:e} exceeds 1e-3 of {scale:e}",
                    which.name(),
                    v.name(),
                    e.abs
                );
                let cg32 = tforward(&c32.model, s, PrimalSolver::Cg);
                let cg64 = tforward(&c64.model, s, PrimalSolver::Cg);
                worst_cg = worst_cg.max(safe_cmp(&widen(&cg32.qacc), &reference).rel());
                worst_cg64 = worst_cg64.max(safe_cmp(&widen(&cg64.qacc), &reference).rel());
            }
            Some(why) => {
                println!(
                    "MEASURED f32 {} {} newton state {k}: contact structure DIFFERS from MuJoCo's: {why}",
                    which.name(),
                    v.name()
                );
                unequal.push(k);
            }
        }
    }
    println!(
        "MEASURED f32 {} {}: structure equal in {equal} of {nstates} states (differs in {unequal:?}); on those, worst Newton qacc relative error {worst_newton:e}, worst error {worst_gate_ratio:.3} of the gate over the {compared} gated states (states {at_rest:?} are at rest and held to 1e-4 of the smooth acceleration), Newton solver_niter f32 = f64 in {niter_equal} of {compared}, worst CG qacc relative error {worst_cg:e} (f64 CG: {worst_cg64:e}, not gated)",
        which.name(),
        v.name()
    );

    // the gates on the structure: no state may newly differ, and the equal count has a floor
    let allowed = expected_unequal(which);
    let new: Vec<usize> = unequal
        .iter()
        .copied()
        .filter(|k| !allowed.contains(k))
        .collect();
    assert!(
        new.is_empty(),
        "{} {}: states {new:?} newly differ from MuJoCo's contact structure in f32 (the measured list is {allowed:?})",
        which.name(),
        v.name()
    );
    assert!(
        equal >= nstates - allowed.len(),
        "{} {}: structure equal in only {equal} of {nstates} states (floor {})",
        which.name(),
        v.name(),
        nstates - allowed.len()
    );
    assert!(equal > 0 && compared == equal, "no state was gated");
}

macro_rules! f32_tests {
    ($($name:ident: $which:expr, $variant:expr;)*) => {
        $(
            #[test]
            fn $name() {
                f32_model_run($which, $variant);
            }
        )*
    };
}

f32_tests! {
    sphere_pyramidal_f32_is_measured: TWhich::Sphere, Variant::Pyramidal;
    sphere_elliptic_f32_is_measured: TWhich::Sphere, Variant::Elliptic;
    box_pyramidal_f32_is_measured: TWhich::Box, Variant::Pyramidal;
    box_elliptic_f32_is_measured: TWhich::Box, Variant::Elliptic;
    stack_pyramidal_f32_is_measured: TWhich::Stack, Variant::Pyramidal;
    stack_elliptic_f32_is_measured: TWhich::Stack, Variant::Elliptic;
    capsules_pyramidal_f32_is_measured: TWhich::Capsules, Variant::Pyramidal;
    capsules_elliptic_f32_is_measured: TWhich::Capsules, Variant::Elliptic;
    pile_pyramidal_f32_is_measured: TWhich::Pile, Variant::Pyramidal;
    pile_elliptic_f32_is_measured: TWhich::Pile, Variant::Elliptic;
    humanoid_pyramidal_f32_is_measured: TWhich::Humanoid, Variant::Pyramidal;
    humanoid_elliptic_f32_is_measured: TWhich::Humanoid, Variant::Elliptic;
    zoo_pyramidal_f32_is_measured: TWhich::Zoo, Variant::Pyramidal;
    zoo_elliptic_f32_is_measured: TWhich::Zoo, Variant::Elliptic;
}

/// The pile in `f32`: 2,400 steps, the penetration over the last second.
#[test]
fn the_pile_in_f32_penetrates_like_the_f64_pile() {
    for v in CONES {
        let g = tgolden(TWhich::Pile, v);
        let settle = &g["settle"];
        let total = settle["steps"].as_u64().unwrap() as usize;
        let frame_steps: Vec<usize> = settle["frames"]
            .as_array()
            .unwrap()
            .iter()
            .map(|fr| fr["step"].as_u64().unwrap() as usize)
            .collect();
        let c = tcompile::<f32>(TWhich::Pile, v);
        let m = &c.model;
        let mut d = Data::new(m);
        let (mut sat, mut floor, mut depth) = (0.0f64, 0.0f64, 0.0f64);
        for n in 1..=total {
            sim_physics::step(m, &mut d);
            assert_eq!(d.warning_collision_overflow, 0);
            if n > total - 240 {
                for i in 0..d.ncon {
                    depth = depth.max(-d.contact_dist[i].to_f64());
                }
            }
            if frame_steps.contains(&n) {
                let qpos = scene_qpos(m, &d.qpos);
                let (pos, quat) = pile_poses_scene(&qpos);
                let (bb, fl, _, _) = box_pile_penetration(&pos, &quat, 0.1);
                sat = sat.max(bb);
                floor = floor.max(fl);
            }
        }
        let speed = (0..m.nbody - 1)
            .map(|b| {
                let q = &d.qvel[6 * b..6 * b + 3];
                f64::from((q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt())
            })
            .fold(0.0f64, f64::max);
        let (mj_sat, mj_floor) = (
            f(&settle["sat_max_box_box_last_second"]),
            f(&settle["sat_max_box_floor_last_second"]),
        );
        println!(
            "MEASURED f32 pile ({}): maximum box-box SAT penetration over the last second {:.4} mm (f64 / MuJoCo: {:.4} mm), box-floor {:.4} mm ({:.4} mm), engine-reported depth {:.4} mm ({:.4} mm), maximum speed at 10 s {speed:.6} m/s ({:.6})",
            v.name(),
            1e3 * sat,
            1e3 * mj_sat,
            1e3 * floor,
            1e3 * mj_floor,
            1e3 * depth,
            1e3 * f(&settle["last_second_max_neg_dist"]),
            f(&settle["end_max_speed"]),
        );
        // the f32 pile meets ADR-0042's bar too
        assert!(sat <= 3e-3, "{}: {sat:e}", v.name());
        assert!(speed < 0.01, "{}: {speed}", v.name());
    }
}

// ---------------------------------------------------------------------------------------
// degenerate branches in f32 (the phase-1c-ii review's finding)
// ---------------------------------------------------------------------------------------
//
// MuJoCo tests `mjc_PlaneCylinder`'s "disk parallel to the plane" against `mjMINVAL^2 = 1e-30` and
// the parallel-axes branches of `mjraw_CapsuleCapsule` and `mjraw_CapsuleBox` against
// `|det| < mjMINVAL = 1e-15`. In `f32` the rounding residue of an upright cylinder's axis
// (`1 - 6e-8`) and of the determinant of parallel axes are far above those, so the general branch
// ran on noise. Measured before the scale-aware `f32` thresholds (`Real::AXIS_RESIDUAL_SQR`,
// `PARALLEL_DET_REL`, `TIE_REL`; the review's probe, 2026-10-02): an upright cylinder 1 mm into a
// plane made 3 contacts at -1 mm at all 360 whole-degree yaws in `f64` and was wrong at 74 of them
// in `f32` (one contact 12 cm deep, or none), where it was launched or sank 18 cm through the plane;
// parallel capsules made 1 contact in place of 2.

type Quat = [f64; 4];

fn qmul(a: Quat, b: Quat) -> Quat {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

/// The rotation of `deg` degrees about the unit `axis`, `[w, x, y, z]`.
fn qaxis(axis: [f64; 3], deg: f64) -> Quat {
    let h = deg.to_radians() / 2.0;
    [
        h.cos(),
        h.sin() * axis[0],
        h.sin() * axis[1],
        h.sin() * axis[2],
    ]
}

/// The column `k` of the rotation matrix of the unit quaternion `q`.
fn qcol(q: Quat, k: usize) -> [f64; 3] {
    let [w, x, y, z] = q;
    let m = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    [m[0][k], m[1][k], m[2][k]]
}

/// The model of a sweep XML (two geoms, a plane or a free body each).
fn sweep_model<R: Real>(file: &str) -> Model<R> {
    compile_constrained::<R>(load_scene(&fixtures().join(file))).model
}

/// The contacts of the pose `qpos` (the physics layout: `[x, y, z, w, qx, qy, qz]` per free body)
/// of a sweep model: `kinematics` then `collide`.
fn contacts_at<R: Real>(m: &Model<R>, qpos: &[f64]) -> Data<R> {
    let mut d = Data::new(m);
    d.qpos = qpos.iter().map(|&x| R::from_f64(x)).collect();
    sim_physics::kinematics(m, &mut d);
    sim_physics::collide(m, &mut d);
    assert_eq!(d.warning_collision_overflow, 0);
    d
}

/// The pose of a free body: position then orientation.
fn pose(p: [f64; 3], q: Quat) -> Vec<f64> {
    vec![p[0], p[1], p[2], q[0], q[1], q[2], q[3]]
}

/// An upright cylinder 1 mm into the plane, at every whole-degree yaw: `f64` makes three contacts
/// at -1 mm (the two upright rim points under the axis and the triangle point, all of them at the
/// bottom disk), `f32` must make the same.
#[test]
fn an_upright_cylinder_makes_its_contacts_in_f32_at_every_yaw() {
    let (m32, m64) = (
        sweep_model::<f32>("contact_sweep_plane_cylinder.xml"),
        sweep_model::<f64>("contact_sweep_plane_cylinder.xml"),
    );
    let (mut wrong32, mut wrong64) = (Vec::new(), Vec::new());
    for yaw in 0..360 {
        let q = qaxis([0.0, 0.0, 1.0], f64::from(yaw));
        let p = pose([0.0, 0.0, 0.099], q);
        let ok = |n: usize, dists: Vec<f64>, tol: f64| {
            n == 3 && dists.iter().all(|d| (d + 1e-3).abs() <= tol)
        };
        let d64 = contacts_at(&m64, &p);
        let d32 = contacts_at(&m32, &p);
        if !ok(d64.ncon, thead(&d64.contact_dist, d64.ncon), 1e-12) {
            wrong64.push(yaw);
        }
        if !ok(d32.ncon, thead(&d32.contact_dist, d32.ncon), 1e-6) {
            wrong32.push(yaw);
        }
    }
    println!(
        "MEASURED f32 upright cylinder on a plane, 360 whole-degree yaws, 1 mm in: 3 contacts at -1 mm in all but {} yaws in f64 and all but {} in f32 (before the scale-aware thresholds: 0 and 74)",
        wrong64.len(),
        wrong32.len()
    );
    assert!(wrong64.is_empty(), "f64: {wrong64:?}");
    assert!(wrong32.is_empty(), "f32: {wrong32:?}");
}

/// Tilts of the cylinder through the noise floor of its axis (1e-7 rad, where the axis residual is
/// pure rounding, to 1e-2): the deepest contact of `f32` stays within 5e-5 m of `f64`'s (the
/// threshold `sqrt(eps)` of the axis residual bounds the error by `r sqrt(eps) = 2e-5 m`; MuJoCo's
/// absolute test left up to 9 cm at 1e-7 and 1.4 mm at 1e-5).
#[test]
fn a_nearly_upright_cylinder_is_accurate_in_f32_through_the_noise_floor() {
    let (m32, m64) = (
        sweep_model::<f32>("contact_sweep_plane_cylinder.xml"),
        sweep_model::<f64>("contact_sweep_plane_cylinder.xml"),
    );
    let mut worst = (0.0f64, 0.0f64, 0.0f64);
    for yaw in [0.0, 30.0, 45.0, 90.0, 137.0] {
        for tilt in [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 1e-2] {
            // the tilt about the x axis, then the yaw about z
            let q = qmul(
                qaxis([0.0, 0.0, 1.0], yaw),
                qaxis([1.0, 0.0, 0.0], f64::to_degrees(tilt)),
            );
            let p = pose([0.0, 0.0, 0.099], q);
            let (d64, d32) = (contacts_at(&m64, &p), contacts_at(&m32, &p));
            assert_eq!(
                d32.ncon, d64.ncon,
                "yaw {yaw} tilt {tilt:e}: {} contacts in f32, {} in f64",
                d32.ncon, d64.ncon
            );
            let deepest = |d: &[f64]| d.iter().fold(0.0f64, |a, &x| a.min(x));
            let (a, b) = (
                deepest(&thead(&d64.contact_dist, d64.ncon)),
                deepest(&thead(&d32.contact_dist, d32.ncon)),
            );
            let e = (a - b).abs();
            if e > worst.0 {
                worst = (e, yaw, tilt);
            }
            assert!(
                e <= 5e-5,
                "yaw {yaw} tilt {tilt:e}: deepest contact {b:e} in f32, {a:e} in f64"
            );
        }
    }
    println!(
        "MEASURED f32 near-upright cylinder: the deepest contact is within {:e} m of f64's over 5 yaws and tilts from 0 to 1e-2 rad (worst at yaw {}, tilt {:e})",
        worst.0, worst.1, worst.2
    );
}

/// The upright cylinder dropped 2 mm above the plane at yaws where `f32` used to fail: it comes to
/// rest (speed below 0.01 m/s, the rest criterion of the engine comparison) within 0.2 mm of where
/// the `f64` run rests, in 3 s. Before: launched and still bouncing at 4 s (yaw 30 and 45), or sunk
/// 18 cm through the plane (yaw 90).
#[test]
fn a_dropped_upright_cylinder_comes_to_rest_in_f32_as_in_f64() {
    let (m32, m64) = (
        sweep_model::<f32>("contact_sweep_plane_cylinder.xml"),
        sweep_model::<f64>("contact_sweep_plane_cylinder.xml"),
    );
    fn run<R: Real>(m: &Model<R>, yaw: f64) -> (f64, f64) {
        let mut d = Data::new(m);
        let q = qaxis([0.0, 0.0, 1.0], yaw);
        d.qpos = pose([0.0, 0.0, 0.102], q)
            .iter()
            .map(|&x| R::from_f64(x))
            .collect();
        for _ in 0..1500 {
            sim_physics::step(m, &mut d);
        }
        assert_eq!(d.warning_collision_overflow, 0);
        let v = (0..3)
            .map(|k| d.qvel[k].to_f64().powi(2))
            .sum::<f64>()
            .sqrt();
        (d.qpos[2].to_f64(), v)
    }
    for yaw in [30.0, 45.0, 90.0] {
        let (z64, v64) = run(&m64, yaw);
        let (z32, v32) = run(&m32, yaw);
        println!(
            "MEASURED f32 upright cylinder dropped 2 mm at yaw {yaw}: after 3 s z = {z32:.6} m, speed {v32:.2e} m/s in f32; z = {z64:.6} m, {v64:.2e} m/s in f64"
        );
        assert!(
            v64 < 0.01 && v32 < 0.01,
            "yaw {yaw}: speeds {v64:e} {v32:e}"
        );
        assert!((z32 - z64).abs() < 2e-4, "yaw {yaw}: z {z32} vs {z64}");
    }
}

/// Parallel capsules (the same orientation, so exactly parallel axes in the matrices, rounding
/// residue aside) overlapping by 1 cm across their axes, shifted along them by 0, 0.1 and 0.2 m:
/// `f64` and `f32` must make the same contacts (the parallel branch keeps up to two). 240 seeded
/// orientations.
#[test]
fn parallel_capsules_keep_their_contacts_in_f32() {
    let (m32, m64) = (
        sweep_model::<f32>("contact_sweep_capsule_capsule.xml"),
        sweep_model::<f64>("contact_sweep_capsule_capsule.xml"),
    );
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let (mut cases, mut wrong, mut two) = (0usize, Vec::new(), 0usize);
    for _ in 0..240 {
        let mut q = [next(), next(), next(), next()];
        let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n < 0.3 {
            continue;
        }
        q.iter_mut().for_each(|x| *x /= n);
        let (across, along) = (qcol(q, 0), qcol(q, 2));
        // the second capsule is the first spun about its own axis: the same axis, other matrix entries
        let spin = 180.0 * next();
        for shift in [0.0, 0.1, 0.2] {
            let off = [
                0.09 * across[0] + shift * along[0],
                0.09 * across[1] + shift * along[1],
                0.09 * across[2] + shift * along[2],
            ];
            let qb = qmul(q, qaxis([0.0, 0.0, 1.0], spin));
            let qpos: Vec<f64> = [pose([0.0; 3], q), pose(off, qb)].concat();
            let (d64, d32) = (contacts_at(&m64, &qpos), contacts_at(&m32, &qpos));
            cases += 1;
            two += usize::from(d64.ncon == 2);
            let same = d32.ncon == d64.ncon
                && (0..d64.ncon)
                    .all(|i| (d32.contact_dist[i].to_f64() - d64.contact_dist[i]).abs() <= 1e-5);
            if !same {
                wrong.push((cases, d32.ncon, d64.ncon));
            }
        }
    }
    println!(
        "MEASURED f32 parallel capsules, {cases} poses (seeded orientations, shifts 0, 0.1, 0.2 m along the axes; f64 makes 2 contacts in {two}): f32 differs from f64 in {} (before: 1 contact in place of 2)",
        wrong.len()
    );
    assert!(
        two >= cases / 2,
        "the cases must exercise two-contact poses"
    );
    assert!(wrong.is_empty(), "f32 differs from f64: {wrong:?}");
}

/// The geometry of the zoo's `capsule_box_face` state (where `f32` used to put a contact 0.1 m off):
/// a capsule (radius 0.05, half-length 0.2) over a box (half-sizes 0.1, 0.2, 0.05) with its lowest
/// point `PEN` below the box's top face.
const CAP_H: f64 = 0.2;
const BOX_HX: f64 = 0.1;
const BOX_HY: f64 = 0.2;
const PEN: f64 = 0.002;

/// The model of that geometry: a free capsule and a free box.
fn capsule_box_model<R: Real>() -> Model<R> {
    let xml = r#"<mujoco><worldbody>
      <body name="a"><freejoint/><geom type="capsule" size="0.05 0.2" mass="1"/></body>
      <body name="b"><freejoint/><geom type="box" size="0.1 0.2 0.05" mass="1"/></body>
    </worldbody></mujoco>"#;
    let scene = sim_scene::mjcf::load(xml, std::env::temp_dir()).expect("imports");
    compile_constrained::<R>(scene).model
}

/// Whether every contact of `d` is a RIGHT contact of a capsule lying parallel to the top face of
/// the box (yawed by `yaw` degrees about the vertical at the origin) with its lowest point `PEN`
/// in: a `dist` of `-PEN` (within 1e-5), a normal along the face's (within 1e-4) and a position
/// over the face (within 1 mm of its footprint). No phantom contact (the review's wrong contact
/// was 0.1 m along the axis from the right place, outside the footprint); how many right contacts
/// there are is not asked.
fn contacts_are_right<R: Real>(d: &Data<R>, yaw: f64, world: [f64; 3]) -> bool {
    let (s, c) = yaw.to_radians().sin_cos();
    (0..d.ncon).all(|i| {
        let (px, py) = (
            d.contact_pos[3 * i].to_f64() - world[0],
            d.contact_pos[3 * i + 1].to_f64() - world[1],
        );
        // the position in the box's frame
        let (bx, by) = (c * px + s * py, -s * px + c * py);
        (d.contact_dist[i].to_f64() + PEN).abs() <= 1e-5
            && d.contact_frame[9 * i + 2].to_f64().abs() >= 1.0 - 1e-4
            && bx.abs() <= BOX_HX + 1e-3
            && by.abs() <= BOX_HY + 1e-3
    })
}

/// The pose of a capsule over the top face of the box (its lowest point `PEN` in) with its axis along
/// the box's x axis, turned by `turn` degrees about the vertical and tilted by `tilt` degrees out
/// of the horizontal (about the box's y axis: the end at +x goes down), the pair yawed by `yaw`, the
/// capsule offset by `(dx, dy)` in the box's frame and the box centred at `world` (the zoo's state has it at
/// (3, 0, 0.05): the position rounding of `f32` is part of what the pose tests).
fn capsule_on_box(world: [f64; 3], yaw: f64, turn: f64, tilt: f64, dx: f64, dy: f64) -> Vec<f64> {
    let qbox = qaxis([0.0, 0.0, 1.0], yaw);
    // the capsule's axis (local z) onto the box's x axis (90 degrees about y, 90 + tilt in all),
    // then turned
    let qcap = qmul(
        qmul(qbox, qaxis([0.0, 0.0, 1.0], turn)),
        qaxis([0.0, 1.0, 0.0], 90.0 + tilt),
    );
    let (ex, ey) = (qcol(qbox, 0), qcol(qbox, 1));
    // the box's top is at 0.05, the capsule's radius 0.05; the end that goes down drops by H sin(tilt)
    let z = 0.05 + 0.05 - PEN + CAP_H * tilt.to_radians().sin().abs();
    let p = [dx * ex[0] + dy * ey[0], dx * ex[1] + dy * ey[1], z];
    let at = |v: [f64; 3]| [v[0] + world[0], v[1] + world[1], v[2] + world[2]];
    [pose(at(p), qcap), pose(world, qbox)].concat()
}

/// A capsule lying along a box edge (its axis the box's x axis, to rounding) on the box's top face,
/// at several yaws: `mjraw_CapsuleBox` skips an edge parallel to the capsule axis by its determinant
/// (`|det| < 1e-15`, plus `1e-6 ma mc` in `f32`); the residue in `f32` was far above the absolute
/// test and put the nearest point 0.1 m off (the review's zoo state 20).
///
/// What can be held in a pose whose capsule axis is parallel to the box top (the exactly aligned
/// one, and the same turned about the vertical, which stays parallel to the face) is that `f32`
/// makes no phantom contact and loses none entirely: every contact, in `f32` and in `f64`, is a
/// right one (a `dist` of `-PEN` within 1e-5, the face's normal, a position over the face), there is at
/// least one, and, aligned with the edge, the FIRST contact is the same in both (the tie rule decides it).
/// The COUNT cannot be held: the second contact of `mjraw_CapsuleBox` is
/// chosen by `axisdir`, the signs of the capsule axis components in the box frame
/// (`halfaxis[i] > 0`), and for an axis parallel to the face one of those components is rounding
/// noise, and the candidate edges and faces tie in distance; `f64` itself makes 1 or 2 contacts
/// under a tilt of 1e-12 rad (measured below), so the number of contacts in such a pose is a
/// property of the noise, in `f64` as in `f32`. Tilted by 2 degrees or more out of the face's plane
/// the pose has no ties, and `f32` must make the same contacts as `f64` (count, `dist` within 1e-5,
/// `pos` within 1e-3). The pair is also placed away from the origin, at the zoo's (3, 0, 0.05) and
/// at (-2.3, 1.7, 0.05): the `f32` rounding of the positions is part of what is tested.
#[test]
fn a_capsule_along_a_box_edge_makes_no_wrong_contact_in_f32() {
    let (m32, m64) = (capsule_box_model::<f32>(), capsule_box_model::<f64>());
    let yaws = [0.0, 7.0, 30.0, 45.0, 90.0, 123.0, 200.0, 300.0];
    let offsets = [(0.0, 0.0), (0.02, 0.0), (0.0, 0.03), (-0.03, 0.02)];
    let worlds = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.05], [-2.3, 1.7, 0.05]];

    // parallel to the face (aligned with the edge, or turned about the vertical): no phantom
    // contact, none lost entirely
    let (mut bad, mut cases, mut fewer) = (Vec::new(), 0usize, 0usize);
    for world in worlds {
        for yaw in yaws {
            for turn in [0.0, 3.0, -7.0] {
                for (dx, dy) in offsets {
                    let qpos = capsule_on_box(world, yaw, turn, 0.0, dx, dy);
                    let (d64, d32) = (contacts_at(&m64, &qpos), contacts_at(&m32, &qpos));
                    cases += 1;
                    fewer += usize::from(d32.ncon < d64.ncon);
                    let right = contacts_are_right(&d32, yaw, world)
                        && contacts_are_right(&d64, yaw, world);
                    // aligned with the edge, the FIRST contact is decided by the tie rule
                    // (`dist2 < bestdist - mjMINVAL`: the first of the candidates at equal
                    // distance wins), which `f64` and `f32` must apply alike: the same first
                    // contact (the review's zoo state 20 put it 0.1 m along the axis)
                    let first = turn != 0.0
                        || (d32.ncon > 0
                            && d64.ncon > 0
                            && (d32.contact_dist[0].to_f64() - d64.contact_dist[0]).abs() <= 1e-5
                            && (0..3).all(|k| {
                                (d32.contact_pos[k].to_f64() - d64.contact_pos[k]).abs() <= 1e-3
                            }));
                    if d32.ncon == 0 || d64.ncon == 0 || !right || !first {
                        bad.push((world, yaw, turn, dx, dy, d32.ncon, d64.ncon));
                    }
                }
            }
        }
    }
    println!(
        "MEASURED f32 capsule parallel to a box face (along the edge or turned), {cases} poses: every contact in f32 and in f64 is a right one (dist -2 mm, the face's normal, over the face), and there is one in {} poses ({} poses make fewer contacts in f32 than in f64, and others more: the second contact follows the sign of rounding noise)",
        cases - bad.len(),
        fewer
    );
    assert!(bad.is_empty(), "a wrong or a lost contact: {bad:?}");

    // f64 itself: a tilt of 1e-12 rad changes the number of contacts of the exact pose
    let mut counts = std::collections::BTreeSet::new();
    for yaw in yaws {
        for tilt in [-1e-12f64, 0.0, 1e-12] {
            for axis in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
                let mut qpos = capsule_on_box([0.0; 3], yaw, 0.0, 0.0, 0.0, 0.0);
                let q = qmul(
                    [qpos[3], qpos[4], qpos[5], qpos[6]],
                    qaxis(axis, tilt.to_degrees()),
                );
                qpos[3..7].copy_from_slice(&q);
                counts.insert(contacts_at(&m64, &qpos).ncon);
            }
        }
    }
    println!(
        "MEASURED f64 capsule along a box edge: under tilts of 1e-12 rad the exact pose makes {counts:?} contacts (MuJoCo's own arithmetic, f64)"
    );
    assert!(
        counts.len() > 1,
        "if f64 never flips, the relaxation above is not justified: {counts:?}"
    );

    // tilted out of the face's plane, the pose has no ties: the same contacts
    let (mut wrong, mut cases) = (Vec::new(), 0usize);
    for world in worlds {
        for yaw in yaws {
            for tilt in [2.0, 5.0, 15.0, -2.0, -5.0, -15.0] {
                for turn in [0.0, 3.0, -7.0] {
                    for (dx, dy) in offsets {
                        let qpos = capsule_on_box(world, yaw, turn, tilt, dx, dy);
                        let (d64, d32) = (contacts_at(&m64, &qpos), contacts_at(&m32, &qpos));
                        cases += 1;
                        let same = d32.ncon == d64.ncon
                            && (0..d64.ncon).all(|i| {
                                (d32.contact_dist[i].to_f64() - d64.contact_dist[i]).abs() <= 1e-5
                                    && (0..3).all(|k| {
                                        (d32.contact_pos[3 * i + k].to_f64()
                                            - d64.contact_pos[3 * i + k])
                                            .abs()
                                            <= 1e-3
                                    })
                            });
                        if !same {
                            wrong.push((world, yaw, tilt, turn, dx, dy, d32.ncon, d64.ncon));
                        }
                    }
                }
            }
        }
    }
    println!(
        "MEASURED f32 capsule tilted 2 to 15 degrees out of a box face's plane, {cases} poses: f32 differs from f64 (count, dist within 1e-5, pos within 1e-3) in {}",
        wrong.len()
    );
    assert!(wrong.is_empty(), "f32 differs from f64: {wrong:?}");
}
