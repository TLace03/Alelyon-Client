//! Inertial frames snapped onto the body frame: MuJoCo's `body_sameframe` (phase 1c-ii review).
//!
//! MuJoCo's compiler marks a body whose inertial pose is within `kFrameEps = 1e-6` of the body
//! frame BODY (`IsNullPose`: position and orientation) and one whose inertial orientation alone
//! is BODYROT; `mj_kinematics2` then calls `mj_local2Global` with that flag, which COPIES the
//! body's position and rotation instead of composing the inertial offset: `xipos` is `xpos`
//! exactly. The first port of this crate always composed, so a free body with
//! `<inertial pos="5e-7 0 0">` had a centre of mass 5e-7 from where MuJoCo puts it, and
//! `subtree_com`, `cdof`, `M`, `qacc` and the contact Jacobians followed (in this fixture `M`
//! at 2e-7 relative and `qacc` at 6e-5), far above the gates; no fixture had such a frame, so
//! nothing showed it.
//!
//! `contact_sameframe.xml` has free, hinge and slide bodies with inertial frames near the body
//! frame (BODY and BODYROT), a general one, and geoms of every `geom_sameframe` kind; its golden
//! file holds what MuJoCo 3.14.0 computed with contacts and constraints off.
//!
//! - the compiled flags and the simple-dof runs equal MuJoCo's, exactly;
//! - the frames, `M`, `qfrc_bias`, `qacc` and one Euler and one RK4 step equal MuJoCo's within
//!   1e-13 relative (bit for bit in `contacts_exact.rs`);
//! - the negative control: with every body's flag set to NONE, which is exactly what the first
//!   port computed, the comparison fails on `xipos` (6.9e-7), `ximat` (1.0e-6), `subtree_com`,
//!   `M` (3.1e-7), `qfrc_bias` (7.3e-6) and `qacc` (6.2e-4) and passes, exactly, on the body
//!   frames and the geom frames, which do not read the inertial frame.

mod common;

use common::contacts::*;
use common::*;
use sim_physics::{Data, SameFrame};

const RTOL: f64 = 1e-13;
const FLOOR: f64 = 1e-14;

fn code(s: SameFrame) -> i64 {
    i64::from(s.code())
}

#[test]
fn the_compiled_inertial_frame_flags_equal_mujocos() {
    let c = inertial_model::<f64>();
    let m = &c.model;
    let g = inertial_golden();
    let a = &g["arrays"];

    // MuJoCo's body_sameframe, including the world's
    let theirs = tints(&a["body_sameframe"]);
    let ours: Vec<i64> = m.body_sameframe.iter().map(|&s| code(s)).collect();
    assert_eq!(ours, theirs, "body_sameframe");
    let theirs = tints(&a["geom_sameframe"]);
    let ours: Vec<i64> = m.geom_sameframe.iter().map(|&s| code(s)).collect();
    assert_eq!(ours, theirs, "geom_sameframe");
    let theirs = tints(&a["dof_simplenum"]);
    let ours: Vec<i64> = m.dof_simplenum.iter().map(|&n| n as i64).collect();
    assert_eq!(ours, theirs, "dof_simplenum");

    // the inertial poses themselves (quaternions [x, y, z, w] in the golden file)
    let xyzw: Vec<f64> = m
        .body_iquat
        .chunks(4)
        .flat_map(|q| [q[1], q[2], q[3], q[0]])
        .collect();
    assert_eq!(safe_cmp(&xyzw, &farr(&a["body_iquat"])).abs, 0.0);
    assert_eq!(safe_cmp(&m.body_ipos, &farr(&a["body_ipos"])).abs, 0.0);

    // the fixture covers what it is for: BODY, BODYROT and NONE bodies, and all five geom kinds
    let body_kinds: std::collections::BTreeSet<i64> = ours_body_kinds(m);
    assert_eq!(
        body_kinds,
        [0, 1, 3].into_iter().collect(),
        "body_sameframe kinds"
    );
    let geom_kinds: std::collections::BTreeSet<i64> =
        m.geom_sameframe.iter().map(|&s| code(s)).collect();
    assert_eq!(
        geom_kinds,
        [0, 1, 2, 3, 4].into_iter().collect(),
        "geom_sameframe kinds"
    );
    println!(
        "MEASURED f64 sameframe: body_sameframe {:?}, geom_sameframe {:?}, dof_simplenum equal to MuJoCo's",
        m.body_sameframe
            .iter()
            .map(|&s| code(s))
            .collect::<Vec<_>>(),
        m.geom_sameframe
            .iter()
            .map(|&s| code(s))
            .collect::<Vec<_>>()
    );
}

/// The kinds of the real bodies (the world's `Body` aside).
fn ours_body_kinds(m: &sim_physics::Model<f64>) -> std::collections::BTreeSet<i64> {
    m.body_sameframe[1..].iter().map(|&s| code(s)).collect()
}

#[test]
fn snapped_inertial_frames_equal_mujocos_kinematics_and_dynamics() {
    let c = inertial_model::<f64>();
    let g = inertial_golden();
    let report = inertial_report(&c.model, &g, RTOL, FLOOR);
    report.print("f64", "sameframe");
    assert!(
        report.all_within(),
        "out of tolerance: {:?}",
        report.failures()
    );
}

/// A snapped frame is the body's EXACTLY (bit for bit), not merely close: the property MuJoCo's
/// copy gives and a composition cannot. Needs no golden file.
#[test]
fn a_snapped_inertial_frame_is_the_bodys_frame_to_the_bit() {
    let c = inertial_model::<f64>();
    let m = &c.model;
    let g = inertial_golden();
    for s in tstates(&g) {
        let mut d = Data::new(m);
        d.qpos = physics_qpos(m, &farr(&s["qpos"]));
        d.qvel = farr(&s["qvel"]);
        sim_physics::forward(m, &mut d);
        for b in 1..m.nbody {
            let (xpos, xipos) = (&d.xpos[3 * b..3 * b + 3], &d.xipos[3 * b..3 * b + 3]);
            let (xmat, ximat) = (&d.xmat[9 * b..9 * b + 9], &d.ximat[9 * b..9 * b + 9]);
            match m.body_sameframe[b] {
                SameFrame::Body => {
                    assert_eq!(xipos, xpos, "body {b}: xipos is xpos");
                    assert_eq!(ximat, xmat, "body {b}: ximat is xmat");
                }
                SameFrame::BodyRot => {
                    assert_eq!(ximat, xmat, "body {b}: ximat is xmat");
                    assert_ne!(xipos, xpos, "body {b}: the offset is kept");
                }
                _ => {
                    assert_ne!(xipos, xpos, "body {b}: a general inertial frame is offset");
                    assert_ne!(ximat, xmat, "body {b}: a general inertial frame is rotated");
                }
            }
        }
    }
}

/// The negative control. With every flag NONE the kinematics compose the inertial offset, which
/// is what the first port did for every body; the comparison with MuJoCo then fails where the
/// snap matters (`xipos` and what is built on it) by about the size of the offsets, and passes
/// where it does not.
#[test]
fn ignoring_body_sameframe_is_caught() {
    let c = inertial_model::<f64>();
    let g = inertial_golden();

    // the positive control: the same comparison passes with the flags as compiled
    let ok = inertial_report(&c.model, &g, RTOL, FLOOR);
    assert!(ok.all_within(), "{:?}", ok.failures());

    let mut broken = c.model.clone();
    broken.body_sameframe.fill(SameFrame::None);
    let report = inertial_report(&broken, &g, RTOL, FLOOR);
    report.print("f64", "sameframe, flags ignored (the first port)");
    let failures = report.failures();
    for name in ["xipos", "ximat", "subtree_com", "M", "qfrc_bias", "qacc"] {
        let e = report.get(name).unwrap_or_else(|| panic!("no {name}"));
        assert!(
            e.rel() > 1e-9 && failures.iter().any(|f| f.contains(name)),
            "{name}: relative error {:e} with the flags ignored",
            e.rel()
        );
    }
    // the body frames and the geoms of bodies that read no inertial frame are untouched
    for name in ["xpos", "xmat", "xquat"] {
        assert_eq!(report.get(name).unwrap().abs, 0.0, "{name}");
    }
    // by about the size of the offsets (5e-7 .. 7e-7 for the snapped bodies)
    let xipos = report.get("xipos").unwrap();
    assert!(
        xipos.abs > 1e-7 && xipos.abs < 1e-5,
        "xipos error {:e}",
        xipos.abs
    );
}
