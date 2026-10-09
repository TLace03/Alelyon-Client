//! Parity of the MJCF importer with MuJoCo's own compiler, on `humanoid.xml`.
//!
//! `fixtures/mujoco/humanoid.xml` is MuJoCo's `model/humanoid/humanoid.xml`
//! (commit a8373cc4e, Apache-2.0; see `fixtures/mujoco/NOTICE`).
//! `fixtures/mujoco/humanoid_golden.json` is what MuJoCo 3.14.0 compiled it to,
//! written by `tools/sim_scene_mujoco_golden.py`. This test imports the same XML
//! with `sim_scene::mjcf::load` and compares every value in the golden file.
//!
//! Tolerances (the spec's): 1e-9 relative on everything, 1e-6 relative on the
//! inertia diagonal; quaternions are compared up to sign. A relative tolerance
//! cannot judge a number that MuJoCo computes as exactly 0 and a port as 1e-18, so
//! every comparison also allows an absolute slack of 1e-12 (the golden values are
//! O(1) metres or kilograms, so that slack is 1e-12 of the scale of the model).
//! The worst relative error of each family is printed (`cargo test -- --nocapture`)
//! and asserted to be at most the tolerance.

use std::fs;
use std::path::PathBuf;

use serde_json::Value;
use sim_scene::{ActuatorKind, JointKind, Scene, Shape, mjcf};

const RTOL: f64 = 1e-9;
const INERTIA_RTOL: f64 = 1e-6;
const ATOL: f64 = 1e-12;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mujoco")
}

fn golden() -> Value {
    let text = fs::read_to_string(dir().join("humanoid_golden.json")).expect("golden file");
    serde_json::from_str(&text).expect("golden JSON")
}

fn xml() -> String {
    fs::read_to_string(dir().join("humanoid.xml")).expect("humanoid.xml")
}

/// FNV-1a, 64 bit: the same function `tools/sim_scene_mujoco_golden.py` uses.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

fn f(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

fn farr<const N: usize>(v: &Value) -> [f64; N] {
    let a = v.as_array().expect("array");
    assert_eq!(a.len(), N);
    let mut out = [0.0; N];
    for (o, x) in out.iter_mut().zip(a) {
        *o = f(x);
    }
    out
}

thread_local! {
    /// The largest absolute difference this test thread has compared (each test
    /// runs on its own thread), printed beside the relative error.
    static MAX_ABS: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

fn track_abs(d: f64) {
    MAX_ABS.with(|m| m.set(m.get().max(d)));
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn worst_abs() -> f64 {
    MAX_ABS.with(std::cell::Cell::get)
}

/// Relative error with an absolute floor.
fn rel(a: f64, b: f64) -> f64 {
    let diff = (a - b).abs();
    track_abs(diff);
    if diff <= ATOL {
        return 0.0;
    }
    diff / a.abs().max(b.abs())
}

fn rel_quiet(a: f64, b: f64) -> f64 {
    let diff = (a - b).abs();
    if diff <= ATOL {
        return 0.0;
    }
    diff / a.abs().max(b.abs())
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    track_abs(max_abs_diff(a, b));
    a.iter()
        .zip(b)
        .map(|(x, y)| rel_quiet(*x, *y))
        .fold(0.0, f64::max)
}

/// Quaternion error up to sign.
fn quat_rel(a: [f64; 4], b: [f64; 4]) -> f64 {
    let neg = [-b[0], -b[1], -b[2], -b[3]];
    let quiet = |x: &[f64], y: &[f64]| {
        x.iter()
            .zip(y)
            .map(|(p, q)| rel_quiet(*p, *q))
            .fold(0.0, f64::max)
    };
    track_abs(max_abs_diff(&a, &b).min(max_abs_diff(&a, &neg)));
    quiet(&a, &b).min(quiet(&a, &neg))
}

#[derive(Default)]
struct Worst {
    pos: f64,
    quat: f64,
    mass: f64,
    inertia: f64,
    inertia_quat: f64,
}

fn load() -> Scene {
    mjcf::load(&xml(), dir()).expect("humanoid.xml imports")
}

#[test]
fn golden_matches_the_fixture_bytes() {
    let g = golden();
    let bytes = fs::read(dir().join("humanoid.xml")).unwrap();
    assert_eq!(g["xml"]["bytes"].as_u64().unwrap(), bytes.len() as u64);
    assert_eq!(
        g["xml"]["fnv1a64"].as_str().unwrap(),
        format!("{:016x}", fnv1a64(&bytes)),
        "humanoid.xml changed without regenerating the golden file (tools/sim_scene_mujoco_golden.py)"
    );
    assert_eq!(g["mujoco_version"], "3.14.0");
    assert_eq!(g["quat_order"], "xyzw");
}

#[test]
fn counts_and_options_match_mujoco() {
    let g = golden();
    let scene = load();
    let c = &g["counts"];
    // MuJoCo counts the world body; the scene does not list it.
    assert_eq!(
        scene.bodies.len() + 1,
        c["nbody"].as_u64().unwrap() as usize
    );
    assert_eq!(scene.joints.len(), c["njnt"].as_u64().unwrap() as usize);
    assert_eq!(scene.geoms.len(), c["ngeom"].as_u64().unwrap() as usize);
    assert_eq!(scene.actuators.len(), c["nu"].as_u64().unwrap() as usize);
    assert_eq!(scene.tendons.len(), c["ntendon"].as_u64().unwrap() as usize);
    assert_eq!(scene.nq(), c["nq"].as_u64().unwrap() as usize);
    assert_eq!(scene.nv(), c["nv"].as_u64().unwrap() as usize);
    assert_eq!(scene.timestep_s, f(&g["option"]["timestep"]));
    assert_eq!(scene.gravity, farr::<3>(&g["option"]["gravity"]));
}

#[test]
fn every_body_matches_mujoco() {
    let g = golden();
    let scene = load();
    let gb = g["bodies"].as_array().unwrap();
    assert_eq!(gb[0]["name"], "world");
    let mut w = Worst::default();
    for (i, body) in scene.bodies.iter().enumerate() {
        let gold = &gb[i + 1];
        assert_eq!(body.name, gold["name"].as_str().unwrap(), "body {i} name");
        // parent, by name
        let parent_name = match body.parent {
            None => "world".to_string(),
            Some(p) => scene.bodies[p.index()].name.clone(),
        };
        assert_eq!(
            parent_name,
            gold["parent"].as_str().unwrap(),
            "{} parent",
            body.name
        );

        w.pos = w.pos.max(max_rel(&body.pos, &farr::<3>(&gold["pos"])));
        w.quat = w.quat.max(quat_rel(body.quat, farr::<4>(&gold["quat"])));

        let (mass, com, diag, iquat) = match &body.inertial {
            Some(x) => (x.mass_kg, x.com, x.diag_inertia, x.inertia_quat),
            None => (0.0, [0.0; 3], [0.0; 3], [0.0, 0.0, 0.0, 1.0]),
        };
        w.mass = w.mass.max(rel(mass, f(&gold["mass"])));
        w.pos = w.pos.max(max_rel(&com, &farr::<3>(&gold["com"])));
        w.inertia = w.inertia.max(max_rel(&diag, &farr::<3>(&gold["inertia"])));
        w.inertia_quat = w
            .inertia_quat
            .max(quat_rel(iquat, farr::<4>(&gold["inertia_quat"])));
    }
    println!(
        "bodies: worst relative error  pos/com {:.2e}  quat {:.2e}  mass {:.2e}  inertia {:.2e}  inertia_quat {:.2e}  (worst absolute difference {:.2e})",
        w.pos,
        w.quat,
        w.mass,
        w.inertia,
        w.inertia_quat,
        worst_abs()
    );
    assert!(w.pos <= RTOL, "body/com position: {}", w.pos);
    assert!(w.quat <= RTOL, "body quaternion: {}", w.quat);
    assert!(w.mass <= RTOL, "mass: {}", w.mass);
    assert!(w.inertia <= INERTIA_RTOL, "inertia diagonal: {}", w.inertia);
    assert!(
        w.inertia_quat <= INERTIA_RTOL,
        "inertia quaternion: {}",
        w.inertia_quat
    );
}

#[test]
fn every_joint_matches_mujoco() {
    let g = golden();
    let scene = load();
    let gj = g["joints"].as_array().unwrap();
    let mut worst = 0.0f64;
    for (i, joint) in scene.joints.iter().enumerate() {
        let gold = &gj[i];
        assert_eq!(joint.name, gold["name"].as_str().unwrap(), "joint {i} name");
        assert_eq!(
            scene.bodies[joint.body.index()].name,
            gold["body"].as_str().unwrap(),
            "{} body",
            joint.name
        );
        let (kind, axis) = match joint.kind {
            JointKind::Free => ("free", [0.0, 0.0, 1.0]),
            JointKind::Ball => ("ball", [0.0, 0.0, 1.0]),
            JointKind::Hinge { axis } => ("hinge", axis),
            JointKind::Slide { axis } => ("slide", axis),
        };
        assert_eq!(kind, gold["type"].as_str().unwrap(), "{} type", joint.name);
        worst = worst.max(max_rel(&axis, &farr::<3>(&gold["axis"])));
        worst = worst.max(max_rel(&joint.pos, &farr::<3>(&gold["pos"])));
        match (joint.range, gold["range"].is_null()) {
            (None, true) => {}
            (Some(r), false) => worst = worst.max(max_rel(&r, &farr::<2>(&gold["range"]))),
            (r, _) => panic!("{} range {:?} vs {}", joint.name, r, gold["range"]),
        }
        worst = worst.max(rel(joint.stiffness, f(&gold["stiffness"])));
        worst = worst.max(rel(joint.damping, f(&gold["damping"])));
        worst = worst.max(rel(joint.armature, f(&gold["armature"])));
        worst = worst.max(rel(joint.frictionloss, f(&gold["frictionloss"])));
    }
    println!(
        "joints: worst relative error {worst:.2e} (worst absolute difference {:.2e})",
        worst_abs()
    );
    assert!(worst <= RTOL, "joints: {worst}");
}

#[test]
fn every_geom_matches_mujoco() {
    let g = golden();
    let scene = load();
    let gg = g["geoms"].as_array().unwrap();
    let mut worst = 0.0f64;
    for (i, geom) in scene.geoms.iter().enumerate() {
        let gold = &gg[i];
        assert_eq!(geom.name, gold["name"].as_str().unwrap(), "geom {i} name");
        let body = match geom.body {
            None => "world",
            Some(b) => scene.bodies[b.index()].name.as_str(),
        };
        assert_eq!(body, gold["body"].as_str().unwrap(), "{} body", geom.name);
        let (kind, size) = match geom.shape {
            Shape::Sphere { r } => ("sphere", [r, 0.0, 0.0]),
            Shape::Capsule { r, half_len } => ("capsule", [r, half_len, 0.0]),
            Shape::Cylinder { r, half_len } => ("cylinder", [r, half_len, 0.0]),
            Shape::Box { half } => ("box", half),
            Shape::Ellipsoid { radii } => ("ellipsoid", radii),
            Shape::Plane { size } => ("plane", size),
            Shape::Mesh { .. } => ("mesh", [0.0; 3]),
        };
        assert_eq!(kind, gold["type"].as_str().unwrap(), "{} type", geom.name);
        worst = worst.max(max_rel(&size, &farr::<3>(&gold["size"])));
        worst = worst.max(max_rel(&geom.pos, &farr::<3>(&gold["pos"])));
        worst = worst.max(quat_rel(geom.quat, farr::<4>(&gold["quat"])));
        worst = worst.max(max_rel(&geom.friction, &farr::<3>(&gold["friction"])));
        assert_eq!(u64::from(geom.contype), gold["contype"].as_u64().unwrap());
        assert_eq!(
            u64::from(geom.conaffinity),
            gold["conaffinity"].as_u64().unwrap()
        );
        assert_eq!(u64::from(geom.condim), gold["condim"].as_u64().unwrap());
        check_contact_params(geom, gold);
    }
    println!(
        "geoms: worst relative error {worst:.2e} (worst absolute difference {:.2e})",
        worst_abs()
    );
    assert!(worst <= RTOL, "geoms: {worst}");
}

/// A geom's contact parameters (`geom_solref`, `geom_solimp`, `geom_solmix`,
/// `geom_priority`, `geom_margin`, `geom_gap`): copied from the XML with no arithmetic,
/// so they equal MuJoCo's compiled values EXACTLY.
fn check_contact_params(geom: &sim_scene::Geom, gold: &Value) {
    let what = &geom.name;
    assert_eq!(geom.solref, farr::<2>(&gold["solref"]), "{what} solref");
    assert_eq!(geom.solimp, farr::<5>(&gold["solimp"]), "{what} solimp");
    assert_eq!(geom.solmix, f(&gold["solmix"]), "{what} solmix");
    assert_eq!(
        i64::from(geom.priority),
        gold["priority"].as_i64().unwrap(),
        "{what} priority"
    );
    assert_eq!(geom.margin, f(&gold["margin"]), "{what} margin");
    assert_eq!(geom.gap, f(&gold["gap"]), "{what} gap");
}

/// The scene's contact exclusions as MuJoCo's `exclude_signature` holds them
/// (`(min body << 16) + max body`, sorted), decoded to body names (`world` is body 0).
fn exclude_names(scene: &Scene) -> Vec<[String; 2]> {
    let internal = |b: Option<sim_scene::BodyId>| b.map_or(0usize, |b| b.index() + 1);
    let name = |i: usize| {
        if i == 0 {
            "world".to_string()
        } else {
            scene.bodies[i - 1].name.clone()
        }
    };
    let mut sigs: Vec<usize> = scene
        .contact_excludes
        .iter()
        .map(|x| {
            let (a, b) = (internal(x.body1), internal(x.body2));
            (a.min(b) << 16) + a.max(b)
        })
        .collect();
    sigs.sort_unstable();
    sigs.iter()
        .map(|s| [name(s >> 16), name(s & 0xFFFF)])
        .collect()
}

fn golden_excludes(v: &Value) -> Vec<[String; 2]> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| {
            [
                p[0].as_str().unwrap().to_string(),
                p[1].as_str().unwrap().to_string(),
            ]
        })
        .collect()
}

#[test]
fn the_humanoids_contact_exclusions_match_mujoco() {
    let g = golden();
    let scene = load();
    assert_eq!(scene.contact_excludes.len(), 2);
    assert_eq!(
        exclude_names(&scene),
        golden_excludes(&g["excludes"]),
        "exclude_signature decoded to body names"
    );
    // the two exclusions are read, not recorded
    assert!(
        scene
            .unsupported
            .iter()
            .all(|u| u.path != "contact/exclude")
    );
}

#[test]
fn the_contact_zoos_contact_parameters_and_exclusions_match_mujoco() {
    // `contact_zoo.xml` (ours): non-default solref (standard and direct), solimp (with a
    // power of 3), solmix, priority, margin, gap, friction and condim on geoms, bit groups,
    // an exclusion and a non-default impratio
    let g = golden();
    let case = &g["contact_case"];
    let bytes = fs::read(dir().join("contact_zoo.xml")).unwrap();
    assert_eq!(case["xml"]["bytes"].as_u64().unwrap(), bytes.len() as u64);
    assert_eq!(
        case["xml"]["fnv1a64"].as_str().unwrap(),
        format!("{:016x}", fnv1a64(&bytes)),
        "contact_zoo.xml changed without regenerating the golden file"
    );
    let scene = mjcf::load(&String::from_utf8(bytes).unwrap(), dir()).expect("imports");
    assert_eq!(
        scene.geoms.len() as u64,
        case["counts"]["ngeom"].as_u64().unwrap()
    );
    assert_eq!(
        scene.bodies.len() as u64 + 1,
        case["counts"]["nbody"].as_u64().unwrap()
    );
    for (geom, gold) in scene.geoms.iter().zip(case["geoms"].as_array().unwrap()) {
        assert_eq!(geom.name, gold["name"].as_str().unwrap());
        assert_eq!(u64::from(geom.contype), gold["contype"].as_u64().unwrap());
        assert_eq!(
            u64::from(geom.conaffinity),
            gold["conaffinity"].as_u64().unwrap()
        );
        assert_eq!(u64::from(geom.condim), gold["condim"].as_u64().unwrap());
        assert_eq!(geom.friction, farr::<3>(&gold["friction"]), "friction");
        check_contact_params(geom, gold);
    }
    assert_eq!(
        exclude_names(&scene),
        golden_excludes(&case["excludes"]),
        "exclude_signature decoded to body names"
    );
    check_options(&scene, &case["option"]);
    // the values really are not the defaults (so equal-to-MuJoCo is not vacuous)
    assert_eq!(scene.options.impratio, 5.0);
    let by = |n: &str| scene.geoms.iter().find(|x| x.name == n).unwrap();
    assert_eq!(by("cb").solref, [0.01, 0.8]);
    assert_eq!(by("cb").priority, 1);
    assert_eq!(by("bb").solref, [-2000.0, -30.0]);
    assert_eq!(by("bb").solimp[4], 3.0);
    assert_eq!(by("sa").solmix, 0.0);
    assert_eq!(by("sb1").solmix, 3.0);
    assert_eq!(by("cy").margin, 0.01);
    assert_eq!(by("cy").gap, 0.02);
    let conds: std::collections::BTreeSet<u32> = scene.geoms.iter().map(|x| x.condim).collect();
    assert_eq!(conds.into_iter().collect::<Vec<_>>(), vec![1, 3, 4, 6]);
    assert_eq!(scene.contact_excludes.len(), 1);
    // nothing of the model is left unsupported, and it imports under strict
    assert!(scene.unsupported.is_empty(), "{:?}", scene.unsupported);
    mjcf::load_with(
        &String::from_utf8(fs::read(dir().join("contact_zoo.xml")).unwrap()).unwrap(),
        dir(),
        &mjcf::LoadOptions {
            strict: true,
            instances: true,
        },
    )
    .expect("a fully modelled scene imports under strict");
}

#[test]
fn every_actuator_and_tendon_matches_mujoco() {
    let g = golden();
    let scene = load();
    let ga = g["actuators"].as_array().unwrap();
    let mut worst = 0.0f64;
    for (i, act) in scene.actuators.iter().enumerate() {
        let gold = &ga[i];
        assert_eq!(act.name, gold["name"].as_str().unwrap());
        assert_eq!(
            scene.joints[act.joint.index()].name,
            gold["joint"].as_str().unwrap()
        );
        let ActuatorKind::Motor { gear, ctrlrange } = act.kind else {
            panic!("{} is not a motor", act.name);
        };
        let gear6 = farr::<6>(&gold["gear"]);
        worst = worst.max(rel(gear, gear6[0]));
        assert!(gear6[1..].iter().all(|&x| x == 0.0));
        match (ctrlrange, gold["ctrlrange"].is_null()) {
            (None, true) => {}
            (Some(r), false) => worst = worst.max(max_rel(&r, &farr::<2>(&gold["ctrlrange"]))),
            (r, _) => panic!("{} ctrlrange {:?} vs {}", act.name, r, gold["ctrlrange"]),
        }
    }
    let gt = g["tendons"].as_array().unwrap();
    for (i, t) in scene.tendons.iter().enumerate() {
        let gold = &gt[i];
        assert_eq!(t.name, gold["name"].as_str().unwrap());
        match (t.range, gold["range"].is_null()) {
            (None, true) => {}
            (Some(r), false) => worst = worst.max(max_rel(&r, &farr::<2>(&gold["range"]))),
            (r, _) => panic!("{} range {:?} vs {}", t.name, r, gold["range"]),
        }
        let terms = gold["joints"].as_array().unwrap();
        assert_eq!(t.joints.len(), terms.len());
        for (a, b) in t.joints.iter().zip(terms) {
            assert_eq!(
                scene.joints[a.joint.index()].name,
                b["joint"].as_str().unwrap()
            );
            worst = worst.max(rel(a.coef, f(&b["coef"])));
        }
        worst = worst.max(rel(t.stiffness, f(&gold["stiffness"])));
        worst = worst.max(rel(t.damping, f(&gold["damping"])));
    }
    println!(
        "actuators and tendons: worst relative error {worst:.2e} (worst absolute difference {:.2e})",
        worst_abs()
    );
    assert!(worst <= RTOL, "actuators/tendons: {worst}");
}

// ---- the soft-constraint parameters (phase 1c-i) ---------------------------------

/// The values that are copied from the XML (`solref`, `solimp`, `margin`,
/// `frictionloss`) must equal MuJoCo's compiled values EXACTLY: the same decimal
/// parse, no arithmetic. The range of an angular joint goes through a degree to
/// radian conversion and is compared to 1e-12 relative. Returns the worst relative
/// range error.
fn check_soft_joints(scene: &Scene, gold: &[Value]) -> f64 {
    assert_eq!(scene.joints.len(), gold.len());
    let mut worst_range = 0.0f64;
    for (joint, g) in scene.joints.iter().zip(gold) {
        let what = &joint.name;
        assert_eq!(joint.name, g["name"].as_str().unwrap());
        // `jnt_limited` and `jnt_range`
        assert_eq!(
            joint.range.is_some(),
            g["limited"].as_bool().unwrap(),
            "{what} limited"
        );
        if let Some(r) = joint.range {
            let gr = farr::<2>(&g["range"]);
            for k in 0..2 {
                let diff = (r[k] - gr[k]).abs();
                let rel = if diff == 0.0 {
                    0.0
                } else {
                    diff / gr[k].abs().max(r[k].abs())
                };
                worst_range = worst_range.max(rel);
                assert!(rel <= 1e-12, "{what} range[{k}]: {} vs {}", r[k], gr[k]);
            }
        }
        // `jnt_solref`, `jnt_solimp`, `jnt_margin`, `dof_solref`, `dof_solimp`,
        // `dof_frictionloss`
        assert_eq!(
            joint.solref_limit,
            farr::<2>(&g["solref_limit"]),
            "{what} solref_limit"
        );
        assert_eq!(
            joint.solimp_limit,
            farr::<5>(&g["solimp_limit"]),
            "{what} solimp_limit"
        );
        assert_eq!(
            joint.solref_friction,
            farr::<2>(&g["solref_friction"]),
            "{what} solref_friction"
        );
        assert_eq!(
            joint.solimp_friction,
            farr::<5>(&g["solimp_friction"]),
            "{what} solimp_friction"
        );
        assert_eq!(joint.margin, f(&g["margin"]), "{what} margin");
        assert_eq!(
            joint.frictionloss,
            f(&g["frictionloss"]),
            "{what} frictionloss"
        );
    }
    worst_range
}

/// `tendon_limited`, `tendon_range`, `tendon_solref_lim`, `tendon_solimp_lim`,
/// `tendon_solref_fri`, `tendon_solimp_fri`, `tendon_margin`, `tendon_frictionloss`.
fn check_soft_tendons(scene: &Scene, gold: &[Value]) {
    assert_eq!(scene.tendons.len(), gold.len());
    for (t, g) in scene.tendons.iter().zip(gold) {
        let what = &t.name;
        assert_eq!(t.name, g["name"].as_str().unwrap());
        assert_eq!(
            t.range.is_some(),
            g["limited"].as_bool().unwrap(),
            "{what} limited"
        );
        if let Some(r) = t.range {
            assert_eq!(r, farr::<2>(&g["range"]), "{what} range (no conversion)");
        }
        assert_eq!(
            t.solref_limit,
            farr::<2>(&g["solref_limit"]),
            "{what} solref_limit"
        );
        assert_eq!(
            t.solimp_limit,
            farr::<5>(&g["solimp_limit"]),
            "{what} solimp_limit"
        );
        assert_eq!(
            t.solref_friction,
            farr::<2>(&g["solref_friction"]),
            "{what} solref_friction"
        );
        assert_eq!(
            t.solimp_friction,
            farr::<5>(&g["solimp_friction"]),
            "{what} solimp_friction"
        );
        assert_eq!(t.margin, f(&g["margin"]), "{what} margin");
        assert_eq!(t.frictionloss, f(&g["frictionloss"]), "{what} frictionloss");
        assert_eq!(t.stiffness, f(&g["stiffness"]), "{what} stiffness");
        assert_eq!(t.damping, f(&g["damping"]), "{what} damping");
        assert_eq!(t.armature, f(&g["armature"]), "{what} armature");
    }
}

/// The solver options: MuJoCo's `opt.solver` (PGS 0, CG 1, Newton 2),
/// `opt.iterations`, `opt.tolerance`, `opt.ls_iterations`, `opt.ls_tolerance`,
/// `opt.cone` (pyramidal 0, elliptic 1) and `opt.impratio`, exactly.
fn check_options(scene: &Scene, g: &Value) {
    let o = &scene.options;
    let solver = match o.solver {
        sim_scene::Solver::Pgs => 0,
        sim_scene::Solver::Cg => 1,
        sim_scene::Solver::Newton => 2,
    };
    assert_eq!(solver, g["solver"].as_i64().unwrap(), "solver");
    assert_eq!(i64::from(o.iterations), g["iterations"].as_i64().unwrap());
    assert_eq!(o.tolerance, f(&g["tolerance"]));
    assert_eq!(
        i64::from(o.ls_iterations),
        g["ls_iterations"].as_i64().unwrap()
    );
    assert_eq!(o.ls_tolerance, f(&g["ls_tolerance"]));
    let cone = match o.cone {
        sim_scene::Cone::Pyramidal => 0,
        sim_scene::Cone::Elliptic => 1,
    };
    assert_eq!(cone, g["cone"].as_i64().unwrap(), "cone");
    assert_eq!(o.impratio, f(&g["impratio"]));
    assert_eq!(scene.timestep_s, f(&g["timestep"]));
}

#[test]
fn the_humanoids_soft_constraint_parameters_and_options_match_mujoco() {
    let g = golden();
    let scene = load();
    let worst = check_soft_joints(&scene, g["joints"].as_array().unwrap());
    check_soft_tendons(&scene, g["tendons"].as_array().unwrap());
    check_options(&scene, &g["option"]);
    // the humanoid's limits carry `solimplimit="0 .99 .01"` (MuJoCo clamps the 0 at run
    // time and keeps it in the model) and nothing else is non-default
    assert!(
        scene
            .joints
            .iter()
            .filter(|j| j.range.is_some())
            .all(|j| j.solimp_limit == [0.0, 0.99, 0.01, 0.5, 2.0])
    );
    println!("humanoid: worst relative error of a joint range {worst:.2e} (gate 1e-12)");
}

#[test]
fn the_constrained_models_non_default_parameters_match_mujoco() {
    // `constrained.xml` (ours): non-default solref, solimp and margin on joints and a
    // tendon, a direct (negative) solref, a ball joint's range in degrees, friction
    // loss on a ball joint, class defaults and non-default solver options
    let g = golden();
    let case = &g["constrained_case"];
    let bytes = fs::read(dir().join("constrained.xml")).unwrap();
    assert_eq!(case["xml"]["bytes"].as_u64().unwrap(), bytes.len() as u64);
    assert_eq!(
        case["xml"]["fnv1a64"].as_str().unwrap(),
        format!("{:016x}", fnv1a64(&bytes)),
        "constrained.xml changed without regenerating the golden file"
    );
    let scene = mjcf::load(&String::from_utf8(bytes).unwrap(), dir()).expect("imports");
    assert_eq!(
        scene.joints.len() as u64,
        case["counts"]["njnt"].as_u64().unwrap()
    );
    let worst = check_soft_joints(&scene, case["joints"].as_array().unwrap());
    check_soft_tendons(&scene, case["tendons"].as_array().unwrap());
    check_options(&scene, &case["option"]);
    // the values really are not the defaults (so equal-to-MuJoCo is not vacuous)
    assert_eq!(scene.options.solver, sim_scene::Solver::Cg);
    assert_eq!(scene.options.iterations, 80);
    assert_eq!(scene.options.cone, sim_scene::Cone::Elliptic);
    assert_eq!(scene.joints[2].solref_limit, [-400.0, -15.0]);
    assert_eq!(scene.joints[3].margin, 0.08);
    assert_eq!(scene.joints[3].frictionloss, 0.15);
    assert_eq!(scene.tendons[0].margin, 0.05);
    assert_eq!(scene.tendons[0].frictionloss, 0.25);
    // a ball joint's range is [0, max_angle] in radians
    let ball = scene.joints[3].range.unwrap();
    assert_eq!(ball[0], 0.0);
    assert!((ball[1] - 50.0f64.to_radians()).abs() < 1e-15);
    // nothing of the model is left unsupported
    assert!(scene.unsupported.is_empty(), "{:?}", scene.unsupported);
    println!("constrained: worst relative error of a joint range {worst:.2e} (gate 1e-12)");
    // and it imports under strict
    mjcf::load_with(
        &String::from_utf8(fs::read(dir().join("constrained.xml")).unwrap()).unwrap(),
        dir(),
        &mjcf::LoadOptions {
            strict: true,
            instances: true,
        },
    )
    .expect("a fully modelled scene imports under strict");
}

#[test]
fn the_import_records_what_it_does_not_model() {
    // humanoid.xml has visual/statistic/texture/material/light/camera/exclude/
    // keyframe sections and geom solver attributes the scene does not model; every
    // one of them must be listed, none dropped silently. Its two fixed tendons (a
    // limit, no spring) and the joint limit parameters (`solimplimit`) are modelled
    // and not listed.
    let scene = load();
    let has = |path: &str, item: &str| {
        scene
            .unsupported
            .iter()
            .any(|u| u.path == path && u.item == item)
    };
    // <visual>: the clip planes and the offscreen size are read (they place and size
    // the cameras), every other setting is recorded
    assert!(!has("visual", "element"));
    assert!(has("visual/map", "@force"));
    assert!(!has("visual/map", "@zfar"));
    assert!(has("visual/rgba", "element"));
    assert!(has("visual/global", "@elevation"));
    assert!(has("visual/global", "@azimuth"));
    assert!(!has("visual/global", "@offwidth"));
    assert!(!has("visual/global", "@offheight"));
    // <statistic center> is read: it overrides the computed model centre
    assert!(!has("statistic", "element"));
    assert!(!has("statistic", "@center"));
    assert!(has("asset/texture", "element"));
    assert!(has("keyframe", "element"));
    assert!(has("worldbody/light[spotlight]", "element"));
    // the two tracking cameras are recorded; the fixed egocentric camera is imported
    assert!(has("worldbody/body[torso]/camera[back]", "element"));
    assert!(has("worldbody/body[torso]/camera[side]", "element"));
    assert!(!has(
        "worldbody/body[torso]/body[head]/camera[egocentric]",
        "element"
    ));
    assert_eq!(
        scene
            .cameras
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>(),
        ["egocentric"]
    );
    assert!(has("default/default[body]/geom", "@group"));
    // modelled since phase 1c-ii (the geoms' contact parameters and the exclusions),
    // so not listed any more
    assert!(!has("default/default[body]/geom", "@solimp"));
    assert!(!has("default/default[body]/geom", "@solref"));
    assert!(!has("contact/exclude", "element"));
    // modelled since phase 1c-i, so not listed any more
    assert!(!has("default/default[body]/joint", "@solimplimit"));
    assert!(!has("tendon/fixed[hamstring_right]", "element"));
    assert!(!has("tendon/fixed[hamstring_left]", "element"));
    assert!(
        scene
            .unsupported
            .iter()
            .all(|u| !u.path.starts_with("tendon"))
    );
    // the humanoid's unsupported list, entry by entry, so that a change in what is
    // recorded (or a new silent drop) is a visible change of this test
    let counts = |item: &str| scene.unsupported.iter().filter(|u| u.item == item).count();
    println!(
        "humanoid unsupported: {} entries ({} elements, {} @group)",
        scene.unsupported.len(),
        counts("element"),
        counts("@group")
    );
    assert_eq!(counts("@solimp") + counts("@solref"), 0);
    // each entry says where, what and why
    for u in &scene.unsupported {
        assert!(
            !u.path.is_empty() && !u.item.is_empty() && !u.reason.is_empty(),
            "{u:?}"
        );
        assert!(u.line > 0, "{u:?}");
    }
}

#[test]
fn strict_import_refuses_what_the_default_import_records() {
    let err = mjcf::load_with(
        &xml(),
        dir(),
        &mjcf::LoadOptions {
            strict: true,
            instances: true,
        },
    )
    .unwrap_err();
    match err {
        sim_scene::SceneError::Mjcf(e) => {
            assert_eq!(e.kind, sim_scene::MjcfErrorKind::Strict);
            assert!(e.line > 0);
        }
        other => panic!("expected an MJCF strict refusal, got {other:?}"),
    }
}

#[test]
fn the_imported_scene_round_trips_through_json() {
    let scene = load();
    let json = scene.to_json().unwrap();
    let back = Scene::from_json(&json).unwrap();
    assert_eq!(scene, back);
}

#[test]
fn import_is_deterministic() {
    assert_eq!(load(), load());
}

#[test]
fn a_scaled_offset_rotated_stl_mesh_matches_mujocos_compile() {
    // `mesh_case.xml` (fixtures/mujoco, written for this test, with `pyramid.stl`
    // written by the generator): a binary STL of an irregular pyramid, scaled by
    // (1.5, 1, 2), placed with an offset and an Euler rotation, at density 700, summed
    // with a sphere of fixed mass. It covers what humanoid.xml cannot: STL decoding,
    // MuJoCo's mesh volume, centre of mass and inertia, the mesh frame folded into
    // the body's inertia, and the parallel-axis sum of a mesh with a primitive.
    let g = golden();
    let case = &g["mesh_case"];
    let bytes = fs::read(dir().join("mesh_case.xml")).unwrap();
    assert_eq!(case["xml"]["bytes"].as_u64().unwrap(), bytes.len() as u64);
    assert_eq!(
        case["xml"]["fnv1a64"].as_str().unwrap(),
        format!("{:016x}", fnv1a64(&bytes)),
        "mesh_case.xml changed without regenerating the golden file"
    );
    let stl = fs::read(dir().join("pyramid.stl")).unwrap();
    assert_eq!(case["stl"]["bytes"].as_u64().unwrap(), stl.len() as u64);
    assert_eq!(
        case["stl"]["fnv1a64"].as_str().unwrap(),
        format!("{:016x}", fnv1a64(&stl)),
        "pyramid.stl differs from the one the golden file was generated with"
    );

    let scene =
        mjcf::load(&String::from_utf8(bytes).unwrap(), dir()).expect("mesh_case.xml imports");
    assert_eq!(scene.meshes.len(), 1);
    assert_eq!(
        scene.meshes[0].vertices.len() as u64,
        case["mesh_vertices"].as_u64().unwrap()
    );
    assert_eq!(
        scene.meshes[0].triangles.len() as u64,
        case["mesh_faces"].as_u64().unwrap()
    );
    let body = scene.bodies.iter().find(|b| b.name == "piece").unwrap();
    let inertial = body.inertial.unwrap();
    let gold = &case["body"];
    let worst = [
        rel(inertial.mass_kg, f(&gold["mass"])),
        max_rel(&inertial.com, &farr::<3>(&gold["com"])),
        max_rel(&inertial.diag_inertia, &farr::<3>(&gold["inertia"])),
        quat_rel(inertial.inertia_quat, farr::<4>(&gold["inertia_quat"])),
    ];
    println!(
        "mesh case: worst relative error  mass {:.2e}  com {:.2e}  inertia {:.2e}  inertia_quat {:.2e}  (worst absolute difference {:.2e})",
        worst[0],
        worst[1],
        worst[2],
        worst[3],
        worst_abs()
    );
    assert!(worst[0] <= RTOL, "mass");
    assert!(worst[1] <= RTOL, "centre of mass");
    assert!(worst[2] <= INERTIA_RTOL, "inertia");
    assert!(worst[3] <= INERTIA_RTOL, "inertia quaternion");
}

#[test]
fn negative_control_a_changed_model_fails_the_parity_comparison() {
    // The comparison is not vacuous: change the head's radius by 1.1% and the head's
    // mass, which the golden file pins to 1e-9, differs by 3.3%, and so does the
    // mesh case when the density changes.
    let g = golden();
    let original = xml();
    let changed = original.replace(
        r#"<geom name="head" type="sphere" size=".09"/>"#,
        r#"<geom name="head" type="sphere" size=".091"/>"#,
    );
    assert_ne!(changed, original, "the replacement did not apply");
    let scene = mjcf::load(&changed, dir()).unwrap();
    let head = scene.bodies.iter().find(|b| b.name == "head").unwrap();
    let gold_head = g["bodies"]
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["name"] == "head")
        .unwrap();
    let error = rel(head.inertial.unwrap().mass_kg, f(&gold_head["mass"]));
    assert!(
        error > 1e-2,
        "a 1.1% radius change moved the mass by only {error}"
    );

    let mesh_xml = fs::read_to_string(dir().join("mesh_case.xml")).unwrap();
    let denser = mesh_xml.replace(r#"density="700""#, r#"density="701""#);
    assert_ne!(denser, mesh_xml);
    let scene = mjcf::load(&denser, dir()).unwrap();
    let mass = scene.bodies[0].inertial.unwrap().mass_kg;
    let gold_mass = f(&g["mesh_case"]["body"]["mass"]);
    assert!(
        rel(mass, gold_mass) > 1e-4,
        "a changed density did not change the mass"
    );
}

/// The generator's own `--check`: regenerates the golden file with MuJoCo and
/// requires it to equal the committed one. Needs the MuJoCo oracle interpreter
/// (the path in `SIM_SCENE_ORACLE_PYTHON`).
/// Without it the test says SKIPPED on stderr and passes; set
/// `SIM_SCENE_REQUIRE_ORACLE=1` to make a missing interpreter a failure. The gate
/// also runs the same command directly.
#[test]
fn the_generator_check_passes() {
    let python = std::env::var("SIM_SCENE_ORACLE_PYTHON").unwrap_or_default();
    if !std::path::Path::new(&python).exists() {
        assert!(
            std::env::var("SIM_SCENE_REQUIRE_ORACLE").is_err(),
            "the MuJoCo oracle interpreter {python} is required and missing"
        );
        eprintln!(
            "SKIPPED the_generator_check_passes: no MuJoCo oracle interpreter (set SIM_SCENE_ORACLE_PYTHON; now {python:?})"
        );
        return;
    }
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../..");
    let script = repo.join("tools/sim_scene_mujoco_golden.py");
    if !script.exists() {
        // A copy of this crate outside the source repository has no generator.
        assert!(
            std::env::var("SIM_SCENE_REQUIRE_ORACLE").is_err(),
            "the golden generator {} is required and missing",
            script.display()
        );
        eprintln!(
            "SKIPPED the_generator_check_passes: no golden generator at {}",
            script.display()
        );
        return;
    }
    let output = std::process::Command::new(&python)
        .arg(&script)
        .arg("--check")
        .output()
        .expect("the oracle interpreter starts");
    assert!(
        output.status.success(),
        "tools/sim_scene_mujoco_golden.py --check failed ({}):\n{}{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    println!("{}", String::from_utf8_lossy(&output.stdout).trim());
}
