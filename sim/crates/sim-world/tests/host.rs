//! The host mirror: determinism of reset, forward kinematics against hand
//! computation and against MuJoCo, and the layout of the buffers.

mod common;

use std::f64::consts::FRAC_PI_2;
use std::path::PathBuf;

use serde_json::Value;
use sim_scene::{Scene, mjcf};
use sim_world::{FieldId, HostWorld, ResetNoise};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../sim-scene/tests/fixtures/mujoco")
}

fn humanoid() -> Scene {
    let xml = std::fs::read_to_string(fixtures().join("humanoid.xml")).unwrap();
    mjcf::load(&xml, fixtures()).unwrap()
}

#[test]
fn two_worlds_reset_with_the_same_seed_are_bitwise_equal() {
    let scene = common::two_body_scene();
    let mut a = HostWorld::new(&scene, 8).unwrap();
    let mut b = HostWorld::new(&scene, 8).unwrap();
    a.reset(12345);
    b.reset(12345);
    assert!(a.bit_eq(&b));
    assert_eq!(a.arena_bytes(), b.arena_bytes());
    // and resetting again, in the same world, restores the same bits
    let before = a.arena_bytes();
    a.reset(999);
    assert_ne!(a.arena_bytes(), before);
    a.reset(12345);
    assert_eq!(a.arena_bytes(), before);
}

#[test]
fn a_different_seed_gives_a_different_world() {
    let scene = common::two_body_scene();
    let mut a = HostWorld::new(&scene, 8).unwrap();
    let mut b = HostWorld::new(&scene, 8).unwrap();
    a.reset(1);
    b.reset(2);
    assert!(!a.bit_eq(&b));
    // seeds that differ only in a high bit also differ
    b.reset(1 | (1 << 40));
    assert!(!a.bit_eq(&b));
}

#[test]
fn different_environments_of_one_world_start_differently() {
    let scene = common::two_body_scene();
    let mut w = HostWorld::new(&scene, 6).unwrap();
    w.reset(77);
    for e in 1..6 {
        assert_ne!(w.qpos(0), w.qpos(e), "env {e} equals env 0");
    }
}

#[test]
fn an_environments_reset_does_not_depend_on_how_many_there_are() {
    let scene = common::two_body_scene();
    let mut small = HostWorld::new(&scene, 2).unwrap();
    let mut large = HostWorld::new(&scene, 50).unwrap();
    small.reset(2026);
    large.reset(2026);
    for env in 0..2 {
        for id in FieldId::ALL {
            assert_eq!(
                small
                    .env_slice(id, env)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                large
                    .env_slice(id, env)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "{id:?} env {env}"
            );
        }
    }
}

#[test]
fn zero_noise_restores_the_exact_reference_pose() {
    let scene = common::two_body_scene();
    let mut w = HostWorld::new(&scene, 3).unwrap();
    w.reset_with(5, &ResetNoise::NONE);
    for env in 0..3 {
        assert_eq!(w.body_pos(env), &[1.0, 2.0, 3.0, 1.5, 2.0, 3.0]);
        assert_eq!(w.body_quat(env), &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
        // free joint: the base's own pose; hinge: 0
        assert_eq!(w.qpos(env), &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert!(w.qvel(env).iter().all(|v| *v == 0.0));
        assert!(w.body_linvel(env).iter().all(|v| *v == 0.0));
        assert!(w.body_angvel(env).iter().all(|v| *v == 0.0));
    }
}

#[test]
fn a_new_world_is_valid_before_any_reset() {
    let scene = common::two_body_scene();
    let w = HostWorld::new(&scene, 2).unwrap();
    for env in 0..2 {
        let q = w.body_quat(env);
        for b in 0..2 {
            let n: f32 = q[4 * b..4 * b + 4].iter().map(|c| c * c).sum();
            assert!((n - 1.0).abs() < 1e-6, "body {b} quaternion is not unit");
        }
    }
}

#[test]
fn surface_state_ctrl_and_velocities_are_all_zero_after_reset() {
    let scene = common::two_body_scene();
    let mut w = HostWorld::new(&scene, 4).unwrap();
    w.reset(31337);
    for env in 0..4 {
        assert_eq!(w.surface(env), &[0.0, 0.0, 0.0, 0.0]);
        assert!(w.qvel(env).iter().all(|v| *v == 0.0));
    }
    // padding is zero too: every byte after the data of each env is 0
    for id in FieldId::ALL {
        let f = *w.layout().field(id);
        let bytes = w.field_bytes(id);
        assert_eq!(bytes.len(), f.size_bytes);
        for env in 0..4 {
            let start = env * f.env_stride_bytes + f.floats_per_env * 4;
            let end = (env + 1) * f.env_stride_bytes;
            assert!(
                bytes[start..end].iter().all(|b| *b == 0),
                "{id:?} padding env {env}"
            );
        }
    }
}

#[test]
fn reset_noise_stays_within_its_amplitude_and_the_joint_ranges() {
    let scene = common::two_body_scene();
    let noise = ResetNoise::default();
    let mut w = HostWorld::new(&scene, 200).unwrap();
    w.reset(8);
    for env in 0..200 {
        let q = w.qpos(env);
        // free position within 1 cm of the reference, hinge within 0.01 rad
        assert!((q[0] - 1.0).abs() <= noise.free_pos_m as f32 + 1e-6);
        assert!((q[1] - 2.0).abs() <= noise.free_pos_m as f32 + 1e-6);
        assert!((q[2] - 3.0).abs() <= noise.free_pos_m as f32 + 1e-6);
        assert!(q[7].abs() <= noise.hinge_rad as f32 + 1e-6);
        let n: f32 = q[3..7].iter().map(|c| c * c).sum();
        assert!((n - 1.0).abs() < 1e-5);
    }
    // and not all the same: the jitter is real
    let distinct: std::collections::HashSet<u32> =
        (0..200).map(|e| w.qpos(e)[7].to_bits()).collect();
    assert!(distinct.len() > 150);
}

#[test]
fn body_poses_always_agree_with_the_jittered_joint_coordinates() {
    // The hinge about z through the arm's origin turns the arm in place: its world
    // orientation is the base's turned 180 degrees about z, then by the hinge angle.
    let scene = common::two_body_scene();
    let mut w = HostWorld::new(&scene, 16).unwrap();
    w.reset(4242);
    for env in 0..16 {
        let q = w.qpos(env);
        let (bp, bq) = w.body_pose(env, 1);
        let (base_p, base_q) = w.body_pose(env, 0);
        // base pose = free-joint coordinates
        assert_eq!(base_p, [q[0], q[1], q[2]]);
        for k in 0..4 {
            assert!((base_q[k] - q[3 + k]).abs() < 1e-6);
        }
        // arm: pos = base + R(base) * (0.5, 0, 0); a z-hinge through the origin does
        // not move it
        let r = sim_scene::pose::quat_rotate(
            [
                f64::from(base_q[0]),
                f64::from(base_q[1]),
                f64::from(base_q[2]),
                f64::from(base_q[3]),
            ],
            [0.5, 0.0, 0.0],
        );
        for k in 0..3 {
            assert!((f64::from(bp[k]) - (f64::from(base_p[k]) + r[k])).abs() < 1e-6);
        }
        // orientation: |q_arm| = 1
        let n: f32 = bq.iter().map(|c| c * c).sum();
        assert!((n - 1.0).abs() < 1e-5);
    }
}

#[test]
fn hinge_about_an_offset_anchor_matches_the_hand_computation() {
    // The arm sits at world (1.5, 2, 3) turned 180 degrees about z. Put the hinge
    // anchor at body-frame (1, 0, 0): in the world that is (1.5 - 1, 2, 3) = (0.5, 2, 3).
    // A hinge angle of +90 degrees about the body z axis rotates the arm about that
    // anchor: the orientation becomes 180 + 90 = 270 degrees about z, (0, 0, sin 135, cos 135)
    // = (0, 0, 0.70710678, -0.70710678), and the origin goes to
    // anchor - R(270) * (1, 0, 0) = (0.5, 2, 3) - (0, -1, 0) = (0.5, 3, 3).
    let mut scene = common::two_body_scene();
    scene.joints[1].pos = [1.0, 0.0, 0.0];
    let mut w = HostWorld::new(&scene, 1).unwrap();
    w.set_qpos(0, &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, FRAC_PI_2])
        .unwrap();
    let (p, q) = w.body_pose(0, 1);
    let s = std::f32::consts::FRAC_1_SQRT_2;
    for (got, want) in p.iter().zip([0.5f32, 3.0, 3.0]) {
        assert!((got - want).abs() < 1e-6, "position {p:?}");
    }
    for (got, want) in q.iter().zip([0.0f32, 0.0, s, -s]) {
        assert!((got - want).abs() < 1e-6, "quaternion {q:?}");
    }
}

#[test]
fn slide_and_ball_joints_match_the_hand_computation() {
    // slide along the arm's own x axis, which points along world -x (arm turned 180
    // degrees about z): 0.3 m of travel moves the arm from x = 1.5 to 1.2
    let mut scene = common::two_body_scene();
    scene.joints[1].kind = sim_scene::JointKind::Slide {
        axis: [1.0, 0.0, 0.0],
    };
    scene.joints[1].range = None;
    let mut w = HostWorld::new(&scene, 1).unwrap();
    w.set_qpos(0, &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.3])
        .unwrap();
    let (p, q) = w.body_pose(0, 1);
    assert!((p[0] - 1.2).abs() < 1e-6 && (p[1] - 2.0).abs() < 1e-6 && (p[2] - 3.0).abs() < 1e-6);
    assert_eq!(q, [0.0, 0.0, 1.0, 0.0]);

    // a ball joint turned by the same rotation as the hinge gives the same pose
    let mut hinge_scene = common::two_body_scene();
    hinge_scene.joints[1].pos = [1.0, 0.0, 0.0];
    let mut ball_scene = hinge_scene.clone();
    ball_scene.joints[1].kind = sim_scene::JointKind::Ball;
    ball_scene.joints[1].range = None;
    let mut hw = HostWorld::new(&hinge_scene, 1).unwrap();
    let mut bw = HostWorld::new(&ball_scene, 1).unwrap();
    hw.set_qpos(0, &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, FRAC_PI_2])
        .unwrap();
    let h = FRAC_PI_2 / 2.0;
    bw.set_qpos(
        0,
        &[
            1.0,
            2.0,
            3.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            h.sin(),
            h.cos(),
        ],
    )
    .unwrap();
    let (hp, hq) = hw.body_pose(0, 1);
    let (bp, bq) = bw.body_pose(0, 1);
    for k in 0..3 {
        assert!((hp[k] - bp[k]).abs() < 1e-6, "position {k}");
    }
    for k in 0..4 {
        assert!((hq[k] - bq[k]).abs() < 1e-6, "quaternion {k}");
    }
}

#[test]
fn set_qpos_refuses_what_it_cannot_use() {
    let scene = common::two_body_scene();
    let mut w = HostWorld::new(&scene, 2).unwrap();
    assert!(w.set_qpos(2, &[0.0; 8]).is_err());
    assert!(w.set_qpos(0, &[0.0; 7]).is_err());
    let mut bad = [0.0; 8];
    bad[3] = f64::NAN;
    assert!(w.set_qpos(0, &bad).is_err());
}

#[test]
fn forward_kinematics_matches_mujoco_on_the_humanoid() {
    // MuJoCo 3.14.0's mj_kinematics for qpos0 and the four keyframes of
    // humanoid.xml (fixtures/mujoco/humanoid_golden.json, "fk"), compared with
    // this crate's forward kinematics. Positions to 5e-6 m (the world state is f32:
    // its resolution at 1.3 m is 1.2e-7 and the 22-joint chain accumulates a few),
    // orientations up to sign.
    let g: Value = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("humanoid_golden.json")).unwrap(),
    )
    .unwrap();
    let scene = humanoid();
    let mut w = HostWorld::new(&scene, 1).unwrap();
    let mut worst_pos = 0.0f64;
    let mut worst_quat = 0.0f64;
    for case in g["fk"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let qpos: Vec<f64> = case["qpos"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        w.set_qpos(0, &qpos).unwrap();
        let xpos = case["xpos"].as_array().unwrap();
        let xquat = case["xquat"].as_array().unwrap();
        assert_eq!(xpos.len(), scene.bodies.len() + 1);
        for (b, body) in scene.bodies.iter().enumerate() {
            let (p, q) = w.body_pose(0, b);
            let gp: Vec<f64> = xpos[b + 1]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let gq: Vec<f64> = xquat[b + 1]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            for k in 0..3 {
                worst_pos = worst_pos.max((f64::from(p[k]) - gp[k]).abs());
            }
            let d_plus: f64 = (0..4)
                .map(|k| (f64::from(q[k]) - gq[k]).abs())
                .fold(0.0, f64::max);
            let d_minus: f64 = (0..4)
                .map(|k| (f64::from(q[k]) + gq[k]).abs())
                .fold(0.0, f64::max);
            worst_quat = worst_quat.max(d_plus.min(d_minus));
            assert!(
                worst_pos < 5e-6 && worst_quat < 5e-6,
                "{name}: body {} ({}): position error {worst_pos:e}, orientation error {worst_quat:e}",
                b,
                body.name
            );
        }
    }
    println!(
        "humanoid FK vs MuJoCo: worst position error {worst_pos:.2e} m, worst orientation error {worst_quat:.2e}"
    );
}

#[test]
fn reset_on_the_humanoid_keeps_every_body_quaternion_unit_and_in_range() {
    let scene = humanoid();
    let mut w = HostWorld::new(&scene, 32).unwrap();
    w.reset(20_260_930);
    for env in 0..32 {
        let q = w.body_quat(env);
        for b in 0..scene.bodies.len() {
            let n: f32 = q[4 * b..4 * b + 4].iter().map(|c| c * c).sum();
            assert!((n - 1.0).abs() < 1e-5, "env {env} body {b}");
        }
        // joint limits respected
        let qpos = w.qpos(env);
        for (j, joint) in scene.joints.iter().enumerate() {
            if let Some([lo, hi]) = joint.range {
                let adr = w.layout().joint_qpos_adr[j] as usize;
                let v = f64::from(qpos[adr]);
                assert!(
                    v >= lo - 1e-6 && v <= hi + 1e-6,
                    "{} = {v} outside [{lo}, {hi}]",
                    joint.name
                );
            }
        }
    }
}
