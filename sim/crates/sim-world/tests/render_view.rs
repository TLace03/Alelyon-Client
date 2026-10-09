//! The render view, byte for byte: hand-computed bytes for a two-body, two-env
//! scene, the layout invariants, and the cameras view.
//!
//! Every expected word below is an IEEE-754 binary32 bit pattern written by hand
//! (not produced by the code under test): 1.0 = 0x3F800000, 2.0 = 0x40000000,
//! 3.0 = 0x40400000, 1.5 = 0x3FC00000, 0.5 = 0x3F000000, 0.25 = 0x3E800000,
//! 4.0 = 0x40800000, 100.0 = 0x42C80000, 63.5 = 0x427E0000, 47.5 = 0x423E0000,
//! 0.1 = 0x3DCCCCCD, 50.0 = 0x42480000, and a sign flip sets bit 31 (-1.0 =
//! 0xBF800000, -2.0 = 0xC0000000, -3.0 = 0xC0400000, -0.5 = 0xBF000000, -0.75 = 0xBF400000).

mod common;

use common::{two_body_scene, words_to_bytes};
use sim_world::{
    CAMERA_RECORD_BYTES, HostWorld, POSE_BYTES, ResetNoise, cameras_view, gather_render_view,
};

/// Environment 0 from the scene's reference pose (zero noise, forward kinematics);
/// environment 1 written directly, so the two differ in every component.
fn world() -> HostWorld {
    let scene = two_body_scene();
    let mut w = HostWorld::new(&scene, 2).unwrap();
    w.reset_with(0, &ResetNoise::NONE);
    // env 1: base at (-1, -2, -3) turned 180 degrees about x; arm at
    // (0.25, 0.5, -0.5) turned 180 degrees about y
    w.body_pos_mut(1)
        .copy_from_slice(&[-1.0, -2.0, -3.0, 0.25, 0.5, -0.5]);
    w.body_quat_mut(1)
        .copy_from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    w
}

#[test]
fn the_render_view_is_the_agreed_layout_with_hand_computed_bytes() {
    let view = gather_render_view(&world());
    assert_eq!(view.poses.len(), 2 * 2 * POSE_BYTES);
    assert_eq!(POSE_BYTES, 32);

    #[rustfmt::skip]
    let expected = words_to_bytes(&[
        // env 0, body 0 (base): pos (1, 2, 3), pad 0, quat (0, 0, 0, 1)
        0x3F80_0000, 0x4000_0000, 0x4040_0000, 0x0000_0000,
        0x0000_0000, 0x0000_0000, 0x0000_0000, 0x3F80_0000,
        // env 0, body 1 (arm): pos (1.5, 2, 3), pad 0, quat (0, 0, 1, 0)
        0x3FC0_0000, 0x4000_0000, 0x4040_0000, 0x0000_0000,
        0x0000_0000, 0x0000_0000, 0x3F80_0000, 0x0000_0000,
        // env 1, body 0: pos (-1, -2, -3), pad 0, quat (1, 0, 0, 0)
        0xBF80_0000, 0xC000_0000, 0xC040_0000, 0x0000_0000,
        0x3F80_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000,
        // env 1, body 1: pos (0.25, 0.5, -0.5), pad 0, quat (0, 1, 0, 0)
        0x3E80_0000, 0x3F00_0000, 0xBF00_0000, 0x0000_0000,
        0x0000_0000, 0x3F80_0000, 0x0000_0000, 0x0000_0000,
    ]);
    assert_eq!(view.poses, expected);
}

#[test]
fn the_view_is_packed_env_major_then_body() {
    let view = gather_render_view(&world());
    // pose index = env * n_bodies + body
    for (env, body, first_word) in [
        (0usize, 0usize, 0x3F80_0000u32), // x = 1.0
        (0, 1, 0x3FC0_0000),              // x = 1.5
        (1, 0, 0xBF80_0000),              // x = -1.0
        (1, 1, 0x3E80_0000),              // x = 0.25
    ] {
        let bytes = view.pose_bytes(env * 2 + body).unwrap();
        assert_eq!(bytes.len(), 32);
        assert_eq!(
            u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            first_word
        );
        // the pad word is zero
        assert_eq!(&bytes[12..16], &[0, 0, 0, 0]);
    }
    assert!(view.pose_bytes(4).is_none());
}

#[test]
fn the_view_is_a_bitwise_copy_of_the_state() {
    let scene = two_body_scene();
    let mut w = HostWorld::new(&scene, 5).unwrap();
    w.reset(99);
    let view = gather_render_view(&w);
    for env in 0..5u32 {
        for body in 0..2usize {
            let bytes = view.pose_bytes(env as usize * 2 + body).unwrap();
            let (p, q) = w.body_pose(env, body);
            for k in 0..3 {
                assert_eq!(&bytes[4 * k..4 * k + 4], &p[k].to_le_bytes());
            }
            for k in 0..4 {
                assert_eq!(&bytes[16 + 4 * k..20 + 4 * k], &q[k].to_le_bytes());
            }
        }
    }
    // gathering twice gives the same bytes
    assert_eq!(view, gather_render_view(&w));
}

#[test]
fn the_view_size_is_envs_times_bodies_times_32() {
    let scene = two_body_scene();
    for n in [1u32, 3, 17] {
        let w = HostWorld::new(&scene, n).unwrap();
        assert_eq!(gather_render_view(&w).poses.len(), n as usize * 2 * 32);
    }
}

#[test]
fn the_cameras_view_has_hand_computed_bytes() {
    let scene = two_body_scene();
    let w = world();
    let view = cameras_view(&scene, &w).unwrap();
    assert_eq!(CAMERA_RECORD_BYTES, 64);
    assert_eq!(view.records.len(), 2 * 2 * 64);

    // the intrinsics and the two pad words, the same for every camera of the scene
    let intrinsics = [
        0x42C8_0000u32, // fx = 100.0
        0x42C8_0000,    // fy = 100.0
        0x427E_0000,    // cx = 63.5  ((128 - 1) / 2)
        0x423E_0000,    // cy = 47.5  ((96 - 1) / 2)
        0x3DCC_CCCD,    // near = 0.1
        0x4248_0000,    // far = 50.0
        0x0000_0000,    // pad
        0x0000_0000,    // pad
    ];
    let mut words: Vec<u32> = Vec::new();
    // env 0, camera "overhead" (world mount): pos (0, 0, 2), quat (0.5, 0.5, 0.5, 0.5)
    words.extend([
        0x0000_0000,
        0x0000_0000,
        0x4000_0000,
        0x0000_0000,
        0x3F00_0000,
        0x3F00_0000,
        0x3F00_0000,
        0x3F00_0000,
    ]);
    words.extend(intrinsics);
    // env 0, camera "wrist" (on the arm, local (1, 0, 0), local quat (.5, .5, .5, .5)):
    // arm at (1.5, 2, 3) with quat (0, 0, 1, 0) rotates (1, 0, 0) to (-1, 0, 0), so the
    // camera is at (0.5, 2, 3); (0, 0, 1, 0) * (.5, .5, .5, .5) = (-.5, .5, .5, -.5)
    words.extend([
        0x3F00_0000,
        0x4000_0000,
        0x4040_0000,
        0x0000_0000,
        0xBF00_0000,
        0x3F00_0000,
        0x3F00_0000,
        0xBF00_0000,
    ]);
    words.extend(intrinsics);
    // env 1, "overhead": unchanged (world mount)
    words.extend([
        0x0000_0000,
        0x0000_0000,
        0x4000_0000,
        0x0000_0000,
        0x3F00_0000,
        0x3F00_0000,
        0x3F00_0000,
        0x3F00_0000,
    ]);
    words.extend(intrinsics);
    // env 1, "wrist": arm at (0.25, 0.5, -0.5) with quat (0, 1, 0, 0) (180 degrees about
    // y) turns (1, 0, 0) into (-1, 0, 0), so the camera is at (-0.75, 0.5, -0.5);
    // (0, 1, 0, 0) * (.5, .5, .5, .5) = (.5, .5, -.5, -.5). -0.75 = 0xBF400000.
    words.extend([
        0xBF40_0000,
        0x3F00_0000,
        0xBF00_0000,
        0x0000_0000,
        0x3F00_0000,
        0x3F00_0000,
        0xBF00_0000,
        0xBF00_0000,
    ]);
    words.extend(intrinsics);
    assert_eq!(view.records, words_to_bytes(&words));
}

#[test]
fn a_body_mounted_camera_follows_its_body() {
    let scene = two_body_scene();
    let mut w = HostWorld::new(&scene, 1).unwrap();
    w.reset_with(0, &ResetNoise::NONE);
    let before = cameras_view(&scene, &w).unwrap();
    // move the arm 1 m along +z and look again: only the wrist camera moves
    w.body_pos_mut(0)[5] += 1.0;
    let after = cameras_view(&scene, &w).unwrap();
    assert_eq!(
        &before.records[..64],
        &after.records[..64],
        "the world camera moved"
    );
    assert_ne!(
        &before.records[64..],
        &after.records[64..],
        "the wrist camera did not follow"
    );
    // z of the wrist camera: 3 + 1
    let z = f32::from_le_bytes(after.records[64 + 8..64 + 12].try_into().unwrap());
    assert_eq!(z, 4.0);
}

#[test]
fn the_cameras_view_refuses_a_world_that_is_not_the_scenes() {
    let scene = two_body_scene();
    let mut other = sim_scene::Scene::new();
    other.bodies.push(sim_scene::Body {
        name: "lonely".into(),
        parent: None,
        pos: [0.0; 3],
        quat: [0.0, 0.0, 0.0, 1.0],
        inertial: None,
    });
    let w = HostWorld::new(&other, 1).unwrap();
    assert!(matches!(
        cameras_view(&scene, &w),
        Err(sim_world::WorldError::SceneMismatch { .. })
    ));
}

#[test]
fn a_scene_without_cameras_has_an_empty_cameras_view() {
    let mut scene = two_body_scene();
    scene.cameras.clear();
    let w = HostWorld::new(&scene, 3).unwrap();
    assert!(cameras_view(&scene, &w).unwrap().records.is_empty());
}
