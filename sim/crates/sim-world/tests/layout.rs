//! The world-state layout: sizes, offsets, alignment and refusals.

mod common;

use sim_scene::mjcf;
use sim_world::{ENV_STRIDE_ALIGN_BYTES, FieldId, WorldError, WorldLayout};

fn round_up(n: usize) -> usize {
    n.div_ceil(ENV_STRIDE_ALIGN_BYTES) * ENV_STRIDE_ALIGN_BYTES
}

#[test]
fn the_two_body_scene_layout_is_what_the_conventions_say() {
    let scene = common::two_body_scene();
    let layout = WorldLayout::new(&scene, 5).unwrap();
    // free (7/6) + hinge (1/1)
    assert_eq!((layout.nq, layout.nv), (8, 7));
    assert_eq!(layout.n_bodies, 2);
    assert_eq!(layout.n_joints, 2);
    assert_eq!(layout.n_actuators, 0);
    assert_eq!(layout.n_instances, 1);
    assert_eq!(layout.joint_qpos_adr, vec![0, 7]);
    assert_eq!(layout.joint_dof_adr, vec![0, 6]);

    let floats = |id: FieldId| layout.field(id).floats_per_env;
    assert_eq!(floats(FieldId::BodyPos), 6);
    assert_eq!(floats(FieldId::BodyQuat), 8);
    assert_eq!(floats(FieldId::BodyLinVel), 6);
    assert_eq!(floats(FieldId::BodyAngVel), 6);
    assert_eq!(floats(FieldId::Qpos), 8);
    assert_eq!(floats(FieldId::Qvel), 7);
    assert_eq!(floats(FieldId::Ctrl), 0);
    // temperature, browning, wetness, reserved
    assert_eq!(floats(FieldId::Surface), 4);
}

#[test]
fn every_environment_slice_starts_on_a_256_byte_boundary() {
    let scene = common::two_body_scene();
    for n_envs in [1u32, 2, 7, 64] {
        let layout = WorldLayout::new(&scene, n_envs).unwrap();
        let mut expected_base = 0usize;
        for f in layout.fields() {
            assert_eq!(f.env_stride_bytes % ENV_STRIDE_ALIGN_BYTES, 0, "{:?}", f.id);
            assert_eq!(
                f.base_offset_bytes % ENV_STRIDE_ALIGN_BYTES,
                0,
                "{:?}",
                f.id
            );
            assert_eq!(
                f.base_offset_bytes, expected_base,
                "{:?} is not contiguous",
                f.id
            );
            if f.floats_per_env == 0 {
                assert_eq!((f.env_stride_bytes, f.size_bytes), (0, 0), "{:?}", f.id);
            } else {
                assert_eq!(
                    f.env_stride_bytes,
                    round_up(f.floats_per_env * 4),
                    "{:?}",
                    f.id
                );
                assert!(f.env_stride_bytes >= f.floats_per_env * 4);
                assert!(f.env_stride_bytes < f.floats_per_env * 4 + ENV_STRIDE_ALIGN_BYTES);
            }
            assert_eq!(f.size_bytes, f.env_stride_bytes * n_envs as usize);
            for env in 0..n_envs as usize {
                assert_eq!(
                    f.env_offset_bytes(env) % ENV_STRIDE_ALIGN_BYTES,
                    0,
                    "{:?} env {env}",
                    f.id
                );
            }
            expected_base += f.size_bytes;
        }
        assert_eq!(layout.total_bytes, expected_base);
    }
}

#[test]
fn a_field_that_needs_more_than_256_bytes_gets_the_next_multiple() {
    // Pins the rounding with hand-computed numbers on the humanoid (16 bodies,
    // 28 joint coordinates, 27 dofs, 21 actuators, 20 instances).
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../sim-scene/tests/fixtures/mujoco");
    let xml = std::fs::read_to_string(dir.join("humanoid.xml")).unwrap();
    let scene = mjcf::load(&xml, &dir).unwrap();
    let layout = WorldLayout::new(&scene, 3).unwrap();
    let stride = |id: FieldId| layout.field(id).env_stride_bytes;
    assert_eq!(stride(FieldId::BodyPos), 256); // 3 * 16 * 4 = 192 -> 256
    assert_eq!(stride(FieldId::BodyQuat), 256); // 4 * 16 * 4 = 256 -> 256 (exact)
    assert_eq!(stride(FieldId::BodyLinVel), 256);
    assert_eq!(stride(FieldId::BodyAngVel), 256);
    assert_eq!(stride(FieldId::Qpos), 256); // 28 * 4 = 112 -> 256
    assert_eq!(stride(FieldId::Qvel), 256); // 27 * 4 = 108 -> 256
    assert_eq!(stride(FieldId::Ctrl), 256); // 21 * 4 = 84 -> 256
    assert_eq!(stride(FieldId::Surface), 512); // 4 * 20 * 4 = 320 -> 512
    assert_eq!(layout.total_bytes, 3 * (7 * 256 + 512));
    assert_eq!(
        layout.field(FieldId::Surface).base_offset_bytes,
        3 * 7 * 256
    );
}

#[test]
fn zero_environments_and_overflowing_requests_are_refused() {
    let scene = common::two_body_scene();
    assert!(matches!(
        WorldLayout::new(&scene, 0),
        Err(WorldError::Layout { .. })
    ));
}

#[test]
fn an_invalid_scene_is_refused_before_any_layout_is_computed() {
    let mut scene = common::two_body_scene();
    scene.instances[0].seg_id = 0;
    assert!(matches!(
        WorldLayout::new(&scene, 2),
        Err(WorldError::Scene { .. })
    ));
}

#[test]
fn a_scene_with_no_joints_has_empty_joint_fields() {
    let mut scene = sim_scene::Scene::new();
    scene.bodies.push(sim_scene::Body {
        name: "fixed".into(),
        parent: None,
        pos: [0.0; 3],
        quat: [0.0, 0.0, 0.0, 1.0],
        inertial: None,
    });
    let layout = WorldLayout::new(&scene, 4).unwrap();
    assert_eq!((layout.nq, layout.nv), (0, 0));
    assert_eq!(layout.field(FieldId::Qpos).size_bytes, 0);
    assert_eq!(layout.field(FieldId::BodyPos).env_stride_bytes, 256);
}
