//! Determinism: the same state steps to the same bits.
//!
//! - Two runs of 1,000 humanoid steps in `f32` from the same state end bitwise equal.
//! - `step_world` over three environments gives, for each environment, the same bits
//!   as that environment alone and as the environments in reverse order: an
//!   environment's result does not depend on how many others there are, or on their
//!   order.
//! - `step_world` agrees with the plain `step` on the same `Data` (so the load and
//!   store through the world, and the quaternion conversion, change nothing), and the
//!   poses it writes are the ones `HostWorld`'s own forward kinematics gives for the
//!   new `qpos`, so `gather_render_view` shows the stepped poses.

mod common;

use common::*;
use sim_physics::{Data, QuatOrder, convert_qpos, step, step_world};
use sim_world::{FieldId, HostWorld, gather_render_view};

fn humanoid32() -> Compiled<f32> {
    compile::<f32>(scene_of(Which::Humanoid))
}

fn bits(a: &[f32]) -> Vec<u32> {
    a.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn a_thousand_humanoid_steps_are_bitwise_reproducible_in_f32() {
    let c = humanoid32();
    let golden = golden_of(Which::Humanoid);
    let state = &golden["states"][3];
    let run = || {
        let mut d = data_from_state(&c.model, state);
        for _ in 0..1000 {
            step(&c.model, &mut d);
        }
        d
    };
    let (a, b) = (run(), run());
    assert_eq!(bits(&a.qpos), bits(&b.qpos));
    assert_eq!(bits(&a.qvel), bits(&b.qvel));
    assert_eq!(a.time.to_bits(), b.time.to_bits());
    // the run is not trivially equal because it is NaN everywhere
    assert!(a.qpos.iter().chain(&a.qvel).all(|x| x.is_finite()));
    assert!(a.qvel.iter().any(|x| x.abs() > 1e-3));
    println!(
        "humanoid f32 after 1000 steps: time {}, max |qvel| {}",
        a.time,
        a.qvel.iter().fold(0.0f32, |m, x| m.max(x.abs()))
    );
}

/// A world of `states.len()` environments, each holding one golden state.
fn world_of(c: &Compiled<f32>, golden: &serde_json::Value, picks: &[usize]) -> HostWorld {
    let mut world = HostWorld::new(&c.scene, picks.len() as u32).expect("world");
    for (env, &k) in picks.iter().enumerate() {
        let s = &golden["states"][k];
        world.set_qpos(env as u32, &farr(&s["qpos"])).expect("qpos");
        for (dst, src) in world
            .env_slice_mut(FieldId::Qvel, env as u32)
            .iter_mut()
            .zip(farr(&s["qvel"]))
        {
            *dst = src as f32;
        }
        for (dst, src) in world.ctrl_mut(env as u32).iter_mut().zip(farr(&s["ctrl"])) {
            *dst = src as f32;
        }
    }
    world
}

const FIELDS: [FieldId; 6] = [
    FieldId::Qpos,
    FieldId::Qvel,
    FieldId::BodyPos,
    FieldId::BodyQuat,
    FieldId::BodyLinVel,
    FieldId::BodyAngVel,
];

fn env_bits(world: &HostWorld, env: u32) -> Vec<Vec<u32>> {
    FIELDS
        .iter()
        .map(|&f| bits(world.env_slice(f, env)))
        .collect()
}

fn step_n(c: &Compiled<f32>, world: &mut HostWorld, n: usize) {
    let mut datas: Vec<Data<f32>> = (0..world.layout().n_envs)
        .map(|_| Data::new(&c.model))
        .collect();
    for _ in 0..n {
        step_world(&c.model, &mut datas, world).expect("step_world");
    }
}

#[test]
fn step_world_is_independent_of_the_other_environments_and_their_order() {
    let c = humanoid32();
    let golden = golden_of(Which::Humanoid);
    let picks = [0usize, 3, 5];
    let steps = 25;

    let mut together = world_of(&c, &golden, &picks);
    step_n(&c, &mut together, steps);

    // each environment alone
    for (env, &k) in picks.iter().enumerate() {
        let mut alone = world_of(&c, &golden, &[k]);
        step_n(&c, &mut alone, steps);
        assert_eq!(
            env_bits(&together, env as u32),
            env_bits(&alone, 0),
            "environment {env} differs from the same environment alone"
        );
    }

    // the environments in reverse order
    let reversed_picks: Vec<usize> = picks.iter().rev().copied().collect();
    let mut reversed = world_of(&c, &golden, &reversed_picks);
    step_n(&c, &mut reversed, steps);
    for env in 0..picks.len() {
        assert_eq!(
            env_bits(&together, env as u32),
            env_bits(&reversed, (picks.len() - 1 - env) as u32),
            "environment {env} differs when the batch is reversed"
        );
    }

    // the environments really are different, and moved
    assert_ne!(env_bits(&together, 0), env_bits(&together, 1));
    assert!(together.qvel(0).iter().any(|v| v.abs() > 1e-3));
}

#[test]
fn step_world_equals_a_plain_step_of_the_same_data() {
    let c = humanoid32();
    let golden = golden_of(Which::Humanoid);
    let mut world = world_of(&c, &golden, &[2]);
    let mut d = Data::<f32>::new(&c.model);
    convert_qpos(
        &c.model,
        world.qpos(0),
        QuatOrder::Xyzw,
        &mut d.qpos,
        QuatOrder::Wxyz,
    );
    d.qvel.copy_from_slice(world.qvel(0));
    d.ctrl.copy_from_slice(world.ctrl(0));
    step(&c.model, &mut d);
    let mut expected = vec![0.0f32; c.model.nq];
    convert_qpos(
        &c.model,
        &d.qpos,
        QuatOrder::Wxyz,
        &mut expected,
        QuatOrder::Xyzw,
    );

    let mut datas = vec![Data::<f32>::new(&c.model)];
    step_world(&c.model, &mut datas, &mut world).unwrap();
    assert_eq!(bits(world.qpos(0)), bits(&expected));
    assert_eq!(bits(world.qvel(0)), bits(&d.qvel));
}

#[test]
fn step_world_poses_are_the_render_view_and_agree_with_the_worlds_own_kinematics() {
    let c = humanoid32();
    let golden = golden_of(Which::Humanoid);
    let mut world = world_of(&c, &golden, &[1, 4]);
    step_n(&c, &mut world, 10);

    // the render view is a bitwise copy of what step_world wrote
    let view = gather_render_view(&world);
    let nb = c.scene.bodies.len();
    for env in 0..2usize {
        for b in 0..nb {
            let (pos, quat) = world.body_pose(env as u32, b);
            let bytes = view.pose_bytes(env * nb + b).expect("pose");
            let mut expect = Vec::new();
            for v in pos {
                expect.extend(v.to_le_bytes());
            }
            expect.extend(0.0f32.to_le_bytes());
            for v in quat {
                expect.extend(v.to_le_bytes());
            }
            assert_eq!(bytes, &expect[..]);
        }
    }

    // sim-world's own f64 forward kinematics of the stepped qpos gives the same poses
    let mut reference = world.clone();
    let mut worst = 0.0f64;
    for env in 0..2u32 {
        let q: Vec<f64> = world.qpos(env).iter().map(|&x| f64::from(x)).collect();
        reference.set_qpos(env, &q).unwrap();
        for b in 0..nb {
            let (p, qa) = world.body_pose(env, b);
            let (rp, rq) = reference.body_pose(env, b);
            for k in 0..3 {
                worst = worst.max(f64::from((p[k] - rp[k]).abs()));
            }
            // a quaternion is its own negation: compare up to sign
            let d1: f32 = (0..4).map(|k| (qa[k] - rq[k]).abs()).fold(0.0, f32::max);
            let d2: f32 = (0..4).map(|k| (qa[k] + rq[k]).abs()).fold(0.0, f32::max);
            worst = worst.max(f64::from(d1.min(d2)));
        }
    }
    println!(
        "MEASURED f32 step_world poses vs sim-world forward kinematics: max_abs_err={worst:e}"
    );
    assert!(worst < 1e-4, "{worst:e}");
}

#[test]
fn step_world_writes_world_frame_body_velocities() {
    // The double pendulum, both links turning about +y. With the new joint angles
    // t1 and rates w1, w2 (read back from the world), the hand-derived velocities
    // of the body frames in the world frame are:
    //   body a (its origin is the hinge): linear 0, angular (0, w1, 0);
    //   body b (its origin is 1 m along the link, r = (-sin t1, 0, -cos t1)):
    //     linear w1 y x r = (-w1 cos t1, 0, w1 sin t1), angular (0, w1 + w2, 0).
    let c = compile::<f32>(load_scene(&fixtures().join("double_pendulum.xml")));
    let mut world = HostWorld::new(&c.scene, 1).unwrap();
    world.set_qpos(0, &[0.7, -0.4]).unwrap();
    world
        .env_slice_mut(FieldId::Qvel, 0)
        .copy_from_slice(&[2.0, 0.5]);
    let mut datas = vec![Data::<f32>::new(&c.model)];
    step_world(&c.model, &mut datas, &mut world).unwrap();
    let (t1, w1, w2) = (world.qpos(0)[0], world.qvel(0)[0], world.qvel(0)[1]);
    let (lin, ang) = (world.body_linvel(0), world.body_angvel(0));
    let close = |a: f32, b: f32| (a - b).abs() < 2e-5;
    let expect_lin = [0.0, 0.0, 0.0, -w1 * t1.cos(), 0.0, w1 * t1.sin()];
    let expect_ang = [0.0, w1, 0.0, 0.0, w1 + w2, 0.0];
    for k in 0..6 {
        assert!(
            close(lin[k], expect_lin[k]),
            "linvel {k}: {} vs {}",
            lin[k],
            expect_lin[k]
        );
        assert!(
            close(ang[k], expect_ang[k]),
            "angvel {k}: {} vs {}",
            ang[k],
            expect_ang[k]
        );
    }
}

#[test]
fn mismatched_sizes_are_refused_and_change_nothing() {
    let c = humanoid32();
    let mut world = HostWorld::new(&c.scene, 2).unwrap();
    let before = world.clone();
    let mut one = vec![Data::<f32>::new(&c.model)];
    assert!(step_world(&c.model, &mut one, &mut world).is_err());
    assert!(world.bit_eq(&before));
    // a model of another scene
    let other = compile::<f32>(load_scene(&fixtures().join("zoo.xml")));
    let mut datas = vec![
        Data::<f32>::new(&other.model),
        Data::<f32>::new(&other.model),
    ];
    assert!(step_world(&other.model, &mut datas, &mut world).is_err());
}
