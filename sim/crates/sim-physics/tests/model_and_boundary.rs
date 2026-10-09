//! The compiled model, and the quaternion boundary with `HostWorld`.
//!
//! - `Model::compile` is held to MuJoCo's compiled arrays (the `arrays` section of
//!   the golden files): body tree, joint and dof addresses, the tree of dofs, masses,
//!   inertias, stiffness, damping, armature, `qpos0` and `qpos_spring`.
//! - The quaternion order is converted in one function, [`sim_physics::convert_qpos`]
//!   (scene and `HostWorld`: `[x, y, z, w]`; physics: `[w, x, y, z]`): a round trip
//!   gives back the same bits, and a hand-written example gives the expected layout.
//! - The body index convention: `scene.bodies[b]` is internal body `b + 1`.

mod common;

use common::*;
use serde_json::Value;
use sim_physics::{
    ActuatorType, JointType, Model, PhysicsError, QuatOrder, convert_qpos, internal_body_to_scene,
    reorder_quat, scene_body_to_internal,
};
use sim_scene::{Body, BodyId, Inertial, Joint, JointKind, Scene};

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

fn usizes(a: &[usize]) -> Vec<i64> {
    a.iter().map(|&x| x as i64).collect()
}

fn close(name: &str, ours: &[f64], reference: &Value) {
    let reference = farr(reference);
    let e = compare(ours, &reference);
    assert!(
        e.within(1e-12, 1e-12),
        "{name}: {:e} (max |ref| {:e})",
        e.abs,
        e.scale
    );
}

#[test]
fn the_compiled_arrays_match_mujoco() {
    for which in MODELS {
        let golden = golden_of(which);
        let a = &golden["arrays"];
        let c = compile::<f64>(scene_of(which));
        let m = &c.model;
        let name = which.name();

        // integer arrays, exactly
        assert_eq!(
            usizes(&m.body_parentid),
            ints(&a["body_parentid"]),
            "{name} body_parentid"
        );
        assert_eq!(
            usizes(&m.body_rootid),
            ints(&a["body_rootid"]),
            "{name} body_rootid"
        );
        assert_eq!(
            usizes(&m.body_weldid),
            ints(&a["body_weldid"]),
            "{name} body_weldid"
        );
        assert_eq!(
            usizes(&m.body_jntnum),
            ints(&a["body_jntnum"]),
            "{name} body_jntnum"
        );
        assert_eq!(
            usizes(&m.body_dofnum),
            ints(&a["body_dofnum"]),
            "{name} body_dofnum"
        );
        let types: Vec<i64> = m
            .jnt_type
            .iter()
            .map(|t| match t {
                JointType::Free => 0,
                JointType::Ball => 1,
                JointType::Slide => 2,
                JointType::Hinge => 3,
            })
            .collect();
        assert_eq!(types, ints(&a["jnt_type"]), "{name} jnt_type");
        assert_eq!(
            usizes(&m.jnt_qposadr),
            ints(&a["jnt_qposadr"]),
            "{name} jnt_qposadr"
        );
        assert_eq!(
            usizes(&m.jnt_dofadr),
            ints(&a["jnt_dofadr"]),
            "{name} jnt_dofadr"
        );
        assert_eq!(
            usizes(&m.jnt_bodyid),
            ints(&a["jnt_bodyid"]),
            "{name} jnt_bodyid"
        );
        assert_eq!(
            usizes(&m.dof_bodyid),
            ints(&a["dof_bodyid"]),
            "{name} dof_bodyid"
        );
        assert_eq!(
            usizes(&m.dof_jntid),
            ints(&a["dof_jntid"]),
            "{name} dof_jntid"
        );
        let parents: Vec<i64> = m.dof_parentid.iter().map(|&x| i64::from(x)).collect();
        assert_eq!(parents, ints(&a["dof_parentid"]), "{name} dof_parentid");

        // float arrays, to 1e-12
        close(&format!("{name} body_mass"), &m.body_mass, &a["body_mass"]);
        close(
            &format!("{name} body_subtreemass"),
            &m.body_subtreemass,
            &a["body_subtreemass"],
        );
        close(&format!("{name} body_pos"), &m.body_pos, &a["body_pos"]);
        close(&format!("{name} body_ipos"), &m.body_ipos, &a["body_ipos"]);
        close(
            &format!("{name} body_inertia"),
            &m.body_inertia,
            &a["body_inertia"],
        );
        close(&format!("{name} jnt_pos"), &m.jnt_pos, &a["jnt_pos"]);
        close(&format!("{name} jnt_axis"), &m.jnt_axis, &a["jnt_axis"]);
        close(
            &format!("{name} jnt_stiffness"),
            &m.jnt_stiffness,
            &a["jnt_stiffness"],
        );
        close(
            &format!("{name} dof_armature"),
            &m.dof_armature,
            &a["dof_armature"],
        );
        close(
            &format!("{name} dof_damping"),
            &m.dof_damping,
            &a["dof_damping"],
        );
        // quaternions: ours are [w, x, y, z], the golden file's [x, y, z, w]
        let to_xyzw = |q: &[f64]| -> Vec<f64> {
            q.chunks(4).flat_map(|c| [c[1], c[2], c[3], c[0]]).collect()
        };
        for (key, ours) in [("body_quat", &m.body_quat), ("body_iquat", &m.body_iquat)] {
            let e = compare_quats_up_to_sign(&to_xyzw(ours), &farr(&a[key]));
            assert!(e.within(1e-12, 1e-12), "{name} {key}: {:e}", e.abs);
        }
        // qpos0 and qpos_spring: the golden file is in the scene layout
        let scene_q0 = scene_qpos(m, &m.qpos0);
        let scene_qs = scene_qpos(m, &m.qpos_spring);
        close(&format!("{name} qpos0"), &scene_q0, &a["qpos0"]);
        close(&format!("{name} qpos_spring"), &scene_qs, &a["qpos_spring"]);
        assert_eq!(m.qpos0, m.qpos_spring);
    }
}

#[test]
fn the_zoo_tree_of_dofs_is_the_one_mujoco_builds() {
    // MuJoCo 3.14.0, m.dof_parentid of zoo.xml (free 0-5; ball 6-8; elbow 9; slider
    // 10; wrist hinge 11 and slide 12 hang from the ball's last dof 8)
    let c = compile::<f64>(scene_of(Which::Zoo));
    assert_eq!(
        c.model.dof_parentid,
        vec![-1, 0, 1, 2, 3, 4, -1, 6, 7, 8, 9, 8, 11]
    );
    assert_eq!(
        (c.model.nq, c.model.nv, c.model.nu, c.model.nbody),
        (15, 13, 2, 7)
    );
    assert_eq!(
        c.model.actuator_type,
        vec![ActuatorType::Motor, ActuatorType::Position]
    );
    assert_eq!(c.model.actuator_ctrllimited, vec![true, true]);
}

#[test]
fn bodies_are_numbered_with_the_world_as_body_zero() {
    assert_eq!(scene_body_to_internal(0), 1);
    assert_eq!(scene_body_to_internal(6), 7);
    assert_eq!(internal_body_to_scene(0), None);
    assert_eq!(internal_body_to_scene(1), Some(0));
    for b in 0..20 {
        assert_eq!(internal_body_to_scene(scene_body_to_internal(b)), Some(b));
    }
    let scene = scene_of(Which::Zoo);
    let c = compile::<f64>(scene.clone());
    assert_eq!(c.model.nbody, scene.bodies.len() + 1);
    // the world: no parent, no mass
    assert_eq!(c.model.body_parentid[0], 0);
    assert_eq!(c.model.body_mass[0], 0.0);
    // scene body "tip" (index 2) is internal body 3 and its parent "mid" is internal 2
    let tip = scene.bodies.iter().position(|b| b.name == "tip").unwrap();
    let mid = scene.bodies.iter().position(|b| b.name == "mid").unwrap();
    assert_eq!(
        c.model.body_parentid[scene_body_to_internal(tip)],
        scene_body_to_internal(mid)
    );
    // the body masses are the scene's
    for (b, body) in scene.bodies.iter().enumerate() {
        let want = body.inertial.map_or(0.0, |i| i.mass_kg);
        assert_eq!(c.model.body_mass[scene_body_to_internal(b)], want);
    }
}

#[test]
fn convert_qpos_round_trips_bit_for_bit() {
    let c = compile::<f64>(scene_of(Which::Zoo));
    let m = &c.model;
    // awkward numbers, so that a reordering mistake cannot hide in symmetric values
    let src: Vec<f64> = (0..m.nq)
        .map(|i| (i as f64 + 0.3) * 1.0000000000000002 / 7.0)
        .collect();
    let mut physics = vec![0.0; m.nq];
    let mut back = vec![0.0; m.nq];
    convert_qpos(m, &src, QuatOrder::Xyzw, &mut physics, QuatOrder::Wxyz);
    convert_qpos(m, &physics, QuatOrder::Wxyz, &mut back, QuatOrder::Xyzw);
    assert_eq!(
        src.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        back.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    assert_ne!(src, physics, "the order really changed");
    // the same order is the identity
    let mut same = vec![0.0; m.nq];
    convert_qpos(m, &src, QuatOrder::Wxyz, &mut same, QuatOrder::Wxyz);
    assert_eq!(src, same);
    // f32 too
    let c32 = compile::<f32>(scene_of(Which::Zoo));
    let src32: Vec<f32> = src.iter().map(|&x| x as f32).collect();
    let mut p32 = vec![0.0f32; m.nq];
    let mut b32 = vec![0.0f32; m.nq];
    convert_qpos(
        &c32.model,
        &src32,
        QuatOrder::Xyzw,
        &mut p32,
        QuatOrder::Wxyz,
    );
    convert_qpos(&c32.model, &p32, QuatOrder::Wxyz, &mut b32, QuatOrder::Xyzw);
    assert_eq!(src32, b32);
}

#[test]
fn convert_qpos_against_a_hand_written_example() {
    // zoo joints in order: free (7), ball (4), hinge (1), slide (1), hinge (1), slide (1)
    let c = compile::<f64>(scene_of(Which::Zoo));
    let scene_layout = [
        // free: position, then quaternion x y z w
        1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9, //
        // ball: quaternion x y z w
        0.4, 0.5, 0.6, 0.7, //
        // hinge, slide, hinge, slide
        10.0, 11.0, 12.0, 13.0,
    ];
    let expected_physics = [
        1.0, 2.0, 3.0, 0.9, 0.1, 0.2, 0.3, // free: position, then w x y z
        0.7, 0.4, 0.5, 0.6, // ball: w x y z
        10.0, 11.0, 12.0, 13.0,
    ];
    let mut physics = [0.0; 15];
    convert_qpos(
        &c.model,
        &scene_layout,
        QuatOrder::Xyzw,
        &mut physics,
        QuatOrder::Wxyz,
    );
    assert_eq!(physics, expected_physics);
    let mut back = [0.0; 15];
    convert_qpos(
        &c.model,
        &expected_physics,
        QuatOrder::Wxyz,
        &mut back,
        QuatOrder::Xyzw,
    );
    assert_eq!(back, scene_layout);

    // one quaternion
    assert_eq!(
        reorder_quat([0.1, 0.2, 0.3, 0.9], QuatOrder::Xyzw, QuatOrder::Wxyz),
        [0.9, 0.1, 0.2, 0.3]
    );
    assert_eq!(
        reorder_quat([0.9, 0.1, 0.2, 0.3], QuatOrder::Wxyz, QuatOrder::Xyzw),
        [0.1, 0.2, 0.3, 0.9]
    );
}

#[test]
fn a_physics_qpos_at_the_scene_pose_is_the_models_reference_pose() {
    // the world's reset pose (sim-world) converted to the physics order is qpos0
    use sim_world::HostWorld;
    let c = compile::<f32>(scene_of(Which::Zoo));
    let world = HostWorld::new(&c.scene, 1).unwrap();
    let mut physics = vec![0.0f32; c.model.nq];
    convert_qpos(
        &c.model,
        world.qpos(0),
        QuatOrder::Xyzw,
        &mut physics,
        QuatOrder::Wxyz,
    );
    for (a, b) in physics.iter().zip(&c.model.qpos0) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }
}

fn one_body_scene(mass: f64, with_joint: bool) -> Scene {
    let mut s = Scene::new();
    s.bodies.push(Body {
        name: "b".into(),
        parent: None,
        pos: [0.0; 3],
        quat: [0.0, 0.0, 0.0, 1.0],
        inertial: Some(Inertial {
            mass_kg: mass,
            com: [0.0; 3],
            diag_inertia: [0.1, 0.1, 0.1],
            inertia_quat: [0.0, 0.0, 0.0, 1.0],
        }),
    });
    if with_joint {
        s.joints.push(Joint {
            name: "j".into(),
            body: BodyId(0),
            kind: JointKind::Hinge {
                axis: [0.0, 0.0, 1.0],
            },
            pos: [0.0; 3],
            range: None,
            stiffness: 0.0,
            damping: 0.0,
            armature: 0.0,
            frictionloss: 0.0,
            solref_limit: sim_scene::DEFAULT_SOLREF,
            solimp_limit: sim_scene::DEFAULT_SOLIMP,
            solref_friction: sim_scene::DEFAULT_SOLREF,
            solimp_friction: sim_scene::DEFAULT_SOLIMP,
            margin: 0.0,
        });
    }
    s
}

#[test]
fn a_moving_body_without_mass_is_refused_but_a_static_one_is_fine() {
    // MuJoCo: "mass and inertia of moving bodies must be larger than mjMINVAL"
    let bad = one_body_scene(0.0, true);
    assert_eq!(
        Model::<f64>::compile(&bad).err(),
        Some(PhysicsError::DegenerateMass { body: 0 })
    );
    let ok = one_body_scene(0.0, false);
    assert!(Model::<f64>::compile(&ok).is_ok());
    let fine = one_body_scene(1.0, true);
    assert!(Model::<f64>::compile(&fine).is_ok());
}

#[test]
fn an_invalid_scene_is_refused_not_panicked_on() {
    let mut s = one_body_scene(1.0, true);
    s.bodies[0].parent = Some(BodyId(0)); // its own parent
    assert!(matches!(
        Model::<f64>::compile(&s),
        Err(PhysicsError::Scene { .. })
    ));
}

#[test]
fn a_scene_without_dofs_or_actuators_steps_without_a_panic() {
    use sim_physics::{Data, Integrator, energy, forward, step};
    for integrator in [Integrator::Euler, Integrator::Rk4] {
        let mut scene = one_body_scene(1.0, false);
        scene.integrator = integrator;
        let (m, nm) = Model::<f64>::compile(&scene).unwrap();
        assert!(nm.is_empty());
        assert_eq!((m.nq, m.nv, m.nu), (0, 0, 0));
        let mut d = Data::new(&m);
        forward(&m, &mut d);
        step(&m, &mut d);
        step(&m, &mut d);
        assert_eq!(d.time, 2.0 * m.timestep);
        // the static body's potential energy: -m g . x = 0 at the origin
        assert_eq!(energy(&m, &mut d), 0.0);
    }
}
