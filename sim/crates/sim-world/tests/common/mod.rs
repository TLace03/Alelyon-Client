//! Scenes the sim-world tests share.

#![allow(dead_code)]

use sim_scene::{
    Body, BodyId, Camera, CameraMount, Geom, Instance, Joint, JointKind, Material, MaterialId,
    Scene, Shape, ShapeRef,
};

/// A free base at (1, 2, 3) and an arm child at local (0.5, 0, 0), turned 180
/// degrees about z, on a hinge about z through the body origin.
///
/// Every number is exactly representable in f32 and every product of the
/// hand-computed render-view test is exact, so the expected bytes can be written
/// by hand.
pub fn two_body_scene() -> Scene {
    let mut scene = Scene::new();
    scene.name = "two-body".into();
    scene.materials.push(Material::named("plain"));
    scene.bodies.push(Body {
        name: "base".into(),
        parent: None,
        pos: [1.0, 2.0, 3.0],
        quat: [0.0, 0.0, 0.0, 1.0],
        inertial: None,
    });
    scene.bodies.push(Body {
        name: "arm".into(),
        parent: Some(BodyId(0)),
        pos: [0.5, 0.0, 0.0],
        quat: [0.0, 0.0, 1.0, 0.0],
        inertial: None,
    });
    scene.joints.push(Joint {
        name: "root".into(),
        body: BodyId(0),
        kind: JointKind::Free,
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
    scene.joints.push(Joint {
        name: "elbow".into(),
        body: BodyId(1),
        kind: JointKind::Hinge {
            axis: [0.0, 0.0, 1.0],
        },
        pos: [0.0; 3],
        range: Some([-1.5, 1.5]),
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
    scene.geoms.push(Geom {
        name: "ball".into(),
        body: Some(BodyId(1)),
        shape: Shape::Sphere { r: 0.1 },
        pos: [0.0; 3],
        quat: [0.0, 0.0, 0.0, 1.0],
        material: MaterialId(0),
        contype: 1,
        conaffinity: 1,
        condim: 3,
        friction: [1.0, 0.005, 0.0001],
        density: 1000.0,
        solref: sim_scene::DEFAULT_SOLREF,
        solimp: sim_scene::DEFAULT_SOLIMP,
        solmix: 1.0,
        priority: 0,
        margin: 0.0,
        gap: 0.0,
    });
    scene.instances.push(Instance {
        body: Some(BodyId(1)),
        mesh_or_geom: ShapeRef::Geom(sim_scene::GeomId(0)),
        material: MaterialId(0),
        local_pos: [0.0; 3],
        local_quat: [0.0, 0.0, 0.0, 1.0],
        seg_id: 1,
    });
    scene.cameras.push(Camera {
        name: "overhead".into(),
        mount: CameraMount::World {
            pos: [0.0, 0.0, 2.0],
            quat: [0.5, 0.5, 0.5, 0.5],
        },
        fx: 100.0,
        fy: 100.0,
        cx: 63.5,
        cy: 47.5,
        near: 0.1,
        far: 50.0,
        width: 128,
        height: 96,
    });
    scene.cameras.push(Camera {
        name: "wrist".into(),
        mount: CameraMount::Body {
            body: BodyId(1),
            local_pos: [1.0, 0.0, 0.0],
            local_quat: [0.5, 0.5, 0.5, 0.5],
        },
        fx: 100.0,
        fy: 100.0,
        cx: 63.5,
        cy: 47.5,
        near: 0.1,
        far: 50.0,
        width: 128,
        height: 96,
    });
    scene.validate().expect("the two-body scene is valid");
    scene
}

/// 32-bit words (typed by hand as IEEE-754 bit patterns) to little-endian bytes.
pub fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}
