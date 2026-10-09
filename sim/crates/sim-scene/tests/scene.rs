//! The scene description: validation, units and conventions, JSON.

use sim_scene::material::Checker;
use sim_scene::{
    Actuator, ActuatorKind, Body, BodyId, Camera, CameraMount, Cone, DEFAULT_SOLIMP,
    DEFAULT_SOLREF, Geom, GeomId, Inertial, Instance, Integrator, Joint, JointId, JointKind,
    Material, MaterialId, Mesh, MeshId, Optical, Scene, SceneError, Shape, ShapeRef, Solver,
    SolverOptions, Tendon, TendonJoint, Unsupported, srgb_to_linear,
};

fn body(name: &str, parent: Option<u32>) -> Body {
    Body {
        name: name.into(),
        parent: parent.map(BodyId),
        pos: [0.0; 3],
        quat: [0.0, 0.0, 0.0, 1.0],
        inertial: Some(Inertial {
            mass_kg: 1.0,
            com: [0.0; 3],
            diag_inertia: [0.1, 0.2, 0.25],
            inertia_quat: [0.0, 0.0, 0.0, 1.0],
        }),
    }
}

fn joint(name: &str, b: u32, kind: JointKind) -> Joint {
    Joint {
        name: name.into(),
        body: BodyId(b),
        kind,
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
    }
}

fn geom(name: &str, b: Option<u32>, shape: Shape) -> Geom {
    Geom {
        name: name.into(),
        body: b.map(BodyId),
        shape,
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
    }
}

/// A scene that uses every kind of record, valid.
fn full_scene() -> Scene {
    let mut s = Scene::new();
    s.name = "full".into();
    s.gravity = [0.0, 0.0, -9.80665];
    s.timestep_s = 0.001;
    s.materials.push(Material::named("steel"));
    let mut painted = Material::named("painted");
    painted.optical = Optical {
        base_colour_linear: [0.8, 0.1, 0.1],
        roughness: 0.3,
        metallic: 0.0,
        emission_linear: [0.0, 0.0, 2.5],
        checker: Some(Checker {
            cells_per_m: 4.0,
            dark_multiplier: 0.25,
        }),
    };
    s.materials.push(painted);
    s.meshes.push(Mesh {
        name: "tetra".into(),
        vertices: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        triangles: vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
    });
    s.bodies.push(body("base", None));
    s.bodies.push(body("link", Some(0)));
    s.bodies.push(body("tip", Some(1)));
    s.joints.push(joint("root", 0, JointKind::Free));
    let mut hinge = joint(
        "elbow",
        1,
        JointKind::Hinge {
            axis: [0.0, 1.0, 0.0],
        },
    );
    hinge.range = Some([-1.5, 1.5]);
    hinge.stiffness = 2.0;
    hinge.damping = 0.1;
    hinge.armature = 0.01;
    hinge.frictionloss = 0.02;
    s.joints.push(hinge);
    s.joints.push(joint(
        "slider",
        2,
        JointKind::Slide {
            axis: [1.0, 0.0, 0.0],
        },
    ));
    s.geoms.push(geom(
        "floor",
        None,
        Shape::Plane {
            size: [0.0, 0.0, 0.05],
        },
    ));
    s.geoms
        .push(geom("ball", Some(0), Shape::Sphere { r: 0.1 }));
    s.geoms.push(geom(
        "arm",
        Some(1),
        Shape::Capsule {
            r: 0.02,
            half_len: 0.2,
        },
    ));
    s.geoms.push(geom(
        "slab",
        Some(1),
        Shape::Box {
            half: [0.1, 0.2, 0.05],
        },
    ));
    s.geoms.push(geom(
        "can",
        Some(2),
        Shape::Cylinder {
            r: 0.03,
            half_len: 0.06,
        },
    ));
    s.geoms.push(geom(
        "egg",
        Some(2),
        Shape::Ellipsoid {
            radii: [0.03, 0.02, 0.04],
        },
    ));
    s.geoms
        .push(geom("shard", Some(2), Shape::Mesh { mesh: MeshId(0) }));
    s.instances.push(Instance {
        body: Some(BodyId(1)),
        mesh_or_geom: ShapeRef::Geom(GeomId(2)),
        material: MaterialId(1),
        local_pos: [0.0, 0.0, 0.01],
        local_quat: [0.0, 0.0, 0.0, 1.0],
        seg_id: 7,
    });
    s.instances.push(Instance {
        body: None,
        mesh_or_geom: ShapeRef::Mesh(MeshId(0)),
        material: MaterialId(0),
        local_pos: [1.0, 2.0, 3.0],
        local_quat: [0.0, 0.0, 0.0, 1.0],
        seg_id: 65535,
    });
    s.cameras.push(Camera {
        name: "fixed".into(),
        mount: CameraMount::World {
            pos: [0.0, -2.0, 1.0],
            quat: [0.5, 0.5, 0.5, 0.5],
        },
        fx: 224.0,
        fy: 224.0,
        cx: 223.5,
        cy: 223.5,
        near: 0.05,
        far: 20.0,
        width: 448,
        height: 448,
    });
    s.cameras.push(Camera {
        name: "wrist".into(),
        mount: CameraMount::Body {
            body: BodyId(2),
            local_pos: [0.0, 0.0, 0.1],
            local_quat: [0.0, 0.0, 0.0, 1.0],
        },
        fx: 112.0,
        fy: 112.0,
        cx: 111.5,
        cy: 111.5,
        near: 0.02,
        far: 5.0,
        width: 224,
        height: 224,
    });
    s.actuators.push(Actuator {
        name: "m".into(),
        joint: JointId(1),
        kind: ActuatorKind::Motor {
            gear: 40.0,
            ctrlrange: Some([-1.0, 1.0]),
        },
    });
    s.actuators.push(Actuator {
        name: "p".into(),
        joint: JointId(2),
        kind: ActuatorKind::Position {
            kp: 100.0,
            gear: 2.0,
            ctrlrange: None,
        },
    });
    s.tendons.push(Tendon {
        name: "t".into(),
        joints: vec![
            TendonJoint {
                joint: JointId(1),
                coef: 0.5,
            },
            TendonJoint {
                joint: JointId(2),
                coef: -0.5,
            },
        ],
        range: Some([-0.3, 2.0]),
        stiffness: 1.0,
        damping: 0.5,
        armature: 0.0,
        frictionloss: 0.0,
        solref_limit: sim_scene::DEFAULT_SOLREF,
        solimp_limit: sim_scene::DEFAULT_SOLIMP,
        solref_friction: sim_scene::DEFAULT_SOLREF,
        solimp_friction: sim_scene::DEFAULT_SOLIMP,
        margin: 0.0,
    });
    s.unsupported.push(Unsupported {
        path: "visual".into(),
        item: "element".into(),
        line: 12,
        reason: "rendering parameters".into(),
    });
    s
}

fn invalid(scene: &Scene, path_contains: &str) {
    match scene.validate() {
        Err(SceneError::Invalid { path, reason }) => assert!(
            path.contains(path_contains),
            "expected a refusal at '{path_contains}', got '{path}': {reason}"
        ),
        other => panic!("expected a refusal at '{path_contains}', got {other:?}"),
    }
}

fn mutate(f: impl FnOnce(&mut Scene)) -> Scene {
    let mut s = full_scene();
    f(&mut s);
    s
}

#[test]
fn the_full_scene_is_valid_and_the_empty_scene_too() {
    full_scene().validate().unwrap();
    Scene::new().validate().unwrap();
    let s = Scene::new();
    assert_eq!(s.gravity, [0.0, 0.0, -9.81]);
    assert_eq!(s.timestep_s, 0.002);
    assert_eq!((s.nq(), s.nv()), (0, 0));
    // free 7/6 + hinge 1/1 + slide 1/1
    let f = full_scene();
    assert_eq!((f.nq(), f.nv()), (9, 8));
}

#[test]
fn json_round_trips_exactly() {
    let s = full_scene();
    let json = s.to_json().unwrap();
    assert_eq!(Scene::from_json(&json).unwrap(), s);
    let pretty = s.to_json_pretty().unwrap();
    assert_eq!(Scene::from_json(&pretty).unwrap(), s);
    // the writer is deterministic
    assert_eq!(json, s.to_json().unwrap());
}

#[test]
fn the_integrator_defaults_to_euler_and_round_trips_as_snake_case() {
    assert_eq!(Scene::new().integrator, Integrator::Euler);
    assert_eq!(Integrator::default(), Integrator::Euler);
    let mut s = full_scene();
    s.integrator = Integrator::Rk4;
    let json = s.to_json().unwrap();
    assert!(json.contains(r#""integrator":"rk4""#), "{json}");
    assert_eq!(Scene::from_json(&json).unwrap(), s);
    // a document written before the field existed reads as Euler
    let without = json.replace(r#""integrator":"rk4","#, "");
    assert!(!without.contains("integrator"));
    assert_eq!(
        Scene::from_json(&without).unwrap().integrator,
        Integrator::Euler
    );
    // only the two modelled integrators exist in JSON
    let bad = json.replace(r#""integrator":"rk4""#, r#""integrator":"implicit""#);
    assert!(matches!(
        Scene::from_json(&bad),
        Err(SceneError::Json { .. })
    ));
}

#[test]
fn json_round_trips_awkward_floats_bit_for_bit() {
    let mut s = full_scene();
    s.gravity = [0.1 + 0.2, 1e-300, -f64::MAX / 1e10];
    s.bodies[0].pos = [std::f64::consts::PI, std::f64::consts::E, 1.0 / 3.0];
    s.meshes[0].vertices[1] = [f32::MIN_POSITIVE, 0.1, 16_777_217.0];
    let back = Scene::from_json(&s.to_json().unwrap()).unwrap();
    for k in 0..3 {
        assert_eq!(back.gravity[k].to_bits(), s.gravity[k].to_bits());
        assert_eq!(
            back.bodies[0].pos[k].to_bits(),
            s.bodies[0].pos[k].to_bits()
        );
        assert_eq!(
            back.meshes[0].vertices[1][k].to_bits(),
            s.meshes[0].vertices[1][k].to_bits()
        );
    }
}

#[test]
fn json_documented_shapes_and_defaults() {
    // A hand-written scene may omit every field that has a documented default.
    let json = r#"{
        "version": 0,
        "materials": [{"name": "m"}],
        "bodies": [{"name": "b", "parent": null}],
        "geoms": [{"name": "g", "body": 0, "shape": {"kind": "sphere", "r": 0.5}, "material": 0}]
    }"#;
    let s = Scene::from_json(json).unwrap();
    assert_eq!(s.gravity, [0.0, 0.0, -9.81]);
    assert_eq!(s.timestep_s, 0.002);
    assert_eq!(s.integrator, Integrator::Euler);
    assert_eq!(s.bodies[0].quat, [0.0, 0.0, 0.0, 1.0]);
    let g = &s.geoms[0];
    assert_eq!((g.contype, g.conaffinity, g.condim), (1, 1, 3));
    assert_eq!(g.friction, [1.0, 0.005, 0.0001]);
    assert_eq!(g.density, 1000.0);
    assert_eq!(s.materials[0], Material::named("m"));
}

#[test]
fn a_position_servo_without_a_gear_has_unit_gear() {
    // scenes written before the field existed keep MuJoCo's default
    let mut value: serde_json::Value =
        serde_json::from_str(&full_scene().to_json().unwrap()).unwrap();
    let kind = &mut value["actuators"][1]["kind"];
    assert_eq!(kind["gear"], 2.0);
    kind.as_object_mut().unwrap().remove("gear");
    let s = Scene::from_json(&value.to_string()).unwrap();
    assert_eq!(
        s.actuators[1].kind,
        ActuatorKind::Position {
            kp: 100.0,
            gear: 1.0,
            ctrlrange: None
        }
    );
}

#[test]
fn soft_constraint_fields_default_to_mujocos_and_round_trip() {
    // a scene written before the fields existed reads with MuJoCo's defaults
    let mut value: serde_json::Value =
        serde_json::from_str(&full_scene().to_json().unwrap()).unwrap();
    for key in ["joints", "tendons"] {
        for item in value[key].as_array_mut().unwrap() {
            let o = item.as_object_mut().unwrap();
            for field in [
                "solref_limit",
                "solimp_limit",
                "solref_friction",
                "solimp_friction",
                "margin",
            ] {
                o.remove(field);
            }
        }
    }
    value.as_object_mut().unwrap().remove("options");
    let s = Scene::from_json(&value.to_string()).unwrap();
    for j in &s.joints {
        assert_eq!(j.solref_limit, DEFAULT_SOLREF);
        assert_eq!(j.solimp_limit, DEFAULT_SOLIMP);
        assert_eq!(j.solref_friction, DEFAULT_SOLREF);
        assert_eq!(j.solimp_friction, DEFAULT_SOLIMP);
        assert_eq!(j.margin, 0.0);
    }
    let t = &s.tendons[0];
    assert_eq!(t.solref_limit, [0.02, 1.0]);
    assert_eq!(t.solimp_limit, [0.9, 0.95, 0.001, 0.5, 2.0]);
    assert_eq!(t.solref_friction, [0.02, 1.0]);
    assert_eq!(t.solimp_friction, [0.9, 0.95, 0.001, 0.5, 2.0]);
    assert_eq!(t.margin, 0.0);
    assert_eq!(s, full_scene());
    assert_eq!(Scene::new().options, SolverOptions::default());
    assert_eq!(s.options, SolverOptions::default());

    // `{}` is the default option set, and a partial set keeps the other defaults
    let json = r#"{"version": 0, "options": {"iterations": 12, "solver": "cg"}}"#;
    let o = Scene::from_json(json).unwrap().options;
    assert_eq!((o.solver, o.iterations), (Solver::Cg, 12));
    assert_eq!(
        (
            o.tolerance,
            o.ls_iterations,
            o.ls_tolerance,
            o.cone,
            o.impratio
        ),
        (1e-8, 50, 0.01, Cone::Pyramidal, 1.0)
    );
    assert_eq!(
        Scene::from_json(r#"{"version": 0, "options": {}}"#)
            .unwrap()
            .options,
        SolverOptions::default()
    );

    // non-default values round-trip bit for bit
    let mut s = full_scene();
    s.joints[1].solref_limit = [-1234.5, -0.1 - 0.2];
    s.joints[1].solimp_limit = [0.1 + 0.2, 0.99, 1e-300, 0.25, 3.5];
    s.joints[2].solref_friction = [0.03, 0.7];
    s.joints[2].solimp_friction = [0.5, 0.6, 0.02, 0.9, 1.0];
    s.joints[2].margin = 1.0 / 3.0;
    s.tendons[0].solref_limit = [0.07, 2.0];
    s.tendons[0].margin = 0.25;
    s.options = SolverOptions {
        solver: Solver::Pgs,
        iterations: 3,
        tolerance: 0.0,
        ls_iterations: 7,
        ls_tolerance: 1.0 / 7.0,
        cone: Cone::Elliptic,
        impratio: 0.5,
    };
    let json = s.to_json().unwrap();
    assert!(json.contains(r#""solver":"pgs""#), "{json}");
    assert!(json.contains(r#""cone":"elliptic""#), "{json}");
    let back = Scene::from_json(&json).unwrap();
    assert_eq!(back, s);
    assert_eq!(
        back.joints[2].margin.to_bits(),
        s.joints[2].margin.to_bits()
    );
    assert_eq!(
        back.options.ls_tolerance.to_bits(),
        s.options.ls_tolerance.to_bits()
    );
}

#[test]
fn soft_constraint_invariants() {
    // joints and tendons carry the same five fields; each is checked
    invalid(
        &mutate(|s| s.joints[1].solref_limit = [0.02, -1.0]),
        "joints[1].solref_limit",
    );
    invalid(
        &mutate(|s| s.joints[1].solref_friction = [-5.0, 0.5]),
        "joints[1].solref_friction",
    );
    invalid(
        &mutate(|s| s.joints[1].solref_limit = [f64::NAN, 1.0]),
        "joints[1].solref_limit",
    );
    // the direct format (both not positive) and the standard one are both fine
    mutate(|s| s.joints[1].solref_limit = [-100.0, -10.0])
        .validate()
        .unwrap();
    mutate(|s| s.joints[1].solref_limit = [0.0, 0.0])
        .validate()
        .unwrap();
    invalid(
        &mutate(|s| s.joints[1].solimp_limit = [0.9, 1.5, 0.001, 0.5, 2.0]),
        "joints[1].solimp_limit",
    );
    invalid(
        &mutate(|s| s.joints[1].solimp_friction = [-0.1, 0.9, 0.001, 0.5, 2.0]),
        "joints[1].solimp_friction",
    );
    invalid(
        &mutate(|s| s.joints[1].solimp_limit = [0.9, 0.95, -0.001, 0.5, 2.0]),
        "joints[1].solimp_limit",
    );
    invalid(
        &mutate(|s| s.joints[1].solimp_limit = [0.9, 0.95, 0.001, 1.5, 2.0]),
        "joints[1].solimp_limit",
    );
    invalid(
        &mutate(|s| s.joints[1].solimp_limit = [0.9, 0.95, 0.001, 0.5, 0.5]),
        "joints[1].solimp_limit",
    );
    invalid(
        &mutate(|s| s.joints[1].solimp_limit = [0.9, 0.95, f64::INFINITY, 0.5, 2.0]),
        "joints[1].solimp_limit",
    );
    // MuJoCo's own humanoid uses an impedance of 0 (it clamps it): allowed
    mutate(|s| s.joints[1].solimp_limit = [0.0, 0.99, 0.01, 0.5, 2.0])
        .validate()
        .unwrap();
    invalid(&mutate(|s| s.joints[1].margin = -0.01), "joints[1].margin");
    invalid(
        &mutate(|s| s.joints[1].margin = f64::NAN),
        "joints[1].margin",
    );
    // tendons
    invalid(
        &mutate(|s| s.tendons[0].solref_limit = [1.0, 0.0]),
        "tendons[0].solref_limit",
    );
    invalid(
        &mutate(|s| s.tendons[0].solref_friction = [0.0, 1.0]),
        "tendons[0].solref_friction",
    );
    invalid(
        &mutate(|s| s.tendons[0].solimp_limit = [0.9, 0.95, 0.001, 0.5, 0.0]),
        "tendons[0].solimp_limit",
    );
    invalid(
        &mutate(|s| s.tendons[0].solimp_friction = [2.0, 0.95, 0.001, 0.5, 2.0]),
        "tendons[0].solimp_friction",
    );
    invalid(&mutate(|s| s.tendons[0].margin = -1.0), "tendons[0].margin");
    // the solver options
    invalid(
        &mutate(|s| s.options.tolerance = -1e-8),
        "options.tolerance",
    );
    invalid(
        &mutate(|s| s.options.tolerance = f64::NAN),
        "options.tolerance",
    );
    invalid(
        &mutate(|s| s.options.ls_tolerance = -0.01),
        "options.ls_tolerance",
    );
    invalid(&mutate(|s| s.options.impratio = 0.0), "options.impratio");
    invalid(
        &mutate(|s| s.options.impratio = f64::INFINITY),
        "options.impratio",
    );
    // a tolerance of 0 never stops early, and zero iterations are legal
    mutate(|s| {
        s.options.tolerance = 0.0;
        s.options.iterations = 0;
        s.options.ls_iterations = 0;
    })
    .validate()
    .unwrap();
}

#[test]
fn unknown_fields_are_refused_at_every_level() {
    let base = full_scene().to_json().unwrap();
    let value: serde_json::Value = serde_json::from_str(&base).unwrap();
    // add an unknown field to the scene, a body, a joint (and its kind), a geom
    // (and its shape), a material (and each of its four parts), an instance (and its
    // shape reference), a camera (and its mount), an actuator (and its kind), a
    // tendon, a mesh and an unsupported record
    let targets: Vec<Vec<serde_json::Value>> = vec![
        vec![],
        vec!["options".into()],
        vec!["bodies".into(), 0.into()],
        vec!["joints".into(), 1.into()],
        vec!["joints".into(), 1.into(), "kind".into()],
        vec!["geoms".into(), 1.into()],
        vec!["geoms".into(), 1.into(), "shape".into()],
        vec!["materials".into(), 1.into()],
        vec!["materials".into(), 1.into(), "mechanical".into()],
        vec!["materials".into(), 1.into(), "thermal".into()],
        vec!["materials".into(), 1.into(), "acoustic".into()],
        vec!["materials".into(), 1.into(), "optical".into()],
        vec![
            "materials".into(),
            1.into(),
            "optical".into(),
            "checker".into(),
        ],
        vec!["bodies".into(), 0.into(), "inertial".into()],
        vec!["instances".into(), 0.into()],
        vec!["instances".into(), 0.into(), "mesh_or_geom".into()],
        vec!["cameras".into(), 1.into()],
        vec!["cameras".into(), 1.into(), "mount".into()],
        vec!["actuators".into(), 0.into()],
        vec!["actuators".into(), 0.into(), "kind".into()],
        vec!["tendons".into(), 0.into()],
        vec!["tendons".into(), 0.into(), "joints".into(), 0.into()],
        vec!["meshes".into(), 0.into()],
        vec!["unsupported".into(), 0.into()],
    ];
    for target in targets {
        let mut v = value.clone();
        let mut node = &mut v;
        for step in &target {
            node = match step {
                serde_json::Value::String(k) => node.get_mut(k.as_str()).unwrap(),
                serde_json::Value::Number(n) => node.get_mut(n.as_u64().unwrap() as usize).unwrap(),
                _ => unreachable!(),
            };
        }
        node.as_object_mut()
            .unwrap()
            .insert("surprise".into(), serde_json::json!(1));
        let text = serde_json::to_string(&v).unwrap();
        match Scene::from_json(&text) {
            Err(SceneError::Json { message }) => {
                assert!(message.contains("surprise"), "{target:?}: {message}")
            }
            other => panic!("{target:?}: an unknown field was accepted: {other:?}"),
        }
    }
}

#[test]
fn json_errors_are_json_errors() {
    assert!(matches!(
        Scene::from_json("{"),
        Err(SceneError::Json { .. })
    ));
    assert!(matches!(
        Scene::from_json("{}"),
        Err(SceneError::Json { .. })
    )); // no version
    assert!(matches!(
        Scene::from_json(r#"{"version": 0, "bodies": [{"parent": null}]}"#),
        Err(SceneError::Json { .. })
    )); // a body without a name
    // an absent optional field is None: a body with no "parent" is a child of the world
    let s = Scene::from_json(r#"{"version": 0, "bodies": [{"name": "b"}]}"#).unwrap();
    assert_eq!(s.bodies[0].parent, None);
    assert!(matches!(
        Scene::from_json(r#"{"version": 1}"#),
        Err(SceneError::Invalid { .. })
    ));
    assert!(matches!(
        Scene::from_json(r#"{"version": 0, "gravity": [0, 0, null]}"#),
        Err(SceneError::Json { .. })
    ));
}

#[test]
fn a_scene_of_another_version_is_refused() {
    invalid(&mutate(|s| s.version = 1), "version");
}

#[test]
fn header_numbers_must_be_finite_and_positive() {
    invalid(&mutate(|s| s.gravity[2] = f64::NAN), "gravity");
    invalid(&mutate(|s| s.gravity[0] = f64::INFINITY), "gravity");
    invalid(&mutate(|s| s.timestep_s = 0.0), "timestep_s");
    invalid(&mutate(|s| s.timestep_s = -1.0), "timestep_s");
    invalid(&mutate(|s| s.timestep_s = f64::NAN), "timestep_s");
}

#[test]
fn body_invariants() {
    invalid(
        &mutate(|s| s.bodies[1].quat = [0.0, 0.0, 0.0, 2.0]),
        "bodies[1].quat",
    );
    invalid(&mutate(|s| s.bodies[1].quat = [0.0; 4]), "bodies[1].quat");
    invalid(&mutate(|s| s.bodies[1].pos[1] = f64::NAN), "bodies[1].pos");
    // a parent must come first
    invalid(
        &mutate(|s| s.bodies[1].parent = Some(BodyId(2))),
        "bodies[1].parent",
    );
    invalid(
        &mutate(|s| s.bodies[0].parent = Some(BodyId(0))),
        "bodies[0].parent",
    );
    invalid(
        &mutate(|s| s.bodies[2].name = "base".into()),
        "bodies[2].name",
    );
    // inertia
    invalid(
        &mutate(|s| s.bodies[0].inertial.as_mut().unwrap().mass_kg = -1.0),
        "mass_kg",
    );
    invalid(
        &mutate(|s| s.bodies[0].inertial.as_mut().unwrap().diag_inertia = [0.1, 0.1, 0.5]),
        "diag_inertia",
    );
    invalid(
        &mutate(|s| s.bodies[0].inertial.as_mut().unwrap().diag_inertia[0] = -0.1),
        "diag_inertia[0]",
    );
    invalid(
        &mutate(|s| s.bodies[0].inertial.as_mut().unwrap().inertia_quat = [1.0; 4]),
        "inertia_quat",
    );
    invalid(
        &mutate(|s| s.bodies[0].inertial.as_mut().unwrap().com[0] = f64::INFINITY),
        "com",
    );
}

#[test]
fn unnamed_things_may_share_the_empty_name() {
    let mut s = full_scene();
    s.bodies[1].name.clear();
    s.bodies[2].name.clear();
    s.validate().unwrap();
}

#[test]
fn joint_invariants() {
    invalid(&mutate(|s| s.joints[1].body = BodyId(9)), "joints[1].body");
    invalid(
        &mutate(|s| {
            s.joints[1].kind = JointKind::Hinge {
                axis: [0.0, 2.0, 0.0],
            }
        }),
        "axis",
    );
    invalid(
        &mutate(|s| {
            s.joints[1].kind = JointKind::Hinge {
                axis: [0.0, 0.0, 0.0],
            }
        }),
        "axis",
    );
    invalid(
        &mutate(|s| s.joints[1].range = Some([1.0, -1.0])),
        "joints[1].range",
    );
    invalid(
        &mutate(|s| s.joints[1].range = Some([0.0, 0.0])),
        "joints[1].range",
    );
    invalid(
        &mutate(|s| s.joints[1].range = Some([f64::NAN, 1.0])),
        "joints[1].range",
    );
    invalid(&mutate(|s| s.joints[1].stiffness = -1.0), "stiffness");
    invalid(&mutate(|s| s.joints[1].damping = -1.0), "damping");
    invalid(&mutate(|s| s.joints[1].armature = -1.0), "armature");
    invalid(
        &mutate(|s| s.joints[1].frictionloss = f64::NAN),
        "frictionloss",
    );
    invalid(
        &mutate(|s| s.joints[1].name = "root".into()),
        "joints[1].name",
    );
    // joints in body order
    invalid(&mutate(|s| s.joints.swap(1, 2)), "joints[2].body");
    invalid(&mutate(|s| s.joints.swap(0, 1)), "joints[1].body");
    // a free joint: top level only, anchored at the origin, unlimited, alone
    invalid(
        &mutate(|s| s.joints[1].kind = JointKind::Free),
        "joints[1].kind",
    );
    invalid(
        &mutate(|s| s.joints[0].pos = [0.1, 0.0, 0.0]),
        "joints[0].pos",
    );
    invalid(
        &mutate(|s| s.joints[0].range = Some([0.0, 1.0])),
        "joints[0].range",
    );
    invalid(
        &mutate(|s| {
            s.joints.insert(
                1,
                joint(
                    "extra",
                    0,
                    JointKind::Hinge {
                        axis: [0.0, 0.0, 1.0],
                    },
                ),
            )
        }),
        "joints[1].body",
    );
}

#[test]
fn degree_of_freedom_rules_match_mujocos() {
    // a body cannot have more than 6 dofs: free (6) is already 6; ball (3) + ball (3) = 6 ok
    let mut s = Scene::new();
    s.bodies.push(body("b", None));
    s.joints.push(joint("ball1", 0, JointKind::Ball));
    s.joints.push(joint(
        "slide",
        0,
        JointKind::Slide {
            axis: [1.0, 0.0, 0.0],
        },
    ));
    s.joints.push(joint(
        "slide2",
        0,
        JointKind::Slide {
            axis: [0.0, 1.0, 0.0],
        },
    ));
    s.validate().unwrap(); // ball, then translations: allowed
    // but a ball cannot be followed by a rotation
    let mut r = Scene::new();
    r.bodies.push(body("b", None));
    r.joints.push(joint("ball1", 0, JointKind::Ball));
    r.joints.push(joint(
        "hinge",
        0,
        JointKind::Hinge {
            axis: [0.0, 0.0, 1.0],
        },
    ));
    invalid(&r, "joints[1].kind");
    let mut r = Scene::new();
    r.bodies.push(body("b", None));
    r.joints.push(joint("ball1", 0, JointKind::Ball));
    r.joints.push(joint("ball2", 0, JointKind::Ball));
    invalid(&r, "joints[1].kind");
    // seven: a hinge-free sequence of 7 slides
    let mut seven = Scene::new();
    seven.bodies.push(body("b", None));
    for i in 0..7 {
        seven.joints.push(joint(
            &format!("s{i}"),
            0,
            JointKind::Slide {
                axis: [1.0, 0.0, 0.0],
            },
        ));
    }
    invalid(&seven, "joints[6].body");
    // a ball joint's range is [0, max_angle]
    let mut ball = Scene::new();
    ball.bodies.push(body("b", None));
    let mut j = joint("ball", 0, JointKind::Ball);
    j.range = Some([0.0, 1.0]);
    ball.joints.push(j.clone());
    ball.validate().unwrap();
    ball.joints[0].range = Some([0.5, 1.0]);
    invalid(&ball, "joints[0].range");
}

#[test]
fn geom_invariants() {
    invalid(
        &mutate(|s| s.geoms[1].shape = Shape::Sphere { r: 0.0 }),
        "shape.r",
    );
    invalid(
        &mutate(|s| s.geoms[1].shape = Shape::Sphere { r: -1.0 }),
        "shape.r",
    );
    invalid(
        &mutate(|s| s.geoms[1].shape = Shape::Sphere { r: f64::NAN }),
        "shape.r",
    );
    invalid(
        &mutate(|s| {
            s.geoms[2].shape = Shape::Capsule {
                r: 0.1,
                half_len: 0.0,
            }
        }),
        "half_len",
    );
    invalid(
        &mutate(|s| {
            s.geoms[3].shape = Shape::Box {
                half: [1.0, 0.0, 1.0],
            }
        }),
        "half[1]",
    );
    invalid(
        &mutate(|s| {
            s.geoms[5].shape = Shape::Ellipsoid {
                radii: [1.0, 1.0, -1.0],
            }
        }),
        "radii[2]",
    );
    invalid(
        &mutate(|s| {
            s.geoms[0].shape = Shape::Plane {
                size: [0.0, 0.0, 0.0],
            }
        }),
        "size[2]",
    );
    invalid(
        &mutate(|s| {
            s.geoms[0].shape = Shape::Plane {
                size: [-1.0, 0.0, 1.0],
            }
        }),
        "size[0]",
    );
    invalid(
        &mutate(|s| s.geoms[6].shape = Shape::Mesh { mesh: MeshId(3) }),
        "shape.mesh",
    );
    invalid(
        &mutate(|s| s.geoms[1].body = Some(BodyId(7))),
        "geoms[1].body",
    );
    invalid(
        &mutate(|s| s.geoms[1].material = MaterialId(5)),
        "geoms[1].material",
    );
    invalid(&mutate(|s| s.geoms[1].quat = [0.0; 4]), "geoms[1].quat");
    invalid(&mutate(|s| s.geoms[1].condim = 2), "condim");
    invalid(&mutate(|s| s.geoms[1].friction[1] = -0.1), "friction[1]");
    invalid(&mutate(|s| s.geoms[1].density = -1.0), "density");
    invalid(
        &mutate(|s| s.geoms[1].name = "floor".into()),
        "geoms[1].name",
    );
    // a plane may belong to the world or a body with no joint in its chain, not a moving body
    invalid(
        &mutate(|s| {
            s.geoms[1].shape = Shape::Plane {
                size: [1.0, 1.0, 0.1],
            }
        }),
        "geoms[1].body",
    );
    // condim 1, 3, 4, 6 are all fine
    for c in [1, 3, 4, 6] {
        let mut s = full_scene();
        s.geoms[1].condim = c;
        s.validate().unwrap();
    }
}

#[test]
fn mesh_invariants() {
    invalid(
        &mutate(|s| s.meshes[0].vertices.truncate(3)),
        "meshes[0].vertices",
    );
    invalid(
        &mutate(|s| s.meshes[0].triangles.clear()),
        "meshes[0].triangles",
    );
    invalid(
        &mutate(|s| s.meshes[0].triangles[1] = [0, 1, 4]),
        "meshes[0].triangles[1]",
    );
    invalid(
        &mutate(|s| s.meshes[0].vertices[2][1] = f32::NAN),
        "meshes[0].vertices[2]",
    );
    invalid(
        &mutate(|s| s.meshes[0].vertices[0][0] = f32::INFINITY),
        "meshes[0].vertices[0]",
    );
}

#[test]
fn instance_invariants() {
    // segmentation id 0 is the background
    invalid(
        &mutate(|s| s.instances[0].seg_id = 0),
        "instances[0].seg_id",
    );
    invalid(
        &mutate(|s| s.instances[1].seg_id = 0),
        "instances[1].seg_id",
    );
    invalid(
        &mutate(|s| s.instances[0].body = Some(BodyId(9))),
        "instances[0].body",
    );
    invalid(
        &mutate(|s| s.instances[0].mesh_or_geom = ShapeRef::Geom(GeomId(99))),
        "mesh_or_geom",
    );
    invalid(
        &mutate(|s| s.instances[1].mesh_or_geom = ShapeRef::Mesh(MeshId(4))),
        "mesh_or_geom",
    );
    invalid(
        &mutate(|s| s.instances[0].material = MaterialId(8)),
        "instances[0].material",
    );
    invalid(
        &mutate(|s| s.instances[0].local_quat = [0.0; 4]),
        "local_quat",
    );
    // an instance of a geom must sit on the geom's own body
    invalid(&mutate(|s| s.instances[0].body = None), "instances[0].body");
    invalid(
        &mutate(|s| s.instances[0].body = Some(BodyId(2))),
        "instances[0].body",
    );
}

#[test]
fn camera_invariants() {
    invalid(&mutate(|s| s.cameras[0].fx = 0.0), "fx");
    invalid(&mutate(|s| s.cameras[0].fy = -3.0), "fy");
    invalid(&mutate(|s| s.cameras[0].cx = f32::NAN), "cx");
    invalid(&mutate(|s| s.cameras[0].near = 0.0), "near");
    invalid(&mutate(|s| s.cameras[0].far = s.cameras[0].near), "far");
    invalid(&mutate(|s| s.cameras[0].width = 0), "width");
    invalid(&mutate(|s| s.cameras[0].height = 0), "width");
    invalid(
        &mutate(|s| {
            s.cameras[0].mount = CameraMount::World {
                pos: [0.0; 3],
                quat: [0.0; 4],
            }
        }),
        "mount.quat",
    );
    invalid(
        &mutate(|s| {
            s.cameras[1].mount = CameraMount::Body {
                body: BodyId(9),
                local_pos: [0.0; 3],
                local_quat: [0.0, 0.0, 0.0, 1.0],
            }
        }),
        "mount.body",
    );
    invalid(
        &mutate(|s| s.cameras[1].name = "fixed".into()),
        "cameras[1].name",
    );
}

#[test]
fn actuator_and_tendon_invariants() {
    invalid(
        &mutate(|s| s.actuators[0].joint = JointId(9)),
        "actuators[0].joint",
    );
    // free and ball joints cannot be driven here
    invalid(
        &mutate(|s| s.actuators[0].joint = JointId(0)),
        "actuators[0].joint",
    );
    invalid(
        &mutate(|s| {
            s.actuators[0].kind = ActuatorKind::Motor {
                gear: f64::NAN,
                ctrlrange: None,
            }
        }),
        "gear",
    );
    invalid(
        &mutate(|s| {
            s.actuators[0].kind = ActuatorKind::Motor {
                gear: 1.0,
                ctrlrange: Some([1.0, -1.0]),
            }
        }),
        "ctrlrange",
    );
    invalid(
        &mutate(|s| {
            s.actuators[1].kind = ActuatorKind::Position {
                kp: -1.0,
                gear: 1.0,
                ctrlrange: None,
            }
        }),
        "kp",
    );
    invalid(
        &mutate(|s| {
            s.actuators[1].kind = ActuatorKind::Position {
                kp: 1.0,
                gear: f64::NAN,
                ctrlrange: None,
            }
        }),
        "gear",
    );
    invalid(
        &mutate(|s| s.tendons[0].joints.clear()),
        "tendons[0].joints",
    );
    invalid(
        &mutate(|s| s.tendons[0].joints[1].joint = JointId(0)),
        "joints[1].joint",
    );
    invalid(
        &mutate(|s| s.tendons[0].joints[0].joint = JointId(30)),
        "joints[0].joint",
    );
    invalid(
        &mutate(|s| s.tendons[0].joints[0].coef = f64::INFINITY),
        "coef",
    );
    invalid(
        &mutate(|s| s.tendons[0].range = Some([2.0, -0.3])),
        "tendons[0].range",
    );
    invalid(&mutate(|s| s.tendons[0].damping = -1.0), "damping");
}

#[test]
fn material_defaults_are_the_documented_ones() {
    let m = Material::named("x");
    assert_eq!(m.mechanical.density, 1000.0);
    assert_eq!(m.mechanical.friction, [1.0, 0.005, 0.0001]);
    assert_eq!(m.mechanical.restitution, 0.0);
    assert_eq!(m.thermal.conductivity_w_mk, 0.2);
    assert_eq!(m.thermal.heat_capacity_j_kgk, 1200.0);
    assert_eq!(m.thermal.emissivity, 0.9);
    assert_eq!(m.acoustic.absorption, [0.05, 0.05, 0.05]);
    assert_eq!(m.acoustic.scattering, 0.1);
    assert_eq!(m.acoustic.transmission, 0.0);
    assert_eq!(m.optical.base_colour_linear, [0.5, 0.5, 0.5]);
    assert_eq!(m.optical.roughness, 0.5);
    assert_eq!(m.optical.metallic, 0.0);
    assert_eq!(m.optical.emission_linear, [0.0, 0.0, 0.0]);
    assert_eq!(m.optical.checker, None);
    m.validate("m").unwrap();
    // a material with only a name in JSON is the default
    let from_json: Material = serde_json::from_str(r#"{"name": "x"}"#).unwrap();
    assert_eq!(from_json, m);
    // a part may be partial
    let partial: Material =
        serde_json::from_str(r#"{"name": "x", "optical": {"metallic": 1.0}}"#).unwrap();
    assert_eq!(partial.optical.metallic, 1.0);
    assert_eq!(partial.optical.roughness, 0.5);
}

#[test]
fn material_physical_ranges_are_enforced() {
    type Mutation = fn(&mut Material);
    let bad: Vec<(&str, Mutation)> = vec![
        ("mechanical.density", |m| m.mechanical.density = 0.0),
        ("mechanical.density", |m| m.mechanical.density = f32::NAN),
        ("mechanical.friction[2]", |m| {
            m.mechanical.friction[2] = -1.0
        }),
        ("mechanical.restitution", |m| m.mechanical.restitution = 1.5),
        ("mechanical.restitution", |m| {
            m.mechanical.restitution = -0.1
        }),
        ("thermal.conductivity_w_mk", |m| {
            m.thermal.conductivity_w_mk = 0.0
        }),
        ("thermal.heat_capacity_j_kgk", |m| {
            m.thermal.heat_capacity_j_kgk = -5.0
        }),
        ("thermal.emissivity", |m| m.thermal.emissivity = 1.01),
        ("acoustic.absorption[1]", |m| m.acoustic.absorption[1] = 1.1),
        ("acoustic.scattering", |m| m.acoustic.scattering = -0.5),
        ("acoustic.transmission", |m| m.acoustic.transmission = 2.0),
        ("acoustic.absorption[0]", |m| {
            m.acoustic.absorption = [0.7, 0.1, 0.1];
            m.acoustic.transmission = 0.5;
        }),
        ("optical.base_colour_linear[0]", |m| {
            m.optical.base_colour_linear[0] = 1.2
        }),
        ("optical.base_colour_linear[2]", |m| {
            m.optical.base_colour_linear[2] = -0.1
        }),
        ("optical.roughness", |m| m.optical.roughness = 1.5),
        ("optical.metallic", |m| m.optical.metallic = -0.5),
        ("optical.emission_linear[1]", |m| {
            m.optical.emission_linear[1] = -1.0
        }),
        ("optical.checker.cells_per_m", |m| {
            m.optical.checker = Some(Checker {
                cells_per_m: 0.0,
                dark_multiplier: 0.5,
            })
        }),
        ("optical.checker.dark_multiplier", |m| {
            m.optical.checker = Some(Checker {
                cells_per_m: 2.0,
                dark_multiplier: 1.5,
            })
        }),
    ];
    for (path, f) in bad {
        let mut s = full_scene();
        f(&mut s.materials[0]);
        invalid(&s, &format!("materials[0].{path}"));
    }
    // emission above 1 is allowed (linear radiance)
    let mut s = full_scene();
    s.materials[0].optical.emission_linear = [10.0, 0.0, 0.0];
    s.validate().unwrap();
    s.materials[1].name = "steel".into();
    invalid(&s, "materials[1].name");
}

#[test]
fn srgb_is_converted_with_the_standard_eotf() {
    // IEC 61966-2-1. The expected values are computed with an independent
    // evaluation of the standard's formula in Python (double precision).
    let close = |a: f32, b: f64| assert!((f64::from(a) - b).abs() < 1e-6, "{a} vs {b}");
    close(srgb_to_linear(0.0), 0.0);
    close(srgb_to_linear(1.0), 1.0);
    close(srgb_to_linear(0.5), 0.21404114048223255);
    close(srgb_to_linear(0.8), 0.6038273388553378);
    close(srgb_to_linear(0.2), 0.033104766570885055);
    close(srgb_to_linear(0.6), 0.31854677812509186);
    // the linear segment, at its edge
    close(srgb_to_linear(0.04045), 0.0031308049535603713);
    close(srgb_to_linear(0.02), 0.02 / 12.92);
    // monotone
    let mut last = -1.0f32;
    for i in 0..=100 {
        let v = srgb_to_linear(i as f32 / 100.0);
        assert!(v > last);
        last = v;
    }
}
