//! `Model::compile` never ignores part of a scene silently: it lists what this phase
//! does not model (geom pairs that would need MuJoCo's convex path, a fixed tendon's
//! spring, damper and armature, a request for the PGS solver, and every
//! `Scene::unsupported` record).
//!
//! Joint limits, joint and tendon friction loss, the limits of fixed tendons and (since
//! phase 1c-ii) contacts between planes, spheres, capsules, cylinders (against planes and
//! spheres) and boxes are modelled, so they are not listed any more. What stays listed is a
//! geom pair with an ellipsoid or a mesh, and capsule-cylinder, cylinder-cylinder and
//! cylinder-box: `NotModelled::Collision`, one entry per type pair. The humanoid and the 1b
//! zoo list nothing of the kind, and the double pendulum lists nothing at all.

mod common;

use common::*;
use sim_physics::{GeomType, Model, NotModelled};
use sim_scene::{
    Body, BodyId, Geom, Inertial, Joint, JointId, JointKind, Material, MaterialId, Scene, Shape,
    Solver, Tendon, TendonJoint,
};

#[test]
fn the_humanoid_lists_what_the_importer_recorded_and_no_collision() {
    let scene = scene_of(Which::Humanoid);
    let (m, list) = Model::<f64>::compile(&scene).unwrap();
    for n in &list {
        println!("humanoid NotModelled: {n}");
    }

    // the humanoid has ranged joints and two limited tendons, and none of them is
    // listed: limits are simulated
    assert!(scene.joints.iter().any(|j| j.range.is_some()));
    assert_eq!(scene.tendons.len(), 2);
    assert!(scene.tendons.iter().all(|t| t.range.is_some()));
    assert!(!list.iter().any(|n| matches!(
        n,
        NotModelled::TendonPassive { .. } | NotModelled::PgsSolver
    )));

    // its pairs are plane-capsule, plane-sphere, capsule-capsule, sphere-capsule and
    // sphere-sphere: all ported, so no `Collision` entry; they are in the candidate list
    assert!(
        !list
            .iter()
            .any(|n| matches!(n, NotModelled::Collision { .. })),
        "{list:?}"
    );
    assert!(m.candidates.len() > 20, "{}", m.candidates.len());
    println!(
        "humanoid contacts: {} candidate geom pairs, ncon_max {}, nefc_max {}",
        m.candidates.len(),
        m.ncon_max,
        m.nefc_max
    );

    // every unsupported record of the scene, unchanged and in order
    let unsupported: Vec<(&str, &str, u32, &str)> = list
        .iter()
        .filter_map(|n| match n {
            NotModelled::Unsupported {
                path,
                item,
                line,
                reason,
            } => Some((path.as_str(), item.as_str(), *line, reason.as_str())),
            _ => None,
        })
        .collect();
    let expected: Vec<(&str, &str, u32, &str)> = scene
        .unsupported
        .iter()
        .map(|u| (u.path.as_str(), u.item.as_str(), u.line, u.reason.as_str()))
        .collect();
    assert_eq!(unsupported, expected);
    assert!(!expected.is_empty());
    // the tendons are no longer in the scene's unsupported list for merely existing,
    // the joints' solver parameters are read, not recorded, and (phase 1c-ii) neither are
    // the geoms' contact parameters or the exclusions
    assert!(!expected.iter().any(|(path, ..)| path.starts_with("tendon")));
    assert!(
        !expected
            .iter()
            .any(|(path, item, ..)| path.contains("joint") && *item == "@solimplimit"),
        "{expected:?}"
    );
    assert!(
        !expected
            .iter()
            .any(|(path, item, ..)| path.contains("exclude")
                || *item == "@solref"
                || *item == "@solimp"),
        "{expected:?}"
    );

    // and nothing else is in the list
    assert_eq!(list.len(), expected.len());
}

#[test]
fn the_zoo_lists_nothing() {
    let (m, list) = Model::<f64>::compile(&scene_of(Which::Zoo)).unwrap();
    assert!(list.is_empty(), "{list:?}");
    // its geoms (boxes, capsules, a sphere) do make candidate pairs
    assert!(!m.candidates.is_empty());
}

#[test]
fn the_double_pendulum_lists_nothing() {
    // the two spheres are parent and child, which MuJoCo's collision filter skips
    for file in ["double_pendulum.xml", "double_pendulum_euler.xml"] {
        let (m, list) = Model::<f64>::compile(&load_scene(&fixtures().join(file))).unwrap();
        assert!(list.is_empty(), "{file}: {list:?}");
        assert!(m.candidates.is_empty(), "{file}");
    }
}

// ---- small scenes that isolate each kind of entry

fn body(name: &str, parent: Option<u32>, with_joint_mass: bool) -> Body {
    Body {
        name: name.into(),
        parent: parent.map(BodyId),
        pos: [0.0; 3],
        quat: [0.0, 0.0, 0.0, 1.0],
        inertial: with_joint_mass.then_some(Inertial {
            mass_kg: 1.0,
            com: [0.0; 3],
            diag_inertia: [0.1, 0.1, 0.1],
            inertia_quat: [0.0, 0.0, 0.0, 1.0],
        }),
    }
}

fn hinge(name: &str, b: u32) -> Joint {
    Joint {
        name: name.into(),
        body: BodyId(b),
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
    }
}

fn geom(name: &str, b: Option<u32>, shape: Shape, contype: u32, conaffinity: u32) -> Geom {
    Geom {
        name: name.into(),
        body: b.map(BodyId),
        shape,
        pos: [0.0; 3],
        quat: [0.0, 0.0, 0.0, 1.0],
        material: MaterialId(0),
        contype,
        conaffinity,
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

fn sphere(name: &str, b: Option<u32>, contype: u32, conaffinity: u32) -> Geom {
    geom(name, b, Shape::Sphere { r: 0.1 }, contype, conaffinity)
}

fn scene_with(bodies: Vec<Body>, joints: Vec<Joint>, geoms: Vec<Geom>) -> Scene {
    let mut s = Scene::new();
    s.materials.push(Material::named("m"));
    s.bodies = bodies;
    s.joints = joints;
    s.geoms = geoms;
    s
}

fn list_of(s: &Scene) -> Vec<NotModelled> {
    Model::<f64>::compile(s).unwrap().1
}

fn tendon(stiffness: f64, damping: f64, armature: f64) -> Tendon {
    Tendon {
        name: "t".into(),
        joints: vec![TendonJoint {
            joint: JointId(0),
            coef: 1.0,
        }],
        range: Some([-1.0, 1.0]),
        stiffness,
        damping,
        armature,
        frictionloss: 0.5,
        solref_limit: sim_scene::DEFAULT_SOLREF,
        solimp_limit: sim_scene::DEFAULT_SOLIMP,
        solref_friction: sim_scene::DEFAULT_SOLREF,
        solimp_friction: sim_scene::DEFAULT_SOLIMP,
        margin: 0.0,
    }
}

#[test]
fn friction_loss_limits_and_limited_tendons_are_modelled_and_not_listed() {
    let mut s = scene_with(vec![body("a", None, true)], vec![hinge("h", 0)], vec![]);
    assert!(list_of(&s).is_empty());
    s.joints[0].frictionloss = 0.25;
    s.joints[0].range = Some([-1.0, 2.0]);
    s.joints[0].margin = 0.1;
    s.tendons.push(tendon(0.0, 0.0, 0.0));
    assert_eq!(
        list_of(&s),
        vec![],
        "nothing is ignored, so nothing is listed"
    );
    // and the model carries them
    let (m, _) = Model::<f64>::compile(&s).unwrap();
    assert_eq!(m.dof_frictionloss, vec![0.25]);
    assert_eq!(m.jnt_range, vec![-1.0, 2.0]);
    assert_eq!(m.ntendon, 1);
    assert_eq!(m.tendon_frictionloss, vec![0.5]);
    // 2 limit rows of the joint, 1 friction row, 2 limit rows and 1 friction row of the tendon
    assert_eq!(m.nefc_max, 1 + 2 + 1 + 2);
}

#[test]
fn a_fixed_tendons_spring_damper_and_armature_are_listed() {
    let mut s = scene_with(vec![body("a", None, true)], vec![hinge("h", 0)], vec![]);
    s.tendons.push(tendon(3.0, 0.0, 0.0));
    let list = list_of(&s);
    assert_eq!(
        list,
        vec![NotModelled::TendonPassive {
            tendon: 0,
            stiffness: 3.0,
            damping: 0.0,
            armature: 0.0
        }]
    );
    assert!(list[0].to_string().contains("tendon 0"));
    // each of the three terms triggers the entry
    for (k, d, a) in [(0.0, 0.5, 0.0), (0.0, 0.0, 0.1)] {
        let mut s = scene_with(vec![body("a", None, true)], vec![hinge("h", 0)], vec![]);
        s.tendons.push(tendon(k, d, a));
        assert_eq!(list_of(&s).len(), 1, "damping {d}, armature {a}");
    }
}

#[test]
fn a_request_for_the_pgs_solver_is_listed_and_newton_runs() {
    let mut s = scene_with(vec![body("a", None, true)], vec![hinge("h", 0)], vec![]);
    s.options.solver = Solver::Pgs;
    let (m, list) = Model::<f64>::compile(&s).unwrap();
    assert_eq!(list, vec![NotModelled::PgsSolver]);
    assert!(list[0].to_string().contains("PGS"));
    assert_eq!(m.opt.solver, sim_physics::PrimalSolver::Newton);
    // Newton and CG are modelled and not listed
    for (solver, expect) in [
        (Solver::Newton, sim_physics::PrimalSolver::Newton),
        (Solver::Cg, sim_physics::PrimalSolver::Cg),
    ] {
        s.options.solver = solver;
        let (m, list) = Model::<f64>::compile(&s).unwrap();
        assert!(list.is_empty(), "{list:?}");
        assert_eq!(m.opt.solver, expect);
    }
}

/// The candidate pairs of a compiled scene as `(g1, g2)`.
fn pairs_of(s: &Scene) -> Vec<(usize, usize)> {
    let (m, _) = Model::<f64>::compile(s).unwrap();
    m.candidates.iter().map(|c| (c.g1, c.g2)).collect()
}

#[test]
fn contacts_follow_mujocos_collision_filter() {
    // two free-standing moving bodies with collidable geoms: one candidate pair (a
    // sphere-sphere pair is a ported collider, so it is a candidate and not listed)
    let two = |ca: u32, cb: u32| {
        scene_with(
            vec![body("a", None, true), body("b", None, true)],
            vec![hinge("ha", 0), hinge("hb", 1)],
            vec![sphere("ga", Some(0), ca, ca), sphere("gb", Some(1), cb, cb)],
        )
    };
    assert_eq!(pairs_of(&two(1, 1)), vec![(0, 1)]);
    assert!(list_of(&two(1, 1)).is_empty());
    // contype/conaffinity bitmasks that do not overlap: no pair
    assert!(pairs_of(&two(1, 2)).is_empty());
    // a geom that collides with nothing
    assert!(pairs_of(&two(0, 1)).is_empty());

    // parent and child (both moving): filtered, as MuJoCo's weldparent filter does
    let chain = scene_with(
        vec![body("a", None, true), body("b", Some(0), true)],
        vec![hinge("ha", 0), hinge("hb", 1)],
        vec![sphere("ga", Some(0), 1, 1), sphere("gb", Some(1), 1, 1)],
    );
    assert!(pairs_of(&chain).is_empty());
    // and with the parent filter disabled they collide
    let (mut m, _) = Model::<f64>::compile(&chain).unwrap();
    m.disable.filterparent = true;
    m.rebuild_contact_pairs();
    assert_eq!(m.candidates.len(), 1);

    // a geom of a static body against a moving one: a pair; two static ones: not
    let with_floor = scene_with(
        vec![body("a", None, true)],
        vec![hinge("ha", 0)],
        vec![sphere("floor", None, 1, 1), sphere("ga", Some(0), 1, 1)],
    );
    assert_eq!(pairs_of(&with_floor).len(), 1);
    let statics = scene_with(
        vec![body("s", None, true)],
        vec![],
        vec![sphere("floor", None, 1, 1), sphere("gs", Some(0), 1, 1)],
    );
    assert!(pairs_of(&statics).is_empty());

    // two geoms on one body never collide with each other
    let same = scene_with(
        vec![body("a", None, true)],
        vec![hinge("ha", 0)],
        vec![sphere("g1", Some(0), 1, 1), sphere("g2", Some(0), 1, 1)],
    );
    assert!(pairs_of(&same).is_empty());

    // an excluded body pair has no candidate
    let mut excluded = two(1, 1);
    excluded.contact_excludes.push(sim_scene::ContactExclude {
        name: String::new(),
        body1: Some(BodyId(0)),
        body2: Some(BodyId(1)),
    });
    assert!(pairs_of(&excluded).is_empty());
}

#[test]
fn a_geom_pair_with_no_collider_is_listed_by_type_pair_and_makes_no_contact() {
    // an ellipsoid and a cylinder that each meet a box (but not each other: the bit groups
    // keep them apart): ellipsoid-box and cylinder-box have no collider in this phase
    let ellipsoid = Shape::Ellipsoid {
        radii: [0.1, 0.2, 0.3],
    };
    let cylinder = Shape::Cylinder {
        r: 0.1,
        half_len: 0.2,
    };
    let boxx = Shape::Box {
        half: [0.1, 0.1, 0.1],
    };
    let s = scene_with(
        vec![
            body("e", None, true),
            body("x", None, true),
            body("c", None, true),
        ],
        vec![hinge("he", 0), hinge("hx", 1), hinge("hc", 2)],
        vec![
            geom("ge", Some(0), ellipsoid, 1, 1),
            geom("gx", Some(1), boxx, 3, 3),
            geom("gc", Some(2), cylinder, 2, 2),
        ],
    );
    let (m, list) = Model::<f64>::compile(&s).unwrap();
    assert_eq!(
        list,
        vec![
            NotModelled::Collision {
                type1: GeomType::Ellipsoid,
                type2: GeomType::Box,
                pairs: 1,
                first_pair: (0, 1),
            },
            NotModelled::Collision {
                type1: GeomType::Cylinder,
                type2: GeomType::Box,
                pairs: 1,
                first_pair: (2, 1),
            },
        ]
    );
    assert!(list[0].to_string().contains("Ellipsoid-Box"));
    // neither pair is a candidate: they make no contact (and need no slots or rows)
    assert!(m.candidates.is_empty());
    assert_eq!((m.ncon_max, m.nefc_max), (0, 0));
}

#[test]
fn plane_plane_is_dropped_as_mujoco_drops_it_and_ellipsoid_and_mesh_planes_are_listed() {
    // two planes never collide (MuJoCo's table has no function for them): not even listed
    let plane = |name: &str, b: Option<u32>| {
        geom(
            name,
            b,
            Shape::Plane {
                size: [1.0, 1.0, 0.1],
            },
            1,
            1,
        )
    };
    // WHY the pair is dropped (the review found this test passing for a reason it did not
    // state). Body `a` has no joint, so `filterBodyPair` drops its pairs with the world (both
    // weld to the world, both without dofs) before the collision table is consulted. The
    // reviewer's remedy, a joint on `a`, is refused by the scene: a plane can only belong to the
    // world or to a body with no joint in its chain (checked below). So in an importable scene
    // two planes are ALWAYS static together, always dropped by the weld rule, and the table's
    // NULL entry for plane-plane (MuJoCo drops it at `driver.c:599-601`) is unreachable; this
    // test shows the drop, the refusal that makes the NULL entry unreachable, and, as the
    // positive control, that the same body with a sphere in place of the second plane and a joint
    // IS a candidate (so the empty list above is the filter's doing, not an empty scene).
    let s = scene_with(
        vec![body("a", None, true)],
        vec![],
        vec![plane("p0", None), plane("p1", Some(0))],
    );
    let (m, list) = Model::<f64>::compile(&s).unwrap();
    assert!(list.is_empty());
    assert!(m.candidates.is_empty());
    assert_eq!((m.ncon_max, m.nefc_max), (0, 0));
    // a plane on a body that has a joint is refused by the scene: no importable scene can make
    // the pair pass the body filter
    let jointed = scene_with(
        vec![body("a", None, true)],
        vec![hinge("h", 0)],
        vec![plane("p0", None), plane("p1", Some(0))],
    );
    let err = Model::<f64>::compile(&jointed).unwrap_err().to_string();
    assert!(
        err.contains("a plane can only belong to the world or a body with no joint"),
        "{err}"
    );
    // the positive control: a plane and a sphere on the jointed body are a plane-sphere candidate
    let control = scene_with(
        vec![body("a", None, true)],
        vec![hinge("h", 0)],
        vec![
            plane("p0", None),
            geom("s", Some(0), Shape::Sphere { r: 0.1 }, 1, 1),
        ],
    );
    let (mc, list) = Model::<f64>::compile(&control).unwrap();
    assert!(list.is_empty());
    assert_eq!(mc.candidates.len(), 1);
    assert_eq!(
        mc.candidates[0].collider,
        sim_physics::Collider::PlaneSphere
    );
    // a plane and an ellipsoid on a moving body: a convex-path pair, listed
    let s = scene_with(
        vec![body("a", None, true)],
        vec![hinge("h", 0)],
        vec![
            plane("p0", None),
            geom(
                "e",
                Some(0),
                Shape::Ellipsoid {
                    radii: [0.1, 0.1, 0.1],
                },
                1,
                1,
            ),
        ],
    );
    assert_eq!(
        list_of(&s),
        vec![NotModelled::Collision {
            type1: GeomType::Plane,
            type2: GeomType::Ellipsoid,
            pairs: 1,
            first_pair: (0, 1),
        }]
    );
}
