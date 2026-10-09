//! Physics checks of the contacts (phase 1c-ii), each against a prediction derived on paper from
//! MuJoCo's soft contact model and from the source's row formulas, not from MuJoCo's output.
//!
//! **The model.** A contact row with Jacobian `J`, reference acceleration `aref = -B v - K I
//! (pos - margin)` and regularisation `R = (1 - I) A / I` (`A` the row's approximate inverse
//! inertia, `D = 1 / R`) pulls the acceleration to the minimum of `0.5 (a - a_s)' M (a - a_s) +
//! s(J a - aref)`, where an active row costs `0.5 D (J a - aref)^2`. The standard `solref =
//! (tc, zeta)` gives `K = 1 / (d_max^2 tc^2 zeta^2)` with `d_max = solimp[1]`, and `tc` is raised
//! to `2 h` (`refsafe`); the impedance `I` runs from `d_min` to `d_max` over the penetration
//! `p = includemargin - dist` (a power law with the midpoint and power of `solimp`).
//!
//! **A contact at rest.** At rest (`v = 0`, `a = 0`) every row has `pos = dist` and `aref = K I(p)
//! p`, so its force is `D K I p` and the contact force is built from the rows. With `A` the sum of
//! `body_invweight0[2 b]` (translation) over the contact's two bodies (the world has 0):
//!
//! - frictionless (`condim 1`): one row, `F = K I(p)^2 p / ((1 - I(p)) A)`;
//! - elliptic: the normal row has `R = (1 - I) A / I`, and at rest the friction rows have zero
//!   `jar` while the normal one is negative, so the cone is in its BOTTOM zone, every row is
//!   quadratic, and `F` is the frictionless formula: the friction `f0` and `impratio` do not
//!   enter;
//! - pyramidal: `2 (dim - 1)` rows `J_n +- f_k J_k`, each with `diagApprox = A (1 + f0^2)` and,
//!   after the impratio pass, `R = 2 mu^2 (1 - I) A (1 + f0^2) / I` for every row, `mu^2 =
//!   f0^2 / impratio`; the tangential parts cancel and the normal force is the sum of the rows,
//!   `F = 2 (dim - 1) K I(p)^2 p / ((1 - I(p)) 2 mu^2 A (1 + f0^2))`. A pyramidal contact is
//!   therefore stiffer or softer than a frictionless one by a factor that depends on `f0` and on
//!   `impratio`.
//!
//! Each prediction is solved for `p` by bisection (`I(p)` makes it nonlinear), the bodies are
//! put at those penetrations at rest, and the solver must find `qacc = 0` with the predicted
//! forces; the dynamic runs start away from equilibrium and must settle there.
//!
//! **Sliding** (elliptic cone, middle zone): `|f_t| = f0 f_n` exactly (`force_j = -force_0 / T U_j
//! f_{j-1}`), so a sliding sphere decelerates at `f0 g` once its vertical motion has settled,
//! until it rolls: the rolling onset of a solid sphere is `t = 2 v0 / (7 f0 g)`. The pyramidal cone
//! carries spring load on the other pair of edges, so its sliding friction is below `f0 f_n` and is
//! reported, not predicted. **Spinning**: with `condim 3` nothing resists the spin about the
//! normal; with `condim 4` (elliptic, middle zone) the torsional friction decelerates it at
//! `friction[2] N / I_zz`.

use sim_physics::{
    ConstraintState, Data, Model, PrimalSolver, contact_force, scene_body_to_internal, step,
};
use sim_scene::mjcf;

mod common;
use common::contacts::*;
use common::{f, farr};

const G: f64 = 9.81;
/// The solver tolerance of these checks: they are about the soft-contact model, not about how
/// well a solver converges.
const SOLVE_TO: f64 = 1e-14;

fn compile(xml: &str) -> Model<f64> {
    let scene = mjcf::load(xml, std::env::temp_dir()).expect("imports");
    let (m, nm) = Model::<f64>::compile(&scene).expect("compiles");
    assert!(nm.is_empty(), "{nm:?}");
    m
}

/// MuJoCo's impedance as its documentation gives it (independent of the engine's code): the
/// value at distance `x = |pos - margin|` from the margin.
fn impedance(solimp: [f64; 5], pos_minus_margin: f64) -> f64 {
    let [d0, d1, width, mid, power] = solimp;
    if d0 == d1 {
        return 0.5 * (d0 + d1);
    }
    let x = (pos_minus_margin.abs() / width).min(1.0);
    if x >= 1.0 {
        return d1;
    }
    let y = if x <= mid {
        x.powf(power) / mid.powf(power - 1.0)
    } else {
        1.0 - (1.0 - x).powf(power) / (1.0 - mid).powf(power - 1.0)
    };
    d0 + y * (d1 - d0)
}

/// The kinds of contact the prediction knows.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Frictionless,
    /// Pyramidal with this `dim` (3: four rows).
    Pyramidal(usize),
    Elliptic,
}

/// Everything the prediction of a contact's normal force needs.
#[derive(Clone, Copy)]
struct Soft {
    solref: [f64; 2],
    solimp: [f64; 5],
    timestep: f64,
    /// The sum of the translational inverse weights of the two bodies.
    a: f64,
    kind: Kind,
    f0: f64,
    impratio: f64,
}

impl Soft {
    /// `K = 1 / (d_max^2 tc^2 zeta^2)` with refsafe's `tc >= 2 h`.
    fn stiffness(&self) -> f64 {
        let tc = self.solref[0].max(2.0 * self.timestep);
        1.0 / (self.solimp[1] * self.solimp[1] * tc * tc * self.solref[1] * self.solref[1])
    }

    /// The normal force of one contact at rest with penetration `p`.
    fn force(&self, p: f64) -> f64 {
        let (k, i) = (self.stiffness(), impedance(self.solimp, p));
        let spring = k * i * i * p / ((1.0 - i) * self.a);
        match self.kind {
            Kind::Frictionless | Kind::Elliptic => spring,
            Kind::Pyramidal(dim) => {
                let mu2 = self.f0 * self.f0 / self.impratio;
                // 2 (dim - 1) rows, each D K I p with R = 2 mu^2 (1 - I) A (1 + f0^2) / I
                (2 * (dim - 1)) as f64 * spring / (2.0 * mu2 * (1.0 + self.f0 * self.f0))
            }
        }
    }

    /// The penetration at which one contact carries `target` (bisection: the force grows with
    /// `p`).
    fn penetration(&self, target: f64) -> f64 {
        let g = |p: f64| self.force(p) - target;
        let (mut lo, mut hi) = (0.0f64, 0.5f64);
        assert!(g(lo) < 0.0 && g(hi) > 0.0, "the bracket must hold the root");
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if g(mid) > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        0.5 * (lo + hi)
    }
}

fn tighten(m: &Model<f64>, solver: PrimalSolver) -> Model<f64> {
    let mut mm = m.clone();
    mm.opt.solver = solver;
    mm.opt.tolerance = SOLVE_TO;
    mm
}

// ---------------------------------------------------------------------------
// (a) a sphere resting on a plane
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct SphereCase {
    label: &'static str,
    condim: u32,
    cone: &'static str,
    f0: f64,
    impratio: f64,
}

const SPHERE_CASES: [SphereCase; 9] = [
    SphereCase {
        label: "condim 1",
        condim: 1,
        cone: "pyramidal",
        f0: 0.5,
        impratio: 1.0,
    },
    SphereCase {
        label: "condim 1, elliptic",
        condim: 1,
        cone: "elliptic",
        f0: 0.5,
        impratio: 1.0,
    },
    SphereCase {
        label: "pyramidal f0 0.5 impratio 1",
        condim: 3,
        cone: "pyramidal",
        f0: 0.5,
        impratio: 1.0,
    },
    SphereCase {
        label: "pyramidal f0 1 impratio 1",
        condim: 3,
        cone: "pyramidal",
        f0: 1.0,
        impratio: 1.0,
    },
    SphereCase {
        label: "pyramidal f0 0.5 impratio 4",
        condim: 3,
        cone: "pyramidal",
        f0: 0.5,
        impratio: 4.0,
    },
    SphereCase {
        label: "pyramidal f0 1 impratio 4",
        condim: 3,
        cone: "pyramidal",
        f0: 1.0,
        impratio: 4.0,
    },
    SphereCase {
        label: "elliptic f0 0.5 impratio 1",
        condim: 3,
        cone: "elliptic",
        f0: 0.5,
        impratio: 1.0,
    },
    SphereCase {
        label: "elliptic f0 1 impratio 4",
        condim: 3,
        cone: "elliptic",
        f0: 1.0,
        impratio: 4.0,
    },
    SphereCase {
        label: "elliptic f0 0.5 impratio 4",
        condim: 3,
        cone: "elliptic",
        f0: 0.5,
        impratio: 4.0,
    },
];

fn sphere_xml(c: &SphereCase, z: f64) -> String {
    format!(
        r#"<mujoco><option timestep="0.002" cone="{}" impratio="{}"/>
           <worldbody>
             <geom name="floor" type="plane" size="5 5 .1" condim="{}" friction="{} 0.005 0.0001"/>
             <body name="ball" pos="0 0 {z}"><freejoint/>
               <geom name="ball_geom" type="sphere" size=".1" mass="1" condim="{}" friction="{} 0.005 0.0001"/>
             </body>
           </worldbody></mujoco>"#,
        c.cone, c.impratio, c.condim, c.f0, c.condim, c.f0
    )
}

fn soft_of_sphere(m: &Model<f64>, c: &SphereCase) -> Soft {
    Soft {
        solref: [0.02, 1.0],
        solimp: [0.9, 0.95, 0.001, 0.5, 2.0],
        timestep: m.timestep,
        a: m.body_invweight0[2 * scene_body_to_internal(0)],
        kind: match (c.condim, c.cone) {
            (1, _) => Kind::Frictionless,
            (3, "elliptic") => Kind::Elliptic,
            (3, _) => Kind::Pyramidal(3),
            _ => unreachable!(),
        },
        f0: c.f0,
        impratio: c.impratio,
    }
}

/// The predicted rest penetration of every case, and the facts that make this a test of the
/// impratio and friction dependence: the elliptic and frictionless contacts agree, the
/// pyramidal ones depend on `f0` and on impratio.
#[test]
fn a_sphere_resting_on_a_plane_sits_at_the_predicted_penetration() {
    let mut predicted = Vec::new();
    for c in &SPHERE_CASES {
        let m = compile(&sphere_xml(c, 0.1));
        let soft = soft_of_sphere(&m, c);
        // the model's own inverse weight is a free sphere's 1 / m, derived here
        assert!((soft.a - 1.0).abs() < 1e-12, "{}: A = {}", c.label, soft.a);
        let mg = G;
        let p = soft.penetration(mg);
        predicted.push((c.label, p));

        for solver in [PrimalSolver::Newton, PrimalSolver::Cg] {
            let mm = tighten(&m, solver);
            let mut d = Data::new(&mm);
            d.qpos[2] = 0.1 - p; // dist = -p, includemargin = 0
            sim_physics::forward(&mm, &mut d);
            assert_eq!(d.warning_collision_overflow, 0);
            assert_eq!(d.ncon, 1, "{}", c.label);
            // every row has pos = dist, and the penetration is includemargin - dist
            let dist = d.contact_dist[0];
            let pen = d.contact_includemargin[0] - dist;
            for r in 0..d.nefc {
                assert_eq!(
                    d.efc_pos[r],
                    if r == 0 || soft.kind != Kind::Elliptic {
                        dist
                    } else {
                        0.0
                    }
                );
            }
            assert!((pen - p).abs() <= 1e-14, "{}: p {pen:e} vs {p:e}", c.label);
            let force = contact_force(&mm, &d, 0)[0];
            let a_max = d.qacc.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            let from_formula = soft.force(pen);
            println!(
                "MEASURED f64 sphere at rest ({}, {solver:?}): predicted penetration {p:e} m; normal force {force:e} N (m g {mg:e}, formula at the measured penetration {from_formula:e}); max |qacc| {a_max:e}",
                c.label
            );
            assert!(
                (force - mg).abs() <= 1e-9 * mg,
                "{} {solver:?}: force {force}",
                c.label
            );
            assert!(
                (from_formula - force).abs() <= 1e-9 * force,
                "{} {solver:?}: formula {from_formula} vs force {force}",
                c.label
            );
            // the tangential parts vanish by symmetry
            let f6 = contact_force(&mm, &d, 0);
            assert!(f6[1].abs() <= 1e-9 * mg && f6[2].abs() <= 1e-9 * mg);
            assert!(a_max <= 1e-8, "{} {solver:?}: |qacc| {a_max:e}", c.label);
        }
    }
    let p_of = |label: &str| predicted.iter().find(|(l, _)| *l == label).unwrap().1;
    // elliptic contacts do not feel f0 or impratio, and are the frictionless contact. Checked on
    // the ENGINE (the review: comparing the test's own predictions compares one match arm with
    // itself): each elliptic case, placed at the FRICTIONLESS penetration, carries m g with a
    // vanishing acceleration, whatever its f0 and impratio
    let frictionless = p_of("condim 1");
    for label in [
        "condim 1, elliptic",
        "elliptic f0 0.5 impratio 1",
        "elliptic f0 1 impratio 4",
        "elliptic f0 0.5 impratio 4",
    ] {
        let case = SPHERE_CASES
            .iter()
            .find(|c| c.label == label)
            .unwrap_or_else(|| panic!("no case {label}"));
        let m = compile(&sphere_xml(case, 0.1));
        for solver in [PrimalSolver::Newton, PrimalSolver::Cg] {
            let mm = tighten(&m, solver);
            let mut d = Data::new(&mm);
            d.qpos[2] = 0.1 - frictionless;
            sim_physics::forward(&mm, &mut d);
            assert_eq!(d.warning_collision_overflow, 0);
            let force = contact_force(&mm, &d, 0)[0];
            let a_max = d.qacc.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            assert!(
                (force - G).abs() <= 1e-9 * G && a_max <= 1e-8,
                "{label} {solver:?}: at the frictionless penetration the normal force is {force:e} (m g {G:e}), max |qacc| {a_max:e}"
            );
        }
        // and the prediction agrees (a statement about the formulas, not the engine)
        assert_eq!(p_of(label), frictionless, "{label}");
    }
    // pyramidal contacts do: the penetration differs from the frictionless one and across f0
    // and impratio, except where the factor 2 (dim - 1) / (2 mu^2 (1 + f0^2)) is exactly 1:
    // f0 = 1 with impratio 1 (2 mu^2 (1 + f0^2) = 4 = 2 (dim - 1)), which then IS the
    // frictionless penetration
    let pyr = [
        p_of("pyramidal f0 0.5 impratio 1"),
        p_of("pyramidal f0 1 impratio 1"),
        p_of("pyramidal f0 0.5 impratio 4"),
        p_of("pyramidal f0 1 impratio 4"),
    ];
    assert!(
        (pyr[1] - frictionless).abs() <= 1e-12 * frictionless,
        "the factor is 1"
    );
    for (i, a) in pyr.iter().enumerate() {
        if i != 1 {
            assert!(
                (a - frictionless).abs() > 0.02 * frictionless,
                "{a:e} vs {frictionless:e}"
            );
        }
        for b in &pyr[i + 1..] {
            assert!((a - b).abs() > 0.02 * a.max(*b), "{a:e} vs {b:e}");
        }
    }
    println!(
        "MEASURED f64 sphere at rest: predicted penetration frictionless/elliptic {frictionless:e} m; pyramidal f0 0.5 impratio 1 {:e}, f0 1 impratio 1 {:e}, f0 0.5 impratio 4 {:e}, f0 1 impratio 4 {:e}",
        pyr[0], pyr[1], pyr[2], pyr[3]
    );
}

/// Dropped from just above the plane, the sphere settles at the predicted penetration.
#[test]
fn a_sphere_dropped_on_a_plane_settles_at_the_predicted_penetration() {
    for c in &SPHERE_CASES {
        let m = compile(&sphere_xml(c, 0.1));
        let soft = soft_of_sphere(&m, c);
        let p = soft.penetration(G);
        for solver in [PrimalSolver::Newton, PrimalSolver::Cg] {
            let mm = tighten(&m, solver);
            let mut d = Data::new(&mm);
            d.qpos[2] = 0.1 + 0.02; // 2 cm above the plane (a short fall, not touching)
            for _ in 0..2000 {
                step(&mm, &mut d);
            }
            let measured = 0.1 - d.qpos[2];
            println!(
                "MEASURED f64 sphere dropped ({}, {solver:?}): predicted rest penetration {p:e} m, measured {measured:e} m after 4 s, |qvel| {:e}",
                c.label,
                d.qvel.iter().fold(0.0f64, |m, x| m.max(x.abs()))
            );
            assert!(
                (measured - p).abs() <= 1e-6 * p,
                "{} {solver:?}: {measured:e} vs {p:e}",
                c.label
            );
            assert!(d.qvel.iter().all(|v| v.abs() < 1e-8));
        }
    }
}

// ---------------------------------------------------------------------------
// (b) a box resting on a plane
// ---------------------------------------------------------------------------

fn box_xml(cone: &str, margin: f64, z: f64) -> String {
    format!(
        r#"<mujoco><option timestep="0.002" cone="{cone}"/>
           <worldbody>
             <geom name="floor" type="plane" size="5 5 .1" friction="0.5 0.005 0.0001"/>
             <body name="block" pos="0 0 {z}"><freejoint/>
               <geom name="block_geom" type="box" size=".15 .1 .05" mass="1" margin="{margin}" friction="0.5 0.005 0.0001"/>
             </body>
           </worldbody></mujoco>"#
    )
}

/// Each corner carries `m g / 4` at the penetration the contact model predicts; with a margin
/// the box rests at `dist = includemargin - p`.
#[test]
fn a_box_resting_on_a_plane_carries_a_quarter_of_its_weight_on_each_corner() {
    for cone in ["pyramidal", "elliptic"] {
        for margin in [0.0, 0.01] {
            let m = compile(&box_xml(cone, margin, 0.05));
            let soft = Soft {
                solref: [0.02, 1.0],
                solimp: [0.9, 0.95, 0.001, 0.5, 2.0],
                timestep: m.timestep,
                a: m.body_invweight0[2],
                kind: if cone == "elliptic" {
                    Kind::Elliptic
                } else {
                    Kind::Pyramidal(3)
                },
                f0: 0.5,
                impratio: 1.0,
            };
            let p = soft.penetration(G / 4.0);
            for solver in [PrimalSolver::Newton, PrimalSolver::Cg] {
                let mm = tighten(&m, solver);
                let mut d = Data::new(&mm);
                d.qpos[2] = 0.05 + margin - p;
                sim_physics::forward(&mm, &mut d);
                assert_eq!(d.warning_collision_overflow, 0);
                assert_eq!(d.ncon, 4, "{cone} margin {margin}: four corners");
                let mut worst = 0.0f64;
                for i in 0..4 {
                    let dist = d.contact_dist[i];
                    assert!(
                        (d.contact_includemargin[i] - dist - p).abs() <= 1e-13,
                        "corner {i}: dist {dist:e}, p {p:e}"
                    );
                    let force = contact_force(&mm, &d, i)[0];
                    worst = worst.max((force - G / 4.0).abs() / (G / 4.0));
                }
                let a_max = d.qacc.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                println!(
                    "MEASURED f64 box at rest ({cone}, margin {margin}, {solver:?}): predicted penetration {p:e} m, rests at dist = includemargin - p = {:e}; worst corner force error {worst:e} (of m g / 4); max |qacc| {a_max:e}",
                    d.contact_dist[0]
                );
                assert!(
                    worst <= 1e-9,
                    "{cone} margin {margin} {solver:?}: {worst:e}"
                );
                assert!(
                    a_max <= 1e-8,
                    "{cone} margin {margin} {solver:?}: {a_max:e}"
                );
            }
            // dropped from a little above, the box settles there
            let mm = tighten(&m, PrimalSolver::Newton);
            let mut d = Data::new(&mm);
            d.qpos[2] = 0.05 + margin + 0.01;
            for _ in 0..2000 {
                step(&mm, &mut d);
            }
            let measured = 0.05 + margin - d.qpos[2];
            println!(
                "MEASURED f64 box dropped ({cone}, margin {margin}): predicted rest penetration {p:e} m, measured {measured:e} m after 4 s"
            );
            assert!(
                (measured - p).abs() <= 1e-6 * p,
                "{cone} margin {margin}: {measured:e}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// (c) a stack of three boxes
// ---------------------------------------------------------------------------

const STACK_HALF_HEIGHT: f64 = 0.05;

fn stack_xml(cone: &str, z: [f64; 3]) -> String {
    format!(
        r#"<mujoco><option timestep="0.002" cone="{cone}"/>
           <worldbody>
             <geom name="floor" type="plane" size="5 5 .1" friction="0.5 0.005 0.0001"/>
             <body name="bottom" pos="0 0 {}"><freejoint/>
               <geom name="bottom_geom" type="box" size=".3 .3 .05" mass="1" friction="0.5 0.005 0.0001"/></body>
             <body name="middle" pos="0 0 {}"><freejoint/>
               <geom name="middle_geom" type="box" size=".25 .25 .05" mass="1" friction="0.5 0.005 0.0001"/></body>
             <body name="top" pos="0 0 {}"><freejoint/>
               <geom name="top_geom" type="box" size=".2 .2 .05" mass="1" friction="0.5 0.005 0.0001"/></body>
           </worldbody></mujoco>"#,
        z[0], z[1], z[2]
    )
}

/// The three interfaces: geom pairs, the weight each carries (in units of `m g`), and `A`.
fn stack_interfaces(m: &Model<f64>) -> [((usize, usize), f64, f64); 3] {
    let tw = |b: usize| m.body_invweight0[2 * b];
    [
        ((0, 1), 3.0, tw(1)),
        ((1, 2), 2.0, tw(1) + tw(2)),
        ((2, 3), 1.0, tw(2) + tw(3)),
    ]
}

/// Interface `k` carries `(3 - k) m g` over its measured number of contacts, with `A =
/// tw(lower) + tw(upper)`; the stack is put at the predicted penetrations and is at rest, and
/// a stack that starts touching settles there.
#[test]
fn a_stack_of_three_boxes_carries_its_weight_at_every_interface() {
    for cone in ["pyramidal", "elliptic"] {
        let m = compile(&stack_xml(cone, [0.05, 0.15, 0.25]));
        let kind = if cone == "elliptic" {
            Kind::Elliptic
        } else {
            Kind::Pyramidal(3)
        };
        let interfaces = stack_interfaces(&m);
        // the number of contacts of each interface, measured on the touching stack
        let counts: Vec<usize> = {
            let mut d = Data::new(&m);
            d.qpos[2] = 0.05 - 1e-4;
            d.qpos[9] = 0.15 - 2e-4;
            d.qpos[16] = 0.25 - 3e-4;
            sim_physics::forward(&m, &mut d);
            assert_eq!(d.warning_collision_overflow, 0, "box-box overflow");
            interfaces
                .iter()
                .map(|(pair, _, _)| {
                    (0..d.ncon)
                        .filter(|&i| (d.contact_geom[2 * i], d.contact_geom[2 * i + 1]) == *pair)
                        .count()
                })
                .collect()
        };
        assert!(counts.iter().all(|&n| n > 0), "{counts:?}");
        let mut pens = [0.0f64; 3];
        for (k, ((_, weights, a), n)) in interfaces.iter().zip(&counts).enumerate() {
            let soft = Soft {
                solref: [0.02, 1.0],
                solimp: [0.9, 0.95, 0.001, 0.5, 2.0],
                timestep: m.timestep,
                a: *a,
                kind,
                f0: 0.5,
                impratio: 1.0,
            };
            pens[k] = soft.penetration(weights * G / *n as f64);
        }
        // the stack at rest: each box rests on the one below by its interface's penetration
        let z0 = STACK_HALF_HEIGHT - pens[0];
        let z1 = z0 + STACK_HALF_HEIGHT + STACK_HALF_HEIGHT - pens[1];
        let z2 = z1 + STACK_HALF_HEIGHT + STACK_HALF_HEIGHT - pens[2];
        for solver in [PrimalSolver::Newton, PrimalSolver::Cg] {
            let mm = tighten(&m, solver);
            let mut d = Data::new(&mm);
            d.qpos[2] = z0;
            d.qpos[9] = z1;
            d.qpos[16] = z2;
            sim_physics::forward(&mm, &mut d);
            assert_eq!(d.warning_collision_overflow, 0);
            let mut worst = 0.0f64;
            for (k, ((pair, weights, _), n)) in interfaces.iter().zip(&counts).enumerate() {
                let idx: Vec<usize> = (0..d.ncon)
                    .filter(|&i| (d.contact_geom[2 * i], d.contact_geom[2 * i + 1]) == *pair)
                    .collect();
                assert_eq!(idx.len(), *n, "interface {k}");
                let target = weights * G / *n as f64;
                for &i in &idx {
                    let force = contact_force(&mm, &d, i)[0];
                    worst = worst.max((force - target).abs() / target);
                    let pen = d.contact_includemargin[i] - d.contact_dist[i];
                    assert!(
                        (pen - pens[k]).abs() <= 1e-12,
                        "interface {k}: {pen:e} vs {:e}",
                        pens[k]
                    );
                }
            }
            let a_max = d.qacc.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            println!(
                "MEASURED f64 stack at rest ({cone}, {solver:?}): contacts per interface {counts:?}, predicted penetrations {:e} {:e} {:e} m; worst contact force error {worst:e} (of its share of (3 - k) m g); max |qacc| {a_max:e}",
                pens[0], pens[1], pens[2]
            );
            assert!(worst <= 1e-9, "{cone} {solver:?}: {worst:e}");
            assert!(a_max <= 1e-8, "{cone} {solver:?}: {a_max:e}");
        }
        // starting at the XML's touching pose, the stack settles at those penetrations
        let mm = tighten(&m, PrimalSolver::Newton);
        let mut d = Data::new(&mm);
        for _ in 0..6000 {
            step(&mm, &mut d);
        }
        for (k, ((pair, _, _), _)) in interfaces.iter().zip(&counts).enumerate() {
            let idx: Vec<usize> = (0..d.ncon)
                .filter(|&i| (d.contact_geom[2 * i], d.contact_geom[2 * i + 1]) == *pair)
                .collect();
            assert!(!idx.is_empty(), "the interface touches");
            // the mean penetration of the interface's contacts: a settled box keeps a tilt of
            // a few nanoradians, which spreads the individual contacts by a few nanometres
            let pens_here: Vec<f64> = idx
                .iter()
                .map(|&i| d.contact_includemargin[i] - d.contact_dist[i])
                .collect();
            let mean = pens_here.iter().sum::<f64>() / pens_here.len() as f64;
            let spread = pens_here
                .iter()
                .fold(0.0f64, |m, p| m.max((p - mean).abs()));
            println!(
                "MEASURED f64 stack settled ({cone}): interface {k} mean penetration predicted {:e} m, measured {mean:e} m after 12 s (spread of its {} contacts {spread:e} m)",
                pens[k],
                idx.len()
            );
            assert!(
                (mean - pens[k]).abs() <= 1e-6 * pens[k],
                "{cone} interface {k}: {mean:e}"
            );
        }
        assert!(d.qvel.iter().all(|v| v.abs() < 1e-8));
    }
}

// ---------------------------------------------------------------------------
// (d), (e) a sliding and a spinning sphere
// ---------------------------------------------------------------------------

fn slider_xml(cone: &str, condim: u32, friction: &str) -> String {
    format!(
        r#"<mujoco><option timestep="0.002" cone="{cone}"/>
           <worldbody>
             <geom name="floor" type="plane" size="5 5 .1" condim="{condim}" friction="{friction}"/>
             <body name="ball" pos="0 0 .1"><freejoint/>
               <geom name="ball_geom" type="sphere" size=".1" mass="1" condim="{condim}" friction="{friction}"/>
             </body>
           </worldbody></mujoco>"#
    )
}

/// One step's record of the contact of a sliding or spinning sphere.
struct Sample {
    t: f64,
    /// Whether the sphere touches the plane in this step (a contact that is a constraint).
    in_contact: bool,
    /// The sphere's acceleration `qacc` (world-frame linear, body-frame angular).
    qacc: [f64; 6],
    /// `contact_force`: normal, two tangents, torsion, two rolling (zero without contact).
    force: [f64; 6],
    zone: ConstraintState,
    qvel: [f64; 6],
}

/// Runs the sphere from the pose `z` (the centre height) with the velocity `v` for `seconds`,
/// recording each step's contact.
fn run_sphere(m: &Model<f64>, z: f64, v: [f64; 6], seconds: f64) -> Vec<Sample> {
    let mm = tighten(m, PrimalSolver::Newton);
    let mut d = Data::new(&mm);
    d.qpos[2] = z;
    d.qvel.copy_from_slice(&v);
    let n = (seconds / mm.timestep).round() as usize;
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        step(&mm, &mut d);
        assert_eq!(d.warning_collision_overflow, 0);
        let in_contact = d.ncon > 0 && d.contact_efc_address[0] >= 0;
        let (force, zone) = if in_contact {
            (
                contact_force(&mm, &d, 0),
                d.efc_state[d.contact_efc_address[0] as usize],
            )
        } else {
            ([0.0; 6], ConstraintState::Satisfied)
        };
        out.push(Sample {
            t: (k + 1) as f64 * mm.timestep,
            in_contact,
            qacc: d.qacc[..6].try_into().unwrap(),
            force,
            zone,
            qvel: d.qvel[..6].try_into().unwrap(),
        });
    }
    out
}

/// A sphere sliding on a plane (elliptic cone, condim 3, f0 = 0.5, 2 m/s).
///
/// **What the soft cone does.** In the cone (middle) zone the friction force is exactly `f0` times
/// the normal force, and the normal force is `F_n = D0 (f0 |jar_t| - jar_n) / (1 + mu^2)`, with
/// `jar_t` the slip's reference acceleration `a_t + B v_slip`. At rest the second term is a small
/// spring, but for a slip of 2 m/s `B v_slip` is 200 m/s^2, `D0` is 10 or more, and the cone
/// pushes the sphere up (58 N in the first contact step) until it leaves the plane; it returns
/// and is pushed again. The sphere HOPS while it slides: a contact step in the cone zone, then
/// free flight, again and again, until the slip is small and the contact stays in the bottom zone.
/// MuJoCo does the same (its own run of this sphere is in the golden file and ours equals it to
/// the last bit), so a steady `F_n = m g` and a deceleration of exactly `f0 g` are NOT what the
/// soft cone produces, and the spec's "within 1e-6 of `mu g`" cannot hold (measured below).
///
/// What does hold exactly, from the force law and Newton's second law (`a_z = (F_n - m g) / m`,
/// `a_x = -F_t / m`, `F_t = f0 F_n` in the cone zone, and nothing in free flight):
///
/// 1. in every contact step in the cone zone, `|F_t| = f0 F_n` to 1e-9;
/// 2. in every step in the cone zone or in free flight, `a_x = -f0 (g + a_z)`, to 1e-9 of `f0 g`
///    (the impulse balance: the deceleration is `f0 g` plus `f0` times the vertical acceleration);
/// 3. so over the sliding phase the mean deceleration is `f0 (g + Delta v_z / T)`, which is `f0 g`
///    only when the sphere returns to its vertical velocity;
/// 4. the sphere ends rolling at `5 v0 / 7`, whatever the force history (the angular momentum
///    about the line of contact is conserved: the normal force and gravity act on its vertical
///    line), within 1e-3 here, and the contact leaves the cone zone near the predicted rolling
///    onset `2 v0 / (7 f0 g)` (within 10%);
/// 5. with the pyramidal cone the ratio of friction to normal force is below `f0` (F8) and is
///    reported; with `condim 1` there is no horizontal acceleration at all.
#[test]
fn a_sliding_sphere_follows_the_impulse_balance_of_the_elliptic_cone() {
    let (f0, v0) = (0.5, 2.0);
    let friction = format!("{f0} 0.005 0.0001");
    let t_roll = 2.0 * v0 / (7.0 * f0 * G);

    // ---- elliptic, condim 3, from the touching pose (as the golden file's run)
    let m = compile(&slider_xml("elliptic", 3, &friction));
    let run = run_sphere(&m, 0.1, [v0, 0.0, 0.0, 0.0, 0.0, 0.0], 2.0 * t_roll);

    // the contact leaves the cone zone for good at the onset
    let onset = run
        .iter()
        .position(|s| s.in_contact && s.zone != ConstraintState::Cone)
        .expect("the sphere starts to roll");
    assert!(
        run[onset..]
            .iter()
            .all(|s| !s.in_contact || s.zone != ConstraintState::Cone),
        "the cone zone returns after the onset"
    );
    let t_onset = run[onset].t;

    // 1 and 2: the cone ratio and the impulse balance, in every step of the sliding phase
    let (mut worst_ratio, mut worst_balance) = (0.0f64, 0.0f64);
    let (mut cone_steps, mut flight_steps) = (0usize, 0usize);
    for s in &run[..onset] {
        if s.in_contact {
            assert_eq!(s.zone, ConstraintState::Cone, "t = {}", s.t);
            let ft = s.force[1].hypot(s.force[2]);
            worst_ratio = worst_ratio.max((ft / s.force[0] - f0).abs() / f0);
            cone_steps += 1;
        } else {
            flight_steps += 1;
        }
        worst_balance = worst_balance.max((s.qacc[0] + f0 * (G + s.qacc[2])).abs() / (f0 * G));
    }
    println!(
        "MEASURED f64 sliding sphere (elliptic, v0 {v0} m/s): {cone_steps} contact steps, all in the cone zone, and {flight_steps} steps of free flight before the contact leaves the cone zone at t = {t_onset:.4} s (predicted rolling onset {t_roll:.4} s); worst |f_t / f_n - f0| / f0 = {worst_ratio:e}; worst |a_x + f0 (g + a_z)| / (f0 g) = {worst_balance:e}"
    );
    assert!(
        cone_steps >= 10 && flight_steps >= 10,
        "the sphere should hop while it slides"
    );
    assert!(worst_ratio <= 1e-9, "{worst_ratio:e}");
    assert!(worst_balance <= 1e-9, "{worst_balance:e}");

    // 3: the mean deceleration over the sliding phase and its identity
    let last = &run[onset - 1];
    let mean_decel = (v0 - last.qvel[0]) / last.t;
    let identity = f0 * (G + last.qvel[2] / last.t);
    println!(
        "MEASURED f64 sliding sphere (elliptic): mean deceleration over the sliding phase {mean_decel:.6} m/s^2 = {:.6} of f0 g (the spec predicted 1 within 1e-6); the impulse balance f0 (g + Delta v_z / T) = {identity:.6}, difference {:e} of f0 g; Delta v_z = {:.6} m/s over T = {:.4} s",
        mean_decel / (f0 * G),
        (mean_decel - identity).abs() / (f0 * G),
        last.qvel[2],
        last.t
    );
    assert!((mean_decel - identity).abs() <= 1e-9 * f0 * G);

    // 4: the rolling onset and the final speed
    let v_end = run.last().unwrap().qvel[0];
    println!(
        "MEASURED f64 sliding sphere (elliptic): the contact leaves the cone zone at t = {t_onset:.4} s against the predicted rolling onset {t_roll:.4} s (error {:.2}%); final speed {v_end:.6} m/s against 5 v0 / 7 = {:.6} m/s (error {:.3e} of v0)",
        100.0 * (t_onset - t_roll) / t_roll,
        5.0 * v0 / 7.0,
        (v_end - 5.0 * v0 / 7.0).abs() / v0
    );
    // the cone zone ends before the ideal rolling onset (the soft friction lets the slip go
    // a little early), within 10% of it
    assert!(
        t_onset < t_roll && t_onset > 0.9 * t_roll,
        "{t_onset} vs {t_roll}"
    );
    assert!((v_end - 5.0 * v0 / 7.0).abs() <= 1e-3 * v0, "{v_end}");

    // 5a: the pyramidal cone, reported
    let mp = compile(&slider_xml("pyramidal", 3, &friction));
    let runp = run_sphere(&mp, 0.1, [v0, 0.0, 0.0, 0.0, 0.0, 0.0], 2.0 * t_roll);
    let onset_p = runp
        .iter()
        .position(|s| s.in_contact && s.zone == ConstraintState::Quadratic)
        .unwrap_or(runp.len());
    let ratios: Vec<f64> = runp[..onset_p]
        .iter()
        .filter(|s| s.in_contact)
        .map(|s| s.force[1].hypot(s.force[2]) / s.force[0])
        .collect();
    let worst_p = ratios.iter().cloned().fold(0.0f64, f64::max);
    let last_p = &runp[onset_p.max(1) - 1];
    println!(
        "MEASURED f64 sliding sphere (pyramidal): over {} contact steps of the sliding phase friction / normal force is {worst_p:.6} at most (f0 = {f0}); mean deceleration {:.6} of f0 g; final speed {:.6} m/s",
        ratios.len(),
        (v0 - last_p.qvel[0]) / last_p.t / (f0 * G),
        runp.last().unwrap().qvel[0]
    );
    // the pyramid is inscribed in the cone: never above f0
    assert!(worst_p <= f0 * (1.0 + 1e-12), "{worst_p}");

    // 5a': the instantaneous friction / normal force at the rest penetration for a range of slip
    // speeds, both cones. F8 (INFERRED in the spec) expected the pyramid to stay below f0 while it
    // slides because the two edges across the slide carry spring load; measured: at the first
    // contact step the other three edges are `Satisfied` once the slip is above about 0.1 m/s (the
    // sphere is pushed up, so their `jar` is positive), a single edge is active, and the ratio is
    // exactly f0. Below that the contact is in its soft-stick regime (ratio below f0 for both cones).
    for cone in ["pyramidal", "elliptic"] {
        let mc = tighten(
            &compile(&slider_xml(cone, 3, &friction)),
            PrimalSolver::Newton,
        );
        let mut line = String::new();
        for vs in [0.01, 0.05, 0.1, 0.3, 1.0, 2.0] {
            let mut d = Data::new(&mc);
            d.qpos[2] = 0.1 - 3.67e-4;
            d.qvel[0] = vs;
            sim_physics::forward(&mc, &mut d);
            assert_eq!(d.warning_collision_overflow, 0, "box-box overflow");
            let force = contact_force(&mc, &d, 0);
            let ratio = force[1].hypot(force[2]) / force[0];
            line += &format!(" {vs}: {ratio:.6}");
            assert!(ratio <= f0 * (1.0 + 1e-12), "{cone} {vs}: {ratio}");
            if vs >= 0.3 {
                assert!(
                    (ratio - f0).abs() <= 1e-9,
                    "{cone}: sliding at {vs} m/s has friction / normal = {ratio}"
                );
            }
        }
        println!(
            "MEASURED f64 sliding sphere ({cone}): instantaneous friction / normal force at slip speed (m/s):{line}"
        );
    }

    // 5b: a frictionless contact (condim 1) has no horizontal acceleration at all
    let m1 = compile(&slider_xml("elliptic", 1, &friction));
    let run1 = run_sphere(&m1, 0.1, [v0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0);
    let worst_x = run1.iter().map(|s| s.qacc[0].abs()).fold(0.0f64, f64::max);
    println!(
        "MEASURED f64 sliding sphere (condim 1): max horizontal acceleration {worst_x:e}; final speed {:.12} m/s",
        run1.last().unwrap().qvel[0]
    );
    assert!(worst_x <= 1e-12, "{worst_x:e}");
    assert!((run1.last().unwrap().qvel[0] - v0).abs() <= 1e-12 * v0);
}

/// A spinning sphere: `condim 3` has nothing that resists the spin about the normal; `condim 4`
/// (elliptic) has torsional friction. In its cone zone `|F_tors| = friction[2] F_n` exactly, so
/// the spin decelerates at `friction[2] F_n / I_zz` with `F_n = m (g + a_z)`: the per-step identity
/// `alpha = friction[2] m (g + a_z) / I_zz` is gated to 1e-9, and equals the spec's `friction[2] N
/// / I_zz` with `N = m g` only when the sphere has no vertical acceleration; as for the sliding
/// sphere, the soft cone's `B v_spin` term pushes the sphere up (it hops) while it spins fast, so
/// the spec's "within 1e-6" is not what the cone does (measured below).
#[test]
fn a_spinning_sphere_keeps_its_spin_with_condim_3_and_slows_by_torsional_friction_with_condim_4() {
    // sphere of radius 0.1 and mass 1: I_zz = 2/5 m r^2
    let inertia = 0.4 * 1.0 * 0.1 * 0.1;
    let w0 = 30.0;
    // ---- condim 3: nothing resists the spin about the normal
    for cone in ["pyramidal", "elliptic"] {
        let m = compile(&slider_xml(cone, 3, "0.5 0.005 0.0001"));
        let run = run_sphere(
            &m,
            0.1 - 3.6718184246016636e-4,
            [0.0, 0.0, 0.0, 0.0, 0.0, w0],
            1.0,
        );
        let drift = run
            .iter()
            .map(|s| (s.qvel[5] - w0).abs())
            .fold(0.0f64, f64::max);
        println!(
            "MEASURED f64 spinning sphere (condim 3, {cone}): max |omega_z - omega_0| over 1 s = {drift:e} (omega_0 = {w0} rad/s)"
        );
        assert!(drift <= 1e-12 * w0, "{cone}: {drift:e}");
    }

    // ---- condim 4, elliptic
    let f_tors = 0.005;
    let friction = format!("0.5 {f_tors} 0.0001");
    let m = compile(&slider_xml("elliptic", 4, &friction));
    let alpha_nominal = f_tors * G / inertia; // friction[2] N / I_zz with N = m g
    let t_stop = w0 / alpha_nominal;
    let run = run_sphere(&m, 0.1, [0.0, 0.0, 0.0, 0.0, 0.0, w0], 0.9 * t_stop);
    let (mut worst_ratio, mut worst_alpha, mut cone_steps) = (0.0f64, 0.0f64, 0usize);
    let mut z_accel = 0.0f64;
    for s in run
        .iter()
        .filter(|s| s.in_contact && s.zone == ConstraintState::Cone)
    {
        // the torsional force of the cone is friction[2] times the normal force
        worst_ratio = worst_ratio.max((s.force[3].abs() / s.force[0] - f_tors).abs() / f_tors);
        // and decelerates the spin: alpha = F_tors / I_zz = friction[2] m (g + a_z) / I_zz
        let alpha_pred = f_tors * (G + s.qacc[2]) / inertia;
        worst_alpha = worst_alpha.max((-s.qacc[5] - alpha_pred).abs() / alpha_pred);
        z_accel = z_accel.max(s.qacc[2].abs());
        cone_steps += 1;
    }
    println!(
        "MEASURED f64 spinning sphere (condim 4, elliptic): {cone_steps} contact steps in the cone zone of {} (spin stops at {t_stop:.3} s); worst |F_tors / F_n - friction[2]| / friction[2] = {worst_ratio:e}; worst relative error of alpha = friction[2] m (g + a_z) / I_zz: {worst_alpha:e}; the vertical acceleration reaches {z_accel:.3} m/s^2 (the spec's alpha = friction[2] N / I_zz = {alpha_nominal:.4} rad/s^2 assumed none)",
        run.len()
    );
    assert!(cone_steps >= 10);
    assert!(worst_ratio <= 1e-9, "{worst_ratio:e}");
    assert!(worst_alpha <= 1e-9, "{worst_alpha:e}");
    // the mean deceleration over the run, against the nominal one and the impulse balance
    let last = run.last().unwrap();
    let mean_alpha = (w0 - last.qvel[5]) / last.t;
    let balance = f_tors * (G + last.qvel[2] / last.t) / inertia;
    println!(
        "MEASURED f64 spinning sphere (condim 4, elliptic): mean deceleration {mean_alpha:.4} rad/s^2 = {:.4} of friction[2] N / I_zz (impulse balance friction[2] m (g + Delta v_z / T) / I_zz = {balance:.4}, if every step were in the cone zone)",
        mean_alpha / alpha_nominal,
    );

    // ---- condim 4, pyramidal: reported
    let mp = compile(&slider_xml("pyramidal", 4, &friction));
    let runp = run_sphere(&mp, 0.1, [0.0, 0.0, 0.0, 0.0, 0.0, w0], 0.5 * t_stop);
    let s = runp.last().unwrap();
    println!(
        "MEASURED f64 spinning sphere (condim 4, pyramidal): mean torsional deceleration / (friction[2] N / I_zz) = {:.4} over {:.4} s (reported, not predicted)",
        (w0 - s.qvel[5]) / s.t / alpha_nominal,
        s.t
    );
}

// ---------------------------------------------------------------------------
// (f) the pile
// ---------------------------------------------------------------------------

/// The penetration of the eight-box pile, both cones, against ADR-0042's bar of 3 mm.
#[test]
fn the_box_pile_penetrates_less_than_3_mm() {
    for v in CONES {
        let c = tcompile::<f64>(TWhich::Pile, v);
        let g = tgolden(TWhich::Pile, v);
        let settle = &g["settle"];
        let m = &c.model;

        // the positive control: the Rust metric on MuJoCo's own poses gives the generator's value
        let mut worst_control = 0.0f64;
        let mut mujoco_sat = 0.0f64;
        let mut mujoco_floor = 0.0f64;
        for fr in settle["frames"].as_array().unwrap() {
            let (pos, quat) = pile_poses_scene(&farr(&fr["qpos"]));
            let (bb, floor, pairs, penetrating) = box_pile_penetration(&pos, &quat, 0.1);
            let want = &fr["sat"];
            worst_control = worst_control
                .max((bb - f(&want["box_box_max"])).abs())
                .max((floor - f(&want["box_floor_max"])).abs());
            assert_eq!(pairs as u64, want["box_box_pairs"].as_u64().unwrap());
            assert_eq!(
                penetrating as u64,
                want["box_box_penetrating"].as_u64().unwrap()
            );
            mujoco_sat = mujoco_sat.max(bb);
            mujoco_floor = mujoco_floor.max(floor);
        }
        println!(
            "MEASURED f64 pile {} positive control: the Rust SAT metric on MuJoCo's poses differs from the generator's numpy value by at most {worst_control:e} m",
            v.name()
        );
        assert!(worst_control <= 1e-12, "{worst_control:e}");

        // our run: 2,400 steps, the last second recorded
        let total = settle["steps"].as_u64().unwrap() as usize;
        let frame_steps: Vec<usize> = settle["frames"]
            .as_array()
            .unwrap()
            .iter()
            .map(|fr| fr["step"].as_u64().unwrap() as usize)
            .collect();
        let mut d = Data::new(m);
        let (mut our_sat, mut our_floor, mut worst_depth) = (0.0f64, 0.0f64, 0.0f64);
        let mut ncon_peak = 0usize;
        for n in 1..=total {
            step(m, &mut d);
            assert_eq!(d.warning_collision_overflow, 0);
            if n > total - 240 {
                for i in 0..d.ncon {
                    worst_depth = worst_depth.max(-d.contact_dist[i]);
                }
                ncon_peak = ncon_peak.max(d.ncon);
            }
            if frame_steps.contains(&n) {
                let qpos = common::scene_qpos(m, &d.qpos);
                let (pos, quat) = pile_poses_scene(&qpos);
                let (bb, floor, _, _) = box_pile_penetration(&pos, &quat, 0.1);
                our_sat = our_sat.max(bb);
                our_floor = our_floor.max(floor);
            }
        }
        let speed = (0..m.nbody - 1)
            .map(|b| {
                let q = &d.qvel[6 * b..6 * b + 3];
                (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt()
            })
            .fold(0.0f64, f64::max);
        println!(
            "MEASURED f64 pile ({}, 8 boxes, edge 0.2 m, 2,400 steps of {:.6} s): maximum box-box SAT penetration over the last second {:.4} mm (MuJoCo's: {:.4} mm, difference {:.3e} m); maximum box-floor penetration {:.4} mm (MuJoCo's {:.4} mm); engine-reported maximum depth {:.4} mm (MuJoCo's {:.4} mm); maximum speed at 10 s {speed:.6} m/s (MuJoCo's {:.6}); rest criterion 0.01 m/s; peak contacts {ncon_peak}",
            v.name(),
            m.timestep,
            1e3 * our_sat,
            1e3 * mujoco_sat,
            our_sat - mujoco_sat,
            1e3 * our_floor,
            1e3 * mujoco_floor,
            1e3 * worst_depth,
            1e3 * f(&settle["last_second_max_neg_dist"]),
            f(&settle["end_max_speed"]),
        );
        println!(
            "MEASURED context: the engine comparison's 2.9 mm for MuJoCo (the engine comparison) is 1,000 boxes of edge 0.2 m with CG at 100 iterations (Newton was infeasible), a separating-axis maximum over 10 frames of the last second, with the pile not at rest (0.12 m/s at 10 s); this run is 8 boxes with Newton at rest, so the two figures are not like for like"
        );
        // ADR-0042 P1: box-pile penetration of 3 mm or less
        assert!(our_sat <= 3e-3, "{}: {our_sat:e} m", v.name());
        assert!(
            speed < 0.01,
            "{}: the pile is not at rest, {speed} m/s",
            v.name()
        );
    }
}
