//! A free body in no gravity keeps its angular momentum.
//!
//! A rigid body with an asymmetric inertia tumbles (Euler's equations) but, with no
//! external torque, its angular momentum about its centre of mass, written in the
//! world frame, is constant. That is a check of the engine that needs no oracle: the
//! inertia matrix, the Coriolis (gyroscopic) term of the Newton-Euler pass, the free
//! joint's convention (linear velocity in the world frame, angular velocity in the
//! body frame) and the quaternion integration all have to be right for it to hold.
//! `fixtures/free_body.xml` has the centre of mass off the body origin and the
//! principal axes rotated in the body frame, so the check is not an accident of
//! symmetry.
//!
//! The spec's gate is `|dL| / |L| <= 1e-9` over 1,000 RK4 steps in `f64`. How small
//! that is depends on the timestep, because MuJoCo's RK4 is only second order on the
//! orientation: it rotates the body by the stage-AVERAGED body-frame angular velocity
//! (`mj_RungeKutta` hands `mj_integratePos` the weighted sum of the stage velocities),
//! which ignores that the body-frame velocity changes within the step. The error of
//! the angular momentum is then `O(steps * h^3)`. This is MuJoCo's behaviour, not the
//! port's: the golden file holds what MuJoCo 3.14.0 measures for the same body, and
//! the port reproduces it. At MuJoCo's default 2 ms the 1,000 steps lose 2.1e-7, so
//! the gate is run at `h = 1e-4` (1.05e-10 in MuJoCo), where the integration error is
//! below it; 2 ms and 0.5 ms are compared with MuJoCo's values.

mod common;

use common::*;
use sim_physics::faults::{Faults, step_faulted};
use sim_physics::{Data, Model, Real, energy_vel, forward, step};
use sim_scene::Scene;

fn scene_at(h: f64) -> Scene {
    let mut s = load_scene(&fixtures().join("free_body.xml"));
    s.timestep_s = h;
    s
}

fn quat_normalised(q: [f64; 4]) -> [f64; 4] {
    let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

/// A 3x3 rotation matrix, row-major, from `[w, x, y, z]`.
fn mat(q: [f64; 4]) -> [[f64; 3]; 3] {
    let [w, x, y, z] = q;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn mul_mat_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

fn transpose(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [0, 1, 2].map(|i| [0, 1, 2].map(|j| m[j][i]))
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// What the check needs of the scene: the principal moments and the rotation of the
/// principal axes in the body frame, `[w, x, y, z]`.
struct Body {
    diag: [f64; 3],
    iquat_wxyz: [f64; 4],
}

fn body_of(scene: &Scene) -> Body {
    let i = scene.bodies[0].inertial.expect("an inertial");
    let q = quat_normalised(i.inertia_quat); // [x, y, z, w]
    Body {
        diag: i.diag_inertia,
        iquat_wxyz: [q[3], q[0], q[1], q[2]],
    }
}

/// The angular momentum about the centre of mass, in the world frame, of a state
/// (`qpos` with the quaternion `[w, x, y, z]` at 3..7, `qvel` with the body-frame
/// angular velocity at 3..6).
fn angular_momentum(b: &Body, qpos: &[f64], qvel: &[f64]) -> [f64; 3] {
    let body = mat([qpos[3], qpos[4], qpos[5], qpos[6]]);
    let ri = mat(b.iquat_wxyz);
    let in_principal = mul_mat_vec(&transpose(&ri), [qvel[3], qvel[4], qvel[5]]);
    let l_principal = [
        b.diag[0] * in_principal[0],
        b.diag[1] * in_principal[1],
        b.diag[2] * in_principal[2],
    ];
    mul_mat_vec(&body, mul_mat_vec(&ri, l_principal))
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

struct Outcome {
    max_dl: f64,
    final_dl: f64,
    max_dke: f64,
    l0: f64,
}

/// The golden file's start state.
fn start() -> ([f64; 4], Vec<f64>) {
    let g = read_json(&fixtures().join("free_body_golden.json"));
    let q = farr(&g["quat0"]);
    (quat_normalised([q[0], q[1], q[2], q[3]]), farr(&g["qvel0"]))
}

fn run<R: Real>(h: f64, steps: usize, faults: Option<&Faults>) -> Outcome {
    let scene = scene_at(h);
    let b = body_of(&scene);
    let (model, nm) = Model::<R>::compile(&scene).expect("compiles");
    assert!(nm.is_empty(), "{nm:?}");
    let mut d = Data::new(&model);
    let (q0, v0) = start();
    d.qpos[3..7].copy_from_slice(&q0.map(R::from_f64));
    for (dst, src) in d.qvel.iter_mut().zip(&v0) {
        *dst = R::from_f64(*src);
    }
    let wide = |d: &Data<R>| (widen(&d.qpos), widen(&d.qvel));
    let (q, v) = wide(&d);
    let l0 = angular_momentum(&b, &q, &v);
    forward(&model, &mut d);
    let ke0 = energy_vel(&model, &mut d).to_f64();
    let (mut max_dl, mut max_dke) = (0.0f64, 0.0f64);
    for _ in 0..steps {
        match faults {
            None => step(&model, &mut d),
            Some(f) => step_faulted(&model, &mut d, f),
        }
        let (q, v) = wide(&d);
        let dl = norm(sub(angular_momentum(&b, &q, &v), l0)) / norm(l0);
        max_dl = max_dl.max(if dl.is_nan() { f64::INFINITY } else { dl });
        forward(&model, &mut d);
        let ke = energy_vel(&model, &mut d).to_f64();
        max_dke = max_dke.max((ke - ke0).abs() / ke0);
    }
    let (q, v) = wide(&d);
    Outcome {
        max_dl,
        final_dl: norm(sub(angular_momentum(&b, &q, &v), l0)) / norm(l0),
        max_dke,
        l0: norm(l0),
    }
}

fn golden_run(h: f64) -> (f64, f64) {
    let g = read_json(&fixtures().join("free_body_golden.json"));
    let r = g["runs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| f(&r["timestep"]) == h)
        .unwrap_or_else(|| panic!("no golden run at h = {h}"));
    assert_eq!(r["steps"].as_u64(), Some(1000));
    (f(&r["max_dl_over_l"]), f(&r["final_dl_over_l"]))
}

#[test]
fn the_free_body_golden_belongs_to_its_xml() {
    let g = read_json(&fixtures().join("free_body_golden.json"));
    let bytes = std::fs::read(fixtures().join("free_body.xml")).unwrap();
    assert_eq!(
        g["xml"]["fnv1a64"].as_str(),
        Some(format!("{:016x}", fnv1a64(&bytes)).as_str())
    );
}

/// The spec's gate: `|dL| / |L| <= 1e-9` over 1,000 RK4 steps, in `f64`.
#[test]
fn a_free_asymmetric_body_in_no_gravity_keeps_its_angular_momentum_f64() {
    let h = 1e-4;
    let o = run::<f64>(h, 1000, None);
    let (mujoco_max, _) = golden_run(h);
    println!(
        "MEASURED f64 free-body rotation (1000 RK4 steps, h = {h}): max |dL|/|L|={:e} final |dL|/|L|={:e} max |dKE|/KE={:e} (|L|={}) ; MuJoCo 3.14.0: {mujoco_max:e}",
        o.max_dl, o.final_dl, o.max_dke, o.l0
    );
    assert!(o.max_dl <= 1e-9, "|dL|/|L| = {:e}", o.max_dl);
    // the body really tumbled: the check is not on a state that never moved
    assert!(o.l0 > 0.1);
    // and the port loses what MuJoCo loses
    assert!(
        (o.max_dl - mujoco_max).abs() <= 1e-3 * mujoco_max,
        "{:e} vs MuJoCo {mujoco_max:e}",
        o.max_dl
    );
}

/// At MuJoCo's own timesteps the loss is MuJoCo's loss: second order in `h`.
#[test]
fn the_angular_momentum_loss_at_larger_timesteps_is_mujocos() {
    let mut previous: Option<f64> = None;
    for h in [2e-3, 5e-4] {
        let o = run::<f64>(h, 1000, None);
        let (mujoco_max, mujoco_final) = golden_run(h);
        println!(
            "MEASURED f64 free-body rotation (1000 RK4 steps, h = {h}): max |dL|/|L|={:e} final |dL|/|L|={:e} ; MuJoCo 3.14.0: max {mujoco_max:e} final {mujoco_final:e}",
            o.max_dl, o.final_dl
        );
        assert!((o.max_dl - mujoco_max).abs() <= 1e-3 * mujoco_max);
        assert!((o.final_dl - mujoco_final).abs() <= 1e-3 * mujoco_final);
        if let Some(prev) = previous {
            // a timestep 4 times smaller over the same number of steps (a run 4
            // times shorter): the loss falls by about 19 times in MuJoCo
            assert!(prev / o.max_dl > 10.0, "{prev:e} then {:e}", o.max_dl);
        }
        previous = Some(o.max_dl);
    }
}

#[test]
fn the_same_in_f32_is_measured() {
    for h in [2e-3, 1e-4] {
        let o = run::<f32>(h, 1000, None);
        println!(
            "MEASURED f32 free-body rotation (1000 RK4 steps, h = {h}): max |dL|/|L|={:e} final |dL|/|L|={:e} max |dKE|/KE={:e}",
            o.max_dl, o.final_dl, o.max_dke
        );
        assert!(o.max_dl < 1e-3, "{:e}", o.max_dl);
    }
}

/// The check can fail: with the Coriolis (gyroscopic) term's sign flipped the body
/// does not keep its angular momentum.
#[test]
fn the_check_fails_with_a_flipped_gyroscopic_term() {
    let o = run::<f64>(1e-4, 1000, None);
    assert!(o.max_dl <= 1e-9, "positive control");
    let faults = Faults {
        flip_coriolis_sign: true,
        ..Faults::NONE
    };
    let bad = run::<f64>(1e-4, 1000, Some(&faults));
    println!(
        "free body with a flipped gyroscopic sign: max |dL|/|L| = {:e}",
        bad.max_dl
    );
    assert!(
        bad.max_dl > 1e-5,
        "|dL|/|L| = {:e} with the fault; the check cannot fail",
        bad.max_dl
    );
}
