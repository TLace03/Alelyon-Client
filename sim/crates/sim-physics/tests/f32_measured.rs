//! The same states and the same pendulum in `f32`: MEASURED, and gated loosely.
//!
//! `f32` is what the GPU port will run. This test runs the whole engine in `f32`
//! on the states of the golden files and reports, for every array, the worst
//! `max |f32 result - MuJoCo's f64 value| / max |MuJoCo's value|` over the states
//! (`MEASURED f32 ...` lines; they are the GPU port's starting expectation). The
//! gates are the spec's loose ones: `qacc` within 1e-3 relative, and the energy
//! drift of the `f32` RK4 double pendulum at 10 s within 1e-3.
//!
//! Where the `f32` error comes from is not decided here: M is built from products
//! of rotated inertias and the solve divides by the condition number of M; the
//! comparison is made against MuJoCo's `f64` values, so it includes the rounding of
//! the `f32` model constants (rotations, inertias) as well as of the step.

mod common;

use common::*;

fn run(which: Which) {
    let report = check_model::<f32>(which, &TOL_MEASURE, None);
    report.print("f32", which.name());
    let qacc = report.get("qacc").expect("qacc was compared");
    assert!(
        qacc.rel() <= 1e-3,
        "{}: f32 qacc relative error {:e} exceeds 1e-3",
        which.name(),
        qacc.rel()
    );
}

#[test]
fn humanoid_f32_is_measured() {
    run(Which::Humanoid);
}

#[test]
fn zoo_f32_is_measured() {
    run(Which::Zoo);
}

#[test]
fn f32_rk4_pendulum_energy_drift_at_10_s() {
    let s = pendulum_series::<f32>("double_pendulum.xml", "pendulum_rk4_golden.json");
    let ours = drift(s.ours.iter().map(|(_, e)| *e));
    let mujoco = drift(s.golden.iter().copied());
    let (worst, at) = worst_energy_error(&s);
    let state = s.state_err.iter().cloned().fold(0.0, f64::max);
    let dev = max_deviation(s.ours.iter().map(|(_, e)| *e));
    println!("MEASURED f32 pendulum RK4 energy_drift_10s: ours={ours:e} mujoco(f64)={mujoco:e}");
    println!("MEASURED f32 pendulum RK4 max_deviation_from_E0: {dev:e}");
    println!(
        "MEASURED f32 pendulum RK4 energy_series_vs_f64_mujoco: max_rel_err={worst:e} (at step {at})"
    );
    println!(
        "MEASURED f32 pendulum RK4 state_series_vs_f64_mujoco: max_abs_err(qpos,qvel)={state:e}"
    );
    assert!(ours.abs() <= 1e-3, "f32 RK4 drift {ours:e}");
}

#[test]
fn f32_euler_pendulum_energy_drift_at_10_s() {
    let s = pendulum_series::<f32>("double_pendulum_euler.xml", "pendulum_euler_golden.json");
    let ours = drift(s.ours.iter().map(|(_, e)| *e));
    let mujoco = drift(s.golden.iter().copied());
    println!("MEASURED f32 pendulum Euler energy_drift_10s: ours={ours:e} mujoco(f64)={mujoco:e}");
    // the Euler drift is a property of the method, not of the precision: f32 loses
    // what f64 loses, within 1 % of E(0)
    assert!((ours - mujoco).abs() < 1e-2, "{ours:e} vs {mujoco:e}");
}
