//! The double pendulum of the engine comparison's multiphysics benchmark
//! (dt = 1/240, `qpos0 = [1.2, -0.7]`, 10 s), RK4 and Euler, in `f64`.
//!
//! `fixtures/pendulum_{rk4,euler}_golden.json` hold MuJoCo's `qpos`, `qvel` and
//! energy of the STATE at step 0, 10, ..., 2400. The engine is run for 2400 steps and
//! its energy series is compared with MuJoCo's at every recorded step.
//!
//! What is gated:
//! - the engine's series equals MuJoCo's to 1e-9 relative at every recorded step,
//!   for RK4 and for Euler, and the drift equals MuJoCo's;
//! - the Euler drift is bounded (the benchmark's -5.2 % with a bounded oscillation);
//! - the RK4 drift `|E(10 s) - E(0)| / |E(0)|` is no worse than MuJoCo's own.
//!   MuJoCo 3.14.0 drifts by -2.02e-6 on this model in the state-based series here
//!   (the benchmark's record `mj_pend_RK4_r0.json`, which reads `d.energy` after
//!   each step, says -2.05e-6). An earlier spec quoted -6e-7 and a 1e-6 gate:
//!   -6e-7 is MuJoCo RK4's drift in the comparison's OTHER double pendulum (the
//!   rigid-engine scene, relative to a 19.62 J swing), not in this one.

mod common;

use common::*;

#[test]
fn rk4_energy_series_matches_mujoco() {
    let s = pendulum_series::<f64>("double_pendulum.xml", "pendulum_rk4_golden.json");
    let (worst, at) = worst_energy_error(&s);
    let ours = drift(s.ours.iter().map(|(_, e)| *e));
    let mujoco = drift(s.golden.iter().copied());
    let state = s.state_err.iter().cloned().fold(0.0, f64::max);
    println!(
        "MEASURED f64 pendulum RK4 energy_series: max_rel_err={worst:e} (at step {at}) tol_rel=1e-9"
    );
    println!("MEASURED f64 pendulum RK4 state_series: max_abs_err(qpos,qvel)={state:e}");
    println!(
        "MEASURED f64 pendulum RK4 energy_drift_10s: ours={ours:e} mujoco={mujoco:e} difference={:e}",
        ours - mujoco
    );
    assert!(worst <= 1e-9, "energy series: {worst:e} at step {at}");
    // the same drift as MuJoCo's, to 1e-9 of the energy (a consequence of the series match)
    assert!((ours - mujoco).abs() <= 1e-9, "{ours:e} vs {mujoco:e}");
}

#[test]
fn euler_energy_series_matches_mujoco_and_drift_is_bounded() {
    let s = pendulum_series::<f64>("double_pendulum_euler.xml", "pendulum_euler_golden.json");
    let (worst, at) = worst_energy_error(&s);
    let ours = drift(s.ours.iter().map(|(_, e)| *e));
    let mujoco = drift(s.golden.iter().copied());
    let dev = max_deviation(s.ours.iter().map(|(_, e)| *e));
    let state = s.state_err.iter().cloned().fold(0.0, f64::max);
    println!(
        "MEASURED f64 pendulum Euler energy_series: max_rel_err={worst:e} (at step {at}) tol_rel=1e-9"
    );
    println!("MEASURED f64 pendulum Euler state_series: max_abs_err(qpos,qvel)={state:e}");
    println!(
        "MEASURED f64 pendulum Euler energy_drift_10s: ours={ours:e} mujoco={mujoco:e}; max deviation from E(0): {dev:e}"
    );
    assert!(worst <= 1e-9, "energy series: {worst:e} at step {at}");
    // bounded: the benchmark's Euler loses 5.2 % and oscillates inside 5.5 %
    assert!(
        ours < 0.0 && dev < 0.06,
        "drift {ours:e}, deviation {dev:e}"
    );
}

/// The RK4 drift at 10 s is no worse than MuJoCo's own on the same model (the
/// series test above already holds it equal to 1e-9; this states the bar the ADR
/// uses, an energy error of about 2e-6, in a form that cannot be met by drifting
/// further).
#[test]
fn rk4_energy_drift_is_no_worse_than_mujocos() {
    let s = pendulum_series::<f64>("double_pendulum.xml", "pendulum_rk4_golden.json");
    let ours = drift(s.ours.iter().map(|(_, e)| *e));
    let mujoco = drift(s.golden.iter().copied());
    println!("MEASURED f64 pendulum RK4 drift at 10 s: ours {ours:e}, MuJoCo {mujoco:e}");
    assert!(
        ours.abs() <= mujoco.abs() * (1.0 + 1e-9),
        "|drift| = {:e} exceeds MuJoCo's {:e}",
        ours.abs(),
        mujoco.abs()
    );
    assert!(
        mujoco.abs() < 3e-6,
        "the oracle's own drift moved: {mujoco:e}"
    );
}
