//! The smooth dynamics in `f64` against MuJoCo 3.14.0, state by state.
//!
//! `fixtures/{humanoid,zoo}_golden.json` hold what MuJoCo computed from `K = 8`
//! states of each model (written by `tools/sim_physics_mujoco_golden.py`, with the
//! constraints switched off): `M`, `qfrc_bias`, `qfrc_passive`, `qfrc_actuator`,
//! `qacc`, the pose and the velocity of every body, the energy, one Euler step, one
//! RK4 step, and a 100-step trajectory with the model's own integrator. Each
//! comparison is held to the spec's tolerance (relative to the array's max-abs,
//! floored at 1e-12 absolute): 1e-10 for `M`, `qfrc_*`, `xpos`, `xquat`, velocities
//! and energies, 1e-9 for `qacc`, 1e-10 for one step, 1e-9 / 1e-8 / 1e-6 for
//! trajectory steps 1 / 10 / 100. The worst error of every array is printed
//! (`cargo test -- --nocapture`), one `MEASURED` line each.

mod common;

use common::*;

fn run(which: Which) {
    let report = check_model::<f64>(which, &TOL_F64, None);
    report.print("f64", which.name());
    assert!(
        report.all_within(),
        "{}: out of tolerance: {:?}",
        which.name(),
        report.failures()
    );
}

#[test]
fn humanoid_f64_matches_mujoco() {
    run(Which::Humanoid);
}

#[test]
fn zoo_f64_matches_mujoco() {
    run(Which::Zoo);
}

/// The golden states are not all alike: the comparison would pass vacuously on a
/// file whose arrays were all zero, so the files must carry real content.
#[test]
fn the_golden_states_are_not_degenerate() {
    for which in MODELS {
        let g = golden_of(which);
        let states = g["states"].as_array().unwrap();
        assert_eq!(states.len(), 8);
        for key in ["M", "qfrc_bias", "qacc"] {
            let all_zero = states
                .iter()
                .all(|s| farr(&s[key]).iter().all(|x| *x == 0.0));
            assert!(!all_zero, "{} {key} is all zero", which.name());
        }
        for key in ["xpos", "xvel"] {
            let all_zero = states
                .iter()
                .all(|s| frows(&s[key]).iter().all(|x| *x == 0.0));
            assert!(!all_zero, "{} {key} is all zero", which.name());
        }
        // the clamping state: some control outside its actuator range
        let ctrl = farr(&states[7]["ctrl"]);
        let outside = match which {
            Which::Humanoid => ctrl.iter().any(|x| x.abs() > 1.0),
            Which::Zoo => ctrl[0].abs() > 1.0 || ctrl[1].abs() > 0.2,
        };
        assert!(
            outside,
            "{}: no control outside its range in the last state",
            which.name()
        );
        // trajectories moved
        let t = frows(&states[7]["trajectory"]["qvel"]);
        assert!(t.iter().any(|x| x.abs() > 1e-3));
    }
}
