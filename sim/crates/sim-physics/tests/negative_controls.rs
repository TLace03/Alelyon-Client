//! Negative controls: the comparison with MuJoCo must FAIL when the engine is wrong.
//!
//! A check that has never been seen to fail proves nothing, so each control first
//! runs the very comparison the parity tests use on the unfaulted engine (the
//! positive control, which must pass) and then on an engine with one piece broken
//! by test-only fault injection (`sim_physics::faults::Faults`, hidden from the
//! public API): a flipped sign in the Coriolis term of `rne`, the armature dropped
//! from `M`, and the Euler integrator without implicit damping. Each must fail, and
//! it must fail where the fault is, not everywhere.

mod common;

use common::*;
use sim_physics::faults::Faults;

fn positive_control(which: Which) {
    let r = check_model::<f64>(which, &TOL_F64, None);
    assert!(
        r.all_within(),
        "positive control failed: {:?}",
        r.failures()
    );
    // and with the fault machinery switched on but no fault set: the same
    let r = check_model::<f64>(which, &TOL_F64, Some(&Faults::NONE));
    assert!(
        r.all_within(),
        "the faulted path with no fault must agree: {:?}",
        r.failures()
    );
}

fn failures_of(which: Which, faults: Faults) -> Vec<String> {
    let r = check_model::<f64>(which, &TOL_F64, Some(&faults));
    r.print("f64-fault", which.name());
    r.failures()
}

#[test]
fn a_flipped_coriolis_sign_in_rne_is_caught() {
    positive_control(Which::Humanoid);
    let failed = failures_of(
        Which::Humanoid,
        Faults {
            flip_coriolis_sign: true,
            ..Faults::NONE
        },
    );
    assert!(!failed.is_empty(), "the flipped sign was not caught");
    // the bias force is where it is wrong; the inertia matrix and the passive and
    // actuator forces do not depend on it
    assert!(failed.iter().any(|n| n == "qfrc_bias"), "{failed:?}");
    assert!(failed.iter().any(|n| n == "qacc"), "{failed:?}");
    for ok in [
        "M",
        "qfrc_passive",
        "qfrc_actuator",
        "xpos",
        "xquat",
        "xvel",
    ] {
        assert!(
            !failed.iter().any(|n| n == ok),
            "{ok} must not depend on the fault: {failed:?}"
        );
    }
    // the zoo catches it too
    positive_control(Which::Zoo);
    let failed = failures_of(
        Which::Zoo,
        Faults {
            flip_coriolis_sign: true,
            ..Faults::NONE
        },
    );
    assert!(failed.iter().any(|n| n == "qfrc_bias"), "{failed:?}");
}

#[test]
fn dropping_the_armature_from_m_is_caught() {
    for which in MODELS {
        positive_control(which);
        let failed = failures_of(
            which,
            Faults {
                drop_armature: true,
                ..Faults::NONE
            },
        );
        assert!(
            failed.iter().any(|n| n == "M"),
            "{}: {failed:?}",
            which.name()
        );
        assert!(
            failed.iter().any(|n| n == "qacc"),
            "{}: {failed:?}",
            which.name()
        );
        // the forces and the kinematics do not depend on M
        for ok in [
            "qfrc_bias",
            "qfrc_passive",
            "qfrc_actuator",
            "xpos",
            "xquat",
            "xvel",
        ] {
            assert!(
                !failed.iter().any(|n| n == ok),
                "{}: {ok} must not depend on the fault: {failed:?}",
                which.name()
            );
        }
    }
}

#[test]
fn euler_without_implicit_damping_is_caught() {
    for which in MODELS {
        positive_control(which);
        let failed = failures_of(
            which,
            Faults {
                euler_without_implicit_damping: true,
                ..Faults::NONE
            },
        );
        // only the Euler step differs; the forward pass and the RK4 step do not use it
        assert!(
            failed.iter().any(|n| n == "euler_step.qvel"),
            "{}: {failed:?}",
            which.name()
        );
        for ok in ["M", "qfrc_bias", "qacc", "rk4_step.qpos", "rk4_step.qvel"] {
            assert!(
                !failed.iter().any(|n| n == ok),
                "{}: {ok} must not depend on the fault: {failed:?}",
                which.name()
            );
        }
    }
}

/// The three faults together break everything they should, and the comparison is
/// not so lax that a small perturbation passes: perturbing one entry of one array
/// by 1e-6 relative is seen by the 1e-10 tolerance.
#[test]
fn the_tolerance_is_tight_enough_to_see_a_1e6_perturbation() {
    let golden = golden_of(Which::Humanoid);
    let state = &golden["states"][0];
    let reference = farr(&state["qfrc_bias"]);
    let mut perturbed = reference.clone();
    let k = reference
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .unwrap()
        .0;
    perturbed[k] *= 1.0 + 1e-6;
    let e = compare(&perturbed, &reference);
    assert!(!e.within(1e-10, 1e-12));
    // and an exact copy passes
    assert!(compare(&reference, &reference).within(1e-10, 1e-12));
}
