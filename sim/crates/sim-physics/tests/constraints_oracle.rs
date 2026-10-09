//! The constraint golden files are what the generator script produces from the oracle.
//!
//! Runs `tools/sim_constraints_mujoco_golden.py --check` with the MuJoCo oracle
//! interpreter when it exists (it regenerates the three constraint golden files in
//! memory with MuJoCo 3.14.0 and compares them with the committed ones) and says SKIPPED
//! on stderr when it does not; set `SIM_PHYSICS_REQUIRE_ORACLE=1` to make a missing
//! interpreter a failure. The gate also runs the same command directly.

use std::path::PathBuf;

#[test]
fn the_constraint_generator_check_passes() {
    let python = std::env::var("SIM_PHYSICS_ORACLE_PYTHON").unwrap_or_default();
    if !std::path::Path::new(&python).exists() {
        assert!(
            std::env::var("SIM_PHYSICS_REQUIRE_ORACLE").is_err(),
            "the MuJoCo oracle interpreter {python} is required and missing"
        );
        eprintln!(
            "SKIPPED the_constraint_generator_check_passes: no MuJoCo oracle interpreter (set SIM_PHYSICS_ORACLE_PYTHON; now {python:?})"
        );
        return;
    }
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../..");
    let script = repo.join("tools/sim_constraints_mujoco_golden.py");
    if !script.exists() {
        // A copy of this crate outside the source repository has no generator.
        assert!(
            std::env::var("SIM_PHYSICS_REQUIRE_ORACLE").is_err(),
            "the golden generator {} is required and missing",
            script.display()
        );
        eprintln!(
            "SKIPPED the_constraint_generator_check_passes: no golden generator at {}",
            script.display()
        );
        return;
    }
    let output = std::process::Command::new(&python)
        .arg(&script)
        .arg("--check")
        .output()
        .expect("the oracle interpreter starts");
    assert!(
        output.status.success(),
        "tools/sim_constraints_mujoco_golden.py --check failed ({}):\n{}{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    println!("{}", String::from_utf8_lossy(&output.stdout).trim());
}
