//! The compiled model of phase 1c-i against MuJoCo 3.14.0's own compile: the limit,
//! friction-loss and fixed-tendon arrays, the approximate inverse inertias that
//! `mj_setConst` computes at `qpos0` (`dof_invweight0`, `tendon_invweight0`,
//! `body_invweight0`), `stat.meaninertia`, and the solver options.
//!
//! Integers are compared exactly. Values copied from the XML (`solref`, `solimp`,
//! `margin`, `frictionloss`, the tendon coefficients) are compared exactly too: the
//! same decimal parse, no arithmetic. Values that went through arithmetic (the
//! degree to radian conversion of a range, the inverse inertias, the mean inertia) are
//! compared to 1e-12 relative to the array's largest absolute value. Every error is
//! printed as a `MEASURED` line.

mod common;

use common::cons::*;
use common::*;
use serde_json::Value;
use sim_physics::{JointType, PrimalSolver};

fn exact(name: &str, ours: &[f64], reference: &Value) {
    let reference = farr(reference);
    assert_eq!(ours, reference.as_slice(), "{name}");
}

fn close(label: &str, name: &str, ours: &[f64], reference: &Value) {
    let reference = farr(reference);
    let e = compare(ours, &reference);
    println!(
        "MEASURED f64 {label} arrays {name}: max_abs_err={:e} array_max_abs={:e} rel={:e} tol_rel=1e-12",
        e.abs,
        e.scale,
        e.rel()
    );
    assert!(
        e.within(1e-12, 1e-15),
        "{label} {name}: {:e} (max |ref| {:e})",
        e.abs,
        e.scale
    );
}

fn usizes(a: &[usize]) -> Vec<i64> {
    a.iter().map(|&x| x as i64).collect()
}

fn bools(a: &[bool]) -> Vec<i64> {
    a.iter().map(|&x| i64::from(x)).collect()
}

#[test]
fn the_constraint_arrays_match_mujoco() {
    for which in CMODELS {
        let g = cgolden(which);
        let a = &g["arrays"];
        let c = ccompile::<f64>(which);
        let m = &c.model;
        let name = which.name();

        // joints: the limit
        assert_eq!(
            bools(&m.jnt_limited),
            ints(&a["jnt_limited"]),
            "{name} jnt_limited"
        );
        // MuJoCo keeps a range for an unlimited joint too; ours is [0, 0] there
        let gr = farr(&a["jnt_range"]);
        let mut ours_r = Vec::new();
        let mut gold_r = Vec::new();
        for j in 0..m.njnt {
            if m.jnt_limited[j] {
                ours_r.extend_from_slice(&m.jnt_range[2 * j..2 * j + 2]);
                gold_r.extend_from_slice(&gr[2 * j..2 * j + 2]);
            }
        }
        close(
            name,
            "jnt_range (limited joints)",
            &ours_r,
            &serde_json::json!(gold_r),
        );
        exact("jnt_margin", &m.jnt_margin, &a["jnt_margin"]);
        exact("jnt_solref", &m.jnt_solref, &a["jnt_solref"]);
        exact("jnt_solimp", &m.jnt_solimp, &a["jnt_solimp"]);

        // a ball joint's range is [0, max_angle]
        for j in 0..m.njnt {
            if m.jnt_type[j] == JointType::Ball && m.jnt_limited[j] {
                assert_eq!(m.jnt_range[2 * j], 0.0);
                assert!(m.jnt_range[2 * j + 1] > 0.0);
            }
        }

        // dofs: friction loss, and the approximate inverse inertia at qpos0
        exact(
            "dof_frictionloss",
            &m.dof_frictionloss,
            &a["dof_frictionloss"],
        );
        exact("dof_solref", &m.dof_solref, &a["dof_solref"]);
        exact("dof_solimp", &m.dof_solimp, &a["dof_solimp"]);
        close(
            name,
            "dof_invweight0",
            &m.dof_invweight0,
            &a["dof_invweight0"],
        );
        close(
            name,
            "body_invweight0",
            &m.body_invweight0,
            &a["body_invweight0"],
        );
        close(
            name,
            "meaninertia",
            &[m.meaninertia],
            &serde_json::json!([f(&a["meaninertia"])]),
        );

        // fixed tendons
        assert_eq!(
            usizes(&m.tendon_adr),
            ints(&a["tendon_adr"]),
            "{name} tendon_adr"
        );
        assert_eq!(
            usizes(&m.tendon_num),
            ints(&a["tendon_num"]),
            "{name} tendon_num"
        );
        assert_eq!(
            usizes(&m.wrap_objid),
            ints(&a["wrap_objid"]),
            "{name} wrap_objid"
        );
        assert_eq!(
            usizes(&m.ten_j_rowadr),
            ints(&a["ten_J_rowadr"]),
            "{name} ten_J_rowadr"
        );
        assert_eq!(
            usizes(&m.ten_j_rownnz),
            ints(&a["ten_J_rownnz"]),
            "{name} ten_J_rownnz"
        );
        assert_eq!(
            usizes(&m.ten_j_colind),
            ints(&a["ten_J_colind"]),
            "{name} ten_J_colind"
        );
        assert_eq!(
            bools(&m.tendon_limited),
            ints(&a["tendon_limited"]),
            "{name} tendon_limited"
        );
        exact("wrap_prm", &m.wrap_prm, &a["wrap_prm"]);
        let tr = farr(&a["tendon_range"]);
        let mut ours_t = Vec::new();
        let mut gold_t = Vec::new();
        for t in 0..m.ntendon {
            if m.tendon_limited[t] {
                ours_t.extend_from_slice(&m.tendon_range[2 * t..2 * t + 2]);
                gold_t.extend_from_slice(&tr[2 * t..2 * t + 2]);
            }
        }
        assert_eq!(ours_t, gold_t, "{name} tendon_range");
        exact("tendon_margin", &m.tendon_margin, &a["tendon_margin"]);
        exact(
            "tendon_frictionloss",
            &m.tendon_frictionloss,
            &a["tendon_frictionloss"],
        );
        exact(
            "tendon_solref_lim",
            &m.tendon_solref_lim,
            &a["tendon_solref_lim"],
        );
        exact(
            "tendon_solimp_lim",
            &m.tendon_solimp_lim,
            &a["tendon_solimp_lim"],
        );
        exact(
            "tendon_solref_fri",
            &m.tendon_solref_fri,
            &a["tendon_solref_fri"],
        );
        exact(
            "tendon_solimp_fri",
            &m.tendon_solimp_fri,
            &a["tendon_solimp_fri"],
        );
        close(
            name,
            "tendon_invweight0",
            &m.tendon_invweight0,
            &a["tendon_invweight0"],
        );
    }
}

#[test]
fn the_solver_options_match_mujoco() {
    for which in CMODELS {
        let g = cgolden(which);
        let o = &g["option"];
        let c = ccompile::<f64>(which);
        let opt = &c.model.opt;
        let name = which.name();
        // MuJoCo's solver codes: PGS 0, CG 1, Newton 2
        let code = match opt.solver {
            PrimalSolver::Cg => 1,
            PrimalSolver::Newton => 2,
        };
        assert_eq!(code, o["solver"].as_i64().unwrap(), "{name} solver");
        assert_eq!(
            opt.iterations as i64,
            o["iterations"].as_i64().unwrap(),
            "{name}"
        );
        assert_eq!(opt.tolerance, f(&o["tolerance"]), "{name}");
        assert_eq!(
            opt.ls_iterations as i64,
            o["ls_iterations"].as_i64().unwrap(),
            "{name}"
        );
        assert_eq!(opt.ls_tolerance, f(&o["ls_tolerance"]), "{name}");
        assert_eq!(opt.impratio, f(&o["impratio"]), "{name}");
        assert_eq!(c.model.timestep, f(&o["timestep"]), "{name}");
        let cone = match opt.cone {
            sim_physics::Cone::Pyramidal => 0,
            sim_physics::Cone::Elliptic => 1,
        };
        assert_eq!(cone, o["cone"].as_i64().unwrap(), "{name} cone");
    }
}

/// The golden models are not degenerate: the comparison above would pass vacuously
/// on a model with nothing in it.
#[test]
fn the_golden_models_carry_non_default_soft_constraint_parameters() {
    let humanoid = ccompile::<f64>(CWhich::Humanoid).model;
    // MuJoCo's humanoid: limits on, the limit impedance `0 .99 .01`, tendons, no friction loss
    assert_eq!(humanoid.jnt_limited.iter().filter(|&&l| l).count(), 21);
    assert_eq!(humanoid.ntendon, 2);
    assert!(humanoid.dof_frictionloss.iter().all(|&x| x == 0.0));
    assert!(humanoid.nefc_max >= 21 * 2 + 4);

    let c = ccompile::<f64>(CWhich::Constrained).model;
    // 7 dofs have friction loss (a ball joint's three among them), 5 joints are limited
    // (a ball joint's limit is one row, the other four joints' two each), the tendon has
    // one friction row and two limit rows
    assert_eq!(c.dof_frictionloss.iter().filter(|&&x| x != 0.0).count(), 7);
    assert_eq!(c.jnt_limited, vec![true; 5]);
    assert_eq!(c.nefc_max, 7 + (4 * 2 + 1) + (1 + 2));
    // one limit uses the direct format (both entries negative), one has a time
    // constant below twice the timestep (the integrator-safety rule acts on it)
    assert!(c.jnt_solref.chunks(2).any(|s| s[0] < 0.0 && s[1] < 0.0));
    assert!(
        c.jnt_solref
            .chunks(2)
            .any(|s| s[0] > 0.0 && s[0] < 2.0 * c.timestep)
    );
    assert!(c.jnt_margin.iter().all(|&x| x > 0.0));
    // the cart is MuJoCo's "simple" slide body: 1 / mass, not 1 / (mass + armature)
    assert_eq!(c.dof_invweight0[0], 0.5);
    assert_eq!(c.body_invweight0[2], 0.5);
    assert_eq!(c.opt.solver, PrimalSolver::Cg);
    assert_eq!(c.opt.iterations, 80);

    let zoo = ccompile::<f64>(CWhich::Zoo).model;
    // no friction-loss or limit row: what is left of `nefc_max` is the contact slots of the
    // zoo's candidate pairs (phase 1c-ii), `2 (dim - 1)` rows each (1 for dim 1)
    let contact_rows: usize = zoo
        .candidates
        .iter()
        .map(|c| c.slot_count * if c.condim == 1 { 1 } else { 2 * (c.condim - 1) })
        .sum();
    assert_eq!(zoo.nefc_max, contact_rows);
    assert_eq!(zoo.ntendon, 0);
}

/// `Model<f32>` is the `f64` model rounded once: every new array too.
#[test]
fn the_f32_model_is_the_f64_model_rounded() {
    for which in CMODELS {
        let m64 = ccompile::<f64>(which).model;
        let m32 = ccompile::<f32>(which).model;
        let r = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
        assert_eq!(
            m32.dof_invweight0,
            r(&m64.dof_invweight0),
            "{}",
            which.name()
        );
        assert_eq!(m32.tendon_invweight0, r(&m64.tendon_invweight0));
        assert_eq!(m32.body_invweight0, r(&m64.body_invweight0));
        assert_eq!(m32.jnt_solimp, r(&m64.jnt_solimp));
        assert_eq!(m32.dof_solref, r(&m64.dof_solref));
        assert_eq!(m32.meaninertia, m64.meaninertia as f32);
        assert_eq!(m32.opt.tolerance, m64.opt.tolerance as f32);
        assert_eq!(m32.nefc_max, m64.nefc_max);
        assert_eq!(m32.jnt_limited, m64.jnt_limited);
    }
}
