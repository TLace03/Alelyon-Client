//! What the constraint tests (phase 1c-i) share: the three models of the constraint
//! golden files, their states, and the comparison of one state with MuJoCo.
//!
//! `fixtures/{humanoid,constrained,zoo}_constraints_golden.json` (written by
//! `tools/sim_constraints_mujoco_golden.py`) hold, for eight states of each model,
//! what MuJoCo 3.14.0 computed with contacts and islands off and the constraints on:
//! the constraint rows, `qacc_smooth`, the Newton and CG solutions, the converged
//! optimum, one Euler and one RK4 step, and a 100-step trajectory.

#![allow(dead_code)]

use serde_json::Value;
use sim_physics::{
    ConstraintState, ConstraintType, Data, Model, PrimalSolver, Real, forward, step,
};
use sim_scene::Scene;

use super::{
    Compiled, ErrStat, Report, compare, compile_constrained, f, farr, fixtures, fnv1a64, frows,
    humanoid_dir, integrator_of, load_scene, physics_qpos, scene_qpos, widen,
};

/// The three models of the constraint golden files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CWhich {
    /// MuJoCo's humanoid: hinge limits with `solimplimit`, two limited tendons.
    Humanoid,
    /// Ours: every kind of soft constraint with non-default parameters.
    Constrained,
    /// The phase-1b zoo: no constraint at all.
    Zoo,
}

pub const CMODELS: [CWhich; 3] = [CWhich::Humanoid, CWhich::Constrained, CWhich::Zoo];

impl CWhich {
    pub fn name(self) -> &'static str {
        match self {
            CWhich::Humanoid => "humanoid",
            CWhich::Constrained => "constrained",
            CWhich::Zoo => "zoo",
        }
    }

    pub fn xml_path(self) -> std::path::PathBuf {
        match self {
            CWhich::Humanoid => humanoid_dir().join("humanoid.xml"),
            CWhich::Constrained => humanoid_dir().join("constrained.xml"),
            CWhich::Zoo => fixtures().join("zoo.xml"),
        }
    }

    pub fn golden_path(self) -> std::path::PathBuf {
        fixtures().join(format!("{}_constraints_golden.json", self.name()))
    }
}

/// The golden file of `which`, checked against the bytes of its XML.
pub fn cgolden(which: CWhich) -> Value {
    let g = super::read_json(&which.golden_path());
    let bytes = std::fs::read(which.xml_path()).expect("xml bytes");
    assert_eq!(g["xml"]["bytes"].as_u64(), Some(bytes.len() as u64));
    assert_eq!(
        g["xml"]["fnv1a64"].as_str(),
        Some(format!("{:016x}", fnv1a64(&bytes)).as_str()),
        "{} changed since its golden file was generated: rerun tools/sim_constraints_mujoco_golden.py",
        which.name()
    );
    g
}

pub fn cscene(which: CWhich) -> Scene {
    load_scene(&which.xml_path())
}

/// The model with its constraints on, and checked against the golden file's sizes
/// and options.
pub fn ccompile<R: Real>(which: CWhich) -> Compiled<R> {
    let scene = cscene(which);
    let g = cgolden(which);
    assert_eq!(
        scene.integrator,
        integrator_of(g["integrator"].as_str().unwrap())
    );
    let mut c = compile_constrained::<R>(scene);
    // the 1c-i golden files were generated with mjDSBL_CONTACT set (phase 1c-ii adds contacts)
    c.model.disable.contact = true;
    let counts = &g["counts"];
    assert_eq!(c.model.nbody as u64, counts["nbody"].as_u64().unwrap());
    assert_eq!(c.model.njnt as u64, counts["njnt"].as_u64().unwrap());
    assert_eq!(c.model.nq as u64, counts["nq"].as_u64().unwrap());
    assert_eq!(c.model.nv as u64, counts["nv"].as_u64().unwrap());
    assert_eq!(c.model.nu as u64, counts["nu"].as_u64().unwrap());
    assert_eq!(c.model.ntendon as u64, counts["ntendon"].as_u64().unwrap());
    c
}

/// A `Data` holding the golden state's inputs, `qacc_warmstart` included.
pub fn cdata<R: Real>(model: &Model<R>, state: &Value) -> Data<R> {
    let mut d = Data::new(model);
    d.qpos = physics_qpos(model, &farr(&state["qpos"]));
    d.qvel = farr(&state["qvel"])
        .iter()
        .map(|&x| R::from_f64(x))
        .collect();
    d.ctrl = farr(&state["ctrl"])
        .iter()
        .map(|&x| R::from_f64(x))
        .collect();
    d.qacc_warmstart = farr(&state["qacc_warmstart"])
        .iter()
        .map(|&x| R::from_f64(x))
        .collect();
    d
}

pub fn states(g: &Value) -> &Vec<Value> {
    g["states"].as_array().unwrap()
}

pub fn solver_of(name: &str) -> PrimalSolver {
    match name {
        "newton" => PrimalSolver::Newton,
        "cg" => PrimalSolver::Cg,
        other => panic!("unknown solver {other}"),
    }
}

pub const SOLVERS: [&str; 2] = ["newton", "cg"];

pub fn type_code(t: ConstraintType) -> i64 {
    i64::from(t.code())
}

pub fn state_code(s: ConstraintState) -> i64 {
    i64::from(s.code())
}

pub fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

/// The first `n` entries of a `Data` array, widened.
pub fn head<R: Real>(a: &[R], n: usize) -> Vec<f64> {
    a[..n].iter().map(|x| x.to_f64()).collect()
}

/// Records `err` for `name` in `report` with the tolerance `rtol` and floor `floor`.
pub fn rec(report: &mut Report, name: &str, err: ErrStat, rtol: f64, floor: f64) {
    report.record(name, err, rtol, floor);
}

/// Runs `forward` on the golden state `state` of `m` (with the solver `solver`) and
/// returns the data.
pub fn run_forward<R: Real>(m: &Model<R>, state: &Value, solver: PrimalSolver) -> Data<R> {
    let mut mm = m.clone();
    mm.opt.solver = solver;
    let mut d = cdata(&mm, state);
    forward(&mm, &mut d);
    d
}

/// `n` steps of `m` (own integrator, or `integrator`) from the golden state, with
/// the state's `qacc_warmstart`, solver `solver`.
pub fn run_steps<R: Real>(
    m: &Model<R>,
    state: &Value,
    solver: PrimalSolver,
    integrator: sim_physics::Integrator,
    n: usize,
) -> Data<R> {
    let mut mm = m.clone();
    mm.opt.solver = solver;
    mm.integrator = integrator;
    let mut d = cdata(&mm, state);
    for _ in 0..n {
        step(&mm, &mut d);
    }
    d
}

/// The relative error of `ours` against `reference`, in the harness's terms.
pub fn cmp(ours: &[f64], reference: &[f64]) -> ErrStat {
    compare(ours, reference)
}

/// The golden `efc_J` as a flat vector.
pub fn golden_rows(v: &Value) -> Vec<f64> {
    frows(v)
}

/// A scene-layout `qpos` of `d`, widened.
pub fn qpos_of<R: Real>(m: &Model<R>, d: &Data<R>) -> Vec<f64> {
    scene_qpos(m, &d.qpos)
}

/// The widened `qvel` of `d`.
pub fn qvel_of<R: Real>(d: &Data<R>) -> Vec<f64> {
    widen(&d.qvel)
}

/// A number from the golden file.
pub fn num(v: &Value) -> f64 {
    f(v)
}
