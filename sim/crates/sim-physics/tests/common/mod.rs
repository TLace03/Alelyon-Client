//! What the sim-physics tests share: the models, the golden files, and the
//! comparison functions.
//!
//! Every comparison reports `max_abs_err`, the `array_max_abs` it is relative to,
//! and `rel = max_abs_err / array_max_abs`, and the tests print them as lines
//! starting with `MEASURED`, so the gate run (`cargo test -- --nocapture`) can be
//! collected with `grep MEASURED`.

#![allow(dead_code)]

pub mod cons;
pub mod contacts;

use std::fs;
use std::path::PathBuf;

use serde_json::Value;
use sim_physics::{Data, Integrator, Model, NotModelled, QuatOrder, Real, convert_qpos};
use sim_scene::{Scene, mjcf};

/// FNV-1a, 64 bit: the same function `tools/sim_physics_mujoco_golden.py` uses.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// `crates/sim-physics/tests/fixtures`.
pub fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// The sim-scene fixture directory that holds the vendored `humanoid.xml`
/// (reused by relative path, not copied).
pub fn humanoid_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../sim-scene/tests/fixtures/mujoco")
}

/// The three models of the golden files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which {
    Humanoid,
    Zoo,
}

impl Which {
    pub fn name(self) -> &'static str {
        match self {
            Which::Humanoid => "humanoid",
            Which::Zoo => "zoo",
        }
    }

    pub fn xml_path(self) -> PathBuf {
        match self {
            Which::Humanoid => humanoid_dir().join("humanoid.xml"),
            Which::Zoo => fixtures().join("zoo.xml"),
        }
    }

    pub fn golden_path(self) -> PathBuf {
        fixtures().join(format!("{}_golden.json", self.name()))
    }
}

pub const MODELS: [Which; 2] = [Which::Humanoid, Which::Zoo];

pub fn load_scene(path: &std::path::Path) -> Scene {
    let xml = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    mjcf::load(&xml, path.parent().expect("a parent directory"))
        .unwrap_or_else(|e| panic!("{} imports: {e}", path.display()))
}

pub fn scene_of(which: Which) -> Scene {
    load_scene(&which.xml_path())
}

pub fn read_json(path: &std::path::Path) -> Value {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).expect("golden JSON")
}

pub fn golden_of(which: Which) -> Value {
    let g = read_json(&which.golden_path());
    // the golden file belongs to the XML bytes it was generated from
    let bytes = fs::read(which.xml_path()).expect("xml bytes");
    assert_eq!(g["xml"]["bytes"].as_u64(), Some(bytes.len() as u64));
    assert_eq!(
        g["xml"]["fnv1a64"].as_str(),
        Some(format!("{:016x}", fnv1a64(&bytes)).as_str()),
        "{} changed since its golden file was generated: rerun tools/sim_physics_mujoco_golden.py",
        which.name()
    );
    g
}

pub fn f(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

pub fn farr(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(f).collect()
}

/// A list of equal-length rows flattened.
pub fn frows(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array of rows")
        .iter()
        .flat_map(farr)
        .collect()
}

/// A model and the scene it came from.
pub struct Compiled<R: Real> {
    pub scene: Scene,
    pub model: Model<R>,
    pub not_modelled: Vec<NotModelled>,
}

/// The model of a scene as phase 1b's golden files were generated: MuJoCo with
/// `mjDSBL_CONSTRAINT` set (the smooth dynamics only; the humanoid has joint limits
/// that the phase-1b comparison does not include), so `disable.constraint` is set.
pub fn compile<R: Real>(scene: Scene) -> Compiled<R> {
    let mut c = compile_constrained::<R>(scene);
    c.model.disable.constraint = true;
    // (`disable.constraint` already disables collision, as in MuJoCo; the contact flag says so)
    c.model.disable.contact = true;
    c
}

/// The model of a scene with its constraints on (the default of `Model::compile`).
pub fn compile_constrained<R: Real>(scene: Scene) -> Compiled<R> {
    let (model, not_modelled) = Model::<R>::compile(&scene).expect("compiles");
    Compiled {
        scene,
        model,
        not_modelled,
    }
}

pub fn integrator_of(name: &str) -> Integrator {
    match name {
        "Euler" => Integrator::Euler,
        "RK4" => Integrator::Rk4,
        other => panic!("unknown integrator {other}"),
    }
}

/// A scene-layout `qpos` (quaternions `[x, y, z, w]`) as the physics `qpos`.
pub fn physics_qpos<R: Real>(model: &Model<R>, scene_qpos: &[f64]) -> Vec<R> {
    let src: Vec<R> = scene_qpos.iter().map(|&x| R::from_f64(x)).collect();
    let mut dst = vec![R::ZERO; model.nq];
    convert_qpos(model, &src, QuatOrder::Xyzw, &mut dst, QuatOrder::Wxyz);
    dst
}

/// A physics `qpos` as the scene layout, widened to `f64`.
pub fn scene_qpos<R: Real>(model: &Model<R>, qpos: &[R]) -> Vec<f64> {
    let mut dst = vec![R::ZERO; model.nq];
    convert_qpos(model, qpos, QuatOrder::Wxyz, &mut dst, QuatOrder::Xyzw);
    dst.iter().map(|x| x.to_f64()).collect()
}

/// A `Data` holding the golden state's inputs.
pub fn data_from_state<R: Real>(model: &Model<R>, state: &Value) -> Data<R> {
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
    d
}

pub fn widen<R: Real>(a: &[R]) -> Vec<f64> {
    a.iter().map(|x| x.to_f64()).collect()
}

/// The result of comparing one array with its reference.
#[derive(Clone, Copy, Debug, Default)]
pub struct ErrStat {
    /// The largest `|ours - reference|`.
    pub abs: f64,
    /// The largest `|reference|` of the array.
    pub scale: f64,
}

impl ErrStat {
    pub fn rel(&self) -> f64 {
        if self.scale > 0.0 {
            self.abs / self.scale
        } else {
            self.abs
        }
    }

    /// Whether the error is within `rtol` of the array's max-abs, floored at `floor`
    /// absolute: `abs <= max(rtol * scale, floor)`.
    pub fn within(&self, rtol: f64, floor: f64) -> bool {
        self.abs <= (rtol * self.scale).max(floor)
    }

    /// The error as a multiple of its allowance (<= 1 passes).
    pub fn ratio(&self, rtol: f64, floor: f64) -> f64 {
        self.abs / (rtol * self.scale).max(floor)
    }
}

/// Compares two arrays element by element.
pub fn compare(ours: &[f64], reference: &[f64]) -> ErrStat {
    assert_eq!(ours.len(), reference.len(), "array lengths");
    let mut e = ErrStat::default();
    for (a, b) in ours.iter().zip(reference) {
        let diff = (a - b).abs();
        // NaN compares false everywhere: make it the worst error, never a pass
        e.abs = if diff.is_nan() {
            f64::INFINITY
        } else {
            e.abs.max(diff)
        };
        e.scale = e.scale.max(b.abs());
    }
    e
}

/// Compares two sets of unit quaternions (`[x, y, z, w]` rows) up to sign, row by row.
pub fn compare_quats_up_to_sign(ours: &[f64], reference: &[f64]) -> ErrStat {
    assert_eq!(ours.len(), reference.len());
    let mut e = ErrStat::default();
    for (a, b) in ours.chunks(4).zip(reference.chunks(4)) {
        let same = compare(a, b).abs;
        let neg: Vec<f64> = b.iter().map(|x| -x).collect();
        let flipped = compare(a, &neg).abs;
        e.abs = e.abs.max(same.min(flipped));
        e.scale = e.scale.max(b.iter().fold(0.0f64, |m, x| m.max(x.abs())));
    }
    e
}

/// How bad an error is: its multiple of the allowance, or, for an ungated
/// (infinite) tolerance, its error relative to the array's max-abs.
fn badness(err: &ErrStat, rtol: f64, floor: f64) -> f64 {
    if rtol.is_finite() {
        err.ratio(rtol, floor)
    } else {
        err.rel()
    }
}

/// The worst error of each named array over the states of one run, with the
/// allowance it is held to.
#[derive(Default)]
pub struct Report {
    rows: Vec<(String, ErrStat, f64, f64)>,
}

impl Report {
    /// Records `err` for `name` (keeping the worst by `ratio`), with the tolerance
    /// `rtol` (relative to the array's max-abs) and absolute floor `floor`.
    pub fn record(&mut self, name: &str, err: ErrStat, rtol: f64, floor: f64) {
        match self.rows.iter_mut().find(|r| r.0 == name) {
            Some(row) => {
                if badness(&err, rtol, floor) > badness(&row.1, row.2, row.3) {
                    row.1 = err;
                }
            }
            None => self.rows.push((name.to_string(), err, rtol, floor)),
        }
    }

    /// The worst error recorded for `name`.
    pub fn get(&self, name: &str) -> Option<ErrStat> {
        self.rows.iter().find(|r| r.0 == name).map(|r| r.1)
    }

    /// The largest absolute error recorded for any array (0 when every comparison is exact).
    pub fn max_abs(&self) -> f64 {
        self.rows.iter().map(|r| r.1.abs).fold(0.0, f64::max)
    }

    /// Whether every recorded array is within its allowance.
    pub fn all_within(&self) -> bool {
        self.rows
            .iter()
            .all(|(_, e, rtol, floor)| e.within(*rtol, *floor))
    }

    /// The names of the arrays that are out of tolerance.
    pub fn failures(&self) -> Vec<String> {
        self.rows
            .iter()
            .filter(|(_, e, rtol, floor)| !e.within(*rtol, *floor))
            .map(|(n, ..)| n.clone())
            .collect()
    }

    /// Prints one `MEASURED` line per array.
    pub fn print(&self, precision: &str, model: &str) {
        for (name, e, rtol, floor) in &self.rows {
            println!(
                "MEASURED {precision} {model} {name}: max_abs_err={:e} array_max_abs={:e} rel={:e} tol_rel={:e} floor_abs={:e} {}",
                e.abs,
                e.scale,
                e.rel(),
                rtol,
                floor,
                if !rtol.is_finite() {
                    "ungated"
                } else if e.within(*rtol, *floor) {
                    "ok"
                } else {
                    "OUT OF TOLERANCE"
                }
            );
        }
    }
}

// ---------------------------------------------------------------------------
// the per-state comparison with MuJoCo
// ---------------------------------------------------------------------------

use sim_physics::faults::{Faults, forward_faulted, step_faulted};
use sim_physics::{body_velocity, energy, energy_pos, energy_vel, forward, step};

/// The tolerances of one comparison, each relative to the array's max-abs with an
/// absolute floor. `None` means "measured, not gated".
#[derive(Clone, Copy, Debug)]
pub struct Tol {
    /// `M`, `qfrc_*`, `xpos`, `xquat`, body velocities, energies.
    pub arrays: Option<f64>,
    /// `qacc`.
    pub qacc: Option<f64>,
    /// One Euler step and one RK4 step.
    pub step: Option<f64>,
    /// Trajectory steps 1, 10 and 100.
    pub traj: [Option<f64>; 3],
    /// The absolute floor under every tolerance.
    pub floor: f64,
}

/// The spec's f64 tolerances.
pub const TOL_F64: Tol = Tol {
    arrays: Some(1e-10),
    qacc: Some(1e-9),
    step: Some(1e-10),
    traj: [Some(1e-9), Some(1e-8), Some(1e-6)],
    floor: 1e-12,
};

/// Nothing gated: the f32 run, which is measured.
pub const TOL_MEASURE: Tol = Tol {
    arrays: None,
    qacc: None,
    step: None,
    traj: [None, None, None],
    floor: 0.0,
};

fn rec(report: &mut Report, name: &str, e: ErrStat, rtol: Option<f64>, floor: f64) {
    report.record(name, e, rtol.unwrap_or(f64::INFINITY), floor);
}

fn run_forward<R: Real>(m: &Model<R>, d: &mut Data<R>, faults: Option<&Faults>) {
    match faults {
        None => forward(m, d),
        Some(f) => forward_faulted(m, d, f),
    }
}

fn run_step<R: Real>(m: &Model<R>, d: &mut Data<R>, faults: Option<&Faults>) {
    match faults {
        None => step(m, d),
        Some(f) => step_faulted(m, d, f),
    }
}

/// Compares everything the golden state holds with the engine run in precision
/// `R`, recording the worst error of each array in `report`. `faults` injects a
/// fault (the negative controls); `None` is the public API.
pub fn check_state<R: Real>(
    c: &Compiled<R>,
    own: Integrator,
    state: &Value,
    tol: &Tol,
    faults: Option<&Faults>,
    report: &mut Report,
) {
    let m = &c.model;
    let floor = tol.floor;
    let nbody = m.nbody;

    // ---- the forward pass
    let mut d = data_from_state(m, state);
    run_forward(m, &mut d, faults);
    let g = |key: &str| farr(&state[key]);
    rec(
        report,
        "M",
        compare(&widen(&d.qm), &g("M")),
        tol.arrays,
        floor,
    );
    rec(
        report,
        "qfrc_bias",
        compare(&widen(&d.qfrc_bias), &g("qfrc_bias")),
        tol.arrays,
        floor,
    );
    rec(
        report,
        "qfrc_passive",
        compare(&widen(&d.qfrc_passive), &g("qfrc_passive")),
        tol.arrays,
        floor,
    );
    rec(
        report,
        "qfrc_actuator",
        compare(&widen(&d.qfrc_actuator), &g("qfrc_actuator")),
        tol.arrays,
        floor,
    );
    rec(
        report,
        "qacc",
        compare(&widen(&d.qacc), &g("qacc")),
        tol.qacc,
        floor,
    );
    rec(
        report,
        "xpos",
        compare(&widen(&d.xpos), &frows(&state["xpos"])),
        tol.arrays,
        floor,
    );
    // xquat: [w, x, y, z] inside, compared as [x, y, z, w] up to sign
    let mut xq = Vec::with_capacity(4 * nbody);
    for i in 0..nbody {
        let q = &d.xquat[4 * i..4 * i + 4];
        xq.extend([q[1], q[2], q[3], q[0]].map(|x| x.to_f64()));
    }
    rec(
        report,
        "xquat",
        compare_quats_up_to_sign(&xq, &frows(&state["xquat"])),
        tol.arrays,
        floor,
    );
    // body velocities as MuJoCo's mj_objectVelocity returns them: [angular, linear]
    let mut xv = Vec::with_capacity(6 * nbody);
    for i in 0..nbody {
        xv.extend(body_velocity(m, &d, i).map(|x| x.to_f64()));
    }
    rec(
        report,
        "xvel",
        compare(&xv, &frows(&state["xvel"])),
        tol.arrays,
        floor,
    );
    let golden_energy = farr(&state["energy"]);
    let pot = energy_pos(m, &mut d).to_f64();
    let kin = energy_vel(m, &mut d).to_f64();
    rec(
        report,
        "energy_pos",
        compare(&[pot], &golden_energy[0..1]),
        tol.arrays,
        floor,
    );
    rec(
        report,
        "energy_vel",
        compare(&[kin], &golden_energy[1..2]),
        tol.arrays,
        floor,
    );

    // ---- one Euler step and one RK4 step from the same state
    for (label, integ) in [
        ("euler_step", Integrator::Euler),
        ("rk4_step", Integrator::Rk4),
    ] {
        let mut mm = m.clone();
        mm.integrator = integ;
        let mut d = data_from_state(&mm, state);
        run_step(&mm, &mut d, faults);
        let r = &state[label];
        rec(
            report,
            &format!("{label}.qpos"),
            compare(&scene_qpos(&mm, &d.qpos), &farr(&r["qpos"])),
            tol.step,
            floor,
        );
        rec(
            report,
            &format!("{label}.qvel"),
            compare(&widen(&d.qvel), &farr(&r["qvel"])),
            tol.step,
            floor,
        );
        rec(
            report,
            &format!("{label}.time"),
            compare(&[d.time.to_f64()], &[f(&r["time"])]),
            tol.step,
            floor,
        );
    }

    // ---- a 100-step trajectory with the model's own integrator
    let mut mm = m.clone();
    mm.integrator = own;
    let mut d = data_from_state(&mm, state);
    let traj = &state["trajectory"];
    let steps: Vec<usize> = traj["steps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let mut done = 0usize;
    for (k, &n) in steps.iter().enumerate() {
        while done < n {
            run_step(&mm, &mut d, faults);
            done += 1;
        }
        let qpos = &traj["qpos"][k];
        let qvel = &traj["qvel"][k];
        rec(
            report,
            &format!("trajectory[{n}].qpos"),
            compare(&scene_qpos(&mm, &d.qpos), &farr(qpos)),
            tol.traj[k],
            floor,
        );
        rec(
            report,
            &format!("trajectory[{n}].qvel"),
            compare(&widen(&d.qvel), &farr(qvel)),
            tol.traj[k],
            floor,
        );
    }
}

/// Runs [`check_state`] over every state of `which`.
pub fn check_model<R: Real>(which: Which, tol: &Tol, faults: Option<&Faults>) -> Report {
    let golden = golden_of(which);
    let scene = scene_of(which);
    let own = integrator_of(golden["integrator"].as_str().unwrap());
    assert_eq!(
        scene.integrator, own,
        "the importer reads the integrator the XML names"
    );
    let c = compile::<R>(scene);
    // the model agrees with the golden file on its size
    let counts = &golden["counts"];
    assert_eq!(c.model.nbody as u64, counts["nbody"].as_u64().unwrap());
    assert_eq!(c.model.njnt as u64, counts["njnt"].as_u64().unwrap());
    assert_eq!(c.model.nq as u64, counts["nq"].as_u64().unwrap());
    assert_eq!(c.model.nv as u64, counts["nv"].as_u64().unwrap());
    assert_eq!(c.model.nu as u64, counts["nu"].as_u64().unwrap());
    assert_eq!(
        c.model.timestep.to_f64(),
        R::from_f64(f(&golden["option"]["timestep"])).to_f64()
    );
    let mut report = Report::default();
    for state in golden["states"].as_array().unwrap() {
        check_state(&c, own, state, tol, faults, &mut report);
    }
    report
}

// ---------------------------------------------------------------------------
// the double pendulum series
// ---------------------------------------------------------------------------

pub struct Series {
    /// `(step, energy)` of the engine at every recorded step.
    pub ours: Vec<(usize, f64)>,
    /// MuJoCo's energy at the same steps.
    pub golden: Vec<f64>,
    /// The largest difference of `qpos` and `qvel` from MuJoCo's, per recorded step.
    pub state_err: Vec<f64>,
}

pub fn pendulum_series<R: Real>(xml: &str, golden_file: &str) -> Series {
    let path = fixtures().join(xml);
    let scene = load_scene(&path);
    let g = read_json(&fixtures().join(golden_file));
    // the golden file belongs to the XML bytes
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(
        g["xml"]["fnv1a64"].as_str(),
        Some(format!("{:016x}", fnv1a64(&bytes)).as_str())
    );
    assert_eq!(
        scene.integrator,
        integrator_of(g["integrator"].as_str().unwrap())
    );
    let c = compile::<R>(scene);
    assert!(c.not_modelled.is_empty(), "{:?}", c.not_modelled);
    let m = &c.model;
    let steps = g["steps"].as_u64().unwrap() as usize;
    let every = g["every"].as_u64().unwrap() as usize;
    let series = g["series"].as_array().unwrap();
    assert_eq!(series.len(), steps / every + 1);

    let mut d = Data::new(m);
    d.qpos = farr(&g["qpos0"]).iter().map(|&x| R::from_f64(x)).collect();
    let mut out = Series {
        ours: Vec::new(),
        golden: Vec::new(),
        state_err: Vec::new(),
    };
    let mut record = |d: &mut Data<R>, n: usize| {
        let row = &series[n / every];
        assert_eq!(row["step"].as_u64().unwrap() as usize, n);
        forward(m, d);
        let e = energy(m, d).to_f64();
        let g_energy = farr(&row["energy"]);
        out.ours.push((n, e));
        out.golden.push(g_energy[0] + g_energy[1]);
        let q = compare(&widen(&d.qpos), &farr(&row["qpos"])).abs;
        let v = compare(&widen(&d.qvel), &farr(&row["qvel"])).abs;
        out.state_err.push(q.max(v));
    };
    record(&mut d, 0);
    for n in 1..=steps {
        step(m, &mut d);
        if n % every == 0 {
            record(&mut d, n);
        }
    }
    out
}

/// Worst `|E_ours - E_mujoco| / |E_mujoco|` over the recorded steps, and where.
pub fn worst_energy_error(s: &Series) -> (f64, usize) {
    let mut worst = (0.0f64, 0usize);
    for ((n, e), g) in s.ours.iter().zip(&s.golden) {
        let rel = (e - g).abs() / g.abs();
        if rel.is_nan() || rel > worst.0 {
            worst = (if rel.is_nan() { f64::INFINITY } else { rel }, *n);
        }
    }
    worst
}

pub fn drift(energies: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = energies.collect();
    (v[v.len() - 1] - v[0]) / v[0].abs()
}

pub fn max_deviation(energies: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = energies.collect();
    v.iter().map(|e| (e - v[0]).abs()).fold(0.0, f64::max) / v[0].abs()
}
