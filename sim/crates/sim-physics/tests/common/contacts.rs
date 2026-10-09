//! What the contact tests (phase 1c-ii) share: the models of the contact golden files, their
//! variants, and the comparison of a contact list and of the constraint structure with MuJoCo.
//!
//! `fixtures/<model>_<variant>_contacts_golden.json` (written by
//! `tools/sim_contacts_mujoco_golden.py`) hold, for the states of each model, what MuJoCo
//! 3.14.0 computed with islands off and contacts and constraints on: the contact list, the
//! constraint rows, the Newton and CG solutions, the converged optimum, one Euler and one RK4
//! step and a 100-step trajectory. The variants are `pyramidal` and `elliptic` (the cone, set on
//! `scene.options` before `Model::compile`), and for the zoo `midphase` and `filterparent`
//! (structure only: the flag is set on the compiled model and the candidate list rebuilt).

#![allow(dead_code)]

use serde_json::Value;
use sim_physics::{Cone, Data, Integrator, Model, PrimalSolver, Real};
use sim_scene::Scene;

use super::cons::{golden_rows, head, ints, rec, type_code};
use super::{
    Compiled, ErrStat, Report, compare, compare_quats_up_to_sign, compile_constrained, f, farr,
    fixtures, fnv1a64, humanoid_dir, integrator_of, load_scene, physics_qpos, scene_qpos, widen,
};

/// The models of the contact golden files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TWhich {
    /// A plane, a resting sphere and a sliding sphere.
    Sphere,
    /// One box on a plane.
    Box,
    /// Three boxes, each smaller than the one below.
    Stack,
    /// Five capsules.
    Capsules,
    /// Eight boxes, the comparison's pile in small.
    Pile,
    /// MuJoCo's humanoid.
    Humanoid,
    /// The contact zoo: every collider and every parameter.
    Zoo,
}

/// Every model of the contact golden files.
pub const TMODELS: [TWhich; 7] = [
    TWhich::Sphere,
    TWhich::Box,
    TWhich::Stack,
    TWhich::Capsules,
    TWhich::Pile,
    TWhich::Humanoid,
    TWhich::Zoo,
];

/// A variant of a golden file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// The pyramidal cone (MuJoCo's default).
    Pyramidal,
    /// The elliptic cone.
    Elliptic,
    /// Pyramidal, with `mjDSBL_MIDPHASE` set (structure only, the zoo).
    Midphase,
    /// Pyramidal, with `mjDSBL_FILTERPARENT` set (structure only, the zoo).
    Filterparent,
}

/// The two cone variants.
pub const CONES: [Variant; 2] = [Variant::Pyramidal, Variant::Elliptic];

impl Variant {
    /// The file-name part.
    pub fn name(self) -> &'static str {
        match self {
            Variant::Pyramidal => "pyramidal",
            Variant::Elliptic => "elliptic",
            Variant::Midphase => "midphase",
            Variant::Filterparent => "filterparent",
        }
    }

    /// The friction cone of the variant.
    pub fn cone(self) -> Cone {
        match self {
            Variant::Elliptic => Cone::Elliptic,
            _ => Cone::Pyramidal,
        }
    }
}

impl TWhich {
    /// The model's name in the file names.
    pub fn name(self) -> &'static str {
        match self {
            TWhich::Sphere => "sphere",
            TWhich::Box => "box",
            TWhich::Stack => "stack",
            TWhich::Capsules => "capsules",
            TWhich::Pile => "pile",
            TWhich::Humanoid => "humanoid",
            TWhich::Zoo => "zoo",
        }
    }

    /// The XML of the model.
    pub fn xml_path(self) -> std::path::PathBuf {
        let file = match self {
            TWhich::Sphere => "contact_sphere.xml",
            TWhich::Box => "contact_box.xml",
            TWhich::Stack => "contact_stack.xml",
            TWhich::Capsules => "contact_capsules.xml",
            TWhich::Pile => "contact_pile.xml",
            TWhich::Humanoid => "humanoid.xml",
            TWhich::Zoo => "contact_zoo.xml",
        };
        humanoid_dir().join(file)
    }

    /// The golden file of a variant.
    pub fn golden_path(self, v: Variant) -> std::path::PathBuf {
        fixtures().join(format!("{}_{}_contacts_golden.json", self.name(), v.name()))
    }
}

/// The golden file of `which` and `v`, checked against the bytes of its XML.
pub fn tgolden(which: TWhich, v: Variant) -> Value {
    let g = super::read_json(&which.golden_path(v));
    let bytes = std::fs::read(which.xml_path()).expect("xml bytes");
    assert_eq!(g["xml"]["bytes"].as_u64(), Some(bytes.len() as u64));
    assert_eq!(
        g["xml"]["fnv1a64"].as_str(),
        Some(format!("{:016x}", fnv1a64(&bytes)).as_str()),
        "{} changed since its golden file was generated: rerun tools/sim_contacts_mujoco_golden.py",
        which.name()
    );
    assert_eq!(g["mujoco_version"], "3.14.0");
    g
}

/// The scene of `which` with the cone of `v` (set on its options, as MuJoCo's `m.opt.cone` is).
pub fn tscene(which: TWhich, v: Variant) -> Scene {
    let mut s = load_scene(&which.xml_path());
    s.options.cone = v.cone();
    s
}

/// The model of `which` and `v`, checked against the golden file's sizes and options. The
/// structure-only variants set their flag on the model and rebuild the candidate list.
pub fn tcompile<R: Real>(which: TWhich, v: Variant) -> Compiled<R> {
    let g = tgolden(which, v);
    let scene = tscene(which, v);
    assert_eq!(
        scene.integrator,
        integrator_of(g["integrator"].as_str().unwrap())
    );
    // the importer reads the options the XML names (the cone is the variant's)
    assert_eq!(
        scene.options.impratio,
        g["option"]["impratio"].as_f64().unwrap()
    );
    let mut c = if matches!(v, Variant::Midphase | Variant::Filterparent) {
        // the flag goes on the f64 model, the list is rebuilt there, and the f32 model is
        // that rounded once
        let (mut m64, nm) = Model::<f64>::compile(&scene).expect("compiles");
        match v {
            Variant::Midphase => m64.disable.midphase = true,
            Variant::Filterparent => m64.disable.filterparent = true,
            _ => unreachable!(),
        }
        m64.rebuild_contact_pairs();
        Compiled {
            scene,
            model: m64.rounded_to::<R>(),
            not_modelled: nm,
        }
    } else {
        compile_constrained::<R>(scene)
    };
    let counts = &g["counts"];
    assert_eq!(c.model.nbody as u64, counts["nbody"].as_u64().unwrap());
    assert_eq!(c.model.njnt as u64, counts["njnt"].as_u64().unwrap());
    assert_eq!(c.model.ngeom as u64, counts["ngeom"].as_u64().unwrap());
    assert_eq!(c.model.nq as u64, counts["nq"].as_u64().unwrap());
    assert_eq!(c.model.nv as u64, counts["nv"].as_u64().unwrap());
    assert_eq!(c.model.nu as u64, counts["nu"].as_u64().unwrap());
    // a model with no ellipsoid, mesh or cylinder pair outside the ported ones lists no
    // `Collision` entry
    assert!(
        !c.not_modelled
            .iter()
            .any(|n| matches!(n, sim_physics::NotModelled::Collision { .. })),
        "{}: {:?}",
        which.name(),
        c.not_modelled
    );
    c.model.disable.contact = false;
    c
}

/// A `Data` holding the golden state's inputs, `qacc_warmstart` included.
pub fn tdata<R: Real>(model: &Model<R>, state: &Value) -> Data<R> {
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

/// The states of a golden file.
pub fn tstates(g: &Value) -> &Vec<Value> {
    g["states"].as_array().unwrap()
}

/// The golden file's solver block name as a solver.
pub fn tsolver_of(name: &str) -> PrimalSolver {
    match name {
        "newton" => PrimalSolver::Newton,
        "cg" => PrimalSolver::Cg,
        other => panic!("unknown solver {other}"),
    }
}

/// The two solvers' block names.
pub const TSOLVERS: [&str; 2] = ["newton", "cg"];

/// Integers of a golden array.
pub fn tints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

/// The first `n` entries of a `Data` array, widened.
pub fn thead<R: Real>(a: &[R], n: usize) -> Vec<f64> {
    a[..n].iter().map(|x| x.to_f64()).collect()
}

/// The rows of a golden matrix flattened.
pub fn trows(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().flat_map(farr).collect()
}

/// A model with the cone of `v` set, run through `forward` on a golden state.
pub fn tforward<R: Real>(m: &Model<R>, state: &Value, solver: PrimalSolver) -> Data<R> {
    let mut mm = m.clone();
    mm.opt.solver = solver;
    let mut d = tdata(&mm, state);
    sim_physics::forward(&mm, &mut d);
    assert_eq!(d.warning_collision_overflow, 0, "box-box overflow");
    d
}

/// The contact list of `d` as `(geom pair, dist, pos, frame, includemargin, friction, solref,
/// solreffriction, solimp, dim, exclude, efc_address)` per contact, field by field (floats
/// widened).
pub struct ContactRow {
    /// The geom ids.
    pub geom: [usize; 2],
    /// The distance.
    pub dist: f64,
    /// The position.
    pub pos: Vec<f64>,
    /// The frame (9).
    pub frame: Vec<f64>,
    /// `includemargin`.
    pub includemargin: f64,
    /// Friction (5).
    pub friction: Vec<f64>,
    /// `solref` (2).
    pub solref: Vec<f64>,
    /// `solreffriction` (2).
    pub solreffriction: Vec<f64>,
    /// `solimp` (5).
    pub solimp: Vec<f64>,
    /// The dimension.
    pub dim: usize,
    /// The exclude flag.
    pub exclude: i64,
    /// The first constraint row, or -1.
    pub efc_address: i64,
}

/// Our contact `i`.
pub fn our_contact<R: Real>(d: &Data<R>, i: usize) -> ContactRow {
    let w = |a: &[R], n: usize| -> Vec<f64> {
        a[n * i..n * i + n].iter().map(|x| x.to_f64()).collect()
    };
    ContactRow {
        geom: [d.contact_geom[2 * i], d.contact_geom[2 * i + 1]],
        dist: d.contact_dist[i].to_f64(),
        pos: w(&d.contact_pos, 3),
        frame: w(&d.contact_frame, 9),
        includemargin: d.contact_includemargin[i].to_f64(),
        friction: w(&d.contact_friction, 5),
        solref: w(&d.contact_solref, 2),
        solreffriction: w(&d.contact_solreffriction, 2),
        solimp: w(&d.contact_solimp, 5),
        dim: d.contact_dim[i],
        exclude: i64::from(d.contact_exclude[i]),
        efc_address: i64::from(d.contact_efc_address[i]),
    }
}

/// MuJoCo's contact `i` of a golden state.
pub fn golden_contact(c: &Value) -> ContactRow {
    let arr = |k: &str| farr(&c[k]);
    let geom = tints(&c["geom"]);
    ContactRow {
        geom: [geom[0] as usize, geom[1] as usize],
        dist: c["dist"].as_f64().unwrap(),
        pos: arr("pos"),
        frame: arr("frame"),
        includemargin: c["includemargin"].as_f64().unwrap(),
        friction: arr("friction"),
        solref: arr("solref"),
        solreffriction: arr("solreffriction"),
        solimp: arr("solimp"),
        dim: c["dim"].as_u64().unwrap() as usize,
        exclude: c["exclude"].as_i64().unwrap(),
        efc_address: c["efc_address"].as_i64().unwrap(),
    }
}

/// The result of comparing the two contact lists of one state.
pub struct ListCompare {
    /// The first difference of an integer field or of the count, in words (`None`: equal).
    pub first_integer_difference: Option<String>,
    /// The worst error of each float field, in field order: `dist`, `pos`, `frame`,
    /// `includemargin`, `friction`, `solref`, `solreffriction`, `solimp`.
    pub floats: Vec<(&'static str, ErrStat)>,
}

/// Compares our contact list with MuJoCo's: integers (`ncon`, geoms, `dim`, `exclude`,
/// `efc_address`) exactly, in order, floats by error.
pub fn compare_lists<R: Real>(d: &Data<R>, golden: &Value) -> ListCompare {
    let gc = golden["contacts"].as_array().unwrap();
    let mut first = None;
    if d.ncon != gc.len() {
        first = Some(format!("ncon {} vs MuJoCo's {}", d.ncon, gc.len()));
    }
    let names = [
        "dist",
        "pos",
        "frame",
        "includemargin",
        "friction",
        "solref",
        "solreffriction",
        "solimp",
    ];
    let mut ours: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
    let mut theirs: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
    for (i, g) in gc.iter().enumerate().take(d.ncon) {
        let a = our_contact(d, i);
        let b = golden_contact(g);
        if first.is_none() {
            if a.geom != b.geom {
                first = Some(format!(
                    "contact {i}: geoms {:?} vs MuJoCo's {:?}",
                    a.geom, b.geom
                ));
            } else if a.dim != b.dim {
                first = Some(format!("contact {i}: dim {} vs {}", a.dim, b.dim));
            } else if a.exclude != b.exclude {
                first = Some(format!(
                    "contact {i}: exclude {} vs {}",
                    a.exclude, b.exclude
                ));
            } else if a.efc_address != b.efc_address {
                first = Some(format!(
                    "contact {i}: efc_address {} vs {}",
                    a.efc_address, b.efc_address
                ));
            }
        }
        ours[0].push(a.dist);
        theirs[0].push(b.dist);
        ours[1].extend(&a.pos);
        theirs[1].extend(&b.pos);
        ours[2].extend(&a.frame);
        theirs[2].extend(&b.frame);
        ours[3].push(a.includemargin);
        theirs[3].push(b.includemargin);
        ours[4].extend(&a.friction);
        theirs[4].extend(&b.friction);
        ours[5].extend(&a.solref);
        theirs[5].extend(&b.solref);
        ours[6].extend(&a.solreffriction);
        theirs[6].extend(&b.solreffriction);
        ours[7].extend(&a.solimp);
        theirs[7].extend(&b.solimp);
    }
    let floats = names
        .iter()
        .enumerate()
        .map(|(k, &n)| (n, compare(&ours[k], &theirs[k])))
        .collect();
    ListCompare {
        first_integer_difference: first,
        floats,
    }
}

/// The tolerance of the contact structure comparisons: relative to the array's largest absolute
/// value, with an absolute floor (the 1c-i ones).
pub const STRUCTURE_RTOL: f64 = 1e-12;
/// The absolute floor under [`STRUCTURE_RTOL`].
pub const STRUCTURE_FLOOR: f64 = 1e-14;

/// Compares two arrays, an infinite error when their lengths differ (a different number of rows
/// or contacts is the worst error, not a panic: the negative controls run through this).
pub fn safe_cmp(ours: &[f64], reference: &[f64]) -> ErrStat {
    if ours.len() != reference.len() {
        return ErrStat {
            abs: f64::INFINITY,
            scale: reference.iter().fold(0.0f64, |m, x| m.max(x.abs())),
        };
    }
    compare(ours, reference)
}

/// Records the contact list and the constraint structure of `d` (after `forward`) against
/// golden state `s` in `report`: every float array by its error, and returns the descriptions
/// of every integer difference (the first contact whose geoms, `dim`, `exclude` or `efc_address`
/// differ, `ncon`, `nefc`, `ne`, `nf`, `nl`, `efc_type`, `efc_id`); an empty list means every
/// integer matches. Never panics on a difference.
pub fn record_structure<R: Real>(
    m: &Model<R>,
    d: &Data<R>,
    s: &Value,
    report: &mut Report,
) -> Vec<String> {
    let mut list_notes = Vec::new();
    let mut row_notes = Vec::new();
    // the contact list: integers exactly, floats by error
    let list = compare_lists(d, s);
    if let Some(first) = list.first_integer_difference {
        list_notes.push(format!("state {}: {first}", s["name"]));
    }
    for (name, e) in list.floats {
        report.record(
            &format!("contact.{name}"),
            e,
            STRUCTURE_RTOL,
            STRUCTURE_FLOOR,
        );
    }

    // which rows exist, exactly
    for (name, ours, key) in [
        ("nefc", d.nefc, "nefc"),
        ("ne", d.ne, "ne"),
        ("nf", d.nf, "nf"),
        ("nl", d.nl, "nl"),
    ] {
        let theirs = s[key].as_u64().unwrap() as usize;
        if ours != theirs {
            row_notes.push(format!(
                "state {}: {name} {ours} vs MuJoCo's {theirs}",
                s["name"]
            ));
        }
    }
    let types: Vec<i64> = d.efc_type[..d.nefc].iter().map(|&t| type_code(t)).collect();
    if types != ints(&s["efc_type"]) {
        row_notes.push(format!("state {}: efc_type differs", s["name"]));
    }
    let ids: Vec<i64> = d.efc_id[..d.nefc].iter().map(|&i| i as i64).collect();
    if ids != ints(&s["efc_id"]) {
        row_notes.push(format!("state {}: efc_id differs", s["name"]));
    }
    // the integer comparisons as arrays of their own: the count of differences, allowed 0 (an
    // allowance of half a difference, so that any difference fails and the worst count is kept)
    for (name, notes) in [
        ("integers.contact_list", &list_notes),
        ("integers.rows", &row_notes),
    ] {
        report.record(
            name,
            ErrStat {
                abs: notes.len() as f64,
                scale: 0.0,
            },
            1.0,
            0.5,
        );
    }

    let nv = m.nv;
    let nefc = d.nefc;
    let g = |key: &str| farr(&s[key]);
    let mut put = |name: &str, ours: Vec<f64>, reference: Vec<f64>| {
        rec(
            report,
            name,
            safe_cmp(&ours, &reference),
            STRUCTURE_RTOL,
            STRUCTURE_FLOOR,
        );
    };
    put("efc_J", head(&d.efc_j, nefc * nv), golden_rows(&s["efc_J"]));
    put("efc_pos", head(&d.efc_pos, nefc), g("efc_pos"));
    put("efc_margin", head(&d.efc_margin, nefc), g("efc_margin"));
    put(
        "efc_frictionloss",
        head(&d.efc_frictionloss, nefc),
        g("efc_frictionloss"),
    );
    put(
        "efc_diagApprox",
        head(&d.efc_diag_approx, nefc),
        g("efc_diagApprox"),
    );
    put("efc_R", head(&d.efc_r, nefc), g("efc_R"));
    put("efc_D", head(&d.efc_d, nefc), g("efc_D"));
    put(
        "efc_KBIP",
        head(&d.efc_kbip, 4 * nefc),
        golden_rows(&s["efc_KBIP"]),
    );
    put("efc_aref", head(&d.efc_aref, nefc), g("efc_aref"));
    put("efc_vel", head(&d.efc_vel, nefc), g("efc_vel"));
    put("qacc_smooth", widen(&d.qacc_smooth), g("qacc_smooth"));
    list_notes.extend(row_notes);
    list_notes
}

/// The tolerance of the solution comparisons (the 1c-i ones): relative to the array's largest
/// absolute value, with an absolute floor.
pub const SOLVE_RTOL: f64 = 1e-9;
/// The absolute floor under [`SOLVE_RTOL`].
pub const SOLVE_FLOOR: f64 = 1e-12;

/// Records the solution of the solver block `solver` (`"newton"` or `"cg"`) of golden state `s`:
/// `qacc`, `efc_force`, `qfrc_constraint`, the contacts' `mu` and `H` and `contact_force`, each by
/// its error (an array of another length is the worst error), and returns whether every row ended
/// in MuJoCo's zone.
pub fn record_solution<R: Real>(
    m: &Model<R>,
    d: &Data<R>,
    s: &Value,
    solver: &str,
    report: &mut Report,
) -> bool {
    let r = &s[solver];
    let (n, ncon) = (d.nefc, d.ncon);
    let mut put = |name: &str, ours: Vec<f64>, reference: Vec<f64>| {
        rec(
            report,
            name,
            safe_cmp(&ours, &reference),
            SOLVE_RTOL,
            SOLVE_FLOOR,
        );
    };
    put("qacc", widen(&d.qacc), farr(&r["qacc"]));
    put("efc_force", head(&d.efc_force, n), farr(&r["efc_force"]));
    put(
        "qfrc_constraint",
        widen(&d.qfrc_constraint),
        farr(&r["qfrc_constraint"]),
    );
    put(
        "contact_mu",
        head(&d.contact_mu, ncon),
        farr(&r["contact_mu"]),
    );
    put(
        "contact_H",
        head(&d.contact_h, 36 * ncon),
        trows(&r["contact_H"]),
    );
    let forces: Vec<f64> = (0..ncon)
        .flat_map(|i| sim_physics::contact_force(m, d, i))
        .map(|x| x.to_f64())
        .collect();
    put("contact_force", forces, trows(&r["contact_force"]));
    let zones: Vec<i64> = d.efc_state[..n]
        .iter()
        .map(|&z| super::cons::state_code(z))
        .collect();
    zones == ints(&r["efc_state"])
}

/// [`tcompile`] with the compile-time faults of `faults` (the negative controls).
pub fn tcompile_faulted<R: Real>(
    which: TWhich,
    v: Variant,
    faults: &sim_physics::faults::Faults,
) -> Compiled<R> {
    let scene = tscene(which, v);
    let (model, not_modelled) =
        sim_physics::faults::compile_faulted::<R>(&scene, faults).expect("compiles");
    let mut c = Compiled {
        scene,
        model,
        not_modelled,
    };
    c.model.disable.contact = false;
    c
}

/// Runs `forward` (Newton, `faults` injected when given) on every state of `which` and `v`,
/// recording the structure; returns the report and the integer-difference notes.
pub fn run_structure<R: Real>(
    c: &Compiled<R>,
    g: &Value,
    faults: Option<&sim_physics::faults::Faults>,
) -> (Report, Vec<String>) {
    let mut report = Report::default();
    let mut notes = Vec::new();
    for s in tstates(g) {
        let mut mm = c.model.clone();
        mm.opt.solver = PrimalSolver::Newton;
        let mut d = tdata(&mm, s);
        match faults {
            None => sim_physics::forward(&mm, &mut d),
            Some(f) => sim_physics::faults::forward_faulted(&mm, &mut d, f),
        }
        notes.extend(record_structure(&mm, &d, s, &mut report));
        assert_eq!(d.warning_collision_overflow, 0, "box-box overflow");
    }
    (report, notes)
}

// ---------------------------------------------------------------------------
// the collider sweep (`contact_pairs_golden.json`)
// ---------------------------------------------------------------------------

const SWEEP_RTOL: f64 = 1e-12;
const SWEEP_FLOOR: f64 = 1e-14;

/// The four float fields of a contact, in the sweep golden file.
pub const SWEEP_FIELDS: [&str; 4] = ["dist", "pos", "frame", "includemargin"];

pub struct SweepOutcome {
    /// Cases whose contact count, geoms, `exclude` flags or order differ from MuJoCo's.
    pub integer_failures: Vec<String>,
    /// The worst error of each float field.
    pub report: Report,
    pub cases: usize,
    pub cases_with_contacts: usize,
    pub contacts: usize,
    pub excluded: usize,
}

pub fn sweep_model_of(variant: &Value) -> (sim_scene::Scene, Model<f64>) {
    let file = variant["xml"]["file"].as_str().unwrap();
    let path = fixtures().join(file);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{file}: {e}"));
    assert_eq!(
        variant["xml"]["bytes"].as_u64(),
        Some(bytes.len() as u64),
        "{file}"
    );
    assert_eq!(
        variant["xml"]["fnv1a64"].as_str(),
        Some(format!("{:016x}", fnv1a64(&bytes)).as_str()),
        "{file} changed since its golden file was generated: rerun tools/sim_contacts_mujoco_golden.py"
    );
    let scene = load_scene(&path);
    let (model, not_modelled) = Model::<f64>::compile(&scene).expect("compiles");
    assert!(not_modelled.is_empty(), "{file}: {not_modelled:?}");
    (scene, model)
}

pub fn run_sweep_variant(
    variant: &Value,
    faults: Option<&sim_physics::faults::Faults>,
) -> SweepOutcome {
    let (_scene, model) = sweep_model_of(variant);
    let cases = variant["cases"].as_array().unwrap();
    let mut ours: Vec<Vec<f64>> = vec![Vec::new(); SWEEP_FIELDS.len()];
    let mut theirs: Vec<Vec<f64>> = vec![Vec::new(); SWEEP_FIELDS.len()];
    let mut integer_failures = Vec::new();
    let (mut with_contacts, mut contacts, mut excluded) = (0, 0, 0);
    for (k, case) in cases.iter().enumerate() {
        // the qpos of the case: the plane colliders have no body for geom a
        let mut scene_qpos = Vec::new();
        for q in case["qpos"].as_array().unwrap() {
            if !q.is_null() {
                scene_qpos.extend(farr(q));
            }
        }
        let mut d = Data::new(&model);
        d.qpos = physics_qpos(&model, &scene_qpos);
        sim_physics::kinematics(&model, &mut d);
        match faults {
            None => sim_physics::collide(&model, &mut d),
            Some(f) => sim_physics::faults::collide_faulted(&model, &mut d, f),
        }

        let gc = case["contacts"].as_array().unwrap();
        assert_eq!(gc.len() as u64, case["ncon"].as_u64().unwrap());
        if gc.len() != d.ncon {
            integer_failures.push(format!(
                "case {k}: ncon {} vs MuJoCo's {}",
                d.ncon,
                gc.len()
            ));
            continue;
        }
        if !gc.is_empty() {
            with_contacts += 1;
        }
        contacts += gc.len();
        for (i, g) in gc.iter().enumerate() {
            let geom = [
                d.contact_geom[2 * i] as i64,
                d.contact_geom[2 * i + 1] as i64,
            ];
            let want: Vec<i64> = g["geom"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap())
                .collect();
            if want != geom {
                integer_failures.push(format!("case {k} contact {i}: geoms {geom:?} vs {want:?}"));
            }
            if i64::from(d.contact_exclude[i]) != g["exclude"].as_i64().unwrap() {
                integer_failures.push(format!(
                    "case {k} contact {i}: exclude {} vs {}",
                    d.contact_exclude[i], g["exclude"]
                ));
            }
            if g["exclude"].as_i64() == Some(1) {
                excluded += 1;
            }
            ours[0].push(d.contact_dist[i]);
            theirs[0].push(f(&g["dist"]));
            ours[1].extend_from_slice(&d.contact_pos[3 * i..3 * i + 3]);
            theirs[1].extend(farr(&g["pos"]));
            ours[2].extend_from_slice(&d.contact_frame[9 * i..9 * i + 9]);
            theirs[2].extend(farr(&g["frame"]));
            ours[3].push(d.contact_includemargin[i]);
            theirs[3].push(f(&g["includemargin"]));
        }
        assert_eq!(d.warning_collision_overflow, 0, "case {k}");
    }
    let mut report = Report::default();
    for (n, name) in SWEEP_FIELDS.iter().enumerate() {
        report.record(name, compare(&ours[n], &theirs[n]), SWEEP_RTOL, SWEEP_FLOOR);
    }
    SweepOutcome {
        integer_failures,
        report,
        cases: cases.len(),
        cases_with_contacts: with_contacts,
        contacts,
        excluded,
    }
}

// ---------------------------------------------------------------------------
// the engine comparison's penetration metric
// ---------------------------------------------------------------------------

/// A Rust copy of the engine comparison's `box_pile_penetration`
/// (the comparison's benchmark script, 15 separating axes, no arena walls): for
/// boxes of half-size `half` at `pos` with orientations `quat` (`[w, x, y, z]`), the largest
/// box-box penetration (the minimum overlap over the 15 axes, floored at 0), the largest
/// box-floor penetration, and the numbers of pairs tested and penetrating.
pub fn box_pile_penetration(
    pos: &[[f64; 3]],
    quat: &[[f64; 4]],
    half: f64,
) -> (f64, f64, usize, usize) {
    let rot = |q: [f64; 4]| -> [[f64; 3]; 3] {
        let [w, x, y, z] = q;
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    };
    let n = pos.len();
    let rs: Vec<[[f64; 3]; 3]> = quat.iter().map(|&q| rot(q)).collect();
    // the box-floor penetration: the half-extent along z minus the height
    let mut floor_max = 0.0f64;
    for i in 0..n {
        let ext = half * rs[i][2].iter().map(|x| x.abs()).sum::<f64>();
        floor_max = floor_max.max((ext - pos[i][2]).max(0.0));
    }
    let rc = 2.0 * half * 3.0f64.sqrt();
    let col = |r: &[[f64; 3]; 3], k: usize| [r[0][k], r[1][k], r[2][k]];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let (mut bb_max, mut pairs, mut penetrating) = (0.0f64, 0usize, 0usize);
    for i in 0..n {
        for j in i + 1..n {
            let d = [
                pos[i][0] - pos[j][0],
                pos[i][1] - pos[j][1],
                pos[i][2] - pos[j][2],
            ];
            let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if dist >= rc {
                continue;
            }
            pairs += 1;
            let (a, b) = (&rs[i], &rs[j]);
            let t = [
                pos[j][0] - pos[i][0],
                pos[j][1] - pos[i][1],
                pos[j][2] - pos[i][2],
            ];
            let mut axes: Vec<[f64; 3]> = Vec::with_capacity(15);
            for k in 0..3 {
                axes.push(col(a, k));
            }
            for k in 0..3 {
                axes.push(col(b, k));
            }
            for ka in 0..3 {
                for kb in 0..3 {
                    axes.push(cross(col(a, ka), col(b, kb)));
                }
            }
            let mut depth = f64::INFINITY;
            for ax in axes {
                let nrm = (ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]).sqrt();
                let ov = if nrm > 1e-9 {
                    let axn = [ax[0] / nrm, ax[1] / nrm, ax[2] / nrm];
                    let radius = |r: &[[f64; 3]; 3]| {
                        half * (0..3)
                            .map(|l| (axn[0] * r[0][l] + axn[1] * r[1][l] + axn[2] * r[2][l]).abs())
                            .sum::<f64>()
                    };
                    let sep = (t[0] * axn[0] + t[1] * axn[1] + t[2] * axn[2]).abs();
                    radius(a) + radius(b) - sep
                } else {
                    f64::INFINITY
                };
                depth = depth.min(ov);
            }
            let bb = depth.max(0.0);
            bb_max = bb_max.max(bb);
            if bb > 1e-9 {
                penetrating += 1;
            }
        }
    }
    (bb_max, floor_max, pairs, penetrating)
}

/// Positions and `[w, x, y, z]` orientations of the pile's boxes from a scene-layout `qpos`
/// (`[x, y, z, qx, qy, qz, qw]` per free body).
pub fn pile_poses_scene(qpos: &[f64]) -> (Vec<[f64; 3]>, Vec<[f64; 4]>) {
    let n = qpos.len() / 7;
    let pos = (0..n)
        .map(|b| [qpos[7 * b], qpos[7 * b + 1], qpos[7 * b + 2]])
        .collect();
    let quat = (0..n)
        .map(|b| {
            [
                qpos[7 * b + 6],
                qpos[7 * b + 3],
                qpos[7 * b + 4],
                qpos[7 * b + 5],
            ]
        })
        .collect();
    (pos, quat)
}

// ---------------------------------------------------------------------------------------
// the inertial-frame model (`body_sameframe`)
// ---------------------------------------------------------------------------------------

/// The golden file of `contact_sameframe.xml` (`sameframe_inertial_golden.json`), checked against
/// the bytes of its XML.
pub fn inertial_golden() -> Value {
    let g = super::read_json(&fixtures().join("sameframe_inertial_golden.json"));
    let bytes = std::fs::read(inertial_xml_path()).expect("xml bytes");
    assert_eq!(g["xml"]["bytes"].as_u64(), Some(bytes.len() as u64));
    assert_eq!(
        g["xml"]["fnv1a64"].as_str(),
        Some(format!("{:016x}", fnv1a64(&bytes)).as_str()),
        "contact_sameframe.xml changed since its golden file was generated: rerun tools/sim_contacts_mujoco_golden.py"
    );
    assert_eq!(g["mujoco_version"], "3.14.0");
    g
}

/// The XML of the inertial-frame model.
pub fn inertial_xml_path() -> std::path::PathBuf {
    humanoid_dir().join("contact_sameframe.xml")
}

/// The model of `contact_sameframe.xml` as the oracle ran it: the smooth dynamics only
/// (constraints and contacts off), with no `NotModelled` entry.
pub fn inertial_model<R: Real>() -> Compiled<R> {
    let c = super::compile::<R>(load_scene(&inertial_xml_path()));
    assert!(c.not_modelled.is_empty(), "{:?}", c.not_modelled);
    c
}

/// Every comparison of `m` with the inertial-frame golden `g` (the frames, `M`, `qfrc_bias`,
/// `qacc`, one Euler and one RK4 step), the worst error of each named array over the states in
/// the report, each held to `rtol` of the array's largest value (floor `floor`).
pub fn inertial_report(m: &Model<f64>, g: &Value, rtol: f64, floor: f64) -> Report {
    let mut report = Report::default();
    let nbody = m.nbody;
    for s in tstates(g) {
        let mut d = Data::new(m);
        d.qpos = physics_qpos(m, &farr(&s["qpos"]));
        d.qvel = farr(&s["qvel"]);
        sim_physics::forward(m, &mut d);
        assert_eq!(d.warning_collision_overflow, 0, "box-box overflow");
        let mut put = |name: &str, ours: &[f64], theirs: Vec<f64>| {
            rec(&mut report, name, safe_cmp(ours, &theirs), rtol, floor);
        };
        put("xpos", &d.xpos, trows(&s["xpos"]));
        put("xmat", &d.xmat, trows(&s["xmat"]));
        put("xipos", &d.xipos, trows(&s["xipos"]));
        put("ximat", &d.ximat, trows(&s["ximat"]));
        put("subtree_com", &d.subtree_com, trows(&s["subtree_com"]));
        put("geom_xpos", &d.geom_xpos, trows(&s["geom_xpos"]));
        put("geom_xmat", &d.geom_xmat, trows(&s["geom_xmat"]));
        put("M", &d.qm, farr(&s["M"]));
        put("qfrc_bias", &d.qfrc_bias, farr(&s["qfrc_bias"]));
        put("qacc", &d.qacc, farr(&s["qacc"]));
        // xquat: [w, x, y, z] inside, compared as [x, y, z, w] up to sign
        let mut xq = Vec::with_capacity(4 * nbody);
        for i in 0..nbody {
            let q = &d.xquat[4 * i..4 * i + 4];
            xq.extend([q[1], q[2], q[3], q[0]]);
        }
        rec(
            &mut report,
            "xquat",
            compare_quats_up_to_sign(&xq, &trows(&s["xquat"])),
            rtol,
            floor,
        );
        for (label, integ) in [
            ("euler_step", Integrator::Euler),
            ("rk4_step", Integrator::Rk4),
        ] {
            let mut mm = m.clone();
            mm.integrator = integ;
            let mut dd = Data::new(&mm);
            dd.qpos = physics_qpos(&mm, &farr(&s["qpos"]));
            dd.qvel = farr(&s["qvel"]);
            sim_physics::step(&mm, &mut dd);
            assert_eq!(dd.warning_collision_overflow, 0, "box-box overflow");
            let want = &s[label];
            rec(
                &mut report,
                &format!("{label}.qpos"),
                safe_cmp(&scene_qpos(&mm, &dd.qpos), &farr(&want["qpos"])),
                rtol,
                floor,
            );
            rec(
                &mut report,
                &format!("{label}.qvel"),
                safe_cmp(&dd.qvel, &farr(&want["qvel"])),
                rtol,
                floor,
            );
        }
    }
    report
}

// ---------------------------------------------------------------------------------------
// the settle runs
// ---------------------------------------------------------------------------------------

/// The settle run of `which`: our poses and velocities at the golden file's recorded steps (the
/// scene's own initial pose run from rest at the golden file's timestep, the initial velocities
/// of its named bodies applied).
pub fn tsettle<R: Real>(which: TWhich, v: Variant, g: &Value) -> Vec<(usize, Data<R>)> {
    let mut c = tcompile::<R>(which, v);
    let s = &g["settle"];
    let h = f(&s["timestep"]);
    c.model.timestep = R::from_f64(h);
    let m = &c.model;
    let mut d = Data::new(m);
    // the initial velocities of named bodies (6 numbers, the free joint's dofs)
    for (name, v6) in s["initial_qvel"].as_object().unwrap() {
        let b = c
            .scene
            .bodies
            .iter()
            .position(|b| b.name == *name)
            .unwrap_or_else(|| panic!("no body {name}"));
        let adr = m.body_dofadr[sim_physics::scene_body_to_internal(b)];
        for (k, x) in farr(v6).iter().enumerate() {
            d.qvel[adr + k] = R::from_f64(*x);
        }
    }
    let frames: Vec<usize> = s["frames"]
        .as_array()
        .unwrap()
        .iter()
        .map(|fr| fr["step"].as_u64().unwrap() as usize)
        .collect();
    let total = s["steps"].as_u64().unwrap() as usize;
    let mut out = Vec::new();
    for n in 1..=total {
        sim_physics::step(m, &mut d);
        assert_eq!(d.warning_collision_overflow, 0);
        if frames.contains(&n) {
            out.push((n, d.clone()));
        }
    }
    out
}
