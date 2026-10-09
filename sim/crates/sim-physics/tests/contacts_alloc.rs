//! No allocation inside a step (phase 1c-ii).
//!
//! - **Buffers.** Every `Data` buffer (the state, every scratch array, the contact arrays and the
//!   solver's workspace) keeps its pointer and its capacity over 1,000 steps of each contact
//!   fixture, both cones: a step that reallocated one would move it. The list of buffers is checked
//!   against the number of arrays `Data`'s `Debug` output shows, so a buffer added to `Data`
//!   without being listed here fails the test.
//! - **Source.** The step modules (the collision driver, the colliders, the contact frames, the
//!   Jacobians, the constraint rows, the solver, the smooth dynamics, the linear algebra, the
//!   factorisation, the shared maths, the energies and `step_world`) contain no `Vec::`, `vec!`,
//!   `collect`, `to_vec`, `Box::new`, `sort`, `clone()`, `push` or `with_capacity` outside their
//!   `#[cfg(test)]` modules and comments. The candidate list, which allocates, is built when the
//!   model is compiled and lives in `candidates.rs`. Positive control: the scan finds every one
//!   of those in a test string.

mod common;

use common::contacts::*;
use sim_physics::{Data, Real, step};

/// `(pointer, capacity)` of every `Vec` of a `Data`.
fn buffers<R: Real>(d: &Data<R>) -> Vec<(usize, usize)> {
    macro_rules! b {
        ($($v:expr),* $(,)?) => { vec![$((($v).as_ptr() as usize, ($v).capacity())),*] };
    }
    let ws = &d.ws;
    b![
        d.qpos,
        d.qvel,
        d.ctrl,
        d.qacc_warmstart,
        d.xpos,
        d.xquat,
        d.xmat,
        d.xipos,
        d.ximat,
        d.xanchor,
        d.xaxis,
        d.subtree_com,
        d.cdof,
        d.cinert,
        d.crb,
        d.cvel,
        d.cdof_dot,
        d.cacc,
        d.cfrc_body,
        d.qm,
        d.qld,
        d.qld_diag_inv,
        d.qh,
        d.qh_diag_inv,
        d.qfrc_bias,
        d.qfrc_spring,
        d.qfrc_damper,
        d.qfrc_passive,
        d.qfrc_actuator,
        d.qfrc_smooth,
        d.qacc_smooth,
        d.qfrc_constraint,
        d.geom_xpos,
        d.geom_xmat,
        d.qacc,
        d.qacc_step,
        d.ten_length,
        d.ten_j,
        d.ten_velocity,
        d.contact_dist,
        d.contact_pos,
        d.contact_frame,
        d.contact_includemargin,
        d.contact_friction,
        d.contact_solref,
        d.contact_solreffriction,
        d.contact_solimp,
        d.contact_mu,
        d.contact_h,
        d.contact_dim,
        d.contact_geom,
        d.contact_exclude,
        d.contact_efc_address,
        d.cand_ncon,
        d.cand_start,
        d.cand_overflow,
        d.pre_dist,
        d.pre_pos,
        d.pre_frame,
        d.con_jac1p,
        d.con_jac2p,
        d.con_jac1r,
        d.con_jac2r,
        d.con_jacdifp,
        d.con_jacdifr,
        d.con_jac,
        d.con_edge,
        d.efc_type,
        d.efc_id,
        d.efc_j,
        d.efc_pos,
        d.efc_margin,
        d.efc_frictionloss,
        d.efc_diag_approx,
        d.efc_r,
        d.efc_d,
        d.efc_kbip,
        d.efc_aref,
        d.efc_vel,
        d.efc_b,
        d.efc_force,
        d.efc_state,
        d.rk_x,
        d.rk_f,
        d.rk_dx,
        ws.jaref,
        ws.jv,
        ws.ma,
        ws.mv,
        ws.grad,
        ws.mgrad,
        ws.search,
        ws.quad,
        ws.oldstate,
        ws.gradold,
        ws.mgradold,
        ws.graddif,
        ws.mgraddif,
        ws.d,
        ws.cholupd,
        ws.l,
        ws.lcone,
        ws.ltj,
    ]
}

/// The number of array-valued fields `Data` prints in its `Debug` output (every `Vec`, those of the
/// workspace included, and the two-element `energy`).
fn debug_arrays<R: Real>(d: &Data<R>) -> usize {
    format!("{d:?}").matches(": [").count()
}

fn thousand_steps(which: TWhich, v: Variant) {
    let c = tcompile::<f64>(which, v);
    let m = &c.model;
    let mut d = Data::new(m);
    let before = buffers(&d);
    // every buffer is listed (the `energy` array is the one array that is not a `Vec`)
    assert_eq!(
        debug_arrays(&d),
        before.len() + 1,
        "Data has a buffer that this test does not list"
    );
    // a timestep that keeps the humanoid's fall inside the run
    let (mut contacts, mut peak) = (0usize, 0usize);
    for n in 0..1000 {
        step(m, &mut d);
        assert_eq!(d.warning_collision_overflow, 0);
        let now = buffers(&d);
        assert_eq!(
            now,
            before,
            "{} {}: a buffer moved or changed capacity in step {n}",
            which.name(),
            v.name()
        );
        if d.ncon > 0 {
            contacts += 1;
        }
        peak = peak.max(d.ncon);
    }
    println!(
        "MEASURED allocation {} {}: {} buffers keep their pointer and capacity over 1,000 steps ({contacts} steps with contacts, peak {peak} contacts of at most {})",
        which.name(),
        v.name(),
        before.len(),
        m.ncon_max
    );
    assert!(
        contacts > 0 || which == TWhich::Humanoid,
        "the run never made a contact"
    );
}

macro_rules! alloc_tests {
    ($($name:ident: $which:expr;)*) => {
        $(
            #[test]
            fn $name() {
                for v in CONES {
                    thousand_steps($which, v);
                }
            }
        )*
    };
}

alloc_tests! {
    no_buffer_of_the_sphere_scene_moves_in_a_thousand_steps: TWhich::Sphere;
    no_buffer_of_the_box_scene_moves_in_a_thousand_steps: TWhich::Box;
    no_buffer_of_the_stack_moves_in_a_thousand_steps: TWhich::Stack;
    no_buffer_of_the_capsules_move_in_a_thousand_steps: TWhich::Capsules;
    no_buffer_of_the_pile_moves_in_a_thousand_steps: TWhich::Pile;
    no_buffer_of_the_humanoid_moves_in_a_thousand_steps: TWhich::Humanoid;
    no_buffer_of_the_zoo_moves_in_a_thousand_steps: TWhich::Zoo;
}

// ---------------------------------------------------------------------------
// the source scan
// ---------------------------------------------------------------------------

/// What a step module must not contain (outside comments and its test module).
const FORBIDDEN: [&str; 9] = [
    "Vec::",
    "vec!",
    "collect",
    "to_vec",
    "Box::new",
    "sort",
    "clone()",
    "push(",
    "with_capacity",
];

/// The step modules: everything a step runs (the review found the numerical kernels, `linalg.rs`
/// and `factor.rs`, the shared `math.rs`, `energy.rs` and `world.rs`'s `step_world` missing from
/// an earlier list; the buffer-pointer test above cannot see a transient `Vec`).
const STEP_MODULES: [&str; 14] = [
    "collision.rs",
    "collide_primitive.rs",
    "collide_box.rs",
    "contact.rs",
    "jac.rs",
    "constraint.rs",
    "solver.rs",
    "smooth.rs",
    "integrate.rs",
    "linalg.rs",
    "factor.rs",
    "math.rs",
    "energy.rs",
    "world.rs",
];

/// The code of a source file: without its comments and without its `#[cfg(test)]` module (which
/// closes every file of this crate).
fn code_of(source: &str) -> String {
    let body = source.split("#[cfg(test)]").next().unwrap();
    body.lines()
        .map(|line| line.split("//").next().unwrap())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The forbidden calls a source text contains.
fn allocating_calls(code: &str) -> Vec<&'static str> {
    FORBIDDEN
        .iter()
        .copied()
        .filter(|p| code.contains(p))
        .collect()
}

#[test]
fn the_scan_finds_every_allocating_call_in_a_test_string() {
    let bad = r#"
        let a = Vec::new();
        let b = vec![0.0; 3];
        let c: Vec<_> = it.collect();
        let d = slice.to_vec();
        let e = Box::new(1);
        v.sort_by_key(|x| x);
        let f = a.clone();
        a.push(1);
        let g = Vec::with_capacity(4);
    "#;
    let found = allocating_calls(&code_of(bad));
    assert_eq!(found, FORBIDDEN.to_vec(), "the scan must find each pattern");
    // a comment and a test module are not scanned
    assert!(allocating_calls(&code_of("let x = 1; // Vec::new() in a comment")).is_empty());
    assert!(
        allocating_calls(&code_of(
            "let x = 1;\n#[cfg(test)]\nmod tests { fn f() { let v = vec![1]; } }"
        ))
        .is_empty()
    );
}

#[test]
fn the_step_modules_contain_no_allocating_call() {
    let src = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    for file in STEP_MODULES {
        let text =
            std::fs::read_to_string(src.join(file)).unwrap_or_else(|e| panic!("{file}: {e}"));
        let found = allocating_calls(&code_of(&text));
        assert!(found.is_empty(), "{file} contains {found:?}");
    }
    // the compile-time candidate list does allocate: it is the one module that is not a step
    // module (and the scan would see it)
    let text = std::fs::read_to_string(src.join("candidates.rs")).unwrap();
    assert!(!allocating_calls(&code_of(&text)).is_empty());
    println!(
        "MEASURED allocation scan: {} step modules contain none of {:?}; candidates.rs (compile time) does",
        STEP_MODULES.len(),
        FORBIDDEN
    );
}
