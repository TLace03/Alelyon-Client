//! Determinism with contacts in every step: the same state steps to the same bits.
//!
//! The 1b and 1c-i determinism tests run with contacts off; these run the pile (eight boxes
//! tumbling onto a floor) and the humanoid (falling onto its floor from `qpos0` at h = 0.005) with
//! contacts on, in `f32`, with both cones and both solvers:
//!
//! - two runs of 1,000 steps from the same state end bitwise equal (and contacts were active in
//!   the run: the equality is not between two runs that never touched anything);
//! - `step_world` over three environments gives, for each environment, the same bits as that
//!   environment alone and as the environments in reverse order: the contact list, the warm
//!   start, the workspace and the factors of an environment are its own, and the collision step
//!   has no cross-environment state;
//! - `step_world` agrees bit for bit with the plain `step` on the same `Data` (the load and store
//!   through the world change nothing, with the contact forces in the step).
//!
//! The same `cargo test -p sim-physics -- --nocapture --test-threads=1` in debug and with
//! `--release` print identical `MEASURED` lines once sorted (the README records the gate).

mod common;

use common::contacts::*;
use common::*;
use sim_physics::{Data, PrimalSolver, QuatOrder, convert_qpos, step, step_world};
use sim_world::{FieldId, HostWorld};

fn bits(a: &[f32]) -> Vec<u32> {
    a.iter().map(|x| x.to_bits()).collect()
}

fn model32(which: TWhich, v: Variant, solver: PrimalSolver) -> Compiled<f32> {
    let mut c = tcompile::<f32>(which, v);
    c.model.opt.solver = solver;
    if which == TWhich::Humanoid {
        // the falling humanoid of the settle runs
        c.model.timestep = 0.005;
    }
    c
}

/// 1,000 steps from the model's own initial pose; the data, how many steps had contacts that were
/// constraints, and the largest `ncon`.
fn run(c: &Compiled<f32>, n: usize) -> (Data<f32>, usize, usize) {
    let mut d = Data::new(&c.model);
    let (mut active, mut peak) = (0usize, 0usize);
    for _ in 0..n {
        step(&c.model, &mut d);
        assert_eq!(d.warning_collision_overflow, 0);
        if d.nefc > 0 {
            active += 1;
        }
        peak = peak.max(d.ncon);
    }
    (d, active, peak)
}

fn thousand_steps(which: TWhich, v: Variant, solver: PrimalSolver) {
    let c = model32(which, v, solver);
    let (a, active_a, peak_a) = run(&c, 1000);
    let (b, active_b, peak_b) = run(&c, 1000);
    assert_eq!(bits(&a.qpos), bits(&b.qpos));
    assert_eq!(bits(&a.qvel), bits(&b.qvel));
    assert_eq!(bits(&a.qacc), bits(&b.qacc));
    assert_eq!(bits(&a.qacc_warmstart), bits(&b.qacc_warmstart));
    assert_eq!(a.time.to_bits(), b.time.to_bits());
    assert_eq!((a.ncon, a.nefc), (b.ncon, b.nefc));
    assert_eq!(bits(&a.efc_force[..a.nefc]), bits(&b.efc_force[..b.nefc]));
    assert_eq!(
        bits(&a.contact_dist[..a.ncon]),
        bits(&b.contact_dist[..b.ncon])
    );
    assert_eq!(
        bits(&a.contact_pos[..3 * a.ncon]),
        bits(&b.contact_pos[..3 * b.ncon])
    );
    assert_eq!(a.contact_geom[..2 * a.ncon], b.contact_geom[..2 * b.ncon]);
    assert_eq!((active_a, peak_a), (active_b, peak_b));
    // not trivially equal because it is NaN everywhere, and contacts were in play
    assert!(a.qpos.iter().chain(&a.qvel).all(|x| x.is_finite()));
    assert!(
        active_a >= 100,
        "{} steps of 1,000 had constraint rows: the run did not test the contacts",
        active_a
    );
    println!(
        "MEASURED f32 {} {} {solver:?} after 1000 steps: time {}, steps with contact rows {active_a} of 1000, peak contacts {peak_a}, final ncon {} nefc {}, max |qvel| {}",
        which.name(),
        v.name(),
        a.time,
        a.ncon,
        a.nefc,
        a.qvel.iter().fold(0.0f32, |m, x| m.max(x.abs())),
    );
}

macro_rules! determinism_tests {
    ($($name:ident: $which:expr, $variant:expr, $solver:expr;)*) => {
        $(
            #[test]
            fn $name() {
                thousand_steps($which, $variant, $solver);
            }
        )*
    };
}

determinism_tests! {
    a_thousand_pile_steps_are_bitwise_reproducible_in_f32_pyramidal_newton: TWhich::Pile, Variant::Pyramidal, PrimalSolver::Newton;
    a_thousand_pile_steps_are_bitwise_reproducible_in_f32_pyramidal_cg: TWhich::Pile, Variant::Pyramidal, PrimalSolver::Cg;
    a_thousand_pile_steps_are_bitwise_reproducible_in_f32_elliptic_newton: TWhich::Pile, Variant::Elliptic, PrimalSolver::Newton;
    a_thousand_pile_steps_are_bitwise_reproducible_in_f32_elliptic_cg: TWhich::Pile, Variant::Elliptic, PrimalSolver::Cg;
    a_thousand_humanoid_fall_steps_are_bitwise_reproducible_in_f32_pyramidal_newton: TWhich::Humanoid, Variant::Pyramidal, PrimalSolver::Newton;
    a_thousand_humanoid_fall_steps_are_bitwise_reproducible_in_f32_pyramidal_cg: TWhich::Humanoid, Variant::Pyramidal, PrimalSolver::Cg;
    a_thousand_humanoid_fall_steps_are_bitwise_reproducible_in_f32_elliptic_newton: TWhich::Humanoid, Variant::Elliptic, PrimalSolver::Newton;
    a_thousand_humanoid_fall_steps_are_bitwise_reproducible_in_f32_elliptic_cg: TWhich::Humanoid, Variant::Elliptic, PrimalSolver::Cg;
}

/// A world of `picks.len()` environments, each holding one golden state.
fn world_of(c: &Compiled<f32>, golden: &serde_json::Value, picks: &[usize]) -> HostWorld {
    let mut world = HostWorld::new(&c.scene, picks.len() as u32).expect("world");
    for (env, &k) in picks.iter().enumerate() {
        let s = &golden["states"][k];
        world.set_qpos(env as u32, &farr(&s["qpos"])).expect("qpos");
        for (dst, src) in world
            .env_slice_mut(FieldId::Qvel, env as u32)
            .iter_mut()
            .zip(farr(&s["qvel"]))
        {
            *dst = src as f32;
        }
        for (dst, src) in world.ctrl_mut(env as u32).iter_mut().zip(farr(&s["ctrl"])) {
            *dst = src as f32;
        }
    }
    world
}

const FIELDS: [FieldId; 6] = [
    FieldId::Qpos,
    FieldId::Qvel,
    FieldId::BodyPos,
    FieldId::BodyQuat,
    FieldId::BodyLinVel,
    FieldId::BodyAngVel,
];

fn env_bits(world: &HostWorld, env: u32) -> Vec<Vec<u32>> {
    FIELDS
        .iter()
        .map(|&f| bits(world.env_slice(f, env)))
        .collect()
}

fn step_n(c: &Compiled<f32>, world: &mut HostWorld, n: usize) -> usize {
    let mut datas: Vec<Data<f32>> = (0..world.layout().n_envs)
        .map(|_| Data::new(&c.model))
        .collect();
    let mut active = 0usize;
    for _ in 0..n {
        step_world(&c.model, &mut datas, world).expect("step_world");
        assert!(datas.iter().all(|d| d.warning_collision_overflow == 0));
        active += datas.iter().filter(|d| d.nefc > 0).count();
    }
    active
}

fn batch_independence(which: TWhich, v: Variant, solver: PrimalSolver) {
    let c = model32(which, v, solver);
    let golden = tgolden(which, v);
    // contact-rich states of each model
    let picks: [usize; 3] = match which {
        TWhich::Pile => [2, 4, 7],
        _ => [0, 3, 5],
    };
    let steps = 25;

    let mut together = world_of(&c, &golden, &picks);
    let active = step_n(&c, &mut together, steps);
    assert!(active > 0, "no environment had a contact row in the run");

    // each environment alone
    for (env, &k) in picks.iter().enumerate() {
        let mut alone = world_of(&c, &golden, &[k]);
        step_n(&c, &mut alone, steps);
        assert_eq!(
            env_bits(&together, env as u32),
            env_bits(&alone, 0),
            "{} {} {solver:?}: environment {env} differs from the same environment alone",
            which.name(),
            v.name()
        );
    }

    // the environments in reverse order
    let reversed_picks: Vec<usize> = picks.iter().rev().copied().collect();
    let mut reversed = world_of(&c, &golden, &reversed_picks);
    step_n(&c, &mut reversed, steps);
    for env in 0..picks.len() {
        assert_eq!(
            env_bits(&together, env as u32),
            env_bits(&reversed, (picks.len() - 1 - env) as u32),
            "{} {} {solver:?}: environment {env} differs when the batch is reversed",
            which.name(),
            v.name()
        );
    }

    // the environments really are different, and moved
    assert_ne!(env_bits(&together, 0), env_bits(&together, 1));
    assert!(together.qvel(0).iter().any(|v| v.abs() > 1e-3));
    println!(
        "MEASURED f32 {} {} {solver:?}: step_world over 3 environments for {steps} steps, {active} environment-steps with contact rows; each environment equals itself alone and in reverse order, bitwise",
        which.name(),
        v.name()
    );
}

#[test]
fn step_world_is_independent_of_the_other_environments_and_their_order_with_contacts() {
    for v in CONES {
        for solver in [PrimalSolver::Newton, PrimalSolver::Cg] {
            batch_independence(TWhich::Pile, v, solver);
            batch_independence(TWhich::Humanoid, v, solver);
        }
    }
}

#[test]
fn step_world_equals_a_plain_step_of_the_same_data_with_contacts() {
    for which in [TWhich::Pile, TWhich::Humanoid, TWhich::Zoo] {
        for v in CONES {
            for solver in [PrimalSolver::Newton, PrimalSolver::Cg] {
                let c = model32(which, v, solver);
                let golden = tgolden(which, v);
                let k = if which == TWhich::Pile { 3 } else { 2 };
                let mut world = world_of(&c, &golden, &[k]);
                let mut d = Data::<f32>::new(&c.model);
                convert_qpos(
                    &c.model,
                    world.qpos(0),
                    QuatOrder::Xyzw,
                    &mut d.qpos,
                    QuatOrder::Wxyz,
                );
                d.qvel.copy_from_slice(world.qvel(0));
                d.ctrl.copy_from_slice(world.ctrl(0));
                let mut datas = vec![Data::<f32>::new(&c.model)];
                // several steps, so the warm start the two carry is compared too
                for _ in 0..5 {
                    step(&c.model, &mut d);
                    step_world(&c.model, &mut datas, &mut world).unwrap();
                    let mut expected = vec![0.0f32; c.model.nq];
                    convert_qpos(
                        &c.model,
                        &d.qpos,
                        QuatOrder::Wxyz,
                        &mut expected,
                        QuatOrder::Xyzw,
                    );
                    let label = format!("{} {} {solver:?}", which.name(), v.name());
                    assert_eq!(bits(world.qpos(0)), bits(&expected), "{label}");
                    assert_eq!(bits(world.qvel(0)), bits(&d.qvel), "{label}");
                    assert_eq!(
                        bits(&datas[0].qacc_warmstart),
                        bits(&d.qacc_warmstart),
                        "{label}"
                    );
                    assert_eq!(datas[0].ncon, d.ncon, "{label}");
                }
            }
        }
    }
}
