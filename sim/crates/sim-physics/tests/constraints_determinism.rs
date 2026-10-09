//! Determinism with the constraints active: the same state steps to the same bits.
//!
//! The 1b determinism tests (`determinism.rs`) run with the constraints off; these run
//! the two models that have constraints with them on, in `f32`, with both solvers:
//!
//! - two runs of 1,000 steps from the same state end bitwise equal (and the rows of the
//!   constraint were active in the run: the equality is not between two runs that never
//!   touched a limit);
//! - `step_world` over three environments gives, for each environment, the same bits as that
//!   environment alone and as the environments in reverse order: the warm start, the
//!   workspace and the factor of an environment are its own;
//! - `step_world` agrees bit for bit with the plain `step` on the same `Data` (the load
//!   and store through the world change nothing, with the constraint force in the step).

mod common;

use common::cons::*;
use common::*;
use sim_physics::{Data, PrimalSolver, QuatOrder, convert_qpos, step, step_world};
use sim_world::{FieldId, HostWorld};

fn bits(a: &[f32]) -> Vec<u32> {
    a.iter().map(|x| x.to_bits()).collect()
}

fn model32(which: CWhich, solver: PrimalSolver) -> Compiled<f32> {
    let mut c = ccompile::<f32>(which);
    c.model.opt.solver = solver;
    c
}

/// `n` steps from golden state `k`; the data, and how many steps had constraint rows.
fn run(c: &Compiled<f32>, golden: &serde_json::Value, k: usize, n: usize) -> (Data<f32>, usize) {
    let mut d = cdata(&c.model, &golden["states"][k]);
    let mut active = 0usize;
    for _ in 0..n {
        step(&c.model, &mut d);
        if d.nefc > 0 {
            active += 1;
        }
    }
    (d, active)
}

fn thousand_steps(which: CWhich, solver: PrimalSolver, k: usize) {
    let c = model32(which, solver);
    let golden = cgolden(which);
    let (a, active_a) = run(&c, &golden, k, 1000);
    let (b, active_b) = run(&c, &golden, k, 1000);
    assert_eq!(bits(&a.qpos), bits(&b.qpos));
    assert_eq!(bits(&a.qvel), bits(&b.qvel));
    assert_eq!(bits(&a.qacc), bits(&b.qacc));
    assert_eq!(bits(&a.qacc_warmstart), bits(&b.qacc_warmstart));
    assert_eq!(a.time.to_bits(), b.time.to_bits());
    assert_eq!(a.nefc, b.nefc);
    assert_eq!(bits(&a.efc_force[..a.nefc]), bits(&b.efc_force[..b.nefc]));
    assert_eq!(active_a, active_b);
    // not trivially equal because it is NaN everywhere, and the constraints were in play
    assert!(a.qpos.iter().chain(&a.qvel).all(|x| x.is_finite()));
    assert!(a.qvel.iter().any(|x| x.abs() > 1e-3));
    assert!(
        active_a > 0,
        "no step of the run had a constraint row: the run did not test the constraints"
    );
    println!(
        "{} {solver:?} f32 after 1000 steps: time {}, max |qvel| {}, steps with constraint rows {active_a} of 1000, final nefc {}",
        which.name(),
        a.time,
        a.qvel.iter().fold(0.0f32, |m, x| m.max(x.abs())),
        a.nefc,
    );
}

#[test]
fn a_thousand_constrained_humanoid_steps_are_bitwise_reproducible_in_f32() {
    thousand_steps(CWhich::Humanoid, PrimalSolver::Newton, 3);
    thousand_steps(CWhich::Humanoid, PrimalSolver::Cg, 3);
}

#[test]
fn a_thousand_constrained_steps_of_the_constrained_model_are_bitwise_reproducible_in_f32() {
    thousand_steps(CWhich::Constrained, PrimalSolver::Newton, 3);
    thousand_steps(CWhich::Constrained, PrimalSolver::Cg, 3);
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
        active += datas.iter().filter(|d| d.nefc > 0).count();
    }
    active
}

fn batch_independence(which: CWhich, solver: PrimalSolver) {
    let c = model32(which, solver);
    let golden = cgolden(which);
    let picks = [0usize, 3, 5];
    let steps = 25;

    let mut together = world_of(&c, &golden, &picks);
    let active = step_n(&c, &mut together, steps);
    assert!(active > 0, "no environment had a constraint row in the run");

    // each environment alone
    for (env, &k) in picks.iter().enumerate() {
        let mut alone = world_of(&c, &golden, &[k]);
        step_n(&c, &mut alone, steps);
        assert_eq!(
            env_bits(&together, env as u32),
            env_bits(&alone, 0),
            "{} {solver:?}: environment {env} differs from the same environment alone",
            which.name()
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
            "{} {solver:?}: environment {env} differs when the batch is reversed",
            which.name()
        );
    }

    // the environments really are different, and moved
    assert_ne!(env_bits(&together, 0), env_bits(&together, 1));
    assert!(together.qvel(0).iter().any(|v| v.abs() > 1e-3));
}

#[test]
fn step_world_is_independent_of_the_other_environments_and_their_order_with_constraints() {
    for solver in [PrimalSolver::Newton, PrimalSolver::Cg] {
        batch_independence(CWhich::Humanoid, solver);
        batch_independence(CWhich::Constrained, solver);
    }
}

#[test]
fn step_world_equals_a_plain_step_of_the_same_data_with_constraints() {
    for which in [CWhich::Humanoid, CWhich::Constrained] {
        for solver in [PrimalSolver::Newton, PrimalSolver::Cg] {
            let c = model32(which, solver);
            let golden = cgolden(which);
            let mut world = world_of(&c, &golden, &[2]);
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
                assert_eq!(
                    bits(world.qpos(0)),
                    bits(&expected),
                    "{} {solver:?}",
                    which.name()
                );
                assert_eq!(
                    bits(world.qvel(0)),
                    bits(&d.qvel),
                    "{} {solver:?}",
                    which.name()
                );
                assert_eq!(
                    bits(&datas[0].qacc_warmstart),
                    bits(&d.qacc_warmstart),
                    "{} {solver:?}",
                    which.name()
                );
            }
        }
    }
}
