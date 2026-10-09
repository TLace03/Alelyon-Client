//! Step times of every contact fixture, measured and not gated (phase 1c-ii).
//!
//! The median of 1,000 steps (200 in a debug build, whose times mean little) from each model's own
//! initial pose, Newton, the model's own integrator and both cones, in `f64` and `f32`, on one thread
//! of the machine the test runs on. The lines start with `TIMING`, not `MEASURED`, so that the
//! `MEASURED` lines of a debug and a release run stay identical. The README quotes the release
//! numbers with the machine's busy state noted at the time; a time here depends on what else the
//! machine is doing, which is why nothing gates it.

mod common;

use std::time::Instant;

use common::contacts::*;
use sim_physics::{Data, PrimalSolver, Real, step};

fn median_step_micros<R: Real>(which: TWhich, v: Variant) -> (f64, f64, f64, usize) {
    let mut c = tcompile::<R>(which, v);
    c.model.opt.solver = PrimalSolver::Newton;
    let n = if cfg!(debug_assertions) { 200 } else { 1000 };
    let mut d = Data::new(&c.model);
    let mut times = Vec::with_capacity(n);
    let mut with_contacts = 0usize;
    for _ in 0..n {
        let t = Instant::now();
        step(&c.model, &mut d);
        times.push(t.elapsed().as_secs_f64() * 1e6);
        with_contacts += usize::from(d.nefc > 0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[n / 2], times[0], times[n - 1], with_contacts)
}

#[test]
fn the_step_time_of_every_contact_fixture_is_measured() {
    let build = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    for which in TMODELS {
        for v in CONES {
            let (med, min, max, active) = median_step_micros::<f64>(which, v);
            println!(
                "TIMING f64 {} {} newton ({build}): median {med:.1} us per step (min {min:.1}, max {max:.1}); {active} steps had contact rows",
                which.name(),
                v.name()
            );
            let (med, min, max, active) = median_step_micros::<f32>(which, v);
            println!(
                "TIMING f32 {} {} newton ({build}): median {med:.1} us per step (min {min:.1}, max {max:.1}); {active} steps had contact rows",
                which.name(),
                v.name()
            );
        }
    }
}
