//! The step scheduler: rate boundaries, the contract's rule and exact counts.

use sim_contract::{RATES_V0, Rates, Sense};
use sim_world::Schedule;

/// Control tick `k` is due at physics step `ceil(k * p / c)`.
fn control_steps_up_to(c: u64, p: u64, last_step: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let mut k = 0u64;
    loop {
        let n = (k * p).div_ceil(c);
        if n > last_step {
            break;
        }
        out.push(n);
        k += 1;
    }
    out
}

#[test]
fn control_is_due_at_the_first_physics_step_at_or_after_each_control_time() {
    for (c, p) in [
        (50u32, 1000u32),
        (30, 100),
        (7, 1000),
        (1, 1),
        (1000, 1000),
        (3, 7),
        (999, 1000),
    ] {
        let s = Schedule::new(c, p).unwrap();
        let expect = control_steps_up_to(u64::from(c), u64::from(p), 3000);
        let got: Vec<u64> = (0..=3000u64).filter(|&n| s.ticks(n).control).collect();
        assert_eq!(got, expect, "control {c} Hz, physics {p} Hz");
    }
}

#[test]
fn rate_boundaries_at_50_control_over_1000_physics() {
    let s = Schedule::new(50, 1000).unwrap();
    // control every 20 physics steps
    for n in 0..2000u64 {
        assert_eq!(s.ticks(n).control, n % 20 == 0, "step {n}");
        assert!(s.ticks(n).physics);
    }
    // 10 Hz sight on a 50 Hz control clock: every 5th control tick, i.e. every 100 steps
    for n in 0..2000u64 {
        assert_eq!(
            s.ticks(n).sense(Sense::Sight),
            n % 100 == 0,
            "sight at step {n}"
        );
    }
    // 100 Hz touch is faster than control: due at every control tick, once
    for n in 0..2000u64 {
        assert_eq!(
            s.ticks(n).sense(Sense::Touch),
            n % 20 == 0,
            "touch at step {n}"
        );
    }
    // audio and proprioception ride every control tick
    for n in 0..200u64 {
        assert_eq!(s.ticks(n).sense(Sense::Sound), n % 20 == 0);
        assert_eq!(s.ticks(n).sense(Sense::Proprioception), n % 20 == 0);
    }
    // the boundaries: step 99 not due, 100 due, 101 not
    assert!(!s.ticks(99).sense(Sense::Sight));
    assert!(s.ticks(100).sense(Sense::Sight));
    assert!(!s.ticks(101).sense(Sense::Sight));
}

#[test]
fn step_zero_is_always_due() {
    let s = Schedule::new(10, 1000).unwrap();
    let t = s.ticks(0);
    assert!(t.physics && t.control);
    assert_eq!(t.control_step, 0);
    assert_eq!(t.due_senses().count(), 6, "every sense samples at step 0");
}

#[test]
fn a_sense_is_due_exactly_when_the_contract_says_on_the_control_clock() {
    for (c, p) in [
        (50u32, 1000u32),
        (30, 100),
        (10, 1000),
        (100, 1000),
        (7, 50),
    ] {
        let s = Schedule::new(c, p).unwrap();
        for n in 0..5000u64 {
            let t = s.ticks(n);
            if !t.control {
                assert!(
                    t.due_senses().next().is_none(),
                    "a sense is due without a control tick"
                );
                continue;
            }
            for sense in Sense::ALL {
                assert_eq!(
                    t.sense(sense),
                    Rates::due(sense, t.control_step, c),
                    "{sense:?} at physics step {n}, control tick {}",
                    t.control_step
                );
            }
        }
    }
}

#[test]
fn exact_counts_over_one_simulated_second() {
    let s = Schedule::new(50, 1000).unwrap();
    let mut control = 0;
    let mut sight = 0;
    let mut smell = 0;
    for n in 0..1000u64 {
        let t = s.ticks(n);
        control += u32::from(t.control);
        sight += u32::from(t.sense(Sense::Sight));
        smell += u32::from(t.sense(Sense::Smell));
    }
    assert_eq!(control, 50);
    assert_eq!(sight, 10);
    assert_eq!(smell, 10);
}

#[test]
fn equal_rates_make_everything_due_at_every_step() {
    let s = Schedule::new(1000, 1000).unwrap();
    for n in 0..100u64 {
        let t = s.ticks(n);
        assert!(t.physics && t.control);
        assert_eq!(t.control_step, n);
    }
}

#[test]
fn ticks_is_a_pure_function_and_survives_huge_steps() {
    let s = Schedule::new(50, 1000).unwrap();
    assert_eq!(s.ticks(123_456), s.ticks(123_456));
    // u128 arithmetic: no overflow near the top of u64
    let n = (u64::MAX / 20) * 20;
    assert!(s.ticks(n).control);
    assert!(!s.ticks(n + 1).control);
    let big = Schedule::new(u32::MAX, u32::MAX).unwrap();
    assert!(big.ticks(u64::MAX).control);
}

#[test]
fn an_unusable_schedule_has_nothing_due_and_says_why() {
    for (c, p) in [(0u32, 1000u32), (50, 0), (2000, 1000), (0, 0)] {
        let s = Schedule {
            control_hz: c,
            physics_hz: p,
        };
        assert!(s.validate().is_err(), "{c}/{p} must be refused");
        assert!(Schedule::new(c, p).is_err());
        for n in [0u64, 1, 20, 100] {
            let t = s.ticks(n);
            assert!(!t.physics && !t.control, "{c}/{p} step {n}");
            assert!(t.due_senses().next().is_none());
        }
    }
}

#[test]
fn custom_rates_are_honoured() {
    let rates = Rates {
        rgb_hz: 25,
        ..RATES_V0
    };
    let s = Schedule::new(50, 1000).unwrap();
    for n in 0..400u64 {
        assert_eq!(
            s.ticks_with(&rates, n).sense(Sense::Sight),
            n % 40 == 0,
            "step {n}"
        );
    }
}
