//! Philox4x32-10 against its published known-answer vectors, and the determinism
//! of the simulator's stream addressing.
//!
//! Known-answer source: the Random123 library, D. E. Shaw Research
//! (https://github.com/DEShawResearch/random123), file `tests/kat_vectors`. Its
//! header says "For each generator, we test: gen(0, 0), gen(fff, fff) and
//! gen(ctr=digits_of_pi, key=more_digits_of_pi)", and its `philox4x32 10` lines are
//! reproduced below (counter words, key words, expected output words, in the file's
//! order). The generator is implemented from the paper (J. K. Salmon, M. A. Moraes,
//! R. O. Dror, D. E. Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3", SC11); the
//! vectors were read from the library's file (fetched on 2026-09-30), not produced
//! by this implementation.

use sim_world::rng::{STREAM_PHYSICS, STREAM_RESET, STREAM_SENSE_NOISE, signed_unit, unit_open};
use sim_world::{Rng, philox4x32_10};

/// `philox4x32 10 <ctr0..3> <key0..1>   <out0..3>` lines of `tests/kat_vectors`.
const KAT: [([u32; 4], [u32; 2], [u32; 4]); 3] = [
    (
        [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000],
        [0x0000_0000, 0x0000_0000],
        [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8],
    ),
    (
        [0xffff_ffff, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff],
        [0xffff_ffff, 0xffff_ffff],
        [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd],
    ),
    (
        [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
        [0xa409_3822, 0x299f_31d0],
        [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1],
    ),
];

#[test]
fn philox4x32_10_reproduces_the_random123_known_answers() {
    for (i, (ctr, key, expected)) in KAT.iter().enumerate() {
        assert_eq!(
            &philox4x32_10(*ctr, *key),
            expected,
            "Random123 kat_vectors philox4x32 10, vector {}",
            i + 1
        );
    }
}

#[test]
fn the_known_answers_are_sensitive_to_every_input() {
    // Negative control: one flipped counter bit or key bit changes every output
    // word (avalanche), so an implementation that passes the vectors above is not
    // passing because the output ignores part of its input.
    let (ctr, key, expected) = KAT[2];
    let mut flipped = ctr;
    flipped[0] ^= 1;
    let out = philox4x32_10(flipped, key);
    for k in 0..4 {
        assert_ne!(
            out[k], expected[k],
            "word {k} unchanged by a flipped counter bit"
        );
    }
    for word in 0..4 {
        let mut c = ctr;
        c[word] ^= 0x8000_0000;
        assert_ne!(
            philox4x32_10(c, key),
            expected,
            "counter word {word} ignored"
        );
    }
    for word in 0..2 {
        let mut k = key;
        k[word] ^= 0x8000_0000;
        assert_ne!(philox4x32_10(ctr, k), expected, "key word {word} ignored");
    }
}

#[test]
fn the_same_seed_env_step_stream_and_lane_give_the_same_block() {
    let a = Rng::new(0xDEAD_BEEF_0123_4567, 17);
    let b = Rng::new(0xDEAD_BEEF_0123_4567, 17);
    assert_eq!(a, b);
    for step in [
        0u64,
        1,
        2,
        1000,
        u64::from(u32::MAX),
        u64::from(u32::MAX) + 1,
        u64::MAX,
    ] {
        for stream in [STREAM_RESET, STREAM_SENSE_NOISE, STREAM_PHYSICS] {
            for lane in [0u32, 1, 7, u32::MAX] {
                assert_eq!(a.block(step, stream, lane), b.block(step, stream, lane));
            }
        }
    }
}

#[test]
fn a_block_does_not_depend_on_the_order_it_is_asked_for() {
    let rng = Rng::new(42, 3);
    let forward: Vec<[u32; 4]> = (0..64).map(|s| rng.block(s, STREAM_RESET, 5)).collect();
    let mut reversed: Vec<[u32; 4]> = (0..64)
        .rev()
        .map(|s| rng.block(s, STREAM_RESET, 5))
        .collect();
    reversed.reverse();
    assert_eq!(forward, reversed);
}

#[test]
fn different_envs_have_different_streams() {
    let seed = 20_260_930;
    let mut firsts = std::collections::HashSet::new();
    let mut keys = std::collections::HashSet::new();
    for env in 0..4096u32 {
        let rng = Rng::new(seed, env);
        assert!(keys.insert(rng.key()), "key collision at env {env}");
        assert!(
            firsts.insert(rng.block(0, STREAM_RESET, 0)),
            "block collision at env {env}"
        );
    }
}

#[test]
fn different_seeds_and_every_seed_bit_matter() {
    let base = Rng::new(0, 0);
    // a seed that differs only in one of the 64 bits gives a different stream
    for bit in 0..64 {
        let other = Rng::new(1u64 << bit, 0);
        assert_ne!(
            base.key(),
            other.key(),
            "seed bit {bit} did not change the key"
        );
        assert_ne!(base.block(0, 0, 0), other.block(0, 0, 0));
    }
    // and so does every bit of the environment id
    for bit in 0..32 {
        assert_ne!(base.key(), Rng::new(0, 1u32 << bit).key(), "env bit {bit}");
    }
}

#[test]
fn step_stream_and_lane_address_independent_blocks() {
    let rng = Rng::new(7, 1);
    let reference = rng.block(10, STREAM_RESET, 2);
    assert_ne!(reference, rng.block(11, STREAM_RESET, 2));
    assert_ne!(reference, rng.block(10, STREAM_SENSE_NOISE, 2));
    assert_ne!(reference, rng.block(10, STREAM_RESET, 3));
    // the high word of the step is part of the address
    assert_ne!(rng.block(5, 0, 0), rng.block(5 + (1u64 << 32), 0, 0));
}

#[test]
fn unit_conversions_stay_strictly_inside_their_intervals() {
    for w in [0u32, 1, 0x7fff_ffff, 0x8000_0000, u32::MAX - 1, u32::MAX] {
        let u = unit_open(w);
        assert!(u > 0.0 && u < 1.0, "unit_open({w:#x}) = {u}");
        let s = signed_unit(w);
        assert!(s > -1.0 && s < 1.0, "signed_unit({w:#x}) = {s}");
    }
    assert_eq!(unit_open(0), 0.5 / 4_294_967_296.0);
}

#[test]
fn the_stream_is_unbiased_to_a_smoke_test() {
    // Not a statistical certification of Philox (that is the literature's); a
    // guard that the conversions and addressing did not introduce an obvious bias.
    let rng = Rng::new(123, 9);
    let n = 50_000u32;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for lane in 0..n {
        for v in rng.signed_units(0, STREAM_RESET, lane) {
            sum += v;
            sum_sq += v * v;
        }
    }
    let count = f64::from(n) * 4.0;
    let mean = sum / count;
    let var = sum_sq / count - mean * mean;
    assert!(mean.abs() < 0.01, "mean {mean}");
    // a uniform variable on (-1, 1) has variance 1/3
    assert!((var - 1.0 / 3.0).abs() < 0.01, "variance {var}");
}
