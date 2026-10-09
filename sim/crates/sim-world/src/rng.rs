//! Philox4x32-10: the simulator's only source of randomness.
//!
//! Philox is a counter-based generator: the output is a pure function of a
//! 128-bit counter and a 64-bit key, with no state to advance, so any number can
//! be recomputed from `(key, counter)` on any device, in any order, by any number
//! of threads, and a GPU kernel can produce exactly the numbers the CPU does.
//!
//! Implemented from the paper: J. K. Salmon, M. A. Moraes, R. O. Dror and
//! D. E. Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3", SC11 (2011),
//! section 2 (the Philox round: two 32x32 -> 64 bit multiplications whose halves
//! are permuted and xored into the other counter words and the key, a Weyl
//! sequence bumping the key between rounds). The constants are the paper's
//! (and the Random123 library's `philox.h`): multipliers `0xD2511F53` and
//! `0xCD9E8D57`, Weyl increments `0x9E3779B9` and `0xBB67AE85` (the golden ratio
//! and sqrt(3) - 1 in 32-bit fixed point), ten rounds.
//!
//! Invariants:
//! - [`philox4x32_10`] is the published function: it reproduces the Random123
//!   known-answer vectors (tested in `tests/rng.rs` against the library's own
//!   `tests/kat_vectors` file).
//! - [`Rng`] is keyed by `(seed, env_id)` and addressed by
//!   `(step, stream, lane)`; the same four numbers always give the same
//!   block, whatever order they are asked in and however many environments
//!   exist. Different `env_id`s (or seeds) give different streams.
//! - The 64-bit seed and the 32-bit environment id do not fit a 64-bit Philox key
//!   together, so the key is derived: it is the first two output words of one
//!   Philox block with the seed as the key and `[env_id, KEY_DOMAIN, 0, 0]` as the
//!   counter (the same construction as `fold_in` in counter-based generators such
//!   as JAX's). The derived key depends on every bit of both inputs.
//! - The counter is `[step_lo, step_hi, stream, lane]`: the full 64-bit step, a
//!   caller-chosen stream id (see the `STREAM_*` constants) and a lane that
//!   addresses independent consumers within a stream (a joint, a thread). One
//!   `(step, stream, lane)` yields one block of four 32-bit words.
//! - Conversions to floats are fixed: [`unit_open`] maps a word to the open
//!   interval (0, 1) in `f64` as `(w + 0.5) / 2^32`; [`signed_unit`] to (-1, 1).
//! - There is no other randomness in the simulator: no clock, no OS entropy, no
//!   `HashMap` iteration order feeds a result.

/// First Philox multiplier (Random123 `PHILOX_M4x32_0`).
const M0: u32 = 0xD251_1F53;
/// Second Philox multiplier (Random123 `PHILOX_M4x32_1`).
const M1: u32 = 0xCD9E_8D57;
/// First Weyl key increment (golden ratio, `PHILOX_W32_0`).
const W0: u32 = 0x9E37_79B9;
/// Second Weyl key increment (sqrt(3) - 1, `PHILOX_W32_1`).
const W1: u32 = 0xBB67_AE85;

/// Domain word of the key-derivation block ("KEY1" in ASCII), so the derived
/// key can never equal the output of a sampling block of the same words.
const KEY_DOMAIN: u32 = 0x4B45_5931;

/// Stream: the randomness of an environment reset (initial pose noise).
pub const STREAM_RESET: u32 = 1;
/// Stream: measurement and process noise of the senses (later phases).
pub const STREAM_SENSE_NOISE: u32 = 2;
/// Stream: contact and friction jitter (later phases).
pub const STREAM_PHYSICS: u32 = 3;

fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let p = u64::from(a) * u64::from(b);
    ((p >> 32) as u32, p as u32)
}

fn round(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (hi0, lo0) = mulhilo(M0, ctr[0]);
    let (hi1, lo1) = mulhilo(M1, ctr[2]);
    [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
}

/// Philox4x32 with ten rounds: the 128-bit block for `counter` under `key`.
pub fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut ctr = counter;
    let mut k = key;
    for r in 0..10 {
        if r > 0 {
            k[0] = k[0].wrapping_add(W0);
            k[1] = k[1].wrapping_add(W1);
        }
        ctr = round(ctr, k);
    }
    ctr
}

/// Maps a 32-bit word to the open interval (0, 1): `(w + 0.5) / 2^32`.
pub fn unit_open(word: u32) -> f64 {
    (f64::from(word) + 0.5) * (1.0 / 4_294_967_296.0)
}

/// Maps a 32-bit word to the open interval (-1, 1): `2 * unit_open(w) - 1`.
pub fn signed_unit(word: u32) -> f64 {
    2.0 * unit_open(word) - 1.0
}

/// The stream generator of one environment under one seed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rng {
    key: [u32; 2],
}

impl Rng {
    /// The generator for `env_id` under `seed`. See the module note for how the
    /// 96 bits of `(seed, env_id)` become the 64-bit key.
    pub fn new(seed: u64, env_id: u32) -> Rng {
        let seed_key = [seed as u32, (seed >> 32) as u32];
        let derived = philox4x32_10([env_id, KEY_DOMAIN, 0, 0], seed_key);
        Rng {
            key: [derived[0], derived[1]],
        }
    }

    /// The 64-bit Philox key this generator uses.
    pub fn key(&self) -> [u32; 2] {
        self.key
    }

    /// The four words at `(step, stream, lane)`.
    pub fn block(&self, step: u64, stream: u32, lane: u32) -> [u32; 4] {
        philox4x32_10([step as u32, (step >> 32) as u32, stream, lane], self.key)
    }

    /// Four values in (-1, 1) at `(step, stream, lane)`.
    pub fn signed_units(&self, step: u64, stream: u32, lane: u32) -> [f64; 4] {
        self.block(step, stream, lane).map(signed_unit)
    }
}
