//! Audio: one continuous mono stream, delivered a chunk per tick.
//!
//! Sound is the one sense with no gaps. Each tick's bundle carries the chunk of
//! the stream that belongs to it, and the stream is numbered in samples so a
//! consumer can prove it lost none.
//!
//! Invariants:
//! - Contract v0 audio is mono, 32-bit float, at 16,000 Hz.
//! - Chunk `n + 1` starts at the sample index where chunk `n` ended:
//!   `next.first_sample_index == prev.first_sample_index + prev.sample_count()`.
//!   [`AudioContinuity::check`] verifies it. A chunk may be empty (zero
//!   samples); an empty chunk still sits at its `first_sample_index`.
//! - The samples are either on the host (`Host`) or stay on the device
//!   (`Device`), like frames; in both cases they are `f32`.
//! - The contract fixes the sample type and rate, not the amplitude scale.

use serde::{Deserialize, Serialize};

use crate::check::all_finite_f32;
use crate::{ContractError, DeviceBufferId};

/// The sample rate of contract v0 audio.
pub const AUDIO_SAMPLE_RATE_HZ_V0: u32 = 16_000;

/// Where a chunk's samples are.
///
/// JSON: an object with a `kind` of `"host"` (with `values`) or `"device"`
/// (with `buffer`, `byte_offset` and `count`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AudioSamples {
    /// The samples, mono, in time order.
    Host {
        /// One `f32` per sample.
        values: Vec<f32>,
    },
    /// The samples are `f32` in a device buffer.
    Device {
        /// The buffer.
        buffer: DeviceBufferId,
        /// Where the first sample starts, a multiple of 4.
        byte_offset: u64,
        /// How many samples (4 bytes each) there are.
        count: u64,
    },
}

/// One tick's audio.
///
/// Invariants (checked by [`AudioChunk::validate`]):
/// - `sample_rate_hz` is [`AUDIO_SAMPLE_RATE_HZ_V0`].
/// - Host samples are finite. A device reference's `byte_offset` is a multiple
///   of 4 and its extent fits in a `u64`.
/// - `first_sample_index + sample_count()` fits in a `u64`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioChunk {
    /// Samples per second; 16,000 in v0.
    pub sample_rate_hz: u32,
    /// The index, in the whole stream since the episode began, of this chunk's
    /// first sample.
    pub first_sample_index: u64,
    /// The samples, or where they are.
    pub samples: AudioSamples,
}

impl AudioChunk {
    /// A v0 chunk with its samples on the host.
    pub fn host(first_sample_index: u64, values: Vec<f32>) -> Self {
        Self {
            sample_rate_hz: AUDIO_SAMPLE_RATE_HZ_V0,
            first_sample_index,
            samples: AudioSamples::Host { values },
        }
    }

    /// A v0 chunk with no samples yet: the channel exists, with no content.
    pub fn empty(first_sample_index: u64) -> Self {
        Self::host(first_sample_index, Vec::new())
    }

    /// How many samples the chunk holds.
    pub fn sample_count(&self) -> u64 {
        match &self.samples {
            AudioSamples::Host { values } => values.len() as u64,
            AudioSamples::Device { count, .. } => *count,
        }
    }

    /// The index the next chunk must start at. `None` if it would not fit a `u64`.
    pub fn end_sample_index(&self) -> Option<u64> {
        self.first_sample_index.checked_add(self.sample_count())
    }

    /// Checks the sample rate, the samples and the index range.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.sample_rate_hz != AUDIO_SAMPLE_RATE_HZ_V0 {
            return Err(ContractError::OutOfRange {
                field: "sample_rate_hz",
                reason: "v0 audio is 16000 Hz",
            });
        }
        match &self.samples {
            AudioSamples::Host { values } => all_finite_f32("samples", values)?,
            AudioSamples::Device {
                byte_offset, count, ..
            } => {
                if byte_offset % 4 != 0 {
                    return Err(ContractError::Misaligned {
                        field: "byte_offset",
                        alignment: 4,
                        found: *byte_offset,
                    });
                }
                let fits = count
                    .checked_mul(4)
                    .and_then(|bytes| byte_offset.checked_add(bytes))
                    .is_some();
                if !fits {
                    return Err(ContractError::OutOfRange {
                        field: "count",
                        reason: "the samples' extent does not fit in 64 bits",
                    });
                }
            }
        }
        if self.end_sample_index().is_none() {
            return Err(ContractError::OutOfRange {
                field: "first_sample_index",
                reason: "the chunk's end does not fit in 64 bits",
            });
        }
        Ok(())
    }
}

/// The continuity rule between two consecutive chunks.
pub struct AudioContinuity;

impl AudioContinuity {
    /// Checks that `next` follows `prev` with no gap and no overlap, at one
    /// sample rate: `next.first_sample_index` must equal `prev`'s end.
    pub fn check(prev: &AudioChunk, next: &AudioChunk) -> Result<(), ContractError> {
        if prev.sample_rate_hz != next.sample_rate_hz {
            return Err(ContractError::OutOfRange {
                field: "sample_rate_hz",
                reason: "the sample rate changed between chunks",
            });
        }
        let expected = prev.end_sample_index().ok_or(ContractError::OutOfRange {
            field: "first_sample_index",
            reason: "the previous chunk's end does not fit in 64 bits",
        })?;
        if next.first_sample_index != expected {
            return Err(ContractError::AudioDiscontinuity {
                expected_first_sample_index: expected,
                found_first_sample_index: next.first_sample_index,
            });
        }
        Ok(())
    }
}
