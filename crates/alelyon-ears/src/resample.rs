//! From whatever the device or file delivers to the recogniser's 16 kHz mono.
//!
//! A streaming windowed-sinc resampler: each output sample is a weighted sum of the input around its exact
//! fractional position, with a low-pass at 95% of the lower Nyquist so that nothing above 8 kHz folds back
//! into speech when 48 kHz comes down to 16 kHz. The weights are renormalised per output sample, so a
//! constant input gives exactly that constant out at every phase.
//!
//! Output sample `n` sits at input position `n * in_rate / out_rate`, kept as an exact integer part and
//! remainder rather than as an accumulating float, so every decision about which input an output needs is
//! the same however the input arrives: chunks of any size give the same samples, and as many, as one call
//! over the whole signal (`tests`).

use std::f64::consts::PI;

/// Zero crossings of the sinc kept on each side of the centre, at the lower of the two rates.
const ZERO_CROSSINGS: f64 = 16.0;
/// The low-pass sits this far below the lower Nyquist frequency.
const CUTOFF: f64 = 0.95;

pub struct Resampler {
    in_rate: u64,
    out_rate: u64,
    /// Cycles per input sample.
    cutoff: f64,
    /// Kernel half-width in input samples.
    half_width: f64,
    reach: u64,
    /// `buffer[0]` is input sample number `dropped`.
    buffer: Vec<f32>,
    dropped: u64,
    received: u64,
    /// The next output sample's number.
    next: u64,
    flushed: bool,
}

impl Resampler {
    pub fn new(in_rate: u32, out_rate: u32) -> Self {
        assert!(in_rate > 0 && out_rate > 0, "sample rates must be positive");
        let lower = (f64::from(out_rate) / f64::from(in_rate)).min(1.0);
        let half_width = ZERO_CROSSINGS / lower;
        Self {
            in_rate: u64::from(in_rate),
            out_rate: u64::from(out_rate),
            cutoff: 0.5 * lower * CUTOFF,
            half_width,
            reach: half_width.ceil() as u64 + 1,
            buffer: Vec::new(),
            dropped: 0,
            received: 0,
            next: 0,
            flushed: false,
        }
    }

    /// Output for this chunk of input; samples whose kernel reaches past it wait for the next chunk.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        assert!(!self.flushed, "process after flush");
        self.buffer.extend_from_slice(input);
        self.received += input.len() as u64;
        let mut out = Vec::new();
        loop {
            let (whole, _) = self.centre(self.next);
            if whole + self.reach >= self.received {
                break;
            }
            out.push(self.sample(self.next));
            self.next += 1;
        }
        // Drop what no future output can reach.
        let (whole, _) = self.centre(self.next);
        let keep_from = whole.saturating_sub(self.reach).max(self.dropped);
        let drop = (keep_from - self.dropped) as usize;
        if drop > 0 {
            self.buffer.drain(..drop.min(self.buffer.len()));
            self.dropped = keep_from;
        }
        out
    }

    /// The rest of the output, as if the input continued in silence: every output sample whose position
    /// falls inside the input received.
    pub fn flush(&mut self) -> Vec<f32> {
        self.flushed = true;
        let mut out = Vec::new();
        while self.next * self.in_rate < self.received * self.out_rate {
            out.push(self.sample(self.next));
            self.next += 1;
        }
        self.buffer.clear();
        out
    }

    /// Exact position of output sample `n` in the input: whole samples, then the fraction.
    fn centre(&self, n: u64) -> (u64, f64) {
        let num = u128::from(n) * u128::from(self.in_rate);
        let out = u128::from(self.out_rate);
        ((num / out) as u64, (num % out) as f64 / out as f64)
    }

    fn sample(&self, n: u64) -> f32 {
        let (whole, frac) = self.centre(n);
        let first = whole as i64 - self.reach as i64;
        let last = whole as i64 + self.reach as i64;
        let mut acc = 0.0f64;
        let mut weight_sum = 0.0f64;
        for i in first..=last {
            // distance from the output's position, in input samples
            let t = (i - whole as i64) as f64 - frac;
            let w = self.kernel(t);
            if w == 0.0 {
                continue;
            }
            weight_sum += w;
            // before the start and after the end the input is silence
            if i >= self.dropped as i64 && (i as u64) < self.received {
                acc += w * f64::from(self.buffer[(i as u64 - self.dropped) as usize]);
            }
        }
        if weight_sum.abs() > 1e-12 {
            (acc / weight_sum) as f32
        } else {
            0.0
        }
    }

    fn kernel(&self, t: f64) -> f64 {
        if t.abs() >= self.half_width {
            return 0.0;
        }
        let x = 2.0 * self.cutoff * t;
        let sinc = if x.abs() < 1e-12 { 1.0 } else { (PI * x).sin() / (PI * x) };
        // Blackman window over [-half_width, half_width].
        let u = (t / self.half_width + 1.0) / 2.0;
        let window = 0.42 - 0.5 * (2.0 * PI * u).cos() + 0.08 * (4.0 * PI * u).cos();
        2.0 * self.cutoff * sinc * window
    }
}

/// Interleaved frames of `channels` samples, averaged to one channel.
pub fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Everything at once: `input` at `in_rate` to `out_rate`.
pub fn resample_all(input: &[f32], in_rate: u32, out_rate: u32) -> Vec<f32> {
    if in_rate == out_rate {
        return input.to_vec();
    }
    let mut resampler = Resampler::new(in_rate, out_rate);
    let mut out = resampler.process(input);
    out.extend(resampler.flush());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f64, rate: u32, seconds: f64) -> Vec<f32> {
        let n = (f64::from(rate) * seconds) as usize;
        (0..n).map(|i| (2.0 * PI * freq * i as f64 / f64::from(rate)).sin() as f32 * 0.5).collect()
    }

    /// Power of `freq` in `signal` (Goertzel), relative to a full-length unit-amplitude tone.
    fn power_at(signal: &[f32], freq: f64, rate: u32) -> f64 {
        let w = 2.0 * PI * freq / f64::from(rate);
        let coeff = 2.0 * w.cos();
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for &x in signal {
            let s = f64::from(x) + coeff * s1 - s2;
            s2 = s1;
            s1 = s;
        }
        let power = s1 * s1 + s2 * s2 - coeff * s1 * s2;
        power / (signal.len() as f64 * signal.len() as f64 / 4.0)
    }

    #[test]
    fn a_constant_stays_that_constant() {
        let out = resample_all(&vec![0.25; 48_000], 48_000, 16_000);
        // away from the edges, where the input's own start and end are steps
        for &s in &out[200..out.len() - 200] {
            assert!((s - 0.25).abs() < 1e-5, "{s}");
        }
    }

    #[test]
    fn the_length_follows_the_ratio() {
        for (inr, outr) in [(48_000u32, 16_000u32), (44_100, 16_000), (22_050, 16_000), (8_000, 16_000)] {
            let out = resample_all(&vec![0.0; inr as usize * 2], inr, outr);
            assert_eq!(out.len(), outr as usize * 2, "{inr}->{outr}");
        }
    }

    #[test]
    fn speech_band_tones_pass_and_what_would_alias_is_removed() {
        let input = tone(1_000.0, 48_000, 1.0);
        let out = resample_all(&input, 48_000, 16_000);
        let kept = power_at(&out[400..out.len() - 400], 1_000.0, 16_000);
        assert!((kept - 0.25).abs() < 0.01, "a 1 kHz tone keeps its power: {kept}");

        // 12 kHz cannot exist at 16 kHz; unfiltered it would fold to 4 kHz.
        let high = tone(12_000.0, 48_000, 1.0);
        let folded = resample_all(&high, 48_000, 16_000);
        let leak = power_at(&folded[400..folded.len() - 400], 4_000.0, 16_000);
        assert!(leak < 0.25 * 1e-4, "folded power must be below -40 dB: {leak}");
    }

    #[test]
    fn chunks_of_any_size_give_the_one_shot_result() {
        for (inr, outr) in [(44_100u32, 16_000u32), (48_000, 16_000), (8_000, 16_000)] {
            let input = tone(440.0, inr, 0.5);
            let whole = resample_all(&input, inr, outr);
            let mut resampler = Resampler::new(inr, outr);
            let mut pieces = Vec::new();
            let mut at = 0;
            let mut size = 1;
            while at < input.len() {
                let end = (at + size).min(input.len());
                pieces.extend(resampler.process(&input[at..end]));
                at = end;
                size = size * 7 % 997 + 1;
            }
            pieces.extend(resampler.flush());
            assert_eq!(pieces.len(), whole.len(), "{inr}->{outr}");
            for (a, b) in pieces.iter().zip(&whole) {
                assert!((a - b).abs() < 1e-6, "{inr}->{outr}");
            }
        }
    }

    #[test]
    fn the_buffer_stays_bounded_while_streaming() {
        let mut resampler = Resampler::new(48_000, 16_000);
        for _ in 0..1_000 {
            resampler.process(&[0.1; 480]);
        }
        assert!(resampler.buffer.len() < 480 + 4 * resampler.reach as usize, "{}", resampler.buffer.len());
    }

    #[test]
    fn downmix_averages_the_channels() {
        assert_eq!(downmix(&[1.0, 0.0, 0.5, 0.5], 2), vec![0.5, 0.5]);
        assert_eq!(downmix(&[0.3, 0.6], 1), vec![0.3, 0.6]);
    }
}
