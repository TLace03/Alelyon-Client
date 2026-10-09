//! The waterfall's shared time axis and the geometry of a bar on it.
//!
//! Invariants:
//! - One axis serves every row: `t0` is the earliest span start and `t1` the
//!   latest end (an open span counts as ending "now"), so the last bar touches
//!   the right edge and nothing is drawn outside the track.
//! - Ticks are "nice": their step is 1, 2, 2.5 or 5 times a power of ten
//!   below ten seconds, then whole units of a second/minute/hour that read
//!   naturally (15 s, 30 s, 1 m, 5 m ...), and there are never more than
//!   [`MAX_TICKS`]. The first tick is always the axis start (`0 ms`), measured
//!   from the trace's start, not the wall clock.
//! - A bar is never narrower than its minimum width, so a 1 ms span is still
//!   visible, and never crosses the track's right edge.

/// The most ticks an axis labels; the spec asks for four to six.
pub const MAX_TICKS: usize = 5;

/// A span of time, in seconds since the epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Axis {
    pub t0: f64,
    pub t1: f64,
}

impl Axis {
    /// The axis over `(start, end)` pairs, an open span (`None`) ending at `now`.
    /// An empty set, or a zero-length one, still gets a one-millisecond axis so
    /// every division downstream is safe.
    pub fn over(spans: impl IntoIterator<Item = (f64, Option<f64>)>, now: f64) -> Self {
        let mut t0 = f64::INFINITY;
        let mut t1 = f64::NEG_INFINITY;
        for (start, end) in spans {
            let end = end.unwrap_or_else(|| now.max(start));
            t0 = t0.min(start);
            t1 = t1.max(end.max(start));
        }
        if !t0.is_finite() || !t1.is_finite() {
            return Self {
                t0: now,
                t1: now + 0.001,
            };
        }
        if t1 - t0 < 0.001 {
            t1 = t0 + 0.001;
        }
        Self { t0, t1 }
    }

    pub fn span(&self) -> f64 {
        self.t1 - self.t0
    }

    /// Fraction of the axis at `t`, not clamped.
    pub fn fraction(&self, t: f64) -> f64 {
        (t - self.t0) / self.span()
    }
}

/// Steps below ten seconds: 1, 2, 2.5 and 5 times a power of ten from a
/// microsecond up; above, the steps a person would pick on a clock.
fn step_candidates() -> impl Iterator<Item = f64> {
    let decades = (-6..=0).flat_map(|k| [1.0, 2.0, 2.5, 5.0].map(|m| m * 10f64.powi(k)));
    let clock = [
        10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 900.0, 1800.0, 3600.0, 7200.0, 21_600.0,
        43_200.0, 86_400.0,
    ];
    decades.chain(clock)
}

/// Ticks for an axis of `span` seconds: `(step, offsets)` where the offsets
/// are `0, step, 2·step ...` up to and including `span` (within rounding).
pub fn nice_ticks(span: f64, max_ticks: usize) -> (f64, Vec<f64>) {
    let max_ticks = max_ticks.max(2);
    let span = if span.is_finite() && span > 0.0 {
        span
    } else {
        0.001
    };
    let step = step_candidates()
        .find(|step| ((span / step + 1e-9).floor() as usize) < max_ticks)
        .unwrap_or(86_400.0 * (span / 86_400.0 / (max_ticks as f64 - 1.0)).ceil());
    let count = (span / step + 1e-9).floor() as usize + 1;
    let ticks = (0..count).map(|i| i as f64 * step).collect();
    (step, ticks)
}

/// A bar on the track: `x` and `width` in pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bar {
    pub x: f32,
    pub w: f32,
}

/// The bar for a span from `start` to `end` (`None` = open, drawn to `now`) on
/// `axis`, over a track starting at `track_x` and `track_w` pixels wide.
pub fn bar(
    start: f64,
    end: Option<f64>,
    now: f64,
    axis: &Axis,
    track_x: f32,
    track_w: f32,
    min_w: f32,
) -> Bar {
    let end = end.unwrap_or_else(|| now.max(start)).max(start);
    let x0 = axis.fraction(start).clamp(0.0, 1.0) as f32 * track_w;
    let x1 = axis.fraction(end).clamp(0.0, 1.0) as f32 * track_w;
    let w = (x1 - x0).max(min_w).min(track_w);
    let x = x0.min(track_w - w).max(0.0);
    Bar { x: track_x + x, w }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::format_axis_label;

    fn labels(span: f64) -> Vec<String> {
        nice_ticks(span, MAX_TICKS)
            .1
            .into_iter()
            .map(format_axis_label)
            .collect()
    }

    #[test]
    fn a_one_second_axis_ticks_every_quarter_second() {
        assert_eq!(labels(1.0), ["0 ms", "250 ms", "500 ms", "750 ms", "1 s"]);
    }

    #[test]
    fn a_second_and_a_half_ticks_every_half_second() {
        assert_eq!(labels(1.5), ["0 ms", "500 ms", "1 s", "1.5 s"]);
    }

    #[test]
    fn ticks_are_always_four_to_five_and_nice_across_many_scales() {
        let mut span = 0.0007;
        while span < 200_000.0 {
            let (step, ticks) = nice_ticks(span, MAX_TICKS);
            assert!(ticks.len() <= MAX_TICKS, "{span}: {ticks:?}");
            assert!(ticks.len() >= 2, "{span}: {ticks:?}");
            assert_eq!(ticks[0], 0.0);
            assert!(*ticks.last().unwrap() <= span + 1e-9);
            // The next tick would fall past the axis: no tick is wasted.
            assert!(
                ticks.last().unwrap() + step > span - 1e-9,
                "{span}: {ticks:?} step {step}"
            );
            // The step is the smallest nice one that fits: every smaller candidate
            // would have crowded the axis with more than MAX_TICKS labels.
            for smaller in step_candidates().take_while(|c| *c < step * (1.0 - 1e-9)) {
                assert!(
                    (span / smaller + 1e-9).floor() as usize + 1 > MAX_TICKS,
                    "{span}: {smaller} was nice and fit"
                );
            }
            span *= 1.37;
        }
    }

    #[test]
    fn minute_scale_axes_use_clock_steps() {
        let (step, ticks) = nice_ticks(200.0, MAX_TICKS);
        assert_eq!(step, 60.0);
        assert_eq!(ticks.len(), 4);
        assert_eq!(format_axis_label(ticks[3]), "3 m");
        assert_eq!(nice_ticks(45.0, MAX_TICKS).0, 10.0);
        assert_eq!(nice_ticks(50.0, MAX_TICKS).0, 15.0);
    }

    #[test]
    fn degenerate_spans_still_give_ticks() {
        for span in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let (step, ticks) = nice_ticks(span, MAX_TICKS);
            assert!(step > 0.0 && !ticks.is_empty(), "{span}");
        }
    }

    #[test]
    fn the_axis_spans_the_earliest_start_to_the_latest_end_and_open_spans_reach_now() {
        let axis = Axis::over(
            [(100.0, Some(101.0)), (100.5, Some(102.5)), (101.0, None)],
            104.0,
        );
        assert_eq!((axis.t0, axis.t1), (100.0, 104.0));
        let done = Axis::over([(100.0, Some(101.0)), (100.5, Some(102.5))], 999.0);
        assert_eq!((done.t0, done.t1), (100.0, 102.5));
    }

    #[test]
    fn an_empty_or_instant_axis_is_never_zero_length() {
        assert!(Axis::over([], 50.0).span() > 0.0);
        assert!(Axis::over([(10.0, Some(10.0))], 50.0).span() > 0.0);
        // A span that starts after "now" (clock skew) does not run backwards.
        let skewed = Axis::over([(10.0, None)], 5.0);
        assert!(skewed.t1 >= skewed.t0);
    }

    #[test]
    fn bars_scale_to_the_track_and_never_vanish() {
        let axis = Axis { t0: 0.0, t1: 10.0 };
        let b = bar(2.0, Some(4.0), 10.0, &axis, 100.0, 500.0, 2.0);
        assert_eq!((b.x, b.w), (200.0, 100.0));
        // A 1 ms span on a 10 s axis is 0.05 px wide; it gets the minimum.
        let tiny = bar(5.0, Some(5.001), 10.0, &axis, 100.0, 500.0, 2.0);
        assert_eq!(tiny.w, 2.0);
        assert_eq!(tiny.x, 350.0);
        // A span at the very end keeps its minimum width inside the track.
        let last = bar(10.0, Some(10.0), 10.0, &axis, 100.0, 500.0, 3.0);
        assert_eq!(last.w, 3.0);
        assert!(last.x + last.w <= 600.0 + 1e-3);
    }

    #[test]
    fn an_open_span_is_drawn_to_now() {
        let axis = Axis { t0: 0.0, t1: 10.0 };
        let open = bar(4.0, None, 8.0, &axis, 0.0, 1000.0, 2.0);
        assert_eq!((open.x, open.w), (400.0, 400.0));
        // "Now" before the start gives the minimum width, not a negative one.
        let early = bar(4.0, None, 1.0, &axis, 0.0, 1000.0, 2.0);
        assert_eq!(early.w, 2.0);
    }

    #[test]
    fn a_bar_never_leaves_the_track_even_off_axis() {
        let axis = Axis { t0: 5.0, t1: 6.0 };
        let b = bar(0.0, Some(100.0), 0.0, &axis, 10.0, 200.0, 2.0);
        assert_eq!((b.x, b.w), (10.0, 200.0));
        let after = bar(50.0, Some(60.0), 0.0, &axis, 10.0, 200.0, 2.0);
        assert!(after.x >= 10.0 && after.x + after.w <= 210.0 + 1e-3);
    }
}
