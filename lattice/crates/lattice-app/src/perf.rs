//! `LATTICE_PERF=1` instrumentation: how often the interface builds its widget
//! tree (`view()` calls), rebuilds a canvas cache and draws a frame, reported
//! every ten seconds.
//!
//! This is how the "zero frames while idle" rule is measured: with nothing
//! running and nobody touching the window the three numbers stay at zero.
//! `frames` is the number of `RedrawRequested` events the widget tree received
//! (see `ui::frames`): it is what shows a frame that builds no view and
//! tessellates nothing, such as a repaint of cached geometry.
//!
//! Invariants:
//! - With the variable unset, the counters do nothing (one relaxed load) and
//!   no thread exists.
//! - With it set, the reporter is an ordinary thread, not an iced
//!   subscription: a subscription's tick would itself be a message, which
//!   costs a `view()` call and a redraw and would spoil the measurement.
//! - A line is printed when any count is non-zero, and once for the first
//!   idle interval after activity (the proof that it went quiet); further idle
//!   intervals are silent until something happens again.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

static VIEWS: AtomicU64 = AtomicU64::new(0);
static REBUILDS: AtomicU64 = AtomicU64::new(0);
static FRAMES: AtomicU64 = AtomicU64::new(0);
static ENABLED: OnceLock<bool> = OnceLock::new();

/// The reporting interval.
pub const INTERVAL: Duration = Duration::from_secs(10);

/// True when `LATTICE_PERF=1`.
pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var("LATTICE_PERF").is_ok_and(|v| v == "1"))
}

// Tests count on the calling thread, so that tests running side by side (and
// `LATTICE_PERF` unset) do not see each other's views, rebuilds and frames.
#[cfg(test)]
thread_local! {
    static LOCAL: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
    static LOCAL_FRAMES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Take and reset this thread's counts `(views, rebuilds)` (tests only).
#[cfg(test)]
pub fn take_local() -> (u64, u64) {
    LOCAL.with(|c| c.replace((0, 0)))
}

/// Take and reset this thread's frame count (tests only).
#[cfg(test)]
pub fn take_local_frames() -> u64 {
    LOCAL_FRAMES.with(|c| c.replace(0))
}

/// Count one `view()` call.
pub fn view_called() {
    if enabled() {
        VIEWS.fetch_add(1, Ordering::Relaxed);
    }
    #[cfg(test)]
    LOCAL.with(|c| {
        let (v, r) = c.get();
        c.set((v + 1, r));
    });
}

/// Count one canvas cache rebuild (a call of a `Cache::draw` closure).
pub fn cache_rebuilt() {
    if enabled() {
        REBUILDS.fetch_add(1, Ordering::Relaxed);
    }
    #[cfg(test)]
    LOCAL.with(|c| {
        let (v, r) = c.get();
        c.set((v, r + 1));
    });
}

/// Count one frame: the widget tree received `RedrawRequested`.
pub fn frame_drawn() {
    if enabled() {
        FRAMES.fetch_add(1, Ordering::Relaxed);
    }
    #[cfg(test)]
    LOCAL_FRAMES.with(|c| c.set(c.get() + 1));
}

/// Take and reset the counters: `(views, rebuilds, frames)`.
pub fn take() -> (u64, u64, u64) {
    (
        VIEWS.swap(0, Ordering::Relaxed),
        REBUILDS.swap(0, Ordering::Relaxed),
        FRAMES.swap(0, Ordering::Relaxed),
    )
}

/// The line for one interval, or `None` when it stays silent. `idle_reported`
/// remembers whether the current quiet spell has already been reported.
pub fn report_line(
    views: u64,
    rebuilds: u64,
    frames: u64,
    idle_reported: &mut bool,
) -> Option<String> {
    if views != 0 || rebuilds != 0 || frames != 0 {
        *idle_reported = false;
    } else if *idle_reported {
        return None;
    } else {
        *idle_reported = true;
    }
    Some(format!(
        "perf: views={views} rebuilds={rebuilds} frames={frames} (last {} s)",
        INTERVAL.as_secs()
    ))
}

/// Start the reporter thread when `LATTICE_PERF=1`. Does nothing otherwise.
pub fn start_reporter() {
    if !enabled() {
        return;
    }
    let spawned = std::thread::Builder::new()
        .name("lattice-perf".into())
        .spawn(|| {
            let mut idle_reported = false;
            loop {
                std::thread::sleep(INTERVAL);
                let (views, rebuilds, frames) = take();
                if let Some(line) = report_line(views, rebuilds, frames, &mut idle_reported) {
                    eprintln!("{line}");
                }
            }
        });
    if let Err(error) = spawned {
        eprintln!("perf: could not start the reporter thread: {error}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activity_is_reported_every_interval() {
        let mut idle = false;
        assert_eq!(
            report_line(12, 3, 7, &mut idle).as_deref(),
            Some("perf: views=12 rebuilds=3 frames=7 (last 10 s)")
        );
        assert_eq!(
            report_line(0, 1, 0, &mut idle).as_deref(),
            Some("perf: views=0 rebuilds=1 frames=0 (last 10 s)")
        );
        assert_eq!(
            report_line(1, 0, 0, &mut idle).as_deref(),
            Some("perf: views=1 rebuilds=0 frames=0 (last 10 s)")
        );
    }

    #[test]
    fn frames_alone_count_as_activity() {
        // A frame that repaints cached geometry builds no view and no cache: it is
        // exactly the activity `frames=` exists to show.
        let mut idle = false;
        assert!(report_line(0, 0, 0, &mut idle).is_some(), "first idle line");
        assert_eq!(report_line(0, 0, 0, &mut idle), None);
        assert_eq!(
            report_line(0, 0, 3, &mut idle).as_deref(),
            Some("perf: views=0 rebuilds=0 frames=3 (last 10 s)"),
            "frames with no views and no rebuilds are reported"
        );
        assert!(!idle, "and end the quiet spell");
        assert!(
            report_line(0, 0, 0, &mut idle).is_some(),
            "the next quiet interval is reported once, as after any activity"
        );
        assert_eq!(report_line(0, 0, 0, &mut idle), None);
    }

    #[test]
    fn the_first_idle_interval_is_reported_once_then_silence_until_activity() {
        let mut idle = false;
        assert_eq!(
            report_line(0, 0, 0, &mut idle).as_deref(),
            Some("perf: views=0 rebuilds=0 frames=0 (last 10 s)")
        );
        assert_eq!(report_line(0, 0, 0, &mut idle), None);
        assert_eq!(report_line(0, 0, 0, &mut idle), None);
        // Activity resumes, then goes quiet again: the new quiet spell is reported once.
        assert!(report_line(4, 1, 2, &mut idle).is_some());
        assert!(report_line(0, 0, 0, &mut idle).is_some());
        assert_eq!(report_line(0, 0, 0, &mut idle), None);
    }

    #[test]
    fn take_resets_the_counters() {
        // The counters only move when LATTICE_PERF=1, which the test process does
        // not set, so drive them directly.
        VIEWS.fetch_add(3, Ordering::Relaxed);
        REBUILDS.fetch_add(2, Ordering::Relaxed);
        FRAMES.fetch_add(5, Ordering::Relaxed);
        assert_eq!(take(), (3, 2, 5));
        assert_eq!(take(), (0, 0, 0));
    }

    #[test]
    fn frames_are_counted_per_thread_in_tests() {
        let _ = take_local_frames();
        frame_drawn();
        frame_drawn();
        assert_eq!(take_local_frames(), 2);
        assert_eq!(take_local_frames(), 0);
        std::thread::spawn(|| {
            frame_drawn();
            assert_eq!(take_local_frames(), 1);
        })
        .join()
        .unwrap();
        assert_eq!(
            take_local_frames(),
            0,
            "another thread's frames are not ours"
        );
    }
}
