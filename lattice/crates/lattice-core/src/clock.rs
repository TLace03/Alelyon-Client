//! The clock, as a value that can be replaced.
//!
//! Time enters this crate in three places: the timestamps of run events, the
//! `current_time` tool, and the run store's write throttle. All of them read a
//! [`Clock`], seconds since the Unix epoch, so a test can hold time still or
//! move it by hand, and no assertion depends on the wall clock.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Seconds since the Unix epoch.
pub type Clock = Arc<dyn Fn() -> f64 + Send + Sync>;

/// The system clock. A clock set before 1970 reads as the epoch itself.
pub fn system_clock() -> Clock {
    Arc::new(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs_f64())
            .unwrap_or(0.0)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_system_clock_is_after_2020() {
        assert!(system_clock()() > 1_577_836_800.0);
    }
}
