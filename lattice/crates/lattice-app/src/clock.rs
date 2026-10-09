//! Time and number formatting for the interface: pure functions over seconds
//! since the Unix epoch, so every one of them is testable without a clock.
//!
//! Invariants:
//! - Nothing here reads the wall clock except [`now`]; `view()` never calls it.
//!   The application samples `now` when a message arrives and passes it down,
//!   which keeps drawing deterministic and screenshots reproducible.
//! - Spans carry ISO 8601 timestamps in UTC as Python's `isoformat()` writes
//!   them (`2026-09-30T05:21:05.123456+00:00`, microseconds omitted when zero).
//!   [`parse_iso`] accepts that form plus `Z` and other numeric offsets;
//!   [`iso_from_epoch`] writes it.
//! - Clock times are shown in UTC and say so. The local UTC offset needs either
//!   an operating-system call (unsafe) or a new dependency, and a wrong local
//!   time is worse than an honest UTC one.

use std::time::{SystemTime, UNIX_EPOCH};

use lattice_protocol::RunStatus;

/// "Today" in the runs list is the last 24 hours, not the calendar day: the
/// local calendar day needs the local UTC offset (see the module header), and a
/// run that finished a minute ago must never be filed under "Earlier".
pub const TODAY_WINDOW_SECS: f64 = 86_400.0;

/// Seconds since the Unix epoch, now.
pub fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's
/// `days_from_civil`).
pub fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (i64::from(month) + 9) % 12;
    let doy = (153 * mp + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The inverse of [`days_from_civil`].
pub fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn digits(bytes: &[u8]) -> Option<i64> {
    if bytes.is_empty() || !bytes.iter().all(u8::is_ascii_digit) {
        return None;
    }
    bytes.iter().try_fold(0i64, |acc, b| {
        acc.checked_mul(10)?.checked_add(i64::from(b - b'0'))
    })
}

/// Parse `YYYY-MM-DDTHH:MM:SS[.ffffff][Z|+HH:MM|-HH:MM]` into epoch seconds.
/// Anything else is `None`.
pub fn parse_iso(text: &str) -> Option<f64> {
    let b = text.as_bytes();
    if b.len() < 19
        || b[4] != b'-'
        || b[7] != b'-'
        || !(b[10] == b'T' || b[10] == b' ')
        || b[13] != b':'
        || b[16] != b':'
    {
        return None;
    }
    let year = digits(&b[0..4])?;
    let month = digits(&b[5..7])? as u32;
    let day = digits(&b[8..10])? as u32;
    let hour = digits(&b[11..13])?;
    let minute = digits(&b[14..16])?;
    let second = digits(&b[17..19])?;
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 60
    {
        return None;
    }
    let mut rest = &b[19..];
    let mut fraction = 0.0;
    if rest.first() == Some(&b'.') {
        let end = rest[1..]
            .iter()
            .position(|c| !c.is_ascii_digit())
            .map_or(rest.len(), |p| p + 1);
        let frac = &rest[1..end];
        if frac.is_empty() {
            return None;
        }
        // Only the first nine digits matter to an f64 of epoch seconds.
        let used = &frac[..frac.len().min(9)];
        fraction = digits(used)? as f64 / 10f64.powi(used.len() as i32);
        rest = &rest[end..];
    }
    let offset = match rest {
        [] | [b'Z'] | [b'z'] => 0,
        [sign @ (b'+' | b'-'), h1, h2, b':', m1, m2] => {
            let h = digits(&[*h1, *h2])?;
            let m = digits(&[*m1, *m2])?;
            if h > 23 || m > 59 {
                return None;
            }
            let secs = h * 3600 + m * 60;
            if *sign == b'+' { secs } else { -secs }
        }
        _ => return None,
    };
    let days = days_from_civil(year, month, day);
    Some((days * 86_400 + hour * 3600 + minute * 60 + second - offset) as f64 + fraction)
}

/// Epoch seconds as Python's `datetime.isoformat()` writes a UTC time.
pub fn iso_from_epoch(t: f64) -> String {
    let micros = (t * 1e6).round() as i64;
    let secs = micros.div_euclid(1_000_000);
    let frac = micros.rem_euclid(1_000_000);
    let (y, mo, d) = civil_from_days(secs.div_euclid(86_400));
    let sod = secs.rem_euclid(86_400);
    let (h, mi, s) = (sod / 3600, (sod % 3600) / 60, sod % 60);
    if frac == 0 {
        format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}+00:00")
    } else {
        format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}.{frac:06}+00:00")
    }
}

/// `05:21:05.123 UTC`.
pub fn clock_time(t: f64) -> String {
    let millis = (t * 1000.0).floor() as i64;
    let sod = millis.div_euclid(1000).rem_euclid(86_400);
    let ms = millis.rem_euclid(1000);
    format!(
        "{:02}:{:02}:{:02}.{ms:03} UTC",
        sod / 3600,
        (sod % 3600) / 60,
        sod % 60
    )
}

const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// `Sep 28` (UTC date).
pub fn short_date(t: f64) -> String {
    let (_, m, d) = civil_from_days((t.floor() as i64).div_euclid(86_400));
    format!("{} {d}", MONTHS[(m - 1) as usize])
}

/// A number with thousands separators: `1,200`.
pub fn format_count(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// `1,200 → 34 tokens`, with `?` for a half the model server did not report.
pub fn format_usage(input: Option<u64>, output: Option<u64>) -> String {
    let half = |n: Option<u64>| n.map_or_else(|| "?".to_string(), format_count);
    format!("{} → {} tokens", half(input), half(output))
}

/// Up to three decimals, trailing zeros trimmed: `1.5`, `250`, `0.5`.
fn trimmed(x: f64) -> String {
    let s = format!("{x:.3}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() {
        "0".to_string()
    } else {
        s.to_string()
    }
}

/// A duration for a row or header: `<1 ms`, `2.4 ms`, `250 ms`, `1.24 s`,
/// `12.3 s`, `2 m 03 s`, `1 h 05 m`.
pub fn format_duration(secs: f64) -> String {
    if secs.is_nan() || secs <= 0.0 {
        return "0 ms".to_string();
    }
    if secs < 0.001 {
        return "<1 ms".to_string();
    }
    if secs < 1.0 {
        let ms = secs * 1000.0;
        return if ms < 10.0 {
            format!("{ms:.1} ms")
        } else if ms.round() < 1000.0 {
            format!("{:.0} ms", ms)
        } else {
            "1.00 s".to_string()
        };
    }
    if secs < 10.0 {
        return format!("{secs:.2} s");
    }
    if secs < 60.0 {
        return format!("{secs:.1} s");
    }
    let total = secs.round() as u64;
    if total < 3600 {
        format!("{} m {:02} s", total / 60, total % 60)
    } else {
        format!("{} h {:02} m", total / 3600, (total % 3600) / 60)
    }
}

/// An axis tick label: `0 ms`, `250 ms`, `1.5 s`, `1 m 30 s`, `1 h`.
pub fn format_axis_label(t: f64) -> String {
    if t.abs() < 1e-12 {
        return "0 ms".to_string();
    }
    if t < 1.0 {
        return format!("{} ms", trimmed(t * 1000.0));
    }
    if t < 60.0 {
        return format!("{} s", trimmed(t));
    }
    let total = t.round() as u64;
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    match (h, m, s) {
        (0, m, 0) => format!("{m} m"),
        (0, m, s) => format!("{m} m {s} s"),
        (h, 0, 0) => format!("{h} h"),
        (h, m, 0) => format!("{h} h {m} m"),
        (h, m, s) => format!("{h} h {m} m {s} s"),
    }
}

/// How long ago, for the runs list.
pub fn format_relative(t: f64, now: f64) -> String {
    let d = now - t;
    if d < 10.0 {
        "just now".to_string()
    } else if d < 60.0 {
        format!("{} s ago", d as u64)
    } else if d < 3600.0 {
        format!("{} min ago", (d / 60.0) as u64)
    } else if d < 86_400.0 {
        format!("{} h ago", (d / 3600.0) as u64)
    } else if d < 2.0 * 86_400.0 {
        "yesterday".to_string()
    } else {
        short_date(t)
    }
}

/// The group a run is listed under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RunGroup {
    Running,
    Today,
    Earlier,
}

impl RunGroup {
    pub fn title(self) -> &'static str {
        match self {
            RunGroup::Running => "Running",
            RunGroup::Today => "Today",
            RunGroup::Earlier => "Earlier",
        }
    }
}

pub fn group_of(status: RunStatus, created_at: f64, now: f64) -> RunGroup {
    if status.is_active() {
        RunGroup::Running
    } else if now - created_at < TODAY_WINDOW_SECS {
        RunGroup::Today
    } else {
        RunGroup::Earlier
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_dates_round_trip_across_centuries() {
        for days in [
            -800_000i64,
            -1,
            0,
            1,
            59,
            60,
            365,
            18_000,
            20_000,
            2_932_896,
        ] {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days, "{y}-{m}-{d}");
        }
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        // 2024 is a leap year; 2100 is not.
        assert_eq!(civil_from_days(days_from_civil(2024, 2, 29)), (2024, 2, 29));
        assert_eq!(
            civil_from_days(days_from_civil(2100, 3, 1) - 1),
            (2100, 2, 28)
        );
    }

    #[test]
    fn parses_pythons_isoformat_with_and_without_microseconds() {
        let a = parse_iso("2026-09-30T05:21:05.123456+00:00").unwrap();
        let b = parse_iso("2026-09-30T05:21:05+00:00").unwrap();
        assert!((a - b - 0.123456).abs() < 1e-6, "{a} {b}");
        assert_eq!(parse_iso("1970-01-01T00:00:00+00:00"), Some(0.0));
        assert_eq!(parse_iso("1970-01-01T00:00:01Z"), Some(1.0));
        // A numeric offset is subtracted: 01:00 at +01:00 is 00:00 UTC.
        assert_eq!(parse_iso("1970-01-01T01:00:00+01:00"), Some(0.0));
        assert_eq!(parse_iso("1970-01-01T00:00:00-02:30"), Some(9000.0));
    }

    #[test]
    fn refuses_malformed_timestamps() {
        for bad in [
            "",
            "yesterday",
            "2026-13-01T00:00:00+00:00",
            "2026-09-30T25:00:00+00:00",
            "2026-09-30T05:21:05.+00:00",
            "2026-09-30T05:21:05+0000",
            "2026-09-30",
            "2026-09-30T05:21:05+00:00trailing",
        ] {
            assert_eq!(parse_iso(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn iso_round_trips_and_matches_pythons_form() {
        let t = parse_iso("2026-09-30T05:21:05.123456+00:00").unwrap();
        assert_eq!(iso_from_epoch(t), "2026-09-30T05:21:05.123456+00:00");
        // Python omits the fraction when the microsecond is zero.
        assert_eq!(iso_from_epoch(0.0), "1970-01-01T00:00:00+00:00");
        assert_eq!(
            parse_iso(&iso_from_epoch(1_800_000_000.25)),
            Some(1_800_000_000.25)
        );
    }

    #[test]
    fn clock_time_is_utc_with_milliseconds() {
        let t = parse_iso("2026-09-30T05:21:05.123456+00:00").unwrap();
        assert_eq!(clock_time(t), "05:21:05.123 UTC");
        assert_eq!(short_date(t), "Sep 30");
    }

    #[test]
    fn counts_and_usage_read_like_the_spec() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(999), "999");
        assert_eq!(format_count(1200), "1,200");
        assert_eq!(format_count(1_234_567), "1,234,567");
        assert_eq!(format_usage(Some(1200), Some(34)), "1,200 → 34 tokens");
        assert_eq!(format_usage(None, Some(34)), "? → 34 tokens");
        assert_eq!(format_usage(Some(5), None), "5 → ? tokens");
        assert_eq!(format_usage(None, None), "? → ? tokens");
    }

    #[test]
    fn durations_pick_a_readable_unit() {
        assert_eq!(format_duration(-1.0), "0 ms");
        assert_eq!(format_duration(0.0004), "<1 ms");
        assert_eq!(format_duration(0.0024), "2.4 ms");
        assert_eq!(format_duration(0.25), "250 ms");
        assert_eq!(format_duration(0.9996), "1.00 s");
        assert_eq!(format_duration(1.238), "1.24 s");
        assert_eq!(format_duration(12.34), "12.3 s");
        assert_eq!(format_duration(123.0), "2 m 03 s");
        assert_eq!(format_duration(3900.0), "1 h 05 m");
    }

    #[test]
    fn axis_labels_match_the_spec_examples() {
        assert_eq!(format_axis_label(0.0), "0 ms");
        assert_eq!(format_axis_label(0.25), "250 ms");
        assert_eq!(format_axis_label(1.0), "1 s");
        assert_eq!(format_axis_label(1.5), "1.5 s");
        assert_eq!(format_axis_label(1.25), "1.25 s");
        assert_eq!(format_axis_label(0.0005), "0.5 ms");
        assert_eq!(format_axis_label(90.0), "1 m 30 s");
        assert_eq!(format_axis_label(120.0), "2 m");
        assert_eq!(format_axis_label(3600.0), "1 h");
    }

    #[test]
    fn relative_times_step_through_units() {
        let now = 10_000_000.0;
        assert_eq!(format_relative(now - 3.0, now), "just now");
        assert_eq!(format_relative(now - 30.0, now), "30 s ago");
        assert_eq!(format_relative(now - 150.0, now), "2 min ago");
        assert_eq!(format_relative(now - 7300.0, now), "2 h ago");
        assert_eq!(format_relative(now - 90_000.0, now), "yesterday");
        let september = parse_iso("2026-09-01T00:00:00+00:00").unwrap();
        assert_eq!(
            format_relative(september, september + 5.0 * 86_400.0),
            "Sep 1"
        );
        // A timestamp from the future (clock skew) is "just now", never negative.
        assert_eq!(format_relative(now + 500.0, now), "just now");
    }

    #[test]
    fn a_just_finished_run_is_never_filed_under_earlier() {
        let now = 1_000_000.0;
        assert_eq!(
            group_of(RunStatus::Running, now - 5e5, now),
            RunGroup::Running
        );
        assert_eq!(
            group_of(RunStatus::Completed, now - 60.0, now),
            RunGroup::Today
        );
        assert_eq!(
            group_of(RunStatus::Failed, now - 86_000.0, now),
            RunGroup::Today
        );
        assert_eq!(
            group_of(RunStatus::Refused, now - 90_000.0, now),
            RunGroup::Earlier
        );
    }
}
