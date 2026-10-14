//! Paid cloud spending, counted on this PC so a paid part stops at the person's monthly cap.
//!
//! A paid price tier ("Up to $5 a month", "Up to $20 a month", or a number of the person's own) is the person's
//! consent to pay up to its cap and no more. Every paid call is counted here, in US dollars, per calendar month in
//! UTC, in a file that survives a restart ([`Spend`]):
//! - **before** a call is sent, its worst case (the request's input estimate and its output limit, at the model's
//!   published per-token [`Price`]) is reserved under an operating-system lock on a file beside the count, and the call
//!   is refused when that could take the month's total past the cap ([`Refusal::Cap`], until the next month starts);
//! - **after** it answers, the reservation is replaced by the cost the provider reports for it (OpenRouter's
//!   `usage.cost`), else by the provider's reported token counts at the published price. A call that failed after it
//!   was sent is never settled, so its worst case stays counted: the count errs toward the cap, never past it.
//!
//! The count is shared by every part (a cap is the person's whole monthly budget). A count file that cannot be read,
//! locked or written refuses the call: nothing paid goes uncounted. Nothing here moves a part to another model or
//! another provider; a capped part pauses until the month turns or the person raises the cap.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::{Value, json};

/// A model's published price, in US dollars per million tokens.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Price {
    pub input_per_m: f64,
    pub output_per_m: f64,
}

impl Price {
    /// A price that can be counted: both figures finite and not negative. None otherwise (an unknown price).
    pub fn known(input_per_m: f64, output_per_m: f64) -> Option<Price> {
        let ok = |n: f64| n.is_finite() && n >= 0.0;
        (ok(input_per_m) && ok(output_per_m)).then_some(Price { input_per_m, output_per_m })
    }

    /// Whether calls at this price cost nothing.
    pub fn is_free(&self) -> bool {
        self.input_per_m == 0.0 && self.output_per_m == 0.0
    }

    /// What `input` and `output` tokens cost, in US dollars.
    pub fn cost(&self, input: u64, output: u64) -> f64 {
        (input as f64 * self.input_per_m + output as f64 * self.output_per_m) / 1_000_000.0
    }
}

/// A calendar month in UTC, as `YYYY-MM`.
pub fn month_key(at: i64) -> String {
    let t = time::OffsetDateTime::from_unix_timestamp(at).unwrap_or(time::OffsetDateTime::UNIX_EPOCH);
    format!("{:04}-{:02}", t.year(), u8::from(t.month()))
}

/// 00:00 UTC on the first day of the month after the one `at` falls in.
pub fn next_month_start(at: i64) -> i64 {
    let t = time::OffsetDateTime::from_unix_timestamp(at).unwrap_or(time::OffsetDateTime::UNIX_EPOCH);
    let (year, month) = match t.month() {
        time::Month::December => (t.year() + 1, time::Month::January),
        m => (t.year(), m.next()),
    };
    time::Date::from_calendar_date(year, month, 1)
        .map(|d| d.midnight().assume_utc().unix_timestamp())
        .unwrap_or(at + 31 * 86_400)
}

/// Why a paid call may not go.
#[derive(Clone, Debug, PartialEq)]
pub enum Refusal {
    /// It could take the month's total past the cap: the count starts again then (the next month, 00:00 UTC).
    Cap { back_at: i64 },
    /// The count could not be read, locked or written: in one sentence.
    Ledger(String),
}

/// A call's worst case, counted before it is sent; [`Spend::settle`] replaces it with what the call cost.
#[derive(Clone, Debug, PartialEq)]
pub struct Reservation {
    part: String,
    month: String,
    usd: f64,
}

impl Reservation {
    pub fn usd(&self) -> f64 {
        self.usd
    }
}

/// The ledger's file under a home folder.
pub fn path_in(home: &Path) -> PathBuf {
    home.join(".alelyon").join("cloud").join("paid-spend.json")
}

/// The ledger's file for this user, or None without a home folder.
pub fn default_path() -> Option<PathBuf> {
    ["USERPROFILE", "HOME"]
        .iter()
        .find_map(|k| std::env::var_os(k).filter(|v| !v.is_empty()))
        .map(|h| path_in(Path::new(&h)))
}

/// The longest a call waits for the ledger's lock before it is refused.
pub const LOCK_WAIT: std::time::Duration = std::time::Duration::from_secs(3);

const BUSY: &str = "The spending count is busy or cannot be opened, so nothing paid was sent.";

/// The month's paid spending per part, kept in a file. Every change is made under an operating-system lock on a file
/// beside it (`paid-spend.lock`), held for the whole read, check and write, so two programs on this PC cannot both see
/// room under the cap that only one has.
#[derive(Debug)]
pub struct Spend {
    path: PathBuf,
    lock: Mutex<()>,
}

impl Spend {
    pub fn at(path: PathBuf) -> Spend {
        Spend { path, lock: Mutex::new(()) }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The counts for `month` (another month's counts are none); Err for a file that is there and unreadable.
    fn read(&self, month: &str) -> Result<BTreeMap<String, f64>, String> {
        let text = match std::fs::read_to_string(&self.path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
            Err(_) => return Err("The spending count could not be read, so nothing paid was sent.".into()),
        };
        let value = serde_json::from_str::<Value>(&text)
            .map_err(|_| "The spending count's file is damaged, so nothing paid was sent.".to_string())?;
        if value.get("month").and_then(Value::as_str) != Some(month) {
            return Ok(BTreeMap::new());
        }
        Ok(value
            .get("parts")
            .and_then(Value::as_object)
            .map(|m| m.iter().filter_map(|(k, v)| Some((k.clone(), v.as_f64().filter(|n| n.is_finite() && *n >= 0.0)?))).collect())
            .unwrap_or_default())
    }

    fn write(&self, month: &str, counts: &BTreeMap<String, f64>) -> Result<(), String> {
        let text = json!({"month": month, "parts": counts}).to_string();
        crate::fsx::atomic_write(&self.path, text.as_bytes()).map_err(|_| "The spending count could not be kept.".to_string())
    }

    fn locked<T>(&self, work: impl FnOnce() -> Result<T, Refusal>) -> Result<T, Refusal> {
        let _held = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(|_| Refusal::Ledger(BUSY.into()))?;
        }
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(self.path.with_extension("lock"))
            .map_err(|_| Refusal::Ledger(BUSY.into()))?;
        let deadline = std::time::Instant::now() + LOCK_WAIT;
        loop {
            match file.try_lock() {
                Ok(()) => break,
                Err(std::fs::TryLockError::WouldBlock) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(_) => return Err(Refusal::Ledger(BUSY.into())),
            }
        }
        let result = work();
        let _ = file.unlock();
        result
    }

    /// The month's total so far (as of `now`, UTC), every part and reservation included; 0 when unreadable.
    pub fn used(&self, now: i64) -> f64 {
        self.read(&month_key(now)).map(|c| c.values().sum()).unwrap_or(0.0)
    }

    /// One part's spending this month so far.
    pub fn used_by(&self, part: &str, now: i64) -> f64 {
        self.read(&month_key(now)).ok().and_then(|c| c.get(part).copied()).unwrap_or(0.0)
    }

    /// Reserve a call's worst case, `most` dollars, for `part` before it is sent: refused when it could take the month's
    /// total past `cap`, or when the count cannot be locked, read or written. A cap that is not a finite positive
    /// number refuses every call.
    pub fn reserve(&self, part: &str, most: f64, cap: f64, now: i64) -> Result<Reservation, Refusal> {
        let month = month_key(now);
        let most = if most.is_finite() { most.max(0.0) } else { f64::INFINITY };
        if !(cap.is_finite() && cap > 0.0) || !most.is_finite() {
            return Err(Refusal::Cap { back_at: next_month_start(now) });
        }
        self.locked(|| {
            let mut counts = self.read(&month).map_err(Refusal::Ledger)?;
            let total: f64 = counts.values().sum();
            if total + most > cap {
                return Err(Refusal::Cap { back_at: next_month_start(now) });
            }
            *counts.entry(part.to_owned()).or_insert(0.0) += most;
            self.write(&month, &counts).map_err(Refusal::Ledger)?;
            Ok(Reservation { part: part.to_owned(), month: month.clone(), usd: most })
        })
    }

    /// Replace a reservation with what the call cost. A call that failed after it was sent keeps its reservation.
    pub fn settle(&self, reservation: &Reservation, cost: f64) -> Result<(), Refusal> {
        let cost = if cost.is_finite() { cost.max(0.0) } else { reservation.usd };
        self.locked(|| {
            let mut counts = self.read(&reservation.month).map_err(Refusal::Ledger)?;
            let total = counts.entry(reservation.part.clone()).or_insert(0.0);
            *total = (*total - reservation.usd + cost).max(0.0);
            self.write(&reservation.month, &counts).map_err(Refusal::Ledger)
        })
    }

    /// Whether the month's total has reached `cap` (within a cent): no paid call of any part fits any more.
    pub fn capped(&self, cap: f64, now: i64) -> bool {
        !(cap.is_finite() && cap > 0.0) || self.used(now) >= cap - 0.01
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::TempDir;

    const NOW: i64 = 1_791_633_600; // 2026-10-10 12:00 UTC

    #[test]
    fn a_price_counts_tokens_and_an_unknown_price_is_none() {
        let p = Price::known(0.2, 0.6).unwrap();
        assert!((p.cost(1_000_000, 500_000) - 0.5).abs() < 1e-12);
        assert!(Price::known(f64::NAN, 1.0).is_none());
        assert!(Price::known(-1.0, 1.0).is_none());
        assert!(Price::known(0.0, 0.0).unwrap().is_free());
    }

    #[test]
    fn months_are_utc_calendar_months() {
        assert_eq!(month_key(NOW), "2026-10");
        assert_eq!(month_key(next_month_start(NOW)), "2026-11");
        assert_eq!(month_key(next_month_start(NOW) - 1), "2026-10");
        // December rolls into the next year
        let december = time::Date::from_calendar_date(2026, time::Month::December, 15).unwrap().midnight().assume_utc().unix_timestamp();
        assert_eq!(month_key(next_month_start(december)), "2027-01");
    }

    #[test]
    fn a_call_that_could_pass_the_cap_is_refused_and_settles_to_what_it_cost() {
        let dir = TempDir::new("spend-cap");
        let spend = Spend::at(path_in(dir.path()));
        let r = spend.reserve("mind", 3.0, 5.0, NOW).unwrap();
        assert_eq!(spend.used(NOW), 3.0, "the worst case is counted before the call");
        spend.settle(&r, 0.25).unwrap();
        assert_eq!(spend.used_by("mind", NOW), 0.25);
        assert_eq!(spend.reserve("mind", 4.76, 5.0, NOW), Err(Refusal::Cap { back_at: next_month_start(NOW) }), "past the cap");
        let r = spend.reserve("eyes", 4.75, 5.0, NOW).expect("up to the cap exactly, shared by every part");
        assert!(spend.capped(5.0, NOW));
        spend.settle(&r, 4.0).unwrap();
        assert!((spend.used(NOW) - 4.25).abs() < 1e-9);
        // the next month starts again from nothing
        assert_eq!(spend.used(next_month_start(NOW)), 0.0);
        assert!(spend.reserve("mind", 1.0, 5.0, next_month_start(NOW)).is_ok());
    }

    #[test]
    fn no_cap_or_an_unknown_cost_refuses() {
        let dir = TempDir::new("spend-nocap");
        let spend = Spend::at(path_in(dir.path()));
        for cap in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(spend.reserve("mind", 0.01, cap, NOW), Err(Refusal::Cap { .. })), "{cap}");
        }
        assert!(matches!(spend.reserve("mind", f64::NAN, 5.0, NOW), Err(Refusal::Cap { .. })), "an unknown worst case is never sent");
        assert_eq!(spend.used(NOW), 0.0, "nothing counted, nothing written");
    }

    #[test]
    fn a_count_that_cannot_be_read_or_locked_refuses() {
        let dir = TempDir::new("spend-closed");
        let path = path_in(dir.path());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not json").unwrap();
        assert!(matches!(Spend::at(path.clone()).reserve("mind", 0.01, 5.0, NOW), Err(Refusal::Ledger(_))));
        std::fs::write(&path, b"{}").unwrap();
        let holder = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path.with_extension("lock")).unwrap();
        holder.lock().unwrap();
        let refused = Spend::at(path.clone()).reserve("mind", 0.01, 5.0, NOW);
        assert!(matches!(&refused, Err(Refusal::Ledger(w)) if w.contains("nothing paid was sent")), "{refused:?}");
        holder.unlock().unwrap();
        assert!(Spend::at(path).reserve("mind", 0.01, 5.0, NOW).is_ok());
    }

    /// Eight ledgers on one file (a handle and a lock each, as two processes would have) reserve at once: the month's
    /// total never passes the cap, and exactly as many reservations succeed as fit.
    #[test]
    fn concurrent_reservations_never_pass_the_cap() {
        let dir = TempDir::new("spend-race");
        let path = path_in(dir.path());
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let (path, barrier) = (path.clone(), barrier.clone());
                std::thread::spawn(move || {
                    let spend = Spend::at(path);
                    barrier.wait();
                    (0..5).filter(|_| spend.reserve("mind", 0.5, 5.0, NOW).is_ok()).count()
                })
            })
            .collect();
        let granted: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
        assert_eq!(granted, 10, "$5 holds ten reservations of 50 cents");
        assert!(Spend::at(path).used(NOW) <= 5.0);
    }
}
