//! Cloudflare Workers AI's free allocation, metered on this PC so a part stops before it is spent.
//!
//! Cloudflare gives every account 10,000 Neurons a day, starting again at 00:00 UTC; past that, an account on
//! Workers Free is refused and one on Workers Paid is billed. Alelyon cannot see which plan an account is on, so it
//! counts each call's Neurons itself, from Cloudflare's published per-model rates ([`rate`], read 2026-10-10 from
//! <https://developers.cloudflare.com/workers-ai/platform/pricing/>), and refuses a call that could take the day's
//! count past [`STOP_AT`] (9,000 Neurons: a margin of 1,000 below the allocation, for what the meter estimates rather
//! than knows). The count is per account, shared by every part on it (eyes, embeddings, voice), and kept in a file
//! ([`Meter`]) so a restart does not reset it. A model whose rate the meter does not know is never called.
//!
//! **What is estimated.** Before a call the meter reserves its most: input tokens from the text's length (one token
//! per three characters, rounded up, plus eight), the output's limit, an image as [`IMAGE_TOKENS`] input tokens, and
//! speech as two words a second. After the call it records the provider's own token counts where the answer has them
//! (`usage`), else the same estimate. Cloudflare's page gives no rate for an image sent to a vision model; the meter
//! counts one as 6,404 input tokens (four of Llama 3.2 Vision's 1,601-token tiles), an assumption, not a measurement.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::{Value, json};

/// The free allocation a day, in Neurons.
pub const FREE_PER_DAY: f64 = 10_000.0;
/// Where the meter stops: no call may take the day's count past this.
pub const STOP_AT: f64 = 9_000.0;
/// The input tokens an image is counted as (see the module's note).
pub const IMAGE_TOKENS: u64 = 6_404;

/// A model's Neurons.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rate {
    /// Per million input tokens, and per million output tokens.
    Tokens { input: f64, output: f64 },
    /// Per minute of audio.
    AudioMinute(f64),
}

/// Cloudflare's published rate for `model`, or None for a model the meter does not know.
pub fn rate(model: &str) -> Option<Rate> {
    let tokens = |input: f64, output: f64| Some(Rate::Tokens { input, output });
    match model.trim() {
        "@cf/meta/llama-3.2-11b-vision-instruct" => tokens(4_410.0, 61_493.0),
        "@cf/baai/bge-m3" => tokens(1_075.0, 0.0),
        "@cf/qwen/qwen3-embedding-0.6b" => tokens(1_075.0, 0.0),
        "@cf/baai/bge-small-en-v1.5" => tokens(1_841.0, 0.0),
        "@cf/baai/bge-base-en-v1.5" => tokens(6_058.0, 0.0),
        "@cf/baai/bge-large-en-v1.5" => tokens(18_582.0, 0.0),
        "@cf/pfnet/plamo-embedding-1b" => tokens(1_689.0, 0.0),
        "@cf/myshell-ai/melotts" => Some(Rate::AudioMinute(18.63)),
        "@cf/openai/whisper-large-v3-turbo" => Some(Rate::AudioMinute(46.63)),
        _ => None,
    }
}

/// What one call uses.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Use {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub images: u64,
    pub audio_seconds: f64,
}

/// Input tokens counted for `chars` characters of text: one per three, rounded up, plus eight.
pub fn tokens_for(chars: usize) -> u64 {
    (chars as u64).div_ceil(3) + 8
}

/// Seconds of speech counted for `text`: two words a second, at least one.
pub fn speech_seconds(text: &str) -> f64 {
    (text.split_whitespace().count() as f64 / 2.0).max(1.0)
}

/// The Neurons `used` costs on `model`, or None for a model the meter does not know.
pub fn neurons(model: &str, used: Use) -> Option<f64> {
    Some(match rate(model)? {
        Rate::Tokens { input, output } => {
            let tokens_in = used.input_tokens + used.images * IMAGE_TOKENS;
            (tokens_in as f64 * input + used.output_tokens as f64 * output) / 1_000_000.0
        }
        Rate::AudioMinute(per) => used.audio_seconds / 60.0 * per,
    })
}

/// The day a time falls on, in UTC (days since 1970).
pub fn utc_day(at: i64) -> i64 {
    at.div_euclid(86_400)
}

/// The next 00:00 UTC after `at`.
pub fn next_midnight(at: i64) -> i64 {
    (utc_day(at) + 1) * 86_400
}

/// The day's count per account, kept in a file. Every change to it (a reservation, its settling) is made under
/// an operating-system lock on a file beside it (`cloudflare-neurons.lock`, `std::fs::File::lock`, which is
/// `LockFileEx` on Windows), held for the whole read, check and write, so two programs on this PC
/// counting on one account cannot both see room that only one has. The lock is waited for at most [`LOCK_WAIT`];
/// past that, or with a count file that cannot be read, the call is refused: nothing goes unmetered.
#[derive(Debug)]
pub struct Meter {
    path: PathBuf,
    lock: Mutex<()>,
}

/// The longest a call waits for the meter's lock before it is refused.
pub const LOCK_WAIT: std::time::Duration = std::time::Duration::from_secs(3);

/// Why a call may not go.
#[derive(Clone, Debug, PartialEq)]
pub enum Refusal {
    /// It could take the day's count past [`STOP_AT`]: the allocation starts again then (the next 00:00 UTC).
    Spent { back_at: i64 },
    /// The meter could not be read, locked or written: in one sentence.
    Meter(String),
}

/// A call's worst case, counted before it is sent; [`Meter::settle`] replaces it with what the call used.
#[derive(Clone, Debug, PartialEq)]
pub struct Reservation {
    account: String,
    day: i64,
    neurons: f64,
}

impl Reservation {
    pub fn neurons(&self) -> f64 {
        self.neurons
    }
}

/// The meter's file under a home folder.
pub fn path_in(home: &Path) -> PathBuf {
    home.join(".alelyon").join("cloud").join("cloudflare-neurons.json")
}

/// The meter's file for this user, or None without a home folder.
pub fn default_path() -> Option<PathBuf> {
    ["USERPROFILE", "HOME"]
        .iter()
        .find_map(|k| std::env::var_os(k).filter(|v| !v.is_empty()))
        .map(|h| path_in(Path::new(&h)))
}

const BUSY: &str = "The free-allocation meter is busy or cannot be opened, so nothing was sent to Cloudflare.";

impl Meter {
    pub fn at(path: PathBuf) -> Meter {
        Meter { path, lock: Mutex::new(()) }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The counts as kept, for `day` (another day's counts are none); Err for a file that is there and unreadable.
    fn read(&self, day: i64) -> Result<BTreeMap<String, f64>, String> {
        let text = match std::fs::read_to_string(&self.path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
            Err(_) => return Err("The free-allocation meter could not be read, so nothing was sent to Cloudflare.".into()),
        };
        let value = serde_json::from_str::<Value>(&text)
            .map_err(|_| "The free-allocation meter's file is damaged, so nothing was sent to Cloudflare.".to_string())?;
        if value.get("day").and_then(Value::as_i64) != Some(day) {
            return Ok(BTreeMap::new());
        }
        Ok(value
            .get("accounts")
            .and_then(Value::as_object)
            .map(|m| m.iter().filter_map(|(k, v)| Some((k.clone(), v.as_f64().filter(|n| n.is_finite() && *n >= 0.0)?))).collect())
            .unwrap_or_default())
    }

    fn write(&self, day: i64, counts: &BTreeMap<String, f64>) -> Result<(), String> {
        let text = json!({"day": day, "accounts": counts}).to_string();
        // whole, then into place, as lattice-core writes every file it keeps
        crate::fsx::atomic_write(&self.path, text.as_bytes()).map_err(|_| "The free-allocation meter could not be kept.".to_string())
    }

    /// Run `work` holding this process's lock and the operating system's lock on the file beside the meter, waiting
    /// at most [`LOCK_WAIT`] for the latter.
    fn locked<T>(&self, work: impl FnOnce() -> Result<T, Refusal>) -> Result<T, Refusal> {
        let _held = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(|_| Refusal::Meter(BUSY.into()))?;
        }
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(self.path.with_extension("lock"))
            .map_err(|_| Refusal::Meter(BUSY.into()))?;
        let deadline = std::time::Instant::now() + LOCK_WAIT;
        loop {
            match file.try_lock() {
                Ok(()) => break,
                Err(std::fs::TryLockError::WouldBlock) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(_) => return Err(Refusal::Meter(BUSY.into())),
            }
        }
        let result = work();
        let _ = file.unlock();
        result
    }

    /// The Neurons counted today (UTC, as of `now`) for `account`, reservations included (0 when unreadable).
    pub fn used(&self, account: &str, now: i64) -> f64 {
        self.read(utc_day(now)).ok().and_then(|c| c.get(account).copied()).unwrap_or(0.0)
    }

    /// Reserve a call's worst case, `most` Neurons, before it is sent: refused when it could take the day's count
    /// past [`STOP_AT`], or when the meter cannot be locked, read or written.
    pub fn reserve(&self, account: &str, most: f64, now: i64) -> Result<Reservation, Refusal> {
        let day = utc_day(now);
        self.locked(|| {
            let mut counts = self.read(day).map_err(Refusal::Meter)?;
            let total = counts.entry(account.to_owned()).or_insert(0.0);
            if *total + most.max(0.0) > STOP_AT {
                return Err(Refusal::Spent { back_at: next_midnight(now) });
            }
            *total += most.max(0.0);
            self.write(day, &counts).map_err(Refusal::Meter)?;
            Ok(Reservation { account: account.to_owned(), day, neurons: most.max(0.0) })
        })
    }

    /// Replace a reservation with what the call used. A call that failed after it was sent keeps its reservation
    /// (it is never settled), so the count errs toward the allocation, never past it.
    pub fn settle(&self, reservation: &Reservation, used: f64) -> Result<(), Refusal> {
        self.locked(|| {
            let mut counts = self.read(reservation.day).map_err(Refusal::Meter)?;
            let total = counts.entry(reservation.account.clone()).or_insert(0.0);
            *total = (*total - reservation.neurons + used.max(0.0)).max(0.0);
            self.write(reservation.day, &counts).map_err(Refusal::Meter)
        })
    }

    /// Count `neurons` for `account` today (a call counted without a reservation, or a test's).
    pub fn add(&self, account: &str, neurons: f64, now: i64) -> Result<f64, Refusal> {
        let day = utc_day(now);
        self.locked(|| {
            let mut counts = self.read(day).map_err(Refusal::Meter)?;
            let total = counts.entry(account.to_owned()).or_insert(0.0);
            *total += neurons.max(0.0);
            let total = *total;
            self.write(day, &counts).map_err(Refusal::Meter)?;
            Ok(total)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::TempDir;

    const NOW: i64 = 1_791_633_600; // 2026-10-10 12:00 UTC

    #[test]
    fn each_models_neurons_follow_cloudflares_rates() {
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        // a million input tokens of bge-m3 is 1,075 Neurons
        assert!(close(neurons("@cf/baai/bge-m3", Use { input_tokens: 1_000_000, ..Use::default() }).unwrap(), 1_075.0));
        assert!(close(neurons("@cf/qwen/qwen3-embedding-0.6b", Use { input_tokens: 2_000, ..Use::default() }).unwrap(), 2.15));
        // vision: input and output priced apart; an image counts as its tokens
        let v = neurons("@cf/meta/llama-3.2-11b-vision-instruct", Use { input_tokens: 1_000, output_tokens: 1_000, images: 1, ..Use::default() }).unwrap();
        assert!(close(v, (7_404.0 * 4_410.0 + 1_000.0 * 61_493.0) / 1e6), "{v}");
        // speech: per audio minute
        assert!(close(neurons("@cf/myshell-ai/melotts", Use { audio_seconds: 60.0, ..Use::default() }).unwrap(), 18.63));
        assert!(close(neurons("@cf/openai/whisper-large-v3-turbo", Use { audio_seconds: 30.0, ..Use::default() }).unwrap(), 23.315));
        assert_eq!(neurons("@cf/meta/llama-3.1-8b-instruct", Use::default()), None, "a model the meter does not know");
        assert_eq!(tokens_for(0), 8);
        assert_eq!(tokens_for(10), 12);
        assert_eq!(speech_seconds("one two three"), 1.5);
        assert_eq!(speech_seconds(""), 1.0);
    }

    #[test]
    fn the_count_is_shared_by_the_account_and_stops_before_the_allocation() {
        let dir = TempDir::new("meter-shared");
        let meter = Meter::at(path_in(dir.path()));
        const { assert!(STOP_AT < FREE_PER_DAY) };
        meter.add("acct", 5_000.0, NOW).unwrap(); // the voice
        meter.add("acct", 3_900.0, NOW).unwrap(); // the embeddings
        assert_eq!(meter.used("acct", NOW), 8_900.0, "every part on the account counts toward one allocation");
        assert_eq!(meter.used("other", NOW), 0.0, "another account has its own");
        assert_eq!(meter.reserve("acct", 100.5, NOW), Err(Refusal::Spent { back_at: next_midnight(NOW) }), "past the stop: wait for 00:00 UTC");
        let r = meter.reserve("acct", 100.0, NOW).expect("up to the stop exactly");
        assert_eq!(meter.used("acct", NOW), 9_000.0, "the worst case is counted before the call");
        meter.settle(&r, 40.0).unwrap();
        assert_eq!(meter.used("acct", NOW), 8_940.0, "settled to what the call used");
        assert_eq!(next_midnight(NOW), NOW + 12 * 3600);
    }

    #[test]
    fn the_count_starts_again_at_utc_midnight_and_survives_a_restart() {
        let dir = TempDir::new("meter-day");
        let path = path_in(dir.path());
        Meter::at(path.clone()).add("acct", 8_999.0, NOW).unwrap();
        // a restart: a new meter on the same file
        let again = Meter::at(path.clone());
        assert_eq!(again.used("acct", NOW + 60), 8_999.0, "the day's count is kept on disk");
        assert!(again.reserve("acct", 2.0, NOW + 60).is_err());
        // the next day, from 00:00 UTC
        let midnight = next_midnight(NOW);
        assert_eq!(again.used("acct", midnight - 1), 8_999.0);
        assert_eq!(again.used("acct", midnight), 0.0);
        let r = again.reserve("acct", 2.0, midnight).unwrap();
        again.settle(&r, 1.0).unwrap();
        assert_eq!(Meter::at(path).used("acct", midnight), 1.0, "yesterday's count is gone");
    }

    /// Eight threads, each with a meter (and so a file handle and an operating-system lock) of its own on one file,
    /// as two processes would have, reserve at once: the day's count never passes the stop, and exactly as many
    /// reservations succeed as fit. Threads with separate handles stand in for processes: the lock is per handle.
    #[test]
    fn concurrent_reservations_on_separate_handles_never_pass_the_stop() {
        let dir = TempDir::new("meter-race");
        let path = path_in(dir.path());
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let (path, barrier) = (path.clone(), barrier.clone());
                std::thread::spawn(move || {
                    let meter = Meter::at(path);
                    barrier.wait();
                    (0..5).filter(|_| meter.reserve("acct", 400.0, NOW).is_ok()).count()
                })
            })
            .collect();
        let granted: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
        let used = Meter::at(path).used("acct", NOW);
        assert_eq!(granted, 22, "9,000 holds 22 reservations of 400");
        assert_eq!(used, 22.0 * 400.0);
        assert!(used <= STOP_AT);
    }

    #[test]
    fn a_meter_that_cannot_be_read_or_locked_refuses() {
        let dir = TempDir::new("meter-closed");
        let path = path_in(dir.path());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not json").unwrap();
        assert!(matches!(Meter::at(path.clone()).reserve("acct", 1.0, NOW), Err(Refusal::Meter(_))), "a damaged file is no room");
        std::fs::write(&path, b"{}").unwrap();
        // another holder keeps the lock past the wait: refused, never sent unmetered
        let holder = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path.with_extension("lock")).unwrap();
        holder.lock().unwrap();
        let began = std::time::Instant::now();
        let refused = Meter::at(path.clone()).reserve("acct", 1.0, NOW);
        assert!(matches!(&refused, Err(Refusal::Meter(w)) if w.contains("nothing was sent")), "{refused:?}");
        assert!(began.elapsed() >= LOCK_WAIT && began.elapsed() < LOCK_WAIT * 3, "a bounded wait");
        holder.unlock().unwrap();
        assert!(Meter::at(path).reserve("acct", 1.0, NOW).is_ok());
    }
}
