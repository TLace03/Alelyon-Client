//! Where a window keeps what a person set: `~/.alelyon/angel`, beside Sinai's own state.
//!
//! Measured on 2026-10-01: the dock's layout lived in `%APPDATA%\Alelyon`. A file that a process
//! inside the Claude desktop app creates there lands in the app's private store, where it hides the
//! real one from every process inside the app, so a window a Claude session started and the user's
//! own launches kept two layouts (465 B in the store, 613 B real). The appearance went to the same
//! folder. The home directory is not redirected, and Sinai's own state already lives in
//! `~/.alelyon/angel`. This is the same resolver for a window's two files, by the same rules:
//!
//! - MOVING IN, once: only a start no AI coding agent made (`CLAUDECODE` and `AI_AGENT` unset)
//!   moves anything. Inside the app, an agent's process sees the app's copy of the old files, not
//!   the user's.
//! - A copy, never a move: each file is checked by SHA-256 and listed in `MIGRATED-window.json`,
//!   which is written last. The old files are left exactly as they were.
//! - Until that marker exists, an agent's run keeps the files in `<temp>/alelyon-agent-state/Angel`,
//!   so nothing but the move ever writes them into the new home first.
//!
//! `ANGEL_DOCK` and `ANGEL_APPEARANCE` still name a file, which is used as it is; a start that names
//! both moves nothing. A run decides where its files are once, at its first read and after the
//! move, so a window that is open when another start moves the files in keeps the folder it began
//! with.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::SystemTime;

#[cfg(test)]
#[path = "state_home_tests.rs"]
mod tests;

/// The window's own marker, beside the loop's `MIGRATED.json` rather than shared with it: the loop
/// and the window move different folders, each at its own first start, and one marker would let
/// whichever moved first stop the other.
pub const MARKER: &str = "MIGRATED-window.json";
/// The home, under the user's home directory, as `state_home.home_dir()` builds it.
pub const HOME: [&str; 2] = [".alelyon", "angel"];
/// An agent's folder before the move, under the temporary directory, as `state_home.agent_dir()`
/// builds it.
pub const AGENT_STATE: [&str; 2] = ["alelyon-agent-state", "Angel"];
/// The values of `CLAUDECODE` that mean an agent: `state_home._TRUTHY`.
const TRUTHY: [&str; 4] = ["1", "true", "yes", "on"];
/// A copy in progress. Not the loop's `.moving`: the loop's move clears every `.moving` file in
/// this folder, and the loop and the window can make their first starts at the same moment.
const PARTIAL: &str = ".copying";

/// A file the window keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum File {
    /// The dock's arrangement (`dock.rs`).
    Dock,
    /// How a person has shaped their Sinai (`appearance.rs`).
    Appearance,
}

impl File {
    pub const ALL: [File; 2] = [File::Dock, File::Appearance];

    /// Its name in the home, the same as it was in the old folder.
    pub fn name(self) -> &'static str {
        match self {
            File::Dock => "angel-dock.json",
            File::Appearance => "sinai-appearance.json",
        }
    }

    /// The variable that names it explicitly.
    pub fn variable(self) -> &'static str {
        match self {
            File::Dock => "ANGEL_DOCK",
            File::Appearance => "ANGEL_APPEARANCE",
        }
    }
}

/// A process's environment, one variable at a time: the real one or a test's. Read as
/// `std::env::var` reads it, which is how the window has always read these variables.
pub type Env<'a> = &'a dyn Fn(&str) -> Option<String>;

/// What checks a copy: SHA-256 as lowercase hex, unless a test makes a copy fail it.
type Digest<'a> = &'a dyn Fn(&Path) -> Result<String, String>;

fn process_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// True when an AI coding agent started this process. Claude Code sets `CLAUDECODE=1` and
/// `AI_AGENT` in every process it starts (measured 2026-10-01, `state_home.agent_session`).
pub fn agent_session(env: Env) -> bool {
    let claude = env("CLAUDECODE").unwrap_or_default();
    if TRUTHY.contains(&claude.trim().to_lowercase().as_str()) {
        return true;
    }
    !env("AI_AGENT").unwrap_or_default().trim().is_empty()
}

/// The window's state home, `~/.alelyon/angel`, under the user's home directory `home`.
pub fn home_dir(home: &Path) -> PathBuf {
    HOME.iter().fold(home.to_path_buf(), |path, part| path.join(part))
}

/// Where the window kept its files before W3, resolved exactly as it was.
pub fn legacy_dir(env: Env) -> Option<PathBuf> {
    let base = env("APPDATA")
        .or_else(|| env("XDG_CONFIG_HOME"))
        .or_else(|| env("HOME").map(|h| format!("{h}/.config")))?;
    Some(PathBuf::from(base).join("Alelyon"))
}

/// An agent's folder before the user's first start has moved the old files in.
pub fn agent_dir(temp: &Path) -> PathBuf {
    AGENT_STATE
        .iter()
        .fold(temp.to_path_buf(), |path, part| path.join(part))
}

/// The file `file`'s variable names, when it names one.
fn explicit(file: File, env: Env) -> Option<PathBuf> {
    let value = env(file.variable())?;
    (!value.trim().is_empty()).then(|| PathBuf::from(value))
}

/// The folder the window's files live in for this process, or none without a home directory.
/// Pure: it creates and moves nothing.
pub fn state_dir(env: Env, home: Option<&Path>, temp: &Path) -> Option<PathBuf> {
    let target = home_dir(home?);
    if agent_session(env) && !target.join(MARKER).exists() {
        return Some(agent_dir(temp));
    }
    Some(target)
}

/// What a start's move did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    /// Both variables name their files, so nothing moved.
    Explicit,
    /// There is no home directory: nothing moved, and only a file a variable names is kept.
    NoHome,
    /// The marker exists: an earlier start moved the files in.
    Already,
    /// An AI coding agent started this process.
    Deferred,
    /// There were no old files. The home is marked all the same.
    Nothing,
    /// The old files were copied in and checked.
    Moved,
    /// A copy failed or did not verify. No marker was written, so the next start tries again.
    Failed(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Report {
    pub status: Status,
    /// The old folder, when the move looked in it.
    pub from: Option<PathBuf>,
    /// The home, when there is one.
    pub to: Option<PathBuf>,
    /// How many files were copied.
    pub files: usize,
}

impl Report {
    /// One line for the window's log.
    pub fn describe(&self) -> String {
        let shown = |path: &Option<PathBuf>| {
            path.as_ref()
                .map_or_else(|| "(none)".to_string(), |p| p.display().to_string())
        };
        match &self.status {
            Status::Moved => format!(
                "moved {} file(s) from {} to {} (copy; the old files are kept)",
                self.files,
                shown(&self.from),
                shown(&self.to)
            ),
            Status::Nothing => format!("new state home {} (nothing to move)", shown(&self.to)),
            Status::Deferred => "not moved: an AI coding agent started this process; the next \
                                 start that no agent makes moves the old files in"
                .into(),
            Status::Explicit => "ANGEL_DOCK and ANGEL_APPEARANCE name the files; nothing moved".into(),
            Status::NoHome => "no home directory: only a file a variable names is saved".into(),
            Status::Already => format!("state home {}", shown(&self.to)),
            Status::Failed(why) => format!(
                "not moved ({why}); this run keeps the old files in {}, and the next start tries again",
                shown(&self.from)
            ),
        }
    }
}

/// Copy the old files into the home once, if this process may. Returns what it did.
///
/// `home` is the user's home directory; with none, nothing moves. A copy that does not verify
/// fails before the marker is written, so the next start tries again.
pub fn move_in(env: Env, home: Option<&Path>, now: SystemTime) -> Report {
    move_in_checked_by(env, home, now, &sha256)
}

fn move_in_checked_by(env: Env, home: Option<&Path>, now: SystemTime, digest: Digest) -> Report {
    let mut report = Report {
        status: Status::Explicit,
        from: None,
        to: None,
        files: 0,
    };
    if File::ALL.iter().all(|&file| explicit(file, env).is_some()) {
        return report;
    }
    let Some(home) = home else {
        report.status = Status::NoHome;
        return report;
    };
    let target = home_dir(home);
    report.to = Some(target.clone());
    if target.join(MARKER).exists() {
        report.status = Status::Already;
        return report;
    }
    if agent_session(env) {
        report.status = Status::Deferred;
        return report;
    }
    report.from = legacy_dir(env);
    report.status = match copy_in(report.from.as_deref(), &target, now, digest) {
        Ok(0) => Status::Nothing,
        Ok(files) => {
            report.files = files;
            Status::Moved
        }
        Err(why) => Status::Failed(why),
    };
    report
}

/// Copy each old file there is into `target`, check it, and write the marker last.
fn copy_in(
    source: Option<&Path>,
    target: &Path,
    now: SystemTime,
    digest: Digest,
) -> Result<usize, String> {
    let at = |path: &Path, e: std::io::Error| format!("{}: {e}", path.display());
    std::fs::create_dir_all(target).map_err(|e| at(target, e))?;
    let mut copied = Vec::new();
    for file in File::ALL {
        let dest = target.join(file.name());
        let partial = target.join(format!("{}{PARTIAL}", file.name()));
        // Left by a start that died mid-copy. Only this file's own partial is ever removed.
        match std::fs::remove_file(&partial) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(at(&partial, e)),
            _ => {}
        }
        let Some(path) = source
            .map(|dir| dir.join(file.name()))
            .filter(|p| p.is_file())
        else {
            continue;
        };
        {
            let mut from = std::fs::File::open(&path).map_err(|e| at(&path, e))?;
            let mut out = std::fs::File::create(&partial).map_err(|e| at(&partial, e))?;
            std::io::copy(&mut from, &mut out)
                .and_then(|_| out.sync_all())
                .map_err(|e| at(&partial, e))?;
        }
        std::fs::rename(&partial, &dest).map_err(|e| at(&dest, e))?;
        let sha = digest(&path)?;
        if digest(&dest)? != sha {
            return Err(format!("the copy of {} did not verify", file.name()));
        }
        let bytes = std::fs::metadata(&path).map_err(|e| at(&path, e))?.len();
        copied.push(serde_json::json!({"file": file.name(), "bytes": bytes, "sha256": sha}));
    }
    let record = serde_json::json!({
        "moved_at_utc": utc_iso(now),
        "from": source.map(|dir| dir.display().to_string()),
        "to": target.display().to_string(),
        "files": copied,
        "note": "copied, not moved: the old files are left as they were",
    });
    let text = serde_json::to_string_pretty(&record).map_err(|e| e.to_string())?;
    let marker = target.join(MARKER);
    let temporary = target.join(format!("{MARKER}.tmp"));
    std::fs::write(&temporary, text).map_err(|e| at(&temporary, e))?;
    std::fs::rename(&temporary, &marker).map_err(|e| at(&marker, e))?;
    Ok(copied.len())
}

/// A file's SHA-256 as lowercase hex, which is what `state_home._sha` records.
fn sha256(path: &Path) -> Result<String, String> {
    use sha2::Digest as _;
    let at = |e: std::io::Error| format!("{}: {e}", path.display());
    let mut file = std::fs::File::open(path).map_err(at)?;
    let mut hasher = sha2::Sha256::new();
    std::io::copy(&mut file, &mut hasher).map_err(at)?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// `now` in UTC, ISO 8601 to the microsecond, without a date-time crate.
fn utc_iso(now: SystemTime) -> String {
    let since = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let seconds = since.as_secs() as i64;
    let (days, clock) = (seconds.div_euclid(86_400), seconds.rem_euclid(86_400));
    // Days since 1970-01-01 to a civil date: H. Hinnant, "chrono-Compatible Low-Level Date
    // Algorithms", days_from_civil's inverse.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:06}+00:00",
        clock / 3_600,
        clock % 3_600 / 60,
        clock % 60,
        since.subsec_micros()
    )
}

/// Where `file` is for a run whose start's move reported `report`. Pure.
pub fn resolve(
    file: File,
    env: Env,
    report: &Report,
    home: Option<&Path>,
    temp: &Path,
) -> Option<PathBuf> {
    if let Some(path) = explicit(file, env) {
        return Some(path);
    }
    if let Status::Failed(_) = report.status {
        // The move did not finish. This run stays on the old file, which the next start copies
        // in, rather than writing into a home that copy would then overwrite.
        return legacy_dir(env).map(|dir| dir.join(file.name()));
    }
    Some(state_dir(env, home, temp)?.join(file.name()))
}

/// Where a run keeps its files, and what its start's move did.
pub struct Run {
    pub report: Report,
    dock: Option<PathBuf>,
    appearance: Option<PathBuf>,
    /// The files this run kept on their old copies because the move failed.
    stayed: Vec<File>,
}

impl Run {
    /// Move the old files in, if this start may, then decide where each file is.
    pub fn start(env: Env, home: Option<&Path>, temp: &Path, now: SystemTime) -> Run {
        let report = move_in(env, home, now);
        let stayed = match report.status {
            Status::Failed(_) => File::ALL
                .into_iter()
                .filter(|&file| explicit(file, env).is_none())
                .collect(),
            _ => Vec::new(),
        };
        Run {
            dock: resolve(File::Dock, env, &report, home, temp),
            appearance: resolve(File::Appearance, env, &report, home, temp),
            report,
            stayed,
        }
    }

    pub fn path(&self, file: File) -> Option<&Path> {
        match file {
            File::Dock => self.dock.as_deref(),
            File::Appearance => self.appearance.as_deref(),
        }
    }

    /// What a person is told about `file` when this start's move failed.
    pub fn notice(&self, file: File) -> Option<String> {
        let Status::Failed(why) = &self.report.status else {
            return None;
        };
        self.stayed.contains(&file).then(|| {
            format!(
                "Not moved to ~/.alelyon/angel ({why}); this run keeps the old file, \
                 and the next start tries again"
            )
        })
    }
}

/// This run's files. The first call moves the old files in, if this start may, and logs where the
/// run keeps them; every later call returns the same answer.
pub fn run() -> &'static Run {
    static RUN: OnceLock<Run> = OnceLock::new();
    RUN.get_or_init(|| {
        let home = std::env::home_dir();
        let run = Run::start(
            &process_env,
            home.as_deref(),
            &std::env::temp_dir(),
            SystemTime::now(),
        );
        println!("[angel] state: {}", run.report.describe());
        for file in File::ALL {
            match run.path(file) {
                Some(path) => println!("[angel] {} at {}", file.name(), path.display()),
                None => println!("[angel] {}: no folder, so it is not saved", file.name()),
            }
        }
        run
    })
}

/// Where this run keeps `file`.
pub fn saved_at(file: File) -> Option<PathBuf> {
    run().path(file).map(Path::to_path_buf)
}

/// What a person is told about `file`, when this run's move failed.
pub fn notice(file: File) -> Option<String> {
    run().notice(file)
}
