//! W3: the window's layout and appearance live in ~/.alelyon/angel, and the old
//! %APPDATA% files are moved in once (`state_home.rs`), by the same rules as
//! Sinai's own state, and these tests follow that state's own tests.
//!
//! The most expensive failure is losing what a person already set, so the
//! properties pinned first are: the old files are copied whole and exactly, they
//! are never changed, and no AI agent's process is ever the one that moves them.
//! Every test passes its own home, temporary directory and environment; none
//! reads or writes the real ones.

use super::*;
use std::collections::BTreeMap;

const AGENT: &[(&str, &str)] = &[("CLAUDECODE", "1")];

/// A directory of this test's own, removed when the test ends, failed or not.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Scratch {
        let nanos = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "angel-state-home-{name}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir(&dir).unwrap();
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // Only the directory this test created, by the name it was given.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
    let vars: BTreeMap<String, String> = pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    move |name| vars.get(name).cloned()
}

/// `APPDATA` pointing at `root`, plus `more`.
fn appdata(root: &Path, more: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
    let mut pairs = vec![("APPDATA", root.to_str().unwrap())];
    pairs.extend_from_slice(more);
    env(&pairs)
}

/// An old %APPDATA% folder holding both files, and their bytes.
fn old_files(root: &Path) -> BTreeMap<String, Vec<u8>> {
    let files: BTreeMap<String, Vec<u8>> = [
        (
            "angel-dock.json",
            &br#"{"schema": "alelyon.angel.dock", "version": 2}"#[..],
        ),
        (
            "sinai-appearance.json",
            &br#"{"schema": "sinai-appearance", "version": 1}"#[..],
        ),
    ]
    .into_iter()
    .map(|(name, bytes)| (name.to_string(), bytes.to_vec()))
    .collect();
    let folder = root.join("Alelyon");
    std::fs::create_dir_all(&folder).unwrap();
    for (name, bytes) in &files {
        std::fs::write(folder.join(name), bytes).unwrap();
    }
    files
}

fn tree(folder: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut found = BTreeMap::new();
    for entry in std::fs::read_dir(folder).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            found.insert(
                path.file_name().unwrap().to_string_lossy().into_owned(),
                std::fs::read(&path).unwrap(),
            );
        }
    }
    found
}

fn marker(target: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(target.join(MARKER)).unwrap()).unwrap()
}

// ---- where the files are ----------------------------------------------------

#[test]
fn an_explicit_file_is_used_as_it_is() {
    let s = Scratch::new("explicit");
    let (home, temp) = (s.0.join("home"), s.0.join("temp"));
    let dock = s.0.join("x").join("layout.json");
    let appearance = s.0.join("y").join("look.json");
    let named = env(&[
        ("ANGEL_DOCK", dock.to_str().unwrap()),
        ("ANGEL_APPEARANCE", appearance.to_str().unwrap()),
        ("CLAUDECODE", "1"),
    ]);
    let report = move_in(&named, Some(&home), SystemTime::now());
    assert_eq!(report.status, Status::Explicit);
    assert_eq!(
        resolve(File::Dock, &named, &report, Some(&home), &temp),
        Some(dock)
    );
    assert_eq!(
        resolve(File::Appearance, &named, &report, Some(&home), &temp),
        Some(appearance)
    );
    assert!(!home.exists(), "a start that names both files touched the home");
}

#[test]
fn a_blank_variable_names_no_file() {
    let s = Scratch::new("blank");
    let (home, temp) = (s.0.join("home"), s.0.join("temp"));
    let blank = env(&[("ANGEL_DOCK", "  ")]);
    let report = Report {
        status: Status::Already,
        from: None,
        to: None,
        files: 0,
    };
    assert_eq!(
        resolve(File::Dock, &blank, &report, Some(&home), &temp),
        Some(home.join(".alelyon").join("angel").join("angel-dock.json"))
    );
}

#[test]
fn a_persons_own_start_uses_the_home() {
    let s = Scratch::new("own");
    let (home, temp) = (s.0.join("home"), s.0.join("temp"));
    let target = home.join(".alelyon").join("angel");
    assert_eq!(home_dir(&home), target);
    assert_eq!(state_dir(&env(&[]), Some(&home), &temp), Some(target));
}

#[test]
fn an_agent_uses_temp_until_the_move_then_the_home() {
    let s = Scratch::new("agent-temp");
    let (home, temp) = (s.0.join("home"), s.0.join("temp"));
    assert_eq!(
        state_dir(&env(AGENT), Some(&home), &temp),
        Some(temp.join("alelyon-agent-state").join("Angel"))
    );
    // The user's first start, with no old files to move.
    move_in(&appdata(&s.0.join("none"), &[]), Some(&home), SystemTime::now());
    assert_eq!(
        state_dir(&env(AGENT), Some(&home), &temp),
        Some(home.join(".alelyon").join("angel"))
    );
}

#[test]
fn either_marker_is_an_agent() {
    assert!(agent_session(&env(&[("CLAUDECODE", "1")])));
    assert!(agent_session(&env(&[("CLAUDECODE", " True ")])));
    assert!(agent_session(&env(&[(
        "AI_AGENT",
        "claude-code_2-1-284_agent"
    )])));
}

#[test]
fn a_false_marker_is_no_agent() {
    for value in ["", "0", "false"] {
        assert!(!agent_session(&env(&[("CLAUDECODE", value)])), "{value:?}");
    }
    assert!(!agent_session(&env(&[("AI_AGENT", " ")])));
    assert!(!agent_session(&env(&[])));
}

#[test]
fn the_old_folder_is_resolved_as_it_always_was() {
    assert_eq!(
        legacy_dir(&env(&[("APPDATA", "R"), ("XDG_CONFIG_HOME", "X")])),
        Some(PathBuf::from("R").join("Alelyon"))
    );
    assert_eq!(
        legacy_dir(&env(&[("XDG_CONFIG_HOME", "X"), ("HOME", "H")])),
        Some(PathBuf::from("X").join("Alelyon"))
    );
    assert_eq!(
        legacy_dir(&env(&[("HOME", "H")])),
        Some(PathBuf::from("H/.config").join("Alelyon"))
    );
    assert_eq!(legacy_dir(&env(&[])), None);
}

#[test]
fn with_no_home_directory_only_a_named_file_is_kept() {
    let s = Scratch::new("no-home");
    let dock = s.0.join("layout.json");
    let named = env(&[("ANGEL_DOCK", dock.to_str().unwrap())]);
    let run = Run::start(&named, None, &s.0, SystemTime::now());
    assert_eq!(run.report.status, Status::NoHome);
    assert_eq!(run.path(File::Dock), Some(dock.as_path()));
    assert_eq!(run.path(File::Appearance), None);
}

// ---- moving in ----------------------------------------------------------------

#[test]
fn the_first_own_start_copies_everything_once_and_leaves_the_old_folder() {
    let s = Scratch::new("first-start");
    let (roaming, home) = (s.0.join("roaming"), s.0.join("home"));
    let files = old_files(&roaming);
    let person = appdata(&roaming, &[]);
    let before = tree(&roaming.join("Alelyon"));

    let report = move_in(&person, Some(&home), SystemTime::now());

    let target = home.join(".alelyon").join("angel");
    assert_eq!(report.status, Status::Moved);
    assert_eq!(report.files, files.len());
    let mut moved = tree(&target);
    let record = moved.remove(MARKER).unwrap();
    assert_eq!(moved, before, "every file, byte for byte");
    assert_eq!(
        tree(&roaming.join("Alelyon")),
        before,
        "the old folder is untouched"
    );
    let record: serde_json::Value = serde_json::from_slice(&record).unwrap();
    let listed: Vec<&str> = record["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["file"].as_str().unwrap())
        .collect();
    assert_eq!(listed, files.keys().map(String::as_str).collect::<Vec<_>>());
    assert_eq!(
        record["from"].as_str(),
        Some(roaming.join("Alelyon").to_str().unwrap())
    );

    let again = move_in(&person, Some(&home), SystemTime::now());
    assert_eq!(again.status, Status::Already);
    let mut kept = before.clone();
    kept.insert(MARKER.to_string(), std::fs::read(target.join(MARKER)).unwrap());
    assert_eq!(tree(&target), kept, "a second start changed the home");
}

#[test]
fn the_record_holds_each_files_sha256() {
    let s = Scratch::new("sha");
    let (roaming, home) = (s.0.join("roaming"), s.0.join("home"));
    std::fs::create_dir_all(roaming.join("Alelyon")).unwrap();
    // FIPS 180-2, appendix B.1: SHA-256("abc").
    std::fs::write(roaming.join("Alelyon").join("angel-dock.json"), b"abc").unwrap();
    move_in(&appdata(&roaming, &[]), Some(&home), SystemTime::now());
    let record = marker(&home.join(".alelyon").join("angel"));
    assert_eq!(
        record["files"],
        serde_json::json!([{
            "file": "angel-dock.json",
            "bytes": 3,
            "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        }])
    );
}

#[test]
fn an_agent_never_moves_anything() {
    let s = Scratch::new("agent-never");
    let (roaming, home) = (s.0.join("roaming"), s.0.join("home"));
    old_files(&roaming);
    let report = move_in(&appdata(&roaming, AGENT), Some(&home), SystemTime::now());
    assert_eq!(report.status, Status::Deferred);
    assert!(!home.join(".alelyon").exists());
    let report = move_in(
        &appdata(&roaming, &[("AI_AGENT", "codex")]),
        Some(&home),
        SystemTime::now(),
    );
    assert_eq!(report.status, Status::Deferred);
    assert!(!home.join(".alelyon").exists());
}

#[test]
fn with_no_old_files_the_home_is_still_marked() {
    let s = Scratch::new("nothing");
    let home = s.0.join("home");
    let report = move_in(&appdata(&s.0.join("none"), &[]), Some(&home), SystemTime::now());
    assert_eq!(report.status, Status::Nothing);
    assert_eq!(
        marker(&home.join(".alelyon").join("angel"))["files"],
        serde_json::json!([])
    );
}

#[test]
fn the_loops_marker_does_not_stop_the_windows_move() {
    // The loop's first start can come before the window's: it moves its own
    // folder and leaves MIGRATED.json, which says nothing about these files.
    let s = Scratch::new("loop-first");
    let (roaming, home) = (s.0.join("roaming"), s.0.join("home"));
    let files = old_files(&roaming);
    let target = home.join(".alelyon").join("angel");
    std::fs::create_dir_all(&target).unwrap();
    std::fs::write(target.join("MIGRATED.json"), b"{\"files\": []}").unwrap();
    assert_ne!(MARKER, "MIGRATED.json");

    let report = move_in(&appdata(&roaming, &[]), Some(&home), SystemTime::now());

    assert_eq!(report.status, Status::Moved);
    for (name, bytes) in &files {
        assert_eq!(&std::fs::read(target.join(name)).unwrap(), bytes);
    }
    assert_eq!(
        std::fs::read(target.join("MIGRATED.json")).unwrap(),
        b"{\"files\": []}",
        "the window's move changed the loop's marker"
    );
}

#[test]
fn a_start_that_names_one_file_still_moves_both() {
    // Trying a dock template with ANGEL_DOCK set is the README's own advice. The
    // appearance it does not name comes from the home, so the move must run, and
    // it takes both files, or the layout would be left behind for good.
    let s = Scratch::new("one-named");
    let (roaming, home, temp) = (s.0.join("roaming"), s.0.join("home"), s.0.join("temp"));
    let files = old_files(&roaming);
    let template = s.0.join("template-copy.json");
    let trying = appdata(&roaming, &[("ANGEL_DOCK", template.to_str().unwrap())]);

    let run = Run::start(&trying, Some(&home), &temp, SystemTime::now());

    let target = home.join(".alelyon").join("angel");
    assert_eq!(run.report.status, Status::Moved);
    assert_eq!(run.path(File::Dock), Some(template.as_path()));
    assert_eq!(
        run.path(File::Appearance),
        Some(target.join("sinai-appearance.json").as_path())
    );
    for (name, bytes) in &files {
        assert_eq!(&std::fs::read(target.join(name)).unwrap(), bytes);
    }
}

#[test]
fn an_interrupted_move_leaves_nothing_behind() {
    // A partial copy, or a half-written file with no marker yet, can only be
    // left by a start that died during its move.
    let s = Scratch::new("interrupted");
    let (roaming, home) = (s.0.join("roaming"), s.0.join("home"));
    let files = old_files(&roaming);
    std::fs::remove_file(roaming.join("Alelyon").join("sinai-appearance.json")).unwrap();
    let target = home.join(".alelyon").join("angel");
    std::fs::create_dir_all(&target).unwrap();
    std::fs::write(target.join("angel-dock.json"), b"half a layout").unwrap();
    std::fs::write(target.join("angel-dock.json.copying"), b"partial").unwrap();
    std::fs::write(target.join("sinai-appearance.json.copying"), b"partial").unwrap();
    std::fs::write(target.join("unrelated.moving"), b"the loop's").unwrap();

    let report = move_in(&appdata(&roaming, &[]), Some(&home), SystemTime::now());

    assert_eq!(report.status, Status::Moved);
    assert_eq!(
        std::fs::read(target.join("angel-dock.json")).unwrap(),
        files["angel-dock.json"]
    );
    assert!(!target.join("angel-dock.json.copying").exists());
    assert!(!target.join("sinai-appearance.json.copying").exists());
    assert!(
        target.join("unrelated.moving").exists(),
        "the window's move removed a file that is not its own"
    );
}

#[test]
fn a_copy_that_does_not_verify_writes_no_marker() {
    let s = Scratch::new("no-verify");
    let (roaming, home) = (s.0.join("roaming"), s.0.join("home"));
    old_files(&roaming);
    let target = home.join(".alelyon").join("angel");
    let corrupt = |path: &Path| {
        if path.starts_with(&target) {
            Ok("corrupt".to_string())
        } else {
            sha256(path)
        }
    };

    let report = move_in_checked_by(
        &appdata(&roaming, &[]),
        Some(&home),
        SystemTime::now(),
        &corrupt,
    );

    assert!(
        matches!(&report.status, Status::Failed(why) if why.contains("did not verify")),
        "{report:?}"
    );
    assert!(!target.join(MARKER).exists(), "the next start must try again");
}

#[test]
fn a_failed_move_keeps_the_run_on_the_old_files() {
    // Writing into the home after a failed move would be undone by the next
    // start's copy; the old files are what that copy takes.
    let s = Scratch::new("failed");
    let (roaming, home, temp) = (s.0.join("roaming"), s.0.join("home"), s.0.join("temp"));
    let person = appdata(&roaming, &[]);
    let failed = Report {
        status: Status::Failed("disk full".into()),
        from: Some(roaming.join("Alelyon")),
        to: Some(home.join(".alelyon").join("angel")),
        files: 0,
    };
    for file in File::ALL {
        assert_eq!(
            resolve(file, &person, &failed, Some(&home), &temp),
            Some(roaming.join("Alelyon").join(file.name()))
        );
    }
    // The run says so where each file's notices are shown.
    let run = Run {
        report: failed,
        dock: None,
        appearance: None,
        stayed: vec![File::Appearance],
    };
    assert!(run.notice(File::Appearance).unwrap().contains("disk full"));
    assert_eq!(run.notice(File::Dock), None);
}

#[test]
fn a_move_that_cannot_place_a_copy_fails_and_the_run_stays_on_the_old_files() {
    // A directory where the copy of the layout must go cannot be replaced. The move
    // reports it, and that run keeps both old paths, as resolve says.
    let s = Scratch::new("unreadable");
    let (roaming, home, temp) = (s.0.join("roaming"), s.0.join("home"), s.0.join("temp"));
    old_files(&roaming);
    let blocked = home.join(".alelyon").join("angel").join("angel-dock.json");
    std::fs::create_dir_all(blocked.join("in-the-way")).unwrap();

    let run = Run::start(&appdata(&roaming, &[]), Some(&home), &temp, SystemTime::now());

    assert!(
        matches!(run.report.status, Status::Failed(_)),
        "{:?}",
        run.report
    );
    assert!(!home.join(".alelyon").join("angel").join(MARKER).exists());
    for file in File::ALL {
        assert_eq!(
            run.path(file),
            Some(roaming.join("Alelyon").join(file.name()).as_path())
        );
        assert!(run.notice(file).is_some());
    }
    assert!(
        blocked.join("in-the-way").is_dir(),
        "the move removed something it did not create"
    );
}

#[test]
fn the_marker_is_stamped_in_utc() {
    let at = std::time::UNIX_EPOCH + std::time::Duration::from_millis(1_727_000_000_250);
    assert_eq!(utc_iso(at), "2024-09-22T10:13:20.250000+00:00");
    assert_eq!(utc_iso(std::time::UNIX_EPOCH), "1970-01-01T00:00:00.000000+00:00");
    // A leap day, and the last second of a century that is not a leap year.
    let leap = std::time::UNIX_EPOCH + std::time::Duration::from_secs(951_782_400);
    assert_eq!(utc_iso(leap), "2000-02-29T00:00:00.000000+00:00");
    let century = std::time::UNIX_EPOCH + std::time::Duration::from_secs(4_107_542_399);
    assert_eq!(utc_iso(century), "2100-02-28T23:59:59.000000+00:00");
}
