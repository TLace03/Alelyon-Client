//! Helpers for this crate's tests: temporary directories, a wait-with-a-deadline
//! and one real browser at a time.
//!
//! Every test that touches the disk works inside a directory made here, under
//! the system temporary directory, and never in the real `globals/`. A directory
//! is named with a random suffix and removed when its `TempDir` drops; only
//! that directory, which this module created, is ever removed. The other
//! files are the turn files of [`one_real_browser`] and [`stop_keys`], empty
//! files in the system temporary directory that are locked and never removed.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError, mpsc};
use std::time::Duration;

use uuid::Uuid;

pub(crate) struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub(crate) fn new(tag: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("lattice-core-{tag}-{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&path).expect("a temporary directory");
        Self { path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Whether `path` lies inside `root`, compared after `.` and `..` are folded
/// away, nothing read from the disk.
pub(crate) fn is_within(root: &Path, path: &Path) -> bool {
    crate::state::tidy(path).starts_with(crate::state::tidy(root))
}

/// Refuse a test's write outside its own directory: a fixture handed the
/// reader's own home (as the by-hand tests of the labs' agents are) must
/// still never write a stand-in into the reader's real `~/.alelyon`.
#[track_caller]
pub(crate) fn assert_within(root: &Path, path: &Path) {
    assert!(
        is_within(root, path),
        "a test wrote outside its own directory: {} is not under {}",
        path.display(),
        root.display()
    );
}

/// Run `work` on a thread and wait for its answer for at most `seconds`:
/// a hang is a failed test with a name, not a stuck run.
pub(crate) fn within<T: Send + 'static>(
    what: &str,
    seconds: u64,
    work: impl FnOnce() -> T + Send + 'static,
) -> T {
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = sender.send(work());
    });
    receiver
        .recv_timeout(Duration::from_secs(seconds))
        .unwrap_or_else(|_| panic!("{what} did not finish within {seconds} s"))
}

/// How long a test's browser has to answer its first DevTools request: every
/// test starts one on a fresh profile, and a fresh profile's first start on a
/// loaded machine has taken over 30 s (the browser's own
/// [`crate::browser::launch::START_WAIT`]) with no other test's browser
/// running. A bound on a hang; a browser that answers sooner goes on at once.
pub(crate) const BROWSER_START_WAIT: Duration = Duration::from_secs(180);

/// How long a test's browser action waits for its page to load: a page this
/// test serves on loopback has stayed `about:blank` past the browser's own
/// 15 s ([`crate::browser::session::LOAD_WAIT`]) on a loaded machine. A bound on a hang.
pub(crate) const BROWSER_LOAD_WAIT: Duration = Duration::from_secs(120);

/// A turn on something one test at a time may use (a real browser, the stop
/// keys): held for the whole of a test, and given up when it drops.
pub(crate) struct Turn {
    // Unlocked as the file closes; it drops before the process's own turn.
    _machine: File,
    _process: MutexGuard<'static, ()>,
}

/// The real browser's turn file in the system temporary directory.
const TURN_FILE: &str = "lattice-core-real-browser.lock";
/// The stop keys' turn file in the system temporary directory.
const STOP_KEYS_FILE: &str = "lattice-core-stop-keys.lock";

/// Wait for a real browser's turn. Each such test starts a browser of its own
/// (17 processes for Edge here), and several at once on a loaded machine left
/// a browser that did not answer on its DevTools pipe within 30 s, or still
/// running 20 s after its pipe closed: a different test failed each run, in
/// several sessions' checks, and each passed alone. The turn is taken in this
/// process first, then on this machine (an exclusive lock on one file in the
/// system temporary directory, which every session's run under this user
/// shares), so two sessions' runs take turns too. A failed test's turn passes
/// on; it does not poison the next, and a process that ends gives its lock up
/// with it.
pub(crate) fn one_real_browser() -> Turn {
    static IN_PROCESS: Mutex<()> = Mutex::new(());
    turn(&IN_PROCESS, TURN_FILE)
}

/// Wait for the stop keys' turn, held by a test that registers auto mode's
/// stop keys. A hotkey is one program's at a time, and in this process each
/// test has keys of its own; another session's run of the same tests
/// registers the same keys, which refused this run's ("held by another
/// program"). Taken as [`one_real_browser`]'s is.
pub(crate) fn stop_keys() -> Turn {
    static IN_PROCESS: Mutex<()> = Mutex::new(());
    turn(&IN_PROCESS, STOP_KEYS_FILE)
}

fn turn(in_process: &'static Mutex<()>, file: &str) -> Turn {
    let process = in_process.lock().unwrap_or_else(PoisonError::into_inner);
    let path = std::env::temp_dir().join(file);
    let machine = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    machine
        .lock()
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    Turn {
        _machine: machine,
        _process: process,
    }
}

#[cfg(test)]
mod tests {
    use std::fs::{OpenOptions, TryLockError};

    /// While a test holds a real browser's turn, another handle on the turn
    /// file (another process's, in effect) cannot take it; once the turn
    /// drops, it can.
    #[test]
    fn a_real_browser_turn_holds_the_machines_lock_until_it_drops() {
        let path = std::env::temp_dir().join(super::TURN_FILE);
        let turn = super::one_real_browser();
        let other = OpenOptions::new().write(true).open(&path).unwrap();
        assert!(
            matches!(other.try_lock(), Err(TryLockError::WouldBlock)),
            "the turn file is locked while the turn is held"
        );
        drop(turn);
        // Another run's test may take the turn first; this waits for it.
        other.lock().unwrap();
    }

    /// A path is inside a root only below it, and `..` cannot climb out.
    #[test]
    fn a_path_is_within_a_root_only_below_it() {
        let root = std::env::temp_dir().join("lattice-core-within");
        assert!(super::is_within(&root, &root.join("home").join(".alelyon")));
        assert!(super::is_within(&root, &root));
        assert!(!super::is_within(&root, &root.join("..").join("elsewhere")));
        assert!(!super::is_within(
            &root,
            &std::env::temp_dir().join("lattice-core-within-not")
        ));
        assert!(!super::is_within(&root.join("home"), &root));
    }

    #[test]
    #[should_panic(expected = "a test wrote outside its own directory")]
    fn a_write_outside_the_tests_directory_is_refused() {
        let root = std::env::temp_dir().join("lattice-core-within");
        super::assert_within(&root, &root.join("..").join(".alelyon"));
    }
}
