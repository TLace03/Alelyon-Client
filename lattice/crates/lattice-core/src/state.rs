//! Where Lattice keeps its state.
//!
//! Parity with the Python runtime's `paths._resolve`: the native Lattice and
//! the Python Lattice must agree on the state directory, or the model registry
//! one edits is not the one the other reads. First match wins:
//!
//! 1. `ALELYON_HOME`, when set and not blank: that directory (made absolute),
//!    state in `<it>/globals`.
//! 2. `ALELYON_FORCE_PACKAGED` in {`1`, `true`, `yes`, `on`} (trimmed, any case):
//!    the per-user directory (below).
//! 3. A source checkout: the first ancestor of the running executable that
//!    holds a `pyproject.toml`. A development build lives under
//!    `<checkout>/.../lattice_native/target/...`, so it finds its
//!    own checkout, per worktree, exactly as the Python runtime does.
//! 4. The per-user directory: `%LOCALAPPDATA%` (else `%APPDATA%`, else
//!    `~/AppData/Local`) `/Alelyon` on Windows, `~/Library/Application
//!    Support/Alelyon` on macOS, else `$XDG_DATA_HOME/alelyon` or
//!    `~/.local/share/alelyon`; state in `<it>/globals`, `installed = true`.
//!
//! `ALELYON_HOME` is expanded as Python's `Path(...).expanduser()` expands it
//! for a leading `~` alone or `~/...` (`~\...` too on Windows): the home
//! directory is `USERPROFILE`, else `HOMEDRIVE` joined to `HOMEPATH` (the drive
//! may be absent) on Windows, and `HOME` elsewhere.
//!
//! Deliberate differences from Python: `ALELYON_HOME` is made absolute and
//! tidied lexically (`.` and `..`), not by resolving symlinks; a checkout
//! is not searched from a path that runs through `site-packages`, which only
//! a Python file can do; `~user` (another account's home) is not expanded
//! (Python guesses it from the current profile's parent directory, or refuses),
//! so `~bob/x` stays the relative name `~bob/x` under the working directory;
//! and where Python has no home directory at all and raises, the temporary
//! directory stands in (a state directory relative to the working directory
//! would move with it), as does Unix without `HOME` (Python asks the password
//! database).
//!
//! Invariant: resolving never touches the disk beyond looking for
//! `pyproject.toml`, and never creates a directory. Whoever writes creates the
//! directory it writes into, at the moment it writes.

use std::path::{Component, Path, PathBuf};

use crate::env::{self, Env, ProcessEnv};

const APP_DIR: &str = "Alelyon";

/// The platform, as far as the state directory's location is concerned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Platform {
    Windows,
    MacOs,
    Other,
}

impl Platform {
    pub fn host() -> Self {
        if cfg!(windows) {
            Platform::Windows
        } else if cfg!(target_os = "macos") {
            Platform::MacOs
        } else {
            Platform::Other
        }
    }
}

/// The state root and the directory under it that holds Lattice's files.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateRoot {
    /// The checkout root, or the state root when nothing is checked out.
    pub root: PathBuf,
    /// The state home: `<root>/globals`.
    pub globals: PathBuf,
    /// True when no source checkout backs `root`.
    pub installed: bool,
}

impl StateRoot {
    /// A state root at `root` (not an installation): for tests and for callers
    /// that were told where the state is.
    pub fn at(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            globals: root.join("globals"),
            root,
            installed: false,
        }
    }

    /// `<globals>/lattice_native/runs`: where runs are kept.
    pub fn runs_dir(&self) -> PathBuf {
        self.globals.join("lattice_native").join("runs")
    }

    /// `<globals>/lattice_chat`: the chat store the web Lattice shares
    /// (the native chat's spec S1; Python's `GLOBALS_DIR / "lattice_chat"`).
    pub fn chat_dir(&self) -> PathBuf {
        self.globals.join("lattice_chat")
    }

    /// `<globals>/lattice_native/chat/answering`: one lock file per thread
    /// while this process writes its answer (the native chat's spec §3.3.5). It
    /// is outside the shared store, so the store's layout is unchanged.
    pub fn chat_locks_dir(&self) -> PathBuf {
        self.native_chat_dir().join("answering")
    }

    /// `<globals>/lattice_native/chat`: the agent chat's own state (`<native>/chat`
    /// in the chat core's spec §5.7).
    pub fn native_chat_dir(&self) -> PathBuf {
        self.globals.join("lattice_native").join("chat")
    }

    /// `<native>/chat/runs`: the agent chat's turn traces, in the run store's
    /// format (the chat core's spec §5.7, T8), apart from the run
    /// manager's `runs/`.
    pub fn chat_runs_dir(&self) -> PathBuf {
        self.native_chat_dir().join("runs")
    }

    /// `<native>/chat/no-hooks`: the empty folder every git call names as
    /// `core.hooksPath` (spec §8.2).
    pub fn no_hooks_dir(&self) -> PathBuf {
        self.native_chat_dir().join("no-hooks")
    }
}

/// The state root for this process.
pub fn resolve() -> StateRoot {
    let exe = std::env::current_exe().ok();
    resolve_with(&ProcessEnv, exe.as_deref(), Platform::host())
}

/// [`resolve`] over an environment, an executable path and a platform of the
/// caller's choosing.
pub fn resolve_with(env: &dyn Env, exe: Option<&Path>, platform: Platform) -> StateRoot {
    if let Some(home) =
        env::text(env, "ALELYON_HOME").filter(|home| !crate::py::strip(home).is_empty())
    {
        return StateRoot::at(absolute(&expand_user(&home, env, platform)));
    }
    if forced_packaged(env) {
        return packaged(env, platform);
    }
    if let Some(root) = exe.and_then(checkout_of) {
        return StateRoot::at(root);
    }
    packaged(env, platform)
}

/// `ALELYON_FORCE_PACKAGED`: `(value or "").strip().lower() in {"1", "true", "yes", "on"}`.
pub fn forced_packaged(env: &dyn Env) -> bool {
    env::text(env, "ALELYON_FORCE_PACKAGED").is_some_and(|value| {
        matches!(
            crate::py::strip(&value).to_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn packaged(env: &dyn Env, platform: Platform) -> StateRoot {
    let root = user_state_dir(env, platform);
    StateRoot {
        globals: root.join("globals"),
        root,
        installed: true,
    }
}

/// The first ancestor of `exe` that holds a `pyproject.toml`.
fn checkout_of(exe: &Path) -> Option<PathBuf> {
    exe.ancestors()
        .skip(1)
        .find(|dir| dir.join("pyproject.toml").is_file())
        .map(Path::to_path_buf)
}

/// The platform's per-user application-data directory for Lattice.
pub fn user_state_dir(env: &dyn Env, platform: Platform) -> PathBuf {
    let non_empty = |name: &str| env::text(env, name).filter(|value| !value.is_empty());
    match platform {
        Platform::Windows => match non_empty("LOCALAPPDATA").or_else(|| non_empty("APPDATA")) {
            Some(base) => PathBuf::from(base).join(APP_DIR),
            None => home_dir(env, platform)
                .join("AppData")
                .join("Local")
                .join(APP_DIR),
        },
        Platform::MacOs => home_dir(env, platform)
            .join("Library")
            .join("Application Support")
            .join(APP_DIR),
        Platform::Other => match non_empty("XDG_DATA_HOME") {
            Some(base) => PathBuf::from(base).join(APP_DIR.to_lowercase()),
            None => home_dir(env, platform)
                .join(".local")
                .join("share")
                .join(APP_DIR.to_lowercase()),
        },
    }
}

/// `Path.home()`: `USERPROFILE` (else `HOMEDRIVE` + `HOMEPATH`) on Windows,
/// `HOME` elsewhere. With none of them set, the temporary directory: a state
/// directory relative to the working directory would move with it.
pub(crate) fn home_dir(env: &dyn Env, platform: Platform) -> PathBuf {
    let non_empty = |name: &str| env::text(env, name).filter(|value| !value.is_empty());
    let home = match platform {
        Platform::Windows => non_empty("USERPROFILE").or_else(|| {
            // `ntpath.expanduser`: HOMEPATH, behind HOMEDRIVE when there is one.
            let path = non_empty("HOMEPATH")?;
            let drive = non_empty("HOMEDRIVE").unwrap_or_default();
            Some(format!("{drive}{path}"))
        }),
        _ => non_empty("HOME"),
    };
    home.map(PathBuf::from).unwrap_or_else(std::env::temp_dir)
}

/// `Path.expanduser()` for a leading `~` alone or followed by a separator (`/`,
/// and `\` on Windows only, where Python treats it as one).
fn expand_user(path: &str, env: &dyn Env, platform: Platform) -> PathBuf {
    let separators: &[char] = if platform == Platform::Windows {
        &['/', '\\']
    } else {
        &['/']
    };
    let rest = match path.strip_prefix('~') {
        Some(rest) if rest.is_empty() || rest.starts_with(separators) => rest,
        _ => return PathBuf::from(path),
    };
    let mut home = home_dir(env, platform);
    let rest = rest.trim_start_matches(separators);
    if !rest.is_empty() {
        home.push(rest);
    }
    home
}

/// `os.path.normpath` of an absolute path (the writer lease's state home):
/// `.` and `..` folded away lexically, nothing read from the disk.
pub(crate) fn tidy(path: &Path) -> PathBuf {
    absolute(path)
}

/// An absolute path with `.` and `..` folded away lexically.
fn absolute(path: &Path) -> PathBuf {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    let mut tidy = PathBuf::new();
    for component in joined.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                tidy.pop();
            }
            other => tidy.push(other.as_os_str()),
        }
    }
    tidy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::MapEnv;
    use crate::testkit::TempDir;

    fn win() -> Platform {
        Platform::Windows
    }

    #[test]
    fn alelyon_home_wins_and_is_made_absolute() {
        let dir = TempDir::new("state-home");
        let home = dir.path().join("a").join("..").join("b");
        let env = MapEnv::new()
            .with("ALELYON_HOME", home.as_os_str())
            .with("ALELYON_FORCE_PACKAGED", "1");
        let state = resolve_with(&env, Some(Path::new("/nowhere/lattice")), win());
        assert_eq!(state.root, dir.path().join("b"));
        assert_eq!(state.globals, dir.path().join("b").join("globals"));
        assert!(!state.installed);
    }

    #[test]
    fn a_blank_alelyon_home_is_ignored() {
        let env = MapEnv::new()
            .with("ALELYON_HOME", "  \t")
            .with("ALELYON_FORCE_PACKAGED", "yes")
            .with("LOCALAPPDATA", "D:\\Profiles\\x\\AppData\\Local");
        let state = resolve_with(&env, None, win());
        assert!(state.installed, "a blank ALELYON_HOME is not an answer");
        assert_eq!(
            state.root,
            PathBuf::from("D:\\Profiles\\x\\AppData\\Local").join("Alelyon")
        );
    }

    #[test]
    fn a_relative_alelyon_home_is_anchored_at_the_working_directory() {
        let env = MapEnv::new().with("ALELYON_HOME", "some/where/./else");
        let state = resolve_with(&env, None, win());
        let expected = std::env::current_dir()
            .unwrap()
            .join("some")
            .join("where")
            .join("else");
        assert_eq!(state.root, expected);
    }

    #[test]
    fn a_leading_tilde_expands_to_the_home_directory() {
        let env = MapEnv::new()
            .with("ALELYON_HOME", "~/lat")
            .with("USERPROFILE", "D:\\Profiles\\x");
        let state = resolve_with(&env, None, win());
        assert_eq!(state.root, PathBuf::from("D:\\Profiles\\x").join("lat"));
        let bare = MapEnv::new()
            .with("ALELYON_HOME", "~")
            .with("USERPROFILE", "D:\\Profiles\\x");
        assert_eq!(
            resolve_with(&bare, None, win()).root,
            PathBuf::from("D:\\Profiles\\x")
        );
        let not_a_home = MapEnv::new()
            .with("ALELYON_HOME", "~other")
            .with("USERPROFILE", "D:\\Profiles\\x");
        assert!(
            resolve_with(&not_a_home, None, win())
                .root
                .ends_with("~other"),
            "only a bare tilde is expanded"
        );
        // `~user` forms stay a documented deviation: they are not expanded.
        let other_user = MapEnv::new()
            .with("ALELYON_HOME", "~bob/x")
            .with("USERPROFILE", "D:\\Profiles\\x")
            .with("HOME", "/home/x");
        for platform in [Platform::Windows, Platform::Other] {
            let root = resolve_with(&other_user, None, platform).root;
            assert!(
                root.ends_with(Path::new("~bob").join("x")),
                "{platform:?}: {root:?}"
            );
        }
        // A backslash after the tilde is a separator on Windows only.
        let backslash = MapEnv::new()
            .with("ALELYON_HOME", "~\\lat")
            .with("USERPROFILE", "D:\\Profiles\\x")
            .with("HOME", "/home/x");
        assert_eq!(
            resolve_with(&backslash, None, Platform::Windows).root,
            PathBuf::from("D:\\Profiles\\x").join("lat")
        );
        assert!(
            !resolve_with(&backslash, None, Platform::Other)
                .root
                .starts_with("/home/x"),
            "on Unix `~\\lat` is one odd name, not the home directory"
        );
    }

    #[test]
    fn a_tilde_expands_from_homepath_alone_as_ntpath_expanduser_does() {
        // `ntpath.expanduser`: USERPROFILE, else HOMEDRIVE joined to HOMEPATH, and
        // the drive may be absent.
        let env = MapEnv::new()
            .with("ALELYON_HOME", "~/lat")
            .with("HOMEPATH", "D:\\Profiles\\x");
        assert_eq!(
            resolve_with(&env, None, win()).root,
            PathBuf::from("D:\\Profiles\\x").join("lat")
        );
        let with_drive = MapEnv::new()
            .with("ALELYON_HOME", "~")
            .with("HOMEDRIVE", "D:")
            .with("HOMEPATH", "\\h");
        assert_eq!(
            resolve_with(&with_drive, None, win()).root,
            PathBuf::from("D:\\h")
        );
        // USERPROFILE wins over both.
        let both = MapEnv::new()
            .with("ALELYON_HOME", "~/lat")
            .with("USERPROFILE", "D:\\Profiles\\p")
            .with("HOMEDRIVE", "D:")
            .with("HOMEPATH", "\\h");
        assert_eq!(
            resolve_with(&both, None, win()).root,
            PathBuf::from("D:\\Profiles\\p").join("lat")
        );
    }

    #[test]
    fn force_packaged_accepts_exactly_python_s_truthy_words() {
        for (value, expected) in [
            ("1", true),
            ("true", true),
            ("YES", true),
            (" On ", true),
            ("0", false),
            ("no", false),
            ("", false),
            ("2", false),
            ("truee", false),
        ] {
            let env = MapEnv::new().with("ALELYON_FORCE_PACKAGED", value);
            assert_eq!(forced_packaged(&env), expected, "{value:?}");
        }
        assert!(!forced_packaged(&MapEnv::new()));
    }

    #[test]
    fn a_checkout_is_the_first_ancestor_with_a_pyproject() {
        let dir = TempDir::new("state-checkout");
        let outer = dir.path().join("outer");
        let inner = outer.join("inner");
        let exe_dir = inner
            .join("alelyon")
            .join("frontend")
            .join("target")
            .join("debug");
        std::fs::create_dir_all(&exe_dir).unwrap();
        std::fs::write(outer.join("pyproject.toml"), "").unwrap();
        let exe = exe_dir.join("lattice.exe");
        let env = MapEnv::new();
        assert_eq!(
            resolve_with(&env, Some(&exe), win()).root,
            outer,
            "the outer one, the inner has none"
        );
        std::fs::write(inner.join("pyproject.toml"), "").unwrap();
        let state = resolve_with(&env, Some(&exe), win());
        assert_eq!(state.root, inner, "the nearest wins");
        assert_eq!(state.globals, inner.join("globals"));
        assert!(!state.installed);
        assert!(!inner.join("globals").exists(), "resolving creates nothing");
    }

    #[test]
    fn without_a_checkout_the_per_user_directory_is_used() {
        let dir = TempDir::new("state-none");
        let exe = dir.path().join("bin").join("lattice.exe");
        std::fs::create_dir_all(exe.parent().unwrap()).unwrap();
        // The temporary directory sits under no pyproject.toml of its own.
        let env = MapEnv::new().with("LOCALAPPDATA", "C:\\L");
        let state = resolve_with(&env, Some(&exe), win());
        // A stray pyproject.toml above the system temporary directory would
        // make this a checkout; that is the machine's doing, not the rule's.
        if !state.installed {
            assert!(exe.ancestors().any(|d| d.join("pyproject.toml").is_file()));
        } else {
            assert_eq!(state.root, PathBuf::from("C:\\L").join("Alelyon"));
            assert_eq!(
                state.globals,
                PathBuf::from("C:\\L").join("Alelyon").join("globals")
            );
        }
    }

    #[test]
    fn the_per_user_directory_follows_each_platform_s_convention() {
        let local = MapEnv::new()
            .with("LOCALAPPDATA", "L")
            .with("APPDATA", "A")
            .with("USERPROFILE", "U");
        assert_eq!(
            user_state_dir(&local, Platform::Windows),
            PathBuf::from("L").join("Alelyon")
        );
        let roaming = MapEnv::new().with("APPDATA", "A").with("USERPROFILE", "U");
        assert_eq!(
            user_state_dir(&roaming, Platform::Windows),
            PathBuf::from("A").join("Alelyon")
        );
        let empty_local = MapEnv::new().with("LOCALAPPDATA", "").with("APPDATA", "A");
        assert_eq!(
            user_state_dir(&empty_local, Platform::Windows),
            PathBuf::from("A").join("Alelyon"),
            "an empty variable is unset, as in Python"
        );
        let profile = MapEnv::new().with("USERPROFILE", "U");
        assert_eq!(
            user_state_dir(&profile, Platform::Windows),
            PathBuf::from("U")
                .join("AppData")
                .join("Local")
                .join("Alelyon")
        );
        let drive = MapEnv::new()
            .with("HOMEDRIVE", "D:")
            .with("HOMEPATH", "\\h");
        assert_eq!(
            user_state_dir(&drive, Platform::Windows),
            PathBuf::from("D:\\h")
                .join("AppData")
                .join("Local")
                .join("Alelyon")
        );

        let mac = MapEnv::new().with("HOME", "/Users/x");
        assert_eq!(
            user_state_dir(&mac, Platform::MacOs),
            PathBuf::from("/Users/x")
                .join("Library")
                .join("Application Support")
                .join("Alelyon")
        );

        let xdg = MapEnv::new()
            .with("XDG_DATA_HOME", "/data")
            .with("HOME", "/home/x");
        assert_eq!(
            user_state_dir(&xdg, Platform::Other),
            PathBuf::from("/data").join("alelyon")
        );
        let home = MapEnv::new().with("HOME", "/home/x");
        assert_eq!(
            user_state_dir(&home, Platform::Other),
            PathBuf::from("/home/x")
                .join(".local")
                .join("share")
                .join("alelyon")
        );
    }

    #[test]
    fn a_state_root_names_the_run_directory() {
        let state = StateRoot::at("/r");
        assert_eq!(
            state.runs_dir(),
            PathBuf::from("/r")
                .join("globals")
                .join("lattice_native")
                .join("runs")
        );
    }

    #[test]
    fn a_state_root_names_the_chat_runs_apart_from_the_runs() {
        let state = StateRoot::at("/r");
        let native = PathBuf::from("/r").join("globals").join("lattice_native");
        assert_eq!(state.chat_runs_dir(), native.join("chat").join("runs"));
        assert_ne!(state.chat_runs_dir(), state.runs_dir());
        assert!(!state.chat_runs_dir().starts_with(state.chat_dir()));
    }

    #[test]
    fn a_state_root_names_the_chat_store_and_the_answer_locks() {
        let state = StateRoot::at("/r");
        let globals = PathBuf::from("/r").join("globals");
        assert_eq!(state.chat_dir(), globals.join("lattice_chat"));
        assert_eq!(
            state.chat_locks_dir(),
            globals
                .join("lattice_native")
                .join("chat")
                .join("answering")
        );
        assert!(
            !state.chat_locks_dir().starts_with(state.chat_dir()),
            "the answer locks stay out of the shared store"
        );
    }
}
