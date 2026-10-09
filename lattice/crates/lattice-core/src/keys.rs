//! API keys: resolved by name, never stored, never printed.
//!
//! Parity with the Python runtime's `keys.get_key`: the process
//! environment first (trimmed, non-empty), then Windows Credential Manager
//! (below), then the env files, in this order, with earlier files winning:
//!
//! 1. the file named by `FAM_ENV_PATH`;
//! 2. `<root>/FAMEnvironment.env`;
//! 3. `<root>/engines/FAMEnvironment.env`.
//!
//! A file is lines of `KEY = value`. Blank lines, `#` comments and lines with no
//! `=` are skipped; a line splits at its first `=`; key and value are trimmed;
//! the value then loses every leading and trailing `"` and then every leading
//! and trailing `'` (Python's `strip('"').strip("'")`, which strips more than
//! one pair; the goldens pin it). An unreadable file is skipped silently. As in
//! Python, a key with an EMPTY value in an earlier file shadows the same key
//! in a later one, and a later duplicate line in one file replaces an earlier
//! one.
//!
//! Credential Manager (decided 2026-10-07: a key typed into
//! Alelyon is kept there, never in a plain-text file): a generic credential
//! whose target is `Alelyon/<NAME>` ([`vault_target`]), read through the
//! environment ([`Env::secret`]), so only the real process consults it and no
//! test does. It comes after the process environment (a variable set for one
//! run still wins) and before the files (a key saved in Alelyon wins over a
//! stale line in an env file). Python does not read it yet: until it does, a
//! key saved only there is seen by the native Lattice and Alelyon, not by the
//! web or PyQt Lattice.
//!
//! Deliberate differences from Python, all in the direction of less exposure:
//! - Python's fourth candidate is a hard-coded path in one developer's home
//!   directory; the client does not carry it.
//! - A key NAME must match `^[A-Z][A-Z0-9_]{2,63}$` or the lookup is refused
//!   (`None`); Python accepts any name.
//! - A file over 1 MiB is skipped (Python reads it whole).
//! - Files are read on every lookup and not cached, so a key added while
//!   Lattice runs is seen; Python parses once and offers `reload_keys()`.
//!
//! Invariant: a value is returned only as a [`SecretString`], whose `Debug` and
//! `Display` print `***`; this module never logs or formats a value.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub use lattice_agents::SecretString;

use crate::env::{self, Env, ProcessEnv};
use crate::py;
use crate::state::{self, StateRoot};

/// The largest env file that is read.
const MAX_ENV_FILE_BYTES: u64 = 1024 * 1024;

/// True when `name` may be looked up: `^[A-Z][A-Z0-9_]{2,63}$`.
pub fn valid_key_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    (3..=64).contains(&bytes.len())
        && bytes[0].is_ascii_uppercase()
        && bytes[1..]
            .iter()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || *b == b'_')
}

/// `_parse_env_file` over the text of a file: `(key, value)` pairs in file
/// order, later duplicates included.
pub fn parse_env_text(text: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for line in py::splitlines(text) {
        let line = py::strip(line);
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = py::strip(key);
        let value = py::strip(value).trim_matches('"').trim_matches('\'');
        if !key.is_empty() {
            pairs.push((key.to_owned(), value.to_owned()));
        }
    }
    pairs
}

/// The lookups for one environment and state root.
#[derive(Clone)]
pub struct KeyStore {
    env: Arc<dyn Env>,
    root: PathBuf,
}

impl KeyStore {
    pub fn new(env: Arc<dyn Env>, state: &StateRoot) -> Self {
        Self {
            env,
            root: state.root.clone(),
        }
    }

    /// The env files searched, in order (those that exist or not).
    pub fn candidates(&self) -> Vec<PathBuf> {
        let mut files = Vec::new();
        if let Some(explicit) =
            env::text(self.env.as_ref(), "FAM_ENV_PATH").filter(|p| !p.is_empty())
        {
            files.push(PathBuf::from(explicit));
        }
        files.push(self.root.join("FAMEnvironment.env"));
        files.push(self.root.join("engines").join("FAMEnvironment.env"));
        files
    }

    /// The key called `name`, if there is one. Never the empty string.
    pub fn get(&self, name: &str) -> Option<SecretString> {
        if !valid_key_name(name) {
            return None;
        }
        let from_env = env::text(self.env.as_ref(), name)
            .map(|value| py::strip(&value).to_owned())
            .filter(|value| !value.is_empty());
        if let Some(value) = from_env {
            return Some(SecretString::new(value));
        }
        if let Some(value) = self
            .env
            .secret(name)
            .map(|value| py::strip(&value).to_owned())
            .filter(|value| !value.is_empty())
        {
            return Some(SecretString::new(value));
        }
        for file in self.candidates() {
            let Some(text) = read_env_file(&file) else {
                continue;
            };
            // The last line of the file that names the key wins; a file that
            // names it at all decides, even with an empty value.
            if let Some((_, value)) = parse_env_text(&text)
                .into_iter()
                .rev()
                .find(|(k, _)| k == name)
            {
                let value = py::strip(&value);
                return (!value.is_empty()).then(|| SecretString::new(value));
            }
        }
        None
    }

    /// `has_key`: is there a non-empty value for `name`?
    pub fn has(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// Where the value `get` would return comes from, without returning it.
    pub fn source(&self, name: &str) -> Option<KeySource> {
        if !valid_key_name(name) {
            return None;
        }
        if env::text(self.env.as_ref(), name).is_some_and(|value| !py::strip(&value).is_empty()) {
            return Some(KeySource::Environment);
        }
        if self
            .env
            .secret(name)
            .is_some_and(|value| !py::strip(&value).is_empty())
        {
            return Some(KeySource::CredentialManager);
        }
        for file in self.candidates() {
            let Some(text) = read_env_file(&file) else {
                continue;
            };
            if let Some((_, value)) = parse_env_text(&text)
                .into_iter()
                .rev()
                .find(|(k, _)| k == name)
            {
                return (!py::strip(&value).is_empty()).then_some(KeySource::File(file));
            }
        }
        None
    }
}

/// Where a key's value was found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeySource {
    /// The process environment.
    Environment,
    /// Windows Credential Manager, under [`vault_target`].
    CredentialManager,
    /// An env file (a plain-text file).
    File(PathBuf),
}

/// The Credential Manager target a key called `name` is kept under.
pub fn vault_target(name: &str) -> String {
    format!("Alelyon/{name}")
}

/// The longest key value kept (Credential Manager's limit for a generic credential).
pub const MAX_KEY_BYTES: usize = lattice_sys::cred::MAX_BLOB;

/// The value kept in Credential Manager for `name`, as text, or `None` (none
/// kept, a name that may not be looked up, or a value that is not UTF-8).
pub fn vault_read(name: &str) -> Option<String> {
    if !valid_key_name(name) {
        return None;
    }
    let mut bytes = lattice_sys::cred::read_generic(&vault_target(name)).ok()??;
    let value = String::from_utf8(bytes.clone()).ok();
    bytes.iter_mut().for_each(|b| *b = 0);
    value
}

/// Keep `value` for `name` in Credential Manager (replacing any earlier one).
/// The value is trimmed as `get` trims it; an empty one is refused.
pub fn vault_store(name: &str, value: &SecretString) -> Result<(), String> {
    if !valid_key_name(name) {
        return Err(
            "a key name is 3 to 64 capital letters, digits or underscores, starting with a letter"
                .into(),
        );
    }
    let value = py::strip(value.expose());
    if value.is_empty() {
        return Err("the key is empty".into());
    }
    if value.len() > MAX_KEY_BYTES {
        return Err(format!("a key is at most {MAX_KEY_BYTES} bytes"));
    }
    lattice_sys::cred::write_generic(&vault_target(name), "Alelyon", value.as_bytes())
        .map_err(|e| format!("Credential Manager did not keep the key: {e}"))
}

/// Forget the key kept for `name` in Credential Manager. `Ok(false)` when none was kept.
pub fn vault_remove(name: &str) -> Result<bool, String> {
    if !valid_key_name(name) {
        return Ok(false);
    }
    lattice_sys::cred::delete_generic(&vault_target(name))
        .map_err(|e| format!("Credential Manager did not forget the key: {e}"))
}

impl std::fmt::Debug for KeyStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeyStore")
            .field("root", &self.root)
            .finish_non_exhaustive()
    }
}

/// The text of an env file, or `None` when it is not a readable UTF-8 file
/// within the size bound.
fn read_env_file(path: &Path) -> Option<String> {
    if !path.is_file() {
        return None;
    }
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .ok()?
        .take(MAX_ENV_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() as u64 > MAX_ENV_FILE_BYTES {
        return None;
    }
    String::from_utf8(bytes).ok()
}

/// `get_key(name)` for this process: its environment and its state root.
pub fn get_key(name: &str) -> Option<SecretString> {
    KeyStore::new(Arc::new(ProcessEnv), &state::resolve()).get(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::MapEnv;
    use crate::testkit::TempDir;

    fn store(env: MapEnv, root: &Path) -> KeyStore {
        KeyStore::new(Arc::new(env), &StateRoot::at(root))
    }

    fn expose(secret: Option<SecretString>) -> Option<String> {
        secret.map(|s| s.expose().to_owned())
    }

    #[test]
    fn key_names_are_environment_variable_names() {
        for good in ["ABC", "OPENAI_API_KEY", "HF_TOKEN", "A1_", &"A".repeat(64)] {
            assert!(valid_key_name(good), "{good}");
        }
        for bad in [
            "",
            "AB",
            "abc",
            "1ABC",
            "_ABC",
            "ABC-D",
            "ABC D",
            "ABC\n",
            &"A".repeat(65),
            "ÄBC",
        ] {
            assert!(!valid_key_name(bad), "{bad:?}");
        }
    }

    #[test]
    fn an_env_file_is_read_like_python_reads_it() {
        let text = "# comment\n\n  A_KEY = one  \nB_KEY=\"two\"\nC_KEY='three'\nno equals here\n=novalue\nD_KEY = a=b=c\nE_KEY =\nA_KEY = again\nQ_KEY = \"\"\"x\"\"\"\nM_KEY = \"'x'\"\nexport N_KEY=z\n";
        let pairs = parse_env_text(text);
        let get = |key: &str| {
            pairs
                .iter()
                .rev()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.as_str())
        };
        assert_eq!(
            get("A_KEY"),
            Some("again"),
            "a later line replaces an earlier one"
        );
        assert_eq!(get("B_KEY"), Some("two"));
        assert_eq!(get("C_KEY"), Some("three"));
        assert_eq!(get("D_KEY"), Some("a=b=c"), "a line splits at its first =");
        assert_eq!(get("E_KEY"), Some(""));
        assert_eq!(
            get("Q_KEY"),
            Some("x"),
            "every leading and trailing quote goes, as Python's strip does"
        );
        assert_eq!(
            get("M_KEY"),
            Some("x"),
            "double quotes first, then single quotes"
        );
        assert_eq!(
            get("export N_KEY"),
            Some("z"),
            "an `export` prefix is part of the key, as in Python"
        );
        assert_eq!(get("no equals here"), None);
        assert_eq!(pairs.iter().filter(|(k, _)| k.is_empty()).count(), 0);
    }

    #[test]
    fn the_process_environment_beats_every_file_and_is_trimmed() {
        let dir = TempDir::new("keys-env");
        std::fs::write(
            dir.path().join("FAMEnvironment.env"),
            "SOME_KEY = from-file\n",
        )
        .unwrap();
        let env = MapEnv::new().with("SOME_KEY", "  from-env \n");
        assert_eq!(
            expose(store(env, dir.path()).get("SOME_KEY")).as_deref(),
            Some("from-env")
        );
        let blank = MapEnv::new().with("SOME_KEY", "   ");
        assert_eq!(
            expose(store(blank, dir.path()).get("SOME_KEY")).as_deref(),
            Some("from-file"),
            "a blank variable is not an answer"
        );
    }

    #[test]
    fn earlier_files_win_and_an_empty_value_shadows() {
        let dir = TempDir::new("keys-order");
        let explicit = dir.path().join("explicit.env");
        std::fs::write(&explicit, "ALPHA_KEY = explicit\nBETA_KEY =\n").unwrap();
        std::fs::write(
            dir.path().join("FAMEnvironment.env"),
            "ALPHA_KEY = root\nBETA_KEY = root\nGAMMA_KEY = root\n",
        )
        .unwrap();
        std::fs::create_dir_all(dir.path().join("engines")).unwrap();
        std::fs::write(
            dir.path().join("engines").join("FAMEnvironment.env"),
            "ALPHA_KEY = engines\nGAMMA_KEY = engines\nDELTA_KEY = engines\n",
        )
        .unwrap();
        let keys = store(
            MapEnv::new().with("FAM_ENV_PATH", explicit.as_os_str()),
            dir.path(),
        );
        assert_eq!(expose(keys.get("ALPHA_KEY")).as_deref(), Some("explicit"));
        assert_eq!(
            keys.get("BETA_KEY").map(|_| ()),
            None,
            "an empty value in an earlier file shadows a later one"
        );
        assert_eq!(expose(keys.get("GAMMA_KEY")).as_deref(), Some("root"));
        assert_eq!(expose(keys.get("DELTA_KEY")).as_deref(), Some("engines"));
        assert_eq!(keys.get("EPSILON_KEY").map(|_| ()), None);
        assert!(keys.has("DELTA_KEY") && !keys.has("BETA_KEY"));
        assert_eq!(keys.candidates().len(), 3);
    }

    #[test]
    fn unreadable_missing_and_oversize_files_are_skipped() {
        let dir = TempDir::new("keys-skip");
        std::fs::write(
            dir.path().join("FAMEnvironment.env"),
            b"\xff\xfe not utf-8 \xff",
        )
        .unwrap();
        std::fs::create_dir_all(dir.path().join("engines")).unwrap();
        let mut big = b"BIG_KEY = x\n".to_vec();
        big.resize(MAX_ENV_FILE_BYTES as usize + 1, b'#');
        std::fs::write(dir.path().join("engines").join("FAMEnvironment.env"), big).unwrap();
        let keys = store(
            MapEnv::new().with("FAM_ENV_PATH", dir.path().join("missing.env").as_os_str()),
            dir.path(),
        );
        assert!(keys.get("BIG_KEY").is_none());
        // A directory where a file is expected is not a file.
        let as_dir = store(
            MapEnv::new().with("FAM_ENV_PATH", dir.path().as_os_str()),
            dir.path(),
        );
        assert!(as_dir.get("BIG_KEY").is_none());
    }

    #[test]
    fn an_invalid_name_is_refused_before_anything_is_read() {
        let dir = TempDir::new("keys-name");
        std::fs::write(dir.path().join("FAMEnvironment.env"), "lower = 1\nAB = 2\n").unwrap();
        let env = MapEnv::new().with("lower", "x").with("AB", "y");
        let keys = store(env, dir.path());
        assert!(keys.get("lower").is_none());
        assert!(keys.get("AB").is_none());
    }

    #[test]
    fn a_secret_prints_as_stars_through_the_store() {
        let dir = TempDir::new("keys-print");
        let env = MapEnv::new().with("PRINT_KEY", "very-secret-value");
        let secret = store(env, dir.path()).get("PRINT_KEY").unwrap();
        assert_eq!(format!("{secret:?} {secret}"), "*** ***");
        let keys = store(MapEnv::new(), dir.path());
        assert!(!format!("{keys:?}").contains("very-secret"));
    }

    #[test]
    fn the_process_environment_is_read_through_process_env_and_first() {
        // PATH is set on every platform this runs on, and is a valid key name.
        let dir = TempDir::new("keys-process");
        std::fs::write(dir.path().join("FAMEnvironment.env"), "PATH = from-file\n").unwrap();
        let keys = KeyStore::new(Arc::new(ProcessEnv), &StateRoot::at(dir.path()));
        let value = keys.get("PATH").expect("the process environment has PATH");
        assert!(!value.expose().is_empty() && value.expose() != "from-file");
        assert_eq!(value.expose(), value.expose().trim());
    }

    #[test]
    fn the_free_function_reads_this_process() {
        assert!(get_key("lower_case_is_refused").is_none());
        assert!(get_key("LATTICE_CORE_SURELY_UNSET_KEY_NAME").is_none());
    }

    /// An environment with a credential store of its own (the real one is never read by a test).
    struct VaultEnv(MapEnv, std::collections::BTreeMap<String, String>);

    impl Env for VaultEnv {
        fn var(&self, name: &str) -> Option<std::ffi::OsString> {
            self.0.var(name)
        }
        fn secret(&self, name: &str) -> Option<String> {
            self.1.get(name).cloned()
        }
    }

    #[test]
    fn credential_manager_comes_after_the_environment_and_before_the_files() {
        let dir = TempDir::new("keys-vault");
        std::fs::write(
            dir.path().join("FAMEnvironment.env"),
            "A_KEY = from-file
B_KEY = from-file
C_KEY = from-file
",
        )
        .unwrap();
        let vault = [
            ("A_KEY", "from-vault"),
            ("B_KEY", "  from-vault  "),
            ("D_KEY", "   "),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
        let keys = KeyStore::new(
            Arc::new(VaultEnv(MapEnv::new().with("A_KEY", "from-env"), vault)),
            &StateRoot::at(dir.path()),
        );
        assert_eq!(keys.get("A_KEY").unwrap().expose(), "from-env");
        assert_eq!(keys.source("A_KEY"), Some(KeySource::Environment));
        assert_eq!(
            keys.get("B_KEY").unwrap().expose(),
            "from-vault",
            "trimmed, and before the file"
        );
        assert_eq!(keys.source("B_KEY"), Some(KeySource::CredentialManager));
        assert_eq!(keys.get("C_KEY").unwrap().expose(), "from-file");
        assert_eq!(
            keys.source("C_KEY"),
            Some(KeySource::File(dir.path().join("FAMEnvironment.env")))
        );
        assert!(keys.get("D_KEY").is_none(), "a blank kept value is no key");
        assert_eq!(keys.source("D_KEY"), None);
    }

    #[test]
    fn a_test_environment_never_reads_the_real_credential_manager() {
        assert_eq!(MapEnv::new().secret("ANTHROPIC_API_KEY"), None);
    }

    /// The real Credential Manager, under a name no real key uses, left as it was found.
    #[cfg(windows)]
    #[test]
    fn a_key_is_kept_read_and_forgotten_in_credential_manager() {
        let name = format!("ZZ_LATTICE_CORE_TEST_{}", std::process::id());
        assert!(vault_read(&name).is_none());
        vault_store(&name, &SecretString::new("  not-a-real-key  ")).unwrap();
        assert_eq!(vault_read(&name).as_deref(), Some("not-a-real-key"));
        assert_eq!(ProcessEnv.secret(&name).as_deref(), Some("not-a-real-key"));
        assert!(vault_store(&name, &SecretString::new("   ")).is_err());
        assert!(vault_store("lower_case", &SecretString::new("x")).is_err());
        assert!(vault_remove(&name).unwrap());
        assert!(!vault_remove(&name).unwrap());
        assert!(vault_read(&name).is_none());
    }
}
