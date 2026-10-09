//! The process environment, as a value that can be replaced.
//!
//! Every rule in this crate that reads the environment (the state root, the key
//! search, the model registry's path, the local model's address and name) reads
//! it through [`Env`]. Production hands them [`ProcessEnv`]; tests hand them a
//! [`MapEnv`], so no test has to change the real process environment (which is
//! global to every test thread, and in Rust 2024 an `unsafe` operation).
//!
//! Invariant: a lookup never fails and never allocates more than the value; an
//! unset variable and one this map does not hold are the same answer, `None`.

use std::collections::BTreeMap;
use std::ffi::OsString;

/// Where environment variables come from.
pub trait Env: Send + Sync {
    /// The raw value, or `None` when the variable is not set.
    fn var(&self, name: &str) -> Option<OsString>;

    /// An API key kept by the system's credential store under `name` (Windows
    /// Credential Manager, `crate::keys::vault_target`), or `None`. Only the
    /// real process environment has one; a test environment has none, so no
    /// test reads the person's real credentials.
    fn secret(&self, _name: &str) -> Option<String> {
        None
    }
}

/// The real process environment.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProcessEnv;

impl Env for ProcessEnv {
    fn var(&self, name: &str) -> Option<OsString> {
        std::env::var_os(name)
    }

    fn secret(&self, name: &str) -> Option<String> {
        crate::keys::vault_read(name)
    }
}

/// A fixed set of variables, for tests and for callers that want to run the
/// services against an environment of their own.
#[derive(Clone, Debug, Default)]
pub struct MapEnv(BTreeMap<String, OsString>);

impl MapEnv {
    pub fn new() -> Self {
        Self::default()
    }

    /// This map with `name` set to `value`.
    pub fn with(mut self, name: &str, value: impl Into<OsString>) -> Self {
        self.0.insert(name.to_owned(), value.into());
        self
    }

    pub fn set(&mut self, name: &str, value: impl Into<OsString>) {
        self.0.insert(name.to_owned(), value.into());
    }
}

impl Env for MapEnv {
    fn var(&self, name: &str) -> Option<OsString> {
        self.0.get(name).cloned()
    }
}

/// A variable as text (Python's `os.environ.get`): undecodable bytes become
/// U+FFFD rather than making the variable disappear.
pub(crate) fn text(env: &dyn Env, name: &str) -> Option<String> {
    env.var(name)
        .map(|value| value.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_map_env_answers_only_what_it_holds() {
        let env = MapEnv::new().with("A", "1");
        assert_eq!(text(&env, "A").as_deref(), Some("1"));
        assert_eq!(text(&env, "B"), None);
    }

    #[test]
    fn the_process_env_reads_the_real_environment() {
        // PATH exists on every platform this runs on (Windows spells it `Path`
        // and looks names up case-insensitively).
        assert!(ProcessEnv.var("PATH").is_some());
        assert!(
            ProcessEnv
                .var("LATTICE_CORE_SURELY_UNSET_VARIABLE")
                .is_none()
        );
    }
}
