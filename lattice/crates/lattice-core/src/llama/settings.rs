//! How the managed server runs, and which build it is (the chat core's spec
//! §22 LR3, LR5; `llama_server.load_settings` and `install`).
//!
//! **Settings (LR5)** are `~/.alelyon/llama/settings.json`, shared with the
//! Python side and read-only here: `ctx_size`, `gpu_layers` and `parallel`
//! (an integer of at least 0, else Python's default: 8192, 999, 1) and
//! `idle_seconds` (a number above 0, else 600). A file that cannot be read,
//! is not UTF-8 text Python's `read_text` takes, is not JSON, or is not an
//! object gives every default. Deviations, each toward a value the server
//! can use:
//! - a boolean is not a number here (Python's `isinstance(True, int)` holds,
//!   so Python would pass `-c True` on the command line);
//! - an integer beyond `u64`, and a file holding `NaN` or `Infinity` (which
//!   Python reads), give the default;
//! - the idle time is capped at [`MAX_IDLE`], so a timer can always be armed.
//!
//! Which model runs is not a setting: it is the model Local uses
//! (`files::selected_model`), as in Python.
//!
//! **The build (LR3).** Python's `install` copies a llama.cpp build into
//! `~/.alelyon/llama` and writes `manifest.json` with each file's SHA-256.
//! The core only reads it: [`manifest_sha256`] is the binary's recorded
//! SHA-256 when the binary is the pinned install's, and `None` (UNMEASURED,
//! never guessed) otherwise.

use std::path::Path;
use std::time::Duration;

use super::files::LlamaPaths;
use crate::chat::pyjson::{self, PyValue};
use crate::exec::spawn::comparable;

pub const DEFAULT_CTX: u64 = 8192;
/// Every layer that fits; llama.cpp clamps.
pub const DEFAULT_GPU_LAYERS: u64 = 999;
pub const DEFAULT_PARALLEL: u64 = 1;
pub const DEFAULT_IDLE_SECONDS: f64 = 600.0;
/// The longest idle time a timer is armed for.
pub const MAX_IDLE: Duration = Duration::from_secs(30 * 24 * 3600);

/// How the server runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    pub ctx_size: u64,
    pub gpu_layers: u64,
    pub parallel: u64,
    pub idle_seconds: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            ctx_size: DEFAULT_CTX,
            gpu_layers: DEFAULT_GPU_LAYERS,
            parallel: DEFAULT_PARALLEL,
            idle_seconds: DEFAULT_IDLE_SECONDS,
        }
    }
}

impl Settings {
    /// How long an idle server keeps its model loaded.
    pub fn idle(&self) -> Duration {
        Duration::try_from_secs_f64(self.idle_seconds)
            .unwrap_or(MAX_IDLE)
            .min(MAX_IDLE)
    }
}

/// `load_settings` over the file's bytes.
pub fn parse(bytes: &[u8]) -> Settings {
    let mut settings = Settings::default();
    let Ok(text) = std::str::from_utf8(bytes) else {
        return settings;
    };
    let Ok(raw) = pyjson::loads(text) else {
        return settings;
    };
    if !raw.is_object() {
        return settings;
    }
    let count = |key: &str| match raw.get(key) {
        Some(PyValue::Int(digits)) if !digits.starts_with('-') => digits.parse::<u64>().ok(),
        _ => None,
    };
    if let Some(value) = count("ctx_size") {
        settings.ctx_size = value;
    }
    if let Some(value) = count("gpu_layers") {
        settings.gpu_layers = value;
    }
    if let Some(value) = count("parallel") {
        settings.parallel = value;
    }
    let idle = match raw.get("idle_seconds") {
        Some(PyValue::Int(digits)) => digits.parse::<f64>().ok(),
        Some(PyValue::Float(value)) => Some(*value),
        _ => None,
    };
    if let Some(value) = idle.filter(|value| *value > 0.0) {
        settings.idle_seconds = value;
    }
    settings
}

/// `load_settings`: the shared file, or every default.
pub fn load(paths: &LlamaPaths) -> Settings {
    std::fs::read(paths.settings_file())
        .map(|bytes| parse(&bytes))
        .unwrap_or_default()
}

/// The SHA-256 `manifest.json` records for `binary`, when `binary` is the
/// pinned install's own file; `None` otherwise.
///
/// LR3a: `binary` is a final path (links resolved, long names), so the
/// install folder is compared by its final path too: `llama_dir` is opened one
/// component at a time (a link to a share or a device is refused before it is
/// followed) and its final path read back, so a junction or a link on the way
/// to `~/.alelyon/llama`, or an 8.3 name in it, still names the same folder.
/// Both sides are compared as Windows compares names, without regard to case.
pub fn manifest_sha256(paths: &LlamaPaths, binary: &Path) -> Option<String> {
    let parent = binary.parent()?.to_string_lossy().into_owned();
    let install = final_dir(&paths.llama_dir)?;
    if comparable(&parent) != comparable(&install.to_string_lossy()) {
        return None;
    }
    let name = binary.file_name()?.to_str()?;
    let text = String::from_utf8(std::fs::read(paths.manifest_file()).ok()?).ok()?;
    let manifest = pyjson::loads(&text).ok()?;
    let sha = manifest.get("files")?.get(name)?.as_str()?;
    (sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit()))
        .then(|| sha.to_ascii_lowercase())
}

/// `dir`'s final path without its verbatim prefix, when it is a folder
/// reached through local links only.
#[cfg(windows)]
fn final_dir(dir: &Path) -> Option<std::path::PathBuf> {
    use crate::localfs::{LinkRule, open_walk};
    let walked = open_walk(dir, lattice_sys::fs::Access::Attributes, LinkRule::AnyLocal).ok()?;
    walked
        .is_dir
        .then(|| std::path::PathBuf::from(crate::workspace::shown_path(&walked.final_path)))
}

#[cfg(not(windows))]
fn final_dir(dir: &Path) -> Option<std::path::PathBuf> {
    std::fs::canonicalize(dir).ok().filter(|path| path.is_dir())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::TempDir;
    use std::path::PathBuf;

    /// LR3a's provenance: the binary is judged by its final path, so the
    /// install folder must be too. With `~/.alelyon/llama` a junction to the
    /// real install, the binary `find_binary` returns (the real folder's
    /// path) is still the pinned install's, and its recorded SHA-256 is
    /// read; the configured path spelled in another case is the same folder;
    /// a binary in another folder is still not the install's (the control).
    /// Mutant: `llama_dir` compared by its text again.
    #[cfg(windows)]
    #[test]
    fn the_install_folder_is_compared_by_its_final_path() {
        let dir = TempDir::new("llama-manifest-junction");
        let real = dir.path().join("real-llama");
        std::fs::create_dir_all(&real).unwrap();
        std::fs::write(real.join("llama-server.exe"), b"MZ").unwrap();
        let sha = "cd".repeat(32);
        std::fs::write(
            real.join("manifest.json"),
            format!("{{\"files\": {{\"llama-server.exe\": \"{sha}\"}}}}"),
        )
        .unwrap();
        let linked = dir.path().join("llama");
        std::fs::create_dir(&linked).unwrap();
        lattice_sys::fs::seam::create_junction(&linked, &real).unwrap();
        let paths = LlamaPaths {
            home: dir.path().to_path_buf(),
            llama_dir: linked.clone(),
            models_dir: dir.path().join("models"),
            binary: linked.join("llama-server.exe"),
            binary_from_env: false,
        };
        let binary = super::super::files::find_binary(&paths).unwrap();
        assert!(
            !binary.starts_with(&linked),
            "find_binary answers the final path: {}",
            binary.display()
        );
        assert_eq!(manifest_sha256(&paths, &binary), Some(sha.clone()));
        let upper = LlamaPaths {
            llama_dir: PathBuf::from(linked.to_string_lossy().to_uppercase()),
            ..paths.clone()
        };
        assert_eq!(manifest_sha256(&upper, &binary), Some(sha));
        let elsewhere = dir.path().join("other");
        std::fs::create_dir_all(&elsewhere).unwrap();
        assert_eq!(
            manifest_sha256(&paths, &elsewhere.join("llama-server.exe")),
            None
        );
    }

    #[test]
    fn settings_follow_pythons_validation_and_defaults() {
        assert_eq!(parse(b""), Settings::default());
        assert_eq!(parse(b"[]"), Settings::default());
        assert_eq!(
            parse(b"\xef\xbb\xbf{\"ctx_size\": 4096}"),
            Settings::default(),
            "read_text keeps a BOM; json.loads refuses it"
        );
        let all =
            parse(br#"{"ctx_size": 4096, "gpu_layers": 0, "parallel": 2, "idle_seconds": 1.5}"#);
        assert_eq!(
            all,
            Settings {
                ctx_size: 4096,
                gpu_layers: 0,
                parallel: 2,
                idle_seconds: 1.5
            }
        );
        let bad = parse(
            br#"{"ctx_size": -1, "gpu_layers": 1.5, "parallel": "2", "idle_seconds": 0, "model": "x"}"#,
        );
        assert_eq!(bad, Settings::default());
        assert_eq!(parse(br#"{"idle_seconds": -3}"#).idle_seconds, 600.0);
        assert_eq!(parse(br#"{"idle_seconds": 7}"#).idle_seconds, 7.0);
        assert_eq!(
            parse(br#"{"ctx_size": true, "idle_seconds": true}"#),
            Settings::default(),
            "a boolean is no number here"
        );
        assert_eq!(
            parse(br#"{"ctx_size": 99999999999999999999999}"#).ctx_size,
            DEFAULT_CTX
        );
        assert_eq!(
            parse(br#"{"ctx_size": 2048, "idle_seconds": NaN}"#),
            Settings::default()
        );
        assert_eq!(parse(br#"{"idle_seconds": 1e300}"#).idle(), MAX_IDLE);
        assert_eq!(
            parse(br#"{"idle_seconds": 0.25}"#).idle(),
            Duration::from_millis(250)
        );
    }

    #[test]
    fn the_manifest_names_the_pinned_binarys_sha256_and_nothing_else() {
        let dir = TempDir::new("llama-manifest");
        let paths = LlamaPaths {
            home: dir.path().to_path_buf(),
            llama_dir: dir.path().join("llama"),
            models_dir: dir.path().join("models"),
            binary: dir.path().join("llama").join("llama-server.exe"),
            binary_from_env: false,
        };
        std::fs::create_dir_all(&paths.llama_dir).unwrap();
        assert_eq!(manifest_sha256(&paths, &paths.binary), None, "no manifest");
        let sha = "AB".repeat(32);
        std::fs::write(
            paths.manifest_file(),
            format!("{{\"source\": \"x\", \"files\": {{\"llama-server.exe\": \"{sha}\", \"ggml.dll\": \"00\"}}}}"),
        )
        .unwrap();
        assert_eq!(
            manifest_sha256(&paths, &paths.binary),
            Some(sha.to_lowercase())
        );
        assert_eq!(
            manifest_sha256(
                &paths,
                &dir.path().join("elsewhere").join("llama-server.exe")
            ),
            None,
            "a binary from ALELYON_LLAMA_SERVER elsewhere is not the pinned install's"
        );
        assert_eq!(
            manifest_sha256(&paths, &paths.llama_dir.join("ggml.dll")),
            None,
            "not a SHA-256"
        );
    }
}
