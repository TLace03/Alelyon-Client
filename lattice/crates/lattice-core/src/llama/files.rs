//! The managed server's files: where they are, which binary runs, which GGUF
//! models a name may mean, and the model Local uses (the chat core's spec
//! §22 LR3, LR4; ADR-0041 decisions 3, 4 and 6).
//!
//! Ported from `llama_server.py` (`alelyon_home`, `models_dir`, `llama_dir`,
//! `find_binary`, `list_models`, `resolve_model`) and
//! `local_model.selected_model`:
//! - **Where.** `~/.alelyon` is the home directory's (`USERPROFILE`, else
//!   `HOMEDRIVE` + `HOMEPATH`, as `Path.home()` reads them; `HOME` elsewhere),
//!   read through the injected [`Env`], so a test points every path at a
//!   temporary folder. `ALELYON_MODELS_DIR` and `ALELYON_LLAMA_SERVER` are
//!   used when set and not blank after Python's `strip()`.
//! - **The binary (LR3)** is `ALELYON_LLAMA_SERVER`, else
//!   `~/.alelyon/llama/llama-server.exe`. It is configured, never searched
//!   for. A path holding `ollama` in any case is refused, as Python refuses
//!   it. Stricter than Python (LR3): it must be absolute and on a drive of this
//!   machine, so starting it never reaches a network share, and on Windows an
//!   `.exe` (X2b): a `.cmd` or `.bat` wrapper would run through `cmd.exe`,
//!   where a model's file name is shell text.
//! - **Models (LR4)** are GGUF files in the models folder and one level below
//!   it. `mmproj*` files (vision projectors) are left out, and a file is
//!   listed only when it starts with the `GGUF` magic. A name is a file stem,
//!   matched exactly; a name given with `.gguf` means that file. Nothing is
//!   guessed: `qwen3` never means `qwen3-8b`, and case counts.
//! - **The model Local uses** is the `model` field of
//!   `<globals>/analyst_model.json` (at most 8,192 bytes), trimmed, or none.
//!   No variable overrides it and there is no default: a default named a
//!   download (ADR-0041 decision 6).
//!
//! - **Final paths (LR3a).** The binary, the models folder, each folder below
//!   it and each model are opened one component at a time (`localfs`); a
//!   link whose target is a share, a remote drive or a device path is refused
//!   before it is followed, so nothing connects to it, and the binary's rules
//!   are checked again on its final path, which is what starts. These walks
//!   pass through no reparse point but a link or a storage-only one: an app
//!   execution alias (whose data may name a `.cmd`, or an Ollama install's
//!   copy) or any other kind is refused at every component.
//!
//! Reads only: nothing here writes, starts or downloads anything.

use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::env::{self, Env};
use crate::local_model;
#[cfg(windows)]
use crate::localfs::{LinkRule, WalkError, open_walk_links_only};
use crate::py;
use crate::state::{self, Platform, StateRoot};
#[cfg(windows)]
use lattice_sys::fs::Access;

/// The variable that names the binary.
pub const BINARY_ENV: &str = "ALELYON_LLAMA_SERVER";
/// The variable that names the models folder.
pub const MODELS_ENV: &str = "ALELYON_MODELS_DIR";
/// The binary's name in the pinned install.
pub const BINARY_NAME: &str = if cfg!(windows) {
    "llama-server.exe"
} else {
    "llama-server"
};
/// What a GGUF file starts with.
pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const GGUF_SUFFIX: &str = ".gguf";
/// Vision projectors: companions of a model, never models.
const MMPROJ: &str = "mmproj";

/// Where the managed server's files are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlamaPaths {
    /// `~/.alelyon`.
    pub home: PathBuf,
    /// `~/.alelyon/llama`: the pinned install with its `manifest.json`, the
    /// shared `settings.json`, the probe records and the server logs.
    pub llama_dir: PathBuf,
    /// `ALELYON_MODELS_DIR`, else `~/.alelyon/models`.
    pub models_dir: PathBuf,
    /// The binary the core would start.
    pub binary: PathBuf,
    /// Whether [`LlamaPaths::binary`] came from `ALELYON_LLAMA_SERVER`.
    pub binary_from_env: bool,
}

impl LlamaPaths {
    /// The paths this environment names, on this platform.
    pub fn from_env(env: &dyn Env) -> Self {
        Self::from_env_on(env, Platform::host())
    }

    /// The paths this environment names, on `platform`.
    pub fn from_env_on(env: &dyn Env, platform: Platform) -> Self {
        let home = state::home_dir(env, platform).join(".alelyon");
        let llama_dir = home.join("llama");
        let models_dir = stripped(env, MODELS_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join("models"));
        let (binary, binary_from_env) = match stripped(env, BINARY_ENV) {
            Some(path) => (PathBuf::from(path), true),
            None => (llama_dir.join(BINARY_NAME), false),
        };
        Self {
            home,
            llama_dir,
            models_dir,
            binary,
            binary_from_env,
        }
    }

    /// `~/.alelyon/llama/settings.json`, shared with the Python side (LR5).
    pub fn settings_file(&self) -> PathBuf {
        self.llama_dir.join("settings.json")
    }

    /// `~/.alelyon/llama/manifest.json`, which Python's `install` writes.
    pub fn manifest_file(&self) -> PathBuf {
        self.llama_dir.join("manifest.json")
    }

    /// `~/.alelyon/llama/logs`: one log per model (LR6).
    pub fn logs_dir(&self) -> PathBuf {
        self.llama_dir.join("logs")
    }
}

/// `os.environ.get(name, "").strip()`, when that is not empty.
fn stripped(env: &dyn Env, name: &str) -> Option<String> {
    env::text(env, name)
        .map(|value| py::strip(&value).to_owned())
        .filter(|value| !value.is_empty())
}

/// Why the configured binary cannot be started.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryProblem {
    /// The path holds `ollama` (any case): an Ollama install's copy.
    Ollama,
    /// Not an absolute path.
    NotAbsolute,
    /// Not a program: its name does not end in `.exe` (a `.cmd` or `.bat`
    /// wrapper would run through `cmd.exe`, X2b).
    NotExe,
    /// Not on a drive of this machine (a share, a mapped network drive).
    NotLocal,
    /// No file there.
    Missing,
}

/// `_is_ollama_path`: does the path hold `ollama`, in any case?
pub fn is_ollama_path(path: &Path) -> bool {
    path.to_string_lossy().to_lowercase().contains("ollama")
}

/// Is `path` on a fixed, removable, RAM or optical drive of this machine?
/// Decided from the path's text and the drive's type, before anything opens
/// it, so a share is never contacted.
fn on_a_local_drive(path: &Path) -> bool {
    #[cfg(windows)]
    {
        use lattice_sys::fs::{DriveType, PathKind, drive_type, path_kind};
        path_kind(path) == PathKind::Drive
            && matches!(
                drive_type(path),
                DriveType::Fixed | DriveType::Removable | DriveType::RamDisk | DriveType::CdRom
            )
    }
    #[cfg(not(windows))]
    {
        let _ = path;
        true
    }
}

/// Does `path` name a Windows program by its name: `.exe` in any case, with
/// nothing after it (a trailing dot or space, which Windows drops, is
/// refused rather than trimmed)? Elsewhere any name is.
fn is_exe_name(path: &Path) -> bool {
    if !cfg!(windows) {
        return true;
    }
    path.file_name()
        .map(|name| name.to_string_lossy().to_ascii_lowercase())
        .is_some_and(|name| name.len() > ".exe".len() && name.ends_with(".exe"))
}

/// `find_binary`, with LR3's stricter rules: the binary to start, or why not.
/// The checks run in this order, so an Ollama path is named as one even when
/// it does not exist, and nothing is opened on a share.
pub fn find_binary(paths: &LlamaPaths) -> Result<PathBuf, BinaryProblem> {
    let candidate = &paths.binary;
    if is_ollama_path(candidate) {
        return Err(BinaryProblem::Ollama);
    }
    if !candidate.is_absolute() {
        return Err(BinaryProblem::NotAbsolute);
    }
    if !is_exe_name(candidate) {
        return Err(BinaryProblem::NotExe);
    }
    if !on_a_local_drive(candidate) {
        return Err(BinaryProblem::NotLocal);
    }
    final_binary(candidate)
}

/// LR3a: the binary opened one component at a time, a link to anything but
/// a local path refused before it is followed, and every rule checked again
/// on the final path read back from the handle (an Ollama install's copy
/// reached through a link or a junction is still an Ollama install's copy).
/// The final path, without its verbatim prefix, is what is started.
#[cfg(windows)]
fn final_binary(candidate: &Path) -> Result<PathBuf, BinaryProblem> {
    let walked = match open_walk_links_only(candidate, Access::Attributes, LinkRule::AnyLocal) {
        Ok(walked) => walked,
        Err(WalkError::NotFound | WalkError::Io(_)) => return Err(BinaryProblem::Missing),
        // An app execution alias or another reparse point that is not a link
        // (or a link whose data cannot be read) is not an .exe file.
        Err(WalkError::BadLink(_)) => return Err(BinaryProblem::NotExe),
        Err(_) => return Err(BinaryProblem::NotLocal),
    };
    if walked.is_dir {
        return Err(BinaryProblem::Missing);
    }
    let real = PathBuf::from(crate::workspace::shown_path(&walked.final_path));
    if is_ollama_path(&real) {
        return Err(BinaryProblem::Ollama);
    }
    if !is_exe_name(&real) {
        return Err(BinaryProblem::NotExe);
    }
    if !on_a_local_drive(&real) {
        return Err(BinaryProblem::NotLocal);
    }
    Ok(real)
}

#[cfg(not(windows))]
fn final_binary(candidate: &Path) -> Result<PathBuf, BinaryProblem> {
    if !candidate.is_file() {
        return Err(BinaryProblem::Missing);
    }
    Ok(candidate.to_path_buf())
}

/// LR3a: a folder reached through local links only (a link whose target is
/// a share, a remote drive or a device path is refused before it is
/// followed, so nothing connects to it).
#[cfg(windows)]
fn local_dir(path: &Path) -> bool {
    matches!(
        open_walk_links_only(path, Access::Attributes, LinkRule::AnyLocal),
        Ok(walked) if walked.is_dir
    )
}

#[cfg(not(windows))]
fn local_dir(path: &Path) -> bool {
    path.is_dir()
}

/// LR3a: a file reached through local links only, opened for reading.
#[cfg(windows)]
pub(crate) fn local_file(path: &Path) -> Option<fs::File> {
    match open_walk_links_only(path, Access::Read, LinkRule::AnyLocal) {
        Ok(walked) if !walked.is_dir => Some(walked.file),
        _ => None,
    }
}

#[cfg(not(windows))]
pub(crate) fn local_file(path: &Path) -> Option<fs::File> {
    if !path.is_file() {
        return None;
    }
    fs::File::open(path).ok()
}

/// Does an open file start with the `GGUF` magic?
fn starts_with_magic(file: &mut fs::File) -> bool {
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).is_ok() && magic == GGUF_MAGIC
}

/// A GGUF model the folder holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalModel {
    /// The file's stem: what a person picks.
    pub name: String,
    pub path: PathBuf,
    /// Bytes.
    pub size: u64,
}

/// `gguf_header.is_gguf`: does the file start with the `GGUF` magic? A file
/// that cannot be opened, or is shorter than four bytes, does not; nor does
/// one reached through a link to a share or a device (LR3a).
pub fn is_gguf(path: &Path) -> bool {
    local_file(path).is_some_and(|mut file| starts_with_magic(&mut file))
}

/// `PurePath.stem`: the name without its last suffix. A name whose only dot
/// leads it (`.gguf`), or ends it, has no suffix.
fn python_stem(name: &str) -> &str {
    match name.rfind('.') {
        Some(at) if at > 0 && at < name.len() - 1 => &name[..at],
        _ => name,
    }
}

/// Does `name` match the glob `*.gguf`? Without regard to case on Windows,
/// where Python's `Path.glob` matches that way; exactly elsewhere.
fn matches_gguf(name: &str) -> bool {
    if cfg!(windows) {
        name.to_lowercase().ends_with(GGUF_SUFFIX)
    } else {
        name.ends_with(GGUF_SUFFIX)
    }
}

/// The order `sorted()` gives Python paths: part by part, without regard to
/// case on Windows.
fn sort_key(path: &Path) -> Vec<String> {
    path.components()
        .map(|part| {
            let text = part.as_os_str().to_string_lossy();
            if cfg!(windows) {
                text.to_lowercase()
            } else {
                text.into_owned()
            }
        })
        .collect()
}

/// The entries of `dir` that match `*.gguf`, whatever they are (the glob's
/// last part matches files, folders and links alike).
fn matching_entries(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if matches_gguf(&entry.file_name().to_string_lossy()) {
            out.push(entry.path());
        }
    }
}

/// `sorted(base.glob(pattern))` for `*.gguf` (depth 0) or `*/*.gguf` (depth
/// 1, through the folders, and links to folders, directly below `base`).
fn glob_gguf(base: &Path, depth: u8) -> Vec<PathBuf> {
    let mut found = Vec::new();
    if depth == 0 {
        matching_entries(base, &mut found);
    } else if let Ok(entries) = fs::read_dir(base) {
        for entry in entries.flatten() {
            let path = entry.path();
            if local_dir(&path) {
                matching_entries(&path, &mut found);
            }
        }
    }
    found.sort_by_cached_key(|path| sort_key(path));
    found
}

/// `list_models`: the GGUF models in `dir` and one level below it, by name
/// without regard to case. When two files share a stem, the first found
/// wins: the folder's own before a subfolder's, then in path order.
///
/// LR3a: the folder, each folder below it and each file are reached through
/// local links only; a link (or a configured folder) whose target is a share,
/// a remote drive or a device path is passed over before it is followed.
pub fn list_models(dir: &Path) -> Vec<LocalModel> {
    if !local_dir(dir) {
        return Vec::new();
    }
    let mut seen = HashSet::new();
    let mut found = Vec::new();
    for path in glob_gguf(dir, 0).into_iter().chain(glob_gguf(dir, 1)) {
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        if name.to_lowercase().starts_with(MMPROJ) {
            continue;
        }
        let Some(mut file) = local_file(&path) else {
            continue;
        };
        if !starts_with_magic(&mut file) {
            continue;
        }
        let Ok(meta) = file.metadata() else {
            continue;
        };
        let stem = python_stem(&name).to_owned();
        if seen.insert(stem.clone()) {
            found.push(LocalModel {
                name: stem,
                path,
                size: meta.len(),
            });
        }
    }
    found.sort_by_cached_key(|model| model.name.to_lowercase());
    found
}

/// Why a name means no model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelProblem {
    /// No name: nothing is chosen.
    NoneChosen,
    /// No GGUF file has that stem.
    NotFound,
}

/// The stem a name asks for: trimmed, with a trailing `.gguf` (any case)
/// removed; `None` for a blank name.
fn wanted_stem(name: &str) -> Option<String> {
    let wanted = py::strip(name);
    if wanted.is_empty() {
        return None;
    }
    if wanted.to_lowercase().ends_with(GGUF_SUFFIX) {
        let keep = wanted.chars().count().saturating_sub(GGUF_SUFFIX.len());
        return Some(wanted.chars().take(keep).collect());
    }
    Some(wanted.to_owned())
}

/// `resolve_model` over models already listed.
pub fn resolve_in(name: &str, models: &[LocalModel]) -> Result<LocalModel, ModelProblem> {
    let stem = wanted_stem(name).ok_or(ModelProblem::NoneChosen)?;
    models
        .iter()
        .find(|model| model.name == stem)
        .cloned()
        .ok_or(ModelProblem::NotFound)
}

/// `resolve_model`: the model a name means, exactly, or why none.
pub fn resolve_model(name: &str, dir: &Path) -> Result<LocalModel, ModelProblem> {
    resolve_in(name, &list_models(dir))
}

/// The model Local uses: `analyst_model.json`'s `model`, or `""`.
pub fn selected_model(state: &StateRoot) -> String {
    local_model::stored_model(state).unwrap_or_default()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::env::MapEnv;
    use crate::testkit::TempDir;

    pub(crate) fn gguf(path: &Path) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        // The magic, version 3, no tensors, no keys: the smallest header.
        let mut bytes = GGUF_MAGIC.to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        fs::write(path, bytes).unwrap();
    }

    fn names(models: &[LocalModel]) -> Vec<&str> {
        models.iter().map(|model| model.name.as_str()).collect()
    }

    #[test]
    fn every_path_follows_the_home_folder_and_the_two_variables() {
        let env = MapEnv::new().with("USERPROFILE", r"D:\Profiles\x");
        let paths = LlamaPaths::from_env_on(&env, Platform::Windows);
        let home = PathBuf::from(r"D:\Profiles\x").join(".alelyon");
        assert_eq!(paths.home, home);
        assert_eq!(paths.llama_dir, home.join("llama"));
        assert_eq!(paths.models_dir, home.join("models"));
        assert_eq!(paths.binary, home.join("llama").join(BINARY_NAME));
        assert!(!paths.binary_from_env);
        assert_eq!(
            paths.settings_file(),
            home.join("llama").join("settings.json")
        );
        assert_eq!(paths.logs_dir(), home.join("llama").join("logs"));

        let env = env
            .with(MODELS_ENV, "  D:\\gguf  ")
            .with(BINARY_ENV, "\tE:\\llama\\llama-server.exe\n");
        let paths = LlamaPaths::from_env_on(&env, Platform::Windows);
        assert_eq!(
            paths.models_dir,
            PathBuf::from("D:\\gguf"),
            "trimmed as Python trims"
        );
        assert_eq!(paths.binary, PathBuf::from("E:\\llama\\llama-server.exe"));
        assert!(paths.binary_from_env);

        let blank = MapEnv::new()
            .with("USERPROFILE", r"D:\Profiles\x")
            .with(MODELS_ENV, "   ")
            .with(BINARY_ENV, " ");
        let paths = LlamaPaths::from_env_on(&blank, Platform::Windows);
        assert_eq!(
            paths.models_dir,
            home.join("models"),
            "a blank variable is unset"
        );
        assert!(!paths.binary_from_env);
    }

    #[test]
    fn no_ollama_variable_moves_any_path() {
        let env = MapEnv::new().with("USERPROFILE", r"D:\Profiles\x");
        let with_ollama = env
            .clone()
            .with("OLLAMA_MODELS", r"D:\ollama\models")
            .with("OLLAMA_HOST", "0.0.0.0")
            .with("OLLAMA_BASE_URL", "http://198.51.100.7:11434");
        assert_eq!(
            LlamaPaths::from_env_on(&env, Platform::Windows),
            LlamaPaths::from_env_on(&with_ollama, Platform::Windows)
        );
    }

    fn paths_with_binary(binary: PathBuf) -> LlamaPaths {
        LlamaPaths {
            home: PathBuf::new(),
            llama_dir: PathBuf::new(),
            models_dir: PathBuf::new(),
            binary,
            binary_from_env: true,
        }
    }

    #[test]
    fn the_binary_is_refused_from_an_ollama_install_a_relative_path_or_a_share() {
        let dir = TempDir::new("llama-binary");
        let real = dir.path().join(BINARY_NAME);
        fs::write(&real, b"MZ").unwrap();
        assert_eq!(find_binary(&paths_with_binary(real.clone())), Ok(real));

        let ollama = dir.path().join("Ollama").join("lib").join(BINARY_NAME);
        fs::create_dir_all(ollama.parent().unwrap()).unwrap();
        fs::write(&ollama, b"MZ").unwrap();
        assert_eq!(
            find_binary(&paths_with_binary(ollama)),
            Err(BinaryProblem::Ollama),
            "an existing file in an Ollama install is still refused"
        );
        assert_eq!(
            find_binary(&paths_with_binary(PathBuf::from(
                r"C:\OLLAMA-x\llama-server.exe"
            ))),
            Err(BinaryProblem::Ollama)
        );
        assert_eq!(
            find_binary(&paths_with_binary(PathBuf::from("llama-server.exe"))),
            Err(BinaryProblem::NotAbsolute)
        );
        assert_eq!(
            find_binary(&paths_with_binary(dir.path().join("absent.exe"))),
            Err(BinaryProblem::Missing)
        );
        let folder = dir.path().join("folder.exe");
        fs::create_dir(&folder).unwrap();
        assert_eq!(
            find_binary(&paths_with_binary(folder)),
            Err(BinaryProblem::Missing),
            "a folder is not the binary"
        );
    }

    /// X2b (spec §22.6): the binary must be an `.exe`; a `.cmd` or `.bat`
    /// wrapper (which `cmd.exe` would run, with a model's name live as shell
    /// text) is refused, even when it exists.
    /// Mutant: the `.exe` check dropped.
    #[cfg(windows)]
    #[test]
    fn the_binary_must_be_an_exe() {
        let dir = TempDir::new("llama-binary-exe");
        for name in [
            "run-server.cmd",
            "run-server.bat",
            "RUN.CMD",
            "llama-server",
            "llama-server.ps1",
            "llama-server.exe.cmd",
            "llama-server.exe.",
            "llama-server.com",
        ] {
            let path = dir.path().join(name);
            fs::write(&path, b"@exit /b 0\r\n").unwrap();
            assert_eq!(
                find_binary(&paths_with_binary(path)),
                Err(BinaryProblem::NotExe),
                "{name}"
            );
        }
        fs::create_dir(dir.path().join("ok")).unwrap();
        let upper = dir.path().join("ok").join("LLAMA-SERVER.EXE");
        fs::write(&upper, b"MZ").unwrap();
        assert_eq!(find_binary(&paths_with_binary(upper.clone())), Ok(upper));
    }

    #[cfg(windows)]
    #[test]
    fn a_binary_on_a_share_is_refused_by_its_text_before_anything_opens_it() {
        for path in [
            r"\\198.51.100.7\share\llama-server.exe",
            r"\\?\UNC\198.51.100.7\share\llama-server.exe",
        ] {
            assert_eq!(
                find_binary(&paths_with_binary(PathBuf::from(path))),
                Err(BinaryProblem::NotLocal),
                "{path}"
            );
        }
    }

    /// `\\?\GLOBALROOT\Device\…\<file>`: a device path that reaches a local
    /// file, the offline stand-in for a link to a place that is not a drive.
    #[cfg(windows)]
    fn device_path_of(path: &Path) -> PathBuf {
        let opened = lattice_sys::fs::open_no_follow(path, Access::Attributes).unwrap();
        let nt = lattice_sys::fs::seam::nt_path(&opened.file).unwrap();
        PathBuf::from(format!(r"\\?\GLOBALROOT{}", nt.display()))
    }

    /// A test folder's own final path, without its verbatim prefix.
    #[cfg(windows)]
    fn real_root(dir: &TempDir) -> PathBuf {
        let walked =
            crate::localfs::open_walk(dir.path(), Access::Attributes, LinkRule::AnyLocal).unwrap();
        PathBuf::from(crate::workspace::shown_path(&walked.final_path))
    }

    /// LR3a (spec §22.6): the binary's checks run on its final path. A
    /// junction into an Ollama install is an Ollama install, and a link whose
    /// target is a device path is refused before it is followed; a junction
    /// to a local binary elsewhere starts that binary, by its final path.
    /// Mutant: `final_binary` returns the configured path once it is a file.
    #[cfg(windows)]
    #[test]
    fn the_binary_is_judged_by_its_final_path() {
        use crate::localfs::record;
        let dir = TempDir::new("llama-binary-final");
        let root = real_root(&dir);
        let install = root.join("Programs").join("Ollama").join("lib");
        fs::create_dir_all(&install).unwrap();
        fs::write(install.join(BINARY_NAME), b"MZ").unwrap();
        let tools = root.join("tools");
        fs::create_dir_all(tools.join("llama")).unwrap();
        lattice_sys::fs::seam::create_junction(&tools.join("llama"), &install).unwrap();
        assert_eq!(
            find_binary(&paths_with_binary(tools.join("llama").join(BINARY_NAME))),
            Err(BinaryProblem::Ollama),
            "a junction into an Ollama install"
        );

        let elsewhere = root.join("builds").join("b1");
        fs::create_dir_all(&elsewhere).unwrap();
        let real = elsewhere.join(BINARY_NAME);
        fs::write(&real, b"MZ").unwrap();
        fs::create_dir_all(tools.join("current")).unwrap();
        lattice_sys::fs::seam::create_junction(&tools.join("current"), &elsewhere).unwrap();
        assert_eq!(
            find_binary(&paths_with_binary(tools.join("current").join(BINARY_NAME))),
            Ok(real.clone()),
            "a local junction is followed, and its target is what starts"
        );

        let device = tools.join("device").join(BINARY_NAME);
        fs::create_dir_all(device.parent().unwrap()).unwrap();
        std::os::windows::fs::symlink_file(device_path_of(&real), &device).unwrap();
        let _ = record::take();
        assert_eq!(
            find_binary(&paths_with_binary(device)),
            Err(BinaryProblem::NotLocal),
            "a link to a device path"
        );
        assert!(
            !record::take()
                .iter()
                .any(|path| path.to_string_lossy().contains("GLOBALROOT")),
            "the device path was never opened"
        );
    }

    /// LR3a: a binary link to a share is refused before it is followed: no
    /// open of the share's path, and no wait for a connection.
    #[cfg(windows)]
    #[test]
    fn a_binary_link_to_a_share_is_refused_before_it_is_followed() {
        use crate::localfs::record;
        let dir = TempDir::new("llama-binary-unc");
        let link = dir.path().join(BINARY_NAME);
        std::os::windows::fs::symlink_file(r"\\198.51.100.7\x\llama-server.exe", &link).unwrap();
        let started = std::time::Instant::now();
        let _ = record::take();
        assert_eq!(
            find_binary(&paths_with_binary(link)),
            Err(BinaryProblem::NotLocal)
        );
        let opened = record::take();
        assert!(
            !opened
                .iter()
                .any(|path| path.to_string_lossy().contains("198.51.100.7")),
            "{opened:?}"
        );
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
    }

    /// X2b, LR3a (the verifier's probe, phaseHA/logs/VERIFY-probe-appexeclink):
    /// an app execution alias named `llama-server.exe` is refused whatever
    /// its data names (a `.cmd` wrapper, or an Ollama install's copy), and
    /// so is a symlink to one; a symlink to a plain binary is still
    /// accepted (the positive control).
    /// Mutants: `link_target` walks through an alias as it walks through
    /// other reparse points; `final_binary` walks with `open_walk` (which
    /// passes through an unknown reparse point).
    #[cfg(windows)]
    #[test]
    fn an_app_execution_alias_is_never_the_binary() {
        use lattice_sys::fs::seam::create_app_exec_link;
        let dir = TempDir::new("llama-binary-alias");
        let root = real_root(&dir);
        let wrapper = root.join("wrapper.cmd");
        fs::write(
            &wrapper,
            b"@echo off
",
        )
        .unwrap();
        let ollama = root.join("Ollama").join(BINARY_NAME);
        fs::create_dir_all(ollama.parent().unwrap()).unwrap();
        fs::write(&ollama, b"MZ").unwrap();
        let bin = root.join("bin");
        let obin = root.join("obin");
        fs::create_dir_all(&bin).unwrap();
        fs::create_dir_all(&obin).unwrap();
        create_app_exec_link(&bin.join(BINARY_NAME), &wrapper).unwrap();
        create_app_exec_link(&obin.join(BINARY_NAME), &ollama).unwrap();
        assert_eq!(
            find_binary(&paths_with_binary(bin.join(BINARY_NAME))),
            Err(BinaryProblem::NotExe),
            "an alias to a .cmd"
        );
        assert_eq!(
            find_binary(&paths_with_binary(obin.join(BINARY_NAME))),
            Err(BinaryProblem::NotExe),
            "an alias to an Ollama install's copy"
        );
        let via = root.join("via").join(BINARY_NAME);
        fs::create_dir_all(via.parent().unwrap()).unwrap();
        std::os::windows::fs::symlink_file(bin.join(BINARY_NAME), &via).unwrap();
        assert_eq!(
            find_binary(&paths_with_binary(via)),
            Err(BinaryProblem::NotExe),
            "a symlink to an alias"
        );
        let real = root.join("real").join(BINARY_NAME);
        fs::create_dir_all(real.parent().unwrap()).unwrap();
        fs::write(&real, b"MZ").unwrap();
        let to_real = root.join("link").join(BINARY_NAME);
        fs::create_dir_all(to_real.parent().unwrap()).unwrap();
        std::os::windows::fs::symlink_file(&real, &to_real).unwrap();
        assert_eq!(find_binary(&paths_with_binary(to_real)), Ok(real));
        // A reparse point of a tag no filter handles is no binary either.
        let odd = root.join("odd").join(BINARY_NAME);
        fs::create_dir_all(odd.parent().unwrap()).unwrap();
        lattice_sys::fs::seam::create_reparse_file(&odd, 0x8000_00F7, b"lattice").unwrap();
        assert_eq!(
            find_binary(&paths_with_binary(odd)),
            Err(BinaryProblem::NotExe),
            "an unknown reparse point"
        );
        // A model that is an alias is never opened as one.
        let models = root.join("models");
        gguf(&models.join("plain.gguf"));
        create_app_exec_link(&models.join("alias.gguf"), &models.join("plain.gguf")).unwrap();
        assert!(!is_gguf(&models.join("alias.gguf")));
        assert_eq!(names(&list_models(&models)), ["plain"]);
    }

    /// LR3a: the models folder, the folders below it and each model are
    /// reached through local links only. A model linked to a device path is
    /// passed over before it is followed; a local junction below the folder
    /// is still listed (the positive control).
    /// Mutant: `local_file` opens with `File::open`, following any link (and
    /// `gguf::read_header` back on `File::open`: the last assertions). (A
    /// `local_dir` that follows links survives here: every file below a folder
    /// so linked is still refused by `local_file`, so the listing is the same.)
    #[cfg(windows)]
    #[test]
    fn a_model_linked_off_the_drive_is_passed_over() {
        let dir = TempDir::new("llama-models-links");
        let root = dir.path().join("models");
        gguf(&root.join("plain.gguf"));
        let away = dir.path().join("away");
        gguf(&away.join("joined.gguf"));
        fs::create_dir_all(root.join("linked")).unwrap();
        lattice_sys::fs::seam::create_junction(&root.join("linked"), &away).unwrap();
        let outside = dir.path().join("outside.gguf");
        gguf(&outside);
        std::os::windows::fs::symlink_file(device_path_of(&outside), root.join("device.gguf"))
            .unwrap();
        let hidden = dir.path().join("hidden");
        gguf(&hidden.join("behind.gguf"));
        std::os::windows::fs::symlink_dir(device_path_of(&hidden), root.join("devdir")).unwrap();
        assert_eq!(names(&list_models(&root)), ["joined", "plain"]);
        assert!(!is_gguf(&root.join("device.gguf")));
        assert!(is_gguf(&root.join("plain.gguf")));
        // The header reader opens a model the same way (it used File::open,
        // which followed the link to the device path).
        assert!(super::super::gguf::read_header(&root.join("plain.gguf")).is_ok());
        assert_eq!(
            super::super::gguf::read_header(&root.join("device.gguf")).err(),
            Some(super::super::gguf::GgufError::Io)
        );
    }

    /// LR3a: a models folder on a share, and a model linked to a share, are
    /// never opened: nothing connects to the share.
    #[cfg(windows)]
    #[test]
    fn a_models_folder_or_model_on_a_share_is_never_opened() {
        use crate::localfs::record;
        let dir = TempDir::new("llama-models-unc");
        let root = dir.path().join("models");
        gguf(&root.join("plain.gguf"));
        std::os::windows::fs::symlink_file(r"\\198.51.100.7\x\m.gguf", root.join("share.gguf"))
            .unwrap();
        let started = std::time::Instant::now();
        let _ = record::take();
        assert_eq!(names(&list_models(&root)), ["plain"]);
        assert!(list_models(Path::new(r"\\198.51.100.7\x\models")).is_empty());
        let opened = record::take();
        assert!(
            !opened
                .iter()
                .any(|path| path.to_string_lossy().contains("198.51.100.7")),
            "{opened:?}"
        );
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
    }

    #[test]
    fn models_are_gguf_files_at_depth_zero_or_one_without_projectors() {
        let dir = TempDir::new("llama-models");
        let root = dir.path();
        gguf(&root.join("qwen3-8b.gguf"));
        gguf(&root.join("Alpha.gguf"));
        gguf(&root.join("sub").join("beta-q4.gguf"));
        gguf(&root.join("sub").join("deeper").join("gamma.gguf"));
        gguf(&root.join("mmproj-qwen3.gguf"));
        gguf(&root.join("MMPROJ-big.gguf"));
        fs::write(root.join("fake.gguf"), b"not a model").unwrap();
        fs::write(root.join("tiny.gguf"), b"GG").unwrap();
        fs::write(root.join("notes.txt"), b"GGUF").unwrap();
        fs::create_dir_all(root.join("folder.gguf")).unwrap();
        let models = list_models(root);
        assert_eq!(names(&models), ["Alpha", "beta-q4", "qwen3-8b"]);
        let beta = &models[1];
        assert_eq!(beta.path, root.join("sub").join("beta-q4.gguf"));
        assert_eq!(beta.size, 24);
        assert!(list_models(&root.join("absent")).is_empty());
    }

    #[test]
    fn a_stem_in_the_folder_wins_over_the_same_stem_below_it() {
        let dir = TempDir::new("llama-models-dup");
        let root = dir.path();
        gguf(&root.join("b").join("same.gguf"));
        gguf(&root.join("a").join("same.gguf"));
        gguf(&root.join("same.gguf"));
        let models = list_models(root);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].path, root.join("same.gguf"));
        fs::remove_file(root.join("same.gguf")).unwrap();
        let models = list_models(root);
        assert_eq!(
            models[0].path,
            root.join("a").join("same.gguf"),
            "then path order"
        );
    }

    #[test]
    fn a_stem_is_pythons_stem() {
        assert_eq!(python_stem("a.b.gguf"), "a.b");
        assert_eq!(python_stem(".gguf"), ".gguf");
        assert_eq!(python_stem("x."), "x.");
        assert_eq!(python_stem("x"), "x");
    }

    /// LF6 (spec §22.2): a model name is never guessed.
    #[test]
    fn a_model_name_is_never_guessed() {
        let dir = TempDir::new("llama-resolve");
        let root = dir.path();
        gguf(&root.join("qwen3-8b.gguf"));
        gguf(&root.join("mmproj-qwen3.gguf"));
        gguf(&root.join("sub").join("Mixed-Case.gguf"));
        assert_eq!(resolve_model("qwen3-8b", root).unwrap().name, "qwen3-8b");
        assert_eq!(
            resolve_model("  qwen3-8b.gguf ", root).unwrap().name,
            "qwen3-8b",
            "a name given with .gguf means that file"
        );
        assert_eq!(
            resolve_model("qwen3-8b.GGUF", root).unwrap().name,
            "qwen3-8b"
        );
        for (name, problem) in [
            ("qwen3", ModelProblem::NotFound),
            ("qwen3-8", ModelProblem::NotFound),
            ("Qwen3-8B", ModelProblem::NotFound),
            ("mixed-case", ModelProblem::NotFound),
            ("mmproj-qwen3", ModelProblem::NotFound),
            ("mmproj-qwen3.gguf", ModelProblem::NotFound),
            ("sub/Mixed-Case", ModelProblem::NotFound),
            ("qwen3:8b", ModelProblem::NotFound),
            ("", ModelProblem::NoneChosen),
            ("   ", ModelProblem::NoneChosen),
        ] {
            assert_eq!(resolve_model(name, root), Err(problem), "{name:?}");
        }
        assert_eq!(
            resolve_model("Mixed-Case", root).unwrap().name,
            "Mixed-Case"
        );
    }

    #[test]
    fn the_selected_model_is_the_files_with_no_variable_and_no_default() {
        let dir = TempDir::new("llama-selected");
        let state = StateRoot::at(dir.path());
        assert_eq!(selected_model(&state), "", "no file: nothing is chosen");
        fs::create_dir_all(&state.globals).unwrap();
        fs::write(
            state.globals.join("analyst_model.json"),
            br#"{"model": "  qwen3-8b "}"#,
        )
        .unwrap();
        assert_eq!(selected_model(&state), "qwen3-8b");
        fs::write(state.globals.join("analyst_model.json"), b"{\"model\": 3}").unwrap();
        assert_eq!(selected_model(&state), "");
    }
}
