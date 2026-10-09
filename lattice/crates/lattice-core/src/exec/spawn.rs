//! The environment a child gets, and the one way the chat core starts a child
//! (the chat core's spec §7.6 X5, X7, X8; §8.2 for git). The allowlist is a
//! port of Python's `SAFE_AGENT_ENV_NAMES` (`agent/session.py`), pinned by the
//! parity golden `agent/env_allowlist.json`.
//!
//! **X7.** A child's environment is exactly the names in
//! [`SAFE_AGENT_ENV_NAMES`], with the values this process has (read through the
//! injectable [`Env`]), with two changes for every child (commands, git, later
//! MCP servers):
//! - `PATH` loses its empty and relative entries, and every entry inside the
//!   workspace root or `<globals>` ([`filter_path`]);
//! - `NoDefaultCurrentDirectoryInExePath=1` is added
//!   ([`NATIVE_ADDITIONS`]), so neither `CreateProcess` nor `cmd.exe` looks in
//!   the current folder (the workspace) for a program a grandchild starts.
//!
//! Nothing else: no provider key, no `ALELYON_*`, no `LATTICE_*`. The block
//! replaces this process's environment entirely; nothing is inherited.
//!
//! **X5, X8.** [`spawn`] hands the block to `lattice_sys::process::spawn`,
//! which starts the program by its absolute path with an explicit handle list,
//! a `NUL` stdin and pipes, inside a Job Object that ends the whole tree.

use std::ffi::{OsStr, OsString};
use std::io;
use std::path::{Path, PathBuf};

use lattice_sys::fs::{DriveType, PathKind, drive_type, path_kind};
use lattice_sys::process::{Child, JobLimits, SpawnRequest};

use crate::env::Env;

/// Python's `SAFE_AGENT_ENV_NAMES`, in its order.
pub const SAFE_AGENT_ENV_NAMES: [&str; 14] = [
    "PATH",
    "SystemRoot",
    "windir",
    "COMSPEC",
    "PATHEXT",
    "USERPROFILE",
    "HOME",
    "TEMP",
    "TMP",
    "TMPDIR",
    "APPDATA",
    "LOCALAPPDATA",
    "LANG",
    "LC_ALL",
];

/// What the native Lattice adds to every child's environment.
pub const NATIVE_ADDITIONS: [(&str, &str); 1] = [("NoDefaultCurrentDirectoryInExePath", "1")];

/// A path as compared here: backslashes, no verbatim prefix, no trailing
/// separator, lower case.
pub(crate) fn comparable(text: &str) -> String {
    let unquoted = text.trim().trim_matches('"');
    let slashes = unquoted.replace('/', "\\");
    let plain = if let Some(rest) = slashes.strip_prefix(r"\\?\UNC\") {
        format!(r"\\{rest}")
    } else if let Some(rest) = slashes.strip_prefix(r"\\?\") {
        rest.to_owned()
    } else {
        slashes
    };
    plain.trim_end_matches('\\').to_lowercase()
}

/// The comparable forms of `path`: as written, and as the file system resolves
/// it (short names expanded, links followed) when it is on a local drive and
/// exists. A network path is never resolved, so nothing connects to it.
pub(crate) fn forms(path: &Path) -> Vec<String> {
    let mut out = vec![comparable(&path.to_string_lossy())];
    let local = path_kind(path) == PathKind::Drive && drive_type(path) != DriveType::Remote;
    if local && let Ok(real) = std::fs::canonicalize(path) {
        out.push(comparable(&real.to_string_lossy()));
    }
    out
}

/// Whether any form of `entry` is `root` or below it, for any form of `root`.
pub(crate) fn inside(entry: &[String], root: &[String]) -> bool {
    entry.iter().any(|entry| {
        root.iter().any(|root| {
            !root.is_empty() && (entry == root || entry.starts_with(&format!("{root}\\")))
        })
    })
}

/// `PATH` without empty or relative entries and without entries inside the
/// workspace root or `<globals>`, the rest in order and as written.
pub fn filter_path(value: &OsStr, workspace: Option<&Path>, globals: &Path) -> OsString {
    let text = value.to_string_lossy();
    let mut roots = vec![forms(globals)];
    if let Some(workspace) = workspace {
        roots.push(forms(workspace));
    }
    let kept: Vec<&str> = text
        .split(';')
        .filter(|entry| {
            let unquoted = entry.trim().trim_matches('"');
            if unquoted.is_empty() {
                return false;
            }
            let path = Path::new(unquoted);
            if !matches!(
                path_kind(path),
                PathKind::Drive | PathKind::Unc | PathKind::VolumeGuid
            ) {
                return false;
            }
            let entry_forms = forms(path);
            !roots.iter().any(|root| inside(&entry_forms, root))
        })
        .collect();
    OsString::from(kept.join(";"))
}

/// The environment block every child gets (X7).
pub fn child_environment(
    env: &dyn Env,
    workspace: Option<&Path>,
    globals: &Path,
) -> Vec<(OsString, OsString)> {
    let mut block = Vec::with_capacity(SAFE_AGENT_ENV_NAMES.len() + NATIVE_ADDITIONS.len());
    for name in SAFE_AGENT_ENV_NAMES {
        let Some(value) = env.var(name) else {
            continue;
        };
        let value = if name == "PATH" {
            filter_path(&value, workspace, globals)
        } else {
            value
        };
        block.push((OsString::from(name), value));
    }
    for (name, value) in NATIVE_ADDITIONS {
        block.push((OsString::from(name), OsString::from(value)));
    }
    block
}

/// A child to start: an absolute program, its argv (its own name first), and
/// the folder it runs in.
#[derive(Clone, Debug)]
pub struct ChildSpec {
    pub program: PathBuf,
    pub argv: Vec<OsString>,
    pub cwd: PathBuf,
    pub limits: JobLimits,
}

/// Start `spec` with the X7 environment built from `env`, the workspace root
/// and `<globals>`.
pub fn spawn(
    spec: &ChildSpec,
    env: &dyn Env,
    workspace: Option<&Path>,
    globals: &Path,
) -> io::Result<Child> {
    let block = child_environment(env, workspace, globals);
    lattice_sys::process::spawn(&SpawnRequest {
        program: &spec.program,
        argv: &spec.argv,
        cwd: &spec.cwd,
        env: &block,
        limits: spec.limits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::MapEnv;
    use crate::testkit::TempDir;

    fn names(block: &[(OsString, OsString)]) -> Vec<String> {
        block
            .iter()
            .map(|(name, _)| name.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn only_the_allowlisted_names_and_the_native_addition_reach_a_child() {
        let mut env = MapEnv::new()
            .with("PATH", r"C:\Windows\System32")
            .with("SystemRoot", r"C:\Windows")
            .with("TEMP", r"C:\Temp")
            .with("LANG", "en_US.UTF-8");
        // Assembled so no source holds a credential-shaped value.
        for name in [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "ALELYON_HOME",
            "ALELYON_SECRET",
            "LATTICE_PERF",
            "HF_TOKEN",
            "Path_extra",
        ] {
            env.set(name, format!("{name}-sentinel"));
        }
        let block = child_environment(&env, None, Path::new(r"C:\state\globals"));
        assert_eq!(
            names(&block),
            [
                "PATH",
                "SystemRoot",
                "TEMP",
                "LANG",
                "NoDefaultCurrentDirectoryInExePath"
            ]
        );
        assert!(
            !block
                .iter()
                .any(|(_, value)| value.to_string_lossy().contains("sentinel"))
        );
        assert_eq!(block[4].1, OsString::from("1"));
    }

    #[test]
    fn path_loses_empty_relative_workspace_and_state_entries() {
        let workspace = TempDir::new("spawn-ws");
        let globals = TempDir::new("spawn-globals");
        let ws = workspace.path().to_string_lossy().into_owned();
        let gl = globals.path().to_string_lossy().into_owned();
        let ws_upper = ws.to_uppercase();
        let entries = [
            r"C:\Windows\System32".to_owned(),
            String::new(),
            "bin".to_owned(),
            r".\tools".to_owned(),
            r"\rooted".to_owned(),
            "C:relative".to_owned(),
            ws.clone(),
            format!(r"{ws}\node_modules\.bin"),
            format!("\"{ws}\\scripts\""),
            format!(r"{ws_upper}\BIN\"),
            ws.replace('\\', "/"),
            format!("{gl}\\lattice_native"),
            gl.clone(),
            format!("{ws}2"),
            r"C:\Program Files\Git\cmd".to_owned(),
            r"\\server\share\bin".to_owned(),
            r"\\.\pipe\x".to_owned(),
        ];
        let joined = entries.join(";");
        let filtered = filter_path(OsStr::new(&joined), Some(workspace.path()), globals.path());
        assert_eq!(
            filtered.to_string_lossy(),
            [
                r"C:\Windows\System32".to_owned(),
                format!("{ws}2"),
                r"C:\Program Files\Git\cmd".to_owned(),
                r"\\server\share\bin".to_owned(),
            ]
            .join(";")
        );
    }

    #[cfg(windows)]
    #[test]
    fn an_entry_that_names_the_workspace_by_another_spelling_is_dropped() {
        // The same folder through a directory link inside the temporary root:
        // its canonical form is the workspace.
        let root = TempDir::new("spawn-alias");
        let workspace = root.path().join("project");
        std::fs::create_dir(&workspace).unwrap();
        std::fs::create_dir(workspace.join("bin")).unwrap();
        let alias = root.path().join("alias");
        std::os::windows::fs::symlink_dir(&workspace, &alias).unwrap();
        let entry = alias.join("bin");
        let globals = root.path().join("globals");
        let filtered = filter_path(
            entry.as_os_str(),
            Some(&std::fs::canonicalize(&workspace).unwrap()),
            &globals,
        );
        assert_eq!(filtered, OsString::new());
    }
}
