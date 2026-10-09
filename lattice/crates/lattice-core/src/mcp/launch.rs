//! How an MCP server starts (the chat core's spec §12, "Spawn"; X2, X5,
//! X7, X8, T5). Not a port.
//!
//! - **The program.** A bare name is resolved by X2's rules
//!   ([`resolve_program`]): Lattice's own `PATH`, never an entry inside the
//!   folder or `<globals>`, never an app execution alias. A path is used as
//!   written (a relative one against the server's folder). A `.cmd` or `.bat`
//!   (`npx` is one) runs through `%SystemRoot%\System32\cmd.exe /d /v:off /s
//!   /c`, only when nothing in its line has a meaning to `cmd.exe`
//!   (`lattice_sys::process::batch_command_line`); a PowerShell script is
//!   refused (its command should be `powershell.exe -File`).
//! - **The environment** is X7's block (no provider key, no `ALELYON_*`, no
//!   `LATTICE_*`; `PATH` without the folder's or `<globals>`' entries), plus
//!   each variable the entry names: from the reader's own file, its value as
//!   written (it may replace a name of the block); from a folder's file, the
//!   value Lattice's own environment has (a folder never replaces a name of
//!   the block).
//! - **The folder** it runs in: the entry's `cwd`, else the folder whose file
//!   declared it, else the reader's home.
//! - **The Job** ([`LIMITS`]) holds the server and everything it starts;
//!   dropping the child ends them all.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use lattice_sys::process::{
    BatchRequest, Child, JobLimits, SpawnRequest, spawn_batch_with_input, spawn_with_input,
};

use super::config::{Scope, ServerEntry};
use crate::env::Env;
use crate::exec::resolve::{Ineligible, resolve_program};
use crate::exec::spawn::child_environment;

/// A server's Job: more processes and memory than a command's, since a
/// server may run a runtime and its workers (a Node or Python process tree).
pub const LIMITS: JobLimits = JobLimits {
    active_processes: 128,
    job_memory: 8 * 1024 * 1024 * 1024,
};

/// How the program starts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum How {
    /// A program, started directly with this argv (its own name first).
    Direct {
        program: PathBuf,
        argv: Vec<OsString>,
    },
    /// A batch file, through `cmd.exe`.
    Batch {
        cmd: PathBuf,
        script: PathBuf,
        args: Vec<OsString>,
    },
}

/// A server's start, planned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub how: How,
    pub cwd: PathBuf,
    pub env: Vec<(OsString, OsString)>,
}

impl Plan {
    /// The program that runs, as the reader is shown it.
    pub fn program(&self) -> String {
        match &self.how {
            How::Direct { program, .. } => shown(program),
            How::Batch { script, .. } => format!("{} (through cmd.exe)", shown(script)),
        }
    }

    /// Start it, with stdin and stdout as pipes.
    pub fn start(&self) -> std::io::Result<Child> {
        match &self.how {
            How::Direct { program, argv } => spawn_with_input(&SpawnRequest {
                program,
                argv,
                cwd: &self.cwd,
                env: &self.env,
                limits: LIMITS,
            }),
            How::Batch { cmd, script, args } => spawn_batch_with_input(&BatchRequest {
                cmd,
                script,
                args,
                cwd: &self.cwd,
                env: &self.env,
                limits: LIMITS,
            }),
        }
    }
}

/// A path as the reader reads it: without the `\\?\` prefix.
pub fn shown(path: &Path) -> String {
    let text = path.display().to_string();
    text.strip_prefix(r"\\?\")
        .map(str::to_owned)
        .unwrap_or(text)
}

fn has_extension(text: &str, extension: &str) -> bool {
    let lower = text.trim_end_matches(['.', ' ']).to_ascii_lowercase();
    lower.len() > extension.len() && lower.ends_with(extension)
}

/// The base a relative path of this entry is read against.
fn base(entry: &ServerEntry, folder: Option<&Path>, env: &dyn Env) -> Option<PathBuf> {
    match (&entry.key.scope, folder) {
        (Scope::Folder { .. }, Some(folder)) => Some(folder.to_path_buf()),
        _ => env.var("USERPROFILE").map(PathBuf::from),
    }
}

/// The folder the server runs in.
fn cwd_of(entry: &ServerEntry, folder: Option<&Path>, env: &dyn Env) -> Result<PathBuf, String> {
    let base = base(entry, folder, env);
    let cwd = match &entry.cwd {
        Some(cwd) if Path::new(cwd).is_absolute() => PathBuf::from(cwd),
        Some(cwd) => base
            .ok_or_else(|| {
                format!("Its folder {cwd} is relative, and there is nothing to read it against.")
            })?
            .join(cwd),
        None => base.ok_or_else(|| "Give it a folder to run in (cwd).".to_owned())?,
    };
    if !cwd.is_dir() {
        return Err(format!(
            "The folder it runs in, {}, does not exist.",
            shown(&cwd)
        ));
    }
    Ok(cwd)
}

fn cmd_exe(env: &dyn Env) -> Result<PathBuf, String> {
    let root = env
        .var("SystemRoot")
        .ok_or_else(|| "SystemRoot is not set, so cmd.exe cannot be found.".to_owned())?;
    Ok(PathBuf::from(root).join("System32").join("cmd.exe"))
}

/// The program `command` names, and whether it is a batch file.
fn program_of(
    command: &str,
    cwd: &Path,
    env: &dyn Env,
    workspace: Option<&Path>,
    globals: &Path,
) -> Result<(PathBuf, bool), String> {
    if has_extension(command, ".ps1") {
        return Err("A PowerShell script runs through powershell.exe -File: make that its command and the script its first argument.".to_owned());
    }
    let batch = has_extension(command, ".cmd") || has_extension(command, ".bat");
    let as_path = Path::new(command);
    if as_path.is_absolute() || command.contains(['/', '\\']) {
        let path = if as_path.is_absolute() {
            as_path.to_path_buf()
        } else {
            cwd.join(as_path)
        };
        if !path.is_file() {
            return Err(format!("There is no program at {}.", shown(&path)));
        }
        return Ok((path, batch));
    }
    // A bare name; one written with its batch extension is looked for by
    // its stem and must come back as that batch file.
    let stem = if batch {
        command
            .trim_end_matches(['.', ' '])
            .rsplit_once('.')
            .map_or(command, |(stem, _)| stem)
    } else {
        command
    };
    match resolve_program(stem, env, workspace, globals) {
        Ok(resolved) if !batch => Ok((resolved.real, false)),
        Err(Ineligible::BatchFile(found)) if found.is_absolute() => {
            let wanted = !batch
                || found
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name.eq_ignore_ascii_case(command.trim_end_matches(['.', ' ']))
                    });
            if wanted {
                Ok((found, true))
            } else {
                Err(format!("{command} was not found on Lattice's PATH."))
            }
        }
        Ok(_) => Err(format!("{command} was not found on Lattice's PATH.")),
        Err(Ineligible::Script(_)) => Err(format!(
            "{command} runs a script; make its interpreter the command and the script its argument."
        )),
        Err(Ineligible::NotFound) => Err(format!(
            "No program called {command} was found on Lattice's PATH (outside the folder)."
        )),
        Err(other) => Err(other.sentence().to_owned()),
    }
}

/// The environment: X7's block and the variables the entry names.
fn environment_of(
    entry: &ServerEntry,
    env: &dyn Env,
    workspace: Option<&Path>,
    globals: &Path,
) -> Vec<(OsString, OsString)> {
    let mut block = child_environment(env, workspace, globals);
    let folder = matches!(entry.key.scope, Scope::Folder { .. });
    for (name, value) in &entry.env {
        let value = match value {
            Some(value) => Some(OsString::from(value)),
            None => env.var(name),
        };
        let Some(value) = value else {
            continue;
        };
        let at = block
            .iter()
            .position(|(have, _)| have.to_string_lossy().eq_ignore_ascii_case(name));
        match at {
            // A folder never replaces a name of X7's block (its PATH filter).
            Some(_) if folder => {}
            Some(at) => block[at] = (OsString::from(name), value),
            None => block.push((OsString::from(name), value)),
        }
    }
    block
}

/// Plan the start of `entry`. `folder` is the folder whose file declared it
/// (`None` for the reader's own); `workspace` is the attached folder whose
/// `PATH` entries X2 and X7 leave out.
pub fn plan(
    entry: &ServerEntry,
    folder: Option<&Path>,
    env: &dyn Env,
    workspace: Option<&Path>,
    globals: &Path,
) -> Result<Plan, String> {
    let cwd = cwd_of(entry, folder, env)?;
    let (program, batch) = program_of(&entry.command, &cwd, env, workspace, globals)?;
    let args: Vec<OsString> = entry.args.iter().map(OsString::from).collect();
    let how = if batch {
        How::Batch {
            cmd: cmd_exe(env)?,
            script: program,
            args,
        }
    } else {
        let mut argv = vec![program.clone().into_os_string()];
        argv.extend(args);
        How::Direct { program, argv }
    };
    Ok(Plan {
        how,
        cwd,
        env: environment_of(entry, env, workspace, globals),
    })
}
