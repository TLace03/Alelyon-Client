//! Which program a bare command name runs (the chat core's spec §7.6 X2,
//! §8.2: the git runner resolves `git.exe` the same way). Not a port: the
//! Python agent host starts known CLIs by name; this is the native rule.
//!
//! [`resolve_program`] searches the `PATH` of Lattice's own environment, entry
//! by entry, and answers an absolute path or why the name is not eligible:
//! - Empty and relative entries are skipped, and so are entries inside the
//!   workspace root or `<globals>` (a program planted in the folder the agent
//!   works in is never chosen) and entries that are not on a local drive (a
//!   share is never probed, so nothing connects to it).
//! - In each entry, the name is tried with each `PATHEXT` extension in order.
//!   `<name>.exe` or `<name>.com` is the program. A `.bat` or `.cmd` that comes
//!   first is **ineligible**: Windows runs a batch file through `cmd.exe`,
//!   which is a shell; so are other script hosts' files, and a `<name>.ps1` in
//!   the same entry (PowerShell would run it). `npm`, `npx` and `yarn` are
//!   `.cmd` shims, so they always ask.
//! - A name that already ends in `.exe` or `.com` is looked for as it is; one
//!   ending in `.bat`, `.cmd` or `.ps1` is ineligible at once.
//! - The program found is opened one component at a time (`localfs`): every
//!   link of the chain, in the `PATH` entry's folders and at the program
//!   itself, is followed only when its target is a local path, and refused
//!   before it is followed otherwise (X2c). Its final path is then checked
//!   again by X2's own rules: a file, on a local drive, not inside the
//!   workspace or `<globals>`. A `PATH` entry reached through a link off the
//!   drive is skipped before anything below it is probed. No reparse point
//!   but a link or a storage-only one (WOF, dedup) is passed through: an app
//!   execution alias, at the name itself or anywhere in the chain, makes the
//!   name `Unsafe` (its data could name a `.cmd`, or any program).
//!
//! A bare name is never handed to `CreateProcessW` to search for itself (that
//! search includes the current folder): the spawn takes this absolute path.

use std::path::{Path, PathBuf};

use lattice_sys::fs::{Access, DriveType, PathKind, drive_type, open_no_follow, path_kind};

use super::spawn::{forms, inside};
use crate::env::Env;
use crate::localfs::{LinkRule, open_walk_links_only};

/// What `PATHEXT` is when the environment does not say (Windows' default).
const DEFAULT_PATHEXT: &str = ".COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC";

/// A program a name resolved to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolved {
    /// As found in its `PATH` entry.
    pub path: PathBuf,
    /// Its final path, read from its handle (links followed, long names).
    pub real: PathBuf,
}

/// Why a name cannot be run directly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ineligible {
    /// Not a bare program name (a path, a drive, a wildcard, a trailing dot).
    NotABareName,
    /// A `.bat` or `.cmd` file comes first: it runs through `cmd.exe`.
    BatchFile(PathBuf),
    /// A PowerShell script or another script host's file comes first.
    Script(PathBuf),
    /// Nothing of that name in a usable `PATH` entry.
    NotFound,
    /// Found, but it is not a local file outside the workspace and Lattice's
    /// state (a link elsewhere, a folder, a file that cannot be opened).
    Unsafe(PathBuf),
}

impl Ineligible {
    /// One sentence for the approval card.
    pub fn sentence(&self) -> &'static str {
        match self {
            Self::NotABareName => "Only a program's bare name can be allowed to run directly.",
            Self::BatchFile(_) => {
                "That name runs a batch file through cmd.exe, a shell, so it always asks."
            }
            Self::Script(_) => "That name runs a script, so it always asks.",
            Self::NotFound => "No program of that name was found outside this folder.",
            Self::Unsafe(_) => "That program is not a local file outside this folder.",
        }
    }
}

fn bare(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && !name.ends_with(['.', ' '])
        && !name.starts_with(' ')
        && !name.chars().any(|c| {
            matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') || c.is_control()
        })
}

fn has_extension(name: &str, extension: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.len() > extension.len() && lower.ends_with(extension)
}

/// What a probe of one candidate path found.
enum Probe {
    Absent,
    /// A file, or a link to a local path.
    Present,
    /// A link to a share or a device: never followed.
    Unsafe,
}

/// Look at `candidate` without following a link to anything but a local path.
fn probe(candidate: &Path) -> Probe {
    let Ok(opened) = open_no_follow(candidate, Access::Attributes) else {
        return Probe::Absent;
    };
    if opened.is_dir && opened.link.is_none() {
        return Probe::Absent;
    }
    match opened.link {
        None => Probe::Present,
        Some(link) => match link.target {
            Some(target) if link.relative => {
                let joined = candidate.parent().unwrap_or(Path::new("")).join(target);
                if path_kind(&joined) == PathKind::Drive {
                    Probe::Present
                } else {
                    Probe::Unsafe
                }
            }
            Some(target)
                if path_kind(&target) == PathKind::Drive
                    && drive_type(&target) != DriveType::Remote =>
            {
                Probe::Present
            }
            _ => Probe::Unsafe,
        },
    }
}

/// Open the candidate one component at a time, following a link only when
/// its target is local (X2c: every hop of the chain, not only the first, is
/// read before it is followed), read its final path, and refuse anything that
/// is not a file on a local drive outside the workspace and `<globals>`.
fn checked(candidate: PathBuf, roots: &[Vec<String>]) -> Result<Resolved, Ineligible> {
    let unsafe_one = || Ineligible::Unsafe(candidate.clone());
    let walked = open_walk_links_only(&candidate, Access::Attributes, LinkRule::AnyLocal)
        .map_err(|_| unsafe_one())?;
    if walked.is_dir {
        return Err(unsafe_one());
    }
    let real = walked.final_path;
    if path_kind(&real) != PathKind::Drive || drive_type(&real) == DriveType::Remote {
        return Err(unsafe_one());
    }
    let real_forms = forms(&real);
    if roots.iter().any(|root| inside(&real_forms, root)) {
        return Err(unsafe_one());
    }
    Ok(Resolved {
        path: candidate,
        real,
    })
}

/// Is a `PATH` entry a folder reached through local links only? A link on
/// the way whose target is a share, a remote drive or a device path is
/// refused before it is followed, so nothing below it is probed (X2c).
fn local_folder(folder: &Path) -> bool {
    matches!(
        open_walk_links_only(folder, Access::Attributes, LinkRule::AnyLocal),
        Ok(walked) if walked.is_dir
    )
}

/// Resolve `name` by X2's rules (see the module header).
pub fn resolve_program(
    name: &str,
    env: &dyn Env,
    workspace: Option<&Path>,
    globals: &Path,
) -> Result<Resolved, Ineligible> {
    if !bare(name) {
        return Err(Ineligible::NotABareName);
    }
    for script in [".bat", ".cmd"] {
        if has_extension(name, script) {
            return Err(Ineligible::BatchFile(PathBuf::from(name)));
        }
    }
    if has_extension(name, ".ps1") {
        return Err(Ineligible::Script(PathBuf::from(name)));
    }
    let direct = has_extension(name, ".exe") || has_extension(name, ".com");
    let pathext: Vec<String> = env
        .var("PATHEXT")
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| DEFAULT_PATHEXT.to_owned())
        .split(';')
        .map(|extension| extension.trim().to_ascii_lowercase())
        .filter(|extension| extension.starts_with('.') && extension.len() > 1)
        .collect();
    let mut roots = vec![forms(globals)];
    if let Some(workspace) = workspace {
        roots.push(forms(workspace));
    }
    let path = env.var("PATH").unwrap_or_default();
    for entry in path.to_string_lossy().split(';') {
        let entry = entry.trim().trim_matches('"');
        if entry.is_empty() {
            continue;
        }
        let folder = Path::new(entry);
        if path_kind(folder) != PathKind::Drive || drive_type(folder) == DriveType::Remote {
            continue;
        }
        if roots.iter().any(|root| inside(&forms(folder), root)) {
            continue;
        }
        if !local_folder(folder) {
            continue;
        }
        if direct {
            let candidate = folder.join(name);
            match probe(&candidate) {
                Probe::Absent => continue,
                Probe::Unsafe => return Err(Ineligible::Unsafe(candidate)),
                Probe::Present => return checked(candidate, &roots),
            }
        }
        let script = folder.join(format!("{name}.ps1"));
        let has_script = !matches!(probe(&script), Probe::Absent);
        for extension in &pathext {
            let candidate = folder.join(format!("{name}{extension}"));
            match probe(&candidate) {
                Probe::Absent => continue,
                Probe::Unsafe => return Err(Ineligible::Unsafe(candidate)),
                Probe::Present => {}
            }
            return match extension.as_str() {
                ".exe" | ".com" if has_script => Err(Ineligible::Script(script)),
                ".exe" | ".com" => checked(candidate, &roots),
                ".bat" | ".cmd" => Err(Ineligible::BatchFile(candidate)),
                _ => Err(Ineligible::Script(candidate)),
            };
        }
        if has_script {
            return Err(Ineligible::Script(script));
        }
    }
    Err(Ineligible::NotFound)
}

#[cfg(all(test, windows))]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;
    use crate::env::MapEnv;
    use crate::testkit::TempDir;

    /// A file that looks like a program; nothing here ever runs it.
    fn plant(dir: &Path, name: &str) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, b"MZ not a real program").unwrap();
        path
    }

    fn env_with(path: &[&Path], pathext: &str) -> MapEnv {
        let joined = path
            .iter()
            .map(|entry| entry.display().to_string())
            .collect::<Vec<_>>()
            .join(";");
        MapEnv::new().with("PATH", joined).with("PATHEXT", pathext)
    }

    const PATHEXT: &str = ".COM;.EXE;.BAT;.CMD";

    struct Layout {
        _root: TempDir,
        root: PathBuf,
        workspace: PathBuf,
        globals: PathBuf,
    }

    fn layout(tag: &str) -> Layout {
        let root_dir = TempDir::new(tag);
        let root = std::fs::canonicalize(root_dir.path()).unwrap();
        let workspace = root.join("workspace");
        let globals = root.join("globals");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&globals).unwrap();
        Layout {
            _root: root_dir,
            root,
            workspace,
            globals,
        }
    }

    fn resolve(name: &str, env: &MapEnv, at: &Layout) -> Result<Resolved, Ineligible> {
        resolve_program(name, env, Some(&at.workspace), &at.globals)
    }

    /// CF5: a planted `cargo.exe` in the workspace is not chosen, even with the
    /// workspace first on `PATH`.
    #[test]
    fn a_program_planted_in_the_workspace_is_never_chosen() {
        let at = layout("resolve-planted");
        plant(&at.workspace, "cargo.exe");
        plant(&at.workspace.join("bin"), "cargo.exe");
        let tools = at.root.join("tools");
        let real = plant(&tools, "cargo.exe");
        let env = env_with(&[&at.workspace, &at.workspace.join("bin"), &tools], PATHEXT);
        let resolved = resolve("cargo", &env, &at).unwrap();
        assert_eq!(resolved.path, real);
        assert_eq!(resolved.real, real);
        // Only the workspace on PATH: nothing to run.
        let env = env_with(&[&at.workspace], PATHEXT);
        assert_eq!(resolve("cargo", &env, &at), Err(Ineligible::NotFound));
        // The same for Lattice's own state.
        plant(&at.globals, "git.exe");
        let env = env_with(&[&at.globals, &tools], PATHEXT);
        assert_eq!(resolve("git", &env, &at), Err(Ineligible::NotFound));
    }

    /// CF5: a name that resolves to a batch file or a script is ineligible.
    #[test]
    fn batch_files_and_scripts_are_ineligible() {
        let at = layout("resolve-scripts");
        let shims = at.root.join("node");
        let npm = plant(&shims, "npm.cmd");
        let later = at.root.join("later");
        plant(&later, "npm.exe");
        let env = env_with(&[&shims, &later], PATHEXT);
        // The first entry holds only a .cmd: it is what a shell would run.
        assert_eq!(resolve("npm", &env, &at), Err(Ineligible::BatchFile(npm)));
        // .bat too, and asked for by its own name.
        let bat = plant(&shims, "build.bat");
        assert_eq!(resolve("build", &env, &at), Err(Ineligible::BatchFile(bat)));
        for name in ["npm.cmd", "build.BAT", "go.ps1"] {
            assert!(
                matches!(
                    resolve(name, &env, &at),
                    Err(Ineligible::BatchFile(_) | Ineligible::Script(_))
                ),
                "{name}"
            );
        }
        // A .ps1 beside an .exe in the same entry: PowerShell could run it.
        let both = at.root.join("both");
        plant(&both, "tool.exe");
        let script = plant(&both, "tool.ps1");
        let env = env_with(&[&both], PATHEXT);
        assert_eq!(resolve("tool", &env, &at), Err(Ineligible::Script(script)));
        // Another script host's file that PATHEXT puts first.
        let hosts = at.root.join("hosts");
        let vbs = plant(&hosts, "report.vbs");
        plant(&hosts, "report.exe");
        let env = env_with(&[&hosts], ".VBS;.EXE");
        assert_eq!(resolve("report", &env, &at), Err(Ineligible::Script(vbs)));
    }

    #[test]
    fn the_pathext_order_decides_within_one_entry() {
        let at = layout("resolve-order");
        let dir = at.root.join("dir");
        let exe = plant(&dir, "tool.exe");
        let cmd = plant(&dir, "tool.cmd");
        let env = env_with(&[&dir], ".COM;.EXE;.BAT;.CMD");
        assert_eq!(resolve("tool", &env, &at).unwrap().path, exe);
        let env = env_with(&[&dir], ".CMD;.EXE");
        assert_eq!(resolve("tool", &env, &at), Err(Ineligible::BatchFile(cmd)));
        // A name that ends in .exe is looked for as it is.
        let env = env_with(&[&dir], ".CMD;.EXE");
        assert_eq!(resolve("TOOL.EXE", &env, &at).unwrap().real, exe);
        // An unset PATHEXT is Windows' default (.COM before .EXE before .BAT).
        let env = MapEnv::new().with("PATH", dir.display().to_string());
        assert_eq!(resolve("tool", &env, &at).unwrap().path, exe);
    }

    #[test]
    fn only_a_bare_name_is_resolved() {
        let at = layout("resolve-bare");
        let env = env_with(&[&at.root], PATHEXT);
        for name in [
            "",
            ".",
            "..",
            "a/b",
            r"a\b",
            "C:x",
            r"..\cargo",
            "cargo.",
            "cargo ",
            "*",
            "a?b",
            "a|b",
            "a\"b",
            "a\tb",
        ] {
            assert_eq!(
                resolve(name, &env, &at),
                Err(Ineligible::NotABareName),
                "{name:?}"
            );
        }
    }

    #[test]
    fn relative_and_network_entries_are_skipped_without_a_connection() {
        let at = layout("resolve-net");
        let tools = at.root.join("tools");
        let real = plant(&tools, "tool.exe");
        let started = Instant::now();
        let env = MapEnv::new()
            .with(
                "PATH",
                format!(r"\\198.51.100.7\x;.;tools;;\rooted;{}", tools.display()),
            )
            .with("PATHEXT", PATHEXT);
        assert_eq!(resolve("tool", &env, &at).unwrap().path, real);
        // A probe of the share would take seconds; it is never made.
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn a_link_into_the_workspace_or_to_a_share_is_refused() {
        let at = layout("resolve-links");
        let planted = plant(&at.workspace, "tool.exe");
        let tools = at.root.join("tools");
        std::fs::create_dir_all(&tools).unwrap();
        let link = tools.join("tool.exe");
        std::os::windows::fs::symlink_file(&planted, &link).unwrap();
        let env = env_with(&[&tools], PATHEXT);
        assert_eq!(resolve("tool", &env, &at), Err(Ineligible::Unsafe(link)));
        let shared = at.root.join("shared");
        std::fs::create_dir_all(&shared).unwrap();
        let to_share = shared.join("tool.exe");
        std::os::windows::fs::symlink_file(r"\\198.51.100.7\x\tool.exe", &to_share).unwrap();
        let started = Instant::now();
        let env = env_with(&[&shared], PATHEXT);
        assert_eq!(
            resolve("tool", &env, &at),
            Err(Ineligible::Unsafe(to_share))
        );
        assert!(started.elapsed() < Duration::from_secs(2));
        // A folder named like a program is not a program.
        let odd = at.root.join("odd");
        std::fs::create_dir_all(odd.join("tool.exe")).unwrap();
        let env = env_with(&[&odd], PATHEXT);
        assert_eq!(resolve("tool", &env, &at), Err(Ineligible::NotFound));
    }

    /// `\\?\GLOBALROOT\Device\…`: a device path that reaches a local file or
    /// folder, the offline stand-in for a hop to a share.
    fn device_path_of(path: &Path) -> PathBuf {
        let opened = open_no_follow(path, Access::Attributes).unwrap();
        let nt = lattice_sys::fs::seam::nt_path(&opened.file).unwrap();
        PathBuf::from(format!(r"\\?\GLOBALROOT{}", nt.display()))
    }

    /// X2c (spec §22.6): the whole link chain is checked, not its first hop.
    /// `tools\tool.exe` links to a local `stage\tool.exe`, which links to a
    /// device path: refused before the second hop is followed. A chain of
    /// local hops still resolves (the positive control), to its final file.
    /// Mutant: `checked` opens with `File::open`, following the chain.
    #[test]
    fn every_hop_of_a_link_chain_is_checked() {
        let at = layout("resolve-chain");
        let real = plant(&at.root.join("real"), "tool.exe");
        let stage = at.root.join("stage");
        std::fs::create_dir_all(&stage).unwrap();
        let tools = at.root.join("tools");
        std::fs::create_dir_all(&tools).unwrap();
        std::os::windows::fs::symlink_file(device_path_of(&real), stage.join("tool.exe")).unwrap();
        let first = tools.join("tool.exe");
        std::os::windows::fs::symlink_file(stage.join("tool.exe"), &first).unwrap();
        let env = env_with(&[&tools], PATHEXT);
        assert_eq!(resolve("tool", &env, &at), Err(Ineligible::Unsafe(first)));

        let local = at.root.join("local");
        std::fs::create_dir_all(&local).unwrap();
        let hop = at.root.join("hop");
        std::fs::create_dir_all(&hop).unwrap();
        std::os::windows::fs::symlink_file(&real, hop.join("tool.exe")).unwrap();
        std::os::windows::fs::symlink_file(hop.join("tool.exe"), local.join("tool.exe")).unwrap();
        let env = env_with(&[&local], PATHEXT);
        let resolved = resolve("tool", &env, &at).unwrap();
        assert_eq!(resolved.path, local.join("tool.exe"));
        assert_eq!(
            resolved.real, real,
            "the final file, through two local hops"
        );
    }

    /// X2b, X2c (the verifier's AppExecLink probe): an app execution alias is
    /// never the program, whether the name itself is one (as in
    /// `WindowsApps`) or a symlink's chain ends at one; its data names a
    /// `.cmd` here, which `CreateProcessW` would run through `cmd.exe`.
    /// Mutant: `link_target` walks through an alias as it walks through
    /// other reparse points (the chain case resolves).
    #[test]
    fn an_app_execution_alias_is_never_the_program() {
        use lattice_sys::fs::seam::create_app_exec_link;
        let at = layout("resolve-alias");
        let wrapper = at.root.join("wrapper.cmd");
        std::fs::write(
            &wrapper,
            b"@echo off
",
        )
        .unwrap();
        let apps = at.root.join("apps");
        std::fs::create_dir_all(&apps).unwrap();
        let alias = apps.join("tool.exe");
        create_app_exec_link(&alias, &wrapper).unwrap();
        let env = env_with(&[&apps], PATHEXT);
        assert_eq!(
            resolve("tool", &env, &at),
            Err(Ineligible::Unsafe(alias.clone()))
        );
        let tools = at.root.join("tools");
        std::fs::create_dir_all(&tools).unwrap();
        let first = tools.join("tool.exe");
        std::os::windows::fs::symlink_file(&alias, &first).unwrap();
        let env = env_with(&[&tools], PATHEXT);
        assert_eq!(resolve("tool", &env, &at), Err(Ineligible::Unsafe(first)));
    }

    /// X2c: a `PATH` entry whose folder is reached through a link off the
    /// drive is skipped before anything below it is probed; the next entry
    /// answers.
    /// Mutant: the folder walk dropped (the entry is probed through its link).
    #[test]
    fn a_path_entry_linked_off_the_drive_is_skipped() {
        let at = layout("resolve-entry-link");
        let hidden = at.root.join("hidden");
        plant(&hidden, "tool.exe");
        let via = at.root.join("via");
        std::os::windows::fs::symlink_dir(device_path_of(&hidden), &via).unwrap();
        let later = at.root.join("later");
        let fallback = plant(&later, "tool.exe");
        let env = env_with(&[&via, &later], PATHEXT);
        assert_eq!(resolve("tool", &env, &at).unwrap().path, fallback);
    }

    /// X2c: a chain whose last hop is a share is refused before that hop is
    /// followed: no wait for a connection.
    #[test]
    fn a_chain_to_a_share_is_refused_without_a_connection() {
        let at = layout("resolve-chain-unc");
        let stage = at.root.join("stage");
        std::fs::create_dir_all(&stage).unwrap();
        std::os::windows::fs::symlink_file(r"\\198.51.100.7\x\tool.exe", stage.join("tool.exe"))
            .unwrap();
        let tools = at.root.join("tools");
        std::fs::create_dir_all(&tools).unwrap();
        let first = tools.join("tool.exe");
        std::os::windows::fs::symlink_file(stage.join("tool.exe"), &first).unwrap();
        let started = Instant::now();
        let env = env_with(&[&tools], PATHEXT);
        assert_eq!(resolve("tool", &env, &at), Err(Ineligible::Unsafe(first)));
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn every_refusal_has_one_sentence() {
        for refusal in [
            Ineligible::NotABareName,
            Ineligible::BatchFile(PathBuf::new()),
            Ineligible::Script(PathBuf::new()),
            Ineligible::NotFound,
            Ineligible::Unsafe(PathBuf::new()),
        ] {
            let sentence = refusal.sentence();
            assert!(sentence.ends_with('.') && sentence.len() > 10, "{sentence}");
        }
    }
}
