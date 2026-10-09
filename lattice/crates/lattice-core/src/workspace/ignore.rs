//! WP8 for a folder without git: `.latticeignore` (gitignore syntax) and the
//! built-in defaults (the chat core's spec §6.2 WP8, design §5.1).
//!
//! The defaults (`*.env`, `.env*`, `FAMEnvironment.env`, `*.pem`, `*.key`,
//! `id_*`, `.ssh/`) are where secrets live; they are refused as
//! [`Verdict::Default`] whatever `.latticeignore` says, because a `!` line in
//! a file of the folder must not re-include them (FT4: nothing in the folder
//! grants anything). `.latticeignore` is read from the workspace's root only,
//! through the no-follow walk; a nested `.latticeignore` is not read. Patterns
//! match without regard to case, as Windows names do. A folder treated as one
//! without git by FT6 uses these rules too; git's own ignore files are not
//! read there.

use std::io::Read;
use std::path::Path;

use ignore::Match;
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use lattice_sys::fs::Access;

use crate::localfs::{LinkRule, WalkError, open_walk};

/// The built-in defaults.
pub const DEFAULTS: [&str; 7] = [
    "*.env",
    ".env*",
    "FAMEnvironment.env",
    "*.pem",
    "*.key",
    "id_*",
    ".ssh/",
];
/// The file a folder may hold its own rules in.
pub const LATTICEIGNORE: &str = ".latticeignore";
/// The largest `.latticeignore` read.
pub const MAX_LATTICEIGNORE: u64 = 256 * 1024;

/// What the rules say about one path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Allowed,
    /// A built-in default matched: a file that may hold secrets.
    Default,
    /// `.latticeignore` matched.
    Ignored,
}

/// Why the rules could not be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RulesError {
    /// `.latticeignore` is a link outside the folder, or could not be read.
    Unreadable,
    /// `.latticeignore` is larger than [`MAX_LATTICEIGNORE`], or not UTF-8.
    TooLarge,
}

/// The ignore rules of one folder without git.
#[derive(Clone, Debug)]
pub struct IgnoreRules {
    defaults: Gitignore,
    folder: Option<Gitignore>,
}

fn builder() -> GitignoreBuilder {
    let mut builder = GitignoreBuilder::new(".");
    builder.case_insensitive(true).expect("a fixed option");
    builder
}

impl IgnoreRules {
    /// The defaults alone.
    pub fn defaults_only() -> Self {
        let mut builder = builder();
        for line in DEFAULTS {
            builder.add_line(None, line).expect("the defaults parse");
        }
        Self {
            defaults: builder.build().expect("the defaults build"),
            folder: None,
        }
    }

    /// The defaults and the root's `.latticeignore` (`root` is the
    /// workspace's final path). A line that does not parse is left out, as
    /// git leaves out a bad pattern.
    pub fn load(root: &Path) -> Result<Self, RulesError> {
        let mut rules = Self::defaults_only();
        let path = root.join(LATTICEIGNORE);
        let walked = match open_walk(&path, Access::Read, LinkRule::Inside(root)) {
            Ok(walked) => walked,
            Err(WalkError::NotFound) => return Ok(rules),
            Err(_) => return Err(RulesError::Unreadable),
        };
        if walked.is_dir {
            return Ok(rules);
        }
        let mut bytes = Vec::new();
        walked
            .file
            .take(MAX_LATTICEIGNORE + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| RulesError::Unreadable)?;
        if bytes.len() as u64 > MAX_LATTICEIGNORE {
            return Err(RulesError::TooLarge);
        }
        let text = String::from_utf8(bytes).map_err(|_| RulesError::TooLarge)?;
        let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
        let mut builder = builder();
        for line in text.lines() {
            let _ = builder.add_line(None, line.trim_end_matches('\r'));
        }
        rules.folder = builder.build().ok();
        Ok(rules)
    }

    /// The verdict for `path` (relative, forward slashes), or for any folder
    /// above it: a file inside an ignored folder is ignored.
    pub fn check(&self, path: &str, is_dir: bool) -> Verdict {
        if matches!(
            self.defaults.matched_path_or_any_parents(path, is_dir),
            Match::Ignore(_)
        ) {
            return Verdict::Default;
        }
        match &self.folder {
            Some(folder)
                if matches!(
                    folder.matched_path_or_any_parents(path, is_dir),
                    Match::Ignore(_)
                ) =>
            {
                Verdict::Ignored
            }
            _ => Verdict::Allowed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::TempDir;

    fn root(dir: &TempDir) -> std::path::PathBuf {
        open_walk(dir.path(), Access::Attributes, LinkRule::AnyLocal)
            .unwrap()
            .final_path
    }

    /// The defaults refuse what may hold secrets, at any depth and in any
    /// case, and `.latticeignore` cannot re-include them.
    /// Mutant: the defaults matched with case.
    #[test]
    fn the_defaults_hold_and_a_folder_cannot_undo_them() {
        let dir = TempDir::new("ignore-defaults");
        std::fs::write(
            dir.path().join(LATTICEIGNORE),
            "!*.env\n!.ssh/\nbuild/\n/top.txt\n",
        )
        .unwrap();
        let rules = IgnoreRules::load(&root(&dir)).unwrap();
        for path in [
            ".env",
            ".env.local",
            "FAMEnvironment.env",
            "famenvironment.ENV",
            "sub/deep/x.env",
            "cert.PEM",
            "a/server.key",
            "id_rsa",
            "keys/id_ed25519.pub",
            ".ssh/config",
            "home/.SSH/known_hosts",
        ] {
            assert_eq!(rules.check(path, false), Verdict::Default, "{path}");
        }
        assert_eq!(rules.check("build/out.txt", false), Verdict::Ignored);
        assert_eq!(rules.check("build", true), Verdict::Ignored);
        assert_eq!(rules.check("top.txt", false), Verdict::Ignored);
        assert_eq!(rules.check("sub/top.txt", false), Verdict::Allowed);
        assert_eq!(rules.check("src/main.rs", false), Verdict::Allowed);
        assert_eq!(rules.check("environment.txt", false), Verdict::Allowed);
    }

    #[test]
    fn no_latticeignore_means_the_defaults_alone() {
        let dir = TempDir::new("ignore-none");
        let rules = IgnoreRules::load(&root(&dir)).unwrap();
        assert_eq!(rules.check("x.env", false), Verdict::Default);
        assert_eq!(rules.check("build/out.txt", false), Verdict::Allowed);
    }
}
