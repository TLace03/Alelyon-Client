//! Opening a path one component at a time, never following a link before
//! reading where it points (the chat core's spec §6.2 WP10 step 1, §6.3
//! FT6). Not a port: Python's `os.path.realpath` follows every link first and
//! checks afterwards, and a link to `\\host\share` connects as it is followed.
//!
//! [`open_walk`] takes an absolute path, normalises it by its text (`.` and
//! `..` as Windows does, before the file system sees them), and opens each
//! component from its drive's root down with `lattice_sys::fs::open_no_follow`
//! (`FILE_FLAG_OPEN_REPARSE_POINT`). A component that is a symlink or a
//! junction has its target read first:
//! - a target that is not local (a UNC or `\\?\UNC\` path, a device path, a
//!   path on a `DRIVE_REMOTE` drive) is refused **before** it is followed, so
//!   no SMB or WebDAV connection is made;
//! - with [`LinkRule::Inside`], a target outside the given root is refused
//!   before it is followed too;
//! - otherwise the walk starts again from the target's own root, with the
//!   rest of the path appended, at most [`MAX_LINKS`] times.
//!
//! An app execution alias (`IO_REPARSE_TAG_APPEXECLINK`) is refused at any
//! component (`WalkError::BadLink`): starting it starts whatever its data
//! names, so no walk treats it as a plain file (spec §22.6 X2b, LR3a, X2c).
//! Other reparse points (a cloud placeholder, a deduplicated or compressed
//! file) are not links and are walked through as they are by [`open_walk`];
//! [`open_walk_links_only`], for a program to start and the llama.cpp files,
//! refuses them as well, except a WOF-compressed or deduplicated file, which
//! is stored differently but redirects nothing. The path that is opened at
//! the end is read back from its handle
//! ([`Walked::final_path`]: every component's long name, links resolved), and
//! with [`LinkRule::Inside`] it must still be inside the root.
//!
//! Nothing here writes, and nothing here opens a path whose text is not local:
//! [`is_local_text`] decides that by the text alone (and `GetDriveTypeW`,
//! which opens nothing), and `lattice_sys` refuses such a path again.

use std::ffi::OsString;
use std::fs::File;
use std::io;
use std::path::{Component, Path, PathBuf, Prefix};

use lattice_sys::fs::{
    Access, DriveType, LinkKind, Opened, PathKind, drive_type, file_identity, final_path,
    is_storage_only_tag, open_no_follow, path_kind,
};

use crate::exec::spawn::comparable;

/// How many links one walk follows before it gives up (Windows itself stops
/// at 63).
pub const MAX_LINKS: u32 = 32;

/// Which links a walk may follow.
#[derive(Clone, Copy, Debug)]
pub enum LinkRule<'a> {
    /// Any link to a local path.
    AnyLocal,
    /// Only a link whose target is inside this folder (a canonical path), and
    /// the path opened at the end must be inside it as well.
    Inside(&'a Path),
}

/// Why a walk stopped.
#[derive(Debug)]
pub enum WalkError {
    /// The path, or a link's target, is not local; nothing was opened there.
    NotLocal(PathBuf),
    /// A link's target, or the path opened, is outside the root.
    Outside(PathBuf),
    /// The path is not absolute.
    NotAbsolute,
    /// More than [`MAX_LINKS`] links.
    TooManyLinks,
    /// A link whose target could not be read.
    BadLink(PathBuf),
    /// No such file or folder.
    NotFound,
    /// Anything else the file system said.
    Io(io::Error),
}

impl WalkError {
    fn from_io(error: io::Error) -> Self {
        if error.kind() == io::ErrorKind::NotFound {
            Self::NotFound
        } else {
            Self::Io(error)
        }
    }
}

/// What a walk opened.
#[derive(Debug)]
pub struct Walked {
    /// The handle, with the access asked for.
    pub file: File,
    pub is_dir: bool,
    /// The path the handle names, read back from it (`\\?\C:\…`).
    pub final_path: PathBuf,
}

/// Is `path`, by its text, on this machine? A drive letter on a drive that is
/// not `DRIVE_REMOTE`, or a volume GUID path. A UNC, `\\?\UNC\` or device
/// path is not. A rooted or relative path is judged by the base it will be
/// joined to, so it is not judged here (`false`).
pub fn is_local_text(path: &Path) -> bool {
    match path_kind(path) {
        PathKind::Drive => drive_type(path) != DriveType::Remote,
        PathKind::VolumeGuid => true,
        PathKind::Unc | PathKind::Device | PathKind::Rooted | PathKind::Relative => false,
    }
}

/// `path` with a verbatim drive prefix removed (`\\?\C:\x` reads `C:\x`), so
/// std's component parser sees `.` and `..`.
fn plain(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    for prefix in [r"\\?\", r"\??\"] {
        if let Some(rest) = text.strip_prefix(prefix) {
            let bytes = rest.as_bytes();
            if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
                return PathBuf::from(rest.replace('/', "\\"));
            }
        }
    }
    PathBuf::from(text.replace('/', "\\"))
}

/// The root (`C:\` or `\\?\Volume{…}\`) and the normal components of an
/// absolute local path, with `.` dropped and `..` taken back lexically, as
/// Windows resolves them before the file system is asked.
fn split(path: &Path) -> Result<(PathBuf, Vec<OsString>), WalkError> {
    let kind = path_kind(path);
    if !matches!(kind, PathKind::Drive | PathKind::VolumeGuid) {
        return Err(match kind {
            PathKind::Unc | PathKind::Device => WalkError::NotLocal(path.to_path_buf()),
            _ => WalkError::NotAbsolute,
        });
    }
    let plain = plain(path);
    let mut root: Option<PathBuf> = None;
    let mut parts: Vec<OsString> = Vec::new();
    for component in plain.components() {
        match component {
            Component::Prefix(prefix) => {
                root = Some(match prefix.kind() {
                    Prefix::Disk(letter) | Prefix::VerbatimDisk(letter) => {
                        PathBuf::from(format!(r"{}:\", char::from(letter)))
                    }
                    Prefix::Verbatim(name) => {
                        PathBuf::from(format!(r"\\?\{}\", name.to_string_lossy()))
                    }
                    _ => return Err(WalkError::NotLocal(path.to_path_buf())),
                });
            }
            Component::RootDir | Component::CurDir => {}
            Component::ParentDir => {
                parts.pop();
            }
            Component::Normal(name) => parts.push(name.to_owned()),
        }
    }
    let root = root.ok_or(WalkError::NotAbsolute)?;
    if !is_local_text(&root) {
        return Err(WalkError::NotLocal(path.to_path_buf()));
    }
    Ok((root, parts))
}

fn join(root: &Path, parts: &[OsString]) -> PathBuf {
    let mut out = root.to_path_buf();
    for part in parts {
        out.push(part);
    }
    out
}

/// True when `path` is `root` or inside it, compared case-insensitively
/// without verbatim prefixes.
pub fn is_inside(path: &Path, root: &Path) -> bool {
    let path = comparable(&path.to_string_lossy());
    let root = comparable(&root.to_string_lossy());
    !root.is_empty() && (path == root || path.starts_with(&format!("{root}\\")))
}

/// Every call this module makes to `lattice_sys::fs::open_no_follow`, recorded
/// in tests only: a falsifier reads it to show a path that is not local never
/// reached an open.
#[cfg(test)]
pub(crate) mod record {
    use std::cell::RefCell;
    use std::path::{Path, PathBuf};

    thread_local! {
        static OPENED: RefCell<Vec<PathBuf>> = const { RefCell::new(Vec::new()) };
    }

    pub(crate) fn note(path: &Path) {
        OPENED.with(|opened| opened.borrow_mut().push(path.to_path_buf()));
    }

    /// The paths opened on this thread since the last call.
    pub(crate) fn take() -> Vec<PathBuf> {
        OPENED.with(|opened| std::mem::take(&mut *opened.borrow_mut()))
    }
}

fn open_one(path: &Path, access: Access) -> io::Result<Opened> {
    #[cfg(test)]
    record::note(path);
    open_no_follow(path, access)
}

/// Open `path` for its attributes without following a link at its last
/// component; earlier components are opened as they are, so `path`'s folder
/// must already be a final path (no links in it). A path whose text is not
/// local is refused without an open.
pub fn probe(path: &Path) -> io::Result<Opened> {
    if !is_local_text(path) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "a network or device path is not opened",
        ));
    }
    open_one(path, Access::Attributes)
}

/// Which reparse points that are not links a walk passes through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reparse {
    /// Every one but an app execution alias (a placeholder's or a compressed
    /// file's content is the file's own).
    Data,
    /// Only a WOF-compressed or deduplicated file.
    StorageOnly,
}

/// Where a link at `at` (whose folder is `parent`) leads, if it may be
/// followed under `rule`; `None` when the reparse point is not a link and may
/// be walked through under `reparse`. An app execution alias, and under
/// [`Reparse::StorageOnly`] any reparse point that is neither a link nor
/// storage-only, is refused as a `BadLink`.
fn link_target(
    opened: &Opened,
    at: &Path,
    parent: &Path,
    rule: LinkRule<'_>,
    reparse: Reparse,
) -> Result<Option<PathBuf>, WalkError> {
    let Some(link) = &opened.link else {
        return Ok(None);
    };
    match link.kind {
        LinkKind::AppExecLink => return Err(WalkError::BadLink(at.to_path_buf())),
        LinkKind::Other if reparse == Reparse::Data || is_storage_only_tag(link.tag) => {
            return Ok(None);
        }
        LinkKind::Other => return Err(WalkError::BadLink(at.to_path_buf())),
        LinkKind::Symlink | LinkKind::Junction => {}
    }
    let Some(target) = &link.target else {
        return Err(WalkError::BadLink(at.to_path_buf()));
    };
    let target = if link.relative {
        match path_kind(target) {
            PathKind::Relative => parent.join(target),
            PathKind::Rooted => {
                let (root, _) = split(parent)?;
                root.join(target.to_string_lossy().trim_start_matches(['\\', '/']))
            }
            _ => target.clone(),
        }
    } else {
        target.clone()
    };
    if !is_local_text(&target) {
        return Err(WalkError::NotLocal(target));
    }
    if let LinkRule::Inside(root) = rule {
        let (target_root, target_parts) = split(&target)?;
        if !is_inside(&join(&target_root, &target_parts), root) {
            return Err(WalkError::Outside(target));
        }
    }
    Ok(Some(target))
}

/// Open `path` as the module header says. `access` is the access of the
/// handle returned; every component on the way is opened for its attributes
/// only.
pub fn open_walk(path: &Path, access: Access, rule: LinkRule<'_>) -> Result<Walked, WalkError> {
    walk(path, access, rule, Reparse::Data)
}

/// [`open_walk`] that passes through no reparse point but a symlink, a
/// junction, a WOF-compressed or a deduplicated file: for a program that will
/// be started and for the llama.cpp server's files, where a cloud
/// placeholder or a third party's reparse point has no business (X2b, LR3a,
/// X2c).
pub fn open_walk_links_only(
    path: &Path,
    access: Access,
    rule: LinkRule<'_>,
) -> Result<Walked, WalkError> {
    walk(path, access, rule, Reparse::StorageOnly)
}

fn walk(
    path: &Path,
    access: Access,
    rule: LinkRule<'_>,
    reparse: Reparse,
) -> Result<Walked, WalkError> {
    if !matches!(path_kind(path), PathKind::Drive | PathKind::VolumeGuid) {
        return Err(match path_kind(path) {
            PathKind::Unc | PathKind::Device => WalkError::NotLocal(path.to_path_buf()),
            _ => WalkError::NotAbsolute,
        });
    }
    let (mut root, mut parts) = split(path)?;
    if let LinkRule::Inside(inside) = rule
        && !is_inside(&join(&root, &parts), inside)
    {
        return Err(WalkError::Outside(path.to_path_buf()));
    }
    let mut links = 0u32;
    'walk: loop {
        let mut current = root.clone();
        for at in 0..parts.len() {
            let candidate = current.join(&parts[at]);
            let last = at + 1 == parts.len();
            let opened = open_one(&candidate, if last { access } else { Access::Attributes })
                .map_err(WalkError::from_io)?;
            if let Some(target) = link_target(&opened, &candidate, &current, rule, reparse)? {
                links += 1;
                if links > MAX_LINKS {
                    return Err(WalkError::TooManyLinks);
                }
                let (target_root, mut target_parts) = split(&target)?;
                target_parts.extend(parts[at + 1..].iter().cloned());
                root = target_root;
                parts = target_parts;
                continue 'walk;
            }
            if last {
                return finish(opened, access, rule);
            }
            if !opened.is_dir {
                return Err(WalkError::NotFound);
            }
            current = candidate;
        }
        // The path is a root itself.
        let opened = open_one(&root, access).map_err(WalkError::from_io)?;
        return finish(opened, access, rule);
    }
}

/// The handle the caller gets: the no-follow handle, or, for a reparse point
/// that is not a link (a placeholder or a compressed file, whose content the
/// file system supplies only on a normal open), a normal open of the same
/// file, checked to be the same file by its identity.
fn finish(opened: Opened, access: Access, rule: LinkRule<'_>) -> Result<Walked, WalkError> {
    let final_path = final_path(&opened.file).map_err(WalkError::Io)?;
    if let LinkRule::Inside(root) = rule
        && !is_inside(&final_path, root)
    {
        return Err(WalkError::Outside(final_path));
    }
    let file = match (&opened.link, access) {
        (Some(_), Access::Read) if !opened.is_dir => {
            let reopened = File::open(&final_path).map_err(WalkError::from_io)?;
            let same = file_identity(&reopened).map_err(WalkError::Io)?
                == file_identity(&opened.file).map_err(WalkError::Io)?;
            if !same {
                return Err(WalkError::Outside(final_path));
            }
            reopened
        }
        _ => opened.file,
    };
    Ok(Walked {
        file,
        is_dir: opened.is_dir,
        final_path,
    })
}

#[cfg(all(test, windows))]
mod tests {
    use std::io::Read;

    use super::*;
    use crate::testkit::TempDir;

    fn real(dir: &TempDir) -> PathBuf {
        let opened = open_walk(dir.path(), Access::Attributes, LinkRule::AnyLocal).unwrap();
        opened.final_path
    }

    #[test]
    fn a_plain_file_opens_with_its_long_final_path() {
        let dir = TempDir::new("localfs-plain");
        std::fs::create_dir_all(dir.path().join("a").join("b")).unwrap();
        std::fs::write(dir.path().join("a").join("b").join("f.txt"), b"hi").unwrap();
        let mut walked = open_walk(
            &dir.path().join(r"a\.\b\..\b\f.txt"),
            Access::Read,
            LinkRule::AnyLocal,
        )
        .unwrap();
        assert!(!walked.is_dir);
        let mut text = String::new();
        walked.file.read_to_string(&mut text).unwrap();
        assert_eq!(text, "hi");
        assert!(walked.final_path.to_string_lossy().starts_with(r"\\?\"));
        assert!(walked.final_path.ends_with(r"a\b\f.txt"));
        assert!(matches!(
            open_walk(&dir.path().join("nope"), Access::Read, LinkRule::AnyLocal),
            Err(WalkError::NotFound)
        ));
    }

    /// FT6 and WP10 step 1: a network or device path is refused by its text,
    /// and nothing is handed to an open. The documentation address is used;
    /// no packet is sent anywhere.
    /// Mutant: the text check in `open_walk` dropped (the path reaches
    /// `open_no_follow`, which the recorder sees).
    #[test]
    fn a_network_or_device_path_is_never_opened() {
        let _ = record::take();
        for path in [
            r"\\198.51.100.7\x\f",
            r"//198.51.100.7/x/f",
            r"\\?\UNC\198.51.100.7\x\f",
            r"\\.\UNC\198.51.100.7\x\f",
            r"\\.\pipe\x",
            r"\\?\GLOBALROOT\Device\Mup\198.51.100.7\x",
        ] {
            let outcome = open_walk(Path::new(path), Access::Read, LinkRule::AnyLocal);
            assert!(
                matches!(outcome, Err(WalkError::NotLocal(_))),
                "{path}: {outcome:?}"
            );
        }
        assert_eq!(record::take(), Vec::<PathBuf>::new(), "nothing was opened");
        assert!(!is_local_text(Path::new(r"\\198.51.100.7\x")));
        assert!(is_local_text(Path::new(r"C:\Windows")));
        assert!(is_local_text(Path::new(r"\\?\C:\Windows")));
        assert!(!is_local_text(Path::new(r"relative\x")));
    }

    #[test]
    fn a_junction_inside_is_followed_and_one_outside_is_refused_before() {
        let dir = TempDir::new("localfs-junction");
        let root = dir.path().join("root");
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(root.join("real")).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(root.join("real").join("f.txt"), b"in").unwrap();
        std::fs::write(outside.join("g.txt"), b"out").unwrap();
        std::fs::create_dir(root.join("inner")).unwrap();
        lattice_sys::fs::seam::create_junction(&root.join("inner"), &root.join("real")).unwrap();
        std::fs::create_dir(root.join("away")).unwrap();
        lattice_sys::fs::seam::create_junction(&root.join("away"), &outside).unwrap();
        let canonical = real(&dir).join("root");
        let walked = open_walk(
            &root.join("inner").join("f.txt"),
            Access::Read,
            LinkRule::Inside(&canonical),
        )
        .unwrap();
        assert!(walked.final_path.ends_with(r"root\real\f.txt"));
        let _ = record::take();
        let refused = open_walk(
            &root.join("away").join("g.txt"),
            Access::Read,
            LinkRule::Inside(&canonical),
        );
        assert!(matches!(refused, Err(WalkError::Outside(_))), "{refused:?}");
        let opened = record::take();
        assert!(
            !opened.iter().any(|path| path.ends_with("g.txt")),
            "refused before it was followed: {opened:?}"
        );
        // AnyLocal follows a local junction anywhere.
        let walked = open_walk(
            &root.join("away").join("g.txt"),
            Access::Read,
            LinkRule::AnyLocal,
        )
        .unwrap();
        assert!(walked.final_path.ends_with(r"outside\g.txt"));
    }

    #[test]
    fn a_path_outside_the_root_is_refused_by_its_text() {
        let dir = TempDir::new("localfs-outside");
        let root = dir.path().join("root");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(dir.path().join("x.txt"), b"x").unwrap();
        let canonical = real(&dir).join("root");
        let outcome = open_walk(
            &root.join(r"..\x.txt"),
            Access::Read,
            LinkRule::Inside(&canonical),
        );
        assert!(matches!(outcome, Err(WalkError::Outside(_))), "{outcome:?}");
    }
}
