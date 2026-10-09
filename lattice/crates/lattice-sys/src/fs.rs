//! Files: identity, final paths, links read before they are followed, drive
//! types, `ReplaceFileW` with a backup name, and moves that never replace.
//! Not a port; the chat core's spec §2.1, §4.1 (`WorkspaceId`), §6.2 WP10,
//! §6.1 (`DRIVE_REMOTE`), §7.5 step 5 and §5.8 ND2 say why each exists.
//!
//! Invariants:
//! - Nothing here removes a file or a directory. A failed or partial operation
//!   leaves every file it touched under some name (see [`replace_file`]).
//! - [`open_no_follow`] never opens a network or device path: a UNC path, a
//!   `\\?\UNC\` path, a device path or a path on a remote drive is refused by
//!   its text before `CreateFileW` is called, so no SMB or WebDAV connection
//!   can start here. It also never follows a link at the path's last
//!   component; it reports the link and its target instead.
//! - Paths cross into Win32 as NUL-terminated UTF-16; an interior NUL is
//!   refused (`wide::to_wide`).

use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

/// Where a file lives: its volume's serial number and its 128-bit id there.
/// Equal identities are the same file, whatever names lead to it (a hard link,
/// a rename within the volume). On FAT and exFAT the id comes from a directory
/// entry's position and is reused, which callers must allow for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FileIdentity {
    pub volume_serial: u64,
    pub file_id: [u8; 16],
}

impl FileIdentity {
    /// The 24 bytes to hash: the serial (little-endian) then the id.
    pub fn to_bytes(&self) -> [u8; 24] {
        let mut bytes = [0u8; 24];
        bytes[..8].copy_from_slice(&self.volume_serial.to_le_bytes());
        bytes[8..].copy_from_slice(&self.file_id);
        bytes
    }
}

/// What a path's text says it is, before anything opens it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PathKind {
    /// `C:\…`, `C:/…` or `\\?\C:\…`.
    Drive,
    /// `\\server\share\…`, `//server/share/…`, `\\?\UNC\…`, `\\.\UNC\…`.
    Unc,
    /// `\\?\Volume{…}\…`: a local volume by its GUID.
    VolumeGuid,
    /// Any other `\\?\…`, `\\.\…` or `\??\…` path: a device or object name.
    Device,
    /// `\…` or `/…`: the root of whatever drive is current.
    Rooted,
    /// Everything else, including drive-relative `C:file`.
    Relative,
}

fn starts_with_ignore_case(text: &str, prefix: &str) -> bool {
    text.len() >= prefix.len()
        && text.as_bytes()[..prefix.len()].eq_ignore_ascii_case(prefix.as_bytes())
}

fn is_drive_prefix(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

/// The kind of path `path`'s text names. Lexical only: no file is touched.
pub fn path_kind(path: &Path) -> PathKind {
    let text = path.to_string_lossy();
    for verbatim in [r"\\?\", r"\\.\", r"\??\"] {
        if let Some(rest) = text.strip_prefix(verbatim) {
            if starts_with_ignore_case(rest, r"UNC\") {
                return PathKind::Unc;
            }
            if is_drive_prefix(rest)
                && (rest.len() == 2 || rest.as_bytes()[2] == b'\\' || rest.as_bytes()[2] == b'/')
            {
                return PathKind::Drive;
            }
            if starts_with_ignore_case(rest, "Volume{") {
                return PathKind::VolumeGuid;
            }
            return PathKind::Device;
        }
    }
    let bytes = text.as_bytes();
    let separator = |b: u8| b == b'\\' || b == b'/';
    if bytes.len() >= 2 && separator(bytes[0]) && separator(bytes[1]) {
        return PathKind::Unc;
    }
    if is_drive_prefix(&text) && bytes.len() >= 3 && separator(bytes[2]) {
        return PathKind::Drive;
    }
    if bytes.first().copied().is_some_and(separator) {
        return PathKind::Rooted;
    }
    PathKind::Relative
}

/// A drive's type, as `GetDriveTypeW` reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DriveType {
    Unknown,
    NoRootDir,
    Removable,
    Fixed,
    /// A network drive (a mapped share, WebDAV).
    Remote,
    CdRom,
    RamDisk,
}

/// The type of the drive `path` is on. A UNC path is `Remote` without any
/// call; a device, rooted or relative path is `Unknown`. Only a drive letter or
/// a volume GUID's root is asked about, and asking opens nothing.
pub fn drive_type(path: &Path) -> DriveType {
    let text = path.to_string_lossy();
    let root = match path_kind(path) {
        PathKind::Unc => return DriveType::Remote,
        PathKind::Device | PathKind::Rooted | PathKind::Relative => return DriveType::Unknown,
        PathKind::Drive => {
            let rest = [r"\\?\", r"\\.\", r"\??\"]
                .iter()
                .find_map(|prefix| text.strip_prefix(prefix))
                .unwrap_or(&text);
            format!(r"{}\", &rest[..2])
        }
        PathKind::VolumeGuid => {
            let start = text.find("Volume{").unwrap_or(0);
            match text[start..].find('}') {
                Some(end) => format!(r"\\?\{}\", &text[start..start + end + 1]),
                None => return DriveType::Unknown,
            }
        }
    };
    imp::drive_type_of_root(&root)
}

/// What kind of link a reparse point is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LinkKind {
    /// `IO_REPARSE_TAG_SYMLINK`.
    Symlink,
    /// `IO_REPARSE_TAG_MOUNT_POINT`: a junction, or a volume mount point.
    Junction,
    /// `IO_REPARSE_TAG_APPEXECLINK`: an app execution alias (the files in
    /// `%LOCALAPPDATA%\Microsoft\WindowsApps`). `CreateProcessW` starts the
    /// program its data names, which may be a batch file or any other path,
    /// so the chat core never follows, opens or starts one (spec §22.6 X2b,
    /// LR3a, X2c). Its target is not read: it is refused, not judged.
    AppExecLink,
    /// Any other reparse tag (a cloud placeholder, a deduplicated or
    /// compressed file, a third party's tag, …).
    Other,
}

/// `IO_REPARSE_TAG_APPEXECLINK`.
pub const TAG_APPEXECLINK: u32 = 0x8000_001B;
/// `IO_REPARSE_TAG_WOF`: a file compressed by the Windows Overlay Filter
/// (`compact /exe`, CompactOS).
pub const TAG_WOF: u32 = 0x8000_0017;
/// `IO_REPARSE_TAG_DEDUP`: a file whose data Data Deduplication keeps.
pub const TAG_DEDUP: u32 = 0x8000_0013;

/// Is `tag` one whose file is stored differently but is still the file at
/// that path, with nothing redirected: a WOF-compressed or deduplicated file?
/// These are the only reparse points besides symlinks and junctions that a
/// program path may pass through.
pub fn is_storage_only_tag(tag: u32) -> bool {
    matches!(tag, TAG_WOF | TAG_DEDUP)
}

/// A reparse point at a path, read without following it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Link {
    pub tag: u32,
    pub kind: LinkKind,
    /// Where it points. An absolute target is in verbatim form (`\\?\C:\…`,
    /// `\\?\UNC\server\share\…`, `\\?\Volume{…}\`), so [`path_kind`] reads it;
    /// a relative symlink's target is as stored. `None` when the tag carries no
    /// path or the data is malformed.
    pub target: Option<PathBuf>,
    /// A symlink stored relative to its own folder.
    pub relative: bool,
}

/// What [`open_no_follow`] opens.
#[derive(Debug)]
pub struct Opened {
    /// The file, the directory, or the link itself (never its target).
    pub file: File,
    pub is_dir: bool,
    /// Set when the path's last component is a reparse point.
    pub link: Option<Link>,
}

/// The access an [`open_no_follow`] handle asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    /// Attributes only: enough for identity, the final path and link targets,
    /// and allowed on files the reader cannot read.
    Attributes,
    /// Read the content too.
    Read,
}

fn network_refusal(path: &Path) -> Option<io::Error> {
    let kind = path_kind(path);
    let refused = match kind {
        PathKind::Unc | PathKind::Device => true,
        PathKind::Drive | PathKind::VolumeGuid => drive_type(path) == DriveType::Remote,
        PathKind::Rooted | PathKind::Relative => false,
    };
    refused.then(|| {
        io::Error::new(
            io::ErrorKind::PermissionDenied,
            "a network or device path is not opened",
        )
    })
}

/// Open `path` without following a link at its last component, and without
/// ever opening a network or device path.
///
/// Links in earlier components are followed by the system as usual, so a
/// caller that must not follow any link opens each prefix in turn. A directory
/// opens as a directory. When the last component is a reparse point, the
/// handle is to the link itself and [`Opened::link`] says where it points.
pub fn open_no_follow(path: &Path, access: Access) -> io::Result<Opened> {
    if let Some(refusal) = network_refusal(path) {
        return Err(refusal);
    }
    imp::open_no_follow(path, access)
}

/// The identity of an open file (`FILE_ID_INFO`).
pub fn file_identity(file: &File) -> io::Result<FileIdentity> {
    imp::file_identity(file)
}

/// A file's four times (`FILE_BASIC_INFO`), each in 100-nanosecond ticks
/// since 1601-01-01 UTC (a `FILETIME`). `change` moves on any change to the
/// file's data or metadata, including a program setting `last_write` back,
/// which `last_write` alone cannot show (spec §8.2, the checkpoint's cache).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FileTimes {
    pub creation: i64,
    pub last_access: i64,
    pub last_write: i64,
    pub change: i64,
}

/// The times of an open file (`FILE_BASIC_INFO`, through
/// `GetFileInformationByHandleEx`). Elsewhere: the metadata's times, with
/// the status-change time as `change`.
pub fn file_times(file: &File) -> io::Result<FileTimes> {
    imp::file_times(file)
}

/// The attributes of an open file (`FILE_ATTRIBUTE_*`, from
/// `FILE_ATTRIBUTE_TAG_INFO`): read-only, directory, reparse point and the
/// rest, as Python's `stat` reads them on Windows for `st_mode` (the writer
/// lease's checkout identity, spec §6.5). Elsewhere: `0x10` for a folder,
/// `0x1` for a file without write permission.
pub fn file_attributes(file: &File) -> io::Result<u32> {
    imp::file_attributes(file)
}

/// The path an open handle names, with every link resolved and every
/// component's long name (`GetFinalPathNameByHandleW`, normalised, with a drive
/// letter). On Windows it is in verbatim form (`\\?\C:\…`).
pub fn final_path(file: &File) -> io::Result<PathBuf> {
    imp::final_path(file)
}

/// The name of the file system an open file is on (`NTFS`, `ReFS`, `FAT32`,
/// `exFAT`, …; `GetVolumeInformationByHandleW`). File ids on FAT and exFAT
/// come from directory-entry positions and are reused, so the chat core keeps
/// no trust across sessions for a folder there (spec §4.1). Empty on other
/// targets.
pub fn file_system_name(file: &File) -> io::Result<String> {
    imp::file_system_name(file)
}

/// `FILE_PERSISTENT_ACLS`: the volume keeps access-control lists (NTFS and
/// ReFS do; FAT and exFAT do not, so a DACL set there is not kept).
pub const FILE_PERSISTENT_ACLS: u32 = 0x0000_0008;

/// The flags of the file system an open file is on
/// (`GetVolumeInformationByHandleW`'s `lpFileSystemFlags`: `FILE_PERSISTENT_ACLS`
/// and its kin). Elsewhere [`FILE_PERSISTENT_ACLS`], since POSIX permissions
/// persist.
pub fn volume_flags(file: &File) -> io::Result<u32> {
    imp::volume_flags(file)
}

/// The SID of `NT AUTHORITY\SYSTEM`.
pub const SYSTEM_SID: &str = "S-1-5-18";
/// The SID of `BUILTIN\Administrators`.
pub const ADMINISTRATORS_SID: &str = "S-1-5-32-544";

/// Make a new folder at `path` that only SYSTEM, Administrators and the
/// current user may open (spec §22.6 LR6a): an explicit, protected DACL of
/// three full-control ACEs, so nothing is inherited from the parent folder,
/// whatever the parent grants. This is what CPython's `tempfile.mkdtemp`
/// applies on Windows. `AlreadyExists` when the name is taken; a network or
/// device path is refused by its text before anything is created.
pub fn create_private_dir(path: &Path) -> io::Result<()> {
    if let Some(refusal) = network_refusal(path) {
        return Err(refusal);
    }
    imp::create_private_dir(path)
}

/// Create the new file `path` (never an existing one: `AlreadyExists`) with
/// the same private, protected DACL as [`create_private_dir`], open for
/// writing.
pub fn create_private_file(path: &Path) -> io::Result<File> {
    if let Some(refusal) = network_refusal(path) {
        return Err(refusal);
    }
    imp::create_private_file(path)
}

/// The current process's user, as a SID string (`S-1-5-21-…`).
pub fn current_user_sid() -> io::Result<String> {
    imp::current_user_sid()
}

/// One access-control entry of a DACL, read back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ace {
    /// The trustee, as a SID string.
    pub sid: String,
    /// An access-allowed ACE (else access-denied, or another type).
    pub allow: bool,
    /// The ACE came from a parent (`INHERITED_ACE`).
    pub inherited: bool,
    pub mask: u32,
}

/// A path's DACL, read back with `GetNamedSecurityInfoW`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dacl {
    /// `SE_DACL_PROTECTED`: the parent's inheritable ACEs do not flow in.
    pub protected: bool,
    pub aces: Vec<Ace>,
}

/// Read the DACL of a local file or folder (never a network or device path).
pub fn read_dacl(path: &Path) -> io::Result<Dacl> {
    if let Some(refusal) = network_refusal(path) {
        return Err(refusal);
    }
    imp::read_dacl(path)
}

/// How `ReplaceFileW` failed.
#[derive(Debug)]
pub enum ReplaceError {
    /// 1176 (`ERROR_UNABLE_TO_MOVE_REPLACEMENT`). With a backup name, as here,
    /// both files keep their names: the target is unchanged.
    UnableToMoveReplacement,
    /// 1177 (`ERROR_UNABLE_TO_MOVE_REPLACEMENT_2`). The target now has the
    /// backup's name, and the replacement still has its own: the target's
    /// path is empty until the caller moves one of them back.
    UnableToMoveReplacement2,
    /// Any other failure; no file was changed (the call did not get as far).
    Io(io::Error),
}

impl std::fmt::Display for ReplaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnableToMoveReplacement => f.write_str("the replacement could not be renamed"),
            Self::UnableToMoveReplacement2 => f.write_str("the replacement could not be moved"),
            Self::Io(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ReplaceError {}

/// Replace `target` with `replacement`, keeping the old `target` as `backup`
/// (`ReplaceFileW` with a backup name, so the old bytes are never lost). The
/// target keeps its attributes and its security descriptor.
///
/// All three paths must be on one volume; `backup` must not exist.
pub fn replace_file(target: &Path, replacement: &Path, backup: &Path) -> Result<(), ReplaceError> {
    #[cfg(any(test, feature = "test-support"))]
    {
        if let Some(fault) = seam::take(target) {
            return seam::act(fault, target, backup);
        }
    }
    imp::replace_file(target, replacement, backup)
}

/// Move `from` to `to` (a file or a directory) on one volume, refusing when
/// `to` exists (`AlreadyExists`). Across volumes it fails (`CrossesDevices`)
/// and moves nothing: copying and verifying is the caller's business, and
/// nothing here unlinks the source.
pub fn move_no_replace(from: &Path, to: &Path) -> io::Result<()> {
    #[cfg(any(test, feature = "test-support"))]
    {
        if seam::take_move(to) {
            return Err(io::Error::other("injected move failure (test seam)"));
        }
    }
    imp::move_no_replace(from, to)
}

/// Test seams other crates' tests use: an injected `ReplaceFileW` failure,
/// an injected failure of a move that never replaces, and making a junction. Compiled only for this crate's tests or with the
/// `test-support` feature, which only `[dev-dependencies]` enable.
#[cfg(any(test, feature = "test-support"))]
pub mod seam {
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    use super::ReplaceError;

    /// Which documented partial failure [`super::replace_file`] simulates.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum ReplaceFault {
        /// 1176: nothing moves; the error is returned.
        UnableToMoveReplacement,
        /// 1177: the target moves to the backup name; the error is returned.
        UnableToMoveReplacement2,
    }

    static FAULTS: Mutex<Vec<(PathBuf, ReplaceFault)>> = Mutex::new(Vec::new());
    static MOVE_FAULTS: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

    /// The next [`super::move_no_replace`] onto exactly `to` fails (an error
    /// that is not a sharing violation, so no caller retries it), and moves
    /// nothing. Inject it twice to fail two moves.
    pub fn inject_move_fault(to: &Path) {
        MOVE_FAULTS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(to.to_path_buf());
    }

    pub(super) fn take_move(to: &Path) -> bool {
        let mut faults = MOVE_FAULTS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match faults.iter().position(|path| path == to) {
            Some(at) => {
                faults.remove(at);
                true
            }
            None => false,
        }
    }

    /// The next [`super::replace_file`] of exactly `target` fails as `fault`
    /// says, leaving the files as the real failure would.
    pub fn inject_replace_fault(target: &Path, fault: ReplaceFault) {
        FAULTS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push((target.to_path_buf(), fault));
    }

    pub(super) fn take(target: &Path) -> Option<ReplaceFault> {
        let mut faults = FAULTS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let at = faults.iter().position(|(path, _)| path == target)?;
        Some(faults.remove(at).1)
    }

    pub(super) fn act(
        fault: ReplaceFault,
        target: &Path,
        backup: &Path,
    ) -> Result<(), ReplaceError> {
        match fault {
            ReplaceFault::UnableToMoveReplacement => Err(ReplaceError::UnableToMoveReplacement),
            ReplaceFault::UnableToMoveReplacement2 => {
                match super::move_no_replace(target, backup) {
                    Ok(()) => Err(ReplaceError::UnableToMoveReplacement2),
                    Err(error) => Err(ReplaceError::Io(error)),
                }
            }
        }
    }

    /// Make `link` (a new, empty directory) a junction to the absolute local
    /// directory `target`.
    pub fn create_junction(link: &Path, target: &Path) -> std::io::Result<()> {
        super::imp::create_junction(link, target)
    }

    /// Make `link` (a new path) an app execution alias whose data names
    /// `target`, as the Store's `WindowsApps` aliases do
    /// (`IO_REPARSE_TAG_APPEXECLINK`, version 3: a package id, an app user
    /// model id, the target path and an app type, each NUL-terminated). No
    /// privilege is needed: the caller owns the new, empty file.
    pub fn create_app_exec_link(link: &Path, target: &Path) -> std::io::Result<()> {
        super::imp::create_app_exec_link(link, target)
    }

    /// Make `path` (a new file) a reparse point with a Microsoft `tag` and
    /// `data`, for a tag no filter handles: a reparse point that is neither a
    /// link nor storage-only, which a links-only walk must refuse.
    pub fn create_reparse_file(path: &Path, tag: u32, data: &[u8]) -> std::io::Result<()> {
        super::imp::create_reparse_file(path, tag, data)
    }

    /// The path an open handle names in the object manager's form
    /// (`\Device\HarddiskVolume3\…`; `VOLUME_NAME_NT`), so a test can make a
    /// link whose target is a device path to a local file
    /// (`\?\GLOBALROOT\Device\…`): a target that is not a drive path, which
    /// a link check must refuse before it follows it, and which opens nothing
    /// on the network if a regression follows it.
    pub fn nt_path(file: &std::fs::File) -> std::io::Result<PathBuf> {
        super::imp::nt_path(file)
    }
}

/// Read a reparse point's data (`REPARSE_DATA_BUFFER`) into a [`Link`]. Every
/// offset is checked against the buffer, so malformed data gives `target: None`.
#[cfg_attr(not(windows), allow(dead_code))]
fn parse_reparse(data: &[u8]) -> Option<Link> {
    const TAG_SYMLINK: u32 = 0xA000_000C;
    const TAG_MOUNT_POINT: u32 = 0xA000_0003;
    const SYMLINK_FLAG_RELATIVE: u32 = 1;
    let u16_at = |at: usize| -> Option<u16> {
        Some(u16::from_le_bytes(data.get(at..at + 2)?.try_into().ok()?))
    };
    let u32_at = |at: usize| -> Option<u32> {
        Some(u32::from_le_bytes(data.get(at..at + 4)?.try_into().ok()?))
    };
    let tag = u32_at(0)?;
    let (kind, path_buffer, relative) = match tag {
        TAG_SYMLINK => (
            LinkKind::Symlink,
            20,
            u32_at(16).is_some_and(|flags| flags & SYMLINK_FLAG_RELATIVE != 0),
        ),
        TAG_MOUNT_POINT => (LinkKind::Junction, 16, false),
        TAG_APPEXECLINK => {
            return Some(Link {
                tag,
                kind: LinkKind::AppExecLink,
                target: None,
                relative: false,
            });
        }
        _ => {
            return Some(Link {
                tag,
                kind: LinkKind::Other,
                target: None,
                relative: false,
            });
        }
    };
    let substitute = (|| {
        let offset = usize::from(u16_at(8)?);
        let length = usize::from(u16_at(10)?);
        let start = path_buffer + offset;
        let bytes = data.get(start..start.checked_add(length)?)?;
        if bytes.len() % 2 != 0 {
            return None;
        }
        let units: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        Some(String::from_utf16_lossy(&units))
    })();
    let target = substitute.map(|name| {
        // The NT prefix `\??\` reads as the verbatim prefix `\\?\`.
        match name.strip_prefix(r"\??\") {
            Some(rest) => PathBuf::from(format!(r"\\?\{rest}")),
            None => PathBuf::from(name),
        }
    });
    Some(Link {
        tag,
        kind,
        target,
        relative,
    })
}

#[cfg(windows)]
mod imp {
    use std::ffi::c_void;
    use std::fs::File;
    use std::io;
    use std::mem::{MaybeUninit, size_of};
    use std::os::windows::io::{AsRawHandle, FromRawHandle};
    use std::path::{Path, PathBuf};
    use std::ptr;

    use std::os::windows::io::{OwnedHandle, RawHandle};

    use windows_sys::Win32::Foundation::{
        ERROR_UNABLE_TO_MOVE_REPLACEMENT, ERROR_UNABLE_TO_MOVE_REPLACEMENT_2, GENERIC_READ,
        GENERIC_WRITE, GetLastError, HANDLE, INVALID_HANDLE_VALUE, LocalFree,
    };
    use windows_sys::Win32::Security::Authorization::{
        ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW,
        GetNamedSecurityInfoW, SDDL_REVISION_1, SE_FILE_OBJECT,
    };
    use windows_sys::Win32::Security::{
        ACCESS_ALLOWED_ACE, ACE_HEADER, ACL, ACL_SIZE_INFORMATION, AclSizeInformation,
        DACL_SECURITY_INFORMATION, GetAce, GetAclInformation, GetSecurityDescriptorControl,
        GetTokenInformation, INHERITED_ACE, PSECURITY_DESCRIPTOR, PSID, SE_DACL_PROTECTED,
        SECURITY_ATTRIBUTES, SECURITY_DESCRIPTOR_CONTROL, TOKEN_QUERY, TOKEN_USER, TokenUser,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CREATE_NEW, CreateDirectoryW, CreateFileW, FILE_ATTRIBUTE_DIRECTORY, FILE_ATTRIBUTE_NORMAL,
        FILE_ATTRIBUTE_REPARSE_POINT, FILE_ATTRIBUTE_TAG_INFO, FILE_FLAG_BACKUP_SEMANTICS,
        FILE_FLAG_OPEN_REPARSE_POINT, FILE_ID_INFO, FILE_NAME_NORMALIZED, FILE_READ_ATTRIBUTES,
        FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, FileAttributeTagInfo, FileIdInfo,
        GetDriveTypeW, GetFileInformationByHandleEx, GetFinalPathNameByHandleW,
        GetVolumeInformationByHandleW, MAXIMUM_REPARSE_DATA_BUFFER_SIZE, MOVEFILE_WRITE_THROUGH,
        MoveFileExW, OPEN_EXISTING, ReplaceFileW, SYNCHRONIZE, VOLUME_NAME_DOS,
    };
    // A separate import, so the one above stays the line the never-delete
    // guard lists for ReplaceFileW.
    use windows_sys::Win32::Storage::FileSystem::{FILE_BASIC_INFO, FileBasicInfo};
    use windows_sys::Win32::System::IO::DeviceIoControl;
    use windows_sys::Win32::System::Ioctl::FSCTL_GET_REPARSE_POINT;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};
    use windows_sys::core::PWSTR;

    use super::{
        Access, Ace, Dacl, DriveType, FileIdentity, FileTimes, Link, Opened, ReplaceError,
        parse_reparse,
    };
    use crate::wide::{from_wide, to_wide};

    // GetDriveTypeW's answers (WinBase.h). Their constants live in a windows-sys
    // feature this crate does not take for six numbers.
    const DRIVE_NO_ROOT_DIR: u32 = 1;
    const DRIVE_REMOVABLE: u32 = 2;
    const DRIVE_FIXED: u32 = 3;
    const DRIVE_REMOTE: u32 = 4;
    const DRIVE_CDROM: u32 = 5;
    const DRIVE_RAMDISK: u32 = 6;

    pub(super) fn drive_type_of_root(root: &str) -> DriveType {
        let Ok(wide) = to_wide(root.as_ref()) else {
            return DriveType::Unknown;
        };
        // SAFETY: `wide` is a NUL-terminated UTF-16 string that outlives the call;
        // GetDriveTypeW only reads it.
        let answer = unsafe { GetDriveTypeW(wide.as_ptr()) };
        match answer {
            DRIVE_NO_ROOT_DIR => DriveType::NoRootDir,
            DRIVE_REMOVABLE => DriveType::Removable,
            DRIVE_FIXED => DriveType::Fixed,
            DRIVE_REMOTE => DriveType::Remote,
            DRIVE_CDROM => DriveType::CdRom,
            DRIVE_RAMDISK => DriveType::RamDisk,
            _ => DriveType::Unknown,
        }
    }

    /// Open with `FILE_FLAG_OPEN_REPARSE_POINT` and `FILE_FLAG_BACKUP_SEMANTICS`
    /// (so a directory opens too), sharing read, write and delete so the open
    /// blocks no other program.
    fn open_raw(path: &Path, access: u32) -> io::Result<File> {
        let wide = to_wide(path.as_os_str())?;
        // SAFETY: `wide` is NUL-terminated and outlives the call; the security
        // attributes and template are null, which CreateFileW allows.
        let handle = unsafe {
            CreateFileW(
                wide.as_ptr(),
                access,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                ptr::null(),
                OPEN_EXISTING,
                FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
                ptr::null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `handle` is a valid handle CreateFileW just returned, owned by
        // nothing else; the File takes ownership and closes it once.
        Ok(unsafe { File::from_raw_handle(handle) })
    }

    pub(super) fn open_no_follow(path: &Path, access: Access) -> io::Result<Opened> {
        let rights = match access {
            Access::Attributes => FILE_READ_ATTRIBUTES | SYNCHRONIZE,
            Access::Read => GENERIC_READ | SYNCHRONIZE,
        };
        let file = open_raw(path, rights)?;
        let mut info = MaybeUninit::<FILE_ATTRIBUTE_TAG_INFO>::zeroed();
        // SAFETY: the handle is open; `info` is a writable buffer of exactly the
        // size passed, of the type the FileAttributeTagInfo class fills.
        let ok = unsafe {
            GetFileInformationByHandleEx(
                file.as_raw_handle(),
                FileAttributeTagInfo,
                info.as_mut_ptr().cast::<c_void>(),
                size_of::<FILE_ATTRIBUTE_TAG_INFO>() as u32,
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the call succeeded, so it wrote the whole structure (and a
        // zeroed one is a valid value of this plain-integer struct anyway).
        let info = unsafe { info.assume_init() };
        let is_dir = info.FileAttributes & FILE_ATTRIBUTE_DIRECTORY != 0;
        let link = if info.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            Some(read_link(&file, info.ReparseTag)?)
        } else {
            None
        };
        Ok(Opened { file, is_dir, link })
    }

    fn read_link(file: &File, tag: u32) -> io::Result<Link> {
        let mut buffer = vec![0u8; MAXIMUM_REPARSE_DATA_BUFFER_SIZE as usize];
        let mut returned = 0u32;
        // SAFETY: the handle is open; the output buffer is writable for the
        // length passed; no input buffer and no OVERLAPPED (a synchronous call).
        let ok = unsafe {
            DeviceIoControl(
                file.as_raw_handle(),
                FSCTL_GET_REPARSE_POINT,
                ptr::null(),
                0,
                buffer.as_mut_ptr().cast::<c_void>(),
                buffer.len() as u32,
                &mut returned,
                ptr::null_mut(),
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        buffer.truncate(returned as usize);
        Ok(parse_reparse(&buffer).unwrap_or(Link {
            tag,
            kind: if tag == super::TAG_APPEXECLINK {
                super::LinkKind::AppExecLink
            } else {
                super::LinkKind::Other
            },
            target: None,
            relative: false,
        }))
    }

    pub(super) fn file_attributes(file: &File) -> io::Result<u32> {
        let mut info = MaybeUninit::<FILE_ATTRIBUTE_TAG_INFO>::zeroed();
        // SAFETY: the handle is open; `info` is a writable buffer of exactly the
        // size passed, of the type the FileAttributeTagInfo class fills.
        let ok = unsafe {
            GetFileInformationByHandleEx(
                file.as_raw_handle(),
                FileAttributeTagInfo,
                info.as_mut_ptr().cast::<c_void>(),
                size_of::<FILE_ATTRIBUTE_TAG_INFO>() as u32,
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the call succeeded and filled the plain-integer structure.
        let info = unsafe { info.assume_init() };
        Ok(info.FileAttributes)
    }

    pub(super) fn file_times(file: &File) -> io::Result<FileTimes> {
        let mut info = MaybeUninit::<FILE_BASIC_INFO>::zeroed();
        // SAFETY: the handle is open; `info` is a writable buffer of exactly the
        // size passed, of the type the FileBasicInfo class fills.
        let ok = unsafe {
            GetFileInformationByHandleEx(
                file.as_raw_handle(),
                FileBasicInfo,
                info.as_mut_ptr().cast::<c_void>(),
                size_of::<FILE_BASIC_INFO>() as u32,
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the call succeeded and filled the plain-integer structure.
        let info = unsafe { info.assume_init() };
        Ok(FileTimes {
            creation: info.CreationTime,
            last_access: info.LastAccessTime,
            last_write: info.LastWriteTime,
            change: info.ChangeTime,
        })
    }

    pub(super) fn file_identity(file: &File) -> io::Result<FileIdentity> {
        let mut info = MaybeUninit::<FILE_ID_INFO>::zeroed();
        // SAFETY: the handle is open; `info` is a writable buffer of exactly the
        // size passed, of the type the FileIdInfo class fills.
        let ok = unsafe {
            GetFileInformationByHandleEx(
                file.as_raw_handle(),
                FileIdInfo,
                info.as_mut_ptr().cast::<c_void>(),
                size_of::<FILE_ID_INFO>() as u32,
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the call succeeded and filled the plain-integer structure.
        let info = unsafe { info.assume_init() };
        Ok(FileIdentity {
            volume_serial: info.VolumeSerialNumber,
            file_id: info.FileId.Identifier,
        })
    }

    pub(super) fn file_system_name(file: &File) -> io::Result<String> {
        // MAX_PATH + 1, the size the documentation asks for.
        let mut name = [0u16; 261];
        // SAFETY: the handle is open; the file-system-name buffer is writable
        // for the length passed, in UTF-16 units; every other out-pointer is
        // null with a zero size, which the API allows.
        let ok = unsafe {
            GetVolumeInformationByHandleW(
                file.as_raw_handle(),
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                name.as_mut_ptr(),
                name.len() as u32,
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        let end = name
            .iter()
            .position(|unit| *unit == 0)
            .unwrap_or(name.len());
        Ok(String::from_utf16_lossy(&name[..end]))
    }

    pub(super) fn volume_flags(file: &File) -> io::Result<u32> {
        let mut flags = 0u32;
        // SAFETY: the handle is open; the flags out-pointer is writable; every
        // other buffer is null with a zero size, which the API allows.
        let ok = unsafe {
            GetVolumeInformationByHandleW(
                file.as_raw_handle(),
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
                &mut flags,
                ptr::null_mut(),
                0,
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(flags)
    }

    /// A security descriptor made from SDDL, freed with `LocalFree` on drop.
    struct Descriptor(PSECURITY_DESCRIPTOR);

    impl Descriptor {
        fn from_sddl(sddl: &str) -> io::Result<Self> {
            let wide = to_wide(sddl.as_ref())?;
            let mut descriptor: PSECURITY_DESCRIPTOR = ptr::null_mut();
            // SAFETY: `wide` is NUL-terminated and outlives the call; the
            // out-pointer is writable; the size pointer may be null.
            let ok = unsafe {
                ConvertStringSecurityDescriptorToSecurityDescriptorW(
                    wide.as_ptr(),
                    SDDL_REVISION_1,
                    &mut descriptor,
                    ptr::null_mut(),
                )
            };
            if ok == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(Self(descriptor))
        }

        fn attributes(&self) -> SECURITY_ATTRIBUTES {
            SECURITY_ATTRIBUTES {
                nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
                lpSecurityDescriptor: self.0,
                bInheritHandle: 0,
            }
        }
    }

    impl Drop for Descriptor {
        fn drop(&mut self) {
            // SAFETY: the descriptor came from
            // ConvertStringSecurityDescriptorToSecurityDescriptorW or
            // GetNamedSecurityInfoW, both of which ask for LocalFree; it is
            // freed once.
            unsafe { LocalFree(self.0) };
        }
    }

    /// SYSTEM, Administrators and `user`, full control, protected; `inherit`
    /// adds object and container inheritance (for a folder).
    fn private_sddl(user: &str, inherit: bool) -> String {
        let flags = if inherit { "OICI" } else { "" };
        format!("D:P(A;{flags};FA;;;SY)(A;{flags};FA;;;BA)(A;{flags};FA;;;{user})")
    }

    pub(super) fn current_user_sid() -> io::Result<String> {
        let mut token: HANDLE = ptr::null_mut();
        // SAFETY: GetCurrentProcess returns a pseudo-handle that needs no
        // closing; the out-pointer is writable.
        let ok = unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: OpenProcessToken succeeded, so `token` is an open handle
        // owned by nothing else; the OwnedHandle closes it once.
        let token = unsafe { OwnedHandle::from_raw_handle(token as RawHandle) };
        let mut needed = 0u32;
        // SAFETY: a null buffer of size 0 asks for the size needed; that call
        // fails by design (ERROR_INSUFFICIENT_BUFFER).
        unsafe {
            GetTokenInformation(
                token.as_raw_handle(),
                TokenUser,
                ptr::null_mut(),
                0,
                &mut needed,
            )
        };
        if needed == 0 {
            return Err(io::Error::last_os_error());
        }
        // u64 units keep the buffer aligned for TOKEN_USER.
        let mut buffer = vec![0u64; (needed as usize).div_ceil(8)];
        // SAFETY: the buffer is writable for `needed` bytes and aligned for
        // TOKEN_USER; the token is open with TOKEN_QUERY.
        let ok = unsafe {
            GetTokenInformation(
                token.as_raw_handle(),
                TokenUser,
                buffer.as_mut_ptr().cast::<c_void>(),
                needed,
                &mut needed,
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the call filled a TOKEN_USER at the buffer's start; its SID
        // points into the same buffer, which outlives `sid_string`.
        let sid = unsafe { (*buffer.as_ptr().cast::<TOKEN_USER>()).User.Sid };
        sid_string(sid)
    }

    /// A SID as its string form.
    fn sid_string(sid: PSID) -> io::Result<String> {
        let mut text: PWSTR = ptr::null_mut();
        // SAFETY: `sid` is a valid SID for the duration of the call; the
        // out-pointer is writable. The string is one LocalAlloc block.
        let ok = unsafe { ConvertSidToStringSidW(sid, &mut text) };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut length = 0usize;
        // SAFETY: the string is NUL-terminated; it is read up to its NUL.
        while unsafe { *text.add(length) } != 0 {
            length += 1;
        }
        // SAFETY: `length` units were just read as part of this string.
        let units = unsafe { std::slice::from_raw_parts(text, length) };
        let out = String::from_utf16_lossy(units);
        // SAFETY: the block came from ConvertSidToStringSidW, which asks for
        // LocalFree; it is freed once and not used after.
        unsafe { LocalFree(text.cast()) };
        Ok(out)
    }

    pub(super) fn create_private_dir(path: &Path) -> io::Result<()> {
        let descriptor = Descriptor::from_sddl(&private_sddl(&current_user_sid()?, true))?;
        let attributes = descriptor.attributes();
        let wide = to_wide(path.as_os_str())?;
        // SAFETY: `wide` is NUL-terminated and outlives the call; the
        // attributes and the descriptor they point at are alive.
        let ok = unsafe { CreateDirectoryW(wide.as_ptr(), &attributes) };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    pub(super) fn create_private_file(path: &Path) -> io::Result<File> {
        let descriptor = Descriptor::from_sddl(&private_sddl(&current_user_sid()?, false))?;
        let attributes = descriptor.attributes();
        let wide = to_wide(path.as_os_str())?;
        // SAFETY: `wide` is NUL-terminated and outlives the call; the
        // attributes and the descriptor they point at are alive; CREATE_NEW
        // never opens an existing file; no template.
        let handle = unsafe {
            CreateFileW(
                wide.as_ptr(),
                GENERIC_WRITE,
                0,
                &attributes,
                CREATE_NEW,
                FILE_ATTRIBUTE_NORMAL,
                ptr::null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: a valid handle CreateFileW just returned, owned by nothing
        // else; the File closes it once.
        Ok(unsafe { File::from_raw_handle(handle) })
    }

    pub(super) fn read_dacl(path: &Path) -> io::Result<Dacl> {
        // The access-allowed and access-denied ACE types (winnt.h).
        const ACCESS_ALLOWED_ACE_TYPE: u8 = 0;
        const ACCESS_DENIED_ACE_TYPE: u8 = 1;
        let wide = to_wide(path.as_os_str())?;
        let mut dacl: *mut ACL = ptr::null_mut();
        let mut descriptor: PSECURITY_DESCRIPTOR = ptr::null_mut();
        // SAFETY: `wide` is NUL-terminated and outlives the call; the DACL and
        // descriptor out-pointers are writable; the others are null, which the
        // API allows when that information is not asked for.
        let status = unsafe {
            GetNamedSecurityInfoW(
                wide.as_ptr(),
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION,
                ptr::null_mut(),
                ptr::null_mut(),
                &mut dacl,
                ptr::null_mut(),
                &mut descriptor,
            )
        };
        if status != 0 {
            return Err(io::Error::from_raw_os_error(status as i32));
        }
        // Freed on every path below.
        let descriptor = Descriptor(descriptor);
        let mut control: SECURITY_DESCRIPTOR_CONTROL = 0;
        let mut revision = 0u32;
        // SAFETY: the descriptor is the one the call returned; both
        // out-pointers are writable.
        let ok = unsafe { GetSecurityDescriptorControl(descriptor.0, &mut control, &mut revision) };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut aces = Vec::new();
        if !dacl.is_null() {
            let mut size = ACL_SIZE_INFORMATION::default();
            // SAFETY: `dacl` points into the live descriptor; `size` is the
            // structure this class fills, of exactly the size passed.
            let ok = unsafe {
                GetAclInformation(
                    dacl,
                    (&mut size as *mut ACL_SIZE_INFORMATION).cast::<c_void>(),
                    size_of::<ACL_SIZE_INFORMATION>() as u32,
                    AclSizeInformation,
                )
            };
            if ok == 0 {
                return Err(io::Error::last_os_error());
            }
            for index in 0..size.AceCount {
                let mut ace: *mut c_void = ptr::null_mut();
                // SAFETY: `index` is below the ACL's count; the out-pointer is
                // writable; the ACE points into the live descriptor.
                if unsafe { GetAce(dacl, index, &mut ace) } == 0 {
                    return Err(io::Error::last_os_error());
                }
                // SAFETY: every ACE starts with an ACE_HEADER.
                let header = unsafe { *ace.cast::<ACE_HEADER>() };
                let allow = header.AceType == ACCESS_ALLOWED_ACE_TYPE;
                if !allow && header.AceType != ACCESS_DENIED_ACE_TYPE {
                    aces.push(Ace {
                        sid: String::new(),
                        allow: false,
                        inherited: u32::from(header.AceFlags) & INHERITED_ACE != 0,
                        mask: 0,
                    });
                    continue;
                }
                // SAFETY: an access-allowed or access-denied ACE has the
                // ACCESS_ALLOWED_ACE layout (the denied one is identical); its
                // SID starts at SidStart, inside the ACE.
                let (mask, sid) = unsafe {
                    let body = ace.cast::<ACCESS_ALLOWED_ACE>();
                    (
                        (*body).Mask,
                        ptr::addr_of_mut!((*body).SidStart).cast::<c_void>(),
                    )
                };
                aces.push(Ace {
                    sid: sid_string(sid)?,
                    allow,
                    inherited: u32::from(header.AceFlags) & INHERITED_ACE != 0,
                    mask,
                });
            }
        }
        Ok(Dacl {
            protected: control & SE_DACL_PROTECTED != 0,
            aces,
        })
    }

    #[cfg(any(test, feature = "test-support"))]
    pub(super) fn nt_path(file: &File) -> io::Result<PathBuf> {
        use windows_sys::Win32::Storage::FileSystem::VOLUME_NAME_NT;
        handle_path(file, FILE_NAME_NORMALIZED | VOLUME_NAME_NT)
    }

    pub(super) fn final_path(file: &File) -> io::Result<PathBuf> {
        handle_path(file, FILE_NAME_NORMALIZED | VOLUME_NAME_DOS)
    }

    fn handle_path(file: &File, flags: u32) -> io::Result<PathBuf> {
        let mut buffer = vec![0u16; 512];
        loop {
            // SAFETY: the handle is open and `buffer` is writable for the length
            // passed, in UTF-16 units.
            let length = unsafe {
                GetFinalPathNameByHandleW(
                    file.as_raw_handle(),
                    buffer.as_mut_ptr(),
                    buffer.len() as u32,
                    flags,
                )
            } as usize;
            if length == 0 {
                return Err(io::Error::last_os_error());
            }
            if length < buffer.len() {
                return Ok(PathBuf::from(from_wide(&buffer[..length])));
            }
            // Too small: `length` is the size needed, terminator included.
            buffer.resize(length + 1, 0);
        }
    }

    pub(super) fn replace_file(
        target: &Path,
        replacement: &Path,
        backup: &Path,
    ) -> Result<(), ReplaceError> {
        let wide = |path: &Path| to_wide(path.as_os_str()).map_err(ReplaceError::Io);
        let (target, replacement, backup) = (wide(target)?, wide(replacement)?, wide(backup)?);
        // SAFETY: the three strings are NUL-terminated and outlive the call; the
        // exclude and reserved arguments are null, as the API requires.
        let ok = unsafe {
            ReplaceFileW(
                target.as_ptr(),
                replacement.as_ptr(),
                backup.as_ptr(),
                0,
                ptr::null(),
                ptr::null(),
            )
        };
        if ok != 0 {
            return Ok(());
        }
        // SAFETY: GetLastError has no preconditions; it reads this thread's code.
        match unsafe { GetLastError() } {
            ERROR_UNABLE_TO_MOVE_REPLACEMENT => Err(ReplaceError::UnableToMoveReplacement),
            ERROR_UNABLE_TO_MOVE_REPLACEMENT_2 => Err(ReplaceError::UnableToMoveReplacement2),
            code => Err(ReplaceError::Io(io::Error::from_raw_os_error(code as i32))),
        }
    }

    pub(super) fn move_no_replace(from: &Path, to: &Path) -> io::Result<()> {
        let (from, to) = (to_wide(from.as_os_str())?, to_wide(to.as_os_str())?);
        // SAFETY: both strings are NUL-terminated and outlive the call. Without
        // MOVEFILE_REPLACE_EXISTING an existing target fails the call, and
        // without MOVEFILE_COPY_ALLOWED a move across volumes fails instead of
        // copying and deleting.
        let ok = unsafe { MoveFileExW(from.as_ptr(), to.as_ptr(), MOVEFILE_WRITE_THROUGH) };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    #[cfg(any(test, feature = "test-support"))]
    pub(super) fn create_app_exec_link(link: &Path, target: &Path) -> io::Result<()> {
        let mut data: Vec<u8> = 3u32.to_le_bytes().to_vec();
        let target = target.to_string_lossy();
        for text in [
            "Lattice.Test_0000000000000",
            "Lattice.Test_0000000000000!App",
            target.as_ref(),
            "0",
        ] {
            for unit in text.encode_utf16().chain([0]) {
                data.extend_from_slice(&unit.to_le_bytes());
            }
        }
        create_reparse_file(link, super::TAG_APPEXECLINK, &data)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub(super) fn create_reparse_file(path: &Path, tag: u32, data: &[u8]) -> io::Result<()> {
        use windows_sys::Win32::System::Ioctl::FSCTL_SET_REPARSE_POINT;

        let length = u16::try_from(data.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "reparse data too long"))?;
        let mut buffer = Vec::with_capacity(8 + data.len());
        buffer.extend_from_slice(&tag.to_le_bytes());
        buffer.extend_from_slice(&length.to_le_bytes());
        buffer.extend_from_slice(&0u16.to_le_bytes());
        buffer.extend_from_slice(data);
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        let file = open_raw(path, GENERIC_WRITE | SYNCHRONIZE)?;
        let mut returned = 0u32;
        // SAFETY: the handle is open for writing; the input buffer is a complete
        // REPARSE_DATA_BUFFER of the length passed; no output buffer and no
        // OVERLAPPED (a synchronous call).
        let ok = unsafe {
            DeviceIoControl(
                file.as_raw_handle(),
                FSCTL_SET_REPARSE_POINT,
                buffer.as_ptr().cast::<c_void>(),
                buffer.len() as u32,
                ptr::null_mut(),
                0,
                &mut returned,
                ptr::null_mut(),
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// `link` must be a new, empty directory; `target` an absolute local path.
    #[cfg(any(test, feature = "test-support"))]
    pub(super) fn create_junction(link: &Path, target: &Path) -> io::Result<()> {
        use windows_sys::Win32::System::Ioctl::FSCTL_SET_REPARSE_POINT;

        let text = target.to_string_lossy();
        let plain = text.strip_prefix(r"\\?\").unwrap_or(&text);
        if !(plain.len() >= 3 && plain.as_bytes()[1] == b':') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a junction's target is an absolute local path",
            ));
        }
        let substitute: Vec<u16> = format!(r"\??\{plain}").encode_utf16().collect();
        let print: Vec<u16> = plain.encode_utf16().collect();
        let sub_bytes = substitute.len() * 2;
        let print_bytes = print.len() * 2;
        // Header (8), the four offsets (8), the names each NUL-terminated.
        let data_length = 8 + sub_bytes + 2 + print_bytes + 2;
        let mut buffer = Vec::with_capacity(8 + data_length);
        buffer.extend_from_slice(&0xA000_0003u32.to_le_bytes());
        buffer.extend_from_slice(&(data_length as u16).to_le_bytes());
        buffer.extend_from_slice(&0u16.to_le_bytes());
        buffer.extend_from_slice(&0u16.to_le_bytes());
        buffer.extend_from_slice(&(sub_bytes as u16).to_le_bytes());
        buffer.extend_from_slice(&((sub_bytes + 2) as u16).to_le_bytes());
        buffer.extend_from_slice(&(print_bytes as u16).to_le_bytes());
        for unit in substitute
            .iter()
            .chain([&0])
            .chain(print.iter())
            .chain([&0])
        {
            buffer.extend_from_slice(&unit.to_le_bytes());
        }
        let file = open_raw(link, GENERIC_WRITE | SYNCHRONIZE)?;
        let mut returned = 0u32;
        // SAFETY: the handle is open for writing; the input buffer is a complete
        // mount-point REPARSE_DATA_BUFFER of the length passed; no output buffer
        // and no OVERLAPPED (a synchronous call).
        let ok = unsafe {
            DeviceIoControl(
                file.as_raw_handle(),
                FSCTL_SET_REPARSE_POINT,
                buffer.as_ptr().cast::<c_void>(),
                buffer.len() as u32,
                ptr::null_mut(),
                0,
                &mut returned,
                ptr::null_mut(),
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}

/// Portable stand-ins, used only by tests on other targets.
#[cfg(not(windows))]
mod imp {
    use std::fs::File;
    use std::io;
    use std::path::{Path, PathBuf};

    use super::{Access, DriveType, FileIdentity, FileTimes, Opened, ReplaceError};

    pub(super) fn drive_type_of_root(_root: &str) -> DriveType {
        DriveType::Fixed
    }

    pub(super) fn open_no_follow(path: &Path, _access: Access) -> io::Result<Opened> {
        let meta = std::fs::symlink_metadata(path)?;
        if meta.file_type().is_symlink() {
            // A symlink itself cannot be opened portably.
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "a link is not opened on this platform",
            ));
        }
        Ok(Opened {
            file: File::open(path)?,
            is_dir: meta.is_dir(),
            link: None,
        })
    }

    pub(super) fn file_attributes(file: &File) -> io::Result<u32> {
        let meta = file.metadata()?;
        let mut attributes = 0;
        if meta.is_dir() {
            attributes |= 0x10;
        }
        if meta.permissions().readonly() {
            attributes |= 0x1;
        }
        Ok(attributes)
    }

    pub(super) fn file_times(file: &File) -> io::Result<FileTimes> {
        use std::os::unix::fs::MetadataExt;
        let meta = file.metadata()?;
        // FILETIME ticks: 100 ns since 1601 (11,644,473,600 s before 1970).
        let ticks =
            |seconds: i64, nanos: i64| (seconds + 11_644_473_600) * 10_000_000 + nanos / 100;
        Ok(FileTimes {
            creation: ticks(meta.ctime(), meta.ctime_nsec()),
            last_access: ticks(meta.atime(), meta.atime_nsec()),
            last_write: ticks(meta.mtime(), meta.mtime_nsec()),
            change: ticks(meta.ctime(), meta.ctime_nsec()),
        })
    }

    pub(super) fn file_identity(file: &File) -> io::Result<FileIdentity> {
        use std::os::unix::fs::MetadataExt;
        let meta = file.metadata()?;
        let mut file_id = [0u8; 16];
        file_id[..8].copy_from_slice(&meta.ino().to_le_bytes());
        Ok(FileIdentity {
            volume_serial: meta.dev(),
            file_id,
        })
    }

    pub(super) fn final_path(file: &File) -> io::Result<PathBuf> {
        use std::os::fd::AsRawFd;
        std::fs::read_link(format!("/proc/self/fd/{}", file.as_raw_fd()))
    }

    pub(super) fn replace_file(
        target: &Path,
        replacement: &Path,
        backup: &Path,
    ) -> Result<(), ReplaceError> {
        std::fs::hard_link(target, backup).map_err(ReplaceError::Io)?;
        std::fs::rename(replacement, target).map_err(ReplaceError::Io)
    }

    pub(super) fn move_no_replace(from: &Path, to: &Path) -> io::Result<()> {
        if std::fs::symlink_metadata(to).is_ok() {
            return Err(io::Error::from(io::ErrorKind::AlreadyExists));
        }
        std::fs::rename(from, to)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub(super) fn nt_path(_file: &File) -> io::Result<PathBuf> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "NT paths exist only on Windows",
        ))
    }

    #[cfg(any(test, feature = "test-support"))]
    pub(super) fn create_reparse_file(_path: &Path, _tag: u32, _data: &[u8]) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "reparse points exist only on Windows",
        ))
    }

    #[cfg(any(test, feature = "test-support"))]
    pub(super) fn create_app_exec_link(_link: &Path, _target: &Path) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "app execution aliases exist only on Windows",
        ))
    }

    #[cfg(any(test, feature = "test-support"))]
    pub(super) fn create_junction(_link: &Path, _target: &Path) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "junctions exist only on Windows",
        ))
    }

    pub(super) fn file_system_name(_file: &File) -> io::Result<String> {
        Ok(String::new())
    }

    pub(super) fn volume_flags(_file: &File) -> io::Result<u32> {
        Ok(super::FILE_PERSISTENT_ACLS)
    }

    pub(super) fn create_private_dir(path: &Path) -> io::Result<()> {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new().mode(0o700).create(path)
    }

    pub(super) fn create_private_file(path: &Path) -> io::Result<File> {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
    }

    pub(super) fn current_user_sid() -> io::Result<String> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "SIDs exist only on Windows",
        ))
    }

    pub(super) fn read_dacl(_path: &Path) -> io::Result<super::Dacl> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "DACLs exist only on Windows",
        ))
    }
}

#[cfg(all(test, windows))]
mod tests {
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    use super::seam::{ReplaceFault, create_junction, inject_replace_fault};
    use super::*;

    /// A directory of this test's own under the system temporary directory,
    /// removed when the test ends. Nothing outside it is touched.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "lattice-sys-{tag}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(std::fs::canonicalize(path).unwrap())
        }

        fn path(&self) -> &Path {
            &self.0
        }

        /// Every file and directory under the scratch folder, relative.
        fn tree(&self) -> BTreeSet<String> {
            fn walk(root: &Path, dir: &Path, out: &mut BTreeSet<String>) {
                for entry in std::fs::read_dir(dir).unwrap() {
                    let entry = entry.unwrap();
                    let path = entry.path();
                    out.insert(path.strip_prefix(root).unwrap().display().to_string());
                    let kind = entry.file_type().unwrap();
                    if kind.is_dir() && !kind.is_symlink() {
                        walk(root, &path, out);
                    }
                }
            }
            let mut out = BTreeSet::new();
            walk(&self.0, &self.0, &mut out);
            out
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            // The test's own temporary folder, and only it.
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn open(path: &Path) -> Opened {
        open_no_follow(path, Access::Read).unwrap()
    }

    #[test]
    fn path_kinds_are_read_from_the_text_alone() {
        let cases = [
            (r"C:\work", PathKind::Drive),
            ("C:/work", PathKind::Drive),
            (r"\\?\C:\work", PathKind::Drive),
            (r"\\?\c:", PathKind::Drive),
            (r"\??\C:\work", PathKind::Drive),
            (r"\\server\share\f", PathKind::Unc),
            ("//server/share/f", PathKind::Unc),
            (r"\\?\UNC\server\share", PathKind::Unc),
            (r"\\?\unc\server\share", PathKind::Unc),
            (r"\\.\UNC\server\share", PathKind::Unc),
            (r"\??\UNC\198.51.100.7\x", PathKind::Unc),
            (
                r"\\?\Volume{0f1e2d3c-0000-0000-0000-000000000000}\",
                PathKind::VolumeGuid,
            ),
            (r"\\.\PhysicalDrive0", PathKind::Device),
            (r"\\.\pipe\x", PathKind::Device),
            (r"\\?\GLOBALROOT\Device\X", PathKind::Device),
            (r"\work", PathKind::Rooted),
            ("/work", PathKind::Rooted),
            ("C:work", PathKind::Relative),
            (r"..\x", PathKind::Relative),
            ("a.txt", PathKind::Relative),
            ("", PathKind::Relative),
        ];
        for (text, kind) in cases {
            assert_eq!(path_kind(Path::new(text)), kind, "{text}");
        }
    }

    #[test]
    fn a_fixed_drive_is_fixed_and_a_unc_path_is_remote_without_a_call() {
        let scratch = Scratch::new("drive");
        assert_eq!(drive_type(scratch.path()), DriveType::Fixed);
        assert_eq!(
            drive_type(Path::new(r"\\198.51.100.7\x")),
            DriveType::Remote
        );
        assert_eq!(drive_type(Path::new(r"\\.\pipe\x")), DriveType::Unknown);
        assert_eq!(drive_type(Path::new("relative")), DriveType::Unknown);
    }

    #[test]
    fn identity_follows_the_file_not_its_name() {
        let scratch = Scratch::new("identity");
        let a = scratch.path().join("a.txt");
        let b = scratch.path().join("b.txt");
        std::fs::write(&a, "a").unwrap();
        std::fs::write(&b, "b").unwrap();
        let id_a = file_identity(&open(&a).file).unwrap();
        assert_eq!(id_a, file_identity(&open(&a).file).unwrap());
        assert_ne!(id_a, file_identity(&open(&b).file).unwrap());
        // A hard link is the same file; a rename within the volume keeps the id.
        let linked = scratch.path().join("linked.txt");
        std::fs::hard_link(&a, &linked).unwrap();
        assert_eq!(id_a, file_identity(&open(&linked).file).unwrap());
        let renamed = scratch.path().join("renamed.txt");
        move_no_replace(&a, &renamed).unwrap();
        assert_eq!(id_a, file_identity(&open(&renamed).file).unwrap());
        assert_ne!(id_a.file_id, [0u8; 16]);
        assert_eq!(&id_a.to_bytes()[..8], &id_a.volume_serial.to_le_bytes());
    }

    #[test]
    fn the_final_path_names_the_real_file_with_its_long_name() {
        let scratch = Scratch::new("final");
        let dir = scratch.path().join("A Long Directory Name");
        std::fs::create_dir(&dir).unwrap();
        let file = dir.join("file.txt");
        std::fs::write(&file, "x").unwrap();
        // Through a directory symlink, the final path is the real one.
        let via = scratch.path().join("via");
        std::os::windows::fs::symlink_dir(&dir, &via).unwrap();
        let through = std::fs::File::open(via.join("file.txt")).unwrap();
        let real = final_path(&through).unwrap();
        assert_eq!(real, file, "{}", real.display());
        assert_eq!(path_kind(&real), PathKind::Drive);
    }

    #[test]
    fn a_plain_file_and_a_directory_open_as_themselves() {
        let scratch = Scratch::new("plain");
        let file = scratch.path().join("f.txt");
        std::fs::write(&file, "x").unwrap();
        let opened = open(&file);
        assert!(!opened.is_dir && opened.link.is_none());
        let dir = open_no_follow(scratch.path(), Access::Attributes).unwrap();
        assert!(dir.is_dir && dir.link.is_none());
        assert_eq!(final_path(&dir.file).unwrap(), scratch.path());
    }

    #[test]
    fn a_file_symlink_is_opened_as_the_link_and_its_target_is_reported() {
        let scratch = Scratch::new("symlink");
        let target = scratch.path().join("target.txt");
        std::fs::write(&target, "secret-looking but local").unwrap();
        let absolute = scratch.path().join("absolute.lnk");
        std::os::windows::fs::symlink_file(&target, &absolute).unwrap();
        let opened = open(&absolute);
        let link = opened.link.expect("a link");
        assert_eq!(link.kind, LinkKind::Symlink);
        assert!(!link.relative);
        let reported = link.target.unwrap();
        assert_eq!(path_kind(&reported), PathKind::Drive);
        assert_eq!(reported, target);
        // The handle is the link itself, not its target.
        assert_eq!(final_path(&opened.file).unwrap(), absolute);

        let relative = scratch.path().join("relative.lnk");
        std::os::windows::fs::symlink_file("target.txt", &relative).unwrap();
        let link = open(&relative).link.unwrap();
        assert!(link.relative);
        assert_eq!(link.target.unwrap(), PathBuf::from("target.txt"));
    }

    #[test]
    fn a_junction_is_reported_with_its_target_and_not_followed() {
        let scratch = Scratch::new("junction");
        let target = scratch.path().join("target");
        std::fs::create_dir(&target).unwrap();
        std::fs::write(target.join("inside.txt"), "x").unwrap();
        let junction = scratch.path().join("j");
        std::fs::create_dir(&junction).unwrap();
        create_junction(&junction, &target).unwrap();
        let opened = open_no_follow(&junction, Access::Attributes).unwrap();
        let link = opened.link.expect("a junction");
        assert_eq!(link.kind, LinkKind::Junction);
        assert_eq!(link.target.unwrap(), target);
        assert_eq!(final_path(&opened.file).unwrap(), junction);
        // The system still follows it for an ordinary open below it.
        assert_eq!(
            std::fs::read_to_string(junction.join("inside.txt")).unwrap(),
            "x"
        );
    }

    #[test]
    fn a_link_to_a_network_share_is_read_and_never_followed() {
        let scratch = Scratch::new("unclink");
        let link_path = scratch.path().join("f");
        // TEST-NET-2 (RFC 5737): nothing answers there. Making the link opens
        // nothing; reading it opens only the link.
        std::os::windows::fs::symlink_file(r"\\198.51.100.7\x\f", &link_path).unwrap();
        let started = Instant::now();
        let link = open_no_follow(&link_path, Access::Attributes)
            .unwrap()
            .link
            .unwrap();
        let target = link.target.unwrap();
        assert_eq!(path_kind(&target), PathKind::Unc, "{}", target.display());
        // A connection attempt to an unrouted address would take seconds.
        assert!(started.elapsed() < Duration::from_secs(2));
        // And the target itself is refused by its text, before any open.
        let refused = open_no_follow(&target, Access::Attributes).unwrap_err();
        assert_eq!(refused.kind(), io::ErrorKind::PermissionDenied);
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn network_and_device_paths_are_refused_by_their_text() {
        for text in [
            r"\\198.51.100.7\x\f",
            "//198.51.100.7/x/f",
            r"\\?\UNC\198.51.100.7\x\f",
            r"\\.\PhysicalDrive0",
            r"\\?\GLOBALROOT\Device\Null",
        ] {
            let error = open_no_follow(Path::new(text), Access::Attributes).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::PermissionDenied, "{text}");
        }
    }

    #[test]
    fn malformed_reparse_data_gives_no_target() {
        assert!(parse_reparse(&[]).is_none());
        // A symlink header whose offsets point past the buffer.
        let mut data = vec![0u8; 24];
        data[..4].copy_from_slice(&0xA000_000Cu32.to_le_bytes());
        data[8..10].copy_from_slice(&100u16.to_le_bytes());
        data[10..12].copy_from_slice(&8u16.to_le_bytes());
        let link = parse_reparse(&data).unwrap();
        assert_eq!(link.kind, LinkKind::Symlink);
        assert!(link.target.is_none());
        // An unknown tag is reported as Other, with no path.
        let other = parse_reparse(&0x9000_001Au32.to_le_bytes()).unwrap();
        assert_eq!(other.kind, LinkKind::Other);
        assert!(other.target.is_none());
        // An app execution alias is named as one, and its target is not read.
        let alias = parse_reparse(&0x8000_001Bu32.to_le_bytes()).unwrap();
        assert_eq!(alias.kind, LinkKind::AppExecLink);
        assert!(alias.target.is_none());
        assert!(is_storage_only_tag(TAG_WOF) && is_storage_only_tag(TAG_DEDUP));
        assert!(!is_storage_only_tag(TAG_APPEXECLINK) && !is_storage_only_tag(0x9000_001A));
    }

    #[test]
    fn replace_file_keeps_the_old_bytes_under_the_backup_name() {
        let scratch = Scratch::new("replace");
        let target = scratch.path().join("t.txt");
        let replacement = scratch.path().join(".t.txt.lattice-1-0.tmp");
        let backup = scratch.path().join(".t.txt.lattice-bak-1-0");
        std::fs::write(&target, "old").unwrap();
        std::fs::write(&replacement, "new").unwrap();
        let id_before = file_identity(&open(&target).file).unwrap();
        replace_file(&target, &replacement, &backup).unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "new");
        assert_eq!(std::fs::read_to_string(&backup).unwrap(), "old");
        assert!(
            !replacement.exists(),
            "the replacement took the target's name"
        );
        // The target path now names the replacement's file; the old file lives on
        // as the backup.
        assert_eq!(file_identity(&open(&backup).file).unwrap(), id_before);
        assert_eq!(
            scratch.tree(),
            ["t.txt", ".t.txt.lattice-bak-1-0"]
                .into_iter()
                .map(String::from)
                .collect()
        );
    }

    #[test]
    fn an_injected_move_fault_fails_one_move_onto_its_path_and_moves_nothing() {
        let scratch = Scratch::new("movefault");
        let from = scratch.path().join("from.txt");
        let to = scratch.path().join("to.txt");
        std::fs::write(&from, "bytes").unwrap();
        let before = scratch.tree();
        super::seam::inject_move_fault(&to);
        let error = move_no_replace(&from, &to).unwrap_err();
        assert_eq!(error.raw_os_error(), None, "not a sharing violation");
        assert_eq!(scratch.tree(), before);
        // Taken once: the next move is real.
        move_no_replace(&from, &to).unwrap();
        assert_eq!(std::fs::read_to_string(&to).unwrap(), "bytes");
    }

    #[test]
    fn an_injected_1176_leaves_both_files_under_their_names() {
        let scratch = Scratch::new("r1176");
        let target = scratch.path().join("t.txt");
        let replacement = scratch.path().join("r.tmp");
        let backup = scratch.path().join("b.bak");
        std::fs::write(&target, "old").unwrap();
        std::fs::write(&replacement, "new").unwrap();
        let before = scratch.tree();
        inject_replace_fault(&target, ReplaceFault::UnableToMoveReplacement);
        let error = replace_file(&target, &replacement, &backup).unwrap_err();
        assert!(
            matches!(error, ReplaceError::UnableToMoveReplacement),
            "{error}"
        );
        assert_eq!(scratch.tree(), before);
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "old");
        assert_eq!(std::fs::read_to_string(&replacement).unwrap(), "new");
        // The fault was taken once: the next call is real.
        replace_file(&target, &replacement, &backup).unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "new");
    }

    #[test]
    fn an_injected_1177_moves_the_target_to_the_backup_and_loses_nothing() {
        let scratch = Scratch::new("r1177");
        let target = scratch.path().join("t.txt");
        let replacement = scratch.path().join("r.tmp");
        let backup = scratch.path().join("b.bak");
        std::fs::write(&target, "old").unwrap();
        std::fs::write(&replacement, "new").unwrap();
        inject_replace_fault(&target, ReplaceFault::UnableToMoveReplacement2);
        let error = replace_file(&target, &replacement, &backup).unwrap_err();
        assert!(
            matches!(error, ReplaceError::UnableToMoveReplacement2),
            "{error}"
        );
        assert!(!target.exists(), "1177 leaves the target's path empty");
        assert_eq!(std::fs::read_to_string(&backup).unwrap(), "old");
        assert_eq!(std::fs::read_to_string(&replacement).unwrap(), "new");
        // What a caller does next: move the new bytes into place, never replacing.
        move_no_replace(&replacement, &target).unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "new");
    }

    #[test]
    fn a_move_never_replaces_and_moves_directories_too() {
        let scratch = Scratch::new("move");
        let a = scratch.path().join("a.txt");
        let b = scratch.path().join("b.txt");
        std::fs::write(&a, "a").unwrap();
        std::fs::write(&b, "b").unwrap();
        let before = scratch.tree();
        let error = move_no_replace(&a, &b).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(scratch.tree(), before);
        assert_eq!(std::fs::read_to_string(&b).unwrap(), "b");
        let dir = scratch.path().join("d");
        std::fs::create_dir(&dir).unwrap();
        std::fs::write(dir.join("inner.txt"), "i").unwrap();
        let moved = scratch.path().join("removed").join("1");
        std::fs::create_dir(scratch.path().join("removed")).unwrap();
        move_no_replace(&dir, &moved).unwrap();
        assert_eq!(
            std::fs::read_to_string(moved.join("inner.txt")).unwrap(),
            "i"
        );
        assert!(!dir.exists());
        let missing = move_no_replace(&scratch.path().join("nope"), &scratch.path().join("x"));
        assert_eq!(missing.unwrap_err().kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn a_nul_in_a_path_is_refused_before_any_call() {
        let error = open_no_follow(Path::new("a\0b"), Access::Attributes).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    /// The SIDs of a DACL's ACEs, sorted, and whether any is inherited.
    fn trustees(dacl: &Dacl) -> (Vec<String>, bool) {
        let mut sids: Vec<String> = dacl.aces.iter().map(|ace| ace.sid.clone()).collect();
        sids.sort();
        (sids, dacl.aces.iter().any(|ace| ace.inherited))
    }

    /// LR6a (spec §22.6): a private folder, and a private file inside it, carry
    /// an explicit, protected DACL that names SYSTEM, Administrators and the
    /// current user only, each allowed, and inherit nothing, under a parent
    /// whose inheritable ACEs reach a plain folder made beside them (the
    /// positive control). Read back with `GetNamedSecurityInfoW`.
    /// Mutant: `create_private_dir` made with a null security descriptor.
    #[test]
    fn a_private_folder_and_its_file_inherit_nothing_and_name_three_sids() {
        let scratch = Scratch::new("private");
        let plain = scratch.path().join("plain");
        std::fs::create_dir(&plain).unwrap();
        let (_, plain_inherits) = trustees(&read_dacl(&plain).unwrap());
        assert!(
            plain_inherits,
            "the control: a plain folder here inherits its parent's ACEs"
        );
        let user = current_user_sid().unwrap();
        assert!(user.starts_with("S-1-5-"), "{user}");
        let mut expected = vec![
            SYSTEM_SID.to_owned(),
            ADMINISTRATORS_SID.to_owned(),
            user.clone(),
        ];
        expected.sort();

        let private = scratch.path().join("private");
        create_private_dir(&private).unwrap();
        let dacl = read_dacl(&private).unwrap();
        println!("the private folder's DACL: {dacl:?}");
        assert!(
            dacl.protected,
            "protected: the parent's ACEs do not flow in"
        );
        assert!(dacl.aces.iter().all(|ace| ace.allow));
        assert_eq!(trustees(&dacl), (expected.clone(), false));

        let key = private.join("key");
        let mut file = create_private_file(&key).unwrap();
        std::io::Write::write_all(&mut file, b"token\n").unwrap();
        drop(file);
        assert_eq!(std::fs::read(&key).unwrap(), b"token\n");
        let dacl = read_dacl(&key).unwrap();
        println!("the key file's DACL: {dacl:?}");
        assert!(dacl.protected);
        assert_eq!(trustees(&dacl), (expected, false));

        // Neither ever opens or replaces what is there.
        assert_eq!(
            create_private_dir(&private).unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(
            create_private_file(&key).unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(std::fs::read(&key).unwrap(), b"token\n");
        for path in [r"\\198.51.100.7\x\d", r"\\?\GLOBALROOT\Device\Null"] {
            assert_eq!(
                create_private_dir(Path::new(path)).unwrap_err().kind(),
                io::ErrorKind::PermissionDenied,
                "{path}"
            );
        }
    }

    /// The temporary folder's file system is named; on this machine's system
    /// drive it is NTFS (a FAT or exFAT volume could not be made here).
    #[test]
    fn the_file_system_is_named() {
        let opened = open_no_follow(&std::env::temp_dir(), Access::Attributes).unwrap();
        let name = file_system_name(&opened.file).unwrap();
        println!("the temporary folder is on {name}");
        assert!(!name.is_empty());
        assert!(name.chars().all(|c| c.is_ascii_alphanumeric()), "{name}");
    }
}
