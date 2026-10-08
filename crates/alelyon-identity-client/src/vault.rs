//! "Stay signed in": the refresh token kept on this PC between runs, sealed with Windows' DPAPI under the signed-in
//! Windows user (with this crate's own entropy), so another Windows user, or the file copied to another PC, cannot
//! open it. When the person does not ask to stay signed in, nothing should be kept: the session ends with the app.
//! Where DPAPI is not available nothing is kept either; a token is never written in the clear.
//!
//! What this does not protect against: code running as the same Windows user can ask DPAPI to open the file, as this
//! crate does. DPAPI separates users and machines, not programs.

use std::fs;
use std::path::{Path, PathBuf};

/// Mixed into the seal, so another program's DPAPI blob is not mistaken for this one. It is the desktop app's
/// version-1 label: changing it would leave every session already kept unopenable.
const ENTROPY: &[u8] = b"alelyon.centcom.session/v1";

/// Where the sealed session lives: `~/.alelyon/alelyon/session.sealed`, or the file `ALELYON_SESSION_FILE` names.
pub fn path() -> Option<PathBuf> {
    path_or("ALELYON_SESSION_FILE")
}

/// [`path`], with the file named by the environment variable `var` (when it is set and not empty) in place of the
/// default, so an application can keep its own setting's name.
pub fn path_or(var: &str) -> Option<PathBuf> {
    if let Some(p) = std::env::var_os(var).filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(p));
    }
    std::env::var_os("USERPROFILE").map(|h| PathBuf::from(h).join(".alelyon").join("alelyon").join("session.sealed"))
}

/// Seal `secret` and write it, replacing what was there (a whole new file moved over the old).
pub fn keep(at: &Path, secret: &str) -> Result<(), String> {
    let sealed = seal(secret.as_bytes())?;
    if let Some(dir) = at.parent() {
        fs::create_dir_all(dir).map_err(|e| format!("{} could not be made: {e}", dir.display()))?;
    }
    let partial = at.with_extension("sealing");
    fs::write(&partial, &sealed).map_err(|e| format!("the session could not be kept: {e}"))?;
    fs::rename(&partial, at).map_err(|e| format!("the session could not be kept: {e}"))
}

/// The kept secret, if there is one this user can open.
pub fn open(at: &Path) -> Option<String> {
    let sealed = fs::read(at).ok()?;
    String::from_utf8(unseal(&sealed).ok()?).ok()
}

/// Forget the kept session (sign out, or a token the service refused).
pub fn forget(at: &Path) {
    let _ = fs::remove_file(at);
}

#[cfg(windows)]
mod dpapi {
    #[repr(C)]
    pub struct Blob {
        pub len: u32,
        pub data: *mut u8,
    }

    #[link(name = "crypt32")]
    unsafe extern "system" {
        pub fn CryptProtectData(
            input: *const Blob,
            desc: *const u16,
            entropy: *const Blob,
            reserved: *const u8,
            prompt: *const u8,
            flags: u32,
            out: *mut Blob,
        ) -> i32;
        pub fn CryptUnprotectData(
            input: *const Blob,
            desc: *mut *mut u16,
            entropy: *const Blob,
            reserved: *const u8,
            prompt: *const u8,
            flags: u32,
            out: *mut Blob,
        ) -> i32;
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn LocalFree(mem: *mut u8) -> *mut u8;
    }

    /// CRYPTPROTECT_UI_FORBIDDEN: never show a dialog.
    pub const UI_FORBIDDEN: u32 = 0x1;
}

#[cfg(windows)]
fn run(input: &[u8], protect: bool) -> Result<Vec<u8>, String> {
    use dpapi::*;
    let blob_in = Blob { len: input.len() as u32, data: input.as_ptr() as *mut u8 };
    let entropy = Blob { len: ENTROPY.len() as u32, data: ENTROPY.as_ptr() as *mut u8 };
    let mut out = Blob { len: 0, data: std::ptr::null_mut() };
    // SAFETY: the input and entropy blobs point at live slices for the call; on success the system allocates `out`
    // with LocalAlloc, it is copied before LocalFree, and it is freed exactly once.
    unsafe {
        let ok = if protect {
            CryptProtectData(&blob_in, std::ptr::null(), &entropy, std::ptr::null(), std::ptr::null(), UI_FORBIDDEN, &mut out)
        } else {
            CryptUnprotectData(&blob_in, std::ptr::null_mut(), &entropy, std::ptr::null(), std::ptr::null(), UI_FORBIDDEN, &mut out)
        };
        if ok == 0 || out.data.is_null() {
            return Err(
                if protect { "Windows could not seal the session" } else { "this session cannot be opened by this Windows user" }.into()
            );
        }
        let bytes = std::slice::from_raw_parts(out.data, out.len as usize).to_vec();
        LocalFree(out.data);
        Ok(bytes)
    }
}

#[cfg(windows)]
fn seal(secret: &[u8]) -> Result<Vec<u8>, String> {
    run(secret, true)
}

#[cfg(windows)]
fn unseal(sealed: &[u8]) -> Result<Vec<u8>, String> {
    run(sealed, false)
}

#[cfg(not(windows))]
fn seal(_: &[u8]) -> Result<Vec<u8>, String> {
    Err("Stay signed in needs Windows' DPAPI here; nothing was kept".into())
}

#[cfg(not(windows))]
fn unseal(_: &[u8]) -> Result<Vec<u8>, String> {
    Err("no DPAPI".into())
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    /// A fresh, empty directory of this test run's own.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("alelyon-identity-client-test-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_kept_session_opens_again_and_its_file_is_not_the_token() {
        let dir = scratch("vault");
        let at = dir.join("session.sealed");
        assert_eq!(open(&at), None, "nothing kept");
        keep(&at, "opaque-refresh-token-123").unwrap();
        let raw = fs::read(&at).unwrap();
        assert!(!raw.windows(10).any(|w| w == b"opaque-ref"), "the file does not carry the token in the clear");
        assert_eq!(open(&at).as_deref(), Some("opaque-refresh-token-123"));
        keep(&at, "rotated").unwrap();
        assert_eq!(open(&at).as_deref(), Some("rotated"), "replaced whole");
        fs::write(&at, b"not a dpapi blob").unwrap();
        assert_eq!(open(&at), None, "a damaged file opens as nothing");
        forget(&at);
        assert!(!at.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unset_setting_keeps_the_session_under_the_users_profile() {
        let p = path_or("ALELYON_IDENTITY_CLIENT_TEST_NEVER_SET").unwrap();
        assert!(p.ends_with(Path::new(".alelyon").join("alelyon").join("session.sealed")));
    }
}
