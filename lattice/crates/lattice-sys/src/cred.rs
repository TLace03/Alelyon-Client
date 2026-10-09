//! Windows Credential Manager: generic credentials, read, written and removed
//! by target name (`CredReadW`, `CredWriteW`, `CredDeleteW`). Not a port.
//!
//! Decided 2026-10-07: an API key typed into Alelyon is kept
//! in Credential Manager, never in a plain-text file. A credential is the
//! signed-in user's: Windows encrypts it with their logon (DPAPI) and shows it
//! under Control Panel > Credential Manager > Windows Credentials, where it can
//! be seen and removed by hand. It is persisted for this user on this computer
//! (`CRED_PERSIST_LOCAL_MACHINE`), never roamed.
//!
//! Removing a credential is a deletion this crate otherwise never does; it is
//! the reader's own "forget this key" and touches nothing on disk that Lattice
//! keeps (the never-delete guard lists the call).
//!
//! The value is handed back as bytes and the copies this module makes are
//! zeroed before they are dropped; the caller decides how long it lives.

use std::io;

/// The largest value a generic credential holds (`CRED_MAX_CREDENTIAL_BLOB_SIZE`).
pub const MAX_BLOB: usize = 5 * 512;

/// The value stored under `target`, or `None` when there is none.
pub fn read_generic(target: &str) -> io::Result<Option<Vec<u8>>> {
    imp::read_generic(target)
}

/// Store `blob` under `target` (replacing any earlier value), for `user`, a
/// name shown beside it in Credential Manager.
pub fn write_generic(target: &str, user: &str, blob: &[u8]) -> io::Result<()> {
    if blob.len() > MAX_BLOB {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "a credential holds at most 2,560 bytes",
        ));
    }
    imp::write_generic(target, user, blob)
}

/// Remove the credential under `target`. `Ok(false)` when there was none.
pub fn delete_generic(target: &str) -> io::Result<bool> {
    imp::delete_generic(target)
}

#[cfg(windows)]
mod imp {
    use std::io;
    use std::ptr;

    use windows_sys::Win32::Foundation::{ERROR_NOT_FOUND, GetLastError};
    use windows_sys::Win32::Security::Credentials::{
        CRED_PERSIST_LOCAL_MACHINE, CRED_TYPE_GENERIC, CREDENTIALW, CredDeleteW, CredFree,
        CredReadW, CredWriteW,
    };

    use crate::wide::to_wide;

    pub(super) fn read_generic(target: &str) -> io::Result<Option<Vec<u8>>> {
        let target = to_wide(std::ffi::OsStr::new(target))?;
        let mut found: *mut CREDENTIALW = ptr::null_mut();
        // SAFETY: `target` is a NUL-terminated UTF-16 string that outlives the
        // call; `found` receives a buffer the system allocates, freed below.
        let ok = unsafe { CredReadW(target.as_ptr(), CRED_TYPE_GENERIC, 0, &mut found) };
        if ok == 0 {
            // SAFETY: reads this thread's last error, set by the failed call.
            let error = unsafe { GetLastError() };
            return if error == ERROR_NOT_FOUND {
                Ok(None)
            } else {
                Err(io::Error::from_raw_os_error(error as i32))
            };
        }
        // SAFETY: on success `found` points to a CREDENTIALW the system filled,
        // whose blob is `CredentialBlobSize` bytes at `CredentialBlob` (or null
        // when the size is zero); it stays valid until `CredFree`.
        let value = unsafe {
            let credential = &*found;
            let size = credential.CredentialBlobSize as usize;
            let value = if size == 0 || credential.CredentialBlob.is_null() {
                Vec::new()
            } else {
                std::slice::from_raw_parts(credential.CredentialBlob, size).to_vec()
            };
            if size > 0 && !credential.CredentialBlob.is_null() {
                ptr::write_bytes(credential.CredentialBlob, 0, size);
            }
            CredFree(found.cast());
            value
        };
        Ok(Some(value))
    }

    pub(super) fn write_generic(target: &str, user: &str, blob: &[u8]) -> io::Result<()> {
        let mut target = to_wide(std::ffi::OsStr::new(target))?;
        let mut user = to_wide(std::ffi::OsStr::new(user))?;
        let mut copy = blob.to_vec();
        let credential = CREDENTIALW {
            Flags: 0,
            Type: CRED_TYPE_GENERIC,
            TargetName: target.as_mut_ptr(),
            Comment: ptr::null_mut(),
            LastWritten: Default::default(),
            CredentialBlobSize: copy.len() as u32,
            CredentialBlob: if copy.is_empty() {
                ptr::null_mut()
            } else {
                copy.as_mut_ptr()
            },
            Persist: CRED_PERSIST_LOCAL_MACHINE,
            AttributeCount: 0,
            Attributes: ptr::null_mut(),
            TargetAlias: ptr::null_mut(),
            UserName: user.as_mut_ptr(),
        };
        // SAFETY: every pointer in `credential` points into a buffer above that
        // outlives the call; the system copies what it keeps.
        let ok = unsafe { CredWriteW(&credential, 0) };
        let result = if ok == 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        };
        copy.iter_mut().for_each(|b| *b = 0);
        result
    }

    pub(super) fn delete_generic(target: &str) -> io::Result<bool> {
        let target = to_wide(std::ffi::OsStr::new(target))?;
        // SAFETY: `target` is a NUL-terminated UTF-16 string that outlives the call.
        let ok = unsafe { CredDeleteW(target.as_ptr(), CRED_TYPE_GENERIC, 0) };
        if ok != 0 {
            return Ok(true);
        }
        // SAFETY: reads this thread's last error, set by the failed call.
        let error = unsafe { GetLastError() };
        if error == ERROR_NOT_FOUND {
            Ok(false)
        } else {
            Err(io::Error::from_raw_os_error(error as i32))
        }
    }
}

#[cfg(not(windows))]
mod imp {
    use std::io;

    fn unsupported() -> io::Error {
        io::Error::new(
            io::ErrorKind::Unsupported,
            "Credential Manager is Windows only",
        )
    }

    pub(super) fn read_generic(_: &str) -> io::Result<Option<Vec<u8>>> {
        Ok(None)
    }

    pub(super) fn write_generic(_: &str, _: &str, _: &[u8]) -> io::Result<()> {
        Err(unsupported())
    }

    pub(super) fn delete_generic(_: &str) -> io::Result<bool> {
        Ok(false)
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    /// Writes, reads and removes one credential of its own, under a target no
    /// real key uses, and leaves none behind.
    #[test]
    fn a_credential_round_trips_and_is_removed() {
        let target = format!(
            "lattice-sys-test/{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
        );
        assert_eq!(read_generic(&target).unwrap(), None);
        write_generic(&target, "lattice-sys test", b"not a secret \x00\xff").unwrap();
        assert_eq!(
            read_generic(&target).unwrap().as_deref(),
            Some(&b"not a secret \x00\xff"[..])
        );
        write_generic(&target, "lattice-sys test", b"replaced").unwrap();
        assert_eq!(
            read_generic(&target).unwrap().as_deref(),
            Some(&b"replaced"[..])
        );
        assert!(delete_generic(&target).unwrap());
        assert!(!delete_generic(&target).unwrap());
        assert_eq!(read_generic(&target).unwrap(), None);
    }

    #[test]
    fn an_oversized_value_is_refused_before_windows_sees_it() {
        let error = write_generic("lattice-sys-test/never", "t", &[0u8; MAX_BLOB + 1]).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }
}
