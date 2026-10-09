//! UTF-16 strings for Win32 calls (not a port).
//!
//! Every path handed to a `W` function goes through [`to_wide`], which refuses
//! an interior NUL instead of letting Windows read a shorter path than the
//! caller meant.

use std::ffi::{OsStr, OsString};
use std::io;
use std::os::windows::ffi::{OsStrExt, OsStringExt};

/// `text` as NUL-terminated UTF-16. An interior NUL is `InvalidInput`.
pub(crate) fn to_wide(text: &OsStr) -> io::Result<Vec<u16>> {
    let mut wide: Vec<u16> = text.encode_wide().collect();
    if wide.contains(&0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "a path or argument holds a NUL character",
        ));
    }
    wide.push(0);
    Ok(wide)
}

/// UTF-16 code units (no terminator) back to an `OsString`.
pub(crate) fn from_wide(units: &[u16]) -> OsString {
    OsString::from_wide(units)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_nul_inside_is_refused_and_a_terminator_is_added() {
        assert_eq!(to_wide(OsStr::new("ab")).unwrap(), [97, 98, 0]);
        assert_eq!(
            to_wide(OsStr::new("a\0b")).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(from_wide(&[97, 98]), OsString::from("ab"));
    }
}
