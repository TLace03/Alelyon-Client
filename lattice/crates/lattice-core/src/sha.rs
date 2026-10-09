//! SHA-256 as lowercase hex, of bytes or of a file read in pieces.
//!
//! The chat core names blobs by their hash, checks a file's bytes before and
//! after a write, and verifies a copy before the original is moved aside
//! (the chat core's spec §5.7, §5.8 ND2, §7.5).

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use sha2::{Digest, Sha256};

fn hex(digest: &[u8]) -> String {
    use std::fmt::Write;
    let mut text = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

/// The SHA-256 of `bytes`, 64 lowercase hexadecimal characters.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// The SHA-256 of everything `reader` yields, and how many bytes that was.
pub fn sha256_reader(mut reader: impl Read) -> io::Result<(String, u64)> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        total += read as u64;
    }
    Ok((hex(&hasher.finalize()), total))
}

/// The SHA-256 of a file's bytes, and its length.
pub fn sha256_file(path: &Path) -> io::Result<(String, u64)> {
    sha256_reader(File::open(path)?)
}

/// True when `text` is 64 lowercase hexadecimal characters: a name a blob may
/// have, checked before it becomes a path.
pub fn is_sha256_hex(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_digests() {
        // FIPS 180-2 test vectors.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let long = vec![b'a'; 200_000];
        let (streamed, length) = sha256_reader(&long[..]).unwrap();
        assert_eq!(streamed, sha256_hex(&long));
        assert_eq!(length, 200_000);
        assert!(is_sha256_hex(&streamed));
        assert!(!is_sha256_hex("../etc"));
        assert!(!is_sha256_hex(&streamed.to_uppercase()));
    }
}
