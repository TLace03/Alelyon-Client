//! ND4 and NF2: nothing in lattice-core or lattice-sys removes a file or a
//! folder except where the never-delete rule allows it (spec §5.8).
//!
//! The words in [`FORBIDDEN`] may appear in code only where listed below. They
//! are `remove_file`, `remove_dir`, `remove_dir_all`, `DeleteFileW`,
//! `RemoveDirectoryW` and `unlink`, and (spec 22.6 ND4a, from the review's
//! nd4-word-list-misses-deletion-routes) the other routes to a deletion:
//! delete-on-close opens (`custom_flags`, `FILE_FLAG_DELETE_ON_CLOSE`), a
//! disposition set by handle (`FileDispositionInfo` and its kin), a rename by
//! handle that may replace (`FileRenameInfo`), `MoveFileExW`'s
//! `MOVEFILE_DELAY_UNTIL_REBOOT` (a null target deletes at the next boot) and
//! `MOVEFILE_REPLACE_EXISTING`, `fs::rename` (std's rename replaces an
//! existing target), the `tempfile` crate's auto-deleting types, and the ANSI,
//! v2, app-container and transacted deletes, `SHFileOperation`,
//! `IFileOperation` and `NtDeleteFile`. Each has a fixture the guard catches.
//!
//! The verifier's probe (phaseHA/logs/VERIFY-probe-nd4a) found five shapes
//! that hid a route from a word list; the guard now reads them as routes too
//! (`routes`), on code whose paths are squeezed (`a :: b` reads `a::b`, lines
//! kept):
//! - any path that ends in `::rename` (std's, tokio's, or a module imported
//!   under another name), a `use` of an `fs` module that names `rename`
//!   (`use std::fs::{rename as mv}`), and a glob import of an `fs` module;
//! - `SetFileInformationByHandle` anywhere but a call whose class is a named
//!   class that cannot delete or rename (`FileBasicInfo`, …): a class given by
//!   number (`4` is `FileDispositionInfo`) is a disposition;
//!   `NtSetInformationFile` and its `Zw` twin anywhere;
//! - `FILE_FLAG_DELETE_ON_CLOSE` by number: an integer literal equal to
//!   `0x0400_0000` anywhere, or with that bit set among a `CreateFile*`
//!   call's arguments; the NT opens (`NtCreateFile`, `NtOpenFile`), whose
//!   `FILE_DELETE_ON_CLOSE` option is a number too;
//! - `ReplaceFileW` (without a backup name the old target is gone), allowed
//!   only at lattice-sys's `replace_file`, which always passes one.
//!
//! They may appear in code only:
//! - in `lattice-core/src/fsx.rs`, whose two removals are of Lattice's own
//!   temporaries (ND3): a temporary file (`remove_own_temporary`), and an
//!   empty temporary folder (`remove_own_temporary_dir`, the managed
//!   llama.cpp server's token folder once its token file is gone, row C7);
//! - at the run store's two existing removals of its own temporaries
//!   (`store.rs`: the leftovers on taking the writer lock, and its temporary
//!   after a failed rename), listed by their exact text;
//! - at the test kit's removal of the temporary folders tests make
//!   (`testkit.rs`, `TempDir`'s `Drop`), listed by its exact text;
//! - at the three renames that replace by design, listed by their exact text:
//!   the run store's atomic write of its own file (`store.rs`), and
//!   lattice-sys's non-Windows fallbacks of `replace_file` (after a hard-link
//!   backup) and `move_no_replace` (after a check that the target is absent);
//! - at lattice-sys's `ReplaceFileW` import and its one call, in
//!   `replace_file`, which always names a backup.
//!
//! Deviation: a bare `TempDir` is not forbidden, because the test kit's own
//! `TempDir` is that name; the `tempfile` crate's is caught by its path.
//!
//! Test modules are not shipped and are skipped (`scanner`). The scanner and
//! the guard are each shown to catch a fixture before they are trusted.

mod scanner;

use scanner::{Source, code_only, is_test_file, sources, without_test_modules, word_lines};

/// The removals the never-delete rule names first (spec §5.8 ND4).
const REMOVALS: [&str; 6] = [
    "remove_file",
    "remove_dir",
    "remove_dir_all",
    "DeleteFileW",
    "RemoveDirectoryW",
    "unlink",
];

/// Every word the guard refuses: [`REMOVALS`], then ND4a's other routes.
const FORBIDDEN: &[&str] = &[
    "remove_file",
    "remove_dir",
    "remove_dir_all",
    "DeleteFileW",
    "RemoveDirectoryW",
    "unlink",
    // Delete on close (`OpenOptionsExt::custom_flags`, CreateFileW's flag).
    "custom_flags",
    "FILE_FLAG_DELETE_ON_CLOSE",
    "FILE_DELETE_ON_CLOSE",
    // SetFileInformationByHandle / NtSetInformationFile dispositions and
    // renames by handle.
    "FileDispositionInfo",
    "FileDispositionInfoEx",
    "FILE_DISPOSITION_INFO",
    "FILE_DISPOSITION_INFO_EX",
    "FileDispositionInformation",
    "FileDispositionInformationEx",
    "FileRenameInfo",
    "FileRenameInfoEx",
    // MoveFileExW: delete at the next boot (null target), replace a target.
    "MOVEFILE_DELAY_UNTIL_REBOOT",
    "MOVEFILE_REPLACE_EXISTING",
    // std's (and tokio's) rename replaces an existing target: see `routes`.
    // The tempfile crate's types delete when dropped.
    "tempfile",
    "NamedTempFile",
    "TempPath",
    "SpooledTempFile",
    // The other Win32 and NT deletes.
    "DeleteFileA",
    "DeleteFile2W",
    "DeleteFile2A",
    "DeleteFileFromAppW",
    "DeleteFileTransactedW",
    "DeleteFileTransactedA",
    "RemoveDirectoryA",
    "RemoveDirectory2W",
    "RemoveDirectory2A",
    "RemoveDirectoryFromAppW",
    "RemoveDirectoryTransactedW",
    "RemoveDirectoryTransactedA",
    "SHFileOperationW",
    "SHFileOperationA",
    "IFileOperation",
    "NtDeleteFile",
    "ZwDeleteFile",
    // ND4a's probe: dispositions set by NT class number, NT opens whose
    // delete-on-close option is a number, the disposition flags, and
    // ReplaceFileW (no backup name: the old target is gone).
    "NtSetInformationFile",
    "ZwSetInformationFile",
    "NtCreateFile",
    "ZwCreateFile",
    "NtOpenFile",
    "ZwOpenFile",
    "FILE_DISPOSITION_DELETE",
    "FILE_DISPOSITION_FLAG_DELETE",
    "ReplaceFileW",
    "ReplaceFileA",
    "ReplaceFile",
];

/// `SetFileInformationByHandle` classes that neither delete nor rename.
const QUIET_CLASSES: [&str; 5] = [
    "FileBasicInfo",
    "FileAllocationInfo",
    "FileEndOfFileInfo",
    "FileIoPriorityHintInfo",
    "FileCaseSensitiveInfo",
];

/// `FILE_FLAG_DELETE_ON_CLOSE`.
const DELETE_ON_CLOSE: u128 = 0x0400_0000;

/// The listed sites outside fsx.rs: (file, exact trimmed line, how many times).
const LISTED: [(&str, &str, usize); 8] = [
    (
        "lattice-core/src/store.rs",
        "let _ = fs::remove_file(entry.path());",
        1,
    ),
    (
        "lattice-core/src/store.rs",
        "let _ = fs::remove_file(&temporary);",
        1,
    ),
    (
        "lattice-core/src/testkit.rs",
        "let _ = std::fs::remove_dir_all(&self.path);",
        1,
    ),
    (
        "lattice-core/src/store.rs",
        "match fs::rename(&temporary, target) {",
        1,
    ),
    (
        "lattice-sys/src/fs.rs",
        "std::fs::rename(replacement, target).map_err(ReplaceError::Io)",
        1,
    ),
    ("lattice-sys/src/fs.rs", "std::fs::rename(from, to)", 1),
    (
        "lattice-sys/src/fs.rs",
        "MoveFileExW, OPEN_EXISTING, ReplaceFileW, SYNCHRONIZE, VOLUME_NAME_DOS,",
        1,
    ),
    ("lattice-sys/src/fs.rs", "ReplaceFileW(", 1),
];

/// Every use of a forbidden word in `code`, and every route `routes` reads:
/// (line, word).
fn removals(code: &str) -> Vec<(usize, &'static str)> {
    let code = squeezed(code);
    let mut found = Vec::new();
    for &word in FORBIDDEN {
        for line in word_lines(&code, word) {
            found.push((line, word));
        }
    }
    found.extend(routes(&code));
    found.sort();
    found.dedup();
    found
}

fn is_identifier_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// `code` with the white space around every `::` removed (`std::fs ::
/// rename` reads `std::fs::rename`). A line break removed there is put back
/// at the end of the line, so every later line keeps its number.
fn squeezed(code: &str) -> String {
    let chars: Vec<char> = code.chars().collect();
    let mut out = String::with_capacity(code.len());
    let mut owed = 0usize;
    let mut at = 0;
    while at < chars.len() {
        let c = chars[at];
        if c == ':' && chars.get(at + 1) == Some(&':') {
            while out.ends_with(|c: char| c.is_whitespace()) {
                if out.pop() == Some('\n') {
                    owed += 1;
                }
            }
            out.push_str("::");
            at += 2;
            while at < chars.len() && chars[at].is_whitespace() {
                if chars[at] == '\n' {
                    owed += 1;
                }
                at += 1;
            }
            continue;
        }
        out.push(c);
        if c == '\n' {
            out.extend(std::iter::repeat_n('\n', owed));
            owed = 0;
        }
        at += 1;
    }
    out.extend(std::iter::repeat_n('\n', owed));
    out
}

/// The line `offset` is on.
fn line_of(code: &str, offset: usize) -> usize {
    code[..offset].matches('\n').count() + 1
}

/// Where `word` occurs in `code` as a whole identifier: byte offsets.
fn word_offsets<'a>(code: &'a str, word: &'a str) -> impl Iterator<Item = usize> + 'a {
    code.match_indices(word).filter_map(move |(at, _)| {
        let before = code[..at].chars().next_back();
        let after = code[at + word.len()..].chars().next();
        (!before.is_some_and(is_identifier_char) && !after.is_some_and(is_identifier_char))
            .then_some(at)
    })
}

/// The top-level arguments of the call whose `(` is at `open`, trimmed, and
/// the offset of its `)`; `None` when the parentheses do not close.
fn call_arguments(code: &str, open: usize) -> Option<(Vec<&str>, usize)> {
    let mut depth = 0usize;
    let mut start = open + 1;
    let mut arguments = Vec::new();
    for (index, c) in code[open..].char_indices() {
        let at = open + index;
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    arguments.push(code[start..at].trim());
                    return Some((arguments, at));
                }
            }
            ',' if depth == 1 => {
                arguments.push(code[start..at].trim());
                start = at + 1;
            }
            _ => {}
        }
    }
    None
}

/// The value of a Rust integer literal (`0x0400_0000`, `67108864u32`), or
/// `None` for anything else.
fn integer_value(token: &str) -> Option<u128> {
    let text = token.replace('_', "");
    let (digits, radix) = if let Some(rest) = text.strip_prefix("0x") {
        (rest.to_owned(), 16)
    } else if let Some(rest) = text.strip_prefix("0o") {
        (rest.to_owned(), 8)
    } else if let Some(rest) = text.strip_prefix("0b") {
        (rest.to_owned(), 2)
    } else {
        (text.clone(), 10)
    };
    let suffixes = [
        "u128", "i128", "usize", "isize", "u64", "i64", "u32", "i32", "u16", "i16", "u8", "i8",
    ];
    let digits = suffixes
        .iter()
        .find_map(|suffix| digits.strip_suffix(suffix))
        .unwrap_or(&digits);
    u128::from_str_radix(digits, radix).ok()
}

/// Every integer literal in `code`: (byte offset, value).
fn integer_literals(code: &str) -> Vec<(usize, u128)> {
    let mut out = Vec::new();
    let bytes = code.as_bytes();
    let mut at = 0;
    while at < bytes.len() {
        let starts = bytes[at].is_ascii_digit()
            && !code[..at]
                .chars()
                .next_back()
                .is_some_and(is_identifier_char);
        if !starts {
            at += 1;
            continue;
        }
        let end = code[at..]
            .find(|c: char| !is_identifier_char(c))
            .map_or(code.len(), |length| at + length);
        if let Some(value) = integer_value(&code[at..end]) {
            out.push((at, value));
        }
        at = end;
    }
    out
}

/// ND4a's routes that a word list cannot see (the module header): (line,
/// what).
fn routes(code: &str) -> Vec<(usize, &'static str)> {
    let mut found = Vec::new();
    // A path ending in `::rename`.
    for (at, _) in code.match_indices("::rename") {
        let after = code[at + "::rename".len()..].chars().next();
        if !after.is_some_and(is_identifier_char) {
            found.push((line_of(code, at), "::rename"));
        }
    }
    // A `use` of an `fs` module that names `rename`, or imports everything.
    let uses: Vec<(usize, usize)> = word_offsets(code, "use")
        .map(|at| {
            (
                at,
                code[at..]
                    .find(';')
                    .map_or(code.len(), |length| at + length),
            )
        })
        .collect();
    for &(at, end) in &uses {
        let item = &code[at..end];
        let from_fs = word_offsets(item, "fs").next().is_some();
        if from_fs && word_offsets(item, "rename").next().is_some() {
            found.push((line_of(code, at), "use fs rename"));
        }
        if from_fs && item.contains("fs::*") {
            found.push((line_of(code, at), "use fs::*"));
        }
    }
    // SetFileInformationByHandle, unless a call with a quiet named class, or
    // a name in a `use` that does not rename it.
    for at in word_offsets(code, "SetFileInformationByHandle") {
        let rest = &code[at + "SetFileInformationByHandle".len()..];
        let next = rest.trim_start();
        let open = at + "SetFileInformationByHandle".len() + (rest.len() - next.len());
        let quiet = next.starts_with('(')
            && call_arguments(code, open).is_some_and(|(arguments, _)| {
                arguments.get(1).is_some_and(|class| {
                    let name = class.rsplit("::").next().unwrap_or(class);
                    QUIET_CLASSES.contains(&name)
                })
            });
        let imported = uses.iter().any(|&(start, end)| (start..end).contains(&at))
            && word_offsets(next, "as").next() != Some(0);
        if !quiet && !imported {
            found.push((line_of(code, at), "SetFileInformationByHandle"));
        }
    }
    // FILE_FLAG_DELETE_ON_CLOSE by number.
    let literals = integer_literals(code);
    for (at, value) in &literals {
        if *value == DELETE_ON_CLOSE {
            found.push((line_of(code, *at), "0x0400_0000"));
        }
    }
    for opener in [
        "CreateFileW",
        "CreateFileA",
        "CreateFile2",
        "CreateFileTransactedW",
    ] {
        for at in word_offsets(code, opener) {
            let rest = &code[at + opener.len()..];
            if !rest.trim_start().starts_with('(') {
                continue;
            }
            let open = at + opener.len() + (rest.len() - rest.trim_start().len());
            let close = call_arguments(code, open).map_or(code.len(), |(_, close)| close);
            if literals.iter().any(|(offset, value)| {
                (open..close).contains(offset) && value & DELETE_ON_CLOSE != 0
            }) {
                found.push((line_of(code, at), "delete on close by number"));
            }
        }
    }
    found
}

/// The removals in `source` that no rule allows.
fn violations(source: &Source) -> Vec<String> {
    if source.relative == "lattice-core/src/fsx.rs" {
        return Vec::new();
    }
    removals(&source.code)
        .into_iter()
        .filter(|(line, _)| {
            !LISTED
                .iter()
                .any(|(file, text, _)| *file == source.relative && *text == source.line(*line))
        })
        .map(|(line, word)| format!("{}:{line}: {word}: {}", source.relative, source.line(line)))
        .collect()
}

fn fixture(relative: &str, text: &str) -> Source {
    Source {
        relative: relative.to_owned(),
        original: text.to_owned(),
        code: without_test_modules(&code_only(text)),
    }
}

#[test]
fn the_scanner_skips_test_modules_and_nothing_else() {
    let text = "\
//! A header may say remove_file.
pub fn outside() {
    std::fs::remove_file(\"a\").ok();
    let note = \"never unlink here\";
}

#[cfg(test)]
mod tests {
    #[test]
    fn inside() {
        std::fs::remove_dir_all(\"t\").ok();
        let braces = '{';
    }
}

#[cfg(all(test, windows))]
#[allow(dead_code)]
pub(crate) mod windows_tests {
    fn inside() { DeleteFileW(); }
}

#[cfg(any(test, feature = \"test-support\"))]
pub mod seam {
    pub fn shipped_with_the_feature() { unlink(); }
}

#[cfg(test)]
mod declared;
";
    let found = removals(&fixture("lattice-core/src/x.rs", text).code);
    println!("fixture findings: {found:?}");
    assert_eq!(found, [(3, "remove_file"), (24, "unlink")]);
    assert!(is_test_file(std::path::Path::new(
        "src/chat/privacy_tests.rs"
    )));
    assert!(is_test_file(std::path::Path::new("src/manager/tests.rs")));
    assert!(!is_test_file(std::path::Path::new("src/testkit.rs")));
}

#[test]
fn the_guard_catches_a_planted_removal() {
    // NF2's mutant fixture: a source file that removes a file.
    let planted = fixture(
        "lattice-core/src/planted.rs",
        "pub fn tidy(path: &std::path::Path) {\n    let _ = std::fs::remove_file(path);\n}\n",
    );
    let found = violations(&planted);
    println!("planted fixture: {found:?}");
    assert_eq!(found.len(), 1, "{found:?}");
    // A listed line elsewhere is still a violation.
    let moved = fixture(
        "lattice-core/src/manager.rs",
        "fn f() {\n    let _ = fs::remove_file(&temporary);\n}\n",
    );
    assert_eq!(violations(&moved).len(), 1);
    // `remove_dir` inside `remove_dir_all` is one use, of the longer word.
    assert_eq!(
        removals(&code_only("fs::remove_dir_all(p);\n")),
        [(1, "remove_dir_all")]
    );
}

/// ND4a: the lexer cases the review found. Each hides a removal from the old
/// `code_only`; each must now be caught.
#[test]
fn the_scanner_reads_raw_byte_and_c_strings_and_escaped_quotes() {
    let cases = [
        (
            "a raw byte string ending in a backslash",
            "let _root = br\"C:\\\";\nlet _ = std::fs::remove_file(path);\nlet _note = \"\";\n",
            2,
        ),
        (
            "a raw C string ending in a backslash",
            "let _up = cr\"..\\\";\nstd::fs::remove_dir_all(p).ok();\nlet _n = \"\";\n",
            2,
        ),
        (
            "a raw byte string with an odd number of inner quotes",
            "let _say = br#\"say \"hi\"#;\nstd::fs::remove_file(p).ok();\nlet _n = \"\";\n",
            2,
        ),
        (
            "an escaped quote character literal",
            "let q = ['\\'','\"'];\nstd::fs::remove_file(p).ok();\nlet _n = \"\";\n",
            2,
        ),
        (
            "a string continued over an LF line end (the removal keeps its line number)",
            "let s = \"one \\\n    two\";\nstd::fs::remove_file(p).ok();\n",
            3,
        ),
        (
            "a string continued over a CRLF line end",
            "let s = \"one \\\r\n    two\";\r\nstd::fs::remove_file(p).ok();\r\n",
            3,
        ),
    ];
    for (what, text, line) in cases {
        let found = removals(&fixture("lattice-core/src/x.rs", text).code);
        println!("{what}: {found:?}");
        assert!(
            found.iter().any(|(at, _)| *at == line),
            "{what}: the removal on line {line} is hidden ({found:?})"
        );
    }
    // Nothing is invented either: a raw byte string's backslash, a raw
    // identifier and an escaped backslash literal leave no code behind.
    let quiet = "let s = br\"\\\";\nlet t = r#type;\nlet u = '\\\\';\n";
    let code = code_only(quiet);
    println!("quiet: {code:?}");
    assert_eq!(code, "let s = br\"\";\nlet t = r#type;\nlet u = ' ';\n");
}

/// ND4a: each of the other deletion routes is caught in a shipped file.
#[test]
fn the_guard_catches_every_other_deletion_route() {
    let routes = [
        (
            "delete on close through std",
            "OpenOptions::new().access_mode(0x0001_0000).custom_flags(0x0400_0000).open(p)",
        ),
        (
            "the delete-on-close flag by name",
            "let flags = FILE_FLAG_DELETE_ON_CLOSE;",
        ),
        (
            "a disposition set by handle",
            "unsafe { SetFileInformationByHandle(h, FileDispositionInfo, info, size) };",
        ),
        (
            "the extended disposition",
            "SetFileInformationByHandle(h, FileDispositionInfoEx, info, size);",
        ),
        (
            "a delete at the next boot",
            "unsafe { MoveFileExW(from, std::ptr::null(), MOVEFILE_DELAY_UNTIL_REBOOT) };",
        ),
        (
            "a move that replaces",
            "MoveFileExW(from, to, MOVEFILE_REPLACE_EXISTING);",
        ),
        (
            "std's rename onto an existing file",
            "std::fs::rename(&draft, &existing)?;",
        ),
        ("tokio's rename", "tokio::fs::rename(a, b).await?;"),
        (
            "a tempfile type that deletes on drop",
            "let scratch = tempfile::NamedTempFile::new()?;",
        ),
        ("a tempfile import", "use tempfile::TempDir;"),
        ("an ANSI delete", "unsafe { DeleteFileA(p) };"),
        ("a v2 folder removal", "unsafe { RemoveDirectory2W(p, 0) };"),
        ("a shell delete", "unsafe { SHFileOperationW(&mut op) };"),
        ("an NT delete", "NtDeleteFile(&attributes);"),
        // The verifier's probe (phaseHA/logs/VERIFY-probe-nd4a).
        (
            "an import of fs::rename under another name",
            "use std::fs::{rename as mv};\n    mv(a, b).unwrap();",
        ),
        ("std::fs :: rename, spaced", "std::fs :: rename(a, b);"),
        (
            "a path split over lines",
            "std::fs\n        ::rename(a, b);",
        ),
        (
            "a module imported under another name",
            "use std::fs as f;\n    f::rename(a, b);",
        ),
        ("a glob import of fs", "use std::fs::*;\n    rename(a, b);"),
        (
            "a disposition by class number",
            "unsafe { SetFileInformationByHandle(h, 4, &info as *const _ as _, 1) };",
        ),
        (
            "a disposition by a class constant of another name",
            "unsafe { SetFileInformationByHandle(h, CLASS, info, size) };",
        ),
        (
            "SetFileInformationByHandle imported under another name",
            "use windows_sys::Win32::Storage::FileSystem::SetFileInformationByHandle as set;",
        ),
        (
            "an NT disposition",
            "NtSetInformationFile(h, &mut io, info, size, 13);",
        ),
        (
            "delete on close by number",
            "let h = unsafe { CreateFileW(p, 0x10000, 0, null(), 3, 0x0400_0000, null_mut()) };",
        ),
        (
            "delete on close by number, unseparated",
            "let flags = 0x04000000;",
        ),
        (
            "delete on close by number among other flags",
            "unsafe { CreateFileW(p, 0x10000, 0, null(), 3, 0x0400_0080, null_mut()) };",
        ),
        ("delete on close in decimal", "let flags: u32 = 67108864;"),
        (
            "an NT open",
            "NtCreateFile(&mut h, access, &attributes, &mut io, null(), 0, 0, 1, 0x1000, null(), 0);",
        ),
        (
            "ReplaceFileW without a backup",
            "unsafe { ReplaceFileW(target, source, null(), 0, null(), null()) };",
        ),
    ];
    for (what, line) in routes {
        let planted = fixture(
            "lattice-core/src/convo/x.rs",
            &format!("pub fn tidy() {{\n    {line}\n}}\n"),
        );
        let found = violations(&planted);
        println!("{what}: {found:?}");
        assert!(!found.is_empty(), "{what}: not caught");
    }
    // A listed rename elsewhere is still a violation, and an ordinary method
    // named `rename` (the store's thread rename) is not a file rename.
    let moved = fixture(
        "lattice-core/src/manager.rs",
        "fn f() {\n    match fs::rename(&temporary, target) {}\n}\n",
    );
    assert_eq!(violations(&moved).len(), 1);
    let thread_rename = fixture(
        "lattice-core/src/chat/mod.rs",
        "fn f() {\n    store.rename(&id, &title)?;\n}\n",
    );
    assert!(violations(&thread_rename).is_empty());
    // The control: what the new routes must not catch.
    let quiet = fixture(
        "lattice-core/src/convo/x.rs",
        "use std::fs::{self, File, OpenOptions};\n\
         use lattice_sys::fs::{Access, open_no_follow};\n\
         fn rename(&self, id: &str) {}\n\
         fn f() {\n\
             store.rename(&id, &title)?;\n\
             let _ = runner.rename_branch(x);\n\
             unsafe { SetFileInformationByHandle(h, FileBasicInfo, info, size) };\n\
             unsafe { SetFileInformationByHandle(h, FileEndOfFileInfo, info, size) };\n\
             let h = unsafe { CreateFileW(p, GENERIC_READ, 0, null(), 3, 0x0200_0000, null_mut()) };\n\
             let size = 0x0400_0001u64 + 64;\n\
             let _ = lattice_sys::fs::replace_file(target, replacement, backup);\n\
         }\n",
    );
    let found = violations(&quiet);
    println!("quiet: {found:?}");
    assert!(found.is_empty(), "{found:?}");
    // Squeezing keeps every later line's number.
    assert_eq!(
        removals(&code_only(
            "let a = std::fs\n    ::rename(x, y);\nstd::fs::remove_file(p);\n"
        )),
        [(1, "::rename"), (3, "remove_file")]
    );
}

#[test]
fn nothing_removes_a_file_outside_fsx_and_the_listed_sites() {
    let sources = sources(&["lattice-core", "lattice-sys"]);
    assert!(sources.len() > 20, "read only {} files", sources.len());
    let mut found = Vec::new();
    for source in &sources {
        found.extend(violations(source));
    }
    assert!(
        found.is_empty(),
        "a removal outside the never-delete rule's allowances:\n{}",
        found.join("\n")
    );
    // Each listed site is still there exactly as listed, so a change to it is
    // a change to this guard too.
    for (file, text, count) in LISTED {
        let source = sources
            .iter()
            .find(|source| source.relative == file)
            .unwrap();
        let seen = removals(&source.code)
            .into_iter()
            .filter(|(line, _)| source.line(*line) == text)
            .count();
        assert_eq!(seen, count, "{file}: {text}");
    }
    // fsx.rs removes in two places, each inside the function that removes
    // only Lattice's own temporaries: a file, and an empty folder.
    let fsx = sources
        .iter()
        .find(|source| source.relative == "lattice-core/src/fsx.rs")
        .unwrap();
    let uses: Vec<(usize, &str)> = removals(&fsx.code)
        .into_iter()
        .filter(|(_, word)| REMOVALS.contains(word))
        .collect();
    assert_eq!(
        uses.iter().map(|(_, word)| *word).collect::<Vec<_>>(),
        ["remove_file", "remove_dir"],
        "{uses:?}"
    );
    let enclosing = |line: usize| {
        fsx.original
            .lines()
            .take(line)
            .collect::<Vec<_>>()
            .iter()
            .rev()
            .find(|text| text.trim_start().starts_with("pub fn "))
            .map(|text| text.trim().to_owned())
            .unwrap()
    };
    for ((line, word), function) in uses.iter().zip([
        "pub fn remove_own_temporary(",
        "pub fn remove_own_temporary_dir(",
    ]) {
        assert!(
            enclosing(*line).starts_with(function),
            "{word} at fsx.rs:{line} is in {}",
            enclosing(*line)
        );
    }
}
