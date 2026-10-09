//! X1, X3 and X6's text rules, in memory (the chat core's spec §16.4
//! CF1, CF2, CF4, CF12, CF17).

use std::path::PathBuf;

use super::*;

fn resolved(path: &str, real: &str) -> Resolved {
    Resolved {
        path: PathBuf::from(path),
        real: PathBuf::from(real),
    }
}

/// CF2: every metacharacter of X1, `--%`, CR, LF, a tab, other controls and
/// non-ASCII characters each make a command ineligible, in every position.
#[test]
fn cf2_every_x1_character_makes_a_command_ineligible() {
    let base = "cargo test --release";
    // The spec's list, written out here so a character dropped from the
    // constant is missed by the code and caught by this test.
    let x1 = "; & | < > ^ $ ( ) { } [ ] ` @ % ! ' # ,";
    let x1: Vec<char> = x1.split(' ').map(|c| c.chars().next().unwrap()).collect();
    assert_eq!(x1.len(), 20);
    for c in x1 {
        for text in [
            format!("{c}cargo test"),
            format!("cargo{c} test"),
            format!("cargo test {c}"),
            format!("cargo te{c}st"),
        ] {
            assert_eq!(
                eligible(&text),
                Err(NotEligible::Metacharacter(c)),
                "{text:?}"
            );
        }
    }
    for c in [
        '\r', '\n', '\t', '\u{0}', '\u{1b}', '\u{7f}', '\u{a0}', '\u{2013}', '\u{201c}',
        '\u{ff1b}', '\u{202e}',
    ] {
        let text = format!("cargo test{c}--release");
        assert_eq!(
            eligible(&text),
            Err(NotEligible::NotPrintableAscii),
            "{text:?}"
        );
    }
    // `--%` holds `%`, a metacharacter, so it is refused as one; the
    // stop-parsing check stays behind it.
    assert_eq!(
        eligible("cargo --% test"),
        Err(NotEligible::Metacharacter('%'))
    );
    assert_eq!(
        eligible("cargo \"--%\" test"),
        Err(NotEligible::Metacharacter('%'))
    );
    assert_eq!(
        eligible(base),
        Ok(vec!["cargo".into(), "test".into(), "--release".into()])
    );
}

/// X1's quoting: a whole token may be quoted (spaces inside); any other
/// quote, an empty quoted token and a backslash before the closing quote
/// make the command ineligible.
#[test]
fn x1_quotes_wrap_whole_tokens_only() {
    assert_eq!(
        eligible("cargo test \"my test name\" -q"),
        Ok(vec![
            "cargo".into(),
            "test".into(),
            "my test name".into(),
            "-q".into()
        ])
    );
    for text in [
        "cargo te\"st\"",
        "cargo \"test",
        "cargo \"te\"st",
        "cargo \"\"",
        "cargo \"a\\\"",
        "cargo \"a\"\"b\"",
        "\"cargo\"x test",
    ] {
        assert_eq!(eligible(text), Err(NotEligible::Quote), "{text:?}");
    }
}

/// X1's other limits: at most 64 tokens, a bare program name, not empty.
#[test]
fn x1_tokens_and_the_program_name() {
    let many = format!("cargo{}", " x".repeat(63));
    assert!(eligible(&many).is_ok());
    let too_many = format!("cargo{}", " x".repeat(64));
    assert_eq!(eligible(&too_many), Err(NotEligible::TooManyTokens));
    for text in [
        r"C:\tools\cargo.exe test",
        "./cargo test",
        "..\\cargo test",
        "c:cargo test",
        "\"my tool\" x",
        "*.exe x",
        "x?y z",
        "a=b c",
    ] {
        assert!(eligible(text).is_err(), "{text:?}");
    }
    assert_eq!(
        eligible(r"tools\cargo test"),
        Err(NotEligible::NotABareName)
    );
    assert_eq!(eligible("   "), Err(NotEligible::Empty));
    assert_eq!(eligible(""), Err(NotEligible::Empty));
    assert!(eligible("g++ -v").is_ok());
    assert!(eligible("rustfmt.exe --check src/lib.rs").is_ok());
}

/// CF1 (the text side): `cargo test; calc` is never eligible, so it can
/// never match a `cargo test` entry; neither can any command with more
/// after a separator.
#[test]
fn cf1_a_chained_command_is_never_eligible() {
    for text in [
        "cargo test; calc",
        "cargo test;calc",
        "cargo test & calc",
        "cargo test && calc",
        "cargo test | calc",
        "cargo test\ncalc",
        "cargo test\r\ncalc",
        "cargo test `calc`",
        "cargo test $(calc)",
    ] {
        assert!(eligible(text).is_err(), "{text:?}");
    }
}

/// CF12 (the PowerShell names): `sc`, `curl`, `where`, `sort`, `fc` and the
/// others PowerShell 5.1 resolves itself are ineligible, in any case; the
/// same programs named with `.exe` are not PowerShell's names.
#[test]
fn cf12_powershell_names_are_ineligible() {
    for text in [
        "sc config.txt hello",
        "curl https://example.invalid/ -o x",
        "where cargo",
        "sort a.txt",
        "fc a.txt b.txt",
        "SC config.txt hello",
        "Curl x",
        "ls src",
        "cat a.txt",
        "echo hi",
        "rm x",
        "del x",
        "copy a b",
        "start x",
        "wget x",
        "gc x",
        "iwr x",
        "Get-Content x",
        "Remove-Item x",
        "mkdir x",
    ] {
        assert_eq!(eligible(text), Err(NotEligible::PowerShellName), "{text:?}");
    }
    for text in [
        "sc.exe query",
        "curl.exe -V",
        "where.exe cargo",
        "cargo build",
        "git status",
    ] {
        assert!(eligible(text).is_ok(), "{text:?}");
    }
}

/// The golden itself: the names the reviewer saw on this machine are there,
/// lower case and in ordinal order, with no path or user in it.
#[test]
fn the_powershell_names_golden_holds_the_reviewers_names() {
    let names = ps51_names();
    assert!(names.len() > 500, "{} names", names.len());
    for name in [
        "sc",
        "curl",
        "where",
        "sort",
        "fc",
        "wget",
        "ls",
        "dir",
        "cat",
        "echo",
        "gc",
        "set-content",
        "invoke-webrequest",
        "where-object",
        "sort-object",
        "format-custom",
        "start",
        "kill",
    ] {
        assert!(names.contains(name), "{name}");
    }
    for name in ["cargo", "git", "rustc", "python", "node"] {
        assert!(!names.contains(name), "{name}");
    }
    let parsed: serde_json::Value = serde_json::from_str(PS51_NAMES_JSON).unwrap();
    let list: Vec<&str> = parsed["names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|name| name.as_str().unwrap())
        .collect();
    let mut sorted = list.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(list, sorted, "ordinal order, no duplicates");
    assert!(list.iter().all(|name| *name == name.to_lowercase()));
    let lower = PS51_NAMES_JSON.to_lowercase();
    assert!(!lower.contains("users"), "no profile path");
    assert!(!lower.contains(r":\\"), "no drive path");
}

/// X3's stems: a version suffix and the extension are cut, case ignored.
#[test]
fn x3_stems() {
    for (name, stem) in [
        ("python3.12", "python"),
        ("python3.12.exe", "python"),
        ("python3", "python"),
        ("PowerShell.EXE", "powershell"),
        (
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
            "powershell",
        ),
        ("pythonw.exe", "pythonw"),
        ("pip3.11", "pip"),
        ("node-18", "node"),
        ("cargo.exe", "cargo"),
        ("x86", "x"),
        ("7z.exe", "7z"),
        ("g++", "g++"),
        ("rustc", "rustc"),
    ] {
        assert_eq!(stem_of(name), stem, "{name}");
    }
}

/// CF4 and CF12 (the entry side): interpreters and launchers are refused as
/// entries by the resolved program's file stem, and by the final path its
/// handle reads (a link named like a harmless tool to `python.exe` is
/// refused); a program without an argument is never an entry.
#[test]
fn cf4_cf12_interpreters_are_refused_as_entries_by_file_stem() {
    for (typed, path) in [
        ("python3.12", r"C:\py\python3.12.exe"),
        (
            "PowerShell.EXE",
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
        ),
        ("pythonw", r"C:\py\pythonw.exe"),
        ("dotnet", r"C:\Program Files\dotnet\dotnet.exe"),
        ("schtasks", r"C:\Windows\System32\schtasks.exe"),
        ("ssh", r"C:\Windows\System32\OpenSSH\ssh.exe"),
        ("cmd", r"C:\Windows\System32\cmd.exe"),
        ("node", r"C:\node\node.exe"),
        ("uvx", r"C:\uv\uvx.exe"),
        ("wsl", r"C:\Windows\System32\wsl.exe"),
        ("certutil", r"C:\Windows\System32\certutil.exe"),
        ("msiexec", r"C:\Windows\System32\msiexec.exe"),
        ("pwsh", r"C:\Program Files\PowerShell\7\pwsh.exe"),
    ] {
        let program = resolved(path, path);
        assert!(refused_as_entry(typed, &program), "{typed}");
        assert!(
            !may_be_entry(&[typed.to_owned(), "-m".to_owned()], &program),
            "{typed}"
        );
    }
    // The name typed and the file found are harmless; the final path is not.
    let linked = resolved(r"C:\tools\build.exe", r"C:\Python312\python.exe");
    assert!(refused_as_entry("build", &linked));
    // The name typed is an interpreter's, the file found is not.
    assert!(refused_as_entry(
        "python",
        &resolved(r"C:\x\tool.exe", r"C:\x\tool.exe")
    ));
    let cargo = resolved(r"C:\cargo\bin\cargo.exe", r"C:\cargo\bin\cargo.exe");
    assert!(!refused_as_entry("cargo", &cargo));
    assert!(may_be_entry(&["cargo".into(), "test".into()], &cargo));
    assert!(!may_be_entry(&["cargo".into()], &cargo), "a program alone");
}

/// CF17 (the text side) and X6: each bidi control is found; a Unicode dash
/// is not one.
#[test]
fn cf17_bidi_controls() {
    for c in BIDI_CONTROLS {
        assert!(has_bidi(&format!("echo a{c}b")), "{:04x}", u32::from(c));
    }
    assert!(!has_bidi("echo a\u{2013}b \u{201c}q\u{201d}"));
    assert_eq!(
        BIDI_SENTENCE,
        "The command contains characters that change how text is displayed."
    );
}
