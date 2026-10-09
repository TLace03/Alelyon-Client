//! Which command text may ever run without a shell, and which program may
//! ever be allowed to (the chat core's spec §7.6 X1, X3, X6). Not a
//! port.
//!
//! **X1, eligibility for a standing match** ([`eligible`]). A command is
//! eligible only when all of these hold; anything else always asks:
//! - it is printable ASCII (no CR, LF, tab or other control character, no
//!   character past `~`);
//! - it holds none of [`METACHARACTERS`], the union of cmd's and
//!   PowerShell's special characters;
//! - it splits on spaces into at most [`MAX_TOKENS`] tokens; a token may be
//!   wrapped in double quotes with no quote inside and no backslash before
//!   the closing quote, and any other `"` makes the command ineligible;
//! - no token is `--%` (PowerShell's stop-parsing token);
//! - `argv[0]` is a bare program name, `[A-Za-z0-9._+-]+`, with no path
//!   separator;
//! - `argv[0]` is **not** a Windows PowerShell 5.1 alias, function or cmdlet
//!   name under `-NoProfile`, compared without case against the golden
//!   `tests/goldens/agent/ps51_names.json`, which `tools/ps51_command_names.ps1`
//!   writes. A command approved once runs in PowerShell (X6), where `sc` is
//!   `Set-Content` and `curl` is `Invoke-WebRequest`, while a standing entry
//!   runs `sc.exe` or `curl.exe` directly: the same text would run two
//!   programs.
//!
//! **X6's bidi refusal** ([`has_bidi`]): a command holding U+202A–U+202E,
//! U+2066–U+2069, U+200E or U+200F is refused outright, because those
//! characters make the card read differently from what PowerShell runs
//! (Trojan Source). Other non-ASCII and control characters are shown
//! escaped in the card and the dialog (CP2, `ports::escape_for_dialog`).
//!
//! **X3, entry rules** ([`refused_as_entry`]). An entry is a program and at
//! least one argument. Interpreters and launchers are refused as entries,
//! by the **file stem** of the resolved program (and of the name typed, and
//! of the final path its handle reads), compared without case after a
//! trailing version suffix is cut (`python3.12` → `python`, `PowerShell.EXE`
//! → `powershell`), against [`REFUSED_STEMS`].

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::OnceLock;

use super::resolve::Resolved;

/// The longest `command` the tool takes, in characters.
pub const MAX_COMMAND_CHARS: usize = 8000;
/// The most tokens an eligible command splits into.
pub const MAX_TOKENS: usize = 64;

/// X1's metacharacters: cmd's and PowerShell's special characters together.
pub const METACHARACTERS: [char; 20] = [
    ';', '&', '|', '<', '>', '^', '$', '(', ')', '{', '}', '[', ']', '`', '@', '%', '!', '\'', '#',
    ',',
];

/// X6: the characters that reorder how text is displayed.
pub const BIDI_CONTROLS: [char; 11] = [
    '\u{202a}', '\u{202b}', '\u{202c}', '\u{202d}', '\u{202e}', '\u{2066}', '\u{2067}', '\u{2068}',
    '\u{2069}', '\u{200e}', '\u{200f}',
];

/// X6's refusal of a command with a bidi control.
pub const BIDI_SENTENCE: &str =
    "The command contains characters that change how text is displayed.";

/// X3: interpreters and launchers, which are never an entry.
pub const REFUSED_STEMS: [&str; 40] = [
    "powershell",
    "pwsh",
    "cmd",
    "conhost",
    "bash",
    "sh",
    "wsl",
    "wt",
    "explorer",
    "python",
    "pythonw",
    "py",
    "pyw",
    "pip",
    "uv",
    "uvx",
    "node",
    "npx",
    "npm",
    "deno",
    "bun",
    "dotnet",
    "ruby",
    "perl",
    "php",
    "wscript",
    "cscript",
    "mshta",
    "rundll32",
    "regsvr32",
    "msiexec",
    "forfiles",
    "schtasks",
    "winget",
    "certutil",
    "bitsadmin",
    "ssh",
    "env",
    "start",
    // Not in the spec's list, and as plainly a launcher: it starts any
    // program under another account.
    "runas",
];

/// The PowerShell names golden (X1), as `tools/ps51_command_names.ps1` wrote it.
const PS51_NAMES_JSON: &str = include_str!("../../tests/goldens/agent/ps51_names.json");

#[derive(serde::Deserialize)]
struct Ps51Names {
    names: Vec<String>,
}

/// The alias, function and cmdlet names of Windows PowerShell 5.1, lower case.
pub fn ps51_names() -> &'static BTreeSet<String> {
    static NAMES: OnceLock<BTreeSet<String>> = OnceLock::new();
    NAMES.get_or_init(|| {
        let parsed: Ps51Names =
            serde_json::from_str(PS51_NAMES_JSON).expect("the ps51_names golden is JSON");
        parsed
            .names
            .into_iter()
            .map(|name| name.to_lowercase())
            .collect()
    })
}

/// Whether PowerShell 5.1 would take `name` as an alias, function or cmdlet.
pub fn is_powershell_name(name: &str) -> bool {
    ps51_names().contains(&name.to_lowercase())
}

/// Why a command can never match a standing entry (it always asks).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NotEligible {
    Empty,
    /// A character that is not printable ASCII (a control, CR, LF, non-ASCII).
    NotPrintableAscii,
    Metacharacter(char),
    /// A double quote not wrapping a whole token.
    Quote,
    /// PowerShell's `--%`.
    StopParsing,
    TooManyTokens,
    /// `argv[0]` is not a bare program name.
    NotABareName,
    /// `argv[0]` is a PowerShell alias, function or cmdlet.
    PowerShellName,
}

impl NotEligible {
    /// One sentence for the card.
    pub fn sentence(&self) -> &'static str {
        match self {
            Self::Empty => "The command is empty.",
            Self::NotPrintableAscii => {
                "The command has characters other than printable ASCII, so it always asks."
            }
            Self::Metacharacter(_) => {
                "The command has a shell character (such as ; & | > $), so it always asks."
            }
            Self::Quote => {
                "The command quotes text in a way only a shell reads, so it always asks."
            }
            Self::StopParsing => "The command uses PowerShell's --%, so it always asks.",
            Self::TooManyTokens => "The command has too many words to allow, so it always asks.",
            Self::NotABareName => "Only a program's bare name can be allowed to run directly.",
            Self::PowerShellName => {
                "In PowerShell that name is a command of its own, not the program, so it always asks."
            }
        }
    }
}

fn bare_program(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'+' | b'-'))
}

/// X1: the argv of an eligible command, or why it is not eligible.
pub fn eligible(text: &str) -> Result<Vec<String>, NotEligible> {
    if text.chars().any(|c| !(' '..='~').contains(&c)) {
        return Err(NotEligible::NotPrintableAscii);
    }
    if let Some(c) = text.chars().find(|c| METACHARACTERS.contains(c)) {
        return Err(NotEligible::Metacharacter(c));
    }
    let argv = tokens(text)?;
    // `%` is a metacharacter, so `--%` cannot get here; kept so the rule
    // holds if the set ever changes.
    if argv.iter().any(|word| word == "--%") {
        return Err(NotEligible::StopParsing);
    }
    if argv.len() > MAX_TOKENS {
        return Err(NotEligible::TooManyTokens);
    }
    let Some(program) = argv.first() else {
        return Err(NotEligible::Empty);
    };
    if !bare_program(program) {
        return Err(NotEligible::NotABareName);
    }
    if is_powershell_name(program) {
        return Err(NotEligible::PowerShellName);
    }
    Ok(argv)
}

/// Split printable ASCII on spaces. A token wrapped in double quotes may hold
/// spaces; it holds no other quote, is not empty (PowerShell 5.1 drops an
/// empty argument that a direct start would pass), and has no backslash
/// before its closing quote. Any other `"` is refused.
fn tokens(text: &str) -> Result<Vec<String>, NotEligible> {
    let bytes = text.as_bytes();
    let mut argv = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        if bytes[at] == b' ' {
            at += 1;
            continue;
        }
        if bytes[at] == b'"' {
            let close = text[at + 1..]
                .find('"')
                .map(|offset| at + 1 + offset)
                .ok_or(NotEligible::Quote)?;
            let inner = &text[at + 1..close];
            let after = bytes.get(close + 1).copied();
            if inner.is_empty() || inner.ends_with('\\') || !matches!(after, None | Some(b' ')) {
                return Err(NotEligible::Quote);
            }
            argv.push(inner.to_owned());
            at = close + 1;
        } else {
            let end = text[at..]
                .find(' ')
                .map_or(text.len(), |offset| at + offset);
            let word = &text[at..end];
            if word.contains('"') {
                return Err(NotEligible::Quote);
            }
            argv.push(word.to_owned());
            at = end;
        }
        if argv.len() > MAX_TOKENS {
            return Err(NotEligible::TooManyTokens);
        }
    }
    Ok(argv)
}

/// X6: whether `text` holds a character that changes how text is displayed.
pub fn has_bidi(text: &str) -> bool {
    text.chars().any(|c| BIDI_CONTROLS.contains(&c))
}

/// A program's file stem as X3 compares it: lower case, the extension and a
/// trailing version suffix (`3.12`, `-3`, `_2`) cut.
pub fn stem_of(name: &str) -> String {
    let file = name
        .rsplit(['\\', '/'])
        .next()
        .unwrap_or(name)
        .to_lowercase();
    let file = file.trim_end_matches(['.', ' ']);
    let stem = match file.rsplit_once('.') {
        Some((stem, extension))
            if !stem.is_empty() && extension.chars().all(|c| c.is_ascii_alphabetic()) =>
        {
            stem
        }
        _ => file,
    };
    let cut = stem.trim_end_matches(|c: char| c.is_ascii_digit() || c == '.');
    let cut = cut.trim_end_matches(['-', '_']);
    if cut.is_empty() {
        stem.to_owned()
    } else {
        cut.to_owned()
    }
}

/// X3: whether a program may never be an entry. The stems of the name typed,
/// of the program found on `PATH`, and of the final path read from its
/// handle are all compared.
pub fn refused_as_entry(typed: &str, resolved: &Resolved) -> bool {
    let stems = [
        stem_of(typed),
        stem_of(&resolved.path.to_string_lossy()),
        stem_of(&resolved.real.to_string_lossy()),
    ];
    stems
        .iter()
        .any(|stem| REFUSED_STEMS.contains(&stem.as_str()))
}

/// X3: whether `argv`, resolved to `resolved`, may become an entry (with
/// [`eligible`] already passed): a program and at least one argument, and
/// not an interpreter or launcher.
pub fn may_be_entry(argv: &[String], resolved: &Resolved) -> bool {
    argv.len() >= 2 && !refused_as_entry(&argv[0], resolved)
}

/// A path's file stem by [`stem_of`].
pub fn path_stem(path: &Path) -> String {
    stem_of(&path.to_string_lossy())
}

#[cfg(test)]
mod tests;
