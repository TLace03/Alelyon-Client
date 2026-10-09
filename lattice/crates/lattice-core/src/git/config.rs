//! Reading a repository's git configuration file natively, for FT6
//! (the chat core's spec §6.3): which files and folders it names, before
//! any git process is started in the folder. Not a port of git's parser; it
//! follows git's documented syntax (`git help config`, "Syntax") and is
//! stricter where git is lax, because anything it cannot parse makes the
//! folder count as one without git (fail closed).
//!
//! The syntax read:
//! - comments start with `#` or `;` (outside quotes) and run to the line end;
//!   a UTF-8 byte-order mark at the start is skipped;
//! - a section is `[name]`, `[name "subsection"]` (`\"` and `\\` escaped in
//!   the subsection) or the old `[name.subsection]`; section and variable names
//!   are compared without regard to case, a subsection with it;
//! - a variable is `name`, `name =` or `name = value`; a name starts with a
//!   letter and holds letters, digits and `-`; a value may be quoted in parts,
//!   holds the escapes `\\`, `\"`, `\n`, `\t` and `\b`, continues onto the next
//!   line after a final `\`, and loses the whitespace around it outside quotes.
//!
//! Anything else (an unknown escape, an unclosed quote or section, a bad name,
//! a NUL) is an error.

/// One variable: its full key (`section.name` or `section.subsection.name`,
/// the section and the name lower-cased) and its value (`None`: no `=`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub key: String,
    pub value: Option<String>,
    /// The 1-based line the variable starts on.
    pub line: usize,
}

/// Why a file is not configuration git would read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    pub line: usize,
    pub what: &'static str,
}

fn is_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '-'
}

/// Parse a whole configuration file.
pub fn parse(text: &str) -> Result<Vec<Entry>, ParseError> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let chars: Vec<char> = text.chars().collect();
    let mut at = 0usize;
    let mut line = 1usize;
    let mut section: Option<String> = None;
    let mut entries = Vec::new();
    let error = |line: usize, what: &'static str| Err(ParseError { line, what });
    while at < chars.len() {
        let c = chars[at];
        match c {
            '\n' => {
                line += 1;
                at += 1;
            }
            ' ' | '\t' | '\r' => at += 1,
            '#' | ';' => {
                while at < chars.len() && chars[at] != '\n' {
                    at += 1;
                }
            }
            '\0' => return error(line, "a NUL byte"),
            '[' => {
                at += 1;
                let mut name = String::new();
                while at < chars.len() && (is_name_char(chars[at]) || chars[at] == '.') {
                    name.push(chars[at].to_ascii_lowercase());
                    at += 1;
                }
                if name.is_empty() {
                    return error(line, "an empty section name");
                }
                let mut full = name;
                if at < chars.len() && (chars[at] == ' ' || chars[at] == '\t') {
                    while at < chars.len() && (chars[at] == ' ' || chars[at] == '\t') {
                        at += 1;
                    }
                    if at >= chars.len() || chars[at] != '"' {
                        return error(line, "a subsection must be quoted");
                    }
                    if full.contains('.') {
                        return error(line, "a dotted section name with a subsection");
                    }
                    at += 1;
                    let mut sub = String::new();
                    loop {
                        match chars.get(at) {
                            None | Some('\n') => return error(line, "an unclosed subsection"),
                            Some('"') => {
                                at += 1;
                                break;
                            }
                            Some('\\') => {
                                match chars.get(at + 1) {
                                    Some('\n') | None => {
                                        return error(line, "an unclosed subsection");
                                    }
                                    Some(&escaped) => sub.push(escaped),
                                }
                                at += 2;
                            }
                            Some('\0') => return error(line, "a NUL byte"),
                            Some(&other) => {
                                sub.push(other);
                                at += 1;
                            }
                        }
                    }
                    full = format!("{full}.{sub}");
                } else if let Some((head, tail)) = full.split_once('.') {
                    // The old `[section.subsection]` form: the subsection is
                    // lower-cased, as git does.
                    if head.is_empty() || tail.is_empty() {
                        return error(line, "a bad section name");
                    }
                }
                if chars.get(at) != Some(&']') {
                    return error(line, "an unclosed section header");
                }
                at += 1;
                section = Some(full);
            }
            c if c.is_ascii_alphabetic() => {
                let Some(current) = &section else {
                    return error(line, "a variable before any section");
                };
                let start_line = line;
                let mut name = String::new();
                while at < chars.len() && is_name_char(chars[at]) {
                    name.push(chars[at].to_ascii_lowercase());
                    at += 1;
                }
                while at < chars.len() && (chars[at] == ' ' || chars[at] == '\t') {
                    at += 1;
                }
                let value = match chars.get(at) {
                    None | Some('\n') | Some('\r') | Some('#') | Some(';') => None,
                    Some('=') => {
                        at += 1;
                        let (value, used, lines) = parse_value(&chars[at..], line)?;
                        at += used;
                        line += lines;
                        Some(value)
                    }
                    Some(_) => return error(line, "a bad variable name"),
                };
                entries.push(Entry {
                    key: format!("{current}.{name}"),
                    value,
                    line: start_line,
                });
            }
            _ => return error(line, "a line git would not read"),
        }
    }
    Ok(entries)
}

/// A value after its `=`: (the value, chars used, line ends passed inside it).
fn parse_value(chars: &[char], line: usize) -> Result<(String, usize, usize), ParseError> {
    let mut out = String::new();
    let mut pending_space = String::new();
    let mut quoted = false;
    let mut at = 0usize;
    let mut lines = 0usize;
    let mut started = false;
    loop {
        let Some(&c) = chars.get(at) else {
            if quoted {
                return Err(ParseError {
                    line: line + lines,
                    what: "an unclosed quote",
                });
            }
            break;
        };
        match c {
            '\n' if !quoted => break,
            '\n' => {
                return Err(ParseError {
                    line: line + lines,
                    what: "an unclosed quote",
                });
            }
            '\r' if !quoted && chars.get(at + 1).is_none_or(|next| *next == '\n') => {
                at += 1;
            }
            '#' | ';' if !quoted => {
                while at < chars.len() && chars[at] != '\n' {
                    at += 1;
                }
                break;
            }
            ' ' | '\t' if !quoted => {
                if started {
                    pending_space.push(c);
                }
                at += 1;
            }
            '"' => {
                out.push_str(&pending_space);
                pending_space.clear();
                started = true;
                quoted = !quoted;
                at += 1;
            }
            '\\' => {
                let next = chars.get(at + 1).copied();
                let escaped = match next {
                    Some('\n') => {
                        lines += 1;
                        at += 2;
                        continue;
                    }
                    Some('\r') if chars.get(at + 2) == Some(&'\n') => {
                        lines += 1;
                        at += 3;
                        continue;
                    }
                    Some('\\') => '\\',
                    Some('"') => '"',
                    Some('n') => '\n',
                    Some('t') => '\t',
                    Some('b') => '\u{8}',
                    _ => {
                        return Err(ParseError {
                            line: line + lines,
                            what: "an unknown escape",
                        });
                    }
                };
                out.push_str(&pending_space);
                pending_space.clear();
                started = true;
                out.push(escaped);
                at += 2;
            }
            '\0' => {
                return Err(ParseError {
                    line: line + lines,
                    what: "a NUL byte",
                });
            }
            other => {
                out.push_str(&pending_space);
                pending_space.clear();
                started = true;
                out.push(other);
                at += 1;
            }
        }
    }
    Ok((out, at, lines))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(text: &str) -> Vec<(String, Option<String>)> {
        parse(text)
            .unwrap()
            .into_iter()
            .map(|entry| (entry.key, entry.value))
            .collect()
    }

    fn one(key: &str, value: Option<&str>) -> (String, Option<String>) {
        (key.to_owned(), value.map(str::to_owned))
    }

    #[test]
    fn sections_names_and_values_read_as_git_reads_them() {
        let text = "\u{feff}# a comment\n[core]\n\tbare = false\n\tExcludesFile = \"C:/x y/ignore\" ; why\n[includeIf \"gitdir:C:/W/\"]\n  path = ../shared.inc\n[Include]\npath=a\\\\b\n[remote.Origin]\n\turl\n[x]\n\tv = a \"b  c\" d  # tail\n\tw = one\\\n two\n";
        assert_eq!(
            pairs(text),
            [
                one("core.bare", Some("false")),
                one("core.excludesfile", Some("C:/x y/ignore")),
                one("includeif.gitdir:C:/W/.path", Some("../shared.inc")),
                one("include.path", Some("a\\b")),
                one("remote.origin.url", None),
                one("x.v", Some("a b  c d")),
                one("x.w", Some("one two")),
            ]
        );
        assert_eq!(pairs("[a]\r\n\tb = c\r\n"), [one("a.b", Some("c"))]);
        assert_eq!(
            pairs("[a \"q\\\"x\"]\nb = 1\n"),
            [one("a.q\"x.b", Some("1"))]
        );
    }

    /// Anything git would not read is an error, so FT6 fails closed.
    #[test]
    fn what_git_would_not_read_is_an_error() {
        for text in [
            "b = 1\n",
            "[a\nb = 1\n",
            "[a \"unclosed]\n",
            "[a]\nb = \"open\n",
            "[a]\nb = \\q\n",
            "[a]\n1b = c\n",
            "[a]\nb c = d\n",
            "[]\n",
            "[a]\n\0\n",
            "{x}\n",
        ] {
            assert!(parse(text).is_err(), "{text:?}");
        }
    }
}
