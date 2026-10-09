//! Front matter, as commands and skills carry it: `key: value` lines between
//! two `---` lines at the start of a Markdown file. Not a port.
//!
//! Only the top level is read: a key's value is the rest of its line,
//! unquoted. A line that is indented, a list item or has no `:` is skipped,
//! so a nested value is read as empty. A file without front matter is all
//! body. A byte-order mark before it is set aside.

/// The front matter's top-level pairs, in order, and the body after it.
pub(crate) fn split(text: &str) -> (Vec<(String, String)>, &str) {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let Some(rest) = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))
    else {
        return (Vec::new(), text);
    };
    let mut offset = 0;
    let mut end = None;
    for line in rest.split_inclusive('\n') {
        if line.trim_end() == "---" {
            end = Some((offset, offset + line.len()));
            break;
        }
        offset += line.len();
    }
    let Some((head_end, body_start)) = end else {
        return (Vec::new(), text);
    };
    let mut pairs = Vec::new();
    for line in rest[..head_end].lines() {
        if line.starts_with([' ', '\t', '-', '#']) {
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        pairs.push((key.trim().to_owned(), unquote(value)));
    }
    (pairs, rest[body_start..].trim_start_matches(['\r', '\n']))
}

/// A value without its surrounding quotes.
pub(crate) fn unquote(value: &str) -> String {
    let value = value.trim();
    for quote in ['"', '\''] {
        if let Some(inner) = value
            .strip_prefix(quote)
            .and_then(|rest| rest.strip_suffix(quote))
        {
            return inner.to_owned();
        }
    }
    value.to_owned()
}

/// The value of `key`, if the front matter sets it.
pub(crate) fn value<'a>(pairs: &'a [(String, String)], key: &str) -> Option<&'a str> {
    pairs
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn front_matter_is_read_at_its_top_level_only() {
        let (pairs, body) = split(
            "\u{feff}---\r\ndescription: \"Run the tests\"\nallowed-tools: Bash(git:*)\nmeta:\n  nested: x\n- item\n---\r\n\r\nBody $ARGUMENTS\n",
        );
        assert_eq!(value(&pairs, "description"), Some("Run the tests"));
        assert_eq!(value(&pairs, "allowed-tools"), Some("Bash(git:*)"));
        assert_eq!(value(&pairs, "meta"), Some(""));
        assert!(value(&pairs, "nested").is_none());
        assert_eq!(body, "Body $ARGUMENTS\n");
        let (pairs, body) = split("No front matter.\n---\n");
        assert!(pairs.is_empty() && body == "No front matter.\n---\n");
        let (pairs, body) = split("---\nunclosed: yes\n");
        assert!(pairs.is_empty() && body.starts_with("---"));
    }
}
