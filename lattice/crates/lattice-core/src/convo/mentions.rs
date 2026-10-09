//! @-mentions: files the reader names in a message with `@path`, which the
//! agent's model is given with it (the tool-parity direction of
//! 2026-10-08, as Cursor and Claude Code take them).
//!
//! - **Which:** each `@` at the start of the message or after a space,
//!   followed by a path of the conversation's folder (trailing punctuation
//!   left off); at most [`MAX_MENTIONS`] files, the first ones named.
//! - **How they are read:** as the agent's `read_file` reads (the folder's
//!   path rules, `.gitignore` and `.latticeignore`, staged changes, 2,000
//!   lines and 2 MiB), so a mention reaches nothing the agent could not read
//!   itself. A path that cannot be read is said so, to the model and to the
//!   reader.
//! - **Where they go:** one user item before the message, at most
//!   [`MAX_CHARS`] characters, in this turn only (as the message's images
//!   are); later turns have the message's words, `@path` included, and the
//!   agent reads a file again when it needs it. For a model off this PC the
//!   item passes the secret tripwire as every input item does: one holding a
//!   key-shaped string is withheld, and the message still goes.
//! - **The reader** is told which files were read and which were not (a
//!   notice in the conversation).
//!
//! Not a port: the web Lattice reads no mentions.

use crate::tools::read::{ReadContext, ReadFileArgs, read_file};

/// The most files one message mentions.
pub const MAX_MENTIONS: usize = 5;
/// The most characters of the mentioned files given to the model.
pub const MAX_CHARS: usize = 64 * 1024;

/// The paths `text` mentions (`@path`), in order, each once, at most
/// [`MAX_MENTIONS`].
pub fn paths(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for word in text.split_whitespace() {
        let Some(path) = word.strip_prefix('@') else {
            continue;
        };
        let path = path.trim_end_matches(['.', ',', ';', ':', '!', '?', ')', ']', '\'', '"']);
        if path.is_empty() || path.contains("://") || out.iter().any(|known| known == path) {
            continue;
        }
        out.push(path.to_owned());
        if out.len() == MAX_MENTIONS {
            break;
        }
    }
    out
}

/// What reading the mentions gave: the model's item (`None` when nothing was
/// mentioned) and the reader's notice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Read {
    pub item: Option<String>,
    pub notice: Option<String>,
}

/// Read the files `text` mentions, as `read_file` reads them.
pub fn read(ctx: &ReadContext<'_>, text: &str) -> Read {
    let wanted = paths(text);
    if wanted.is_empty() {
        return Read {
            item: None,
            notice: None,
        };
    }
    let mut item = String::from(
        "The user mentioned these files in the message that follows; each is shown as read_file shows it.",
    );
    let (mut read_ok, mut not_read) = (Vec::new(), Vec::new());
    for path in &wanted {
        let part = match read_file(
            ctx,
            &ReadFileArgs {
                path: path.clone(),
                offset: None,
                limit: None,
            },
        ) {
            Ok(shown) => {
                read_ok.push(format!("@{path}"));
                shown
            }
            Err(why) => {
                not_read.push(format!("@{path} ({})", why.0));
                format!("@{path} was not read: {}", why.0)
            }
        };
        item.push_str("\n\n");
        item.push_str(&part);
    }
    if item.chars().count() > MAX_CHARS {
        let cut = item
            .char_indices()
            .nth(MAX_CHARS)
            .map_or(item.len(), |(at, _)| at);
        item.truncate(cut);
        item.push_str("\n[... the rest was left out; read the files with read_file]");
    }
    let mut notice = Vec::new();
    if !read_ok.is_empty() {
        notice.push(format!("The agent was given {}.", read_ok.join(", ")));
    }
    if !not_read.is_empty() {
        notice.push(format!("Not read: {}.", not_read.join("; ")));
    }
    Read {
        item: Some(item),
        notice: Some(notice.join(" ")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mentions_are_the_at_words_once_each_and_at_most_five() {
        assert_eq!(
            paths("Compare @src/a.rs with @src/b.rs, and @src/a.rs again (see @docs/x.md)."),
            ["src/a.rs", "src/b.rs", "docs/x.md"]
        );
        assert!(paths("mail me at me@example.com or see https://x.y/@z").is_empty());
        assert!(paths("@ alone, and @https://example.com").is_empty());
        assert_eq!(paths("@a @b @c @d @e @f @g").len(), MAX_MENTIONS);
    }
}
