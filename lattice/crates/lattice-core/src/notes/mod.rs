//! Notes: explanations attached to lines of a folder's files, shown beside
//! them in an editor and never written into the files.
//!
//! A note names its file (relative to the folder, forward slashes), the lines
//! it was written about, and those lines' text as they were then, with that
//! text's SHA-256. When the file is read again, [`place`] finds the lines:
//! where they were ([`Place::At`]), elsewhere in the file, after lines above
//! them were added or removed ([`Place::Moved`], the nearest such place), or
//! nowhere, because they were changed ([`Place::Stale`]). A stale note is kept
//! and shown as such until it is removed.
//!
//! A note is the reader's own, or the agent's: an agent that staged an edit
//! with a `why` leaves that reason on the lines the edit wrote, tied to its
//! chat and its change. Until the change is kept, the lines are not on disk,
//! so its note reads as stale against the file there.
//!
//! The store is one file, `<native>/chat/notes.json`, keyed by the folder's
//! workspace id and written through `fsx`'s atomic write; a file Lattice
//! cannot read is never written over.

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use serde::{Deserialize, Serialize};

use crate::clock::Clock;
use crate::state::StateRoot;

/// The longest note, in bytes.
pub const MAX_TEXT: usize = 4 * 1024;
/// The most lines a note is about; a longer span is cut to its first lines.
pub const MAX_LINES: u32 = 200;
/// The most notes kept for one folder.
pub const MAX_NOTES: usize = 2000;

/// Who wrote a note.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Author {
    Reader,
    Agent,
}

/// One note.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Note {
    pub id: String,
    /// Relative to the folder, forward slashes.
    pub path: String,
    /// The first and last line it is about (1-based, inclusive), as written.
    pub start: u32,
    pub end: u32,
    /// Those lines' text then, joined by `\n` (no carriage returns).
    pub quote: String,
    /// SHA-256 of `quote`.
    pub sha256: String,
    pub text: String,
    pub author: Author,
    /// The chat and the change an agent's note came from.
    #[serde(default)]
    pub chat: Option<String>,
    #[serde(default)]
    pub change: Option<String>,
    pub created: f64,
}

/// A note to add.
#[derive(Clone, Debug, PartialEq)]
pub struct NewNote {
    pub path: String,
    pub start: u32,
    pub end: u32,
    pub text: String,
    pub author: Author,
    pub chat: Option<String>,
    pub change: Option<String>,
}

/// Where a note's lines are in a file's text now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    At { start: u32, end: u32 },
    Moved { start: u32, end: u32 },
    Stale,
}

impl Place {
    /// Its lines, when they were found.
    pub fn lines(self) -> Option<(u32, u32)> {
        match self {
            Place::At { start, end } | Place::Moved { start, end } => Some((start, end)),
            Place::Stale => None,
        }
    }
}

/// A text's lines, without their line ends.
fn lines(text: &str) -> Vec<&str> {
    text.split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .collect()
}

/// Lines `start..=end` of `text` (1-based), joined by `\n`; `None` when the
/// span is not inside it.
pub fn quote_of(text: &str, start: u32, end: u32) -> Option<String> {
    let all = lines(text);
    if start == 0 || end < start || end as usize > all.len() {
        return None;
    }
    Some(all[start as usize - 1..end as usize].join("\n"))
}

/// Where `note`'s lines are in `text`.
pub fn place(note: &Note, text: &str) -> Place {
    let all = lines(text);
    let want: Vec<&str> = note.quote.split('\n').collect();
    let n = want.len();
    if n == 0 || n > all.len() {
        return Place::Stale;
    }
    let found = |at: usize| all[at..at + n] == want[..];
    let was = note.start.saturating_sub(1) as usize;
    if was + n <= all.len() && found(was) {
        return Place::At {
            start: note.start,
            end: note.end,
        };
    }
    // The nearest place holding the same lines.
    let best = (0..=all.len() - n)
        .filter(|&at| found(at))
        .min_by_key(|&at| at.abs_diff(was));
    match best {
        Some(at) => Place::Moved {
            start: at as u32 + 1,
            end: (at + n) as u32,
        },
        None => Place::Stale,
    }
}

/// The lines of `new` that differ from `old`, by their common first and last
/// lines (1-based, inclusive); a change that only removed lines gives the line
/// after the removal (or the last line). `None` when nothing differs or `new`
/// is empty.
pub fn changed_lines(old: &str, new: &str) -> Option<(u32, u32)> {
    let (a, b) = (lines(old), lines(new));
    if a == b || new.is_empty() {
        return None;
    }
    let first = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let last = a[first..]
        .iter()
        .rev()
        .zip(b[first..].iter().rev())
        .take_while(|(x, y)| x == y)
        .count();
    let end = b.len() - last;
    if end > first {
        Some((first as u32 + 1, end as u32))
    } else {
        let line = (first + 1).min(b.len()) as u32;
        Some((line, line))
    }
}

/// The lines of `text` holding the first occurrence of `needle` (1-based,
/// inclusive); `None` when it is empty or absent.
pub fn lines_holding(text: &str, needle: &str) -> Option<(u32, u32)> {
    if needle.trim().is_empty() {
        return None;
    }
    let at = text.find(needle)?;
    let start = text[..at].matches('\n').count() as u32 + 1;
    let inner = needle.trim_end_matches(['\n', '\r']);
    let end = start + inner.matches('\n').count() as u32;
    Some((start, end))
}

#[derive(Default, Serialize, Deserialize)]
struct Stored {
    #[serde(default)]
    v: u32,
    /// By the folder's workspace id.
    #[serde(default)]
    folders: std::collections::BTreeMap<String, Vec<Note>>,
}

/// The notes of every folder.
pub struct Notes {
    path: PathBuf,
    clock: Clock,
    lock: Mutex<()>,
}

fn guard(mutex: &Mutex<()>) -> MutexGuard<'_, ()> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl Notes {
    pub fn new(state: &StateRoot, clock: Clock) -> Self {
        Self {
            path: state.native_chat_dir().join("notes.json"),
            clock,
            lock: Mutex::new(()),
        }
    }

    fn read(&self) -> Result<Stored, String> {
        match std::fs::read(&self.path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|_| {
                "notes.json is not a record Lattice can read, so it was left as it is.".to_owned()
            }),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Stored::default()),
            Err(_) => Err("Lattice could not read notes.json.".to_owned()),
        }
    }

    fn write(&self, stored: &Stored) -> Result<(), String> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|_| "Lattice could not make its settings folder.".to_owned())?;
        }
        let bytes = serde_json::to_vec_pretty(stored).unwrap_or_default();
        crate::fsx::atomic_write(&self.path, &bytes)
            .map_err(|_| "Lattice could not save notes.json.".to_owned())
    }

    /// The folder's notes on `path`, oldest first.
    pub fn of_file(&self, folder: &str, path: &str) -> Result<Vec<Note>, String> {
        let _held = guard(&self.lock);
        let stored = self.read()?;
        Ok(stored
            .folders
            .get(folder)
            .map(|notes| notes.iter().filter(|n| n.path == path).cloned().collect())
            .unwrap_or_default())
    }

    /// Every note of the folder, oldest first.
    pub fn of_folder(&self, folder: &str) -> Result<Vec<Note>, String> {
        let _held = guard(&self.lock);
        Ok(self.read()?.folders.remove(folder).unwrap_or_default())
    }

    /// Add a note about lines of `text`, the file's text it was written
    /// against. A span past [`MAX_LINES`] is cut to its first lines.
    pub fn add(&self, folder: &str, text: &str, new: NewNote) -> Result<Note, String> {
        let words = new.text.trim();
        if words.is_empty() {
            return Err("Write the note first.".to_owned());
        }
        if words.len() > MAX_TEXT {
            return Err("A note is at most 4 KiB.".to_owned());
        }
        if new.path.is_empty() || new.path.starts_with('/') || new.path.contains('\\') {
            return Err("A note names a file inside the folder.".to_owned());
        }
        let end = new.end.min(new.start.saturating_add(MAX_LINES - 1));
        let quote = quote_of(text, new.start, end)
            .ok_or_else(|| "Those lines are not in the file.".to_owned())?;
        let _held = guard(&self.lock);
        let mut stored = self.read()?;
        let notes = stored.folders.entry(folder.to_owned()).or_default();
        if notes.len() >= MAX_NOTES {
            return Err(format!(
                "A folder keeps at most {MAX_NOTES} notes; remove some first."
            ));
        }
        let note = Note {
            id: format!("n_{}", &uuid::Uuid::new_v4().simple().to_string()[..12]),
            path: new.path,
            start: new.start,
            end,
            sha256: crate::sha::sha256_hex(quote.as_bytes()),
            quote,
            text: words.to_owned(),
            author: new.author,
            chat: new.chat,
            change: new.change,
            created: (self.clock)(),
        };
        notes.push(note.clone());
        stored.v = 1;
        self.write(&stored)?;
        Ok(note)
    }

    /// Remove a note.
    pub fn remove(&self, folder: &str, id: &str) -> Result<(), String> {
        let _held = guard(&self.lock);
        let mut stored = self.read()?;
        let notes = stored
            .folders
            .get_mut(folder)
            .ok_or_else(|| "That note is not here.".to_owned())?;
        let before = notes.len();
        notes.retain(|n| n.id != id);
        if notes.len() == before {
            return Err("That note is not here.".to_owned());
        }
        if notes.is_empty() {
            stored.folders.remove(folder);
        }
        self.write(&stored)
    }
}

#[cfg(test)]
mod tests;
