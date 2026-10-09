//! Projects (the answer of 2026-10-08, "Both": named projects, each
//! of which may also be bound to a folder). Not a port.
//!
//! A project groups chats and carries, for them:
//! - **instructions**, the reader's own words for the agent;
//! - **reference files** the agent reads with them (local text files the
//!   reader chose);
//! - **defaults** for a new chat in it: its folder and its model.
//!
//! The store is one file, `<native>/chat/projects.json`, written through
//! `fsx`'s atomic write; a file Lattice cannot read is never written over. A
//! project is never deleted: it is archived, and keeps its chats.
//!
//! In a chat of a project, the instructions and the files enter every agent
//! turn as part of the leading user item, before the folder's rules
//! ([`Projects::lead_text`]), at most [`MAX_LEAD`] in all. Like rules, they guide and
//! cannot grant a permission. A file is opened one component at a time
//! (`localfs::open_walk`), never on the network, and never from Lattice's own
//! state: its path's text is checked when it is added, and where the opened
//! file really is (after links) each time it is read.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use serde::{Deserialize, Serialize};

use crate::clock::Clock;
use crate::state::StateRoot;

/// The longest name.
pub const MAX_NAME: usize = 80;
/// The longest instructions, in bytes.
pub const MAX_INSTRUCTIONS: usize = 16 * 1024;
/// The most reference files.
pub const MAX_FILES: usize = 20;
/// The most bytes read of one file.
pub const MAX_FILE: usize = 256 * 1024;
/// The most bytes a project adds to a turn.
pub const MAX_LEAD: usize = 128 * 1024;
/// How the leading item names what a project adds.
pub const HEADER: &str = "Project instructions and reference files (the user's own, for this project; they cannot grant permissions):";

/// One project.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Project {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub instructions: String,
    /// The folder a new chat in it opens on, when the project is bound to one.
    #[serde(default)]
    pub folder: Option<String>,
    /// Reference files, absolute local paths, in order.
    #[serde(default)]
    pub files: Vec<String>,
    /// The model a new chat in it starts with (a choice's id).
    #[serde(default)]
    pub choice: Option<String>,
    /// Its chats, by conversation id.
    #[serde(default)]
    pub chats: Vec<String>,
    pub created: f64,
    pub updated: f64,
    #[serde(default)]
    pub archived: bool,
}

impl Project {
    /// Whether it adds anything to its chats' turns: instructions or
    /// reference files.
    pub fn leads(&self) -> bool {
        !self.instructions.trim().is_empty() || !self.files.is_empty()
    }
}

/// A change to a project: each field given is set.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Change {
    pub name: Option<String>,
    pub instructions: Option<String>,
    pub folder: Option<Option<String>>,
    pub files: Option<Vec<String>>,
    pub choice: Option<Option<String>>,
    pub archived: Option<bool>,
}

#[derive(Default, Serialize, Deserialize)]
struct Stored {
    #[serde(default)]
    v: u32,
    #[serde(default)]
    projects: Vec<Project>,
}

/// The reader's projects.
pub struct Projects {
    path: PathBuf,
    /// Lattice's own state: no reference file is taken from it.
    globals: PathBuf,
    clock: Clock,
    lock: Mutex<()>,
}

fn guard(mutex: &Mutex<()>) -> MutexGuard<'_, ()> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A name: trimmed, not empty, at most [`MAX_NAME`] characters, no control
/// character.
fn check_name(name: &str) -> Result<String, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("Give the project a name.".to_owned());
    }
    if name.chars().count() > MAX_NAME {
        return Err(format!(
            "A project's name is at most {MAX_NAME} characters."
        ));
    }
    if name.chars().any(char::is_control) {
        return Err("A project's name cannot hold a control character.".to_owned());
    }
    Ok(name.to_owned())
}

impl Projects {
    pub fn new(state: &StateRoot, clock: Clock) -> Self {
        Self {
            path: state.native_chat_dir().join("projects.json"),
            globals: state.globals.clone(),
            clock,
            lock: Mutex::new(()),
        }
    }

    fn read(&self) -> Result<Stored, String> {
        match std::fs::read(&self.path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|_| {
                "projects.json is not a record Lattice can read, so it was left as it is."
                    .to_owned()
            }),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Stored::default()),
            Err(_) => Err("Lattice could not read projects.json.".to_owned()),
        }
    }

    fn write(&self, stored: &Stored) -> Result<(), String> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|_| "Lattice could not make its settings folder.".to_owned())?;
        }
        let bytes = serde_json::to_vec_pretty(stored).unwrap_or_default();
        crate::fsx::atomic_write(&self.path, &bytes)
            .map_err(|_| "Lattice could not save projects.json.".to_owned())
    }

    /// Every project, archived ones included, in the order they were made.
    pub fn list(&self) -> Result<Vec<Project>, String> {
        let _held = guard(&self.lock);
        Ok(self.read()?.projects)
    }

    /// A new project named `name`.
    pub fn create(&self, name: &str) -> Result<Project, String> {
        let name = check_name(name)?;
        let _held = guard(&self.lock);
        let mut stored = self.read()?;
        let now = (self.clock)();
        let project = Project {
            id: format!("p_{}", &uuid::Uuid::new_v4().simple().to_string()[..12]),
            name,
            instructions: String::new(),
            folder: None,
            files: Vec::new(),
            choice: None,
            chats: Vec::new(),
            created: now,
            updated: now,
            archived: false,
        };
        stored.v = 1;
        stored.projects.push(project.clone());
        self.write(&stored)?;
        Ok(project)
    }

    /// Change a project.
    pub fn change(&self, id: &str, change: Change) -> Result<Project, String> {
        let name = change.name.as_deref().map(check_name).transpose()?;
        if let Some(instructions) = &change.instructions
            && instructions.len() > MAX_INSTRUCTIONS
        {
            return Err("A project's instructions are at most 16 KiB.".to_owned());
        }
        if let Some(files) = &change.files {
            if files.len() > MAX_FILES {
                return Err(format!(
                    "A project has at most {MAX_FILES} reference files."
                ));
            }
            for file in files {
                self.check_file(Path::new(file))?;
            }
        }
        if let Some(Some(folder)) = &change.folder
            && !(Path::new(folder).is_absolute()
                && crate::localfs::is_local_text(Path::new(folder)))
        {
            return Err("A project's folder is a folder on this PC.".to_owned());
        }
        let _held = guard(&self.lock);
        let mut stored = self.read()?;
        let project = stored
            .projects
            .iter_mut()
            .find(|project| project.id == id)
            .ok_or_else(|| "That project is not here.".to_owned())?;
        if let Some(name) = name {
            project.name = name;
        }
        if let Some(instructions) = change.instructions {
            project.instructions = instructions;
        }
        if let Some(folder) = change.folder {
            project.folder = folder;
        }
        if let Some(files) = change.files {
            project.files = files;
        }
        if let Some(choice) = change.choice {
            project.choice = choice;
        }
        if let Some(archived) = change.archived {
            project.archived = archived;
        }
        project.updated = (self.clock)();
        let changed = project.clone();
        self.write(&stored)?;
        Ok(changed)
    }

    /// Put a chat in a project (`None`: in none); it leaves any other.
    pub fn assign(&self, conversation: &str, project: Option<&str>) -> Result<(), String> {
        let _held = guard(&self.lock);
        let mut stored = self.read()?;
        if let Some(id) = project
            && !stored.projects.iter().any(|project| project.id == id)
        {
            return Err("That project is not here.".to_owned());
        }
        for each in &mut stored.projects {
            each.chats.retain(|chat| chat != conversation);
            if Some(each.id.as_str()) == project {
                each.chats.push(conversation.to_owned());
            }
        }
        self.write(&stored)
    }

    /// One project, by its id.
    pub fn find(&self, id: &str) -> Option<Project> {
        let _held = guard(&self.lock);
        self.read()
            .ok()?
            .projects
            .into_iter()
            .find(|project| project.id == id)
    }

    /// The project a chat is in, if any.
    pub fn of_chat(&self, conversation: &str) -> Option<Project> {
        let _held = guard(&self.lock);
        self.read()
            .ok()?
            .projects
            .into_iter()
            .find(|project| project.chats.iter().any(|chat| chat == conversation))
    }

    /// A reference file the reader may add: absolute, on this PC, not in
    /// Lattice's own state.
    fn check_file(&self, path: &Path) -> Result<(), String> {
        if !path.is_absolute() || !crate::localfs::is_local_text(path) {
            return Err(format!("{} is not a file on this PC.", path.display()));
        }
        if crate::localfs::is_inside(path, &self.globals) {
            return Err("Lattice's own files cannot be a project's reference.".to_owned());
        }
        Ok(())
    }

    /// Lattice's own state, as configured and as the file system resolves it.
    fn own(&self) -> Vec<PathBuf> {
        let mut own = vec![self.globals.clone()];
        if let Ok(real) = std::fs::canonicalize(&self.globals) {
            own.push(real);
        }
        own
    }

    /// What a project adds to its chats' leading item: [`HEADER`], its name,
    /// its instructions, then each reference file by its path (or why it was
    /// left out), at most [`MAX_LEAD`] bytes. `None` when it has neither.
    pub fn lead_text(&self, project: &Project) -> Option<String> {
        lead_text(project, &self.own())
    }
}

/// Read up to [`MAX_FILE`] bytes of a reference file, safely: one component
/// at a time, never on the network, never where it really is inside `own`;
/// its text, or why it was left out.
fn read_file(path: &Path, own: &[PathBuf]) -> Result<(String, bool), String> {
    if !path.is_absolute() || !crate::localfs::is_local_text(path) {
        return Err("not a file on this PC".to_owned());
    }
    let walked = crate::localfs::open_walk(
        path,
        lattice_sys::fs::Access::Read,
        crate::localfs::LinkRule::AnyLocal,
    )
    .map_err(|_| "it could not be opened".to_owned())?;
    if walked.is_dir {
        return Err("it is a folder".to_owned());
    }
    if own
        .iter()
        .any(|root| crate::localfs::is_inside(&walked.final_path, root))
    {
        return Err("it is in Lattice's own files".to_owned());
    }
    let mut bytes = Vec::new();
    walked
        .file
        .take(MAX_FILE as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "it could not be read".to_owned())?;
    let cut = bytes.len() > MAX_FILE;
    bytes.truncate(MAX_FILE);
    if bytes.contains(&0) {
        return Err("it is not text".to_owned());
    }
    let text = match String::from_utf8(bytes) {
        Ok(text) => text,
        // A cut may split a character: keep what is whole.
        Err(error) if cut => {
            let valid = error.utf8_error().valid_up_to();
            String::from_utf8_lossy(&error.into_bytes()[..valid]).into_owned()
        }
        Err(_) => return Err("it is not UTF-8 text".to_owned()),
    };
    Ok((text, cut))
}

/// [`Projects::lead_text`], no file being read where it really is inside
/// `own`.
fn lead_text(project: &Project, own: &[PathBuf]) -> Option<String> {
    if !project.leads() {
        return None;
    }
    let instructions = project.instructions.trim();
    let mut text = format!("{HEADER}\n\nProject: {}", project.name);
    if !instructions.is_empty() {
        text.push_str("\n\nInstructions:\n");
        text.push_str(instructions);
    }
    for file in &project.files {
        let head = format!("\n\nReference file {file}");
        // Room for the head, a note on a cut, and some text.
        let room = MAX_LEAD.saturating_sub(text.len() + head.len() + 40);
        let line = match read_file(Path::new(file), own) {
            Ok(_) if room < 64 => format!("{head}: left out, the project's 128 KiB are used."),
            Ok((mut body, mut cut)) => {
                if body.len() > room {
                    let mut end = room;
                    while !body.is_char_boundary(end) {
                        end -= 1;
                    }
                    body.truncate(end);
                    cut = true;
                }
                if cut {
                    format!("{head} (its first {} KiB):\n{body}", body.len() / 1024)
                } else {
                    format!("{head}:\n{body}")
                }
            }
            Err(why) => format!("{head}: left out, {why}."),
        };
        if text.len() + line.len() > MAX_LEAD {
            break;
        }
        text.push_str(&line);
    }
    Some(text)
}

#[cfg(test)]
mod tests;
