//! Which model writes the editor's completions, and whether they are on:
//! `<native>/completion.json`, Lattice's own state, written whole.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::state::StateRoot;

/// Where completions come from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "lowercase")]
pub enum Choice {
    /// A GGUF file in the models folder, run by the core's own llama.cpp
    /// server: nothing leaves the machine.
    Local { model: String },
    /// A model of a connected provider, by its id there.
    Hosted { provider: String, model: String },
}

impl Choice {
    /// The model's id or file name.
    pub fn model(&self) -> &str {
        match self {
            Choice::Local { model } | Choice::Hosted { model, .. } => model,
        }
    }
}

/// The reader's settings.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Settings {
    /// Whether the editor asks for completions as the reader types.
    #[serde(default)]
    pub on: bool,
    #[serde(default)]
    pub choice: Option<Choice>,
}

#[derive(Serialize, Deserialize)]
struct File {
    v: u32,
    #[serde(flatten)]
    settings: Settings,
}

/// The settings file.
pub fn file(state: &StateRoot) -> PathBuf {
    state.globals.join("lattice_native").join("completion.json")
}

/// The settings (off, with no model, when the file is missing or unreadable).
pub fn load(path: &Path) -> Settings {
    std::fs::read(path)
        .ok()
        .filter(|bytes| bytes.len() <= 64 * 1024)
        .and_then(|bytes| serde_json::from_slice::<File>(&bytes).ok())
        .map(|file| file.settings)
        .unwrap_or_default()
}

/// Write the settings whole: to a file beside it, then put in its place.
pub fn save(path: &Path, settings: &Settings) -> Result<(), String> {
    let failed = || "Lattice could not save the completion settings.".to_owned();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|_| failed())?;
    }
    let bytes = serde_json::to_vec_pretty(&File { v: 1, settings: settings.clone() }).map_err(|_| failed())?;
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = PathBuf::from(temporary);
    std::fs::write(&temporary, bytes).map_err(|_| failed())?;
    crate::fsx::replace_shared(&temporary, path).map_err(|_| failed())
}
