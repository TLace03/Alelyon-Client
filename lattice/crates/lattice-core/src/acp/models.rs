//! The model each lab agent answers with: the reader picks it per agent.
//!
//! - **What is offered** is what the adapter's `session/new` answers with
//!   ([`offered`]): Codex's `models` (each model at a reasoning effort, set
//!   with `session/set_model`), else a `model` config option (Claude Code's:
//!   `default`, `opus`, `sonnet`, ..., set with `session/set_config_option`).
//!   Each session the pool opens records it, and Check models opens one only
//!   to read it ([`super::pool::Pool::check_models`]).
//! - **The reader's pick** is kept per agent in `<agents dir>/models.json`
//!   with what was offered; every new session of that agent opens on it, set
//!   the way that adapter takes it ([`model_for`]). Without a pick each keeps
//!   its measured default ([`super::Agent::model`]). A session already open
//!   keeps the model it opened with.
//!
//! Nothing here reaches the network: the adapters do, with the reader's own
//! sign-in, when a session opens.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::Agent;
use super::session::Model;

/// Held while the file is read and written.
static WRITING: Mutex<()> = Mutex::new(());

/// One model an agent offers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Offer {
    pub id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
}

/// How an agent takes its model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum How {
    /// `session/set_model {modelId}`.
    SetModel,
    /// `session/set_config_option {configId: "model", value}`.
    ConfigOption,
}

/// What an agent offered when a session last opened.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Offered {
    pub how: Option<How>,
    pub models: Vec<Offer>,
    /// The model its session opened on.
    pub current: Option<String>,
}

fn text(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_owned()
}

/// The models a `session/new` answer offers: its `models` (set with
/// `session/set_model`), else its `model` config option, whose choices may be
/// grouped.
pub fn offered(reply: &Value) -> Offered {
    if let Some(models) = reply.get("models")
        && let Some(available) = models.get("availableModels").and_then(Value::as_array)
        && !available.is_empty()
    {
        return Offered {
            how: Some(How::SetModel),
            models: available
                .iter()
                .filter_map(|m| {
                    let id = text(m, "modelId");
                    (!id.is_empty()).then(|| Offer {
                        name: Some(text(m, "name"))
                            .filter(|n| !n.is_empty())
                            .unwrap_or_else(|| id.clone()),
                        description: text(m, "description"),
                        id,
                    })
                })
                .collect(),
            current: Some(text(models, "currentModelId")).filter(|c| !c.is_empty()),
        };
    }
    let option = reply
        .get("configOptions")
        .and_then(Value::as_array)
        .and_then(|all| {
            all.iter()
                .find(|o| o.get("id").and_then(Value::as_str) == Some("model"))
        });
    let Some(option) = option else {
        return Offered::default();
    };
    let mut models = Vec::new();
    let mut add = |choice: &Value| {
        let id = text(choice, "value");
        if !id.is_empty() && !models.iter().any(|m: &Offer| m.id == id) {
            models.push(Offer {
                name: Some(text(choice, "name"))
                    .filter(|n| !n.is_empty())
                    .unwrap_or_else(|| id.clone()),
                description: text(choice, "description"),
                id,
            });
        }
    };
    for choice in option
        .get("options")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match choice.get("options").and_then(Value::as_array) {
            // A group of choices.
            Some(group) => group.iter().for_each(&mut add),
            None => add(choice),
        }
    }
    Offered {
        how: (!models.is_empty()).then_some(How::ConfigOption),
        models,
        current: Some(text(option, "currentValue")).filter(|c| !c.is_empty()),
    }
}

#[derive(Default, Serialize, Deserialize)]
struct File {
    #[serde(default)]
    v: u32,
    #[serde(default)]
    picks: BTreeMap<String, String>,
    #[serde(default)]
    offered: BTreeMap<String, Offered>,
}

/// The file in the agents' folder.
pub fn file(dir: &Path) -> PathBuf {
    dir.join("models.json")
}

fn load(dir: &Path) -> File {
    std::fs::read(file(dir))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

fn save(dir: &Path, data: &File) -> Result<(), String> {
    let failed = || "Lattice could not keep the agents' models.".to_owned();
    std::fs::create_dir_all(dir).map_err(|_| failed())?;
    let bytes = serde_json::to_vec_pretty(data).map_err(|_| failed())?;
    let path = file(dir);
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = PathBuf::from(temporary);
    std::fs::write(&temporary, bytes).map_err(|_| failed())?;
    crate::fsx::replace_shared(&temporary, &path).map_err(|_| failed())
}

/// Keep what `agent` offered (a session opened, or Check models).
pub fn remember_offered(dir: &Path, agent: Agent, offered: &Offered) -> Result<(), String> {
    if offered.models.is_empty() {
        return Ok(());
    }
    let _held = WRITING.lock().unwrap_or_else(|p| p.into_inner());
    let mut data = load(dir);
    data.v = 1;
    data.offered
        .insert(agent.choice().to_owned(), offered.clone());
    save(dir, &data)
}

/// What `agent` last offered, and the reader's pick.
pub fn known(dir: &Path, agent: Agent) -> (Offered, Option<String>) {
    let data = load(dir);
    (
        data.offered
            .get(agent.choice())
            .cloned()
            .unwrap_or_default(),
        data.picks.get(agent.choice()).cloned(),
    )
}

/// Keep the reader's pick for `agent` (one it offered), or `None` for its default.
pub fn pick(dir: &Path, agent: Agent, model: Option<&str>) -> Result<(), String> {
    let _held = WRITING.lock().unwrap_or_else(|p| p.into_inner());
    let mut data = load(dir);
    match model {
        None => {
            data.picks.remove(agent.choice());
        }
        Some(model) => {
            let offered = data
                .offered
                .get(agent.choice())
                .cloned()
                .unwrap_or_default();
            if !offered.models.iter().any(|m| m.id == model) {
                return Err(format!(
                    "{} does not offer that model: check its models again.",
                    agent.label()
                ));
            }
            data.picks
                .insert(agent.choice().to_owned(), model.to_owned());
        }
    }
    data.v = 1;
    save(dir, &data)
}

/// The model a new session of `agent` opens on: the reader's pick, set the way
/// the agent offered it, else its default.
pub fn model_for(dir: &Path, agent: Agent) -> Model {
    let (offered, picked) = known(dir, agent);
    match (picked, offered.how) {
        (Some(model), Some(How::SetModel)) => Model::SetModel(model),
        (Some(model), Some(How::ConfigOption)) => Model::ConfigOption(model),
        _ => agent.model(),
    }
}
