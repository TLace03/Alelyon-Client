//! Checklists any agent gives the reader, outside Lattice's chats: a local
//! store (`tools/user_steps.py` writes it) the reader's window shows and ticks.
//!
//! - **Where:** `~/.alelyon/steps/`, or the folder `ALELYON_STEPS_DIR` names
//!   ([`dir`]).
//! - **A list** is two files, so the two sides never write the same one:
//!   `<id>.json`, written once by the agent (`title`, `steps`, `agent`,
//!   `folder`, `notify`), and `<id>.state.json`, written here, whole, beside
//!   itself first (`checked`, `notify`, `done`, `done_at`).
//! - **Done:** checking the last step marks the list done when its switch is
//!   on; with it off, the reader's Done does. The agent that asked waits on the
//!   state file (`user_steps.py wait`), so whatever agent it is learns it then.
//!   Done is kept: a later check does not undo it.
//! - Anything unreadable, oversized or not in this shape is left out, never
//!   guessed.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::env::Env;

/// The most of a list's file read.
pub const MAX_FILE: u64 = 64 * 1024;
/// The most steps a list has.
pub const MAX_STEPS: usize = 20;
/// The most lists shown.
pub const MAX_LISTS: usize = 50;

/// The store's folder.
pub fn dir(env: &dyn Env) -> PathBuf {
    match crate::env::text(env, "ALELYON_STEPS_DIR").filter(|d| !d.trim().is_empty()) {
        Some(named) => PathBuf::from(named),
        None => crate::state::home_dir(env, crate::state::Platform::host()).join(".alelyon").join("steps"),
    }
}

/// Whether `id` is a list's id: `steps_` and 16 lower-case hex digits.
pub fn is_id(id: &str) -> bool {
    lattice_protocol::conversation::is_steps_id(id)
}

/// One list, as the store has it.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalList {
    pub id: String,
    pub title: String,
    pub steps: Vec<String>,
    /// Who asked, as the agent named itself.
    pub agent: String,
    /// The folder it was asked from.
    pub folder: String,
    pub checked: Vec<bool>,
    pub notify: bool,
    pub done: bool,
    pub created: f64,
}

impl ExternalList {
    pub fn complete(&self) -> bool {
        !self.checked.is_empty() && self.checked.iter().all(|c| *c)
    }
}

#[derive(Default, Serialize, Deserialize)]
struct State {
    v: u32,
    #[serde(default)]
    checked: Vec<bool>,
    #[serde(default)]
    notify: Option<bool>,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    done_at: Option<f64>,
}

fn read_capped(path: &Path) -> Option<Vec<u8>> {
    use std::io::Read;
    let file = std::fs::File::open(path).ok()?;
    let mut bytes = Vec::new();
    file.take(MAX_FILE + 1).read_to_end(&mut bytes).ok()?;
    (bytes.len() as u64 <= MAX_FILE).then_some(bytes)
}

fn text(value: &Value, key: &str, most: usize) -> Option<String> {
    let text = value.get(key)?.as_str()?.trim();
    (!text.is_empty() && text.chars().count() <= most && !text.chars().any(char::is_control)).then(|| text.to_owned())
}

/// The list `id` in `dir`, or `None` when it is missing or not in shape.
pub fn read(dir: &Path, id: &str) -> Option<ExternalList> {
    if !is_id(id) {
        return None;
    }
    let given: Value = serde_json::from_slice(&read_capped(&dir.join(format!("{id}.json")))?).ok()?;
    if given.get("id").and_then(Value::as_str) != Some(id) {
        return None;
    }
    let title = text(&given, "title", 80)?;
    let steps: Vec<String> = given
        .get("steps")?
        .as_array()?
        .iter()
        .map(|s| {
            s.as_str()
                .map(str::trim)
                .filter(|s| !s.is_empty() && s.chars().count() <= 200 && !s.chars().any(char::is_control))
                .map(str::to_owned)
        })
        .collect::<Option<_>>()?;
    if steps.is_empty() || steps.len() > MAX_STEPS {
        return None;
    }
    let state: State = read_capped(&dir.join(format!("{id}.state.json")))
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let mut checked = state.checked;
    checked.resize(steps.len(), false);
    Some(ExternalList {
        id: id.to_owned(),
        agent: text(&given, "agent", 80).unwrap_or_else(|| "An agent".to_owned()),
        folder: given.get("folder").and_then(Value::as_str).unwrap_or("").to_owned(),
        notify: state.notify.unwrap_or_else(|| given.get("notify").and_then(Value::as_bool).unwrap_or(true)),
        done: state.done,
        created: given.get("created").and_then(Value::as_f64).unwrap_or(0.0),
        title,
        steps,
        checked,
    })
}

/// Every list not yet done, oldest first.
pub fn open(dir: &Path) -> Vec<ExternalList> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut found: Vec<ExternalList> = entries
        .filter_map(Result::ok)
        .filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_suffix(".json")).map(str::to_owned))
        .filter(|stem| is_id(stem))
        .filter_map(|id| read(dir, &id))
        .filter(|list| !list.done)
        .collect();
    found.sort_by(|a, b| a.created.total_cmp(&b.created).then(a.id.cmp(&b.id)));
    found.truncate(MAX_LISTS);
    found
}

fn write(dir: &Path, list: &ExternalList, at: f64) -> Result<(), String> {
    let state = State {
        v: 1,
        checked: list.checked.clone(),
        notify: Some(list.notify),
        done: list.done,
        done_at: list.done.then_some(at),
    };
    let bytes = serde_json::to_vec_pretty(&state).map_err(|_| "The checklist could not be saved.".to_owned())?;
    crate::fsx::atomic_write(&dir.join(format!("{}.state.json", list.id)), &bytes)
        .map_err(|_| "The checklist could not be saved.".to_owned())
}

/// The longest title, in characters.
pub const MAX_TITLE_CHARS: usize = 80;
/// The longest step, in characters.
pub const MAX_STEP_CHARS: usize = 200;
/// The longest agent's name, in characters.
pub const MAX_AGENT_CHARS: usize = 80;

fn one_line(text: &str, most: usize, what: &str) -> Result<String, String> {
    let text = crate::secrets::redact(text.trim());
    if text.is_empty() || text.chars().count() > most || text.chars().any(char::is_control) {
        return Err(format!("{what} is one line of 1 to {most} characters."));
    }
    Ok(text)
}

/// Give the reader a list: write `<id>.json` in `dir` once, as
/// `tools/user_steps.py add` writes it (`v`, `id`, `title`, `steps`, `agent`,
/// `folder`, `notify`, `created`), and return its id. Each text is one line,
/// redacted before it is written; a list that is not in shape is refused and
/// nothing is written. An empty `agent` is "An agent".
pub fn give(dir: &Path, title: &str, steps: &[String], agent: &str, folder: &str, notify: bool, at: f64) -> Result<String, String> {
    let title = one_line(title, MAX_TITLE_CHARS, "The title")?;
    if steps.is_empty() || steps.len() > MAX_STEPS {
        return Err(format!("A checklist has 1 to {MAX_STEPS} steps."));
    }
    let steps = steps.iter().map(|s| one_line(s, MAX_STEP_CHARS, "Each step")).collect::<Result<Vec<_>, _>>()?;
    let agent = if agent.trim().is_empty() { "An agent".to_owned() } else { one_line(agent, MAX_AGENT_CHARS, "The agent's name")? };
    let id = format!("steps_{}", &uuid::Uuid::new_v4().simple().to_string()[..16]);
    let value = serde_json::json!({
        "v": 1, "id": id, "title": title, "steps": steps, "agent": agent,
        "folder": folder, "notify": notify, "created": at,
    });
    let mut bytes = serde_json::to_vec_pretty(&value).map_err(|_| "The checklist could not be saved.".to_owned())?;
    bytes.push(b'\n');
    std::fs::create_dir_all(dir).map_err(|_| "The checklists' folder could not be made.".to_owned())?;
    match crate::fsx::write_once(&dir.join(format!("{id}.json")), &bytes) {
        Ok(true) => Ok(id),
        _ => Err("The checklist could not be saved.".to_owned()),
    }
}

fn found(dir: &Path, id: &str) -> Result<ExternalList, String> {
    read(dir, id).ok_or_else(|| "That checklist is gone.".to_owned())
}

/// The reader checked (or unchecked) step `step` (from 0) of `id`; done when
/// that completes a list whose switch is on.
pub fn check(dir: &Path, id: &str, step: usize, done: bool, at: f64) -> Result<ExternalList, String> {
    let mut list = found(dir, id)?;
    let Some(slot) = list.checked.get_mut(step) else { return Err("That step is not on the checklist.".to_owned()) };
    *slot = done;
    if list.notify && list.complete() {
        list.done = true;
    }
    write(dir, &list, at)?;
    Ok(list)
}

/// Whether finishing `id` marks it done by itself.
pub fn set_notify(dir: &Path, id: &str, notify: bool, at: f64) -> Result<ExternalList, String> {
    let mut list = found(dir, id)?;
    list.notify = notify;
    if notify && list.complete() {
        list.done = true;
    }
    write(dir, &list, at)?;
    Ok(list)
}

/// The reader's Done on `id`.
pub fn mark_done(dir: &Path, id: &str, at: f64) -> Result<ExternalList, String> {
    let mut list = found(dir, id)?;
    list.done = true;
    write(dir, &list, at)?;
    Ok(list)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::TempDir;

    /// A list as `tools/user_steps.py add` writes it.
    fn given(dir: &Path, id: &str, steps: &[&str], notify: bool) {
        std::fs::create_dir_all(dir).unwrap();
        let value = serde_json::json!({
            "v": 1, "id": id, "title": "Connect OpenRouter", "steps": steps,
            "agent": "Claude Code session 9b9e", "folder": "C:/work", "notify": notify, "created": 1.0,
        });
        std::fs::write(dir.join(format!("{id}.json")), value.to_string()).unwrap();
    }

    const A: &str = "steps_0011223344556677";
    const B: &str = "steps_8899aabbccddeeff";

    #[test]
    fn a_list_is_read_as_the_agent_wrote_it_and_checked_until_done() {
        let dir = TempDir::new("user-steps");
        let store = dir.path().join("steps");
        given(&store, A, &["Open the Models page.", "Sign in."], true);
        let open = super::open(&store);
        assert_eq!(open.len(), 1);
        assert_eq!((open[0].agent.as_str(), open[0].checked.clone()), ("Claude Code session 9b9e", vec![false, false]));
        let once = check(&store, A, 0, true, 2.0).unwrap();
        assert!(!once.done, "one of two");
        let twice = check(&store, A, 1, true, 3.0).unwrap();
        assert!(twice.done, "the last check finishes a list whose switch is on");
        assert!(super::open(&store).is_empty(), "done lists leave the open ones");
        let state: Value = serde_json::from_slice(&std::fs::read(store.join(format!("{A}.state.json"))).unwrap()).unwrap();
        assert_eq!(state["done"], true);
        assert_eq!(state["checked"], serde_json::json!([true, true]));
        assert_eq!(state["done_at"], 3.0);
        // Done is kept: a later uncheck does not undo it.
        assert!(check(&store, A, 1, false, 4.0).unwrap().done);
    }

    #[test]
    fn with_the_switch_off_only_done_finishes_and_bad_files_are_left_out() {
        let dir = TempDir::new("user-steps-off");
        let store = dir.path().join("steps");
        given(&store, B, &["Restart Alelyon."], false);
        assert!(!check(&store, B, 0, true, 1.0).unwrap().done, "switch off: checked, not done");
        assert!(!set_notify(&store, B, false, 1.5).unwrap().done);
        assert!(mark_done(&store, B, 2.0).unwrap().done);
        assert!(check(&store, B, 5, true, 3.0).is_err(), "a step past the end");
        assert!(check(&store, "steps_ffffffffffffffff", 0, true, 3.0).is_err(), "a list that is not there");
        // Not lists: a bad id, a file whose id differs, an empty list, a step of two lines.
        std::fs::write(store.join("notes.json"), "{}").unwrap();
        std::fs::write(store.join(format!("{A}.json")), serde_json::json!({"id": B, "title": "x", "steps": ["a"]}).to_string()).unwrap();
        given(&store, "steps_1111111111111111", &[], true);
        given(&store, "steps_2222222222222222", &["two\nlines"], true);
        assert!(super::open(&store).is_empty(), "{:?}", super::open(&store));
        // Switching it on when every step is checked finishes it.
        given(&store, "steps_3333333333333333", &["x"], false);
        check(&store, "steps_3333333333333333", 0, true, 1.0).unwrap();
        assert!(set_notify(&store, "steps_3333333333333333", true, 2.0).unwrap().done);
    }

    #[test]
    fn a_given_list_is_written_as_the_python_tool_writes_it_and_bad_lists_write_nothing() {
        let dir = TempDir::new("user-steps-give");
        let store = dir.path().join("steps");
        let steps = vec!["Open the Models page.".to_owned(), "Sign in.".to_owned()];
        let id = give(&store, "Connect a provider", &steps, "", "C:/work", true, 7.5).unwrap();
        assert!(is_id(&id));
        let written: Value = serde_json::from_slice(&std::fs::read(store.join(format!("{id}.json"))).unwrap()).unwrap();
        let mut keys: Vec<&str> = written.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["agent", "created", "folder", "id", "notify", "steps", "title", "v"]);
        assert_eq!((written["v"].clone(), written["agent"].clone(), written["created"].clone()), (1.into(), "An agent".into(), 7.5.into()));
        let list = read(&store, &id).unwrap();
        assert_eq!((list.title.as_str(), list.steps.clone(), list.notify), ("Connect a provider", steps.clone(), true));
        // Refused, and nothing written: no steps, too many, two lines, too long a title.
        let many: Vec<String> = (0..=MAX_STEPS).map(|n| format!("Step {n}.")).collect();
        assert!(give(&store, "x", &[], "", "", true, 1.0).unwrap_err().contains("1 to 20 steps"));
        assert!(give(&store, "x", &many, "", "", true, 1.0).is_err());
        assert!(give(&store, "x", &["two\nlines".to_owned()], "", "", true, 1.0).unwrap_err().contains("Each step"));
        assert!(give(&store, &"t".repeat(81), &steps, "", "", true, 1.0).unwrap_err().contains("The title"));
        assert_eq!(std::fs::read_dir(&store).unwrap().count(), 1);
    }

    #[test]
    fn the_store_is_the_named_folder_else_the_homes() {
        let env = crate::env::MapEnv::new().with("ALELYON_STEPS_DIR", "D:/steps");
        assert_eq!(dir(&env), PathBuf::from("D:/steps"));
        let env = crate::env::MapEnv::new().with("USERPROFILE", "D:/Profiles/r").with("HOME", "D:/Profiles/r");
        assert!(dir(&env).ends_with(Path::new(".alelyon").join("steps")));
    }
}
