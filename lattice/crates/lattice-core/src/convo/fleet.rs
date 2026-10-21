//! The fleet as another agent sees it: what the app's agent bridge answers for the reader's own agent (in another
//! harness, through `alelyon-cli`). Two reads, both blocking (they read the topics' records):
//!
//! - [`AgentChat::fleet_topics`]: one group's topics, each with the step under way, its to-do list, whether it is
//!   working, what it did last, and how many things it waits for from the reader;
//! - [`AgentChat::fleet_waiting`]: every question and approval waiting for the reader, in every topic, named by its
//!   group and topic, a question with the options it offered.
//!
//! [`AgentChat::fleet_state`] is the two whole-fleet reads together with a digest of them, so that a watcher (the
//! bridge's `fleet_watch`) answers only when something changed: a column, a wait, a step, what was spent.
//!
//! A question is answered through the chat's own `answer`, as the window answers it. An approval is listed and never
//! decided here: it widens what an agent may do, so only the reader's click in the window decides it.

use std::collections::HashMap;

use lattice_protocol::conversation::{TodoItem, TodoStatus};
use serde::Serialize;

use super::agent::AgentChat;
use super::item::{Item, Payload};
use super::task_tools::{self, Waiting};
use crate::tools::ask::AskQuestionArgs;

/// One topic of a group, as the fleet reads it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TopicView {
    pub id: String,
    pub title: String,
    /// A turn is under way.
    pub running: bool,
    /// The step under way: its latest to-do list's first step in progress.
    pub doing: Option<String>,
    pub todos_done: usize,
    pub todos_total: usize,
    /// The questions and approvals it waits for.
    pub waiting: usize,
    pub updated: f64,
    /// Its latest to-do list, step by step.
    pub todos: Vec<TodoItem>,
    /// What it did last, newest first: its last tool calls as their summaries say (secrets already redacted when they
    /// were recorded), at most [`RECENT`].
    pub recent: Vec<String>,
}

/// How many of a topic's last actions [`TopicView::recent`] holds.
pub const RECENT: usize = 5;

/// One thing waiting for the reader.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Wait {
    /// The group the topic is in, and its name as shown (`None`: in no group).
    pub group: Option<String>,
    pub group_name: Option<String>,
    pub topic: String,
    pub topic_title: String,
    pub call: String,
    /// `question` or `approval`.
    pub kind: &'static str,
    /// The question, or what the approval would allow.
    pub text: String,
    /// The options a question offered (an approval has none).
    pub options: Vec<String>,
}

/// The whole fleet at one moment: every group's card and everything waiting, with a digest of both.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FleetState {
    /// Changes when anything shown below changes; equal digests, equal fleets.
    pub digest: String,
    pub groups: Vec<GroupCard>,
    pub waiting: Vec<Wait>,
}

/// A group's card with its project.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GroupCard {
    pub project: Option<String>,
    #[serde(flatten)]
    pub card: super::board::Card,
}

/// The options `ask_question` offered in the call `call`, from its recorded arguments.
fn options(items: &[Item], call: &str) -> Vec<String> {
    items
        .iter()
        .find_map(|item| match item {
            Item::ToolCall { call_id, tool, arguments: Payload::Inline(args), .. }
                if call_id == call && tool == "ask_question" =>
            {
                serde_json::from_str::<AskQuestionArgs>(args).ok().map(|a| a.options)
            }
            _ => None,
        })
        .unwrap_or_default()
}

impl AgentChat {
    fn titles(&self) -> HashMap<String, (String, f64)> {
        match self.inner.config.store.list() {
            crate::chat::store::IndexState::Rows(rows) => {
                rows.into_iter().map(|r| (r.id, (r.title, r.updated))).collect()
            }
            _ => HashMap::new(),
        }
    }

    /// Every group's card, project by project (work in no project first), each project's highest priority first.
    pub fn fleet_groups(&self) -> Vec<(Option<String>, super::board::Card)> {
        let (groups, _) = self.groups();
        let mut projects: Vec<Option<String>> =
            groups.iter().filter(|(g, _)| !g.archived).map(|(g, _)| g.project.clone()).collect();
        projects.sort();
        projects.dedup();
        projects
            .into_iter()
            .flat_map(|project| {
                let cards = self.board(project.as_deref());
                cards.into_iter().map(move |card| (project.clone(), card))
            })
            .collect()
    }

    /// Every group's card and everything waiting, with their digest (`FleetState`).
    pub fn fleet_state(&self) -> FleetState {
        use std::hash::{Hash, Hasher};
        let groups: Vec<GroupCard> =
            self.fleet_groups().into_iter().map(|(project, card)| GroupCard { project, card }).collect();
        let waiting = self.fleet_waiting();
        let shown = serde_json::to_string(&(&groups, &waiting)).unwrap_or_default();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        shown.hash(&mut hasher);
        FleetState { digest: format!("{:016x}", hasher.finish()), groups, waiting }
    }

    /// The reader's sessions that match `query` by the words said in them (`super::recall`), the best match of each,
    /// best first, at most `limit` sessions: to find one and pick it up where it stopped.
    pub fn search_sessions(&self, query: &str, limit: usize) -> Vec<super::recall::Hit> {
        let mut recall = self.inner.recall.lock().unwrap_or_else(|p| p.into_inner());
        recall.refresh(&*self.inner.config.store);
        let mut seen = std::collections::HashSet::new();
        recall
            .search(query, None, limit.saturating_mul(4).max(limit))
            .into_iter()
            .filter(|hit| seen.insert(hit.conversation.clone()))
            .take(limit)
            .collect()
    }

    /// The topics of the group `group`, in the order they joined (`None`: no such group).
    pub fn fleet_topics(&self, group: &str) -> Option<Vec<TopicView>> {
        let group = self.inner.groups.list().into_iter().find(|g| g.id == group)?;
        let titles = self.titles();
        Some(
            group
                .topics
                .iter()
                .filter_map(|id| {
                    let (title, updated) = titles.get(id)?.clone();
                    let items = self.inner.sidecars.read_items(id).map(|log| log.items).unwrap_or_default();
                    let todos = super::todos::latest(&items);
                    Some(TopicView {
                        id: id.clone(),
                        title,
                        running: self.inner.loaded(id).is_some_and(|c| c.state().running.is_some()),
                        doing: todos.iter().find(|t| t.status == TodoStatus::InProgress).map(|t| t.content.clone()),
                        todos_done: todos.iter().filter(|t| t.status == TodoStatus::Done).count(),
                        todos_total: todos.len(),
                        waiting: task_tools::waiting(&self.inner, id).len(),
                        updated,
                        recent: items
                            .iter()
                            .rev()
                            .filter_map(|item| match item {
                                Item::ToolCall { summary, .. } if !summary.trim().is_empty() => Some(summary.clone()),
                                _ => None,
                            })
                            .take(RECENT)
                            .collect(),
                        todos,
                    })
                })
                .collect(),
        )
    }

    /// Every question and approval waiting for the reader, the most recently active topic first.
    pub fn fleet_waiting(&self) -> Vec<Wait> {
        let mut titles: Vec<(String, (String, f64))> = self.titles().into_iter().collect();
        titles.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
        let (groups, _) = self.groups();
        let mut out = Vec::new();
        for (topic, (topic_title, _)) in titles {
            let waits = task_tools::waiting(&self.inner, &topic);
            if waits.is_empty() {
                continue;
            }
            let group = groups.iter().find(|(g, _)| g.topics.contains(&topic));
            let items = self.inner.sidecars.read_items(&topic).map(|log| log.items).unwrap_or_default();
            for wait in waits {
                let (call, kind, text) = match wait {
                    Waiting::Question { call, text } => (call, "question", text),
                    Waiting::Approval { call, what } => (call, "approval", what),
                };
                out.push(Wait {
                    group: group.map(|(g, _)| g.id.clone()),
                    group_name: group.map(|(_, name)| name.clone()),
                    topic: topic.clone(),
                    topic_title: topic_title.clone(),
                    options: if kind == "question" { options(&items, &call) } else { Vec::new() },
                    call,
                    kind,
                    text,
                });
            }
        }
        out
    }
}
