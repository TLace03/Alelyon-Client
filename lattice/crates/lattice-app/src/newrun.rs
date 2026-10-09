//! The state and rules of the "New run" overlay.
//!
//! Invariants:
//! - The person is told where the task will go BEFORE they start it: the
//!   disclosure line is derived from the chosen model's locality, and a remote
//!   model's line names the model and says the task, the tool results and the
//!   conversation leave this machine.
//! - Start is available only with a non-empty task, a chosen agent, and a model
//!   that is ready. A model that is not ready stays in the list with its refusal
//!   text, and choosing it changes nothing but shows that text.
//! - A refusal from `start()` is shown as it came, in the overlay, and the
//!   overlay stays open with the task intact.
//! - The task text is never trimmed silently into something else: it is sent as
//!   typed apart from surrounding whitespace, which the service also trims.

use std::fmt;

use iced::widget::text_editor;
use lattice_protocol::{AgentInfo, Locality, ModelChoice, StartRun};

/// The widget id of the task editor, so opening the overlay can focus it.
pub const TASK_INPUT_ID: &str = "new-run-task";

/// The disclosure for a local model.
pub const LOCAL_DISCLOSURE: &str = "Runs on this machine. Nothing leaves it.";

/// What the overlay says about where a run goes.
pub fn disclosure(locality: Locality, model_label: &str) -> String {
    match locality {
        Locality::Local => LOCAL_DISCLOSURE.to_string(),
        Locality::Remote => format!(
            "This task, tool results and the conversation go to {model_label}, off this machine."
        ),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentOption {
    pub id: String,
    pub label: String,
}

impl fmt::Display for AgentOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.label)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelOption {
    pub id: String,
    pub label: String,
    pub locality: Locality,
    pub ready: bool,
    pub refusal: Option<String>,
}

impl fmt::Display for ModelOption {
    /// A ready model: `Local model · on this machine`. A model that is not
    /// ready: `Workstation model · unavailable: <why>`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.ready {
            let place = match self.locality {
                Locality::Local => "on this machine",
                Locality::Remote => "off this machine",
            };
            write!(f, "{} · {place}", self.label)
        } else {
            write!(
                f,
                "{} · unavailable: {}",
                self.label,
                self.refusal.as_deref().unwrap_or("not ready")
            )
        }
    }
}

impl From<&ModelChoice> for ModelOption {
    fn from(model: &ModelChoice) -> Self {
        Self {
            id: model.id.clone(),
            label: model.label.clone(),
            locality: model.locality,
            ready: model.ready,
            refusal: model.refusal.clone(),
        }
    }
}

impl From<&AgentInfo> for AgentOption {
    fn from(agent: &AgentInfo) -> Self {
        Self {
            id: agent.id.clone(),
            label: agent.label.clone(),
        }
    }
}

pub struct NewRun {
    pub task: text_editor::Content,
    pub agents: Vec<AgentOption>,
    pub models: Vec<ModelOption>,
    pub agent: Option<AgentOption>,
    pub model: Option<ModelOption>,
    /// A refusal to show: from `start()`, or the reason a model cannot be chosen.
    pub notice: Option<String>,
}

impl NewRun {
    /// The overlay as it opens: the first agent, and the first ready model,
    /// preferring one that stays on this machine.
    pub fn open(agents: &[AgentInfo], models: &[ModelChoice]) -> Self {
        let agents: Vec<AgentOption> = agents.iter().map(AgentOption::from).collect();
        let models: Vec<ModelOption> = models.iter().map(ModelOption::from).collect();
        let model = models
            .iter()
            .find(|m| m.ready && m.locality == Locality::Local)
            .or_else(|| models.iter().find(|m| m.ready))
            .cloned();
        Self {
            task: text_editor::Content::new(),
            agent: agents.first().cloned(),
            model,
            agents,
            models,
            notice: None,
        }
    }

    /// The task as typed.
    pub fn task_text(&self) -> String {
        self.task.text()
    }

    pub fn can_start(&self) -> bool {
        !self.task_text().trim().is_empty()
            && self.agent.is_some()
            && self.model.as_ref().is_some_and(|m| m.ready)
    }

    /// The disclosure for the chosen model, if there is one.
    pub fn disclosure(&self) -> Option<String> {
        self.model
            .as_ref()
            .map(|m| disclosure(m.locality, &m.label))
    }

    /// Choose a model. One that is not ready is refused: the choice stays as it
    /// was and the notice says why.
    pub fn pick_model(&mut self, model: ModelOption) {
        if model.ready {
            self.model = Some(model);
            self.notice = None;
        } else {
            self.notice = Some(
                model
                    .refusal
                    .clone()
                    .unwrap_or_else(|| "That model is not ready.".to_string()),
            );
        }
    }

    pub fn pick_agent(&mut self, agent: AgentOption) {
        self.agent = Some(agent);
        self.notice = None;
    }

    /// The request to make, or `None` while Start is not available.
    pub fn request(&self) -> Option<StartRun> {
        if !self.can_start() {
            return None;
        }
        Some(StartRun {
            task: self.task_text().trim().to_string(),
            agent: self.agent.as_ref()?.id.clone(),
            model: self.model.as_ref()?.id.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demo::{ADVISOR, ASSISTANT, DemoService, LOCAL_MODEL, OFFLINE_MODEL, REMOTE_MODEL};
    use lattice_protocol::RunService;

    fn overlay() -> NewRun {
        let service = DemoService::instant();
        NewRun::open(&service.agents(), &service.models())
    }

    fn type_task(overlay: &mut NewRun, text: &str) {
        overlay
            .task
            .perform(text_editor::Action::Edit(text_editor::Edit::Paste(
                std::sync::Arc::new(text.to_string()),
            )));
    }

    #[test]
    fn a_local_model_says_nothing_leaves_the_machine() {
        assert_eq!(
            disclosure(Locality::Local, "Local model"),
            "Runs on this machine. Nothing leaves it."
        );
    }

    #[test]
    fn a_remote_model_names_itself_and_says_the_task_leaves() {
        assert_eq!(
            disclosure(Locality::Remote, "Hosted model"),
            "This task, tool results and the conversation go to Hosted model, off this machine."
        );
    }

    #[test]
    fn the_overlay_opens_on_the_first_agent_and_a_ready_local_model() {
        let o = overlay();
        assert_eq!(o.agent.as_ref().unwrap().id, ASSISTANT);
        assert_eq!(o.model.as_ref().unwrap().id, LOCAL_MODEL);
        assert_eq!(o.disclosure().as_deref(), Some(LOCAL_DISCLOSURE));
        assert!(o.notice.is_none());
    }

    #[test]
    fn start_needs_a_task_an_agent_and_a_ready_model() {
        let mut o = overlay();
        assert!(!o.can_start() && o.request().is_none(), "no task yet");
        type_task(&mut o, "   \n  ");
        assert!(!o.can_start(), "whitespace is not a task");
        type_task(&mut o, "What time is it?");
        assert!(o.can_start());
        let request = o.request().unwrap();
        assert_eq!(
            request,
            StartRun {
                task: "What time is it?".into(),
                agent: ASSISTANT.into(),
                model: LOCAL_MODEL.into()
            }
        );
        o.agent = None;
        assert!(!o.can_start());
        o.agent = o.agents.first().cloned();
        o.model = None;
        assert!(!o.can_start());
    }

    #[test]
    fn an_unready_model_is_listed_with_its_refusal_but_cannot_be_chosen() {
        let mut o = overlay();
        let offline = o
            .models
            .iter()
            .find(|m| m.id == OFFLINE_MODEL)
            .cloned()
            .unwrap();
        let shown = offline.to_string();
        assert!(
            shown.contains("Workstation model")
                && shown.contains("unavailable:")
                && shown.contains("not running"),
            "{shown}"
        );
        o.pick_model(offline);
        assert_eq!(
            o.model.as_ref().unwrap().id,
            LOCAL_MODEL,
            "the choice did not change"
        );
        assert!(
            o.notice.as_deref().unwrap().contains("not running"),
            "the refusal text is shown"
        );
        // Choosing a ready model afterwards clears the notice.
        let remote = o
            .models
            .iter()
            .find(|m| m.id == REMOTE_MODEL)
            .cloned()
            .unwrap();
        o.pick_model(remote);
        assert_eq!(o.model.as_ref().unwrap().id, REMOTE_MODEL);
        assert!(o.notice.is_none());
    }

    #[test]
    fn choosing_a_remote_model_changes_the_disclosure_before_start() {
        let mut o = overlay();
        let remote = o
            .models
            .iter()
            .find(|m| m.id == REMOTE_MODEL)
            .cloned()
            .unwrap();
        o.pick_model(remote);
        assert_eq!(
            o.disclosure().as_deref(),
            Some(
                "This task, tool results and the conversation go to Hosted model, off this machine."
            )
        );
        let shown = o.model.as_ref().unwrap().to_string();
        assert!(shown.contains("off this machine"), "{shown}");
    }

    #[test]
    fn the_request_carries_the_chosen_agent_and_trimmed_task() {
        let mut o = overlay();
        let advisor = o.agents.iter().find(|a| a.id == ADVISOR).cloned().unwrap();
        o.pick_agent(advisor);
        type_task(&mut o, "  Which model translates contracts?  ");
        let request = o.request().unwrap();
        assert_eq!(request.agent, ADVISOR);
        assert_eq!(request.task, "Which model translates contracts?");
    }

    #[test]
    fn with_no_local_model_ready_the_first_ready_model_is_chosen() {
        let service = DemoService::instant();
        let mut models = service.models();
        models.retain(|m| m.locality == Locality::Remote || !m.ready);
        let o = NewRun::open(&service.agents(), &models);
        assert_eq!(o.model.as_ref().unwrap().id, REMOTE_MODEL);
        let none = NewRun::open(
            &service.agents(),
            &models
                .iter()
                .filter(|m| !m.ready)
                .cloned()
                .collect::<Vec<_>>(),
        );
        assert!(none.model.is_none() && none.disclosure().is_none() && !none.can_start());
    }
}
