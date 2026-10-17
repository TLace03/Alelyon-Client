//! Steering helpers (`super::helpers`), end to end over the agent harness. The
//! agent and its helpers each answer from a scripted model of their own (a
//! helper's run is told apart by its instructions), so a helper working in
//! the background never takes the agent's next step: a helper sent to work in
//! the background answers through `wait_agent`; a message reaches a working
//! helper before its next step; a helper that answered is asked again with
//! its task and answer as its input; one still working when its turn ends is
//! stopped and says so; a closed helper is forgotten; and `spawn_agents`
//! gives back an answer per task.

use std::sync::Arc;
use std::time::Duration;

use futures::stream::BoxStream;
use lattice_agents::model::{
    GenerationTrace, InputItem, Model, ModelError, ModelEvent, ModelRequest,
};
use lattice_agents::testing::{ScriptedModel, ScriptedStep};
use serde_json::{Value, json};

use super::agent_tests::{H, call, say};
use super::helpers::{EDITING_HELPER_SYSTEM, ENDED_WITH_TURN, HELPER_SYSTEM};
use super::prompt_agent;

/// The agent's steps from one script, its helpers' from another.
pub(crate) struct Routed {
    pub(crate) agent: ScriptedModel,
    pub(crate) helper: ScriptedModel,
}

impl Model for Routed {
    fn name(&self) -> &str {
        "scripted-model"
    }

    fn config_for_trace(&self) -> Value {
        Value::Null
    }

    fn generation_trace(&self) -> GenerationTrace {
        GenerationTrace::Bare
    }

    fn stream(&self, request: ModelRequest) -> BoxStream<'static, Result<ModelEvent, ModelError>> {
        if request.system == HELPER_SYSTEM || request.system == EDITING_HELPER_SYSTEM {
            self.helper.stream(request)
        } else {
            self.agent.stream(request)
        }
    }
}

pub(crate) fn routed(h: &H, agent: Vec<ScriptedStep>, helper: ScriptedModel) -> Arc<Routed> {
    let model = Arc::new(Routed {
        agent: ScriptedModel::new(agent),
        helper,
    });
    h.answer_with(model.clone());
    model
}

fn result_of(calls: &[ModelRequest], call_id: &str) -> String {
    calls
        .iter()
        .flat_map(|request| request.input.iter())
        .find_map(|item| match item {
            InputItem::ToolResult { call_id: id, output } if id == call_id => Some(output.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no result for {call_id}"))
}

#[test]
fn a_background_helper_works_while_the_agent_does_and_wait_gives_back_its_answer() {
    let h = H::new("helper-background");
    let ws = h.workspace();
    let model = routed(
        &h,
        vec![
            call("spawn_agent", json!({"task": "What does a.txt hold?", "background": true}), "c1"),
            call("wait_agent", json!({}), "c2"),
            // Nothing is left to give back.
            call("wait_agent", json!({}), "c3"),
            say("It holds a."),
        ],
        ScriptedModel::new(vec![
            call("read_file", json!({"path": "a.txt"}), "h1"),
            say("a.txt holds the letter a."),
        ]),
    );
    let id = h.agent(None, "ask a helper in the background", &ws);
    h.turns_end(&id, 1);
    let calls = model.agent.calls();
    assert_eq!(
        result_of(&calls, "c1"),
        "Helper a1 is working. wait_agent gives back its answer; it is stopped if your turn ends first."
    );
    assert_eq!(result_of(&calls, "c2"), "[Helper a1 answered]\na.txt holds the letter a.");
    assert_eq!(
        result_of(&calls, "c3"),
        "No helper is working, and every answer has been given back."
    );
    assert_eq!(model.helper.calls().len(), 2, "the helper read, then answered");
}

#[test]
fn a_message_reaches_a_working_helper_before_its_next_step() {
    let h = H::new("helper-steer");
    let ws = h.workspace();
    // The helper's first step takes a while, so the message arrives during it.
    let helper = ScriptedModel::new(vec![
        call("read_file", json!({"path": "a.txt"}), "h1"),
        say("a.txt holds a; and I checked b too."),
    ])
    .with_delay(Duration::from_millis(400));
    let model = routed(
        &h,
        vec![
            call("spawn_agent", json!({"task": "Read a.txt.", "background": true}), "c1"),
            call("send_agent", json!({"agent": "a1", "message": "STEER: check b too"}), "c2"),
            call("wait_agent", json!({"agents": ["a1"]}), "c3"),
            say("Done."),
        ],
        helper,
    );
    let id = h.agent(None, "steer a helper", &ws);
    h.turns_end(&id, 1);
    let calls = model.agent.calls();
    assert_eq!(result_of(&calls, "c2"), "Helper a1 reads your message before its next step.");
    let helper = model.helper.calls();
    assert_eq!(helper.len(), 2);
    assert!(
        format!("{:?}", helper[1].input).contains("STEER: check b too"),
        "its next step sees the message: {:?}",
        helper[1].input
    );
    assert!(!format!("{:?}", helper[0].input).contains("STEER"));
    assert_eq!(
        result_of(&calls, "c3"),
        "[Helper a1 answered]\na.txt holds a; and I checked b too."
    );
}

#[test]
fn a_helper_that_answered_is_asked_again_with_its_task_and_answer() {
    let h = H::new("helper-resume");
    let ws = h.workspace();
    let model = routed(
        &h,
        vec![
            call("spawn_agent", json!({"task": "TASK: what is in a.txt?"}), "c1"),
            call("send_agent", json!({"agent": "a1", "message": "AGAIN: and its length?"}), "c2"),
            call("wait_agent", json!({"agents": ["a1"]}), "c3"),
            say("Done."),
        ],
        ScriptedModel::new(vec![say("FIRST: the letter a."), say("SECOND: one letter.")]),
    );
    let id = h.agent(None, "ask a helper twice", &ws);
    h.turns_end(&id, 1);
    let calls = model.agent.calls();
    assert_eq!(
        result_of(&calls, "c1"),
        "FIRST: the letter a.\n\n[This was helper a1: send_agent asks it more.]"
    );
    assert_eq!(
        result_of(&calls, "c2"),
        "Helper a1 is working on it, with its task and its earlier answers. wait_agent gives back its answer."
    );
    assert_eq!(result_of(&calls, "c3"), "[Helper a1 answered]\nSECOND: one letter.");
    let helper = model.helper.calls();
    assert_eq!(
        helper[1].input,
        vec![
            InputItem::User("TASK: what is in a.txt?".into()),
            InputItem::Assistant { text: Some("FIRST: the letter a.".into()), tool_calls: Vec::new() },
            InputItem::User("AGAIN: and its length?".into()),
        ]
    );
}

#[test]
fn a_helper_still_working_when_its_turn_ends_is_stopped_and_says_so() {
    let h = H::new("helper-turn-end");
    let ws = h.workspace();
    let helper = ScriptedModel::new(vec![say("too late")]).with_delay(Duration::from_secs(3));
    let model = routed(
        &h,
        vec![
            call("spawn_agent", json!({"task": "Read a.txt slowly.", "background": true}), "c1"),
            say("I answer without waiting."),
            // The next turn.
            call("wait_agent", json!({}), "c2"),
            call("close_agent", json!({"agent": "a1"}), "c3"),
            call("wait_agent", json!({"agents": ["a1"]}), "c4"),
            say("Closed."),
        ],
        helper,
    );
    let id = h.agent(None, "send a slow helper", &ws);
    h.turns_end(&id, 1);
    let next = h.agent(Some(&id), "what did it say?", &ws);
    assert_eq!(next, id);
    h.turns_end(&id, 2);
    let calls = model.agent.calls();
    assert_eq!(result_of(&calls, "c2"), format!("[Helper a1 has no answer] {ENDED_WITH_TURN}"));
    assert_eq!(result_of(&calls, "c3"), "Helper a1 is closed.");
    assert!(result_of(&calls, "c4").contains("There is no helper a1"), "{}", result_of(&calls, "c4"));
}

#[test]
fn spawn_agents_gives_back_an_answer_per_task() {
    let h = H::new("helper-fan-out");
    let ws = h.workspace();
    let tasks: Vec<String> = (1..=5).map(|n| format!("Task number {n}.")).collect();
    let model = routed(
        &h,
        vec![
            call("spawn_agents", json!({"tasks": tasks}), "c1"),
            call("spawn_agents", json!({"tasks": vec!["t"; 17]}), "c2"),
            call("spawn_agents", json!({"tasks": ["fine", "  "]}), "c3"),
            say("Done."),
        ],
        ScriptedModel::new((0..5).map(|_| say("ANSWERED"))),
    );
    let id = h.agent(None, "fan out", &ws);
    h.turns_end(&id, 1);
    let calls = model.agent.calls();
    let fanned = result_of(&calls, "c1");
    for n in 1..=5 {
        assert!(fanned.contains(&format!("[Task {n} (helper a")), "{fanned}");
    }
    assert_eq!(fanned.matches("answered]\nANSWERED").count(), 5, "{fanned}");
    assert_eq!(model.helper.calls().len(), 5);
    assert!(result_of(&calls, "c2").contains("At most 16 tasks"));
    assert!(result_of(&calls, "c3").contains("Every task needs its words"));
}

#[test]
fn the_steering_tools_are_offered_with_a_folder_only() {
    for mode in [lattice_protocol::conversation::Mode::Ask, lattice_protocol::conversation::Mode::Agent] {
        let tools = prompt_agent::tools(mode);
        for name in ["spawn_agents", "send_agent", "wait_agent", "close_agent"] {
            assert!(tools.iter().any(|t| t.name == name), "{name} in {mode:?}");
        }
    }
    let without = prompt_agent::tools_without_folder();
    for name in ["spawn_agents", "send_agent", "wait_agent", "close_agent"] {
        assert!(!without.iter().any(|t| t.name == name), "{name}: no folder, no helper");
    }
}
