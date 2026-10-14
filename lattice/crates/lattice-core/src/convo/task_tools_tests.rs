//! The main chat's task tools (`super::task_tools`), end to end over the agent
//! harness: a task started from a chip asks a question; the main chat's agent
//! lists it with what it waits for, answers it with the user's words, sends it
//! an instruction, and stops it; a topic that is not a task is refused, and a
//! message is never routed into a task's thread.

use std::sync::Arc;

use futures::stream::BoxStream;
use lattice_agents::model::{GenerationTrace, InputItem, Model, ModelError, ModelEvent, ModelRequest};
use lattice_agents::testing::ScriptedModel;
use lattice_protocol::conversation::{Accepted, ConversationEventKind, Mode};
use serde_json::{Value, json};

use super::agent_tests::{H, call, local, say};
use super::recall::Route;
use super::task_tools::HEADER;

const PROMPT: &str = "Update the README badge to point at build.yml, then report.";

/// Answers the task's turns from one script and the main chat's from another,
/// told apart by the task's prompt in the request.
struct Router {
    main: Arc<ScriptedModel>,
    task: Arc<ScriptedModel>,
}

impl Model for Router {
    fn name(&self) -> &str {
        "router"
    }
    fn config_for_trace(&self) -> Value {
        Value::Null
    }
    fn generation_trace(&self) -> GenerationTrace {
        GenerationTrace::Bare
    }
    fn stream(&self, request: ModelRequest) -> BoxStream<'static, Result<ModelEvent, ModelError>> {
        let is_task = request.input.iter().any(|item| matches!(item, InputItem::User(text) if text.contains(PROMPT)));
        if is_task { self.task.stream(request) } else { self.main.stream(request) }
    }
}

fn result_of(model: &ScriptedModel, call_id: &str) -> String {
    model
        .calls()
        .iter()
        .flat_map(|request| request.input.iter())
        .find_map(|item| match item {
            InputItem::ToolResult { call_id: id, output } if id == call_id => Some(output.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no result for {call_id}"))
}

/// Mutants: a question relayed to the wrong place; a message to a topic that
/// is not a task; the task's thread chosen by routing.
#[test]
fn the_main_chat_lists_answers_messages_and_stops_its_tasks() {
    let h = H::new("task-tools");
    let ws = h.workspace();
    let main = Arc::new(ScriptedModel::new([
        call("suggest_task", json!({"title": "Fix the badge", "summary": "The badge names the old workflow.", "prompt": PROMPT}), "s1"),
        say("I suggested a task."),
    ]));
    let task = Arc::new(ScriptedModel::new([call(
        "ask_question",
        json!({"question": "Which branch should the badge follow?", "options": ["main", "release"]}),
        "q1",
    )]));
    h.answer_with(Arc::new(Router { main: main.clone(), task: task.clone() }));
    let id = h.agent(None, "Fix the build.", &ws);
    let events = h.turns_end(&id, 1);
    let suggested = events
        .iter()
        .find_map(|e| match &e.kind {
            ConversationEventKind::TaskSuggested { task, .. } => Some(task.clone()),
            _ => None,
        })
        .unwrap();
    let Accepted::Started { conversation: started, .. } =
        h.runtime.block_on(h.chat.start_task(&id, &suggested, "local".into(), local(), Mode::Agent)).unwrap()
    else {
        panic!("started")
    };
    let topic = started.id.clone();
    h.events_until(&topic, |events| events.iter().any(|e| matches!(e.kind, ConversationEventKind::Question { .. })));

    // The main chat's agent lists the task and what it waits for, then answers with the user's word.
    main.enqueue(call("list_tasks", json!({}), "l1"));
    main.enqueue(call("answer_task", json!({"task": topic, "call_id": "q1", "answer": "main"}), "a1"));
    main.enqueue(call("answer_task", json!({"task": id, "call_id": "q1", "answer": "x"}), "a2"));
    main.enqueue(say("Answered the task."));
    task.enqueue(say("The badge follows main now."));
    let sent = h.send(Some(&id), "What do my tasks need? Tell the badge one: main.", "local", local(), Mode::Agent, Some(&ws));
    assert!(sent.is_ok(), "{sent:?}");
    h.turns_end(&id, 2);
    let listed = result_of(&main, "l1");
    assert!(listed.starts_with(HEADER) && listed.contains(&topic) && listed.contains("\u{201c}Fix the badge\u{201d}"), "{listed}");
    assert!(listed.contains("waits for the user's answer (q1): Which branch should the badge follow?"), "{listed}");
    assert!(result_of(&main, "a1").contains("Answered"));
    assert!(result_of(&main, "a2").contains("not one of the user's tasks"), "the main topic is no task");
    h.turns_end(&topic, 1);
    let answered = task.calls().last().unwrap().input.iter().any(|item| matches!(item, InputItem::ToolResult { call_id, output } if call_id == "q1" && output.contains("main")));
    assert!(answered, "the task heard the user's answer");

    // An instruction to the task, and a stop of one that is not working.
    main.enqueue(call("message_task", json!({"task": topic, "text": "Also update the docs badge."}), "m1"));
    main.enqueue(call("stop_task", json!({"task": topic}), "x1"));
    main.enqueue(say("Done."));
    task.enqueue(say("Docs badge updated."));
    let sent = h.send(Some(&id), "Tell it to fix the docs badge too.", "local", local(), Mode::Agent, Some(&ws));
    assert!(sent.is_ok(), "{sent:?}");
    h.turns_end(&id, 3);
    assert!(result_of(&main, "m1").contains("Fix the badge"), "{}", result_of(&main, "m1"));
    h.turns_end(&topic, 2);
    assert!(h.texts(&topic).iter().any(|t| t == "Also update the docs badge."));

    // A message about the task's own words is not routed into its thread.
    assert_ne!(h.chat.route("badge build.yml README follows main docs", Some(&id)), Route::Return(topic.clone()));
}
