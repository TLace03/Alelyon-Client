//! The agent moves the work to another project (`super::switch`), end to end
//! over the agent harness: the turn is told the reader's projects; the
//! reader is asked once; Stay here changes nothing; Switch begins a new topic
//! in the project's folder with the agent's prompt, recorded as moved on from
//! this topic; a project that is unknown, has no folder or holds this topic
//! already is refused with a sentence the model reads.

use lattice_agents::model::{InputItem, ModelRequest};
use lattice_protocol::conversation::{AgentChatService, ConversationEventKind};
use serde_json::json;

use super::agent_tests::{H, call, say, user_texts};
use super::switch::{self, HEADER, STAY, STAYED, SWITCH};
use crate::projects::Change;

const PROMPT: &str = "Fix the login page's error message on the website.";

fn project(h: &H, name: &str, folder: Option<String>) -> String {
    let made = h.runtime.block_on(h.chat.project_create(name.into())).unwrap();
    h.runtime
        .block_on(h.chat.project_change(made.id.clone(), Change { folder: Some(folder), ..Change::default() }))
        .unwrap();
    made.id
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

/// The question `call` asked in `id`, once it is asked: its text and options.
fn asked(h: &H, id: &str, call: &str) -> (String, Vec<String>) {
    let found = |events: &[lattice_protocol::conversation::ConversationEvent]| {
        events.iter().find_map(|e| match &e.kind {
            ConversationEventKind::Question { call_id, text, options } if call_id == call => Some((text.clone(), options.clone())),
            _ => None,
        })
    };
    found(&h.events_until(id, |events| found(events).is_some())).unwrap()
}

/// Mutants: the move made without the reader's word; Stay here moving it
/// anyway; the new topic sent no folder or no project.
#[test]
fn the_agent_moves_the_work_to_a_project_only_when_the_reader_says_switch() {
    let h = H::new("switch-project");
    let ws = h.workspace();
    let site = h.scratch.path().join("site");
    std::fs::create_dir_all(&site).unwrap();
    let website = project(&h, "Website", Some(site.to_string_lossy().into_owned()));
    let model = h.script(vec![
        call("switch_project", json!({"project": "website", "prompt": PROMPT, "why": "The login page is the website's."}), "s1"),
        say("Staying here."),
    ]);
    let first = h.agent(None, "Fix the login page's error message.", &ws);
    // The reader is asked once, and the turn was told the projects.
    let (text, options) = asked(&h, &first, "s1");
    assert!(user_texts(&model.calls()[0]).iter().any(|t| t.contains(HEADER) && t.contains("\u{201c}Website\u{201d}")));
    assert!(text.contains("\u{201c}Website\u{201d}") && text.contains(PROMPT) && text.contains("the website's"), "{text}");
    assert_eq!(options, [SWITCH, STAY]);
    h.chat.answer(&first, "s1", STAY.into()).unwrap();
    h.turns_end(&first, 1);
    assert_eq!(result_of(&model.calls(), "s1"), STAYED);
    let listed = |h: &H| h.runtime.block_on(h.chat.list()).unwrap().conversations;
    assert_eq!(listed(&h).len(), 1, "staying begins nothing");

    // Switch: a new topic in the project's folder, with the prompt, moved on from this one.
    model.enqueue(call("switch_project", json!({"project": "Website", "prompt": PROMPT}), "s2"));
    model.enqueue(say("Moved to the website."));
    model.enqueue(say("Looking at the login page."));
    let accepted = h.send(Some(&first), "Please, the website.", "local", super::agent_tests::local(), lattice_protocol::conversation::Mode::Agent, Some(&ws));
    assert!(accepted.is_ok(), "{accepted:?}");
    asked(&h, &first, "s2");
    h.chat.answer(&first, "s2", SWITCH.into()).unwrap();
    h.turns_end(&first, 2);
    let said = result_of(&model.calls(), "s2");
    assert!(said.contains("new topic in the project \u{201c}Website\u{201d}"), "{said}");
    // The project's folder is not trusted yet: it began in Ask mode, and the agent was told why.
    assert!(said.ends_with(switch::UNTRUSTED), "{said}");
    let moved = listed(&h).into_iter().find(|c| c.id != first).expect("a new topic");
    assert_eq!(moved.mode, lattice_protocol::conversation::Mode::Ask);
    let folder = moved.workspace.as_ref().expect("in a folder");
    assert_eq!(folder.name, "site", "{folder:?}");
    h.turns_end(&moved.id, 1);
    assert_eq!(h.texts(&moved.id)[0], PROMPT);
    assert_eq!(h.chat.inner.projects.of_chat(&moved.id).map(|p| p.id), Some(website.clone()));
    let origin = &h.chat.task_origins()[&moved.id];
    assert!(origin.switch && origin.from == first && origin.project.as_deref() == Some(website.as_str()), "{origin:?}");
}

/// Mutant: a project with no folder, an unknown one or this topic's own
/// begun anyway.
#[test]
fn unknown_folderless_and_own_projects_are_refused_with_a_sentence() {
    let h = H::new("switch-refused");
    let ws = h.workspace();
    project(&h, "Notes", None);
    let model = h.script(vec![
        call("switch_project", json!({"project": "Nowhere", "prompt": PROMPT}), "a"),
        call("switch_project", json!({"project": "Notes", "prompt": PROMPT}), "b"),
        call("switch_project", json!({"project": "Notes", "prompt": ""}), "c"),
        say("Done."),
    ]);
    let id = h.agent(None, "Tidy up.", &ws);
    h.turns_end(&id, 1);
    let calls = model.calls();
    assert!(result_of(&calls, "a").contains("no project by that name") && result_of(&calls, "a").contains("\u{201c}Notes\u{201d}"));
    assert!(result_of(&calls, "b").contains("has no folder"));
    assert!(result_of(&calls, "c").contains("A prompt is 1 to 2,000"));
    assert_eq!(h.runtime.block_on(h.chat.list()).unwrap().conversations.len(), 1);
}

#[test]
fn a_turn_is_told_the_other_projects_and_its_own_and_nothing_without_others() {
    let p = |id: &str, name: &str, archived: bool| -> crate::projects::Project {
        serde_json::from_value(json!({"id": id, "name": name, "archived": archived, "created": 1.0, "updated": 1.0})).unwrap()
    };
    let all = vec![p("a", "Website", false), p("b", "Lattice", false), p("c", "Old", true)];
    let text = switch::lead_text(&all, Some(&all[1])).unwrap();
    assert!(text.starts_with(HEADER));
    assert!(text.contains("\u{201c}Website\u{201d}.") && text.ends_with("This topic is in \u{201c}Lattice\u{201d}."), "{text}");
    assert!(!text.contains("Old"), "an archived project is not offered");
    assert_eq!(switch::lead_text(&all[1..2], Some(&all[1])), None, "no other project, nothing said");
    assert!(switch::lead_text(&all[..1], None).unwrap().ends_with("This topic is in none."));
}
