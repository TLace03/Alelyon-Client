//! Projects in agent turns ("Both"): a new chat sent in a project
//! joins it, and every turn of a project's chat leads with the project's
//! instructions and reference files, before the folder's rules; a chat with
//! no folder in such a project is an agent turn in either mode, offering only
//! `ask_question` there; a chat put in a project later gets it from its next
//! turn; a new chat that runs as a plain turn joins its project too, and a
//! plain turn says it does not carry the project's instructions.

use std::sync::Arc;

use lattice_agents::model::{InputItem, ModelRequest};
use lattice_protocol::conversation::{
    Accepted, ConversationEventKind, Mode, SendRequest, TurnKind,
};

use super::agent::words;
use super::agent_tests::{H, local, say};
use super::caps::{ModelCaps, Tri};
use crate::projects::{Change, HEADER};

fn lead(request: &ModelRequest) -> String {
    match request.input.first() {
        Some(InputItem::User(text)) => text.clone(),
        other => panic!("{other:?}"),
    }
}

fn send(
    h: &H,
    conversation: Option<&str>,
    mode: Mode,
    workspace: Option<&str>,
    project: Option<&str>,
) -> (String, TurnKind) {
    match h
        .runtime
        .block_on(h.chat.send(SendRequest {
            conversation: conversation.map(str::to_owned),
            text: "Draft the summary.".into(),
            choice: "local".into(),
            shown: local(),
            mode,
            workspace: workspace.map(str::to_owned),
            edit_of: None,
            project: project.map(str::to_owned),
            images: Vec::new(),
        }))
        .unwrap()
    {
        Accepted::Started {
            conversation,
            turn_kind,
            ..
        } => (conversation.id, turn_kind),
        other => panic!("{other:?}"),
    }
}

use lattice_protocol::conversation::AgentChatService;

fn project_with(h: &H, instructions: &str, files: Vec<String>) -> String {
    let made = h
        .runtime
        .block_on(h.chat.project_create("Launch".into()))
        .unwrap();
    h.runtime
        .block_on(h.chat.project_change(
            made.id.clone(),
            Change {
                instructions: Some(instructions.into()),
                files: Some(files),
                ..Change::default()
            },
        ))
        .unwrap();
    made.id
}

#[test]
fn a_new_chat_in_a_project_joins_it_and_leads_with_its_instructions_and_files() {
    let h = H::new("projects-turn");
    let ws = h.workspace();
    let notes = h.scratch.path().join("brief.md");
    std::fs::write(&notes, "The launch is on Friday.").unwrap();
    let project = project_with(
        &h,
        "Write for investors.",
        vec![notes.display().to_string()],
    );
    let model = h.script(vec![say("Here it is.")]);
    let (id, kind) = send(&h, None, Mode::Agent, Some(&ws), Some(&project));
    assert_eq!(kind, TurnKind::Agent);
    h.turns_end(&id, 1);
    let projects = h.runtime.block_on(h.chat.projects()).unwrap();
    assert_eq!(
        projects[0].chats,
        [id.clone()],
        "the chat joined its project"
    );
    let lead = lead(&model.calls()[0]);
    assert!(lead.starts_with(HEADER), "{lead}");
    assert!(lead.contains("Project: Launch"));
    assert!(lead.contains("Instructions:\nWrite for investors."));
    assert!(lead.contains("The launch is on Friday."));
}

#[test]
fn a_chat_with_no_folder_in_a_project_is_an_agent_turn_in_either_mode() {
    let h = H::new("projects-nofolder");
    // No project: Ask mode with no folder is the plain turn.
    h.script(vec![say("plain")]);
    let (_, kind) = send(&h, None, Mode::Ask, None, None);
    assert_eq!(kind, TurnKind::Plain);
    let project = project_with(&h, "Answer in French.", vec![]);
    let model = h.script(vec![say("Bonjour.")]);
    let (id, kind) = send(&h, None, Mode::Ask, None, Some(&project));
    assert_eq!(
        kind,
        TurnKind::Agent,
        "the project's instructions reach the model"
    );
    h.turns_end(&id, 1);
    let first = &model.calls()[0];
    let tools: Vec<&str> = first.tools.iter().map(|tool| tool.name.as_str()).collect();
    assert_eq!(
        tools,
        [
            "ask_question",
            "suggest_task",
            "withdraw_task",
            "write_artifact",
            "read_artifact",
            "update_todos"
        ]
    );
    assert!(lead(first).contains("Answer in French."));
}

#[test]
fn a_chat_put_in_a_project_later_gets_it_from_its_next_turn() {
    let h = H::new("projects-later");
    let ws = h.workspace();
    let model = h.script(vec![say("first")]);
    let (id, _) = send(&h, None, Mode::Agent, Some(&ws), None);
    h.turns_end(&id, 1);
    assert!(
        !model.calls()[0]
            .input
            .iter()
            .any(|item| matches!(item, InputItem::User(text) if text.starts_with(HEADER)))
    );
    let project = project_with(&h, "Be brief.", vec![]);
    h.runtime
        .block_on(h.chat.project_assign(id.clone(), Some(project)))
        .unwrap();
    let model = h.script(vec![say("second")]);
    send(&h, Some(&id), Mode::Agent, None, None);
    h.turns_end(&id, 2);
    assert!(lead(&model.calls()[0]).contains("Be brief."));
}

#[test]
fn a_plain_turn_joins_its_project_and_says_what_it_does_not_carry() {
    let h = H::with("projects-plain", &[], |config| {
        config.caps = Some(Arc::new(|_| ModelCaps {
            tools: Tri::No,
            vision: Tri::Unknown,
            context_tokens: None,
        }));
    });
    let said = |events: &[lattice_protocol::conversation::ConversationEvent]| {
        events.iter().any(|event| {
            matches!(&event.kind, ConversationEventKind::Notice { text } if text == words::PROJECT_PLAIN)
        })
    };
    let quiet = h
        .runtime
        .block_on(h.chat.project_create("Quiet".into()))
        .unwrap();
    h.script(vec![say("plain")]);
    let (id, kind) = send(&h, None, Mode::Ask, None, Some(&quiet.id));
    assert_eq!(kind, TurnKind::Plain);
    assert!(
        !said(&h.turns_end(&id, 1)),
        "the project has nothing to carry"
    );
    let projects = h.runtime.block_on(h.chat.projects()).unwrap();
    assert_eq!(
        projects[0].chats,
        [id.clone()],
        "the plain chat joined its project"
    );
    let project = project_with(&h, "Answer in French.", vec![]);
    h.script(vec![say("plain")]);
    let (other, kind) = send(&h, None, Mode::Agent, None, Some(&project));
    assert_eq!(kind, TurnKind::Plain, "this model cannot call tools here");
    assert!(said(&h.turns_end(&other, 1)));
}
