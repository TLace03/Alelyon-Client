//! Skills in agent turns ("plugins"): a turn lists the reader's
//! own skills and a trusted folder's in its leading item, after the rules,
//! and offers `use_skill` and `read_skill_file` in both modes with the skills'
//! note; a turn without a skill offers neither.

use lattice_agents::model::{InputItem, ModelRequest};
use lattice_protocol::conversation::{Accepted, Mode};
use serde_json::json;

use super::agent_tests::{H, call, local, say};
use super::prompt_agent::SKILLS_NOTE;
use crate::skills::{HEADER, user_skills_dir};

fn lead(request: &ModelRequest) -> String {
    match request.input.first() {
        Some(InputItem::User(text)) => text.clone(),
        other => panic!("{other:?}"),
    }
}

fn tool_names(request: &ModelRequest) -> Vec<String> {
    request.tools.iter().map(|tool| tool.name.clone()).collect()
}

/// What the tool calls returned, in order.
fn results(request: &ModelRequest) -> Vec<String> {
    request
        .input
        .iter()
        .filter_map(|item| match item {
            InputItem::ToolResult { output, .. } => Some(output.clone()),
            _ => None,
        })
        .collect()
}

fn write(path: &std::path::Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

fn send(h: &H, mode: Mode, workspace: Option<&str>) -> String {
    match h
        .send(None, "Tidy the notes.", "local", local(), mode, workspace)
        .unwrap()
    {
        Accepted::Started { conversation, .. } => conversation.id,
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_turn_lists_the_skills_and_reads_one_and_its_file() {
    let h = H::new("skills-turn");
    let mine = user_skills_dir(&h.state).join("guide");
    write(
        &mine.join("SKILL.md"),
        "---\ndescription: How the user keeps notes.\n---\nRead references/steps.md first.\n",
    );
    write(
        &mine.join("references").join("steps.md"),
        "Step one: sort by date.",
    );
    write(
        &h.folder
            .join(".claude")
            .join("skills")
            .join("house")
            .join("SKILL.md"),
        "---\nname: house\ndescription: This folder's house style.\n---\nUse short lines.\n",
    );
    write(
        &h.folder
            .join(".claude")
            .join("skills")
            .join("house")
            .join("style.md"),
        "Lines under 80 characters.",
    );
    let ws = h.workspace();
    let model = h.script(vec![
        call("use_skill", json!({"name": "guide"}), "c1"),
        call(
            "read_skill_file",
            json!({"name": "guide", "path": "references/steps.md"}),
            "c2",
        ),
        call(
            "read_skill_file",
            json!({"name": "house", "path": "style.md"}),
            "c3",
        ),
        call("use_skill", json!({"name": "missing"}), "c4"),
        say("Done."),
    ]);
    let id = send(&h, Mode::Ask, Some(&ws));
    h.turns_end(&id, 1);
    let calls = model.calls();
    let first = &calls[0];
    let tools = tool_names(first);
    assert!(tools.iter().any(|name| name == "use_skill"), "{tools:?}");
    assert!(
        tools.iter().any(|name| name == "read_skill_file"),
        "both modes"
    );
    assert!(first.system.ends_with(SKILLS_NOTE));
    let lead = lead(first);
    assert!(lead.contains(HEADER), "{lead}");
    assert!(lead.contains("\n- guide (yours): How the user keeps notes."));
    assert!(lead.contains("\n- house (.claude/skills/house): This folder's house style."));
    let answered = results(calls.last().unwrap());
    assert!(
        answered[0].starts_with(
            "Skill guide ((yours) guide/SKILL.md):\n\nRead references/steps.md first."
        )
    );
    assert!(
        answered[0]
            .ends_with("Its other files (read one with read_skill_file): references/steps.md")
    );
    assert_eq!(answered[1], "Step one: sort by date.");
    assert!(
        answered[2].contains("Lines under 80 characters."),
        "read as read_file reads: {}",
        answered[2]
    );
    assert!(
        answered[3].contains("There is no skill named missing"),
        "{}",
        answered[3]
    );
}

#[test]
fn a_turn_without_a_skill_offers_no_skill_tool() {
    let h = H::new("skills-none");
    let ws = h.workspace();
    let model = h.script(vec![say("Done.")]);
    let id = send(&h, Mode::Agent, Some(&ws));
    h.turns_end(&id, 1);
    let first = &model.calls()[0];
    let tools = tool_names(first);
    assert!(
        !tools.iter().any(|name| name.contains("skill")),
        "{tools:?}"
    );
    assert!(!first.system.contains(SKILLS_NOTE));
    assert!(!lead(first).contains(HEADER));
}

#[test]
fn a_skill_step_is_named_by_the_skill_and_its_file() {
    use super::views::call_summary;
    assert_eq!(
        call_summary("use_skill", r#"{"name": "guide"}"#),
        ("use_skill guide".to_owned(), Some("guide".to_owned()))
    );
    assert_eq!(
        call_summary(
            "read_skill_file",
            r#"{"name": "guide", "path": "references/steps.md"}"#
        ),
        (
            "read_skill_file guide: references/steps.md".to_owned(),
            Some("guide: references/steps.md".to_owned())
        )
    );
}
