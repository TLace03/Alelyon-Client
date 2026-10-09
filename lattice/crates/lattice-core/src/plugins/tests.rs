//! Plugins: added only from a folder on this PC holding a manifest, under a
//! name not taken; what one brings; switched off and taken off the list; a
//! store Lattice cannot read left as it is; its MCP servers with its folder
//! filled in; and its commands and skills joining the lists while it is on.

use std::sync::Arc;

use super::*;
use crate::testkit::TempDir;

fn store(dir: &TempDir) -> (Plugins, StateRoot) {
    let state = StateRoot::at(dir.path().join("state"));
    std::fs::create_dir_all(&state.globals).unwrap();
    let clock: Clock = Arc::new(|| 1_000.0);
    (Plugins::new(&state, clock), state)
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

/// A plugin folder in Claude Code's layout: a manifest, two commands, a
/// skill, an agent, hooks and an MCP server.
fn plugin(at: &Path, name: &str) {
    write(
        &at.join(".claude-plugin").join("plugin.json"),
        &format!(
            r#"{{"name": "{name}", "version": "0.1.0", "description": "Checks evidence.", "author": {{"name": "Alelyon"}}}}"#
        ),
    );
    write(
        &at.join("commands").join("check.md"),
        "---\ndescription: Check the evidence\n---\nCheck $ARGUMENTS.",
    );
    write(
        &at.join("commands").join("deep").join("audit.md"),
        "Audit it.",
    );
    write(
        &at.join("skills").join("evidence").join("SKILL.md"),
        "---\nname: evidence\ndescription: Tie results to evidence.\n---\nRead references/how.md.",
    );
    write(
        &at.join("skills")
            .join("evidence")
            .join("references")
            .join("how.md"),
        "Observed, declared, unmeasured.",
    );
    write(
        &at.join("agents").join("reviewer.md"),
        "---\nname: reviewer\n---\nReview.",
    );
    write(&at.join("hooks").join("hooks.json"), r#"{"hooks": {}}"#);
    write(
        &at.join(".mcp.json"),
        r#"{"mcpServers": {"evidence-server": {"command": "${CLAUDE_PLUGIN_ROOT}/bin/server.exe", "args": ["--root", "${CLAUDE_PLUGIN_ROOT}"]}}}"#,
    );
}

#[test]
fn a_plugin_is_added_from_a_folder_with_a_manifest_and_shows_what_it_brings() {
    let dir = TempDir::new("plugins-add");
    let (plugins, state) = store(&dir);
    let folder = dir.path().join("alelyon-plugin");
    plugin(&folder, "alelyon");
    let added = plugins.add(&folder).unwrap();
    assert_eq!(
        (
            added.name.as_str(),
            added.version.as_str(),
            added.author.as_str()
        ),
        ("alelyon", "0.1.0", "Alelyon")
    );
    assert!(added.on, "a plugin the reader chose is on");
    assert_eq!(added.commands, ["alelyon:check", "alelyon:deep:audit"]);
    assert_eq!(added.skills, ["alelyon:evidence"]);
    assert_eq!(added.agents, ["reviewer"]);
    assert!(added.hooks);
    assert_eq!(added.mcp_servers, ["evidence-server"]);
    assert_eq!(added.problem, None);
    assert_eq!(plugins.list().unwrap(), [added]);
    // Refused: a relative path, a share, a folder with no manifest, a bad
    // name, the same folder again, another folder under the same name, and
    // Lattice's own state.
    let bare = dir.path().join("bare");
    std::fs::create_dir_all(&bare).unwrap();
    let badly = dir.path().join("badly");
    write(
        &badly.join(".claude-plugin").join("plugin.json"),
        r#"{"name": "two words"}"#,
    );
    let twin = dir.path().join("twin");
    plugin(&twin, "Alelyon");
    let inside = state.globals.join("p");
    plugin(&inside, "inside");
    // A junction from outside Lattice's state into it (its real place is inside), and one inside it leading out (its
    // path is inside): each half of the check refuses one.
    let kept = state.globals.join("kept");
    plugin(&kept, "kept");
    let into = dir.path().join("into");
    std::fs::create_dir(&into).unwrap();
    lattice_sys::fs::seam::create_junction(&into, &kept).unwrap();
    let real = dir.path().join("real-plugin");
    plugin(&real, "real");
    let out = state.globals.join("out");
    std::fs::create_dir(&out).unwrap();
    lattice_sys::fs::seam::create_junction(&out, &real).unwrap();
    for (folder, why) in [
        (into.clone(), "Lattice's own files"),
        (out.clone(), "Lattice's own files"),
        (PathBuf::from("relative"), "a folder on this PC"),
        (
            PathBuf::from(r"\\server\share\plugin"),
            "a folder on this PC",
        ),
        (bare.clone(), "no .claude-plugin/plugin.json"),
        (badly.clone(), "needs a name"),
        (folder.clone(), "That plugin is on the list already"),
        (twin.clone(), "named Alelyon is on the list already"),
        (inside.clone(), "Lattice's own files"),
    ] {
        let refused = plugins.add(&folder).unwrap_err();
        assert!(refused.contains(why), "{}: {refused}", folder.display());
    }
}

#[test]
fn a_plugin_is_switched_off_and_taken_off_the_list_and_its_folder_stays() {
    let dir = TempDir::new("plugins-off");
    let (plugins, state) = store(&dir);
    let folder = dir.path().join("p");
    plugin(&folder, "tools");
    plugins.add(&folder).unwrap();
    assert_eq!(enabled(&state), [("tools".to_owned(), folder.clone())]);
    plugins.set_on("tools", false).unwrap();
    assert!(enabled(&state).is_empty());
    assert!(!plugins.list().unwrap()[0].on);
    plugins.set_on("tools", true).unwrap();
    plugins.remove("tools").unwrap();
    assert!(plugins.list().unwrap().is_empty());
    assert!(
        folder.join(".claude-plugin").join("plugin.json").is_file(),
        "never deleted"
    );
    assert!(plugins.set_on("tools", true).is_err());
}

#[test]
fn a_store_lattice_cannot_read_is_left_as_it_is() {
    let dir = TempDir::new("plugins-broken");
    let (plugins, state) = store(&dir);
    let file = state.native_chat_dir().join("plugins.json");
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    std::fs::write(&file, "{not json").unwrap();
    let folder = dir.path().join("p");
    plugin(&folder, "tools");
    assert!(plugins.add(&folder).is_err());
    assert!(enabled(&state).is_empty());
    assert_eq!(std::fs::read_to_string(&file).unwrap(), "{not json");
}

#[test]
fn a_plugins_mcp_servers_name_its_folder_and_are_checked_as_pasted_entries() {
    let dir = TempDir::new("plugins-mcp");
    let (plugins, _) = store(&dir);
    let folder = dir.path().join("p");
    plugin(&folder, "tools");
    plugins.add(&folder).unwrap();
    let entries = plugins.mcp_entries("tools").unwrap();
    assert_eq!(entries.len(), 1);
    let (name, entry) = &entries[0];
    assert_eq!(name, "evidence-server");
    let root = folder.display().to_string();
    assert_eq!(entry["command"], format!("{root}/bin/server.exe"));
    assert_eq!(entry["args"][1], root.as_str());
    write(
        &folder.join(".mcp.json"),
        r#"{"mcpServers": {"web": {"url": "https://example.com/mcp"}}}"#,
    );
    assert!(
        plugins.mcp_entries("tools").is_err(),
        "a remote server is refused as a pasted one is"
    );
}

#[test]
fn a_plugins_commands_and_skills_join_the_lists_while_it_is_on() {
    let dir = TempDir::new("plugins-lists");
    let (plugins, state) = store(&dir);
    let folder = dir.path().join("p");
    plugin(&folder, "alelyon");
    plugins.add(&folder).unwrap();
    // A plugin whose skills folder is a junction out of it: nothing there is read.
    let linked = dir.path().join("linked");
    write(
        &linked.join(".claude-plugin").join("plugin.json"),
        r#"{"name": "linked"}"#,
    );
    let outside = dir.path().join("outside");
    write(&outside.join("leak").join("SKILL.md"), "Outside text.");
    std::fs::create_dir(linked.join("skills")).unwrap();
    lattice_sys::fs::seam::create_junction(&linked.join("skills"), &outside).unwrap();
    // And its commands folder too.
    let elsewhere = dir.path().join("elsewhere");
    write(&elsewhere.join("x.md"), "Outside command.");
    std::fs::create_dir(linked.join("commands")).unwrap();
    lattice_sys::fs::seam::create_junction(&linked.join("commands"), &elsewhere).unwrap();
    plugins.add(&linked).unwrap();
    let commands = crate::commands::load(&state, None);
    let named: Vec<(&str, &str)> = commands
        .commands
        .iter()
        .map(|command| (command.name.as_str(), command.path.as_str()))
        .collect();
    assert_eq!(
        named,
        [
            ("alelyon:check", "(plugin alelyon) commands/check.md"),
            (
                "alelyon:deep:audit",
                "(plugin alelyon) commands/deep/audit.md"
            ),
        ]
    );
    assert!(
        commands
            .commands
            .iter()
            .all(|command| command.source == crate::commands::Source::Plugin)
    );
    assert!(
        commands
            .notices
            .join("\n")
            .contains("(plugin linked) commands/x.md could not be read"),
        "{:?}",
        commands.notices
    );
    let skills = crate::skills::load(&state, None);
    assert!(
        skills.find("linked:leak").is_none(),
        "a link out of the plugin is not followed"
    );
    let notices = skills.notices.join("\n");
    assert!(
        notices.contains("(plugin linked) skills/leak/SKILL.md could not be read"),
        "{notices}"
    );
    let skill = skills.find("alelyon:evidence").unwrap();
    assert_eq!(skill.source, crate::skills::Source::Plugin);
    assert!(
        skills
            .lead_text()
            .unwrap()
            .contains("\n- alelyon:evidence (plugin alelyon): Tie results to evidence.")
    );
    let crate::skills::Place::Local(skill_dir) = &skill.place else {
        panic!("{:?}", skill.place)
    };
    assert_eq!(
        crate::skills::read_user_file(skill_dir, "references/how.md").unwrap(),
        "Observed, declared, unmeasured."
    );
    plugins.set_on("alelyon", false).unwrap();
    plugins.set_on("linked", false).unwrap();
    assert!(crate::commands::load(&state, None).commands.is_empty());
    assert!(crate::skills::load(&state, None).skills.is_empty());
}
