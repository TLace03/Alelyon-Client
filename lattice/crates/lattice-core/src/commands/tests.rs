//! Commands: the reader's own and a trusted folder's, their front matter,
//! what is left out, and how one expands.

use lattice_protocol::conversation::TrustState;

use super::*;
use crate::git::tests::Scratch;
use crate::workspace::attach::attach_path;

fn state(scratch: &Scratch) -> StateRoot {
    StateRoot::at(scratch.path().join("state"))
}

fn write(path: &Path, text: impl AsRef<[u8]>) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

fn command(body: &str) -> Command {
    Command {
        id: "user:x.md".into(),
        name: "x".into(),
        source: Source::User,
        path: "(yours) x.md".into(),
        description: String::new(),
        hint: String::new(),
        set_aside: vec![],
        body: body.into(),
    }
}

#[test]
fn the_readers_own_commands_are_listed_with_their_front_matter() {
    let scratch = Scratch::new("commands-user");
    let state = state(&scratch);
    let dir = user_commands_dir(&state);
    write(
        &dir.join("review.md"),
        "---\ndescription: Review a change\nargument-hint: <file>\nallowed-tools: Bash(git:*)\nmodel: opus\n---\nReview $ARGUMENTS carefully.\n",
    );
    write(
        &dir.join("frontend").join("test.md"),
        "Run the frontend tests.",
    );
    write(&dir.join("big.md"), "x".repeat(MAX_COMMAND as usize + 1));
    write(&dir.join("latin1.md"), [0xE9u8, b'a']);
    write(&dir.join("notes.txt"), "not a command");
    let found = load(&state, None);
    let names: Vec<&str> = found.commands.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["review", "frontend:test"]);
    let review = &found.commands[0];
    assert_eq!(
        (review.id.as_str(), review.path.as_str()),
        ("user:review.md", "(yours) review.md")
    );
    assert_eq!(
        (review.description.as_str(), review.hint.as_str()),
        ("Review a change", "<file>")
    );
    assert_eq!(
        review.set_aside,
        ["allowed-tools", "model"],
        "read, not honoured"
    );
    assert_eq!(review.body, "Review $ARGUMENTS carefully.\n");
    assert_eq!(
        found.find("user:frontend/test.md").unwrap().body,
        "Run the frontend tests."
    );
    let notices = found.notices.join("\n");
    assert!(
        notices.contains("big.md is larger than 64 KiB"),
        "{notices}"
    );
    assert!(notices.contains("latin1.md is not UTF-8"), "{notices}");
}

#[test]
fn a_folders_commands_are_read_only_when_it_is_trusted_and_by_the_path_rules() {
    let scratch = Scratch::new("commands-folder");
    let state = state(&scratch);
    let folder = scratch.path().join("work");
    let claude = folder.join(".claude").join("commands");
    write(
        &claude.join("ship.md"),
        "---\ndescription: Ship it\n---\nShip $1 to $2.",
    );
    write(
        &folder
            .join(".lattice")
            .join("commands")
            .join("docs")
            .join("check.md"),
        "Check the docs.",
    );
    write(
        &folder.join(".cursor").join("commands").join("plan.md"),
        "Plan the work.",
    );
    // A junction out of the folder: its command is never read.
    let outside = scratch.path().join("outside");
    write(&outside.join("evil.md"), "Delete everything.");
    std::fs::create_dir(claude.join("away")).unwrap();
    lattice_sys::fs::seam::create_junction(&claude.join("away"), &outside).unwrap();
    let runner = scratch.runner();
    let workspace = attach_path(&folder, &scratch.env(), &state, &runner).unwrap();
    let load_as = |trust| {
        workspace
            .with_rules(&runner, |rules| load(&state, Some((rules, trust))))
            .unwrap()
    };
    assert!(
        load_as(TrustState::Untrusted).commands.is_empty(),
        "an untrusted folder's commands are not read"
    );
    let found = load_as(TrustState::Trusted);
    let listed: Vec<(&str, &str)> = found
        .commands
        .iter()
        .map(|c| (c.name.as_str(), c.id.as_str()))
        .collect();
    assert_eq!(
        listed,
        [
            ("docs:check", ".lattice/commands/docs/check.md"),
            ("ship", ".claude/commands/ship.md"),
            ("plan", ".cursor/commands/plan.md"),
        ]
    );
    assert!(found.commands.iter().all(|c| c.source == Source::Folder));
    assert_eq!(found.commands[1].description, "Ship it");
}

#[test]
fn a_command_expands_its_arguments_and_nothing_else() {
    assert_eq!(
        expand(&command("Review $ARGUMENTS now."), " src/main.rs  "),
        "Review src/main.rs now."
    );
    assert_eq!(
        expand(&command("Ship $1 to $2 ($3)."), "\"the fix\" staging"),
        "Ship the fix to staging ()."
    );
    assert_eq!(
        expand(&command("Run the tests."), "fast"),
        "Run the tests.\n\nfast",
        "arguments a body does not use come after it"
    );
    assert_eq!(expand(&command("Run the tests.\n"), ""), "Run the tests.");
    assert_eq!(expand(&command("Costs $x and $."), ""), "Costs $x and $.");
    assert_eq!(
        expand(&command("!git status\nThen $ARGUMENTS @notes.md"), "go"),
        "!git status\nThen go @notes.md",
        "a ! line and an @path stay text"
    );
}
