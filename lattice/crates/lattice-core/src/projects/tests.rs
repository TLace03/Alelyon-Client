//! Projects: the store (made, changed, archived, never written over when it
//! cannot be read), chats' membership, and what a project adds to a turn.

use std::sync::Arc;

use super::*;
use crate::testkit::TempDir;

fn store(dir: &TempDir) -> (Projects, StateRoot) {
    let state = StateRoot::at(dir.path().join("state"));
    std::fs::create_dir_all(&state.globals).unwrap();
    let clock: Clock = Arc::new(|| 1_000.0);
    (Projects::new(&state, clock), state)
}

#[test]
fn a_project_is_made_changed_and_archived_and_never_deleted() {
    let dir = TempDir::new("projects-store");
    let (projects, _) = store(&dir);
    assert!(projects.list().unwrap().is_empty());
    assert!(projects.create("  ").is_err(), "a name is needed");
    assert!(projects.create(&"x".repeat(MAX_NAME + 1)).is_err());
    let made = projects.create("  Launch plan ").unwrap();
    assert_eq!(made.name, "Launch plan");
    assert!(
        made.id.starts_with("p_") && made.id.len() == 14,
        "{}",
        made.id
    );
    let changed = projects
        .change(
            &made.id,
            Change {
                instructions: Some("Write for investors.".into()),
                choice: Some(Some("endpoint:hosted".into())),
                ..Change::default()
            },
        )
        .unwrap();
    assert_eq!(changed.instructions, "Write for investors.");
    assert_eq!(changed.choice.as_deref(), Some("endpoint:hosted"));
    assert!(
        projects
            .change(
                &made.id,
                Change {
                    instructions: Some("x".repeat(MAX_INSTRUCTIONS + 1)),
                    ..Change::default()
                }
            )
            .is_err()
    );
    let archived = projects
        .change(
            &made.id,
            Change {
                archived: Some(true),
                ..Change::default()
            },
        )
        .unwrap();
    assert!(archived.archived);
    assert_eq!(projects.list().unwrap().len(), 1, "archived, still listed");
    assert!(projects.change("p_nothere", Change::default()).is_err());
}

#[test]
fn a_chat_is_in_one_project_at_a_time() {
    let dir = TempDir::new("projects-chats");
    let (projects, _) = store(&dir);
    let first = projects.create("First").unwrap();
    let second = projects.create("Second").unwrap();
    projects.assign("c1", Some(&first.id)).unwrap();
    assert_eq!(projects.of_chat("c1").unwrap().id, first.id);
    projects.assign("c1", Some(&second.id)).unwrap();
    assert_eq!(projects.of_chat("c1").unwrap().id, second.id);
    let all = projects.list().unwrap();
    assert!(all[0].chats.is_empty() && all[1].chats == ["c1"]);
    projects.assign("c1", None).unwrap();
    assert!(projects.of_chat("c1").is_none());
    assert!(projects.assign("c1", Some("p_nothere")).is_err());
}

#[test]
fn a_store_lattice_cannot_read_is_left_as_it_is() {
    let dir = TempDir::new("projects-broken");
    let (projects, state) = store(&dir);
    let file = state.native_chat_dir().join("projects.json");
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    std::fs::write(&file, "{not json").unwrap();
    assert!(projects.list().is_err());
    assert!(projects.create("New").is_err());
    assert_eq!(std::fs::read_to_string(&file).unwrap(), "{not json");
}

#[test]
fn reference_files_are_local_and_never_lattices_own() {
    let dir = TempDir::new("projects-files");
    let (projects, state) = store(&dir);
    let made = projects.create("Docs").unwrap();
    let inside = state.globals.join("keys.env");
    let files = |list: Vec<String>| Change {
        files: Some(list),
        ..Change::default()
    };
    let refused = projects
        .change(&made.id, files(vec![inside.display().to_string()]))
        .unwrap_err();
    assert!(refused.contains("Lattice's own files"), "{refused}");
    for bad in [r"\\server\share\notes.md", "notes.md"] {
        assert!(
            projects.change(&made.id, files(vec![bad.into()])).is_err(),
            "{bad}"
        );
    }
    let many: Vec<String> = (0..=MAX_FILES).map(|n| format!(r"C:\x\{n}.md")).collect();
    assert!(projects.change(&made.id, files(many)).is_err());
}

#[test]
fn a_project_adds_its_instructions_and_readable_files_to_a_turn() {
    let dir = TempDir::new("projects-lead");
    let (projects, _) = store(&dir);
    let made = projects.create("Launch").unwrap();
    assert!(projects.lead_text(&made).is_none(), "nothing to add");
    let notes = dir.path().join("notes.md");
    std::fs::write(&notes, "Ship on Friday.").unwrap();
    let big = dir.path().join("big.txt");
    std::fs::write(&big, "y".repeat(MAX_FILE + 10)).unwrap();
    let binary = dir.path().join("logo.png");
    std::fs::write(&binary, [137u8, 80, 78, 71, 0, 1, 2]).unwrap();
    let missing = dir.path().join("gone.md");
    let folder = dir.path().to_path_buf();
    let changed = projects
        .change(
            &made.id,
            Change {
                instructions: Some("Be brief.".into()),
                files: Some(
                    [&notes, &binary, &missing, &folder, &big]
                        .iter()
                        .map(|path| path.display().to_string())
                        .collect(),
                ),
                ..Change::default()
            },
        )
        .unwrap();
    let text = projects.lead_text(&changed).unwrap();
    assert!(text.starts_with(HEADER), "{text}");
    assert!(text.contains("Project: Launch") && text.contains("Instructions:\nBe brief."));
    assert!(text.contains("Ship on Friday."));
    assert!(
        text.contains("big.txt (its first "),
        "a long file is cut to the room left"
    );
    assert!(text.contains("logo.png: left out, it is not text."));
    assert!(text.contains("gone.md: left out, it could not be opened."));
    assert!(text.contains(": left out, it is a folder."));
    assert!(text.len() <= MAX_LEAD);
}

#[test]
fn a_reference_file_whose_real_place_is_lattices_own_is_left_out() {
    let dir = TempDir::new("projects-own");
    let (projects, state) = store(&dir);
    std::fs::write(state.globals.join("keys.env"), "SECRET_TOKEN=x").unwrap();
    // A junction outside Lattice's state, into it: the path's text passes.
    let link = dir.path().join("notes");
    std::fs::create_dir(&link).unwrap();
    lattice_sys::fs::seam::create_junction(&link, &state.globals).unwrap();
    let made = projects.create("Docs").unwrap();
    let changed = projects
        .change(
            &made.id,
            Change {
                files: Some(vec![link.join("keys.env").display().to_string()]),
                ..Change::default()
            },
        )
        .unwrap();
    let text = projects.lead_text(&changed).unwrap();
    assert!(
        text.contains("keys.env: left out, it is in Lattice's own files."),
        "{text}"
    );
    assert!(!text.contains("SECRET_TOKEN"));
}
