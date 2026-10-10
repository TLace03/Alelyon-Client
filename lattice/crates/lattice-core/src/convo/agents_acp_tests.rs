//! The labs' agents through the whole chat service (`crate::acp`): a chat
//! sent with an agent's choice is answered by that agent and saved as any
//! answer is. By hand only: it spends a little of the reader's
//! subscriptions, and needs the adapters installed in the folder
//! `LATTICE_ACP_AGENTS` names (linked into the scratch state's agents folder).

use lattice_protocol::Shown;
use lattice_protocol::conversation::{Accepted, Mode};

use super::agent_tests::{H, statuses};
use crate::acp::{Agent, agents_dir};

#[test]
#[ignore]
fn a_chat_with_an_agents_choice_is_answered_by_that_agent() {
    let Some(adapters) = std::env::var_os("LATTICE_ACP_AGENTS") else {
        panic!("set LATTICE_ACP_AGENTS")
    };
    // The reader's own home, app data and PATH: each CLI finds its login there.
    let real: Vec<(String, String)> = ["USERPROFILE", "HOME", "APPDATA", "LOCALAPPDATA", "PATH"]
        .iter()
        .filter_map(|name| {
            std::env::var(name)
                .ok()
                .map(|value| ((*name).to_owned(), value))
        })
        .collect();
    let extra: Vec<(&str, &str)> = real.iter().map(|(n, v)| (n.as_str(), v.as_str())).collect();
    let h = H::with("acp-chat", &extra, |_| {});
    let dir = agents_dir(&h.state);
    std::fs::create_dir_all(dir.parent().unwrap()).unwrap();
    let linked = std::process::Command::new("cmd")
        .args(["/c", "mklink", "/J"])
        .arg(&dir)
        .arg(&adapters)
        .output()
        .unwrap();
    assert!(linked.status.success(), "{linked:?}");
    for agent in Agent::ALL {
        let shown = Shown {
            locality: lattice_protocol::Locality::Remote,
            label: agent.label().to_owned(),
        };
        let accepted = h
            .send(
                None,
                "Reply with the single word: ready. Do not use any tools.",
                agent.choice(),
                shown,
                Mode::Ask,
                None,
            )
            .unwrap();
        let Accepted::Started { conversation, .. } = accepted else {
            panic!("{accepted:?}")
        };
        let events = h.turns_end(&conversation.id, 1);
        let texts = h.texts(&conversation.id);
        let errors: Vec<String> = h
            .store
            .load(&conversation.id)
            .into_iter()
            .map(|t| t.error)
            .collect();
        eprintln!(
            "{}: {:?} {:?} errors {:?}",
            agent.label(),
            statuses(&events),
            texts,
            errors
        );
        assert!(
            texts.last().unwrap().to_lowercase().contains("ready"),
            "{texts:?}"
        );
    }
    // The link, not the adapters, goes.
    let _ = std::fs::remove_dir(&dir);
}

/// By hand, as above: each agent's edit waits for the core's own dialog,
/// which shows the change. Declined, the file is as it was; allowed, it
/// changes. Codex opens in the mode where it asks before every edit.
#[test]
#[ignore]
fn an_agents_edit_shows_its_change_and_waits_for_the_readers_yes() {
    let Some(adapters) = std::env::var_os("LATTICE_ACP_AGENTS") else {
        panic!("set LATTICE_ACP_AGENTS")
    };
    let real: Vec<(String, String)> = ["USERPROFILE", "HOME", "APPDATA", "LOCALAPPDATA", "PATH"]
        .iter()
        .filter_map(|name| std::env::var(name).ok().map(|value| ((*name).to_owned(), value)))
        .collect();
    let extra: Vec<(&str, &str)> = real.iter().map(|(n, v)| (n.as_str(), v.as_str())).collect();
    let h = H::with("acp-gate", &extra, |_| {});
    let dir = agents_dir(&h.state);
    std::fs::create_dir_all(dir.parent().unwrap()).unwrap();
    let linked = std::process::Command::new("cmd")
        .args(["/c", "mklink", "/J"])
        .arg(&dir)
        .arg(&adapters)
        .output()
        .unwrap();
    assert!(linked.status.success(), "{linked:?}");
    for agent in Agent::ALL {
        for allow in [false, true] {
            let shown = || Shown {
                locality: lattice_protocol::Locality::Remote,
                label: agent.label().to_owned(),
            };
            // A first turn opens the agent's session in a folder of its own
            // (the reader says yes to sending off this machine).
            *h.confirm.answer.lock().unwrap() = true;
            let accepted = h
                .send(
                    None,
                    "Reply with the single word: ready. Do not use any tools.",
                    agent.choice(),
                    shown(),
                    Mode::Agent,
                    None,
                )
                .unwrap();
            let Accepted::Started { conversation, .. } = accepted else {
                panic!("{accepted:?}")
            };
            let id = conversation.id.clone();
            h.turns_end(&id, 1);
            let file = dir.join("work").join(&id).join("gate.txt");
            std::fs::write(&file, "hello world\n").unwrap();
            *h.confirm.answer.lock().unwrap() = allow;
            h.send(
                Some(&id),
                "In gate.txt, change the word hello to goodbye with your file editing tool, not a shell command. Then reply done.",
                agent.choice(),
                shown(),
                Mode::Agent,
                None,
            )
            .unwrap();
            h.turns_end(&id, 2);
            let asked: Vec<Vec<String>> = h
                .confirm
                .asked()
                .into_iter()
                .filter_map(|request| match request {
                    crate::ports::ConfirmRequest::AgentPermission { diff, .. } => Some(diff),
                    _ => None,
                })
                .collect();
            let now = std::fs::read_to_string(&file).unwrap();
            eprintln!(
                "{} allow={allow}: file {now:?}; last change shown {:?}",
                agent.label(),
                asked.last()
            );
            let shown = asked.last().expect("it asked");
            assert!(shown.iter().any(|l| l.starts_with("gate.txt")), "{shown:?}");
            assert!(shown.iter().any(|l| l.starts_with("- ") && l.contains("hello")), "{shown:?}");
            assert!(shown.iter().any(|l| l.starts_with("+ ") && l.contains("goodbye")), "{shown:?}");
            if allow {
                assert!(now.contains("goodbye"), "{now:?}");
            } else {
                assert_eq!(now, "hello world\n", "declined: nothing written");
            }
        }
    }
    let _ = std::fs::remove_dir(&dir);
}

/// By hand, as above, with LATTICE_UNDO_FOLDER naming an empty folder outside
/// the app-data folders: an agent's turn in a git folder is watched; what it
/// changed is one effect, named in a note, and listed in the Changes panel.
#[test]
#[ignore]
fn an_agents_turn_in_a_git_folder_lists_what_it_changed() {
    use lattice_protocol::conversation::{AgentChatService, ConversationEventKind};
    let (Some(adapters), Some(folder)) = (
        std::env::var_os("LATTICE_ACP_AGENTS"),
        std::env::var_os("LATTICE_UNDO_FOLDER").map(std::path::PathBuf::from),
    ) else {
        panic!("set LATTICE_ACP_AGENTS and LATTICE_UNDO_FOLDER")
    };
    let real: Vec<(String, String)> = ["USERPROFILE", "HOME", "APPDATA", "LOCALAPPDATA", "PATH"]
        .iter()
        .filter_map(|name| std::env::var(name).ok().map(|value| ((*name).to_owned(), value)))
        .collect();
    let extra: Vec<(&str, &str)> = real.iter().map(|(n, v)| (n.as_str(), v.as_str())).collect();
    let h = H::with("acp-undo", &extra, |_| {});
    let dir = agents_dir(&h.state);
    std::fs::create_dir_all(dir.parent().unwrap()).unwrap();
    let linked = std::process::Command::new("cmd")
        .args(["/c", "mklink", "/J"])
        .arg(&dir)
        .arg(&adapters)
        .output()
        .unwrap();
    assert!(linked.status.success(), "{linked:?}");
    let git = |args: &[&str]| {
        let out = std::process::Command::new("git").args(args).current_dir(&folder).output().unwrap();
        assert!(out.status.success(), "{args:?}: {out:?}");
    };
    git(&["init", "-q"]);
    std::fs::write(folder.join("gate.txt"), "hello world\n").unwrap();
    git(&["add", "gate.txt"]);
    git(&["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "start"]);
    let view = h.runtime.block_on(h.chat.attach_native(None, folder.clone())).unwrap();
    let ws = view.workspace.id.clone();
    h.runtime.block_on(h.chat.trust(&ws)).unwrap();
    *h.confirm.answer.lock().unwrap() = true;
    for agent in Agent::ALL {
        std::fs::write(folder.join("gate.txt"), "hello world\n").unwrap();
        let shown = Shown {
            locality: lattice_protocol::Locality::Remote,
            label: agent.label().to_owned(),
        };
        let accepted = h
            .send(
                None,
                "In gate.txt, change the word hello to goodbye with your file editing tool, not a shell command. Then reply done.",
                agent.choice(),
                shown,
                Mode::Agent,
                Some(&ws),
            )
            .unwrap();
        let Accepted::Started { conversation, .. } = accepted else {
            panic!("{accepted:?}")
        };
        let events = h.turns_end(&conversation.id, 1);
        let effect: Vec<String> = events
            .iter()
            .find_map(|event| match &event.kind {
                ConversationEventKind::CommandEffect { files, .. } => {
                    Some(files.iter().map(|f| f.path.clone()).collect())
                }
                _ => None,
            })
            .unwrap_or_default();
        let notes: Vec<String> = events
            .iter()
            .filter_map(|event| match &event.kind {
                ConversationEventKind::Notice { text } => Some(text.clone()),
                _ => None,
            })
            .collect();
        let changes = h.runtime.block_on(h.chat.changes(&conversation.id)).unwrap().changes;
        eprintln!("{}: effect {effect:?}; notes {notes:?}; changes {:?}", agent.label(),
            changes.iter().map(|c| (&c.path, c.kind)).collect::<Vec<_>>());
        assert_eq!(effect, ["gate.txt"]);
        assert!(notes.iter().any(|n| n.starts_with(&format!("{} changed 1 file: gate.txt", agent.label()))), "{notes:?}");
        let change = changes.iter().find(|c| c.path == "gate.txt").expect("listed").id.clone();
        // Undo stages putting it back; Keep of that writes it, and the folder
        // is free for the next chat.
        h.runtime
            .block_on(h.chat.review(
                &conversation.id,
                vec![lattice_protocol::conversation::ReviewOp::Undo { change, hunks: None, note: None }],
            ))
            .map(|outcome| eprintln!("  undo: {:?}", outcome.results))
            .unwrap();
        let back = h
            .runtime
            .block_on(h.chat.changes(&conversation.id))
            .unwrap()
            .changes
            .into_iter()
            .find(|c| c.path == "gate.txt" && c.kind == lattice_protocol::conversation::ChangeKind::CommandUndo)
            .expect("the undo staged")
            .id;
        h.runtime
            .block_on(h.chat.review(
                &conversation.id,
                vec![lattice_protocol::conversation::ReviewOp::Keep { change: back, hunks: None }],
            ))
            .map(|outcome| eprintln!("  keep: {:?}", outcome.results))
            .unwrap();
        assert_eq!(std::fs::read_to_string(folder.join("gate.txt")).unwrap(), "hello world
");
    }
    let _ = std::fs::remove_dir(&dir);
}
