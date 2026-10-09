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
