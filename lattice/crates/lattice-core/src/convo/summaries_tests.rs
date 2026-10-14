//! Topic summaries (`super::summaries`): when one is due, what the request
//! asks from, what a reply gives, what a turn's leading item gets, the store
//! (the reader's model kept, an unreadable file never written over), and,
//! over the agent harness, a topic summarised after its tenth turn and the
//! next topic's first turn told about it.

use super::agent_tests::{H, local, say, user_texts};
use super::summaries::{self, EVERY, HEADER_OTHERS, HEADER_OWN, MIN_TURNS, Summaries, Summary};
use crate::chat::store::StoredTurn;
use crate::chat::transcript::NewTurn;
use crate::state::StateRoot;
use crate::testkit::TempDir;
use lattice_agents::model::OutputItem;
use lattice_protocol::conversation::Mode;

fn summary(text: &str, turns: usize) -> Summary {
    Summary { text: text.into(), turns, at: 1.0, by: "Local".into() }
}

fn turn(role: &str, text: &str) -> StoredTurn {
    let new = if role == "assistant" { NewTurn::assistant(text, "local") } else { NewTurn::user(text) };
    new.stored("t".into(), 1.0)
}

#[test]
fn a_topic_is_due_after_ten_turns_then_every_eight_and_the_store_keeps_the_readers_model() {
    let dir = TempDir::new("summaries-store");
    let state = StateRoot::at(dir.path().join("state"));
    let s = Summaries::new(&state);
    assert!(!s.due("t", MIN_TURNS - 1));
    assert!(s.due("t", MIN_TURNS));
    s.keep("t", summary("Fixing the tax.", MIN_TURNS)).unwrap();
    assert!(!s.due("t", MIN_TURNS + EVERY - 1));
    assert!(s.due("t", MIN_TURNS + EVERY));
    assert_eq!(s.get("t").unwrap().text, "Fixing the tax.");
    assert_eq!(s.choice(), None, "Lattice picks until the reader does");
    s.set_choice(Some("endpoint:hosted".into())).unwrap();
    assert_eq!(s.choice().as_deref(), Some("endpoint:hosted"));
    assert_eq!(s.get("t").unwrap().turns, MIN_TURNS, "the choice kept the summaries");
    // A file Lattice cannot read is never written over.
    let path = state.native_chat_dir().join("summaries.json");
    std::fs::write(&path, "not json").unwrap();
    assert!(s.keep("t", summary("x", 30)).is_err());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "not json");
}

#[test]
fn the_request_holds_the_newest_turns_oldest_first_and_a_reply_is_trimmed_and_cut() {
    let turns = vec![turn("user", "Fix the tax."), turn("assistant", "Done: the page no longer adds it.")];
    let request = summaries::request("Payment page", &turns);
    let lattice_agents::model::InputItem::User(text) = &request.input[0] else { panic!("one user item") };
    assert!(text.starts_with("Topic: Payment page"));
    assert!(text.find("User: Fix the tax.").unwrap() < text.find("Agent: Done").unwrap(), "{text}");
    assert!(request.tools.is_empty());
    let reply = vec![OutputItem::Message { text: "<think>plan</think>  The user fixed the tax.  ".into() }];
    assert_eq!(summaries::text_of(&reply).as_deref(), Some("The user fixed the tax."));
    let long = vec![OutputItem::Message { text: "x".repeat(5_000) }];
    assert_eq!(summaries::text_of(&long).unwrap().chars().count(), summaries::MAX_CHARS);
    assert_eq!(summaries::text_of(&[]), None);
}

#[test]
fn a_turn_is_told_its_own_topic_and_the_others_in_brief_within_the_limit() {
    let own = summary("Fixing the payment page.", 12);
    let others = vec![("Guide".to_string(), summary("Writing the onboarding guide.", 10)), ("Deploy".to_string(), summary("The canary deploy.", 10))];
    let text = summaries::lead_text(Some(&own), &others).unwrap();
    assert!(text.starts_with(HEADER_OWN));
    assert!(text.contains(HEADER_OTHERS) && text.contains("\u{201c}Guide\u{201d}: Writing the onboarding guide."));
    assert_eq!(summaries::lead_text(None, &[]), None);
    let many: Vec<(String, Summary)> = (0..10).map(|n| (format!("T{n}"), summary(&"y".repeat(1_000), 10))).collect();
    assert!(summaries::lead_text(None, &many).unwrap().chars().count() <= summaries::LEAD_CHARS);
}

/// Over the agent harness: a topic is summarised after its tenth turn, by the
/// model this PC runs (the harness's), and the next topic's first turn is told
/// about it.
#[test]
fn a_topic_is_summarised_after_its_tenth_turn_and_the_next_topic_hears_of_it() {
    let h = H::new("summaries-across-topics");
    let ws = h.workspace();
    let mut script: Vec<_> = (0..5).map(|n| say(&format!("Answer {n}."))).collect();
    script.push(say("The user is fixing the payment page's tax."));
    script.push(say("Starting the guide."));
    let model = h.script(script);
    let first = h.agent(None, "Fix the payment page's tax.", &ws);
    h.turns_end(&first, 1);
    for n in 1..5 {
        let accepted = h.send(Some(&first), &format!("Next step {n}."), "local", local(), Mode::Agent, Some(&ws));
        assert!(accepted.is_ok(), "{accepted:?}");
        h.turns_end(&first, n + 1);
    }
    let kept = Summaries::new(&h.state);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while kept.get(&first).is_none() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let written = kept.get(&first).expect("the topic was summarised after its tenth turn");
    assert_eq!(written.text, "The user is fixing the payment page's tax.");
    assert_eq!(written.turns, MIN_TURNS);
    let second = h.agent(None, "Write the onboarding guide.", &ws);
    h.turns_end(&second, 1);
    let calls = model.calls();
    let texts = user_texts(calls.last().unwrap());
    assert!(texts.iter().any(|t| t.contains(HEADER_OTHERS) && t.contains("payment page's tax")), "{texts:?}");
}
