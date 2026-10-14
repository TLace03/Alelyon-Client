//! Recall (`super::recall`): the index over the store's chats, kept up to
//! date by reading only what changed; matches ranked by their words and
//! needing two of the query's; the turn's own topic left out; what a turn's
//! leading item gets; the tool; and, over the agent harness, a new topic's
//! first turn carrying what an earlier topic said.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::json;

use super::agent_tests::{H, say, user_texts};
use super::recall::{self, HEADER, Recall};
use crate::chat::memory::MemoryTranscriptStore;
use crate::chat::transcript::{NewTurn, TranscriptStore};

fn store() -> MemoryTranscriptStore {
    let next = Arc::new(AtomicU64::new(1));
    let ticks = Arc::new(AtomicU64::new(0));
    MemoryTranscriptStore::new(
        Arc::new(move || format!("t{}", next.fetch_add(1, Ordering::Relaxed))),
        Arc::new(move || 1_790_000_000.0 + ticks.fetch_add(1, Ordering::Relaxed) as f64),
    )
}

fn chat(store: &MemoryTranscriptStore, user: &str, answer: &str) -> String {
    let (row, _) = store.first_message(user, "local").unwrap();
    store.append(&row.id, NewTurn::assistant(answer, "local")).unwrap();
    row.id
}

#[test]
fn the_best_match_comes_first_and_one_common_word_brings_nothing() {
    let s = store();
    let pay = chat(&s, "The payment page totals are off by the tax.", "The tax is added twice, in cart.total() and in the page.");
    let _docs = chat(&s, "Write the README for the payment service.", "Done: the README names the service's routes.");
    let mut r = Recall::default();
    assert_eq!(r.refresh(&s), 2, "both chats read once");
    let hits = r.search("why is the tax added twice on the payment page", None, 5);
    assert_eq!(hits[0].conversation, pay);
    assert!(hits[0].snippet.contains("tax"), "{:?}", hits[0]);
    assert!(r.search("payment", None, 5).len() >= 2, "a one-word query needs that word only");
    assert!(r.search("payment kubernetes", None, 5).is_empty(), "two words asked, one matched: nothing");
    assert!(r.search("the of and", None, 5).is_empty(), "no words left");
}

#[test]
fn only_changed_chats_are_read_again_and_the_own_topic_is_left_out() {
    let s = store();
    let a = chat(&s, "Rotate the webhook secret every ninety days.", "Noted: ninety days.");
    let mut r = Recall::default();
    r.refresh(&s);
    assert_eq!(r.refresh(&s), 0, "nothing changed, nothing read");
    s.append(&a, NewTurn::user("And log every webhook rotation.")).unwrap();
    assert_eq!(r.refresh(&s), 1, "the changed chat only");
    assert!(!r.search("webhook rotation logged", None, 5).is_empty());
    assert!(r.search("webhook rotation logged", Some(&a), 5).is_empty(), "the turn's own topic is not recalled");
}

#[test]
fn a_turns_lead_holds_at_most_three_matches_under_its_header_and_the_tool_answers() {
    let s = store();
    for n in 0..5 {
        chat(&s, &format!("Deploy the gateway to region {n} with the canary flag."), "Deployed with the canary flag.");
    }
    let mut r = Recall::default();
    r.refresh(&s);
    let hits = r.search("deploy gateway canary", None, 5);
    let lead = recall::lead_text(&hits).unwrap();
    assert!(lead.starts_with(HEADER));
    assert_eq!(lead.lines().count(), 1 + recall::LEAD_HITS);
    assert!(lead.chars().count() <= recall::LEAD_CHARS);
    assert_eq!(recall::lead_text(&[]), None);
    let text = recall::tool_text(&mut r, &s, &json!({"query": "gateway canary", "limit": 2})).unwrap();
    assert_eq!(text.lines().count(), 3, "{text}");
    assert!(recall::tool_text(&mut r, &s, &json!({"query": "  "})).is_err());
    let none = recall::tool_text(&mut r, &s, &json!({"query": "kubernetes helm"})).unwrap();
    assert!(none.starts_with("Nothing"), "{none}");
}

/// Over the agent harness: what one topic said reaches the first turn of
/// another, in its leading item, and the turn's own words are not recalled
/// back to it.
#[test]
fn a_new_topics_first_turn_carries_what_an_earlier_topic_said() {
    let h = H::new("recall-across-topics");
    let ws = h.workspace();
    let model = h.script(vec![say("The webhook secret rotates every ninety days."), say("It rotates every ninety days, as decided earlier.")]);
    let first = h.agent(None, "Decide how often the stripe webhook secret rotates.", &ws);
    h.turns_end(&first, 1);
    let second = h.agent(None, "How often does the stripe webhook secret rotate?", &ws);
    h.turns_end(&second, 1);
    let calls = model.calls();
    let texts = user_texts(&calls[1]);
    let lead = texts.iter().find(|t| t.contains(HEADER)).unwrap_or_else(|| panic!("no recall in {texts:?}"));
    assert!(lead.contains("ninety days"), "{lead}");
    assert!(!user_texts(&calls[0]).iter().any(|t| t.contains(HEADER)), "nothing earlier to recall in the first topic");
}

/// By hand (`--ignored --nocapture`): the index at the store's full size, 60
/// chats of 400 turns, its first build, a search, and a refresh with nothing
/// changed.
#[test]
#[ignore]
fn timing_at_the_stores_full_size() {
    let s = store();
    for c in 0..60 {
        let (row, _) = s.first_message(&format!("Topic {c}: the gateway canary deploy for region {c}"), "local").unwrap();
        for t in 0..399 {
            let text = format!("Turn {t} of topic {c}: we changed the payment webhook retry policy and the canary threshold to {t} percent, then reviewed the logs.");
            s.append(&row.id, if t % 2 == 0 { NewTurn::assistant(text, "local") } else { NewTurn::user(text) }).unwrap();
        }
    }
    let mut r = Recall::default();
    let started = std::time::Instant::now();
    r.refresh(&s);
    let read = started.elapsed();
    let started = std::time::Instant::now();
    let hits = r.search("webhook retry canary threshold", None, 3);
    let first = started.elapsed();
    let started = std::time::Instant::now();
    r.refresh(&s);
    let _ = r.search("payment logs reviewed", None, 3);
    let again = started.elapsed();
    println!("read 24,000 turns: {read:?}; first search (builds postings): {first:?}; refresh with nothing changed + search: {again:?}; hits {}", hits.len());
}

#[test]
fn a_message_continues_its_topic_returns_to_an_earlier_one_or_starts_a_new_one() {
    use super::recall::{NEW_AFTER, Route};
    let s = store();
    let pay = chat(&s, "The payment page adds the tax twice to the totals.", "The tax is added in cart.total() and again in the page.");
    let docs = chat(&s, "Write the onboarding guide for new engineers.", "The guide covers setup, the build and the review steps.");
    let mut r = Recall::default();
    r.refresh(&s);
    // Continuing the guide: it stays.
    assert_eq!(r.route("add the review steps to the onboarding guide", Some(&docs)), Route::Continue);
    // Back to the payment page's tax: it returns there.
    assert_eq!(r.route("the payment page tax totals are still added twice", Some(&docs)), Route::Return(pay.clone()));
    // Nothing shown yet: the matching topic, else a new one.
    assert_eq!(r.route("payment page tax totals added twice", None), Route::Return(pay.clone()));
    assert_eq!(r.route("kubernetes helm chart upgrade", None), Route::New);
    // A short topic keeps an unrelated message; a long one sharing none of its words gives way to a new topic.
    assert_eq!(r.route("kubernetes helm chart upgrade", Some(&docs)), Route::Continue);
    for n in 0..NEW_AFTER {
        s.append(&docs, NewTurn::user(format!("guide section {n} about setup"))).unwrap();
    }
    r.refresh(&s);
    assert_eq!(r.route("kubernetes helm chart upgrade", Some(&docs)), Route::New);
}
