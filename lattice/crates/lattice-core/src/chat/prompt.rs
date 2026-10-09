//! What a plain chat turn sends (the native chat's spec §2.3, §3.3.6; row C5 of
//! the chat core's spec): a port of the web Lattice's
//! `conversation_messages(question, [], history)`
//! (the Python runtime's `engine.py`) with its default domain,
//! `GENERAL`.
//!
//! - The system message is the persona, a blank line, and the open-mode rules.
//!   Both are copied byte for byte from `domain.py` and `engine.py`, including
//!   the rules' em dashes (`\u{2014}`). Until a later fix repaired its
//!   cp1252 mojibake, `engine.py` held the three characters
//!   `\u{e2}\u{20ac}\u{201d}` where each dash was meant, the web sent those,
//!   and this port copied them; it now sends the dash, as the web does, and
//!   `chat/messages.json` pins it.
//! - Then the last [`HISTORY_TURNS`] turns of the history, each cut to
//!   [`HISTORY_CHARS`] code points, as the reader's (any role but
//!   `"assistant"`) or the model's, a turn whose text is blank after Python's
//!   `strip` left out. The window is taken before blank turns are left out,
//!   as in Python.
//! - Then the question, after "No tool was queried for this question." and a
//!   blank line. The history ends with the question when it was saved first,
//!   so the model reads it twice: the web's known quirk, kept for parity.
//!
//! Nothing here reads, writes or logs anything.

use lattice_agents::model::InputItem;

use super::store::StoredTurn;
use crate::py;

/// How many of the last turns of the history are sent.
pub const HISTORY_TURNS: usize = 8;
/// How many code points of each history turn are sent.
pub const HISTORY_CHARS: usize = 4000;

/// `GENERAL`'s persona (`domain.Vocabulary.persona`).
pub const PERSONA: &str =
    "You are Lattice, Alelyon's assistant. You answer the way a knowledgeable colleague would.";

/// `engine._open_rules(GENERAL.vocabulary)`.
pub const OPEN_RULES: &str = "How to use the tool data:\n- The figures above were computed deterministically by this machine's own tools, from captured data, moments ago. Where they answer the question they are better than your recollection \u{2014} quote them exactly as rendered and keep their as-of stamps.\n- Never attribute a figure to a tool that did not return it. If you give a number of your own \u{2014} an estimate, a worked calculation, something you know \u{2014} say so plainly in the sentence that carries it.\n- If the tools returned nothing relevant, answer anyway from what you know, and say which tool would hold the measured version.\n\nOtherwise answer normally: reason it through, explain, work through the arithmetic, write code, ask a clarifying question if the request is genuinely ambiguous. Be direct and concrete; skip the preamble and the disclaimers about being an AI.";

/// What precedes the question when no tool ran (`f"No {tool_noun} was
/// queried for this question.\n\n"`).
pub const NO_TOOL: &str = "No tool was queried for this question.\n\n";

/// A plain turn's request: the system message and the input items.
#[derive(Clone, Debug, PartialEq)]
pub struct Messages {
    pub system: String,
    pub input: Vec<InputItem>,
}

/// The system message: the persona, a blank line, the rules.
pub fn system() -> String {
    format!("{PERSONA}\n\n{OPEN_RULES}")
}

/// `conversation_messages(question, [], history)`.
pub fn messages(question: &str, history: &[StoredTurn]) -> Messages {
    let window = &history[history.len().saturating_sub(HISTORY_TURNS)..];
    let mut input = Vec::new();
    for turn in window {
        let text: String = turn.text.chars().take(HISTORY_CHARS).collect();
        if py::strip(&text).is_empty() {
            continue;
        }
        input.push(if turn.role == "assistant" {
            InputItem::Assistant {
                text: Some(text),
                tool_calls: Vec::new(),
            }
        } else {
            InputItem::User(text)
        });
    }
    input.push(InputItem::User(format!("{NO_TOOL}{question}")));
    Messages {
        system: system(),
        input,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(role: &str, text: &str) -> StoredTurn {
        StoredTurn {
            id: "t".into(),
            ts: 1.0,
            role: role.into(),
            text: text.into(),
            tools: Vec::new(),
            facts: Vec::new(),
            unsupported: Vec::new(),
            provider: String::new(),
            error: String::new(),
            constrained: false,
            truncated: false,
            cancelled: false,
            prompt_tokens: None,
            completion_tokens: None,
            superseded: false,
        }
    }

    #[test]
    fn the_window_is_taken_before_blank_turns_are_left_out() {
        let mut history: Vec<StoredTurn> =
            (0..10).map(|i| turn("user", &format!("t{i}"))).collect();
        history[9].text = "  \u{3000} ".into();
        history[8].role = "assistant".into();
        let request = messages("q?", &history);
        // Turns 2..=9 are the window; 9 is blank.
        assert_eq!(request.input.len(), 8);
        assert_eq!(request.input[0], InputItem::User("t2".into()));
        assert_eq!(
            request.input[6],
            InputItem::Assistant {
                text: Some("t8".into()),
                tool_calls: Vec::new()
            }
        );
        assert_eq!(
            request.input[7],
            InputItem::User("No tool was queried for this question.\n\nq?".into())
        );
        assert!(request.system.starts_with(PERSONA));
        assert!(
            !request.system.contains("q?"),
            "the question is not in the system message"
        );
    }

    #[test]
    fn a_turn_is_cut_at_4000_code_points() {
        let long = "\u{1f600}".repeat(HISTORY_CHARS + 5);
        let request = messages("q", &[turn("system", &long)]);
        let InputItem::User(text) = &request.input[0] else {
            panic!("{:?}", request.input[0])
        };
        assert_eq!(text.chars().count(), HISTORY_CHARS);
    }
}
