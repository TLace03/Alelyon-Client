//! Bounds on the text that enters events and spans.
//!
//! The protocol says every string in an event or span is already bounded by its
//! producer. The producers here are the SDK port and the model, and neither
//! promises a bound (a model may stream megabytes; a tool may return anything),
//! so everything passes through this module on its way into a run's log:
//!
//! - a string is cut to [`MAX_TEXT_CHARS`] characters (20,000), ending in `…`
//!   when it was cut, and never in the middle of a character;
//! - a JSON value (a span's data) is cut string by string the same way, and its
//!   strings together may hold at most [`MAX_VALUE_CHARS`] characters: past that,
//!   the remaining strings become `…`. A span's data is a few messages and a few
//!   numbers, so the budget is never reached by anything but a runaway.
//!
//! Invariant: the result is valid UTF-8 and never longer than the bound, and the
//! structure (keys, array lengths, numbers, booleans) is untouched.

use serde_json::Value;

/// The longest string an event or span field may hold.
pub const MAX_TEXT_CHARS: usize = 20_000;
/// The most characters all the strings of one span's data may hold together.
pub const MAX_VALUE_CHARS: usize = 200_000;

const ELLIPSIS: char = '\u{2026}';

/// `text`, cut to at most `max_chars` characters (ending in `…` when cut).
pub fn cap_text(text: &str, max_chars: usize) -> String {
    match text.char_indices().nth(max_chars) {
        None => text.to_owned(),
        Some((_, _)) if max_chars == 0 => String::new(),
        Some(_) => {
            let keep: String = text.chars().take(max_chars - 1).collect();
            format!("{keep}{ELLIPSIS}")
        }
    }
}

/// Bound every string of `value` in place, spending `budget` characters in all.
pub fn bound_value(value: &mut Value, budget: &mut usize) {
    match value {
        Value::String(text) => {
            let allowed = MAX_TEXT_CHARS.min(*budget);
            let bounded = if allowed == 0 && !text.is_empty() {
                ELLIPSIS.to_string()
            } else {
                cap_text(text, allowed)
            };
            *budget = budget.saturating_sub(bounded.chars().count());
            *text = bounded;
        }
        Value::Array(items) => items.iter_mut().for_each(|item| bound_value(item, budget)),
        Value::Object(map) => map.values_mut().for_each(|item| bound_value(item, budget)),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn short_text_is_untouched_and_long_text_ends_in_an_ellipsis() {
        assert_eq!(cap_text("hello", 5), "hello");
        assert_eq!(cap_text("hello!", 5), "hell\u{2026}");
        assert_eq!(cap_text("", 0), "");
        assert_eq!(cap_text("a", 0), "");
        let long = "x".repeat(MAX_TEXT_CHARS + 500);
        let capped = cap_text(&long, MAX_TEXT_CHARS);
        assert_eq!(capped.chars().count(), MAX_TEXT_CHARS);
        assert!(capped.ends_with('\u{2026}'));
    }

    #[test]
    fn the_cut_never_splits_a_character() {
        let text = "\u{1F600}".repeat(30);
        let capped = cap_text(&text, 10);
        assert_eq!(capped.chars().count(), 10);
        assert!(capped.starts_with("\u{1F600}") && capped.ends_with('\u{2026}'));
        assert_eq!(cap_text("h\u{e9}llo w\u{f6}rld", 8), "h\u{e9}llo w\u{2026}");
    }

    #[test]
    fn a_json_value_is_bounded_string_by_string_and_in_total() {
        let mut value = json!({
            "type": "generation",
            "n": 3,
            "ok": true,
            "input": [{"role": "user", "content": "x".repeat(MAX_TEXT_CHARS + 1)}, {"role": "assistant", "content": "short"}],
            "nothing": null
        });
        let mut budget = MAX_VALUE_CHARS;
        bound_value(&mut value, &mut budget);
        assert_eq!(value["n"], 3);
        assert_eq!(value["ok"], true);
        assert_eq!(value["nothing"], Value::Null);
        assert_eq!(
            value["input"][0]["content"]
                .as_str()
                .unwrap()
                .chars()
                .count(),
            MAX_TEXT_CHARS
        );
        assert_eq!(value["input"][1]["content"], "short");
        assert_eq!(value["input"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn the_total_budget_runs_out() {
        let piece = "y".repeat(MAX_TEXT_CHARS);
        let mut value = Value::Array((0..15).map(|_| Value::String(piece.clone())).collect());
        let mut budget = MAX_VALUE_CHARS;
        bound_value(&mut value, &mut budget);
        let lengths: Vec<usize> = value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().chars().count())
            .collect();
        assert_eq!(lengths[..10], [MAX_TEXT_CHARS; 10]);
        assert!(
            lengths[10..].iter().all(|len| *len == 1),
            "what is past the budget is an ellipsis: {lengths:?}"
        );
        assert_eq!(budget, 0);
        let mut empty = json!([""]);
        let mut none = 0;
        bound_value(&mut empty, &mut none);
        assert_eq!(empty, json!([""]), "an empty string costs nothing");
    }
}
