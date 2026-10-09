//! Python's text formats, reproduced exactly where the SDK's output depends on them.
//!
//! Python's text formats, reproduced exactly where the SDK's output depends on them.
//!
//! Ports the observable behaviour of two Python facilities that
//! openai-agents-python 0.22.3 uses when it writes what a person or a model
//! later reads:
//!
//! - `json.dumps(...)` of a string, used by `Handoff.get_transfer_message`
//!   (`{"assistant": "<name>"}`): the default separators put a space after the
//!   colon and `ensure_ascii` escapes everything outside printable ASCII.
//! - `datetime.now(timezone.utc).isoformat()`, used for every span timestamp:
//!   microseconds are written only when they are not zero.
//!
//! Not ported: the text of `json.JSONDecodeError`. The SDK includes it in a tool
//! error only when `OPENAI_AGENTS_DONT_LOG_TOOL_DATA` is switched off; by default
//! it is ON, the model is told `Invalid JSON input for tool <name>` and nothing
//! else, and this port always behaves as the SDK does by default.
//!
//! Invariant: nothing here panics on any input.

use std::fmt::Write as _;

use time::OffsetDateTime;

/// `json.dumps(text)` for one string: quotes included, `ensure_ascii=True`.
pub(crate) fn python_json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(ch),
            _ => {
                let mut units = [0u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
    out
}

/// `json.dumps({"assistant": name})`: what a handoff tool answers with.
pub(crate) fn transfer_message(agent_name: &str) -> String {
    format!("{{\"assistant\": {}}}", python_json_string(agent_name))
}

/// `datetime.isoformat()` of a UTC instant: `2026-09-30T05:21:05.123456+00:00`,
/// with the fraction left out when the microsecond is zero, as Python does.
pub(crate) fn iso_utc(instant: OffsetDateTime) -> String {
    let mut out = String::with_capacity(32);
    let _ = write!(
        out,
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
        instant.year(),
        u8::from(instant.month()),
        instant.day(),
        instant.hour(),
        instant.minute(),
        instant.second()
    );
    let micros = instant.microsecond();
    if micros != 0 {
        let _ = write!(out, ".{micros:06}");
    }
    out.push_str("+00:00");
    out
}

/// The current time in the SDK's timestamp form.
pub(crate) fn iso_now() -> String {
    iso_utc(OffsetDateTime::now_utc())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strings_are_escaped_like_json_dumps_with_ensure_ascii() {
        assert_eq!(python_json_string("Model advisor"), "\"Model advisor\"");
        assert_eq!(
            python_json_string("Mod\u{e8}l adv\u{ef}sor \u{1F600} \u{7f}"),
            "\"Mod\\u00e8l adv\\u00efsor \\ud83d\\ude00 \\u007f\""
        );
        assert_eq!(
            python_json_string("a\"b\\c\n\t\r\u{8}\u{c}\u{1}"),
            r#""a\"b\\c\n\t\r\b\f\u0001""#
        );
    }

    #[test]
    fn the_transfer_message_has_python_separators() {
        assert_eq!(
            transfer_message("Model advisor"),
            r#"{"assistant": "Model advisor"}"#
        );
    }

    #[test]
    fn timestamps_omit_a_zero_fraction_like_isoformat() {
        let base = OffsetDateTime::from_unix_timestamp(1_790_000_000).unwrap();
        assert_eq!(iso_utc(base), "2026-09-21T14:13:20+00:00");
        let with_micros = base.replace_microsecond(123_456).unwrap();
        assert_eq!(iso_utc(with_micros), "2026-09-21T14:13:20.123456+00:00");
        let one_micro = base.replace_microsecond(7).unwrap();
        assert_eq!(iso_utc(one_micro), "2026-09-21T14:13:20.000007+00:00");
    }

    #[test]
    fn the_current_time_has_the_sdk_shape() {
        let now = iso_now();
        assert!(now.ends_with("+00:00"), "{now}");
        assert_eq!(now.as_bytes()[10], b'T');
        assert!(now.len() == 25 || now.len() == 32, "{now}");
    }
}
