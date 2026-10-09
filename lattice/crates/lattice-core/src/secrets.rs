//! Does a task look like it contains a credential?
//!
//! The `secrets_stay_local` guardrail refuses a task that would send something
//! that looks like a secret to a model that is not on this machine. This module
//! is its detector: the nine patterns below, written by hand rather than as
//! regular expressions (a regex engine would be a new dependency for a handful of
//! simple shapes). Each is the pattern named in its comment. The start of a key
//! needs nothing alphanumeric (a letter or digit, in the Unicode sense) in front
//! of it, so `ask-me-anything` is not a key, but a key stays a key behind a
//! separator: after `_` (`api_key_sk-...`), `-`, `=`, `:`, a quote or a space.
//!
//! ```text
//! sk-[A-Za-z0-9_-]{16,}                    API keys of the `sk-` family
//! gh[pousr]_[A-Za-z0-9]{20,}               GitHub tokens
//! github_pat_[A-Za-z0-9_]{22,}             GitHub fine-grained tokens
//! AKIA[0-9A-Z]{16}\b                       AWS access key ids
//! -----BEGIN [A-Z ]*PRIVATE KEY-----       private key blocks
//! xox[baprs]-[A-Za-z0-9-]{10,}             Slack tokens
//! (sk_live_|sk_test_|rk_live_)[A-Za-z0-9]{16,}    Stripe keys
//! AIza[0-9A-Za-z_-]{35}                    Google API keys
//! hf_[A-Za-z0-9]{30,}                      Hugging Face tokens
//! ```
//!
//! This is a tripwire for the obvious, not a secret scanner: a secret in another
//! shape passes (an AWS secret access key has none of its own), and a lookalike
//! (an example key in a tutorial) trips it. The trip is cheap (the person picks a
//! local model or removes the text) and a missed secret is not, so the patterns
//! lean toward matching.
//!
//! [`looks_like_secret`] answers yes or no. [`redact`] gives the text back with
//! each match replaced by [`REDACTED`], for a record that must not keep the
//! secret: a match runs to the end of the run of characters the key is made of,
//! and a private key block to its `END` line (or to the end of the text when the
//! block was cut off), so none of a key's body is left behind.
//!
//! Invariant: the detector's answer is only yes or no. The matched text is never
//! returned, logged or put in a message, so a refusal cannot repeat the secret.

/// What [`redact`] puts where a secret was.
pub const REDACTED: &str = "[redacted: looks like a secret]";

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The number of characters from `at` that satisfy `class`, counted up to
/// `limit`.
fn run(chars: &[char], at: usize, limit: usize, class: impl Fn(char) -> bool) -> usize {
    chars[at.min(chars.len())..]
        .iter()
        .take(limit)
        .take_while(|c| class(**c))
        .count()
}

fn starts_with(chars: &[char], at: usize, literal: &str) -> bool {
    literal
        .chars()
        .enumerate()
        .all(|(offset, expected)| chars.get(at + offset) == Some(&expected))
}

/// A key may start at `at`: no letter or digit comes just before it. (`\b` would
/// also refuse `_`, which is what hid `api_key_sk-...`.)
fn boundary_before(chars: &[char], at: usize) -> bool {
    at == 0 || !chars[at - 1].is_alphanumeric()
}

/// A token that is `prefix`, then at least `least` characters of `class`, with a
/// key boundary before it: its end is the end of the whole run of `class`. A
/// candidate that fails the minimum never scanned past it (a run shorter than
/// `least` ends the scan), and one that matches is skipped over, so a long input
/// cannot make the scan quadratic.
fn prefixed_run(
    chars: &[char],
    at: usize,
    prefix: &str,
    least: usize,
    class: impl Fn(char) -> bool,
) -> Option<usize> {
    if !(boundary_before(chars, at) && starts_with(chars, at, prefix)) {
        return None;
    }
    let body = at + prefix.chars().count();
    let len = run(chars, body, usize::MAX, class);
    (len >= least).then_some(body + len)
}

fn sk_key(chars: &[char], at: usize) -> Option<usize> {
    prefixed_run(chars, at, "sk-", 16, |c| {
        c.is_ascii_alphanumeric() || c == '_' || c == '-'
    })
}

fn github_token(chars: &[char], at: usize) -> Option<usize> {
    let kind = *chars.get(at + 2)?;
    if !(starts_with(chars, at, "gh") && matches!(kind, 'p' | 'o' | 'u' | 's' | 'r')) {
        return None;
    }
    prefixed_run(chars, at, &format!("gh{kind}_"), 20, |c| {
        c.is_ascii_alphanumeric()
    })
}

fn github_fine_grained(chars: &[char], at: usize) -> Option<usize> {
    prefixed_run(chars, at, "github_pat_", 22, |c| {
        c.is_ascii_alphanumeric() || c == '_'
    })
}

fn aws_key(chars: &[char], at: usize) -> Option<usize> {
    if !(boundary_before(chars, at) && starts_with(chars, at, "AKIA")) {
        return None;
    }
    let body = at + 4;
    let run_len = run(chars, body, 16, |c| {
        c.is_ascii_digit() || c.is_ascii_uppercase()
    });
    // `{16}` takes exactly sixteen; the `\b` after them needs a non-word next.
    (run_len == 16 && chars.get(body + 16).is_none_or(|next| !is_word(*next))).then_some(body + 16)
}

/// The end of a `[A-Z ]*PRIVATE KEY-----` tail that begins at `from`: `[A-Z ]*` is
/// greedy and `PRIVATE KEY` is itself made of `[A-Z ]`, so the pattern matches
/// exactly when the whole run of `[A-Z ]` ends in `PRIVATE KEY` and the dashes
/// follow it.
fn key_label_end(chars: &[char], from: usize) -> Option<usize> {
    let len = run(chars, from, usize::MAX, |c| {
        c.is_ascii_uppercase() || c == ' '
    });
    let end = from + len;
    let tail: String = chars[from..end].iter().collect();
    (tail.ends_with("PRIVATE KEY") && starts_with(chars, end, "-----")).then_some(end + 5)
}

fn private_key(chars: &[char], at: usize) -> Option<usize> {
    const BEGIN: &str = "-----BEGIN ";
    const END: &str = "-----END ";
    if !starts_with(chars, at, BEGIN) {
        return None;
    }
    let header_end = key_label_end(chars, at + BEGIN.len())?;
    // The key's body follows the header: take it up to the END line, or to the
    // end of the text when the block was cut off.
    let mut from = header_end;
    while from < chars.len() {
        if starts_with(chars, from, END)
            && let Some(end) = key_label_end(chars, from + END.len())
        {
            return Some(end);
        }
        from += 1;
    }
    Some(chars.len())
}

fn slack_token(chars: &[char], at: usize) -> Option<usize> {
    let kind = *chars.get(at + 3)?;
    if !(starts_with(chars, at, "xox") && matches!(kind, 'b' | 'a' | 'p' | 'r' | 's')) {
        return None;
    }
    prefixed_run(chars, at, &format!("xox{kind}-"), 10, |c| {
        c.is_ascii_alphanumeric() || c == '-'
    })
}

fn stripe_key(chars: &[char], at: usize) -> Option<usize> {
    ["sk_live_", "sk_test_", "rk_live_"]
        .iter()
        .find_map(|prefix| prefixed_run(chars, at, prefix, 16, |c| c.is_ascii_alphanumeric()))
}

fn google_key(chars: &[char], at: usize) -> Option<usize> {
    prefixed_run(chars, at, "AIza", 35, |c| {
        c.is_ascii_alphanumeric() || c == '_' || c == '-'
    })
}

fn hugging_face_token(chars: &[char], at: usize) -> Option<usize> {
    prefixed_run(chars, at, "hf_", 30, |c| c.is_ascii_alphanumeric())
}

/// Where the longest secret-shaped match that starts at `at` ends, if one does.
fn match_end(chars: &[char], at: usize) -> Option<usize> {
    [
        sk_key(chars, at),
        github_token(chars, at),
        github_fine_grained(chars, at),
        aws_key(chars, at),
        private_key(chars, at),
        slack_token(chars, at),
        stripe_key(chars, at),
        google_key(chars, at),
        hugging_face_token(chars, at),
    ]
    .into_iter()
    .flatten()
    .max()
}

/// True when `text` contains something shaped like one of the credentials above.
pub fn looks_like_secret(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    (0..chars.len()).any(|at| match_end(&chars, at).is_some())
}

/// The secret-shaped matches in `chars`, as `[start, end)` character
/// ranges, found as [`redact`] finds them: the longest match at the earliest
/// place, then on after it.
pub fn secret_spans(chars: &[char]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut at = 0;
    while at < chars.len() {
        match match_end(chars, at) {
            Some(end) if end > at => {
                spans.push((at, end));
                at = end;
            }
            _ => at += 1,
        }
    }
    spans
}

/// `text` with each secret-shaped match replaced by [`REDACTED`]; the text
/// between the matches is untouched.
pub fn redact(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    for (start, end) in secret_spans(&chars) {
        out.extend(&chars[at..start]);
        out.push_str(REDACTED);
        at = end;
    }
    out.extend(&chars[at..]);
    out
}

/// Replace each secret-looking part of every string in `value`, in place (a
/// walk over arrays and objects; keys are left alone). True when anything was
/// replaced. The run manager redacts a refused run's events with it, and the
/// chat core every record it writes from a turn (T15).
pub fn redact_strings(value: &mut serde_json::Value) -> bool {
    use serde_json::Value;
    match value {
        Value::String(text) => {
            let clean = redact(text);
            let changed = clean != *text;
            *text = clean;
            changed
        }
        Value::Array(items) => items
            .iter_mut()
            .fold(false, |any, item| redact_strings(item) | any),
        Value::Object(map) => map
            .values_mut()
            .fold(false, |any, item| redact_strings(item) | any),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The sample credentials below are assembled from pieces, so that no source
    // file holds a string that a secret scanner would report as a real one.
    const SK: &str = concat!("s", "k-");
    const GHP: &str = concat!("g", "hp_");
    const GHO: &str = concat!("g", "ho_");
    const GHU: &str = concat!("g", "hu_");
    const GHS: &str = concat!("g", "hs_");
    const GHR: &str = concat!("g", "hr_");
    const AKIA: &str = concat!("AK", "IA");
    const XOXB: &str = concat!("x", "oxb-");
    const XOXP: &str = concat!("x", "oxp-");
    const XOXA: &str = concat!("x", "oxa-");
    const XOXR: &str = concat!("x", "oxr-");
    const XOXS: &str = concat!("x", "oxs-");
    const BEGIN: &str = concat!("-----", "BEGIN ");
    const END: &str = concat!("PRIVATE ", "KEY-----");
    const STRIPE_SK_LIVE: &str = concat!("s", "k_live_");
    const STRIPE_SK_TEST: &str = concat!("s", "k_test_");
    const STRIPE_RK_LIVE: &str = concat!("r", "k_live_");
    const GITHUB_PAT: &str = concat!("github", "_pat_");
    const GOOGLE: &str = concat!("AI", "za");
    const HUGGING_FACE: &str = concat!("h", "f_");

    /// One sample of every shape, and of each way a key may be introduced.
    fn caught() -> Vec<String> {
        vec![
            format!("my key is {SK}abcdefghijklmnop please"),
            format!("{SK}proj-AAAAAAAAAAAAAAAA_BBBB-CCCC"),
            format!("token {GHP}abcdefghijklmnopqrst12"),
            format!("{GHO}ABCDEFGHIJKLMNOPQRSTUVWX"),
            format!("{GHU}12345678901234567890"),
            format!("{GHS}12345678901234567890"),
            format!("{GHR}12345678901234567890"),
            format!("id {AKIA}ABCDEFGHIJKLMNOP."),
            format!("{AKIA}0123456789ABCDEF"),
            format!("{BEGIN}{END}\nMIIE"),
            format!("{BEGIN}RSA {END}"),
            format!("{BEGIN}ENCRYPTED {END}"),
            format!("{BEGIN}OPENSSH {END}"),
            format!("{XOXB}1234567890-abcdefghij"),
            format!("{XOXP}1234567890"),
            format!("{XOXA}0123456789"),
            format!("{XOXR}0123456789"),
            format!("{XOXS}0123456789"),
            format!("({SK}abcdefghijklmnop)"),
            format!("line one\n{SK}abcdefghijklmnop\nline three"),
            format!("caf\u{e9} {SK}abcdefghijklmnop"),
            // Stripe.
            format!("{STRIPE_SK_LIVE}abcdefghijklmnop"),
            format!("{STRIPE_SK_TEST}ABCDEFGHIJKLMNOPQRSTUVWX"),
            format!("{STRIPE_RK_LIVE}0123456789abcdef0123"),
            format!("key={STRIPE_SK_LIVE}abcdefghijklmnop"),
            // GitHub fine-grained tokens.
            format!("{GITHUB_PAT}abcdefghijklmnopqrstuv"),
            format!("{GITHUB_PAT}11ABCDEFG0_abcdefghijklmnopqrstuvwxyz0123456789"),
            // Google API keys.
            format!("{GOOGLE}SyA1234567890abcdefghijklmnopqrstuvw"),
            format!("{GOOGLE}abcdefghijklmnopqrstuvwxyz0123456-_"),
            // Hugging Face tokens.
            format!("{HUGGING_FACE}abcdefghijklmnopqrstuvwxyzABCD"),
            format!("token: {HUGGING_FACE}abcdefghijklmnopqrstuvwxyzABCDEFGH"),
            // A key is still a key when a name, a separator or an assignment comes first.
            format!("_{SK}abcdefghijklmnop"),
            format!("api_key_{SK}abcdefghijklmnop"),
            format!("OPENAI_KEY={SK}abcdefghijklmnop"),
            format!("KEY:{SK}abcdefghijklmnop"),
            format!("key-{SK}abcdefghijklmnop"),
            format!("my_token_{GHP}abcdefghijklmnopqrst12"),
            format!("AWS_KEY_{AKIA}ABCDEFGHIJKLMNOP"),
            format!("x_{XOXB}1234567890-abcdefghij"),
            format!("hf_token_{HUGGING_FACE}abcdefghijklmnopqrstuvwxyzABCD"),
            format!("_{STRIPE_SK_LIVE}abcdefghijklmnop"),
        ]
    }

    #[test]
    fn each_credential_shape_is_caught() {
        for text in caught() {
            assert!(looks_like_secret(&text), "{text}");
        }
    }

    #[test]
    fn redaction_replaces_each_match_and_leaves_the_rest() {
        let key = format!("{SK}abcdefghijklmnopqrstuvwx");
        assert_eq!(
            redact(&format!("use {key} now")),
            format!("use {REDACTED} now")
        );
        assert_eq!(
            redact(&format!(
                "{key}\n{key},{GHP}abcdefghijklmnopqrst12;{AKIA}ABCDEFGHIJKLMNOP"
            )),
            format!("{REDACTED}\n{REDACTED},{REDACTED};{REDACTED}"),
            "each match, with what lies between them kept"
        );
        assert_eq!(
            redact(&format!("OPENAI_KEY={key}")),
            format!("OPENAI_KEY={REDACTED}"),
            "the name before a key stays, the key goes"
        );
        // Nothing to redact: the same text back, character for character.
        for text in [
            "",
            "What is 6 * 7?",
            "caf\u{e9} \u{1F600} ask-me-anything: sk is short",
        ] {
            assert_eq!(redact(text), text);
        }
        // A private key block goes through its END line, and no further.
        let block =
            format!("{BEGIN}RSA {END}\nMIIEvAIBADANBgkqhkiG9w0B\nZm9vYmFy\n-----END RSA {END}");
        assert_eq!(
            redact(&format!("before\n{block}\nafter")),
            format!("before\n{REDACTED}\nafter")
        );
        // A block that was cut off goes to the end of the text.
        assert_eq!(
            redact(&format!(
                "x {BEGIN}{END}\nMIIEvAIBADANBgkqhkiG9w0B\nZm9vYmFy"
            )),
            format!("x {REDACTED}")
        );
    }

    #[test]
    fn nothing_that_was_caught_survives_redaction() {
        for text in caught() {
            let clean = redact(&text);
            assert_ne!(clean, text, "{text}");
            assert!(!looks_like_secret(&clean), "{text} -> {clean}");
            assert!(clean.contains(REDACTED));
            for body in ["abcdefghijklmnop", "0123456789abcdef", "ABCDEFGHIJKLMNOP"] {
                assert!(!clean.contains(body), "{text} -> {clean}");
            }
        }
    }

    #[test]
    fn ordinary_text_and_near_misses_pass() {
        for text in [
            String::new(),
            "What is 6 * 7?".to_owned(),
            "Which model should I use to translate a contract?".to_owned(),
            format!("{SK}short"),
            format!("{SK}abcdefghijklmn"),
            format!("ta{SK}abcdefghijklmnopqrstuvwx"),
            format!("a{SK}abcdefghijklmnopqrstuv"),
            "ghx_abcdefghijklmnopqrstuvwxyz".to_owned(),
            format!("{GHP}short"),
            "ghp-abcdefghijklmnopqrstuvwxyz".to_owned(),
            format!("{AKIA}ABCDEFGHIJKLMNO"),
            format!("{AKIA}ABCDEFGHIJKLMNOPQ"),
            format!("{AKIA}abcdefghijklmnop"),
            format!("x{AKIA}ABCDEFGHIJKLMNOP"),
            format!("{BEGIN}CERTIFICATE-----"),
            format!("{BEGIN}PUBLIC KEY-----"),
            format!("{BEGIN}PRIVATE KEYS-----"),
            format!("{BEGIN}private key-----"),
            format!("{BEGIN}PRIVATE KEY----"),
            "xoxz-0123456789".to_owned(),
            format!("{XOXB}123"),
            "xoxb_1234567890".to_owned(),
            format!("\u{e9}{SK}abcdefghijklmnop"),
            format!("1{SK}abcdefghijklmnop"),
            // The new shapes, short or not shaped that way.
            format!("{STRIPE_SK_LIVE}short"),
            format!("{STRIPE_SK_LIVE}abcdefghijklmn"),
            "sk_prod_abcdefghijklmnopqrstuvwx".to_owned(),
            format!("a{STRIPE_SK_LIVE}abcdefghijklmnopqrst"),
            format!("{GITHUB_PAT}short"),
            format!("{GITHUB_PAT}abcdefghijklmnopqrstu"),
            format!("{GOOGLE}short"),
            format!("{GOOGLE}abcdefghijklmnopqrstuvwxyz012345"),
            format!("x{GOOGLE}SyA1234567890abcdefghijklmnopqrstuvw"),
            format!("{HUGGING_FACE}short"),
            format!("{HUGGING_FACE}abcdefghijklmnopqrstuvwxyzABC"),
            format!("x{HUGGING_FACE}abcdefghijklmnopqrstuvwxyzABCD"),
            // Ordinary words, short hex and UUIDs.
            "The quick brown fox jumps over the lazy dog".to_owned(),
            "a1b2c3d4e5f60718".to_owned(),
            "deadbeef".to_owned(),
            "9f8b1c2d-4e5f-4a6b-8c7d-0e1f2a3b4c5d".to_owned(),
            "550e8400-e29b-41d4-a716-446655440000".to_owned(),
            "hf is short for half; github is a site; AI is not a key".to_owned(),
            "task-specific and ask-me-anything: sk is an abbreviation".to_owned(),
            "0123456789abcdef0123456789abcdef".to_owned(),
        ] {
            assert!(!looks_like_secret(&text), "{text}");
        }
    }

    #[test]
    fn a_word_boundary_is_unicode_aware_and_an_aws_key_needs_one_after_it() {
        assert!(
            !looks_like_secret(&format!("{AKIA}ABCDEFGHIJKLMNOP\u{e9}")),
            "a letter after the id is not a boundary"
        );
        assert!(
            looks_like_secret(&format!("{AKIA}ABCDEFGHIJKLMNOP-tail")),
            "a dash is one"
        );
        assert!(looks_like_secret(&format!("={SK}abcdefghijklmnop")));
        assert!(
            looks_like_secret(&format!("_{SK}abcdefghijklmnop")),
            "an underscore is a separator before a key, not a part of a word"
        );
    }

    #[test]
    fn long_inputs_are_scanned_without_panicking() {
        let filler = SK.repeat(6_000);
        assert!(looks_like_secret(&filler));
        let noise = "\u{1F600}x".repeat(10_000);
        assert!(!looks_like_secret(&noise));
        // A long run that never completes a pattern is scanned once, not once per start.
        let almost = format!("{AKIA}{}", "A".repeat(20_000));
        assert!(!looks_like_secret(&almost));
        let dashes = format!("{BEGIN}{}", "A".repeat(20_000));
        assert!(!looks_like_secret(&dashes));
    }
}
