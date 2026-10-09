//! Which figures in an answer no tool backs (the native chat's spec §3.3.6; row
//! C5 of the chat core's spec): a port of the web Lattice's
//! `grounding.check(prose, facts=[], question=q)`
//! (the Python runtime's `grounding.py`), the check a plain chat
//! turn runs with no facts.
//!
//! A mention is what `_NUM_RE` finds: an optional sign (`-`, `+`, `−`) and
//! dollar, then either digits grouped by commas or plain digits, each with an
//! optional fraction, then optional whitespace and a unit or scale (`%`,
//! `bp`/`bps`, `x`, `k`, `m`, `mm`, `b`, `bn`, `tn`, in any case), not inside a
//! word or after a point, and not followed by a digit, a letter, `_` or a
//! point and a digit. The scanner here is written by hand (the workspace has
//! no regular-expression crate) and tries the same alternatives in the same
//! order with the same backtracking, so it finds the same matches.
//!
//! With no facts, a mention is supported only when it is a year (no
//! decimals, 1900–2100, no unit) or when its digits appear verbatim in the
//! question; every other mention is unsupported. The result is the
//! unsupported mentions' texts, in order.
//!
//! Deviation D5: Python's `\d`, `\w` and case folding are Unicode-aware. Here
//! a digit is an ASCII digit, a word character is `char::is_alphanumeric()`
//! or `_`, and letters fold in ASCII only. Text with a digit of another
//! script (or a letter whose case folds into ASCII, such as the Kelvin sign)
//! can be read differently; `chat/grounding.json` lists such cases as D5.

use crate::py;

/// One figure found in the prose.
#[derive(Clone, Debug, PartialEq)]
pub struct Mention {
    /// As written, stripped: `"$12.4bn"`.
    pub text: String,
    /// The number, scaled by a `k`/`m`/`mm`/`b`/`bn`/`tn` suffix.
    pub value: f64,
    /// Digits after the point, as written.
    pub decimals: usize,
    /// Code point offsets of the whole match.
    pub start: usize,
    pub end: usize,
    /// `"%"`, `"bp"`, `"x"` or `""`.
    pub unit: &'static str,
    /// Written with a scale suffix.
    pub scaled: bool,
}

/// The years that are not market figures.
pub const YEAR_LOW: f64 = 1900.0;
pub const YEAR_HIGH: f64 = 2100.0;

/// The suffix alternatives, in the pattern's order: `bps?` is `bps`, then `bp`,
/// and the pattern's own `bp` follows it. (Where two alternatives can both match,
/// as `m` and `mm`, the lookahead that forbids a following letter leaves only one.)
const SUFFIXES: [&str; 11] = ["%", "bps", "bp", "bp", "k", "mm", "m", "bn", "b", "tn", "x"];

fn digit(c: char) -> bool {
    c.is_ascii_digit()
}

fn word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn sign(c: char) -> bool {
    matches!(c, '-' | '+' | '\u{2212}')
}

fn digits_from(chars: &[char], at: usize) -> usize {
    chars[at.min(chars.len())..]
        .iter()
        .take_while(|c| digit(**c))
        .count()
}

/// The ways `(?:\.\d+)?` can end at `at`, in backtracking order.
fn fractions(chars: &[char], at: usize) -> Vec<usize> {
    let mut ends = Vec::new();
    if chars.get(at) == Some(&'.') {
        let run = digits_from(chars, at + 1);
        for take in (1..=run).rev() {
            ends.push(at + 1 + take);
        }
    }
    ends.push(at);
    ends
}

/// `\s*(suffix)?` and the three lookaheads after a number ending at `at`:
/// where the match ends and the suffix, or `None`.
fn tail(chars: &[char], at: usize) -> Option<(usize, String)> {
    let spaces = chars[at..].iter().take_while(|c| py::is_space(**c)).count();
    for taken in (0..=spaces).rev() {
        let here = at + taken;
        let mut options: Vec<(usize, String)> = SUFFIXES
            .iter()
            .filter_map(|suffix| {
                let end = here + suffix.chars().count();
                let written: String = chars.get(here..end)?.iter().collect();
                written
                    .eq_ignore_ascii_case(suffix)
                    .then_some((end, written))
            })
            .collect();
        options.push((here, String::new()));
        for (end, suffix) in options {
            let next = chars.get(end).copied();
            let after = chars.get(end + 1).copied();
            let clear = !next.is_some_and(digit)
                && !next.is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                && !(next == Some('.') && after.is_some_and(digit));
            if clear {
                return Some((end, suffix));
            }
        }
    }
    None
}

/// The number part of each alternative at `at` (past the sign and dollar),
/// in backtracking order: where it ends.
fn numbers(chars: &[char], at: usize) -> Vec<usize> {
    let mut ends = Vec::new();
    let run = digits_from(chars, at);
    // `\d{1,3}(?:,\d{3})+(?:\.\d+)?`
    for lead in (1..=run.min(3)).rev() {
        let mut groups = Vec::new();
        let mut here = at + lead;
        while chars.get(here) == Some(&',') && digits_from(chars, here + 1) >= 3 {
            here += 4;
            groups.push(here);
        }
        for end in groups.iter().rev() {
            ends.extend(fractions(chars, *end));
        }
    }
    // `\d+(?:\.\d+)?`
    for take in (1..=run).rev() {
        ends.extend(fractions(chars, at + take));
    }
    ends
}

/// The match at `start`, if `_NUM_RE` matches there.
fn match_at(chars: &[char], start: usize) -> Option<Mention> {
    if start > 0 && (word(chars[start - 1]) || chars[start - 1] == '.') {
        return None;
    }
    let mut at = start;
    if chars.get(at).copied().is_some_and(sign) {
        at += 1;
    }
    if chars.get(at) == Some(&'$') {
        at += 1;
    }
    for number_end in numbers(chars, at) {
        if let Some((end, suffix)) = tail(chars, number_end) {
            return Some(mention(chars, start, number_end, end, &suffix));
        }
    }
    None
}

fn mention(chars: &[char], start: usize, number_end: usize, end: usize, suffix: &str) -> Mention {
    let raw: String = chars[start..number_end].iter().collect();
    // `_to_float`: commas, the dollar and the minus sign, then a leading `+`.
    let plain = raw.replace([',', '$'], "").replace('\u{2212}', "-");
    let mut value: f64 = plain.trim_start_matches('+').parse().unwrap_or(0.0);
    let decimals = plain
        .split_once('.')
        .map_or(0, |(_, fraction)| fraction.len());
    let (mut unit, mut scaled) = ("", false);
    match suffix.to_ascii_lowercase().as_str() {
        "%" => unit = "%",
        "bp" | "bps" => unit = "bp",
        "x" => unit = "x",
        "k" => (value, scaled) = (value * 1e3, true),
        "m" | "mm" => (value, scaled) = (value * 1e6, true),
        "b" | "bn" => (value, scaled) = (value * 1e9, true),
        "tn" => (value, scaled) = (value * 1e12, true),
        _ => {}
    }
    let whole: String = chars[start..end].iter().collect();
    Mention {
        text: py::strip(&whole).to_owned(),
        value,
        decimals,
        start,
        end,
        unit,
        scaled,
    }
}

/// `find_mentions(prose)`.
pub fn find_mentions(prose: &str) -> Vec<Mention> {
    let chars: Vec<char> = prose.chars().collect();
    let mut found = Vec::new();
    let mut at = 0;
    while at < chars.len() {
        match match_at(&chars, at) {
            Some(mention) => {
                at = mention.end.max(at + 1);
                found.push(mention);
            }
            None => at += 1,
        }
    }
    found
}

/// `_literal_tokens(text)`: the numerals (`\d+(?:\.\d+)?`) written in it.
fn literal_tokens(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut found = Vec::new();
    let mut at = 0;
    while at < chars.len() {
        let run = digits_from(&chars, at);
        if run == 0 {
            at += 1;
            continue;
        }
        let mut end = at + run;
        if chars.get(end) == Some(&'.') {
            let fraction = digits_from(&chars, end + 1);
            if fraction > 0 {
                end += 1 + fraction;
            }
        }
        found.push(chars[at..end].iter().collect());
        at = end;
    }
    found
}

/// `_is_supported` with no facts: a year, or a numeral the question holds.
fn supported(mention: &Mention, literals: &[String]) -> bool {
    if mention.decimals == 0
        && (YEAR_LOW..=YEAR_HIGH).contains(&mention.value)
        && mention.unit.is_empty()
    {
        return true;
    }
    // `men.text.lstrip("+-−$").split()[0].rstrip("%xkmbnt").rstrip(".")`
    let bare = mention
        .text
        .trim_start_matches(['+', '-', '\u{2212}', '$'])
        .split(py::is_space)
        .find(|part| !part.is_empty())
        .unwrap_or("")
        .trim_end_matches(['%', 'x', 'k', 'm', 'b', 'n', 't'])
        .trim_end_matches('.')
        .replace(',', "");
    literals.contains(&bare)
}

/// `[m.text for m in check(prose, [], question=question).unsupported]`.
pub fn unsupported_without_facts(prose: &str, question: &str) -> Vec<String> {
    let literals = literal_tokens(question);
    find_mentions(prose)
        .into_iter()
        .filter(|mention| !supported(mention, &literals))
        .map(|mention| mention.text)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(prose: &str) -> Vec<String> {
        find_mentions(prose).into_iter().map(|m| m.text).collect()
    }

    #[test]
    fn a_prefix_of_a_longer_number_is_not_a_figure() {
        assert_eq!(texts("663.705"), ["663.705"]);
        assert_eq!(texts("closed at 663.70."), ["663.70"]);
        assert_eq!(texts("v1.2.3"), Vec::<String>::new());
        assert_eq!(texts("5kg"), Vec::<String>::new());
        assert_eq!(texts("5 km"), ["5"]);
    }

    #[test]
    fn grouped_digits_fall_back_to_plain_ones() {
        assert_eq!(texts("1,234.5"), ["1,234.5"]);
        assert_eq!(texts("1,23"), ["1", "23"]);
        assert_eq!(texts("1,2345"), ["1", "2345"]);
    }

    #[test]
    fn years_and_echoes_of_the_question_are_supported() {
        assert_eq!(unsupported_without_facts("In 2024 it rose 5%", ""), ["5%"]);
        assert_eq!(
            unsupported_without_facts("over 30 days", "the last 30 days"),
            Vec::<String>::new()
        );
        assert_eq!(
            unsupported_without_facts("2k", ""),
            Vec::<String>::new(),
            "2,000 reads as a year"
        );
        let scaled = &find_mentions("$12.4bn")[0];
        assert!(scaled.scaled && (scaled.value - 12.4e9).abs() < 1.0 && scaled.decimals == 1);
    }
}
