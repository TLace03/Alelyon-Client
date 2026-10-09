//! Text width estimates, so canvases and one-line rows can fit text without a
//! shaping pass.
//!
//! iced 0.14's `text` widget has no ellipsis, and a canvas cannot measure text
//! before it draws it. Rows therefore truncate with an estimate of the advance
//! width per character (in ems, tuned on Segoe UI) and clip as a second line of
//! defence. The estimate errs on the wide side, so a truncated title never
//! overflows its column; the price is a few pixels of slack.
//!
//! Invariants: `truncate_to_width` never returns text wider (by this estimate)
//! than asked, always cuts on a character boundary, and returns its input
//! untouched when it fits.

use std::borrow::Cow;

/// Estimated advance of `ch` in ems.
pub fn advance_em(ch: char) -> f32 {
    match ch {
        ' ' => 0.29,
        'i' | 'l' | 'j' | '.' | ',' | ':' | ';' | '\'' | '|' | '!' => 0.30,
        'f' | 't' | 'r' | 'I' | '(' | ')' | '[' | ']' | '-' | '/' | '\\' => 0.40,
        'm' | 'w' => 0.82,
        'M' | 'W' => 0.92,
        '0'..='9' => 0.56,
        'A'..='Z' => 0.66,
        'a'..='z' => 0.54,
        '\u{2E80}'..='\u{9FFF}' | '\u{AC00}'..='\u{D7AF}' | '\u{FF00}'..='\u{FFEF}' => 1.0,
        c if c.is_ascii() => 0.58,
        _ => 0.70,
    }
}

/// Estimated width in pixels of `text` at `size` pixels.
pub fn text_width(text: &str, size: f32) -> f32 {
    text.chars().map(advance_em).sum::<f32>() * size
}

/// A fixed-pitch estimate for monospace text (Cascadia Code, Consolas).
pub fn mono_width(text: &str, size: f32) -> f32 {
    text.chars().count() as f32 * size * 0.60
}

/// The cut of `text` that fits `max_width`, or `None` when the whole text fits.
fn cut(text: &str, size: f32, max_width: f32) -> Option<String> {
    if text_width(text, size) <= max_width {
        return None;
    }
    let ellipsis = advance_em('…').max(0.70) * size;
    let budget = (max_width - ellipsis).max(0.0);
    let mut used = 0.0;
    let mut end = 0;
    for (i, ch) in text.char_indices() {
        let w = advance_em(ch) * size;
        if used + w > budget {
            break;
        }
        used += w;
        end = i + ch.len_utf8();
    }
    let mut out = text[..end].trim_end().to_string();
    out.push('…');
    Some(out)
}

/// `text` if it fits in `max_width` pixels, else the longest prefix that fits
/// together with a trailing `…`. Newlines and tabs count as one space, since a
/// one-line row cannot show them.
pub fn truncate_to_width(text: &str, size: f32, max_width: f32) -> Cow<'_, str> {
    const WHITESPACE: [char; 3] = ['\n', '\r', '\t'];
    if text.contains(WHITESPACE) {
        let flat = text.replace(WHITESPACE, " ");
        return Cow::Owned(cut(&flat, size, max_width).unwrap_or(flat));
    }
    match cut(text, size, max_width) {
        Some(out) => Cow::Owned(out),
        None => Cow::Borrowed(text),
    }
}

/// The first of `candidates` that `installed` accepts.
///
/// cosmic-text falls back to the default family when one is missing, not down
/// a list, so the list is resolved once at start against the system's fonts.
pub fn pick_family(
    candidates: &[&'static str],
    installed: impl Fn(&str) -> bool,
) -> Option<&'static str> {
    candidates.iter().copied().find(|name| installed(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_that_fits_is_returned_untouched() {
        let out = truncate_to_width("List the models", 13.0, 400.0);
        assert!(matches!(out, Cow::Borrowed("List the models")));
    }

    #[test]
    fn truncation_fits_the_budget_and_ends_with_an_ellipsis() {
        let text = "Summarise the quarterly report and list every open action item for the team";
        for width in [40.0f32, 90.0, 150.0, 230.0] {
            let out = truncate_to_width(text, 13.0, width);
            assert!(out.ends_with('…'), "{out}");
            assert!(text_width(&out, 13.0) <= width + 1e-3, "{width}: {out}");
            assert!(text.starts_with(out.trim_end_matches('…')));
        }
    }

    #[test]
    fn truncation_never_splits_a_character() {
        let text = "日本語のタスクを実行してください、ありがとう";
        let out = truncate_to_width(text, 13.0, 80.0);
        assert!(out.ends_with('…'));
        assert!(text_width(&out, 13.0) <= 80.0 + 1e-3);
        let tiny = truncate_to_width(text, 13.0, 3.0);
        assert_eq!(tiny, "…");
    }

    #[test]
    fn newlines_become_spaces_in_a_one_line_row() {
        assert_eq!(
            truncate_to_width("one\ntwo\tthree", 13.0, 500.0),
            "one two three"
        );
    }

    #[test]
    fn family_resolution_takes_the_first_installed_candidate() {
        let installed = |name: &str| name == "Segoe UI";
        assert_eq!(
            pick_family(&["Segoe UI Variable Text", "Segoe UI"], installed),
            Some("Segoe UI")
        );
        assert_eq!(pick_family(&["Cascadia Code", "Consolas"], installed), None);
        assert_eq!(
            pick_family(&["Cascadia Code", "Consolas"], |n| n == "Cascadia Code"
                || n == "Consolas"),
            Some("Cascadia Code")
        );
    }

    #[test]
    fn wide_letters_measure_wider_than_narrow_ones() {
        assert!(text_width("mmmm", 12.0) > text_width("iiii", 12.0) * 2.0);
        assert!(mono_width("12345", 12.0) > 0.0);
    }
}
