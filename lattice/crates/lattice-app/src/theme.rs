//! Lattice's black and gold: the palette of `lattice_desktop/src/styles/tokens.css`
//! as iced colours, the span-kind colours, fonts, and every widget style.
//!
//! Invariants:
//! - The palette constants are the tokens' values, one for one; a test reads the
//!   token file when it is present and fails if they drift apart.
//! - Every colour used for text or for a span kind holds a contrast of at least
//!   4.5:1 (WCAG 2.x relative luminance) against both `--surface` and
//!   `--canvas`, and the text on a gold button holds 4.5:1 against the gold.
//!   A test asserts it for the whole list, so a new colour must join the list.
//! - Widget styles read these constants, never the iced `Theme`'s generated
//!   palette, so a change of theme cannot recolour the interface by accident.
//!   The `Theme::custom` below exists so iced's own defaults (markdown text,
//!   text colour of unstyled widgets) start from the right values.

use std::sync::OnceLock;

use iced::widget::{button, container, pick_list, scrollable, text_editor};
use iced::{Background, Border, Color, Font, Shadow, Theme, Vector, border, font, overlay};

use lattice_protocol::RunStatus;

use crate::spans::SpanKind;
use crate::textmetrics::pick_family;

pub const CANVAS: Color = Color::from_rgb8(0x00, 0x00, 0x00);
pub const SIDEBAR: Color = Color::from_rgb8(0x0b, 0x0b, 0x0b);
pub const SURFACE: Color = Color::from_rgb8(0x12, 0x12, 0x12);
pub const RAISED: Color = Color::from_rgb8(0x1a, 0x1a, 0x1a);
pub const HOVER: Color = Color::from_rgb8(0x22, 0x22, 0x22);
pub const LINE: Color = Color::from_rgb8(0x26, 0x26, 0x26);
pub const LINE_SOFT: Color = Color::from_rgb8(0x1b, 0x1b, 0x1b);

pub const TEXT: Color = Color::from_rgb8(0xec, 0xe8, 0xdc);
pub const TEXT_DIM: Color = Color::from_rgb8(0xaa, 0xa5, 0x92);
pub const TEXT_FAINT: Color = Color::from_rgb8(0x8f, 0x8a, 0x7c);

pub const GOLD: Color = Color::from_rgb8(0xe6, 0xc4, 0x6a);
pub const GOLD_STRONG: Color = Color::from_rgb8(0xf0, 0xd3, 0x8b);
pub const GOLD_DIM: Color = Color::from_rgb8(0x8f, 0x7a, 0x3e);
pub const GOLD_WASH: Color = Color::from_rgba8(0xe6, 0xc4, 0x6a, 0.10);
pub const ON_GOLD: Color = Color::from_rgb8(0x0a, 0x0a, 0x0a);

pub const CAUTION: Color = Color::from_rgb8(0xea, 0x80, 0x27);
pub const DANGER: Color = Color::from_rgb8(0xd9, 0x70, 0x7f);
pub const POSITIVE: Color = Color::from_rgb8(0x54, 0xb9, 0xa5);

/// The model-call colour: a cool blue-grey that is not any of the brand's
/// warm colours, so model spans read apart from agents and tools at a glance.
pub const MODEL: Color = Color::from_rgb8(0x9f, 0xb4, 0xd9);

/// The colour of a span kind's bar and chip. A guardrail is red only when its
/// tripwire fired.
pub fn kind_color(kind: SpanKind, triggered: bool) -> Color {
    match kind {
        SpanKind::Agent => GOLD,
        SpanKind::Tool => POSITIVE,
        SpanKind::Handoff => CAUTION,
        SpanKind::Guardrail if triggered => DANGER,
        SpanKind::Guardrail => TEXT_DIM,
        SpanKind::Model => MODEL,
        SpanKind::Task | SpanKind::Turn => TEXT_FAINT,
        SpanKind::Custom | SpanKind::Mcp | SpanKind::Voice | SpanKind::Unknown => TEXT_DIM,
    }
}

/// The colour of a run's state dot and status word.
pub fn status_color(status: RunStatus) -> Color {
    match status {
        RunStatus::Running => GOLD,
        RunStatus::Completed => POSITIVE,
        RunStatus::Failed | RunStatus::Refused => DANGER,
        RunStatus::Stopped | RunStatus::Interrupted => TEXT_FAINT,
    }
}

pub fn status_word(status: RunStatus) -> &'static str {
    match status {
        RunStatus::Running => "Running",
        RunStatus::Completed => "Completed",
        RunStatus::Failed => "Failed",
        RunStatus::Refused => "Refused",
        RunStatus::Stopped => "Stopped",
        RunStatus::Interrupted => "Interrupted",
    }
}

pub fn with_alpha(color: Color, alpha: f32) -> Color {
    Color { a: alpha, ..color }
}

fn channel(c: f32) -> f64 {
    let c = f64::from(c);
    if c <= 0.03928 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// WCAG 2.x relative luminance.
pub fn luminance(color: Color) -> f64 {
    0.2126 * channel(color.r) + 0.7152 * channel(color.g) + 0.0722 * channel(color.b)
}

/// WCAG 2.x contrast ratio, from 1 to 21.
pub fn contrast(a: Color, b: Color) -> f64 {
    let (la, lb) = (luminance(a), luminance(b));
    let (hi, lo) = if la >= lb { (la, lb) } else { (lb, la) };
    (hi + 0.05) / (lo + 0.05)
}

/// The iced theme: iced's defaults (unstyled text, markdown links) start from
/// the brand's colours.
pub fn theme() -> Theme {
    Theme::custom(
        "Lattice".to_string(),
        iced::theme::Palette {
            background: CANVAS,
            text: TEXT,
            primary: GOLD,
            success: POSITIVE,
            warning: CAUTION,
            danger: DANGER,
        },
    )
}

/// The window's own style: canvas background, brand text colour.
pub fn app_style(_theme: &Theme) -> iced::theme::Style {
    iced::theme::Style {
        background_color: CANVAS,
        text_color: TEXT,
    }
}

// ----------------------------------------------------------------- fonts

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fonts {
    pub ui: Font,
    pub ui_strong: Font,
    pub mono: Font,
}

impl Default for Fonts {
    fn default() -> Self {
        Self::from_names(None, None)
    }
}

const UI_FAMILIES: [&str; 2] = ["Segoe UI Variable Text", "Segoe UI"];
const MONO_FAMILIES: [&str; 2] = ["Cascadia Code", "Consolas"];

impl Fonts {
    pub fn from_names(ui: Option<&'static str>, mono: Option<&'static str>) -> Self {
        let ui_font = ui.map_or(Font::DEFAULT, Font::with_name);
        Self {
            ui: ui_font,
            ui_strong: Font {
                weight: font::Weight::Semibold,
                ..ui_font
            },
            mono: mono.map_or(Font::MONOSPACE, Font::with_name),
        }
    }

    /// Resolve the families against the fonts iced's font system found. The
    /// system is the one iced draws with, so a family found here is one that
    /// draws; it loads the system fonts once, which iced would have done anyway.
    pub fn detect() -> Self {
        let installed = |name: &str| -> bool {
            let system = iced::advanced::graphics::text::font_system();
            let Ok(mut guard) = system.write() else {
                return false;
            };
            guard.raw().db().faces().any(|face| {
                face.families
                    .iter()
                    .any(|(family, _)| family.eq_ignore_ascii_case(name))
            })
        };
        Self::from_names(
            pick_family(&UI_FAMILIES, installed),
            pick_family(&MONO_FAMILIES, installed),
        )
    }
}

static FONTS: OnceLock<Fonts> = OnceLock::new();

/// Fix the fonts for the process. The first call wins.
pub fn set_fonts(fonts: Fonts) {
    let _ = FONTS.set(fonts);
}

/// The process's fonts (the generic families until [`set_fonts`] runs).
pub fn fonts() -> Fonts {
    FONTS.get().copied().unwrap_or_default()
}

// ---------------------------------------------------------------- styles

fn rounded(radius: f32) -> Border {
    border::rounded(radius)
}

fn outlined(radius: f32, color: Color) -> Border {
    Border {
        color,
        width: 1.0,
        radius: radius.into(),
    }
}

pub fn primary_button(_: &Theme, status: button::Status) -> button::Style {
    let (bg, text) = match status {
        button::Status::Active => (GOLD, ON_GOLD),
        button::Status::Hovered => (GOLD_STRONG, ON_GOLD),
        button::Status::Pressed => (GOLD_DIM, ON_GOLD),
        button::Status::Disabled => (LINE, TEXT_FAINT),
    };
    button::Style {
        background: Some(Background::Color(bg)),
        text_color: text,
        border: rounded(8.0),
        ..button::Style::default()
    }
}

pub fn secondary_button(_: &Theme, status: button::Status) -> button::Style {
    let (bg, text, edge) = match status {
        button::Status::Active => (RAISED, TEXT, LINE),
        button::Status::Hovered => (HOVER, TEXT, GOLD_DIM),
        button::Status::Pressed => (LINE, TEXT, GOLD_DIM),
        button::Status::Disabled => (SURFACE, TEXT_FAINT, LINE),
    };
    button::Style {
        background: Some(Background::Color(bg)),
        text_color: text,
        border: outlined(8.0, edge),
        ..button::Style::default()
    }
}

pub fn ghost_button(_: &Theme, status: button::Status) -> button::Style {
    let (bg, text) = match status {
        button::Status::Active => (None, TEXT_DIM),
        button::Status::Hovered => (Some(HOVER), TEXT),
        button::Status::Pressed => (Some(LINE), TEXT),
        button::Status::Disabled => (None, TEXT_FAINT),
    };
    button::Style {
        background: bg.map(Background::Color),
        text_color: text,
        border: rounded(6.0),
        ..button::Style::default()
    }
}

/// One segment of a segmented control (Timeline | Graph, Overview | Input | Output).
pub fn segment_button(selected: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| {
        let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
        let (bg, text, edge) = if selected {
            (RAISED, GOLD, GOLD_DIM)
        } else if hovered {
            (HOVER, TEXT, LINE)
        } else {
            (SURFACE, TEXT_DIM, LINE)
        };
        button::Style {
            background: Some(Background::Color(bg)),
            text_color: text,
            border: outlined(8.0, edge),
            ..button::Style::default()
        }
    }
}

/// A row of the runs list.
pub fn list_row(selected: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| {
        let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
        let bg = if selected {
            Some(RAISED)
        } else if hovered {
            Some(HOVER)
        } else {
            None
        };
        let edge = if selected {
            GOLD_DIM
        } else {
            Color::TRANSPARENT
        };
        button::Style {
            background: bg.map(Background::Color),
            text_color: TEXT,
            border: outlined(8.0, edge),
            ..button::Style::default()
        }
    }
}

/// The rail's mode button.
pub fn rail_button(selected: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| {
        let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
        let (bg, text) = if selected {
            (Some(GOLD_WASH), GOLD)
        } else if hovered {
            (Some(HOVER), TEXT)
        } else {
            (None, TEXT_DIM)
        };
        button::Style {
            background: bg.map(Background::Color),
            text_color: text,
            border: rounded(10.0),
            ..button::Style::default()
        }
    }
}

pub fn sidebar(_: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(SIDEBAR)),
        text_color: Some(TEXT),
        ..container::Style::default()
    }
}

pub fn canvas_bg(_: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(CANVAS)),
        text_color: Some(TEXT),
        ..container::Style::default()
    }
}

/// A raised panel on the canvas.
pub fn panel(_: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(SURFACE)),
        text_color: Some(TEXT),
        border: outlined(12.0, LINE),
        ..container::Style::default()
    }
}

/// A flat surface without a border (the detail column).
pub fn surface(_: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(SURFACE)),
        text_color: Some(TEXT),
        ..container::Style::default()
    }
}

pub fn line(_: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(LINE)),
        ..container::Style::default()
    }
}

/// A block of code or a long text: darker than the panel it sits in.
pub fn well(_: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(CANVAS)),
        text_color: Some(TEXT),
        border: outlined(8.0, LINE),
        ..container::Style::default()
    }
}

/// The floating card of the new-run overlay.
pub fn card(_: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(SURFACE)),
        text_color: Some(TEXT),
        border: outlined(18.0, LINE),
        shadow: Shadow {
            color: Color::from_rgba(0.0, 0.0, 0.0, 0.55),
            offset: Vector::new(0.0, 12.0),
            blur_radius: 40.0,
        },
        ..container::Style::default()
    }
}

pub fn scrim(_: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(Color::from_rgba(0.0, 0.0, 0.0, 0.62))),
        ..container::Style::default()
    }
}

pub fn banner(_: &Theme) -> container::Style {
    container::Style {
        background: Some(Background::Color(RAISED)),
        text_color: Some(GOLD),
        border: Border {
            color: LINE,
            width: 0.0,
            radius: 0.0.into(),
        },
        ..container::Style::default()
    }
}

/// A small pill: tinted background, coloured text and edge.
pub fn chip(color: Color) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(with_alpha(color, 0.14))),
        text_color: Some(color),
        border: outlined(6.0, with_alpha(color, 0.45)),
        ..container::Style::default()
    }
}

/// A filled dot.
pub fn dot(color: Color) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(color)),
        border: rounded(4.0),
        ..container::Style::default()
    }
}

/// A notice with a coloured edge: the run's error or a refusal.
pub fn notice(color: Color) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(with_alpha(color, 0.10))),
        text_color: Some(TEXT),
        border: outlined(8.0, with_alpha(color, 0.55)),
        ..container::Style::default()
    }
}

pub fn scrollbars(theme: &Theme, status: scrollable::Status) -> scrollable::Style {
    let mut style = scrollable::default(theme, status);
    let (thumb, hovered) = match status {
        scrollable::Status::Active { .. } => (LINE, false),
        scrollable::Status::Hovered {
            is_vertical_scrollbar_hovered,
            is_horizontal_scrollbar_hovered,
            ..
        } => (
            GOLD_DIM,
            is_vertical_scrollbar_hovered || is_horizontal_scrollbar_hovered,
        ),
        scrollable::Status::Dragged { .. } => (GOLD, true),
    };
    for rail in [&mut style.vertical_rail, &mut style.horizontal_rail] {
        rail.background = None;
        rail.scroller.background = Background::Color(if hovered { thumb } else { LINE });
        rail.scroller.border = rounded(3.0);
    }
    style.container = container::Style::default();
    style
}

pub fn editor(_: &Theme, status: text_editor::Status) -> text_editor::Style {
    let edge = match status {
        text_editor::Status::Focused { .. } => GOLD,
        text_editor::Status::Hovered => GOLD_DIM,
        _ => LINE,
    };
    text_editor::Style {
        background: Background::Color(CANVAS),
        border: outlined(8.0, edge),
        placeholder: TEXT_FAINT,
        value: TEXT,
        selection: with_alpha(GOLD, 0.30),
    }
}

pub fn picker(_: &Theme, status: pick_list::Status) -> pick_list::Style {
    let edge = match status {
        pick_list::Status::Active => LINE,
        pick_list::Status::Hovered | pick_list::Status::Opened { .. } => GOLD_DIM,
    };
    pick_list::Style {
        text_color: TEXT,
        placeholder_color: TEXT_FAINT,
        handle_color: TEXT_DIM,
        background: Background::Color(CANVAS),
        border: outlined(8.0, edge),
    }
}

pub fn picker_menu(_: &Theme) -> overlay::menu::Style {
    overlay::menu::Style {
        background: Background::Color(RAISED),
        border: outlined(8.0, LINE),
        text_color: TEXT,
        selected_text_color: ON_GOLD,
        selected_background: Background::Color(GOLD),
        shadow: Shadow {
            color: Color::from_rgba(0.0, 0.0, 0.0, 0.5),
            offset: Vector::new(0.0, 8.0),
            blur_radius: 24.0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (name, colour) for every colour the interface uses for text or for a span kind.
    fn text_and_kind_colors() -> Vec<(&'static str, Color)> {
        vec![
            ("text", TEXT),
            ("text-dim", TEXT_DIM),
            ("text-faint", TEXT_FAINT),
            ("gold", GOLD),
            ("gold-strong", GOLD_STRONG),
            ("caution", CAUTION),
            ("danger", DANGER),
            ("positive", POSITIVE),
            ("model", MODEL),
            ("kind agent", kind_color(SpanKind::Agent, false)),
            ("kind tool", kind_color(SpanKind::Tool, false)),
            ("kind handoff", kind_color(SpanKind::Handoff, false)),
            (
                "kind guardrail (passed)",
                kind_color(SpanKind::Guardrail, false),
            ),
            (
                "kind guardrail (triggered)",
                kind_color(SpanKind::Guardrail, true),
            ),
            ("kind model", kind_color(SpanKind::Model, false)),
            ("kind task", kind_color(SpanKind::Task, false)),
            ("kind turn", kind_color(SpanKind::Turn, false)),
            ("kind custom", kind_color(SpanKind::Custom, false)),
            ("kind unknown", kind_color(SpanKind::Unknown, false)),
            ("status running", status_color(RunStatus::Running)),
            ("status completed", status_color(RunStatus::Completed)),
            ("status failed", status_color(RunStatus::Failed)),
            ("status stopped", status_color(RunStatus::Stopped)),
        ]
    }

    #[test]
    fn wcag_formula_matches_its_reference_values() {
        assert!((contrast(Color::BLACK, Color::WHITE) - 21.0).abs() < 1e-9);
        assert!((contrast(Color::WHITE, Color::WHITE) - 1.0).abs() < 1e-9);
        // #767676 on white is the classic 4.54:1 boundary case.
        let grey = Color::from_rgb8(0x76, 0x76, 0x76);
        let ratio = contrast(grey, Color::WHITE);
        assert!((4.5..4.6).contains(&ratio), "{ratio}");
    }

    #[test]
    fn every_text_and_kind_colour_holds_45_to_1_on_surface_and_canvas() {
        let mut failures = Vec::new();
        for (name, color) in text_and_kind_colors() {
            for (bg_name, bg) in [("surface", SURFACE), ("canvas", CANVAS)] {
                let ratio = contrast(color, bg);
                if ratio < 4.5 {
                    failures.push(format!("{name} on {bg_name}: {ratio:.2}"));
                }
            }
        }
        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn text_on_the_gold_button_holds_45_to_1() {
        for (name, bg) in [
            ("gold", GOLD),
            ("gold-strong", GOLD_STRONG),
            ("gold-dim", GOLD_DIM),
        ] {
            let ratio = contrast(ON_GOLD, bg);
            assert!(ratio >= 4.5, "on-gold on {name}: {ratio:.2}");
        }
        // Text on the raised surfaces the interface puts text on.
        for bg in [RAISED, HOVER, SIDEBAR] {
            assert!(
                contrast(TEXT, bg) >= 4.5
                    && contrast(TEXT_DIM, bg) >= 4.5
                    && contrast(GOLD, bg) >= 4.5
            );
        }
    }

    #[test]
    fn the_new_model_colour_is_distinct_from_every_brand_kind_colour() {
        for other in [GOLD, POSITIVE, CAUTION, DANGER, TEXT_DIM, TEXT_FAINT] {
            let d =
                (MODEL.r - other.r).abs() + (MODEL.g - other.g).abs() + (MODEL.b - other.b).abs();
            assert!(d > 0.25, "{other:?}");
        }
    }

    fn token_hex(css: &str, name: &str) -> Option<[u8; 3]> {
        let needle = format!("--{name}:");
        let line = css.lines().find(|l| l.trim_start().starts_with(&needle))?;
        let value = line.split(':').nth(1)?.trim().trim_end_matches(';').trim();
        let hex = value.strip_prefix('#')?;
        if hex.len() != 6 {
            return None;
        }
        Some([
            u8::from_str_radix(&hex[0..2], 16).ok()?,
            u8::from_str_radix(&hex[2..4], 16).ok()?,
            u8::from_str_radix(&hex[4..6], 16).ok()?,
        ])
    }

    #[test]
    fn the_palette_is_the_token_files_palette() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../lattice_desktop/src/styles/tokens.css");
        let Ok(css) = std::fs::read_to_string(&path) else {
            // The crate is built outside the repository (a published copy): the
            // constants below are then the only source, and the spec's values are
            // asserted directly instead.
            assert_eq!(GOLD.into_rgba8()[..3], [0xe6, 0xc4, 0x6a]);
            return;
        };
        let pairs: [(&str, Color); 15] = [
            ("canvas", CANVAS),
            ("sidebar", SIDEBAR),
            ("surface", SURFACE),
            ("raised", RAISED),
            ("hover", HOVER),
            ("line", LINE),
            ("text", TEXT),
            ("text-dim", TEXT_DIM),
            ("text-faint", TEXT_FAINT),
            ("gold", GOLD),
            ("gold-strong", GOLD_STRONG),
            ("gold-dim", GOLD_DIM),
            ("caution", CAUTION),
            ("danger", DANGER),
            ("positive", POSITIVE),
        ];
        for (name, color) in pairs {
            let hex =
                token_hex(&css, name).unwrap_or_else(|| panic!("--{name} not found in tokens.css"));
            assert_eq!(color.into_rgba8()[..3], hex, "--{name}");
        }
    }

    #[test]
    fn font_resolution_falls_back_to_the_generic_families() {
        let none = Fonts::from_names(None, None);
        assert_eq!(none.ui, Font::DEFAULT);
        assert_eq!(none.mono, Font::MONOSPACE);
        assert_eq!(none.ui_strong.weight, font::Weight::Semibold);
        let named = Fonts::from_names(Some("Segoe UI"), Some("Consolas"));
        assert_eq!(named.ui.family, font::Family::Name("Segoe UI"));
        assert_eq!(named.mono.family, font::Family::Name("Consolas"));
    }

    #[test]
    fn detection_resolves_only_the_named_families_or_the_generic_ones() {
        // Reads the machine's installed fonts (CPU only): whatever it finds must be
        // one of the candidates, else the generic family, never anything else.
        let fonts = Fonts::detect();
        match fonts.ui.family {
            font::Family::Name(name) => assert!(UI_FAMILIES.contains(&name), "{name}"),
            other => assert_eq!(other, font::Family::SansSerif),
        }
        match fonts.mono.family {
            font::Family::Name(name) => assert!(MONO_FAMILIES.contains(&name), "{name}"),
            other => assert_eq!(other, font::Family::Monospace),
        }
        assert_eq!(fonts.ui_strong.weight, font::Weight::Semibold);
        // The first candidate wins over a later one when both are installed.
        assert_eq!(
            pick_family(&UI_FAMILIES, |_| true),
            Some("Segoe UI Variable Text")
        );
        assert_eq!(
            pick_family(&MONO_FAMILIES, |name| name == "Consolas"),
            Some("Consolas")
        );
    }

    #[test]
    fn status_words_and_colours_cover_every_status() {
        for status in [
            RunStatus::Running,
            RunStatus::Completed,
            RunStatus::Failed,
            RunStatus::Refused,
            RunStatus::Stopped,
            RunStatus::Interrupted,
        ] {
            assert!(!status_word(status).is_empty());
            let _ = status_color(status);
        }
        assert_eq!(
            status_color(RunStatus::Failed),
            status_color(RunStatus::Refused)
        );
    }
}
