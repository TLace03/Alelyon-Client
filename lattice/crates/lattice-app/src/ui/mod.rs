//! The window's widget tree: pure functions of `&App`.
//!
//! Layout (spec): a 56 px rail, a 300 px runs column, the centre (run header,
//! Timeline | Graph toggle, the view, the output), and a 380 px span-detail
//! column that exists only while a span is selected.
//!
//! Invariants:
//! - `view()` reads state and builds widgets; it parses nothing, builds no span
//!   tree and calls no clock. Everything it shows was prepared in `update`.
//! - Every string a person reads is drawn as plain text, except the run's final
//!   output, which iced's markdown widget renders (no HTML, no scripts, links
//!   never opened).
//! - The two canvases (waterfall, graph) draw from caches that `update` clears
//!   when their data changes; see `app.rs`.

mod detail_view;
mod frames;
mod graphview;
mod mark;
mod overlay;
mod runs;
mod waterfall;

use iced::widget::{
    Column, Row, Text, button, canvas, column, container, markdown, responsive, row, scrollable,
    space, stack, text,
};
use iced::{Alignment, Color, Length, border, padding};

use crate::app::{App, Message, Selected, ViewMode};
use crate::clock::{format_count, format_duration};
use crate::perf;
use crate::runstate::RunState;
use crate::theme::{self, fonts};

pub type Element<'a> = iced::Element<'a, Message>;

/// The runs column's width.
pub const RUNS_W: f32 = 300.0;
/// The span-detail column's width.
pub const DETAIL_W: f32 = 380.0;
/// The rail's width.
pub const RAIL_W: f32 = 56.0;
/// The narrowest the centre column is kept before the side columns give way.
pub const CENTRE_MIN: f32 = 420.0;
/// The narrowest the runs column gets when the window is too narrow for the spec's widths.
pub const RUNS_MIN: f32 = 240.0;
/// The narrowest the detail column gets in a narrow window.
pub const DETAIL_MIN: f32 = 300.0;

/// The widths of the runs and detail columns in a window `window_width` wide.
///
/// The spec's 300 and 380 px are used whenever the centre column keeps
/// [`CENTRE_MIN`] pixels. In a narrower window (1000 px is the minimum, which
/// would leave the centre 264 px with the detail column open) the detail column
/// gives way first, down to [`DETAIL_MIN`], then the runs column, down to
/// [`RUNS_MIN`]. The centre never shrinks to make room for them.
pub fn column_widths(window_width: f32, detail_shown: bool) -> (f32, f32) {
    let rules = if detail_shown { 3.0 } else { 2.0 };
    let mut runs = RUNS_W;
    let mut detail = if detail_shown { DETAIL_W } else { 0.0 };
    let centre = window_width - RAIL_W - rules - runs - detail;
    let mut deficit = (CENTRE_MIN - centre).max(0.0);
    if detail_shown {
        let give = deficit.min(detail - DETAIL_MIN);
        detail -= give;
        deficit -= give;
    }
    runs -= deficit.min(runs - RUNS_MIN);
    (runs, detail)
}

pub(crate) fn label<'a>(content: impl text::IntoFragment<'a>, size: f32, color: Color) -> Text<'a> {
    text(content).size(size).color(color).font(fonts().ui)
}

pub(crate) fn strong<'a>(
    content: impl text::IntoFragment<'a>,
    size: f32,
    color: Color,
) -> Text<'a> {
    text(content)
        .size(size)
        .color(color)
        .font(fonts().ui_strong)
}

pub(crate) fn mono<'a>(content: impl text::IntoFragment<'a>, size: f32, color: Color) -> Text<'a> {
    text(content).size(size).color(color).font(fonts().mono)
}

/// A one-pixel vertical rule.
fn rule_v<'a>() -> Element<'a> {
    container(space())
        .width(1)
        .height(Length::Fill)
        .style(theme::line)
        .into()
}

pub fn view(app: &App) -> Element<'_> {
    perf::view_called();
    let mut page = Column::new().width(Length::Fill).height(Length::Fill);
    if let Some(banner) = &app.options.banner {
        page = page.push(
            container(label(banner.as_str(), 12.0, theme::GOLD))
                .width(Length::Fill)
                .center_x(Length::Fill)
                .padding([5.0, 14.0])
                .style(theme::banner),
        );
    }
    let detail_shown = app.selected.as_ref().is_some_and(|s| s.detail.is_some());
    // The columns' widths depend on the window's width, which `responsive` supplies.
    let body = responsive(move |size| {
        let (runs_w, detail_w) = column_widths(size.width, detail_shown);
        let mut body = Row::new()
            .height(Length::Fill)
            .push(rail(app))
            .push(rule_v())
            .push(runs::view(app, runs_w))
            .push(rule_v())
            .push(centre(app));
        if let Some(sel) = app.selected.as_ref().filter(|s| s.detail.is_some()) {
            body = body
                .push(rule_v())
                .push(detail_view::view(app, sel, detail_w));
        }
        body.into()
    });
    let base: Element<'_> = container(page.push(body))
        .width(Length::Fill)
        .height(Length::Fill)
        .style(theme::canvas_bg)
        .into();
    let root: Element<'_> = match &app.overlay {
        Some(new_run) => stack![base, overlay::view(app, new_run)].into(),
        None => base,
    };
    // Counts the frames for `LATTICE_PERF`; invisible to everything else.
    frames::Frames::new(root).into()
}

// ------------------------------------------------------------------ rail

fn rail(app: &App) -> Element<'_> {
    let mark = canvas(mark::Mark {
        cache: &app.mark_cache,
    })
    .width(34)
    .height(34);
    let bar = |w: f32, offset: f32| {
        container(container(space().width(w).height(3)).style(theme::dot(theme::GOLD)))
            .padding(padding::left(offset))
    };
    let icon = column![bar(16.0, 0.0), bar(11.0, 4.0), bar(13.0, 8.0)].spacing(3);
    let traces = button(
        column![icon, label("Traces", 10.5, theme::GOLD)]
            .spacing(6)
            .align_x(Alignment::Center)
            .width(Length::Fill),
    )
    .width(Length::Fill)
    .padding([9.0, 2.0])
    .style(theme::rail_button(true));
    container(
        column![
            container(mark).padding([6.0, 0.0]).center_x(Length::Fill),
            traces
        ]
        .spacing(14)
        .padding([14.0, 6.0])
        .width(Length::Fill),
    )
    .width(RAIL_W)
    .height(Length::Fill)
    .style(theme::sidebar)
    .into()
}

// ---------------------------------------------------------------- centre

fn centre(app: &App) -> Element<'_> {
    let Some(sel) = &app.selected else {
        return container(
            column![
                strong("No run selected", 16.0, theme::TEXT),
                label(
                    "Pick a run on the left, or start one with New run (Ctrl+N).",
                    13.0,
                    theme::TEXT_DIM
                ),
                app.notice.as_deref().map_or_else(
                    || Element::from(space()),
                    |n| container(label(n, 12.5, theme::TEXT))
                        .padding(10)
                        .style(theme::notice(theme::CAUTION))
                        .into()
                ),
            ]
            .spacing(8)
            .align_x(Alignment::Center),
        )
        .center(Length::Fill)
        .into();
    };
    let mut col = Column::new()
        .spacing(10)
        .padding(12)
        .width(Length::Fill)
        .height(Length::Fill)
        .push(header(app, sel));
    if let Some(notice) = &app.notice {
        col = col.push(
            container(label(notice.as_str(), 12.5, theme::TEXT))
                .padding(10)
                .width(Length::Fill)
                .style(theme::notice(theme::CAUTION)),
        );
    }
    col = col.push(toolbar(app, sel));
    let has_output = !sel.output.is_empty() || !notices(&sel.state).is_empty();
    let visual = match app.view_mode {
        ViewMode::Timeline => timeline(app, sel),
        ViewMode::Graph => graphview::view(app, sel),
    };
    col = col.push(
        container(visual)
            .width(Length::Fill)
            .height(Length::FillPortion(3)),
    );
    if has_output {
        col = col.push(output_panel(sel));
    }
    col.into()
}

/// `text` cut to `max` characters with an ellipsis, for a header.
fn clip_chars(text: &str, max: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max {
        flat
    } else {
        format!("{}…", flat.chars().take(max).collect::<String>().trim_end())
    }
}

fn stat<'a>(name: &'a str, value: String) -> Element<'a> {
    column![
        label(name, 10.5, theme::TEXT_FAINT),
        label(value, 13.0, theme::TEXT)
    ]
    .spacing(2)
    .into()
}

/// The run's duration: to its end, or to now while it works.
fn run_duration(state: &RunState, now: f64) -> f64 {
    let end = state.ended_at.or(state.summary.ended_at).unwrap_or(now);
    (end - state.summary.created_at).max(0.0)
}

fn header<'a>(app: &'a App, sel: &'a Selected) -> Element<'a> {
    let state = &sel.state;
    let summary = &state.summary;
    let status_color = theme::status_color(state.status);
    let status = container(
        row![
            container(space().width(8).height(8)).style(theme::dot(
                if state.status.is_active() && app.pulse() == 1 {
                    theme::GOLD_DIM
                } else {
                    status_color
                }
            )),
            label(theme::status_word(state.status), 12.0, status_color)
        ]
        .spacing(7)
        .align_y(Alignment::Center),
    )
    .padding([3.0, 9.0])
    .style(theme::chip(status_color));
    let tokens = state.usage.map_or_else(
        || "not reported".to_string(),
        |u| {
            format!(
                "{} → {}",
                format_count(u.input_tokens),
                format_count(u.output_tokens)
            )
        },
    );
    let errors = sel.totals.errors + usize::from(state.error.is_some() && sel.totals.errors == 0);
    let stats = row![
        stat("Agent", summary.agent_label.clone()),
        stat("Model", summary.model_label.clone()),
        stat("Duration", format_duration(run_duration(state, app.now))),
        stat("Tokens", tokens),
        stat("Tool calls", format_count(sel.totals.tool_calls as u64)),
        stat("Handoffs", format_count(sel.totals.handoffs as u64)),
        stat("Errors", format_count(errors as u64)),
    ]
    .spacing(24)
    .wrap()
    .vertical_spacing(8);
    let title = row![
        strong(clip_chars(&summary.task, 240), 16.5, theme::TEXT).width(Length::Fill),
        status
    ]
    .spacing(12)
    .align_y(Alignment::Start);
    container(column![title, stats].spacing(12))
        .padding(14)
        .width(Length::Fill)
        .style(theme::panel)
        .into()
}

fn segmented<'a, T: Copy + PartialEq + 'a>(
    items: &[(T, &'a str)],
    current: T,
    message: impl Fn(T) -> Message,
) -> Element<'a> {
    let mut segments = Row::new().spacing(4);
    for (value, name) in items {
        // The chosen segment looks the same hovered or not and pressing it changes
        // nothing, so it is not pressable (no status to flip, so no idle frame).
        segments = segments.push(
            button(label(
                *name,
                12.5,
                if *value == current {
                    theme::GOLD
                } else {
                    theme::TEXT_DIM
                },
            ))
            .padding([5.0, 14.0])
            .style(theme::segment_button(*value == current))
            .on_press_maybe((*value != current).then(|| message(*value))),
        );
    }
    segments.into()
}

fn toolbar<'a>(app: &'a App, sel: &'a Selected) -> Element<'a> {
    let toggle = segmented(
        &[(ViewMode::Timeline, "Timeline"), (ViewMode::Graph, "Graph")],
        app.view_mode,
        Message::SetView,
    );
    let mut bar = Row::new()
        .spacing(10)
        .align_y(Alignment::Center)
        .push(toggle)
        .push(space::horizontal());
    if sel.state.status.is_active() {
        bar = bar.push(
            button(label("Stop run", 12.5, theme::TEXT))
                .padding([5.0, 14.0])
                .style(theme::secondary_button)
                .on_press(Message::StopRun),
        );
    }
    bar.into()
}

fn timeline<'a>(app: &'a App, sel: &'a Selected) -> Element<'a> {
    let program = waterfall::Waterfall {
        rows: &sel.rows,
        axis: sel.axis,
        now: app.now,
        selected: sel.selected_row(),
        scroll: sel.scroll,
        known_size: sel.viewport,
        cache: &app.waterfall_cache,
    };
    container(canvas(program).width(Length::Fill).height(Length::Fill))
        .width(Length::Fill)
        .height(Length::Fill)
        .style(theme::panel)
        .into()
}

// --------------------------------------------------------------- output

/// The notices a run carries: a refusal, an error, how it ended.
fn notices(state: &RunState) -> Vec<(Color, String, String)> {
    use lattice_protocol::RunStatus;
    let mut out = Vec::new();
    if let Some((name, message)) = &state.refusal {
        out.push((
            theme::DANGER,
            format!("Refused by the {name} guardrail"),
            message.clone(),
        ));
    }
    if let Some(error) = &state.error {
        out.push((theme::DANGER, "This run failed".to_string(), error.clone()));
    } else if state.status == RunStatus::Failed && state.refusal.is_none() {
        out.push((
            theme::DANGER,
            "This run failed".to_string(),
            "No reason was recorded.".to_string(),
        ));
    }
    match state.status {
        RunStatus::Stopped => out.push((
            theme::TEXT_DIM,
            "Stopped".to_string(),
            "This run was stopped before it finished.".to_string(),
        )),
        RunStatus::Interrupted => out.push((
            theme::CAUTION,
            "Interrupted".to_string(),
            "This run was still working when Lattice last closed.".to_string(),
        )),
        _ => {}
    }
    out
}

fn markdown_settings() -> markdown::Settings {
    let f = fonts();
    markdown::Settings::with_text_size(
        14,
        markdown::Style {
            font: f.ui,
            inline_code_highlight: markdown::Highlight {
                background: theme::RAISED.into(),
                border: border::rounded(4),
            },
            inline_code_padding: padding::left(2).right(2),
            inline_code_color: theme::TEXT,
            inline_code_font: f.mono,
            code_block_font: f.mono,
            link_color: theme::GOLD,
        },
    )
}

fn output_panel<'a>(sel: &'a Selected) -> Element<'a> {
    let mut title = Row::new()
        .spacing(8)
        .align_y(Alignment::Center)
        .push(strong("Output", 12.0, theme::TEXT_DIM));
    if sel.output.streaming {
        title = title.push(label("arriving…", 11.5, theme::GOLD));
    }
    title = title.push(space::horizontal());
    if !sel.output.is_empty() {
        title = title.push(
            button(label("Copy", 11.5, theme::TEXT_DIM))
                .padding([3.0, 10.0])
                .style(theme::ghost_button)
                .on_press(Message::Copy(sel.output.shown.clone())),
        );
    }
    let mut body = Column::new().spacing(10).width(Length::Fill);
    for (color, heading, message) in notices(&sel.state) {
        body = body.push(
            container(
                column![
                    strong(heading, 12.5, color),
                    label(message, 13.0, theme::TEXT)
                ]
                .spacing(3),
            )
            .padding(10)
            .width(Length::Fill)
            .style(theme::notice(color)),
        );
    }
    if !sel.output.is_empty() {
        body = body.push(
            markdown::view(sel.output.content.items(), markdown_settings())
                .map(Message::LinkClicked),
        );
    }
    container(
        column![
            title,
            scrollable(container(body).padding(padding::right(12)))
                .height(Length::Fill)
                .style(theme::scrollbars)
        ]
        .spacing(8),
    )
    .padding(14)
    .width(Length::Fill)
    .height(Length::FillPortion(2))
    .style(theme::panel)
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runstate::RunState;
    use lattice_protocol::{Locality, RunStatus, RunSummary};

    fn state(status: RunStatus) -> RunState {
        RunState::new(RunSummary {
            id: "0123456789abcdef".into(),
            task: "t".into(),
            agent: "a".into(),
            agent_label: "A".into(),
            model: "m".into(),
            model_label: "M".into(),
            locality: Locality::Local,
            status,
            created_at: 100.0,
            updated_at: 100.0,
            ended_at: None,
            trace_id: String::new(),
            usage: None,
            output: None,
            error: None,
            spans: 0,
        })
    }

    #[test]
    fn the_specs_widths_are_kept_whenever_the_centre_stays_usable() {
        assert_eq!(column_widths(1440.0, true), (300.0, 380.0));
        assert_eq!(column_widths(1440.0, false), (300.0, 0.0));
        // Just wide enough for the centre to keep its minimum: 56 + 3 + 300 + 380 + 420.
        assert_eq!(column_widths(1159.0, true), (300.0, 380.0));
        assert_eq!(
            column_widths(1000.0, false),
            (300.0, 0.0),
            "no detail column: 1000 px is plenty"
        );
    }

    #[test]
    fn a_narrow_window_takes_from_the_detail_column_first_then_the_runs_column() {
        let (runs, detail) = column_widths(1100.0, true);
        assert_eq!(
            runs, 300.0,
            "the runs column is untouched while the detail column can give"
        );
        assert!((300.0..380.0).contains(&detail), "{detail}");
        let centre = 1100.0 - RAIL_W - 3.0 - runs - detail;
        assert!((centre - CENTRE_MIN).abs() < 1e-3, "{centre}");
        // The 1000 px minimum window: both give, and the centre gets what is left.
        let (runs, detail) = column_widths(1000.0, true);
        assert_eq!((runs, detail), (RUNS_MIN, DETAIL_MIN));
        assert!(
            1000.0 - RAIL_W - 3.0 - runs - detail > 395.0,
            "the centre is close to usable, not 264 px"
        );
        // Even a silly window never gives a negative or grown column.
        for width in [0.0, 300.0, 700.0, 5000.0] {
            let (r, d) = column_widths(width, true);
            assert!(
                (RUNS_MIN..=RUNS_W).contains(&r) && (DETAIL_MIN..=DETAIL_W).contains(&d),
                "{width}: {r} {d}"
            );
        }
    }

    #[test]
    fn a_failed_run_always_says_so_even_without_a_reason() {
        let n = notices(&state(RunStatus::Failed));
        assert_eq!(n.len(), 1);
        assert_eq!(n[0].1, "This run failed");
        assert_eq!(n[0].2, "No reason was recorded.");
    }

    #[test]
    fn a_refusal_names_the_guardrail_and_repeats_only_its_message() {
        let mut s = state(RunStatus::Refused);
        s.refusal = Some((
            "secrets_stay_local".into(),
            "The task looks like it contains a secret.".into(),
        ));
        let n = notices(&s);
        assert_eq!(n[0].1, "Refused by the secrets_stay_local guardrail");
        assert_eq!(n[0].2, "The task looks like it contains a secret.");
        assert_eq!(n.len(), 1, "a refusal is not also reported as a failure");
    }

    #[test]
    fn running_and_completed_runs_have_no_notice_and_stopped_ones_say_so() {
        assert!(notices(&state(RunStatus::Running)).is_empty());
        assert!(notices(&state(RunStatus::Completed)).is_empty());
        assert_eq!(notices(&state(RunStatus::Stopped))[0].1, "Stopped");
        assert_eq!(notices(&state(RunStatus::Interrupted))[0].1, "Interrupted");
        let mut failed = state(RunStatus::Failed);
        failed.error = Some("The model server did not answer.".into());
        assert_eq!(notices(&failed)[0].2, "The model server did not answer.");
    }

    #[test]
    fn a_header_task_is_flattened_and_bounded() {
        assert_eq!(clip_chars("one\n two\tthree", 100), "one two three");
        let long = "x".repeat(500);
        let clipped = clip_chars(&long, 240);
        assert_eq!(clipped.chars().count(), 241);
        assert!(clipped.ends_with('…'));
    }

    #[test]
    fn the_run_duration_stops_at_the_end_and_counts_up_while_working() {
        let mut s = state(RunStatus::Running);
        assert_eq!(run_duration(&s, 112.5), 12.5);
        s.ended_at = Some(104.0);
        assert_eq!(run_duration(&s, 500.0), 4.0);
        assert_eq!(
            run_duration(&state(RunStatus::Running), 50.0),
            0.0,
            "a clock before the start is zero, not negative"
        );
    }
}
