//! The runs column: the "New run" button and the grouped list of runs.
//!
//! The list is built inside a `lazy` widget keyed on the list's version, the
//! selected run, the running dot's pulse phase and the minute, so it is
//! rebuilt when a row's content changes, not on every message. Its rows are
//! owned data (`'static`), which is what `lazy` needs.

use iced::widget::{Column, Row, button, column, container, lazy, row, scrollable, space, text};
use iced::{Alignment, Length};

use super::{Element, label, strong};
use crate::app::{App, Message};
use crate::runlist::{ListEntry, RunRow, minute_bucket};
use crate::textmetrics::truncate_to_width;
use crate::theme;

/// The width a row's text can use in a runs column `column_w` wide: the column,
/// less its gutters, the button's padding, the dot and the gap after it.
fn text_width(column_w: f32) -> f32 {
    column_w - 2.0 * 12.0 - 2.0 * 10.0 - 8.0 - 10.0 - 6.0
}

pub(super) fn view(app: &App, width: f32) -> Element<'_> {
    let new_run = button(
        row![
            strong("New run", 13.5, theme::ON_GOLD),
            space::horizontal(),
            label("Ctrl+N", 11.0, theme::ON_GOLD)
        ]
        .align_y(Alignment::Center),
    )
    .width(Length::Fill)
    .padding([9.0, 14.0])
    .style(theme::primary_button);
    let new_run = if app.status.refusal.is_none() {
        new_run.on_press(Message::OpenNewRun)
    } else {
        new_run
    };

    let dependency = (
        app.list_version,
        app.selected.as_ref().map(|s| s.id().to_string()),
        app.pulse(),
        minute_bucket(app.now),
        width as u32,
    );
    let entries = &app.entries;
    let list = lazy(dependency, move |(_, selected, pulse, _, width)| {
        list_view(
            entries,
            selected.as_deref(),
            *pulse,
            text_width(*width as f32),
        )
    });

    let mut footer = Column::new();
    if app.hidden_runs > 0 {
        footer = footer.push(
            container(label(
                format!(
                    "The {} newest runs are listed; {} older ones are not.",
                    crate::runlist::LIST_CAP,
                    app.hidden_runs
                ),
                11.0,
                theme::TEXT_FAINT,
            ))
            .padding([8.0, 14.0]),
        );
    }
    if let Some(reason) = &app.status.refusal {
        footer = footer.push(
            container(
                container(label(reason.clone(), 12.0, theme::TEXT))
                    .padding(10)
                    .width(Length::Fill)
                    .style(theme::notice(theme::CAUTION)),
            )
            .padding([4.0, 12.0]),
        );
    }
    if app.entries.is_empty() {
        footer = footer.push(
            container(label(
                "No runs yet. Start one with New run.",
                12.5,
                theme::TEXT_DIM,
            ))
            .padding([8.0, 14.0]),
        );
    }

    let status = container(
        column![
            label(app.status.runtime.clone(), 10.5, theme::TEXT_FAINT),
            label(
                format!("Traces stay on {}.", app.status.traces),
                10.5,
                theme::TEXT_FAINT
            )
        ]
        .spacing(2),
    )
    .padding([8.0, 14.0]);
    container(column![
        container(new_run).padding(12),
        scrollable(list)
            .height(Length::Fill)
            .style(theme::scrollbars),
        footer,
        status
    ])
    .width(width)
    .height(Length::Fill)
    .style(theme::sidebar)
    .into()
}

fn list_view(
    entries: &[ListEntry],
    selected: Option<&str>,
    pulse: u8,
    text_w: f32,
) -> Column<'static, Message> {
    let mut list = Column::new()
        .spacing(2)
        .padding([0.0, 12.0])
        .width(Length::Fill);
    for entry in entries {
        match entry {
            ListEntry::Header { group, count } => {
                list = list.push(
                    container(strong(
                        format!("{} · {count}", group.title().to_uppercase()),
                        10.5,
                        theme::TEXT_FAINT,
                    ))
                    .padding([10.0, 4.0]),
                );
            }
            ListEntry::Run(run) => {
                list = list.push(run_row(
                    run,
                    selected == Some(run.id.as_str()),
                    pulse,
                    text_w,
                ))
            }
        }
    }
    list
}

fn run_row(run: &RunRow, selected: bool, pulse: u8, text_w: f32) -> Element<'static> {
    use lattice_protocol::RunStatus;
    let color = if run.status == RunStatus::Running && pulse == 1 {
        theme::GOLD_DIM
    } else {
        theme::status_color(run.status)
    };
    let dot = column![
        space().height(4),
        container(space().width(8).height(8)).style(theme::dot(color))
    ];
    let task = truncate_to_width(&run.task, 13.0, text_w).into_owned();
    let meta = truncate_to_width(&run.meta, 11.5, text_w).into_owned();
    let mut lines = Column::new()
        .spacing(3)
        .width(Length::Fill)
        .push(label(task, 13.0, theme::TEXT).wrapping(text::Wrapping::None))
        .push(label(meta, 11.5, theme::TEXT_DIM).wrapping(text::Wrapping::None));
    if run.remote {
        lines = lines.push(
            container(label("leaves this machine", 10.5, theme::CAUTION))
                .padding([1.0, 6.0])
                .style(theme::chip(theme::CAUTION)),
        );
    }
    lines = lines.push(
        Row::new()
            .push(label(run.when.clone(), 11.0, theme::TEXT_FAINT))
            .push(space::horizontal())
            .push(label(run.tokens.clone(), 11.0, theme::TEXT_FAINT)),
    );
    // The selected row looks the same hovered or not and pressing it does
    // nothing, so it is not pressable: no status to flip, so no frame when the
    // pointer crosses it, and no message for a click that changes nothing.
    button(row![dot, lines].spacing(10))
        .width(Length::Fill)
        .padding([9.0, 10.0])
        .style(theme::list_row(selected))
        .on_press_maybe((!selected).then(|| Message::SelectRun(run.id.clone())))
        .into()
}
