//! The right column: what the selected span says about itself.
//!
//! A kind chip, the title, the start time (UTC) and the duration, then a
//! segmented control (Overview | Input | Output) over the sections built by
//! `detail.rs`. Every text is plain; long ones sit in their own scrollable with
//! a Copy button, and the whole body scrolls.

use iced::widget::{Column, Row, button, column, container, row, scrollable, space};
use iced::{Alignment, Length};

use super::{Element, label, mono, segmented, strong};
use crate::app::{App, DetailTab, Message, Selected};
use crate::detail::{Block, DetailModel, Field, Tone};
use crate::theme;

/// The tallest a single text block grows before it scrolls inside itself.
const BLOCK_MAX_H: f32 = 280.0;

pub(super) fn view<'a>(app: &'a App, sel: &'a Selected, width: f32) -> Element<'a> {
    let Some(model) = &sel.detail else {
        return space().into();
    };
    let color = theme::kind_color(
        model.kind,
        model.overview.iter().any(|f| f.tone == Tone::Danger),
    );
    let chip = container(label(model.kind.label(), 11.5, color))
        .padding([2.0, 9.0])
        .style(theme::chip(color));
    let close = button(label("Close", 11.5, theme::TEXT_DIM))
        .padding([3.0, 9.0])
        .style(theme::ghost_button)
        .on_press(Message::SelectSpan(None));
    let top = Row::new()
        .align_y(Alignment::Center)
        .push(chip)
        .push(space::horizontal())
        .push(close);
    let times = label(
        format!("Started {} · {}", model.started, model.duration),
        11.5,
        theme::TEXT_FAINT,
    );
    let tabs = segmented(
        &[
            (DetailTab::Overview, "Overview"),
            (DetailTab::Input, "Input"),
            (DetailTab::Output, "Output"),
        ],
        app.tab,
        Message::SetTab,
    );
    let body = match app.tab {
        DetailTab::Overview => overview(model),
        DetailTab::Input => blocks(&model.input, "No input was recorded for this span."),
        DetailTab::Output => blocks(&model.output, "No output was recorded for this span yet."),
    };
    container(
        column![
            top,
            strong(model.title.clone(), 15.0, theme::TEXT),
            times,
            tabs,
            scrollable(container(body).padding(iced::padding::right(12)))
                .height(Length::Fill)
                .style(theme::scrollbars)
        ]
        .spacing(10),
    )
    .padding(16)
    .width(width)
    .height(Length::Fill)
    .style(theme::surface)
    .into()
}

fn field_view(field: &Field) -> Element<'static> {
    let color = match field.tone {
        Tone::Normal => theme::TEXT,
        Tone::Danger => theme::DANGER,
        Tone::Positive => theme::POSITIVE,
    };
    let value = if field.mono {
        mono(field.value.clone(), 12.0, color)
    } else {
        label(field.value.clone(), 13.0, color)
    };
    column![label(field.label.clone(), 10.5, theme::TEXT_FAINT), value]
        .spacing(2)
        .into()
}

fn overview(model: &DetailModel) -> Element<'static> {
    let mut col = Column::new().spacing(12).width(Length::Fill);
    if let Some(error) = &model.error {
        let mut problem = Column::new()
            .spacing(4)
            .push(strong("Error", 11.5, theme::DANGER))
            .push(label(error.message.clone(), 13.0, theme::TEXT));
        if let Some(data) = &error.data {
            problem = problem.push(text_block(data, true));
        }
        col = col.push(
            container(problem)
                .padding(10)
                .width(Length::Fill)
                .style(theme::notice(theme::DANGER)),
        );
    }
    for field in &model.overview {
        col = col.push(field_view(field));
    }
    col.into()
}

fn blocks(blocks: &[Block], empty: &'static str) -> Element<'static> {
    if blocks.is_empty() {
        return label(empty, 13.0, theme::TEXT_DIM).into();
    }
    let mut col = Column::new().spacing(14).width(Length::Fill);
    for block in blocks {
        let header = row![
            label(block.label.clone(), 11.0, theme::TEXT_FAINT),
            space::horizontal(),
            button(label("Copy", 11.0, theme::TEXT_DIM))
                .padding([2.0, 9.0])
                .style(theme::ghost_button)
                .on_press(Message::Copy(block.text.clone()))
        ]
        .align_y(Alignment::Center);
        col = col.push(column![header, text_block(&block.text, block.mono)].spacing(4));
    }
    col.into()
}

/// Long text in a well of its own that scrolls when it is taller than
/// [`BLOCK_MAX_H`].
fn text_block(text: &str, monospace: bool) -> Element<'static> {
    let content = if monospace {
        mono(text.to_string(), 12.0, theme::TEXT)
    } else {
        label(text.to_string(), 13.0, theme::TEXT)
    };
    container(
        scrollable(
            container(content)
                .padding(iced::padding::right(10))
                .width(Length::Fill),
        )
        .style(theme::scrollbars),
    )
    .padding(10)
    .width(Length::Fill)
    .max_height(BLOCK_MAX_H)
    .style(theme::well)
    .into()
}
