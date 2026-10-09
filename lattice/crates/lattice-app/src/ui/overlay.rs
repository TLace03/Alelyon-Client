//! The "New run" overlay: task, agent, model, the disclosure of where the run
//! goes (shown before Start), and Start.
//!
//! The overlay covers the window with a scrim that swallows clicks (`opaque`),
//! so nothing behind it can be pressed; Escape closes it (see `App::on_key`).

use iced::widget::{
    Column, Row, button, column, container, mouse_area, opaque, pick_list, row, space, text_editor,
};
use iced::{Alignment, Length};

use super::{Element, label, strong};
use crate::app::{App, Message};
use crate::newrun::{NewRun, TASK_INPUT_ID};
use crate::theme;
use lattice_protocol::Locality;

const CARD_W: f32 = 580.0;

pub(super) fn view<'a>(_app: &'a App, new_run: &'a NewRun) -> Element<'a> {
    let editor = text_editor(&new_run.task)
        .id(TASK_INPUT_ID)
        .placeholder("Describe the task in your own words")
        .on_action(Message::TaskEdited)
        .height(130)
        .padding(10)
        .size(14)
        .font(theme::fonts().ui)
        .style(theme::editor);

    let agents = pick_list(
        new_run.agents.clone(),
        new_run.agent.clone(),
        Message::PickAgent,
    )
    .width(Length::Fill)
    .padding(9)
    .text_size(13.5)
    .font(theme::fonts().ui)
    .style(theme::picker)
    .menu_style(theme::picker_menu);
    let models = pick_list(
        new_run.models.clone(),
        new_run.model.clone(),
        Message::PickModel,
    )
    .placeholder("No model is ready")
    .width(Length::Fill)
    .padding(9)
    .text_size(13.5)
    .font(theme::fonts().ui)
    .style(theme::picker)
    .menu_style(theme::picker_menu);

    let field = |name: &'static str, control: Element<'a>| -> Element<'a> {
        column![label(name, 11.5, theme::TEXT_DIM), control]
            .spacing(5)
            .into()
    };

    // Where the run goes, before it starts.
    let disclosure: Element<'a> = match (&new_run.model, new_run.disclosure()) {
        (Some(model), Some(line)) => {
            let color = if model.locality == Locality::Remote {
                theme::CAUTION
            } else {
                theme::POSITIVE
            };
            container(label(line, 13.0, theme::TEXT))
                .padding(10)
                .width(Length::Fill)
                .style(theme::notice(color))
                .into()
        }
        _ => container(label(
            "Choose a model that is ready. None is available right now.",
            13.0,
            theme::TEXT_DIM,
        ))
        .padding(10)
        .width(Length::Fill)
        .style(theme::notice(theme::TEXT_FAINT))
        .into(),
    };

    let mut card = Column::new()
        .spacing(14)
        .push(strong("New run", 18.0, theme::TEXT))
        .push(field("Task", editor.into()))
        .push(row![field("Agent", agents.into()), field("Model", models.into())].spacing(12))
        .push(disclosure);
    if let Some(notice) = &new_run.notice {
        card = card.push(
            container(label(notice.as_str(), 13.0, theme::TEXT))
                .padding(10)
                .width(Length::Fill)
                .style(theme::notice(theme::DANGER)),
        );
    }
    let mut start = button(strong(
        "Start",
        13.5,
        if new_run.can_start() {
            theme::ON_GOLD
        } else {
            theme::TEXT_FAINT
        },
    ))
    .padding([9.0, 22.0])
    .style(theme::primary_button);
    if new_run.can_start() {
        start = start.on_press(Message::Start);
    }
    let cancel = button(label("Cancel", 13.5, theme::TEXT_DIM))
        .padding([9.0, 16.0])
        .style(theme::ghost_button)
        .on_press(Message::CloseNewRun);
    card = card.push(
        Row::new()
            .spacing(8)
            .align_y(Alignment::Center)
            .push(label("Esc closes", 11.5, theme::TEXT_FAINT))
            .push(space::horizontal())
            .push(cancel)
            .push(start),
    );

    let card = container(card).padding(22).width(CARD_W).style(theme::card);
    // The scrim swallows clicks; a click on it (outside the card) closes the overlay.
    let scrim = mouse_area(
        container(opaque(card))
            .center(Length::Fill)
            .style(theme::scrim),
    )
    .on_press(Message::CloseNewRun);
    opaque(scrim)
}
