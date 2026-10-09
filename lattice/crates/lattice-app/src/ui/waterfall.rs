//! The waterfall: one canvas holding the tree column (chevron, kind chip, title,
//! duration) and the bar track on one shared time axis.
//!
//! Compute rules kept here:
//! - The big layer (axis, gridlines, visible rows, scrollbar) is drawn through
//!   `Cache::draw`, so it is tessellated only after `update` clears the cache:
//!   new or ended span, selection, collapse, axis change (the tick), scroll, or
//!   a size change. Each rebuild is counted for `LATTICE_PERF`.
//! - Hover is not in that layer. It lives in this canvas's own state and is drawn
//!   as a small uncached rectangle above it, and moving the mouse asks only for
//!   a redraw, never for a `view()`.
//! - Only the rows that intersect the viewport are drawn (see
//!   `wfmodel::visible_range`); scrolling, clicks, the scrollbar and the wheel
//!   are handled here and reported as messages.

use iced::widget::canvas::{self, Action, Cache, Event, Frame, Geometry, Path, Stroke, Text};
use iced::widget::text::Alignment as TextAlign;
use iced::{Color, Pixels, Point, Rectangle, Size, alignment, mouse, window};

use crate::app::Message;
use crate::axis::{self, Axis, MAX_TICKS, nice_ticks};
use crate::clock::{format_axis_label, format_duration};
use crate::perf;
use crate::textmetrics::{mono_width, truncate_to_width};
use crate::theme::{self, fonts};
use crate::wfmodel::{self, HEADER_H, Metrics, ROW_H, SCROLLBAR_W, WfRow};

/// Pixels per wheel notch: what iced's own scrollable uses.
const WHEEL_LINE: f32 = 60.0;
/// The narrowest a bar is drawn.
const MIN_BAR: f32 = 3.0;
const BAR_H: f32 = 12.0;

pub struct Waterfall<'a> {
    pub rows: &'a [WfRow],
    pub axis: Axis,
    pub now: f64,
    /// The selected row, as an index into `rows`.
    pub selected: Option<usize>,
    pub scroll: f32,
    /// The size the application last recorded for this canvas.
    pub known_size: Size,
    pub cache: &'a Cache,
}

/// The canvas's own state: hover and scrollbar drag, which never need a message.
#[derive(Default)]
pub struct WaterfallState {
    hover: Option<usize>,
    /// While dragging the scrollbar thumb: where inside the thumb it was grabbed.
    grab: Option<f32>,
}

fn text_at(
    content: impl Into<String>,
    at: Point,
    size: f32,
    color: Color,
    font: iced::Font,
    align: TextAlign,
) -> Text {
    Text {
        content: content.into(),
        position: at,
        color,
        size: Pixels(size),
        font,
        align_x: align,
        align_y: alignment::Vertical::Center,
        ..Text::default()
    }
}

fn rounded(x: f32, y: f32, w: f32, h: f32, radius: f32) -> Path {
    Path::rounded_rectangle(
        Point::new(x, y),
        Size::new(w, h),
        radius.min(w / 2.0).min(h / 2.0).into(),
    )
}

impl Waterfall<'_> {
    fn paint(&self, frame: &mut Frame, size: Size) {
        let f = fonts();
        let m = Metrics::new(size.width);
        let span = self.axis.span();

        // The axis: a gridline at every nice tick, and a label where it fits: the
        // number of ticks follows the track's width, and a label that would touch
        // the previous one is left out.
        let max_ticks = ((m.track_w / 84.0) as usize).clamp(2, MAX_TICKS);
        let (_, ticks) = nice_ticks(span, max_ticks);
        let mut previous_right = f32::NEG_INFINITY;
        for tick in &ticks {
            let x = m.track_x + (tick / span) as f32 * m.track_w;
            frame.stroke(
                &Path::line(Point::new(x, HEADER_H - 6.0), Point::new(x, size.height)),
                Stroke::default().with_color(theme::LINE).with_width(1.0),
            );
            let label = format_axis_label(*tick);
            let half = mono_width(&label, 11.0) / 2.0;
            let cx = x.clamp(
                m.track_x - 4.0 + half,
                (size.width - SCROLLBAR_W - 2.0 - half).max(m.track_x),
            );
            if cx - half >= previous_right + 8.0 {
                previous_right = cx + half;
                frame.fill_text(text_at(
                    label,
                    Point::new(cx, HEADER_H / 2.0 - 3.0),
                    11.0,
                    theme::TEXT_FAINT,
                    f.mono,
                    TextAlign::Center,
                ));
            }
        }
        frame.fill_text(text_at(
            "Spans",
            Point::new(12.0, HEADER_H / 2.0 - 3.0),
            11.0,
            theme::TEXT_FAINT,
            f.ui_strong,
            TextAlign::Left,
        ));
        frame.stroke(
            &Path::line(Point::new(0.0, HEADER_H), Point::new(size.width, HEADER_H)),
            Stroke::default().with_color(theme::LINE).with_width(1.0),
        );
        frame.stroke(
            &Path::line(
                Point::new(m.tree_w + 4.0, HEADER_H),
                Point::new(m.tree_w + 4.0, size.height),
            ),
            Stroke::default().with_color(theme::LINE).with_width(1.0),
        );

        // Only the rows that intersect the viewport, clipped under the header.
        let area = Rectangle::new(
            Point::new(0.0, HEADER_H),
            Size::new(size.width, (size.height - HEADER_H).max(0.0)),
        );
        let visible = wfmodel::visible_range(self.scroll, size.height, self.rows.len());
        frame.with_clip(area, |frame| {
            for index in visible {
                let y = HEADER_H + index as f32 * ROW_H - self.scroll;
                self.paint_row(frame, &self.rows[index], index, y, &m, size.width);
            }
        });

        if let Some((top, height)) = wfmodel::thumb(self.scroll, self.rows.len(), size.height) {
            frame.fill(
                &rounded(size.width - SCROLLBAR_W + 3.0, top, 4.0, height, 2.0),
                theme::TEXT_FAINT,
            );
        }
    }

    fn paint_row(
        &self,
        frame: &mut Frame,
        row: &WfRow,
        index: usize,
        y: f32,
        m: &Metrics,
        width: f32,
    ) {
        let f = fonts();
        let cy = y + ROW_H / 2.0;
        let color = theme::kind_color(row.kind, row.triggered);

        if self.selected == Some(index) {
            frame.fill_rectangle(
                Point::new(0.0, y),
                Size::new((width - SCROLLBAR_W).max(0.0), ROW_H),
                theme::GOLD_WASH,
            );
            frame.fill_rectangle(Point::new(0.0, y), Size::new(2.0, ROW_H), theme::GOLD);
        }

        // Tree column: chevron, kind chip, title, duration.
        let cx = wfmodel::chevron_x(row.depth);
        if row.has_children {
            let triangle = Path::new(|b| {
                if row.collapsed {
                    b.move_to(Point::new(cx - 2.5, cy - 4.5));
                    b.line_to(Point::new(cx + 3.0, cy));
                    b.line_to(Point::new(cx - 2.5, cy + 4.5));
                } else {
                    b.move_to(Point::new(cx - 4.5, cy - 2.5));
                    b.line_to(Point::new(cx + 4.5, cy - 2.5));
                    b.line_to(Point::new(cx, cy + 3.0));
                }
                b.close();
            });
            frame.fill(&triangle, theme::TEXT_DIM);
        }
        let chip_x = cx + 11.0;
        frame.fill(
            &rounded(chip_x, cy - 8.0, 16.0, 16.0, 4.0),
            theme::with_alpha(color, 0.20),
        );
        frame.fill_text(text_at(
            row.kind.glyph(),
            Point::new(chip_x + 8.0, cy),
            10.0,
            color,
            f.ui_strong,
            TextAlign::Center,
        ));

        let title_x = chip_x + 16.0 + 8.0;
        let now = self.now;
        let end = row.end.unwrap_or_else(|| now.max(row.start));
        let duration = format_duration(end - row.start);
        let duration_w = mono_width(&duration, 11.0);
        let title_w = m.tree_w - title_x - duration_w - 14.0;
        if title_w > 24.0 {
            let title = truncate_to_width(&row.title, 12.5, title_w).into_owned();
            let title_color = if row.error {
                theme::DANGER
            } else {
                theme::TEXT
            };
            frame.fill_text(text_at(
                title,
                Point::new(title_x, cy),
                12.5,
                title_color,
                f.ui,
                TextAlign::Left,
            ));
        }
        frame.fill_text(text_at(
            duration,
            Point::new(m.tree_w - 6.0, cy),
            11.0,
            theme::TEXT_DIM,
            f.mono,
            TextAlign::Right,
        ));

        // Bar track: one bar per row on the shared axis; an open span runs to "now"
        // and fades at its growing edge.
        let bar = axis::bar(
            row.start, row.end, now, &self.axis, m.track_x, m.track_w, MIN_BAR,
        );
        let open = row.end.is_none();
        let body = rounded(bar.x, cy - BAR_H / 2.0, bar.w, BAR_H, 3.0);
        if open {
            frame.fill(&body, theme::with_alpha(color, 0.35));
            let solid = (bar.w - 14.0).max(0.0);
            if solid > 0.0 {
                frame.fill(
                    &rounded(bar.x, cy - BAR_H / 2.0, solid, BAR_H, 3.0),
                    theme::with_alpha(color, 0.95),
                );
            }
        } else {
            frame.fill(&body, theme::with_alpha(color, 0.92));
        }
        if row.error {
            frame.stroke(
                &body,
                Stroke::default().with_color(theme::DANGER).with_width(1.5),
            );
            let track_end = m.track_x + m.track_w;
            let marker_x = if bar.x + bar.w + 10.0 <= track_end {
                bar.x + bar.w + 9.0
            } else {
                (bar.x - 9.0).max(m.track_x)
            };
            frame.fill(&Path::circle(Point::new(marker_x, cy), 4.5), theme::DANGER);
            frame.fill_text(text_at(
                "!",
                Point::new(marker_x, cy),
                8.0,
                theme::ON_GOLD,
                f.ui_strong,
                TextAlign::Center,
            ));
        }
    }
}

impl canvas::Program<Message> for Waterfall<'_> {
    type State = WaterfallState;

    fn update(
        &self,
        state: &mut WaterfallState,
        event: &Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<Action<Message>> {
        let size = bounds.size();
        let rows = self.rows.len();
        // Report the canvas's size once it differs from what the application knows:
        // the application needs it to scroll a selected row into view.
        if let Event::Window(window::Event::RedrawRequested(_)) = event {
            let changed = (size.width - self.known_size.width).abs() > 0.5
                || (size.height - self.known_size.height).abs() > 0.5;
            return changed.then(|| Action::publish(Message::WaterfallSize(size)));
        }
        let Event::Mouse(event) = event else {
            return None;
        };
        let local = cursor.position_in(bounds);
        let gutter = size.width - SCROLLBAR_W - 2.0;
        match event {
            mouse::Event::CursorMoved { .. } => {
                if let Some(grab) = state.grab {
                    let y = cursor.position().map(|p| p.y - bounds.y)?;
                    let scroll = wfmodel::scroll_from_thumb_top(y - grab, rows, size.height);
                    return Some(Action::publish(Message::Scrolled(scroll)).and_capture());
                }
                let hover = local
                    .filter(|p| p.x < gutter)
                    .and_then(|p| wfmodel::row_at(p.y, self.scroll, rows));
                if hover != state.hover {
                    state.hover = hover;
                    return Some(Action::request_redraw());
                }
                None
            }
            mouse::Event::CursorLeft => state.hover.take().map(|_| Action::request_redraw()),
            mouse::Event::ButtonPressed(mouse::Button::Left) => {
                let p = local?;
                if p.x >= gutter {
                    let (top, height) = wfmodel::thumb(self.scroll, rows, size.height)?;
                    if p.y >= top && p.y <= top + height {
                        state.grab = Some(p.y - top);
                        return Some(Action::capture());
                    }
                    // Beside the thumb: page towards the click.
                    let page = wfmodel::viewport_height(size.height);
                    let target = if p.y < top {
                        self.scroll - page
                    } else {
                        self.scroll + page
                    };
                    return Some(Action::publish(Message::Scrolled(target)).and_capture());
                }
                let index = wfmodel::row_at(p.y, self.scroll, rows)?;
                let row = &self.rows[index];
                let message = if row.has_children && wfmodel::on_chevron(p.x, row.depth) {
                    Message::ToggleCollapse(row.span_id.clone())
                } else {
                    Message::SelectSpan(Some(row.span_id.clone()))
                };
                Some(Action::publish(message).and_capture())
            }
            mouse::Event::ButtonReleased(mouse::Button::Left) => {
                state.grab.take().map(|_| Action::capture())
            }
            mouse::Event::WheelScrolled { delta } => {
                local?;
                let dy = match delta {
                    mouse::ScrollDelta::Lines { y, .. } => y * WHEEL_LINE,
                    mouse::ScrollDelta::Pixels { y, .. } => *y,
                };
                let target = wfmodel::clamp_scroll(self.scroll - dy, rows, size.height);
                ((target - self.scroll).abs() > f32::EPSILON)
                    .then(|| Action::publish(Message::Scrolled(target)).and_capture())
            }
            _ => None,
        }
    }

    fn draw(
        &self,
        state: &WaterfallState,
        renderer: &iced::Renderer,
        _theme: &iced::Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry> {
        let size = bounds.size();
        let big = self.cache.draw(renderer, size, |frame| {
            perf::cache_rebuilt();
            self.paint(frame, size);
        });
        let mut layers = vec![big];
        if let Some(index) = state.hover.filter(|&i| i < self.rows.len()) {
            let y = HEADER_H + index as f32 * ROW_H - self.scroll;
            if y + ROW_H > HEADER_H && y < size.height {
                let mut frame = Frame::new(renderer, size);
                let top = y.max(HEADER_H);
                let bottom = (y + ROW_H).min(size.height);
                frame.fill_rectangle(
                    Point::new(0.0, top),
                    Size::new((size.width - SCROLLBAR_W).max(0.0), (bottom - top).max(0.0)),
                    Color {
                        a: 0.05,
                        ..theme::TEXT
                    },
                );
                layers.push(frame.into_geometry());
            }
        }
        layers
    }

    fn mouse_interaction(
        &self,
        state: &WaterfallState,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        if state.grab.is_some() {
            return mouse::Interaction::Grabbing;
        }
        match (state.hover, cursor.position_in(bounds)) {
            (Some(_), Some(_)) => mouse::Interaction::Pointer,
            _ => mouse::Interaction::default(),
        }
    }
}
