//! The Graph view: the run's agent's static graph, drawn from the layered
//! layout in `graph.rs`.
//!
//! Agents that have an agent span in this trace are drawn at full strength and
//! the rest dimmed; tool edges are dotted, hand-off edges gold with arrowheads,
//! and the start and end are rounded pills. The geometry is cached: it is
//! rebuilt only when the run changes or an agent starts running.

use iced::widget::canvas::{self, Cache, Frame, Geometry, LineDash, Path, Stroke, Text};
use iced::widget::text::Alignment as TextAlign;
use iced::widget::{Column, canvas as canvas_widget, column, container, responsive, scrollable};
use iced::{Color, Length, Pixels, Point, Rectangle, Size, alignment, mouse};

use super::{Element, label};
use crate::app::{App, Message, Selected};
use crate::graph::{self, AGENT_TEXT, EdgeLine, GraphLayout, NodeBox, TOOL_TEXT};
use crate::perf;
use crate::textmetrics::truncate_to_width;
use crate::theme::{self, fonts};
use lattice_protocol::{EdgeKind, NodeKind};

/// True when a graph `graph` wide and high needs no scrolling in `available`
/// space: the picture is drawn as it is, with no scrollbar beside it.
pub(super) fn fits(graph: Size, available: Size) -> bool {
    graph.width <= available.width + 0.5 && graph.height <= available.height + 0.5
}

pub(super) fn view<'a>(app: &'a App, sel: &'a Selected) -> Element<'a> {
    let Some(layout) = &sel.graph else {
        return container(label(
            "This run's agent has no graph on record.",
            13.0,
            theme::TEXT_DIM,
        ))
        .center(Length::Fill)
        .style(theme::panel)
        .into();
    };
    let active = graph::node_activity(layout, &sel.active_agents);
    let cache = &app.graph_cache;
    // A graph that fits its panel is drawn as it is. Only one that does not gets a
    // scrollable, and that one fills the panel, so its scrollbars sit at the
    // panel's edge and not beside the picture.
    let drawing = responsive(move |available| {
        let program = GraphView {
            layout,
            active: active.clone(),
            cache,
        };
        let canvas = canvas_widget(program)
            .width(Length::Fixed(layout.width))
            .height(Length::Fixed(layout.height));
        if fits(Size::new(layout.width, layout.height), available) {
            container(canvas)
                .center_x(Length::Fill)
                .height(Length::Fill)
                .into()
        } else {
            scrollable(container(canvas).center_x(Length::Fixed(layout.width.max(available.width))))
                .direction(scrollable::Direction::Both {
                    vertical: scrollable::Scrollbar::new(),
                    horizontal: scrollable::Scrollbar::new(),
                })
                .style(theme::scrollbars)
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        }
    });
    let legend = label(
        "Bright agents ran in this trace; dimmed ones did not. Dotted lines are tools; gold arrows are hand-offs.",
        11.5,
        theme::TEXT_FAINT,
    );
    let body: Column<'a, Message> = column![
        container(drawing).width(Length::Fill).height(Length::Fill),
        legend
    ]
    .spacing(8);
    container(body)
        .padding(12)
        .width(Length::Fill)
        .height(Length::Fill)
        .style(theme::panel)
        .into()
}

struct GraphView<'a> {
    layout: &'a GraphLayout,
    /// Per node: drawn at full strength.
    active: Vec<bool>,
    cache: &'a Cache,
}

fn text_centered(content: String, at: Point, size: f32, color: Color, font: iced::Font) -> Text {
    Text {
        content,
        position: at,
        color,
        size: Pixels(size),
        font,
        align_x: TextAlign::Center,
        align_y: alignment::Vertical::Center,
        ..Text::default()
    }
}

fn dimmed(color: Color, on: bool) -> Color {
    theme::with_alpha(color, if on { color.a } else { color.a * 0.38 })
}

impl GraphView<'_> {
    fn paint(&self, frame: &mut Frame) {
        // Edges first, so nodes sit on top of them.
        for edge in &self.layout.edges {
            self.paint_edge(frame, edge, graph::edge_active(&self.active, edge));
        }
        for (index, node) in self.layout.nodes.iter().enumerate() {
            self.paint_node(frame, node, self.active[index]);
        }
    }

    fn paint_node(&self, frame: &mut Frame, node: &NodeBox, on: bool) {
        let f = fonts();
        let r = node.rect;
        let (fill, edge, text, size, font, radius) = match node.kind {
            NodeKind::Agent => (
                theme::RAISED,
                theme::GOLD,
                theme::TEXT,
                AGENT_TEXT,
                f.ui_strong,
                10.0,
            ),
            NodeKind::Tool | NodeKind::Mcp => (
                theme::SURFACE,
                theme::POSITIVE,
                theme::TEXT,
                TOOL_TEXT,
                f.mono,
                7.0,
            ),
            NodeKind::Start | NodeKind::End => (
                theme::SURFACE,
                theme::TEXT_FAINT,
                theme::TEXT_DIM,
                11.5,
                f.ui,
                r.h / 2.0,
            ),
        };
        let path =
            Path::rounded_rectangle(Point::new(r.x, r.y), Size::new(r.w, r.h), radius.into());
        frame.fill(&path, dimmed(fill, on));
        frame.stroke(
            &path,
            Stroke::default().with_color(dimmed(edge, on)).with_width(
                if node.kind == NodeKind::Agent {
                    1.6
                } else {
                    1.2
                },
            ),
        );
        let shown = truncate_to_width(&node.label, size, r.w - 16.0).into_owned();
        frame.fill_text(text_centered(
            shown,
            Point::new(r.x + r.w / 2.0, r.y + r.h / 2.0),
            size,
            dimmed(text, on),
            font,
        ));
    }

    fn paint_edge(&self, frame: &mut Frame, edge: &EdgeLine, on: bool) {
        let (from, to) = (
            Point::new(edge.from.x, edge.from.y),
            Point::new(edge.to.x, edge.to.y),
        );
        let (color, width, dotted) = match edge.kind {
            EdgeKind::Handoff => (theme::GOLD, 1.8, false),
            EdgeKind::Tool | EdgeKind::Mcp => (theme::POSITIVE, 1.4, true),
            EdgeKind::Start | EdgeKind::End => (theme::TEXT_FAINT, 1.3, false),
        };
        let color = dimmed(color, on);
        let curve = Path::new(|b| {
            b.move_to(from);
            if edge.back {
                // Out to the left, up and round, into the target's left side.
                let reach = 70.0;
                b.bezier_curve_to(
                    Point::new(from.x - reach, from.y),
                    Point::new(to.x - reach, to.y),
                    to,
                );
            } else if matches!(edge.kind, EdgeKind::Tool | EdgeKind::Mcp) {
                let dx = (to.x - from.x) / 2.0;
                b.bezier_curve_to(
                    Point::new(from.x + dx, from.y),
                    Point::new(to.x - dx, to.y),
                    to,
                );
            } else {
                let dy = (to.y - from.y) / 2.0;
                b.bezier_curve_to(
                    Point::new(from.x, from.y + dy),
                    Point::new(to.x, to.y - dy),
                    to,
                );
            }
        });
        let dash = [2.0f32, 5.0];
        let mut stroke = Stroke::default().with_color(color).with_width(width);
        if dotted {
            stroke.line_dash = LineDash {
                segments: &dash,
                offset: 0,
            };
            stroke.line_cap = canvas::LineCap::Round;
        }
        frame.stroke(&curve, stroke);
        if !dotted {
            // The arrowhead points along the last stretch of the curve.
            let (dx, dy) = if edge.back { (1.0, 0.0) } else { (0.0, 1.0) };
            let head = Path::new(|b| {
                b.move_to(to);
                b.line_to(Point::new(
                    to.x - dx * 9.0 - dy * 5.0,
                    to.y - dy * 9.0 - dx * 5.0,
                ));
                b.line_to(Point::new(
                    to.x - dx * 9.0 + dy * 5.0,
                    to.y - dy * 9.0 + dx * 5.0,
                ));
                b.close();
            });
            frame.fill(&head, color);
        }
    }
}

impl canvas::Program<Message> for GraphView<'_> {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &iced::Renderer,
        _theme: &iced::Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry> {
        let geometry = self.cache.draw(renderer, bounds.size(), |frame| {
            perf::cache_rebuilt();
            self.paint(frame);
        });
        vec![geometry]
    }
}
