//! `Frames`: a transparent wrapper around the root element that counts the
//! `RedrawRequested` events the widget tree receives, for `LATTICE_PERF`.
//!
//! iced hands every widget a `window::Event::RedrawRequested` once per frame it
//! is about to draw (and once more when a redraw's own messages or layout change
//! make the window loop update the tree again before drawing). The number of
//! those events over ten seconds is the number of frames the window cost, which
//! `views=` and `rebuilds=` cannot show: a frame that only repaints cached
//! geometry builds no view and tessellates nothing.
//!
//! Invariants:
//! - The wrapper is invisible to iced. It forwards `size`, `size_hint`, `tag`,
//!   `state`, `children`, `diff`, `layout`, `draw`, `update`, `mouse_interaction`,
//!   `operate` and `overlay` to the wrapped element, and it shares the wrapped
//!   element's widget-state tree instead of adding a level to it, so wrapping
//!   changes neither the layout, the pixels, the events a widget sees, nor the
//!   state a widget keeps between views.
//! - It never publishes a message, never requests a redraw and holds no
//!   subscription: it only increments a counter (one relaxed atomic add, and
//!   only when `LATTICE_PERF=1`), so measuring frames cannot cause any.

use iced::advanced::layout::{self, Layout};
use iced::advanced::widget::{Operation, Tree, tree};
use iced::advanced::{Clipboard, Shell, Widget, mouse, overlay, renderer};
use iced::{Element, Event, Length, Rectangle, Size, Vector, window};

use crate::perf;

/// The root element, with its frames counted.
pub(super) struct Frames<'a, Message, Theme, Renderer> {
    content: Element<'a, Message, Theme, Renderer>,
}

impl<'a, Message, Theme, Renderer> Frames<'a, Message, Theme, Renderer> {
    pub(super) fn new(content: Element<'a, Message, Theme, Renderer>) -> Self {
        Self { content }
    }
}

impl<Message, Theme, Renderer> Widget<Message, Theme, Renderer>
    for Frames<'_, Message, Theme, Renderer>
where
    Renderer: renderer::Renderer,
{
    fn size(&self) -> Size<Length> {
        self.content.as_widget().size()
    }

    fn size_hint(&self) -> Size<Length> {
        self.content.as_widget().size_hint()
    }

    fn tag(&self) -> tree::Tag {
        self.content.as_widget().tag()
    }

    fn state(&self) -> tree::State {
        self.content.as_widget().state()
    }

    fn children(&self) -> Vec<Tree> {
        self.content.as_widget().children()
    }

    fn diff(&self, tree: &mut Tree) {
        self.content.as_widget().diff(tree);
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        self.content.as_widget_mut().layout(tree, renderer, limits)
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        style: &renderer::Style,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        self.content
            .as_widget()
            .draw(tree, renderer, theme, style, layout, cursor, viewport);
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        self.content
            .as_widget_mut()
            .operate(tree, layout, renderer, operation);
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        renderer: &Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        viewport: &Rectangle,
    ) {
        if let Event::Window(window::Event::RedrawRequested(_)) = event {
            perf::frame_drawn();
        }
        self.content.as_widget_mut().update(
            tree, event, layout, cursor, renderer, clipboard, shell, viewport,
        );
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        self.content
            .as_widget()
            .mouse_interaction(tree, layout, cursor, viewport, renderer)
    }

    fn overlay<'b>(
        &'b mut self,
        tree: &'b mut Tree,
        layout: Layout<'b>,
        renderer: &Renderer,
        viewport: &Rectangle,
        translation: Vector,
    ) -> Option<overlay::Element<'b, Message, Theme, Renderer>> {
        self.content
            .as_widget_mut()
            .overlay(tree, layout, renderer, viewport, translation)
    }
}

impl<'a, Message, Theme, Renderer> From<Frames<'a, Message, Theme, Renderer>>
    for Element<'a, Message, Theme, Renderer>
where
    Message: 'a,
    Theme: 'a,
    Renderer: renderer::Renderer + 'a,
{
    fn from(frames: Frames<'a, Message, Theme, Renderer>) -> Self {
        Element::new(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use iced::advanced::renderer::Headless;
    use iced::time::Instant;
    use iced::widget::{button, column, container, text};
    use iced::{Color, Font, Pixels, Point, event, keyboard};

    #[derive(Clone, Debug, PartialEq)]
    enum Msg {
        Pressed,
    }

    type Root<'a> = Element<'a, Msg, iced::Theme, iced::Renderer>;

    /// A small interactive tree: a text and a button that publishes `Pressed`.
    fn content<'a>() -> Root<'a> {
        container(
            column![
                text("Counting frames"),
                button(text("Press")).on_press(Msg::Pressed)
            ]
            .spacing(6),
        )
        .padding(12)
        .into()
    }

    fn renderer() -> iced::Renderer {
        block_on(<iced::Renderer as Headless>::new(
            Font::DEFAULT,
            Pixels(16.0),
            Some("tiny-skia"),
        ))
        .expect("the software renderer")
    }

    const SIZE: Size = Size::new(240.0, 120.0);

    /// One element, built, laid out and ready to be poked, either bare or wrapped.
    struct Rig<'a> {
        element: Root<'a>,
        tree: Tree,
        node: layout::Node,
        renderer: iced::Renderer,
        /// What the last event published, asked for and did.
        messages: Vec<Msg>,
        redraw: window::RedrawRequest,
        status: iced::event::Status,
    }

    impl Rig<'_> {
        fn new(wrapped: bool) -> Self {
            let mut element = if wrapped {
                Frames::new(content()).into()
            } else {
                content()
            };
            let mut tree = Tree::new(element.as_widget());
            let renderer = renderer();
            let node = element.as_widget_mut().layout(
                &mut tree,
                &renderer,
                &layout::Limits::new(Size::ZERO, SIZE),
            );
            Self {
                element,
                tree,
                node,
                renderer,
                messages: Vec::new(),
                redraw: window::RedrawRequest::Wait,
                status: event::Status::Ignored,
            }
        }

        fn send(&mut self, event: Event, cursor: mouse::Cursor) {
            self.messages.clear();
            let mut shell = Shell::new(&mut self.messages);
            self.element.as_widget_mut().update(
                &mut self.tree,
                &event,
                Layout::new(&self.node),
                cursor,
                &self.renderer,
                &mut iced::advanced::clipboard::Null,
                &mut shell,
                &Rectangle::with_size(SIZE),
            );
            self.redraw = shell.redraw_request();
            self.status = shell.event_status();
        }

        fn interaction(&self, cursor: mouse::Cursor) -> mouse::Interaction {
            self.element.as_widget().mouse_interaction(
                &self.tree,
                Layout::new(&self.node),
                cursor,
                &Rectangle::with_size(SIZE),
                &self.renderer,
            )
        }

        fn pixels(&mut self, cursor: mouse::Cursor) -> Vec<u8> {
            renderer::Renderer::reset(&mut self.renderer, Rectangle::with_size(SIZE));
            self.element.as_widget().draw(
                &self.tree,
                &mut self.renderer,
                &iced::Theme::Dark,
                &renderer::Style {
                    text_color: Color::WHITE,
                },
                Layout::new(&self.node),
                cursor,
                &Rectangle::with_size(SIZE),
            );
            self.renderer.screenshot(
                Size::new(SIZE.width as u32, SIZE.height as u32),
                1.0,
                Color::BLACK,
            )
        }
    }

    fn redraw_event() -> Event {
        Event::Window(window::Event::RedrawRequested(Instant::now()))
    }

    fn bounds_of(layout: Layout<'_>, out: &mut Vec<Rectangle>) {
        out.push(layout.bounds());
        for child in layout.children() {
            bounds_of(child, out);
        }
    }

    /// The button's centre, found from the layout (the last widget in it).
    fn on_button(rig: &Rig<'_>) -> Point {
        let mut all = Vec::new();
        bounds_of(Layout::new(&rig.node), &mut all);
        all.iter()
            .rev()
            .find(|b| b.width > 20.0 && b.height > 10.0 && b.height < 60.0)
            .map(|b| b.center())
            .expect("the button")
    }

    #[test]
    fn wrapping_changes_neither_the_layout_nor_the_tree_nor_the_pixels() {
        let (mut bare, mut wrapped) = (Rig::new(false), Rig::new(true));
        let (mut a, mut b) = (Vec::new(), Vec::new());
        bounds_of(Layout::new(&bare.node), &mut a);
        bounds_of(Layout::new(&wrapped.node), &mut b);
        assert_eq!(a, b, "the same layout, widget by widget");
        assert_eq!(
            wrapped.element.as_widget().size(),
            bare.element.as_widget().size()
        );
        assert_eq!(
            wrapped.element.as_widget().size_hint(),
            bare.element.as_widget().size_hint()
        );
        assert_eq!(
            wrapped.tree.tag, bare.tree.tag,
            "the wrapper is not a level of its own in the widget-state tree"
        );
        assert_eq!(wrapped.tree.children.len(), bare.tree.children.len());
        // A frame later, with the pointer on the button and off it.
        let over = on_button(&wrapped);
        for cursor in [
            mouse::Cursor::Unavailable,
            mouse::Cursor::Available(over),
            mouse::Cursor::Available(Point::new(1.0, 1.0)),
        ] {
            bare.send(redraw_event(), cursor);
            wrapped.send(redraw_event(), cursor);
            assert_eq!(
                bare.pixels(cursor),
                wrapped.pixels(cursor),
                "the same pixels for {cursor:?}"
            );
        }
    }

    #[test]
    fn events_reach_the_content_and_come_back_unchanged() {
        let (mut bare, mut wrapped) = (Rig::new(false), Rig::new(true));
        let over = on_button(&wrapped);
        let cursor = mouse::Cursor::Available(over);
        // The first frame records the button's status, as the window's does.
        bare.send(redraw_event(), mouse::Cursor::Unavailable);
        wrapped.send(redraw_event(), mouse::Cursor::Unavailable);
        let script = [
            Event::Mouse(mouse::Event::CursorMoved { position: over }),
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)),
            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)),
            Event::Keyboard(keyboard::Event::ModifiersChanged(
                keyboard::Modifiers::empty(),
            )),
            Event::Mouse(mouse::Event::CursorMoved {
                position: Point::new(1.0, 1.0),
            }),
        ];
        let mut published = 0;
        let mut redraws = 0;
        for event in script {
            bare.send(event.clone(), cursor);
            wrapped.send(event, cursor);
            assert_eq!(bare.messages, wrapped.messages);
            assert_eq!(bare.redraw, wrapped.redraw);
            assert_eq!(bare.status, wrapped.status);
            assert_eq!(bare.interaction(cursor), wrapped.interaction(cursor));
            published += wrapped.messages.len();
            redraws += usize::from(wrapped.redraw == window::RedrawRequest::NextFrame);
        }
        assert_eq!(published, 1, "the press was published, once, through it");
        assert!(
            redraws >= 1,
            "the hover change reached the shell: {redraws}"
        );
        assert_eq!(
            wrapped.interaction(cursor),
            mouse::Interaction::Pointer,
            "the button's cursor comes through"
        );
        assert_eq!(
            wrapped.interaction(mouse::Cursor::Unavailable),
            mouse::Interaction::None
        );
    }

    #[test]
    fn only_redraw_requests_are_counted_and_counting_asks_for_nothing() {
        let mut wrapped = Rig::new(true);
        let _ = perf::take_local_frames();
        let cursor = mouse::Cursor::Unavailable;
        for _ in 0..3 {
            wrapped.send(redraw_event(), cursor);
            // Counting is not a reason to draw again, or to say anything.
            assert_eq!(wrapped.redraw, window::RedrawRequest::Wait);
            assert!(wrapped.messages.is_empty());
            assert_eq!(wrapped.status, event::Status::Ignored);
        }
        for event in [
            Event::Mouse(mouse::Event::CursorMoved {
                position: Point::new(3.0, 3.0),
            }),
            Event::Mouse(mouse::Event::CursorLeft),
            Event::Keyboard(keyboard::Event::ModifiersChanged(
                keyboard::Modifiers::empty(),
            )),
            Event::Window(window::Event::Focused),
            Event::Window(window::Event::Unfocused),
        ] {
            wrapped.send(event, cursor);
            assert_eq!(wrapped.redraw, window::RedrawRequest::Wait);
        }
        assert_eq!(
            perf::take_local_frames(),
            3,
            "three RedrawRequested, and nothing else counted"
        );
    }

    #[test]
    fn the_windows_root_is_wrapped_so_its_frames_are_counted() {
        // The real tree, through the harness: the counter sits at its root.
        let mut h = crate::testkit::Harness::new(
            std::sync::Arc::new(crate::demo::DemoService::instant()),
            crate::app::Options {
                select_latest: true,
                ..crate::app::Options::default()
            },
            Size::new(1440.0, 900.0),
        );
        h.redraw();
        let _ = perf::take_local_frames();
        h.draw();
        assert_eq!(perf::take_local_frames(), 1, "one draw, one frame");
        h.draw();
        h.draw();
        assert_eq!(perf::take_local_frames(), 2);
    }
}
