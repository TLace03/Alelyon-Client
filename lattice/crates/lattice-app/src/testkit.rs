//! A headless harness that drives the real widget tree, for tests.
//!
//! It builds `ui::view(&app)`, lays it out, feeds it input events and draws it
//! with iced's software renderer (tiny-skia) into pixels. There is no window,
//! no compositor and no GPU: the renderer is asked for the `"tiny-skia"` backend
//! by name, and iced's fallback renderer only tries the GPU one when no backend
//! is named.
//!
//! What it lets the tests prove: which messages an input produces (a click on a
//! row, a wheel scroll), that hovering asks for a redraw but publishes no
//! message and does not rebuild the big cache, that a message costs exactly one
//! `view()` call, and what the window looks like (`save_png`).
//!
//! Two ways to drive the tree. `Harness::send` builds it afresh for every event:
//! simple, and enough to test what an input publishes. `Harness::live` and
//! `Harness::pump` do what iced's window loop does: build it once, hand it many
//! events, keep the widgets' per-instance state between them, and rebuild only
//! after a message. That is the one to measure with (`perftests.rs`): a widget
//! that remembers its last status (a button, a scrollable, a canvas) can only ask
//! for a frame when it sees successive events on the same instance.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use futures::executor::block_on;
use iced::advanced::Shell;
use iced::advanced::clipboard;
use iced::advanced::layout::{self, Layout};
use iced::advanced::renderer::{self, Headless};
use iced::advanced::widget::Tree;
use iced::time::Instant;
use iced::{Color, Event, Font, Pixels, Point, Rectangle, Size, event, mouse, window};
use lattice_protocol::RunService;

use crate::app::{App, Message, Options};
use crate::ui;

pub struct Harness {
    pub app: App,
    renderer: iced::Renderer,
    tree: Option<Tree>,
    pub size: Size,
    pub cursor: mouse::Cursor,
    /// Messages the interface produced and the harness has applied, in order.
    pub log: Vec<Message>,
}

/// What one input event did.
pub struct Sent {
    pub messages: Vec<Message>,
    pub redraw: window::RedrawRequest,
    pub status: event::Status,
}

impl Harness {
    pub fn new(service: Arc<dyn RunService>, options: Options, size: Size) -> Self {
        let renderer = block_on(<iced::Renderer as Headless>::new(
            Font::DEFAULT,
            Pixels(16.0),
            Some("tiny-skia"),
        ))
        .expect("iced's software renderer");
        let (app, _boot_task) = App::boot(service, options);
        Self {
            app,
            renderer,
            tree: None,
            size,
            cursor: mouse::Cursor::Unavailable,
            log: Vec::new(),
        }
    }

    /// Build the current widget tree, lay it out, and hand it to `f`.
    fn with_ui<R>(
        &mut self,
        f: impl FnOnce(&mut crate::ui::Element<'_>, &mut Tree, &mut iced::Renderer, &layout::Node) -> R,
    ) -> R {
        let Self {
            app,
            renderer,
            tree,
            size,
            ..
        } = self;
        let mut element = ui::view(app);
        match tree {
            Some(tree) => tree.diff(element.as_widget()),
            None => *tree = Some(Tree::new(element.as_widget())),
        }
        let tree = tree.as_mut().expect("the tree was just made");
        let limits = layout::Limits::new(Size::ZERO, *size);
        let node = element.as_widget_mut().layout(tree, renderer, &limits);
        f(&mut element, tree, renderer, &node)
    }

    /// The bounds of the widget that is exactly `size` big, found by walking the
    /// layout (the waterfall canvas, whose size the application records).
    pub fn canvas_at(&mut self, size: Size) -> Option<Rectangle> {
        fn find(layout: Layout<'_>, size: Size) -> Option<Rectangle> {
            let b = layout.bounds();
            if (b.width - size.width).abs() < 0.5 && (b.height - size.height).abs() < 0.5 {
                return Some(b);
            }
            layout.children().find_map(|child| find(child, size))
        }
        self.with_ui(|_, _, _, node| find(Layout::new(node), size))
    }

    /// Deliver one input event to the widget tree.
    pub fn send(&mut self, event: Event) -> Sent {
        let (cursor, size) = (self.cursor, self.size);
        self.with_ui(|element, tree, renderer, node| {
            let mut messages = Vec::new();
            let mut shell = Shell::new(&mut messages);
            element.as_widget_mut().update(
                tree,
                &event,
                Layout::new(node),
                cursor,
                renderer,
                &mut clipboard::Null,
                &mut shell,
                &Rectangle::with_size(size),
            );
            let (redraw, status) = (shell.redraw_request(), shell.event_status());
            Sent {
                messages,
                redraw,
                status,
            }
        })
    }

    /// Apply messages to the application, as iced does after events: one
    /// `update` per message, then (in the real loop) one `view()`.
    pub fn apply(&mut self, messages: Vec<Message>) {
        for message in messages {
            self.log.push(message.clone());
            let _task = self.app.update(message);
        }
    }

    /// One redraw as the windowing loop performs it: the widgets see
    /// `RedrawRequested` (the canvas reports its size then), their messages are
    /// applied, and the loop repeats until nothing more is published.
    pub fn redraw(&mut self) -> usize {
        let mut applied = 0;
        for _ in 0..4 {
            let sent = self.send(Event::Window(
                window::Event::RedrawRequested(Instant::now()),
            ));
            if sent.messages.is_empty() {
                break;
            }
            applied += sent.messages.len();
            self.apply(sent.messages);
        }
        applied
    }

    /// Draw the tree into the software renderer, the way iced's window loop
    /// does: the same widget tree first sees `RedrawRequested` (widgets such as
    /// buttons work out their hover and pressed status then) and is then drawn.
    pub fn draw(&mut self) {
        let (cursor, size) = (self.cursor, self.size);
        let theme = self.app.theme();
        self.with_ui(|element, tree, renderer, node| {
            let mut ignored = Vec::new();
            let mut shell = Shell::new(&mut ignored);
            let redraw = Event::Window(window::Event::RedrawRequested(Instant::now()));
            element.as_widget_mut().update(
                tree,
                &redraw,
                Layout::new(node),
                cursor,
                renderer,
                &mut clipboard::Null,
                &mut shell,
                &Rectangle::with_size(size),
            );
            renderer::Renderer::reset(renderer, Rectangle::with_size(size));
            element.as_widget().draw(
                tree,
                renderer,
                &theme,
                &renderer::Style {
                    text_color: crate::theme::TEXT,
                },
                Layout::new(node),
                cursor,
                &Rectangle::with_size(size),
            );
        });
    }

    /// Move the pointer.
    pub fn move_to(&mut self, position: Point) -> Sent {
        self.cursor = mouse::Cursor::Available(position);
        self.send(Event::Mouse(mouse::Event::CursorMoved { position }))
    }

    /// Press and release the left button at `position`, applying what it publishes.
    pub fn click(&mut self, position: Point) -> Vec<Message> {
        let mut produced = self.move_to(position).messages;
        produced.extend(
            self.send(Event::Mouse(mouse::Event::ButtonPressed(
                mouse::Button::Left,
            )))
            .messages,
        );
        produced.extend(
            self.send(Event::Mouse(mouse::Event::ButtonReleased(
                mouse::Button::Left,
            )))
            .messages,
        );
        self.apply(produced.clone());
        produced
    }

    /// Scroll the wheel by `lines` notches at `position` (positive scrolls up).
    pub fn wheel(&mut self, position: Point, lines: f32) -> Vec<Message> {
        self.move_to(position);
        let sent = self.send(Event::Mouse(mouse::Event::WheelScrolled {
            delta: mouse::ScrollDelta::Lines { x: 0.0, y: lines },
        }));
        self.apply(sent.messages.clone());
        sent.messages
    }

    /// The window's pixels, RGBA, as the software renderer draws them.
    pub fn pixels(&mut self) -> Vec<u8> {
        self.draw();
        self.renderer.screenshot(
            Size::new(self.size.width as u32, self.size.height as u32),
            1.0,
            Color::BLACK,
        )
    }

    /// Write the window as a PNG (for looking at).
    pub fn save_png(&mut self, path: &Path) {
        let pixels = self.pixels();
        let png =
            crate::screenshot::encode(&pixels, self.size.width as u32, self.size.height as u32)
                .expect("a well-formed capture");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("the folder");
        }
        std::fs::write(path, png).expect("the file");
    }
}

// ------------------------------------------------------------ the real loop

/// What one turn of the window loop cost, as `Harness::pump` measured it.
#[derive(Clone, Debug, Default)]
pub struct Pumped {
    /// Time in the widget tree's `update` (and the mouse-interaction query) for
    /// the input event: the work iced does for an event before it knows whether
    /// a frame is needed.
    pub update: Duration,
    /// The input event's update asked for the next frame.
    pub asked_for_frame: bool,
    /// Frames drawn by the turn: each one is `RedrawRequested` plus a draw.
    pub frames: u32,
    /// Time in those frames (the redraw's update and the draw of the tree).
    pub frame_time: Duration,
    /// Messages the turn produced and the application handled.
    pub messages: Vec<Message>,
    /// Times the tree was built again (`view()` and layout) after the first build.
    pub rebuilds: u32,
}

/// The widget tree as iced's window loop holds it between events: built once
/// (`view()` and layout), then handed many events, until a message makes the
/// application change and the loop builds it again.
///
/// `Harness::send` above builds the tree for every event, which is easier to
/// reason about but not what iced does: widgets keep per-instance fields (a
/// button's last status, a canvas's last mouse interaction) that only exist
/// while the same instance sees successive events, and the real cost of an
/// event is the traversal, not a rebuild. Measurements use this.
pub struct Live<'a> {
    element: crate::ui::Element<'a>,
    tree: &'a mut Tree,
    node: layout::Node,
    renderer: &'a mut iced::Renderer,
    theme: iced::Theme,
    size: Size,
    pub cursor: mouse::Cursor,
    /// Messages the tree has published and nobody has taken yet.
    pub messages: Vec<Message>,
    /// True if some widget's overlay was open at an update (not modelled here).
    pub overlay_open: bool,
}

/// What one `Live::update` saw.
#[derive(Clone, Copy, Debug)]
pub struct Updated {
    pub redraw: window::RedrawRequest,
    pub interaction: mouse::Interaction,
    pub layout_changed: bool,
    pub elapsed: Duration,
}

/// What one `Live::frame` did.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    /// What the redraw's own update asked for next.
    pub redraw: window::RedrawRequest,
    pub elapsed: Duration,
}

impl Live<'_> {
    /// `UserInterface::update` for the no-overlay case: the overlay query iced
    /// makes first, every event through the root widget, a layout again when a
    /// widget invalidated it, and the mouse interaction last.
    pub fn update(&mut self, events: &[Event]) -> Updated {
        let started = Instant::now();
        let viewport = Rectangle::with_size(self.size);
        let mut redraw = window::RedrawRequest::Wait;
        let mut layout_changed = false;
        let overlay = self.element.as_widget_mut().overlay(
            self.tree,
            Layout::new(&self.node),
            self.renderer,
            &viewport,
            iced::Vector::ZERO,
        );
        self.overlay_open |= overlay.is_some();
        drop(overlay);
        for event in events {
            let mut shell = Shell::new(&mut self.messages);
            self.element.as_widget_mut().update(
                self.tree,
                event,
                Layout::new(&self.node),
                self.cursor,
                self.renderer,
                &mut clipboard::Null,
                &mut shell,
                &viewport,
            );
            redraw = redraw.min(shell.redraw_request());
            shell.revalidate_layout(|| {
                layout_changed = true;
                self.node = self.element.as_widget_mut().layout(
                    self.tree,
                    self.renderer,
                    &layout::Limits::new(Size::ZERO, self.size),
                );
            });
        }
        let interaction = self.element.as_widget().mouse_interaction(
            self.tree,
            Layout::new(&self.node),
            self.cursor,
            &viewport,
            self.renderer,
        );
        Updated {
            redraw,
            interaction,
            layout_changed,
            elapsed: started.elapsed(),
        }
    }

    /// The tree's answer to `RedrawRequested`, which the window loop repeats
    /// (up to three times) while it publishes messages or moves the layout.
    pub fn redraw_update(&mut self) -> Updated {
        self.update(&[Event::Window(
            window::Event::RedrawRequested(Instant::now()),
        )])
    }

    /// One frame as the window loop performs it: `RedrawRequested` through the
    /// tree (again, without a rebuild, while it moves the layout: at most three
    /// times), then the draw. If the redraw published messages the frame stops
    /// before drawing and the caller builds the tree again (as iced does).
    pub fn frame(&mut self) -> Frame {
        let started = Instant::now();
        let mut last = self.redraw_update();
        for _ in 0..2 {
            if !self.messages.is_empty() || !last.layout_changed {
                break;
            }
            last = self.redraw_update();
        }
        if self.messages.is_empty() {
            self.draw();
        }
        Frame {
            redraw: last.redraw,
            elapsed: started.elapsed(),
        }
    }

    /// Draw the tree into the software renderer.
    pub fn draw(&mut self) -> Duration {
        let started = Instant::now();
        let viewport = Rectangle::with_size(self.size);
        renderer::Renderer::reset(self.renderer, viewport);
        self.element.as_widget().draw(
            self.tree,
            self.renderer,
            &self.theme,
            &renderer::Style {
                text_color: crate::theme::TEXT,
            },
            Layout::new(&self.node),
            self.cursor,
            &viewport,
        );
        started.elapsed()
    }

    /// The window's pixels as the software renderer drew the tree last.
    pub fn capture(&mut self) -> Vec<u8> {
        self.renderer.screenshot(
            Size::new(self.size.width as u32, self.size.height as u32),
            1.0,
            Color::BLACK,
        )
    }

    /// Take the messages published so far.
    pub fn take_messages(&mut self) -> Vec<Message> {
        std::mem::take(&mut self.messages)
    }
}

impl Harness {
    /// Drop the widgets' remembered state, so that the next tree is built as if
    /// the window had just opened.
    pub fn forget_tree(&mut self) {
        self.tree = None;
    }

    /// How many widgets of the kind `tag` names the widget-state tree holds (as
    /// of the last build).
    pub fn count_widgets(&self, tag: iced::advanced::widget::tree::Tag) -> usize {
        fn walk(tree: &Tree, tag: iced::advanced::widget::tree::Tag) -> usize {
            usize::from(tree.tag == tag)
                + tree
                    .children
                    .iter()
                    .map(|child| walk(child, tag))
                    .sum::<usize>()
        }
        self.tree.as_ref().map_or(0, |tree| walk(tree, tag))
    }

    /// Build the widget tree once, as iced's loop does, and run `f` on it. The
    /// widget states persist across calls (as iced's cache keeps them).
    pub fn live<R>(&mut self, f: impl FnOnce(&mut Live<'_>) -> R) -> R {
        let Self {
            app,
            renderer,
            tree,
            size,
            cursor,
            ..
        } = self;
        let theme = app.theme();
        let mut element = ui::view(app);
        let tree = match tree {
            Some(tree) => {
                tree.diff(element.as_widget());
                tree
            }
            None => tree.insert(Tree::new(element.as_widget())),
        };
        let node =
            element
                .as_widget_mut()
                .layout(tree, renderer, &layout::Limits::new(Size::ZERO, *size));
        let mut live = Live {
            element,
            tree,
            node,
            renderer,
            theme,
            size: *size,
            cursor: *cursor,
            messages: Vec::new(),
            overlay_open: false,
        };
        let out = f(&mut live);
        *cursor = live.cursor;
        out
    }

    /// One turn of iced's window loop for one input event (`AboutToWait`, then the
    /// frames it leads to): the tree handles the event; messages are applied and
    /// the tree is built again; a requested redraw delivers `RedrawRequested` and
    /// draws. Returns what it cost and produced. A hover-only event (no
    /// messages) keeps one tree for the whole turn, as the real loop does.
    pub fn pump(&mut self, event: Event) -> Pumped {
        let mut out = Pumped::default();
        let mut events = vec![event];
        let mut frame_wanted = false;
        let mut first = true;
        for round in 0..24 {
            let mut published: Vec<Message> = Vec::new();
            self.live(|live| {
                if !events.is_empty() {
                    let updated = live.update(&std::mem::take(&mut events));
                    if first {
                        out.update = updated.elapsed;
                        out.asked_for_frame = updated.redraw == window::RedrawRequest::NextFrame;
                        first = false;
                    }
                    if updated.redraw == window::RedrawRequest::NextFrame {
                        frame_wanted = true;
                    }
                    published = live.take_messages();
                    if !published.is_empty() {
                        // The loop applies them, builds the tree again and redraws.
                        frame_wanted = true;
                        return;
                    }
                }
                if frame_wanted {
                    let frame = live.frame();
                    out.frame_time += frame.elapsed;
                    published = live.take_messages();
                    frame_wanted = if published.is_empty() {
                        out.frames += 1;
                        frame.redraw == window::RedrawRequest::NextFrame
                    } else {
                        // The redraw published: build the tree again and redraw.
                        true
                    };
                }
            });
            let rebuild = !published.is_empty();
            if rebuild {
                out.messages.extend(published.iter().cloned());
                self.apply(published);
                out.rebuilds += 1;
            }
            if !rebuild && !frame_wanted {
                break;
            }
            assert!(round < 23, "the window loop did not settle: a redraw loop");
        }
        out
    }
}
