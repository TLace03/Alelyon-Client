//! What an input event costs, measured on the real widget tree.
//!
//! The rule is that the window's compute is a rounding error. A pointer
//! that moves without changing anything visible must therefore cost (almost)
//! nothing: no frame, and an `update` of the widget tree measured in
//! microseconds. These tests drive the demo window through the loop iced runs
//! (`Harness::live`: one built tree, many events, the widgets' per-instance
//! state kept between them; `Harness::send` builds the tree afresh for every
//! event, so widgets that remember their last status never get to ask for
//! anything) and record, for a stream of `CursorMoved` events over each part of
//! the window, how long the tree's `update` took, whether the event asked for a
//! frame, and what a frame costs.
//!
//! `LATTICE_PERF_REPORT=1 cargo test perftests -- --nocapture` prints the
//! tables. Bounds on times are loose (a test build is unoptimised): they trip on
//! a gross regression, not on noise. The frame counts are exact.

use std::sync::Arc;
use std::time::Duration;

use iced::{Event, Point, Rectangle, Size, mouse, window};

use crate::app::Options;
use crate::demo::DemoService;
use crate::perf;
use crate::testkit::{Harness, Live};
use crate::wfmodel::{HEADER_H, ROW_H};

const WINDOW: Size = Size::new(1440.0, 900.0);
/// Events per scenario.
const N: usize = 500;

fn options() -> Options {
    Options {
        select_latest: true,
        ..Options::default()
    }
}

/// A harness that has drawn once, so the canvases know their size: the newest
/// demo run with its first span selected and the detail column open, as
/// `--select-latest` leaves it.
fn settled() -> Harness {
    let mut h = Harness::new(Arc::new(DemoService::instant()), options(), WINDOW);
    h.redraw();
    h.draw();
    let _ = perf::take_local();
    let _ = perf::take_local_frames();
    h
}

/// The results of a stream of moves.
#[derive(Clone, Debug, Default)]
struct Stat {
    moves: usize,
    /// Moves whose `update` asked for the next frame.
    asked: usize,
    /// Frames the tree drew (`RedrawRequested` deliveries, from the counter).
    frames: u64,
    /// Frames that asked for yet another frame.
    chained: usize,
    /// Moves that asked for anything at all (a frame now or at a later time).
    asked_at_all: usize,
    /// The distinct mouse interactions (the cursor icon) the tree answered with.
    interactions: Vec<mouse::Interaction>,
    update_mean: Duration,
    update_max: Duration,
    frame_mean: Duration,
}

impl Stat {
    fn line(&self, name: &str) -> String {
        format!(
            "{name:<38} moves={:<4} asked={:<4} frames={:<4} update mean={:>7.1}us max={:>7.1}us  frame mean={:>7.1}us",
            self.moves,
            self.asked,
            self.frames,
            micros(self.update_mean),
            micros(self.update_max),
            micros(self.frame_mean),
        )
    }
}

fn micros(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

fn report() -> bool {
    std::env::var_os("LATTICE_PERF_REPORT").is_some()
}

fn moved(at: Point) -> Event {
    Event::Mouse(mouse::Event::CursorMoved { position: at })
}

/// Put the pointer on `entry` (the one legitimate frame of arriving is drawn and
/// not counted), then move it through `points` on the same built tree, as iced
/// would deliver them: one `AboutToWait` per event and a frame when one is asked
/// for. The first frame after the build records each widget's status, as it
/// does in the window before the pointer arrives.
fn moves(h: &mut Harness, entry: Point, points: &[Point]) -> Stat {
    let mut stat = Stat::default();
    let mut frame_total = Duration::ZERO;
    let mut update_total = Duration::ZERO;
    h.live(|live: &mut Live<'_>| {
        live.frame();
        live.cursor = mouse::Cursor::Available(entry);
        live.update(&[moved(entry)]);
        live.frame();
        let _ = perf::take_local_frames();
        for &p in points {
            live.cursor = mouse::Cursor::Available(p);
            let updated = live.update(&[moved(p)]);
            assert!(
                live.messages.is_empty(),
                "a pointer move publishes nothing: {:?}",
                live.messages
            );
            stat.moves += 1;
            update_total += updated.elapsed;
            stat.update_max = stat.update_max.max(updated.elapsed);
            stat.asked_at_all += usize::from(updated.redraw != window::RedrawRequest::Wait);
            if !stat.interactions.contains(&updated.interaction) {
                stat.interactions.push(updated.interaction);
            }
            if updated.redraw == window::RedrawRequest::NextFrame {
                stat.asked += 1;
                let frame = live.frame();
                frame_total += frame.elapsed;
                if frame.redraw == window::RedrawRequest::NextFrame {
                    stat.chained += 1;
                }
            }
        }
    });
    stat.frames = perf::take_local_frames();
    if stat.moves > 0 {
        stat.update_mean = update_total / stat.moves as u32;
    }
    if stat.asked > 0 {
        stat.frame_mean = frame_total / stat.asked as u32;
    }
    stat
}

/// `count` points along a horizontal line at `y`, from `x0` to `x1`.
fn along(x0: f32, x1: f32, y: f32, count: usize) -> Vec<Point> {
    (0..count)
        .map(|i| Point::new(x0 + (x1 - x0) * i as f32 / count as f32, y))
        .collect()
}

/// The waterfall canvas's bounds.
fn canvas(h: &mut Harness) -> Rectangle {
    let size = h.app.selected.as_ref().expect("a run is selected").viewport;
    h.canvas_at(size).expect("the waterfall canvas")
}

fn row_y(b: Rectangle, row: usize) -> f32 {
    b.y + HEADER_H + (row as f32 + 0.5) * ROW_H
}

/// The parts of the demo window the scenarios move over, for a 1440 x 900 window
/// with the detail column open.
struct Parts {
    waterfall: Rectangle,
}

impl Parts {
    fn of(h: &mut Harness) -> Self {
        Self {
            waterfall: canvas(h),
        }
    }

    /// A point inside waterfall row `row`, at `x` from the canvas's left.
    fn row(&self, row: usize, x: f32) -> Point {
        Point::new(self.waterfall.x + x, row_y(self.waterfall, row))
    }

    /// The output panel's text (below the waterfall, in the centre column).
    fn output_y(&self) -> f32 {
        self.waterfall.y + self.waterfall.height + 90.0
    }

    fn region(&self, at: Point) -> &'static str {
        if self.waterfall.contains(at) {
            "waterfall"
        } else if at.x < 57.0 {
            "rail"
        } else if at.x < 358.0 {
            "runs list"
        } else if at.x >= WINDOW.width - 380.0 {
            "detail"
        } else if at.y < self.waterfall.y {
            "header/toolbar"
        } else {
            "output panel"
        }
    }
}

/// Text long enough to make a paragraph's hit test cost something: forty
/// paragraphs of about 300 characters, one with a link.
fn long_output() -> String {
    let mut text = String::new();
    for i in 0..40 {
        text.push_str(&format!(
            "Paragraph {i}: the local model runs on this machine and nothing leaves it, \
             while the hosted model sends the text off this machine and handles much \
             longer inputs; the workstation model is not running right now, so it is not \
             an option until you start it, and the choice depends on how private the text is. \
             {}\n\n",
            if i == 7 {
                "See [the notes](https://example.invalid/notes) for more."
            } else {
                ""
            }
        ));
    }
    text
}

/// The scenarios of the task, `N` moves each, on one harness: the pointer is
/// already on the thing it moves over.
fn scenarios(h: &mut Harness) -> Vec<(&'static str, Stat)> {
    let parts = Parts::of(h);
    let b = parts.waterfall;
    let mut out = Vec::new();

    // Within one waterfall row.
    let y = row_y(b, 2);
    let row = along(b.x + 201.0, b.x + b.width - 60.0, y, N);
    out.push((
        "waterfall, inside one row",
        moves(h, Point::new(b.x + 200.0, y), &row),
    ));

    // The hovered row changes on every move.
    let alternate: Vec<Point> = (0..N).map(|i| parts.row(1 + (i % 2) * 3, 260.0)).collect();
    out.push((
        "waterfall, row changes each move",
        moves(h, parts.row(4, 260.0), &alternate),
    ));

    // Inside one runs-list row's button (the second row: the first is selected).
    let runs = along(81.0, 340.0, 210.0, N);
    out.push((
        "runs list, inside one row",
        moves(h, Point::new(80.0, 210.0), &runs),
    ));

    // Over the output's Markdown text.
    let markdown = along(401.0, 900.0, parts.output_y(), N);
    out.push((
        "output panel (markdown)",
        moves(h, Point::new(400.0, parts.output_y()), &markdown),
    ));

    // Over the detail column's text.
    let detail = along(1081.0, 1400.0, 165.0, N);
    out.push((
        "detail column",
        moves(h, Point::new(1080.0, 165.0), &detail),
    ));
    out
}

#[test]
fn every_region_is_measured_and_a_move_that_changes_nothing_asks_for_no_frame() {
    let mut h = settled();
    let table = scenarios(&mut h);
    if report() {
        println!("---- per-event cost, {N} CursorMoved per scenario");
        for (name, stat) in &table {
            println!("{}", stat.line(name));
        }
    }
    for (name, stat) in &table {
        assert_eq!(stat.moves, N, "{name}");
        // A gross-regression bound, generous for an unoptimised build.
        assert!(
            stat.update_mean < Duration::from_millis(2),
            "{name}: {:?} per move",
            stat.update_mean
        );
        assert_eq!(stat.chained, 0, "{name}: a frame never asks for another");
        assert_eq!(
            stat.asked_at_all, stat.asked,
            "{name}: nothing is asked for at a later time either"
        );
        assert_eq!(
            stat.interactions.len(),
            1,
            "{name}: the cursor icon does not flap: {:?}",
            stat.interactions
        );
        if *name == "waterfall, row changes each move" {
            assert_eq!(stat.asked, N, "{name}: a new row is a new frame");
            assert_eq!(stat.frames, N as u64, "{name}: and exactly one");
            assert!(
                stat.frame_mean < Duration::from_millis(1),
                "{name}: a frame costs {:?}",
                stat.frame_mean
            );
        } else {
            assert_eq!(
                (stat.asked, stat.frames),
                (0, 0),
                "{name}: moves that change nothing ask for no frame"
            );
        }
    }
    // What the pointer is over decides its icon: a hand over a row and a button,
    // the arrow over text.
    let icon = |name: &str| {
        table
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, s)| s.interactions[0])
            .expect("a scenario")
    };
    assert_eq!(
        icon("waterfall, inside one row"),
        mouse::Interaction::Pointer
    );
    assert_eq!(
        icon("runs list, inside one row"),
        mouse::Interaction::Pointer
    );
    assert_eq!(icon("output panel (markdown)"), mouse::Interaction::None);
    assert_eq!(icon("detail column"), mouse::Interaction::None);
}

#[test]
fn a_click_is_one_message_one_rebuild_and_one_frame() {
    let mut h = settled();
    let parts = Parts::of(&mut h);
    let at = parts.row(3, 150.0);
    // The pointer arrives on the row: one frame for the hover, no message.
    h.cursor = mouse::Cursor::Available(at);
    let arrived = h.pump(moved(at));
    assert_eq!((arrived.frames, arrived.rebuilds), (1, 0));
    assert!(arrived.messages.is_empty());
    // The canvas selects on press: the message, the new tree, the frame.
    let pressed = h.pump(Event::Mouse(mouse::Event::ButtonPressed(
        mouse::Button::Left,
    )));
    assert!(
        matches!(
            pressed.messages.as_slice(),
            [crate::app::Message::SelectSpan(Some(_))]
        ),
        "{:?}",
        pressed.messages
    );
    assert_eq!(
        (pressed.frames, pressed.rebuilds),
        (1, 1),
        "one message, one rebuild, one frame"
    );
    // Releasing publishes nothing and draws nothing.
    let released = h.pump(Event::Mouse(mouse::Event::ButtonReleased(
        mouse::Button::Left,
    )));
    assert_eq!(
        (released.frames, released.rebuilds, released.messages.len()),
        (0, 0, 0)
    );
    assert!(!released.asked_for_frame);
    // Pressing the row that is already selected is the same span again, and still
    // costs one frame, not a loop of them.
    let again = h.pump(Event::Mouse(mouse::Event::ButtonPressed(
        mouse::Button::Left,
    )));
    assert!(again.frames <= 1 && again.rebuilds <= 1, "{again:?}");
    if report() {
        println!(
            "click: update {:.1}us, frame {:.1}us",
            micros(pressed.update),
            micros(pressed.frame_time)
        );
    }
}

#[test]
fn a_long_output_costs_no_more_to_move_over_than_a_short_one() {
    let mut h = settled();
    let parts = Parts::of(&mut h);
    h.app
        .selected
        .as_mut()
        .expect("a run is selected")
        .output
        .set(&long_output(), false);
    let y = parts.output_y();
    let points = along(401.0, 900.0, y, N);
    let stat = moves(&mut h, Point::new(400.0, y), &points);
    if report() {
        println!("{}", stat.line("output panel (long markdown)"));
    }
    assert_eq!((stat.asked, stat.frames), (0, 0));
    assert!(
        stat.update_mean < Duration::from_millis(2),
        "{:?} per move over forty paragraphs",
        stat.update_mean
    );
}

#[test]
fn an_idle_window_and_a_redundant_leave_ask_for_nothing() {
    let mut h = settled();
    let parts = Parts::of(&mut h);
    h.live(|live: &mut Live<'_>| {
        live.frame();
        let _ = perf::take_local_frames();
        for _ in 0..N {
            let updated = live.update(&[]);
            assert_eq!(updated.redraw, window::RedrawRequest::Wait);
        }
        // Leaving a window the pointer was never in changes nothing.
        live.cursor = mouse::Cursor::Unavailable;
        for _ in 0..3 {
            let updated = live.update(&[Event::Mouse(mouse::Event::CursorLeft)]);
            assert_eq!(updated.redraw, window::RedrawRequest::Wait);
        }
        // Arriving is not a change either; the move that follows it is.
        let at = parts.row(2, 300.0);
        live.cursor = mouse::Cursor::Available(at);
        let updated = live.update(&[Event::Mouse(mouse::Event::CursorEntered)]);
        assert_eq!(updated.redraw, window::RedrawRequest::Wait);
        let updated = live.update(&[moved(at)]);
        assert_eq!(updated.redraw, window::RedrawRequest::NextFrame);
        live.frame();
        // Leaving takes the hover away: one frame, and the second leave none.
        live.cursor = mouse::Cursor::Unavailable;
        let updated = live.update(&[Event::Mouse(mouse::Event::CursorLeft)]);
        assert_eq!(updated.redraw, window::RedrawRequest::NextFrame);
        live.frame();
        let updated = live.update(&[Event::Mouse(mouse::Event::CursorLeft)]);
        assert_eq!(updated.redraw, window::RedrawRequest::Wait);
    });
    assert_eq!(
        perf::take_local_frames(),
        2,
        "only the hover appearing and going away drew frames"
    );
}

/// How a synthetic `WM_MOUSEMOVE` reaches the window. winit turns each into
/// `CursorEntered` + `CursorMoved` when the window does not think the pointer is
/// inside it, and asks Windows for a leave notice (`TrackMouseEvent`); if the
/// real pointer is elsewhere that notice can come straight back as
/// `CursorLeft`. Each synthetic move is then a hover that appears and a hover
/// that goes away: two frames, however still the pointer "stays" in its row.
#[test]
fn a_synthetic_move_that_enters_and_leaves_costs_two_frames_and_a_real_one_none() {
    let mut h = settled();
    let parts = Parts::of(&mut h);
    let at = parts.row(2, 300.0);
    h.live(|live: &mut Live<'_>| {
        live.frame();
        let _ = perf::take_local_frames();
        let cycles = 50;
        for _ in 0..cycles {
            // Arrive and move (one batch), then the leave notice (the next).
            live.cursor = mouse::Cursor::Available(at);
            let arrived = live.update(&[Event::Mouse(mouse::Event::CursorEntered), moved(at)]);
            if arrived.redraw == window::RedrawRequest::NextFrame {
                live.frame();
            }
            live.cursor = mouse::Cursor::Unavailable;
            let left = live.update(&[Event::Mouse(mouse::Event::CursorLeft)]);
            if left.redraw == window::RedrawRequest::NextFrame {
                live.frame();
            }
        }
        assert_eq!(
            perf::take_local_frames(),
            2 * cycles,
            "hover on, hover off: two frames per synthetic move"
        );
        // The same three events handled together (the leave already happened when
        // the loop looks) leave nothing to draw.
        live.cursor = mouse::Cursor::Unavailable;
        let together = live.update(&[
            Event::Mouse(mouse::Event::CursorEntered),
            moved(at),
            Event::Mouse(mouse::Event::CursorLeft),
        ]);
        assert_eq!(together.redraw, window::RedrawRequest::Wait);
    });
}

// ------------------------------------------------- frames versus pixels

/// One pointer position of a sweep.
#[derive(Clone, Copy, Debug)]
struct Sample {
    at: Point,
    /// The move's `update` asked for the next frame.
    asked: bool,
    /// The window would look different from what is on screen.
    changed: bool,
}

/// What the window looks like after a frame with the pointer at `at` (or
/// outside the window): a tree built from scratch, so it depends on nothing but
/// the position.
fn window_with_pointer_at(h: &mut Harness, at: Option<Point>) -> Vec<u8> {
    h.forget_tree();
    h.live(|live| {
        live.cursor = at.map_or(mouse::Cursor::Unavailable, mouse::Cursor::Available);
        if let Some(p) = at {
            live.update(&[moved(p)]);
        }
        live.frame();
        live.capture()
    })
}

/// Move the pointer through `points` and, at each, compare what the tree asks
/// for with what the window would look like. The screen shows the last frame
/// that was drawn, so a move that leaves the pixels alone should ask for no
/// frame ("wasted" if it does), and one that changes them must ask for one
/// ("missing" if it does not).
fn sweep(h: &mut Harness, points: &[Point]) -> Vec<Sample> {
    // Pass one: what the tree asks for, on one long-lived tree.
    let asked: Vec<bool> = h.live(|live: &mut Live<'_>| {
        live.frame();
        points
            .iter()
            .map(|&at| {
                live.cursor = mouse::Cursor::Available(at);
                let updated = live.update(&[moved(at)]);
                let asked = updated.redraw == window::RedrawRequest::NextFrame;
                if asked {
                    live.frame();
                }
                asked
            })
            .collect()
    });
    // Pass two: what each position looks like, and what was on screen before it.
    let mut on_screen = window_with_pointer_at(h, None);
    let mut samples = Vec::with_capacity(points.len());
    for (&at, &asked) in points.iter().zip(&asked) {
        let expected = window_with_pointer_at(h, Some(at));
        let changed = expected != on_screen;
        if asked {
            on_screen = expected;
        }
        samples.push(Sample { at, asked, changed });
    }
    h.forget_tree();
    samples
}

/// A grid of points `step` apart over `size`.
fn grid(size: Size, step: f32) -> Vec<Point> {
    let mut points = Vec::new();
    let mut y = step / 2.0;
    while y < size.height {
        let mut x = step / 2.0;
        while x < size.width {
            points.push(Point::new(x, y));
            x += step;
        }
        y += step;
    }
    points
}

/// Diagnostic, slow (a screenshot per position): moves the pointer over a grid of
/// the window and reports every position whose frame changed no pixel ("wasted")
/// or whose missing frame would leave the screen stale ("missing"). Run with
/// `cargo test --release perftests -- --ignored --nocapture`, and
/// `LATTICE_PERF_GRID=<px>` for the grid step.
#[test]
#[ignore = "diagnostic: a screenshot per pointer position"]
fn a_move_asks_for_a_frame_exactly_when_the_window_would_look_different() {
    let mut h = settled();
    let parts = Parts::of(&mut h);
    let step = std::env::var("LATTICE_PERF_GRID")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(90.0);
    let samples = sweep(&mut h, &grid(WINDOW, step));
    let wasted: Vec<&Sample> = samples.iter().filter(|s| s.asked && !s.changed).collect();
    let missing: Vec<&Sample> = samples.iter().filter(|s| !s.asked && s.changed).collect();
    println!(
        "---- sweep: {} positions, {} asked for a frame, {} changed the pixels, {} wasted, {} missing",
        samples.len(),
        samples.iter().filter(|s| s.asked).count(),
        samples.iter().filter(|s| s.changed).count(),
        wasted.len(),
        missing.len()
    );
    for s in &wasted {
        println!(
            "wasted  at ({:.0}, {:.0}) in {}",
            s.at.x,
            s.at.y,
            parts.region(s.at)
        );
    }
    for s in &missing {
        println!(
            "missing at ({:.0}, {:.0}) in {}",
            s.at.x,
            s.at.y,
            parts.region(s.at)
        );
    }
    assert!(
        missing.is_empty(),
        "the screen would go stale at {:?}",
        missing.iter().map(|s| s.at).collect::<Vec<_>>()
    );
}

/// The selected run's row looks the same hovered or not, so the pointer leaving it
/// for the list's heading (or the heading for the row) changes no pixel and asks
/// for no frame. (Crossing the list's edge is another matter: iced's scrollable
/// asks for a frame there, see the sweep.)
#[test]
fn moving_between_the_selected_row_and_its_heading_asks_for_nothing() {
    let mut h = settled();
    h.live(|live: &mut Live<'_>| {
        live.frame();
        let row = Point::new(200.0, 130.0);
        let heading = Point::new(200.0, 78.0);
        // Arrive inside the list (its scrollable asks for the one frame of that).
        live.cursor = mouse::Cursor::Available(heading);
        live.update(&[moved(heading)]);
        live.frame();
        let _ = perf::take_local_frames();
        for at in [row, heading, row, heading] {
            live.cursor = mouse::Cursor::Available(at);
            let updated = live.update(&[moved(at)]);
            assert_eq!(
                updated.redraw,
                window::RedrawRequest::Wait,
                "the selected row changes nothing when hovered: {at:?}"
            );
        }
    });
    assert_eq!(perf::take_local_frames(), 0);
}

/// The one frame per crossing that is left. iced's scrollable asks for a frame
/// when the pointer enters or leaves it, because its status (hovered or not)
/// may change how it is drawn; ours does not (see `theme::scrollbars`), but the
/// widget cannot know that. So a crossing costs one frame, and only the crossing:
/// the moves before and after it, inside or outside, cost none.
#[test]
fn crossing_the_edge_of_a_scrollable_costs_at_most_one_frame_per_crossing() {
    let mut h = settled();
    // Each line crosses one panel's scrollable: the runs list (x 57 to 358), the
    // output panel, the detail column. `y` is in a gap of each (no row, no button).
    for (name, y, outside, inside) in [
        (
            "runs list",
            420.0_f32,
            (30.0_f32, 45.0_f32),
            (100.0_f32, 330.0_f32),
        ),
        ("detail column", 300.0, (1048.0, 1056.0), (1100.0, 1400.0)),
    ] {
        h.forget_tree();
        let phases = [
            ("outside", along(outside.0, outside.1, y, 20)),
            ("inside", along(inside.0, inside.1, y, 100)),
            ("outside again", along(outside.0, outside.1, y, 20)),
        ];
        h.live(|live: &mut Live<'_>| {
            live.frame();
            let _ = perf::take_local_frames();
            for (phase, points) in &phases {
                let mut asked = 0;
                for &p in points {
                    live.cursor = mouse::Cursor::Available(p);
                    let updated = live.update(&[moved(p)]);
                    if updated.redraw == window::RedrawRequest::NextFrame {
                        asked += 1;
                        live.frame();
                    }
                }
                let crossing = usize::from(*phase != "outside");
                let allowed = if *phase == "inside" { 1 } else { crossing };
                assert!(
                    asked <= allowed,
                    "{name}, {phase}: {asked} frames for {} moves",
                    points.len()
                );
            }
        });
    }
}
