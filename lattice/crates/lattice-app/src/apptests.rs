//! Tests of the application as a whole, driven through the headless harness:
//! the compute rules of the spec (what redraws, what subscribes, what a batch
//! and a hover cost) and the behaviour of the interface (selection, collapse,
//! keyboard, the New run overlay), against the demonstration service.
//!
//! Set `LATTICE_TEST_SHOTS=<folder>` to have the rendering tests also write
//! PNGs of the window there (drawn by the software renderer, no GPU).

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures::StreamExt;
use futures::executor::block_on;
use futures::stream::BoxStream;
use iced::keyboard::{Key, Modifiers, key::Named};
use iced::widget::text_editor;
use iced::{Point, Rectangle, Size, event, window};
use lattice_protocol::{
    AgentInfo, ModelChoice, Refusal, RefusalKind, RunDetail, RunEvent, RunEventKind, RunService,
    RunStatus, RunSummary, ServiceStatus, SpanRecord, StartRun,
};

use crate::app::{App, DetailTab, Follow, Message, Options, ViewMode, follow_stream};
use crate::demo::{ADVISOR, ASSISTANT, DemoService, LOCAL_MODEL, REMOTE_MODEL};
use crate::perf;
use crate::testkit::Harness;
use crate::wfmodel::{HEADER_H, ROW_H, chevron_x};

fn window_size() -> Size {
    Size::new(1440.0, 900.0)
}

fn options() -> Options {
    Options {
        select_latest: true,
        ..Options::default()
    }
}

fn harness() -> Harness {
    Harness::new(Arc::new(DemoService::instant()), options(), window_size())
}

/// A harness that has drawn once, so the canvases have reported their size and
/// built their caches; the perf counts are reset to zero.
fn settled() -> Harness {
    let mut h = harness();
    // `select_latest` selected the first span; most tests start with none selected.
    h.apply(vec![Message::SelectSpan(None)]);
    h.redraw();
    h.draw();
    let _ = perf::take_local();
    h
}

fn canvas_bounds(h: &mut Harness) -> Rectangle {
    let size = h.app.selected.as_ref().expect("a selected run").viewport;
    assert!(
        size.width > 100.0 && size.height > 100.0,
        "the waterfall reported its size: {size:?}"
    );
    // The canvas sits in a panel in the centre column: find it by walking the layout.
    h.canvas_at(size)
        .expect("the waterfall canvas is in the layout")
}

fn row_point(b: Rectangle, row: usize, x: f32) -> Point {
    Point::new(b.x + x, b.y + HEADER_H + (row as f32 + 0.5) * ROW_H)
}

fn shots() -> Option<PathBuf> {
    std::env::var_os("LATTICE_TEST_SHOTS").map(PathBuf::from)
}

fn shot(h: &mut Harness, name: &str) {
    let pixels = h.pixels();
    assert_eq!(pixels.len(), (h.size.width * h.size.height * 4.0) as usize);
    // Not a blank window: the software renderer drew real content.
    let distinct: std::collections::HashSet<[u8; 4]> = pixels
        .chunks_exact(4)
        .step_by(7)
        .map(|p| [p[0], p[1], p[2], p[3]])
        .collect();
    assert!(
        distinct.len() > 8,
        "{name}: only {} distinct colours",
        distinct.len()
    );
    if let Some(dir) = shots() {
        h.save_png(&dir.join(format!("{name}.png")));
    }
}

fn key(named: Named) -> Message {
    Message::Key(
        Key::Named(named),
        Modifiers::empty(),
        event::Status::Ignored,
    )
}

fn selected_span(h: &Harness) -> Option<String> {
    h.app.selected.as_ref().and_then(|s| s.span.clone())
}

fn rows(h: &Harness) -> usize {
    h.app.selected.as_ref().map_or(0, |s| s.rows.len())
}

// ------------------------------------------------------------- subscriptions

#[test]
fn idle_the_only_subscription_is_the_keyboard() {
    let app = App::new(Arc::new(DemoService::instant()), Options::default());
    assert_eq!(app.subscription().units(), 1);
    let h = harness();
    assert!(h.app.selected.is_some());
    assert_eq!(
        h.app.subscription().units(),
        1,
        "a finished run selected: still only the keyboard"
    );
}

fn start(service: &DemoService) -> RunSummary {
    service
        .start(StartRun {
            task: "Which model suits translating a contract?".into(),
            agent: ASSISTANT.into(),
            model: LOCAL_MODEL.into(),
        })
        .unwrap()
}

fn finish(service: &DemoService, id: &str) {
    service.stop(id).ok();
    let _ = block_on(service.follow(id, 0).unwrap().collect::<Vec<_>>());
}

#[test]
fn follow_and_the_tick_exist_only_for_the_selected_run_while_it_works() {
    let service = Arc::new(DemoService::new());
    let run = start(&service);
    let mut app = App::new(service.clone(), Options::default());

    // A run is working but nothing follows it: the keyboard and the slow poll.
    assert_eq!(app.subscription().units(), 2);

    // Selected and working: the keyboard, its follow stream, and the tick. The poll goes.
    app.select_run(&run.id);
    assert!(app.selected.as_ref().unwrap().state.following);
    assert_eq!(app.subscription().units(), 3);

    // Deselect by selecting a finished run: back to the keyboard and the poll.
    let finished = app
        .runs
        .iter()
        .find(|r| !r.status.is_active())
        .unwrap()
        .id
        .clone();
    app.select_run(&finished);
    assert_eq!(app.subscription().units(), 2);

    // The working run ends: nothing runs but the keyboard.
    finish(&service, &run.id);
    app.refresh_runs();
    assert_eq!(app.subscription().units(), 1);
}

#[test]
fn the_follow_subscription_is_identified_by_its_run_alone() {
    use std::hash::{Hash, Hasher};
    let service: Arc<dyn RunService> = Arc::new(DemoService::instant());
    let hash = |after: u64| {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        Follow {
            service: service.clone(),
            run: "0123456789abcdef".into(),
            after,
        }
        .hash(&mut hasher);
        hasher.finish()
    };
    assert_eq!(
        hash(0),
        hash(999),
        "advancing `after` must not restart the stream"
    );
}

#[test]
fn a_followed_run_arrives_as_one_message_per_batch_then_says_it_ended() {
    let service: Arc<dyn RunService> = Arc::new(DemoService::instant());
    let id = service.runs()[0].id.clone();
    let messages: Vec<Message> = block_on(
        follow_stream(&Follow {
            service: service.clone(),
            run: id.clone(),
            after: 0,
        })
        .collect(),
    );
    assert_eq!(
        messages.len(),
        2,
        "a finished run replays as one batch and then ends: {messages:?}"
    );
    let Message::Batch { run, events } = &messages[0] else {
        panic!("a batch first")
    };
    assert_eq!(run, &id);
    assert!(events.len() > 20, "the whole batch is in one message");
    assert!(matches!(
        events.last().unwrap().kind,
        RunEventKind::End { .. }
    ));
    assert!(matches!(&messages[1], Message::FollowEnded(r) if r == &id));
    // A run the service does not know is reported, not ignored.
    let missing = block_on(
        follow_stream(&Follow {
            service,
            run: "ffffffffffffffff".into(),
            after: 0,
        })
        .collect::<Vec<_>>(),
    );
    assert!(matches!(missing.as_slice(), [Message::FollowFailed(..)]));
}

// -------------------------------------------- a run that is still working

/// The first `keep` events of a finished run, as a working run's detail.
fn working_detail(full: &RunDetail, keep: usize) -> RunDetail {
    let events: Vec<RunEvent> = full
        .events
        .iter()
        .filter(|e| e.seq as usize <= keep)
        .cloned()
        .collect();
    let mut spans: std::collections::HashMap<String, SpanRecord> = std::collections::HashMap::new();
    for e in &events {
        match &e.kind {
            RunEventKind::SpanStart { span } => {
                spans.entry(span.id.clone()).or_insert_with(|| span.clone());
            }
            RunEventKind::SpanEnd { span } => {
                spans.insert(span.id.clone(), span.clone());
            }
            _ => {}
        }
    }
    let mut spans: Vec<SpanRecord> = spans.into_values().collect();
    spans.sort_by_key(|s| s.order);
    let mut run = full.run.clone();
    run.status = RunStatus::Running;
    run.ended_at = None;
    // A working run has no final output or usage yet: those arrive with `Result`.
    run.output = None;
    run.usage = None;
    run.error = None;
    RunDetail {
        run,
        trace: None,
        spans,
        events,
    }
}

/// A service whose one run is frozen part-way, so tests can deliver the rest.
struct Frozen {
    inner: DemoService,
    id: String,
    detail: RunDetail,
    finished: AtomicBool,
}

impl RunService for Frozen {
    fn status(&self) -> ServiceStatus {
        self.inner.status()
    }
    fn agents(&self) -> Vec<AgentInfo> {
        self.inner.agents()
    }
    fn models(&self) -> Vec<ModelChoice> {
        self.inner.models()
    }
    fn runs(&self) -> Vec<RunSummary> {
        let mut runs = self.inner.runs();
        for run in &mut runs {
            if run.id == self.id {
                run.status = if self.finished.load(Ordering::SeqCst) {
                    RunStatus::Completed
                } else {
                    RunStatus::Running
                };
            }
        }
        runs
    }
    fn run(&self, id: &str) -> Result<RunDetail, Refusal> {
        if id == self.id {
            Ok(self.detail.clone())
        } else {
            self.inner.run(id)
        }
    }
    fn start(&self, request: StartRun) -> Result<RunSummary, Refusal> {
        self.inner.start(request)
    }
    fn stop(&self, id: &str) -> Result<(), Refusal> {
        self.inner.stop(id)
    }
    fn follow(&self, id: &str, after: u64) -> Result<BoxStream<'static, Vec<RunEvent>>, Refusal> {
        self.inner.follow(id, after)
    }
}

fn frozen(keep: usize) -> (Arc<Frozen>, Vec<RunEvent>) {
    let inner = DemoService::instant();
    let id = inner.runs()[0].id.clone();
    let full = inner.run(&id).unwrap();
    let detail = working_detail(&full, keep);
    let tail: Vec<RunEvent> = full
        .events
        .iter()
        .filter(|e| e.seq as usize > keep)
        .cloned()
        .collect();
    (
        Arc::new(Frozen {
            inner,
            id,
            detail,
            finished: AtomicBool::new(false),
        }),
        tail,
    )
}

#[test]
fn a_working_run_grows_with_the_tick_and_a_batch_finishes_it() {
    let (service, tail) = frozen(30);
    let id = service.id.clone();
    let mut app = App::new(service.clone(), Options::default());
    app.select_run(&id);
    {
        let sel = app.selected.as_ref().unwrap();
        assert!(sel.state.following && sel.state.status.is_active());
        assert!(
            sel.state.spans.iter().any(|s| s.end.is_none()),
            "some span is still open"
        );
        assert!(
            sel.rows.iter().any(|r| r.end.is_none()),
            "an open row draws to now"
        );
    }
    assert_eq!(app.subscription().units(), 3, "keyboard, follow, tick");

    // A tick moves the axis to the clock and changes nothing else.
    let before = app.selected.as_ref().unwrap().axis;
    std::thread::sleep(std::time::Duration::from_millis(20));
    let _ = app.update(Message::Tick);
    let after = app.selected.as_ref().unwrap().axis;
    assert!(
        after.t1 > before.t1,
        "an open span makes the axis follow the clock"
    );
    assert_eq!(after.t0, before.t0);

    // A batch for another run is ignored.
    let last_seq = app.selected.as_ref().unwrap().state.last_seq;
    let _ = app.update(Message::Batch {
        run: "ffffffffffffffff".into(),
        events: tail.clone(),
    });
    assert_eq!(app.selected.as_ref().unwrap().state.last_seq, last_seq);

    // The rest arrives as one batch: the run completes, the output appears, nothing follows it.
    service.finished.store(true, Ordering::SeqCst);
    let _ = app.update(Message::Batch {
        run: id.clone(),
        events: tail.clone(),
    });
    let sel = app.selected.as_ref().unwrap();
    assert_eq!(sel.state.status, RunStatus::Completed);
    assert!(!sel.state.following);
    assert!(
        sel.state.spans.iter().all(|s| s.end.is_some()),
        "every span ended"
    );
    assert!(sel.rows.iter().all(|r| r.end.is_some()));
    assert!(sel.output.shown.starts_with("**Short answer:**"));
    assert_eq!(
        app.subscription().units(),
        1,
        "the run ended: only the keyboard is left"
    );

    // The same batch again changes nothing.
    let rows_before = rows_of(&app);
    let _ = app.update(Message::Batch {
        run: id,
        events: tail,
    });
    assert_eq!(rows_of(&app), rows_before);
}

fn rows_of(app: &App) -> usize {
    app.selected.as_ref().unwrap().rows.len()
}

#[test]
fn a_stream_that_ends_early_reads_the_run_again_instead_of_waiting() {
    let (service, _tail) = frozen(30);
    let id = service.id.clone();
    let mut app = App::new(service.clone(), Options::default());
    app.select_run(&id);
    assert!(app.selected.as_ref().unwrap().state.following);
    let _ = app.update(Message::FollowEnded(id));
    assert!(
        !app.selected.as_ref().unwrap().state.following,
        "no stream is coming: stop following"
    );
    assert!(app.subscription().units() >= 1);
}

// ------------------------------------------------- redraw and cache rules

#[test]
fn the_first_draw_builds_each_canvas_once_and_a_second_draw_builds_nothing() {
    let mut h = harness();
    h.redraw();
    let _ = perf::take_local();
    h.draw();
    let (_, first) = perf::take_local();
    assert_eq!(first, 2, "the mark and the waterfall build once each");
    h.draw();
    h.draw();
    let (_, again) = perf::take_local();
    assert_eq!(again, 0, "nothing changed: the caches serve the geometry");
}

#[test]
fn hovering_asks_for_a_redraw_but_publishes_nothing_and_keeps_the_big_cache() {
    let mut h = settled();
    let b = canvas_bounds(&mut h);
    let sent = h.move_to(row_point(b, 2, 220.0));
    assert!(
        sent.messages.is_empty(),
        "hover publishes no message, so no view() is built"
    );
    assert_eq!(
        sent.redraw,
        window::RedrawRequest::NextFrame,
        "but the hover layer needs a frame"
    );
    assert_eq!(
        sent.status,
        event::Status::Ignored,
        "hover captures nothing"
    );
    h.draw();
    assert_eq!(
        perf::take_local().1,
        0,
        "the big cache is untouched by hover"
    );

    // Moving within the same row asks for nothing.
    let same = h.move_to(row_point(b, 2, 260.0));
    assert!(same.messages.is_empty());
    assert_eq!(same.redraw, window::RedrawRequest::Wait);

    // Moving to another row moves the hover layer only.
    let next = h.move_to(row_point(b, 4, 260.0));
    assert!(next.messages.is_empty());
    assert_eq!(next.redraw, window::RedrawRequest::NextFrame);
    h.draw();
    assert_eq!(perf::take_local().1, 0);

    // Leaving the canvas clears the hover with one more redraw.
    let away = h.move_to(Point::new(5.0, 5.0));
    assert!(away.messages.is_empty());
    assert_eq!(away.redraw, window::RedrawRequest::NextFrame);
    let quiet = h.move_to(Point::new(6.0, 5.0));
    assert_eq!(
        quiet.redraw,
        window::RedrawRequest::Wait,
        "and then nothing at all"
    );
}

#[test]
fn streamed_text_updates_the_output_but_never_rebuilds_the_canvases() {
    let (service, tail) = frozen(30);
    let id = service.id.clone();
    let mut h = Harness::new(service, options(), window_size());
    h.redraw();
    h.draw();
    let _ = perf::take_local();
    let last = h.app.selected.as_ref().unwrap().state.last_seq;

    // A burst of streamed text: five deltas in one batch.
    let deltas: Vec<RunEvent> = (1..=5)
        .map(|i| RunEvent {
            seq: last + i,
            at: 0.0,
            kind: RunEventKind::Delta {
                agent: "Lattice assistant".into(),
                text: format!("word{i} "),
            },
        })
        .collect();
    h.apply(vec![Message::Batch {
        run: id.clone(),
        events: deltas,
    }]);
    let output = h.app.selected.as_ref().unwrap().output.shown.clone();
    assert!(
        output.ends_with("word1 word2 word3 word4 word5 "),
        "{output:?}"
    );
    h.draw();
    assert_eq!(
        perf::take_local().1,
        0,
        "text arrived, and neither the waterfall nor anything else was rebuilt"
    );

    // A span ending is what the waterfall draws: one rebuild.
    let ended = tail
        .iter()
        .find(|e| matches!(e.kind, RunEventKind::SpanEnd { .. }))
        .cloned()
        .unwrap();
    h.apply(vec![Message::Batch {
        run: id,
        events: vec![RunEvent {
            seq: last + 10,
            ..ended
        }],
    }]);
    h.draw();
    assert_eq!(
        perf::take_local().1,
        1,
        "a span ended: the waterfall is rebuilt once"
    );
}

#[test]
fn the_graph_is_built_once_and_served_from_its_cache_after() {
    let mut h = settled();
    h.apply(vec![Message::SetView(ViewMode::Graph)]);
    h.draw();
    assert_eq!(
        perf::take_local().1,
        1,
        "the graph is built on first sight (the waterfall is not drawn)"
    );
    h.draw();
    h.draw();
    assert_eq!(perf::take_local().1, 0);
    // Coming back to the timeline rebuilds the waterfall once.
    h.apply(vec![Message::SetView(ViewMode::Timeline)]);
    h.draw();
    assert_eq!(perf::take_local().1, 1);
}

#[test]
fn an_idle_window_asks_for_nothing() {
    let mut h = settled();
    // Nothing is running: a redraw request finds nothing to publish and nothing to redraw.
    assert_eq!(h.redraw(), 0);
    let sent = h.send(iced::Event::Window(window::Event::RedrawRequested(
        iced::time::Instant::now(),
    )));
    assert!(sent.messages.is_empty());
    assert_eq!(sent.redraw, window::RedrawRequest::Wait);
}

#[test]
fn a_click_selects_a_span_and_rebuilds_the_waterfall_cache_exactly_once() {
    let mut h = settled();
    let b = canvas_bounds(&mut h);
    let produced = h.click(row_point(b, 1, 150.0));
    assert!(
        matches!(produced.as_slice(), [Message::SelectSpan(Some(_))]),
        "{produced:?}"
    );
    let id = selected_span(&h).expect("the span is selected");
    assert_eq!(h.app.selected.as_ref().unwrap().rows[1].span_id, id);
    assert!(
        h.app.selected.as_ref().unwrap().detail.is_some(),
        "the detail column has content"
    );
    h.draw();
    assert_eq!(
        perf::take_local().1,
        1,
        "selection rebuilds the waterfall's cache once"
    );
    h.draw();
    assert_eq!(perf::take_local().1, 0);
}

#[test]
fn the_chevron_collapses_a_row_and_its_descendants_leave_the_list() {
    let mut h = settled();
    let b = canvas_bounds(&mut h);
    let all = rows(&h);
    assert!(all > 10, "the demo run has many spans, got {all}");
    let produced = h.click(row_point(b, 0, chevron_x(0)));
    assert!(
        matches!(produced.as_slice(), [Message::ToggleCollapse(_)]),
        "{produced:?}"
    );
    assert_eq!(rows(&h), 1, "the task span holds every other span");
    assert!(h.app.selected.as_ref().unwrap().rows[0].collapsed);
    h.draw();
    assert_eq!(perf::take_local().1, 1);
    // Clicking it again expands the tree.
    h.click(row_point(b, 0, chevron_x(0)));
    assert_eq!(rows(&h), all);
}

#[test]
fn collapsing_the_parent_of_the_selected_span_moves_the_selection_to_the_parent() {
    let mut h = settled();
    let (task, child) = {
        let sel = h.app.selected.as_ref().unwrap();
        (sel.rows[0].span_id.clone(), sel.rows[2].span_id.clone())
    };
    h.apply(vec![Message::SelectSpan(Some(child))]);
    h.apply(vec![Message::ToggleCollapse(task.clone())]);
    assert_eq!(
        selected_span(&h),
        Some(task),
        "a hidden selection falls back to the collapsed ancestor"
    );
}

#[test]
fn the_wheel_scrolls_the_waterfall_within_its_content() {
    let mut h = settled();
    let b = canvas_bounds(&mut h);
    let inside = row_point(b, 3, 200.0);
    let produced = h.wheel(inside, -3.0);
    assert!(
        matches!(produced.as_slice(), [Message::Scrolled(y)] if *y > 0.0),
        "{produced:?}"
    );
    let scroll = h.app.selected.as_ref().unwrap().scroll;
    assert!(scroll > 0.0);
    h.draw();
    assert_eq!(
        perf::take_local().1,
        1,
        "a scroll rebuilds the visible rows once"
    );
    // Scrolling back past the top stops at the top.
    h.wheel(inside, 500.0);
    assert_eq!(h.app.selected.as_ref().unwrap().scroll, 0.0);
    // And at the top there is nothing more to scroll: no message at all.
    assert!(h.wheel(inside, 1.0).is_empty());
    // A wheel over the runs column does not move the waterfall.
    assert!(
        h.wheel(Point::new(150.0, 400.0), -3.0)
            .iter()
            .all(|m| !matches!(m, Message::Scrolled(_)))
    );
}

#[test]
fn resizing_the_canvas_is_reported_once_and_rebuilds_its_cache() {
    let mut h = settled();
    h.size = Size::new(1200.0, 800.0);
    assert!(h.redraw() >= 1, "the canvas reports its new size");
    h.draw();
    assert!(
        perf::take_local().1 >= 1,
        "a size change rebuilds the geometry"
    );
    assert_eq!(h.redraw(), 0, "and then it is quiet");
}

// ----------------------------------------------------------- the keyboard

#[test]
fn arrow_keys_walk_the_tree_and_collapse_it() {
    let mut h = settled();
    h.apply(vec![key(Named::ArrowDown)]);
    assert_eq!(
        selected_span(&h),
        Some(h.app.selected.as_ref().unwrap().rows[0].span_id.clone()),
        "Down with nothing selected picks the first row"
    );
    h.apply(vec![key(Named::ArrowDown), key(Named::ArrowDown)]);
    assert_eq!(h.app.selected.as_ref().unwrap().selected_row(), Some(2));
    h.apply(vec![key(Named::ArrowUp)]);
    assert_eq!(h.app.selected.as_ref().unwrap().selected_row(), Some(1));

    // Left on an open row with children collapses it; Right expands it; Right again goes to the first child.
    let all = rows(&h);
    h.apply(vec![key(Named::ArrowLeft)]);
    assert!(rows(&h) < all, "Left collapsed the agent span");
    assert!(h.app.selected.as_ref().unwrap().rows[1].collapsed);
    h.apply(vec![key(Named::ArrowRight)]);
    assert_eq!(rows(&h), all, "Right expanded it");
    h.apply(vec![key(Named::ArrowRight)]);
    assert_eq!(
        h.app.selected.as_ref().unwrap().selected_row(),
        Some(2),
        "Right on an open row goes to its first child"
    );
    // Left on a leaf goes to its parent.
    h.apply(vec![key(Named::End)]);
    let last = rows(&h) - 1;
    assert_eq!(h.app.selected.as_ref().unwrap().selected_row(), Some(last));
    h.apply(vec![key(Named::ArrowLeft)]);
    assert!(h.app.selected.as_ref().unwrap().selected_row().unwrap() < last);
    h.apply(vec![key(Named::Home)]);
    assert_eq!(h.app.selected.as_ref().unwrap().selected_row(), Some(0));
}

#[test]
fn keys_a_widget_captured_do_not_move_the_selection_and_the_graph_view_ignores_them() {
    let mut h = settled();
    h.apply(vec![Message::Key(
        Key::Named(Named::ArrowDown),
        Modifiers::empty(),
        event::Status::Captured,
    )]);
    assert_eq!(
        selected_span(&h),
        None,
        "a captured key belongs to the widget that took it"
    );
    h.apply(vec![
        Message::SetView(ViewMode::Graph),
        key(Named::ArrowDown),
    ]);
    assert_eq!(selected_span(&h), None, "arrow keys walk the timeline only");
}

#[test]
fn keys_scroll_the_selected_row_into_view() {
    let mut h = settled();
    h.apply(vec![key(Named::End)]);
    let sel = h.app.selected.as_ref().unwrap();
    assert!(
        sel.scroll > 0.0,
        "the last row is below the fold, so the list scrolled"
    );
    let row = sel.selected_row().unwrap() as f32;
    let (top, bottom) = (row * ROW_H, row * ROW_H + ROW_H);
    let view = sel.viewport.height - HEADER_H;
    assert!(
        top >= sel.scroll - 0.01 && bottom <= sel.scroll + view + 0.01,
        "row {row} in {}..{}",
        sel.scroll,
        sel.scroll + view
    );
}

// ------------------------------------------------------------ New run

fn type_task(h: &mut Harness, text: &str) {
    h.apply(vec![Message::TaskEdited(text_editor::Action::Edit(
        text_editor::Edit::Paste(Arc::new(text.to_string())),
    ))]);
}

#[test]
fn ctrl_n_opens_the_overlay_escape_closes_it_and_it_works_from_inside_the_editor() {
    let mut h = settled();
    let ctrl_n = Message::Key(
        Key::Character("n".into()),
        Modifiers::COMMAND,
        event::Status::Ignored,
    );
    h.apply(vec![ctrl_n.clone()]);
    assert!(h.app.overlay.is_some());
    // Arrow keys do not move the timeline's selection while the overlay is up.
    h.apply(vec![key(Named::ArrowDown)]);
    assert_eq!(selected_span(&h), None);
    // Escape closes it even when the editor captured the key (it unfocuses on Escape).
    h.apply(vec![Message::Key(
        Key::Named(Named::Escape),
        Modifiers::empty(),
        event::Status::Captured,
    )]);
    assert!(h.app.overlay.is_none());
    // A plain "n" does nothing.
    h.apply(vec![Message::Key(
        Key::Character("n".into()),
        Modifiers::empty(),
        event::Status::Ignored,
    )]);
    assert!(h.app.overlay.is_none());
}

#[test]
fn start_is_unavailable_until_there_is_a_task_and_then_selects_the_new_run() {
    let mut h = settled();
    h.apply(vec![Message::OpenNewRun]);
    assert!(h.app.overlay.as_ref().unwrap().request().is_none());
    // Start with nothing typed does nothing.
    let runs = h.app.runs.len();
    h.apply(vec![Message::Start]);
    assert!(h.app.overlay.is_some() && h.app.runs.len() == runs);

    type_task(&mut h, "Which model suits translating a contract?");
    assert!(h.app.overlay.as_ref().unwrap().can_start());
    h.apply(vec![Message::Start]);
    assert!(h.app.overlay.is_none(), "the overlay closed");
    assert_eq!(h.app.runs.len(), runs + 1);
    let selected = h.app.selected.as_ref().unwrap();
    assert_eq!(selected.id(), h.app.runs[0].id, "the new run is on screen");
    // Let the demo run finish so its thread ends before the test does.
    let id = selected.id().to_string();
    let service = h.app.service.clone();
    let _ = block_on(service.follow(&id, 0).unwrap().collect::<Vec<_>>());
}

/// A service that refuses every start with a sentence.
struct Refusing(DemoService);

impl RunService for Refusing {
    fn status(&self) -> ServiceStatus {
        self.0.status()
    }
    fn agents(&self) -> Vec<AgentInfo> {
        self.0.agents()
    }
    fn models(&self) -> Vec<ModelChoice> {
        self.0.models()
    }
    fn runs(&self) -> Vec<RunSummary> {
        self.0.runs()
    }
    fn run(&self, id: &str) -> Result<RunDetail, Refusal> {
        self.0.run(id)
    }
    fn start(&self, _request: StartRun) -> Result<RunSummary, Refusal> {
        Err(Refusal::new(
            RefusalKind::Conflict,
            "Three runs are already working. Wait for one to finish, or stop it.",
        ))
    }
    fn stop(&self, id: &str) -> Result<(), Refusal> {
        self.0.stop(id)
    }
    fn follow(&self, id: &str, after: u64) -> Result<BoxStream<'static, Vec<RunEvent>>, Refusal> {
        self.0.follow(id, after)
    }
}

#[test]
fn a_refusal_from_start_is_shown_in_the_overlay_which_stays_open_with_the_task() {
    let mut h = Harness::new(
        Arc::new(Refusing(DemoService::instant())),
        options(),
        window_size(),
    );
    h.redraw();
    h.apply(vec![Message::OpenNewRun]);
    type_task(&mut h, "Anything at all");
    h.apply(vec![Message::Start]);
    let overlay = h.app.overlay.as_ref().expect("the overlay stays open");
    assert_eq!(
        overlay.notice.as_deref(),
        Some("Three runs are already working. Wait for one to finish, or stop it.")
    );
    assert_eq!(
        overlay.task_text().trim(),
        "Anything at all",
        "the task is kept"
    );
    shot(&mut h, "overlay-refused");
}

#[test]
fn an_unready_model_cannot_be_picked_and_a_remote_one_changes_the_disclosure() {
    let mut h = settled();
    h.apply(vec![Message::OpenNewRun]);
    let (offline, remote) = {
        let o = h.app.overlay.as_ref().unwrap();
        (
            o.models.iter().find(|m| !m.ready).cloned().unwrap(),
            o.models
                .iter()
                .find(|m| m.id == REMOTE_MODEL)
                .cloned()
                .unwrap(),
        )
    };
    h.apply(vec![Message::PickModel(offline)]);
    let o = h.app.overlay.as_ref().unwrap();
    assert_eq!(o.model.as_ref().unwrap().id, LOCAL_MODEL);
    assert!(o.notice.as_deref().unwrap().contains("not running"));
    h.apply(vec![Message::PickModel(remote)]);
    let o = h.app.overlay.as_ref().unwrap();
    assert!(o.disclosure().unwrap().ends_with("off this machine."));
    h.apply(vec![Message::PickAgent(
        o.agents.iter().find(|a| a.id == ADVISOR).cloned().unwrap(),
    )]);
    assert_eq!(
        h.app.overlay.as_ref().unwrap().agent.as_ref().unwrap().id,
        ADVISOR
    );
}

#[test]
fn stopping_a_working_run_ends_it_and_the_interface_follows() {
    let service = Arc::new(DemoService::new());
    let run = start(&service);
    let mut app = App::new(service.clone(), Options::default());
    app.select_run(&run.id);
    let _ = app.update(Message::StopRun);
    let events = block_on(service.follow(&run.id, 0).unwrap().collect::<Vec<_>>()).concat();
    let _ = app.update(Message::Batch {
        run: run.id.clone(),
        events,
    });
    let sel = app.selected.as_ref().unwrap();
    assert_eq!(sel.state.status, RunStatus::Stopped);
    assert!(!sel.state.following);
    assert!(
        sel.rows.iter().all(|r| r.end.is_some()),
        "a stopped run leaves no span running"
    );
    assert_eq!(
        app.runs[0].status,
        RunStatus::Stopped,
        "the list learned of it"
    );
}

// ------------------------------------------------- launch options, notices

#[test]
fn select_latest_selects_the_newest_run_and_its_first_span() {
    let h = harness();
    let sel = h.app.selected.as_ref().unwrap();
    assert_eq!(sel.id(), h.app.runs[0].id);
    assert_eq!(sel.span.as_deref(), Some(sel.rows[0].span_id.as_str()));
    assert!(sel.detail.is_some());
    let none = Harness::new(
        Arc::new(DemoService::instant()),
        Options::default(),
        window_size(),
    );
    assert!(none.app.selected.is_none() && none.app.overlay.is_none());
}

#[test]
fn a_run_started_at_launch_is_selected_and_working() {
    let service = Arc::new(DemoService::new());
    let request = StartRun {
        task: "Which model suits translating a contract?".into(),
        agent: ASSISTANT.into(),
        model: LOCAL_MODEL.into(),
    };
    let (app, _task) = App::boot(
        service.clone(),
        Options {
            start_run: Some(request),
            select_latest: true,
            ..Options::default()
        },
    );
    let sel = app.selected.as_ref().unwrap();
    assert!(
        sel.state.status.is_active(),
        "the newest run is the one just started"
    );
    let id = sel.id().to_string();
    finish(&service, &id);
}

#[test]
fn a_service_that_cannot_run_agents_says_why_and_new_run_is_refused() {
    struct Unavailable(DemoService);
    impl RunService for Unavailable {
        fn status(&self) -> ServiceStatus {
            ServiceStatus {
                refusal: Some("The agent runtime did not start.".into()),
                ..self.0.status()
            }
        }
        fn agents(&self) -> Vec<AgentInfo> {
            self.0.agents()
        }
        fn models(&self) -> Vec<ModelChoice> {
            self.0.models()
        }
        fn runs(&self) -> Vec<RunSummary> {
            self.0.runs()
        }
        fn run(&self, id: &str) -> Result<RunDetail, Refusal> {
            self.0.run(id)
        }
        fn start(&self, request: StartRun) -> Result<RunSummary, Refusal> {
            self.0.start(request)
        }
        fn stop(&self, id: &str) -> Result<(), Refusal> {
            self.0.stop(id)
        }
        fn follow(
            &self,
            id: &str,
            after: u64,
        ) -> Result<BoxStream<'static, Vec<RunEvent>>, Refusal> {
            self.0.follow(id, after)
        }
    }
    let mut h = Harness::new(
        Arc::new(Unavailable(DemoService::instant())),
        Options::default(),
        window_size(),
    );
    h.apply(vec![Message::OpenNewRun]);
    assert!(h.app.overlay.is_none());
    assert_eq!(
        h.app.notice.as_deref(),
        Some("The agent runtime did not start.")
    );
    shot(&mut h, "runtime-unavailable");
}

#[test]
fn selecting_a_run_the_service_no_longer_has_tells_the_person() {
    let mut h = settled();
    h.apply(vec![Message::SelectRun("ffffffffffffffff".into())]);
    assert_eq!(
        h.app.notice.as_deref(),
        Some("There is no run with that id.")
    );
    assert!(h.app.selected.is_some(), "the run on screen stays");
}

// ------------------------------------------------------------- rendering

#[test]
fn the_timeline_with_a_selected_span_renders() {
    let mut h = settled();
    h.apply(vec![Message::SelectSpan(Some(
        h.app.selected.as_ref().unwrap().rows[4].span_id.clone(),
    ))]);
    shot(&mut h, "timeline-selected");
}

#[test]
fn every_run_in_the_history_renders_in_both_views() {
    let mut h = settled();
    let ids: Vec<String> = h.app.runs.iter().map(|r| r.id.clone()).collect();
    for (i, id) in ids.iter().enumerate() {
        h.apply(vec![
            Message::SelectRun(id.clone()),
            Message::SelectSpan(None),
        ]);
        h.redraw();
        shot(&mut h, &format!("run-{i}-timeline"));
        h.apply(vec![Message::SetView(ViewMode::Graph)]);
        shot(&mut h, &format!("run-{i}-graph"));
        h.apply(vec![Message::SetView(ViewMode::Timeline)]);
    }
}

#[test]
fn the_detail_column_renders_every_span_kind_and_every_tab() {
    let mut h = settled();
    let ids: Vec<String> = h
        .app
        .selected
        .as_ref()
        .unwrap()
        .rows
        .iter()
        .map(|r| r.span_id.clone())
        .collect();
    for id in &ids {
        for tab in [DetailTab::Overview, DetailTab::Input, DetailTab::Output] {
            h.apply(vec![
                Message::SelectSpan(Some(id.clone())),
                Message::SetTab(tab),
            ]);
            h.draw();
        }
    }
    h.apply(vec![Message::SetTab(DetailTab::Overview)]);
    shot(&mut h, "detail-overview");
}

#[test]
fn a_working_run_and_the_new_run_overlay_render() {
    let (service, _tail) = frozen(34);
    let mut h = Harness::new(
        service.clone(),
        Options {
            select_latest: true,
            ..Options::default()
        },
        window_size(),
    );
    h.redraw();
    shot(&mut h, "working-run");
    h.apply(vec![Message::OpenNewRun]);
    type_task(
        &mut h,
        "Compare my local and hosted models for summarising a long report.",
    );
    shot(&mut h, "overlay-open");
}

#[test]
fn the_banner_shows_only_when_asked() {
    let mut with = Harness::new(
        Arc::new(DemoService::instant()),
        Options {
            banner: Some(crate::BANNER.into()),
            select_latest: true,
            ..Options::default()
        },
        window_size(),
    );
    with.redraw();
    shot(&mut with, "banner");
}

#[test]
fn a_narrow_window_still_lays_out() {
    let mut h = Harness::new(
        Arc::new(DemoService::instant()),
        options(),
        Size::new(1000.0, 640.0),
    );
    h.redraw();
    let viewport = h.app.selected.as_ref().unwrap().viewport;
    let b = h.canvas_at(viewport).expect("canvas");
    assert!(
        b.width > 60.0,
        "the centre column keeps a usable width at the minimum window size: {b:?}"
    );
    shot(&mut h, "minimum-size");
}

// ------------------------------------------------------ launch tab and span

#[test]
fn view_graph_opens_the_selected_run_on_the_graph_tab() {
    let mut h = Harness::new(
        Arc::new(DemoService::instant()),
        Options {
            select_latest: true,
            view: Some(ViewMode::Graph),
            ..Options::default()
        },
        window_size(),
    );
    assert_eq!(h.app.view_mode, ViewMode::Graph);
    assert!(h.app.selected.is_some(), "the newest run is selected");
    h.redraw();
    let _ = perf::take_local();
    h.draw();
    assert_eq!(
        perf::take_local().1,
        2,
        "the mark and the graph are drawn; the waterfall is not on screen"
    );
    // Without the option the window opens on the timeline, as before.
    let timeline = Harness::new(Arc::new(DemoService::instant()), options(), window_size());
    assert_eq!(timeline.app.view_mode, ViewMode::Timeline);
}

/// The frozen service, but `start` hands back its one working run, as a run
/// started at launch would be: working, with nothing recorded yet.
struct Launching(Arc<Frozen>);

impl RunService for Launching {
    fn status(&self) -> ServiceStatus {
        self.0.status()
    }
    fn agents(&self) -> Vec<AgentInfo> {
        self.0.agents()
    }
    fn models(&self) -> Vec<ModelChoice> {
        self.0.models()
    }
    fn runs(&self) -> Vec<RunSummary> {
        self.0.runs()
    }
    fn run(&self, id: &str) -> Result<RunDetail, Refusal> {
        self.0.run(id)
    }
    fn start(&self, _request: StartRun) -> Result<RunSummary, Refusal> {
        Ok(self
            .0
            .runs()
            .into_iter()
            .find(|r| r.id == self.0.id)
            .expect("the frozen run is listed"))
    }
    fn stop(&self, id: &str) -> Result<(), Refusal> {
        self.0.stop(id)
    }
    fn follow(&self, id: &str, after: u64) -> Result<BoxStream<'static, Vec<RunEvent>>, Refusal> {
        self.0.follow(id, after)
    }
}

fn launch_request() -> StartRun {
    StartRun {
        task: "Which model suits translating a contract?".into(),
        agent: ASSISTANT.into(),
        model: LOCAL_MODEL.into(),
    }
}

/// The events of the run, split just after its first span starts.
fn split_after_first_span(tail: &[RunEvent]) -> (Vec<RunEvent>, Vec<RunEvent>) {
    let at = tail
        .iter()
        .position(|e| matches!(e.kind, RunEventKind::SpanStart { .. }))
        .expect("the run has spans");
    (tail[..=at].to_vec(), tail[at + 1..].to_vec())
}

#[test]
fn a_run_started_at_launch_gets_its_first_span_selected_once_it_has_one() {
    let (frozen, tail) = frozen(0);
    let id = frozen.id.clone();
    let (mut app, _task) = App::boot(
        Arc::new(Launching(frozen)),
        Options {
            select_latest: true,
            start_run: Some(launch_request()),
            ..Options::default()
        },
    );
    let sel = app.selected.as_ref().expect("the started run is selected");
    assert_eq!(sel.id(), id);
    assert!(
        sel.rows.is_empty() && sel.span.is_none(),
        "nothing recorded yet"
    );
    assert!(sel.detail.is_none());

    let (first, rest) = split_after_first_span(&tail);
    let _ = app.update(Message::Batch {
        run: id.clone(),
        events: first,
    });
    let sel = app.selected.as_ref().unwrap();
    assert_eq!(
        sel.span.as_deref(),
        Some(sel.rows[0].span_id.as_str()),
        "the first span that came is the selected one"
    );
    assert!(sel.detail.is_some(), "and its detail is open");

    // It is chosen once: the rest of the run does not move the selection.
    let chosen = sel.span.clone();
    let _ = app.update(Message::Batch {
        run: id,
        events: rest,
    });
    assert_eq!(app.selected.as_ref().unwrap().span, chosen);
    assert!(!app.first_span_pending);
}

#[test]
fn choosing_a_span_or_a_run_yourself_ends_the_wait_for_the_first_span() {
    let start = |frozen: &Arc<Frozen>| {
        App::boot(
            Arc::new(Launching(frozen.clone())),
            Options {
                select_latest: true,
                start_run: Some(launch_request()),
                ..Options::default()
            },
        )
        .0
    };

    // The person closes the detail (selects no span) before any span exists.
    let (frozen_a, tail) = frozen(0);
    let id = frozen_a.id.clone();
    let mut app = start(&frozen_a);
    assert!(app.first_span_pending);
    let _ = app.update(Message::SelectSpan(None));
    assert!(
        !app.first_span_pending,
        "they chose: nothing is chosen for them"
    );
    let (first, _) = split_after_first_span(&tail);
    let _ = app.update(Message::Batch {
        run: id.clone(),
        events: first.clone(),
    });
    assert_eq!(app.selected.as_ref().unwrap().span, None);

    // Or they open another run first.
    let (frozen_b, _) = frozen(0);
    let mut app = start(&frozen_b);
    let other = app
        .runs
        .iter()
        .find(|r| r.id != frozen_b.id)
        .expect("the demo history has other runs")
        .id
        .clone();
    app.select_run(&other);
    assert!(!app.first_span_pending);
    let _ = app.update(Message::Batch {
        run: frozen_b.id.clone(),
        events: first,
    });
    assert_eq!(
        app.selected.as_ref().unwrap().id(),
        other,
        "the batch was for the run they left"
    );
}

#[test]
fn a_run_started_without_select_latest_selects_no_span() {
    let (frozen, tail) = frozen(0);
    let id = frozen.id.clone();
    let (mut app, _task) = App::boot(
        Arc::new(Launching(frozen)),
        Options {
            start_run: Some(launch_request()),
            ..Options::default()
        },
    );
    assert!(!app.first_span_pending, "only --select-latest waits");
    let (first, _) = split_after_first_span(&tail);
    let _ = app.update(Message::Batch {
        run: id,
        events: first,
    });
    let sel = app.selected.as_ref().unwrap();
    assert!(!sel.rows.is_empty() && sel.span.is_none());
}

// ------------------------------------------------------------ the graph panel

fn scrollables(h: &Harness) -> usize {
    let probe: iced::widget::Scrollable<'_, Message> =
        iced::widget::scrollable(iced::widget::text(""));
    h.count_widgets(iced::advanced::Widget::<Message, iced::Theme, iced::Renderer>::tag(&probe))
}

/// The scrollables in the window with `mode` showing the run `id`.
fn scrollables_showing(h: &mut Harness, id: &str, mode: ViewMode) -> usize {
    h.apply(vec![
        Message::SelectRun(id.to_string()),
        Message::SelectSpan(None),
        Message::SetView(mode),
    ]);
    h.redraw();
    h.draw();
    scrollables(h)
}

#[test]
fn a_graph_that_fits_its_panel_has_no_scrollbars_and_one_that_does_not_scrolls() {
    // At the default size every demonstration graph fits: the Graph tab has the
    // scrollables the Timeline tab has (the runs list, the output) and no more.
    let mut h = settled_window(window_size());
    let ids: Vec<String> = h.app.runs.iter().map(|r| r.id.clone()).collect();
    assert!(ids.len() >= 4);
    for id in &ids {
        let timeline = scrollables_showing(&mut h, id, ViewMode::Timeline);
        let graph = scrollables_showing(&mut h, id, ViewMode::Graph);
        assert_eq!(
            graph, timeline,
            "run {id}: the graph fits, so it is not put in a scrollable"
        );
    }
    // In the smallest window the tall graph cannot fit: it scrolls, and the
    // scrollable is there only because it is needed.
    let mut small = settled_window(Size::new(1000.0, 640.0));
    let id = small.app.runs[0].id.clone();
    let timeline = scrollables_showing(&mut small, &id, ViewMode::Timeline);
    let graph = scrollables_showing(&mut small, &id, ViewMode::Graph);
    assert_eq!(
        graph,
        timeline + 1,
        "too big for its panel: one scrollable more"
    );
    shot(&mut small, "minimum-size-graph");
}

/// A harness of `size` that has drawn once.
fn settled_window(size: Size) -> Harness {
    let mut h = Harness::new(Arc::new(DemoService::instant()), options(), size);
    h.apply(vec![Message::SelectSpan(None)]);
    h.redraw();
    h.draw();
    h
}

// ------------------------------------------- what can be pressed, and what not

#[test]
fn the_selected_row_and_the_chosen_tab_are_not_pressable_and_the_others_are() {
    let mut h = settled();
    // The selected run's row: pressing it would change nothing, so it publishes
    // nothing (and its hover cannot change anything either).
    assert!(h.click(Point::new(200.0, 130.0)).is_empty());
    // The chosen tab likewise; the other one switches the view.
    assert!(h.click(Point::new(407.0, 129.0)).is_empty());
    assert_eq!(h.app.view_mode, ViewMode::Timeline);
    let produced = h.click(Point::new(480.0, 129.0));
    assert!(
        matches!(produced.as_slice(), [Message::SetView(ViewMode::Graph)]),
        "{produced:?}"
    );
    assert_eq!(h.app.view_mode, ViewMode::Graph);
    // Now Graph is the chosen one and Timeline the way back.
    assert!(h.click(Point::new(480.0, 129.0)).is_empty());
    let produced = h.click(Point::new(407.0, 129.0));
    assert!(
        matches!(produced.as_slice(), [Message::SetView(ViewMode::Timeline)]),
        "{produced:?}"
    );
    // Another run's row selects that run.
    let other = h.app.runs[1].id.clone();
    let produced = h.click(Point::new(200.0, 215.0));
    assert!(
        matches!(produced.as_slice(), [Message::SelectRun(id)] if *id == other),
        "{produced:?}"
    );
    assert_eq!(h.app.selected.as_ref().unwrap().id(), other);
}
