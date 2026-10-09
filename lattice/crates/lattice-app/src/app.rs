//! The application: its state, the messages that change it, and the
//! subscriptions that feed it.
//!
//! This file is where the interface's "rounding error" budget is kept. In order:
//!
//! 1. Nothing redraws unless state changed: there is no `window::frames()`
//!    subscription and `unconditional-rendering` is off, so iced redraws only
//!    after a message or an input event that a widget asks a redraw for.
//! 2. Subscriptions exist only while they have work: the run `follow` stream for
//!    the selected run while it is running; a 250 ms tick only then; a 2 s poll
//!    only while some run that is not being followed is working. Idle, the only
//!    subscription is the keyboard.
//! 3. A run's events arrive in batches, one `Message` per batch.
//! 4. Canvas caches are cleared only when what they draw changed (see
//!    [`App::clear_waterfall`] and [`App::clear_graph`]); hover lives in the
//!    canvas's own state and a small uncached layer.
//! 5. Span trees, rows, details and Markdown are built here, in `update`, and
//!    kept; `view()` only reads them.
//!
//! The service is called from `update` (its methods return promptly by contract).
//! Everything the interface knows about the selected run is in [`Selected`]; a
//! run that is not selected costs nothing.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use futures::stream::BoxStream;
use iced::keyboard::{self, Key, Modifiers, key::Named};
use iced::widget::{canvas, markdown, text_editor};
use iced::{Size, Subscription, Task, Theme, event, window};
use lattice_protocol::{
    AgentInfo, ModelChoice, RunDetail, RunEvent, RunId, RunService, RunSummary, ServiceStatus,
    StartRun,
};

use crate::axis::Axis;
use crate::clock;
use crate::detail::{self, DetailModel};
use crate::graph::{self, GraphLayout};
use crate::newrun::{AgentOption, ModelOption, NewRun, TASK_INPUT_ID};
use crate::runlist::{self, ListEntry};
use crate::runstate::{Changes, RunState};
use crate::spans::{SpanKind, SpanTree, Totals, kind_of, totals, visible_rows};
use crate::theme;
use crate::wfmodel::{self, Nav, NavResult, WfRow};

/// How often open spans and elapsed times grow while the selected run works.
pub const TICK: Duration = Duration::from_millis(250);
/// How often the runs list is refreshed while a run that is not followed works.
pub const POLL: Duration = Duration::from_secs(2);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewMode {
    Timeline,
    Graph,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetailTab {
    Overview,
    Input,
    Output,
}

/// The screenshot the lead asked for on the command line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScreenshotRequest {
    pub path: PathBuf,
    pub after_ms: u64,
}

/// What the window is started with.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// A thin banner across the top (the demonstration notice), if any.
    pub banner: Option<String>,
    /// Select the newest run and its first span at launch.
    pub select_latest: bool,
    pub screenshot: Option<ScreenshotRequest>,
    /// Start this run at launch.
    pub start_run: Option<StartRun>,
    /// The tab to open on (Timeline when not given).
    pub view: Option<ViewMode>,
}

#[derive(Clone, Debug)]
pub enum Message {
    /// A batch of a followed run's events.
    Batch {
        run: RunId,
        events: Vec<RunEvent>,
    },
    /// The followed stream ended.
    FollowEnded(RunId),
    FollowFailed(RunId, String),
    /// The 250 ms tick while the selected run works.
    Tick,
    /// The 2 s poll while an unfollowed run works.
    Poll,
    SelectRun(RunId),
    SelectSpan(Option<String>),
    ToggleCollapse(String),
    /// The waterfall's scroll offset changed (wheel or scrollbar).
    Scrolled(f32),
    /// The waterfall canvas has this size now.
    WaterfallSize(Size),
    SetView(ViewMode),
    SetTab(DetailTab),
    Key(Key, Modifiers, event::Status),
    OpenNewRun,
    CloseNewRun,
    TaskEdited(text_editor::Action),
    PickAgent(AgentOption),
    PickModel(ModelOption),
    Start,
    StopRun,
    Copy(String),
    /// A Markdown link was clicked. Links are never opened: the interface has
    /// no way out of the process, by design.
    LinkClicked(String),
    CaptureScreenshot,
    Screenshot(window::Screenshot),
}

/// Streamed-text Markdown, parsed once per change.
#[derive(Default)]
pub struct OutputView {
    pub content: markdown::Content,
    /// The text `content` was parsed from.
    pub shown: String,
    pub streaming: bool,
}

impl OutputView {
    pub fn is_empty(&self) -> bool {
        self.shown.is_empty()
    }

    /// Show `text`. Text that only grew is appended, so a long stream is parsed
    /// incrementally; anything else is parsed afresh.
    pub fn set(&mut self, text: &str, streaming: bool) {
        self.streaming = streaming;
        if text == self.shown {
            return;
        }
        if !self.shown.is_empty() && text.starts_with(&self.shown) {
            self.content.push_str(&text[self.shown.len()..]);
        } else {
            self.content = markdown::Content::parse(text);
        }
        self.shown = text.to_string();
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

/// Everything the interface holds about the run on screen.
pub struct Selected {
    pub state: RunState,
    pub tree: SpanTree,
    pub collapsed: HashSet<String>,
    pub rows: Vec<WfRow>,
    pub totals: Totals,
    pub axis: Axis,
    /// Names of the agents that have an agent span in this trace.
    pub active_agents: HashSet<String>,
    /// The layout of the run's agent's graph, when the agent is known.
    pub graph: Option<GraphLayout>,
    /// The selected span's id.
    pub span: Option<String>,
    pub detail: Option<DetailModel>,
    pub output: OutputView,
    pub scroll: f32,
    /// The waterfall canvas's size, as the canvas last reported it.
    pub viewport: Size,
}

impl Selected {
    pub fn new(state: RunState, agents: &[AgentInfo], now: f64, viewport: Size) -> Self {
        let graph = agents
            .iter()
            .find(|a| a.id == state.summary.agent)
            .map(|a| graph::layout(&a.graph));
        let mut selected = Self {
            state,
            tree: SpanTree::default(),
            collapsed: HashSet::new(),
            rows: Vec::new(),
            totals: Totals::default(),
            axis: Axis::over([], now),
            active_agents: HashSet::new(),
            graph,
            span: None,
            detail: None,
            output: OutputView::default(),
            scroll: 0.0,
            viewport,
        };
        selected.rebuild(now);
        selected.refresh_output();
        selected
    }

    pub fn id(&self) -> &str {
        &self.state.summary.id
    }

    /// Rebuild everything derived from the spans. Called when spans change or a
    /// row is collapsed or expanded, never from `view()`.
    pub fn rebuild(&mut self, now: f64) {
        self.tree = SpanTree::build(&self.state.spans);
        let rows = visible_rows(&self.state.spans, &self.tree, &self.collapsed);
        self.rows = wfmodel::build_rows(&self.state, &rows);
        self.totals = totals(self.state.spans.iter().map(|s| &s.rec));
        self.refresh_time(now);
        self.active_agents = self
            .state
            .spans
            .iter()
            .filter(|s| kind_of(&s.rec) == SpanKind::Agent)
            .filter_map(|s| s.rec.data_str("name").map(str::to_string))
            .collect();
        self.fix_selection();
        self.refresh_detail(now);
        self.scroll = wfmodel::clamp_scroll(self.scroll, self.rows.len(), self.viewport.height);
    }

    /// The axis and open bars follow the clock: the only thing a tick changes.
    pub fn refresh_time(&mut self, now: f64) {
        self.axis = Axis::over(
            self.state
                .spans
                .iter()
                .map(|s| (s.start, self.state.display_end(s))),
            now,
        );
    }

    /// If the selected span is hidden (a collapsed ancestor) or gone, select the
    /// nearest visible ancestor instead.
    fn fix_selection(&mut self) {
        let Some(id) = self.span.clone() else { return };
        let visible: HashSet<&str> = self.rows.iter().map(|r| r.span_id.as_str()).collect();
        if visible.contains(id.as_str()) {
            return;
        }
        let mut at = self.state.span_index(&id);
        self.span = None;
        while let Some(i) = at {
            at = self.tree.parent.get(i).copied().flatten();
            if let Some(p) = at
                && visible.contains(self.state.spans[p].rec.id.as_str())
            {
                self.span = Some(self.state.spans[p].rec.id.clone());
                return;
            }
        }
    }

    pub fn refresh_detail(&mut self, now: f64) {
        self.detail = self
            .span
            .as_deref()
            .and_then(|id| self.state.span_by_id(id))
            .map(|entry| detail::build(entry, self.state.display_end(entry), now));
    }

    pub fn refresh_output(&mut self) {
        match self.state.output_text() {
            Some((text, streaming)) => self.output.set(&text, streaming),
            None => self.output.clear(),
        }
    }

    /// True when the selected span has not ended in a run that is working, so its
    /// duration is still counting.
    pub fn selected_is_open(&self) -> bool {
        self.span
            .as_deref()
            .and_then(|id| self.state.span_by_id(id))
            .is_some_and(|entry| self.state.is_open(entry))
    }

    /// The row index of the selected span, among the visible rows.
    pub fn selected_row(&self) -> Option<usize> {
        let id = self.span.as_deref()?;
        self.rows.iter().position(|r| r.span_id == id)
    }
}

/// The subscription data of the follow stream. It hashes by run only: `after` is
/// where the stream starts, and changing it as events arrive must not restart it.
#[derive(Clone)]
pub(crate) struct Follow {
    pub(crate) service: Arc<dyn RunService>,
    pub(crate) run: RunId,
    pub(crate) after: u64,
}

impl Hash for Follow {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.run.hash(state);
    }
}

pub(crate) fn follow_stream(follow: &Follow) -> BoxStream<'static, Message> {
    let run = follow.run.clone();
    match follow.service.follow(&follow.run, follow.after) {
        Ok(stream) => {
            let batch_run = run.clone();
            stream
                .map(move |events| Message::Batch {
                    run: batch_run.clone(),
                    events,
                })
                .chain(futures::stream::once(
                    async move { Message::FollowEnded(run) },
                ))
                .boxed()
        }
        Err(refusal) => {
            futures::stream::once(async move { Message::FollowFailed(run, refusal.message) })
                .boxed()
        }
    }
}

pub struct App {
    pub(crate) service: Arc<dyn RunService>,
    pub(crate) options: Options,
    pub(crate) theme: Theme,
    pub(crate) status: ServiceStatus,
    pub(crate) agents: Vec<AgentInfo>,
    pub(crate) models: Vec<ModelChoice>,
    pub(crate) runs: Vec<RunSummary>,
    pub(crate) entries: Vec<ListEntry>,
    pub(crate) hidden_runs: usize,
    /// Bumped whenever `entries` change; the runs list rebuilds on it.
    pub(crate) list_version: u64,
    pub(crate) now: f64,
    pub(crate) selected: Option<Selected>,
    pub(crate) view_mode: ViewMode,
    pub(crate) tab: DetailTab,
    pub(crate) overlay: Option<NewRun>,
    /// A refusal or failure to tell the person, shown under the run header.
    pub(crate) notice: Option<String>,
    /// `--select-latest` with a run started at launch: the run has no span yet, so
    /// its first span is selected when one appears (and never after anything else
    /// has been selected).
    pub(crate) first_span_pending: bool,
    pub(crate) waterfall_cache: canvas::Cache,
    pub(crate) graph_cache: canvas::Cache,
    pub(crate) mark_cache: canvas::Cache,
}

impl App {
    pub fn new(service: Arc<dyn RunService>, options: Options) -> Self {
        let now = clock::now();
        let mut app = Self {
            status: service.status(),
            agents: service.agents(),
            models: service.models(),
            runs: Vec::new(),
            entries: Vec::new(),
            hidden_runs: 0,
            list_version: 0,
            now,
            selected: None,
            view_mode: ViewMode::Timeline,
            tab: DetailTab::Overview,
            overlay: None,
            notice: None,
            first_span_pending: false,
            waterfall_cache: canvas::Cache::new(),
            graph_cache: canvas::Cache::new(),
            mark_cache: canvas::Cache::new(),
            theme: theme::theme(),
            service,
            options,
        };
        app.refresh_runs();
        app
    }

    /// The application as iced boots it: with the launch options applied and the
    /// screenshot timer started when one was asked for.
    pub fn boot(service: Arc<dyn RunService>, options: Options) -> (Self, Task<Message>) {
        let mut app = Self::new(service, options.clone());
        let mut started = false;
        if let Some(request) = options.start_run {
            match app.service.start(request) {
                Ok(summary) => {
                    started = true;
                    app.refresh_runs();
                    app.select_run(&summary.id);
                }
                Err(refusal) => app.notice = Some(refusal.message),
            }
        }
        if options.select_latest
            && let Some(newest) = app.runs.first().map(|r| r.id.clone())
        {
            app.select_run(&newest);
            let first = app
                .selected
                .as_ref()
                .and_then(|s| s.rows.first())
                .map(|r| r.span_id.clone());
            match first {
                Some(first) => app.select_span(Some(first)),
                // A run just started has no span yet: take the first one that comes.
                None => app.first_span_pending = started,
            }
        }
        if let Some(mode) = options.view {
            app.view_mode = mode;
        }
        let task = match options.screenshot {
            // The sleep is created inside the future so that it starts on iced's runtime.
            Some(request) => Task::perform(
                async move { tokio::time::sleep(Duration::from_millis(request.after_ms)).await },
                |()| Message::CaptureScreenshot,
            ),
            None => Task::none(),
        };
        (app, task)
    }

    // ------------------------------------------------------------ caches

    /// The waterfall's rows, bars, axis or selection changed.
    pub(crate) fn clear_waterfall(&self) {
        self.waterfall_cache.clear();
    }

    /// The graph's active set (or the run) changed.
    pub(crate) fn clear_graph(&self) {
        self.graph_cache.clear();
    }

    // ----------------------------------------------------------- the list

    /// Re-read the runs list and re-word it, bumping the version only when
    /// something a row shows has changed.
    pub(crate) fn refresh_runs(&mut self) {
        self.now = clock::now();
        self.status = self.service.status();
        let runs = self.service.runs();
        let (entries, hidden) = runlist::build_entries(&runs, self.now);
        if entries != self.entries || hidden != self.hidden_runs {
            self.entries = entries;
            self.hidden_runs = hidden;
            self.list_version += 1;
        }
        self.runs = runs;
    }

    /// True while a run that is not the one being followed is working.
    fn unfollowed_run_working(&self) -> bool {
        let followed = self
            .selected
            .as_ref()
            .filter(|s| s.state.following)
            .map(|s| s.id());
        self.runs
            .iter()
            .any(|r| r.status.is_active() && Some(r.id.as_str()) != followed)
    }

    /// The running dot alternates twice a second, driven by the tick.
    pub(crate) fn pulse(&self) -> u8 {
        if self.selected.as_ref().is_some_and(|s| s.state.following) {
            ((self.now * 2.0) as u64 % 2) as u8
        } else {
            0
        }
    }

    // --------------------------------------------------------- selection

    pub(crate) fn select_run(&mut self, id: &str) {
        self.now = clock::now();
        if self.selected.as_ref().is_some_and(|s| s.id() == id) {
            return;
        }
        match self.service.run(id) {
            Ok(detail) => self.load(detail),
            Err(refusal) => self.notice = Some(refusal.message),
        }
    }

    fn load(&mut self, detail: RunDetail) {
        let state = RunState::from_detail(detail);
        // The canvas keeps its size from one run to the next, so it need not report it again.
        let viewport = self.selected.as_ref().map_or(Size::ZERO, |s| s.viewport);
        self.selected = Some(Selected::new(state, &self.agents, self.now, viewport));
        self.notice = None;
        self.first_span_pending = false;
        self.clear_waterfall();
        self.clear_graph();
    }

    pub(crate) fn select_span(&mut self, id: Option<String>) {
        // Whoever selects a span (a click, a key, or the wait for the first one)
        // ends the wait for the first one.
        self.first_span_pending = false;
        let now = self.now;
        let Some(sel) = &mut self.selected else {
            return;
        };
        sel.span = id.filter(|id| sel.state.span_by_id(id).is_some());
        sel.refresh_detail(now);
        if let Some(row) = sel.selected_row() {
            sel.scroll =
                wfmodel::ensure_visible(sel.scroll, row, sel.rows.len(), sel.viewport.height);
        }
        self.clear_waterfall();
    }

    fn toggle_collapse(&mut self, id: &str) {
        let now = self.now;
        let Some(sel) = &mut self.selected else {
            return;
        };
        if !sel.collapsed.remove(id) {
            sel.collapsed.insert(id.to_string());
        }
        sel.rebuild(now);
        if let Some(row) = sel.selected_row() {
            sel.scroll =
                wfmodel::ensure_visible(sel.scroll, row, sel.rows.len(), sel.viewport.height);
        }
        self.clear_waterfall();
    }

    fn apply_batch(&mut self, run: &str, events: &[RunEvent]) {
        self.now = clock::now();
        let now = self.now;
        let Some(sel) = self.selected.as_mut().filter(|s| s.id() == run) else {
            return;
        };
        let previous_agents = sel.active_agents.len();
        let changes: Changes = sel.state.apply_batch(events);
        if changes.applied == 0 {
            return;
        }
        // Only a span starting or ending, or the run ending, changes what the
        // waterfall and the graph draw. Streamed text does not, and the clock is the
        // tick's business, so a burst of deltas leaves both caches alone.
        let structural = changes.spans || changes.ended;
        if structural {
            sel.rebuild(now);
        }
        if changes.text || changes.meta {
            sel.refresh_output();
        }
        let agents_changed = sel.active_agents.len() != previous_agents;
        if structural {
            self.clear_waterfall();
        }
        if structural || agents_changed {
            self.clear_graph();
        }
        if changes.ended {
            self.refresh_runs();
        }
        if self.first_span_pending {
            let first = self
                .selected
                .as_ref()
                .filter(|s| s.id() == run)
                .and_then(|s| s.rows.first())
                .map(|r| r.span_id.clone());
            if let Some(first) = first {
                self.select_span(Some(first));
            }
        }
    }

    /// The followed stream ended. If the run still says it is working, the
    /// service closed the stream early: read the run again rather than wait for
    /// events that will not come.
    fn follow_ended(&mut self, run: &str) {
        let Some(sel) = self.selected.as_ref().filter(|s| s.id() == run) else {
            return;
        };
        if !sel.state.following {
            return;
        }
        match self.service.run(run) {
            Ok(detail) => {
                let keep_span = sel.span.clone();
                let keep_collapsed = sel.collapsed.clone();
                self.now = clock::now();
                let mut state = RunState::from_detail(detail);
                state.following = false;
                let mut fresh = Selected::new(state, &self.agents, self.now, sel.viewport);
                fresh.collapsed = keep_collapsed;
                fresh.span = keep_span;
                fresh.rebuild(self.now);
                self.selected = Some(fresh);
                self.clear_waterfall();
                self.clear_graph();
            }
            Err(refusal) => self.notice = Some(refusal.message),
        }
        self.refresh_runs();
    }

    fn navigate(&mut self, key: Nav) {
        let Some(sel) = &self.selected else { return };
        let page =
            ((wfmodel::viewport_height(sel.viewport.height) / wfmodel::ROW_H) as usize).max(1);
        match wfmodel::navigate(&sel.rows, sel.selected_row(), key, page) {
            NavResult::Select(row) => {
                let id = sel.rows[row].span_id.clone();
                self.select_span(Some(id));
            }
            NavResult::Toggle(id) => self.toggle_collapse(&id),
            NavResult::Nothing => {}
        }
    }

    fn on_key(&mut self, key: Key, modifiers: Modifiers, status: event::Status) -> Task<Message> {
        match key.as_ref() {
            Key::Named(Named::Escape) => {
                self.overlay = None;
            }
            Key::Character(c) if modifiers.command() && c.eq_ignore_ascii_case("n") => {
                return self.open_new_run();
            }
            Key::Named(named)
                if status == event::Status::Ignored
                    && self.overlay.is_none()
                    && self.view_mode == ViewMode::Timeline =>
            {
                let nav = match named {
                    Named::ArrowUp => Nav::Up,
                    Named::ArrowDown => Nav::Down,
                    Named::ArrowLeft => Nav::Left,
                    Named::ArrowRight => Nav::Right,
                    Named::Home => Nav::Home,
                    Named::End => Nav::End,
                    Named::PageUp => Nav::PageUp,
                    Named::PageDown => Nav::PageDown,
                    _ => return Task::none(),
                };
                self.navigate(nav);
            }
            _ => {}
        }
        Task::none()
    }

    /// Open the New run overlay and put the caret in its task editor.
    fn open_new_run(&mut self) -> Task<Message> {
        if let Some(reason) = &self.status.refusal {
            self.notice = Some(reason.clone());
            return Task::none();
        }
        if self.overlay.is_none() {
            self.models = self.service.models();
            self.overlay = Some(NewRun::open(&self.agents, &self.models));
            return iced::widget::operation::focus(TASK_INPUT_ID);
        }
        Task::none()
    }

    fn start(&mut self) -> Task<Message> {
        let Some(request) = self.overlay.as_ref().and_then(NewRun::request) else {
            return Task::none();
        };
        match self.service.start(request) {
            Ok(summary) => {
                self.overlay = None;
                self.refresh_runs();
                self.select_run(&summary.id);
            }
            Err(refusal) => {
                if let Some(overlay) = &mut self.overlay {
                    overlay.notice = Some(refusal.message);
                }
            }
        }
        Task::none()
    }

    // ------------------------------------------------------------ update

    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Batch { run, events } => self.apply_batch(&run, &events),
            Message::FollowEnded(run) => self.follow_ended(&run),
            Message::FollowFailed(run, reason) => {
                if self.selected.as_ref().is_some_and(|s| s.id() == run) {
                    self.notice = Some(reason);
                }
            }
            Message::Tick => {
                self.now = clock::now();
                let now = self.now;
                if let Some(sel) = &mut self.selected {
                    sel.refresh_time(now);
                    if sel.selected_is_open() {
                        sel.refresh_detail(now);
                    }
                }
                self.clear_waterfall();
            }
            Message::Poll => self.refresh_runs(),
            Message::SelectRun(id) => {
                self.select_run(&id);
                self.refresh_runs();
            }
            Message::SelectSpan(id) => self.select_span(id),
            Message::ToggleCollapse(id) => self.toggle_collapse(&id),
            Message::Scrolled(offset) => {
                if let Some(sel) = &mut self.selected {
                    sel.scroll = wfmodel::clamp_scroll(offset, sel.rows.len(), sel.viewport.height);
                }
                self.clear_waterfall();
            }
            Message::WaterfallSize(size) => {
                if let Some(sel) = &mut self.selected {
                    sel.viewport = size;
                    sel.scroll = wfmodel::clamp_scroll(sel.scroll, sel.rows.len(), size.height);
                }
                self.clear_waterfall();
            }
            Message::SetView(mode) => {
                self.view_mode = mode;
                self.clear_waterfall();
                self.clear_graph();
            }
            Message::SetTab(tab) => self.tab = tab,
            Message::Key(key, modifiers, status) => return self.on_key(key, modifiers, status),
            Message::OpenNewRun => return self.open_new_run(),
            Message::CloseNewRun => self.overlay = None,
            Message::TaskEdited(action) => {
                if let Some(overlay) = &mut self.overlay {
                    overlay.task.perform(action);
                    overlay.notice = None;
                }
            }
            Message::PickAgent(agent) => {
                if let Some(overlay) = &mut self.overlay {
                    overlay.pick_agent(agent);
                }
            }
            Message::PickModel(model) => {
                if let Some(overlay) = &mut self.overlay {
                    overlay.pick_model(model);
                }
            }
            Message::Start => return self.start(),
            Message::StopRun => {
                if let Some(id) = self
                    .selected
                    .as_ref()
                    .filter(|s| s.state.status.is_active())
                    .map(|s| s.id().to_string())
                {
                    if let Err(refusal) = self.service.stop(&id) {
                        self.notice = Some(refusal.message);
                    }
                    self.refresh_runs();
                }
            }
            Message::Copy(text) => return iced::clipboard::write(text),
            Message::LinkClicked(_) => {}
            Message::CaptureScreenshot => {
                return window::oldest()
                    .and_then(window::screenshot)
                    .map(Message::Screenshot);
            }
            Message::Screenshot(shot) => {
                if let Some(request) = &self.options.screenshot
                    && let Err(error) = crate::screenshot::save(&request.path, &shot)
                {
                    eprintln!("lattice: screenshot failed: {error}");
                    crate::screenshot::mark_failed();
                }
                return iced::exit();
            }
        }
        Task::none()
    }

    // ----------------------------------------------------- subscriptions

    pub fn subscription(&self) -> Subscription<Message> {
        let mut subscriptions = vec![event::listen_with(key_events)];
        if let Some(sel) = self
            .selected
            .as_ref()
            .filter(|s| s.state.following && s.state.status.is_active())
        {
            let follow = Follow {
                service: self.service.clone(),
                run: sel.id().to_string(),
                after: sel.state.last_seq,
            };
            subscriptions.push(Subscription::run_with(follow, follow_stream));
            subscriptions.push(iced::time::every(TICK).map(|_| Message::Tick));
        }
        if self.unfollowed_run_working() {
            subscriptions.push(iced::time::every(POLL).map(|_| Message::Poll));
        }
        Subscription::batch(subscriptions)
    }

    pub fn theme(&self) -> Theme {
        self.theme.clone()
    }
}

/// Keyboard events for the interface. Events a widget captured (typing in the
/// task editor) are delivered too, marked as captured, so Escape and Ctrl+N
/// work from inside the editor while arrow keys there stay the editor's.
fn key_events(event: iced::Event, status: event::Status, _window: window::Id) -> Option<Message> {
    match event {
        iced::Event::Keyboard(keyboard::Event::KeyPressed { key, modifiers, .. }) => {
            Some(Message::Key(key, modifiers, status))
        }
        _ => None,
    }
}
