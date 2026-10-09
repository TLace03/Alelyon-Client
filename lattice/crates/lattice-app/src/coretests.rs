//! The window over the real service (`lattice-core`), driven headless.
//!
//! The rest of the app's tests run against the in-memory demonstration service.
//! These run the same window over `CoreService`, with its scripted development
//! model and temporary state, to check what only the real service can show: that
//! its events reduce to the state its stored detail gives, that its agent graph
//! and its span names agree (the window matches them by name), and that the New
//! run overlay is filled from its models with the right disclosure.
//!
//! No test here opens a window or a GPU device (the harness uses the software
//! renderer), touches the real `globals/`, or reads the process environment: the
//! service gets a temporary state root and an empty environment.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use futures::StreamExt;
use futures::executor::block_on;
use iced::Size;
use iced::widget::text_editor;
use lattice_core::{CoreConfig, CoreService, MapEnv, StateRoot};
use lattice_protocol::{Locality, RunEventKind, RunService, RunStatus, StartRun};

use crate::app::{Follow, Message, Options, follow_stream};
use crate::testkit::Harness;

const ANSWER_START: &str = "**This is the development model.**";

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos());
        let path = std::env::temp_dir().join(format!(
            "lattice-app-core-{}-{}-{nanos}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn service(state: &Path, env: MapEnv, development: bool) -> Arc<CoreService> {
    let mut config = CoreConfig::new(StateRoot::at(state), Arc::new(env));
    config.development = development;
    config.dev_step_delay = Duration::ZERO;
    Arc::new(CoreService::new(config))
}

fn size() -> Size {
    Size::new(1440.0, 900.0)
}

fn dev_run(service: &CoreService, task: &str) -> String {
    service
        .start(StartRun {
            task: task.into(),
            agent: "lattice-assistant".into(),
            model: "dev:scripted".into(),
        })
        .unwrap()
        .id
}

fn finish(service: &CoreService, id: &str) {
    let batches: Vec<_> = block_on(service.follow(id, 0).unwrap().collect());
    let last = batches.last().and_then(|batch| batch.last()).unwrap();
    assert!(matches!(last.kind, RunEventKind::End { .. }));
}

#[test]
fn the_window_shows_a_finished_run_of_the_real_service() {
    let scratch = Scratch::new();
    let core = service(scratch.path(), MapEnv::new(), true);
    let id = dev_run(&core, "Which model should I use to translate a contract?");
    finish(&core, &id);

    let mut h = Harness::new(
        core.clone(),
        Options {
            select_latest: true,
            ..Options::default()
        },
        size(),
    );
    let selected = h.app.selected.as_ref().expect("the newest run is selected");
    assert_eq!(selected.id(), id);
    assert_eq!(selected.state.status, RunStatus::Completed);
    assert_eq!(selected.state.spans.len(), 16, "every span of the run");
    assert_eq!(
        selected.tree.preorder().len(),
        16,
        "each span exactly once in the tree"
    );
    assert_eq!(selected.rows.len(), 16, "and one row each in the waterfall");
    assert!(
        selected
            .state
            .spans
            .iter()
            .all(|span| !selected.state.is_open(span))
    );
    let (text, streaming) = selected.state.output_text().expect("an output");
    assert!(!streaming && text.starts_with(ANSWER_START), "{text}");
    assert_eq!(selected.state.usage.unwrap().total_tokens, 806);
    assert_eq!(selected.state.turns, Some(4));
    assert_eq!(selected.state.last_agent.as_deref(), Some("Model advisor"));
    assert_eq!(selected.totals.errors, 0);

    // The window matches agent spans to the graph's nodes by name.
    let mut active: Vec<&str> = selected.active_agents.iter().map(String::as_str).collect();
    active.sort_unstable();
    assert_eq!(active, ["Lattice assistant", "Model advisor"]);
    let graph = h.app.agents[0].graph.clone();
    for name in &active {
        assert!(
            graph.nodes.iter().any(|node| node.label == *name),
            "the graph has a node for {name}"
        );
    }
    assert!(selected.graph.is_some(), "the graph was laid out");

    // Drawn by the software renderer: not a blank window.
    h.redraw();
    h.draw();
    let pixels = h.pixels();
    let distinct: std::collections::HashSet<[u8; 4]> = pixels
        .chunks_exact(4)
        .step_by(11)
        .map(|p| [p[0], p[1], p[2], p[3]])
        .collect();
    assert!(distinct.len() > 8, "{} distinct colours", distinct.len());
}

#[test]
fn a_run_started_in_the_overlay_is_followed_batch_by_batch_and_agrees_with_its_stored_detail() {
    let scratch = Scratch::new();
    let core = service(scratch.path(), MapEnv::new(), true);
    let mut h = Harness::new(core.clone(), Options::default(), size());
    assert!(h.app.runs.is_empty(), "a fresh state has no runs");

    h.apply(vec![Message::OpenNewRun]);
    {
        let overlay = h.app.overlay.as_ref().unwrap();
        assert_eq!(overlay.agents.len(), 1);
        assert_eq!(overlay.agents[0].label, "Lattice assistant");
        assert_eq!(
            overlay.model.as_ref().unwrap().id,
            "dev:scripted",
            "the development model is the first ready model on this machine"
        );
        assert_eq!(
            overlay.disclosure().unwrap(),
            "Runs on this machine. Nothing leaves it."
        );
    }
    h.apply(vec![Message::TaskEdited(text_editor::Action::Edit(
        text_editor::Edit::Paste(Arc::new("What is 6 * 7?".to_string())),
    ))]);
    h.apply(vec![Message::Start]);
    assert!(h.app.overlay.is_none());
    let id = h.app.selected.as_ref().unwrap().id().to_string();
    assert_eq!(h.app.runs[0].id, id);

    // What the subscription would deliver: the followed events, one message per batch.
    let messages: Vec<Message> = block_on(
        follow_stream(&Follow {
            service: core.clone(),
            run: id.clone(),
            after: 0,
        })
        .collect(),
    );
    assert!(
        messages.len() >= 2,
        "batches, then the end: {}",
        messages.len()
    );
    assert!(matches!(messages.last(), Some(Message::FollowEnded(run)) if run == &id));
    h.apply(messages);

    let live = &h.app.selected.as_ref().unwrap().state;
    assert_eq!(live.status, RunStatus::Completed);
    assert!(!live.following);
    assert!(
        live.live.is_empty(),
        "the streamed text was cleared by the complete message"
    );

    // The same run read back from the store's detail.
    let stored = crate::runstate::RunState::from_detail(core.run(&id).unwrap());
    assert_eq!(live.status, stored.status);
    assert_eq!(live.spans.len(), stored.spans.len());
    assert_eq!(live.output, stored.output);
    assert_eq!(live.usage, stored.usage);
    assert_eq!(live.turns, stored.turns);
    assert_eq!(live.messages, stored.messages);
    assert_eq!(live.error, stored.error);
    let ids = |state: &crate::runstate::RunState| {
        let mut ids: Vec<String> = state.spans.iter().map(|s| s.rec.id.clone()).collect();
        ids.sort();
        ids
    };
    assert_eq!(ids(live), ids(&stored));
    assert!(
        live.spans.iter().all(|span| span.rec.ended_at.is_some()),
        "the ends superseded the starts"
    );
    assert_eq!(h.app.runs[0].status, RunStatus::Completed);
}

#[test]
fn the_overlay_lists_the_registrys_models_and_discloses_where_each_one_runs() {
    let scratch = Scratch::new();
    let state = StateRoot::at(scratch.path());
    std::fs::create_dir_all(&state.globals).unwrap();
    std::fs::write(
        state.globals.join("model_endpoints.json"),
        r#"{"version": 1, "endpoints": [
            {"id": "hosted", "label": "Hosted model", "base_url": "https://models.example.test/v1", "model": "big", "api_key_name": "HOSTED_API_KEY"},
            {"id": "openai", "enabled": true},
            {"id": "on-this-machine", "label": "A server on this machine", "base_url": "http://127.0.0.1:8080/v1", "model": "local-model"}
        ]}"#,
    )
    .unwrap();
    let env = MapEnv::new().with("HOSTED_API_KEY", "fixture-value");
    let core = service(scratch.path(), env, false);
    let mut h = Harness::new(core, Options::default(), size());
    h.apply(vec![Message::OpenNewRun]);

    let (hosted, openai, local) = {
        let overlay = h.app.overlay.as_ref().unwrap();
        assert_eq!(
            overlay.models.len(),
            18,
            "every endpoint of the registry, and no development model"
        );
        // The managed llama.cpp row is listed, and says why this runtime cannot
        // use it yet (ADR-0041).
        let managed = overlay
            .models
            .iter()
            .find(|m| m.id == "endpoint:llamacpp-local")
            .unwrap();
        assert!(!managed.ready && managed.locality == Locality::Local && managed.refusal.is_some());
        assert!(overlay.models.iter().all(|m| m.id != "dev:scripted"));
        (
            overlay
                .models
                .iter()
                .find(|m| m.id == "endpoint:hosted")
                .cloned()
                .unwrap(),
            overlay
                .models
                .iter()
                .find(|m| m.id == "endpoint:openai")
                .cloned()
                .unwrap(),
            overlay
                .models
                .iter()
                .find(|m| m.id == "endpoint:on-this-machine")
                .cloned()
                .unwrap(),
        )
    };
    assert_eq!((hosted.locality, hosted.ready), (Locality::Remote, true));
    assert_eq!((openai.locality, openai.ready), (Locality::Remote, false));
    assert_eq!((local.locality, local.ready), (Locality::Local, true));
    assert_eq!(
        h.app.overlay.as_ref().unwrap().model.as_ref().unwrap().id,
        "endpoint:on-this-machine",
        "it opens on a ready model that stays on this machine"
    );

    // A model without a key is listed with its reason and cannot be picked.
    h.apply(vec![Message::PickModel(openai.clone())]);
    let overlay = h.app.overlay.as_ref().unwrap();
    assert_eq!(
        overlay.model.as_ref().unwrap().id,
        "endpoint:on-this-machine"
    );
    let notice = overlay.notice.as_deref().unwrap();
    assert_eq!(notice, "It needs an API key, and none is set.");
    assert!(
        !notice.contains("OPENAI_API_KEY"),
        "a key name never reaches the window"
    );
    assert!(
        openai
            .to_string()
            .contains("unavailable: It needs an API key")
    );

    // A remote one is disclosed before the run starts.
    h.apply(vec![Message::PickModel(hosted)]);
    assert_eq!(
        h.app.overlay.as_ref().unwrap().disclosure().unwrap(),
        "This task, tool results and the conversation go to Hosted model, off this machine."
    );
}

#[test]
fn a_refusal_from_the_real_service_is_shown_in_the_overlay_as_it_came() {
    let scratch = Scratch::new();
    let mut config = CoreConfig::new(StateRoot::at(scratch.path()), Arc::new(MapEnv::new()));
    config.development = true;
    // Runs that stay open, so that three of them use up the service's slots.
    config.dev_step_delay = Duration::from_secs(3_600);
    let core = Arc::new(CoreService::new(config));
    let held: Vec<String> = (0..3)
        .map(|n| dev_run(&core, &format!("run {n}")))
        .collect();

    let mut h = Harness::new(core.clone(), Options::default(), size());
    h.apply(vec![Message::OpenNewRun]);
    h.apply(vec![Message::TaskEdited(text_editor::Action::Edit(
        text_editor::Edit::Paste(Arc::new("One run too many".to_string())),
    ))]);
    h.apply(vec![Message::Start]);
    let overlay = h
        .app
        .overlay
        .as_ref()
        .expect("the overlay stays open on a refusal");
    assert_eq!(
        overlay.notice.as_deref(),
        Some("3 runs are already working. Wait for one to finish, or stop it.")
    );
    assert_eq!(overlay.task_text(), "One run too many", "the task is kept");
    assert_eq!(h.app.runs.len(), 3, "and no run was made");

    // Stopping one makes room, and the same overlay then starts the run.
    core.stop(&held[0]).unwrap();
    finish(&core, &held[0]);
    h.apply(vec![Message::Start]);
    assert!(h.app.overlay.is_none(), "the run started");
    assert_eq!(h.app.runs.len(), 4);
    for id in held[1..].iter().chain([&h.app.runs[0].id.clone()]) {
        core.stop(id).unwrap();
        finish(&core, id);
    }
}

#[test]
fn the_screenshot_flags_work_with_the_real_service_too() {
    // `--screenshot` and `--select-latest` are options of the window, not of a
    // service: the same `Options` boot the real one.
    let scratch = Scratch::new();
    let core = service(scratch.path(), MapEnv::new(), true);
    let id = dev_run(&core, "Which model?");
    finish(&core, &id);
    let h = Harness::new(
        core,
        Options {
            select_latest: true,
            ..Options::default()
        },
        size(),
    );
    let selected = h.app.selected.as_ref().unwrap();
    assert_eq!(selected.id(), id);
    assert!(
        selected.span.is_some(),
        "--select-latest also selects the first span"
    );
}
