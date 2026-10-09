//! The run manager, end to end and at its bounds.
//!
//! Everything here runs in temporary directories, with a `MapEnv` for the
//! environment; no test reads or writes the real `globals/`. Models are the
//! scripted development model (with no delay unless a test wants a run to stay
//! open), scripted models handed in through the model-factory hook, or, in the
//! two tests that exercise the real client, a stub HTTP server on 127.0.0.1.

use std::collections::HashSet;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener};
use std::path::Path;
use std::sync::atomic::AtomicBool;

use futures::executor::block_on;
use lattice_agents::OutputItem;
use lattice_agents::testing::{ScriptedModel, ScriptedStep, assistant_message, function_call};
use lattice_protocol::{EndStatus, RunEventKind};
use serde_json::{Value, json};

use super::*;
use crate::env::MapEnv;
use crate::testkit::{TempDir, within};

const ASSISTANT: &str = "lattice-assistant";

fn config(dir: &Path, env: MapEnv, development: bool) -> CoreConfig {
    let mut config = CoreConfig::new(StateRoot::at(dir), Arc::new(env));
    config.development = development;
    config.dev_step_delay = Duration::ZERO;
    config
}

fn dev_service(dir: &Path) -> CoreService {
    CoreService::new(config(dir, MapEnv::new(), true))
}

fn request(task: &str, model: &str) -> StartRun {
    StartRun {
        task: task.into(),
        agent: ASSISTANT.into(),
        model: model.into(),
    }
}

/// Every event of a run from `after`, waiting (at most 60 s) for its `End`.
fn events_of(service: &CoreService, id: &str, after: u64) -> Vec<RunEvent> {
    let service = service.clone();
    let id = id.to_owned();
    within("following a run to its end", 60, move || {
        let batches: Vec<Vec<RunEvent>> = block_on(service.follow(&id, after).unwrap().collect());
        batches.into_iter().flatten().collect()
    })
}

fn end_status(events: &[RunEvent]) -> EndStatus {
    match events.last().map(|event| &event.kind) {
        Some(RunEventKind::End { status }) => *status,
        other => panic!("the last event is not End: {other:?}"),
    }
}

fn spans_of<'a>(events: &'a [RunEvent], kind: &str) -> Vec<&'a SpanRecord> {
    events
        .iter()
        .filter_map(|event| match &event.kind {
            RunEventKind::SpanEnd { span } if span.data_type() == kind => Some(span),
            _ => None,
        })
        .collect()
}

/// `serde_json` reads a float back to within one unit in the last place (it
/// has a fast path, and its exact one is a feature this crate does not turn on),
/// which is a fraction of a microsecond in a timestamp: times are compared to a microsecond.
fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

fn same_event(a: &RunEvent, b: &RunEvent) -> bool {
    near(a.at, b.at)
        && a == &RunEvent {
            at: a.at,
            ..b.clone()
        }
}

fn same_summary(a: &RunSummary, b: &RunSummary) -> bool {
    near(a.created_at, b.created_at)
        && near(a.updated_at, b.updated_at)
        && a.ended_at.zip(b.ended_at).is_none_or(|(x, y)| near(x, y))
        && a.ended_at.is_some() == b.ended_at.is_some()
        && a == &RunSummary {
            created_at: a.created_at,
            updated_at: a.updated_at,
            ended_at: a.ended_at,
            ..b.clone()
        }
}

fn persisted_lines(dir: &Path, id: &str) -> Vec<String> {
    let path = StateRoot::at(dir)
        .runs_dir()
        .join(format!("{id}.events.jsonl"));
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect()
}

fn named<'a>(events: &'a [RunEvent], span_type: &str, name: &str) -> Option<&'a SpanRecord> {
    spans_of(events, span_type)
        .into_iter()
        .find(|span| span.data_str("name") == Some(name))
}

#[test]
fn a_development_run_records_a_complete_coherent_history() {
    let dir = TempDir::new("manager-dev");
    let service = dev_service(dir.path());
    let summary = service
        .start(request(
            "What is 6 * 7, and which model should I use?",
            "dev:scripted",
        ))
        .unwrap();
    assert!(is_run_id(&summary.id));
    assert_eq!(summary.status, RunStatus::Running);
    assert_eq!(summary.task, "What is 6 * 7, and which model should I use?");
    assert_eq!(
        (summary.agent.as_str(), summary.agent_label.as_str()),
        (ASSISTANT, "Lattice assistant")
    );
    assert_eq!(summary.model, "dev:scripted");
    assert_eq!(summary.locality, Locality::Local);
    assert!(summary.trace_id.starts_with("trace_") && summary.trace_id.len() == 38);

    let events = events_of(&service, &summary.id, 0);
    for (index, event) in events.iter().enumerate() {
        assert!(event.seq > 0);
        if index > 0 {
            assert!(event.seq > events[index - 1].seq, "seq rises");
            assert!(event.at >= events[index - 1].at, "time does not go back");
        }
    }
    assert_eq!(events[0].seq, 1, "seq starts at 1");
    assert!(
        matches!(&events[0].kind, RunEventKind::TraceStart { trace_id, .. } if *trace_id == summary.trace_id)
    );
    assert_eq!(end_status(&events), EndStatus::Completed);
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e.kind, RunEventKind::End { .. }))
            .count(),
        1
    );
    let last_but_one = &events[events.len() - 2].kind;
    assert!(
        matches!(last_but_one, RunEventKind::Result { .. }),
        "Result comes just before End: {last_but_one:?}"
    );

    // The spans the development run makes.
    assert_eq!(spans_of(&events, "agent").len(), 2);
    assert_eq!(spans_of(&events, "function").len(), 3);
    assert_eq!(spans_of(&events, "handoff").len(), 1);
    assert_eq!(spans_of(&events, "guardrail").len(), 1);
    assert_eq!(spans_of(&events, "generation").len(), 4);
    assert_eq!(
        spans_of(&events, "custom").len(),
        5,
        "one task and four turns"
    );
    let started = events
        .iter()
        .filter(|e| matches!(e.kind, RunEventKind::SpanStart { .. }))
        .count();
    let ended = events
        .iter()
        .filter(|e| matches!(e.kind, RunEventKind::SpanEnd { .. }))
        .count();
    assert_eq!((started, ended), (16, 16), "every span that starts ends");
    let guardrail = spans_of(&events, "guardrail")[0];
    assert_eq!(guardrail.data_str("name"), Some("secrets_stay_local"));
    assert_eq!(guardrail.span_data["triggered"], false);
    let sum = named(&events, "function", "calculate").unwrap();
    assert_eq!(sum.span_data["input"], "{\"expression\":\"6 * 7\"}");
    assert_eq!(sum.span_data["output"], "42");
    let listing = named(&events, "function", "list_models").unwrap();
    let rows: Value = serde_json::from_str(listing.span_data["output"].as_str().unwrap()).unwrap();
    assert!(
        rows.as_array()
            .unwrap()
            .iter()
            .all(|row| row["id"].as_str() != Some("dev:scripted"))
    );
    let generation = spans_of(&events, "generation");
    assert_eq!(
        generation[0].span_data["model"], "scripted-development-model",
        "the development model records its spans in full"
    );
    assert_eq!(generation[0].span_data["usage"]["input_tokens"], 120);
    let handoff = spans_of(&events, "handoff")[0];
    assert_eq!(handoff.span_data["from_agent"], "Lattice assistant");
    assert_eq!(handoff.span_data["to_agent"], "Model advisor");

    // The order things happened in is the order they are recorded in.
    let position = |predicate: &dyn Fn(&RunEventKind) -> bool| {
        events.iter().position(|e| predicate(&e.kind)).unwrap()
    };
    for name in ["current_time", "calculate", "list_models"] {
        let called =
            position(&|k| matches!(k, RunEventKind::ToolCall { name: n, .. } if n == name));
        let output = position(&|k| {
            matches!(k, RunEventKind::ToolOutput { call_id, .. } if call_id.starts_with("call_dev_") && {
                // The output of the call that `called` announced.
                matches!(&events[called].kind, RunEventKind::ToolCall { call_id: c, .. } if c == call_id)
            })
        });
        let span = named(&events, "function", name).unwrap().id.clone();
        let span_start =
            position(&|k| matches!(k, RunEventKind::SpanStart { span: s } if s.id == span));
        let span_end =
            position(&|k| matches!(k, RunEventKind::SpanEnd { span: s } if s.id == span));
        assert!(
            called < span_start,
            "{name}: the call is announced before its span starts"
        );
        assert!(
            span_start < span_end && span_end < output,
            "{name}: its output is announced after its span ended"
        );
    }
    let agent_event =
        position(&|k| matches!(k, RunEventKind::Agent { name } if name == "Lattice assistant"));
    let first_agent_span =
        position(&|k| matches!(k, RunEventKind::SpanStart { span } if span.data_type() == "agent"));
    assert!(
        agent_event < first_agent_span,
        "the agent change is announced before its span starts"
    );
    let handoff_event = position(&|k| matches!(k, RunEventKind::Handoff { .. }));
    assert!(events.iter().any(|e| matches!(&e.kind, RunEventKind::HandoffRequested { agent } if agent == "Lattice assistant")));
    assert!(events.iter().any(|e| matches!(&e.kind, RunEventKind::Handoff { from, to } if from == "Lattice assistant" && to == "Model advisor")));
    assert!(
        events
            .iter()
            .any(|e| matches!(&e.kind, RunEventKind::Agent { name } if name == "Model advisor"))
    );
    assert!(handoff_event > agent_event);

    // The answer streamed, and arrived whole.
    let streamed: String = events
        .iter()
        .filter_map(|e| match &e.kind {
            RunEventKind::Delta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(streamed, crate::devmodel::DevModel::answer());
    match &events[events.len() - 2].kind {
        RunEventKind::Result {
            output,
            usage,
            turns,
            last_agent,
        } => {
            assert_eq!(output, crate::devmodel::DevModel::answer());
            assert_eq!(*turns, 4);
            assert_eq!(last_agent, "Model advisor");
            let usage = usage.unwrap();
            assert_eq!(
                (usage.requests, usage.input_tokens, usage.output_tokens),
                (4, 670, 136)
            );
        }
        other => panic!("{other:?}"),
    }

    // The summary in the list.
    let listed = service.runs();
    assert_eq!(listed.len(), 1);
    let summary = &listed[0];
    assert_eq!(summary.status, RunStatus::Completed);
    assert_eq!(summary.spans, 16);
    assert!(
        summary
            .output
            .as_deref()
            .unwrap()
            .starts_with("**This is the development model.**")
    );
    assert_eq!(summary.usage.unwrap().total_tokens, 806);
    assert!(summary.ended_at.unwrap() >= summary.created_at && summary.error.is_none());

    // What is on disk: no delta, in order, the same as the events in memory.
    let lines = persisted_lines(dir.path(), &summary.id);
    let parsed: Vec<RunEvent> = lines
        .iter()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert!(
        parsed
            .iter()
            .all(|e| !matches!(e.kind, RunEventKind::Delta { .. }))
    );
    assert!(parsed.windows(2).all(|pair| pair[0].seq < pair[1].seq));
    let expected: Vec<RunEvent> = events
        .iter()
        .filter(|e| !matches!(e.kind, RunEventKind::Delta { .. }))
        .cloned()
        .collect();
    assert_eq!(
        parsed.len(),
        expected.len(),
        "the same number of events is on disk and in memory"
    );
    for (on_disk, in_memory) in parsed.iter().zip(&expected) {
        assert_eq!(on_disk.seq, in_memory.seq);
        assert!(
            same_event(on_disk, in_memory),
            "event {} differs between disk and memory",
            on_disk.seq
        );
    }
    let on_disk = RunStore::new(StateRoot::at(dir.path()).runs_dir())
        .read_summary(&summary.id)
        .unwrap();
    assert!(
        same_summary(&on_disk, summary),
        "the final summary is written: {on_disk:?} vs {summary:?}"
    );

    // And what `run` says.
    let detail = service.run(&summary.id).unwrap();
    assert_eq!(&detail.run, summary);
    assert_eq!(
        detail.events, expected,
        "the events in memory, as they were recorded"
    );
    assert_eq!(detail.spans.len(), 16);
    assert!(
        detail
            .spans
            .windows(2)
            .all(|pair| pair[0].order < pair[1].order),
        "spans are sorted by order"
    );
    assert!(
        detail.spans.iter().all(|span| span.ended_at.is_some()),
        "the ends supersede the starts"
    );
    let trace = detail.trace.unwrap();
    assert_eq!(trace.id, summary.trace_id);
    assert_eq!(trace.workflow_name, "Lattice assistant");
    assert!(trace.started_at.is_some() && trace.ended_at.is_some());
    assert_eq!(service.persist_failures(), 0);
    assert_eq!(service.dropped(&summary.id), Some((0, 0)));
}

#[test]
fn following_an_ended_run_gives_the_rest_and_ends() {
    let dir = TempDir::new("manager-follow");
    let service = dev_service(dir.path());
    let id = service
        .start(request("Which model?", "dev:scripted"))
        .unwrap()
        .id;
    let all = events_of(&service, &id, 0);
    let middle = all[all.len() / 2].seq;
    let rest = events_of(&service, &id, middle);
    assert_eq!(
        rest.first().map(|e| e.seq),
        all.iter().map(|e| e.seq).find(|seq| *seq > middle)
    );
    assert_eq!(rest.last(), all.last());
    assert!(rest.iter().all(|e| e.seq > middle));
    let last = all.last().unwrap().seq;
    assert!(
        events_of(&service, &id, last).is_empty(),
        "nothing after the last event: the stream ends at once"
    );
    assert!(events_of(&service, &id, last + 1_000).is_empty());
    let batches: Vec<Vec<RunEvent>> = {
        let service = service.clone();
        let id = id.clone();
        within("collecting batches", 30, move || {
            block_on(service.follow(&id, 0).unwrap().collect())
        })
    };
    assert!(
        batches.iter().all(|batch| !batch.is_empty()),
        "an empty batch is never sent"
    );
    assert_eq!(batches.iter().map(Vec::len).sum::<usize>(), all.len());
    assert!(
        matches!(
            batches.last().unwrap().last().unwrap().kind,
            RunEventKind::End { .. }
        ),
        "the batch that holds End is the last"
    );
}

#[test]
fn a_live_follower_receives_batches_and_the_stream_ends_after_end() {
    let dir = TempDir::new("manager-live");
    let mut cfg = config(dir.path(), MapEnv::new(), true);
    cfg.dev_step_delay = Duration::from_millis(60);
    let service = CoreService::new(cfg);
    let id = service
        .start(request("Which model?", "dev:scripted"))
        .unwrap()
        .id;
    // Follow at once: most events are not there yet.
    let batches: Vec<Vec<RunEvent>> = {
        let service = service.clone();
        let id = id.clone();
        within("following a live run", 60, move || {
            block_on(service.follow(&id, 0).unwrap().collect())
        })
    };
    let flat: Vec<&RunEvent> = batches.iter().flatten().collect();
    assert!(
        batches.len() > 3,
        "events arrive over time, in several batches: {}",
        batches.len()
    );
    assert_eq!(flat.first().unwrap().seq, 1);
    assert!(flat.windows(2).all(|pair| pair[1].seq > pair[0].seq));
    assert!(matches!(
        flat.last().unwrap().kind,
        RunEventKind::End {
            status: EndStatus::Completed
        }
    ));
    assert_eq!(service.runs()[0].status, RunStatus::Completed);
}

#[test]
fn start_refuses_what_it_cannot_run() {
    let dir = TempDir::new("manager-refuse");
    let service = dev_service(dir.path());
    let refused = |request: StartRun| service.start(request).unwrap_err();

    let empty = refused(request("   \n\t ", "dev:scripted"));
    assert_eq!(
        (empty.kind, empty.message.as_str()),
        (RefusalKind::Invalid, "Write the task first.")
    );
    let long = refused(request(&"x".repeat(MAX_TASK_CHARS + 1), "dev:scripted"));
    assert_eq!(long.kind, RefusalKind::Invalid);
    assert!(long.message.contains("20,000"));
    let padded = format!("  {}  ", "x".repeat(MAX_TASK_CHARS));
    assert!(
        service.start(request(&padded, "dev:scripted")).is_ok(),
        "the bound is on the trimmed task"
    );
    assert_eq!(
        service.runs()[0].task.chars().count(),
        MAX_TASK_CHARS,
        "and the task is stored trimmed"
    );
    let wide = "\u{1F600}".repeat(MAX_TASK_CHARS);
    assert!(
        service.start(request(&wide, "dev:scripted")).is_ok(),
        "characters, not bytes"
    );

    let mut wrong_agent = request("hi", "dev:scripted");
    wrong_agent.agent = "model-advisor".into();
    assert_eq!(refused(wrong_agent).message, "That agent does not exist.");
    assert_eq!(
        refused(request("hi", "endpoint:nope")).message,
        "That model does not exist."
    );
    assert_eq!(
        refused(request("hi", "local")).message,
        "That model does not exist."
    );
    let unready = refused(request("hi", "endpoint:openai"));
    assert_eq!(unready.kind, RefusalKind::Invalid);
    assert_eq!(unready.message, "It is turned off in the model registry.");
    let anthropic = refused(request("hi", "endpoint:anthropic"));
    assert!(anthropic.message.contains("Messages API"));

    let plain = CoreService::new(config(dir.path(), MapEnv::new(), false));
    assert_eq!(
        plain
            .start(request("hi", "dev:scripted"))
            .unwrap_err()
            .message,
        "That model does not exist.",
        "without development mode there is no scripted model"
    );
}

#[test]
fn a_fourth_active_run_is_a_conflict_and_stop_ends_a_run_as_stopped() {
    let dir = TempDir::new("manager-conflict");
    let mut cfg = config(dir.path(), MapEnv::new(), true);
    cfg.dev_step_delay = Duration::from_secs(3_600);
    let service = CoreService::new(cfg);
    let ids: Vec<String> = (0..MAX_ACTIVE_RUNS)
        .map(|n| {
            service
                .start(request(&format!("run {n}"), "dev:scripted"))
                .unwrap()
                .id
        })
        .collect();
    assert_eq!(
        ids.iter().collect::<HashSet<_>>().len(),
        MAX_ACTIVE_RUNS,
        "ids are unique"
    );
    let fourth = service
        .start(request("one too many", "dev:scripted"))
        .unwrap_err();
    assert_eq!(fourth.kind, RefusalKind::Conflict);
    assert_eq!(
        fourth.message,
        "3 runs are already working. Wait for one to finish, or stop it."
    );
    assert_eq!(
        service.runs().len(),
        3,
        "a refused run leaves nothing behind"
    );

    service.stop(&ids[0]).unwrap();
    let events = events_of(&service, &ids[0], 0);
    assert_eq!(end_status(&events), EndStatus::Stopped);
    assert!(!events.iter().any(|e| matches!(
        e.kind,
        RunEventKind::Error { .. } | RunEventKind::Result { .. }
    )));
    let started: HashSet<&str> = events
        .iter()
        .filter_map(|e| match &e.kind {
            RunEventKind::SpanStart { span } => Some(span.id.as_str()),
            _ => None,
        })
        .collect();
    let ended: HashSet<&str> = events
        .iter()
        .filter_map(|e| match &e.kind {
            RunEventKind::SpanEnd { span } => Some(span.id.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        !started.is_empty() && started == ended,
        "every open span was closed by the stop"
    );
    let stopped = service.runs().into_iter().find(|r| r.id == ids[0]).unwrap();
    assert_eq!(stopped.status, RunStatus::Stopped);
    assert!(stopped.ended_at.is_some() && stopped.error.is_none());
    assert_eq!(
        service.stop(&ids[0]).unwrap_err().kind,
        RefusalKind::Conflict,
        "a run that is not running cannot be stopped"
    );
    assert_eq!(
        service.stop(&ids[0]).unwrap_err().message,
        "That run is not running."
    );

    // A slot is free again.
    let again = service
        .start(request("now there is room", "dev:scripted"))
        .unwrap();
    for id in ids[1..].iter().chain([&again.id]) {
        service.stop(id).unwrap();
        assert_eq!(end_status(&events_of(&service, id, 0)), EndStatus::Stopped);
    }
}

#[test]
fn stop_and_run_and_follow_refuse_unknown_and_malformed_ids_without_touching_the_disk() {
    let dir = TempDir::new("manager-ids");
    let service = dev_service(dir.path());
    for bad in [
        "",
        "nope",
        "../../../etc/passwd",
        "0123456789ABCDEF",
        "0123456789abcde",
        "0123456789abcdef0",
        "..\\..\\x",
    ] {
        assert_eq!(
            service.stop(bad).unwrap_err().kind,
            RefusalKind::NotFound,
            "{bad}"
        );
        assert_eq!(
            service.run(bad).unwrap_err().kind,
            RefusalKind::NotFound,
            "{bad}"
        );
        assert!(
            matches!(
                service.follow(bad, 0),
                Err(Refusal {
                    kind: RefusalKind::NotFound,
                    ..
                })
            ),
            "{bad}"
        );
    }
    assert_eq!(
        service.stop("0123456789abcdef").unwrap_err().kind,
        RefusalKind::NotFound,
        "well-formed but unknown"
    );
    assert!(
        !StateRoot::at(dir.path()).globals.exists(),
        "nothing was created on the disk"
    );
}

#[test]
fn a_run_that_was_working_when_lattice_closed_is_interrupted_on_the_next_start() {
    let dir = TempDir::new("manager-reload");
    let store = RunStore::new(StateRoot::at(dir.path()).runs_dir());
    let summary = |id: &str, status: RunStatus, created: f64| RunSummary {
        id: id.to_owned(),
        task: "an old task".into(),
        agent: ASSISTANT.into(),
        agent_label: "Lattice assistant".into(),
        model: "endpoint:ollama-local".into(),
        model_label: "Ollama (this machine)".into(),
        locality: Locality::Local,
        status,
        created_at: created,
        updated_at: created + 5.0,
        ended_at: (status != RunStatus::Running).then_some(created + 5.0),
        trace_id: "trace_00000000000000000000000000000000".into(),
        usage: None,
        output: None,
        error: None,
        spans: 3,
    };
    store
        .write_summary(
            &summary("1111111111111111", RunStatus::Running, 100.0),
            true,
        )
        .unwrap();
    store
        .write_summary(
            &summary("2222222222222222", RunStatus::Completed, 200.0),
            true,
        )
        .unwrap();
    let mut file = store.open_events("1111111111111111").unwrap();
    RunStore::append_event(
        &mut file,
        &RunEvent {
            seq: 1,
            at: 100.0,
            kind: RunEventKind::Agent {
                name: "Lattice assistant".into(),
            },
        },
    )
    .unwrap();
    RunStore::append_event(
        &mut file,
        &RunEvent {
            seq: 3,
            at: 101.0,
            kind: RunEventKind::Message {
                agent: "a".into(),
                text: "so far".into(),
            },
        },
    )
    .unwrap();
    drop(file);

    let service = dev_service(dir.path());
    let runs = service.runs();
    let ids: Vec<&str> = runs.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(
        ids,
        ["2222222222222222", "1111111111111111"],
        "newest first"
    );
    let old = &runs[1];
    assert_eq!(old.status, RunStatus::Interrupted);
    assert_eq!(
        old.error.as_deref(),
        Some("Ended when Lattice last closed.")
    );
    assert_eq!(
        old.ended_at,
        Some(105.0),
        "it ended at its last sign of life"
    );
    assert_eq!(runs[0].status, RunStatus::Completed);
    assert_eq!(
        store.read_summary("1111111111111111").unwrap().status,
        RunStatus::Interrupted,
        "and the record was rewritten"
    );

    // Events are read only when asked for.
    let detail = service.run("1111111111111111").unwrap();
    assert_eq!(detail.run.status, RunStatus::Interrupted);
    assert_eq!(detail.events.len(), 2, "the file is read as it is");
    assert_eq!(
        events_of(&service, "1111111111111111", 0).len(),
        2,
        "an interrupted run's stream ends, though it has no End"
    );
    assert!(service.stop("1111111111111111").is_err());
    // A third start finds it interrupted already and leaves it alone.
    let again = dev_service(dir.path());
    assert_eq!(again.runs()[1].status, RunStatus::Interrupted);
}

#[test]
fn at_most_two_hundred_runs_are_loaded_newest_first() {
    let dir = TempDir::new("manager-two-hundred");
    let store = RunStore::new(StateRoot::at(dir.path()).runs_dir());
    for n in 0..230u64 {
        let summary = RunSummary {
            id: format!("{n:016x}"),
            task: format!("task {n}"),
            agent: ASSISTANT.into(),
            agent_label: "Lattice assistant".into(),
            model: "endpoint:ollama-local".into(),
            model_label: "Ollama (this machine)".into(),
            locality: Locality::Local,
            status: RunStatus::Completed,
            created_at: 1_000.0 + n as f64,
            updated_at: 1_000.0 + n as f64,
            ended_at: Some(1_000.0 + n as f64),
            trace_id: "trace_00000000000000000000000000000000".into(),
            usage: None,
            output: None,
            error: None,
            spans: 0,
        };
        store.write_summary(&summary, false).unwrap();
    }
    let runs = dev_service(dir.path()).runs();
    assert_eq!(runs.len(), LOADED_RUNS);
    assert_eq!(runs[0].task, "task 229");
    assert_eq!(runs[LOADED_RUNS - 1].task, "task 30");
}

/// A model that answers each call with the next scripted step and is counted.
fn scripted(steps: Vec<ScriptedStep>) -> (Arc<ScriptedModel>, ModelFactory) {
    let model = Arc::new(ScriptedModel::new(steps));
    let handed = model.clone();
    let factory: ModelFactory = Arc::new(move |_config| Some(handed.clone() as Arc<dyn Model>));
    (model, factory)
}

fn with_factory(dir: &Path, env: MapEnv, factory: ModelFactory) -> CoreService {
    let mut cfg = config(dir, env, false);
    cfg.model_factory = Some(factory);
    CoreService::new(cfg)
}

fn write_registry(dir: &Path, json: &str) {
    let globals = StateRoot::at(dir).globals;
    std::fs::create_dir_all(&globals).unwrap();
    std::fs::write(globals.join("model_endpoints.json"), json).unwrap();
}

const REMOTE_REGISTRY: &str = r#"{"version": 1, "endpoints": [
    {"id": "hosted", "label": "Hosted model", "base_url": "https://models.example.test/v1", "model": "big-one", "api_key_name": "HOSTED_API_KEY"},
    {"id": "on-this-machine", "label": "A server on this machine", "base_url": "http://127.0.0.1:8080/v1", "model": "local-model"}
]}"#;

/// A model server on this machine that this runtime can talk to. The built-in
/// local row is the platform's managed llama.cpp server, which this runtime does
/// not start yet (ADR-0041), so a test that needs a model on this machine names
/// one of its own. The scripted factory means nothing is ever called at it.
const LOCAL_REGISTRY: &str = r#"{"version": 1, "endpoints": [
    {"id": "on-this-machine", "label": "A server on this machine", "base_url": "http://127.0.0.1:8080/v1", "model": "local-model"}
]}"#;
const LOCAL_MODEL: &str = "endpoint:on-this-machine";

#[test]
fn a_remote_model_and_a_secret_looking_task_end_refused_with_no_model_call() {
    let dir = TempDir::new("manager-refused");
    write_registry(dir.path(), REMOTE_REGISTRY);
    // Assembled from pieces: no source file holds a string a secret scanner would report.
    let secret = concat!("s", "k-abcdefghijklmnopqrstuvwxyz0123456789");
    let (model, factory) = scripted(vec![ScriptedStep::respond(vec![assistant_message(
        "should never be said",
    )])]);
    let env = MapEnv::new().with("HOSTED_API_KEY", "fixture-key-value");
    let service = with_factory(dir.path(), env, factory);
    let hosted = service
        .models()
        .into_iter()
        .find(|m| m.id == "endpoint:hosted")
        .unwrap();
    assert!(
        hosted.ready && hosted.locality == Locality::Remote,
        "{hosted:?}"
    );

    let summary = service
        .start(request(
            &format!("Please call the API with {secret} and tell me a joke"),
            "endpoint:hosted",
        ))
        .unwrap();
    assert_eq!(summary.locality, Locality::Remote);
    let events = events_of(&service, &summary.id, 0);
    assert_eq!(end_status(&events), EndStatus::Refused);
    assert_eq!(model.calls().len(), 0, "the model was never called");
    assert_eq!(model.remaining_steps(), 1);
    let refusal = events
        .iter()
        .find_map(|e| match &e.kind {
            RunEventKind::Guardrail { name, message } => Some((name.clone(), message.clone())),
            _ => None,
        })
        .unwrap();
    assert_eq!(refusal.0, "secrets_stay_local");
    assert_eq!(
        refusal.1,
        "The task contains what looks like a secret, and the chosen model is not on this machine."
    );
    assert_eq!(
        spans_of(&events, "generation").len(),
        0,
        "no generation span: nothing reached a model"
    );
    assert_eq!(
        spans_of(&events, "guardrail")[0].span_data["triggered"],
        true
    );
    assert!(!events.iter().any(|e| matches!(
        e.kind,
        RunEventKind::Result { .. } | RunEventKind::Error { .. }
    )));
    let summary = service.runs().remove(0);
    assert_eq!(summary.status, RunStatus::Refused);
    assert!(summary.error.is_none());

    // The refusal repeats nothing of the secret, and neither does any event (the
    // stored task loses it too: see `a_run_refused_by_a_guardrail_keeps_no_copy...`).
    let everything = format!("{events:?}");
    let refusal_text = format!("{refusal:?}");
    assert!(!refusal_text.contains(secret) && !refusal_text.contains(concat!("s", "k-abc")));
    assert!(!everything.contains("fixture-key-value") && !everything.contains("HOSTED_API_KEY"));

    // The same task on a model that is on this machine goes through. (One window
    // records in a directory at a time: the first service is closed first.)
    drop(service);
    let (local_model, factory) = scripted(vec![ScriptedStep::respond(vec![assistant_message(
        "Sure.",
    )])]);
    let service = with_factory(dir.path(), MapEnv::new(), factory);
    let local = service
        .start(request(&format!("Use {secret} please"), LOCAL_MODEL))
        .unwrap();
    assert_eq!(
        end_status(&events_of(&service, &local.id, 0)),
        EndStatus::Completed
    );
    assert_eq!(local_model.calls().len(), 1);
}

#[test]
fn a_remote_run_without_a_secret_calls_the_model_and_the_key_never_reaches_a_record() {
    let dir = TempDir::new("manager-remote");
    write_registry(dir.path(), REMOTE_REGISTRY);
    let (model, factory) = scripted(vec![
        ScriptedStep::respond(vec![assistant_message("Hello from the hosted model.")])
            .with_tokens(30, 7),
    ]);
    let env = MapEnv::new().with("HOSTED_API_KEY", "fixture-key-value-9876");
    let service = with_factory(dir.path(), env, factory);
    let summary = service
        .start(request("Say hello", "endpoint:hosted"))
        .unwrap();
    assert_eq!(summary.model_label, "Hosted model");
    let events = events_of(&service, &summary.id, 0);
    assert_eq!(end_status(&events), EndStatus::Completed);
    assert_eq!(model.calls().len(), 1);
    assert_eq!(
        model.calls()[0].system,
        "You are the Lattice assistant. Answer concisely. Use the current_time tool for the time and the date, and the calculate tool for arithmetic, instead of guessing. If the person asks which model to use, hand the conversation to the Model advisor."
    );
    let tool_names: Vec<String> = model.calls()[0]
        .tools
        .iter()
        .map(|t| t.name.clone())
        .collect();
    assert_eq!(
        tool_names,
        ["current_time", "calculate", "transfer_to_model_advisor"]
    );
    let guardrail = spans_of(&events, "guardrail")[0];
    assert_eq!(guardrail.span_data["triggered"], false);

    let mut everything = format!("{events:?}");
    everything.push_str(
        &std::fs::read_to_string(
            StateRoot::at(dir.path())
                .runs_dir()
                .join(format!("{}.run.json", summary.id)),
        )
        .unwrap(),
    );
    everything.push_str(&persisted_lines(dir.path(), &summary.id).join("\n"));
    everything.push_str(&format!("{:?} {:?}", service.runs(), service.models()));
    for forbidden in ["fixture-key-value-9876", "HOSTED_API_KEY"] {
        assert!(
            !everything.contains(forbidden),
            "{forbidden} reached a record or the interface"
        );
    }
}

#[test]
fn model_failures_become_one_sentence_each() {
    let cases: Vec<(ModelError, &str)> = vec![
        (
            ModelError::Connection("x".into()),
            "Could not reach the model at http://127.0.0.1:8080/v1.",
        ),
        (ModelError::Status(500), "The model server answered 500."),
        (ModelError::Timeout, "The model did not answer in time."),
        (
            ModelError::Protocol("bad chunk with details".into()),
            "The model server sent something this runtime could not read.",
        ),
        (
            ModelError::Failed("internal detail http://secret.example/x".into()),
            "The model could not be used.",
        ),
    ];
    for (error, sentence) in cases {
        let dir = TempDir::new("manager-model-error");
        let (_, factory) = scripted(vec![ScriptedStep::error(error.clone())]);
        write_registry(dir.path(), LOCAL_REGISTRY);
        let service = with_factory(dir.path(), MapEnv::new(), factory);
        let summary = service.start(request("hi", LOCAL_MODEL)).unwrap();
        let events = events_of(&service, &summary.id, 0);
        assert_eq!(end_status(&events), EndStatus::Failed, "{error}");
        let message = events
            .iter()
            .find_map(|e| match &e.kind {
                RunEventKind::Error { message } => Some(message.clone()),
                _ => None,
            })
            .unwrap();
        assert_eq!(message, sentence, "{error}");
        assert!(
            !message.contains("secret.example") && !message.contains("bad chunk"),
            "the model's own words are not repeated"
        );
        let listed = service.runs().remove(0);
        assert_eq!(
            (listed.status, listed.error.as_deref()),
            (RunStatus::Failed, Some(sentence))
        );
        let agent_span = spans_of(&events, "agent")[0];
        assert!(
            agent_span.error.is_some(),
            "the trace shows the failure on the agent span"
        );
    }
}

#[test]
fn a_spent_token_budget_fails_the_run_in_one_sentence_and_marks_the_model_call() {
    let dir = TempDir::new("manager-truncated");
    let (_, factory) = scripted(vec![ScriptedStep::error(ModelError::Truncated)]);
    write_registry(dir.path(), LOCAL_REGISTRY);
    let service = with_factory(dir.path(), MapEnv::new(), factory);
    let summary = service.start(request("think hard", LOCAL_MODEL)).unwrap();
    let events = events_of(&service, &summary.id, 0);
    assert_eq!(end_status(&events), EndStatus::Failed);
    let message = events
        .iter()
        .find_map(|e| match &e.kind {
            RunEventKind::Error { message } => Some(message.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(message, "The model ran out of tokens before answering.");
    // As in the SDK, a `ModelBehaviorError` is reported where it happened (the
    // model call), not as a generic failure of the agent.
    assert!(spans_of(&events, "generation")[0].error.is_some());
    assert!(spans_of(&events, "agent")[0].error.is_none());
}

#[test]
fn a_model_that_refuses_fails_the_run_with_what_it_said_bounded() {
    let dir = TempDir::new("manager-refusal");
    let long = "No. ".repeat(400);
    for (said, expected) in [
        (
            "I can't help with that.".to_owned(),
            "The model refused: I can't help with that.".to_owned(),
        ),
        (
            long.clone(),
            format!("The model refused: {}", crate::bound::cap_text(&long, 500)),
        ),
    ] {
        let (model, factory) = scripted(vec![ScriptedStep::respond(vec![OutputItem::Refusal {
            text: said,
        }])]);
        write_registry(dir.path(), LOCAL_REGISTRY);
        let service = with_factory(dir.path(), MapEnv::new(), factory);
        let summary = service.start(request("hi", LOCAL_MODEL)).unwrap();
        let events = events_of(&service, &summary.id, 0);
        assert_eq!(end_status(&events), EndStatus::Failed);
        let message = events
            .iter()
            .find_map(|e| match &e.kind {
                RunEventKind::Error { message } => Some(message.clone()),
                _ => None,
            })
            .unwrap();
        assert_eq!(message, expected);
        assert!(message.chars().count() <= "The model refused: ".len() + 500);
        assert_eq!(model.calls().len(), 1);
        let listed = service.runs().remove(0);
        assert_eq!(
            (listed.status, listed.error.as_deref()),
            (RunStatus::Failed, Some(expected.as_str()))
        );
        let agent_span = spans_of(&events, "agent")[0];
        assert_eq!(
            agent_span.error.as_ref().map(|e| e.message.as_str()),
            Some("Error in agent run"),
            "the SDK marks a refusal as a generic agent failure"
        );
        drop(service);
    }
}

#[test]
fn a_model_that_never_answers_finally_ends_at_the_turn_limit() {
    let dir = TempDir::new("manager-turns");
    let steps: Vec<ScriptedStep> = (0..12)
        .map(|n| {
            ScriptedStep::respond(vec![function_call(
                "current_time",
                "{}",
                format!("call_{n}"),
            )])
        })
        .collect();
    let (model, factory) = scripted(steps);
    write_registry(dir.path(), LOCAL_REGISTRY);
    let service = with_factory(dir.path(), MapEnv::new(), factory);
    let summary = service.start(request("loop", LOCAL_MODEL)).unwrap();
    let events = events_of(&service, &summary.id, 0);
    assert_eq!(end_status(&events), EndStatus::Failed);
    assert!(events.iter().any(|e| matches!(&e.kind, RunEventKind::Error { message } if message == "Stopped after 10 turns without a final answer.")));
    assert_eq!(model.calls().len(), 10);
}

#[test]
fn a_run_that_cannot_be_saved_still_runs_and_the_failure_is_counted() {
    let dir = TempDir::new("manager-unsaveable");
    // A file where the store's directory must go.
    let globals = StateRoot::at(dir.path()).globals;
    std::fs::create_dir_all(&globals).unwrap();
    std::fs::write(globals.join("lattice_native"), "in the way").unwrap();
    let service = dev_service(dir.path());
    let summary = service
        .start(request("Which model?", "dev:scripted"))
        .unwrap();
    let events = events_of(&service, &summary.id, 0);
    assert_eq!(
        end_status(&events),
        EndStatus::Completed,
        "the run does not depend on the disk"
    );
    assert!(service.persist_failures() > 0);
    let detail = service.run(&summary.id).unwrap();
    assert_eq!(
        detail.spans.len(),
        16,
        "and it can still be read, from memory"
    );
    assert!(globals.join("lattice_native").is_file());
}

/// A stub of an OpenAI-compatible server on 127.0.0.1: answers each connection
/// with `reply` (an SSE body) and records the request.
struct Stub {
    addr: SocketAddr,
    requests: Arc<Mutex<Vec<(String, String)>>>,
    stop: Arc<AtomicBool>,
}

impl Stub {
    fn start(sse: &'static str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (seen, stopping) = (requests.clone(), stop.clone());
        std::thread::spawn(move || {
            while !stopping.load(Ordering::Relaxed) {
                let Ok((mut socket, _)) = listener.accept() else {
                    std::thread::sleep(Duration::from_millis(5));
                    continue;
                };
                socket.set_nonblocking(false).unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut buffer = Vec::new();
                let mut chunk = [0u8; 4096];
                let head_end = loop {
                    if let Some(at) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
                        break at;
                    }
                    match socket.read(&mut chunk) {
                        Ok(0) | Err(_) => break buffer.len(),
                        Ok(n) => buffer.extend_from_slice(&chunk[..n]),
                    }
                };
                let head = String::from_utf8_lossy(&buffer[..head_end]).into_owned();
                let length = head
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                    })
                    .unwrap_or(0);
                let mut body = buffer.get(head_end + 4..).unwrap_or_default().to_vec();
                while body.len() < length {
                    match socket.read(&mut chunk) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => body.extend_from_slice(&chunk[..n]),
                    }
                }
                seen.lock()
                    .unwrap()
                    .push((head, String::from_utf8_lossy(&body).into_owned()));
                let _ = socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n");
                let _ = socket.write_all(sse.as_bytes());
                let _ = socket.flush();
            }
        });
        Self {
            addr,
            requests,
            stop,
        }
    }
}

impl Drop for Stub {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

const STUB_SSE: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"The answer \"}}]}\n\n\
data: {\"choices\":[{\"delta\":{\"content\":\"is 42.\"}}]}\n\n\
data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":5,\"total_tokens\":16}}\n\n\
data: [DONE]\n\n";

#[test]
fn a_registry_endpoint_runs_through_the_real_client_and_its_key_stays_out_of_the_records() {
    let dir = TempDir::new("manager-real-client");
    let stub = Stub::start(STUB_SSE);
    write_registry(
        dir.path(),
        &format!(
            r#"{{"version": 1, "endpoints": [{{"id": "stub", "label": "Stub server", "base_url": "http://{}/v1", "model": "stub-model", "api_key_name": "STUB_API_KEY"}}]}}"#,
            stub.addr
        ),
    );
    let env = MapEnv::new().with("STUB_API_KEY", "fixture-secret-value-4711");
    let service = CoreService::new(config(dir.path(), env, false));
    let choice = service
        .models()
        .into_iter()
        .find(|m| m.id == "endpoint:stub")
        .unwrap();
    assert!(
        choice.ready && choice.locality == Locality::Local,
        "a loopback address is on this machine: {choice:?}"
    );

    let summary = service
        .start(request("What is 6 * 7?", "endpoint:stub"))
        .unwrap();
    let events = events_of(&service, &summary.id, 0);
    assert_eq!(end_status(&events), EndStatus::Completed);
    let streamed: String = events
        .iter()
        .filter_map(|e| match &e.kind {
            RunEventKind::Delta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(streamed, "The answer is 42.");
    match &events[events.len() - 2].kind {
        RunEventKind::Result {
            output,
            usage,
            turns,
            ..
        } => {
            assert_eq!(output, "The answer is 42.");
            assert_eq!(*turns, 1);
            assert_eq!(
                usage.map(|u| (u.input_tokens, u.output_tokens, u.total_tokens)),
                Some((11, 5, 16))
            );
        }
        other => panic!("{other:?}"),
    }
    let generation = spans_of(&events, "generation");
    assert_eq!(generation[0].span_data["model"], "stub-model");
    assert_eq!(generation[0].span_data["usage"]["total_tokens"], 16);

    // The stub saw the key, as a bearer token, and the request the SDK builds.
    let seen = stub.requests.lock().unwrap().clone();
    assert_eq!(seen.len(), 1);
    assert!(
        seen[0].0.starts_with("POST /v1/chat/completions"),
        "{}",
        seen[0].0
    );
    assert!(
        seen[0]
            .0
            .to_ascii_lowercase()
            .contains("authorization: bearer fixture-secret-value-4711")
    );
    let body: Value = serde_json::from_str(&seen[0].1).unwrap();
    assert_eq!(body["model"], "stub-model");
    assert_eq!(body["stream"], true);
    assert_eq!(
        body["messages"][1],
        json!({"role": "user", "content": "What is 6 * 7?"})
    );

    // And nothing that was recorded holds it.
    let mut everything = format!("{events:?}");
    everything.push_str(
        &std::fs::read_to_string(
            StateRoot::at(dir.path())
                .runs_dir()
                .join(format!("{}.run.json", summary.id)),
        )
        .unwrap(),
    );
    everything.push_str(&persisted_lines(dir.path(), &summary.id).join("\n"));
    for forbidden in ["fixture-secret-value-4711", "STUB_API_KEY"] {
        assert!(!everything.contains(forbidden), "{forbidden} was recorded");
    }
}

#[test]
fn a_server_that_is_not_there_fails_the_run_with_the_model_s_address() {
    let dir = TempDir::new("manager-unreachable");
    let port = {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap().port()
    };
    write_registry(
        dir.path(),
        &format!(
            r#"{{"version": 1, "endpoints": [{{"id": "gone", "label": "Gone", "base_url": "http://127.0.0.1:{port}/v1/chat/completions", "model": "m"}}]}}"#
        ),
    );
    let service = CoreService::new(config(dir.path(), MapEnv::new(), false));
    let summary = service.start(request("hi", "endpoint:gone")).unwrap();
    let events = events_of(&service, &summary.id, 0);
    assert_eq!(end_status(&events), EndStatus::Failed);
    let message = events
        .iter()
        .find_map(|e| match &e.kind {
            RunEventKind::Error { message } => Some(message.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        message,
        format!("Could not reach the model at http://127.0.0.1:{port}/v1."),
        "the /chat/completions suffix was stripped from the address"
    );
}

#[test]
fn status_agents_and_models_describe_the_service() {
    let dir = TempDir::new("manager-describe");
    let service = dev_service(dir.path());
    let status = service.status();
    assert_eq!(
        status.runtime,
        format!(
            "lattice-agents {} (a port of openai-agents 0.22.3)",
            env!("CARGO_PKG_VERSION")
        )
    );
    assert_eq!(status.traces, "this machine");
    assert!(status.refusal.is_none());
    let agents = service.agents();
    assert_eq!(agents.len(), 1);
    assert_eq!(agents[0].id, ASSISTANT);
    let models = service.models();
    assert_eq!(models[0].id, "dev:scripted");
    // The managed llama.cpp row is listed, on this machine, and says why this
    // runtime cannot use it yet (ADR-0041).
    assert!(models.iter().any(|m| m.id == "endpoint:llamacpp-local"
        && !m.ready
        && m.locality == Locality::Local
        && m.refusal.is_some()));
    let plain = CoreService::new(config(dir.path(), MapEnv::new(), false));
    assert!(plain.models().iter().all(|m| m.id != "dev:scripted"));
    assert!(service.runs().is_empty());
}

#[test]
fn a_service_can_be_dropped_inside_a_runtime_and_with_a_run_working() {
    let dir = TempDir::new("manager-drop");
    let mut cfg = config(dir.path(), MapEnv::new(), true);
    cfg.dev_step_delay = Duration::from_secs(3_600);
    let id = {
        let service = CoreService::new(cfg);
        service
            .start(request("still working", "dev:scripted"))
            .unwrap()
            .id
    };
    // The run was left `running` on disk: the next start calls it interrupted.
    let store = RunStore::new(StateRoot::at(dir.path()).runs_dir());
    assert_eq!(store.read_summary(&id).unwrap().status, RunStatus::Running);
    assert_eq!(
        dev_service(dir.path()).runs()[0].status,
        RunStatus::Interrupted
    );
    // Dropping inside an async context is fine too.
    within("dropping a service inside a runtime", 30, || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime.block_on(async {
            let dir = TempDir::new("manager-drop-inner");
            let service = dev_service(dir.path());
            drop(service);
        });
    });
}

#[test]
fn run_ids_are_sixteen_lowercase_hex_characters_and_do_not_repeat() {
    let ids: HashSet<String> = (0..20_000).map(|_| new_run_id()).collect();
    assert_eq!(ids.len(), 20_000);
    assert!(ids.iter().all(|id| is_run_id(id)));
}

// ---- the bounds, on an entry directly

fn entry_over(dir: &Path, clock: Clock) -> (Arc<RunEntry>, Arc<Shared>) {
    let shared = Arc::new(Shared {
        store: RunStore::new(StateRoot::at(dir).runs_dir()),
        clock: clock.clone(),
        persist_failures: AtomicU64::new(0),
        writer: AtomicBool::new(true),
    });
    let now = clock();
    let summary = RunSummary {
        id: "00000000000000ee".into(),
        task: "t".into(),
        agent: ASSISTANT.into(),
        agent_label: "Lattice assistant".into(),
        model: "dev:scripted".into(),
        model_label: "Dev".into(),
        locality: Locality::Local,
        status: RunStatus::Running,
        created_at: now,
        updated_at: now,
        ended_at: None,
        trace_id: "trace_00000000000000000000000000000000".into(),
        usage: None,
        output: None,
        error: None,
        spans: 0,
    };
    (
        RunEntry::new(shared.clone(), summary, true, true, now),
        shared,
    )
}

fn fixed_clock() -> Clock {
    Arc::new(|| 1_790_000_000.0)
}

#[test]
fn persisted_events_are_capped_and_the_events_that_end_a_run_are_always_kept() {
    let dir = TempDir::new("manager-cap");
    let (entry, shared) = entry_over(dir.path(), fixed_clock());
    for n in 0..(MAX_PERSISTED_EVENTS + 500) {
        entry.append(RunEventKind::Agent {
            name: format!("a{n}"),
        });
    }
    entry.append(RunEventKind::Result {
        output: "done".into(),
        usage: None,
        turns: 1,
        last_agent: "a".into(),
    });
    entry.append(RunEventKind::Error {
        message: "e".into(),
    });
    entry.append(RunEventKind::Guardrail {
        name: "g".into(),
        message: "m".into(),
    });
    entry.append(RunEventKind::End {
        status: EndStatus::Failed,
    });
    entry.append(RunEventKind::Agent {
        name: "after the end".into(),
    });
    let state = lock(&entry.state);
    assert_eq!(state.dropped_events, 500);
    assert_eq!(state.events.len(), MAX_PERSISTED_EVENTS + 4);
    assert!(matches!(
        state.events.last().unwrap().kind,
        RunEventKind::End { .. }
    ));
    assert!(
        state
            .events
            .windows(2)
            .all(|pair| pair[1].seq == pair[0].seq + 1),
        "dropped events take no seq"
    );
    drop(state);
    let on_disk = shared.store.read_events(&entry.id);
    assert_eq!(on_disk.len(), MAX_PERSISTED_EVENTS + 4);
    assert!(matches!(
        on_disk.last().unwrap().kind,
        RunEventKind::End {
            status: EndStatus::Failed
        }
    ));
    assert_eq!(
        shared.store.read_summary(&entry.id).unwrap().status,
        RunStatus::Failed
    );
}

#[test]
fn live_deltas_are_capped_by_count_and_by_size_and_never_written() {
    let dir = TempDir::new("manager-deltas");
    let (entry, shared) = entry_over(dir.path(), fixed_clock());
    for _ in 0..(MAX_LIVE_DELTAS + 100) {
        entry.append(RunEventKind::Delta {
            agent: "a".into(),
            text: "x".into(),
        });
    }
    {
        let state = lock(&entry.state);
        assert_eq!(state.live_deltas, MAX_LIVE_DELTAS);
        assert_eq!(state.dropped_deltas, 100);
        assert_eq!(state.events.len(), MAX_LIVE_DELTAS);
    }
    entry.append(RunEventKind::Message {
        agent: "a".into(),
        text: "whole".into(),
    });
    entry.append(RunEventKind::End {
        status: EndStatus::Completed,
    });
    assert_eq!(
        shared.store.read_events(&entry.id).len(),
        2,
        "only the message and the end are on disk"
    );

    let dir = TempDir::new("manager-delta-size");
    let (entry, _) = entry_over(dir.path(), fixed_clock());
    let big = "y".repeat(1024 * 1024);
    for _ in 0..10 {
        entry.append(RunEventKind::Delta {
            agent: "a".into(),
            text: big.clone(),
        });
    }
    let state = lock(&entry.state);
    assert_eq!(
        state.live_deltas,
        MAX_LIVE_DELTA_CHARS / (1024 * 1024),
        "the text budget stops the deltas"
    );
    assert_eq!(state.dropped_deltas, 10 - state.live_deltas as u64);
}

#[test]
fn stream_events_map_one_to_one_and_are_bounded() {
    use lattice_agents::RunItemName;
    let huge = "z".repeat(MAX_TEXT_CHARS + 100);
    let mapped = |event: StreamEvent| map_stream_event(event).expect("an SDK event is recorded");
    match mapped(StreamEvent::RawTextDelta {
        agent: "A".into(),
        delta: huge.clone(),
    }) {
        RunEventKind::Delta { agent, text } => {
            assert_eq!(agent, "A");
            assert_eq!(text.chars().count(), MAX_TEXT_CHARS);
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(
        mapped(StreamEvent::AgentUpdated { agent: "A".into() }),
        RunEventKind::Agent { name: "A".into() }
    );
    let item = |name, item| StreamEvent::RunItem { name, item };
    assert_eq!(
        mapped(item(
            RunItemName::MessageOutputCreated,
            RunItem::MessageOutput {
                agent: "A".into(),
                text: "hi".into()
            }
        )),
        RunEventKind::Message {
            agent: "A".into(),
            text: "hi".into()
        }
    );
    assert_eq!(
        mapped(item(
            RunItemName::ToolCalled,
            RunItem::ToolCall {
                agent: "A".into(),
                call_id: "c".into(),
                name: "n".into(),
                arguments: huge.clone()
            }
        )),
        RunEventKind::ToolCall {
            agent: "A".into(),
            name: "n".into(),
            call_id: "c".into(),
            arguments: cap_text(&huge, MAX_TEXT_CHARS)
        }
    );
    assert_eq!(
        mapped(item(
            RunItemName::ToolOutput,
            RunItem::ToolOutput {
                agent: "A".into(),
                call_id: "c".into(),
                output: "o".into()
            }
        )),
        RunEventKind::ToolOutput {
            agent: "A".into(),
            call_id: "c".into(),
            output: "o".into()
        }
    );
    assert_eq!(
        mapped(item(
            RunItemName::HandoffRequested,
            RunItem::HandoffCall {
                agent: "A".into(),
                call_id: "c".into(),
                tool_name: "transfer_to_b".into()
            }
        )),
        RunEventKind::HandoffRequested { agent: "A".into() }
    );
    assert_eq!(
        mapped(item(
            RunItemName::HandoffOccurred,
            RunItem::HandoffOutput {
                agent: "A".into(),
                call_id: "c".into(),
                source_agent: "A".into(),
                target_agent: "B".into(),
                output: "{}".into()
            }
        )),
        RunEventKind::Handoff {
            from: "A".into(),
            to: "B".into()
        }
    );
    assert_eq!(
        mapped(item(
            RunItemName::ReasoningItemCreated,
            RunItem::Reasoning {
                agent: "A".into(),
                text: "hm".into()
            }
        )),
        RunEventKind::Reasoning {
            agent: "A".into(),
            text: "hm".into()
        }
    );
}

#[test]
fn a_native_only_stream_event_is_not_a_run_event() {
    // The run manager configures no session, so its runs never send this; if
    // one did, it would leave no record.
    assert_eq!(
        map_stream_event(StreamEvent::SessionWriteFailed {
            message: "the disk is full".into()
        }),
        None
    );
    // Nor does it configure approvals.
    assert_eq!(
        map_stream_event(StreamEvent::ToolApprovalRequested {
            agent: "A".into(),
            call_id: "c".into(),
            tool: "t".into(),
            arguments: json!({}),
        }),
        None
    );
    assert_eq!(
        map_stream_event(StreamEvent::ToolApprovalResolved {
            agent: "A".into(),
            call_id: "c".into(),
            approved: true,
        }),
        None
    );
    // Nor does it steer.
    assert_eq!(
        map_stream_event(StreamEvent::Steered {
            agent: "A".into(),
            text: "t".into(),
        }),
        None
    );
}

#[test]
fn a_span_is_bounded_before_it_is_recorded() {
    let huge = "q".repeat(MAX_TEXT_CHARS * 3);
    let span = SpanRecord {
        order: 1,
        id: "span_x".into(),
        trace_id: "trace_x".into(),
        parent_id: None,
        started_at: "2026-09-30T05:21:05.123456+00:00".into(),
        ended_at: None,
        span_data: json!({"type": "generation", "input": [{"role": "user", "content": huge.clone()}], "output": [huge.clone()], "model": "m"}),
        error: Some(lattice_protocol::SpanError {
            message: huge.clone(),
            data: Some(json!({"error": huge})),
        }),
    };
    let bounded = bounded_span(&span);
    assert_eq!(
        bounded.span_data["input"][0]["content"]
            .as_str()
            .unwrap()
            .chars()
            .count(),
        MAX_TEXT_CHARS
    );
    assert_eq!(bounded.span_data["model"], "m");
    assert_eq!(
        bounded.error.as_ref().unwrap().message.chars().count(),
        MAX_TEXT_CHARS
    );
    assert_eq!(
        bounded.error.unwrap().data.unwrap()["error"]
            .as_str()
            .unwrap()
            .chars()
            .count(),
        MAX_TEXT_CHARS
    );
    assert_eq!(bounded.order, 1);
}

#[test]
fn the_summary_is_written_at_most_once_a_second_and_always_at_the_end() {
    let dir = TempDir::new("manager-throttle");
    let time = Arc::new(Mutex::new(1_790_000_000.0f64));
    let clock: Clock = {
        let time = time.clone();
        Arc::new(move || *time.lock().unwrap())
    };
    let (entry, shared) = entry_over(dir.path(), clock);
    shared.store.write_summary(&entry.summary(), true).unwrap();
    let span_end = || RunEventKind::SpanEnd {
        span: SpanRecord {
            order: 1,
            id: "span_a".into(),
            trace_id: "t".into(),
            parent_id: None,
            started_at: "x".into(),
            ended_at: Some("y".into()),
            span_data: json!({"type": "function"}),
            error: None,
        },
    };
    let on_disk = || shared.store.read_summary(&entry.id).unwrap();
    entry.append(span_end());
    entry.append(span_end());
    assert_eq!(
        on_disk().spans,
        0,
        "within a second of the last write: not written again"
    );
    assert_eq!(
        entry.summary().spans,
        2,
        "but the summary in memory is current"
    );
    *time.lock().unwrap() += 1.5;
    entry.append(span_end());
    assert_eq!(on_disk().spans, 3, "a second later it is");
    *time.lock().unwrap() += 0.1;
    entry.append(span_end());
    assert_eq!(on_disk().spans, 3);
    entry.append(RunEventKind::Result {
        output: "x".repeat(SUMMARY_OUTPUT_CHARS + 50),
        usage: Some(lattice_protocol::Usage {
            requests: 1,
            input_tokens: 2,
            output_tokens: 3,
            total_tokens: 5,
        }),
        turns: 1,
        last_agent: "a".into(),
    });
    entry.append(RunEventKind::End {
        status: EndStatus::Completed,
    });
    let last = on_disk();
    assert_eq!(
        (last.status, last.spans),
        (RunStatus::Completed, 4),
        "the end is always written"
    );
    assert_eq!(
        last.output.as_deref().unwrap().chars().count(),
        SUMMARY_OUTPUT_CHARS,
        "the list carries a bounded output"
    );
    assert_eq!(last.usage.unwrap().total_tokens, 5);
    assert!(last.ended_at.is_some());
}

#[test]
fn only_the_sixteen_most_recent_finished_runs_keep_their_events_in_memory() {
    let dir = TempDir::new("manager-evict");
    let service = dev_service(dir.path());
    let mut ids = Vec::new();
    for n in 0..(CACHED_FINISHED_RUNS + 3) {
        let id = service
            .start(request(&format!("run {n}"), "dev:scripted"))
            .unwrap()
            .id;
        events_of(&service, &id, 0);
        ids.push(id);
        // Keep at most three working at once.
        while service
            .runs()
            .iter()
            .filter(|r| r.status.is_active())
            .count()
            >= MAX_ACTIVE_RUNS
        {
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    // Wait for every run to finish and be noted.
    for id in &ids {
        events_of(&service, id, 0);
    }
    let held: Vec<bool> = ids
        .iter()
        .map(|id| lock(&service.inner.find(id).unwrap().state).loaded)
        .collect();
    assert!(
        held.iter().filter(|loaded| **loaded).count() <= CACHED_FINISHED_RUNS,
        "{held:?}"
    );
    let evicted = ids
        .iter()
        .zip(&held)
        .find(|(_, loaded)| !**loaded)
        .map(|(id, _)| id.clone())
        .expect("an early run was evicted");
    // An evicted run reads back from disk: its events without the deltas.
    let detail = service.run(&evicted).unwrap();
    assert!(
        !detail.events.is_empty()
            && matches!(detail.events.last().unwrap().kind, RunEventKind::End { .. })
    );
    assert_eq!(detail.spans.len(), 16);
    let again = events_of(&service, &evicted, 0);
    assert_eq!(
        again, detail.events,
        "what is read back is the recorded events"
    );
}

#[test]
fn a_run_whose_events_could_not_be_saved_is_never_evicted_from_memory() {
    let dir = TempDir::new("manager-unsaved-evict");
    let globals = StateRoot::at(dir.path()).globals;
    std::fs::create_dir_all(&globals).unwrap();
    std::fs::write(globals.join("lattice_native"), "in the way").unwrap();
    let service = dev_service(dir.path());
    let mut ids = Vec::new();
    for n in 0..(CACHED_FINISHED_RUNS + 4) {
        let id = service
            .start(request(&format!("run {n}"), "dev:scripted"))
            .unwrap()
            .id;
        events_of(&service, &id, 0);
        ids.push(id);
    }
    for id in &ids {
        // Their events exist nowhere else, so every one of them is still here.
        assert!(
            lock(&service.inner.find(id).unwrap().state).loaded,
            "{id} was evicted"
        );
        assert_eq!(service.run(id).unwrap().spans.len(), 16);
    }
    assert!(service.persist_failures() > 0);
}

#[test]
fn runs_execute_on_two_worker_threads_named_lattice_runs() {
    let dir = TempDir::new("manager-runtime");
    let service = dev_service(dir.path());
    let handle = service.inner.handle.clone().expect("the runtime started");
    assert_eq!(handle.metrics().num_workers(), 2);
    let spawned = handle.spawn(async { std::thread::current().name().map(str::to_owned) });
    let name = within("a task on the runtime", 30, move || {
        handle.block_on(spawned).unwrap()
    });
    assert_eq!(name.as_deref(), Some("lattice-runs"));
}

/// The runtime a shell would share with the service: two workers, named so a
/// test can tell its threads from the service's own `lattice-runs`.
fn shell_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("shell-runtime")
        .enable_all()
        .build()
        .unwrap()
}

#[test]
fn a_service_given_a_handle_builds_no_runtime_and_runs_on_the_shared_one() {
    let shell = shell_runtime();
    let dir = TempDir::new("manager-shared-runtime");
    let service = CoreService::with_handle(
        config(dir.path(), MapEnv::new(), true),
        shell.handle().clone(),
    );
    assert!(service.inner.runtime.is_none());
    assert!(service.status().refusal.is_none());
    let handle = service.inner.handle.clone().expect("the shared handle");
    let spawned = handle.spawn(async { std::thread::current().name().map(str::to_owned) });
    let name = within("a task on the shared runtime", 30, move || {
        handle.block_on(spawned).unwrap()
    });
    assert_eq!(name.as_deref(), Some("shell-runtime"));

    let summary = service
        .start(request("What is 6 * 7?", "dev:scripted"))
        .unwrap();
    let events = events_of(&service, &summary.id, 0);
    assert_eq!(end_status(&events), EndStatus::Completed);
    // A finished run leaves nothing behind on the runtime it borrowed.
    eventually("the finished run's tasks end", 30, || {
        (shell.metrics().num_alive_tasks() == 0).then_some(())
    });
}

#[test]
fn dropping_a_service_on_a_shared_runtime_ends_its_runs_and_leaves_the_runtime_running() {
    let shell = shell_runtime();
    let dir = TempDir::new("manager-shared-drop");
    let mut cfg = config(dir.path(), MapEnv::new(), true);
    cfg.dev_step_delay = Duration::from_secs(3_600);
    let service = CoreService::with_handle(cfg, shell.handle().clone());
    let id = service
        .start(request("still working", "dev:scripted"))
        .unwrap()
        .id;
    // The run and its supervisor are working on the shell's runtime.
    eventually("the run's tasks are alive", 30, || {
        (shell.metrics().num_alive_tasks() >= 2).then_some(())
    });
    drop(service);
    // They end with the service, as an owned runtime's would.
    eventually("the dropped service's tasks end", 30, || {
        (shell.metrics().num_alive_tasks() == 0).then_some(())
    });
    // The runtime itself is not the service's: it still runs work.
    let handle = shell.handle().clone();
    let answer = within("a task after the drop", 30, move || {
        handle.block_on(handle.spawn(async { 6 * 7 })).unwrap()
    });
    assert_eq!(answer, 42);
    // As with an owned runtime: left `running` on disk, interrupted next time.
    assert_eq!(
        runs_store(dir.path()).read_summary(&id).unwrap().status,
        RunStatus::Running
    );
    assert_eq!(
        dev_service(dir.path()).runs()[0].status,
        RunStatus::Interrupted
    );
}

// ---- what an adversarial review found (the native Lattice's review)

/// Probe until `probe` answers, or fail after `seconds`.
fn eventually<T>(what: &str, seconds: u64, mut probe: impl FnMut() -> Option<T>) -> T {
    let deadline = std::time::Instant::now() + Duration::from_secs(seconds);
    loop {
        if let Some(found) = probe() {
            return found;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "{what} did not happen within {seconds} s"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn runs_store(dir: &Path) -> RunStore {
    RunStore::new(StateRoot::at(dir).runs_dir())
}

fn follow_to_the_end(service: &CoreService, id: &str) -> Vec<RunEvent> {
    events_of(service, id, 0)
}

const SECRET: &str = concat!("s", "k-abcdefghijklmnopqrstuvwxyz0123456789");
const REDACTED: &str = "[redacted: looks like a secret]";

#[test]
fn a_saved_ollama_row_says_where_it_pointed_and_is_refused_before_any_client_is_built() {
    // Ollama is retired (ADR-0041). With no address of its own the row pointed at
    // OLLAMA_BASE_URL, here a remote one, and the window must still say so.
    let dir = TempDir::new("manager-ollama-retired");
    write_registry(
        dir.path(),
        r#"{"version": 1, "endpoints": [{"id": "mine", "label": "Mine", "kind": "ollama", "model": "m"}]}"#,
    );
    let built: Arc<Mutex<Vec<String>>> = Arc::default();
    let factory: ModelFactory = {
        let built = built.clone();
        Arc::new(move |config| {
            lock(&built).push(config.base_url.clone());
            None
        })
    };
    let env = MapEnv::new().with("OLLAMA_BASE_URL", "http://192.168.1.50:11434");
    let service = with_factory(dir.path(), env, factory);
    let mine = service
        .models()
        .into_iter()
        .find(|m| m.id == "endpoint:mine")
        .unwrap();
    assert_eq!(
        mine.locality,
        Locality::Remote,
        "the window must not say this stays on the machine"
    );
    assert!(!mine.ready);

    let refusal = service
        .start(request(
            &format!("use {SECRET} to call the API"),
            "endpoint:mine",
        ))
        .unwrap_err();
    assert_eq!(
        refusal.message,
        "Ollama is retired (ADR-0041): use a llama.cpp model instead."
    );
    assert!(
        lock(&built).is_empty(),
        "no client was built for a retired row"
    );
    assert!(service.runs().is_empty(), "nothing was recorded");
}

#[test]
fn a_connection_failure_names_the_address_without_a_query_or_credentials() {
    for (given, sentence) in [
        (
            "http://user:pw@h.example.test:11434/v1?token=abc#frag",
            "Could not reach the model at http://h.example.test:11434/v1.",
        ),
        (
            "https://h.example.test/v1?key=abc",
            "Could not reach the model at https://h.example.test/v1.",
        ),
        ("not a url", "Could not reach the model."),
    ] {
        let (events, status) = ending(
            Ok(Err(RunError::Model(ModelError::Connection("x".into())))),
            Some(given),
        );
        assert_eq!(status, EndStatus::Failed);
        match events.as_slice() {
            [RunEventKind::Error { message }] => assert_eq!(message, sentence, "{given}"),
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn a_second_window_reads_the_first_windows_runs_and_never_rewrites_them() {
    let dir = TempDir::new("manager-two-windows");
    let mut first = config(dir.path(), MapEnv::new(), true);
    first.dev_step_delay = Duration::from_millis(350);
    let a = CoreService::new(first);
    let id = a
        .start(request("What is 6 * 7?", "dev:scripted"))
        .unwrap()
        .id;

    // A second window starts on the same state while the run is working.
    let b = dev_service(dir.path());
    let store = runs_store(dir.path());

    // It sees the run as it is, and did not end it, in memory or on disk.
    let seen = b.runs();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].id, id);
    assert_eq!(
        seen[0].status,
        RunStatus::Running,
        "the second window marked a live run interrupted"
    );
    assert_eq!(
        store.read_summary(&id).unwrap().status,
        RunStatus::Running,
        "the second window rewrote the first window's record"
    );

    // It cannot record runs, and says why, in the status and in the refusal.
    let said = "Another Lattice window is recording runs. Start runs there, or close it.";
    let refusal = b.start(request("hi", "dev:scripted")).unwrap_err();
    assert_eq!(refusal.kind, RefusalKind::Unavailable);
    assert_eq!(refusal.message, said);
    assert_eq!(b.status().refusal.as_deref(), Some(said));
    assert_eq!(
        a.status().refusal,
        None,
        "the window that records has nothing to say"
    );
    assert_eq!(
        b.stop(&id).unwrap_err().kind,
        RefusalKind::Conflict,
        "a run that belongs to another window is stopped there"
    );

    // It follows the same run to its end...
    let followed = {
        let b = b.clone();
        let id = id.clone();
        within("a second window following a run", 60, move || {
            let batches: Vec<Vec<RunEvent>> = block_on(b.follow(&id, 0).unwrap().collect());
            batches.into_iter().flatten().collect::<Vec<RunEvent>>()
        })
    };
    assert_eq!(end_status(&followed), EndStatus::Completed);
    assert_eq!(followed.first().map(|e| e.seq), Some(1));

    // ...and what the first window left on disk is what it finished with.
    assert_eq!(
        end_status(&follow_to_the_end(&a, &id)),
        EndStatus::Completed
    );
    let on_disk = eventually("the final summary", 30, || {
        store
            .read_summary(&id)
            .filter(|summary| summary.status != RunStatus::Running)
    });
    assert_eq!(on_disk.status, RunStatus::Completed);
    let caught_up = eventually("the second window listing the finished run", 30, || {
        b.runs()
            .into_iter()
            .find(|run| run.id == id && run.status != RunStatus::Running)
    });
    assert_eq!(caught_up.status, RunStatus::Completed);
    assert_eq!(b.persist_failures(), 0);
}

#[test]
fn a_second_window_sees_runs_the_first_started_after_it_did() {
    let dir = TempDir::new("manager-two-windows-later");
    let a = dev_service(dir.path());
    let first = a.start(request("one", "dev:scripted")).unwrap().id;
    follow_to_the_end(&a, &first);
    let b = dev_service(dir.path());
    assert_eq!(b.runs().len(), 1);
    let second = a.start(request("two", "dev:scripted")).unwrap().id;
    follow_to_the_end(&a, &second);
    let listed = eventually("the new run in the second window", 15, || {
        let runs = b.runs();
        (runs.len() == 2).then_some(runs)
    });
    assert_eq!(listed[0].id, second, "newest first");
    assert_eq!(listed[0].task, "two");
    // What the second window reads is what the first recorded: the events it
    // keeps on disk, which are all but the live text deltas.
    let recorded: Vec<u64> = follow_to_the_end(&a, &second)
        .iter()
        .filter(|e| !matches!(e.kind, RunEventKind::Delta { .. }))
        .map(|e| e.seq)
        .collect();
    let read: Vec<u64> = follow_to_the_end(&b, &second)
        .iter()
        .map(|e| e.seq)
        .collect();
    assert_eq!(read, recorded);
}

#[test]
fn a_second_window_takes_over_when_the_first_closes() {
    let dir = TempDir::new("manager-two-windows-takeover");
    let mut first = config(dir.path(), MapEnv::new(), true);
    first.dev_step_delay = Duration::from_secs(3_600);
    let a = CoreService::new(first);
    let abandoned = a.start(request("never ends", "dev:scripted")).unwrap().id;
    let b = dev_service(dir.path());
    assert!(
        b.start(request("hi", "dev:scripted")).is_err(),
        "the first window still holds the directory"
    );
    assert!(b.status().refusal.is_some());

    drop(a);

    let second = b
        .start(request("now it is mine", "dev:scripted"))
        .expect("the directory is free once the first window is gone");
    assert_eq!(
        end_status(&follow_to_the_end(&b, &second.id)),
        EndStatus::Completed
    );
    assert_eq!(b.status().refusal, None);
    // The window that took over ends what the window that left had not.
    let old = b.runs().into_iter().find(|r| r.id == abandoned).unwrap();
    assert_eq!(old.status, RunStatus::Interrupted);
    assert_eq!(
        runs_store(dir.path())
            .read_summary(&abandoned)
            .unwrap()
            .status,
        RunStatus::Interrupted
    );
}

#[test]
fn a_stop_that_arrives_before_the_run_has_its_control_is_not_lost() {
    // The run is listed from the moment it is inserted, and its control exists a
    // durable write of its summary later. Stop it the moment it is listed.
    for round in 0..15 {
        let dir = TempDir::new("manager-stop-race");
        let mut cfg = config(dir.path(), MapEnv::new(), true);
        cfg.dev_step_delay = Duration::from_millis(400);
        let service = CoreService::new(cfg);
        let watcher = {
            let service = service.clone();
            std::thread::spawn(move || {
                let deadline = std::time::Instant::now() + Duration::from_secs(30);
                loop {
                    if let Some(run) = service.runs().into_iter().find(|r| r.status.is_active()) {
                        return (run.id.clone(), service.stop(&run.id));
                    }
                    assert!(std::time::Instant::now() < deadline, "no run appeared");
                    std::thread::yield_now();
                }
            })
        };
        let summary = service.start(request("race", "dev:scripted")).unwrap();
        let (id, stopped) = watcher.join().unwrap();
        assert_eq!(id, summary.id);
        if let Err(refusal) = stopped {
            panic!(
                "round {round}: stop() said `{}` about a run that was still starting",
                refusal.message
            );
        }
        assert_eq!(
            end_status(&events_of(&service, &id, 0)),
            EndStatus::Stopped,
            "round {round}"
        );
    }
}

#[test]
fn a_hostile_events_file_cannot_panic_the_service() {
    let dir = TempDir::new("manager-hostile-seq");
    let store = runs_store(dir.path());
    let line = |seq: u64| {
        serde_json::to_string(&RunEvent {
            seq,
            at: 1.0,
            kind: RunEventKind::Agent { name: "a".into() },
        })
        .unwrap()
    };
    let files: [(&str, Vec<u64>, Vec<u64>); 4] = [
        ("aaaaaaaaaaaaaaa1", vec![1, u64::MAX], vec![1]),
        ("aaaaaaaaaaaaaaa2", vec![u64::MAX], vec![]),
        (
            "aaaaaaaaaaaaaaa3",
            vec![1, 2, u64::MAX - 1, u64::MAX],
            vec![1, 2],
        ),
        ("aaaaaaaaaaaaaaa4", vec![u64::MAX - 3, 1, 2], vec![1, 2]),
    ];
    for (n, (id, seqs, _)) in files.iter().enumerate() {
        let summary = RunSummary {
            id: (*id).to_owned(),
            task: "old".into(),
            agent: ASSISTANT.into(),
            agent_label: "Lattice assistant".into(),
            model: "dev:scripted".into(),
            model_label: "Dev".into(),
            locality: Locality::Local,
            status: RunStatus::Completed,
            created_at: 100.0 + n as f64,
            updated_at: 100.0 + n as f64,
            ended_at: Some(100.0 + n as f64),
            trace_id: "trace_00000000000000000000000000000000".into(),
            usage: None,
            output: None,
            error: None,
            spans: 0,
        };
        store.write_summary(&summary, false).unwrap();
        let text: String = seqs.iter().map(|seq| line(*seq) + "\n").collect();
        std::fs::write(store.events_path(id).unwrap(), text).unwrap();
    }
    let service = dev_service(dir.path());
    for (id, _, kept) in &files {
        let detail = service.run(id).expect("the run is listed");
        let seqs: Vec<u64> = detail.events.iter().map(|e| e.seq).collect();
        assert_eq!(&seqs, kept, "{id}");
        let followed: Vec<u64> = events_of(&service, id, 0).iter().map(|e| e.seq).collect();
        assert_eq!(&followed, kept, "{id}");
    }
}

#[cfg(windows)]
#[test]
fn a_finished_run_holds_no_file_open() {
    // Asking for exclusive access to the events file succeeds only when no
    // handle to it is open. A handle kept per run for the life of the service
    // runs a long-lived window out of descriptors.
    use std::os::windows::fs::OpenOptionsExt;

    let dir = TempDir::new("manager-handles");
    let service = dev_service(dir.path());
    let id = service.start(request("hi", "dev:scripted")).unwrap().id;
    assert_eq!(
        end_status(&events_of(&service, &id, 0)),
        EndStatus::Completed
    );
    let path = runs_store(dir.path()).events_path(&id).unwrap();
    eventually("the events file to be let go of", 10, || {
        std::fs::OpenOptions::new()
            .write(true)
            .share_mode(0)
            .open(&path)
            .ok()
    });
}

#[test]
fn a_run_refused_by_a_guardrail_keeps_no_copy_of_the_secret_it_was_refused_for() {
    let dir = TempDir::new("manager-redacted-task");
    write_registry(dir.path(), REMOTE_REGISTRY);
    let (model, factory) = scripted(vec![ScriptedStep::respond(vec![assistant_message(
        "should never be said",
    )])]);
    let env = MapEnv::new().with("HOSTED_API_KEY", "fixture-key-value");
    let service = with_factory(dir.path(), env, factory);
    let summary = service
        .start(request(
            &format!("Please call the API with {SECRET} and tell me a joke"),
            "endpoint:hosted",
        ))
        .unwrap();
    let events = events_of(&service, &summary.id, 0);
    assert_eq!(end_status(&events), EndStatus::Refused);
    assert_eq!(model.calls().len(), 0);

    let store = runs_store(dir.path());
    let stored = eventually("the final summary", 30, || {
        store
            .read_summary(&summary.id)
            .filter(|summary| summary.status == RunStatus::Refused)
    });
    let expected = format!("Please call the API with {REDACTED} and tell me a joke");
    assert_eq!(stored.task, expected, "the record on disk");
    assert_eq!(service.runs()[0].task, expected, "the list");

    let on_disk = std::fs::read_to_string(store.summary_path(&summary.id).unwrap()).unwrap();
    let events_on_disk = persisted_lines(dir.path(), &summary.id).join("\n");
    let detail = format!("{:?}", service.run(&summary.id).unwrap());
    for (what, text) in [
        ("run.json", on_disk),
        ("events.jsonl", events_on_disk),
        ("the run as the window reads it", detail),
    ] {
        assert!(!text.contains(SECRET), "{what} kept the secret");
        assert!(!text.contains("abcdefghijklmnop"), "{what} kept part of it");
    }
}

#[test]
fn an_event_that_carries_a_secret_is_cleaned_wherever_it_is_and_one_that_does_not_is_left_alone() {
    let event = |kind| RunEvent {
        seq: 3,
        at: 1.5,
        kind,
    };
    let carrying = [
        RunEventKind::Message {
            agent: "a".into(),
            text: format!("the key is {SECRET}."),
        },
        RunEventKind::ToolCall {
            agent: "a".into(),
            name: "t".into(),
            call_id: "c".into(),
            arguments: format!("{{\"key\":\"{SECRET}\"}}"),
        },
        RunEventKind::SpanEnd {
            span: SpanRecord {
                order: 1,
                id: "span_1".into(),
                trace_id: "trace_1".into(),
                parent_id: None,
                started_at: "2026-09-30T05:21:05.123456+00:00".into(),
                ended_at: None,
                span_data: json!({"type": "generation", "input": [{"role": "user", "content": format!("use {SECRET}")}]}),
                error: None,
            },
        },
    ];
    for kind in carrying {
        let dirty = event(kind);
        let clean = redacted_event(&dirty).expect("it held a secret");
        assert_eq!((clean.seq, clean.at), (3, 1.5));
        let text = format!("{clean:?}");
        assert!(
            !text.contains("abcdefghijklmnop") && text.contains(REDACTED),
            "{text}"
        );
    }
    let innocent = event(RunEventKind::Message {
        agent: "a".into(),
        text: "What is 6 * 7?".into(),
    });
    assert!(redacted_event(&innocent).is_none());
}

#[test]
fn a_finished_run_lets_go_of_its_events_file_and_a_working_one_holds_it() {
    let dir = TempDir::new("manager-handles-count");
    let mut cfg = config(dir.path(), MapEnv::new(), true);
    cfg.dev_step_delay = Duration::from_millis(200);
    let service = CoreService::new(cfg);
    assert_eq!(service.open_event_files(), 0);
    let id = service.start(request("one", "dev:scripted")).unwrap().id;
    eventually("the working run's file to be open", 30, || {
        (service.open_event_files() == 1).then_some(())
    });
    assert_eq!(
        end_status(&events_of(&service, &id, 0)),
        EndStatus::Completed
    );
    eventually("the finished run's file to be closed", 30, || {
        (service.open_event_files() == 0).then_some(())
    });
    // Many runs do not add up to many handles.
    for n in 0..5 {
        let id = service
            .start(request(&format!("run {n}"), "dev:scripted"))
            .unwrap()
            .id;
        events_of(&service, &id, 0);
    }
    eventually("no run's file to be open", 30, || {
        (service.open_event_files() == 0).then_some(())
    });
}

#[test]
fn a_window_started_before_any_run_exists_sees_the_first_windows_runs_and_cannot_take_the_directory()
 {
    let dir = TempDir::new("manager-two-windows-unclaimed");
    let a = dev_service(dir.path());
    let b = dev_service(dir.path());
    assert!(b.runs().is_empty());
    assert_eq!(
        b.status().refusal,
        None,
        "nothing records yet, so nothing stands in the way"
    );

    let id = a.start(request("one", "dev:scripted")).unwrap().id;
    follow_to_the_end(&a, &id);
    eventually("the first window's run in the second", 15, || {
        b.runs().into_iter().find(|run| run.id == id)
    });
    let refused = b.start(request("two", "dev:scripted")).unwrap_err();
    assert_eq!(refused.kind, RefusalKind::Unavailable);
    assert_eq!(
        b.status().refusal.as_deref(),
        Some("Another Lattice window is recording runs. Start runs there, or close it.")
    );
}
