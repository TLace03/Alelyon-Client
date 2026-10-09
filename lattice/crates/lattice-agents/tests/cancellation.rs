//! Cancelling a run: `Immediate` aborts what is in flight, `AfterTurn` lets the
//! turn in progress finish. Every path must close every span and the trace, and
//! must never leave the run task or a tool future behind.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures::StreamExt;
use lattice_agents::testing::{ScriptedModel, ScriptedStep, assistant_message, function_call};
use lattice_agents::{
    Agent, CancelMode, FunctionTool, RunConfig, RunError, run_streamed, strict_object_schema,
};
use serde_json::json;
use tokio::sync::Notify;

use common::{Collector, agent_with, echo};

/// Sets its flag when dropped: proof that a future was abandoned, not left running.
struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

fn hanging_tool(started: Arc<Notify>, dropped: Arc<AtomicBool>) -> FunctionTool {
    FunctionTool::new(
        "hang",
        "",
        strict_object_schema(json!({}), &[]),
        move |_, _| {
            let started = started.clone();
            let dropped = dropped.clone();
            async move {
                let _guard = DropFlag(dropped);
                started.notify_one();
                tokio::time::sleep(Duration::from_secs(3600)).await;
                Ok(String::new())
            }
        },
    )
}

fn assert_everything_closed(collector: &Collector) {
    let starts = collector.span_starts();
    let ends = collector.span_ends();
    assert_eq!(starts.len(), ends.len(), "every span that started ended");
    for start in &starts {
        assert!(
            ends.iter().any(|end| end.id == start.id),
            "span {} never ended",
            start.id
        );
    }
    assert_eq!(collector.trace_starts().len(), 1);
    assert_eq!(collector.trace_ends().len(), 1);
    assert_eq!(
        collector.calls().last().unwrap(),
        "trace_end",
        "the trace ends last"
    );
}

async fn within<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(10), future)
        .await
        .expect("the run must end promptly")
}

#[tokio::test]
async fn an_immediate_cancel_aborts_a_model_call_that_would_take_an_hour() {
    let model = Arc::new(
        ScriptedModel::new([vec![assistant_message("never")]])
            .with_delay(Duration::from_secs(3600)),
    );
    let collector = Collector::new();
    let mut config = RunConfig::new(model.clone());
    config.processors = vec![collector.clone()];
    let mut handle = run_streamed(agent_with("Assistant", "", vec![]), "Go.".into(), config);

    // Wait until the model has actually been called.
    while model.calls().is_empty() {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    handle.control.cancel(CancelMode::Immediate);
    handle.control.cancel(CancelMode::Immediate);

    let result = within(handle.result).await.unwrap();
    assert!(matches!(result, Err(RunError::Cancelled)), "{result:?}");
    let mut names = Vec::new();
    while let Some(event) = within(handle.events.next()).await {
        names.push(event.sdk_name());
    }
    assert_eq!(
        names,
        ["agent_updated_stream_event"],
        "the events stop where the run did"
    );
    assert_eq!(
        model.remaining_steps(),
        1,
        "the abandoned call consumed nothing"
    );
    assert_everything_closed(&collector);
    let ends = collector.span_ends();
    assert!(
        ends.iter().all(|span| span.error.is_none()),
        "a cancellation is not an error span"
    );
}

#[tokio::test]
async fn an_immediate_cancel_abandons_a_running_tool_and_closes_its_span() {
    let started = Arc::new(Notify::new());
    let dropped = Arc::new(AtomicBool::new(false));
    let agent = agent_with(
        "Assistant",
        "",
        vec![hanging_tool(started.clone(), dropped.clone())],
    );
    let model = Arc::new(ScriptedModel::new([vec![function_call(
        "hang", "{}", "call_1",
    )]]));
    let collector = Collector::new();
    let mut config = RunConfig::new(model);
    config.processors = vec![collector.clone()];
    let handle = run_streamed(agent, "Go.".into(), config);

    within(started.notified()).await;
    assert!(!dropped.load(Ordering::SeqCst));
    handle.control.cancel(CancelMode::Immediate);
    let result = within(handle.result).await.unwrap();

    assert!(matches!(result, Err(RunError::Cancelled)), "{result:?}");
    assert!(
        dropped.load(Ordering::SeqCst),
        "the tool's future was dropped, not left running"
    );
    assert_everything_closed(&collector);
    let ends = collector.span_ends();
    let function = ends
        .iter()
        .find(|s| s.data_type() == "function")
        .expect("the tool's span exists");
    assert!(function.ended_at.is_some());
    // The turn that was cut short still reports what it used.
    let turn = ends
        .iter()
        .find(|s| s.data_str("name") == Some("turn"))
        .unwrap();
    assert_eq!(turn.span_data["data"]["usage"]["input_tokens"], 0);
}

#[tokio::test]
async fn an_after_turn_cancel_lets_the_turn_finish_and_stops_before_the_next() {
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let tool = {
        let started = started.clone();
        let release = release.clone();
        FunctionTool::new(
            "gate",
            "",
            strict_object_schema(json!({}), &[]),
            move |_, _| {
                let started = started.clone();
                let release = release.clone();
                async move {
                    started.notify_one();
                    release.notified().await;
                    Ok("gate opened".to_owned())
                }
            },
        )
    };
    let model = Arc::new(ScriptedModel::new([
        vec![function_call("gate", "{}", "call_1")],
        vec![assistant_message("never asked for")],
    ]));
    let collector = Collector::new();
    let mut config = RunConfig::new(model.clone());
    config.processors = vec![collector.clone()];
    let mut handle = run_streamed(
        agent_with("Assistant", "", vec![tool]),
        "Go.".into(),
        config,
    );

    within(started.notified()).await;
    handle.control.cancel(CancelMode::AfterTurn);
    release.notify_one();

    let result = within(handle.result).await.unwrap();
    assert!(matches!(result, Err(RunError::Cancelled)), "{result:?}");
    let mut names = Vec::new();
    while let Some(event) = within(handle.events.next()).await {
        names.push(event.sdk_name());
    }
    assert_eq!(
        names,
        ["agent_updated_stream_event", "tool_called", "tool_output"],
        "the turn in progress finished"
    );
    assert_eq!(model.calls().len(), 1, "no second turn started");
    assert_eq!(model.remaining_steps(), 1);
    assert_everything_closed(&collector);
    let ends = collector.span_ends();
    let function = ends.iter().find(|s| s.data_type() == "function").unwrap();
    assert_eq!(function.span_data["output"], "gate opened");
    assert_eq!(
        ends.iter()
            .filter(|s| s.data_str("name") == Some("turn"))
            .count(),
        1
    );
}

#[tokio::test]
async fn an_after_turn_cancel_does_not_discard_a_final_answer() {
    let model = Arc::new(
        ScriptedModel::new([
            ScriptedStep::respond(vec![function_call("echo", "{\"text\":\"x\"}", "call_1")]),
            ScriptedStep::respond(vec![assistant_message("the answer")]),
        ])
        .with_delay(Duration::from_millis(150)),
    );
    let handle = run_streamed(
        agent_with("Assistant", "", vec![echo()]),
        "Go.".into(),
        RunConfig::new(model.clone()),
    );
    // The second (final) model call is in flight.
    while model.calls().len() < 2 {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    handle.control.cancel(CancelMode::AfterTurn);
    let result = within(handle.result)
        .await
        .unwrap()
        .expect("the final turn completes");
    assert_eq!(result.final_output, "the answer");
}

#[tokio::test]
async fn a_run_cancelled_before_it_starts_emits_nothing_and_still_closes_its_trace() {
    let model = Arc::new(ScriptedModel::new([vec![assistant_message("never")]]));
    let collector = Collector::new();
    let mut config = RunConfig::new(model.clone());
    config.processors = vec![collector.clone()];
    let mut handle = run_streamed(Agent::builder("Assistant").build(), "Go.".into(), config);
    handle.control.cancel(CancelMode::Immediate);

    let result = within(handle.result).await.unwrap();
    assert!(matches!(result, Err(RunError::Cancelled)));
    assert!(
        within(handle.events.next()).await.is_none(),
        "no event of a run that never started"
    );
    assert!(model.calls().is_empty());
    assert_everything_closed(&collector);
}

#[tokio::test]
async fn cancelling_a_finished_run_changes_nothing() {
    let model = Arc::new(ScriptedModel::new([vec![assistant_message("done")]]));
    let handle = run_streamed(
        Agent::builder("Assistant").build(),
        "Go.".into(),
        RunConfig::new(model),
    );
    let control = handle.control.clone();
    let result = within(handle.result)
        .await
        .unwrap()
        .expect("the run finishes");
    control.cancel(CancelMode::Immediate);
    control.cancel(CancelMode::AfterTurn);
    assert_eq!(result.final_output, "done");
}
