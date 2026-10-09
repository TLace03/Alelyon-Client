//! Behaviour of the agent loop beyond what the SDK goldens pin: the data a run
//! returns, what models are asked, how tools and guardrails interleave, how
//! failures are recorded, and how the trace is shaped.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use futures::future::BoxFuture;
use futures::{FutureExt, StreamExt};
use lattice_agents::testing::{ScriptedStep, assistant_message, function_call};
use lattice_agents::{
    Agent, ApprovalDecision, ApprovalPort, ApprovalRequest, CancelMode, FunctionTool,
    GenerationTrace, GuardrailOutput, InputGuardrail, InputItem, ModelError, ModelResponse,
    ModelSettings, OutputGuardrail, OutputItem, RunConfig, RunError, RunItem, Session,
    SessionError, StreamEvent, ToolCallItem, ToolError, TraceProcessor, TraceRecord, run_streamed,
    strict_object_schema,
};
use lattice_protocol::{SpanRecord, Usage};
use serde_json::{Value, json};

use common::{
    Collector, MemorySession, PlainModel, ScriptedApprovals, agent_with, clock, delete_file, echo,
    explode, lookup, run_items_to_end, run_to_end, scripted, text_response, weather,
};

fn usage(input: u64, output: u64) -> Usage {
    Usage {
        requests: 1,
        input_tokens: input,
        output_tokens: output,
        total_tokens: input + output,
    }
}

fn step(output: Vec<OutputItem>) -> ScriptedStep {
    ScriptedStep::respond(output)
}

fn find_span<'a>(spans: &'a [SpanRecord], kind: &str) -> &'a SpanRecord {
    spans
        .iter()
        .find(|s| common_kind(s) == kind)
        .unwrap_or_else(|| panic!("no {kind} span"))
}

fn common_kind(span: &SpanRecord) -> String {
    match span.data_type() {
        "custom" => span.data_str("name").unwrap_or("custom").to_owned(),
        other => other.to_owned(),
    }
}

// ---------------------------------------------------------------------------
// The result
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_finished_run_returns_its_output_agent_usage_turns_and_items() {
    let agent = agent_with("Assistant", "Be brief.", vec![echo()]);
    let model = scripted(vec![
        step(vec![
            assistant_message("Let me echo."),
            function_call("echo", "{\"text\":\"hi\"}", "call_1"),
        ])
        .with_tokens(20, 5),
        step(vec![assistant_message("Echoed: hi")]).with_tokens(30, 4),
    ]);
    let outcome = run_to_end(agent, "Echo hi.", model, |_| {}).await;
    let result = outcome.result.expect("the run finishes");

    assert_eq!(result.final_output, "Echoed: hi");
    assert_eq!(result.last_agent, "Assistant");
    assert_eq!(result.turns, 2);
    assert_eq!(
        result.usage,
        Some(Usage {
            requests: 2,
            input_tokens: 50,
            output_tokens: 9,
            total_tokens: 59
        })
    );
    assert_eq!(
        result.new_items,
        vec![
            RunItem::MessageOutput {
                agent: "Assistant".into(),
                text: "Let me echo.".into()
            },
            RunItem::ToolCall {
                agent: "Assistant".into(),
                call_id: "call_1".into(),
                name: "echo".into(),
                arguments: "{\"text\":\"hi\"}".into()
            },
            RunItem::ToolOutput {
                agent: "Assistant".into(),
                call_id: "call_1".into(),
                output: "echo:hi".into()
            },
            RunItem::MessageOutput {
                agent: "Assistant".into(),
                text: "Echoed: hi".into()
            },
        ]
    );
    // A response with text AND a tool call is not a final answer.
    assert_eq!(
        outcome.names,
        [
            "agent_updated_stream_event",
            "raw_text_delta",
            "message_output_created",
            "tool_called",
            "tool_output",
            "raw_text_delta",
            "message_output_created",
        ]
    );
}

#[tokio::test]
async fn usage_is_absent_when_no_model_call_reported_any() {
    let model = PlainModel::new(GenerationTrace::Bare, vec![Ok(text_response("hi", None))]);
    let collector = Collector::new();
    let mut config = RunConfig::new(model);
    config.processors = vec![collector.clone()];
    let handle = run_streamed(agent_with("A", "", vec![]), "hello".into(), config);
    let result = handle.result.await.unwrap().unwrap();
    assert_eq!(result.usage, None);
    // With nothing reported, no task or turn span claims usage either.
    let ends = collector.span_ends();
    assert!(
        find_span(&ends, "task").span_data["data"]
            .get("usage")
            .is_none()
    );
    assert!(
        find_span(&ends, "turn").span_data["data"]
            .get("usage")
            .is_none()
    );
}

#[tokio::test]
async fn an_empty_response_is_an_empty_final_output() {
    let model = PlainModel::new(GenerationTrace::Bare, vec![Ok(ModelResponse::default())]);
    let handle = run_streamed(
        agent_with("A", "", vec![]),
        "hello".into(),
        RunConfig::new(model),
    );
    assert_eq!(handle.result.await.unwrap().unwrap().final_output, "");
}

#[tokio::test]
async fn only_the_last_message_of_a_response_is_the_final_output() {
    let model = PlainModel::new(
        GenerationTrace::Bare,
        vec![Ok(ModelResponse {
            output: vec![
                OutputItem::Reasoning { text: "hmm".into() },
                OutputItem::Message {
                    text: "first".into(),
                },
                OutputItem::Message {
                    text: "second".into(),
                },
            ],
            usage: None,
        })],
    );
    let collector = Collector::new();
    let mut config = RunConfig::new(model.clone());
    config.processors = vec![collector];
    let mut handle = run_streamed(agent_with("A", "", vec![]), "hello".into(), config);
    let mut names = Vec::new();
    while let Some(event) = handle.events.next().await {
        if let StreamEvent::RunItem { name, .. } = event {
            names.push(name.as_str());
        }
    }
    let result = handle.result.await.unwrap().unwrap();
    assert_eq!(result.final_output, "second");
    assert_eq!(
        names,
        [
            "reasoning_item_created",
            "message_output_created",
            "message_output_created"
        ]
    );
}

// ---------------------------------------------------------------------------
// What the model is asked
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_second_agent_sees_the_whole_conversation_and_only_its_own_tools() {
    let specialist = Agent::builder("Specialist agent")
        .instructions("You are the specialist.")
        .tool(lookup())
        .build();
    let triage = Agent::builder("Triage agent")
        .instructions("Route the request.")
        .handoff_description("Routes.")
        .handoff_to(specialist)
        .tool(echo())
        .build();
    let model = scripted(vec![
        step(vec![function_call(
            "transfer_to_specialist_agent",
            "{}",
            "call_h",
        )]),
        step(vec![assistant_message("Done.")]),
    ]);
    let outcome = run_to_end(triage, "Find the manual.", model.clone(), |_| {}).await;
    outcome.result.expect("the run finishes");

    let calls = model.calls();
    assert_eq!(calls.len(), 2);
    // First call: the triage agent's instructions, its tool and the handoff tool.
    assert_eq!(calls[0].system, "Route the request.");
    assert_eq!(
        calls[0].input,
        vec![InputItem::User("Find the manual.".into())]
    );
    let names: Vec<&str> = calls[0].tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, ["echo", "transfer_to_specialist_agent"]);
    let handoff_tool = &calls[0].tools[1];
    assert_eq!(
        handoff_tool.description,
        "Handoff to the Specialist agent agent to handle the request. "
    );
    assert_eq!(handoff_tool.parameters["properties"], json!({}));
    assert!(handoff_tool.strict);
    // Second call: the specialist, with the conversation so far and its own tool.
    assert_eq!(calls[1].system, "You are the specialist.");
    assert_eq!(
        calls[1].input,
        vec![
            InputItem::User("Find the manual.".into()),
            InputItem::Assistant {
                text: None,
                tool_calls: vec![ToolCallItem {
                    call_id: "call_h".into(),
                    name: "transfer_to_specialist_agent".into(),
                    arguments: "{}".into()
                }]
            },
            InputItem::ToolResult {
                call_id: "call_h".into(),
                output: r#"{"assistant": "Specialist agent"}"#.into()
            },
        ]
    );
    let names: Vec<&str> = calls[1].tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, ["lookup"]);
}

#[tokio::test]
async fn an_ignored_handoff_is_answered_before_the_executed_one() {
    let billing = Agent::builder("Billing agent").build();
    let support = Agent::builder("Support agent").build();
    let router = Agent::builder("Router")
        .handoff_to(billing)
        .handoff_to(support)
        .build();
    let model = scripted(vec![
        step(vec![
            function_call("transfer_to_billing_agent", "{}", "call_b"),
            function_call("transfer_to_support_agent", "{}", "call_s"),
        ]),
        step(vec![assistant_message("Billing here.")]),
    ]);
    let outcome = run_to_end(router, "Both.", model.clone(), |_| {}).await;
    outcome.result.expect("the run finishes");
    let second = &model.calls()[1];
    let results: Vec<&InputItem> = second
        .input
        .iter()
        .filter(|i| matches!(i, InputItem::ToolResult { .. }))
        .collect();
    assert_eq!(
        results,
        vec![
            &InputItem::ToolResult {
                call_id: "call_s".into(),
                output: "Multiple handoffs detected, ignoring this one.".into()
            },
            &InputItem::ToolResult {
                call_id: "call_b".into(),
                output: r#"{"assistant": "Billing agent"}"#.into()
            },
        ]
    );
}

#[tokio::test]
async fn function_calls_in_a_handoff_response_run_first() {
    let specialist = Agent::builder("Specialist agent").build();
    let triage = Agent::builder("Triage agent")
        .tool(echo())
        .handoff_to(specialist)
        .build();
    let model = scripted(vec![
        step(vec![
            function_call("transfer_to_specialist_agent", "{}", "call_h"),
            function_call("echo", "{\"text\":\"x\"}", "call_e"),
        ]),
        step(vec![assistant_message("Done.")]),
    ]);
    let outcome = run_to_end(triage, "Go.", model.clone(), |_| {}).await;
    outcome.result.expect("the run finishes");
    assert_eq!(
        outcome.names,
        [
            "agent_updated_stream_event",
            "handoff_requested",
            "tool_called",
            "tool_output",
            "handoff_occured",
            "agent_updated_stream_event",
            "raw_text_delta",
            "message_output_created",
        ]
    );
    let results: Vec<String> = model.calls()[1]
        .input
        .iter()
        .filter_map(|i| match i {
            InputItem::ToolResult { call_id, .. } => Some(call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(results, ["call_e", "call_h"]);
}

#[tokio::test]
async fn model_settings_resolve_per_field_and_a_forced_tool_choice_applies_once() {
    let agent = Agent::builder("A")
        .tool(echo())
        .model_settings(ModelSettings {
            temperature: Some(0.2),
            max_tokens: Some(100),
            tool_choice: Some("required".into()),
            ..ModelSettings::default()
        })
        .build();
    let model = scripted(vec![
        step(vec![function_call("echo", "{\"text\":\"x\"}", "call_1")]),
        step(vec![assistant_message("done")]),
    ]);
    let outcome = run_to_end(agent, "Go.", model.clone(), |config| {
        config.model_settings = Some(ModelSettings {
            temperature: Some(0.9),
            ..ModelSettings::default()
        });
    })
    .await;
    outcome.result.expect("the run finishes");
    let calls = model.calls();
    assert_eq!(
        calls[0].settings.temperature,
        Some(0.9),
        "the run's setting overrides the agent's"
    );
    assert_eq!(
        calls[0].settings.max_tokens,
        Some(100),
        "an unset run field keeps the agent's"
    );
    assert_eq!(calls[0].settings.tool_choice.as_deref(), Some("required"));
    assert_eq!(
        calls[1].settings.tool_choice, None,
        "a forced tool choice is reset once the agent used a tool"
    );
}

// ---------------------------------------------------------------------------
// Tools running
// ---------------------------------------------------------------------------

fn sleeper(name: &'static str, millis: u64) -> FunctionTool {
    FunctionTool::new(
        name,
        "",
        strict_object_schema(json!({}), &[]),
        move |_, _| async move {
            tokio::time::sleep(Duration::from_millis(millis)).await;
            Ok(format!("{name} done"))
        },
    )
}

#[tokio::test(start_paused = true)]
async fn function_calls_run_concurrently_and_report_in_call_order() {
    let agent = agent_with("A", "", vec![sleeper("slow", 100), sleeper("fast", 10)]);
    let model = scripted(vec![
        step(vec![
            function_call("slow", "{}", "call_slow"),
            function_call("fast", "{}", "call_fast"),
        ]),
        step(vec![assistant_message("ok")]),
    ]);
    let started = tokio::time::Instant::now();
    let outcome = run_to_end(agent, "Go.", model.clone(), |_| {}).await;
    outcome.result.expect("the run finishes");
    // Concurrent: the slower tool's 100 ms, not the sum of 110 ms.
    assert_eq!(started.elapsed(), Duration::from_millis(100));

    let outputs: Vec<&str> = outcome
        .events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::RunItem {
                item: RunItem::ToolOutput { output, .. },
                ..
            } => Some(output.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        outputs,
        ["slow done", "fast done"],
        "outputs are appended in call order"
    );

    // Spans start in call order even though the fast one ends first.
    let starts = outcome.collector.span_starts();
    let function_starts: Vec<&str> = starts
        .iter()
        .filter(|s| s.data_type() == "function")
        .map(|s| s.data_str("name").unwrap())
        .collect();
    assert_eq!(function_starts, ["slow", "fast"]);
    let ends = outcome.collector.span_ends();
    let function_ends: Vec<&str> = ends
        .iter()
        .filter(|s| s.data_type() == "function")
        .map(|s| s.data_str("name").unwrap())
        .collect();
    assert_eq!(function_ends, ["fast", "slow"]);
}

#[tokio::test]
async fn tools_and_guardrails_see_the_run_context_and_the_task() {
    let seen_by_guardrail = Arc::new(std::sync::Mutex::new(String::new()));
    let seen = seen_by_guardrail.clone();
    let guardrail = InputGuardrail::new("look", move |context, text| {
        let seen = seen.clone();
        async move {
            let tag = context
                .run_context
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_default();
            *seen.lock().unwrap() = format!("{tag}|{}|{text}", context.agent);
            GuardrailOutput::pass()
        }
    });
    let tool = FunctionTool::new(
        "who",
        "",
        strict_object_schema(json!({}), &[]),
        |context, _| async move {
            let tag = context
                .run_context
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_default();
            Ok(format!("{tag}|{}|{}", context.agent, context.call_id))
        },
    );
    let agent = Agent::builder("Assistant")
        .tool(tool)
        .input_guardrail(guardrail)
        .build();
    let model = scripted(vec![
        step(vec![function_call("who", "{}", "call_w")]),
        step(vec![assistant_message("ok")]),
    ]);
    let outcome = run_to_end(agent, "the task", model, |config| {
        config.context = Arc::new("ctx".to_owned());
    })
    .await;
    let result = outcome.result.expect("the run finishes");
    assert_eq!(*seen_by_guardrail.lock().unwrap(), "ctx|Assistant|the task");
    let output = result.new_items.iter().find_map(|item| match item {
        RunItem::ToolOutput { output, .. } => Some(output.clone()),
        _ => None,
    });
    assert_eq!(output.as_deref(), Some("ctx|Assistant|call_w"));
}

#[tokio::test]
async fn a_model_that_asks_for_a_missing_tool_ends_the_run_and_marks_the_turn() {
    let model = scripted(vec![step(vec![function_call("nope", "{}", "call_n")])]);
    let outcome = run_to_end(
        agent_with("Assistant", "", vec![echo()]),
        "Go.",
        model,
        |_| {},
    )
    .await;
    let error = outcome.result.expect_err("the run fails");
    assert_eq!(error.sdk_name(), "ModelBehaviorError");
    assert_eq!(error.to_string(), "Tool nope not found in agent Assistant");
    // Nothing from the bad response was announced.
    assert_eq!(outcome.names, ["agent_updated_stream_event"]);

    let ends = outcome.collector.span_ends();
    let turn = find_span(&ends, "turn");
    let turn_error = turn
        .error
        .as_ref()
        .expect("the turn span carries the error");
    assert_eq!(turn_error.message, "Tool not found");
    assert_eq!(
        turn_error.data.as_ref().unwrap(),
        &json!({"tool_name": "nope"})
    );
    assert!(
        find_span(&ends, "agent").error.is_none(),
        "model misbehaviour is not a generic agent error"
    );
}

// ---------------------------------------------------------------------------
// Failure
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_model_failure_ends_the_run_and_is_recorded_on_the_generation_and_agent_spans() {
    let model = scripted(vec![ScriptedStep::error(ModelError::Status(500))]);
    let outcome = run_to_end(agent_with("Assistant", "", vec![]), "Go.", model, |_| {}).await;
    let error = outcome.result.expect_err("the run fails");
    assert!(
        matches!(error, RunError::Model(ModelError::Status(500))),
        "{error:?}"
    );

    let ends = outcome.collector.span_ends();
    let generation = find_span(&ends, "generation").error.as_ref().unwrap();
    assert_eq!(generation.message, "Error");
    assert_eq!(generation.data.as_ref().unwrap()["name"], "Status");
    let agent = find_span(&ends, "agent").error.as_ref().unwrap();
    assert_eq!(agent.message, "Error in agent run");
    assert_eq!(
        agent.data.as_ref().unwrap()["error"],
        "the model server answered HTTP 500"
    );
    assert_eq!(*outcome.collector.calls().last().unwrap(), "trace_end");
}

#[tokio::test]
async fn model_error_text_is_redacted_from_spans_when_sensitive_data_is_off() {
    let model = scripted(vec![ScriptedStep::error(ModelError::Failed(
        "the key sk-123 was rejected".into(),
    ))]);
    let outcome = run_to_end(
        agent_with("Assistant", "", vec![]),
        "Go.",
        model,
        |config| {
            config.include_sensitive_data = false;
        },
    )
    .await;
    outcome.result.expect_err("the run fails");
    let ends = outcome.collector.span_ends();
    let generation = find_span(&ends, "generation").error.as_ref().unwrap();
    assert_eq!(
        generation.data.as_ref().unwrap()["message"],
        "Error details are redacted."
    );
    let agent = find_span(&ends, "agent").error.as_ref().unwrap();
    assert_eq!(
        agent.data.as_ref().unwrap()["error"],
        "Error details are redacted."
    );
    let dumped = serde_json::to_string(&ends).unwrap();
    assert!(!dumped.contains("sk-123"));
}

#[tokio::test]
async fn a_stream_without_a_final_response_is_a_model_behaviour_error() {
    struct Silent;
    impl lattice_agents::Model for Silent {
        fn name(&self) -> &str {
            "silent"
        }
        fn config_for_trace(&self) -> Value {
            Value::Null
        }
        fn stream(
            &self,
            _: lattice_agents::ModelRequest,
        ) -> futures::stream::BoxStream<'static, Result<lattice_agents::ModelEvent, ModelError>>
        {
            futures::stream::empty().boxed()
        }
    }
    let handle = run_streamed(
        agent_with("A", "", vec![]),
        "hello".into(),
        RunConfig::new(Arc::new(Silent)),
    );
    let error = handle.result.await.unwrap().unwrap_err();
    assert_eq!(error.to_string(), "Model did not produce a final response!");
    assert_eq!(error.sdk_name(), "ModelBehaviorError");
}

/// A guardrail's private detail: never hex, so never inside a random id.
const INFO_MARKER: &str = "zqx-guardrail-private-detail";

#[tokio::test]
async fn a_tripped_input_guardrail_never_calls_the_model_and_reports_its_info() {
    // The marker holds letters no hex digit is ("z", "q", "x"), so a span's
    // random trace or span id can never contain it (the old three-letter
    // marker was inside some id in about 2% of runs, which failed the
    // check below at random).
    let guardrail = InputGuardrail::new("refuse", |_, _| async {
        GuardrailOutput::trip(json!({"matched": INFO_MARKER}))
    });
    let agent = Agent::builder("Assistant")
        .input_guardrail(guardrail)
        .build();
    let model = scripted(vec![]);
    let outcome = run_to_end(agent, "Go.", model.clone(), |_| {}).await;
    match outcome.result.expect_err("the run is refused") {
        RunError::InputGuardrailTripwire {
            guardrail,
            output_info,
        } => {
            assert_eq!(guardrail, "refuse");
            assert_eq!(output_info, json!({"matched": INFO_MARKER}));
        }
        other => panic!("{other:?}"),
    }
    assert!(
        model.calls().is_empty(),
        "a refused task is never sent to a model"
    );
    // The check's own account of its decision is not recorded in any span.
    assert!(
        !serde_json::to_string(&outcome.collector.span_ends())
            .unwrap()
            .contains(INFO_MARKER)
    );
}

#[tokio::test]
async fn input_guardrails_belong_to_the_first_agent_only_and_run_in_order_until_one_trips() {
    let ran = Arc::new(AtomicUsize::new(0));
    let counter = ran.clone();
    let second_agent_guardrail = InputGuardrail::new("second_agent", move |_, _| {
        let counter = counter.clone();
        async move {
            counter.fetch_add(100, Ordering::SeqCst);
            GuardrailOutput::pass()
        }
    });
    let specialist = Agent::builder("Specialist")
        .input_guardrail(second_agent_guardrail)
        .build();
    let order = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let log = |name: &'static str, trip: bool, order: Arc<std::sync::Mutex<Vec<String>>>| {
        InputGuardrail::new(name, move |_, _| {
            let order = order.clone();
            async move {
                order.lock().unwrap().push(name.to_owned());
                if trip {
                    GuardrailOutput::trip(Value::Null)
                } else {
                    GuardrailOutput::pass()
                }
            }
        })
    };
    let first = Agent::builder("First")
        .input_guardrail(log("g1", false, order.clone()))
        .input_guardrail(log("g2", true, order.clone()))
        .input_guardrail(log("g3", false, order.clone()))
        .handoff_to(specialist.clone())
        .build();
    let outcome = run_to_end(first, "Go.", scripted(vec![]), |_| {}).await;
    assert!(
        matches!(outcome.result, Err(RunError::InputGuardrailTripwire { ref guardrail, .. }) if guardrail == "g2")
    );
    assert_eq!(
        *order.lock().unwrap(),
        ["g1", "g2"],
        "checks run in order and stop at the first tripwire"
    );

    // A passing run: only the first agent's guardrails ran, never the specialist's.
    let passing = Agent::builder("First2")
        .input_guardrail(log("g1", false, order.clone()))
        .handoff_to(specialist)
        .build();
    let model = scripted(vec![
        step(vec![function_call("transfer_to_specialist", "{}", "c1")]),
        step(vec![assistant_message("hi")]),
    ]);
    let outcome = run_to_end(passing, "Go.", model, |_| {}).await;
    outcome.result.expect("the run finishes");
    assert_eq!(ran.load(Ordering::SeqCst), 0);
}

#[tokio::test(start_paused = true)]
async fn output_guardrails_run_alongside_each_other_and_the_first_to_trip_wins() {
    let slow_pass = OutputGuardrail::new("slow_pass", |_, _| async {
        tokio::time::sleep(Duration::from_millis(500)).await;
        GuardrailOutput::pass()
    });
    let quick_trip = OutputGuardrail::new("quick_trip", |_, text| async move {
        GuardrailOutput::trip(json!({"text_len": text.len()}))
    });
    let agent = Agent::builder("Assistant")
        .output_guardrail(slow_pass)
        .output_guardrail(quick_trip)
        .build();
    let model = scripted(vec![step(vec![assistant_message("An answer.")])]);
    let outcome = run_to_end(agent, "Go.", model, |_| {}).await;
    match outcome.result.expect_err("the answer is refused") {
        RunError::OutputGuardrailTripwire {
            guardrail,
            output_info,
        } => {
            assert_eq!(guardrail, "quick_trip");
            assert_eq!(output_info, json!({"text_len": 10}));
        }
        other => panic!("{other:?}"),
    }
    let ends = outcome.collector.span_ends();
    let guardrails: Vec<(&str, bool)> = ends
        .iter()
        .filter(|s| s.data_type() == "guardrail")
        .map(|s| {
            (
                s.data_str("name").unwrap(),
                s.span_data["triggered"].as_bool().unwrap(),
            )
        })
        .collect();
    // Both ran (their spans exist and ended); the interrupted one did not trip.
    assert_eq!(guardrails, [("quick_trip", true), ("slow_pass", false)]);
    let starts = outcome.collector.span_starts();
    let started: Vec<&str> = starts
        .iter()
        .filter(|s| s.data_type() == "guardrail")
        .map(|s| s.data_str("name").unwrap())
        .collect();
    assert_eq!(
        started,
        ["slow_pass", "quick_trip"],
        "output guardrails start in registration order"
    );
}

#[tokio::test]
async fn the_turn_counter_counts_handoff_turns_too() {
    let second = Agent::builder("Second").build();
    let first = Agent::builder("First").handoff_to(second).build();
    let model = scripted(vec![
        step(vec![function_call("transfer_to_second", "{}", "c1")]),
        step(vec![assistant_message("never")]),
    ]);
    let outcome = run_to_end(first, "Go.", model.clone(), |config| config.max_turns = 1).await;
    match outcome
        .result
        .expect_err("the second turn is over the limit")
    {
        RunError::MaxTurnsExceeded { max_turns } => assert_eq!(max_turns, 1),
        other => panic!("{other:?}"),
    }
    assert_eq!(model.calls().len(), 1);
    let ends = outcome.collector.span_ends();
    let second_agent = ends
        .iter()
        .filter(|s| s.data_type() == "agent")
        .find(|s| s.data_str("name") == Some("Second"))
        .unwrap();
    let error = second_agent
        .error
        .as_ref()
        .expect("the agent whose turn was refused carries the error");
    assert_eq!(error.message, "Max turns exceeded");
    assert_eq!(error.data.as_ref().unwrap(), &json!({"max_turns": 1}));
}

// ---------------------------------------------------------------------------
// The trace
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_trace_carries_the_runs_settings_and_ends_last() {
    let model = scripted(vec![step(vec![assistant_message("hi")])]);
    let outcome = run_to_end(
        agent_with("Assistant", "", vec![]),
        "Go.",
        model,
        |config| {
            config.workflow_name = "My flow".into();
            config.trace_id = Some("trace_0123456789abcdef0123456789abcdef".into());
            config.group_id = Some("group_1".into());
            config.trace_metadata = Some(json!({"owner": "test"}));
        },
    )
    .await;
    outcome.result.expect("the run finishes");

    let starts = outcome.collector.trace_starts();
    let ends = outcome.collector.trace_ends();
    assert_eq!((starts.len(), ends.len()), (1, 1));
    let TraceRecord {
        id,
        workflow_name,
        group_id,
        metadata,
        started_at,
        ended_at,
    } = starts[0].clone();
    assert_eq!(id, "trace_0123456789abcdef0123456789abcdef");
    assert_eq!(workflow_name, "My flow");
    assert_eq!(group_id.as_deref(), Some("group_1"));
    assert_eq!(metadata, Some(json!({"owner": "test"})));
    assert!(!started_at.is_empty() && ended_at.is_none());
    assert!(ends[0].ended_at.is_some());

    let spans = outcome.collector.span_ends();
    assert!(spans.iter().all(|s| s.trace_id == id));
    assert_eq!(
        find_span(&spans, "task").span_data["data"]["name"],
        "My flow"
    );
    let calls = outcome.collector.calls();
    assert_eq!(calls.first().unwrap(), "trace_start");
    assert_eq!(calls.last().unwrap(), "trace_end");
}

#[tokio::test]
async fn spans_have_sdk_ids_dense_orders_valid_parents_and_matching_ends() {
    let agent = agent_with("Assistant", "", vec![echo()]);
    let model = scripted(vec![
        step(vec![function_call("echo", "{\"text\":\"1\"}", "c1")]),
        step(vec![assistant_message("done")]),
    ]);
    let outcome = run_to_end(agent, "Go.", model, |_| {}).await;
    outcome.result.expect("the run finishes");

    let starts = outcome.collector.span_starts();
    let ends = outcome.collector.span_ends();
    assert_eq!(starts.len(), ends.len(), "every span that starts ends");
    let orders: Vec<u64> = starts.iter().map(|s| s.order).collect();
    assert_eq!(
        orders,
        (1..=starts.len() as u64).collect::<Vec<_>>(),
        "orders count up from 1 as spans start"
    );
    let ids: Vec<&str> = starts.iter().map(|s| s.id.as_str()).collect();
    for span in &starts {
        assert!(
            span.id.starts_with("span_") && span.id.len() == 29,
            "{}",
            span.id
        );
        assert!(
            span.trace_id.starts_with("trace_") && span.trace_id.len() == 38,
            "{}",
            span.trace_id
        );
        assert!(span.ended_at.is_none(), "the start record is open");
        assert!(
            matches!(span.started_at.len(), 25 | 32) && span.started_at.ends_with("+00:00"),
            "{}",
            span.started_at
        );
        if let Some(parent) = &span.parent_id {
            let parent_order = starts
                .iter()
                .find(|s| &s.id == parent)
                .expect("a parent exists")
                .order;
            assert!(
                parent_order < span.order,
                "a parent starts before its children"
            );
        }
    }
    assert_eq!(
        ids.len(),
        ids.iter().collect::<std::collections::HashSet<_>>().len()
    );
    for end in &ends {
        let ended_at = end.ended_at.as_deref().expect("the end record is closed");
        assert!(
            matches!(ended_at.len(), 25 | 32) && ended_at.ends_with("+00:00"),
            "{ended_at}"
        );
        assert!(ended_at >= end.started_at.as_str());
    }
    let task = find_span(&ends, "task");
    assert!(task.parent_id.is_none());
    assert_eq!(
        find_span(&ends, "agent").parent_id.as_deref(),
        Some(task.id.as_str())
    );
}

#[tokio::test]
async fn without_task_and_turn_spans_the_agent_span_is_the_root() {
    let agent = agent_with("Assistant", "", vec![echo()]);
    let model = scripted(vec![
        step(vec![function_call("echo", "{\"text\":\"1\"}", "c1")]),
        step(vec![assistant_message("done")]),
    ]);
    let outcome = run_to_end(agent, "Go.", model, |config| {
        config.include_task_and_turn_spans = false
    })
    .await;
    outcome.result.expect("the run finishes");
    let forest = outcome.collector.forest();
    assert_eq!(forest.len(), 1);
    assert_eq!(forest[0]["kind"], "agent");
    let kinds: Vec<&str> = forest[0]["children"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["kind"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, ["generation", "function", "generation"]);
    assert!(
        !serde_json::to_string(&forest)
            .unwrap()
            .contains("\"usage\":{\"input_tokens\"")
    );
}

#[tokio::test]
async fn full_generation_spans_record_model_configuration_input_output_and_usage() {
    let model = PlainModel::new(
        GenerationTrace::Full,
        vec![Ok(ModelResponse {
            output: vec![OutputItem::Message {
                text: "hello".into(),
            }],
            usage: Some(usage(1200, 34)),
        })],
    );
    let collector = Collector::new();
    let mut config = RunConfig::new(model);
    config.processors = vec![collector.clone()];
    config.model_settings = Some(ModelSettings {
        temperature: Some(0.5),
        ..ModelSettings::default()
    });
    let agent = agent_with("Assistant", "Be brief.", vec![]);
    let result = run_streamed(agent, "hi there".into(), config)
        .result
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result.usage, Some(usage(1200, 34)));

    let ends = collector.span_ends();
    let data = &find_span(&ends, "generation").span_data;
    assert_eq!(data["model"], "plain-model");
    assert_eq!(data["model_config"]["temperature"], json!(0.5));
    assert_eq!(data["model_config"]["top_p"], Value::Null);
    assert_eq!(data["model_config"]["base_url"], "http://127.0.0.1:1/v1");
    assert_eq!(
        data["input"],
        json!([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi there"}])
    );
    assert_eq!(
        data["output"][0]["output"][0]["content"][0]["text"],
        "hello"
    );
    assert_eq!(data["usage"]["input_tokens"], 1200);
    assert_eq!(data["usage"]["output_tokens"], 34);
    assert_eq!(data["usage"]["requests"], 1);

    // The turn and task spans carry the same counts.
    assert_eq!(
        find_span(&ends, "turn").span_data["data"]["usage"]["input_tokens"],
        1200
    );
    assert_eq!(
        find_span(&ends, "task").span_data["data"]["usage"]["total_tokens"],
        1234
    );
}

#[tokio::test]
async fn full_generation_spans_blank_input_and_output_when_sensitive_data_is_off() {
    let model = PlainModel::new(
        GenerationTrace::Full,
        vec![Ok(text_response("hello", Some(usage(5, 1))))],
    );
    let collector = Collector::new();
    let mut config = RunConfig::new(model);
    config.processors = vec![collector.clone()];
    config.include_sensitive_data = false;
    run_streamed(
        agent_with("Assistant", "Be brief.", vec![]),
        "secret task".into(),
        config,
    )
    .result
    .await
    .unwrap()
    .unwrap();

    let ends = collector.span_ends();
    let data = &find_span(&ends, "generation").span_data;
    assert_eq!(data["input"], Value::Null);
    assert_eq!(data["output"], Value::Null);
    assert_eq!(
        data["model"], "plain-model",
        "the model name and usage are not sensitive data"
    );
    assert_eq!(data["usage"]["input_tokens"], 5);
    assert!(
        !serde_json::to_string(&ends)
            .unwrap()
            .contains("secret task")
    );
}

// ---------------------------------------------------------------------------
// Robustness
// ---------------------------------------------------------------------------

struct Panicker;

impl TraceProcessor for Panicker {
    fn on_trace_start(&self, _: &TraceRecord) {
        panic!("processor failure");
    }
    fn on_trace_end(&self, _: &TraceRecord) {
        panic!("processor failure");
    }
    fn on_span_start(&self, _: &SpanRecord) {
        panic!("processor failure");
    }
    fn on_span_end(&self, _: &SpanRecord) {
        panic!("processor failure");
    }
}

#[tokio::test]
async fn a_panicking_processor_does_not_kill_the_run_and_is_counted() {
    let good = Collector::new();
    let model = scripted(vec![step(vec![assistant_message("still here")])]);
    let mut config = RunConfig::new(model);
    config.processors = vec![Arc::new(Panicker), good.clone()];
    let mut handle = run_streamed(agent_with("Assistant", "", vec![]), "Go.".into(), config);
    let control = handle.control.clone();
    let mut names = Vec::new();
    while let Some(event) = handle.events.next().await {
        names.push(event.sdk_name());
    }
    let result = handle
        .result
        .await
        .unwrap()
        .expect("the run survives its processor");
    assert_eq!(result.final_output, "still here");
    assert_eq!(names.first(), Some(&"agent_updated_stream_event"));
    // trace start + end, task, agent and turn and generation span starts and ends.
    assert_eq!(control.processor_panics(), 2 + 2 * 4);
    assert_eq!(
        good.span_ends().len(),
        4,
        "the processors after the failing one still hear everything"
    );
    assert_eq!(good.calls().last().unwrap(), "trace_end");
}

#[tokio::test]
async fn a_panicking_tool_is_a_failed_call_not_a_dead_run() {
    let tool = FunctionTool::new(
        "bad",
        "",
        strict_object_schema(json!({}), &[]),
        |_, _| async move {
            let fail = std::hint::black_box(true);
            if fail {
                panic!("tool bug with secret-value");
            }
            Ok::<String, ToolError>(String::new())
        },
    );
    let model = scripted(vec![
        step(vec![function_call("bad", "{}", "c1")]),
        step(vec![assistant_message("recovered")]),
    ]);
    let outcome = run_to_end(
        agent_with("Assistant", "", vec![tool]),
        "Go.",
        model.clone(),
        |_| {},
    )
    .await;
    assert_eq!(
        outcome.result.expect("the run finishes").final_output,
        "recovered"
    );
    let second = &model.calls()[1];
    let output = second.input.iter().find_map(|i| match i {
        InputItem::ToolResult { output, .. } => Some(output.clone()),
        _ => None,
    });
    let output = output.unwrap();
    assert!(
        output.contains("the tool panicked") && !output.contains("secret-value"),
        "{output}"
    );
}

#[tokio::test]
async fn a_consumer_that_drops_the_event_stream_does_not_stop_the_run() {
    let model = scripted(vec![step(vec![assistant_message("finished anyway")])]);
    let handle = run_streamed(
        agent_with("Assistant", "", vec![]),
        "Go.".into(),
        RunConfig::new(model),
    );
    let lattice_agents::RunHandle { events, result, .. } = handle;
    drop(events);
    assert_eq!(
        result.await.unwrap().unwrap().final_output,
        "finished anyway"
    );
}

#[tokio::test]
async fn parallel_calls_to_the_same_tool_and_unknown_json_keys_are_handled() {
    let model = scripted(vec![
        step(vec![
            function_call("clock", "{\"city\":\"Oslo\",\"extra\":1}", "c1"),
            function_call("clock", "{\"city\":\"Rome\"}", "c2"),
            function_call("weather", "{\"city\":\"Oslo\"}", "c3"),
        ]),
        step(vec![assistant_message("ok")]),
    ]);
    let outcome = run_to_end(
        agent_with("A", "", vec![clock(), weather()]),
        "Go.",
        model,
        |_| {},
    )
    .await;
    let result = outcome.result.expect("the run finishes");
    let outputs: Vec<&str> = result
        .new_items
        .iter()
        .filter_map(|item| match item {
            RunItem::ToolOutput { output, .. } => Some(output.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(outputs, ["noon in Oslo", "noon in Rome", "sunny in Oslo"]);
}

#[tokio::test]
async fn an_erroring_tool_does_not_stop_its_siblings() {
    let model = scripted(vec![
        step(vec![
            function_call("explode", "{\"reason\":\"x\"}", "c1"),
            function_call("echo", "{\"text\":\"ok\"}", "c2"),
        ]),
        step(vec![assistant_message("done")]),
    ]);
    let outcome = run_to_end(
        agent_with("A", "", vec![explode(), echo()]),
        "Go.",
        model,
        |_| {},
    )
    .await;
    let result = outcome.result.expect("the run finishes");
    let outputs: Vec<&str> = result
        .new_items
        .iter()
        .filter_map(|item| match item {
            RunItem::ToolOutput { output, .. } => Some(output.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        outputs,
        [
            "An error occurred while running the tool. Please try again. Error: boom: x",
            "echo:ok"
        ]
    );
}

// ---------------------------------------------------------------------------
// A model that misbehaves badly
// ---------------------------------------------------------------------------

/// A model that panics, at the point chosen by `when`.
struct PanickingModel {
    when: PanicWhen,
    full_trace: bool,
}

#[derive(Clone, Copy)]
enum PanicWhen {
    StartingTheCall,
    WhileStreaming,
    AskedForItsName,
}

impl lattice_agents::Model for PanickingModel {
    fn name(&self) -> &str {
        if matches!(self.when, PanicWhen::AskedForItsName) {
            panic!("name bug");
        }
        "panicking-model"
    }

    fn config_for_trace(&self) -> Value {
        Value::Null
    }

    fn generation_trace(&self) -> GenerationTrace {
        if self.full_trace {
            GenerationTrace::Full
        } else {
            GenerationTrace::Bare
        }
    }

    fn stream(
        &self,
        _: lattice_agents::ModelRequest,
    ) -> futures::stream::BoxStream<'static, Result<lattice_agents::ModelEvent, ModelError>> {
        match self.when {
            PanicWhen::StartingTheCall => panic!("stream bug secret-in-model-panic"),
            PanicWhen::WhileStreaming => {
                futures::stream::poll_fn(|_| -> std::task::Poll<Option<_>> {
                    panic!("poll bug secret-in-model-panic")
                })
                .boxed()
            }
            PanicWhen::AskedForItsName => futures::stream::empty().boxed(),
        }
    }
}

#[tokio::test]
async fn a_model_that_panics_is_a_failed_call_that_does_not_repeat_the_panic_message() {
    for when in [PanicWhen::StartingTheCall, PanicWhen::WhileStreaming] {
        let collector = Collector::new();
        let mut config = RunConfig::new(Arc::new(PanickingModel {
            when,
            full_trace: false,
        }));
        config.processors = vec![collector.clone()];
        let handle = run_streamed(agent_with("Assistant", "", vec![]), "Go.".into(), config);
        let error = handle.result.await.unwrap().expect_err("the run fails");
        assert!(
            matches!(&error, RunError::Model(ModelError::Failed(text)) if text == "the model panicked"),
            "{error:?}"
        );
        let ends = collector.span_ends();
        assert!(
            !serde_json::to_string(&ends)
                .unwrap()
                .contains("secret-in-model-panic")
        );
        assert_eq!(collector.span_starts().len(), ends.len());
        assert_eq!(collector.calls().last().unwrap(), "trace_end");
    }
}

#[tokio::test]
async fn a_run_task_that_unwinds_still_ends_its_spans_and_its_trace() {
    // Asking the model for its name is outside every shield, so this panic really does
    // unwind the run task: the trace must still end and every span must still close.
    let collector = Collector::new();
    let mut config = RunConfig::new(Arc::new(PanickingModel {
        when: PanicWhen::AskedForItsName,
        full_trace: true,
    }));
    config.processors = vec![collector.clone()];
    let mut handle = run_streamed(agent_with("Assistant", "", vec![]), "Go.".into(), config);
    let joined = handle.result.await;
    assert!(joined.is_err(), "the task panicked");
    while handle.events.next().await.is_some() {}
    assert_eq!(
        collector.span_starts().len(),
        collector.span_ends().len(),
        "every span that started ended"
    );
    assert_eq!(collector.calls().last().unwrap(), "trace_end");
    assert_eq!(collector.trace_ends().len(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_run_works_the_same_on_a_multi_threaded_runtime() {
    let specialist = Agent::builder("Specialist agent").tool(lookup()).build();
    let triage = Agent::builder("Triage agent")
        .tool(weather())
        .handoff_to(specialist)
        .build();
    let model = scripted(vec![
        step(vec![
            function_call("weather", "{\"city\":\"Oslo\"}", "c1"),
            function_call("transfer_to_specialist_agent", "{}", "c2"),
        ]),
        step(vec![function_call("lookup", "{\"query\":\"q\"}", "c3")]),
        step(vec![assistant_message("finished")]),
    ]);
    let outcome = run_to_end(triage, "Go.", model, |_| {}).await;
    let result = outcome.result.expect("the run finishes");
    assert_eq!(
        (
            result.final_output.as_str(),
            result.last_agent.as_str(),
            result.turns
        ),
        ("finished", "Specialist agent", 3)
    );
    assert_eq!(
        outcome.collector.span_starts().len(),
        outcome.collector.span_ends().len()
    );
}

// ---------------------------------------------------------------------------
// Call ids (SDK `run_internal/tool_planning.py`, `_dedupe_processed_response_invocations`)
// ---------------------------------------------------------------------------

fn function_spans(outcome: &common::Outcome) -> usize {
    outcome
        .collector
        .span_ends()
        .iter()
        .filter(|span| span.data_type() == "function")
        .count()
}

#[tokio::test]
async fn an_identical_duplicate_call_in_one_response_runs_once() {
    // The same id, the same tool, the same arguments (the second with its keys in
    // another order and more spacing, which the SDK's fingerprint ignores).
    let model = scripted(vec![
        step(vec![
            function_call("lookup", "{\"query\":\"q\"}", "c1"),
            function_call("lookup", "{ \"query\": \"q\" }", "c1"),
            function_call("echo", "{\"text\":\"b\",\"extra\":1}", "c2"),
            function_call("echo", "{\"extra\":1,\"text\":\"b\"}", "c2"),
        ]),
        step(vec![assistant_message("done")]),
    ]);
    let outcome = run_to_end(
        agent_with("A", "", vec![lookup(), echo()]),
        "Go.",
        model.clone(),
        |_| {},
    )
    .await;
    let result = outcome.result.as_ref().expect("the run finishes");
    assert_eq!(function_spans(&outcome), 2, "one span per distinct call");
    let announced: Vec<&str> = result
        .new_items
        .iter()
        .filter_map(|item| match item {
            RunItem::ToolCall { call_id, .. } => Some(call_id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(announced, ["c1", "c2"], "a skipped call is not announced");
    let outputs = result
        .new_items
        .iter()
        .filter(|item| matches!(item, RunItem::ToolOutput { .. }))
        .count();
    assert_eq!(outputs, 2);

    // What the model is told next holds each call, and each answer, once.
    let next = &model.calls()[1];
    let calls_in_history: usize = next
        .input
        .iter()
        .map(|item| match item {
            InputItem::Assistant { tool_calls, .. } => tool_calls.len(),
            _ => 0,
        })
        .sum();
    let results_in_history = next
        .input
        .iter()
        .filter(|item| matches!(item, InputItem::ToolResult { .. }))
        .count();
    assert_eq!((calls_in_history, results_in_history), (2, 2));
}

#[tokio::test]
async fn a_reused_call_id_for_a_different_invocation_is_a_behaviour_error_before_anything_runs() {
    for response in [
        // Different arguments.
        vec![
            function_call("echo", "{\"text\":\"a\"}", "c1"),
            function_call("echo", "{\"text\":\"b\"}", "c1"),
        ],
        // Different tools.
        vec![
            function_call("echo", "{\"text\":\"a\"}", "c1"),
            function_call("lookup", "{\"text\":\"a\"}", "c1"),
        ],
    ] {
        let model = scripted(vec![step(response)]);
        let outcome = run_to_end(
            agent_with("A", "", vec![echo(), lookup()]),
            "Go.",
            model,
            |_| {},
        )
        .await;
        let error = outcome.result.as_ref().expect_err("the run fails");
        assert_eq!(error.sdk_name(), "ModelBehaviorError");
        assert_eq!(
            error.to_string(),
            "Model reused a tool call ID for a different invocation in one response. Use a unique call ID for each tool invocation."
        );
        assert_eq!(function_spans(&outcome), 0, "no tool started");
        assert!(
            !outcome.names.contains(&"tool_called"),
            "nothing from the bad response was announced: {:?}",
            outcome.names
        );
    }

    // A function and a handoff may not share an id either.
    let specialist = Agent::builder("Specialist agent").build();
    let triage = Agent::builder("Triage agent")
        .tool(echo())
        .handoff_to(specialist)
        .build();
    let model = scripted(vec![step(vec![
        function_call("echo", "{\"text\":\"a\"}", "c1"),
        function_call("transfer_to_specialist_agent", "{}", "c1"),
    ])]);
    let outcome = run_to_end(triage, "Go.", model, |_| {}).await;
    let error = outcome.result.as_ref().expect_err("the run fails");
    assert_eq!(error.sdk_name(), "ModelBehaviorError");
    assert!(error.to_string().starts_with("Model reused a tool call ID"));
    assert_eq!(function_spans(&outcome), 0);
}

#[tokio::test]
async fn a_refusal_ends_the_run_only_when_the_response_asks_for_nothing_else() {
    // Nothing else in the response: ModelRefusalError (`turn_resolution.py:952-988`),
    // after the turn's items are announced, and the agent span says so.
    let model = scripted(vec![step(vec![
        assistant_message("Sorry."),
        OutputItem::Refusal {
            text: "I can't help with that.".into(),
        },
    ])]);
    let outcome = run_to_end(agent_with("A", "", vec![echo()]), "Go.", model, |_| {}).await;
    let error = outcome.result.as_ref().expect_err("the run fails");
    assert_eq!(error.sdk_name(), "ModelRefusalError");
    assert_eq!(
        error.to_string(),
        "Model refused to produce output: I can't help with that."
    );
    assert!(
        matches!(error, RunError::ModelRefusal { refusal } if refusal == "I can't help with that.")
    );
    assert_eq!(
        outcome.names,
        [
            "agent_updated_stream_event",
            "raw_text_delta",
            "message_output_created"
        ]
    );
    let ends = outcome.collector.span_ends();
    let agent = find_span(&ends, "agent").error.as_ref().unwrap();
    assert_eq!(agent.message, "Error in agent run");
    assert_eq!(
        agent.data.as_ref().unwrap()["error"],
        "Model refused to produce output: I can't help with that."
    );

    // With a call in the same response the call is what happens, as in the SDK.
    let model = scripted(vec![
        step(vec![
            OutputItem::Refusal {
                text: "I won't say more.".into(),
            },
            function_call("echo", "{\"text\":\"a\"}", "c1"),
        ]),
        step(vec![assistant_message("done")]),
    ]);
    let outcome = run_to_end(agent_with("A", "", vec![echo()]), "Go.", model, |_| {}).await;
    assert_eq!(
        outcome.result.expect("the run finishes").final_output,
        "done"
    );
}

#[tokio::test]
async fn a_call_with_an_empty_id_is_refused_with_the_sdks_words_before_anything_runs() {
    let model = scripted(vec![step(vec![
        function_call("echo", "{\"text\":\"a\"}", "c1"),
        function_call("echo", "{\"text\":\"b\"}", ""),
    ])]);
    let outcome = run_to_end(agent_with("A", "", vec![echo()]), "Go.", model, |_| {}).await;
    let error = outcome.result.as_ref().expect_err("the run fails");
    assert_eq!(error.sdk_name(), "ModelBehaviorError");
    assert_eq!(
        error.to_string(),
        "Tool invocations require a non-empty string call ID before execution."
    );
    assert_eq!(function_spans(&outcome), 0, "not even the first call ran");
}

// ---------------------------------------------------------------------------
// List input (`run_streamed_items`)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_run_with_no_input_is_refused_before_it_starts_anything() {
    let model = scripted(vec![step(vec![assistant_message("never")])]);
    let outcome = run_items_to_end(
        agent_with("Assistant", "", vec![]),
        vec![],
        model.clone(),
        |_| {},
    )
    .await;
    match outcome.result {
        Err(RunError::User(message)) => assert_eq!(message, "the run has no input"),
        other => panic!("expected the no-input refusal, got {other:?}"),
    }
    assert!(outcome.events.is_empty(), "no event");
    assert!(outcome.collector.calls().is_empty(), "no trace and no span");
    assert!(model.calls().is_empty(), "no model call");
}

#[tokio::test]
async fn the_string_path_is_the_list_of_one_user_item() {
    let steps = || {
        vec![
            step(vec![function_call("echo", "{\"text\":\"hi\"}", "call_1")]),
            step(vec![assistant_message("Echoed: hi")]),
        ]
    };
    let text_model = scripted(steps());
    let text = run_to_end(
        agent_with("Assistant", "Be brief.", vec![echo()]),
        "Echo hi.",
        text_model.clone(),
        |_| {},
    )
    .await;
    let items_model = scripted(steps());
    let items = run_items_to_end(
        agent_with("Assistant", "Be brief.", vec![echo()]),
        vec![InputItem::User("Echo hi.".into())],
        items_model.clone(),
        |_| {},
    )
    .await;
    assert_eq!(text.names, items.names);
    assert_eq!(text.collector.forest(), items.collector.forest());
    assert_eq!(text_model.calls(), items_model.calls());
    assert_eq!(
        text.result.unwrap().final_output,
        items.result.unwrap().final_output
    );
}

/// A guardrail that remembers every text it was given.
fn recording_guardrail(seen: Arc<std::sync::Mutex<Vec<String>>>) -> InputGuardrail {
    InputGuardrail::new("record", move |_, text| {
        seen.lock().unwrap().push(text);
        async { GuardrailOutput::pass() }
    })
}

#[tokio::test]
async fn an_input_guardrail_sees_the_last_user_item_or_nothing_when_there_is_none() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let agent = Agent::builder("Assistant")
        .input_guardrail(recording_guardrail(seen.clone()))
        .build();
    let outcome = run_items_to_end(
        agent.clone(),
        vec![
            InputItem::User("first".into()),
            InputItem::Assistant {
                text: Some("between".into()),
                tool_calls: vec![],
            },
            InputItem::User("second".into()),
            InputItem::Assistant {
                text: Some("after".into()),
                tool_calls: vec![],
            },
        ],
        scripted(vec![step(vec![assistant_message("ok")])]),
        |_| {},
    )
    .await;
    outcome.result.expect("the run finishes");
    let outcome = run_items_to_end(
        agent,
        vec![InputItem::Assistant {
            text: Some("no user here".into()),
            tool_calls: vec![],
        }],
        scripted(vec![step(vec![assistant_message("ok")])]),
        |_| {},
    )
    .await;
    outcome.result.expect("the run finishes");
    assert_eq!(*seen.lock().unwrap(), ["second", ""]);
}

// ---------------------------------------------------------------------------
// Sessions (`RunConfig::session`)
// ---------------------------------------------------------------------------

/// How a test session answers one kind of call.
#[derive(Clone, Copy, PartialEq)]
enum Answer {
    Fine,
    Fail,
    Panic,
    Never,
}

/// A session whose reads and writes answer as a test chooses, over a
/// [`MemorySession`] that keeps what it is given.
struct ScriptedSession {
    read: Answer,
    write: Answer,
    inner: Arc<MemorySession>,
}

impl ScriptedSession {
    fn new(read: Answer, write: Answer) -> Arc<Self> {
        Arc::new(Self {
            read,
            write,
            inner: MemorySession::new(),
        })
    }
}

impl Session for ScriptedSession {
    fn get_items(
        &self,
        limit: Option<usize>,
    ) -> BoxFuture<'static, Result<Vec<InputItem>, SessionError>> {
        match self.read {
            Answer::Fine => self.inner.get_items(limit),
            Answer::Fail => async { Err(SessionError::new("the store is locked")) }.boxed(),
            Answer::Panic => panic!("a session bug with conversation text"),
            Answer::Never => futures::future::pending().boxed(),
        }
    }

    fn add_items(&self, items: Vec<InputItem>) -> BoxFuture<'static, Result<(), SessionError>> {
        match self.write {
            Answer::Fine => self.inner.add_items(items),
            Answer::Fail => async { Err(SessionError::new("the disk is full")) }.boxed(),
            Answer::Panic => async { panic!("a session bug with conversation text") }.boxed(),
            Answer::Never => futures::future::pending().boxed(),
        }
    }
}

fn with_session(session: Arc<dyn Session>) -> impl FnOnce(&mut RunConfig) {
    move |config: &mut RunConfig| config.session = Some(session)
}

fn write_failures(outcome: &common::Outcome) -> Vec<String> {
    outcome
        .events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::SessionWriteFailed { message } => Some(message.clone()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn a_session_is_read_first_and_its_items_come_before_the_input_in_every_model_call() {
    let session = MemorySession::holding(vec![
        InputItem::User("earlier".into()),
        InputItem::Assistant {
            text: Some("an earlier answer".into()),
            tool_calls: vec![],
        },
    ]);
    let model = scripted(vec![
        step(vec![function_call("echo", "{\"text\":\"a\"}", "call_1")]),
        step(vec![assistant_message("done")]),
    ]);
    let outcome = run_to_end(
        agent_with("Assistant", "", vec![echo()]),
        "now",
        model.clone(),
        with_session(session.clone()),
    )
    .await;
    outcome.result.expect("the run finishes");
    assert_eq!(session.reads(), [None], "read once, every item");
    let calls = model.calls();
    assert_eq!(
        calls[0].input,
        [
            InputItem::User("earlier".into()),
            InputItem::Assistant {
                text: Some("an earlier answer".into()),
                tool_calls: vec![],
            },
            InputItem::User("now".into()),
        ]
    );
    assert_eq!(calls[1].input[..3], calls[0].input[..]);
    let call = ToolCallItem {
        call_id: "call_1".into(),
        name: "echo".into(),
        arguments: "{\"text\":\"a\"}".into(),
    };
    assert_eq!(
        session.adds(),
        [
            vec![InputItem::User("now".into())],
            vec![
                InputItem::Assistant {
                    text: None,
                    tool_calls: vec![call],
                },
                InputItem::ToolResult {
                    call_id: "call_1".into(),
                    output: "echo:a".into(),
                },
            ],
            vec![InputItem::Assistant {
                text: Some("done".into()),
                tool_calls: vec![],
            }],
        ]
    );
}

#[tokio::test]
async fn a_session_that_cannot_be_read_ends_the_run_before_it_starts_anything() {
    for (read, said) in [
        (
            Answer::Fail,
            "the session could not be read: the store is locked",
        ),
        (
            Answer::Panic,
            "the session could not be read: the session panicked",
        ),
    ] {
        let model = scripted(vec![step(vec![assistant_message("never")])]);
        let outcome = run_to_end(
            agent_with("Assistant", "", vec![]),
            "Go.",
            model.clone(),
            with_session(ScriptedSession::new(read, Answer::Fine)),
        )
        .await;
        match &outcome.result {
            Err(RunError::User(message)) => assert_eq!(message, said),
            other => panic!("expected a session refusal, got {other:?}"),
        }
        assert!(outcome.events.is_empty(), "no event");
        assert!(outcome.collector.calls().is_empty(), "no trace and no span");
        assert!(model.calls().is_empty(), "no model call");
        assert!(!format!("{:?}", outcome.result).contains("conversation text"));
    }
}

#[tokio::test]
async fn a_session_that_cannot_be_written_is_announced_once_and_the_run_goes_on() {
    for (write, said) in [
        (Answer::Fail, "the disk is full"),
        (Answer::Panic, "the session panicked"),
    ] {
        let model = scripted(vec![
            step(vec![function_call("echo", "{\"text\":\"a\"}", "call_1")]),
            step(vec![assistant_message("done")]),
        ]);
        let outcome = run_to_end(
            agent_with("Assistant", "", vec![echo()]),
            "Go.",
            model,
            with_session(ScriptedSession::new(Answer::Fine, write)),
        )
        .await;
        assert_eq!(
            write_failures(&outcome),
            [said],
            "three writes failed, one event"
        );
        // The SDK's stream is unchanged by it.
        assert_eq!(
            outcome.names,
            [
                "agent_updated_stream_event",
                "tool_called",
                "tool_output",
                "raw_text_delta",
                "message_output_created"
            ]
        );
        assert_eq!(
            outcome.result.expect("the run finishes").final_output,
            "done"
        );
    }
}

#[tokio::test]
async fn a_failed_turn_saves_nothing_and_the_completed_turns_are_kept() {
    let session = MemorySession::new();
    let model = scripted(vec![
        step(vec![function_call("echo", "{\"text\":\"a\"}", "call_1")]),
        ScriptedStep::error(ModelError::Status(500)),
    ]);
    let outcome = run_to_end(
        agent_with("Assistant", "", vec![echo()]),
        "Go.",
        model,
        with_session(session.clone()),
    )
    .await;
    outcome.result.expect_err("the second model call fails");
    let adds = session.adds();
    assert_eq!(adds.len(), 2, "the input, then the one completed turn");
    assert_eq!(adds[0], [InputItem::User("Go.".into())]);
    assert_eq!(adds[1].len(), 2);
}

#[tokio::test]
async fn a_run_cancelled_while_its_session_is_read_ends_cancelled_with_no_trace() {
    let model = scripted(vec![step(vec![assistant_message("never")])]);
    let collector = Collector::new();
    let mut config = RunConfig::new(model.clone());
    config.processors = vec![collector.clone()];
    config.session = Some(ScriptedSession::new(Answer::Never, Answer::Fine));
    let handle = run_streamed(agent_with("Assistant", "", vec![]), "Go.".into(), config);
    handle.control.cancel(CancelMode::Immediate);
    let result = tokio::time::timeout(Duration::from_secs(30), handle.result)
        .await
        .expect("the cancel is honoured")
        .expect("the run task does not panic");
    assert!(matches!(result, Err(RunError::Cancelled)), "{result:?}");
    assert!(collector.calls().is_empty());
    assert!(model.calls().is_empty());
}

// ---------------------------------------------------------------------------
// Tool approval (`NeedsApproval`, `RunConfig::approvals`)
// ---------------------------------------------------------------------------

fn approval_events(outcome: &common::Outcome) -> Vec<StreamEvent> {
    outcome
        .events
        .iter()
        .filter(|event| {
            matches!(
                event,
                StreamEvent::ToolApprovalRequested { .. }
                    | StreamEvent::ToolApprovalResolved { .. }
            )
        })
        .cloned()
        .collect()
}

#[tokio::test]
async fn an_approval_is_asked_for_with_the_parsed_arguments_and_announced_both_ways() {
    let port = ScriptedApprovals::new(vec![ApprovalDecision::Approve]);
    let model = scripted(vec![
        step(vec![function_call(
            "delete_file",
            "{\"path\":\"a.txt\"}",
            "call_d",
        )]),
        step(vec![assistant_message("Deleted.")]),
    ]);
    let outcome = run_to_end(
        agent_with("Assistant", "", vec![delete_file()]),
        "Go.",
        model,
        |config| config.approvals = Some(port.clone()),
    )
    .await;
    outcome.result.as_ref().expect("the run finishes");
    let asked = port.asked();
    assert_eq!(asked.len(), 1);
    assert_eq!(
        (
            asked[0].agent.as_str(),
            asked[0].call_id.as_str(),
            asked[0].tool.as_str()
        ),
        ("Assistant", "call_d", "delete_file")
    );
    assert_eq!(asked[0].arguments, json!({"path": "a.txt"}));
    assert_eq!(
        approval_events(&outcome),
        [
            StreamEvent::ToolApprovalRequested {
                agent: "Assistant".into(),
                call_id: "call_d".into(),
                tool: "delete_file".into(),
                arguments: json!({"path": "a.txt"}),
            },
            StreamEvent::ToolApprovalResolved {
                agent: "Assistant".into(),
                call_id: "call_d".into(),
                approved: true,
            },
        ]
    );
    // Asked after the call was announced and answered before its output.
    let order: Vec<&str> = outcome.events.iter().map(StreamEvent::sdk_name).collect();
    let at = |name: &str| order.iter().position(|n| *n == name).unwrap();
    assert!(at("tool_called") < at("tool_approval_requested"));
    assert!(at("tool_approval_resolved") < at("tool_output"));
}

#[tokio::test]
async fn arguments_that_are_not_an_object_are_answered_without_asking() {
    let port = ScriptedApprovals::new(vec![ApprovalDecision::Approve]);
    let model = scripted(vec![
        step(vec![function_call("delete_file", "{\"path\": ", "call_d")]),
        step(vec![assistant_message("Retried.")]),
    ]);
    let outcome = run_to_end(
        agent_with("Assistant", "", vec![delete_file()]),
        "Go.",
        model,
        |config| config.approvals = Some(port.clone()),
    )
    .await;
    let result = outcome.result.as_ref().expect("the run finishes");
    assert!(port.asked().is_empty(), "nothing valid to approve");
    assert!(approval_events(&outcome).is_empty());
    let output = result.new_items.iter().find_map(|item| match item {
        RunItem::ToolOutput { output, .. } => Some(output.as_str()),
        _ => None,
    });
    assert_eq!(
        output,
        Some(
            "An error occurred while running the tool. Please try again. \
             Error: Invalid JSON input for tool delete_file"
        )
    );
}

/// A port that answers only once `released` is notified.
struct HeldPort {
    released: Arc<tokio::sync::Notify>,
}

impl ApprovalPort for HeldPort {
    fn request(&self, _: ApprovalRequest) -> BoxFuture<'static, ApprovalDecision> {
        let released = self.released.clone();
        async move {
            released.notified().await;
            ApprovalDecision::Approve
        }
        .boxed()
    }
}

#[tokio::test]
async fn the_other_calls_of_a_response_run_while_one_waits_and_their_outputs_come_first() {
    // The wait ends only when the second call's handler has run: if a waiting
    // call held up the others, this run would never finish. The free call's
    // output comes before the approved one's, as in the SDK (spec 22.6 A3a;
    // the golden approval_needed_before_free_calls pins the full order).
    let released = Arc::new(tokio::sync::Notify::new());
    let signal = released.clone();
    let other = FunctionTool::new(
        "other",
        "",
        strict_object_schema(json!({}), &[]),
        move |_, _| {
            let signal = signal.clone();
            async move {
                signal.notify_one();
                Ok("other ran".to_owned())
            }
        },
    );
    let model = scripted(vec![
        step(vec![
            function_call("delete_file", "{\"path\":\"a.txt\"}", "call_d"),
            function_call("other", "{}", "call_o"),
        ]),
        step(vec![assistant_message("Both.")]),
    ]);
    let run = run_to_end(
        agent_with("Assistant", "", vec![delete_file(), other]),
        "Go.",
        model,
        |config| config.approvals = Some(Arc::new(HeldPort { released })),
    );
    let outcome = tokio::time::timeout(Duration::from_secs(30), run)
        .await
        .expect("the other call ran while the first waited");
    let result = outcome.result.expect("the run finishes");
    let outputs: Vec<(&str, &str)> = result
        .new_items
        .iter()
        .filter_map(|item| match item {
            RunItem::ToolOutput {
                call_id, output, ..
            } => Some((call_id.as_str(), output.as_str())),
            _ => None,
        })
        .collect();
    assert_eq!(
        outputs,
        [("call_o", "other ran"), ("call_d", "deleted a.txt")]
    );
}

/// A port that panics.
struct PanickingPort;

impl ApprovalPort for PanickingPort {
    fn request(&self, _: ApprovalRequest) -> BoxFuture<'static, ApprovalDecision> {
        panic!("a port bug")
    }
}

#[tokio::test]
async fn a_port_that_panics_rejects_and_the_run_goes_on() {
    let model = scripted(vec![
        step(vec![function_call(
            "delete_file",
            "{\"path\":\"a.txt\"}",
            "call_d",
        )]),
        step(vec![assistant_message("Kept.")]),
    ]);
    let outcome = run_to_end(
        agent_with("Assistant", "", vec![delete_file()]),
        "Go.",
        model,
        |config| config.approvals = Some(Arc::new(PanickingPort)),
    )
    .await;
    let result = outcome.result.as_ref().expect("the run finishes");
    let output = result.new_items.iter().find_map(|item| match item {
        RunItem::ToolOutput { output, .. } => Some(output.as_str()),
        _ => None,
    });
    assert_eq!(output, Some("Tool execution was not approved."));
    assert_eq!(
        approval_events(&outcome).last(),
        Some(&StreamEvent::ToolApprovalResolved {
            agent: "Assistant".into(),
            call_id: "call_d".into(),
            approved: false,
        })
    );
}

#[tokio::test]
async fn a_rejected_calls_span_records_its_input_and_no_output_and_the_note_reaches_the_model() {
    let note = "x".repeat(lattice_agents::MAX_NOTE_CHARS + 10);
    let port = ScriptedApprovals::new(vec![ApprovalDecision::Reject {
        note: Some(note.clone()),
    }]);
    let model = scripted(vec![
        step(vec![function_call(
            "delete_file",
            "{\"path\":\"a.txt\"}",
            "call_d",
        )]),
        step(vec![assistant_message("Kept.")]),
    ]);
    let outcome = run_to_end(
        agent_with("Assistant", "", vec![delete_file()]),
        "Go.",
        model.clone(),
        |config| config.approvals = Some(port.clone()),
    )
    .await;
    outcome.result.expect("the run finishes");
    let ends = outcome.collector.span_ends();
    let function = find_span(&ends, "function");
    assert_eq!(function.span_data["input"], json!("{\"path\":\"a.txt\"}"));
    assert_eq!(function.span_data["output"], Value::Null);
    assert!(function.error.is_none());
    let told = model.calls()[1]
        .input
        .iter()
        .find_map(|item| match item {
            InputItem::ToolResult { output, .. } => Some(output.clone()),
            _ => None,
        })
        .unwrap();
    let expected = format!(
        "Tool execution was not approved. The user said: {}",
        &note[..lattice_agents::MAX_NOTE_CHARS]
    );
    assert_eq!(told, expected);
}

// ---------------------------------------------------------------------------
// Steering (`RunControl::steer`); the falsifiers AF5 and AF7 are in src/run/af_tests.rs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_steered_message_is_announced_and_saved_with_the_turn_that_took_it() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let (signal, wait) = (started.clone(), release.clone());
    let held = FunctionTool::new(
        "wait",
        "",
        strict_object_schema(json!({}), &[]),
        move |_, _| {
            let (signal, wait) = (signal.clone(), wait.clone());
            async move {
                signal.notify_one();
                wait.notified().await;
                Ok("waited".to_owned())
            }
        },
    );
    let session = MemorySession::new();
    let mut config = RunConfig::new(scripted(vec![
        step(vec![function_call("wait", "{}", "call_1")]),
        step(vec![assistant_message("Done.")]),
    ]));
    config.session = Some(session.clone());
    let mut handle = run_streamed(
        agent_with("Assistant", "", vec![held]),
        "Go.".into(),
        config,
    );
    started.notified().await;
    handle.control.steer("also this".into()).unwrap();
    release.notify_one();
    let mut events = Vec::new();
    while let Some(event) = handle.events.next().await {
        events.push(event);
    }
    handle.result.await.unwrap().expect("the run finishes");
    assert!(events.contains(&StreamEvent::Steered {
        agent: "Assistant".into(),
        text: "also this".into(),
    }));
    let adds = session.adds();
    assert_eq!(adds.len(), 3, "the input and two turns");
    assert_eq!(
        adds[2],
        [
            InputItem::User("also this".into()),
            InputItem::Assistant {
                text: Some("Done.".into()),
                tool_calls: vec![],
            },
        ]
    );
}
