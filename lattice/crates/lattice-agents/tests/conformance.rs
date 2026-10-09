//! The port against the real SDK.
//!
//! `tools/lattice_sdk_goldens.py` runs openai-agents 0.22.3 over the scenarios
//! below (with its own scripted model and a collecting trace processor that
//! exports nothing) and writes what happened to `tests/goldens/<scenario>.json`:
//! the output or the exception's name, the stream events in order, and the span
//! forest. This file replays each scenario through `lattice-agents` and demands
//! the same, field for field. A green run here means the port produces the span
//! tree, the stream events and the error text the SDK produces for these
//! scenarios; it says nothing about scenarios that are not listed.
//!
//! The comparison is exact except for what the goldens' header lists as
//! excluded (ids, timestamps, `Span.export()`'s optional `metadata`). Anything
//! that must differ on purpose belongs in [`DEVIATIONS`] with its reason.

mod common;

use std::sync::Arc;

use lattice_agents::testing::{ScriptedStep, assistant_message, function_call};
use lattice_agents::{
    Agent, ApprovalDecision, ApprovalPort, FunctionTool, GuardrailOutput, InputGuardrail,
    InputItem, ModelError, OutputGuardrail, OutputItem, RunConfig, RunItem, Session, StreamEvent,
    ToolCallItem, ToolErrorPolicy,
};
use serde_json::{Value, json};

use common::{
    Collector, MemorySession, Outcome, ScriptedApprovals, Start, clock, delete_file, echo, explode,
    lookup, remove, run_into, scripted, weather,
};

/// Deliberate differences from the recorded SDK behaviour:
/// `(scenario, JSON pointer into the normalised output, reason)`. The pointed-to
/// value is removed from BOTH sides before comparing.
const DEVIATIONS: &[(&str, &str, &str)] = &[
    (
        "model_error",
        "/error",
        "The SDK raises the model's own exception, whose Python class name (`Exception`) is the golden. \
         The port's error is `RunError::Model(ModelError)`, named `ModelError` by `sdk_name()`.",
    ),
    (
        "model_error",
        "/spans/0/children/0/children/0/children/0/error/data/name",
        "The SDK records the exception's Python class name on the generation span; the port records the \
         `ModelError` variant (`Failed`). The message text next to it is compared.",
    ),
];

/// What a scenario's run starts from: one string (the `run_streamed` path every
/// golden recorded before list input exercises) or a list of input items.
enum ScenarioInput {
    Text(&'static str),
    Items(Vec<InputItem>),
}

struct Scenario {
    name: &'static str,
    agent: Arc<Agent>,
    input: ScenarioInput,
    steps: Vec<ScriptedStep>,
    max_turns: Option<u32>,
    include_sensitive_data: bool,
    include_task_and_turn_spans: bool,
    workflow_name: Option<&'static str>,
    /// The golden also holds what each model call was given (`model_inputs`).
    record_model_inputs: bool,
    /// The run keeps its conversation in a session (a [`MemorySession`]); the
    /// golden holds what each `add_items` call was given (`session_add_items`).
    session: bool,
    /// Runs made after the first, in order, on the same model, session and
    /// trace collector (their streams concatenate; their traces are roots).
    then: Vec<&'static str>,
    /// The run asks a port that gives these decisions, in order (the goldens
    /// are the SDK's interrupted run and its resume, stitched into one).
    approvals: Option<Vec<ApprovalDecision>>,
    /// The golden also holds every `tool_called` and `tool_output` event with
    /// its call id (`tool_events`).
    record_tool_events: bool,
}

impl Scenario {
    fn new(
        name: &'static str,
        agent: Arc<Agent>,
        input: &'static str,
        steps: Vec<ScriptedStep>,
    ) -> Self {
        Self {
            name,
            agent,
            input: ScenarioInput::Text(input),
            steps,
            max_turns: None,
            include_sensitive_data: true,
            include_task_and_turn_spans: true,
            workflow_name: None,
            record_model_inputs: false,
            session: false,
            then: Vec::new(),
            approvals: None,
            record_tool_events: false,
        }
    }

    /// The run asks for approval and is given these decisions; the golden
    /// records the model inputs.
    fn deciding(mut self, decisions: Vec<ApprovalDecision>) -> Self {
        self.approvals = Some(decisions);
        self.record_model_inputs = true;
        self
    }

    /// The run keeps its conversation in a session; the golden records the
    /// model inputs and the session's writes.
    fn with_session(mut self) -> Self {
        self.session = true;
        self.record_model_inputs = true;
        self
    }

    /// Another run, after the ones before, from this message.
    fn then(mut self, input: &'static str) -> Self {
        self.then.push(input);
        self
    }

    /// A scenario started from a list of input items; its golden records the
    /// model inputs.
    fn from_items(
        name: &'static str,
        agent: Arc<Agent>,
        input: Vec<InputItem>,
        steps: Vec<ScriptedStep>,
    ) -> Self {
        Self {
            input: ScenarioInput::Items(input),
            record_model_inputs: true,
            ..Self::new(name, agent, "", steps)
        }
    }

    /// The golden records which call each tool event belongs to.
    fn with_tool_events(mut self) -> Self {
        self.record_tool_events = true;
        self
    }

    fn max_turns(mut self, max_turns: u32) -> Self {
        self.max_turns = Some(max_turns);
        self
    }

    fn sensitive_data_off(mut self) -> Self {
        self.include_sensitive_data = false;
        self
    }

    fn without_task_and_turn_spans(mut self) -> Self {
        self.include_task_and_turn_spans = false;
        self
    }

    fn workflow(mut self, name: &'static str) -> Self {
        self.workflow_name = Some(name);
        self
    }
}

fn step(output: Vec<OutputItem>) -> ScriptedStep {
    ScriptedStep::respond(output)
}

fn assistant(tools: Vec<FunctionTool>) -> Arc<Agent> {
    Agent::builder("Assistant")
        .instructions("Be brief.")
        .tools(tools)
        .build()
}

/// The Triage agent of the goldens that has a tool and a handoff (`echo`, and
/// `transfer_to_specialist_agent`).
fn assistant_with_a_handoff() -> Arc<Agent> {
    Agent::builder("Triage agent")
        .instructions("Route the request.")
        .tool(echo())
        .handoff_to(
            Agent::builder("Specialist agent")
                .instructions("You are the specialist.")
                .build(),
        )
        .build()
}

fn custom_policy(tool: FunctionTool) -> FunctionTool {
    tool.with_error_policy(ToolErrorPolicy::Custom(Arc::new(|error| {
        format!("custom:{error}")
    })))
}

fn explode_custom() -> FunctionTool {
    let mut tool = custom_policy(explode());
    tool.name = "explode_custom".into();
    tool
}

fn echo_custom() -> FunctionTool {
    let mut tool = custom_policy(echo());
    tool.name = "echo_custom".into();
    tool
}

fn allow_input(name: &'static str) -> InputGuardrail {
    InputGuardrail::new(name, |_, _| async {
        GuardrailOutput {
            tripwire_triggered: false,
            output_info: json!({"checked": true}),
        }
    })
}

fn allow_output(name: &'static str) -> OutputGuardrail {
    OutputGuardrail::new(name, |_, _| async {
        GuardrailOutput {
            tripwire_triggered: false,
            output_info: json!({"checked": true}),
        }
    })
}

fn scenarios() -> Vec<Scenario> {
    let specialist = Agent::builder("Specialist agent")
        .instructions("You are the specialist.")
        .tool(lookup())
        .build();
    let triage = Agent::builder("Triage agent")
        .instructions("Route the request.")
        .handoff_to(specialist)
        .build();
    let billing = Agent::builder("Billing agent")
        .instructions("Billing.")
        .build();
    let support = Agent::builder("Support agent")
        .instructions("Support.")
        .build();
    let router = Agent::builder("Router")
        .instructions("Route.")
        .handoff_to(billing)
        .handoff_to(support)
        .build();
    let echo_and_handoff = Agent::builder("Triage agent")
        .instructions("Route the request.")
        .tool(echo())
        .handoff_to(
            Agent::builder("Specialist agent")
                .instructions("You are the specialist.")
                .build(),
        )
        .build();

    let with_input_guardrails = |guardrails: Vec<InputGuardrail>| {
        let mut builder = Agent::builder("Assistant").instructions("Be brief.");
        for guardrail in guardrails {
            builder = builder.input_guardrail(guardrail);
        }
        builder.build()
    };
    let with_output_guardrails = |guardrails: Vec<OutputGuardrail>| {
        let mut builder = Agent::builder("Assistant").instructions("Be brief.");
        for guardrail in guardrails {
            builder = builder.output_guardrail(guardrail);
        }
        builder.build()
    };
    let refuse_all = InputGuardrail::new("refuse_all", |_, _| async {
        GuardrailOutput::trip(json!({"reason": "refused"}))
    });
    let no_output = OutputGuardrail::new("no_output", |_, _| async {
        GuardrailOutput::trip(json!({"reason": "refused"}))
    });

    vec![
        Scenario::new(
            "text_only",
            assistant(vec![]),
            "Say hello.",
            vec![step(vec![assistant_message("Hello there.")]).with_tokens(11, 3)],
        ),
        Scenario::new(
            "one_tool",
            assistant(vec![echo()]),
            "Echo hi.",
            vec![
                step(vec![function_call("echo", "{\"text\":\"hi\"}", "call_1")]).with_tokens(20, 5),
                step(vec![assistant_message("Echoed: hi")]).with_tokens(30, 4),
            ],
        ),
        Scenario::new(
            "two_parallel_tools",
            assistant(vec![weather(), clock()]),
            "Weather and time in Oslo.",
            vec![
                step(vec![
                    function_call("weather", "{\"city\":\"Oslo\"}", "call_w"),
                    function_call("clock", "{\"city\":\"Oslo\"}", "call_c"),
                ]),
                step(vec![assistant_message("Sunny at noon in Oslo.")]),
            ],
        ),
        Scenario::new(
            "handoff_then_tool",
            triage,
            "Find the manual.",
            vec![
                step(vec![function_call(
                    "transfer_to_specialist_agent",
                    "{}",
                    "call_h",
                )])
                .with_tokens(9, 2),
                step(vec![function_call(
                    "lookup",
                    "{\"query\":\"manual\"}",
                    "call_t",
                )])
                .with_tokens(15, 6),
                step(vec![assistant_message("The manual was found.")]).with_tokens(25, 7),
            ],
        ),
        Scenario::new(
            "multiple_handoffs",
            router,
            "Bill me and support me.",
            vec![
                step(vec![
                    function_call("transfer_to_billing_agent", "{}", "call_b"),
                    function_call("transfer_to_support_agent", "{}", "call_s"),
                ]),
                step(vec![assistant_message("Billing here.")]),
            ],
        ),
        Scenario::new(
            "input_guardrail_pass",
            with_input_guardrails(vec![allow_input("allow_all")]),
            "Say ok.",
            vec![step(vec![assistant_message("ok")])],
        ),
        Scenario::new(
            "input_guardrail_trip",
            with_input_guardrails(vec![refuse_all]),
            "Do something forbidden.",
            vec![],
        ),
        Scenario::new(
            "output_guardrail_trip",
            with_output_guardrails(vec![no_output]),
            "Say something.",
            vec![step(vec![assistant_message("Something.")]).with_tokens(8, 2)],
        ),
        Scenario::new(
            "tool_error_default_policy",
            assistant(vec![explode()]),
            "Try the tool.",
            vec![
                step(vec![function_call(
                    "explode",
                    "{\"reason\":\"test\"}",
                    "call_x",
                )]),
                step(vec![assistant_message("The tool failed.")]),
            ],
        ),
        Scenario::new(
            "tool_error_custom_policy",
            assistant(vec![explode_custom(), echo_custom()]),
            "Two tools, both fail.",
            vec![
                step(vec![
                    function_call("explode_custom", "{\"reason\":\"test\"}", "call_x"),
                    function_call("echo_custom", "{\"text\": \"hi\"", "call_j"),
                ]),
                step(vec![assistant_message("Both failed.")]),
            ],
        ),
        Scenario::new(
            "tool_invalid_json",
            assistant(vec![echo()]),
            "Call echo badly.",
            vec![
                step(vec![function_call("echo", "{\"text\": \"hi\"", "call_j")]),
                step(vec![assistant_message("Retried.")]),
            ],
        ),
        Scenario::new(
            "tool_non_object_arguments",
            assistant(vec![echo()]),
            "Call echo with a list.",
            vec![
                step(vec![function_call("echo", "[1, 2]", "call_l")]),
                step(vec![assistant_message("Retried.")]),
            ],
        ),
        Scenario::new(
            "tool_not_found",
            assistant(vec![echo()]),
            "Call a tool that is not there.",
            vec![step(vec![function_call("nope", "{}", "call_n")])],
        ),
        Scenario::new(
            "text_with_tool_call",
            assistant(vec![echo()]),
            "Say something and echo.",
            vec![
                step(vec![
                    assistant_message("Let me check."),
                    function_call("echo", "{\"text\":\"x\"}", "call_1"),
                ]),
                step(vec![assistant_message("Done.")]),
            ],
        ),
        Scenario::new(
            "handoff_with_function",
            echo_and_handoff,
            "Echo and hand off.",
            vec![
                step(vec![
                    function_call("transfer_to_specialist_agent", "{}", "call_h"),
                    function_call("echo", "{\"text\":\"x\"}", "call_e"),
                ]),
                step(vec![assistant_message("Specialist here.")]),
            ],
        ),
        Scenario::new(
            "reasoning_then_answer",
            assistant(vec![]),
            "Think, then answer.",
            vec![step(vec![
                OutputItem::Reasoning {
                    text: "think hard".into(),
                },
                assistant_message("The answer."),
            ])],
        ),
        Scenario::new(
            "two_input_guardrails",
            with_input_guardrails(vec![
                allow_input("first_check"),
                allow_input("second_check"),
            ]),
            "Say ok.",
            vec![step(vec![assistant_message("ok")])],
        ),
        Scenario::new(
            "two_output_guardrails",
            with_output_guardrails(vec![
                allow_output("first_output_check"),
                allow_output("second_output_check"),
            ]),
            "Say ok.",
            vec![step(vec![assistant_message("ok")])],
        ),
        Scenario::new(
            "no_task_turn_spans",
            assistant(vec![echo()]),
            "Echo hi.",
            vec![
                step(vec![function_call("echo", "{\"text\":\"hi\"}", "call_1")]).with_tokens(20, 5),
                step(vec![assistant_message("Echoed: hi")]).with_tokens(30, 4),
            ],
        )
        .without_task_and_turn_spans(),
        Scenario::new(
            "custom_workflow_name",
            assistant(vec![]),
            "Say hello.",
            vec![step(vec![assistant_message("Hello there.")]).with_tokens(11, 3)],
        )
        .workflow("My flow"),
        // An exact repeat of a call in one response is skipped (the second spaced
        // differently); the port and the SDK run and announce each distinct call once.
        Scenario::new(
            "duplicate_call_in_one_response",
            assistant(vec![lookup(), echo()]),
            "Look it up, twice.",
            vec![
                step(vec![
                    function_call("lookup", "{\"query\":\"q\"}", "call_1"),
                    function_call("lookup", "{ \"query\": \"q\" }", "call_1"),
                    function_call("echo", "{\"text\":\"b\"}", "call_2"),
                    function_call("echo", "{\"text\":\"b\"}", "call_2"),
                ]),
                step(vec![assistant_message("Done.")]),
            ],
        ),
        Scenario::new(
            "reused_call_id",
            assistant(vec![echo()]),
            "Echo twice.",
            vec![step(vec![
                function_call("echo", "{\"text\":\"a\"}", "call_1"),
                function_call("echo", "{\"text\":\"b\"}", "call_1"),
            ])],
        ),
        Scenario::new(
            "reused_call_id_across_a_function_and_a_handoff",
            assistant_with_a_handoff(),
            "Echo and hand off.",
            vec![step(vec![
                function_call("echo", "{\"text\":\"a\"}", "call_1"),
                function_call("transfer_to_specialist_agent", "{}", "call_1"),
            ])],
        ),
        Scenario::new(
            "empty_call_id",
            assistant(vec![echo()]),
            "Echo.",
            vec![step(vec![
                function_call("echo", "{\"text\":\"a\"}", "call_1"),
                function_call("echo", "{\"text\":\"b\"}", ""),
            ])],
        ),
        Scenario::new(
            "refusal",
            assistant(vec![echo()]),
            "Do something forbidden.",
            vec![
                step(vec![OutputItem::Refusal {
                    text: "I can't help with that.".into(),
                }])
                .with_tokens(9, 6),
            ],
        ),
        Scenario::new(
            "refusal_with_function_call",
            assistant(vec![echo()]),
            "Echo, if you will.",
            vec![
                step(vec![
                    OutputItem::Refusal {
                        text: "I won't say more.".into(),
                    },
                    function_call("echo", "{\"text\":\"a\"}", "call_1"),
                ]),
                step(vec![assistant_message("Done.")]),
            ],
        ),
        Scenario::new(
            "model_error",
            assistant(vec![]),
            "Say hello.",
            vec![ScriptedStep::error(ModelError::Failed("boom".into()))],
        ),
        // Three steps for two allowed turns: the third must never be consumed.
        Scenario::new(
            "max_turns",
            assistant(vec![echo()]),
            "Loop.",
            vec![
                step(vec![function_call("echo", "{\"text\":\"1\"}", "call_1")]).with_tokens(10, 1),
                step(vec![function_call("echo", "{\"text\":\"2\"}", "call_2")]).with_tokens(10, 1),
                step(vec![function_call("echo", "{\"text\":\"3\"}", "call_3")]).with_tokens(10, 1),
            ],
        )
        .max_turns(2),
        Scenario::new(
            "sensitive_data_off",
            assistant(vec![echo(), explode()]),
            "Two tools, one fails.",
            vec![
                step(vec![
                    function_call("echo", "{\"text\":\"secret\"}", "call_e"),
                    function_call("explode", "{\"reason\":\"secret\"}", "call_x"),
                ]),
                step(vec![assistant_message("Done.")]),
            ],
        )
        .sensitive_data_off(),
        // ---- list input: the items reach the model first, in order, unchanged
        Scenario::from_items(
            "list_input_two_messages",
            assistant(vec![]),
            vec![user("Say hello."), user("In French, please.")],
            vec![step(vec![assistant_message("Bonjour.")]).with_tokens(12, 2)],
        ),
        Scenario::from_items(
            "list_input_with_tool_history",
            assistant(vec![echo()]),
            vec![
                user("Echo hi."),
                InputItem::Assistant {
                    text: None,
                    tool_calls: vec![ToolCallItem {
                        call_id: "call_0".into(),
                        name: "echo".into(),
                        arguments: "{\"text\":\"hi\"}".into(),
                    }],
                },
                InputItem::ToolResult {
                    call_id: "call_0".into(),
                    output: "echo:hi".into(),
                },
                user("Now echo bye."),
            ],
            vec![
                step(vec![function_call("echo", "{\"text\":\"bye\"}", "call_1")])
                    .with_tokens(20, 5),
                step(vec![assistant_message("Echoed: bye")]).with_tokens(30, 4),
            ],
        ),
        // The guardrail trips on the LAST user message only; a port that checked
        // the first would call the model.
        Scenario::from_items(
            "list_input_guardrail_on_last_user",
            with_input_guardrails(vec![refuse_forbidden("last_user_check")]),
            vec![
                user("Say hello."),
                InputItem::Assistant {
                    text: Some("Hello.".into()),
                    tool_calls: vec![],
                },
                user("Now do something forbidden."),
            ],
            vec![],
        ),
        // ---- sessions: what the run reads first, and what it saves when
        Scenario::new(
            "session_one_tool_turn",
            assistant(vec![echo()]),
            "Echo hi.",
            vec![
                step(vec![function_call("echo", "{\"text\":\"hi\"}", "call_1")]).with_tokens(20, 5),
                step(vec![assistant_message("Echoed: hi")]).with_tokens(30, 4),
            ],
        )
        .with_session(),
        Scenario::new(
            "session_two_runs",
            assistant(vec![echo()]),
            "Echo hi.",
            vec![
                step(vec![function_call("echo", "{\"text\":\"hi\"}", "call_1")]),
                step(vec![assistant_message("Echoed: hi")]),
                step(vec![assistant_message("Second answer.")]),
            ],
        )
        .with_session()
        .then("And again?"),
        Scenario::new(
            "session_input_guardrail_trip",
            with_input_guardrails(vec![InputGuardrail::new("refuse_all", |_, _| async {
                GuardrailOutput::trip(json!({"reason": "refused"}))
            })]),
            "Do something forbidden.",
            vec![],
        )
        .with_session(),
        Scenario::new(
            "session_output_guardrail_trip",
            with_output_guardrails(vec![OutputGuardrail::new("no_output", |_, _| async {
                GuardrailOutput::trip(json!({"reason": "refused"}))
            })]),
            "Say something.",
            vec![step(vec![assistant_message("Something.")]).with_tokens(8, 2)],
        )
        .with_session(),
        // ---- tool approval: one run that waits in place, against the SDK's
        // interrupted run and its resume, stitched (see the generator)
        Scenario::new(
            "approval_approve",
            assistant(vec![delete_file()]),
            "Delete a.txt.",
            vec![
                step(vec![function_call(
                    "delete_file",
                    "{\"path\":\"a.txt\"}",
                    "call_d",
                )]),
                step(vec![assistant_message("Deleted.")]),
            ],
        )
        .deciding(vec![ApprovalDecision::Approve]),
        Scenario::new(
            "approval_reject",
            assistant(vec![delete_file()]),
            "Delete a.txt.",
            vec![
                step(vec![function_call(
                    "delete_file",
                    "{\"path\":\"a.txt\"}",
                    "call_d",
                )]),
                step(vec![assistant_message("I did not delete it.")]),
            ],
        )
        .deciding(vec![ApprovalDecision::Reject { note: None }]),
        Scenario::new(
            "approval_reject_with_note",
            assistant(vec![delete_file()]),
            "Delete a.txt.",
            vec![
                step(vec![function_call(
                    "delete_file",
                    "{\"path\":\"a.txt\"}",
                    "call_d",
                )]),
                step(vec![assistant_message("Understood.")]),
            ],
        )
        .deciding(vec![ApprovalDecision::Reject {
            note: Some("not that file".into()),
        }]),
        Scenario::new(
            "approval_predicate_mixed",
            assistant(vec![remove()]),
            "Remove notes.txt and secret.txt.",
            vec![
                step(vec![
                    function_call("remove", "{\"path\":\"notes.txt\"}", "call_n"),
                    function_call("remove", "{\"path\":\"secret.txt\"}", "call_s"),
                ]),
                step(vec![assistant_message("Both removed.")]),
            ],
        )
        .deciding(vec![ApprovalDecision::Approve]),
        // Spec 22.6 A3a: calls that need approval before and between calls
        // that need none; outputs, events and session writes in the SDK's order.
        Scenario::new(
            "approval_needed_before_free_calls",
            assistant(vec![delete_file(), remove(), echo()]),
            "Delete a.txt, remove notes.txt and secret.txt, and echo hi.",
            vec![
                step(vec![
                    function_call("delete_file", "{\"path\":\"a.txt\"}", "call_d"),
                    function_call("remove", "{\"path\":\"notes.txt\"}", "call_n"),
                    function_call("remove", "{\"path\":\"secret.txt\"}", "call_s"),
                    function_call("echo", "{\"text\":\"hi\"}", "call_e"),
                ]),
                step(vec![assistant_message("Done, except secret.txt.")]),
            ],
        )
        .deciding(vec![
            ApprovalDecision::Approve,
            ApprovalDecision::Reject { note: None },
        ])
        .with_session()
        .with_tool_events(),
        Scenario::new(
            "approval_reject_then_answer",
            assistant(vec![echo(), delete_file()]),
            "Echo hi, then delete a.txt.",
            vec![
                step(vec![function_call("echo", "{\"text\":\"hi\"}", "call_e")]).with_tokens(20, 5),
                step(vec![function_call(
                    "delete_file",
                    "{\"path\":\"a.txt\"}",
                    "call_d",
                )])
                .with_tokens(30, 6),
                step(vec![assistant_message("Echoed, and a.txt was kept.")]).with_tokens(40, 7),
            ],
        )
        .deciding(vec![ApprovalDecision::Reject { note: None }]),
    ]
}

fn user(text: &str) -> InputItem {
    InputItem::User(text.to_owned())
}

/// Trips when the text it is given mentions "forbidden" (the generator's
/// `_refuse_forbidden_last_user`).
fn refuse_forbidden(name: &'static str) -> InputGuardrail {
    InputGuardrail::new(name, |_, text| async move {
        if text.contains("forbidden") {
            GuardrailOutput::trip(json!({"checked": true}))
        } else {
            GuardrailOutput {
                tripwire_triggered: false,
                output_info: json!({"checked": true}),
            }
        }
    })
}

/// One model call's input in the goldens' reduced form (see the generator's
/// header): user and assistant text, function calls and their outputs.
fn reduced_input(items: &[InputItem]) -> Value {
    let mut reduced = Vec::new();
    for item in items {
        match item {
            // The goldens hold no images (the SDK's runs here send none).
            InputItem::User(text) | InputItem::UserImages { text, .. } => {
                reduced.push(json!({"role": "user", "text": text}))
            }
            InputItem::Assistant { text, tool_calls } => {
                if let Some(text) = text {
                    reduced.push(json!({"role": "assistant", "text": text}));
                }
                for call in tool_calls {
                    reduced.push(json!({
                        "type": "function_call",
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": call.arguments,
                    }));
                }
            }
            InputItem::ToolResult { call_id, output } => reduced.push(json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            })),
        }
    }
    Value::Array(reduced)
}

/// Run a scenario and describe it the way the goldens do.
async fn describe(scenario: Scenario) -> Value {
    let model = scripted(scenario.steps.clone());
    let session = scenario.session.then(MemorySession::new);
    let tweak = |config: &mut RunConfig| {
        if let Some(max_turns) = scenario.max_turns {
            config.max_turns = max_turns;
        }
        if let Some(workflow_name) = scenario.workflow_name {
            config.workflow_name = workflow_name.to_owned();
        }
        config.include_sensitive_data = scenario.include_sensitive_data;
        config.include_task_and_turn_spans = scenario.include_task_and_turn_spans;
    };
    let mut starts = vec![match &scenario.input {
        ScenarioInput::Text(text) => Start::Text((*text).to_owned()),
        ScenarioInput::Items(items) => Start::Items(items.clone()),
    }];
    starts.extend(
        scenario
            .then
            .iter()
            .map(|text| Start::Text((*text).to_owned())),
    );
    let collector = Collector::new();
    let mut names: Vec<&'static str> = Vec::new();
    let mut tool_events: Vec<Value> = Vec::new();
    let mut last: Option<Outcome> = None;
    for start in starts {
        let outcome = run_into(
            scenario.agent.clone(),
            start,
            model.clone(),
            collector.clone(),
            |config: &mut RunConfig| {
                tweak(config);
                config.session = session.clone().map(|session| session as Arc<dyn Session>);
                config.approvals = scenario
                    .approvals
                    .clone()
                    .map(|decisions| ScriptedApprovals::new(decisions) as Arc<dyn ApprovalPort>);
            },
        )
        .await;
        names.extend(outcome.names.iter().copied());
        for event in &outcome.events {
            if let StreamEvent::RunItem {
                item: RunItem::ToolCall { call_id, .. },
                ..
            } = event
            {
                tool_events.push(json!(["tool_called", call_id]));
            }
            if let StreamEvent::RunItem {
                item: RunItem::ToolOutput { call_id, .. },
                ..
            } = event
            {
                tool_events.push(json!(["tool_output", call_id]));
            }
        }
        let failed = outcome.result.is_err();
        last = Some(outcome);
        if failed {
            break;
        }
    }
    let outcome = last.expect("a scenario makes at least one run");

    let mut golden = serde_json::Map::new();
    match &outcome.result {
        Ok(result) => {
            golden.insert("final_output".into(), json!(result.final_output));
        }
        Err(error) => {
            golden.insert("error".into(), json!(error.sdk_name()));
        }
    }
    golden.insert("stream".into(), json!(names));
    golden.insert("spans".into(), Value::Array(collector.forest()));
    if scenario.record_model_inputs {
        let calls: Vec<Value> = model
            .calls()
            .iter()
            .map(|call| reduced_input(&call.input))
            .collect();
        golden.insert("model_inputs".into(), Value::Array(calls));
    }
    if let Some(session) = &session {
        let adds: Vec<Value> = session
            .adds()
            .iter()
            .map(|items| reduced_input(items))
            .collect();
        golden.insert("session_add_items".into(), Value::Array(adds));
    }
    if scenario.record_tool_events {
        golden.insert("tool_events".into(), Value::Array(tool_events));
    }
    Value::Object(golden)
}

fn read_golden(name: &str) -> Value {
    let path = format!("{}/tests/goldens/{name}.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read golden {path}: {error}"));
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("golden {path} is not JSON: {error}"))
}

fn remove_pointer(value: &mut Value, pointer: &str) {
    let Some((parent, key)) = pointer.rsplit_once('/') else {
        return;
    };
    if let Some(target) = value.pointer_mut(parent) {
        match target {
            Value::Object(map) => {
                map.remove(key);
            }
            Value::Array(items) => {
                if let Ok(index) = key.parse::<usize>()
                    && index < items.len()
                {
                    items.remove(index);
                }
            }
            _ => {}
        }
    }
}

async fn conforms(name: &str) {
    let scenario = scenarios()
        .into_iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("no scenario {name}"));
    let mut actual = describe(scenario).await;
    let mut golden = read_golden(name);
    for (scenario, pointer, _reason) in DEVIATIONS {
        if *scenario == name {
            assert!(
                actual.pointer(pointer).is_some(),
                "deviation {pointer} of {name} points at nothing in the port's output"
            );
            assert!(
                golden.pointer(pointer).is_some(),
                "deviation {pointer} of {name} points at nothing in the golden"
            );
            remove_pointer(&mut actual, pointer);
            remove_pointer(&mut golden, pointer);
        }
    }
    assert!(
        actual == golden,
        "scenario {name} differs from the SDK's recorded behaviour\n--- port ---\n{}\n--- SDK golden ---\n{}",
        serde_json::to_string_pretty(&actual).unwrap(),
        serde_json::to_string_pretty(&golden).unwrap()
    );
}

macro_rules! scenario_tests {
    ($($name:ident),* $(,)?) => {
        $(
            #[tokio::test]
            async fn $name() {
                conforms(stringify!($name)).await;
            }
        )*
    };
}

scenario_tests!(
    text_only,
    one_tool,
    two_parallel_tools,
    handoff_then_tool,
    multiple_handoffs,
    input_guardrail_pass,
    input_guardrail_trip,
    output_guardrail_trip,
    tool_error_default_policy,
    tool_error_custom_policy,
    tool_invalid_json,
    tool_non_object_arguments,
    tool_not_found,
    text_with_tool_call,
    handoff_with_function,
    reasoning_then_answer,
    two_input_guardrails,
    two_output_guardrails,
    no_task_turn_spans,
    custom_workflow_name,
    duplicate_call_in_one_response,
    reused_call_id,
    reused_call_id_across_a_function_and_a_handoff,
    empty_call_id,
    refusal,
    refusal_with_function_call,
    model_error,
    max_turns,
    sensitive_data_off,
    list_input_two_messages,
    list_input_with_tool_history,
    list_input_guardrail_on_last_user,
    session_one_tool_turn,
    session_two_runs,
    session_input_guardrail_trip,
    session_output_guardrail_trip,
    approval_approve,
    approval_reject,
    approval_reject_with_note,
    approval_predicate_mixed,
    approval_needed_before_free_calls,
    approval_reject_then_answer,
);

/// A golden nobody replays proves nothing, and a scenario without a golden
/// would compare against a file that is not there.
#[test]
fn every_golden_has_a_scenario_and_every_scenario_a_golden() {
    let mut on_disk: Vec<String> =
        std::fs::read_dir(format!("{}/tests/goldens", env!("CARGO_MANIFEST_DIR")))
            .unwrap()
            .filter_map(|entry| {
                let name = entry.unwrap().file_name().into_string().unwrap();
                name.strip_suffix(".json").map(str::to_owned)
            })
            .collect();
    on_disk.sort();
    let mut declared: Vec<String> = scenarios().into_iter().map(|s| s.name.to_owned()).collect();
    declared.sort();
    assert_eq!(on_disk, declared);
}

/// Every scenario has a test function, so none is silently left unreplayed.
#[test]
fn every_scenario_is_replayed_by_a_test() {
    let source = std::fs::read_to_string(format!(
        "{}/tests/conformance.rs",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let macro_body = source
        .split("scenario_tests!(")
        .nth(1)
        .expect("the macro invocation");
    let replayed: Vec<&str> = macro_body
        .split(");")
        .next()
        .unwrap()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    for scenario in scenarios() {
        assert!(
            replayed.contains(&scenario.name),
            "scenario {} has no test",
            scenario.name
        );
    }
}

#[test]
fn deviations_name_a_reason_and_a_recorded_scenario() {
    let names: Vec<&str> = scenarios().iter().map(|s| s.name).collect();
    for (scenario, pointer, reason) in DEVIATIONS {
        assert!(names.contains(scenario), "unknown scenario {scenario}");
        assert!(pointer.starts_with('/') && !reason.trim().is_empty());
    }
}

/// The custom-policy tools reuse the plain tools' handlers under other names,
/// which must be exactly what the scripted model calls.
#[test]
fn the_custom_policy_tools_carry_the_names_the_goldens_call() {
    assert_eq!(explode_custom().name, "explode_custom");
    assert_eq!(echo_custom().name, "echo_custom");
}
