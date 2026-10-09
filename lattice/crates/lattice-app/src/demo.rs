//! An in-memory [`RunService`] with scripted runs: the demonstration data behind
//! `--demo`, and the fake the interface's tests run against.
//!
//! What it offers: two agents (the built-in pair, Lattice assistant handing off
//! to Model advisor), three models (a local one that is ready, a remote one that
//! is ready, one that is not ready and says why), four finished runs
//! (completed, completed with a tool and a hand-off, failed, refused), and
//! `start()`, which replays a realistic event sequence over about four seconds
//! on a background thread: trace start; task, agent, turn, generation, function,
//! hand-off and guardrail spans with plausible timings; streamed text in bursts;
//! a result; the end.
//!
//! Invariants (the tests hold the service to the protocol's own):
//! - Event `seq` numbers start at 1, rise by one per run and are assigned when an
//!   event is published, so a follower that has seen `n` and asks for events after
//!   `n` misses nothing.
//! - Parents start before their children and a span's `order` is assigned at its
//!   start; a `SpanEnd` carries the whole span, which supersedes its `SpanStart`.
//! - `End` is always the last event; the stream a follower gets ends after the
//!   batch that holds it, and ends at once for a run that has already ended and
//!   has nothing after `after`.
//! - Batches are whatever accumulated since the follower last polled, so a burst
//!   of streamed text costs one wake-up.
//! - No thread outlives its run, and `stop()` ends a run promptly with every open
//!   span closed.
//! - Nothing here touches the network, the disk or a model: the "model" is a script.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use futures::stream::{BoxStream, Stream};
use lattice_protocol::{
    AgentGraph, AgentInfo, EdgeKind, EndStatus, GraphEdge, GraphNode, Locality, ModelChoice,
    ModelKind, NodeKind, Refusal, RefusalKind, RunDetail, RunEvent, RunEventKind, RunService,
    RunStatus, RunSummary, ServiceStatus, SpanError, SpanRecord, StartRun, TraceInfo, Usage,
};
use serde_json::{Value, json};

use crate::clock::{self, iso_from_epoch};

/// A source of "now", in seconds since the epoch.
pub type Clock = Arc<dyn Fn() -> f64 + Send + Sync>;

/// How many demonstration runs may work at once.
pub const MAX_RUNNING: usize = 3;

pub const ASSISTANT: &str = "lattice-assistant";
pub const ADVISOR: &str = "model-advisor";
pub const LOCAL_MODEL: &str = "local";
pub const REMOTE_MODEL: &str = "endpoint:hosted-demo";
pub const OFFLINE_MODEL: &str = "endpoint:workstation";

const ASSISTANT_LABEL: &str = "Lattice assistant";
const ADVISOR_LABEL: &str = "Model advisor";
const GUARDRAIL: &str = "secrets_stay_local";
const OFFLINE_REFUSAL: &str =
    "The workstation endpoint is not running. Start it, then open this list again.";

// --------------------------------------------------------------- catalogue

fn node(kind: NodeKind, label: &str, root: bool) -> GraphNode {
    let prefix = match kind {
        NodeKind::Start => "start",
        NodeKind::End => "end",
        NodeKind::Agent => "agent",
        NodeKind::Tool => "tool",
        NodeKind::Mcp => "mcp",
    };
    GraphNode {
        id: format!("{prefix}:{label}"),
        kind,
        label: label.to_string(),
        root,
    }
}

fn edge(source: &GraphNode, target: &GraphNode, kind: EdgeKind) -> GraphEdge {
    GraphEdge {
        source: source.id.clone(),
        target: target.id.clone(),
        kind,
    }
}

fn assistant_graph() -> AgentGraph {
    let start = node(NodeKind::Start, "__start__", false);
    let assistant = node(NodeKind::Agent, ASSISTANT_LABEL, true);
    let time = node(NodeKind::Tool, "current_time", false);
    let calc = node(NodeKind::Tool, "calculate", false);
    let advisor = node(NodeKind::Agent, ADVISOR_LABEL, false);
    let models = node(NodeKind::Tool, "list_models", false);
    let end = node(NodeKind::End, "__end__", false);
    AgentGraph {
        edges: vec![
            edge(&start, &assistant, EdgeKind::Start),
            edge(&assistant, &time, EdgeKind::Tool),
            edge(&assistant, &calc, EdgeKind::Tool),
            edge(&assistant, &advisor, EdgeKind::Handoff),
            edge(&advisor, &models, EdgeKind::Tool),
            edge(&advisor, &end, EdgeKind::End),
        ],
        nodes: vec![start, assistant, time, calc, advisor, models, end],
    }
}

fn advisor_graph() -> AgentGraph {
    let start = node(NodeKind::Start, "__start__", false);
    let advisor = node(NodeKind::Agent, ADVISOR_LABEL, true);
    let models = node(NodeKind::Tool, "list_models", false);
    let end = node(NodeKind::End, "__end__", false);
    AgentGraph {
        edges: vec![
            edge(&start, &advisor, EdgeKind::Start),
            edge(&advisor, &models, EdgeKind::Tool),
            edge(&advisor, &end, EdgeKind::End),
        ],
        nodes: vec![start, advisor, models, end],
    }
}

fn catalogue_agents() -> Vec<AgentInfo> {
    vec![
        AgentInfo {
            id: ASSISTANT.into(),
            label: ASSISTANT_LABEL.into(),
            description: "Answers questions, tells the time, does arithmetic, and hands model questions to the Model advisor.".into(),
            graph: assistant_graph(),
        },
        AgentInfo { id: ADVISOR.into(), label: ADVISOR_LABEL.into(), description: "Explains which model suits a task, from the models you have set up.".into(), graph: advisor_graph() },
    ]
}

fn catalogue_models() -> Vec<ModelChoice> {
    vec![
        ModelChoice {
            id: LOCAL_MODEL.into(),
            label: "Local model".into(),
            kind: ModelKind::Ollama,
            locality: Locality::Local,
            ready: true,
            refusal: None,
        },
        ModelChoice {
            id: REMOTE_MODEL.into(),
            label: "Hosted model".into(),
            kind: ModelKind::OpenaiCompatible,
            locality: Locality::Remote,
            ready: true,
            refusal: None,
        },
        ModelChoice {
            id: OFFLINE_MODEL.into(),
            label: "Workstation model".into(),
            kind: ModelKind::OpenaiCompatible,
            locality: Locality::Local,
            ready: false,
            refusal: Some(OFFLINE_REFUSAL.into()),
        },
    ]
}

/// The model name a generation span records for a model choice.
fn served_model(model_id: &str) -> &'static str {
    match model_id {
        REMOTE_MODEL => "gpt-4.1-mini",
        OFFLINE_MODEL => "workstation-70b",
        _ => "llama3.2:3b",
    }
}

// ------------------------------------------------------------------ script

/// What a scripted run ends as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Completed,
    /// The model server stops answering.
    Failed,
    /// The input guardrail refuses the task before any model sees it.
    Refused,
}

/// A span handle while a script is being built.
#[derive(Clone)]
struct Open {
    id: String,
    order: u64,
    parent: Option<String>,
    started_at: String,
}

struct Builder {
    base: f64,
    trace_id: String,
    seed: u64,
    counter: u64,
    order: u64,
    events: Vec<(f64, RunEventKind)>,
}

fn hex_of(seed: u64, counter: u64, width: usize) -> String {
    // A cheap avalanche (splitmix64) so ids look like the SDK's random hex.
    let mut out = String::new();
    let mut state = seed ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    while out.len() < width {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        out.push_str(&format!("{:016x}", z ^ (z >> 31)));
    }
    out.truncate(width);
    out
}

impl Builder {
    fn new(base: f64, seed: u64) -> Self {
        Self {
            base,
            trace_id: format!("trace_{}", hex_of(seed, 0, 32)),
            seed,
            counter: 0,
            order: 0,
            events: Vec::new(),
        }
    }

    fn event(&mut self, t: f64, kind: RunEventKind) {
        self.events.push((t, kind));
    }

    fn start(&mut self, t: f64, parent: Option<&Open>, data: Value) -> Open {
        self.counter += 1;
        self.order += 1;
        let open = Open {
            id: format!("span_{}", hex_of(self.seed, self.counter, 24)),
            order: self.order,
            parent: parent.map(|p| p.id.clone()),
            started_at: iso_from_epoch(self.base + t),
        };
        let span = self.record(&open, None, data, None);
        self.event(t, RunEventKind::SpanStart { span });
        open
    }

    fn end(&mut self, t: f64, open: &Open, data: Value, error: Option<SpanError>) {
        let span = self.record(open, Some(iso_from_epoch(self.base + t)), data, error);
        self.event(t, RunEventKind::SpanEnd { span });
    }

    fn record(
        &self,
        open: &Open,
        ended_at: Option<String>,
        data: Value,
        error: Option<SpanError>,
    ) -> SpanRecord {
        SpanRecord {
            order: open.order,
            id: open.id.clone(),
            trace_id: self.trace_id.clone(),
            parent_id: open.parent.clone(),
            started_at: open.started_at.clone(),
            ended_at,
            span_data: data,
            error,
        }
    }

    /// Streamed text as bursts: a few small deltas published together, a short
    /// pause, the next burst. Returns the time the last burst was published.
    fn stream_text(&mut self, from: f64, agent: &str, text: &str) -> f64 {
        let words: Vec<&str> = text.split_inclusive(char::is_whitespace).collect();
        let mut t = from;
        for burst in words.chunks(9) {
            for group in burst.chunks(3) {
                self.event(
                    t,
                    RunEventKind::Delta {
                        agent: agent.to_string(),
                        text: group.concat(),
                    },
                );
            }
            t += 0.11;
        }
        t
    }
}

fn agent_data(name: &str, finished: bool) -> Value {
    if !finished {
        return json!({"type": "agent", "name": name, "handoffs": null, "tools": null, "output_type": null});
    }
    let (handoffs, tools): (Vec<&str>, Vec<&str>) = if name == ASSISTANT_LABEL {
        (vec![ADVISOR_LABEL], vec!["current_time", "calculate"])
    } else {
        (vec![], vec!["list_models"])
    };
    json!({"type": "agent", "name": name, "handoffs": handoffs, "tools": tools, "output_type": "str"})
}

fn task_data(usage: Option<&Usage>) -> Value {
    let mut data = json!({"sdk_span_type": "task", "name": "Agent workflow"});
    if let Some(u) = usage {
        data["usage"] = json!({"requests": u.requests, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens, "total_tokens": u.total_tokens});
    }
    json!({"type": "custom", "name": "task", "data": data})
}

fn turn_data(turn: u32, agent: &str, usage: Option<(u64, u64)>) -> Value {
    let mut data = json!({"sdk_span_type": "turn", "turn": turn, "agent_name": agent});
    if let Some((i, o)) = usage {
        data["usage"] =
            json!({"requests": 1, "input_tokens": i, "output_tokens": o, "total_tokens": i + o});
    }
    json!({"type": "custom", "name": "turn", "data": data})
}

fn generation_data(
    model: &str,
    input: &Value,
    output: Option<Value>,
    usage: Option<(u64, u64)>,
) -> Value {
    json!({
        "type": "generation",
        "input": input,
        "output": output,
        "model": model,
        "model_config": {"temperature": 0.2, "top_p": null, "max_tokens": null, "stream": true},
        "usage": usage.map(|(i, o)| json!({"input_tokens": i, "output_tokens": o})),
    })
}

const ASSISTANT_SYSTEM: &str = "You are Lattice's assistant. Answer briefly. Use the tools when they help. Hand questions about which model to use to the Model advisor.";
const ADVISOR_SYSTEM: &str = "You are Lattice's model advisor. Explain which of the user's configured models suits a task, and say plainly which ones leave this machine.";

fn tool_call(id: &str, name: &str, arguments: &str) -> Value {
    json!({"id": id, "type": "function", "function": {"name": name, "arguments": arguments}})
}

fn final_answer(task: &str) -> String {
    format!(
        "**Short answer:** for \"{}\", use the *Local model* when the text is private and the *Hosted model* when it is longer than the local model's context window.\n\n\
- **Local model** runs on this machine. Nothing leaves it. It is comfortable up to about 8,000 tokens.\n\
- **Hosted model** sends the text off this machine, and handles much longer inputs.\n\n\
I checked the models you have set up. The Workstation model is not running right now, so it is not an option until you start it.",
        clip(task, 80)
    )
}

fn clip(text: &str, max_chars: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max_chars {
        flat
    } else {
        format!(
            "{}…",
            flat.chars().take(max_chars).collect::<String>().trim_end()
        )
    }
}

/// The whole scripted run: `(offset in seconds, event)`, in order.
pub fn script(
    base: f64,
    seed: u64,
    task: &str,
    agent: &str,
    model_id: &str,
    outcome: Outcome,
) -> Vec<(f64, RunEventKind)> {
    let mut b = Builder::new(base, seed);
    let model = served_model(model_id);
    let starts_with_assistant = agent == ASSISTANT;
    let first_agent = if starts_with_assistant {
        ASSISTANT_LABEL
    } else {
        ADVISOR_LABEL
    };
    let system = if starts_with_assistant {
        ASSISTANT_SYSTEM
    } else {
        ADVISOR_SYSTEM
    };
    let user = json!({"role": "user", "content": task});
    let system_msg = json!({"role": "system", "content": system});

    let trace_id = b.trace_id.clone();
    b.event(
        0.0,
        RunEventKind::TraceStart {
            trace_id: trace_id.clone(),
            workflow_name: "Agent workflow".into(),
            at: iso_from_epoch(base),
        },
    );
    let task_span = b.start(0.0, None, task_data(None));
    let agent_span = b.start(0.012, Some(&task_span), agent_data(first_agent, false));
    b.event(
        0.012,
        RunEventKind::Agent {
            name: first_agent.into(),
        },
    );

    // The input guardrail runs on the first agent, before its first model call.
    let guard = b.start(
        0.020,
        Some(&agent_span),
        json!({"type": "guardrail", "name": GUARDRAIL, "triggered": null}),
    );
    if outcome == Outcome::Refused {
        b.end(
            0.046,
            &guard,
            json!({"type": "guardrail", "name": GUARDRAIL, "triggered": true}),
            Some(SpanError {
                message: "Guardrail tripwire triggered".into(),
                data: Some(json!({"guardrail": GUARDRAIL})),
            }),
        );
        b.end(
            0.050,
            &agent_span,
            agent_data(first_agent, true),
            Some(SpanError {
                message: "Guardrail tripwire triggered".into(),
                data: Some(json!({"guardrail": GUARDRAIL})),
            }),
        );
        b.event(0.052, RunEventKind::Guardrail { name: GUARDRAIL.into(), message: "The task looks like it contains a secret, so it was not sent to any model. Remove the secret and start the run again.".into() });
        b.end(0.055, &task_span, task_data(None), None);
        b.event(
            0.056,
            RunEventKind::TraceEnd {
                trace_id,
                at: iso_from_epoch(base + 0.056),
            },
        );
        b.event(
            0.057,
            RunEventKind::End {
                status: EndStatus::Refused,
            },
        );
        return b.events;
    }
    b.end(
        0.046,
        &guard,
        json!({"type": "guardrail", "name": GUARDRAIL, "triggered": false}),
        None,
    );

    let mut usage_in = 0u64;
    let mut usage_out = 0u64;
    let mut requests = 0u64;
    let mut t;

    // Turn 1: the model reasons aloud and asks for the time.
    let turn1 = b.start(0.052, Some(&agent_span), turn_data(1, first_agent, None));
    let input1 = json!([system_msg, user]);
    let gen1 = b.start(
        0.060,
        Some(&turn1),
        generation_data(model, &input1, None, None),
    );
    if outcome == Outcome::Failed {
        b.end(
            0.94,
            &gen1,
            generation_data(model, &input1, None, None),
            Some(SpanError {
                message: "The model server did not answer.".into(),
                data: Some(json!({"status": 503, "attempts": 1})),
            }),
        );
        b.end(0.95, &turn1, turn_data(1, first_agent, None), None);
        b.end(
            0.96,
            &agent_span,
            agent_data(first_agent, true),
            Some(SpanError {
                message: "Model request failed".into(),
                data: None,
            }),
        );
        b.event(0.97, RunEventKind::Error { message: "The model server did not answer (HTTP 503). Nothing was lost; start the run again when it is back.".into() });
        b.end(0.98, &task_span, task_data(None), None);
        b.event(
            0.99,
            RunEventKind::TraceEnd {
                trace_id,
                at: iso_from_epoch(base + 0.99),
            },
        );
        b.event(
            1.0,
            RunEventKind::End {
                status: EndStatus::Failed,
            },
        );
        return b.events;
    }
    let lead_in = if starts_with_assistant {
        "I'll look up the current time first, then decide who should answer."
    } else {
        "Let me check which models you have set up."
    };
    t = b.stream_text(0.30, first_agent, lead_in);
    let call_name = if starts_with_assistant {
        "current_time"
    } else {
        "list_models"
    };
    let call_id = "call_1";
    let out1 = json!([{"role": "assistant", "content": lead_in, "tool_calls": [tool_call(call_id, call_name, "{}")]}]);
    let (i1, o1) = (412u64, 38u64);
    usage_in += i1;
    usage_out += o1;
    requests += 1;
    t = t.max(1.05);
    b.end(
        t,
        &gen1,
        generation_data(model, &input1, Some(out1.clone()), Some((i1, o1))),
        None,
    );
    b.event(
        t + 0.002,
        RunEventKind::Message {
            agent: first_agent.into(),
            text: lead_in.into(),
        },
    );
    b.event(
        t + 0.004,
        RunEventKind::ToolCall {
            agent: first_agent.into(),
            name: call_name.into(),
            call_id: call_id.into(),
            arguments: "{}".into(),
        },
    );
    let tool_output = if starts_with_assistant {
        "Wednesday 2026-09-30, 05:21:05 UTC".to_string()
    } else {
        "Local model (ready, on this machine); Hosted model (ready, off this machine); Workstation model (not running)".to_string()
    };
    let func = b.start(t + 0.006, Some(&turn1), json!({"type": "function", "name": call_name, "input": "{}", "output": null, "mcp_data": null}));
    let t_func_end = t + 0.006 + 0.31;
    b.end(t_func_end, &func, json!({"type": "function", "name": call_name, "input": "{}", "output": tool_output, "mcp_data": null}), None);
    b.event(
        t_func_end + 0.002,
        RunEventKind::ToolOutput {
            agent: first_agent.into(),
            call_id: call_id.into(),
            output: tool_output.clone(),
        },
    );
    b.end(
        t_func_end + 0.004,
        &turn1,
        turn_data(1, first_agent, Some((i1, o1))),
        None,
    );
    t = t_func_end + 0.006;

    let mut history = vec![
        system_msg.clone(),
        user.clone(),
        out1[0].clone(),
        json!({"role": "tool", "tool_call_id": call_id, "content": tool_output}),
    ];

    // Turn 2 (assistant only): the model recognises a model question and hands off.
    let mut turn_no = 1u32;
    let mut current = first_agent;
    let mut current_span = agent_span;
    if starts_with_assistant {
        turn_no += 1;
        let turn2 = b.start(
            t,
            Some(&current_span),
            turn_data(turn_no, ASSISTANT_LABEL, None),
        );
        let gen2 = b.start(
            t + 0.008,
            Some(&turn2),
            generation_data(model, &json!(history.clone()), None, None),
        );
        let hand_text = "This is a question about models, so I am passing it to the Model advisor.";
        let t_text = b.stream_text(t + 0.30, ASSISTANT_LABEL, hand_text);
        let t_gen_end = t_text.max(t + 0.75);
        let out2 = json!([{"role": "assistant", "content": hand_text, "tool_calls": [tool_call("call_2", "transfer_to_model_advisor", "{}")]}]);
        let (i2, o2) = (466u64, 27u64);
        usage_in += i2;
        usage_out += o2;
        requests += 1;
        b.end(
            t_gen_end,
            &gen2,
            generation_data(
                model,
                &json!(history.clone()),
                Some(out2.clone()),
                Some((i2, o2)),
            ),
            None,
        );
        b.event(
            t_gen_end + 0.002,
            RunEventKind::Message {
                agent: ASSISTANT_LABEL.into(),
                text: hand_text.into(),
            },
        );
        b.event(
            t_gen_end + 0.004,
            RunEventKind::HandoffRequested {
                agent: ASSISTANT_LABEL.into(),
            },
        );
        let handoff = b.start(
            t_gen_end + 0.006,
            Some(&turn2),
            json!({"type": "handoff", "from_agent": ASSISTANT_LABEL, "to_agent": null}),
        );
        b.end(
            t_gen_end + 0.011,
            &handoff,
            json!({"type": "handoff", "from_agent": ASSISTANT_LABEL, "to_agent": ADVISOR_LABEL}),
            None,
        );
        b.event(
            t_gen_end + 0.012,
            RunEventKind::Handoff {
                from: ASSISTANT_LABEL.into(),
                to: ADVISOR_LABEL.into(),
            },
        );
        b.end(
            t_gen_end + 0.014,
            &turn2,
            turn_data(turn_no, ASSISTANT_LABEL, Some((i2, o2))),
            None,
        );
        b.end(
            t_gen_end + 0.016,
            &current_span,
            agent_data(ASSISTANT_LABEL, true),
            None,
        );
        t = t_gen_end + 0.02;

        // The advisor takes over: its own agent span, one tool call, the answer.
        current = ADVISOR_LABEL;
        current_span = b.start(t, Some(&task_span), agent_data(ADVISOR_LABEL, false));
        b.event(
            t,
            RunEventKind::Agent {
                name: ADVISOR_LABEL.into(),
            },
        );
        turn_no += 1;
        let turn3 = b.start(
            t + 0.01,
            Some(&current_span),
            turn_data(turn_no, ADVISOR_LABEL, None),
        );
        // The advisor sees the conversation so far under its own instructions.
        history[0] = json!({"role": "system", "content": ADVISOR_SYSTEM});
        history.push(out2[0].clone());
        history.push(json!({"role": "tool", "tool_call_id": "call_2", "content": "{\"assistant\": \"Model advisor\"}"}));
        let gen3 = b.start(
            t + 0.02,
            Some(&turn3),
            generation_data(model, &json!(history.clone()), None, None),
        );
        let t3_text = b.stream_text(
            t + 0.20,
            ADVISOR_LABEL,
            "Let me check which models you have set up.",
        );
        let t3_end = t3_text.max(t + 0.50);
        let out3 = json!([{"role": "assistant", "content": "Let me check which models you have set up.", "tool_calls": [tool_call("call_3", "list_models", "{}")]}]);
        let (i3, o3) = (503u64, 31u64);
        usage_in += i3;
        usage_out += o3;
        requests += 1;
        b.end(
            t3_end,
            &gen3,
            generation_data(
                model,
                &json!(history.clone()),
                Some(out3.clone()),
                Some((i3, o3)),
            ),
            None,
        );
        b.event(
            t3_end + 0.002,
            RunEventKind::Message {
                agent: ADVISOR_LABEL.into(),
                text: "Let me check which models you have set up.".into(),
            },
        );
        b.event(
            t3_end + 0.004,
            RunEventKind::ToolCall {
                agent: ADVISOR_LABEL.into(),
                name: "list_models".into(),
                call_id: "call_3".into(),
                arguments: "{}".into(),
            },
        );
        let models_out = "Local model (ready, on this machine); Hosted model (ready, off this machine); Workstation model (not running)";
        let f3 = b.start(t3_end + 0.006, Some(&turn3), json!({"type": "function", "name": "list_models", "input": "{}", "output": null, "mcp_data": null}));
        b.end(t3_end + 0.146, &f3, json!({"type": "function", "name": "list_models", "input": "{}", "output": models_out, "mcp_data": null}), None);
        b.event(
            t3_end + 0.148,
            RunEventKind::ToolOutput {
                agent: ADVISOR_LABEL.into(),
                call_id: "call_3".into(),
                output: models_out.into(),
            },
        );
        b.end(
            t3_end + 0.15,
            &turn3,
            turn_data(turn_no, ADVISOR_LABEL, Some((i3, o3))),
            None,
        );
        history.push(out3[0].clone());
        history.push(json!({"role": "tool", "tool_call_id": "call_3", "content": models_out}));
        t = t3_end + 0.16;
    }

    // The final turn: the answer, streamed.
    turn_no += 1;
    let answer = final_answer(task);
    let last = b.start(t, Some(&current_span), turn_data(turn_no, current, None));
    let gen_last = b.start(
        t + 0.01,
        Some(&last),
        generation_data(model, &json!(history.clone()), None, None),
    );
    let t_answer = b.stream_text(t + 0.25, current, &answer);
    let t_last_end = t_answer.max(t + 0.9);
    let (il, ol) = (611u64, 142u64);
    usage_in += il;
    usage_out += ol;
    requests += 1;
    let out_last = json!([{"role": "assistant", "content": answer.clone()}]);
    b.end(
        t_last_end,
        &gen_last,
        generation_data(model, &json!(history), Some(out_last), Some((il, ol))),
        None,
    );
    b.event(
        t_last_end + 0.002,
        RunEventKind::Message {
            agent: current.into(),
            text: answer.clone(),
        },
    );
    b.end(
        t_last_end + 0.004,
        &last,
        turn_data(turn_no, current, Some((il, ol))),
        None,
    );
    b.end(
        t_last_end + 0.006,
        &current_span,
        agent_data(current, true),
        None,
    );
    let usage = Usage {
        requests,
        input_tokens: usage_in,
        output_tokens: usage_out,
        total_tokens: usage_in + usage_out,
    };
    b.end(
        t_last_end + 0.008,
        &task_span,
        task_data(Some(&usage)),
        None,
    );
    b.event(
        t_last_end + 0.010,
        RunEventKind::Result {
            output: answer,
            usage: Some(usage),
            turns: turn_no,
            last_agent: current.into(),
        },
    );
    b.event(
        t_last_end + 0.012,
        RunEventKind::TraceEnd {
            trace_id,
            at: iso_from_epoch(base + t_last_end + 0.012),
        },
    );
    b.event(
        t_last_end + 0.014,
        RunEventKind::End {
            status: EndStatus::Completed,
        },
    );
    b.events
}

// ----------------------------------------------------------------- service

struct RunLog {
    summary: RunSummary,
    /// Every published event, deltas included (followers need them).
    events: Vec<RunEvent>,
    wakers: Vec<Waker>,
    ended: bool,
    stop_requested: bool,
}

struct RunShared {
    log: Mutex<RunLog>,
}

impl RunShared {
    fn lock(&self) -> MutexGuard<'_, RunLog> {
        self.log
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

struct Inner {
    clock: Clock,
    /// Skip the real-time pauses (tests): timestamps still follow the script.
    instant: bool,
    agents: Vec<AgentInfo>,
    models: Vec<ModelChoice>,
    /// Newest first.
    runs: Mutex<Vec<Arc<RunShared>>>,
    next_run: Mutex<u64>,
}

/// The demonstration service. Cloning shares the same runs.
#[derive(Clone)]
pub struct DemoService {
    inner: Arc<Inner>,
}

fn summary_after(summary: &mut RunSummary, event: &RunEvent) {
    summary.updated_at = event.at;
    match &event.kind {
        RunEventKind::SpanEnd { .. } => summary.spans += 1,
        RunEventKind::Result { output, usage, .. } => {
            summary.output = Some(output.chars().take(2000).collect());
            summary.usage = *usage;
        }
        RunEventKind::Error { message } => summary.error = Some(message.clone()),
        RunEventKind::End { status } => {
            summary.status = (*status).into();
            summary.ended_at = Some(event.at);
        }
        _ => {}
    }
}

impl RunLog {
    fn publish(&mut self, batch: Vec<(f64, RunEventKind)>) {
        for (at, kind) in batch {
            let seq = self.events.last().map_or(0, |e| e.seq) + 1;
            let event = RunEvent { seq, at, kind };
            summary_after(&mut self.summary, &event);
            if matches!(event.kind, RunEventKind::End { .. }) {
                self.ended = true;
            }
            self.events.push(event);
        }
        for waker in self.wakers.drain(..) {
            waker.wake();
        }
    }
}

impl DemoService {
    /// The real-time service `--demo` runs on.
    pub fn new() -> Self {
        Self::with(Arc::new(clock::now), false)
    }

    /// A service whose runs finish without pausing, for tests.
    pub fn instant() -> Self {
        Self::with(Arc::new(clock::now), true)
    }

    pub fn with(clock: Clock, instant: bool) -> Self {
        let service = Self {
            inner: Arc::new(Inner {
                clock,
                instant,
                agents: catalogue_agents(),
                models: catalogue_models(),
                runs: Mutex::new(Vec::new()),
                next_run: Mutex::new(1),
            }),
        };
        service.seed_history();
        service
    }

    fn now(&self) -> f64 {
        (self.inner.clock)()
    }

    fn new_run_id(&self) -> String {
        let mut next = self
            .inner
            .next_run
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let n = *next;
        *next += 1;
        hex_of(0x1A77_1CE0_DE30_0001, n, 16)
    }

    fn agent_label(&self, id: &str) -> Option<String> {
        self.inner
            .agents
            .iter()
            .find(|a| a.id == id)
            .map(|a| a.label.clone())
    }

    /// Finished runs, so the list is not empty on first launch. Newest first.
    fn seed_history(&self) {
        let now = self.now();
        let history: [(&str, &str, &str, f64, Outcome); 4] = [
            (
                "Which of my models is best for summarising long documents?",
                ASSISTANT,
                LOCAL_MODEL,
                22.0 * 60.0,
                Outcome::Completed,
            ),
            (
                "Summarise this repository's README with the hosted model.",
                ASSISTANT,
                REMOTE_MODEL,
                5.0 * 3600.0,
                Outcome::Failed,
            ),
            (
                "Store my API key for the billing script.",
                ASSISTANT,
                LOCAL_MODEL,
                30.0 * 3600.0,
                Outcome::Refused,
            ),
            (
                "Which model should I use to translate a contract?",
                ADVISOR,
                REMOTE_MODEL,
                4.0 * 86_400.0,
                Outcome::Completed,
            ),
        ];
        let mut runs = Vec::new();
        for (task, agent, model, age, outcome) in history {
            let base = now - age;
            let id = self.new_run_id();
            let events = script(
                base,
                id.bytes()
                    .fold(7u64, |a, b| a.wrapping_mul(31).wrapping_add(u64::from(b))),
                task,
                agent,
                model,
                outcome,
            );
            let summary = self.summary_for(&id, task, agent, model, base, &events);
            let mut log = RunLog {
                summary,
                events: Vec::new(),
                wakers: Vec::new(),
                ended: false,
                stop_requested: false,
            };
            log.publish(events.into_iter().map(|(t, k)| (base + t, k)).collect());
            runs.push(Arc::new(RunShared {
                log: Mutex::new(log),
            }));
        }
        *self.inner.runs.lock().unwrap_or_else(|p| p.into_inner()) = runs;
    }

    fn summary_for(
        &self,
        id: &str,
        task: &str,
        agent: &str,
        model: &str,
        base: f64,
        events: &[(f64, RunEventKind)],
    ) -> RunSummary {
        let choice = self.inner.models.iter().find(|m| m.id == model);
        let trace_id = events
            .iter()
            .find_map(|(_, k)| {
                if let RunEventKind::TraceStart { trace_id, .. } = k {
                    Some(trace_id.clone())
                } else {
                    None
                }
            })
            .unwrap_or_default();
        RunSummary {
            id: id.to_string(),
            task: task.to_string(),
            agent: agent.to_string(),
            agent_label: self.agent_label(agent).unwrap_or_else(|| agent.to_string()),
            model: model.to_string(),
            model_label: choice.map_or_else(|| model.to_string(), |m| m.label.clone()),
            locality: choice.map_or(Locality::Local, |m| m.locality),
            status: RunStatus::Running,
            created_at: base,
            updated_at: base,
            ended_at: None,
            trace_id,
            usage: None,
            output: None,
            error: None,
            spans: 0,
        }
    }

    fn find(&self, id: &str) -> Result<Arc<RunShared>, Refusal> {
        if !lattice_protocol::is_run_id(id) {
            return Err(Refusal::new(RefusalKind::Invalid, "That is not a run id."));
        }
        self.inner
            .runs
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .find(|r| r.lock().summary.id == id)
            .cloned()
            .ok_or_else(|| Refusal::new(RefusalKind::NotFound, "There is no run with that id."))
    }
}

impl Default for DemoService {
    fn default() -> Self {
        Self::new()
    }
}

/// The follower's stream: yields whatever has accumulated since it last polled.
struct Follow {
    shared: Arc<RunShared>,
    after: u64,
    finished: bool,
}

impl Stream for Follow {
    type Item = Vec<RunEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }
        let shared = self.shared.clone();
        let mut log = shared.lock();
        let from = log.events.partition_point(|e| e.seq <= self.after);
        if from < log.events.len() {
            let batch: Vec<RunEvent> = log.events[from..].to_vec();
            self.after = batch.last().map_or(self.after, |e| e.seq);
            if batch
                .iter()
                .any(|e| matches!(e.kind, RunEventKind::End { .. }))
            {
                self.finished = true;
            }
            return Poll::Ready(Some(batch));
        }
        if log.ended {
            self.finished = true;
            return Poll::Ready(None);
        }
        if !log.wakers.iter().any(|w| w.will_wake(cx.waker())) {
            log.wakers.push(cx.waker().clone());
        }
        Poll::Pending
    }
}

/// Sleep until `clock() >= until`, in slices short enough to notice a stop.
/// Returns false when the run was asked to stop.
fn wait_until(shared: &RunShared, clock: &Clock, until: f64) -> bool {
    loop {
        if shared.lock().stop_requested {
            return false;
        }
        let remaining = until - clock();
        if remaining <= 0.0 {
            return true;
        }
        std::thread::sleep(Duration::from_secs_f64(remaining.min(0.02)));
    }
}

fn play(
    shared: Arc<RunShared>,
    clock: Clock,
    instant: bool,
    base: f64,
    script: Vec<(f64, RunEventKind)>,
) {
    let mut open: HashMap<String, SpanRecord> = HashMap::new();
    let mut open_order: Vec<String> = Vec::new();
    let mut trace_id = String::new();
    let mut i = 0;
    let mut stopped = false;
    while i < script.len() {
        let t = script[i].0;
        let mut j = i;
        while j < script.len() && (script[j].0 - t).abs() < 1e-9 {
            j += 1;
        }
        if !instant && !wait_until(&shared, &clock, base + t) {
            stopped = true;
            break;
        }
        if instant && shared.lock().stop_requested {
            stopped = true;
            break;
        }
        let mut batch = Vec::with_capacity(j - i);
        for (offset, kind) in &script[i..j] {
            match kind {
                RunEventKind::SpanStart { span } => {
                    open_order.push(span.id.clone());
                    open.insert(span.id.clone(), span.clone());
                }
                RunEventKind::SpanEnd { span } => {
                    open.remove(&span.id);
                }
                RunEventKind::TraceStart { trace_id: id, .. } => trace_id = id.clone(),
                _ => {}
            }
            batch.push((base + offset, kind.clone()));
        }
        shared.lock().publish(batch);
        i = j;
    }
    if stopped {
        // Close what is open, newest first, then end: a stopped run leaves no span running.
        let now = clock();
        let mut closing: Vec<(f64, RunEventKind)> = Vec::new();
        for id in open_order.iter().rev() {
            if let Some(span) = open.get(id) {
                let mut ended = span.clone();
                ended.ended_at = Some(iso_from_epoch(now));
                closing.push((now, RunEventKind::SpanEnd { span: ended }));
            }
        }
        closing.push((
            now,
            RunEventKind::TraceEnd {
                trace_id,
                at: iso_from_epoch(now),
            },
        ));
        closing.push((
            now,
            RunEventKind::End {
                status: EndStatus::Stopped,
            },
        ));
        shared.lock().publish(closing);
    }
}

impl RunService for DemoService {
    fn status(&self) -> ServiceStatus {
        ServiceStatus {
            runtime: "Demonstration data (not the agent runtime)".into(),
            traces: "this machine".into(),
            refusal: None,
        }
    }

    fn agents(&self) -> Vec<AgentInfo> {
        self.inner.agents.clone()
    }

    fn models(&self) -> Vec<ModelChoice> {
        self.inner.models.clone()
    }

    fn runs(&self) -> Vec<RunSummary> {
        self.inner
            .runs
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .map(|r| r.lock().summary.clone())
            .collect()
    }

    fn run(&self, id: &str) -> Result<RunDetail, Refusal> {
        let shared = self.find(id)?;
        let log = shared.lock();
        let mut spans: HashMap<&str, &SpanRecord> = HashMap::new();
        let mut trace: Option<TraceInfo> = None;
        for event in &log.events {
            match &event.kind {
                RunEventKind::SpanStart { span } => {
                    spans.entry(span.id.as_str()).or_insert(span);
                }
                RunEventKind::SpanEnd { span } => {
                    spans.insert(span.id.as_str(), span);
                }
                RunEventKind::TraceStart {
                    trace_id,
                    workflow_name,
                    at,
                } => {
                    trace = Some(TraceInfo {
                        id: trace_id.clone(),
                        workflow_name: workflow_name.clone(),
                        started_at: Some(at.clone()),
                        ended_at: None,
                    });
                }
                RunEventKind::TraceEnd { at, .. } => {
                    if let Some(t) = &mut trace {
                        t.ended_at = Some(at.clone());
                    }
                }
                _ => {}
            }
        }
        let mut spans: Vec<SpanRecord> = spans.into_values().cloned().collect();
        spans.sort_by_key(|s| (s.order, s.id.clone()));
        let events = log
            .events
            .iter()
            .filter(|e| !matches!(e.kind, RunEventKind::Delta { .. }))
            .cloned()
            .collect();
        Ok(RunDetail {
            run: log.summary.clone(),
            trace,
            spans,
            events,
        })
    }

    fn start(&self, request: StartRun) -> Result<RunSummary, Refusal> {
        let task = request.task.trim();
        if task.is_empty() {
            return Err(Refusal::new(RefusalKind::Invalid, "Write the task first."));
        }
        if self.agent_label(&request.agent).is_none() {
            return Err(Refusal::new(
                RefusalKind::Invalid,
                "That agent does not exist.",
            ));
        }
        let Some(model) = self.inner.models.iter().find(|m| m.id == request.model) else {
            return Err(Refusal::new(
                RefusalKind::Invalid,
                "That model does not exist.",
            ));
        };
        if !model.ready {
            return Err(Refusal::new(
                RefusalKind::Invalid,
                model
                    .refusal
                    .clone()
                    .unwrap_or_else(|| "That model is not ready.".into()),
            ));
        }
        let running = self.runs().iter().filter(|r| r.status.is_active()).count();
        if running >= MAX_RUNNING {
            return Err(Refusal::new(
                RefusalKind::Conflict,
                format!(
                    "{MAX_RUNNING} runs are already working. Wait for one to finish, or stop it."
                ),
            ));
        }
        let base = self.now();
        let id = self.new_run_id();
        let lowered = task.to_lowercase();
        let outcome = if ["api key", "password", "secret", "token"]
            .iter()
            .any(|w| lowered.contains(w))
        {
            Outcome::Refused
        } else {
            Outcome::Completed
        };
        let seed = id
            .bytes()
            .fold(11u64, |a, b| a.wrapping_mul(131).wrapping_add(u64::from(b)));
        let events = script(base, seed, task, &request.agent, &request.model, outcome);
        let summary = self.summary_for(&id, task, &request.agent, &request.model, base, &events);
        let shared = Arc::new(RunShared {
            log: Mutex::new(RunLog {
                summary: summary.clone(),
                events: Vec::new(),
                wakers: Vec::new(),
                ended: false,
                stop_requested: false,
            }),
        });
        self.inner
            .runs
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(0, shared.clone());
        let (clock, instant) = (self.inner.clock.clone(), self.inner.instant);
        let spawned = std::thread::Builder::new()
            .name("lattice-demo-run".into())
            .spawn(move || play(shared, clock, instant, base, events));
        if spawned.is_err() {
            return Err(Refusal::new(
                RefusalKind::Unavailable,
                "The demonstration run could not start a thread.",
            ));
        }
        Ok(summary)
    }

    fn stop(&self, id: &str) -> Result<(), Refusal> {
        let shared = self.find(id)?;
        let mut log = shared.lock();
        if log.ended || !log.summary.status.is_active() {
            return Err(Refusal::new(
                RefusalKind::Conflict,
                "That run is not running.",
            ));
        }
        log.stop_requested = true;
        Ok(())
    }

    fn follow(&self, id: &str, after: u64) -> Result<BoxStream<'static, Vec<RunEvent>>, Refusal> {
        let shared = self.find(id)?;
        Ok(Box::pin(Follow {
            shared,
            after,
            finished: false,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runstate::RunState;
    use crate::spans::{SpanKind, SpanTree, kind_of};
    use futures::StreamExt;
    use futures::executor::block_on;

    fn request(task: &str) -> StartRun {
        StartRun {
            task: task.into(),
            agent: ASSISTANT.into(),
            model: LOCAL_MODEL.into(),
        }
    }

    fn collect(service: &DemoService, id: &str, after: u64) -> Vec<RunEvent> {
        let batches: Vec<Vec<RunEvent>> = block_on(service.follow(id, after).unwrap().collect());
        batches.into_iter().flatten().collect()
    }

    fn assert_coherent(events: &[RunEvent]) {
        assert!(!events.is_empty());
        for (i, e) in events.iter().enumerate() {
            assert_eq!(e.seq, i as u64 + 1, "seq rises by one from 1");
        }
        assert!(matches!(events[0].kind, RunEventKind::TraceStart { .. }));
        assert!(
            matches!(events.last().unwrap().kind, RunEventKind::End { .. }),
            "End is last"
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e.kind, RunEventKind::End { .. }))
                .count(),
            1
        );

        let mut started: HashMap<&str, &SpanRecord> = HashMap::new();
        let mut ended: HashMap<&str, &SpanRecord> = HashMap::new();
        for e in events {
            match &e.kind {
                RunEventKind::SpanStart { span } => {
                    assert!(span.ended_at.is_none(), "a start record is open");
                    if let Some(parent) = &span.parent_id {
                        let p = started
                            .get(parent.as_str())
                            .unwrap_or_else(|| panic!("parent {parent} starts before its child"));
                        assert!(p.order < span.order, "parents have the lower order");
                    }
                    assert!(
                        started.insert(span.id.as_str(), span).is_none(),
                        "a span starts once"
                    );
                }
                RunEventKind::SpanEnd { span } => {
                    let s = started
                        .get(span.id.as_str())
                        .expect("an end follows its start");
                    assert_eq!(
                        (s.order, &s.parent_id, &s.started_at),
                        (span.order, &span.parent_id, &span.started_at),
                        "an end keeps the start's identity"
                    );
                    assert!(span.ended_at.is_some());
                    assert!(
                        ended.insert(span.id.as_str(), span).is_none(),
                        "a span ends once"
                    );
                }
                _ => {}
            }
        }
        assert_eq!(
            started.len(),
            ended.len(),
            "every span that starts also ends"
        );
    }

    #[test]
    fn start_then_follow_yields_a_coherent_sequence_ending_in_end() {
        let service = DemoService::instant();
        let before = service.runs().len();
        let summary = service
            .start(request("Which model suits translating a contract?"))
            .unwrap();
        assert_eq!(summary.status, RunStatus::Running);
        assert!(lattice_protocol::is_run_id(&summary.id));
        let events = collect(&service, &summary.id, 0);
        assert_coherent(&events);
        let end = events.last().unwrap();
        assert!(matches!(
            end.kind,
            RunEventKind::End {
                status: EndStatus::Completed
            }
        ));

        // The story the demo tells: both agents, a tool, a hand-off, a guardrail, streamed text, one result.
        let names: Vec<String> = events
            .iter()
            .filter_map(|e| {
                if let RunEventKind::Agent { name } = &e.kind {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(names, [ASSISTANT_LABEL, ADVISOR_LABEL]);
        assert!(
            events
                .iter()
                .any(|e| matches!(e.kind, RunEventKind::Handoff { .. }))
        );
        assert!(events.iter().any(
            |e| matches!(&e.kind, RunEventKind::ToolCall { name, .. } if name == "current_time")
        ));
        assert!(events.iter().any(
            |e| matches!(&e.kind, RunEventKind::ToolCall { name, .. } if name == "list_models")
        ));
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e.kind, RunEventKind::Result { .. }))
                .count(),
            1
        );
        let deltas = events
            .iter()
            .filter(|e| matches!(e.kind, RunEventKind::Delta { .. }))
            .count();
        assert!(deltas > 20, "text streams in many deltas, got {deltas}");

        // Each agent's streamed text is followed by its complete message, which replaces it.
        let mut state = RunState::new(service.runs()[0].clone());
        for e in &events {
            state.apply(e);
        }
        assert!(state.live.is_empty());
        assert_eq!(state.status, RunStatus::Completed);
        assert!(
            state
                .output
                .as_deref()
                .unwrap()
                .starts_with("**Short answer:**")
        );

        // The runs list updated: one more run, the new one first, now completed with usage.
        let runs = service.runs();
        assert_eq!(runs.len(), before + 1);
        assert_eq!(runs[0].id, summary.id);
        assert_eq!(runs[0].status, RunStatus::Completed);
        assert!(
            runs[0]
                .usage
                .is_some_and(|u| u.total_tokens > 0 && u.requests == 4)
        );
        assert!(runs[0].spans > 10);
    }

    #[test]
    fn deltas_arrive_in_bursts_that_a_slow_follower_receives_as_one_batch() {
        let service = DemoService::instant();
        let summary = service
            .start(request("Which model suits translating a contract?"))
            .unwrap();
        // Wait for the run to finish, then follow from the start: everything is one batch.
        loop {
            if service.runs()[0].status != RunStatus::Running {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        let batches: Vec<Vec<RunEvent>> =
            block_on(service.follow(&summary.id, 0).unwrap().collect());
        assert_eq!(batches.len(), 1, "an ended run replays as a single batch");
        assert!(matches!(
            batches[0].last().unwrap().kind,
            RunEventKind::End { .. }
        ));
    }

    #[test]
    fn following_a_finished_run_after_its_last_event_ends_at_once() {
        let service = DemoService::instant();
        let id = service.runs()[0].id.clone();
        let all = collect(&service, &id, 0);
        assert_coherent(&all);
        let last = all.last().unwrap().seq;
        let rest: Vec<Vec<RunEvent>> = block_on(service.follow(&id, last).unwrap().collect());
        assert!(rest.is_empty());
        // Following from the middle gives exactly the rest.
        let middle = all.len() as u64 / 2;
        let tail = collect(&service, &id, middle);
        assert_eq!(tail.len(), all.len() - middle as usize);
        assert_eq!(tail[0].seq, middle + 1);
    }

    #[test]
    fn the_history_has_a_completed_a_failed_and_a_refused_run() {
        let service = DemoService::instant();
        let runs = service.runs();
        let statuses: Vec<RunStatus> = runs.iter().map(|r| r.status).collect();
        assert_eq!(
            statuses,
            [
                RunStatus::Completed,
                RunStatus::Failed,
                RunStatus::Refused,
                RunStatus::Completed
            ]
        );
        for run in &runs {
            assert!(lattice_protocol::is_run_id(&run.id), "{}", run.id);
            let detail = service.run(&run.id).unwrap();
            assert_coherent(&collect(&service, &run.id, 0));
            // The detail replays into a state with nothing open and the right end.
            let state = RunState::from_detail(detail);
            assert!(!state.following);
            assert_eq!(state.status, run.status);
            assert!(state.spans.iter().all(|s| s.end.is_some()));
            // And its spans form one tree rooted at the task span.
            let tree = SpanTree::build(&state.spans);
            assert_eq!(tree.roots.len(), 1, "{}: one root", run.task);
            assert_eq!(kind_of(&state.spans[tree.roots[0]].rec), SpanKind::Task);
        }
        // Newest first, strictly older as we go down.
        assert!(runs.windows(2).all(|w| w[0].created_at > w[1].created_at));
        let failed = service.run(&runs[1].id).unwrap();
        assert!(
            failed.spans.iter().any(|s| s.error.is_some()),
            "the failed run has an error span"
        );
        assert!(failed.run.error.is_some());
        let refused = service.run(&runs[2].id).unwrap();
        assert!(
            refused.spans.iter().all(|s| kind_of(s) != SpanKind::Model),
            "no model saw a refused task"
        );
        assert!(
            refused
                .spans
                .iter()
                .any(|s| s.span_data["triggered"] == json!(true))
        );
        assert!(
            refused
                .events
                .iter()
                .any(|e| matches!(e.kind, RunEventKind::Guardrail { .. }))
        );
    }

    #[test]
    fn a_task_that_mentions_a_secret_is_refused_by_the_guardrail() {
        let service = DemoService::instant();
        let summary = service
            .start(request("Remember this password for the next task."))
            .unwrap();
        let events = collect(&service, &summary.id, 0);
        assert_coherent(&events);
        assert!(matches!(
            events.last().unwrap().kind,
            RunEventKind::End {
                status: EndStatus::Refused
            }
        ));
        let message = events
            .iter()
            .find_map(|e| {
                if let RunEventKind::Guardrail { message, .. } = &e.kind {
                    Some(message.clone())
                } else {
                    None
                }
            })
            .unwrap();
        assert!(
            !message.contains("password"),
            "the refusal never repeats what matched"
        );
    }

    #[test]
    fn detail_of_a_run_has_sorted_spans_and_no_deltas() {
        let service = DemoService::instant();
        let summary = service
            .start(request("Which model suits translating a contract?"))
            .unwrap();
        let _ = collect(&service, &summary.id, 0);
        let detail = service.run(&summary.id).unwrap();
        assert!(detail.spans.windows(2).all(|w| w[0].order < w[1].order));
        assert!(
            detail
                .events
                .iter()
                .all(|e| !matches!(e.kind, RunEventKind::Delta { .. }))
        );
        assert_eq!(detail.trace.as_ref().unwrap().id, detail.run.trace_id);
        assert!(detail.trace.as_ref().unwrap().ended_at.is_some());
        // Superseded records are gone: every span in the detail is the ended one.
        assert!(detail.spans.iter().all(|s| s.ended_at.is_some()));
    }

    #[test]
    fn start_refuses_what_it_cannot_run_with_a_sentence() {
        let service = DemoService::instant();
        let empty = service.start(request("   ")).unwrap_err();
        assert_eq!(empty.kind, RefusalKind::Invalid);
        assert_eq!(empty.message, "Write the task first.");
        let mut bad_agent = request("x");
        bad_agent.agent = "nobody".into();
        assert_eq!(
            service.start(bad_agent).unwrap_err().kind,
            RefusalKind::Invalid
        );
        let mut bad_model = request("x");
        bad_model.model = "endpoint:missing".into();
        assert_eq!(
            service.start(bad_model).unwrap_err().kind,
            RefusalKind::Invalid
        );
        let mut unready = request("x");
        unready.model = OFFLINE_MODEL.into();
        let refusal = service.start(unready).unwrap_err();
        assert_eq!(
            refusal.message, OFFLINE_REFUSAL,
            "the model's own refusal text is what the person reads"
        );
        assert_eq!(
            service.stop("../../etc/passwd").unwrap_err().kind,
            RefusalKind::Invalid
        );
        assert_eq!(
            service.stop("ffffffffffffffff").unwrap_err().kind,
            RefusalKind::NotFound
        );
        assert!(service.follow("nope", 0).is_err());
        assert!(service.run("ffffffffffffffff").is_err());
        // Nothing was started by any of that.
        assert_eq!(service.runs().len(), 4);
    }

    #[test]
    fn stop_ends_a_working_run_promptly_with_every_span_closed() {
        let service = DemoService::new(); // real time: the run would take about four seconds
        let summary = service
            .start(request("Which model suits translating a contract?"))
            .unwrap();
        std::thread::sleep(Duration::from_millis(400));
        assert!(service.runs()[0].status.is_active());
        service.stop(&summary.id).unwrap();
        let started = std::time::Instant::now();
        let events = collect(&service, &summary.id, 0);
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "a stop is prompt, took {:?}",
            started.elapsed()
        );
        assert_coherent(&events);
        assert!(matches!(
            events.last().unwrap().kind,
            RunEventKind::End {
                status: EndStatus::Stopped
            }
        ));
        assert_eq!(service.runs()[0].status, RunStatus::Stopped);
        // A second stop is a conflict, not a crash.
        assert_eq!(
            service.stop(&summary.id).unwrap_err().kind,
            RefusalKind::Conflict
        );
    }

    #[test]
    fn at_most_three_runs_work_at_once() {
        let service = DemoService::new();
        let ids: Vec<String> = (0..MAX_RUNNING)
            .map(|i| {
                service
                    .start(request(&format!("Task number {i}")))
                    .unwrap()
                    .id
            })
            .collect();
        let refusal = service.start(request("One too many")).unwrap_err();
        assert_eq!(refusal.kind, RefusalKind::Conflict);
        for id in &ids {
            service.stop(id).unwrap();
        }
        for id in &ids {
            let events = collect(&service, id, 0);
            assert!(matches!(
                events.last().unwrap().kind,
                RunEventKind::End {
                    status: EndStatus::Stopped
                }
            ));
        }
        // With none working, a new run is accepted again (and stopped to leave no thread behind).
        let again = service.start(request("Room again")).unwrap();
        service.stop(&again.id).unwrap();
        let _ = collect(&service, &again.id, 0);
    }

    #[test]
    fn a_live_run_reaches_a_polling_follower_in_more_than_one_batch() {
        let service = DemoService::new();
        let summary = service
            .start(request("Which model suits translating a contract?"))
            .unwrap();
        let mut stream = service.follow(&summary.id, 0).unwrap();
        let first = block_on(stream.next()).unwrap();
        assert!(matches!(first[0].kind, RunEventKind::TraceStart { .. }));
        let second = block_on(stream.next()).unwrap();
        assert!(
            second[0].seq > first.last().unwrap().seq,
            "the second batch continues where the first stopped"
        );
        drop(stream);
        service.stop(&summary.id).unwrap();
        let _ = collect(&service, &summary.id, 0);
    }

    #[test]
    fn scripts_are_deterministic_and_use_python_style_timestamps() {
        let a = script(
            1_800_000_000.0,
            42,
            "Task",
            ASSISTANT,
            LOCAL_MODEL,
            Outcome::Completed,
        );
        let b = script(
            1_800_000_000.0,
            42,
            "Task",
            ASSISTANT,
            LOCAL_MODEL,
            Outcome::Completed,
        );
        assert_eq!(a, b);
        let RunEventKind::TraceStart { trace_id, at, .. } = &a[0].1 else {
            panic!("trace start first")
        };
        assert!(trace_id.starts_with("trace_") && trace_id.len() == 6 + 32);
        assert_eq!(clock::parse_iso(at), Some(1_800_000_000.0));
        for (offset, kind) in &a {
            if let RunEventKind::SpanStart { span } | RunEventKind::SpanEnd { span } = kind {
                assert!(
                    span.id.starts_with("span_") && span.id.len() == 5 + 24,
                    "{}",
                    span.id
                );
                assert!(clock::parse_iso(&span.started_at).is_some());
                let started = clock::parse_iso(&span.started_at).unwrap();
                assert!(
                    started <= 1_800_000_000.0 + offset + 1e-6,
                    "a span never starts after the event that announces it"
                );
            }
        }
        assert!(
            a.windows(2).all(|w| w[0].0 <= w[1].0),
            "offsets never go backwards"
        );
        // The advisor-only script has no hand-off.
        let solo = script(0.0, 1, "Task", ADVISOR, LOCAL_MODEL, Outcome::Completed);
        assert!(
            !solo
                .iter()
                .any(|(_, k)| matches!(k, RunEventKind::Handoff { .. }))
        );
    }

    #[test]
    fn the_catalogue_offers_ready_and_unready_models_and_two_agents() {
        let service = DemoService::instant();
        let models = service.models();
        assert_eq!(models.len(), 3);
        assert!(
            models
                .iter()
                .any(|m| m.ready && m.locality == Locality::Local)
        );
        assert!(
            models
                .iter()
                .any(|m| m.ready && m.locality == Locality::Remote)
        );
        let unready: Vec<&ModelChoice> = models.iter().filter(|m| !m.ready).collect();
        assert_eq!(unready.len(), 1);
        assert!(unready[0].refusal.as_deref().is_some_and(|r| !r.is_empty()));
        let agents = service.agents();
        assert_eq!(agents.len(), 2);
        assert!(
            agents[0]
                .graph
                .nodes
                .iter()
                .any(|n| n.label == "list_models")
        );
        assert!(
            agents[0]
                .graph
                .edges
                .iter()
                .any(|e| e.kind == EdgeKind::Handoff)
        );
        assert_eq!(service.status().traces, "this machine");
    }
}
