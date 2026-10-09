//! The built-in agents: what they know, what they can call, and what stops them.
//!
//! Two agents, as the Python Lattice's assistant pair:
//!
//! - **Lattice assistant** (`lattice-assistant`): answers with read-only tools,
//!   the clock and arithmetic, and hands questions about which model to use to
//! - **Model advisor**: knows the models this Lattice can use and where each one
//!   runs, through one tool, `list_models`.
//!
//! Every tool is read-only and total: none writes, runs a program, opens a
//! network connection or reads a file.
//!
//! - `current_time {}` gives the UTC time and the weekday (the clock is
//!   injectable);
//! - `calculate {expression}` evaluates arithmetic with the hand-written
//!   evaluator in [`crate::calc`], which cannot run code;
//! - `list_models {}` lists the choices as `{id, label, kind, locality, ready}`
//!   (locality is "this machine" or "leaves this machine"). It carries no
//!   address, no key name and no key: the tool's result goes back to the model,
//!   and a remote model must not learn where the person's servers are or what
//!   keys they hold.
//!
//! Tool failures use a custom policy: the model hears `The tool failed: <reason>`
//! (the whole text at most 500 characters), and the reason is the tool's own
//! sentence, never a Rust error or a path.
//!
//! The one guardrail, `secrets_stay_local`, is an input guardrail on the
//! assistant. It reads the run context ([`AgentRunContext`], which says whether
//! the chosen model is on this machine). For a model that is not local, a task
//! that looks like it holds a credential ([`crate::secrets`]) trips it before any
//! model is called; for a local model it passes. When the context is missing the
//! guardrail assumes the model is NOT local: failing closed.
//!
//! Invariant: what a guardrail's `output_info` says never repeats the matched
//! text, only the reason.

use std::sync::Arc;

use lattice_agents::{
    Agent, FunctionTool, GuardrailOutput, InputGuardrail, ToolContext, ToolError, ToolErrorPolicy,
    agent_graph, strict_object_schema,
};
use lattice_protocol::{AgentInfo, Locality, ModelChoice, ModelKind};
use serde_json::{Value, json};

use crate::bound::cap_text;
use crate::calc;
use crate::clock::Clock;
use crate::secrets::looks_like_secret;

/// The id of the assistant, the agent a person starts a run with.
pub const ASSISTANT_ID: &str = "lattice-assistant";
pub const ASSISTANT_LABEL: &str = "Lattice assistant";
pub const ADVISOR_LABEL: &str = "Model advisor";
pub const GUARDRAIL_NAME: &str = "secrets_stay_local";

const ASSISTANT_DESCRIPTION: &str = "Answers with read-only tools: the clock, arithmetic and your model list. Hands model questions to the Model advisor.";
const MAX_TOOL_ERROR_CHARS: usize = 500;

const ASSISTANT_INSTRUCTIONS: &str = "You are the Lattice assistant. Answer concisely. Use the current_time tool for the time and the date, and the calculate tool for arithmetic, instead of guessing. If the person asks which model to use, hand the conversation to the Model advisor.";
const ADVISOR_INSTRUCTIONS: &str = "You are the Model advisor. Call list_models, then recommend one of the models it lists for the person's task. Say plainly which models leave this machine and which run on this machine. Answer concisely.";

pub const REASON_LOCAL: &str = "The model runs on this machine.";
pub const REASON_SECRET: &str =
    "The task contains what looks like a secret, and the chosen model is not on this machine.";
pub const REASON_CLEAN: &str = "The task contains nothing that looks like a secret.";

/// What the run tells its tools and guardrails (`RunConfig::context`).
#[derive(Clone, Debug)]
pub struct AgentRunContext {
    /// Whether the chosen model runs on this machine.
    pub model_local: bool,
    /// The choices `list_models` describes (the development model is left out
    /// of its answer).
    pub models: Vec<ModelChoice>,
}

/// The agents, built once.
pub struct Catalog {
    assistant: Arc<Agent>,
    /// What `agents()` answers, worked out once (the graph is a walk).
    infos: Vec<AgentInfo>,
}

/// `The tool failed: <reason>`, at most 500 characters in all.
fn tool_failed(error: &ToolError) -> String {
    cap_text(
        &format!("The tool failed: {}", error.message()),
        MAX_TOOL_ERROR_CHARS,
    )
}

fn policy() -> ToolErrorPolicy {
    ToolErrorPolicy::Custom(Arc::new(tool_failed))
}

/// The UTC time and weekday, as the model reads them.
fn time_text(clock: &Clock) -> String {
    let seconds = clock().floor() as i64;
    match time::OffsetDateTime::from_unix_timestamp(seconds) {
        Ok(at) => json!({
            "utc": format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
                at.year(),
                u8::from(at.month()),
                at.day(),
                at.hour(),
                at.minute(),
                at.second()
            ),
            "weekday": at.weekday().to_string(),
        })
        .to_string(),
        Err(_) => json!({"utc": null, "weekday": null}).to_string(),
    }
}

fn current_time_tool(clock: Clock) -> FunctionTool {
    FunctionTool::new(
        "current_time",
        "The current date and time in UTC, with the weekday.",
        strict_object_schema(json!({}), &[]),
        move |_, _| {
            let clock = clock.clone();
            async move { Ok(time_text(&clock)) }
        },
    )
    .with_error_policy(policy())
}

fn calculate_tool() -> FunctionTool {
    FunctionTool::new(
        "calculate",
        "Evaluate an arithmetic expression. Numbers, + - * / // % ** and parentheses only; Python's rules for // % and **.",
        strict_object_schema(
            json!({"expression": {"type": "string", "description": "For example: (12 + 30) * 2 / 7"}}),
            &["expression"],
        ),
        |_, arguments| async move {
            let expression = arguments
                .get("expression")
                .and_then(Value::as_str)
                .ok_or_else(|| ToolError::new("expression must be a string"))?;
            calc::calculate(expression).map_err(|error| ToolError::new(error.to_string()))
        },
    )
    .with_error_policy(policy())
}

fn kind_name(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::Ollama => "ollama",
        ModelKind::OpenaiCompatible => "openai-compatible",
        ModelKind::Anthropic => "anthropic",
        ModelKind::Llamacpp => "llamacpp",
        ModelKind::Development => "development",
    }
}

/// What `list_models` answers, from the run's context.
fn models_text(context: &AgentRunContext) -> String {
    let rows: Vec<Value> = context
        .models
        .iter()
        .filter(|model| model.kind != ModelKind::Development)
        .map(|model| {
            json!({
                "id": model.id,
                "label": model.label,
                "kind": kind_name(model.kind),
                "locality": match model.locality {
                    Locality::Local => "this machine",
                    Locality::Remote => "leaves this machine",
                },
                "ready": model.ready,
            })
        })
        .collect();
    Value::Array(rows).to_string()
}

fn list_models_tool() -> FunctionTool {
    FunctionTool::new(
        "list_models",
        "The models this Lattice can use: id, label, kind, whether each runs on this machine or leaves it, and whether it is ready.",
        strict_object_schema(json!({}), &[]),
        |context: ToolContext, _| async move {
            match context.run_context.downcast_ref::<AgentRunContext>() {
                Some(run) => Ok(models_text(run)),
                None => Err(ToolError::new("the model list is not available in this run")),
            }
        },
    )
    .with_error_policy(policy())
}

fn secrets_stay_local() -> InputGuardrail {
    InputGuardrail::new(GUARDRAIL_NAME, |context, task| async move {
        let model_local = context
            .run_context
            .downcast_ref::<AgentRunContext>()
            .is_some_and(|run| run.model_local);
        if model_local {
            return GuardrailOutput {
                tripwire_triggered: false,
                output_info: json!({"reason": REASON_LOCAL}),
            };
        }
        if looks_like_secret(&task) {
            return GuardrailOutput::trip(json!({"reason": REASON_SECRET}));
        }
        GuardrailOutput {
            tripwire_triggered: false,
            output_info: json!({"reason": REASON_CLEAN}),
        }
    })
}

impl Catalog {
    /// Build the agents over a clock (the only thing a tool reads besides its
    /// arguments and the run's context).
    pub fn new(clock: Clock) -> Self {
        let advisor = Agent::builder(ADVISOR_LABEL)
            .handoff_description("Knows which models this Lattice can use and where each one runs.")
            .instructions(ADVISOR_INSTRUCTIONS)
            .tool(list_models_tool())
            .build();
        let assistant = Agent::builder(ASSISTANT_LABEL)
            .instructions(ASSISTANT_INSTRUCTIONS)
            .tool(current_time_tool(clock))
            .tool(calculate_tool())
            .handoff_to(advisor)
            .input_guardrail(secrets_stay_local())
            .build();
        let infos = vec![AgentInfo {
            id: ASSISTANT_ID.to_owned(),
            label: ASSISTANT_LABEL.to_owned(),
            description: ASSISTANT_DESCRIPTION.to_owned(),
            graph: agent_graph(&assistant),
        }];
        Self { assistant, infos }
    }

    /// The agents a person can start a run with.
    pub fn agents(&self) -> Vec<AgentInfo> {
        self.infos.clone()
    }

    /// The agent with this id.
    pub fn agent(&self, id: &str) -> Option<Arc<Agent>> {
        (id == ASSISTANT_ID).then(|| self.assistant.clone())
    }

    /// The name of the assistant's handoff tool (`transfer_to_model_advisor`),
    /// as the SDK built it: the development model calls it by this name.
    pub fn handoff_tool_name(&self) -> String {
        self.assistant
            .handoffs
            .first()
            .map(|handoff| handoff.tool_name.clone())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use lattice_protocol::{EdgeKind, NodeKind};

    use super::*;

    fn clock_at(seconds: f64) -> Clock {
        Arc::new(move || seconds)
    }

    fn catalog() -> Catalog {
        Catalog::new(clock_at(1_790_000_000.0))
    }

    fn context(local: bool) -> Arc<AgentRunContext> {
        Arc::new(AgentRunContext {
            model_local: local,
            models: vec![
                ModelChoice {
                    id: "dev:scripted".into(),
                    label: "Development model (scripted)".into(),
                    kind: ModelKind::Development,
                    locality: Locality::Local,
                    ready: true,
                    refusal: None,
                },
                ModelChoice {
                    id: "endpoint:on-this-machine".into(),
                    label: "A server on this machine".into(),
                    kind: ModelKind::OpenaiCompatible,
                    locality: Locality::Local,
                    ready: true,
                    refusal: None,
                },
                ModelChoice {
                    id: "endpoint:openai".into(),
                    label: "OpenAI".into(),
                    kind: ModelKind::OpenaiCompatible,
                    locality: Locality::Remote,
                    ready: false,
                    refusal: Some("It needs an API key, and none is set.".into()),
                },
            ],
        })
    }

    fn tool_context(local: bool) -> ToolContext {
        ToolContext {
            run_context: context(local),
            agent: ASSISTANT_LABEL.into(),
            call_id: "call".into(),
        }
    }

    fn tool<'a>(agent: &'a Agent, name: &str) -> &'a FunctionTool {
        agent.tools.iter().find(|t| t.name == name).unwrap()
    }

    async fn call(tool: &FunctionTool, arguments: Value) -> Result<String, ToolError> {
        (tool.handler)(tool_context(true), arguments).await
    }

    #[test]
    fn the_assistant_has_its_tools_its_handoff_and_its_guardrail() {
        let catalog = catalog();
        let agents = catalog.agents();
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0].id, "lattice-assistant");
        assert_eq!(agents[0].label, "Lattice assistant");
        assert_eq!(
            agents[0].description,
            "Answers with read-only tools: the clock, arithmetic and your model list. Hands model questions to the Model advisor."
        );
        let assistant = catalog.agent("lattice-assistant").unwrap();
        let names: Vec<&str> = assistant.tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["current_time", "calculate"]);
        assert_eq!(assistant.input_guardrails.len(), 1);
        assert_eq!(assistant.input_guardrails[0].name, "secrets_stay_local");
        assert!(assistant.output_guardrails.is_empty());
        assert_eq!(assistant.handoffs.len(), 1);
        let advisor = &assistant.handoffs[0].agent;
        assert_eq!(advisor.name, "Model advisor");
        assert_eq!(
            advisor.handoff_description.as_deref(),
            Some("Knows which models this Lattice can use and where each one runs.")
        );
        assert_eq!(advisor.tools.len(), 1);
        assert_eq!(advisor.tools[0].name, "list_models");
        assert_eq!(catalog.handoff_tool_name(), "transfer_to_model_advisor");
        assert!(catalog.agent("model-advisor").is_none());
        assert!(catalog.agent("").is_none());
    }

    #[test]
    fn the_graph_is_the_sdks_walk_of_the_pair() {
        let graph = catalog().agents().remove(0).graph;
        let kinds: Vec<(NodeKind, &str)> = graph
            .nodes
            .iter()
            .map(|n| (n.kind, n.label.as_str()))
            .collect();
        assert_eq!(
            kinds,
            [
                (NodeKind::Start, "Start"),
                (NodeKind::Agent, "Lattice assistant"),
                (NodeKind::Tool, "current_time"),
                (NodeKind::Tool, "calculate"),
                (NodeKind::Agent, "Model advisor"),
                (NodeKind::Tool, "list_models"),
                (NodeKind::End, "End"),
            ]
        );
        assert!(graph.nodes.iter().filter(|n| n.root).count() == 1);
        assert!(graph.edges.iter().any(|e| e.kind == EdgeKind::Handoff));
        assert!(graph.edges.iter().any(|e| e.kind == EdgeKind::End));
    }

    #[test]
    fn every_tool_has_a_strict_schema_and_the_custom_error_policy() {
        let assistant = catalog().agent(ASSISTANT_ID).unwrap();
        let advisor = assistant.handoffs[0].agent.clone();
        for tool in assistant.tools.iter().chain(advisor.tools.iter()) {
            assert!(tool.strict, "{}", tool.name);
            assert_eq!(tool.parameters["type"], "object", "{}", tool.name);
            assert_eq!(
                tool.parameters["additionalProperties"], false,
                "{}",
                tool.name
            );
            assert!(
                matches!(tool.error_policy, ToolErrorPolicy::Custom(_)),
                "{}",
                tool.name
            );
        }
        assert_eq!(
            tool(&assistant, "calculate").parameters["required"],
            json!(["expression"])
        );
    }

    #[test]
    fn the_failure_text_is_the_reason_capped_at_five_hundred_characters() {
        assert_eq!(
            tool_failed(&ToolError::new("division by zero")),
            "The tool failed: division by zero"
        );
        let long = tool_failed(&ToolError::new("x".repeat(2_000)));
        assert_eq!(long.chars().count(), MAX_TOOL_ERROR_CHARS);
        assert!(long.starts_with("The tool failed: xxx") && long.ends_with('\u{2026}'));
    }

    #[tokio::test]
    async fn current_time_reads_the_injected_clock() {
        // 1,790,000,000 s after the epoch is Monday 2026-09-21 14:13:20 UTC (checked
        // against Python's `datetime`); the fraction of a second is dropped.
        let catalog = Catalog::new(clock_at(1_790_000_000.9));
        let assistant = catalog.agent(ASSISTANT_ID).unwrap();
        let answer = call(tool(&assistant, "current_time"), json!({}))
            .await
            .unwrap();
        let value: Value = serde_json::from_str(&answer).unwrap();
        assert_eq!(value["utc"], "2026-09-21T14:13:20Z");
        assert_eq!(value["weekday"], "Monday");
        let far = Catalog::new(clock_at(1e30));
        let assistant = far.agent(ASSISTANT_ID).unwrap();
        let answer = call(tool(&assistant, "current_time"), json!({}))
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&answer).unwrap()["utc"],
            Value::Null
        );
    }

    #[tokio::test]
    async fn calculate_answers_and_refuses_in_the_tools_own_words() {
        let assistant = catalog().agent(ASSISTANT_ID).unwrap();
        let calculate = tool(&assistant, "calculate");
        assert_eq!(
            call(calculate, json!({"expression": "6 * 7"}))
                .await
                .unwrap(),
            "42"
        );
        assert_eq!(
            call(calculate, json!({"expression": "1 / 4"}))
                .await
                .unwrap(),
            "0.25"
        );
        let refused = call(
            calculate,
            json!({"expression": "__import__('os').system('x')"}),
        )
        .await
        .unwrap_err();
        assert_eq!(
            refused.message(),
            "Only numbers, + - * / // % ** and parentheses are allowed."
        );
        assert_eq!(
            call(calculate, json!({"expression": "1 / 0"}))
                .await
                .unwrap_err()
                .message(),
            "division by zero"
        );
        for bad in [
            json!({}),
            json!({"expression": 5}),
            json!({"expression": null}),
            json!({"exp": "1"}),
        ] {
            assert_eq!(
                call(calculate, bad).await.unwrap_err().message(),
                "expression must be a string"
            );
        }
        let ToolErrorPolicy::Custom(format) = &calculate.error_policy else {
            panic!("custom policy")
        };
        assert_eq!(
            format(&ToolError::new("division by zero")),
            "The tool failed: division by zero"
        );
    }

    #[tokio::test]
    async fn list_models_names_no_address_no_key_and_no_development_model() {
        let assistant = catalog().agent(ASSISTANT_ID).unwrap();
        let advisor = assistant.handoffs[0].agent.clone();
        let list = tool(&advisor, "list_models");
        let answer = call(list, json!({})).await.unwrap();
        let rows: Value = serde_json::from_str(&answer).unwrap();
        assert_eq!(
            rows,
            json!([
                {"id": "endpoint:on-this-machine", "label": "A server on this machine", "kind": "openai-compatible", "locality": "this machine", "ready": true},
                {"id": "endpoint:openai", "label": "OpenAI", "kind": "openai-compatible", "locality": "leaves this machine", "ready": false},
            ])
        );
        for forbidden in ["http", "KEY", "refusal", "dev:scripted", "base_url"] {
            assert!(!answer.contains(forbidden), "{forbidden} in {answer}");
        }
        let without = (list.handler)(
            ToolContext {
                run_context: Arc::new(()),
                agent: "x".into(),
                call_id: "c".into(),
            },
            json!({}),
        )
        .await;
        assert_eq!(
            without.unwrap_err().message(),
            "the model list is not available in this run"
        );
    }

    async fn verdict(local: Option<bool>, task: &str) -> GuardrailOutput {
        let catalog = catalog();
        let assistant = catalog.agent(ASSISTANT_ID).unwrap();
        let guardrail = &assistant.input_guardrails[0];
        let run_context: Arc<dyn std::any::Any + Send + Sync> = match local {
            Some(local) => context(local),
            None => Arc::new(()),
        };
        (guardrail.check)(
            lattice_agents::GuardrailContext {
                run_context,
                agent: ASSISTANT_LABEL.into(),
            },
            task.to_owned(),
        )
        .await
    }

    #[tokio::test]
    async fn secrets_stay_local_trips_only_for_a_remote_model_and_a_secret_looking_task() {
        // Assembled from pieces: no source file holds a string a secret scanner would report.
        let secret_task = concat!(
            "please use ",
            "s",
            "k-abcdefghijklmnopqrstuv to call the API"
        );
        let tripped = verdict(Some(false), secret_task).await;
        assert!(tripped.tripwire_triggered);
        assert_eq!(tripped.output_info, json!({"reason": REASON_SECRET}));
        assert!(
            !tripped
                .output_info
                .to_string()
                .contains(concat!("s", "k-abc")),
            "the match is never repeated"
        );

        let local = verdict(Some(true), secret_task).await;
        assert!(!local.tripwire_triggered);
        assert_eq!(
            local.output_info,
            json!({"reason": "The model runs on this machine."})
        );

        let clean = verdict(Some(false), "What is 6 * 7?").await;
        assert!(!clean.tripwire_triggered);

        let unknown = verdict(None, secret_task).await;
        assert!(
            unknown.tripwire_triggered,
            "a missing context is treated as a remote model"
        );
        assert_eq!(
            REASON_SECRET,
            "The task contains what looks like a secret, and the chosen model is not on this machine."
        );
    }
}
