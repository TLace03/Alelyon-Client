//! Guardrails: checks that can stop a run.
//!
//! Ports `agents.guardrail.InputGuardrail` / `OutputGuardrail` and
//! `GuardrailFunctionOutput` (0.22.3), as closures over the text they check.
//!
//! - An input guardrail sees the text of the last user item the run started
//!   with (for a run started from a string, that string) and runs on the
//!   FIRST agent only, before its first model call. A tripped guardrail stops
//!   the run with [`crate::RunError::InputGuardrailTripwire`]; the model is
//!   never called.
//! - An output guardrail sees the final output of the agent that produced it.
//!   A tripped guardrail stops the run with
//!   [`crate::RunError::OutputGuardrailTripwire`]; the answer is not returned
//!   (the events that carried it have already been sent).
//! - Each check that runs gets a `guardrail` span with its name and whether it
//!   tripped.
//!
//! Deviations, and why:
//! - The SDK hands an input guardrail the input as a list of input items
//!   (`[{"role": "user", "content": ...}]`, or the whole list a run was started
//!   from); this port hands it the text of the LAST user item of that list (the
//!   empty string when it has none). For a run started from a string the two
//!   carry the same text. Pinned by the golden
//!   `list_input_guardrail_on_last_user`.
//! - Input guardrails run one at a time and stop at the first tripwire. The SDK
//!   starts an agent's input guardrails together, so when one trips the others'
//!   `guardrail` spans exist too (see [`crate::run`]).
//!
//! `output_info` is whatever the check wants to report about its decision. It
//! reaches the caller inside the error and never enters a span or an event, so
//! a check can describe what it matched without that text being recorded.

use std::fmt;
use std::sync::Arc;

use futures::future::BoxFuture;
use serde_json::Value;

use crate::RunContext;

/// A check's verdict (`GuardrailFunctionOutput`).
#[derive(Clone, Debug, PartialEq)]
pub struct GuardrailOutput {
    /// True stops the run.
    pub tripwire_triggered: bool,
    /// The check's own account of why, for the caller. Not recorded anywhere.
    pub output_info: Value,
}

impl GuardrailOutput {
    pub fn pass() -> Self {
        Self {
            tripwire_triggered: false,
            output_info: Value::Null,
        }
    }

    pub fn trip(output_info: Value) -> Self {
        Self {
            tripwire_triggered: true,
            output_info,
        }
    }
}

/// What a running check knows.
#[derive(Clone)]
pub struct GuardrailContext {
    /// `RunConfig::context`, as the caller gave it.
    pub run_context: RunContext,
    /// The agent the check is guarding.
    pub agent: String,
}

impl fmt::Debug for GuardrailContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GuardrailContext")
            .field("agent", &self.agent)
            .finish_non_exhaustive()
    }
}

pub type GuardrailCheck =
    Arc<dyn Fn(GuardrailContext, String) -> BoxFuture<'static, GuardrailOutput> + Send + Sync>;

#[derive(Clone)]
pub struct InputGuardrail {
    pub name: String,
    pub check: GuardrailCheck,
}

#[derive(Clone)]
pub struct OutputGuardrail {
    pub name: String,
    pub check: GuardrailCheck,
}

impl InputGuardrail {
    pub fn new<F, Fut>(name: impl Into<String>, check: F) -> Self
    where
        F: Fn(GuardrailContext, String) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = GuardrailOutput> + Send + 'static,
    {
        Self {
            name: name.into(),
            check: boxed(check),
        }
    }
}

impl OutputGuardrail {
    pub fn new<F, Fut>(name: impl Into<String>, check: F) -> Self
    where
        F: Fn(GuardrailContext, String) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = GuardrailOutput> + Send + 'static,
    {
        Self {
            name: name.into(),
            check: boxed(check),
        }
    }
}

fn boxed<F, Fut>(check: F) -> GuardrailCheck
where
    F: Fn(GuardrailContext, String) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = GuardrailOutput> + Send + 'static,
{
    use futures::FutureExt;
    Arc::new(move |context, text| check(context, text).boxed())
}

impl fmt::Debug for InputGuardrail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InputGuardrail")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for OutputGuardrail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OutputGuardrail")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}
