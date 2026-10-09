//! The development model: a script that stands in for a language model.
//!
//! Offered as `dev:scripted` only when the service was built with development
//! mode on (`lattice --dev-model`). It is not a language model and says so in
//! its answer. It exists so that the whole path (a run, its spans, the handoff,
//! the tools, the streamed answer, the store, the window) can be exercised and
//! looked at without a model server and without a key.
//!
//! The script, one step per model call, each after a delay (350 ms by default,
//! so a run takes about a second and a half and can be watched):
//!
//! 1. `current_time` and `calculate("6 * 7")` called together (two parallel
//!    function calls);
//! 2. the handoff to the Model advisor, by the tool name the SDK built for it;
//! 3. `list_models`, called by the advisor;
//! 4. a Markdown answer saying what it is and what happened.
//!
//! It wraps [`lattice_agents::testing::ScriptedModel`] but records its
//! generation spans in full (a model name, the messages, the usage, as a real
//! model's spans are), where the SDK's own scripted model leaves them empty: a
//! run that looks like a run is the point of it.
//!
//! Invariant: it never touches the network, the disk or a key, and it makes
//! one call per step, so a task that reaches it is answered the same way each
//! time, whatever the task says.

use std::sync::Arc;
use std::time::Duration;

use futures::stream::BoxStream;
use lattice_agents::testing::{ScriptedModel, ScriptedStep, assistant_message, function_call};
use lattice_agents::{Model, ModelError, ModelEvent, ModelRequest};
use serde_json::{Value, json};

/// The name its generation spans record.
pub const DEV_MODEL_NAME: &str = "scripted-development-model";

/// The pause before each answer.
pub const DEFAULT_STEP_DELAY: Duration = Duration::from_millis(350);

const ANSWER: &str = "**This is the development model.** It is a script, not a language model, so nothing in this answer was written by an AI.\n\nWhat happened in this run:\n\n1. It asked for the time and for `6 * 7` in the same step: two tools, run together.\n2. It handed the conversation to the **Model advisor**.\n3. The advisor called `list_models`, and this answer followed.\n\nPick a real model from the list to run your own task.";

/// The scripted development model.
pub struct DevModel {
    inner: Arc<ScriptedModel>,
}

impl DevModel {
    /// A model that plays the script once, `delay` before each step.
    /// `handoff_tool` is the name of the assistant's handoff tool.
    pub fn new(handoff_tool: &str, delay: Duration) -> Self {
        let steps = vec![
            ScriptedStep::respond(vec![
                function_call("current_time", "{}", "call_dev_time"),
                function_call("calculate", r#"{"expression":"6 * 7"}"#, "call_dev_sum"),
            ])
            .with_tokens(120, 24),
            ScriptedStep::respond(vec![function_call(handoff_tool, "{}", "call_dev_handoff")])
                .with_tokens(160, 12),
            ScriptedStep::respond(vec![function_call("list_models", "{}", "call_dev_models")])
                .with_tokens(90, 10),
            ScriptedStep::respond(vec![assistant_message(ANSWER)]).with_tokens(300, 90),
        ];
        Self {
            inner: Arc::new(ScriptedModel::new(steps).with_delay(delay)),
        }
    }

    /// How many times the model has been called.
    pub fn call_count(&self) -> usize {
        self.inner.calls().len()
    }

    /// The scripted answer's text.
    pub fn answer() -> &'static str {
        ANSWER
    }
}

impl Model for DevModel {
    fn name(&self) -> &str {
        DEV_MODEL_NAME
    }

    fn config_for_trace(&self) -> Value {
        json!({"scripted": true})
    }

    fn stream(&self, request: ModelRequest) -> BoxStream<'static, Result<ModelEvent, ModelError>> {
        self.inner.stream(request)
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use lattice_agents::{InputItem, ModelSettings, OutputItem};

    use super::*;

    fn request() -> ModelRequest {
        ModelRequest {
            system: String::new(),
            input: vec![InputItem::User("hi".into())],
            tools: Vec::new(),
            settings: ModelSettings::default(),
        }
    }

    async fn output(model: &DevModel) -> Vec<OutputItem> {
        let mut stream = model.stream(request());
        while let Some(event) = stream.next().await {
            if let Ok(ModelEvent::Done(response)) = event {
                return response.output;
            }
        }
        panic!("no final response");
    }

    #[tokio::test]
    async fn the_script_calls_the_tools_then_hands_off_then_lists_then_answers() {
        let model = DevModel::new("transfer_to_model_advisor", Duration::ZERO);
        let first = output(&model).await;
        assert_eq!(first.len(), 2);
        assert!(
            matches!(&first[0], OutputItem::FunctionCall { name, .. } if name == "current_time")
        );
        assert!(
            matches!(&first[1], OutputItem::FunctionCall { name, arguments, .. } if name == "calculate" && arguments.contains("6 * 7"))
        );
        let second = output(&model).await;
        assert!(
            matches!(&second[..], [OutputItem::FunctionCall { name, .. }] if name == "transfer_to_model_advisor")
        );
        let third = output(&model).await;
        assert!(
            matches!(&third[..], [OutputItem::FunctionCall { name, .. }] if name == "list_models")
        );
        let fourth = output(&model).await;
        match &fourth[..] {
            [OutputItem::Message { text }] => {
                assert!(
                    text.contains("development model") && text.contains("not a language model")
                );
                assert_eq!(text, DevModel::answer());
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(model.call_count(), 4);
    }

    #[test]
    fn it_records_generation_spans_in_full_under_its_own_name() {
        let model = DevModel::new("t", Duration::ZERO);
        assert_eq!(model.name(), "scripted-development-model");
        assert_eq!(
            model.generation_trace(),
            lattice_agents::GenerationTrace::Full
        );
        assert_eq!(model.config_for_trace(), json!({"scripted": true}));
    }
}
