//! A scripted model, for tests and for the development model.
//!
//! Ports `agents.testing.ScriptedModel` and its `assistant_message` /
//! `function_call` builders (0.22.3). Each call to the model consumes one
//! scripted step: a list of output items, or an error. The model records every
//! request it receives.
//!
//! It behaves like the SDK's scripted model in the three ways a conformance test
//! can see:
//! - its generation spans are EMPTY ([`GenerationTrace::Bare`]): the SDK's
//!   scripted model opens `generation_span(disabled=False)` and fills nothing
//!   in, so no model name, input, output or usage is recorded;
//! - a step with no usage reports `Usage(requests=1)` (one request, no tokens),
//!   which is the SDK's default for a scripted step;
//! - calling it with no steps left is an error
//!   (`UnexpectedModelCall`), never a silent empty answer.
//!
//! It streams an answer's text in small chunks so that live-text handling is
//! exercised; the SDK's scripted model emits one delta per message, and the
//! number of chunks is not something a conformance test compares.
//!
//! [`ScriptedModel::with_delay`] makes every call wait before it answers, which
//! is how the cancellation tests hold a model call open.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use futures::stream::{self, BoxStream};
use lattice_protocol::Usage;
use serde_json::Value;

use crate::model::{
    GenerationTrace, Model, ModelError, ModelEvent, ModelRequest, ModelResponse, OutputItem,
};

/// The characters of an answer sent in one `TextDelta`.
const CHUNK_CHARS: usize = 8;

/// One scripted model call.
#[derive(Clone, Debug, PartialEq)]
pub struct ScriptedStep {
    outcome: Result<Vec<OutputItem>, ModelError>,
    usage: Option<Usage>,
}

impl ScriptedStep {
    pub fn respond(output: Vec<OutputItem>) -> Self {
        Self {
            outcome: Ok(output),
            usage: None,
        }
    }

    pub fn error(error: ModelError) -> Self {
        Self {
            outcome: Err(error),
            usage: None,
        }
    }

    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// One request with these token counts.
    pub fn with_tokens(self, input_tokens: u64, output_tokens: u64) -> Self {
        self.with_usage(Usage {
            requests: 1,
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
        })
    }
}

impl From<Vec<OutputItem>> for ScriptedStep {
    fn from(output: Vec<OutputItem>) -> Self {
        Self::respond(output)
    }
}

impl From<ModelError> for ScriptedStep {
    fn from(error: ModelError) -> Self {
        Self::error(error)
    }
}

#[derive(Default)]
struct State {
    steps: VecDeque<ScriptedStep>,
    calls: Vec<ModelRequest>,
}

pub struct ScriptedModel {
    state: Arc<Mutex<State>>,
    delay: Option<Duration>,
}

impl ScriptedModel {
    pub fn new<S: Into<ScriptedStep>>(steps: impl IntoIterator<Item = S>) -> Self {
        let steps = steps.into_iter().map(Into::into).collect();
        Self {
            state: Arc::new(Mutex::new(State {
                steps,
                calls: Vec::new(),
            })),
            delay: None,
        }
    }

    /// Every call waits `delay` before it answers.
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = Some(delay);
        self
    }

    /// Append a step (`ScriptedModel.enqueue`).
    pub fn enqueue(&self, step: impl Into<ScriptedStep>) {
        self.lock().steps.push_back(step.into());
    }

    /// Every request the model has received, oldest first.
    pub fn calls(&self) -> Vec<ModelRequest> {
        self.lock().calls.clone()
    }

    /// Steps not yet consumed.
    pub fn remaining_steps(&self) -> usize {
        self.lock().steps.len()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Model for ScriptedModel {
    fn name(&self) -> &str {
        "scripted-model"
    }

    fn config_for_trace(&self) -> Value {
        Value::Null
    }

    fn generation_trace(&self) -> GenerationTrace {
        GenerationTrace::Bare
    }

    fn stream(&self, request: ModelRequest) -> BoxStream<'static, Result<ModelEvent, ModelError>> {
        let call_number = {
            let mut state = self.lock();
            state.calls.push(request);
            state.calls.len()
        };
        let state = self.state.clone();
        let delay = self.delay;
        stream::once(async move {
            if let Some(delay) = delay {
                tokio::time::sleep(delay).await;
            }
            let step = state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .steps
                .pop_front();
            events_for(step, call_number)
        })
        .flat_map(stream::iter)
        .boxed()
    }
}

fn events_for(
    step: Option<ScriptedStep>,
    call_number: usize,
) -> Vec<Result<ModelEvent, ModelError>> {
    let Some(step) = step else {
        return vec![Err(ModelError::Failed(format!(
            "Unexpected streaming model call #{call_number}: no scripted steps remain."
        )))];
    };
    let output = match step.outcome {
        Ok(output) => output,
        Err(error) => return vec![Err(error)],
    };
    let mut events = Vec::new();
    for item in &output {
        match item {
            OutputItem::Message { text } => {
                let chars: Vec<char> = text.chars().collect();
                for chunk in chars.chunks(CHUNK_CHARS) {
                    events.push(Ok(ModelEvent::TextDelta(chunk.iter().collect())));
                }
            }
            OutputItem::Reasoning { text } if !text.is_empty() => {
                events.push(Ok(ModelEvent::ReasoningDelta(text.clone())));
            }
            OutputItem::Reasoning { .. }
            | OutputItem::FunctionCall { .. }
            | OutputItem::Refusal { .. } => {}
        }
    }
    let usage = step.usage.unwrap_or(Usage {
        requests: 1,
        ..Usage::default()
    });
    events.push(Ok(ModelEvent::Done(ModelResponse {
        output,
        usage: Some(usage),
    })));
    events
}

/// A plain assistant answer (`assistant_message`).
pub fn assistant_message(text: impl Into<String>) -> OutputItem {
    OutputItem::Message { text: text.into() }
}

/// A function call the model makes (`function_call`). `arguments` is the JSON
/// text the model wrote.
pub fn function_call(
    name: impl Into<String>,
    arguments: impl Into<String>,
    call_id: impl Into<String>,
) -> OutputItem {
    OutputItem::FunctionCall {
        call_id: call_id.into(),
        name: name.into(),
        arguments: arguments.into(),
    }
}

/// [`function_call`] with the arguments as JSON, written compactly the way the
/// SDK's `function_call(..., {...})` does (`separators=(",", ":")`).
pub fn function_call_json(
    name: impl Into<String>,
    arguments: &Value,
    call_id: impl Into<String>,
) -> OutputItem {
    function_call(name, arguments.to_string(), call_id)
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;
    use crate::model::ModelSettings;

    fn request(text: &str) -> ModelRequest {
        ModelRequest {
            system: String::new(),
            input: vec![crate::model::InputItem::User(text.into())],
            tools: vec![],
            settings: ModelSettings::default(),
        }
    }

    async fn collect(model: &ScriptedModel, text: &str) -> Vec<Result<ModelEvent, ModelError>> {
        model.stream(request(text)).collect().await
    }

    #[tokio::test]
    async fn an_answer_streams_in_chunks_then_completes_with_default_usage() {
        let model = ScriptedModel::new([vec![assistant_message("Hello there, friend.")]]);
        let events = collect(&model, "hi").await;
        let deltas: Vec<String> = events
            .iter()
            .filter_map(|event| match event {
                Ok(ModelEvent::TextDelta(text)) => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(deltas.concat(), "Hello there, friend.");
        assert!(deltas.len() > 1 && deltas.iter().all(|d| d.chars().count() <= CHUNK_CHARS));
        match events.last().unwrap() {
            Ok(ModelEvent::Done(response)) => {
                assert_eq!(
                    response.usage,
                    Some(Usage {
                        requests: 1,
                        ..Usage::default()
                    })
                );
                assert_eq!(
                    response.output,
                    vec![assistant_message("Hello there, friend.")]
                );
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chunks_never_split_a_character() {
        let model = ScriptedModel::new([vec![assistant_message(
            "h\u{e9}llo \u{1F600} w\u{f6}rld \u{1F600}\u{1F600}!",
        )]]);
        let events = collect(&model, "hi").await;
        let text: String = events
            .iter()
            .filter_map(|event| match event {
                Ok(ModelEvent::TextDelta(text)) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "h\u{e9}llo \u{1F600} w\u{f6}rld \u{1F600}\u{1F600}!");
    }

    #[tokio::test]
    async fn it_records_calls_and_consumes_steps_in_order() {
        let model = ScriptedModel::new([
            ScriptedStep::respond(vec![assistant_message("one")]).with_tokens(10, 2),
            ScriptedStep::respond(vec![assistant_message("two")]),
        ]);
        assert_eq!(model.remaining_steps(), 2);
        let first = collect(&model, "a").await;
        match first.last().unwrap() {
            Ok(ModelEvent::Done(response)) => assert_eq!(response.usage.unwrap().total_tokens, 12),
            other => panic!("{other:?}"),
        }
        let _ = collect(&model, "b").await;
        assert_eq!(model.remaining_steps(), 0);
        let calls = model.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            calls[1].input,
            vec![crate::model::InputItem::User("b".into())]
        );
    }

    #[tokio::test]
    async fn an_error_step_and_an_empty_script_are_errors() {
        let model = ScriptedModel::new([ScriptedStep::error(ModelError::Status(500))]);
        assert_eq!(
            collect(&model, "a").await,
            vec![Err(ModelError::Status(500))]
        );
        let events = collect(&model, "b").await;
        match &events[..] {
            [Err(ModelError::Failed(message))] => assert!(message.contains("call #2"), "{message}"),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn the_stream_is_lazy_and_a_dropped_stream_consumes_nothing() {
        let model = ScriptedModel::new([vec![assistant_message("never read")]]);
        drop(model.stream(request("a")));
        assert_eq!(model.remaining_steps(), 1);
        assert_eq!(model.calls().len(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn a_delayed_model_waits_before_answering() {
        let model = ScriptedModel::new([vec![assistant_message("late")]])
            .with_delay(Duration::from_secs(5));
        let mut stream = model.stream(request("a"));
        let started = tokio::time::Instant::now();
        let first = stream.next().await;
        assert!(first.is_some());
        assert!(started.elapsed() >= Duration::from_secs(5));
    }

    #[test]
    fn builders_make_the_items_the_sdk_makes() {
        assert_eq!(
            function_call("echo", "{\"text\":\"hi\"}", "call_1"),
            OutputItem::FunctionCall {
                call_id: "call_1".into(),
                name: "echo".into(),
                arguments: "{\"text\":\"hi\"}".into()
            }
        );
        assert_eq!(
            function_call_json("echo", &serde_json::json!({"text": "hi"}), "call_1"),
            function_call("echo", "{\"text\":\"hi\"}", "call_1")
        );
    }
}
