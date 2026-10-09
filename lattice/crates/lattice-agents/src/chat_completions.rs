//! An OpenAI-compatible `/chat/completions` model client: the only real model.
//!
//! Ports the request side of `agents.models.openai_chatcompletions` and
//! `chatcmpl_converter` (`items_to_messages`, `tool_to_openai`,
//! `convert_handoff_tool`, `convert_tool_choice`) and the response side of
//! `chatcmpl_stream_handler` (0.22.3), over `reqwest` instead of the `openai`
//! package, for these item kinds: user text, assistant text, function calls and
//! their results.
//!
//! What this client refuses to do, because it is the one place a task, its tool
//! results and (optionally) a credential leave the process:
//! - It never follows a redirect (`Policy::none()`): a `302` would otherwise
//!   carry the `Authorization` header to a host the caller did not choose. A
//!   redirect is an error, [`ModelError::Status`], like any other non-2xx.
//! - For a `local` model it uses no proxy at all, so an environment or system
//!   proxy cannot route "local" traffic off the machine.
//! - The API key is a [`SecretString`], sent only as an `Authorization: Bearer`
//!   header marked sensitive (so HTTP debug output redacts it), and only when a
//!   key is set. A base URL that embeds credentials (`http://user:pw@host`) is
//!   refused, because the HTTP client would turn them into a `Basic` header.
//! - Errors never carry the response body, the URL, the key, or the transport's
//!   own message (which names the URL): non-2xx is `Status(code)`, connect and
//!   read failures are `Connection` with a fixed phrase, silence is `Timeout`,
//!   and an unreadable chunk is `Protocol` with a fixed reason.
//! - What a server may stream is bounded: `MAX_RESPONSE_BYTES` for a whole
//!   response (text, reasoning, refusal, and every tool call's id, name and
//!   arguments, plus a fixed [`TOOL_CALL_ENTRY_COST`] for each tool-call entry,
//!   so a burst of empty entries is paid for too), `MAX_TOOL_CALLS` distinct tool
//!   calls, `MAX_EVENT_BYTES` for one event.
//!
//! The wire format:
//! - Request: `model`, `messages` (the system message first, then the items
//!   converted as the SDK does: an assistant's text and tool calls are ONE
//!   assistant message with `tool_calls`, each tool result a `tool` message),
//!   `tools` only when there are any, `tool_choice` / `parallel_tool_calls` only
//!   when there are tools and the setting is set, `stream: true`,
//!   `stream_options: {"include_usage": true}` unless usage is switched off
//!   (local servers report none otherwise), and the sampling fields when set.
//! - Response: server-sent events, read as the SDK reads them
//!   (`chatcmpl_stream_handler.py`). Of the `choices` of a chunk only the one
//!   with `index` 0 is read (the first, when the server numbers none).
//!   `delta.content` is text, `delta.reasoning_content` or `delta.reasoning` is
//!   reasoning, `delta.refusal` is a refusal, tool calls are assembled by `index`
//!   (`function.arguments` pieces are concatenated; a non-empty `function.name`
//!   REPLACES the name and a non-empty `id` the id, the latest of each winning,
//!   as in the SDK), and the last chunk's `usage` is the response's usage.
//! - A call the server gave no `id` keeps the empty id; the runner refuses it
//!   with the SDK's words rather than this client inventing one.
//! - A `content_filter` finish with no text, calls or refusal is reported as the
//!   refusal "Response withheld by the provider's content filter."; a `length`
//!   finish with no text, calls or refusal is [`ModelError::Truncated`] (a
//!   reasoning model that spent its whole budget thinking).
//!
//! Deviations, and why:
//! - The SDK returns a response's function calls before its message when the
//!   first tool-call delta arrived before the first text delta. This client
//!   always returns reasoning, then the message, then the calls by index. The
//!   conversion for the next request merges them into one assistant message
//!   either way; only the order of the run-item events differs, in a case real
//!   servers rarely produce.
//! - A stream that ends without `[DONE]` and without a `finish_reason` is a
//!   `Protocol` error, not a shorter answer: a dropped connection must not turn
//!   into a final output.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::time::Duration;

use futures::StreamExt;
use futures::stream::{self, BoxStream};
use lattice_protocol::Usage;
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderValue};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::model::{
    InputItem, Model, ModelError, ModelEvent, ModelRequest, ModelResponse, ModelSettings,
    OutputItem, ToolCallItem, ToolSpec,
};
use crate::secret::SecretString;

/// The most a server may stream for one response (text, reasoning, refusal and
/// tool calls together), and the longest single event, before the client gives
/// up. A model that legitimately exceeds it is indistinguishable from a broken
/// or hostile server.
pub const MAX_RESPONSE_BYTES: usize = 32 * 1024 * 1024;

/// What each tool-call entry of a chunk costs toward [`MAX_RESPONSE_BYTES`], on
/// top of the bytes of its id, name and arguments: an entry with nothing in it
/// still takes a slot and a parse, and a stream of them must not go on for free.
pub const TOOL_CALL_ENTRY_COST: usize = 64;

/// The most distinct tool calls (by `index`) one response may hold. A real model
/// makes a handful.
pub const MAX_TOOL_CALLS: usize = 128;

/// What a provider that filtered a response away says, in the SDK's words
/// (`chatcmpl_stream_handler.py:1133-1180`).
pub const CONTENT_FILTER_REFUSAL: &str = "Response withheld by the provider's content filter.";

/// The most one server-sent event (its lines, gathered before its blank line)
/// may hold: far more than any real delta, small enough that a server that
/// never ends a line cannot make the client buffer without limit.
const MAX_EVENT_BYTES: usize = 8 * 1024 * 1024;

const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(120);

/// How to reach one chat model.
#[derive(Clone)]
pub struct ChatCompletionsConfig {
    /// The API root, already in `/v1` form (`http://127.0.0.1:11434/v1`); the
    /// client appends `/chat/completions`.
    pub base_url: String,
    pub model: String,
    pub api_key: Option<SecretString>,
    /// The endpoint is on this machine: use no proxy.
    pub local: bool,
    /// Longest wait to establish a connection (default 10 s).
    pub connect_timeout: Duration,
    /// Longest silence between two reads (default 120 s).
    pub read_timeout: Duration,
}

impl ChatCompletionsConfig {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            api_key: None,
            local: false,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            read_timeout: DEFAULT_READ_TIMEOUT,
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<SecretString>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn local(mut self, local: bool) -> Self {
        self.local = local;
        self
    }

    pub fn with_timeouts(mut self, connect: Duration, read: Duration) -> Self {
        self.connect_timeout = connect;
        self.read_timeout = read;
        self
    }
}

impl fmt::Debug for ChatCompletionsConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatCompletionsConfig")
            .field("base_url", &sanitize_url_for_trace(&self.base_url))
            .field("model", &self.model)
            .field("api_key", &self.api_key)
            .field("local", &self.local)
            .field("connect_timeout", &self.connect_timeout)
            .field("read_timeout", &self.read_timeout)
            .finish()
    }
}

pub struct ChatCompletionsModel {
    client: reqwest::Client,
    url: String,
    model: String,
    api_key: Option<SecretString>,
    base_url_for_trace: String,
}

impl fmt::Debug for ChatCompletionsModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatCompletionsModel")
            .field("base_url", &self.base_url_for_trace)
            .field("model", &self.model)
            .field("api_key", &self.api_key)
            .finish_non_exhaustive()
    }
}

impl ChatCompletionsModel {
    pub fn new(config: ChatCompletionsConfig) -> Result<Self, ModelError> {
        let parsed = reqwest::Url::parse(&config.base_url)
            .map_err(|_| ModelError::Failed("the model base URL is not a valid URL".into()))?;
        if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
            return Err(ModelError::Failed(
                "the model base URL must be an http or https URL".into(),
            ));
        }
        if !parsed.username().is_empty() || parsed.password().is_some() {
            return Err(ModelError::Failed(
                "the model base URL must not contain credentials".into(),
            ));
        }

        let mut builder = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(config.connect_timeout)
            .read_timeout(config.read_timeout)
            .user_agent(concat!("Lattice/", env!("CARGO_PKG_VERSION")));
        if config.local {
            builder = builder.no_proxy();
        }
        let client = builder.build().map_err(|_| {
            ModelError::Failed("could not set up the HTTP client for the model".into())
        })?;

        Ok(Self {
            client,
            url: format!("{}/chat/completions", config.base_url.trim_end_matches('/')),
            model: config.model,
            api_key: config.api_key,
            base_url_for_trace: sanitize_url_for_trace(&config.base_url),
        })
    }
}

impl Model for ChatCompletionsModel {
    fn name(&self) -> &str {
        &self.model
    }

    fn config_for_trace(&self) -> Value {
        json!({"base_url": self.base_url_for_trace})
    }

    fn stream(&self, request: ModelRequest) -> BoxStream<'static, Result<ModelEvent, ModelError>> {
        let body = chat_request_body(&self.model, &request);
        let start = Start {
            client: self.client.clone(),
            url: self.url.clone(),
            api_key: self.api_key.clone(),
            body,
        };
        stream::unfold(State::Start(Box::new(start)), step).boxed()
    }
}

/// `sanitize_url_for_trace`: a URL with any credentials, query and fragment
/// removed, or the empty string when it cannot be parsed. What any message that
/// names the model's address should print.
pub fn sanitize_url_for_trace(url: &str) -> String {
    let Ok(parsed) = reqwest::Url::parse(url) else {
        return String::new();
    };
    let Some(host) = parsed.host_str() else {
        return String::new();
    };
    let port = parsed.port().map(|p| format!(":{p}")).unwrap_or_default();
    // Python's `urlsplit` keeps an absent path absent; `Url` reports it as "/".
    let without_suffix = url.split(['?', '#']).next().unwrap_or(url);
    let path = if parsed.path() == "/" && !without_suffix.ends_with('/') {
        ""
    } else {
        parsed.path()
    };
    format!("{}://{host}{port}{path}", parsed.scheme())
}

// ---------------------------------------------------------------------------
// The request
// ---------------------------------------------------------------------------

/// A tool as the wire carries it (`tool_to_openai` / `convert_handoff_tool`).
fn tool_to_openai(tool: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": tool.strict,
        }
    })
}

/// `Converter.convert_tool_choice`: the three modes are strings; anything else
/// names one function.
fn convert_tool_choice(choice: &str) -> Value {
    match choice {
        "auto" | "required" | "none" => json!(choice),
        name => json!({"type": "function", "function": {"name": name}}),
    }
}

fn tool_call_to_openai(call: &ToolCallItem) -> Value {
    let arguments = if call.arguments.is_empty() {
        "{}"
    } else {
        call.arguments.as_str()
    };
    json!({
        "id": call.call_id,
        "type": "function",
        "function": {"name": call.name, "arguments": arguments},
    })
}

#[derive(Default)]
struct PendingAssistant {
    content: Option<String>,
    tool_calls: Vec<Value>,
}

impl PendingAssistant {
    fn into_message(self) -> Value {
        let mut message = Map::new();
        message.insert("role".into(), json!("assistant"));
        message.insert(
            "content".into(),
            self.content.map_or(Value::Null, Value::String),
        );
        if !self.tool_calls.is_empty() {
            message.insert("tool_calls".into(), Value::Array(self.tool_calls));
        }
        Value::Object(message)
    }
}

fn flush(pending: &mut Option<PendingAssistant>, out: &mut Vec<Value>) {
    if let Some(assistant) = pending.take() {
        out.push(assistant.into_message());
    }
}

/// `Converter.items_to_messages`, with the system message first.
///
/// An assistant's text and its tool calls are one assistant message: tool calls
/// attach to the assistant message in progress, and an assistant text joins the
/// one in progress when that one already has tool calls and no text (a turn
/// whose calls came before its text). A user message or a tool result ends the
/// assistant message in progress.
pub fn chat_messages(system: &str, input: &[InputItem]) -> Vec<Value> {
    messages(system, input, false)
}

/// [`chat_messages`] for a trace: an image's bytes are replaced by
/// `[image: <media type>, <n> bytes]`, so a span never carries a screenshot.
pub fn chat_messages_for_trace(system: &str, input: &[InputItem]) -> Vec<Value> {
    messages(system, input, true)
}

fn image_part(image: &crate::model::Image, for_trace: bool) -> Value {
    let url = if for_trace {
        format!(
            "[image: {}, {} bytes]",
            image.media_type,
            image.base64.len() / 4 * 3
        )
    } else {
        format!("data:{};base64,{}", image.media_type, image.base64)
    };
    json!({"type": "image_url", "image_url": {"url": url}})
}

fn messages(system: &str, input: &[InputItem], for_trace: bool) -> Vec<Value> {
    let mut out = Vec::with_capacity(input.len() + 1);
    if !system.is_empty() {
        out.push(json!({"role": "system", "content": system}));
    }
    let mut pending: Option<PendingAssistant> = None;
    for item in input {
        match item {
            InputItem::User(text) => {
                flush(&mut pending, &mut out);
                out.push(json!({"role": "user", "content": text}));
            }
            // The SDK's `input_image` parts, as its Converter sends them: one
            // user message whose content is the text, then each image.
            InputItem::UserImages { text, images } => {
                flush(&mut pending, &mut out);
                let mut content = vec![json!({"type": "text", "text": text})];
                content.extend(images.iter().map(|image| image_part(image, for_trace)));
                out.push(json!({"role": "user", "content": content}));
            }
            InputItem::ToolResult { call_id, output } => {
                flush(&mut pending, &mut out);
                out.push(json!({"role": "tool", "tool_call_id": call_id, "content": output}));
            }
            InputItem::Assistant { text, tool_calls } => {
                if let Some(text) = text {
                    let joins_pending = matches!(&pending, Some(p) if !p.tool_calls.is_empty() && p.content.is_none());
                    if joins_pending {
                        if let Some(p) = pending.as_mut() {
                            p.content = Some(text.clone());
                        }
                    } else {
                        flush(&mut pending, &mut out);
                        pending = Some(PendingAssistant {
                            content: Some(text.clone()),
                            tool_calls: Vec::new(),
                        });
                    }
                }
                for call in tool_calls {
                    pending
                        .get_or_insert_with(PendingAssistant::default)
                        .tool_calls
                        .push(tool_call_to_openai(call));
                }
            }
        }
    }
    flush(&mut pending, &mut out);
    out
}

/// The JSON body of one streaming chat completion request.
pub fn chat_request_body(model: &str, request: &ModelRequest) -> Value {
    let settings: &ModelSettings = &request.settings;
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert(
        "messages".into(),
        Value::Array(chat_messages(&request.system, &request.input)),
    );
    if !request.tools.is_empty() {
        body.insert(
            "tools".into(),
            Value::Array(request.tools.iter().map(tool_to_openai).collect()),
        );
        if let Some(choice) = &settings.tool_choice {
            body.insert("tool_choice".into(), convert_tool_choice(choice));
        }
        if let Some(parallel) = settings.parallel_tool_calls {
            body.insert("parallel_tool_calls".into(), json!(parallel));
        }
    }
    body.insert("stream".into(), json!(true));
    if settings.include_usage.unwrap_or(true) {
        body.insert("stream_options".into(), json!({"include_usage": true}));
    }
    if let Some(temperature) = settings.temperature {
        body.insert("temperature".into(), json!(temperature));
    }
    if let Some(top_p) = settings.top_p {
        body.insert("top_p".into(), json!(top_p));
    }
    if let Some(max_tokens) = settings.max_tokens {
        body.insert("max_tokens".into(), json!(max_tokens));
    }
    Value::Object(body)
}

// ---------------------------------------------------------------------------
// The response
// ---------------------------------------------------------------------------

/// Server-sent events: `data:` lines gathered until a blank line.
#[derive(Default)]
struct SseDecoder {
    buffer: Vec<u8>,
    /// The first `searched` bytes of `buffer` are known to hold no newline, so a
    /// long line arriving in small pieces is scanned once, not once per piece.
    searched: usize,
    data: Vec<String>,
}

impl SseDecoder {
    /// Feed bytes; returns the `data` of every event completed by them.
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>, ModelError> {
        self.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        let mut consumed = 0;
        loop {
            let from = consumed.max(self.searched);
            let Some(offset) = self.buffer[from..].iter().position(|b| *b == b'\n') else {
                break;
            };
            let end = from + offset;
            let mut line = &self.buffer[consumed..end];
            if line.last() == Some(&b'\r') {
                line = &line[..line.len() - 1];
            }
            let line = String::from_utf8_lossy(line).into_owned();
            consumed = end + 1;
            if line.is_empty() {
                if !self.data.is_empty() {
                    events.push(self.data.join("\n"));
                    self.data.clear();
                }
            } else if let Some((field, value)) = line.split_once(':') {
                if field == "data" {
                    self.data
                        .push(value.strip_prefix(' ').unwrap_or(value).to_owned());
                }
            } else if line == "data" {
                self.data.push(String::new());
            }
        }
        self.buffer.drain(..consumed);
        self.searched = self.buffer.len();
        let pending: usize = self.data.iter().map(String::len).sum::<usize>() + self.buffer.len();
        if pending > MAX_EVENT_BYTES {
            return Err(ModelError::Protocol("a stream event is too large".into()));
        }
        Ok(events)
    }

    /// The stream ended: an event whose blank line never came still counts.
    fn finish(&mut self) -> Option<String> {
        if !self.buffer.is_empty() {
            let line = String::from_utf8_lossy(&self.buffer).into_owned();
            self.buffer.clear();
            if let Some((field, value)) = line.trim_end_matches('\r').split_once(':')
                && field == "data"
            {
                self.data
                    .push(value.strip_prefix(' ').unwrap_or(value).to_owned());
            }
        }
        if self.data.is_empty() {
            None
        } else {
            let joined = self.data.join("\n");
            self.data.clear();
            Some(joined)
        }
    }
}

#[derive(Deserialize)]
struct Chunk {
    #[serde(default)]
    choices: Option<Vec<Choice>>,
    #[serde(default)]
    usage: Option<UsageChunk>,
    #[serde(default)]
    error: Option<Value>,
}

#[derive(Deserialize)]
struct Choice {
    /// Which choice this is. A server that asks for one completion still numbers
    /// it 0; some number none.
    #[serde(default)]
    index: Option<i64>,
    #[serde(default)]
    delta: Option<Delta>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    /// Set by the OpenAI API when the model declines to answer.
    #[serde(default)]
    refusal: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

/// The one choice of a chunk that is read (`chatcmpl_stream_handler.py:682`):
/// the entry with `index` 0; when no entry carries an index at all, the first.
fn choice_zero(choices: Option<Vec<Choice>>) -> Option<Choice> {
    let choices = choices?;
    if choices.iter().any(|choice| choice.index.is_some()) {
        choices.into_iter().find(|choice| choice.index == Some(0))
    } else {
        choices.into_iter().next()
    }
}

#[derive(Deserialize)]
struct ToolCallDelta {
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<FunctionDelta>,
}

#[derive(Deserialize)]
struct FunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Deserialize)]
struct UsageChunk {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
    #[serde(default)]
    total_tokens: Option<u64>,
}

#[derive(Default)]
struct PartialCall {
    id: String,
    name: String,
    arguments: String,
}

/// Turns chunks into events and, at the end, the complete response.
#[derive(Default)]
struct Assembler {
    text: String,
    reasoning: String,
    refusal: String,
    calls: BTreeMap<usize, PartialCall>,
    usage: Option<Usage>,
    finished: bool,
    saw_content_filter: bool,
    saw_length: bool,
    bytes: usize,
}

impl Assembler {
    fn account(&mut self, length: usize) -> Result<(), ModelError> {
        self.bytes = self.bytes.saturating_add(length);
        if self.bytes > MAX_RESPONSE_BYTES {
            return Err(ModelError::Protocol("the response is too large".into()));
        }
        Ok(())
    }

    fn push(&mut self, data: &str) -> Result<Vec<ModelEvent>, ModelError> {
        let chunk: Chunk = serde_json::from_str(data).map_err(|_| {
            ModelError::Protocol("a stream chunk is not valid JSON of the expected shape".into())
        })?;
        if chunk.error.is_some() {
            return Err(ModelError::Protocol(
                "the model server reported an error in the stream".into(),
            ));
        }
        if let Some(usage) = chunk.usage {
            let input_tokens = usage.prompt_tokens.unwrap_or(0);
            let output_tokens = usage.completion_tokens.unwrap_or(0);
            self.usage = Some(Usage {
                requests: 1,
                input_tokens,
                output_tokens,
                total_tokens: usage
                    .total_tokens
                    .unwrap_or(input_tokens.saturating_add(output_tokens)),
            });
        }
        let mut events = Vec::new();
        let Some(choice) = choice_zero(chunk.choices) else {
            return Ok(events);
        };
        match choice.finish_reason.as_deref() {
            Some("content_filter") => self.saw_content_filter = true,
            Some("length") => self.saw_length = true,
            _ => {}
        }
        if choice.finish_reason.is_some() {
            self.finished = true;
        }
        let Some(delta) = choice.delta else {
            return Ok(events);
        };
        if let Some(content) = delta.content.filter(|c| !c.is_empty()) {
            self.account(content.len())?;
            self.text.push_str(&content);
            events.push(ModelEvent::TextDelta(content));
        }
        if let Some(reasoning) = delta
            .reasoning_content
            .or(delta.reasoning)
            .filter(|r| !r.is_empty())
        {
            self.account(reasoning.len())?;
            self.reasoning.push_str(&reasoning);
            events.push(ModelEvent::ReasoningDelta(reasoning));
        }
        if let Some(refusal) = delta.refusal.filter(|r| !r.is_empty()) {
            self.account(refusal.len())?;
            self.refusal.push_str(&refusal);
        }
        for (position, call) in delta.tool_calls.unwrap_or_default().into_iter().enumerate() {
            let index = call.index.unwrap_or(position);
            if !self.calls.contains_key(&index) && self.calls.len() >= MAX_TOOL_CALLS {
                return Err(ModelError::Protocol(
                    "the response has too many tool calls".into(),
                ));
            }
            // Everything the entry carries is paid for, and the entry itself.
            let function = call.function.as_ref();
            self.account(
                TOOL_CALL_ENTRY_COST
                    + call.id.as_ref().map_or(0, String::len)
                    + function
                        .and_then(|f| f.name.as_ref())
                        .map_or(0, String::len)
                    + function
                        .and_then(|f| f.arguments.as_ref())
                        .map_or(0, String::len),
            )?;
            let partial = self.calls.entry(index).or_default();
            // The latest non-empty id wins (`chatcmpl_stream_handler.py:1037-1041`).
            if let Some(id) = call.id.filter(|id| !id.is_empty()) {
                partial.id = id;
            }
            if let Some(function) = call.function {
                // A non-empty name REPLACES the name: the SDK says it is "correct
                // from the first function call chunk" (`:1033-1035`), and a server
                // that repeats it in every delta must not make `calculatecalculate`.
                if let Some(name) = function.name.filter(|name| !name.is_empty()) {
                    partial.name = name;
                }
                if let Some(arguments) = function.arguments {
                    partial.arguments.push_str(&arguments);
                }
            }
        }
        Ok(events)
    }

    /// The complete response, or why there is none.
    fn into_response(mut self) -> Result<ModelResponse, ModelError> {
        let nothing_made = |assembler: &Self| {
            assembler.text.is_empty() && assembler.refusal.is_empty() && assembler.calls.is_empty()
        };
        // A provider that filtered the whole answer away: a refusal, not silence.
        if self.saw_content_filter && nothing_made(&self) {
            self.refusal = CONTENT_FILTER_REFUSAL.to_owned();
        }
        // A budget that ran out before any answer is not an empty answer.
        if self.saw_length && nothing_made(&self) {
            return Err(ModelError::Truncated);
        }
        let mut output = Vec::new();
        if !self.reasoning.is_empty() {
            output.push(OutputItem::Reasoning {
                text: self.reasoning,
            });
        }
        if !self.text.is_empty() {
            output.push(OutputItem::Message { text: self.text });
        }
        if !self.refusal.is_empty() {
            output.push(OutputItem::Refusal { text: self.refusal });
        }
        for call in self.calls.into_values() {
            if call.name.is_empty() && call.arguments.is_empty() {
                continue;
            }
            // No id stays no id: the runner refuses a call without one.
            output.push(OutputItem::FunctionCall {
                call_id: call.id,
                name: call.name,
                arguments: call.arguments,
            });
        }
        Ok(ModelResponse {
            output,
            usage: self.usage,
        })
    }
}

// ---------------------------------------------------------------------------
// The stream
// ---------------------------------------------------------------------------

struct Start {
    client: reqwest::Client,
    url: String,
    api_key: Option<SecretString>,
    body: Value,
}

struct Reading {
    response: reqwest::Response,
    decoder: SseDecoder,
    assembler: Assembler,
    pending: VecDeque<Result<ModelEvent, ModelError>>,
    finished: bool,
}

enum State {
    Start(Box<Start>),
    Reading(Box<Reading>),
    Finished,
}

fn map_transport_error(error: &reqwest::Error) -> ModelError {
    if error.is_timeout() {
        ModelError::Timeout
    } else if error.is_connect() {
        ModelError::Connection("could not connect to the model server".into())
    } else if error.is_redirect() {
        ModelError::Connection("the model server redirected the request".into())
    } else {
        ModelError::Connection("the connection to the model server failed".into())
    }
}

async fn send(start: Start) -> Result<reqwest::Response, ModelError> {
    let body = serde_json::to_vec(&start.body)
        .map_err(|_| ModelError::Failed("could not encode the model request".into()))?;
    let mut request = start
        .client
        .post(&start.url)
        .header(ACCEPT, "text/event-stream")
        .header(CONTENT_TYPE, "application/json")
        .body(body);
    if let Some(key) = &start.api_key {
        let mut value =
            HeaderValue::from_str(&format!("Bearer {}", key.expose())).map_err(|_| {
                ModelError::Failed("the API key cannot be sent as an HTTP header".into())
            })?;
        value.set_sensitive(true);
        request = request.header(AUTHORIZATION, value);
    }
    let response = request
        .send()
        .await
        .map_err(|error| map_transport_error(&error))?;
    let status = response.status();
    if !status.is_success() {
        // The body is dropped unread: it can echo the request, and the caller
        // is told only the status.
        return Err(ModelError::Status(status.as_u16()));
    }
    Ok(response)
}

async fn step(state: State) -> Option<(Result<ModelEvent, ModelError>, State)> {
    let mut reading = match state {
        State::Finished => return None,
        State::Start(start) => match send(*start).await {
            Ok(response) => Box::new(Reading {
                response,
                decoder: SseDecoder::default(),
                assembler: Assembler::default(),
                pending: VecDeque::new(),
                finished: false,
            }),
            Err(error) => return Some((Err(error), State::Finished)),
        },
        State::Reading(reading) => reading,
    };

    loop {
        if let Some(item) = reading.pending.pop_front() {
            let stop = item.is_err() || matches!(item, Ok(ModelEvent::Done(_)));
            return Some((
                item,
                if stop {
                    State::Finished
                } else {
                    State::Reading(reading)
                },
            ));
        }
        if reading.finished {
            return None;
        }
        match reading.response.chunk().await {
            Ok(Some(bytes)) => match reading.decoder.push(&bytes) {
                Ok(events) => {
                    for data in events {
                        handle_event(&mut reading, &data);
                        if reading.finished {
                            break;
                        }
                    }
                }
                Err(error) => {
                    reading.pending.push_back(Err(error));
                    reading.finished = true;
                }
            },
            Ok(None) => {
                if let Some(data) = reading.decoder.finish() {
                    handle_event(&mut reading, &data);
                }
                if !reading.finished {
                    if reading.assembler.finished {
                        complete(&mut reading);
                    } else {
                        reading.pending.push_back(Err(ModelError::Protocol(
                            "the stream ended before the model finished".into(),
                        )));
                        reading.finished = true;
                    }
                }
            }
            Err(error) => {
                reading.pending.push_back(Err(map_transport_error(&error)));
                reading.finished = true;
            }
        }
    }
}

/// One event's data: `[DONE]` completes the response, anything else is a chunk.
fn handle_event(reading: &mut Reading, data: &str) {
    let data = data.trim();
    if data == "[DONE]" {
        complete(reading);
        return;
    }
    if data.is_empty() {
        return;
    }
    match reading.assembler.push(data) {
        Ok(events) => reading.pending.extend(events.into_iter().map(Ok)),
        Err(error) => {
            reading.pending.push_back(Err(error));
            reading.finished = true;
        }
    }
}

fn complete(reading: &mut Reading) {
    let response = std::mem::take(&mut reading.assembler).into_response();
    reading.pending.push_back(response.map(ModelEvent::Done));
    reading.finished = true;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: format!("{name} tool"),
            parameters: json!({"type": "object"}),
            strict: true,
        }
    }

    fn call(id: &str, name: &str, arguments: &str) -> ToolCallItem {
        ToolCallItem {
            call_id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    fn request(
        input: Vec<InputItem>,
        tools: Vec<ToolSpec>,
        settings: ModelSettings,
    ) -> ModelRequest {
        ModelRequest {
            system: "Be brief.".into(),
            input,
            tools,
            settings,
        }
    }

    #[test]
    fn messages_put_the_system_prompt_first_and_merge_an_assistants_text_with_its_calls() {
        let messages = chat_messages(
            "Be brief.",
            &[
                InputItem::User("Weather in Oslo?".into()),
                InputItem::Assistant {
                    text: Some("Checking.".into()),
                    tool_calls: vec![],
                },
                InputItem::Assistant {
                    text: None,
                    tool_calls: vec![
                        call("c1", "weather", "{\"city\":\"Oslo\"}"),
                        call("c2", "clock", ""),
                    ],
                },
                InputItem::ToolResult {
                    call_id: "c1".into(),
                    output: "sunny".into(),
                },
                InputItem::ToolResult {
                    call_id: "c2".into(),
                    output: "noon".into(),
                },
                InputItem::Assistant {
                    text: Some("Sunny at noon.".into()),
                    tool_calls: vec![],
                },
            ],
        );
        assert_eq!(
            messages,
            vec![
                json!({"role": "system", "content": "Be brief."}),
                json!({"role": "user", "content": "Weather in Oslo?"}),
                json!({"role": "assistant", "content": "Checking.", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "weather", "arguments": "{\"city\":\"Oslo\"}"}},
                    {"id": "c2", "type": "function", "function": {"name": "clock", "arguments": "{}"}},
                ]}),
                json!({"role": "tool", "tool_call_id": "c1", "content": "sunny"}),
                json!({"role": "tool", "tool_call_id": "c2", "content": "noon"}),
                json!({"role": "assistant", "content": "Sunny at noon."}),
            ]
        );
    }

    #[test]
    fn calls_that_precede_their_text_still_make_one_assistant_message() {
        let messages = chat_messages(
            "",
            &[
                InputItem::Assistant {
                    text: None,
                    tool_calls: vec![call("c1", "t", "{}")],
                },
                InputItem::Assistant {
                    text: Some("Working on it.".into()),
                    tool_calls: vec![],
                },
            ],
        );
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["content"], "Working on it.");
        assert_eq!(messages[0]["tool_calls"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn two_plain_assistant_texts_stay_two_messages_and_no_system_means_none() {
        let messages = chat_messages(
            "",
            &[
                InputItem::Assistant {
                    text: Some("a".into()),
                    tool_calls: vec![],
                },
                InputItem::Assistant {
                    text: Some("b".into()),
                    tool_calls: vec![],
                },
            ],
        );
        assert_eq!(
            messages,
            vec![
                json!({"role": "assistant", "content": "a"}),
                json!({"role": "assistant", "content": "b"})
            ]
        );
    }

    #[test]
    fn tools_and_tool_settings_appear_only_when_there_are_tools() {
        let settings = ModelSettings {
            tool_choice: Some("required".into()),
            parallel_tool_calls: Some(false),
            temperature: Some(0.5),
            max_tokens: Some(64),
            ..ModelSettings::default()
        };
        let without = chat_request_body(
            "m",
            &request(vec![InputItem::User("hi".into())], vec![], settings.clone()),
        );
        assert!(without.get("tools").is_none());
        assert!(without.get("tool_choice").is_none());
        assert!(without.get("parallel_tool_calls").is_none());
        assert_eq!(without["temperature"], json!(0.5));
        assert_eq!(without["max_tokens"], json!(64));
        assert_eq!(without["stream"], json!(true));

        let with = chat_request_body(
            "m",
            &request(
                vec![InputItem::User("hi".into())],
                vec![spec("echo")],
                settings,
            ),
        );
        assert_eq!(with["tools"][0]["type"], "function");
        assert_eq!(with["tools"][0]["function"]["name"], "echo");
        assert_eq!(with["tools"][0]["function"]["strict"], json!(true));
        assert_eq!(with["tool_choice"], "required");
        assert_eq!(with["parallel_tool_calls"], json!(false));
    }

    #[test]
    fn usage_is_requested_by_default_and_a_named_tool_choice_forces_that_function() {
        let body = chat_request_body(
            "m",
            &request(vec![], vec![spec("echo")], ModelSettings::default()),
        );
        assert_eq!(body["stream_options"], json!({"include_usage": true}));
        assert!(
            body.get("temperature").is_none()
                && body.get("top_p").is_none()
                && body.get("max_tokens").is_none()
        );
        let off = ModelSettings {
            include_usage: Some(false),
            tool_choice: Some("echo".into()),
            ..ModelSettings::default()
        };
        let body = chat_request_body("m", &request(vec![], vec![spec("echo")], off));
        assert!(body.get("stream_options").is_none());
        assert_eq!(
            body["tool_choice"],
            json!({"type": "function", "function": {"name": "echo"}})
        );
    }

    #[test]
    fn urls_are_sanitised_for_traces() {
        assert_eq!(
            sanitize_url_for_trace("http://user:pw@127.0.0.1:11434/v1?x=1#f"),
            "http://127.0.0.1:11434/v1"
        );
        assert_eq!(
            sanitize_url_for_trace("https://api.example.com/v1/"),
            "https://api.example.com/v1/"
        );
        assert_eq!(
            sanitize_url_for_trace("https://api.example.com"),
            "https://api.example.com"
        );
        assert_eq!(sanitize_url_for_trace("not a url"), "");
    }

    #[test]
    fn a_base_url_with_credentials_or_a_bad_scheme_is_refused() {
        for bad in [
            "http://user:pw@127.0.0.1/v1",
            "ftp://host/v1",
            "nonsense",
            "http://",
        ] {
            let refused =
                ChatCompletionsModel::new(ChatCompletionsConfig::new(bad, "m")).unwrap_err();
            assert!(!refused.to_string().contains("pw"), "{refused}");
        }
        assert!(
            ChatCompletionsModel::new(
                ChatCompletionsConfig::new("http://127.0.0.1:1/v1", "m").local(true)
            )
            .is_ok()
        );
    }

    #[test]
    fn debug_output_never_shows_the_key_or_url_credentials() {
        let config = ChatCompletionsConfig::new("https://host.example/v1", "m")
            .with_api_key("sk-secret-123");
        assert!(!format!("{config:?}").contains("sk-secret-123"));
        let model = ChatCompletionsModel::new(config).unwrap();
        let printed = format!("{model:?}");
        assert!(
            !printed.contains("sk-secret-123") && printed.contains("***"),
            "{printed}"
        );
    }

    #[test]
    fn sse_events_are_reassembled_across_chunk_boundaries() {
        let mut decoder = SseDecoder::default();
        assert!(decoder.push(b"data: {\"a\":").unwrap().is_empty());
        assert!(decoder.push(b"1}\r\n").unwrap().is_empty());
        let events = decoder
            .push(b"\r\n: keep-alive\n\ndata: [DONE]\n\n")
            .unwrap();
        assert_eq!(events, vec!["{\"a\":1}".to_owned(), "[DONE]".to_owned()]);
        assert!(decoder.finish().is_none());

        let mut decoder = SseDecoder::default();
        assert!(decoder.push(b"data: last").unwrap().is_empty());
        assert_eq!(decoder.finish().as_deref(), Some("last"));
    }

    #[test]
    fn a_response_larger_than_the_cap_is_refused_however_it_is_sliced() {
        let mut assembler = Assembler::default();
        let piece = "x".repeat(1024 * 1024);
        let chunk = format!("{{\"choices\":[{{\"delta\":{{\"content\":\"{piece}\"}}}}]}}");
        let mut refused_after = None;
        for count in 1..=40 {
            if assembler.push(&chunk).is_err() {
                refused_after = Some(count);
                break;
            }
        }
        assert_eq!(refused_after, Some(MAX_RESPONSE_BYTES / (1024 * 1024) + 1));

        // Tool-call arguments count too.
        let mut assembler = Assembler::default();
        let arguments = format!(
            "{{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":0,\"function\":{{\"name\":\"t\",\"arguments\":\"{piece}\"}}}}]}}}}]}}"
        );
        let refused = (0..40).any(|_| assembler.push(&arguments).is_err());
        assert!(refused);
    }

    #[test]
    fn an_event_fed_one_byte_at_a_time_comes_out_whole() {
        let mut decoder = SseDecoder::default();
        let wire = b"data: {\"a\": \"long line\"}\r\n\r\ndata: second\n\n";
        let mut events = Vec::new();
        for byte in wire {
            events.extend(decoder.push(&[*byte]).unwrap());
        }
        assert_eq!(
            events,
            vec!["{\"a\": \"long line\"}".to_owned(), "second".to_owned()]
        );
    }

    #[test]
    fn an_oversized_event_is_refused() {
        let mut decoder = SseDecoder::default();
        let block = vec![b'x'; 1024 * 1024];
        let mut refused = false;
        for _ in 0..40 {
            if decoder.push(&block).is_err() {
                refused = true;
                break;
            }
        }
        assert!(refused);
    }

    #[test]
    fn tool_call_deltas_split_across_chunks_are_assembled_by_index() {
        let mut assembler = Assembler::default();
        let chunks = [
            r#"{"choices":[{"delta":{"content":"Let me "}}]}"#,
            r#"{"choices":[{"delta":{"content":"check."}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"weather","arguments":""}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"ci"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_b","function":{"name":"clock","arguments":"{}"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\":\"Oslo\"}"}}]}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            r#"{"choices":[],"usage":{"prompt_tokens":1200,"completion_tokens":34,"total_tokens":1234}}"#,
        ];
        let mut deltas = String::new();
        for chunk in chunks {
            for event in assembler.push(chunk).unwrap() {
                if let ModelEvent::TextDelta(text) = event {
                    deltas.push_str(&text);
                }
            }
        }
        assert_eq!(deltas, "Let me check.");
        let response = assembler.into_response().unwrap();
        assert_eq!(
            response.output,
            vec![
                OutputItem::Message {
                    text: "Let me check.".into()
                },
                OutputItem::FunctionCall {
                    call_id: "call_a".into(),
                    name: "weather".into(),
                    arguments: "{\"city\":\"Oslo\"}".into()
                },
                OutputItem::FunctionCall {
                    call_id: "call_b".into(),
                    name: "clock".into(),
                    arguments: "{}".into()
                },
            ]
        );
        assert_eq!(
            response.usage,
            Some(Usage {
                requests: 1,
                input_tokens: 1200,
                output_tokens: 34,
                total_tokens: 1234
            })
        );
    }

    #[test]
    fn reasoning_from_either_field_is_kept_apart_from_the_answer() {
        let mut assembler = Assembler::default();
        assembler
            .push(r#"{"choices":[{"delta":{"reasoning_content":"think "}}]}"#)
            .unwrap();
        assembler
            .push(r#"{"choices":[{"delta":{"reasoning":"more"}}]}"#)
            .unwrap();
        assembler
            .push(r#"{"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}"#)
            .unwrap();
        assert!(assembler.finished);
        let response = assembler.into_response().unwrap();
        assert_eq!(
            response.output,
            vec![
                OutputItem::Reasoning {
                    text: "think more".into()
                },
                OutputItem::Message {
                    text: "answer".into()
                }
            ]
        );
        assert_eq!(response.usage, None);
    }

    #[test]
    fn malformed_and_error_chunks_are_protocol_errors_with_fixed_reasons() {
        let mut assembler = Assembler::default();
        let malformed = assembler.push("{not json with secret-token").unwrap_err();
        assert!(
            matches!(&malformed, ModelError::Protocol(reason) if !reason.contains("secret-token"))
        );
        let reported = assembler
            .push(r#"{"error":{"message":"bad key sk-123"}}"#)
            .unwrap_err();
        assert!(!reported.to_string().contains("sk-123"));
    }

    #[test]
    fn a_call_keeps_the_empty_id_the_server_gave_it_and_empty_stubs_are_dropped() {
        let mut assembler = Assembler::default();
        assembler.push(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"t","arguments":"{}"}},{"index":1,"function":{"name":"","arguments":""}}]}}]}"#).unwrap();
        let response = assembler.into_response().unwrap();
        assert_eq!(
            response.output,
            vec![OutputItem::FunctionCall {
                call_id: String::new(),
                name: "t".into(),
                arguments: "{}".into()
            }],
            "no id is invented: the runner refuses a call without one"
        );
    }

    #[test]
    fn refusals_and_finish_reasons_are_read_as_the_sdk_reads_them() {
        let finish = |chunks: &[&str]| {
            let mut assembler = Assembler::default();
            for chunk in chunks {
                assembler.push(chunk).unwrap();
            }
            assembler.into_response()
        };
        // A refusal is its own output item, after the text, before the calls.
        let response = finish(&[
            r#"{"choices":[{"index":0,"delta":{"content":"Sorry. "}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"refusal":"I can't "}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"refusal":"do that."},"finish_reason":"stop"}]}"#,
        ])
        .unwrap();
        assert_eq!(
            response.output,
            vec![
                OutputItem::Message {
                    text: "Sorry. ".into()
                },
                OutputItem::Refusal {
                    text: "I can't do that.".into()
                }
            ]
        );
        // Content filter: a refusal only when nothing else was made.
        let withheld =
            finish(&[r#"{"choices":[{"delta":{"content":""},"finish_reason":"content_filter"}]}"#])
                .unwrap();
        assert_eq!(
            withheld.output,
            vec![OutputItem::Refusal {
                text: CONTENT_FILTER_REFUSAL.into()
            }]
        );
        let kept = finish(&[
            r#"{"choices":[{"delta":{"content":"half"},"finish_reason":"content_filter"}]}"#,
        ])
        .unwrap();
        assert_eq!(
            kept.output,
            vec![OutputItem::Message {
                text: "half".into()
            }]
        );
        // Reasoning is not an answer: the filter still withholds, the length still fails.
        let reasoning_then_filter = finish(&[
            r#"{"choices":[{"delta":{"reasoning":"hm"}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"content_filter"}]}"#,
        ])
        .unwrap();
        assert_eq!(reasoning_then_filter.output.len(), 2);
        assert_eq!(
            finish(&[
                r#"{"choices":[{"delta":{"reasoning":"hm"}}]}"#,
                r#"{"choices":[{"delta":{},"finish_reason":"length"}]}"#,
            ])
            .unwrap_err(),
            ModelError::Truncated
        );
        // A refusal, or a call, or text, is something: no error for `length`.
        for chunk in [
            r#"{"choices":[{"delta":{"refusal":"no"},"finish_reason":"length"}]}"#,
            r#"{"choices":[{"delta":{"content":"a"},"finish_reason":"length"}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"t","arguments":"{}"}}]},"finish_reason":"length"}]}"#,
        ] {
            assert!(finish(&[chunk]).is_ok(), "{chunk}");
        }
        // Both flags with nothing made: the refusal the SDK synthesises wins.
        let both = finish(&[
            r#"{"choices":[{"delta":{},"finish_reason":"content_filter"}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"length"}]}"#,
        ])
        .unwrap();
        assert!(matches!(&both.output[..], [OutputItem::Refusal { .. }]));
    }

    #[test]
    fn only_the_choice_with_index_zero_is_read_and_a_server_that_numbers_none_is_read_from_its_first()
     {
        let choices = |json: &str| -> Option<Choice> {
            let chunk: Chunk = serde_json::from_str(json).unwrap();
            choice_zero(chunk.choices)
        };
        let content = |choice: Option<Choice>| {
            choice
                .and_then(|choice| choice.delta)
                .and_then(|delta| delta.content)
        };
        assert_eq!(
            content(choices(
                r#"{"choices":[{"index":1,"delta":{"content":"b"}},{"index":0,"delta":{"content":"a"}}]}"#
            ))
            .as_deref(),
            Some("a")
        );
        assert_eq!(
            content(choices(
                r#"{"choices":[{"index":1,"delta":{"content":"b"}}]}"#
            )),
            None
        );
        assert_eq!(
            content(choices(
                r#"{"choices":[{"delta":{"content":"first"}},{"delta":{"content":"second"}}]}"#
            ))
            .as_deref(),
            Some("first")
        );
        assert!(choices(r#"{"choices":[]}"#).is_none());
    }

    #[test]
    fn a_tool_call_name_is_replaced_not_joined_and_the_latest_id_wins() {
        let mut assembler = Assembler::default();
        for chunk in [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"wea","arguments":"{"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"","function":{"name":"","arguments":"}"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"b","function":{"name":"ther"}}]},"finish_reason":"tool_calls"}]}"#,
        ] {
            assembler.push(chunk).unwrap();
        }
        assert_eq!(
            assembler.into_response().unwrap().output,
            vec![OutputItem::FunctionCall {
                call_id: "b".into(),
                name: "ther".into(),
                arguments: "{}".into()
            }]
        );
    }

    #[test]
    fn tool_call_entries_are_paid_for_and_capped() {
        let mut assembler = Assembler::default();
        let entry = |index: usize| {
            format!(r#"{{"choices":[{{"delta":{{"tool_calls":[{{"index":{index}}}]}}}}]}}"#)
        };
        for index in 0..MAX_TOOL_CALLS {
            assembler.push(&entry(index)).unwrap();
        }
        // The same index again is not a new call, but it is not free either.
        assembler.push(&entry(0)).unwrap();
        assert_eq!(assembler.bytes, (MAX_TOOL_CALLS + 1) * TOOL_CALL_ENTRY_COST);
        assert_eq!(
            assembler.push(&entry(MAX_TOOL_CALLS)).unwrap_err(),
            ModelError::Protocol("the response has too many tool calls".into())
        );
    }
}
