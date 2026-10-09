//! Function tools: what the model can call, and what happens when a call fails.
//!
//! Ports `agents.tool.FunctionTool` and the failure handling the SDK wraps
//! around `@function_tool` functions (`_FailureHandlingFunctionToolInvoker`,
//! `default_tool_error_function`, `_build_handled_function_tool_error_handler`,
//! `_parse_function_tool_json_input`; 0.22.3), for tools written as Rust
//! closures over a JSON argument object.
//!
//! Failure semantics, copied from the SDK and checked against goldens recorded
//! from it (`tool_error_default_policy`, `tool_invalid_json`,
//! `tool_non_object_arguments`, `sensitive_data_off`):
//! - Arguments that are not JSON (or not a JSON object) and handlers that
//!   return an error are not fatal to the run. The model receives the error
//!   policy's text as the tool's output and can try again.
//! - The SDK's default policy answers `An error occurred while running the tool.
//!   Please try again. Error: <error text>`; a custom policy answers whatever it
//!   returns, given the same error.
//! - The tool's function span records `{"message": "Error running tool
//!   (non-fatal)", "data": {"tool_name", "error"}}`. The error text is replaced by
//!   `Tool execution failed. Error details are redacted.` when the run does not
//!   trace sensitive data (`get_trace_tool_error`).
//! - Unreadable arguments are `Invalid JSON input for tool <name>`, and nothing
//!   more: the SDK keeps tool data out of error text unless
//!   `OPENAI_AGENTS_DONT_LOG_TOOL_DATA` is switched off, and it is ON by
//!   default. This port has no such switch and always behaves as the SDK does by
//!   default, so a model's malformed arguments (which may contain anything) are
//!   never echoed into a span or back to the model, and the parser's own message
//!   is not reproduced. JSON that is valid but not an object is
//!   `Invalid JSON input for tool <name>: expected a JSON object`, which the SDK
//!   says in either mode.
//!
//! Deviations, and why:
//! - The SDK validates the arguments against a pydantic model built from the
//!   function's signature. Here the handler receives the parsed JSON object and
//!   validates it itself (or relies on the model honouring `parameters`).
//! - A handler that panics is a failed tool call, not a dead run: Python turns
//!   any `Exception` into a tool error, and a panic is Rust's nearest thing. The
//!   panic's message is not repeated (it may hold data).
//! - `failure_error_function=None` (raise instead of answering the model) has
//!   no counterpart: every failure is answered.
//!
//! Approval (`needs_approval`, `RunState.approve()` / `.reject()`; pinned by the
//! goldens `approval_*`): a call whose tool needs approval WAITS IN PLACE for
//! the caller's [`ApprovalPort`] before its handler runs (see [`crate::run`]).
//! - [`NeedsApproval::Always`] asks for every call; a
//!   [`NeedsApproval::Predicate`] decides per call from the parsed arguments,
//!   and one that panics counts as `Always`.
//! - Approve: the handler runs as usual. Reject: the handler never runs and the
//!   model is told the SDK's `"Tool execution was not approved."`, followed by
//!   `" The user said: <note>"` when the rejection carries a note (at most
//!   2,000 characters of it; an empty note is no note). That is the text the
//!   SDK sends for `reject(item, rejection_message=<the same text>)`, which is
//!   how the golden `approval_reject_with_note` was recorded.
//! - A call that needs approval when no port is configured is rejected with
//!   `"Tool execution was not approved. No one was asked."`: it fails closed.
//!   A port that panics rejects, with no note.
//! - The decision comes only from the port. Nothing in a tool's output, the
//!   model's text or the arguments can approve a call.
//!
//! An MCP server's tool (`FunctionTool::mcp_server`): the caller brings it as
//! a function tool that names its server, and its function span carries
//! `mcp_data: {"server": <name>}` while it runs, as the SDK's
//! `MCPUtil.invoke_mcp_tool` sets it. No SDK golden pins this yet, and the
//! SDK's `mcp_tools` listing span is not emitted (the caller lists a server's
//! tools once per server process, not at every turn).
//!
//! Deviations in approval, and why:
//! - The SDK asks for approval of a call whose arguments are not valid JSON (or
//!   not an object) and answers the error after approval (`tool.py:509-511`).
//!   This port answers the error at once, without asking: there is nothing
//!   valid to approve.
//! - A rejected call's function span keeps `output: null`, as the SDK's single
//!   interrupted span does; there is no SDK equivalent of the port's span
//!   covering the wait (the SDK's resume is a second run), see [`crate::run`].

use std::fmt;
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use lattice_protocol::SpanError;
use serde_json::{Value, json};

use crate::run::RunContext;

/// Shown in a span instead of the error text when sensitive data is off
/// (`REDACTED_TOOL_ERROR_MESSAGE`).
pub(crate) const REDACTED_TOOL_ERROR: &str = "Tool execution failed. Error details are redacted.";

/// What a running tool knows about its call.
#[derive(Clone)]
pub struct ToolContext {
    /// `RunConfig::context`, as the caller gave it.
    pub run_context: RunContext,
    /// The name of the agent making the call.
    pub agent: String,
    pub call_id: String,
}

impl fmt::Debug for ToolContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolContext")
            .field("agent", &self.agent)
            .field("call_id", &self.call_id)
            .finish_non_exhaustive()
    }
}

/// A tool handler's failure. The message goes to the model (through the error
/// policy), so it should say what went wrong without repeating secrets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolError {
    message: String,
}

impl ToolError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ToolError {}

impl From<String> for ToolError {
    fn from(message: String) -> Self {
        Self { message }
    }
}

impl From<&str> for ToolError {
    fn from(message: &str) -> Self {
        Self {
            message: message.to_owned(),
        }
    }
}

pub type ToolHandler =
    Arc<dyn Fn(ToolContext, Value) -> BoxFuture<'static, Result<String, ToolError>> + Send + Sync>;
pub type ToolErrorFormatter = Arc<dyn Fn(&ToolError) -> String + Send + Sync>;

/// What the model is told when a tool call fails.
#[derive(Clone, Default)]
pub enum ToolErrorPolicy {
    /// The SDK's `default_tool_error_function`.
    #[default]
    SdkDefault,
    /// The caller's own text, given the error the SDK would have raised.
    Custom(ToolErrorFormatter),
}

impl fmt::Debug for ToolErrorPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolErrorPolicy::SdkDefault => f.write_str("SdkDefault"),
            ToolErrorPolicy::Custom(_) => f.write_str("Custom(..)"),
        }
    }
}

/// The SDK's answer to a call that was not approved
/// (`DEFAULT_APPROVAL_REJECTION_MESSAGE`).
pub(crate) const NOT_APPROVED: &str = "Tool execution was not approved.";
/// The answer when a call needs approval and no port is configured.
pub(crate) const NO_ONE_ASKED: &str = "Tool execution was not approved. No one was asked.";
/// The most characters of a rejection's note the model is told.
pub const MAX_NOTE_CHARS: usize = 2_000;

pub type ApprovalPredicate = Arc<dyn Fn(&ToolContext, &Value) -> bool + Send + Sync>;

/// Whether a call must be approved before its handler runs (`needs_approval`).
#[derive(Clone, Default)]
pub enum NeedsApproval {
    /// Never asks (`needs_approval=False`).
    #[default]
    Never,
    /// Every call with readable arguments asks (`needs_approval=True`).
    Always,
    /// Decided per call from the parsed arguments; a predicate that panics
    /// counts as [`NeedsApproval::Always`].
    Predicate(ApprovalPredicate),
}

impl NeedsApproval {
    /// Whether the call with these (parsed) arguments must be approved.
    pub(crate) fn asks(&self, context: &ToolContext, arguments: &Value) -> bool {
        match self {
            NeedsApproval::Never => false,
            NeedsApproval::Always => true,
            NeedsApproval::Predicate(decide) => {
                catch_unwind(AssertUnwindSafe(|| decide(context, arguments))).unwrap_or(true)
            }
        }
    }
}

impl fmt::Debug for NeedsApproval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NeedsApproval::Never => f.write_str("Never"),
            NeedsApproval::Always => f.write_str("Always"),
            NeedsApproval::Predicate(_) => f.write_str("Predicate(..)"),
        }
    }
}

/// A call waiting for a decision.
#[derive(Clone, Debug, PartialEq)]
pub struct ApprovalRequest {
    /// The agent making the call.
    pub agent: String,
    pub call_id: String,
    /// The tool's name.
    pub tool: String,
    /// The call's arguments, parsed (always a JSON object).
    pub arguments: Value,
}

/// What the reader decided about one call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApprovalDecision {
    Approve,
    /// The handler does not run; a note, if any, is passed on to the model.
    Reject {
        note: Option<String>,
    },
}

/// Where a run asks whether a call may run (the caller's; in Lattice, the
/// reader). The run waits in place, so the other calls of the same response
/// keep running meanwhile.
pub trait ApprovalPort: Send + Sync {
    /// Resolves when the reader decides. Dropped, unresolved, when the run is
    /// cancelled.
    fn request(&self, request: ApprovalRequest) -> BoxFuture<'static, ApprovalDecision>;
}

/// What the model is told about a rejected call.
pub(crate) fn rejection_output(note: Option<&str>) -> String {
    match note.map(str::trim).filter(|note| !note.is_empty()) {
        None => NOT_APPROVED.to_owned(),
        Some(note) => {
            let note: String = note.chars().take(MAX_NOTE_CHARS).collect();
            format!("{NOT_APPROVED} The user said: {note}")
        }
    }
}

/// A tool the model can call.
#[derive(Clone)]
pub struct FunctionTool {
    pub name: String,
    pub description: String,
    /// JSON Schema of the arguments object.
    pub parameters: Value,
    /// Whether `parameters` is a strict-mode schema.
    pub strict: bool,
    pub handler: ToolHandler,
    pub error_policy: ToolErrorPolicy,
    /// Whether a call must be approved first (default never).
    pub needs_approval: NeedsApproval,
    /// The MCP server the tool belongs to, if it is one's: its function span
    /// carries `mcp_data: {"server": <name>}` while it runs, as the SDK's
    /// `invoke_mcp_tool` sets it. This crate starts no server; the caller
    /// brings the tool.
    pub mcp_server: Option<String>,
}

impl FunctionTool {
    /// A strict-mode tool with the SDK's default error policy.
    pub fn new<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        handler: F,
    ) -> Self
    where
        F: Fn(ToolContext, Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String, ToolError>> + Send + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            strict: true,
            handler: Arc::new(move |context, arguments| handler(context, arguments).boxed()),
            error_policy: ToolErrorPolicy::SdkDefault,
            needs_approval: NeedsApproval::Never,
            mcp_server: None,
        }
    }

    /// Mark the tool as a tool of the MCP server `server` (its spans say so).
    pub fn with_mcp_server(mut self, server: impl Into<String>) -> Self {
        self.mcp_server = Some(server.into());
        self
    }

    pub fn with_needs_approval(mut self, needs_approval: NeedsApproval) -> Self {
        self.needs_approval = needs_approval;
        self
    }

    pub fn with_error_policy(mut self, policy: ToolErrorPolicy) -> Self {
        self.error_policy = policy;
        self
    }

    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Two tools are the same tool when they share a handler; the agent graph
    /// draws one node for a tool that several agents hold.
    pub(crate) fn identity(&self) -> usize {
        Arc::as_ptr(&self.handler) as *const () as usize
    }
}

impl fmt::Debug for FunctionTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FunctionTool")
            .field("name", &self.name)
            .field("strict", &self.strict)
            .field("error_policy", &self.error_policy)
            .field("needs_approval", &self.needs_approval)
            .finish_non_exhaustive()
    }
}

/// `{"type": "object", "properties": ..., "required": [...],
/// "additionalProperties": false}`: the strict-mode schema of an arguments
/// object.
pub fn strict_object_schema(properties: Value, required: &[&str]) -> Value {
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}

/// How one call ended, for the runner: the text the model receives and, when
/// the call failed, the error its function span records.
pub(crate) struct Invocation {
    pub output: String,
    pub span_error: Option<SpanError>,
}

/// Call `tool` with `arguments` (the model's JSON text) and turn any failure
/// into the SDK's non-fatal outcome.
pub(crate) async fn invoke(
    tool: &FunctionTool,
    context: ToolContext,
    arguments: &str,
    include_sensitive_data: bool,
) -> Invocation {
    let error = match run(tool, context, arguments).await {
        Ok(output) => {
            return Invocation {
                output,
                span_error: None,
            };
        }
        Err(error) => error,
    };
    let default_text =
        || format!("An error occurred while running the tool. Please try again. Error: {error}");
    let output = match &tool.error_policy {
        ToolErrorPolicy::SdkDefault => default_text(),
        // A formatter that panics must not take the run down: the model hears
        // the SDK's default text instead.
        ToolErrorPolicy::Custom(format) => {
            catch_unwind(AssertUnwindSafe(|| format(&error))).unwrap_or_else(|_| default_text())
        }
    };
    let traced = if include_sensitive_data {
        error.message().to_owned()
    } else {
        REDACTED_TOOL_ERROR.to_owned()
    };
    Invocation {
        output,
        span_error: Some(SpanError {
            message: "Error running tool (non-fatal)".to_owned(),
            data: Some(json!({"tool_name": tool.name, "error": traced})),
        }),
    }
}

async fn run(
    tool: &FunctionTool,
    context: ToolContext,
    arguments: &str,
) -> Result<String, ToolError> {
    let parsed = parse_arguments(&tool.name, arguments)?;
    let future = match catch_unwind(AssertUnwindSafe(|| (tool.handler)(context, parsed))) {
        Ok(future) => future,
        Err(_) => return Err(ToolError::new("the tool panicked")),
    };
    match AssertUnwindSafe(future).catch_unwind().await {
        Ok(result) => result,
        Err(_) => Err(ToolError::new("the tool panicked")),
    }
}

/// `_parse_function_tool_json_input`: empty text is an empty object; anything
/// else must be a JSON object. The error for text that is not JSON leaves the
/// text out (see the module documentation).
pub(crate) fn parse_arguments(tool_name: &str, arguments: &str) -> Result<Value, ToolError> {
    if arguments.is_empty() {
        return Ok(json!({}));
    }
    match serde_json::from_str::<Value>(arguments) {
        Ok(value @ Value::Object(_)) => Ok(value),
        Ok(_) => Err(ToolError::new(format!(
            "Invalid JSON input for tool {tool_name}: expected a JSON object"
        ))),
        Err(_) => Err(ToolError::new(format!(
            "Invalid JSON input for tool {tool_name}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> ToolContext {
        ToolContext {
            run_context: Arc::new(()),
            agent: "Assistant".into(),
            call_id: "call_1".into(),
        }
    }

    fn echo() -> FunctionTool {
        FunctionTool::new(
            "echo",
            "Repeat the text back.",
            strict_object_schema(json!({"text": {"type": "string"}}), &["text"]),
            |_, arguments| async move {
                Ok(format!("echo:{}", arguments["text"].as_str().unwrap_or("")))
            },
        )
    }

    fn failing() -> FunctionTool {
        FunctionTool::new(
            "explode",
            "Always fails.",
            strict_object_schema(json!({}), &[]),
            |_, _| async { Err(ToolError::new("boom: test")) },
        )
    }

    #[tokio::test]
    async fn a_successful_call_returns_its_output() {
        let result = invoke(&echo(), context(), "{\"text\":\"hi\"}", true).await;
        assert_eq!(result.output, "echo:hi");
        assert!(result.span_error.is_none());
    }

    #[tokio::test]
    async fn empty_arguments_are_an_empty_object() {
        let seen = FunctionTool::new("probe", "", json!({}), |_, arguments| async move {
            Ok(arguments.to_string())
        });
        assert_eq!(invoke(&seen, context(), "", true).await.output, "{}");
    }

    #[tokio::test]
    async fn a_handler_error_follows_the_sdk_default_policy() {
        let result = invoke(&failing(), context(), "{}", true).await;
        assert_eq!(
            result.output,
            "An error occurred while running the tool. Please try again. Error: boom: test"
        );
        let error = result.span_error.unwrap();
        assert_eq!(error.message, "Error running tool (non-fatal)");
        assert_eq!(
            error.data.unwrap(),
            json!({"tool_name": "explode", "error": "boom: test"})
        );
    }

    #[tokio::test]
    async fn the_span_error_text_is_redacted_when_sensitive_data_is_off_but_the_model_still_hears_it()
     {
        let result = invoke(&failing(), context(), "{}", false).await;
        assert!(result.output.ends_with("Error: boom: test"));
        assert_eq!(
            result.span_error.unwrap().data.unwrap()["error"],
            json!(REDACTED_TOOL_ERROR)
        );
    }

    #[tokio::test]
    async fn unreadable_arguments_are_named_but_never_echoed() {
        let result = invoke(&echo(), context(), "{secret-token: nope", true).await;
        assert_eq!(
            result.output,
            "An error occurred while running the tool. Please try again. Error: Invalid JSON input for tool echo"
        );
        assert!(!result.output.contains("secret-token"));
        let error = result.span_error.unwrap();
        assert_eq!(error.message, "Error running tool (non-fatal)");
        assert_eq!(
            error.data.unwrap(),
            json!({"tool_name": "echo", "error": "Invalid JSON input for tool echo"})
        );
    }

    #[tokio::test]
    async fn arguments_that_are_not_an_object_say_so() {
        let result = invoke(&echo(), context(), "[1]", true).await;
        assert_eq!(
            result.output,
            "An error occurred while running the tool. Please try again. \
             Error: Invalid JSON input for tool echo: expected a JSON object"
        );
    }

    #[tokio::test]
    async fn a_custom_policy_supplies_the_text_and_sees_the_sdk_error() {
        let tool = echo().with_error_policy(ToolErrorPolicy::Custom(Arc::new(|error| {
            format!("custom:{error}")
        })));
        let result = invoke(&tool, context(), "nope", true).await;
        assert_eq!(result.output, "custom:Invalid JSON input for tool echo");
        let tool = failing().with_error_policy(ToolErrorPolicy::Custom(Arc::new(|error| {
            format!("custom:{error}")
        })));
        let result = invoke(&tool, context(), "{}", true).await;
        assert_eq!(result.output, "custom:boom: test");
        // The span records the real error, not the custom text.
        assert_eq!(
            result.span_error.unwrap().data.unwrap()["error"],
            json!("boom: test")
        );
    }

    #[tokio::test]
    async fn a_custom_policy_that_panics_falls_back_to_the_sdk_default_text() {
        let tool = failing().with_error_policy(ToolErrorPolicy::Custom(Arc::new(|_| {
            panic!("formatter bug")
        })));
        let result = invoke(&tool, context(), "{}", true).await;
        assert_eq!(
            result.output,
            "An error occurred while running the tool. Please try again. Error: boom: test"
        );
    }

    #[tokio::test]
    async fn a_panicking_handler_is_a_failed_call_that_does_not_repeat_the_panic_message() {
        let tool = FunctionTool::new("panicky", "", json!({}), |_, _| async move {
            let fail = std::hint::black_box(true);
            if fail {
                panic!("secret-in-panic");
            }
            Ok(String::new())
        });
        let result = invoke(&tool, context(), "{}", true).await;
        assert!(result.output.contains("the tool panicked"));
        assert!(!result.output.contains("secret-in-panic"));
        let sync_panic = FunctionTool {
            handler: Arc::new(|_, _| panic!("sync secret")),
            ..failing()
        };
        let result = invoke(&sync_panic, context(), "{}", true).await;
        assert!(result.output.contains("the tool panicked"));
        assert!(!result.output.contains("sync secret"));
    }

    #[test]
    fn strict_schemas_forbid_extra_properties() {
        let schema = strict_object_schema(json!({"a": {"type": "string"}}), &["a"]);
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["additionalProperties"], json!(false));
        assert_eq!(schema["required"], json!(["a"]));
    }

    #[test]
    fn a_rejection_tells_the_model_the_sdks_text_and_at_most_two_thousand_characters_of_a_note() {
        assert_eq!(rejection_output(None), "Tool execution was not approved.");
        assert_eq!(
            rejection_output(Some("  ")),
            "Tool execution was not approved."
        );
        assert_eq!(
            rejection_output(Some("not that one")),
            "Tool execution was not approved. The user said: not that one"
        );
        let long = "\u{e9}".repeat(MAX_NOTE_CHARS + 50);
        let told = rejection_output(Some(&long));
        let note = told
            .strip_prefix("Tool execution was not approved. The user said: ")
            .unwrap();
        assert_eq!(note.chars().count(), MAX_NOTE_CHARS);
    }

    #[test]
    fn a_predicate_decides_per_call_and_one_that_panics_asks() {
        let context = context();
        let secret = NeedsApproval::Predicate(Arc::new(|_, arguments| {
            arguments["path"] == json!("secret.txt")
        }));
        assert!(secret.asks(&context, &json!({"path": "secret.txt"})));
        assert!(!secret.asks(&context, &json!({"path": "notes.txt"})));
        let broken = NeedsApproval::Predicate(Arc::new(|_, _| panic!("predicate bug")));
        assert!(broken.asks(&context, &json!({})));
        assert!(NeedsApproval::Always.asks(&context, &json!({})));
        assert!(!NeedsApproval::default().asks(&context, &json!({})));
        assert!(!echo().needs_approval.asks(&context, &json!({})));
    }

    #[test]
    fn a_cloned_tool_keeps_its_identity() {
        let tool = echo();
        assert_eq!(tool.identity(), tool.clone().identity());
        assert_ne!(tool.identity(), echo().identity());
    }
}
