//! The Chat Completions client against an in-process HTTP stub.
//!
//! A tokio `TcpListener` on 127.0.0.1 serves canned server-sent events, so
//! nothing here touches a network beyond the loopback interface. Each test also
//! pins one of the things this client must never do: follow a redirect, leak the
//! key or a response body into an error, route "local" traffic through a proxy,
//! or hang forever on a stalled stream.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use lattice_agents::{
    Agent, ChatCompletionsConfig, ChatCompletionsModel, FunctionTool, InputItem, Model, ModelError,
    ModelEvent, ModelRequest, ModelSettings, OutputItem, RunConfig, RunError, RunResult,
    ToolCallItem, ToolSpec, run_streamed, strict_object_schema,
};
use lattice_protocol::Usage;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

#[derive(Clone, Debug)]
struct Recorded {
    request_line: String,
    headers: Vec<(String, String)>,
    body: String,
}

impl Recorded {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    fn json(&self) -> Value {
        serde_json::from_str(&self.body).expect("the request body is JSON")
    }
}

#[derive(Clone)]
enum Reply {
    /// 200 with these pieces written one at a time, then the connection closes.
    Sse(Vec<Vec<u8>>),
    /// 200 with these pieces, then silence with the connection held open.
    SseThenStall(Vec<Vec<u8>>),
    Status {
        code: u16,
        body: &'static str,
    },
    Redirect(String),
}

struct Stub {
    addr: SocketAddr,
    recorded: Arc<Mutex<Vec<Recorded>>>,
    connections: Arc<AtomicUsize>,
    /// Stalled connections the client closed (see `Reply::SseThenStall`).
    closed_by_client: Arc<AtomicUsize>,
}

impl Stub {
    fn base_url(&self) -> String {
        format!("http://{}/v1", self.addr)
    }

    fn requests(&self) -> Vec<Recorded> {
        self.recorded.lock().unwrap().clone()
    }
}

async fn read_request(socket: &mut TcpStream) -> Option<Recorded> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        if let Some(position) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            break position;
        }
        let read = socket.read(&mut chunk).await.ok()?;
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
    };
    let head = String::from_utf8_lossy(&buffer[..header_end]).into_owned();
    let mut lines = head.split("\r\n");
    let request_line = lines.next()?.to_owned();
    let headers: Vec<(String, String)> = lines
        .filter_map(|line| {
            line.split_once(':')
                .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
        })
        .collect();
    let length = headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, value)| value.parse::<usize>().ok())
        .unwrap_or(0);
    let mut body = buffer[header_end + 4..].to_vec();
    while body.len() < length {
        let read = socket.read(&mut chunk).await.ok()?;
        if read == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..read]);
    }
    Some(Recorded {
        request_line,
        headers,
        body: String::from_utf8_lossy(&body).into_owned(),
    })
}

async fn write_sse_head_and_pieces(socket: &mut TcpStream, pieces: Vec<Vec<u8>>) {
    let _ = socket
        .write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
        )
        .await;
    for piece in pieces {
        let _ = socket.write_all(&piece).await;
        let _ = socket.flush().await;
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

async fn serve(
    mut socket: TcpStream,
    reply: Reply,
    recorded: Arc<Mutex<Vec<Recorded>>>,
    closed_by_client: Arc<AtomicUsize>,
) {
    let Some(request) = read_request(&mut socket).await else {
        return;
    };
    recorded.lock().unwrap().push(request);
    match reply {
        Reply::Sse(pieces) => write_sse_head_and_pieces(&mut socket, pieces).await,
        Reply::SseThenStall(pieces) => {
            write_sse_head_and_pieces(&mut socket, pieces).await;
            // Hold the connection open and silent until the client closes it.
            let mut sink = [0u8; 256];
            loop {
                match socket.read(&mut sink).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
            }
            closed_by_client.fetch_add(1, Ordering::SeqCst);
        }
        Reply::Status { code, body } => {
            let response = format!(
                "HTTP/1.1 {code} Status\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = socket.write_all(response.as_bytes()).await;
        }
        Reply::Redirect(location) => {
            let response = format!(
                "HTTP/1.1 302 Found\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            );
            let _ = socket.write_all(response.as_bytes()).await;
        }
    }
}

async fn spawn_stub(reply: Reply) -> Stub {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a loopback port");
    let addr = listener.local_addr().unwrap();
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let connections = Arc::new(AtomicUsize::new(0));
    let closed_by_client = Arc::new(AtomicUsize::new(0));
    let (recorded_for_task, connections_for_task, closed_for_task) = (
        recorded.clone(),
        connections.clone(),
        closed_by_client.clone(),
    );
    tokio::spawn(async move {
        loop {
            let Ok((socket, _)) = listener.accept().await else {
                return;
            };
            connections_for_task.fetch_add(1, Ordering::SeqCst);
            tokio::spawn(serve(
                socket,
                reply.clone(),
                recorded_for_task.clone(),
                closed_for_task.clone(),
            ));
        }
    });
    Stub {
        addr,
        recorded,
        connections,
        closed_by_client,
    }
}

fn sse(events: &[&str]) -> Vec<Vec<u8>> {
    events
        .iter()
        .map(|event| format!("data: {event}\n\n").into_bytes())
        .collect()
}

fn model_for(
    stub: &Stub,
    tweak: impl FnOnce(ChatCompletionsConfig) -> ChatCompletionsConfig,
) -> ChatCompletionsModel {
    ChatCompletionsModel::new(tweak(
        ChatCompletionsConfig::new(stub.base_url(), "test-model").local(true),
    ))
    .expect("the model builds")
}

fn hello() -> ModelRequest {
    ModelRequest {
        system: "Be brief.".into(),
        input: vec![InputItem::User("hi".into())],
        tools: vec![],
        settings: ModelSettings::default(),
    }
}

async fn collect(
    model: &ChatCompletionsModel,
    request: ModelRequest,
) -> Vec<Result<ModelEvent, ModelError>> {
    tokio::time::timeout(
        Duration::from_secs(20),
        model.stream(request).collect::<Vec<_>>(),
    )
    .await
    .expect("the stream must end")
}

fn deltas(events: &[Result<ModelEvent, ModelError>]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            Ok(ModelEvent::TextDelta(text)) => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

fn done(events: &[Result<ModelEvent, ModelError>]) -> &lattice_agents::ModelResponse {
    match events.last() {
        Some(Ok(ModelEvent::Done(response))) => response,
        other => panic!("expected the stream to end with Done, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

#[tokio::test]
async fn text_streams_in_pieces_and_completes_with_the_last_chunks_usage() {
    let stub = spawn_stub(Reply::Sse(sse(&[
        r#"{"choices":[{"delta":{"role":"assistant","content":"Hel"}}]}"#,
        r#"{"choices":[{"delta":{"content":"lo"}}]}"#,
        r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        r#"{"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15}}"#,
        "[DONE]",
    ])))
    .await;
    let model = model_for(&stub, |c| c);
    let events = collect(&model, hello()).await;

    assert_eq!(deltas(&events), "Hello");
    assert!(
        events[..events.len() - 1]
            .iter()
            .all(|e| matches!(e, Ok(ModelEvent::TextDelta(_))))
    );
    let response = done(&events);
    assert_eq!(
        response.output,
        vec![OutputItem::Message {
            text: "Hello".into()
        }]
    );
    assert_eq!(
        response.usage,
        Some(Usage {
            requests: 1,
            input_tokens: 12,
            output_tokens: 3,
            total_tokens: 15
        })
    );
}

#[tokio::test]
async fn tool_calls_split_across_events_and_writes_are_assembled_by_index() {
    // Events are also cut in the middle of a line, as a real network does. The
    // name comes whole in the first delta of a call, as the SDK expects
    // (`chatcmpl_stream_handler.py:1033-1035`): a later non-empty name would
    // replace it, not extend it (see `a_later_non_empty_name_replaces_...`).
    let whole = concat!(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"weather\",\"arguments\":\"\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_b\",\"function\":{\"name\":\"clock\",\"arguments\":\"{}\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"Oslo\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n",
    );
    let bytes = whole.as_bytes();
    let pieces: Vec<Vec<u8>> = bytes.chunks(37).map(<[u8]>::to_vec).collect();
    let stub = spawn_stub(Reply::Sse(pieces)).await;
    let events = collect(&model_for(&stub, |c| c), hello()).await;
    let response = done(&events);
    assert_eq!(
        response.output,
        vec![
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
        response.usage, None,
        "no usage chunk means no usage, not zero usage"
    );
}

#[tokio::test]
async fn reasoning_text_is_reported_apart_from_the_answer() {
    let stub = spawn_stub(Reply::Sse(sse(&[
        r#"{"choices":[{"delta":{"reasoning_content":"thinking"}}]}"#,
        r#"{"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}"#,
        "[DONE]",
    ])))
    .await;
    let events = collect(&model_for(&stub, |c| c), hello()).await;
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Ok(ModelEvent::ReasoningDelta(text)) if text == "thinking"))
    );
    assert_eq!(deltas(&events), "answer");
    assert_eq!(
        done(&events).output,
        vec![
            OutputItem::Reasoning {
                text: "thinking".into()
            },
            OutputItem::Message {
                text: "answer".into()
            }
        ]
    );
}

#[tokio::test]
async fn a_stream_that_stops_before_the_model_finished_is_an_error_not_a_short_answer() {
    let stub = spawn_stub(Reply::Sse(sse(&[
        r#"{"choices":[{"delta":{"content":"half an ans"}}]}"#,
    ])))
    .await;
    let events = collect(&model_for(&stub, |c| c), hello()).await;
    assert_eq!(deltas(&events), "half an ans");
    match events.last().unwrap() {
        Err(ModelError::Protocol(reason)) => {
            assert!(reason.contains("before the model finished"), "{reason}")
        }
        other => panic!("{other:?}"),
    }
    assert!(!events.iter().any(|e| matches!(e, Ok(ModelEvent::Done(_)))));
}

#[tokio::test]
async fn a_finished_stream_without_done_still_completes() {
    let stub = spawn_stub(Reply::Sse(sse(&[
        r#"{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}"#,
    ])))
    .await;
    let events = collect(&model_for(&stub, |c| c), hello()).await;
    assert_eq!(
        done(&events).output,
        vec![OutputItem::Message { text: "ok".into() }]
    );
}

#[tokio::test]
async fn a_malformed_chunk_is_a_protocol_error_that_does_not_repeat_it() {
    let stub = spawn_stub(Reply::Sse(sse(&[
        r#"{"choices": [oops secret-in-chunk"#,
        "[DONE]",
    ])))
    .await;
    let events = collect(&model_for(&stub, |c| c), hello()).await;
    match events.last().unwrap() {
        Err(error @ ModelError::Protocol(_)) => {
            assert!(!error.to_string().contains("secret-in-chunk"))
        }
        other => panic!("{other:?}"),
    }
}

// ---------------------------------------------------------------------------
// The request
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_request_is_a_streaming_chat_completion_with_the_documented_shape() {
    let stub = spawn_stub(Reply::Sse(sse(&[
        r#"{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}"#,
        "[DONE]",
    ])))
    .await;
    let request = ModelRequest {
        system: "Be brief.".into(),
        input: vec![
            InputItem::User("Weather?".into()),
            InputItem::Assistant {
                text: Some("Checking.".into()),
                tool_calls: vec![],
            },
            InputItem::Assistant {
                text: None,
                tool_calls: vec![ToolCallItem {
                    call_id: "c1".into(),
                    name: "weather".into(),
                    arguments: "{\"city\":\"Oslo\"}".into(),
                }],
            },
            InputItem::ToolResult {
                call_id: "c1".into(),
                output: "sunny".into(),
            },
        ],
        tools: vec![ToolSpec {
            name: "weather".into(),
            description: "The weather.".into(),
            parameters: json!({"type": "object", "properties": {"city": {"type": "string"}}}),
            strict: true,
        }],
        settings: ModelSettings {
            temperature: Some(0.3),
            top_p: Some(0.9),
            max_tokens: Some(128),
            tool_choice: Some("auto".into()),
            parallel_tool_calls: Some(true),
            include_usage: None,
        },
    };
    collect(
        &model_for(&stub, |c| c.with_api_key("test-key-123")),
        request,
    )
    .await;

    let recorded = stub.requests();
    assert_eq!(recorded.len(), 1);
    let request = &recorded[0];
    assert_eq!(request.request_line, "POST /v1/chat/completions HTTP/1.1");
    assert_eq!(request.header("authorization"), Some("Bearer test-key-123"));
    assert_eq!(request.header("content-type"), Some("application/json"));
    assert_eq!(request.header("accept"), Some("text/event-stream"));
    assert!(
        request
            .header("user-agent")
            .unwrap_or("")
            .starts_with("Lattice/"),
        "{:?}",
        request.header("user-agent")
    );

    let body = request.json();
    assert_eq!(body["model"], "test-model");
    assert_eq!(body["stream"], json!(true));
    assert_eq!(body["stream_options"], json!({"include_usage": true}));
    assert_eq!(body["temperature"], json!(0.3));
    assert_eq!(body["top_p"], json!(0.9));
    assert_eq!(body["max_tokens"], json!(128));
    assert_eq!(body["tool_choice"], "auto");
    assert_eq!(body["parallel_tool_calls"], json!(true));
    assert_eq!(
        body["tools"],
        json!([{
            "type": "function",
            "function": {
                "name": "weather",
                "description": "The weather.",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                "strict": true
            }
        }])
    );
    assert_eq!(
        body["messages"],
        json!([
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "Checking.", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "weather", "arguments": "{\"city\":\"Oslo\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        ])
    );
}

#[tokio::test]
async fn without_tools_there_are_no_tool_fields_and_without_a_key_no_authorization_header() {
    let stub = spawn_stub(Reply::Sse(sse(&[
        r#"{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}"#,
        "[DONE]",
    ])))
    .await;
    let mut request = hello();
    request.settings.tool_choice = Some("required".into());
    request.settings.parallel_tool_calls = Some(false);
    request.settings.include_usage = Some(false);
    collect(&model_for(&stub, |c| c), request).await;
    let recorded = stub.requests();
    let body = recorded[0].json();
    for absent in [
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "stream_options",
        "temperature",
    ] {
        assert!(
            body.get(absent).is_none(),
            "{absent} must be omitted: {body}"
        );
    }
    assert!(recorded[0].header("authorization").is_none());
}

// ---------------------------------------------------------------------------
// Failure, and what it must not say
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_redirect_is_not_followed_and_the_key_goes_nowhere_else() {
    let elsewhere = spawn_stub(Reply::Sse(sse(&["[DONE]"]))).await;
    let redirector = spawn_stub(Reply::Redirect(format!(
        "http://{}/v1/chat/completions",
        elsewhere.addr
    )))
    .await;
    let events = collect(
        &model_for(&redirector, |c| c.with_api_key("test-key-123")),
        hello(),
    )
    .await;

    assert_eq!(events.len(), 1);
    assert!(
        matches!(&events[0], Err(ModelError::Status(302))),
        "{events:?}"
    );
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(
        elsewhere.connections.load(Ordering::SeqCst),
        0,
        "the redirect target was never contacted"
    );
    assert!(elsewhere.requests().is_empty());
}

#[tokio::test]
async fn a_server_error_is_a_status_and_says_nothing_of_its_body() {
    let stub = spawn_stub(Reply::Status {
        code: 500,
        body: "internal secret-body-text with sk-live-999",
    })
    .await;
    let events = collect(
        &model_for(&stub, |c| c.with_api_key("test-key-123")),
        hello(),
    )
    .await;
    assert_eq!(events.len(), 1);
    let error = events[0].as_ref().unwrap_err();
    assert_eq!(*error, ModelError::Status(500));
    for text in [error.to_string(), format!("{error:?}")] {
        assert!(
            !text.contains("secret-body-text")
                && !text.contains("sk-live-999")
                && !text.contains("test-key-123"),
            "{text}"
        );
    }
    let stub = spawn_stub(Reply::Status {
        code: 401,
        body: "bad key test-key-123",
    })
    .await;
    let events = collect(
        &model_for(&stub, |c| c.with_api_key("test-key-123")),
        hello(),
    )
    .await;
    assert_eq!(*events[0].as_ref().unwrap_err(), ModelError::Status(401));
}

#[tokio::test]
async fn a_stalled_stream_times_out_after_the_read_timeout() {
    let stub = spawn_stub(Reply::SseThenStall(sse(&[
        r#"{"choices":[{"delta":{"content":"start"}}]}"#,
    ])))
    .await;
    let model = model_for(&stub, |c| {
        c.with_timeouts(Duration::from_secs(5), Duration::from_millis(300))
    });
    let started = std::time::Instant::now();
    let events = collect(&model, hello()).await;
    assert_eq!(deltas(&events), "start");
    assert_eq!(
        *events.last().unwrap().as_ref().unwrap_err(),
        ModelError::Timeout
    );
    assert!(started.elapsed() < Duration::from_secs(10));
}

#[tokio::test]
async fn an_unreachable_server_is_a_connection_error_that_names_no_address() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let model = ChatCompletionsModel::new(
        ChatCompletionsConfig::new(format!("http://{addr}/v1"), "m")
            .local(true)
            .with_api_key("test-key-123"),
    )
    .unwrap();
    let events = collect(&model, hello()).await;
    let error = events[0].as_ref().unwrap_err();
    assert!(matches!(error, ModelError::Connection(_)), "{error:?}");
    for text in [error.to_string(), format!("{error:?}")] {
        assert!(
            !text.contains("127.0.0.1")
                && !text.contains(&addr.port().to_string())
                && !text.contains("test-key-123"),
            "{text}"
        );
    }
}

#[tokio::test]
async fn the_key_never_appears_in_debug_output_of_the_model_or_its_config() {
    let config = ChatCompletionsConfig::new("http://127.0.0.1:1/v1", "m")
        .with_api_key("test-key-123")
        .local(true);
    assert!(!format!("{config:?}").contains("test-key-123"));
    let model = ChatCompletionsModel::new(config).unwrap();
    assert!(!format!("{model:?}").contains("test-key-123"));
    assert_eq!(
        model.config_for_trace(),
        json!({"base_url": "http://127.0.0.1:1/v1"})
    );
    assert_eq!(model.name(), "m");
}

#[tokio::test]
async fn dropping_the_stream_closes_the_connection() {
    let stub = spawn_stub(Reply::SseThenStall(sse(&[
        r#"{"choices":[{"delta":{"content":"a"}}]}"#,
    ])))
    .await;
    let model = model_for(&stub, |c| c);
    let mut stream = model.stream(hello());
    let first = tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .unwrap();
    assert!(matches!(first, Some(Ok(ModelEvent::TextDelta(_)))));
    assert_eq!(
        stub.closed_by_client.load(Ordering::SeqCst),
        0,
        "the server is still holding the request open"
    );

    drop(stream);
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while stub.closed_by_client.load(Ordering::SeqCst) == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "the client never closed the abandoned connection"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        stub.connections.load(Ordering::SeqCst),
        1,
        "abandoning a request does not retry it"
    );
}

// ---------------------------------------------------------------------------
// Proxies: a "local" model must never go through one
// ---------------------------------------------------------------------------

/// Counts connections and drops them: a proxy that would not have worked.
async fn spawn_black_hole_proxy(port: u16) -> Arc<AtomicUsize> {
    let listener = TcpListener::bind(("127.0.0.1", port))
        .await
        .expect("bind the proxy port");
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = hits.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            counter.fetch_add(1, Ordering::SeqCst);
            let mut sink = [0u8; 1024];
            let _ = socket.read(&mut sink).await;
        }
    });
    hits
}

/// The body of the proxy test. It runs in a child process, because the proxy
/// is configured through the environment and changing the environment of a
/// running multi-threaded test binary is not sound. Ignored so a normal run
/// skips it; the parent test below runs it explicitly.
#[tokio::test]
#[ignore = "runs only as the child of local_models_ignore_a_configured_proxy"]
async fn proxy_child() {
    let stub = spawn_stub(Reply::Sse(sse(&[
        r#"{"choices":[{"delta":{"content":"direct"},"finish_reason":"stop"}]}"#,
        "[DONE]",
    ])))
    .await;
    let port: u16 = std::env::var("LATTICE_PROXY_TEST_PORT")
        .expect("set by the parent")
        .parse()
        .unwrap();
    let hits = spawn_black_hole_proxy(port).await;

    // Control: a remote model honours the proxy, so the test can see it work.
    let remote =
        ChatCompletionsModel::new(ChatCompletionsConfig::new(stub.base_url(), "m").local(false))
            .unwrap();
    let _ = collect(&remote, hello()).await;
    assert!(
        hits.load(Ordering::SeqCst) >= 1,
        "control failed: the environment proxy was not used by a non-local model"
    );
    assert!(
        stub.requests().is_empty(),
        "control failed: the request should have gone to the proxy"
    );

    // The property: a local model goes straight to its server.
    let before = hits.load(Ordering::SeqCst);
    let local =
        ChatCompletionsModel::new(ChatCompletionsConfig::new(stub.base_url(), "m").local(true))
            .unwrap();
    let events = collect(&local, hello()).await;
    assert_eq!(deltas(&events), "direct");
    assert_eq!(
        hits.load(Ordering::SeqCst),
        before,
        "a local model must not use the proxy"
    );
    assert_eq!(stub.requests().len(), 1);
}

#[test]
fn local_models_ignore_a_configured_proxy() {
    // The child listens on this port as a proxy that drops what it receives, and is told by
    // the environment (the only channel that reaches the HTTP client) to use it.
    let port = {
        let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        probe.local_addr().unwrap().port()
    };
    let exe = std::env::current_exe().unwrap();
    let output = std::process::Command::new(exe)
        .args([
            "--ignored",
            "--exact",
            "proxy_child",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("HTTP_PROXY", format!("http://127.0.0.1:{port}"))
        .env("http_proxy", format!("http://127.0.0.1:{port}"))
        .env_remove("NO_PROXY")
        .env_remove("no_proxy")
        .env("LATTICE_PROXY_TEST_PORT", port.to_string())
        .output()
        .expect("the child test process starts");
    assert!(
        output.status.success(),
        "child failed\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

// ---------------------------------------------------------------------------
// The response cap on tool-call deltas, and the SDK's response semantics
// (`chatcmpl_stream_handler.py`, `turn_resolution.py`), as an adversarial
// review found them missing.
// ---------------------------------------------------------------------------

/// A run of one agent (with an `echo` tool) over a model that talks to `stub`.
async fn run_against(stub: &Stub) -> Result<RunResult, RunError> {
    let agent = Agent::builder("A")
        .instructions("Be brief.")
        .tool(FunctionTool::new(
            "echo",
            "Say something back.",
            strict_object_schema(json!({}), &[]),
            |_, _| async { Ok("echoed".to_owned()) },
        ))
        .build();
    let handle = run_streamed(
        agent,
        "hi".to_owned(),
        RunConfig::new(Arc::new(model_for(stub, |c| c))),
    );
    tokio::time::timeout(Duration::from_secs(60), handle.result)
        .await
        .expect("the run must end")
        .expect("the run task must not panic")
}

fn protocol_reason(events: &[Result<ModelEvent, ModelError>]) -> &str {
    match events.last() {
        Some(Err(ModelError::Protocol(reason))) => reason,
        other => panic!("expected a Protocol error, got {other:?}"),
    }
}

#[tokio::test]
async fn tool_call_ids_count_toward_the_response_cap() {
    // Six events of a 7 MiB id each: every event is under the event cap, the
    // total (42 MiB) is over the response cap. Before the fix nothing counted
    // an id, and the stream simply completed.
    let id = "i".repeat(7 * 1024 * 1024);
    let events: Vec<String> = (0..6)
        .map(|index| {
            format!(
                r#"{{"choices":[{{"delta":{{"tool_calls":[{{"index":{index},"id":"{id}"}}]}}}}]}}"#
            )
        })
        .collect();
    let mut wire: Vec<&str> = events.iter().map(String::as_str).collect();
    wire.push("[DONE]");
    let stub = spawn_stub(Reply::Sse(sse(&wire))).await;
    let events = collect(&model_for(&stub, |c| c), hello()).await;
    assert_eq!(protocol_reason(&events), "the response is too large");
}

#[tokio::test]
async fn tool_call_entries_cost_something_and_distinct_indices_are_capped_at_128() {
    // 200 bare entries in one event: no name, no arguments, no id.
    let entries: Vec<String> = (0..200)
        .map(|index| format!(r#"{{"index":{index}}}"#))
        .collect();
    let burst = format!(
        r#"{{"choices":[{{"delta":{{"tool_calls":[{}]}}}}]}}"#,
        entries.join(",")
    );
    let stub = spawn_stub(Reply::Sse(sse(&[&burst, "[DONE]"]))).await;
    let events = collect(&model_for(&stub, |c| c), hello()).await;
    assert_eq!(
        protocol_reason(&events),
        "the response has too many tool calls"
    );

    // The same across events.
    let chunks: Vec<String> = (0..200)
        .map(|index| {
            format!(r#"{{"choices":[{{"delta":{{"tool_calls":[{{"index":{index}}}]}}}}]}}"#)
        })
        .collect();
    let mut wire: Vec<&str> = chunks.iter().map(String::as_str).collect();
    wire.push("[DONE]");
    let stub = spawn_stub(Reply::Sse(sse(&wire))).await;
    let events = collect(&model_for(&stub, |c| c), hello()).await;
    assert_eq!(
        protocol_reason(&events),
        "the response has too many tool calls"
    );

    // Exactly 128 real calls are fine.
    let chunks: Vec<String> = (0..128)
        .map(|index| {
            format!(
                r#"{{"choices":[{{"delta":{{"tool_calls":[{{"index":{index},"id":"c{index}","function":{{"name":"t","arguments":"{{}}"}}}}]}}}}]}}"#
            )
        })
        .collect();
    let mut wire: Vec<&str> = chunks.iter().map(String::as_str).collect();
    wire.push(r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#);
    wire.push("[DONE]");
    let stub = spawn_stub(Reply::Sse(sse(&wire))).await;
    let events = collect(&model_for(&stub, |c| c), hello()).await;
    assert_eq!(done(&events).output.len(), 128);
}

#[tokio::test]
async fn a_repeated_tool_call_name_stays_one_name_and_the_latest_id_wins() {
    // A server that repeats `function.name` in every delta of one call: the SDK
    // (`chatcmpl_stream_handler.py:1033-1041`) assigns it, and keeps the latest id.
    let stub = spawn_stub(Reply::Sse(sse(&[
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_first","function":{"name":"calculate","arguments":"{\"expression\":"}}]}}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_second","function":{"name":"calculate","arguments":"\"6 * 7\"}"}}]}}]}"#,
        r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
        "[DONE]",
    ])))
    .await;
    let events = collect(&model_for(&stub, |c| c), hello()).await;
    assert_eq!(
        done(&events).output,
        vec![OutputItem::FunctionCall {
            call_id: "call_second".into(),
            name: "calculate".into(),
            arguments: "{\"expression\":\"6 * 7\"}".into(),
        }]
    );
}

#[tokio::test]
async fn a_later_non_empty_name_replaces_the_earlier_one_and_an_empty_one_does_not() {
    let stub = spawn_stub(Reply::Sse(sse(&[
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"wea","arguments":""}}]}}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"ther","arguments":"{}"}}]}}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"","arguments":""}}]},"finish_reason":"tool_calls"}]}"#,
        "[DONE]",
    ])))
    .await;
    let events = collect(&model_for(&stub, |c| c), hello()).await;
    match &done(&events).output[..] {
        [OutputItem::FunctionCall { name, .. }] => assert_eq!(name, "ther"),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn only_the_choice_with_index_zero_is_read() {
    let stub = spawn_stub(Reply::Sse(sse(&[
        r#"{"choices":[{"index":1,"delta":{"content":"WRONG "}},{"index":0,"delta":{"content":"right"}}]}"#,
        r#"{"choices":[{"index":1,"delta":{"content":" WRONG"},"finish_reason":"stop"}]}"#,
        r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        "[DONE]",
    ])))
    .await;
    let events = collect(&model_for(&stub, |c| c), hello()).await;
    assert_eq!(deltas(&events), "right");
    assert_eq!(
        done(&events).output,
        vec![OutputItem::Message {
            text: "right".into()
        }]
    );

    // A server that never numbers its choices is read from its first entry.
    let stub = spawn_stub(Reply::Sse(sse(&[
        r#"{"choices":[{"delta":{"content":"plain"},"finish_reason":"stop"}]}"#,
        "[DONE]",
    ])))
    .await;
    let events = collect(&model_for(&stub, |c| c), hello()).await;
    assert_eq!(deltas(&events), "plain");
}

#[tokio::test]
async fn a_refusal_fails_the_run_instead_of_completing_with_nothing() {
    // `delta.refusal` builds a refusal part and the SDK raises ModelRefusalError
    // (`chatcmpl_stream_handler.py:944-990`, `turn_resolution.py:952-988`).
    let stub = spawn_stub(Reply::Sse(sse(&[
        r#"{"choices":[{"index":0,"delta":{"refusal":"I can't "}}]}"#,
        r#"{"choices":[{"index":0,"delta":{"refusal":"help with that."}}]}"#,
        r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        "[DONE]",
    ])))
    .await;
    let error = run_against(&stub)
        .await
        .expect_err("a refusal is not an empty final output");
    assert_eq!(error.sdk_name(), "ModelRefusalError");
    assert_eq!(
        error.to_string(),
        "Model refused to produce output: I can't help with that."
    );
}

#[tokio::test]
async fn a_content_filtered_response_with_no_output_is_a_refusal() {
    // `chatcmpl_stream_handler.py:1133-1180`.
    let stub = spawn_stub(Reply::Sse(sse(&[
        r#"{"choices":[{"index":0,"delta":{"content":""},"finish_reason":"content_filter"}]}"#,
        "[DONE]",
    ])))
    .await;
    let error = run_against(&stub)
        .await
        .expect_err("withheld, not answered");
    assert_eq!(error.sdk_name(), "ModelRefusalError");
    assert!(
        error
            .to_string()
            .contains("Response withheld by the provider's content filter."),
        "{error}"
    );

    // A content filter that still emitted text is left as it is.
    let stub = spawn_stub(Reply::Sse(sse(&[
        r#"{"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":"content_filter"}]}"#,
        "[DONE]",
    ])))
    .await;
    let result = run_against(&stub).await.expect("text was produced");
    assert_eq!(result.final_output, "partial");
}

#[tokio::test]
async fn running_out_of_tokens_before_any_answer_fails_the_run() {
    // A reasoning model that spends all of `max_tokens` thinking
    // (`chatcmpl_stream_handler.py:1190-1205`): ModelBehaviorError.
    let stub = spawn_stub(Reply::Sse(sse(&[
        r#"{"choices":[{"index":0,"delta":{"reasoning_content":"hmm, let me think"}}]}"#,
        r#"{"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}"#,
        "[DONE]",
    ])))
    .await;
    let error = run_against(&stub)
        .await
        .expect_err("no answer is not a completed run");
    assert_eq!(error.sdk_name(), "ModelBehaviorError");
    assert!(
        error.to_string().contains("finish_reason='length'"),
        "{error}"
    );

    // Truncated after some text: the text is the answer, as in the SDK.
    let stub = spawn_stub(Reply::Sse(sse(&[
        r#"{"choices":[{"index":0,"delta":{"content":"The answer is"},"finish_reason":"length"}]}"#,
        "[DONE]",
    ])))
    .await;
    assert_eq!(
        run_against(&stub)
            .await
            .expect("text was produced")
            .final_output,
        "The answer is"
    );
}

#[tokio::test]
async fn a_call_without_an_id_is_refused_not_given_an_invented_one() {
    // `tool_planning.py:406-410`: "Tool invocations require a non-empty string
    // call ID before execution." The server here never sends an id.
    let stub = spawn_stub(Reply::Sse(sse(&[
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"echo","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#,
        "[DONE]",
    ])))
    .await;
    let error = run_against(&stub).await.expect_err("an id is required");
    assert_eq!(error.sdk_name(), "ModelBehaviorError");
    assert_eq!(
        error.to_string(),
        "Tool invocations require a non-empty string call ID before execution."
    );
    // The model was asked once: nothing ran, so nothing was sent back.
    assert_eq!(stub.requests().len(), 1);
}
