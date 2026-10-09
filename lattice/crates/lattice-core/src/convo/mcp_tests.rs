//! An agent turn with an MCP server (the chat core's spec §12, "Calls";
//! §9.2's Mcp rows): the server's tools are offered in Agent mode only, every
//! call asks (the card, then the native `McpCall` dialog), a no runs nothing,
//! "allow always" goes through `AllowMcpTool`, an allowed tool runs without
//! asking under the allowlist's rule, and a server that does not start is a
//! notice. The server is the real stub process (`lattice-mcp-stub`).

#![cfg(windows)]

use lattice_protocol::conversation::{
    AgentChatService, ApprovalDetail, ApprovalKind, ConversationEvent, ConversationEventKind,
    DecidedBy, Decision, Mode,
};
use serde_json::json;

use super::agent_tests::{H, call, local, say};
use crate::mcp::config::{self, ServerKey};
use crate::ports::ConfirmRequest;

/// Declare the stub as `stub` in the reader's file, with `args`, and enable
/// it (the recording dialog answers yes).
fn declare(h: &H, args: &[&str]) -> ServerKey {
    let stub = crate::mcp::hub_tests::stub();
    config::put_user_server(
        &h.state,
        "stub",
        &json!({"command": stub.display().to_string(), "args": args}),
    )
    .unwrap();
    let key = ServerKey::user("stub");
    assert!(
        h.runtime
            .block_on(h.chat.mcp_enable(key.clone(), None))
            .unwrap()
    );
    key
}

fn approval(events: &[ConversationEvent]) -> Option<(String, ApprovalDetail, bool)> {
    events.iter().find_map(|event| match &event.kind {
        ConversationEventKind::ApprovalRequested {
            call_id,
            kind: ApprovalKind::Mcp,
            detail,
            allow_always_offer,
        } => Some((call_id.clone(), detail.clone(), *allow_always_offer)),
        _ => None,
    })
}

fn outputs(events: &[ConversationEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match &event.kind {
            ConversationEventKind::ToolOutput { preview, .. } => Some(preview.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn an_mcp_tool_is_offered_in_agent_mode_and_its_call_asks_on_the_card_then_in_the_dialog() {
    let h = H::new("mcp-turn");
    let workspace = h.workspace();
    declare(&h, &[]);
    let model = h.script(vec![
        call("mcp__stub__echo", json!({"text": "hello there"}), "c1"),
        say("It said hello."),
    ]);
    let id = h.agent(None, "Use the echo tool.", &workspace);
    let events = h.events_until(&id, |events| approval(events).is_some());
    let (call_id, detail, offer) = approval(&events).unwrap();
    assert_eq!(call_id, "c1");
    assert!(offer, "allow always is offered");
    let ApprovalDetail::Mcp {
        server,
        tool,
        arguments_preview,
    } = detail
    else {
        panic!("{detail:?}")
    };
    assert_eq!((server.as_str(), tool.as_str()), ("stub", "echo"));
    assert!(
        arguments_preview.contains("hello there"),
        "{arguments_preview}"
    );
    let summary = events.iter().find_map(|event| match &event.kind {
        ConversationEventKind::ToolCall { summary, .. } => Some(summary.clone()),
        _ => None,
    });
    assert!(summary.unwrap().starts_with("echo (stub) "));
    h.runtime
        .block_on(h.chat.decide(&id, "c1", Decision::Approve))
        .unwrap();
    let events = h.turns_end(&id, 1);
    assert_eq!(outputs(&events), ["hello there"]);
    let asked = h.confirm.asked();
    assert!(
        asked.iter().any(|request| matches!(
            request,
            ConfirmRequest::McpCall { server, tool, .. } if server == "stub" && tool == "echo"
        )),
        "{asked:?}"
    );
    let requests = model.calls();
    let first = &requests[0];
    assert!(
        first
            .tools
            .iter()
            .any(|tool| tool.name == "mcp__stub__echo")
    );
    assert!(first.system.ends_with(crate::convo::prompt_agent::MCP_NOTE));
    assert!(requests[1].input.iter().any(|item| matches!(
        item,
        lattice_agents::model::InputItem::ToolResult { output, .. } if output == "hello there"
    )));
}

#[test]
fn ask_mode_offers_no_mcp_tool() {
    let h = H::new("mcp-ask");
    let workspace = h.workspace();
    declare(&h, &[]);
    let model = h.script(vec![say("Nothing to do.")]);
    let id = match h
        .send(
            None,
            "Look around.",
            "local",
            local(),
            Mode::Ask,
            Some(&workspace),
        )
        .unwrap()
    {
        lattice_protocol::conversation::Accepted::Started { conversation, .. } => conversation.id,
        other => panic!("{other:?}"),
    };
    h.turns_end(&id, 1);
    let first = &model.calls()[0];
    assert!(
        !first
            .tools
            .iter()
            .any(|tool| tool.name.starts_with("mcp__"))
    );
    assert!(!first.system.contains("mcp__"));
}

/// A no in the native dialog is a Reject with no note (CP3): the server is
/// never called, and the model hears it was not approved.
#[test]
fn a_no_in_the_dialog_runs_nothing() {
    let h = H::new("mcp-no");
    let workspace = h.workspace();
    let record = h.scratch.path().join("calls.record");
    declare(&h, &["--record", &record.display().to_string()]);
    h.script(vec![
        call("mcp__stub__echo", json!({"text": "x"}), "c1"),
        say("Understood."),
    ]);
    let id = h.agent(None, "Use echo.", &workspace);
    h.events_until(&id, |events| approval(events).is_some());
    *h.confirm.answer.lock().unwrap() = false;
    h.runtime
        .block_on(h.chat.decide(&id, "c1", Decision::Approve))
        .unwrap();
    let events = h.turns_end(&id, 1);
    let resolved = events.iter().find_map(|event| match &event.kind {
        ConversationEventKind::ApprovalResolved { approved, by, .. } => {
            Some((*approved, by.clone()))
        }
        _ => None,
    });
    assert_eq!(resolved, Some((false, DecidedBy::Reader)));
    let methods = std::fs::read_to_string(&record).unwrap_or_default();
    assert!(!methods.contains("tools/call"), "{methods}");
}

/// "Allow always" from the card: the `AllowMcpTool` dialog, then this call
/// runs; the next call of that tool runs under the allowlist's rule, with
/// no card and no dialog.
#[test]
fn allow_always_runs_this_call_and_the_next_one_without_asking() {
    let h = H::new("mcp-allow");
    let workspace = h.workspace();
    let key = declare(&h, &[]);
    h.script(vec![
        call("mcp__stub__echo", json!({"text": "one"}), "c1"),
        call("mcp__stub__echo", json!({"text": "two"}), "c2"),
        say("Twice."),
    ]);
    let id = h.agent(None, "Echo twice.", &workspace);
    h.events_until(&id, |events| approval(events).is_some());
    h.runtime
        .block_on(h.chat.allow_mcp_always(&id, "c1"))
        .unwrap();
    let events = h.turns_end(&id, 1);
    assert_eq!(outputs(&events), ["one", "two"]);
    let asks: Vec<&ConversationEvent> = events
        .iter()
        .filter(|event| matches!(event.kind, ConversationEventKind::ApprovalRequested { .. }))
        .collect();
    assert_eq!(asks.len(), 1, "only the first call asked");
    let by_rule = events.iter().any(|event| matches!(
        &event.kind,
        ConversationEventKind::ApprovalResolved { call_id, approved: true, by: DecidedBy::Policy { rule } }
            if call_id == "c2" && rule == "mcp allowlist"
    ));
    assert!(by_rule, "{events:#?}");
    let asked = h.confirm.asked();
    assert!(
        asked
            .iter()
            .any(|r| matches!(r, ConfirmRequest::AllowMcpTool { tool, .. } if tool == "echo"))
    );
    assert!(
        !asked
            .iter()
            .any(|r| matches!(r, ConfirmRequest::McpCall { .. })),
        "no call dialog"
    );
    let entry = config::load_user(&h.state).servers.remove(0);
    assert!(h.chat.mcp().approvals().allowed(&entry, "echo"));
    assert_eq!(entry.key, key);
}

#[test]
fn a_server_that_does_not_start_is_a_notice_and_the_turn_goes_on() {
    let h = H::new("mcp-dead");
    let workspace = h.workspace();
    declare(&h, &["--exit", "4"]);
    let model = h.script(vec![say("Done without it.")]);
    let id = h.agent(None, "Try.", &workspace);
    let events = h.turns_end(&id, 1);
    let notice = events.iter().find_map(|event| match &event.kind {
        ConversationEventKind::Notice { text } if text.contains("did not start") => {
            Some(text.clone())
        }
        _ => None,
    });
    assert!(notice.unwrap().contains("stub: exiting with 4"));
    assert!(
        !model.calls()[0]
            .tools
            .iter()
            .any(|tool| tool.name.starts_with("mcp__"))
    );
    assert!(!model.calls()[0].system.contains("mcp__"));
}

/// The Tools view's calls through the service: the overview, a stop, and a
/// server no longer enabled after its approval ends.
#[test]
fn the_tools_view_sees_and_steers_the_servers() {
    let h = H::new("mcp-view");
    let key = declare(&h, &[]);
    h.runtime
        .block_on(h.chat.mcp_start(key.clone(), None))
        .unwrap();
    let overview = h.runtime.block_on(h.chat.mcp_overview(None));
    let view = &overview.servers[0];
    assert_eq!(view.status, crate::mcp::ServerStatus::Running);
    assert_eq!(view.tools.len(), 8);
    h.runtime.block_on(h.chat.mcp_stop(key.clone()));
    let overview = h.runtime.block_on(h.chat.mcp_overview(None));
    assert_eq!(
        overview.servers[0].status,
        crate::mcp::ServerStatus::Stopped
    );
    h.runtime.block_on(h.chat.mcp_disable(key.clone())).unwrap();
    let overview = h.runtime.block_on(h.chat.mcp_overview(None));
    assert!(!overview.servers[0].enabled);
    assert!(h.runtime.block_on(h.chat.mcp_start(key, None)).is_err());
}
