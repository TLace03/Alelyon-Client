//! A plugin's MCP servers copied into the reader's own settings
//! (`AgentChat::plugin_mcp_copy`): its folder filled in, a name the settings
//! hold already left as it is, and each copied server enabled only through the
//! core's own `EnableMcpServer` dialog. The server is the real stub's program
//! (`lattice-mcp-stub`); nothing starts here.

#![cfg(windows)]

use serde_json::json;

use super::agent_tests::H;
use crate::mcp::config;
use crate::ports::ConfirmRequest;

fn write(path: &std::path::Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

#[test]
fn a_plugins_servers_are_copied_enabled_through_the_dialog_and_never_over_the_readers_own() {
    let h = H::new("plugins-copy");
    let stub = crate::mcp::hub_tests::stub().display().to_string();
    // The reader's own server named `taken`.
    config::put_user_server(
        &h.state,
        "taken",
        &json!({"command": stub, "args": ["mine"]}),
    )
    .unwrap();
    let folder = h.scratch.path().join("tools-plugin");
    write(
        &folder.join(".claude-plugin").join("plugin.json"),
        r#"{"name": "tools"}"#,
    );
    write(
        &folder.join(".mcp.json"),
        &json!({"mcpServers": {
            "copied": {"command": stub, "args": ["--root", "${CLAUDE_PLUGIN_ROOT}"]},
            "taken": {"command": stub, "args": ["theirs"]},
        }})
        .to_string(),
    );
    h.runtime
        .block_on(h.chat.plugin_add_native(folder.clone()))
        .unwrap();
    let copied = h
        .runtime
        .block_on(h.chat.plugin_mcp_copy("tools".into(), None))
        .unwrap();
    assert_eq!(copied.enabled, ["copied"]);
    assert_eq!(copied.kept, ["taken"]);
    assert!(copied.not_enabled.is_empty());
    // The reader's own `taken` is as it was; `copied` names the plugin's folder.
    let taken = h
        .runtime
        .block_on(h.chat.mcp_masked("taken".into()))
        .unwrap();
    assert_eq!(taken["args"], json!(["mine"]));
    let mine = h
        .runtime
        .block_on(h.chat.mcp_masked("copied".into()))
        .unwrap();
    assert_eq!(
        mine["args"],
        json!(["--root", folder.display().to_string()])
    );
    // Enabled only through the core's own dialog, which named it.
    let asked: Vec<String> = h
        .confirm
        .asked()
        .into_iter()
        .filter_map(|request| match request {
            ConfirmRequest::EnableMcpServer { name, .. } => Some(name),
            _ => None,
        })
        .collect();
    assert_eq!(asked, ["copied"]);
}
