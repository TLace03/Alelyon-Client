//! Finding connectors (`super::connectors`), end to end over the agent
//! harness: the catalog is searched by words; a suggestion asks the user with
//! the server's command and adds it to their MCP settings only on a yes; a
//! no adds nothing; a name already there is never replaced; and a server the
//! catalog does not hold is refused.

use lattice_agents::model::{InputItem, ModelRequest};
use serde_json::json;

use super::agent_tests::{H, call, say};
use super::connectors::{CATALOG, find};
use crate::mcp::config;
use crate::ports::ConfirmRequest;

fn result_of(calls: &[ModelRequest], call_id: &str) -> String {
    calls
        .iter()
        .flat_map(|request| request.input.iter())
        .find_map(|item| match item {
            InputItem::ToolResult { call_id: id, output } if id == call_id => Some(output.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no result for {call_id}"))
}

#[test]
fn the_catalog_is_searched_by_words_best_first() {
    assert_eq!(find("").len(), CATALOG.len());
    assert_eq!(find("browser automation")[0].id, "playwright");
    assert_eq!(find("time zone")[0].id, "time");
    assert!(find("spreadsheet").is_empty());
    for connector in CATALOG {
        assert!(
            config::entry_of(connector.id, &json!({"command": connector.command, "args": connector.args}), &config::Scope::User, "test").is_ok(),
            "{} is an entry Lattice uses",
            connector.id
        );
    }
}

#[test]
fn a_connector_is_added_to_the_users_settings_only_after_their_yes() {
    let h = H::new("connectors");
    let ws = h.workspace();
    let model = h.script(vec![
        call("find_connectors", json!({"query": "browser"}), "c1"),
        call("suggest_connector", json!({"id": "playwright", "why": "To test the site in a browser."}), "c2"),
        call("suggest_connector", json!({"id": "playwright", "why": "Again."}), "c3"),
        call("find_connectors", json!({"query": "playwright"}), "c4"),
        call("suggest_connector", json!({"id": "nosuch", "why": "x"}), "c5"),
        say("Added it."),
    ]);
    let id = h.agent(None, "help me test the site", &ws);
    h.turns_end(&id, 1);
    let calls = model.calls();
    assert!(result_of(&calls, "c1").starts_with("playwright: "), "{}", result_of(&calls, "c1"));
    assert!(result_of(&calls, "c2").starts_with("Added playwright"), "{}", result_of(&calls, "c2"));
    assert!(h.confirm.asked().iter().any(|request| matches!(
        request,
        ConfirmRequest::AddConnector { name, command, why }
            if name == "playwright" && command == "npx @playwright/mcp@latest" && why == "To test the site in a browser."
    )));
    let declared = config::load_user(&h.state);
    let added = declared.servers.iter().find(|s| s.key.name == "playwright").expect("added");
    assert_eq!(added.command_line(), "npx @playwright/mcp@latest");
    assert!(result_of(&calls, "c3").contains("already in the user's MCP settings"));
    assert!(result_of(&calls, "c4").contains("Already in the user's MCP settings."));
    assert!(result_of(&calls, "c5").contains("no connector nosuch"));
    // A no adds nothing.
    *h.confirm.answer.lock().unwrap() = false;
    let model = h.script(vec![
        call("suggest_connector", json!({"id": "time", "why": "For time zones."}), "c6"),
        say("Fine."),
    ]);
    h.agent(Some(&id), "and time zones?", &ws);
    h.turns_end(&id, 2);
    assert!(result_of(&model.calls(), "c6").contains("did not add"));
    assert!(!config::load_user(&h.state).servers.iter().any(|s| s.key.name == "time"));
}
