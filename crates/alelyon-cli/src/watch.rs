//! `alelyon-cli fleet watch`: the fleet in a terminal, printed again each time it changes. It calls the app's
//! `fleet_watch` over and over with the last digest it saw, so it prints only when something changed: every group
//! (its column, what it waits for, its tasks running, its steps), then every question and approval waiting, with
//! the ids `fleet answer` takes.

use serde_json::Value;

fn text<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap_or("")
}

fn number(value: &Value, key: &str) -> u64 {
    value.get(key).and_then(Value::as_u64).unwrap_or(0)
}

/// The column, in the board's words.
fn column(status: &str) -> &'static str {
    match status {
        "waiting" => "WAITS FOR YOU",
        "in_progress" => "in progress",
        "done" => "done",
        _ => "planned",
    }
}

/// The fleet (`fleet_watch`'s `fleet`) as lines for a terminal.
pub fn summary(fleet: &Value) -> String {
    let mut out = Vec::new();
    let groups = fleet.get("groups").and_then(Value::as_array).cloned().unwrap_or_default();
    if groups.is_empty() {
        out.push("No groups yet.".to_string());
    }
    for group in &groups {
        let mut line = format!("[{}] {}", column(text(group, "status")), text(group, "name"));
        if let Some(project) = group.get("project").and_then(Value::as_str) {
            line.push_str(&format!(" ({project})"));
        }
        let waits: Vec<&str> = group.get("waits").and_then(Value::as_array).map(|w| w.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
        if !waits.is_empty() {
            line.push_str(&format!(" - waits for {}", waits.join(", ")));
        }
        let running = number(group, "tasks_running");
        if running > 0 {
            line.push_str(&format!(" - {running} task{} running", if running == 1 { "" } else { "s" }));
        }
        let total = number(group, "todos_total");
        if total > 0 {
            line.push_str(&format!(" - {} of {total} steps", number(group, "todos_done")));
        }
        out.push(line);
    }
    let waiting = fleet.get("waiting").and_then(Value::as_array).cloned().unwrap_or_default();
    if !waiting.is_empty() {
        out.push(String::new());
        out.push(format!("Waiting for you ({}):", waiting.len()));
        for wait in &waiting {
            let place = match wait.get("group_name").and_then(Value::as_str) {
                Some(group) => format!("{group} / {}", text(wait, "topic_title")),
                None => text(wait, "topic_title").to_string(),
            };
            let what = if text(wait, "kind") == "approval" { "asks to" } else { "asks" };
            out.push(format!("  {place} {what}: {}", text(wait, "text")));
            let options: Vec<&str> =
                wait.get("options").and_then(Value::as_array).map(|o| o.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
            if !options.is_empty() {
                out.push(format!("    options: {}", options.join(" | ")));
            }
            if text(wait, "kind") == "question" {
                out.push(format!("    answer: alelyon-cli fleet answer {} {} \"...\"", text(wait, "topic"), text(wait, "call")));
            } else {
                out.push("    approve or refuse it in the app".to_string());
            }
        }
    }
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_fleet_reads_as_its_groups_then_what_waits_with_how_to_answer() {
        let fleet = json!({
            "digest": "0123",
            "groups": [
                {"group": "g1", "name": "Parser", "project": "site", "status": "waiting", "waits": ["a question"],
                 "tasks_running": 1, "todos_done": 1, "todos_total": 3},
                {"group": "g2", "name": "Docs", "project": null, "status": "done", "waits": [], "tasks_running": 0,
                 "todos_done": 0, "todos_total": 0}
            ],
            "waiting": [
                {"group": "g1", "group_name": "Parser", "topic": "t1", "topic_title": "CSV export", "call": "q1",
                 "kind": "question", "text": "Which format first?", "options": ["CSV", "JSON"]},
                {"group": null, "group_name": null, "topic": "t9", "topic_title": "Cleanup", "call": "c4",
                 "kind": "approval", "text": "run `rm -r build`", "options": []}
            ]
        });
        let lines = summary(&fleet);
        assert_eq!(
            lines,
            [
                "[WAITS FOR YOU] Parser (site) - waits for a question - 1 task running - 1 of 3 steps",
                "[done] Docs",
                "",
                "Waiting for you (2):",
                "  Parser / CSV export asks: Which format first?",
                "    options: CSV | JSON",
                "    answer: alelyon-cli fleet answer t1 q1 \"...\"",
                "  Cleanup asks to: run `rm -r build`",
                "    approve or refuse it in the app",
            ]
            .join("\n")
        );
        assert_eq!(summary(&json!({"groups": [], "waiting": []})), "No groups yet.");
    }
}
