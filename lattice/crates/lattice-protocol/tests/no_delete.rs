//! NF3: the services the interface calls have no way to delete a conversation
//! (spec §4.3 and the never-delete rule ND1).
//!
//! The check reads the two service traits out of the crate's sources and lists
//! their methods; a method whose name says it removes something fails it. It is
//! first shown to catch a fixture trait that has one.

use std::path::Path;

/// The method names declared in `pub trait <name>` in `source`.
fn trait_methods(source: &str, name: &str) -> Vec<String> {
    let header = format!("pub trait {name}");
    let start = source
        .find(&header)
        .unwrap_or_else(|| panic!("no `{header}` in the source"));
    let body_start = start + source[start..].find('{').expect("a trait body");
    let mut depth = 0usize;
    let mut end = body_start;
    for (offset, c) in source[body_start..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = body_start + offset;
                    break;
                }
            }
            _ => {}
        }
    }
    source[body_start..end]
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .filter_map(|line| {
            let rest = line.trim_start().strip_prefix("fn ")?;
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            Some(name)
        })
        .collect()
}

/// Names that would remove a conversation or its record.
fn removals(methods: &[String]) -> Vec<String> {
    methods
        .iter()
        .filter(|name| {
            [
                "delete", "remove", "purge", "erase", "destroy", "clear", "wipe",
            ]
            .iter()
            .any(|word| name.split('_').any(|part| part == *word))
        })
        .cloned()
        .collect()
}

fn source(file: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join(file);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn the_check_catches_a_trait_with_a_delete() {
    let fixture = "pub trait AgentChatService: Send {\n    fn list(&self);\n    \
                   fn delete(&self, id: &str) -> Result<(), Refusal>;\n    \
                   fn remove_thread(&self);\n}\n";
    let found = removals(&trait_methods(fixture, "AgentChatService"));
    println!("mutant fixture: {found:?}");
    assert_eq!(found, ["delete", "remove_thread"]);
    // A narrowing that removes nothing is not caught.
    let fine = "pub trait X {\n    fn revoke(&self);\n    fn cancel_queued(&self);\n}\n";
    assert!(removals(&trait_methods(fine, "X")).is_empty());
}

#[test]
fn neither_service_can_delete_a_conversation() {
    let agent = trait_methods(&source("conversation.rs"), "AgentChatService");
    assert!(agent.len() >= 30, "found only {agent:?}");
    assert!(agent.iter().any(|name| name == "archive"));
    let chat = trait_methods(&source("chat.rs"), "ChatService");
    assert!(chat.len() >= 9, "found only {chat:?}");
    for (name, methods) in [("AgentChatService", agent), ("ChatService", chat)] {
        let found = removals(&methods);
        assert!(found.is_empty(), "{name} can delete: {found:?}");
    }
}
