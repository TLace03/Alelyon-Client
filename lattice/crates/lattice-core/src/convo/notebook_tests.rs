//! Notebooks by cell (`crate::tools::notebook`), end to end over the agent
//! harness: a notebook is read cell by cell with its outputs; a cell's source
//! is replaced (its stale outputs dropped), a cell inserted (with a new id)
//! and one deleted, each staged for review with nothing on disk; the staged
//! notebook is what a later read shows; a file that is not a notebook is
//! refused; and the file is written as Jupyter writes it.

use lattice_agents::model::{InputItem, ModelRequest};
use lattice_protocol::conversation::AgentChatService;
use serde_json::json;

use super::agent_tests::{H, call, files, say};
use crate::tools::notebook::written;

/// A notebook as Jupyter 4.5 writes it: keys sorted, one space of indent.
const NOTEBOOK: &str = r##"{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "cell-one",
   "metadata": {},
   "source": [
    "# Sales\n",
    "A look at the numbers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "cell-two",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "OLD-OUTPUT 42\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgo="
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(40 + 2)"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
"##;

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
fn a_notebook_is_written_back_as_jupyter_writes_it() {
    let notebook: serde_json::Value = serde_json::from_str(NOTEBOOK).unwrap();
    assert_eq!(written(&notebook).unwrap(), NOTEBOOK);
}

#[test]
fn a_notebooks_cells_are_read_and_edited_one_at_a_time_for_review() {
    let h = H::new("notebook");
    std::fs::write(h.folder.join("sales.ipynb"), NOTEBOOK).unwrap();
    std::fs::write(h.folder.join("plain.json"), "{}").unwrap();
    let ws = h.workspace();
    let before = files(&h.folder);
    let model = h.script(vec![
        call("read_notebook", json!({"path": "sales.ipynb"}), "c1"),
        call(
            "notebook_edit",
            json!({"path": "sales.ipynb", "cell_id": "cell-two", "new_source": "print(NEW_SOURCE)\nprint(2)"}),
            "c2",
        ),
        call(
            "notebook_edit",
            json!({"path": "sales.ipynb", "cell_id": "0", "edit_mode": "insert", "cell_type": "markdown", "new_source": "INSERTED NOTE"}),
            "c3",
        ),
        call("notebook_edit", json!({"path": "sales.ipynb", "cell_id": "0", "edit_mode": "delete"}), "c4"),
        call("read_notebook", json!({"path": "sales.ipynb"}), "c5"),
        call("notebook_edit", json!({"path": "sales.ipynb", "cell_id": "2", "new_source": "x"}), "c6"),
        call("read_notebook", json!({"path": "plain.json"}), "c7"),
        say("Edited the notebook."),
    ]);
    let id = h.agent(None, "tidy the notebook", &ws);
    h.turns_end(&id, 1);
    let calls = model.calls();
    let read = result_of(&calls, "c1");
    assert!(read.contains("sales.ipynb: 2 cells"), "{read}");
    assert!(read.contains("--- cell 0 id cell-one (markdown)\n# Sales\nA look at the numbers."), "{read}");
    assert!(read.contains("--- cell 1 id cell-two (code)\nprint(40 + 2)"), "{read}");
    assert!(read.contains("OLD-OUTPUT 42"), "a text output is shown: {read}");
    assert!(read.contains("[image/png output not shown]"), "an image is named: {read}");
    assert!(result_of(&calls, "c2").starts_with("Replaced cell 1's source. Staged `sales.ipynb`"), "{}", result_of(&calls, "c2"));
    assert!(result_of(&calls, "c3").starts_with("Inserted a markdown cell at 1."), "{}", result_of(&calls, "c3"));
    assert!(result_of(&calls, "c4").starts_with("Deleted cell 0."), "{}", result_of(&calls, "c4"));
    // The staged notebook: the inserted note first, then the edited code cell
    // with no stale output.
    let staged = result_of(&calls, "c5");
    assert!(staged.contains("sales.ipynb: 2 cells"), "{staged}");
    let note = staged.find("INSERTED NOTE").expect("the inserted cell");
    let code = staged.find("print(NEW_SOURCE)\nprint(2)").expect("the new source");
    assert!(note < code, "{staged}");
    assert!(!staged.contains("OLD-OUTPUT") && !staged.contains("# Sales"), "{staged}");
    let inserted_id = staged
        .lines()
        .find(|line| line.starts_with("--- cell 0 id "))
        .map(|line| line["--- cell 0 id ".len().."--- cell 0 id ".len() + 8].to_owned())
        .expect("the new cell has an id");
    assert!(inserted_id.chars().all(|c| c.is_ascii_hexdigit()), "{inserted_id}");
    assert!(result_of(&calls, "c6").contains("There is no cell 2"));
    assert!(result_of(&calls, "c7").contains("Give a notebook"));
    // Staged for review, not on disk.
    assert_eq!(files(&h.folder), before, "nothing reached the disk");
    let changes = h.runtime.block_on(h.chat.changes(&id)).unwrap();
    assert!(changes.changes.iter().any(|change| change.path == "sales.ipynb"), "{changes:?}");
}
