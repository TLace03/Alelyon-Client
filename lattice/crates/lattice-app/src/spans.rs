//! Spans as the interface understands them: their kind, their title, the tree
//! they form, which rows of the tree are visible, and a run's totals.
//!
//! Invariants:
//! - [`SpanTree::build`] places every input span exactly once, whatever the
//!   input: orphans (a `parent_id` that names no span) become roots, a span
//!   that names itself becomes a root, and a cycle is cut at its earliest span
//!   (lowest `order`), so the result is always a forest.
//! - Siblings, and roots, are ordered by `order` (ties broken by input
//!   position), which the protocol guarantees is a valid tree order.
//! - Building is iterative: a chain of ten thousand nested spans cannot
//!   overflow the stack.
//! - Nothing here is called from `view()`; the application builds the tree when
//!   spans change and keeps the result.

use std::collections::{HashMap, HashSet};

use lattice_protocol::SpanRecord;
use serde_json::Value;

/// What a span is, for colour, glyph and title.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpanKind {
    Agent,
    /// The SDK's `function` span: a tool call.
    Tool,
    /// The SDK's `generation` and `response` spans: a model call.
    Model,
    Handoff,
    Guardrail,
    /// The SDK's task span (`custom` with `sdk_span_type = "task"`): the whole run.
    Task,
    /// The SDK's turn span (`custom` with `sdk_span_type = "turn"`).
    Turn,
    /// Any other `custom` span.
    Custom,
    /// `mcp_tools`.
    Mcp,
    /// `transcription`, `speech`, `speech_group`.
    Voice,
    Unknown,
}

impl SpanKind {
    /// The word for the kind chip in the detail column.
    pub fn label(self) -> &'static str {
        match self {
            SpanKind::Agent => "Agent",
            SpanKind::Tool => "Tool",
            SpanKind::Model => "Model",
            SpanKind::Handoff => "Handoff",
            SpanKind::Guardrail => "Guardrail",
            SpanKind::Task => "Task",
            SpanKind::Turn => "Turn",
            SpanKind::Custom => "Custom",
            SpanKind::Mcp => "MCP",
            SpanKind::Voice => "Voice",
            SpanKind::Unknown => "Span",
        }
    }

    /// One ASCII letter for the chip in the waterfall's tree column.
    pub fn glyph(self) -> &'static str {
        match self {
            SpanKind::Agent => "A",
            SpanKind::Tool => "T",
            SpanKind::Model => "M",
            SpanKind::Handoff => "H",
            SpanKind::Guardrail => "G",
            SpanKind::Task => "R",
            SpanKind::Turn => "N",
            SpanKind::Custom => "C",
            SpanKind::Mcp => "P",
            SpanKind::Voice => "V",
            SpanKind::Unknown => "?",
        }
    }
}

pub fn kind_of(span: &SpanRecord) -> SpanKind {
    match span.data_type() {
        "agent" => SpanKind::Agent,
        "function" => SpanKind::Tool,
        "generation" | "response" => SpanKind::Model,
        "handoff" => SpanKind::Handoff,
        "guardrail" => SpanKind::Guardrail,
        "custom" => match span.sdk_span_type() {
            Some("task") => SpanKind::Task,
            Some("turn") => SpanKind::Turn,
            _ => match span.data_str("name") {
                Some("task") => SpanKind::Task,
                Some("turn") => SpanKind::Turn,
                _ => SpanKind::Custom,
            },
        },
        "mcp_tools" => SpanKind::Mcp,
        "transcription" | "speech" | "speech_group" => SpanKind::Voice,
        _ => SpanKind::Unknown,
    }
}

fn non_empty(text: Option<&str>) -> Option<&str> {
    text.filter(|t| !t.trim().is_empty())
}

/// The one-line title of a span.
pub fn title_of(span: &SpanRecord) -> String {
    match kind_of(span) {
        SpanKind::Agent => non_empty(span.data_str("name"))
            .unwrap_or("Agent")
            .to_string(),
        SpanKind::Tool => non_empty(span.data_str("name"))
            .unwrap_or("Tool call")
            .to_string(),
        SpanKind::Model => match span.data_type() {
            "response" => "Response".to_string(),
            _ => non_empty(span.data_str("model"))
                .unwrap_or("Model call")
                .to_string(),
        },
        SpanKind::Handoff => {
            let from = non_empty(span.data_str("from_agent")).unwrap_or("?");
            let to = non_empty(span.data_str("to_agent")).unwrap_or("?");
            format!("{from} → {to}")
        }
        SpanKind::Guardrail => non_empty(span.data_str("name"))
            .unwrap_or("Guardrail")
            .to_string(),
        SpanKind::Task => "Task".to_string(),
        SpanKind::Turn => match span
            .span_data
            .get("data")
            .and_then(|d| d.get("turn"))
            .and_then(Value::as_u64)
        {
            Some(n) => format!("Turn {n}"),
            None => "Turn".to_string(),
        },
        SpanKind::Custom => non_empty(span.data_str("name"))
            .unwrap_or("Custom")
            .to_string(),
        SpanKind::Mcp => match non_empty(span.data_str("server")) {
            Some(server) => format!("MCP · {server}"),
            None => "MCP tools".to_string(),
        },
        SpanKind::Voice => match span.data_type() {
            "transcription" => "Transcription",
            "speech" => "Speech",
            _ => "Speech group",
        }
        .to_string(),
        SpanKind::Unknown => span.data_type().to_string(),
    }
}

/// True when a guardrail span records that its tripwire fired.
pub fn guardrail_triggered(span: &SpanRecord) -> bool {
    span.span_data
        .get("triggered")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Anything that holds a span record, so the tree can be built over the run
/// state's entries without cloning every record's (possibly large) payload.
pub trait HasSpan {
    fn span(&self) -> &SpanRecord;
}

impl HasSpan for SpanRecord {
    fn span(&self) -> &SpanRecord {
        self
    }
}

/// The spans as a forest of indices into the slice they were built from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpanTree {
    pub parent: Vec<Option<usize>>,
    pub children: Vec<Vec<usize>>,
    pub roots: Vec<usize>,
}

impl SpanTree {
    pub fn build<S: HasSpan>(spans: &[S]) -> Self {
        let n = spans.len();
        let mut by_id: HashMap<&str, usize> = HashMap::with_capacity(n);
        for (i, span) in spans.iter().enumerate() {
            by_id.entry(span.span().id.as_str()).or_insert(i);
        }
        let mut parent: Vec<Option<usize>> = spans
            .iter()
            .enumerate()
            .map(|(i, span)| {
                span.span()
                    .parent_id
                    .as_deref()
                    .and_then(|p| by_id.get(p).copied())
                    .filter(|&p| p != i)
            })
            .collect();

        // Each span has at most one parent, so each component has at most one
        // cycle. Walk every parent chain once; a chain that runs into itself is a
        // cycle, cut at its earliest span.
        let mut state = vec![0u8; n]; // 0 unvisited, 1 on the current chain, 2 settled
        for start in 0..n {
            if state[start] != 0 {
                continue;
            }
            let mut chain: Vec<usize> = Vec::new();
            let mut cur = start;
            loop {
                match state[cur] {
                    2 => break,
                    1 => {
                        let at = chain.iter().position(|&x| x == cur).unwrap_or(0);
                        let cycle = &chain[at..];
                        if let Some(&cut) =
                            cycle.iter().min_by_key(|&&i| (spans[i].span().order, i))
                        {
                            parent[cut] = None;
                        }
                        break;
                    }
                    _ => {
                        state[cur] = 1;
                        chain.push(cur);
                        match parent[cur] {
                            Some(p) => cur = p,
                            None => break,
                        }
                    }
                }
            }
            for &i in &chain {
                state[i] = 2;
            }
        }

        let mut children = vec![Vec::new(); n];
        let mut roots = Vec::new();
        for (i, p) in parent.iter().enumerate() {
            match p {
                Some(p) => children[*p].push(i),
                None => roots.push(i),
            }
        }
        let key = |i: &usize| (spans[*i].span().order, *i);
        roots.sort_by_key(key);
        for list in &mut children {
            list.sort_by_key(key);
        }
        Self {
            parent,
            children,
            roots,
        }
    }

    /// Every span index once, parents before their children, siblings in order.
    pub fn preorder(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.parent.len());
        let mut stack: Vec<usize> = self.roots.iter().rev().copied().collect();
        while let Some(i) = stack.pop() {
            out.push(i);
            stack.extend(self.children[i].iter().rev().copied());
        }
        out
    }
}

/// One visible row of the waterfall's tree column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Row {
    /// Index into the span slice.
    pub index: usize,
    pub depth: u16,
    pub has_children: bool,
    pub collapsed: bool,
}

/// The rows to show: the preorder, minus everything below a collapsed span.
pub fn visible_rows<S: HasSpan>(
    spans: &[S],
    tree: &SpanTree,
    collapsed: &HashSet<String>,
) -> Vec<Row> {
    let mut rows = Vec::with_capacity(spans.len());
    let mut stack: Vec<(usize, u16)> = tree.roots.iter().rev().map(|&i| (i, 0)).collect();
    while let Some((i, depth)) = stack.pop() {
        let has_children = !tree.children[i].is_empty();
        let is_collapsed = has_children && collapsed.contains(&spans[i].span().id);
        rows.push(Row {
            index: i,
            depth,
            has_children,
            collapsed: is_collapsed,
        });
        if has_children && !is_collapsed {
            let next = depth.saturating_add(1);
            stack.extend(tree.children[i].iter().rev().map(|&c| (c, next)));
        }
    }
    rows
}

/// What a run did, counted from its spans.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Totals {
    pub spans: usize,
    pub tool_calls: usize,
    pub handoffs: usize,
    pub model_calls: usize,
    pub errors: usize,
}

pub fn totals<'a>(spans: impl IntoIterator<Item = &'a SpanRecord>) -> Totals {
    let mut t = Totals::default();
    for span in spans {
        t.spans += 1;
        match kind_of(span) {
            SpanKind::Tool => t.tool_calls += 1,
            SpanKind::Handoff => t.handoffs += 1,
            SpanKind::Model => t.model_calls += 1,
            _ => {}
        }
        if span.error.is_some() {
            t.errors += 1;
        }
    }
    t
}

#[cfg(test)]
pub(crate) fn test_span(id: &str, parent: Option<&str>, order: u64, data: Value) -> SpanRecord {
    SpanRecord {
        order,
        id: id.to_string(),
        trace_id: "trace_t".to_string(),
        parent_id: parent.map(str::to_string),
        started_at: "2026-09-30T05:21:05.000000+00:00".to_string(),
        ended_at: Some("2026-09-30T05:21:06.000000+00:00".to_string()),
        span_data: data,
        error: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn plain(id: &str, parent: Option<&str>, order: u64) -> SpanRecord {
        test_span(id, parent, order, json!({"type": "agent", "name": id}))
    }

    /// A small deterministic generator, so the property test is seeded.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n.max(1)
        }
    }

    #[test]
    fn every_kind_and_title_of_the_sdk_span_types() {
        let cases = [
            (
                json!({"type":"agent","name":"Lattice assistant","handoffs":["Model advisor"],"tools":[],"output_type":"str"}),
                SpanKind::Agent,
                "Lattice assistant",
            ),
            (
                json!({"type":"function","name":"current_time","input":"{}","output":"noon"}),
                SpanKind::Tool,
                "current_time",
            ),
            (
                json!({"type":"generation","model":"llama3.2:3b","input":[],"output":[]}),
                SpanKind::Model,
                "llama3.2:3b",
            ),
            (
                json!({"type":"generation","model":null}),
                SpanKind::Model,
                "Model call",
            ),
            (
                json!({"type":"response","response_id":"resp_1"}),
                SpanKind::Model,
                "Response",
            ),
            (
                json!({"type":"handoff","from_agent":"Lattice assistant","to_agent":"Model advisor"}),
                SpanKind::Handoff,
                "Lattice assistant → Model advisor",
            ),
            (
                json!({"type":"handoff","from_agent":"A","to_agent":null}),
                SpanKind::Handoff,
                "A → ?",
            ),
            (
                json!({"type":"guardrail","name":"secrets_stay_local","triggered":false}),
                SpanKind::Guardrail,
                "secrets_stay_local",
            ),
            (
                json!({"type":"custom","name":"task","data":{"sdk_span_type":"task","name":"Agent workflow"}}),
                SpanKind::Task,
                "Task",
            ),
            (
                json!({"type":"custom","name":"turn","data":{"sdk_span_type":"turn","turn":2,"agent_name":"A"}}),
                SpanKind::Turn,
                "Turn 2",
            ),
            (
                json!({"type":"custom","name":"turn","data":{"sdk_span_type":"turn"}}),
                SpanKind::Turn,
                "Turn",
            ),
            (
                json!({"type":"custom","name":"my step","data":{"x":1}}),
                SpanKind::Custom,
                "my step",
            ),
            (
                json!({"type":"mcp_tools","server":"files","result":["read"]}),
                SpanKind::Mcp,
                "MCP · files",
            ),
            (json!({"type":"mcp_tools"}), SpanKind::Mcp, "MCP tools"),
            (
                json!({"type":"transcription","input":{"data":"","format":"pcm"}}),
                SpanKind::Voice,
                "Transcription",
            ),
            (json!({"type":"speech"}), SpanKind::Voice, "Speech"),
            (
                json!({"type":"speech_group"}),
                SpanKind::Voice,
                "Speech group",
            ),
            (
                json!({"type":"something_new"}),
                SpanKind::Unknown,
                "something_new",
            ),
            (json!({}), SpanKind::Unknown, "unknown"),
        ];
        for (data, kind, title) in cases {
            let span = test_span("s", None, 1, data.clone());
            assert_eq!(kind_of(&span), kind, "{data}");
            assert_eq!(title_of(&span), title, "{data}");
            assert!(!kind.glyph().is_empty() && !kind.label().is_empty());
        }
    }

    #[test]
    fn siblings_are_ordered_by_order_not_by_input_position() {
        let spans = vec![
            plain("root", None, 1),
            plain("late", Some("root"), 9),
            plain("early", Some("root"), 3),
            plain("mid", Some("root"), 5),
        ];
        let tree = SpanTree::build(&spans);
        let names: Vec<&str> = tree
            .preorder()
            .iter()
            .map(|&i| spans[i].id.as_str())
            .collect();
        assert_eq!(names, ["root", "early", "mid", "late"]);
    }

    #[test]
    fn orphans_self_parents_and_cycles_still_appear_once() {
        let spans = vec![
            plain("a", Some("missing"), 4), // orphan
            plain("b", Some("b"), 2),       // names itself
            plain("c", Some("d"), 7),       // c <-> d cycle; d has the lower order
            plain("d", Some("c"), 6),
            plain("e", Some("d"), 8),
        ];
        let tree = SpanTree::build(&spans);
        let mut order = tree.preorder();
        assert_eq!(order.len(), spans.len());
        order.sort_unstable();
        assert_eq!(order, [0, 1, 2, 3, 4]);
        // The cycle is cut at its earliest span: d becomes a root, c hangs below it.
        assert!(tree.roots.contains(&3), "{:?}", tree.roots);
        assert_eq!(tree.parent[2], Some(3));
        assert_eq!(tree.parent[4], Some(3));
    }

    #[test]
    fn a_seeded_property_test_over_random_forests_with_orphans_and_cycles() {
        let mut rng = Lcg(0x5EED_1234_ABCD);
        for case in 0..300 {
            let n = rng.below(60) as usize;
            let ids: Vec<String> = (0..n).map(|i| format!("span_{i}")).collect();
            let mut spans = Vec::with_capacity(n);
            for i in 0..n {
                let parent = match rng.below(10) {
                    0 => None,
                    1 => Some("span_missing".to_string()), // orphan
                    2 => Some(ids[i].clone()),             // self
                    _ => Some(ids[rng.below(n as u64) as usize].clone()), // anything, cycles included
                };
                // Orders are not unique in general; duplicates stress the tie-break.
                spans.push(plain_owned(&ids[i], parent, rng.below(20)));
            }
            let tree = SpanTree::build(&spans);
            let pre = tree.preorder();
            assert_eq!(pre.len(), n, "case {case}: every span exactly once");
            let mut seen = vec![false; n];
            for &i in &pre {
                assert!(!seen[i], "case {case}: span {i} twice");
                seen[i] = true;
            }
            let position: Vec<usize> = {
                let mut p = vec![0; n];
                for (k, &i) in pre.iter().enumerate() {
                    p[i] = k;
                }
                p
            };
            for i in 0..n {
                if let Some(p) = tree.parent[i] {
                    assert!(
                        position[p] < position[i],
                        "case {case}: parent {p} after child {i}"
                    );
                    assert!(tree.children[p].contains(&i));
                }
            }
            for list in tree.children.iter().chain(std::iter::once(&tree.roots)) {
                assert!(
                    list.windows(2)
                        .all(|w| (spans[w[0]].order, w[0]) <= (spans[w[1]].order, w[1])),
                    "case {case}: siblings out of order"
                );
            }
        }
    }

    fn plain_owned(id: &str, parent: Option<String>, order: u64) -> SpanRecord {
        test_span(
            id,
            parent.as_deref(),
            order,
            json!({"type": "agent", "name": id}),
        )
    }

    #[test]
    fn a_very_deep_chain_does_not_overflow_the_stack() {
        let n = 20_000;
        let spans: Vec<SpanRecord> = (0..n)
            .map(|i| {
                plain_owned(
                    &format!("s{i}"),
                    if i == 0 {
                        None
                    } else {
                        Some(format!("s{}", i - 1))
                    },
                    i as u64,
                )
            })
            .collect();
        let tree = SpanTree::build(&spans);
        assert_eq!(tree.preorder().len(), n);
        let rows = visible_rows(&spans, &tree, &HashSet::new());
        assert_eq!(rows.len(), n);
        assert_eq!(rows.last().unwrap().depth as usize, n - 1);
    }

    #[test]
    fn collapsing_hides_descendants_but_keeps_the_collapsed_row() {
        let spans = vec![
            plain("task", None, 1),
            plain("agent", Some("task"), 2),
            plain("tool", Some("agent"), 3),
            plain("turn2", Some("task"), 4),
        ];
        let tree = SpanTree::build(&spans);
        let all = visible_rows(&spans, &tree, &HashSet::new());
        assert_eq!(
            all.iter().map(|r| (r.index, r.depth)).collect::<Vec<_>>(),
            [(0, 0), (1, 1), (2, 2), (3, 1)]
        );
        assert!(all[0].has_children && !all[0].collapsed);
        assert!(!all[2].has_children);

        let collapsed: HashSet<String> = ["agent".to_string()].into();
        let rows = visible_rows(&spans, &tree, &collapsed);
        assert_eq!(rows.iter().map(|r| r.index).collect::<Vec<_>>(), [0, 1, 3]);
        assert!(rows[1].collapsed && rows[1].has_children);

        // Collapsing a leaf changes nothing: there is nothing below it.
        let leaf: HashSet<String> = ["tool".to_string()].into();
        assert_eq!(visible_rows(&spans, &tree, &leaf).len(), 4);
        assert!(!visible_rows(&spans, &tree, &leaf)[2].collapsed);

        let root: HashSet<String> = ["task".to_string()].into();
        assert_eq!(visible_rows(&spans, &tree, &root).len(), 1);
    }

    #[test]
    fn totals_count_tools_handoffs_model_calls_and_errors() {
        let mut errored = test_span("t2", None, 4, json!({"type":"function","name":"calculate"}));
        errored.error = Some(lattice_protocol::SpanError {
            message: "Error running tool (non-fatal)".into(),
            data: None,
        });
        let spans = vec![
            test_span("a", None, 1, json!({"type":"agent","name":"A"})),
            test_span(
                "t1",
                None,
                2,
                json!({"type":"function","name":"current_time"}),
            ),
            errored,
            test_span("g", None, 5, json!({"type":"generation","model":"m"})),
            test_span(
                "h",
                None,
                6,
                json!({"type":"handoff","from_agent":"A","to_agent":"B"}),
            ),
        ];
        let t = totals(&spans);
        assert_eq!(
            t,
            Totals {
                spans: 5,
                tool_calls: 2,
                handoffs: 1,
                model_calls: 1,
                errors: 1
            }
        );
    }

    #[test]
    fn guardrail_triggered_reads_the_flag() {
        assert!(guardrail_triggered(&test_span(
            "g",
            None,
            1,
            json!({"type":"guardrail","name":"x","triggered":true})
        )));
        assert!(!guardrail_triggered(&test_span(
            "g",
            None,
            1,
            json!({"type":"guardrail","name":"x","triggered":false})
        )));
        assert!(!guardrail_triggered(&test_span(
            "g",
            None,
            1,
            json!({"type":"guardrail","name":"x"})
        )));
    }
}
