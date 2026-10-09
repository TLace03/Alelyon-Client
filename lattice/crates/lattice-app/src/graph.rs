//! Layered layout of an agent's static graph (`AgentGraph`), in logical pixels.
//!
//! The picture is top to bottom: the start pill, then the root agent, then the
//! agents it hands off to (one layer per hop), then the end pill. An agent's
//! tools are stacked to its right, level with it. Edges are anchors only; the
//! canvas decides how to curve them.
//!
//! Invariants (each one has a test):
//! - Deterministic: the same graph always gives the same boxes and anchors.
//! - Every node with a unique id is placed exactly once, and no two boxes
//!   overlap; all boxes lie inside `width x height`.
//! - Every edge in the result has both endpoints in the result; an edge that
//!   names a node the graph does not contain is dropped, not guessed at.
//! - Cycles are fine: layers come from a breadth-first walk from the root
//!   agents, so a hand-off back to an earlier agent is drawn as a back edge.

use std::collections::{HashMap, HashSet, VecDeque};

use lattice_protocol::{AgentGraph, EdgeKind, NodeKind};

use crate::textmetrics::text_width;

/// Space left and right of the picture.
pub const MARGIN: f32 = 28.0;
/// Space above and below it. Small on purpose: the panel is about 390 px high in
/// a 900 px window and the demonstration graph is 360 px of picture, so a larger
/// margin would make the graph scroll for the sake of its own padding.
pub const MARGIN_Y: f32 = 12.0;
pub const AGENT_H: f32 = 44.0;
pub const TOOL_H: f32 = 28.0;
pub const PILL_H: f32 = 30.0;
pub const PILL_W: f32 = 84.0;
const TOOL_GAP: f32 = 8.0;
const AGENT_TO_TOOLS: f32 = 44.0;
const BLOCK_GAP: f32 = 56.0;
const LAYER_GAP: f32 = 64.0;
pub const AGENT_TEXT: f32 = 13.0;
pub const TOOL_TEXT: f32 = 12.0;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub fn right(&self) -> f32 {
        self.x + self.w
    }
    pub fn bottom(&self) -> f32 {
        self.y + self.h
    }
    pub fn top_center(&self) -> Pt {
        Pt {
            x: self.x + self.w / 2.0,
            y: self.y,
        }
    }
    pub fn bottom_center(&self) -> Pt {
        Pt {
            x: self.x + self.w / 2.0,
            y: self.bottom(),
        }
    }
    pub fn left_center(&self) -> Pt {
        Pt {
            x: self.x,
            y: self.y + self.h / 2.0,
        }
    }
    pub fn right_center(&self) -> Pt {
        Pt {
            x: self.right(),
            y: self.y + self.h / 2.0,
        }
    }
    /// True when the interiors intersect (touching edges do not overlap).
    pub fn overlaps(&self, other: &Rect) -> bool {
        self.x < other.right() - 1e-3
            && other.x < self.right() - 1e-3
            && self.y < other.bottom() - 1e-3
            && other.y < self.bottom() - 1e-3
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pt {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NodeBox {
    pub id: String,
    pub kind: NodeKind,
    /// What is drawn in the box.
    pub label: String,
    pub rect: Rect,
    pub layer: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EdgeLine {
    /// Indices into `GraphLayout::nodes`.
    pub source: usize,
    pub target: usize,
    pub kind: EdgeKind,
    pub from: Pt,
    pub to: Pt,
    /// The target is not below the source: draw the edge round the side.
    pub back: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GraphLayout {
    pub nodes: Vec<NodeBox>,
    pub edges: Vec<EdgeLine>,
    pub width: f32,
    pub height: f32,
}

struct Block {
    agent: Option<usize>,
    tools: Vec<usize>,
    w: f32,
    h: f32,
}

fn node_size(kind: NodeKind, label: &str) -> (f32, f32) {
    match kind {
        NodeKind::Start | NodeKind::End => (PILL_W, PILL_H),
        NodeKind::Agent => (
            (text_width(label, AGENT_TEXT) + 36.0).clamp(140.0, 280.0),
            AGENT_H,
        ),
        NodeKind::Tool | NodeKind::Mcp => (
            (text_width(label, TOOL_TEXT) + 28.0).clamp(96.0, 240.0),
            TOOL_H,
        ),
    }
}

fn shown_label(kind: NodeKind, label: &str) -> String {
    match kind {
        NodeKind::Start => "Start".to_string(),
        NodeKind::End => "End".to_string(),
        _ if label.trim().is_empty() => "(unnamed)".to_string(),
        _ => label.to_string(),
    }
}

pub fn layout(graph: &AgentGraph) -> GraphLayout {
    // Unique nodes, first occurrence wins.
    let mut ids: HashMap<&str, usize> = HashMap::new();
    let mut nodes: Vec<&lattice_protocol::GraphNode> = Vec::new();
    for node in &graph.nodes {
        if !ids.contains_key(node.id.as_str()) {
            ids.insert(node.id.as_str(), nodes.len());
            nodes.push(node);
        }
    }
    let kind = |i: usize| nodes[i].kind;
    let agents: Vec<usize> = (0..nodes.len())
        .filter(|&i| kind(i) == NodeKind::Agent)
        .collect();
    let is_tool = |i: usize| matches!(kind(i), NodeKind::Tool | NodeKind::Mcp);

    let edge_ends: Vec<Option<(usize, usize)>> = graph
        .edges
        .iter()
        .map(|e| Some((*ids.get(e.source.as_str())?, *ids.get(e.target.as_str())?)))
        .collect();

    // Tools stack beside the first agent that names them.
    let mut tools_of: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut attached: HashSet<usize> = HashSet::new();
    for (edge, ends) in graph.edges.iter().zip(&edge_ends) {
        if let (EdgeKind::Tool | EdgeKind::Mcp, Some((s, t))) = (edge.kind, ends)
            && kind(*s) == NodeKind::Agent
            && is_tool(*t)
            && attached.insert(*t)
        {
            tools_of.entry(*s).or_default().push(*t);
        }
    }
    let orphan_tools: Vec<usize> = (0..nodes.len())
        .filter(|&i| is_tool(i) && !attached.contains(&i))
        .collect();

    // Agent depth: breadth-first over hand-off edges from the root agents.
    let mut handoffs: HashMap<usize, Vec<usize>> = HashMap::new();
    for (edge, ends) in graph.edges.iter().zip(&edge_ends) {
        if let (EdgeKind::Handoff, Some((s, t))) = (edge.kind, ends)
            && kind(*s) == NodeKind::Agent
            && kind(*t) == NodeKind::Agent
        {
            handoffs.entry(*s).or_default().push(*t);
        }
    }
    let mut roots: Vec<usize> = agents.iter().copied().filter(|&i| nodes[i].root).collect();
    for (edge, ends) in graph.edges.iter().zip(&edge_ends) {
        if let (EdgeKind::Start, Some((s, t))) = (edge.kind, ends)
            && kind(*s) == NodeKind::Start
            && kind(*t) == NodeKind::Agent
            && !roots.contains(t)
        {
            roots.push(*t);
        }
    }
    if roots.is_empty()
        && let Some(&first) = agents.first()
    {
        roots.push(first);
    }
    let mut depth: HashMap<usize, usize> = HashMap::new();
    let walk = |seed: usize, depth: &mut HashMap<usize, usize>| {
        if depth.contains_key(&seed) {
            return;
        }
        depth.insert(seed, 0);
        let mut queue = VecDeque::from([seed]);
        while let Some(a) = queue.pop_front() {
            let d = depth[&a];
            for &next in handoffs.get(&a).map(Vec::as_slice).unwrap_or(&[]) {
                if let std::collections::hash_map::Entry::Vacant(slot) = depth.entry(next) {
                    slot.insert(d + 1);
                    queue.push_back(next);
                }
            }
        }
    };
    for &r in &roots {
        walk(r, &mut depth);
    }
    for &a in &agents {
        walk(a, &mut depth); // agents nothing reaches start their own walk at the top
    }
    let max_depth = agents.iter().map(|a| depth[a]).max().unwrap_or(0);

    // Layers of blocks: starts, one per agent depth, ends.
    let mut layers: Vec<Vec<Block>> = Vec::new();
    let pill_block = |i: usize| {
        let (w, h) = node_size(kind(i), &nodes[i].label);
        Block {
            agent: Some(i),
            tools: Vec::new(),
            w,
            h,
        }
    };
    let starts: Vec<usize> = (0..nodes.len())
        .filter(|&i| kind(i) == NodeKind::Start)
        .collect();
    let ends: Vec<usize> = (0..nodes.len())
        .filter(|&i| kind(i) == NodeKind::End)
        .collect();
    if !starts.is_empty() {
        layers.push(starts.iter().map(|&i| pill_block(i)).collect());
    }
    for d in 0..=max_depth {
        let mut layer = Vec::new();
        for &a in agents.iter().filter(|a| depth[a] == d) {
            let stack = tools_of.get(&a).cloned().unwrap_or_default();
            let (aw, ah) = node_size(NodeKind::Agent, &nodes[a].label);
            let tool_w = stack
                .iter()
                .map(|&t| node_size(kind(t), &nodes[t].label).0)
                .fold(0.0f32, f32::max);
            let stack_h =
                stack.len() as f32 * TOOL_H + stack.len().saturating_sub(1) as f32 * TOOL_GAP;
            let w = if stack.is_empty() {
                aw
            } else {
                aw + AGENT_TO_TOOLS + tool_w
            };
            layer.push(Block {
                agent: Some(a),
                tools: stack,
                w,
                h: ah.max(stack_h),
            });
        }
        if d == 0 && !orphan_tools.is_empty() {
            let tool_w = orphan_tools
                .iter()
                .map(|&t| node_size(kind(t), &nodes[t].label).0)
                .fold(0.0f32, f32::max);
            let h = orphan_tools.len() as f32 * TOOL_H
                + orphan_tools.len().saturating_sub(1) as f32 * TOOL_GAP;
            layer.push(Block {
                agent: None,
                tools: orphan_tools.clone(),
                w: tool_w,
                h,
            });
        }
        if !layer.is_empty() {
            layers.push(layer);
        }
    }
    if !ends.is_empty() {
        layers.push(ends.iter().map(|&i| pill_block(i)).collect());
    }

    // Sizes of the whole picture.
    let layer_w = |layer: &Vec<Block>| {
        layer.iter().map(|b| b.w).sum::<f32>() + BLOCK_GAP * layer.len().saturating_sub(1) as f32
    };
    let layer_h = |layer: &Vec<Block>| layer.iter().map(|b| b.h).fold(0.0f32, f32::max);
    let inner_w = layers.iter().map(layer_w).fold(0.0f32, f32::max);
    let inner_h =
        layers.iter().map(layer_h).sum::<f32>() + LAYER_GAP * layers.len().saturating_sub(1) as f32;
    let width = inner_w + 2.0 * MARGIN;
    let height = inner_h + 2.0 * MARGIN_Y;

    // Place.
    let mut placed: Vec<Option<(Rect, usize)>> = vec![None; nodes.len()];
    let mut y = MARGIN_Y;
    for (li, layer) in layers.iter().enumerate() {
        let lh = layer_h(layer);
        let mut x = (width - layer_w(layer)) / 2.0;
        for block in layer {
            let top = y + (lh - block.h) / 2.0;
            if let Some(a) = block.agent {
                let (aw, ah) = node_size(kind(a), &nodes[a].label);
                placed[a] = Some((
                    Rect {
                        x,
                        y: top + (block.h - ah) / 2.0,
                        w: aw,
                        h: ah,
                    },
                    li,
                ));
            }
            let tools_x = match block.agent {
                Some(a) => x + node_size(kind(a), &nodes[a].label).0 + AGENT_TO_TOOLS,
                None => x,
            };
            let mut ty = top;
            for &t in &block.tools {
                let (tw, th) = node_size(kind(t), &nodes[t].label);
                placed[t] = Some((
                    Rect {
                        x: tools_x,
                        y: ty,
                        w: tw,
                        h: th,
                    },
                    li,
                ));
                ty += th + TOOL_GAP;
            }
            x += block.w + BLOCK_GAP;
        }
        y += lh + LAYER_GAP;
    }

    let mut out_nodes = Vec::with_capacity(nodes.len());
    let mut out_index: HashMap<usize, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        if let Some((rect, layer)) = placed[i] {
            out_index.insert(i, out_nodes.len());
            out_nodes.push(NodeBox {
                id: node.id.clone(),
                kind: node.kind,
                label: shown_label(node.kind, &node.label),
                rect,
                layer,
            });
        }
    }
    let mut out_edges = Vec::new();
    for (edge, ends) in graph.edges.iter().zip(&edge_ends) {
        let Some((s, t)) = ends else { continue };
        let (Some(&si), Some(&ti)) = (out_index.get(s), out_index.get(t)) else {
            continue;
        };
        let (src, dst) = (&out_nodes[si], &out_nodes[ti]);
        let (from, to, back) = match edge.kind {
            EdgeKind::Tool | EdgeKind::Mcp => {
                (src.rect.right_center(), dst.rect.left_center(), false)
            }
            _ if dst.layer > src.layer => (src.rect.bottom_center(), dst.rect.top_center(), false),
            _ => (src.rect.left_center(), dst.rect.left_center(), true),
        };
        out_edges.push(EdgeLine {
            source: si,
            target: ti,
            kind: edge.kind,
            from,
            to,
            back,
        });
    }
    GraphLayout {
        nodes: out_nodes,
        edges: out_edges,
        width,
        height,
    }
}

/// Which nodes to draw at full strength: an agent that has an agent span in the
/// trace (`active` holds their names), the tools of such an agent, and the
/// start and end pills while any agent is active.
pub fn node_activity(layout: &GraphLayout, active: &HashSet<String>) -> Vec<bool> {
    let mut on: Vec<bool> = layout
        .nodes
        .iter()
        .map(|n| n.kind == NodeKind::Agent && active.contains(&n.label))
        .collect();
    let any = on.iter().any(|&a| a);
    for edge in &layout.edges {
        if matches!(edge.kind, EdgeKind::Tool | EdgeKind::Mcp)
            && layout.nodes[edge.source].kind == NodeKind::Agent
            && on[edge.source]
        {
            on[edge.target] = true;
        }
    }
    for (i, node) in layout.nodes.iter().enumerate() {
        if matches!(node.kind, NodeKind::Start | NodeKind::End) {
            on[i] = any;
        }
    }
    on
}

/// An edge is drawn at full strength when both of its ends are.
pub fn edge_active(on: &[bool], edge: &EdgeLine) -> bool {
    on[edge.source] && on[edge.target]
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_protocol::{GraphEdge, GraphNode};

    fn node(id: &str, kind: NodeKind, label: &str, root: bool) -> GraphNode {
        GraphNode {
            id: id.into(),
            kind,
            label: label.into(),
            root,
        }
    }

    fn edge(source: &str, target: &str, kind: EdgeKind) -> GraphEdge {
        GraphEdge {
            source: source.into(),
            target: target.into(),
            kind,
        }
    }

    /// The built-in pair: Lattice assistant hands off to Model advisor.
    fn pair() -> AgentGraph {
        AgentGraph {
            nodes: vec![
                node("start", NodeKind::Start, "__start__", false),
                node("agent-lattice", NodeKind::Agent, "Lattice assistant", true),
                node("tool-current_time", NodeKind::Tool, "current_time", false),
                node("tool-calculate", NodeKind::Tool, "calculate", false),
                node("agent-advisor", NodeKind::Agent, "Model advisor", false),
                node("tool-list_models", NodeKind::Tool, "list_models", false),
                node("end", NodeKind::End, "__end__", false),
            ],
            edges: vec![
                edge("start", "agent-lattice", EdgeKind::Start),
                edge("agent-lattice", "tool-current_time", EdgeKind::Tool),
                edge("agent-lattice", "tool-calculate", EdgeKind::Tool),
                edge("agent-lattice", "agent-advisor", EdgeKind::Handoff),
                edge("agent-advisor", "tool-list_models", EdgeKind::Tool),
                edge("agent-advisor", "end", EdgeKind::End),
            ],
        }
    }

    fn assert_sane(graph: &AgentGraph, l: &GraphLayout) {
        for (i, a) in l.nodes.iter().enumerate() {
            assert!(
                a.rect.x >= 0.0
                    && a.rect.y >= 0.0
                    && a.rect.right() <= l.width + 1e-3
                    && a.rect.bottom() <= l.height + 1e-3,
                "{} outside {}x{}: {:?}",
                a.id,
                l.width,
                l.height,
                a.rect
            );
            for b in &l.nodes[i + 1..] {
                assert!(!a.rect.overlaps(&b.rect), "{} overlaps {}", a.id, b.id);
            }
        }
        let unique: HashSet<&str> = graph.nodes.iter().map(|n| n.id.as_str()).collect();
        assert_eq!(
            l.nodes.len(),
            unique.len(),
            "every unique node is placed exactly once"
        );
        let placed: HashSet<&str> = l.nodes.iter().map(|n| n.id.as_str()).collect();
        assert_eq!(placed, unique);
        for e in &l.edges {
            assert!(e.source < l.nodes.len() && e.target < l.nodes.len());
        }
        let expected = graph
            .edges
            .iter()
            .filter(|e| unique.contains(e.source.as_str()) && unique.contains(e.target.as_str()))
            .count();
        assert_eq!(
            l.edges.len(),
            expected,
            "an edge is dropped only when an endpoint is missing"
        );
    }

    #[test]
    fn the_built_in_pair_reads_top_to_bottom_with_tools_beside_their_agent() {
        let g = pair();
        let l = layout(&g);
        assert_sane(&g, &l);
        let by = |id: &str| l.nodes.iter().find(|n| n.id == id).unwrap();
        let (start, lattice, advisor, end) = (
            by("start"),
            by("agent-lattice"),
            by("agent-advisor"),
            by("end"),
        );
        assert!(start.rect.bottom() < lattice.rect.y);
        assert!(lattice.rect.bottom() < advisor.rect.y);
        assert!(advisor.rect.bottom() < end.rect.y);
        assert_eq!(
            (start.layer, lattice.layer, advisor.layer, end.layer),
            (0, 1, 2, 3)
        );
        for tool in ["tool-current_time", "tool-calculate"] {
            let t = by(tool);
            assert!(
                t.rect.x >= lattice.rect.right(),
                "{tool} sits to the right of its agent"
            );
            assert_eq!(t.layer, lattice.layer);
        }
        assert!(by("tool-list_models").rect.x >= advisor.rect.right());
        // Start and end pills say so, whatever the graph calls them.
        assert_eq!(start.label, "Start");
        assert_eq!(end.label, "End");
        // Handoff and start/end edges run downwards; tool edges run sideways.
        for e in &l.edges {
            match e.kind {
                EdgeKind::Tool => assert!(e.to.x > e.from.x && !e.back),
                _ => assert!(e.to.y > e.from.y && !e.back),
            }
        }
    }

    #[test]
    fn agents_with_spans_are_active_and_carry_their_tools_and_the_pills() {
        let l = layout(&pair());
        let idx = |id: &str| l.nodes.iter().position(|n| n.id == id).unwrap();
        let none = node_activity(&l, &HashSet::new());
        assert!(
            none.iter().all(|&a| !a),
            "no agent span yet: everything is dimmed"
        );

        let only_first: HashSet<String> = ["Lattice assistant".to_string()].into();
        let on = node_activity(&l, &only_first);
        assert!(
            on[idx("agent-lattice")] && on[idx("tool-current_time")] && on[idx("tool-calculate")]
        );
        assert!(
            !on[idx("agent-advisor")] && !on[idx("tool-list_models")],
            "the advisor did not run"
        );
        assert!(on[idx("start")] && on[idx("end")]);

        let both: HashSet<String> =
            ["Lattice assistant".to_string(), "Model advisor".to_string()].into();
        assert!(node_activity(&l, &both).iter().all(|&a| a));

        // An edge is full strength only when both ends are.
        for edge in &l.edges {
            let strong = edge_active(&on, edge);
            assert_eq!(strong, on[edge.source] && on[edge.target]);
        }
        let handoff = l
            .edges
            .iter()
            .find(|e| e.kind == EdgeKind::Handoff)
            .unwrap();
        assert!(!edge_active(&on, handoff), "the hand-off never happened");
    }

    #[test]
    fn layout_is_deterministic() {
        let g = pair();
        assert_eq!(layout(&g), layout(&g));
    }

    #[test]
    fn a_handoff_cycle_terminates_and_draws_a_back_edge() {
        let g = AgentGraph {
            nodes: vec![
                node("a", NodeKind::Agent, "A", true),
                node("b", NodeKind::Agent, "B", false),
            ],
            edges: vec![
                edge("a", "b", EdgeKind::Handoff),
                edge("b", "a", EdgeKind::Handoff),
            ],
        };
        let l = layout(&g);
        assert_sane(&g, &l);
        assert_eq!(l.nodes.iter().find(|n| n.id == "a").unwrap().layer, 0);
        assert_eq!(l.nodes.iter().find(|n| n.id == "b").unwrap().layer, 1);
        assert!(l.edges.iter().any(|e| e.back), "B -> A goes back up");
        assert!(l.edges.iter().any(|e| !e.back), "A -> B goes down");
    }

    #[test]
    fn an_edge_to_a_missing_node_is_dropped() {
        let mut g = pair();
        g.edges
            .push(edge("agent-lattice", "ghost", EdgeKind::Handoff));
        g.edges
            .push(edge("ghost", "agent-advisor", EdgeKind::Handoff));
        let l = layout(&g);
        assert_sane(&g, &l);
        assert_eq!(l.edges.len(), 6);
    }

    #[test]
    fn empty_and_edgeless_graphs_lay_out() {
        let empty = layout(&AgentGraph::default());
        assert!(empty.nodes.is_empty() && empty.width >= 0.0 && empty.height >= 0.0);
        let lone = AgentGraph {
            nodes: vec![node("a", NodeKind::Agent, "Solo", false)],
            edges: vec![],
        };
        let l = layout(&lone);
        assert_sane(&lone, &l);
    }

    #[test]
    fn duplicate_ids_are_placed_once_and_unattached_tools_are_not_lost() {
        let g = AgentGraph {
            nodes: vec![
                node("a", NodeKind::Agent, "A", true),
                node("a", NodeKind::Agent, "A again", false),
                node("loose", NodeKind::Tool, "loose_tool", false),
            ],
            edges: vec![],
        };
        let l = layout(&g);
        assert_sane(&g, &l);
        assert_eq!(l.nodes.len(), 2);
    }

    struct Lcg(u64);
    impl Lcg {
        fn below(&mut self, n: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) % n.max(1)
        }
    }

    #[test]
    fn a_seeded_property_test_over_random_graphs() {
        let mut rng = Lcg(0xABCDEF);
        for case in 0..200 {
            let agents = 1 + rng.below(6) as usize;
            let mut nodes = vec![
                node("start", NodeKind::Start, "s", false),
                node("end", NodeKind::End, "e", false),
            ];
            let mut edges = vec![];
            for a in 0..agents {
                nodes.push(node(
                    &format!("agent-{a}"),
                    NodeKind::Agent,
                    &format!(
                        "Agent number {a} with a fairly long name {}",
                        "x".repeat(rng.below(30) as usize)
                    ),
                    a == 0,
                ));
                for t in 0..rng.below(5) {
                    let id = format!("tool-{a}-{t}");
                    nodes.push(node(
                        &id,
                        if rng.below(4) == 0 {
                            NodeKind::Mcp
                        } else {
                            NodeKind::Tool
                        },
                        &format!("tool_{a}_{t}"),
                        false,
                    ));
                    edges.push(edge(&format!("agent-{a}"), &id, EdgeKind::Tool));
                }
                for _ in 0..rng.below(3) {
                    edges.push(edge(
                        &format!("agent-{a}"),
                        &format!("agent-{}", rng.below(agents as u64 + 1)),
                        EdgeKind::Handoff,
                    ));
                }
                if rng.below(3) == 0 {
                    edges.push(edge(&format!("agent-{a}"), "end", EdgeKind::End));
                }
            }
            edges.push(edge("start", "agent-0", EdgeKind::Start));
            let g = AgentGraph { nodes, edges };
            let l = layout(&g);
            assert_sane(&g, &l);
            assert_eq!(l, layout(&g), "case {case}: deterministic");
        }
    }
}
