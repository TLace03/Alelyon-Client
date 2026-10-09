//! The static shape of an agent: agents, their tools, and who they can hand off to.
//!
//! Ports the walk in `agents.extensions.visualization` (0.22.3): `get_all_nodes`,
//! `get_all_edges` and `_GraphNodeIds`, producing data
//! ([`lattice_protocol::AgentGraph`]) where the SDK produces Graphviz DOT.
//!
//! The walk, as the SDK makes it:
//! - `start -> root agent`;
//! - `agent -> tool` for each tool the agent holds;
//! - `agent -> target` for each handoff, then the walk continues into the
//!   target, once per agent (agents are identified by allocation, not by name);
//! - `agent -> end` for an agent with no handoffs.
//!
//! Deviations, and why:
//! - The SDK draws a tool edge in both directions (dotted, agent to tool and
//!   tool back); one `Tool` edge from the agent says the same thing as data.
//! - Identical edges are written once. The SDK repeats an edge when an agent
//!   lists the same target twice; a duplicate carries no information.
//! - No MCP nodes: this port has no MCP servers. `NodeKind::Mcp` and
//!   `EdgeKind::Mcp` stay in the protocol for when it does.
//!
//! Node ids are `<kind>:<label>` (`agent:Model advisor`, `tool:calculate`) with
//! `-2`, `-3`, ... appended when two nodes would share one, so ids are stable,
//! unique and deterministic: the same agent always produces the same graph.
//! `start` and `end` are the literal ids `start` and `end`.

use std::collections::{HashMap, HashSet};

use lattice_protocol::{AgentGraph, EdgeKind, GraphEdge, GraphNode, NodeKind};

use crate::agent::Agent;

struct Builder {
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
    used_ids: HashSet<String>,
    /// Allocation of an agent -> its node id.
    agents: HashMap<usize, String>,
    /// Identity of a tool (its handler) -> its node id.
    tools: HashMap<usize, String>,
    edge_seen: HashSet<(String, String, EdgeKind)>,
}

impl Builder {
    fn unique_id(&mut self, prefix: &str, label: &str) -> String {
        let base = format!("{prefix}:{label}");
        let mut id = base.clone();
        let mut n = 2;
        while !self.used_ids.insert(id.clone()) {
            id = format!("{base}-{n}");
            n += 1;
        }
        id
    }

    fn add_node(&mut self, kind: NodeKind, prefix: &str, label: &str, root: bool) -> String {
        let id = self.unique_id(prefix, label);
        self.nodes.push(GraphNode {
            id: id.clone(),
            kind,
            label: label.to_owned(),
            root,
        });
        id
    }

    fn add_edge(&mut self, source: &str, target: &str, kind: EdgeKind) {
        if self
            .edge_seen
            .insert((source.to_owned(), target.to_owned(), kind))
        {
            self.edges.push(GraphEdge {
                source: source.to_owned(),
                target: target.to_owned(),
                kind,
            });
        }
    }

    /// Nodes first (the SDK's `_GraphNodeIds.visit` order: the agent, its
    /// tools, then each handoff target in turn), so ids are assigned in a
    /// stable order.
    fn visit_nodes(&mut self, agent: &Agent, identity: usize, root: bool) {
        if self.agents.contains_key(&identity) {
            return;
        }
        let id = self.add_node(NodeKind::Agent, "agent", &agent.name, root);
        self.agents.insert(identity, id);
        for tool in &agent.tools {
            let tool_identity = tool.identity();
            if !self.tools.contains_key(&tool_identity) {
                let id = self.add_node(NodeKind::Tool, "tool", &tool.name, false);
                self.tools.insert(tool_identity, id);
            }
        }
        for handoff in &agent.handoffs {
            self.visit_nodes(
                &handoff.agent,
                std::sync::Arc::as_ptr(&handoff.agent) as usize,
                false,
            );
        }
    }

    /// Edges in the SDK's `_get_all_edges` order.
    fn visit_edges(&mut self, agent: &Agent, identity: usize, visited: &mut HashSet<usize>) {
        if !visited.insert(identity) {
            return;
        }
        let agent_id = self.agents[&identity].clone();
        for tool in &agent.tools {
            let tool_id = self.tools[&tool.identity()].clone();
            self.add_edge(&agent_id, &tool_id, EdgeKind::Tool);
        }
        for handoff in &agent.handoffs {
            let target_identity = std::sync::Arc::as_ptr(&handoff.agent) as usize;
            let target_id = self.agents[&target_identity].clone();
            self.add_edge(&agent_id, &target_id, EdgeKind::Handoff);
            self.visit_edges(&handoff.agent, target_identity, visited);
        }
        if agent.handoffs.is_empty() {
            self.add_edge(&agent_id, "end", EdgeKind::End);
        }
    }
}

/// The graph of `agent`: itself, its tools, and everything it can hand off to.
pub fn agent_graph(agent: &Agent) -> AgentGraph {
    let mut builder = Builder {
        nodes: Vec::new(),
        edges: Vec::new(),
        used_ids: HashSet::from(["start".to_owned(), "end".to_owned()]),
        agents: HashMap::new(),
        tools: HashMap::new(),
        edge_seen: HashSet::new(),
    };
    builder.nodes.push(GraphNode {
        id: "start".into(),
        kind: NodeKind::Start,
        label: "Start".into(),
        root: false,
    });
    let root_identity = agent as *const Agent as usize;
    builder.visit_nodes(agent, root_identity, true);
    builder.nodes.push(GraphNode {
        id: "end".into(),
        kind: NodeKind::End,
        label: "End".into(),
        root: false,
    });

    let root_id = builder.agents[&root_identity].clone();
    builder.add_edge("start", &root_id, EdgeKind::Start);
    builder.visit_edges(agent, root_identity, &mut HashSet::new());
    AgentGraph {
        nodes: builder.nodes,
        edges: builder.edges,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::tool::{FunctionTool, strict_object_schema};

    fn tool(name: &str) -> FunctionTool {
        FunctionTool::new(
            name,
            "",
            strict_object_schema(json!({}), &[]),
            |_, _| async { Ok(String::new()) },
        )
    }

    fn edge(source: &str, target: &str, kind: EdgeKind) -> GraphEdge {
        GraphEdge {
            source: source.into(),
            target: target.into(),
            kind,
        }
    }

    #[test]
    fn the_lattice_pair_walks_like_the_sdks_visualization() {
        let advisor = Agent::builder("Model advisor")
            .tool(tool("list_models"))
            .build();
        let assistant = Agent::builder("Lattice assistant")
            .tool(tool("current_time"))
            .tool(tool("calculate"))
            .handoff_to(advisor)
            .build();
        let graph = agent_graph(&assistant);

        let labels: Vec<(&str, NodeKind, bool)> = graph
            .nodes
            .iter()
            .map(|n| (n.id.as_str(), n.kind, n.root))
            .collect();
        assert_eq!(
            labels,
            [
                ("start", NodeKind::Start, false),
                ("agent:Lattice assistant", NodeKind::Agent, true),
                ("tool:current_time", NodeKind::Tool, false),
                ("tool:calculate", NodeKind::Tool, false),
                ("agent:Model advisor", NodeKind::Agent, false),
                ("tool:list_models", NodeKind::Tool, false),
                ("end", NodeKind::End, false),
            ]
        );
        assert_eq!(
            graph.edges,
            [
                edge("start", "agent:Lattice assistant", EdgeKind::Start),
                edge(
                    "agent:Lattice assistant",
                    "tool:current_time",
                    EdgeKind::Tool
                ),
                edge("agent:Lattice assistant", "tool:calculate", EdgeKind::Tool),
                edge(
                    "agent:Lattice assistant",
                    "agent:Model advisor",
                    EdgeKind::Handoff
                ),
                edge("agent:Model advisor", "tool:list_models", EdgeKind::Tool),
                edge("agent:Model advisor", "end", EdgeKind::End),
            ]
        );
    }

    #[test]
    fn a_single_agent_goes_straight_to_the_end() {
        let solo = Agent::builder("Solo").build();
        let graph = agent_graph(&solo);
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(
            graph.edges,
            [
                edge("start", "agent:Solo", EdgeKind::Start),
                edge("agent:Solo", "end", EdgeKind::End)
            ]
        );
    }

    #[test]
    fn agents_are_identified_by_allocation_and_names_may_collide() {
        let first = Agent::builder("Helper").build();
        let second = Agent::builder("Helper").build();
        let root = Agent::builder("Root")
            .handoff_to(first)
            .handoff_to(second)
            .build();
        let graph = agent_graph(&root);
        let ids: Vec<&str> = graph.nodes.iter().map(|n| n.id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "start",
                "agent:Root",
                "agent:Helper",
                "agent:Helper-2",
                "end"
            ]
        );
        assert_eq!(
            graph
                .edges
                .iter()
                .filter(|e| e.kind == EdgeKind::Handoff)
                .count(),
            2
        );
    }

    #[test]
    fn a_target_reached_twice_is_drawn_once_and_a_shared_tool_is_one_node() {
        let shared = tool("shared");
        let leaf = Agent::builder("Leaf").tool(shared.clone()).build();
        let root = Agent::builder("Root")
            .tool(shared)
            .handoff_to(leaf.clone())
            .handoff_to(leaf)
            .build();
        let graph = agent_graph(&root);
        assert_eq!(
            graph
                .nodes
                .iter()
                .filter(|n| n.kind == NodeKind::Tool)
                .count(),
            1
        );
        assert_eq!(
            graph
                .nodes
                .iter()
                .filter(|n| n.kind == NodeKind::Agent)
                .count(),
            2
        );
        assert_eq!(
            graph
                .edges
                .iter()
                .filter(|e| e.kind == EdgeKind::Handoff)
                .count(),
            1
        );
        assert_eq!(
            graph
                .edges
                .iter()
                .filter(|e| e.kind == EdgeKind::Tool)
                .count(),
            2
        );
    }

    #[test]
    fn the_graph_is_deterministic_and_every_edge_has_endpoints() {
        let a = Agent::builder("A").tool(tool("t")).build();
        let b = Agent::builder("B").tool(tool("t")).handoff_to(a).build();
        let first = agent_graph(&b);
        assert_eq!(first, agent_graph(&b));
        let ids: HashSet<&str> = first.nodes.iter().map(|n| n.id.as_str()).collect();
        assert_eq!(ids.len(), first.nodes.len());
        for edge in &first.edges {
            assert!(
                ids.contains(edge.source.as_str()) && ids.contains(edge.target.as_str()),
                "{edge:?}"
            );
        }
        assert!(ids.contains("tool:t") && ids.contains("tool:t-2"));
    }
}
