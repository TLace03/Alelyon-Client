//! Replay: an agent turn's model input, from the shared transcript and the
//! sidecar (the chat core's spec §3.4 UT1, §4.4, §4.6, §5.4, §5.7.2, T1).
//! Not a port.
//!
//! 1. The visible shared turns (the last 400, as `load` gives them), oldest
//!    first, **stopping before the current turn's user record** (UT1): the
//!    run's own input is `[User(text)]`, so the reader's words reach the
//!    model exactly once.
//! 2. For each user turn: `User(text)`; then that turn's sidecar items in
//!    order, each run of `ToolCall`s as one `Assistant{tool_calls}` followed
//!    by its `ToolResult`s; then `Assistant{text}` for the visible answer
//!    that follows it (an empty answer, an error turn, is left out). A
//!    sidecar item is replayed only when its turn is a visible user turn, so
//!    a turn superseded by either Lattice hides its tool trail and nothing is
//!    removed (§5.4).
//!    - **Reasoning** (§22.8 RP3): each `Reasoning` item of the turn goes
//!      with the assistant output that followed it, as `<think>…</think>`
//!      text ([`think_block`]) at the start of that assistant message: the
//!      tool-call message it preceded, or the visible answer. Every target
//!      the agents crate speaks to is a chat-completions endpoint, which
//!      takes reasoning only as text; no provider-native reasoning item is
//!      sent. Reasoning after the last call of a turn with no visible answer
//!      has no assistant output to go with and is left out.
//!    - **A call with no result** (Lattice closed while it waited on an
//!      approval or a question, §4.6) is answered with [`CLOSED_BEFORE`],
//!      once.
//!    - **Every target gets the same items** (Amendment 4, RP1/RP2): tool
//!      items read under a local model replay to a remote target in full.
//!      Their number ([`local_results`]) is what `FirstRemoteSend` states, as
//!      information; a refused dialog sends nothing (the caller's). The
//!      tripwire (T3) sees every item before a request leaves.
//! 3. **A budget**, in characters: `min(context_tokens × 3, 96,000)`, or
//!    96,000 when the context is unknown. The last two agent turns are kept
//!    whole; older tool results become `[output elided: <tool> <summary>, <n>
//!    bytes]` and older reasoning is left out; if still over, older turns lose their tool items, then the
//!    oldest exchanges go whole. The caller tells the model once, in the
//!    leading user item, how many exchanges were left out ([`lead_item`]).
//!
//! [`SidecarSession`] hands the result to the run as its `Session`; it
//! persists nothing (`add_items` is a no-op): the sidecar's tool items are
//! written from the stream (§5.7.2), so a call waiting on an approval is
//! recorded before anything completes.

use std::collections::{HashMap, HashSet};

use futures::future::BoxFuture;
use lattice_agents::model::{InputItem, ToolCallItem};
use lattice_agents::session::{Session, SessionError};
use lattice_protocol::Locality;

use super::item::{Item, Payload};
use crate::chat::store::StoredTurn;

/// What the model reads for a call that never got its answer.
pub const CLOSED_BEFORE: &str = "The user closed Lattice before deciding; the call did not run.";
/// The budget when the context is unknown, in characters.
pub const DEFAULT_BUDGET: usize = 96_000;
/// Characters per token, conservatively.
pub const CHARS_PER_TOKEN: usize = 3;
/// Agent turns kept whole by the budget.
pub const KEEP_WHOLE: usize = 2;

/// The budget for a model with `context_tokens` (§5.7.2 step 3).
pub fn budget(context_tokens: Option<u32>) -> usize {
    context_tokens.map_or(DEFAULT_BUDGET, |tokens| {
        (tokens as usize)
            .saturating_mul(CHARS_PER_TOKEN)
            .min(DEFAULT_BUDGET)
    })
}

/// How to replay.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// The current turn's user record: replay stops before it (UT1).
    pub stop_before: Option<String>,
    pub context_tokens: Option<u32>,
}

/// The replayed input and what was left out.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Replay {
    pub items: Vec<InputItem>,
    /// Earlier exchanges left out whole by the budget.
    pub left_out: usize,
}

#[derive(Clone, Debug)]
struct Call {
    call_id: String,
    tool: String,
    arguments: String,
    summary: String,
}

#[derive(Clone, Debug)]
struct Group {
    /// The reasoning recorded before these calls (RP3).
    reasoning: Vec<String>,
    calls: Vec<Call>,
    /// (call, output) in recorded order.
    results: Vec<(String, String)>,
}

#[derive(Clone, Debug, Default)]
struct Exchange {
    user: Option<String>,
    groups: Vec<Group>,
    /// The reasoning recorded after the turn's last call: it goes with the
    /// visible answer (RP3).
    reasoning: Vec<String>,
    answer: Option<String>,
    local: bool,
}

/// A turn's sidecar trail.
#[derive(Clone, Debug, Default)]
struct Trail {
    local: bool,
    groups: Vec<Group>,
    /// Reasoning not yet followed by a call.
    reasoning: Vec<String>,
}

/// Reasoning as the text a chat-completions target reads at the start of
/// an assistant message (RP3): one `<think>…</think>` block.
pub fn think_block(parts: &[String]) -> Option<String> {
    (!parts.is_empty()).then(|| format!("<think>\n{}\n</think>\n\n", parts.join("\n\n")))
}

fn reasoning_chars(parts: &[String]) -> usize {
    parts.iter().map(String::len).sum()
}

impl Exchange {
    fn has_tools(&self) -> bool {
        !self.groups.is_empty()
    }

    fn chars(&self) -> usize {
        let mut n = self.user.as_ref().map_or(0, String::len)
            + self.answer.as_ref().map_or(0, String::len)
            + reasoning_chars(&self.reasoning);
        for group in &self.groups {
            n += reasoning_chars(&group.reasoning);
            n += group
                .calls
                .iter()
                .map(|call| call.arguments.len() + call.tool.len())
                .sum::<usize>();
            n += group
                .results
                .iter()
                .map(|(_, out)| out.len())
                .sum::<usize>();
        }
        n
    }

    /// Leave the reasoning out (the budget, RP3).
    fn drop_reasoning(&mut self) {
        self.reasoning.clear();
        for group in &mut self.groups {
            group.reasoning.clear();
        }
    }
}

fn payload_text(payload: &Payload, read_blob: &dyn Fn(&str) -> Option<Vec<u8>>) -> String {
    match payload {
        Payload::Inline(text) => text.clone(),
        Payload::Blob {
            sha256, preview, ..
        } => read_blob(sha256)
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            .unwrap_or_else(|| preview.clone()),
    }
}

/// The sidecar's tool items, reasoning and locality, by turn.
fn by_turn(items: &[Item], read_blob: &dyn Fn(&str) -> Option<Vec<u8>>) -> HashMap<String, Trail> {
    let mut out: HashMap<String, Trail> = HashMap::new();
    for item in items {
        match item {
            Item::TurnStart { turn, resolved, .. } => {
                out.entry(turn.clone()).or_default().local = *resolved == Locality::Local;
            }
            Item::Reasoning { turn, text, .. } => {
                out.entry(turn.clone())
                    .or_default()
                    .reasoning
                    .push(payload_text(text, read_blob));
            }
            Item::ToolCall {
                turn,
                call_id,
                tool,
                arguments,
                summary,
                ..
            } => {
                let trail = out.entry(turn.clone()).or_default();
                let reasoning = std::mem::take(&mut trail.reasoning);
                let groups = &mut trail.groups;
                let call = Call {
                    call_id: call_id.clone(),
                    tool: tool.clone(),
                    arguments: payload_text(arguments, read_blob),
                    summary: summary.clone(),
                };
                match groups.last_mut() {
                    Some(group) if group.results.is_empty() => {
                        group.reasoning.extend(reasoning);
                        group.calls.push(call);
                    }
                    _ => groups.push(Group {
                        reasoning,
                        calls: vec![call],
                        results: Vec::new(),
                    }),
                }
            }
            Item::ToolResult {
                turn,
                call_id,
                output,
                ..
            } => {
                let groups = &mut out.entry(turn.clone()).or_default().groups;
                if let Some(group) = groups
                    .iter_mut()
                    .rev()
                    .find(|group| group.calls.iter().any(|call| &call.call_id == call_id))
                    && !group.results.iter().any(|(id, _)| id == call_id)
                {
                    group
                        .results
                        .push((call_id.clone(), payload_text(output, read_blob)));
                }
            }
            // A native regenerate: the turn's earlier tool trail is hidden
            // (§5.4); nothing is removed from the record.
            Item::Superseded { turns, .. } => {
                for turn in turns {
                    if let Some(trail) = out.get_mut(turn) {
                        trail.groups.clear();
                        trail.reasoning.clear();
                    }
                }
            }
            _ => {}
        }
    }
    // A call with no result gets [`CLOSED_BEFORE`], once (§4.6).
    for trail in out.values_mut() {
        for group in trail.groups.iter_mut() {
            let answered: HashSet<String> =
                group.results.iter().map(|(id, _)| id.clone()).collect();
            for call in &group.calls {
                if !answered.contains(&call.call_id) {
                    group
                        .results
                        .push((call.call_id.clone(), CLOSED_BEFORE.to_owned()));
                }
            }
        }
    }
    out
}

/// The visible exchanges up to (not including) `stop_before`.
fn exchanges(
    turns: &[StoredTurn],
    items: &[Item],
    stop_before: Option<&str>,
    read_blob: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Vec<Exchange> {
    let mut tools = by_turn(items, read_blob);
    // A message's images are named, not sent again (`super::images`).
    let pictures = super::images::counts(items);
    let mut out: Vec<Exchange> = Vec::new();
    for turn in turns {
        if turn.role == "user" {
            if Some(turn.id.as_str()) == stop_before {
                break;
            }
            let trail = tools.remove(&turn.id).unwrap_or_default();
            let user = match pictures.get(&turn.id) {
                Some(&count) if count > 0 => {
                    format!("{}\n\n{}", turn.text, super::images::note(count))
                }
                _ => turn.text.clone(),
            };
            out.push(Exchange {
                user: Some(user),
                groups: trail.groups,
                reasoning: trail.reasoning,
                answer: None,
                local: trail.local,
            });
        } else if !turn.text.is_empty() {
            match out.last_mut() {
                Some(exchange) if exchange.answer.is_none() => {
                    exchange.answer = Some(turn.text.clone());
                }
                _ => out.push(Exchange {
                    answer: Some(turn.text.clone()),
                    ..Exchange::default()
                }),
            }
        }
    }
    out
}

/// The number of tool results read on this machine that a remote target
/// will be sent: what `FirstRemoteSend` states (T1, as information since
/// Amendment 4 RP2).
pub fn local_results(turns: &[StoredTurn], items: &[Item], stop_before: Option<&str>) -> u32 {
    let none = |_: &str| None;
    exchanges(turns, items, stop_before, &none)
        .iter()
        .filter(|exchange| exchange.local)
        .map(|exchange| {
            exchange
                .groups
                .iter()
                .map(|group| group.results.len() as u32)
                .sum::<u32>()
        })
        .sum()
}

/// Build the replay (see the module header).
pub fn replay(
    turns: &[StoredTurn],
    items: &[Item],
    options: &Options,
    read_blob: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Replay {
    let mut exchanges = exchanges(turns, items, options.stop_before.as_deref(), read_blob);
    // The budget.
    let limit = budget(options.context_tokens);
    let total = |exchanges: &[Exchange]| exchanges.iter().map(Exchange::chars).sum::<usize>();
    let whole_from = {
        let agent_turns: Vec<usize> = exchanges
            .iter()
            .enumerate()
            .filter(|(_, exchange)| exchange.has_tools())
            .map(|(at, _)| at)
            .collect();
        agent_turns
            .len()
            .checked_sub(KEEP_WHOLE)
            .map_or(0, |first| agent_turns[first])
    };
    if total(&exchanges) > limit {
        for exchange in &mut exchanges[..whole_from] {
            exchange.drop_reasoning();
            for group in &mut exchange.groups {
                let names: HashMap<String, (String, String)> = group
                    .calls
                    .iter()
                    .map(|call| {
                        (
                            call.call_id.clone(),
                            (call.tool.clone(), call.summary.clone()),
                        )
                    })
                    .collect();
                for (call_id, output) in &mut group.results {
                    let (tool, summary) = names.get(call_id).cloned().unwrap_or_default();
                    *output = format!("[output elided: {tool} {summary}, {} bytes]", output.len());
                }
            }
        }
    }
    let mut at = 0;
    while total(&exchanges) > limit && at < whole_from {
        exchanges[at].groups.clear();
        exchanges[at].reasoning.clear();
        at += 1;
    }
    let mut left_out = 0;
    while total(&exchanges) > limit && exchanges.len() > 1 {
        exchanges.remove(0);
        left_out += 1;
    }
    let mut input = Vec::new();
    for exchange in exchanges {
        if let Some(user) = exchange.user {
            input.push(InputItem::User(user));
        }
        for group in exchange.groups {
            input.push(InputItem::Assistant {
                text: think_block(&group.reasoning),
                tool_calls: group
                    .calls
                    .into_iter()
                    .map(|call| ToolCallItem {
                        call_id: call.call_id,
                        name: call.tool,
                        arguments: call.arguments,
                    })
                    .collect(),
            });
            for (call_id, output) in group.results {
                input.push(InputItem::ToolResult { call_id, output });
            }
        }
        if let Some(answer) = exchange.answer {
            let text = match think_block(&exchange.reasoning) {
                Some(reasoning) => reasoning + &answer,
                None => answer,
            };
            input.push(InputItem::Assistant {
                text: Some(text),
                tool_calls: Vec::new(),
            });
        }
    }
    Replay {
        items: input,
        left_out,
    }
}

/// Continue's input (§4.4): the replay up to and including its last tool
/// result, with no new user item.
pub fn through_last_result(mut items: Vec<InputItem>) -> Vec<InputItem> {
    let last = items
        .iter()
        .rposition(|item| matches!(item, InputItem::ToolResult { .. }));
    if let Some(last) = last {
        items.truncate(last + 1);
    }
    items
}

/// The leading user item: the folder's rules (never the system prompt,
/// §6.4), how many earlier exchanges were left out, and what the agent must
/// hear at this model call (a conflict, a review, a note).
pub fn lead_item(rules: Option<&str>, left_out: usize, notes: &[String]) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    if let Some(rules) = rules {
        parts.push(rules.to_owned());
    }
    if left_out > 0 {
        parts.push(format!(
            "{left_out} earlier exchanges of this conversation were left out to fit the model's context."
        ));
    }
    parts.extend(notes.iter().cloned());
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

/// The run's `Session`: the replay, computed before the run; nothing is
/// persisted through it (§3.4, §5.7.2).
pub struct SidecarSession {
    items: Vec<InputItem>,
}

impl SidecarSession {
    pub fn new(items: Vec<InputItem>) -> Self {
        Self { items }
    }
}

impl Session for SidecarSession {
    fn get_items(
        &self,
        limit: Option<usize>,
    ) -> BoxFuture<'static, Result<Vec<InputItem>, SessionError>> {
        let items = match limit {
            Some(limit) => self.items[self.items.len().saturating_sub(limit)..].to_vec(),
            None => self.items.clone(),
        };
        Box::pin(async move { Ok(items) })
    }

    fn add_items(&self, _items: Vec<InputItem>) -> BoxFuture<'static, Result<(), SessionError>> {
        Box::pin(async { Ok(()) })
    }
}
