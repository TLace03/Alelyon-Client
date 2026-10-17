//! The project manager's board: each group of a project as a card in a
//! column (Planned, In progress, Waiting on you, Done), in the reader's order
//! of priority, with its progress and what it waits for.
//!
//! - **The column follows the work** ([`derived`]): a group waits on the
//!   reader while any of its topics asks for an approval, an answer or a
//!   decision; it is in progress while any works; it is done once its to-do
//!   lists are all done; it is planned while nothing in it has started or none
//!   of its steps is done yet; otherwise in progress.
//! - **The reader can move a card** (`groups::Groups::set_move`): the move is
//!   kept until the group's work changes (its topics' turns, [`work`]), and
//!   the card then follows the work again.
//! - **Progress:** the steps done of its topics' latest to-do lists, and its
//!   tasks, how many and how many work now.
//! - **Priority** is the reader's order (`groups::Groups::raise`); a task
//!   waiting for a slot starts by it (`super::tasks::start_next`).
//! - **The plan** (`groups::Plan`): its goal, its milestones done, the next
//!   one, and the first one past its date and not done (the group is behind;
//!   nothing moves the date).
//! - **Cost and time** ([`Spent`]), measured from the topics' records: the
//!   tokens each answer's provider reported (an answer whose provider reported
//!   none is counted apart, never as zero), the time from each message to its
//!   answer, and the commands that may cost money (`exec::cost`) the agent
//!   asked to run and how many the reader approved. A price is known only where it is: an
//!   answer made on this PC costs nothing, one by a lab's agent is on the
//!   reader's own subscription; any other answer's price is not known here,
//!   and the card says how many.

use std::collections::HashMap;

use super::groups::{Group, Milestone, Move, Plan, Status};

/// What the board reads of one topic.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TopicState {
    /// Its turns so far.
    pub turns: u64,
    pub running: bool,
    /// It waits for the reader: an approval, an answer to a question, a decision.
    pub approval: bool,
    pub question: bool,
    pub decision: bool,
    /// Its latest to-do list: steps done, and in all.
    pub todos_done: usize,
    pub todos_total: usize,
    /// It is a task's topic.
    pub task: bool,
    /// What its answers spent.
    pub spent: Spent,
}

/// What answers spent: measured, never estimated.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Spent {
    /// Tokens in and out, as the providers reported them.
    pub tokens_in: u64,
    pub tokens_out: u64,
    /// Answers whose provider reported no tokens (not counted above).
    pub unreported: usize,
    /// Seconds from each message to its answer.
    pub seconds: f64,
    /// Answers made on this PC (no charge), by a lab's agent on the reader's
    /// own subscription, and by any other provider (price not known here).
    pub local: usize,
    pub subscription: usize,
    pub unpriced: usize,
    /// Commands that may cost money the agent asked to run, and those the reader approved.
    pub costly_asked: usize,
    pub costly_approved: usize,
}

impl Spent {
    /// Both, added.
    pub fn plus(self, other: Spent) -> Spent {
        Spent {
            tokens_in: self.tokens_in + other.tokens_in,
            tokens_out: self.tokens_out + other.tokens_out,
            unreported: self.unreported + other.unreported,
            seconds: self.seconds + other.seconds,
            local: self.local + other.local,
            subscription: self.subscription + other.subscription,
            unpriced: self.unpriced + other.unpriced,
            costly_asked: self.costly_asked + other.costly_asked,
            costly_approved: self.costly_approved + other.costly_approved,
        }
    }
}

/// How an answer's provider is priced: made on this PC, by a lab's agent on
/// the reader's subscription, or by another provider (price not known here).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Priced {
    Local,
    Subscription,
    Unknown,
}

/// The pricing of the provider an answer names (`chat::vocab::Resolution::provider`).
pub fn priced(provider: &str) -> Priced {
    if provider.starts_with("llamacpp:") {
        Priced::Local
    } else if provider.starts_with("agent:") {
        Priced::Subscription
    } else {
        Priced::Unknown
    }
}

/// One stored turn as [`spent_by`] reads it: `(ts, user, provider, tokens in,
/// tokens out)`.
pub type TurnSpend<'a> = (f64, bool, &'a str, Option<i64>, Option<i64>);

/// What a topic's stored turns spent, oldest first.
pub fn spent_by(turns: &[TurnSpend]) -> Spent {
    let mut spent = Spent::default();
    let mut asked: Option<f64> = None;
    for (ts, user, provider, tokens_in, tokens_out) in turns {
        if *user {
            asked = Some(*ts);
            continue;
        }
        match (tokens_in, tokens_out) {
            (Some(i), Some(o)) => {
                spent.tokens_in += (*i).max(0) as u64;
                spent.tokens_out += (*o).max(0) as u64;
            }
            _ => spent.unreported += 1,
        }
        if let Some(at) = asked.take() {
            spent.seconds += (ts - at).max(0.0);
        }
        match priced(provider) {
            Priced::Local => spent.local += 1,
            Priced::Subscription => spent.subscription += 1,
            Priced::Unknown => spent.unpriced += 1,
        }
    }
    spent
}

/// One card.
#[derive(Clone, Debug, PartialEq)]
pub struct Card {
    pub group: String,
    pub name: String,
    pub status: Status,
    /// The reader put it in this column.
    pub moved: bool,
    /// Its topics that are not tasks', its tasks, and its tasks working now.
    pub topics: usize,
    pub tasks: usize,
    pub tasks_running: usize,
    pub todos_done: usize,
    pub todos_total: usize,
    /// What it waits for, in the reader's words: "an approval", "a question", "a decision".
    pub waits: Vec<&'static str>,
    /// Its place in the reader's order, 0 the highest.
    pub rank: usize,
    /// Its plan's goal (empty without one), its milestones done and in all,
    /// the next one, and the one it is behind on.
    pub goal: String,
    pub milestones_done: usize,
    pub milestones_total: usize,
    pub next: Option<Milestone>,
    pub behind: Option<Milestone>,
    /// What its topics and tasks spent.
    pub spent: Spent,
}

/// The column a group's work gives.
pub fn derived(states: &[TopicState]) -> Status {
    let (done, total): (usize, usize) =
        states.iter().fold((0, 0), |(d, t), s| (d + s.todos_done, t + s.todos_total));
    if states.iter().any(|s| s.approval || s.question || s.decision) {
        Status::Waiting
    } else if states.iter().any(|s| s.running) {
        Status::InProgress
    } else if total > 0 && done == total {
        Status::Done
    } else if work(states) == 0 || (total > 0 && done == 0) {
        Status::Planned
    } else {
        Status::InProgress
    }
}

/// A group's work, by which a move of its card lapses: its topics' turns.
pub fn work(states: &[TopicState]) -> u64 {
    states.iter().map(|s| s.turns).sum()
}

/// The cards of `project`'s unarchived groups (`groups` with their shown
/// names), highest priority first. `state` reads a topic (`None`: no longer
/// listed).
#[allow(clippy::too_many_arguments)]
pub fn cards(
    groups: &[(Group, String)],
    ranks: &HashMap<String, usize>,
    moves: &HashMap<String, Move>,
    plans: &HashMap<String, Plan>,
    today: &str,
    project: Option<&str>,
    state: impl Fn(&str) -> Option<TopicState>,
) -> Vec<Card> {
    let mut out: Vec<Card> = groups
        .iter()
        .filter(|(g, _)| !g.archived && g.project.as_deref() == project)
        .map(|(g, name)| {
            let states: Vec<TopicState> = g.topics.iter().filter_map(|t| state(t)).collect();
            let kept = moves.get(&g.id).filter(|m| m.work == work(&states));
            let mut waits = Vec::new();
            for (any, words) in [
                (states.iter().any(|s| s.approval), "an approval"),
                (states.iter().any(|s| s.question), "a question"),
                (states.iter().any(|s| s.decision), "a decision"),
            ] {
                if any {
                    waits.push(words);
                }
            }
            let plan = plans.get(&g.id);
            Card {
                spent: states.iter().fold(Spent::default(), |all, s| all.plus(s.spent)),
                goal: plan.map(|p| p.goal.clone()).unwrap_or_default(),
                milestones_done: plan.map_or(0, |p| p.milestones.iter().filter(|m| m.done).count()),
                milestones_total: plan.map_or(0, |p| p.milestones.len()),
                next: plan.and_then(super::groups::next).cloned(),
                behind: plan.and_then(|p| super::groups::behind(p, today)).cloned(),
                group: g.id.clone(),
                name: name.clone(),
                status: kept.map_or_else(|| derived(&states), |m| m.status),
                moved: kept.is_some(),
                topics: states.iter().filter(|s| !s.task).count(),
                tasks: states.iter().filter(|s| s.task).count(),
                tasks_running: states.iter().filter(|s| s.task && s.running).count(),
                todos_done: states.iter().map(|s| s.todos_done).sum(),
                todos_total: states.iter().map(|s| s.todos_total).sum(),
                waits,
                rank: ranks.get(&g.id).copied().unwrap_or(usize::MAX),
            }
        })
        .collect();
    out.sort_by_key(|c| c.rank);
    out
}
