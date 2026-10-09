//! The runs column's rows: grouped, bounded and worded once, when the list or
//! the minute changes, never in `view()`.
//!
//! Invariants:
//! - Groups appear in the order Running, Today, Earlier and a group with no run
//!   is not shown; inside a group runs keep newest first.
//! - The list never shows more than [`LIST_CAP`] runs, so laying it out costs a
//!   bounded amount however long the history grows; the number left out is
//!   reported so the interface can say so.
//! - A row's relative time changes only with the minute, which is what the
//!   `lazy` widget's dependency in the view keys on.

use lattice_protocol::{Locality, RunStatus, RunSummary};

use crate::clock::{RunGroup, format_count, format_relative, group_of};

/// The most runs the list lays out.
pub const LIST_CAP: usize = 200;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunRow {
    pub id: String,
    /// The whole task; the row truncates it to its width when it draws.
    pub task: String,
    /// `Lattice assistant · Local model`.
    pub meta: String,
    pub status: RunStatus,
    /// The run's model is on another machine: the row says so.
    pub remote: bool,
    pub when: String,
    /// `1,234 tokens`, or empty when the server reported no usage.
    pub tokens: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ListEntry {
    Header { group: RunGroup, count: usize },
    Run(RunRow),
}

pub fn row_of(run: &RunSummary, now: f64) -> RunRow {
    RunRow {
        id: run.id.clone(),
        task: run.task.clone(),
        meta: format!("{} · {}", run.agent_label, run.model_label),
        status: run.status,
        remote: run.locality == Locality::Remote,
        when: if run.status.is_active() {
            "working".to_string()
        } else {
            format_relative(run.updated_at.max(run.created_at), now)
        },
        tokens: run
            .usage
            .map(|u| format!("{} tokens", format_count(u.total_tokens)))
            .unwrap_or_default(),
    }
}

/// The entries to list and how many runs were left out by the cap.
pub fn build_entries(runs: &[RunSummary], now: f64) -> (Vec<ListEntry>, usize) {
    let mut sorted: Vec<&RunSummary> = runs.iter().collect();
    sorted.sort_by(|a, b| {
        b.created_at
            .partial_cmp(&a.created_at)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let shown = sorted.len().min(LIST_CAP);
    let hidden = sorted.len() - shown;
    let mut entries = Vec::with_capacity(shown + 3);
    for group in [RunGroup::Running, RunGroup::Today, RunGroup::Earlier] {
        let members: Vec<&&RunSummary> = sorted
            .iter()
            .take(shown)
            .filter(|r| group_of(r.status, r.created_at, now) == group)
            .collect();
        if members.is_empty() {
            continue;
        }
        entries.push(ListEntry::Header {
            group,
            count: members.len(),
        });
        entries.extend(members.into_iter().map(|r| ListEntry::Run(row_of(r, now))));
    }
    (entries, hidden)
}

/// A number that changes once a minute, for a `lazy` dependency: the list's
/// relative times are the only thing that ages.
pub fn minute_bucket(now: f64) -> u64 {
    (now.max(0.0) / 60.0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_protocol::Usage;

    fn run(id: u8, status: RunStatus, created: f64, remote: bool) -> RunSummary {
        RunSummary {
            id: format!("{id:016x}"),
            task: format!("task {id}"),
            agent: "lattice-assistant".into(),
            agent_label: "Lattice assistant".into(),
            model: "local".into(),
            model_label: if remote {
                "Hosted model".into()
            } else {
                "Local model".into()
            },
            locality: if remote {
                Locality::Remote
            } else {
                Locality::Local
            },
            status,
            created_at: created,
            updated_at: created + 5.0,
            ended_at: None,
            trace_id: String::new(),
            usage: None,
            output: None,
            error: None,
            spans: 0,
        }
    }

    #[test]
    fn groups_come_in_order_and_empty_groups_are_left_out() {
        let now = 1_000_000.0;
        let runs = vec![
            run(1, RunStatus::Completed, now - 100.0, false),
            run(2, RunStatus::Running, now - 10.0, false),
            run(3, RunStatus::Failed, now - 200_000.0, true),
            run(4, RunStatus::Refused, now - 300_000.0, false),
        ];
        let (entries, hidden) = build_entries(&runs, now);
        assert_eq!(hidden, 0);
        let shape: Vec<String> = entries
            .iter()
            .map(|e| match e {
                ListEntry::Header { group, count } => format!("{}:{count}", group.title()),
                ListEntry::Run(r) => r.id.clone(),
            })
            .collect();
        let expected: Vec<String> = vec![
            "Running:1".into(),
            format!("{:016x}", 2),
            "Today:1".into(),
            format!("{:016x}", 1),
            "Earlier:2".into(),
            format!("{:016x}", 3),
            format!("{:016x}", 4),
        ];
        assert_eq!(shape, expected);

        let (only_today, _) = build_entries(&runs[..1], now);
        assert_eq!(only_today.len(), 2);
        assert!(matches!(
            only_today[0],
            ListEntry::Header {
                group: RunGroup::Today,
                count: 1
            }
        ));
        assert!(build_entries(&[], now).0.is_empty());
    }

    #[test]
    fn the_list_is_bounded_and_says_how_many_it_left_out() {
        let now = 1_000_000.0;
        let runs: Vec<RunSummary> = (0..(LIST_CAP + 37))
            .map(|i| RunSummary {
                id: format!("{i:016x}"),
                ..run(1, RunStatus::Completed, now - i as f64, false)
            })
            .collect();
        let (entries, hidden) = build_entries(&runs, now);
        assert_eq!(hidden, 37);
        let shown = entries
            .iter()
            .filter(|e| matches!(e, ListEntry::Run(_)))
            .count();
        assert_eq!(shown, LIST_CAP);
        // The newest are the ones kept.
        let ListEntry::Run(first) = &entries[1] else {
            panic!("first run")
        };
        assert_eq!(first.id, format!("{:016x}", 0));
    }

    #[test]
    fn a_row_is_worded_for_a_person() {
        let now = 1_000_000.0;
        let mut r = run(9, RunStatus::Completed, now - 150.0, true);
        r.usage = Some(Usage {
            requests: 3,
            input_tokens: 1000,
            output_tokens: 234,
            total_tokens: 1234,
        });
        let row = row_of(&r, now);
        assert_eq!(row.meta, "Lattice assistant · Hosted model");
        assert!(row.remote);
        assert_eq!(row.tokens, "1,234 tokens");
        assert_eq!(row.when, "2 min ago");
        let working = row_of(&run(1, RunStatus::Running, now - 3.0, false), now);
        assert_eq!(working.when, "working");
        assert_eq!(working.tokens, "");
        assert!(!working.remote);
    }

    #[test]
    fn the_minute_bucket_changes_only_once_a_minute() {
        assert_eq!(minute_bucket(59.9), minute_bucket(0.0));
        assert_ne!(minute_bucket(60.0), minute_bucket(59.9));
        assert_eq!(minute_bucket(-5.0), 0);
    }
}
