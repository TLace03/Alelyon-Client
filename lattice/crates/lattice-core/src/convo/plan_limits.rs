//! What the reader's plan allows of the harness: tasks running at once,
//! helper agents working at once, and whether schedules and the project
//! manager (a group's plan and milestones) are on.
//!
//! - **Set by the application** ([`super::agent::AgentChat::set_plan_limits`]):
//!   the window reads the signed-in account's plan and passes its limits in;
//!   signed out, it passes Free's. Until it passes any (`None`), nothing here
//!   limits anything, so an application that sells no plan keeps today's
//!   behaviour.
//! - **Where they apply:** the tasks at once are the lower of the reader's
//!   own setting and the plan's ([`effective`]); a turn's helper places are
//!   the plan's ([`helper_places`]); `schedule_prompt` is refused and a due
//!   schedule waits without a plan that has schedules; a group's plan tools
//!   are not offered without the project manager.
//! - These limits are the application's: Lattice is public source, so they
//!   hold for the application as shipped.

use super::task_runs::Limit;

/// What a plan allows of the harness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanLimits {
    /// Tasks running at once ([`UNLIMITED`]: no limit).
    pub background_tasks: u32,
    /// Helper agents working at once in a turn ([`UNLIMITED`]: no limit).
    pub helper_agents: u32,
    /// Scheduled prompts and wake-ups.
    pub schedules: bool,
    /// A group's plan and milestones (the board is the application's).
    pub project_manager: bool,
}

impl PlanLimits {
    /// Free's.
    pub const FREE: PlanLimits = PlanLimits {
        background_tasks: 2,
        helper_agents: 4,
        schedules: false,
        project_manager: false,
    };
}

/// A count with no limit.
pub const UNLIMITED: u32 = u32::MAX;

/// The words for a feature that comes with a paid plan.
pub fn paid_only(feature: &str) -> String {
    format!(
        "{feature} come with a paid plan (Pro and up). The user can upgrade in Settings, under Account."
    )
}

/// The tasks that may run at once: the lower of the reader's own setting and
/// the plan's.
pub fn effective(own: Limit, plan: Option<PlanLimits>) -> Limit {
    match (own, plan) {
        (own, None) => own,
        (own, Some(plan)) if plan.background_tasks == UNLIMITED => own,
        (Limit::At(n), Some(plan)) => Limit::At(n.min(plan.background_tasks.max(1))),
        (Limit::None, Some(plan)) => Limit::At(plan.background_tasks.max(1)),
    }
}

/// Whether the plan, not the reader's own setting, is what holds the tasks
/// at `limit`.
pub fn plan_holds(own: Limit, plan: Option<PlanLimits>) -> bool {
    match (own, plan) {
        (_, None) => false,
        (_, Some(plan)) if plan.background_tasks == UNLIMITED => false,
        (Limit::None, Some(_)) => true,
        (Limit::At(n), Some(plan)) => plan.background_tasks.max(1) < n,
    }
}

/// A turn's helper places: the plan's, or the harness's own four.
pub fn helper_places(plan: Option<PlanLimits>) -> usize {
    match plan {
        None => super::helpers::MAX_HELPERS,
        Some(plan) if plan.helper_agents == UNLIMITED => tokio::sync::Semaphore::MAX_PERMITS,
        Some(plan) => plan.helper_agents.max(1) as usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lower_of_the_readers_limit_and_the_plans_holds() {
        let free = Some(PlanLimits::FREE);
        assert_eq!(effective(Limit::At(5), None), Limit::At(5), "no plan: the reader's own");
        assert_eq!(effective(Limit::None, None), Limit::None);
        assert_eq!(effective(Limit::At(5), free), Limit::At(2));
        assert_eq!(effective(Limit::At(1), free), Limit::At(1), "the reader may set fewer");
        assert_eq!(effective(Limit::None, free), Limit::At(2), "no limit of the reader's: the plan's");
        assert!(plan_holds(Limit::At(5), free) && !plan_holds(Limit::At(1), free));
        assert!(!plan_holds(Limit::At(5), None));
        assert_eq!(helper_places(None), super::super::helpers::MAX_HELPERS);
        assert_eq!(helper_places(free), 4);
        let pro = PlanLimits { background_tasks: 6, helper_agents: 12, schedules: true, project_manager: true };
        assert_eq!((effective(Limit::None, Some(pro)), helper_places(Some(pro))), (Limit::At(6), 12));
        let max = PlanLimits { background_tasks: UNLIMITED, helper_agents: UNLIMITED, ..pro };
        assert_eq!(effective(Limit::None, Some(max)), Limit::None, "unlimited: the reader's own");
        assert_eq!(effective(Limit::At(5), Some(max)), Limit::At(5));
        assert!(!plan_holds(Limit::None, Some(max)));
        assert!(helper_places(Some(max)) > 1_000_000);
    }
}
