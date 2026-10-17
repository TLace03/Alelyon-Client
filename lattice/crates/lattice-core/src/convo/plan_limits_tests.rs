//! What the reader's plan allows of the harness (`super::plan_limits`), end
//! to end over the agent harness: with no plan set nothing changes; on Free
//! the tasks at once are capped at two, a turn has the plan's helper places,
//! `schedule_prompt` is refused and a due schedule waits, and a group's topic
//! is not offered the plan's tools (and they are refused); a paid plan lifts
//! each of them.

use std::time::Duration;

use lattice_agents::model::{InputItem, ModelRequest};
use lattice_agents::testing::ScriptedModel;
use serde_json::json;

use super::agent_tests::{H, call, say};
use super::groups::Placement;
use super::helpers_steer_tests::routed;
use super::plan_limits::PlanLimits;
use super::task_runs::Limit;

const PRO: PlanLimits = PlanLimits { background_tasks: 6, helper_agents: 12, schedules: true, project_manager: true };

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
fn the_plan_caps_the_tasks_at_once_and_no_plan_changes_nothing() {
    let h = H::new("plan-tasks");
    h.chat.set_task_limit(Limit::None).unwrap();
    assert_eq!(h.chat.effective_task_limit(), Limit::None, "no plan: the reader's own");
    h.chat.set_plan_limits(Some(PlanLimits::FREE));
    assert_eq!(h.chat.effective_task_limit(), Limit::At(2));
    let full = super::task_runs::full_of(&h.chat.inner);
    assert!(full.contains("the most your plan allows"), "{full}");
    h.chat.set_task_limit(Limit::At(1)).unwrap();
    assert_eq!(h.chat.effective_task_limit(), Limit::At(1), "the reader may run fewer");
    assert!(super::task_runs::full_of(&h.chat.inner).contains("your limit allows"));
    h.chat.set_plan_limits(Some(PRO));
    h.chat.set_task_limit(Limit::None).unwrap();
    assert_eq!(h.chat.effective_task_limit(), Limit::At(6));
    h.chat.set_plan_limits(None);
    assert_eq!(h.chat.effective_task_limit(), Limit::None);
}

#[test]
fn a_turn_has_the_plans_helper_places_and_on_free_schedules_wait_for_a_paid_plan() {
    let h = H::new("plan-free");
    let ws = h.workspace();
    // Two places, not the harness's own four, so the plan's number is the one seen.
    h.chat.set_plan_limits(Some(PlanLimits { helper_agents: 2, ..PlanLimits::FREE }));
    let helper = ScriptedModel::new((0..2).map(|_| say("working"))).with_delay(Duration::from_secs(3));
    let background = |n: usize| call("spawn_agent", json!({"task": format!("Task {n}."), "background": true}), &format!("s{n}"));
    let model = routed(
        &h,
        vec![
            background(1),
            background(2),
            background(3),
            call("schedule_prompt", json!({"title": "Check CI", "prompt": "Look at the build.", "in_minutes": 60}), "c1"),
            say("Done."),
        ],
        helper,
    );
    let id = h.agent(None, "send three helpers and schedule a check", &ws);
    h.turns_end(&id, 1);
    let calls = model.agent.calls();
    for n in 1..=2 {
        assert!(result_of(&calls, &format!("s{n}")).contains("is working"), "helper {n}");
    }
    assert!(result_of(&calls, "s3").contains("All 2 of this turn's helper places are taken"), "{}", result_of(&calls, "s3"));
    let refused = result_of(&calls, "c1");
    assert!(refused.contains("come with a paid plan"), "{refused}");
    assert!(h.chat.schedules().is_empty(), "nothing was scheduled");
    assert!(
        !h.confirm.asked().iter().any(|r| matches!(r, crate::ports::ConfirmRequest::Schedule { .. })),
        "the user was never asked"
    );
}

#[test]
fn on_free_a_groups_topic_has_no_plan_tools_and_a_paid_plan_brings_them_back() {
    let h = H::new("plan-pm");
    let ws = h.workspace();
    let model = h.script(vec![say("Started.")]);
    let topic = h.agent(None, "Build the checkout.", &ws);
    h.turns_end(&topic, 1);
    h.chat.group_add(&topic, &Placement::New, None).unwrap();
    h.chat.set_plan_limits(Some(PlanLimits::FREE));
    model.enqueue(say("Planned nothing."));
    h.agent(Some(&topic), "Plan it.", &ws);
    h.turns_end(&topic, 2);
    let offered = |calls: &[ModelRequest]| -> Vec<String> {
        calls.last().unwrap().tools.iter().map(|t| t.name.clone()).collect()
    };
    assert!(!offered(&model.calls()).iter().any(|t| t == "plan_group"), "not on Free");
    h.chat.set_plan_limits(Some(PRO));
    model.enqueue(say("Now I can plan."));
    h.agent(Some(&topic), "Plan it now.", &ws);
    h.turns_end(&topic, 3);
    assert!(offered(&model.calls()).iter().any(|t| t == "plan_group"), "back on a paid plan");
}
