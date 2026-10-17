//! Groups' plans (`super::groups::Plan`): the agent proposes and ticks, the
//! reader edits; the agent never sets or moves a date the plan has, and an
//! edited plan is not replaced; a milestone past its date puts the group
//! behind, on the board and in its turns.

use lattice_agents::model::{InputItem, ModelRequest};
use serde_json::json;

use super::agent_tests::{H, call, say};
use super::groups::{self, Milestone, PLAN_HEADER, Placement, Plan, behind, is_date, next, today};

fn plan(goal: &str, milestones: &[(&str, Option<&str>, bool)]) -> Plan {
    Plan {
        goal: goal.into(),
        milestones: milestones
            .iter()
            .map(|(t, d, done)| Milestone { title: (*t).into(), due: d.map(str::to_owned), done: *done })
            .collect(),
        edited: false,
    }
}

fn result_of(calls: &[ModelRequest], call_id: &str) -> String {
    calls
        .iter()
        .flat_map(|r| r.input.iter())
        .find_map(|i| match i {
            InputItem::ToolResult { call_id: id, output } if id == call_id => Some(output.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

/// Mutants: a date not checked; behind on a done or undated milestone;
/// today off by a day.
#[test]
fn dates_are_checked_and_a_milestone_past_its_date_and_not_done_is_behind() {
    assert!(is_date("2026-10-15") && is_date("2028-02-29"));
    for bad in ["2026-02-30", "2026-13-01", "26-10-15", "2026-1-5", "soon", ""] {
        assert!(!is_date(bad), "{bad}");
    }
    assert_eq!(today(1_792_022_400.0), "2026-10-15");
    let p = plan("Ship checkout", &[("Cart", Some("2026-10-10"), true), ("Cards", Some("2026-10-14"), false), ("Refunds", None, false)]);
    assert_eq!(behind(&p, "2026-10-15").map(|m| m.title.as_str()), Some("Cards"));
    assert_eq!(behind(&p, "2026-10-14"), None, "due today is not behind");
    assert_eq!(next(&p).map(|m| m.title.as_str()), Some("Cards"));
}

/// Mutants: the agent moving a date or unticking; an edited plan replaced by
/// the agent; a bad date kept.
#[test]
fn the_agent_proposes_and_ticks_the_reader_edits_and_dates_stay_put() {
    let h = H::new("plans-store");
    let (row, _) = h.store.first_message("Build the checkout.", "local").unwrap();
    let g = h.chat.group_add(&row.id, &Placement::New, None).unwrap();
    let store = &h.chat.inner.groups;
    store.set_plan(&g.id, plan("Ship checkout", &[("Cart", Some("2026-10-20"), false), ("Cards", None, false)]), false).unwrap();
    store.tick(&g.id, "1", true).unwrap();
    // The agent proposes again, with another date for Cart: the date stays, and Cart stays done.
    store.set_plan(&g.id, plan("Ship checkout", &[("cart", Some("2026-11-30"), false), ("Cards", Some("2026-10-25"), false)]), false).unwrap();
    let kept = h.chat.group_plan(&g.id).unwrap();
    assert_eq!(kept.milestones[0].due.as_deref(), Some("2026-10-20"));
    assert!(kept.milestones[0].done);
    assert_eq!(kept.milestones[1].due.as_deref(), Some("2026-10-25"), "a new date where there was none");
    assert!(store.set_plan(&g.id, plan("x", &[("y", Some("2026-02-30"), false)]), false).is_err());
    assert!(store.tick(&g.id, "Nope", true).is_err());

    // The reader edits it: their dates stand, and the agent's proposal is refused.
    let mut mine = kept.clone();
    mine.milestones[1].due = Some("2026-11-01".into());
    let edited = h.chat.set_group_plan(&g.id, mine).unwrap().unwrap();
    assert!(edited.edited);
    let refused = store.set_plan(&g.id, plan("Other", &[]), false).unwrap_err();
    assert_eq!(refused, groups::words::EDITED);
    assert_eq!(h.chat.group_plan(&g.id).unwrap().milestones[1].due.as_deref(), Some("2026-11-01"));
    // The agent can still tick.
    assert!(store.tick(&g.id, "Cards", true).unwrap().done);
    // Cleared by the reader.
    assert_eq!(h.chat.set_group_plan(&g.id, Plan::default()).unwrap(), None);
    assert!(h.chat.group_plan(&g.id).is_none());
}

/// Over the harness: a turn in a group's topic offers the plan's tools and is
/// told the plan; the agent proposes a plan and ticks a milestone; a topic in
/// no group gets neither; the board shows the plan and that the group is
/// behind.
/// Mutants: the tools offered everywhere, or nowhere; the plan not told.
#[test]
fn a_groups_turns_are_told_its_plan_and_its_agent_plans_and_ticks() {
    let h = H::new("plans-turns");
    let ws = h.workspace();
    let model = h.script(vec![say("Started.")]);
    let topic = h.agent(None, "Build the checkout.", &ws);
    h.turns_end(&topic, 1);
    let g = h.chat.group_add(&topic, &Placement::New, None).unwrap();
    h.chat.inner.groups.set_plan(&g.id, plan("Ship checkout", &[("Cart", Some("2020-01-01"), false), ("Cards", None, false)]), false).unwrap();

    model.enqueue(call("plan_group", json!({"goal": "Ship checkout", "milestones": [{"title": "Cart", "due": "2030-01-01"}, {"title": "Cards"}, {"title": "Refunds"}]}), "p1"));
    model.enqueue(call("tick_milestone", json!({"milestone": "Cart"}), "k1"));
    model.enqueue(say("Planned and ticked."));
    h.agent(Some(&topic), "Plan it.", &ws);
    h.turns_end(&topic, 2);
    let calls = model.calls();
    let asked = calls.iter().rev().find(|r| r.input.iter().any(|i| matches!(i, InputItem::User(t) if t.contains("Plan it.")))).unwrap();
    let tools: Vec<&str> = asked.tools.iter().map(|t| t.name.as_str()).collect();
    assert!(tools.contains(&"plan_group") && tools.contains(&"tick_milestone"), "{tools:?}");
    let lead = asked.input.iter().find_map(|i| match i { InputItem::User(t) if t.contains(PLAN_HEADER) => Some(t.clone()), _ => None }).expect("the plan is told");
    assert!(lead.contains("Goal: Ship checkout") && lead.contains("1. [ ] Cart (due 2020-01-01)") && lead.contains("The group is behind"), "{lead}");
    assert!(result_of(&calls, "p1").contains("3 milestones"), "{}", result_of(&calls, "p1"));
    assert!(result_of(&calls, "k1").contains("is marked done"));
    let now = h.chat.group_plan(&g.id).unwrap();
    assert_eq!(now.milestones.iter().map(|m| (m.title.as_str(), m.due.as_deref(), m.done)).collect::<Vec<_>>(),
        [("Cart", Some("2020-01-01"), true), ("Cards", None, false), ("Refunds", None, false)]);

    // On the board: the plan's progress and the next milestone; nothing behind once Cart is done.
    let card = h.chat.board(None).into_iter().find(|c| c.group == g.id).unwrap();
    assert_eq!((card.goal.as_str(), card.milestones_done, card.milestones_total), ("Ship checkout", 1, 3));
    assert_eq!(card.next.map(|m| m.title), Some("Cards".to_owned()));
    assert_eq!(card.behind, None);
    let mut late = now.clone();
    late.milestones[1].due = Some("2020-02-01".into());
    h.chat.set_group_plan(&g.id, late).unwrap();
    let card = h.chat.board(None).into_iter().find(|c| c.group == g.id).unwrap();
    assert_eq!(card.behind.map(|m| m.title), Some("Cards".to_owned()));

    // A topic in no group: no plan tools, no plan told.
    model.enqueue(say("Hi."));
    let loose = h.agent(None, "Something else.", &ws);
    h.turns_end(&loose, 1);
    let calls = model.calls();
    let last = calls.last().unwrap();
    assert!(!last.tools.iter().any(|t| t.name == "plan_group"));
    assert!(!last.input.iter().any(|i| matches!(i, InputItem::User(t) if t.contains(PLAN_HEADER) || t.contains("has no plan yet"))));
}
