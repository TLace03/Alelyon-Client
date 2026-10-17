//! Groups (`super::groups`): a message with no group open joins the group
//! whose topics match it, or starts one; in an open group it stays there; a
//! task joins its topic's group; the reader renames, merges, moves and
//! archives; earlier work is grouped once and waits for review.

use super::agent_tests::H;
use super::groups::{Placement, Plan};
use super::recall::Route;
use super::task_runs::Origin;
use crate::chat::transcript::NewTurn;

fn topic(h: &H, user: &str, answer: &str) -> String {
    let (row, _) = h.store.first_message(user, "local").unwrap();
    h.store.append(&row.id, NewTurn::assistant(answer, "local")).unwrap();
    row.id
}

/// Topics on other things, so the matched words are rare enough to score as
/// they would among a reader's many chats.
fn others(h: &H) {
    for (user, answer) in [
        ("Water the garden tomatoes.", "Watered twice."),
        ("Fix the bicycle brake cable.", "Cable replaced."),
        ("Bake sourdough bread loaves.", "Two loaves baked."),
        ("Translate the menu into French.", "Menu translated."),
        ("Tune the guitar strings.", "Strings tuned."),
        ("Paint the fence white.", "Fence painted."),
        ("Book the dentist visit.", "Visit booked."),
        ("Sort the photo albums.", "Albums sorted."),
    ] {
        topic(h, user, answer);
    }
}

fn names(h: &H) -> Vec<String> {
    h.chat.groups().0.into_iter().filter(|(g, _)| !g.archived).map(|(_, name)| name).collect()
}

/// Mutants: placement ignoring the scores (everything in one group, or a new
/// group each time); an open group's message leaving it; a group from
/// another project taking the message.
#[test]
fn a_message_joins_the_group_that_matches_it_or_starts_its_own_and_an_open_group_keeps_it() {
    let h = H::new("groups-route");
    others(&h);
    let pay = topic(&h, "Add stripe webhooks to checkout payments.", "The checkout payment webhooks are in.");
    let login = topic(&h, "Design the login page with email codes.", "The login page sends email codes.");
    let pay_group = h.chat.group_add(&pay, &Placement::New, None).unwrap();
    let login_group = h.chat.group_add(&login, &Placement::New, None).unwrap();
    assert_ne!(pay_group.id, login_group.id);
    assert_eq!(names(&h), ["Add stripe webhooks to checkout payments.", "Design the login page with email codes."]);

    let (route, into) = h.chat.route_work("Retry failed stripe webhooks for checkout payments.", None, None, None);
    assert_eq!((route, into), (Route::Return(pay.clone()), Placement::Group(pay_group.id.clone())));
    let (route, into) = h.chat.route_work("Write a poem about rivers and mountains.", None, None, None);
    assert_eq!((route, into), (Route::New, Placement::New));
    // A group of another project is not a match for work in no project.
    let elsewhere = h.chat.route_work("Retry failed stripe webhooks for checkout payments.", None, None, Some("p_other"));
    assert_eq!(elsewhere, (Route::New, Placement::New));

    // The login group is open: a message about payments stays in it, on its newest topic.
    let (route, into) = h.chat.route_work("Retry failed stripe webhooks for checkout payments.", Some(&login), Some(&login_group.id), None);
    assert_eq!((route, into), (Route::Continue, Placement::Group(login_group.id.clone())));
    let (route, into) = h.chat.route_work("Retry failed stripe webhooks for checkout payments.", None, Some(&login_group.id), None);
    assert_eq!((route, into), (Route::Return(login.clone()), Placement::Group(login_group.id.clone())));

    // An empty group the reader named: its first message starts its first topic.
    let named = h.chat.group_create("Release notes", None).unwrap();
    assert_eq!(h.chat.route_work("Draft them.", None, Some(&named.id), None), (Route::New, Placement::Group(named.id.clone())));
    assert!(h.chat.group_create("  ", None).is_err(), "a group the reader makes has a name");
}

/// Mutants: a task's match leading to a new topic rather than the chat the
/// task was started from; a matched group none of whose chats match starting
/// a new topic rather than continuing its newest; a group's name and plan
/// left out of the match.
#[test]
fn work_given_with_no_group_open_goes_to_the_chat_that_already_does_it() {
    let h = H::new("groups-desk");
    others(&h);
    // The payments group: its chat, a task started from it that did the chargebacks, and a later chat.
    let pay = topic(&h, "Add stripe webhooks to checkout payments.", "The checkout payment webhooks are in.");
    let pay_group = h.chat.group_add(&pay, &Placement::New, None).unwrap();
    let task = topic(&h, "Handle chargeback disputes and partial refunds.", "Chargeback disputes now refund partially.");
    h.chat.inner.task_runs.started(&task, Origin { from: pay.clone(), title: "Chargebacks".into(), at: 1.0, ..Origin::default() }).unwrap();
    h.chat.inner.groups.follow(&task, &pay, 10.0).unwrap();
    let later = topic(&h, "Rename the invoice email subject line.", "Renamed.");
    h.chat.group_add(&later, &Placement::Group(pay_group.id.clone()), None).unwrap();

    // What the task did goes to the chat that started it: the task's thread is its own, and no new topic begins.
    let (route, into) = h.chat.route_work("More chargeback disputes need partial refunds.", None, None, None);
    assert_eq!((route, into), (Route::Return(pay.clone()), Placement::Group(pay_group.id.clone())));

    // The docs group matches by its name and plan alone (none of its chats has these words): its continuing chat.
    let docs = topic(&h, "Set up the static site generator.", "Hugo is set up.");
    let docs_group = h.chat.group_add(&docs, &Placement::New, None).unwrap();
    h.chat.group_rename(&docs_group.id, "API reference").unwrap();
    let plan = Plan { goal: "Write the API reference for every endpoint.".into(), milestones: Vec::new(), edited: false };
    h.chat.inner.groups.set_plan(&docs_group.id, plan, true).unwrap();
    let (route, into) = h.chat.route_work("Write the API reference for the export endpoints.", None, None, None);
    assert_eq!((route, into), (Route::Return(docs.clone()), Placement::Group(docs_group.id.clone())));

    // Work no group does starts a group of its own.
    assert_eq!(h.chat.route_work("Write a poem about rivers and mountains.", None, None, None), (Route::New, Placement::New));
}

/// Mutants: a task left out of its topic's group; a moved topic left in two
/// groups; a merge that loses topics; an emptied unnamed group kept.
#[test]
fn tasks_follow_their_topic_and_the_reader_renames_moves_merges_and_archives() {
    let h = H::new("groups-adjust");
    let a = topic(&h, "Build the search index for products.", "Indexed.");
    let b = topic(&h, "Tune the search ranking weights.", "Tuned.");
    let c = topic(&h, "Write the onboarding emails.", "Written.");
    let g1 = h.chat.group_add(&a, &Placement::New, None).unwrap();
    h.chat.group_add(&b, &Placement::Group(g1.id.clone()), None).unwrap();
    let g2 = h.chat.group_add(&c, &Placement::New, None).unwrap();

    // A task started from `a` joins its group.
    h.chat.inner.groups.follow("task-1", &a, 10.0).unwrap();
    let of = |id: &str| h.chat.inner.groups.of(id).map(|g| g.id);
    assert_eq!(of("task-1"), Some(g1.id.clone()));
    h.chat.inner.groups.follow("task-2", "not-grouped", 10.0).unwrap();
    assert_eq!(of("task-2"), None);

    // Rename; an empty name gives back the first topic's title.
    assert_eq!(h.chat.group_rename(&g1.id, "Search").unwrap().name, "Search");
    assert_eq!(names(&h), ["Search", "Write the onboarding emails."]);
    h.chat.group_rename(&g1.id, "").unwrap();
    assert_eq!(names(&h)[0], "Build the search index for products.");

    // Move `c` into the search group: its own, unnamed and empty, is gone.
    h.chat.group_add(&c, &Placement::Group(g1.id.clone()), None).unwrap();
    assert_eq!(of(&c), Some(g1.id.clone()));
    assert!(h.chat.groups().0.iter().all(|(g, _)| g.id != g2.id), "an emptied group nobody named goes");
    // Split it out again, into a group of its own.
    let g3 = h.chat.group_add(&c, &Placement::New, None).unwrap();
    assert_eq!(h.chat.inner.groups.list().iter().filter(|g| g.topics.contains(&c)).count(), 1);

    // Merge the search group into it: every topic and task moves, the search group is gone.
    let merged = h.chat.group_merge(&g1.id, &g3.id).unwrap();
    assert_eq!(merged.topics, [c.clone(), a.clone(), b.clone(), "task-1".to_owned()]);
    assert!(h.chat.group_merge(&g3.id, &g3.id).is_err());
    assert_eq!(h.chat.groups().0.len(), 1);

    // Archived: kept, its topics too.
    let archived = h.chat.group_archive(&g3.id, true).unwrap();
    assert!(archived.archived && archived.topics.len() == 4);
    assert!(names(&h).is_empty());
}

/// Mutants: earlier work grouped twice; a task grouped apart from its
/// topic; projects mixed in one group; no review asked.
#[test]
fn earlier_work_is_grouped_once_by_project_with_tasks_beside_their_topic_and_waits_for_review() {
    let h = H::new("groups-existing");
    others(&h);
    let a = topic(&h, "Set up the payment checkout with stripe.", "Checkout uses stripe.");
    let b = topic(&h, "Stripe payment checkout refunds are missing.", "Refunds added to the checkout.");
    let c = topic(&h, "Plan the marketing site's blog.", "The blog has a plan.");
    let t = topic(&h, "Task: write the refund tests.", "Tests written.");
    h.chat.inner.task_runs.started(&t, Origin { from: b.clone(), title: "Refund tests".into(), at: 1.0, ..Origin::default() }).unwrap();
    let shop = h.runtime.block_on(h.chat.project_create("Shop".into())).unwrap();
    h.runtime.block_on(h.chat.project_assign(c.clone(), Some(shop.id.clone()))).unwrap();

    let made = h.chat.group_existing().unwrap();
    let groups = h.chat.inner.groups.list();
    assert_eq!(made, groups.len());
    let pay = groups.iter().find(|g| g.topics.contains(&a)).unwrap();
    assert_eq!(pay.topics, [a.clone(), b.clone(), t.clone()], "the task beside its topic");
    assert_eq!(pay.project, None);
    let blog = groups.iter().find(|g| g.topics.contains(&c)).unwrap();
    assert_eq!(blog.project.as_deref(), Some(shop.id.as_str()));
    assert!(h.chat.groups().1, "the grouping waits for the reader's review");

    h.chat.group_keep().unwrap();
    assert!(!h.chat.groups().1);
    topic(&h, "Another stripe checkout question.", "Answered.");
    assert_eq!(h.chat.group_existing().unwrap(), 0, "earlier work is grouped once");
    assert_eq!(h.chat.inner.groups.list().len(), made);
}
