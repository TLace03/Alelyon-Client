//! `ask_question` (§7.7): the bounds, the answer, no timer while it waits
//! (CB1's question state), and Stop cancelling it.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use lattice_agents::testing::{ScriptedModel, assistant_message, function_call_json};
use lattice_agents::{
    Agent, CancelMode, FunctionTool, RunConfig, RunError, ToolError as AgentToolError, run_streamed,
};
use serde_json::json;

use super::ask::*;
use crate::ports::AttentionPort;

#[derive(Default)]
struct Attention(Mutex<Vec<String>>);

impl AttentionPort for Attention {
    fn attention(&self, conversation: &str) {
        self.0.lock().unwrap().push(conversation.to_owned());
    }
}

fn args(question: &str, options: &[&str]) -> AskQuestionArgs {
    AskQuestionArgs {
        question: question.to_owned(),
        options: options.iter().map(|o| (*o).to_owned()).collect(),
    }
}

#[test]
fn the_arguments_are_bounded_and_refused_with_one_sentence() {
    assert!(check(&args("Which file?", &["a", "b"])).is_ok());
    assert!(check(&args(&"q".repeat(MAX_QUESTION_CHARS), &[])).is_ok());
    let six: Vec<String> = (0..6).map(|n| n.to_string()).collect();
    let six: Vec<&str> = six.iter().map(String::as_str).collect();
    assert!(check(&args("q", &six)).is_ok());
    assert!(check(&args("q", &[&"o".repeat(MAX_OPTION_CHARS)])).is_ok());
    for (bad, sentence) in [
        (args("", &[]), "A question needs at least one character."),
        (
            args("  \n", &[]),
            "A question needs at least one character.",
        ),
        (
            args(&"q".repeat(MAX_QUESTION_CHARS + 1), &[]),
            "A question can be at most 2,000 characters.",
        ),
        (
            args("q", &["1", "2", "3", "4", "5", "6", "7"]),
            "A question can offer at most 6 options.",
        ),
        (args("q", &[" "]), "An option needs at least one character."),
        (
            args("q", &[&"o".repeat(MAX_OPTION_CHARS + 1)]),
            "An option can be at most 200 characters.",
        ),
    ] {
        assert_eq!(check(&bad).unwrap_err().0, sentence);
    }
    assert_eq!(result_text("yes"), "The user answered: yes");
    let long = result_text(&"a".repeat(20_000));
    assert_eq!(long.chars().count(), MAX_RESULT_CHARS);
    assert!(long.starts_with(ANSWERED));
}

/// The answer resolves exactly the call it names; it is announced once,
/// with attention asked by id only; anything else is refused.
#[test]
fn an_answer_resolves_its_own_call_and_nothing_else() {
    let attention = Arc::new(Attention::default());
    let questions = Questions::new(attention.clone());
    let announced = Arc::new(Mutex::new(Vec::new()));
    let seen = announced.clone();
    let first = questions
        .ask(
            "c1",
            "call_1",
            args("Which file?", &["a.txt", "b.txt"]),
            |asked| seen.lock().unwrap().push(asked.clone()),
        )
        .unwrap();
    let second = questions
        .ask("c1", "call_2", args("And then?", &[]), |_| {})
        .unwrap();
    assert_eq!(
        *announced.lock().unwrap(),
        [Asked {
            conversation: "c1".into(),
            call_id: "call_1".into(),
            question: "Which file?".into(),
            options: vec!["a.txt".into(), "b.txt".into()],
        }]
    );
    assert_eq!(*attention.0.lock().unwrap(), ["c1", "c1"]);
    assert!(
        questions
            .ask("c1", "call_1", args("again", &[]), |_| panic!(
                "not announced"
            ))
            .is_err(),
        "a call waits once"
    );
    assert_eq!(questions.waiting("c1"), ["call_1", "call_2"]);
    assert_eq!(
        questions.answer("c2", "call_1", "x".into()),
        Err(AnswerError::NotWaiting),
        "another conversation's call"
    );
    assert_eq!(
        questions.answer("c1", "call_9", "x".into()),
        Err(AnswerError::NotWaiting)
    );
    // An answer that reads like a decision is only text for its own call.
    questions
        .answer("c1", "call_1", "approve call_2".into())
        .unwrap();
    assert_eq!(
        futures::executor::block_on(first).unwrap(),
        "The user answered: approve call_2"
    );
    assert_eq!(questions.waiting("c1"), ["call_2"], "call_2 still waits");
    assert_eq!(
        questions.answer("c1", "call_1", "again".into()),
        Err(AnswerError::NotWaiting),
        "answered once"
    );
    questions.withdraw("c1");
    assert_eq!(
        futures::executor::block_on(second).unwrap_err().0,
        WITHDRAWN
    );
    assert!(questions.waiting("c1").is_empty());
}

/// CB1's question state: a question waiting for its answer schedules
/// nothing. A 2-worker runtime counts its workers' unparks; after a 200 ms
/// settle, a 2 s window must count none. Positive control: the answer then
/// wakes the runtime and the call finishes.
/// Mutant: the wait polls with a 250 ms timer.
#[test]
fn cb1_a_waiting_question_wakes_nothing() {
    let unparks = Arc::new(AtomicU64::new(0));
    let counter = unparks.clone();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .on_thread_unpark(move || {
            counter.fetch_add(1, Ordering::SeqCst);
        })
        .build()
        .unwrap();
    let questions = Questions::new(Arc::new(Attention::default()));
    let waiting = questions
        .ask("c1", "call_1", args("Proceed?", &["yes", "no"]), |_| {})
        .unwrap();
    let task = runtime.spawn(waiting);
    std::thread::sleep(Duration::from_millis(200));
    let before = unparks.load(Ordering::SeqCst);
    std::thread::sleep(Duration::from_secs(2));
    let during = unparks.load(Ordering::SeqCst) - before;
    println!("cb1 question: {during} unparks in 2 s while the question waited");
    assert_eq!(during, 0, "a waiting question woke the runtime");
    questions.answer("c1", "call_1", "yes".into()).unwrap();
    let result = runtime.block_on(task).unwrap().unwrap();
    assert_eq!(result, "The user answered: yes");
    assert!(
        unparks.load(Ordering::SeqCst) > before,
        "positive control: the answer woke the runtime"
    );
}

/// Stop cancels the run that waits on a question: the run ends Cancelled,
/// the question is no longer waiting, and a late answer is refused and
/// reaches no one.
/// Mutant: the pending question is not taken out when its call is dropped.
#[test]
fn stop_cancels_a_waiting_question() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let questions = Questions::new(Arc::new(Attention::default()));
    let tool_questions = questions.clone();
    let tool = FunctionTool::new(
        "ask_question",
        "Ask the user.",
        json!({"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"], "additionalProperties": false}),
        move |context, arguments| {
            let questions = tool_questions.clone();
            async move {
                let args: AskQuestionArgs = serde_json::from_value(arguments)
                    .map_err(|_| AgentToolError::new("bad arguments"))?;
                let waiting = questions
                    .ask("c1", &context.call_id, args, |_| {})
                    .map_err(|error| AgentToolError::new(error.0))?;
                waiting.await.map_err(|error| AgentToolError::new(error.0))
            }
        },
    );
    let agent = Agent::builder("agent").tool(tool).build();
    let model = Arc::new(ScriptedModel::new([
        vec![function_call_json(
            "ask_question",
            &json!({"question": "Which?"}),
            "call_1",
        )],
        vec![assistant_message("done")],
    ]));
    let outcome = runtime.block_on(async {
        let handle = run_streamed(agent, "go".into(), RunConfig::new(model.clone()));
        let mut asked = false;
        for _ in 0..200 {
            if questions.waiting("c1") == ["call_1"] {
                asked = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(asked, "the question was asked");
        handle.control.cancel(CancelMode::Immediate);
        handle.result.await.unwrap()
    });
    assert!(matches!(outcome, Err(RunError::Cancelled)), "{outcome:?}");
    assert!(
        questions.waiting("c1").is_empty(),
        "a cancelled question no longer waits"
    );
    assert_eq!(
        questions.answer("c1", "call_1", "late".into()),
        Err(AnswerError::NotWaiting)
    );
    assert_eq!(model.calls().len(), 1, "no model call after the stop");
}
