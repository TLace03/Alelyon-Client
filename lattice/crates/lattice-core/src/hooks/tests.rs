use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{Value, json};

use super::run::{self, Happening, Verdict};
use super::*;
use crate::env::MapEnv;
use crate::state::StateRoot;
use crate::testkit::TempDir;

fn read(value: Value, source: Source, family: Family) -> (Vec<Hook>, Vec<Problem>) {
    formats::read(&value, &source, family, Path::new("C:/x/hooks.json"), None)
}

#[test]
fn claude_codes_hooks_are_read_with_their_matchers_and_timeouts() {
    let (hooks, problems) = read(
        json!({"hooks": {
            "PreToolUse": [{"matcher": "Bash|Edit", "hooks": [{"type": "command", "command": "check.sh", "timeout": 5}]}],
            "PostToolUse": [{"hooks": [{"type": "command", "command": "fmt.sh"}]}],
            "BeforeModel": [{"hooks": [{"type": "command", "command": "n.sh"}]}],
            "Stop": [{"hooks": [{"type": "prompt", "prompt": "done?"}]}]
        }, "permissions": {"allow": ["Bash"]}, "env": {"KEY": "secret-value"}}),
        Source::ClaudeCode,
        Family::Claude,
    );
    assert_eq!(hooks.len(), 2);
    let pre = hooks.iter().find(|h| h.event == Event::PreToolUse).unwrap();
    let post = hooks.iter().find(|h| h.event == Event::PostToolUse).unwrap();
    assert_eq!((pre.event, pre.command.as_str(), pre.timeout), (Event::PreToolUse, "check.sh", Duration::from_secs(5)));
    assert!(pre.applies("run_command"), "Bash is Lattice's run_command");
    assert!(pre.applies("edit_file") && !pre.applies("read_file"));
    assert_eq!(post.timeout, formats::DEFAULT_TIMEOUT);
    assert!(post.applies("anything"));
    let said: Vec<&str> = problems.iter().map(|p| p.sentence.as_str()).collect();
    assert!(said.iter().any(|s| s.contains("BeforeModel hooks are not run")), "{said:?}");
    assert!(said.iter().any(|s| s.contains("type \"prompt\" is not run")), "{said:?}");
}

#[test]
fn cursors_and_geminis_events_map_onto_lattices() {
    let (cursor, _) = read(
        json!({"version": 1, "hooks": {
            "beforeShellExecution": [{"command": "./audit.sh"}],
            "afterFileEdit": [{"command": "./fmt.sh", "timeout": 10}],
            "beforeSubmitPrompt": [{"command": "./prompt.sh"}],
            "stop": [{"command": "./stop.sh"}]
        }}),
        Source::Cursor,
        Family::Cursor,
    );
    let by: Vec<(Event, &str)> = cursor.iter().map(|h| (h.event, h.native.as_str())).collect();
    assert!(by.contains(&(Event::PreToolUse, "beforeShellExecution")));
    assert!(by.contains(&(Event::UserPromptSubmit, "beforeSubmitPrompt")));
    let shell = cursor.iter().find(|h| h.native == "beforeShellExecution").unwrap();
    assert!(shell.applies("run_command") && !shell.applies("edit_file"));
    let edit = cursor.iter().find(|h| h.native == "afterFileEdit").unwrap();
    assert!(edit.applies("write_file") && !edit.applies("run_command"));
    assert_eq!(edit.timeout, Duration::from_secs(10));

    let (gemini, _) = read(
        json!({"hooks": {
            "BeforeTool": [{"matcher": "write_file|replace", "hooks": [{"type": "command", "command": "g.sh", "timeout": 5000}]}],
            "BeforeAgent": [{"hooks": [{"type": "command", "command": "p.sh"}]}],
            "enabled": true
        }}),
        Source::GeminiCli,
        Family::Gemini,
    );
    let tool = gemini.iter().find(|h| h.native == "BeforeTool").unwrap();
    assert_eq!(tool.timeout, Duration::from_secs(5), "Gemini's timeouts are milliseconds");
    assert!(tool.applies("edit_file"), "Gemini's replace is Lattice's edit_file");
    assert_eq!(gemini.iter().find(|h| h.native == "BeforeAgent").unwrap().event, Event::UserPromptSubmit);

    // Antigravity's named hooks, one switched off.
    let (named, _) = read(
        json!({"hooks": {
            "guard": {"PreToolUse": [{"matcher": "run_command", "hooks": [{"type": "command", "command": "a.sh"}]}]},
            "off": {"enabled": false, "PostToolUse": [{"hooks": [{"type": "command", "command": "b.sh"}]}]}
        }}),
        Source::Antigravity,
        Family::Gemini,
    );
    assert_eq!(named.len(), 1);
    assert!(named[0].applies("run_command"));
}

#[test]
fn a_hooks_approval_is_kept_for_its_exact_text() {
    let dir = TempDir::new("hooks-approvals");
    let path = dir.path().join("hook_approvals.json");
    let (hooks, _) = read(json!({"hooks": {"Stop": [{"hooks": [{"command": "a.sh"}]}]}}), Source::Lattice, Family::Claude);
    let hook = &hooks[0];
    assert!(!approvals::allowed(&path, hook));
    approvals::allow(&path, hook, 1.0).unwrap();
    assert!(approvals::allowed(&path, hook));
    let mut changed = hook.clone();
    changed.command = "a.sh --more".into();
    assert!(!approvals::allowed(&path, &changed), "a changed command asks again");
    assert!(approvals::revoke(&path, &hook.digest()).unwrap());
    assert!(!approvals::allowed(&path, hook));
    std::fs::write(&path, "garbage").unwrap();
    assert!(!approvals::allowed(&path, hook), "an unreadable file allows nothing");
}

fn hook(family: Family, event: Event, native: &str) -> Hook {
    Hook {
        source: Source::Lattice,
        family,
        event,
        native: native.into(),
        matcher: Matcher::All,
        command: "x".into(),
        timeout: Duration::from_secs(5),
        file: PathBuf::from("C:/x/hooks.json"),
        plugin_root: None,
    }
}

#[test]
fn each_tools_answer_is_read_as_that_tool_reads_it() {
    let pre = hook(Family::Claude, Event::PreToolUse, "PreToolUse");
    assert_eq!(run::verdict(&pre, Some(2), "", "not on main\n").block.as_deref(), Some("not on main"));
    let deny = json!({"hookSpecificOutput": {"permissionDecision": "deny", "permissionDecisionReason": "no rm"}}).to_string();
    assert_eq!(run::verdict(&pre, Some(0), &deny, "").block.as_deref(), Some("no rm"));
    let rewrite = json!({"hookSpecificOutput": {"updatedInput": {"command": "ls"}}}).to_string();
    assert_eq!(run::verdict(&pre, Some(0), &rewrite, "").input, Some(json!({"command": "ls"})));
    let failed = run::verdict(&pre, Some(1), "", "oops");
    assert!(failed.block.is_none() && failed.notes[0].contains("code 1"), "another code does not block");
    assert!(run::verdict(&pre, None, "", "").notes[0].contains("in time"));

    let prompt = hook(Family::Claude, Event::UserPromptSubmit, "UserPromptSubmit");
    assert_eq!(run::verdict(&prompt, Some(0), "Today is Friday.\n", "").context, vec!["Today is Friday."]);

    let cursor = hook(Family::Cursor, Event::PreToolUse, "beforeShellExecution");
    let denied = json!({"permission": "deny", "agent_message": "use the wrapper"}).to_string();
    assert_eq!(run::verdict(&cursor, Some(0), &denied, "").block.as_deref(), Some("use the wrapper"));
    let stop = hook(Family::Cursor, Event::Stop, "stop");
    let again = json!({"followup_message": "run the tests"}).to_string();
    assert_eq!(run::verdict(&stop, Some(0), &again, "").block.as_deref(), Some("run the tests"), "go on with this");

    let gemini = hook(Family::Gemini, Event::PostToolUse, "AfterTool");
    let more = json!({"hookSpecificOutput": {"additionalContext": "3 tests failed"}}).to_string();
    assert_eq!(run::verdict(&gemini, Some(0), &more, "").context, vec!["3 tests failed"]);
    let halt = json!({"continue": false, "stopReason": "budget"}).to_string();
    assert_eq!(run::verdict(&gemini, Some(0), &halt, "").halt.as_deref(), Some("budget"));
}

#[test]
fn a_hook_is_given_its_tools_own_json() {
    let input = json!({"path": "src/a.rs", "content": "x"});
    let what = Happening {
        session: "c1",
        folder: Some(Path::new("C:/work")),
        tool: Some("write_file"),
        input: Some(&input),
        ..Happening::default()
    };
    let claude = run::payload(&hook(Family::Claude, Event::PreToolUse, "PreToolUse"), &what);
    assert_eq!(claude["tool_name"], "Write");
    assert_eq!(claude["hook_event_name"], "PreToolUse");
    assert_eq!(claude["session_id"], "c1");
    assert!(claude["tool_input"]["file_path"].as_str().unwrap().ends_with("a.rs"));
    let cursor = run::payload(&hook(Family::Cursor, Event::PreToolUse, "preToolUse"), &what);
    assert_eq!(cursor["conversation_id"], "c1");
    assert_eq!(cursor["workspace_roots"][0], "C:/work");
    let gemini = run::payload(&hook(Family::Gemini, Event::PreToolUse, "BeforeTool"), &what);
    assert_eq!(gemini["tool_name"], "write_file");
}

#[test]
fn files_are_found_where_each_tool_keeps_them_and_a_folders_only_when_trusted() {
    let dir = TempDir::new("hooks-sources");
    let home = dir.path().join("home");
    let folder = dir.path().join("work");
    let write = |path: PathBuf, text: &str| {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    };
    let claude = r#"{"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "c.sh"}]}]}}"#;
    write(home.join(".claude").join("settings.json"), claude);
    write(home.join(".cursor").join("hooks.json"), r#"{"version": 1, "hooks": {"stop": [{"command": "u.sh"}]}}"#);
    write(home.join(".gemini").join("settings.json"), r#"{"hooks": {"AfterAgent": [{"hooks": [{"type": "command", "command": "g.sh"}]}]}}"#);
    write(folder.join(".claude").join("settings.json"), claude.replace("c.sh", "f.sh").as_str());
    write(folder.join(".cursor").join("hooks.json"), "not json");
    let env = MapEnv::new().with("USERPROFILE", home.as_os_str()).with("HOME", home.as_os_str());
    let state = StateRoot::at(dir.path().join("state"));
    std::fs::create_dir_all(&state.globals).unwrap();
    write(sources::lattice_file(&state), r#"{"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "l.sh"}]}]}}"#);

    let untrusted = sources::discover(&env, &state, Some(&folder), false);
    let commands: Vec<&str> = untrusted.hooks.iter().map(|h| h.command.as_str()).collect();
    assert_eq!(commands, vec!["l.sh", "c.sh", "u.sh", "g.sh"], "a folder's own are read only when trusted");

    let trusted = sources::discover(&env, &state, Some(&folder), true);
    assert!(trusted.hooks.iter().any(|h| h.command == "f.sh" && h.source == Source::Folder(".claude/settings.json".into())));
    assert!(trusted.problems.iter().any(|p| p.file.ends_with("hooks.json") && p.sentence.contains("not JSON")));
    assert_eq!(trusted.for_event(Event::Stop, None).len(), 4);
}

#[test]
fn a_hook_runs_with_its_event_on_its_standard_input() {
    let dir = TempDir::new("hooks-run");
    let env = crate::env::ProcessEnv;
    let globals = dir.path().join("globals");
    std::fs::create_dir_all(&globals).unwrap();
    let Some((shell, _)) = run::shell(&env, &globals) else {
        eprintln!("UNMEASURED: no shell on this machine");
        return;
    };
    let bash = shell.file_name().is_some_and(|n| n.eq_ignore_ascii_case("bash.exe"));
    let mut h = hook(Family::Claude, Event::PreToolUse, "PreToolUse");
    // The hook reads its input and blocks a command that names `rm`.
    h.command = if bash {
        r#"grep -q '"command":"rm' && { echo 'no rm here' >&2; exit 2; } || exit 0"#.into()
    } else {
        r#"findstr /c:"\"command\":\"rm" >nul && (echo no rm here 1>&2 & exit /b 2) || exit /b 0"#.into()
    };
    let rm = json!({"command": "rm -rf x"});
    let what = Happening { session: "c1", folder: Some(dir.path()), tool: Some("run_command"), input: Some(&rm), ..Happening::default() };
    let ran = run::execute(&env, &globals, &h, dir.path(), Some(dir.path()), &run::payload(&h, &what)).unwrap();
    let verdict = run::verdict(&h, ran.code, &ran.stdout, &ran.stderr);
    assert_eq!(verdict.block.as_deref(), Some("no rm here"), "{ran:?}");
    let ls = json!({"command": "ls"});
    let what = Happening { input: Some(&ls), ..what };
    let ran = run::execute(&env, &globals, &h, dir.path(), Some(dir.path()), &run::payload(&h, &what)).unwrap();
    assert_eq!(ran.code, Some(0), "{ran:?}");

    // A hook that does not end is ended at its timeout.
    let mut slow = h.clone();
    slow.timeout = Duration::from_millis(500);
    slow.command = if bash { "sleep 30".into() } else { "ping -n 30 127.0.0.1 >nul".into() };
    let started = std::time::Instant::now();
    let ran = run::execute(&env, &globals, &slow, dir.path(), None, &json!({})).unwrap();
    assert_eq!(ran.code, None);
    assert!(started.elapsed() < Duration::from_secs(10));
}


#[test]
fn every_tools_other_events_map_onto_lattices_and_the_rest_say_why_not() {
    let (hooks, problems) = read(
        json!({"version": 1, "hooks": {
            "subagentStart": [{"command": "a"}], "subagentStop": [{"command": "b"}],
            "preCompact": [{"command": "c"}], "sessionEnd": [{"command": "d"}],
            "afterAgentResponse": [{"command": "e"}], "postToolUseFailure": [{"command": "f"}],
            "beforeTabFileRead": [{"command": "g"}], "afterTabFileEdit": [{"command": "h"}],
            "workspaceOpen": [{"command": "i"}], "afterAgentThought": [{"command": "j"}]
        }}),
        Source::Cursor,
        Family::Cursor,
    );
    let events: std::collections::BTreeSet<Event> = hooks.iter().map(|h| h.event).collect();
    for event in [
        Event::SubagentStart, Event::SubagentStop, Event::PreCompact, Event::SessionEnd, Event::AfterResponse,
        Event::PostToolUseFailure, Event::TabRead, Event::TabEdit, Event::WorkspaceOpen,
    ] {
        assert!(events.contains(&event), "{event:?}");
    }
    assert!(problems.iter().any(|p| p.sentence.contains("afterAgentThought hooks are not run: Lattice does not keep the model's thinking")));
    let (claude, _) = read(
        json!({"hooks": {"Notification": [{"hooks": [{"command": "n"}]}], "SubagentStop": [{"hooks": [{"command": "s"}]}],
                         "PreCompact": [{"matcher": "manual", "hooks": [{"command": "p"}]}], "SessionEnd": [{"hooks": [{"command": "e"}]}],
                         "SessionStart": [{"matcher": "resume", "hooks": [{"command": "r"}]}, {"matcher": "startup", "hooks": [{"command": "u"}]}]}}),
        Source::ClaudeCode,
        Family::Claude,
    );
    let found = Found { hooks: claude, problems: Vec::new() };
    let what = run::What::default();
    let start: Vec<&str> = found.for_event(Event::SessionStart, what.subject(Event::SessionStart)).iter().map(|h| h.command.as_str()).collect();
    assert_eq!(start, ["u"], "a chat's start is `startup`, not `resume`");
    assert!(found.for_event(Event::PreCompact, what.subject(Event::PreCompact)).is_empty(), "turns left out are `auto`, not `manual`");
    assert_eq!(found.for_event(Event::Notification, what.subject(Event::Notification)).len(), 1);
    let (gemini, problems) = read(
        json!({"hooks": {"PreCompress": [{"hooks": [{"command": "c"}]}], "BeforeModel": [{"hooks": [{"command": "m"}]}]}}),
        Source::GeminiCli,
        Family::Gemini,
    );
    assert_eq!(gemini[0].event, Event::PreCompact);
    assert!(problems[0].sentence.contains("BeforeModel hooks are not run: Lattice does not let a hook rewrite"));
    assert!(Event::TabRead.background() && Event::SessionEnd.background() && !Event::PreToolUse.background());
}

#[test]
fn antigravitys_named_hooks_are_read_as_its_documentation_gives_them() {
    let (hooks, _) = read(
        json!({
            "guard": {"PreToolUse": [{"matcher": "run_command|view_file", "hooks": [{"type": "command", "command": "g.sh", "timeout": 5}]}]},
            "nudge": {"PreInvocation": [{"command": "p.sh"}], "Stop": [{"command": "s.sh"}], "PostInvocation": [{"command": "i.sh"}]},
            "off": {"enabled": false, "PostToolUse": [{"matcher": "*", "hooks": [{"command": "o.sh"}]}]}
        }),
        Source::Antigravity,
        Family::Antigravity,
    );
    let by = |c: &str| hooks.iter().find(|h| h.command == c).unwrap();
    assert_eq!(hooks.len(), 4, "the switched-off hook is not read");
    assert_eq!(by("g.sh").timeout, Duration::from_secs(5), "seconds");
    assert_eq!(by("p.sh").timeout, formats::ANTIGRAVITY_TIMEOUT);
    assert!(by("g.sh").applies("read_file"), "Antigravity's view_file is Lattice's read_file");
    assert_eq!(by("p.sh").event, Event::UserPromptSubmit);
    assert_eq!((by("s.sh").event, by("i.sh").event), (Event::Stop, Event::Stop));

    // Its JSON in, and its answers read.
    let input = json!({"command": "ls"});
    let what = Happening { session: "c1", folder: Some(Path::new("C:/work")), tool: Some("run_command"), input: Some(&input), ..Happening::default() };
    let given = run::payload(by("g.sh"), &what);
    assert_eq!(given["conversationId"], "c1");
    assert_eq!(given["workspacePaths"][0], "C:/work");
    assert_eq!(given["toolCall"]["name"], "run_command");
    assert_eq!(given["toolCall"]["args"]["command"], "ls");
    let deny = json!({"decision": "deny", "reason": "not here"}).to_string();
    assert_eq!(run::verdict(by("g.sh"), Some(0), &deny, "").block.as_deref(), Some("not here"));
    let go_on = json!({"decision": "continue", "reason": "tests first"}).to_string();
    assert_eq!(run::verdict(by("s.sh"), Some(0), &go_on, "").block.as_deref(), Some("tests first"));
    let forced = json!({"terminationBehavior": "force_continue", "injectSteps": [{"userMessage": "check the logs"}]}).to_string();
    assert_eq!(run::verdict(by("i.sh"), Some(0), &forced, "").block.as_deref(), Some("check the logs"));
    let told = json!({"injectSteps": [{"ephemeralMessage": "the build is red"}]}).to_string();
    assert_eq!(run::verdict(by("p.sh"), Some(0), &told, "").context, vec!["the build is red"]);
}

#[test]
fn only_hooks_already_allowed_run_for_an_event_that_does_not_ask() {
    let dir = TempDir::new("hooks-allowed-only");
    let env = crate::env::ProcessEnv;
    let state = StateRoot::at(dir.path().join("state"));
    std::fs::create_dir_all(&state.globals).unwrap();
    let Some((shell, _)) = run::shell(&env, &state.globals) else {
        eprintln!("UNMEASURED: no shell on this machine");
        return;
    };
    let bash = shell.file_name().is_some_and(|n| n.eq_ignore_ascii_case("bash.exe"));
    let mut h = hook(Family::Cursor, Event::TabRead, "beforeTabFileRead");
    h.command = if bash {
        r#"grep -q 'API_KEY' && echo '{"permission": "deny"}' || echo '{"permission": "allow"}'"#.into()
    } else {
        "echo {\"permission\": \"deny\"}".into()
    };
    let found = Found { hooks: vec![h.clone()], problems: Vec::new() };
    let what = run::What { file: Some(".env".into()), content: Some("API_KEY=1".into()), ..Default::default() };
    let before = run::run_allowed(&env, &state, &found, Event::TabRead, "", Some(dir.path()), &what);
    assert_eq!(before, Verdict::default(), "not allowed: it did not run");
    approvals::allow(&approvals::file(&state), &h, 1.0).unwrap();
    let after = run::run_allowed(&env, &state, &found, Event::TabRead, "", Some(dir.path()), &what);
    assert!(after.block.is_some(), "allowed, it ran and refused: {after:?}");
    let given = run::payload(&h, &Happening { file: Some(".env"), content: Some("API_KEY=1"), folder: Some(dir.path()), ..Happening::default() });
    assert!(given["file_path"].as_str().unwrap().ends_with(".env"));
    assert_eq!(given["content"], "API_KEY=1");
}
