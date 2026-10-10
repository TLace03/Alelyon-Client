//! Running a hook as the tool it is from runs it: its event as that tool's
//! JSON on standard input, its answer read as that tool reads it.

use std::ffi::OsString;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use lattice_sys::process::{JobLimits, SpawnRequest};
use serde_json::{Map, Value, json};

use super::{Event, Family, Hook};
use crate::env::Env;

/// The most of a hook's output that is read, per stream.
pub const MAX_OUTPUT: u64 = 1024 * 1024;
/// The most of a hook's words the agent or the reader is given.
pub const MAX_WORDS: usize = 4_000;

/// A Lattice tool's name as `family`'s hooks know it (their matchers and
/// `tool_name`); MCP tools are `mcp__<server>__<tool>` in all of them.
pub fn tool_name(family: Family, tool: &str) -> String {
    let named = match family {
        Family::Claude => match tool {
            "run_command" => "Bash",
            "read_file" => "Read",
            "edit_file" => "Edit",
            "write_file" => "Write",
            "glob" => "Glob",
            "grep" => "Grep",
            "list_dir" => "LS",
            "web_search" => "WebSearch",
            "browser_read" => "WebFetch",
            "spawn_agent" => "Task",
            "todo_write" => "TodoWrite",
            _ => tool,
        },
        Family::Cursor => match tool {
            "run_command" => "Shell",
            "read_file" => "Read",
            "edit_file" | "write_file" => "Write",
            "delete_file" => "Delete",
            "grep" => "Grep",
            "glob" => "Glob",
            "list_dir" => "LS",
            "spawn_agent" => "Task",
            _ => tool,
        },
        Family::Gemini => match tool {
            "run_command" => "run_shell_command",
            "edit_file" => "replace",
            "grep" => "search_file_content",
            "list_dir" => "list_directory",
            "web_search" => "google_web_search",
            "browser_read" => "web_fetch",
            _ => tool,
        },
    };
    named.to_owned()
}

/// The event's name in `hook`'s own tool.
fn event_name(hook: &Hook) -> &str {
    &hook.native
}

/// What happened, for the hook's input.
#[derive(Clone, Debug, Default)]
pub struct Happening<'a> {
    /// The conversation's id.
    pub session: &'a str,
    pub folder: Option<&'a Path>,
    pub tool: Option<&'a str>,
    pub input: Option<&'a Value>,
    pub output: Option<&'a str>,
    pub prompt: Option<&'a str>,
    /// A Stop hook already had the agent go on in this turn.
    pub stop_active: bool,
}

/// A tool's arguments as hooks read them: Lattice's own, with `file_path`
/// (absolute) beside a `path`, as Claude Code's and Cursor's tools name it.
fn tool_input(input: Option<&Value>, folder: Option<&Path>) -> Value {
    let mut value = input.cloned().unwrap_or_else(|| json!({}));
    if let (Some(map), Some(folder)) = (value.as_object_mut(), folder)
        && let Some(path) = map.get("path").and_then(Value::as_str).map(str::to_owned)
        && !map.contains_key("file_path")
    {
        map.insert("file_path".into(), json!(folder.join(path).display().to_string()));
    }
    value
}

/// The JSON `hook` is given on its standard input.
pub fn payload(hook: &Hook, what: &Happening<'_>) -> Value {
    let cwd = what.folder.map(|f| f.display().to_string()).unwrap_or_default();
    let mut map = Map::new();
    let mut put = |k: &str, v: Value| {
        map.insert(k.to_owned(), v);
    };
    match hook.family {
        Family::Claude | Family::Gemini => {
            put("session_id", json!(what.session));
            put("transcript_path", Value::Null);
            put("cwd", json!(cwd));
            put("hook_event_name", json!(event_name(hook)));
            if hook.family == Family::Gemini {
                put("timestamp", json!(chrono_now()));
            } else {
                put("permission_mode", json!("default"));
            }
        }
        Family::Cursor => {
            put("conversation_id", json!(what.session));
            put("generation_id", json!(what.session));
            put("hook_event_name", json!(event_name(hook)));
            put("workspace_roots", json!(if cwd.is_empty() { vec![] } else { vec![cwd.clone()] }));
            put("cwd", json!(cwd));
        }
    }
    if let Some(tool) = what.tool {
        let input = tool_input(what.input, what.folder);
        put("tool_name", json!(tool_name(hook.family, tool)));
        if hook.family == Family::Cursor {
            if let Some(command) = input.get("command").cloned() {
                put("command", command);
            }
            if let Some(file) = input.get("file_path").cloned() {
                put("file_path", file);
            }
        }
        put("tool_input", input);
        if let Some(output) = what.output {
            match hook.family {
                Family::Claude => put("tool_response", json!({"output": output})),
                Family::Gemini => put("tool_response", json!({"llmContent": output, "returnDisplay": output})),
                Family::Cursor => put("tool_output", json!(output)),
            }
        }
    }
    if let Some(prompt) = what.prompt {
        put("prompt", json!(prompt));
    }
    match hook.event {
        Event::Stop => {
            put("stop_hook_active", json!(what.stop_active));
            if hook.family == Family::Cursor {
                put("status", json!("completed"));
                put("loop_count", json!(u32::from(what.stop_active)));
            }
        }
        Event::SessionStart => put("source", json!("startup")),
        _ => {}
    }
    Value::Object(map)
}

fn chrono_now() -> String {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    format!("{}", now.as_secs())
}

/// What a hook decided.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Verdict {
    /// Block the action (for Stop: have the agent go on), with this reason.
    pub block: Option<String>,
    /// Words added to what the agent reads.
    pub context: Vec<String>,
    /// The tool's arguments, changed.
    pub input: Option<Value>,
    /// End the turn now, with this reason.
    pub halt: Option<String>,
    /// A hook that failed (it did not block), for the reader.
    pub notes: Vec<String>,
}

impl Verdict {
    /// Several hooks' verdicts together: any block blocks; words are joined.
    pub fn merge(&mut self, other: Verdict) {
        if let Some(reason) = other.block {
            self.block = Some(match self.block.take() {
                Some(before) => format!("{before}\n{reason}"),
                None => reason,
            });
        }
        self.context.extend(other.context);
        if other.input.is_some() {
            self.input = other.input;
        }
        if self.halt.is_none() {
            self.halt = other.halt;
        }
        self.notes.extend(other.notes);
    }
}

fn words(text: &str) -> String {
    let text = crate::secrets::redact(text.trim());
    match text.char_indices().nth(MAX_WORDS) {
        Some((at, _)) => format!("{}[...]", &text[..at]),
        None => text,
    }
}

fn text_of(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(words).filter(|t| !t.is_empty())
}

/// How `hook` answered: its exit code (`None` when it did not end), and what
/// it wrote.
pub fn verdict(hook: &Hook, code: Option<u32>, stdout: &str, stderr: &str) -> Verdict {
    let mut v = Verdict::default();
    let blocked = || {
        let said = words(stderr);
        if said.is_empty() { "A hook blocked it.".to_owned() } else { said }
    };
    match code {
        Some(2) => {
            v.block = Some(blocked());
            return v;
        }
        Some(0) => {}
        Some(other) => {
            v.notes.push(format!("{} hook exited with code {other}: {}", hook.native, words(stderr)));
            return v;
        }
        None => {
            v.notes.push(format!("{} hook did not answer in time and was ended.", hook.native));
            return v;
        }
    }
    let trimmed = stdout.trim();
    let Some(answer) = trimmed.starts_with('{').then(|| serde_json::from_str::<Value>(trimmed).ok()).flatten() else {
        // Plain words: context, for a prompt or a chat's start (as Claude Code reads them).
        if !trimmed.is_empty() && matches!(hook.event, Event::UserPromptSubmit | Event::SessionStart) && hook.family == Family::Claude {
            v.context.push(words(trimmed));
        }
        return v;
    };
    let specific = answer.get("hookSpecificOutput").cloned().unwrap_or(Value::Null);
    match hook.family {
        Family::Claude => {
            if answer.get("continue").and_then(Value::as_bool) == Some(false) {
                v.halt = Some(text_of(&answer, "stopReason").unwrap_or_else(|| "A hook ended the turn.".to_owned()));
            }
            let reason = text_of(&answer, "reason");
            if answer.get("decision").and_then(Value::as_str) == Some("block") {
                v.block = Some(reason.clone().unwrap_or_else(|| "A hook blocked it.".to_owned()));
            }
            if specific.get("permissionDecision").and_then(Value::as_str) == Some("deny") {
                v.block = Some(text_of(&specific, "permissionDecisionReason").unwrap_or_else(|| "A hook denied it.".to_owned()));
            }
            if let Some(input) = specific.get("updatedInput").filter(|i| i.is_object()) {
                v.input = Some(input.clone());
            }
            if let Some(context) = text_of(&specific, "additionalContext") {
                v.context.push(context);
            }
        }
        Family::Cursor => {
            let message = text_of(&answer, "agent_message").or_else(|| text_of(&answer, "user_message"));
            if answer.get("permission").and_then(Value::as_str) == Some("deny") {
                v.block = Some(message.clone().unwrap_or_else(|| "A hook denied it.".to_owned()));
            }
            if answer.get("continue").and_then(Value::as_bool) == Some(false) {
                v.block = Some(message.clone().unwrap_or_else(|| "A hook blocked it.".to_owned()));
            }
            if let Some(followup) = text_of(&answer, "followup_message") {
                v.block = Some(followup);
            }
            if let Some(input) = answer.get("updated_input").filter(|i| i.is_object()) {
                v.input = Some(input.clone());
            }
            if let Some(context) = text_of(&answer, "additional_context") {
                v.context.push(context);
            }
        }
        Family::Gemini => {
            if answer.get("continue").and_then(Value::as_bool) == Some(false) {
                v.halt = Some(text_of(&answer, "stopReason").unwrap_or_else(|| "A hook ended the turn.".to_owned()));
            }
            if matches!(answer.get("decision").and_then(Value::as_str), Some("deny" | "block")) {
                v.block = Some(text_of(&answer, "reason").unwrap_or_else(|| "A hook denied it.".to_owned()));
            }
            if let Some(input) = specific.get("tool_input").filter(|i| i.is_object()) {
                v.input = Some(input.clone());
            }
            if let Some(context) = text_of(&specific, "additionalContext") {
                v.context.push(context);
            }
        }
    }
    v
}

/// What a hook wrote, and how it ended (`None`: it did not end in time).
#[derive(Clone, Debug, PartialEq)]
pub struct Ran {
    pub code: Option<u32>,
    pub stdout: String,
    pub stderr: String,
}

/// The shell a hook's command runs in: Git for Windows' bash when git is
/// installed (as Claude Code runs hooks on Windows), else `cmd.exe`.
pub fn shell(env: &dyn Env, globals: &Path) -> Option<(PathBuf, Vec<OsString>)> {
    if let Ok(git) = crate::exec::resolve::resolve_program("git", env, None, globals) {
        for dir in git.real.ancestors().skip(1).take(4) {
            let bash = dir.join("bin").join("bash.exe");
            if bash.is_file() {
                return Some((bash, vec!["bash.exe".into(), "-c".into()]));
            }
        }
    }
    let root = env.var("SystemRoot").map(PathBuf::from)?;
    let cmd = root.join("System32").join("cmd.exe");
    cmd.is_file().then(|| (cmd, vec!["cmd.exe".into(), "/d".into(), "/s".into(), "/c".into()]))
}

/// Run `hook` with `input` on its standard input, in `cwd`, ending its tree at
/// its timeout. Blocking: call it off the runtime's threads.
pub fn execute(env: &dyn Env, globals: &Path, hook: &Hook, cwd: &Path, project: Option<&Path>, input: &Value) -> Result<Ran, String> {
    let (program, mut argv) = shell(env, globals).ok_or_else(|| "No shell was found to run hooks in.".to_owned())?;
    let mut command = hook.command.clone();
    if let Some(root) = &hook.plugin_root {
        command = command.replace("${CLAUDE_PLUGIN_ROOT}", &root.display().to_string());
    }
    argv.push(command.into());
    let mut block = crate::exec::spawn::child_environment(env, project, globals);
    let mut add = |name: &str, value: &Path| block.push((name.into(), value.as_os_str().to_owned()));
    if let Some(project) = project {
        add("CLAUDE_PROJECT_DIR", project);
        add("CURSOR_PROJECT_DIR", project);
        add("GEMINI_PROJECT_DIR", project);
    }
    if let Some(root) = &hook.plugin_root {
        add("CLAUDE_PLUGIN_ROOT", root);
    }
    let mut child = lattice_sys::process::spawn_with_input(&SpawnRequest {
        program: &program,
        argv: &argv,
        cwd,
        env: &block,
        limits: JobLimits::default(),
    })
    .map_err(|_| "The hook could not start.".to_owned())?;
    let bytes = serde_json::to_vec(input).unwrap_or_default();
    let stdin = child.take_stdin();
    let writer = std::thread::spawn(move || {
        if let Some(mut stdin) = stdin {
            let _ = stdin.write_all(&bytes);
        }
    });
    let read = |pipe: Option<std::fs::File>| {
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            if let Some(pipe) = pipe {
                let _ = pipe.take(MAX_OUTPUT).read_to_end(&mut bytes);
            }
            bytes
        })
    };
    let (out, err) = (read(child.take_stdout()), read(child.take_stderr()));
    let code = child.wait(Some(hook.timeout)).ok().flatten();
    let _ = child.kill_tree();
    let _ = writer.join();
    Ok(Ran {
        code,
        stdout: String::from_utf8_lossy(&out.join().unwrap_or_default()).into_owned(),
        stderr: String::from_utf8_lossy(&err.join().unwrap_or_default()).into_owned(),
    })
}

/// The folder a hook runs in: Cursor's own hooks run from `~/.cursor`, as
/// Cursor runs them; every other hook in the chat's folder, else beside its file.
pub fn cwd_for(hook: &Hook, folder: Option<&Path>) -> PathBuf {
    let beside = hook.file.parent().map(Path::to_path_buf).unwrap_or_default();
    match (hook.family, &hook.source, folder) {
        (Family::Cursor, super::Source::Cursor, _) => beside,
        (_, _, Some(folder)) => folder.to_path_buf(),
        _ => beside,
    }
}
