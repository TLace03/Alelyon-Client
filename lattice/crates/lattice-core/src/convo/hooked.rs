//! The reader's hooks in Lattice's agent turns ([`crate::hooks`]): read once
//! when a turn begins, each asked for in the core's own dialog the first time
//! it would run, then run at its event.
//!
//! - **A tool** ([`around`]): its `PreToolUse` hooks run before it (a block
//!   ends the call with the hook's reason; changed arguments are used, except
//!   for a command or an MCP call the reader already approved, whose approved
//!   arguments stand), its `PostToolUse` hooks after it (their words follow
//!   the result the agent reads).
//! - **A message** (`UserPromptSubmit`, in `turn::begin` before anything is
//!   written): a block refuses the message with the hook's reason; words go
//!   just before the message.
//! - **A new chat** (`SessionStart`): words join the turn's leading context.
//! - **The agent stopping** (`Stop`, when a turn ends well): a hook that has it
//!   go on queues its reason as the next message, shown as the hook's; at most
//!   [`MAX_FOLLOW_UPS`] in a row.
//!
//! A hook that fails (another exit code, a timeout) is told in the chat as a
//! notice and blocks nothing.

use std::path::PathBuf;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use serde_json::Value;

use super::agent::{Convo, Inner};
use super::turn::TurnTools;
use crate::hooks::run::{self, Happening, Verdict};
use crate::hooks::{Event, Found, Hook, approvals};
use crate::ports::{ConfirmRequest, Initiated};
use crate::tools::read::ToolError;
use lattice_protocol::conversation::ConversationEventKind;

/// The most messages Stop hooks may send in a row.
pub const MAX_FOLLOW_UPS: u32 = 5;
/// How a Stop hook's message starts, so the reader and the next Stop hook know it.
pub const FOLLOW_UP: &str = "A Stop hook asks you to go on: ";

/// What happened, owned, for a hook run off the runtime.
#[derive(Clone, Debug, Default)]
pub(crate) struct Owned {
    pub tool: Option<String>,
    pub input: Option<Value>,
    pub output: Option<String>,
    pub prompt: Option<String>,
    pub stop_active: bool,
}

/// Ask the reader for `hook` the first time; `true` when it may run.
async fn may_run(inner: &Arc<Inner>, hook: &Hook, folder: Option<&std::path::Path>) -> bool {
    let path = approvals::file(&inner.config.state);
    if approvals::allowed(&path, hook) {
        return true;
    }
    let request = ConfirmRequest::RunHook {
        source: hook.source.label(),
        event: format!("{} ({})", hook.native, hook.event.label()),
        matcher: hook.matcher.text(),
        command: hook.command.clone(),
        file: hook.file.display().to_string(),
        folder: folder.map(|f| f.display().to_string()),
    };
    let yes = inner.confirmer.ask(&format!("hook:{}", hook.digest()), request, Initiated::Page).await;
    yes && approvals::allow(&path, hook, inner.now()).is_ok()
}

/// Run the hooks of `event` (for a tool event, those for `what.tool`), each
/// allowed by the reader, one after another; their verdicts together.
pub(crate) async fn fire(inner: &Arc<Inner>, found: &Arc<Found>, session: &str, folder: Option<PathBuf>, event: Event, what: Owned) -> Verdict {
    let hooks: Vec<Hook> = found.for_event(event, what.tool.as_deref()).into_iter().cloned().collect();
    let mut all = Verdict::default();
    for hook in hooks {
        if !may_run(inner, &hook, folder.as_deref()).await {
            continue;
        }
        let (env, globals) = (inner.config.env.clone(), inner.config.state.globals.clone());
        let (what, folder, session) = (what.clone(), folder.clone(), session.to_owned());
        let ran = tokio::task::spawn_blocking(move || {
            let happening = Happening {
                session: &session,
                folder: folder.as_deref(),
                tool: what.tool.as_deref(),
                input: what.input.as_ref(),
                output: what.output.as_deref(),
                prompt: what.prompt.as_deref(),
                stop_active: what.stop_active,
            };
            let input = run::payload(&hook, &happening);
            let cwd = run::cwd_for(&hook, folder.as_deref());
            run::execute(env.as_ref(), &globals, &hook, &cwd, folder.as_deref(), &input)
                .map(|ran| run::verdict(&hook, ran.code, &ran.stdout, &ran.stderr))
        })
        .await;
        match ran {
            Ok(Ok(verdict)) => all.merge(verdict),
            Ok(Err(why)) => all.notes.push(why),
            Err(_) => all.notes.push("A hook stopped unexpectedly.".to_owned()),
        }
    }
    all
}

/// The folder as hooks are given it: its path as people write it.
pub(crate) fn folder_of(workspace: &crate::workspace::Workspace) -> PathBuf {
    PathBuf::from(crate::workspace::shown_path(&workspace.root))
}

/// Tell the chat what failed.
pub(crate) fn tell(convo: &Convo, notes: &[String]) {
    for note in notes {
        convo.log.push(ConversationEventKind::Notice { text: format!("Hook: {note}") });
    }
}

/// The words hooks add to a result.
fn added(words: &[String]) -> String {
    words.iter().map(|w| format!("\n\n[A hook adds] {w}")).collect()
}

/// `work` (the tool `name` with `args`) with its hooks around it. `rewrite`
/// is `false` for a call the reader approved as it is (a command, an MCP
/// call): a hook's changed arguments are not used for it.
pub(crate) fn around(
    tools: Arc<TurnTools>,
    name: &str,
    args: Value,
    rewrite: bool,
    work: impl FnOnce(Value) -> BoxFuture<'static, Result<String, ToolError>> + Send + 'static,
) -> BoxFuture<'static, Result<String, ToolError>> {
    if tools.hooks.hooks.is_empty() {
        return work(args);
    }
    let name = name.to_owned();
    async move {
        let folder = tools.workspace.as_ref().map(folder_of);
        let what = Owned { tool: Some(name.clone()), input: Some(args.clone()), ..Owned::default() };
        let pre = fire(&tools.inner, &tools.hooks, &tools.convo.id, folder.clone(), Event::PreToolUse, what).await;
        tell(&tools.convo, &pre.notes);
        if let Some(reason) = pre.block.or(pre.halt) {
            return Err(ToolError(format!("A hook blocked this {name} call: {reason}")));
        }
        let mut words = pre.context;
        let args = match pre.input {
            Some(changed) if rewrite => changed,
            Some(_) => {
                words.push("a hook's change to these arguments was not used: you approved them as they were.".to_owned());
                args
            }
            None => args,
        };
        let result = work(args.clone()).await;
        let output = match &result {
            Ok(text) => text.clone(),
            Err(error) => error.0.clone(),
        };
        let what = Owned { tool: Some(name), input: Some(args), output: Some(output), ..Owned::default() };
        let post = fire(&tools.inner, &tools.hooks, &tools.convo.id, folder, Event::PostToolUse, what).await;
        tell(&tools.convo, &post.notes);
        words.extend(post.block);
        words.extend(post.context);
        if words.is_empty() {
            return result;
        }
        match result {
            Ok(text) => Ok(format!("{text}{}", added(&words))),
            Err(error) => Err(ToolError(format!("{}{}", error.0, added(&words)))),
        }
    }
    .boxed()
}
