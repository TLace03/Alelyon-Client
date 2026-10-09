//! `run_command`: every command asks unless a standing entry matches
//! (the chat core's spec §7.6, §9.2, §9.3; OD2). Not a port.
//!
//! A call goes through three steps, each its own function, so no command can
//! start without passing the ones before it:
//! 1. [`prepare`] checks the call and computes the card (X12): the text at
//!    most 8,000 characters and with no bidi control (X6: refused outright),
//!    the timeout 1–3,600 s (600 by default), the working folder through the
//!    path rules and a folder (X10), the policy engine's verdict (§9.2: Ask
//!    mode, an untrusted folder, a refused folder and another command running
//!    refuse; staged changes turn even a standing match into a question), the
//!    standing match (X1–X4), and `allow_always_offer` (X1–X3 hold).
//! 2. [`approve`] is the reader's Approve from the page, which **only asks**:
//!    a standing match needs no dialog; otherwise the core itself opens the
//!    native `RunCommand` dialog (`ConfirmPort`, CP1–CP3), and the command
//!    may run only if the reader confirms there. `Approve` is refused with a
//!    conflict while staged changes wait (the command gate) or another
//!    command runs in the folder (X14). The [`Approval`] it gives is the only
//!    way to reach step 3.
//! 3. [`run`] (blocking: one thread waits, one reads each pipe) claims the
//!    folder's one command slot (X14), takes the writer lease and the
//!    before-command checkpoint (X11: both before the spawn; if either fails
//!    the command does not run), and starts the program in a Job Object with
//!    the X7 environment, a `NUL` stdin and pipes (X5, X8):
//!    - **a standing match** runs its resolved program directly, with the
//!      argv that matched (no shell);
//!    - **approved once** runs `%SystemRoot%\System32\WindowsPowerShell\v1.0\
//!      powershell.exe -NoProfile -NonInteractive -EncodedCommand <base64 of
//!      UTF-16LE text>`, by absolute path, never from `PATH`: the text
//!      arrives exactly as the card showed it, with no command-line quoting
//!      in between (X6).
//!
//!    Stop, the timeout and the program's exit each end the whole tree. The
//!    after-command hook runs before the slot is given back, so a Keep can
//!    never land inside a command's before/after window.
//!
//! **Output (X9).** One blocking reader thread per pipe writes into one
//! buffer per command of at most 8 MiB: the first 1 MiB is kept, then a 7 MiB
//! ring. At exit the kept bytes become one blob in the conversation's record
//! (`BlobKind::Output`, redacted, T15). **The model receives at most 32
//! KiB:** the first 8 KiB and the last 24 KiB of the ANSI-stripped text, with
//! `[… N bytes omitted …]`, then "Exit code N after T s", "Stopped by the
//! user" or "Timed out after T s".
//!
//! **X14.** [`CommandSlots`] holds at most one running command per
//! workspace: a second is refused ("Another command is still running in this
//! folder."), and the review's Keep asks it whether a command runs.
//!
//! **Background commands** (tool parity with Claude Code's and Codex's:
//! `run_command {background: true}`): the same
//! three steps, the same card and dialog (which says it keeps running), the
//! same gates when it starts (staged changes, the lease, the before-command
//! checkpoint), but one of [`MAX_BACKGROUND`] background slots of the folder
//! instead of its one command slot, so a server or a watcher does not hold
//! every other command and the review's Keep. It has no timeout: it runs
//! until it exits, is stopped, or its conversation or Lattice closes (the Job
//! Object ends the tree). Its after-command hook runs when it ends, so its
//! effect is the files changed over its whole life; a foreground command's
//! effect while one runs may include the background command's writes.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::ffi::OsString;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use lattice_protocol::conversation::{
    AllowEntry, ApprovalDetail, CallId, ChangedPath, CheckpointId, CommandMode, ExitReason, Mode,
};
use lattice_sys::process::{Child, JobLimits};
use serde::Deserialize;

use super::allowlist::{Candidate, Permissions};
use super::command::{
    BIDI_SENTENCE, MAX_COMMAND_CHARS, NotEligible, eligible, has_bidi, may_be_entry,
};
use super::resolve::{Resolved, resolve_program};
use super::spawn::ChildSpec;
use crate::convo::sidecar::BlobKind;
use crate::env::Env;
use crate::git::runner::GitRunner;
use crate::policy::{
    AskKind, Because, Gates, Lease, PathClass, Reason, Standing, Target, ToolClass, Verdict, decide,
};
use crate::ports::{ConfirmRequest, Confirmer, Initiated, escape_for_dialog};
use crate::staging::Staging;
use crate::text::ansi_strip;
use crate::tools::read::ToolError;
use crate::workspace::paths::Want;
use crate::workspace::{Workspace, shown_path};

/// The timeout when the call names none.
pub const DEFAULT_TIMEOUT_S: u32 = 600;
/// The longest timeout a call may name.
pub const MAX_TIMEOUT_S: u32 = 3600;
/// X9: the bytes kept from the start of a command's output.
pub const HEAD_BYTES: usize = 1024 * 1024;
/// X9: the ring kept after the head.
pub const RING_BYTES: usize = 7 * 1024 * 1024;
/// X9: the model's share of the start of the stripped text.
pub const MODEL_HEAD: usize = 8 * 1024;
/// X9: the model's share of the end of the stripped text.
pub const MODEL_TAIL: usize = 24 * 1024;

/// What the reader is told on every card and dialog (X12).
pub const RUNS_AS_YOU: &str = "Runs as you, with your permissions. It can change files outside the folder and use the network; Lattice cannot prevent either on Windows yet.";
const NOT_CONFIRMED: &str = "You did not confirm running this command, so it did not run.";
const NOT_RECORDED: &str =
    "Lattice could not record the folder's state first, so the command did not run.";

/// `run_command`'s arguments.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct RunCommandArgs {
    pub command: String,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub timeout_s: Option<u32>,
    /// Keep running after the call returns (see the module header).
    #[serde(default)]
    pub background: bool,
}

/// Background commands a workspace may run at once.
pub const MAX_BACKGROUND: usize = 3;

/// Why a background command was not started: the folder's are all in use.
pub const BACKGROUND_FULL: &str =
    "Three background commands already run in this folder: stop one with stop_command first.";

// ------------------------------------------------------------------- X14

/// The running command of each workspace (X14), shared by every
/// conversation of the process.
#[derive(Clone, Debug, Default)]
pub struct CommandSlots(Arc<Mutex<Slots>>);

/// Each workspace's foreground command (X14) and its background commands.
#[derive(Debug, Default)]
pub struct Slots {
    foreground: BTreeMap<String, CallId>,
    background: BTreeMap<String, BTreeSet<CallId>>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl CommandSlots {
    /// Whether a command runs in the workspace `id` now.
    pub fn running(&self, workspace: &str) -> bool {
        lock(&self.0).foreground.contains_key(workspace)
    }

    /// The call running in the workspace `id`, if one is.
    pub fn holder(&self, workspace: &str) -> Option<CallId> {
        lock(&self.0).foreground.get(workspace).cloned()
    }

    /// How many background commands run in the workspace `id`.
    pub fn background(&self, workspace: &str) -> usize {
        lock(&self.0).background.get(workspace).map_or(0, BTreeSet::len)
    }

    /// Whether any command, foreground or background, runs in the workspace
    /// `id` (the writer lease is kept while one does).
    pub fn busy(&self, workspace: &str) -> bool {
        self.running(workspace) || self.background(workspace) > 0
    }

    /// Take the workspace's slot for `call`; `None` while another holds it.
    pub fn claim(&self, workspace: &str, call: &str) -> Option<SlotGuard> {
        let mut slots = lock(&self.0);
        if slots.foreground.contains_key(workspace) {
            return None;
        }
        slots.foreground.insert(workspace.to_owned(), call.to_owned());
        Some(SlotGuard {
            slots: self.clone(),
            workspace: workspace.to_owned(),
            background: None,
        })
    }

    /// Take one of the workspace's background slots for `call`; `None` when
    /// [`MAX_BACKGROUND`] are taken.
    pub fn claim_background(&self, workspace: &str, call: &str) -> Option<SlotGuard> {
        let mut slots = lock(&self.0);
        let calls = slots.background.entry(workspace.to_owned()).or_default();
        if calls.len() >= MAX_BACKGROUND || calls.contains(call) {
            return None;
        }
        calls.insert(call.to_owned());
        Some(SlotGuard {
            slots: self.clone(),
            workspace: workspace.to_owned(),
            background: Some(call.to_owned()),
        })
    }
}

/// A held slot; dropping it gives the slot back.
#[derive(Debug)]
pub struct SlotGuard {
    slots: CommandSlots,
    workspace: String,
    /// A background slot's call; `None` for the foreground slot.
    background: Option<CallId>,
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        let mut slots = lock(&self.slots.0);
        match &self.background {
            None => {
                slots.foreground.remove(&self.workspace);
            }
            Some(call) => {
                if let Some(calls) = slots.background.get_mut(&self.workspace) {
                    calls.remove(call);
                    if calls.is_empty() {
                        slots.background.remove(&self.workspace);
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------ ports

/// Starts a child: the core's one spawn (`exec::spawn::spawn`), or a test's
/// recorder around it.
pub trait Launcher: Send + Sync {
    fn launch(&self, spec: &ChildSpec, workspace: &Path) -> io::Result<Child>;
}

/// The real launcher: the X7 environment from Lattice's own.
pub struct SpawnLauncher {
    pub env: Arc<dyn Env>,
    pub globals: PathBuf,
}

impl Launcher for SpawnLauncher {
    fn launch(&self, spec: &ChildSpec, workspace: &Path) -> io::Result<Child> {
        super::spawn::spawn(spec, self.env.as_ref(), Some(workspace), &self.globals)
    }
}

/// What happened after a command (`changes::effects`, row E8).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AfterCommand {
    /// Lines the model reads after the output.
    pub notes: Vec<String>,
    /// The command's effect inside the folder, when Lattice could see it
    /// (a git folder): the checkpoints before and after, and the paths that
    /// changed between them.
    pub effect: Option<EffectSummary>,
}

/// A command's effect as the `CommandEffect` event carries it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectSummary {
    pub before: CheckpointId,
    pub after: CheckpointId,
    pub files: Vec<ChangedPath>,
}

/// What a command needs from the rest of the core around its run: the
/// writer lease (§6.5) and the checkpoints before and after it (§8.4, row
/// E8).
pub trait CommandHooks: Send + Sync {
    /// The workspace's writer lease: `Held` (taken now if it was free) or
    /// `Elsewhere`.
    fn lease(&self) -> Lease;
    /// The before-command checkpoint; `Err` is why it could not be taken,
    /// and the command then does not run.
    fn before(&self, call: &CallId) -> Result<Option<CheckpointId>, String>;
    /// After exit, stop or timeout, before the slot is given back.
    fn after(&self, call: &CallId, before: Option<CheckpointId>) -> AfterCommand;
}

/// What a command works with.
pub struct CommandContext<'a> {
    pub workspace: &'a Workspace,
    /// For the path rules of the working folder.
    pub runner: &'a GitRunner,
    /// The conversation's staged changes (the command gate) and its record
    /// (the output blob).
    pub staging: &'a Staging,
    pub mode: Mode,
    pub trusted: bool,
    pub slots: &'a CommandSlots,
    pub permissions: &'a Permissions,
    pub confirmer: &'a Confirmer,
    pub launcher: &'a dyn Launcher,
    pub hooks: &'a dyn CommandHooks,
    /// Lattice's own environment: X2's `PATH`, and `SystemRoot` for
    /// `powershell.exe`.
    pub env: &'a dyn Env,
    pub globals: &'a Path,
    /// The remote label the output goes to, when the turn's target is remote.
    pub remote: Option<String>,
}

impl CommandContext<'_> {
    fn resolve(&self, name: &str) -> Option<Resolved> {
        resolve_program(name, self.env, Some(&self.workspace.root), self.globals).ok()
    }
}

// ---------------------------------------------------------------- prepare

/// A checked call and its card.
#[derive(Clone, Debug)]
pub struct Prepared {
    /// The command exactly as the model wrote it.
    pub text: String,
    /// Workspace-relative, forward slashes; `""` is the root.
    pub cwd: String,
    /// The working folder's final path, without a verbatim prefix.
    pub cwd_path: PathBuf,
    pub cwd_class: PathClass,
    pub timeout_s: u32,
    /// It keeps running after the call returns (no timeout).
    pub background: bool,
    /// The policy's verdict when the call arrived.
    pub verdict: Verdict,
    /// The card (X12).
    pub detail: ApprovalDetail,
    /// X1–X3 hold: the card may offer "Allow always".
    pub allow_always_offer: bool,
    /// What "Allow always" would store, when it is offered.
    pub candidate: Option<Candidate>,
}

/// The working folder (X10): the root, or a folder that passes WP1–WP11.
fn working_folder(
    ctx: &CommandContext<'_>,
    cwd: Option<&str>,
) -> Result<(String, PathBuf, PathClass), ToolError> {
    let request = cwd.unwrap_or("").trim();
    let request = request.trim_matches(['/', '\\']);
    if request.is_empty() || request == "." {
        return Ok((
            String::new(),
            PathBuf::from(shown_path(&ctx.workspace.root)),
            PathClass::Normal,
        ));
    }
    let resolved = ctx
        .workspace
        .with_rules(ctx.runner, |rules| rules.resolve(request, Want::Existing))
        .map_err(|_| ToolError::new("Lattice could not read this folder's .latticeignore."))??;
    if !resolved.is_dir {
        return Err(ToolError::new("cwd must be a folder in the workspace."));
    }
    Ok((
        resolved.derived,
        PathBuf::from(shown_path(&resolved.final_path)),
        resolved.class,
    ))
}

/// The policy's gates. A background command is not held by the folder's
/// foreground command (nor holds it): its own limit is checked apart.
fn gates(ctx: &CommandContext<'_>, lease: Lease, background: bool) -> Gates {
    Gates {
        staged_waiting: ctx.staging.waiting(),
        command_running: !background && ctx.slots.running(&ctx.workspace.id),
        lease,
    }
}

/// The standing entry `text` matches in `cwd`, with what it resolved to.
fn standing(ctx: &CommandContext<'_>, text: &str, cwd: &str) -> Option<(AllowEntry, Resolved)> {
    let mut found = None;
    let entry = ctx.permissions.matching(ctx.workspace, text, cwd, |name| {
        found = ctx.resolve(name);
        found.clone()
    })?;
    Some((entry, found?))
}

/// Step 1 (see the module header).
pub fn prepare(ctx: &CommandContext<'_>, args: &RunCommandArgs) -> Result<Prepared, ToolError> {
    let text = args.command.clone();
    if text.chars().count() > MAX_COMMAND_CHARS {
        return Err(ToolError::new(
            "The command is longer than 8,000 characters.",
        ));
    }
    if has_bidi(&text) {
        return Err(ToolError::new(BIDI_SENTENCE));
    }
    if text.trim().is_empty() {
        return Err(ToolError::new(NotEligible::Empty.sentence()));
    }
    if args.background && ctx.slots.background(&ctx.workspace.id) >= MAX_BACKGROUND {
        return Err(ToolError::new(BACKGROUND_FULL));
    }
    let timeout_s = match args.timeout_s {
        // A background command has no timeout.
        _ if args.background => 0,
        None => DEFAULT_TIMEOUT_S,
        Some(seconds) if (1..=MAX_TIMEOUT_S).contains(&seconds) => seconds,
        Some(_) => {
            return Err(ToolError::new(
                "timeout_s must be between 1 and 3,600 seconds.",
            ));
        }
    };
    let (cwd, cwd_path, cwd_class) = working_folder(ctx, args.cwd.as_deref())?;
    let matched = standing(ctx, &text, &cwd);
    let verdict = decide(
        ctx.mode,
        ctx.trusted,
        ToolClass::RunCommand,
        &Target::Command { cwd: cwd_class },
        &gates(ctx, Lease::Free, args.background),
        &Standing {
            command_entry: matched.as_ref().map(|(entry, _)| entry.id.clone()),
            mcp_allowlisted: false,
        },
    );
    if let Verdict::Refuse(reason) = &verdict {
        return Err(ToolError::new(reason.sentence()));
    }
    let candidate = match eligible(&text) {
        Ok(argv) if matched.is_none() => ctx.resolve(&argv[0]).and_then(|resolved| {
            may_be_entry(&argv, &resolved).then(|| Candidate {
                argv,
                resolved,
                cwd: cwd.clone(),
            })
        }),
        _ => None,
    };
    let mode = match verdict {
        Verdict::Allow(Because::Standing { .. }) => CommandMode::Direct,
        _ if matched.is_some() => CommandMode::Direct,
        _ => CommandMode::PowerShell,
    };
    let detail = ApprovalDetail::Command {
        text: escape_for_dialog(&text),
        cwd: cwd.clone(),
        mode,
        timeout_s,
        remote: ctx.remote.clone(),
        staged_waiting: ctx.staging.waiting(),
        background: args.background,
    };
    Ok(Prepared {
        text,
        cwd,
        cwd_path,
        cwd_class,
        timeout_s,
        background: args.background,
        verdict,
        detail,
        allow_always_offer: candidate.is_some(),
        candidate,
    })
}

// ---------------------------------------------------------------- approve

/// Why a command was not approved or not run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refused {
    pub sentence: String,
    /// The protocol's `Conflict` (rather than `Invalid`): the gate is shut,
    /// another command runs, the lease is held elsewhere.
    pub conflict: bool,
    /// The reader refused in the native dialog: a `Reject` with no note
    /// (§9.3, CP3).
    pub rejected: bool,
}

impl Refused {
    fn because(reason: Reason) -> Self {
        Self {
            sentence: reason.sentence().to_owned(),
            conflict: reason.is_conflict(),
            rejected: false,
        }
    }

    fn conflict(sentence: impl Into<String>) -> Self {
        Self {
            sentence: sentence.into(),
            conflict: true,
            rejected: false,
        }
    }

    fn invalid(sentence: impl Into<String>) -> Self {
        Self {
            sentence: sentence.into(),
            conflict: false,
            rejected: false,
        }
    }
}

/// The command gate's sentence (§7.5).
pub fn staged_first(waiting: u32) -> String {
    if waiting == 1 {
        "Review 1 staged change first.".to_owned()
    } else {
        format!("Review {waiting} staged changes first.")
    }
}

/// How an approved command runs.
#[derive(Clone, Debug)]
enum How {
    /// A standing entry: its program, directly, with the argv that matched.
    Direct {
        entry: String,
        argv: Vec<String>,
        program: Resolved,
    },
    /// Approved once, in the native dialog: Windows PowerShell 5.1.
    PowerShell,
}

/// A command the reader approved (or a standing entry allowed). Only
/// [`approve`] makes one.
#[derive(Clone, Debug)]
pub struct Approval {
    how: How,
}

impl Approval {
    pub fn mode(&self) -> CommandMode {
        match self.how {
            How::Direct { .. } => CommandMode::Direct,
            How::PowerShell => CommandMode::PowerShell,
        }
    }

    /// The standing entry that allowed it, when one did.
    pub fn entry(&self) -> Option<&str> {
        match &self.how {
            How::Direct { entry, .. } => Some(entry),
            How::PowerShell => None,
        }
    }
}

/// Step 2: the reader's Approve (see the module header). `call` keys CP3:
/// a refusal in the dialog is not asked again for the same call.
pub async fn approve(
    ctx: &CommandContext<'_>,
    prepared: &Prepared,
    call: &CallId,
) -> Result<Approval, Refused> {
    let matched = standing(ctx, &prepared.text, &prepared.cwd);
    let verdict = decide(
        ctx.mode,
        ctx.trusted,
        ToolClass::RunCommand,
        &Target::Command {
            cwd: prepared.cwd_class,
        },
        &gates(ctx, Lease::Free, prepared.background),
        &Standing {
            command_entry: matched.as_ref().map(|(entry, _)| entry.id.clone()),
            mcp_allowlisted: false,
        },
    );
    match verdict {
        Verdict::Refuse(reason) => Err(Refused::because(reason)),
        Verdict::Ask(AskKind::GateStaged { waiting }) => {
            Err(Refused::conflict(staged_first(waiting)))
        }
        Verdict::Allow(Because::Standing { entry }) => {
            let Some((_, program)) = matched else {
                return Err(Refused::invalid(
                    "The allowed command no longer matches; it asks again.",
                ));
            };
            let argv = eligible(&prepared.text).map_err(|why| Refused::invalid(why.sentence()))?;
            Ok(Approval {
                how: How::Direct {
                    entry,
                    argv,
                    program,
                },
            })
        }
        Verdict::Ask(AskKind::Command) => {
            let confirmed = ctx
                .confirmer
                .ask(
                    &format!("run-command:{call}"),
                    ConfirmRequest::RunCommand {
                        text: prepared.text.clone(),
                        cwd: prepared.cwd.clone(),
                        mode: CommandMode::PowerShell,
                        background: prepared.background,
                    },
                    Initiated::Page,
                )
                .await;
            if confirmed {
                Ok(Approval {
                    how: How::PowerShell,
                })
            } else {
                Err(Refused {
                    sentence: NOT_CONFIRMED.to_owned(),
                    conflict: false,
                    rejected: true,
                })
            }
        }
        Verdict::Ask(_) | Verdict::Allow(_) => {
            Err(Refused::invalid("That command cannot be approved here."))
        }
    }
}

// -------------------------------------------------------------------- run

/// Ends a running command's whole tree from another thread (Stop, closing
/// the conversation).
#[derive(Clone, Debug, Default)]
pub struct StopHandle(Arc<StopInner>);

#[derive(Debug, Default)]
struct StopInner {
    stopped: AtomicBool,
    child: Mutex<Option<Arc<Child>>>,
}

impl StopHandle {
    /// Stop: the command does not start, or its whole tree ends now.
    pub fn stop(&self) {
        self.0.stopped.store(true, Ordering::SeqCst);
        if let Some(child) = lock(&self.0.child).as_ref() {
            let _ = child.kill_tree();
        }
    }

    pub fn is_stopped(&self) -> bool {
        self.0.stopped.load(Ordering::SeqCst)
    }

    fn hold(&self, child: &Arc<Child>) {
        *lock(&self.0.child) = Some(Arc::clone(child));
        if self.is_stopped() {
            let _ = child.kill_tree();
        }
    }

    fn release(&self) {
        *lock(&self.0.child) = None;
    }
}

/// X9: a command's output, kept within 8 MiB.
#[derive(Debug, Default)]
pub struct OutputBuffer {
    head: Vec<u8>,
    ring: VecDeque<u8>,
    /// Bytes neither in the head nor in the ring.
    dropped: u64,
    total: u64,
    lines: u64,
}

impl OutputBuffer {
    pub fn push(&mut self, mut bytes: &[u8]) {
        self.total += bytes.len() as u64;
        self.lines += bytes.iter().filter(|byte| **byte == b'\n').count() as u64;
        let room = HEAD_BYTES.saturating_sub(self.head.len());
        if room > 0 {
            let take = room.min(bytes.len());
            self.head.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
        }
        if bytes.len() >= RING_BYTES {
            self.dropped += (self.ring.len() + bytes.len() - RING_BYTES) as u64;
            self.ring.clear();
            self.ring.extend(&bytes[bytes.len() - RING_BYTES..]);
            return;
        }
        let over = (self.ring.len() + bytes.len()).saturating_sub(RING_BYTES);
        if over > 0 {
            self.ring.drain(..over);
            self.dropped += over as u64;
        }
        self.ring.extend(bytes);
    }

    /// Every byte written.
    pub fn total(&self) -> u64 {
        self.total
    }

    pub fn lines(&self) -> u64 {
        self.lines
    }

    /// Bytes dropped between the head and the ring.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// The kept bytes: the head, then the ring.
    pub fn kept(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.head.len() + self.ring.len());
        out.extend_from_slice(&self.head);
        out.extend(self.ring.iter());
        out
    }
}

/// Output bytes as text: UTF-16LE when they read as it (a byte-order mark,
/// or most odd bytes zero), else UTF-8 with replacements.
pub fn output_text(bytes: &[u8]) -> String {
    let utf16 = if bytes.starts_with(&[0xFF, 0xFE]) {
        Some(&bytes[2..])
    } else if bytes.len() >= 4 {
        let pairs = bytes.len() / 2;
        let zero_odd = bytes.chunks_exact(2).filter(|pair| pair[1] == 0).count();
        (zero_odd * 10 >= pairs * 9).then_some(bytes)
    } else {
        None
    };
    match utf16 {
        Some(body) => {
            let units: Vec<u16> = body
                .chunks_exact(2)
                .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                .collect();
            String::from_utf16_lossy(&units)
        }
        None => String::from_utf8_lossy(bytes).into_owned(),
    }
}

fn floor_char(text: &str, at: usize) -> usize {
    let mut at = at.min(text.len());
    while !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}

fn ceil_char(text: &str, at: usize) -> usize {
    let mut at = at.min(text.len());
    while !text.is_char_boundary(at) {
        at += 1;
    }
    at
}

/// X9: what the model reads of an output: the ANSI-stripped text, cut to its
/// first 8 KiB and last 24 KiB with the bytes between counted (`dropped`
/// adds what the buffer itself did not keep), then `status`.
pub fn model_text(kept: &[u8], dropped: u64, status: &str) -> String {
    let text = ansi_strip::strip(&output_text(kept));
    let mut out = String::new();
    if text.len() <= MODEL_HEAD + MODEL_TAIL && dropped == 0 {
        out.push_str(&text);
    } else {
        let head_end = floor_char(&text, MODEL_HEAD);
        let tail_start = ceil_char(&text, text.len().saturating_sub(MODEL_TAIL)).max(head_end);
        let omitted = (tail_start - head_end) as u64 + dropped;
        out.push_str(&text[..head_end]);
        if !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(&format!("[\u{2026} {omitted} bytes omitted \u{2026}]\n"));
        out.push_str(&text[tail_start..]);
    }
    if text.is_empty() {
        out.push_str("(no output)");
    }
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(status);
    out
}

/// Base64 (standard alphabet, padded) of `bytes`.
pub fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// X6: `-EncodedCommand`'s value for `text`.
pub fn encoded_command(text: &str) -> String {
    let bytes: Vec<u8> = text.encode_utf16().flat_map(u16::to_le_bytes).collect();
    base64(&bytes)
}

/// `%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe`, from
/// Lattice's own environment; never searched for.
pub fn powershell_path(env: &dyn Env) -> Option<PathBuf> {
    let root = PathBuf::from(env.var("SystemRoot")?);
    if !root.is_absolute() {
        return None;
    }
    Some(
        root.join("System32")
            .join("WindowsPowerShell")
            .join("v1.0")
            .join("powershell.exe"),
    )
}

/// What a run came to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandOutcome {
    pub mode: CommandMode,
    /// The argv the program was started with.
    pub argv: Vec<String>,
    pub reason: ExitReason,
    pub code: Option<i32>,
    pub duration_ms: u64,
    /// Bytes and lines the command wrote.
    pub bytes: u64,
    pub lines: u64,
    /// The kept output's blob in the conversation's record.
    pub output_blob: Option<String>,
    /// What the model reads (X9).
    pub model_text: String,
    pub after: AfterCommand,
}

pub fn status_line(reason: ExitReason, code: Option<i32>, seconds: f64) -> String {
    match reason {
        ExitReason::Exited => format!("Exit code {} after {seconds:.1} s", code.unwrap_or(-1)),
        ExitReason::Stopped => "Stopped by the user".to_owned(),
        ExitReason::TimedOut => format!("Timed out after {seconds:.1} s"),
        ExitReason::Failed => "The command could not be started.".to_owned(),
    }
}

/// The most bytes of a progress report's tail (§11.2).
pub const TAIL_PREVIEW: usize = 512;

/// A running command's counters after a chunk: bytes and lines written so
/// far, and the last [`TAIL_PREVIEW`] bytes as text (§11.2: one report per
/// chunk at most; the follow coalescer keeps the last per batch).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Progress {
    pub bytes: u64,
    pub lines: u64,
    pub tail: String,
}

/// Where a command's progress goes (the conversation's events).
pub type ProgressSink = Arc<dyn Fn(Progress) + Send + Sync>;

impl OutputBuffer {
    /// The last `n` bytes kept, without copying the buffer.
    pub fn tail(&self, n: usize) -> Vec<u8> {
        let from_ring = n.min(self.ring.len());
        let from_head = (n - from_ring).min(self.head.len());
        let mut out = Vec::with_capacity(from_head + from_ring);
        out.extend_from_slice(&self.head[self.head.len() - from_head..]);
        out.extend(self.ring.iter().skip(self.ring.len() - from_ring));
        out
    }
}

fn reader(
    pipe: Option<std::fs::File>,
    buffer: Arc<Mutex<OutputBuffer>>,
    progress: Option<ProgressSink>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let Some(mut pipe) = pipe else {
            return;
        };
        let mut chunk = vec![0u8; 64 * 1024];
        while let Ok(read) = pipe.read(&mut chunk) {
            if read == 0 {
                break;
            }
            let report = {
                let mut kept = lock(&buffer);
                kept.push(&chunk[..read]);
                progress.as_ref().map(|_| Progress {
                    bytes: kept.total(),
                    lines: kept.lines(),
                    tail: crate::secrets::redact(&ansi_strip::strip(&output_text(
                        &kept.tail(TAIL_PREVIEW),
                    ))),
                })
            };
            if let (Some(progress), Some(report)) = (&progress, report) {
                progress(report);
            }
        }
    })
}

/// Step 3 (see the module header). Blocking: run it off the interface's
/// thread.
pub fn run(
    ctx: &CommandContext<'_>,
    prepared: &Prepared,
    approval: &Approval,
    call: &CallId,
    stop: &StopHandle,
) -> Result<CommandOutcome, Refused> {
    run_with_progress(ctx, prepared, approval, call, stop, None)
}

/// [`run`], reporting the output's counters after each chunk to `progress`
/// (the agent chat's `CommandProgress`, row E11).
pub fn run_with_progress(
    ctx: &CommandContext<'_>,
    prepared: &Prepared,
    approval: &Approval,
    call: &CallId,
    stop: &StopHandle,
    progress: Option<ProgressSink>,
) -> Result<CommandOutcome, Refused> {
    let buffer = Arc::new(Mutex::new(OutputBuffer::default()));
    run_into(ctx, prepared, approval, call, stop, progress, buffer)
}

/// [`run_with_progress`], writing the output into `buffer`, which the caller
/// may read while the command runs (a background command's `command_output`).
pub fn run_into(
    ctx: &CommandContext<'_>,
    prepared: &Prepared,
    approval: &Approval,
    call: &CallId,
    stop: &StopHandle,
    progress: Option<ProgressSink>,
    buffer: Arc<Mutex<OutputBuffer>>,
) -> Result<CommandOutcome, Refused> {
    let waiting = ctx.staging.waiting();
    if waiting > 0 {
        return Err(Refused::conflict(staged_first(waiting)));
    }
    let claimed = if prepared.background {
        ctx.slots.claim_background(&ctx.workspace.id, call)
    } else {
        ctx.slots.claim(&ctx.workspace.id, call)
    };
    let Some(slot) = claimed else {
        if prepared.background {
            return Err(Refused::conflict(BACKGROUND_FULL));
        }
        return Err(Refused::because(Reason::CommandRunning));
    };
    if ctx.hooks.lease() == Lease::Elsewhere {
        return Err(Refused::because(Reason::LeaseElsewhere));
    }
    let (program, argv): (PathBuf, Vec<String>) = match &approval.how {
        How::Direct { argv, program, .. } => (program.path.clone(), argv.clone()),
        How::PowerShell => {
            let Some(program) = powershell_path(ctx.env) else {
                return Err(Refused::invalid(
                    "Windows PowerShell could not be found, so the command did not run.",
                ));
            };
            (
                program,
                vec![
                    "powershell.exe".to_owned(),
                    "-NoProfile".to_owned(),
                    "-NonInteractive".to_owned(),
                    "-EncodedCommand".to_owned(),
                    encoded_command(&prepared.text),
                ],
            )
        }
    };
    if stop.is_stopped() {
        drop(slot);
        return Ok(finished(
            ctx,
            approval,
            argv,
            ExitReason::Stopped,
            None,
            Duration::ZERO,
            &OutputBuffer::default(),
            AfterCommand::default(),
        ));
    }
    let before = ctx
        .hooks
        .before(call)
        .map_err(|_| Refused::invalid(NOT_RECORDED))?;
    let spec = ChildSpec {
        program,
        argv: argv.iter().map(OsString::from).collect(),
        cwd: prepared.cwd_path.clone(),
        limits: JobLimits::default(),
    };
    let started = Instant::now();
    let mut child = match ctx.launcher.launch(&spec, &ctx.workspace.root) {
        Ok(child) => child,
        Err(_) => {
            let after = ctx.hooks.after(call, before);
            drop(slot);
            return Ok(finished(
                ctx,
                approval,
                argv,
                ExitReason::Failed,
                None,
                started.elapsed(),
                &OutputBuffer::default(),
                after,
            ));
        }
    };
    let readers = [
        reader(child.take_stdout(), Arc::clone(&buffer), progress.clone()),
        reader(child.take_stderr(), Arc::clone(&buffer), progress),
    ];
    let child = Arc::new(child);
    stop.hold(&child);
    let waited = if prepared.background {
        child.wait(None)
    } else {
        child.wait(Some(Duration::from_secs(u64::from(prepared.timeout_s))))
    };
    let elapsed = started.elapsed();
    let (reason, code) = match waited {
        _ if stop.is_stopped() => (ExitReason::Stopped, None),
        Ok(Some(code)) => (ExitReason::Exited, Some(code as i32)),
        Ok(None) => (ExitReason::TimedOut, None),
        Err(_) => (ExitReason::Failed, None),
    };
    // Exit, stop and timeout all end the whole tree (X8).
    let _ = child.kill_tree();
    stop.release();
    for handle in readers {
        let _ = handle.join();
    }
    let after = ctx.hooks.after(call, before);
    drop(slot);
    let buffer = lock(&buffer);
    Ok(finished(
        ctx, approval, argv, reason, code, elapsed, &buffer, after,
    ))
}

#[allow(clippy::too_many_arguments)]
fn finished(
    ctx: &CommandContext<'_>,
    approval: &Approval,
    argv: Vec<String>,
    reason: ExitReason,
    code: Option<i32>,
    elapsed: Duration,
    buffer: &OutputBuffer,
    after: AfterCommand,
) -> CommandOutcome {
    let kept = buffer.kept();
    let output_blob = if kept.is_empty() {
        None
    } else {
        ctx.staging
            .sidecar()
            .put_blob(&kept, BlobKind::Output)
            .ok()
            .map(|blob| blob.sha256)
    };
    let mut model = model_text(
        &kept,
        buffer.dropped(),
        &status_line(reason, code, elapsed.as_secs_f64()),
    );
    for note in &after.notes {
        model.push('\n');
        model.push_str(note);
    }
    CommandOutcome {
        mode: approval.mode(),
        argv,
        reason,
        code,
        duration_ms: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
        bytes: buffer.total(),
        lines: buffer.lines(),
        output_blob,
        model_text: model,
        after,
    }
}

#[cfg(test)]
mod tests;
