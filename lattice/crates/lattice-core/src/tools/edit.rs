//! The staging tools: `edit_file`, `write_file` and `delete_file`
//! (the chat core's spec §7.1, §7.4.1). Not a port.
//!
//! **They write nothing to the folder** (OD2, ST1). Each one resolves its
//! path by the path rules (WP1–WP11, on the derived long-name path; a file to
//! be created has its parent resolved and its leaf appended), reads the
//! conversation's staged view of the file, computes the bytes a Keep would
//! leave, and records them in the conversation's staging (`staging`), whose
//! record is the sidecar. The reader's Keep is the write (row E3).
//!
//! - `edit_file`: an exact match of `old_string` in the **staged view**, once
//!   unless `replace_all` (`staging::apply_op`: CRLF files match on their LF
//!   form and stay CRLF; the byte-order mark stays). UTF-8 text only.
//! - `write_file`: creates a file, or replaces one this conversation has read
//!   (`read_file`) or staged; otherwise "Read the file before replacing it."
//!   At most 2 MiB of UTF-8. A replaced file keeps its byte-order mark and,
//!   when they were uniform, its line ends. The parent folder must exist:
//!   no folder is made.
//! - `delete_file`: the file must exist in the staged view; folders are
//!   refused.
//!
//! The policy engine is asked first (Ask mode and an untrusted folder refuse;
//! §9.2), and again with the path's class. The answer tells the model the
//! truth (ST5): "Staged `<path>` (+a −b). It is not on disk until the user
//! keeps it."

use lattice_protocol::TurnId;
use lattice_protocol::conversation::{CallId, Mode};
use serde::Deserialize;

use super::read::{MAX_FILE_BYTES, Overlay, Staged, ToolError, Unread, read_bounded, rules_error};
use crate::convo::item::{BaseState, EditOp};
use crate::git::runner::GitRunner;
use crate::policy::{Gates, Lease, PathClass, Standing, Target, ToolClass, Verdict, decide};
use crate::staging::{Action, Base, Proposal, Recorded, Staging, apply_op, encode};
use crate::workspace::Workspace;
use crate::workspace::paths::{PathError, PathRules, Resolved, Want};

/// What every staging tool works with.
pub struct StageContext<'a> {
    pub workspace: &'a Workspace,
    pub runner: &'a GitRunner,
    pub staging: &'a Staging,
    pub mode: Mode,
    /// The folder is trusted (FT3: Agent mode needs it).
    pub trusted: bool,
    pub turn: &'a TurnId,
    pub call: &'a CallId,
}

/// `edit_file`'s arguments.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct EditFileArgs {
    pub path: String,
    pub old_string: String,
    pub new_string: String,
    #[serde(default)]
    pub replace_all: bool,
}

/// `write_file`'s arguments.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct WriteFileArgs {
    pub path: String,
    pub content: String,
}

/// `delete_file`'s arguments.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct DeleteFileArgs {
    pub path: String,
}

/// The policy's word on a staging call (§9.2): Ask mode, an untrusted folder
/// and a refused path refuse; anything else stages.
fn allowed(ctx: &StageContext<'_>, target: Target) -> Result<(), ToolError> {
    let gates = Gates {
        staged_waiting: 0,
        command_running: false,
        lease: Lease::Free,
    };
    match decide(
        ctx.mode,
        ctx.trusted,
        ToolClass::Stage,
        &target,
        &gates,
        &Standing::default(),
    ) {
        Verdict::Refuse(reason) => Err(ToolError::new(reason.sentence())),
        Verdict::Allow(_) | Verdict::Ask(_) => Ok(()),
    }
}

/// The path rules' answer for a staging tool: the path may be created; a
/// folder is refused. With it, whether the file is an authority file (ST4),
/// by the class the rules gave the derived path, never the request.
fn target(
    ctx: &StageContext<'_>,
    rules: &PathRules<'_>,
    request: &str,
    folder: &'static str,
) -> Result<(Resolved, bool), ToolError> {
    let resolved = rules.resolve(request, Want::MayCreate)?;
    if resolved.is_dir {
        return Err(ToolError::new(folder));
    }
    allowed(ctx, Target::Path(resolved.class))?;
    let authority = resolved.class == PathClass::Authority;
    Ok((resolved, authority))
}

/// What the staged view holds at a resolved path.
enum View {
    /// Its bytes; `staged` when they are this conversation's staged bytes.
    Bytes { bytes: Vec<u8>, staged: bool },
    /// No file (none on disk, or a staged deletion).
    Absent,
}

fn too_large(size: u64) -> ToolError {
    ToolError::new(format!(
        "That file is {size} bytes; Lattice stages files of at most 2 MiB (2,097,152 bytes)."
    ))
}

/// The staged view of `resolved`: staged bytes first, then the disk through
/// the handle the path rules opened.
fn view(ctx: &StageContext<'_>, resolved: &mut Resolved) -> Result<View, ToolError> {
    match ctx.staging.staged(&resolved.derived) {
        Some(Staged::Bytes(bytes)) => {
            return Ok(View::Bytes {
                bytes,
                staged: true,
            });
        }
        Some(Staged::Deleted) => return Ok(View::Absent),
        None => {}
    }
    let Some(file) = resolved.file.take() else {
        return Ok(View::Absent);
    };
    match read_bounded(file) {
        Ok(bytes) => Ok(View::Bytes {
            bytes,
            staged: false,
        }),
        Err(Unread::TooLarge(size)) => Err(too_large(size)),
        Err(Unread::Unreadable) => Err(PathError::Unreadable.into()),
    }
}

/// The model's answer (ST5).
fn told(recorded: &Recorded) -> String {
    format!(
        "Staged `{}` (+{} \u{2212}{}). It is not on disk until the user keeps it.",
        recorded.change.path, recorded.added, recorded.removed
    )
}

fn record(ctx: &StageContext<'_>, proposal: Proposal) -> Result<String, ToolError> {
    ctx.staging
        .record(proposal, ctx.turn, ctx.call)
        .map(|recorded| told(&recorded))
        .map_err(|error| ToolError::new(error.sentence()))
}

/// The base a first staging records: the disk bytes just read, or an absent
/// file. A staged view means a live change, whose own base stands.
fn base_of(view: &View) -> Base {
    match view {
        View::Bytes {
            bytes,
            staged: false,
        } => Base::of_bytes(bytes.clone()),
        View::Bytes { staged: true, .. } | View::Absent => Base::absent(),
    }
}

/// `edit_file` (§7.4.1).
pub fn edit_file(ctx: &StageContext<'_>, args: &EditFileArgs) -> Result<String, ToolError> {
    allowed(ctx, Target::None)?;
    if args.old_string.is_empty() {
        return Err(ToolError::new(
            "old_string is empty; use write_file to create or replace a file.",
        ));
    }
    if args.old_string == args.new_string {
        return Err(ToolError::new(
            "old_string and new_string are the same, so nothing would change.",
        ));
    }
    let _one = ctx.staging.serial();
    ctx.workspace
        .with_rules(ctx.runner, |rules| {
            let (mut resolved, authority) = target(
                ctx,
                rules,
                &args.path,
                "That is a folder; edit_file changes files.",
            )?;
            let view = view(ctx, &mut resolved)?;
            let View::Bytes { bytes, .. } = &view else {
                return Err(PathError::Missing.into());
            };
            let op = EditOp {
                old_string: args.old_string.clone(),
                new_string: args.new_string.clone(),
                replace_all: args.replace_all,
            };
            let new = apply_op(bytes, &op).map_err(|error| ToolError::new(error.sentence()))?;
            if new.len() as u64 > MAX_FILE_BYTES {
                return Err(ToolError::new(
                    "The edited file would be over 2 MiB (2,097,152 bytes), so nothing was staged.",
                ));
            }
            record(
                ctx,
                Proposal {
                    path: resolved.derived.clone(),
                    authority,
                    base: base_of(&view),
                    new: Some(new),
                    action: Action::Edit(op),
                },
            )
        })
        .map_err(rules_error)?
}

/// `write_file` (§7.4.1).
pub fn write_file(ctx: &StageContext<'_>, args: &WriteFileArgs) -> Result<String, ToolError> {
    allowed(ctx, Target::None)?;
    if args.content.len() as u64 > MAX_FILE_BYTES {
        return Err(ToolError::new(
            "write_file writes at most 2 MiB (2,097,152 bytes).",
        ));
    }
    let _one = ctx.staging.serial();
    ctx.workspace
        .with_rules(ctx.runner, |rules| {
            let (mut resolved, authority) = target(
                ctx,
                rules,
                &args.path,
                "That is a folder; write_file writes files.",
            )
            .map_err(|error| {
                if error == ToolError::from(PathError::Missing) {
                    ToolError::new("That folder does not exist; write_file does not make folders.")
                } else {
                    error
                }
            })?;
            let live = ctx.staging.live(&resolved.derived);
            if live.is_none() && resolved.exists && !ctx.staging.has_read(&resolved.derived) {
                return Err(ToolError::new("Read the file before replacing it."));
            }
            let view = view(ctx, &mut resolved)?;
            let base = base_of(&view);
            // The base whose encoding a replaced file keeps: the live change's,
            // or the disk's just read.
            let encoding = match live.as_ref().map(|change| &change.base) {
                Some(BaseState::Present { eol, bom, .. }) => Some((*eol, *bom)),
                Some(BaseState::Absent) => None,
                None => match &base.state {
                    BaseState::Present { eol, bom, .. } => Some((*eol, *bom)),
                    BaseState::Absent => None,
                },
            };
            let new = match encoding {
                Some((eol, bom)) => encode(&args.content, eol, bom),
                None => args.content.as_bytes().to_vec(),
            };
            if new.len() as u64 > MAX_FILE_BYTES {
                return Err(ToolError::new(
                    "write_file writes at most 2 MiB (2,097,152 bytes).",
                ));
            }
            record(
                ctx,
                Proposal {
                    path: resolved.derived.clone(),
                    authority,
                    base,
                    new: Some(new),
                    action: Action::Write,
                },
            )
        })
        .map_err(rules_error)?
}

/// `delete_file` (§7.4.1): the file must exist in the staged view.
pub fn delete_file(ctx: &StageContext<'_>, args: &DeleteFileArgs) -> Result<String, ToolError> {
    allowed(ctx, Target::None)?;
    let _one = ctx.staging.serial();
    ctx.workspace
        .with_rules(ctx.runner, |rules| {
            let (mut resolved, authority) = target(
                ctx,
                rules,
                &args.path,
                "delete_file removes files only; that is a folder.",
            )?;
            let base = match ctx.staging.staged(&resolved.derived) {
                Some(Staged::Bytes(_)) => Base::absent(),
                Some(Staged::Deleted) => return Err(PathError::Missing.into()),
                None => {
                    let Some(file) = resolved.file.take() else {
                        return Err(PathError::Missing.into());
                    };
                    let size = file
                        .metadata()
                        .map_err(|_| ToolError::from(PathError::Unreadable))?
                        .len();
                    if size <= MAX_FILE_BYTES {
                        match read_bounded(file) {
                            Ok(bytes) => Base::of_bytes(bytes),
                            Err(Unread::TooLarge(size)) => return Err(too_large(size)),
                            Err(Unread::Unreadable) => return Err(PathError::Unreadable.into()),
                        }
                    } else {
                        Base::of_reader(file).map_err(|_| ToolError::from(PathError::Unreadable))?
                    }
                }
            };
            record(
                ctx,
                Proposal {
                    path: resolved.derived.clone(),
                    authority,
                    base,
                    new: None,
                    action: Action::Delete,
                },
            )
        })
        .map_err(rules_error)?
}
