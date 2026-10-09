//! The policy engine: one pure function that says whether a tool call, a
//! Keep or a command may go ahead, must ask the reader, or is refused
//! (the chat core's spec §9.1–§9.2). Not a port.
//!
//! [`decide`] reads no file, no clock and no environment: everything it needs
//! arrives as arguments (the mode, whether the folder is trusted, the tool, the
//! target's path class, the gates, and whether a standing approval matched).
//! The callers compute those facts (path rules, X1–X4 matching, the lease);
//! the engine only combines them, so the whole table is testable in memory.
//!
//! Precedence, where the table's rows meet: a refusal comes first (Ask mode,
//! an untrusted folder, a refused path, the lease held elsewhere, a command
//! already running), then the staged-changes gate (which turns even a standing
//! match into a question), then a standing match, then the question itself.
//!
//! Every verdict names its rule ([`Verdict::rule`]) so the record can say why
//! something ran or did not (`ApprovalDecided{by: Policy{rule}}`).

use lattice_protocol::conversation::Mode;

/// A path refused by the path rules (§6.2, §9.1). The class is computed on the
/// derived long-name path, never on the request string.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PathRefusal {
    Outside,
    GitDir,
    Ignored,
    SecretDefault,
    LatticeState,
    Device,
    Stream,
    Unc,
    ShortName,
    RemoteLink,
    TrailingDotOrSpace,
    TooLong,
    Escape,
}

impl PathRefusal {
    /// The sentence a refused tool call answers with.
    pub fn sentence(self) -> &'static str {
        match self {
            Self::Outside => "That path is outside the folder.",
            Self::GitDir => "Lattice does not read or change the .git folder.",
            Self::Ignored => "git ignores that file, so it is not shown.",
            Self::SecretDefault => "That file may hold secrets, so it is not used.",
            Self::LatticeState => "That is Lattice's own state, so it is not used.",
            Self::Device => "That name is a device, not a file.",
            Self::Stream => "Alternate data streams are not used.",
            Self::Unc => "Network paths are not used.",
            Self::ShortName => "Use the file's full name.",
            Self::RemoteLink => "That link points outside the folder or to the network.",
            Self::TrailingDotOrSpace => "A name may not end in a dot or a space.",
            Self::TooLong => "That path is too long.",
            Self::Escape => "That path leaves the folder.",
        }
    }
}

/// A path's class (§9.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PathClass {
    Normal,
    /// A file that moves a boundary or a program (ST4): kept one at a time,
    /// through a native dialog, and never in Keep All.
    Authority,
    Refused(PathRefusal),
}

/// What is asking.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ToolClass {
    /// `list_dir`, `glob`, `read_file`, `grep`.
    Read,
    AskQuestion,
    /// `edit_file`, `write_file`, `delete_file`: they stage, they do not write.
    Stage,
    RunCommand,
    /// A tool of an MCP server (§12).
    Mcp,
    /// An action in the agent's browser; `sensitive` when its effect is one
    /// another person will see, moves money, deletes or changes an account.
    Browser {
        sensitive: bool,
    },
    /// An action on the whole desktop in auto mode; `asks` when its effect
    /// moves money or changes an account ("Money and accounts
    /// ask").
    Desktop {
        asks: bool,
    },
    /// The reader's Keep of one change (`what` says which kind).
    Keep(KeepKind),
    /// One change of a Keep All.
    KeepAll,
}

/// What a Keep keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeepKind {
    Change,
    Restore,
    CommandUndo,
}

/// What a call is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    None,
    /// A file tool's path, or a change's path, by class.
    Path(PathClass),
    /// A command, by its working folder's class.
    Command {
        cwd: PathClass,
    },
}

/// The writer lease (§6.5) as the caller found it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Lease {
    /// No one holds it; the action may take it.
    Free,
    /// This conversation holds it.
    Held,
    /// Another conversation, the PyQt IDE or a web agent run holds it.
    Elsewhere,
}

/// The facts that shut or open the gates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Gates {
    /// Changes Pending, Rebased or in Conflict in this conversation.
    pub staged_waiting: u32,
    /// A command is running in this workspace (X14).
    pub command_running: bool,
    pub lease: Lease,
}

/// What the standing approvals say (X1–X4 are matched by the caller).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Standing {
    /// The id of the entry this exact command matched, if one did.
    pub command_entry: Option<String>,
    /// The MCP tool is on its server's allowlist.
    pub mcp_allowlisted: bool,
}

/// Why a call may go ahead.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Because {
    Read,
    Question,
    Stage,
    /// Staged as an authority file.
    StageAuthority,
    /// A standing entry matched (`ApprovalDecided{by: Standing{entry}}`).
    Standing {
        entry: String,
    },
    Mcp,
    /// A browser action whose effect does not ask.
    Browser,
    /// A desktop action in auto mode whose effect does not ask.
    AutoMode,
    Keep,
}

/// What the reader is asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AskKind {
    /// Approve a command, through the native `RunCommand` dialog.
    Command,
    /// Staged changes wait for review: Run is disabled until they are.
    GateStaged {
        waiting: u32,
    },
    /// Keep an authority file, through the native `KeepAuthority` dialog.
    KeepAuthority,
    Mcp,
    /// A browser action whose effect asks, through the native `BrowserAct`
    /// dialog.
    Browser,
    /// A desktop action that moves money or changes an account, through the
    /// native `DesktopAct` dialog.
    Desktop,
}

/// Why a call is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reason {
    AskMode,
    Untrusted,
    Path(PathRefusal),
    /// X14: another command runs in this workspace.
    CommandRunning,
    /// X14: a Keep waits for the running command.
    KeepWhileCommand,
    LeaseElsewhere,
    /// Keep All leaves authority files to an individual Keep.
    AuthorityKeptAlone,
}

impl Reason {
    /// One sentence for the model or the reader.
    pub fn sentence(self) -> &'static str {
        match self {
            Self::AskMode => "That is not available in Ask mode.",
            Self::Untrusted => "Trust this folder to use Agent mode.",
            Self::Path(refusal) => refusal.sentence(),
            Self::CommandRunning => "Another command is still running in this folder.",
            Self::KeepWhileCommand => {
                "A command is still running in this folder; Keep when it has finished."
            }
            Self::LeaseElsewhere => "Another Lattice agent is editing this folder.",
            Self::AuthorityKeptAlone => {
                "This file changes how tools or Lattice behave; keep it on its own."
            }
        }
    }

    /// Whether the protocol's refusal kind is `Conflict` (rather than `Invalid`).
    pub fn is_conflict(self) -> bool {
        matches!(
            self,
            Self::CommandRunning | Self::KeepWhileCommand | Self::LeaseElsewhere
        )
    }
}

/// The engine's answer.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Verdict {
    Allow(Because),
    Ask(AskKind),
    Refuse(Reason),
}

impl Verdict {
    /// The rule's name, as the record keeps it.
    pub fn rule(&self) -> &'static str {
        match self {
            Verdict::Allow(Because::Read) => "read tools",
            Verdict::Allow(Because::Question) => "ask question",
            Verdict::Allow(Because::Stage) => "stage",
            Verdict::Allow(Because::StageAuthority) => "stage authority file",
            Verdict::Allow(Because::Standing { .. }) => "standing entry",
            Verdict::Allow(Because::Mcp) => "mcp allowlist",
            Verdict::Allow(Because::Browser) => "browser view or edit",
            Verdict::Allow(Because::AutoMode) => "auto mode",
            Verdict::Allow(Because::Keep) => "keep",
            Verdict::Ask(AskKind::Command) => "ask command",
            Verdict::Ask(AskKind::GateStaged { .. }) => "staged changes first",
            Verdict::Ask(AskKind::KeepAuthority) => "keep authority file",
            Verdict::Ask(AskKind::Mcp) => "ask mcp",
            Verdict::Ask(AskKind::Browser) => "ask browser",
            Verdict::Ask(AskKind::Desktop) => "ask desktop",
            Verdict::Refuse(Reason::AskMode) => "ask mode",
            Verdict::Refuse(Reason::Untrusted) => "untrusted folder",
            Verdict::Refuse(Reason::Path(_)) => "path refused",
            Verdict::Refuse(Reason::CommandRunning) => "one command per folder",
            Verdict::Refuse(Reason::KeepWhileCommand) => "keep waits for command",
            Verdict::Refuse(Reason::LeaseElsewhere) => "lease held elsewhere",
            Verdict::Refuse(Reason::AuthorityKeptAlone) => "authority kept alone",
        }
    }
}

fn path_of(target: &Target) -> Option<PathClass> {
    match *target {
        Target::Path(class) => Some(class),
        Target::Command { cwd } => Some(cwd),
        Target::None => None,
    }
}

/// The refusals every acting tool meets first: Ask mode, an untrusted
/// folder, a refused path.
fn may_act(mode: Mode, trusted: bool, target: &Target) -> Option<Reason> {
    if mode == Mode::Ask {
        return Some(Reason::AskMode);
    }
    if !trusted {
        return Some(Reason::Untrusted);
    }
    match path_of(target) {
        Some(PathClass::Refused(refusal)) => Some(Reason::Path(refusal)),
        _ => None,
    }
}

/// The verdict for one call (§9.2).
pub fn decide(
    mode: Mode,
    trusted: bool,
    tool: ToolClass,
    target: &Target,
    gates: &Gates,
    standing: &Standing,
) -> Verdict {
    use Verdict::{Allow, Ask, Refuse};
    let authority = path_of(target) == Some(PathClass::Authority);
    if let ToolClass::Read = tool {
        return match path_of(target) {
            Some(PathClass::Refused(refusal)) => Refuse(Reason::Path(refusal)),
            _ => Allow(Because::Read),
        };
    }
    if let ToolClass::AskQuestion = tool {
        return Allow(Because::Question);
    }
    if let Some(reason) = may_act(mode, trusted, target) {
        return Refuse(reason);
    }
    match tool {
        ToolClass::Read => Allow(Because::Read),
        ToolClass::AskQuestion => Allow(Because::Question),
        ToolClass::Stage if authority => Allow(Because::StageAuthority),
        ToolClass::Stage => Allow(Because::Stage),
        ToolClass::RunCommand => {
            if gates.lease == Lease::Elsewhere {
                Refuse(Reason::LeaseElsewhere)
            } else if gates.command_running {
                Refuse(Reason::CommandRunning)
            } else if gates.staged_waiting > 0 {
                Ask(AskKind::GateStaged {
                    waiting: gates.staged_waiting,
                })
            } else if let Some(entry) = &standing.command_entry {
                Allow(Because::Standing {
                    entry: entry.clone(),
                })
            } else {
                Ask(AskKind::Command)
            }
        }
        ToolClass::Mcp if standing.mcp_allowlisted => Allow(Because::Mcp),
        ToolClass::Mcp => Ask(AskKind::Mcp),
        ToolClass::Browser { sensitive: true } => Ask(AskKind::Browser),
        ToolClass::Browser { sensitive: false } => Allow(Because::Browser),
        ToolClass::Desktop { asks: true } => Ask(AskKind::Desktop),
        ToolClass::Desktop { asks: false } => Allow(Because::AutoMode),
        ToolClass::Keep(_) | ToolClass::KeepAll => {
            if gates.lease == Lease::Elsewhere {
                Refuse(Reason::LeaseElsewhere)
            } else if gates.command_running {
                Refuse(Reason::KeepWhileCommand)
            } else if authority && tool == ToolClass::KeepAll {
                Refuse(Reason::AuthorityKeptAlone)
            } else if authority {
                Ask(AskKind::KeepAuthority)
            } else {
                Allow(Because::Keep)
            }
        }
    }
}

#[cfg(test)]
mod tests;
