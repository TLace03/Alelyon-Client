//! Artifacts: documents the agent makes beside the chat, the
//! "artifacts" (2026-10-08), as Claude and Antigravity have them: a plan, a
//! report, a walkthrough, a page. The reader reads one apart from the answer,
//! keeps coming back to it, and sees each version the agent saves.
//!
//! - `write_artifact` (`name`, `title`, `kind`, `content`) saves a version: a
//!   name the conversation has not used starts an artifact, and the same name
//!   again is its next version. `read_artifact` (`name`, `version`) reads one
//!   back, so the agent can revise it in a later turn. Both are offered in both
//!   modes and in a turn with no folder, with no approval: they write only the
//!   conversation's own record, never a file of the folder.
//! - **Kinds:** `markdown` (shown formatted), `html` and `svg` (shown as
//!   source, and previewed in a browser of their own that has no network and
//!   none of the reader's sign-ins: `crate::browser::preview`, through
//!   `AgentChat::preview_artifact`), and `text` (shown as it is).
//! - **The record:** each version is an [`Item::ArtifactSaved`], its text kept
//!   as any payload is (inline up to 16 KiB, else a blob), and the event of the
//!   same name says which version was saved; the window reads the text with
//!   `AgentChat::artifact`. The records become the same events when a
//!   conversation is reopened.
//! - **Bounds:** a name is 1 to 48 lowercase letters, digits and single
//!   hyphens; a title one line of at most 80 characters; a version's text at
//!   most 256 KiB of UTF-8; at most 32 artifacts in a conversation and 20
//!   versions of each. The text is redacted before it is written (T15), so
//!   what the reader sees, what the agent reads back and the record agree.
//!
//! Not a port: the web Lattice has no artifacts.

use std::collections::BTreeMap;

use lattice_protocol::conversation::{ArtifactKind, ArtifactView, ConversationEventKind};
use lattice_protocol::{Refusal, RefusalKind};
use serde::Deserialize;
use serde_json::Value;

use super::agent::{Inner, lock, refuse};
use super::item::{Item, Payload};
use super::turn::TurnTools;
use crate::secrets;
use crate::tools::read::{ToolError, parse_args};

/// The longest name, in characters.
pub const MAX_NAME_CHARS: usize = 48;
/// The longest title, in characters.
pub const MAX_TITLE_CHARS: usize = 80;
/// The most bytes of one version's text.
pub const MAX_BYTES: usize = 256 * 1024;
/// The most artifacts in one conversation.
pub const MAX_ARTIFACTS: usize = 32;
/// The most versions of one artifact.
pub const MAX_VERSIONS: usize = 20;
/// The most characters `read_artifact` returns.
pub const MAX_READ_CHARS: usize = 64 * 1024;

/// The reader's sentences.
pub mod words {
    pub const NONE: &str = "That artifact is not in this conversation.";
    pub const NO_VERSION: &str = "That version of the artifact is not in this conversation.";
    pub const UNREADABLE: &str =
        "Lattice could not read this artifact from the conversation's record.";
}

/// `write_artifact`'s arguments.
#[derive(Clone, Debug, Deserialize)]
pub struct WriteArgs {
    pub name: String,
    pub title: String,
    pub kind: ArtifactKind,
    pub content: String,
}

/// `read_artifact`'s arguments.
#[derive(Clone, Debug, Deserialize)]
pub struct ReadArgs {
    pub name: String,
    #[serde(default)]
    pub version: Option<u32>,
}

/// One saved version, as the records state it.
#[derive(Clone, Debug, PartialEq)]
pub struct Version {
    pub title: String,
    pub kind: ArtifactKind,
    pub text: Payload,
    pub bytes: u64,
}

/// True when `name` can name an artifact: 1 to 48 lowercase ASCII letters,
/// digits and hyphens, with no hyphen first, last or twice in a row.
pub fn is_name(name: &str) -> bool {
    (1..=MAX_NAME_CHARS).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        && !name.starts_with('-')
        && !name.ends_with('-')
        && !name.contains("--")
}

/// Every artifact the records name, by name, each with its versions in order.
pub fn artifacts(items: &[Item]) -> BTreeMap<String, Vec<Version>> {
    let mut out: BTreeMap<String, Vec<Version>> = BTreeMap::new();
    for item in items {
        if let Item::ArtifactSaved {
            name,
            title,
            kind,
            text,
            bytes,
            ..
        } = item
        {
            out.entry(name.clone()).or_default().push(Version {
                title: title.clone(),
                kind: *kind,
                text: text.clone(),
                bytes: *bytes,
            });
        }
    }
    out
}

/// `write_artifact` and `read_artifact`, on the blocking pool.
pub(crate) fn tool(
    tools: &TurnTools,
    name: &str,
    call: &str,
    args: &Value,
) -> Result<String, ToolError> {
    if name == "read_artifact" {
        read_tool(tools, &parse_args(args)?)
    } else {
        write_tool(tools, call, &parse_args(args)?)
    }
}

fn write_tool(tools: &TurnTools, call: &str, args: &WriteArgs) -> Result<String, ToolError> {
    let name = args.name.trim();
    if !is_name(name) {
        return Err(ToolError::new(
            "An artifact's name is 1 to 48 lowercase letters, digits and single hyphens, such as plan or api-report.",
        ));
    }
    let title = args.title.trim();
    if title.is_empty()
        || title.chars().count() > MAX_TITLE_CHARS
        || title.chars().any(char::is_control)
    {
        return Err(ToolError::new(
            "An artifact's title is one line of 1 to 80 characters.",
        ));
    }
    if args.content.trim().is_empty() || args.content.len() > MAX_BYTES {
        return Err(ToolError::new(
            "An artifact's content is 1 byte to 256 KiB of text.",
        ));
    }
    let (title, content) = (secrets::redact(title), secrets::redact(&args.content));
    let redacted = title != args.title.trim() || content != args.content;
    let inner = &tools.inner;
    let convo = &tools.convo;
    let _saving = lock(&inner.artifacts);
    let sidecar = convo.state().sidecar.clone().ok_or_else(|| {
        ToolError::new("This conversation has no record yet, so nothing was saved.")
    })?;
    let items = inner
        .sidecars
        .read_items(&convo.id)
        .map(|log| log.items)
        .map_err(|_| {
            ToolError::new(
                "Lattice could not read this conversation's record, so nothing was saved.",
            )
        })?;
    let known = artifacts(&items);
    let version = match known.get(name) {
        Some(versions) if versions.len() >= MAX_VERSIONS => {
            return Err(ToolError::new(format!(
                "{name} has {MAX_VERSIONS} versions already, the most one artifact keeps; save the next under a new name."
            )));
        }
        Some(versions) => versions.len() + 1,
        None if known.len() >= MAX_ARTIFACTS => {
            return Err(ToolError::new(format!(
                "This conversation has {MAX_ARTIFACTS} artifacts already, the most it keeps; save a new version of one instead."
            )));
        }
        None => 1,
    };
    let (text, _) = sidecar.payload(&content).map_err(|_| {
        ToolError::new("Lattice could not save the artifact's text, so nothing was saved.")
    })?;
    let version = version as u32;
    let bytes = content.len() as u64;
    let item = Item::ArtifactSaved {
        turn: tools.turn.clone(),
        call_id: call.to_owned(),
        name: name.to_owned(),
        title: title.clone(),
        kind: args.kind,
        version,
        text,
        bytes,
        at: inner.now(),
    };
    if sidecar.append(&item).is_err() {
        return Err(ToolError::new(
            "Lattice could not save the artifact, so nothing was saved.",
        ));
    }
    convo.log.push(ConversationEventKind::ArtifactSaved {
        name: name.to_owned(),
        title,
        kind: args.kind,
        version,
        bytes,
    });
    let mut said = format!(
        "Saved {name}, version {version}. The user sees it beside the chat; saving it again under the same name makes its next version."
    );
    if redacted {
        said.push_str(&format!(
            " Text that looked like a secret was replaced with {}.",
            secrets::REDACTED
        ));
    }
    Ok(said)
}

fn read_tool(tools: &TurnTools, args: &ReadArgs) -> Result<String, ToolError> {
    let view = read(
        &tools.inner,
        &tools.convo.id,
        args.name.trim(),
        args.version,
    )
    .map_err(|refusal| ToolError::new(refusal.message))?;
    let mut text = view.text;
    let total = text.chars().count();
    if total > MAX_READ_CHARS {
        let cut = text
            .char_indices()
            .nth(MAX_READ_CHARS)
            .map_or(text.len(), |(at, _)| at);
        text.truncate(cut);
        text.push_str(&format!(
            "\n[... {} more characters not shown]",
            total - MAX_READ_CHARS
        ));
    }
    Ok(format!(
        "{} ({}), version {} of {}:\n\n{text}",
        view.title,
        kind_word(view.kind),
        view.version,
        view.versions
    ))
}

fn kind_word(kind: ArtifactKind) -> &'static str {
    match kind {
        ArtifactKind::Markdown => "markdown",
        ArtifactKind::Html => "html",
        ArtifactKind::Svg => "svg",
        ArtifactKind::Text => "text",
    }
}

/// One version of the artifact `name` of `conversation` (the latest when
/// `version` is `None`), with its text read from the record.
pub(crate) fn read(
    inner: &Inner,
    conversation: &str,
    name: &str,
    version: Option<u32>,
) -> Result<ArtifactView, Refusal> {
    if !is_name(name) {
        return Err(refuse(RefusalKind::NotFound, words::NONE));
    }
    let items = inner
        .sidecars
        .read_items(conversation)
        .map(|log| log.items)
        .map_err(|_| refuse(RefusalKind::Unavailable, words::UNREADABLE))?;
    let known = artifacts(&items);
    let versions = known
        .get(name)
        .ok_or_else(|| refuse(RefusalKind::NotFound, words::NONE))?;
    let number = version.unwrap_or(versions.len() as u32);
    let chosen = number
        .checked_sub(1)
        .and_then(|at| versions.get(at as usize))
        .ok_or_else(|| refuse(RefusalKind::NotFound, words::NO_VERSION))?;
    let text = match &chosen.text {
        Payload::Inline(text) => text.clone(),
        Payload::Blob { sha256, .. } => {
            let bytes = inner
                .sidecars
                .read_blob(conversation, sha256)
                .map_err(|_| refuse(RefusalKind::Unavailable, words::UNREADABLE))?;
            String::from_utf8(bytes)
                .map_err(|_| refuse(RefusalKind::Unavailable, words::UNREADABLE))?
        }
    };
    Ok(ArtifactView {
        name: name.to_owned(),
        title: chosen.title.clone(),
        kind: chosen.kind,
        version: number,
        versions: versions.len() as u32,
        text,
    })
}
