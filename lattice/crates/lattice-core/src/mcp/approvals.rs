//! The reader's MCP decisions (the chat core's spec §12, "Server
//! approval" and "Calls"; FT4, FT5). Not a port.
//!
//! `<native>/chat/mcp_approvals.json`:
//! `{"v": 1, "servers": [...], "tools": [...], "switches": [...], "revoked": [...]}`.
//! - **servers**: `{scope, name, sha256, approved_at}`: the reader enabled
//!   this entry, pinned to the SHA-256 of its canonical JSON
//!   ([`super::config::canonical`]). An entry whose hash has no approval is
//!   not enabled, so any change to it asks again.
//! - **tools**: `{scope, name, server_sha256, tool, allowed_at}`: "allow
//!   always" for one tool of one server, through `ConfirmPort(AllowMcpTool)`.
//!   It holds only while the server's entry has that hash.
//! - **switches**: `{scope, name, tool, on, at}`: a tool the reader switched
//!   off (or on again), so the model is not offered it; `"*"` is the whole
//!   server, for a folder's server (whose file Lattice never writes).
//! - **revoked**: `{scope, name, tool, revoked_at}`: the end of an approval
//!   (`tool` is `""`) or of an "allow always". As trust's revocations (FT5),
//!   nothing is removed: the newest record decides, and a tie goes to the
//!   revocation (fail closed).
//!
//! A `scope` is `{"kind": "user"}` or `{"kind": "folder", "id", "path"}`, a
//! folder matched by both its id and its canonical path, as trust is (§4.1).
//! A file that cannot be read or parsed approves nothing and is never
//! overwritten. Only the reader's own actions write here: nothing in a
//! folder, a model's output or a server's reply does (FT4).

use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use super::config::{Scope, ServerEntry, ServerKey};
use crate::clock::Clock;
use crate::exec::spawn::comparable;
use crate::fsx;
use crate::state::StateRoot;

/// The file's name, in `<native>/chat`.
pub const APPROVALS_FILE: &str = "mcp_approvals.json";
/// The file's version.
pub const APPROVALS_VERSION: u32 = 1;
/// The largest approvals file read.
pub const MAX_APPROVALS_FILE: u64 = 4 * 1024 * 1024;
/// A switch's `tool` for the whole server.
pub const WHOLE_SERVER: &str = "*";

/// Why a decision was not saved.
pub const NOT_SAVED: &str = "Lattice's MCP approvals could not be read, so nothing was changed.";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ScopeRecord {
    User,
    Folder { id: String, path: String },
}

impl ScopeRecord {
    fn of(scope: &Scope) -> Self {
        match scope {
            Scope::User => Self::User,
            Scope::Folder { id, path } => Self::Folder {
                id: id.clone(),
                path: path.clone(),
            },
        }
    }

    fn matches(&self, scope: &Scope) -> bool {
        match (self, scope) {
            (Self::User, Scope::User) => true,
            (
                Self::Folder { id, path },
                Scope::Folder {
                    id: other_id,
                    path: other_path,
                },
            ) => id == other_id && comparable(path) == comparable(other_path),
            _ => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ServerRecord {
    scope: ScopeRecord,
    name: String,
    sha256: String,
    approved_at: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ToolRecord {
    scope: ScopeRecord,
    name: String,
    server_sha256: String,
    tool: String,
    allowed_at: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct SwitchRecord {
    scope: ScopeRecord,
    name: String,
    tool: String,
    on: bool,
    at: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct RevokedRecord {
    scope: ScopeRecord,
    name: String,
    /// `""`: the server's approval; otherwise the tool's "allow always".
    tool: String,
    revoked_at: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct ApprovalsFile {
    v: u32,
    #[serde(default)]
    servers: Vec<ServerRecord>,
    #[serde(default)]
    tools: Vec<ToolRecord>,
    #[serde(default)]
    switches: Vec<SwitchRecord>,
    #[serde(default)]
    revoked: Vec<RevokedRecord>,
}

/// The reader's MCP decisions, read from the file at each question (it is
/// small), written only by the reader's actions.
pub struct Approvals {
    file: PathBuf,
    clock: Clock,
    write: Mutex<()>,
}

fn newest(times: impl Iterator<Item = f64>) -> f64 {
    times.fold(f64::NEG_INFINITY, f64::max)
}

impl Approvals {
    pub fn new(state: &StateRoot, clock: Clock) -> Self {
        Self {
            file: state.native_chat_dir().join(APPROVALS_FILE),
            clock,
            write: Mutex::default(),
        }
    }

    /// `<native>/chat/mcp_approvals.json`.
    pub fn file(&self) -> &std::path::Path {
        &self.file
    }

    fn read(&self) -> Result<ApprovalsFile, ()> {
        let bytes = match std::fs::read(&self.file) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(ApprovalsFile {
                    v: APPROVALS_VERSION,
                    ..ApprovalsFile::default()
                });
            }
            Err(_) => return Err(()),
        };
        if bytes.len() as u64 > MAX_APPROVALS_FILE {
            return Err(());
        }
        let file: ApprovalsFile = serde_json::from_slice(&bytes).map_err(|_| ())?;
        if file.v != APPROVALS_VERSION {
            return Err(());
        }
        Ok(file)
    }

    /// Read, change and write the file under the write lock, or say why not.
    fn change(&self, edit: impl FnOnce(&mut ApprovalsFile, f64)) -> Result<(), String> {
        let _held = self
            .write
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut file = self.read().map_err(|_| NOT_SAVED.to_owned())?;
        edit(&mut file, (self.clock)());
        let parent = self.file.parent().ok_or_else(|| NOT_SAVED.to_owned())?;
        std::fs::create_dir_all(parent).map_err(|_| NOT_SAVED.to_owned())?;
        let bytes = serde_json::to_vec_pretty(&file).map_err(|_| NOT_SAVED.to_owned())?;
        fsx::atomic_write(&self.file, &bytes)
            .map_err(|_| "Lattice's MCP approvals could not be saved.".to_owned())
    }

    fn revoked_at(file: &ApprovalsFile, key: &ServerKey, tool: &str) -> f64 {
        newest(
            file.revoked
                .iter()
                .filter(|record| {
                    record.scope.matches(&key.scope)
                        && record.name == key.name
                        && record.tool == tool
                })
                .map(|record| record.revoked_at),
        )
    }

    /// Is this entry enabled: an approval of exactly its hash, newer than
    /// any revocation of the server's approval?
    pub fn approved(&self, entry: &ServerEntry) -> bool {
        let Ok(file) = self.read() else {
            return false;
        };
        let approved = newest(
            file.servers
                .iter()
                .filter(|record| {
                    record.scope.matches(&entry.key.scope)
                        && record.name == entry.key.name
                        && record.sha256 == entry.sha256
                })
                .map(|record| record.approved_at),
        );
        approved.is_finite() && approved > Self::revoked_at(&file, &entry.key, "")
    }

    /// Record that the reader enabled `entry` (its hash).
    pub fn approve(&self, entry: &ServerEntry) -> Result<(), String> {
        self.change(|file, now| {
            file.servers.push(ServerRecord {
                scope: ScopeRecord::of(&entry.key.scope),
                name: entry.key.name.clone(),
                sha256: entry.sha256.clone(),
                approved_at: now,
            });
        })
    }

    /// End the server's approval (it asks again before it next starts).
    pub fn revoke(&self, key: &ServerKey) -> Result<(), String> {
        self.revoke_named(key, "")
    }

    fn revoke_named(&self, key: &ServerKey, tool: &str) -> Result<(), String> {
        self.change(|file, now| {
            file.revoked.push(RevokedRecord {
                scope: ScopeRecord::of(&key.scope),
                name: key.name.clone(),
                tool: tool.to_owned(),
                revoked_at: now,
            });
        })
    }

    /// Is `tool` of this entry allowed always (for exactly its hash)?
    pub fn allowed(&self, entry: &ServerEntry, tool: &str) -> bool {
        self.allowed_for(&entry.key, &entry.sha256, tool)
    }

    /// [`Approvals::allowed`] by the server's key and its entry's hash.
    pub fn allowed_for(&self, key: &ServerKey, sha256: &str, tool: &str) -> bool {
        let Ok(file) = self.read() else {
            return false;
        };
        let allowed = newest(
            file.tools
                .iter()
                .filter(|record| {
                    record.scope.matches(&key.scope)
                        && record.name == key.name
                        && record.server_sha256 == sha256
                        && record.tool == tool
                })
                .map(|record| record.allowed_at),
        );
        allowed.is_finite() && allowed > Self::revoked_at(&file, key, tool)
    }

    /// Record "allow always" for `tool` of `entry`. The caller has had the
    /// reader's yes in `ConfirmPort(AllowMcpTool)`.
    pub fn allow(&self, entry: &ServerEntry, tool: &str) -> Result<(), String> {
        self.allow_for(&entry.key, &entry.sha256, tool)
    }

    /// [`Approvals::allow`] by the server's key and its entry's hash.
    pub fn allow_for(&self, key: &ServerKey, sha256: &str, tool: &str) -> Result<(), String> {
        if tool.is_empty() || tool == WHOLE_SERVER {
            return Err("That is not a tool's name.".to_owned());
        }
        self.change(|file, now| {
            file.tools.push(ToolRecord {
                scope: ScopeRecord::of(&key.scope),
                name: key.name.clone(),
                server_sha256: sha256.to_owned(),
                tool: tool.to_owned(),
                allowed_at: now,
            });
        })
    }

    /// End "allow always" for `tool`: its calls ask again.
    pub fn disallow(&self, key: &ServerKey, tool: &str) -> Result<(), String> {
        if tool.is_empty() || tool == WHOLE_SERVER {
            return Err("That is not a tool's name.".to_owned());
        }
        self.revoke_named(key, tool)
    }

    /// Is `tool` (or [`WHOLE_SERVER`]) switched on? The newest switch
    /// decides; with none, it is on.
    pub fn is_on(&self, key: &ServerKey, tool: &str) -> bool {
        let Ok(file) = self.read() else {
            return true;
        };
        file.switches
            .iter()
            .filter(|record| {
                record.scope.matches(&key.scope) && record.name == key.name && record.tool == tool
            })
            .max_by(|a, b| a.at.total_cmp(&b.at))
            .is_none_or(|record| record.on)
    }

    /// Switch `tool` (or [`WHOLE_SERVER`]) on or off.
    pub fn switch(&self, key: &ServerKey, tool: &str, on: bool) -> Result<(), String> {
        self.change(|file, now| {
            file.switches.push(SwitchRecord {
                scope: ScopeRecord::of(&key.scope),
                name: key.name.clone(),
                tool: tool.to_owned(),
                on,
                at: now,
            });
        })
    }
}
