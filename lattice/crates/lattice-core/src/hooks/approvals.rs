//! The hooks the reader allowed: `<native>/chat/hook_approvals.json`, each
//! kept by [`Hook::digest`] (its source, event, matcher, exact command and
//! file), so any change to a hook asks again. An unreadable file allows
//! nothing; it is written whole, beside itself first.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use super::Hook;
use crate::state::StateRoot;

static WRITING: Mutex<()> = Mutex::new(());

/// One allowed hook.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Allowed {
    pub digest: String,
    pub source: String,
    pub event: String,
    pub command: String,
    /// Seconds since the epoch.
    pub at: f64,
}

#[derive(Default, Serialize, Deserialize)]
struct File {
    v: u32,
    allowed: Vec<Allowed>,
}

pub fn file(state: &StateRoot) -> PathBuf {
    state.native_chat_dir().join("hook_approvals.json")
}

fn read(path: &Path) -> Vec<Allowed> {
    std::fs::read(path)
        .ok()
        .filter(|b| b.len() <= 4 * 1024 * 1024)
        .and_then(|b| serde_json::from_slice::<File>(&b).ok())
        .filter(|f| f.v == 1)
        .map(|f| f.allowed)
        .unwrap_or_default()
}

fn write(path: &Path, allowed: Vec<Allowed>) -> Result<(), String> {
    let failed = || "Lattice could not save the hooks you allowed.".to_owned();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|_| failed())?;
    }
    let bytes = serde_json::to_vec_pretty(&File { v: 1, allowed }).map_err(|_| failed())?;
    crate::fsx::atomic_write(path, &bytes).map_err(|_| failed())
}

/// Whether `hook`, exactly as it is, was allowed.
pub fn allowed(path: &Path, hook: &Hook) -> bool {
    let digest = hook.digest();
    read(path).iter().any(|a| a.digest == digest)
}

/// Every allowed hook.
pub fn list(path: &Path) -> Vec<Allowed> {
    read(path)
}

/// Keep `hook` allowed.
pub fn allow(path: &Path, hook: &Hook, at: f64) -> Result<(), String> {
    let _held = WRITING.lock().unwrap_or_else(|e| e.into_inner());
    let mut all = read(path);
    let digest = hook.digest();
    if !all.iter().any(|a| a.digest == digest) {
        all.push(Allowed {
            digest,
            source: hook.source.label(),
            event: hook.native.clone(),
            command: hook.command.clone(),
            at,
        });
    }
    write(path, all)
}

/// Take back the allowance `digest`; `false` when there was none.
pub fn revoke(path: &Path, digest: &str) -> Result<bool, String> {
    let _held = WRITING.lock().unwrap_or_else(|e| e.into_inner());
    let mut all = read(path);
    let before = all.len();
    all.retain(|a| a.digest != digest);
    if all.len() == before {
        return Ok(false);
    }
    write(path, all).map(|()| true)
}
