//! The preview browser: where the reader previews an HTML or SVG artifact the
//! agent saved ("artifacts", 2026-10-08). Not a port.
//!
//! It is a copy of the installed Edge (else Chrome) of its own, apart from
//! the agent's browser: its own profile ([`profile_dir`]), so none of the
//! reader's sign-ins, and no network at all ([`NO_NETWORK`]):
//! - every request goes to a proxy that is not there (`127.0.0.1:9`), this
//!   PC's own addresses included (`<-loopback>`);
//! - every name resolves to nothing, so no lookup leaves the PC;
//! - WebRTC may send UDP only through that proxy;
//! - its own background traffic (updates, sync, safe browsing) is off.
//!
//! A page is opened as a `data:` address, so its origin is opaque (no cookies
//! or storage), with a Content Security Policy that lets its own inline script
//! and style run and loads nothing from elsewhere ([`page`]). The DevTools
//! pipes, the Job Object and the start wait are the agent's browser's
//! ([`super::launch`]); the agent has no tool that reaches this browser.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use lattice_protocol::conversation::ArtifactKind;
use lattice_sys::process::Child;
use serde_json::json;
use tokio::runtime::Handle;

use super::cdp::{Cdp, Endpoint};
use super::launch;
use crate::env::Env;
use crate::state::StateRoot;

/// The flags that take the network away.
pub const NO_NETWORK: [&str; 8] = [
    "--proxy-server=http://127.0.0.1:9",
    "--proxy-bypass-list=<-loopback>",
    "--host-resolver-rules=MAP * ~NOTFOUND",
    "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-domain-reliability",
    "--no-pings",
];

/// The Content Security Policy every previewed page carries: its own inline
/// script and style, and pictures, fonts and media written into it; nothing
/// fetched, framed or posted.
pub const POLICY: &str = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; media-src data: blob:; form-action 'none'; base-uri 'none'";

/// The window's size.
const SIZE: (u32, u32) = (1280, 860);

/// `<globals>/lattice_native/browser/preview-profile`.
pub fn profile_dir(state: &StateRoot) -> PathBuf {
    state
        .globals
        .join("lattice_native")
        .join("browser")
        .join("preview-profile")
}

/// The preview browser's arguments: the agent's browser's, with
/// [`NO_NETWORK`] before the first page.
pub fn arguments(program: &Path, profile: &Path, headless: bool) -> Vec<OsString> {
    let mut argv = launch::arguments(program, profile, headless, SIZE);
    let page = argv.pop();
    argv.extend(NO_NETWORK.iter().map(OsString::from));
    argv.extend(page);
    argv
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// The document a preview shows: the policy first, then the artifact (an SVG
/// inside a page of its own).
pub fn page(kind: ArtifactKind, title: &str, text: &str) -> String {
    let head = format!(
        "<!doctype html><meta charset=\"utf-8\"><meta http-equiv=\"Content-Security-Policy\" content=\"{POLICY}\"><meta http-equiv=\"x-dns-prefetch-control\" content=\"off\">"
    );
    match kind {
        ArtifactKind::Svg => format!(
            "{head}<title>{}</title><body style=\"margin:0;display:grid;place-items:center;min-height:100vh\">{text}</body>",
            escape(title)
        ),
        _ => format!("{head}{text}"),
    }
}

/// `html` as a `data:` address.
pub fn data_url(html: &str) -> String {
    format!(
        "data:text/html;charset=utf-8;base64,{}",
        super::ws::base64(html.as_bytes())
    )
}

struct Live {
    cdp: Arc<Cdp>,
    child: Mutex<Option<Child>>,
}

/// The preview browser, started at its first preview.
pub struct PreviewBrowser {
    state: StateRoot,
    env: Arc<dyn Env>,
    handle: Handle,
    headless: bool,
    program: Option<PathBuf>,
    profile: Option<PathBuf>,
    live: tokio::sync::Mutex<Option<Arc<Live>>>,
}

impl PreviewBrowser {
    pub fn new(state: StateRoot, env: Arc<dyn Env>, handle: Handle) -> Self {
        Self {
            state,
            env,
            handle,
            headless: false,
            program: None,
            profile: None,
            live: tokio::sync::Mutex::new(None),
        }
    }

    /// Tests: headless, and another program or profile.
    pub fn with_test_config(
        mut self,
        headless: bool,
        program: Option<PathBuf>,
        profile: Option<PathBuf>,
    ) -> Self {
        self.headless = headless;
        self.program = program;
        self.profile = profile;
        self
    }

    async fn live(&self) -> Result<Arc<Live>, String> {
        let mut slot = self.live.lock().await;
        if let Some(live) = slot.as_ref()
            && live.cdp.closed().is_none()
        {
            return Ok(live.clone());
        }
        if let Some(old) = slot.take() {
            drop(old.child.lock().unwrap_or_else(|p| p.into_inner()).take());
        }
        let program = match &self.program {
            Some(program) => program.clone(),
            None => launch::find_browser(self.env.as_ref()).ok_or_else(|| {
                "Neither Microsoft Edge nor Google Chrome is installed where its installer puts it.".to_owned()
            })?,
        };
        let profile = self
            .profile
            .clone()
            .unwrap_or_else(|| profile_dir(&self.state));
        let (headless, env, globals) =
            (self.headless, self.env.clone(), self.state.globals.clone());
        let started = self
            .handle
            .spawn_blocking(move || {
                std::fs::create_dir_all(&profile).map_err(|_| {
                    "The preview browser's profile folder could not be made.".to_owned()
                })?;
                launch::start_with(
                    &program,
                    &arguments(&program, &profile, headless),
                    env.as_ref(),
                    &globals,
                )
            })
            .await
            .map_err(|_| "The preview browser could not start.".to_owned())??;
        let launch::Started { child, pipes, .. } = started;
        let cdp = Cdp::connect(Endpoint::Pipe(pipes), Arc::new(|_| {}))?;
        if let Err(why) = cdp
            .call("Browser.getVersion", json!({}), None, launch::START_WAIT)
            .await
        {
            let open = cdp.closed().is_none();
            drop(cdp);
            return Err(self
                .handle
                .spawn_blocking(move || {
                    launch::start_failure(child, &why, open, launch::START_WAIT)
                })
                .await
                .unwrap_or_else(|_| "The preview browser did not start.".to_owned()));
        }
        let live = Arc::new(Live {
            cdp,
            child: Mutex::new(Some(child)),
        });
        *slot = Some(live.clone());
        Ok(live)
    }

    /// Show `html` in a tab of its own, in front, the browser started first
    /// if it is not.
    pub async fn show(&self, html: &str) -> Result<(), String> {
        let live = self.live().await?;
        let created = live
            .cdp
            .call(
                "Target.createTarget",
                json!({"url": data_url(html)}),
                None,
                launch::START_WAIT,
            )
            .await
            .map_err(|why| format!("The preview could not be opened: {why}"))?;
        if let Some(target) = created.get("targetId").and_then(|t| t.as_str()) {
            let _ = live
                .cdp
                .call(
                    "Target.activateTarget",
                    json!({"targetId": target}),
                    None,
                    launch::START_WAIT,
                )
                .await;
        }
        Ok(())
    }

    /// End it, if it runs.
    pub async fn stop(&self) {
        if let Some(live) = self.live.lock().await.take() {
            drop(live.child.lock().unwrap_or_else(|p| p.into_inner()).take());
        }
    }

    /// The process ids of its tree (tests).
    #[cfg(test)]
    pub(crate) async fn process_ids(&self) -> Vec<u32> {
        match self.live.lock().await.as_ref() {
            Some(live) => live
                .child
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .as_ref()
                .and_then(|child| child.process_ids().ok())
                .unwrap_or_default(),
            None => Vec::new(),
        }
    }
}
