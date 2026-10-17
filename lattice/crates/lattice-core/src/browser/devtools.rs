//! What a developer's tools give, for the agent's browser: a script run in
//! the page, what the page wrote to its console, the requests it made (and a
//! response's text), elements found by their words and forms filled by
//! reference, and the page's size and colour scheme. Not a port.
//!
//! - **A script** ([`AgentBrowser::run_script`]) runs in the page as the
//!   page's own would, its value returned as JSON, at most [`MAX_TEXT`]
//!   characters, redacted. Its stated effect asks as a click's does, and a
//!   script that reaches a password, a card, cookies or a site's stored data
//!   is refused before it runs ([`policy::script_looks`]).
//! - **The console and the network** are kept from the page's events, the
//!   newest [`MAX_KEPT`] of each, for as long as the browser runs. A response
//!   is read only when asked for, as text, redacted.
//! - **Finding** ([`AgentBrowser::find`]) marks the visible elements whose
//!   words match with a reference (`r1`, `r2` …) and says where each is;
//!   **filling** ([`AgentBrowser::fill`]) sets a field, a list or a box by
//!   its reference as typing would, and never a password or card field.
//! - **The viewport** ([`AgentBrowser::set_viewport`]): the page's size, a
//!   phone's width, and light or dark.
//! - **Tabs** ([`AgentBrowser::tabs`]): the browser's pages listed, a new one
//!   opened (the same address rules), another made the agent's, one closed
//!   (never the last).
//! - **Upload** ([`AgentBrowser::upload`]): a file of the folder put in a
//!   page's file field by reference, as a person picking it would; the caller
//!   checks the path and asks the reader first.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Duration;

use serde::Deserialize;
use serde_json::{Value, json};

use super::cdp::Event;
use super::policy::{self, Effect};
use super::session::{AgentBrowser, Live, Shot, VIEWPORT, evaluate};
use crate::secrets;

/// The most console lines and requests kept.
pub const MAX_KEPT: usize = 500;
/// The most lines or requests one answer lists.
pub const MAX_LISTED: usize = 100;
/// The most characters of a script's value or a response's text.
pub const MAX_TEXT: usize = 20_000;
/// The longest script.
pub const MAX_SCRIPT: usize = 10_000;
/// How long a script may run.
const SCRIPT_WAIT: Duration = Duration::from_secs(10);
/// The smallest and largest page.
pub const MIN_SIZE: (u32, u32) = (320, 240);
pub const MAX_SIZE: (u32, u32) = (3840, 2160);

/// One line the page wrote to its console, or an error it threw.
#[derive(Clone, Debug, PartialEq)]
pub struct ConsoleLine {
    /// `log`, `info`, `warning`, `error` or `debug`.
    pub level: String,
    pub text: String,
    /// Where it came from, `address:line`, when said.
    pub source: String,
}

/// One request the page made.
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    pub id: String,
    pub method: String,
    pub url: String,
    /// The resource's kind (`Document`, `XHR`, `Fetch`, `Script` …).
    pub kind: String,
    pub status: Option<u16>,
    pub mime: String,
    /// Why it failed, when it did.
    pub failed: Option<String>,
}

/// The page's size and colour scheme.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
    /// `Some(true)` dark, `Some(false)` light, `None` as the system says.
    pub dark: Option<bool>,
}

impl Default for Viewport {
    fn default() -> Self {
        Self { width: VIEWPORT.0, height: VIEWPORT.1, dark: None }
    }
}

impl Viewport {
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn cut(text: &str, most: usize) -> String {
    let mut out: String = text.chars().take(most).collect();
    if text.chars().count() > most {
        out.push_str(" [cut]");
    }
    out
}

/// A console event, kept.
pub(super) fn on_console(kept: &Mutex<VecDeque<ConsoleLine>>, event: &Event) {
    let p = &event.params;
    let line = match event.method.as_str() {
        "Runtime.consoleAPICalled" => {
            let text = p
                .get("args")
                .and_then(Value::as_array)
                .map(|args| {
                    args.iter()
                        .map(|a| {
                            a.get("value")
                                .map(|v| match v {
                                    Value::String(s) => s.clone(),
                                    other => other.to_string(),
                                })
                                .or_else(|| a.get("description").and_then(Value::as_str).map(str::to_owned))
                                .unwrap_or_default()
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .unwrap_or_default();
            let frame = p.get("stackTrace").and_then(|s| s.get("callFrames")).and_then(|f| f.get(0));
            let source = frame
                .map(|f| {
                    format!(
                        "{}:{}",
                        f.get("url").and_then(Value::as_str).unwrap_or(""),
                        f.get("lineNumber").and_then(Value::as_u64).map_or(0, |n| n + 1)
                    )
                })
                .unwrap_or_default();
            let level = match p.get("type").and_then(Value::as_str).unwrap_or("log") {
                "warning" => "warning",
                "error" | "assert" => "error",
                "debug" => "debug",
                "info" => "info",
                _ => "log",
            };
            ConsoleLine { level: level.to_owned(), text, source }
        }
        "Runtime.exceptionThrown" => {
            let details = p.get("exceptionDetails");
            let text = details
                .and_then(|d| d.get("exception"))
                .and_then(|e| e.get("description"))
                .and_then(Value::as_str)
                .or_else(|| details.and_then(|d| d.get("text")).and_then(Value::as_str))
                .unwrap_or("An error was thrown.")
                .to_owned();
            let source = details
                .map(|d| {
                    format!(
                        "{}:{}",
                        d.get("url").and_then(Value::as_str).unwrap_or(""),
                        d.get("lineNumber").and_then(Value::as_u64).map_or(0, |n| n + 1)
                    )
                })
                .unwrap_or_default();
            ConsoleLine { level: "error".to_owned(), text, source }
        }
        "Log.entryAdded" => {
            let entry = p.get("entry");
            let field = |k: &str| entry.and_then(|e| e.get(k)).and_then(Value::as_str).unwrap_or("").to_owned();
            let level = match field("level").as_str() {
                "error" => "error",
                "warning" => "warning",
                "verbose" => "debug",
                _ => "info",
            };
            ConsoleLine { level: level.to_owned(), text: field("text"), source: field("url") }
        }
        _ => return,
    };
    let line = ConsoleLine { text: secrets::redact(&cut(&line.text, 2_000)), ..line };
    let mut kept = lock(kept);
    kept.push_back(line);
    while kept.len() > MAX_KEPT {
        kept.pop_front();
    }
}

/// A network event, kept.
pub(super) fn on_network(kept: &Mutex<VecDeque<Request>>, event: &Event) {
    let p = &event.params;
    let Some(id) = p.get("requestId").and_then(Value::as_str) else { return };
    let mut kept = lock(kept);
    match event.method.as_str() {
        "Network.requestWillBeSent" => {
            let request = p.get("request");
            let field = |k: &str| request.and_then(|r| r.get(k)).and_then(Value::as_str).unwrap_or("").to_owned();
            // A redirect sends the same id again: the request is the new one.
            kept.retain(|r| r.id != id);
            kept.push_back(Request {
                id: id.to_owned(),
                method: field("method"),
                url: cut(&field("url"), 2_000),
                kind: p.get("type").and_then(Value::as_str).unwrap_or("Other").to_owned(),
                status: None,
                mime: String::new(),
                failed: None,
            });
            while kept.len() > MAX_KEPT {
                kept.pop_front();
            }
        }
        "Network.responseReceived" => {
            if let Some(r) = kept.iter_mut().rev().find(|r| r.id == id) {
                let response = p.get("response");
                r.status = response.and_then(|x| x.get("status")).and_then(Value::as_f64).map(|s| s as u16);
                r.mime = response.and_then(|x| x.get("mimeType")).and_then(Value::as_str).unwrap_or("").to_owned();
            }
        }
        "Network.loadingFailed" => {
            if let Some(r) = kept.iter_mut().rev().find(|r| r.id == id) {
                r.failed = Some(p.get("errorText").and_then(Value::as_str).unwrap_or("failed").to_owned());
            }
        }
        _ => {}
    }
}

/// Set the page's size and colour scheme.
pub(super) async fn apply_viewport(live: &Live, session: Option<&str>, viewport: Viewport) -> Result<(), String> {
    live.cdp
        .call(
            "Emulation.setDeviceMetricsOverride",
            json!({"width": viewport.width, "height": viewport.height, "deviceScaleFactor": 1, "mobile": viewport.width < 700}),
            session,
            super::session::CALL,
        )
        .await?;
    let scheme = match viewport.dark {
        Some(true) => "dark",
        Some(false) => "light",
        None => "",
    };
    let _ = live
        .cdp
        .call(
            "Emulation.setEmulatedMedia",
            json!({"features": [{"name": "prefers-color-scheme", "value": scheme}]}),
            session,
            super::session::CALL,
        )
        .await;
    Ok(())
}

/// The console lines, newest last: at most [`MAX_LISTED`], only errors and
/// warnings when `problems`.
pub fn console_text(lines: &VecDeque<ConsoleLine>, problems: bool) -> String {
    let shown: Vec<&ConsoleLine> =
        lines.iter().filter(|l| !problems || l.level == "error" || l.level == "warning").collect();
    if shown.is_empty() {
        return if problems { "The console has no errors or warnings.".to_owned() } else { "The console is empty.".to_owned() };
    }
    let skip = shown.len().saturating_sub(MAX_LISTED);
    let mut out = String::new();
    if skip > 0 {
        out.push_str(&format!("({skip} older lines not shown)\n"));
    }
    for l in &shown[skip..] {
        let source = if l.source.is_empty() { String::new() } else { format!(" ({})", l.source) };
        out.push_str(&format!("[{}] {}{source}\n", l.level, l.text));
    }
    out
}

/// The requests, newest last: at most [`MAX_LISTED`], those whose address
/// holds `filter` when given.
pub fn network_text(requests: &VecDeque<Request>, filter: Option<&str>) -> String {
    let filter = filter.map(str::to_lowercase).filter(|f| !f.trim().is_empty());
    let shown: Vec<&Request> =
        requests.iter().filter(|r| filter.as_deref().is_none_or(|f| r.url.to_lowercase().contains(f))).collect();
    if shown.is_empty() {
        return "No requests match.".to_owned();
    }
    let skip = shown.len().saturating_sub(MAX_LISTED);
    let mut out = String::new();
    if skip > 0 {
        out.push_str(&format!("({skip} older requests not shown)\n"));
    }
    for r in &shown[skip..] {
        let status = match (&r.failed, r.status) {
            (Some(why), _) => format!("failed: {why}"),
            (None, Some(s)) => s.to_string(),
            (None, None) => "pending".to_owned(),
        };
        out.push_str(&format!("{} {} {} {} [{}]\n", r.id, r.method, status, r.url, r.kind));
    }
    out
}

/// One element [`FIND_JS`] found.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Found {
    #[serde(rename = "ref")]
    pub reference: String,
    pub tag: String,
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub label: String,
    pub x: f64,
    pub y: f64,
    #[serde(default)]
    pub kind: String,
}

/// The visible elements whose words hold the query (every word of it), marked
/// with a reference: at most 30, actionable ones first. Called with the query.
const FIND_JS: &str = r#"((query) => {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  const pick = (v) => (v || '').toString().replace(/\s+/g, ' ').trim().slice(0, 160);
  const all = Array.from(document.querySelectorAll('a, button, input, select, textarea, summary, label, h1, h2, h3, h4, [role], [contenteditable="true"], [aria-label], [placeholder]'));
  let next = Number(document.documentElement.getAttribute('data-lattice-next') || '1');
  const found = [];
  for (const el of all) {
    const box = el.getBoundingClientRect();
    if (box.width < 1 || box.height < 1 || box.bottom < 0 || box.right < 0 || box.top > innerHeight || box.left > innerWidth) continue;
    const style = getComputedStyle(el);
    if (style.visibility === 'hidden' || style.display === 'none') continue;
    const label = pick(el.getAttribute('aria-label') || el.innerText || el.value || el.getAttribute('placeholder') || el.getAttribute('title') || el.getAttribute('name'));
    const hay = (label + ' ' + (el.getAttribute('name') || '') + ' ' + (el.id || '')).toLowerCase();
    if (!words.every((w) => hay.includes(w))) continue;
    let ref = el.getAttribute('data-lattice-ref');
    if (!ref) { ref = 'r' + next++; el.setAttribute('data-lattice-ref', ref); }
    const type = (el.getAttribute('type') || '').toLowerCase();
    const auto = (el.getAttribute('autocomplete') || '').toLowerCase();
    const names = ((el.getAttribute('name') || '') + ' ' + (el.id || '')).toLowerCase();
    let kind = '';
    if (type === 'password' || auto.includes('password')) kind = 'password';
    else if (auto.startsWith('cc-') || /card.?num|cvc|cvv|csc|security.?code/.test(names)) kind = 'card';
    const actionable = el.matches('a, button, input, select, textarea, summary, [role="button"], [role="link"], [role="checkbox"], [role="tab"], [contenteditable="true"]');
    found.push({ref, tag: el.tagName.toLowerCase(), role: el.getAttribute('role') || '', label, x: Math.round(box.left + box.width / 2), y: Math.round(box.top + box.height / 2), kind, actionable});
    if (found.length >= 60) break;
  }
  document.documentElement.setAttribute('data-lattice-next', String(next));
  found.sort((a, b) => (b.actionable ? 1 : 0) - (a.actionable ? 1 : 0));
  return found.slice(0, 30);
})"#;

/// Fill the element with a reference: a text field or area as typing would,
/// a list by an option's text or value, a box with `true` or `false`.
/// Called with the reference and the value.
const FILL_JS: &str = r#"((ref, value) => {
  const el = document.querySelector('[data-lattice-ref="' + ref.replace(/"/g, '') + '"]');
  if (!el) return {kind: 'gone'};
  const type = (el.getAttribute('type') || '').toLowerCase();
  const auto = (el.getAttribute('autocomplete') || '').toLowerCase();
  const names = ((el.getAttribute('name') || '') + ' ' + (el.id || '')).toLowerCase();
  if (type === 'password' || auto.includes('password')) return {kind: 'password'};
  if (auto.startsWith('cc-') || /card.?num|cvc|cvv|csc|security.?code/.test(names)) return {kind: 'card'};
  el.scrollIntoView({block: 'center'});
  el.focus();
  const fire = () => { el.dispatchEvent(new Event('input', {bubbles: true})); el.dispatchEvent(new Event('change', {bubbles: true})); };
  if (el.tagName === 'SELECT') {
    const option = Array.from(el.options).find((o) => o.value === value || o.text.trim() === value);
    if (!option) return {kind: 'no-option'};
    el.value = option.value; fire(); return {kind: 'ok'};
  }
  if (type === 'checkbox' || type === 'radio') {
    const on = value === 'true';
    if (el.checked !== on) el.click();
    return {kind: 'ok'};
  }
  if (el.isContentEditable) { el.textContent = value; fire(); return {kind: 'ok'}; }
  if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
    const proto = el.tagName === 'INPUT' ? HTMLInputElement.prototype : HTMLTextAreaElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, value);
    fire(); return {kind: 'ok'};
  }
  return {kind: 'not-a-field'};
})"#;

#[derive(Deserialize, Default)]
struct Filled {
    #[serde(default)]
    kind: String,
}

impl AgentBrowser {
    /// Run `script` in the page and say its value, as JSON. A script that
    /// reaches a password, a card, cookies or a site's stored data is refused
    /// before it runs ([`policy::script_looks`]).
    pub async fn run_script(&self, script: &str, _stated: Effect) -> Result<String, String> {
        let script = script.trim();
        if script.is_empty() {
            return Err("Give the script to run.".to_owned());
        }
        if script.chars().count() > MAX_SCRIPT {
            return Err(format!("A script is at most {MAX_SCRIPT} characters."));
        }
        if let Some(why) = policy::script_looks(script) {
            return Err(why.to_owned());
        }
        let live = self.agent_page().await?;
        let session = live.session();
        let answer = live
            .cdp
            .call(
                "Runtime.evaluate",
                json!({"expression": script, "returnByValue": true, "awaitPromise": true, "timeout": SCRIPT_WAIT.as_millis() as u64}),
                session.as_deref(),
                SCRIPT_WAIT + Duration::from_secs(5),
            )
            .await
            .map_err(|why| format!("The script could not run: {why}"))?;
        if let Some(details) = answer.get("exceptionDetails") {
            let why = details
                .get("exception")
                .and_then(|e| e.get("description"))
                .and_then(Value::as_str)
                .or_else(|| details.get("text").and_then(Value::as_str))
                .unwrap_or("an error");
            return Err(secrets::redact(&cut(&format!("The script threw: {why}"), MAX_TEXT)));
        }
        let result = answer.get("result").cloned().unwrap_or(Value::Null);
        let value = match result.get("value") {
            Some(v) => v.to_string(),
            None => result
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("undefined")
                .to_owned(),
        };
        Ok(secrets::redact(&cut(&value, MAX_TEXT)))
    }

    /// What the page wrote to its console (errors and warnings only, when
    /// `problems`).
    pub async fn console(&self, problems: bool) -> Result<String, String> {
        let live = self.agent_page().await?;
        let lines = lock(&live.events.console);
        Ok(console_text(&lines, problems))
    }

    /// The requests the page made (whose address holds `filter`, when given).
    pub async fn network(&self, filter: Option<&str>) -> Result<String, String> {
        let live = self.agent_page().await?;
        let requests = lock(&live.events.network);
        Ok(network_text(&requests, filter))
    }

    /// The text of the response to a request the page made, by its id.
    pub async fn response_text(&self, id: &str) -> Result<String, String> {
        let live = self.agent_page().await?;
        if !lock(&live.events.network).iter().any(|r| r.id == id) {
            return Err("No request kept has that id: list them with browser_network first.".to_owned());
        }
        let session = live.session();
        let body = live
            .cdp
            .call("Network.getResponseBody", json!({"requestId": id}), session.as_deref(), super::session::CALL)
            .await
            .map_err(|why| format!("Its response could not be read: {why}"))?;
        if body.get("base64Encoded").and_then(Value::as_bool) == Some(true) {
            let n = body.get("body").and_then(Value::as_str).map_or(0, |b| b.len() * 3 / 4);
            return Ok(format!("The response is not text (about {n} bytes)."));
        }
        let text = body.get("body").and_then(Value::as_str).unwrap_or("");
        Ok(secrets::redact(&cut(text, MAX_TEXT)))
    }

    /// The visible elements whose words hold `query`, each marked with a
    /// reference and its point in the screenshot's coordinates.
    pub async fn find(&self, query: &str) -> Result<Vec<Found>, String> {
        let query = query.trim();
        if query.is_empty() || query.chars().count() > 200 {
            return Err("Say what to find, in at most 200 characters.".to_owned());
        }
        let live = self.agent_page().await?;
        let expression = format!("{FIND_JS}({})", serde_json::to_string(query).unwrap_or_default());
        evaluate(&live, &expression).await.map_err(|why| format!("The page could not be searched: {why}"))
    }

    /// Fill the element `reference` ([`Self::find`]'s) with `value`; never a
    /// password or card field.
    pub async fn fill(&self, reference: &str, value: &str) -> Result<Shot, String> {
        if value.chars().count() > super::session::MAX_TYPE {
            return Err("That is more text than one action types (4,000 characters).".to_owned());
        }
        let live = self.agent_page().await?;
        let expression = format!(
            "{FILL_JS}({}, {})",
            serde_json::to_string(reference).unwrap_or_default(),
            serde_json::to_string(value).unwrap_or_default()
        );
        let filled: Filled = evaluate(&live, &expression).await.unwrap_or_default();
        match filled.kind.as_str() {
            "ok" => self.after(&live).await,
            "password" => Err("That is a password field: the agent never types a password. Sign in yourself in the agent's browser.".to_owned()),
            "card" => Err("That is a payment card field: the agent never types card details. Fill it in yourself.".to_owned()),
            "gone" => Err("No element has that reference now: find it again with browser_find.".to_owned()),
            "no-option" => Err("The list has no option with that text or value.".to_owned()),
            _ => Err("That element is not a field, a list or a box.".to_owned()),
        }
    }

    /// Set the page's size, and light or dark (`None`: as the system says).
    pub async fn set_viewport(&self, width: u32, height: u32, dark: Option<bool>) -> Result<Shot, String> {
        if !(MIN_SIZE.0..=MAX_SIZE.0).contains(&width) || !(MIN_SIZE.1..=MAX_SIZE.1).contains(&height) {
            return Err(format!(
                "A page is {} to {} pixels wide and {} to {} high.",
                MIN_SIZE.0, MAX_SIZE.0, MIN_SIZE.1, MAX_SIZE.1
            ));
        }
        let live = self.agent_page().await?;
        let viewport = Viewport { width, height, dark };
        let session = live.session();
        apply_viewport(&live, session.as_deref(), viewport).await?;
        *lock(&live.viewport) = viewport;
        self.after(&live).await
    }
}

/// What `browser_tabs` does.
#[derive(Clone, Debug, PartialEq)]
pub enum TabAct {
    List,
    /// Open a new page at an address (checked as `open` checks) and work in it.
    New(String),
    /// Work in the page at this place in the list, from 1.
    Select(usize),
    /// Close the page at this place in the list.
    Close(usize),
}

impl AgentBrowser {
    /// The browser's pages, and what was done to them: each with its place,
    /// title and address, the agent's marked.
    pub async fn tabs(&self, act: TabAct) -> Result<String, String> {
        let live = self.agent_page().await?;
        let mut did = String::new();
        match &act {
            TabAct::List => {}
            TabAct::New(url) => {
                let url = self.check(url)?;
                let made = live.cdp.call("Target.createTarget", json!({"url": url}), None, super::session::CALL).await?;
                let id = made.get("targetId").and_then(Value::as_str).ok_or("The browser opened no page.")?.to_owned();
                super::session::attach(&live, &id).await?;
                did = "Opened a new tab; the agent works in it now.".to_owned();
            }
            TabAct::Select(n) | TabAct::Close(n) => {
                let pages = super::session::pages(&live).await?;
                let Some((id, title, _)) = n.checked_sub(1).and_then(|i| pages.get(i)).cloned() else {
                    return Err(format!("There is no tab {n}: the browser has {}.", pages.len()));
                };
                if let TabAct::Select(_) = act {
                    super::session::attach(&live, &id).await?;
                    did = format!("The agent works in tab {n} (\u{201c}{title}\u{201d}) now.");
                } else {
                    if pages.len() == 1 {
                        return Err("That is the browser's only tab: it stays open.".to_owned());
                    }
                    let current = super::session::current_target(&live);
                    live.cdp.call("Target.closeTarget", json!({"targetId": id}), None, super::session::CALL).await?;
                    if current.as_deref() == Some(id.as_str()) {
                        let rest = super::session::pages(&live).await?;
                        if let Some((next, _, _)) = rest.first() {
                            super::session::attach(&live, next).await?;
                        }
                    }
                    did = format!("Closed tab {n} (\u{201c}{title}\u{201d}).");
                }
            }
        }
        let pages = super::session::pages(&live).await?;
        let current = super::session::current_target(&live);
        let mut out = if did.is_empty() { String::new() } else { format!("{did}\n") };
        out.push_str("Tabs:");
        for (i, (id, title, url)) in pages.iter().enumerate() {
            let mine = if current.as_deref() == Some(id.as_str()) { " (the agent's)" } else { "" };
            out.push_str(&format!("\n{}. {}{mine} \u{2014} {}", i + 1, if title.is_empty() { "untitled" } else { title }, url));
        }
        Ok(out)
    }

    /// Put the file at `path` (its final path, checked by the caller) in the
    /// file field `reference` ([`Self::find`]'s).
    pub async fn upload(&self, reference: &str, path: &std::path::Path) -> Result<Shot, String> {
        let live = self.agent_page().await?;
        let session = live.session();
        let expression = format!(
            "document.querySelector('[data-lattice-ref=\"' + {} + '\"]')",
            serde_json::to_string(reference).unwrap_or_default()
        );
        let found = live
            .cdp
            .call("Runtime.evaluate", json!({"expression": expression}), session.as_deref(), super::session::CALL)
            .await?;
        let result = found.get("result");
        let Some(object) = result.and_then(|r| r.get("objectId")).and_then(Value::as_str) else {
            return Err("No element has that reference now: find it again with browser_find.".to_owned());
        };
        let is_file = evaluate::<bool>(
            &live,
            &format!(
                "(() => {{ const el = document.querySelector('[data-lattice-ref=\"' + {} + '\"]'); return !!el && el.tagName === 'INPUT' && (el.getAttribute('type') || '').toLowerCase() === 'file'; }})()",
                serde_json::to_string(reference).unwrap_or_default()
            ),
        )
        .await
        .unwrap_or(false);
        if !is_file {
            return Err("That element is not a file field.".to_owned());
        }
        // A file name the browser reads plainly: no `\\?\` prefix.
        let plain = path.to_string_lossy().trim_start_matches(r"\\?\").to_owned();
        live.cdp
            .call("DOM.setFileInputFiles", json!({"files": [plain], "objectId": object}), session.as_deref(), super::session::CALL)
            .await
            .map_err(|why| format!("The file could not be put in the field: {why}"))?;
        self.after(&live).await
    }
}
