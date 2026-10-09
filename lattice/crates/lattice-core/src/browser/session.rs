//! [`AgentBrowser`]: the agent's own browser, its page, and what the agent
//! does in it, as a person does: it looks at the page (a screenshot), clicks
//! at a point, types, presses keys, scrolls and goes back. Not a port.
//!
//! - **Started lazily**, at the first action (or the reader's Show), and
//!   stopped by the reader or when Lattice closes (the Job ends the tree).
//! - **One page at a time**: the first page target, attached through
//!   `Target.attachToTarget` (flat sessions); a page another one opened (a new
//!   tab) becomes the page the agent works in.
//! - **The viewport** is fixed (`Emulation.setDeviceMetricsOverride`,
//!   1280 x 800, scale 1), so a screenshot's pixels are the page's coordinates
//!   and a click lands where the model saw the thing it clicks.
//! - **Every action ends with a screenshot** (PNG), the page's address and
//!   its title, once the page has settled: a short pause, then until no
//!   frame is loading and a navigation of the page (one Lattice asked for,
//!   or one the page asked for) has loaded, at most 15 s. What the model
//!   sees next.
//! - **Guards**: an address passes [`super::policy::check_url`]; downloads are
//!   denied (`Browser.setDownloadBehavior`); a JavaScript dialog is dismissed
//!   and named in the next result; typing is refused into a password field,
//!   a card field, or a field inside another site's frame (the reader signs
//!   in and pays themselves). The guard reads the focused element with one
//!   fixed expression; the agent itself has no way to run script.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use lattice_agents::model::Image;
use serde_json::{Value, json};
use tokio::runtime::Handle;
use tokio::sync::watch;

use super::cdp::{Cdp, Endpoint, Event};
use super::launch;
use super::policy::{self, Effect, Press, Target};
use crate::env::Env;
use crate::state::StateRoot;

/// The page's size, in CSS pixels (= screenshot pixels).
pub const VIEWPORT: (u32, u32) = (1280, 800);
/// How long one DevTools request waits.
const CALL: Duration = Duration::from_secs(30);
/// How long an action waits for its page to stop loading.
pub const LOAD_WAIT: Duration = Duration::from_secs(15);
/// The pause after an action before the page is looked at.
const SETTLE: Duration = Duration::from_millis(350);
/// The longest text typed in one action.
pub const MAX_TYPE: usize = 4000;

/// The page's address, title and visible text, read by a fixed expression
/// (never the agent's script).
const PAGE_TEXT_JS: &str = r#"(() => ({
  url: location.href,
  title: document.title,
  text: document.body ? document.body.innerText : ''
}))()"#;

/// The web results of Bing's results page (`li.b_algo`; its ads are other
/// elements), read by a fixed expression.
const RESULTS_JS: &str = r#"(() => Array.from(document.querySelectorAll('#b_results > li.b_algo'))
  .map(r => {
    const a = r.querySelector('h2 a');
    const s = r.querySelector('.b_caption p, p[class^="b_lineclamp"], .b_snippet, .b_algoSlug');
    const words = e => e ? (e.innerText || e.textContent || '').replace(/\s+/g, ' ').trim() : '';
    return { title: words(a), url: a ? a.href : '', snippet: words(s) };
  })
  .filter(h => h.url))()"#;

/// Where `web_search` searches: Bing. Measured on 2026-10-09: DuckDuckGo's
/// pages and Brave Search answered the agent's browser with a challenge for
/// humans (which the agent never answers); Bing answered with results.
pub const SEARCH_PAGE: &str = "https://www.bing.com/search";

/// The most results a search gives.
pub const MAX_HITS: usize = 10;

/// Unpadded (or padded) base64url's bytes; `None` for any other character.
fn base64url_decode(text: &str) -> Option<Vec<u8>> {
    let mut bits = 0u32;
    let mut held = 0u8;
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    for c in text.trim_end_matches('=').bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'-' | b'+' => 62,
            b'_' | b'/' => 63,
            _ => return None,
        };
        bits = (bits << 6) | u32::from(v);
        held += 6;
        if held >= 8 {
            held -= 8;
            out.push((bits >> held) as u8);
        }
    }
    Some(out)
}

/// What a page says, as `browser_read` gives it.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Deserialize)]
pub struct PageText {
    pub url: String,
    pub title: String,
    pub text: String,
}

/// One result of a web search.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Deserialize)]
pub struct Hit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// A result's own address: Bing's redirect (`/ck/a?...&u=a1<base64url of
/// the address>`) unwrapped; `None` for anything that is not a web address.
pub fn landing(url: &str) -> Option<String> {
    let parsed = url::Url::parse(url).ok()?;
    let host = parsed.host_str().unwrap_or("");
    let own = if (host == "bing.com" || host.ends_with(".bing.com")) && parsed.path() == "/ck/a" {
        let u = parsed.query_pairs().find(|(k, _)| k == "u").map(|(_, v)| v.into_owned())?;
        let bytes = base64url_decode(u.strip_prefix("a1")?)?;
        String::from_utf8(bytes).ok()?
    } else {
        url.to_owned()
    };
    let own = url::Url::parse(&own).ok()?;
    matches!(own.scheme(), "http" | "https").then(|| own.to_string())
}

/// The focused element, read before typing and before Enter or Space (a
/// [`policy::Target`]): its `kind` is `none`, `ok`, `password`, `card` or
/// `frame` (inside another site's frame).
const FOCUS_JS: &str = r#"(() => {
  let el = document.activeElement; let depth = 0;
  while (el && depth < 8) {
    if (el.tagName === 'IFRAME' || el.tagName === 'FRAME') {
      let inner = null;
      try { inner = el.contentDocument; } catch (e) { inner = null; }
      if (!inner || !inner.activeElement) return {kind: 'frame', frame: true};
      el = inner.activeElement;
    } else if (el.shadowRoot && el.shadowRoot.activeElement) {
      el = el.shadowRoot.activeElement;
    } else break;
    depth++;
  }
  if (!el || el === el.ownerDocument.body || el === el.ownerDocument.documentElement) return {kind: 'none'};
  const attr = (name) => el.getAttribute(name) || '';
  const pick = (v) => (v || '').toString().replace(/\s+/g, ' ').trim().slice(0, 200);
  const type = attr('type').toLowerCase();
  const auto = attr('autocomplete').toLowerCase();
  if (type === 'password' || auto.includes('password')) return {kind: 'password'};
  const names = (attr('name') + ' ' + (el.id || '')).toLowerCase();
  if (auto.startsWith('cc-') || /card.?num|cvc|cvv|csc|security.?code/.test(names)) return {kind: 'card'};
  const role = attr('role').toLowerCase();
  const said = (attr('aria-label') + ' ' + attr('placeholder')).toLowerCase();
  const search = type === 'search' || role === 'searchbox' || !!el.closest('[role="search"], search')
    || /(^|[^a-z0-9])(q|query|search|search_query|field-keywords|keywords)([^a-z0-9]|$)/.test(names)
    || said.includes('search');
  const editor = el.tagName === 'TEXTAREA' || el.isContentEditable;
  let submit = '';
  const form = el.form || el.closest('form');
  if (form) {
    const button = form.querySelector('button[type="submit"], input[type="submit"], button:not([type])');
    if (button) submit = pick(button.getAttribute('aria-label') || button.innerText || button.value);
  }
  return {kind: 'ok', tag: el.tagName.toLowerCase(), actionable: true,
    label: pick(attr('aria-label') || attr('placeholder') || (editor ? '' : el.innerText) || attr('title')),
    testid: pick(attr('data-testid')), editor, search, submit};
})()"#;

/// What is at a point of the page (a [`policy::Target`]), read before a
/// click stated as `view` or `edit`: called with the point's coordinates.
const TARGET_JS: &str = r#"((x, y) => {
  let doc = document; let el = null; let depth = 0;
  while (depth < 8) {
    el = doc.elementFromPoint(x, y);
    if (!el) return {};
    while (el.shadowRoot) {
      const inner = el.shadowRoot.elementFromPoint(x, y);
      if (!inner || inner === el) break;
      el = inner;
    }
    if (el.tagName !== 'IFRAME' && el.tagName !== 'FRAME') break;
    let inner = null;
    try { inner = el.contentDocument; } catch (e) { inner = null; }
    if (!inner) return {frame: true, tag: 'iframe'};
    const box = el.getBoundingClientRect();
    x -= box.left; y -= box.top; doc = inner; depth++;
  }
  const pick = (v) => (v || '').toString().replace(/\s+/g, ' ').trim().slice(0, 200);
  const hit = el.closest('button, a[href], input, select, summary, label, [role="button"], [role="link"], [role="menuitem"], [role="menuitemradio"], [role="menuitemcheckbox"], [role="tab"], [role="option"], [role="checkbox"], [role="radio"], [role="switch"]');
  const at = hit || el;
  let label = pick(at.getAttribute('aria-label')) || pick(at.innerText) || pick(at.value) || pick(at.getAttribute('title'));
  if (!label && at.querySelector) {
    const art = at.querySelector('img[alt], svg[aria-label], [aria-label]');
    if (art) label = pick(art.getAttribute('alt') || art.getAttribute('aria-label'));
  }
  return {kind: 'ok', tag: at.tagName.toLowerCase(), actionable: !!hit, label,
    testid: pick(at.getAttribute('data-testid'))};
})"#;

/// What the agent sees after an action.
#[derive(Clone, Debug, PartialEq)]
pub struct Shot {
    pub url: String,
    pub title: String,
    pub image: Image,
    /// Something the reader and the model should know (a dialog dismissed, a
    /// new tab followed, a download refused).
    pub notes: Vec<String>,
}

impl Shot {
    /// The words a tool result gives with it.
    pub fn summary(&self) -> String {
        let mut text = format!(
            "The page is {} ({}). A screenshot ({} x {} pixels, the page's coordinates) comes with the next message.",
            if self.title.is_empty() {
                "untitled"
            } else {
                &self.title
            },
            self.url,
            VIEWPORT.0,
            VIEWPORT.1
        );
        for note in &self.notes {
            text.push(' ');
            text.push_str(note);
        }
        text
    }
}

/// The browser's state, as the window shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BrowserStatus {
    Stopped,
    Starting,
    Running { url: String, title: String },
    Failed(String),
}

/// How the browser is started (tests change it).
#[derive(Clone, Debug)]
pub struct BrowserConfig {
    pub headless: bool,
    /// Open loopback and private addresses too: tests only, for a page they
    /// serve themselves.
    pub allow_local: bool,
    /// Another profile folder (tests).
    pub profile: Option<std::path::PathBuf>,
    /// Another program (tests); else Edge or Chrome where installed.
    pub program: Option<std::path::PathBuf>,
    /// How long the browser has to answer its first DevTools request:
    /// [`launch::START_WAIT`], else longer for tests, whose every start is a
    /// fresh profile's first.
    pub start_wait: Duration,
    /// How long an action waits for its page to stop loading: [`LOAD_WAIT`],
    /// else longer for tests on a loaded machine.
    pub load_wait: Duration,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            headless: false,
            allow_local: false,
            profile: None,
            program: None,
            start_wait: launch::START_WAIT,
            load_wait: LOAD_WAIT,
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// What the page's events tell the actions.
#[derive(Default)]
struct PageEvents {
    /// Frames loading now (start minus stop).
    loading: AtomicU32,
    /// The page's own frame (a page target's main frame has its id).
    main_frame: Mutex<String>,
    /// A navigation of the page is under way: one Lattice asked for (open,
    /// back) or the page asked for (a link, a form, a script), until its load
    /// event, a navigation within the document, or a restore from the
    /// back-forward cache.
    navigating: AtomicBool,
    notes: Mutex<Vec<String>>,
    /// A page another one opened, waiting to be followed.
    opened: Mutex<Option<String>>,
}

impl PageEvents {
    fn is_main(&self, params: &Value) -> bool {
        let frame = params
            .get("frameId")
            .or_else(|| params.get("frame").and_then(|frame| frame.get("id")))
            .and_then(Value::as_str);
        frame.is_some_and(|frame| *lock(&self.main_frame) == frame)
    }
}

/// A running browser.
struct Live {
    child: Mutex<Option<lattice_sys::process::Child>>,
    cdp: Arc<Cdp>,
    /// The page target and its session.
    page: Mutex<Option<(String, String)>>,
    events: Arc<PageEvents>,
}

impl Live {
    fn session(&self) -> Option<String> {
        lock(&self.page)
            .as_ref()
            .map(|(_, session)| session.clone())
    }
}

/// The agent's browser.
pub struct AgentBrowser {
    state: StateRoot,
    env: Arc<dyn Env>,
    handle: Handle,
    config: BrowserConfig,
    live: tokio::sync::Mutex<Option<Arc<Live>>>,
    status: Mutex<BrowserStatus>,
    changed: Arc<watch::Sender<u64>>,
}

fn on_event(events: &PageEvents, event: Event) {
    match event.method.as_str() {
        "Page.frameStartedLoading" => {
            events.loading.fetch_add(1, Ordering::SeqCst);
        }
        "Page.frameStoppedLoading" => {
            let _ = events
                .loading
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                    Some(n.saturating_sub(1))
                });
        }
        "Page.frameRequestedNavigation" if events.is_main(&event.params) => {
            events.navigating.store(true, Ordering::SeqCst);
        }
        "Page.loadEventFired" => events.navigating.store(false, Ordering::SeqCst),
        "Page.navigatedWithinDocument" if events.is_main(&event.params) => {
            events.navigating.store(false, Ordering::SeqCst);
        }
        "Page.frameNavigated"
            if events.is_main(&event.params)
                && event.params.get("type").and_then(Value::as_str)
                    == Some("BackForwardCacheRestore") =>
        {
            events.navigating.store(false, Ordering::SeqCst);
        }
        "Page.javascriptDialogOpening" => {
            let message = event
                .params
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("");
            let shown: String = message.chars().take(200).collect();
            lock(&events.notes).push(format!(
                "The page showed a dialog (\"{shown}\"); it was dismissed."
            ));
        }
        "Target.targetCreated" => {
            let info = event.params.get("targetInfo");
            let is_page = info.and_then(|i| i.get("type")).and_then(Value::as_str) == Some("page");
            let opened_by_page = info
                .and_then(|i| i.get("openerId"))
                .and_then(Value::as_str)
                .is_some();
            if is_page
                && opened_by_page
                && let Some(id) = info.and_then(|i| i.get("targetId")).and_then(Value::as_str)
            {
                *lock(&events.opened) = Some(id.to_owned());
            }
        }
        "Browser.downloadWillBegin" | "Page.downloadWillBegin" => {
            lock(&events.notes)
                .push("The page tried to download a file; downloads are refused.".to_owned());
        }
        _ => {}
    }
}

impl AgentBrowser {
    pub fn new(state: StateRoot, env: Arc<dyn Env>, handle: Handle) -> Self {
        let (changed, _) = watch::channel(0);
        Self {
            state,
            env,
            handle,
            config: BrowserConfig::default(),
            live: tokio::sync::Mutex::new(None),
            status: Mutex::new(BrowserStatus::Stopped),
            changed: Arc::new(changed),
        }
    }

    /// The same browser started another way (tests).
    pub fn with_config(mut self, config: BrowserConfig) -> Self {
        self.config = config;
        self
    }

    /// The profile folder its sign-ins live in.
    pub fn profile(&self) -> std::path::PathBuf {
        self.config
            .profile
            .clone()
            .unwrap_or_else(|| launch::profile_dir(&self.state))
    }

    pub fn status(&self) -> BrowserStatus {
        lock(&self.status).clone()
    }

    /// The running browser's processes (its Job's list), or none (tests).
    #[cfg(test)]
    pub(crate) fn process_ids(&self) -> Vec<u32> {
        let Ok(slot) = self.live.try_lock() else {
            return Vec::new();
        };
        slot.as_ref()
            .and_then(|live| {
                lock(&live.child)
                    .as_ref()
                    .and_then(|child| child.process_ids().ok())
            })
            .unwrap_or_default()
    }

    /// Bumped at every change of state.
    pub fn changed(&self) -> watch::Receiver<u64> {
        self.changed.subscribe()
    }

    fn set_status(&self, status: BrowserStatus) {
        *lock(&self.status) = status;
        self.changed.send_modify(|n| *n = n.wrapping_add(1));
    }

    fn check(&self, url: &str) -> Result<String, String> {
        if self.config.allow_local {
            return url::Url::parse(url)
                .map(|url| url.to_string())
                .map_err(|_| "That is not an address.".to_owned());
        }
        policy::check_url(url)
    }

    /// The running browser, started now if it is not.
    async fn live(&self) -> Result<Arc<Live>, String> {
        let mut slot = self.live.lock().await;
        if let Some(live) = slot.as_ref()
            && live.cdp.closed().is_none()
        {
            return Ok(live.clone());
        }
        if let Some(old) = slot.take() {
            drop(lock(&old.child).take());
        }
        self.set_status(BrowserStatus::Starting);
        match self.start().await {
            Ok(live) => {
                *slot = Some(live.clone());
                self.set_status(BrowserStatus::Running {
                    url: "about:blank".to_owned(),
                    title: String::new(),
                });
                Ok(live)
            }
            Err(why) => {
                self.set_status(BrowserStatus::Failed(why.clone()));
                Err(why)
            }
        }
    }

    /// The running browser for an action of the agent's, its page brought to
    /// the front of its window first: a tab the reader opened (`open_tab`)
    /// may cover it, and the agent acts on, and looks at, its own page.
    async fn agent_page(&self) -> Result<Arc<Live>, String> {
        let live = self.live().await?;
        let target = lock(&live.page).as_ref().map(|(target, _)| target.clone());
        if let Some(target) = target {
            let _ = live
                .cdp
                .call(
                    "Target.activateTarget",
                    json!({"targetId": target}),
                    None,
                    CALL,
                )
                .await;
        }
        Ok(live)
    }

    async fn start(&self) -> Result<Arc<Live>, String> {
        let program = match &self.config.program {
            Some(program) => program.clone(),
            None => launch::find_browser(self.env.as_ref()).ok_or_else(|| {
                "Neither Microsoft Edge nor Google Chrome is installed where its installer puts it.".to_owned()
            })?,
        };
        let (profile, headless, env, globals) = (
            self.profile(),
            self.config.headless,
            self.env.clone(),
            self.state.globals.clone(),
        );
        let started = self
            .handle
            .spawn_blocking(move || {
                launch::start(
                    &program,
                    &profile,
                    headless,
                    VIEWPORT,
                    env.as_ref(),
                    &globals,
                )
            })
            .await
            .map_err(|_| "The browser could not start.".to_owned())??;
        let launch::Started { child, pipes, .. } = started;
        let events = Arc::new(PageEvents::default());
        let seen = events.clone();
        let cdp = Cdp::connect(
            Endpoint::Pipe(pipes),
            Arc::new(move |event| on_event(&seen, event)),
        )?;
        // The first answer says the browser has started; a copy that ends at
        // once (another holds the profile) closes its pipe instead.
        let wait = self.config.start_wait;
        if let Err(why) = cdp
            .call(
                "Target.setDiscoverTargets",
                json!({"discover": true}),
                None,
                wait,
            )
            .await
        {
            let open = cdp.closed().is_none();
            drop(cdp);
            return Err(self
                .handle
                .spawn_blocking(move || launch::start_failure(child, &why, open, wait))
                .await
                .unwrap_or_else(|_| "The browser could not start.".to_owned()));
        }
        let _ = cdp
            .call(
                "Browser.setDownloadBehavior",
                json!({"behavior": "deny"}),
                None,
                CALL,
            )
            .await;
        let live = Arc::new(Live {
            child: Mutex::new(Some(child)),
            cdp,
            page: Mutex::new(None),
            events,
        });
        let target = first_page(&live.cdp).await?;
        attach(&live, &target).await?;
        Ok(live)
    }

    /// Stop the browser (the Job ends its tree).
    pub async fn stop(&self) {
        let live = self.live.lock().await.take();
        if let Some(live) = live {
            live.cdp.close();
            let child = lock(&live.child).take();
            let _ = self
                .handle
                .spawn_blocking(move || {
                    if let Some(child) = child {
                        let _ = child.wait(Some(Duration::from_secs(2)));
                    }
                })
                .await;
        }
        self.set_status(BrowserStatus::Stopped);
    }

    /// Start it (the reader's Show), at `url` when given.
    pub async fn show(&self, url: Option<&str>) -> Result<Shot, String> {
        match url {
            Some(url) => self.open(url).await,
            None => self.look().await,
        }
    }

    /// Open an address in a tab of its own, in front (the reader's own
    /// action: a link of Alelyon's account panel), the browser started first
    /// if it is not. The agent keeps its page: a tab Lattice opens has no
    /// opener, so the session does not follow it. The address passes the same
    /// policy as the agent's.
    pub async fn open_tab(&self, url: &str) -> Result<(), String> {
        let url = self.check(url)?;
        let live = self.live().await?;
        let created = live
            .cdp
            .call("Target.createTarget", json!({"url": url}), None, CALL)
            .await
            .map_err(|why| format!("The page could not be opened: {why}"))?;
        let target = created
            .get("targetId")
            .and_then(Value::as_str)
            .ok_or_else(|| "The browser opened no tab.".to_owned())?;
        let _ = live
            .cdp
            .call(
                "Target.activateTarget",
                json!({"targetId": target}),
                None,
                CALL,
            )
            .await;
        Ok(())
    }

    /// The agent's page's `document.visibilityState`, read without bringing
    /// it to the front (tests).
    #[cfg(test)]
    pub(crate) async fn agent_page_visibility(&self) -> String {
        let Some(live) = self.live.lock().await.clone() else {
            return String::new();
        };
        let session = live.session();
        live.cdp
            .call(
                "Runtime.evaluate",
                json!({"expression": "document.visibilityState", "returnByValue": true}),
                session.as_deref(),
                CALL,
            )
            .await
            .ok()
            .and_then(|value| {
                value
                    .get("result")
                    .and_then(|result| result.get("value"))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .unwrap_or_default()
    }

    /// The addresses of the browser's pages, in its order (tests).
    #[cfg(test)]
    pub(crate) async fn page_urls(&self) -> Vec<String> {
        let Some(live) = self.live.lock().await.clone() else {
            return Vec::new();
        };
        let targets = live
            .cdp
            .call("Target.getTargets", json!({}), None, CALL)
            .await
            .unwrap_or(Value::Null);
        targets
            .get("targetInfos")
            .and_then(Value::as_array)
            .map(|all| {
                all.iter()
                    .filter(|t| t.get("type").and_then(Value::as_str) == Some("page"))
                    .filter_map(|t| t.get("url").and_then(Value::as_str).map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Open an address in the page.
    pub async fn open(&self, url: &str) -> Result<Shot, String> {
        let url = self.check(url)?;
        let live = self.agent_page().await?;
        let session = live.session();
        live.events.navigating.store(true, Ordering::SeqCst);
        live.cdp
            .call(
                "Page.navigate",
                json!({"url": url}),
                session.as_deref(),
                CALL,
            )
            .await
            .map_err(|why| format!("The page could not be opened: {why}"))?;
        self.after(&live).await
    }

    /// The page's address, title and visible text.
    pub async fn page_text(&self) -> Result<PageText, String> {
        let live = self.agent_page().await?;
        evaluate(&live, PAGE_TEXT_JS)
            .await
            .map_err(|why| format!("The page's text could not be read: {why}"))
    }

    /// Search the web in the page ([`SEARCH_PAGE`]): its results, at most [`MAX_HITS`].
    pub async fn search(&self, query: &str) -> Result<Vec<Hit>, String> {
        self.search_at(SEARCH_PAGE, query).await
    }

    /// [`Self::search`] on a results page of the same shape at `page` (a test's).
    pub async fn search_at(&self, page: &str, query: &str) -> Result<Vec<Hit>, String> {
        let query = query.trim();
        if query.is_empty() {
            return Err("Say what to search for.".to_owned());
        }
        if query.chars().count() > 500 {
            return Err("A search is at most 500 characters.".to_owned());
        }
        let mut url = url::Url::parse(page).map_err(|_| "That is not an address.".to_owned())?;
        url.query_pairs_mut().append_pair("q", query);
        self.open(url.as_str()).await?;
        let live = self.agent_page().await?;
        let hits: Vec<Hit> = evaluate(&live, RESULTS_JS)
            .await
            .map_err(|why| format!("The results could not be read: {why}"))?;
        Ok(hits
            .into_iter()
            .filter_map(|hit| {
                Some(Hit {
                    url: landing(&hit.url)?,
                    ..hit
                })
            })
            .take(MAX_HITS)
            .collect())
    }

    /// Look at the page as it is.
    pub async fn look(&self) -> Result<Shot, String> {
        let live = self.agent_page().await?;
        self.after(&live).await
    }

    /// Click at a point of the page, in its coordinates. A click stated as
    /// `view` or `edit` whose target looks like an action that asks is
    /// refused ([`policy::click_looks`]).
    pub async fn click(
        &self,
        x: f64,
        y: f64,
        double: bool,
        stated: Effect,
    ) -> Result<Shot, String> {
        let (width, height) = (f64::from(VIEWPORT.0), f64::from(VIEWPORT.1));
        if !(0.0..width).contains(&x) || !(0.0..height).contains(&y) {
            return Err(format!(
                "({x}, {y}) is outside the page, which is {} x {} pixels.",
                VIEWPORT.0, VIEWPORT.1
            ));
        }
        let live = self.agent_page().await?;
        if !stated.asks() {
            // Both are finite numbers inside the page: the expression is fixed.
            let target = read_target(&live, format!("{TARGET_JS}({x:.1}, {y:.1})")).await?;
            if let Some(looks) = policy::click_looks(&target) {
                return Err(policy::misstated(looks, &target));
            }
        }
        let session = live.session();
        let s = session.as_deref();
        live.cdp
            .call(
                "Input.dispatchMouseEvent",
                json!({"type": "mouseMoved", "x": x, "y": y}),
                s,
                CALL,
            )
            .await?;
        let clicks = if double { 2 } else { 1 };
        for count in 1..=clicks {
            for kind in ["mousePressed", "mouseReleased"] {
                live.cdp
                    .call(
                        "Input.dispatchMouseEvent",
                        json!({"type": kind, "x": x, "y": y, "button": "left", "clickCount": count}),
                        s,
                        CALL,
                    )
                    .await?;
            }
        }
        self.after(&live).await
    }

    /// Type text into the focused field, unless it is one the reader fills in
    /// themselves.
    pub async fn type_text(&self, text: &str) -> Result<Shot, String> {
        if text.is_empty() {
            return Err("Give the text to type.".to_owned());
        }
        if text.chars().count() > MAX_TYPE {
            return Err("That is more text than one action types (4,000 characters).".to_owned());
        }
        let live = self.agent_page().await?;
        let session = live.session();
        let s = session.as_deref();
        let focus = read_target(&live, FOCUS_JS.to_owned()).await?;
        match focus.kind.as_str() {
            "password" => {
                return Err("That is a password field: the agent never types a password. Sign in yourself in the agent's browser.".to_owned());
            }
            "card" => {
                return Err("That is a payment card field: the agent never types card details. Fill it in yourself.".to_owned());
            }
            "frame" => {
                return Err("The focused field is inside another site's frame, which the agent does not type into. Fill it in yourself.".to_owned());
            }
            "ok" => {}
            _ => {
                return Err("No field has the focus: click the field first.".to_owned());
            }
        }
        live.cdp
            .call("Input.insertText", json!({"text": text}), s, CALL)
            .await?;
        self.after(&live).await
    }

    /// Press one key, with its modifiers. Enter or Space stated as `view` or
    /// `edit` that looks like sending or an action that asks is refused
    /// ([`policy::press_looks`]).
    pub async fn press(&self, press: &Press, stated: Effect) -> Result<Shot, String> {
        let live = self.agent_page().await?;
        if !stated.asks() && matches!(press.key.as_str(), "Enter" | " ") {
            let focused = read_target(&live, FOCUS_JS.to_owned()).await?;
            if let Some(looks) = policy::press_looks(press, &focused) {
                return Err(policy::misstated(looks, &focused));
            }
        }
        let session = live.session();
        let s = session.as_deref();
        let mut down = json!({
            "type": if press.text.is_some() { "keyDown" } else { "rawKeyDown" },
            "key": press.key,
            "code": press.code,
            "windowsVirtualKeyCode": press.virtual_key,
            "nativeVirtualKeyCode": press.virtual_key,
            "modifiers": press.modifiers,
        });
        if let Some(text) = &press.text {
            down["text"] = json!(text);
            down["unmodifiedText"] = json!(text);
        }
        live.cdp
            .call("Input.dispatchKeyEvent", down, s, CALL)
            .await?;
        live.cdp
            .call(
                "Input.dispatchKeyEvent",
                json!({
                    "type": "keyUp",
                    "key": press.key,
                    "code": press.code,
                    "windowsVirtualKeyCode": press.virtual_key,
                    "nativeVirtualKeyCode": press.virtual_key,
                    "modifiers": press.modifiers,
                }),
                s,
                CALL,
            )
            .await?;
        self.after(&live).await
    }

    /// Scroll the page at a point by whole pixels (down and right positive).
    pub async fn scroll(&self, x: f64, y: f64, dx: f64, dy: f64) -> Result<Shot, String> {
        let live = self.agent_page().await?;
        let session = live.session();
        live.cdp
            .call(
                "Input.dispatchMouseEvent",
                json!({"type": "mouseWheel", "x": x, "y": y, "deltaX": dx, "deltaY": dy}),
                session.as_deref(),
                CALL,
            )
            .await?;
        self.after(&live).await
    }

    /// Go back one page.
    pub async fn back(&self) -> Result<Shot, String> {
        let live = self.agent_page().await?;
        let session = live.session();
        let s = session.as_deref();
        let history = live
            .cdp
            .call("Page.getNavigationHistory", json!({}), s, CALL)
            .await?;
        let current = history
            .get("currentIndex")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        let entries = history
            .get("entries")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let Some(entry) = (current > 0)
            .then(|| entries.get((current - 1) as usize))
            .flatten()
        else {
            return Err("There is no page to go back to.".to_owned());
        };
        let id = entry.get("id").and_then(Value::as_i64).unwrap_or(0);
        live.events.navigating.store(true, Ordering::SeqCst);
        live.cdp
            .call(
                "Page.navigateToHistoryEntry",
                json!({"entryId": id}),
                s,
                CALL,
            )
            .await?;
        self.after(&live).await
    }

    /// After an action: follow a new tab, dismiss a dialog, wait for the
    /// page to settle, and look.
    async fn after(&self, live: &Arc<Live>) -> Result<Shot, String> {
        let mut notes = Vec::new();
        tokio::time::sleep(SETTLE).await;
        let opened = lock(&live.events.opened).take();
        if let Some(target) = opened {
            attach(live, &target).await?;
            notes.push("The page opened a new tab; the agent works in it now.".to_owned());
        }
        let session = live.session();
        let s = session.as_deref();
        // A dialog blocks the page until it is answered: dismiss it.
        let dialogs: Vec<String> = std::mem::take(&mut *lock(&live.events.notes));
        if dialogs
            .iter()
            .any(|note| note.starts_with("The page showed a dialog"))
        {
            let _ = live
                .cdp
                .call(
                    "Page.handleJavaScriptDialog",
                    json!({"accept": false}),
                    s,
                    CALL,
                )
                .await;
        }
        notes.extend(dialogs);
        let waited = std::time::Instant::now();
        while (live.events.loading.load(Ordering::SeqCst) > 0
            || live.events.navigating.load(Ordering::SeqCst))
            && waited.elapsed() < self.config.load_wait
        {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        // A navigation that never loaded (cancelled, a download refused)
        // does not hold up the next action.
        live.events.navigating.store(false, Ordering::SeqCst);
        let shot = live
            .cdp
            .call("Page.captureScreenshot", json!({"format": "png"}), s, CALL)
            .await
            .map_err(|why| format!("The page could not be looked at: {why}"))?;
        let data = shot
            .get("data")
            .and_then(Value::as_str)
            .ok_or_else(|| "The browser gave no screenshot.".to_owned())?
            .to_owned();
        let info = live
            .cdp
            .call("Target.getTargetInfo", json!({}), s, CALL)
            .await
            .unwrap_or(Value::Null);
        let field = |key: &str| {
            info.get("targetInfo")
                .and_then(|i| i.get(key))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned()
        };
        let (url, title) = (field("url"), field("title"));
        self.set_status(BrowserStatus::Running {
            url: url.clone(),
            title: title.clone(),
        });
        Ok(Shot {
            url,
            title,
            image: Image {
                media_type: "image/png".to_owned(),
                base64: data,
            },
            notes,
        })
    }
}

impl Drop for AgentBrowser {
    fn drop(&mut self) {
        if let Ok(mut slot) = self.live.try_lock()
            && let Some(live) = slot.take()
        {
            live.cdp.close();
            drop(lock(&live.child).take());
        }
    }
}

/// What one fixed expression says of the page (an empty target when it
/// throws or says nothing).
/// A fixed expression's value in the page, as `T`.
async fn evaluate<T: serde::de::DeserializeOwned>(live: &Live, expression: &str) -> Result<T, String> {
    let session = live.session();
    let value = live
        .cdp
        .call(
            "Runtime.evaluate",
            json!({"expression": expression, "returnByValue": true}),
            session.as_deref(),
            CALL,
        )
        .await?;
    let found = value
        .get("result")
        .and_then(|result| result.get("value"))
        .cloned()
        .unwrap_or(Value::Null);
    serde_json::from_value(found).map_err(|_| "the page did not answer as expected".to_owned())
}

async fn read_target(live: &Live, expression: String) -> Result<Target, String> {
    let session = live.session();
    let value = live
        .cdp
        .call(
            "Runtime.evaluate",
            json!({"expression": expression, "returnByValue": true}),
            session.as_deref(),
            CALL,
        )
        .await?;
    let found = value
        .get("result")
        .and_then(|result| result.get("value"))
        .cloned()
        .unwrap_or(Value::Null);
    Ok(serde_json::from_value(found).unwrap_or_default())
}

/// The browser's first page, made when it has none.
async fn first_page(cdp: &Cdp) -> Result<String, String> {
    let targets = cdp.call("Target.getTargets", json!({}), None, CALL).await?;
    let page = targets
        .get("targetInfos")
        .and_then(Value::as_array)
        .and_then(|all| {
            all.iter()
                .find(|t| t.get("type").and_then(Value::as_str) == Some("page"))
        })
        .and_then(|t| t.get("targetId"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    match page {
        Some(id) => Ok(id),
        None => cdp
            .call(
                "Target.createTarget",
                json!({"url": "about:blank"}),
                None,
                CALL,
            )
            .await?
            .get("targetId")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| "The browser opened no page.".to_owned()),
    }
}

/// Make `target` the page the agent works in.
async fn attach(live: &Live, target: &str) -> Result<(), String> {
    let attached = live
        .cdp
        .call(
            "Target.attachToTarget",
            json!({"targetId": target, "flatten": true}),
            None,
            CALL,
        )
        .await?;
    let session = attached
        .get("sessionId")
        .and_then(Value::as_str)
        .ok_or_else(|| "The browser gave no session for its page.".to_owned())?
        .to_owned();
    let s = Some(session.as_str());
    live.cdp.call("Page.enable", json!({}), s, CALL).await?;
    live.cdp
        .call(
            "Emulation.setDeviceMetricsOverride",
            json!({"width": VIEWPORT.0, "height": VIEWPORT.1, "deviceScaleFactor": 1, "mobile": false}),
            s,
            CALL,
        )
        .await?;
    let _ = live
        .cdp
        .call(
            "Target.activateTarget",
            json!({"targetId": target}),
            None,
            CALL,
        )
        .await;
    live.events.loading.store(0, Ordering::SeqCst);
    *lock(&live.events.main_frame) = target.to_owned();
    *lock(&live.page) = Some((target.to_owned(), session));
    Ok(())
}
