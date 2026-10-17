//! [`Desktop`]: the agent's hands on the whole desktop in auto mode. It looks
//! at one display (the primary, unless the agent chose another; a picture
//! scaled to fit 1280 x 800), clicks with any button at a point of that
//! picture, drags, types, presses and holds keys, scrolls, reads and fills the
//! clipboard and opens a program from the Start menu, as a person does,
//! through a [`Driver`] (the system's, `lattice_sys::desktop`; a test's own).
//!
//! - **Never on Alelyon, a password manager or Windows' prompts**
//!   ([`super::policy::protected`]): a click or a drag on such a window, or
//!   typing or a key press while one is in front, is refused before any input
//!   is sent; such a program is never opened.
//! - **Never into a password box**: typing is refused while the focus is a
//!   classic password box (the driver's `focus_is_password`). A password
//!   field another toolkit draws is not seen; the agent has no passwords to
//!   type, and says so in its instructions.
//! - **Every action ends with a picture** of the display, once it has had a
//!   moment to settle: what the model sees next. [`Desktop::zoom`] shows an
//!   area of it at the display's own pixels.
//! - **The stop hotkey** (Ctrl+Alt+End), armed while auto mode is on, stops
//!   every running agent turn ([`Desktop::arm_hotkey`]).

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use lattice_agents::model::Image;
pub use lattice_sys::desktop::{Button, Display};
use lattice_sys::desktop::{Screen, Window};
use tokio::runtime::Handle;

use super::policy::{self, Press};

/// The pause after an action before the screen is looked at.
const SETTLE: Duration = Duration::from_millis(600);
/// The longest text typed in one action.
pub const MAX_TYPE: usize = 4000;
/// The longest text put on the clipboard, and read from it.
pub const MAX_CLIPBOARD: usize = 100_000;
/// The longest name of a program to open.
pub const MAX_APP: usize = 100;
/// The pauses while a program is opened from the Start menu: after it opens,
/// after the name is typed, and after Enter, for the program to start.
const START_MENU: Duration = Duration::from_millis(800);
const PROGRAM_START: Duration = Duration::from_millis(1500);
/// `VK_LWIN` and `VK_RETURN`.
const WINDOWS_KEY: u16 = 0x5B;
const ENTER: u16 = 0x0D;
/// The stop hotkey: Ctrl+Alt+End.
pub const STOP_KEYS: &str = "Ctrl+Alt+End";

/// What the desktop is, to the agent: the system's, or a test's.
pub trait Driver: Send + Sync {
    /// The primary screen's size.
    fn screen_size(&self) -> Result<(u32, u32), String>;
    /// Every display, the primary first (empty: only the primary, at its size).
    fn displays(&self) -> Vec<Display>;
    /// The pixels of an area of the desktop.
    fn capture_area(&self, left: i32, top: i32, width: u32, height: u32) -> Result<Screen, String>;
    fn window_at(&self, x: i32, y: i32) -> Option<Window>;
    fn foreground(&self) -> Option<Window>;
    fn focus_is_password(&self) -> bool;
    /// Click `count` times (1 to 3) with `button` at a point of the desktop.
    fn press_at(&self, x: i32, y: i32, button: Button, count: u32) -> Result<(), String>;
    fn drag(&self, from: (i32, i32), to: (i32, i32)) -> Result<(), String>;
    fn type_text(&self, text: &str) -> Result<(), String>;
    fn key(&self, virtual_key: u16, modifiers: u32) -> Result<(), String>;
    fn hold_key(&self, virtual_key: u16, modifiers: u32, millis: u64) -> Result<(), String>;
    fn scroll(&self, x: i32, y: i32, down: i32, right: i32) -> Result<(), String>;
    fn clipboard_text(&self) -> Result<Option<String>, String>;
    fn set_clipboard_text(&self, text: &str) -> Result<(), String>;
}

/// The system's desktop (`lattice_sys::desktop`).
pub struct SystemDriver;

fn sentence(error: std::io::Error) -> String {
    let text = error.to_string();
    let mut chars = text.trim_end_matches('.').chars();
    match chars.next() {
        Some(first) => format!("{}{}.", first.to_uppercase(), chars.as_str()),
        None => "The desktop did not answer.".to_owned(),
    }
}

impl Driver for SystemDriver {
    fn screen_size(&self) -> Result<(u32, u32), String> {
        lattice_sys::desktop::screen_size().map_err(sentence)
    }
    fn displays(&self) -> Vec<Display> {
        lattice_sys::desktop::displays()
    }
    fn capture_area(&self, left: i32, top: i32, width: u32, height: u32) -> Result<Screen, String> {
        lattice_sys::desktop::capture_area(left, top, width, height).map_err(sentence)
    }
    fn window_at(&self, x: i32, y: i32) -> Option<Window> {
        lattice_sys::desktop::window_at(x, y)
    }
    fn foreground(&self) -> Option<Window> {
        lattice_sys::desktop::foreground()
    }
    fn focus_is_password(&self) -> bool {
        lattice_sys::desktop::focus_is_password()
    }
    fn press_at(&self, x: i32, y: i32, button: Button, count: u32) -> Result<(), String> {
        lattice_sys::desktop::press_at(x, y, button, count).map_err(sentence)
    }
    fn drag(&self, from: (i32, i32), to: (i32, i32)) -> Result<(), String> {
        lattice_sys::desktop::drag(from, to).map_err(sentence)
    }
    fn type_text(&self, text: &str) -> Result<(), String> {
        lattice_sys::desktop::type_text(text).map_err(sentence)
    }
    fn key(&self, virtual_key: u16, modifiers: u32) -> Result<(), String> {
        lattice_sys::desktop::key(virtual_key, modifiers).map_err(sentence)
    }
    fn hold_key(&self, virtual_key: u16, modifiers: u32, millis: u64) -> Result<(), String> {
        lattice_sys::desktop::hold_key(virtual_key, modifiers, millis).map_err(sentence)
    }
    fn scroll(&self, x: i32, y: i32, down: i32, right: i32) -> Result<(), String> {
        lattice_sys::desktop::scroll(x, y, down, right).map_err(sentence)
    }
    fn clipboard_text(&self) -> Result<Option<String>, String> {
        lattice_sys::desktop::clipboard_text().map_err(sentence)
    }
    fn set_clipboard_text(&self, text: &str) -> Result<(), String> {
        lattice_sys::desktop::set_clipboard_text(text).map_err(sentence)
    }
}

/// What the agent sees after an action.
#[derive(Clone, Debug, PartialEq)]
pub struct Look {
    /// The window in front, as its title names it.
    pub front: String,
    pub image: Image,
    pub width: u32,
    pub height: u32,
}

impl Look {
    /// The words a tool result gives with it.
    pub fn summary(&self) -> String {
        format!(
            "{} is in front. A picture of the screen ({} x {} pixels, the coordinates to click in) comes with the next message.",
            if self.front.is_empty() {
                "No window"
            } else {
                &self.front
            },
            self.width,
            self.height
        )
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The agent's hands on the whole desktop.
pub struct Desktop {
    driver: Arc<dyn Driver>,
    handle: Handle,
    /// This process: Alelyon itself.
    own: u32,
    /// The stop hotkey's modifiers and virtual key ([`STOP_KEYS`]).
    keys: (u32, u32),
    hotkey: Mutex<Option<lattice_sys::desktop::Hotkey>>,
    /// The display the agent works on, as [`Driver::displays`] lists them.
    display: Mutex<usize>,
}

fn front_title(window: Option<Window>) -> String {
    window
        .map(|window| {
            if window.title.trim().is_empty() {
                window
                    .exe
                    .as_deref()
                    .and_then(std::path::Path::file_stem)
                    .map(|stem| stem.to_string_lossy().into_owned())
                    .unwrap_or_default()
            } else {
                window.title
            }
        })
        .unwrap_or_default()
}

/// The display numbered `chosen` (the primary when there is no such display).
fn area(driver: &dyn Driver, chosen: usize) -> Result<Display, String> {
    let displays = driver.displays();
    if let Some(display) = displays.get(chosen).or_else(|| displays.first()) {
        return Ok(*display);
    }
    let (width, height) = driver.screen_size()?;
    Ok(Display {
        left: 0,
        top: 0,
        width,
        height,
        primary: true,
    })
}

/// A point of `display`'s picture on the desktop; refused outside the picture.
fn point(display: &Display, x: f64, y: f64) -> Result<(i32, i32), String> {
    let (picture_w, picture_h, scale) = policy::fit(display.width, display.height);
    if !(0.0..f64::from(picture_w)).contains(&x) || !(0.0..f64::from(picture_h)).contains(&y) {
        return Err(format!(
            "({x}, {y}) is outside the picture, which is {picture_w} x {picture_h} pixels."
        ));
    }
    let (screen_x, screen_y) = policy::to_screen(x, y, scale);
    Ok((display.left + screen_x, display.top + screen_y))
}

fn guard(window: Option<&Window>, own: u32) -> Result<(), String> {
    match window.and_then(|window| policy::protected(window.exe.as_deref(), window.pid, own)) {
        Some(why) => Err(why.to_owned()),
        None => Ok(()),
    }
}

impl Desktop {
    pub fn new(driver: Arc<dyn Driver>, handle: Handle) -> Self {
        Self {
            driver,
            handle,
            own: std::process::id(),
            keys: (
                lattice_sys::desktop::MOD_CONTROL | lattice_sys::desktop::MOD_ALT,
                lattice_sys::desktop::VK_END,
            ),
            hotkey: Mutex::new(None),
            display: Mutex::new(0),
        }
    }

    /// The system's desktop.
    pub fn system(handle: Handle) -> Self {
        Self::new(Arc::new(SystemDriver), handle)
    }

    /// The same desktop with another process counted as Alelyon (tests).
    pub fn with_own(mut self, own: u32) -> Self {
        self.own = own;
        self
    }

    /// The same desktop with other stop keys (tests: keys nobody presses).
    pub fn with_stop_keys(mut self, modifiers: u32, virtual_key: u32) -> Self {
        self.keys = (modifiers, virtual_key);
        self
    }

    async fn blocking<T: Send + 'static>(
        &self,
        work: impl FnOnce(&dyn Driver) -> Result<T, String> + Send + 'static,
    ) -> Result<T, String> {
        let driver = self.driver.clone();
        self.handle
            .spawn_blocking(move || work(driver.as_ref()))
            .await
            .unwrap_or_else(|_| Err("The desktop could not be reached.".to_owned()))
    }

    /// The window in front, as its title names it (empty when none is).
    pub async fn front(&self) -> String {
        self.blocking(|driver| Ok(front_title(driver.foreground())))
            .await
            .unwrap_or_default()
    }

    fn chosen(&self) -> usize {
        *lock(&self.display)
    }

    /// Look at the display as it is.
    pub async fn look(&self) -> Result<Look, String> {
        let chosen = self.chosen();
        self.blocking(move |driver| {
            let display = area(driver, chosen)?;
            let screen =
                driver.capture_area(display.left, display.top, display.width, display.height)?;
            let (image, width, height, _) =
                super::image::picture(&screen.bgra, screen.width, screen.height)?;
            Ok(Look {
                front: front_title(driver.foreground()),
                image,
                width,
                height,
            })
        })
        .await
    }

    /// The displays, the primary first.
    pub async fn displays(&self) -> Result<Vec<Display>, String> {
        self.blocking(|driver| {
            let displays = driver.displays();
            if displays.is_empty() {
                Ok(vec![area(driver, 0)?])
            } else {
                Ok(displays)
            }
        })
        .await
    }

    /// Work on display `index` from now on, and look at it.
    pub async fn show_display(&self, index: usize) -> Result<Look, String> {
        let count = self.displays().await?.len();
        if index >= count {
            return Err(format!(
                "There is no display {index}: this desktop has {count}, numbered from 0."
            ));
        }
        *lock(&self.display) = index;
        self.look().await
    }

    /// An area of the display's picture, from one corner to the other, at the
    /// display's own pixels (the picture is scaled down to fit).
    pub async fn zoom(&self, from: (f64, f64), to: (f64, f64)) -> Result<Look, String> {
        let chosen = self.chosen();
        self.blocking(move |driver| {
            let display = area(driver, chosen)?;
            let a = point(&display, from.0.min(to.0), from.1.min(to.1))?;
            let b = point(&display, from.0.max(to.0), from.1.max(to.1))?;
            let (width, height) = ((b.0 - a.0).max(1) as u32, (b.1 - a.1).max(1) as u32);
            let screen = driver.capture_area(a.0, a.1, width, height)?;
            let (image, width, height, _) =
                super::image::picture(&screen.bgra, screen.width, screen.height)?;
            Ok(Look {
                front: front_title(driver.foreground()),
                image,
                width,
                height,
            })
        })
        .await
    }

    async fn settle_and_look(&self) -> Result<Look, String> {
        tokio::time::sleep(SETTLE).await;
        self.look().await
    }

    /// Click `count` times (1 to 3) with `button` at a point of the picture.
    pub async fn click(&self, x: f64, y: f64, button: Button, count: u32) -> Result<Look, String> {
        let (own, chosen) = (self.own, self.chosen());
        self.blocking(move |driver| {
            let (screen_x, screen_y) = point(&area(driver, chosen)?, x, y)?;
            guard(driver.window_at(screen_x, screen_y).as_ref(), own)?;
            driver.press_at(screen_x, screen_y, button, count.clamp(1, 3))
        })
        .await?;
        self.settle_and_look().await
    }

    /// Drag with the left button from one point of the picture to another;
    /// neither end may be on a window the agent never acts on.
    pub async fn drag(&self, from: (f64, f64), to: (f64, f64)) -> Result<Look, String> {
        let (own, chosen) = (self.own, self.chosen());
        self.blocking(move |driver| {
            let display = area(driver, chosen)?;
            let start = point(&display, from.0, from.1)?;
            let end = point(&display, to.0, to.1)?;
            guard(driver.window_at(start.0, start.1).as_ref(), own)?;
            guard(driver.window_at(end.0, end.1).as_ref(), own)?;
            driver.drag(start, end)
        })
        .await?;
        self.settle_and_look().await
    }

    /// Hold one key with its modifiers for `seconds` (0.1 to 10), unless
    /// Alelyon, a password manager or a Windows prompt is in front.
    pub async fn hold(&self, press: &Press, seconds: f64) -> Result<Look, String> {
        if !seconds.is_finite() {
            return Err("Give how long to hold it, in seconds.".to_owned());
        }
        let millis = (seconds.clamp(0.1, 10.0) * 1000.0).round() as u64;
        let (key, modifiers, own) = (press.virtual_key as u16, press.modifiers, self.own);
        self.blocking(move |driver| {
            guard(driver.foreground().as_ref(), own)?;
            driver.hold_key(key, modifiers, millis)
        })
        .await?;
        self.settle_and_look().await
    }

    /// The clipboard's text (at most [`MAX_CLIPBOARD`] characters), when it
    /// holds text. Its caller asks the reader first.
    pub async fn clipboard(&self) -> Result<Option<String>, String> {
        self.blocking(|driver| {
            Ok(driver
                .clipboard_text()?
                .map(|text| text.chars().take(MAX_CLIPBOARD).collect()))
        })
        .await
    }

    /// Put text on the clipboard.
    pub async fn set_clipboard(&self, text: &str) -> Result<(), String> {
        if text.chars().count() > MAX_CLIPBOARD {
            return Err("That is more text than the clipboard takes here (100,000 characters).".to_owned());
        }
        let text = text.to_owned();
        self.blocking(move |driver| driver.set_clipboard_text(&text)).await
    }

    /// Open a program as a person does: the Start menu, its name, Enter. A
    /// program the agent never acts on is not opened.
    pub async fn open(&self, app: &str) -> Result<Look, String> {
        let app = app.trim().to_owned();
        if app.is_empty() || app.chars().count() > MAX_APP || app.chars().any(char::is_control) {
            return Err("Give the program's name, as the Start menu shows it (one line, at most 100 characters).".to_owned());
        }
        if let Some(why) = policy::protected_name(&app) {
            return Err(why.to_owned());
        }
        let own = self.own;
        self.blocking(move |driver| {
            guard(driver.foreground().as_ref(), own)?;
            driver.key(WINDOWS_KEY, 0)
        })
        .await?;
        tokio::time::sleep(START_MENU).await;
        self.blocking(move |driver| {
            guard(driver.foreground().as_ref(), own)?;
            driver.type_text(&app)
        })
        .await?;
        tokio::time::sleep(START_MENU).await;
        self.blocking(move |driver| {
            guard(driver.foreground().as_ref(), own)?;
            driver.key(ENTER, 0)
        })
        .await?;
        tokio::time::sleep(PROGRAM_START).await;
        self.look().await
    }

    /// Type text into whatever has the focus, unless Alelyon, a password
    /// manager or a Windows prompt is in front, or the focus is a password box.
    pub async fn type_text(&self, text: &str) -> Result<Look, String> {
        if text.is_empty() {
            return Err("Give the text to type.".to_owned());
        }
        if text.chars().count() > MAX_TYPE {
            return Err("That is more text than one action types (4,000 characters).".to_owned());
        }
        let (text, own) = (text.to_owned(), self.own);
        self.blocking(move |driver| {
            guard(driver.foreground().as_ref(), own)?;
            if driver.focus_is_password() {
                return Err("The focus is a password box: the agent never types a password. Sign in yourself.".to_owned());
            }
            driver.type_text(&text)
        })
        .await?;
        self.settle_and_look().await
    }

    /// Press one key with its modifiers, unless Alelyon, a password manager or
    /// a Windows prompt is in front.
    pub async fn press(&self, press: &Press) -> Result<Look, String> {
        let (key, modifiers, own) = (press.virtual_key as u16, press.modifiers, self.own);
        self.blocking(move |driver| {
            guard(driver.foreground().as_ref(), own)?;
            driver.key(key, modifiers)
        })
        .await?;
        self.settle_and_look().await
    }

    /// Scroll the window in the middle of the display by wheel notches.
    pub async fn scroll(&self, direction: &str, notches: i32) -> Result<Look, String> {
        let notches = notches.clamp(1, 30);
        let (down, right) = match direction {
            "up" => (-notches, 0),
            "left" => (0, -notches),
            "right" => (0, notches),
            _ => (notches, 0),
        };
        let (own, chosen) = (self.own, self.chosen());
        self.blocking(move |driver| {
            let display = area(driver, chosen)?;
            let (x, y) = (
                display.left + display.width as i32 / 2,
                display.top + display.height as i32 / 2,
            );
            guard(driver.window_at(x, y).as_ref(), own)?;
            driver.scroll(x, y, down, right)
        })
        .await?;
        self.settle_and_look().await
    }

    /// Arm the stop hotkey ([`STOP_KEYS`]): `on_press` runs on its own thread
    /// each time it is pressed. Already armed: nothing changes.
    pub fn arm_hotkey(&self, on_press: Box<dyn Fn() + Send>) -> Result<(), String> {
        let mut slot = lock(&self.hotkey);
        if slot.is_some() {
            return Ok(());
        }
        let hotkey = lattice_sys::desktop::register_hotkey(self.keys.0, self.keys.1, on_press)
            .map_err(|_| {
                format!("{STOP_KEYS} is held by another program, so it cannot stop the agent.")
            })?;
        *slot = Some(hotkey);
        Ok(())
    }

    /// Disarm the stop hotkey.
    pub fn disarm_hotkey(&self) {
        drop(lock(&self.hotkey).take());
    }

    pub fn hotkey_armed(&self) -> bool {
        lock(&self.hotkey).is_some()
    }
}

/// Whether auto mode is on (`<native>/chat/auto_mode.json`): switched on only
/// through the core's own dialog (`ConfirmRequest::AutoMode`).
pub mod prefs {
    use serde_json::{Value, json};

    use crate::fsx;
    use crate::state::StateRoot;

    pub fn file(state: &StateRoot) -> std::path::PathBuf {
        state.native_chat_dir().join("auto_mode.json")
    }

    pub fn on(state: &StateRoot) -> bool {
        std::fs::read(file(state))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
            .is_some_and(|value| value.get("on").and_then(Value::as_bool) == Some(true))
    }

    pub(crate) fn set(state: &StateRoot, on: bool) -> Result<(), String> {
        let path = file(state);
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|_| "Lattice could not make its settings folder.".to_owned())?;
        }
        let bytes = serde_json::to_vec_pretty(&json!({"v": 1, "on": on})).unwrap_or_default();
        fsx::atomic_write(&path, &bytes)
            .map_err(|_| "Lattice could not save the setting.".to_owned())
    }
}
