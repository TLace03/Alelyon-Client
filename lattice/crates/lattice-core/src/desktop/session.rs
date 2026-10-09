//! [`Desktop`]: the agent's hands on the whole desktop in auto mode. It looks
//! at the primary screen (a picture scaled to fit 1280 x 800), clicks at a
//! point of that picture, types, presses keys and scrolls, as a person does,
//! through a [`Driver`] (the system's, `lattice_sys::desktop`; a test's own).
//!
//! - **Never on Alelyon, a password manager or Windows' prompts**
//!   ([`super::policy::protected`]): a click on such a window, or typing or a
//!   key press while one is in front, is refused before any input is sent.
//! - **Never into a password box**: typing is refused while the focus is a
//!   classic password box (the driver's `focus_is_password`). A password
//!   field another toolkit draws is not seen; the agent has no passwords to
//!   type, and says so in its instructions.
//! - **Every action ends with a picture** of the screen, once it has had a
//!   moment to settle: what the model sees next.
//! - **The stop hotkey** (Ctrl+Alt+End), armed while auto mode is on, stops
//!   every running agent turn ([`Desktop::arm_hotkey`]).

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use lattice_agents::model::Image;
use lattice_sys::desktop::{Screen, Window};
use tokio::runtime::Handle;

use super::policy::{self, Press};

/// The pause after an action before the screen is looked at.
const SETTLE: Duration = Duration::from_millis(600);
/// The longest text typed in one action.
pub const MAX_TYPE: usize = 4000;
/// The stop hotkey: Ctrl+Alt+End.
pub const STOP_KEYS: &str = "Ctrl+Alt+End";

/// What the desktop is, to the agent: the system's, or a test's.
pub trait Driver: Send + Sync {
    fn screen_size(&self) -> Result<(u32, u32), String>;
    fn capture(&self) -> Result<Screen, String>;
    fn window_at(&self, x: i32, y: i32) -> Option<Window>;
    fn foreground(&self) -> Option<Window>;
    fn focus_is_password(&self) -> bool;
    fn click(&self, x: i32, y: i32, double: bool) -> Result<(), String>;
    fn type_text(&self, text: &str) -> Result<(), String>;
    fn key(&self, virtual_key: u16, modifiers: u32) -> Result<(), String>;
    fn scroll(&self, x: i32, y: i32, down: i32, right: i32) -> Result<(), String>;
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
    fn capture(&self) -> Result<Screen, String> {
        lattice_sys::desktop::capture().map_err(sentence)
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
    fn click(&self, x: i32, y: i32, double: bool) -> Result<(), String> {
        lattice_sys::desktop::click(x, y, double).map_err(sentence)
    }
    fn type_text(&self, text: &str) -> Result<(), String> {
        lattice_sys::desktop::type_text(text).map_err(sentence)
    }
    fn key(&self, virtual_key: u16, modifiers: u32) -> Result<(), String> {
        lattice_sys::desktop::key(virtual_key, modifiers).map_err(sentence)
    }
    fn scroll(&self, x: i32, y: i32, down: i32, right: i32) -> Result<(), String> {
        lattice_sys::desktop::scroll(x, y, down, right).map_err(sentence)
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

    /// Look at the screen as it is.
    pub async fn look(&self) -> Result<Look, String> {
        self.blocking(|driver| {
            let screen = driver.capture()?;
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

    /// Click (twice when `double`) at a point of the picture.
    pub async fn click(&self, x: f64, y: f64, double: bool) -> Result<Look, String> {
        let own = self.own;
        self.blocking(move |driver| {
            let (width, height) = driver.screen_size()?;
            let (picture_w, picture_h, scale) = policy::fit(width, height);
            if !(0.0..f64::from(picture_w)).contains(&x)
                || !(0.0..f64::from(picture_h)).contains(&y)
            {
                return Err(format!(
                    "({x}, {y}) is outside the picture, which is {picture_w} x {picture_h} pixels."
                ));
            }
            let (screen_x, screen_y) = policy::to_screen(x, y, scale);
            guard(driver.window_at(screen_x, screen_y).as_ref(), own)?;
            driver.click(screen_x, screen_y, double)
        })
        .await?;
        self.settle_and_look().await
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

    /// Scroll the window in the middle of the screen by wheel notches.
    pub async fn scroll(&self, direction: &str, notches: i32) -> Result<Look, String> {
        let notches = notches.clamp(1, 30);
        let (down, right) = match direction {
            "up" => (-notches, 0),
            "left" => (0, -notches),
            "right" => (0, notches),
            _ => (notches, 0),
        };
        let own = self.own;
        self.blocking(move |driver| {
            let (width, height) = driver.screen_size()?;
            let (x, y) = (width as i32 / 2, height as i32 / 2);
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
