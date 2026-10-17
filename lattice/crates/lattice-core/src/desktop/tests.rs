//! The desktop's hands over a fake driver: what they refuse before any input
//! is sent, what they send, and the picture they return. No test here
//! touches the real desktop.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use lattice_sys::desktop::{Screen, Window};

use super::policy::parse_keys;
use super::session::{Button, Desktop, Display, Driver};

/// A desktop of `size` pixels whose windows the test chooses, recording
/// every input sent.
pub(crate) struct FakeDesktop {
    pub size: (u32, u32),
    pub sent: Mutex<Vec<String>>,
    pub front: Mutex<Option<Window>>,
    pub under: Mutex<Option<Window>>,
    pub password: Mutex<bool>,
    /// The displays it says it has (empty: only the primary, at `size`).
    pub displays: Mutex<Vec<Display>>,
    /// Each area looked at: "left,top width x height".
    pub captured: Mutex<Vec<String>>,
    pub clipboard: Mutex<Option<String>>,
    /// A window at one point only (else `under`).
    pub under_at: Mutex<Option<((i32, i32), Window)>>,
}

impl FakeDesktop {
    pub(crate) fn new(width: u32, height: u32) -> Arc<Self> {
        Arc::new(Self {
            size: (width, height),
            sent: Mutex::default(),
            front: Mutex::new(Some(window(
                r"C:\Windows\System32\notepad.exe",
                "notes.txt - Notepad",
                7,
            ))),
            under: Mutex::new(Some(window(
                r"C:\Windows\System32\notepad.exe",
                "notes.txt - Notepad",
                7,
            ))),
            password: Mutex::new(false),
            displays: Mutex::default(),
            captured: Mutex::default(),
            clipboard: Mutex::default(),
            under_at: Mutex::default(),
        })
    }

    pub(crate) fn sent(&self) -> Vec<String> {
        self.sent.lock().unwrap().clone()
    }
}

pub(crate) fn window(exe: &str, title: &str, pid: u32) -> Window {
    Window {
        pid,
        exe: Some(PathBuf::from(exe)),
        title: title.to_owned(),
        class: "Notepad".to_owned(),
    }
}

impl Driver for FakeDesktop {
    fn screen_size(&self) -> Result<(u32, u32), String> {
        Ok(self.size)
    }
    fn displays(&self) -> Vec<Display> {
        self.displays.lock().unwrap().clone()
    }
    fn capture_area(&self, left: i32, top: i32, width: u32, height: u32) -> Result<Screen, String> {
        self.captured
            .lock()
            .unwrap()
            .push(format!("{left},{top} {width}x{height}"));
        Ok(Screen {
            width,
            height,
            bgra: vec![90; width as usize * height as usize * 4],
        })
    }
    fn window_at(&self, x: i32, y: i32) -> Option<Window> {
        if let Some((at, window)) = self.under_at.lock().unwrap().clone()
            && at == (x, y)
        {
            return Some(window);
        }
        self.under.lock().unwrap().clone()
    }
    fn foreground(&self) -> Option<Window> {
        self.front.lock().unwrap().clone()
    }
    fn focus_is_password(&self) -> bool {
        *self.password.lock().unwrap()
    }
    fn press_at(&self, x: i32, y: i32, button: Button, count: u32) -> Result<(), String> {
        self.sent.lock().unwrap().push(format!(
            "click {x},{y}{}{}",
            match count {
                1 => "",
                2 => " twice",
                _ => " three times",
            },
            match button {
                Button::Left => "",
                Button::Right => " right",
                Button::Middle => " middle",
            }
        ));
        Ok(())
    }
    fn drag(&self, from: (i32, i32), to: (i32, i32)) -> Result<(), String> {
        self.sent
            .lock()
            .unwrap()
            .push(format!("drag {},{} to {},{}", from.0, from.1, to.0, to.1));
        Ok(())
    }
    fn hold_key(&self, virtual_key: u16, modifiers: u32, millis: u64) -> Result<(), String> {
        self.sent
            .lock()
            .unwrap()
            .push(format!("hold {virtual_key} {modifiers} {millis}ms"));
        Ok(())
    }
    fn clipboard_text(&self) -> Result<Option<String>, String> {
        Ok(self.clipboard.lock().unwrap().clone())
    }
    fn set_clipboard_text(&self, text: &str) -> Result<(), String> {
        *self.clipboard.lock().unwrap() = Some(text.to_owned());
        Ok(())
    }
    fn type_text(&self, text: &str) -> Result<(), String> {
        self.sent.lock().unwrap().push(format!("type {text}"));
        Ok(())
    }
    fn key(&self, virtual_key: u16, modifiers: u32) -> Result<(), String> {
        self.sent
            .lock()
            .unwrap()
            .push(format!("key {virtual_key} {modifiers}"));
        Ok(())
    }
    fn scroll(&self, x: i32, y: i32, down: i32, right: i32) -> Result<(), String> {
        self.sent
            .lock()
            .unwrap()
            .push(format!("scroll {x},{y} {down},{right}"));
        Ok(())
    }
}

fn rig(width: u32, height: u32) -> (Arc<FakeDesktop>, Desktop, tokio::runtime::Runtime) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let fake = FakeDesktop::new(width, height);
    let desktop = Desktop::new(fake.clone(), runtime.handle().clone()).with_own(4242);
    (fake, desktop, runtime)
}

#[test]
fn a_click_on_the_picture_lands_on_its_screen_point_and_a_picture_comes_back() {
    let (fake, desktop, runtime) = rig(2560, 1440);
    let look = runtime
        .block_on(desktop.click(640.0, 360.0, Button::Left, 1))
        .unwrap();
    assert_eq!(fake.sent(), ["click 1280,720"]);
    assert_eq!((look.width, look.height), (1280, 720));
    assert!(look.image.base64.starts_with("iVBORw0KGgo"), "a PNG");
    assert!(
        look.summary()
            .starts_with("notes.txt - Notepad is in front"),
        "{}",
        look.summary()
    );
    runtime.block_on(desktop.click(10.0, 10.0, Button::Left, 2)).unwrap();
    assert_eq!(fake.sent()[1], "click 20,20 twice");
    let outside = runtime
        .block_on(desktop.click(1280.0, 100.0, Button::Left, 1))
        .unwrap_err();
    assert!(outside.contains("outside the picture"), "{outside}");
    assert_eq!(fake.sent().len(), 2, "nothing sent for a point outside");
}

#[test]
fn alelyon_a_password_manager_and_windows_prompts_get_no_input() {
    let (fake, desktop, runtime) = rig(1600, 1000);
    *fake.under.lock().unwrap() = Some(super::tests::window(
        r"C:\Program Files\1Password\app\8\1Password.exe",
        "1Password",
        9,
    ));
    let refused = runtime
        .block_on(desktop.click(100.0, 100.0, Button::Left, 1))
        .unwrap_err();
    assert!(refused.contains("password manager"), "{refused}");
    *fake.front.lock().unwrap() = Some(super::tests::window(
        r"D:\src\centcom\target\release\centcom.exe",
        "Alelyon",
        11,
    ));
    let refused = runtime.block_on(desktop.type_text("hello")).unwrap_err();
    assert!(refused.contains("Alelyon"), "{refused}");
    let refused = runtime
        .block_on(desktop.press(&parse_keys("Enter").unwrap()))
        .unwrap_err();
    assert!(refused.contains("Alelyon"), "{refused}");
    // This very process is Alelyon, whatever its file is called.
    *fake.front.lock().unwrap() = Some(super::tests::window(r"C:\x\other.exe", "Other", 4242));
    assert!(
        runtime
            .block_on(desktop.type_text("hello"))
            .unwrap_err()
            .contains("Alelyon")
    );
    *fake.front.lock().unwrap() = Some(super::tests::window(
        r"C:\Windows\System32\consent.exe",
        "User Account Control",
        12,
    ));
    let refused = runtime.block_on(desktop.type_text("yes")).unwrap_err();
    assert!(refused.contains("only you answer"), "{refused}");
    assert!(
        fake.sent().is_empty(),
        "no input reached any of them: {:?}",
        fake.sent()
    );
}

#[test]
fn typing_into_a_password_box_is_refused_and_other_typing_goes_ahead() {
    let (fake, desktop, runtime) = rig(1280, 800);
    *fake.password.lock().unwrap() = true;
    let refused = runtime.block_on(desktop.type_text("hunter2")).unwrap_err();
    assert!(refused.contains("password"), "{refused}");
    assert!(fake.sent().is_empty());
    *fake.password.lock().unwrap() = false;
    runtime.block_on(desktop.type_text("hello there")).unwrap();
    assert_eq!(fake.sent(), ["type hello there"]);
    assert!(runtime.block_on(desktop.type_text("")).is_err());
    let long = "x".repeat(super::session::MAX_TYPE + 1);
    assert!(runtime.block_on(desktop.type_text(&long)).is_err());
}

#[test]
fn keys_and_scrolls_are_sent_with_their_modifiers() {
    let (fake, desktop, runtime) = rig(2560, 1440);
    runtime
        .block_on(desktop.press(&parse_keys("Ctrl+A").unwrap()))
        .unwrap();
    runtime
        .block_on(desktop.press(&parse_keys("Shift+Tab").unwrap()))
        .unwrap();
    runtime.block_on(desktop.scroll("down", 3)).unwrap();
    runtime.block_on(desktop.scroll("up", 99)).unwrap();
    assert_eq!(
        fake.sent(),
        [
            "key 65 2",
            "key 9 8",
            "scroll 1280,720 3,0",
            "scroll 1280,720 -30,0"
        ]
    );
}

#[test]
fn right_middle_and_triple_clicks_drags_and_held_keys_reach_their_points() {
    let (fake, desktop, runtime) = rig(2560, 1440);
    runtime.block_on(desktop.click(640.0, 360.0, Button::Right, 1)).unwrap();
    runtime.block_on(desktop.click(100.0, 50.0, Button::Left, 3)).unwrap();
    runtime.block_on(desktop.click(100.0, 50.0, Button::Middle, 9)).unwrap();
    runtime.block_on(desktop.drag((10.0, 20.0), (300.0, 200.0))).unwrap();
    runtime.block_on(desktop.hold(&parse_keys("Space").unwrap(), 2.0)).unwrap();
    runtime.block_on(desktop.hold(&parse_keys("Ctrl+A").unwrap(), 60.0)).unwrap();
    assert_eq!(
        fake.sent(),
        [
            "click 1280,720 right",
            "click 200,100 three times",
            "click 200,100 three times middle",
            "drag 20,40 to 600,400",
            "hold 32 0 2000ms",
            "hold 65 2 10000ms",
        ]
    );
    let outside = runtime.block_on(desktop.drag((10.0, 20.0), (1300.0, 20.0))).unwrap_err();
    assert!(outside.contains("outside the picture"), "{outside}");
    // A drag that ends on a protected window sends nothing.
    *fake.under_at.lock().unwrap() =
        Some(((600, 400), window(r"C:\Program Files\Bitwarden\Bitwarden.exe", "Bitwarden", 9)));
    let refused = runtime.block_on(desktop.drag((10.0, 20.0), (300.0, 200.0))).unwrap_err();
    assert!(refused.contains("password manager"), "{refused}");
    assert_eq!(fake.sent().len(), 6);
}

#[test]
fn a_second_display_is_looked_at_and_clicked_on_where_it_is() {
    let (fake, desktop, runtime) = rig(2560, 1440);
    *fake.displays.lock().unwrap() = vec![
        Display { left: 0, top: 0, width: 2560, height: 1440, primary: true },
        Display { left: 2560, top: 0, width: 1920, height: 1080, primary: false },
    ];
    let listed = runtime.block_on(desktop.displays()).unwrap();
    assert_eq!(listed.len(), 2);
    let look = runtime.block_on(desktop.show_display(1)).unwrap();
    assert_eq!((look.width, look.height), (1280, 720));
    assert_eq!(fake.captured.lock().unwrap().last().unwrap(), "2560,0 1920x1080");
    runtime.block_on(desktop.click(640.0, 360.0, Button::Left, 1)).unwrap();
    assert_eq!(fake.sent(), ["click 3520,540"]);
    let missing = runtime.block_on(desktop.show_display(2)).unwrap_err();
    assert!(missing.contains("There is no display 2"), "{missing}");
}

#[test]
fn a_zoom_looks_at_its_area_at_the_displays_own_pixels() {
    let (fake, desktop, runtime) = rig(2560, 1440);
    let look = runtime.block_on(desktop.zoom((100.0, 50.0), (300.0, 150.0))).unwrap();
    assert_eq!(fake.captured.lock().unwrap().last().unwrap(), "200,100 400x200");
    assert_eq!((look.width, look.height), (400, 200), "not scaled down: it fits");
    assert!(runtime.block_on(desktop.zoom((0.0, 0.0), (2000.0, 10.0))).is_err());
}

#[test]
fn the_clipboard_is_read_and_filled_and_a_program_is_opened_from_the_start_menu() {
    let (fake, desktop, runtime) = rig(1280, 800);
    assert_eq!(runtime.block_on(desktop.clipboard()).unwrap(), None);
    runtime.block_on(desktop.set_clipboard("copied text")).unwrap();
    assert_eq!(runtime.block_on(desktop.clipboard()).unwrap().as_deref(), Some("copied text"));
    assert!(runtime.block_on(desktop.set_clipboard(&"x".repeat(100_001))).is_err());
    runtime.block_on(desktop.open("Paint")).unwrap();
    assert_eq!(fake.sent(), ["key 91 0", "type Paint", "key 13 0"]);
    let refused = runtime.block_on(desktop.open("1Password")).unwrap_err();
    assert!(refused.contains("password manager"), "{refused}");
    assert!(runtime.block_on(desktop.open("two\nlines")).is_err());
    assert_eq!(fake.sent().len(), 3, "nothing sent for a refused program");
}
