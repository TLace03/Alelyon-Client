//! The desktop's hands over a fake driver: what they refuse before any input
//! is sent, what they send, and the picture they return. No test here
//! touches the real desktop.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use lattice_sys::desktop::{Screen, Window};

use super::policy::parse_keys;
use super::session::{Desktop, Driver};

/// A desktop of `size` pixels whose windows the test chooses, recording
/// every input sent.
pub(crate) struct FakeDesktop {
    pub size: (u32, u32),
    pub sent: Mutex<Vec<String>>,
    pub front: Mutex<Option<Window>>,
    pub under: Mutex<Option<Window>>,
    pub password: Mutex<bool>,
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
    fn capture(&self) -> Result<Screen, String> {
        let (width, height) = self.size;
        Ok(Screen {
            width,
            height,
            bgra: vec![90; width as usize * height as usize * 4],
        })
    }
    fn window_at(&self, _x: i32, _y: i32) -> Option<Window> {
        self.under.lock().unwrap().clone()
    }
    fn foreground(&self) -> Option<Window> {
        self.front.lock().unwrap().clone()
    }
    fn focus_is_password(&self) -> bool {
        *self.password.lock().unwrap()
    }
    fn click(&self, x: i32, y: i32, double: bool) -> Result<(), String> {
        self.sent.lock().unwrap().push(format!(
            "click {x},{y}{}",
            if double { " twice" } else { "" }
        ));
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
        .block_on(desktop.click(640.0, 360.0, false))
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
    runtime.block_on(desktop.click(10.0, 10.0, true)).unwrap();
    assert_eq!(fake.sent()[1], "click 20,20 twice");
    let outside = runtime
        .block_on(desktop.click(1280.0, 100.0, false))
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
        .block_on(desktop.click(100.0, 100.0, false))
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
