//! The desktop's read-only calls and its hotkey, on this machine's own screen.
//! No test here sends input: clicks, typing and keys move the person's own
//! mouse and keyboard, so they are tried by hand only.

#![cfg(windows)]

use lattice_sys::desktop;

#[test]
fn the_primary_screen_is_captured_whole() {
    let (width, height) = match desktop::screen_size() {
        Ok(size) => size,
        Err(why) => {
            eprintln!("no screen in this session: {why}");
            return;
        }
    };
    let screen = desktop::capture().unwrap();
    assert_eq!((screen.width, screen.height), (width, height));
    assert_eq!(screen.bgra.len(), width as usize * height as usize * 4);
}

#[test]
fn a_window_names_its_process() {
    if let Some(front) = desktop::foreground() {
        assert!(front.pid > 0, "{front:?}");
    }
    if let Ok((width, height)) = desktop::screen_size()
        && let Some(window) = desktop::window_at(width as i32 / 2, height as i32 / 2)
    {
        assert!(!window.class.is_empty(), "{window:?}");
    }
    // Whatever is focused, the answer is a plain yes or no.
    let _ = desktop::focus_is_password();
}

/// Ctrl+Alt+Shift+F24: keys no keyboard of this PC presses.
#[test]
fn a_hotkey_is_held_while_it_lives_and_freed_when_it_drops() {
    const F24: u32 = 0x87;
    let keys = desktop::MOD_CONTROL | desktop::MOD_ALT | desktop::MOD_SHIFT;
    let held = desktop::register_hotkey(keys, F24, Box::new(|| {})).unwrap();
    assert!(
        desktop::register_hotkey(keys, F24, Box::new(|| {})).is_err(),
        "a second registration of the same keys is refused"
    );
    drop(held);
    let again = desktop::register_hotkey(keys, F24, Box::new(|| {})).unwrap();
    drop(again);
}

/// By hand: moves the mouse to the middle of the screen, clicks there, types
/// "lattice" and presses Escape. Run it with something harmless in front.
#[test]
#[ignore = "it moves your mouse and types"]
fn by_hand_input_reaches_the_window_in_front() {
    let (width, height) = desktop::screen_size().unwrap();
    desktop::click(width as i32 / 2, height as i32 / 2, false).unwrap();
    desktop::type_text("lattice").unwrap();
    desktop::key(0x1B, 0).unwrap();
    desktop::scroll(width as i32 / 2, height as i32 / 2, 1, 0).unwrap();
}
