//! The whole desktop, for the agent's auto mode (as chosen on
//! 2026-10-08: "If users want to enable auto mode with bypass permissions then
//! whole desktop should be the route"). Not a port.
//!
//! - [`capture`]: the primary screen's pixels (`BitBlt` with `CAPTUREBLT`
//!   into a 32-bit top-down DIB, `GetDIBits`).
//! - [`window_at`] and [`foreground`]: the top-level window at a point or in
//!   front, with its process's executable (`WindowFromPoint`, `GetAncestor`,
//!   `QueryFullProcessImageNameW`), so the chat core can refuse to act on
//!   Alelyon itself, a password manager or Windows' own sign-in prompts.
//! - [`focus_is_password`]: whether the focused control is a classic password
//!   box (an `Edit` with `ES_PASSWORD`, through `GetGUIThreadInfo`).
//! - [`displays`] and [`capture_area`]: each display's place on the desktop
//!   (`EnumDisplayMonitors`) and the pixels of an area of it.
//! - [`click`], [`press_at`], [`drag`], [`type_text`], [`key`], [`hold_key`]
//!   and [`scroll`]: input as a person gives it (`SendInput`), at points of the
//!   whole desktop (`MOUSEEVENTF_VIRTUALDESK`). Windows does not let them reach
//!   a window of a higher integrity level (an elevated program, a UAC prompt).
//! - [`clipboard_text`] and [`set_clipboard_text`]: the clipboard's text
//!   (`CF_UNICODETEXT`).
//! - [`register_hotkey`]: one global hotkey on a thread of its own
//!   (`RegisterHotKey`, a `GetMessageW` loop that waits without a timer),
//!   unregistered when the [`Hotkey`] drops.
//!
//! Every call that reads or gives a position runs in the per-monitor-aware
//! DPI context (`SetThreadDpiAwarenessContext`), so pixels and points are
//! the screen's own whatever the process's DPI mode.

use std::io;
use std::path::PathBuf;

/// The primary screen's pixels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Screen {
    pub width: u32,
    pub height: u32,
    /// Blue, green, red and an unused byte per pixel, the top row first.
    pub bgra: Vec<u8>,
}

/// A top-level window and its process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Window {
    pub pid: u32,
    /// The process's executable, when Windows names it to this process.
    pub exe: Option<PathBuf>,
    pub title: String,
    pub class: String,
}

/// One display: its place on the desktop, in the desktop's own pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Display {
    pub left: i32,
    pub top: i32,
    pub width: u32,
    pub height: u32,
    pub primary: bool,
}

/// A mouse button.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Button {
    Left,
    Right,
    Middle,
}

/// Hotkey modifiers (`MOD_ALT`, `MOD_CONTROL`, `MOD_SHIFT`).
pub const MOD_ALT: u32 = 1;
pub const MOD_CONTROL: u32 = 2;
pub const MOD_SHIFT: u32 = 4;
/// `VK_END`.
pub const VK_END: u32 = 0x23;

/// The primary screen's size in its own pixels.
pub fn screen_size() -> io::Result<(u32, u32)> {
    imp::screen_size()
}

/// The primary screen, as it is now.
pub fn capture() -> io::Result<Screen> {
    imp::capture()
}

/// Every display, the primary first, then from left to right.
pub fn displays() -> Vec<Display> {
    let mut list = imp::displays();
    list.sort_by_key(|d| (!d.primary, d.left, d.top));
    list
}

/// The pixels of an area of the desktop, as it is now.
pub fn capture_area(left: i32, top: i32, width: u32, height: u32) -> io::Result<Screen> {
    imp::capture_area(left, top, width, height)
}

/// The top-level window at a point of the primary screen.
pub fn window_at(x: i32, y: i32) -> Option<Window> {
    imp::window_at(x, y)
}

/// The window in front.
pub fn foreground() -> Option<Window> {
    imp::foreground()
}

/// Whether the focused control of the window in front is a classic password
/// box (`Edit` with `ES_PASSWORD`). A password field drawn by a browser or
/// another toolkit is not seen here.
pub fn focus_is_password() -> bool {
    imp::focus_is_password()
}

/// Click (twice when `double`) at a point of the primary screen.
pub fn click(x: i32, y: i32, double: bool) -> io::Result<()> {
    imp::click(x, y, double)
}

/// Click `count` times (1 to 3) with `button` at a point of the desktop.
pub fn press_at(x: i32, y: i32, button: Button, count: u32) -> io::Result<()> {
    imp::press_at(x, y, button, count.clamp(1, 3))
}

/// Press the left button at one point of the desktop, move to another and
/// let go there.
pub fn drag(from: (i32, i32), to: (i32, i32)) -> io::Result<()> {
    imp::drag(from, to)
}

/// Hold one virtual key with its modifiers (as [`key`]) for `millis`
/// milliseconds, at most 10 seconds.
pub fn hold_key(virtual_key: u16, modifiers: u32, millis: u64) -> io::Result<()> {
    imp::hold_key(virtual_key, modifiers, millis.min(10_000))
}

/// The clipboard's text, when it holds text.
pub fn clipboard_text() -> io::Result<Option<String>> {
    imp::clipboard_text()
}

/// Put `text` on the clipboard, in place of what it held.
pub fn set_clipboard_text(text: &str) -> io::Result<()> {
    imp::set_clipboard_text(text)
}

/// Type `text` into whatever has the focus: each character as itself
/// (`KEYEVENTF_UNICODE`), a line end as Enter and a tab as Tab.
pub fn type_text(text: &str) -> io::Result<()> {
    imp::type_text(text)
}

/// Press one virtual key with its modifiers (the browser's bits: Alt 1,
/// Ctrl 2, Shift 8), holding them for the press.
pub fn key(virtual_key: u16, modifiers: u32) -> io::Result<()> {
    imp::key(virtual_key, modifiers)
}

/// Scroll at a point: `down` and `right` notches of the wheel (negative is up
/// or left).
pub fn scroll(x: i32, y: i32, down: i32, right: i32) -> io::Result<()> {
    imp::scroll(x, y, down, right)
}

/// A position on a screen of `size` pixels as `SendInput`'s absolute
/// coordinate, 0 to 65535 across it.
pub fn normalize(position: i32, size: u32) -> i32 {
    let last = i64::from(size.max(2) - 1);
    let position = i64::from(position).clamp(0, last);
    ((position * 65535 + last / 2) / last) as i32
}

/// A global hotkey, registered while this lives.
pub struct Hotkey {
    #[cfg_attr(not(windows), allow(dead_code))]
    thread: u32,
    join: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Hotkey {
    fn drop(&mut self) {
        imp::quit(self.thread);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

/// Register a global hotkey (`modifiers` of [`MOD_ALT`], [`MOD_CONTROL`] and
/// [`MOD_SHIFT`], and a virtual key): `on_press` runs on the hotkey's own
/// thread each time it is pressed. It fails when another program holds the
/// same keys.
pub fn register_hotkey(
    modifiers: u32,
    virtual_key: u32,
    on_press: Box<dyn Fn() + Send>,
) -> io::Result<Hotkey> {
    let (thread, join) = imp::register_hotkey(modifiers, virtual_key, on_press)?;
    Ok(Hotkey {
        thread,
        join: Some(join),
    })
}

#[cfg(windows)]
mod imp {
    use std::ffi::c_void;
    use std::io;
    use std::mem::size_of;
    use std::path::PathBuf;
    use std::ptr::null_mut;
    use std::sync::mpsc;

    use windows_sys::Win32::Foundation::{CloseHandle, GlobalFree, HWND, LPARAM, POINT, RECT};
    use windows_sys::Win32::Graphics::Gdi::{
        BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BitBlt, CAPTUREBLT, CreateCompatibleBitmap,
        CreateCompatibleDC, DIB_RGB_COLORS, DeleteDC, DeleteObject, EnumDisplayMonitors, GetDC,
        GetDIBits, GetMonitorInfoW, HDC, HMONITOR, MONITORINFO, ReleaseDC, SRCCOPY, SelectObject,
    };
    use windows_sys::Win32::System::DataExchange::{
        CloseClipboard, EmptyClipboard, GetClipboardData, IsClipboardFormatAvailable,
        OpenClipboard, SetClipboardData,
    };
    use windows_sys::Win32::System::Memory::{GMEM_MOVEABLE, GlobalAlloc, GlobalLock, GlobalUnlock};
    use windows_sys::Win32::System::Threading::{
        GetCurrentThreadId, OpenProcess, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION,
        QueryFullProcessImageNameW,
    };
    use windows_sys::Win32::UI::HiDpi::{
        DPI_AWARENESS_CONTEXT, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
        SetThreadDpiAwarenessContext,
    };
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        INPUT, INPUT_0, INPUT_KEYBOARD, INPUT_MOUSE, KEYBDINPUT, KEYEVENTF_EXTENDEDKEY,
        KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, MOD_NOREPEAT, MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_HWHEEL,
        MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP,
        MOUSEEVENTF_MOVE, MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, MOUSEEVENTF_VIRTUALDESK,
        MOUSEEVENTF_WHEEL, MOUSEINPUT, RegisterHotKey, SendInput, UnregisterHotKey,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        ES_PASSWORD, GA_ROOT, GUITHREADINFO, GWL_STYLE, GetAncestor, GetClassNameW,
        GetForegroundWindow, GetGUIThreadInfo, GetMessageW, GetSystemMetrics, GetWindowLongW,
        GetWindowTextW, GetWindowThreadProcessId, MSG, PM_NOREMOVE, PeekMessageW,
        PostThreadMessageW, SM_CXSCREEN, SM_CXVIRTUALSCREEN, SM_CYSCREEN, SM_CYVIRTUALSCREEN,
        SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN, WM_HOTKEY, WM_QUIT, WindowFromPoint,
    };

    use super::{Button, Display, Screen, Window, normalize};

    /// `MONITORINFOF_PRIMARY`.
    const PRIMARY: u32 = 1;
    /// `CF_UNICODETEXT`.
    const UNICODE_TEXT: u32 = 13;
    /// The longest clipboard text read, in UTF-16 units.
    const MAX_CLIPBOARD_UNITS: usize = 1 << 20;

    /// The thread's DPI context, per-monitor aware while this lives.
    struct PhysicalPixels(DPI_AWARENESS_CONTEXT);

    impl PhysicalPixels {
        fn new() -> Self {
            // SAFETY: a documented context constant; the call only changes
            // this thread's DPI context and returns the previous one (null
            // when it fails, which leaves the context as it was).
            Self(unsafe {
                SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
            })
        }
    }

    impl Drop for PhysicalPixels {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: restores the context this thread had before.
                unsafe { SetThreadDpiAwarenessContext(self.0) };
            }
        }
    }

    fn size() -> io::Result<(i32, i32)> {
        // SAFETY: GetSystemMetrics takes a documented index and reads only.
        let (width, height) =
            unsafe { (GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)) };
        if width <= 0 || height <= 0 {
            return Err(io::Error::other("there is no screen to look at"));
        }
        Ok((width, height))
    }

    pub(super) fn screen_size() -> io::Result<(u32, u32)> {
        let _physical = PhysicalPixels::new();
        size().map(|(width, height)| (width as u32, height as u32))
    }

    pub(super) fn capture() -> io::Result<Screen> {
        let _physical = PhysicalPixels::new();
        let (width, height) = size()?;
        capture_pixels(0, 0, width, height)
    }

    pub(super) fn capture_area(left: i32, top: i32, width: u32, height: u32) -> io::Result<Screen> {
        let _physical = PhysicalPixels::new();
        if width == 0 || height == 0 || width > 16_384 || height > 16_384 {
            return Err(io::Error::other("that area of the desktop cannot be looked at"));
        }
        capture_pixels(left, top, width as i32, height as i32)
    }

    /// `width` x `height` pixels of the desktop from (`left`, `top`).
    fn capture_pixels(left: i32, top: i32, width: i32, height: i32) -> io::Result<Screen> {
        // SAFETY: a null window asks for the whole screen's device context;
        // it is released below on every path.
        let screen = unsafe { GetDC(null_mut()) };
        if screen.is_null() {
            return Err(io::Error::other("the screen could not be read"));
        }
        // SAFETY: `screen` is a valid device context; the memory context and
        // the bitmap made from it are deleted below on every path.
        let (memory, bitmap) = unsafe {
            (
                CreateCompatibleDC(screen),
                CreateCompatibleBitmap(screen, width, height),
            )
        };
        let mut pixels = vec![0u8; width as usize * height as usize * 4];
        let mut result = Err(io::Error::other("the screen could not be copied"));
        if !memory.is_null() && !bitmap.is_null() {
            // SAFETY: both contexts and the bitmap are valid; the bitmap is
            // selected into the memory context for the copy and the previous
            // object selected back before the bitmap is read.
            let copied = unsafe {
                let previous = SelectObject(memory, bitmap);
                let copied = BitBlt(
                    memory,
                    0,
                    0,
                    width,
                    height,
                    screen,
                    left,
                    top,
                    SRCCOPY | CAPTUREBLT,
                );
                SelectObject(memory, previous);
                copied
            };
            if copied != 0 {
                let mut info = BITMAPINFO {
                    bmiHeader: BITMAPINFOHEADER {
                        biSize: size_of::<BITMAPINFOHEADER>() as u32,
                        biWidth: width,
                        // Negative: the top row first.
                        biHeight: -height,
                        biPlanes: 1,
                        biBitCount: 32,
                        biCompression: BI_RGB,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                // SAFETY: `pixels` holds `height` rows of `width` 32-bit
                // pixels, exactly what the header asks GetDIBits to write;
                // the bitmap is no longer selected into any context.
                let lines = unsafe {
                    GetDIBits(
                        memory,
                        bitmap,
                        0,
                        height as u32,
                        pixels.as_mut_ptr().cast::<c_void>(),
                        &mut info,
                        DIB_RGB_COLORS,
                    )
                };
                if lines == height {
                    result = Ok(());
                }
            }
        }
        // SAFETY: each handle was made above and is released once; a null
        // one is skipped.
        unsafe {
            if !bitmap.is_null() {
                DeleteObject(bitmap);
            }
            if !memory.is_null() {
                DeleteDC(memory);
            }
            ReleaseDC(null_mut(), screen);
        }
        result.map(|()| Screen {
            width: width as u32,
            height: height as u32,
            bgra: pixels,
        })
    }

    unsafe extern "system" fn each_display(
        monitor: HMONITOR,
        _context: HDC,
        _clip: *mut RECT,
        data: LPARAM,
    ) -> windows_sys::core::BOOL {
        let mut info = MONITORINFO {
            cbSize: size_of::<MONITORINFO>() as u32,
            ..Default::default()
        };
        // SAFETY: `monitor` came from EnumDisplayMonitors and `info` is
        // writable with its size set; `data` is the address of the list
        // `displays` passed, alive for the whole enumeration.
        unsafe {
            if GetMonitorInfoW(monitor, &mut info) != 0 {
                let r = info.rcMonitor;
                (*(data as *mut Vec<Display>)).push(Display {
                    left: r.left,
                    top: r.top,
                    width: (r.right - r.left).max(0) as u32,
                    height: (r.bottom - r.top).max(0) as u32,
                    primary: info.dwFlags & PRIMARY != 0,
                });
            }
        }
        1
    }

    pub(super) fn displays() -> Vec<Display> {
        let _physical = PhysicalPixels::new();
        let mut list: Vec<Display> = Vec::new();
        // SAFETY: no device context and no clip: every display; the callback
        // writes only into `list`, which outlives the call.
        unsafe {
            EnumDisplayMonitors(
                null_mut(),
                std::ptr::null(),
                Some(each_display),
                &mut list as *mut Vec<Display> as LPARAM,
            )
        };
        list
    }

    fn text_of(read: impl Fn(*mut u16, i32) -> i32) -> String {
        let mut buffer = vec![0u16; 512];
        let length = read(buffer.as_mut_ptr(), buffer.len() as i32).max(0) as usize;
        String::from_utf16_lossy(&buffer[..length.min(buffer.len())])
    }

    fn describe(window: HWND) -> Option<Window> {
        if window.is_null() {
            return None;
        }
        // SAFETY: GetAncestor reads the window tree; a null result keeps the
        // window itself.
        let root = unsafe { GetAncestor(window, GA_ROOT) };
        let window = if root.is_null() { window } else { root };
        let mut pid = 0u32;
        // SAFETY: `pid` is writable; the window handle came from the system.
        unsafe { GetWindowThreadProcessId(window, &mut pid) };
        // SAFETY: each buffer is writable for the length passed.
        let title = text_of(|buffer, length| unsafe { GetWindowTextW(window, buffer, length) });
        // SAFETY: as above.
        let class = text_of(|buffer, length| unsafe { GetClassNameW(window, buffer, length) });
        Some(Window {
            pid,
            exe: executable(pid),
            title,
            class,
        })
    }

    fn executable(pid: u32) -> Option<PathBuf> {
        if pid == 0 {
            return None;
        }
        // SAFETY: the limited query right is all QueryFullProcessImageNameW
        // needs; the handle is closed below.
        let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if process.is_null() {
            return None;
        }
        let mut buffer = vec![0u16; 1024];
        let mut length = buffer.len() as u32;
        // SAFETY: the buffer is writable for `length` units; `length` is
        // writable and is set to what was written.
        let ok = unsafe {
            QueryFullProcessImageNameW(
                process,
                PROCESS_NAME_WIN32,
                buffer.as_mut_ptr(),
                &mut length,
            )
        };
        // SAFETY: the handle was opened above.
        unsafe { CloseHandle(process) };
        (ok != 0).then(|| {
            PathBuf::from(String::from_utf16_lossy(
                &buffer[..(length as usize).min(buffer.len())],
            ))
        })
    }

    pub(super) fn window_at(x: i32, y: i32) -> Option<Window> {
        let _physical = PhysicalPixels::new();
        // SAFETY: WindowFromPoint reads the window under a point.
        describe(unsafe { WindowFromPoint(POINT { x, y }) })
    }

    pub(super) fn foreground() -> Option<Window> {
        // SAFETY: GetForegroundWindow reads which window is in front.
        describe(unsafe { GetForegroundWindow() })
    }

    pub(super) fn focus_is_password() -> bool {
        // SAFETY: reads the window in front and its thread; a null window
        // gives thread 0, for which GetGUIThreadInfo reads the foreground
        // thread.
        let thread = unsafe { GetWindowThreadProcessId(GetForegroundWindow(), null_mut()) };
        let mut info = GUITHREADINFO {
            cbSize: size_of::<GUITHREADINFO>() as u32,
            ..Default::default()
        };
        // SAFETY: `info` is writable and its size is set.
        if unsafe { GetGUIThreadInfo(thread, &mut info) } == 0 || info.hwndFocus.is_null() {
            return false;
        }
        let focus = info.hwndFocus;
        // SAFETY: the buffer is writable for the length passed.
        let class = text_of(|buffer, length| unsafe { GetClassNameW(focus, buffer, length) });
        // ES_PASSWORD means a password box only on an edit control.
        let edit = class.eq_ignore_ascii_case("Edit")
            || class.to_ascii_lowercase().starts_with("richedit");
        // SAFETY: reads the control's style.
        edit && unsafe { GetWindowLongW(focus, GWL_STYLE) } & ES_PASSWORD != 0
    }

    fn mouse(dx: i32, dy: i32, data: u32, flags: u32) -> INPUT {
        INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dx,
                    dy,
                    mouseData: data,
                    dwFlags: flags,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        }
    }

    fn keyboard(virtual_key: u16, scan: u16, flags: u32) -> INPUT {
        INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: virtual_key,
                    wScan: scan,
                    dwFlags: flags,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        }
    }

    fn send(inputs: &[INPUT]) -> io::Result<()> {
        // SAFETY: the slice holds `inputs.len()` initialised INPUTs of the
        // size passed.
        let sent = unsafe {
            SendInput(
                inputs.len() as u32,
                inputs.as_ptr(),
                size_of::<INPUT>() as i32,
            )
        };
        if sent as usize == inputs.len() {
            Ok(())
        } else {
            Err(io::Error::other(
                "Windows did not take the input (a window of a higher level, such as an elevated program, is in front)",
            ))
        }
    }

    /// A move to a point of the whole desktop (every display).
    fn move_to(x: i32, y: i32) -> io::Result<INPUT> {
        // SAFETY: GetSystemMetrics takes documented indices and reads only.
        let (left, top, width, height) = unsafe {
            (
                GetSystemMetrics(SM_XVIRTUALSCREEN),
                GetSystemMetrics(SM_YVIRTUALSCREEN),
                GetSystemMetrics(SM_CXVIRTUALSCREEN),
                GetSystemMetrics(SM_CYVIRTUALSCREEN),
            )
        };
        if width <= 0 || height <= 0 {
            let (width, height) = size()?;
            return Ok(mouse(
                normalize(x, width as u32),
                normalize(y, height as u32),
                0,
                MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE,
            ));
        }
        Ok(mouse(
            normalize(x - left, width as u32),
            normalize(y - top, height as u32),
            0,
            MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
        ))
    }

    pub(super) fn press_at(x: i32, y: i32, button: Button, count: u32) -> io::Result<()> {
        let _physical = PhysicalPixels::new();
        let (down, up) = match button {
            Button::Left => (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
            Button::Right => (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
            Button::Middle => (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
        };
        let mut inputs = vec![move_to(x, y)?];
        for _ in 0..count {
            inputs.push(mouse(0, 0, 0, down));
            inputs.push(mouse(0, 0, 0, up));
        }
        send(&inputs)
    }

    /// The steps a drag moves in, and the pause after each (a program sees
    /// a drag only when the pointer moves while the button is down).
    const DRAG_STEPS: i32 = 12;
    const DRAG_PAUSE: std::time::Duration = std::time::Duration::from_millis(15);

    pub(super) fn drag(from: (i32, i32), to: (i32, i32)) -> io::Result<()> {
        let _physical = PhysicalPixels::new();
        send(&[move_to(from.0, from.1)?, mouse(0, 0, 0, MOUSEEVENTF_LEFTDOWN)])?;
        let mut moved = Ok(());
        for step in 1..=DRAG_STEPS {
            std::thread::sleep(DRAG_PAUSE);
            let x = from.0 + (to.0 - from.0) * step / DRAG_STEPS;
            let y = from.1 + (to.1 - from.1) * step / DRAG_STEPS;
            moved = move_to(x, y).and_then(|input| send(&[input]));
            if moved.is_err() {
                break;
            }
        }
        // The button is let go whatever happened, so it is never left down.
        let released = send(&[mouse(0, 0, 0, MOUSEEVENTF_LEFTUP)]);
        moved.and(released)
    }

    pub(super) fn clipboard_text() -> io::Result<Option<String>> {
        let _open = Clipboard::open()?;
        // SAFETY: the clipboard is open on this thread; the handle it gives
        // stays the clipboard's, read while locked and unlocked below.
        unsafe {
            if IsClipboardFormatAvailable(UNICODE_TEXT) == 0 {
                return Ok(None);
            }
            let handle = GetClipboardData(UNICODE_TEXT);
            if handle.is_null() {
                return Ok(None);
            }
            let units = GlobalLock(handle) as *const u16;
            if units.is_null() {
                return Ok(None);
            }
            let mut length = 0usize;
            while length < MAX_CLIPBOARD_UNITS && *units.add(length) != 0 {
                length += 1;
            }
            let text = String::from_utf16_lossy(std::slice::from_raw_parts(units, length));
            GlobalUnlock(handle);
            Ok(Some(text))
        }
    }

    pub(super) fn set_clipboard_text(text: &str) -> io::Result<()> {
        let units: Vec<u16> = text.encode_utf16().chain(std::iter::once(0)).collect();
        // SAFETY: a movable block of exactly the units' size, filled while
        // locked; once the clipboard takes it, it is the clipboard's, and it
        // is freed here only when the clipboard did not take it.
        unsafe {
            let block = GlobalAlloc(GMEM_MOVEABLE, units.len() * 2);
            if block.is_null() {
                return Err(io::Error::other("there was no memory for the clipboard's text"));
            }
            let target = GlobalLock(block) as *mut u16;
            if target.is_null() {
                GlobalFree(block);
                return Err(io::Error::other("there was no memory for the clipboard's text"));
            }
            std::ptr::copy_nonoverlapping(units.as_ptr(), target, units.len());
            GlobalUnlock(block);
            let _open = match Clipboard::open() {
                Ok(open) => open,
                Err(error) => {
                    GlobalFree(block);
                    return Err(error);
                }
            };
            EmptyClipboard();
            if SetClipboardData(UNICODE_TEXT, block).is_null() {
                GlobalFree(block);
                return Err(io::Error::other("the clipboard did not take the text"));
            }
        }
        Ok(())
    }

    /// The clipboard, open on this thread while this lives.
    struct Clipboard;

    impl Clipboard {
        fn open() -> io::Result<Clipboard> {
            // SAFETY: opens the clipboard for this thread, with no window.
            if unsafe { OpenClipboard(null_mut()) } == 0 {
                return Err(io::Error::other("another program is using the clipboard"));
            }
            Ok(Clipboard)
        }
    }

    impl Drop for Clipboard {
        fn drop(&mut self) {
            // SAFETY: this thread opened the clipboard.
            unsafe { CloseClipboard() };
        }
    }

    pub(super) fn click(x: i32, y: i32, double: bool) -> io::Result<()> {
        let _physical = PhysicalPixels::new();
        let mut inputs = vec![move_to(x, y)?];
        for _ in 0..if double { 2 } else { 1 } {
            inputs.push(mouse(0, 0, 0, MOUSEEVENTF_LEFTDOWN));
            inputs.push(mouse(0, 0, 0, MOUSEEVENTF_LEFTUP));
        }
        send(&inputs)
    }

    /// `VK_RETURN`, `VK_TAB`, `VK_SHIFT`, `VK_CONTROL`, `VK_MENU`.
    const RETURN: u16 = 0x0D;
    const TAB: u16 = 0x09;
    const SHIFT: u16 = 0x10;
    const CONTROL: u16 = 0x11;
    const MENU: u16 = 0x12;

    pub(super) fn type_text(text: &str) -> io::Result<()> {
        let mut inputs = Vec::new();
        for character in text.chars() {
            match character {
                '\r' => {}
                '\n' => {
                    inputs.push(keyboard(RETURN, 0, 0));
                    inputs.push(keyboard(RETURN, 0, KEYEVENTF_KEYUP));
                }
                '\t' => {
                    inputs.push(keyboard(TAB, 0, 0));
                    inputs.push(keyboard(TAB, 0, KEYEVENTF_KEYUP));
                }
                _ => {
                    let mut units = [0u16; 2];
                    for unit in character.encode_utf16(&mut units) {
                        inputs.push(keyboard(0, *unit, KEYEVENTF_UNICODE));
                        inputs.push(keyboard(0, *unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP));
                    }
                }
            }
        }
        send(&inputs)
    }

    /// The keys Windows sends as extended (arrows, Insert, Delete, Home, End,
    /// Page Up, Page Down, the context menu key).
    fn extended(virtual_key: u16) -> bool {
        matches!(virtual_key, 0x21..=0x28 | 0x2D | 0x2E | 0x5D)
    }

    pub(super) fn key(virtual_key: u16, modifiers: u32) -> io::Result<()> {
        let held: Vec<u16> = [(2, CONTROL), (1, MENU), (8, SHIFT)]
            .into_iter()
            .filter(|(bit, _)| modifiers & bit != 0)
            .map(|(_, key)| key)
            .collect();
        let flags = if extended(virtual_key) {
            KEYEVENTF_EXTENDEDKEY
        } else {
            0
        };
        let mut inputs: Vec<INPUT> = held.iter().map(|key| keyboard(*key, 0, 0)).collect();
        inputs.push(keyboard(virtual_key, 0, flags));
        inputs.push(keyboard(virtual_key, 0, flags | KEYEVENTF_KEYUP));
        inputs.extend(
            held.iter()
                .rev()
                .map(|key| keyboard(*key, 0, KEYEVENTF_KEYUP)),
        );
        send(&inputs)
    }

    pub(super) fn hold_key(virtual_key: u16, modifiers: u32, millis: u64) -> io::Result<()> {
        let held: Vec<u16> = [(2, CONTROL), (1, MENU), (8, SHIFT)]
            .into_iter()
            .filter(|(bit, _)| modifiers & bit != 0)
            .map(|(_, key)| key)
            .chain(std::iter::once(virtual_key))
            .collect();
        let flags = |key: u16| if extended(key) { KEYEVENTF_EXTENDEDKEY } else { 0 };
        let down: Vec<INPUT> = held.iter().map(|key| keyboard(*key, 0, flags(*key))).collect();
        let pressed = send(&down);
        if pressed.is_ok() {
            std::thread::sleep(std::time::Duration::from_millis(millis));
        }
        // Every key is let go whatever happened, so none is left held.
        let up: Vec<INPUT> = held
            .iter()
            .rev()
            .map(|key| keyboard(*key, 0, flags(*key) | KEYEVENTF_KEYUP))
            .collect();
        let released = send(&up);
        pressed.and(released)
    }

    /// One notch of the wheel (`WHEEL_DELTA`).
    const NOTCH: i32 = 120;

    pub(super) fn scroll(x: i32, y: i32, down: i32, right: i32) -> io::Result<()> {
        let _physical = PhysicalPixels::new();
        let mut inputs = vec![move_to(x, y)?];
        if down != 0 {
            // A positive wheel turn scrolls up.
            inputs.push(mouse(0, 0, (-down * NOTCH) as u32, MOUSEEVENTF_WHEEL));
        }
        if right != 0 {
            inputs.push(mouse(0, 0, (right * NOTCH) as u32, MOUSEEVENTF_HWHEEL));
        }
        send(&inputs)
    }

    pub(super) fn register_hotkey(
        modifiers: u32,
        virtual_key: u32,
        on_press: Box<dyn Fn() + Send>,
    ) -> io::Result<(u32, std::thread::JoinHandle<()>)> {
        let (told, heard) = mpsc::channel();
        let join = std::thread::Builder::new()
            .name("lattice-hotkey".into())
            .spawn(move || {
                // SAFETY: reads this thread's id.
                let thread = unsafe { GetCurrentThreadId() };
                let mut message = MSG::default();
                // SAFETY: makes this thread's message queue exist before its
                // id is handed out, so a quit posted at once is not lost.
                unsafe { PeekMessageW(&mut message, null_mut(), 0, 0, PM_NOREMOVE) };
                // SAFETY: a thread hotkey (no window), id 1 of this thread.
                let registered =
                    unsafe { RegisterHotKey(null_mut(), 1, modifiers | MOD_NOREPEAT, virtual_key) }
                        != 0;
                let _ = told.send(if registered {
                    Ok(thread)
                } else {
                    Err(io::Error::last_os_error())
                });
                if !registered {
                    return;
                }
                // SAFETY: `message` is writable; GetMessageW waits for this
                // thread's next message (no timer) and returns 0 on WM_QUIT.
                while unsafe { GetMessageW(&mut message, null_mut(), 0, 0) } > 0 {
                    if message.message == WM_HOTKEY {
                        on_press();
                    }
                }
                // SAFETY: the hotkey this thread registered.
                unsafe { UnregisterHotKey(null_mut(), 1) };
            })?;
        match heard.recv() {
            Ok(Ok(thread)) => Ok((thread, join)),
            Ok(Err(error)) => {
                let _ = join.join();
                Err(error)
            }
            Err(_) => Err(io::Error::other("the hotkey's thread ended")),
        }
    }

    pub(super) fn quit(thread: u32) {
        // SAFETY: posts WM_QUIT to the hotkey's thread, whose queue exists;
        // a thread that has ended makes the call fail harmlessly.
        unsafe { PostThreadMessageW(thread, WM_QUIT, 0, 0) };
    }
}

/// Portable stand-in, used only by tests on other targets.
#[cfg(not(windows))]
mod imp {
    use std::io;

    use super::{Button, Display, Screen, Window};

    fn unsupported<T>() -> io::Result<T> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "the desktop is driven only on Windows",
        ))
    }

    pub(super) fn screen_size() -> io::Result<(u32, u32)> {
        unsupported()
    }
    pub(super) fn capture() -> io::Result<Screen> {
        unsupported()
    }
    pub(super) fn capture_area(_left: i32, _top: i32, _width: u32, _height: u32) -> io::Result<Screen> {
        unsupported()
    }
    pub(super) fn displays() -> Vec<Display> {
        Vec::new()
    }
    pub(super) fn press_at(_x: i32, _y: i32, _button: Button, _count: u32) -> io::Result<()> {
        unsupported()
    }
    pub(super) fn drag(_from: (i32, i32), _to: (i32, i32)) -> io::Result<()> {
        unsupported()
    }
    pub(super) fn hold_key(_key: u16, _modifiers: u32, _millis: u64) -> io::Result<()> {
        unsupported()
    }
    pub(super) fn clipboard_text() -> io::Result<Option<String>> {
        unsupported()
    }
    pub(super) fn set_clipboard_text(_text: &str) -> io::Result<()> {
        unsupported()
    }
    pub(super) fn window_at(_x: i32, _y: i32) -> Option<Window> {
        None
    }
    pub(super) fn foreground() -> Option<Window> {
        None
    }
    pub(super) fn focus_is_password() -> bool {
        false
    }
    pub(super) fn click(_x: i32, _y: i32, _double: bool) -> io::Result<()> {
        unsupported()
    }
    pub(super) fn type_text(_text: &str) -> io::Result<()> {
        unsupported()
    }
    pub(super) fn key(_key: u16, _modifiers: u32) -> io::Result<()> {
        unsupported()
    }
    pub(super) fn scroll(_x: i32, _y: i32, _down: i32, _right: i32) -> io::Result<()> {
        unsupported()
    }
    pub(super) fn register_hotkey(
        _modifiers: u32,
        _key: u32,
        _on_press: Box<dyn Fn() + Send>,
    ) -> io::Result<(u32, std::thread::JoinHandle<()>)> {
        unsupported()
    }
    pub(super) fn quit(_thread: u32) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_position_is_spread_over_sendinputs_absolute_range() {
        assert_eq!(normalize(0, 1920), 0);
        assert_eq!(normalize(1919, 1920), 65535);
        assert_eq!(normalize(-5, 1920), 0, "clamped");
        assert_eq!(normalize(5000, 1920), 65535, "clamped");
        let middle = normalize(960, 1920);
        assert!((32700..32800).contains(&middle), "{middle}");
    }

    /// Reads only: the displays Windows lists, the primary first.
    #[cfg(windows)]
    #[test]
    fn the_displays_are_listed_the_primary_first() {
        let displays = displays();
        assert!(!displays.is_empty(), "a desktop with no display");
        assert!(displays[0].primary, "{displays:?}");
        assert_eq!(displays.iter().filter(|d| d.primary).count(), 1, "{displays:?}");
        assert!(displays.iter().all(|d| d.width > 0 && d.height > 0), "{displays:?}");
        let (width, height) = screen_size().unwrap();
        assert_eq!((displays[0].width, displays[0].height), (width, height), "the primary's size");
    }
}
