//! What the agent may do on the whole desktop in auto mode, decided in one
//! place. Pure: no clock, no file, no screen.
//!
//! - **Windows it never acts on** ([`protected`]): Alelyon itself (this
//!   process, or a program named as Alelyon's own: CENTCOM, Lattice, Sinai's
//!   Angel), a password manager, and Windows' own sign-in, lock and
//!   permission prompts. A click on one, or typing or a key press while one
//!   is in front, is refused with the reason.
//! - **What still asks** ([`asks`]): chosen on 2026-10-08,
//!   "Money and accounts ask": in auto mode a purchase or a payment (`buy`)
//!   and a sign-in, security, consent or account change (`account`) wait for
//!   the reader; `view`, `edit`, `share` and `delete` go ahead.
//! - **The picture** ([`fit`]): the screen scaled to fit 1280 x 800, the
//!   coordinates the model clicks in; [`to_screen`] maps a point back.

use std::path::Path;

pub use crate::browser::policy::{Effect, Press, parse_keys};

/// The largest picture of the screen the model is shown.
pub const PICTURE: (u32, u32) = (1280, 800);

/// Programs that are Alelyon's own (their file stems, lower case).
pub const ALELYON: [&str; 5] = [
    "centcom",
    "alelyon",
    "lattice",
    "projectangel",
    "angel-native",
];

/// Password managers (file stems, lower case, or a stem's start).
pub const PASSWORD_MANAGERS: [&str; 14] = [
    "1password",
    "bitwarden",
    "keepass",
    "keepassxc",
    "lastpass",
    "dashlane",
    "keeper",
    "roboform",
    "nordpass",
    "enpass",
    "passwordsafe",
    "pwsafe",
    "protonpass",
    "proton pass",
];

/// Windows' own sign-in, lock and permission prompts.
pub const WINDOWS_PROMPTS: [&str; 5] = [
    "credentialuibroker",
    "consent",
    "logonui",
    "lockapp",
    "credentialenrollmentmanager",
];

/// Why the agent may not act on a window of `exe` (in process `pid`), or
/// `None` when it may. `own` is this process's id.
pub fn protected(exe: Option<&Path>, pid: u32, own: u32) -> Option<&'static str> {
    if pid == own {
        return Some("That is Alelyon itself, which the agent never controls.");
    }
    let stem = exe
        .and_then(Path::file_stem)
        .map(|stem| stem.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if stem.is_empty() {
        return None;
    }
    if ALELYON.iter().any(|name| stem == *name) {
        return Some("That is Alelyon itself, which the agent never controls.");
    }
    if PASSWORD_MANAGERS.iter().any(|name| stem.starts_with(name)) {
        return Some("That is a password manager, which the agent never controls.");
    }
    if WINDOWS_PROMPTS.iter().any(|name| stem == *name) {
        return Some("That is Windows' own sign-in or permission prompt, which only you answer.");
    }
    None
}

/// Whether an action of `effect` waits for the reader in auto mode.
pub fn asks(effect: Effect) -> bool {
    matches!(effect, Effect::Buy | Effect::Account)
}

/// The picture's size for a screen of `width` x `height` and its scale
/// (picture pixels per screen pixel, at most 1).
pub fn fit(width: u32, height: u32) -> (u32, u32, f64) {
    let (width, height) = (width.max(1), height.max(1));
    let scale = (f64::from(PICTURE.0) / f64::from(width))
        .min(f64::from(PICTURE.1) / f64::from(height))
        .min(1.0);
    let out = |side: u32| ((f64::from(side) * scale).round() as u32).max(1);
    (out(width), out(height), scale)
}

/// A point of the picture on the screen.
pub fn to_screen(x: f64, y: f64, scale: f64) -> (i32, i32) {
    ((x / scale).round() as i32, (y / scale).round() as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alelyon_password_managers_and_windows_prompts_are_never_acted_on() {
        let own = 4242;
        assert!(protected(None, own, own).unwrap().contains("Alelyon"));
        for (exe, kind) in [
            (r"D:\src\centcom\target\release\centcom.exe", "Alelyon"),
            (r"D:\Apps\Alelyon\ProjectAngel.exe", "Alelyon"),
            (
                r"C:\Program Files\1Password\app\8\1Password.exe",
                "password manager",
            ),
            (
                r"C:\Program Files\Bitwarden\Bitwarden.exe",
                "password manager",
            ),
            (
                r"C:\Program Files\KeePassXC\KeePassXC.exe",
                "password manager",
            ),
            (r"C:\Windows\System32\CredentialUIBroker.exe", "sign-in"),
            (r"C:\Windows\System32\consent.exe", "sign-in"),
        ] {
            let why = protected(Some(Path::new(exe)), 1, own).unwrap_or_else(|| panic!("{exe}"));
            assert!(why.contains(kind), "{exe}: {why}");
        }
        for exe in [
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Windows\explorer.exe",
            r"C:\Windows\System32\notepad.exe",
            r"C:\Program Files\Spotify\Spotify.exe",
        ] {
            assert_eq!(protected(Some(Path::new(exe)), 1, own), None, "{exe}");
        }
        assert_eq!(
            protected(None, 1, own),
            None,
            "an unnamed program is judged by its acts"
        );
    }

    #[test]
    fn in_auto_mode_money_and_accounts_still_ask() {
        for (effect, asked) in [
            (Effect::View, false),
            (Effect::Edit, false),
            (Effect::Share, false),
            (Effect::Delete, false),
            (Effect::Buy, true),
            (Effect::Account, true),
        ] {
            assert_eq!(asks(effect), asked, "{effect:?}");
        }
    }

    #[test]
    fn the_screen_is_scaled_to_fit_the_picture_and_points_map_back() {
        assert_eq!(fit(2560, 1440), (1280, 720, 0.5));
        assert_eq!(fit(1280, 800), (1280, 800, 1.0));
        assert_eq!(fit(1024, 768), (1024, 768, 1.0), "never enlarged");
        let (w, h, scale) = fit(3840, 2160);
        assert_eq!((w, h), (1280, 720));
        assert_eq!(to_screen(640.0, 360.0, scale), (1920, 1080));
        let (_, h, _) = fit(1920, 1200);
        assert_eq!(h, 800);
    }
}
