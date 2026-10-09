//! What the agent's browser may open and do, decided in one place. Pure: no
//! clock, no file, no network.
//!
//! - **Addresses** ([`check_url`]): `http` and `https` only (and
//!   `about:blank`), with a host that names the public web. Refused: this PC
//!   and its own services (`localhost`, loopback addresses), a private or
//!   link-local network (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`,
//!   `169.254.0.0/16`, `fc00::/7`, `fe80::/10`, `0.0.0.0`), a name with no dot
//!   (an intranet name) or under `.local`, `.internal`, `.localhost`,
//!   `.home.arpa`, and an address carrying a user name or password. The
//!   browser is a way to use websites as a person does, never a way to reach
//!   the PC or the network it sits on.
//! - **Effects** ([`Effect`]): every click and key press the agent asks for
//!   says what it does. `view` and `edit` go ahead; `share` (another person
//!   will see it: a post, a message, a comment, a reaction, a follow), `buy`
//!   (money moves or an order is placed), `delete` and `account` (sign-in,
//!   security or settings) ask the reader first, as the charter's rules ask
//!   before what another person will see and what cannot be undone.
//! - **What it looks like** ([`click_looks`], [`press_looks`]): a second
//!   guard, because the agent may misstate an effect. Before a click or a
//!   press stated as `view` or `edit`, the page says what is there; a button
//!   or link whose label reads like posting, paying, deleting or an account
//!   change ("Post", "Buy now", "Delete", "Log out", "Accept all"), and Enter
//!   in a message box, are refused with the effect they look like, so the
//!   agent restates it and the reader is asked. Words only: a guard that can
//!   miss a label, never one that lets a stated `share` through unasked.
//! - **Keys** ([`parse_keys`]): a press is a key name with optional `Ctrl`,
//!   `Alt` and `Shift`, such as `Enter`, `Tab`, `Ctrl+A`, `Shift+Tab`.

use std::net::{Ipv4Addr, Ipv6Addr};

/// The longest address opened.
pub const MAX_URL: usize = 4096;

/// What one action does, as the agent states it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Effect {
    /// Opens, reads, scrolls, searches: nothing lasting.
    View,
    /// Changes something only the reader sees, and can be undone (a draft, a
    /// filter, a form not yet sent).
    Edit,
    /// Another person will see it.
    Share,
    /// Money moves or an order is placed.
    Buy,
    /// Something is removed.
    Delete,
    /// Sign-in, security, privacy or account settings.
    Account,
}

impl Effect {
    pub const ALL: [Effect; 6] = [
        Effect::View,
        Effect::Edit,
        Effect::Share,
        Effect::Buy,
        Effect::Delete,
        Effect::Account,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Effect::View => "view",
            Effect::Edit => "edit",
            Effect::Share => "share",
            Effect::Buy => "buy",
            Effect::Delete => "delete",
            Effect::Account => "account",
        }
    }

    /// The effect a name gives, if it is one.
    pub fn parse(name: &str) -> Option<Effect> {
        Effect::ALL
            .into_iter()
            .find(|effect| effect.name().eq_ignore_ascii_case(name.trim()))
    }

    /// Whether the reader is asked first.
    pub fn asks(self) -> bool {
        matches!(
            self,
            Effect::Share | Effect::Buy | Effect::Delete | Effect::Account
        )
    }

    /// What the reader is told it does.
    pub fn words(self) -> &'static str {
        match self {
            Effect::View => "opens or reads something",
            Effect::Edit => "changes something only you see",
            Effect::Share => "something another person will see",
            Effect::Buy => "a purchase or an order",
            Effect::Delete => "deletes something",
            Effect::Account => "changes a sign-in, a consent, security or an account setting",
        }
    }
}

/// What the page says is at a click's point, or in the focused field.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(default)]
pub struct Target {
    /// `ok`, `none`, `password`, `card` or `frame` (the focused field only).
    pub kind: String,
    /// The element's tag, lower case.
    pub tag: String,
    /// A button, a link or a control (not text the click landed in).
    pub actionable: bool,
    /// Its label: `aria-label`, its text, its value or its title.
    pub label: String,
    /// Its `data-testid`, which sites name their buttons by.
    pub testid: String,
    /// Inside another site's frame, which the page cannot read.
    pub frame: bool,
    /// A box one writes a message in (a `textarea`, an editable element).
    pub editor: bool,
    /// A search field.
    pub search: bool,
    /// The label of the button that sends the field's form.
    pub submit: String,
}

/// Labels longer than this many words are content (a headline, a post's
/// text), not the name of an action.
const LABEL_WORDS: usize = 5;

/// The words of a label that read like an action that asks, by effect, in
/// the order a label is matched (a purchase first). Two words match side by
/// side.
const LOOKS: [(Effect, &[&str]); 4] = [
    (
        Effect::Buy,
        &[
            "buy",
            "purchase",
            "pay",
            "checkout",
            "check out",
            "order",
            "place order",
            "donate",
            "subscribe",
            "book now",
        ],
    ),
    (
        Effect::Delete,
        &["delete", "remove", "discard", "trash", "erase"],
    ),
    (
        Effect::Account,
        &[
            "authorize",
            "authorise",
            "deactivate",
            "logout",
            "signout",
            "sign out",
            "log out",
            "accept",
            "agree",
            "allow",
            "grant",
            "password",
            "unsubscribe",
        ],
    ),
    (
        Effect::Share,
        &[
            "post", "tweet", "retweet", "repost", "reply", "send", "comment", "publish", "share",
            "follow", "unfollow", "like", "submit", "upload", "invite", "react",
        ],
    ),
];

/// A label's words, lower case: runs of letters and digits, a `camelCase`
/// identifier split at its capitals.
pub fn words_of(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut previous_lower = false;
    for c in text.chars() {
        if c.is_alphanumeric() {
            if c.is_uppercase() && previous_lower && !word.is_empty() {
                words.push(std::mem::take(&mut word));
            }
            previous_lower = c.is_lowercase() || c.is_ascii_digit();
            word.extend(c.to_lowercase());
        } else {
            previous_lower = false;
            if !word.is_empty() {
                words.push(std::mem::take(&mut word));
            }
        }
    }
    if !word.is_empty() {
        words.push(word);
    }
    words
}

/// The effect a label's words read like, if one that asks.
pub fn label_looks(words: &[String]) -> Option<Effect> {
    LOOKS.iter().find_map(|(effect, phrases)| {
        phrases
            .iter()
            .any(|phrase| {
                let parts: Vec<&str> = phrase.split(' ').collect();
                words
                    .windows(parts.len())
                    .any(|window| window.iter().zip(&parts).all(|(word, part)| word == part))
            })
            .then_some(*effect)
    })
}

/// What a click stated as `view` or `edit` looks like, when its target is a
/// button, a link or a control whose short label or test id reads like an
/// action that asks.
pub fn click_looks(target: &Target) -> Option<Effect> {
    if target.frame || !target.actionable {
        return None;
    }
    let mut words = words_of(&target.label);
    if words.len() > LABEL_WORDS {
        words.clear();
    }
    words.extend(words_of(&target.testid));
    label_looks(&words)
}

/// What a key press stated as `view` or `edit` looks like: Enter (or Space)
/// activates the focused button, sends its field's form, or, in a message
/// box, often sends the message (Shift+Enter is a new line). A search field
/// is looked up, never sent.
pub fn press_looks(press: &Press, focused: &Target) -> Option<Effect> {
    let enter = press.key == "Enter" && press.modifiers & 8 == 0;
    let space = press.key == " " && press.modifiers & 3 == 0;
    if !(enter || space) || focused.search || focused.frame {
        return None;
    }
    if enter && focused.editor {
        return Some(Effect::Share);
    }
    let mut words = words_of(&focused.label);
    if words.len() > LABEL_WORDS {
        words.clear();
    }
    words.extend(words_of(&focused.testid));
    if enter {
        words.extend(words_of(&focused.submit));
    }
    label_looks(&words)
}

/// The refusal of an action that looks like `looks` but was stated as `view`
/// or `edit`: the agent restates it, and then the reader is asked.
pub fn misstated(looks: Effect, target: &Target) -> String {
    let named = if target.label.is_empty() {
        &target.testid
    } else {
        &target.label
    };
    let named: String = named.chars().take(80).collect();
    let what = if named.is_empty() {
        String::new()
    } else {
        format!(" (\"{named}\")")
    };
    let like = match looks {
        Effect::Buy => "a purchase or an order",
        Effect::Delete => "a deletion",
        Effect::Account => "a sign-in, consent, security or account change",
        Effect::Share | Effect::View | Effect::Edit => "something another person will see",
    };
    format!(
        "That looks like {like}{what}, not view or edit. If it is, state its effect as {}, and the user is asked first.",
        looks.name()
    )
}

fn private_v4(ip: Ipv4Addr) -> bool {
    ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_unspecified()
        || ip.is_broadcast()
        || ip.is_multicast()
        || ip.octets()[0] == 0
        // 100.64.0.0/10, carrier-grade NAT, is not the public web either.
        || (ip.octets()[0] == 100 && (ip.octets()[1] & 0xc0) == 64)
}

fn private_v6(ip: Ipv6Addr) -> bool {
    if let Some(v4) = ip.to_ipv4_mapped() {
        return private_v4(v4);
    }
    let first = ip.segments()[0];
    ip.is_loopback()
        || ip.is_unspecified()
        || ip.is_multicast()
        || (first & 0xfe00) == 0xfc00
        || (first & 0xffc0) == 0xfe80
}

/// Names that are never the public web.
const LOCAL_SUFFIXES: [&str; 5] = [".local", ".internal", ".localhost", ".home.arpa", ".lan"];

/// The address the browser may open, normalised, or the sentence why not.
pub fn check_url(text: &str) -> Result<String, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("Give an address to open.".to_owned());
    }
    if text.len() > MAX_URL {
        return Err("That address is too long.".to_owned());
    }
    if text.eq_ignore_ascii_case("about:blank") {
        return Ok("about:blank".to_owned());
    }
    // A bare name ("example.com/page") is read as https.
    let with_scheme = if text.contains("://") {
        text.to_owned()
    } else {
        format!("https://{text}")
    };
    let url = url::Url::parse(&with_scheme)
        .map_err(|_| "That is not a web address the browser can open.".to_owned())?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err("The agent's browser opens only http and https addresses.".to_owned());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("An address with a user name or password in it is not opened.".to_owned());
    }
    let host = match url.host() {
        Some(url::Host::Domain(name)) => {
            let name = name.trim_end_matches('.').to_ascii_lowercase();
            if name == "localhost"
                || !name.contains('.')
                || LOCAL_SUFFIXES.iter().any(|suffix| name.ends_with(suffix))
            {
                return Err(
                    "That name is on this PC or its local network, which the agent's browser does not open."
                        .to_owned(),
                );
            }
            name
        }
        Some(url::Host::Ipv4(ip)) if private_v4(ip) => {
            return Err(
                "That address is this PC or a private network, which the agent's browser does not open."
                    .to_owned(),
            );
        }
        Some(url::Host::Ipv6(ip)) if private_v6(ip) => {
            return Err(
                "That address is this PC or a private network, which the agent's browser does not open."
                    .to_owned(),
            );
        }
        Some(host) => host.to_string(),
        None => return Err("That address has no host.".to_owned()),
    };
    debug_assert!(!host.is_empty());
    Ok(url.to_string())
}

/// The site an address belongs to, as the reader names it: the host without
/// `www.` (`mail.google.com` stays itself; the Connections page groups by
/// it).
pub fn site_of(url: &str) -> Option<String> {
    let url = url::Url::parse(url).ok()?;
    let host = url.host_str()?.to_ascii_lowercase();
    Some(host.strip_prefix("www.").map(str::to_owned).unwrap_or(host))
}

/// One key press: a key and its modifiers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Press {
    /// The DOM `key` value (`Enter`, `a`, `ArrowDown`).
    pub key: String,
    /// The DOM `code` value (`Enter`, `KeyA`, `ArrowDown`).
    pub code: String,
    /// Windows' virtual-key code.
    pub virtual_key: u32,
    /// CDP's modifier bits: Alt 1, Ctrl 2, Meta 4, Shift 8.
    pub modifiers: u32,
    /// What typing it inserts, for a printable key without Ctrl or Alt.
    pub text: Option<String>,
}

/// The keys a press may name, beside a single letter or digit.
const NAMED: [(&str, &str, u32); 18] = [
    ("Enter", "Enter", 13),
    ("Tab", "Tab", 9),
    ("Escape", "Escape", 27),
    ("Backspace", "Backspace", 8),
    ("Delete", "Delete", 46),
    ("Space", "Space", 32),
    ("ArrowUp", "ArrowUp", 38),
    ("ArrowDown", "ArrowDown", 40),
    ("ArrowLeft", "ArrowLeft", 37),
    ("ArrowRight", "ArrowRight", 39),
    ("Home", "Home", 36),
    ("End", "End", 35),
    ("PageUp", "PageUp", 33),
    ("PageDown", "PageDown", 34),
    ("F5", "F5", 116),
    ("Insert", "Insert", 45),
    ("ContextMenu", "ContextMenu", 93),
    ("F1", "F1", 112),
];

/// `Ctrl+A`, `Shift+Tab`, `Enter`, `a`: one press, or why not.
pub fn parse_keys(text: &str) -> Result<Press, String> {
    let parts: Vec<&str> = text.split('+').map(str::trim).collect();
    if parts.iter().any(|part| part.is_empty()) || parts.is_empty() {
        return Err(
            "Name one key, with Ctrl, Alt or Shift before it: Enter, Tab, Ctrl+A.".to_owned(),
        );
    }
    let (key, modifiers) = parts.split_last().expect("not empty");
    let mut bits = 0u32;
    for modifier in modifiers {
        bits |= match modifier.to_ascii_lowercase().as_str() {
            "alt" => 1,
            "ctrl" | "control" => 2,
            "shift" => 8,
            _ => return Err(format!("{modifier} is not Ctrl, Alt or Shift.")),
        };
    }
    if let Some((name, code, vk)) = NAMED
        .iter()
        .find(|(name, _, _)| name.eq_ignore_ascii_case(key))
    {
        let text = match (*name, bits & 3) {
            ("Enter", 0) => Some("\r".to_owned()),
            ("Space", 0) => Some(" ".to_owned()),
            _ => None,
        };
        let key = if *name == "Space" { " " } else { name };
        return Ok(Press {
            key: key.to_owned(),
            code: (*code).to_owned(),
            virtual_key: *vk,
            modifiers: bits,
            text,
        });
    }
    let mut chars = key.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) if c.is_ascii_alphanumeric() => {
            let upper = c.to_ascii_uppercase();
            let code = if c.is_ascii_digit() {
                format!("Digit{c}")
            } else {
                format!("Key{upper}")
            };
            let shifted = bits & 8 != 0;
            let typed = if c.is_ascii_alphabetic() && shifted {
                upper.to_string()
            } else {
                c.to_ascii_lowercase().to_string()
            };
            Ok(Press {
                key: typed.clone(),
                code,
                virtual_key: u32::from(upper),
                modifiers: bits,
                text: (bits & 3 == 0).then_some(typed),
            })
        }
        _ => Err(format!("{key} is not a key the browser presses.")),
    }
}
