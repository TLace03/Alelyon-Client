//! Walkthroughs (tool parity with Antigravity's walkthroughs and screenshot
//! records): `write_walkthrough` saves what the agent did in this turn with
//! its browser or on the desktop as an HTML artifact, its pictures in the
//! order they were taken, each with its caption and the agent's note.
//!
//! - **The pictures** are those the agent's browser and the desktop's tools
//!   showed the model in this turn ([`Evidence`]): at most [`MAX_KEPT`] are
//!   kept, the first ones, each as it was shown. A walkthrough takes at most
//!   [`MAX_PICTURES`] of them, all unless it names some.
//! - **It is an artifact** like `write_artifact`'s, of kind `html`, saved in
//!   the conversation's record (the same name again is its next version) and
//!   previewed in the browser that has no network. Its words are redacted;
//!   its pictures are embedded as they were shown, and are at most
//!   [`MAX_BYTES`] with the page.

use std::sync::Mutex;

use lattice_agents::model::Image;
use lattice_protocol::conversation::ArtifactKind;
use serde::Deserialize;
use serde_json::Value;

use super::artifacts;
use super::turn::TurnTools;
use crate::secrets;
use crate::tools::read::{ToolError, parse_args};

/// The pictures one turn keeps for a walkthrough.
pub const MAX_KEPT: usize = 24;
/// The pictures one walkthrough shows.
pub const MAX_PICTURES: usize = 12;
/// The most bytes of a walkthrough's page, its pictures included.
pub const MAX_BYTES: usize = 8 * 1024 * 1024;
/// The longest summary, and the longest note.
pub const MAX_SUMMARY_CHARS: usize = 4000;
pub const MAX_NOTE_CHARS: usize = 600;

/// One picture the turn showed the model: what it was, and the image.
#[derive(Clone, Debug, PartialEq)]
pub struct Shown {
    pub caption: String,
    pub image: Image,
}

/// The turn's pictures, in the order they were shown.
#[derive(Debug, Default)]
pub struct Evidence(Mutex<Vec<Shown>>);

impl Evidence {
    /// Keep a picture, while fewer than [`MAX_KEPT`] are kept.
    pub fn keep(&self, caption: &str, image: &Image) {
        let mut kept = self.0.lock().unwrap_or_else(|p| p.into_inner());
        if kept.len() < MAX_KEPT {
            kept.push(Shown { caption: caption.to_owned(), image: image.clone() });
        }
    }

    pub fn all(&self) -> Vec<Shown> {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }
}

#[derive(Debug, Deserialize)]
struct Note {
    picture: usize,
    note: String,
}

#[derive(Debug, Deserialize)]
struct Args {
    name: String,
    title: String,
    summary: String,
    #[serde(default)]
    pictures: Option<Vec<usize>>,
    #[serde(default)]
    notes: Vec<Note>,
}

/// Text for the page, escaped.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// The image's media type, when it is one a page shows.
fn media(image: &Image) -> Option<&str> {
    matches!(image.media_type.as_str(), "image/png" | "image/jpeg" | "image/gif" | "image/webp")
        .then_some(image.media_type.as_str())
}

/// The page: the title, the summary, then each picture (numbered as the turn
/// took them) with its caption and note. Its words are redacted.
pub fn page(title: &str, summary: &str, pictures: &[(usize, &Shown, Option<&str>)]) -> String {
    let words = |text: &str| escape(&secrets::redact(text));
    let mut html = format!(
        "<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>{}</title>\n<style>body{{font-family:system-ui,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#1b1b1b;background:#fff}}figure{{margin:24px 0}}img{{max-width:100%;border:1px solid #ccc}}figcaption{{margin-top:6px;color:#444}}.note{{display:block;margin-top:4px;color:#1b1b1b}}</style></head><body>\n<h1>{}</h1>\n",
        words(title),
        words(title)
    );
    for paragraph in summary.split("\n\n").filter(|p| !p.trim().is_empty()) {
        html.push_str(&format!("<p>{}</p>\n", words(paragraph.trim())));
    }
    for (number, shown, note) in pictures {
        let Some(media) = media(&shown.image) else {
            continue;
        };
        html.push_str(&format!(
            "<figure><img alt=\"Picture {number}\" src=\"data:{media};base64,{}\"><figcaption>{number}. {}",
            shown.image.base64,
            words(&shown.caption)
        ));
        if let Some(note) = note {
            html.push_str(&format!("<span class=\"note\">{}</span>", words(note)));
        }
        html.push_str("</figcaption></figure>\n");
    }
    html.push_str("</body></html>\n");
    html
}

/// `write_walkthrough`: this turn's pictures as an HTML artifact.
pub(crate) fn tool(tools: &TurnTools, call: &str, args: &Value) -> Result<String, ToolError> {
    let args: Args = parse_args(args)?;
    let shown = tools.evidence.all();
    if shown.is_empty() {
        return Err(ToolError::new(
            "This turn has no pictures of the agent's browser or of the screen to show: a walkthrough is made of them.",
        ));
    }
    if args.summary.trim().is_empty() || args.summary.chars().count() > MAX_SUMMARY_CHARS {
        return Err(ToolError::new("A walkthrough's summary is 1 to 4,000 characters."));
    }
    let chosen: Vec<usize> = match &args.pictures {
        Some(numbers) => numbers.clone(),
        None => (1..=shown.len().min(MAX_PICTURES)).collect(),
    };
    if chosen.is_empty() || chosen.len() > MAX_PICTURES {
        return Err(ToolError::new("A walkthrough shows 1 to 12 pictures."));
    }
    if let Some(missing) = chosen.iter().find(|n| **n == 0 || **n > shown.len()) {
        return Err(ToolError::new(format!(
            "There is no picture {missing}: this turn has {} so far, numbered from 1 in the order they were shown.",
            shown.len()
        )));
    }
    if args.notes.iter().any(|n| n.note.chars().count() > MAX_NOTE_CHARS) {
        return Err(ToolError::new("A note is at most 600 characters."));
    }
    let pictures: Vec<(usize, &Shown, Option<&str>)> = chosen
        .iter()
        .map(|n| {
            let note = args.notes.iter().find(|note| note.picture == *n).map(|note| note.note.as_str());
            (*n, &shown[n - 1], note)
        })
        .collect();
    let html = page(args.title.trim(), &args.summary, &pictures);
    if html.len() > MAX_BYTES {
        return Err(ToolError::new(
            "Those pictures make a page of more than 8 MiB: choose fewer of them.",
        ));
    }
    let said = artifacts::save(tools, call, args.name.trim(), args.title.trim(), ArtifactKind::Html, &html, false)?;
    Ok(format!("{said} It shows {} pictures of this turn.", pictures.len()))
}
