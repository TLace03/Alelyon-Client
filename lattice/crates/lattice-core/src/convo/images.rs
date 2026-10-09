//! Images in the composer: pictures the reader attaches to a message, which
//! the agent's model sees with the words (the tool-parity direction,
//! 2026-10-08, as Claude Code, Cursor and Codex take them).
//!
//! - **What is taken:** at most [`MAX_IMAGES`] PNG or JPEG pictures of at most
//!   [`MAX_BYTES`] each, checked by their first bytes as well as their stated
//!   type. A message still needs words.
//! - **Where they go:** an agent turn only, in the message the model is sent
//!   now (`InputItem::UserImages`). A plain turn (no folder, or a model
//!   without tools) refuses them, as does a model this session learned cannot
//!   see; a model that refuses a message with images is learned so
//!   (`Inner::no_vision`). A send while a turn runs refuses them rather than
//!   queue them.
//! - **Off this PC:** the secret checks read words, not pictures, so every
//!   send with images to a model off this PC asks first
//!   (`ConfirmRequest::SendImages`), however often the conversation was
//!   confirmed before.
//! - **The record:** each picture is a blob of the conversation's sidecar,
//!   written as given, and an [`Item::ImagesAttached`] names them with the
//!   user turn, so a regenerate sends them again. The event of the same name
//!   carries their count and size only. Later turns replay the message's
//!   words with a note that images were attached; the pictures are not sent
//!   again.
//!
//! Not a port: the web Lattice takes no images.

use lattice_agents::model::Image;
use lattice_protocol::conversation::{ConversationEventKind, UserImage};
use lattice_protocol::{Refusal, RefusalKind};
use serde::{Deserialize, Serialize};

use super::agent::{Convo, refuse};
use super::item::Item;
use super::sidecar::BlobKind;

/// The most images on one message.
pub const MAX_IMAGES: usize = 4;
/// The most bytes of one image.
pub const MAX_BYTES: usize = 5 * 1024 * 1024;

/// The reader's sentences.
pub mod words {
    pub const TOO_MANY: &str = "Attach at most 4 images to a message.";
    pub const NOT_AN_IMAGE: &str = "Only PNG and JPEG images can be attached.";
    pub const TOO_LARGE: &str = "An attached image can be at most 5 MB.";
    pub const PLAIN: &str = "Images go to the agent: attach a folder to this chat (or use Agent mode with the browser on), then send them.";
    pub const CANNOT_SEE: &str = "This model did not accept images earlier in this session; choose another model, or send without them.";
    pub const RUNNING: &str = "Images can be sent once the agent's turn has ended; the words alone can wait in the queue.";
    pub const NOT_SAVED: &str = "Lattice could not save the attached images, so nothing was sent.";
    pub const NOT_CONFIRMED: &str =
        "The images were not sent: they go to a model off this PC only when you confirm.";
}

/// One stored image, as its record names it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredImage {
    pub media_type: String,
    pub sha256: String,
    pub bytes: u64,
}

/// One image, checked and decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decoded {
    pub media_type: &'static str,
    pub bytes: Vec<u8>,
}

/// The type `bytes` begin as: PNG or JPEG, else nothing.
fn sniff(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("image/jpeg")
    } else {
        None
    }
}

/// Standard base64 (padding optional), refusing anything else.
pub fn decode_base64(text: &str) -> Option<Vec<u8>> {
    fn value(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some(u32::from(c - b'A')),
            b'a'..=b'z' => Some(u32::from(c - b'a') + 26),
            b'0'..=b'9' => Some(u32::from(c - b'0') + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let body = text.trim_end_matches('=');
    if text.len() - body.len() > 2 || body.len() % 4 == 1 {
        return None;
    }
    let mut out = Vec::with_capacity(body.len() / 4 * 3 + 3);
    for chunk in body.as_bytes().chunks(4) {
        let mut n = 0u32;
        for (i, c) in chunk.iter().enumerate() {
            n |= value(*c)? << (18 - 6 * i);
        }
        out.push((n >> 16) as u8);
        if chunk.len() > 2 {
            out.push((n >> 8) as u8);
        }
        if chunk.len() > 3 {
            out.push(n as u8);
        }
    }
    Some(out)
}

/// The attached images, checked: their number, size and type.
pub fn check(images: &[UserImage]) -> Result<Vec<Decoded>, Refusal> {
    if images.len() > MAX_IMAGES {
        return Err(refuse(RefusalKind::Invalid, words::TOO_MANY));
    }
    images
        .iter()
        .map(|image| {
            // Base64 is a third longer than the bytes it holds.
            if image.base64.len() > MAX_BYTES / 3 * 4 + 4 {
                return Err(refuse(RefusalKind::Invalid, words::TOO_LARGE));
            }
            let bytes = decode_base64(&image.base64)
                .ok_or_else(|| refuse(RefusalKind::Invalid, words::NOT_AN_IMAGE))?;
            if bytes.len() > MAX_BYTES {
                return Err(refuse(RefusalKind::Invalid, words::TOO_LARGE));
            }
            match sniff(&bytes) {
                Some(kind) if kind == image.media_type => Ok(Decoded {
                    media_type: kind,
                    bytes,
                }),
                _ => Err(refuse(RefusalKind::Invalid, words::NOT_AN_IMAGE)),
            }
        })
        .collect()
}

/// Write `images` with the user turn `turn`: each a blob as given, then the
/// record, then the event. The images as the model is sent them.
pub(crate) fn store(
    convo: &Convo,
    turn: &str,
    images: &[Decoded],
    at: f64,
) -> Result<Vec<Image>, Refusal> {
    let not_saved = || refuse(RefusalKind::Unavailable, words::NOT_SAVED);
    let sidecar = convo.state().sidecar.clone().ok_or_else(not_saved)?;
    let mut stored = Vec::with_capacity(images.len());
    for image in images {
        let blob = sidecar
            .put_blob(&image.bytes, BlobKind::Staged)
            .map_err(|_| not_saved())?;
        stored.push(StoredImage {
            media_type: image.media_type.to_owned(),
            sha256: blob.sha256,
            bytes: blob.bytes,
        });
    }
    let bytes = stored.iter().map(|image| image.bytes).sum();
    let count = stored.len() as u32;
    sidecar
        .append(&Item::ImagesAttached {
            turn: turn.to_owned(),
            images: stored,
            at,
        })
        .map_err(|_| not_saved())?;
    convo.log.push(ConversationEventKind::ImagesAttached {
        turn: turn.to_owned(),
        count,
        bytes,
    });
    Ok(images.iter().map(model_image).collect())
}

fn model_image(image: &Decoded) -> Image {
    Image {
        media_type: image.media_type.to_owned(),
        base64: crate::browser::ws::base64(&image.bytes),
    }
}

/// The images recorded with the user turn `turn`, as the model is sent them
/// (a picture whose blob cannot be read is left out).
pub fn of_turn(
    items: &[Item],
    turn: &str,
    read_blob: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Vec<Image> {
    items
        .iter()
        .filter_map(|item| match item {
            Item::ImagesAttached {
                turn: t, images, ..
            } if t == turn => Some(images),
            _ => None,
        })
        .flatten()
        .filter_map(|image| {
            let bytes = read_blob(&image.sha256)?;
            Some(Image {
                media_type: image.media_type.clone(),
                base64: crate::browser::ws::base64(&bytes),
            })
        })
        .collect()
}

/// How many images the record names with each user turn.
pub fn counts(items: &[Item]) -> std::collections::HashMap<String, usize> {
    let mut out = std::collections::HashMap::new();
    for item in items {
        if let Item::ImagesAttached { turn, images, .. } = item {
            *out.entry(turn.clone()).or_default() += images.len();
        }
    }
    out
}

/// What a later turn reads in place of a message's images.
pub fn note(count: usize) -> String {
    format!(
        "[{count} image{} attached to this message; not shown again]",
        if count == 1 { " was" } else { "s were" }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_reads_back_what_the_encoder_wrote_and_refuses_the_rest() {
        for n in 0..40 {
            let data: Vec<u8> = (0..n).map(|i| (i * 37 + 11) as u8).collect();
            let text = crate::browser::ws::base64(&data);
            assert_eq!(decode_base64(&text), Some(data.clone()), "{n}");
            assert_eq!(decode_base64(text.trim_end_matches('=')), Some(data));
        }
        for bad in ["a", "ab!c", "abc===", "a b="] {
            assert_eq!(decode_base64(bad), None, "{bad}");
        }
    }

    #[test]
    fn only_png_and_jpeg_of_the_type_they_state_are_taken() {
        let png = b"\x89PNG\r\n\x1a\nrest".to_vec();
        let jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0, 1, 2];
        let image = |media_type: &str, bytes: &[u8]| UserImage {
            media_type: media_type.into(),
            base64: crate::browser::ws::base64(bytes),
        };
        let ok = check(&[image("image/png", &png), image("image/jpeg", &jpeg)]).unwrap();
        assert_eq!(ok[0].media_type, "image/png");
        assert_eq!(ok[1].bytes, jpeg);
        for (bad, why) in [
            (vec![image("image/jpeg", &png)], words::NOT_AN_IMAGE),
            (vec![image("image/gif", b"GIF89a")], words::NOT_AN_IMAGE),
            (
                vec![image("image/png", &vec![0x89; MAX_BYTES + 1])],
                words::TOO_LARGE,
            ),
            (
                vec![image("image/png", &png); MAX_IMAGES + 1],
                words::TOO_MANY,
            ),
        ] {
            assert_eq!(check(&bad).unwrap_err().message, why);
        }
        let not_base64 = UserImage {
            media_type: "image/png".into(),
            base64: "not base64!".into(),
        };
        assert_eq!(
            check(&[not_base64]).unwrap_err().message,
            words::NOT_AN_IMAGE
        );
    }
}
