//! `--screenshot`: write the window's pixels to a PNG.
//!
//! Invariants: the file is written only from a well-formed capture (the pixel
//! buffer is exactly `width x height x 4` bytes of RGBA), and a failure is
//! reported and remembered so the process can exit non-zero, because a
//! verification script that "took a screenshot" and got nothing must not
//! believe it succeeded.

use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use iced::window::Screenshot;

static FAILED: AtomicBool = AtomicBool::new(false);

/// Remember that a requested screenshot could not be written.
pub fn mark_failed() {
    FAILED.store(true, Ordering::SeqCst);
}

/// True when a requested screenshot could not be written.
pub fn failed() -> bool {
    FAILED.load(Ordering::SeqCst)
}

/// Encode `rgba` (8 bits per channel, row by row) as a PNG.
pub fn encode(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 {
        return Err("the capture is empty".to_string());
    }
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or("the capture is too large")?;
    if rgba.len() != expected {
        return Err(format!(
            "the capture has {} bytes, expected {expected} for {width}x{height}",
            rgba.len()
        ));
    }
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
        writer.write_image_data(rgba).map_err(|e| e.to_string())?;
    }
    Ok(out)
}

/// Write `shot` to `path`, creating its folder when it is missing.
pub fn save(path: &Path, shot: &Screenshot) -> Result<(), String> {
    let png = encode(&shot.rgba, shot.size.width, shot.size.height)?;
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
    }
    fs::write(path, png).map_err(|e| format!("could not write {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced::Size;

    fn decode(bytes: &[u8]) -> (u32, u32, Vec<u8>) {
        let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        buf.truncate(info.buffer_size());
        (info.width, info.height, buf)
    }

    #[test]
    fn a_capture_round_trips_through_png() {
        let pixels: Vec<u8> = (0..3 * 2 * 4).map(|i| (i * 9) as u8).collect();
        let bytes = encode(&pixels, 3, 2).unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let (w, h, decoded) = decode(&bytes);
        assert_eq!((w, h), (3, 2));
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn malformed_captures_are_refused_not_written() {
        assert!(encode(&[0; 12], 3, 2).is_err(), "too few bytes");
        assert!(encode(&[0; 100], 3, 2).is_err(), "too many bytes");
        assert!(encode(&[], 0, 0).is_err());
        assert!(encode(&[0; 4], 0, 1).is_err());
    }

    #[test]
    fn save_writes_a_png_and_makes_its_folder() {
        let dir = std::env::temp_dir().join(format!("lattice-shot-test-{}", std::process::id()));
        let path = dir.join("nested").join("shot.png");
        let shot = Screenshot::new(vec![255u8; 4 * 4 * 4], Size::new(4, 4), 1.0);
        save(&path, &shot).unwrap();
        let (w, h, pixels) = decode(&fs::read(&path).unwrap());
        assert_eq!((w, h), (4, 4));
        assert!(pixels.iter().all(|&b| b == 255));
        let _ = fs::remove_dir_all(&dir);
        // A capture whose size does not match its bytes is not written.
        let bad = Screenshot::new(vec![0u8; 10], Size::new(4, 4), 1.0);
        assert!(save(&dir.join("bad.png"), &bad).is_err());
        assert!(!dir.join("bad.png").exists());
    }
}
