//! The screen as the model sees it: scaled by area averaging to the picture's
//! size ([`super::policy::fit`]) and encoded as a PNG, then base64.

use lattice_agents::model::Image;

/// `bgra` (`width` x `height`, the top row first) averaged down to `out_w` x
/// `out_h` RGB pixels.
pub fn scale(bgra: &[u8], width: u32, height: u32, out_w: u32, out_h: u32) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let (ow, oh) = (out_w.max(1) as usize, out_h.max(1) as usize);
    let mut out = vec![0u8; ow * oh * 3];
    if w == 0 || h == 0 || bgra.len() < w * h * 4 {
        return out;
    }
    for oy in 0..oh {
        let y0 = oy * h / oh;
        let y1 = ((oy + 1) * h / oh).clamp(y0 + 1, h);
        for ox in 0..ow {
            let x0 = ox * w / ow;
            let x1 = ((ox + 1) * w / ow).clamp(x0 + 1, w);
            let (mut red, mut green, mut blue, mut count) = (0u32, 0u32, 0u32, 0u32);
            for y in y0..y1 {
                let row = &bgra[(y * w + x0) * 4..(y * w + x1) * 4];
                for pixel in row.chunks_exact(4) {
                    blue += u32::from(pixel[0]);
                    green += u32::from(pixel[1]);
                    red += u32::from(pixel[2]);
                    count += 1;
                }
            }
            let at = (oy * ow + ox) * 3;
            out[at] = (red / count) as u8;
            out[at + 1] = (green / count) as u8;
            out[at + 2] = (blue / count) as u8;
        }
    }
    out
}

/// RGB pixels as a PNG.
pub fn png(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::Fast);
        let mut writer = encoder
            .write_header()
            .map_err(|_| "The screen could not be encoded.".to_owned())?;
        writer
            .write_image_data(rgb)
            .map_err(|_| "The screen could not be encoded.".to_owned())?;
    }
    Ok(bytes)
}

/// The picture of a screen for the model.
pub fn picture(bgra: &[u8], width: u32, height: u32) -> Result<(Image, u32, u32, f64), String> {
    let (out_w, out_h, scale_by) = super::policy::fit(width, height);
    let rgb = scale(bgra, width, height, out_w, out_h);
    let bytes = png(&rgb, out_w, out_h)?;
    Ok((
        Image {
            media_type: "image/png".to_owned(),
            base64: crate::browser::ws::base64(&bytes),
        },
        out_w,
        out_h,
        scale_by,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_block_of_pixels_averages_to_one_and_a_png_is_written() {
        // 2 x 2 BGRA: red, green, blue, white.
        let bgra = [0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0];
        let one = scale(&bgra, 2, 2, 1, 1);
        assert_eq!(one, [127, 127, 127]);
        let same = scale(&bgra, 2, 2, 2, 2);
        assert_eq!(same, [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]);
        let bytes = png(&same, 2, 2).unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let decoder = png::Decoder::new(std::io::Cursor::new(&bytes));
        let mut reader = decoder.read_info().unwrap();
        let mut back = vec![0; reader.output_buffer_size().unwrap()];
        reader.next_frame(&mut back).unwrap();
        assert_eq!(back, same, "it decodes to the same pixels");
    }

    #[test]
    fn a_large_screen_becomes_a_picture_of_the_fitted_size() {
        let (width, height) = (2560u32, 1440u32);
        let bgra = vec![200u8; width as usize * height as usize * 4];
        let (image, w, h, scale_by) = picture(&bgra, width, height).unwrap();
        assert_eq!((w, h, scale_by), (1280, 720, 0.5));
        assert_eq!(image.media_type, "image/png");
        assert!(image.base64.starts_with("iVBORw0KGgo"));
    }
}
