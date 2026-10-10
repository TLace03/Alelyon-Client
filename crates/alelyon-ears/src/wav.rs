//! WAV in and out: the recogniser takes 16-bit mono WAV, and WAV is the one file format read without
//! Media Foundation.
//!
//! The reader takes integer PCM at 8, 16, 24 and 32 bits, IEEE float at 32 and 64 bits, and the extensible
//! header that wraps either; any other encoding is refused by name rather than read as noise. Chunks it does
//! not need (LIST, fact, cue, ...) are skipped, odd-sized chunks included.

use std::fmt;

#[derive(Debug, PartialEq)]
pub enum WavError {
    NotWav,
    Truncated,
    NoFormat,
    NoData,
    Unsupported(String),
}

impl fmt::Display for WavError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WavError::NotWav => write!(f, "not a WAV file (no RIFF/WAVE header)"),
            WavError::Truncated => write!(f, "the WAV file ends inside a chunk"),
            WavError::NoFormat => write!(f, "the WAV file has no fmt chunk before its data"),
            WavError::NoData => write!(f, "the WAV file has no data chunk"),
            WavError::Unsupported(what) => write!(f, "unsupported WAV encoding: {what}"),
        }
    }
}

impl std::error::Error for WavError {}

/// Decoded audio: interleaved samples in [-1, 1].
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub channels: u16,
    pub rate: u32,
}

impl Audio {
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / usize::from(self.channels)
        }
    }

    pub fn seconds(&self) -> f64 {
        if self.rate == 0 {
            0.0
        } else {
            self.frames() as f64 / f64::from(self.rate)
        }
    }
}

/// `samples` (mono, [-1, 1]) as a 16-bit PCM WAV file at `rate`.
pub fn encode_pcm16(samples: &[f32], rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

const PCM: u16 = 1;
const IEEE_FLOAT: u16 = 3;
const EXTENSIBLE: u16 = 0xFFFE;

fn u16_at(b: &[u8], i: usize) -> Result<u16, WavError> {
    b.get(i..i + 2).map(|s| u16::from_le_bytes([s[0], s[1]])).ok_or(WavError::Truncated)
}

fn u32_at(b: &[u8], i: usize) -> Result<u32, WavError> {
    b.get(i..i + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])).ok_or(WavError::Truncated)
}

/// Read a WAV file's samples.
pub fn decode(bytes: &[u8]) -> Result<Audio, WavError> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(WavError::NotWav);
    }
    let mut at = 12;
    let mut format: Option<(u16, u16, u32, u16)> = None; // (encoding, channels, rate, bits)
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let size = u32_at(bytes, at + 4)? as usize;
        let body = at + 8;
        match id {
            b"fmt " => {
                if size < 16 || body + size > bytes.len() {
                    return Err(WavError::Truncated);
                }
                let mut encoding = u16_at(bytes, body)?;
                let channels = u16_at(bytes, body + 2)?;
                let rate = u32_at(bytes, body + 4)?;
                let bits = u16_at(bytes, body + 14)?;
                if encoding == EXTENSIBLE {
                    // cbSize, valid bits, channel mask, then the sub-format GUID whose first two bytes are
                    // the real encoding tag.
                    if size < 40 {
                        return Err(WavError::Truncated);
                    }
                    encoding = u16_at(bytes, body + 24)?;
                }
                format = Some((encoding, channels, rate, bits));
            }
            b"data" => {
                let (encoding, channels, rate, bits) = format.ok_or(WavError::NoFormat)?;
                // A data chunk whose stated size runs past the file (a recorder that never finalised its
                // header) is read to the end of the file.
                let end = (body + size).min(bytes.len());
                let samples = samples_of(&bytes[body..end], encoding, bits)?;
                if channels == 0 || rate == 0 {
                    return Err(WavError::Unsupported(format!("{channels} channels at {rate} Hz")));
                }
                return Ok(Audio { samples, channels, rate });
            }
            _ => {}
        }
        // Chunks are padded to an even length.
        at = body + size + (size & 1);
    }
    if format.is_none() {
        Err(WavError::NoFormat)
    } else {
        Err(WavError::NoData)
    }
}

fn samples_of(data: &[u8], encoding: u16, bits: u16) -> Result<Vec<f32>, WavError> {
    let out = match (encoding, bits) {
        (PCM, 8) => data.iter().map(|&b| (f32::from(b) - 128.0) / 128.0).collect(),
        (PCM, 16) => data.chunks_exact(2).map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0).collect(),
        (PCM, 24) => data
            .chunks_exact(3)
            .map(|c| {
                let v = i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8;
                v as f32 / 8_388_608.0
            })
            .collect(),
        (PCM, 32) => data
            .chunks_exact(4)
            .map(|c| (f64::from(i32::from_le_bytes([c[0], c[1], c[2], c[3]])) / 2_147_483_648.0) as f32)
            .collect(),
        (IEEE_FLOAT, 32) => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        (IEEE_FLOAT, 64) => data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        (e, b) => return Err(WavError::Unsupported(format!("format tag {e:#06x} at {b} bits"))),
    };
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(encoding: u16, channels: u16, rate: u32, bits: u16, data: &[u8], extra: &[u8]) -> Vec<u8> {
        let block = channels * bits / 8;
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&encoding.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&rate.to_le_bytes());
        fmt.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
        fmt.extend_from_slice(&block.to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());
        fmt.extend_from_slice(extra);
        let mut out = b"RIFF\0\0\0\0WAVE".to_vec();
        // an odd-sized chunk before fmt, padded, which the reader must step over
        out.extend_from_slice(b"LIST");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(b"abc\0");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        out.extend_from_slice(&fmt);
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn what_is_written_reads_back() {
        let samples = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.25];
        let audio = decode(&encode_pcm16(&samples, 16_000)).unwrap();
        assert_eq!((audio.channels, audio.rate), (1, 16_000));
        for (a, b) in audio.samples.iter().zip(&samples) {
            assert!((a - b).abs() < 1.0 / 16_000.0, "{a} vs {b}");
        }
        assert!((audio.seconds() - 6.0 / 16_000.0).abs() < 1e-9);
    }

    #[test]
    fn twenty_four_bit_and_float_and_extensible_read() {
        // 24-bit: 0x400000 is +0.5, 0xC00000 is -0.5
        let a = decode(&header(PCM, 1, 48_000, 24, &[0, 0, 0x40, 0, 0, 0xC0], &[])).unwrap();
        assert_eq!(a.samples, vec![0.5, -0.5]);
        let mut f = Vec::new();
        f.extend_from_slice(&0.75f32.to_le_bytes());
        f.extend_from_slice(&(-0.25f32).to_le_bytes());
        let b = decode(&header(IEEE_FLOAT, 2, 44_100, 32, &f, &[])).unwrap();
        assert_eq!((b.samples.clone(), b.channels, b.frames()), (vec![0.75, -0.25], 2, 1));
        // WAVE_FORMAT_EXTENSIBLE wrapping 16-bit PCM: cbSize 22, valid bits, mask, sub-format GUID
        let mut ext = Vec::new();
        ext.extend_from_slice(&22u16.to_le_bytes());
        ext.extend_from_slice(&16u16.to_le_bytes());
        ext.extend_from_slice(&3u32.to_le_bytes());
        ext.extend_from_slice(&1u16.to_le_bytes());
        ext.extend_from_slice(&[0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71]);
        let c = decode(&header(EXTENSIBLE, 1, 16_000, 16, &16384i16.to_le_bytes(), &ext)).unwrap();
        assert_eq!(c.samples, vec![0.5]);
    }

    #[test]
    fn what_cannot_be_read_is_refused_by_name() {
        assert_eq!(decode(b"not a wav file at all"), Err(WavError::NotWav));
        assert!(matches!(decode(&header(2, 1, 8_000, 4, &[0; 8], &[])), Err(WavError::Unsupported(_))));
        let mut no_data = header(PCM, 1, 16_000, 16, &[], &[]);
        no_data.truncate(no_data.len() - 8);
        assert_eq!(decode(&no_data), Err(WavError::NoData));
    }

    #[test]
    fn an_unfinalised_data_size_is_read_to_the_end_of_the_file() {
        let mut bytes = header(PCM, 1, 16_000, 16, &[0, 0x40, 0, 0xC0], &[]);
        let at = bytes.len() - 4 - 4;
        bytes[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(decode(&bytes).unwrap().samples, vec![0.5, -0.5]);
    }
}
