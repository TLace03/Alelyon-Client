//! Audio files Windows can play, decoded by Media Foundation: MP3, M4A/AAC, WMA, FLAC and WAV among them.
//!
//! The first audio stream is asked for as 32-bit float at its own rate and channel count; the caller mixes
//! it down and resamples. Nothing here downloads or installs a codec: a format Windows cannot decode is
//! refused with Media Foundation's own reason.

use std::path::Path;

use windows::core::HSTRING;
use windows::Win32::Media::MediaFoundation::{
    IMFAttributes, IMFSample, MFAudioFormat_Float, MFCreateMediaType, MFCreateSourceReaderFromURL, MFMediaType_Audio, MFShutdown,
    MFStartup, MFSTARTUP_FULL, MF_MT_AUDIO_NUM_CHANNELS, MF_MT_AUDIO_SAMPLES_PER_SECOND, MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE,
    MF_SOURCE_READERF_ENDOFSTREAM, MF_SOURCE_READER_ALL_STREAMS, MF_SOURCE_READER_FIRST_AUDIO_STREAM, MF_VERSION,
};
use windows::Win32::System::Com::{CoInitializeEx, CoUninitialize, COINIT_MULTITHREADED};

use crate::wav::Audio;

/// Decode `path`'s first audio stream to interleaved f32.
pub fn decode_file(path: &Path) -> Result<Audio, String> {
    // SAFETY: COM and Media Foundation are started and stopped on this thread around the calls that use them.
    unsafe {
        // Already initialised in another mode on this thread is fine: Media Foundation works in either.
        let com = CoInitializeEx(None, COINIT_MULTITHREADED);
        let started = MFStartup(MF_VERSION, MFSTARTUP_FULL);
        let result = match &started {
            Ok(()) => decode(path),
            Err(e) => Err(format!("Media Foundation would not start: {e}")),
        };
        if started.is_ok() {
            let _ = MFShutdown();
        }
        if com.is_ok() {
            CoUninitialize();
        }
        result
    }
}

unsafe fn decode(path: &Path) -> Result<Audio, String> {
    let name = path.display();
    let url = HSTRING::from(path.as_os_str());
    let reader = MFCreateSourceReaderFromURL(&url, None::<&IMFAttributes>)
        .map_err(|e| format!("{name}: Windows cannot open this as audio ({e})"))?;
    let audio = MF_SOURCE_READER_FIRST_AUDIO_STREAM.0 as u32;
    reader.SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS.0 as u32, false).map_err(|e| format!("{name}: {e}"))?;
    reader.SetStreamSelection(audio, true).map_err(|e| format!("{name}: it has no audio stream ({e})"))?;
    let want = MFCreateMediaType().map_err(|e| format!("{name}: {e}"))?;
    want.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio).map_err(|e| format!("{name}: {e}"))?;
    want.SetGUID(&MF_MT_SUBTYPE, &MFAudioFormat_Float).map_err(|e| format!("{name}: {e}"))?;
    reader
        .SetCurrentMediaType(audio, None, &want)
        .map_err(|e| format!("{name}: Windows has no decoder that turns this into PCM ({e})"))?;
    let got = reader.GetCurrentMediaType(audio).map_err(|e| format!("{name}: {e}"))?;
    let channels = got.GetUINT32(&MF_MT_AUDIO_NUM_CHANNELS).map_err(|e| format!("{name}: no channel count ({e})"))?;
    let rate = got.GetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND).map_err(|e| format!("{name}: no sample rate ({e})"))?;
    if channels == 0 || rate == 0 {
        return Err(format!("{name}: the decoder reports {channels} channels at {rate} Hz"));
    }
    let mut samples = Vec::new();
    loop {
        let mut flags = 0u32;
        let mut sample: Option<IMFSample> = None;
        reader.ReadSample(audio, 0, None, Some(&mut flags), None, Some(&mut sample)).map_err(|e| {
            format!("{name}: decoding stopped at {:.1} s ({e})", samples.len() as f64 / f64::from(rate * channels))
        })?;
        if let Some(s) = sample {
            let buffer = s.ConvertToContiguousBuffer().map_err(|e| format!("{name}: {e}"))?;
            let mut ptr = std::ptr::null_mut();
            let mut len = 0u32;
            buffer.Lock(&mut ptr, None, Some(&mut len)).map_err(|e| format!("{name}: {e}"))?;
            let count = len as usize / 4;
            samples.extend((0..count).map(|i| std::ptr::read_unaligned((ptr as *const f32).add(i))));
            let _ = buffer.Unlock();
        }
        if flags & MF_SOURCE_READERF_ENDOFSTREAM.0 as u32 != 0 {
            break;
        }
    }
    Ok(Audio { samples, channels: channels as u16, rate })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("angel-ears-mf-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_wav_decodes_through_media_foundation_as_through_the_wav_reader() {
        let dir = scratch("wav");
        let path = dir.join("tone.wav");
        let tone: Vec<f32> = (0..16_000).map(|i| ((i as f32) * 0.05).sin() * 0.5).collect();
        std::fs::write(&path, crate::wav::encode_pcm16(&tone, 16_000)).unwrap();
        let decoded = decode_file(&path);
        let _ = std::fs::remove_dir_all(&dir);
        let audio = decoded.unwrap();
        assert_eq!((audio.channels, audio.rate), (1, 16_000));
        assert_eq!(audio.samples.len(), tone.len());
        for (a, b) in audio.samples.iter().zip(&tone) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn a_file_that_is_not_audio_is_refused_with_a_reason() {
        let dir = scratch("bad");
        let path = dir.join("notes.mp3");
        std::fs::write(&path, b"this is not an mp3").unwrap();
        let result = decode_file(&path);
        let _ = std::fs::remove_dir_all(&dir);
        let err = result.unwrap_err();
        assert!(err.contains("notes.mp3"), "{err}");
    }
}
