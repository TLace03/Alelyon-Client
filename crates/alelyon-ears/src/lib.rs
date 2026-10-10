//! The ears: real-time transcription of the microphone, the computer's own audio and audio files, on this PC.
//!
//! Audio is captured (`capture`, Windows' WASAPI) or decoded (`media`, Media Foundation; `wav`), brought to
//! 16 kHz mono (`resample`), cut into utterances (`vad`), read by whisper.cpp's server (`whisper`) while it is
//! spoken, and turned into settled and settling words (`agreement`, `stream`). The service (`server`) hands those
//! words, as events, to the window and to Sinai's hearing. `setup` says where the recogniser program and the
//! speech model are looked for, and what is missing.
//!
//! The same stream serves captions, dictation, the computer's own audio and files for people, and Sinai, which acts
//! only on speech addressed to it.

pub mod agreement;
pub mod events;
pub mod files;
pub mod resample;
pub mod setup;
pub mod stream;
pub mod vad;
pub mod wav;
pub mod whisper;

#[cfg(windows)]
pub mod capture;
#[cfg(windows)]
pub mod media;
#[cfg(windows)]
pub mod recognizer;
#[cfg(windows)]
pub mod server;
