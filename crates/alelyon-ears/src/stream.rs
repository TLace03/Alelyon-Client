//! One source's speech, turned into events as it is spoken.
//!
//! 16 kHz mono audio goes in, in pieces of any size. Every 30 ms frame feeds the speech detector
//! (`vad`). While nobody is speaking the last 270 ms are kept, so an utterance starts with the sound that
//! proved it was speech and a little before it. During speech, every partial interval, the utterance so far
//! is read again and the reading goes through local agreement (`agreement`): settled words come out as
//! `Partial.stable`, the rest as `Partial.settling`. When the detector ends the utterance, one last reading
//! settles everything and comes out as a `Final` with word times on the stream's clock.
//!
//! The recogniser is called on the calling thread. The service gives each source its own thread and feeds
//! it from the capture thread through a channel, so a slow reading delays this source's events and loses
//! no audio.

use std::collections::VecDeque;

use crate::agreement::Agreement;
use crate::vad::{self, VadEvent, VoiceActivityDetector, FRAME_SAMPLES};
use crate::whisper::{join, Recognizer, Word};

pub const RATE: u32 = 16_000;
/// Frames kept before speech starts: 180 ms of lead-in plus the 90 ms that proved it was speech.
const PREROLL_FRAMES: usize = 9;
/// Characters of what was said before that go to the recogniser as context.
const CONTEXT_CHARS: usize = 200;
/// A reading of less audio than this is not worth a call.
const MIN_READ_S: f64 = 0.4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Mic,
    Pc,
    File,
}

impl Source {
    pub fn name(self) -> &'static str {
        match self {
            Source::Mic => "mic",
            Source::Pc => "pc",
            Source::File => "file",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// Speech began `at` seconds into the stream.
    SpeechStart { source: Source, utterance: u64, at: f64 },
    /// The utterance so far: words that have settled, and words still settling.
    Partial { source: Source, utterance: u64, stable: String, settling: String },
    /// The utterance is over. Word times are seconds on the stream's clock.
    Final { source: Source, utterance: u64, text: String, start: f64, end: f64, words: Vec<Word> },
    /// It sounded like speech but was too short, or nothing was recognised in it.
    Dropped { source: Source, utterance: u64 },
    /// A reading failed; `stage` says which.
    Error { source: Source, utterance: u64, stage: &'static str, message: String },
}

/// The lowest level at which live speech starts (30 ms RMS), below the detector's 0.020. Measured 2026-10-03 on a
/// headset microphone (Razer, at its Windows gain): a quiet room read 0.0001 and the speech 0.014 at
/// the 90th percentile and 0.028 at the 99th, so at 0.020 nothing started in 30 s of talking, and at 0.004
/// all three sentences were heard. The gates still sit above the room's measured noise (four times it to start),
/// so a noisy room raises them on its own: the C920 webcam microphone's room (0.0032) gives 0.0128.
pub const LIVE_START_RMS: f32 = 0.004;
/// ...and where it counts as silence again, in the loop's proportion (0.008 to 0.020).
pub const LIVE_CONTINUE_RMS: f32 = LIVE_START_RMS * vad::CONTINUE_RMS / vad::START_RMS;

#[derive(Clone, Debug)]
pub struct Config {
    pub source: Source,
    /// Read the utterance again during speech. Off for files, where only finals matter.
    pub partials: bool,
    /// Always offered to the recogniser before the context, e.g. the wake word's spelling.
    pub base_prompt: Option<String>,
    /// The detector's floors: where speech starts and where it counts as silence (30 ms RMS).
    pub floors: (f32, f32),
}

impl Config {
    pub fn live(source: Source) -> Self {
        Self { source, partials: true, base_prompt: None, floors: (LIVE_START_RMS, LIVE_CONTINUE_RMS) }
    }

    pub fn file() -> Self {
        Self { source: Source::File, partials: false, base_prompt: None, floors: (vad::START_RMS, vad::CONTINUE_RMS) }
    }
}

pub struct Transcriber<R: Recognizer> {
    recognizer: R,
    config: Config,
    vad: VoiceActivityDetector,
    /// Samples received so far, the stream's clock.
    clock: u64,
    frame: Vec<f32>,
    preroll: VecDeque<Vec<f32>>,
    utterance: u64,
    in_utterance: bool,
    audio: Vec<f32>,
    start_sample: u64,
    agreement: Agreement,
    speaking: bool,
    /// The end of what was said before: context for the next reading.
    context: String,
    /// The stream's clock at the last reading in progress, and the audio that must pass before the next.
    last_partial: u64,
    partial_gap_s: f64,
}

impl<R: Recognizer> Transcriber<R> {
    pub fn new(recognizer: R, config: Config) -> Self {
        let mut vad = VoiceActivityDetector::default();
        (vad.start_rms, vad.continue_rms) = config.floors;
        Self {
            recognizer,
            config,
            vad,
            clock: 0,
            frame: Vec::with_capacity(FRAME_SAMPLES),
            preroll: VecDeque::with_capacity(PREROLL_FRAMES + 1),
            utterance: 0,
            in_utterance: false,
            audio: Vec::new(),
            start_sample: 0,
            agreement: Agreement::default(),
            speaking: false,
            context: String::new(),
            last_partial: 0,
            partial_gap_s: f64::from(vad::PARTIAL_INTERVAL_S),
        }
    }

    pub fn recognizer(&self) -> &R {
        &self.recognizer
    }

    /// Seconds of audio received.
    pub fn elapsed(&self) -> f64 {
        self.clock as f64 / f64::from(RATE)
    }

    pub fn in_speech(&self) -> bool {
        self.in_utterance
    }

    /// Sinai is speaking: raise the bar for starting speech, as the loop does.
    pub fn set_speaking(&mut self, on: bool) {
        self.speaking = on;
    }

    /// The lowest levels at which speech starts and continues (30 ms RMS). The loop's 0.020 and 0.008 suit a
    /// microphone near the mouth at a normal gain; a quiet one needs lower floors. The gates still sit above the
    /// room's measured noise.
    pub fn set_floors(&mut self, start_rms: f32, continue_rms: f32) {
        self.vad.start_rms = start_rms;
        self.vad.continue_rms = continue_rms;
    }

    /// Feed 16 kHz mono audio; returns what it caused.
    pub fn push(&mut self, audio: &[f32]) -> Vec<Event> {
        let mut events = Vec::new();
        for &s in audio {
            self.frame.push(s);
            if self.frame.len() == FRAME_SAMPLES {
                let frame = std::mem::replace(&mut self.frame, Vec::with_capacity(FRAME_SAMPLES));
                self.clock += FRAME_SAMPLES as u64;
                self.frame_done(frame, &mut events);
            }
        }
        events
    }

    /// The stream is over: an utterance still open is finished as it stands.
    pub fn finish(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        if !self.frame.is_empty() {
            let frame = std::mem::take(&mut self.frame);
            self.clock += frame.len() as u64;
            if self.in_utterance {
                self.audio.extend_from_slice(&frame);
            }
        }
        if self.in_utterance {
            let spoken = self.vad.speech_duration_s() - self.vad.quiet_s();
            self.vad.reset();
            if spoken >= vad::MIN_UTTERANCE_S {
                self.final_reading(&mut events);
            } else {
                self.drop_utterance(&mut events);
            }
        }
        events
    }

    fn frame_done(&mut self, frame: Vec<f32>, events: &mut Vec<Event>) {
        let rms = vad::rms(&frame);
        if self.in_utterance {
            self.audio.extend_from_slice(&frame);
        } else {
            self.preroll.push_back(frame);
            if self.preroll.len() > PREROLL_FRAMES {
                self.preroll.pop_front();
            }
        }
        let was_in_speech = self.vad.in_speech;
        for event in self.vad.push(rms, self.speaking) {
            match event {
                VadEvent::SpeechStart => self.begin(events),
                VadEvent::PartialDue => {
                    if self.config.partials {
                        self.partial_reading(events);
                    }
                }
                VadEvent::SpeechEnd => self.final_reading(events),
            }
        }
        // The detector discards a too-short burst without an event: the utterance it opened is dropped.
        if was_in_speech && !self.vad.in_speech && self.in_utterance {
            self.drop_utterance(events);
        }
    }

    fn begin(&mut self, events: &mut Vec<Event>) {
        self.utterance += 1;
        self.in_utterance = true;
        self.audio = self.preroll.drain(..).flatten().collect();
        self.start_sample = self.clock - self.audio.len() as u64;
        self.agreement = Agreement::default();
        events.push(Event::SpeechStart { source: self.config.source, utterance: self.utterance, at: self.seconds(self.start_sample) });
    }

    fn seconds(&self, sample: u64) -> f64 {
        sample as f64 / f64::from(RATE)
    }

    fn prompt(&self) -> Option<String> {
        let mut p = String::new();
        if let Some(base) = &self.config.base_prompt {
            p.push_str(base);
        }
        if !self.context.is_empty() {
            if !p.is_empty() {
                p.push(' ');
            }
            p.push_str(&self.context);
        }
        (!p.is_empty()).then_some(p)
    }

    fn read(&mut self) -> Result<Vec<Word>, String> {
        let prompt = self.prompt();
        self.recognizer.transcribe(&self.audio, prompt.as_deref()).map(|t| t.words())
    }

    /// A reading of the utterance so far. Readings wait for one another: after one that took r seconds, the
    /// next waits for at least 2r seconds of audio (never less than the loop's 0.6 s). Measured 2026-10-03 on
    /// the CPU test model: readings of about 1.1 s every 0.6 s put the captions 9 s behind the speaker within
    /// 30 s, and the audio still queued at the end was never read. On the RX a reading takes 0.16-0.32 s, so
    /// the cadence stays at 0.6 s there.
    fn partial_reading(&mut self, events: &mut Vec<Event>) {
        if (self.audio.len() as f64) < MIN_READ_S * f64::from(RATE) {
            return;
        }
        if self.seconds(self.clock - self.last_partial) < self.partial_gap_s {
            return;
        }
        let began = std::time::Instant::now();
        let reading = self.read();
        let took = began.elapsed().as_secs_f64();
        self.last_partial = self.clock;
        self.partial_gap_s = (2.0 * took).max(f64::from(vad::PARTIAL_INTERVAL_S));
        match reading {
            Ok(words) => {
                let (_, settling) = self.agreement.insert(words);
                let stable = join(self.agreement.committed());
                let settling = join(&settling);
                self.vad.unfinished = vad::sounds_unfinished(&format!("{stable} {settling}"));
                events.push(Event::Partial { source: self.config.source, utterance: self.utterance, stable, settling });
            }
            Err(message) => {
                events.push(Event::Error { source: self.config.source, utterance: self.utterance, stage: "partial", message })
            }
        }
    }

    fn final_reading(&mut self, events: &mut Vec<Event>) {
        let reading = self.read();
        let words = match reading {
            // The reading of the whole utterance has the most context, so it is the record, even where it
            // differs from words that settled earlier: settling keeps live captions still, the final is the
            // best reading there is (the window replaces the line, and Sinai hears this one).
            Ok(words) => words.into_iter().filter(|w| !w.key().is_empty()).collect(),
            Err(message) => {
                events.push(Event::Error { source: self.config.source, utterance: self.utterance, stage: "final", message });
                // What had settled, and what was still settling, is the best record there is.
                self.agreement.flush();
                self.agreement.committed().to_vec()
            }
        };
        let offset = self.seconds(self.start_sample);
        let end = self.seconds(self.start_sample + self.audio.len() as u64);
        let text = join(&words);
        if text.is_empty() {
            self.drop_utterance(events);
            return;
        }
        let words: Vec<Word> = words.iter().map(|w| w.shifted(offset)).collect();
        self.remember(&text);
        events.push(Event::Final { source: self.config.source, utterance: self.utterance, text, start: offset, end, words });
        self.close();
    }

    fn drop_utterance(&mut self, events: &mut Vec<Event>) {
        events.push(Event::Dropped { source: self.config.source, utterance: self.utterance });
        self.close();
    }

    fn close(&mut self) {
        self.in_utterance = false;
        self.audio.clear();
        self.agreement = Agreement::default();
    }

    fn remember(&mut self, text: &str) {
        let mut context = format!("{} {}", self.context, text).trim().to_string();
        if context.chars().count() > CONTEXT_CHARS {
            let skip = context.chars().count() - CONTEXT_CHARS;
            context = context.chars().skip(skip).collect();
            // start on a word boundary
            if let Some(space) = context.find(' ') {
                context = context[space + 1..].to_string();
            }
        }
        self.context = context;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whisper::{transcript_of, Scripted, Transcript};

    fn tone(seconds: f64, amp: f32) -> Vec<f32> {
        let n = (seconds * f64::from(RATE)) as usize;
        (0..n).map(|i| if i % 2 == 0 { amp } else { -amp }).collect()
    }

    fn feed<R: Recognizer>(t: &mut Transcriber<R>, parts: &[(f64, f32)]) -> Vec<Event> {
        parts.iter().flat_map(|&(s, a)| t.push(&tone(s, a))).collect()
    }

    fn finals(events: &[Event]) -> Vec<String> {
        events.iter().filter_map(|e| match e { Event::Final { text, .. } => Some(text.clone()), _ => None }).collect()
    }

    #[test]
    fn an_utterance_streams_partials_then_one_final_on_the_stream_clock() {
        let script = vec![
            Ok(transcript_of("he hoped", 0.3, 0.3)),
            Ok(transcript_of("he hoped there", 0.3, 0.3)),
            Ok(transcript_of("he hoped there would be stew", 0.3, 0.3)),
        ];
        let mut t = Transcriber::new(Scripted::new(script), Config::live(Source::Mic));
        // 1.5 s of speech holds two partial readings; words without a full stop sound unfinished, which
        // holds the turn open for 1.8 s of quiet rather than 0.9 (as in the loop), so 2.5 s of quiet follows.
        let events = feed(&mut t, &[(2.0, 0.0005), (1.5, 0.1), (2.5, 0.0005)]);
        let kinds: Vec<&str> = events
            .iter()
            .map(|e| match e {
                Event::SpeechStart { .. } => "start",
                Event::Partial { .. } => "partial",
                Event::Final { .. } => "final",
                Event::Dropped { .. } => "dropped",
                Event::Error { .. } => "error",
            })
            .collect();
        assert_eq!(kinds, ["start", "partial", "partial", "final"]);
        match &events[2] {
            Event::Partial { stable, settling, .. } => {
                assert_eq!(stable, "he hoped");
                assert_eq!(settling, "there");
            }
            other => panic!("{other:?}"),
        }
        match &events[3] {
            Event::Final { text, start, words, .. } => {
                assert_eq!(text, "he hoped there would be stew");
                // speech started 2.0 s in; the utterance opens 270 ms of preroll before the 90 ms proof
                assert!((*start - (2.0 + 0.09 - 0.27)).abs() < 0.031, "{start}");
                assert!((words[0].start - (start + 0.3)).abs() < 1e-9, "word times move onto the stream clock");
            }
            other => panic!("{other:?}"),
        }
    }

    /// Answers every reading with the same words, after `delay`.
    struct Slow {
        delay: std::time::Duration,
    }

    impl Recognizer for Slow {
        fn transcribe(&mut self, _audio: &[f32], _prompt: Option<&str>) -> Result<Transcript, String> {
            std::thread::sleep(self.delay);
            Ok(transcript_of("still talking", 0.3, 0.3))
        }
    }

    fn partials(events: &[Event]) -> usize {
        events.iter().filter(|e| matches!(e, Event::Partial { .. })).count()
    }

    #[test]
    fn slow_readings_space_out_the_readings_in_progress() {
        let speech = [(1.0, 0.0005), (4.2, 0.1)];
        let mut fast = Transcriber::new(Slow { delay: std::time::Duration::ZERO }, Config::live(Source::Mic));
        let quick = partials(&feed(&mut fast, &speech));
        assert!(quick >= 5, "a reading every 0.6 s of speech: {quick}");
        // a reading of 0.4 s makes the next wait for 0.8 s of audio, so every other request is skipped
        let mut slow = Transcriber::new(Slow { delay: std::time::Duration::from_millis(400) }, Config::live(Source::Mic));
        let paced = partials(&feed(&mut slow, &speech));
        assert!(paced >= 2 && paced <= (quick + 1) / 2, "{paced} readings against {quick}");
    }

    #[test]
    fn live_speech_starts_at_the_lower_floor_and_files_keep_the_loops() {
        let quiet_speaker = [(1.0, 0.0001), (1.0, 0.01), (2.5, 0.0001)];
        let mut live = Transcriber::new(Scripted::new(vec![Ok(transcript_of("hello", 0.3, 0.3)); 4]), Config::live(Source::Mic));
        assert_eq!(finals(&feed(&mut live, &quiet_speaker)), ["hello"], "speech at 0.01 is heard live");
        let mut file = Transcriber::new(Scripted::new(vec![]), Config::file());
        assert!(feed(&mut file, &quiet_speaker).is_empty(), "the file transcriber keeps the loop's 0.020");
    }

    #[test]
    fn the_final_reading_corrects_a_word_that_settled_wrongly() {
        let script = vec![
            Ok(transcript_of("stew four", 0.3, 0.3)),
            Ok(transcript_of("stew four dinner", 0.3, 0.3)),
            Ok(transcript_of("stew for dinner", 0.3, 0.3)),
        ];
        let mut t = Transcriber::new(Scripted::new(script), Config::live(Source::Mic));
        let events = feed(&mut t, &[(1.0, 0.0005), (1.5, 0.1), (2.5, 0.0005)]);
        let settled: Vec<String> = events
            .iter()
            .filter_map(|e| match e { Event::Partial { stable, .. } => Some(stable.clone()), _ => None })
            .collect();
        assert_eq!(settled.last().map(String::as_str), Some("stew four"), "two readings agreed on the wrong word");
        assert_eq!(finals(&events), ["stew for dinner"], "the whole-utterance reading is the record");
    }

    #[test]
    fn a_cough_is_dropped_without_a_reading() {
        let mut t = Transcriber::new(Scripted::new(vec![]), Config::live(Source::Mic));
        let events = feed(&mut t, &[(1.0, 0.0005), (0.15, 0.1), (2.0, 0.0005)]);
        assert!(matches!(events.as_slice(), [Event::SpeechStart { .. }, Event::Dropped { .. }]), "{events:?}");
        assert!(t.recognizer().calls.is_empty(), "nothing was read");
    }

    #[test]
    fn a_failed_final_reading_keeps_what_had_settled() {
        let script = vec![
            Ok(transcript_of("open the", 0.3, 0.3)),
            Ok(transcript_of("open the door", 0.3, 0.3)),
            Err("whisper-server did not finish its answer".to_string()),
        ];
        let mut t = Transcriber::new(Scripted::new(script), Config::live(Source::Mic));
        let events = feed(&mut t, &[(1.0, 0.0005), (1.5, 0.1), (2.5, 0.0005)]);
        assert!(events.iter().any(|e| matches!(e, Event::Error { stage: "final", .. })));
        assert_eq!(finals(&events), ["open the door"], "the settled words and the last settling ones survive");
    }

    #[test]
    fn what_was_said_before_becomes_the_next_prompt() {
        let script = vec![Ok(transcript_of("first part", 0.3, 0.3)), Ok(transcript_of("second part", 0.3, 0.3))];
        let mut config = Config::file();
        config.base_prompt = Some("Sinai".into());
        let mut t = Transcriber::new(Scripted::new(script), config);
        let mut events = feed(&mut t, &[(0.5, 0.0005), (0.6, 0.1), (1.2, 0.0005), (0.6, 0.1), (1.2, 0.0005)]);
        events.extend(t.finish());
        assert_eq!(finals(&events), ["first part", "second part"]);
        let prompts: Vec<Option<String>> = t.recognizer().calls.iter().map(|(_, p)| p.clone()).collect();
        assert_eq!(prompts, [Some("Sinai".into()), Some("Sinai first part".into())]);
    }

    #[test]
    fn a_stream_that_ends_mid_utterance_is_finished() {
        let mut t = Transcriber::new(Scripted::new(vec![Ok(transcript_of("unfinished thought", 0.2, 0.3))]), Config::file());
        let mut events = feed(&mut t, &[(0.5, 0.0005), (1.0, 0.1)]);
        assert!(finals(&events).is_empty());
        events.extend(t.finish());
        assert_eq!(finals(&events), ["unfinished thought"]);
    }

    #[test]
    fn nothing_recognised_is_dropped_not_reported_as_empty_speech() {
        let mut t = Transcriber::new(Scripted::new(vec![Ok(Transcript::default())]), Config::file());
        let events = feed(&mut t, &[(0.5, 0.0005), (1.0, 0.1), (1.2, 0.0005)]);
        assert!(events.iter().any(|e| matches!(e, Event::Dropped { .. })));
        assert!(finals(&events).is_empty());
    }

    #[test]
    fn sinai_speaking_keeps_its_own_voice_from_starting_an_utterance() {
        let mut t = Transcriber::new(Scripted::new(vec![]), Config::live(Source::Mic));
        t.set_speaking(true);
        let events = feed(&mut t, &[(1.0, 0.0005), (1.0, 0.03)]);
        assert!(events.is_empty(), "{events:?}");
    }
}
