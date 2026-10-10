//! Where speech starts and stops: energy-gated segmentation with hysteresis and a hangover.
//!
//! A port of the voice activity detector Sinai's hearing has always used, constant for constant, so the ears
//! cut an utterance where Sinai's hearing has been cutting it. The thresholds float above a measured noise floor
//! (one fixed number cannot serve a quiet room and a loud one), a run of loud frames proves speech has
//! started, and quiet must last before it has ended, longer when what was said so far sounds unfinished.
//! `tests` mirrors that detector's own test cases.

/// One energy reading per 30 ms of audio.
pub const FRAME_S: f32 = 0.03;
/// Samples in one frame at the ears' 16 kHz.
pub const FRAME_SAMPLES: usize = 480;
/// Speech must exceed this to begin (a floor; the noise-relative gate can raise it).
pub const START_RMS: f32 = 0.020;
/// ...and fall below this to count as silence.
pub const CONTINUE_RMS: f32 = 0.008;
/// Consecutive loud frames before speech starts (90 ms).
pub const START_FRAMES: u32 = 3;
/// Quiet this long ends the utterance...
pub const END_SILENCE_S: f32 = 0.90;
/// ...unless what was said so far sounds unfinished.
pub const HOLD_SILENCE_S: f32 = 1.80;
/// Shorter than this is a cough, not a turn.
pub const MIN_UTTERANCE_S: f32 = 0.35;
/// Hard stop, matching the recorder's buffer and whisper's 30 s window.
pub const MAX_UTTERANCE_S: f32 = 30.0;
/// Transcribe-so-far cadence during speech.
pub const PARTIAL_INTERVAL_S: f32 = 0.60;
/// The gates sit this many times above the room's measured noise floor.
pub const NOISE_START_MULT: f32 = 4.0;
pub const NOISE_CONTINUE_MULT: f32 = 2.0;
/// How fast the floor tracks the room, per quiet frame.
pub const NOISE_EMA: f32 = 0.02;
/// While Sinai is speaking its own voice can reach an open microphone; a higher bar during playback is the
/// cheap guard (acoustic echo cancellation is a recorded gap, as in the loop).
pub const SPEAKING_RMS_MULTIPLIER: f32 = 2.5;

/// What one frame caused, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VadEvent {
    SpeechStart,
    PartialDue,
    SpeechEnd,
}

/// A pause after one of these is a person thinking, not a person finished (`listening.CONTINUATION_WORDS`).
const CONTINUATION_WORDS: &[&str] = &[
    "and", "but", "so", "or", "nor", "because", "that", "which", "if", "when", "while", "where", "with", "to",
    "for", "of", "the", "a", "an", "my", "your", "our", "their", "as", "at", "on", "in", "from", "into",
    "about", "over", "under", "than", "then", "though", "although", "since", "unless", "until", "whether",
    "like", "just", "um", "uh", "er", "hmm", "well", "actually", "basically", "literally", "sort", "kind",
];

/// Does this partial transcript sound like a sentence still in progress? (`listening.sounds_unfinished`)
///
/// Any of three signals holds the turn open: it ends on a word people continue from, on a comma or a dash,
/// or it carries no terminal punctuation at all. Cheap, wrong sometimes, and wrong in the forgiving direction.
pub fn sounds_unfinished(text: &str) -> bool {
    let stripped = text.trim();
    let Some(last) = stripped.chars().last() else {
        return false;
    };
    if ",;:-\u{2014}".contains(last) {
        return true;
    }
    if !".!?".contains(last) {
        return true;
    }
    let trimmed = stripped.trim_end_matches(|c| ".!?\"')".contains(c));
    match trimmed.split_whitespace().last() {
        Some(word) => {
            let word = word.trim_matches(|c| ",.;:!?\"')".contains(c)).to_lowercase();
            CONTINUATION_WORDS.contains(&word.as_str())
        }
        None => false,
    }
}

/// The root mean square of one frame of samples in [-1, 1]; 0 for an empty frame.
pub fn rms(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum: f64 = frame.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum / frame.len() as f64).sqrt() as f32
}

#[derive(Clone, Debug)]
pub struct VoiceActivityDetector {
    pub frame_s: f32,
    pub start_rms: f32,
    pub continue_rms: f32,
    pub start_frames: u32,
    pub end_silence_s: f32,
    pub hold_silence_s: f32,
    pub min_utterance_s: f32,
    pub max_utterance_s: f32,
    pub partial_interval_s: f32,
    pub speaking_multiplier: f32,
    pub in_speech: bool,
    /// Set from the latest partial transcript: holds the turn open while it sounds like it is still going.
    pub unfinished: bool,
    pub noise_floor: f32,
    loud_run: u32,
    quiet_s: f32,
    speech_s: f32,
    since_partial_s: f32,
}

impl Default for VoiceActivityDetector {
    fn default() -> Self {
        Self {
            frame_s: FRAME_S,
            start_rms: START_RMS,
            continue_rms: CONTINUE_RMS,
            start_frames: START_FRAMES,
            end_silence_s: END_SILENCE_S,
            hold_silence_s: HOLD_SILENCE_S,
            min_utterance_s: MIN_UTTERANCE_S,
            max_utterance_s: MAX_UTTERANCE_S,
            partial_interval_s: PARTIAL_INTERVAL_S,
            speaking_multiplier: SPEAKING_RMS_MULTIPLIER,
            in_speech: false,
            unfinished: false,
            noise_floor: 0.0015,
            loud_run: 0,
            quiet_s: 0.0,
            speech_s: 0.0,
            since_partial_s: 0.0,
        }
    }
}

impl VoiceActivityDetector {
    /// Seconds of the current utterance so far, the frames that proved it was speech included.
    pub fn speech_duration_s(&self) -> f32 {
        self.speech_s
    }

    /// Seconds of trailing quiet in the current utterance.
    pub fn quiet_s(&self) -> f32 {
        self.quiet_s
    }

    pub fn end_silence_now_s(&self) -> f32 {
        if self.unfinished {
            self.hold_silence_s
        } else {
            self.end_silence_s
        }
    }

    pub fn reset(&mut self) {
        self.in_speech = false;
        self.unfinished = false;
        self.loud_run = 0;
        self.quiet_s = 0.0;
        self.speech_s = 0.0;
        self.since_partial_s = 0.0;
    }

    /// Feed one frame's RMS; returns the events it caused, in order. `speaking` says whether Sinai is
    /// talking now, which raises the start gate so its own voice is less likely to be heard as speech.
    pub fn push(&mut self, rms: f32, speaking: bool) -> Vec<VadEvent> {
        let mut events = Vec::new();

        // Track the room while nobody is speaking, then sit above it. The clamp stops a loud frame from
        // dragging the floor up with it.
        if !self.in_speech && !speaking {
            let target = rms.min(self.noise_floor * 4.0 + 1e-6);
            self.noise_floor += NOISE_EMA * (target - self.noise_floor);
        }

        let mut start_gate = self.start_rms.max(self.noise_floor * NOISE_START_MULT);
        if speaking {
            // Sinai's own voice from the speakers does not get quieter because a microphone's floor was lowered
            // (the ears' live floor is 0.004): while it talks the gate keeps the loop's own, from START_RMS. With
            // the loop's floors this is the loop's rule unchanged.
            start_gate = start_gate.max(START_RMS) * self.speaking_multiplier;
        }
        let continue_gate = self.continue_rms.max(self.noise_floor * NOISE_CONTINUE_MULT);

        if !self.in_speech {
            if rms >= start_gate {
                self.loud_run += 1;
                if self.loud_run >= self.start_frames {
                    self.in_speech = true;
                    // The frames that proved it was speech are part of the utterance.
                    self.speech_s = self.start_frames as f32 * self.frame_s;
                    self.quiet_s = 0.0;
                    self.since_partial_s = 0.0;
                    events.push(VadEvent::SpeechStart);
                }
            } else {
                self.loud_run = 0;
            }
            return events;
        }

        self.speech_s += self.frame_s;
        self.since_partial_s += self.frame_s;
        self.quiet_s = if rms >= continue_gate { 0.0 } else { self.quiet_s + self.frame_s };

        if self.since_partial_s >= self.partial_interval_s && self.quiet_s == 0.0 {
            self.since_partial_s = 0.0;
            events.push(VadEvent::PartialDue);
        }

        if self.speech_s >= self.max_utterance_s {
            events.push(VadEvent::SpeechEnd);
            self.reset();
            return events;
        }

        if self.quiet_s >= self.end_silence_now_s() {
            // The trailing silence is not part of what was said.
            let spoken = self.speech_s - self.quiet_s;
            self.reset();
            if spoken >= self.min_utterance_s {
                events.push(VadEvent::SpeechEnd);
            }
            // Too short: discarded silently; a cough must not start a turn.
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOUD: f32 = 0.1;
    const QUIET: f32 = 0.001;

    fn feed(vad: &mut VoiceActivityDetector, rms: f32, seconds: f32, speaking: bool) -> Vec<VadEvent> {
        let frames = (seconds / FRAME_S).round() as usize;
        (0..frames).flat_map(|_| vad.push(rms, speaking)).collect()
    }

    #[test]
    fn speech_starts_after_three_loud_frames_and_ends_after_the_hangover() {
        let mut vad = VoiceActivityDetector::default();
        feed(&mut vad, QUIET, 1.0, false);
        assert_eq!(vad.push(LOUD, false), vec![]);
        assert_eq!(vad.push(LOUD, false), vec![]);
        assert_eq!(vad.push(LOUD, false), vec![VadEvent::SpeechStart]);
        let during = feed(&mut vad, LOUD, 1.0, false);
        assert!(during.contains(&VadEvent::PartialDue), "a partial is due every 0.6 s of speech");
        assert!(!during.contains(&VadEvent::SpeechEnd));
        let after = feed(&mut vad, QUIET, 0.87, false);
        assert!(!after.contains(&VadEvent::SpeechEnd), "0.87 s of quiet is a pause, not an end");
        assert_eq!(feed(&mut vad, QUIET, 0.06, false), vec![VadEvent::SpeechEnd]);
        assert!(!vad.in_speech);
    }

    #[test]
    fn a_cough_is_discarded() {
        let mut vad = VoiceActivityDetector::default();
        feed(&mut vad, QUIET, 1.0, false);
        let mut events = feed(&mut vad, LOUD, 0.12, false);
        events.extend(feed(&mut vad, QUIET, 2.0, false));
        assert_eq!(events, vec![VadEvent::SpeechStart], "0.12 s of sound is too short to be a turn");
    }

    #[test]
    fn an_unfinished_sentence_holds_the_turn_open() {
        let mut vad = VoiceActivityDetector::default();
        feed(&mut vad, QUIET, 1.0, false);
        feed(&mut vad, LOUD, 1.0, false);
        vad.unfinished = true;
        assert!(!feed(&mut vad, QUIET, 1.5, false).contains(&VadEvent::SpeechEnd));
        assert!(feed(&mut vad, QUIET, 0.4, false).contains(&VadEvent::SpeechEnd));
    }

    #[test]
    fn the_hard_stop_ends_a_thirty_second_utterance() {
        let mut vad = VoiceActivityDetector::default();
        feed(&mut vad, QUIET, 1.0, false);
        let events = feed(&mut vad, LOUD, 31.0, false);
        assert_eq!(events.iter().filter(|e| **e == VadEvent::SpeechEnd).count(), 1);
    }

    #[test]
    fn a_noisy_room_raises_the_gate() {
        let mut quiet_room = VoiceActivityDetector::default();
        let mut noisy_room = VoiceActivityDetector::default();
        feed(&mut quiet_room, QUIET, 10.0, false);
        feed(&mut noisy_room, 0.008, 10.0, false);
        assert!(noisy_room.noise_floor > quiet_room.noise_floor * 3.0);
        // 0.025 clears the floor gate in the quiet room but not four times the noisy room's floor.
        assert!(feed(&mut quiet_room, 0.025, 0.3, false).contains(&VadEvent::SpeechStart));
        assert!(!feed(&mut noisy_room, 0.025, 0.3, false).contains(&VadEvent::SpeechStart));
    }

    #[test]
    fn sinai_speaking_raises_the_start_gate() {
        let mut vad = VoiceActivityDetector::default();
        feed(&mut vad, QUIET, 1.0, false);
        // 0.03 starts speech normally but not while Sinai speaks (gate 0.020 x 2.5 = 0.050).
        assert!(!feed(&mut vad, 0.03, 0.3, true).contains(&VadEvent::SpeechStart));
        assert!(feed(&mut vad, 0.03, 0.3, false).contains(&VadEvent::SpeechStart));
    }

    #[test]
    fn unfinished_matches_the_loop() {
        assert!(sounds_unfinished("I was thinking that"));
        assert!(sounds_unfinished("I went to the store and"));
        assert!(sounds_unfinished("Well,"));
        assert!(sounds_unfinished("So the plan is \u{2014}"));
        assert!(sounds_unfinished("I bought milk and."), "a continuation word overrides whisper's full stop");
        assert!(!sounds_unfinished("What time is it?"), "pronouns were removed from the list (2026-09-17)");
        assert!(!sounds_unfinished("That is all."));
        assert!(!sounds_unfinished("   "));
    }

    #[test]
    fn rms_of_a_full_scale_square_wave_is_one() {
        let frame: Vec<f32> = (0..FRAME_SAMPLES).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        assert!((rms(&frame) - 1.0).abs() < 1e-6);
        assert_eq!(rms(&[]), 0.0);
    }
}
