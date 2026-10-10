//! A recording, start to finish: decoded audio in, timed lines out, as plain text, SRT, WebVTT or JSON.
//!
//! A file is not live, so it is planned before it is read: the speech is found with the live detector, speech
//! close together is read as one piece of up to 28 s (the recogniser's window is 30 s), and speech longer than
//! that is cut at its quietest moment. Each piece is read once, with what came before as context, and a
//! progress callback reports how far through the speech the job is, so the window can show a real progress
//! bar rather than a spinner over an unknown wait.
//!
//! Why not the live transcriber: it must cut at every pause, as speech arrives, and a cut costs the
//! recogniser its context. MEASURED 2026-10-03 on one fixed sample of 50 LibriSpeech test-clean clips
//! (931 words; base.en on the CPU; the same server for both arms): cutting at every pause 5.26% WER;
//! these planned pieces 4.51% (42 edits); each clip sent whole 4.62% (43). One sample and one small model:
//! evidence that the plan costs no accuracy here, not a figure for the turbo model.

use serde_json::{json, Value};

use crate::stream::RATE;
use crate::vad::{self, VadEvent, VoiceActivityDetector, FRAME_SAMPLES};
use crate::whisper::{join, Recognizer, Word};

/// The longest piece read at once, and the earliest a piece of continuous speech is cut.
const MAX_CHUNK_S: f64 = 28.0;
const MIN_CUT_S: f64 = 15.0;
/// Speech separated by a pause no longer than this is read together.
const JOIN_GAP_S: f64 = 3.0;
/// Kept before speech starts and after it stops, so no syllable is clipped.
const LEAD_S: f64 = 0.3;
const TAIL_S: f64 = 0.3;

#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub words: Vec<Word>,
}

#[derive(Debug, Default)]
pub struct FileTranscript {
    pub lines: Vec<Line>,
    /// Readings that failed, by the time they began; the rest of the file is still transcribed.
    pub errors: Vec<String>,
    pub seconds: f64,
    /// The job was stopped before the end of the audio.
    pub cancelled: bool,
}

/// A recording's loudest moments (its 99th-percentile 30 ms level) are read by the detector at this level at
/// least: a quiet recording is turned up, by at most `MAX_GAIN`, and none is turned down. Without it, a recording
/// whose speech stays under the detector's 0.020 start (a phone memo, a quiet microphone; one headset's speech
/// read 0.014-0.028, measured 2026-10-03) has no speech regions and comes back empty. Only the detector's
/// readings are scaled; the recogniser hears the audio as it is.
const LEVEL_TARGET: f32 = 0.1;
const MAX_GAIN: f32 = 20.0;

fn samples(seconds: f64) -> usize {
    (seconds * f64::from(RATE)).round() as usize
}

/// How much the detector turns this recording up (see `LEVEL_TARGET`).
pub fn level_gain(audio: &[f32]) -> f32 {
    let mut levels: Vec<f32> = audio.chunks(FRAME_SAMPLES).map(vad::rms).collect();
    levels.sort_by(f32::total_cmp);
    match levels.get(((levels.len().max(1) - 1) as f64 * 0.99).round() as usize) {
        Some(&loud) if loud > 0.0 => (LEVEL_TARGET / loud).clamp(1.0, MAX_GAIN),
        _ => 1.0,
    }
}

/// Where the speech is, in samples: what the live detector would call utterances, with a little lead and
/// tail kept, and the silence that ended each one left out.
pub fn speech_regions(audio: &[f32]) -> Vec<(usize, usize)> {
    let mut vad = VoiceActivityDetector::default();
    let gain = level_gain(audio);
    let mut regions = Vec::new();
    let mut start = None;
    for (i, frame) in audio.chunks(FRAME_SAMPLES).enumerate() {
        let quiet_before = vad.quiet_s();
        let was_in_speech = vad.in_speech;
        let frame_end = i * FRAME_SAMPLES + frame.len();
        let mut ended = false;
        for event in vad.push(vad::rms(frame) * gain, false) {
            match event {
                VadEvent::SpeechStart => {
                    let proved = vad::START_FRAMES as usize * FRAME_SAMPLES;
                    start = Some(frame_end.saturating_sub(proved + samples(LEAD_S)));
                }
                VadEvent::SpeechEnd => {
                    ended = true;
                    if let Some(s) = start.take() {
                        // the silence that ended it is not speech (the 30 s hard stop has none)
                        let trailing = if vad.speech_duration_s() == 0.0 && quiet_before > 0.0 {
                            samples(f64::from(quiet_before + vad::FRAME_S))
                        } else {
                            0
                        };
                        let end = (frame_end.saturating_sub(trailing) + samples(TAIL_S)).min(audio.len());
                        regions.push((s, end.max(s)));
                    }
                }
                VadEvent::PartialDue => {}
            }
        }
        // A burst too short to be speech leaves speech without an event: it was a cough, not a region.
        if was_in_speech && !vad.in_speech && !ended {
            start = None;
        }
    }
    if let Some(s) = start {
        regions.push((s, audio.len()));
    }
    regions
}

/// The pieces to read: speech close together joined, nothing longer than `MAX_CHUNK_S`, long speech cut at
/// its quietest 30 ms between `MIN_CUT_S` and `MAX_CHUNK_S` into the piece.
pub fn plan(audio: &[f32]) -> Vec<(usize, usize)> {
    let (max, min_cut, gap) = (samples(MAX_CHUNK_S), samples(MIN_CUT_S), samples(JOIN_GAP_S));
    let mut chunks: Vec<(usize, usize)> = Vec::new();
    let mut current: Option<(usize, usize)> = None;
    for (s, e) in speech_regions(audio) {
        current = match current {
            Some((cs, ce)) if s.saturating_sub(ce) <= gap && e - cs <= max => Some((cs, e)),
            Some(done) => {
                chunks.push(done);
                Some((s, e))
            }
            None => Some((s, e)),
        };
        while let Some((cs, ce)) = current {
            if ce - cs <= max {
                break;
            }
            let cut = quietest(audio, cs + min_cut, cs + max);
            chunks.push((cs, cut));
            current = Some((cut, ce));
        }
    }
    chunks.extend(current);
    chunks
}

/// The start of the quietest frame in `from..to`.
fn quietest(audio: &[f32], from: usize, to: usize) -> usize {
    let mut best = (f32::MAX, to);
    let mut at = from;
    while at + FRAME_SAMPLES <= to.min(audio.len()) {
        let r = vad::rms(&audio[at..at + FRAME_SAMPLES]);
        if r < best.0 {
            best = (r, at);
        }
        at += FRAME_SAMPLES;
    }
    best.1
}

/// Transcribe 16 kHz mono audio. `progress` receives the fraction of the speech done, from 0 to 1, and
/// returns whether to go on: false stops the job, and what was transcribed so far is returned with
/// `cancelled` set.
pub fn transcribe<R: Recognizer>(
    audio: &[f32],
    mut recognizer: R,
    base_prompt: Option<String>,
    mut progress: impl FnMut(f64) -> bool,
) -> FileTranscript {
    let mut out = FileTranscript { seconds: audio.len() as f64 / f64::from(RATE), ..Default::default() };
    let chunks = plan(audio);
    let total: usize = chunks.iter().map(|(s, e)| e - s).sum::<usize>().max(1);
    let mut done = 0usize;
    let mut context = String::new();
    if !progress(0.0) {
        out.cancelled = true;
        return out;
    }
    for (s, e) in chunks {
        let prompt = [base_prompt.as_deref().unwrap_or(""), context.as_str()].join(" ").trim().to_string();
        let offset = s as f64 / f64::from(RATE);
        match recognizer.transcribe(&audio[s..e], (!prompt.is_empty()).then_some(prompt.as_str())) {
            Ok(t) => {
                for seg in t.segments {
                    let words: Vec<Word> = seg.words.iter().filter(|w| !w.key().is_empty()).map(|w| w.shifted(offset)).collect();
                    let text = if words.is_empty() { seg.text.trim().to_string() } else { join(&words) };
                    if text.is_empty() {
                        continue;
                    }
                    let start = words.first().map(|w| w.start).unwrap_or(seg.start + offset);
                    let end = words.last().map(|w| w.end).unwrap_or(seg.end + offset);
                    context = remember(&context, &text);
                    out.lines.push(Line { start, end, text, words });
                }
            }
            Err(message) => out.errors.push(format!("at {offset:.1} s: {message}")),
        }
        done += e - s;
        if !progress(done as f64 / total as f64) {
            out.cancelled = true;
            return out;
        }
    }
    progress(1.0);
    out
}

/// The last 200 characters of what was said, from a word boundary: the next piece's context.
fn remember(context: &str, text: &str) -> String {
    let all = format!("{context} {text}");
    let all = all.trim();
    let count = all.chars().count();
    if count <= 200 {
        return all.to_string();
    }
    let tail: String = all.chars().skip(count - 200).collect();
    match tail.find(' ') {
        Some(i) => tail[i + 1..].to_string(),
        None => tail,
    }
}

fn clock(seconds: f64, sep: char) -> String {
    let ms = (seconds.max(0.0) * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02}{sep}{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

pub fn to_text(t: &FileTranscript) -> String {
    let mut out = String::new();
    for line in &t.lines {
        out.push_str(&line.text);
        out.push('\n');
    }
    out
}

/// Longest subtitle cue, in characters (two lines of the 42 broadcasters use) and in seconds.
const CUE_CHARS: usize = 84;
const CUE_SECONDS: f64 = 6.0;
/// A pause this long between two words is a good place to start a new cue.
const CUE_PAUSE_S: f64 = 0.6;

/// A line cut into subtitle cues by its word times: none longer than `CUE_CHARS` or `CUE_SECONDS`, broken
/// after punctuation or at a pause where one is near, and every cue's text on at most two lines.
pub fn cues(line: &Line) -> Vec<(f64, f64, String)> {
    if line.words.is_empty() {
        return vec![(line.start, line.end, line.text.clone())];
    }
    let mut out = Vec::new();
    let mut current: Vec<&Word> = Vec::new();
    for (i, w) in line.words.iter().enumerate() {
        if let Some(first) = current.first() {
            let text_len = current.iter().map(|w| w.text.len()).sum::<usize>() + w.text.len();
            let too_long = text_len > CUE_CHARS || w.end - first.start > CUE_SECONDS;
            let last = current[current.len() - 1];
            // Where the recogniser left the full stops out, a capitalised word other than "I" is most often
            // a sentence starting.
            let next = w.text.trim();
            let sentence_start = next.chars().next().is_some_and(char::is_uppercase) && next != "I" && !next.starts_with("I'");
            let natural = last.text.trim_end().ends_with(['.', ',', '?', '!', ';', ':'])
                || w.start - last.end >= CUE_PAUSE_S
                || sentence_start;
            // Break at a natural point once the cue is half full, and anywhere once it would be too long.
            let half = current.iter().map(|w| w.text.len()).sum::<usize>() >= CUE_CHARS / 2;
            if too_long || (natural && half) {
                out.push(cue_of(&current));
                current.clear();
            }
        }
        current.push(w);
        if i == line.words.len() - 1 {
            out.push(cue_of(&current));
        }
    }
    out
}

fn cue_of(words: &[&Word]) -> (f64, f64, String) {
    let text: String = words.iter().map(|w| w.text.as_str()).collect::<String>().trim().to_string();
    (words[0].start, words[words.len() - 1].end, two_lines(&text))
}

/// Text over 42 characters broken into two lines at the space nearest the middle.
fn two_lines(text: &str) -> String {
    if text.chars().count() <= CUE_CHARS / 2 {
        return text.to_string();
    }
    let middle = text.len() / 2;
    let split = text
        .char_indices()
        .filter(|(_, c)| *c == ' ')
        .map(|(i, _)| i)
        .min_by_key(|i| i.abs_diff(middle));
    match split {
        Some(i) => format!("{}\n{}", &text[..i], &text[i + 1..]),
        None => text.to_string(),
    }
}

pub fn to_srt(t: &FileTranscript) -> String {
    let mut out = String::new();
    let mut n = 0;
    for line in &t.lines {
        for (start, end, text) in cues(line) {
            n += 1;
            out.push_str(&format!("{n}\n{} --> {}\n{text}\n\n", clock(start, ','), clock(end, ',')));
        }
    }
    out
}

pub fn to_vtt(t: &FileTranscript) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for line in &t.lines {
        for (start, end, text) in cues(line) {
            out.push_str(&format!("{} --> {}\n{text}\n\n", clock(start, '.'), clock(end, '.')));
        }
    }
    out
}

pub fn to_json(t: &FileTranscript, source: &str) -> Value {
    json!({
        "source": source,
        "seconds": (t.seconds * 1000.0).round() / 1000.0,
        "errors": t.errors,
        "lines": t.lines.iter().map(|l| json!({
            "start": (l.start * 1000.0).round() / 1000.0,
            "end": (l.end * 1000.0).round() / 1000.0,
            "text": l.text,
            "words": l.words.iter().map(|w| json!({"w": w.text, "s": (w.start * 1000.0).round() / 1000.0, "e": (w.end * 1000.0).round() / 1000.0})).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whisper::{transcript_of, Scripted};

    fn tone(seconds: f64, amp: f32) -> Vec<f32> {
        (0..(seconds * f64::from(RATE)) as usize).map(|i| if i % 2 == 0 { amp } else { -amp }).collect()
    }

    /// Two pieces of speech four seconds apart: read as two pieces.
    fn recording() -> Vec<f32> {
        let mut a = tone(0.5, 0.0005);
        a.extend(tone(1.0, 0.1));
        a.extend(tone(4.0, 0.0005));
        a.extend(tone(1.0, 0.1));
        a.extend(tone(1.5, 0.0005));
        a
    }

    fn seconds(chunks: &[(usize, usize)]) -> Vec<(f64, f64)> {
        chunks.iter().map(|&(s, e)| (s as f64 / f64::from(RATE), e as f64 / f64::from(RATE))).collect()
    }

    #[test]
    fn a_quiet_recording_is_turned_up_for_the_detector_and_a_loud_one_is_left_alone() {
        // speech at 0.01, under the detector's 0.020 start, in a quiet room
        let mut quiet = tone(0.5, 0.0001);
        quiet.extend(tone(1.0, 0.01));
        quiet.extend(tone(1.5, 0.0001));
        assert!((level_gain(&quiet) - 10.0).abs() < 0.01, "{}", level_gain(&quiet));
        let found = seconds(&speech_regions(&quiet));
        assert_eq!(found.len(), 1, "the speech is found: {found:?}");
        assert!(found[0].0 < 0.5 && found[0].1 > 1.5, "{found:?}");
        assert_eq!(level_gain(&recording()), 1.0, "speech at 0.1 is read as it is");
        assert_eq!(level_gain(&tone(2.0, 0.5)), 1.0, "nothing is turned down");
        assert_eq!(level_gain(&tone(2.0, 0.0)), 1.0, "digital silence is left alone");
        assert_eq!(level_gain(&tone(2.0, 0.0001)), MAX_GAIN, "a room with nobody in it is turned up at most 20x");
        assert!(speech_regions(&tone(3.0, 0.0001)).is_empty(), "and still holds no speech");
    }

    #[test]
    fn speech_close_together_is_read_as_one_piece() {
        let mut a = tone(0.5, 0.0005);
        a.extend(tone(1.0, 0.1));
        a.extend(tone(1.5, 0.0005));
        a.extend(tone(1.0, 0.1));
        a.extend(tone(1.5, 0.0005));
        let chunks = seconds(&plan(&a));
        assert_eq!(chunks.len(), 1, "{chunks:?}");
        let (s, e) = chunks[0];
        assert!(s < 0.5 && s > 0.0, "the piece starts just before the speech: {s}");
        assert!(e > 4.0 && e < 4.5, "and ends just after it, not with the closing silence: {e}");
        assert_eq!(seconds(&plan(&recording())).len(), 2, "four seconds apart is two pieces");
    }

    #[test]
    fn long_speech_is_cut_at_its_quietest_moment() {
        let mut a = tone(0.5, 0.0005);
        a.extend(tone(19.5, 0.1));
        a.extend(tone(0.3, 0.004)); // a breath, too short to end the speech
        a.extend(tone(20.0, 0.1));
        a.extend(tone(1.5, 0.0005));
        let chunks = seconds(&plan(&a));
        assert_eq!(chunks.len(), 2, "{chunks:?}");
        assert!((chunks[0].1 - 20.0).abs() < 0.35, "cut in the breath at 20 s: {chunks:?}");
        assert!(chunks.iter().all(|(s, e)| e - s <= MAX_CHUNK_S + 1e-9), "{chunks:?}");
        assert_eq!(chunks[0].1, chunks[1].0, "nothing is skipped at the cut");
    }

    #[test]
    fn a_cough_is_not_speech() {
        let mut a = tone(1.0, 0.0005);
        a.extend(tone(0.15, 0.1));
        a.extend(tone(2.0, 0.0005));
        assert!(plan(&a).is_empty());
    }

    #[test]
    fn a_recording_becomes_timed_lines_with_progress_to_the_end() {
        let script = vec![Ok(transcript_of("good morning", 0.3, 0.3)), Ok(transcript_of("how are you", 0.3, 0.3))];
        let mut seen = Vec::new();
        let t = transcribe(&recording(), Scripted::new(script), None, |f| {
            seen.push(f);
            true
        });
        assert_eq!(t.lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>(), ["good morning", "how are you"]);
        assert!(t.lines[0].start < t.lines[0].end && t.lines[0].end <= t.lines[1].start);
        assert_eq!(seen.first(), Some(&0.0));
        assert_eq!(seen.last(), Some(&1.0));
        assert!(seen.windows(2).all(|w| w[0] <= w[1]), "progress never goes backwards: {seen:?}");
        assert!((t.seconds - 8.0).abs() < 1e-9);
    }

    #[test]
    fn a_failed_piece_is_recorded_and_the_rest_still_transcribed() {
        let script = vec![Err("whisper-server answered 500".to_string()), Ok(transcript_of("still here", 0.3, 0.3))];
        let t = transcribe(&recording(), Scripted::new(script), None, |_| true);
        assert_eq!(t.errors.len(), 1);
        assert!(t.errors[0].starts_with("at 0.2 s:"), "the failure says where: {:?}", t.errors);
        assert_eq!(t.lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>(), ["still here"]);
    }

    #[test]
    fn a_job_stopped_part_way_keeps_what_it_had_and_says_so() {
        let script = vec![Ok(transcript_of("good morning", 0.3, 0.3)), Ok(transcript_of("never read", 0.3, 0.3))];
        let mut calls = 0;
        let t = transcribe(&recording(), Scripted::new(script), None, |f| {
            calls += 1;
            f < 0.5
        });
        assert!(t.cancelled);
        assert_eq!(t.lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>(), ["good morning"]);
        assert!(calls > 1);
    }

    #[test]
    fn a_long_line_becomes_short_cues_broken_at_punctuation_and_pauses() {
        let text = "Yesterday you were trembling for a health that is dear to you. Today you fear for your own. \
                    Tomorrow it will be anxiety about money, the day after tomorrow the diatribe of a slanderer";
        let words: Vec<Word> = text
            .split_whitespace()
            .enumerate()
            .map(|(i, w)| Word { text: format!(" {w}"), start: i as f64 * 0.35, end: i as f64 * 0.35 + 0.3, probability: 0.9 })
            .collect();
        let line = Line { start: 0.0, end: words.last().unwrap().end, text: text.into(), words };
        let cues = cues(&line);
        assert!(cues.len() >= 3, "{cues:?}");
        for (start, end, cue) in &cues {
            assert!(end - start <= CUE_SECONDS + 1e-9, "{start}..{end}: {cue}");
            assert!(cue.lines().count() <= 2 && cue.lines().all(|l| l.chars().count() <= CUE_CHARS), "{cue}");
        }
        assert!(cues[0].2.ends_with("to you."), "the first cue ends at the full stop: {:?}", cues[0].2);
        let rejoined: Vec<String> = cues.iter().map(|c| c.2.replace('\n', " ")).collect();
        assert_eq!(rejoined.join(" "), text, "no word lost or repeated");
    }

    #[test]
    fn subtitles_have_their_formats_clocks() {
        let t = FileTranscript {
            lines: vec![Line { start: 3661.5, end: 3662.25, text: "Hello".into(), words: vec![] }],
            errors: vec![],
            seconds: 3663.0,
            cancelled: false,
        };
        assert_eq!(to_srt(&t), "1\n01:01:01,500 --> 01:01:02,250\nHello\n\n");
        assert_eq!(to_vtt(&t), "WEBVTT\n\n01:01:01.500 --> 01:01:02.250\nHello\n\n");
        assert_eq!(to_text(&t), "Hello\n");
        assert_eq!(to_json(&t, "a.wav")["lines"][0]["end"], 3662.25);
    }
}
