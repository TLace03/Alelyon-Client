//! The ears' events as JSON: what the window and Sinai's hearing receive over the socket.
//!
//! Every message is one JSON object with a `type` beginning `ears.`, so a hub that relays them can carry them
//! beside its own types without a clash. Times are seconds; word lists carry `w` (the word as the
//! recogniser wrote it), `s`, `e` and `p` (start, end, probability).

use serde_json::{json, Value};

use crate::stream::Event;
use crate::whisper::Word;

fn words(ws: &[Word]) -> Value {
    Value::Array(
        ws.iter()
            .map(|w| json!({"w": w.text, "s": round3(w.start), "e": round3(w.end), "p": (f64::from(w.probability) * 1000.0).round() / 1000.0}))
            .collect(),
    )
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// One transcription event, tagged with the session it belongs to (a dictation, a file job, the
/// microphone's listening session).
pub fn event(e: &Event, session: &str) -> Value {
    match e {
        Event::SpeechStart { source, utterance, at } => {
            json!({"type": "ears.speech", "source": source.name(), "session": session, "utterance": utterance, "state": "start", "at": round3(*at)})
        }
        Event::Partial { source, utterance, stable, settling } => {
            json!({"type": "ears.partial", "source": source.name(), "session": session, "utterance": utterance, "stable": stable, "settling": settling})
        }
        Event::Final { source, utterance, text, start, end, words: ws } => json!({
            "type": "ears.final", "source": source.name(), "session": session, "utterance": utterance,
            "text": text, "start": round3(*start), "end": round3(*end), "words": words(ws)
        }),
        Event::Dropped { source, utterance } => {
            json!({"type": "ears.speech", "source": source.name(), "session": session, "utterance": utterance, "state": "dropped"})
        }
        Event::Error { source, utterance, stage, message } => json!({
            "type": "ears.error", "source": source.name(), "session": session, "utterance": utterance, "stage": stage, "message": message
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::Source;

    #[test]
    fn a_final_carries_its_words_with_rounded_times() {
        let e = Event::Final {
            source: Source::Mic,
            utterance: 3,
            text: "Hi Sinai".into(),
            start: 1.23456,
            end: 2.5,
            words: vec![Word { text: " Hi".into(), start: 1.30001, end: 1.5, probability: 0.98765 }],
        };
        let v = event(&e, "listen-1");
        assert_eq!(v["type"], "ears.final");
        assert_eq!(v["source"], "mic");
        assert_eq!(v["session"], "listen-1");
        assert_eq!(v["start"], 1.235);
        assert_eq!(v["words"][0]["w"], " Hi");
        assert_eq!(v["words"][0]["s"], 1.3);
        assert_eq!(v["words"][0]["p"], 0.988);
    }

    #[test]
    fn every_event_type_is_namespaced() {
        let all = [
            Event::SpeechStart { source: Source::Pc, utterance: 1, at: 0.5 },
            Event::Partial { source: Source::Pc, utterance: 1, stable: "a".into(), settling: "b".into() },
            Event::Dropped { source: Source::Pc, utterance: 1 },
            Event::Error { source: Source::Pc, utterance: 1, stage: "final", message: "x".into() },
        ];
        for e in &all {
            assert!(event(e, "s").get("type").and_then(Value::as_str).unwrap().starts_with("ears."));
        }
    }
}
