//! Sinai's expressions, as shapes its face takes.
//!
//! A tone arrives with each sentence Sinai says, chosen from the marks it wrote,
//! along with a lid, a tilt and a warmth ([`tone_pose`]). Those three were all
//! a fixed head could show: there was no smile, because there was no way to
//! move the corners of a mouth that was a single baked mesh. The bust carries the 34
//! expression units MakeHuman ships -- the muscle actions of the face, close to
//! the action units of the Facial Action Coding System -- so a tone now also
//! names which of them it uses and how far. The lid, tilt and warmth still
//! apply on top, as they did.
//!
//! Each tone's units follow the action units the expression is known by:
//! a smile is the lip-corner puller with the cheek raised (AU 6 + 12); surprise
//! is raised brows, raised upper lids and parted lips (AU 1 + 2 + 5 + 25); sadness
//! is the inner brow raised and the lip corners pulled down (AU 1 + 4 + 15);
//! anger is the brows drawn down, the lids tightened and the lips pressed
//! (AU 4 + 7 + 24). How far each one goes was set by looking at the result.

use crate::body::Body;

/// The units a tone uses, and how far. Names are the bake's.
pub fn tone_units(tone: &str) -> &'static [(&'static str, f32)] {
    match tone {
        "smile" => &[
            ("expression/mouth-corner-puller", 0.55),
            ("expression/eye-left-slit", 0.18),
            ("expression/eye-right-slit", 0.18),
        ],
        "laugh" => &[
            ("expression/mouth-corner-puller", 0.90),
            ("expression/mouth-upward-retraction", 0.25),
            ("expression/eye-left-slit", 0.40),
            ("expression/eye-right-slit", 0.40),
            ("expression/eyebrows-left-up", 0.15),
            ("expression/eyebrows-right-up", 0.15),
        ],
        "warm" => &[
            ("expression/mouth-corner-puller", 0.35),
            ("expression/eyebrows-left-inner-up", 0.20),
            ("expression/eyebrows-right-inner-up", 0.20),
            ("expression/eye-left-slit", 0.12),
            ("expression/eye-right-slit", 0.12),
        ],
        "approve" => &[
            ("expression/mouth-corner-puller", 0.30),
            ("expression/mouth-compression", 0.20),
        ],
        "celebrate" => &[
            ("expression/mouth-corner-puller", 0.80),
            ("expression/eyebrows-left-up", 0.50),
            ("expression/eyebrows-right-up", 0.50),
            ("expression/eye-left-opened-up", 0.20),
            ("expression/eye-right-opened-up", 0.20),
        ],
        "wink" => &[
            ("expression/mouth-corner-puller", 0.45),
            ("expression/eye-left-closure", 0.95),
            ("expression/eyebrows-left-down", 0.15),
            ("expression/eye-right-slit", 0.15),
        ],
        "thinking" => &[
            ("expression/eyebrows-left-up", 0.45),
            ("expression/eyebrows-right-down", 0.25),
            ("expression/mouth-compression", 0.30),
            ("expression/eye-left-slit", 0.15),
            ("expression/eye-right-slit", 0.15),
        ],
        "surprise" => &[
            ("expression/eyebrows-left-up", 0.75),
            ("expression/eyebrows-right-up", 0.75),
            ("expression/eyebrows-left-inner-up", 0.30),
            ("expression/eyebrows-right-inner-up", 0.30),
            ("expression/eye-left-opened-up", 0.55),
            ("expression/eye-right-opened-up", 0.55),
            ("expression/mouth-parling", 0.50),
        ],
        "worried" => &[
            ("expression/eyebrows-left-inner-up", 0.70),
            ("expression/eyebrows-right-inner-up", 0.70),
            ("expression/eyebrows-left-down", 0.15),
            ("expression/eyebrows-right-down", 0.15),
            ("expression/mouth-retraction", 0.30),
            ("expression/mouth-compression", 0.10),
        ],
        "sad" => &[
            ("expression/eyebrows-left-inner-up", 0.60),
            ("expression/eyebrows-right-inner-up", 0.60),
            ("expression/eyebrows-left-down", 0.15),
            ("expression/eyebrows-right-down", 0.15),
            ("expression/mouth-depression", 0.60),
            ("expression/eye-left-slit", 0.12),
            ("expression/eye-right-slit", 0.12),
        ],
        "angry" => &[
            ("expression/eyebrows-left-down", 0.85),
            ("expression/eyebrows-right-down", 0.85),
            ("expression/eye-left-slit", 0.30),
            ("expression/eye-right-slit", 0.30),
            ("expression/mouth-compression", 0.50),
            ("expression/nose-left-elevation", 0.25),
            ("expression/nose-right-elevation", 0.25),
            ("expression/eye-left-opened-up", 0.15),
            ("expression/eye-right-opened-up", 0.15),
        ],
        _ => &[],
    }
}

/// Every tone a sentence can carry, for the creator's preview.
pub const TONES: [&str; 11] = [
    "smile", "laugh", "warm", "approve", "celebrate", "wink", "thinking", "surprise", "worried", "sad", "angry",
];

/// The lid, tilt and warmth that come with a tone, for the creator's preview,
/// where no sentence is carrying them. The sender keeps its own table of these;
/// the Angel window's tests hold the two equal.
pub fn tone_pose(tone: &str) -> (f32, f32, f32) {
    match tone {
        "laugh" => (0.22, -0.05, 1.0),
        "smile" => (0.10, -0.03, 0.8),
        "warm" => (0.12, 0.04, 1.0),
        "approve" => (0.06, -0.02, 0.6),
        "celebrate" => (-0.10, -0.06, 0.9),
        "wink" => (0.14, 0.07, 0.7),
        "thinking" => (0.10, 0.10, -0.2),
        "surprise" => (-0.30, -0.04, 0.1),
        "worried" => (0.08, 0.06, -0.5),
        "sad" => (0.20, 0.03, -0.8),
        "angry" => (0.26, -0.02, -0.6),
        _ => (0.0, 0.0, 0.0),
    }
}

/// The face's current expression, easing from one tone to the next.
pub struct Face {
    /// Target index and current weight for every expression unit the bust has.
    units: Vec<(usize, f32)>,
}

impl Face {
    pub fn new(body: &Body) -> Face {
        let units = (0..body.target_count())
            .filter(|&t| body.target_name(t).starts_with("expression/"))
            .map(|t| (t, 0.0))
            .collect();
        Face { units }
    }

    /// Move part of the way towards `tone`. Returns whether anything moved
    /// enough to draw again; a face that has arrived stops costing a re-pose.
    pub fn ease_toward(&mut self, body: &Body, tone: &str, ease: f32) -> bool {
        let wanted = tone_units(tone);
        let mut moved = false;
        for (t, w) in self.units.iter_mut() {
            let name = body.target_name(*t);
            let goal = wanted.iter().find(|(n, _)| *n == name).map_or(0.0, |(_, v)| *v);
            if *w == goal {
                continue;
            }
            let next = *w + (goal - *w) * ease.clamp(0.0, 1.0);
            *w = if (next - goal).abs() < 1e-3 { goal } else { next };
            moved = true;
        }
        moved
    }

    /// The units in use, for the pose.
    pub fn weights(&self) -> impl Iterator<Item = (usize, f32)> + '_ {
        self.units.iter().copied().filter(|&(_, w)| w != 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body() -> &'static Body {
        static BODY: std::sync::OnceLock<Body> = std::sync::OnceLock::new();
        BODY.get_or_init(Body::load)
    }

    #[test]
    fn every_tone_uses_units_the_bust_has_within_their_range() {
        for tone in TONES {
            let units = tone_units(tone);
            assert!(!units.is_empty(), "{tone} has no shape");
            for (name, w) in units {
                assert!(body().target(name).is_some(), "{tone} uses {name}, which was not baked");
                assert!(*w > 0.0 && *w <= 1.0, "{tone} drives {name} at {w}");
            }
        }
        assert!(tone_units("").is_empty() && tone_units("unknown").is_empty());
    }

    #[test]
    fn a_face_eases_into_a_tone_and_stops_when_it_arrives() {
        let b = body();
        let mut face = Face::new(b);
        assert_eq!(face.weights().count(), 0);
        let mut steps = 0;
        while face.ease_toward(b, "smile", 0.2) {
            steps += 1;
            assert!(steps < 200, "the face never arrived");
        }
        let smile = b.target("expression/mouth-corner-puller").unwrap();
        assert!(face.weights().any(|(t, w)| t == smile && (w - 0.55).abs() < 1e-6));
        assert!(!face.ease_toward(b, "smile", 0.2), "an arrived face kept asking to be drawn");
        while face.ease_toward(b, "", 0.5) {}
        assert_eq!(face.weights().count(), 0, "the face did not come back to rest");
    }
}
