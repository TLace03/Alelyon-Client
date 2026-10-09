use super::*;
use crate::body::Body;

fn catalog() -> &'static Catalog {
    Catalog::builtin()
}

fn body() -> &'static Body {
    static BODY: std::sync::OnceLock<Body> = std::sync::OnceLock::new();
    BODY.get_or_init(Body::load)
}

fn control(id: &str) -> &'static Control {
    catalog().control(id).unwrap_or_else(|| panic!("no control {id}"))
}

#[test]
fn every_control_drives_shapes_the_baked_bust_has() {
    // The catalog and the binary are written by different tools at different
    // times. A control naming a target the file lacks would be a slider that
    // silently does nothing.
    let c = catalog();
    assert!(c.controls.len() >= 140, "the catalog lost controls: {}", c.controls.len());
    for control in &c.controls {
        for name in control.all_targets() {
            assert!(body().target(&name).is_some(), "{} drives {name}, which the bake did not keep", control.id);
        }
    }
    for category in &c.categories {
        assert!(c.in_category(&category.id).next().is_some(), "category {} is empty", category.id);
    }
}

#[test]
fn every_baked_shape_belongs_to_a_control_or_an_expression() {
    let used: BTreeSet<String> = catalog().controls.iter().flat_map(|c| c.all_targets()).collect();
    for t in 0..body().target_count() {
        let name = body().target_name(t);
        assert!(
            used.contains(name) || name.starts_with("expression/"),
            "{name} is baked but nothing can set it"
        );
    }
}

#[test]
fn a_slider_drives_its_low_end_below_zero_and_its_high_end_above() {
    let c = control("nose.width");
    let find = |n: &str| body().target(n);
    let mut a = Appearance::default();
    a.set(c, None, -0.4);
    assert_eq!(a.weights(catalog(), find), vec![(find("nose/nose-scale-horiz-decr").unwrap(), 0.4)]);
    a.set(c, None, 0.7);
    assert_eq!(a.weights(catalog(), find), vec![(find("nose/nose-scale-horiz-incr").unwrap(), 0.7)]);
    a.set(c, None, 5.0);
    assert_eq!(a.get(c, None), 1.0, "a value past the end was not held at the end");
    a.set(c, None, f32::NAN);
    assert_eq!(a.get(c, None), 0.0, "NaN reached a target weight");
    assert!(a.weights(catalog(), find).is_empty());
}

#[test]
fn a_limited_slider_stops_short_and_says_why() {
    let c = control("torso.chest-muscles");
    assert!(c.why.is_some());
    let mut a = Appearance::default();
    a.set(c, None, 1.0);
    let w = a.weights(catalog(), |n| body().target(n));
    assert_eq!(w, vec![(body().target("torso/torso-muscle-pectoral-incr").unwrap(), 0.5)]);
}

#[test]
fn a_shape_runs_from_nothing_to_all_of_it() {
    let c = control("head.round");
    assert_eq!(c.range(), (0.0, 1.0));
    let mut a = Appearance::default();
    a.set(c, None, -0.5);
    assert_eq!(a.get(c, None), 0.0, "a shape went below nothing");
    a.set(c, None, 0.6);
    assert_eq!(a.weights(catalog(), |n| body().target(n)), vec![(body().target("head/head-round").unwrap(), 0.6)]);
}

#[test]
fn sides_move_together_until_unlinked_and_the_left_wins_when_relinked() {
    let c = control("eyes.size");
    assert!(c.sided);
    let mut a = Appearance::default();
    assert!(a.is_linked(c));
    a.set(c, Some(Side::Right), 0.3);
    assert_eq!((a.get(c, Some(Side::Left)), a.get(c, Some(Side::Right))), (0.3, 0.3));
    a.unlink(c);
    a.set(c, Some(Side::Right), -0.2);
    assert_eq!((a.get(c, Some(Side::Left)), a.get(c, Some(Side::Right))), (0.3, -0.2));
    let w = a.weights(catalog(), |n| body().target(n));
    assert!(w.contains(&(body().target("eyes/l-eye-scale-incr").unwrap(), 0.3)));
    assert!(w.contains(&(body().target("eyes/r-eye-scale-decr").unwrap(), 0.2)));
    a.link(c);
    assert_eq!((a.get(c, Some(Side::Left)), a.get(c, Some(Side::Right))), (0.3, 0.3));
    a.reset(c);
    assert!(a.is_default_shape() && a.is_linked(c));
}

#[test]
fn an_appearance_survives_the_file() {
    let mut a = Appearance::default();
    a.set(control("nose.length"), None, -0.55);
    a.set(control("head.diamond"), None, 0.25);
    let ears = control("ears.protrusion");
    a.unlink(ears);
    a.set(ears, Some(Side::Left), 0.4);
    a.palette.set("lattice", Some([10, 200, 255]));
    let mut kept = Appearance::default();
    kept.set(control("mouth.width"), None, 0.5);
    let saved = Saved {
        current: a.clone(),
        looks: vec![("Wide".into(), kept.clone())],
        notice: String::new(),
    };
    let back = Saved::from_json(&saved.to_json(), catalog()).unwrap();
    assert_eq!(back.current, a);
    assert_eq!(back.looks, vec![("Wide".to_string(), kept)]);
    assert!(back.notice.is_empty(), "{}", back.notice);
}

#[test]
fn a_file_from_another_version_keeps_what_it_can_and_says_what_it_set_aside() {
    let text = r##"{"schema": "sinai-appearance", "version": 1,
        "values": {"nose.length": 3.0, "nose.gone": 0.5, "eyes.size.l": 0.2, "eyes.size.r": -0.6, "chin.width": "wide"},
        "unlinked": ["nose.length"],
        "colours": {"glow": "#00ff00", "fill": "teal"}}"##;
    let s = Saved::from_json(text, catalog()).unwrap();
    let a = &s.current;
    assert_eq!(a.get(control("nose.length"), None), 1.0, "an out-of-range value was not brought into range");
    let eyes = control("eyes.size");
    assert_eq!(
        (a.get(eyes, Some(Side::Left)), a.get(eyes, Some(Side::Right))),
        (0.2, 0.2),
        "a linked control saved with two different sides was not settled the way linking settles it"
    );
    assert_eq!(a.palette.glow, Some([0, 255, 0]));
    assert_eq!(a.palette.fill, None);
    for word in ["nose.gone", "brought into", "chin.width", "fill"] {
        assert!(s.notice.contains(word), "the notice does not mention {word}: {}", s.notice);
    }
}

#[test]
fn a_file_that_is_not_an_appearance_is_refused() {
    for text in ["", "not json", r#"{"schema": "angel-dock", "version": 1}"#, r#"{"schema": "sinai-appearance", "version": 9}"#] {
        assert!(Saved::from_json(text, catalog()).is_err(), "{text:?} was read as an appearance");
    }
}

#[test]
fn a_share_code_carries_an_appearance_to_another_window() {
    let mut a = Appearance::default();
    a.set(control("chin.cleft"), None, 0.8);
    a.set(control("asym.eyes-height"), None, -0.33);
    let lobe = control("ears.lobe");
    a.unlink(lobe);
    a.set(lobe, Some(Side::Right), 0.5);
    a.palette.set("iris", Some([40, 90, 200]));
    let code = a.share_code();
    assert!(code.starts_with(SHARE_PREFIX));
    let (b, notes) = Appearance::from_share_code(&code, catalog()).unwrap();
    assert!(notes.is_empty(), "{notes:?}");
    for c in &catalog().controls {
        for k in c.keys() {
            assert!((a.value(&k) - b.value(&k)).abs() <= 0.5 / 127.0 + 1e-6, "{k}: {} -> {}", a.value(&k), b.value(&k));
        }
    }
    assert!(!b.is_linked(lobe), "sides set apart came back linked");
    assert_eq!(b.palette, a.palette);
}

#[test]
fn a_damaged_share_code_is_refused_rather_than_misread() {
    let mut a = Appearance::default();
    a.set(control("nose.width"), None, 0.5);
    let code = a.share_code();
    let mut chars: Vec<char> = code.chars().collect();
    let k = chars.len() - 4;
    chars[k] = if chars[k] == 'A' { 'B' } else { 'A' };
    let broken: String = chars.into_iter().collect();
    assert!(Appearance::from_share_code(&broken, catalog()).is_err());
    assert!(Appearance::from_share_code("SINAI1:", catalog()).is_err());
    assert!(Appearance::from_share_code("hello", catalog()).is_err());
    assert!(Appearance::from_share_code(&code[..code.len() - 3], catalog()).is_err());
}

#[test]
fn no_two_settings_share_a_hash_in_a_code() {
    let mut seen = HashMap::new();
    for c in &catalog().controls {
        for k in c.keys() {
            if let Some(other) = seen.insert(fnv1a(k.as_bytes()), k.clone()) {
                panic!("{k} and {other} would be confused in a share code");
            }
        }
    }
}

#[test]
fn a_random_sinai_is_repeatable_plausible_and_can_stay_in_one_category() {
    let a = Appearance::random(catalog(), 7, None);
    assert_eq!(a, Appearance::random(catalog(), 7, None), "the same seed gave two Sinais");
    assert_ne!(a, Appearance::random(catalog(), 8, None));
    for c in &catalog().controls {
        for k in c.keys() {
            let v = a.value(&k);
            let bound = match (&c.kind, c.category.as_str()) {
                (Kind::Shape { .. }, _) => 0.45,
                (_, "asym") => 0.15,
                _ => 0.5,
            };
            assert!(v.abs() <= bound + 1e-6, "{k} = {v}, past the random range {bound}");
        }
        if c.sided {
            assert!(a.is_linked(c), "randomising unlinked {}", c.id);
        }
    }
    let nose = Appearance::random(catalog(), 3, Some("nose"));
    for c in catalog().controls.iter().filter(|c| c.category != "nose") {
        assert_eq!(nose.get(c, None), 0.0, "randomising the nose moved {}", c.id);
    }
    assert!(!nose.is_default_shape());
}

#[test]
fn one_category_can_be_taken_from_another_appearance() {
    let mut mine = Appearance::default();
    mine.set(control("nose.width"), None, 0.3);
    mine.set(control("mouth.width"), None, 0.3);
    let mut theirs = Appearance::default();
    theirs.set(control("mouth.width"), None, -0.8);
    let mixed = mine.with_category_from(&theirs, catalog(), "mouth");
    assert_eq!(mixed.get(control("nose.width"), None), 0.3);
    assert_eq!(mixed.get(control("mouth.width"), None), -0.8);
}

#[test]
fn the_brand_look_is_exact_and_a_chosen_body_colour_has_its_own_shadow() {
    let brand = Palette::default().resolve();
    assert_eq!(brand.lattice, brand::LATTICE);
    assert_eq!(brand.fill, brand::FILL);
    assert_eq!(brand.shadow, brand::SHADOW);
    assert_eq!(brand.iris[3], 0.0, "the brand look tints the irises");
    let mut p = Palette::default();
    p.set("fill", Some([100, 50, 0]));
    let c = p.resolve();
    assert!(c.shadow[0] < c.fill[0] * 0.5 && c.shadow[0] > 0.0);
    p.set("iris", Some([0, 0, 255]));
    assert_eq!(p.resolve().iris, [0.0, 0.0, 1.0, 1.0]);
    assert_eq!(Palette::default().shown("lattice"), [217, 180, 91]);
}

#[test]
fn undo_and_redo_walk_the_history_both_ways() {
    let c = control("nose.width");
    let mut h = History::default();
    let mut a = Appearance::default();
    assert!(!h.can_undo());
    for v in [0.1, 0.2, 0.3] {
        h.record(&a);
        a.set(c, None, v);
    }
    h.record(&a);
    h.record(&a);
    a = h.undo(&a).unwrap();
    assert_eq!(a.get(c, None), 0.3, "recording the same state twice made an empty undo step");
    a = h.undo(&a).unwrap();
    assert_eq!(a.get(c, None), 0.2);
    a = h.redo(&a).unwrap();
    assert_eq!(a.get(c, None), 0.3);
    h.record(&a);
    a.set(c, None, 0.9);
    assert!(!h.can_redo(), "a new change kept a redo that no longer follows from it");
}

#[test]
fn saving_replaces_the_file_whole_and_leaves_nothing_beside_it() {
    let dir = std::env::temp_dir().join(format!("angel-appearance-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("sinai-appearance.json");
    let mut s = Saved::default();
    s.current.set(control("chin.width"), None, 0.25);
    s.save_to(&path).unwrap();
    s.current.set(control("chin.width"), None, -0.25);
    s.save_to(&path).unwrap();
    let back = Saved::from_json(&std::fs::read_to_string(&path).unwrap(), catalog()).unwrap();
    assert_eq!(back.current.get(control("chin.width"), None), -0.25);
    let names: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert_eq!(names.len(), 1, "a temporary file was left behind: {names:?}");
    std::fs::remove_file(&path).unwrap();
    std::fs::remove_dir(&dir).unwrap();
}

#[test]
fn the_looks_that_ship_are_real_shapes_within_range() {
    let looks = &catalog().looks;
    assert!(looks.len() >= 6, "the catalog lost its looks: {}", looks.len());
    let mut seen = BTreeSet::new();
    for (label, look) in looks {
        assert!(seen.insert(label.clone()), "two looks are called {label}");
        assert!(!look.is_default_shape(), "{label} is Sinai as it ships");
        assert_eq!(look.palette, Palette::default(), "{label} carries colours; looks are shapes only");
        for c in &catalog().controls {
            let (lo, hi) = c.range();
            for k in c.keys() {
                let v = look.value(&k);
                assert!(v >= lo && v <= hi, "{label} sets {k} to {v}");
            }
            if c.sided {
                assert!(look.is_linked(c), "{label} sets {} apart per side", c.id);
            }
        }
        assert!(!look.weights(catalog(), |n| body().target(n)).is_empty(), "{label} drives no shape");
    }
}


#[test]
fn a_saved_appearance_survives_the_move_into_the_state_home() {
    // W3: the appearance moves from %APPDATA%\Alelyon to ~/.alelyon/angel once,
    // beside the layout, and reads back as the Sinai the person shaped.
    use crate::state_home::{File, Run, Status};
    let root = std::env::temp_dir().join(format!(
        "angel-appearance-move-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    let (roaming, home, temp) = (root.join("roaming"), root.join("home"), root.join("temp"));
    let mut s = Saved::default();
    s.current.set(control("chin.width"), None, 0.25);
    s.save_to(&roaming.join("Alelyon").join("sinai-appearance.json")).unwrap();
    let roaming_text = roaming.to_str().unwrap().to_string();
    let person = |name: &str| (name == "APPDATA").then(|| roaming_text.clone());
    let now = std::time::SystemTime::now();

    let run = Run::start(&person, Some(&home), &temp, now);
    assert_eq!(run.report.status, Status::Moved);
    let path = run.path(File::Appearance).unwrap();
    assert_eq!(path, home.join(".alelyon").join("angel").join("sinai-appearance.json"));
    let back = Saved::from_json(&std::fs::read_to_string(path).unwrap(), catalog()).unwrap();
    assert_eq!(back.current.get(control("chin.width"), None), 0.25);
    assert_eq!(back.to_json(), s.to_json());

    // ANGEL_APPEARANCE still points it elsewhere, after the move as before it.
    let other = root.join("another-sinai.json");
    let other_text = other.to_str().unwrap().to_string();
    let named = |name: &str| match name {
        "ANGEL_APPEARANCE" => Some(other_text.clone()),
        _ => person(name),
    };
    let again = Run::start(&named, Some(&home), &temp, now);
    assert_eq!(again.report.status, Status::Already);
    assert_eq!(again.path(File::Appearance), Some(other.as_path()));
    std::fs::remove_dir_all(&root).unwrap();
}
