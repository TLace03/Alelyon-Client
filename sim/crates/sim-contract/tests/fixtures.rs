//! The committed JSON fixtures a Python training loop reads.
//!
//! `fixtures/v0/` holds one JSON file per type of the contract (one per variant
//! for the tagged enums), plus `index.json`, which names the Rust type each file
//! holds. This test builds every value in code (`tests/common/mod.rs`), writes it
//! the way the files are written, and compares it with the committed bytes: a
//! change to a type, to its JSON shape, or to an example value turns this red
//! until the fixtures are regenerated and the diff reviewed. It also reads every
//! committed file back into its type and validates it, so a fixture can never
//! hold a value the contract itself would refuse.
//!
//! To regenerate after an intended change, from the workspace folder:
//!
//! ```text
//! SIM_CONTRACT_REGENERATE_FIXTURES=1 cargo test --locked -p sim-contract --test fixtures
//! ```
//!
//! That run rewrites the files and removes any file it no longer lists; run the
//! test again without the variable to check the result.

mod common;

use std::collections::BTreeSet;
use std::fmt::Debug;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::Serialize;
use serde::de::DeserializeOwned;
use sim_contract::{
    Action, AudioChunk, Bundle, Capture, ContractError, FrameRef, FrameSemantic, GroundTruth,
    Identity, Perturbation, Proprio, RATES_V0, Rates, Reset, SCHEMA_VERSION, SpeciesTable,
    SpeciesVector, TouchMaps,
};

const REGENERATE: &str = "SIM_CONTRACT_REGENERATE_FIXTURES";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join("v0")
}

fn regenerating() -> bool {
    std::env::var(REGENERATE).is_ok_and(|value| value == "1")
}

/// How a fixture is written: pretty JSON, a trailing newline, LF line endings.
fn render<T: Serialize>(value: &T) -> String {
    let mut text = serde_json::to_string_pretty(value).unwrap();
    text.push('\n');
    text
}

/// Writes (when regenerating) or compares one fixture, reads it back, and
/// validates it. Returns the `index.json` entry.
fn fixture<T>(
    stem: &str,
    type_name: &str,
    value: &T,
    validate: impl Fn(&T) -> Result<(), ContractError>,
) -> (String, String)
where
    T: Serialize + DeserializeOwned + PartialEq + Debug,
{
    let file = format!("{stem}.json");
    let path = fixture_dir().join(&file);
    let expected = render(value);
    if regenerating() {
        fs::create_dir_all(fixture_dir()).unwrap();
        fs::write(&path, &expected).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    }
    let on_disk = fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}\nregenerate with {REGENERATE}=1 (see the header of tests/fixtures.rs)",
            path.display()
        )
    });
    assert_eq!(
        on_disk, expected,
        "{file} is out of date with the code; regenerate with {REGENERATE}=1 and review the diff"
    );
    let parsed: T = serde_json::from_str(&on_disk).unwrap();
    assert_eq!(&parsed, value, "{file} does not read back to its value");
    validate(&parsed).unwrap_or_else(|e| panic!("{file} holds a value the contract refuses: {e}"));
    (file, type_name.to_string())
}

/// Every fixture, as listed here. Built (and, when
/// regenerating, written) once per test process, however many tests ask.
fn every_fixture() -> Vec<(String, String)> {
    static ENTRIES: OnceLock<Vec<(String, String)>> = OnceLock::new();
    ENTRIES.get_or_init(build_every_fixture).clone()
}

fn build_every_fixture() -> Vec<(String, String)> {
    let smell = common::smell_table();
    let taste = common::taste_table();
    vec![
        fixture(
            "identity",
            "Identity",
            &common::identity(),
            Identity::validate,
        ),
        fixture(
            "capture_proprio",
            "Capture<Proprio>",
            &common::capture_proprio(),
            |c: &Capture<Proprio>| c.validate_time().and_then(|()| c.value.validate()),
        ),
        fixture(
            "frame_ref_rgb",
            "FrameRef",
            &common::rgb(),
            |f: &FrameRef| f.validate_v0(FrameSemantic::Rgb),
        ),
        fixture(
            "frame_ref_depth",
            "FrameRef",
            &common::depth(),
            |f: &FrameRef| f.validate_v0(FrameSemantic::DepthMetres),
        ),
        fixture(
            "frame_ref_segmentation",
            "FrameRef",
            &common::segmentation(),
            |f: &FrameRef| f.validate_v0(FrameSemantic::Segmentation),
        ),
        fixture(
            "audio_chunk_host",
            "AudioChunk",
            &common::audio_host(),
            AudioChunk::validate,
        ),
        fixture(
            "audio_chunk_device",
            "AudioChunk",
            &common::audio_device(),
            AudioChunk::validate,
        ),
        fixture("proprio", "Proprio", &common::proprio(), Proprio::validate),
        fixture(
            "touch_maps",
            "TouchMaps",
            &common::touch_maps(),
            TouchMaps::validate,
        ),
        fixture(
            "species_table_smell",
            "SpeciesTable",
            &smell,
            SpeciesTable::validate,
        ),
        fixture(
            "species_table_taste",
            "SpeciesTable",
            &taste,
            SpeciesTable::validate,
        ),
        fixture(
            "species_vector_smell",
            "SpeciesVector",
            &common::smell_vector(),
            |v: &SpeciesVector| v.validate(&smell),
        ),
        fixture(
            "species_vector_taste",
            "SpeciesVector",
            &common::taste_vector(),
            |v: &SpeciesVector| v.validate(&taste),
        ),
        fixture("bundle", "Bundle", &common::bundle_full(), |b: &Bundle| {
            b.validate(&smell, &taste)
        }),
        fixture(
            "bundle_minimal",
            "Bundle",
            &common::bundle_minimal(),
            |b: &Bundle| {
                let empty = SpeciesTable {
                    version: 0,
                    species: Vec::new(),
                };
                b.validate(&empty, &empty)
            },
        ),
        fixture("action", "Action", &common::action_pose(), Action::validate),
        fixture(
            "action_look_at",
            "Action",
            &common::action_look_at(),
            Action::validate,
        ),
        fixture("reset", "Reset", &common::reset(), Reset::validate),
        fixture(
            "ground_truth",
            "GroundTruth",
            &common::ground_truth(),
            GroundTruth::validate,
        ),
        fixture(
            "perturbation_offset",
            "Perturbation",
            &common::perturbation_offset(),
            Perturbation::validate,
        ),
        fixture(
            "perturbation_delay",
            "Perturbation",
            &common::perturbation_delay(),
            Perturbation::validate,
        ),
        fixture(
            "perturbation_render_from_perturbed_state",
            "Perturbation",
            &common::perturbation_render_from_perturbed_state(),
            Perturbation::validate,
        ),
        fixture("rates", "Rates", &RATES_V0, |_: &Rates| Ok(())),
    ]
}

/// The fixtures that are evaluation only although their type is not: a `FrameRef`
/// is an observation when it is an RGB or depth frame, but a segmentation frame
/// is ground truth (the contract's decision), so a loop that
/// reads the index must not feed it to a model.
const EVALUATION_ONLY_FILES: [&str; 1] = ["frame_ref_segmentation.json"];

/// `index.json`: which Rust type each fixture holds, and which of them are
/// evaluation only.
fn index(entries: &[(String, String)]) -> serde_json::Value {
    let evaluation_only_types = ["GroundTruth", "Perturbation"];
    let files: serde_json::Map<String, serde_json::Value> = entries
        .iter()
        .map(|(file, type_name)| {
            let evaluation_only = evaluation_only_types.contains(&type_name.as_str())
                || EVALUATION_ONLY_FILES.contains(&file.as_str());
            (
                file.clone(),
                serde_json::json!({
                    "type": type_name,
                    "evaluation_only": evaluation_only,
                }),
            )
        })
        .collect();
    serde_json::json!({ "schema_version": SCHEMA_VERSION, "fixtures": files })
}

#[test]
fn the_committed_fixtures_match_the_code_and_the_contract() {
    let entries = every_fixture();

    // index.json, written the same way.
    let index = index(&entries);
    let path = fixture_dir().join("index.json");
    let expected = render(&index);
    if regenerating() {
        fs::write(&path, &expected).unwrap();
    }
    let on_disk = fs::read_to_string(&path).unwrap();
    assert_eq!(on_disk, expected, "index.json is out of date; regenerate");

    // No fixture file is left over, and none is unlisted.
    let listed: BTreeSet<String> = entries
        .iter()
        .map(|(file, _)| file.clone())
        .chain(std::iter::once("index.json".to_string()))
        .collect();
    let present: BTreeSet<String> = fs::read_dir(fixture_dir())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    if regenerating() {
        for stale in present.difference(&listed) {
            fs::remove_file(fixture_dir().join(stale)).unwrap();
        }
        return;
    }
    assert_eq!(
        present, listed,
        "the fixture directory and the list in tests/fixtures.rs disagree"
    );
}

#[test]
fn every_type_of_the_contract_has_a_fixture() {
    let entries = every_fixture();
    let types: BTreeSet<&str> = entries.iter().map(|(_, t)| t.as_str()).collect();
    for required in [
        "Identity",
        "Capture<Proprio>",
        "FrameRef",
        "AudioChunk",
        "Proprio",
        "TouchMaps",
        "SpeciesTable",
        "SpeciesVector",
        "Bundle",
        "Action",
        "Reset",
        "GroundTruth",
        "Perturbation",
        "Rates",
    ] {
        assert!(types.contains(required), "no fixture holds a {required}");
    }
    // Both audio storages, all three perturbations, and both camera actions are
    // shown, because a reader has to handle each.
    let files: BTreeSet<&str> = entries.iter().map(|(f, _)| f.as_str()).collect();
    for file in [
        "audio_chunk_host.json",
        "audio_chunk_device.json",
        "perturbation_offset.json",
        "perturbation_delay.json",
        "perturbation_render_from_perturbed_state.json",
        "action.json",
        "action_look_at.json",
    ] {
        assert!(files.contains(file), "{file} is missing");
    }
}

#[test]
fn the_index_marks_the_segmentation_frame_and_ground_truth_evaluation_only() {
    let entries = every_fixture();
    let index = index(&entries);
    let files = &index["fixtures"];
    // Segmentation moved from the bundle to ground truth: its frame fixture is
    // marked, the observation frames are not, and the bundle holds none.
    assert_eq!(
        files["frame_ref_segmentation.json"]["evaluation_only"],
        true
    );
    assert_eq!(files["frame_ref_rgb.json"]["evaluation_only"], false);
    assert_eq!(files["frame_ref_depth.json"]["evaluation_only"], false);
    assert_eq!(files["ground_truth.json"]["evaluation_only"], true);
    assert_eq!(files["bundle.json"]["evaluation_only"], false);
    // Every file named in EVALUATION_ONLY_FILES exists, so the list cannot go
    // stale unnoticed.
    for file in EVALUATION_ONLY_FILES {
        assert!(files.get(file).is_some(), "{file} is not a fixture");
    }
}

#[test]
fn the_fixtures_are_plain_json_a_python_loop_can_read() {
    // No fixture holds a null where a number belongs (serde_json writes a NaN
    // as null) and every file ends with exactly one newline and uses LF.
    for entry in fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|ext| ext != "json") {
            continue;
        }
        let text = fs::read_to_string(&path).unwrap();
        assert!(
            text.ends_with("}\n") && !text.ends_with("\n\n"),
            "{}",
            path.display()
        );
        assert!(
            !text.contains('\r'),
            "{} has CR line endings",
            path.display()
        );
        serde_json::from_str::<serde_json::Value>(&text)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    }
}

#[test]
fn the_full_bundle_fixture_is_a_tick_where_every_sense_is_due() {
    // The example tick (step 50 at 50 Hz) is chosen so every sampled channel of
    // the full bundle is genuinely due under the v0 rates.
    use sim_contract::Sense;
    let bundle = common::bundle_full();
    for sense in Sense::ALL {
        assert!(
            Rates::due(sense, bundle.identity.step, common::CONTROL_HZ),
            "{sense:?}"
        );
    }
    assert!(
        (bundle.identity.sim_time_s * f64::from(common::CONTROL_HZ) - bundle.identity.step as f64)
            .abs()
            < 1e-9
    );
}
