//! The contract's behaviour: JSON round trips for every type, what each
//! `validate` accepts and refuses, the rates rule, and the strict reading of
//! inputs.

mod common;

use std::fmt::Debug;

use serde::Serialize;
use serde::de::DeserializeOwned;
use sim_contract::{
    Action, AudioChunk, AudioContinuity, AudioSamples, BasePose, Bundle, CameraAction, Capture,
    ContactEvent, ContractError, DeviceBufferId, EntityId, FrameDtype, FrameRef, FrameSemantic,
    GroundTruth, Identity, MAX_IDENTIFIER_LEN, NamedValue, Perturbation, Proprio, RATES_V0, Rates,
    Reset, SCHEMA_VERSION, SceneId, SegFrame, SegTable, Sense, SensorActions, Sniff, Species,
    SpeciesTable, SpeciesVector, StateDeltaId, TouchMap, TouchMaps,
};

// -- JSON round trips -------------------------------------------------------

/// Through text and through a `serde_json::Value`, each back to an equal value.
fn round_trip<T>(value: &T)
where
    T: Serialize + DeserializeOwned + PartialEq + Debug,
{
    let text = serde_json::to_string(value).unwrap();
    let back: T = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"));
    assert_eq!(&back, value, "{text}");
    let tree = serde_json::to_value(value).unwrap();
    let back: T = serde_json::from_value(tree).unwrap();
    assert_eq!(&back, value);
}

#[test]
fn every_type_survives_a_json_round_trip() {
    round_trip(&common::identity());
    round_trip(&common::capture_proprio());
    round_trip(&common::rgb());
    round_trip(&common::depth());
    round_trip(&common::segmentation());
    round_trip(&common::seg_frame());
    round_trip(&common::seg_table());
    round_trip(&common::shown_seg_frame());
    round_trip(&common::audio_host());
    round_trip(&common::audio_device());
    round_trip(&common::proprio());
    round_trip(&common::touch_maps());
    round_trip(&common::smell_table());
    round_trip(&common::smell_vector());
    round_trip(&common::bundle_full());
    round_trip(&common::bundle_minimal());
    round_trip(&common::action_pose());
    round_trip(&common::action_look_at());
    round_trip(&common::reset());
    round_trip(&common::ground_truth());
    round_trip(&common::perturbation_offset());
    round_trip(&common::perturbation_delay());
    round_trip(&common::perturbation_render_from_perturbed_state());
    round_trip(&RATES_V0);
    for sense in Sense::ALL {
        round_trip(&sense);
    }
    for dtype in [
        FrameDtype::U8,
        FrameDtype::F16,
        FrameDtype::U16,
        FrameDtype::F32,
    ] {
        round_trip(&dtype);
    }
}

#[test]
fn enums_use_the_documented_json_names() {
    let json = |value: &dyn erased::Json| value.json();
    assert_eq!(json(&Sense::Proprioception), "\"proprioception\"");
    assert_eq!(json(&FrameDtype::F16), "\"f16\"");
    assert_eq!(json(&FrameSemantic::DepthMetres), "\"depth_metres\"");
    let offset = serde_json::to_value(common::perturbation_offset()).unwrap();
    assert_eq!(offset["kind"], "offset");
    assert_eq!(offset["sense"], "sight");
    let render = serde_json::to_value(common::perturbation_render_from_perturbed_state()).unwrap();
    assert_eq!(render["kind"], "render_from_perturbed_state");
    assert_eq!(render["state_delta_id"], "plate_colder_by_20k");
    let look = serde_json::to_value(common::action_look_at()).unwrap();
    assert_eq!(look["sensor"]["camera"]["kind"], "look_at");
    let audio = serde_json::to_value(common::audio_host()).unwrap();
    assert_eq!(audio["samples"]["kind"], "host");
    assert!(audio["samples"]["values"].is_array());
}

mod erased {
    pub trait Json {
        fn json(&self) -> String;
    }
    impl<T: serde::Serialize> Json for T {
        fn json(&self) -> String {
            serde_json::to_string(self).unwrap()
        }
    }
}

#[test]
fn f32_values_survive_json_bit_for_bit() {
    // A deterministic walk over f32 bit patterns, skipping NaN and infinity.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut checked = 0u32;
    while checked < 20_000 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let value = f32::from_bits((state >> 32) as u32);
        if !value.is_finite() {
            continue;
        }
        let text = serde_json::to_string(&value).unwrap();
        let back: f32 = serde_json::from_str(&text).unwrap();
        assert_eq!(back.to_bits(), value.to_bits(), "{value:e} wrote {text}");
        checked += 1;
    }
}

#[test]
fn json_cannot_carry_nan_so_validate_refuses_it_first() {
    let mut proprio = common::proprio();
    proprio.joint_positions[1] = f32::NAN;
    // serde_json writes a NaN as null, and null does not read back as a number.
    let text = serde_json::to_string(&proprio).unwrap();
    assert!(text.contains("null"), "{text}");
    assert!(serde_json::from_str::<Proprio>(&text).is_err());
    assert_eq!(
        proprio.validate(),
        Err(ContractError::NonFinite {
            field: "joint_positions"
        })
    );
}

// -- the samples are valid --------------------------------------------------

#[test]
fn every_example_value_is_valid() {
    common::identity().validate().unwrap();
    common::rgb().validate_v0(FrameSemantic::Rgb).unwrap();
    common::depth()
        .validate_v0(FrameSemantic::DepthMetres)
        .unwrap();
    common::segmentation()
        .validate_v0(FrameSemantic::Segmentation)
        .unwrap();
    common::seg_frame().validate().unwrap();
    common::seg_table().validate().unwrap();
    common::audio_host().validate().unwrap();
    common::audio_device().validate().unwrap();
    common::proprio().validate().unwrap();
    common::touch_maps().validate().unwrap();
    common::smell_table().validate().unwrap();
    common::taste_table().validate().unwrap();
    common::smell_vector()
        .validate(&common::smell_table())
        .unwrap();
    common::taste_vector()
        .validate(&common::taste_table())
        .unwrap();
    common::action_pose().validate().unwrap();
    common::action_look_at().validate().unwrap();
    common::reset().validate().unwrap();
    common::ground_truth().validate().unwrap();
    common::perturbation_offset().validate().unwrap();
    common::perturbation_delay().validate().unwrap();
    common::perturbation_render_from_perturbed_state()
        .validate()
        .unwrap();
    for bundle in [common::bundle_full(), common::bundle_minimal()] {
        bundle
            .validate(&common::smell_table(), &common::taste_table())
            .unwrap();
    }
}

// -- identity ---------------------------------------------------------------

#[test]
fn an_identity_carries_the_schema_version_and_refuses_another() {
    assert_eq!(SCHEMA_VERSION, 0);
    assert_eq!(common::identity().schema_version, SCHEMA_VERSION);
    let mut identity = common::identity();
    identity.schema_version = 1;
    assert_eq!(
        identity.validate(),
        Err(ContractError::SchemaVersion { found: 1 })
    );
    let mut identity = common::identity();
    identity.sim_time_s = f64::INFINITY;
    assert!(matches!(
        identity.validate(),
        Err(ContractError::NonFinite { .. })
    ));
    identity.sim_time_s = -0.5;
    assert!(matches!(
        identity.validate(),
        Err(ContractError::OutOfRange { .. })
    ));
}

#[test]
fn a_bundle_written_under_another_schema_version_is_refused() {
    let mut bundle = common::bundle_full();
    bundle.identity.schema_version = 7;
    let err = bundle
        .validate(&common::smell_table(), &common::taste_table())
        .unwrap_err();
    assert_eq!(err, ContractError::SchemaVersion { found: 7 });
}

// -- frames -----------------------------------------------------------------

#[test]
fn v0_frame_formats_are_the_ones_the_contract_fixes() {
    let rgb = FrameRef::v0_rgb();
    assert_eq!(
        (rgb.width, rgb.height, rgb.channels, rgb.dtype),
        (448, 448, 3, FrameDtype::U8)
    );
    assert_eq!(rgb.row_stride_bytes, 448 * 3);
    assert_eq!(rgb.semantic, FrameSemantic::Rgb);
    let depth = FrameRef::v0_depth();
    assert_eq!(
        (depth.width, depth.height, depth.channels, depth.dtype),
        (448, 448, 1, FrameDtype::F16)
    );
    assert_eq!(depth.row_stride_bytes, 448 * 2);
    assert_eq!(depth.semantic, FrameSemantic::DepthMetres);
    let seg = FrameRef::v0_segmentation();
    assert_eq!(
        (seg.width, seg.height, seg.channels, seg.dtype),
        (448, 448, 1, FrameDtype::U16)
    );
    assert_eq!(seg.row_stride_bytes, 448 * 2);
    assert_eq!(seg.semantic, FrameSemantic::Segmentation);
    assert_eq!(rgb.byte_extent(), Some(448 * 448 * 3));
    let placed = rgb.at(DeviceBufferId(42), 4096);
    assert_eq!(
        (placed.buffer, placed.byte_offset),
        (DeviceBufferId(42), 4096)
    );
    assert_eq!(placed.width, 448);
}

#[test]
fn a_frame_may_pad_its_rows_but_not_overlap_them() {
    let mut frame = FrameRef::v0_rgb();
    frame.row_stride_bytes = 1536; // padded to a 256-byte multiple
    frame.validate().unwrap();
    frame.row_stride_bytes = 1343;
    assert_eq!(
        frame.validate(),
        Err(ContractError::Stride {
            required_bytes: 1344,
            found_bytes: 1343
        })
    );
}

#[test]
fn frame_layout_is_checked_against_the_dtype() {
    let mut frame = FrameRef::v0_depth(); // f16: 2-byte elements
    frame.byte_offset = 3;
    assert_eq!(
        frame.validate(),
        Err(ContractError::Misaligned {
            field: "byte_offset",
            alignment: 2,
            found: 3
        })
    );
    let mut frame = FrameRef::v0_depth();
    frame.row_stride_bytes = 897;
    assert_eq!(
        frame.validate(),
        Err(ContractError::Misaligned {
            field: "row_stride_bytes",
            alignment: 2,
            found: 897
        })
    );
    // The same offset is fine for one-byte elements.
    let mut frame = FrameRef::v0_rgb();
    frame.byte_offset = 3;
    frame.validate().unwrap();
    // A packed f32 row is 4 bytes per element.
    let float = FrameRef {
        buffer: DeviceBufferId(1),
        byte_offset: 0,
        width: 4,
        height: 2,
        channels: 3,
        dtype: FrameDtype::F32,
        row_stride_bytes: 47,
        semantic: FrameSemantic::Rgb,
    };
    assert_eq!(
        float.validate(),
        Err(ContractError::Stride {
            required_bytes: 48,
            found_bytes: 47
        })
    );
}

#[test]
fn empty_dimensions_and_overflowing_extents_are_refused() {
    for (field, edit) in [
        (
            "width",
            (|f: &mut FrameRef| f.width = 0) as fn(&mut FrameRef),
        ),
        ("height", |f| f.height = 0),
        ("channels", |f| f.channels = 0),
    ] {
        let mut frame = FrameRef::v0_rgb();
        edit(&mut frame);
        assert_eq!(frame.validate(), Err(ContractError::Empty { field }));
    }
    let mut frame = FrameRef::v0_rgb();
    frame.byte_offset = u64::MAX - 1000;
    assert!(matches!(
        frame.validate(),
        Err(ContractError::OutOfRange {
            field: "byte_offset",
            ..
        })
    ));
}

#[test]
fn a_frame_must_be_the_v0_format_of_its_channel() {
    let rgb_as_depth = FrameRef::v0_rgb().validate_v0(FrameSemantic::DepthMetres);
    assert!(matches!(rgb_as_depth, Err(ContractError::Format { .. })));
    let mut wrong_dtype = FrameRef::v0_depth();
    wrong_dtype.dtype = FrameDtype::F32;
    wrong_dtype.row_stride_bytes = 448 * 4;
    assert!(matches!(
        wrong_dtype.validate_v0(FrameSemantic::DepthMetres),
        Err(ContractError::Format { .. })
    ));
    let mut wrong_channels = FrameRef::v0_rgb();
    wrong_channels.channels = 4;
    wrong_channels.row_stride_bytes = 448 * 4;
    assert!(matches!(
        wrong_channels.validate_v0(FrameSemantic::Rgb),
        Err(ContractError::Format { .. })
    ));
    let mut small = FrameRef::v0_segmentation();
    small.width = 224;
    assert!(matches!(
        small.validate_v0(FrameSemantic::Segmentation),
        Err(ContractError::Format { .. })
    ));
    // A frame that fails the layout check fails that check first.
    let mut broken = FrameRef::v0_rgb();
    broken.width = 0;
    assert!(matches!(
        broken.validate_v0(FrameSemantic::Rgb),
        Err(ContractError::Empty { .. })
    ));
}

// -- audio ------------------------------------------------------------------

#[test]
fn audio_must_continue_where_the_previous_chunk_ended() {
    let first = AudioChunk::host(0, vec![0.0; 320]);
    let second = AudioChunk::host(320, vec![0.0; 320]);
    let third = AudioChunk::host(640, vec![0.0; 320]);
    AudioContinuity::check(&first, &second).unwrap();
    AudioContinuity::check(&second, &third).unwrap();
    // A skipped chunk is a gap.
    assert_eq!(
        AudioContinuity::check(&first, &third),
        Err(ContractError::AudioDiscontinuity {
            expected_first_sample_index: 320,
            found_first_sample_index: 640
        })
    );
    // A repeated chunk is an overlap.
    assert_eq!(
        AudioContinuity::check(&second, &AudioChunk::host(0, vec![0.0; 320])),
        Err(ContractError::AudioDiscontinuity {
            expected_first_sample_index: 640,
            found_first_sample_index: 0
        })
    );
    // An off-by-one is caught too.
    assert!(AudioContinuity::check(&first, &AudioChunk::host(321, vec![])).is_err());
}

#[test]
fn an_empty_chunk_keeps_its_place_in_the_stream() {
    let empty = AudioChunk::empty(640);
    assert_eq!(empty.sample_count(), 0);
    assert_eq!(empty.end_sample_index(), Some(640));
    AudioContinuity::check(&AudioChunk::host(320, vec![0.0; 320]), &empty).unwrap();
    AudioContinuity::check(&empty, &AudioChunk::host(640, vec![0.5])).unwrap();
    assert!(AudioContinuity::check(&empty, &AudioChunk::host(641, vec![0.5])).is_err());
}

#[test]
fn continuity_counts_device_samples_and_checks_the_rate() {
    let device = common::audio_device(); // 16000..16320
    assert_eq!(device.sample_count(), 320);
    AudioContinuity::check(&device, &AudioChunk::host(16_320, vec![0.0])).unwrap();
    assert!(AudioContinuity::check(&device, &AudioChunk::host(16_000, vec![0.0])).is_err());
    let mut other_rate = AudioChunk::host(16_320, vec![0.0]);
    other_rate.sample_rate_hz = 48_000;
    assert!(matches!(
        AudioContinuity::check(&device, &other_rate),
        Err(ContractError::OutOfRange {
            field: "sample_rate_hz",
            ..
        })
    ));
}

#[test]
fn continuity_does_not_overflow_at_the_end_of_the_index_space() {
    let last = AudioChunk::host(u64::MAX, vec![0.0]);
    assert!(matches!(
        last.validate(),
        Err(ContractError::OutOfRange { .. })
    ));
    assert!(matches!(
        AudioContinuity::check(&last, &AudioChunk::host(0, vec![])),
        Err(ContractError::OutOfRange { .. })
    ));
}

#[test]
fn audio_is_16_khz_mono_f32_and_finite() {
    let mut chunk = common::audio_host();
    chunk.sample_rate_hz = 44_100;
    assert!(matches!(
        chunk.validate(),
        Err(ContractError::OutOfRange {
            field: "sample_rate_hz",
            ..
        })
    ));
    let chunk = AudioChunk::host(0, vec![0.0, f32::NAN]);
    assert_eq!(
        chunk.validate(),
        Err(ContractError::NonFinite { field: "samples" })
    );
    let misaligned = AudioChunk {
        sample_rate_hz: 16_000,
        first_sample_index: 0,
        samples: AudioSamples::Device {
            buffer: DeviceBufferId(1),
            byte_offset: 6,
            count: 10,
        },
    };
    assert_eq!(
        misaligned.validate(),
        Err(ContractError::Misaligned {
            field: "byte_offset",
            alignment: 4,
            found: 6
        })
    );
    let huge = AudioChunk {
        sample_rate_hz: 16_000,
        first_sample_index: 0,
        samples: AudioSamples::Device {
            buffer: DeviceBufferId(1),
            byte_offset: 0,
            count: u64::MAX,
        },
    };
    assert!(matches!(
        huge.validate(),
        Err(ContractError::OutOfRange { field: "count", .. })
    ));
}

// -- proprioception ---------------------------------------------------------

#[test]
fn proprioception_lists_are_parallel_and_the_pose_is_a_unit_quaternion() {
    let mut proprio = common::proprio();
    proprio.joint_velocities.pop();
    assert_eq!(
        proprio.validate(),
        Err(ContractError::LengthMismatch {
            field: "joint_velocities",
            expected: 3,
            found: 2
        })
    );
    let mut proprio = common::proprio();
    proprio.joint_velocities[2] = f32::INFINITY;
    assert_eq!(
        proprio.validate(),
        Err(ContractError::NonFinite {
            field: "joint_velocities"
        })
    );
    let mut proprio = common::proprio();
    proprio.base_pose = Some(BasePose {
        position: [0.0; 3],
        orientation_quat: [0.0, 0.0, 0.0, 2.0],
    });
    assert!(matches!(
        proprio.validate(),
        Err(ContractError::OutOfRange {
            field: "base_pose.orientation_quat",
            ..
        })
    ));
    Proprio::empty().validate().unwrap();
}

// -- touch ------------------------------------------------------------------

fn one_sensor() -> TouchMap {
    common::touch_maps().sensors.remove(0)
}

#[test]
fn touch_maps_are_validated_for_shape() {
    let mut map = one_sensor();
    map.pressure_pa.pop();
    assert_eq!(
        map.validate(),
        Err(ContractError::LengthMismatch {
            field: "pressure_pa",
            expected: 4,
            found: 3
        })
    );
    let mut map = one_sensor();
    map.shear_pa.push([0.0, 0.0]);
    assert_eq!(
        map.validate(),
        Err(ContractError::LengthMismatch {
            field: "shear_pa",
            expected: 4,
            found: 5
        })
    );
    let mut map = one_sensor();
    map.temperature_k.clear();
    assert_eq!(
        map.validate(),
        Err(ContractError::LengthMismatch {
            field: "temperature_k",
            expected: 4,
            found: 0
        })
    );
    let mut map = one_sensor();
    map.rows = 3; // the lists still hold 2 x 2
    assert!(matches!(
        map.validate(),
        Err(ContractError::LengthMismatch { expected: 6, .. })
    ));
    let mut map = one_sensor();
    map.cols = 0;
    assert_eq!(map.validate(), Err(ContractError::Empty { field: "cols" }));
}

#[test]
fn touch_values_are_physical() {
    let mut map = one_sensor();
    map.pressure_pa[0] = -1.0;
    assert!(matches!(
        map.validate(),
        Err(ContractError::OutOfRange {
            field: "pressure_pa",
            ..
        })
    ));
    let mut map = one_sensor();
    map.temperature_k[3] = 0.0;
    assert!(matches!(
        map.validate(),
        Err(ContractError::OutOfRange {
            field: "temperature_k",
            ..
        })
    ));
    let mut map = one_sensor();
    map.shear_pa[1][0] = f32::NAN;
    assert_eq!(
        map.validate(),
        Err(ContractError::NonFinite { field: "shear_pa" })
    );
}

#[test]
fn touch_sensor_ids_are_unique_and_no_sensors_is_valid() {
    let mut maps = TouchMaps {
        sensors: vec![one_sensor(), one_sensor()],
    };
    assert_eq!(
        maps.validate(),
        Err(ContractError::DuplicateId {
            field: "sensors",
            id: 4
        })
    );
    maps.sensors[1].sensor_id = 5;
    maps.validate().unwrap();
    TouchMaps::default().validate().unwrap();
}

// -- smell and taste --------------------------------------------------------

#[test]
fn a_species_table_refuses_duplicate_cids() {
    let mut table = common::smell_table();
    table.species.push(Species {
        pubchem_cid: 702,
        name: "ethyl alcohol".to_string(),
    });
    assert_eq!(
        table.validate(),
        Err(ContractError::DuplicateCid { pubchem_cid: 702 })
    );
    // The check is on the CID, not the name: one name may stand for two CIDs.
    let mut table = common::smell_table();
    table.species.push(Species {
        pubchem_cid: 962,
        name: "ethanol".to_string(),
    });
    table.validate().unwrap();
    // 0 is not a CID.
    let mut table = common::smell_table();
    table.species[0].pubchem_cid = 0;
    assert!(matches!(
        table.validate(),
        Err(ContractError::OutOfRange {
            field: "pubchem_cid",
            ..
        })
    ));
}

#[test]
fn a_vector_is_checked_against_its_table() {
    let table = common::smell_table();
    // A duplicate in the table is refused when a vector is checked against it.
    let mut duplicated = table.clone();
    duplicated.species[1].pubchem_cid = 702;
    assert_eq!(
        common::smell_vector().validate(&duplicated),
        Err(ContractError::DuplicateCid { pubchem_cid: 702 })
    );
    // Another table version.
    let mut vector = common::smell_vector();
    vector.table_version = 2;
    assert_eq!(
        vector.validate(&table),
        Err(ContractError::TableVersion {
            vector: 2,
            table: 1
        })
    );
    // Wrong length, short and long.
    for len in [2usize, 4] {
        let mut vector = common::smell_vector();
        vector.concentrations.resize(len, 0.0);
        assert_eq!(
            vector.validate(&table),
            Err(ContractError::LengthMismatch {
                field: "concentrations",
                expected: 3,
                found: len as u64
            })
        );
    }
    // Negative and NaN concentrations.
    let mut vector = common::smell_vector();
    vector.concentrations[0] = -1e-9;
    assert!(matches!(
        vector.validate(&table),
        Err(ContractError::OutOfRange { .. })
    ));
    vector.concentrations[0] = f32::NAN;
    assert!(matches!(
        vector.validate(&table),
        Err(ContractError::NonFinite { .. })
    ));
}

#[test]
fn an_empty_table_is_refused_for_a_non_empty_vector_only() {
    let empty_table = SpeciesTable {
        version: 1,
        species: Vec::new(),
    };
    empty_table.validate().unwrap();
    SpeciesVector::empty(1).validate(&empty_table).unwrap();
    let vector = SpeciesVector {
        table_version: 1,
        concentrations: vec![0.5],
    };
    assert_eq!(
        vector.validate(&empty_table),
        Err(ContractError::EmptyTableForNonEmptyVector)
    );
}

#[test]
fn species_are_found_by_cid_not_by_name() {
    let table = common::taste_table();
    assert_eq!(table.position_of(5988), Some(1));
    assert_eq!(table.position_of(33_032), Some(2));
    assert_eq!(table.position_of(1), None);
}

// -- the bundle -------------------------------------------------------------

fn bundle_json() -> serde_json::Value {
    serde_json::to_value(common::bundle_full()).unwrap()
}

#[test]
fn a_bundle_writes_every_channel_even_when_it_was_not_sampled() {
    let value = serde_json::to_value(common::bundle_minimal()).unwrap();
    let object = value.as_object().unwrap();
    let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "audio",
            "depth",
            "identity",
            "proprioception",
            "rgb",
            "smell",
            "taste",
            "touch"
        ]
    );
    for key in ["rgb", "depth", "touch", "smell", "taste"] {
        assert!(object[key].is_null(), "{key} is null, not left out");
    }
    // The channels that always carry a value carry an empty one.
    assert_eq!(
        value["audio"]["value"]["samples"]["values"]
            .as_array()
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        value["proprioception"]["value"]["joint_positions"]
            .as_array()
            .unwrap()
            .len(),
        0
    );
}

#[test]
fn reading_a_bundle_with_a_channel_left_out_is_an_error() {
    for key in [
        "identity",
        "rgb",
        "depth",
        "audio",
        "proprioception",
        "touch",
        "smell",
        "taste",
    ] {
        let mut value = bundle_json();
        value.as_object_mut().unwrap().remove(key);
        let err = serde_json::from_value::<Bundle>(value)
            .unwrap_err()
            .to_string();
        assert!(err.contains(key), "{key}: {err}");
    }
    // Present and null is a channel that was not sampled.
    let mut value = bundle_json();
    value["rgb"] = serde_json::Value::Null;
    let bundle: Bundle = serde_json::from_value(value).unwrap();
    assert!(bundle.rgb.is_none());
}

#[test]
fn depth_comes_only_with_rgb() {
    let mut bundle = common::bundle_full();
    bundle.rgb = None;
    let tables = (common::smell_table(), common::taste_table());
    assert_eq!(
        bundle.validate(&tables.0, &tables.1),
        Err(ContractError::ExtraChannelWithoutRgb { channel: "depth" })
    );
    bundle.depth = None;
    bundle.validate(&tables.0, &tables.1).unwrap();
    // Sight alone, with no depth, is fine.
    let mut bundle = common::bundle_full();
    bundle.depth = None;
    bundle.validate(&tables.0, &tables.1).unwrap();
}

#[test]
fn a_segmentation_frame_is_refused_in_every_frame_channel_of_a_bundle() {
    // Segmentation is evaluation only: it has no channel of its own in a bundle,
    // and the two frame channels a bundle has refuse a segmentation frame.
    let tables = (common::smell_table(), common::taste_table());
    for channel in ["rgb", "depth"] {
        let mut bundle = common::bundle_full();
        let slot = if channel == "rgb" {
            &mut bundle.rgb
        } else {
            &mut bundle.depth
        };
        *slot = Some(Capture::new(1.0, common::segmentation()));
        let err = bundle.validate(&tables.0, &tables.1).unwrap_err();
        assert!(
            matches!(err, ContractError::Channel { channel: c, .. } if c == channel),
            "{channel}: {err:?}"
        );
        assert!(
            matches!(err.root(), ContractError::Format { .. }),
            "{channel}: {err:?}"
        );
    }
}

#[test]
fn a_bundle_refuses_a_frame_in_the_wrong_channel_and_names_the_channel() {
    let tables = (common::smell_table(), common::taste_table());
    let mut bundle = common::bundle_full();
    bundle.rgb = Some(Capture::new(1.0, common::depth()));
    let err = bundle.validate(&tables.0, &tables.1).unwrap_err();
    assert!(
        matches!(err, ContractError::Channel { channel: "rgb", .. }),
        "{err:?}"
    );
    assert!(matches!(err.root(), ContractError::Format { .. }));
    assert!(err.to_string().starts_with("rgb: "), "{err}");

    let mut bundle = common::bundle_full();
    bundle.audio.value.sample_rate_hz = 8_000;
    let err = bundle.validate(&tables.0, &tables.1).unwrap_err();
    assert!(matches!(
        err,
        ContractError::Channel {
            channel: "audio",
            ..
        }
    ));

    let mut bundle = common::bundle_full();
    bundle.touch.as_mut().unwrap().value.sensors[0].rows = 9;
    let err = bundle.validate(&tables.0, &tables.1).unwrap_err();
    assert!(matches!(
        err,
        ContractError::Channel {
            channel: "touch",
            ..
        }
    ));

    let mut bundle = common::bundle_full();
    bundle.proprioception.captured_at_s = f64::NAN;
    let err = bundle.validate(&tables.0, &tables.1).unwrap_err();
    assert!(matches!(
        err,
        ContractError::Channel {
            channel: "proprioception",
            ..
        }
    ));
    assert!(matches!(err.root(), ContractError::NonFinite { .. }));
}

#[test]
fn a_bundle_checks_smell_and_taste_against_their_own_tables() {
    let tables = (common::smell_table(), common::taste_table());
    let mut bundle = common::bundle_full();
    bundle.smell.as_mut().unwrap().value.table_version = 9;
    let err = bundle.validate(&tables.0, &tables.1).unwrap_err();
    assert!(matches!(
        err,
        ContractError::Channel {
            channel: "smell",
            ..
        }
    ));
    assert_eq!(
        *err.root(),
        ContractError::TableVersion {
            vector: 9,
            table: 1
        }
    );
    let mut bundle = common::bundle_full();
    bundle.taste.as_mut().unwrap().value.concentrations.pop();
    let err = bundle.validate(&tables.0, &tables.1).unwrap_err();
    assert!(matches!(
        err,
        ContractError::Channel {
            channel: "taste",
            ..
        }
    ));
    // Swapping the tables is caught: the smell vector is not a taste vector.
    let mut bundle = common::bundle_full();
    bundle
        .smell
        .as_mut()
        .unwrap()
        .value
        .concentrations
        .push(0.0);
    assert!(bundle.validate(&tables.0, &tables.1).is_err());
}

#[test]
fn an_empty_bundle_has_every_channel_and_no_content() {
    let identity = Identity::new(1, 2, 0, 0.0, 3);
    let bundle = Bundle::empty(identity, 640);
    assert!(bundle.rgb.is_none() && bundle.depth.is_none());
    assert!(bundle.touch.is_none() && bundle.smell.is_none() && bundle.taste.is_none());
    assert_eq!(bundle.audio.value.sample_count(), 0);
    assert_eq!(bundle.audio.value.first_sample_index, 640);
    assert_eq!(bundle.proprioception.value, Proprio::empty());
    let empty_table = SpeciesTable {
        version: 0,
        species: Vec::new(),
    };
    bundle.validate(&empty_table, &empty_table).unwrap();
}

// -- actions and resets -----------------------------------------------------

fn action_json() -> serde_json::Value {
    serde_json::to_value(common::action_pose()).unwrap()
}

#[test]
fn inputs_refuse_unknown_fields() {
    // Action, at each level.
    let mut value = action_json();
    value["extra"] = 1.into();
    assert!(Action::from_json(&value.to_string()).is_err());
    let mut value = action_json();
    value["sensor"]["extra"] = 1.into();
    assert!(Action::from_json(&value.to_string()).is_err());
    let mut value = action_json();
    value["sensor"]["camera"]["extra"] = 1.into();
    assert!(Action::from_json(&value.to_string()).is_err());
    let mut value = action_json();
    value["sensor"]["sniff"]["extra"] = 1.into();
    assert!(Action::from_json(&value.to_string()).is_err());
    // A misspelt field is an unknown field, not a silent default.
    let mut value = action_json();
    let sniff = value["sensor"].as_object_mut().unwrap();
    let moved = sniff.remove("sniff").unwrap();
    sniff.insert("sniffs".to_string(), moved);
    assert!(Action::from_json(&value.to_string()).is_err());
    // Reset.
    let mut value = serde_json::to_value(common::reset()).unwrap();
    value["extra"] = true.into();
    assert!(Reset::from_json(&value.to_string()).is_err());
    // Perturbation, every variant.
    for perturbation in [
        common::perturbation_offset(),
        common::perturbation_delay(),
        common::perturbation_render_from_perturbed_state(),
    ] {
        let mut value = serde_json::to_value(&perturbation).unwrap();
        value["extra"] = 0.into();
        assert!(Perturbation::from_json(&value.to_string()).is_err());
        let text = serde_json::to_string(&perturbation).unwrap();
        assert_eq!(Perturbation::from_json(&text).unwrap(), perturbation);
    }
}

#[test]
fn from_json_reads_a_valid_input_and_names_the_failure_of_an_invalid_one() {
    let text = serde_json::to_string(&common::action_pose()).unwrap();
    assert_eq!(Action::from_json(&text).unwrap(), common::action_pose());
    let text = serde_json::to_string(&common::reset()).unwrap();
    assert_eq!(Reset::from_json(&text).unwrap(), common::reset());
    assert!(matches!(
        Action::from_json("{not json"),
        Err(ContractError::Json { .. })
    ));
    // Valid JSON, invalid value: validation runs after reading.
    let mut bad = common::action_pose();
    bad.sensor.sniff = Some(Sniff {
        intensity: 1.5,
        duration_s: 0.5,
    });
    let text = serde_json::to_string(&bad).unwrap();
    assert!(matches!(
        Action::from_json(&text),
        Err(ContractError::OutOfRange {
            field: "sniff.intensity",
            ..
        })
    ));
}

#[test]
fn an_action_may_leave_its_optional_sensor_actions_out() {
    let action =
        Action::from_json(r#"{"identity_step": 3, "joint_targets": [0.5], "sensor": {}}"#).unwrap();
    assert_eq!(action.sensor, SensorActions::default());
    assert!(Action::from_json(r#"{"identity_step": 3, "joint_targets": [0.5]}"#).is_err());
    assert!(Action::from_json(r#"{"joint_targets": [0.5], "sensor": {}}"#).is_err());
}

#[test]
fn sniff_intensity_is_in_zero_to_one() {
    let sniff = |intensity, duration_s| Sniff {
        intensity,
        duration_s,
    };
    sniff(0.0, 0.1).validate().unwrap();
    sniff(1.0, 0.1).validate().unwrap();
    for bad in [-0.01, 1.01, f32::NAN, f32::INFINITY] {
        assert!(sniff(bad, 0.1).validate().is_err(), "{bad}");
    }
    for bad in [0.0, -1.0, f32::NAN] {
        assert!(sniff(0.5, bad).validate().is_err(), "{bad}");
    }
}

#[test]
fn camera_actions_are_checked() {
    let pose = |q: [f32; 4]| CameraAction::Pose {
        position: [0.0; 3],
        orientation_quat: q,
    };
    pose([0.0, 0.0, 0.0, 1.0]).validate().unwrap();
    pose([0.0, 0.0, 0.0, 1.0005]).validate().unwrap(); // within f32 slack
    assert!(pose([0.0, 0.0, 0.0, 0.0]).validate().is_err());
    assert!(pose([0.0, 0.0, 0.0, 1.1]).validate().is_err());
    assert!(pose([f32::NAN, 0.0, 0.0, 1.0]).validate().is_err());
    let look = |target: [f32; 3], up: [f32; 3]| CameraAction::LookAt { target, up };
    look([1.0, 2.0, 3.0], [0.0, 0.0, 1.0]).validate().unwrap();
    assert!(look([1.0, 2.0, 3.0], [0.0; 3]).validate().is_err());
    assert!(
        look([f32::INFINITY, 0.0, 0.0], [0.0, 0.0, 1.0])
            .validate()
            .is_err()
    );
    assert!(look([0.0; 3], [0.0, f32::NAN, 1.0]).validate().is_err());
}

#[test]
fn joint_targets_must_be_finite() {
    let mut action = common::action_pose();
    action.joint_targets[1] = f32::NAN;
    assert_eq!(
        action.validate(),
        Err(ContractError::NonFinite {
            field: "joint_targets"
        })
    );
}

#[test]
fn scene_ids_cannot_name_a_path() {
    let reset = |scene: &str| Reset {
        env_id: 0,
        seed: 0,
        scene: SceneId(scene.to_string()),
    };
    for good in ["kitchen", "manipulation_table_v0", "a", "scene-2.1", "0abc"] {
        reset(good).validate().unwrap();
        assert_eq!(reset(good).scene.as_str(), good);
    }
    let too_long = "a".repeat(MAX_IDENTIFIER_LEN + 1);
    let longest = "a".repeat(MAX_IDENTIFIER_LEN);
    reset(&longest).validate().unwrap();
    for bad in [
        "",
        "..",
        "../x",
        "a/b",
        "a\\b",
        ".hidden",
        "-flag",
        "Upper",
        "has space",
        "caf\u{e9}",
        "nul\0",
        too_long.as_str(),
    ] {
        assert_eq!(
            reset(bad).validate(),
            Err(ContractError::BadIdentifier { field: "scene" }),
            "{bad:?}"
        );
    }
}

// -- ground truth -----------------------------------------------------------

#[test]
fn ground_truth_is_checked_for_physical_sense() {
    let truth = common::ground_truth;
    let mut bad = truth();
    bad.temperatures_k[0].value = 0.0;
    assert!(matches!(
        bad.validate(),
        Err(ContractError::OutOfRange {
            field: "temperatures_k",
            ..
        })
    ));
    let mut bad = truth();
    bad.reaction_progress[0].value = 1.5;
    assert!(matches!(
        bad.validate(),
        Err(ContractError::OutOfRange {
            field: "reaction_progress",
            ..
        })
    ));
    let mut bad = truth();
    bad.reaction_progress[1].value = f64::NAN;
    assert!(matches!(
        bad.validate(),
        Err(ContractError::NonFinite { .. })
    ));
    let mut bad = truth();
    bad.reaction_progress[1].name = bad.reaction_progress[0].name.clone();
    assert_eq!(
        bad.validate(),
        Err(ContractError::DuplicateName {
            field: "reaction_progress"
        })
    );
    let mut bad = truth();
    bad.temperatures_k.push(NamedValue {
        name: String::new(),
        value: 300.0,
    });
    assert_eq!(
        bad.validate(),
        Err(ContractError::Empty {
            field: "temperatures_k"
        })
    );
    let mut bad = truth();
    bad.contacts[0].b = bad.contacts[0].a;
    assert!(bad.validate().is_err());
    let mut bad = truth();
    bad.contacts[0].normal_force_n = -1.0;
    assert!(bad.validate().is_err());
    let mut bad = truth();
    bad.contacts.push(ContactEvent {
        a: EntityId(1),
        b: EntityId(3),
        normal_force_n: 1.0,
        position: [f32::NAN, 0.0, 0.0],
    });
    assert!(bad.validate().is_err());
    let mut bad = truth();
    bad.object_ids_through_occlusion.push((EntityId(2), true));
    assert_eq!(
        bad.validate(),
        Err(ContractError::DuplicateId {
            field: "object_ids_through_occlusion",
            id: 2
        })
    );
    let mut bad = truth();
    bad.identity.schema_version = 3;
    assert!(matches!(
        bad.validate(),
        Err(ContractError::SchemaVersion { .. })
    ));
    GroundTruth::empty(common::identity()).validate().unwrap();
}

#[test]
fn ground_truth_carries_segmentation_frames() {
    let truth = common::ground_truth();
    assert_eq!(truth.segmentation.len(), 1);
    let seg = &truth.segmentation[0];
    assert_eq!(seg.camera, 0);
    assert_eq!(seg.frame.captured_at_s, 1.0);
    assert_eq!(seg.frame.value, common::segmentation());
    let frame = &seg.frame.value;
    assert_eq!(
        (
            frame.semantic,
            frame.dtype,
            frame.channels,
            frame.width,
            frame.height
        ),
        (FrameSemantic::Segmentation, FrameDtype::U16, 1, 448, 448)
    );
    seg.validate().unwrap();
    truth.validate().unwrap();

    // The JSON shape a Python loop reads.
    let value = serde_json::to_value(&truth).unwrap();
    let first = &value["segmentation"][0];
    assert_eq!(first["camera"], 0);
    assert_eq!(first["frame"]["captured_at_s"], 1.0);
    assert_eq!(first["frame"]["value"]["semantic"], "segmentation");
    assert_eq!(first["frame"]["value"]["dtype"], "u16");
    assert_eq!(first["frame"]["value"]["width"], 448);

    // Any number of cameras, and none, is valid.
    let mut two_cameras = common::ground_truth();
    two_cameras.segmentation.push(SegFrame {
        camera: 1,
        frame: Capture::new(1.0, FrameRef::v0_segmentation().at(DeviceBufferId(10), 0)),
    });
    two_cameras.validate().unwrap();
    let empty = GroundTruth::empty(common::identity());
    assert!(empty.segmentation.is_empty());
    empty.validate().unwrap();

    // Like every other list, each key is always written (`null` for a table that
    // this record does not carry), and reading a record with one left out is an
    // error.
    for key in ["segmentation", "seg_table", "shown_segmentation"] {
        let mut value = serde_json::to_value(common::ground_truth()).unwrap();
        value.as_object_mut().unwrap().remove(key);
        let err = serde_json::from_value::<GroundTruth>(value)
            .unwrap_err()
            .to_string();
        assert!(err.contains(key), "{key}: {err}");
    }
    let mut no_table = common::ground_truth();
    no_table.seg_table = None;
    let value = serde_json::to_value(&no_table).unwrap();
    assert!(value["seg_table"].is_null());
    assert_eq!(
        serde_json::from_value::<GroundTruth>(value).unwrap(),
        no_table
    );
    no_table.validate().unwrap();
}

#[test]
fn a_record_has_at_most_one_true_seg_frame_per_camera() {
    // Two cameras, one frame each: fine.
    let mut truth = common::ground_truth();
    truth.segmentation.push(SegFrame {
        camera: 1,
        frame: Capture::new(1.0, FrameRef::v0_segmentation().at(DeviceBufferId(10), 0)),
    });
    truth.validate().unwrap();

    // The same camera twice is refused, naming the channel and the camera, even
    // when the second frame is otherwise valid and at another time.
    let mut bad = common::ground_truth();
    bad.segmentation.push(SegFrame {
        camera: 0,
        frame: Capture::new(0.98, FrameRef::v0_segmentation().at(DeviceBufferId(11), 0)),
    });
    let err = bad.validate().unwrap_err();
    assert_eq!(
        err,
        ContractError::Channel {
            channel: "segmentation",
            cause: Box::new(ContractError::DuplicateId {
                field: "camera",
                id: 0
            }),
        }
    );
    assert_eq!(err.to_string(), "segmentation: camera lists id 0 twice");

    // A duplicate that is not adjacent, on a camera other than 0.
    let mut bad = common::ground_truth();
    for camera in [3, 4, 3] {
        bad.segmentation.push(SegFrame {
            camera,
            frame: Capture::new(1.0, common::segmentation()),
        });
    }
    assert_eq!(
        bad.validate().unwrap_err().root(),
        &ContractError::DuplicateId {
            field: "camera",
            id: 3
        }
    );
}

#[test]
fn the_seg_table_maps_ids_to_entities_and_is_checked() {
    let table = common::seg_table();
    table.validate().unwrap();
    let truth = common::ground_truth();
    assert_eq!(truth.seg_table.as_ref(), Some(&table));
    // The JSON is a list of [seg_id, entity_id] pairs under the episode.
    assert_eq!(
        serde_json::to_value(&table).unwrap(),
        serde_json::json!({ "episode_id": 12, "entries": [[41, 1], [7, 2], [23, 5]] })
    );
    // The example's ids are assigned, not derived from the entities: neither
    // equal to the entity id nor in its order.
    assert!(table.entries.iter().all(|(id, e)| u64::from(*id) != e.0));
    assert!(!table.entries.windows(2).all(|w| w[0].0 < w[1].0));
    // No entries is a valid table; the largest id is a valid id.
    SegTable {
        episode_id: 0,
        entries: Vec::new(),
    }
    .validate()
    .unwrap();
    SegTable {
        episode_id: 0,
        entries: vec![(u16::MAX, EntityId(u64::MAX))],
    }
    .validate()
    .unwrap();

    // 0 is background and never an entity.
    let mut bad = common::seg_table();
    bad.entries.push((0, EntityId(9)));
    assert!(matches!(
        bad.validate(),
        Err(ContractError::OutOfRange {
            field: "seg_table.seg_id",
            ..
        })
    ));
    // Seg ids are unique.
    let mut bad = common::seg_table();
    bad.entries.push((7, EntityId(9)));
    assert_eq!(
        bad.validate(),
        Err(ContractError::DuplicateId {
            field: "seg_table.seg_id",
            id: 7
        })
    );
    // Entity ids are unique.
    let mut bad = common::seg_table();
    bad.entries.push((8, EntityId(5)));
    assert_eq!(
        bad.validate(),
        Err(ContractError::DuplicateId {
            field: "seg_table.entity_id",
            id: 5
        })
    );
    // A record carries its table's refusals, and a record with no table is fine.
    for bad_entry in [(0, EntityId(9)), (7, EntityId(9)), (8, EntityId(5))] {
        let mut truth = common::ground_truth();
        truth.seg_table.as_mut().unwrap().entries.push(bad_entry);
        assert!(truth.validate().is_err(), "{bad_entry:?}");
    }
    let mut truth = common::ground_truth();
    truth.seg_table = None;
    truth.validate().unwrap();
}

#[test]
fn a_shown_seg_frame_goes_beside_the_true_one_and_is_checked() {
    let truth = common::ground_truth();
    // Beside, never instead: the record holds the true frame for the camera and
    // the shown one, and they are different frames.
    assert_eq!(truth.segmentation.len(), 1);
    assert_eq!(truth.shown_segmentation.len(), 1);
    let shown = &truth.shown_segmentation[0];
    assert_eq!(shown.frame.camera, truth.segmentation[0].camera);
    assert_ne!(shown.frame, truth.segmentation[0]);
    truth.validate().unwrap();
    shown.frame.validate().unwrap();

    // The JSON shape a Python loop reads.
    let value = serde_json::to_value(&truth).unwrap();
    let first = &value["shown_segmentation"][0];
    assert_eq!(
        first["perturbation"],
        serde_json::json!({ "kind": "offset", "sense": "sight", "seconds": -0.05 })
    );
    assert_eq!(first["frame"]["camera"], 0);
    assert_eq!(first["frame"]["frame"]["captured_at_s"], 0.95);
    assert_eq!(first["frame"]["frame"]["value"]["semantic"], "segmentation");

    // No sight perturbation active: no shown frames, and that is valid.
    let mut none = common::ground_truth();
    none.shown_segmentation.clear();
    none.validate().unwrap();

    // Every kind of sight perturbation is accepted.
    for perturbation in [
        Perturbation::Offset {
            sense: Sense::Sight,
            seconds: 0.25,
        },
        Perturbation::Delay {
            sense: Sense::Sight,
            seconds: 0.1,
        },
        Perturbation::RenderFromPerturbedState {
            sense: Sense::Sight,
            state_delta_id: StateDeltaId("plate_colder_by_20k".to_string()),
        },
    ] {
        let mut ok = common::ground_truth();
        ok.shown_segmentation[0].perturbation = perturbation;
        ok.validate().unwrap();
    }
}

#[test]
fn a_shown_frame_with_no_true_frame_for_its_camera_is_refused() {
    let refused = ContractError::Channel {
        channel: "shown_segmentation",
        cause: Box::new(ContractError::ShownWithoutTrueSegmentation { camera: 0 }),
    };
    // No true frames at all.
    let mut bad = common::ground_truth();
    bad.segmentation.clear();
    assert_eq!(bad.validate().unwrap_err(), refused);
    // True frames, but none for the shown frame's camera.
    let mut bad = common::ground_truth();
    bad.segmentation[0].camera = 1;
    assert_eq!(bad.validate().unwrap_err(), refused);
    // The shown frame is for the camera that has no true frame.
    let mut bad = common::ground_truth();
    bad.shown_segmentation[0].frame.camera = 2;
    assert_eq!(
        bad.validate().unwrap_err(),
        ContractError::Channel {
            channel: "shown_segmentation",
            cause: Box::new(ContractError::ShownWithoutTrueSegmentation { camera: 2 }),
        }
    );
    assert_eq!(
        bad.validate().unwrap_err().to_string(),
        "shown_segmentation: a shown segmentation frame for camera 2 has no true \
         segmentation frame for that camera in the same record"
    );
    // A later shown frame is checked too, not just the first.
    let mut bad = common::ground_truth();
    let mut second = common::shown_seg_frame();
    second.frame.camera = 5;
    bad.shown_segmentation.push(second);
    assert_eq!(
        bad.validate().unwrap_err().root(),
        &ContractError::ShownWithoutTrueSegmentation { camera: 5 }
    );
}

#[test]
fn a_shown_frame_needs_a_sight_perturbation_and_a_valid_frame() {
    for sense in Sense::ALL {
        if sense == Sense::Sight {
            continue;
        }
        for perturbation in [
            Perturbation::Offset {
                sense,
                seconds: 0.25,
            },
            Perturbation::Delay {
                sense,
                seconds: 0.1,
            },
            Perturbation::RenderFromPerturbedState {
                sense,
                state_delta_id: StateDeltaId("plate_colder_by_20k".to_string()),
            },
        ] {
            let mut bad = common::ground_truth();
            bad.shown_segmentation[0].perturbation = perturbation.clone();
            let err = bad.validate().unwrap_err();
            assert!(
                matches!(
                    err,
                    ContractError::Channel {
                        channel: "shown_segmentation",
                        ..
                    }
                ),
                "{perturbation:?}: {err:?}"
            );
            assert!(
                matches!(
                    err.root(),
                    ContractError::OutOfRange {
                        field: "perturbation.sense",
                        ..
                    }
                ),
                "{perturbation:?}: {err:?}"
            );
        }
    }
    // The perturbation is checked as a perturbation.
    let mut bad = common::ground_truth();
    bad.shown_segmentation[0].perturbation = Perturbation::Delay {
        sense: Sense::Sight,
        seconds: -0.1,
    };
    assert!(matches!(
        bad.validate().unwrap_err().root(),
        ContractError::OutOfRange {
            field: "delay.seconds",
            ..
        }
    ));
    // The frame is checked as a segmentation frame.
    let mut bad = common::ground_truth();
    bad.shown_segmentation[0].frame.frame.value = common::depth();
    let err = bad.validate().unwrap_err();
    assert!(
        matches!(
            err,
            ContractError::Channel {
                channel: "shown_segmentation",
                ..
            }
        ),
        "{err:?}"
    );
    assert!(matches!(err.root(), ContractError::Format { .. }));
    let mut bad = common::ground_truth();
    bad.shown_segmentation[0].frame.frame.captured_at_s = f64::INFINITY;
    assert_eq!(
        bad.validate().unwrap_err().root(),
        &ContractError::NonFinite {
            field: "captured_at_s"
        }
    );
}

#[test]
fn ground_truth_refuses_a_bad_segmentation_frame_and_names_the_channel() {
    // A frame of another semantic, another element type, another size.
    for wrong in [
        common::rgb(),
        common::depth(),
        FrameRef {
            width: 224,
            ..common::segmentation()
        },
        FrameRef {
            dtype: FrameDtype::U8,
            row_stride_bytes: 448,
            ..common::segmentation()
        },
    ] {
        let mut bad = common::ground_truth();
        bad.segmentation[0].frame.value = wrong;
        let err = bad.validate().unwrap_err();
        assert!(
            matches!(
                err,
                ContractError::Channel {
                    channel: "segmentation",
                    ..
                }
            ),
            "{err:?}"
        );
        assert!(
            matches!(err.root(), ContractError::Format { .. }),
            "{err:?}"
        );
        assert!(err.to_string().starts_with("segmentation: "), "{err}");
    }
    // A frame whose layout is broken fails the layout check first.
    let mut bad = common::ground_truth();
    bad.segmentation[0].frame.value.width = 0;
    assert_eq!(
        bad.validate().unwrap_err().root(),
        &ContractError::Empty { field: "width" }
    );
    // A capture time that is not a number.
    let mut bad = common::ground_truth();
    bad.segmentation[0].frame.captured_at_s = f64::NAN;
    assert_eq!(
        bad.validate().unwrap_err().root(),
        &ContractError::NonFinite {
            field: "captured_at_s"
        }
    );
    // The check reaches every frame, not just the first.
    let mut bad = common::ground_truth();
    bad.segmentation.push(SegFrame {
        camera: 1,
        frame: Capture::new(1.0, common::rgb()),
    });
    assert!(matches!(
        bad.validate(),
        Err(ContractError::Channel {
            channel: "segmentation",
            ..
        })
    ));
}

#[test]
fn a_seg_table_must_be_of_the_records_episode() {
    // The example: the table's episode is the record's.
    let truth = common::ground_truth();
    assert_eq!(
        truth.seg_table.as_ref().unwrap().episode_id,
        truth.identity.episode_id
    );
    truth.validate().unwrap();

    // The table of another episode is refused, naming the field and both ids.
    let mut bad = common::ground_truth();
    bad.seg_table.as_mut().unwrap().episode_id = 13;
    let err = bad.validate().unwrap_err();
    assert_eq!(
        err,
        ContractError::EpisodeMismatch {
            field: "seg_table.episode_id",
            expected: 12,
            found: 13
        }
    );
    assert_eq!(
        err.to_string(),
        "seg_table.episode_id is episode 13, but the record is of episode 12"
    );

    // The same from the other side: the record moves to another episode.
    let mut bad = common::ground_truth();
    bad.identity.episode_id = 13;
    assert_eq!(
        bad.validate().unwrap_err(),
        ContractError::EpisodeMismatch {
            field: "seg_table.episode_id",
            expected: 13,
            found: 12
        }
    );
    // Both moved together: valid again.
    bad.seg_table.as_mut().unwrap().episode_id = 13;
    bad.validate().unwrap();

    // A record that carries no table has nothing to match.
    let mut none = common::ground_truth();
    none.identity.episode_id = 99;
    none.seg_table = None;
    none.validate().unwrap();
}

#[test]
fn a_record_has_at_most_one_shown_frame_per_camera() {
    // Two cameras, each with a true frame and a shown frame: fine.
    let mut two = common::ground_truth();
    two.segmentation.push(SegFrame {
        camera: 1,
        frame: Capture::new(1.0, FrameRef::v0_segmentation().at(DeviceBufferId(10), 0)),
    });
    let mut second = common::shown_seg_frame();
    second.frame.camera = 1;
    two.shown_segmentation.push(second);
    two.validate().unwrap();

    // The same camera shown twice is refused, naming the channel and the camera,
    // even under another perturbation and at another time.
    let mut bad = common::ground_truth();
    let mut again = common::shown_seg_frame();
    again.perturbation = Perturbation::Delay {
        sense: Sense::Sight,
        seconds: 0.1,
    };
    again.frame.frame.captured_at_s = 0.9;
    bad.shown_segmentation.push(again);
    let err = bad.validate().unwrap_err();
    assert_eq!(
        err,
        ContractError::Channel {
            channel: "shown_segmentation",
            cause: Box::new(ContractError::DuplicateId {
                field: "camera",
                id: 0
            }),
        }
    );
    assert_eq!(
        err.to_string(),
        "shown_segmentation: camera lists id 0 twice"
    );

    // A duplicate that is not adjacent to the frame it repeats.
    let mut bad = two;
    let mut third = common::shown_seg_frame();
    third.frame.camera = 1;
    bad.shown_segmentation.push(common::shown_seg_frame());
    bad.shown_segmentation.push(third);
    assert_eq!(
        bad.validate().unwrap_err().root(),
        &ContractError::DuplicateId {
            field: "camera",
            id: 0
        }
    );
}

#[test]
fn occlusion_records_are_pairs_of_entity_and_visibility_in_json() {
    let value = serde_json::to_value(common::ground_truth()).unwrap();
    assert_eq!(
        value["object_ids_through_occlusion"],
        serde_json::json!([[2, false], [5, true]])
    );
}

// -- perturbations ----------------------------------------------------------

#[test]
fn a_perturbation_applies_to_one_sense() {
    assert_eq!(common::perturbation_offset().sense(), Sense::Sight);
    assert_eq!(common::perturbation_delay().sense(), Sense::Sound);
    assert_eq!(
        common::perturbation_render_from_perturbed_state().sense(),
        Sense::Touch
    );
    for sense in Sense::ALL {
        Perturbation::Offset {
            sense,
            seconds: 0.0,
        }
        .validate()
        .unwrap();
    }
}

#[test]
fn an_offset_is_signed_and_a_delay_is_not() {
    for seconds in [-0.25, 0.0, 0.25] {
        Perturbation::Offset {
            sense: Sense::Touch,
            seconds,
        }
        .validate()
        .unwrap();
    }
    Perturbation::Delay {
        sense: Sense::Touch,
        seconds: 0.0,
    }
    .validate()
    .unwrap();
    assert!(matches!(
        Perturbation::Delay {
            sense: Sense::Touch,
            seconds: -0.001
        }
        .validate(),
        Err(ContractError::OutOfRange {
            field: "delay.seconds",
            ..
        })
    ));
    for seconds in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            Perturbation::Offset {
                sense: Sense::Smell,
                seconds
            }
            .validate()
            .is_err()
        );
        assert!(
            Perturbation::Delay {
                sense: Sense::Smell,
                seconds
            }
            .validate()
            .is_err()
        );
    }
}

#[test]
fn a_state_delta_id_is_an_identifier() {
    let render = |id: &str| Perturbation::RenderFromPerturbedState {
        sense: Sense::Sight,
        state_delta_id: StateDeltaId(id.to_string()),
    };
    render("plate_colder_by_20k").validate().unwrap();
    for bad in ["", "../x", "Has Space", ".x"] {
        assert_eq!(
            render(bad).validate(),
            Err(ContractError::BadIdentifier {
                field: "state_delta_id"
            }),
            "{bad:?}"
        );
    }
}

#[test]
fn a_perturbation_of_an_unknown_kind_or_sense_is_refused() {
    assert!(Perturbation::from_json(r#"{"kind":"scramble","sense":"sight"}"#).is_err());
    assert!(
        Perturbation::from_json(r#"{"kind":"offset","sense":"hearing","seconds":0.1}"#).is_err()
    );
    assert!(Perturbation::from_json(r#"{"kind":"offset","sense":"sight"}"#).is_err());
    assert!(Perturbation::from_json(r#"{"kind":"delay","sense":"sight","seconds":-1.0}"#).is_err());
}

// -- rates ------------------------------------------------------------------

#[test]
fn the_v0_rates_are_the_ones_the_contract_fixes() {
    assert_eq!(
        RATES_V0,
        Rates {
            rgb_hz: 10,
            audio_hz: 16_000,
            touch_hz: 100,
            smell_hz: 10,
            taste_hz: 10
        }
    );
    assert_eq!(RATES_V0.hz(Sense::Sight, 50), 10);
    assert_eq!(RATES_V0.hz(Sense::Sound, 50), 16_000);
    assert_eq!(RATES_V0.hz(Sense::Touch, 50), 100);
    assert_eq!(RATES_V0.hz(Sense::Smell, 50), 10);
    assert_eq!(RATES_V0.hz(Sense::Taste, 50), 10);
    assert_eq!(RATES_V0.hz(Sense::Proprioception, 50), 50);
}

fn due_steps(sense: Sense, control_hz: u32, steps: u64) -> Vec<u64> {
    (0..steps)
        .filter(|&step| Rates::due(sense, step, control_hz))
        .collect()
}

#[test]
fn slow_senses_are_due_every_few_steps() {
    // 50 Hz control, 10 Hz senses: every fifth step.
    for sense in [Sense::Sight, Sense::Smell, Sense::Taste] {
        assert_eq!(due_steps(sense, 50, 21), [0, 5, 10, 15, 20], "{sense:?}");
    }
    // 10 Hz control: every step. 100 Hz control: every tenth.
    assert_eq!(due_steps(Sense::Sight, 10, 4), [0, 1, 2, 3]);
    assert_eq!(due_steps(Sense::Sight, 100, 31), [0, 10, 20, 30]);
}

#[test]
fn a_rate_that_does_not_divide_the_control_rate_is_still_exact() {
    // 25 Hz control, 10 Hz sense: samples at 0, 0.1, 0.2, 0.3, 0.4 s are
    // delivered at the first step at or after them (0.04 s apart).
    assert_eq!(due_steps(Sense::Sight, 25, 11), [0, 3, 5, 8, 10]);
    // 30 Hz control: every third step.
    assert_eq!(due_steps(Sense::Sight, 30, 10), [0, 3, 6, 9]);
}

#[test]
fn senses_at_or_above_the_control_rate_are_due_every_step() {
    for control_hz in [10, 20, 50, 100] {
        assert_eq!(
            due_steps(Sense::Touch, control_hz, 20),
            (0..20).collect::<Vec<_>>(),
            "touch at {control_hz} Hz control"
        );
    }
    // Touch at 100 Hz under a 200 Hz control loop: every other step.
    assert_eq!(due_steps(Sense::Touch, 200, 7), [0, 2, 4, 6]);
}

#[test]
fn audio_and_proprioception_are_due_at_every_step() {
    for control_hz in [1, 7, 50, 1000] {
        for step in [0, 1, 2, 99, u64::MAX] {
            assert!(Rates::due(Sense::Sound, step, control_hz));
            assert!(Rates::due(Sense::Proprioception, step, control_hz));
        }
    }
}

#[test]
fn over_a_long_run_each_sense_is_due_at_its_rate() {
    for control_hz in [10u32, 25, 30, 50, 60, 100, 144] {
        for sense in [Sense::Sight, Sense::Touch, Sense::Smell, Sense::Taste] {
            let seconds = 20u64;
            let steps = u64::from(control_hz) * seconds;
            // Steps 0..=steps cover `seconds` seconds, including time 0.
            let due = (0..=steps)
                .filter(|&step| Rates::due(sense, step, control_hz))
                .count() as u64;
            let rate = u64::from(RATES_V0.hz(sense, control_hz));
            let expected = if rate >= u64::from(control_hz) {
                steps + 1
            } else {
                rate * seconds + 1
            };
            assert_eq!(due, expected, "{sense:?} at {control_hz} Hz control");
        }
    }
}

#[test]
fn due_is_a_pure_function_of_its_arguments_and_cannot_overflow() {
    for _ in 0..3 {
        assert_eq!(
            due_steps(Sense::Sight, 25, 200),
            due_steps(Sense::Sight, 25, 200)
        );
    }
    // The largest step and rate multiply past u64; the rule uses 128 bits.
    let big = u64::MAX - 5;
    let _ = Rates::due(Sense::Sight, big, 1);
    let _ = Rates::due(Sense::Touch, big, 1);
    // With a 1 Hz control loop a 10 Hz sense is due at every step.
    assert!(Rates::due(Sense::Sight, big, 1));
}

#[test]
fn a_stopped_clock_samples_nothing_and_custom_rates_work() {
    for sense in Sense::ALL {
        assert!(!Rates::due(sense, 0, 0), "{sense:?}");
    }
    let slow = Rates {
        rgb_hz: 2,
        audio_hz: 16_000,
        touch_hz: 0,
        smell_hz: 1,
        taste_hz: 1,
    };
    let steps: Vec<u64> = (0..21)
        .filter(|&s| slow.is_due(Sense::Sight, s, 10))
        .collect();
    assert_eq!(steps, [0, 5, 10, 15, 20]);
    assert!(
        !slow.is_due(Sense::Touch, 0, 10),
        "a rate of 0 is never sampled"
    );
    assert_eq!(
        Rates::due(Sense::Sight, 5, 50),
        RATES_V0.is_due(Sense::Sight, 5, 50)
    );
}
