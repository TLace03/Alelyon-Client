//! Ground truth and perturbations are evaluation only: no field of a [`Bundle`]
//! can hold them, directly or through any type a bundle holds.
//!
//! WHY. A model shown the temperature at a point, a reaction's progress, a
//! contact event or the identity of an occluded object has been handed the
//! answer to what the senses are meant to let it infer. Training on it, or
//! letting it into a model's context at evaluation time, would make every
//! cross-sense result meaningless. See the note in `ground_truth.rs`.
//!
//! Segmentation is on the evaluation-only side too: its per-instance ids persist
//! through occlusion, so a model that sees them has been handed the object
//! identities the object-permanence test measures. It was a channel of the
//! bundle once; the contract owner moved it to `GroundTruth`, and
//! `a_bundle_cannot_carry_a_segmentation_frame` below is the negative control.
//! The same goes for what makes sense of the ids (the per-episode `SegTable`) and
//! for the segmentation shown under a sight perturbation (`ShownSegFrame`), both
//! decided by the contract owner.
//!
//! HOW, at compile time. The first test below does not so much run as compile:
//!
//! 1. It destructures [`Bundle`] and every type a bundle is made of with no
//!    `..` wildcard, and pins the exact type of every field. Adding a field
//!    (of any type), removing one or changing one's type stops this file from
//!    compiling.
//! 2. It passes each channel to a function that only accepts types this file
//!    lists as observations. [`GroundTruth`] and [`Perturbation`] are not
//!    listed, and the assertions at the bottom stop compiling if someone lists
//!    them.
//!
//! So the way to put evaluation data into a bundle is to edit this file, and
//! the person editing it has just read this note. The runtime tests then check
//! the same thing in the JSON a Python training loop actually reads.

mod common;

use std::collections::BTreeSet;

use sim_contract::{
    AudioChunk, AudioSamples, BasePose, Bundle, Capture, DeviceBufferId, FrameDtype, FrameRef,
    FrameSemantic, GroundTruth, Identity, Perturbation, Proprio, SegFrame, SegTable, ShownSegFrame,
    SpeciesVector, TouchMap, TouchMaps,
};

/// The types an observation is made of. A bundle's channels must be these.
trait Observation {}
impl Observation for Identity {}
impl Observation for FrameRef {}
impl Observation for AudioChunk {}
impl Observation for Proprio {}
impl Observation for TouchMaps {}
impl Observation for SpeciesVector {}
impl<T: Observation> Observation for Capture<T> {}
impl<T: Observation> Observation for Option<T> {}

fn observation<T: Observation>(_: &T) {}

/// Compiles only if `T` does not implement [`Observation`]: with the impl, the
/// two blanket impls below overlap and `some_item` is ambiguous.
trait NotAnObservation<A> {
    fn some_item() {}
}
impl<T: ?Sized> NotAnObservation<()> for T {}
impl<T: ?Sized + Observation> NotAnObservation<u8> for T {}

#[test]
fn a_bundle_is_made_of_observations_and_nothing_else() {
    // Compile-time: neither evaluation-only type is an observation.
    let _ = <GroundTruth as NotAnObservation<_>>::some_item;
    let _ = <Perturbation as NotAnObservation<_>>::some_item;
    let _ = <SegFrame as NotAnObservation<_>>::some_item;
    let _ = <SegTable as NotAnObservation<_>>::some_item;
    let _ = <ShownSegFrame as NotAnObservation<_>>::some_item;

    let bundle = common::bundle_full();

    // Every field of Bundle, exhaustively, each with its exact type.
    let Bundle {
        identity,
        rgb,
        depth,
        audio,
        proprioception,
        touch,
        smell,
        taste,
    } = &bundle;
    let _: &Identity = identity;
    let _: &Option<Capture<FrameRef>> = rgb;
    let _: &Option<Capture<FrameRef>> = depth;
    let _: &Capture<AudioChunk> = audio;
    let _: &Capture<Proprio> = proprioception;
    let _: &Option<Capture<TouchMaps>> = touch;
    let _: &Option<Capture<SpeciesVector>> = smell;
    let _: &Option<Capture<SpeciesVector>> = taste;
    observation(identity);
    observation(rgb);
    observation(depth);
    observation(audio);
    observation(proprioception);
    observation(touch);
    observation(smell);
    observation(taste);

    // The types inside, exhaustively.
    let Identity {
        schema_version,
        env_id,
        episode_id,
        step,
        sim_time_s,
        seed,
    } = *identity;
    let _: (u32, u32, u64, u64, f64, u64) =
        (schema_version, env_id, episode_id, step, sim_time_s, seed);

    let Capture {
        captured_at_s,
        value,
    } = rgb.as_ref().unwrap();
    let _: (&f64, &FrameRef) = (captured_at_s, value);

    let FrameRef {
        buffer,
        byte_offset,
        width,
        height,
        channels,
        dtype,
        row_stride_bytes,
        semantic,
    } = rgb.as_ref().unwrap().value;
    let _: (
        DeviceBufferId,
        u64,
        u32,
        u32,
        u8,
        FrameDtype,
        u32,
        FrameSemantic,
    ) = (
        buffer,
        byte_offset,
        width,
        height,
        channels,
        dtype,
        row_stride_bytes,
        semantic,
    );

    let AudioChunk {
        sample_rate_hz,
        first_sample_index,
        samples,
    } = &audio.value;
    let _: (&u32, &u64) = (sample_rate_hz, first_sample_index);
    match samples {
        AudioSamples::Host { values } => {
            let _: &Vec<f32> = values;
        }
        AudioSamples::Device {
            buffer,
            byte_offset,
            count,
        } => {
            let _: (&DeviceBufferId, &u64, &u64) = (buffer, byte_offset, count);
        }
    }

    let Proprio {
        joint_positions,
        joint_velocities,
        base_pose,
    } = &proprioception.value;
    let _: (&Vec<f32>, &Vec<f32>, &Option<BasePose>) =
        (joint_positions, joint_velocities, base_pose);
    let BasePose {
        position,
        orientation_quat,
    } = base_pose.as_ref().unwrap();
    let _: (&[f32; 3], &[f32; 4]) = (position, orientation_quat);

    let TouchMaps { sensors } = &touch.as_ref().unwrap().value;
    let _: &Vec<TouchMap> = sensors;
    let TouchMap {
        sensor_id,
        rows,
        cols,
        pressure_pa,
        shear_pa,
        temperature_k,
    } = &sensors[0];
    let _: (&u32, &u32, &u32) = (sensor_id, rows, cols);
    let _: (&Vec<f32>, &Vec<[f32; 2]>, &Vec<f32>) = (pressure_pa, shear_pa, temperature_k);

    let SpeciesVector {
        table_version,
        concentrations,
    } = &smell.as_ref().unwrap().value;
    let _: (&u32, &Vec<f32>) = (table_version, concentrations);
}

/// Every object key at any depth of a JSON value.
fn keys(value: &serde_json::Value, into: &mut BTreeSet<String>) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, inner) in map {
                into.insert(key.clone());
                keys(inner, into);
            }
        }
        serde_json::Value::Array(items) => items.iter().for_each(|item| keys(item, into)),
        _ => {}
    }
}

#[test]
fn no_ground_truth_key_appears_anywhere_in_a_bundles_json() {
    // The keys that belong to ground truth alone. Keys both types use for other
    // reasons (`identity`, `position`, `value`, `name`) say nothing either way.
    let truth_only = [
        "temperatures_k",
        "reaction_progress",
        "contacts",
        "object_ids_through_occlusion",
        "normal_force_n",
        "segmentation",
        "seg_table",
        "shown_segmentation",
    ];
    let mut truth_keys = BTreeSet::new();
    keys(
        &serde_json::to_value(common::ground_truth()).unwrap(),
        &mut truth_keys,
    );
    for key in truth_only {
        assert!(
            truth_keys.contains(key),
            "the sweep is stale: ground truth has no key {key}"
        );
    }
    for bundle in [common::bundle_full(), common::bundle_minimal()] {
        let mut found = BTreeSet::new();
        keys(&serde_json::to_value(&bundle).unwrap(), &mut found);
        let leaked: Vec<_> = truth_only
            .iter()
            .filter(|key| found.contains(**key))
            .collect();
        assert!(
            leaked.is_empty(),
            "ground truth reached a bundle: {leaked:?}"
        );
    }
}

/// Every string value at any depth of a JSON value.
fn strings<'a>(value: &'a serde_json::Value, into: &mut BTreeSet<&'a str>) {
    match value {
        serde_json::Value::String(text) => {
            into.insert(text);
        }
        serde_json::Value::Object(map) => map.values().for_each(|inner| strings(inner, into)),
        serde_json::Value::Array(items) => items.iter().for_each(|item| strings(item, into)),
        _ => {}
    }
}

/// Negative control for the contract's decision: the
/// segmentation channel is gone from the bundle and lives in ground truth.
#[test]
fn a_bundle_cannot_carry_a_segmentation_frame() {
    // Compile-time. `SegFrame`, the record that carries a segmentation frame,
    // and the types that go with it (the per-episode id table, and the shown
    // frame with its perturbation) are evaluation only: none is an observation.
    let _ = <SegFrame as NotAnObservation<_>>::some_item;
    let _ = <SegTable as NotAnObservation<_>>::some_item;
    let _ = <ShownSegFrame as NotAnObservation<_>>::some_item;

    // Compile-time. Bundle destructured with no `..`: every field it has is
    // named here, so a `segmentation` field (or any other) added back stops this
    // file compiling. The field types are pinned too, so none of them can be a
    // list or record that carries segmentation frames.
    let Bundle {
        identity,
        rgb,
        depth,
        audio,
        proprioception,
        touch,
        smell,
        taste,
    } = common::bundle_full();
    let _: Identity = identity;
    let _: Option<Capture<FrameRef>> = rgb;
    let _: Option<Capture<FrameRef>> = depth;
    let _: Capture<AudioChunk> = audio;
    let _: Capture<Proprio> = proprioception;
    let _: Option<Capture<TouchMaps>> = touch;
    let _: Option<Capture<SpeciesVector>> = smell;
    let _: Option<Capture<SpeciesVector>> = taste;

    // Run time. Nothing in a bundle's JSON is named or valued `segmentation`: no
    // key, and no frame whose semantic says so.
    for bundle in [common::bundle_full(), common::bundle_minimal()] {
        let value = serde_json::to_value(&bundle).unwrap();
        let mut found = BTreeSet::new();
        keys(&value, &mut found);
        assert!(!found.contains("segmentation"), "a segmentation key");
        let mut texts = BTreeSet::new();
        strings(&value, &mut texts);
        assert!(!texts.contains("segmentation"), "a segmentation frame");
    }

    // Ground truth is where it went.
    let truth = common::ground_truth();
    let _: &Vec<SegFrame> = &truth.segmentation;
    let _: &Option<SegTable> = &truth.seg_table;
    let _: &Vec<ShownSegFrame> = &truth.shown_segmentation;
    let value = serde_json::to_value(&truth).unwrap();
    let mut texts = BTreeSet::new();
    strings(&value, &mut texts);
    assert!(texts.contains("segmentation"));
}

#[test]
fn a_bundle_and_ground_truth_cannot_be_read_as_each_other() {
    let bundle = serde_json::to_string(&common::bundle_full()).unwrap();
    let truth = serde_json::to_string(&common::ground_truth()).unwrap();
    assert!(serde_json::from_str::<GroundTruth>(&bundle).is_err());
    assert!(serde_json::from_str::<Bundle>(&truth).is_err());
}

#[test]
fn a_perturbation_is_not_part_of_a_bundle_either() {
    let mut found = BTreeSet::new();
    keys(
        &serde_json::to_value(common::bundle_full()).unwrap(),
        &mut found,
    );
    for key in ["seconds", "state_delta_id", "sense"] {
        assert!(!found.contains(key), "{key} appears in a bundle");
    }
}
