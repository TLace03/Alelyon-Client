//! Canonical example values for every type of the contract.
//!
//! These are the values the committed JSON fixtures (`fixtures/v0/`) hold, and
//! the values the other tests build on. Every one of them is valid. They are
//! examples of the schema's shape, not simulator output: the buffer ids are
//! placeholders; the species are examples whose PubChem CIDs were checked
//! against PubChem on 2026-10-01 (see the crate README).
//!
//! The example tick is step 50 of episode 12 of environment 3, under a control
//! rate of 50 Hz (so simulated time is 1.0 s), a step at which every sense of
//! the v0 rates is due.

// Each test binary uses a different part of this module.
#![allow(dead_code)]

use std::f32::consts::FRAC_1_SQRT_2;

use sim_contract::{
    Action, AudioChunk, AudioSamples, BasePose, Bundle, CameraAction, Capture, ContactEvent,
    DeviceBufferId, EntityId, FrameRef, GroundTruth, Identity, NamedValue, Perturbation, Proprio,
    Reset, SceneId, SegFrame, SegTable, Sense, SensorActions, ShownSegFrame, Sniff, Species,
    SpeciesTable, SpeciesVector, StateDeltaId, TouchMap, TouchMaps,
};

/// The control rate the example tick runs at.
pub const CONTROL_HZ: u32 = 50;

pub fn identity() -> Identity {
    Identity::new(3, 12, 50, 1.0, 20_260_930)
}

pub fn rgb() -> FrameRef {
    FrameRef::v0_rgb().at(DeviceBufferId(7), 0)
}

pub fn depth() -> FrameRef {
    FrameRef::v0_depth().at(DeviceBufferId(8), 0)
}

/// A segmentation frame. Evaluation only: it appears in [`GroundTruth`], never in
/// a bundle.
pub fn segmentation() -> FrameRef {
    FrameRef::v0_segmentation().at(DeviceBufferId(9), 256)
}

/// Camera 0's segmentation frame for the example tick.
pub fn seg_frame() -> SegFrame {
    SegFrame {
        camera: 0,
        frame: Capture::new(1.0, segmentation()),
    }
}

pub fn audio_host() -> AudioChunk {
    AudioChunk::host(
        16_000,
        vec![0.0, 0.125, 0.25, 0.125, 0.0, -0.125, -0.25, -0.125],
    )
}

/// 320 samples is one tick of 16 kHz audio at a 50 Hz control rate.
pub fn audio_device() -> AudioChunk {
    AudioChunk {
        sample_rate_hz: 16_000,
        first_sample_index: 16_000,
        samples: AudioSamples::Device {
            buffer: DeviceBufferId(11),
            byte_offset: 0,
            count: 320,
        },
    }
}

pub fn proprio() -> Proprio {
    Proprio {
        joint_positions: vec![0.0, -0.5, 1.25],
        joint_velocities: vec![0.0, 0.125, -0.25],
        base_pose: Some(BasePose {
            position: [0.0, 0.0, 0.75],
            // A quarter turn about +Z, scalar last.
            orientation_quat: [0.0, 0.0, FRAC_1_SQRT_2, FRAC_1_SQRT_2],
        }),
    }
}

pub fn capture_proprio() -> Capture<Proprio> {
    Capture::new(1.0, proprio())
}

pub fn touch_maps() -> TouchMaps {
    TouchMaps {
        sensors: vec![TouchMap {
            sensor_id: 4,
            rows: 2,
            cols: 2,
            pressure_pa: vec![0.0, 120.5, 0.0, 0.0],
            shear_pa: vec![[0.0, 0.0], [1.5, -0.5], [0.0, 0.0], [0.0, 0.0]],
            temperature_k: vec![305.15, 310.5, 305.15, 305.15],
        }],
    }
}

fn species(pubchem_cid: u64, name: &str) -> Species {
    Species {
        pubchem_cid,
        name: name.to_string(),
    }
}

/// Example odorants: ethanol (CID 702), acetic acid (176), hexanal (6184).
pub fn smell_table() -> SpeciesTable {
    SpeciesTable {
        version: 1,
        species: vec![
            species(702, "ethanol"),
            species(176, "acetic acid"),
            species(6184, "hexanal"),
        ],
    }
}

/// Example tastants: sodium chloride (CID 5234), sucrose (5988), L-glutamic acid (33032).
pub fn taste_table() -> SpeciesTable {
    SpeciesTable {
        version: 1,
        species: vec![
            species(5234, "sodium chloride"),
            species(5988, "sucrose"),
            species(33_032, "L-glutamic acid"),
        ],
    }
}

/// mol/m^3 at the nose.
pub fn smell_vector() -> SpeciesVector {
    SpeciesVector {
        table_version: 1,
        concentrations: vec![0.0, 0.000_25, 0.000_1],
    }
}

/// mol/L at the tongue contact.
pub fn taste_vector() -> SpeciesVector {
    SpeciesVector {
        table_version: 1,
        concentrations: vec![0.05, 0.0, 0.002],
    }
}

/// The episode's segmentation table, as written at reset. The ids are assigned,
/// not derived from the entities: they are neither equal to the entity ids nor
/// in their order.
pub fn seg_table() -> SegTable {
    SegTable {
        episode_id: 12,
        entries: vec![(41, EntityId(1)), (7, EntityId(2)), (23, EntityId(5))],
    }
}

/// Camera 0's segmentation of the frame shown under the example sight
/// perturbation (a 0.05 s earlier capture), beside the true one.
pub fn shown_seg_frame() -> ShownSegFrame {
    ShownSegFrame {
        perturbation: perturbation_offset(),
        frame: SegFrame {
            camera: 0,
            frame: Capture::new(
                0.95,
                FrameRef::v0_segmentation().at(DeviceBufferId(10), 256),
            ),
        },
    }
}

/// Every channel present and sampled. (Segmentation is not a channel: it is
/// evaluation only and lives in [`ground_truth`].)
pub fn bundle_full() -> Bundle {
    Bundle {
        identity: identity(),
        rgb: Some(Capture::new(1.0, rgb())),
        depth: Some(Capture::new(1.0, depth())),
        audio: Capture::new(1.0, audio_device()),
        proprioception: capture_proprio(),
        touch: Some(Capture::new(1.0, touch_maps())),
        smell: Some(Capture::new(1.0, smell_vector())),
        taste: Some(Capture::new(1.0, taste_vector())),
    }
}

/// The first tick of an episode with every channel present and none with
/// content: the Phase 1 "schema whole before content arrives" bundle.
pub fn bundle_minimal() -> Bundle {
    Bundle::empty(Identity::new(3, 12, 0, 0.0, 20_260_930), 0)
}

pub fn action_pose() -> Action {
    Action {
        identity_step: 50,
        joint_targets: vec![0.0, -0.25, 1.5],
        sensor: SensorActions {
            camera: Some(CameraAction::Pose {
                position: [0.5, -0.25, 1.0],
                orientation_quat: [0.0, 0.0, FRAC_1_SQRT_2, FRAC_1_SQRT_2],
            }),
            sniff: Some(Sniff {
                intensity: 0.75,
                duration_s: 0.5,
            }),
        },
    }
}

pub fn action_look_at() -> Action {
    Action {
        identity_step: 51,
        joint_targets: vec![0.0, -0.25, 1.5],
        sensor: SensorActions {
            camera: Some(CameraAction::LookAt {
                target: [0.0, 0.0, 0.5],
                up: [0.0, 0.0, 1.0],
            }),
            sniff: None,
        },
    }
}

pub fn reset() -> Reset {
    Reset {
        env_id: 3,
        seed: 20_260_931,
        scene: SceneId("manipulation_table_v0".to_string()),
    }
}

fn named(name: &str, value: f64) -> NamedValue {
    NamedValue {
        name: name.to_string(),
        value,
    }
}

pub fn ground_truth() -> GroundTruth {
    GroundTruth {
        identity: identity(),
        temperatures_k: vec![named("plate_surface", 373.15), named("object_core", 301.5)],
        reaction_progress: vec![named("maillard", 0.125), named("water_loss", 0.0625)],
        contacts: vec![ContactEvent {
            a: EntityId(1),
            b: EntityId(2),
            normal_force_n: 4.5,
            position: [0.25, 0.0, 0.5],
        }],
        object_ids_through_occlusion: vec![(EntityId(2), false), (EntityId(5), true)],
        segmentation: vec![seg_frame()],
        seg_table: Some(seg_table()),
        shown_segmentation: vec![shown_seg_frame()],
    }
}

pub fn perturbation_offset() -> Perturbation {
    Perturbation::Offset {
        sense: Sense::Sight,
        seconds: -0.05,
    }
}

pub fn perturbation_delay() -> Perturbation {
    Perturbation::Delay {
        sense: Sense::Sound,
        seconds: 0.1,
    }
}

pub fn perturbation_render_from_perturbed_state() -> Perturbation {
    Perturbation::RenderFromPerturbedState {
        sense: Sense::Touch,
        state_delta_id: StateDeltaId("plate_colder_by_20k".to_string()),
    }
}
