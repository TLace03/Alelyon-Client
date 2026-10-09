//! What the simulator takes back: actions and scene resets.
//!
//! These are the contract's inputs to the simulator. They come from a training
//! loop (Python, hand-written or generated), so they are read strictly: an
//! unknown field is refused rather than ignored, because a misspelt field would
//! otherwise silently do nothing. Every type here (and the types nested in
//! them) rejects unknown fields, and `from_json` reads and validates in one step.
//!
//! Invariants:
//! - Every number is finite. Joint targets are radians (revolute) or metres
//!   (prismatic), in the articulation's joint order.
//! - A camera `Pose` is in the world frame: right-handed, metres, +Z up; its
//!   quaternion is `[x, y, z, w]` with length 1. A `LookAt` aims the camera at
//!   `target` (a world point, metres) with `up` as the world direction that
//!   should appear up; `up` is not the zero vector. The camera's own axes are
//!   the OpenCV camera frame (x right, y down, z forward along the view axis;
//!   see [`crate::FrameRef`]): `orientation_quat` is the rotation from that
//!   frame to the world, and `LookAt` points the +z axis at `target`.
//! - A sniff's intensity is in `0..=1`; its duration is positive seconds. `None`
//!   means no sniff this tick.
//! - A scene id is an identifier: it can reach a path on disk, so it is
//!   1 to 128 characters from `[a-z0-9_.-]`, starting with a letter or digit.
//! - Omitting an `Option` field (`camera`, `sniff`) reads as `None`. Nothing else
//!   may be omitted.

use serde::{Deserialize, Serialize};

use crate::ContractError;
use crate::check::{all_finite_f32, finite_f32, identifier, parse, unit_quaternion};

/// A camera command.
///
/// JSON: an object with a `kind` of `"pose"` (with `position`,
/// `orientation_quat`) or `"look_at"` (with `target`, `up`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CameraAction {
    /// Put the camera at a pose.
    Pose {
        /// Metres, world frame.
        position: [f32; 3],
        /// Unit quaternion `[x, y, z, w]`, camera to world.
        orientation_quat: [f32; 4],
    },
    /// Aim the camera at a point.
    LookAt {
        /// The world point to look at, metres.
        target: [f32; 3],
        /// The world direction that should appear up; not zero.
        up: [f32; 3],
    },
}

impl CameraAction {
    /// Checks finiteness, the quaternion's length and that `up` is not zero.
    pub fn validate(&self) -> Result<(), ContractError> {
        match self {
            CameraAction::Pose {
                position,
                orientation_quat,
            } => {
                all_finite_f32("camera.position", position)?;
                unit_quaternion("camera.orientation_quat", orientation_quat)
            }
            CameraAction::LookAt { target, up } => {
                all_finite_f32("camera.target", target)?;
                all_finite_f32("camera.up", up)?;
                if up.iter().all(|&c| c == 0.0) {
                    return Err(ContractError::OutOfRange {
                        field: "camera.up",
                        reason: "the up direction cannot be the zero vector",
                    });
                }
                Ok(())
            }
        }
    }
}

/// A sniff: the sensor action that drives the nose's sniff dynamics.
///
/// Invariants: `intensity` is finite and in `0..=1`; `duration_s` is finite and
/// above 0.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sniff {
    /// How hard to sniff: 0 is resting airflow, 1 is the strongest sniff.
    pub intensity: f32,
    /// How long the sniff lasts, seconds.
    pub duration_s: f32,
}

impl Sniff {
    /// Checks the intensity and the duration.
    pub fn validate(&self) -> Result<(), ContractError> {
        finite_f32("sniff.intensity", self.intensity)?;
        finite_f32("sniff.duration_s", self.duration_s)?;
        if !(0.0..=1.0).contains(&self.intensity) {
            return Err(ContractError::OutOfRange {
                field: "sniff.intensity",
                reason: "intensity is in 0..=1",
            });
        }
        if self.duration_s <= 0.0 {
            return Err(ContractError::OutOfRange {
                field: "sniff.duration_s",
                reason: "a sniff lasts a positive time",
            });
        }
        Ok(())
    }
}

/// The actions on the senses themselves (active perception): where the camera
/// is or looks, and whether to sniff.
///
/// Invariants: each present action is valid. Both absent is valid.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SensorActions {
    /// A camera command, or `None` to leave the camera as it is.
    #[serde(default)]
    pub camera: Option<CameraAction>,
    /// A sniff, or `None` for no sniff this tick.
    #[serde(default)]
    pub sniff: Option<Sniff>,
}

impl SensorActions {
    /// Checks each present action.
    pub fn validate(&self) -> Result<(), ContractError> {
        if let Some(camera) = &self.camera {
            camera.validate()?;
        }
        if let Some(sniff) = &self.sniff {
            sniff.validate()?;
        }
        Ok(())
    }
}

/// What Sinai sends back for one tick.
///
/// Invariants (checked by [`Action::validate`]): every joint target is finite;
/// the sensor actions are valid. `identity_step` is the `step` of the bundle
/// the action answers; the simulator applies it to the tick that follows.
/// Unknown fields are refused on read.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Action {
    /// The `Identity::step` of the bundle this action answers.
    pub identity_step: u64,
    /// Targets for the body's joints: radians (revolute) or metres (prismatic).
    pub joint_targets: Vec<f32>,
    /// The sensors' own actions.
    pub sensor: SensorActions,
}

impl Action {
    /// Checks the joint targets and the sensor actions.
    pub fn validate(&self) -> Result<(), ContractError> {
        all_finite_f32("joint_targets", &self.joint_targets)?;
        self.sensor.validate()
    }

    /// Reads an action from JSON, refusing unknown fields, and validates it.
    pub fn from_json(text: &str) -> Result<Self, ContractError> {
        let action: Action = parse(text)?;
        action.validate()?;
        Ok(action)
    }
}

/// The identity of a scene the simulator can load.
///
/// Invariants (checked by [`SceneId::validate`]): 1 to 128 characters from
/// `[a-z0-9_.-]`, starting with a letter or digit, so it cannot name a path
/// outside the scene directory.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SceneId(pub String);

impl SceneId {
    /// The id as text.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Checks the identifier's shape.
    pub fn validate(&self) -> Result<(), ContractError> {
        identifier("scene", &self.0)
    }
}

/// Start a new episode in an environment slot.
///
/// Invariants (checked by [`Reset::validate`]): the scene id is valid. The
/// simulator starts a new episode of `env_id` from `scene` seeded with `seed`;
/// the same scene and seed on the same device and driver give the same episode
/// bit for bit. Unknown fields are refused on read.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reset {
    /// The environment slot to reset.
    pub env_id: u32,
    /// The new episode's seed.
    pub seed: u64,
    /// The scene to load.
    pub scene: SceneId,
}

impl Reset {
    /// Checks the scene id.
    pub fn validate(&self) -> Result<(), ContractError> {
        self.scene.validate()
    }

    /// Reads a reset from JSON, refusing unknown fields, and validates it.
    pub fn from_json(text: &str) -> Result<Self, ContractError> {
        let reset: Reset = parse(text)?;
        reset.validate()?;
        Ok(reset)
    }
}
