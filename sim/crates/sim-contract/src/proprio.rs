//! Proprioception: the body's own joints and base.
//!
//! Invariants:
//! - Joint positions and velocities are parallel lists: entry `i` of each
//!   describes joint `i`, in the articulation's joint order, and the two lists
//!   have the same length.
//! - Units are SI: radians and radians per second for revolute joints, metres
//!   and metres per second for prismatic ones.
//! - Poses are in the world frame: right-handed, metres, +Z up. A quaternion is
//!   `[x, y, z, w]` (scalar last) and has length 1.
//! - An empty `Proprio` (no joints, no base pose) is valid: the channel exists
//!   before a body does.

use serde::{Deserialize, Serialize};

use crate::ContractError;
use crate::check::{all_finite_f32, unit_quaternion};

/// A body's base pose in the world frame.
///
/// Invariants: finite position; `orientation_quat` is `[x, y, z, w]` with
/// length 1 (within 1e-3).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BasePose {
    /// Metres, world frame.
    pub position: [f32; 3],
    /// Unit quaternion `[x, y, z, w]`, body to world.
    pub orientation_quat: [f32; 4],
}

impl BasePose {
    /// Checks the position and the quaternion.
    pub fn validate(&self) -> Result<(), ContractError> {
        all_finite_f32("base_pose.position", &self.position)?;
        unit_quaternion("base_pose.orientation_quat", &self.orientation_quat)
    }
}

/// What the body knows about itself this tick.
///
/// Invariants (checked by [`Proprio::validate`]): the two joint lists have the
/// same length; every number is finite; a base pose, if present, is valid.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Proprio {
    /// Joint positions: radians (revolute) or metres (prismatic).
    pub joint_positions: Vec<f32>,
    /// Joint velocities: rad/s (revolute) or m/s (prismatic).
    pub joint_velocities: Vec<f32>,
    /// The base pose, for a body whose base is free to move; `None` for a body
    /// fixed to the world.
    pub base_pose: Option<BasePose>,
}

impl Proprio {
    /// No joints and no base pose.
    pub fn empty() -> Self {
        Self {
            joint_positions: Vec::new(),
            joint_velocities: Vec::new(),
            base_pose: None,
        }
    }

    /// Checks the lists' lengths, every number and the base pose.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.joint_positions.len() != self.joint_velocities.len() {
            return Err(ContractError::LengthMismatch {
                field: "joint_velocities",
                expected: self.joint_positions.len() as u64,
                found: self.joint_velocities.len() as u64,
            });
        }
        all_finite_f32("joint_positions", &self.joint_positions)?;
        all_finite_f32("joint_velocities", &self.joint_velocities)?;
        if let Some(pose) = &self.base_pose {
            pose.validate()?;
        }
        Ok(())
    }
}
