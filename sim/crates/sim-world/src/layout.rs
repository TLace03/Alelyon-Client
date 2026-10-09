//! The structure-of-arrays layout of the world state, per environment slot.
//!
//! The world state of ADR-0042 lives in device memory as one array per field,
//! with the environment slot as the leading index, so thousands of environments
//! step together. This module computes where each field's data is for a given
//! [`Scene`] and a number of environments. It is the contract between the host
//! mirror ([`crate::HostWorld`]) and the GPU kernels of later phases: a kernel
//! reads `field.env_stride_bytes` and `field.base_offset_bytes`, not a constant.
//!
//! Fields (all `f32`, little-endian):
//!
//! | Field | Floats per environment |
//! |---|---|
//! | `BodyPos` | `3 * n_bodies`: world position of each body, metres |
//! | `BodyQuat` | `4 * n_bodies`: world orientation, unit `[x, y, z, w]` |
//! | `BodyLinVel` | `3 * n_bodies`: world linear velocity, m/s |
//! | `BodyAngVel` | `3 * n_bodies`: world angular velocity, rad/s |
//! | `Qpos` | `nq`: joint positions (free 7: position, quaternion `[x, y, z, w]`; ball 4: quaternion; hinge and slide 1) |
//! | `Qvel` | `nv`: joint velocities (free 6, ball 3, hinge and slide 1) |
//! | `Ctrl` | `n_actuators`: control inputs |
//! | `Surface` | `4 * n_instances`: per instance temperature K, browning 0..1, wetness 0..1, reserved; all zero until phase 2 |
//!
//! Invariants:
//! - Joint coordinates follow MuJoCo's `nq`/`nv` conventions (a free joint is 7
//!   and 6, a ball joint 4 and 3, a hinge or slide 1 and 1), in the scene's joint
//!   order. The quaternion in `Qpos` is `[x, y, z, w]` like every quaternion in
//!   this crate, NOT MuJoCo's `[w, x, y, z]`.
//! - Each field is its own array: environment `e`'s data starts at
//!   `base_offset_bytes + e * env_stride_bytes`, and the per-environment stride is
//!   the data size rounded up to a multiple of [`ENV_STRIDE_ALIGN_BYTES`] (256), so
//!   every environment's slice of every field starts on a 256-byte boundary. The
//!   padding is zero.
//! - A field with no data (a scene without actuators has no `Ctrl`) has stride 0
//!   and size 0.
//! - Fields are laid out one after another in the order of the table above in a
//!   single arena; every base offset is a multiple of 256.
//! - Sizes are computed with checked arithmetic; a request that overflows is
//!   refused, not wrapped.

use sim_scene::Scene;

use crate::error::WorldError;

/// The alignment of each environment's slice of each field, bytes.
pub const ENV_STRIDE_ALIGN_BYTES: usize = 256;

/// Bytes in one `f32`.
const F32: usize = 4;

/// One field of the world state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FieldId {
    /// Body world positions.
    BodyPos,
    /// Body world orientations.
    BodyQuat,
    /// Body world linear velocities.
    BodyLinVel,
    /// Body world angular velocities.
    BodyAngVel,
    /// Joint positions.
    Qpos,
    /// Joint velocities.
    Qvel,
    /// Actuator controls.
    Ctrl,
    /// Per-instance surface state.
    Surface,
}

impl FieldId {
    /// Every field, in layout order.
    pub const ALL: [FieldId; 8] = [
        FieldId::BodyPos,
        FieldId::BodyQuat,
        FieldId::BodyLinVel,
        FieldId::BodyAngVel,
        FieldId::Qpos,
        FieldId::Qvel,
        FieldId::Ctrl,
        FieldId::Surface,
    ];

    /// The position of this field in [`FieldId::ALL`].
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Where one field's data is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Field {
    /// Which field.
    pub id: FieldId,
    /// `f32` values per environment (without padding).
    pub floats_per_env: usize,
    /// Bytes from the start of one environment's data to the next's: the data
    /// size rounded up to a multiple of 256 (0 when there is no data).
    pub env_stride_bytes: usize,
    /// Byte offset of environment 0's data in the arena.
    pub base_offset_bytes: usize,
    /// Bytes the whole field occupies: `n_envs * env_stride_bytes`.
    pub size_bytes: usize,
}

impl Field {
    /// Byte offset of environment `env`'s data in the arena.
    pub fn env_offset_bytes(&self, env: usize) -> usize {
        self.base_offset_bytes + env * self.env_stride_bytes
    }

    /// `f32` values per environment including the padding (the stride in floats).
    pub fn env_stride_floats(&self) -> usize {
        self.env_stride_bytes / F32
    }
}

/// The layout of the world state for a scene and a number of environments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorldLayout {
    /// Environment slots.
    pub n_envs: u32,
    /// Bodies in the scene (the world is not one).
    pub n_bodies: u32,
    /// Joints in the scene.
    pub n_joints: u32,
    /// Length of the joint position vector.
    pub nq: u32,
    /// Length of the joint velocity vector.
    pub nv: u32,
    /// Actuators (control values per environment).
    pub n_actuators: u32,
    /// Renderable instances (surface-state records per environment).
    pub n_instances: u32,
    /// Per joint: where its coordinates start in `Qpos`.
    pub joint_qpos_adr: Vec<u32>,
    /// Per joint: where its degrees of freedom start in `Qvel`.
    pub joint_dof_adr: Vec<u32>,
    fields: [Field; 8],
    /// Bytes of the whole arena.
    pub total_bytes: usize,
}

impl WorldLayout {
    /// The layout for `scene` with `n_envs` environment slots.
    ///
    /// The scene is validated first. Refuses zero environments and sizes that
    /// overflow `usize` or `u32`.
    pub fn new(scene: &Scene, n_envs: u32) -> Result<WorldLayout, WorldError> {
        scene.validate()?;
        if n_envs == 0 {
            return Err(WorldError::Layout {
                reason: "at least one environment is required",
            });
        }
        let too_big = WorldError::Layout {
            reason: "the scene is too large for 32-bit counts or the arena overflows",
        };
        let n_bodies = u32::try_from(scene.bodies.len()).map_err(|_| too_big.clone())?;
        let n_joints = u32::try_from(scene.joints.len()).map_err(|_| too_big.clone())?;
        let n_actuators = u32::try_from(scene.actuators.len()).map_err(|_| too_big.clone())?;
        let n_instances = u32::try_from(scene.instances.len()).map_err(|_| too_big.clone())?;
        let nq = u32::try_from(scene.nq()).map_err(|_| too_big.clone())?;
        let nv = u32::try_from(scene.nv()).map_err(|_| too_big.clone())?;

        let mut joint_qpos_adr = Vec::with_capacity(scene.joints.len());
        let mut joint_dof_adr = Vec::with_capacity(scene.joints.len());
        let (mut q, mut v) = (0u32, 0u32);
        for joint in &scene.joints {
            joint_qpos_adr.push(q);
            joint_dof_adr.push(v);
            q += joint.kind.nq() as u32;
            v += joint.kind.nv() as u32;
        }

        let floats = |id: FieldId| -> usize {
            let nb = n_bodies as usize;
            match id {
                FieldId::BodyPos | FieldId::BodyLinVel | FieldId::BodyAngVel => 3 * nb,
                FieldId::BodyQuat => 4 * nb,
                FieldId::Qpos => nq as usize,
                FieldId::Qvel => nv as usize,
                FieldId::Ctrl => n_actuators as usize,
                FieldId::Surface => 4 * n_instances as usize,
            }
        };

        let mut fields = [Field {
            id: FieldId::BodyPos,
            floats_per_env: 0,
            env_stride_bytes: 0,
            base_offset_bytes: 0,
            size_bytes: 0,
        }; 8];
        let mut base = 0usize;
        for (slot, id) in fields.iter_mut().zip(FieldId::ALL) {
            let floats_per_env = floats(id);
            let bytes = floats_per_env
                .checked_mul(F32)
                .ok_or_else(|| too_big.clone())?;
            let stride = bytes
                .div_ceil(ENV_STRIDE_ALIGN_BYTES)
                .checked_mul(ENV_STRIDE_ALIGN_BYTES)
                .ok_or_else(|| too_big.clone())?;
            let size = stride
                .checked_mul(n_envs as usize)
                .ok_or_else(|| too_big.clone())?;
            *slot = Field {
                id,
                floats_per_env,
                env_stride_bytes: stride,
                base_offset_bytes: base,
                size_bytes: size,
            };
            base = base.checked_add(size).ok_or_else(|| too_big.clone())?;
        }

        Ok(WorldLayout {
            n_envs,
            n_bodies,
            n_joints,
            nq,
            nv,
            n_actuators,
            n_instances,
            joint_qpos_adr,
            joint_dof_adr,
            fields,
            total_bytes: base,
        })
    }

    /// The layout of one field.
    pub fn field(&self, id: FieldId) -> &Field {
        &self.fields[id.index()]
    }

    /// Every field, in layout order.
    pub fn fields(&self) -> &[Field; 8] {
        &self.fields
    }
}
