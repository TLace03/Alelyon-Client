//! The host mirror of the world state.
//!
//! One `Vec<f32>` per field of [`crate::WorldLayout`], so the host can build,
//! inspect and compare exactly what the device buffers will hold, and upload a
//! field with one copy. It is also the CPU reference the GPU kernels of later
//! phases are checked against.
//!
//! Invariants:
//! - Each field's `Vec` has `n_envs * env_stride_bytes / 4` floats: the padding of
//!   [`crate::WorldLayout`] is present and always zero, so [`HostWorld::field_bytes`]
//!   is byte for byte the device buffer of that field.
//! - A new world is at the scene's reference pose with zero velocities, zero
//!   controls and zero surface state (all zero until phase 2); body quaternions are
//!   unit, so a world is valid from the moment it exists.
//! - [`HostWorld::reset`] is a pure function of the scene, the seed and the noise:
//!   two worlds of the same scene and size reset with the same seed are bitwise
//!   equal, and different seeds give different worlds (when the scene has a joint
//!   to jitter). The jitter of environment `e` comes only from
//!   `Rng::new(seed, e)` at `(step 0, STREAM_RESET, lane)`, so it does not depend
//!   on how many environments there are or the order they are reset in.
//! - A reset rebuilds the body poses by forward kinematics from the jittered joint
//!   coordinates, so `BodyPos`/`BodyQuat` always agree with `Qpos`.
//! - Per-environment accessors take `env: u32` and, like slice indexing, panic when
//!   `env >= n_envs`; they never panic on the data.

use sim_scene::pose::{quat_from_axis_angle, quat_mul, quat_normalize};
use sim_scene::{JointKind, Scene};

use crate::error::WorldError;
use crate::kinematics::Kinematics;
use crate::layout::{FieldId, WorldLayout};
use crate::rng::{Rng, STREAM_RESET};

/// How far a reset moves the initial pose, per kind of joint coordinate.
///
/// Each coordinate is offset by a uniform amount in (-amplitude, amplitude)
/// drawn from the environment's Philox stream. A hinge or slide is clamped into
/// its range; a ball joint's rotation angle is clamped to its maximum angle; a
/// free joint is moved in position (each axis) and rotated by a small rotation
/// vector (each component). Zero amplitude leaves the reference pose exact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResetNoise {
    /// Slide joints, metres.
    pub slide_m: f64,
    /// Hinge joints, radians.
    pub hinge_rad: f64,
    /// Ball joints, radians per rotation-vector component.
    pub ball_rad: f64,
    /// Free joints' position, metres per axis.
    pub free_pos_m: f64,
    /// Free joints' orientation, radians per rotation-vector component.
    pub free_rot_rad: f64,
}

impl ResetNoise {
    /// No jitter: a reset restores the exact reference pose.
    pub const NONE: ResetNoise = ResetNoise {
        slide_m: 0.0,
        hinge_rad: 0.0,
        ball_rad: 0.0,
        free_pos_m: 0.0,
        free_rot_rad: 0.0,
    };
}

impl Default for ResetNoise {
    /// A small jitter, enough that different seeds give visibly different
    /// starts and small enough that a reset never leaves a joint's range:
    /// 1 mm for slides, 0.01 rad for hinge, ball and free rotations, 1 cm for
    /// free positions.
    fn default() -> Self {
        ResetNoise {
            slide_m: 1e-3,
            hinge_rad: 1e-2,
            ball_rad: 1e-2,
            free_pos_m: 1e-2,
            free_rot_rad: 1e-2,
        }
    }
}

/// The host mirror of the world state of `n_envs` environments.
#[derive(Clone, Debug)]
pub struct HostWorld {
    layout: WorldLayout,
    data: [Vec<f32>; 8],
    seed: u64,
    kin: Kinematics,
}

impl HostWorld {
    /// A world of `n_envs` environments of `scene`, at the reference pose.
    pub fn new(scene: &Scene, n_envs: u32) -> Result<HostWorld, WorldError> {
        let layout = WorldLayout::new(scene, n_envs)?;
        let data = std::array::from_fn(|i| vec![0.0f32; layout.fields()[i].size_bytes / 4]);
        let mut world = HostWorld {
            layout,
            data,
            seed: 0,
            kin: Kinematics::new(scene),
        };
        world.reset_with(0, &ResetNoise::NONE);
        Ok(world)
    }

    /// The layout of this world.
    pub fn layout(&self) -> &WorldLayout {
        &self.layout
    }

    /// The seed of the last reset (0 for a new world).
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Resets every environment from the scene's reference pose with the
    /// default [`ResetNoise`], seeded by `seed`.
    pub fn reset(&mut self, seed: u64) {
        self.reset_with(seed, &ResetNoise::default());
    }

    /// [`HostWorld::reset`] with explicit noise.
    pub fn reset_with(&mut self, seed: u64, noise: &ResetNoise) {
        self.seed = seed;
        for v in self.data.iter_mut() {
            v.fill(0.0);
        }
        let n_envs = self.layout.n_envs;
        for env in 0..n_envs {
            let rng = Rng::new(seed, env);
            let mut q = self.kin.qpos0.clone();
            for (j, joint) in self.kin.joints.iter().enumerate() {
                let lane = 2 * j as u32;
                let a = rng.signed_units(0, STREAM_RESET, lane);
                let adr = joint.qpos_adr;
                match joint.kind {
                    JointKind::Hinge { .. } | JointKind::Slide { .. } => {
                        let amp = if matches!(joint.kind, JointKind::Hinge { .. }) {
                            noise.hinge_rad
                        } else {
                            noise.slide_m
                        };
                        let mut v = q[adr] + amp * a[0];
                        if let Some([lo, hi]) = joint.range {
                            v = v.clamp(lo, hi);
                        }
                        q[adr] = v;
                    }
                    JointKind::Ball => {
                        let rv = [
                            noise.ball_rad * a[0],
                            noise.ball_rad * a[1],
                            noise.ball_rad * a[2],
                        ];
                        let mut angle = (rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]).sqrt();
                        if let Some([_, max]) = joint.range {
                            angle = angle.min(max);
                        }
                        let dq = quat_from_axis_angle(rv, angle);
                        let base = [q[adr], q[adr + 1], q[adr + 2], q[adr + 3]];
                        q[adr..adr + 4].copy_from_slice(&quat_normalize(quat_mul(base, dq)));
                    }
                    JointKind::Free => {
                        let b = rng.signed_units(0, STREAM_RESET, lane + 1);
                        for k in 0..3 {
                            q[adr + k] += noise.free_pos_m * a[k];
                        }
                        let rv = [
                            noise.free_rot_rad * a[3],
                            noise.free_rot_rad * b[0],
                            noise.free_rot_rad * b[1],
                        ];
                        let angle = (rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]).sqrt();
                        let dq = quat_from_axis_angle(rv, angle);
                        let base = [q[adr + 3], q[adr + 4], q[adr + 5], q[adr + 6]];
                        q[adr + 3..adr + 7].copy_from_slice(&quat_normalize(quat_mul(base, dq)));
                    }
                }
            }
            self.write_configuration(env, &q);
        }
    }

    /// Sets environment `env`'s joint coordinates to `qpos` (`nq` values, in the
    /// layout of [`crate::WorldLayout`]; quaternions `[x, y, z, w]`, renormalised)
    /// and recomputes every body's world pose by forward kinematics. Velocities,
    /// controls and surface state are left as they are.
    ///
    /// Refuses an environment out of range, a wrong length and non-finite values.
    pub fn set_qpos(&mut self, env: u32, qpos: &[f64]) -> Result<(), WorldError> {
        if env >= self.layout.n_envs {
            return Err(WorldError::Layout {
                reason: "environment index out of range",
            });
        }
        if qpos.len() != self.layout.nq as usize {
            return Err(WorldError::Layout {
                reason: "qpos has the wrong length for the scene",
            });
        }
        if qpos.iter().any(|v| !v.is_finite()) {
            return Err(WorldError::Layout {
                reason: "qpos must be finite",
            });
        }
        self.write_configuration(env, qpos);
        Ok(())
    }

    /// Forward kinematics of `q`, written to `BodyPos`, `BodyQuat` and `Qpos` of `env`.
    fn write_configuration(&mut self, env: u32, q: &[f64]) {
        let poses = self.kin.forward(q);
        for (b, pose) in poses.iter().enumerate() {
            let pos = self.env_slice_mut(FieldId::BodyPos, env);
            for k in 0..3 {
                pos[3 * b + k] = pose.pos[k] as f32;
            }
            let quat = self.env_slice_mut(FieldId::BodyQuat, env);
            for k in 0..4 {
                quat[4 * b + k] = pose.quat[k] as f32;
            }
        }
        let qpos = self.env_slice_mut(FieldId::Qpos, env);
        for (dst, src) in qpos.iter_mut().zip(q) {
            *dst = *src as f32;
        }
    }

    /// The data of environment `env` in `field`, without padding. Panics if
    /// `env >= n_envs`.
    pub fn env_slice(&self, field: FieldId, env: u32) -> &[f32] {
        let f = self.layout.field(field);
        let start = env as usize * f.env_stride_floats();
        &self.data[field.index()][start..start + f.floats_per_env]
    }

    /// Mutable [`HostWorld::env_slice`].
    pub fn env_slice_mut(&mut self, field: FieldId, env: u32) -> &mut [f32] {
        let f = *self.layout.field(field);
        let start = env as usize * f.env_stride_floats();
        &mut self.data[field.index()][start..start + f.floats_per_env]
    }

    /// The whole of `field` (all environments, with padding) as floats.
    pub fn field(&self, field: FieldId) -> &[f32] {
        &self.data[field.index()]
    }

    /// The whole of `field` as little-endian bytes: the device buffer's contents.
    pub fn field_bytes(&self, field: FieldId) -> Vec<u8> {
        self.data[field.index()]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    }

    /// Every field, concatenated in layout order: the arena's contents
    /// (`layout().total_bytes` bytes).
    pub fn arena_bytes(&self) -> Vec<u8> {
        FieldId::ALL
            .into_iter()
            .flat_map(|id| self.field_bytes(id))
            .collect()
    }

    /// Whether two worlds hold the same bits (not the same values: `NaN` equals
    /// itself and `-0.0` differs from `0.0`) and the same layout.
    pub fn bit_eq(&self, other: &HostWorld) -> bool {
        self.layout == other.layout
            && self.data.iter().zip(&other.data).all(|(a, b)| {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
            })
    }

    /// Body world positions of `env`: `3 * n_bodies` floats.
    pub fn body_pos(&self, env: u32) -> &[f32] {
        self.env_slice(FieldId::BodyPos, env)
    }
    /// Mutable [`HostWorld::body_pos`].
    pub fn body_pos_mut(&mut self, env: u32) -> &mut [f32] {
        self.env_slice_mut(FieldId::BodyPos, env)
    }
    /// Body world orientations of `env`: `4 * n_bodies` floats, `[x, y, z, w]`.
    pub fn body_quat(&self, env: u32) -> &[f32] {
        self.env_slice(FieldId::BodyQuat, env)
    }
    /// Mutable [`HostWorld::body_quat`].
    pub fn body_quat_mut(&mut self, env: u32) -> &mut [f32] {
        self.env_slice_mut(FieldId::BodyQuat, env)
    }
    /// Body world linear velocities of `env`: `3 * n_bodies` floats.
    pub fn body_linvel(&self, env: u32) -> &[f32] {
        self.env_slice(FieldId::BodyLinVel, env)
    }
    /// Body world angular velocities of `env`: `3 * n_bodies` floats.
    pub fn body_angvel(&self, env: u32) -> &[f32] {
        self.env_slice(FieldId::BodyAngVel, env)
    }
    /// Joint positions of `env`: `nq` floats.
    pub fn qpos(&self, env: u32) -> &[f32] {
        self.env_slice(FieldId::Qpos, env)
    }
    /// Joint velocities of `env`: `nv` floats.
    pub fn qvel(&self, env: u32) -> &[f32] {
        self.env_slice(FieldId::Qvel, env)
    }
    /// Controls of `env`: one float per actuator.
    pub fn ctrl(&self, env: u32) -> &[f32] {
        self.env_slice(FieldId::Ctrl, env)
    }
    /// Mutable [`HostWorld::ctrl`].
    pub fn ctrl_mut(&mut self, env: u32) -> &mut [f32] {
        self.env_slice_mut(FieldId::Ctrl, env)
    }
    /// Per-instance surface state of `env`: 4 floats per instance (temperature
    /// K, browning 0..1, wetness 0..1, reserved).
    pub fn surface(&self, env: u32) -> &[f32] {
        self.env_slice(FieldId::Surface, env)
    }

    /// The pose of body `body` in `env`: position and `[x, y, z, w]` quaternion.
    /// Panics if `env` or `body` is out of range.
    pub fn body_pose(&self, env: u32, body: usize) -> ([f32; 3], [f32; 4]) {
        let p = self.body_pos(env);
        let q = self.body_quat(env);
        (
            [p[3 * body], p[3 * body + 1], p[3 * body + 2]],
            [
                q[4 * body],
                q[4 * body + 1],
                q[4 * body + 2],
                q[4 * body + 3],
            ],
        )
    }
}
