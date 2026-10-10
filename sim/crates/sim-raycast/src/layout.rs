//! Byte layouts shared with the kernels, and the layout of the output frames.
//!
//! Every record is a whole number of 16-byte vec4s, little-endian `f32` or
//! `u32` words, exactly as the std430 structs in `kernels/common.glsl` read
//! them. A change to a struct there is a change to its writer here.
//!
//! Output frames follow contract v0 ([`sim_contract::FrameRef`]): frame `f`
//! (= `env * cameras_per_env + camera`) of each channel starts at
//! `f * frame_bytes` in that channel's buffer, and its rows are tightly packed.

use sim_contract::{DeviceBufferId, FrameRef, FrameSemantic};

use crate::math::{Pose, Quat, Vec3};

/// Bytes of one BVH node.
pub const NODE_BYTES: usize = 32;
/// Bytes of one triangle (first vertex and two edges).
pub const TRI_BYTES: usize = 48;
/// Bytes of one triangle's three vertex normals.
pub const TRI_NORMAL_BYTES: usize = 48;
/// Bytes of one material.
pub const MATERIAL_BYTES: usize = 48;
/// Bytes of one pose (the world-state record the physics writes per body).
pub const POSE_BYTES: usize = 32;
/// Bytes of one static instance.
pub const INSTANCE_STATIC_BYTES: usize = 64;
/// Bytes of one posed instance (per environment, written on the device).
pub const INSTANCE_POSED_BYTES: usize = 80;
/// Bytes of one camera.
pub const CAMERA_BYTES: usize = 64;
/// Bytes of one body's motion parameters (the benchmark's physics stand-in).
pub const MOTION_BYTES: usize = 32;

/// The body index of an instance that is not attached to a body.
pub const STATIC_BODY: u32 = u32::MAX;

/// A little-endian writer of 32-bit words.
#[derive(Default)]
pub struct Words(pub Vec<u8>);

impl Words {
    /// Append an `f32`.
    pub fn f(&mut self, x: f32) -> &mut Self {
        self.0.extend_from_slice(&x.to_le_bytes());
        self
    }

    /// Append a `u32`.
    pub fn u(&mut self, x: u32) -> &mut Self {
        self.0.extend_from_slice(&x.to_le_bytes());
        self
    }

    /// Append a 3-vector and a fourth word given as raw bits.
    pub fn v3_bits(&mut self, v: Vec3, w: u32) -> &mut Self {
        self.f(v[0]).f(v[1]).f(v[2]).u(w)
    }

    /// Append a 3-vector and a fourth `f32`.
    pub fn v3_f(&mut self, v: Vec3, w: f32) -> &mut Self {
        self.f(v[0]).f(v[1]).f(v[2]).f(w)
    }

    /// Append a quaternion.
    pub fn quat(&mut self, q: Quat) -> &mut Self {
        self.f(q[0]).f(q[1]).f(q[2]).f(q[3])
    }
}

/// The 32-byte pose record of the world state: position, a pad word, then the
/// quaternion.
pub fn pose_bytes(p: &Pose) -> [u8; POSE_BYTES] {
    let mut w = Words::default();
    w.v3_f(p.position, 0.0).quat(p.rotation);
    w.0.try_into().expect("a pose is 32 bytes")
}

/// Read the `i`-th little-endian `f32` of `bytes`.
pub fn f32_at(bytes: &[u8], i: usize) -> f32 {
    f32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
}

/// Read the `i`-th little-endian `u32` of `bytes`.
pub fn u32_at(bytes: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
}

/// The pose record at index `i` of a pose buffer's bytes.
pub fn pose_at(bytes: &[u8], i: usize) -> Pose {
    let r = &bytes[i * POSE_BYTES..(i + 1) * POSE_BYTES];
    Pose {
        position: [f32_at(r, 0), f32_at(r, 1), f32_at(r, 2)],
        rotation: [f32_at(r, 4), f32_at(r, 5), f32_at(r, 6), f32_at(r, 7)],
    }
}

/// The size of the frames one renderer writes, and where frame `f` of each
/// channel starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameLayout {
    /// Pixels per row.
    pub width: u32,
    /// Rows.
    pub height: u32,
}

impl FrameLayout {
    /// The contract v0 frame, 448 x 448.
    pub const V0: FrameLayout = FrameLayout {
        width: sim_contract::V0_FRAME_WIDTH,
        height: sim_contract::V0_FRAME_HEIGHT,
    };

    /// Pixels in one frame.
    pub fn pixels(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Bytes of one RGB frame (3 bytes a pixel).
    pub fn rgb_bytes(&self) -> u64 {
        3 * self.pixels()
    }

    /// Bytes of one depth frame (binary16 a pixel).
    pub fn depth_bytes(&self) -> u64 {
        2 * self.pixels()
    }

    /// Bytes of one segmentation frame (uint16 a pixel).
    pub fn seg_bytes(&self) -> u64 {
        2 * self.pixels()
    }

    /// The contract reference to frame `f` of a channel held in `buffer`.
    pub fn frame_ref(&self, semantic: FrameSemantic, buffer: DeviceBufferId, f: u64) -> FrameRef {
        let base = match semantic {
            FrameSemantic::Rgb => FrameRef::v0_rgb(),
            FrameSemantic::DepthMetres => FrameRef::v0_depth(),
            FrameSemantic::Segmentation => FrameRef::v0_segmentation(),
        };
        let per_frame = match semantic {
            FrameSemantic::Rgb => self.rgb_bytes(),
            FrameSemantic::DepthMetres => self.depth_bytes(),
            FrameSemantic::Segmentation => self.seg_bytes(),
        };
        let channels = u32::from(base.channels);
        FrameRef {
            width: self.width,
            height: self.height,
            row_stride_bytes: self.width * channels * base.dtype.bytes(),
            ..base
        }
        .at(buffer, f * per_frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v0_frame_refs_pass_the_contract_check_at_every_frame() {
        let l = FrameLayout::V0;
        for f in [0u64, 1, 7, 1023] {
            for (semantic, per) in [
                (FrameSemantic::Rgb, l.rgb_bytes()),
                (FrameSemantic::DepthMetres, l.depth_bytes()),
                (FrameSemantic::Segmentation, l.seg_bytes()),
            ] {
                let r = l.frame_ref(semantic, DeviceBufferId(9), f);
                r.validate_v0(semantic).unwrap();
                assert_eq!(r.byte_offset, f * per);
                assert_eq!(r.byte_extent(), Some(per));
            }
        }
    }

    #[test]
    fn a_pose_record_round_trips() {
        let p = Pose::new([1.0, -2.0, 3.5], [0.0, 0.6, 0.0, 0.8]);
        let b = pose_bytes(&p);
        assert_eq!(pose_at(&b, 0), p);
        assert_eq!(f32_at(&b, 3), 0.0);
    }
}
