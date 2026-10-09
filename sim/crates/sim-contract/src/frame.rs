//! Camera frames as references to device memory.
//!
//! The renderer writes frames into GPU buffers and the training loop reads them
//! from the same buffers, so a frame never round-trips through the host: the
//! bundle carries a [`FrameRef`], not pixels. This module defines what the
//! reference says (which buffer, where, how the pixels are laid out) and checks
//! that the layout is consistent with the pixel type.
//!
//! Invariants:
//! - A frame is `height` rows of `width` pixels of `channels` interleaved
//!   channels, each channel one `dtype` element; row `y` starts at
//!   `byte_offset + y * row_stride_bytes`.
//! - Rows may be padded (`row_stride_bytes` larger than a packed row) but never
//!   overlap, and every offset and stride is a multiple of the element size.
//! - Contract v0 fixes three formats, all 448 x 448: RGB `u8` x 3, depth `f16`
//!   in metres x 1, segmentation `u16` x 1 (an instance id per pixel). Depth is
//!   pixel-aligned with the RGB frame and rides in the bundle with it.
//!   **Segmentation is evaluation only**: it is carried by
//!   [`crate::GroundTruth`], never by a [`crate::Bundle`], and is pixel-aligned
//!   with the RGB frame of the same camera and tick.
//!
//! The camera conventions (OpenCV camera frame, intrinsics, depth, RGB encoding)
//! are fixed by the contract owner and stated on [`FrameRef`].

use serde::{Deserialize, Serialize};

use crate::ContractError;

/// Width of every v0 frame, in pixels.
pub const V0_FRAME_WIDTH: u32 = 448;
/// Height of every v0 frame, in pixels.
pub const V0_FRAME_HEIGHT: u32 = 448;

/// The identity of a device buffer, assigned by the simulator.
///
/// Opaque: the number means nothing to the consumer except as the handle to
/// give the device-side interop layer. It is only meaningful inside the process
/// (and device) that issued it, and a fixture's value is a placeholder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceBufferId(pub u64);

/// The element type of one channel of a frame.
///
/// JSON: `"u8"`, `"f16"`, `"u16"`, `"f32"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameDtype {
    /// Unsigned 8-bit integer.
    U8,
    /// IEEE 754 binary16.
    F16,
    /// Unsigned 16-bit integer.
    U16,
    /// IEEE 754 binary32.
    F32,
}

impl FrameDtype {
    /// Bytes in one element.
    pub const fn bytes(self) -> u32 {
        match self {
            FrameDtype::U8 => 1,
            FrameDtype::F16 | FrameDtype::U16 => 2,
            FrameDtype::F32 => 4,
        }
    }
}

/// What the pixels of a frame mean.
///
/// JSON: `"rgb"`, `"depth_metres"`, `"segmentation"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameSemantic {
    /// Colour, red then green then blue.
    Rgb,
    /// The z of the camera frame (distance along the viewing axis, not along
    /// the pixel's ray), in metres; `+inf` where the ray hits nothing.
    DepthMetres,
    /// An instance id per pixel naming the object seen there. **Evaluation
    /// only**: carried by [`crate::GroundTruth`], never by a bundle. Ids are
    /// stable for a whole episode (an object keeps its id through occlusion and
    /// motion) and 0 is background. That is a property the simulator must keep;
    /// this crate checks one frame's format, not ids across ticks.
    Segmentation,
}

/// A frame that lives in device memory.
///
/// Camera conventions, fixed by the contract owner:
/// - **Camera frame**: OpenCV. x is right, y is down, z is forward along the
///   view axis. A camera `Pose` quaternion rotates this frame into the world.
/// - **Intrinsics** are in pixels of the 448 x 448 frame, with pixel centres at
///   integer coordinates: the centre of the top-left pixel is (0, 0), so a
///   centred principal point is `((W - 1) / 2, (H - 1) / 2)`, that is
///   (223.5, 223.5). The v0 bundle carries no intrinsics; wherever they are
///   stated they are in these terms.
/// - **Depth** ([`FrameSemantic::DepthMetres`]) is z in the camera frame, along
///   the view axis (not the distance along the pixel's ray), in metres. A pixel
///   whose ray hits nothing is `+inf` (`f16` bits `0x7C00`). This is a pixel on
///   the device, not a value this crate carries or checks: the crate's "no
///   infinity" rule is about the numbers in its own types, which JSON must be
///   able to hold.
/// - **RGB** ([`FrameSemantic::Rgb`]) is sRGB-encoded `u8`: 8 bits per channel
///   with the sRGB transfer curve applied, not linear light.
///
/// Invariants (checked by [`FrameRef::validate`]):
/// - `width`, `height` and `channels` are at least 1.
/// - `row_stride_bytes >= width * channels * dtype.bytes()`, and both
///   `row_stride_bytes` and `byte_offset` are multiples of `dtype.bytes()`.
/// - `byte_offset` plus the frame's extent fits in a `u64`.
///
/// [`FrameRef::validate_v0`] adds the v0 formats: the dtype, channel count and
/// 448 x 448 size that go with the frame's `semantic`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FrameRef {
    /// The device buffer holding the pixels.
    pub buffer: DeviceBufferId,
    /// Where the first row starts inside the buffer.
    pub byte_offset: u64,
    /// Pixels per row.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// Interleaved channels per pixel.
    pub channels: u8,
    /// The element type of each channel.
    pub dtype: FrameDtype,
    /// Bytes from the start of one row to the start of the next.
    pub row_stride_bytes: u32,
    /// What the pixels mean.
    pub semantic: FrameSemantic,
}

impl FrameRef {
    /// The v0 RGB format (448 x 448 x 3, `u8`, tightly packed) in the
    /// placeholder buffer 0 at offset 0. Place it with [`FrameRef::at`].
    pub fn v0_rgb() -> Self {
        Self::packed(3, FrameDtype::U8, FrameSemantic::Rgb)
    }

    /// The v0 depth format (448 x 448 x 1, `f16` metres, tightly packed).
    pub fn v0_depth() -> Self {
        Self::packed(1, FrameDtype::F16, FrameSemantic::DepthMetres)
    }

    /// The v0 segmentation format (448 x 448 x 1, `u16`, tightly packed).
    pub fn v0_segmentation() -> Self {
        Self::packed(1, FrameDtype::U16, FrameSemantic::Segmentation)
    }

    fn packed(channels: u8, dtype: FrameDtype, semantic: FrameSemantic) -> Self {
        Self {
            buffer: DeviceBufferId(0),
            byte_offset: 0,
            width: V0_FRAME_WIDTH,
            height: V0_FRAME_HEIGHT,
            channels,
            dtype,
            row_stride_bytes: V0_FRAME_WIDTH * u32::from(channels) * dtype.bytes(),
            semantic,
        }
    }

    /// The same frame layout in another place.
    #[must_use]
    pub fn at(self, buffer: DeviceBufferId, byte_offset: u64) -> Self {
        Self {
            buffer,
            byte_offset,
            ..self
        }
    }

    /// Bytes in one tightly packed row.
    pub fn packed_row_bytes(&self) -> u64 {
        u64::from(self.width) * u64::from(self.channels) * u64::from(self.dtype.bytes())
    }

    /// Bytes from the first byte of the first row to one past the last pixel of
    /// the last row (the last row is not padded). `None` for a frame with no
    /// rows, or when the extent does not fit a `u64`.
    pub fn byte_extent(&self) -> Option<u64> {
        let rows_before_last = u64::from(self.height.checked_sub(1)?);
        u64::from(self.row_stride_bytes)
            .checked_mul(rows_before_last)?
            .checked_add(self.packed_row_bytes())
    }

    /// Checks the dimensions and stride against the dtype.
    pub fn validate(&self) -> Result<(), ContractError> {
        for (field, value) in [
            ("width", u64::from(self.width)),
            ("height", u64::from(self.height)),
            ("channels", u64::from(self.channels)),
        ] {
            if value == 0 {
                return Err(ContractError::Empty { field });
            }
        }
        let required = self.packed_row_bytes();
        if u64::from(self.row_stride_bytes) < required {
            return Err(ContractError::Stride {
                required_bytes: required,
                found_bytes: u64::from(self.row_stride_bytes),
            });
        }
        let element = u64::from(self.dtype.bytes());
        for (field, value) in [
            ("row_stride_bytes", u64::from(self.row_stride_bytes)),
            ("byte_offset", self.byte_offset),
        ] {
            if value % element != 0 {
                return Err(ContractError::Misaligned {
                    field,
                    alignment: element,
                    found: value,
                });
            }
        }
        let fits = self
            .byte_extent()
            .and_then(|extent| self.byte_offset.checked_add(extent))
            .is_some();
        if !fits {
            return Err(ContractError::OutOfRange {
                field: "byte_offset",
                reason: "the frame's extent does not fit in 64 bits",
            });
        }
        Ok(())
    }

    /// [`FrameRef::validate`], and the v0 format of a frame sent as `expected`:
    /// its `semantic` is `expected`, its dtype and channel count are the ones v0
    /// fixes for that semantic, and it is 448 x 448.
    pub fn validate_v0(&self, expected: FrameSemantic) -> Result<(), ContractError> {
        self.validate()?;
        if self.semantic != expected {
            return Err(ContractError::Format {
                reason: "the frame's semantic is not the channel's",
            });
        }
        let (channels, dtype) = match expected {
            FrameSemantic::Rgb => (3, FrameDtype::U8),
            FrameSemantic::DepthMetres => (1, FrameDtype::F16),
            FrameSemantic::Segmentation => (1, FrameDtype::U16),
        };
        if self.channels != channels {
            return Err(ContractError::Format {
                reason: "wrong channel count for the channel",
            });
        }
        if self.dtype != dtype {
            return Err(ContractError::Format {
                reason: "wrong element type for the channel",
            });
        }
        if self.width != V0_FRAME_WIDTH || self.height != V0_FRAME_HEIGHT {
            return Err(ContractError::Format {
                reason: "v0 frames are 448 x 448",
            });
        }
        Ok(())
    }
}
