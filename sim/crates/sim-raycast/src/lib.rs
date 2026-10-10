//! The CPU ray caster of Sinai's simulator: the scene the renderer draws, its
//! packing, the pinhole camera, and the host reference renderer.
//!
//! This is the renderer's algorithm on the processor, in Rust `f32`. A scene
//! description ([`scene::SceneDesc`], converted from the one scene description
//! by [`from_scene`]) is packed into flat records ([`scene::PackedScene`]: a
//! BVH per mesh, materials and instances, in the byte layouts of
//! [`layout`]); [`reference`] casts one primary ray per pixel through the
//! posed instances, shades the closest hit with physically based materials
//! under a sun, a sky and a ground, and tone-maps it to sRGB. It writes the
//! observation contract's three camera channels (RGB, depth in `f16` metres,
//! and the segmentation id) in their byte layouts.
//!
//! It opens no graphics device and has no GPU dependency. The device renderer
//! (`sim-render`, on a compute device) runs the same algorithm in compute
//! kernels and re-exports these modules, so its frames can be checked against
//! this one pixel by pixel; a desktop app can draw the simulator with it on the
//! processor alone.
//!
//! Modules: [`scene`] (what is rendered), [`reference`](mod@reference) (the
//! ray caster), [`camera`] (the pinhole model), [`from_scene`] (from the one
//! scene description), [`scenes`] (built-in scenes for benchmarks and tests),
//! and helpers ([`bvh`], [`mesh`], [`math`], [`layout`], [`f16`]).

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod bvh;
pub mod camera;
pub mod f16;
pub mod from_scene;
pub mod layout;
pub mod math;
pub mod mesh;
pub mod reference;
pub mod scene;
pub mod scenes;

use std::fmt;

/// Why a scene was refused: it cannot be rendered as described.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SceneError(pub String);

impl fmt::Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "scene refused: {}", self.0)
    }
}

impl std::error::Error for SceneError {}
