//! Sinai's face, as one crate that every window drawing Sinai links, so they all draw one Sinai.
//!
//! - [`body`]: the baked bust (`assets/sinai_body.bin`, compiled in) and every morph target it carries, with the
//!   arithmetic a window needs whenever a shape changes. Plain arithmetic: no window, no device.
//! - [`expression`]: the tones Sinai's face takes, as weights on the bust's expression units.
//! - [`appearance`]: a person's shaping of Sinai (the creator's catalog, `assets/sinai_controls.json`), its
//!   share code and the file it is kept in.
//! - [`state_home`]: where a window keeps that file, and the one-time checked copy into it.
//! - [`shaders`]: the WGSL that draws the bust and its world. WGSL has no includes, so each is a whole module.

pub mod appearance;
pub mod body;
pub mod expression;
pub mod state_home;

/// The WGSL a window compiles to draw Sinai: the bust, the full-screen copy, the valley and the sky.
pub mod shaders {
    /// The bust: its body, its eyes and the lattice it is made of.
    pub const ANGEL: &str = include_str!("angel.wgsl");
    /// The full-screen copy of the offscreen frame.
    pub const BLIT: &str = include_str!("blit.wgsl");
    /// The valley below Sinai.
    pub const TERRAIN: &str = include_str!("terrain.wgsl");
    /// The sky over it: its colour, its sun and its stars.
    pub const SKY: &str = include_str!("sky.wgsl");
}
