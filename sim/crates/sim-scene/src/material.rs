//! Materials: what a surface is made of, for every sense.
//!
//! One record per material, in four parts, each read by a different consumer of
//! the one world state: the mechanical part by physics, the thermal part by heat
//! transport (phase 2 and later), the acoustic part by sound propagation, the
//! optical part by the renderer. Lane R owns and reviews the optical part.
//!
//! Invariants (checked by [`Material::validate`]):
//! - Every number is finite. Every field has a documented default (the `Default`
//!   of its part) and a JSON file may omit a whole part or any field of it.
//! - Mechanical: `density` > 0 kg/m^3; every friction coefficient >= 0;
//!   `restitution` in `[0, 1]`.
//! - Thermal: conductivity > 0 W/(m K); heat capacity > 0 J/(kg K); emissivity in
//!   `[0, 1]`.
//! - Acoustic: each band's `absorption` in `[0, 1]` (the fraction of incident
//!   energy absorbed, bands low, mid, high); `scattering` in `[0, 1]`;
//!   `transmission` in `[0, 1]`; and in every band `absorption + transmission <= 1`
//!   (the rest is reflected; a surface cannot return negative energy).
//! - Optical (glTF 2.0 metallic-roughness semantics, agreed with Lane R):
//!   `base_colour_linear` is LINEAR (not sRGB), each channel in `[0, 1]`, like
//!   glTF's `baseColorFactor` without alpha; `roughness` is perceptual
//!   roughness in `[0, 1]`; `metallic` in `[0, 1]`; `emission_linear` is linear
//!   radiance, each channel >= 0 (values above 1 are allowed). The optional
//!   procedural `checker` (a stand-in for textures) has `cells_per_m` > 0 and
//!   `dark_multiplier` in `[0, 1]`. Colours authored in sRGB (MJCF `rgba`) are
//!   converted with [`srgb_to_linear`] on import.
//! - Where a geom carries its own `density` and `friction`, physics reads the
//!   geom's: the mechanical part is the material's reference values, the ones a
//!   scene author starts a geom from.

use serde::{Deserialize, Serialize};

use crate::error::{Result, SceneError};

/// A material.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Material {
    /// The material's name. Unique within a scene when non-empty.
    pub name: String,
    /// Mechanical properties, read by physics.
    #[serde(default)]
    pub mechanical: Mechanical,
    /// Thermal properties, read by heat transport.
    #[serde(default)]
    pub thermal: Thermal,
    /// Acoustic properties, read by sound propagation.
    #[serde(default)]
    pub acoustic: Acoustic,
    /// Optical properties, read by the renderer. Owned by Lane R.
    #[serde(default)]
    pub optical: Optical,
}

impl Material {
    /// A material with every property at its documented default.
    pub fn named(name: impl Into<String>) -> Material {
        Material {
            name: name.into(),
            mechanical: Mechanical::default(),
            thermal: Thermal::default(),
            acoustic: Acoustic::default(),
            optical: Optical::default(),
        }
    }

    /// Checks the physical ranges listed in the module note. `path` names this
    /// material in the scene (`materials[2]`).
    pub fn validate(&self, path: &str) -> Result<()> {
        let m = &self.mechanical;
        positive_f32(&format!("{path}.mechanical.density"), m.density)?;
        for (i, &f) in m.friction.iter().enumerate() {
            at_least_f32(&format!("{path}.mechanical.friction[{i}]"), f, 0.0)?;
        }
        unit_f32(&format!("{path}.mechanical.restitution"), m.restitution)?;

        let t = &self.thermal;
        positive_f32(
            &format!("{path}.thermal.conductivity_w_mk"),
            t.conductivity_w_mk,
        )?;
        positive_f32(
            &format!("{path}.thermal.heat_capacity_j_kgk"),
            t.heat_capacity_j_kgk,
        )?;
        unit_f32(&format!("{path}.thermal.emissivity"), t.emissivity)?;

        let a = &self.acoustic;
        for (i, &x) in a.absorption.iter().enumerate() {
            unit_f32(&format!("{path}.acoustic.absorption[{i}]"), x)?;
        }
        unit_f32(&format!("{path}.acoustic.scattering"), a.scattering)?;
        unit_f32(&format!("{path}.acoustic.transmission"), a.transmission)?;
        for (i, &x) in a.absorption.iter().enumerate() {
            if x + a.transmission > 1.0 + 1e-6 {
                return Err(SceneError::invalid(
                    format!("{path}.acoustic.absorption[{i}]"),
                    "absorption + transmission must not exceed 1 (the rest is reflected)",
                ));
            }
        }

        let o = &self.optical;
        for (i, &c) in o.base_colour_linear.iter().enumerate() {
            unit_f32(&format!("{path}.optical.base_colour_linear[{i}]"), c)?;
        }
        unit_f32(&format!("{path}.optical.roughness"), o.roughness)?;
        unit_f32(&format!("{path}.optical.metallic"), o.metallic)?;
        for (i, &e) in o.emission_linear.iter().enumerate() {
            at_least_f32(&format!("{path}.optical.emission_linear[{i}]"), e, 0.0)?;
        }
        if let Some(checker) = &o.checker {
            positive_f32(
                &format!("{path}.optical.checker.cells_per_m"),
                checker.cells_per_m,
            )?;
            unit_f32(
                &format!("{path}.optical.checker.dark_multiplier"),
                checker.dark_multiplier,
            )?;
        }
        Ok(())
    }
}

/// Mechanical properties.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Mechanical {
    /// Mass density, kg/m^3. Default 1000 (MuJoCo's geom default, water).
    pub density: f32,
    /// Sliding, torsional and rolling friction coefficients (MuJoCo's order and
    /// meaning). Default `[1, 0.005, 0.0001]`, MuJoCo's geom default.
    pub friction: [f32; 3],
    /// Coefficient of restitution in `[0, 1]`. Default 0 (perfectly inelastic;
    /// MuJoCo's soft contacts have no restitution parameter of their own).
    pub restitution: f32,
}

impl Default for Mechanical {
    fn default() -> Self {
        Mechanical {
            density: 1000.0,
            friction: [1.0, 0.005, 0.0001],
            restitution: 0.0,
        }
    }
}

/// Thermal properties.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Thermal {
    /// Thermal conductivity, W/(m K). Default 0.2, a typical polymer.
    pub conductivity_w_mk: f32,
    /// Specific heat capacity, J/(kg K). Default 1200, a typical polymer.
    pub heat_capacity_j_kgk: f32,
    /// Emissivity in `[0, 1]`. Default 0.9, a matte non-metal.
    pub emissivity: f32,
}

impl Default for Thermal {
    fn default() -> Self {
        Thermal {
            conductivity_w_mk: 0.2,
            heat_capacity_j_kgk: 1200.0,
            emissivity: 0.9,
        }
    }
}

/// Acoustic properties.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Acoustic {
    /// Energy absorption coefficient per band, `[low, mid, high]`, each in
    /// `[0, 1]`. Default `[0.05, 0.05, 0.05]`, a hard, smooth surface.
    pub absorption: [f32; 3],
    /// Scattering coefficient in `[0, 1]` (the fraction of reflected energy
    /// that is scattered rather than specular). Default 0.1.
    pub scattering: f32,
    /// Transmission coefficient in `[0, 1]` (the fraction of incident energy
    /// that passes through). Default 0 (opaque to sound).
    pub transmission: f32,
}

impl Default for Acoustic {
    fn default() -> Self {
        Acoustic {
            absorption: [0.05, 0.05, 0.05],
            scattering: 0.1,
            transmission: 0.0,
        }
    }
}

/// Optical properties, with glTF 2.0 metallic-roughness semantics. Owned and
/// reviewed by Lane R.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Optical {
    /// Base colour, LINEAR (not sRGB) in `[0, 1]`, no alpha: glTF's
    /// `baseColorFactor`. Default `[0.5, 0.5, 0.5]`, a mid grey in linear light.
    pub base_colour_linear: [f32; 3],
    /// Perceptual roughness in `[0, 1]` (glTF `roughnessFactor`). Default 0.5.
    pub roughness: f32,
    /// Metallic in `[0, 1]` (glTF `metallicFactor`). Default 0 (dielectric).
    pub metallic: f32,
    /// Emitted radiance per channel, LINEAR, >= 0 (glTF `emissiveFactor`
    /// without the 0..1 cap). Default 0 (not a light).
    pub emission_linear: [f32; 3],
    /// An optional procedural checker pattern, a stand-in for textures: the
    /// surface alternates between the base colour and the base colour times
    /// `dark_multiplier`, in square cells of side `1 / cells_per_m` metres laid
    /// out in the shape's local frame. Default `None` (a plain colour).
    pub checker: Option<Checker>,
}

impl Default for Optical {
    fn default() -> Self {
        Optical {
            base_colour_linear: [0.5, 0.5, 0.5],
            roughness: 0.5,
            metallic: 0.0,
            emission_linear: [0.0; 3],
            checker: None,
        }
    }
}

/// A procedural checker pattern on the optical part.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checker {
    /// Cells per metre along each of the two surface axes, > 0.
    pub cells_per_m: f32,
    /// The factor the dark cells multiply the base colour by, in `[0, 1]`.
    pub dark_multiplier: f32,
}

/// The standard sRGB electro-optical transfer function (IEC 61966-2-1): the
/// linear-light value of an sRGB-encoded channel in `[0, 1]`.
///
/// `c <= 0.04045` gives `c / 12.92`, otherwise `((c + 0.055) / 1.055)^2.4`. The
/// MJCF importer applies it to MuJoCo's `rgba`, which it treats as authored in
/// sRGB (MuJoCo does not say; this is the documented assumption).
pub fn srgb_to_linear(c: f32) -> f32 {
    let c = f64::from(c);
    let linear = if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    };
    linear as f32
}

fn finite_f32(path: &str, value: f32) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(SceneError::invalid(path, "must be finite"))
    }
}

fn positive_f32(path: &str, value: f32) -> Result<()> {
    finite_f32(path, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(SceneError::invalid(path, "must be greater than 0"))
    }
}

fn at_least_f32(path: &str, value: f32, minimum: f32) -> Result<()> {
    finite_f32(path, value)?;
    if value >= minimum {
        Ok(())
    } else {
        Err(SceneError::invalid(
            path,
            format!("must be at least {minimum}"),
        ))
    }
}

fn unit_f32(path: &str, value: f32) -> Result<()> {
    finite_f32(path, value)?;
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(SceneError::invalid(path, "must be in [0, 1]"))
    }
}
