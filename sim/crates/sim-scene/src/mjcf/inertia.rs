//! Geom volume and diagonal inertia from shape and density: MuJoCo's rules.
//!
//! Ports `mjCGeom::GetVolume` (user_objects.cc:3394-3443, the volume cases) and
//! `mjCGeom::SetInertia` (3472-3690, the volume cases) of commit a8373cc4e. The
//! shell-inertia cases (`shellinertia="true"`) are not ported: the attribute is
//! refused.
//!
//! Invariants:
//! - Every expression keeps MuJoCo's textual operation order, so the results agree
//!   with the compiled model bit for bit where the platform's `f64` does.
//! - A shape with no volume rule (a plane) has volume 0 and inertia 0.
//! - For a mesh, volume and inertia come from the processed mesh: the volume at
//!   unit density, and the equivalent inertia box (`GetInertiaBoxPtr`).

use std::f64::consts::PI;

use super::mesh::ProcessedMesh;
use super::specs::GeomType;

/// `GetVolume`, volume cases.
pub(crate) fn volume(ty: GeomType, size: [f64; 3], mesh: Option<&ProcessedMesh>) -> f64 {
    match ty {
        GeomType::Mesh => mesh.map_or(0.0, |m| m.volume),
        GeomType::Sphere => {
            let radius = size[0];
            4.0 * PI * radius * radius * radius / 3.0
        }
        GeomType::Capsule => {
            let height = 2.0 * size[1];
            let radius = size[0];
            PI * (radius * radius * height + 4.0 * radius * radius * radius / 3.0)
        }
        GeomType::Cylinder => {
            let height = 2.0 * size[1];
            let radius = size[0];
            PI * radius * radius * height
        }
        GeomType::Ellipsoid => 4.0 * PI * size[0] * size[1] * size[2] / 3.0,
        GeomType::Box => size[0] * size[1] * size[2] * 8.0,
        GeomType::Plane => 0.0,
    }
}

/// `SetInertia`, volume cases: the diagonal inertia in the geom's own frame for
/// a geom of mass `mass`.
pub(crate) fn diag_inertia(
    ty: GeomType,
    size: [f64; 3],
    mass: f64,
    mesh: Option<&ProcessedMesh>,
) -> [f64; 3] {
    match ty {
        GeomType::Mesh => {
            let Some(m) = mesh else {
                return [0.0; 3];
            };
            let b = m.boxsz;
            [
                mass * (b[1] * b[1] + b[2] * b[2]) / 3.0,
                mass * (b[0] * b[0] + b[2] * b[2]) / 3.0,
                mass * (b[0] * b[0] + b[1] * b[1]) / 3.0,
            ]
        }
        GeomType::Sphere => {
            let i = 2.0 * mass * size[0] * size[0] / 5.0;
            [i, i, i]
        }
        GeomType::Capsule => {
            let height = 2.0 * size[1];
            let radius = size[0];
            // mass * (sphere_vol / total_vol)
            let sphere_mass = mass * 4.0 * radius / (4.0 * radius + 3.0 * height);
            let cylinder_mass = mass - sphere_mass;
            // cylinder part
            let mut i0 = cylinder_mass * (3.0 * radius * radius + height * height) / 12.0;
            let mut i1 = i0;
            let mut i2 = cylinder_mass * radius * radius / 2.0;
            // two hemispheres, displaced along the third axis
            let sphere_inertia = 2.0 * sphere_mass * radius * radius / 5.0;
            i0 += sphere_inertia + sphere_mass * height * (3.0 * radius + 2.0 * height) / 8.0;
            i1 += sphere_inertia + sphere_mass * height * (3.0 * radius + 2.0 * height) / 8.0;
            i2 += sphere_inertia;
            [i0, i1, i2]
        }
        GeomType::Cylinder => {
            let height = 2.0 * size[1];
            let radius = size[0];
            let i = mass * (3.0 * radius * radius + height * height) / 12.0;
            [i, i, mass * radius * radius / 2.0]
        }
        GeomType::Ellipsoid => {
            let s00 = size[0] * size[0];
            let s11 = size[1] * size[1];
            let s22 = size[2] * size[2];
            [
                mass * (s11 + s22) / 5.0,
                mass * (s00 + s22) / 5.0,
                mass * (s00 + s11) / 5.0,
            ]
        }
        GeomType::Box => {
            let s00 = size[0] * size[0];
            let s11 = size[1] * size[1];
            let s22 = size[2] * size[2];
            [
                mass * (s11 + s22) / 3.0,
                mass * (s00 + s22) / 3.0,
                mass * (s00 + s11) / 3.0,
            ]
        }
        GeomType::Plane => [0.0; 3],
    }
}
