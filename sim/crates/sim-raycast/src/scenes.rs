//! Built-in scenes for the benchmark and the tests.
//!
//! [`tabletop`] is the shape of contract v0's phase-1 task (a simple
//! manipulation scene): a floor, a table, six objects on it and a three-part
//! hand above them. It is a STAND-IN for the physics' scene: its bodies move on the
//! analytic paths of [`Motion`] (the device renderer's pose animator, and
//! [`crate::reference::animate_poses`] on the host), not by physics.

use crate::camera::{Intrinsics, look_at};
use crate::math::{Pose, normalize, quat_axis_angle};
use crate::mesh::Mesh;
use crate::scene::{Instance, Lighting, Material, SceneDesc};

/// One body's motion: it turns about `pivot` at `omega`
/// rad/s about +z and oscillates by `amplitude` at `frequency` Hz.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Motion {
    /// The rest pose.
    pub rest: Pose,
    /// The point it turns about, world frame.
    pub pivot: [f32; 3],
    /// Turn rate about +z, rad/s.
    pub omega: f32,
    /// Oscillation amplitude, metres per axis.
    pub amplitude: [f32; 3],
    /// Oscillation frequency, Hz.
    pub frequency: f32,
}

/// The table top's upper surface, metres above the floor.
pub const TABLE_TOP_Z: f32 = 0.76;

/// The tabletop scene with its bodies' motions (one per body), its floor one
/// cell (two triangles). That is 1.28x faster at N = 1024 than the
/// baseline's 16 x 16 cells, for the same pixels' materials (levers window,
/// 2026-10-01).
pub fn tabletop() -> (SceneDesc, Vec<Motion>) {
    tabletop_with_floor(1)
}

/// [`tabletop`] with its floor plane cut into `floor_cells` x `floor_cells`
/// cells (two triangles each). The floor's colour is procedural, so the cut
/// changes no pixel's material, only the floor's BVH (a lever of the design
/// note's section 4.3; the v0 baseline used 16, the default is now 1).
pub fn tabletop_with_floor(floor_cells: u32) -> (SceneDesc, Vec<Motion>) {
    let meshes = vec![
        Mesh::plane(90.0, 90.0, floor_cells, floor_cells), // 0 floor
        Mesh::cuboid([0.6, 0.4, 0.02]),                    // 1 table top
        Mesh::cuboid([0.025, 0.025, 0.37]),                // 2 table leg
        Mesh::cuboid([0.03, 0.03, 0.03]),                  // 3 small cube
        Mesh::cuboid([0.05, 0.035, 0.025]),                // 4 block
        Mesh::sphere(0.04, 32, 16),                        // 5 ball
        Mesh::cylinder(0.03, 0.05, 48),                    // 6 can
        Mesh::sphere(0.025, 24, 12),                       // 7 small ball
        Mesh::cuboid([0.05, 0.07, 0.012]),                 // 8 palm
        Mesh::cuboid([0.008, 0.012, 0.045]),               // 9 finger
    ];
    let checker = |c: [f32; 3], r: f32, cells: f32, dark: f32| Material {
        checker_cells_per_metre: cells,
        checker_dark: dark,
        ..Material::plain(c, r)
    };
    let materials = vec![
        checker([0.55, 0.55, 0.52], 0.9, 1.0, 0.6), // 0 floor tiles
        Material::plain([0.45, 0.30, 0.18], 0.55),  // 1 wood
        Material::plain([0.20, 0.20, 0.22], 0.4),   // 2 dark legs
        Material::plain([0.80, 0.08, 0.06], 0.35),  // 3 red
        Material::plain([0.10, 0.55, 0.15], 0.5),   // 4 green
        Material::plain([0.08, 0.20, 0.75], 0.25),  // 5 blue
        Material {
            metallic: 1.0,
            ..Material::plain([0.90, 0.75, 0.30], 0.3)
        }, // 6 brass can
        Material::plain([0.55, 0.15, 0.60], 0.45),  // 7 purple
        Material::plain([0.85, 0.85, 0.82], 0.6),   // 8 white hand
    ];
    let mut instances = Vec::new();
    let mut seg = 0u16;
    let mut add = |body: Option<u32>, mesh: u32, material: u32, offset: Pose| {
        seg += 1;
        instances.push(Instance {
            body,
            mesh,
            material,
            offset,
            seg_id: seg,
        });
    };
    add(None, 0, 0, Pose::IDENTITY);
    add(None, 1, 1, Pose::at([0.0, 0.0, TABLE_TOP_Z - 0.02]));
    for (x, y) in [(0.55, 0.35), (-0.55, 0.35), (0.55, -0.35), (-0.55, -0.35)] {
        add(None, 2, 2, Pose::at([x, y, 0.37]));
    }
    // the bodies: each instance sits at its body's origin
    let bodies: [(u32, u32, [f32; 3], f32); 9] = [
        (3, 3, [0.20, 0.10, TABLE_TOP_Z + 0.03], 0.3),
        (4, 4, [-0.15, 0.18, TABLE_TOP_Z + 0.025], -0.6),
        (5, 5, [0.05, -0.20, TABLE_TOP_Z + 0.04], 0.0),
        (6, 6, [-0.25, -0.10, TABLE_TOP_Z + 0.05], 0.0),
        (7, 7, [0.32, -0.12, TABLE_TOP_Z + 0.025], 0.0),
        (3, 5, [-0.05, 0.02, TABLE_TOP_Z + 0.03], 0.8),
        (8, 8, [0.0, 0.0, TABLE_TOP_Z + 0.22], 0.0),
        (9, 8, [0.0, 0.055, TABLE_TOP_Z + 0.17], 0.0),
        (9, 8, [0.0, -0.055, TABLE_TOP_Z + 0.17], 0.0),
    ];
    let mut motions = Vec::new();
    for (b, &(mesh, material, p, yaw)) in bodies.iter().enumerate() {
        add(Some(b as u32), mesh, material, Pose::IDENTITY);
        let rest = Pose::new(p, quat_axis_angle([0.0, 0.0, 1.0], yaw));
        let hand = b >= 6;
        motions.push(Motion {
            rest,
            pivot: [0.0, 0.0, 0.0],
            // the objects circle the table's centre together; the hand sweeps
            omega: if hand { 0.0 } else { 0.35 },
            amplitude: if hand { [0.22, 0.12, 0.05] } else { [0.0; 3] },
            frequency: if hand { 0.25 } else { 0.0 },
        });
    }
    let scene = SceneDesc {
        meshes,
        materials,
        instances,
        bodies_per_env: bodies.len() as u32,
        lighting: Lighting {
            sun_direction: normalize([0.35, -0.25, 0.9]),
            sun_irradiance: [3.2, 3.0, 2.8],
            sky_zenith: [0.22, 0.32, 0.52],
            horizon_boost: 1.5,
            ground: [0.22, 0.20, 0.18],
            exposure: 1.0,
            shadows: true,
        },
    };
    (scene, motions)
}

/// The tabletop camera of environment `env`: a view over the table from one
/// corner, its eye moved by up to 5 cm per axis by an integer hash of `env`, so
/// environments differ.
pub fn tabletop_camera(env: u32, width: u32, height: u32) -> (Pose, Intrinsics) {
    let h = |k: u32| {
        let mut v = env.wrapping_mul(0x9E37_79B9) ^ k.wrapping_mul(0x85EB_CA6B);
        v ^= v >> 16;
        v = v.wrapping_mul(0x7FEB_352D);
        v ^= v >> 15;
        (v >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0
    };
    let eye = [1.05 + 0.05 * h(1), -0.85 + 0.05 * h(2), 1.40 + 0.05 * h(3)];
    let pose = look_at(eye, [0.0, 0.0, TABLE_TOP_Z + 0.02], [0.0, 0.0, 1.0]).expect("not vertical");
    (
        pose,
        Intrinsics::from_hfov(width, height, 60f32.to_radians(), 0.05, 200.0),
    )
}
