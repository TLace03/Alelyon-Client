//! The MJCF importer: its rules, one at a time, with expected values derived
//! independently of the importer (closed forms and hand computation), and its
//! refusals with their kind, path and line.
//!
//! Parity with MuJoCo's own compile of a real model is `mjcf_parity.rs`; this file
//! covers what that model does not: orientation alternatives, `fromto`, the
//! inertia rules for explicit, derived and mixed cases, defaults and classes,
//! limits and units, actuators, tendons, materials, instances, STL meshes and
//! every refusal.

use std::f64::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};

use sim_scene::pose::{quat_conj, quat_mul, quat_rotate};
use sim_scene::{
    ActuatorKind, Cone, DEFAULT_SOLIMP, DEFAULT_SOLREF, Integrator, JointKind, MjcfError,
    MjcfErrorKind, Scene, SceneError, Shape, Solver, SolverOptions, mjcf, srgb_to_linear,
};

fn wrap(body: &str) -> String {
    format!(r#"<mujoco model="t">{body}</mujoco>"#)
}

fn no_assets() -> PathBuf {
    std::env::temp_dir()
}

fn load(xml: &str) -> Scene {
    match mjcf::load(xml, no_assets()) {
        Ok(s) => s,
        Err(e) => panic!("import failed: {e}\n{xml}"),
    }
}

fn mjcf_err(xml: &str) -> MjcfError {
    match mjcf::load(xml, no_assets()) {
        Err(SceneError::Mjcf(e)) => e,
        other => panic!("expected an MJCF refusal, got {other:?}\n{xml}"),
    }
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!(
        (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0),
        "{what}: {a} vs {b}"
    );
}

fn close_all(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        close(*x, *y, tol, what);
    }
}

fn same_rotation(a: [f64; 4], b: [f64; 4], what: &str) {
    // equal up to sign
    let d = quat_mul(quat_conj(a), b);
    assert!(
        (d[3].abs() - 1.0).abs() < 1e-12
            && d[0].abs() < 1e-12
            && d[1].abs() < 1e-12
            && d[2].abs() < 1e-12,
        "{what}: {a:?} vs {b:?}"
    );
}

const S: f64 = std::f64::consts::FRAC_1_SQRT_2;

// ---- inertia from geoms ---------------------------------------------------

#[test]
fn a_sphere_has_the_closed_form_mass_and_inertia() {
    let s = load(&wrap(
        r#"<worldbody><body name="b" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body></worldbody>"#,
    ));
    let m = 1000.0 * 4.0 / 3.0 * PI * 0.1f64.powi(3);
    let i = 0.4 * m * 0.01;
    let inertial = s.bodies[0].inertial.unwrap();
    close(inertial.mass_kg, m, 1e-12, "mass");
    close_all(&inertial.diag_inertia, &[i, i, i], 1e-12, "inertia");
    assert_eq!(inertial.com, [0.0; 3]);
    assert_eq!(inertial.inertia_quat, [0.0, 0.0, 0.0, 1.0]);
    assert_eq!(s.bodies[0].pos, [0.0, 0.0, 1.0]);
    assert_eq!(s.name, "t");
}

#[test]
fn every_primitive_has_mujocos_closed_forms() {
    // mass at the default density 1000, inertia in the geom's own axes
    let cases: Vec<(&str, f64, [f64; 3])> = vec![
        // box half-extents 1, 2, 3: volume 8*6 = 48
        (r#"type="box" size="1 2 3""#, 48000.0, {
            let m = 48000.0;
            [
                m * (4.0 + 9.0) / 3.0,
                m * (1.0 + 9.0) / 3.0,
                m * (1.0 + 4.0) / 3.0,
            ]
        }),
        // cylinder r 0.5 half-length 1: volume pi r^2 h = pi*0.25*2
        (
            r#"type="cylinder" size="0.5 1""#,
            1000.0 * PI * 0.25 * 2.0,
            {
                let m = 1000.0 * PI * 0.25 * 2.0;
                let ixy = m * (3.0 * 0.25 + 4.0) / 12.0;
                [ixy, ixy, m * 0.25 / 2.0]
            },
        ),
        // ellipsoid 1, 2, 3: volume 4/3 pi a b c
        (
            r#"type="ellipsoid" size="1 2 3""#,
            1000.0 * 4.0 / 3.0 * PI * 6.0,
            {
                let m = 1000.0 * 4.0 / 3.0 * PI * 6.0;
                [m * 13.0 / 5.0, m * 10.0 / 5.0, m * 5.0 / 5.0]
            },
        ),
    ];
    for (attrs, mass, diag) in cases {
        let s = load(&wrap(&format!(
            r#"<worldbody><body><freejoint/><geom {attrs}/></body></worldbody>"#
        )));
        let i = s.bodies[0].inertial.unwrap();
        close(i.mass_kg, mass, 1e-12, attrs);
        close_all(&i.diag_inertia, &diag, 1e-12, attrs);
    }
}

#[test]
fn a_capsule_is_a_cylinder_and_two_hemispheres() {
    // r = 0.1, half-length 0.3 (cylinder part 0.6 long)
    let (r, h) = (0.1f64, 0.6f64);
    let v_cyl = PI * r * r * h;
    let v_sph = 4.0 / 3.0 * PI * r.powi(3);
    let m = 1000.0 * (v_cyl + v_sph);
    let (m_cyl, m_sph) = (1000.0 * v_cyl, 1000.0 * v_sph);
    // axial: cylinder m r^2 / 2 + sphere 2/5 m r^2
    let i_axial = m_cyl * r * r / 2.0 + 0.4 * m_sph * r * r;
    // transverse: cylinder m (3 r^2 + h^2)/12 + sphere 2/5 m r^2 + hemispheres'
    // parallel-axis term, each hemisphere (m/2) with its centre 3r/8 + h/2 from the middle
    let d = h / 2.0 + 3.0 * r / 8.0;
    let i_trans = m_cyl * (3.0 * r * r + h * h) / 12.0
        + (0.4 * r * r - (3.0 * r / 8.0).powi(2)) * m_sph
        + m_sph * d * d;
    let s = load(&wrap(
        r#"<worldbody><body><freejoint/><geom type="capsule" size="0.1 0.3"/></body></worldbody>"#,
    ));
    let i = s.bodies[0].inertial.unwrap();
    close(i.mass_kg, m, 1e-12, "mass");
    close_all(
        &i.diag_inertia,
        &[i_trans, i_trans, i_axial],
        1e-12,
        "capsule inertia",
    );
}

#[test]
fn mass_overrides_density_and_zero_mass_removes_the_geom() {
    let s = load(&wrap(
        r#"<worldbody><body><freejoint/>
            <geom type="sphere" size="0.1" mass="2"/>
            <geom type="sphere" size="0.1" pos="1 0 0" mass="0"/>
        </body></worldbody>"#,
    ));
    let i = s.bodies[0].inertial.unwrap();
    close(i.mass_kg, 2.0, 1e-12, "mass");
    // the massless geom contributes nothing, so the body is the first sphere alone
    assert_eq!(i.com, [0.0; 3]);
    // density of the geom with an explicit mass is the implied one
    close(
        s.geoms[0].density,
        2.0 / (4.0 / 3.0 * PI * 0.001),
        1e-12,
        "implied density",
    );
    assert_eq!(s.geoms[1].density, 0.0);
}

#[test]
fn two_spheres_use_the_parallel_axis_theorem_and_sorted_principal_axes() {
    let s = load(&wrap(
        r#"<worldbody><body><freejoint/>
            <geom type="sphere" size="0.1" pos="0.5 0 0"/>
            <geom type="sphere" size="0.1" pos="-0.5 0 0"/>
        </body></worldbody>"#,
    ));
    let m1 = 1000.0 * 4.0 / 3.0 * PI * 0.001;
    let i_self = 0.4 * m1 * 0.01;
    let along = 2.0 * i_self; // about x
    let across = 2.0 * (i_self + m1 * 0.25); // about y and z
    let i = s.bodies[0].inertial.unwrap();
    close(i.mass_kg, 2.0 * m1, 1e-12, "mass");
    close_all(&i.com, &[0.0; 3], 1e-12, "com");
    // MuJoCo sorts the principal moments in decreasing order
    close_all(&i.diag_inertia, &[across, across, along], 1e-9, "inertia");
    // and the smallest one is about the x axis: the third principal axis is +-x
    let z_axis = quat_rotate(i.inertia_quat, [0.0, 0.0, 1.0]);
    close_all(
        &[z_axis[0].abs(), z_axis[1].abs(), z_axis[2].abs()],
        &[1.0, 0.0, 0.0],
        1e-9,
        "axis",
    );
}

#[test]
fn inertiafromgeom_false_true_and_auto() {
    let geom = r#"<geom type="sphere" size="0.1"/>"#;
    let explicit = r#"<inertial pos="0 0 0.1" mass="2" diaginertia="1 2 2"/>"#;
    // false: a body with only geoms has no mass
    let s = load(&wrap(&format!(
        r#"<compiler inertiafromgeom="false"/><worldbody><body><freejoint/>{geom}</body></worldbody>"#
    )));
    assert_eq!(s.bodies[0].inertial, None);
    // auto (the default): an explicit inertial wins over the geoms
    let s = load(&wrap(&format!(
        r#"<worldbody><body><freejoint/>{explicit}{geom}</body></worldbody>"#
    )));
    let i = s.bodies[0].inertial.unwrap();
    assert_eq!(
        (i.mass_kg, i.com, i.diag_inertia),
        (2.0, [0.0, 0.0, 0.1], [1.0, 2.0, 2.0])
    );
    // true: the geoms win over the explicit inertial
    let s = load(&wrap(&format!(
        r#"<compiler inertiafromgeom="true"/><worldbody><body><freejoint/>{explicit}{geom}</body></worldbody>"#
    )));
    close(
        s.bodies[0].inertial.unwrap().mass_kg,
        1000.0 * 4.0 / 3.0 * PI * 0.001,
        1e-12,
        "mass from the geom",
    );
    // true with no geom keeps the explicit inertial
    let s = load(&wrap(&format!(
        r#"<compiler inertiafromgeom="true"/><worldbody><body><freejoint/>{explicit}</body></worldbody>"#
    )));
    assert_eq!(s.bodies[0].inertial.unwrap().mass_kg, 2.0);
}

#[test]
fn inertiagrouprange_selects_geoms_by_group() {
    let s = load(&wrap(
        r#"<compiler inertiagrouprange="1 1"/><worldbody><body><freejoint/>
            <geom type="sphere" size="0.1" group="0" mass="1"/>
            <geom type="sphere" size="0.1" group="1" mass="3"/>
        </body></worldbody>"#,
    ));
    close(
        s.bodies[0].inertial.unwrap().mass_kg,
        3.0,
        1e-12,
        "only group 1",
    );
}

#[test]
fn an_explicit_inertial_with_a_full_tensor_is_diagonalised() {
    // a physical (positive, triangle-inequality) tensor with off-diagonal terms: it is
    // diagonalised and the moments sorted
    let full = [2.0, 2.5, 3.0, 0.3, 0.2, 0.1];
    let s = load(&wrap(&format!(
        r#"<worldbody><body><freejoint/><inertial pos="0 0 0" mass="1" fullinertia="{} {} {} {} {} {}"/></body></worldbody>"#,
        full[0], full[1], full[2], full[3], full[4], full[5]
    )));
    let i = s.bodies[0].inertial.unwrap();
    // principal moments are in decreasing order
    assert!(i.diag_inertia[0] >= i.diag_inertia[1] && i.diag_inertia[1] >= i.diag_inertia[2]);
    // trace is invariant
    close(i.diag_inertia.iter().sum::<f64>(), 7.5, 1e-12, "trace");
    // R diag R^T reproduces the tensor
    let axes =
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].map(|e| quat_rotate(i.inertia_quat, e));
    let mut rebuilt = [[0.0; 3]; 3];
    for (k, a) in axes.iter().enumerate() {
        for r in 0..3 {
            for c in 0..3 {
                rebuilt[r][c] += i.diag_inertia[k] * a[r] * a[c];
            }
        }
    }
    let want = [[2.0, 0.3, 0.2], [0.3, 2.5, 0.1], [0.2, 0.1, 3.0]];
    for r in 0..3 {
        for c in 0..3 {
            close(rebuilt[r][c], want[r][c], 1e-9, "tensor");
        }
    }
}

#[test]
fn an_explicit_inertial_with_a_rotated_frame_is_kept_as_written() {
    let s = load(&wrap(
        r#"<worldbody><body><freejoint/>
            <inertial pos="1 2 3" mass="4" diaginertia="3 2 1" euler="0 0 90"/></body></worldbody>"#,
    ));
    let i = s.bodies[0].inertial.unwrap();
    assert_eq!(
        (i.mass_kg, i.com, i.diag_inertia),
        (4.0, [1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    );
    same_rotation(i.inertia_quat, [0.0, 0.0, S, S], "inertial frame");
}

#[test]
fn inertia_refusals() {
    let body = |inner: &str| {
        wrap(&format!(
            r#"<worldbody><body><freejoint/>{inner}</body></worldbody>"#
        ))
    };
    // A + B < C
    let e = mjcf_err(&body(
        r#"<inertial pos="0 0 0" mass="1" diaginertia="1 1 3"/>"#,
    ));
    assert_eq!(e.kind, MjcfErrorKind::Inconsistent);
    assert!(e.message.contains("A + B >= C"), "{e}");
    // fullinertia with diaginertia, or with an orientation alternative
    let e = mjcf_err(&body(
        r#"<inertial pos="0 0 0" mass="1" diaginertia="1 1 1" fullinertia="1 1 1 0 0 0"/>"#,
    ));
    assert_eq!(e.kind, MjcfErrorKind::Inconsistent);
    let e = mjcf_err(&body(
        r#"<inertial pos="0 0 0" mass="1" euler="0 0 1" fullinertia="1 1 1 0 0 0"/>"#,
    ));
    assert_eq!(e.kind, MjcfErrorKind::Inconsistent);
    // a full tensor with no positive eigenvalues
    let e = mjcf_err(&body(
        r#"<inertial pos="0 0 0" mass="1" fullinertia="0 0 0 0 0 0"/>"#,
    ));
    assert!(e.message.contains("positive eigenvalues"), "{e}");
    // a required attribute missing
    assert_eq!(
        mjcf_err(&body(r#"<inertial pos="0 0 0" diaginertia="1 1 1"/>"#)).kind,
        MjcfErrorKind::MissingAttribute
    );
    // two inertials
    let e = mjcf_err(&body(
        r#"<inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>"#,
    ));
    assert_eq!(e.kind, MjcfErrorKind::Duplicate);
    // an inertial in the world body
    let e = mjcf_err(&wrap(
        r#"<worldbody><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/></worldbody>"#,
    ));
    assert_eq!(e.kind, MjcfErrorKind::Inconsistent);
    // a negative geom mass is clamped by boundmass = 0 as in MuJoCo, but a negative density is an error
    let e = mjcf_err(&body(r#"<geom type="sphere" size="0.1" density="-1"/>"#));
    assert!(e.message.contains("negative"), "{e}");
}

// ---- orientations ------------------------------------------------------------

fn body_quat(attrs: &str, compiler: &str) -> [f64; 4] {
    let s = load(&wrap(&format!(
        r#"{compiler}<worldbody><body {attrs}><geom type="sphere" size="0.1"/></body></worldbody>"#
    )));
    s.bodies[0].quat
}

#[test]
fn orientation_alternatives_resolve_as_mujoco_does() {
    // euler, degrees (default): 90 about z
    same_rotation(
        body_quat(r#"euler="0 0 90""#, ""),
        [0.0, 0.0, S, S],
        "euler deg",
    );
    // radians
    same_rotation(
        body_quat(
            r#"euler="0 0 1.5707963267948966""#,
            r#"<compiler angle="radian"/>"#,
        ),
        [0.0, 0.0, S, S],
        "euler rad",
    );
    // eulerseq: the sequence names the axis of each angle in order
    same_rotation(
        body_quat(r#"euler="90 0 0""#, r#"<compiler eulerseq="zyx"/>"#),
        [0.0, 0.0, S, S],
        "eulerseq zyx",
    );
    // intrinsic xyz: x by 90 then y by 90 about the new axes = R_x R_y
    let rx = [S, 0.0, 0.0, S];
    let ry = [0.0, S, 0.0, S];
    same_rotation(
        body_quat(r#"euler="90 90 0""#, ""),
        quat_mul(rx, ry),
        "euler xyz moving axes",
    );
    // fixed-axes sequence (capitals): R_y R_x
    same_rotation(
        body_quat(r#"euler="90 90 0""#, r#"<compiler eulerseq="XYZ"/>"#),
        quat_mul(ry, rx),
        "euler XYZ fixed axes",
    );
    // axisangle
    same_rotation(
        body_quat(r#"axisangle="0 0 2 90""#, ""),
        [0.0, 0.0, S, S],
        "axisangle",
    );
    // xyaxes: x -> world y, y -> world -x is a 90 degree turn about z
    same_rotation(
        body_quat(r#"xyaxes="0 1 0 -1 0 0""#, ""),
        [0.0, 0.0, S, S],
        "xyaxes",
    );
    // xyaxes with a y axis that is not orthogonal is orthogonalised
    same_rotation(
        body_quat(r#"xyaxes="0 1 0 -1 1 0""#, ""),
        [0.0, 0.0, S, S],
        "xyaxes gram-schmidt",
    );
    // zaxis: +z onto +x is a 90 degree turn about +y
    same_rotation(body_quat(r#"zaxis="3 0 0""#, ""), [0.0, S, 0.0, S], "zaxis");
    // quat is normalised. NB the MJCF attribute is MuJoCo's [w, x, y, z]; the scene's is [x, y, z, w]
    same_rotation(
        body_quat(r#"quat="2 0 0 0""#, ""),
        [0.0, 0.0, 0.0, 1.0],
        "quat normalised",
    );
    same_rotation(
        body_quat(r#"quat="3 0 0 3""#, ""),
        [0.0, 0.0, S, S],
        "quat wxyz -> xyzw",
    );
    same_rotation(
        body_quat(r#"quat="0 0 3 3""#, ""),
        [0.0, S, S, 0.0],
        "quat wxyz -> xyzw (y, z)",
    );
}

#[test]
fn orientation_refusals() {
    let body = |attrs: &str| {
        wrap(&format!(
            r#"<worldbody><body {attrs}><geom size="1"/></body></worldbody>"#
        ))
    };
    for attrs in [
        r#"quat="1 0 0 0" euler="0 0 1""#,
        r#"euler="0 0 1" axisangle="0 0 1 1""#,
        r#"zaxis="0 0 1" xyaxes="1 0 0 0 1 0""#,
    ] {
        let e = mjcf_err(&body(attrs));
        assert_eq!(e.kind, MjcfErrorKind::BadValue, "{attrs}");
        assert!(e.message.contains("multiple orientation"), "{e}");
    }
    assert!(
        mjcf_err(&body(r#"quat="0 0 0 0""#))
            .message
            .contains("zero quaternion")
    );
    assert!(
        mjcf_err(&body(r#"axisangle="0 0 0 90""#))
            .message
            .contains("axisangle too small")
    );
    assert!(
        mjcf_err(&body(r#"zaxis="0 0 0""#))
            .message
            .contains("zaxis too small")
    );
    assert!(
        mjcf_err(&body(r#"xyaxes="0 0 0 0 1 0""#))
            .message
            .contains("xaxis too small")
    );
    assert!(
        mjcf_err(&body(r#"xyaxes="1 0 0 2 0 0""#))
            .message
            .contains("yaxis too small")
    );
    assert!(
        mjcf_err(&body(r#"quat="1 0 0""#))
            .message
            .contains("not have enough data")
    );
    assert!(
        mjcf_err(&body(r#"pos="1 2 3 4""#))
            .message
            .contains("too much data")
    );
    assert!(
        mjcf_err(&body(r#"pos="1 two 3""#))
            .message
            .contains("not a number")
    );
    assert!(
        mjcf_err(&body(r#"pos="1 nan 3""#))
            .message
            .contains("not finite")
    );
    let e = mjcf_err(&wrap(r#"<compiler eulerseq="xyq"/><worldbody/>"#));
    assert!(e.message.contains("eulerseq"), "{e}");
}

// ---- geoms -------------------------------------------------------------------

#[test]
fn fromto_places_and_sizes_a_capsule() {
    let s = load(&wrap(
        r#"<worldbody><body><freejoint/>
            <geom type="capsule" fromto="1 2 3 4 6 3" size="0.05"/>
            <geom type="box" fromto="0 0 0 0 0 -2" size="0.1"/>
            <geom type="ellipsoid" fromto="0 0 0 2 0 0" size="0.3"/>
            <geom type="cylinder" fromto="0 0 0 0 0 0.5" size="0.2"/>
        </body></worldbody>"#,
    ));
    // (1,2,3) -> (4,6,3): length 5; MuJoCo points the geom's z axis along from - to
    let g = &s.geoms[0];
    assert_eq!(
        g.shape,
        Shape::Capsule {
            r: 0.05,
            half_len: 2.5
        }
    );
    close_all(&g.pos, &[2.5, 4.0, 3.0], 1e-12, "midpoint");
    let axis = quat_rotate(g.quat, [0.0, 0.0, 1.0]);
    close_all(&axis, &[-0.6, -0.8, 0.0], 1e-12, "z axis along from - to");
    // a box and an ellipsoid with fromto: half-extents (size, size, length/2)
    assert_eq!(
        s.geoms[1].shape,
        Shape::Box {
            half: [0.1, 0.1, 1.0]
        }
    );
    assert_eq!(
        s.geoms[2].shape,
        Shape::Ellipsoid {
            radii: [0.3, 0.3, 1.0]
        }
    );
    close_all(&s.geoms[1].pos, &[0.0, 0.0, -1.0], 1e-12, "box midpoint");
    // (0,0,0) -> (0,0,-2): from - to = (0, 0, 2), so z stays +z
    let up = quat_rotate(s.geoms[1].quat, [0.0, 0.0, 1.0]);
    close_all(&up, &[0.0, 0.0, 1.0], 1e-12, "box axis");
    assert_eq!(
        s.geoms[3].shape,
        Shape::Cylinder {
            r: 0.2,
            half_len: 0.25
        }
    );
}

#[test]
fn fromto_refusals() {
    let geom = |attrs: &str| {
        wrap(&format!(
            r#"<worldbody><body><freejoint/><geom {attrs}/></body></worldbody>"#
        ))
    };
    let e = mjcf_err(&geom(
        r#"type="capsule" fromto="0 0 0 0 0 1" pos="1 0 0" size="0.1""#,
    ));
    assert!(e.message.contains("both pos and fromto"), "{e}");
    let e = mjcf_err(&geom(r#"type="sphere" fromto="0 0 0 0 0 1" size="0.1""#));
    assert!(e.message.contains("fromto requires"), "{e}");
    let e = mjcf_err(&geom(r#"type="capsule" fromto="0 0 0 0 0 0" size="0.1""#));
    assert!(e.message.contains("too close"), "{e}");
    let e = mjcf_err(&geom(r#"type="sphere" size="0""#));
    assert!(e.message.contains("size 0 must be positive"), "{e}");
    let e = mjcf_err(&geom(r#"type="capsule" size="0.1""#));
    assert!(e.message.contains("size 1 must be positive"), "{e}");
    let e = mjcf_err(&geom(r#"type="sphere" size="0.1" condim="2""#));
    assert!(e.message.contains("condim"), "{e}");
    let e = mjcf_err(&geom(r#"type="hfield" size="1 1 1 1""#));
    assert_eq!(e.kind, MjcfErrorKind::UnsupportedAttribute);
    let e = mjcf_err(&geom(r#"type="mesh""#));
    assert_eq!(e.kind, MjcfErrorKind::MissingAttribute);
    let e = mjcf_err(&geom(r#"type="mesh" mesh="nope""#));
    assert_eq!(e.kind, MjcfErrorKind::UnknownReference);
    let e = mjcf_err(&geom(r#"type="sphere" size="1" contype="-1""#));
    assert_eq!(e.kind, MjcfErrorKind::BadValue);
}

#[test]
fn geom_attributes_pass_through() {
    let s = load(&wrap(
        r#"<worldbody><geom name="p" type="plane" size="0 0 0.05" contype="2" conaffinity="4" condim="6" friction="0.7"/></worldbody>"#,
    ));
    let g = &s.geoms[0];
    assert_eq!(
        g.shape,
        Shape::Plane {
            size: [0.0, 0.0, 0.05]
        }
    );
    assert_eq!((g.contype, g.conaffinity, g.condim), (2, 4, 6));
    // a one-number friction sets the sliding coefficient and keeps MuJoCo's other defaults
    assert_eq!(g.friction, [0.7, 0.005, 0.0001]);
    assert_eq!(g.body, None, "a world geom belongs to the world");
}

// ---- classes -----------------------------------------------------------------

#[test]
fn default_classes_inherit_nest_and_override() {
    let s = load(&wrap(
        r#"<default>
             <geom type="box" size="1 2 3" density="500"/>
             <default class="a">
               <default class="b"><geom density="100"/></default>
               <geom size="0.5 0.5 0.5"/>
             </default>
           </default>
           <worldbody>
             <body name="x" childclass="a"><freejoint/>
               <geom name="x_geom"/>
               <body name="y" childclass="b">
                 <joint type="slide"/>
                 <geom name="y_geom"/>
                 <geom name="y_over" class="a"/>
               </body>
             </body>
             <body name="z"><geom name="z_geom"/></body>
           </worldbody>"#,
    ));
    let by = |n: &str| s.geoms.iter().find(|g| g.name == n).unwrap();
    // class a: box 0.5, density 500 (inherited from main)
    assert_eq!(by("x_geom").shape, Shape::Box { half: [0.5; 3] });
    assert_eq!(by("x_geom").density, 500.0);
    // class b nests in a, declared BEFORE a's own geom, and still sees a's size
    // (a's elements are all read before its nested classes): box 0.5, density 100
    assert_eq!(by("y_geom").shape, Shape::Box { half: [0.5; 3] });
    assert_eq!(by("y_geom").density, 100.0);
    // a class attribute overrides childclass
    assert_eq!(by("y_over").density, 500.0);
    // no childclass: the main class
    assert_eq!(
        by("z_geom").shape,
        Shape::Box {
            half: [1.0, 2.0, 3.0]
        }
    );
    assert_eq!(by("z_geom").density, 500.0);
}

#[test]
fn childclass_is_inherited_by_descendants_until_overridden() {
    let s = load(&wrap(
        r#"<default>
             <default class="c1"><geom size="0.1" density="1"/></default>
             <default class="c2"><geom size="0.2" density="2"/></default>
           </default>
           <worldbody>
             <body childclass="c1"><freejoint/><geom name="g1"/>
               <body><joint type="slide"/><geom name="g2"/>
                 <body childclass="c2"><joint type="slide"/><geom name="g3"/>
                   <body><joint type="slide"/><geom name="g4"/></body>
                 </body>
               </body>
             </body>
           </worldbody>"#,
    ));
    let r = |n: &str| match s.geoms.iter().find(|g| g.name == n).unwrap().shape {
        Shape::Sphere { r } => r,
        other => panic!("{other:?}"),
    };
    assert_eq!((r("g1"), r("g2"), r("g3"), r("g4")), (0.1, 0.1, 0.2, 0.2));
}

#[test]
fn a_class_that_sets_an_orientation_alternative_wins_over_an_elements_quat() {
    // MuJoCo's ReadAlternative leaves the orientation kind alone when `quat` is read,
    // so the class's euler still applies. (Ported behaviour, see specs.rs.)
    let s = load(&wrap(
        r#"<default><geom euler="0 0 90"/></default>
           <worldbody><body><freejoint/><geom type="sphere" size="0.1" quat="0 1 0 0"/></body></worldbody>"#,
    ));
    same_rotation(
        s.geoms[0].quat,
        [0.0, 0.0, S, S],
        "class euler beats element quat",
    );
}

#[test]
fn class_refusals() {
    let e = mjcf_err(&wrap(
        r#"<worldbody><body childclass="nope"><geom size="1"/></body></worldbody>"#,
    ));
    assert_eq!(e.kind, MjcfErrorKind::UnknownReference);
    let e = mjcf_err(&wrap(
        r#"<worldbody><body><geom class="nope" size="1"/></body></worldbody>"#,
    ));
    assert_eq!(e.kind, MjcfErrorKind::UnknownReference);
    let e = mjcf_err(&wrap(
        r#"<default><default class="a"/><default class="a"/></default>"#,
    ));
    assert_eq!(e.kind, MjcfErrorKind::Duplicate);
    let e = mjcf_err(&wrap(r#"<default class="renamed"/>"#));
    assert!(e.message.contains("main"), "{e}");
    let e = mjcf_err(&wrap(r#"<default><default/></default>"#));
    assert!(e.message.contains("empty class name"), "{e}");
    // name and class are not allowed inside a default element
    let e = mjcf_err(&wrap(r#"<default><geom name="g"/></default>"#));
    assert_eq!(e.kind, MjcfErrorKind::UnsupportedAttribute);
    // other actuator kinds in a default would feed the shared actuator state
    let e = mjcf_err(&wrap(r#"<default><general gear="2"/></default>"#));
    assert_eq!(e.kind, MjcfErrorKind::UnsupportedElement);
}

// ---- joints ------------------------------------------------------------------

#[test]
fn joint_units_limits_and_axes() {
    let s = load(&wrap(
        r#"<worldbody><body name="a"><freejoint/><geom size="0.1"/>
             <body name="b" pos="0 0 1">
               <joint name="h" type="hinge" axis="0 3 4" range="-90 45" stiffness="2" damping="0.5" armature="0.01" frictionloss="0.2" pos="0.1 0 0"/>
               <joint name="s" type="slide" axis="1 0 0" range="-0.5 0.25"/>
               <joint name="free_range" type="hinge" range="0 0" limited="false"/>
               <geom size="0.1"/>
             </body>
           </body></worldbody>"#,
    ));
    let h = &s.joints[1];
    assert_eq!(h.name, "h");
    // angular limits in radians; the axis is normalised (0, 3, 4) / 5
    let JointKind::Hinge { axis } = h.kind else {
        panic!()
    };
    close_all(&axis, &[0.0, 0.6, 0.8], 1e-12, "axis");
    let r = h.range.unwrap();
    close_all(&r, &[-PI / 2.0, PI / 4.0], 1e-12, "hinge range");
    assert_eq!(
        (h.stiffness, h.damping, h.armature, h.frictionloss),
        (2.0, 0.5, 0.01, 0.2)
    );
    assert_eq!(h.pos, [0.1, 0.0, 0.0]);
    // slide limits are metres: no degree conversion
    assert_eq!(s.joints[2].range, Some([-0.5, 0.25]));
    // limited="false" removes the range
    assert_eq!(s.joints[3].range, None);
}

#[test]
fn radians_when_the_compiler_says_so_and_ball_limits() {
    let s = load(&wrap(
        r#"<compiler angle="radian"/><worldbody><body><joint type="ball" range="0 0.5"/><geom size="0.1"/>
             <body><joint type="hinge" range="-1 1"/><geom size="0.1"/></body></body></worldbody>"#,
    ));
    assert_eq!(s.joints[0].range, Some([0.0, 0.5]));
    assert_eq!(s.joints[1].range, Some([-1.0, 1.0]));
    // degrees apply to a ball joint too
    let s = load(&wrap(
        r#"<worldbody><body><joint type="ball" range="0 90"/><geom size="0.1"/></body></worldbody>"#,
    ));
    close(s.joints[0].range.unwrap()[1], PI / 2.0, 1e-12, "ball range");
}

#[test]
fn joint_refusals() {
    let j = |compiler: &str, attrs: &str| {
        wrap(&format!(
            r#"{compiler}<worldbody><body><geom size="0.1"/><joint {attrs}/></body></worldbody>"#
        ))
    };
    // a range without `limited` needs autolimits
    let e = mjcf_err(&j(r#"<compiler autolimits="false"/>"#, r#"range="-1 1""#));
    assert_eq!(e.kind, MjcfErrorKind::Inconsistent);
    assert!(e.message.contains("autolimits"), "{e}");
    // ... but is fine with an explicit limited
    load(&j(
        r#"<compiler autolimits="false"/>"#,
        r#"range="-1 1" limited="true""#,
    ));
    // limited with no usable range
    assert!(
        mjcf_err(&j("", r#"limited="true""#))
            .message
            .contains("range[0] should be smaller")
    );
    assert!(
        mjcf_err(&j("", r#"limited="true" range="1 -1""#))
            .message
            .contains("range[0] should be smaller")
    );
    // an inverted range under auto is simply not a limit (MuJoCo's rule)
    assert_eq!(load(&j("", r#"range="1 -1""#)).joints[0].range, None);
    assert!(
        mjcf_err(&j("", r#"type="ball" range="10 20""#))
            .message
            .contains("range[0] should be 0")
    );
    assert!(
        mjcf_err(&j("", r#"axis="0 0 0""#))
            .message
            .contains("axis too small")
    );
    // polynomial coefficients beyond the constant are not modelled
    assert_eq!(
        mjcf_err(&j("", r#"stiffness="1 2""#)).kind,
        MjcfErrorKind::UnsupportedAttribute
    );
    load(&j("", r#"stiffness="1 0 0""#));
    for attr in [
        "ref=\"1\"",
        "springref=\"1\"",
        "springdamper=\"1 1\"",
        "actuatorfrcrange=\"-1 1\"",
        "actuatorgravcomp=\"true\"",
    ] {
        assert_eq!(
            mjcf_err(&j("", attr)).kind,
            MjcfErrorKind::UnsupportedAttribute,
            "{attr}"
        );
    }
    // free joints: top level only
    let e = mjcf_err(&wrap(
        r#"<worldbody><body><freejoint/><geom size="0.1"/><body><freejoint/><geom size="0.1"/></body></body></worldbody>"#,
    ));
    assert!(e.message.contains("top level"), "{e}");
    let e = mjcf_err(&wrap(r#"<worldbody><joint/></worldbody>"#));
    assert!(e.message.contains("world body cannot have joints"), "{e}");
    let e = mjcf_err(&wrap(
        r#"<worldbody><body><joint name="a"/><joint name="a"/></body></worldbody>"#,
    ));
    assert_eq!(e.kind, MjcfErrorKind::Duplicate);
}

#[test]
fn bodies_and_geoms_and_joints_are_listed_in_body_order_like_mujoco() {
    // B is defined (as a child of A) BEFORE A's own joint and geom in the XML;
    // MuJoCo lists joints and geoms per body in body order, so A's come first.
    let s = load(&wrap(
        r#"<worldbody>
             <geom name="floor" type="plane" size="0 0 0.1"/>
             <body name="A">
               <body name="B" pos="0 0 1"><joint name="jb" type="slide"/><geom name="gb" size="0.1"/></body>
               <joint name="ja" type="slide"/><geom name="ga" size="0.1"/>
             </body>
             <geom name="late_floor" type="plane" size="0 0 0.1" pos="0 0 -1"/>
           </worldbody>"#,
    ));
    let names = |it: Vec<&str>| it.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    assert_eq!(
        names(s.bodies.iter().map(|b| b.name.as_str()).collect()),
        ["A", "B"]
    );
    assert_eq!(
        names(s.joints.iter().map(|j| j.name.as_str()).collect()),
        ["ja", "jb"]
    );
    // world geoms first (both of them), then A's, then B's
    assert_eq!(
        names(s.geoms.iter().map(|g| g.name.as_str()).collect()),
        ["floor", "late_floor", "ga", "gb"]
    );
}

// ---- soft-constraint parameters and solver options ---------------------------

#[test]
fn soft_constraint_parameters_default_to_mujocos_values() {
    let s = load(&wrap(&format!(
        r#"{TWO_JOINTS}<tendon><fixed name="t"><joint joint="j1" coef="1"/></fixed></tendon>"#
    )));
    // MuJoCo's own defaults, as the oracle compiles them (tools/sim_scene_mujoco_golden.py)
    assert_eq!(DEFAULT_SOLREF, [0.02, 1.0]);
    assert_eq!(DEFAULT_SOLIMP, [0.9, 0.95, 0.001, 0.5, 2.0]);
    for j in &s.joints {
        assert_eq!(j.solref_limit, DEFAULT_SOLREF, "{}", j.name);
        assert_eq!(j.solimp_limit, DEFAULT_SOLIMP, "{}", j.name);
        assert_eq!(j.solref_friction, DEFAULT_SOLREF, "{}", j.name);
        assert_eq!(j.solimp_friction, DEFAULT_SOLIMP, "{}", j.name);
        assert_eq!(j.margin, 0.0, "{}", j.name);
    }
    let t = &s.tendons[0];
    assert_eq!(t.solref_limit, DEFAULT_SOLREF);
    assert_eq!(t.solimp_limit, DEFAULT_SOLIMP);
    assert_eq!(t.solref_friction, DEFAULT_SOLREF);
    assert_eq!(t.solimp_friction, DEFAULT_SOLIMP);
    assert_eq!(t.margin, 0.0);
    assert_eq!(s.options, SolverOptions::default());
}

#[test]
fn soft_constraint_parameters_are_read_per_joint_and_per_tendon() {
    let s = load(&wrap(
        r#"<worldbody><body><geom size="0.1"/>
             <joint name="h" type="hinge" range="-30 40" margin="5" solreflimit="0.05 0.8"
                    solimplimit="0.8 0.9 0.01 0.4 3" solreffriction="0.03 0.7"
                    solimpfriction="0.7 0.8 0.002 0.3 3" frictionloss="0.5"/>
             <body><geom size="0.1"/><joint name="b" type="ball" range="0 45" margin="3"/>
             <body><geom size="0.1"/><joint name="d" type="slide" range="-0.5 0.5" margin="0.1" solreflimit="-300 -4"/>
             </body></body></body></worldbody>
           <tendon><fixed name="t" limited="true" range="-1 2" margin="0.2" frictionloss="0.3"
                    solreflimit="0.04 0.9" solimplimit="0.5 0.6 0.02 0.5 1"
                    solreffriction="0.07 0.6" solimpfriction="0.2 0.3 0.5 0.7 4">
             <joint joint="h" coef="1"/><joint joint="d" coef="-2"/></fixed></tendon>"#,
    ));
    let h = &s.joints[0];
    assert_eq!(h.solref_limit, [0.05, 0.8]);
    assert_eq!(h.solimp_limit, [0.8, 0.9, 0.01, 0.4, 3.0]);
    assert_eq!(h.solref_friction, [0.03, 0.7]);
    assert_eq!(h.solimp_friction, [0.7, 0.8, 0.002, 0.3, 3.0]);
    // the margin is a number in the joint's own unit: MuJoCo does not turn it from
    // degrees into radians (the oracle compiles margin="5" on a hinge to 5)
    assert_eq!(h.margin, 5.0);
    close_all(
        &h.range.unwrap(),
        &[-PI / 6.0, 2.0 * PI / 9.0],
        1e-12,
        "hinge range",
    );
    // a ball joint's range is [0, max_angle], in radians
    let b = &s.joints[1];
    assert_eq!(b.margin, 3.0);
    close_all(&b.range.unwrap(), &[0.0, PI / 4.0], 1e-12, "ball range");
    assert_eq!(b.solref_limit, DEFAULT_SOLREF);
    // MuJoCo's "direct" format: negative stiffness and damping
    assert_eq!(s.joints[2].solref_limit, [-300.0, -4.0]);
    assert_eq!(s.joints[2].margin, 0.1);
    let t = &s.tendons[0];
    assert_eq!(t.solref_limit, [0.04, 0.9]);
    assert_eq!(t.solimp_limit, [0.5, 0.6, 0.02, 0.5, 1.0]);
    assert_eq!(t.solref_friction, [0.07, 0.6]);
    assert_eq!(t.solimp_friction, [0.2, 0.3, 0.5, 0.7, 4.0]);
    assert_eq!((t.margin, t.frictionloss), (0.2, 0.3));
    assert_eq!(t.range, Some([-1.0, 2.0]));
    // none of the parameters is recorded as unsupported any more
    for u in &s.unsupported {
        assert!(
            ![
                "@solreflimit",
                "@solimplimit",
                "@solreffriction",
                "@solimpfriction",
                "@margin"
            ]
            .contains(&u.item.as_str())
                || u.path.contains("geom"),
            "{u:?}"
        );
    }
}

#[test]
fn shorter_solref_and_solimp_lists_keep_the_rest_from_the_class() {
    // MuJoCo reads `solref` and `solimp` as lists of up to 2 and 5 numbers; the
    // entries that are not given keep the value of the element's default class
    let s = load(&wrap(
        r#"<default><joint solreflimit="0.04 0.9" solimplimit="0.7 0.8 0.003 0.4 3"/></default>
           <worldbody><body><geom size="0.1"/>
             <joint name="a" solreflimit="0.06" solimplimit="0.6 0.65"/>
             <joint name="b" axis="1 0 0"/>
             <joint name="c" axis="0 1 0" solimplimit="0.5"/>
           </body></worldbody>"#,
    ));
    assert_eq!(s.joints[0].solref_limit, [0.06, 0.9]);
    assert_eq!(s.joints[0].solimp_limit, [0.6, 0.65, 0.003, 0.4, 3.0]);
    assert_eq!(s.joints[1].solref_limit, [0.04, 0.9]);
    assert_eq!(s.joints[1].solimp_limit, [0.7, 0.8, 0.003, 0.4, 3.0]);
    assert_eq!(s.joints[2].solimp_limit, [0.5, 0.8, 0.003, 0.4, 3.0]);
}

#[test]
fn soft_constraint_parameters_follow_default_classes() {
    let s = load(&wrap(
        r#"<default>
             <joint solreffriction="0.03 0.5" margin="0.2"/>
             <tendon solreflimit="0.05 1" margin="0.3"/>
             <default class="stiff"><joint solreffriction="0.01 1"/><tendon margin="0.4"/></default>
           </default>
           <worldbody><body childclass="stiff"><geom size="0.1"/>
             <joint name="j1" type="hinge"/><joint name="j2" type="slide" class="main" margin="0.7"/>
             <body><geom size="0.1"/><joint name="ball" type="ball"/></body>
           </body></worldbody>
           <tendon><fixed name="t1" class="stiff"><joint joint="j1" coef="1"/></fixed>
                   <fixed name="t2"><joint joint="j1" coef="1"/></fixed></tendon>"#,
    ));
    // j1 inherits the "stiff" class through `childclass`; j2 names the main class
    assert_eq!(
        (s.joints[0].solref_friction, s.joints[0].margin),
        ([0.01, 1.0], 0.2)
    );
    assert_eq!(
        (s.joints[1].solref_friction, s.joints[1].margin),
        ([0.03, 0.5], 0.7)
    );
    assert_eq!(
        (s.joints[2].solref_friction, s.joints[2].margin),
        ([0.01, 1.0], 0.2)
    );
    assert_eq!(
        (s.tendons[0].solref_limit, s.tendons[0].margin),
        ([0.05, 1.0], 0.4)
    );
    assert_eq!(
        (s.tendons[1].solref_limit, s.tendons[1].margin),
        ([0.05, 1.0], 0.3)
    );
}

#[test]
fn soft_constraint_refusals() {
    let j = |attrs: &str| {
        wrap(&format!(
            r#"<worldbody><body><geom size="0.1"/><joint {attrs}/></body></worldbody>"#
        ))
    };
    // too many numbers for the attribute
    assert_eq!(
        mjcf_err(&j(r#"solreflimit="1 2 3""#)).kind,
        MjcfErrorKind::BadValue
    );
    assert_eq!(
        mjcf_err(&j(r#"solimpfriction="1 2 3 4 5 6""#)).kind,
        MjcfErrorKind::BadValue
    );
    assert_eq!(
        mjcf_err(&j(r#"margin="abc""#)).kind,
        MjcfErrorKind::BadValue
    );
    // a mixed solref (one positive, one not) is what MuJoCo replaces with the default
    // and warns about; the scene refuses it
    match mjcf::load(&j(r#"solreflimit="0.02 -1""#), no_assets()) {
        Err(SceneError::Invalid { path, .. }) => assert!(path.contains("solref_limit"), "{path}"),
        other => panic!("{other:?}"),
    }
    // a negative margin, an impedance above 1 and a power below 1 are refused
    for (attrs, field) in [
        (r#"margin="-0.1""#, "margin"),
        (r#"solimpfriction="0.9 1.5""#, "solimp_friction"),
        (r#"solimplimit="0.9 0.95 0.001 0.5 0.5""#, "solimp_limit"),
        (r#"solimplimit="0.9 0.95 -1""#, "solimp_limit"),
    ] {
        match mjcf::load(&j(attrs), no_assets()) {
            Err(SceneError::Invalid { path, .. }) => {
                assert!(path.contains(field), "{attrs}: {path}")
            }
            other => panic!("{attrs}: {other:?}"),
        }
    }
    // the same on a tendon
    let t = |attrs: &str| {
        wrap(&format!(
            r#"{TWO_JOINTS}<tendon><fixed {attrs}><joint joint="j1" coef="1"/></fixed></tendon>"#
        ))
    };
    assert_eq!(
        mjcf_err(&t(r#"solreflimit="1 2 3""#)).kind,
        MjcfErrorKind::BadValue
    );
    assert!(matches!(
        mjcf::load(&t(r#"solreffriction="-1 0.5""#), no_assets()),
        Err(SceneError::Invalid { .. })
    ));
}

#[test]
fn solver_options_are_read_with_mujocos_defaults() {
    // MuJoCo 3.14.0's own defaults (oracle: m.opt after compiling an empty <option/>)
    let d = SolverOptions::default();
    assert_eq!(
        (d.solver, d.iterations, d.tolerance, d.ls_iterations),
        (Solver::Newton, 100, 1e-8, 50)
    );
    assert_eq!(
        (d.ls_tolerance, d.cone, d.impratio),
        (0.01, Cone::Pyramidal, 1.0)
    );
    assert_eq!(load(&wrap(r#"<worldbody/>"#)).options, d);
    assert_eq!(load(&wrap(r#"<option/><worldbody/>"#)).options, d);

    let s = load(&wrap(
        r#"<option solver="CG" iterations="7" tolerance="1e-6" ls_iterations="9" ls_tolerance="0.2" cone="elliptic" impratio="3"/><worldbody/>"#,
    ));
    assert_eq!(
        s.options,
        SolverOptions {
            solver: Solver::Cg,
            iterations: 7,
            tolerance: 1e-6,
            ls_iterations: 9,
            ls_tolerance: 0.2,
            cone: Cone::Elliptic,
            impratio: 3.0,
        }
    );
    for (text, expect) in [
        ("PGS", Solver::Pgs),
        ("CG", Solver::Cg),
        ("Newton", Solver::Newton),
    ] {
        let s = load(&wrap(&format!(r#"<option solver="{text}"/><worldbody/>"#)));
        assert_eq!(s.options.solver, expect, "{text}");
    }
    // two <option> elements accumulate, a later attribute wins
    let s = load(&wrap(
        r#"<option iterations="5" tolerance="1e-3"/><option iterations="6"/><worldbody/>"#,
    ));
    assert_eq!((s.options.iterations, s.options.tolerance), (6, 1e-3));
    // a tolerance of 0 is legal (it never stops early), as are 0 iterations
    let s = load(&wrap(
        r#"<option tolerance="0" iterations="0" ls_iterations="0"/><worldbody/>"#,
    ));
    assert_eq!(
        (
            s.options.tolerance,
            s.options.iterations,
            s.options.ls_iterations
        ),
        (0.0, 0, 0)
    );
}

#[test]
fn solver_option_refusals() {
    // MuJoCo's keywords are case sensitive: "newton" is refused as it refuses it
    for attr in [
        r#"solver="newton""#,
        r#"cone="Elliptic""#,
        r#"iterations="-1""#,
        r#"ls_iterations="-3""#,
        r#"tolerance="-1e-8""#,
        r#"ls_tolerance="-1""#,
        r#"impratio="0""#,
        r#"iterations="2.5""#,
    ] {
        assert_eq!(
            mjcf_err(&wrap(&format!("<option {attr}/>"))).kind,
            MjcfErrorKind::BadValue,
            "{attr}"
        );
    }
    // a <flag> stays refused: it would switch parts of the physics on and off
    assert_eq!(
        mjcf_err(&wrap(r#"<option><flag constraint="disable"/></option>"#)).kind,
        MjcfErrorKind::UnsupportedElement
    );
}

// ---- actuators and tendons ---------------------------------------------------

const TWO_JOINTS: &str = r#"<worldbody><body><geom size="0.1"/>
    <joint name="j1" type="hinge"/><joint name="j2" type="slide"/>
    <body><geom size="0.1"/><joint name="ball" type="ball"/></body>
  </body></worldbody>"#;

#[test]
fn motors_and_position_actuators() {
    let s = load(&wrap(&format!(
        r#"<default><motor ctrlrange="-2 2" ctrllimited="true" gear="3"/><position kp="10"/></default>
           {TWO_JOINTS}
           <actuator>
             <motor name="m" joint="j1"/>
             <motor name="m2" joint="j2" gear="5" ctrllimited="false"/>
             <motor name="m3" joint="j1" ctrlrange="-1 1" ctrllimited="auto"/>
             <position name="p" joint="j1"/>
             <position name="p2" joint="j2" kp="7" ctrlrange="0 1"/>
           </actuator>"#
    )));
    let k = |i: usize| s.actuators[i].kind;
    assert_eq!(
        k(0),
        ActuatorKind::Motor {
            gear: 3.0,
            ctrlrange: Some([-2.0, 2.0])
        }
    );
    assert_eq!(
        k(1),
        ActuatorKind::Motor {
            gear: 5.0,
            ctrlrange: None
        }
    );
    // auto limits: a ctrlrange is a limit
    assert_eq!(
        k(2),
        ActuatorKind::Motor {
            gear: 3.0,
            ctrlrange: Some([-1.0, 1.0])
        }
    );
    // the position default's kp=10 applies to a position actuator; the default's
    // motor settings are shared, gear included (MuJoCo keeps one actuator default
    // per class, so gear=3 reaches <position> as it does in MuJoCo)
    assert_eq!(
        k(3),
        ActuatorKind::Position {
            kp: 10.0,
            gear: 3.0,
            ctrlrange: Some([-2.0, 2.0])
        }
    );
    assert_eq!(
        k(4),
        ActuatorKind::Position {
            kp: 7.0,
            gear: 3.0,
            ctrlrange: Some([0.0, 1.0])
        }
    );
    assert_eq!(s.actuators[0].joint.index(), 0);
    assert_eq!(s.actuators[1].joint.index(), 1);
}

#[test]
fn a_position_actuators_own_gear_is_kept() {
    // it was once read and then dropped: nothing an importer reads may vanish
    let s = load(&wrap(&format!(
        r#"{TWO_JOINTS}<actuator><position joint="j1" kp="4" gear="2.5"/></actuator>"#
    )));
    assert_eq!(
        s.actuators[0].kind,
        ActuatorKind::Position {
            kp: 4.0,
            gear: 2.5,
            ctrlrange: None
        }
    );
}

#[test]
fn a_position_actuator_without_a_default_has_unit_kp_and_a_motor_unit_gear() {
    let s = load(&wrap(&format!(
        r#"{TWO_JOINTS}<actuator><position joint="j1"/><motor joint="j2"/></actuator>"#
    )));
    assert_eq!(
        s.actuators[0].kind,
        ActuatorKind::Position {
            kp: 1.0,
            gear: 1.0,
            ctrlrange: None
        }
    );
    assert_eq!(
        s.actuators[1].kind,
        ActuatorKind::Motor {
            gear: 1.0,
            ctrlrange: None
        }
    );
}

#[test]
fn actuator_refusals() {
    let act = |compiler: &str, inner: &str| {
        wrap(&format!(
            "{compiler}{TWO_JOINTS}<actuator>{inner}</actuator>"
        ))
    };
    assert_eq!(
        mjcf_err(&act("", r#"<motor joint="nope"/>"#)).kind,
        MjcfErrorKind::UnknownReference
    );
    assert_eq!(
        mjcf_err(&act("", r#"<motor/>"#)).kind,
        MjcfErrorKind::MissingAttribute
    );
    // a ball joint cannot be driven by a one-value actuator
    assert_eq!(
        mjcf_err(&act("", r#"<motor joint="ball"/>"#)).kind,
        MjcfErrorKind::Inconsistent
    );
    let e = mjcf_err(&act("", r#"<motor joint="j1" gear="1 0 0 0 0 1"/>"#));
    assert!(e.message.contains("gear[1..]"), "{e}");
    let e = mjcf_err(&act(
        "",
        r#"<motor joint="j1" ctrlrange="1 -1" ctrllimited="true"/>"#,
    ));
    assert!(e.message.contains("invalid control range"), "{e}");
    let e = mjcf_err(&act(
        r#"<compiler autolimits="false"/>"#,
        r#"<motor joint="j1" ctrlrange="-1 1"/>"#,
    ));
    assert!(e.message.contains("autolimits"), "{e}");
    for tag in [
        "general",
        "velocity",
        "intvelocity",
        "damper",
        "cylinder",
        "muscle",
        "adhesion",
        "pid",
    ] {
        let e = mjcf_err(&act("", &format!(r#"<{tag} joint="j1"/>"#)));
        assert_eq!(e.kind, MjcfErrorKind::UnsupportedElement, "{tag}");
    }
    for attr in [
        "forcerange=\"-1 1\"",
        "forcelimited=\"true\"",
        "armature=\"1\"",
        "lengthrange=\"0 1\"",
        "tendon=\"t\"",
        "site=\"s\"",
        "body=\"b\"",
        "kv=\"1\"",
    ] {
        let e = mjcf_err(&act("", &format!(r#"<position joint="j1" {attr}/>"#)));
        assert_eq!(e.kind, MjcfErrorKind::UnsupportedAttribute, "{attr}");
    }
}

#[test]
fn fixed_tendons_are_imported_and_listed_only_for_their_passive_terms() {
    let s = load(&wrap(&format!(
        r#"{TWO_JOINTS}<tendon>
             <fixed name="t1" range="-0.3 2" stiffness="3" damping="0.5"><joint joint="j1" coef=".5"/><joint joint="j2" coef="-.5"/></fixed>
             <fixed name="t2"><joint joint="j1" coef="1"/></fixed>
             <fixed name="t3" armature="0.1" limited="true" range="0 1" frictionloss="2"><joint joint="j1" coef="1"/></fixed>
           </tendon>"#
    )));
    assert_eq!(s.tendons.len(), 3);
    let t = &s.tendons[0];
    assert_eq!(t.name, "t1");
    assert_eq!(t.range, Some([-0.3, 2.0]));
    assert_eq!((t.stiffness, t.damping), (3.0, 0.5));
    assert_eq!(t.joints.len(), 2);
    assert_eq!((t.joints[0].joint.index(), t.joints[0].coef), (0, 0.5));
    assert_eq!((t.joints[1].joint.index(), t.joints[1].coef), (1, -0.5));
    // no range, no limit; tendon range is not converted from degrees
    assert_eq!(s.tendons[1].range, None);
    // a tendon is no longer listed merely for existing: the limit and the friction
    // loss are simulated, so only a spring, a damper and an armature are recorded
    let listed = |path: &str, item: &str| {
        s.unsupported
            .iter()
            .any(|u| u.path == path && u.item == item)
    };
    assert!(listed("tendon/fixed[t1]", "@stiffness"));
    assert!(listed("tendon/fixed[t1]", "@damping"));
    assert!(!listed("tendon/fixed[t1]", "@armature"));
    assert!(!listed("tendon/fixed[t1]", "element"));
    assert!(s.unsupported.iter().all(|u| u.path != "tendon/fixed[t2]"));
    assert!(listed("tendon/fixed[t3]", "@armature"));
    assert!(!listed("tendon/fixed[t3]", "@stiffness"));
    assert_eq!(s.tendons[2].frictionloss, 2.0);
    assert_eq!(s.tendons[2].range, Some([0.0, 1.0]));
}

#[test]
fn tendon_refusals() {
    let t = |inner: &str| wrap(&format!("{TWO_JOINTS}<tendon>{inner}</tendon>"));
    assert_eq!(
        mjcf_err(&t(r#"<fixed><joint joint="nope" coef="1"/></fixed>"#)).kind,
        MjcfErrorKind::UnknownReference
    );
    assert_eq!(
        mjcf_err(&t(r#"<fixed/>"#)).kind,
        MjcfErrorKind::Inconsistent
    );
    assert_eq!(
        mjcf_err(&t(r#"<fixed><joint joint="j1"/></fixed>"#)).kind,
        MjcfErrorKind::MissingAttribute
    );
    assert_eq!(
        mjcf_err(&t(r#"<fixed><joint joint="ball" coef="1"/></fixed>"#)).kind,
        MjcfErrorKind::Inconsistent
    );
    assert_eq!(
        mjcf_err(&t(r#"<spatial><site site="a"/><site site="b"/></spatial>"#)).kind,
        MjcfErrorKind::UnsupportedElement
    );
    let e = mjcf_err(&t(
        r#"<fixed limited="true" range="1 0"><joint joint="j1" coef="1"/></fixed>"#,
    ));
    assert!(e.message.contains("invalid limits"), "{e}");
    assert_eq!(
        mjcf_err(&t(
            r#"<fixed springlength="1"><joint joint="j1" coef="1"/></fixed>"#
        ))
        .kind,
        MjcfErrorKind::UnsupportedAttribute
    );
}

// ---- materials, instances ----------------------------------------------------

#[test]
fn colours_become_linear_materials() {
    let s = load(&wrap(
        r#"<asset>
             <material name="m" rgba="0.2 0.4 0.6 1" metallic="0.7" roughness="0.2" emission="2"/>
           </asset>
           <worldbody>
             <geom name="plain" type="plane" size="0 0 1"/>
             <geom name="red" type="sphere" size="1" rgba="1 0.5 0 1"/>
             <geom name="again" type="sphere" size="1" rgba="1 0.5 0 1"/>
             <geom name="named" type="sphere" size="1" material="m"/>
             <geom name="over" type="sphere" size="1" material="m" rgba="0.5 0.5 0.9 1"/>
           </worldbody>"#,
    ));
    let mat = |g: &str| {
        let id = s.geoms.iter().find(|x| x.name == g).unwrap().material;
        &s.materials[id.index()]
    };
    // the asset material, first in the list, rgba read as sRGB and converted
    let m = &s.materials[0];
    assert_eq!(m.name, "m");
    for (got, srgb) in m.optical.base_colour_linear.iter().zip([0.2f32, 0.4, 0.6]) {
        assert_eq!(*got, srgb_to_linear(srgb));
    }
    assert_eq!((m.optical.metallic, m.optical.roughness), (0.7, 0.2));
    for (got, srgb) in m.optical.emission_linear.iter().zip([0.2f32, 0.4, 0.6]) {
        assert_eq!(*got, srgb_to_linear(srgb) * 2.0);
    }
    assert_eq!(mat("named").name, "m");
    // an rgba (not the default) gets its own material, shared by equal colours
    assert_eq!(mat("red").name, "rgba(1,0.5,0)");
    assert_eq!(
        mat("red").optical.base_colour_linear,
        [1.0, srgb_to_linear(0.5), 0.0]
    );
    assert_eq!(
        s.geoms.iter().find(|g| g.name == "red").unwrap().material,
        s.geoms.iter().find(|g| g.name == "again").unwrap().material
    );
    // MuJoCo's default colour is the default material
    assert_eq!(mat("plain").name, "rgba(0.5,0.5,0.5)");
    // an rgba over a named material keeps the material's other properties
    let over = mat("over");
    assert_eq!(over.name, "m+rgba(0.5,0.5,0.9)");
    assert_eq!((over.optical.metallic, over.optical.roughness), (0.7, 0.2));
    assert_eq!(over.optical.base_colour_linear[2], srgb_to_linear(0.9));
    s.validate().unwrap();
}

#[test]
fn alpha_is_recorded_and_unknown_materials_are_refused() {
    let s = load(&wrap(
        r#"<worldbody><geom type="sphere" size="1" rgba="1 0 0 0.5"/></worldbody>"#,
    ));
    assert!(
        s.unsupported
            .iter()
            .any(|u| u.item == "@rgba" && u.reason.contains("alpha"))
    );
    let e = mjcf_err(&wrap(
        r#"<worldbody><geom type="sphere" size="1" material="nope"/></worldbody>"#,
    ));
    assert_eq!(e.kind, MjcfErrorKind::UnknownReference);
    let e = mjcf_err(&wrap(
        r#"<asset><material name="a"/><material name="a"/></asset>"#,
    ));
    assert_eq!(e.kind, MjcfErrorKind::Duplicate);
}

#[test]
fn one_instance_per_geom_with_segmentation_ids_from_one() {
    let s = load(&wrap(
        r#"<worldbody><geom name="floor" type="plane" size="0 0 1"/>
             <body><freejoint/><geom size="0.1"/><geom size="0.2" pos="1 0 0"/></body>
           </worldbody>"#,
    ));
    assert_eq!(s.instances.len(), 3);
    for (i, inst) in s.instances.iter().enumerate() {
        assert_eq!(inst.seg_id as usize, i + 1, "0 is the background");
        assert_eq!(
            inst.mesh_or_geom,
            sim_scene::ShapeRef::Geom(sim_scene::GeomId(i as u32))
        );
        assert_eq!(inst.body, s.geoms[i].body);
        assert_eq!(inst.material, s.geoms[i].material);
        assert_eq!(inst.local_quat, [0.0, 0.0, 0.0, 1.0]);
    }
    assert_eq!(s.instances[0].body, None, "the floor is static");
    let none = mjcf::load_with(
        &wrap(r#"<worldbody><geom type="plane" size="0 0 1"/></worldbody>"#),
        no_assets(),
        &mjcf::LoadOptions {
            strict: false,
            instances: false,
        },
    )
    .unwrap();
    assert!(none.instances.is_empty());
}

// ---- document level ----------------------------------------------------------

#[test]
fn option_gravity_timestep_and_defaults() {
    let s = load(&wrap(
        r#"<option timestep="0.01" gravity="0 0 -1.62"/><worldbody/>"#,
    ));
    assert_eq!((s.timestep_s, s.gravity), (0.01, [0.0, 0.0, -1.62]));
    let s = load(&wrap(r#"<worldbody/>"#));
    assert_eq!((s.timestep_s, s.gravity), (0.002, [0.0, 0.0, -9.81]));
    assert!(
        mjcf_err(&wrap(r#"<option timestep="0"/>"#))
            .message
            .contains("positive")
    );
    // the solver settings that are not modelled are recorded, wind and flags are refused
    let s = load(&wrap(r#"<option noslip_iterations="5" jacobian="dense"/>"#));
    for item in ["@noslip_iterations", "@jacobian"] {
        assert!(
            s.unsupported
                .iter()
                .any(|u| u.item == item && u.path == "option"),
            "{item}"
        );
    }
    // the modelled ones are read, and not recorded
    let s = load(&wrap(r#"<option iterations="50"/>"#));
    assert_eq!(s.options.iterations, 50);
    assert!(s.unsupported.iter().all(|u| u.item != "@iterations"));
    // the integrator: Euler (the default) and RK4 are read, not recorded
    assert_eq!(load(&wrap(r#"<worldbody/>"#)).integrator, Integrator::Euler);
    for (text, expect) in [("Euler", Integrator::Euler), ("RK4", Integrator::Rk4)] {
        let s = load(&wrap(&format!(
            r#"<option integrator="{text}"/><worldbody/>"#
        )));
        assert_eq!(s.integrator, expect, "{text}");
        assert!(
            s.unsupported.iter().all(|u| u.item != "@integrator"),
            "{text} is modelled and must not be recorded"
        );
    }
    // implicit, implicitfast and discrete stay recorded; the scene keeps Euler
    for text in ["implicit", "implicitfast", "discrete"] {
        let s = load(&wrap(&format!(
            r#"<option integrator="{text}"/><worldbody/>"#
        )));
        assert_eq!(s.integrator, Integrator::Euler, "{text}");
        assert!(
            s.unsupported
                .iter()
                .any(|u| u.item == "@integrator" && u.path == "option"),
            "{text}"
        );
    }
    // an unknown keyword is refused, as MuJoCo refuses it
    assert_eq!(
        mjcf_err(&wrap(r#"<option integrator="rk4"/>"#)).kind,
        MjcfErrorKind::BadValue
    );
    for attr in ["wind=\"1 0 0\"", "density=\"1.2\"", "viscosity=\"0.1\""] {
        assert_eq!(
            mjcf_err(&wrap(&format!("<option {attr}/>"))).kind,
            MjcfErrorKind::UnsupportedAttribute,
            "{attr}"
        );
    }
    assert_eq!(
        mjcf_err(&wrap(r#"<option><flag gravity="disable"/></option>"#)).kind,
        MjcfErrorKind::UnsupportedElement
    );
}

#[test]
fn what_is_refused_names_where() {
    // (document, kind, path fragment, line)
    let cases: Vec<(String, MjcfErrorKind, &str, u32)> = vec![
        (wrap("<equality/>"), MjcfErrorKind::UnsupportedElement, "equality", 1),
        (wrap("<sensor/>"), MjcfErrorKind::UnsupportedElement, "sensor", 1),
        (wrap("<include file=\"x.xml\"/>"), MjcfErrorKind::UnsupportedElement, "include", 1),
        (wrap("<size memory=\"1M\"/>"), MjcfErrorKind::UnsupportedElement, "size", 1),
        (
            wrap("<worldbody><body><frame/></body></worldbody>"),
            MjcfErrorKind::UnsupportedElement,
            "worldbody/body/frame",
            1,
        ),
        (
            wrap("<worldbody><body><composite/></body></worldbody>"),
            MjcfErrorKind::UnsupportedElement,
            "composite",
            1,
        ),
        (
            wrap(r#"<worldbody><body mocap="true"><geom size="1"/></body></worldbody>"#),
            MjcfErrorKind::UnsupportedAttribute,
            "worldbody/body",
            1,
        ),
        (
            wrap(r#"<worldbody><body gravcomp="1"><geom size="1"/></body></worldbody>"#),
            MjcfErrorKind::UnsupportedAttribute,
            "worldbody/body",
            1,
        ),
        (
            wrap(r#"<worldbody><body><geom size="1" shellinertia="true"/></body></worldbody>"#),
            MjcfErrorKind::UnsupportedAttribute,
            "worldbody/body/geom",
            1,
        ),
        (
            wrap(r#"<worldbody><body><geom size="1" fitscale="2"/></body></worldbody>"#),
            MjcfErrorKind::UnsupportedAttribute,
            "geom",
            1,
        ),
        (
            wrap(r#"<worldbody><geom size="1" bogus="1"/></worldbody>"#),
            MjcfErrorKind::UnsupportedAttribute,
            "worldbody/geom",
            1,
        ),
        (
            wrap(r#"<compiler boundmass="1"/>"#),
            MjcfErrorKind::UnsupportedAttribute,
            "compiler",
            1,
        ),
        (
            wrap(r#"<compiler balanceinertia="true"/>"#),
            MjcfErrorKind::UnsupportedAttribute,
            "compiler",
            1,
        ),
        (
            wrap(r#"<compiler fusestatic="true"/>"#),
            MjcfErrorKind::UnsupportedAttribute,
            "compiler",
            1,
        ),
        (
            wrap(r#"<contact><bogus geom1="a" geom2="b"/></contact>"#),
            MjcfErrorKind::UnsupportedElement,
            "contact/bogus",
            1,
        ),
        (
            wrap(r#"<contact><exclude body1="a" body2="b" nope="1"/></contact>"#),
            MjcfErrorKind::UnsupportedAttribute,
            "contact/exclude",
            1,
        ),
        (
            wrap(r#"<asset><hfield name="h" nrow="1" ncol="1" size="1 1 1 1"/></asset>"#),
            MjcfErrorKind::UnsupportedElement,
            "asset/hfield",
            1,
        ),
        (
            "<mujoco>\n<worldbody>\n<body>\n<geom size=\"1\" nope=\"1\"/>\n</body>\n</worldbody>\n</mujoco>".to_string(),
            MjcfErrorKind::UnsupportedAttribute,
            "worldbody/body/geom",
            4,
        ),
        ("<other/>".to_string(), MjcfErrorKind::UnsupportedElement, "other", 1),
    ];
    for (xml, kind, path, line) in cases {
        let e = mjcf_err(&xml);
        assert_eq!(e.kind, kind, "{xml}\n{e}");
        assert!(
            e.path.contains(path),
            "path '{}' does not contain '{path}'\n{xml}",
            e.path
        );
        assert_eq!(e.line, line, "{xml}");
        // the message is a sentence a person can act on
        assert!(e.message.len() > 10);
        assert!(e.to_string().contains("MJCF"));
    }
}

#[test]
fn malformed_xml_and_dtds_are_refused() {
    for xml in [
        "",
        "<mujoco>",
        "<mujoco></wrong>",
        "not xml",
        "<mujoco/><mujoco/>",
    ] {
        let e = mjcf_err(xml);
        assert_eq!(e.kind, MjcfErrorKind::Xml, "{xml:?}");
    }
    // a DTD (entity expansion) is refused
    let e = mjcf_err("<!DOCTYPE mujoco [<!ENTITY a \"b\">]><mujoco/>");
    assert_eq!(e.kind, MjcfErrorKind::Xml);
}

#[test]
fn names_are_unique_and_world_is_reserved() {
    let e = mjcf_err(&wrap(
        r#"<worldbody><body name="a"/><body name="a"/></worldbody>"#,
    ));
    assert_eq!(e.kind, MjcfErrorKind::Duplicate);
    let e = mjcf_err(&wrap(
        r#"<worldbody><geom name="g" size="1"/><geom name="g" size="1"/></worldbody>"#,
    ));
    assert_eq!(e.kind, MjcfErrorKind::Duplicate);
    let e = mjcf_err(&wrap(r#"<worldbody><body name="world"/></worldbody>"#));
    assert_eq!(e.kind, MjcfErrorKind::Duplicate);
    let e = mjcf_err(&wrap(r#"<worldbody name="x"/>"#));
    assert!(e.message.contains("world body"), "{e}");
}

#[test]
fn the_default_import_records_and_the_strict_import_refuses() {
    let xml = wrap(
        r#"<visual><global fovy="45"/></visual>
           <default><geom group="1"/></default>
           <worldbody><light pos="0 0 3"/><camera name="c" pos="0 0 1"/><site name="s"/>
             <body><geom size="1" group="2"/></body></worldbody>
           <contact><pair geom1="a" geom2="b"/></contact>
           <keyframe><key name="k" qpos=""/></keyframe>"#,
    );
    let s = load(&xml);
    let listed: Vec<(&str, &str)> = s
        .unsupported
        .iter()
        .map(|u| (u.path.as_str(), u.item.as_str()))
        .collect();
    for want in [
        ("visual/global", "@fovy"),
        ("default/geom", "@group"),
        ("worldbody/light", "element"),
        ("worldbody/site[s]", "element"),
        ("worldbody/body/geom", "@group"),
        ("contact/pair", "element"),
        ("keyframe", "element"),
    ] {
        assert!(
            listed.contains(&want),
            "{want:?} not recorded in {listed:?}"
        );
    }
    // a fixed camera is imported, not recorded
    assert!(!listed.contains(&("worldbody/camera[c]", "element")));
    assert_eq!(s.cameras.len(), 1);
    assert_eq!(s.cameras[0].name, "c");
    let strict = mjcf::LoadOptions {
        strict: true,
        instances: true,
    };
    let e = match mjcf::load_with(&xml, no_assets(), &strict) {
        Err(SceneError::Mjcf(e)) => e,
        other => panic!("{other:?}"),
    };
    assert_eq!(e.kind, MjcfErrorKind::Strict);
    // a document with nothing to record imports under strict
    mjcf::load_with(
        &wrap(
            r#"<worldbody><body><freejoint/><geom type="sphere" size="0.1"/></body></worldbody>"#,
        ),
        no_assets(),
        &strict,
    )
    .unwrap();
}

// ---- STL meshes ------------------------------------------------------------

/// A binary STL of the triangles.
fn stl(tris: &[[[f32; 3]; 3]]) -> Vec<u8> {
    let mut out = vec![0u8; 80];
    out.extend_from_slice(&(tris.len() as u32).to_le_bytes());
    for t in tris {
        out.extend_from_slice(&[0u8; 12]); // normal (ignored)
        for v in t {
            for c in v {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        out.extend_from_slice(&[0u8; 2]);
    }
    out
}

/// The 12 outward counter-clockwise triangles of the axis-aligned cube of
/// half-extent `h` centred on `c`.
fn cube(h: f32, c: [f32; 3]) -> Vec<[[f32; 3]; 3]> {
    let p = |x: f32, y: f32, z: f32| [c[0] + x * h, c[1] + y * h, c[2] + z * h];
    let (p000, p100, p110, p010) = (
        p(-1., -1., -1.),
        p(1., -1., -1.),
        p(1., 1., -1.),
        p(-1., 1., -1.),
    );
    let (p001, p101, p111, p011) = (
        p(-1., -1., 1.),
        p(1., -1., 1.),
        p(1., 1., 1.),
        p(-1., 1., 1.),
    );
    let quads = [
        [p000, p010, p110, p100], // -z
        [p001, p101, p111, p011], // +z
        [p000, p100, p101, p001], // -y
        [p010, p011, p111, p110], // +y
        [p000, p001, p011, p010], // -x
        [p100, p110, p111, p101], // +x
    ];
    quads
        .iter()
        .flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
        .collect()
}

fn assets(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("mjcf_import")
        .join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn mesh_scene(dir: &Path, mesh_attrs: &str, geom_attrs: &str) -> Result<Scene, SceneError> {
    let xml = wrap(&format!(
        r#"<asset><mesh name="m" file="m.stl" {mesh_attrs}/></asset>
           <worldbody><body><freejoint/><geom type="mesh" mesh="m" {geom_attrs}/></body></worldbody>"#
    ));
    mjcf::load(&xml, dir)
}

#[test]
fn a_cube_mesh_has_the_mass_and_inertia_of_the_equivalent_box() {
    let dir = assets("cube");
    fs::write(dir.join("m.stl"), stl(&cube(0.5, [0.0; 3]))).unwrap();
    let s = mesh_scene(&dir, "", "").unwrap();
    // 8 distinct vertices, 12 triangles (vertices de-duplicated by bit pattern)
    assert_eq!(
        (s.meshes[0].vertices.len(), s.meshes[0].triangles.len()),
        (8, 12)
    );
    assert_eq!(
        s.geoms[0].shape,
        Shape::Mesh {
            mesh: sim_scene::MeshId(0)
        }
    );
    let b = load(&wrap(
        r#"<worldbody><body><freejoint/><geom type="box" size="0.5 0.5 0.5"/></body></worldbody>"#,
    ));
    let (im, ib) = (s.bodies[0].inertial.unwrap(), b.bodies[0].inertial.unwrap());
    close(im.mass_kg, 1000.0, 1e-6, "mesh mass");
    close(im.mass_kg, ib.mass_kg, 1e-6, "mass vs box");
    close_all(&im.diag_inertia, &ib.diag_inertia, 1e-6, "inertia vs box");
    close(im.diag_inertia[0], 1000.0 / 6.0, 1e-6, "m a^2 / 6");
    close_all(&im.com, &[0.0; 3], 1e-9, "com");
}

#[test]
fn an_offset_mesh_keeps_its_vertices_and_moves_the_body_com() {
    let dir = assets("offset");
    fs::write(dir.join("m.stl"), stl(&cube(0.5, [1.0, 2.0, 3.0]))).unwrap();
    let s = mesh_scene(&dir, "", r#"pos="0 0 10""#).unwrap();
    // the stored mesh is in its own frame, as authored
    assert!(s.meshes[0].vertices.contains(&[1.5, 2.5, 3.5]));
    // the geom keeps its authored pose; the centre of mass is the mesh centre in the geom frame
    assert_eq!(s.geoms[0].pos, [0.0, 0.0, 10.0]);
    let i = s.bodies[0].inertial.unwrap();
    close_all(&i.com, &[1.0, 2.0, 13.0], 1e-6, "com");
    close(i.mass_kg, 1000.0, 1e-6, "mass");
    close(
        i.diag_inertia[1],
        1000.0 / 6.0,
        1e-6,
        "inertia about the centre of mass",
    );
}

#[test]
fn mesh_scale_and_mirroring() {
    let dir = assets("scale");
    fs::write(dir.join("m.stl"), stl(&cube(0.5, [0.0; 3]))).unwrap();
    let s = mesh_scene(&dir, r#"scale="2 2 2""#, "").unwrap();
    let i = s.bodies[0].inertial.unwrap();
    close(i.mass_kg, 8000.0, 1e-6, "mass at scale 2");
    close(
        i.diag_inertia[0],
        8000.0 * 4.0 / 6.0,
        1e-6,
        "inertia at scale 2",
    );
    assert!(s.meshes[0].vertices.contains(&[1.0, 1.0, 1.0]));
    // a mirrored scale flips the winding so the volume stays positive
    let m = mesh_scene(&dir, r#"scale="-1 1 1""#, "").unwrap();
    close(
        m.bodies[0].inertial.unwrap().mass_kg,
        1000.0,
        1e-6,
        "mirrored mass",
    );
    let plain = mesh_scene(&dir, "", "").unwrap();
    for (a, b) in m.meshes[0].triangles.iter().zip(&plain.meshes[0].triangles) {
        assert_eq!((a[0], a[1], a[2]), (b[0], b[2], b[1]));
    }
    // the mesh default class sets the scale
    let xml = wrap(
        r#"<default><mesh scale="2 2 2"/></default>
           <asset><mesh name="m" file="m.stl"/></asset>
           <worldbody><body><freejoint/><geom type="mesh" mesh="m"/></body></worldbody>"#,
    );
    close(
        mjcf::load(&xml, &dir).unwrap().bodies[0]
            .inertial
            .unwrap()
            .mass_kg,
        8000.0,
        1e-6,
        "class scale",
    );
}

#[test]
fn inverted_winding_is_legacy_positive_and_exact_negative() {
    let dir = assets("inverted");
    let inverted: Vec<[[f32; 3]; 3]> = cube(0.5, [0.0; 3])
        .iter()
        .map(|t| [t[0], t[2], t[1]])
        .collect();
    fs::write(dir.join("m.stl"), stl(&inverted)).unwrap();
    // MuJoCo's default (legacy) takes every pyramid volume positive
    let s = mesh_scene(&dir, "", "").unwrap();
    close(
        s.bodies[0].inertial.unwrap().mass_kg,
        1000.0,
        1e-6,
        "legacy mass",
    );
    // exact keeps the sign and refuses a mesh with negative volume
    let e = match mesh_scene(&dir, r#"inertia="exact""#, "") {
        Err(SceneError::Mjcf(e)) => e,
        other => panic!("{other:?}"),
    };
    assert_eq!(e.kind, MjcfErrorKind::Asset);
    assert!(e.message.contains("negative"), "{e}");
    // exact on a well-wound cube works
    fs::write(dir.join("m.stl"), stl(&cube(0.5, [0.0; 3]))).unwrap();
    close(
        mesh_scene(&dir, r#"inertia="exact""#, "").unwrap().bodies[0]
            .inertial
            .unwrap()
            .mass_kg,
        1000.0,
        1e-6,
        "exact mass",
    );
    for other in ["convex", "shell"] {
        let e = match mesh_scene(&dir, &format!(r#"inertia="{other}""#), "") {
            Err(SceneError::Mjcf(e)) => e,
            other => panic!("{other:?}"),
        };
        assert_eq!(e.kind, MjcfErrorKind::UnsupportedAttribute);
    }
}

#[test]
fn a_mesh_name_defaults_to_the_file_stem() {
    let dir = assets("stem");
    fs::write(dir.join("widget.stl"), stl(&cube(0.5, [0.0; 3]))).unwrap();
    let xml = wrap(
        r#"<asset><mesh file="widget.stl"/></asset>
           <worldbody><body><freejoint/><geom type="mesh" mesh="widget"/></body></worldbody>"#,
    );
    assert_eq!(mjcf::load(&xml, &dir).unwrap().meshes[0].name, "widget");
}

#[test]
fn meshdir_and_assetdir_locate_files_inside_the_asset_directory() {
    let dir = assets("meshdir");
    fs::create_dir_all(dir.join("meshes")).unwrap();
    fs::write(dir.join("meshes").join("m.stl"), stl(&cube(0.5, [0.0; 3]))).unwrap();
    for compiler in [
        r#"<compiler meshdir="meshes"/>"#,
        r#"<compiler assetdir="meshes"/>"#,
    ] {
        let xml = wrap(&format!(
            r#"{compiler}<asset><mesh name="m" file="m.stl"/></asset>
               <worldbody><body><freejoint/><geom type="mesh" mesh="m"/></body></worldbody>"#
        ));
        mjcf::load(&xml, &dir).unwrap();
    }
}

#[test]
fn mesh_file_refusals() {
    let dir = assets("refusals");
    fs::write(dir.join("m.stl"), stl(&cube(0.5, [0.0; 3]))).unwrap();
    let load_file = |file: &str, compiler: &str| {
        let xml = wrap(&format!(
            r#"{compiler}<asset><mesh name="m" file="{file}"/></asset><worldbody/>"#
        ));
        match mjcf::load(&xml, &dir) {
            Err(SceneError::Mjcf(e)) => e,
            other => panic!("{file}: {other:?}"),
        }
    };
    // nothing outside the asset directory is read
    for file in [
        "../m.stl",
        "sub/../../m.stl",
        "/etc/passwd.stl",
        "C:/Windows/x.stl",
        "C:\\x.stl",
    ] {
        let e = load_file(file, "");
        assert_eq!(e.kind, MjcfErrorKind::Asset, "{file}");
        assert!(
            e.message.contains("relative path") || e.message.contains("cannot be opened"),
            "{file}: {e}"
        );
    }
    let e = load_file("m.stl", r#"<compiler meshdir="../"/>"#);
    assert_eq!(e.kind, MjcfErrorKind::Asset);
    // missing file, wrong extension, a directory
    assert_eq!(load_file("nope.stl", "").kind, MjcfErrorKind::Asset);
    assert!(load_file("m.obj", "").message.contains("only binary STL"));
    fs::create_dir_all(dir.join("d.stl")).unwrap();
    assert_eq!(load_file("d.stl", "").kind, MjcfErrorKind::Asset);
    // an ASCII STL
    fs::write(dir.join("ascii.stl"), b"solid cube\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid cube\n").unwrap();
    let e = load_file("ascii.stl", "");
    assert!(e.message.contains("ASCII"), "{e}");
    // a size that disagrees with the triangle count
    let mut bytes = stl(&cube(0.5, [0.0; 3]));
    bytes.truncate(bytes.len() - 7);
    fs::write(dir.join("short.stl"), &bytes).unwrap();
    assert!(load_file("short.stl", "").message.contains("wrong size"));
    // too short for a header, empty, too many triangles
    fs::write(dir.join("tiny.stl"), [0u8; 10]).unwrap();
    assert!(load_file("tiny.stl", "").message.contains("invalid header"));
    fs::write(dir.join("empty.stl"), []).unwrap();
    assert!(load_file("empty.stl", "").message.contains("empty"));
    let mut many = vec![0u8; 84];
    many[80..84].copy_from_slice(&200_001u32.to_le_bytes());
    fs::write(dir.join("many.stl"), &many).unwrap();
    assert!(
        load_file("many.stl", "")
            .message
            .contains("between 1 and 200000")
    );
    // a huge file is refused before it is read
    let huge = fs::File::create(dir.join("huge.stl")).unwrap();
    huge.set_len(84 + 200_000 * 50 + 1).unwrap();
    drop(huge);
    assert!(load_file("huge.stl", "").message.contains("larger than"));
    // a vertex beyond MuJoCo's bound
    let mut far = stl(&cube(0.5, [0.0; 3]));
    far[84 + 12..84 + 16].copy_from_slice(&2e9f32.to_le_bytes());
    fs::write(dir.join("far.stl"), &far).unwrap();
    assert!(load_file("far.stl", "").message.contains("maximum bounds"));
    // a non-finite vertex
    let mut nan = stl(&cube(0.5, [0.0; 3]));
    nan[84 + 12..84 + 16].copy_from_slice(&f32::NAN.to_le_bytes());
    fs::write(dir.join("nan.stl"), &nan).unwrap();
    assert!(
        load_file("nan.stl", "").message.contains("exceeds")
            || load_file("nan.stl", "").message.contains("finite")
    );
}

#[test]
fn degenerate_meshes_are_refused() {
    let dir = assets("degenerate");
    // a flat triangle fan has no volume
    let flat = vec![
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
    ];
    fs::write(dir.join("m.stl"), stl(&flat)).unwrap();
    let e = match mesh_scene(&dir, "", "") {
        Err(SceneError::Mjcf(e)) => e,
        other => panic!("{other:?}"),
    };
    assert_eq!(e.kind, MjcfErrorKind::Asset);
    // only three vertices
    fs::write(
        dir.join("m.stl"),
        stl(&[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
    )
    .unwrap();
    let e = match mesh_scene(&dir, "", "") {
        Err(SceneError::Mjcf(e)) => e,
        other => panic!("{other:?}"),
    };
    assert!(e.message.contains("4 vertices"), "{e}");
}

#[test]
fn a_primitive_that_fits_a_mesh_is_refused() {
    let dir = assets("fit");
    fs::write(dir.join("m.stl"), stl(&cube(0.5, [0.0; 3]))).unwrap();
    let xml = wrap(
        r#"<asset><mesh name="m" file="m.stl"/></asset>
           <worldbody><body><geom type="box" mesh="m"/></body></worldbody>"#,
    );
    match mjcf::load(&xml, &dir) {
        Err(SceneError::Mjcf(e)) => assert_eq!(e.kind, MjcfErrorKind::UnsupportedAttribute),
        other => panic!("{other:?}"),
    }
}

#[test]
fn mesh_attribute_refusals() {
    let dir = assets("meshattrs");
    fs::write(dir.join("m.stl"), stl(&cube(0.5, [0.0; 3]))).unwrap();
    for attr in [
        "refpos=\"1 0 0\"",
        "refquat=\"0 1 0 0\"",
        "vertex=\"0 0 0 1 0 0 0 1 0 0 0 1\"",
        "builtin=\"sphere\"",
        "maxhullvert=\"4\"",
    ] {
        let xml = wrap(&format!(
            r#"<asset><mesh name="m" file="m.stl" {attr}/></asset>"#
        ));
        match mjcf::load(&xml, &dir) {
            Err(SceneError::Mjcf(e)) => {
                assert_eq!(e.kind, MjcfErrorKind::UnsupportedAttribute, "{attr}")
            }
            other => panic!("{attr}: {other:?}"),
        }
    }
    // missing file attribute
    match mjcf::load(&wrap(r#"<asset><mesh name="m"/></asset>"#), &dir) {
        Err(SceneError::Mjcf(e)) => assert_eq!(e.kind, MjcfErrorKind::MissingAttribute),
        other => panic!("{other:?}"),
    }
    // duplicate mesh names
    let xml = wrap(r#"<asset><mesh name="m" file="m.stl"/><mesh name="m" file="m.stl"/></asset>"#);
    match mjcf::load(&xml, &dir) {
        Err(SceneError::Mjcf(e)) => assert_eq!(e.kind, MjcfErrorKind::Duplicate),
        other => panic!("{other:?}"),
    }
}

#[test]
fn an_imported_scene_with_a_mesh_round_trips_through_json() {
    let dir = assets("json");
    fs::write(dir.join("m.stl"), stl(&cube(0.5, [0.0; 3]))).unwrap();
    let s = mesh_scene(&dir, "", "").unwrap();
    assert_eq!(Scene::from_json(&s.to_json().unwrap()).unwrap(), s);
}

#[test]
fn strict_import_refuses_a_fixed_tendon_with_a_spring_but_not_one_without() {
    let strict = mjcf::LoadOptions {
        strict: true,
        instances: true,
    };
    // a tendon whose limit and friction loss are all it has is fully modelled
    let plain = wrap(&format!(
        r#"{TWO_JOINTS}<tendon><fixed name="t" limited="true" range="-1 1" frictionloss="0.1"><joint joint="j1" coef="1"/></fixed></tendon>"#
    ));
    mjcf::load_with(&plain, no_assets(), &strict).unwrap();
    let xml = wrap(&format!(
        r#"{TWO_JOINTS}<tendon><fixed name="t" stiffness="2"><joint joint="j1" coef="1"/></fixed></tendon>"#
    ));
    load(&xml);
    match mjcf::load_with(&xml, no_assets(), &strict) {
        Err(SceneError::Mjcf(e)) => {
            assert_eq!(e.kind, MjcfErrorKind::Strict);
            assert!(e.path.contains("tendon/fixed[t]"), "{e}");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_nesting_limit_holds_on_a_caller_thread_with_a_small_stack() {
    // The import runs on its own thread with a fixed stack, so a caller with a
    // small stack (a Windows main thread has 1 MiB; this one has 256 KiB) can load
    // a document at the nesting limit, and is refused past it, without overflowing.
    fn nested(levels: usize) -> String {
        let mut s = String::from("<worldbody>");
        s.push_str(&"<body>".repeat(levels));
        s.push_str(&"</body>".repeat(levels));
        s.push_str("</worldbody>");
        wrap(&s)
    }
    let (at_limit, past_limit) = (nested(255), nested(300));
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(move || {
            assert_eq!(load(&at_limit).bodies.len(), 255);
            let e = mjcf_err(&past_limit);
            assert_eq!(e.kind, MjcfErrorKind::Inconsistent);
            assert!(e.message.contains("nested more than"), "{e}");
        })
        .unwrap()
        .join()
        .expect("no overflow and no panic on the small-stack caller");
}

#[test]
fn hostile_nesting_and_size_are_refused_not_crashed_on() {
    // 300 nested bodies: past the limit of 256, so refused with the element's path
    let mut deep = String::from("<worldbody>");
    for _ in 0..300 {
        deep.push_str("<body>");
    }
    for _ in 0..300 {
        deep.push_str("</body>");
    }
    deep.push_str("</worldbody>");
    let e = mjcf_err(&wrap(&deep));
    assert_eq!(e.kind, MjcfErrorKind::Inconsistent);
    assert!(e.message.contains("nested more than"), "{e}");
    // 200 nested bodies are fine
    let mut ok = String::from("<worldbody>");
    for _ in 0..200 {
        ok.push_str("<body>");
    }
    for _ in 0..200 {
        ok.push_str("</body>");
    }
    ok.push_str("</worldbody>");
    assert_eq!(load(&wrap(&ok)).bodies.len(), 200);
    // nested default classes obey the same limit
    let mut classes = String::from("<default>");
    for i in 0..300 {
        classes.push_str(&format!(r#"<default class="c{i}">"#));
    }
    for _ in 0..300 {
        classes.push_str("</default>");
    }
    classes.push_str("</default>");
    assert_eq!(mjcf_err(&wrap(&classes)).kind, MjcfErrorKind::Inconsistent);
    // a document over the size limit is refused before it is parsed
    let huge = format!("<mujoco>{}</mujoco>", " ".repeat(65 * 1024 * 1024));
    let e = mjcf_err(&huge);
    assert_eq!(e.kind, MjcfErrorKind::Xml);
    assert!(e.message.contains("larger than"), "{e}");
}

#[test]
fn a_model_with_many_bodies_imports_in_linear_time() {
    // 20000 bodies with one geom each, in a chain of 100-deep trees: a quadratic
    // body-to-geom lookup would take minutes here
    let mut xml = String::from("<worldbody>");
    for t in 0..200 {
        xml.push_str(&format!(
            r#"<body name="t{t}"><freejoint/><geom size="0.1"/>"#
        ));
        for d in 0..99 {
            xml.push_str(&format!(
                r#"<body name="t{t}_{d}" pos="0 0 1"><geom size="0.1"/>"#
            ));
        }
        for _ in 0..99 {
            xml.push_str("</body>");
        }
        xml.push_str("</body>");
    }
    xml.push_str("</worldbody>");
    let started = std::time::Instant::now();
    let s = load(&wrap(&xml));
    assert_eq!(s.bodies.len(), 200 * 100);
    assert_eq!(s.geoms.len(), 200 * 100);
    assert!(
        started.elapsed() < std::time::Duration::from_secs(20),
        "import took {:?}",
        started.elapsed()
    );
}

// ---- contact parameters and exclusions (phase 1c-ii) -------------------------

#[test]
fn a_geoms_contact_parameters_default_to_mujocos() {
    let s = load(&wrap(
        r#"<worldbody><body><freejoint/><geom type="sphere" size="0.1"/></body></worldbody>"#,
    ));
    let g = &s.geoms[0];
    assert_eq!(g.solref, DEFAULT_SOLREF);
    assert_eq!(g.solimp, DEFAULT_SOLIMP);
    assert_eq!((g.solmix, g.priority, g.margin, g.gap), (1.0, 0, 0.0, 0.0));
    assert!(s.contact_excludes.is_empty());
    assert!(s.unsupported.is_empty(), "{:?}", s.unsupported);
}

#[test]
fn the_contact_parameters_are_read_through_default_classes() {
    let s = load(&wrap(
        r#"<default>
             <geom solref=".03 1.5" solimp=".8 .85" solmix="2" priority="1" margin=".01" gap=".02"/>
             <default class="c"><geom solref="-500 -10" priority="3" solimp=".7 .75 .5"/></default>
           </default>
           <worldbody>
             <body><freejoint/><geom name="a" type="sphere" size="0.1"/></body>
             <body><freejoint/><geom name="b" type="sphere" size="0.1" class="c" gap="0"/></body>
             <body><freejoint/><geom name="d" type="sphere" size="0.1" solref=".05" solmix="0.25"/></body>
           </worldbody>"#,
    ));
    let by = |n: &str| s.geoms.iter().find(|g| g.name == n).unwrap();
    // fewer numbers than the attribute holds override the leading entries and keep the
    // class's others (MuJoCo's ReadAttr)
    assert_eq!(by("a").solref, [0.03, 1.5]);
    assert_eq!(by("a").solimp, [0.8, 0.85, 0.001, 0.5, 2.0]);
    assert_eq!((by("a").solmix, by("a").priority), (2.0, 1));
    assert_eq!((by("a").margin, by("a").gap), (0.01, 0.02));
    // a class overrides its parent, an element overrides its class
    assert_eq!(by("b").solref, [-500.0, -10.0]);
    assert_eq!(by("b").solimp, [0.7, 0.75, 0.5, 0.5, 2.0]);
    assert_eq!(by("b").priority, 3);
    assert_eq!((by("b").margin, by("b").gap), (0.01, 0.0));
    assert_eq!(by("d").solref, [0.05, 1.5]);
    assert_eq!(by("d").solmix, 0.25);
    assert!(s.unsupported.is_empty(), "{:?}", s.unsupported);
}

#[test]
fn bad_contact_parameters_are_refused_with_the_field() {
    let g = |attrs: &str| {
        wrap(&format!(
            r#"<worldbody><body><freejoint/><geom type="sphere" size="0.1" {attrs}/></body></worldbody>"#
        ))
    };
    // a negative solmix, margin or gap, a mixed solref and an out-of-range solimp
    for (attrs, field) in [
        (r#"solmix="-1""#, "solmix"),
        (r#"margin="-0.1""#, "margin"),
        (r#"gap="-0.1""#, "gap"),
        (r#"solref="0.02 -1""#, "solref"),
        (r#"solimp="0.9 1.5""#, "solimp"),
        (r#"solimp="0.9 0.95 0.001 0.5 0.5""#, "solimp"),
    ] {
        match mjcf::load(&g(attrs), no_assets()) {
            Err(SceneError::Invalid { path, .. }) => {
                assert!(path.contains(field), "{attrs}: {path}")
            }
            other => panic!("{attrs}: {other:?}"),
        }
    }
    // a malformed value is a format error
    assert_eq!(
        mjcf_err(&g(r#"priority="1.5""#)).kind,
        MjcfErrorKind::BadValue
    );
    assert_eq!(
        mjcf_err(&g(r#"solref="1 2 3""#)).kind,
        MjcfErrorKind::BadValue
    );
    // any integer priority is accepted
    let s = load(&g(r#"priority="-7""#));
    assert_eq!(s.geoms[0].priority, -7);
}

#[test]
fn a_scene_json_without_the_contact_parameters_still_reads_with_their_defaults() {
    let s = load(&wrap(
        r#"<worldbody><body><freejoint/><geom type="sphere" size="0.1" solmix="3" margin="0.2"/></body></worldbody>"#,
    ));
    let mut v: serde_json::Value = serde_json::from_str(&s.to_json().unwrap()).unwrap();
    for geom in v["geoms"].as_array_mut().unwrap() {
        let o = geom.as_object_mut().unwrap();
        for key in ["solref", "solimp", "solmix", "priority", "margin", "gap"] {
            assert!(o.remove(key).is_some(), "{key} is written");
        }
    }
    v.as_object_mut().unwrap().remove("contact_excludes");
    let back = Scene::from_json(&v.to_string()).unwrap();
    let g = &back.geoms[0];
    assert_eq!((g.solref, g.solimp), (DEFAULT_SOLREF, DEFAULT_SOLIMP));
    assert_eq!((g.solmix, g.priority, g.margin, g.gap), (1.0, 0, 0.0, 0.0));
    assert!(back.contact_excludes.is_empty());
}

#[test]
fn contact_exclusions_resolve_body_names_and_the_world() {
    let bodies = r#"<worldbody>
        <geom type="plane" size="1 1 .1"/>
        <body name="a"><freejoint/><geom size=".1"/></body>
        <body name="b"><freejoint/><geom size=".1"/></body>
      </worldbody>"#;
    let s = load(&wrap(&format!(
        r#"{bodies}<contact>
             <exclude name="ab" body1="b" body2="a"/>
             <exclude body1="world" body2="b"/>
           </contact>"#
    )));
    assert_eq!(s.contact_excludes.len(), 2);
    let x = &s.contact_excludes[0];
    assert_eq!(x.name, "ab");
    // scene body 0 is `a`, scene body 1 is `b`; the written order is kept
    assert_eq!(
        (x.body1.map(|b| b.index()), x.body2.map(|b| b.index())),
        (Some(1), Some(0))
    );
    let y = &s.contact_excludes[1];
    assert_eq!(y.name, "");
    assert_eq!((y.body1, y.body2.map(|b| b.index())), (None, Some(1)));
    assert!(s.unsupported.is_empty(), "{:?}", s.unsupported);

    // refusals: an unknown body, the same body twice (also the world twice), a missing
    // attribute, a repeated name
    let bad = |contact: &str| mjcf_err(&wrap(&format!("{bodies}<contact>{contact}</contact>")));
    assert_eq!(
        bad(r#"<exclude body1="a" body2="nope"/>"#).kind,
        MjcfErrorKind::UnknownReference
    );
    assert_eq!(
        bad(r#"<exclude body1="a" body2="a"/>"#).kind,
        MjcfErrorKind::Inconsistent
    );
    assert_eq!(
        bad(r#"<exclude body1="world" body2="world"/>"#).kind,
        MjcfErrorKind::Inconsistent
    );
    assert_eq!(
        bad(r#"<exclude body1="a"/>"#).kind,
        MjcfErrorKind::MissingAttribute
    );
    assert_eq!(
        bad(
            r#"<exclude name="x" body1="a" body2="b"/><exclude name="x" body1="b" body2="world"/>"#
        )
        .kind,
        MjcfErrorKind::Duplicate
    );
}

#[test]
fn a_scene_with_a_bad_exclusion_is_refused_by_validation() {
    use sim_scene::{BodyId, ContactExclude};
    let mut s = load(&wrap(
        r#"<worldbody><body name="a"><freejoint/><geom size=".1"/></body></worldbody>"#,
    ));
    s.contact_excludes.push(ContactExclude {
        name: String::new(),
        body1: Some(BodyId(0)),
        body2: Some(BodyId(5)),
    });
    assert!(matches!(s.validate(), Err(SceneError::Invalid { .. })));
    s.contact_excludes[0].body2 = Some(BodyId(0));
    assert!(matches!(s.validate(), Err(SceneError::Invalid { .. })));
    s.contact_excludes[0].body2 = None;
    s.validate().unwrap();
    // and the JSON of a valid one round-trips
    assert_eq!(Scene::from_json(&s.to_json().unwrap()).unwrap(), s);
}
