//! Parity of the MJCF camera import and of the model statistic with MuJoCo.
//!
//! `fixtures/mujoco/cameras.xml` is ours (see its header); the golden values are
//! MuJoCo 3.14.0's, written by `tools/sim_scene_mujoco_golden.py` into
//! `humanoid_golden.json` (`camera_case`, and `statistic` for every fixture). Three
//! MuJoCo oracles hold the cameras, all on the CPU (no GL context is created):
//!
//! - the renderer's projection (`gl_uv`): where MuJoCo's OpenGL renderer draws a point,
//!   from the frustum `mjv_updateScene` gives the camera and the view `mjr_lookAt` sets
//!   (render_gl3.c `setView`), evaluated by the generator;
//! - `mjv_select`: MuJoCo's own ray through a pixel, and the point it hits (it agrees
//!   with the renderer for a `fovy` camera, and takes the half-width from the viewport
//!   aspect for a `sensorsize` camera, which the renderer does not);
//! - the camera-projection sensor (`cam_project`): the pixel of each site in each `fovy`
//!   camera with a resolution.
//!
//! Tolerances, from the arithmetic. The scene's intrinsics are `f32` (relative
//! rounding 6e-8: under 1e-5 pixel on these images). MuJoCo's renderer and `mjv_select`
//! place the camera in `f32` (`mjvGLCamera`): its position is off by up to a few `f32`
//! ulps of the coordinates (2.4e-7 m at 2.5 m), which moves a point at depth `z` by
//! `fx * 2.4e-7 / z` pixels. Only points the renderer draws are compared, those between
//! the clip planes (`z >= near`, 0.31 m here; nearer points are clipped, and there the
//! float position dominates: 1e-3 pixel at 1 cm); with `fx` at most 260 that bounds the
//! difference by 2e-4 pixel, under the 1e-3 pixel tolerance. Each camera must keep at
//! least 20 of its 25 samples. The camera pose is `f64` and is held to 1e-12. The clip
//! planes are compared bit for bit with MuJoCo's (`near = (float)(znear * extent)`), and
//! so is the model statistic.

use std::fs;
use std::path::PathBuf;

use serde_json::Value;
use sim_scene::mjcf::{self, LoadOptions, Statistic};
use sim_scene::{Camera, CameraMount, MjcfErrorKind, Scene, SceneError};

const PIXEL_TOL: f64 = 1e-3;
const POSE_TOL: f64 = 1e-12;
/// Of each camera's 25 samples, at least this many must lie between its clip planes.
const MIN_DRAWN_SAMPLES: usize = 20;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mujoco")
}

fn golden() -> Value {
    let text = fs::read_to_string(dir().join("humanoid_golden.json")).expect("golden file");
    serde_json::from_str(&text).expect("golden JSON")
}

fn fixture(name: &str) -> String {
    fs::read_to_string(dir().join(name)).expect("fixture")
}

/// FNV-1a, 64 bit: the same function the generator uses.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

fn f(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

fn farr<const N: usize>(v: &Value) -> [f64; N] {
    let a = v.as_array().expect("array");
    assert_eq!(a.len(), N);
    let mut out = [0.0; N];
    for (o, x) in out.iter_mut().zip(a) {
        *o = f(x);
    }
    out
}

fn load_stat(xml: &str) -> (Scene, Statistic) {
    mjcf::load_with_statistic(xml, dir(), &LoadOptions::default()).expect("imports")
}

/// The rotation matrix (row-major) of a unit quaternion `[x, y, z, w]`.
fn mat(q: [f64; 4]) -> [[f64; 3]; 3] {
    let [x, y, z, w] = q;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn apply(a: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2])
}

/// The world pose (position, rotation whose columns are the camera's x, y, z axes) of a
/// scene camera, its body placed where MuJoCo puts it at `qpos0`.
fn world_pose(cam: &Camera, gold: &Value) -> ([f64; 3], [[f64; 3]; 3]) {
    match cam.mount {
        CameraMount::World { pos, quat } => (pos, mat(quat)),
        CameraMount::Body {
            local_pos,
            local_quat,
            ..
        } => {
            let bpos = farr::<3>(&gold["body_xpos"]);
            let brot = mat(farr::<4>(&gold["body_xquat"]));
            let p = apply(&brot, local_pos);
            (
                [bpos[0] + p[0], bpos[1] + p[1], bpos[2] + p[2]],
                mul(&brot, &mat(local_quat)),
            )
        }
    }
}

/// The camera-frame coordinates of the world point `p`.
fn to_camera(pose: &([f64; 3], [[f64; 3]; 3]), p: [f64; 3]) -> [f64; 3] {
    let (t, r) = pose;
    let d = [p[0] - t[0], p[1] - t[1], p[2] - t[2]];
    [0, 1, 2].map(|j| r[0][j] * d[0] + r[1][j] * d[1] + r[2][j] * d[2])
}

/// Pixel coordinates (centres at integers) of the world point `p` in the scene camera.
fn project(cam: &Camera, pose: &([f64; 3], [[f64; 3]; 3]), p: [f64; 3]) -> [f64; 2] {
    let c = to_camera(pose, p);
    [
        f64::from(cam.fx) * c[0] / c[2] + f64::from(cam.cx),
        f64::from(cam.fy) * c[1] / c[2] + f64::from(cam.cy),
    ]
}

fn imported(g: &Value) -> Vec<&Value> {
    g["camera_case"]["cameras"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["mode"] == 0 && c["projection"] == 0)
        .collect()
}

fn camera<'s>(scene: &'s Scene, name: &str) -> &'s Camera {
    scene
        .cameras
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("camera {name} is not imported"))
}

#[test]
fn the_camera_fixture_matches_its_golden_record() {
    let g = golden();
    let case = &g["camera_case"];
    let bytes = fs::read(dir().join("cameras.xml")).unwrap();
    assert_eq!(case["xml"]["bytes"].as_u64().unwrap(), bytes.len() as u64);
    assert_eq!(
        case["xml"]["fnv1a64"].as_str().unwrap(),
        format!("{:016x}", fnv1a64(&bytes)),
        "cameras.xml changed without regenerating the golden file (tools/sim_scene_mujoco_golden.py)"
    );
    assert_eq!(g["mujoco_version"], "3.14.0");
}

#[test]
fn the_model_statistic_matches_mujoco_bit_for_bit() {
    // setStat's extent, mean body size and box centre, on every fixture: the humanoid
    // (whose <statistic center> overrides the computed centre), the mesh case (MuJoCo's
    // mesh bounding radius and its mesh-centred geom frame), the constrained model,
    // the contact zoo (planes, a cylinder, ellipsoids) and the camera fixture (sites,
    // walls of boxes, every joint type)
    let g = golden();
    let cases = [
        ("humanoid.xml", &g["statistic"]),
        ("mesh_case.xml", &g["mesh_case"]["statistic"]),
        ("constrained.xml", &g["constrained_case"]["statistic"]),
        ("contact_zoo.xml", &g["contact_case"]["statistic"]),
        ("cameras.xml", &g["camera_case"]["statistic"]),
    ];
    for (file, gold) in cases {
        let (_, stat) = load_stat(&fixture(file));
        let want = Statistic {
            extent: f(&gold["extent"]),
            meansize: f(&gold["meansize"]),
            center: farr::<3>(&gold["center"]),
        };
        println!(
            "{file}: extent {} meansize {} center {:?}",
            stat.extent, stat.meansize, stat.center
        );
        assert_eq!(
            stat.extent.to_bits(),
            want.extent.to_bits(),
            "{file}: extent {} vs MuJoCo {}",
            stat.extent,
            want.extent
        );
        assert_eq!(
            stat.meansize.to_bits(),
            want.meansize.to_bits(),
            "{file}: meansize {} vs MuJoCo {}",
            stat.meansize,
            want.meansize
        );
        for k in 0..3 {
            assert_eq!(
                stat.center[k].to_bits(),
                want.center[k].to_bits(),
                "{file}: center {:?} vs MuJoCo {:?}",
                stat.center,
                want.center
            );
        }
    }
}

#[test]
fn fixed_cameras_have_mujocos_pose_image_and_clip_planes() {
    let g = golden();
    let (scene, _) = load_stat(&fixture("cameras.xml"));
    let gold = imported(&g);
    // the fixed perspective cameras, in MuJoCo's order
    let names: Vec<&str> = scene.cameras.iter().map(|c| c.name.as_str()).collect();
    let want: Vec<&str> = gold.iter().map(|c| c["name"].as_str().unwrap()).collect();
    assert_eq!(names, want);
    let mut worst_pose = 0.0f64;
    for gc in gold {
        let name = gc["name"].as_str().unwrap();
        let cam = camera(&scene, name);
        // the mount: MuJoCo's local frame, turned by half a turn about x (exact)
        let [x, y, z, w] = farr::<4>(&gc["quat"]);
        let half_turn = [w, z, -y, -x];
        let pos = farr::<3>(&gc["pos"]);
        match (cam.mount, gc["body"].as_str().unwrap()) {
            (CameraMount::World { pos: p, quat }, "world") => {
                assert_eq!(p, pos, "{name}");
                assert_eq!(quat, half_turn, "{name}");
            }
            (
                CameraMount::Body {
                    body,
                    local_pos,
                    local_quat,
                },
                body_name,
            ) => {
                assert_eq!(scene.bodies[body.index()].name, body_name, "{name}");
                assert_eq!(local_pos, pos, "{name}");
                assert_eq!(local_quat, half_turn, "{name}");
            }
            (mount, body) => panic!("{name}: mount {mount:?}, MuJoCo's body {body}"),
        }
        // the image and the clip planes (bit for bit: (float)(znear * extent))
        let image = farr::<2>(&gc["image"]);
        assert_eq!(
            [f64::from(cam.width), f64::from(cam.height)],
            image,
            "{name}"
        );
        assert_eq!(cam.near, f(&gc["frustum"]["near"]) as f32, "{name} near");
        assert_eq!(cam.far, f(&gc["frustum"]["far"]) as f32, "{name} far");
        // the world pose against MuJoCo's cam_xpos and cam_xmat (columns x, y, z of
        // MuJoCo's frame; the scene's y and z are their negatives)
        let (t, r) = world_pose(cam, gc);
        let xpos = farr::<3>(&gc["cam_xpos"]);
        let xmat = farr::<9>(&gc["cam_xmat"]);
        for i in 0..3 {
            worst_pose = worst_pose.max((t[i] - xpos[i]).abs());
            for (j, sign) in [1.0, -1.0, -1.0].into_iter().enumerate() {
                worst_pose = worst_pose.max((r[i][j] - sign * xmat[3 * i + j]).abs());
            }
        }
    }
    println!("camera world pose: worst difference from MuJoCo {worst_pose:.2e}");
    assert!(worst_pose <= POSE_TOL, "{worst_pose}");
}

#[test]
fn every_point_lands_where_mujocos_renderer_draws_it() {
    let g = golden();
    let (scene, _) = load_stat(&fixture("cameras.xml"));
    let mut worst_gl = 0.0f64;
    let mut worst_select_fovy = 0.0f64;
    let mut select_vs_renderer_sensor = 0.0f64;
    let mut n = 0;
    for gc in imported(&g) {
        let name = gc["name"].as_str().unwrap();
        let cam = camera(&scene, name);
        let pose = world_pose(cam, gc);
        let sensor = farr::<2>(&gc["sensorsize"])[1] != 0.0;
        // only what the renderer draws: points between the clip planes
        let drawn: Vec<&Value> = gc["samples"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|s| {
                let z = to_camera(&pose, farr::<3>(&s["point"]))[2];
                z >= f64::from(cam.near) && z <= f64::from(cam.far)
            })
            .collect();
        assert!(
            drawn.len() >= MIN_DRAWN_SAMPLES,
            "{name}: only {} samples between the clip planes",
            drawn.len()
        );
        for s in drawn {
            let uv = project(cam, &pose, farr::<3>(&s["point"]));
            let gl = farr::<2>(&s["gl_uv"]);
            let pixel = farr::<2>(&s["pixel"]);
            let d_gl = (uv[0] - gl[0]).abs().max((uv[1] - gl[1]).abs());
            let d_select = (uv[0] - pixel[0]).abs().max((uv[1] - pixel[1]).abs());
            worst_gl = worst_gl.max(d_gl);
            if sensor {
                select_vs_renderer_sensor = select_vs_renderer_sensor.max(d_select);
            } else {
                worst_select_fovy = worst_select_fovy.max(d_select);
            }
            n += 1;
        }
    }
    println!(
        "{n} samples: worst difference from the renderer {worst_gl:.2e} px; from mjv_select's pixel (fovy cameras) {worst_select_fovy:.2e} px; mjv_select vs the renderer on sensorsize cameras {select_vs_renderer_sensor:.2} px (MuJoCo's own disagreement)"
    );
    assert!(n >= 6 * MIN_DRAWN_SAMPLES, "{n}");
    assert!(worst_gl <= PIXEL_TOL, "{worst_gl}");
    assert!(worst_select_fovy <= PIXEL_TOL, "{worst_select_fovy}");
    // pinned: MuJoCo's picking does not agree with its renderer on a sensorsize camera
    // whose pixels are not square in focal terms (the import follows the renderer)
    assert!(
        select_vs_renderer_sensor > 1.0,
        "{select_vs_renderer_sensor}"
    );
}

#[test]
fn the_camera_projection_sensor_agrees_on_fovy_cameras() {
    let g = golden();
    let case = &g["camera_case"];
    let (scene, _) = load_stat(&fixture("cameras.xml"));
    let sites: Vec<(&str, [f64; 3])> = case["sites"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| (s["name"].as_str().unwrap(), farr::<3>(&s["xpos"])))
        .collect();
    let mut worst = 0.0f64;
    let records = case["camprojection"].as_array().unwrap();
    assert_eq!(records.len(), 6);
    for r in records {
        let cname = r["camera"].as_str().unwrap();
        let gc = imported(&g)
            .into_iter()
            .find(|c| c["name"] == cname)
            .unwrap();
        let cam = camera(&scene, cname);
        let site = sites.iter().find(|s| s.0 == r["site"]).unwrap().1;
        let uv = project(cam, &world_pose(cam, gc), site);
        // the sensor's pixels have their origin at the image corner
        let edge = farr::<2>(&r["uv_edge"]);
        worst = worst
            .max((uv[0] + 0.5 - edge[0]).abs())
            .max((uv[1] + 0.5 - edge[1]).abs());
    }
    println!("camera-projection sensor: worst difference {worst:.2e} px");
    assert!(worst <= PIXEL_TOL, "{worst}");
}

#[test]
fn a_statistic_override_sets_the_clip_planes() {
    let g = golden();
    let case = &g["camera_case"]["override"];
    let xml = fixture("cameras.xml").replacen(
        "<worldbody>",
        "<statistic extent=\"5\" meansize=\"0.3\" center=\"0.5 -0.25 1\"/>\n  <worldbody>",
        1,
    );
    let (scene, stat) = load_stat(&xml);
    assert_eq!(stat.extent, f(&case["statistic"]["extent"]));
    assert_eq!(stat.meansize, f(&case["statistic"]["meansize"]));
    assert_eq!(stat.center, farr::<3>(&case["statistic"]["center"]));
    for cam in &scene.cameras {
        let [near, far] = farr::<2>(&case["frustum_near_far"][cam.name.as_str()]);
        assert_eq!(cam.near, near as f32, "{}", cam.name);
        assert_eq!(cam.far, far as f32, "{}", cam.name);
    }
}

#[test]
fn tracking_and_orthographic_cameras_are_recorded_and_strict_refuses_them() {
    let (scene, _) = load_stat(&fixture("cameras.xml"));
    for (path, why) in [
        ("worldbody/camera[ortho]", "orthographic"),
        (
            "worldbody/body[base]/body[arm]/body[wrist]/body[slider]/camera[follow]",
            "trackcom",
        ),
    ] {
        let u = scene
            .unsupported
            .iter()
            .find(|u| u.path == path)
            .unwrap_or_else(|| panic!("{path} is not recorded: {:?}", scene.unsupported));
        assert_eq!(u.item, "element");
        assert!(u.reason.contains(why), "{u:?}");
    }
    assert!(
        scene
            .cameras
            .iter()
            .all(|c| c.name != "ortho" && c.name != "follow")
    );
    // the sites are recorded too (read only for the extent)
    for site in ["outside", "strut", "tip"] {
        assert!(
            scene
                .unsupported
                .iter()
                .any(|u| u.path.ends_with(&format!("site[{site}]")) && u.item == "element"),
            "{site}"
        );
    }
    let strict = LoadOptions {
        strict: true,
        instances: true,
    };
    let e = match mjcf::load_with(
        r#"<mujoco><worldbody><camera name="t" mode="track"/></worldbody></mujoco>"#,
        dir(),
        &strict,
    ) {
        Err(SceneError::Mjcf(e)) => e,
        other => panic!("{other:?}"),
    };
    assert_eq!(e.kind, MjcfErrorKind::Strict);
    // a fixed camera in a scene with nothing else to record imports under strict
    let s = mjcf::load_with(
        r#"<mujoco><visual><map znear="0.1"/></visual><worldbody><geom size="1"/><camera name="c" pos="0 -3 0" xyaxes="1 0 0 0 0 1"/></worldbody></mujoco>"#,
        dir(),
        &strict,
    )
    .expect("imports under strict");
    assert_eq!(s.cameras.len(), 1);
}

fn mjcf_err(xml: &str) -> sim_scene::MjcfError {
    match mjcf::load(xml, dir()) {
        Err(SceneError::Mjcf(e)) => e,
        other => panic!("expected an MJCF error, got {other:?}"),
    }
}

fn wrap(body: &str) -> String {
    format!("<mujoco><worldbody><geom size=\"1\"/>{body}</worldbody></mujoco>")
}

#[test]
fn what_mujoco_refuses_in_a_camera_or_site_is_refused() {
    // MuJoCo 3.14.0's own errors (measured with tools/probe_mujoco_cameras.py, or read
    // in user_objects.cc)
    for (xml, kind, text) in [
        (
            wrap(r#"<camera fovy="60" sensorsize="0.01 0.01" focal="0.01 0.01"/>"#),
            MjcfErrorKind::Inconsistent,
            "at most one of 'fovy', 'sensorsize'",
        ),
        (
            wrap(r#"<camera focal="0.008 0.008"/>"#),
            MjcfErrorKind::Inconsistent,
            "require sensorsize",
        ),
        (
            wrap(r#"<camera sensorsize="0.01 0.01" focal="0.01 0.01" resolution="0 10"/>"#),
            MjcfErrorKind::Inconsistent,
            "requires positive resolution",
        ),
        (
            wrap(r#"<camera fovy="180"/>"#),
            MjcfErrorKind::BadValue,
            "fovy too large",
        ),
        (
            wrap(r#"<camera mode="targetbody" target="nobody"/>"#),
            MjcfErrorKind::UnknownReference,
            "unknown target body",
        ),
        (
            wrap(r#"<camera name="a"/><camera name="a"/>"#),
            MjcfErrorKind::Duplicate,
            "repeated camera name",
        ),
        (
            "<mujoco><default><camera name=\"a\"/></default></mujoco>".to_string(),
            MjcfErrorKind::UnsupportedAttribute,
            "not allowed in a <default>",
        ),
        (
            wrap(r#"<site type="plane"/>"#),
            MjcfErrorKind::Inconsistent,
            "planes not allowed in site",
        ),
        (
            wrap(r#"<site fromto="0 0 0 1 0 0"/>"#),
            MjcfErrorKind::Inconsistent,
            "fromto requires",
        ),
        (
            wrap(r#"<site type="capsule" pos="1 0 0" fromto="0 0 0 1 0 0"/>"#),
            MjcfErrorKind::Inconsistent,
            "both pos and fromto",
        ),
        (
            wrap(r#"<site type="box" size="0.1 0 0.1"/>"#),
            MjcfErrorKind::Inconsistent,
            "size 1 must be positive",
        ),
        (
            wrap(r#"<site material="nothing"/>"#),
            MjcfErrorKind::UnknownReference,
            "unknown material",
        ),
        (
            wrap(r#"<site name="s"/><site name="s"/>"#),
            MjcfErrorKind::Duplicate,
            "repeated site name",
        ),
        (
            "<mujoco><statistic extent=\"0\"/></mujoco>".to_string(),
            MjcfErrorKind::BadValue,
            "extent must be strictly positive",
        ),
    ] {
        let e = mjcf_err(&xml);
        assert_eq!(e.kind, kind, "{xml}: {e}");
        assert!(e.message.contains(text), "{xml}: {e}");
    }
}

#[test]
fn what_the_scene_cannot_draw_as_a_pinhole_is_refused() {
    // MuJoCo compiles these, but its renderer would divide by zero or draw nothing; the
    // importer refuses them, and refuses an override of the solver's inertia scale
    for (xml, text) in [
        (
            wrap(r#"<camera sensorsize="0.01 0.01" resolution="10 10"/>"#),
            "positive focal lengths",
        ),
        (
            wrap(r#"<camera sensorsize="0.01 0"/>"#),
            "must be positive in both directions",
        ),
        (wrap(r#"<camera fovy="0"/>"#), "fovy must be positive"),
        (
            wrap(r#"<camera resolution="-4 300"/>"#),
            "resolution [-4, 300] must be positive",
        ),
        (
            format!(
                "<mujoco><visual><global offwidth=\"0\"/></visual>{}</mujoco>",
                r#"<worldbody><geom size="1"/><camera/></worldbody>"#
            ),
            "offscreen buffer",
        ),
        (
            format!(
                "<mujoco><visual><map znear=\"2\" zfar=\"1\"/></visual>{}</mujoco>",
                r#"<worldbody><geom size="1"/><camera/></worldbody>"#
            ),
            "0 < near < far",
        ),
        (
            "<mujoco><statistic meaninertia=\"2\"/></mujoco>".to_string(),
            "meaninertia",
        ),
        (wrap(r#"<site type="sphere" mesh="m"/>"#), "mesh"),
        (
            "<mujoco><visual><bogus/></visual></mujoco>".to_string(),
            "<bogus> in <visual>",
        ),
    ] {
        let e = mjcf_err(&xml);
        assert!(e.message.contains(text), "{xml}: {e}");
    }
}

#[test]
fn class_defaults_reach_cameras_and_a_class_fovy_does_not_clash_with_an_element_sensor() {
    // the schema's fovy/sensorsize exclusion is per element, as MuJoCo checks it: a
    // class's fovy and an element's sensorsize make a sensorsize camera
    let xml = r#"<mujoco><default><camera fovy="30" resolution="40 30"/></default>
        <worldbody><geom size="1"/>
          <camera name="a" pos="0 -3 0"/>
          <camera name="b" pos="0 -3 0" sensorsize="0.004 0.003" focal="0.004 0.004"/>
        </worldbody></mujoco>"#;
    let scene = mjcf::load(xml, dir()).expect("imports");
    let a = camera(&scene, "a");
    assert_eq!((a.width, a.height), (40, 30));
    let fy = 30.0 / (2.0 * (30.0f64.to_radians() / 2.0).tan());
    assert!((f64::from(a.fy) - fy).abs() < 1e-4, "{}", a.fy);
    let b = camera(&scene, "b");
    assert_eq!((b.width, b.height), (40, 30));
    assert!((f64::from(b.fx) - 40.0).abs() < 1e-4, "{}", b.fx);
    assert!((f64::from(b.fy) - 40.0).abs() < 1e-4, "{}", b.fy);
}

#[test]
fn negative_control_a_changed_camera_fails_the_parity_comparison() {
    // the comparisons are not vacuous: a 1-degree fovy change on base_cam moves its
    // projected points by pixels, and moving the outside site moves the extent
    let g = golden();
    let original = fixture("cameras.xml");
    let changed = original.replace(r#"fovy="70""#, r#"fovy="71""#);
    assert_ne!(changed, original);
    let (scene, _) = load_stat(&changed);
    let gc = imported(&g)
        .into_iter()
        .find(|c| c["name"] == "base_cam")
        .unwrap();
    let cam = camera(&scene, "base_cam");
    let pose = world_pose(cam, gc);
    let worst = gc["samples"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            let uv = project(cam, &pose, farr::<3>(&s["point"]));
            let gl = farr::<2>(&s["gl_uv"]);
            (uv[0] - gl[0]).abs().max((uv[1] - gl[1]).abs())
        })
        .fold(0.0, f64::max);
    assert!(
        worst > 0.5,
        "a 1-degree fovy change moved points by {worst} px"
    );

    let moved = original.replace(r#"pos="9 0 1""#, r#"pos="9.5 0 1""#);
    assert_ne!(moved, original);
    let (_, stat) = load_stat(&moved);
    assert!(
        (stat.extent - f(&g["camera_case"]["statistic"]["extent"])).abs() > 0.4,
        "{}",
        stat.extent
    );
}
