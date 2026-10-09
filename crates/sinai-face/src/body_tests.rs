use super::*;

/// The Python bake's own answers for a few shapes. Computed there from the
/// same file this module reads, by code written separately; agreement is the
/// evidence that the window draws the shape the bake measured.
const GOLDEN: &str = include_str!("testdata/body_golden.json");

fn body() -> &'static Body {
    static BODY: std::sync::OnceLock<Body> = std::sync::OnceLock::new();
    BODY.get_or_init(Body::load)
}

fn weights_of(b: &Body, case: &serde_json::Value) -> Vec<(usize, f32)> {
    case["weights"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, w)| {
            let t = b.target(name).unwrap_or_else(|| panic!("the golden names {name}, which the file lacks"));
            (t, w.as_f64().unwrap() as f32)
        })
        .collect()
}

fn close(got: f32, want: f64, tol: f64, what: &str) {
    assert!(
        (got as f64 - want).abs() <= tol,
        "{what}: got {got}, the bake computed {want} (tolerance {tol})"
    );
}

#[test]
fn the_window_poses_every_golden_shape_the_way_the_bake_did() {
    let b = body();
    let golden: serde_json::Value = serde_json::from_str(GOLDEN).unwrap();
    let cases = golden["cases"].as_array().unwrap();
    assert!(cases.len() >= 5, "the golden file lost its cases");
    for (k, case) in cases.iter().enumerate() {
        let posed = b.pose(&weights_of(b, case));
        for (v, want) in case["vertices"].as_object().unwrap() {
            let v: usize = v.parse().unwrap();
            for axis in 0..3 {
                close(
                    posed.positions[v][axis],
                    want[axis].as_f64().unwrap(),
                    1e-5,
                    &format!("case {k}, vertex {v}, axis {axis}"),
                );
            }
        }
        for (v, want) in case["normals"].as_object().unwrap() {
            let v: usize = v.parse().unwrap();
            for axis in 0..3 {
                close(
                    posed.normals[v][axis],
                    want[axis].as_f64().unwrap(),
                    1e-4,
                    &format!("case {k}, normal {v}, axis {axis}"),
                );
            }
        }
        for (v, want) in case["ao"].as_object().unwrap() {
            let v: usize = v.parse().unwrap();
            close(posed.ao[v], want.as_f64().unwrap(), 2e-4, &format!("case {k}, occlusion {v}"));
        }
        for (side, key) in ["eye_l", "eye_r"].iter().enumerate() {
            for c in 0..4 {
                close(
                    posed.rig.eyes[side][c],
                    case[*key][c].as_f64().unwrap(),
                    1e-5,
                    &format!("case {k}, {key}[{c}]"),
                );
            }
        }
        for c in 0..2 {
            close(posed.rig.hinge[c], case["hinge"][c].as_f64().unwrap(), 1e-5, &format!("case {k}, hinge[{c}]"));
        }
    }
}

#[test]
fn the_base_shape_reproduces_the_baked_normals_and_occlusion() {
    // The window recomputes both on every change. If its arithmetic differed
    // from the bake's, the shading would jump the first time anything moved.
    let b = body();
    let posed = b.pose(&[]);
    let mut worst_n = 0.0f32;
    let mut worst_ao = 0.0f32;
    for v in 0..b.n_draw {
        let n = b.base_normal(v);
        for axis in 0..3 {
            worst_n = worst_n.max((posed.normals[v][axis] - n[axis]).abs());
        }
        worst_ao = worst_ao.max((posed.ao[v] - b.attrs(v)[0]).abs());
    }
    assert!(worst_n < 1e-4, "normals differ from the bake's by up to {worst_n}");
    assert!(worst_ao < 2e-4, "occlusion differs from the bake's by up to {worst_ao}");
}

#[test]
fn the_head_stays_where_the_scene_puts_it() {
    // Lengthening the neck moves the joint the head turns on. The face must not
    // rise out of the frame; the shoulders drop instead.
    let b = body();
    let rest = b.pose(&[]);
    let neck = b.target("neck/neck-scale-vert-incr").unwrap();
    let long = b.pose(&[(neck, 1.0)]);
    for side in 0..2 {
        for c in 0..3 {
            assert!(
                (long.rig.eyes[side][c] - rest.rig.eyes[side][c]).abs() < 1e-4,
                "a longer neck moved the eyes: {:?} -> {:?}",
                rest.rig.eyes[side],
                long.rig.eyes[side]
            );
        }
    }
    let lowest = |p: &Posed| p.positions.iter().map(|q| q[1]).fold(f32::INFINITY, f32::min);
    assert!(
        lowest(&long) < lowest(&rest) - 0.3,
        "a longer neck did not lower the shoulders: {} -> {}",
        lowest(&rest),
        lowest(&long)
    );
}

#[test]
fn the_eyeballs_sit_on_the_eyes_they_belong_to() {
    // MakeHuman's eye-size targets open or close the lids around an eyeball
    // that keeps its size, which is anatomy: adult eyeballs barely differ, and
    // eyes that look large are large openings. What resizes the eyeballs is the
    // head's width, uniformly. A raised eye carries its eyeball with it.
    let b = body();
    let rest = b.pose(&[]);
    let size = b.target("eyes/l-eye-scale-incr").unwrap();
    let up = b.target("eyes/r-eye-trans-up").unwrap();
    let posed = b.pose(&[(size, 1.0), (up, 1.0)]);
    for side in 0..2 {
        let (start, n) = b.eye_ranges[side];
        let e = posed.rig.eyes[side];
        for k in start..start + n {
            let p = posed.positions[k];
            let d = ((p[0] - e[0]).powi(2) + (p[1] - e[1]).powi(2) + (p[2] - e[2]).powi(2)).sqrt();
            assert!((d - e[3]).abs() < 1e-4, "eyeball vertex {k} is {d} from its centre, radius {}", e[3]);
        }
    }
    assert!((posed.rig.eyes[0][3] - rest.rig.eyes[0][3]).abs() < 1e-4, "opening the left lids resized its eyeball");
    let lids_moved = (0..b.n_draw)
        .filter(|&v| b.attrs(v)[3] == PART_SKIN && rest.positions[v][0] > 0.2)
        .any(|v| (posed.positions[v][1] - rest.positions[v][1]).abs() > 0.01);
    assert!(lids_moved, "a larger left eye moved none of the skin around it");
    assert!(posed.rig.eyes[1][1] > rest.rig.eyes[1][1] + 0.05, "a raised right eye left its eyeball behind");
    assert!((posed.rig.eyes[1][3] - rest.rig.eyes[1][3]).abs() < 1e-4, "raising the right eye resized it");
    let wide = b.pose(&[(b.target("head/head-scale-horiz-incr").unwrap(), 1.0)]);
    for side in 0..2 {
        let grown = wide.rig.eyes[side][3] / rest.rig.eyes[side][3];
        assert!((grown - 1.2).abs() < 0.02, "a wider head grew eyeball {side} by {grown}, not about a fifth");
    }
}

#[test]
fn nothing_on_the_midline_belongs_to_an_eyelid() {
    // The head bake mirrored one eye centre with sign(x), and sign(0) is 0, so
    // six vertices on the bridge of the nose were lid and dipped with every blink.
    let b = body();
    let posed = b.pose(&[]);
    for v in 0..b.n_draw {
        if posed.positions[v][0] == 0.0 {
            assert_eq!(b.attrs(v)[2], 0.0, "midline vertex {v} carries a lid weight");
        }
    }
}

#[test]
fn breathing_moves_the_body_and_never_the_face() {
    let b = body();
    let posed = b.pose(&[]);
    let chin = 1.26 - 2.52; // the head ends here, in head units (bake_body.py: HEAD_TOP - HEAD_H)
    for v in 0..b.n_draw {
        let a = b.attrs(v);
        assert!((0.0..=1.0).contains(&a[5]) && (0.0..=1.0).contains(&a[6]), "breath weights out of range at {v}");
        if posed.positions[v][1] > chin {
            assert_eq!((a[5], a[6]), (0.0, 0.0), "vertex {v}, above the chin, moves with a breath");
        }
    }
    assert!((0..b.n_draw).filter(|&v| b.attrs(v)[5] > 0.5).count() > 300, "almost nothing swells");
    assert!((0..b.n_draw).filter(|&v| b.attrs(v)[6] > 0.5).count() > 500, "almost nothing lifts");
}

#[test]
fn the_weights_the_shader_multiplies_into_angles_stay_in_range() {
    let b = body();
    for v in 0..b.n_draw {
        let a = b.attrs(v);
        assert!((0.0..=1.0).contains(&a[1]), "jaw weight {} at {v}", a[1]);
        assert!((0.0..=1.0).contains(&a[2]), "lid weight {} at {v}", a[2]);
        assert!((0.0..=1.0).contains(&a[4]), "fade {} at {v}", a[4]);
        if a[3] != PART_SKIN {
            assert_eq!(a[2], 0.0, "part {} vertex {v} is weighted as a lid", a[3]);
        }
        if a[3] == PART_BALL || a[3] == PART_TEETH_U {
            assert_eq!(a[1], 0.0, "part {} vertex {v} swings with the jaw", a[3]);
        }
        if a[3] == PART_TEETH_L {
            assert_eq!(a[1], 1.0, "lower tooth vertex {v} does not swing fully with the jaw");
        }
    }
}

#[test]
fn the_highlight_marks_what_a_control_moves_and_nothing_else() {
    let b = body();
    let t = b.target("nose/nose-width1-incr").unwrap();
    let h = b.influence(&[t]);
    assert_eq!(h.len(), b.n_draw);
    assert!((h.iter().copied().fold(0.0f32, f32::max) - 1.0).abs() < 1e-6);
    let lit = h.iter().filter(|&&x| x > 0.0).count();
    assert!(lit > 10 && lit < 200, "a nose-bridge control lights {lit} vertices");
    let data = b.vertex_data(&b.pose(&[]), Some(&h));
    assert_eq!(data.len(), b.n_draw * FLOATS_PER_VERTEX);
    for v in 0..b.n_draw {
        assert_eq!(data[v * FLOATS_PER_VERTEX + 13], h[v]);
    }
}

#[test]
fn a_damaged_file_is_refused_rather_than_drawn() {
    let good = BAKED.to_vec();
    assert!(Body::parse(&good).is_ok());
    assert!(Body::parse(&good[..good.len() - 1]).is_err(), "a truncated file loaded");
    let mut magic = good.clone();
    magic[0] = b'X';
    assert!(Body::parse(&magic).is_err(), "a file with the wrong magic loaded");
    let mut longer = good.clone();
    longer.push(0);
    assert!(Body::parse(&longer).is_err(), "a file with trailing bytes loaded");
    // A vertex count larger than the file: must fail on the count, not allocate.
    let mut huge = good.clone();
    huge[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(Body::parse(&huge).is_err(), "an impossible vertex count loaded");
    // An edge index past the drawn vertices.
    let b = Body::parse(&good).unwrap();
    let n_all = b.base.len();
    let edges_at = HEADER + n_all * 12 + b.n_draw * 12 + b.n_draw * 32;
    let mut bad_edge = good.clone();
    bad_edge[edges_at..edges_at + 4].copy_from_slice(&(b.n_draw as u32).to_le_bytes());
    assert!(Body::parse(&bad_edge).is_err(), "an edge past the drawn vertices loaded");
}

/// What re-shaping costs the CPU on a frame in which a value or an expression
/// changes: the calls the window's shape block makes in `AngelApp::ui`, timed
/// here because this lane does not run the window. It is a microbenchmark, so
/// it is the floor of the cost in the window, which also uploads the vertices,
/// draws, and shares the CPU with everything else. A timing, not a check:
/// `cargo test --locked --release -- --ignored --nocapture reshaping_cost`.
#[test]
#[ignore = "a timing, not a check: run it on purpose"]
fn reshaping_cost_per_changed_frame() {
    use crate::appearance::Catalog;
    use crate::expression::Face;
    use std::hint::black_box;

    let b = body();
    let catalog = Catalog::builtin();
    let (_, look) = catalog.looks.iter().find(|(label, _)| label == "Strong").expect("the look is gone");
    let mut face = Face::new(b);
    while face.ease_toward(b, "laugh", 1.0) {}
    let hovered = catalog.control("eyes.size").expect("the control is gone");
    let everything: Vec<(usize, f32)> = (0..b.target_count()).map(|t| (t, 1.0)).collect();

    let time = |what: &str, run: &mut dyn FnMut() -> Vec<f32>| {
        for _ in 0..20 {
            black_box(run());
        }
        let mut took: Vec<f64> = (0..300)
            .map(|_| {
                let started = std::time::Instant::now();
                black_box(run());
                started.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        took.sort_by(|x, y| x.total_cmp(y));
        println!(
            "{what}: median {:.3} ms, 95th percentile {:.3} ms, slowest {:.3} ms, over {} frames",
            took[took.len() / 2],
            took[took.len() * 95 / 100],
            took[took.len() - 1],
            took.len()
        );
    };
    time("at rest", &mut || b.vertex_data(&b.pose(&[]), None));
    time("the Strong look, a laugh, and the pointer on Eye size", &mut || {
        let mut weights = look.weights(catalog, |n| b.target(n));
        weights.extend(face.weights());
        let posed = b.pose(&weights);
        let targets: Vec<usize> = hovered.all_targets().iter().filter_map(|n| b.target(n)).collect();
        b.vertex_data(&posed, Some(&b.influence(&targets)))
    });
    time(&format!("all {} shapes at full weight", everything.len()), &mut || {
        b.vertex_data(&b.pose(&everything), None)
    });
}
