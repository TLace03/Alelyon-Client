//! The box colliders: sphere-box, capsule-box and box-box.
//!
//! Ports, from MuJoCo 3.14.0 `engine_collision_box.c` (Copyright 2016 Svetoslav Kolev,
//! Apache-2.0; not a DeepMind file): `mju_clampVec`, `mjraw_SphereBox` /
//! `mjc_SphereBox`, `mjraw_CapsuleBox` / `mjc_CapsuleBox`, `clipHalfPlane` and
//! `mjc_BoxBox`. The box-box collider is MuJoCo's separating-axis test over the 15
//! candidate axes followed by Sutherland-Hodgman clipping of the incident face, or a
//! single edge-edge contact.
//!
//! Invariants:
//! - **Same arithmetic, same order as the C source**: the `mji_mulMatTVec3` and
//!   `mju_mulMatTMat3` summation orders, `1 / mju_sqrt(norm2)` multiplied in (not a
//!   division), `pos21` and `pos12` computed separately, `mju_clip` as its ternary, no
//!   fused multiply-add. Constants go through [`Real`]: `mjMINVAL` is `1e-15` in both
//!   precisions, the box-box epsilons are the double values in `f64` and the
//!   `mjUSESINGLE` values in `f32` (`Real::BOXBOX_*`).
//! - **Strict IEEE semantics are relied on.** `mjraw_CapsuleBox` divides by capsule-axis
//!   components that can be 0 and needs the comparisons with the resulting infinities and
//!   NaNs to be false (`e1 > 0`, `e1 < secondpos`). A GPU port must not use fast math.
//! - **Bounded loops only**: box-box tests 15 axes, at most 2 witness corners per box in
//!   the edge case (4 witness pairs), 4 clip passes over at most 12 vertices and a
//!   12-by-12 deduplication; capsule-box tests 2 end points and 12 edges and makes 2
//!   sphere-box calls.
//! - **Box-box overflow.** At most 8 contacts can result in exact arithmetic (a 4-gon
//!   clipped by 4 half-planes), and the pre-contact slice holds 8. MuJoCo calls `mjERROR`
//!   if more are returned; here a count above 8 raises a `debug_assert`, sets
//!   `overflow`, and keeps the first 8.
//!
//! In `f32`, the literals of the C source that MuJoCo's single-precision build evaluates in
//! double (`0.99`, `0.05`, `0.5`) are rounded to `f32` here, and the expression is
//! evaluated in `f32`; `f32` is measured against MuJoCo's double results, not held to them.
//!
//! Deviations from MuJoCo, each changing no `f64` result:
//! - The dead block of `mjraw_CapsuleBox` (the `j == 2` loop at `box.c:310-393`, which
//!   only writes locals that are never read again) is not ported.
//! - `mjraw_SphereBox`'s face index `k` starts at 0; it is uninitialised in C and
//!   reachable only with a NaN position.
//!
//! `f32` deviation (changes no `f64` result): `mjraw_CapsuleBox`'s two absolute tests
//! (`|det| < mjMINVAL`, which skips an edge parallel to the capsule axis, and the tie rule
//! `dist2 < bestdist - mjMINVAL`) gain a relative part ([`Real::PARALLEL_DET_REL`],
//! [`Real::TIE_REL`], zero in `f64`): in `f32` the determinant of an exactly parallel edge is a
//! rounding residue of about `1e-7 * ma * mc`, which the absolute `1e-15` does not skip, and the
//! general branch then puts the nearest point 0.1 m off. See `collide_primitive.rs`.

use crate::collide_primitive::{GeomPose, PreContact};
use crate::math::{
    add_to_scl3, add3, clip, dot3, lit, min_val, mul_mat_t_mat3, mul_mat_t_vec3, mul_mat_vec3,
    normalize3, scl3, sub3,
};
use crate::real::Real;

/// The most vertices a clipped incident face can have (`mjBOXBOX_MAXVERT`).
const MAXVERT: usize = 12;

/// Port of `mju_clampVec`: clamps `vec[i]` to `[-limit[i], limit[i]]` where `limit[i] >
/// 0`.
fn clamp_vec<R: Real>(vec: &mut [R; 3], limit: &[R; 3]) {
    for i in 0..3 {
        // loop over the active limits
        if limit[i] > R::ZERO {
            vec[i] = clip(vec[i], -limit[i], limit[i]);
        }
    }
}

/// Port of `mjraw_SphereBox`: a sphere (`pos1`, radius `size1[0]`) against a box (`pos2`,
/// `mat2`, half-sizes `size2`); a sphere centre inside the box goes to the nearest face.
pub(crate) fn raw_sphere_box<R: Real>(
    con: &mut [PreContact<R>],
    margin: R,
    pos1: [R; 3],
    size1: &[R; 3],
    pos2: [R; 3],
    mat2: &[R; 9],
    size2: &[R; 3],
) -> usize {
    let half = lit::<R>(0.5);
    let two = lit::<R>(2.0);

    let tmp = sub3(pos1, pos2);
    let center = mul_mat_t_vec3(mat2, tmp);

    let mut clamped = center;
    clamp_vec(&mut clamped, size2);

    let mut deepest = center;
    let mut tmp = sub3(clamped, center);
    let mut dist = normalize3(&mut tmp);

    if dist - size1[0] > margin {
        return 0;
    }

    let mut pos = [R::ZERO; 3];
    // the sphere centre is inside the box
    if dist <= min_val::<R>() {
        let mut closest = (size2[0] + size2[1] + size2[2]) * two;
        let mut k = 0usize;
        for i in 0..6usize {
            let face = if i % 2 != 0 {
                size2[i / 2]
            } else {
                -size2[i / 2]
            };
            if closest > (face - center[i / 2]).abs() {
                closest = (face - center[i / 2]).abs();
                k = i;
            }
        }

        let mut nearest = [R::ZERO; 3];
        nearest[k / 2] = if !k.is_multiple_of(2) {
            -R::ONE
        } else {
            R::ONE
        };

        pos = center;
        add_to_scl3(&mut pos, nearest, (size1[0] - closest) / two);
        con[0].normal = mul_mat_vec3(mat2, nearest);
        dist = -closest;
    } else {
        add_to_scl3(&mut deepest, tmp, size1[0]);
        add_to_scl3(&mut pos, clamped, half);
        add_to_scl3(&mut pos, deepest, half);
        con[0].normal = mul_mat_vec3(mat2, tmp);
    }

    let t = mul_mat_vec3(mat2, pos);
    con[0].pos = add3(t, pos2);
    con[0].dist = dist - size1[0];
    con[0].tangent = [R::ZERO; 3];
    1
}

/// Port of `mjc_SphereBox` (`engine_collision_box.c`): sphere against box.
pub fn sphere_box<R: Real>(
    con: &mut [PreContact<R>],
    margin: R,
    g1: &GeomPose<R>,
    g2: &GeomPose<R>,
) -> usize {
    raw_sphere_box(con, margin, g1.pos, &g1.size, g2.pos, &g2.mat, &g2.size)
}

/// Port of `mjraw_CapsuleBox`: a capsule against a box. The closest point between the
/// capsule's segment and the box is found (a face, or an edge of the box against the
/// segment), a sphere-box test is made there, and a second sphere-box test at a "sensible"
/// second point along the segment when the capsule lies at a low enough angle to the box.
// the argument list is MuJoCo's (`mjraw_CapsuleBox`), kept so that the port reads line by line
#[allow(clippy::too_many_arguments)]
pub(crate) fn raw_capsule_box<R: Real>(
    con: &mut [PreContact<R>],
    margin: R,
    pos1: [R; 3],
    mat1: &[R; 9],
    size1: &[R; 3],
    pos2: [R; 3],
    mat2: &[R; 9],
    size2: &[R; 3],
) -> usize {
    let one = R::ONE;
    let two = lit::<R>(2.0);

    let halflength = size1[1];
    let mut secondpos = lit::<R>(-4.0); // initialise to no 2nd contact (valid values are -1 to 1)

    // bring the capsule to the box-local frame (the box centre is at (0, 0, 0)), axes
    // parallel to the world
    let tmp1 = sub3(pos1, pos2);
    let pos = mul_mat_t_vec3(mat2, tmp1); // the capsule position in the box-local frame

    // the capsule's axis, in the same frame
    let axis = mul_mat_t_vec3(mat2, [mat1[2], mat1[5], mat1[8]]);
    // scale to get the actual capsule half-axis
    let halfaxis = scl3(axis, halflength);

    let mut axisdir = 0i32;
    if halfaxis[0] > R::ZERO {
        axisdir += 1;
    }
    if halfaxis[1] > R::ZERO {
        axisdir += 2;
    }
    if halfaxis[2] > R::ZERO {
        axisdir += 4;
    }

    // under this notion "axisdir" and "7 - axisdir" point in opposite directions,
    // essentially the same for a capsule

    // initialise bestdist
    let bestdistmax = margin + two * (size1[0] + halflength + size2[0] + size2[1] + size2[2]);
    let mut bestdist = bestdistmax;
    let mut bestsegmentpos = R::ZERO;

    let mut cltype = -4i32; // the closest type
    let mut clface = -1i32; // the closest face
    let mut clcorner = 0i32; // the closest corner (0..7 in binary)
    let mut cledge = 0usize; // the closest edge axis
    let mut bestboxpos = R::ZERO; // the closest contact point, position on the box's edge

    // test to see if maybe a face of the box is closest to the capsule
    let mut i = -1i32;
    while i <= 1 {
        let mut t1 = pos;
        add_to_scl3(&mut t1, halfaxis, lit::<R>(f64::from(i)));
        let t2 = t1;

        let mut c1 = 0;
        let mut c2 = -1i32;
        for j in 0..3usize {
            if t1[j] < -size2[j] {
                c1 += 1;
                c2 = j as i32;
                t1[j] = -size2[j];
            } else if t1[j] > size2[j] {
                c1 += 1;
                c2 = j as i32;
                t1[j] = size2[j];
            }
        }

        if c1 > 1 {
            i += 2;
            continue;
        }

        let t1 = sub3(t1, t2);
        let dist = dot3(t1, t1);

        if dist < bestdist {
            bestdist = dist;
            bestsegmentpos = lit::<R>(f64::from(i));
            cltype = -2 + i;
            clface = c2;
        }
        i += 2;
    }

    for j in 0..3usize {
        for i in 0..8i32 {
            if (i & (1 << j)) == 0 {
                // trick to get a corner
                let sgn3 = |bit: i32| if i & bit != 0 { one } else { -one };
                let mut tmp3 = [sgn3(1) * size2[0], sgn3(2) * size2[1], sgn3(4) * size2[2]];
                tmp3[j] = R::ZERO;

                // tmp3 is the starting point on the box, the direction along the "j"-th
                // axis is the box edge, pos is the capsule's centre and halfaxis is the
                // capsule direction: find the closest point between the capsule and the edge
                let mut dif = sub3(tmp3, pos);

                let ma = size2[j] * size2[j];
                let mb = -size2[j] * halfaxis[j];
                let mc = size1[1] * size1[1];

                let u = -size2[j] * dif[j];
                let v = dot3(halfaxis, dif);

                // the edge is parallel to the capsule axis: MuJoCo's `|det| < mjMINVAL`, plus the
                // f32 rounding floor of a determinant of parallel axes (0 in f64)
                let det = ma * mc - mb * mb;
                if det.abs() < min_val::<R>() + R::PARALLEL_DET_REL * (ma * mc) {
                    continue;
                }
                let idet = one / det;

                // sX: X = 1 means the middle of the segment, X = 0 or 2 one end or the other
                let mut x1 = (mc * u - mb * v) * idet;
                let mut x2 = (ma * v - mb * u) * idet;

                let mut s1 = 1i32;
                let mut s2 = 1i32;

                if x1 > one {
                    x1 = one;
                    s1 = 2;
                    x2 = (v - mb) * (one / mc);
                } else if x1 < -one {
                    x1 = -one;
                    s1 = 0;
                    x2 = (v + mb) * (one / mc);
                }

                if x2 > one {
                    x2 = one;
                    s2 = 2;
                    x1 = (u - mb) * (one / ma);
                    if x1 > one {
                        x1 = one;
                        s1 = 2;
                    } else if x1 < -one {
                        x1 = -one;
                        s1 = 0;
                    }
                } else if x2 < -one {
                    x2 = -one;
                    s2 = 0;
                    x1 = (u + mb) * (one / ma);
                    if x1 > one {
                        x1 = one;
                        s1 = 2;
                    } else if x1 < -one {
                        x1 = -one;
                        s1 = 0;
                    }
                }

                dif = sub3(tmp3, pos);
                add_to_scl3(&mut dif, halfaxis, -x2);
                dif[j] += size2[j] * x1;

                let dist2 = dot3(dif, dif);

                let c1 = s1 * 3 + s2;

                // the -mjMINVAL might not be necessary. It fixes a numerical problem when the
                // axis is numerically parallel to the box (f32: plus a relative slack, since
                // equal squared distances differ by rounding of that relative size)
                if dist2 < bestdist - (min_val::<R>() + R::TIE_REL * bestdist) {
                    bestdist = dist2;
                    bestsegmentpos = x2;
                    bestboxpos = x1;

                    // c1 < 6 means that the closest point on the box is at the lower end or
                    // in the middle of the edge
                    let c2 = c1 / 6;

                    clcorner = i + (1 << j) * c2; // which corner is the closest
                    cledge = j; // which axis
                    cltype = c1; // save the clamped info
                }
            }
        }
    }

    // cltype: -3, -1: a face is closest to the capsule
    // cltype: 0..8: an edge is closest to the capsule
    // cltype / 3 == 0: the lower corner is closest to the capsule (edges include corners)
    // cltype / 3 == 2: the upper corner is closest to the capsule
    // cltype / 3 == 1: the middle of the edge is closest to the capsule
    // cltype % 3 == 0: the lower corner is closest to the box
    // cltype % 3 == 2: the upper corner is closest to the box
    // cltype % 3 == 1: the middle of the capsule is closest to the box

    // invalid type
    if cltype == -4 {
        return 0;
    }

    // the "goto skip" of the C code: `break 'skip` leaves the second-point search
    'skip: {
        if cltype >= 0 && cltype / 3 != 1 {
            // closest to a corner of the box
            let mut c1 = axisdir ^ clcorner;

            // a hack to find the relative orientation of the capsule and the corner; there
            // are 2 cases: 1: pointing to or away from the corner, 2: oriented along a face
            // or an edge
            if c1 == 0 || c1 == 7 {
                break 'skip; // case 1: no chance of an additional contact
            }

            let mut mul = R::ZERO;
            let mut de = R::ZERO;
            let mut dp = R::ZERO;
            if c1 == 1 || c1 == 2 || c1 == 4 {
                mul = one;
                de = one - bestsegmentpos;
                dp = one + bestsegmentpos;
            }

            if c1 == 3 || c1 == 5 || c1 == 6 {
                mul = -one;
                c1 = 7 - c1;
                dp = one - bestsegmentpos;
                de = one + bestsegmentpos;
            }

            // "de" and "dp" are the distances from the first closest point on the capsule to
            // both ends of it; mul is a direction along the capsule's axis
            let (mut ax, mut ax1, mut ax2) = (0usize, 0usize, 0usize);
            if c1 == 1 {
                (ax, ax1, ax2) = (0, 1, 2);
            }
            if c1 == 2 {
                (ax, ax1, ax2) = (1, 2, 0);
            }
            if c1 == 4 {
                (ax, ax1, ax2) = (2, 0, 1);
            }

            if axis[ax] * axis[ax] > lit::<R>(0.5) {
                // the second point along the edge of the box
                secondpos = de; // the initial position from the
                let e1 = two * size2[ax] / halfaxis[ax].abs();

                if e1 < secondpos {
                    secondpos = e1; // we overshoot, move back to the other corner of the edge
                }
                secondpos *= mul;
            } else {
                // the second point along a face of the box
                secondpos = dp;

                // check for an overshoot again
                let e1 = two * size2[ax1] / halfaxis[ax1].abs();
                if e1 < secondpos {
                    secondpos = e1;
                }

                let e1 = two * size2[ax2] / halfaxis[ax2].abs();
                if e1 < secondpos {
                    secondpos = e1;
                }

                secondpos *= -mul;
            }
        } else if cltype >= 0 && cltype / 3 == 1 {
            // we are on the box's edge; hacks to find the relative orientation of the
            // capsule and the edge; there are 2 cases: c1 = 2^n: the edge and the capsule
            // are oriented in a T configuration (no more contacts); c1 != 2^n: oriented in
            // a cross X configuration
            let mut c1 = axisdir ^ clcorner; // the same trick

            c1 &= 7 - (1 << cledge); // even more hacks

            if c1 != 1 && c1 != 2 && c1 != 4 {
                break 'skip;
            }

            let (mut ax1, mut ax2) = (0usize, 0usize);
            if cledge == 0 {
                (ax1, ax2) = (1, 2);
            }
            if cledge == 1 {
                (ax1, ax2) = (2, 0);
            }
            if cledge == 2 {
                (ax1, ax2) = (0, 1);
            }
            let ax = cledge;

            // then it finds with which face the capsule has a lower angle and switches the
            // axis names
            if axis[ax1].abs() > axis[ax2].abs() {
                ax1 = ax2;
            }
            ax2 = 3 - ax - ax1;

            // keep track of the axis orientation (mul tells in which direction along the
            // capsule to find the second point); all other references to the axis
            // "halfaxis" are with an absolute value
            let mul;
            if c1 & (1 << ax2) != 0 {
                mul = one;
                secondpos = one - bestsegmentpos;
            } else {
                mul = -one;
                secondpos = one + bestsegmentpos;
            }

            // now we have to find out whether we point towards the opposite side or towards
            // one of the sides and also find the farthest point along the capsule that is
            // above the box
            let mut e1 = two * size2[ax2] / halfaxis[ax2].abs();
            if e1 < secondpos {
                secondpos = e1;
            }

            let e2 = if ((axisdir & (1 << ax)) != 0) == ((c1 & (1 << ax2)) != 0) {
                one - bestboxpos
            } else {
                one + bestboxpos
            };

            e1 = size2[ax] * e2 / halfaxis[ax].abs();

            if e1 < secondpos {
                secondpos = e1;
            }

            secondpos *= mul;
        } else if cltype < 0 {
            // similarly we handle the case when one capsule end is closest to a face of the
            // box and find where the other end is pointing to and clamp to the farthest
            // point of the capsule that is above the box
            if clface == -1 {
                break 'skip; // the closest point is inside the box, no need for a second point
            }
            let mul = if cltype == -3 { one } else { -one };

            secondpos = two;

            let mut t1 = pos;
            add_to_scl3(&mut t1, halfaxis, -mul);

            for i in 0..3usize {
                if i as i32 != clface {
                    let mut e1 = (size2[i] - t1[i]) / halfaxis[i] * mul;
                    if e1 > R::ZERO && e1 < secondpos {
                        secondpos = e1;
                    }

                    e1 = (-size2[i] - t1[i]) / halfaxis[i] * mul;
                    if e1 > R::ZERO && e1 < secondpos {
                        secondpos = e1;
                    }
                }
            }
            secondpos *= mul;
        }
    }

    // create a sphere in the original orientation at the first contact point
    let mut t1 = pos;
    add_to_scl3(&mut t1, halfaxis, bestsegmentpos);
    let t2 = add3(mul_mat_vec3(mat2, t1), pos2);

    // collide with the box
    let mut n = raw_sphere_box(con, margin, t2, size1, pos2, mat2, size2);

    if secondpos > lit::<R>(-3.0) {
        // secondpos was modified
        let mut t1 = pos;
        add_to_scl3(&mut t1, halfaxis, secondpos + bestsegmentpos); // note the summation
        let t2 = add3(mul_mat_vec3(mat2, t1), pos2);
        n += raw_sphere_box(&mut con[n..], margin, t2, size1, pos2, mat2, size2);
    }
    n
}

/// Port of `mjc_CapsuleBox` (`engine_collision_box.c`): capsule against box.
pub fn capsule_box<R: Real>(
    con: &mut [PreContact<R>],
    margin: R,
    g1: &GeomPose<R>,
    g2: &GeomPose<R>,
) -> usize {
    raw_capsule_box(
        con, margin, g1.pos, &g1.mat, &g1.size, g2.pos, &g2.mat, &g2.size,
    )
}

/// Port of `clipHalfPlane`: clips the polygon `buf[*cur][..nin]` against the half-plane
/// `sign * v[coord] <= limit`. When every vertex is already inside, the polygon is left
/// untouched (no copies, the common resting case); otherwise the result is written to the
/// other buffer and `*cur` swaps. The `z` of a vertex is interpolated as an attribute.
/// Returns the vertex count.
fn clip_half_plane<R: Real>(
    nin: usize,
    buf: &mut [[[R; 3]; MAXVERT]; 2],
    cur: &mut usize,
    coord: usize,
    sign: R,
    limit: R,
) -> usize {
    let inb = *cur;
    let mut d = [R::ZERO; MAXVERT];
    let mut all_inside = true;
    for k in 0..nin {
        d[k] = sign * buf[inb][k][coord] - limit;
        all_inside &= d[k] <= R::ZERO;
    }
    if all_inside {
        return nin;
    }

    let outb = 1 - inb;
    let mut nout = 0usize;
    for k in 0..nin {
        let p = buf[inb][k];
        let k1 = if k + 1 == nin { 0 } else { k + 1 };
        let (dp, dq) = (d[k], d[k1]);

        // emit p if inside
        if dp <= R::ZERO && nout < MAXVERT {
            buf[outb][nout] = p;
            nout += 1;
        }

        // emit the intersection if the edge strictly crosses the plane
        if ((dp < R::ZERO && dq > R::ZERO) || (dp > R::ZERO && dq < R::ZERO)) && nout < MAXVERT {
            let q = buf[inb][k1];
            let t = dp / (dp - dq);
            buf[outb][nout] = [
                p[0] + t * (q[0] - p[0]),
                p[1] + t * (q[1] - p[1]),
                p[2] + t * (q[2] - p[2]),
            ];
            nout += 1;
        }
    }
    *cur = outb;
    nout
}

/// Port of `mjc_BoxBox` (`engine_collision_box.c`): box against box.
///
/// Stage 1 is the separating-axis test over the 15 candidate axes (face axes preferred over
/// edge axes on near-ties); stage 2 is either a single edge-edge contact (witness points of
/// the supporting edges, enumerating both signs of an ambiguous support corner) or a face
/// manifold: the incident face is clipped against the four side planes of the reference
/// face and every clipped vertex within the margin band becomes a contact, at most 8.
///
/// `drop_last` is the test-only fault that drops the last contact of a face manifold;
/// `overflow` is set when more than 8 contacts came out (which exact arithmetic cannot
/// do), and the first 8 are kept.
pub fn box_box<R: Real>(
    con: &mut [PreContact<R>],
    margin: R,
    g1: &GeomPose<R>,
    g2: &GeomPose<R>,
    drop_last: bool,
    overflow: &mut bool,
) -> usize {
    let (pos1, mat1, size1) = (g1.pos, &g1.mat, &g1.size);
    let (pos2, mat2, size2) = (g2.pos, &g2.mat, &g2.size);
    let one = R::ONE;
    let half = lit::<R>(0.5);

    // rot: the axes of box 2 in the frame of box 1 (columns); pos21: the centre of box 2 in
    // the frame of box 1; pos12: the centre of box 1 in the frame of box 2
    let tmp = sub3(pos2, pos1);
    let pos21 = mul_mat_t_vec3(mat1, tmp);
    let tmp = sub3(pos1, pos2);
    let pos12 = mul_mat_t_vec3(mat2, tmp);
    let rot = mul_mat_t_mat3(mat1, mat2);
    let mut rotabs = [R::ZERO; 9];
    for i in 0..9 {
        rotabs[i] = rot[i].abs();
    }

    // ------------------------------ stage 1: the separating-axis test

    // the separation tests decide contact against no contact, so they carry rounding slack:
    // without it a box pair that genuinely overlaps by less than the rounding error of its
    // own support evaluation is reported as separated, and the boxes pass through each other
    let septol = margin
        + R::BOXBOX_SEPEPS * (size1[0] + size1[1] + size1[2] + size2[0] + size2[1] + size2[2]);

    // the best separation so far (most positive; negative = penetration), and the winning
    // axis: code 0..2 a face of box 1, 3..5 a face of box 2, >= 6 an edge pair (i, j) as
    // 6 + 3 * i + j
    let mut sep_best = -R::MAXVAL;
    let mut code: i32 = -1;

    // the face axes of box 1: the candidate normal is axis i of box 1
    for i in 0..3usize {
        let radius2 =
            rotabs[3 * i] * size2[0] + rotabs[3 * i + 1] * size2[1] + rotabs[3 * i + 2] * size2[2];
        let sep = pos21[i].abs() - size1[i] - radius2;
        if sep > septol {
            return 0;
        }
        if sep > sep_best {
            sep_best = sep;
            code = i as i32;
        }
    }

    // the face axes of box 2: the candidate normal is axis j of box 2
    for j in 0..3usize {
        let radius1 = rotabs[j] * size1[0] + rotabs[3 + j] * size1[1] + rotabs[6 + j] * size1[2];
        let sep = pos12[j].abs() - size2[j] - radius1;
        if sep > septol {
            return 0;
        }
        if sep > sep_best {
            sep_best = sep;
            code = 3 + j as i32;
        }
    }
    let sep_face = sep_best;
    let code_face = code;

    // the edge-cross axes: the candidate direction is axis i of box 1 crossed with axis j of
    // box 2
    for i in 0..3usize {
        for j in 0..3usize {
            // the cross product of e_i with column j of rot, in the frame of box 1;
            // component i is zero
            let (i1, i2) = ((i + 1) % 3, (i + 2) % 3);
            let mut ax1 = -rot[3 * i2 + j];
            let mut ax2 = rot[3 * i1 + j];

            // the cross product of two unit vectors has norm sin(angle); for nearly parallel
            // edges the components above are pure cancellation noise and the direction is
            // meaningless, so require sin(angle) well above rounding; the skipped axes are
            // covered by the face normals, which the cross product converges to as the angle
            // vanishes
            let norm2 = ax1 * ax1 + ax2 * ax2;
            if norm2 < R::BOXBOX_PAREPS {
                continue;
            }
            let inv = one / norm2.sqrt();
            ax1 *= inv;
            ax2 *= inv;

            // the support radius of box 1: component i of the axis is zero by construction
            let radius1 = size1[i1] * ax1.abs() + size1[i2] * ax2.abs();

            // the support radius of box 2: transform the axis to the frame of box 2;
            // component j is zero there, and only components i1, i2 of the axis are nonzero
            let (j1, j2) = ((j + 1) % 3, (j + 2) % 3);
            let a2_1 = ax1 * rot[3 * i1 + j1] + ax2 * rot[3 * i2 + j1];
            let a2_2 = ax1 * rot[3 * i1 + j2] + ax2 * rot[3 * i2 + j2];
            let radius2 = size2[j1] * a2_1.abs() + size2[j2] * a2_2.abs();

            let sep = (ax1 * pos21[i1] + ax2 * pos21[i2]).abs() - radius1 - radius2;
            if sep > septol {
                return 0;
            }

            // an edge axis must beat the best face axis by a bias-scaled amount: on exact
            // ties the face manifold (multiple points) is strictly better for the solver
            if sep - R::BOXBOX_EDGEBIAS * sep.abs() > sep_best && sep > sep_face {
                sep_best = sep;
                code = 6 + 3 * i as i32 + j as i32;
            }
        }
    }

    if code < 0 {
        return 0; // cannot happen: some face axis always sets code
    }

    // a winning edge axis nearly parallel to the best face axis (within ~8 degrees)
    // duplicates it: the face manifold covers the same contact with multiple points, and
    // resting stacks flip between the two codes by rounding noise if the near-tie is
    // allowed to alternate. The face is substituted unless the edge is better by five
    // percent of the face depth (ODE's classic fudge)
    if code >= 6 {
        let i = ((code - 6) / 3) as usize;
        let j = ((code - 6) % 3) as usize;
        let (i1, i2) = ((i + 1) % 3, (i + 2) % 3);
        let mut axis = [R::ZERO; 3];
        axis[i1] = -rot[3 * i2 + j];
        axis[i2] = rot[3 * i1 + j];
        normalize3(&mut axis);
        let face_dot = if code_face < 3 {
            axis[code_face as usize].abs()
        } else {
            let f = (code_face - 3) as usize;
            (axis[0] * rot[f] + axis[1] * rot[3 + f] + axis[2] * rot[6 + f]).abs()
        };
        if face_dot > lit::<R>(0.99)
            && sep_best < sep_face + lit::<R>(0.05) * sep_face.abs() + min_val::<R>()
        {
            code = code_face;
        }
    }

    // ------------------------------ stage 2a: the edge-edge contact

    if code >= 6 {
        let i = ((code - 6) / 3) as usize;
        let j = ((code - 6) % 3) as usize;
        let (i1, i2) = ((i + 1) % 3, (i + 2) % 3);
        let (j1, j2) = ((j + 1) % 3, (j + 2) % 3);

        // the unit separating axis in the frame of box 1, oriented from box 1 toward box 2
        let mut axis = [R::ZERO; 3];
        axis[i1] = -rot[3 * i2 + j];
        axis[i2] = rot[3 * i1 + j];
        normalize3(&mut axis);
        if dot3(axis, pos21) < R::ZERO {
            axis[0] = -axis[0];
            axis[1] = -axis[1];
            axis[2] = -axis[2];
        }

        // the supporting edges: the box 1 edge runs along e_i at a corner selected by the
        // axis signs in (i1, i2); the box 2 edge runs along column j at a corner selected by
        // the signs of the axis in box 2 coordinates. A near-zero component makes the sign
        // choice meaningless -- both edges support the axis -- and rounding can pick the
        // wrong one, producing witness points on the wrong side of the box. Enumerate both
        // signs for any ambiguous component (at most one per box) and keep the closest
        // witness pair.
        let a2 = [
            axis[0] * rot[0] + axis[1] * rot[3] + axis[2] * rot[6],
            axis[0] * rot[1] + axis[1] * rot[4] + axis[2] * rot[7],
            axis[0] * rot[2] + axis[1] * rot[5] + axis[2] * rot[8],
        ];
        let ambig = R::BOXBOX_SGNEPS;
        let mut amb1: i32 = -1;
        let mut amb2: i32 = -1;
        if axis[i1].abs() < ambig {
            amb1 = i1 as i32;
        } else if axis[i2].abs() < ambig {
            amb1 = i2 as i32;
        }
        if a2[j1].abs() < ambig {
            amb2 = j1 as i32;
        } else if a2[j2].abs() < ambig {
            amb2 = j2 as i32;
        }

        let d2 = [rot[j], rot[3 + j], rot[6 + j]];
        let b = d2[i]; // d1 . d2, with d1 = e_i
        let denom = one - b * b;

        let mut w1 = [R::ZERO; 3];
        let mut w2 = [R::ZERO; 3];
        let mut best_d2 = R::MAXVAL;
        for v1 in 0..(if amb1 >= 0 { 2 } else { 1 }) {
            for v2 in 0..(if amb2 >= 0 { 2 } else { 1 }) {
                // the corner of the box 1 edge: support along +axis, the ambiguous component
                // flipped by v1
                let mut c1 = [R::ZERO; 3];
                c1[i1] = if axis[i1] >= R::ZERO {
                    size1[i1]
                } else {
                    -size1[i1]
                };
                c1[i2] = if axis[i2] >= R::ZERO {
                    size1[i2]
                } else {
                    -size1[i2]
                };
                if amb1 >= 0 && v1 != 0 {
                    c1[amb1 as usize] = -c1[amb1 as usize];
                }

                // the corner of the box 2 edge: support along -axis in box 2 coordinates
                let mut cc = [R::ZERO; 3];
                cc[j1] = if a2[j1] >= R::ZERO {
                    -size2[j1]
                } else {
                    size2[j1]
                };
                cc[j2] = if a2[j2] >= R::ZERO {
                    -size2[j2]
                } else {
                    size2[j2]
                };
                if amb2 >= 0 && v2 != 0 {
                    cc[amb2 as usize] = -cc[amb2 as usize];
                }
                let mut c2 = mul_mat_vec3(&rot, cc);
                c2 = add3(c2, pos21);

                // the closest points between the two edge segments (the directions are unit
                // vectors)
                let e = sub3(c2, c1);
                let d1e = e[i]; // d1 . e
                let d2e = dot3(d2, e);
                let mut s = if denom < min_val::<R>() {
                    R::ZERO
                } else {
                    (d1e - b * d2e) / denom
                };

                // clamp into the segments, letting each clamp re-solve the other parameter
                s = clip(s, -size1[i], size1[i]);
                let t = clip(b * s - d2e, -size2[j], size2[j]);
                s = clip(d1e + b * t, -size1[i], size1[i]);

                let mut p1 = c1;
                p1[i] += s;
                let mut p2 = c2;
                add_to_scl3(&mut p2, d2, t);
                let gap = sub3(p2, p1);
                let gap2 = dot3(gap, gap);
                if gap2 < best_d2 {
                    best_d2 = gap2;
                    w1 = p1;
                    w2 = p2;
                }
            }
        }

        // the signed distance along the axis
        let gap = sub3(w2, w1);
        let dist = dot3(gap, axis);
        if dist > septol {
            return 0;
        }

        // the contact at the midpoint of the witness pair: for penetrating edges this is
        // inside both boxes; in the margin band it is midway between the two surfaces
        let mid = [
            half * (w1[0] + w2[0]),
            half * (w1[1] + w2[1]),
            half * (w1[2] + w2[2]),
        ];

        con[0].dist = dist;
        let t = mul_mat_vec3(mat1, mid);
        con[0].pos = add3(t, pos1);
        con[0].normal = mul_mat_vec3(mat1, axis);
        con[0].tangent = [R::ZERO; 3];
        return 1;
    }

    // ------------------------------ stage 2b: the face contact

    // the reference box: the one with the winning face; the incident box: the other one
    let ref1 = code < 3; // is box 1 the reference?
    let a = if ref1 {
        code as usize
    } else {
        (code - 3) as usize
    }; // the face axis of the reference box
    let sizeref = if ref1 { size1 } else { size2 };
    let sizeinc = if ref1 { size2 } else { size1 };
    let posref = if ref1 { pos1 } else { pos2 };
    let matref = if ref1 { mat1 } else { mat2 };
    let posoi = if ref1 { pos21 } else { pos12 }; // the incident centre in the reference frame

    // the incident box axes in the reference frame: rot maps box 2 to box 1, the transpose
    // maps box 1 to box 2; rinc(r, c) = component r of incident axis c, in the reference frame
    let rinc: [R; 9] = if ref1 {
        rot
    } else {
        [
            rot[0], rot[3], rot[6], rot[1], rot[4], rot[7], rot[2], rot[5], rot[8],
        ]
    };

    // the face direction: +1 if the incident box lies along +a, else -1
    let sgn = if posoi[a] >= R::ZERO { one } else { -one };

    // the incident face: the face of the incident box most opposed to the reference face
    // normal
    let mut binc = 0usize;
    for k in 1..3usize {
        if rinc[3 * a + k].abs() > rinc[3 * a + binc].abs() {
            binc = k;
        }
    }
    // the sign making the incident normal oppose
    let tinc = if sgn * rinc[3 * a + binc] > R::ZERO {
        -one
    } else {
        one
    };

    // the corners of the incident face in the reference frame, cyclic winding; the in-plane
    // coordinates are (x, y) = the two non-a reference axes, z is the signed distance above
    // the reference face plane (negative = inside the reference box)
    let (ax, ay) = ((a + 1) % 3, (a + 2) % 3);
    let (bu, bv) = ((binc + 1) % 3, (binc + 2) % 3);
    let mut poly = [[[R::ZERO; 3]; MAXVERT]; 2];

    // the face centre and the in-face half-edge offsets, in the projected (x, y, z)
    // coordinates
    let mut cx = [R::ZERO; 3];
    let mut du = [R::ZERO; 3];
    let mut dv = [R::ZERO; 3];
    for r in 0..3usize {
        let c = if r == 0 {
            ax
        } else if r == 1 {
            ay
        } else {
            a
        };
        cx[r] = posoi[c] + tinc * sizeinc[binc] * rinc[3 * c + binc];
        du[r] = sizeinc[bu] * rinc[3 * c + bu];
        dv[r] = sizeinc[bv] * rinc[3 * c + bv];
    }
    cx[2] = sgn * cx[2] - sizeref[a];
    du[2] *= sgn;
    dv[2] *= sgn;
    let corner_sign: [[R; 2]; 4] = [[one, one], [-one, one], [-one, -one], [one, -one]];
    for k in 0..4usize {
        let (su, sv) = (corner_sign[k][0], corner_sign[k][1]);
        poly[0][k][0] = cx[0] + su * du[0] + sv * dv[0];
        poly[0][k][1] = cx[1] + su * du[1] + sv * dv[1];
        poly[0][k][2] = cx[2] + su * du[2] + sv * dv[2];
    }

    // clip against the four side planes of the reference face; the buffers swap only on
    // passes that actually clip
    let mut nvert = 4usize;
    let mut cur = 0usize;
    nvert = clip_half_plane(nvert, &mut poly, &mut cur, 0, one, sizeref[ax]);
    nvert = clip_half_plane(nvert, &mut poly, &mut cur, 0, -one, sizeref[ax]);
    nvert = clip_half_plane(nvert, &mut poly, &mut cur, 1, one, sizeref[ay]);
    nvert = clip_half_plane(nvert, &mut poly, &mut cur, 1, -one, sizeref[ay]);

    // accept the vertices within the margin band, dropping near-duplicates produced by
    // clipping at polygon corners; the duplicate radius is relative to the reference face
    // scale
    let mut accepted = [[R::ZERO; 3]; MAXVERT];
    let mut naccept = 0usize;
    let dupe2 = R::BOXBOX_DUPEPS * (sizeref[ax] * sizeref[ax] + sizeref[ay] * sizeref[ay]);
    for &v in poly[cur].iter().take(nvert) {
        if v[2] > margin {
            continue;
        }
        let mut dupe = false;
        for a in accepted.iter().take(naccept) {
            let dx = a[0] - v[0];
            let dy = a[1] - v[1];
            if dx * dx + dy * dy < dupe2 {
                dupe = true;
                break;
            }
        }
        if !dupe {
            accepted[naccept] = v;
            naccept += 1;
        }
    }
    if naccept == 0 {
        return 0;
    }

    // the world normal points from geom 1 to geom 2: along +sgn * a of the reference frame
    // when box 1 is the reference, opposite when box 2 is
    let nsign = if ref1 { sgn } else { -sgn };
    let normal = [
        nsign * matref[a],
        nsign * matref[3 + a],
        nsign * matref[6 + a],
    ];

    // at most 8 contacts in exact arithmetic (a 4-gon clipped by 4 half-planes); the
    // pre-contact slice holds 8
    debug_assert!(naccept <= 8, "box-box made {naccept} contacts");
    if naccept > 8 {
        *overflow = true;
        naccept = 8;
    }
    if drop_last {
        naccept -= 1;
    }

    for k in 0..naccept {
        let v = accepted[k];

        // the contact position: on the clipped incident polygon in (x, y), midway between the
        // reference face plane and the incident surface along the face axis
        let mut posc = [R::ZERO; 3];
        posc[ax] = v[0];
        posc[ay] = v[1];
        posc[a] = sgn * (sizeref[a] + half * v[2]);

        con[k].dist = v[2];
        let t = mul_mat_vec3(matref, posc);
        con[k].pos = add3(t, posref);
        con[k].normal = normal;
        con[k].tangent = [R::ZERO; 3];
    }
    naccept
}
