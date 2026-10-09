//! The plane, sphere and capsule colliders: MuJoCo's primitive collision functions.
//!
//! Ports, from MuJoCo 3.14.0 `engine_collision_primitive.c` (Copyright 2021 DeepMind
//! Technologies Limited, Apache-2.0): `mjraw_PlaneSphere` / `mjc_PlaneSphere`,
//! `mjc_PlaneCapsule`, `mjc_PlaneCylinder`, `mjc_PlaneBox`, `mjraw_SphereSphere` /
//! `mjc_SphereSphere`, `mjraw_SphereCapsule` / `mjc_SphereCapsule`,
//! `mjc_SphereCylinder` and `mjraw_CapsuleCapsule` / `mjc_CapsuleCapsule`. The sphere-box
//! and capsule-box colliders are in `collide_box.rs`. The triangle colliders of the file
//! (`mjraw_SphereTriangle`, `mjraw_BoxTriangle`, `mjraw_CapsuleTriangle`) serve flexes
//! and are not ported.
//!
//! Invariants:
//! - **Same arithmetic, same order as the C source**: normalise by multiplying by the
//!   reciprocal (`mju_normalize3`), compute every difference and product in the written
//!   order, write `mju_clip` as its ternary, never fuse a multiply and an add. Constants
//!   go through [`Real`] (`mjMINVAL` is `1e-15` in both precisions).
//! - **Fixed loops, no allocation, no data-dependent iteration**: a collider evaluates a
//!   fixed number of candidate points (the plane-box loop visits at most 8 corners; the
//!   capsule-capsule parallel branch at most 4 sphere pairs) and writes at most
//!   [`Collider::max_contacts`](crate::Collider::max_contacts) pre-contacts into the
//!   slice it is given, which the driver sized from the model.
//! - **Strict IEEE semantics are relied on**: a comparison with infinity or NaN is false
//!   (see `collide_box.rs`, which divides by components that can be zero). A GPU port must
//!   not enable fast math.
//! - The contact normal points from the first geom to the second; the first geom has the
//!   lower geom type (`pushGeomGeom` orients every pair), so a sphere-cylinder pair has
//!   the sphere first and a plane always comes first.
//!
//! Deviation from MuJoCo: none in the arithmetic of `f64`. `mjc_SphereCylinder`'s cap case builds
//! the flipped cap matrix and the plane-sphere call as the C code does.
//!
//! **`f32` deviation (a measured defect of MuJoCo's absolute thresholds in single precision, found
//! in the phase-1c-ii review).** Two degenerate-branch tests compare against `mjMINVAL = 1e-15`
//! (or its square): `mjc_PlaneCylinder`'s "disk parallel to the plane" (`len_sqr >= 1e-30`) and
//! `mjraw_CapsuleCapsule`'s parallel axes (`|det| >= 1e-15`; `mjraw_CapsuleBox` has the same test
//! and a tie rule). In `f32` the rounding residue of an upright cylinder's axis (`1 - 6e-8`) and of
//! the determinant of parallel capsules (a few `2^-24` of `ma * mc`) are far above those values, so
//! the general branch runs on noise: an upright cylinder at 74 of 360 whole-degree yaws made one
//! contact 12 cm deep or none, was launched or sank through its plane; parallel capsules made one
//! contact in place of two. The `f32` thresholds are scale-aware ([`Real::AXIS_RESIDUAL_SQR`],
//! [`Real::PARALLEL_DET_REL`], [`Real::TIE_REL`]); `f64` keeps MuJoCo's tests bit for bit. MuJoCo's
//! own single-precision build shares the absolute thresholds (`mjtype.h`), so this is a deliberate
//! difference from it (UNMEASURED: its single build is not available here).

use crate::math::{
    add_scl3, add_to_scl3, add3, clip, cross, dot3, lit, min_val, normalize3, scl3, sub3,
};
use crate::real::Real;

/// MuJoCo's `mjPreContact`: what a collider reports for one contact, before the driver
/// adds the contact's parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PreContact<R: Real> {
    /// The distance between the surfaces along the normal (negative: penetration).
    pub dist: R,
    /// The contact position, midway between the surfaces.
    pub pos: [R; 3],
    /// The unit normal, from the first geom to the second.
    pub normal: [R; 3],
    /// The first tangent, or zero when the collider does not set one (the driver's
    /// `mju_makeFrame` picks it).
    pub tangent: [R; 3],
}

/// The pose and size of a geom as a collider reads them (`d->geom_xpos`, `d->geom_xmat`,
/// `m->geom_size`).
#[derive(Clone, Copy, Debug)]
pub struct GeomPose<R: Real> {
    /// The world position of the geom frame.
    pub pos: [R; 3],
    /// The rotation matrix of the geom frame, row-major.
    pub mat: [R; 9],
    /// MuJoCo's size vector of the geom's type.
    pub size: [R; 3],
}

// ---------------------------------------------------------------------------------------
// plane colliders
// ---------------------------------------------------------------------------------------

/// Port of `mjraw_PlaneSphere`: a sphere (`pos2`, radius `size2[0]`) against the plane of
/// `(pos1, mat1)`; one contact at most.
pub(crate) fn raw_plane_sphere<R: Real>(
    con: &mut [PreContact<R>],
    margin: R,
    pos1: [R; 3],
    mat1: &[R; 9],
    pos2: [R; 3],
    size2: &[R; 3],
) -> usize {
    // set the normal
    con[0].normal = [mat1[2], mat1[5], mat1[8]];

    // compute the distance, return if too large
    let tmp = [pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2]];
    let cdist = dot3(tmp, con[0].normal);
    if cdist > margin + size2[0] {
        return 0;
    }

    // depth and position
    con[0].dist = cdist - size2[0];
    let tmp = scl3(con[0].normal, -con[0].dist / lit::<R>(2.0) - size2[0]);
    con[0].pos = add3(pos2, tmp);
    con[0].tangent = [R::ZERO; 3];
    1
}

/// Port of `mjc_PlaneSphere` (`engine_collision_primitive.c`): plane against sphere.
pub fn plane_sphere<R: Real>(
    con: &mut [PreContact<R>],
    margin: R,
    g1: &GeomPose<R>,
    g2: &GeomPose<R>,
) -> usize {
    raw_plane_sphere(con, margin, g1.pos, &g1.mat, g2.pos, &g2.size)
}

/// Port of `mjc_PlaneCapsule` (`engine_collision_primitive.c`): plane against capsule;
/// two contacts at most, one per end cap, with the tangent along the capsule's axis.
pub fn plane_capsule<R: Real>(
    con: &mut [PreContact<R>],
    margin: R,
    g1: &GeomPose<R>,
    g2: &GeomPose<R>,
) -> usize {
    // the capsule axis, the segment = the scaled axis
    let axis = [g2.mat[2], g2.mat[5], g2.mat[8]];
    let segment = [
        g2.size[1] * axis[0],
        g2.size[1] * axis[1],
        g2.size[1] * axis[2],
    ];

    // point 1, the sphere-plane test
    let endpoint = add3(g2.pos, segment);
    let n1 = raw_plane_sphere(con, margin, g1.pos, &g1.mat, endpoint, &g2.size);

    // point 2, the sphere-plane test
    let endpoint = sub3(g2.pos, segment);
    let n2 = raw_plane_sphere(&mut con[n1..], margin, g1.pos, &g1.mat, endpoint, &g2.size);

    // align the contact frames with the capsule axis
    if n1 != 0 {
        con[0].tangent = axis;
    }
    if n2 != 0 {
        con[n1].tangent = axis;
    }
    n1 + n2
}

/// Port of `mjc_PlaneCylinder` (`engine_collision_primitive.c`): plane against cylinder;
/// up to four contacts (the two rim points under the axis, then two points of a triangle
/// on the side closer to the plane).
pub fn plane_cylinder<R: Real>(
    con: &mut [PreContact<R>],
    margin: R,
    g1: &GeomPose<R>,
    g2: &GeomPose<R>,
) -> usize {
    let (pos1, mat1) = (g1.pos, &g1.mat);
    let (pos2, mat2, size2) = (g2.pos, &g2.mat, &g2.size);
    let half = lit::<R>(0.5);

    let normal = [mat1[2], mat1[5], mat1[8]];
    let mut axis = [mat2[2], mat2[5], mat2[8]];

    // project, make sure the axis points towards the plane
    let mut prjaxis = dot3(normal, axis);
    if prjaxis > R::ZERO {
        axis = scl3(axis, -R::ONE);
        prjaxis = -prjaxis;
    }

    // the normal distance to the cylinder centre
    let mut vec = [pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2]];
    let dist0 = dot3(vec, normal);

    // remove the component of -normal along the axis, compute the length
    vec = scl3(axis, prjaxis);
    vec = sub3(vec, normal);
    let len_sqr = dot3(vec, vec);

    // general configuration: normalise the vector, scale by the radius (MuJoCo's test is
    // `len_sqr >= mjMINVAL^2`; `R::AXIS_RESIDUAL_SQR` is that in f64 and a rounding-noise floor
    // in f32, where an upright cylinder's residual is ~1e-7 long, not 0)
    if len_sqr >= R::AXIS_RESIDUAL_SQR {
        let scl = size2[0] / len_sqr.sqrt();
        vec[0] *= scl;
        vec[1] *= scl;
        vec[2] *= scl;
    }
    // disk parallel to the plane: pick the x axis of the cylinder, scale by the radius
    else {
        vec = [mat2[0] * size2[0], mat2[3] * size2[0], mat2[6] * size2[0]];
    }

    // project the vector on the normal
    let prjvec = dot3(vec, normal);

    // scale the axis by the half-length
    axis = scl3(axis, size2[1]);
    prjaxis *= size2[1];

    // the first point, construct the contact
    let mut cnt = 0;
    if dist0 + prjaxis + prjvec <= margin {
        con[cnt].dist = dist0 + prjaxis + prjvec;
        con[cnt].pos = add3(pos2, vec);
        con[cnt].pos = add3(con[cnt].pos, axis);
        add_to_scl3(&mut con[cnt].pos, normal, -con[cnt].dist * half);
        con[cnt].normal = normal;
        con[cnt].tangent = [R::ZERO; 3];
        cnt += 1;
    } else {
        return 0; // the nearest point is above the margin: no contacts
    }

    // the second point, construct the contact
    if dist0 - prjaxis + prjvec <= margin {
        con[cnt].dist = dist0 - prjaxis + prjvec;
        con[cnt].pos = add3(pos2, vec);
        con[cnt].pos = sub3(con[cnt].pos, axis);
        add_to_scl3(&mut con[cnt].pos, normal, -con[cnt].dist * half);
        con[cnt].normal = normal;
        con[cnt].tangent = [R::ZERO; 3];
        cnt += 1;
    }

    // try to add triangle points on the side closer to the plane
    let prjvec1 = -prjvec * half;
    if dist0 + prjaxis + prjvec1 <= margin {
        // the sideways vector, vec1
        let mut vec1 = cross(vec, axis);
        normalize3(&mut vec1);
        vec1 = scl3(vec1, size2[0] * lit::<R>(3.0).sqrt() / lit::<R>(2.0));

        // point A
        con[cnt].dist = dist0 + prjaxis + prjvec1;
        con[cnt].pos = add3(pos2, vec1);
        con[cnt].pos = add3(con[cnt].pos, axis);
        add_to_scl3(&mut con[cnt].pos, vec, -half);
        add_to_scl3(&mut con[cnt].pos, normal, -con[cnt].dist * half);
        con[cnt].normal = normal;
        con[cnt].tangent = [R::ZERO; 3];
        cnt += 1;

        // point B
        con[cnt].dist = dist0 + prjaxis + prjvec1;
        con[cnt].pos = sub3(pos2, vec1);
        con[cnt].pos = add3(con[cnt].pos, axis);
        add_to_scl3(&mut con[cnt].pos, vec, -half);
        add_to_scl3(&mut con[cnt].pos, normal, -con[cnt].dist * half);
        con[cnt].normal = normal;
        con[cnt].tangent = [R::ZERO; 3];
        cnt += 1;
    }
    cnt
}

/// Port of `mjc_PlaneBox` (`engine_collision_primitive.c`): plane against box; the first
/// four of the eight corners (in index order) that are below the margin and not above the
/// box centre's plane.
pub fn plane_box<R: Real>(
    con: &mut [PreContact<R>],
    margin: R,
    g1: &GeomPose<R>,
    g2: &GeomPose<R>,
) -> usize {
    let (pos1, mat1) = (g1.pos, &g1.mat);
    let (pos2, mat2, size2) = (g2.pos, &g2.mat, &g2.size);

    // the normal, the difference between the centres, the normal distance
    let norm = [mat1[2], mat1[5], mat1[8]];
    let dif = [pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2]];
    let dist = dot3(dif, norm);

    // test all corners, pick the bottom 4
    let mut cnt = 0;
    for i in 0..8usize {
        // the corner in local coordinates
        let vec = [
            if i & 1 != 0 { size2[0] } else { -size2[0] },
            if i & 2 != 0 { size2[1] } else { -size2[1] },
            if i & 4 != 0 { size2[2] } else { -size2[2] },
        ];

        // the corner in global coordinates relative to the box centre
        let mut corner = crate::math::mul_mat_vec3(mat2, vec);

        // the distance to the plane, skip if too far or pointing up
        let ldist = dot3(norm, corner);
        if dist + ldist > margin || ldist > R::ZERO {
            continue;
        }

        // construct the contact
        con[cnt].dist = dist + ldist;
        con[cnt].normal = norm;
        corner = add3(corner, pos2);
        let v = scl3(norm, -con[cnt].dist / lit::<R>(2.0));
        con[cnt].pos = add3(corner, v);
        con[cnt].tangent = [R::ZERO; 3];

        // count; the maximum is 4
        cnt += 1;
        if cnt >= 4 {
            return 4;
        }
    }
    cnt
}

// ---------------------------------------------------------------------------------------
// sphere and capsule colliders
// ---------------------------------------------------------------------------------------

/// Port of `mjraw_SphereSphere`: sphere 1 (`pos1`, `mat1`, radius `size1[0]`) against
/// sphere 2; the coincident-centre fallback takes the cross product of the two z axes
/// (and `(1, 0, 0)` when those are parallel).
// the argument list is MuJoCo's, kept so that the port reads line by line
#[allow(clippy::too_many_arguments)]
pub(crate) fn raw_sphere_sphere<R: Real>(
    con: &mut [PreContact<R>],
    margin: R,
    pos1: [R; 3],
    mat1: &[R; 9],
    size1: &[R; 3],
    pos2: [R; 3],
    mat2: &[R; 9],
    size2: &[R; 3],
) -> usize {
    // check the bounding spheres (this is called from other functions)
    let dif = [pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2]];
    let cdist_sqr = dot3(dif, dif);
    let min_dist = margin + size1[0] + size2[0];
    if cdist_sqr > min_dist * min_dist {
        return 0;
    }

    // depth and normal
    con[0].dist = cdist_sqr.sqrt() - size1[0] - size2[0];
    con[0].normal = sub3(pos2, pos1);
    let len = normalize3(&mut con[0].normal);

    // if the centres are the same, the normal is the cross product of the z axes; if the
    // z axes are parallel it is [1; 0; 0]
    if len < min_val::<R>() {
        let axis1 = [mat1[2], mat1[5], mat1[8]];
        let axis2 = [mat2[2], mat2[5], mat2[8]];
        con[0].normal = cross(axis1, axis2);
        normalize3(&mut con[0].normal);
    }

    // position
    con[0].pos = scl3(con[0].normal, size1[0] + con[0].dist / lit::<R>(2.0));
    con[0].pos = add3(con[0].pos, pos1);

    // axis
    con[0].tangent = [R::ZERO; 3];
    1
}

/// Port of `mjc_SphereSphere` (`engine_collision_primitive.c`): sphere against sphere.
pub fn sphere_sphere<R: Real>(
    con: &mut [PreContact<R>],
    margin: R,
    g1: &GeomPose<R>,
    g2: &GeomPose<R>,
) -> usize {
    raw_sphere_sphere(
        con, margin, g1.pos, &g1.mat, &g1.size, g2.pos, &g2.mat, &g2.size,
    )
}

/// Port of `mjraw_SphereCapsule`: a sphere against a capsule (the sphere-sphere test at
/// the nearest point of the capsule's segment).
// the argument list is MuJoCo's, kept so that the port reads line by line
#[allow(clippy::too_many_arguments)]
pub(crate) fn raw_sphere_capsule<R: Real>(
    con: &mut [PreContact<R>],
    margin: R,
    pos1: [R; 3],
    mat1: &[R; 9],
    size1: &[R; 3],
    pos2: [R; 3],
    mat2: &[R; 9],
    size2: &[R; 3],
) -> usize {
    // the capsule length and axis
    let len = size2[1];
    let axis = [mat2[2], mat2[5], mat2[8]];

    // the projection, clipped to the segment
    let vec = [pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2]];
    let x = clip(dot3(axis, vec), -len, len);

    // the nearest point on the segment, the sphere-sphere test
    let mut v = scl3(axis, x);
    v = add3(v, pos2);
    raw_sphere_sphere(con, margin, pos1, mat1, size1, v, mat2, size2)
}

/// Port of `mjc_SphereCapsule` (`engine_collision_primitive.c`): sphere against capsule.
pub fn sphere_capsule<R: Real>(
    con: &mut [PreContact<R>],
    margin: R,
    g1: &GeomPose<R>,
    g2: &GeomPose<R>,
) -> usize {
    raw_sphere_capsule(
        con, margin, g1.pos, &g1.mat, &g1.size, g2.pos, &g2.mat, &g2.size,
    )
}

/// Port of `mjc_SphereCylinder` (`engine_collision_primitive.c`): sphere against
/// cylinder, by the side (sphere-sphere at the nearest axis point), the cap (plane-sphere,
/// the normal flipped because the plane would come first) or the corner (sphere-sphere
/// against a point at the rim); a sphere centre inside the cylinder goes to the nearer of
/// the side and the cap, the side winning ties.
pub fn sphere_cylinder<R: Real>(
    con: &mut [PreContact<R>],
    margin: R,
    g1: &GeomPose<R>,
    g2: &GeomPose<R>,
) -> usize {
    let (pos1, mat1, size1) = (g1.pos, &g1.mat, &g1.size);
    let (pos2, mat2, size2) = (g2.pos, &g2.mat, &g2.size);

    // the cylinder sizes and axis
    let radius = size2[0];
    let height = size2[1];
    let axis = [mat2[2], mat2[5], mat2[8]];

    // the sphere projection onto the cylinder axis and plane
    let vec = [pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2]];
    let x = dot3(axis, vec);
    let mut a_proj = scl3(axis, x);
    let mut p_proj = sub3(vec, a_proj);
    let p_proj_sqr = dot3(p_proj, p_proj);

    // the collision type
    let mut collide_side = x.abs() < height;
    let mut collide_cap = p_proj_sqr < radius * radius;
    if collide_side && collide_cap {
        // deep penetration (the sphere origin is inside the cylinder)
        let dist_cap = height - x.abs();
        let dist_radius = radius - p_proj_sqr.sqrt();
        if dist_cap < dist_radius {
            // disable one collision type
            collide_side = false;
        } else {
            collide_cap = false;
        }
    }

    // side collision: sphere-sphere
    if collide_side {
        a_proj = add3(a_proj, pos2);
        return raw_sphere_sphere(con, margin, pos1, mat1, size1, a_proj, mat2, size2);
    }

    // cap collision: plane-sphere
    if collide_cap {
        let flipmat = [
            -mat2[0], mat2[1], -mat2[2], -mat2[3], mat2[4], -mat2[5], -mat2[6], mat2[7], -mat2[8],
        ];
        let (pos_cap, mat_cap) = if x > R::ZERO {
            // the top cap
            (add_scl3(pos2, axis, height), *mat2)
        } else {
            // the bottom cap
            (add_scl3(pos2, axis, -height), flipmat)
        };
        let ncon = raw_plane_sphere(con, margin, pos_cap, &mat_cap, pos1, size1);
        if ncon != 0 {
            // flip the direction of the normal (because mjGEOM_PLANE < mjGEOM_SPHERE <
            // mjGEOM_CYLINDER)
            con[0].normal = scl3(con[0].normal, -R::ONE);
        }
        return ncon;
    }

    // otherwise the corner collision: sphere-sphere (the denominator cannot be 0)
    p_proj = scl3(p_proj, size2[0] / p_proj_sqr.sqrt());
    let mut v = scl3(axis, if x > R::ZERO { height } else { -height });
    v = add3(v, p_proj);
    v = add3(v, pos2);

    // sphere-sphere with a point sphere at the corner
    let size_zero = [R::ZERO; 3];
    raw_sphere_sphere(con, margin, pos1, mat1, size1, v, mat2, &size_zero)
}

/// Port of `mjraw_CapsuleCapsule`: two capsules. Axes with `|det| >= mjMINVAL` are
/// intersected by the closest points of their segments (one sphere-sphere test); parallel
/// axes test up to four end points against the other segment and keep up to two contacts,
/// duplicates included.
// the argument list is MuJoCo's, kept so that the port reads line by line
#[allow(clippy::too_many_arguments)]
pub(crate) fn raw_capsule_capsule<R: Real>(
    con: &mut [PreContact<R>],
    margin: R,
    pos1: [R; 3],
    mat1: &[R; 9],
    size1: &[R; 3],
    pos2: [R; 3],
    mat2: &[R; 9],
    size2: &[R; 3],
) -> usize {
    // the capsule axes (scaled) and the centre difference
    let axis1 = [mat1[2] * size1[1], mat1[5] * size1[1], mat1[8] * size1[1]];
    let axis2 = [mat2[2] * size2[1], mat2[5] * size2[1], mat2[8] * size2[1]];
    let dif = [pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2]];

    // the matrix coefficients and the determinant
    let ma = dot3(axis1, axis1);
    let mb = -dot3(axis1, axis2);
    let mc = dot3(axis2, axis2);
    let u = -dot3(axis1, dif);
    let v = dot3(axis2, dif);
    let det = ma * mc - mb * mb;
    let one = R::ONE;

    // the general configuration (non-parallel axes): MuJoCo's test is `|det| >= mjMINVAL`; the
    // relative part (`PARALLEL_DET_REL`, 0 in f64) is the rounding floor of an f32 determinant
    // of exactly parallel axes
    if det.abs() >= min_val::<R>() + R::PARALLEL_DET_REL * (ma * mc) {
        // the projections, clipped to the segments
        let mut x1 = (mc * u - mb * v) / det;
        let mut x2 = (ma * v - mb * u) / det;

        if x1 > one {
            x1 = one;
            x2 = (v - mb) / mc;
        } else if x1 < -one {
            x1 = -one;
            x2 = (v + mb) / mc;
        }
        if x2 > one {
            x2 = one;
            x1 = clip((u - mb) / ma, -one, one);
        } else if x2 < -one {
            x2 = -one;
            x1 = clip((u + mb) / ma, -one, one);
        }

        // the nearest points, the sphere-sphere test
        let mut vec1 = scl3(axis1, x1);
        vec1 = add3(vec1, pos1);
        let mut vec2 = scl3(axis2, x2);
        vec2 = add3(vec2, pos2);
        raw_sphere_sphere(con, margin, vec1, mat1, size1, vec2, mat2, size2)
    }
    // parallel axes
    else {
        // x1 = 1
        let mut vec1 = add3(pos1, axis1);
        let mut x2 = clip((v - mb) / mc, -one, one);
        let mut vec2 = scl3(axis2, x2);
        vec2 = add3(vec2, pos2);
        let n1 = raw_sphere_sphere(con, margin, vec1, mat1, size1, vec2, mat2, size2);

        // x1 = -1
        vec1 = sub3(pos1, axis1);
        x2 = clip((v + mb) / mc, -one, one);
        vec2 = scl3(axis2, x2);
        vec2 = add3(vec2, pos2);
        let n2 = raw_sphere_sphere(&mut con[n1..], margin, vec1, mat1, size1, vec2, mat2, size2);

        // return if two contacts are already found
        if n1 + n2 >= 2 {
            return n1 + n2;
        }

        // x2 = 1
        vec2 = add3(pos2, axis2);
        let mut x1 = clip((u - mb) / ma, -one, one);
        vec1 = scl3(axis1, x1);
        vec1 = add3(vec1, pos1);
        let n3 = raw_sphere_sphere(
            &mut con[n1 + n2..],
            margin,
            vec1,
            mat1,
            size1,
            vec2,
            mat2,
            size2,
        );

        // return if two contacts are already found
        if n1 + n2 + n3 >= 2 {
            return n1 + n2 + n3;
        }

        // x2 = -1
        vec2 = sub3(pos2, axis2);
        x1 = clip((u + mb) / ma, -one, one);
        vec1 = scl3(axis1, x1);
        vec1 = add3(vec1, pos1);
        let n4 = raw_sphere_sphere(
            &mut con[n1 + n2 + n3..],
            margin,
            vec1,
            mat1,
            size1,
            vec2,
            mat2,
            size2,
        );
        n1 + n2 + n3 + n4
    }
}

/// Port of `mjc_CapsuleCapsule` (`engine_collision_primitive.c`): capsule against capsule.
pub fn capsule_capsule<R: Real>(
    con: &mut [PreContact<R>],
    margin: R,
    g1: &GeomPose<R>,
    g2: &GeomPose<R>,
) -> usize {
    raw_capsule_capsule(
        con, margin, g1.pos, &g1.mat, &g1.size, g2.pos, &g2.mat, &g2.size,
    )
}
