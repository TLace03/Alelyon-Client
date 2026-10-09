//! The collision step: two passes over the static candidate list.
//!
//! Ports, from MuJoCo 3.14.0 `engine_collision_driver.c`: the narrowphase of `mj_collision`
//! (`mj_filterSphere` with `planeGeomDist`, `mj_setContact`, `addPairContacts`) in the form a
//! GPU wants. WHICH geom pairs are tested and in WHICH ORDER is decided when the model is
//! compiled (`candidates.rs`: the filters, the contact order, the per-pair maxima and the mixed
//! contact parameters), so that this file does no search and allocates nothing. The colliders are
//! in `collide_primitive.rs` and `collide_box.rs`, the frames and the contact force in
//! `contact.rs`.
//!
//! # The step: two passes
//!
//! Pass 1 runs one independent unit per candidate `k`: the bounding-sphere test, then the collider,
//! which writes at most `slot_count` pre-contacts (`dist`, `pos`, `normal`, `tangent`) into its own
//! slot range of the pre-contact arrays (`Data::pre_*`, at `slot_offset`), its count into
//! `cand_ncon[k]` and a box-box overflow flag into `cand_overflow[k]`. Between the passes a prefix
//! sum over the counts gives `cand_start[k]`, the index of candidate `k`'s first contact (a scan
//! of at most `candidates.len()` numbers, in list order). Pass 2 runs one independent unit per
//! candidate `k` again, OUT OF PLACE: it gathers the pre-contacts `pre[slot_offset + s]` into
//! `contact[cand_start[k] + s]`, adding for each contact the geoms, the static parameters of its
//! candidate, `includemargin`, `exclude = dist >= includemargin`, the completed frame
//! (`mju_makeFrame`), `efc_address = -1`, `mu = 0` and `H = 0` (`mj_setContact`).
//!
//! Because the pre-contacts and the contacts are separate arrays, every unit of either pass reads
//! only its own inputs and writes only its own outputs, so the units may run in ANY order or in
//! parallel with no atomics, no sort and no data-dependent loop; the result is the same bits (the
//! phase-1c-ii review found that the earlier in-place compaction was safe only in list order:
//! the destination range of one candidate overlaps the source range of an earlier one, so a
//! parallel scatter would race). [`collide_in_order`] runs the passes in orders the caller
//! chooses, and a test holds the result equal to the list order's. The box-box overflow count is
//! reduced after pass 2 (a sum of the flags), so no unit updates a shared counter.
//!
//! Invariants:
//! - **No allocation inside the step.**
//! - **Deterministic**: no clock, no randomness, no unordered iteration; the contact list is a
//!   pure function of `(model, qpos)`.
//! - A contact in the gap (`includemargin <= dist < includemargin + gap`) is generated, kept in
//!   the list, counted in `ncon`, and flagged `exclude`; it has no constraint rows. MuJoCo's
//!   `includemargin` is the margin WITHOUT the gap (3.14.0; older versions used `margin - gap`).
//! - Every collider is held to MuJoCo's contact for contact; the per-pair bounding-sphere test
//!   is MuJoCo's, so a pair the culls of MuJoCo's broadphase and midphase would have dropped
//!   returns nothing here too (those culls only drop pairs whose collider returns nothing).
//!
//! Deviations from MuJoCo: the candidate list replaces the sweep-and-prune, the BVH and the
//! midphase (see `candidates.rs`); there is no `nconmax` overflow (the arrays are sized to the
//! per-pair maxima), so `mjWARN_CONTACTFULL` cannot occur; a box-box pair that returned more
//! than 8 contacts (impossible in exact arithmetic, an `mjERROR` in MuJoCo) raises a
//! `debug_assert`, keeps its first 8, and counts once in `Data::warning_collision_overflow`
//! (per collider call: an RK4 step runs the collision step four times).
//!
//! Not ported, and unreachable from an imported scene: flex collisions, SDF and height-field
//! geoms, explicit `<pair>` entries, the contact-filter callback, sleeping and MuJoCo's convex
//! path (GJK and EPA: [`crate::NotModelled::Collision`] names every pair that would have used it).

use crate::collide_box::{box_box, capsule_box, sphere_box};
use crate::collide_primitive::{
    GeomPose, PreContact, capsule_capsule, plane_box, plane_capsule, plane_cylinder, plane_sphere,
    sphere_capsule, sphere_cylinder, sphere_sphere,
};
use crate::contact::make_frame;
use crate::data::Data;
use crate::math::{v3, v9};
use crate::model::{Candidate, Collider, GeomType, Model};
use crate::real::Real;
use crate::smooth::Faults;

/// The pose of geom `g` as the colliders read it.
fn geom_pose<R: Real>(m: &Model<R>, d: &Data<R>, g: usize) -> GeomPose<R> {
    GeomPose {
        pos: v3(&d.geom_xpos, g),
        mat: v9(&d.geom_xmat, g),
        size: v3(&m.geom_size, g),
    }
}

/// Port of `planeGeomDist`: the signed distance along the normal of plane `g1` to the origin
/// of geom `g2`.
fn plane_geom_dist<R: Real>(d: &Data<R>, g1: usize, g2: usize) -> R {
    let mat1 = &d.geom_xmat[9 * g1..9 * g1 + 9];
    let norm = [mat1[2], mat1[5], mat1[8]];
    let dif = crate::math::sub3(v3(&d.geom_xpos, g2), v3(&d.geom_xpos, g1));
    crate::math::dot3(dif, norm)
}

/// Port of `mj_filterSphere` (with `filterSphere`): whether the bounding spheres of geoms
/// `g1` and `g2`, grown by `margin`, are apart (`true`: discard the pair); a plane is tested
/// by the one-sided distance of the other geom's centre to it.
fn filter_sphere<R: Real>(m: &Model<R>, d: &Data<R>, g1: usize, g2: usize, margin: R) -> bool {
    let (r1, r2) = (m.geom_rbound[g1], m.geom_rbound[g2]);

    // neither geom is a plane
    if r1 > R::ZERO && r2 > R::ZERO {
        let (p1, p2) = (v3(&d.geom_xpos, g1), v3(&d.geom_xpos, g2));
        let dif = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]];
        let distsqr = dif[0] * dif[0] + dif[1] * dif[1] + dif[2] * dif[2];
        let bound = r1 + r2 + margin;
        return distsqr > bound * bound;
    }

    // one geom is a plane
    if m.geom_type[g1] == GeomType::Plane
        && r2 > R::ZERO
        && plane_geom_dist(d, g1, g2) > margin + r2
    {
        return true;
    }
    if m.geom_type[g2] == GeomType::Plane
        && r1 > R::ZERO
        && plane_geom_dist(d, g2, g1) > margin + r1
    {
        return true;
    }
    false
}

/// Runs candidate `k`'s collider: the pre-contacts it reports, and how many.
fn run_collider<R: Real>(
    c: &Candidate<R>,
    g1: &GeomPose<R>,
    g2: &GeomPose<R>,
    faults: &Faults,
    overflow: &mut bool,
    pre: &mut [PreContact<R>; 8],
) -> usize {
    let margin = c.margin_gap;
    match c.collider {
        Collider::PlaneSphere => plane_sphere(pre, margin, g1, g2),
        Collider::PlaneCapsule => plane_capsule(pre, margin, g1, g2),
        Collider::PlaneCylinder => plane_cylinder(pre, margin, g1, g2),
        Collider::PlaneBox => plane_box(pre, margin, g1, g2),
        Collider::SphereSphere => sphere_sphere(pre, margin, g1, g2),
        Collider::SphereCapsule => sphere_capsule(pre, margin, g1, g2),
        Collider::SphereCylinder => sphere_cylinder(pre, margin, g1, g2),
        Collider::SphereBox => sphere_box(pre, margin, g1, g2),
        Collider::CapsuleCapsule => capsule_capsule(pre, margin, g1, g2),
        Collider::CapsuleBox => capsule_box(pre, margin, g1, g2),
        Collider::BoxBox => box_box(
            pre,
            margin,
            g1,
            g2,
            faults.drop_last_boxbox_contact,
            overflow,
        ),
    }
}

/// Port of `mj_collision`: writes `ncon` and the contact arrays of `d` (see the module
/// note). Returns with `ncon = 0` when constraints or contacts are disabled or the model has
/// fewer than two bodies. Needs `kinematics` for the current `qpos` (the geom frames).
pub(crate) fn collision<R: Real>(m: &Model<R>, d: &mut Data<R>, faults: &Faults) {
    // reset the size of the contact array
    d.ncon = 0;

    // return if disabled
    if m.disable.constraint || m.disable.contact || m.nbody < 2 {
        return;
    }
    debug_assert_eq!(
        (m.disable.filterparent, m.disable.midphase),
        (m.candidates_built_filterparent, m.candidates_built_midphase),
        "the candidate list was built with other flags: call Model::rebuild_contact_pairs"
    );

    // ---- pass 1: one independent unit per candidate, writing its own slots
    for k in 0..m.candidates.len() {
        pass1_candidate(m, d, faults, k);
    }

    // ---- the prefix sum of the counts, then pass 2: one independent unit per candidate
    prefix_sum(d);
    for k in 0..m.candidates.len() {
        pass2_candidate(m, d, faults, k);
    }
    reduce_overflow(d);
}

/// Pass 1 for candidate `k` (see the module note): the bounding-sphere test and the collider;
/// writes only candidate `k`'s own slots (`pre_*` at `slot_offset`), `cand_ncon[k]` and
/// `cand_overflow[k]`, and reads only the geom frames and the model.
fn pass1_candidate<R: Real>(m: &Model<R>, d: &mut Data<R>, faults: &Faults, k: usize) {
    let c = &m.candidates[k];
    let (g1, g2) = (geom_pose(m, d, c.g1), geom_pose(m, d, c.g2));
    let mut n = 0usize;
    let mut overflow = false;
    if !filter_sphere(m, d, c.g1, c.g2, c.margin_gap) {
        let mut pre = [PreContact::<R>::default(); 8];
        n = run_collider(c, &g1, &g2, faults, &mut overflow, &mut pre);
        debug_assert!(
            n <= c.slot_count,
            "a collider returned more than its maximum"
        );
        for (s, p) in pre.iter().take(n).enumerate() {
            let idx = c.slot_offset + s;
            d.pre_dist[idx] = p.dist;
            d.pre_pos[3 * idx..3 * idx + 3].copy_from_slice(&p.pos);
            d.pre_frame[6 * idx..6 * idx + 3].copy_from_slice(&p.normal);
            d.pre_frame[6 * idx + 3..6 * idx + 6].copy_from_slice(&p.tangent);
        }
    }
    d.cand_ncon[k] = n;
    d.cand_overflow[k] = overflow;
}

/// The exclusive prefix sum of `cand_ncon` into `cand_start`, and the total into `ncon`: the
/// one sequential step between the passes (a scan of the candidate counts).
fn prefix_sum<R: Real>(d: &mut Data<R>) {
    let mut total = 0usize;
    for k in 0..d.cand_ncon.len() {
        d.cand_start[k] = total;
        total += d.cand_ncon[k];
    }
    d.ncon = total;
}

/// Pass 2 for candidate `k` (see the module note): gathers its pre-contacts into
/// `contact[cand_start[k] + s]` and completes them (`mj_setContact`, `addPairContacts`); reads
/// only candidate `k`'s own pre-contacts and writes only its own contacts, so any order of the
/// candidates gives the same bits.
fn pass2_candidate<R: Real>(m: &Model<R>, d: &mut Data<R>, faults: &Faults, k: usize) {
    let c = &m.candidates[k];
    for s in 0..d.cand_ncon[k] {
        let (src, dst) = (c.slot_offset + s, d.cand_start[k] + s);
        d.contact_dist[dst] = d.pre_dist[src];
        d.contact_pos[3 * dst..3 * dst + 3].copy_from_slice(&d.pre_pos[3 * src..3 * src + 3]);
        d.contact_frame[9 * dst..9 * dst + 6].copy_from_slice(&d.pre_frame[6 * src..6 * src + 6]);

        // mj_setContact and addPairContacts: the geoms and the static parameters
        d.contact_geom[2 * dst] = c.g1;
        d.contact_geom[2 * dst + 1] = c.g2;
        d.contact_dim[dst] = c.condim;
        let includemargin = if faults.includemargin_minus_gap {
            c.includemargin - c.gap
        } else {
            c.includemargin
        };
        d.contact_includemargin[dst] = includemargin;
        d.contact_friction[5 * dst..5 * dst + 5].copy_from_slice(&c.friction);
        d.contact_solref[2 * dst..2 * dst + 2].copy_from_slice(&c.solref);
        d.contact_solreffriction[2 * dst..2 * dst + 2].copy_from_slice(&c.solreffriction);
        d.contact_solimp[5 * dst..5 * dst + 5].copy_from_slice(&c.solimp);

        // a contact in the gap is excluded (kept, counted, no rows)
        d.contact_exclude[dst] = i32::from(d.contact_dist[dst] >= includemargin);

        // complete the frame
        if faults.flip_contact_normal {
            for v in &mut d.contact_frame[9 * dst..9 * dst + 3] {
                *v = -*v;
            }
        }
        make_frame(&mut d.contact_frame[9 * dst..9 * dst + 9]);

        // clear the fields that are computed later
        d.contact_efc_address[dst] = -1;
        d.contact_mu[dst] = R::ZERO;
        d.contact_h[36 * dst..36 * dst + 36].fill(R::ZERO);
    }
}

/// Adds the candidates' box-box overflow flags to `warning_collision_overflow` (a sum over
/// the flags, after pass 2, so that no unit of either pass touches a shared counter).
fn reduce_overflow<R: Real>(d: &mut Data<R>) {
    let mut n = 0usize;
    for &o in &d.cand_overflow {
        n += usize::from(o);
    }
    d.warning_collision_overflow += n;
}

/// The collision step with the units of each pass run in the order the caller gives: `pass1`
/// and `pass2` must each list every candidate index of `m.candidates` exactly once. The
/// contacts are the same bits as [`collide`]'s whatever the orders, which a test asserts (the
/// property that lets a GPU port dispatch the units in parallel, or grouped by collider).
/// A hidden test hook; it does not allocate.
#[doc(hidden)]
pub fn collide_in_order<R: Real>(m: &Model<R>, d: &mut Data<R>, pass1: &[usize], pass2: &[usize]) {
    d.ncon = 0;
    if m.disable.constraint || m.disable.contact || m.nbody < 2 {
        return;
    }
    assert_eq!(pass1.len(), m.candidates.len());
    assert_eq!(pass2.len(), m.candidates.len());
    for &k in pass1 {
        pass1_candidate(m, d, &Faults::NONE, k);
    }
    prefix_sum(d);
    for &k in pass2 {
        pass2_candidate(m, d, &Faults::NONE, k);
    }
    reduce_overflow(d);
}

/// Port of `mj_collision` as a public entry point: the contacts of the current `qpos` in
/// `d` (`ncon` and the contact arrays), with no test faults. Needs [`crate::kinematics`] (the
/// geom frames) to have run for the current state; [`crate::forward`] does that and then
/// builds the constraint rows from the contacts.
pub fn collide<R: Real>(m: &Model<R>, d: &mut Data<R>) {
    collision(m, d, &Faults::NONE);
}
