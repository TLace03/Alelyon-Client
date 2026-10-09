//! The static candidate list of the collision step, built when a model is compiled.
//!
//! Ports, from MuJoCo 3.14.0 `engine_collision_driver.c`, the parts of `mj_collision` that decide
//! WHICH geom pairs are tested and in WHICH ORDER their contacts come out: `filterBodyPair`,
//! `filterBitmask` (through `filterCollisionPair`), `pushGeomGeom` (the orientation of a pair),
//! `contactcompare` (as a static order), `mj_maxContact`, the collision table `mjCOLLISIONFUNC`,
//! `getMargin` and `getGap`; and `mj_contactParam` (how the parameters of a contact are mixed
//! from its two geoms) with the clamp of `mj_assignFriction` in `engine_core_constraint.c`.
//! It allocates (it runs once, in `Model::compile` or `Model::rebuild_contact_pairs`), which is
//! why it is a file of its own: every other collision module is a step module that allocates
//! nothing (`tests/contacts_alloc.rs` scans them).
//!
//! # A static candidate list instead of a broadphase and a midphase
//!
//! MuJoCo prunes with a sweep-and-prune on float-cast intervals, a BVH and OBB midphase and a
//! per-pair bounding-sphere test. Every one of those culls only removes a pair whose collider
//! would return nothing, so an exhaustive static list of the pairs that survive the filters that
//! depend on the MODEL (body weld, both bodies dof-less, parent-child, excluded body pair,
//! `contype`/`conaffinity`), kept in MuJoCo's contact order, plus the same per-pair
//! bounding-sphere test at run time, gives MuJoCo's contacts in MuJoCo's order. The one
//! theoretical exception is a box-box edge contact accepted at `septol = margin + gap + 1e-13 *
//! (sum of half-sizes)`, which is above the inflation of MuJoCo's broadphase; it is measure-zero.
//!
//! [`build_candidates`] orders the list exactly as MuJoCo orders contacts:
//! - body pairs ascending `(b1, b2)` with `b1 < b2` (the broadphase sorts its pairs by signature
//!   `(min << 16) + max`, and skips repeats);
//! - within a body pair, three branches: two single-geom bodies give one pair; otherwise, unless
//!   `disable.midphase` (and when both bodies have a BVH, which every body with a geom has),
//!   MuJoCo runs its midphase and then a STABLE `contactSort` by `(geom[0], geom[1])`, so the
//!   list holds the geom pairs sorted by that key; otherwise all-to-all, `g1` over the lower
//!   body's geoms (outer) and `g2` over the higher body's (inner);
//! - every pair is oriented so that `type(g1) <= type(g2)`, swapping on strict `>` only (the
//!   contact normal points from `geom[0]` to `geom[1]`); the comparator's "undo the swap" test
//!   compares types and never fires, so the effective key is `(geom[0], geom[1])`.
//!
//! The contact parameters of a pair are static (they read model data only), so [`contact_param`]
//! mixes them here, once, in `f64` with MuJoCo's literal arithmetic, and the model rounds them for
//! `f32`: priority wins; at equal priority `condim` is the larger, `solmix` weights the mix (with
//! the `mjMINVAL` cases), `solref` is mixed (or the elementwise minimum when either side is in the
//! direct format), `solimp` is mixed, and friction is the elementwise maximum, unpacked to `[f0,
//! f0, f1, f2, f2]` and clamped to `mjMINMU = 1e-5`; `solreffriction` is zero.
//!
//! Not ported, and unreachable from an imported scene: explicit `<pair>` entries (their compiled
//! defaults come from MuJoCo compiler code that is not in the reference tree), contact adhesion
//! (refused at import), flex collisions, the `o_*` override options (they need the refused
//! `<flag>`), SDF and height-field geoms, mocap bodies (refused at import: MuJoCo makes a mocap
//! body its own weld root, `body_weldid` here does not) and MuJoCo's convex path
//! ([`crate::NotModelled::Collision`] names every pair that would have used it).

use crate::linalg::{max, min};
use crate::model::{Candidate, Collider, GeomType, Model};
use crate::real::Real;
use crate::smooth::Faults;

/// How many geom pairs of one type pair have no collider (see [`crate::NotModelled`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct CollisionCount {
    pub type1: GeomType,
    pub type2: GeomType,
    pub pairs: usize,
    pub first_pair: (usize, usize),
}

/// What MuJoCo's collision table holds for a pair of geom types (`type1 <= type2`).
enum Entry {
    /// A collider this crate ports.
    Ported(Collider),
    /// A collider MuJoCo has and this crate does not (`mjc_Convex`, `mjc_PlaneConvex`).
    NotPorted,
    /// No collider (`NULL`): plane-plane. MuJoCo drops the pair.
    Null,
}

/// `mjCOLLISIONFUNC[type1][type2]` for the geom types the importer can produce
/// (`type1 <= type2`).
fn collision_entry(t1: GeomType, t2: GeomType) -> Entry {
    use GeomType::{Box, Capsule, Cylinder, Plane, Sphere};
    match (t1, t2) {
        (Plane, Plane) => Entry::Null,
        (Plane, Sphere) => Entry::Ported(Collider::PlaneSphere),
        (Plane, Capsule) => Entry::Ported(Collider::PlaneCapsule),
        (Plane, Cylinder) => Entry::Ported(Collider::PlaneCylinder),
        (Plane, Box) => Entry::Ported(Collider::PlaneBox),
        (Sphere, Sphere) => Entry::Ported(Collider::SphereSphere),
        (Sphere, Capsule) => Entry::Ported(Collider::SphereCapsule),
        (Sphere, Cylinder) => Entry::Ported(Collider::SphereCylinder),
        (Sphere, Box) => Entry::Ported(Collider::SphereBox),
        (Capsule, Capsule) => Entry::Ported(Collider::CapsuleCapsule),
        (Capsule, Box) => Entry::Ported(Collider::CapsuleBox),
        (Box, Box) => Entry::Ported(Collider::BoxBox),
        // everything else is MuJoCo's convex path: plane-ellipsoid, plane-mesh, sphere-
        // ellipsoid, sphere-mesh, capsule-ellipsoid, capsule-cylinder, capsule-mesh, every
        // pair with an ellipsoid or a mesh, cylinder-cylinder, cylinder-box, box-mesh
        _ => Entry::NotPorted,
    }
}

/// Port of `filterBodyPair` (`engine_collision_driver.c`) with the sleep arguments 0: the
/// pair is discarded when both bodies are welded together, when neither has dofs, or when
/// one is the (weld) parent of the other (unless `disable.filterparent`; the world is never
/// parent-filtered).
fn filter_body_pair<R: Real>(m: &Model<R>, b1: usize, b2: usize) -> bool {
    let (weld1, weld2) = (m.body_weldid[b1], m.body_weldid[b2]);

    // the same weld body
    if weld1 == weld2 {
        return true;
    }

    // both dof-less: no forces can act, skip
    if m.body_dofnum[weld1] == 0 && m.body_dofnum[weld2] == 0 {
        return true;
    }

    // the weld parent check
    let weldparent1 = m.body_weldid[m.body_parentid[weld1]];
    let weldparent2 = m.body_weldid[m.body_parentid[weld2]];
    if !m.disable.filterparent
        && weld1 != 0
        && weld2 != 0
        && (weld1 == weldparent2 || weld2 == weldparent1)
    {
        return true;
    }

    // all tests passed
    false
}

/// Builds the static candidate list of `m` (see the module note) in the order of MuJoCo's
/// contacts, with the slot ranges of its pre-contacts and the mixed contact parameters of
/// every pair, and counts the geom pairs that have no collider. `faults.nested_order_everywhere`
/// and `faults.friction_mix_min` are the compile-time test faults.
pub(crate) fn build_candidates<R: Real>(
    m: &Model<R>,
    faults: &Faults,
) -> (Vec<Candidate<R>>, Vec<CollisionCount>) {
    let nbody = m.nbody;
    let mut list: Vec<Candidate<R>> = Vec::new();
    let mut counts: Vec<CollisionCount> = Vec::new();
    let mut slot = 0usize;

    for b1 in 0..nbody {
        if m.body_geomnum[b1] == 0 {
            continue;
        }
        for b2 in b1 + 1..nbody {
            if m.body_geomnum[b2] == 0 {
                continue;
            }
            // the body-pair filters (the broadphase's)
            if filter_body_pair(m, b1, b2) {
                continue;
            }
            let signature = ((b1 as u32) << 16) + b2 as u32;
            if m.exclude_signature.binary_search(&signature).is_ok() {
                continue;
            }

            // the geom pairs of the body pair, oriented, past the bitmask filter
            let (ga, na) = (m.body_geomadr[b1] as usize, m.body_geomnum[b1]);
            let (gb, nb) = (m.body_geomadr[b2] as usize, m.body_geomnum[b2]);
            let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(na * nb);
            for g1 in ga..ga + na {
                for g2 in gb..gb + nb {
                    // filterBitmask: a pair collides when either geom's contype meets the
                    // other's conaffinity
                    let pass = (m.geom_contype[g1] & m.geom_conaffinity[g2]) != 0
                        || (m.geom_contype[g2] & m.geom_conaffinity[g1]) != 0;
                    if !pass {
                        continue;
                    }
                    // pushGeomGeom: orient so that type1 <= type2, swapping on strict >
                    if m.geom_type[g1] > m.geom_type[g2] {
                        pairs.push((g2, g1));
                    } else {
                        pairs.push((g1, g2));
                    }
                }
            }

            // the midphase branch sorts the body pair's contacts (stable) by
            // (geom[0], geom[1]); two single-geom bodies and the all-to-all branch keep the
            // nested order
            let single = na == 1 && nb == 1;
            if !single
                && !m.disable.midphase
                && m.body_has_bvh[b1]
                && m.body_has_bvh[b2]
                && !faults.nested_order_everywhere
            {
                pairs.sort_by_key(|&(a, b)| (a, b));
            }

            for (g1, g2) in pairs {
                let (t1, t2) = (m.geom_type[g1], m.geom_type[g2]);
                match collision_entry(t1, t2) {
                    Entry::Null => {}
                    Entry::NotPorted => {
                        match counts.iter_mut().find(|c| c.type1 == t1 && c.type2 == t2) {
                            Some(c) => c.pairs += 1,
                            None => counts.push(CollisionCount {
                                type1: t1,
                                type2: t2,
                                pairs: 1,
                                first_pair: (g1, g2),
                            }),
                        }
                    }
                    Entry::Ported(collider) => {
                        let (condim, solref, solimp, friction) =
                            contact_param(m, g1, g2, faults.friction_mix_min);
                        // getMargin, getGap: sums of the geoms' (in f64, rounded once)
                        let margin = m.geom_margin[g1].to_f64() + m.geom_margin[g2].to_f64();
                        let gap = m.geom_gap[g1].to_f64() + m.geom_gap[g2].to_f64();
                        let slot_count = collider.max_contacts();
                        list.push(Candidate {
                            g1,
                            g2,
                            collider,
                            slot_offset: slot,
                            slot_count,
                            margin_gap: R::from_f64(margin + gap),
                            includemargin: R::from_f64(margin),
                            gap: R::from_f64(gap),
                            condim,
                            solref: solref.map(R::from_f64),
                            solreffriction: [R::ZERO; 2],
                            solimp: solimp.map(R::from_f64),
                            friction: friction.map(R::from_f64),
                        });
                        slot += slot_count;
                    }
                }
            }
        }
    }
    counts.sort_by_key(|c| (c.type1, c.type2));
    (list, counts)
}

/// FNV-1a (64 bit) over the geoms, colliders and slot ranges of a candidate list.
pub(crate) fn fingerprint<R: Real>(list: &[Candidate<R>]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    let mut put = |v: u64| {
        for byte in v.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    };
    put(list.len() as u64);
    for c in list {
        put(c.g1 as u64);
        put(c.g2 as u64);
        put(c.collider as u64);
        put(c.slot_offset as u64);
        put(c.slot_count as u64);
    }
    h
}

/// MuJoCo's `mjMINMU`: the smallest friction coefficient of a contact.
pub(crate) const MIN_MU: f64 = 1e-5;
/// MuJoCo's `mjMINVAL` as an `f64`, for the compile-time mixing.
const MIN_VAL: f64 = 1e-15;

/// The condim, `solref`, `solimp` and 5-vector friction of a contact between geoms `g1`
/// and `g2`: port of `mj_contactParam` (without adhesion) followed by the clamp of
/// `mj_assignFriction`. `friction_min` is the test-only fault that mixes friction by the
/// minimum instead of the maximum.
pub(crate) fn contact_param<R: Real>(
    m: &Model<R>,
    g1: usize,
    g2: usize,
    friction_min: bool,
) -> (usize, [f64; 2], [f64; 5], [f64; 5]) {
    let w = |a: &[R], i: usize, n: usize| -> Vec<f64> {
        a[n * i..n * i + n].iter().map(|x| x.to_f64()).collect()
    };
    let priority1 = m.geom_priority[g1];
    let priority2 = m.geom_priority[g2];
    let condim1 = m.geom_condim[g1] as usize;
    let condim2 = m.geom_condim[g2] as usize;
    let solmix1 = m.geom_solmix[g1].to_f64();
    let solmix2 = m.geom_solmix[g2].to_f64();
    let (solref1, solref2) = (w(&m.geom_solref, g1, 2), w(&m.geom_solref, g2, 2));
    let (solimp1, solimp2) = (w(&m.geom_solimp, g1, 5), w(&m.geom_solimp, g2, 5));
    let (friction1, friction2) = (w(&m.geom_friction, g1, 3), w(&m.geom_friction, g2, 3));

    let condim;
    let mut solref = [0.0f64; 2];
    let mut solimp = [0.0f64; 5];
    let mut fri = [0.0f64; 3];

    // different priority: copy from the item with the higher priority
    if priority1 > priority2 {
        condim = condim1;
        solref.copy_from_slice(&solref1);
        solimp.copy_from_slice(&solimp1);
        fri.copy_from_slice(&friction1);
    } else if priority1 < priority2 {
        condim = condim2;
        solref.copy_from_slice(&solref2);
        solimp.copy_from_slice(&solimp2);
        fri.copy_from_slice(&friction2);
    }
    // same priority
    else {
        // condim: max
        condim = condim1.max(condim2);

        // the solver mix factor
        let mix = if solmix1 >= MIN_VAL && solmix2 >= MIN_VAL {
            solmix1 / (solmix1 + solmix2)
        } else if solmix1 < MIN_VAL && solmix2 < MIN_VAL {
            0.5
        } else if solmix1 < MIN_VAL {
            0.0
        } else {
            1.0
        };

        // the reference, standard: mix
        if solref1[0] > 0.0 && solref2[0] > 0.0 {
            for i in 0..2 {
                solref[i] = mix * solref1[i] + (1.0 - mix) * solref2[i];
            }
        }
        // the reference, direct: min
        else {
            for i in 0..2 {
                solref[i] = min(solref1[i], solref2[i]);
            }
        }

        // the impedance: mix
        for i in 0..5 {
            solimp[i] = mix * solimp1[i] + (1.0 - mix) * solimp2[i];
        }

        // friction: max
        for i in 0..3 {
            fri[i] = if friction_min {
                min(friction1[i], friction2[i])
            } else {
                max(friction1[i], friction2[i])
            };
        }
    }

    // unpack the 5D friction, clamped to mjMINMU (mj_assignFriction)
    let unpacked = [fri[0], fri[0], fri[1], fri[2], fri[2]];
    let mut friction = [0.0f64; 5];
    for i in 0..5 {
        friction[i] = max(MIN_MU, unpacked[i]);
    }
    (condim, solref, solimp, friction)
}
