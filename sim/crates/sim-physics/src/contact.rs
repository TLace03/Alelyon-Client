//! Contact frames and contact forces.
//!
//! Ports, from MuJoCo 3.14.0:
//! - `engine_util_spatial.c`: `mju_makeFrame` (completes a contact frame from its normal and
//!   optional first tangent);
//! - `engine_core_util.c`: `mj_contactForce` (the force of one contact in its frame);
//! - `engine_util_misc.c`: `mju_decodePyramid` (the pyramidal row forces to a contact force).
//!
//! The mixing of a contact's parameters from its two geoms (`mj_contactParam`) is static and is
//! done at compile time (`candidates.rs`).
//!
//! The contact arrays are in [`crate::Data`]: struct-of-arrays sized [`crate::Model::ncon_max`],
//! valid for `0..ncon`, in MuJoCo's contact order (body pairs ascending, then the order of the
//! candidate list within a body pair, then the order the collider emits). A contact's frame is a
//! row-major 3-by-3 whose rows are the normal (from `geom[0]` to `geom[1]`), the first tangent and
//! the second tangent.
//!
//! Invariants: **no allocation**; plain multiply and add only.
//!
//! Not ported, and unreachable from an imported scene: contact adhesion (refused at import;
//! `mj_contactForce` would subtract it).

use sim_scene::Cone;

use crate::data::Data;
use crate::math::{cross, dot3, normalize3, scl3, sub3};
use crate::model::Model;
use crate::real::Real;

/// Port of `mju_makeFrame`: completes the frame `frame` (row-major 3-by-3) whose first row
/// is the contact normal and whose second row is the first tangent or zero: the normal is
/// normalised; an undefined tangent becomes `(0, 1, 0)` when the normal's y component is
/// within 0.5 of zero, else `(0, 0, 1)`; the tangent is made orthogonal to the normal and
/// normalised (a tangent that was the normal itself, an exactly upright capsule's axis,
/// cancels to zero and `mju_normalize3` turns that into `(1, 0, 0)`); the second tangent is
/// their cross product. (MuJoCo's `mjERROR` for a normal shorter than 0.5 is a
/// `debug_assert` here.)
pub(crate) fn make_frame<R: Real>(frame: &mut [R]) {
    let half = R::from_f64(0.5);
    let quarter = R::from_f64(0.25);

    // normalise the x axis
    let mut x = [frame[0], frame[1], frame[2]];
    let norm = normalize3(&mut x);
    debug_assert!(norm >= half, "the x axis of a contact frame is undefined");
    frame[0..3].copy_from_slice(&x);

    // if the y axis is undefined, set it to (0, 1, 0) if possible, otherwise (0, 0, 1)
    let mut y = [frame[3], frame[4], frame[5]];
    if dot3(y, y) < quarter {
        y = [R::ZERO; 3];
        if frame[1] < half && frame[1] > -half {
            y[1] = R::ONE;
        } else {
            y[2] = R::ONE;
        }
    }

    // make the y axis orthogonal to the x axis
    let tmp = scl3(x, dot3(x, y));
    y = sub3(y, tmp);
    normalize3(&mut y);
    frame[3..6].copy_from_slice(&y);

    // z axis = cross(x axis, y axis)
    frame[6..9].copy_from_slice(&cross(x, y));
}

/// Port of `mju_decodePyramid`: the contact force `[normal, tangents...]` from the forces
/// of a pyramidal contact's `2 * (dim - 1)` rows (`pyramid`) and its friction `mu`. A
/// frictionless contact has the one force.
pub(crate) fn decode_pyramid<R: Real>(force: &mut [R], pyramid: &[R], mu: &[R], dim: usize) {
    // special handling of frictionless contacts
    if dim == 1 {
        force[0] = pyramid[0];
        return;
    }

    // force_normal = sum(pyramid0_i + pyramid1_i)
    force[0] = R::ZERO;
    for p in pyramid.iter().take(2 * (dim - 1)) {
        force[0] += *p;
    }

    // force_tangent_i = (pyramid0_i - pyramid1_i) * mu_i
    for i in 0..dim - 1 {
        force[i + 1] = (pyramid[2 * i] - pyramid[2 * i + 1]) * mu[i];
    }
}

/// Port of `mj_contactForce`: the force of contact `i` in the contact frame,
/// `[normal, tangent 1, tangent 2, torsional, rolling 1, rolling 2]` (the entries past the
/// contact's dimension are zero), from the constraint forces of the last forward pass. A
/// contact that has no rows (an excluded one, or an index out of range) gives zeros.
/// Contact adhesion is zero here (adhesion is refused at import).
pub fn contact_force<R: Real>(m: &Model<R>, d: &Data<R>, i: usize) -> [R; 6] {
    let mut result = [R::ZERO; 6];

    // make sure the contact is valid
    if i < d.ncon && d.contact_efc_address[i] >= 0 {
        let adr = d.contact_efc_address[i] as usize;
        let dim = d.contact_dim[i];
        if m.opt.cone == Cone::Pyramidal {
            decode_pyramid(
                &mut result,
                &d.efc_force[adr..],
                &d.contact_friction[5 * i..5 * i + 5],
                dim,
            );
        } else {
            result[..dim].copy_from_slice(&d.efc_force[adr..adr + dim]);
        }
    }
    result
}
