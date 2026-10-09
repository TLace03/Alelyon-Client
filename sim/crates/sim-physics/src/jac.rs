//! Jacobians of points on bodies: the dense `mj_jac` and `mj_jacDifPair`.
//!
//! Ports, from MuJoCo 3.14.0 `engine_core_util.c`: `mj_jac` (the 3-by-`nv` translation
//! and rotation Jacobians of a world point attached to a body) and the dense branch of
//! `mj_jacDifPair` (the difference of the Jacobians of two points, `jac(body 2) -
//! jac(body 1)`, which a contact row uses). The sparse Jacobian, the sparse merged dof
//! chains and the "simple body" fast path are not ported: this crate's Jacobian is
//! dense (phase 1c-i), and the dense path gives the same numbers.
//!
//! Invariants:
//! - **No allocation**: every function writes slices the caller owns (the preallocated
//!   scratch of [`crate::Data`]); they take the two arrays they read (`cdof`,
//!   `subtree_com`) as slices so the caller can borrow the scratch from the same `Data`.
//! - **Same order as MuJoCo**: the dofs of a body's ancestor chain are visited child to
//!   parent, and each column is `cdof` plus the cross product with the point's offset
//!   from the centre of mass of the body's tree, as in the C source. Plain multiply and
//!   add only.
//! - A body welded to the world (no dofs on its weld body) has a zero Jacobian.

use crate::math::{cross, sub3, v3, v6};
use crate::model::Model;
use crate::real::Real;

/// Port of `mj_jac`: the translation Jacobian `jacp` and the rotation Jacobian `jacr`
/// (each 3-by-`nv`, row-major, cleared first) of the world point `point` attached to
/// `body`. Either output may be `None`. `cdof` and `subtree_com` are the arrays of
/// [`crate::Data`] that [`crate::com_pos`] fills.
pub(crate) fn jac<R: Real>(
    m: &Model<R>,
    cdof: &[R],
    subtree_com: &[R],
    jacp: Option<&mut [R]>,
    jacr: Option<&mut [R]>,
    point: [R; 3],
    body: usize,
) {
    let nv = m.nv;
    let mut jacp = jacp;
    let mut jacr = jacr;

    // clear the Jacobians, compute the offset if required
    let mut offset = [R::ZERO; 3];
    if let Some(p) = jacp.as_deref_mut() {
        p[..3 * nv].fill(R::ZERO);
        offset = sub3(point, v3(subtree_com, m.body_rootid[body]));
    }
    if let Some(r) = jacr.as_deref_mut() {
        r[..3 * nv].fill(R::ZERO);
    }

    // skip fixed bodies
    let body = m.body_weldid[body];

    // the weld root has no dofs: nothing to do
    if m.body_dofnum[body] == 0 {
        return;
    }

    // the last dof that affects this (as well as the original) body
    let mut i = (m.body_dofadr[body] + m.body_dofnum[body]) as i32 - 1;

    // backward pass over the dof ancestor chain
    while i >= 0 {
        let iu = i as usize;
        let c = v6(cdof, iu);

        // rotation Jacobian
        if let Some(r) = jacr.as_deref_mut() {
            r[iu] = c[0];
            r[iu + nv] = c[1];
            r[iu + 2 * nv] = c[2];
        }

        // translation Jacobian (corrected for the rotation)
        if let Some(p) = jacp.as_deref_mut() {
            let tmp = cross([c[0], c[1], c[2]], offset);
            p[iu] = c[3] + tmp[0];
            p[iu + nv] = c[4] + tmp[1];
            p[iu + 2 * nv] = c[5] + tmp[2];
        }

        // advance to the parent dof
        i = m.dof_parentid[iu];
    }
}

/// The scratch rows [`jac_dif_pair`] writes (all preallocated in [`crate::Data`]).
pub(crate) struct DifScratch<'a, R: Real> {
    /// `jac(body 1)`, translation, `3 * nv`.
    pub jac1p: &'a mut [R],
    /// `jac(body 2)`, translation, `3 * nv`.
    pub jac2p: &'a mut [R],
    /// `jac(body 2) - jac(body 1)`, translation, `3 * nv`.
    pub jacdifp: &'a mut [R],
    /// The same three for the rotation, written only when `rot` is true.
    pub jac1r: &'a mut [R],
    /// `jac(body 2)`, rotation.
    pub jac2r: &'a mut [R],
    /// `jac(body 2) - jac(body 1)`, rotation.
    pub jacdifr: &'a mut [R],
}

/// Port of the dense branch of `mj_jacDifPair` with `flg_skipcommon` set (which only
/// matters for the sparse chains): the Jacobians of the points `pos1` on `b1` and `pos2`
/// on `b2` and their differences `jac(b2) - jac(b1)`, rotation included when `rot`.
/// Returns the number of columns, `nv` (0 when the model has no dofs).
#[allow(clippy::too_many_arguments)]
pub(crate) fn jac_dif_pair<R: Real>(
    m: &Model<R>,
    cdof: &[R],
    subtree_com: &[R],
    b1: usize,
    b2: usize,
    pos1: [R; 3],
    pos2: [R; 3],
    rot: bool,
    s: &mut DifScratch<'_, R>,
) -> usize {
    let nv = m.nv;
    // skip if no dofs
    if nv == 0 {
        return 0;
    }
    if rot {
        jac(
            m,
            cdof,
            subtree_com,
            Some(&mut *s.jac1p),
            Some(&mut *s.jac1r),
            pos1,
            b1,
        );
        jac(
            m,
            cdof,
            subtree_com,
            Some(&mut *s.jac2p),
            Some(&mut *s.jac2r),
            pos2,
            b2,
        );
    } else {
        jac(m, cdof, subtree_com, Some(&mut *s.jac1p), None, pos1, b1);
        jac(m, cdof, subtree_com, Some(&mut *s.jac2p), None, pos2, b2);
    }

    // differences (mju_sub)
    for k in 0..3 * nv {
        s.jacdifp[k] = s.jac2p[k] - s.jac1p[k];
    }
    if rot {
        for k in 0..3 * nv {
            s.jacdifr[k] = s.jac2r[k] - s.jac1r[k];
        }
    }
    nv
}
