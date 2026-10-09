//! The forward pass of the smooth dynamics: kinematics, composite rigid body
//! inertia, velocities, Newton-Euler bias forces, passive and actuator forces,
//! and the joint accelerations.
//!
//! Ports, from MuJoCo 3.14.0:
//! - `engine_core_smooth.c`: `mj_kinematics1` and the inertial-frame part of
//!   `mj_kinematics2`, `mj_comPos`, `mj_crb`, `mj_comVel`, `mj_rne`, and `mj_tendon`
//!   (fixed tendons: length and Jacobian), the tendon velocity of `mj_fwdVelocity`;
//! - `engine_forward.c`: `mj_fwdPosition`, `mj_fwdVelocity`, `mj_fwdActuation`,
//!   `mj_fwdAcceleration`, `mj_fwdConstraint` (in `constraint.rs`), in that order, and
//!   `mj_forwardSkip`;
//! - `engine_passive.c`: `mj_springdamper` (joint springs and dof dampers).
//!
//! Invariants:
//! - **No allocation**: every function reads the [`Model`] and writes the
//!   preallocated arrays of [`Data`]; locals are fixed-size arrays.
//! - **Plain multiply and add only**: no fused multiply-add, in the operation
//!   order of the C source ([`crate::Real`] says why).
//! - **Same order as MuJoCo**: bodies are visited in index order (parents first)
//!   for forward passes and in reverse for the leaves-first accumulations; dofs
//!   in index order; actuators in index order. A function states where it
//!   differs from the C source.
//! - **Constraints** (phase 1c-i): the forward pass computes `qacc_smooth`, the
//!   acceleration with no constraint force, then builds the constraint rows (joint
//!   and tendon limits, joint and tendon friction loss, in `constraint.rs`) and
//!   solves them (`solver.rs`); `qacc` is `qacc_smooth` when there are no rows and
//!   the solver's result otherwise. With the model's `disable.constraint` set (MuJoCo's
//!   `mjDSBL_CONSTRAINT`, the setting of the phase-1b golden files) `qacc` is always
//!   `qacc_smooth`.
//! - **Contacts** (phase 1c-ii): `kinematics` places the inertial frames through
//!   `mj_local2Global` with `body_sameframe` (an inertial frame within `kFrameEps = 1e-6` of the
//!   body frame is snapped onto it, which MuJoCo does and the phase-1b port did not) and also the
//!   geoms (`geom_xpos`, `geom_xmat`, with `geom_sameframe`), and the forward pass runs the collision step
//!   (`collision.rs`) between the factorisation of `M` and the constraint rows, as `mj_fwdPosition`
//!   does; `crb` writes a simple dof's row of `M` as the constant `dof_M0`, as `mj_crb` does.
//! - Not ported: the divergence guards of `mj_step` (`mj_checkPos`, `mj_checkVel`,
//!   `mj_checkAcc`, which reset the data on a NaN or a value above 1e10), warnings,
//!   sleeping, user callbacks, plugins, activation states, actuator delays,
//!   `qfrc_applied` and `xfrc_applied` (always zero here), gravity compensation,
//!   fluid forces, a tendon's spring, damper and armature (`mj_tendonBias`), and the
//!   other actuator kinds.

use crate::constraint;
use crate::data::Data;
use crate::factor;
use crate::math::{
    add3, axis_angle_to_quat, cross_force, cross_motion, dof_com_hinge, dof_com_slide, dot6,
    inert_com, lit, min_val, mul_inert_vec, mul_mat_vec3, mul_quat, normalize4, quat_to_mat,
    rot_vec_quat, scl3, sub_quat, sub3, v3, v4, v6, v9, v10,
};
use crate::model::{ActuatorType, JointType, Model, SameFrame};
use crate::real::Real;

/// Test-only fault injection for the negative controls of the test suite.
///
/// Each flag breaks one piece of the engine in a known way so that a test can
/// assert that the comparison with MuJoCo catches it. The public entry points
/// ([`crate::step`], [`crate::forward`], ...) pass [`Faults::NONE`]; this type is
/// reachable only through the hidden `*_faulted` functions.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Faults {
    /// Flip the sign of the Coriolis and centrifugal term of `rne`.
    pub flip_coriolis_sign: bool,
    /// Leave the armature out of the diagonal of `M`.
    pub drop_armature: bool,
    /// Integrate Euler explicitly even when joints have damping.
    pub euler_without_implicit_damping: bool,
    /// Flip the sign of the constraints' reference acceleration `efc_aref`.
    pub flip_aref_sign: bool,
    /// Do not instantiate the joint and tendon friction-loss rows.
    pub drop_friction_rows: bool,
    /// Replace the approximate inverse inertia `diagApprox` of every row by 1.
    pub diag_approx_one: bool,
    /// Replace the solvers' exact line search by a step of length 1.
    pub unit_step_line_search: bool,
    /// Flip the normal of every contact (before its frame is completed).
    pub flip_contact_normal: bool,
    /// Combine the friction of a contact's two geoms by the elementwise minimum instead
    /// of MuJoCo's maximum (a compile-time fault, applied by
    /// [`crate::faults::compile_faulted`]).
    pub friction_mix_min: bool,
    /// Drop the last contact of every box-box pair.
    pub drop_last_boxbox_contact: bool,
    /// Read `impratio` as 1 in the contact rows' regularisation.
    pub ignore_impratio: bool,
    /// Set a contact's `includemargin` to `margin - gap` instead of `margin`.
    pub includemargin_minus_gap: bool,
    /// Order the geom pairs of every body pair as the all-to-all loop does, even where
    /// MuJoCo's midphase sorts them (a compile-time fault, applied by
    /// [`crate::faults::compile_faulted`]).
    pub nested_order_everywhere: bool,
    /// Leave the elliptic cone's Hessian out of the Newton solver's factor.
    pub drop_cone_hessian: bool,
}

impl Faults {
    /// No fault.
    pub const NONE: Faults = Faults {
        flip_coriolis_sign: false,
        drop_armature: false,
        euler_without_implicit_damping: false,
        flip_aref_sign: false,
        drop_friction_rows: false,
        diag_approx_one: false,
        unit_step_line_search: false,
        flip_contact_normal: false,
        friction_mix_min: false,
        drop_last_boxbox_contact: false,
        ignore_impratio: false,
        includemargin_minus_gap: false,
        nested_order_everywhere: false,
        drop_cone_hessian: false,
    };
}

/// Port of `mj_kinematics1` and the body inertial frames of `mj_kinematics2`:
/// the world pose of every body frame (`xpos`, `xquat`, `xmat`), the joint anchors
/// and axes in the world (`xanchor`, `xaxis`), and the centre-of-mass frames
/// (`xipos`, `ximat`).
///
/// A free joint takes the body pose straight from `qpos` (the quaternion
/// normalised), as MuJoCo does; every other body composes its parent's pose, its
/// own fixed offset and each of its joints in order. Quaternions in `qpos` are
/// read normalised, never written back.
pub fn kinematics<R: Real>(m: &Model<R>, d: &mut Data<R>) {
    // the world body
    d.xpos[0..3].fill(R::ZERO);
    d.xquat[0..4].copy_from_slice(&[R::ONE, R::ZERO, R::ZERO, R::ZERO]);
    d.xipos[0..3].fill(R::ZERO);
    d.xmat[0..9].fill(R::ZERO);
    d.ximat[0..9].fill(R::ZERO);
    for k in [0, 4, 8] {
        d.xmat[k] = R::ONE;
        d.ximat[k] = R::ONE;
    }

    for i in 1..m.nbody {
        let jntadr = m.body_jntadr[i];
        let jntnum = m.body_jntnum[i];
        let mut xpos;
        let mut xquat;

        if jntnum == 1 && m.jnt_type[jntadr] == JointType::Free {
            // free joint: copy pos and quat from qpos
            let qadr = m.jnt_qposadr[jntadr];
            xpos = [d.qpos[qadr], d.qpos[qadr + 1], d.qpos[qadr + 2]];
            xquat = [
                d.qpos[qadr + 3],
                d.qpos[qadr + 4],
                d.qpos[qadr + 5],
                d.qpos[qadr + 6],
            ];
            normalize4(&mut xquat);
            d.xanchor[3 * jntadr..3 * jntadr + 3].copy_from_slice(&xpos);
            d.xaxis[3 * jntadr..3 * jntadr + 3]
                .copy_from_slice(&m.jnt_axis[3 * jntadr..3 * jntadr + 3]);
        } else {
            // regular or no joint: the fixed offset from the parent
            let pid = m.body_parentid[i];
            let bodypos = v3(&m.body_pos, i);
            let bodyquat = v4(&m.body_quat, i);
            if pid != 0 {
                let pmat = v9(&d.xmat, pid);
                xpos = mul_mat_vec3(&pmat, bodypos);
                xpos = add3(xpos, v3(&d.xpos, pid));
                xquat = mul_quat(v4(&d.xquat, pid), bodyquat);
            } else {
                xpos = bodypos;
                xquat = bodyquat;
            }

            // accumulate the joints
            for jid in jntadr..jntadr + jntnum {
                let qadr = m.jnt_qposadr[jid];
                let jtype = m.jnt_type[jid];

                // axis and anchor in the global frame (before this joint moves)
                let xaxis = rot_vec_quat(v3(&m.jnt_axis, jid), xquat);
                let mut xanchor = rot_vec_quat(v3(&m.jnt_pos, jid), xquat);
                xanchor = add3(xanchor, xpos);

                match jtype {
                    JointType::Slide => {
                        let s = d.qpos[qadr] - m.qpos0[qadr];
                        xpos[0] += xaxis[0] * s;
                        xpos[1] += xaxis[1] * s;
                        xpos[2] += xaxis[2] * s;
                    }
                    JointType::Ball | JointType::Hinge => {
                        // the local rotation
                        let qloc = if jtype == JointType::Ball {
                            let mut q = [
                                d.qpos[qadr],
                                d.qpos[qadr + 1],
                                d.qpos[qadr + 2],
                                d.qpos[qadr + 3],
                            ];
                            normalize4(&mut q);
                            q
                        } else {
                            axis_angle_to_quat(v3(&m.jnt_axis, jid), d.qpos[qadr] - m.qpos0[qadr])
                        };
                        xquat = mul_quat(xquat, qloc);
                        // correct for the off-centre rotation
                        let vec = rot_vec_quat(v3(&m.jnt_pos, jid), xquat);
                        xpos = sub3(xanchor, vec);
                    }
                    // a free joint is alone on its body (validated), handled above
                    JointType::Free => {}
                }
                d.xanchor[3 * jid..3 * jid + 3].copy_from_slice(&xanchor);
                d.xaxis[3 * jid..3 * jid + 3].copy_from_slice(&xaxis);
            }
        }

        normalize4(&mut xquat);
        d.xquat[4 * i..4 * i + 4].copy_from_slice(&xquat);
        d.xpos[3 * i..3 * i + 3].copy_from_slice(&xpos);
        d.xmat[9 * i..9 * i + 9].copy_from_slice(&quat_to_mat(xquat));
    }

    // body inertial frames (mj_kinematics2 via mj_local2Global with body_sameframe: an
    // inertial frame within kFrameEps of the body frame is snapped onto it, as MuJoCo does)
    for i in 1..m.nbody {
        let (p, mat) = local_to_global(
            m,
            d,
            i,
            v3(&m.body_ipos, i),
            v4(&m.body_iquat, i),
            m.body_sameframe[i],
        );
        d.xipos[3 * i..3 * i + 3].copy_from_slice(&p);
        d.ximat[9 * i..9 * i + 9].copy_from_slice(&mat);
    }

    // geom frames (mj_kinematics2 via mj_local2Global with geom_sameframe): bodies in
    // order, each body's geoms in order
    for b in 0..m.nbody {
        let num = m.body_geomnum[b];
        if num == 0 {
            continue;
        }
        let start = m.body_geomadr[b] as usize;
        for g in start..start + num {
            let pos = v3(&m.geom_pos, g);
            let quat = v4(&m.geom_quat, g);
            let (xpos, xmat) = local_to_global(m, d, b, pos, quat, m.geom_sameframe[g]);
            d.geom_xpos[3 * g..3 * g + 3].copy_from_slice(&xpos);
            d.geom_xmat[9 * g..9 * g + 9].copy_from_slice(&xmat);
        }
    }
}

/// Port of `mj_local2Global`: the world position and rotation matrix of the frame
/// `(pos, quat)` in body `body`, using what the compiler found out about the frame
/// (`sameframe`): the frame of the body or of its centre of mass is copied, a frame that
/// differs only by position (`BodyRot`, `InertiaRot`) takes the body's or inertial
/// rotation, and any other is composed (`mj_local2Global`, `engine_core_util.c`).
/// Needs `kinematics` to have written the body and inertial frames.
pub(crate) fn local_to_global<R: Real>(
    m: &Model<R>,
    d: &Data<R>,
    body: usize,
    pos: [R; 3],
    quat: [R; 4],
    sameframe: SameFrame,
) -> ([R; 3], [R; 9]) {
    let _ = m;
    let xpos = match sameframe {
        SameFrame::None | SameFrame::BodyRot | SameFrame::InertiaRot => {
            let p = mul_mat_vec3(&v9(&d.xmat, body), pos);
            add3(p, v3(&d.xpos, body))
        }
        SameFrame::Body => v3(&d.xpos, body),
        SameFrame::Inertia => v3(&d.xipos, body),
    };
    let xmat = match sameframe {
        SameFrame::None => quat_to_mat(mul_quat(v4(&d.xquat, body), quat)),
        SameFrame::Body | SameFrame::BodyRot => v9(&d.xmat, body),
        SameFrame::Inertia | SameFrame::InertiaRot => v9(&d.ximat, body),
    };
    (xpos, xmat)
}

/// Port of `mj_comPos`: maps the body inertias and the dof motion axes to the
/// frame centred at each tree's centre of mass (`subtree_com`, `cinert`, `cdof`).
pub fn com_pos<R: Real>(m: &Model<R>, d: &mut Data<R>) {
    // subtree_com: initialise with the body moment
    for i in 0..m.nbody {
        let p = scl3(v3(&d.xipos, i), m.body_mass[i]);
        d.subtree_com[3 * i..3 * i + 3].copy_from_slice(&p);
    }
    // accumulate to the parent in a backward pass
    for i in (1..m.nbody).rev() {
        let parent = m.body_parentid[i];
        let s = add3(v3(&d.subtree_com, parent), v3(&d.subtree_com, i));
        d.subtree_com[3 * parent..3 * parent + 3].copy_from_slice(&s);
    }
    // normalise
    for i in 0..m.nbody {
        if m.body_subtreemass[i] < min_val::<R>() {
            let x = v3(&d.xipos, i);
            d.subtree_com[3 * i..3 * i + 3].copy_from_slice(&x);
        } else {
            let inv = R::ONE / m.body_subtreemass[i];
            let s = scl3(v3(&d.subtree_com, i), inv);
            d.subtree_com[3 * i..3 * i + 3].copy_from_slice(&s);
        }
    }

    // zero the CoM-frame inertia of the world body
    d.cinert[0..10].fill(R::ZERO);

    // map the inertias to the frame centred at subtree_com
    for i in 1..m.nbody {
        let offset = sub3(v3(&d.xipos, i), v3(&d.subtree_com, m.body_rootid[i]));
        let c = inert_com(
            v3(&m.body_inertia, i),
            &v9(&d.ximat, i),
            offset,
            m.body_mass[i],
        );
        d.cinert[10 * i..10 * i + 10].copy_from_slice(&c);
    }

    // map the motion dofs to the frame centred at subtree_com
    for i in 1..m.nbody {
        for j in m.body_jntadr[i]..m.body_jntadr[i] + m.body_jntnum[i] {
            let da = m.jnt_dofadr[j];
            // com-anchor vector
            let offset = sub3(v3(&d.subtree_com, m.body_rootid[i]), v3(&d.xanchor, j));
            let mut first_rot = da;
            match m.jnt_type[j] {
                JointType::Free => {
                    // translation components: x, y, z in the global frame
                    for k in 0..3 {
                        let mut c = [R::ZERO; 6];
                        c[3 + k] = R::ONE;
                        d.cdof[6 * (da + k)..6 * (da + k) + 6].copy_from_slice(&c);
                    }
                    first_rot = da + 3;
                }
                JointType::Slide => {
                    let c = dof_com_slide(v3(&d.xaxis, j));
                    d.cdof[6 * da..6 * da + 6].copy_from_slice(&c);
                    continue;
                }
                JointType::Hinge => {
                    let c = dof_com_hinge(v3(&d.xaxis, j), offset);
                    d.cdof[6 * da..6 * da + 6].copy_from_slice(&c);
                    continue;
                }
                JointType::Ball => {}
            }
            // ball, and the rotation of a free joint: the identity in the child
            // frame (no subsequent rotations), i.e. the columns of xmat
            for k in 0..3 {
                let axis = [
                    d.xmat[9 * i + k],
                    d.xmat[9 * i + k + 3],
                    d.xmat[9 * i + k + 6],
                ];
                let c = dof_com_hinge(axis, offset);
                d.cdof[6 * (first_rot + k)..6 * (first_rot + k) + 6].copy_from_slice(&c);
            }
        }
    }
}

/// Port of `mj_crb`: the composite rigid body inertia (`crb`) and the joint-space
/// inertia matrix `qm` (dense, symmetric) with the armature on the diagonal.
pub fn crb<R: Real>(m: &Model<R>, d: &mut Data<R>) {
    crb_with(m, d, &Faults::NONE);
}

pub(crate) fn crb_with<R: Real>(m: &Model<R>, d: &mut Data<R>, faults: &Faults) {
    let nv = m.nv;
    // crb = cinert
    d.crb.copy_from_slice(&d.cinert);
    // backward pass over bodies, accumulate composite inertias
    for i in (1..m.nbody).rev() {
        let p = m.body_parentid[i];
        if p > 0 {
            for k in 0..10 {
                let s = d.crb[10 * p + k] + d.crb[10 * i + k];
                d.crb[10 * p + k] = s;
            }
        }
    }

    d.qm.fill(R::ZERO);
    // forward pass over dofs
    for i in 0..nv {
        // a simple dof: fixed diagonal inertia (MuJoCo's `M[adr] = dof_M0[i]`); its row has no
        // other entry. (The test-only `drop_armature` fault takes the general path, so that
        // it still acts on a simple dof.)
        if m.dof_simplenum[i] != 0 && !faults.drop_armature {
            d.qm[i * nv + i] = m.dof_m0[i];
            continue;
        }

        // M(i,i) starts at the armature
        let armature = if faults.drop_armature {
            R::ZERO
        } else {
            m.dof_armature[i]
        };
        d.qm[i * nv + i] = armature;

        // buf = crb_body_i * cdof_i
        let crb_i = v10(&d.crb, m.dof_bodyid[i]);
        let buf = mul_inert_vec(&crb_i, v6(&d.cdof, i));

        // backward pass over the ancestors: M(i,j) += cdof_j * (crb_body_i * cdof_i)
        let mut j = i as i32;
        while j >= 0 {
            let ju = j as usize;
            let s = d.qm[i * nv + ju] + dot6(v6(&d.cdof, ju), buf);
            d.qm[i * nv + ju] = s;
            d.qm[ju * nv + i] = s;
            j = m.dof_parentid[ju];
        }
    }
}

/// Factors `M` into `qld` and `qld_diag_inv` (the `mj_factorM` of `mj_fwdPosition`).
pub(crate) fn factor_m<R: Real>(m: &Model<R>, d: &mut Data<R>) {
    d.qld.copy_from_slice(&d.qm);
    factor::factor(&mut d.qld, &mut d.qld_diag_inv, m.nv);
}

/// `sum_{r < n} dof_r * vec_r`, a 6-vector (`mju_mulDofVec`): `dof` is `n`
/// consecutive 6-vectors starting at `6 * first`, `vec` the `n` numbers starting
/// at `first`. MuJoCo skips the terms whose `vec_r` is zero; the sum is the same
/// for finite `dof`, and no branch depends on the data here.
fn mul_dof_vec<R: Real>(dof: &[R], vec: &[R], first: usize, n: usize) -> [R; 6] {
    let mut res = [R::ZERO; 6];
    for r in 0..n {
        let t = vec[first + r];
        for (c, out) in res.iter_mut().enumerate() {
            *out += dof[6 * (first + r) + c] * t;
        }
    }
    res
}

fn add6<R: Real>(a: [R; 6], b: [R; 6]) -> [R; 6] {
    [
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3],
        a[4] + b[4],
        a[5] + b[5],
    ]
}

/// Port of `mj_comVel`: the body velocities in the CoM-centred frame (`cvel`)
/// and the time derivatives of the dof axes (`cdof_dot`).
pub fn com_vel<R: Real>(m: &Model<R>, d: &mut Data<R>) {
    d.cvel[0..6].fill(R::ZERO);

    for i in 1..m.nbody {
        // cvel = cvel_parent
        let mut cvel = v6(&d.cvel, m.body_parentid[i]);

        // cvel = cvel_parent + cdof * qvel,  cdofdot = cvel x cdof
        let dofnum = m.body_dofnum[i];
        let bda = m.body_dofadr[i];
        let mut cdofdot = [[R::ZERO; 6]; 6];
        let mut j = 0;
        while j < dofnum {
            let jtype = m.jnt_type[m.dof_jntid[bda + j]];
            match jtype {
                JointType::Free | JointType::Ball => {
                    if jtype == JointType::Free {
                        // the translations: cdofdot = 0, then update the velocity
                        let tmp = mul_dof_vec(&d.cdof, &d.qvel, bda + j, 3);
                        cvel = add6(cvel, tmp);
                        j += 3;
                    }
                    // all three cdofdots from the parent velocity (plus the
                    // translations of a free joint), then update the velocity
                    for k in 0..3 {
                        cdofdot[j + k] = cross_motion(cvel, v6(&d.cdof, bda + j + k));
                    }
                    let tmp = mul_dof_vec(&d.cdof, &d.qvel, bda + j, 3);
                    cvel = add6(cvel, tmp);
                    j += 2;
                }
                JointType::Hinge | JointType::Slide => {
                    // in principle the new velocity should be used, but
                    // crossMotion(cdof, cdof) = 0 and the old one is more accurate
                    cdofdot[j] = cross_motion(cvel, v6(&d.cdof, bda + j));
                    let tmp = mul_dof_vec(&d.cdof, &d.qvel, bda + j, 1);
                    cvel = add6(cvel, tmp);
                }
            }
            j += 1;
        }

        d.cvel[6 * i..6 * i + 6].copy_from_slice(&cvel);
        for (k, row) in cdofdot.iter().enumerate().take(dofnum) {
            d.cdof_dot[6 * (bda + k)..6 * (bda + k) + 6].copy_from_slice(row);
        }
    }
}

/// Port of `mj_rne` with `flg_acc = 0` or `1`: the recursive Newton-Euler pass.
///
/// With `flg_acc = false` it writes `qfrc_bias`, the gravity, Coriolis and
/// centrifugal force in joint space; with `true` it writes `M qacc + qfrc_bias`
/// (inverse dynamics) to `qfrc_bias` instead. Needs `com_vel` first.
pub fn rne<R: Real>(m: &Model<R>, d: &mut Data<R>, flg_acc: bool) {
    rne_with(m, d, flg_acc, &Faults::NONE);
}

pub(crate) fn rne_with<R: Real>(m: &Model<R>, d: &mut Data<R>, flg_acc: bool, faults: &Faults) {
    // the world accelerates up at g: cacc = -gravity
    d.cacc[0..6].fill(R::ZERO);
    let minus_one = -R::ONE;
    d.cacc[3] = m.gravity[0] * minus_one;
    d.cacc[4] = m.gravity[1] * minus_one;
    d.cacc[5] = m.gravity[2] * minus_one;

    // forward pass over bodies: accumulate cacc, set cfrc_body
    for i in 1..m.nbody {
        let bda = m.body_dofadr[i];
        let n = m.body_dofnum[i];

        // cacc = cacc_parent + cdofdot * qvel
        let tmp = mul_dof_vec(&d.cdof_dot, &d.qvel, bda, n);
        let mut cacc = add6(v6(&d.cacc, m.body_parentid[i]), tmp);

        // cacc += cdof * qacc
        if flg_acc {
            let tmp = mul_dof_vec(&d.cdof, &d.qacc, bda, n);
            cacc = add6(cacc, tmp);
        }
        d.cacc[6 * i..6 * i + 6].copy_from_slice(&cacc);

        // cfrc_body = cinert * cacc + cvel x (cinert * cvel)
        let cinert = v10(&d.cinert, i);
        let mut cfrc = mul_inert_vec(&cinert, cacc);
        let cvel = v6(&d.cvel, i);
        let tmp = mul_inert_vec(&cinert, cvel);
        let coriolis = cross_force(cvel, tmp);
        for k in 0..6 {
            if faults.flip_coriolis_sign {
                cfrc[k] -= coriolis[k];
            } else {
                cfrc[k] += coriolis[k];
            }
        }
        d.cfrc_body[6 * i..6 * i + 6].copy_from_slice(&cfrc);
    }

    // clear the world's cfrc_body
    d.cfrc_body[0..6].fill(R::ZERO);

    // backward pass over bodies: accumulate cfrc_body from the children
    for i in (1..m.nbody).rev() {
        let j = m.body_parentid[i];
        if j != 0 {
            let s = add6(v6(&d.cfrc_body, j), v6(&d.cfrc_body, i));
            d.cfrc_body[6 * j..6 * j + 6].copy_from_slice(&s);
        }
    }

    // result = cdof * cfrc_body
    for v in 0..m.nv {
        d.qfrc_bias[v] = dot6(v6(&d.cdof, v), v6(&d.cfrc_body, m.dof_bodyid[v]));
    }
}

/// Port of the spring and damper part of `mj_passive` (`mj_springdamper`):
/// joint springs toward `qpos_spring` and dof dampers, summed into
/// `qfrc_passive`.
///
/// MuJoCo's polynomial stiffness and damping terms (`jnt_stiffnesspoly`,
/// `dof_dampingpoly`) are all zero for a scene that came from `sim-scene`, so the
/// force coefficient is the plain stiffness or damping (`mju_polyForce` returns its
/// `linear` argument plus zeros), and the norm that only the polynomial would read
/// is not computed. Gravity compensation, fluid and adhesion forces, tendon and
/// flex springs are out of scope.
pub fn passive<R: Real>(m: &Model<R>, d: &mut Data<R>) {
    d.qfrc_spring.fill(R::ZERO);
    d.qfrc_damper.fill(R::ZERO);
    d.qfrc_passive.fill(R::ZERO);

    // joint-level springs
    for b in 0..m.nbody {
        for j in m.body_jntadr[b]..m.body_jntadr[b] + m.body_jntnum[b] {
            let stiffness = m.jnt_stiffness[j];
            if stiffness == R::ZERO {
                continue;
            }
            let mut padr = m.jnt_qposadr[j];
            let mut dadr = m.jnt_dofadr[j];
            let k = stiffness;
            let mut ball_like = false;
            match m.jnt_type[j] {
                JointType::Free => {
                    // translation: force along the displacement
                    let dif = sub3(
                        [d.qpos[padr], d.qpos[padr + 1], d.qpos[padr + 2]],
                        [
                            m.qpos_spring[padr],
                            m.qpos_spring[padr + 1],
                            m.qpos_spring[padr + 2],
                        ],
                    );
                    let neg_k = -k;
                    d.qfrc_spring[dadr] += dif[0] * neg_k;
                    d.qfrc_spring[dadr + 1] += dif[1] * neg_k;
                    d.qfrc_spring[dadr + 2] += dif[2] * neg_k;
                    // continue with the rotation
                    dadr += 3;
                    padr += 3;
                    ball_like = true;
                }
                JointType::Ball => ball_like = true,
                JointType::Slide | JointType::Hinge => {
                    let x = d.qpos[padr] - m.qpos_spring[padr];
                    d.qfrc_spring[dadr] = -x * k;
                }
            }
            if ball_like {
                // the quaternion difference as an angular "velocity"
                let mut quat = [
                    d.qpos[padr],
                    d.qpos[padr + 1],
                    d.qpos[padr + 2],
                    d.qpos[padr + 3],
                ];
                normalize4(&mut quat);
                let spring = [
                    m.qpos_spring[padr],
                    m.qpos_spring[padr + 1],
                    m.qpos_spring[padr + 2],
                    m.qpos_spring[padr + 3],
                ];
                let dif = sub_quat(quat, spring);
                let neg_k = -k;
                d.qfrc_spring[dadr] += dif[0] * neg_k;
                d.qfrc_spring[dadr + 1] += dif[1] * neg_k;
                d.qfrc_spring[dadr + 2] += dif[2] * neg_k;
            }
        }
    }

    // dof-level dampers
    for i in 0..m.nv {
        let damping = m.dof_damping[i];
        if damping != R::ZERO {
            d.qfrc_damper[i] = -d.qvel[i] * damping;
        }
    }

    // qfrc_passive = qfrc_spring + qfrc_damper
    for i in 0..m.nv {
        d.qfrc_passive[i] = d.qfrc_spring[i] + d.qfrc_damper[i];
    }
}

/// MuJoCo's `mju_isBad`: not a number, or above `mjMAXVAL` in magnitude.
fn is_bad<R: Real>(x: R) -> bool {
    let max = lit::<R>(1e10);
    // written so that a NaN fails both comparisons (x != x, without the lint)
    !(x >= -max && x <= max)
}

/// Port of `mj_fwdActuation` for motors and position servos: clamps `ctrl` to
/// the range when the actuator has one (MuJoCo's `ctrllimited`, which a range
/// switches on under `autolimits`), computes the actuator force, and maps it to
/// joint space with the gear (`qfrc_actuator`).
///
/// Motor: `qfrc = gear * ctrl`. Position servo (kv 0, `ctrl` the target):
/// `force = kp * ctrl - kp * length` with `length = gear * q` (`mj_transmission`
/// for a hinge or slide), and `qfrc = gear * force`. As MuJoCo does, if any clamped control is not a
/// number or exceeds 1e10 in magnitude, every control is read as zero.
pub fn actuation<R: Real>(m: &Model<R>, d: &mut Data<R>) {
    d.qfrc_actuator.fill(R::ZERO);
    if m.nu == 0 {
        return;
    }

    let clamp = |a: usize, x: R| -> R {
        if m.actuator_ctrllimited[a] {
            let (lo, hi) = (m.actuator_ctrlrange[2 * a], m.actuator_ctrlrange[2 * a + 1]);
            if x < lo {
                lo
            } else if x > hi {
                hi
            } else {
                x
            }
        } else {
            x
        }
    };
    let any_bad = (0..m.nu).any(|a| is_bad(clamp(a, d.ctrl[a])));

    for a in 0..m.nu {
        let ctrl = if any_bad {
            R::ZERO
        } else {
            clamp(a, d.ctrl[a])
        };
        let force = match m.actuator_type[a] {
            // gain 1, no bias
            ActuatorType::Motor => R::ONE * ctrl,
            // gain kp, bias -kp * length, length = gear * q (mj_transmission)
            ActuatorType::Position => {
                let kp = m.actuator_kp[a];
                let length = m.actuator_gear[a] * d.qpos[m.actuator_qposadr[a]];
                kp * ctrl - kp * length
            }
        };
        let dof = m.actuator_dofadr[a];
        d.qfrc_actuator[dof] += m.actuator_gear[a] * force;
    }
}

/// Port of `mj_fwdAcceleration`: `qfrc_smooth = qfrc_passive - qfrc_bias +
/// qfrc_actuator` and `qacc_smooth = M^-1 qfrc_smooth` with the factors in `qld`.
pub(crate) fn acceleration<R: Real>(m: &Model<R>, d: &mut Data<R>) {
    for i in 0..m.nv {
        d.qfrc_smooth[i] = (d.qfrc_passive[i] - d.qfrc_bias[i]) + d.qfrc_actuator[i];
    }
    d.qacc_smooth.copy_from_slice(&d.qfrc_smooth);
    factor::solve(
        &mut d.qacc_smooth,
        &d.qld,
        &d.qld_diag_inv,
        m.nv,
        &m.qm_sparsity,
    );
}

/// Port of `mj_tendon` for fixed tendons: `ten_length = sum coef * qpos[adr]` (on
/// `qpos` itself, not on the displacement from `qpos0`) and the dense Jacobian
/// `ten_j` (`coef` in the column of each joint's dof). A joint listed twice in one
/// tendon would add its coefficients; `Model::compile` refuses such a tendon.
pub fn tendon<R: Real>(m: &Model<R>, d: &mut Data<R>) {
    let nv = m.nv;
    d.ten_length.fill(R::ZERO);
    d.ten_j.fill(R::ZERO);
    for i in 0..m.ntendon {
        let adr = m.tendon_adr[i];
        for j in 0..m.tendon_num[i] {
            let k = m.wrap_objid[adr + j];
            let coef = m.wrap_prm[adr + j];
            // add to the length
            d.ten_length[i] += coef * d.qpos[m.jnt_qposadr[k]];
            // add to the Jacobian
            d.ten_j[i * nv + m.jnt_dofadr[k]] += coef;
        }
    }
}

/// The tendon velocity of `mj_fwdVelocity`: `ten_velocity = ten_J qvel`, summed in the
/// order of MuJoCo's sparse product over the sorted dofs of each tendon's row.
pub fn tendon_velocity<R: Real>(m: &Model<R>, d: &mut Data<R>) {
    let nv = m.nv;
    for i in 0..m.ntendon {
        let (adr, nnz) = (m.ten_j_rowadr[i], m.ten_j_rownnz[i]);
        let ind = &m.ten_j_colind[adr..adr + nnz];
        // mju_dotSparse: four interleaved partial sums over blocks of four, then the
        // leftovers one at a time, the Jacobian values read in place
        let (mut r0, mut r1, mut r2, mut r3) = (R::ZERO, R::ZERO, R::ZERO, R::ZERO);
        let mut k = 0;
        while k + 4 <= nnz {
            r0 += d.ten_j[i * nv + ind[k]] * d.qvel[ind[k]];
            r1 += d.ten_j[i * nv + ind[k + 1]] * d.qvel[ind[k + 1]];
            r2 += d.ten_j[i * nv + ind[k + 2]] * d.qvel[ind[k + 2]];
            r3 += d.ten_j[i * nv + ind[k + 3]] * d.qvel[ind[k + 3]];
            k += 4;
        }
        let mut res = (r0 + r2) + (r1 + r3);
        while k < nnz {
            res += d.ten_j[i * nv + ind[k]] * d.qvel[ind[k]];
            k += 1;
        }
        d.ten_velocity[i] = res;
    }
}

/// Port of `mj_forward` (the position, velocity, acceleration and constraint
/// stages): from `qpos`, `qvel`, `ctrl` and `qacc_warmstart` to `qacc` and every
/// intermediate of [`Data`] (the constraint rows, `qacc_smooth`, `qfrc_constraint`,
/// `solver_niter`).
pub fn forward<R: Real>(m: &Model<R>, d: &mut Data<R>) {
    forward_with(m, d, &Faults::NONE);
}

pub(crate) fn forward_with<R: Real>(m: &Model<R>, d: &mut Data<R>, faults: &Faults) {
    // position stage (mj_fwdPosition): kinematics with the tendon lengths, the inertia
    // matrix, and the constraint rows with their impedance
    kinematics(m, d);
    com_pos(m, d);
    tendon(m, d);
    crb_with(m, d, faults);
    factor_m(m, d);
    crate::collision::collision(m, d, faults);
    constraint::make_constraint(m, d, faults);
    // velocity stage (mj_fwdVelocity): the tendon velocity, the constraint
    // references (efc_vel and efc_aref), the bias force
    tendon_velocity(m, d);
    com_vel(m, d);
    passive(m, d);
    constraint::reference_constraint(m, d, faults);
    rne_with(m, d, false, faults);
    // acceleration stage: the actuators, qacc_smooth, then the constraint solve
    actuation(m, d);
    acceleration(m, d);
    constraint::fwd_constraint(m, d, faults);
}

/// Port of `mj_objectVelocity` for `mjOBJ_XBODY` with `flg_local = 0`: the
/// velocity of body `body`'s frame origin as `[angular(3), linear(3)]` in the
/// world orientation. The angular part is the body's angular velocity; the linear
/// part is the velocity of the point that is the body frame's origin. A body
/// that is welded to the world has zero velocity.
///
/// Needs `kinematics`, `com_pos` and `com_vel` for the current state.
pub fn body_velocity<R: Real>(m: &Model<R>, d: &Data<R>, body: usize) -> [R; 6] {
    // dof-less body (static): quick return
    if m.body_dofnum[m.body_weldid[body]] == 0 {
        return [R::ZERO; 6];
    }
    crate::math::transform_motion(
        v6(&d.cvel, body),
        v3(&d.xpos, body),
        v3(&d.subtree_com, m.body_rootid[body]),
    )
}
