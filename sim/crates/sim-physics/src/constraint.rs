//! MuJoCo's soft constraints for joint limits, tendon limits, dry friction loss of joints
//! and fixed tendons, and contacts: instantiation, impedance, reference acceleration, and
//! the constraint cost and force.
//!
//! Ports, from MuJoCo 3.14.0 `engine_core_constraint.c`: `mj_makeConstraint`,
//! `mj_addConstraint`, `mj_instantiateFriction`, `mj_instantiateLimit`,
//! `mj_instantiateContact` with `mj_contactJacobian` (one body per side), `mj_diagApprox`,
//! `mj_makeImpedance` with `getsolparam`, `getposdim` and `getimpedance`,
//! `mj_referenceConstraint`, `mj_constraintUpdate_impl` and `mj_constraintUpdate`; from
//! `engine_forward.c`: `warmstart` and `fwdConstraint`; from `engine_util_blas.c`:
//! `mju_mulMatMat` and `mju_addScl` (as the contact rows use them).
//!
//! Invariants:
//! - **Rows in MuJoCo's order**: equality (none in this phase, so `ne = 0`), then
//!   friction loss (dofs in order, then tendons), then limits (joints in order,
//!   lower side before upper, then tendons), then contacts in contact order (an excluded
//!   contact, one in the gap, has no rows). `efc_type` and `efc_id` identify a row exactly
//!   as MuJoCo does (`mjCNSTR_*` codes 1 to 7; a contact row's id is the contact's index).
//! - **Dense `efc_J`**, `nefc x nv` row-major, preallocated for [`Model::nefc_max`]
//!   rows. A friction-loss or limit row whose Jacobian is entirely zero is not
//!   instantiated (MuJoCo's guard in `mj_addConstraint`); a CONTACT row is never dropped.
//! - **Contact rows** (`mj_instantiateContact`): the Jacobian difference of the two bodies'
//!   points at the contact position is rotated into the contact frame
//!   (`mju_mulMatMat`, skipping zero entries): row 0 the normal, rows 1 and 2 the
//!   tangents, and for `dim > 3` the rotation rows (spin about the normal, then rolling)
//!   from the rotational difference. A frictionless contact (`dim 1`) is one row. A
//!   pyramidal contact is `2 (dim - 1)` rows `J0 +- friction[k-1] Jk`, each with `pos = dist`
//!   and `margin = includemargin`. An elliptic contact is `dim` rows with `pos` and `margin`
//!   on the normal row only. Rows of one contact are consecutive; a contact row increments
//!   `nefc` only (not `nf` or `nl`) and has no friction loss.
//! - **A limit row is active when `dist < margin`**, `dist` being the distance to
//!   the limit (negative past it): a hinge or slide joint on `qpos` itself, a ball
//!   joint on the angle of its quaternion against `max(range[0], range[1])`, a tendon
//!   on its length. The row's Jacobian is `-side` on the dof (hinge, slide), `-axis`
//!   of the rotation vector (ball), `-side` times the tendon Jacobian.
//! - **Impedance and reference**: `R = max(1e-15, (1 - I) A / I)` with `A` the
//!   approximate inverse inertia (`dof_invweight0`, `tendon_invweight0`; for a contact the
//!   `body_invweight0` of the two bodies, translation and rotation), `D = 1 / R`,
//!   `K = 1 / (d_max^2 tc^2 zeta^2)` and `B = 2 / (d_max tc)` for the standard
//!   `solref = (tc, zeta)`, `K = -solref[0] / d_max^2` and `B = -solref[1] / d_max`
//!   for the direct format (both not positive), `K = 0` for friction loss and for the
//!   friction rows of an elliptic contact (which use `solreffriction` instead of `solref`
//!   when it is nonzero); `tc` is raised to `2 * timestep` unless `refsafe` is disabled; the
//!   impedance `I` runs from `d_min` to `d_max` over `width` with the midpoint and power of
//!   `solimp`; `aref = -B v - K I (pos - margin)`. A mixed `solref` is replaced by the
//!   default (MuJoCo also warns; there is no warning channel here).
//! - **Impratio** (`mj_makeImpedance`): for a frictional contact, `R[i+1] = R[i] / impratio`
//!   (the first friction row), `mu = friction[0] * sqrt(R[i+1] / R[i])`; an elliptic
//!   contact's other friction rows get `R[i+j+1] = R[i+1] f0^2 / fj^2`; a pyramidal
//!   contact's rows all get `Rpy = 2 mu^2 R[i]`. This pass runs BEFORE `D = 1 / R` and the
//!   `diagApprox` readjustment. A pyramidal contact's normal stiffness therefore depends on
//!   `friction[0]` and on impratio; an elliptic one's does not.
//! - **Cost and force** (`mj_constraintUpdate_impl`): a limit or non-elliptic contact row is
//!   satisfied (no force) when `J qacc - aref >= 0`, else quadratic with force
//!   `-D (J qacc - aref)`; a friction-loss row is quadratic while `|J qacc - aref| < R f`,
//!   else linear with the constant force `-+f` (zones `LinearNeg`, `LinearPos`); an
//!   elliptic contact is in the top zone (no force), the bottom zone (quadratic in every row)
//!   or the middle zone, where the force is `-Dm (N - mu T) mu` on the normal and the
//!   friction rows share `T` (zone `Cone`, and the cone Hessian is written to `contact_h`
//!   when asked, for the Newton solver). The state of an elliptic contact is replicated over
//!   its `dim` rows.
//! - No allocation: every function writes the arrays of [`Data`]; the Newton and CG
//!   solvers are in `solver.rs`.
//! - Not ported (they need equality constraints, flexes, the discrete integrator, islands
//!   or adhesion): `mj_instantiateEquality`, `mj_projectConstraint`, `mj_Jdotv`,
//!   `mj_addSurfaceVel` and `mj_adhesionRef` (both are no-ops here: surface velocity and
//!   adhesion are refused at import), the sparse Jacobian, the flex contact Jacobians and the
//!   effective metric of `integrator="discrete"`.

use sim_scene::Cone;

use crate::data::Data;
use crate::jac::{DifScratch, jac_dif_pair};
use crate::linalg::{dot, max, min, mul_mat_t_vec, mul_mat_vec, mul_sym_vec_sparse, norm};
use crate::math::{lit, min_val, normalize3, normalize4, quat_to_vel, v3};
use crate::model::{JointType, Model};
use crate::real::Real;
use crate::smooth::Faults;

/// What kind of constraint a row is (MuJoCo's `mjtConstraint`; equality is 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintType {
    /// Dry friction of a dof (`mjCNSTR_FRICTION_DOF`).
    FrictionDof = 1,
    /// Dry friction of a tendon (`mjCNSTR_FRICTION_TENDON`).
    FrictionTendon = 2,
    /// A joint limit (`mjCNSTR_LIMIT_JOINT`).
    LimitJoint = 3,
    /// A tendon length limit (`mjCNSTR_LIMIT_TENDON`).
    LimitTendon = 4,
    /// A frictionless contact, one row (`mjCNSTR_CONTACT_FRICTIONLESS`).
    ContactFrictionless = 5,
    /// A pyramidal frictional contact, `2 (dim - 1)` rows (`mjCNSTR_CONTACT_PYRAMIDAL`).
    ContactPyramidal = 6,
    /// An elliptic frictional contact, `dim` rows (`mjCNSTR_CONTACT_ELLIPTIC`).
    ContactElliptic = 7,
}

impl ConstraintType {
    /// MuJoCo's integer code (`mjtConstraint`).
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// Whether the row belongs to a contact.
    pub const fn is_contact(self) -> bool {
        matches!(
            self,
            ConstraintType::ContactFrictionless
                | ConstraintType::ContactPyramidal
                | ConstraintType::ContactElliptic
        )
    }
}

/// The zone of a row's cost the solver ended in (MuJoCo's `mjtConstraintState`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintState {
    /// Satisfied, zero cost and force (a limit row, or a contact in the top zone).
    Satisfied = 0,
    /// Quadratic cost (friction loss inside its bound, an active limit or contact).
    Quadratic = 1,
    /// Linear cost on the negative side (friction loss: the force is `+f`).
    LinearNeg = 2,
    /// Linear cost on the positive side (friction loss: the force is `-f`).
    LinearPos = 3,
    /// The middle zone of an elliptic cone (every row of the contact has it).
    Cone = 4,
}

impl ConstraintState {
    /// MuJoCo's integer code (`mjtConstraintState`).
    pub const fn code(self) -> i32 {
        self as i32
    }
}

/// MuJoCo's `mjMINIMP`.
const MIN_IMP: f64 = 0.0001;
/// MuJoCo's `mjMAXIMP`.
const MAX_IMP: f64 = 0.9999;

/// Port of the tail of `mj_addConstraint`: records row `d.nefc` (whose Jacobian the
/// caller already wrote to `efc_j`) unless it is a friction-loss or limit row whose
/// Jacobian is entirely zero, and counts it. A contact row is never dropped. Returns
/// whether the row was added.
fn commit_row<R: Real>(
    m: &Model<R>,
    d: &mut Data<R>,
    pos: R,
    margin: R,
    frictionloss: R,
    ty: ConstraintType,
    id: usize,
) -> bool {
    let nv = m.nv;
    let row = d.nefc;
    // dense: make sure the Jacobian is not empty (non-contact rows)
    if !ty.is_contact()
        && !d.efc_j[row * nv..(row + 1) * nv]
            .iter()
            .any(|&x| x != R::ZERO)
    {
        return false;
    }
    d.efc_pos[row] = pos;
    d.efc_margin[row] = margin;
    d.efc_frictionloss[row] = frictionloss;
    d.efc_type[row] = ty;
    d.efc_id[row] = id;
    d.nefc += 1;
    match ty {
        ConstraintType::FrictionDof | ConstraintType::FrictionTendon => d.nf += 1,
        ConstraintType::LimitJoint | ConstraintType::LimitTendon => d.nl += 1,
        ConstraintType::ContactFrictionless
        | ConstraintType::ContactPyramidal
        | ConstraintType::ContactElliptic => {}
    }
    true
}
/// Port of `mj_instantiateFriction`: one row per dof with friction loss, then one
/// per tendon with friction loss.
fn instantiate_friction<R: Real>(m: &Model<R>, d: &mut Data<R>) {
    let nv = m.nv;
    // find frictional dofs
    for i in 0..nv {
        if m.dof_frictionloss[i] == R::ZERO {
            continue;
        }
        let row = d.nefc;
        d.efc_j[row * nv..(row + 1) * nv].fill(R::ZERO);
        d.efc_j[row * nv + i] = R::ONE;
        commit_row(
            m,
            d,
            R::ZERO,
            R::ZERO,
            m.dof_frictionloss[i],
            ConstraintType::FrictionDof,
            i,
        );
    }
    // find frictional tendons
    for i in 0..m.ntendon {
        if m.tendon_frictionloss[i] > R::ZERO {
            let row = d.nefc;
            for k in 0..nv {
                d.efc_j[row * nv + k] = d.ten_j[i * nv + k];
            }
            commit_row(
                m,
                d,
                R::ZERO,
                R::ZERO,
                m.tendon_frictionloss[i],
                ConstraintType::FrictionTendon,
                i,
            );
        }
    }
}

/// Port of `mj_instantiateLimit`: the joint limits (hinge and slide: lower and upper
/// side; ball: the one rotation limit), then the tendon limits.
fn instantiate_limit<R: Real>(m: &Model<R>, d: &mut Data<R>) {
    let nv = m.nv;
    let one = R::ONE;

    // find joint limits
    for i in 0..m.njnt {
        if !m.jnt_limited[i] {
            continue;
        }
        let margin = m.jnt_margin[i];
        match m.jnt_type[i] {
            JointType::Slide | JointType::Hinge => {
                // joint value
                let value = d.qpos[m.jnt_qposadr[i]];
                // process the lower (side -1) and upper (side +1) limits
                for side in [-one, one] {
                    // distance (negative: penetration)
                    let bound = if side < R::ZERO {
                        m.jnt_range[2 * i]
                    } else {
                        m.jnt_range[2 * i + 1]
                    };
                    let dist = side * (bound - value);
                    if dist < margin {
                        let row = d.nefc;
                        d.efc_j[row * nv..(row + 1) * nv].fill(R::ZERO);
                        d.efc_j[row * nv + m.jnt_dofadr[i]] = -side;
                        commit_row(m, d, dist, margin, R::ZERO, ConstraintType::LimitJoint, i);
                    }
                }
            }
            JointType::Ball => {
                // convert the joint quaternion to axis-angle
                let adr = m.jnt_qposadr[i];
                let mut quat = [
                    d.qpos[adr],
                    d.qpos[adr + 1],
                    d.qpos[adr + 2],
                    d.qpos[adr + 3],
                ];
                normalize4(&mut quat);
                let mut angle_axis = quat_to_vel(quat, one);

                // rotation angle, axis normalised
                let value = normalize3(&mut angle_axis);

                // distance, using the max of the range (negative: penetration)
                let dist = max(m.jnt_range[2 * i], m.jnt_range[2 * i + 1]) - value;
                if dist < margin {
                    let row = d.nefc;
                    d.efc_j[row * nv..(row + 1) * nv].fill(R::ZERO);
                    let dof = m.jnt_dofadr[i];
                    for (k, &axis) in angle_axis.iter().enumerate() {
                        d.efc_j[row * nv + dof + k] = axis * -one;
                    }
                    commit_row(m, d, dist, margin, R::ZERO, ConstraintType::LimitJoint, i);
                }
            }
            // a free joint cannot be limited (validated)
            JointType::Free => {}
        }
    }

    // find tendon limits
    for i in 0..m.ntendon {
        if !m.tendon_limited[i] {
            continue;
        }
        // value = length, margin
        let value = d.ten_length[i];
        let margin = m.tendon_margin[i];
        for side in [-one, one] {
            let bound = if side < R::ZERO {
                m.tendon_range[2 * i]
            } else {
                m.tendon_range[2 * i + 1]
            };
            let dist = side * (bound - value);
            if dist < margin {
                let row = d.nefc;
                let scl = -side;
                for k in 0..nv {
                    d.efc_j[row * nv + k] = d.ten_j[i * nv + k] * scl;
                }
                commit_row(m, d, dist, margin, R::ZERO, ConstraintType::LimitTendon, i);
            }
        }
    }
}

/// Port of `mju_mulMatMat`: `res = mat1 * mat2` for the `r1 x c1` row-major `mat1` and the
/// `c1 x c2` row-major `mat2`, exploiting the sparsity of `mat1` (a zero entry of `mat1`
/// adds nothing, which is skipped, as in MuJoCo).
fn mul_mat_mat<R: Real>(res: &mut [R], mat1: &[R], mat2: &[R], r1: usize, c1: usize, c2: usize) {
    res[..r1 * c2].fill(R::ZERO);
    for i in 0..r1 {
        for k in 0..c1 {
            let tmp = mat1[i * c1 + k];
            if tmp != R::ZERO {
                // mju_addToScl(res + i * c2, mat2 + k * c2, tmp, c2)
                for j in 0..c2 {
                    res[i * c2 + j] += mat2[k * c2 + j] * tmp;
                }
            }
        }
    }
}

/// Port of `mj_instantiateContact` (with `mj_contactJacobian` for one body per side, dense):
/// the constraint rows of every contact that is not excluded, in contact order; sets each
/// contact's `efc_address`. See the module note for the rows. MuJoCo reuses `jacdifp` for
/// the pyramid edges; this port writes them into their own scratch (`con_edge`), which
/// changes no value.
fn instantiate_contact<R: Real>(m: &Model<R>, d: &mut Data<R>) {
    let nv = m.nv;
    let ncon = d.ncon;
    if m.disable.contact || ncon == 0 || nv == 0 {
        return;
    }
    let ispyramid = m.opt.cone == Cone::Pyramidal;

    for i in 0..ncon {
        if d.contact_exclude[i] != 0 {
            continue;
        }
        let dim = d.contact_dim[i];
        d.contact_efc_address[i] = d.nefc as i32;

        // the Jacobian difference of the two bodies' points at the contact position
        let pos = v3(&d.contact_pos, i);
        let b1 = m.geom_bodyid[d.contact_geom[2 * i]];
        let b2 = m.geom_bodyid[d.contact_geom[2 * i + 1]];
        let nvj = {
            let Data {
                cdof,
                subtree_com,
                con_jac1p,
                con_jac2p,
                con_jac1r,
                con_jac2r,
                con_jacdifp,
                con_jacdifr,
                ..
            } = &mut *d;
            let mut s = DifScratch {
                jac1p: con_jac1p,
                jac2p: con_jac2p,
                jacdifp: con_jacdifp,
                jac1r: con_jac1r,
                jac2r: con_jac2r,
                jacdifr: con_jacdifr,
            };
            jac_dif_pair(m, cdof, subtree_com, b1, b2, pos, pos, dim > 3, &mut s)
        };

        // skip the contact if no dofs are affected (MuJoCo's NV == 0; the dense Jacobian
        // has nv > 0 columns here, so this does not happen)
        if nvj == 0 {
            d.contact_efc_address[i] = -1;
            d.contact_exclude[i] = 3;
            continue;
        }

        // rotate the Jacobian differences to the contact frame
        {
            let Data {
                contact_frame,
                con_jac,
                con_jacdifp,
                con_jacdifr,
                ..
            } = &mut *d;
            let frame = &contact_frame[9 * i..9 * i + 9];
            mul_mat_mat(
                con_jac,
                frame,
                con_jacdifp,
                if dim > 1 { 3 } else { 1 },
                3,
                nv,
            );
            if dim > 3 {
                mul_mat_mat(&mut con_jac[3 * nv..], frame, con_jacdifr, dim - 3, 3, nv);
            }
        }

        let dist = d.contact_dist[i];
        let includemargin = d.contact_includemargin[i];

        // a frictionless contact
        if dim == 1 {
            let row = d.nefc;
            let (efc_j, con_jac) = (&mut d.efc_j, &d.con_jac);
            efc_j[row * nv..(row + 1) * nv].copy_from_slice(&con_jac[..nv]);
            commit_row(
                m,
                d,
                dist,
                includemargin,
                R::ZERO,
                ConstraintType::ContactFrictionless,
                i,
            );
        }
        // a pyramidal friction cone
        else if ispyramid {
            // one pair of rows per friction dimension
            for k in 1..dim {
                let fk = d.contact_friction[5 * i + k - 1];
                {
                    // the Jacobian of the pair of opposing pyramid edges
                    let Data {
                        con_jac, con_edge, ..
                    } = &mut *d;
                    for c in 0..nv {
                        con_edge[c] = con_jac[c] + con_jac[k * nv + c] * fk;
                    }
                    let neg = -fk;
                    for c in 0..nv {
                        con_edge[nv + c] = con_jac[c] + con_jac[k * nv + c] * neg;
                    }
                }
                for e in 0..2 {
                    let row = d.nefc;
                    let (efc_j, con_edge) = (&mut d.efc_j, &d.con_edge);
                    efc_j[row * nv..(row + 1) * nv]
                        .copy_from_slice(&con_edge[e * nv..(e + 1) * nv]);
                    // pos = dist, margin = includemargin, on both rows
                    commit_row(
                        m,
                        d,
                        dist,
                        includemargin,
                        R::ZERO,
                        ConstraintType::ContactPyramidal,
                        i,
                    );
                }
            }
        }
        // an elliptic friction cone
        else {
            for r in 0..dim {
                let row = d.nefc;
                let (efc_j, con_jac) = (&mut d.efc_j, &d.con_jac);
                efc_j[row * nv..(row + 1) * nv].copy_from_slice(&con_jac[r * nv..(r + 1) * nv]);
                // the normal pos = dist and margin = includemargin, all others 0
                let (pos_r, margin_r) = if r == 0 {
                    (dist, includemargin)
                } else {
                    (R::ZERO, R::ZERO)
                };
                commit_row(
                    m,
                    d,
                    pos_r,
                    margin_r,
                    R::ZERO,
                    ConstraintType::ContactElliptic,
                    i,
                );
            }
        }
    }
}

/// Port of `mj_diagApprox`: the approximate diagonal of the constraint-space inverse
/// inertia, from `invweight0` at `qpos0`. A contact row takes the average translation and
/// rotation inverse weights of its two bodies: a frictionless row the translation, an
/// elliptic contact's rows `tran` (the first three) and `rot` (the others), a pyramidal
/// contact's pair for friction `j` the value `tran + f_j^2 (tran or rot)` (`tran` for the
/// two tangents, `rot` for the spin and rolling). (`faults.diag_approx_one` replaces every row
/// by 1, a test-only fault.)
fn diag_approx<R: Real>(m: &Model<R>, d: &mut Data<R>, faults: &Faults) {
    let nefc = d.nefc;
    let one = R::ONE;
    let mut i = 0usize;
    while i < nefc {
        let id = d.efc_id[i];
        if faults.diag_approx_one {
            d.efc_diag_approx[i] = one;
            i += 1;
            continue;
        }
        match d.efc_type[i] {
            ConstraintType::FrictionDof => d.efc_diag_approx[i] = m.dof_invweight0[id],
            ConstraintType::LimitJoint => {
                d.efc_diag_approx[i] = m.dof_invweight0[m.jnt_dofadr[id]];
            }
            ConstraintType::FrictionTendon | ConstraintType::LimitTendon => {
                d.efc_diag_approx[i] = m.tendon_invweight0[id];
            }
            ConstraintType::ContactFrictionless
            | ConstraintType::ContactPyramidal
            | ConstraintType::ContactElliptic => {
                let dim = d.contact_dim[id];

                // add the average translation and rotation components from both sides
                let mut tran = R::ZERO;
                let mut rot = R::ZERO;
                for side in 0..2 {
                    let bid = m.geom_bodyid[d.contact_geom[2 * id + side]];
                    let weight = one;
                    tran += m.body_invweight0[2 * bid] * weight;
                    rot += m.body_invweight0[2 * bid + 1] * weight;
                }

                match d.efc_type[i] {
                    // frictionless
                    ConstraintType::ContactFrictionless => d.efc_diag_approx[i] = tran,
                    // elliptical
                    ConstraintType::ContactElliptic => {
                        for j in 0..dim {
                            d.efc_diag_approx[i + j] = if j < 3 { tran } else { rot };
                        }
                        // processed dim elements in one iteration; advance the counter
                        i += dim - 1;
                    }
                    // pyramidal
                    _ => {
                        for j in 0..dim - 1 {
                            let fri = d.contact_friction[5 * id + j];
                            let v = tran + fri * fri * (if j < 2 { tran } else { rot });
                            d.efc_diag_approx[i + 2 * j] = v;
                            d.efc_diag_approx[i + 2 * j + 1] = v;
                        }
                        // processed 2 * dim - 2 elements in one iteration; advance the counter
                        i += 2 * dim - 3;
                    }
                }
            }
        }
        i += 1;
    }
}

/// MuJoCo's `power(a, b)`: quick returns for the exponents 1 and 2 (the default).
#[inline]
fn power<R: Real>(a: R, b: R) -> R {
    if b == R::ONE {
        a
    } else if b == lit::<R>(2.0) {
        a * a
    } else {
        a.powf(b)
    }
}

/// Port of `getimpedance`: the impedance `imp` of a row at `pos` (past `margin`) and
/// its derivative `imp'` with respect to `pos`.
fn get_impedance<R: Real>(solimp: &[R; 5], pos: R, margin: R) -> (R, R) {
    let half = lit::<R>(0.5);
    // flat function
    if solimp[0] == solimp[1] || solimp[2] <= min_val::<R>() {
        return (half * (solimp[0] + solimp[1]), R::ZERO);
    }

    // x = abs((pos - margin) / width)
    let mut x = (pos - margin) / solimp[2];
    let mut sgn = R::ONE;
    if x < R::ZERO {
        x = -x;
        sgn = -R::ONE;
    }

    // fully saturated
    if x >= R::ONE || x <= R::ZERO {
        let imp = if x >= R::ONE { solimp[1] } else { solimp[0] };
        return (imp, R::ZERO);
    }

    // linear
    let (y, y_p);
    if solimp[4] == R::ONE {
        y = x;
        y_p = R::ONE;
    }
    // y(x) = a x^p if x <= midpoint
    else if x <= solimp[3] {
        let a = R::ONE / power(solimp[3], solimp[4] - R::ONE);
        y = a * power(x, solimp[4]);
        y_p = solimp[4] * a * power(x, solimp[4] - R::ONE);
    }
    // y(x) = 1 - b (1 - x)^p if x > midpoint
    else {
        let b = R::ONE / power(R::ONE - solimp[3], solimp[4] - R::ONE);
        y = R::ONE - b * power(R::ONE - x, solimp[4]);
        y_p = solimp[4] * b * power(R::ONE - x, solimp[4] - R::ONE);
    }

    // scale
    let imp = solimp[0] + y * (solimp[1] - solimp[0]);
    let imp_p = y_p * sgn * (solimp[1] - solimp[0]) / solimp[2];
    (imp, imp_p)
}

/// Port of `getsolparam`: `solref`, `solreffriction` and `solimp` of row `i` with MuJoCo's
/// repairs (a mixed `solref` becomes the default, `solref[0]` is raised to `2 * timestep`
/// for the standard format unless `refsafe` is disabled, a mixed `solreffriction` becomes
/// zero and gets the same raise, `solimp` is clamped). `solreffriction` applies to contacts
/// only and is zero for every other row.
fn get_solparam<R: Real>(m: &Model<R>, d: &Data<R>, i: usize) -> ([R; 2], [R; 2], [R; 5]) {
    let id = d.efc_id[i];
    let mut solreffriction = [R::ZERO; 2];
    let (sr, si): (&[R], &[R]) = match d.efc_type[i] {
        ConstraintType::LimitJoint => (
            &m.jnt_solref[2 * id..2 * id + 2],
            &m.jnt_solimp[5 * id..5 * id + 5],
        ),
        ConstraintType::FrictionDof => (
            &m.dof_solref[2 * id..2 * id + 2],
            &m.dof_solimp[5 * id..5 * id + 5],
        ),
        ConstraintType::LimitTendon => (
            &m.tendon_solref_lim[2 * id..2 * id + 2],
            &m.tendon_solimp_lim[5 * id..5 * id + 5],
        ),
        ConstraintType::FrictionTendon => (
            &m.tendon_solref_fri[2 * id..2 * id + 2],
            &m.tendon_solimp_fri[5 * id..5 * id + 5],
        ),
        ConstraintType::ContactFrictionless
        | ConstraintType::ContactPyramidal
        | ConstraintType::ContactElliptic => {
            solreffriction.copy_from_slice(&d.contact_solreffriction[2 * id..2 * id + 2]);
            (
                &d.contact_solref[2 * id..2 * id + 2],
                &d.contact_solimp[5 * id..5 * id + 5],
            )
        }
    };
    let mut solref = [sr[0], sr[1]];
    let mut solimp = [si[0], si[1], si[2], si[3], si[4]];

    // check the reference format: standard or direct, cannot be mixed
    if (solref[0] > R::ZERO) != (solref[1] > R::ZERO) {
        // mj_defaultSolRefImp
        solref = [lit::<R>(0.02), R::ONE];
    }

    // integrator safety: impose ref[0] >= 2 * timestep for the standard format
    if !m.disable.refsafe && solref[0] > R::ZERO {
        solref[0] = max(solref[0], lit::<R>(2.0) * m.timestep);
    }

    // check the reference format of solreffriction: the same sign, else its default (0, 0)
    if (solreffriction[0] > R::ZERO) != (solreffriction[1] > R::ZERO) {
        solreffriction = [R::ZERO; 2];
    }

    // integrator safety for solreffriction
    if !m.disable.refsafe && solreffriction[0] > R::ZERO {
        solreffriction[0] = max(solreffriction[0], lit::<R>(2.0) * m.timestep);
    }

    // enforce the constraints on solimp
    let (lo, hi) = (lit::<R>(MIN_IMP), lit::<R>(MAX_IMP));
    solimp[0] = min(hi, max(lo, solimp[0]));
    solimp[1] = min(hi, max(lo, solimp[1]));
    solimp[2] = max(R::ZERO, solimp[2]);
    solimp[3] = min(hi, max(lo, solimp[3]));
    solimp[4] = max(R::ONE, solimp[4]);
    (solref, solreffriction, solimp)
}

/// Port of `getposdim`: the position and the number of rows of the constraint that starts
/// at row `i` (1 except for contacts: `dim` for an elliptic one, `2 (dim - 1)` for a
/// pyramidal one).
fn get_pos_dim<R: Real>(d: &Data<R>, i: usize) -> (R, usize) {
    let id = d.efc_id[i];
    let pos = d.efc_pos[i];
    match d.efc_type[i] {
        ConstraintType::ContactElliptic => (pos, d.contact_dim[id]),
        ConstraintType::ContactPyramidal => (pos, 2 * (d.contact_dim[id] - 1)),
        _ => (pos, 1),
    }
}

/// Port of `mj_makeImpedance`: `efc_r`, `efc_kbip`, `efc_d`, and `efc_diag_approx` adjusted
/// so that `R = (1 - I) A / I`. A contact is processed as one constraint of `dim` rows (the
/// impedance comes from the first row's `pos`); then the frictional contacts' `R` is adjusted
/// (impratio, `Rpy`) and their `mu` set, BEFORE `D = 1 / R` (see the module note).
/// `faults.ignore_impratio` reads impratio as 1, a test-only fault.
fn make_impedance<R: Real>(m: &Model<R>, d: &mut Data<R>, faults: &Faults) {
    let min_val = min_val::<R>();
    let nefc = d.nefc;
    let mut i = 0usize;
    while i < nefc {
        // solref, solreffriction and solimp
        let (solref, solreffriction, solimp) = get_solparam(m, d, i);

        // pos and dim
        let (pos, dim) = get_pos_dim(d, i);

        // imp and impP
        let (imp, imp_p) = get_impedance(&solimp, pos, d.efc_margin[i]);

        // R and KBIP for all the constraint dimensions
        for j in 0..dim {
            // R = (1 - imp) / imp * diagApprox
            d.efc_r[i + j] = max(min_val, (R::ONE - imp) * d.efc_diag_approx[i + j] / imp);

            let tp = d.efc_type[i + j];

            // an elliptic contact uses solreffriction in its non-normal directions, if
            // non-zero
            let elliptic_friction = tp == ConstraintType::ContactElliptic && j > 0;
            let reference = if elliptic_friction
                && (solreffriction[0] != R::ZERO || solreffriction[1] != R::ZERO)
            {
                solreffriction
            } else {
                solref
            };

            let kbip = &mut d.efc_kbip[4 * (i + j)..4 * (i + j) + 4];

            // friction: K = 0
            if tp == ConstraintType::FrictionDof
                || tp == ConstraintType::FrictionTendon
                || elliptic_friction
            {
                kbip[0] = R::ZERO;
            }
            // standard: K = 1 / (d_width^2 timeconst^2 dampratio^2)
            else if reference[0] > R::ZERO {
                kbip[0] = R::ONE
                    / max(
                        min_val,
                        solimp[1]
                            * solimp[1]
                            * reference[0]
                            * reference[0]
                            * reference[1]
                            * reference[1],
                    );
            }
            // direct: K = -solref[0] / d_width^2
            else {
                kbip[0] = -reference[0] / max(min_val, solimp[1] * solimp[1]);
            }

            // standard: B = 2 / (d_width timeconst)
            if reference[1] > R::ZERO {
                kbip[1] = lit::<R>(2.0) / max(min_val, solimp[1] * reference[0]);
            }
            // direct: B = -solref[1] / d_width
            else {
                kbip[1] = -reference[1] / max(min_val, solimp[1]);
            }

            // I = imp, P = imp'
            kbip[2] = imp;
            kbip[3] = imp_p;
        }

        // skip the rest of this constraint
        i += dim;
    }

    // frictional contacts: adjust R in the friction dimensions, set the contact's master mu
    let impratio = if faults.ignore_impratio {
        R::ONE
    } else {
        m.opt.impratio
    };
    let mut i = d.ne + d.nf;
    while i < nefc {
        let ty = d.efc_type[i];
        if ty == ConstraintType::ContactPyramidal || ty == ConstraintType::ContactElliptic {
            // the id, dim and friction
            let id = d.efc_id[i];
            let dim = d.contact_dim[id];
            let friction = [
                d.contact_friction[5 * id],
                d.contact_friction[5 * id + 1],
                d.contact_friction[5 * id + 2],
                d.contact_friction[5 * id + 3],
                d.contact_friction[5 * id + 4],
            ];

            // R[1] = R[0] / impratio
            d.efc_r[i + 1] = d.efc_r[i] / max(min_val, impratio);

            // mu of the regularised cone = mu[1] * sqrt(R[1] / R[0])
            d.contact_mu[id] = friction[0] * (d.efc_r[i + 1] / d.efc_r[i]).sqrt();

            // elliptic
            if ty == ConstraintType::ContactElliptic {
                // the remaining R's such that R[j] * mu[j]^2 = R[1] * mu[1]^2
                for j in 1..dim - 1 {
                    d.efc_r[i + j + 1] =
                        d.efc_r[i + 1] * friction[0] * friction[0] / (friction[j] * friction[j]);
                }

                // skip the rest of this contact
                i += dim;
            }
            // pyramidal: a common R matching the friction impedance of the elliptic model
            else {
                // D0_el = 2 (dim - 1) D_py: normal match; D0_el = 2 mu^2 D_py: friction match
                let mu = d.contact_mu[id];
                let rpy = lit::<R>(2.0) * mu * mu * d.efc_r[i];

                // assign Rpy to all the pyramidal R
                for j in 0..2 * (dim - 1) {
                    d.efc_r[i + j] = rpy;
                }

                // skip the rest of this contact
                i += 2 * (dim - 1);
            }
        } else {
            i += 1;
        }
    }

    // set D = 1 / R
    for i in 0..nefc {
        d.efc_d[i] = R::ONE / d.efc_r[i];
    }

    // adjust diagApprox so that R = (1 - imp) / imp * diagApprox
    for i in 0..nefc {
        let imp = d.efc_kbip[4 * i + 2];
        d.efc_diag_approx[i] = d.efc_r[i] * imp / (R::ONE - imp);
    }
}

/// Port of `mj_makeConstraint` (the friction-loss, limit and contact parts): clears the row
/// counts, instantiates the rows in MuJoCo's order, and computes `efc_diag_approx`,
/// `efc_r`, `efc_d` and `efc_kbip`. With `mjDSBL_CONSTRAINT` set (the model's
/// [`crate::DisableFlags`]) there are no rows. Needs `kinematics`, `tendon` and `collision`
/// for the current `qpos`.
pub(crate) fn make_constraint<R: Real>(m: &Model<R>, d: &mut Data<R>, faults: &Faults) {
    // clear the sizes
    d.ne = 0;
    d.nf = 0;
    d.nl = 0;
    d.nefc = 0;

    // disabled: return
    if m.disable.constraint {
        return;
    }

    // instantiate every row: equality (none), friction loss, limits, contacts
    if !m.disable.frictionloss && !faults.drop_friction_rows {
        instantiate_friction(m, d);
    }
    if !m.disable.limit {
        instantiate_limit(m, d);
    }
    instantiate_contact(m, d);
    debug_assert!(d.nefc <= m.nefc_max);

    // no constraints: return
    if d.nefc == 0 {
        return;
    }

    // compute diagApprox, then KBIP, D, R and the adjusted diagApprox
    diag_approx(m, d, faults);
    make_impedance(m, d, faults);
}

/// Port of `mj_referenceConstraint` (the non-discrete path): `efc_vel = J qvel` and
/// `efc_aref = -B vel - K I (pos - margin)`. (`faults.flip_aref_sign` negates `aref`,
/// a test-only fault.) MuJoCo's `mj_addSurfaceVel` and `mj_adhesionRef` are no-ops here:
/// surface velocity and adhesion are refused at import.
pub(crate) fn reference_constraint<R: Real>(m: &Model<R>, d: &mut Data<R>, faults: &Faults) {
    let nefc = d.nefc;
    let nv = m.nv;
    if nefc == 0 {
        return;
    }

    // efc_vel = J qvel
    mul_mat_vec(&mut d.efc_vel, &d.efc_j, &d.qvel, nefc, nv);

    // aref = -B vel - K I (pos - margin); the explicit step has no shift
    let shift = R::ZERO;
    for i in 0..nefc {
        let kbip = &d.efc_kbip[4 * i..4 * i + 4];
        let aref = (-kbip[1]) * d.efc_vel[i]
            - kbip[0] * kbip[2] * (d.efc_pos[i] - d.efc_margin[i] + shift * d.efc_vel[i]);
        d.efc_aref[i] = if faults.flip_aref_sign { -aref } else { aref };
    }
}

/// The per-row inputs of [`constraint_update_impl`] (MuJoCo passes them as separate
/// arguments, with the contact array).
pub(crate) struct RowParams<'a, R: Real> {
    /// The number of friction-loss rows (the first `nf`; there are no equality rows).
    pub nf: usize,
    /// The number of rows.
    pub nefc: usize,
    /// `efc_D`.
    pub d: &'a [R],
    /// `efc_R`.
    pub r: &'a [R],
    /// `efc_frictionloss`.
    pub frictionloss: &'a [R],
    /// `efc_type`.
    pub etype: &'a [ConstraintType],
    /// `efc_id`.
    pub eid: &'a [usize],
    /// The contacts' `mu` (`contact_mu`).
    pub contact_mu: &'a [R],
    /// The contacts' friction, 5 per contact.
    pub contact_friction: &'a [R],
    /// The contacts' dimensions.
    pub contact_dim: &'a [usize],
}

/// Port of `mj_constraintUpdate_impl`: the force of every row, its zone, and (always
/// computed here) the constraint cost `s_hat(jar)`. `nf` friction-loss rows come first
/// (there are no equality rows). A limit or non-elliptic contact row is quadratic or
/// satisfied; an elliptic contact is mapped to the dual cone space (`U`), decomposed into
/// `N` and `T = norm(U[1..])`, and falls in the top zone (satisfied, no force), the bottom
/// zone (quadratic in all its rows) or the middle zone (the cone: the force
/// `-Dm (N - mu T) mu` on the normal and `-force[0] / T U[j] friction[j-1]` on the others,
/// state `Cone` in every row). With `cone_h` the Hessian of each cone-zone contact is
/// written to its 36 numbers of `contact_h` (the Newton solver's input).
pub(crate) fn constraint_update_impl<R: Real>(
    p: &RowParams<'_, R>,
    jar: &[R],
    state: &mut [ConstraintState],
    force: &mut [R],
    mut cone_h: Option<&mut [R]>,
) -> R {
    let (nf, nefc) = (p.nf, p.nefc);
    let (dd, rr, floss) = (p.d, p.r, p.frictionloss);
    let half = lit::<R>(0.5);
    let mut s = R::ZERO;

    // no constraints: clear the cost, return
    if nefc == 0 {
        return R::ZERO;
    }

    // compute the unconstrained efc_force
    for i in 0..nefc {
        force[i] = -dd[i] * jar[i];
    }

    // update the constraints
    let mut i = 0usize;
    while i < nefc {
        // ==== friction
        if i < nf {
            // linear negative
            if jar[i] <= -rr[i] * floss[i] {
                s += -half * rr[i] * floss[i] * floss[i] - floss[i] * jar[i];
                force[i] = floss[i];
                state[i] = ConstraintState::LinearNeg;
            }
            // linear positive
            else if jar[i] >= rr[i] * floss[i] {
                s += -half * rr[i] * floss[i] * floss[i] + floss[i] * jar[i];
                force[i] = -floss[i];
                state[i] = ConstraintState::LinearPos;
            }
            // quadratic
            else {
                s += half * dd[i] * jar[i] * jar[i];
                state[i] = ConstraintState::Quadratic;
            }
            i += 1;
            continue;
        }

        // ==== limit and contact: a non-negative constraint
        if p.etype[i] != ConstraintType::ContactElliptic {
            // the constraint is satisfied: no cost
            if jar[i] >= R::ZERO {
                force[i] = R::ZERO;
                state[i] = ConstraintState::Satisfied;
            }
            // quadratic
            else {
                s += half * dd[i] * jar[i] * jar[i];
                state[i] = ConstraintState::Quadratic;
            }
            i += 1;
        }
        // ==== a contact with an elliptic cone
        else {
            let id = p.eid[i];
            let mu = p.contact_mu[id];
            let friction = &p.contact_friction[5 * id..5 * id + 5];
            let dim = p.contact_dim[id];

            // map to the regular dual cone space
            let mut u = [R::ZERO; 6];
            u[0] = jar[i] * mu;
            for j in 1..dim {
                u[j] = jar[i + j] * friction[j - 1];
            }

            // decompose into the normal and the tangent
            let n = u[0];
            let t = norm(&u[1..dim]);

            // the top zone
            if n >= mu * t || (t <= R::ZERO && n >= R::ZERO) {
                force[i..i + dim].fill(R::ZERO);
                state[i] = ConstraintState::Satisfied;
            }
            // the bottom zone
            else if mu * n + t <= R::ZERO || (t <= R::ZERO && n < R::ZERO) {
                for j in 0..dim {
                    s += half * dd[i + j] * jar[i + j] * jar[i + j];
                }
                state[i] = ConstraintState::Quadratic;
            }
            // the middle zone
            else {
                // cost: 0.5 D0 / (mu^2 (1 + mu^2)) (N - mu T)^2
                let dm = dd[i] / (mu * mu * (R::ONE + mu * mu));
                let nmt = n - mu * t;
                s += half * dm * nmt * nmt;

                // force: - ds/djar = dU/djar * ds/dU  (dU/djar = diag(mu, friction))
                force[i] = -dm * nmt * mu;
                for j in 1..dim {
                    force[i + j] = -force[i] / t * u[j] * friction[j - 1];
                }

                // set the state
                state[i] = ConstraintState::Cone;

                // the cone Hessian
                if let Some(hall) = cone_h.as_deref_mut() {
                    let h = &mut hall[36 * id..36 * id + 36];

                    // the first row: (1, -mu / T * U)
                    let mut scl = -mu / t;
                    h[0] = R::ONE;
                    for j in 1..dim {
                        h[j] = scl * u[j];
                    }

                    // the upper block: mu N / T^3 * U U'
                    scl = mu * n / (t * t * t);
                    for k in 1..dim {
                        for j in k..dim {
                            h[k * dim + j] = scl * u[j] * u[k];
                        }
                    }

                    // add to the diagonal: (mu^2 - mu N / T) I
                    scl = mu * mu - mu * n / t;
                    for j in 1..dim {
                        h[j * (dim + 1)] += scl;
                    }

                    // pre and post multiply by diag(mu, friction), scale by Dm
                    for k in 0..dim {
                        let scl = dm * (if k == 0 { mu } else { friction[k - 1] });
                        for j in k..dim {
                            h[k * dim + j] *= scl * (if j == 0 { mu } else { friction[j - 1] });
                        }
                    }

                    // make symmetric: copy the upper triangle into the lower
                    for k in 0..dim {
                        for j in k + 1..dim {
                            h[j * dim + k] = h[k * dim + j];
                        }
                    }
                }
            }

            // replicate the state in all the cone dimensions
            for j in 1..dim {
                state[i + j] = state[i];
            }

            // advance to the end of the contact
            i += dim;
        }
    }
    s
}

/// Port of `mj_constraintUpdate`: from `jar` (the solver's `Jaref`, or `efc_b`),
/// writes `efc_state`, `efc_force` and `qfrc_constraint = J' efc_force` and returns
/// the constraint cost. `jar_is_b` selects `efc_b` instead of the workspace's `jaref`.
fn constraint_update<R: Real>(m: &Model<R>, d: &mut Data<R>, jar_is_b: bool) -> R {
    let nv = m.nv;
    let Data {
        nf,
        nefc,
        efc_d,
        efc_r,
        efc_frictionloss,
        efc_type,
        efc_id,
        contact_mu,
        contact_friction,
        contact_dim,
        efc_state,
        efc_force,
        efc_j,
        efc_b,
        qfrc_constraint,
        ws,
        ..
    } = d;
    let jar: &[R] = if jar_is_b { efc_b } else { &ws.jaref };
    let params = RowParams {
        nf: *nf,
        nefc: *nefc,
        d: efc_d,
        r: efc_r,
        frictionloss: efc_frictionloss,
        etype: efc_type,
        eid: efc_id,
        contact_mu,
        contact_friction,
        contact_dim,
    };
    let cost = constraint_update_impl(&params, jar, efc_state, efc_force, None);
    mul_mat_t_vec(qfrc_constraint, efc_j, efc_force, *nefc, nv);
    cost
}

/// Port of `warmstart`: starts the solver from the better, by cost, of
/// `qacc_warmstart` and `qacc_smooth` (`qacc` is set to it), unless
/// `mjDSBL_WARMSTART` is set, which starts from `qacc_smooth` with zero forces.
fn warmstart<R: Real>(m: &Model<R>, d: &mut Data<R>) {
    let nv = m.nv;
    let nefc = d.nefc;
    if !m.disable.warmstart {
        // start with qacc = qacc_warmstart
        d.qacc.copy_from_slice(&d.qacc_warmstart);

        // jar(qacc_warmstart) = J qacc_warmstart - aref
        mul_mat_vec(&mut d.ws.jaref, &d.efc_j, &d.qacc_warmstart, nefc, nv);
        for i in 0..nefc {
            d.ws.jaref[i] -= d.efc_aref[i];
        }

        // update the constraints, save cost(qacc_warmstart)
        let mut cost_warmstart = constraint_update(m, d, false);

        // add Gauss to cost(qacc_warmstart): da = qacc_warmstart - qacc_smooth,
        // Gauss = 0.5 da' M da (the solver's search and Mv arrays are free scratch)
        for i in 0..nv {
            d.ws.search[i] = d.qacc_warmstart[i] - d.qacc_smooth[i];
        }
        mul_sym_vec_sparse(&mut d.ws.mv, &d.qm, &d.ws.search, nv, &m.qm_sparsity);
        cost_warmstart += lit::<R>(0.5) * dot(&d.ws.search, &d.ws.mv);

        // cost(qacc_smooth)
        let cost_smooth = constraint_update(m, d, true);

        // use qacc_smooth if better
        if cost_warmstart > cost_smooth {
            d.qacc.copy_from_slice(&d.qacc_smooth);
        }
    }
    // coldstart with qacc = qacc_smooth, efc_force = 0
    else {
        d.qacc.copy_from_slice(&d.qacc_smooth);
        d.efc_force[..nefc].fill(R::ZERO);
    }
}

/// Port of `fwdConstraint` (monolithic, no islands): clears `qfrc_constraint`, sets
/// `qacc = qacc_smooth` when there are no rows, else computes `efc_b`, warm-starts
/// and runs the model's solver; leaves `qacc`, `efc_force`, `efc_state`,
/// `qfrc_constraint` and `solver_niter`. `mj_discreteGyro` (the last call of
/// `fwdConstraint`) returns at once unless the integrator is `discrete`, so it is
/// not ported: Euler and RK4 never reach its body.
pub(crate) fn fwd_constraint<R: Real>(m: &Model<R>, d: &mut Data<R>, faults: &Faults) {
    let nv = m.nv;
    let nefc = d.nefc;

    // always clear qfrc_constraint
    d.qfrc_constraint.fill(R::ZERO);

    // no constraints: copy the unconstrained acceleration, clear forces, return
    if nefc == 0 {
        d.qacc.copy_from_slice(&d.qacc_smooth);
        d.solver_niter = 0;
        return;
    }

    // efc_b = J qacc_smooth - aref
    mul_mat_vec(&mut d.efc_b, &d.efc_j, &d.qacc_smooth, nefc, nv);
    for i in 0..nefc {
        d.efc_b[i] -= d.efc_aref[i];
    }

    // warmstart the solver
    warmstart(m, d);
    d.solver_niter = 0;

    // run the solver over all constraints (monolithic)
    crate::solver::solve_primal(m, d, faults);
}
