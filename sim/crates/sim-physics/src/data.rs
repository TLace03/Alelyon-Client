//! The per-environment state and every intermediate of a step.
//!
//! Invariants:
//! - [`Data::new`] allocates everything once, from the model's sizes (the constraint
//!   rows from [`Model::nefc_max`]); no step allocates, so a step is a pure function
//!   of `(model, data)` over fixed-size arrays, which is what a GPU port needs.
//! - The state is `qpos`, `qvel`, `ctrl`, `time` and `qacc_warmstart` (the solver's
//!   starting acceleration, which MuJoCo keeps in `mjData` and a step sets to its
//!   own `qacc`). Everything else is derived scratch that a step overwrites; none of
//!   it carries over from one step to the next, so two `Data` with equal state steps
//!   to equal bits.
//! - Layouts (all row-major, flat): `xpos` `3 * nbody`, `xquat` `4 * nbody`
//!   (`[w, x, y, z]`), `xmat` and `ximat` `9 * nbody`, `xipos` `3 * nbody`,
//!   `xanchor` and `xaxis` `3 * njnt`, `subtree_com` `3 * nbody`, `cdof` and
//!   `cdof_dot` `6 * nv` (`[angular, linear]` per dof), `cinert` and `crb`
//!   `10 * nbody`, `cvel` `6 * nbody`, `qm` and `qld` `nv * nv` (dense, symmetric
//!   `M`, and its factorisation), `ten_j` `ntendon * nv`, `efc_j` `nefc_max * nv`
//!   (row `i` at `i * nv`), `efc_kbip` `4 * nefc_max`.
//! - `qm` holds the full symmetric matrix (both triangles). `qld` holds MuJoCo's
//!   `L'DL` factors of it in the lower triangle (`L` below the diagonal, `D` on it)
//!   and the strict upper triangle is not read.
//! - `energy[0]` is the potential energy, `energy[1]` the kinetic energy, valid
//!   after [`crate::energy_pos`] and [`crate::energy_vel`].
//! - The `efc_*` arrays are valid for rows `0..nefc`, in MuJoCo's order (equality
//!   rows, none here; then friction loss, `nf` rows; then limits, `nl` rows; then the
//!   rows of the contacts that are constraints, in contact order);
//!   `ne` is always 0. Past `nefc` they hold whatever an earlier step left.
//! - **Contacts** (phase 1c-ii) are struct-of-arrays sized [`Model::ncon_max`] and valid
//!   for `0..ncon`: `contact_dist`, `contact_includemargin`, `contact_mu` one number per
//!   contact, `contact_pos` 3, `contact_frame` 9 (row-major: normal, tangent 1,
//!   tangent 2), `contact_friction` 5, `contact_solref` and `contact_solreffriction` 2,
//!   `contact_solimp` 5, `contact_h` 36 (the Hessian of the elliptic cone, `dim * dim`
//!   used), `contact_dim`, `contact_geom` 2 (the geom ids, the normal pointing from the
//!   first), `contact_exclude` (0: a constraint; 1: in the gap, excluded but counted; 3: the
//!   model has no dofs) and `contact_efc_address` (the first row, or -1). `geom_xpos` is
//!   `3 * ngeom` and `geom_xmat` `9 * ngeom`. Between the two passes of the collision step
//!   the first `slot_count` entries at a candidate's `slot_offset` of the separate pre-contact
//!   arrays (`pre_dist`, `pre_pos` 3, `pre_frame` 6: normal and tangent) hold its pre-contacts,
//!   `cand_ncon` their counts, `cand_start` the prefix sum of the counts and `cand_overflow` a
//!   box-box overflow flag per candidate (the arrays are separate so that pass 2 is out of place
//!   and its units are independent).
//! - `warning_collision_overflow` counts the box-box collider calls that returned more than 8
//!   contacts (an RK4 step makes four collision steps), which exact arithmetic cannot do; every
//!   test asserts it stays 0.

use crate::constraint::{ConstraintState, ConstraintType};
use crate::model::Model;
use crate::real::Real;

/// The scratch arrays of the primal solvers, allocated once (MuJoCo's
/// `mjPrimalContext`, dense).
#[derive(Clone, Debug)]
pub struct Workspace<R: Real> {
    /// `J qacc - aref`, `nefc_max`.
    pub jaref: Vec<R>,
    /// `J search`, `nefc_max`.
    pub jv: Vec<R>,
    /// `M qacc`, `nv`.
    pub ma: Vec<R>,
    /// `M search`, `nv`.
    pub mv: Vec<R>,
    /// The gradient of the cost, `nv`.
    pub grad: Vec<R>,
    /// The preconditioned gradient (`H \ grad` for Newton, `M \ grad` for CG), `nv`.
    pub mgrad: Vec<R>,
    /// The line-search direction, `nv`.
    pub search: Vec<R>,
    /// The quadratic polynomial of each row's cost along the search direction,
    /// `3 * nefc_max`.
    pub quad: Vec<R>,
    /// The constraint states of the previous iteration, `nefc_max`.
    pub oldstate: Vec<ConstraintState>,
    /// The previous gradient (CG), `nv`.
    pub gradold: Vec<R>,
    /// The previous preconditioned gradient (CG), `nv`.
    pub mgradold: Vec<R>,
    /// `grad - gradold` (CG), `nv`.
    pub graddif: Vec<R>,
    /// `mgrad - mgradold` (CG), `nv`.
    pub mgraddif: Vec<R>,
    /// The constraint inertia `D` of the rows in their quadratic zone, else 0
    /// (Newton), `nefc_max`.
    pub d: Vec<R>,
    /// The scratch of the rank-one Cholesky updates (Newton), `nv`.
    pub cholupd: Vec<R>,
    /// The Cholesky factor of the Hessian `M + J' D J` (Newton), `nv * nv`, lower
    /// triangle.
    pub l: Vec<R>,
    /// The Cholesky factor with the elliptic cones' contributions (Newton), `nv * nv`
    /// (MuJoCo's `Lcone`), valid when a cone is active.
    pub lcone: Vec<R>,
    /// `L' J` of one cone contact, `6 * nv` (MuJoCo's `LTJ`).
    pub ltj: Vec<R>,
}

/// What the constraint solver measured on its last iteration (MuJoCo's
/// `mjSolverStat`, for the last iteration only), and the cost it ended at.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SolverStat<R: Real> {
    /// The primal cost (constraint plus Gauss) at the solution.
    pub cost: R,
    /// The scaled cost improvement of the last iteration.
    pub improvement: R,
    /// The scaled gradient norm after the last iteration.
    pub gradient: R,
    /// The number of rows not satisfied after the last iteration.
    pub nactive: usize,
    /// The number of rows whose zone changed in the last iteration.
    pub nchange: usize,
    /// The number of line-search evaluations of the last iteration.
    pub neval: usize,
    /// The number of rank-one Cholesky updates of the last iteration (Newton), or
    /// the number of rows when the Hessian was refactored.
    pub nupdate: usize,
    /// The largest `neval` over the iterations of the solve (the bound of a GPU port's
    /// line-search dispatch: `ls_iterations + 2`).
    pub neval_max: usize,
    /// The largest `nupdate` over the iterations of the solve, the initial factorisation
    /// (`nefc` rows) included (bounded by `nefc_max` plus the sum of the contacts' dimensions).
    pub nupdate_max: usize,
    /// The largest number of rank-one Cholesky updates of one iteration of the solve, counted
    /// as they were performed: the incremental ones and the cone ones (a rank-loss refactor's
    /// cone updates included), where `nupdate` is overwritten by the row count of a refactor
    /// (MuJoCo's semantics). The factorisation of the first iteration's cones is counted.
    pub nrank1_max: usize,
    /// The number of rank-loss refactorisations of the solve (each a full `J' D J` rebuild and
    /// Cholesky factorisation, `O(nefc nv^2 + nv^3)`), the initial factorisation not counted.
    /// An iteration refactors at most once.
    pub nrefactor: usize,
}

/// One environment: state plus scratch.
#[derive(Clone, Debug)]
pub struct Data<R: Real> {
    /// Simulation time, seconds.
    pub time: R,
    /// Joint positions (`nq`; quaternions `[w, x, y, z]`).
    pub qpos: Vec<R>,
    /// Joint velocities (`nv`).
    pub qvel: Vec<R>,
    /// Actuator controls (`nu`).
    pub ctrl: Vec<R>,
    /// The solver's starting acceleration (`nv`): the better of this and
    /// `qacc_smooth` (by cost) is where the constraint solver starts. A step sets it
    /// to its own `qacc` (MuJoCo's `mj_advance`); [`Data::reset`] zeroes it.
    pub qacc_warmstart: Vec<R>,

    /// Body frame positions in the world, `3 * nbody`.
    pub xpos: Vec<R>,
    /// Body frame orientations in the world, `4 * nbody`, `[w, x, y, z]`.
    pub xquat: Vec<R>,
    /// Body frame rotation matrices, `9 * nbody`.
    pub xmat: Vec<R>,
    /// Centre-of-mass positions in the world, `3 * nbody`.
    pub xipos: Vec<R>,
    /// Inertial frame rotation matrices, `9 * nbody`.
    pub ximat: Vec<R>,
    /// Joint anchors in the world, `3 * njnt`.
    pub xanchor: Vec<R>,
    /// Joint axes in the world, `3 * njnt`.
    pub xaxis: Vec<R>,
    /// Centre of mass of each body's subtree, `3 * nbody`.
    pub subtree_com: Vec<R>,

    /// Motion axes of the dofs in the frame centred at the tree's centre of mass, `6 * nv`.
    pub cdof: Vec<R>,
    /// Body inertias in that frame, `10 * nbody`.
    pub cinert: Vec<R>,
    /// Composite rigid body inertias, `10 * nbody`.
    pub crb: Vec<R>,
    /// Body velocities in that frame, `6 * nbody`.
    pub cvel: Vec<R>,
    /// Time derivatives of `cdof`, `6 * nv`.
    pub cdof_dot: Vec<R>,
    /// Body accelerations (scratch of the Newton-Euler pass), `6 * nbody`.
    pub cacc: Vec<R>,
    /// Body forces (scratch of the Newton-Euler pass), `6 * nbody`.
    pub cfrc_body: Vec<R>,

    /// The joint-space inertia matrix `M`, dense and symmetric, `nv * nv`
    /// (MuJoCo's `qM`), with the armature on the diagonal.
    pub qm: Vec<R>,
    /// The `L'DL` factors of `M`, `nv * nv` (MuJoCo's `qLD`).
    pub qld: Vec<R>,
    /// `1 / D`, `nv` (MuJoCo's `qLDiagInv`).
    pub qld_diag_inv: Vec<R>,
    /// The factors of `M + h diag(damping)` for the Euler step, `nv * nv`
    /// (MuJoCo's `qH`).
    pub qh: Vec<R>,
    /// `1 / D` of `qh`, `nv`.
    pub qh_diag_inv: Vec<R>,

    /// Gravity, Coriolis and centrifugal force in joint space (MuJoCo's
    /// `qfrc_bias`): `M qacc + qfrc_bias` is the inverse dynamics.
    pub qfrc_bias: Vec<R>,
    /// Joint spring force, `nv`.
    pub qfrc_spring: Vec<R>,
    /// Joint damper force, `nv`.
    pub qfrc_damper: Vec<R>,
    /// Passive force: spring plus damper, `nv`.
    pub qfrc_passive: Vec<R>,
    /// Actuator force in joint space, `nv`.
    pub qfrc_actuator: Vec<R>,
    /// `qfrc_passive - qfrc_bias + qfrc_actuator`, `nv`.
    pub qfrc_smooth: Vec<R>,
    /// The acceleration with no constraint force, `M^-1 qfrc_smooth`, `nv`.
    pub qacc_smooth: Vec<R>,
    /// The constraint force in joint space, `J' efc_force`, `nv`.
    pub qfrc_constraint: Vec<R>,
    /// Geom frame positions in the world, `3 * ngeom`.
    pub geom_xpos: Vec<R>,
    /// Geom frame rotation matrices, `9 * ngeom`.
    pub geom_xmat: Vec<R>,

    /// Joint accelerations, `nv`: `qacc_smooth` when there are no constraint rows,
    /// else the constraint solver's result.
    pub qacc: Vec<R>,
    /// The acceleration the Euler step integrates with (`qacc`, or the
    /// damping-implicit one), `nv`.
    pub qacc_step: Vec<R>,

    /// Fixed-tendon lengths, `ntendon`.
    pub ten_length: Vec<R>,
    /// Fixed-tendon Jacobians `d length / d qpos`, dense, `ntendon * nv`.
    pub ten_j: Vec<R>,
    /// Fixed-tendon velocities `ten_J qvel`, `ntendon`.
    pub ten_velocity: Vec<R>,

    /// The number of contacts of this step (excluded ones included).
    pub ncon: usize,
    /// Per contact: the distance between the surfaces along the normal (negative:
    /// penetration).
    pub contact_dist: Vec<R>,
    /// `3 * ncon_max`: the contact position, midway between the surfaces.
    pub contact_pos: Vec<R>,
    /// `9 * ncon_max`: the contact frame, row-major: normal (from `geom[0]` to `geom[1]`),
    /// tangent 1, tangent 2.
    pub contact_frame: Vec<R>,
    /// Per contact: the distance below which the contact is a constraint (`includemargin`).
    pub contact_includemargin: Vec<R>,
    /// `5 * ncon_max`: friction `[f0, f0, f1, f2, f2]`.
    pub contact_friction: Vec<R>,
    /// `2 * ncon_max`: `solref`.
    pub contact_solref: Vec<R>,
    /// `2 * ncon_max`: `solreffriction` (zero unless set).
    pub contact_solreffriction: Vec<R>,
    /// `5 * ncon_max`: `solimp`.
    pub contact_solimp: Vec<R>,
    /// Per contact: the friction of the regularised cone (MuJoCo's `contact.mu`), set with
    /// the impedance.
    pub contact_mu: Vec<R>,
    /// `36 * ncon_max`: the Hessian of the elliptic cone of each contact in the cone zone
    /// (`dim * dim` used, row-major).
    pub contact_h: Vec<R>,
    /// Per contact: its dimension (1, 3, 4 or 6).
    pub contact_dim: Vec<usize>,
    /// `2 * ncon_max`: the geom ids.
    pub contact_geom: Vec<usize>,
    /// Per contact: 0 for a constraint, 1 for a contact in the gap (excluded), 3 when the
    /// model has no dofs.
    pub contact_exclude: Vec<i32>,
    /// Per contact: the first of its constraint rows, or -1.
    pub contact_efc_address: Vec<i32>,
    /// Per candidate: how many pre-contacts its collider reported in pass 1.
    pub cand_ncon: Vec<usize>,
    /// Per candidate: the index of its first contact in the contact arrays, the exclusive prefix
    /// sum of `cand_ncon` (valid after pass 1).
    pub cand_start: Vec<usize>,
    /// Per candidate: whether its box-box collider returned more than 8 contacts in pass 1
    /// (never, in exact arithmetic).
    pub cand_overflow: Vec<bool>,
    /// Pass 1's pre-contact distances, `ncon_max`, at each candidate's `slot_offset`.
    pub pre_dist: Vec<R>,
    /// Pass 1's pre-contact positions, `3 * ncon_max`.
    pub pre_pos: Vec<R>,
    /// Pass 1's pre-contact normals and first tangents, `6 * ncon_max` (normal, tangent).
    pub pre_frame: Vec<R>,
    /// How many box-box collider calls returned more than 8 contacts (never, in exact
    /// arithmetic), summed over the collision steps since this `Data` was made; an RK4 step
    /// makes four collision steps.
    pub warning_collision_overflow: usize,
    /// The fingerprint of the candidate list this `Data` was sized for.
    pub candidates_fingerprint: u64,
    /// Scratch of the contact Jacobians (`3 * nv` each): the two bodies' translation and
    /// rotation Jacobians and their differences, and the contact's rotated rows
    /// (`6 * nv`) and pyramid edges (`2 * nv`).
    pub con_jac1p: Vec<R>,
    /// See `con_jac1p`.
    pub con_jac2p: Vec<R>,
    /// See `con_jac1p`.
    pub con_jac1r: Vec<R>,
    /// See `con_jac1p`.
    pub con_jac2r: Vec<R>,
    /// See `con_jac1p`.
    pub con_jacdifp: Vec<R>,
    /// See `con_jac1p`.
    pub con_jacdifr: Vec<R>,
    /// See `con_jac1p`.
    pub con_jac: Vec<R>,
    /// See `con_jac1p`.
    pub con_edge: Vec<R>,

    /// The number of constraint rows of this step.
    pub nefc: usize,
    /// The number of equality rows (always 0 in this phase).
    pub ne: usize,
    /// The number of friction-loss rows.
    pub nf: usize,
    /// The number of limit rows.
    pub nl: usize,
    /// Per row: what kind of constraint it is.
    pub efc_type: Vec<ConstraintType>,
    /// Per row: the joint, dof or tendon it belongs to (a friction-loss row of a dof:
    /// the dof; of a joint limit: the joint; of a tendon: the tendon).
    pub efc_id: Vec<usize>,
    /// The constraint Jacobian, dense row-major, `nefc_max * nv`.
    pub efc_j: Vec<R>,
    /// Per row: the distance to the limit (negative: past it), 0 for friction loss.
    pub efc_pos: Vec<R>,
    /// Per row: the margin of the limit, 0 for friction loss.
    pub efc_margin: Vec<R>,
    /// Per row: the friction-loss magnitude, 0 for limits.
    pub efc_frictionloss: Vec<R>,
    /// Per row: the approximate diagonal of `J M^-1 J'` (MuJoCo's `efc_diagApprox`).
    pub efc_diag_approx: Vec<R>,
    /// Per row: the regularisation `R`.
    pub efc_r: Vec<R>,
    /// Per row: `1 / R`.
    pub efc_d: Vec<R>,
    /// Per row: the spring, damper and impedance coefficients `[K, B, I, P]`,
    /// `4 * nefc_max`.
    pub efc_kbip: Vec<R>,
    /// Per row: the reference acceleration `-B v - K I (pos - margin)`.
    pub efc_aref: Vec<R>,
    /// Per row: `J qvel`.
    pub efc_vel: Vec<R>,
    /// Per row: `J qacc_smooth - aref` (the warmstart cost of `qacc_smooth`).
    pub efc_b: Vec<R>,
    /// Per row: the constraint force.
    pub efc_force: Vec<R>,
    /// Per row: the zone of the cost the solver ended in.
    pub efc_state: Vec<ConstraintState>,
    /// The number of iterations the constraint solver ran in the last forward pass.
    pub solver_niter: usize,
    /// What the solver measured on its last iteration of the last forward pass.
    pub solver_stat: SolverStat<R>,
    /// The solver's scratch arrays.
    pub ws: Workspace<R>,

    /// Runge-Kutta stage states: `4 * (nq + nv)`, stage `i` at `i * (nq + nv)`
    /// as `[qpos, qvel]` (MuJoCo's `X[i]`).
    pub rk_x: Vec<R>,
    /// Runge-Kutta stage accelerations: `4 * nv` (MuJoCo's `F[i]`).
    pub rk_f: Vec<R>,
    /// Runge-Kutta increments: `2 * nv`, `[velocity, acceleration]` (MuJoCo's `dX`).
    pub rk_dx: Vec<R>,

    /// `[potential, kinetic]` energy.
    pub energy: [R; 2],
}

impl<R: Real> Data<R> {
    /// A new environment at the model's reference pose (`qpos0`), at rest, with
    /// zero controls and zero time. Allocates every array.
    pub fn new(m: &Model<R>) -> Data<R> {
        let z = |n: usize| vec![R::ZERO; n];
        let nefc = m.nefc_max;
        let ncon = m.ncon_max;
        let mut d = Data {
            time: R::ZERO,
            qpos: m.qpos0.clone(),
            qvel: z(m.nv),
            ctrl: z(m.nu),
            qacc_warmstart: z(m.nv),
            xpos: z(3 * m.nbody),
            xquat: z(4 * m.nbody),
            xmat: z(9 * m.nbody),
            xipos: z(3 * m.nbody),
            ximat: z(9 * m.nbody),
            xanchor: z(3 * m.njnt),
            xaxis: z(3 * m.njnt),
            subtree_com: z(3 * m.nbody),
            cdof: z(6 * m.nv),
            cinert: z(10 * m.nbody),
            crb: z(10 * m.nbody),
            cvel: z(6 * m.nbody),
            cdof_dot: z(6 * m.nv),
            cacc: z(6 * m.nbody),
            cfrc_body: z(6 * m.nbody),
            qm: z(m.nv * m.nv),
            qld: z(m.nv * m.nv),
            qld_diag_inv: z(m.nv),
            qh: z(m.nv * m.nv),
            qh_diag_inv: z(m.nv),
            qfrc_bias: z(m.nv),
            qfrc_spring: z(m.nv),
            qfrc_damper: z(m.nv),
            qfrc_passive: z(m.nv),
            qfrc_actuator: z(m.nv),
            qfrc_smooth: z(m.nv),
            qacc_smooth: z(m.nv),
            qfrc_constraint: z(m.nv),
            geom_xpos: z(3 * m.ngeom),
            geom_xmat: z(9 * m.ngeom),
            qacc: z(m.nv),
            qacc_step: z(m.nv),
            ten_length: z(m.ntendon),
            ten_j: z(m.ntendon * m.nv),
            ten_velocity: z(m.ntendon),
            ncon: 0,
            contact_dist: z(ncon),
            contact_pos: z(3 * ncon),
            contact_frame: z(9 * ncon),
            contact_includemargin: z(ncon),
            contact_friction: z(5 * ncon),
            contact_solref: z(2 * ncon),
            contact_solreffriction: z(2 * ncon),
            contact_solimp: z(5 * ncon),
            contact_mu: z(ncon),
            contact_h: z(36 * ncon),
            contact_dim: vec![0; ncon],
            contact_geom: vec![0; 2 * ncon],
            contact_exclude: vec![0; ncon],
            contact_efc_address: vec![-1; ncon],
            cand_ncon: vec![0; m.candidates.len()],
            cand_start: vec![0; m.candidates.len()],
            cand_overflow: vec![false; m.candidates.len()],
            pre_dist: z(ncon),
            pre_pos: z(3 * ncon),
            pre_frame: z(6 * ncon),
            warning_collision_overflow: 0,
            candidates_fingerprint: m.candidates_fingerprint,
            con_jac1p: z(3 * m.nv),
            con_jac2p: z(3 * m.nv),
            con_jac1r: z(3 * m.nv),
            con_jac2r: z(3 * m.nv),
            con_jacdifp: z(3 * m.nv),
            con_jacdifr: z(3 * m.nv),
            con_jac: z(6 * m.nv),
            con_edge: z(2 * m.nv),
            nefc: 0,
            ne: 0,
            nf: 0,
            nl: 0,
            efc_type: vec![ConstraintType::FrictionDof; nefc],
            efc_id: vec![0; nefc],
            efc_j: z(nefc * m.nv),
            efc_pos: z(nefc),
            efc_margin: z(nefc),
            efc_frictionloss: z(nefc),
            efc_diag_approx: z(nefc),
            efc_r: z(nefc),
            efc_d: z(nefc),
            efc_kbip: z(4 * nefc),
            efc_aref: z(nefc),
            efc_vel: z(nefc),
            efc_b: z(nefc),
            efc_force: z(nefc),
            efc_state: vec![ConstraintState::Satisfied; nefc],
            solver_niter: 0,
            solver_stat: SolverStat::default(),
            ws: Workspace {
                jaref: z(nefc),
                jv: z(nefc),
                ma: z(m.nv),
                mv: z(m.nv),
                grad: z(m.nv),
                mgrad: z(m.nv),
                search: z(m.nv),
                quad: z(3 * nefc),
                oldstate: vec![ConstraintState::Satisfied; nefc],
                gradold: z(m.nv),
                mgradold: z(m.nv),
                graddif: z(m.nv),
                mgraddif: z(m.nv),
                d: z(nefc),
                cholupd: z(m.nv),
                l: z(m.nv * m.nv),
                lcone: z(m.nv * m.nv),
                ltj: z(6 * m.nv),
            },
            rk_x: z(4 * (m.nq + m.nv)),
            rk_f: z(4 * m.nv),
            rk_dx: z(2 * m.nv),
            energy: [R::ZERO; 2],
        };
        // the identity frame of the world, which no step writes again
        d.xquat[0] = R::ONE;
        d.xmat[0] = R::ONE;
        d.xmat[4] = R::ONE;
        d.xmat[8] = R::ONE;
        d.ximat[0] = R::ONE;
        d.ximat[4] = R::ONE;
        d.ximat[8] = R::ONE;
        d
    }

    /// Resets the state to the model's reference pose, at rest, with zero
    /// controls, zero time and a zero warmstart; the scratch is left as it is (a
    /// step rewrites it).
    pub fn reset(&mut self, m: &Model<R>) {
        self.time = R::ZERO;
        self.qpos.copy_from_slice(&m.qpos0);
        self.qvel.fill(R::ZERO);
        self.ctrl.fill(R::ZERO);
        self.qacc_warmstart.fill(R::ZERO);
    }

    /// Whether `self` was built for a model of `m`'s sizes.
    pub fn fits(&self, m: &Model<R>) -> bool {
        self.qpos.len() == m.nq
            && self.qvel.len() == m.nv
            && self.ctrl.len() == m.nu
            && self.xpos.len() == 3 * m.nbody
            && self.xanchor.len() == 3 * m.njnt
            && self.ten_length.len() == m.ntendon
            && self.efc_force.len() == m.nefc_max
            && self.efc_j.len() == m.nefc_max * m.nv
            && self.qacc_warmstart.len() == m.nv
            && self.geom_xpos.len() == 3 * m.ngeom
            && self.contact_dist.len() == m.ncon_max
            && self.contact_h.len() == 36 * m.ncon_max
            && self.cand_ncon.len() == m.candidates.len()
            && self.cand_start.len() == m.candidates.len()
            && self.cand_overflow.len() == m.candidates.len()
            && self.pre_dist.len() == m.ncon_max
            && self.pre_pos.len() == 3 * m.ncon_max
            && self.pre_frame.len() == 6 * m.ncon_max
            && self.candidates_fingerprint == m.candidates_fingerprint
    }
}
