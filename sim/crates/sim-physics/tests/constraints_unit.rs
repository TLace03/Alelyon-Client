//! Unit checks of the soft-constraint formulas, one row at a time, each against the formula
//! as MuJoCo's documentation states it, written out again here independently of the
//! engine's code.
//!
//! For a limit or friction-loss row with approximate inverse inertia `A`, impedance `I`
//! (from `solimp = (d0, d1, width, midpoint, power)` and `x = |pos - margin| / width`),
//! `solref = (tc, z)` and `d1 = solimp[1]`:
//!
//! - standard format (`tc > 0`, `z > 0`): `K = 1 / (d1^2 tc^2 z^2)`, `B = 2 / (d1 tc)`,
//!   and `tc` is raised to `2 timestep` unless `refsafe` is disabled;
//! - direct format (`tc <= 0`, `z <= 0`): `K = -tc / d1^2`, `B = -z / d1`;
//! - a mixed format (one positive, one not) is replaced by `(0.02, 1)`;
//! - friction loss has `K = 0`;
//! - `R = (1 - I) A / I`, `D = 1 / R`, `aref = -B v - K I (pos - margin)`;
//! - `solimp` is clamped: `d0, d1, midpoint` into `[0.0001, 0.9999]`, `width >= 0`,
//!   `power >= 1`; a flat impedance (`d0 = d1`, or no width) is `(d0 + d1) / 2`.
//!
//! The derivative of the impedance with respect to `pos` that the engine stores in
//! `efc_KBIP[3]` is checked by a central finite difference of the impedance itself.

use sim_physics::{ConstraintState, ConstraintType, Data, Model, PrimalSolver, forward, step};
use sim_scene::mjcf;

const RTOL: f64 = 1e-12;

fn model(xml: &str) -> Model<f64> {
    let scene = mjcf::load(xml, std::env::temp_dir()).expect("imports");
    Model::<f64>::compile(&scene).expect("compiles").0
}

/// One hinge about `y` carrying a point-like mass at 0.5 (so `I_pivot = 0.27`), limited to
/// `[-30, 30]` degrees with a margin of 0.01 rad.
fn hinge(timestep: f64, solref: &str, solimp: &str, frictionloss: f64) -> Model<f64> {
    model(&format!(
        r#"<mujoco><option timestep="{timestep}" integrator="Euler"/>
           <worldbody><body name="arm">
             <joint name="hinge" type="hinge" axis="0 1 0" range="-30 30" margin="0.01"
                    solreflimit="{solref}" solimplimit="{solimp}" frictionloss="{frictionloss}"/>
             <inertial pos="0 0 0.5" mass="1" diaginertia="0.02 0.02 0.02"/>
           </body></worldbody></mujoco>"#
    ))
}

const UPPER: f64 = 30.0 * std::f64::consts::PI / 180.0;

/// `forward` at joint position `q` and velocity `v`.
fn at(m: &Model<f64>, q: f64, v: f64) -> Data<f64> {
    let mut d = Data::new(m);
    d.qpos[0] = q;
    d.qvel[0] = v;
    forward(m, &mut d);
    d
}

/// MuJoCo's impedance function as documented (the one in the engine is not used here).
fn impedance(solimp: [f64; 5], pos_minus_margin: f64) -> f64 {
    let [d0, d1, width, mid, power] = solimp;
    if d0 == d1 || width <= 1e-15 {
        return 0.5 * (d0 + d1);
    }
    let x = (pos_minus_margin.abs() / width).min(1.0);
    if x >= 1.0 {
        return d1;
    }
    let y = if x <= mid {
        x.powf(power) / mid.powf(power - 1.0)
    } else {
        1.0 - (1.0 - x).powf(power) / (1.0 - mid).powf(power - 1.0)
    };
    d0 + y * (d1 - d0)
}

struct Expected {
    k: f64,
    b: f64,
    i: f64,
    r: f64,
    d: f64,
    aref: f64,
}

/// What a row with these parameters must hold; `solref` and `solimp` are the values after
/// the repairs MuJoCo makes (so the caller states them); `vel` is the row's velocity `J v`.
fn expected(
    solref: [f64; 2],
    solimp: [f64; 5],
    pos: f64,
    margin: f64,
    vel: f64,
    a: f64,
    friction: bool,
) -> Expected {
    let (tc, z) = (solref[0], solref[1]);
    let d1 = solimp[1];
    let k = if friction {
        0.0
    } else if tc > 0.0 {
        1.0 / (d1 * d1 * tc * tc * z * z)
    } else {
        -tc / (d1 * d1)
    };
    let b = if z > 0.0 { 2.0 / (d1 * tc) } else { -z / d1 };
    let i = impedance(solimp, pos - margin);
    let r = (1.0 - i) * a / i;
    Expected {
        k,
        b,
        i,
        r,
        d: 1.0 / r,
        aref: -b * vel - k * i * (pos - margin),
    }
}

fn close(label: &str, name: &str, ours: f64, want: f64) {
    let err = (ours - want).abs();
    let scale = want.abs().max(1e-300);
    assert!(
        err <= RTOL * scale + 1e-300,
        "{label} {name}: ours {ours:e} want {want:e} (relative {:e})",
        err / scale
    );
}

fn check_row(label: &str, d: &Data<f64>, row: usize, want: &Expected) {
    close(label, "K", d.efc_kbip[4 * row], want.k);
    close(label, "B", d.efc_kbip[4 * row + 1], want.b);
    close(label, "I", d.efc_kbip[4 * row + 2], want.i);
    close(label, "R", d.efc_r[row], want.r);
    close(label, "D", d.efc_d[row], want.d);
    close(label, "aref", d.efc_aref[row], want.aref);
}

/// The approximate inverse inertia of the hinge, checked against `1 / I_pivot` first.
fn inverse_inertia(m: &Model<f64>) -> f64 {
    let a = m.dof_invweight0[0];
    let want = 1.0 / (0.02 + 0.5 * 0.5);
    assert!((a - want).abs() < 1e-12 * want, "{a} vs {want}");
    a
}

#[test]
fn limit_rows_follow_the_documented_formulas() {
    // (label, solref, solimp, how far past the limit, velocity)
    let cases: [(&str, &str, &str, f64, f64); 8] = [
        (
            "standard, upper half of the ramp",
            "0.05 0.8",
            "0.8 0.95 0.02 0.4 3",
            0.004,
            0.7,
        ),
        (
            "standard, lower half of the ramp",
            "0.05 0.8",
            "0.6 0.9 0.1 0.5 2",
            0.01,
            -0.4,
        ),
        (
            "linear ramp (power 1)",
            "0.03 1.2",
            "0.6 0.9 0.05 0.5 1",
            0.002,
            0.2,
        ),
        ("direct", "-400 -15", "0.85 0.95 0.02 0.6 2.5", 0.003, 0.5),
        (
            "flat impedance",
            "0.04 1",
            "0.9 0.9 0.001 0.5 2",
            0.003,
            0.1,
        ),
        (
            "saturated impedance",
            "0.04 1",
            "0.5 0.9 0.01 0.5 2",
            0.5,
            0.0,
        ),
        (
            "inside the margin, not yet past the limit",
            "0.05 0.8",
            "0.8 0.95 0.02 0.4 3",
            -0.005,
            0.3,
        ),
        (
            "exactly at the limit",
            "0.05 0.8",
            "0.8 0.95 0.02 0.4 3",
            0.0,
            0.0,
        ),
    ];
    for (label, solref, solimp, past, v) in cases {
        let m = hinge(0.002, solref, solimp, 0.0);
        let a = inverse_inertia(&m);
        let d = at(&m, UPPER + past, v);
        assert_eq!(d.nefc, 1, "{label}");
        assert_eq!(d.efc_type[0], ConstraintType::LimitJoint, "{label}");
        assert_eq!(d.efc_id[0], 0, "{label}");
        // the upper side: distance = hi - q, Jacobian -1
        assert!(
            (d.efc_pos[0] + past).abs() < 1e-14,
            "{label}: pos {:e}",
            d.efc_pos[0]
        );
        assert_eq!(d.efc_margin[0], 0.01);
        assert_eq!(d.efc_j[0], -1.0, "{label}");
        let sr: Vec<f64> = solref.split(' ').map(|s| s.parse().unwrap()).collect();
        let si: Vec<f64> = solimp.split(' ').map(|s| s.parse().unwrap()).collect();
        let want = expected(
            [sr[0], sr[1]],
            [si[0], si[1], si[2], si[3], si[4]],
            d.efc_pos[0],
            0.01,
            // the row's velocity is J v, and J = -1 on the upper side
            -v,
            a,
            false,
        );
        assert_eq!(d.efc_vel[0], -v, "{label}");
        check_row(label, &d, 0, &want);
    }
}

#[test]
fn the_lower_side_has_the_opposite_jacobian_and_the_distance_from_the_lower_bound() {
    let m = hinge(0.002, "0.05 0.8", "0.8 0.95 0.02 0.4 3", 0.0);
    let d = at(&m, -UPPER - 0.004, 0.0);
    assert_eq!(d.nefc, 1);
    assert_eq!(d.efc_j[0], 1.0);
    assert!((d.efc_pos[0] + 0.004).abs() < 1e-14, "{:e}", d.efc_pos[0]);
    // well inside the range: no row
    assert_eq!(at(&m, 0.0, 0.0).nefc, 0);
    // just outside the margin: no row; just inside it: a row
    assert_eq!(at(&m, UPPER - 0.0101, 0.0).nefc, 0);
    assert_eq!(at(&m, UPPER - 0.0099, 0.0).nefc, 1);
}

#[test]
fn friction_rows_have_no_stiffness_and_the_friction_solref() {
    let xml_model = |solref: &str| {
        model(&format!(
            r#"<mujoco><option timestep="0.002" integrator="Euler"/>
               <worldbody><body name="b"><joint name="s" type="slide" axis="1 0 0"
                   frictionloss="0.5" solreffriction="{solref}" solimpfriction="0.9 0.95 0.001 0.5 2"/>
                 <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/></body></worldbody></mujoco>"#
        ))
    };
    let m = xml_model("0.02 1");
    let a = m.dof_invweight0[0];
    assert!(
        (a - 1.0).abs() < 1e-15,
        "a simple slide has inverse inertia 1 / mass: {a}"
    );
    let d = at(&m, 0.0, 0.3);
    assert_eq!(d.nefc, 1);
    assert_eq!(d.efc_type[0], ConstraintType::FrictionDof);
    assert_eq!(d.efc_frictionloss[0], 0.5);
    assert_eq!(d.efc_pos[0], 0.0);
    assert_eq!(d.efc_margin[0], 0.0);
    let want = expected(
        [0.02, 1.0],
        [0.9, 0.95, 0.001, 0.5, 2.0],
        0.0,
        0.0,
        0.3,
        a,
        true,
    );
    check_row("friction", &d, 0, &want);
    assert_eq!(d.efc_kbip[0], 0.0, "friction loss has no stiffness");
    // a solref below twice the timestep is raised for friction too
    let m = xml_model("0.001 1");
    let d = at(&m, 0.0, 0.3);
    let want = expected(
        [0.004, 1.0],
        [0.9, 0.95, 0.001, 0.5, 2.0],
        0.0,
        0.0,
        0.3,
        a,
        true,
    );
    check_row("friction, refsafe", &d, 0, &want);
}

#[test]
fn refsafe_raises_the_time_constant_to_twice_the_timestep_unless_disabled() {
    let solimp = [0.8, 0.95, 0.02, 0.4, 3.0];
    let text = "0.8 0.95 0.02 0.4 3";
    let mut m = hinge(0.01, "0.005 1", text, 0.0);
    let a = inverse_inertia(&m);
    let q = UPPER + 0.004;
    let on = at(&m, q, 0.5);
    let pos = on.efc_pos[0];
    check_row(
        "refsafe on",
        &on,
        0,
        &expected([0.02, 1.0], solimp, pos, 0.01, -0.5, a, false),
    );
    m.disable.refsafe = true;
    let off = at(&m, q, 0.5);
    check_row(
        "refsafe off",
        &off,
        0,
        &expected([0.005, 1.0], solimp, pos, 0.01, -0.5, a, false),
    );
    // a time constant already above the bound, and the direct format, are left alone
    m.disable.refsafe = false;
    m.jnt_solref = vec![0.05, 1.0];
    check_row(
        "above the bound",
        &at(&m, q, 0.5),
        0,
        &expected([0.05, 1.0], solimp, pos, 0.01, -0.5, a, false),
    );
    m.jnt_solref = vec![-300.0, -12.0];
    check_row(
        "direct",
        &at(&m, q, 0.5),
        0,
        &expected([-300.0, -12.0], solimp, pos, 0.01, -0.5, a, false),
    );
}

#[test]
fn a_mixed_solref_is_replaced_by_the_default() {
    let solimp = [0.8, 0.95, 0.02, 0.4, 3.0];
    let mut m = hinge(0.002, "0.05 1", "0.8 0.95 0.02 0.4 3", 0.0);
    let a = inverse_inertia(&m);
    let q = UPPER + 0.004;
    let pos = -0.004;
    for mixed in [[0.05, -1.0], [-0.05, 1.0]] {
        m.jnt_solref = mixed.to_vec();
        check_row(
            "mixed",
            &at(&m, q, 0.5),
            0,
            &expected([0.02, 1.0], solimp, pos, 0.01, -0.5, a, false),
        );
    }
}

#[test]
fn solimp_is_clamped_as_mujoco_clamps_it() {
    let mut m = hinge(0.002, "0.05 1", "0.8 0.95 0.02 0.4 3", 0.0);
    let a = inverse_inertia(&m);
    let q = UPPER + 0.004;
    let pos = -0.004;
    // (raw solimp, solimp after the clamps)
    let cases: [([f64; 5], [f64; 5]); 4] = [
        // d0 and d1 into [1e-4, 0.9999], negative width to 0 (a flat impedance), power to 1
        (
            [0.0, 1.5, -1.0, 0.0, 0.5],
            [0.0001, 0.9999, 0.0, 0.0001, 1.0],
        ),
        // d0 = d1 above the top: flat at the top
        (
            [2.0, 2.0, 0.02, 2.0, 3.0],
            [0.9999, 0.9999, 0.02, 0.9999, 3.0],
        ),
        // a power below 1 is 1
        ([0.5, 0.9, 0.05, 0.5, 0.3], [0.5, 0.9, 0.05, 0.5, 1.0]),
        // the midpoint into the open interval
        ([0.5, 0.9, 0.05, 1.7, 2.0], [0.5, 0.9, 0.05, 0.9999, 2.0]),
    ];
    for (raw, clamped) in cases {
        m.jnt_solimp = raw.to_vec();
        let d = at(&m, q, 0.5);
        check_row(
            &format!("solimp {raw:?}"),
            &d,
            0,
            &expected([0.05, 1.0], clamped, pos, 0.01, -0.5, a, false),
        );
        // the impedance is a probability-like number: strictly inside (0, 1)
        let i = d.efc_kbip[2];
        assert!(i > 0.0 && i < 1.0, "{i}");
    }
}

#[test]
fn the_impedance_derivative_is_the_finite_difference_of_the_impedance() {
    for solimp in [
        "0.8 0.95 0.02 0.4 3",
        "0.6 0.9 0.1 0.5 2",
        "0.6 0.9 0.05 0.5 1",
    ] {
        let m = hinge(0.002, "0.05 1", solimp, 0.0);
        let h = 1e-7;
        // penetrations chosen away from the midpoint's kink and from saturation
        for past in [0.001, 0.003, 0.007, 0.012, 0.06] {
            let q = UPPER + past;
            let d = at(&m, q, 0.0);
            // pos = hi - q: a larger q is a smaller pos
            let i_plus = at(&m, q - h, 0.0).efc_kbip[2]; // pos + h
            let i_minus = at(&m, q + h, 0.0).efc_kbip[2]; // pos - h
            let fd = (i_plus - i_minus) / (2.0 * h);
            let imp_p = d.efc_kbip[3];
            let scale = imp_p.abs().max(1e-12);
            println!(
                "impedance derivative, solimp {solimp}, {past} past the limit: engine {imp_p:e}, finite difference {fd:e}"
            );
            assert!(
                (imp_p - fd).abs() <= 1e-5 * scale,
                "solimp {solimp} past {past}: engine {imp_p:e} vs finite difference {fd:e}"
            );
        }
    }
}

/// A slide with friction loss 0.5 and mass 1, no gravity, so `qacc_smooth = 0`: the row's
/// state and force follow from `jar = J a - aref = a + B v`.
fn friction_slide() -> Model<f64> {
    let mut m = model(
        r#"<mujoco><option timestep="0.002" integrator="Euler" gravity="0 0 0"/>
           <worldbody><body name="b"><joint name="s" type="slide" axis="1 0 0"
               frictionloss="0.5" solreffriction="0.02 1" solimpfriction="0.9 0.95 0.001 0.5 2"/>
             <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/></body></worldbody></mujoco>"#,
    );
    m.opt.tolerance = 1e-14;
    m
}

#[test]
fn friction_loss_is_quadratic_near_zero_velocity_and_linear_beyond() {
    // A = 1, I = 0.9, R = (1 - I) A / I = 1/9, D = 9, B = 2 / (0.95 * 0.02), f = 0.5,
    // R f = 0.0556. At rest in the quadratic zone the joint accelerates by
    // D aref / (1 + D) = -D B v / (1 + D); beyond it the force is -+f and a = -+f / m.
    let b = 2.0 / (0.95 * 0.02);
    let (r, f) = (1.0 / 9.0, 0.5);
    for solver in [PrimalSolver::Newton, PrimalSolver::Cg] {
        let mut m = friction_slide();
        m.opt.solver = solver;
        // linear positive (the joint moves +): friction pushes back with exactly f
        let d = at(&m, 0.0, 0.3);
        assert_eq!(d.efc_state[0], ConstraintState::LinearPos, "{solver:?}");
        close("linear +", "force", d.efc_force[0], -f);
        close("linear +", "qacc", d.qacc[0], -f);
        close("linear +", "qfrc_constraint", d.qfrc_constraint[0], -f);
        // linear negative
        let d = at(&m, 0.0, -0.3);
        assert_eq!(d.efc_state[0], ConstraintState::LinearNeg, "{solver:?}");
        close("linear -", "force", d.efc_force[0], f);
        close("linear -", "qacc", d.qacc[0], f);
        // quadratic: |jar| = B v / (1 + D) < R f
        let v = 1e-4;
        let d = at(&m, 0.0, v);
        assert_eq!(d.efc_state[0], ConstraintState::Quadratic, "{solver:?}");
        let jar = b * v / 10.0;
        assert!(jar < r * f);
        close("quadratic", "force", d.efc_force[0], -9.0 * jar);
        close("quadratic", "qacc", d.qacc[0], -9.0 * b * v / 10.0);
        // exactly at rest: the quadratic zone with no force
        let d = at(&m, 0.0, 0.0);
        assert_eq!(d.efc_state[0], ConstraintState::Quadratic, "{solver:?}");
        assert_eq!(d.efc_force[0].abs(), 0.0);
    }
}

#[test]
fn a_limit_row_is_satisfied_when_it_is_not_pushed_into() {
    // the pendulum at the upper limit with gravity: pressed in, so quadratic with a force
    let m = hinge(0.002, "0.02 1", "0.9 0.95 0.001 0.5 2", 0.0);
    let pressed = at(&m, UPPER + 0.0001, 0.0);
    assert_eq!(pressed.efc_state[0], ConstraintState::Quadratic);
    assert!(pressed.efc_force[0] > 0.0);
    // moving away from the limit fast enough that the reference acceleration pulls it off:
    // inside the margin, the row is satisfied and carries no force
    let away = at(&m, UPPER - 0.005, -2.0);
    assert_eq!(away.nefc, 1);
    assert_eq!(away.efc_state[0], ConstraintState::Satisfied);
    assert_eq!(away.efc_force[0], 0.0);
    assert_eq!(away.qfrc_constraint[0], 0.0);
}

#[test]
fn a_row_with_an_empty_jacobian_is_not_instantiated() {
    // a limited fixed tendon on a slide: length = qpos, below its range at 0
    let mut m = model(
        r#"<mujoco><worldbody><body name="b"><joint name="s" type="slide" axis="1 0 0"/>
             <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/></body></worldbody>
           <tendon><fixed name="t" limited="true" range="0.5 1"><joint joint="s" coef="1"/></fixed></tendon>
           </mujoco>"#,
    );
    let d = at(&m, 0.0, 0.0);
    assert_eq!(d.nefc, 1, "the tendon is below its range: one row");
    assert_eq!(d.efc_type[0], ConstraintType::LimitTendon);
    assert_eq!(d.efc_j[0], 1.0);
    // the same tendon with a zero coefficient has a zero Jacobian and length: MuJoCo does not
    // add the row (its guard in mj_addConstraint), so neither do we
    m.wrap_prm[0] = 0.0;
    let d = at(&m, 0.0, 0.0);
    assert_eq!(d.ten_j[0], 0.0);
    assert_eq!(d.nefc, 0);
    assert_eq!(d.nl, 0);
}

#[test]
fn the_disable_flags_remove_their_rows() {
    let mut m = hinge(0.002, "0.05 1", "0.8 0.95 0.02 0.4 3", 0.3);
    let q = UPPER + 0.004;
    let d = at(&m, q, 0.1);
    assert_eq!((d.nefc, d.nf, d.nl), (2, 1, 1));
    assert_eq!(
        d.efc_type[0],
        ConstraintType::FrictionDof,
        "friction rows come first"
    );
    assert_eq!(d.efc_type[1], ConstraintType::LimitJoint);

    m.disable.frictionloss = true;
    let d = at(&m, q, 0.1);
    assert_eq!((d.nefc, d.nf, d.nl), (1, 0, 1));
    assert_eq!(d.efc_type[0], ConstraintType::LimitJoint);

    m.disable.frictionloss = false;
    m.disable.limit = true;
    let d = at(&m, q, 0.1);
    assert_eq!((d.nefc, d.nf, d.nl), (1, 1, 0));
    assert_eq!(d.efc_type[0], ConstraintType::FrictionDof);

    m.disable.limit = false;
    m.disable.constraint = true;
    let d = at(&m, q, 0.1);
    assert_eq!((d.nefc, d.nf, d.nl), (0, 0, 0));
    // no constraints: the acceleration is the smooth one, bit for bit, and no force
    assert_eq!(d.qacc, d.qacc_smooth);
    assert_eq!(d.qfrc_constraint, vec![0.0]);
    assert_eq!(d.solver_niter, 0);
}

#[test]
fn the_warm_start_is_the_better_of_the_given_one_and_the_smooth_acceleration() {
    let mut m = hinge(0.002, "0.02 1", "0.9 0.95 0.001 0.5 2", 0.0);
    m.opt.tolerance = 1e-12;
    let q = UPPER + 0.01;
    let cold = at(&m, q, 0.0); // zero warm start
    assert!(cold.solver_niter >= 1);
    let optimum = cold.qacc[0];

    let run = |m: &Model<f64>, ws: f64, disable: bool| {
        let mut mm = m.clone();
        mm.disable.warmstart = disable;
        let mut d = Data::new(&mm);
        d.qpos[0] = q;
        d.qacc_warmstart[0] = ws;
        forward(&mm, &mut d);
        d
    };
    println!(
        "warm start iterations: zero {}, perfect {}, garbage {}, disabled (perfect given) {}",
        cold.solver_niter,
        run(&m, optimum, false).solver_niter,
        run(&m, 1e6, false).solver_niter,
        run(&m, optimum, true).solver_niter
    );

    // With no iterations the solver returns where it starts, which shows the start.
    let mut none = m.clone();
    none.opt.iterations = 0;
    let smooth = run(&none, 0.0, true).qacc_smooth[0];
    assert_ne!(smooth, optimum);
    // a perfect warm start costs less than the smooth acceleration: it is used
    assert_eq!(
        run(&none, optimum, false).qacc[0].to_bits(),
        optimum.to_bits()
    );
    // a terrible one costs more: the smooth acceleration is used instead
    assert_eq!(run(&none, 1e6, false).qacc[0].to_bits(), smooth.to_bits());
    // mjDSBL_WARMSTART ignores even a perfect one, and starts with no force
    let disabled = run(&none, optimum, true);
    assert_eq!(disabled.qacc[0].to_bits(), smooth.to_bits());

    // a perfect start is already converged: it needs fewer iterations than one from the smooth
    // acceleration, and both end at the optimum
    let perfect = run(&m, optimum, false);
    let from_smooth = run(&m, 1e6, false);
    assert!(perfect.solver_niter < from_smooth.solver_niter);
    assert!((perfect.qacc[0] - optimum).abs() <= 1e-9 * optimum.abs());
    assert!((from_smooth.qacc[0] - optimum).abs() <= 1e-9 * optimum.abs());
}

#[test]
fn the_step_saves_qacc_as_the_next_warm_start() {
    // mj_advance: qacc_warmstart = qacc after every step
    let m = hinge(0.002, "0.02 1", "0.9 0.95 0.001 0.5 2", 0.0);
    let mut d = Data::new(&m);
    d.qpos[0] = UPPER + 0.01;
    assert_eq!(d.qacc_warmstart[0], 0.0);
    step(&m, &mut d);
    assert_ne!(d.qacc[0], 0.0);
    assert_eq!(d.qacc_warmstart[0].to_bits(), d.qacc[0].to_bits());
}

#[test]
fn data_is_sized_for_the_models_largest_constraint_count_and_reset_clears_the_warm_start() {
    let m = hinge(0.002, "0.02 1", "0.9 0.95 0.001 0.5 2", 0.3);
    // friction loss (1) + the two sides of the limit (2)
    assert_eq!(m.nefc_max, 3);
    let mut d = Data::new(&m);
    assert!(d.fits(&m));
    assert_eq!(d.efc_force.len(), 3);
    assert_eq!(d.efc_j.len(), 3);
    assert_eq!(d.efc_kbip.len(), 12);
    let other = model(
        r#"<mujoco><worldbody><body name="a"><joint type="slide" axis="1 0 0" frictionloss="1"/>
             <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
             <body name="b" pos="0 0 1"><joint type="slide" axis="0 1 0"/>
               <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/></body></body>
           </worldbody></mujoco>"#,
    );
    assert!(!d.fits(&other));
    d.qpos[0] = UPPER + 0.01;
    d.qvel[0] = 1.0;
    step(&m, &mut d);
    assert_ne!(d.qacc_warmstart[0], 0.0);
    d.reset(&m);
    assert_eq!(d.qacc_warmstart, vec![0.0]);
    assert_eq!(d.qvel, vec![0.0]);
    assert_eq!(d.time, 0.0);
    assert!(d.fits(&m));
}
