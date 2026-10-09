//! The boundary with `sim-world`: the quaternion order conversion, and the
//! batched step over a [`HostWorld`].
//!
//! Invariants:
//! - **One conversion function.** `HostWorld`'s `Qpos` stores quaternions as
//!   `[x, y, z, w]` (the scene's convention); this crate stores them as MuJoCo's
//!   `[w, x, y, z]`. [`convert_qpos`] is the only code that reorders them, in both
//!   directions; the body pose written back (`BodyQuat`) is converted by
//!   [`xyzw`], which is the same reordering applied to one quaternion.
//! - **Envs are independent.** [`step_world`] steps each environment from its own
//!   `Data`, `qpos`, `qvel` and `ctrl`; nothing is shared between environments, so
//!   an environment's result does not depend on how many others there are, or on
//!   their order (tested bitwise).
//! - **What is written back.** After the step, for every environment: `Qpos` and
//!   `Qvel` (the new state, quaternions as `[x, y, z, w]`), and for every scene
//!   body `b` (internal body `b + 1`): `BodyPos`, `BodyQuat` (`[x, y, z, w]`) from
//!   the new kinematics, and `BodyLinVel`, `BodyAngVel` as MuJoCo's
//!   `mj_objectVelocity(mjOBJ_XBODY, flg_local = 0)` returns them for the new
//!   state: the world-frame velocity of the body frame's origin, and the
//!   world-frame angular velocity. MuJoCo returns the pair as `[angular, linear]`;
//!   it is split here. After `step_world`, `gather_render_view` shows the stepped
//!   poses. Padding and the other fields are left as they were.
//! - The kinematics and velocities of the new state are one extra position and
//!   velocity pass after the step; the next step's forward pass recomputes them
//!   from `qpos` anyway.
//! - **Contact warnings stay in the `Data`.** The caller owns each environment's [`Data`], and
//!   the box-box overflow count `Data::warning_collision_overflow` accumulates there (per
//!   collider call that returned more than 8 contacts, which exact arithmetic cannot do). MuJoCo
//!   calls `mjERROR` for it; here the collision step keeps the first 8 contacts and
//!   `debug_assert`s, and `step_world` returns `Ok`: a caller that wants to know reads the field
//!   after [`step_world`], per environment.

use sim_world::{FieldId, HostWorld};

use crate::data::Data;
use crate::integrate::step;
use crate::model::{JointType, Model, PhysicsError};
use crate::real::Real;
use crate::smooth::{body_velocity, com_pos, com_vel, kinematics};

/// Which order a quaternion is stored in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuatOrder {
    /// `[x, y, z, w]`, scalar last: the scene, `HostWorld` and `sim-contract`.
    Xyzw,
    /// `[w, x, y, z]`, scalar first: MuJoCo and the inside of this crate.
    Wxyz,
}

/// Reorders one quaternion between the two conventions (`[w, x, y, z]` and
/// `[x, y, z, w]` are each other's rotation by one place).
pub fn reorder_quat<R: Real>(q: [R; 4], from: QuatOrder, to: QuatOrder) -> [R; 4] {
    match (from, to) {
        (QuatOrder::Xyzw, QuatOrder::Wxyz) => [q[3], q[0], q[1], q[2]],
        (QuatOrder::Wxyz, QuatOrder::Xyzw) => [q[1], q[2], q[3], q[0]],
        _ => q,
    }
}

/// `[w, x, y, z]` as `[x, y, z, w]`.
fn xyzw<R: Real>(q: [R; 4]) -> [R; 4] {
    reorder_quat(q, QuatOrder::Wxyz, QuatOrder::Xyzw)
}

/// Converts a joint-coordinate vector between the quaternion orders: copies `src`
/// (in order `from`) to `dst` (in order `to`), reordering the quaternion of every
/// free joint (after its 3 position numbers) and ball joint (all 4 numbers).
/// Hinge and slide coordinates are copied. `src` and `dst` have `model.nq` numbers.
///
/// This is the single place where the physics order and the world order meet.
pub fn convert_qpos<R: Real>(
    model: &Model<R>,
    src: &[R],
    from: QuatOrder,
    dst: &mut [R],
    to: QuatOrder,
) {
    dst[..model.nq].copy_from_slice(&src[..model.nq]);
    for j in 0..model.njnt {
        let adr = match model.jnt_type[j] {
            JointType::Free => model.jnt_qposadr[j] + 3,
            JointType::Ball => model.jnt_qposadr[j],
            JointType::Hinge | JointType::Slide => continue,
        };
        let q = reorder_quat(
            [src[adr], src[adr + 1], src[adr + 2], src[adr + 3]],
            from,
            to,
        );
        dst[adr..adr + 4].copy_from_slice(&q);
    }
}

fn check_sizes(
    model: &Model<f32>,
    datas: &[Data<f32>],
    world: &HostWorld,
) -> Result<(), PhysicsError> {
    let layout = world.layout();
    if datas.len() != layout.n_envs as usize {
        return Err(PhysicsError::Mismatch {
            reason: "the number of Data is not the number of environments of the world",
        });
    }
    if layout.nq as usize != model.nq || layout.nv as usize != model.nv {
        return Err(PhysicsError::Mismatch {
            reason: "the world's nq or nv differs from the model's",
        });
    }
    if layout.n_actuators as usize != model.nu {
        return Err(PhysicsError::Mismatch {
            reason: "the world's number of actuators differs from the model's",
        });
    }
    if layout.n_bodies as usize + 1 != model.nbody {
        return Err(PhysicsError::Mismatch {
            reason: "the world's number of bodies differs from the model's",
        });
    }
    if datas.iter().any(|d| !d.fits(model)) {
        return Err(PhysicsError::Mismatch {
            reason: "a Data was not built for this model",
        });
    }
    Ok(())
}

/// Steps every environment of `world` once, in `f32`: loads each environment's
/// `qpos`, `qvel` and `ctrl` from `world` into `datas[env]`, runs
/// [`crate::step`], and stores the new state and the body poses and velocities
/// back (see the module note for exactly what is written).
///
/// Refuses (and changes nothing) when the sizes of `model`, `datas` and `world`
/// disagree.
pub fn step_world(
    model: &Model<f32>,
    datas: &mut [Data<f32>],
    world: &mut HostWorld,
) -> Result<(), PhysicsError> {
    check_sizes(model, datas, world)?;
    let n_bodies = model.nbody - 1;
    for (env, d) in datas.iter_mut().enumerate() {
        let env = env as u32;
        // load: qpos in the world's order, qvel and ctrl as they are
        convert_qpos(
            model,
            world.qpos(env),
            QuatOrder::Xyzw,
            &mut d.qpos,
            QuatOrder::Wxyz,
        );
        d.qvel.copy_from_slice(world.qvel(env));
        d.ctrl.copy_from_slice(world.ctrl(env));

        step(model, d);

        // the poses and velocities of the new state
        kinematics(model, d);
        com_pos(model, d);
        com_vel(model, d);

        // store: qpos back in the world's order
        convert_qpos(
            model,
            &d.qpos,
            QuatOrder::Wxyz,
            world.env_slice_mut(FieldId::Qpos, env),
            QuatOrder::Xyzw,
        );
        world
            .env_slice_mut(FieldId::Qvel, env)
            .copy_from_slice(&d.qvel);
        for b in 0..n_bodies {
            let i = b + 1;
            let pos = [d.xpos[3 * i], d.xpos[3 * i + 1], d.xpos[3 * i + 2]];
            let quat = xyzw([
                d.xquat[4 * i],
                d.xquat[4 * i + 1],
                d.xquat[4 * i + 2],
                d.xquat[4 * i + 3],
            ]);
            // [angular, linear], both in the world frame
            let vel = body_velocity(model, d, i);
            world.env_slice_mut(FieldId::BodyPos, env)[3 * b..3 * b + 3].copy_from_slice(&pos);
            world.env_slice_mut(FieldId::BodyQuat, env)[4 * b..4 * b + 4].copy_from_slice(&quat);
            world.env_slice_mut(FieldId::BodyLinVel, env)[3 * b..3 * b + 3]
                .copy_from_slice(&vel[3..6]);
            world.env_slice_mut(FieldId::BodyAngVel, env)[3 * b..3 * b + 3]
                .copy_from_slice(&vel[0..3]);
        }
    }
    Ok(())
}
