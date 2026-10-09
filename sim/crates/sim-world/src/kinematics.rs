//! Forward kinematics of a scene, on the host in `f64`.
//!
//! Turns joint coordinates into the world pose of every body. It is the
//! reference the host mirror uses to build body poses from `qpos` at reset; the
//! physics kernels of later phases will compute the same thing on the device.
//!
//! Invariants:
//! - Joint coordinates are laid out as in [`crate::WorldLayout`]: free joints 7
//!   (position, quaternion `[x, y, z, w]`), ball joints 4 (quaternion), hinge and
//!   slide 1, in the scene's joint order. Quaternions in `qpos` are renormalised
//!   here, as MuJoCo's `mj_kinematics` does.
//! - A body's pose is its parent's, composed with the body's own frame, then with
//!   each of its joints in order. A free joint replaces the pose with the joint
//!   coordinates (it is only legal on a child of the world). A hinge or ball joint
//!   rotates the body about its anchor (the joint's `pos`, in the body frame); a
//!   slide translates along its axis in the body frame.
//! - Reference coordinates (`qpos0`): free joint = the body's own pose, ball =
//!   identity, hinge and slide = 0. At `qpos0` every body is where the scene's body
//!   frames put it.
//! - Deterministic: the same inputs give the same bits on a given platform. The
//!   trigonometry is the platform's `f64` `sin`/`cos`; cross-platform bit equality
//!   of the libm is not claimed.

use sim_scene::pose::{Pose, quat_from_axis_angle, quat_mul, quat_normalize, quat_rotate};
use sim_scene::{JointKind, Scene};

/// One joint as the kinematics needs it.
#[derive(Clone, Debug)]
pub(crate) struct KinJoint {
    pub kind: JointKind,
    pub pos: [f64; 3],
    pub range: Option<[f64; 2]>,
    pub qpos_adr: usize,
}

/// The kinematic tree of a scene.
#[derive(Clone, Debug)]
pub(crate) struct Kinematics {
    /// Per body: its parent and its own frame.
    pub bodies: Vec<(Option<usize>, Pose)>,
    /// Joints, in scene order.
    pub joints: Vec<KinJoint>,
    /// Per body: the range of `joints` that belong to it.
    pub joints_of_body: Vec<std::ops::Range<usize>>,
    /// The reference joint coordinates.
    pub qpos0: Vec<f64>,
}

impl Kinematics {
    pub fn new(scene: &Scene) -> Kinematics {
        let bodies: Vec<(Option<usize>, Pose)> = scene
            .bodies
            .iter()
            .map(|b| (b.parent.map(|p| p.index()), Pose::new(b.pos, b.quat)))
            .collect();
        let mut joints = Vec::with_capacity(scene.joints.len());
        let mut qpos0 = Vec::with_capacity(scene.nq());
        let mut adr = 0usize;
        for j in &scene.joints {
            joints.push(KinJoint {
                kind: j.kind,
                pos: j.pos,
                range: j.range,
                qpos_adr: adr,
            });
            match j.kind {
                JointKind::Free => {
                    let body = &scene.bodies[j.body.index()];
                    qpos0.extend_from_slice(&body.pos);
                    qpos0.extend_from_slice(&quat_normalize(body.quat));
                }
                JointKind::Ball => qpos0.extend_from_slice(&[0.0, 0.0, 0.0, 1.0]),
                JointKind::Hinge { .. } | JointKind::Slide { .. } => qpos0.push(0.0),
            }
            adr += j.kind.nq();
        }
        // joints are sorted by body (validated), so each body's joints are contiguous
        let mut joints_of_body = vec![0..0; bodies.len()];
        let mut start = 0usize;
        for (b, range) in joints_of_body.iter_mut().enumerate() {
            let mut end = start;
            while end < scene.joints.len() && scene.joints[end].body.index() == b {
                end += 1;
            }
            *range = start..end;
            start = end;
        }
        Kinematics {
            bodies,
            joints,
            joints_of_body,
            qpos0,
        }
    }

    /// The world pose of every body for joint coordinates `qpos`.
    pub fn forward(&self, qpos: &[f64]) -> Vec<Pose> {
        let mut world: Vec<Pose> = Vec::with_capacity(self.bodies.len());
        for (b, (parent, local)) in self.bodies.iter().enumerate() {
            let parent_pose = match parent {
                Some(p) => world[*p],
                None => Pose::IDENTITY,
            };
            let mut pose = parent_pose.compose(local);
            for joint in &self.joints[self.joints_of_body[b].clone()] {
                pose = apply_joint(pose, joint, &qpos[joint.qpos_adr..]);
            }
            world.push(pose);
        }
        world
    }
}

fn apply_joint(pose: Pose, joint: &KinJoint, q: &[f64]) -> Pose {
    match joint.kind {
        JointKind::Free => Pose::new([q[0], q[1], q[2]], [q[3], q[4], q[5], q[6]]),
        JointKind::Ball => rotate_about_anchor(pose, joint.pos, [q[0], q[1], q[2], q[3]]),
        JointKind::Hinge { axis } => {
            rotate_about_anchor(pose, joint.pos, quat_from_axis_angle(axis, q[0]))
        }
        JointKind::Slide { axis } => {
            let d = quat_rotate(pose.quat, axis);
            Pose {
                pos: [
                    pose.pos[0] + d[0] * q[0],
                    pose.pos[1] + d[1] * q[0],
                    pose.pos[2] + d[2] * q[0],
                ],
                quat: pose.quat,
            }
        }
    }
}

/// Rotates the body by `rotation` (in its own frame) about the anchor `anchor`
/// (in its own frame): the anchor stays where it is in the world.
fn rotate_about_anchor(pose: Pose, anchor: [f64; 3], rotation: [f64; 4]) -> Pose {
    let anchor_world = pose.transform_point(anchor);
    let quat = quat_normalize(quat_mul(pose.quat, quat_normalize(rotation)));
    let offset = quat_rotate(quat, anchor);
    Pose {
        pos: [
            anchor_world[0] - offset[0],
            anchor_world[1] - offset[1],
            anchor_world[2] - offset[2],
        ],
        quat,
    }
}
