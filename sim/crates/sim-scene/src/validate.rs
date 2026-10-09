//! `Scene::validate`: every invariant the scene types promise.
//!
//! Invariants of the checks themselves:
//! - NaN and infinity are refused before any range comparison, so no comparison
//!   is ever made with a NaN.
//! - A refusal names the field as a path into the scene and the rule, and
//!   returns the first violation found, in a fixed order (header, materials,
//!   meshes, bodies, joints, geoms, contact exclusions, instances, cameras,
//!   actuators, tendons), so
//!   the same bad scene always gives the same error.
//! - Unit quaternions are accepted within [`QUAT_TOLERANCE`] of length 1 (the
//!   MJCF importer produces them within 1e-15).
//! - Nothing here panics: ids are range-checked before they are used to index.

use std::collections::HashSet;

use crate::body::{JointKind, Shape};
use crate::error::{Result, SceneError};
use crate::pose::quat_norm;
use crate::scene::{ActuatorKind, Camera, CameraMount, Instance, SCENE_VERSION, Scene, ShapeRef};

/// How far from 1 a unit quaternion's (or axis') length may be.
pub const QUAT_TOLERANCE: f64 = 1e-6;

fn finite(path: &str, v: f64) -> Result<()> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(SceneError::invalid(path, "must be finite"))
    }
}

fn finite_all(path: &str, vs: &[f64]) -> Result<()> {
    for (i, &v) in vs.iter().enumerate() {
        if !v.is_finite() {
            return Err(SceneError::invalid(
                format!("{path}[{i}]"),
                "must be finite",
            ));
        }
    }
    Ok(())
}

fn at_least(path: &str, v: f64, min: f64) -> Result<()> {
    finite(path, v)?;
    if v >= min {
        Ok(())
    } else {
        Err(SceneError::invalid(path, format!("must be at least {min}")))
    }
}

fn positive(path: &str, v: f64) -> Result<()> {
    finite(path, v)?;
    if v > 0.0 {
        Ok(())
    } else {
        Err(SceneError::invalid(path, "must be greater than 0"))
    }
}

fn unit_quat(path: &str, q: [f64; 4]) -> Result<()> {
    finite_all(path, &q)?;
    if (quat_norm(q) - 1.0).abs() <= QUAT_TOLERANCE {
        Ok(())
    } else {
        Err(SceneError::invalid(
            path,
            "must be a unit quaternion [x, y, z, w]",
        ))
    }
}

fn unit_axis(path: &str, a: [f64; 3]) -> Result<()> {
    finite_all(path, &a)?;
    let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if (n - 1.0).abs() <= QUAT_TOLERANCE {
        Ok(())
    } else {
        Err(SceneError::invalid(path, "must be a unit vector"))
    }
}

fn range_pair(path: &str, r: [f64; 2]) -> Result<()> {
    finite_all(path, &r)?;
    if r[0] < r[1] {
        Ok(())
    } else {
        Err(SceneError::invalid(
            path,
            "lower bound must be below upper bound",
        ))
    }
}

/// MuJoCo's two `solref` formats: both positive (time constant, damping ratio) or
/// both not positive (direct stiffness and damping). A mix is what MuJoCo replaces
/// with the default and warns about; a scene refuses it.
fn solref_pair(path: &str, r: [f64; 2]) -> Result<()> {
    finite_all(path, &r)?;
    if (r[0] > 0.0) != (r[1] > 0.0) {
        return Err(SceneError::invalid(
            path,
            "solref must be positive in both entries (time constant, damping ratio) or not positive in both (direct stiffness, damping)",
        ));
    }
    Ok(())
}

/// A `solimp`: `[d_min, d_max, width, midpoint, power]` with the impedances and
/// the midpoint in `[0, 1]`, the width not negative and the power at least 1
/// (the ranges MuJoCo clamps to when it uses them).
fn solimp_five(path: &str, s: [f64; 5]) -> Result<()> {
    finite_all(path, &s)?;
    let unit = |v: f64| (0.0..=1.0).contains(&v);
    if !(unit(s[0]) && unit(s[1]) && unit(s[3])) {
        return Err(SceneError::invalid(
            path,
            "solimp impedances (entries 0 and 1) and midpoint (entry 3) must be in [0, 1]",
        ));
    }
    if s[2] < 0.0 {
        return Err(SceneError::invalid(
            path,
            "solimp width (entry 2) must not be negative",
        ));
    }
    if s[4] < 1.0 {
        return Err(SceneError::invalid(
            path,
            "solimp power (entry 4) must be at least 1",
        ));
    }
    Ok(())
}

fn unique_names<'a>(kind: &str, names: impl Iterator<Item = &'a str>) -> Result<()> {
    let mut seen = HashSet::new();
    for (i, name) in names.enumerate() {
        if !name.is_empty() && !seen.insert(name) {
            return Err(SceneError::invalid(
                format!("{kind}[{i}].name"),
                "name is used twice",
            ));
        }
    }
    Ok(())
}

fn id_in_range(path: &str, id: usize, len: usize, list: &str) -> Result<()> {
    if id < len {
        Ok(())
    } else {
        Err(SceneError::invalid(
            path,
            format!("refers to {list}[{id}] but there are only {len}"),
        ))
    }
}

impl Scene {
    /// Checks every invariant in the module notes of this crate: units are
    /// finite, quaternions are unit, ids are in range, joints respect the
    /// dof rules, shapes have positive sizes, meshes are well-formed, materials
    /// are physical, instances do not use the background segmentation id, and so
    /// on. Returns the first violation.
    pub fn validate(&self) -> Result<()> {
        if self.version != SCENE_VERSION {
            return Err(SceneError::invalid(
                "version",
                format!(
                    "scene version {} is not the supported {SCENE_VERSION}",
                    self.version
                ),
            ));
        }
        finite_all("gravity", &self.gravity)?;
        positive("timestep_s", self.timestep_s)?;
        at_least("options.tolerance", self.options.tolerance, 0.0)?;
        at_least("options.ls_tolerance", self.options.ls_tolerance, 0.0)?;
        positive("options.impratio", self.options.impratio)?;

        unique_names("materials", self.materials.iter().map(|m| m.name.as_str()))?;
        for (i, m) in self.materials.iter().enumerate() {
            m.validate(&format!("materials[{i}]"))?;
        }

        self.validate_meshes()?;
        self.validate_bodies()?;
        self.validate_joints()?;
        self.validate_geoms()?;
        self.validate_contact_excludes()?;
        self.validate_instances()?;
        self.validate_cameras()?;
        self.validate_actuators()?;
        self.validate_tendons()?;
        Ok(())
    }

    fn validate_meshes(&self) -> Result<()> {
        unique_names("meshes", self.meshes.iter().map(|m| m.name.as_str()))?;
        for (i, mesh) in self.meshes.iter().enumerate() {
            let path = format!("meshes[{i}]");
            if mesh.vertices.len() < 4 {
                return Err(SceneError::invalid(
                    format!("{path}.vertices"),
                    "a mesh needs at least 4 vertices",
                ));
            }
            if mesh.triangles.is_empty() {
                return Err(SceneError::invalid(
                    format!("{path}.triangles"),
                    "a mesh needs at least 1 triangle",
                ));
            }
            for (v, p) in mesh.vertices.iter().enumerate() {
                if !p.iter().all(|c| c.is_finite()) {
                    return Err(SceneError::invalid(
                        format!("{path}.vertices[{v}]"),
                        "must be finite",
                    ));
                }
            }
            let n = mesh.vertices.len();
            for (t, tri) in mesh.triangles.iter().enumerate() {
                if tri.iter().any(|&idx| idx as usize >= n) {
                    return Err(SceneError::invalid(
                        format!("{path}.triangles[{t}]"),
                        format!("vertex index out of range (the mesh has {n} vertices)"),
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_bodies(&self) -> Result<()> {
        unique_names("bodies", self.bodies.iter().map(|b| b.name.as_str()))?;
        for (i, body) in self.bodies.iter().enumerate() {
            let path = format!("bodies[{i}]");
            if let Some(parent) = body.parent
                && parent.index() >= i
            {
                return Err(SceneError::invalid(
                    format!("{path}.parent"),
                    "a parent must be listed before its children",
                ));
            }
            finite_all(&format!("{path}.pos"), &body.pos)?;
            unit_quat(&format!("{path}.quat"), body.quat)?;
            if let Some(inertial) = &body.inertial {
                let p = format!("{path}.inertial");
                at_least(&format!("{p}.mass_kg"), inertial.mass_kg, 0.0)?;
                finite_all(&format!("{p}.com"), &inertial.com)?;
                finite_all(&format!("{p}.diag_inertia"), &inertial.diag_inertia)?;
                let d = inertial.diag_inertia;
                for (j, &moment) in d.iter().enumerate() {
                    if moment < 0.0 {
                        return Err(SceneError::invalid(
                            format!("{p}.diag_inertia[{j}]"),
                            "a principal moment cannot be negative",
                        ));
                    }
                }
                let slack = 1e-9 * (d[0] + d[1] + d[2]).max(1e-30);
                if d[0] + d[1] + slack < d[2]
                    || d[0] + d[2] + slack < d[1]
                    || d[1] + d[2] + slack < d[0]
                {
                    return Err(SceneError::invalid(
                        format!("{p}.diag_inertia"),
                        "principal moments must satisfy a + b >= c",
                    ));
                }
                unit_quat(&format!("{p}.inertia_quat"), inertial.inertia_quat)?;
            }
        }
        Ok(())
    }

    fn validate_joints(&self) -> Result<()> {
        unique_names("joints", self.joints.iter().map(|j| j.name.as_str()))?;
        let mut previous_body = 0usize;
        let mut dofs_in_body = vec![0usize; self.bodies.len()];
        let mut joints_in_body = vec![0usize; self.bodies.len()];
        let mut ball_seen = vec![false; self.bodies.len()];
        for (i, joint) in self.joints.iter().enumerate() {
            let path = format!("joints[{i}]");
            let b = joint.body.index();
            id_in_range(&format!("{path}.body"), b, self.bodies.len(), "bodies")?;
            if b < previous_body {
                return Err(SceneError::invalid(
                    format!("{path}.body"),
                    "joints must be listed in non-decreasing body order",
                ));
            }
            previous_body = b;
            finite_all(&format!("{path}.pos"), &joint.pos)?;
            match joint.kind {
                JointKind::Free => {
                    if self.bodies[b].parent.is_some() {
                        return Err(SceneError::invalid(
                            format!("{path}.kind"),
                            "a free joint can only be on a child of the world",
                        ));
                    }
                    if joint.pos != [0.0; 3] {
                        return Err(SceneError::invalid(
                            format!("{path}.pos"),
                            "a free joint's anchor is the body origin (must be zero)",
                        ));
                    }
                    if joint.range.is_some() {
                        return Err(SceneError::invalid(
                            format!("{path}.range"),
                            "a free joint cannot be limited",
                        ));
                    }
                }
                JointKind::Ball => {
                    if let Some(r) = joint.range {
                        finite_all(&format!("{path}.range"), &r)?;
                        if r[0] != 0.0 || r[1] <= 0.0 {
                            return Err(SceneError::invalid(
                                format!("{path}.range"),
                                "a ball joint's range is [0, max_angle] with max_angle > 0",
                            ));
                        }
                    }
                }
                JointKind::Hinge { axis } | JointKind::Slide { axis } => {
                    unit_axis(&format!("{path}.kind.axis"), axis)?;
                    if let Some(r) = joint.range {
                        range_pair(&format!("{path}.range"), r)?;
                    }
                }
            }
            at_least(&format!("{path}.stiffness"), joint.stiffness, 0.0)?;
            at_least(&format!("{path}.damping"), joint.damping, 0.0)?;
            at_least(&format!("{path}.armature"), joint.armature, 0.0)?;
            at_least(&format!("{path}.frictionloss"), joint.frictionloss, 0.0)?;
            solref_pair(&format!("{path}.solref_limit"), joint.solref_limit)?;
            solimp_five(&format!("{path}.solimp_limit"), joint.solimp_limit)?;
            solref_pair(&format!("{path}.solref_friction"), joint.solref_friction)?;
            solimp_five(&format!("{path}.solimp_friction"), joint.solimp_friction)?;
            at_least(&format!("{path}.margin"), joint.margin, 0.0)?;

            // MuJoCo's per-body rules (mjCBody::Compile): at most 6 dofs, a free
            // joint stands alone, and no rotation follows a ball joint.
            joints_in_body[b] += 1;
            dofs_in_body[b] += joint.kind.nv();
            if dofs_in_body[b] > 6 {
                return Err(SceneError::invalid(
                    format!("{path}.body"),
                    "a body cannot have more than 6 degrees of freedom",
                ));
            }
            if matches!(joint.kind, JointKind::Free) && joints_in_body[b] > 1 {
                return Err(SceneError::invalid(
                    format!("{path}.kind"),
                    "a free joint must be the only joint of its body",
                ));
            }
            if joints_in_body[b] > 1
                && self.joints[..i]
                    .iter()
                    .any(|j| j.body == joint.body && matches!(j.kind, JointKind::Free))
            {
                return Err(SceneError::invalid(
                    format!("{path}.kind"),
                    "a free joint must be the only joint of its body",
                ));
            }
            if ball_seen[b] && matches!(joint.kind, JointKind::Ball | JointKind::Hinge { .. }) {
                return Err(SceneError::invalid(
                    format!("{path}.kind"),
                    "a ball joint cannot be followed by a rotation in the same body",
                ));
            }
            if matches!(joint.kind, JointKind::Ball) {
                ball_seen[b] = true;
            }
        }
        Ok(())
    }

    /// Whether a body, or any ancestor of it, has a joint.
    pub(crate) fn body_is_moving(&self) -> Vec<bool> {
        let mut has_joint = vec![false; self.bodies.len()];
        for joint in &self.joints {
            if let Some(slot) = has_joint.get_mut(joint.body.index()) {
                *slot = true;
            }
        }
        let mut moving = vec![false; self.bodies.len()];
        for (i, body) in self.bodies.iter().enumerate() {
            let parent_moving = body
                .parent
                .and_then(|p| moving.get(p.index()).copied())
                .unwrap_or(false);
            moving[i] = has_joint[i] || parent_moving;
        }
        moving
    }

    fn validate_geoms(&self) -> Result<()> {
        unique_names("geoms", self.geoms.iter().map(|g| g.name.as_str()))?;
        let moving = self.body_is_moving();
        for (i, geom) in self.geoms.iter().enumerate() {
            let path = format!("geoms[{i}]");
            if let Some(b) = geom.body {
                id_in_range(
                    &format!("{path}.body"),
                    b.index(),
                    self.bodies.len(),
                    "bodies",
                )?;
            }
            finite_all(&format!("{path}.pos"), &geom.pos)?;
            unit_quat(&format!("{path}.quat"), geom.quat)?;
            id_in_range(
                &format!("{path}.material"),
                geom.material.index(),
                self.materials.len(),
                "materials",
            )?;
            let sp = format!("{path}.shape");
            match geom.shape {
                Shape::Sphere { r } => positive(&format!("{sp}.r"), r)?,
                Shape::Capsule { r, half_len } | Shape::Cylinder { r, half_len } => {
                    positive(&format!("{sp}.r"), r)?;
                    positive(&format!("{sp}.half_len"), half_len)?;
                }
                Shape::Box { half } => {
                    for (j, &h) in half.iter().enumerate() {
                        positive(&format!("{sp}.half[{j}]"), h)?;
                    }
                }
                Shape::Ellipsoid { radii } => {
                    for (j, &h) in radii.iter().enumerate() {
                        positive(&format!("{sp}.radii[{j}]"), h)?;
                    }
                }
                Shape::Plane { size } => {
                    at_least(&format!("{sp}.size[0]"), size[0], 0.0)?;
                    at_least(&format!("{sp}.size[1]"), size[1], 0.0)?;
                    positive(&format!("{sp}.size[2]"), size[2])?;
                    if let Some(b) = geom.body
                        && moving[b.index()]
                    {
                        return Err(SceneError::invalid(
                            format!("{path}.body"),
                            "a plane can only belong to the world or a body with no joint in its chain",
                        ));
                    }
                }
                Shape::Mesh { mesh } => id_in_range(
                    &format!("{sp}.mesh"),
                    mesh.index(),
                    self.meshes.len(),
                    "meshes",
                )?,
            }
            if !matches!(geom.condim, 1 | 3 | 4 | 6) {
                return Err(SceneError::invalid(
                    format!("{path}.condim"),
                    "must be 1, 3, 4 or 6",
                ));
            }
            for (j, &f) in geom.friction.iter().enumerate() {
                at_least(&format!("{path}.friction[{j}]"), f, 0.0)?;
            }
            at_least(&format!("{path}.density"), geom.density, 0.0)?;
            solref_pair(&format!("{path}.solref"), geom.solref)?;
            solimp_five(&format!("{path}.solimp"), geom.solimp)?;
            at_least(&format!("{path}.solmix"), geom.solmix, 0.0)?;
            at_least(&format!("{path}.margin"), geom.margin, 0.0)?;
            at_least(&format!("{path}.gap"), geom.gap, 0.0)?;
        }
        Ok(())
    }

    fn validate_contact_excludes(&self) -> Result<()> {
        for (i, ex) in self.contact_excludes.iter().enumerate() {
            let path = format!("contact_excludes[{i}]");
            for (side, body) in [("body1", ex.body1), ("body2", ex.body2)] {
                if let Some(b) = body {
                    id_in_range(
                        &format!("{path}.{side}"),
                        b.index(),
                        self.bodies.len(),
                        "bodies",
                    )?;
                }
            }
            if ex.body1 == ex.body2 {
                return Err(SceneError::invalid(
                    format!("{path}.body2"),
                    "an exclusion names two different bodies (the world counts as one)",
                ));
            }
        }
        Ok(())
    }

    fn validate_instances(&self) -> Result<()> {
        for (i, inst) in self.instances.iter().enumerate() {
            let path = format!("instances[{i}]");
            self.validate_instance(&path, inst)?;
        }
        Ok(())
    }

    fn validate_instance(&self, path: &str, inst: &Instance) -> Result<()> {
        if let Some(b) = inst.body {
            id_in_range(
                &format!("{path}.body"),
                b.index(),
                self.bodies.len(),
                "bodies",
            )?;
        }
        match inst.mesh_or_geom {
            ShapeRef::Mesh(m) => id_in_range(
                &format!("{path}.mesh_or_geom"),
                m.index(),
                self.meshes.len(),
                "meshes",
            )?,
            ShapeRef::Geom(g) => {
                id_in_range(
                    &format!("{path}.mesh_or_geom"),
                    g.index(),
                    self.geoms.len(),
                    "geoms",
                )?;
                if self.geoms[g.index()].body != inst.body {
                    return Err(SceneError::invalid(
                        format!("{path}.body"),
                        "an instance of a geom must be attached to the geom's own body",
                    ));
                }
            }
        }
        id_in_range(
            &format!("{path}.material"),
            inst.material.index(),
            self.materials.len(),
            "materials",
        )?;
        finite_all(&format!("{path}.local_pos"), &inst.local_pos)?;
        unit_quat(&format!("{path}.local_quat"), inst.local_quat)?;
        if inst.seg_id == 0 {
            return Err(SceneError::invalid(
                format!("{path}.seg_id"),
                "segmentation id 0 is reserved for the background",
            ));
        }
        Ok(())
    }

    fn validate_cameras(&self) -> Result<()> {
        unique_names("cameras", self.cameras.iter().map(|c| c.name.as_str()))?;
        for (i, cam) in self.cameras.iter().enumerate() {
            self.validate_camera(&format!("cameras[{i}]"), cam)?;
        }
        Ok(())
    }

    fn validate_camera(&self, path: &str, cam: &Camera) -> Result<()> {
        match cam.mount {
            CameraMount::World { pos, quat } => {
                finite_all(&format!("{path}.mount.pos"), &pos)?;
                unit_quat(&format!("{path}.mount.quat"), quat)?;
            }
            CameraMount::Body {
                body,
                local_pos,
                local_quat,
            } => {
                id_in_range(
                    &format!("{path}.mount.body"),
                    body.index(),
                    self.bodies.len(),
                    "bodies",
                )?;
                finite_all(&format!("{path}.mount.local_pos"), &local_pos)?;
                unit_quat(&format!("{path}.mount.local_quat"), local_quat)?;
            }
        }
        for (name, v) in [("fx", cam.fx), ("fy", cam.fy)] {
            positive(&format!("{path}.{name}"), f64::from(v))?;
        }
        for (name, v) in [("cx", cam.cx), ("cy", cam.cy)] {
            finite(&format!("{path}.{name}"), f64::from(v))?;
        }
        positive(&format!("{path}.near"), f64::from(cam.near))?;
        finite(&format!("{path}.far"), f64::from(cam.far))?;
        if cam.far <= cam.near {
            return Err(SceneError::invalid(
                format!("{path}.far"),
                "must be greater than near",
            ));
        }
        if cam.width == 0 || cam.height == 0 {
            return Err(SceneError::invalid(
                format!("{path}.width"),
                "image width and height must be positive",
            ));
        }
        Ok(())
    }

    fn validate_actuators(&self) -> Result<()> {
        unique_names("actuators", self.actuators.iter().map(|a| a.name.as_str()))?;
        for (i, act) in self.actuators.iter().enumerate() {
            let path = format!("actuators[{i}]");
            let j = act.joint.index();
            id_in_range(&format!("{path}.joint"), j, self.joints.len(), "joints")?;
            if !matches!(
                self.joints[j].kind,
                JointKind::Hinge { .. } | JointKind::Slide { .. }
            ) {
                return Err(SceneError::invalid(
                    format!("{path}.joint"),
                    "only a hinge or a slide joint can be driven",
                ));
            }
            match act.kind {
                ActuatorKind::Motor { gear, ctrlrange } => {
                    finite(&format!("{path}.kind.gear"), gear)?;
                    if let Some(r) = ctrlrange {
                        range_pair(&format!("{path}.kind.ctrlrange"), r)?;
                    }
                }
                ActuatorKind::Position {
                    kp,
                    gear,
                    ctrlrange,
                } => {
                    at_least(&format!("{path}.kind.kp"), kp, 0.0)?;
                    finite(&format!("{path}.kind.gear"), gear)?;
                    if let Some(r) = ctrlrange {
                        range_pair(&format!("{path}.kind.ctrlrange"), r)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_tendons(&self) -> Result<()> {
        unique_names("tendons", self.tendons.iter().map(|t| t.name.as_str()))?;
        for (i, tendon) in self.tendons.iter().enumerate() {
            let path = format!("tendons[{i}]");
            if tendon.joints.is_empty() {
                return Err(SceneError::invalid(
                    format!("{path}.joints"),
                    "a tendon needs at least one joint",
                ));
            }
            for (k, term) in tendon.joints.iter().enumerate() {
                let p = format!("{path}.joints[{k}]");
                let j = term.joint.index();
                id_in_range(&format!("{p}.joint"), j, self.joints.len(), "joints")?;
                if !matches!(
                    self.joints[j].kind,
                    JointKind::Hinge { .. } | JointKind::Slide { .. }
                ) {
                    return Err(SceneError::invalid(
                        format!("{p}.joint"),
                        "a fixed tendon couples hinge and slide joints",
                    ));
                }
                finite(&format!("{p}.coef"), term.coef)?;
            }
            if let Some(r) = tendon.range {
                range_pair(&format!("{path}.range"), r)?;
            }
            at_least(&format!("{path}.stiffness"), tendon.stiffness, 0.0)?;
            at_least(&format!("{path}.damping"), tendon.damping, 0.0)?;
            at_least(&format!("{path}.armature"), tendon.armature, 0.0)?;
            at_least(&format!("{path}.frictionloss"), tendon.frictionloss, 0.0)?;
            solref_pair(&format!("{path}.solref_limit"), tendon.solref_limit)?;
            solimp_five(&format!("{path}.solimp_limit"), tendon.solimp_limit)?;
            solref_pair(&format!("{path}.solref_friction"), tendon.solref_friction)?;
            solimp_five(&format!("{path}.solimp_friction"), tendon.solimp_friction)?;
            at_least(&format!("{path}.margin"), tendon.margin, 0.0)?;
        }
        Ok(())
    }
}
