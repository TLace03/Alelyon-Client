//! Ground truth: the hidden causes behind a tick, for evaluation only.
//!
//! **EVALUATION ONLY. NEVER AN INPUT TO SINAI.** [`GroundTruth`] is a separate
//! type from [`crate::Bundle`] on purpose, and no field of a bundle can hold it,
//! directly or through any type a bundle holds. A model that is shown the
//! temperature at a point, the progress of a reaction, a contact event, the
//! identity of an occluded object or a segmentation frame (whose per-instance
//! ids persist through occlusion, so they name the very objects the
//! object-permanence test asks the model to track) has been handed the answer to
//! the question the senses are meant to let it answer: whether it infers hidden
//! causes from what it sees, hears, smells and feels. Training on it, or letting it reach a
//! model's context at evaluation time, would make every cross-sense result
//! meaningless. It exists so a test can compare what the model inferred with what
//! was true.
//!
//! The `evaluation_only` integration test pins this at compile time: it
//! destructures every type a `Bundle` is made of with no wildcard, so adding a
//! field (of any type, including `GroundTruth`) stops compiling until a person
//! reads this note and the test.
//!
//! Invariants:
//! - Units are SI: kelvin, newtons, metres.
//! - Names within one list are unique and not empty: a name is the key of a
//!   named point (temperatures) or of a reaction.
//! - Reaction progress is a dimensionless fraction in `0..=1`: 0 not begun, 1
//!   complete (for example `maillard`, `water_loss`).
//! - A contact is between two different entities and pushes with a force that
//!   is not negative.
//! - An entity appears at most once in the occlusion list.
//! - Each segmentation frame ([`SegFrame`]) has a finite capture time and is a
//!   valid v0 segmentation frame: `u16` x 1, 448 x 448
//!   ([`FrameRef::v0_segmentation`]). Its ids are stable for a whole episode (an
//!   object keeps its id through occlusion, motion and across ticks) and 0 is
//!   background. That is a property the simulator must keep: this crate checks
//!   one frame's format and cannot check ids across ticks.
//! - There is at most one true `SegFrame` per camera in a record, and at most
//!   one `ShownSegFrame` per camera: with two, the evaluator could not tell which
//!   frame the model saw.
//! - `segmentation` ALWAYS describes the TRUE state. Under a sight perturbation
//!   the segmentation of the frame actually shown goes in `shown_segmentation`
//!   ([`ShownSegFrame`]), BESIDE the true one and never instead of it: every
//!   shown frame's camera has a true `SegFrame` in the same record, and its
//!   perturbation is a sight one.
//! - A segmentation id means an entity only through the per-episode
//!   [`SegTable`] (`seg_table`), written at reset and fixed for the episode. Id 0
//!   is background and never in the table; seg ids are unique and so are entity
//!   ids. The ids are assigned, NOT derived by any formula from the entity, its
//!   kind or its position, so nothing about identity can be decoded from an id
//!   without the table, and the table lives only in this evaluation channel. The
//!   table's `episode_id` is the record's `identity.episode_id`.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::check::{all_finite_f32, finite_f32, finite_f64};
use crate::{Capture, ContractError, FrameRef, FrameSemantic, Identity, Perturbation, Sense};

/// The identity of a simulated entity (a body, an object, a surface), assigned
/// by the simulator and stable within an episode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EntityId(pub u64);

/// A value with a name.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NamedValue {
    /// The named point or reaction; not empty, unique within its list.
    pub name: String,
    /// Kelvin for a temperature; a fraction in `0..=1` for reaction progress.
    pub value: f64,
}

/// A contact between two entities this tick.
///
/// Invariants: `a != b`; `normal_force_n` is finite and not negative;
/// `position` is finite.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContactEvent {
    /// One entity in contact.
    pub a: EntityId,
    /// The other.
    pub b: EntityId,
    /// Total normal force at the contact, newtons.
    pub normal_force_n: f32,
    /// The contact point, metres, world frame.
    pub position: [f32; 3],
}

/// One camera's segmentation frame for a tick. **Evaluation only.**
///
/// A `u16` instance id per pixel, 448 x 448 in v0
/// ([`FrameRef::v0_segmentation`]), pixel-aligned with the RGB frame of the same
/// camera and tick. Ids are stable for a whole episode, so an object keeps its
/// id through occlusion, and 0 is background. That is a property the simulator
/// must keep; this crate cannot check it across ticks. Because the ids persist
/// through occlusion they would hand a model the object identities that the
/// object-permanence test measures, which is why segmentation is part of
/// [`GroundTruth`] and not of a bundle. What an id means is in the episode's
/// [`SegTable`].
///
/// Invariants (checked by [`SegFrame::validate`]): the capture time is finite
/// and the frame is a valid v0 segmentation frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SegFrame {
    /// The index of the camera that rendered the frame, in the simulator's
    /// camera order.
    pub camera: u32,
    /// The frame and the simulated time it was captured at.
    pub frame: Capture<FrameRef>,
}

impl SegFrame {
    /// Checks the capture time and that the frame is the v0 segmentation format.
    pub fn validate(&self) -> Result<(), ContractError> {
        self.frame
            .validate_time()
            .and_then(|()| self.frame.value.validate_v0(FrameSemantic::Segmentation))
    }
}

/// What a segmentation id means: the per-episode table from seg id to entity.
/// **Evaluation only.**
///
/// Written at reset and fixed for the episode. The ids are assigned, NOT derived
/// by any formula from the entity, its kind or its position, so nothing about an
/// object's identity can be decoded from a seg id without this table. The table
/// lives only in the evaluation channel ([`GroundTruth::seg_table`]); a model
/// never sees it, because with it the ids name the objects.
///
/// JSON: `{ "episode_id": u64, "entries": [[seg_id, entity_id], ...] }`.
///
/// Invariants (checked by [`SegTable::validate`]): no seg id is 0 (0 is
/// background, never an entity), seg ids are unique, and entity ids are unique.
/// That the table is fixed for the whole episode, and that the ids are not
/// derived by a formula, are properties the simulator must keep: this crate
/// checks one table, not the table of one tick against another's.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SegTable {
    /// The episode this table belongs to (an [`Identity::episode_id`]). A
    /// [`GroundTruth`] that carries the table refuses one whose `episode_id` is
    /// not its own `identity.episode_id`.
    pub episode_id: u64,
    /// Each segmentation id and the entity it names, in no particular order.
    pub entries: Vec<(u16, EntityId)>,
}

impl SegTable {
    /// Checks that no seg id is 0, and that seg ids and entity ids are unique.
    pub fn validate(&self) -> Result<(), ContractError> {
        let mut seg_ids = BTreeSet::new();
        let mut entities = BTreeSet::new();
        for (seg_id, entity) in &self.entries {
            if *seg_id == 0 {
                return Err(ContractError::OutOfRange {
                    field: "seg_table.seg_id",
                    reason: "seg id 0 is background and never an entity",
                });
            }
            if !seg_ids.insert(*seg_id) {
                return Err(ContractError::DuplicateId {
                    field: "seg_table.seg_id",
                    id: u64::from(*seg_id),
                });
            }
            if !entities.insert(*entity) {
                return Err(ContractError::DuplicateId {
                    field: "seg_table.entity_id",
                    id: entity.0,
                });
            }
        }
        Ok(())
    }
}

/// The segmentation of the frame actually SHOWN while a sight perturbation is
/// active, with the perturbation that was active. **Evaluation only.**
///
/// It goes in [`GroundTruth::shown_segmentation`], BESIDE the true
/// segmentation of the same camera ([`GroundTruth::segmentation`], which always
/// describes the true state) and never instead of it, so an evaluation can
/// compare what was shown with what was true.
///
/// Invariants (checked by [`GroundTruth::validate`], which also checks the
/// `perturbation` and the `frame` themselves): `perturbation` is one on
/// [`Sense::Sight`], the record holds a true [`SegFrame`] for `frame.camera`, and
/// it holds at most one shown frame for that camera (with two, the evaluator
/// could not tell which one the model saw).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShownSegFrame {
    /// The sight perturbation that was active when the frame was shown.
    pub perturbation: Perturbation,
    /// The segmentation of the frame shown under it.
    pub frame: SegFrame,
}

impl ShownSegFrame {
    /// Checks the perturbation, that it is a sight one, the frame, and that
    /// `true_cameras` (the cameras with a true segmentation frame in the record)
    /// contains the frame's camera.
    fn validate(&self, true_cameras: &BTreeSet<u32>) -> Result<(), ContractError> {
        self.perturbation.validate()?;
        if self.perturbation.sense() != Sense::Sight {
            return Err(ContractError::OutOfRange {
                field: "perturbation.sense",
                reason: "a shown segmentation frame is under a sight perturbation",
            });
        }
        self.frame.validate()?;
        if !true_cameras.contains(&self.frame.camera) {
            return Err(ContractError::ShownWithoutTrueSegmentation {
                camera: self.frame.camera,
            });
        }
        Ok(())
    }
}

/// What was true in the world at one tick. **Evaluation only.**
///
/// See the module note: this is never part of a bundle and never an input.
///
/// Invariants (checked by [`GroundTruth::validate`]): the identity is valid and
/// the rules of the module hold for every list. A failure in a segmentation
/// frame is wrapped as [`ContractError::Channel`] with the channel
/// `"segmentation"` (or `"shown_segmentation"` for a shown frame).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GroundTruth {
    /// The tick this describes; matches the bundle it is evaluated against.
    pub identity: Identity,
    /// Temperature at named points, kelvin.
    pub temperatures_k: Vec<NamedValue>,
    /// Progress of named reactions, `0..=1`.
    pub reaction_progress: Vec<NamedValue>,
    /// Contacts this tick.
    pub contacts: Vec<ContactEvent>,
    /// Each entity whose identity is known through occlusion, and whether it is
    /// visible to the camera this tick.
    pub object_ids_through_occlusion: Vec<(EntityId, bool)>,
    /// The TRUE segmentation frames this tick, at most one per camera. It
    /// always describes the true state, perturbed or not. Evaluation only:
    /// per-instance ids that persist through occlusion are the answer to the
    /// object-permanence test, so they are never part of a bundle.
    pub segmentation: Vec<SegFrame>,
    /// What each segmentation id means, for the episode: `Some` in the record
    /// written at reset, and fixed for the episode. `None` where a record does
    /// not repeat it (read it against the table written at reset). The key is
    /// always present in the JSON, as `null` for `None`.
    #[serde(deserialize_with = "crate::bundle::present")]
    pub seg_table: Option<SegTable>,
    /// The segmentation of the frames actually shown while a sight perturbation
    /// is active, each with its perturbation, at most one per camera. Beside
    /// `segmentation`, never instead of it; empty when no sight perturbation is
    /// active.
    pub shown_segmentation: Vec<ShownSegFrame>,
}

impl GroundTruth {
    /// A record with no causes listed yet.
    pub fn empty(identity: Identity) -> Self {
        Self {
            identity,
            temperatures_k: Vec::new(),
            reaction_progress: Vec::new(),
            contacts: Vec::new(),
            object_ids_through_occlusion: Vec::new(),
            segmentation: Vec::new(),
            seg_table: None,
            shown_segmentation: Vec::new(),
        }
    }

    /// Checks the identity and every list.
    pub fn validate(&self) -> Result<(), ContractError> {
        self.identity.validate()?;
        named_values(
            "temperatures_k",
            &self.temperatures_k,
            |value| value > 0.0,
            "absolute temperature must be above 0 K",
        )?;
        named_values(
            "reaction_progress",
            &self.reaction_progress,
            |value| (0.0..=1.0).contains(&value),
            "reaction progress is a fraction in 0..=1",
        )?;
        for contact in &self.contacts {
            if contact.a == contact.b {
                return Err(ContractError::OutOfRange {
                    field: "contacts",
                    reason: "a contact is between two different entities",
                });
            }
            finite_f32("contacts.normal_force_n", contact.normal_force_n)?;
            if contact.normal_force_n < 0.0 {
                return Err(ContractError::OutOfRange {
                    field: "contacts.normal_force_n",
                    reason: "a normal force cannot be negative",
                });
            }
            all_finite_f32("contacts.position", &contact.position)?;
        }
        let mut seen = BTreeSet::new();
        for (entity, _visible) in &self.object_ids_through_occlusion {
            if !seen.insert(*entity) {
                return Err(ContractError::DuplicateId {
                    field: "object_ids_through_occlusion",
                    id: entity.0,
                });
            }
        }
        let mut cameras = BTreeSet::new();
        for seg in &self.segmentation {
            seg.validate().map_err(|e| e.in_channel("segmentation"))?;
            if !cameras.insert(seg.camera) {
                return Err(ContractError::DuplicateId {
                    field: "camera",
                    id: u64::from(seg.camera),
                }
                .in_channel("segmentation"));
            }
        }
        if let Some(table) = &self.seg_table {
            table.validate()?;
            if table.episode_id != self.identity.episode_id {
                return Err(ContractError::EpisodeMismatch {
                    field: "seg_table.episode_id",
                    expected: self.identity.episode_id,
                    found: table.episode_id,
                });
            }
        }
        let mut shown_cameras = BTreeSet::new();
        for shown in &self.shown_segmentation {
            shown
                .validate(&cameras)
                .map_err(|e| e.in_channel("shown_segmentation"))?;
            if !shown_cameras.insert(shown.frame.camera) {
                return Err(ContractError::DuplicateId {
                    field: "camera",
                    id: u64::from(shown.frame.camera),
                }
                .in_channel("shown_segmentation"));
            }
        }
        Ok(())
    }
}

/// Names are non-empty and unique; values are finite and satisfy `allowed`.
fn named_values(
    field: &'static str,
    values: &[NamedValue],
    allowed: impl Fn(f64) -> bool,
    reason: &'static str,
) -> Result<(), ContractError> {
    let mut names = BTreeSet::new();
    for entry in values {
        if entry.name.is_empty() {
            return Err(ContractError::Empty { field });
        }
        if !names.insert(entry.name.as_str()) {
            return Err(ContractError::DuplicateName { field });
        }
        finite_f64(field, entry.value)?;
        if !allowed(entry.value) {
            return Err(ContractError::OutOfRange { field, reason });
        }
    }
    Ok(())
}
