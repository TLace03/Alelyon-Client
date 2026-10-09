//! Typed indices into the scene's lists.
//!
//! Invariants:
//! - An id is an index into the list of its kind in the same [`crate::Scene`]
//!   (`BodyId(3)` is `scene.bodies[3]`). It is only meaningful for the scene it
//!   came from; `Scene::validate` checks that every id is in range.
//! - The world is not a body: it has no id. A field that can name the world is
//!   an `Option<BodyId>` and `None` means the world (a static, immovable frame).
//! - JSON writes an id as the bare integer.

use serde::{Deserialize, Serialize};

macro_rules! id_type {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub u32);

        impl $name {
            /// The index as a `usize`, for indexing the scene's list.
            pub fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

id_type!(
    /// Index into `Scene::bodies`.
    BodyId
);
id_type!(
    /// Index into `Scene::joints`.
    JointId
);
id_type!(
    /// Index into `Scene::geoms`.
    GeomId
);
id_type!(
    /// Index into `Scene::meshes`.
    MeshId
);
id_type!(
    /// Index into `Scene::materials`.
    MaterialId
);
