//! The one error every `validate` in this crate returns.
//!
//! A refusal names the field or channel it concerns and the rule it broke, so a
//! producer can fix the value and a consumer can log the reason. The variants
//! are data, not sentences, so a test (or the Sinai side's loader) can match on
//! the rule that fired; `Display` writes the sentence.
//!
//! Invariants:
//! - A refusal never echoes a free-text value it was given (a scene id, a
//!   species name). It echoes numbers and the static name of the field.
//! - [`ContractError::Channel`] only ever wraps a different variant, one level
//!   deep, so [`ContractError::root`] always reaches the rule that fired.

use std::fmt;

/// Why a value broke the contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContractError {
    /// The value was written for another schema version than [`crate::SCHEMA_VERSION`].
    SchemaVersion {
        /// The version the value carried.
        found: u32,
    },
    /// A number was NaN or infinite. JSON cannot carry either (`serde_json`
    /// writes `null`, which reads back as an error), so they are refused first.
    NonFinite {
        /// The field that held it.
        field: &'static str,
    },
    /// A number fell outside what the field allows.
    OutOfRange {
        /// The field that held it.
        field: &'static str,
        /// The rule, in a few words.
        reason: &'static str,
    },
    /// A dimension, count or collection that must not be empty was.
    Empty {
        /// The field that was empty.
        field: &'static str,
    },
    /// A collection's length disagreed with the length its shape or table fixes.
    LengthMismatch {
        /// The field whose length was wrong.
        field: &'static str,
        /// The length the shape or table requires.
        expected: u64,
        /// The length found.
        found: u64,
    },
    /// A frame's row stride is smaller than one row of pixels.
    Stride {
        /// Bytes in one tightly packed row.
        required_bytes: u64,
        /// The `row_stride_bytes` found.
        found_bytes: u64,
    },
    /// An offset or stride is not a multiple of the element size.
    Misaligned {
        /// The field that was misaligned.
        field: &'static str,
        /// The required alignment, in bytes.
        alignment: u64,
        /// The value found.
        found: u64,
    },
    /// A frame is not in the format contract v0 fixes for its channel.
    Format {
        /// What differs, in a few words.
        reason: &'static str,
    },
    /// A species table lists one PubChem CID twice.
    DuplicateCid {
        /// The repeated CID.
        pubchem_cid: u64,
    },
    /// Two entries of one list carry the same numeric id.
    DuplicateId {
        /// The list that repeats an id.
        field: &'static str,
        /// The repeated id.
        id: u64,
    },
    /// A shown segmentation frame has no true segmentation frame for its camera
    /// in the same ground-truth record. The shown frame goes beside the true
    /// one, never instead of it.
    ShownWithoutTrueSegmentation {
        /// The camera index the shown frame is for.
        camera: u32,
    },
    /// A value that belongs to one episode names another one than the record it
    /// is in.
    EpisodeMismatch {
        /// The field that holds the episode id.
        field: &'static str,
        /// The episode the record's identity names.
        expected: u64,
        /// The episode the field names.
        found: u64,
    },
    /// Two entries of one list carry the same name.
    DuplicateName {
        /// The list that repeats a name.
        field: &'static str,
    },
    /// A concentration vector has entries but its species table has none.
    EmptyTableForNonEmptyVector,
    /// A vector was written against another version of the species table.
    TableVersion {
        /// The `table_version` the vector carries.
        vector: u32,
        /// The version of the table it was checked against.
        table: u32,
    },
    /// Audio chunk `n + 1` does not start where chunk `n` ended.
    AudioDiscontinuity {
        /// The `first_sample_index` that continuity requires.
        expected_first_sample_index: u64,
        /// The `first_sample_index` found.
        found_first_sample_index: u64,
    },
    /// An identifier is empty, too long, or uses characters outside
    /// `[a-z0-9_.-]` (or starts with `.` or `-`).
    BadIdentifier {
        /// The field that held it.
        field: &'static str,
    },
    /// Depth was sent without the RGB frame it belongs to.
    ExtraChannelWithoutRgb {
        /// The extra channel that was present.
        channel: &'static str,
    },
    /// The text was not valid JSON for the type asked for.
    Json {
        /// The parser's message.
        reason: String,
    },
    /// A rule broke inside one channel of a bundle, or inside the `segmentation`
    /// or `shown_segmentation` frames of a ground-truth record.
    Channel {
        /// The channel.
        channel: &'static str,
        /// The rule that fired inside it.
        cause: Box<ContractError>,
    },
}

impl ContractError {
    /// The error with its channel wrappers removed: the rule that fired.
    pub fn root(&self) -> &ContractError {
        match self {
            ContractError::Channel { cause, .. } => cause.root(),
            other => other,
        }
    }

    /// Names the channel this error happened in.
    pub(crate) fn in_channel(self, channel: &'static str) -> ContractError {
        ContractError::Channel {
            channel,
            cause: Box::new(self),
        }
    }
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContractError::SchemaVersion { found } => {
                write!(
                    f,
                    "schema version {found} is not the supported version {}",
                    crate::SCHEMA_VERSION
                )
            }
            ContractError::NonFinite { field } => {
                write!(f, "{field} is NaN or infinite")
            }
            ContractError::OutOfRange { field, reason } => {
                write!(f, "{field} is out of range: {reason}")
            }
            ContractError::Empty { field } => write!(f, "{field} must not be empty"),
            ContractError::LengthMismatch {
                field,
                expected,
                found,
            } => write!(
                f,
                "{field} has {found} entries where {expected} are required"
            ),
            ContractError::Stride {
                required_bytes,
                found_bytes,
            } => write!(
                f,
                "row_stride_bytes is {found_bytes} but one row needs {required_bytes} bytes"
            ),
            ContractError::Misaligned {
                field,
                alignment,
                found,
            } => write!(f, "{field} is {found}, not a multiple of {alignment}"),
            ContractError::Format { reason } => {
                write!(f, "not the v0 format: {reason}")
            }
            ContractError::DuplicateCid { pubchem_cid } => {
                write!(
                    f,
                    "PubChem CID {pubchem_cid} appears twice in the species table"
                )
            }
            ContractError::DuplicateId { field, id } => {
                write!(f, "{field} lists id {id} twice")
            }
            ContractError::ShownWithoutTrueSegmentation { camera } => write!(
                f,
                "a shown segmentation frame for camera {camera} has no true segmentation \
                 frame for that camera in the same record"
            ),
            ContractError::EpisodeMismatch {
                field,
                expected,
                found,
            } => write!(
                f,
                "{field} is episode {found}, but the record is of episode {expected}"
            ),
            ContractError::DuplicateName { field } => {
                write!(f, "{field} lists one name twice")
            }
            ContractError::EmptyTableForNonEmptyVector => {
                f.write_str("the species table is empty but the concentration vector is not")
            }
            ContractError::TableVersion { vector, table } => write!(
                f,
                "the vector was written for species table version {vector}, not version {table}"
            ),
            ContractError::AudioDiscontinuity {
                expected_first_sample_index,
                found_first_sample_index,
            } => write!(
                f,
                "audio is not continuous: the next chunk must start at sample \
                 {expected_first_sample_index}, it starts at {found_first_sample_index}"
            ),
            ContractError::BadIdentifier { field } => write!(
                f,
                "{field} must be 1 to {} characters from [a-z0-9_.-], starting with a letter or digit",
                crate::MAX_IDENTIFIER_LEN
            ),
            ContractError::ExtraChannelWithoutRgb { channel } => {
                write!(
                    f,
                    "{channel} is sent with the RGB frame and cannot be present without it"
                )
            }
            ContractError::Json { reason } => write!(f, "invalid JSON: {reason}"),
            ContractError::Channel { channel, cause } => write!(f, "{channel}: {cause}"),
        }
    }
}

impl std::error::Error for ContractError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ContractError::Channel { cause, .. } => Some(cause.as_ref()),
            _ => None,
        }
    }
}
