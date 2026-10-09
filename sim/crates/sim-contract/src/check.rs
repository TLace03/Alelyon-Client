//! The small checks every module's `validate` is built from.
//!
//! Invariants:
//! - Every numeric check refuses NaN and infinity before it looks at a range, so
//!   no range comparison is ever made with a NaN.
//! - The checks are private to the crate: the public contract is each type's
//!   `validate`, not these helpers.

use crate::{ContractError, MAX_IDENTIFIER_LEN};

/// How far from 1.0 a quaternion's length may be: the slack an `f32` unit
/// quaternion that went through a few multiplications needs, and tight enough
/// to refuse an unnormalised one.
pub(crate) const QUATERNION_NORM_TOLERANCE: f64 = 1e-3;

pub(crate) fn finite_f32(field: &'static str, value: f32) -> Result<(), ContractError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ContractError::NonFinite { field })
    }
}

pub(crate) fn finite_f64(field: &'static str, value: f64) -> Result<(), ContractError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ContractError::NonFinite { field })
    }
}

pub(crate) fn all_finite_f32(field: &'static str, values: &[f32]) -> Result<(), ContractError> {
    values
        .iter()
        .try_for_each(|&value| finite_f32(field, value))
}

/// Every value finite and at least `minimum`.
pub(crate) fn all_at_least_f32(
    field: &'static str,
    values: &[f32],
    minimum: f32,
    reason: &'static str,
) -> Result<(), ContractError> {
    for &value in values {
        finite_f32(field, value)?;
        if value < minimum {
            return Err(ContractError::OutOfRange { field, reason });
        }
    }
    Ok(())
}

/// A finite `[x, y, z, w]` quaternion of length 1 within
/// [`QUATERNION_NORM_TOLERANCE`].
pub(crate) fn unit_quaternion(field: &'static str, q: &[f32; 4]) -> Result<(), ContractError> {
    all_finite_f32(field, q)?;
    let squared: f64 = q.iter().map(|&c| f64::from(c) * f64::from(c)).sum();
    if (squared.sqrt() - 1.0).abs() > QUATERNION_NORM_TOLERANCE {
        return Err(ContractError::OutOfRange {
            field,
            reason: "a quaternion must have length 1",
        });
    }
    Ok(())
}

/// `[a-z0-9_.-]`, 1 to [`MAX_IDENTIFIER_LEN`] bytes, starting with a letter or
/// digit. Identifiers can reach paths on disk, so a separator, a leading dot
/// (which rules out `..`) and anything non-ASCII are refused before use.
pub(crate) fn identifier(field: &'static str, text: &str) -> Result<(), ContractError> {
    let bytes = text.as_bytes();
    let first_ok = bytes
        .first()
        .is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit());
    let rest_ok = bytes
        .iter()
        .all(|&b| b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'_' | b'.' | b'-'));
    if first_ok && rest_ok && bytes.len() <= MAX_IDENTIFIER_LEN {
        Ok(())
    } else {
        Err(ContractError::BadIdentifier { field })
    }
}

/// Parses JSON into `T`, turning the parser's failure into a [`ContractError`].
pub(crate) fn parse<T: serde::de::DeserializeOwned>(text: &str) -> Result<T, ContractError> {
    serde_json::from_str(text).map_err(|e| ContractError::Json {
        reason: e.to_string(),
    })
}
