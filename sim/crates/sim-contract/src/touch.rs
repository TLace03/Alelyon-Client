//! Touch: tactile maps, one per skin sensor.
//!
//! Invariants:
//! - A sensor's map is `rows x cols` taxels in row-major order: the taxel at row
//!   `r`, column `c` is entry `r * cols + c` of each list, and all three lists
//!   have exactly `rows * cols` entries.
//! - Units are SI: pressure in pascals, shear traction in pascals (tangential,
//!   in the sensor's own `[x, y]` frame), temperature in kelvin.
//! - Pressure is a contact pressure: never negative. Temperature is absolute:
//!   above 0 K. A taxel not in contact reports 0 Pa and the temperature of the
//!   skin there, never NaN.
//! - Sensor ids are unique within one `TouchMaps`. An empty `TouchMaps` (no
//!   sensors) is valid: the channel exists before a skin does.
//! - Contract v0 samples touch at 100 Hz.

use serde::{Deserialize, Serialize};

use crate::ContractError;
use crate::check::{all_at_least_f32, all_finite_f32};

/// One sensor's tactile map.
///
/// Invariants (checked by [`TouchMap::validate`]): `rows` and `cols` are at least
/// 1; each list has `rows * cols` entries; every number is finite; pressure is
/// not negative; temperature is above 0 K.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TouchMap {
    /// Which skin sensor this is; unique within the bundle.
    pub sensor_id: u32,
    /// Taxel rows.
    pub rows: u32,
    /// Taxel columns.
    pub cols: u32,
    /// Normal contact pressure per taxel, pascals.
    pub pressure_pa: Vec<f32>,
    /// Shear traction per taxel, pascals, `[x, y]` in the sensor's frame.
    pub shear_pa: Vec<[f32; 2]>,
    /// Temperature per taxel, kelvin.
    pub temperature_k: Vec<f32>,
}

impl TouchMap {
    /// `rows * cols`, the number of taxels.
    pub fn taxel_count(&self) -> u64 {
        u64::from(self.rows) * u64::from(self.cols)
    }

    /// Checks the shape, finiteness and ranges.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.rows == 0 {
            return Err(ContractError::Empty { field: "rows" });
        }
        if self.cols == 0 {
            return Err(ContractError::Empty { field: "cols" });
        }
        let expected = self.taxel_count();
        for (field, found) in [
            ("pressure_pa", self.pressure_pa.len()),
            ("shear_pa", self.shear_pa.len()),
            ("temperature_k", self.temperature_k.len()),
        ] {
            if found as u64 != expected {
                return Err(ContractError::LengthMismatch {
                    field,
                    expected,
                    found: found as u64,
                });
            }
        }
        all_at_least_f32(
            "pressure_pa",
            &self.pressure_pa,
            0.0,
            "pressure cannot be negative",
        )?;
        for shear in &self.shear_pa {
            all_finite_f32("shear_pa", shear)?;
        }
        all_finite_f32("temperature_k", &self.temperature_k)?;
        if self.temperature_k.iter().any(|&t| t <= 0.0) {
            return Err(ContractError::OutOfRange {
                field: "temperature_k",
                reason: "absolute temperature must be above 0 K",
            });
        }
        Ok(())
    }
}

/// Every skin sensor's map for one tick.
///
/// Invariants (checked by [`TouchMaps::validate`]): every map is valid and
/// sensor ids are unique. No sensors at all is valid.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TouchMaps {
    /// One map per sensor.
    pub sensors: Vec<TouchMap>,
}

impl TouchMaps {
    /// Checks every map and the uniqueness of sensor ids.
    pub fn validate(&self) -> Result<(), ContractError> {
        let mut seen = std::collections::BTreeSet::new();
        for map in &self.sensors {
            map.validate()?;
            if !seen.insert(map.sensor_id) {
                return Err(ContractError::DuplicateId {
                    field: "sensors",
                    id: u64::from(map.sensor_id),
                });
            }
        }
        Ok(())
    }
}
