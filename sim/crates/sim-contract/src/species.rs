//! Smell and taste: concentration vectors over a versioned table of molecules.
//!
//! Mapping molecules to percepts ("roasty", "sweet") is the Sinai side's model,
//! fed by these vectors. So the contract names molecules, never percepts: a
//! species is identified by its PubChem compound id (CID), and a vector is
//! indexed by position in a versioned table of species.
//!
//! Invariants:
//! - A [`SpeciesTable`] lists each CID at most once. CIDs are positive integers;
//!   0 is never a CID. A species `name` is a label for a person (the molecule's
//!   name); nothing keys on it, and it is never a percept word.
//! - A table's version covers its contents *and its order*: adding, removing or
//!   reordering a species is a new version, because a vector is read by position.
//! - A [`SpeciesVector`] has exactly one concentration per species in the table
//!   named by its `table_version`, in table order. Concentrations are finite and
//!   not negative.
//! - Units depend on the channel the vector travels in: smell is mol/m^3 of gas
//!   at the nose, taste is mol/L of solution at the tongue contact.
//! - An empty table with an empty vector is valid: the channel exists before any
//!   species do. An empty table with a non-empty vector is refused.
//! - Contract v0 samples smell and taste at 10 Hz.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::ContractError;
use crate::check::all_at_least_f32;

/// One molecule in a species table.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Species {
    /// The molecule's PubChem compound id; at least 1.
    pub pubchem_cid: u64,
    /// A label for people, such as the molecule's common name.
    pub name: String,
}

/// A versioned list of the molecules a vector is indexed by.
///
/// Invariants (checked by [`SpeciesTable::validate`]): every CID is at least 1
/// and appears once.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeciesTable {
    /// The table's version; covers contents and order.
    pub version: u32,
    /// The molecules, in vector order.
    pub species: Vec<Species>,
}

impl SpeciesTable {
    /// Checks CIDs: positive and unique.
    pub fn validate(&self) -> Result<(), ContractError> {
        let mut seen = BTreeSet::new();
        for species in &self.species {
            if species.pubchem_cid == 0 {
                return Err(ContractError::OutOfRange {
                    field: "pubchem_cid",
                    reason: "a PubChem CID is a positive integer",
                });
            }
            if !seen.insert(species.pubchem_cid) {
                return Err(ContractError::DuplicateCid {
                    pubchem_cid: species.pubchem_cid,
                });
            }
        }
        Ok(())
    }

    /// The vector index of the molecule with this CID.
    pub fn position_of(&self, pubchem_cid: u64) -> Option<usize> {
        self.species
            .iter()
            .position(|species| species.pubchem_cid == pubchem_cid)
    }
}

/// Concentrations of the species of one table.
///
/// Invariants (checked by [`SpeciesVector::validate`] against the table): the
/// table is valid; `table_version` equals the table's version; if the vector is
/// not empty the table is not; the lengths are equal; every concentration is
/// finite and not negative.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpeciesVector {
    /// The version of the [`SpeciesTable`] this vector is indexed by.
    pub table_version: u32,
    /// One per species, in table order: mol/m^3 for smell, mol/L for taste.
    pub concentrations: Vec<f32>,
}

impl SpeciesVector {
    /// The empty vector of a table: the channel exists with no content yet.
    pub fn empty(table_version: u32) -> Self {
        Self {
            table_version,
            concentrations: Vec::new(),
        }
    }

    /// Checks this vector against the table it is indexed by.
    pub fn validate(&self, table: &SpeciesTable) -> Result<(), ContractError> {
        table.validate()?;
        if self.table_version != table.version {
            return Err(ContractError::TableVersion {
                vector: self.table_version,
                table: table.version,
            });
        }
        if table.species.is_empty() && !self.concentrations.is_empty() {
            return Err(ContractError::EmptyTableForNonEmptyVector);
        }
        if self.concentrations.len() != table.species.len() {
            return Err(ContractError::LengthMismatch {
                field: "concentrations",
                expected: table.species.len() as u64,
                found: self.concentrations.len() as u64,
            });
        }
        all_at_least_f32(
            "concentrations",
            &self.concentrations,
            0.0,
            "a concentration cannot be negative",
        )
    }
}
