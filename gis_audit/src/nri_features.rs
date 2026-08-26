//! Parser for FEMA's **National Risk Index (NRI)** county-level table —
//! the node-feature input for this crate.
//!
//! Source (free, public domain, no auth/API key):
//!   <https://hazards.fema.gov/nri/data-resources> (CSV, "Counties" table)
//!   mirrored on OpenFEMA: <https://www.fema.gov/about/openfema/data-sets/national-risk-index-data>
//!
//! The full table has ~130 columns (per-hazard EAL/exposure/frequency for
//! 18 hazard types + social vulnerability + community resilience). This
//! parser only pulls the columns relevant to flood mitigation / disaster
//! routing / zoning-risk use cases, addressed by name so it is resilient to
//! column reordering and tolerant of missing hazard columns (NRI leaves a
//! hazard's columns blank for counties where that hazard is not applicable,
//! e.g. `AVLN` avalanche outside mountain states):
//!
//! - `STCOFIPS` — 5-digit county GEOID (join key against [`crate::county_adjacency`])
//! - `RISK_SCORE` — overall composite risk score (0-100)
//! - `EAL_VALT` — total Expected Annual Loss, all hazards, in dollars
//! - `RFLD_EALT` — Expected Annual Loss, riverine flooding
//! - `CFLD_EALT` — Expected Annual Loss, coastal flooding
//! - `SOVI_SCORE` — Social Vulnerability score (CDC/ATSDR SVI-derived)
//! - `RESL_SCORE` — Community Resilience score
//! - `POPULATION` — county population
//!
//! Any subset of these may be absent from a given NRI release/export; missing
//! columns are filled with `0.0` rather than failing the parse, and which
//! columns were actually found is reported in [`NriFeatures::found_columns`]
//! so callers can tell a genuine zero from a missing column.

use std::collections::HashMap;
use std::fmt;

pub const FEATURE_NAMES: [&str; 7] = [
    "RISK_SCORE",
    "EAL_VALT",
    "RFLD_EALT",
    "CFLD_EALT",
    "SOVI_SCORE",
    "RESL_SCORE",
    "POPULATION",
];

/// Column subset for the live civil-defense/disaster-threat pull in
/// `examples/live_defense_threat_mapping.rs`, sourced from OpenFEMA's
/// Disaster Declarations Summaries API rather than the NRI. Named
/// separately from [`FEATURE_NAMES`] (NRI's schema) because it's a
/// different live source with its own columns; [`NriFeatures::parse`]
/// matches by name against whichever list the caller passes to
/// [`NriFeatures::parse_with_columns`], so both share one parser.
pub const THREAT_FEATURE_NAMES: [&str; 3] = [
    "DECLARATION_COUNT",
    "SMOKEHOUSE_CREEK_CORRIDOR",
    "DAYS_SINCE_LAST_DECLARATION",
];

#[derive(Debug)]
pub enum NriError {
    Io(std::io::Error),
    NoGeoidColumn,
    Empty,
}
impl fmt::Display for NriError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NriError::Io(e) => write!(f, "io error reading NRI csv: {e}"),
            NriError::NoGeoidColumn => {
                write!(f, "no STCOFIPS/GEOID column found in NRI csv header")
            }
            NriError::Empty => write!(f, "NRI csv parsed to zero data rows"),
        }
    }
}
impl std::error::Error for NriError {}
impl From<std::io::Error> for NriError {
    fn from(e: std::io::Error) -> Self {
        NriError::Io(e)
    }
}

pub struct NriFeatures {
    /// GEOID -> feature vector, in the order of whichever column-name list
    /// was passed to [`NriFeatures::parse_with_columns`] (or
    /// [`FEATURE_NAMES`] for plain [`NriFeatures::parse`]).
    pub by_geoid: HashMap<String, Vec<f64>>,
    pub found_columns: Vec<bool>,
    /// The column names actually used for this table, so
    /// [`NriFeatures::align_to`]'s output width is self-describing without
    /// the caller having to remember which name list was passed in.
    pub column_names: Vec<String>,
}

fn split_csv_line(line: &str) -> Vec<String> {
    // Minimal RFC4180-ish CSV split: handles quoted fields containing commas.
    // NRI's export doesn't embed newlines inside quoted fields, so a
    // per-line splitter (no multi-line record support) is sufficient.
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' => {
                if in_quotes && chars.peek() == Some(&'"') {
                    cur.push('"');
                    chars.next();
                } else {
                    in_quotes = !in_quotes;
                }
            }
            ',' if !in_quotes => {
                fields.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    fields.push(cur.trim().to_string());
    fields
}

impl NriFeatures {
    /// Parse using the standard NRI [`FEATURE_NAMES`] column set.
    pub fn parse(raw: &str) -> Result<Self, NriError> {
        Self::parse_with_columns(raw, &FEATURE_NAMES)
    }

    /// Parse using an arbitrary named column set -- e.g.
    /// [`THREAT_FEATURE_NAMES`] for the live OpenFEMA-derived table, or any
    /// other CSV with a GEOID/STCOFIPS column plus named numeric columns.
    /// Column matching, missing-column tolerance, and FIPS zero-padding
    /// behave identically to [`Self::parse`]; only which columns are
    /// looked for changes.
    pub fn parse_with_columns(raw: &str, feature_names: &[&str]) -> Result<Self, NriError> {
        let mut lines = raw
            .lines()
            .skip_while(|l| l.trim_start().starts_with('#') || l.trim().is_empty());
        let header = lines.next().ok_or(NriError::Empty)?;
        let cols = split_csv_line(header);

        let geoid_idx = cols
            .iter()
            .position(|c| c.eq_ignore_ascii_case("STCOFIPS") || c.eq_ignore_ascii_case("GEOID") || c.eq_ignore_ascii_case("COUNTYFIPS"))
            .ok_or(NriError::NoGeoidColumn)?;

        let feature_idx: Vec<Option<usize>> = feature_names
            .iter()
            .map(|name| cols.iter().position(|c| c.eq_ignore_ascii_case(name)))
            .collect();
        let found_columns: Vec<bool> = feature_idx.iter().map(|o| o.is_some()).collect();

        let mut by_geoid = HashMap::new();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let row = split_csv_line(line);
            let Some(geoid_raw) = row.get(geoid_idx) else { continue };
            // FIPS codes are sometimes exported without a leading zero
            // (e.g. Connecticut/Alabama counties starting '0'); zero-pad to 5.
            let geoid = format!("{:0>5}", geoid_raw.trim());
            if geoid.trim().is_empty() || geoid == "00000" {
                continue;
            }

            let feats: Vec<f64> = feature_idx
                .iter()
                .map(|maybe_i| {
                    maybe_i
                        .and_then(|i| row.get(i))
                        .and_then(|s| s.trim().parse::<f64>().ok())
                        .unwrap_or(0.0)
                })
                .collect();
            by_geoid.insert(geoid, feats);
        }

        if by_geoid.is_empty() {
            return Err(NriError::Empty);
        }

        Ok(NriFeatures {
            by_geoid,
            found_columns,
            column_names: feature_names.iter().map(|s| s.to_string()).collect(),
        })
    }

    pub fn from_file(path: &str) -> Result<Self, NriError> {
        let raw = std::fs::read_to_string(path)?;
        Self::parse(&raw)
    }

    pub fn from_file_with_columns(path: &str, feature_names: &[&str]) -> Result<Self, NriError> {
        let raw = std::fs::read_to_string(path)?;
        Self::parse_with_columns(&raw, feature_names)
    }

    /// Build a dense `n x d` feature matrix aligned to a
    /// [`crate::county_adjacency::CountyGraph`]'s node order. Counties
    /// present in the adjacency graph but absent from this table (e.g. NRI
    /// excludes some island-area equivalents some years) get an all-zero
    /// row rather than failing the whole build.
    pub fn align_to(&self, geoid_order: &[String]) -> Vec<Vec<f64>> {
        let width = self.column_names.len();
        geoid_order
            .iter()
            .map(|g| self.by_geoid.get(g).cloned().unwrap_or_else(|| vec![0.0; width]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_subset_of_columns_and_pads_fips() {
        let raw = "STCOFIPS,COUNTY,RISK_SCORE,RFLD_EALT,SOVI_SCORE\n\
                   48201,Harris,87.3,1200000,62.1\n\
                   1003,Baldwin,45.0,50000,,\n";
        let nri = NriFeatures::parse(raw).unwrap();
        assert!(nri.by_geoid.contains_key("48201"));
        assert!(nri.by_geoid.contains_key("01003")); // zero-padded
        assert_eq!(nri.found_columns[0], true); // RISK_SCORE found
        assert_eq!(nri.found_columns[3], false); // CFLD_EALT absent from header
    }
}
