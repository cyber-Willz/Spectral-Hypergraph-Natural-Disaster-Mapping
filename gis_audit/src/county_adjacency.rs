//! Parser for the U.S. Census Bureau **County Adjacency File** — the graph
//! topology input for this crate.
//!
//! Source (free, public domain, no auth/API key):
//!   <https://www.census.gov/geographies/reference-files/time-series/geo/county-adjacency.html>
//! Legacy flat file (tab-delimited, stable format since 2010, easiest to
//! parse):
//!   <https://www2.census.gov/geo/docs/reference/county_adjacency.txt>
//!
//! Two on-the-wire formats are both handled here:
//! - **Legacy/2010**: tab-delimited. A county's *first* neighbor row carries
//!   `County Name`, `County GEOID`, `Neighbor Name`, `Neighbor GEOID`;
//!   subsequent neighbor rows for the *same* county leave the first two
//!   columns blank (fill-forward). Every county also lists itself as a
//!   neighbor.
//! - **2023+**: pipe-delimited (`|`), `County GEOID` and `County Name`
//!   repeated on every row, plus a 5th `Length` column (shared-boundary
//!   length in meters) and no more self-adjacency rows.
//!
//! Each GEOID is the 5-digit state+county FIPS code (e.g. `48201` = Harris
//! County, TX). Nodes in the resulting [`nbsc::graph::Graph`] are indexed
//! 0..n in first-seen order; [`CountyGraph::geoid_of`] /
//! [`CountyGraph::index_of`] translate between that index and the GEOID so
//! predictions and audit trails can be reported in GEOID/county-name terms
//! rather than opaque node indices.

use nbsc::graph::Graph;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug)]
pub enum AdjacencyError {
    Io(std::io::Error),
    Empty,
}

impl fmt::Display for AdjacencyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdjacencyError::Io(e) => write!(f, "io error reading county adjacency file: {e}"),
            AdjacencyError::Empty => write!(f, "county adjacency file parsed to zero rows"),
        }
    }
}
impl std::error::Error for AdjacencyError {}
impl From<std::io::Error> for AdjacencyError {
    fn from(e: std::io::Error) -> Self {
        AdjacencyError::Io(e)
    }
}

/// A US county-adjacency graph: the `nbsc`/`spectral_hypergraph`-compatible
/// [`Graph`] plus the GEOID<->node-index and GEOID->display-name lookups
/// needed to make predictions and audit trails human-readable.
pub struct CountyGraph {
    pub graph: Graph,
    pub geoid_to_index: HashMap<String, usize>,
    pub index_to_geoid: Vec<String>,
    pub index_to_name: Vec<String>,
}

impl CountyGraph {
    pub fn geoid_of(&self, idx: usize) -> &str {
        &self.index_to_geoid[idx]
    }
    pub fn name_of(&self, idx: usize) -> &str {
        &self.index_to_name[idx]
    }
    pub fn index_of(&self, geoid: &str) -> Option<usize> {
        self.geoid_to_index.get(geoid).copied()
    }

    /// Parse either the legacy tab-delimited or the 2023+ pipe-delimited
    /// Census county adjacency format from raw text. Delimiter and
    /// fill-forward behavior are auto-detected per line.
    pub fn parse(raw: &str) -> Result<Self, AdjacencyError> {
        let mut geoid_to_index: HashMap<String, usize> = HashMap::new();
        let mut index_to_geoid: Vec<String> = Vec::new();
        let mut index_to_name: Vec<String> = Vec::new();
        let mut edges: Vec<(usize, usize)> = Vec::new();

        let mut cur_name = String::new();
        let mut cur_geoid = String::new();

        let get_or_insert = |geoid: &str,
                                  name: &str,
                                  geoid_to_index: &mut HashMap<String, usize>,
                                  index_to_geoid: &mut Vec<String>,
                                  index_to_name: &mut Vec<String>|
         -> usize {
            if let Some(&i) = geoid_to_index.get(geoid) {
                return i;
            }
            let i = index_to_geoid.len();
            geoid_to_index.insert(geoid.to_string(), i);
            index_to_geoid.push(geoid.to_string());
            index_to_name.push(name.to_string());
            i
        };

        for line in raw.lines() {
            if line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            let delim = if line.contains('|') { '|' } else { '\t' };
            let cols: Vec<&str> = line.split(delim).map(|c| c.trim()).collect();
            if cols.len() < 4 {
                continue; // malformed/header line, skip rather than fail the whole parse
            }
            let (name, geoid, nbr_name, nbr_geoid) = (cols[0], cols[1], cols[2], cols[3]);

            // fill-forward: legacy format leaves columns 0/1 blank on
            // continuation rows for the same county.
            if !geoid.is_empty() {
                cur_name = name.to_string();
                cur_geoid = geoid.to_string();
            }
            if cur_geoid.is_empty() || nbr_geoid.is_empty() {
                continue;
            }
            // Defensive check: a real GEOID is all ASCII digits. A
            // malformed row (e.g. wrong tab/column count shifting a county
            // *name* into the GEOID slot) would otherwise silently create
            // a bogus graph node instead of failing loudly -- this has
            // bitten this parser's hand-authored fixtures more than once,
            // so skip (rather than crash) any row where either GEOID slot
            // isn't numeric, since a hand-edited data file is more likely
            // than a genuinely non-numeric Census GEOID.
            if !cur_geoid.chars().all(|c| c.is_ascii_digit())
                || !nbr_geoid.chars().all(|c| c.is_ascii_digit())
            {
                continue;
            }

            let u = get_or_insert(
                &cur_geoid,
                &cur_name,
                &mut geoid_to_index,
                &mut index_to_geoid,
                &mut index_to_name,
            );
            let v = get_or_insert(
                nbr_geoid,
                nbr_name,
                &mut geoid_to_index,
                &mut index_to_geoid,
                &mut index_to_name,
            );
            if u != v {
                edges.push((u, v));
            }
        }

        if index_to_geoid.is_empty() {
            return Err(AdjacencyError::Empty);
        }

        let mut graph = Graph::new(index_to_geoid.len());
        for (u, v) in edges {
            graph.add_edge(u, v);
        }

        Ok(CountyGraph { graph, geoid_to_index, index_to_geoid, index_to_name })
    }

    pub fn from_file(path: &str) -> Result<Self, AdjacencyError> {
        let raw = std::fs::read_to_string(path)?;
        Self::parse(&raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_fill_forward_format() {
        let raw = "\
Harris County TX\t48201\tFort Bend County TX\t48157
\t\tMontgomery County TX\t48339
\t\tHarris County TX\t48201
Fort Bend County TX\t48157\tHarris County TX\t48201
\t\tFort Bend County TX\t48157
";
        let cg = CountyGraph::parse(raw).unwrap();
        assert_eq!(cg.graph.n, 3);
        let harris = cg.index_of("48201").unwrap();
        let fb = cg.index_of("48157").unwrap();
        let mont = cg.index_of("48339").unwrap();
        assert!(cg.graph.neighbors[harris].contains(&fb));
        assert!(cg.graph.neighbors[harris].contains(&mont));
        assert!(cg.graph.neighbors[fb].contains(&harris));
    }

    #[test]
    fn pipe_2023_format() {
        let raw = "\
Harris County, TX|48201|Fort Bend County, TX|48157|41210.5
Fort Bend County, TX|48157|Harris County, TX|48201|41210.5
";
        let cg = CountyGraph::parse(raw).unwrap();
        assert_eq!(cg.graph.n, 2);
        assert_eq!(cg.graph.m(), 1);
    }
}
