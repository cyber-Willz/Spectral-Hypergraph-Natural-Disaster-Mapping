//! `gis_audit` — integrates the `spectral_hypergraph`/`nbsc` stack (non-
//! backtracking spectral convolution, GAT/GCN/SAGE baselines) with the
//! `spectral_dqg`/`krylov_ds` stack's non-backtracking (Hashimoto) walk
//! machinery to produce **auditable** GNN predictions over real US
//! county-level GIS graphs.
//!
//! ## The problem this addresses
//! High-stakes GIS decisions (urban zoning, flood mitigation, disaster
//! routing) need auditability: a regulator, zoning board, or emergency
//! manager needs to know not just *what* a model predicted but *which
//! specific neighboring counties, along which specific paths, drove that
//! prediction*. Standard GNN architectures aggregate neighbor features into
//! a fixed-size vector at every layer; by the second or third layer, that
//! aggregation has mixed information from potentially hundreds of counties
//! with no per-source accounting left.
//!
//! ## The approach
//! - [`county_adjacency`] loads the free, public-domain US Census County
//!   Adjacency File as the graph topology.
//! - [`nri_features`] loads FEMA's free, public-domain National Risk Index
//!   county table (flood/hazard EAL, social vulnerability, resilience) as
//!   node features — chosen specifically for the flood-mitigation /
//!   disaster-routing / zoning-risk use case named in the brief.
//! - `nbsc` (this workspace's non-backtracking spectral convolution crate)
//!   or its GAT baseline does the actual prediction; GAT's per-edge
//!   attention weights are exactly the per-neighbor accounting that a
//!   symmetric-normalized GCN layer discards by construction.
//! - [`provenance`] composes those per-layer attention weights via
//!   attention rollout, *restricted to non-backtracking paths* (reusing the
//!   Hashimoto-operator walk structure this codebase already implements for
//!   the Ihara zeta function), to produce a ranked, JSON-exportable audit
//!   trail: which literal chain of counties, and how much of the
//!   prediction, in units an auditor can read directly.

pub mod county_adjacency;
pub mod nri_features;
pub mod provenance;

#[cfg(feature = "live")]
pub mod live_model;

pub use county_adjacency::CountyGraph;
pub use nri_features::NriFeatures;
pub use provenance::{explain_prediction, AuditTrail, LayerAttention};
