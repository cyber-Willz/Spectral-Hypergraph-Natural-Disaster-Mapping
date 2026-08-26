//! End-to-end demo: load the (sample) county adjacency graph + NRI feature
//! table, run a toy risk-propagation "prediction" over it (standing in for
//! a trained NBSC/GAT model — see the README for how to wire in a real
//! trained `nbsc::gat_layer` model instead), and print an audit trail for
//! one target county.
//!
//! Run with:
//!   cargo run -p gis_audit --example flood_risk_audit
//!
//! Swap the two `data/sample_*` paths for the real, full nationwide
//! downloads (see `fetch_real_data.sh`) to run this against the whole US.

use gis_audit::provenance::{explain_prediction, LayerAttention};
use gis_audit::{CountyGraph, NriFeatures};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Resolve relative to this crate's directory (CARGO_MANIFEST_DIR, set by
    // Cargo at compile time), not the process's current working directory --
    // `cargo run` leaves cwd wherever *you* invoked it from (workspace root,
    // gis_audit/, anywhere), not the package directory, so a bare relative
    // "data/..." path only works by accident depending on where you stand.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let adjacency_path = format!("{manifest_dir}/data/sample_county_adjacency.txt");
    let nri_path = format!("{manifest_dir}/data/sample_nri.csv");

    let cg = CountyGraph::from_file(&adjacency_path)?;
    println!("Loaded county graph: {} counties, {} adjacency edges", cg.graph.n, cg.graph.m());

    let nri = NriFeatures::from_file(&nri_path)?;
    let features = nri.align_to(&cg.index_to_geoid);
    println!(
        "Loaded NRI features: {} of {} feature columns present in this export",
        nri.found_columns.iter().filter(|b| **b).count(),
        nri.found_columns.len()
    );

    // --- Stand-in "prediction": is this county flood-mitigation-critical? ---
    // A real deployment replaces this block with a trained
    // nbsc::gat_layer::GatHead (or nbsc::burn_layer::NbscLayer) forward
    // pass, exporting its per-layer alpha_ij into LayerAttention::weights.
    // The auditability story is identical either way -- explain_prediction
    // only needs the per-layer attention weights, not how they were made.
    let rfld_idx = 2; // RFLD_EALT column, see nri_features::FEATURE_NAMES
    for (i, geoid) in cg.index_to_geoid.iter().enumerate() {
        let risk = features[i][rfld_idx];
        println!("  {:<28} {:>9}  riverine-flood EAL ${:>13.0}", cg.name_of(i), geoid, risk);
    }

    // Two GAT-style layers, degree-normalized here as the non-learned
    // fallback described in provenance::LayerAttention::degree_normalized.
    let layer1 = LayerAttention::degree_normalized(&cg.graph, 0.4);
    let layer2 = LayerAttention::degree_normalized(&cg.graph, 0.4);

    let target = cg.index_of("48201").ok_or("Harris County (48201) not in graph")?; // Harris County, TX
    let trail = explain_prediction(&cg, &[layer1, layer2], target, 5);

    println!("\n{}", trail.report());
    println!("--- machine-readable audit record ---");
    println!("{}", trail.to_json_pretty());

    Ok(())
}
