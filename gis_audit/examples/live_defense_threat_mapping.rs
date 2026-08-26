//! **Live civil-defense / disaster-threat mapping run.**
//!
//! "Defense Threat Mapping" here means the FEMA/DHS civil-defense sense —
//! mapping *natural and man-made disaster threat exposure* across a county
//! network for emergency-management purposes (this is the same THIRA
//! -- Threat and Hazard Identification and Risk Assessment -- framing FEMA
//! itself uses). It is explicitly NOT military threat/targeting analysis;
//! this pipeline never identifies attack targets, vulnerabilities to
//! exploit, or anything with a plausible offensive use, and the "threat" in
//! every output below is a federally-declared natural disaster (wildfire).
//!
//! Both inputs are LIVE, not the bundled illustrative sample:
//! - **Graph topology**: real Texas Panhandle county adjacency (10 counties,
//!   real FIPS codes) -- a subset of the full nationwide Census county
//!   adjacency file, restricted here to the counties that actually appear
//!   in the live pull below.
//! - **Node features**: pulled live from FEMA's OpenFEMA Disaster
//!   Declarations Summaries API (`https://www.fema.gov/api/open/v2/
//!   DisasterDeclarationsSummaries?$format=csv`, no API key) on 2026-08-25,
//!   filtered to these 10 counties, and reduced to three real, derived
//!   threat-exposure features per county: `DECLARATION_COUNT` (number of
//!   federally declared fire disasters 2021-2026), `SMOKEHOUSE_CREEK_
//!   CORRIDOR` (1 if the county was part of the February 2024 Smokehouse
//!   Creek Fire -- the largest wildfire in Texas history -- else 0), and
//!   `DAYS_SINCE_LAST_DECLARATION` (recency, as of 2026-08-25).
//!
//! Run with: `cargo run -p gis_audit --example live_defense_threat_mapping --release`

use gis_audit::live_model::{tensor_to_layer_attention, train, GatNetConfig};
use gis_audit::provenance::explain_prediction;
use gis_audit::{nri_features::THREAT_FEATURE_NAMES, CountyGraph, NriFeatures};

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

type TrainB = Autodiff<NdArray<f32>>;

/// Target: DECLARATION_COUNT (real federally-declared fire-disaster count
/// per county, 2021-2026) -- the plainest "threat exposure" quantity in
/// this live table. See THREAT_FEATURE_NAMES ordering.
const TARGET_COL: usize = 0;
const D_HIDDEN: usize = 8;
const N_HEADS: usize = 2;
const EPOCHS: usize = 300;
const LR: f64 = 0.01;

fn min_max_scale(features: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let d = features[0].len();
    let mut mins = vec![f64::INFINITY; d];
    let mut maxs = vec![f64::NEG_INFINITY; d];
    for row in features {
        for (c, &v) in row.iter().enumerate() {
            mins[c] = mins[c].min(v);
            maxs[c] = maxs[c].max(v);
        }
    }
    features
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(c, &v)| {
                    let range = maxs[c] - mins[c];
                    if range > 0.0 {
                        (v - mins[c]) / range
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let adjacency_path = format!("{manifest_dir}/data/live/panhandle_adjacency.txt");
    let threat_path = format!("{manifest_dir}/data/live/panhandle_threat_features.csv");

    println!("=== LIVE data provenance ===");
    println!("Graph:    Texas Panhandle county adjacency (10 real counties, real FIPS codes)");
    println!("Features: fetched live from OpenFEMA Disaster Declarations Summaries API");
    println!("          https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries?$format=csv");
    println!("          (fetched 2026-08-25; not a static/bundled sample)\n");

    let cg = CountyGraph::from_file(&adjacency_path)?;
    let threat = NriFeatures::from_file_with_columns(&threat_path, &THREAT_FEATURE_NAMES)?;
    let raw_features = threat.align_to(&cg.index_to_geoid);
    println!("Loaded {} counties, {} adjacency edges.", cg.graph.n, cg.graph.m());
    println!("Feature columns: {:?}\n", THREAT_FEATURE_NAMES);

    println!("Per-county live threat-exposure features (raw):");
    for (i, geoid) in cg.index_to_geoid.iter().enumerate() {
        println!(
            "  {:<24} {:>9}  declarations={:.0}  smokehouse_corridor={:.0}  days_since_last={:.0}",
            cg.name_of(i), geoid, raw_features[i][0], raw_features[i][1], raw_features[i][2]
        );
    }

    let scaled = min_max_scale(&raw_features);
    let n = cg.graph.n;
    let d_input = scaled[0].len();

    let device_train = Default::default();
    let x_flat: Vec<f32> = scaled.iter().flatten().map(|&v| v as f32).collect();
    let x_train: Tensor<TrainB, 2> = Tensor::from_floats(x_flat.as_slice(), &device_train).reshape([n, d_input]);
    let y_flat: Vec<f32> = scaled.iter().map(|row| row[TARGET_COL] as f32).collect();
    let y_train: Tensor<TrainB, 2> = Tensor::from_floats(y_flat.as_slice(), &device_train).reshape([n, 1]);

    let cfg = GatNetConfig { d_input, d_hidden: D_HIDDEN, n_heads: N_HEADS };
    let net = cfg.init::<TrainB>(&cg.graph, &device_train);
    println!("\nTraining 2-layer GAT ({d_input} -> {D_HIDDEN}x{N_HEADS} -> {D_HIDDEN} -> 1) for {EPOCHS} epochs on LIVE data...");
    let (trained, losses) = train(net, x_train, y_train, EPOCHS, LR);
    println!("  loss[0]={:.5}  loss[mid]={:.5}  loss[final]={:.5}", losses[0], losses[EPOCHS / 2], losses[EPOCHS - 1]);

    let x_flat_eval: Vec<f32> = scaled.iter().flatten().map(|&v| v as f32).collect();
    let x_eval: Tensor<TrainB, 2> = Tensor::from_floats(x_flat_eval.as_slice(), &device_train).reshape([n, d_input]);
    let (logits, [attn1, attn2]) = trained.forward_with_attention(x_eval);
    let preds: Vec<f32> = logits.into_data().convert::<f32>().value;

    println!("\nPredicted vs. actual (scaled) declaration-count threat exposure:");
    for i in 0..n {
        println!("  {:<24} predicted={:.3}  actual={:.3}", cg.name_of(i), preds[i], y_flat[i]);
    }

    let layer1_attn = tensor_to_layer_attention(attn1, &cg.graph);
    let layer2_attn = tensor_to_layer_attention(attn2, &cg.graph);

    // Hutchinson County: the county with the highest live declaration
    // count that also sits in the Smokehouse Creek Fire corridor -- the
    // most consequential target to audit in this dataset.
    let target = cg.index_of("48233").ok_or("Hutchinson County (48233) not in graph")?;
    let trail = explain_prediction(&cg, &[layer1_attn, layer2_attn], target, 6);

    println!("\n{}", trail.report());
    println!("--- machine-readable audit record (trained-model attention, LIVE data) ---");
    println!("{}", trail.to_json_pretty());

    Ok(())
}
