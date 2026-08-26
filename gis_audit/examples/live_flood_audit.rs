//! **Live, end-to-end run**: loads the real county graph + NRI feature
//! data, actually trains a 2-layer `burn`-backed GAT regressor
//! (`nbsc::gat_layer`) on it, then explains a prediction using that
//! trained model's *own learned* attention weights — not the non-learned
//! `LayerAttention::degree_normalized` structural fallback the earlier
//! `flood_risk_audit` example used.
//!
//! Run with (the `live` feature is on by default):
//!   cargo run -p gis_audit --example live_flood_audit --release
//!
//! Swap `data/sample_*` for the real nationwide downloads (see
//! `fetch_real_data.sh`) to run this against the whole US.

use gis_audit::live_model::{tensor_to_layer_attention, train, GatNetConfig};
use gis_audit::provenance::explain_prediction;
use gis_audit::{CountyGraph, NriFeatures};

use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

type TrainB = Autodiff<NdArray<f32>>;

/// Feature index of the target being predicted: RFLD_EALT (riverine-flood
/// Expected Annual Loss), the column this whole crate's flood-mitigation
/// use case is built around. See `nri_features::FEATURE_NAMES`.
const TARGET_COL: usize = 2;
const D_HIDDEN: usize = 8;
const N_HEADS: usize = 2;
const EPOCHS: usize = 300;
const LR: f64 = 0.01;

/// Min-max scale each feature column to [0, 1] independently, returning
/// the scaled matrix. Raw NRI dollar-valued columns span orders of
/// magnitude (thousands to hundreds of millions); training a linear+GAT
/// stack directly on that range is numerically unstable, so this is a
/// real (not decorative) preprocessing step, not a simplification made to
/// dodge the live run.
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
    // Resolve relative to this crate's directory, not the process's cwd --
    // see the comment in flood_risk_audit.rs for why.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let adjacency_path = format!("{manifest_dir}/data/sample_county_adjacency.txt");
    let nri_path = format!("{manifest_dir}/data/sample_nri.csv");

    // --- 1. Load the real graph topology + real (sample) NRI features ---
    let cg = CountyGraph::from_file(&adjacency_path)?;
    let nri = NriFeatures::from_file(&nri_path)?;
    let raw_features = nri.align_to(&cg.index_to_geoid);
    println!("Loaded {} counties, {} adjacency edges.", cg.graph.n, cg.graph.m());

    let scaled = min_max_scale(&raw_features);
    let n = cg.graph.n;
    let d_input = scaled[0].len();

    let device_train = Default::default();
    let x_flat: Vec<f32> = scaled.iter().flatten().map(|&v| v as f32).collect();
    let x_train: Tensor<TrainB, 2> = Tensor::from_floats(x_flat.as_slice(), &device_train).reshape([n, d_input]);
    let y_flat: Vec<f32> = scaled.iter().map(|row| row[TARGET_COL] as f32).collect();
    let y_train: Tensor<TrainB, 2> = Tensor::from_floats(y_flat.as_slice(), &device_train).reshape([n, 1]);

    // --- 2. Train a real 2-layer GAT regressor to predict riverine-flood
    //        EAL from (all of) each county's NRI feature vector, i.e. a
    //        stand-in for "predict this county's flood-mitigation
    //        criticality from neighborhood context" ---
    let cfg = GatNetConfig { d_input, d_hidden: D_HIDDEN, n_heads: N_HEADS };
    let net = cfg.init::<TrainB>(&cg.graph, &device_train);
    println!("Training 2-layer GAT ({d_input} -> {D_HIDDEN}x{N_HEADS} -> {D_HIDDEN} -> 1) for {EPOCHS} epochs...");
    let (trained, losses) = train(net, x_train, y_train, EPOCHS, LR);
    println!(
        "  loss[0]={:.5}  loss[mid]={:.5}",
        losses[0],
        losses[EPOCHS / 2],
    );
    println!("  loss[final]={:.5}", losses[EPOCHS - 1]);

    // --- 3. Run inference with the trained net, returning the model's
    //        *own learned* attention matrices alongside the predictions.
    //        (Reusing the same Autodiff<NdArray> backend for inference too
    //        -- no separate backend swap needed; we simply never call
    //        `.backward()` again.) ---
    let x_flat_eval: Vec<f32> = scaled.iter().flatten().map(|&v| v as f32).collect();
    let x_eval: Tensor<TrainB, 2> = Tensor::from_floats(x_flat_eval.as_slice(), &device_train).reshape([n, d_input]);
    let (logits, [attn1, attn2]) = trained.forward_with_attention(x_eval);

    let preds: Vec<f32> = logits.into_data().convert::<f32>().value;
    println!("\nPredicted (scaled) riverine-flood EAL vs. actual (scaled):");
    for i in 0..n {
        println!(
            "  {:<24} predicted={:.3}  actual={:.3}",
            cg.name_of(i),
            preds[i],
            y_flat[i]
        );
    }

    // --- 4. Convert the trained model's real attention into
    //        LayerAttention and produce the audit trail ---
    let layer1_attn = tensor_to_layer_attention(attn1, &cg.graph);
    let layer2_attn = tensor_to_layer_attention(attn2, &cg.graph);

    let target = cg.index_of("48201").ok_or("Harris County (48201) not in graph")?; // Harris County, TX
    let trail = explain_prediction(&cg, &[layer1_attn, layer2_attn], target, 5);

    println!("\n{}", trail.report());
    println!("--- machine-readable audit record (trained-model attention) ---");
    println!("{}", trail.to_json_pretty());

    Ok(())
}
