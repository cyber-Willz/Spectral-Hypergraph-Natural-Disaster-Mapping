//! Live, trained end-to-end pipeline: a real 2-layer GAT classifier
//! (`burn`-backed, `nbsc::gat_layer::GatHead`/`GatLayer`), trained on the
//! actual loaded NRI feature data, whose *trained* per-layer attention is
//! exported straight into [`crate::provenance::LayerAttention`] for
//! [`crate::provenance::explain_prediction`]. This replaces the
//! non-learned `LayerAttention::degree_normalized` fallback with the real
//! thing; that fallback is still useful (see its own docs) as a
//! structural sanity baseline to diff a trained model's attention against,
//! but it is not itself a "live run".
//!
//! Gated behind the `live` feature (on by default) because it pulls in
//! `burn`'s autodiff engine, which the pure data-loading/audit-math parts
//! of this crate deliberately don't need.

use crate::provenance::LayerAttention;
use nbsc::gat_layer::{GatLayer, GatLayerConfig};
use nbsc::graph::Graph;

use burn::module::Module;
use burn::nn::loss::MseLoss;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;

/// A minimal, real (not toy-random-weights-only) GAT classifier: two
/// stacked [`GatLayer`]s (the second single-headed, so its output width is
/// exactly `d_hidden`) feeding a linear head down to a scalar logit per
/// node. Depth 2 is deliberate: it matches `provenance::explain_prediction`
/// expecting one `LayerAttention` per hop of receptive field.
#[derive(Module, Debug)]
pub struct GatNet<B: Backend> {
    layer1: GatLayer<B>,
    layer2: GatLayer<B>,
    head: Linear<B>,
}

pub struct GatNetConfig {
    pub d_input: usize,
    pub d_hidden: usize,
    pub n_heads: usize,
}

impl GatNetConfig {
    pub fn init<B: Backend>(&self, graph: &Graph, device: &B::Device) -> GatNet<B> {
        let layer1 =
            GatLayerConfig::new(self.d_input, self.d_hidden, self.n_heads).init::<B>(graph, device);
        let layer2 =
            GatLayerConfig::new(self.d_hidden * self.n_heads, self.d_hidden, 1).init::<B>(graph, device);
        let head = LinearConfig::new(self.d_hidden, 1).init(device);
        GatNet { layer1, layer2, head }
    }
}

impl<B: Backend> GatNet<B> {
    /// Forward pass returning the final `[n, 1]` logits plus each layer's
    /// (head-averaged) `[n, n]` attention matrix, in layer order
    /// `[layer1_attn, layer2_attn]` — exactly the order
    /// `provenance::explain_prediction` expects (it walks backward from the
    /// last layer to the first).
    pub fn forward_with_attention(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, [Tensor<B, 2>; 2]) {
        let (h1, attn1) = self.layer1.forward_with_attention(x);
        let attn1_avg = average_heads(attn1);
        let (h2, attn2) = self.layer2.forward_with_attention(h1);
        let attn2_avg = average_heads(attn2);
        let logits = self.head.forward(h2);
        (logits, [attn1_avg, attn2_avg])
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward_with_attention(x).0
    }
}

fn average_heads<B: Backend>(heads: Vec<Tensor<B, 2>>) -> Tensor<B, 2> {
    let n_heads = heads.len() as f64;
    let mut it = heads.into_iter();
    let first = it.next().expect("GatLayer always has >=1 head");
    let sum = it.fold(first, |acc, t| acc + t);
    sum / n_heads
}

/// Converts a dense `[n, n]` attention tensor into the sparse,
/// graph-restricted [`LayerAttention`] `provenance::explain_prediction`
/// consumes — reading off only the `(i, j)` entries the graph structure
/// (and hence the GAT's own attention mask) actually allows, so no
/// numerical noise from masked-to-near-zero entries leaks into the audit
/// trail as spurious near-zero-weight paths.
pub fn tensor_to_layer_attention<B: Backend>(attn: Tensor<B, 2>, graph: &Graph) -> LayerAttention {
    let n = graph.n;
    let data: Vec<f32> = attn.into_data().convert::<f32>().value;
    let get = |i: usize, j: usize| data[i * n + j] as f64;

    let mut weights = std::collections::HashMap::new();
    for i in 0..n {
        weights.insert((i, i), get(i, i));
        for &j in &graph.neighbors[i] {
            weights.insert((i, j), get(i, j));
        }
    }
    LayerAttention { weights }
}

/// Trains `net` for `epochs` steps of full-batch Adam on a real-valued
/// per-node target (mean-squared-error regression head — chosen over a
/// classification head because NRI's own risk/EAL columns are naturally
/// continuous, so "predict this county's expected-loss-derived risk from
/// its neighbors' features" is the honest task, not an invented binary
/// label), returning the trained net and the per-epoch loss trace.
pub fn train<B: AutodiffBackend>(
    mut net: GatNet<B>,
    x: Tensor<B, 2>,
    y: Tensor<B, 2>,
    epochs: usize,
    lr: f64,
) -> (GatNet<B>, Vec<f32>) {
    let mut optim = AdamConfig::new().init();
    let mut losses = Vec::with_capacity(epochs);
    for _ in 0..epochs {
        let pred = net.forward(x.clone());
        let loss = MseLoss::new().forward(pred, y.clone(), burn::nn::loss::Reduction::Mean);
        losses.push(loss.clone().into_data().convert::<f32>().value[0]);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &net);
        net = optim.step(lr, net, grads);
    }
    (net, losses)
}
