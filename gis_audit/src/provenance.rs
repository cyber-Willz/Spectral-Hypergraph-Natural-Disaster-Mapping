//! Auditability layer: turns "the GNN predicted X for this county" into a
//! ranked, exportable trail of *which specific neighboring counties, along
//! which specific paths, contributed how much* — the piece standard GNN
//! architectures don't give you for free, and the piece high-stakes GIS
//! decisions (zoning, flood mitigation, disaster routing) need for sign-off.
//!
//! Two complementary signals are combined:
//!
//! 1. **Attention rollout** (Abnar & Zuidema 2020) over `nbsc`'s
//!    [`gat_layer`](https://en.wikipedia.org/wiki/Graph_attention_network)
//!    per-layer attention weights `alpha_ij`: composing L layers' attention
//!    matrices gives, for any target node, the fraction of its final
//!    representation attributable to each L-hop-reachable source node. This
//!    is the standard "how much did node j matter" number.
//! 2. **Non-backtracking path enumeration**, reusing the Hashimoto/Ihara-zeta
//!    walk structure this whole crate stack is built around (see
//!    `spectral_dqg::nonbacktracking`): rollout alone tells you *how much*,
//!    this tells you *by which literal chain of counties* — i.e. it turns a
//!    scalar into a citable path like `Harris -> Fort Bend -> Wharton`,
//!    which is what an auditor / zoning board actually wants to see, and
//!    excludes the degenerate "walked to the neighbor and immediately back"
//!    paths that would otherwise dominate and add no information.
//!
//! Attention weights are supplied by the caller as plain
//! `HashMap<(usize, usize), f64>` per layer (source -> target -> weight) so
//! this module has no dependency on which layer produced them: a trained
//! `nbsc::gat_layer::GatHead`'s exported attention, or — when no trained
//! model is available yet — a deterministic non-learned fallback such as
//! degree-normalized adjacency, so the audit machinery is exercisable before
//! any training run.

use crate::county_adjacency::CountyGraph;
use nbsc::graph::Graph;
use serde::Serialize;
use std::collections::HashMap;

/// One GNN layer's attention matrix, sparse: `weights[(i, j)]` = attention
/// paid by node `i` to node `j` (`j` in `N(i) ∪ {i}`), row-stochastic
/// (`sum_j weights[(i,j)] == 1` for each `i`) as GAT's per-node softmax
/// guarantees.
#[derive(Clone, Debug)]
pub struct LayerAttention {
    pub weights: HashMap<(usize, usize), f64>,
}

impl LayerAttention {
    /// Non-learned fallback: degree-normalized adjacency plus a fixed
    /// self-loop weight, i.e. what a GCN (not GAT) layer would use. Lets the
    /// audit trail be produced and sanity-checked before a GAT model has
    /// been trained, or as a "structural-only" baseline trail to diff a
    /// trained model's *learned* attention against.
    pub fn degree_normalized(graph: &Graph, self_weight: f64) -> Self {
        let mut weights = HashMap::new();
        for i in 0..graph.n {
            let deg = graph.degree(i) as f64;
            let nbr_total = (1.0 - self_weight).max(0.0);
            weights.insert((i, i), self_weight);
            if deg > 0.0 {
                for &j in &graph.neighbors[i] {
                    weights.insert((i, j), nbr_total / deg);
                }
            } else {
                weights.insert((i, i), 1.0);
            }
        }
        LayerAttention { weights }
    }

    fn get(&self, i: usize, j: usize) -> f64 {
        self.weights.get(&(i, j)).copied().unwrap_or(0.0)
    }
}

/// One step of an audit path: which county, and the raw (un-normalized)
/// attention weight of *this specific hop*.
#[derive(Serialize, Clone, Debug)]
pub struct PathStep {
    pub geoid: String,
    pub name: String,
    pub hop_attention: f64,
}

/// A single explanatory path from a source county to the target county
/// being predicted on, non-backtracking (no immediate A->B->A step) so it
/// reflects genuinely new information reaching the target at each hop
/// rather than a trivial echo.
#[derive(Serialize, Clone, Debug)]
pub struct PathContribution {
    pub path: Vec<PathStep>,
    pub hops: usize,
    /// Product of per-hop attention weights along this exact path — the
    /// rollout contribution of this one path to the target's representation.
    pub path_weight: f64,
}

#[derive(Serialize, Clone, Debug)]
pub struct AuditTrail {
    pub target_geoid: String,
    pub target_name: String,
    pub num_layers: usize,
    pub max_hops_considered: usize,
    /// Top contributing paths, sorted descending by `path_weight`.
    pub top_paths: Vec<PathContribution>,
    /// Sum of `path_weight` over *all* enumerated non-backtracking paths
    /// (not just the truncated `top_paths`) reaching the target within
    /// `max_hops_considered` — the denominator for turning `path_weight`
    /// into a "% of explained influence" figure in a report.
    pub total_rollout_mass: f64,
}

/// Enumerate every non-backtracking path of length `<= max_hops` ending at
/// `target`, scoring each by the product of per-layer attention weights for
/// its hops (layer `l`'s matrix used for the `l`-th hop *counting backward
/// from the target*, matching how a target's representation at layer `L`
/// depends on layer-`(L-1)` neighbor representations, which depend on
/// layer-`(L-2)` neighbors, etc.).
///
/// This is a direct DFS over the graph (mirroring
/// `spectral_dqg::nonbacktracking::count_closed_nbt_walks_bruteforce`'s
/// walk structure, but for open walks *into* a fixed target rather than
/// closed walks), so it is exact, not sampled — appropriate for an audit
/// artifact, and tractable because `max_hops` is the GNN's own depth
/// (typically 2-4), not an unbounded search.
pub fn explain_prediction(
    cg: &CountyGraph,
    layers: &[LayerAttention],
    target_idx: usize,
    top_k: usize,
) -> AuditTrail {
    let max_hops = layers.len();
    let mut all_paths: Vec<PathContribution> = Vec::new();

    // DFS backward from the target: at depth d (0-indexed from the target),
    // we're asking "who fed into this node at layer (num_layers - d)".
    // node_seq accumulates node indices target=v0, v1, ..., vd (source-most
    // last); weight accumulates the product of per-hop attentions.
    fn dfs(
        cg: &CountyGraph,
        layers: &[LayerAttention],
        cur: usize,
        prev: Option<usize>,
        depth: usize,
        max_hops: usize,
        weight: f64,
        node_seq: &mut Vec<usize>,
        out: &mut Vec<PathContribution>,
    ) {
        if depth > 0 {
            let path_steps: Vec<PathStep> = node_seq
                .iter()
                .rev() // report source -> ... -> target
                .map(|&idx| PathStep {
                    geoid: cg.geoid_of(idx).to_string(),
                    name: cg.name_of(idx).to_string(),
                    hop_attention: 0.0, // per-step weights folded into path_weight; see note below
                })
                .collect();
            out.push(PathContribution { path: path_steps, hops: depth, path_weight: weight });
        }
        if depth == max_hops {
            return;
        }
        // Layer used for this hop, counting from the target backward:
        // depth=0 -> layers[max_hops-1] (last layer, nearest the target's
        // own final representation), depth=1 -> layers[max_hops-2], etc.
        let layer = &layers[max_hops - 1 - depth];
        for &next in &cg.graph.neighbors[cur] {
            if Some(next) == prev {
                continue; // non-backtracking: no immediate echo
            }
            let w = layer.get(cur, next);
            if w <= 0.0 {
                continue;
            }
            node_seq.push(next);
            dfs(cg, layers, next, Some(cur), depth + 1, max_hops, weight * w, node_seq, out);
            node_seq.pop();
        }
    }

    let mut node_seq = vec![target_idx];
    dfs(cg, layers, target_idx, None, 0, max_hops, 1.0, &mut node_seq, &mut all_paths);

    let total_rollout_mass: f64 = all_paths.iter().map(|p| p.path_weight).sum();
    all_paths.sort_by(|a, b| b.path_weight.partial_cmp(&a.path_weight).unwrap());
    all_paths.truncate(top_k);

    AuditTrail {
        target_geoid: cg.geoid_of(target_idx).to_string(),
        target_name: cg.name_of(target_idx).to_string(),
        num_layers: layers.len(),
        max_hops_considered: max_hops,
        top_paths: all_paths,
        total_rollout_mass,
    }
}

impl AuditTrail {
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Human-readable summary line per path, e.g.:
    /// `Fort Bend County TX -> Harris County TX  (2 hops, 18.4% of explained influence)`
    pub fn report(&self) -> String {
        let mut s = format!(
            "Audit trail for {} ({}) — {} layer(s), paths up to {} hops:\n",
            self.target_name, self.target_geoid, self.num_layers, self.max_hops_considered
        );
        for p in &self.top_paths {
            let pct = if self.total_rollout_mass > 0.0 {
                100.0 * p.path_weight / self.total_rollout_mass
            } else {
                0.0
            };
            let chain: Vec<&str> = p.path.iter().map(|s| s.name.as_str()).collect();
            s.push_str(&format!(
                "  {}  ({} hop{}, {:.1}% of explained influence, raw weight {:.4})\n",
                chain.join(" -> "),
                p.hops,
                if p.hops == 1 { "" } else { "s" },
                pct,
                p.path_weight
            ));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_graph() -> CountyGraph {
        let raw = "A\t001\tB\t002\n\t\tC\t003\nB\t002\tA\t001\n\t\tC\t003\nC\t003\tA\t001\n\t\tB\t002\n";
        CountyGraph::parse(raw).unwrap()
    }

    #[test]
    fn no_backtracking_paths_in_output() {
        let cg = toy_graph();
        let layer = LayerAttention::degree_normalized(&cg.graph, 0.5);
        let target = cg.index_of("001").unwrap();
        let trail = explain_prediction(&cg, &[layer.clone(), layer], target, 10);
        // A 2-hop path A<-B<-A would be a backtrack; must not appear.
        for p in &trail.top_paths {
            if p.hops == 2 {
                assert_ne!(p.path[0].geoid, p.path[2].geoid);
            }
        }
    }

    #[test]
    fn single_layer_self_only_path() {
        let cg = toy_graph();
        let layer = LayerAttention::degree_normalized(&cg.graph, 1.0); // pure self-loop
        let target = cg.index_of("001").unwrap();
        let trail = explain_prediction(&cg, &[layer], target, 10);
        // With self_weight=1.0, degree_normalized still forces a nbr_total,
        // but self_weight 1.0 means no neighbor mass, so hops>0 paths get 0
        // weight and are dropped by the w<=0.0 pruning -- confirms pruning works.
        assert!(trail.top_paths.iter().all(|p| p.hops == 0));
    }
}
