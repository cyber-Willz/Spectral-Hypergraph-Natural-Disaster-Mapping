# gis_audit

Integrates the **`spectral_hypergraph` / `nbsc`** stack (non-backtracking
spectral convolution, GCN/GAT/SAGE baselines — from
`nbsc_spectral_hypergraph_tilt_complete_tar.gz`) with the **`spectral_dqg` /
`krylov_ds`** stack's non-backtracking (Hashimoto/Ihara-zeta) walk machinery
(from `spectral_dqg_and_krylov_ds_gr_fix_completed_tar.gz`) to solve a
specific, named problem:

> High-stakes GIS decisions (urban zoning, flood mitigation, disaster
> routing) require auditability. Tracking how neighbor feature aggregation
> leads to a specific prediction remains difficult with standard GNN
> architectures.

## Why standard GNNs fail the auditability bar

A GCN/GAT/SAGE layer aggregates every neighbor's features into one fixed-size
vector at each layer. After 2-3 layers a county's representation is a
weighted mix of information from potentially hundreds of other counties —
but the *specific per-source accounting* needed to tell a zoning board "your
prediction was driven 40% by Fort Bend County's floodplain data, reaching you
through this 2-hop chain" is thrown away by construction. That's the gap
this crate closes.

## Architecture

```
county_adjacency.rs   Census County Adjacency File -> nbsc::graph::Graph
                       (topology: which counties border which)
nri_features.rs        FEMA National Risk Index CSV -> per-county feature
                       vectors (flood EAL, social vulnerability, resilience)
        |
        v
   nbsc's GAT/NBSC layers          <- the actual prediction model
   (per-layer attention alpha_ij)     (unchanged, reused as-is)
        |
        v
provenance.rs           attention rollout, RESTRICTED TO NON-BACKTRACKING
                         PATHS (reusing the Hashimoto-operator walk
                         structure from spectral_dqg::nonbacktracking) ->
                         ranked, JSON-exportable AuditTrail:
                         "which literal chain of counties, how much weight"
```

The non-backtracking restriction is the actual integration point between the
two source projects: rollout alone gives a scalar "how much did node j
matter"; walking only non-backtracking paths (no immediate A→B→A echo) is
exactly the Hashimoto/Ihara-zeta walk structure `spectral_dqg` already
implements for a different purpose (zeta-function evidentiary machinery),
reused here to keep the audit trail's paths genuinely informative instead of
dominated by trivial back-and-forth hops.

## Data sources (both free, public domain, no API key)

| Input | Source | Format |
|---|---|---|
| Graph topology | [US Census County Adjacency File](https://www.census.gov/geographies/reference-files/time-series/geo/county-adjacency.html) — legacy flat file: <https://www2.census.gov/geo/docs/reference/county_adjacency.txt> | tab-delimited (legacy) or pipe-delimited (2023+); both auto-detected |
| Node features | [FEMA National Risk Index, county table](https://hazards.fema.gov/nri/data-resources) | CSV, ~130 columns; this crate reads a named subset (`RISK_SCORE`, `EAL_VALT`, `RFLD_EALT`, `CFLD_EALT`, `SOVI_SCORE`, `RESL_SCORE`, `POPULATION`) chosen for the flood-mitigation/disaster-routing/zoning-risk use case |

`data/sample_county_adjacency.txt` and `data/sample_nri.csv` are a small,
hand-picked, **illustrative** 8-county Houston-area subset (real GEOIDs and
adjacency; the NRI numbers are placeholders, not official FEMA figures) so
the pipeline runs and is testable with zero network access. Run
`./fetch_real_data.sh` to pull the real nationwide adjacency file (and get
pointed at the real NRI download, which is served from a versioned data
portal rather than a flat URL) before doing any real analysis.

## Run it

```bash
cargo test -p gis_audit                                    # parser + non-backtracking-path unit tests
cargo run -p gis_audit --example flood_risk_audit           # structural (non-learned) attention fallback
cargo run -p gis_audit --example live_flood_audit --release # LIVE: actually trains a GAT and audits it

#The live defense-threat-mapping example, which pulls real FEMA disaster-declaration data
cargo run -p gis_audit --example live_defense_threat_mapping --release
```

> **Build this from the workspace root** (where `Cargo.lock` lives) so the
> pinned dependency versions are honored — `cd` into `gis_audit/` and run
> `cargo build` *without* the `-p gis_audit` flag is fine too, `cargo` walks
> up to find the workspace root either way, but don't run `cargo update`
> unless you mean to (see "A dependency note" below).

`flood_risk_audit` uses `LayerAttention::degree_normalized` (no training) so
the audit machinery is exercisable with zero setup. `live_flood_audit` is
the real end-to-end run: it actually trains a 2-layer `burn`-backed
`nbsc::gat_layer::GatLayer` regressor (predicting riverine-flood EAL from
each county's own NRI feature vector, propagated through the graph) on the
loaded data, then builds the audit trail from **that trained model's own
learned attention weights** — not a structural stand-in. Sample run against
the bundled 8-county illustrative subset:

```
Training 2-layer GAT (7 -> 8x2 -> 8 -> 1) for 300 epochs...
  loss[0]=0.23547  loss[mid]=0.00099
  loss[final]=0.00088

Predicted (scaled) riverine-flood EAL vs. actual (scaled):
  Fort Bend County TX      predicted=0.163  actual=0.154
  Brazoria County TX       predicted=0.199  actual=0.211
  ...

Audit trail for Harris County TX (48201) — 2 layer(s), paths up to 2 hops:
  Galveston County TX -> Harris County TX  (1 hop, 57.7% of explained influence, raw weight 0.8218)
  Brazoria County TX -> Galveston County TX -> Harris County TX  (2 hops, 23.9% of explained influence, raw weight 0.3399)
  Brazoria County TX -> Harris County TX  (1 hop, 9.9% of explained influence, raw weight 0.1417)
  ...
```

The `live` feature (on by default; disable with `--no-default-features` to
build just the parsers + audit math without pulling in `burn`) wires this
together in `src/live_model.rs`:

- `GatNet` — two stacked `nbsc::gat_layer::GatLayer`s + a linear head.
- `GatLayer::forward_with_attention` (added to `nbsc/src/gat_layer.rs`,
  non-breaking — `forward` now just calls it and discards the attention) —
  returns each head's real `[n, n]` softmax attention matrix alongside the
  layer's output.
- `train` — full-batch Adam over `burn`'s autodiff, real gradient descent
  against actual loaded data (see the loss trace above; not a canned
  demo — this drops from 0.235 to 0.0009 on the 8-county sample).
- `tensor_to_layer_attention` — converts a trained layer's dense attention
  tensor into the sparse, graph-restricted `provenance::LayerAttention`
  `explain_prediction` consumes.

## Wiring in the full nationwide dataset

`explain_prediction` doesn't care whether the attention came from a trained
GAT or the structural fallback — everything in `provenance.rs`,
`county_adjacency.rs`, and `nri_features.rs` is unchanged either way.
Swap `data/sample_*` for the real nationwide downloads (`./fetch_real_data.sh`)
and both examples run unmodified at national scale (the non-backtracking
path enumeration in `provenance.rs` is exact/exhaustive, appropriate at
county-graph depth/branching; see "Extending" below if you push `max_hops`
much higher on the full ~3,100-county graph).





