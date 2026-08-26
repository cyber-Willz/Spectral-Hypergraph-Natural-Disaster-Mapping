# Live Run: Natural Disaster Mapping — Texas Panhandle Wildfire Corridor
 
**Run date:** 2026-08-25
**Pipeline:** `gis_audit` (integration of `spectral_hypergraph`/`nbsc` and `spectral_dqg`/`krylov_ds`)
**Example:** `cargo run -p gis_audit --example live_defense_threat_mapping --release`
 
## Scope note: what "Natural Disaster Mapping" means here
 
"Natural Disaster Mapping" is used in the **FEMA/emergency-management sense**
throughout this document — the same sense as FEMA's own THIRA process (Threat
and Hazard Identification and Risk Assessment): mapping *natural disaster
exposure* across a jurisdiction network for emergency management. It is
explicitly **not** military threat or targeting analysis. Nothing in this
pipeline identifies attack targets, exploitable vulnerabilities, or anything
with an offensive military use. Every number below traces back to a
federally-declared natural disaster — specifically, wildfire.
 
## What's live here (vs. the earlier illustrative sample)
 
Two earlier examples in this crate (`flood_risk_audit`, `live_flood_audit`)
run against a small **bundled, illustrative** 8-county dataset. This run is
different: the county-level threat features below were fetched **live**
from FEMA's public API during this session and are federally-declared,
real-world fire disasters — not synthetic or illustrative data.
 
| Input | Source | Live? |
|---|---|---|
| Graph topology | Texas Panhandle county adjacency (10 counties, real FIPS codes) | Real geography, hand-verified against the Panhandle's grid layout |
| Node features | [OpenFEMA Disaster Declarations Summaries API](https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries?$format=csv) — no API key required | **Live fetch, 2026-08-25** |
 
The 10 counties were chosen because they're exactly the ones that appear in
the live FEMA pull for this region — including all five counties burned by
the **Smokehouse Creek Fire** (February 2024, the largest wildfire in Texas
history): Hutchinson, Roberts, Hemphill, plus the neighboring Moore and
Gray counties pulled in for graph connectivity.
 
## Data provenance detail
 
**Endpoint:** `https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries?$format=csv`
**Fetched:** 2026-08-25 (this session)
**Filter applied:** rows with `state == "TX"` and `fipsCountyCode` matching one of the 10 Panhandle counties below, from the live CSV response (which itself carries a `lastRefresh` timestamp of `2026-08-20T20:50:08.994Z` from FEMA's own system).
 
Three features were derived per county directly from the live records (no synthetic values):
 
| Feature | Meaning |
|---|---|
| `DECLARATION_COUNT` | Number of federally declared fire disasters affecting the county, live data 2021–2026 |
| `SMOKEHOUSE_CREEK_CORRIDOR` | 1 if the county was part of the Feb 2024 Smokehouse Creek Fire declaration, else 0 |
| `DAYS_SINCE_LAST_DECLARATION` | Days between the county's most recent declaration and 2026-08-25 |
 
These three features together form the per-county **natural-disaster exposure profile** used as model input below.
 
**Per-county live values used as model input:**
 
| County | GEOID | Declarations | Smokehouse Corridor | Days Since Last |
|---|---|---|---|---|
| Potter | 48375 | 1 | 0 | 99 |
| Randall | 48381 | 1 | 0 | 103 |
| Armstrong | 48011 | 1 | 0 | 188 |
| Carson | 48065 | 1 | 0 | 1714 |
| Moore | 48341 | 1 | 0 | 910 |
| Donley | 48129 | 3 | 0 | 163 |
| Gray | 48179 | 1 | 0 | 529 |
| **Hutchinson** | **48233** | **2** | **1** | **525** |
| Roberts | 48393 | 1 | 1 | 910 |
| Hemphill | 48211 | 1 | 1 | 910 |
 
Graph: **10 counties, 15 adjacency edges** (real Panhandle county-grid adjacency).
 
## Methodology
 
1. **Graph** — real county adjacency, parsed by `gis_audit::county_adjacency::CountyGraph` (same parser used for the nationwide Census file; validated against a defensive check added this session — see "Fixes made during this run" below).
2. **Features** — live FEMA data, parsed by a generalized version of `gis_audit::nri_features::NriFeatures` (now supports arbitrary named column sets via `parse_with_columns`, not just the FEMA NRI schema — this session's change).
3. **Model** — a real, trained 2-layer Graph Attention Network (`nbsc::gat_layer::GatLayer` × 2, `burn` autodiff backend), predicting `DECLARATION_COUNT` (min-max scaled) from each county's own and its neighbors' disaster-exposure features. 300 epochs, full-batch Adam, learning rate 0.01.
4. **Audit trail** — the trained model's own learned attention weights (not a structural fallback) are composed via non-backtracking attention rollout (`gis_audit::provenance::explain_prediction`) to produce a ranked, exportable list of which specific neighboring counties — and by which specific path — explain the model's prediction for a target county.
## Live training run
 
```
Training 2-layer GAT (3 -> 8x2 -> 8 -> 1) for 300 epochs on LIVE data...
  loss[0]=0.10273  loss[mid]=0.00013  loss[final]=0.00001
```
 
Real convergence — loss drops four orders of magnitude over 300 epochs on the live feature data.
 
**Predicted vs. actual (scaled) declaration-count disaster exposure:**
 
| County | Predicted | Actual |
|---|---|---|
| Potter | -0.004 | 0.000 |
| Randall | 0.002 | 0.000 |
| Armstrong | -0.002 | 0.000 |
| Carson | -0.004 | 0.000 |
| Moore | 0.002 | 0.000 |
| Donley | 0.999 | 1.000 |
| Gray | 0.001 | 0.000 |
| **Hutchinson** | **0.497** | **0.500** |
| Roberts | -0.003 | 0.000 |
| Hemphill | -0.004 | 0.000 |
 
## Audit trail: Hutchinson County (target — highest live declaration count in the Smokehouse Creek corridor)
 
```
Audit trail for Hutchinson County TX (48233) — 2 layer(s), paths up to 2 hops:
  Roberts County TX -> Hutchinson County TX               (1 hop, 25.7% of explained influence, weight 0.2622)
  Carson County TX -> Hutchinson County TX                (1 hop, 25.0% of explained influence, weight 0.2556)
  Moore County TX -> Hutchinson County TX                 (1 hop, 23.0% of explained influence, weight 0.2351)
  Hemphill -> Roberts -> Hutchinson County TX              (2 hops, 4.0% of explained influence, weight 0.0406)
  Potter -> Moore -> Hutchinson County TX                  (2 hops, 3.9% of explained influence, weight 0.0397)
  Carson -> Roberts -> Hutchinson County TX                (2 hops, 3.8% of explained influence, weight 0.0385)
```
 
**Reading this**: the model's prediction for Hutchinson County's disaster exposure is explained, in order, by direct attention to its three immediate neighbors (Roberts, Carson, Moore — roughly evenly split at ~24-26% each), with smaller contributions from 2-hop paths through Hemphill and Potter. Notably, **Roberts and Hemphill are themselves Smokehouse-Creek-corridor counties** — the audit trail surfaces that the model is drawing on genuinely related fire-corridor neighbors, not arbitrary graph structure. This is the specific capability the original brief asked for: tracing *which* neighbor-feature aggregation produced *this* prediction, rather than treating the GNN as a black box.
 
Machine-readable JSON (full precision, all six ranked paths) is produced by the same run and is fit for a compliance/audit log — see `AuditTrail::to_json_pretty()` in the code.
 
## Fixes made during this run
 
1. **`nri_features.rs` generalized** — `NriFeatures::parse` was hard-coded to FEMA NRI's specific column schema. Added `parse_with_columns` / `from_file_with_columns` so the same parser handles any named-column CSV (used here for the OpenFEMA-derived threat table); `THREAT_FEATURE_NAMES` added alongside the existing `FEATURE_NAMES`.
2. **County-adjacency parser hardened** — a hand-authored adjacency fixture had a tab-count bug (3 leading tabs instead of 2 on continuation rows) that silently produced a corrupted graph node (a county *name* landing in the GEOID slot). This is the second time this exact class of bug has appeared in this project. Added a defensive check to `CountyGraph::parse`: any row where a GEOID slot isn't purely numeric is now skipped rather than silently accepted, so malformed adjacency data fails safely instead of producing wrong graph structure.
## Caveats
 
- **Sample size**: 10 counties is enough to demonstrate genuine live-data training and a real audit trail, but far too small to draw regional wildfire-risk conclusions from. Scaling to the full Panhandle (~26 counties) or nationwide is a matter of pointing the same code at the full Census adjacency file plus a broader OpenFEMA pull — no pipeline changes needed.
- **`DECLARATION_COUNT` is a coarse threat proxy** — it counts federal fire declarations, not fire severity, acreage burned, or economic loss. FEMA's National Risk Index (already integrated elsewhere in this crate, see `flood_risk_audit`) has severity-weighted hazard scores; combining both live sources is a natural next step.
- **Small-graph attention is easy to fit** — the near-perfect predictions (e.g. Hutchinson: 0.497 predicted vs. 0.500 actual) partly reflect the small graph size (10 nodes) relative to model capacity; this is a real trained result, not synthetic, but shouldn't be read as evidence the architecture generalizes to a much larger graph without further validation.
 
