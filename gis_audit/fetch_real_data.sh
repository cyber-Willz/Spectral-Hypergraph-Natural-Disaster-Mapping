#!/usr/bin/env bash
# Downloads the real, full-nationwide, free/public-domain datasets this
# crate is designed for, replacing the small illustrative samples in data/.
# Run this yourself (the sandbox this crate was built in has no network
# access to census.gov/fema.gov, only to crates.io/github for building).
set -euo pipefail
cd "$(dirname "$0")"

echo "Fetching US Census county adjacency file (legacy, tab-delimited, stable format)..."
curl -fsSL -o data/county_adjacency_full.txt \
  https://www2.census.gov/geo/docs/reference/county_adjacency.txt

echo
echo "FEMA National Risk Index has no single stable direct-download URL (it's"
echo "versioned and served from a data portal, not a flat file server), so:"
echo "  1. Open https://hazards.fema.gov/nri/data-resources"
echo "  2. Under 'Counties', download the national CSV table"
echo "  3. Save it as data/nri_full.csv"
echo
echo "Then point CountyGraph::from_file / NriFeatures::from_file at the"
echo "*_full.* files instead of the sample_* files."
