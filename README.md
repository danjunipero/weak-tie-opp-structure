# Weak-Tie Opportunity Structure: A Spatial Methods Prototype
![Co-presence map of Gràcia](docs/assets/cpp_map_static-8.png)
Example static map png output for a run on Gracia, Barcelona at a radius of 800m.
```python
[Gracia, Barcelona, Spain] r≈800m
AUI* (grid)=0.455 (occ=0.77, L≈100m, gini_nonempty=0.409); AUI* (net)=0.615 (occ=0.92, step≈50m); NUS=1.000 (method=nodes_all); AccessGini_local=0.235; LUM_no_transit=0.723 (join_share=1.00);
route_entropy=1.097 (norm=0.999); kpaths_edge_cov=0.471; kpaths_len_cov=0.478;
dest_cell_H=0.996; dest_cat_H=0.320; CPP_gini=0.391; top10_share=0.368
```
Console output from the same run on Gracia.

**Purpose.** This repo contains the prototype Python script for my proposed Master’s dissertation. This project investigates whether the physical design of a neighbourhood structurally supports or inhibits the formation and maintenance of ‘weak-tie’ social networks. This script is the quantitative aspect of this research, using OpenStreetMap data for a given neighbourhood it calculates novel metrics that measure what I’m calling ‘Weak-Tie Opportunity Structure’.

This model draws from two key theorists:

- Jane Jacobs (1961): In *The death and life of great American cities*, Jacobs argues that safe, vibrant, socially cohesive neighbourhoods depend on a complex “ballet” of repeated, casual, and low-intensity encounters in the public space. This requires a diverse mix of uses, short-blocks, and distinctions between public and private space.
- Mark Granovetter (1973): Granovetter argues that ‘weak-ties’ (acquaintances) are more important than ‘strong-ties’ (family, close friends) for information flow, economic opportunity, and social mobilisation.

The full Master’s dissertation proposal is a mixed-methods study. This script forms the first part of this - a spatial, structural model that quantifies a neighbourhoods potential for spontaneous encounter. The second part will validate the quantitative model against quantitative/qualitative survey and interview data from a sample of residents. This will gather residents’ perceived safety, social trust, sense of belonging, and informal support networks.

**Core Idea.**
Fine-grain, ubiquitous, mixed-use, walkable urban areas should:

- Distribute everyday destinations across the network (ubiquity)
- Keep short-walk access high across most of the network
- Generate varied, redundant routes to different destinations
- Distribute co-presence across the network rather than bottleneck it through a single corridor

**These conditions increase opportunities for weak-tie formation and maintenance through a higher chance for spotaneous encounters with other residents.**

---

### Methods

Given a place name (e.g. Jericho, Oxford, UK) and a radius, the script:

1. Geocodes the place to a point and builds a circle study area of radius study_radius_m
2. Downloads the walk network within this circle
3. Pulls OSM points of interest (PoIs) and classifies them into social-use categories
4. Computes indicators of ubiquity, short-walk access, inequality of access, route diversity, path coverage, and co-presence concentration
5. Exports a summary CSV, a static map (PNG), and an interactive map (HTML)

---

### Metrics Cheat-Sheet

| Metric | Meaning | Better weak-tie opportunity structure if… |
| --- | --- | --- |
| AUI_raw (grid) | Evenness of PoIs across adaptive grid cells. | Higher (more even) |
| AUI* | Occupancy-weighted AUI: combines coverage (share of non-empty cells) with evenness among the non-empty cells. | Higher |
| AUI* (network) | AUI* computed by sampling along the walk network (not just Euclidean grid). | Higher |
| NUS | Network ubiquity share: fraction of sampled nodes with ≥ min_pois_for_service PoIs reachable within reach_radius_m. | Closer to 1.0 |
| Access Gini | Inequality of reachable PoIs across sampled nodes. | Lower |
| Route entropy (norm.) | Diversity among k-shortest plausible paths (per OD). | Closer to 1.0 |
| Edge coverage (share/length) | Share of simplified graph edges/length touched by plausible daily paths. | Higher |
| CPP Gini | Concentration of simulated co-presence on a few links. | Lower |
| Top-10% CPP share | Load carried by top decile of links. | Lower |
| LUM entropy | Land-use mix entropy across viz grid (excludes transit). | Higher |

---

## Pipeline

1. Study area
    
    Circle polygon of radius study_radius_m around place_name (metric CRS).
    
2. Network
    
    Walk network via OSMnx. This is projected and simplified to a DiGraph with one min-length edge per (u,v) for k-shortest-path calculation.
    
3. PoIs & classification
    
    OSM features reduced to representative points and classified into five social-use categories (plus transit and other). Deduping prevents overcounting multi-tagged places.
    
4. Grids
    - Viz grid: fixed cell size for LUM and residential origin sampling. There is an optional clip to streets via buffered walk-network.
    - Adaptive AUI grid: cell size chosen so expected PoIs per cell ≈ target_pois_per_cell, clamped by [cell_min_m, cell_max_m].
5. Indicators
    - AUI_raw / AUI* on the adaptive grid; optional AUI* on the network via along-edge sampling.
    - NUS: nus_method of either "nodes_all" (evaluate every node) or "segments_sampled" (sample points along edges every sample_step_m and snap to nearest node).
    - Access Gini from the reachable PoI counts.
    - Route entropy, coverage from k-shortest paths between sampled residential origins and sampled PoIs.
    - CPP via OD shortest paths with a toy diurnal weighting, yields per-edge loads and concentration stats.

---

## Configuration

Key fields from cfg:

- place_name (text string)
- study_radius_m (int, metres)
- crs_metric (default: "EPSG:3857") (web mercator projection)
- viz_cell_m, clip_grid_to_streets (bool), street_buffer_m, sjoin_max_dist_m
- aui_modes (list: "grid", "network"), target_pois_per_cell, cell_min_m, cell_max_m, dedupe_radius_m, aui_network_min_step_m
- NUS: nus_method ("nodes_all" recommended for stability), reach_radius_m, min_pois_for_service, sample_step_m
- Trip structure: k_paths, beta, max_origins, dests_per_origin
- Output: map_zoom, outdir
- seed for reproducibility

---

## Outputs

- outputs/cpp_map_static.png — static basemap with CPP coloring and PoI dots.
- outputs/cpp_map_interactive.html — interactive layers for CPP and PoIs.
- outputs/summary_metrics.csv — one row with metrics and config echoes for auditability.

---

## Interpreting results

- Ubiquity & access: High AUI* and high NUS, with low Access Gini, indicate broadly distributed day-to-day functionality and equitable short-walk reach.
- Varied encounter structure: High normalised route entropy and substantial edge coverage indicate many plausible, distinct routes to different amenities, promoting repeated but distributed overlap.
- Concentration and distribution: Lower CPP Gini and Top-10% share suggest co-presence is spread across the network rather than funnelled into a single corridor.
- Mix: Higher LUM entropy correlates with a stronger baseline for everyday social mixing but should be interpreted alongside ubiquity and route diversity.
