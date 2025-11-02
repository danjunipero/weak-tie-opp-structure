![v0.3.0 and v0.9.0 PNG output side by side](docs/assets/comparison.jpeg)
*PNG output of a run on Radcliffe Camera, Oxford. v0.3.0 (left) and v0.9.0 (right)*

Alongside the change log I’ve added a THINKING heading to show the relevance of changes to my research question. Version numbers are completely arbitrary I just thought they looked cool.

# v0.1.0 - Initial Prototype

**GOAL**

Compare “fine-grain” vs “nodal” neighbourhoods in terms of walkability and social interaction potential.

**WHAT WE MEASURED**

- Pulled OSM for walk network and points of interest (PoIs).
- Calculated:
    - Land-use mix entropy (LUM).
    - Amenity dispersion (early AUI approach, later replaced).
    - Route entropy from k-shortest paths.
    - Co-Presence Potential (CPP): how often shortest paths overlap on the same street segments.

**THINKING**

Trying to operationalise whether the street network design structurally allows residents to run into people throughout the area, or whether everyone is funnelled into one “high street” style node when going about their daily life within the area.


# v0.1.1 - OSM boundary handling

**CHANGES**

- Added fallback so that if Nominatim returns a point instead of a polygon, we generate a circular study area by buffering that point (buffer_m metres; default 500 m in current config).
- Otherwise we take the polygon, then buffer it by buffer_m.

**WHY**

OpenStreetMap is inconsistent about returning neighbourhood-scale polygons. We need a study area that reflects where people actually walk for daily needs, not just admin boundaries.


# v0.2.0 - Fixed network handling and route entropy

**CHANGES**

Network handling

- Projected the walk network to a metric CRS (EPSG:3857) so distances are in metres.
- Converted the OSMnx MultiDiGraph to a simplified DiGraph by collapsing parallel edges and keeping only the minimum-length edge per (u, v).
    - Reason: networkx.shortest_simple_paths expects a simple directed graph and did not behave with the raw MultiDiGraph.

Route entropy

- For each origin/destination (OD) pair, compute up to k_paths (default 3) shortest simple paths on that simplified graph.
- Convert each path length to a utility using an exponential decay with parameter beta, then normalise to get a probability distribution over those alternative routes.
- Compute Shannon entropy of that distribution.
- Also compute a normalised version norm_route_entropy by dividing by log(k_paths) so results are 0..1 and thus comparable across places.

**WHY**

Network handling

- Needed a stable base graph so we can actually compute multiple viable walking routes per OD without NetworkX messing up.

Route entropy

- No longer just looking at whether theres a shortest path, but whether a resident has multiple realistically short ways to get somewhere.
- High norm_route_entropy means trips don’t all flow through the same corridor.

**THINKING**

I’m treating route variety itself as structure. If many people can get to daily destinations through multiple viable streets, co-presence is more spatially distributed instead of squeezed into one channel.


# v0.3.0 - CPP and network coverage

**CHANGES**

- Built an OD (origin/destination) set:
    - Sampled residential origins (near buildings).
    - Sampled plausible destinations from the PoI set.
- For each OD pair:
    - Took the single shortest path along the projected walk network.
    - Weighted trips by a simple 24-hour profile (more movement in daytime than at 3 a.m.).
        - Disclaimer: these are placeholder toy diurnal weights
    - Accumulated this into an edge-by-hour load matrix F[edge, hour].
    - Summed over time to get total Co-Presence Potential (CPP) per edge.
- From this, we compute:
    - cpp_gini: Gini coefficient of simulated co-presence load across edges. Higher = social overlap is highly concentrated in a few links.
    - cpp_top10_share: share of total co-presence load carried by the top 10 percent most-loaded edges.
    - edge_cov_frac and edge_len_cov_frac: fraction of edges, and fraction of total edge length in the simplified graph, that are touched by the k-shortest paths from sampled OD pairs.

**WHY**

- If cpp_gini and cpp_top10_share are very high, that means most casual overlap is happening in only one or two corridors.
- If CPP is more even, routine presence is diffused across the neighbourhood instead of bottlenecked.

**THINKING**

CPP becomes a proxy for structural conditions that can support weak-tie maintenance: repeated incidental overlap in public space.


# v0.4.0 - Problems with early AUI and the redesign

**Old Problem**

- Imposed one fixed grid size across the whole study area (e.g. 150 m cells).
    - Counted PoIs per grid cell.
    - Took the Gini of those counts, then used (1 - Gini) as a score for “amenity spread,” which I was calling AUI.
- This was very sensitive to scale
    - If a study area is made larger with no additional PoIs, it included more zero cells.
    - Which pushed AUI down
- AUI also conflated coverage and clustering.
    - I needed the metric to differentiate between low coverage (amenities in one space) and high clustering (amenities within that space being clustered together).

**WHAT CHANGED**

- I no longer use a single fixed grid for AUI. Instead:
    - Estimate how many grid cells we “should” have by targeting roughly one PoI per cell (target_pois_per_cell).
    - Solve for a cell side length L so that the number of cells matches that target.
    - Clamp L between cell_min_m and cell_max_m so it stays at neighbourhood scale.
    - Build this adaptive grid only for AUI calculation.
- Assign deduplicated PoIs to these adaptive cells.
- For that adaptive grid, compute:
    - AUI_occupancy = share of cells that have at least one PoI (what fraction of the area actually participates).
    - Gini of PoI counts across all cells.
    - Gini of PoI counts across only the non-empty cells.
- Then produce two scores:
    - AUI_raw = 1 - Gini(all cells)
    - AUI_star = AUI_occupancy * (1 - Gini(non-empty cells))

**WHY**

AUI_star now answers:

1. Are amenities evenly shared among the locations that have any at all?
2. How much of the neighbourhood actually has amenities in the first place?

**THINKING**

- AUI* ≈ AUI on most sites. AUI* becomes discriminative in very clustered areas (shopping centres, large blocks). I’ve chosen to push it anyway for interpretability, comparability across densities, and robustness when within-cell clustering does happen.
- Amenity ubiquity is now explicitly about whether daily-life destinations are embedded through a meaningful share of the walkable network, instead of hoarded in one corridor.


# v0.4.1 - Tightened PoI dedupe

**WHAT CHANGED**

- Changed default PoI dedupe radius to 5 m.

**WHY**

- A previous version was merging PoIs within ~50 m to avoid counting the same venue twice when OSM had multiple tags for one place (happened in city centres and shopping centres).
- That ended up being too aggressive in fine-grain high street contexts, especially Victorian/terraced areas where multiple distinct shopfronts sit within a few metres.
- 50 m dedupe was fusing genuinely separate amenities into one PoI and reducing diversity.

**RESULTS**

- With a tighter (5 m) dedupe:
    - We keep distinct adjacent amenities instead of merging them.
    - Because there’ll be more PoIs after dedupe, the adaptive AUI cell size AUI_cell_L_m will be smaller.
    - AUI_occupancy and AUI_star adjust accordingly, usually dropping slightly, which is expected.

**THINKING**

Because the adaptive grid’s cell size AUI_cell_L_m changes with deduped PoI count, I’ll always report AUI_cell_L_m with AUI*.


# v0.5.0 - Further AUI* Refinement

**Old Problem**

- Original AUI* Calculation
    - AUI* = occ_share x (1-gini(non-empty))
- Cell size L came from the total polygon area
    - Found that industrial/disused land or large parcels of green space artificially inflated L
        - This collapsed AUI*

**WHAT CHANGED**

Added a new AUI* calculation, so that two AUI* values are produced. 

- AUI* (grid)
    - This is the same as the AUI* introduced in v0.4
- AUI* (net)
    - This is computed along the walk network itself, instead of the adaptive grid.
    - The network is sampled at a step length tied to the adaptive grid L
    - Occupancy/Evenness is measured over the network samples instead of planar cells

- AUI (raw) remains for transparency but is de-emphasised.

**WHY**

Now AUI should remain sensitive to how amenities are spread along actual streets in places with large empty land that could artificially skew the original grid-based AUI.

**THINKING**

This problem was found when looking at Saltley, Birmingham - a fine-grain victorian development with relatively high ubiquity. Saltley also borders an extensive disused industrial area (the *Northern Industrial Corridor*).  AUI* (grid) remained low due to those big blanks, but AUI* (net) is significantly higher, tracking the actual street spread.


# v0.6.0 - Introducing NUS (Network Ubiquity Share)

**WHAT CHANGED**

The grid-based AUI approach tells us about whether amenities are everywhere in principle. It doesn’t directly tell us whether they are walkably close to most of the network.

So I’ve now added NUS.

How NUS is calculated:

- Sample points along the walking network every sample_step_m metres.
- Snap each sampled point to its nearest graph node.
- For each sampled node:
    - Run Dijkstra’s algorithm on the walk network with cutoff reach_radius_m (default 180 m).
    - Count how many distinct PoIs are reachable within that walking distance.
    - Mark the node as “served” if reachable PoIs ≥ min_pois_for_service (default 3).
- Define NUS as the share of sampled nodes that are “served.”

**WHY**

NUS is asking what fraction of the actual walk network gives you short-walk access to multiple everyday destinations.

**THINKING**

NUS is directly interpretable.

If NUS = 0.42, that means about 42 percent of sampled points on the network are within ~180 m of at least three distinct amenities (using current defaults).


# v0.6.1 - Access_Gini_Local with NUS

**OLD PROBLEM**

I realised:

- A neighbourhood can have one high street in one part of the area.
- If half the network falls within 180 m of that one strip, NUS might look high.
- But access is still very spatially concentrated.

**FIXES**

I’ve now added local access inequality: access_gini_local.

**HOW**

- While computing NUS we already count, for each sampled node, how many PoIs are reachable within reach_radius_m
- We keep that full list (the hit counts)
- Now compute the Gini coefficient over those hit counts and call it access_gini_local

**INTERPRETATION**

- Low access_gini_local: access to daily destinations is fairly even.
    - High NUS + low access_gini_local = a walkable network with broadly shared access.
- High access_gini_local: access is uneven and clustered.
    - High NUS + high access_gini_local = node-dominant structure


# v0.6.2 - Cleaning up land-use mix (LUM) categories

**WHAT CHANGED**

I’ve refactored PoI classification so categories are mutually exclusive and socially meaningful. This was due to some overlap in previous iterations. Each PoI is now assigned one of:

- everyday_retail
    
    Daily goods and errands. Convenience stores, supermarket, bakery, butcher, greengrocer, chemist/pharmacy-as-shop, newsagent, off-licence, hairdresser/beauty, plus “errand infrastructure” like post office and bank.
    
- food_drink
    
    Places you eat, pick up prepared food, or socially linger. Cafés, takeaway, restaurants, pubs, bars.
    
- health
    
    Clinical services. GP, walk-in clinic, dentist, physio, optometrist/optician, etc.
    
- services
    
    Civic / collective / social infrastructure. Libraries, places of worship, community centres.
    
- education
    
    All levels and care. Nursery, childcare, school, college, university.
    
- transit
    
    Bus stop, station, platform, taxi rank, etc. We map this but it’s now excluded from land-use mix entropy.
    
- other
    
    Anything else
    

Then I do two things with that:

1. Land-use mix entropy (LUM_entropy_no_transit)
    
    Calculate Shannon entropy (0 to 1) across just the five socially important daily-life categories:
    
    ```python
    ["everyday_retail","food_drink","health","services","education"]
    ```
    
    Transit is excluded here on purpose.
    
2. Per-origin destination category entropy (mean_dest_cat_entropy)
    
    When we simulate residents’ errands, the script tracks how many distinct functional categories each origin actually touches. This tracks if most trips hit multiple categories (e.g. school + pharmacy + corner shop), or just go to fast food multiple times.
    

**THINKING**

This reframes “mixed use” into “can ordinary daily life pull you into different kinds of shared civic/commercial spaces,” which matters for weak-tie exposure.


# v0.6.3 - Polishing

**WHAT CHANGED**

- Cleaned and documented the script.
- Added inline explainers for each major step.

The script now writes out (CSV fields):

- AUI_raw, AUI_star, AUI_cell_L_m, AUI_occupancy
- NUS_share (this is NUS in the code)
- access_gini_local
- LUM_entropy_no_transit
- mean_route_entropy, norm_route_entropy
- edge_cov_frac, edge_len_cov_frac
- mean_dest_cell_entropy (spatial spread of chosen destinations per origin)
- mean_dest_cat_entropy (functional spread of chosen destinations per origin, using the categories above)
- cpp_gini, cpp_top10_share
- counts of PoIs (raw and deduped), number of grid cells, and run parameters for transparency.

The script now plots:

- Co-Presence Potential (CPP) across the walk network, with line colour and width scaled by simulated co-presence load.
- PoIs coloured by category.
- A readable legend.
- Street labels for the highest-CPP segments, where OSM has names.

The script now saves:

- A static PNG map (cpp_map_static.png).
- An interactive Leaflet HTML map (cpp_map_interactive.html) with layer toggles for amenity categories and CPP-weighted street segments.
- A CSV (summary_metrics.csv) with the indicators and run config.


# v0.7.0 - Added clip_grid_to_streets Toggle

**WHAT CHANGED**

- Added a toggle for whether the grid should be clipped to the street network.
    - Originally, the script would always clip to the street with a 30m buffer
- Added street_buffer_m to config, so buffer can be changed.

**WHY**

- While clipping the grid to streets was useful in areas with large amounts of disused industrial land or green space, I found that it skewed LUM in places where empty land is actually something.
    - E.g. Oxford College quads are no-access or controlled-access on OSM. Clipping to the street hollowed these out.
    - Undesirable since those quads are still places of social contact and use, unlike fenced-off empty plots.


# v0.8.0 - Refined NUS Further

**OLD PROBLEM**

- How NUS was calculated:
    - Dropped points along every edge every sample_step_m metres
    - Snapped those points to the nearest graph node
    - For each node, ran Dijkstra’s Algo up to reach_radius_m and counted reachable PoI nodes
    - Marked a point served if it’s reachables ≥ min_pois_for_service
    - NUS = share of sampled points that were served
        - Access_Gini uses the same counts
- This became a problem because
    - It was very sensitive to step size
    - Long edges got more sample points than short ones, which overweighted access on some corridors
    - Two (simulated) areas with identical structure but different edge segmentation produced different NUS metrics.

**WHAT CHANGED**

Now NUS

- Uses every node in the walk graph exactly once as an origin, instead of dropping points
- For each node, runs Dijkstra up to reach_radius_m and counts reachable PoIs and flags them as served just like previously
- So now, NUS = served nodes/total nodes
    - Access_gini now uses these counts too

**THINKING**

- Eliminates dependence on sample_step_m
- No length weighting - thus giving more representation to short streets and cul-de-sacs
- Allows for easier comparability between areas


# v0.9.0 - Replacing Buffer with Study Circle [BETA]

**WHAT CHANGED**

- Replaced polygon retrieval and double buffer (in case of points)
    - Now, the study area is simply a circle with radius study_radius_m centred around place_name
- Added CPP legend in the form of a colour bar to the static_png as well

**WHY**

- Adding buffer to polygons and points returned by OSM made the study radius
    - Difficult to adjust (particularly if study area needed to be smaller)
    - Difficult to control for (due to inconsistencies in polygon shape/size before the additional buffer)
