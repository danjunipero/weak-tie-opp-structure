"""
Neighbourhood walkability / weak-tie opportunity structure
----------------------------------------------------------

This script is my working methods prototype for my intended Masters diss research.
It does four things for a given neighbourhood:

1. Build a circle-of-radius study area around a named place and pull the walk network.
   I explicitly use a circle because OpenStreetMap's administrative boundaries are limiting (see CHANGES.md for more on this).

2. Classify everyday destinations (shops, schools, health, worship, etc.) into meaningful social-use categories instead of just land use.

3. Quantify a bunch of structure indicators:
   - amenity ubiquity across the network (AUI*)
   - network ubiquity / short-walk access (NUS)
   - inequality of access (Access Gini)
   - land-use mix entropy 
   - route entropy and path coverage (whether errands force everyone down one corridor)
   - co-presence concentration (CPP gini)

   These will all feed my theoretical question: does the neighbourhood’s design structurally support repeated 
   casual encounters and overlap in public space (Jacobs, 1961), or is social contact canalised into a single 
   linear node.

4. Produces: a summary CSV row (with metrics + parameters), a static map, and an interactive map.

This is not a polished library. These indicators are not presented as causal claims. At this stage 
they serve as spatial instruments for what I'm calling ‘weak-tie opportunity structure'. These spatial 
indicators will be compared with resident survey data on belonging, trust, perceived safety, and 
informal support. The goal is to test whether neighbourhood form predicts stronger weak-tie networks 
in the Granovetter (1973) sense, i.e. frequent low-intensity ties that enable information flow and social backup.
"""

# standard stack
import os
import math
import random
import warnings
from collections import Counter
import inspect

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe
import contextily as cx
import folium, branca

from shapely.geometry import Point, box, MultiPoint
from shapely.ops import unary_union
import shapely
import osmnx as ox
from osmnx import distance as oxdist

warnings.filterwarnings("ignore", message="The inferred zoom level")
ox.settings.log_console = True


# -------------------------------------------------------------------------------------------------------------------------------
# config
# -------------------------------------------------------------------------------------------------------------------------------

cfg = {
    # neighbourhood for analysis: Neighbourhood, City, Country works best (include postcodes for address points)
    "place_name": "Gracia, Barcelona, Spain",

    # study radius (centered on place_name)
    "study_radius_m": 800,

    # projected CRS for metric work
    "crs_metric": "EPSG:3857",

    # general randomness control (for sampling residential origins etc.)
    "seed": 67,

    # grid for visualisation + LUM join (fixed resolution)
    "viz_cell_m": 150,
    "clip_grid_to_streets": True,  # toggle whether the grid should clip to the network or include everything
    "street_buffer_m": 30,         # how far beyond the network to include
    "sjoin_max_dist_m": 150,

    # adaptive AUI controls (amenity ubiquity grid, which adapts to local needs)
    "aui_modes": ["grid", "network"],
    "target_pois_per_cell": 1.0,   # how many amenities per cell
    "cell_min_m": 100,             # clamp so we don't go too small for dense areas
    "cell_max_m": 220,             # clamp so we don't go too big for sparse areas
    "dedupe_radius_m": 5,          # collapse duplicate OSM points but (try to) keep adjacent shopfronts distinct
    "aui_network_min_step_m": 40,  # minimum step length for network sampling in AUI_star_net

    # network ubiquity / short-walk access
    "nus_method": "nodes_all",     # nodes_all uses all nodes, change to "segments_sampled" to sample along network instead
    "reach_radius_m": 180,         # how far should residents be expected to walk
    "min_pois_for_service": 3,     # node is served if >= this many PoIs are reachable
    "sample_step_m": 60,           # spacing when sampling along the walk network for NUS

    # trip structure / co-presence
    "k_paths": 3,                  # up to K plausible short paths per origin/dest
    "beta": 0.01,                  # softness in penalising longer alternatives in entropy calc
    "max_origins": 60,             # how many residential origins to sample
    "dests_per_origin": 3,         # distinct destinations sampled per origin

    # map output
    "map_zoom": 18,
    "outdir": "outputs",
}

random.seed(cfg["seed"])
np.random.seed(cfg["seed"])
os.makedirs(cfg["outdir"], exist_ok=True)


# -------------------------------------------------------------------------------------------------------------------------------
# defining utilities
# -------------------------------------------------------------------------------------------------------------------------------

def gini(arr):
    """
    standard Gini coefficient.
    """
    # I use it to talk about spatial inequality of amenities, and also to talk 
    # about how concentrated co-presence is on a few street segments.
    
    x = np.asarray(arr, dtype=float)
    if x.size == 0:
        return np.nan
    if np.amin(x) == 0:
        x = x + 1e-12
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    return (n + 1 - 2 * (cumx.sum() / cumx[-1])) / n


def shannon_entropy(counts, normalise_over=None):
    """
    Normalised Shannon entropy.

    counts: iterable of counts per category.
    normalise_over: if I pass an int here (e.g. 5 categories total),
                    I divide by log(normalise_over) so the result is 0..1.
    """
    
    # I use this for:
    # - land-use mix (LUM)
    # - per-origin diversity of destination categories
    # - per-origin spatial spread of destinations
    #
    # Interpretation:
    # 0 means totally concentrated (everything in one category / one cell).
    # 1 means maximally even across the categories considered.

    c = np.array(list(counts), dtype=float)
    tot = c.sum()
    if tot == 0:
        return 0.0
    p = c / tot
    p = p[p > 0]
    H = -(p * np.log(p)).sum()
    if normalise_over and normalise_over > 1:
        H /= math.log(normalise_over)
    return float(H)


def add_basemap_portable(ax, source, **kwargs):
    # Drop kwargs that this contextily version doesn’t support (kept getting warnings)
    sig = inspect.signature(cx.add_basemap)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cx.add_basemap(ax, source=source, **allowed)


def circle_study_polygon(place_name, radius_m, crs_metric):
    """
    Geocode the place name to a circle polygon (EPSG:4326) with radius study_radius_m
    Returns:
        circle_ll  (study polygon in EPSG:4326)
        circle_m   (same polygon in crs_metric)
    """
    try:
        lat, lon = ox.geocode(place_name)
    except Exception:
        raise ValueError(f"Could not geocode: {place_name}")
    pt_ll = gpd.GeoSeries([Point(lon, lat)], crs=4326)
    circle_m = pt_ll.to_crs(crs_metric).buffer(radius_m).iloc[0]
    circle_ll = gpd.GeoSeries([circle_m], crs=crs_metric).to_crs(4326).iloc[0]
    return circle_ll, circle_m


def build_walk_graph(study_poly_ll, crs_metric):
    """
    Pull the walking network inside the study polygon, project it,
    and also build a simplified DiGraph with a single (u,v) edge of
    minimum length. networkx's k-shortest-paths really don't like MultiDiGraphs.
    """
    G_multi = ox.graph_from_polygon(study_poly_ll, network_type="walk")
    G_proj = ox.project_graph(G_multi, to_crs=crs_metric)
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_proj)

    # collapse multigraph edges to one edge per (u,v), keeping the shortest
    G_simple = nx.DiGraph()
    G_simple.add_nodes_from((n, G_proj.nodes[n]) for n in G_proj.nodes())
    for u, v, k, data in G_proj.edges(keys=True, data=True):
        dist = float(data.get("length", 1.0))
        if G_simple.has_edge(u, v):
            if dist < G_simple[u][v]["length"]:
                G_simple[u][v]["length"] = dist
        else:
            G_simple.add_edge(u, v, length=dist)

    return G_proj, G_simple, nodes_gdf, edges_gdf


def classify_poi(row):
    """
    Classify raw OSM features into functional categories that matter for everyday social contact.

    - everyday_retail: errands, shops, hair/barber, post office, etc.
    - food_drink: cafes, takeaway, pubs, bars, restaurants.
    - health: GP, dentist, clinic, pharmacy etc.
    - services: libraries, places of worship, community centres 
    - education: from nursery to university
    - transit: bus stop / station etc. Kept but not counted in LUM.
    - other: anything else
    """
    a  = row.get("amenity")
    s  = row.get("shop")
    hc = row.get("healthcare")
    hw = row.get("highway")
    rw = row.get("railway")
    pt = row.get("public_transport")

    # food / drink / hangout
    if a in {"cafe","fast_food","restaurant","bar","pub"}:
        return "food_drink"

    # health (formal care)
    if a in {"clinic","doctors","dentist","pharmacy"}:
        return "health"
    if hc in {"pharmacy","doctor","dentist","clinic","physiotherapist","optometrist","optician"}:
        return "health"
    if s in {"chemist"}:
        return "health"

    # education (formal + childcare)
    if a in {"school","college","university","nursery","childcare"}:
        return "education"

    # services / civic / social infrastructure
    if a in {"library","place_of_worship","community_centre"}:
        return "services"

    # everyday retail / errands
    if s in {
        "convenience","supermarket","greengrocer","butcher","bakery",
        "newsagent","alcohol","kiosk", "off_licence"
        "hairdresser","beauty","cosmetics","clothes", "retail",
    }:
        return "everyday_retail"
    if a in {"bank","post_office"}:
        return "everyday_retail"

    # transit (kept for mapping + CPP, excluded from land-use mix)
    if (
        (hw == "bus_stop") or
        (rw in {"station"}) or
        (pt in {"station","platform","stop_position"}) or
        (a in {"bus_station","ferry_terminal","taxi"})
    ):
        return "transit"

    return "other"
    # TODO: investigate and fix to account 4 potential borderline cases due to inconsistent OSM tagging. combine edu and services?
    # TODO: Exclude transit from CPP PoI list?


def representative_point_any(geom):
    """
    Reduce polygons/lines to a single point for distance calcs,
    but DON'T lose the fact that many amenities are mapped as areas.
    """
    if geom is None or geom.is_empty:
        return None
    gt = geom.geom_type
    if gt == "Point":
        return geom
    try:
        if gt in ("Polygon","MultiPolygon","GeometryCollection"):
            return geom.representative_point()
        if gt in ("LineString","MultiLineString"):
            return geom.interpolate(0.5, normalized=True)
    except Exception:
        pass
    return geom.centroid


def build_visual_grid(area_poly_m, edges_gdf, cell_m, crs_metric, clip_to_streets=True, street_buffer_m=30):
    """
    A fixed grid (cell_m by cell_m) clipped to the study polygon and also roughly clipped to streets. I use this for:
    - LUM: land-use mix entropy
    - residential origin sampling later

    This is NOT the adaptive AUI grid
    """
    minx, miny, maxx, maxy = area_poly_m.bounds
    xs = np.arange(minx, maxx, cell_m)
    ys = np.arange(miny, maxy, cell_m)
    grid_cells = [box(x, y, x + cell_m, y + cell_m) for x in xs for y in ys]
    grid = gpd.GeoDataFrame(geometry=grid_cells, crs=crs_metric)

    # clip to area and street buffer toggle
    if clip_to_streets:
        try:
            walk_union = shapely.union_all(list(edges_gdf.geometry))
        except Exception:
            walk_union = edges_gdf.geometry.unary_union
        grid = grid[grid.intersects(area_poly_m) & grid.intersects(walk_union.buffer(street_buffer_m))].copy()
    else:
        walk_union = None
        grid = grid[grid.intersects(area_poly_m)].copy()

    grid["cell_id"] = range(len(grid))
    return grid, walk_union


def compute_lum(dest_pts, grid_viz, max_near_m):
    """
    Land-use mix entropy (0..1, higher = more balanced mix) across the viz grid.
    I only consider categories that matter for daily/social life.
    I explicitly exclude 'transit' because bus stops are really not a land use.
    """
    cats_for_lum = ["everyday_retail","food_drink","health","services","education"]
    if dest_pts.empty or grid_viz.empty:
        return 0.0, 0.0

    try:
        join_viz = gpd.sjoin_nearest(
            dest_pts[["geometry","cat"]],
            grid_viz[["cell_id","geometry"]],
            how="left",
            max_distance=max_near_m,
            distance_col="d"
        ).dropna(subset=["cell_id"])
    except Exception:
        try:
            join_viz = gpd.sjoin(
                dest_pts[["geometry","cat"]],
                grid_viz[["cell_id","geometry"]],
                predicate="within",
                how="left"
            ).dropna(subset=["cell_id"])
        except Exception:
            rows = []
            for i, g in dest_pts.iterrows():
                dists = grid_viz.geometry.distance(g.geometry)
                j = int(np.argmin(dists.values))
                if float(dists.iloc[j]) <= max_near_m:
                    rows.append({"cell_id": int(grid_viz.cell_id.iloc[j]), "cat": g["cat"]})
            join_viz = pd.DataFrame(rows)

    share_joined = len(join_viz) / len(dest_pts) if len(dest_pts) else 0.0
    cat_counts = join_viz["cat"].value_counts().reindex(cats_for_lum, fill_value=0).values
    lum_entropy = shannon_entropy(cat_counts, normalise_over=len(cats_for_lum))
    return float(lum_entropy), float(share_joined)


def make_adaptive_aui_grid(area_poly_m, walk_union, dest_pts_dedup, cfg):
    """
    Build the adaptive grid for the Amenity Ubiquity Index (AUI*).
    """
    area_m2 = float(gpd.GeoSeries([area_poly_m], crs=cfg["crs_metric"]).area.iloc[0])
    n_pois = max(1, len(dest_pts_dedup))
    n_cells_target = max(1, int(n_pois / max(0.1, cfg["target_pois_per_cell"])))
    L = max(cfg["cell_min_m"], min(cfg["cell_max_m"], math.sqrt(area_m2 / n_cells_target)))
    
    minx, miny, maxx, maxy = area_poly_m.bounds
    xs2 = np.arange(minx, maxx, L)
    ys2 = np.arange(miny, maxy, L)
    grid_cells = [box(x, y, x + L, y + L) for x in xs2 for y in ys2]
    grid_aui = gpd.GeoDataFrame(geometry=grid_cells, crs=cfg["crs_metric"])

    if walk_union is not None:
        grid_aui = grid_aui[
            grid_aui.intersects(area_poly_m) &
            grid_aui.intersects(walk_union.buffer(cfg["street_buffer_m"]))
        ].copy()
    else:
        grid_aui = grid_aui[grid_aui.intersects(area_poly_m)].copy()

    grid_aui["cell_id"] = range(len(grid_aui))

    if dest_pts_dedup.empty or grid_aui.empty:
        return {"AUI_raw": 0.0, "AUI_star": 0.0, "AUI_cell_L_m": float(L),
                "AUI_occupancy": 0.0, "nonempty_gini": np.nan, "grid_aui": grid_aui}

    try:
        join_aui = gpd.sjoin_nearest(
            dest_pts_dedup[["geometry"]],
            grid_aui[["cell_id","geometry"]],
            how="left",
            max_distance=L*math.sqrt(2)/2
        ).dropna(subset=["cell_id"])
    except Exception:
        rows2 = []
        for i, g in dest_pts_dedup.iterrows():
            d = grid_aui.geometry.distance(g.geometry)
            j = int(np.argmin(d.values))
            rows2.append({"cell_id": int(grid_aui.cell_id.iloc[j])})
        join_aui = pd.DataFrame(rows2)

    counts = join_aui.groupby("cell_id").size().reindex(grid_aui.cell_id, fill_value=0).values
    nonempty_counts = counts[counts > 0]
    occ_share = float((counts > 0).mean())
    AUI_raw = 1 - gini(counts)
    if nonempty_counts.size:
        AUI_star = occ_share * (1 - gini(nonempty_counts))
        ne_gini = gini(nonempty_counts)
    else:
        AUI_star, ne_gini = 0.0, np.nan

    return {
        "AUI_raw": float(AUI_raw),
        "AUI_star": float(AUI_star),
        "AUI_cell_L_m": float(L),
        "AUI_occupancy": float(occ_share),
        "nonempty_gini": float(ne_gini) if not np.isnan(ne_gini) else np.nan,
        "grid_aui": grid_aui
    }

def make_network_aui(G_proj, dest_pts_dedup, L, cfg):
    """
    Now computing AUI_star via network sampling
    """
    if dest_pts_dedup.empty or G_proj.number_of_nodes() == 0:
        return {"AUI_star_net": 0.0, "AUI_raw_net": 0.0, "AUI_sample_step_m": None,
                "AUI_occupancy_net": 0.0, "nonempty_gini_net": np.nan,
                "n_samples": 0}

    step = max(cfg["aui_network_min_step_m"], L/2)

    edges_gdf = ox.graph_to_gdfs(G_proj, nodes=False, edges=True)
    seg_pts = []
    for _, r in edges_gdf.iterrows():
        geom = r.geometry
        if geom.length < 1:
            continue
        n_seg = max(1, int(geom.length // step))
        for t in np.linspace(0, 1, n_seg, endpoint=True):
            seg_pts.append(geom.interpolate(t, normalized=True))

    if not seg_pts:
        return {"AUI_star_net": 0.0, "AUI_raw_net": 0.0, "AUI_sample_step_m": float(step),
                "AUI_occupancy_net": 0.0, "nonempty_gini_net": np.nan, "n_samples": 0}

    samples = gpd.GeoDataFrame(geometry=seg_pts, crs=cfg["crs_metric"])
    sample_nodes = oxdist.nearest_nodes(G_proj, samples.geometry.x.values, samples.geometry.y.values)

    poi_nodes = oxdist.nearest_nodes(
        G_proj, dest_pts_dedup.geometry.x.values, dest_pts_dedup.geometry.y.values
    ) if len(dest_pts_dedup) else np.array([])
    poi_node_set = set(int(n) for n in np.atleast_1d(poi_nodes))

    radius = L/2
    hit_counts = []
    for sn in np.atleast_1d(sample_nodes):
        dist_map = nx.single_source_dijkstra_path_length(G_proj, int(sn), cutoff=radius, weight="length")
        hits_here = sum(1 for node_id in dist_map.keys() if node_id in poi_node_set)
        hit_counts.append(hits_here)

    counts = np.array(hit_counts, dtype=float)
    nonempty = counts[counts > 0]
    occ_share = float((counts > 0).mean())

    AUI_raw_net = 1 - gini(counts)
    if nonempty.size:
        AUI_star_net = occ_share * (1 - gini(nonempty))
        ne_gini = gini(nonempty)
    else:
        AUI_star_net, ne_gini = 0.0, np.nan

    return {
        "AUI_star_net": float(AUI_star_net),
        "AUI_raw_net": float(AUI_raw_net),
        "AUI_sample_step_m": float(step),
        "AUI_occupancy_net": float(occ_share),
        "nonempty_gini_net": float(ne_gini) if not np.isnan(ne_gini) else np.nan,
        "n_samples": int(len(sample_nodes))
    }
    
def compute_network_ubiquity_and_access(G_proj, edges_gdf, dest_pts_dedup, cfg):
    """
    Network Ubiquity Share (NUS):
    - nodes_all: evaluate all graph nodes
    - segments_sampled: sample along the walking network every ~sample_step_m
    """
    method = cfg.get("nus_method", "segments_sampled")

    if method == "nodes_all":
        sample_nodes = np.array(list(G_proj.nodes))
    else:
        seg_pts = []
        for _, r in edges_gdf.iterrows():
            geom = r.geometry
            if geom.length < 1:
                continue
            n_seg = max(1, int(geom.length // cfg["sample_step_m"]))
            for t in np.linspace(0, 1, n_seg, endpoint=True):
                seg_pts.append(geom.interpolate(t, normalized=True))
        samples = gpd.GeoDataFrame(geometry=seg_pts, crs=cfg["crs_metric"])
        sample_nodes = oxdist.nearest_nodes(G_proj, samples.geometry.x.values, samples.geometry.y.values) if len(samples) else np.array([])

    poi_nodes = oxdist.nearest_nodes(
        G_proj, dest_pts_dedup.geometry.x.values, dest_pts_dedup.geometry.y.values
    ) if len(dest_pts_dedup) else np.array([])
    poi_node_set = set(int(n) for n in np.atleast_1d(poi_nodes))

    served_flags, hit_counts = [], []
    for sn in np.atleast_1d(sample_nodes):
        dist_map = nx.single_source_dijkstra_path_length(G_proj, int(sn), cutoff=cfg["reach_radius_m"], weight="length")
        hits_here = sum(1 for node_id in dist_map.keys() if node_id in poi_node_set)
        hit_counts.append(hits_here)
        served_flags.append(1 if hits_here >= cfg["min_pois_for_service"] else 0)

    NUS = float(np.mean(served_flags)) if len(served_flags) else float("nan")
    access_gini_local = gini(hit_counts) if len(hit_counts) else float("nan")
    return {"NUS": NUS, "access_gini_local": access_gini_local, "hit_counts_debug": hit_counts}


def k_shortest_paths_entropy(G_simple, o, d, k, beta):
    """
    For an origin node o and dest node d, get up to k simple shortest paths
    (in terms of edge 'length'), convert path costs into choice probabilities,
    then compute entropy of that distribution.
    """
    try:
        gen = nx.shortest_simple_paths(G_simple, o, d, weight="length")
    except nx.NetworkXNoPath:
        return np.nan, []
    path_costs, edge_lists = [], []
    for _, path in zip(range(k), gen):
        total_len = 0.0
        steps = []
        for u, v in zip(path[:-1], path[1:]):
            total_len += float(G_simple[u][v].get("length", 1.0))
            steps.append((u, v))
        path_costs.append(total_len)
        edge_lists.append(steps)
    if not path_costs:
        return np.nan, []
    path_costs = np.array(path_costs)
    util = np.exp(-beta * (path_costs - path_costs.min()))
    p = util / util.sum()
    p = p[p > 0]
    H = float(-(p * np.log(p)).sum())
    return H, edge_lists


def estimate_trip_structure_and_dest_diversity(G_proj, G_simple, edges_gdf, dest_pts, grid_viz, cfg):
    """
    Trip structure and per-origin diversity.
    """
    # grab buildings to guess where people actually start from
    nodes_xy = [(data["x"], data["y"]) for _, data in G_proj.nodes(data=True)]
    if nodes_xy:
        hull_m = MultiPoint(nodes_xy).convex_hull
        hull_ll = gpd.GeoSeries([hull_m], crs=cfg["crs_metric"]).to_crs(4326).iloc[0]
    else:
        hull_ll = gpd.GeoSeries([grid_viz.unary_union], crs=cfg["crs_metric"]).to_crs(4326).iloc[0]

    try:
        buildings = ox.features_from_polygon(hull_ll, tags={"building": True}).to_crs(cfg["crs_metric"])
    except Exception:
        buildings = gpd.GeoDataFrame(geometry=[], crs=cfg["crs_metric"])

    if buildings.empty:
        res_cells = grid_viz.copy()
    else:
        try:
            build_union = shapely.union_all(list(buildings.geometry))
        except Exception:
            build_union = unary_union(list(buildings.geometry))
        res_cells = grid_viz[grid_viz.distance(build_union) < 50].copy()
        if res_cells.empty:
            res_cells = grid_viz.copy()

    res_pts = res_cells.geometry.centroid
    res_nodes = oxdist.nearest_nodes(G_proj, res_pts.x.values, res_pts.y.values)

    # destination dataframe with poi_id preserved
    if dest_pts.empty:
        d_candidates = np.random.choice(list(G_proj.nodes), size=min(200, G_proj.number_of_nodes()), replace=False)
        dest_df = pd.DataFrame({"poi_id": np.arange(len(d_candidates)), "node": d_candidates, "cat": "other"})
    else:
        tmp = dest_pts.copy()
        tmp["node"] = oxdist.nearest_nodes(G_proj, tmp.geometry.x.values, tmp.geometry.y.values)
        dest_df = tmp[["poi_id","node","cat"]].reset_index(drop=True)

    origins = np.random.choice(res_nodes, size=min(cfg["max_origins"], len(res_nodes)), replace=False)

    route_entropies = []
    edge_hit_set = set()
    dest_cell_entropy_list = []
    dest_cat_entropy_list  = []

    # PoI to viz cell mapping using poi_id
    poi_to_cell = pd.Series(dtype="float64")
    if not dest_pts.empty and not grid_viz.empty:
        try:
            join_tmp = gpd.sjoin_nearest(
                dest_pts[["poi_id","geometry","cat"]],
                grid_viz[["cell_id","geometry"]],
                how="left",
                max_distance=cfg["sjoin_max_dist_m"]
            ).dropna(subset=["cell_id"])
            poi_to_cell = pd.Series(join_tmp["cell_id"].values, index=join_tmp["poi_id"].values)
        except Exception:
            pass

    for o in origins:
        ds = dest_df.sample(min(cfg["dests_per_origin"], len(dest_df)), replace=False,
                            random_state=np.random.randint(0, 10**6))

        # k-shortest path entropy
        for _, r in ds.iterrows():
            d = int(r["node"])
            if o == d:
                continue
            H, kpaths = k_shortest_paths_entropy(G_simple, o, d, k=cfg["k_paths"], beta=cfg["beta"])
            if not math.isnan(H):
                route_entropies.append(H)
            for path_edges in kpaths:
                for (u, v) in path_edges:
                    edge_hit_set.add((u, v))

        # spatial diversity of destinations using poi_id
        if not poi_to_cell.empty:
            cells = poi_to_cell.reindex(ds["poi_id"]).dropna().astype(int).tolist()
            if len(cells) >= 2:
                vals, cnts = np.unique(cells, return_counts=True)
                dest_cell_entropy_list.append(shannon_entropy(cnts, normalise_over=len(cnts)))
            else:
                # count “no spread” as 0 instead of skipping
                dest_cell_entropy_list.append(0.0)

        # functional diversity
        social_use_cats = ["everyday_retail","food_drink","health","services","education"]
        c_local = Counter([c for c in ds["cat"].tolist() if c in social_use_cats])
        if len(c_local) > 1:
            dest_cat_entropy_list.append(shannon_entropy(list(c_local.values()), normalise_over=len(social_use_cats)))
        elif len(c_local) == 1:
            dest_cat_entropy_list.append(0.0)

    # aggregate time
    if route_entropies:
        mean_route_entropy = float(np.mean(route_entropies))
        norm_route_entropy = mean_route_entropy / math.log(cfg["k_paths"])
    else:
        mean_route_entropy = float("nan")
        norm_route_entropy = float("nan")

    total_edges = G_simple.number_of_edges()
    cov_edges = len(edge_hit_set)
    edge_cov_frac = (cov_edges / total_edges) if total_edges else float("nan")

    total_len = sum(float(d.get("length", 0.0)) for _,_,d in G_simple.edges(data=True))
    hit_len = sum(float(G_simple[u][v].get("length", 0.0)) for (u, v) in edge_hit_set)
    edge_len_cov_frac = (hit_len / total_len) if total_len else float("nan")

    if dest_cell_entropy_list:
        mean_dest_cell_entropy = float(np.mean(dest_cell_entropy_list))
    else:
        mean_dest_cell_entropy = 0.0 if not dest_pts.empty else float("nan")

    mean_dest_cat_entropy = float(np.mean(dest_cat_entropy_list)) if dest_cat_entropy_list else (0.0 if not dest_pts.empty else float("nan"))

    return {
        "mean_route_entropy": mean_route_entropy,
        "norm_route_entropy": norm_route_entropy,
        "edge_cov_frac": edge_cov_frac,
        "edge_len_cov_frac": edge_len_cov_frac,
        "mean_dest_cell_entropy": mean_dest_cell_entropy,
        "mean_dest_cat_entropy": mean_dest_cat_entropy,
        "origins_debug": origins,
    }


def estimate_copresence(G_proj, edges_gdf, origins_nodes, dest_pts, cfg):
    """
    Simulated co-presence load (CPP).
    """
    if dest_pts.empty:
        d_candidates = np.random.choice(list(G_proj.nodes), size=min(200, G_proj.number_of_nodes()), replace=False)
        dest_df = pd.DataFrame({"node": d_candidates, "cat": "other"})
    else:
        tmp = dest_pts.copy()
        tmp["node"] = oxdist.nearest_nodes(G_proj, tmp.geometry.x.values, tmp.geometry.y.values)
        dest_df = tmp[["node","cat"]].reset_index(drop=True)

    OD = []
    for o in np.atleast_1d(origins_nodes):
        ds = dest_df.sample(min(cfg["dests_per_origin"], len(dest_df)), replace=False,
                            random_state=np.random.randint(0,10**6))
        for _, r in ds.iterrows():
            d = int(r["node"])
            if o != d:
                OD.append((int(o), d))

    hourly = np.array([
        0.01,0.005,0.005,0.005,0.01,0.02,0.03,0.06,0.08,0.07,0.06,0.06,
        0.05,0.05,0.05,0.06,0.07,0.08,0.06,0.04,0.02,0.015,0.01,0.005
    ])
    hourly = hourly / hourly.sum()

    edges_list = list(G_proj.edges(keys=True))
    edge_to_idx = {e:i for i,e in enumerate(edges_list)}
    F = np.zeros((len(edges_list), 24), dtype=float)

    def shortest_edge_seq(G, s, t):
        try:
            path = nx.shortest_path(G, s, t, weight="length")
        except nx.NetworkXNoPath:
            return []
        return list(zip(path[:-1], path[1:]))

    for s, t in OD:
        seq = shortest_edge_seq(G_proj, s, t)
        if not seq:
            continue
        for h, w_h in enumerate(hourly):
            for (u, v) in seq:
                data = G_proj.get_edge_data(u, v)
                if not data:
                    continue
                key = next(iter(data.keys()))
                F[edge_to_idx[(u, v, key)], h] += w_h

    CPP_edge_total = F.sum(axis=1)
    edges_cpp = edges_gdf.copy()
    edges_cpp["cpp"] = CPP_edge_total
    max_cpp = edges_cpp["cpp"].max() or 1.0
    edges_cpp["w"] = edges_cpp["cpp"] / max_cpp

    cpp_vals = edges_cpp["cpp"].values
    cpp_pos  = cpp_vals[cpp_vals > 0]
    if cpp_pos.size:
        x_sorted = np.sort(cpp_pos)
        n = len(x_sorted)
        cumx = np.cumsum(x_sorted)
        cpp_gini = (n + 1 - 2 * (cumx.sum() / cumx[-1])) / n
        t90 = np.percentile(cpp_pos, 90)
        top10_share = cpp_pos[cpp_pos >= t90].sum() / cpp_pos.sum()
    else:
        cpp_gini, top10_share = float("nan"), float("nan")

    return edges_cpp, {"cpp_gini": cpp_gini, "cpp_top10_share": top10_share}


def plot_static_map(area_poly_m, edges_cpp, dest_pts, cfg, outdir):
    """
    Produce a static map:
    - edges coloured by co-presence load
    - PoIs coloured by category
    - boundary outline
    - street labels for top-CPP segments
    """

    cpp_vals = np.array(edges_cpp["cpp"].fillna(0.0))
    vmin = np.percentile(cpp_vals, 5)
    vmax = np.percentile(cpp_vals, 99)
    norm_col = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap  = mpl.cm.viridis
    colors = cmap(norm_col(cpp_vals))

    # thicker lines for higher co-presence
    q = np.clip(cpp_vals, np.percentile(cpp_vals, 1), np.percentile(cpp_vals, 99))
    w = (q - q.min()) / (q.max() - q.min() + 1e-9)
    lw = 0.3 + 3.8 * (w**0.5)

    palette = {
        "everyday_retail":"#4c78a8",
        "food_drink":"#e45756",
        "health":"#72b7b2",
        "services":"#f58518",
        "education":"#9c8ade",
        "transit":"#54a24b",
        "other":"#9d9da1",
    }

    fig, ax = plt.subplots(figsize=(9,9), dpi=150)
    edges_cpp.plot(ax=ax, color=colors, linewidth=lw, alpha=0.95, zorder=2)

    if not dest_pts.empty:
        for cat, dfc in dest_pts.groupby("cat"):
            dfc.plot(ax=ax, markersize=12, alpha=0.9, color=palette.get(cat, "#9d9da1"),
                     label=cat, zorder=3)

    gpd.GeoDataFrame(geometry=[area_poly_m], crs=cfg["crs_metric"]).boundary.plot(
        ax=ax, color="black", linewidth=1, zorder=4
    )

    x1, y1, x2, y2 = edges_cpp.total_bounds
    ax.set_xlim(x1, x2); ax.set_ylim(y1, y2)

    cx_kw = dict(crs=cfg["crs_metric"], zoom=cfg["map_zoom"])
    try:
        add_basemap_portable(ax, cx.providers.CartoDB.PositronNoLabels, zorder=0, **cx_kw)
        add_basemap_portable(ax, cx.providers.CartoDB.PositronOnlyLabels, zorder=5, **cx_kw)
    except Exception:
        add_basemap_portable(ax, cx.providers.OpenStreetMap.Mapnik, zorder=0, **cx_kw)

    # label a few top-CPP streets if they have names
    lab = edges_cpp.copy()
    if "name" in lab.columns:
        def first_name(val):
            if isinstance(val, list) and val:
                return str(val[0])
            if isinstance(val, str):
                return val
            return None
        lab["name"] = lab["name"].apply(first_name)
        lab = lab.dropna(subset=["name"]).sort_values("cpp", ascending=False).head(6)
        for _, r in lab.iterrows():
            try:
                xx, yy = r.geometry.representative_point().coords[0]
                ax.text(xx, yy, r["name"], fontsize=8, color="black", alpha=0.9,
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")], zorder=6)
            except Exception:
                pass

    # tiny legend for PoI categories
    if not dest_pts.empty:
        poi_handles = []
        for k in ["everyday_retail","food_drink","health","services","education","transit"]:
            if k in dest_pts["cat"].unique():
                poi_handles.append(mpl.patches.Patch(facecolor=palette.get(k, "#9d9da1"),
                                                     edgecolor="none", label=k.replace("_"," ")))
        if poi_handles:
            leg = ax.legend(handles=poi_handles, title="POI category", loc="lower left", frameon=True)
            leg.get_frame().set_alpha(0.9)

        # colour bar for CPP to match the line colours
        sm = mpl.cm.ScalarMappable(norm=norm_col, cmap=cmap)
        sm.set_array(cpp_vals)  # any array; just links the scale
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01)
        cbar.set_label("simulated co-presence load (relative)")
        # tidy ticks
        cbar.ax.tick_params(labelsize=8)
            
    ax.set_title(f"amenities, walk network, and simulated co-presence load for {cfg['place_name']} (r ≈ {cfg['study_radius_m']}m)")
    ax.set_axis_off(); fig.tight_layout()
    static_png = os.path.join(outdir, "cpp_map_static.png")
    fig.savefig(static_png, dpi=220, bbox_inches="tight"); plt.close(fig)
    return static_png


def make_interactive_map(area_poly_m, edges_cpp, dest_pts, cfg, outdir):
    """
    Edges are coloured by co-presence load.
    PoIs are toggleable layers by category.
    """

    edges_wgs = edges_cpp.to_crs(4326).copy()
    pois_wgs  = dest_pts.to_crs(4326).copy()
    center = gpd.GeoSeries([area_poly_m], crs=cfg["crs_metric"]).to_crs(4326).geometry.iloc[0].centroid

    m = folium.Map(location=[center.y, center.x], tiles="CartoDB Positron", zoom_start=15)

    cpp_vals = np.array(edges_cpp["cpp"].fillna(0.0))
    vmin = float(np.percentile(cpp_vals, 5)); vmax = float(np.percentile(cpp_vals, 99))
    colormap = branca.colormap.LinearColormap(colors=["#440154","#31688e","#35b779","#fde725"], vmin=vmin, vmax=vmax)
    colormap.caption = "co-presence load (relative)"; colormap.add_to(m)

    def style_edge(feat):
        val = feat["properties"].get("cpp", 0.0)
        c = colormap(val)
        w_l = 1 + 4 * ((val - vmin)/(vmax - vmin + 1e-9))**0.5
        return {"color": c, "weight": w_l, "opacity": 0.9}

    folium.GeoJson(
        edges_wgs[["cpp","geometry","name"]],
        name="co-presence network",
        style_function=style_edge,
        tooltip=folium.GeoJsonTooltip(fields=["name","cpp"], aliases=["Street","CPP"])
    ).add_to(m)

    palette = {
        "everyday_retail":"#4c78a8","food_drink":"#e45756","health":"#72b7b2",
        "services":"#f58518","education":"#9c8ade","transit":"#54a24b","other":"#9d9da1",
    }

    if not pois_wgs.empty:
        for cat, dfc in pois_wgs.groupby("cat"):
            layer = folium.FeatureGroup(name=f"{cat}", show=True if cat in {"everyday_retail","food_drink"} else False)
            for _, r in dfc.iterrows():
                folium.CircleMarker(location=[r.geometry.y, r.geometry.x], radius=4,
                                    color=palette.get(cat, "#9d9da1"), fill=True, fill_opacity=0.9,
                                    tooltip=f"{cat}").add_to(layer)
            layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    html_path = os.path.join(outdir, "cpp_map_interactive.html"); m.save(html_path)
    return html_path


def export_metrics_csv(metrics, cfg, outdir):
    """
    Create CSV row with:
    - the structural indicators
    - the config values, for reliablity testing
    """
    df = pd.DataFrame([metrics])
    out_csv = os.path.join(outdir, "summary_metrics.csv")
    df.to_csv(out_csv, index=False)
    return out_csv


# -------------------------------------------------------------------------------------------------------------------------------
# main script
# -------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. study polygon + network
    poly_ll, poly_m = circle_study_polygon(cfg["place_name"], cfg["study_radius_m"], cfg["crs_metric"])
    G_proj, G_simple, nodes_gdf, edges_gdf = build_walk_graph(poly_ll, cfg["crs_metric"])

    # 2. pull PoIs from OSM and classify them
    tags = {
        # amenity layer will cover most daily-life stuff we care about
        "amenity": [
            "cafe","fast_food","restaurant","bar","pub",
            "bank","post_office","library",
            "clinic","doctors","dentist","pharmacy",
            "school","college","university","nursery","childcare",
            "place_of_worship","community_centre",
            "bus_station","ferry_terminal","taxi"
        ],
        # 'shop' for corner shops, grocers, barbers etc
        "shop": [
            "convenience","supermarket","greengrocer","butcher","bakery",
            "chemist","newsagent","alcohol","kiosk", "off_licence",
            "hairdresser","beauty","cosmetics","clothes", "retail"
        ],
        # healthcare is annoyingly sometimes mapped separately
        "healthcare": [
            "pharmacy","doctor","dentist","clinic","physiotherapist","optometrist","optician"
        ],
        # transit primitives
        "public_transport": ["station","platform","stop_position"],
        "highway": ["bus_stop"],
        "railway": ["station","halt"]
    }

    pois_raw = ox.features_from_polygon(poly_ll, tags=tags).to_crs(cfg["crs_metric"])
    pois_raw = pois_raw[~pois_raw.geometry.is_empty].copy()
    pois_raw["cat"] = pois_raw.apply(classify_poi, axis=1)

    # reduce geometries to points for snapping/network analysis
    dest_pts = pois_raw.copy()
    dest_pts["geometry"] = dest_pts.geometry.apply(representative_point_any)
    dest_pts = dest_pts[dest_pts.geometry.notna()].copy().set_geometry("geometry").set_crs(cfg["crs_metric"])
    dest_pts = dest_pts.reset_index(drop=True)
    dest_pts["poi_id"] = dest_pts.index
    
    # deduplicate PoIs
    dedup = dest_pts.copy()
    dedup["gx"] = (dedup.geometry.x // cfg["dedupe_radius_m"]).astype(int)
    dedup["gy"] = (dedup.geometry.y // cfg["dedupe_radius_m"]).astype(int)
    dedup = dedup.drop_duplicates(subset=["gx","gy"]).drop(columns=["gx","gy"])

    # build viz grid (fixed cell size) and compute LUM entropy
    grid_viz, walk_union = build_visual_grid(
        poly_m, edges_gdf, cfg["viz_cell_m"], cfg["crs_metric"],
        clip_to_streets=cfg["clip_grid_to_streets"], street_buffer_m=cfg["street_buffer_m"]
    )
    lum_entropy, lum_join_share = compute_lum(dest_pts, grid_viz, cfg["sjoin_max_dist_m"])

    # Now AUI
    aui_grid = make_adaptive_aui_grid(poly_m, walk_union, dedup, cfg)
    aui_net = {}
    if "network" in cfg["aui_modes"]:
        aui_net = make_network_aui(G_proj, dedup, aui_grid["AUI_cell_L_m"], cfg)
    
    # Up Next, NUS
    nus_info = compute_network_ubiquity_and_access(G_proj, edges_gdf, dedup, cfg)

    # Now onto trip structure (route entropy etc.) and per-origin diversity
    trip_info = estimate_trip_structure_and_dest_diversity(G_proj, G_simple, edges_gdf, dest_pts, grid_viz, cfg)

    # Finally, co-presence on the network (how concentrated is overlap?)
    edges_cpp, cpp_info = estimate_copresence(G_proj, edges_gdf, trip_info["origins_debug"], dest_pts, cfg)

    # -------------------------------------------------------------------------------------------------------------------------------
    # outputs
    # -------------------------------------------------------------------------------------------------------------------------------
    static_png_path = plot_static_map(poly_m, edges_cpp, dest_pts, cfg, cfg["outdir"])
    html_path = make_interactive_map(poly_m, edges_cpp, dest_pts, cfg, cfg["outdir"])

    # gathering all metrics into one dict for CSV and console print
    metrics = {
        # AUI grid
        "AUI_raw": aui_grid["AUI_raw"],
        "AUI_star": aui_grid["AUI_star"],
        "AUI_cell_L_m": aui_grid["AUI_cell_L_m"],
        "AUI_occupancy": aui_grid["AUI_occupancy"],
        "AUI_nonempty_gini": aui_grid["nonempty_gini"],

        # AUI network
        "AUI_raw_net": aui_net.get("AUI_raw_net"),
        "AUI_star_net": aui_net.get("AUI_star_net"),
        "AUI_occupancy_net": aui_net.get("AUI_occupancy_net"),
        "AUI_nonempty_gini_net": aui_net.get("nonempty_gini_net"),
        "AUI_sample_step_m": aui_net.get("AUI_sample_step_m"),
        "AUI_n_samples": aui_net.get("n_samples"),

        # access structure
        "NUS_share": nus_info["NUS"],
        "access_gini_local": nus_info["access_gini_local"],

        # land use mix
        "LUM_entropy_no_transit": lum_entropy,
        "LUM_join_share": lum_join_share,

        # trip structure
        "mean_route_entropy": trip_info["mean_route_entropy"],
        "norm_route_entropy": trip_info["norm_route_entropy"],
        "edge_cov_frac": trip_info["edge_cov_frac"],
        "edge_len_cov_frac": trip_info["edge_len_cov_frac"],
        "mean_dest_cell_entropy": trip_info["mean_dest_cell_entropy"],
        "mean_dest_cat_entropy": trip_info["mean_dest_cat_entropy"],

        # co-presence concentration
        "cpp_gini": cpp_info["cpp_gini"],
        "cpp_top10_share": cpp_info["cpp_top10_share"],

        # counts + config echoes
        "pois_total": int(len(dest_pts)),
        "pois_deduped": int(len(dedup)),
        "viz_grid_cells": int(len(grid_viz)),
        "clip_grid_to_streets": bool(cfg["clip_grid_to_streets"]),
        "dedupe_radius_m": cfg["dedupe_radius_m"],
        "sjoin_max_dist_m": cfg["sjoin_max_dist_m"],
        "aui_modes": ",".join(cfg["aui_modes"]),
        "nus_method": cfg["nus_method"],
        "reach_radius_m": cfg["reach_radius_m"],
        "min_pois_for_service": cfg["min_pois_for_service"],
        "viz_cell_m": cfg["viz_cell_m"],
        "AUI_cell_min_m": cfg["cell_min_m"],
        "AUI_cell_max_m": cfg["cell_max_m"],
        "study_radius_m": cfg["study_radius_m"],
        "sample_step_m": cfg["sample_step_m"],
        "k_paths": cfg["k_paths"],
        "beta": cfg["beta"],
        "max_origins": cfg["max_origins"],
        "dests_per_origin": cfg["dests_per_origin"],
        "place_name": cfg["place_name"],
    }

    out_csv = export_metrics_csv(metrics, cfg, cfg["outdir"])

    # final console dump
    print(f"[{cfg['place_name']}] r≈{cfg['study_radius_m']}m")
    print(
        f"AUI* (grid)={metrics['AUI_star']:.3f} "
        f"(occ={metrics['AUI_occupancy']:.2f}, L≈{metrics['AUI_cell_L_m']:.0f}m, gini_nonempty={metrics['AUI_nonempty_gini']:.3f}); "
        + (f"AUI* (net)={metrics['AUI_star_net']:.3f} (occ={metrics['AUI_occupancy_net']:.2f}, step≈{metrics['AUI_sample_step_m']:.0f}m); " if metrics['AUI_star_net'] is not None else "")
        + f"NUS={metrics['NUS_share']:.3f} (method={cfg['nus_method']}); AccessGini_local={metrics['access_gini_local']:.3f}; "
        f"LUM_no_transit={metrics['LUM_entropy_no_transit']:.3f} (join_share={metrics['LUM_join_share']:.2f});\n"
        f"route_entropy={metrics['mean_route_entropy']:.3f} (norm={metrics['norm_route_entropy']:.3f}); "
        f"kpaths_edge_cov={metrics['edge_cov_frac']:.3f}; kpaths_len_cov={metrics['edge_len_cov_frac']:.3f};\n"
        f"dest_cell_H={metrics['mean_dest_cell_entropy']:.3f}; dest_cat_H={metrics['mean_dest_cat_entropy']:.3f}; "
        f"CPP_gini={metrics['cpp_gini']:.3f}; top10_share={metrics['cpp_top10_share']:.3f}"
    )
    print("Note: CPP etc. are simulated structural potentials, not observed counts!")
    print("map (static png):", static_png_path)
    print("map (interactive html):", html_path)
    print("metrics csv:", out_csv)
    # TODO: maybe add hit_counts_debug to csv if I need to explain Access_gini in detail to someone
