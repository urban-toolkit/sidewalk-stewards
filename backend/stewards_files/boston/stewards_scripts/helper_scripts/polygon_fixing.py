"""
polygon_fixing.py
=================
Polygon-level topology operations for sidewalk network fixing.

Functions:
    Geometry helpers:
        geo_to_px, project_to_utm, load_network,
        clip_network_to_tile, clip_polygons_to_tile

    Tile loading:
        load_tile_images, pick_tile_with_network, get_tile_utm_context,
        rasterize_polygons_to_mask

    Centerline:
        compute_centerline, elongate_polygon_centerline

    Polygon operations:
        elongate_polygon, elongate_tile_polygons

    Suggestion generation:
        generate_suggestions

    Display:
        plot_lines, plot_polygons, display_suggestions
"""

import math
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from pathlib import Path
from PIL import Image

from shapely.geometry import (
    LineString, MultiLineString, Point, MultiPoint,
    box as shapely_box,
)
from shapely.ops import unary_union, transform as shapely_transform, linemerge
from shapely import clip_by_rect#, voronoi_diagram
from shapely.ops import voronoi_diagram
from shapely.validation import make_valid
from shapely.prepared import prep
from collections import defaultdict

import pyproj
from rasterio import features as rio_features
from rasterio.transform import from_bounds

# Re-export tile helpers from training utils (avoid duplication)
from tile2net_training_utils import num2deg, get_tile_bounds


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def geo_to_px(coords, bounds, res=256):
    """Convert geographic coordinates to pixel coordinates.

    Parameters
    ----------
    coords : iterable of (x, y)
        Geographic coordinates (lon, lat).
    bounds : tuple
        (west, south, east, north) in EPSG:4326.
    res : int
        Tile resolution in pixels (default 256).

    Returns
    -------
    list of (px_x, px_y)
    """
    w, s, e, n = bounds
    return [((x - w) / (e - w) * res, (n - y) / (n - s) * res) for x, y in coords]


def project_to_utm(geometry_4326, epsg_utm=32619):
    """Project a single shapely geometry from EPSG:4326 to a UTM CRS.

    Parameters
    ----------
    geometry_4326 : shapely geometry
    epsg_utm : int
        Target UTM EPSG code (default 32619 = UTM 19N, covers Boston).

    Returns
    -------
    shapely geometry in projected CRS
    """
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg_utm}", always_xy=True
    )
    return shapely_transform(transformer.transform, geometry_4326)


def load_network(path, label="network"):
    """Load a network shapefile/GeoJSON, reproject to 4326, filter to sidewalk.

    Parameters
    ----------
    path : str or Path
        Path to shapefile or GeoJSON.
    label : str
        Label for print messages.

    Returns
    -------
    GeoDataFrame
        Sidewalk-only features in EPSG:4326.
    """
    print(f"Loading {label}: {path}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    n_all = len(gdf)
    if "f_type" in gdf.columns:
        gdf = gdf[gdf["f_type"] == "sidewalk"].copy()
    print(f"  → {len(gdf):,}/{n_all:,} sidewalk features")
    return gdf


def clip_network_to_tile(gdf_network, bounds):
    """Clip a network GeoDataFrame to tile bounds.

    Parameters
    ----------
    gdf_network : GeoDataFrame
        Network features in EPSG:4326.
    bounds : tuple
        (west, south, east, north).

    Returns
    -------
    GeoDataFrame
        Clipped features (only non-empty geometries kept).
    """
    w, s, e, n = bounds
    subset = gdf_network.cx[w:e, s:n].copy()
    clipped_geoms = []
    keep_idx = []
    for idx, row in subset.iterrows():
        clipped = clip_by_rect(row.geometry, w, s, e, n)
        if clipped is not None and not clipped.is_empty:
            clipped_geoms.append(clipped)
            keep_idx.append(idx)
    if not clipped_geoms:
        return subset.iloc[:0]
    result = subset.loc[keep_idx].copy()
    result.geometry = clipped_geoms
    return result


def clip_polygons_to_tile(gdf_polygons, bounds):
    """Clip a polygon GeoDataFrame to tile bounds, filtering non-polygons.

    Parameters
    ----------
    gdf_polygons : GeoDataFrame
        Polygon features in EPSG:4326.
    bounds : tuple
        (west, south, east, north).

    Returns
    -------
    GeoDataFrame
        Clipped polygon features (Points/Lines from edge clipping removed).
    """
    w, s, e, n = bounds
    tile_box = shapely_box(w, s, e, n)
    subset = gdf_polygons.cx[w:e, s:n].copy()
    clipped_geoms = []
    keep_idx = []
    for idx, row in subset.iterrows():
        clipped = row.geometry.intersection(tile_box)
        if clipped is None or clipped.is_empty:
            continue
        if clipped.geom_type in ("Polygon", "MultiPolygon"):
            clipped_geoms.append(clipped)
            keep_idx.append(idx)
        elif clipped.geom_type == "GeometryCollection":
            polys = [g for g in clipped.geoms
                     if g.geom_type in ("Polygon", "MultiPolygon")]
            if polys:
                clipped_geoms.append(unary_union(polys))
                keep_idx.append(idx)
    if not clipped_geoms:
        return subset.iloc[:0]
    result = subset.loc[keep_idx].copy()
    result.geometry = clipped_geoms
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Tile loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_tile_images(tile_id, tiles_dir, poly_mask_dir=None, gt_mask_dir=None):
    """Load satellite image and optional masks for a tile.

    Parameters
    ----------
    tile_id : str
        Tile ID (e.g. "158598_193996").
    tiles_dir : str or Path
        Directory containing satellite tiles ({tile_id}.jpg).
    poly_mask_dir : str or Path or None
        Directory containing polygon-rasterized T2N masks ({tile_id}.png).
    gt_mask_dir : str or Path or None
        Directory containing ground truth masks ({tile_id}.png).

    Returns
    -------
    sat : ndarray (256, 256, 3) uint8
        Satellite RGB image.
    t2n_mask : ndarray (256, 256) uint8 or None
        T2N polygon mask (255=sidewalk, 0=background). None if poly_mask_dir not given.
    gt_mask : ndarray (256, 256) uint8 or None
        Ground truth mask. None if gt_mask_dir not given.
    """
    tiles_dir = Path(tiles_dir)
    sat = np.array(Image.open(tiles_dir / f"{tile_id}.jpg").convert("RGB"))

    t2n_mask = None
    if poly_mask_dir is not None:
        t2n_mask = np.array(
            Image.open(Path(poly_mask_dir) / f"{tile_id}.png").convert("L")
        )

    gt_mask = None
    if gt_mask_dir is not None:
        gt_mask = np.array(
            Image.open(Path(gt_mask_dir) / f"{tile_id}.png").convert("L")
        )

    return sat, t2n_mask, gt_mask


def pick_tile_with_network(tile_ids, gdf_sidewalk, min_segments=3,
                           seed=None, zoom=19):
    """Pick a random tile that has at least `min_segments` network features.

    Parameters
    ----------
    tile_ids : list of str
        Candidate tile IDs.
    gdf_sidewalk : GeoDataFrame
        Sidewalk network in EPSG:4326 (used for spatial indexing).
    min_segments : int
        Minimum number of network segments required.
    seed : int or None
        Random seed (None = non-deterministic).
    zoom : int
        Tile zoom level (default 19).

    Returns
    -------
    str
        Selected tile ID.

    Raises
    ------
    ValueError
        If no tile meets the minimum segment requirement.
    """
    rng = np.random.RandomState(seed)
    candidates = rng.choice(tile_ids, size=min(100, len(tile_ids)), replace=False)

    for candidate in candidates:
        xtile, ytile = map(int, candidate.split("_"))
        bounds = get_tile_bounds(xtile, ytile, zoom)
        w, s, e, n = bounds
        local = gdf_sidewalk.cx[w:e, s:n]
        if len(local) >= min_segments:
            return candidate

    raise ValueError(
        f"No tile with >= {min_segments} network segments found "
        f"(checked {len(candidates)} candidates)"
    )


def get_tile_utm_context(tile_polys, bounds, epsg_utm=32619):
    """Project tile polygons to UTM and create the tile bounding box in UTM.

    Parameters
    ----------
    tile_polys : GeoDataFrame
        Polygons in EPSG:4326 (output of clip_polygons_to_tile).
    bounds : tuple
        (west, south, east, north) in EPSG:4326.
    epsg_utm : int
        Target UTM EPSG code (default 32619 = UTM 19N, covers Boston).

    Returns
    -------
    tile_polys_proj : GeoDataFrame
        Polygons reprojected to UTM.
    tile_box_utm : shapely Polygon
        Tile bounding box in UTM.
    """
    tile_polys_proj = tile_polys.to_crs(f"EPSG:{epsg_utm}")

    w, s, e, n = bounds
    tile_box_geo = shapely_box(w, s, e, n)
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg_utm}", always_xy=True
    )
    tile_box_utm = shapely_transform(transformer.transform, tile_box_geo)

    return tile_polys_proj, tile_box_utm


def rasterize_polygons_to_mask(
    gdf,
    xtile,
    ytile,
    zoom=19,
    resolution=256,
):
    """Rasterize sidewalk polygons to a binary mask for a specific tile.

    Copied from generate_groundtruth_masks_from_polygons.ipynb
    (originally used to generate the ground truth masks).
    Copied on Thursday 03/05/2026.

    Parameters
    ----------
    gdf : GeoDataFrame
        Sidewalk polygons in EPSG:4326.
    xtile : int
        X tile coordinate.
    ytile : int
        Y tile coordinate.
    zoom : int
        Zoom level (default 19).
    resolution : int
        Output mask resolution in pixels (default 256).

    Returns
    -------
    ndarray (resolution, resolution) uint8
        Binary mask (0 or 255).
    """
    west, south, east, north = get_tile_bounds(xtile, ytile, zoom)

    gdf_clipped = gdf.cx[west:east, south:north]

    if len(gdf_clipped) == 0:
        return np.zeros((resolution, resolution), dtype=np.uint8)

    transform = from_bounds(west, south, east, north, resolution, resolution)

    shapes = [
        (geom, 1)
        for geom in gdf_clipped.geometry
        if geom is not None and geom.is_valid
    ]

    if len(shapes) == 0:
        return np.zeros((resolution, resolution), dtype=np.uint8)

    mask = rio_features.rasterize(
        shapes=shapes,
        out_shape=(resolution, resolution),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    mask = (mask > 0).astype(np.uint8) * 255
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Centerline computation  [DEPRECATED — Voronoi centerline unreliable/slow]
# ─────────────────────────────────────────────────────────────────────────────

def compute_centerline(poly, densify_dist=2.0):
    """DEPRECATED: Voronoi centerline is unreliable for irregular polygons and ~20x slower.

    Compute the centerline (medial axis) of a polygon using Voronoi diagram.

    Densifies the polygon boundary, computes the Voronoi diagram of the
    boundary points, and keeps only edges that lie fully inside the polygon.
    The result is merged into a single LineString or MultiLineString.

    Parameters
    ----------
    poly : shapely Polygon
        Input polygon (projected CRS, e.g. meters).
    densify_dist : float
        Maximum distance between consecutive boundary points (meters).
        Smaller values → more accurate centerline but slower. Default 2.0.

    Returns
    -------
    shapely LineString or MultiLineString, or None if computation fails.
    """
    if poly.is_empty or poly.area < 1e-6:
        return None

    # Densify the boundary so Voronoi has enough points
    boundary = poly.exterior
    densified = boundary.segmentize(densify_dist)
    coords = list(densified.coords)[:-1]  # drop closing duplicate

    if len(coords) < 4:
        return None

    # Compute Voronoi diagram of boundary points
    points = MultiPoint([Point(c) for c in coords])
    try:
        regions = voronoi_diagram(points, envelope=poly.envelope.buffer(1))
    except Exception:
        return None

    # Extract all edges from Voronoi regions, keep only those inside the polygon
    # Use a small negative buffer to avoid edges right on the boundary
    interior = poly.buffer(-densify_dist * 0.3)
    if interior.is_empty:
        interior = poly

    # Use prepared geometry for fast repeated containment checks
    prep_interior = prep(interior)

    edges = []
    for region in regions.geoms:
        if region.geom_type == "Polygon":
            ring = region.exterior
            coords_r = list(ring.coords)
            for i in range(len(coords_r) - 1):
                seg = LineString([coords_r[i], coords_r[i + 1]])
                if prep_interior.contains(seg):
                    edges.append(seg)

    if not edges:
        return None

    merged = linemerge(edges)
    if merged.is_empty:
        return None

    return merged


def _find_endpoints(centerline):
    """DEPRECATED: Part of centerline-based elongation (unreliable).

    Find degree-1 nodes (endpoints) of a centerline geometry.

    Parameters
    ----------
    centerline : LineString or MultiLineString

    Returns
    -------
    list of (x, y) tuples — the endpoint coordinates.
    """
    if centerline.geom_type == "LineString":
        coords = list(centerline.coords)
        if len(coords) < 2:
            return []
        return [coords[0], coords[-1]]

    # MultiLineString: build adjacency graph, find degree-1 nodes
    node_degree = defaultdict(int)
    for line in centerline.geoms:
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        # Round to avoid floating point mismatches
        start = (round(coords[0][0], 6), round(coords[0][1], 6))
        end = (round(coords[-1][0], 6), round(coords[-1][1], 6))
        node_degree[start] += 1
        node_degree[end] += 1

    return [pt for pt, deg in node_degree.items() if deg == 1]


def _local_direction(centerline, endpoint, walk_dist=5.0):
    """DEPRECATED: Part of centerline-based elongation (unreliable).

    Get the outward direction vector at a centerline endpoint.

    Walks `walk_dist` meters along the centerline from the endpoint
    and returns the unit direction vector pointing outward (away from center).

    Parameters
    ----------
    centerline : LineString or MultiLineString
    endpoint : (x, y) tuple
    walk_dist : float
        Distance to walk inward to compute tangent (meters).

    Returns
    -------
    (ux, uy) : unit direction vector pointing outward, or None.
    """
    ep = Point(endpoint)

    # Find the line segment that contains this endpoint
    if centerline.geom_type == "LineString":
        lines = [centerline]
    else:
        lines = list(centerline.geoms)

    best_line = None
    best_dist = float("inf")
    is_start = True

    for line in lines:
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        d_start = ep.distance(Point(coords[0]))
        d_end = ep.distance(Point(coords[-1]))
        if d_start < best_dist:
            best_dist = d_start
            best_line = line
            is_start = True
        if d_end < best_dist:
            best_dist = d_end
            best_line = line
            is_start = False

    if best_line is None or best_dist > 1.0:
        return None

    # Walk inward along the line from the endpoint
    total_len = best_line.length
    walk = min(walk_dist, total_len * 0.4)

    if is_start:
        # endpoint is at start → walk forward along the line
        inner_pt = best_line.interpolate(walk)
        # direction: from inner point toward endpoint (outward)
        dx = endpoint[0] - inner_pt.x
        dy = endpoint[1] - inner_pt.y
    else:
        # endpoint is at end → walk backward from end
        inner_pt = best_line.interpolate(total_len - walk)
        dx = endpoint[0] - inner_pt.x
        dy = endpoint[1] - inner_pt.y

    length = (dx**2 + dy**2) ** 0.5
    if length < 1e-10:
        return None

    return (dx / length, dy / length)


def _local_half_width(poly, endpoint, direction):
    """DEPRECATED: Part of centerline-based elongation (unreliable).

    Estimate the polygon's half-width at an endpoint, perpendicular to direction.

    Casts a short line perpendicular to the direction through the endpoint
    and measures how far it extends inside the polygon on each side.

    Parameters
    ----------
    poly : shapely Polygon
    endpoint : (x, y)
    direction : (ux, uy) unit vector

    Returns
    -------
    float — estimated half-width in CRS units.
    """
    ux, uy = direction
    # Perpendicular direction
    px, py = -uy, ux

    probe_len = 50.0  # max probe distance
    perp_line = LineString([
        (endpoint[0] - px * probe_len, endpoint[1] - py * probe_len),
        (endpoint[0] + px * probe_len, endpoint[1] + py * probe_len),
    ])

    clipped = perp_line.intersection(poly)
    if clipped.is_empty:
        return 1.0  # fallback

    if clipped.geom_type == "LineString":
        return clipped.length / 2.0
    elif clipped.geom_type == "MultiLineString":
        # Take the segment closest to the endpoint
        ep = Point(endpoint)
        best = min(clipped.geoms, key=lambda g: g.distance(ep))
        return best.length / 2.0

    return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Polygon elongation
# ─────────────────────────────────────────────────────────────────────────────

def elongate_polygon_centerline(poly, max_extension, tile_box,
                                return_suggestion_only=False,
                                densify_dist=2.0, walk_dist=5.0):
    """DEPRECATED: Voronoi centerline unreliable for irregular shapes and ~20x slower.

    Extend a polygon from its centerline endpoints along local direction.

    Unlike elongate_polygon() which uses the minimum rotated rectangle,
    this function computes the polygon's centerline (medial axis) and
    extends from its actual endpoints following the local tangent direction.
    This correctly handles L-shapes, S-shapes, C-shapes, and other
    non-rectangular polygons.

    Parameters
    ----------
    poly : shapely Polygon
        The polygon to elongate (in projected CRS, e.g. EPSG:32619).
    max_extension : float
        Maximum extension distance (meters).
    tile_box : shapely Polygon
        Tile boundary for clipping.
    return_suggestion_only : bool
        If True, return only the new extension (difference with original).
    densify_dist : float
        Centerline computation resolution (meters). Default 1.0.
    walk_dist : float
        Distance to walk along centerline for tangent estimation. Default 5.0.

    Returns
    -------
    shapely Polygon or MultiPolygon
        The elongated polygon (or just the extensions), clipped to tile bounds.
    """
    if poly.is_empty or poly.area < 1e-6:
        return poly
    if poly.geom_type not in ("Polygon", "MultiPolygon"):
        return poly

    if not poly.is_valid:
        poly = make_valid(poly)
        if poly.is_empty or poly.geom_type not in ("Polygon", "MultiPolygon"):
            return poly

    # For MultiPolygon, process the largest part
    work_poly = poly
    if poly.geom_type == "MultiPolygon":
        work_poly = max(poly.geoms, key=lambda g: g.area)

    # Compute centerline
    centerline = compute_centerline(work_poly, densify_dist=densify_dist)
    if centerline is None:
        # Fallback to MRR-based elongation
        return elongate_polygon(poly, max_extension, tile_box,
                                return_suggestion_only=return_suggestion_only)

    # Find endpoints and their local directions
    endpoints = _find_endpoints(centerline)
    if len(endpoints) < 2:
        return elongate_polygon(poly, max_extension, tile_box,
                                return_suggestion_only=return_suggestion_only)

    # Build extensions from each endpoint
    extensions = []
    for ep in endpoints:
        direction = _local_direction(centerline, ep, walk_dist=walk_dist)
        if direction is None:
            continue

        half_w = _local_half_width(work_poly, ep, direction)
        ux, uy = direction

        ext_line = LineString([
            ep,
            (ep[0] + ux * max_extension, ep[1] + uy * max_extension),
        ])
        ext_poly = ext_line.buffer(half_w, cap_style="flat")
        extensions.append(ext_poly)

    if not extensions:
        return poly.intersection(tile_box)

    elongated = unary_union([poly] + extensions)

    if return_suggestion_only:
        elongated = elongated.difference(poly)

    elongated = elongated.intersection(tile_box)
    return elongated


def elongate_polygon(poly, max_extension, tile_box, return_suggestion_only=False, side="a"):
    """Extend a polygon along its principal axis from both ends.

    The polygon's minimum rotated rectangle determines orientation.
    Rectangular "tongues" are added at each short edge, matching the
    polygon's width. The result is clipped to tile_box.

    Parameters
    ----------
    poly : shapely Polygon
        The polygon to elongate.
    max_extension : float
        Maximum extension distance (in CRS units — meters if projected).
    tile_box : shapely Polygon
        Tile boundary for clipping.

    Returns
    -------
    shapely Polygon or MultiPolygon
        The elongated polygon, clipped to tile bounds.
    """
    if poly.is_empty or poly.area < 1e-6:
        return poly
    if poly.geom_type not in ("Polygon", "MultiPolygon"):
        return poly

    # Fix invalid geometries (self-intersections from clipping/projection)
    if not poly.is_valid:
        poly = make_valid(poly)
        if poly.is_empty or poly.geom_type not in ("Polygon", "MultiPolygon"):
            return poly

    # Minimum rotated rectangle → orientation
    try:
        mrr = poly.minimum_rotated_rectangle
    except Exception:
        return poly.intersection(tile_box)
    if mrr.is_empty or mrr.geom_type != "Polygon":
        return poly.intersection(tile_box)
    coords = list(mrr.exterior.coords)[:4]

    # Four edges
    edge_lengths = []
    for i in range(4):
        dx = coords[(i + 1) % 4][0] - coords[i][0]
        dy = coords[(i + 1) % 4][1] - coords[i][1]
        edge_lengths.append((dx**2 + dy**2) ** 0.5)

    # Identify long axis vs short axis
    if edge_lengths[0] >= edge_lengths[1]:
        short_len = edge_lengths[1]
        mid_a = ((coords[1][0] + coords[2][0]) / 2,
                 (coords[1][1] + coords[2][1]) / 2)
        mid_b = ((coords[3][0] + coords[0][0]) / 2,
                 (coords[3][1] + coords[0][1]) / 2)
    else:
        short_len = edge_lengths[0]
        mid_a = ((coords[0][0] + coords[1][0]) / 2,
                 (coords[0][1] + coords[1][1]) / 2)
        mid_b = ((coords[2][0] + coords[3][0]) / 2,
                 (coords[2][1] + coords[3][1]) / 2)

    # Direction vector along principal axis
    dx = mid_a[0] - mid_b[0]
    dy = mid_a[1] - mid_b[1]
    length = (dx**2 + dy**2) ** 0.5
    if length < 1e-10:
        return poly.intersection(tile_box)
    ux, uy = dx / length, dy / length

    half_w = short_len / 2

    # Extension lines from each end
    if side=="a":
        ext_line_a = LineString([mid_a,
                                (mid_a[0] + ux * max_extension,
                                mid_a[1] + uy * max_extension)])
        
        # Buffer to match polygon width (flat caps = rectangular extension)
        ext_a = ext_line_a.buffer(half_w, cap_style="flat")
        # Union with original and clip to tile
        elongated = unary_union([poly, ext_a])

    elif side=="b":
        ext_line_b = LineString([mid_b,
                                (mid_b[0] - ux * max_extension,
                                mid_b[1] - uy * max_extension)])
        # Buffer to match polygon width (flat caps = rectangular extension)
        ext_b = ext_line_b.buffer(half_w, cap_style="flat")

        # Union with original and clip to tile
        elongated = unary_union([poly, ext_b])

    if return_suggestion_only:
        elongated = elongated.difference(poly)
    
    elongated = elongated.intersection(tile_box)

    return elongated
    '''
        # Extension lines from each end
        ext_line_a = LineString([mid_a,
                                (mid_a[0] + ux * max_extension,
                                mid_a[1] + uy * max_extension)])
        ext_line_b = LineString([mid_b,
                                (mid_b[0] - ux * max_extension,
                                mid_b[1] - uy * max_extension)])

        # Buffer to match polygon width (flat caps = rectangular extension)
        ext_a = ext_line_a.buffer(half_w, cap_style="flat")
        ext_b = ext_line_b.buffer(half_w, cap_style="flat")

        # Union with original and clip to tile
        elongated = unary_union([poly, ext_a, ext_b])

        if return_suggestion_only:
            elongated = elongated.difference(poly)
        
        elongated = elongated.intersection(tile_box)

        return elongated
    '''

def elongate_tile_polygons(gdf, max_extension, tile_box):
    """Elongate all polygons in a GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame
        Polygons in projected CRS.
    max_extension : float
        Extension distance (meters if projected CRS).
    tile_box : shapely Polygon
        Tile boundary for clipping.

    Returns
    -------
    GeoDataFrame
        Copy with elongated geometries.
    """
    new_geoms = []
    for geom in gdf.geometry:
        if geom.geom_type == "Polygon":
            new_geoms.append(elongate_polygon(geom, max_extension, tile_box))
        elif geom.geom_type == "MultiPolygon":
            parts = [elongate_polygon(p, max_extension, tile_box)
                     for p in geom.geoms]
            new_geoms.append(unary_union(parts))
        else:
            new_geoms.append(geom)
    result = gdf.copy()
    result.geometry = new_geoms
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Suggestion generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_suggestions(
    tile_polys_proj,
    tile_box_utm,
    elongation_dist=10.0,
    compression_dist=2.0,
    min_area_change_frac=0.005,
    max_elongate=None,
    max_compress=None,
    max_remove=None,
    use_centerline=False,
    return_suggestion_only=False,
    convexity_threshold = .6
):
    """Generate polygon modification suggestions for a tile.

    For each polygon, tries elongation, removal, and compression.
    Skips elongations that don't change the tile (e.g. polygon already
    at tile border). Respects per-type limits.

    Parameters
    ----------
    tile_polys_proj : GeoDataFrame
        Polygons in projected CRS (e.g. EPSG:32619), clipped to tile.
    tile_box_utm : shapely Polygon
        Tile bounding box in the same projected CRS.
    elongation_dist : float
        Extension distance in meters.
    compression_dist : float
        Inward buffer distance in meters.
    min_area_change_frac : float
        Minimum area change relative to polygon area to keep a suggestion.
    max_elongate : int or None
        Limit on elongation suggestions (None = unlimited).
    max_compress : int or None
        Limit on compression suggestions (None = unlimited).
    max_remove : int or None
        Limit on removal suggestions (None = unlimited).
    use_centerline : bool
        If True, use centerline-based elongation for non-rectangular polygons.
        Falls back to MRR-based elongation for rectangular ones. Default False.

    Returns
    -------
    list of dict
        Each dict has keys: type, poly_idx, poly_row_idx,
        modified_polys (GeoDataFrame), area_delta_m2, description.
    """
    suggestions = []
    counts = {"elongate": 0, "compress": 0, "remove": 0}
    limits = {
        "elongate": max_elongate if max_elongate is not None else float("inf"),
        "compress": max_compress if max_compress is not None else float("inf"),
        "remove":   max_remove   if max_remove   is not None else float("inf"),
    }

    indices = list(tile_polys_proj.index)

    for pos, idx in enumerate(indices):
        poly = tile_polys_proj.loc[idx, "geometry"]
        if poly.is_empty:
            continue
        if poly.geom_type not in ("Polygon", "MultiPolygon"):
            continue

        poly_area = poly.area

        # ── Elongation ──
        # Skip concave polygons (L/C/S shapes) — MRR elongation is unreliable
        convexity = poly.area / poly.convex_hull.area if poly.convex_hull.area > 0 else 0
        for side in ["a", "b"]:
            if counts["elongate"] < limits["elongate"] and convexity > convexity_threshold:
                if use_centerline:
                    elongated = elongate_polygon_centerline(
                        poly, elongation_dist, tile_box_utm, return_suggestion_only)
                else:
                    # elongated = elongate_polygon(poly, elongation_dist, tile_box_utm)
                    elongated = elongate_polygon(poly, elongation_dist, tile_box_utm, return_suggestion_only,  side=side)

                delta = elongated.area - poly_area
                if delta / max(poly_area, 1e-9) > min_area_change_frac:
                    mod = tile_polys_proj.copy()
                    mod.loc[idx, "geometry"] = elongated
                    suggestions.append({
                        "type": "elongate",
                        "poly_idx": pos,
                        "poly_row_idx": idx,
                        "modified_polys": mod,
                        "area_delta_m2": delta,
                        "description": (f"Elongate polygon {pos} by "
                                        f"{elongation_dist}m  (+{delta:.1f} m²)"),
                    })
                    counts["elongate"] += 1

        # ── Removal ──
        if counts["remove"] < limits["remove"]:
            mod = tile_polys_proj.drop(index=idx).copy()
            suggestions.append({
                "type": "remove",
                "poly_idx": pos,
                "poly_row_idx": idx,
                "modified_polys": mod,
                "area_delta_m2": -poly_area,
                "description": f"Remove polygon {pos}  (−{poly_area:.1f} m²)",
            })
            counts["remove"] += 1

        # ── Compression ──
        if counts["compress"] < limits["compress"]:
            compressed = poly.buffer(-compression_dist)
            if not compressed.is_empty and compressed.area > 0:
                delta = compressed.area - poly_area
                if abs(delta) / max(poly_area, 1e-9) > min_area_change_frac:
                    compressed = compressed.intersection(tile_box_utm)
                    mod = tile_polys_proj.copy()
                    mod.loc[idx, "geometry"] = compressed
                    suggestions.append({
                        "type": "compress",
                        "poly_idx": pos,
                        "poly_row_idx": idx,
                        "modified_polys": mod,
                        "area_delta_m2": delta,
                        "description": (f"Compress polygon {pos} by "
                                        f"{compression_dist}m  ({delta:.1f} m²)"),
                    })
                    counts["compress"] += 1

        # Early exit if all limits reached
        if all(counts[t] >= limits[t] for t in counts):
            break

    return suggestions


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_lines(ax, gdf, bounds, color="cyan", linewidth=1.5, linestyle="-"):
    """Plot a GeoDataFrame of LineString/MultiLineString on a pixel-coord axis.

    Parameters
    ----------
    ax : matplotlib Axes
    gdf : GeoDataFrame
        Line geometries in EPSG:4326.
    bounds : tuple
        (west, south, east, north).
    color, linewidth, linestyle : plot style kwargs.
    """
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        lines = [geom] if isinstance(geom, LineString) else list(geom.geoms)
        for line in lines:
            if len(line.coords) < 2:
                continue
            px = geo_to_px(line.coords, bounds)
            xs, ys = zip(*px)
            ax.plot(xs, ys, color=color, linewidth=linewidth,
                    linestyle=linestyle, solid_capstyle="round")


def plot_polygons(ax, gdf, bounds, facecolor="white", edgecolor="cyan",
                  alpha=0.5, linewidth=0.8):
    """Plot a GeoDataFrame of Polygons as filled patches on a pixel-coord axis.

    Non-polygon geometries (Points, Lines, GeometryCollections with only
    non-polygon parts) are silently skipped.

    Parameters
    ----------
    ax : matplotlib Axes
    gdf : GeoDataFrame
        Polygon geometries in EPSG:4326.
    bounds : tuple
        (west, south, east, north).
    facecolor, edgecolor, alpha, linewidth : patch style kwargs.
    """
    patches = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        elif geom.geom_type == "GeometryCollection":
            polys = [g for g in geom.geoms if g.geom_type == "Polygon"]
        else:
            continue
        for poly in polys:
            px = geo_to_px(poly.exterior.coords, bounds)
            patches.append(MplPolygon(px, closed=True))
    if patches:
        pc = PatchCollection(patches, facecolor=facecolor, edgecolor=edgecolor,
                             alpha=alpha, linewidth=linewidth)
        ax.add_collection(pc)


# Suggestion display colors
_SUG_COLORS = {
    "elongate": {"face": "lime",   "edge": "green",   "highlight": "lime"},
    "remove":   {"face": "red",    "edge": "darkred",  "highlight": "red"},
    "compress": {"face": "orange", "edge": "darkorange", "highlight": "orange"},
}


def display_suggestions(
    suggestions,
    tile_polys,
    sat,
    bounds,
    tile_net=None,
    max_display=10,
    figscale=3.5,
):
    """Display polygon modification suggestions.

    Each row is one suggestion with 3 columns:
      Col 0: Original tile (all polygons, target highlighted)
      Col 1: Modified tile (result of the operation)
      Col 2: Overlay (original cyan + modified in suggestion color)

    Parameters
    ----------
    suggestions : list of dict
        From generate_suggestions().
    tile_polys : GeoDataFrame
        Original polygons in EPSG:4326.
    sat : ndarray (H, W, 3)
        Satellite image.
    bounds : tuple
        (west, south, east, north).
    tile_net : GeoDataFrame or None
        Optional network lines in EPSG:4326.
    max_display : int
        Max rows to show.
    figscale : float
        Size multiplier per subplot.
    """
    n = min(len(suggestions), max_display)
    if n == 0:
        print("No suggestions to display.")
        return

    fig, axes = plt.subplots(n, 3, figsize=(figscale * 3, figscale * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, sug in enumerate(suggestions[:max_display]):
        stype = sug["type"]
        colors = _SUG_COLORS[stype]
        pidx = sug["poly_idx"]
        mod_proj = sug["modified_polys"]
        mod_4326 = mod_proj.to_crs("EPSG:4326")

        target_row = tile_polys.iloc[[pidx]]

        # Col 0: Original with target highlighted
        ax = axes[i, 0]
        ax.imshow(sat, alpha=0.7)
        plot_polygons(ax, tile_polys, bounds, facecolor="cyan",
                      edgecolor="blue", alpha=0.3, linewidth=0.5)
        plot_polygons(ax, target_row, bounds, facecolor=colors["highlight"],
                      edgecolor="black", alpha=0.6, linewidth=1.5)
        if tile_net is not None:
            plot_lines(ax, tile_net, bounds, color="gray",
                       linewidth=0.8, linestyle=":")
        if i == 0:
            ax.set_title("Original\n(target highlighted)", fontsize=9)
        ax.set_ylabel(f"#{i}", fontsize=9, rotation=0, labelpad=15)

        # Col 1: Modified result
        ax = axes[i, 1]
        ax.imshow(sat, alpha=0.7)
        plot_polygons(ax, mod_4326, bounds, facecolor=colors["face"],
                      edgecolor=colors["edge"], alpha=0.4, linewidth=0.8)
        if tile_net is not None:
            plot_lines(ax, tile_net, bounds, color="gray",
                       linewidth=0.8, linestyle=":")
        if i == 0:
            ax.set_title("Modified", fontsize=9)

        # Col 2: Overlay
        ax = axes[i, 2]
        ax.imshow(sat, alpha=0.7)
        plot_polygons(ax, tile_polys, bounds, facecolor="cyan",
                      edgecolor="blue", alpha=0.2, linewidth=0.5)
        plot_polygons(ax, mod_4326, bounds, facecolor=colors["face"],
                      edgecolor=colors["edge"], alpha=0.35, linewidth=1)
        if tile_net is not None:
            plot_lines(ax, tile_net, bounds, color="gray",
                       linewidth=0.8, linestyle=":")
        ax.set_xlabel(sug["description"], fontsize=7)
        if i == 0:
            ax.set_title("Overlay", fontsize=9)

    for ax in axes.flat:
        ax.set_xlim(0, 256)
        ax.set_ylim(256, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f"Polygon Modification Suggestions ({n} shown)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
