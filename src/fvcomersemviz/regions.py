
from __future__ import annotations
"""
Region helpers.

"""

from typing import Optional, Tuple, Sequence, Dict, Any
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon, Point, MultiPoint
from shapely.ops import unary_union
from shapely.prepared import prep as prep_geom


import xarray as xr

# Add to src/fvcomersemviz/regions.py

from typing import Optional, Tuple, Dict, Any
import numpy as np
import xarray as xr

def nearest_node_index(ds: xr.Dataset, lat: float, lon: float) -> Optional[int]:
    """Nearest node by equirectangular distance; assumes 1D 'lat'/'lon' on node center."""
    lat_name = "lat" if "lat" in ds else ("Latitude" if "Latitude" in ds else None)
    lon_name = "lon" if "lon" in ds else ("Longitude" if "Longitude" in ds else None)
    if lat_name is None or lon_name is None:
        return None
    lat_arr = np.asarray(ds[lat_name]).ravel()
    lon_arr = np.asarray(ds[lon_name]).ravel()
    if lat_arr.size != lon_arr.size:
        return None
    dlat = np.deg2rad(lat_arr - lat)
    dlon = np.deg2rad(lon_arr - lon)
    w = np.cos(np.deg2rad(lat))
    dist2 = (dlat**2) + (w * dlon)**2
    return int(np.nanargmin(dist2))

def apply_scope(
    ds: xr.Dataset,
    *,
    region: Optional[Tuple[str, Dict[str, Any]]] = None,
    station: Optional[Tuple[str, float, float]] = None,  # (name, lat, lon)
    verbose: bool = False,
) -> xr.Dataset:
    """Station (nearest node) ? region mask ? domain."""
    if station is not None:
        _, lat, lon = station
        idx = nearest_node_index(ds, lat, lon)
        if idx is not None:
            if "node" in ds.dims: return ds.isel(node=idx)
            if "nele" in ds.dims: return ds.isel(nele=idx)
            if verbose:
                print("[regions/apply_scope] dataset not node/nele-centered; using as-is.")
        elif verbose:
            print("[regions/apply_scope] station selection failed; falling back to region/domain.")
    mask_nodes, mask_elems = build_region_masks(ds, region, verbose=verbose)
    if mask_nodes is None and mask_elems is None:
        return ds
    out = ds
    for v in list(out.data_vars):
        out[v] = apply_prebuilt_mask(out[v], mask_nodes, mask_elems)
    return out


def build_region_masks(
    ds: xr.Dataset,
    region: Optional[Tuple[str, Dict[str, Any]]],
    *,
    verbose: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Return (mask_nodes, mask_elems) or (None, None) if no/failed region.
    """
    if region is None:
        return None, None

    region_name, spec = region
    try:
        if "shapefile" in spec:
            mask_nodes = polygon_mask_from_shapefile(
                ds, spec["shapefile"],
                name_field=spec.get("name_field"),
                name_equals=spec.get("name_equals"),
            )
        elif "csv_boundary" in spec:
            poly = polygon_from_csv_boundary(
                spec["csv_boundary"],
                lon_col=spec.get("lon_col", "lon"),
                lat_col=spec.get("lat_col", "lat"),
                normalize_lon=True,
                sort=spec.get("sort", "auto"),
                convex_hull=spec.get("convex_hull", False),
            )
            mask_nodes = polygon_mask(ds, poly)
        else:
            raise ValueError("Region spec must contain 'shapefile' or 'csv_boundary'.")
    except Exception as e:
        if verbose:
            print(f"[regions/{region_name}] region mask failed: {e}; using full domain.")
        return None, None

    if not np.any(mask_nodes):
        if verbose:
            print(f"[regions/{region_name}] mask empty; using full domain.")
        return None, None

    mask_elems = element_mask_from_node_mask(ds, mask_nodes)
    if mask_elems is None and verbose:
        print(f"[regions/{region_name}] cannot build element mask (no nv); using node mask only.")
    return mask_nodes, mask_elems


def apply_prebuilt_mask(
    da: xr.DataArray,
    mask_nodes: Optional[np.ndarray],
    mask_elems: Optional[np.ndarray],
) -> xr.DataArray:
    """Apply precomputed node/element mask if center matches; else return da unchanged."""
    if "node" in da.dims and mask_nodes is not None:
        idx = np.where(mask_nodes)[0]
        return da.isel(node=idx)
    if "nele" in da.dims and mask_elems is not None:
        idx = np.where(mask_elems)[0]
        return da.isel(nele=idx)
    return da


def _lon_lat_arrays(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    if "lon" in ds and "lat" in ds:
        lon = np.asarray(ds["lon"].values).ravel()
        lat = np.asarray(ds["lat"].values).ravel()
        return lon, lat
    raise ValueError("Region masking expects node coords 'lon' and 'lat'.")


def _normalize_lon_180(lons: np.ndarray) -> np.ndarray:
    """Wrap to [-180, 180]."""
    lons = np.asarray(lons, dtype=float)
    return ((lons + 180.0) % 360.0) - 180.0


def polygon_mask_from_shapefile(
    ds: xr.Dataset,
    shapefile: str,
    name_field: Optional[str] = None,
    name_equals: Optional[str] = None,
    include_boundary: bool = True,   # NEW: inclusive by default
) -> np.ndarray:
    gdf = gpd.read_file(shapefile)
    if name_field is not None and name_equals is not None:
        gdf = gdf[gdf[name_field] == name_equals]
    if gdf.empty:
        raise ValueError("No geometries found in shapefile with given filters.")
    geom = unary_union(gdf.geometry.values)
    return polygon_mask(ds, geom, include_boundary=include_boundary)


def polygon_from_csv_boundary(
    csv_path: str,
    lon_col: str = "lon",
    lat_col: str = "lat",
    *,
    normalize_lon: bool = True,
    sort: str = "auto",          # "auto" (angle sort), "none"
    convex_hull: bool = False,   # if True, ignore sort and use convex hull of points
) -> Polygon:
    """
    Create a polygon from a CSV of boundary coordinates.
    - If points are unordered, use convex_hull=True or sort='auto' to produce a valid ring.
    - Longitudes can be normalized to [-180, 180] to match the dataset.
    """
    df = pd.read_csv(csv_path)
    if lon_col not in df.columns or lat_col not in df.columns:
        lon_col, lat_col = df.columns[:2]
    lons = df[lon_col].astype(float).to_numpy()
    lats = df[lat_col].astype(float).to_numpy()

    if normalize_lon:
        lons = _normalize_lon_180(lons)

    pts = np.column_stack([lons, lats])
    # Drop NaNs / duplicates
    mask_finite = np.isfinite(pts).all(axis=1)
    pts = pts[mask_finite]
    pts = np.unique(pts, axis=0)

    if pts.shape[0] < 3:
        raise ValueError("Need at least 3 distinct points to form a polygon.")

    if convex_hull:
        hull = MultiPoint([Point(x, y) for x, y in pts]).convex_hull
        poly = hull if isinstance(hull, Polygon) else hull.buffer(0)
    else:
        if sort == "auto":
            # Angle sort around centroid to create a simple ring
            cx, cy = pts.mean(axis=0)
            angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
            order = np.argsort(angles)
            ring = pts[order]
        else:
            ring = pts  # assume CSV already provides an ordered ring
        # Ensure closed ring
        if not np.allclose(ring[0], ring[-1]):
            ring = np.vstack([ring, ring[0]])
        poly = Polygon(ring)

    if not poly.is_valid:
        poly = poly.buffer(0)  # fix self-intersections

    return poly


def polygon_mask(
    ds: xr.Dataset,
    polygon: Polygon,
    *,
    include_boundary: bool = True,   # NEW: include boundary points
) -> np.ndarray:
    """Boolean mask for nodes contained in the given polygon (union ok)."""
    lon, lat = _lon_lat_arrays(ds)
    if include_boundary:
        f = np.frompyfunc(lambda x, y: polygon.covers(Point(x, y)), 2, 1)
        return f(lon, lat).astype(bool)
    else:
        P = prep_geom(polygon)
        f = np.frompyfunc(lambda x, y: P.contains(Point(x, y)), 2, 1)
        return f(lon, lat).astype(bool)


def polygon_mask_elements(
    ds: xr.Dataset,
    polygon: Polygon,
    *,
    include_boundary: bool = True,
) -> np.ndarray:
    """Boolean mask for elements contained in polygon using 'lonc'/'latc' (element centers)."""
    if "lonc" not in ds or "latc" not in ds:
        raise ValueError("Need 'lonc'/'latc' for element-center region masking.")
    lons = np.asarray(ds["lonc"].values).ravel()
    lats = np.asarray(ds["latc"].values).ravel()
    if include_boundary:
        f = np.frompyfunc(lambda x, y: polygon.covers(Point(x, y)), 2, 1)
    else:
        P = prep_geom(polygon)
        f = np.frompyfunc(lambda x, y: P.contains(Point(x, y)), 2, 1)
    return f(lons, lats).astype(bool)

def element_mask_from_node_mask(
    ds: xr.Dataset,
    node_mask: np.ndarray,
    *,
    strict: bool = True,
) -> Optional[np.ndarray]:
    """
    Convert a boolean node mask to an element mask using 'nv'.

    strict=True  -> keep an element only if all 3 corner nodes are inside.
    strict=False -> (optional future) keep if >=2 of 3 nodes are inside.
    Returns None if 'nv' is missing or ill-shaped.
    """

