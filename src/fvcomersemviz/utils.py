from __future__ import annotations
"""
Utilities.
"""

from pathlib import Path
import inspect
from typing import Iterable, Tuple, Optional, Dict, Any, Literal, List, Sequence
import os
import numpy as np
import matplotlib.tri as mtri
import xarray as xr
import pandas as pd
import xarray as xr
from pyproj import Geod




from .io import eval_group_or_var  
# utils.py


def is_absolute_z(depth: Any) -> bool:
    """
    Returns True if `depth` requests an absolute z slice (meters),
    e.g. -10.0, ("z_m", -10.0), or {"z_m": -10.0}.
    Returns False for sigma-relative selectors like -0.7 .. 0.0 or "surface"/"bottom"/"depth_avg".
    """
    if depth is None:
        return False

    # numeric → absolute if *not* a sigma-relative number in [-1, 0]
    if isinstance(depth, (float, np.floating, int)):
        return not (-1.0 <= float(depth) <= 0.0)

    # tuple form: ("z_m", -10.0)
    if isinstance(depth, tuple) and len(depth) > 0:
        return depth[0] == "z_m"

    # dict form: {"z_m": -10.0}
    if isinstance(depth, dict):
        return "z_m" in depth

    return False

def resolve_da_with_depth(
    ds: xr.Dataset,
    var: str,
    *,
    depth: Any = "surface",
    groups: Optional[dict] = None,
    verbose: bool = False,
) -> xr.DataArray:
    ds_scoped = ds
    try:
        ds_scoped = select_depth(ds, depth, verbose=verbose)  
    except Exception:
        pass

    da = eval_group_or_var(ds_scoped, var, groups)

    if "siglay" in da.dims:
        if isinstance(depth, (float, np.floating)) and not (-1.0 <= float(depth) <= 0.0):
            da = select_da_by_z(da, ds_scoped, float(depth), verbose=verbose)
        elif isinstance(depth, tuple) and len(depth) > 0 and depth[0] == "z_m":
            da = select_da_by_z(da, ds_scoped, float(depth[1]), verbose=verbose)
        elif isinstance(depth, dict) and "z_m" in depth:
            da = select_da_by_z(da, ds_scoped, float(depth["z_m"]), verbose=verbose)
        return da

    if verbose:
        print(f"[resolve] '{var}' has no 'siglay' — lifting to single layer.")
    sig = xr.DataArray([-0.5], dims=["siglay"], name="siglay")
    da = da.expand_dims(siglay=sig)
    da["siglay"] = sig
    return da



def _lon_lat_arrays_nodes_only(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 1D longitude and latitude arrays from node-based coordinates.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray Dataset expected to contain node-level coordinates
        named 'lon' and 'lat'.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Flattened longitude and latitude arrays (1D).

    Raises
    ------
    ValueError
        If the dataset does not contain 'lon' and 'lat' variables.
    """
    if "lon" in ds and "lat" in ds:
        return np.asarray(ds["lon"].values).ravel(), np.asarray(ds["lat"].values).ravel()
    raise ValueError("Node triangulation requires 'lon' and 'lat' (node coordinates).")


def _lon_lat_arrays_any(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 1D longitude and latitude arrays from either node or cell-center coordinates.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray Dataset that may contain either:
        - 'lon' and 'lat' (node coordinates), or
        - 'lonc' and 'latc' (cell-center coordinates).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Flattened longitude and latitude arrays (1D).

    Raises
    ------
    ValueError
        If neither ('lon','lat') nor ('lonc','latc') are present in the dataset.
    """
    if "lon" in ds and "lat" in ds:
        return np.asarray(ds["lon"].values).ravel(), np.asarray(ds["lat"].values).ravel()
    if "lonc" in ds and "latc" in ds:
        return np.asarray(ds["lonc"].values).ravel(), np.asarray(ds["latc"].values).ravel()
    raise ValueError("Dataset needs 'lon'/'lat' or 'lonc'/'latc'.")


def build_triangulation(ds: xr.Dataset) -> mtri.Triangulation:
    """
    Build a matplotlib Triangulation from FVCOM-style grid data.

    This function constructs a triangular mesh based on available
    node connectivity ('nv') and coordinate variables. If the dataset
    defines the FVCOM connectivity table ('nv'), it uses that to build
    an exact triangulation. Otherwise, it falls back to a Delaunay
    triangulation of available coordinates (node or cell centers).

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray Dataset containing either:
        - 'lon' and 'lat' node coordinates and an 'nv' connectivity table, or
        - only coordinates ('lon','lat') or ('lonc','latc') for fallback Delaunay mesh.

    Returns
    -------
    mtri.Triangulation
        A matplotlib Triangulation object representing the grid topology.

    Raises
    ------
    ValueError
        If 'nv' exists but is not 2D or does not contain exactly 3 indices per triangle.
        If required coordinate variables ('lon'/'lat' or 'lonc'/'latc') are missing.

    Notes
    -----
    - For standard FVCOM grids, 'nv' is expected to be shaped (3, nele) or (nele, 3),
      with 1-based node indices. They are converted internally to 0-based indexing.
    - When 'nv' is absent, the function falls back to a Delaunay triangulation
      built from the available lon/lat coordinate set (nodes or cell centers).
    """
    if "nv" in ds:
        # --- REQUIRE node coords with nv
        x, y = _lon_lat_arrays_nodes_only(ds)
        nv = np.asarray(ds["nv"].values)
        if nv.ndim != 2 or 3 not in nv.shape:
            raise ValueError("nv must be 2D with 3 entries per triangle (shape (3, nele) or (nele, 3)).")
        triangles = (nv.T if nv.shape[0] == 3 else nv) - 1
        triangles = triangles.astype(int)
        return mtri.Triangulation(x, y, triangles=triangles)

    # --- Delaunay fallback: allow either node or element centers
    x, y = _lon_lat_arrays_any(ds)
    return mtri.Triangulation(x, y)

def robust_clims(a: Iterable[float], q: Tuple[float, float] = (5, 95)) -> tuple[float, float]:
    """
    Robust color limits from percentiles; handles NaNs and constant arrays.
    """
    arr = np.asarray(a).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(arr, q)
    if not np.isfinite(lo):
        lo = float(np.nanmin(arr))
    if not np.isfinite(hi):
        hi = float(np.nanmax(arr))
    if lo == hi:
        hi = lo + (abs(lo) if lo != 0 else 1.0)
    return float(lo), float(hi)


def file_prefix(base_dir: str) -> str:
    return os.path.basename(os.path.normpath(base_dir))

def _caller_plot_module_stem(default: str | None = None) -> str | None:
    """
    Walk the call stack and return the module filename stem if the caller is inside
    fvcomersemviz.plots.* (e.g., 'timeseries', 'maps'). Otherwise return default.
    """
    for frame_info in inspect.stack():
        mod = inspect.getmodule(frame_info.frame)
        if not mod:
            continue
        name = getattr(mod, "__name__", "")
        file = getattr(mod, "__file__", None)
        if name.startswith("fvcomersemviz.plots.") and file:
            return Path(file).stem
    return default

def out_dir(base_dir: str, figures_root: str) -> str:
    """
    Return an output directory and ensure it exists.

    Base path:
        FIG_DIR/<basename(BASE_DIR)>/

    If called from a plotting module under fvcomersemviz.plots.*, automatically append
    a subfolder named after that module file (e.g., 'timeseries', 'maps'):
        FIG_DIR/<basename(BASE_DIR)>/<module-stem>/

    You can override the subfolder via environment variable FVCOM_PLOT_SUBDIR.
      - If set to a non-empty value -> use that subfolder name.
      - If set to an empty string   -> disable subfoldering (use base).
    """
    folder = file_prefix(base_dir)
    base = os.path.join(figures_root, folder)
    os.makedirs(base, exist_ok=True)

    # 1) explicit override via env
    env = os.environ.get("FVCOM_PLOT_SUBDIR", None)
    if env is not None:
        sub = env.strip()
        if sub:  # non-empty -> use it
            d = os.path.join(base, sub)
            os.makedirs(d, exist_ok=True)
            return d
        # empty string -> disable subfolder
        return base

    # 2) auto-detect when called from fvcomersemviz.plots.*
    sub = _caller_plot_module_stem(default=None)
    if sub:
        d = os.path.join(base, sub)
        os.makedirs(d, exist_ok=True)
        return d

    # 3) default: just the base
    return base


def weighted_mean_std(
    da: xr.DataArray,
    dims: Iterable[str],
    weights: Optional[xr.DataArray] = None,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Mean and std (±1σ) of `da` reduced over `dims`. If `weights` is given, it's applied
    along its own dims (typically spatial dims like 'node' or 'nele') and broadcast
    over any extra dims in `da` (e.g., 'time', 'siglay').

    - Uses population variance (ddof=0) for stability.
    - Ignores NaNs in data via masking weights where da is NaN.
    - If `dims` is empty, returns (da, zeros_like(da)).
    """
    dims = list(dims)
    if len(dims) == 0:
        return da, xr.zeros_like(da)

    if weights is None:
        return da.mean(dims, skipna=True), da.std(dims, skipna=True)

    # Align and broadcast weights to the data
    try:
        # Label-based select where possible (e.g., after masking by node index coords)
        sel = {d: da[d] for d in weights.dims if d in da.dims and d in da.coords}
        if sel:
            w = weights.sel(sel)
        else:
            w = weights
    except Exception:
        w = weights

    # Align to shared coords (inner join) and broadcast to data shape
    w, da_aligned = xr.align(w, da, join="inner", copy=False)
    w = w.broadcast_like(da_aligned)

    # Mask weights where data is NaN
    w = w.where(da_aligned.notnull())

    wsum = w.sum(dims)
    # Avoid divide-by-zero
    wsum = xr.where(wsum == 0, np.nan, wsum)

    mean = (da_aligned * w).sum(dims, skipna=True) / wsum
    var = ((da_aligned - mean) ** 2 * w).sum(dims, skipna=True) / wsum
    std = np.sqrt(var)
    return mean, std
    
def element_mask_from_node_mask(ds: xr.Dataset, node_mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Convert node mask -> element mask using ds['nv'] (keep element if all 3 nodes inside).
    Returns None if connectivity is unavailable.
    """
    if "nv" not in ds:
        return None
    nv = np.asarray(ds["nv"].values)
    if nv.ndim != 2 or 3 not in nv.shape:
        return None
    tri = (nv.T if nv.shape[0] == 3 else nv) - 1
    tri = tri.astype(int)
    inside = node_mask[tri]
    keep = inside.all(axis=1)
    return keep


def style_get(var: str, styles: Optional[Dict[str, Dict[str, Any]]], key: str, default=None):
    """Lookup styles[var][key] with a safe default."""
    if not styles:
        return default
    s = styles.get(var)
    if not s:
        return default
    return s.get(key, default)

# ---- depth selection (sigma-based, plus deferral for absolute-z) ----
def select_depth(
    ds: xr.Dataset,
    mode: Any,
    *,
    thickness_var: str = "layer_thickness",
    verbose: bool = False,
) -> xr.Dataset:
    """
    Return a view of `ds` sliced in the vertical according to `mode`.

    Accepts:
      - "surface" | "bottom" | "depth_avg"
      - int (sigma index), float in [-1, 0] (sigma value)
      - {"sigma_index": int} | {"sigma_value": float}
      - {"z_m": float} or ("z_m", float) -> deferred (return ds unchanged; slice per-variable via select_da_by_z)
    """
    # No vertical coord? Nothing to slice here.
    if "siglay" not in ds.dims:
        return ds

    # Absolute-z requests are deferred; per-variable functions call select_da_by_z(...)
    if isinstance(mode, dict) and "z_m" in mode:
        return ds
    if isinstance(mode, tuple) and len(mode) > 0 and mode[0] == "z_m":
        return ds
    if isinstance(mode, (float, np.floating)) and not (-1.0 <= float(mode) <= 0.0):
        return ds

    # canonical string modes
    if isinstance(mode, str):
        m = mode.lower()
        if m == "surface":
            return ds.isel(siglay=0)
        if m == "bottom":
            return ds.isel(siglay=-1)
        if m == "depth_avg":
            if thickness_var in ds and "siglay" in ds[thickness_var].dims:
                w = ds[thickness_var]
                w = w / w.sum("siglay")
                return (ds * w).sum("siglay")
            return ds.mean("siglay")
        # else try to parse as number (sigma value)
        try:
            sval = float(m)
            mode = sval
        except Exception:
            raise ValueError(
                "depth must be one of 'surface'|'bottom'|'depth_avg', an int layer index, "
                "a sigma value in [-1,0], or a z request {'z_m': -10}."
            )

    # numeric modes
    if isinstance(mode, (int, np.integer)):
        return ds.isel(siglay=int(mode))

    if isinstance(mode, (float, np.floating)):
        s = float(mode)
        if -1.0 <= s <= 0.0:
            # try coordinate-based nearest selection first
            if "siglay" in ds.coords and ds["siglay"].ndim == 1:
                return ds.sel(siglay=s, method="nearest")
            # fallback: manual nearest in the 1D sigma vector
            sig = ds["siglay"]
            if sig.ndim == 1:
                k = int(np.nanargmin(np.abs(np.asarray(sig.values) - s)))
                return ds.isel(siglay=k)
            raise ValueError("Cannot select by sigma value: 'siglay' is not a 1-D coordinate.")
        # else absolute z was already caught above; return ds
        return ds

    if isinstance(mode, dict):
        if "sigma_index" in mode:
            return ds.isel(siglay=int(mode["sigma_index"]))
        if "sigma_value" in mode:
            return select_depth(ds, float(mode["sigma_value"]), thickness_var=thickness_var, verbose=verbose)

    raise ValueError(f"Unsupported depth selector: {mode!r}")

# slice a DataArray at an absolute (z, metres) depth
def select_da_by_z(
    da: xr.DataArray,
    ds_full: xr.Dataset,
    z_target: float,
    *,
    method: "Literal['nearest']" = "nearest",  # reserved for future
    verbose: bool = False,
) -> xr.DataArray:
    """
    Slice `da` at absolute depth `z_target` (metres, negative downward).
    Works for node-centred (time,siglay,node) and element-centred (time,siglay,nele) data.

    Notes
    -----
    - Builds/uses z/z_nele via ensure_z_from_sigma(ds_full).
    - Aligns z arrays to `da` (after masking/isels).
    - Computes the argmin indexer, then wraps it as an xr.DataArray with dims matching
      the *non-siglay* dims (e.g. ('time','node')), which xarray requires for vectorized isel.
    - Avoids Dask's "chunked array indexer" error by computing the indexer eagerly.
    """
    # Lazily ensure vertical coords on the full dataset (not spatially subset)
    from fvcomersemviz.io import ensure_z_from_sigma
    ds_z = ensure_z_from_sigma(ds_full, verbose=verbose)

    # Align a z-like array to da’s dims/coords
    def _align(zlike: xr.DataArray) -> xr.DataArray:
        out = zlike
        if "time" in da.dims and "time" in out.dims:
            out = out.sel(time=da["time"])
        if "node" in da.dims and "node" in out.dims:
            if ("node" in da.coords) and (getattr(da["node"], "ndim", 1) == 1):
                out = out.sel(node=da["node"])
            else:
                out = out.isel(node=xr.DataArray(np.arange(da.sizes["node"]), dims=("node",)))
        if "nele" in da.dims and "nele" in out.dims:
            if ("nele" in da.coords) and (getattr(da["nele"], "ndim", 1) == 1):
                out = out.sel(nele=da["nele"])
            else:
                out = out.isel(nele=xr.DataArray(np.arange(da.sizes["nele"]), dims=("nele",)))
        return out

    # Helper: compute → numpy → wrap back with proper named dims for xarray vectorized isel
    def _to_da_indexer(idx: xr.DataArray, z_aligned: xr.DataArray) -> xr.DataArray:
        # idx dims are z_aligned.dims without 'siglay' (e.g., ('time','node') or ('node',))
        idx_dims = tuple(d for d in z_aligned.dims if d != "siglay")
        try:
            idx = idx.compute()
        except Exception:
            pass
        return xr.DataArray(np.asarray(idx), dims=idx_dims)

    # --- Node-centred ---
    if "siglay" in da.dims and "node" in da.dims and "z" in ds_z:
        z = _align(ds_z["z"])  # dims like ('time','siglay','node') or subset
        idx = (np.abs(z - z_target)).argmin("siglay").astype("int64")  # dims: ('time','node') or ('node',)
        idx_da = _to_da_indexer(idx, z)
        return da.isel(siglay=idx_da)

    # --- Element-centred ---
    if "siglay" in da.dims and "nele" in da.dims and "z_nele" in ds_z:
        zc = _align(ds_z["z_nele"])  # dims like ('time','siglay','nele') or subset
        idx = (np.abs(zc - z_target)).argmin("siglay").astype("int64")  # dims: ('time','nele') or ('nele',)
        idx_da = _to_da_indexer(idx, zc)
        return da.isel(siglay=idx_da)

    # --- Fallbacks: time-only or other non-spatial cases with 'siglay' ---
    if "siglay" in da.dims and ("node" not in da.dims) and ("nele" not in da.dims):
        # Use first available vertical coordinate (node=0 or nele=0), align to da's time,
        # then vectorized integer indexing over time.
        if "z" in ds_z:
            zref = ds_z["z"].isel(node=0)          # dims: ('time','siglay') or ('siglay',)
        elif "z_nele" in ds_z:
            zref = ds_z["z_nele"].isel(nele=0)     # dims: ('time','siglay') or ('siglay',)
        else:
            raise ValueError("No 'z' or 'z_nele' vertical coordinates available.")
    
        # Align time (if present)
        if "time" in da.dims and "time" in zref.dims:
            zref = zref.sel(time=da["time"])
    
        # Nearest sigma layer per timestep (vectorized over time)
        idx = (np.abs(zref - z_target)).argmin("siglay").astype("int64")  # dims: () or ('time',)
    
        if idx.ndim == 0:
            # truly scalar index
            return da.isel(siglay=int(idx.values))
    
        # vectorized integer indexer: dims must match da except for 'siglay'
        idx_da = xr.DataArray(idx.values, dims=tuple(idx.dims))
        return da.isel(siglay=idx_da)


    # Nothing to do
    return da



# ---- nearest station index (nodes or elements) ----
def nearest_index_for_dim(
    ds: xr.Dataset,
    lat: float,
    lon: float,
    dim: "Literal['node','nele']",
) -> int:
    """Return nearest grid index to (lat, lon) for nodes ('lon'/'lat') or elements ('lonc'/'latc')."""
    from pyproj import Geod
    if dim == "node":
        if "lon" not in ds or "lat" not in ds:
            raise ValueError("Need 'lon'/'lat' for node lookup.")
        xs = np.asarray(ds["lon"].values).ravel()
        ys = np.asarray(ds["lat"].values).ravel()
    elif dim == "nele":
        if "lonc" not in ds or "latc" not in ds:
            raise ValueError("Need 'lonc'/'latc' for element-center lookup.")
        xs = np.asarray(ds["lonc"].values).ravel()
        ys = np.asarray(ds["latc"].values).ravel()
    else:
        raise ValueError("dim must be 'node' or 'nele'.")

    geod = Geod(ellps="WGS84")
    _, _, dist = geod.inv(np.full_like(xs, lon), np.full_like(ys, lat), xs, ys)
    return int(np.nanargmin(dist))


# ---- time & depth label builders for titles / filenames ----
def build_time_window_label(
    months: Optional[Iterable[int]],
    years: Optional[Iterable[int]],
    start_date: Optional[str],
    end_date: Optional[str],
) -> str:
    """e.g., 'Jan-Mar__2020-2021' or '2022-01-01 to 2022-02-01' or 'AllTime'."""
    names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    parts: list[str] = []
    if months:
        m = sorted({int(x) for x in months})
        if len(m) > 1 and m == list(range(m[0], m[-1] + 1)):
            parts.append(f"{names[m[0]-1]}-{names[m[-1]-1]}")
        else:
            parts.append("-".join(names[i-1] for i in m))
    if years:
        y = sorted({int(x) for x in years})
        parts.append(f"{y[0]}-{y[-1]}" if len(y) > 1 else f"{y[0]}")
    if start_date or end_date:
        parts.append(f"{start_date or '...'} to {end_date or '...'}")
    return "__".join(parts) if parts else "AllTime"


def depth_tag(depth: Any) -> str:
    """Map depth selector to a short tag: 'surface', 'bottom', 'zavg', 'k3', 'sigma-0.3', 'zm-10'."""
    if isinstance(depth, str):
        m = depth.lower()
        if m in ("surface", "bottom", "depth_avg"):
            return {"surface": "surface", "bottom": "bottom", "depth_avg": "zavg"}[m]
        # maybe a stringified sigma value
        try:
            depth = float(m)
        except Exception:
            return m
    if isinstance(depth, (int, np.integer)):
        return f"k{int(depth)}"
    if isinstance(depth, (float, np.floating)):
        return f"sigma-{float(depth):g}" if -1.0 <= float(depth) <= 0.0 else f"zm-{abs(float(depth)):g}"
    if isinstance(depth, tuple) and len(depth) > 0 and depth[0] == "z_m":
        return f"zm-{abs(float(depth[1])):g}"
    if isinstance(depth, dict) and "z_m" in depth:
        return f"zm-{abs(float(depth['z_m'])):g}"
    return str(depth)
    
def align_flatten_pair(
    x_da: xr.DataArray,
    y_da: xr.DataArray,
    *,
    sample_max: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Align x,y by index, flatten, drop non-finite; optionally subsample to sample_max."""
    x_al, y_al = xr.align(x_da, y_da, join="inner")
    x = np.asarray(x_al.values).ravel()
    y = np.asarray(y_al.values).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if sample_max is not None and x.size > sample_max:
        rng = rng or np.random.default_rng()
        sel = rng.choice(x.size, size=sample_max, replace=False)
        x = x[sel]; y = y[sel]
    return x, y


def resolve_available_vars(ds: xr.Dataset, names: Sequence[str]) -> List[str]:
    """
    Return subset of 'names' present in ds. Includes a light alias:
    - 'Z6' -> 'Z6_c' if only the latter exists.
    """
    have = set(ds.data_vars)
    out: List[str] = []
    for n in names:
        if n in have:
            out.append(n)
        elif n == "Z6" and "Z6_c" in have:
            out.append("Z6_c")
    return out


def sum_over_all_dims(da: xr.DataArray) -> float:
    """
    Sum over all dims with skipna=True and return a Python float.
    Works for both NumPy- and Dask-backed DataArrays.
    """
    if da.size == 0:
        return np.nan

    # Collapse across all dims in one go (xarray supports a list of dims)
    s = da.sum(list(da.dims), skipna=True)

    # If Dask-backed, compute eagerly to get a NumPy scalar/0-d array
    try:
        # DataArray has .compute() when Dask is available
        if hasattr(s, "compute"):
            s = s.compute()
    except Exception:
        # If compute fails for some reason, fall through to numpy coercion below
        pass

    # At this point s should be a scalar DataArray (NumPy-backed).
    # Convert robustly to a Python float.
    return float(np.asarray(s.data).reshape(()).item())

