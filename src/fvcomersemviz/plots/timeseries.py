# plots/timeseries.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from matplotlib.colors import to_rgb, to_hex
    
from ..io import (
    filter_time,
    eval_group_or_var,

)
from ..regions import (
    polygon_mask_from_shapefile,
    polygon_from_csv_boundary,
    polygon_mask,
    element_mask_from_node_mask,
)
from ..utils import (
    out_dir,
    file_prefix, 
    weighted_mean_std, 
    style_get,
    select_depth, 
    select_da_by_z, 
    build_time_window_label, 
    depth_tag,
    nearest_index_for_dim,
    resolve_da_with_depth,
    is_absolute_z,

)

def _vprint(verbose: bool, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def _space_mean(da: xr.DataArray, ds: xr.Dataset, *, verbose: bool=False) -> xr.DataArray:
    """Mean across all non-time dims; area-weight with art1 if available and alignable.

    If 'art1' exists and spans the same spatial dims, we subset/align the weights
    to the data's coordinates (after any masking/isel) before computing a weighted mean.
    If alignment isn't possible, we fall back to a simple mean with a verbose note.
    """
    space_dims = [d for d in da.dims if d != "time"]
    if not space_dims:
        _vprint(verbose, "[space-mean] No spatial dimensions; returning as is.")
        return da

    if "art1" in ds and all(d in ds["art1"].dims for d in space_dims):
        w = ds["art1"]

        # Align weights to da along each spatial dim using label-based selection
        for d in space_dims:
            if d in w.dims and d in da.dims:
                if da.sizes[d] != w.sizes[d]:
                    if d in da.coords and d in w.coords and da.coords[d].ndim == 1 and w.coords[d].ndim == 1:
                        try:
                            w = w.sel({d: da[d]})
                            _vprint(verbose, f"[space-mean] Matched weights to '{d}' via .sel; size={w.sizes[d]}.")
                        except Exception as e:
                            _vprint(verbose, f"[space-mean] Failed to align weights on '{d}' ({e}); using simple mean.")
                            return da.mean(space_dims, skipna=True)
                    else:
                        _vprint(verbose, f"[space-mean] No coord labels for '{d}' to align weights; using simple mean.")
                        return da.mean(space_dims, skipna=True)

        _vprint(verbose, f"[space-mean] Area-weighted mean over {space_dims} using 'art1'.")
        num = (da * w).sum(space_dims, skipna=True)
        den = w.sum(space_dims, skipna=True)
        return num / den

    _vprint(verbose, f"[space-mean] Simple mean over {space_dims} (no suitable 'art1').")
    return da.mean(space_dims, skipna=True)


def _time_index(da: xr.DataArray) -> pd.DatetimeIndex:
    # Lazy-friendly extraction of time index
    if "time" not in da.dims:
        raise ValueError("Expected 'time' dim for plotting.")
    return pd.DatetimeIndex(da["time"].to_index())

def _space_dims(da: xr.DataArray) -> list:
    return [d for d in da.dims if d not in ("time", "siglay")]

def _align_art1_to_da(ds: xr.Dataset, da: xr.DataArray, verbose: bool=False) -> Optional[xr.DataArray]:
    """Return area weights aligned to the *current* data selection, or None if unsuitable."""
    if "art1" not in ds:
        return None
    w = ds["art1"]
    sd = _space_dims(da)
    # require weights to cover the spatial dims present in da
    if not all(d in w.dims for d in sd):
        if verbose:
            print(f"[3panel] 'art1' dims {list(w.dims)} do not cover space dims {sd}; unweighted.")
        return None
    # Let weighted_mean_std() handle detailed alignment
    return w
    
def _require_vertical(da: xr.DataArray, var: str, where: str = "", verbose: bool = True) -> bool:
    """Return False (and print a clear message) if the array has no 'siglay' depth dim."""
    if "siglay" not in da.dims:
        if verbose:
            dims = list(da.dims)
            prefix = f"[{where}] " if where else ""
            print(f"{prefix}Error: Var '{var}' has no depth dimension; dims={dims}. "
                  "Cannot generate depth-differentiated (surface/bottom/profile) plots.")
        return False
    return True

# ----------------- plotting functions -----------------

def domain_mean_timeseries(
    ds: xr.Dataset,
    variables: List[str],
    *,
    depth: Any = "surface",                      # ← default to surface
    months: Optional[List[int]] = None,
    years: Optional[List[int]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    base_dir: str,
    figures_root: str,
    groups: Optional[Dict[str, Any]] = None,
    linewidth: float = 1.5,
    figsize: tuple = (10, 4),
    dpi: int = 150,
    styles: Optional[Dict[str, Dict[str, Any]]] = None, 
    verbose: bool = True,
    combine_by: Optional[str] = None,            # None | "var"
) -> None:
    """
    Plot domain-mean time series for one or more variables, with optional depth selection
    and time filtering. Writes one or multiple PNGs depending on `combine_by`.

    Workflow
    --------
    1) Time filter: `filter_time(...)` using months/years/date bounds.
    2) For each `var`, resolve + depth-handle via `resolve_da_with_depth(...)`.
       - If `var` has 'siglay', depth selection / abs-z is applied.
       - If `var` is 2-D (time×node/nele), it is auto-lifted to a single 'siglay' layer.
    3) Compute spatial mean over domain using `_space_mean(da, ds, ...)`.
    4) Save per-variable figures, or a combined figure if `combine_by="var"`.

    Notes
    -----
    - Variables without a 'time' dimension are skipped.
    - 2-D variables like `aice(time,node)` are supported transparently.
    """
    if combine_by not in (None, "var"):
        raise ValueError("domain_mean_timeseries: combine_by must be None or 'var'.")

    tag = depth_tag(depth)
    label = build_time_window_label(months, years, start_date, end_date)
    prefix = file_prefix(base_dir)
    outdir = out_dir(base_dir, figures_root)

    _vprint(verbose, f"[domain] Start domain mean time series")
    _vprint(verbose, f"[domain] Depth={depth} -> tag='{tag}' | Time window='{label}'")

    #  time filter once up front (works for both 2-D and 3-D vars)
    ds_t = filter_time(ds, months, years, start_date, end_date)

    # collect series (so we can optionally combine into one plot)
    series: List[Tuple[str, pd.DatetimeIndex, np.ndarray]] = []
    for var in variables:
        _vprint(verbose, f"[domain] Variable '{var}': resolving with depth handling…")
        try:
            # ← unified resolver: handles 'surface'/'bottom'/abs-z AND 2-D fields
            da = resolve_da_with_depth(
                ds_t, var, depth=depth, groups=groups, verbose=verbose
            )
        except Exception as e:
            _vprint(verbose, f"[domain] Skipping '{var}': {e}")
            continue

        m = _space_mean(da, ds, verbose=verbose)
        if "time" not in m.dims:
            _vprint(verbose, f"[domain] '{var}' has no 'time' dimension; skipping.")
            continue
        series.append((var, _time_index(m), m.values))

    if not series:
        _vprint(verbose, "[domain] nothing to plot.")
        return

    # render
    if combine_by == "var":
        fig, ax = plt.subplots(figsize=figsize)
        for (var, t, y) in series:
            color = style_get(var, styles, "line_color", None)
            ax.plot(t, y, lw=linewidth, color=color, label=var)
        ax.set_title(f"Domain — ({tag}, {label})")
        ax.set_xlabel("Time"); ax.set_ylabel("Value")
        ax.legend(loc="best")
        fname = f"{prefix}__Domain__multi__{tag}__{label}__Timeseries__CombinedByVar.png"
        fig.savefig(os.path.join(outdir, fname), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        _vprint(verbose, f"[domain] Saved: {os.path.join(outdir, fname)}")
        return

    # default: one per variable
    for (var, t, y) in series:
        fig, ax = plt.subplots(figsize=figsize)
        color = style_get(var, styles, "line_color", None)
        ax.plot(t, y, lw=linewidth, color=color)
        ax.set_title(f"{var} — Domain ({tag}, {label})")
        ax.set_xlabel("Time"); ax.set_ylabel(var)
        fname = f"{prefix}__Domain__{var}__{tag}__{label}__Timeseries.png"
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        _vprint(verbose, f"[domain] Saved: {path}")


def station_timeseries(
    ds: xr.Dataset,
    variables: List[str],
    stations: List[Tuple[str, float, float]],   # (name, lat, lon)
    *,
    depth: Any = "surface",
    months: Optional[List[int]] = None,
    years: Optional[List[int]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    base_dir: str,
    figures_root: str,
    groups: Optional[Dict[str, Any]] = None,
    linewidth: float = 1.5,
    figsize: tuple = (10, 4),
    dpi: int = 150,
    styles: Optional[Dict[str, Dict[str, Any]]] = None, 
    verbose: bool = True,
    combine_by: Optional[str] = None,           # NEW: None | "var" | "station"
) -> None:
    """
    Plot station time series by sampling the nearest grid column at each station
    (node or element), with optional depth selection and time filtering.

    Workflow
    --------
    1. Depth selection and time filter on `ds` to obtain `ds2`.
    2. For each station, find nearest indices for 'node' and/or 'nele' via `nearest_index_for_dim`.
    3. For each variable, resolve through `eval_group_or_var(ds2, var, groups)`.
       Subset to the station location (node or element). If absolute depth was requested,
       apply `select_da_by_z` using the spatially-aligned subset of `ds2`.
    4. Plot lines according to `combine_by` choice and save PNGs.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset.
    variables : list of str
        Variable or expression names to plot.
    stations : list of (str, float, float)
        Station definitions as (name, lat, lon). Nearest node/element is used.
    depth : Any
        Depth selector: "surface", "bottom", "depth_avg", sigma value, or absolute z forms.
    months, years : list[int], optional
        Time filters (calendar-based).
    start_date, end_date : str, optional
        Inclusive date bounds as "YYYY-MM-DD".
    base_dir : str
        Run root for filename prefixing.
    figures_root : str
        Root folder for outputs.
    groups : dict, optional
        Alias/expression map used by `eval_group_or_var`.
    linewidth : float, default 1.5
        Line width for plotting.
    figsize : tuple, default (10, 4)
        Figure size in inches.
    dpi : int, default 150
        PNG resolution.
    styles : dict, optional
        Optional style dict. Keys can be variable names (for `combine_by="var"`)
        or station names (for `combine_by="station"`). Supports `"line_color"`.
    verbose : bool, default True
        Print progress and skipping reasons.
    combine_by : {None, "var", "station"}, optional
        Controls grouping of lines and files:
        - None: one PNG per (station × variable).
        - "var": one PNG per station; lines = variables.
        - "station": one PNG per variable; lines = stations.

    Returns
    -------
    None
        Figures written to disk. Filenames encode station, depth, and time window.

    Notes
    -----
    - If the DA lacks 'time', that series is skipped.
    - Sparse grids may yield None for node/element indices; function uses whatever is available.
    - Absolute-depth slicing occurs after spatial selection so depths align with the chosen column.
    """
    if combine_by not in (None, "var", "station"):
        raise ValueError("station_timeseries: combine_by must be None, 'var', or 'station'.")

    tag = depth_tag(depth)
    label = build_time_window_label(months, years, start_date, end_date)
    prefix = file_prefix(base_dir)
    outdir = out_dir(base_dir, figures_root)

    _vprint(verbose, f"[station] Start station time series for {len(stations)} station(s)")
    _vprint(verbose, f"[station] Depth={depth} -> tag='{tag}' | Time window='{label}'")
    if not stations:
        _vprint(verbose, "[station] No stations provided; nothing to do.")
        return

    ds_t = filter_time(ds, months, years, start_date, end_date)

    # map station -> nearest indices
    idx_map: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
    for (name, lat, lon) in stations:
        try:
            nidx = nearest_index_for_dim(ds_t, lat, lon, "node")
        except Exception:
            nidx = None
        try:
            eidx = nearest_index_for_dim(ds_t, lat, lon, "nele")
        except Exception:
            eidx = None
        idx_map[name] = (nidx, eidx)

    def one_series(var: str, st_name: str) -> Optional[Tuple[pd.DatetimeIndex, np.ndarray]]:
        node_idx, nele_idx = idx_map[st_name]
    
        # Build a dataset aligned to the station column first (node/element)
        ds_for_z = ds_t
        try:
            # If the grid has nodes, prefer node column; else fall back to element column
            if "node" in ds_t.dims and node_idx is not None:
                ds_for_z = ds_for_z.isel(node=node_idx)
            elif "nele" in ds_t.dims and nele_idx is not None:
                ds_for_z = ds_for_z.isel(nele=nele_idx)
        except Exception:
            pass  # keep ds_for_z as ds_t if selection not possible
    
        try:
            if is_absolute_z(depth):
                # ABSOLUTE-Z: eval first (3D if available), lift if 2D, then slice by z
                da = eval_group_or_var(ds_for_z, var, groups)
                if "siglay" not in da.dims:
                    sig = xr.DataArray([-0.5], dims=["siglay"], name="siglay")
                    da = da.expand_dims(siglay=sig)
                    da["siglay"] = sig
                # derive numeric target_z
                if isinstance(depth, (float, np.floating, int)):
                    target_z = float(depth)
                elif isinstance(depth, tuple):
                    target_z = float(depth[1])
                else:  # dict {"z_m": ...}
                    target_z = float(depth["z_m"])
                da = select_da_by_z(da, ds_for_z, target_z, verbose=verbose)
            else:
                # SURFACE / BOTTOM / DEPTH-AVG / SIGMA-INDEX:
                # do depth selection on the *dataset*, THEN evaluate the expression
                ds_depth = select_depth(ds_for_z, depth, verbose=verbose)
                da = eval_group_or_var(ds_depth, var, groups)
    
            # Ensure we have a time dimension
            if "time" not in da.dims:
                _vprint(verbose, f"[station:{st_name}] '{var}' has no 'time'; skip.")
                return None
    
            # If this DA still has a spatial dim (node/nele) after column selection, squeeze it out
            for d in ("node", "nele"):
                if d in da.dims and da.sizes.get(d, 1) == 1:
                    da = da.isel({d: 0})
    
            return _time_index(da), da.values
    
        except Exception as e:
            _vprint(verbose, f"[station:{st_name}] Skip '{var}': {e}")
            return None



    # combine_by='station' → per variable, lines = stations
    if combine_by == "station":
        for var in variables:
            plotted = []
            fig, ax = plt.subplots(figsize=figsize)
            for (name, _lat, _lon) in stations:
                s = one_series(var, name)
                if s is None:
                    continue
                t, y = s
                color = style_get(name, styles, "line_color", None)
                ax.plot(t, y, lw=linewidth, label=name, color=color)
                plotted.append(name)
            if not plotted:
                plt.close(fig); continue
            ax.set_title(f"{var} — Stations ({tag}, {label})")
            ax.set_xlabel("Time"); ax.set_ylabel(var)
            ax.legend(loc="best")
            fname = f"{prefix}__Station-All__{var}__{tag}__{label}__Timeseries__CombinedByStation.png"
            fig.savefig(os.path.join(outdir, fname), dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            _vprint(verbose, f"[station] Saved: {os.path.join(outdir, fname)}")
        return

    # combine_by='var' → per station, lines = variables
    if combine_by == "var":
        for (name, _lat, _lon) in stations:
            fig, ax = plt.subplots(figsize=figsize)
            plotted = False
            for var in variables:
                s = one_series(var, name)
                if s is None:
                    continue
                t, y = s
                color = style_get(var, styles, "line_color", None)
                ax.plot(t, y, lw=linewidth, label=var, color=color)
                plotted = True
            if not plotted:
                plt.close(fig); continue
            ax.set_title(f"Station {name} — ({tag}, {label})")
            ax.set_xlabel("Time"); ax.set_ylabel("Value")
            ax.legend(loc="best")
            fname = f"{prefix}__Station-{name}__multi__{tag}__{label}__Timeseries__CombinedByVar.png"
            fig.savefig(os.path.join(outdir, fname), dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            _vprint(verbose, f"[station] Saved: {os.path.join(outdir, fname)}")
        return

    # default: one per (station × variable)
    for (name, _lat, _lon) in stations:
        for var in variables:
            s = one_series(var, name)
            if s is None:
                continue
            t, y = s
            fig, ax = plt.subplots(figsize=figsize)
            color = style_get(var, styles, "line_color", None)
            ax.plot(t, y, lw=linewidth, color=color)
            ax.set_title(f"{var} — Station {name} ({tag}, {label})")
            ax.set_xlabel("Time"); ax.set_ylabel(var)
            fname = f"{prefix}__Station-{name}__{var}__{tag}__{label}__Timeseries.png"
            path = os.path.join(outdir, fname)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            _vprint(verbose, f"[station] Saved: {path}")

def region_timeseries(
    ds: xr.Dataset,
    variables: List[str],
    regions: List[Tuple[str, Dict[str, Any]]],   # (region_name, spec)
    *,
    depth: Any = "surface",
    months: Optional[List[int]] = None,
    years: Optional[List[int]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    base_dir: str,
    figures_root: str,
    groups: Optional[Dict[str, Any]] = None,
    linewidth: float = 1.5,
    figsize: tuple = (10, 4),
    dpi: int = 150,
    styles: Optional[Dict[str, Dict[str, Any]]] = None,
    verbose: bool = True,
    combine_by: Optional[str] = None,            # None | "var" | "region"
) -> None:
    """
    Region-mean time series using polygon masks (shapefile or CSV boundary).
    Supports both 3-D (time×siglay×space) and 2-D (time×space) variables.
    For 2-D vars, auto-lifts to a single 'siglay' layer so depth logic is uniform.
    Vertical selection (surface/bottom/depth_avg/sigma/absolute-z) is applied
    AFTER masking the region (important for absolute-z).
    """
    if combine_by not in (None, "var", "region"):
        raise ValueError("region_timeseries: combine_by must be None, 'var', or 'region'.")

    tag   = depth_tag(depth)
    label = build_time_window_label(months, years, start_date, end_date)
    prefix = file_prefix(base_dir)
    outdir = out_dir(base_dir, figures_root)

    if "lon" not in ds or "lat" not in ds:
        raise ValueError("Dataset must contain 'lon' and 'lat' for region masking.")

    _vprint(verbose, f"[region] Start region time series for {len(regions)} region(s)")
    _vprint(verbose, f"[region] Depth={depth} -> tag='{tag}' | Time window='{label}'")

    # Time filter once; depth is handled per-var AFTER masking.
    ds_t = filter_time(ds, months, years, start_date, end_date)

    # ---------- helper: one (region × variable) series ----------
    def region_series(region_name: str, spec: Dict[str, Any], var: str
                      ) -> Optional[Tuple[pd.DatetimeIndex, np.ndarray]]:
        """
        Build a single time series for one (region × variable), applying:
          1) polygon masking (nodes/elements),
          2) vertical selection (surface/bottom/depth_avg/sigma or absolute-z),
          3) spatial mean over the masked area (area-weighted if possible).

        2-D inputs (time×node/nele) are auto-lifted to a single 'siglay' layer
        so vertical logic is uniform.

        Returns
        -------
        (pd.DatetimeIndex, np.ndarray) or None:
            Time index and region-mean values; or None if the mask is empty,
            variable missing, or the result has no 'time' dimension.
        """
        # --- Build node mask (and element mask if topology present) ---
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
                )
                mask_nodes = polygon_mask(ds, poly)
            else:
                raise ValueError("Region spec must have 'shapefile' or 'csv_boundary'.")
        except Exception as e:
            _vprint(verbose, f"[region:{region_name}] Failed to build mask: {e}")
            return None

        if not np.any(mask_nodes):
            _vprint(verbose, f"[region:{region_name}] Empty mask; skip.")
            return None

        mask_elems = element_mask_from_node_mask(ds, mask_nodes)

        # --- Resolve variable/expression without depth first ---
        try:
            da = eval_group_or_var(ds_t, var, groups)
        except Exception as e:
            _vprint(verbose, f"[region:{region_name}] '{var}' missing: {e}")
            return None

        # --- Lift 2-D fields to a single 'siglay' layer ---
        if "siglay" not in da.dims:
            _vprint(verbose, f"[region:{region_name}] '{var}' has no 'siglay' — lifting to single layer.")
            sig = xr.DataArray([-0.5], dims=["siglay"], name="siglay")
            da = da.expand_dims(siglay=sig)
            da["siglay"] = sig

        # --- Apply region mask and build column-aligned dataset subset ---
        ds_sub = ds_t
        if "node" in da.dims:
            idx_nodes = np.where(mask_nodes)[0]
            da     = da.isel(node=idx_nodes)
            ds_sub = ds_sub.isel(node=idx_nodes)
        elif "nele" in da.dims and mask_elems is not None:
            idx_elems = np.where(mask_elems)[0]
            da     = da.isel(nele=idx_elems)
            ds_sub = ds_sub.isel(nele=idx_elems)

        # --- Vertical selection on the masked subset ---
        if "siglay" in da.dims:
            # ensure the DA has a usable name (expressions often produce name=None)
            if da.name is None:
                da = da.rename(var)

            if is_absolute_z(depth):
                # numeric target z (float)
                if isinstance(depth, (float, np.floating, int)):
                    target_z = float(depth)
                elif isinstance(depth, tuple):
                    target_z = float(depth[1])
                else:  # dict {"z_m": ...}
                    target_z = float(depth["z_m"])
                da = select_da_by_z(da, ds_sub, target_z, verbose=verbose)
            else:
                # surface / bottom / depth_avg / sigma index
                # operate on a temporary dataset built from the DA to avoid name=None bugs
                ds_tmp   = da.to_dataset(name=da.name)
                ds_depth = select_depth(ds_tmp, depth, verbose=verbose)
                da       = ds_depth[da.name]

        # --- Must have time dimension before reducing ---
        if "time" not in da.dims:
            _vprint(verbose, f"[region:{region_name}] '{var}' has no 'time'; skip.")
            return None

        # --- Spatial mean (area-weighted if possible) ---
        m = _space_mean(da, ds, verbose=verbose)   # <-- THIS was missing; caused NameError
        return _time_index(m), m.values

    # ---------- combine_by='var' → one plot per region, lines = variables ----------
    if combine_by == "var":
        for (region_name, spec) in regions:
            plotted = []
            fig, ax = plt.subplots(figsize=figsize)
            for var in variables:
                s = region_series(region_name, spec, var)
                if s is None:
                    continue
                t, y = s
                color = style_get(var, styles, "line_color", None)
                ax.plot(t, y, lw=linewidth, label=var, color=color)
                plotted.append(var)
            if not plotted:
                plt.close(fig)
                continue
            ax.set_title(f"Region {region_name} — ({tag}, {label})")
            ax.set_xlabel("Time"); ax.set_ylabel("Value")
            ax.legend(loc="best")
            fname = f"{prefix}__Region-{region_name}__multi__{tag}__{label}__Timeseries__CombinedByVar.png"
            fig.savefig(os.path.join(outdir, fname), dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            _vprint(verbose, f"[region:{region_name}] Saved: {os.path.join(outdir, fname)}")
        return

    # ---------- combine_by='region' → one plot per variable, lines = regions ----------
    if combine_by == "region":
        for var in variables:
            series = []
            for (region_name, spec) in regions:
                s = region_series(region_name, spec, var)
                if s is not None:
                    t, y = s
                    series.append((region_name, t, y))
            if not series:
                continue

            fig, ax = plt.subplots(figsize=figsize)
            for (rname, t, y) in series:
                color = style_get(rname, styles, "line_color", None)
                ax.plot(t, y, lw=linewidth, label=rname, color=color)
            ax.set_title(f"{var} — Regions ({tag}, {label})")
            ax.set_xlabel("Time"); ax.set_ylabel(var)
            ax.legend(loc="best")
            fname = f"{prefix}__Region-All__{var}__{tag}__{label}__Timeseries__CombinedByRegion.png"
            fig.savefig(os.path.join(outdir, fname), dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            _vprint(verbose, f"[region] Saved: {os.path.join(outdir, fname)}")
        return

    # ---------- default: one PNG per (region × variable) ----------
    for (region_name, spec) in regions:
        for var in variables:
            s = region_series(region_name, spec, var)
            if s is None:
                continue
            t, y = s
            fig, ax = plt.subplots(figsize=figsize)
            color = style_get(var, styles, "line_color", None)
            ax.plot(t, y, lw=linewidth, color=color)
            ax.set_title(f"{var} — Region {region_name} ({tag}, {label})")
            ax.set_xlabel("Time"); ax.set_ylabel(var)
            fname = f"{prefix}__Region-{region_name}__{var}__{tag}__{label}__Timeseries.png"
            path = os.path.join(outdir, fname)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            _vprint(verbose, f"[region:{region_name}] Saved: {path}")

def _plot_three_panel(
    *,
    t: np.ndarray,
    surf_mean: np.ndarray, surf_std: np.ndarray,
    bott_mean: np.ndarray, bott_std: np.ndarray,
    zcoord: np.ndarray, prof_mean: np.ndarray, prof_std: np.ndarray,
    title_prefix: str, var: str, label: str,
    outdir: str, prefix: str,
    styles: Optional[Dict[str, Dict[str, Any]]] = None,
    dpi: int = 150, figsize: tuple = (11, 9),
) -> None:
    """
    Internal helper to render a 3-panel figure for a single variable:

      Panel 1: Surface time series (mean ± 1σ)
      Panel 2: Bottom  time series (mean ± 1σ)
      Panel 3: Depth profile vs 'siglay' (mean ± 1σ across time & space)

    Parameters
    ----------
    t : np.ndarray
        1D array of time values suitable for matplotlib plotting (e.g., datetime64 converted).
    surf_mean, bott_mean : np.ndarray
        1D arrays of mean values per time step for surface and bottom.
    surf_std, bott_std : np.ndarray
        1D arrays of standard deviation (±1σ) per time step for surface and bottom.
    zcoord : np.ndarray
        Vertical coordinate (e.g., 'siglay') for the profile panel; increasing or decreasing.
    prof_mean, prof_std : np.ndarray
        1D arrays of mean and standard deviation along `zcoord` (aggregated across time & space).
    title_prefix : str
        Title prefix describing scope (e.g., "Domain", "Station X", "Region Y").
    var : str
        Variable name used for labeling, styling, and filename.
    label : str
        Time window label (e.g., "JJA 2018", "2018-04–2018-10").
    outdir : str
        Directory to save the PNG file.
    prefix : str
        Filename prefix derived from `file_prefix(base_dir)`.
    styles : dict, optional
        Per-variable style dict. Recognized keys:
          - "line_color": color for lines
          - "line_width": float
          - "shade_alpha": float transparency for ±σ fill
          - "shade_color": explicit shade color (otherwise a lightened line color is used)
          - "shade_lighten": float in (0..1) controlling how much to lighten the line color
    dpi : int, default 150
        PNG resolution.
    figsize : tuple, default (11, 9)
        Figure size in inches.

    Returns
    -------
    None
        Saves a PNG named:
        ``<prefix>__<title_prefix>__<var>__3Panel__<label>__Timeseries.png``

    Notes
    -----
    - If `zcoord` appears decreasing (typical for sigma indices), the y-axis is inverted.
    - Colors: if no explicit color provided, uses MPL cycle and derives a lighter shade for the σ band.
    """

    def _lighter(c, amount: float = 0.6) -> str:
        """
        Return a lighter version of color c by blending toward white.
        amount in (0..1): 0 = original, 1 = white.
        """
        r, g, b = to_rgb(c)
        r = r + (1.0 - r) * amount
        g = g + (1.0 - g) * amount
        b = b + (1.0 - b) * amount
        return to_hex((r, g, b))

    # Resolve style for this var (fallbacks are sensible)
    line_color  = style_get(var, styles, "line_color", None)   # None -> MPL default cycle
    line_width  = style_get(var, styles, "line_width", 1.6)
    shade_alpha = style_get(var, styles, "shade_alpha", 0.25)
    # optional explicit shade color; if not provided we'll compute a lighter one from the actual line color
    shade_color_pref = style_get(var, styles, "shade_color", None)
    shade_lighten    = style_get(var, styles, "shade_lighten", 0.6)  # how much to lighten if auto (0..1)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize, constrained_layout=True)

    # --- Panel 1: surface ---
    ax = axes[0]
    # draw line first to discover the *actual* color if user didn't set one
    line_surf, = ax.plot(t, surf_mean, lw=line_width, color=line_color, label="mean", zorder=2)
    actual_color = line_surf.get_color()
    shade_color = shade_color_pref or _lighter(actual_color, amount=shade_lighten)
    ax.fill_between(t, surf_mean - surf_std, surf_mean + surf_std,
                    alpha=shade_alpha, color=shade_color, label="±1σ", zorder=1)
    ax.set_title(f"{title_prefix} — {var} — Surface (±1σ)")
    ax.set_xlabel("Time"); ax.set_ylabel(var)
    ax.legend(loc="best")

    # --- Panel 2: bottom ---
    ax = axes[1]
    line_bott, = ax.plot(t, bott_mean, lw=line_width, color=actual_color, label="mean", zorder=2)
    ax.fill_between(t, bott_mean - bott_std, bott_mean + bott_std,
                    alpha=shade_alpha, color=shade_color, label="±1σ", zorder=1)
    ax.set_title(f"{title_prefix} — {var} — Bottom (±1σ)")
    ax.set_xlabel("Time"); ax.set_ylabel(var)
    ax.legend(loc="best")

    # --- Panel 3: profile vs siglay ---
    ax = axes[2]
    # plot line first to reuse same color, then shade
    line_prof, = ax.plot(prof_mean, zcoord, lw=line_width, color=actual_color, label="mean", zorder=2)
    ax.fill_betweenx(zcoord, prof_mean - prof_std, prof_mean + prof_std,
                     alpha=shade_alpha, color=shade_color, label="±1σ", zorder=1)
    ax.set_title(f"{title_prefix} — {var} — Profile vs siglay (mean ±1σ)")
    ax.set_xlabel(var); ax.set_ylabel("siglay")
    try:
        if np.nanmax(zcoord) > np.nanmin(zcoord) and zcoord[0] >= zcoord[-1]:
            ax.invert_yaxis()
    except Exception:
        pass
    ax.legend(loc="best")

    fname = f"{prefix}__{title_prefix.replace(' ', '-') }__{var}__3Panel__{label}__Timeseries.png"
    fig.savefig(os.path.join(outdir, fname), dpi=dpi, bbox_inches="tight")
    plt.close(fig)



def domain_three_panel(
    ds: xr.Dataset,
    variables: list[str],
    *,
    months=None, years=None, start_date=None, end_date=None,
    base_dir: str, figures_root: str,
    groups: Optional[Dict[str, Any]] = None,
    styles: Optional[Dict[str, Dict[str, Any]]] = None,  
    dpi: int = 150, figsize: tuple = (11, 9), verbose: bool = False,
) -> None:
    """
    Render 3-panel summaries for each variable at the domain scale:

      (1) Surface time series (spatial mean ±1σ)
      (2) Bottom  time series (spatial mean ±1σ)
      (3) Depth profile vs 'siglay' (mean ±1σ across time & space)

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset.
    variables : list of str
        Variable or expression names to plot.
    months, years : list[int], optional
        Time filters.
    start_date, end_date : str, optional
        Inclusive date bounds "YYYY-MM-DD".
    base_dir : str
        Run root for filename prefixing.
    figures_root : str
        Output root directory.
    groups : dict, optional
        Alias/expression map for `eval_group_or_var`.
    styles : dict, optional
        Per-variable style overrides (see `_plot_three_panel` for keys).
    dpi : int, default 150
        PNG resolution.
    figsize : tuple, default (11, 9)
        Figure size for the 3-panel figure.
    verbose : bool, default False
        Print progress messages.

    Returns
    -------
    None
        One PNG per variable saved to disk. Filenames encode scope (Domain) and time label.

    Notes
    -----
    - Surface/bottom time series are computed after `select_depth(ds_t, "surface"/"bottom")`.
    - Spatial mean/std per time step use area weights (`art1`) if available and alignable.
    - If a variable lacks 'siglay', a single-layer profile is plotted using surface stats.
    """
    
    ds_t = filter_time(ds, months, years, start_date, end_date)
    label = build_time_window_label(months, years, start_date, end_date)
    outdir = out_dir(base_dir, figures_root)
    prefix = file_prefix(base_dir)

    if verbose:
        print(f"[3panel/domain] Time label={label}")

    for var in variables:
        try:
            # check the raw (unsliced) field for a vertical dimension first
            da_raw = eval_group_or_var(ds_t, var, groups)
        except Exception as e:
            print(f"[3panel/domain] Skip '{var}': {e}")
            continue
    
        if not _require_vertical(da_raw, var, where="3panel/domain", verbose=verbose):
            continue

        try:
            # Surface & bottom time series: spatial dims only reduced per timestep
            da_surf = eval_group_or_var(select_depth(ds_t, "surface"), var, groups)
            da_bott = eval_group_or_var(select_depth(ds_t, "bottom"),  var, groups)
        except Exception as e:
            print(f"[3panel/domain] Skip '{var}': {e}")
            continue

        if "time" not in da_surf.dims or "time" not in da_bott.dims:
            print(f"[3panel/domain] '{var}' has no time dim; skipping.")
            continue

        t = pd.to_datetime(da_surf["time"].values)
        w_s = _align_art1_to_da(ds, da_surf, verbose)
        w_b = _align_art1_to_da(ds, da_bott, verbose)

        surf_mean, surf_std = weighted_mean_std(da_surf, _space_dims(da_surf), w_s)
        bott_mean, bott_std = weighted_mean_std(da_bott, _space_dims(da_bott), w_b)

        # Profile: mean ±1σ over (time + space) at each siglay
        try:
            da_prof = eval_group_or_var(ds_t, var, groups)
        except Exception as e:
            print(f"[3panel/domain] Profile skip '{var}': {e}")
            continue

        if "siglay" in da_prof.dims:
            w_p = _align_art1_to_da(ds, da_prof, verbose)
            prof_mean, prof_std = weighted_mean_std(da_prof, ["time"] + _space_dims(da_prof), w_p)
            zcoord = da_prof["siglay"].values
        else:
            # No vertical dimension; emulate single-layer profile
            if verbose:
                print(f"[3panel/domain] '{var}' has no 'siglay'; using surface-only profile.")
            zcoord = np.array([0.0])
            m, s = weighted_mean_std(da_surf, _space_dims(da_surf), w_s)
            prof_mean, prof_std = m.mean("time"), s.mean("time")

        _plot_three_panel(
            t=t,
            surf_mean=surf_mean.values, surf_std=surf_std.values,
            bott_mean=bott_mean.values, bott_std=bott_std.values,
            zcoord=zcoord, prof_mean=np.asarray(prof_mean), prof_std=np.asarray(prof_std),
            title_prefix="Domain",
            var=var, label=label, outdir=outdir, prefix=prefix, styles=styles,
            dpi=dpi, figsize=figsize,
        )

def station_three_panel(
    ds: xr.Dataset,
    variables: list[str],
    stations: List[Tuple[str, float, float]],
    *,
    months=None, years=None, start_date=None, end_date=None,
    base_dir: str, figures_root: str, groups: Optional[Dict[str, Any]] = None, styles: Optional[Dict[str, Dict[str, Any]]] = None,
    dpi: int = 150, figsize: tuple = (11, 9), verbose: bool=False,
) -> None:
    """
    Render 3-panel summaries for each (station × variable):

      (1) Surface time series (temporal mean ±1σ at the station column)
      (2) Bottom  time series (temporal mean ±1σ at the station column)
      (3) Depth profile vs 'siglay' (temporal mean ±1σ at the station column)

    At a single station (nearest node/element), spatial variance collapses, so
    the ±σ shading reflects **temporal** variability rather than spatial.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset.
    variables : list of str
        Variable or expression names to plot.
    stations : list of (str, float, float)
        (name, lat, lon); nearest node/element is used for sampling.
    months, years : list[int], optional
        Time filters.
    start_date, end_date : str, optional
        Date bounds "YYYY-MM-DD".
    base_dir : str
        Run root for filename prefixing.
    figures_root : str
        Output root directory.
    groups : dict, optional
        Alias/expression map for `eval_group_or_var`.
    styles : dict, optional
        Per-variable style overrides (see `_plot_three_panel`).
    dpi : int, default 150
        PNG resolution.
    figsize : tuple, default (11, 9)
        Figure size for the 3-panel figure.
    verbose : bool, default False
        Print progress messages.

    Returns
    -------
    None
        One PNG per (station × variable) saved to disk.

    Notes
    -----
    - Surface/bottom arrays are subset to the nearest node/element before plotting.
    - If the profile has 'siglay', profile stats are temporal means/std across time.
      Otherwise a single-layer profile is synthesized from the surface series.
    """
    if not stations:
        return

    ds_t = filter_time(ds, months, years, start_date, end_date)
    label = build_time_window_label(months, years, start_date, end_date)
    prefix = file_prefix(base_dir)
    outdir = out_dir(base_dir, figures_root)

    for (name, lat, lon) in stations:
        for var in variables:
            try:
                # check raw field first for a vertical dim
                da_raw = eval_group_or_var(ds_t, var, groups)
            except Exception as e:
                print(f"[3panel/station {name}] Skip '{var}': {e}")
                continue
    
            if not _require_vertical(da_raw, var, where=f"3panel/station {name}", verbose=verbose):
                continue
            try:
                da_surf = eval_group_or_var(select_depth(ds_t, "surface"), var, groups)
                da_bott = eval_group_or_var(select_depth(ds_t, "bottom"),  var, groups)
                da_prof = eval_group_or_var(ds_t, var, groups)
            except Exception as e:
                print(f"[3panel/station {name}] Skip '{var}': {e}")
                continue

            # select nearest index for the actual grid dimension each array has
            try:
                node_idx = nearest_index_for_dim(ds_t, lat, lon, "node")
            except Exception:
                node_idx = None
            try:
                nele_idx = nearest_index_for_dim(ds_t, lat, lon, "nele")
            except Exception:
                nele_idx = None

            if "node" in da_surf.dims and node_idx is not None: da_surf = da_surf.isel(node=node_idx)
            if "nele" in da_surf.dims and nele_idx is not None: da_surf = da_surf.isel(nele=nele_idx)
            if "node" in da_bott.dims and node_idx is not None: da_bott = da_bott.isel(node=node_idx)
            if "nele" in da_bott.dims and nele_idx is not None: da_bott = da_bott.isel(nele=nele_idx)
            if "node" in da_prof.dims and node_idx is not None: da_prof = da_prof.isel(node=node_idx)
            if "nele" in da_prof.dims and nele_idx is not None: da_prof = da_prof.isel(nele=nele_idx)

            if "time" not in da_surf.dims:
                print(f"[3panel/station {name}] '{var}' has no time dim; skipping.")
                continue

            t = pd.to_datetime(da_surf["time"].values)

            # Spatial σ doesn't exist at a single node — provide temporal σ for shading
            surf_mean, surf_std = da_surf, xr.zeros_like(da_surf)
            bott_mean, bott_std = da_bott, xr.zeros_like(da_bott)

            if "siglay" in da_prof.dims:
                prof_mean = da_prof.mean("time", skipna=True)
                prof_std  = da_prof.std("time",  skipna=True)
                zcoord = da_prof["siglay"].values
            else:
                zcoord = np.array([0.0]); prof_mean = da_surf.mean("time"); prof_std = xr.zeros_like(prof_mean)

            _plot_three_panel(
                t=t,
                surf_mean=surf_mean.values, surf_std=surf_std.values,
                bott_mean=bott_mean.values, bott_std=bott_std.values,
                zcoord=zcoord, prof_mean=np.asarray(prof_mean), prof_std=np.asarray(prof_std),
                title_prefix=f"Station {name}",
                var=var, label=label, outdir=outdir, prefix=prefix, styles=styles,
                dpi=dpi, figsize=figsize,
            )

def region_three_panel(
    ds: xr.Dataset,
    variables: List[str],
    regions: List[Tuple[str, Dict[str, Any]]],
    *,
    months=None, years=None, start_date=None, end_date=None,
    base_dir: str, figures_root: str, groups: Optional[Dict[str, Any]] = None, styles: Optional[Dict[str, Dict[str, Any]]] = None,
    dpi: int = 150, figsize: tuple = (11, 9), verbose: bool=False,
) -> None:
    """
    Render 3-panel summaries for each (region × variable):

      (1) Surface time series (spatial mean ±1σ within region)
      (2) Bottom  time series (spatial mean ±1σ within region)
      (3) Depth profile vs 'siglay' (mean ±1σ across time & space within region)

    Regions are defined by polygon masks from a shapefile or CSV boundary.
    If `art1` is present and alignable, spatial means/std are area-weighted.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset. Must contain 'lon' and 'lat' for region masking.
    variables : list of str
        Variable or expression names to plot.
    regions : list of (str, dict)
        Regions as (region_name, spec). `spec` must include either:
          - {"shapefile": <path>, "name_field": ..., "name_equals": ...}
          - {"csv_boundary": <path>, "lon_col": "lon", "lat_col": "lat"}
    months, years : list[int], optional
        Time filters.
    start_date, end_date : str, optional
        Date bounds "YYYY-MM-DD".
    base_dir : str
        Run root for filename prefixing.
    figures_root : str
        Output root directory.
    groups : dict, optional
        Alias/expression map for `eval_group_or_var`.
    styles : dict, optional
        Per-variable style overrides (see `_plot_three_panel`).
    dpi : int, default 150
        PNG resolution.
    figsize : tuple, default (11, 9)
        Figure size for the 3-panel figure.
    verbose : bool, default False
        Print progress and mask summaries.

    Returns
    -------
    None
        One PNG per (region × variable) saved to disk.

    Notes
    -----
    - Builds node mask from polygon; element mask optionally derived from connectivity.
    - Weights (`art1`) are subset to the same nodes/elements before computing means/std.
    - If the variable lacks 'siglay', a single-layer profile is plotted using temporal
      surface stats over the region.
    """
    
    # Helper: which dims are "space" for a given DataArray
    def space_dims(da: xr.DataArray) -> list[str]:
        return [d for d in da.dims if d not in ("time", "siglay")]

    ds_t = filter_time(ds, months, years, start_date, end_date)
    label = build_time_window_label(months, years, start_date, end_date)
    outdir = out_dir(base_dir, figures_root)
    prefix = file_prefix(base_dir)

    if "lon" not in ds or "lat" not in ds:
        raise ValueError("Dataset must contain 'lon' and 'lat' for region masking.")

    if verbose:
        print(f"[3panel/region] Time label={label} | regions={len(regions)}")

    for region_name, spec in regions:
        # --- Build region node mask ---
        try:
            if "shapefile" in spec:
                if verbose:
                    print(f"[3panel/region {region_name}] Using shapefile: {spec['shapefile']}")
                mask_nodes = polygon_mask_from_shapefile(
                    ds, spec["shapefile"],
                    name_field=spec.get("name_field"),
                    name_equals=spec.get("name_equals"),
                )
            elif "csv_boundary" in spec:
                if verbose:
                    print(f"[3panel/region {region_name}] Using CSV boundary: {spec['csv_boundary']}")
                poly = polygon_from_csv_boundary(
                    spec["csv_boundary"],
                    lon_col=spec.get("lon_col", "lon"),
                    lat_col=spec.get("lat_col", "lat"),
                )
                mask_nodes = polygon_mask(ds, poly)
            else:
                raise ValueError("Region spec must have 'shapefile' or 'csv_boundary'.")
        except Exception as e:
            print(f"[3panel/region {region_name}] failed to build mask: {e}")
            continue

        if not np.any(mask_nodes):
            print(f"[3panel/region {region_name}] mask empty; skipping.")
            continue

        if verbose:
            print(f"[3panel/region {region_name}] Node mask size: {np.count_nonzero(mask_nodes)}")

        # Optional element mask (if connectivity is available)
        mask_elems = element_mask_from_node_mask(ds, mask_nodes)
        if verbose and mask_elems is not None:
            print(f"[3panel/region {region_name}] Element mask size: {np.count_nonzero(mask_elems)}")

        for var in variables:
            # --- Resolve variables at required depths + full for profile ---
            try:
                da_raw = eval_group_or_var(ds_t, var, groups)
            except Exception as e:
                print(f"[3panel/region {region_name}] Skip '{var}': {e}")
                continue
        
            if not _require_vertical(da_raw, var, where=f"3panel/region {region_name}", verbose=verbose):
                continue
            try:
                da_surf = eval_group_or_var(select_depth(ds_t, "surface"), var, groups)
                da_bott = eval_group_or_var(select_depth(ds_t, "bottom"),  var, groups)
                da_prof = eval_group_or_var(ds_t, var, groups)
            except Exception as e:
                print(f"[3panel/region {region_name}] Skip '{var}': {e}")
                continue

            # --- Apply region mask to data ---
            if "node" in da_surf.dims:
                idx_nodes = np.where(mask_nodes)[0]
                da_surf = da_surf.isel(node=idx_nodes)
                if "node" in da_bott.dims:
                    da_bott = da_bott.isel(node=idx_nodes)
                if "node" in da_prof.dims:
                    da_prof = da_prof.isel(node=idx_nodes)
                if verbose:
                    print(f"[3panel/region {region_name}] Selected {len(idx_nodes)} node(s).")
            elif "nele" in da_surf.dims and mask_elems is not None:
                idx_elems = np.where(mask_elems)[0]
                da_surf = da_surf.isel(nele=idx_elems)
                if "nele" in da_bott.dims:
                    da_bott = da_bott.isel(nele=idx_elems)
                if "nele" in da_prof.dims:
                    da_prof = da_prof.isel(nele=idx_elems)
                if verbose:
                    print(f"[3panel/region {region_name}] Selected {len(idx_elems)} element(s).")

            if "time" not in da_surf.dims:
                print(f"[3panel/region {region_name}] '{var}' has no time dim; skipping.")
                continue

            t = pd.to_datetime(da_surf["time"].values)

            # --- Subset weights to EXACT same nodes/elements (avoid alignment errors) ---
            w_base = ds["art1"] if "art1" in ds else None
            w_s = w_b = w_p = None
            if w_base is not None:
                sd_s = space_dims(da_surf)
                sd_b = space_dims(da_bott)
                sd_p = space_dims(da_prof)

                if "node" in w_base.dims and "node" in sd_s + sd_b + sd_p and np.any(mask_nodes):
                    idx_nodes = np.where(mask_nodes)[0]
                    if "node" in sd_s:
                        w_s = w_base.isel(node=idx_nodes)
                    if "node" in sd_b:
                        w_b = w_base.isel(node=idx_nodes)
                    if "node" in sd_p:
                        w_p = w_base.isel(node=idx_nodes)

                if mask_elems is not None and "nele" in w_base.dims:
                    idx_elems = np.where(mask_elems)[0]
                    if "nele" in sd_s:
                        w_s = w_base.isel(nele=idx_elems)
                    if "nele" in sd_b:
                        w_b = w_base.isel(nele=idx_elems)
                    if "nele" in sd_p:
                        w_p = w_base.isel(nele=idx_elems)

            # --- Compute spatial mean ±1σ per time (surface & bottom) ---
            surf_mean, surf_std = weighted_mean_std(da_surf, space_dims(da_surf), w_s)
            bott_mean, bott_std = weighted_mean_std(da_bott, space_dims(da_bott), w_b)

            # --- Profile: mean ±1σ across time + space at each siglay ---
            if "siglay" in da_prof.dims:
                prof_mean, prof_std = weighted_mean_std(da_prof, ["time"] + space_dims(da_prof), w_p)
                zcoord = da_prof["siglay"].values
            else:
                # No vertical dim → single-layer “profile”
                zcoord = np.array([0.0])
                prof_mean = da_surf.mean("time", skipna=True)
                prof_std  = xr.zeros_like(prof_mean)

            if verbose:
                print(f"[3panel/region {region_name}] Saving 3-panel for '{var}' ({label})")

            _plot_three_panel(
                t=t,
                surf_mean=surf_mean.values, surf_std=surf_std.values,
                bott_mean=bott_mean.values, bott_std=bott_std.values,
                zcoord=zcoord, prof_mean=np.asarray(prof_mean), prof_std=np.asarray(prof_std),
                title_prefix=f"Region {region_name}",
                var=var, label=label, outdir=outdir, prefix=prefix, styles=styles,
                dpi=dpi, figsize=figsize,
            )
