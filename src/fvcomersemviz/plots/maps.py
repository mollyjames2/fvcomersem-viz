# fvcomersemviz/maps.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any, Sequence
import os
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from ..io import (
    filter_time,
    eval_group_or_var,
)

from ..utils import (
    out_dir, file_prefix,
    robust_clims,
    build_triangulation,
    style_get,
    select_depth,
    select_da_by_z,
    build_time_window_label,
    depth_tag,
    resolve_da_with_depth,
    is_absolute_z,
)
from ..regions import (
    polygon_mask_from_shapefile,
    polygon_from_csv_boundary,
    polygon_mask,
    element_mask_from_node_mask,
)

# -----------------------------
# Internal helpers kept for this module
# -----------------------------

def _timepoints_to_list(at_time: Optional[Any], at_times: Optional[Sequence[Any]]) -> Optional[List[pd.Timestamp]]:
    if at_times is not None:
        return [pd.to_datetime(t) for t in at_times]
    if at_time is not None:
        return [pd.to_datetime(at_time)]
    return None

def _iso_label(ts: pd.Timestamp) -> str:
    # Compact label for filenames
    return ts.strftime("%Y-%m-%dT%H%M")

def _choose_instants(
    da: xr.DataArray,
    desired: List[pd.Timestamp],
    method: str = "nearest",
) -> List[Tuple[pd.Timestamp, xr.DataArray]]:
    """Return list of (chosen_time, instantaneous-DA) using .sel(method='nearest')."""
    if "time" not in da.dims:
        # No time dimension; return as-is with a simple label
        return [(pd.Timestamp("NaT"), da)]
    out = []
    for want in desired:
        _one = da.sel(time=want, method=method)
        # Capture the *actual* time chosen after nearest selection
        chosen = pd.to_datetime(np.atleast_1d(_one["time"].values)[0])
        out.append((pd.Timestamp(chosen), _one))
    return out

def _plot_tripcolor_full(
    tri: Triangulation,
    *,
    node_values: Optional[np.ndarray] = None,   # length Npoints (NaN outside)
    face_values: Optional[np.ndarray] = None,   # length Ntriangles (NaN outside)
    cmap: str,
    clim: Optional[Tuple[float, float]],
    title: str,
    cbar_label: str,
    fname: str,
    dpi: int,
    figsize: Tuple[float, float],
    shading: str = "gouraud",
    norm=None,
    draw_mesh: bool = False,
    verbose: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    if node_values is not None:
        z = np.ma.masked_invalid(node_values)
        # node-centered; Gouraud works with per-vertex values
        tpc = ax.tripcolor(tri, z, shading=shading, cmap=cmap, norm=norm)
    elif face_values is not None:
        z = np.ma.masked_invalid(face_values)
        # element-centered; pass scalars for each triangle (flat shading)
        tpc = ax.tripcolor(tri, z, shading="flat", cmap=cmap, norm=norm)
    else:
        raise ValueError("Either node_values or face_values must be provided.")

    # Only set explicit clim when no norm is in use
    if clim is not None and norm is None:
        tpc.set_clim(*clim)

    if draw_mesh:
        ax.triplot(tri, color="k", lw=0.3, alpha=0.4, zorder=3)

    ax.set_title(title)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    cbar = fig.colorbar(tpc, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(cbar_label)

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"[maps] Saved {fname}")


# ------------------------------------------------------------
# Mapping Functions
# ------------------------------------------------------------

def domain_map(
    ds: xr.Dataset,
    variables: List[str],
    *,
    depth: Any = "surface",
    months: Optional[List[int]] = None,
    years: Optional[List[int]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    at_time: Optional[Any] = None,
    at_times: Optional[Sequence[Any]] = None,
    time_method: str = "nearest",
    base_dir: str,
    figures_root: str,
    groups: Optional[Dict[str, Any]] = None,
    cmap: str = "viridis",
    clim: Optional[Tuple[float, float]] = None,
    robust_q: Tuple[float, float] = (5, 95),
    dpi: int = 150,
    figsize: Tuple[float, float] = (8, 6),
    shading: str = "gouraud",
    grid_on: bool = False,
    verbose: bool = False,
    styles: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """
    Render domain-wide maps for one or more variables at a chosen depth and time window, 
    saving either **instantaneous** snapshots or the **time mean**
    Handles 3-D (time×siglay×space) and 2-D (time×space) fields. 2-D fields are auto-lifted
    to a single 'siglay' so vertical logic (surface/bottom/depth_avg/abs-z) works uniformly.
    Absolute-z is applied per variable after time filtering.
    
    Workflow
    --------
    1. Depth selection: `select_depth(ds, depth)` (supports sigma keywords, depth-avg,
       and absolute depth via z meters, negative downward). defaults to surface - can be omitted for 2D runs
    2. Time filter: `filter_time(...)` using months/years/date bounds.
    3. For each variable:
       a. Resolve via `eval_group_or_var(ds_t, var, groups)`.
       b. If an absolute z depth was requested, refine with `select_da_by_z(...)`.
       c. Determine plotting center: node ('node' in dims) or element ('nele' in dims).
       d. Decide the time(s) to plot:
          - If `at_time`/`at_times` provided ? plot one PNG per chosen instant
            (selection via `_choose_instants(..., method=time_method)`).
          - Else - plot the mean over 'time' (if present).
       e. Build color scaling:
          - If a `norm` is provided via `styles[var]["norm"]`, it takes priority.
          - Else use explicit vmin/vmax from `styles` (or `clim`), or compute robust
            limits from data using `robust_q` percentiles.
       f. Draw with `_plot_tripcolor_full(...)` on a triangulation from `build_triangulation(ds)`.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset with grid variables (lon/lat, optional 'nv') and fields to map.
    variables : list of str
        Variable or expression names to map (resolved via `groups` if provided).
    depth : Any
        Depth selector passed to `select_depth`, e.g. "surface", "bottom", "depth_avg",
        sigma value, or absolute depth forms: -10.0, ("z_m", -10.0), or {"z_m": -10.0}.
        Absolute depth refinement is applied per variable with `select_da_by_z`.
    months, years : list[int], optional
        Calendar filters (1-12 months, year integers). Can be combined.
    start_date, end_date : str, optional
        Inclusive date bounds "YYYY-MM-DD".
    at_time : Any, optional
        A single desired timestamp (e.g., numpy datetime64, pandas Timestamp, or ISO string)
        for instantaneous plotting.
    at_times : sequence, optional
        Multiple desired timestamps. If both `at_time` and `at_times` are None, a mean over
        the time window is plotted instead.
    time_method : {"nearest", "pad", "backfill"}, default "nearest"
        Selection method passed to `_choose_instants` when matching requested instants to
        available model times.
    base_dir : str
        Run root; used to construct filename prefix via `file_prefix(base_dir)`.
    figures_root : str
        Output root directory; final PNGs are written under this path.
    groups : dict, optional
        Global alias/expression mapping used by `eval_group_or_var`.
    cmap : str, default "viridis"
        Default colormap, can be overridden per-variable in `styles`.
    clim : (float, float), optional
        Default color limits (vmin, vmax) if no per-variable vmin/vmax and no `norm` is provided.
    robust_q : (float, float), default (5, 95)
        Percentiles for robust auto-scaling when limits are not explicitly given.
    dpi : int, default 150
        Output PNG resolution.
    figsize : (float, float), default (8, 6)
        Figure size in inches.
    shading : {"flat", "gouraud"}, default "gouraud"
        Triangulation shading for tripcolor; can be overridden per-variable in `styles`.
    grid_on : bool, default False
        If True, overlay the mesh edges (useful for diagnostics).
    verbose : bool, default False
        Print progress and skip reasons.
    styles : dict, optional
        Per-variable style overrides. For key `var`, recognized entries:
          - "cmap": str or Colormap
          - "norm": matplotlib.colors.Normalize (takes precedence over all limits)
          - "vmin": float
          - "vmax": float
          - "shading": {"flat", "gouraud"}

    Returns
    -------
    None
        Saves one PNG per variable/time selection. Filenames encode scope=Domain,
        variable, depth tag, and time label (instant or mean over window).

    Notes
    -----
    - Plot center is inferred from data dims:
        'node' - node-centered plotting, 'nele' - element-centered plotting.
      Arrays lacking both dims are skipped.
    - If `ds` lacks 'nv', triangulation falls back to Delaunay using available coordinates.
    - Color scaling priority: `norm` > (vmin/vmax in styles) > `clim` > robust percentiles.
    """
    
    # Time filter first; per-var depth is handled by resolve_da_with_depth
    ds_t = filter_time(ds, months, years, start_date, end_date)

    label_window = build_time_window_label(months, years, start_date, end_date)
    tag = depth_tag(depth)
    outdir = out_dir(base_dir, figures_root)
    prefix = file_prefix(base_dir)
    tri = build_triangulation(ds)

    desired = _timepoints_to_list(at_time, at_times)

    for var in variables:
        # unified resolver (handles lifting + surface/bottom/depth_avg/abs-z)
        try:
            da = resolve_da_with_depth(ds_t, var, depth=depth, groups=groups, verbose=verbose)
        except Exception as e:
            print(f"[maps/domain] Skip '{var}': {e}")
            continue

        # Per-var style overrides (with fallback to function args)
        cmap_eff   = style_get(var, styles, "cmap", cmap)
        norm_eff   = style_get(var, styles, "norm", None)
        vmin_style = style_get(var, styles, "vmin", None)
        vmax_style = style_get(var, styles, "vmax", None)
        shading_eff= style_get(var, styles, "shading", shading)

        center = "node" if "node" in da.dims else ("nele" if "nele" in da.dims else None)
        if center is None:
            print(f"[maps/domain] '{var}' has no 'node' or 'nele' dim; skipping.")
            continue

        def _plot_values(vals: np.ndarray, title: str, fname: str):
            # Decide limits: norm has priority; else vmin/vmax; else robust
            if norm_eff is not None:
                clim_eff = None
            else:
                if vmin_style is not None and vmax_style is not None:
                    clim_eff = (vmin_style, vmax_style)
                elif clim is not None:
                    clim_eff = clim
                else:
                    vmin_r, vmax_r = robust_clims(vals, q=robust_q)
                    clim_eff = (vmin_r, vmax_r)

            if center == "node":
                _plot_tripcolor_full(
                    tri, node_values=vals, cmap=cmap_eff, clim=clim_eff, norm=norm_eff,
                    title=title, cbar_label=var, fname=fname, dpi=dpi, figsize=figsize,
                    shading=shading_eff, verbose=verbose, draw_mesh=grid_on,
                )
            else:
                _plot_tripcolor_full(
                    tri, face_values=vals, cmap=cmap_eff, clim=clim_eff, norm=norm_eff,
                    title=title, cbar_label=var, fname=fname, dpi=dpi, figsize=figsize,
                    shading=shading_eff, verbose=verbose, draw_mesh=grid_on,
                )

        if desired:
            for chosen, inst in _choose_instants(da, desired, method=time_method):
                lbl = _iso_label(chosen) if pd.notnull(chosen) else "NoTime"
                title = f"Domain - {var} ({tag}, {lbl})"
                fname = os.path.join(outdir, f"{prefix}__Map-Domain__{var}__{tag}__{lbl}__Instant.png")
                vals = np.asarray(inst.values).ravel()
                _plot_values(vals, title, fname)
        else:
            m = da.mean("time", skipna=True) if "time" in da.dims else da
            vals = np.asarray(m.values).ravel()
            title = f"Domain - {var} ({tag}, {label_window})"
            fname = os.path.join(outdir, f"{prefix}__Map-Domain__{var}__{tag}__{label_window}__Mean.png")
            _plot_values(vals, title, fname)


def region_map(
    ds: xr.Dataset,
    variables: List[str],
    regions: List[Tuple[str, Dict[str, Any]]],
    *,
    depth: Any = "surface",
    months: Optional[List[int]] = None,
    years: Optional[List[int]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    at_time: Optional[Any] = None,
    at_times: Optional[Sequence[Any]] = None,
    time_method: str = "nearest",
    base_dir: str,
    figures_root: str,
    groups: Optional[Dict[str, Any]] = None,
    cmap: str = "viridis",
    clim: Optional[Tuple[float, float]] = None,
    robust_q: Tuple[float, float] = (5, 95),
    dpi: int = 150,
    figsize: Tuple[float, float] = (8, 6),
    shading: str = "gouraud",
    grid_on: bool = False,
    verbose: bool = False,
    styles: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """
    Region-masked maps for one or more variables at a chosen depth and time window.
    Handles 3-D and 2-D fields. 2-D fields are auto-lifted; vertical selection is
    applied **after** masking (for absolute-z correctness).
    
    Render **region-masked** maps for one or more variables at a chosen depth and
    time window, saving instantaneous snapshots or the time mean for each region.

    Workflow
    --------
    1. Ensure masking coordinates: requires 'lon' and 'lat' in `ds`.
    2. Depth selection (`select_depth`) and time filtering (`filter_time`) ? `ds_t`.
    3. Build a triangulation once via `build_triangulation(ds)`.
    4. For each region:
       a. Build node mask from polygon spec:
          - shapefile: `polygon_mask_from_shapefile(...)`
          - CSV boundary: `polygon_from_csv_boundary(...)` ? `polygon_mask(...)`
         Optionally derive element mask via `element_mask_from_node_mask(ds, mask_nodes)`.
       b. For each variable:
          - Resolve via `eval_group_or_var(ds_t, var, groups)`.
          - Apply per-variable absolute depth refinement with `select_da_by_z` if needed.
          - Determine center ('node' or 'nele'); skip if neither present.
          - Mask values **outside** the region (nodes or elements). Compute color limits
            from **in-region** values using priority: `norm` > per-var vmin/vmax > `clim`
            > robust percentiles `robust_q`. If no finite in-region values, fall back to (0,1).
          - Plot either requested instants (`at_time`/`at_times`) using `_choose_instants`
            with `time_method` or the mean over the window.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset with grid variables; must include 'lon' and 'lat' for polygon masking.
    variables : list of str
        Variable or expression names to map (resolved via `groups` if provided).
    regions : list of (str, dict)
        Regions as (region_name, spec). `spec` must include either:
          - {"shapefile": <path>, "name_field": ..., "name_equals": ...}
          - {"csv_boundary": <path>, "lon_col": "lon", "lat_col": "lat"}
    depth : Any
        Depth selector passed to `select_depth` (sigma keywords, depth_avg, sigma value,
        or absolute z forms). Absolute depth refinement is applied per-variable.
    months, years : list[int], optional. default surface - can be omitted for 2D vars
        Calendar-based time filters.
    start_date, end_date : str, optional
        Inclusive date bounds "YYYY-MM-DD".
    at_time : Any, optional
        Single desired timestamp for instantaneous plotting.
    at_times : sequence, optional
        Multiple desired timestamps. If neither `at_time` nor `at_times` is provided,
        the function plots the mean over the time window.
    time_method : {"nearest", "pad", "backfill"}, default "nearest"
        Method for matching requested instants to available model times.
    base_dir : str
        Run root; used for filename prefix via `file_prefix(base_dir)`.
    figures_root : str
        Root directory for outputs.
    groups : dict, optional
        Global alias/expression mapping used by `eval_group_or_var`.
    cmap : str, default "viridis"
        Default colormap; per-variable overrides allowed via `styles`.
    clim : (float, float), optional
        Default color limits if not overridden by styles or norm.
    robust_q : (float, float), default (5, 95)
        Percentiles for robust auto-scaling when explicit limits are absent.
    dpi : int, default 150
        PNG resolution.
    figsize : (float, float), default (8, 6)
        Figure size in inches.
    shading : {"flat", "gouraud"}, default "gouraud"
        Tripcolor shading; can be overridden per-variable in `styles`.
    grid_on : bool, default False
        If True, overlay mesh edges for diagnostics.
    verbose : bool, default False
        Print progress, region diagnostics, and skip reasons.
    styles : dict, optional
        Per-variable style overrides. For key `var`, recognized entries:
          - "cmap": str or Colormap
          - "norm": matplotlib.colors.Normalize (highest priority)
          - "vmin": float
          - "vmax": float
          - "shading": {"flat", "gouraud"}

    Returns
    -------
    None
        Saves PNGs named with scope=Region-<name>, variable, depth tag, and time label
        (instant vs mean over window).

    Notes
    -----
    - Center is inferred from data dims:
        'node' ? node-centered; 'nele' ? element-centered (requires element mask; thus
        'nv' is needed to derive it).
    - Color scaling is computed from **in-region** values only (after masking).
    - If a region mask is empty or a variable lacks required dims, that plot is skipped.
    
    """
    if "lon" not in ds or "lat" not in ds:
        raise ValueError("Dataset must contain 'lon' and 'lat' for region masking.")

    # Time filter first; depth is handled per-var on masked subsets
    ds_t = filter_time(ds, months, years, start_date, end_date)

    label_window = build_time_window_label(months, years, start_date, end_date)
    tag = depth_tag(depth)
    outdir = out_dir(base_dir, figures_root)
    prefix = file_prefix(base_dir)
    tri = build_triangulation(ds)
    desired = _timepoints_to_list(at_time, at_times)

    for region_name, spec in regions:
        try:
            if "shapefile" in spec:
                if verbose:
                    print(f"[maps/region {region_name}] shapefile: {spec['shapefile']}")
                mask_nodes = polygon_mask_from_shapefile(
                    ds, spec["shapefile"],
                    name_field=spec.get("name_field"),
                    name_equals=spec.get("name_equals"),
                )
            elif "csv_boundary" in spec:
                if verbose:
                    print(f"[maps/region {region_name}] CSV boundary: {spec['csv_boundary']}")
                poly = polygon_from_csv_boundary(
                    spec["csv_boundary"],
                    lon_col=spec.get("lon_col", "lon"),
                    lat_col=spec.get("lat_col", "lat"),
                )
                mask_nodes = polygon_mask(ds, poly)
            else:
                raise ValueError("Region spec must have 'shapefile' or 'csv_boundary'.")
        except Exception as e:
            print(f"[maps/region {region_name}] failed to build mask: {e}")
            continue

        if not np.any(mask_nodes):
            print(f"[maps/region {region_name}] mask empty; skipping.")
            continue

        mask_elems = element_mask_from_node_mask(ds, mask_nodes)

        for var in variables:
            # 1) evaluate without depth; 2) lift if needed
            try:
                da = eval_group_or_var(ds_t, var, groups)
            except Exception as e:
                print(f"[maps/region {region_name}] Skip '{var}': {e}")
                continue
            if "siglay" not in da.dims:
                if verbose:
                    print(f"[maps/region {region_name}] '{var}' has no 'siglay' — lifting to single layer.")
                sig = xr.DataArray([-0.5], dims=["siglay"], name="siglay")
                da = da.expand_dims(siglay=sig)
                da["siglay"] = sig

            # 3) apply region mask and build an aligned ds_sub (column subset)
            ds_sub = ds_t
            center = "node" if "node" in da.dims else ("nele" if "nele" in da.dims else None)
            if center is None:
                print(f"[maps/region {region_name}] '{var}' has no 'node' or 'nele' dim; skipping.")
                continue
            if center == "node":
                idx_nodes = np.where(mask_nodes)[0]
                da     = da.isel(node=idx_nodes)
                ds_sub = ds_sub.isel(node=idx_nodes)
            else:
                if mask_elems is None:
                    print(f"[maps/region {region_name}] element-centered map requires 'nv' (cannot mask elements).")
                    continue
                idx_elems = np.where(mask_elems)[0]
                da     = da.isel(nele=idx_elems)
                ds_sub = ds_sub.isel(nele=idx_elems)

            # 4) vertical selection on masked subset
            if da.name is None:
                da = da.rename(var)
            if is_absolute_z(depth):
                if isinstance(depth, (float, np.floating, int)):
                    target_z = float(depth)
                elif isinstance(depth, tuple):
                    target_z = float(depth[1])
                else:
                    target_z = float(depth["z_m"])
                da = select_da_by_z(da, ds_sub, target_z, verbose=verbose)
            else:
                ds_tmp   = da.to_dataset(name=da.name)
                ds_depth = select_depth(ds_tmp, depth, verbose=verbose)
                da       = ds_depth[da.name]

            # Per-var style overrides
            cmap_eff   = style_get(var, styles, "cmap", cmap)
            norm_eff   = style_get(var, styles, "norm", None)
            vmin_style = style_get(var, styles, "vmin", None)
            vmax_style = style_get(var, styles, "vmax", None)
            shading_eff= style_get(var, styles, "shading", shading)

            # --- inside region_map, replace make_masked_full() ---
            def make_masked_full(values_full: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
                """
                values_full is the region-subset array (len == len(idx_nodes) or len(idx_elems)).
                Rebuild a full-domain array and compute clim from the in-region values.
                """
                if center == "node":
                    # full domain length
                    n_nodes = int(ds.dims["node"])
                    full = np.full(n_nodes, np.nan, dtype=float)
                    # put region values back into their original node slots
                    full[idx_nodes] = values_full
                    v_in = values_full  # already only in-region
                else:
                    n_elems = int(ds.dims["nele"])
                    full = np.full(n_elems, np.nan, dtype=float)
                    full[idx_elems] = values_full
                    v_in = values_full  # already only in-region
            
                v_in = v_in[np.isfinite(v_in)]
            
                if norm_eff is not None:
                    clim_eff = None
                else:
                    if vmin_style is not None and vmax_style is not None:
                        clim_eff = (vmin_style, vmax_style)
                    elif clim is not None:
                        clim_eff = clim
                    else:
                        clim_eff = (0.0, 1.0) if v_in.size == 0 else robust_clims(v_in, q=robust_q)
            
                return full, clim_eff
            

            if desired:
                for chosen, inst in _choose_instants(da, desired, method=time_method):
                    lbl = _iso_label(chosen) if pd.notnull(chosen) else "NoTime"
                    title = f"Region {region_name} - {var} ({tag}, {lbl})"
                    fname = os.path.join(outdir, f"{prefix}__Map-Region-{region_name}__{var}__{tag}__{lbl}__Instant.png")
                    vals = np.asarray(inst.values).ravel()
                    full, clim_eff = make_masked_full(vals)
                    if center == "node":
                        _plot_tripcolor_full(
                            tri, node_values=full,
                            cmap=cmap_eff, clim=clim_eff, norm=norm_eff,
                            title=title, cbar_label=var, fname=fname, dpi=dpi,
                            figsize=figsize, shading=shading_eff, verbose=verbose, draw_mesh=grid_on,
                        )
                    else:
                        _plot_tripcolor_full(
                            tri, face_values=full,
                            cmap=cmap_eff, clim=clim_eff, norm=norm_eff,
                            title=title, cbar_label=var, fname=fname, dpi=dpi,
                            figsize=figsize, shading=shading_eff, verbose=verbose, draw_mesh=grid_on,
                        )
            else:
                m = da.mean("time", skipna=True) if "time" in da.dims else da
                vals = np.asarray(m.values).ravel()
                full, clim_eff = make_masked_full(vals)
                title = f"Region {region_name} - {var} ({tag}, {label_window})"
                fname = os.path.join(outdir, f"{prefix}__Map-Region-{region_name}__{var}__{tag}__{label_window}__Mean.png")
                if center == "node":
                    _plot_tripcolor_full(
                        tri, node_values=full,
                        cmap=cmap_eff, clim=clim_eff, norm=norm_eff,
                        title=title, cbar_label=var, fname=fname, dpi=dpi,
                        figsize=figsize, shading=shading_eff, verbose=verbose, draw_mesh=grid_on,
                    )
                else:
                    _plot_tripcolor_full(
                        tri, face_values=full,
                        cmap=cmap_eff, clim=clim_eff, norm=norm_eff,
                        title=title, cbar_label=var, fname=fname, dpi=dpi,
                        figsize=figsize, shading=shading_eff, verbose=verbose, draw_mesh=grid_on,
                    )

