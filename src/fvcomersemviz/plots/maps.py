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
    depth: Any,
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
    ds_depth = select_depth(ds, depth, verbose=verbose)
    ds_t = filter_time(ds_depth, months, years, start_date, end_date)
    label_window = build_time_window_label(months, years, start_date, end_date)
    tag = depth_tag(depth)
    outdir = out_dir(base_dir, figures_root)
    prefix = file_prefix(base_dir)
    tri = build_triangulation(ds)

    desired = _timepoints_to_list(at_time, at_times)

    for var in variables:
        try:
            da = eval_group_or_var(ds_t, var, groups)
            # absolute depth selection (meters, negative downward)
            if (isinstance(depth, (float, np.floating)) and not (-1.0 <= float(depth) <= 0.0)):
                da = select_da_by_z(da, ds_t, float(depth), verbose=verbose)
            elif isinstance(depth, tuple) and len(depth) > 0 and depth[0] == "z_m":
                da = select_da_by_z(da, ds_t, float(depth[1]), verbose=verbose)
            elif isinstance(depth, dict) and "z_m" in depth:
                da = select_da_by_z(da, ds_t, float(depth["z_m"]), verbose=verbose)
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
    depth: Any,
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
    if "lon" not in ds or "lat" not in ds:
        raise ValueError("Dataset must contain 'lon' and 'lat' for region masking.")

    ds_depth = select_depth(ds, depth, verbose=verbose)
    ds_t = filter_time(ds_depth, months, years, start_date, end_date)
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
            try:
                da = eval_group_or_var(ds_t, var, groups)
                # absolute depth selection (meters, negative downward)
                if (isinstance(depth, (float, np.floating)) and not (-1.0 <= float(depth) <= 0.0)):
                    da = select_da_by_z(da, ds_t, float(depth), verbose=verbose)
                elif isinstance(depth, tuple) and len(depth) > 0 and depth[0] == "z_m":
                    da = select_da_by_z(da, ds_t, float(depth[1]), verbose=verbose)
                elif isinstance(depth, dict) and "z_m" in depth:
                    da = select_da_by_z(da, ds_t, float(depth["z_m"]), verbose=verbose)
            except Exception as e:
                print(f"[maps/region {region_name}] Skip '{var}': {e}")
                continue

            # Per-var style overrides
            cmap_eff   = style_get(var, styles, "cmap", cmap)
            norm_eff   = style_get(var, styles, "norm", None)
            vmin_style = style_get(var, styles, "vmin", None)
            vmax_style = style_get(var, styles, "vmax", None)
            shading_eff= style_get(var, styles, "shading", shading)

            center = "node" if "node" in da.dims else ("nele" if "nele" in da.dims else None)
            if center is None:
                print(f"[maps/region {region_name}] '{var}' has no 'node' or 'nele' dim; skipping.")
                continue

            def make_masked_full(values_full: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
                """Mask outside region and compute clim (from in-region values) if needed."""
                if center == "node":
                    full = np.array(values_full, dtype=float).copy()
                    full[~mask_nodes] = np.nan
                    v_in = values_full[mask_nodes]
                else:
                    if mask_elems is None:
                        raise RuntimeError("Element-centered region map requires 'nv' to make element mask.")
                    full = np.array(values_full, dtype=float).copy()
                    full[~mask_elems] = np.nan
                    v_in = values_full[mask_elems]

                v_in = v_in[np.isfinite(v_in)]
                if norm_eff is not None:
                    clim_eff = None
                else:
                    if vmin_style is not None and vmax_style is not None:
                        clim_eff = (vmin_style, vmax_style)
                    elif clim is not None:
                        clim_eff = clim
                    else:
                        if v_in.size == 0:
                            clim_eff = (0.0, 1.0)
                        else:
                            clim_eff = robust_clims(v_in, q=robust_q)
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
