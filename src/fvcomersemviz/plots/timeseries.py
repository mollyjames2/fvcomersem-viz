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


def _space_mean(da: xr.DataArray, ds: xr.Dataset, *, verbose: bool = False) -> xr.DataArray:
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
                    if (
                        d in da.coords
                        and d in w.coords
                        and da.coords[d].ndim == 1
                        and w.coords[d].ndim == 1
                    ):
                        try:
                            w = w.sel({d: da[d]})
                            _vprint(
                                verbose,
                                f"[space-mean] Matched weights to '{d}' via .sel; size={w.sizes[d]}.",
                            )
                        except Exception as e:
                            _vprint(
                                verbose,
                                f"[space-mean] Failed to align weights on '{d}' ({e}); using simple mean.",
                            )
                            return da.mean(space_dims, skipna=True)
                    else:
                        _vprint(
                            verbose,
                            f"[space-mean] No coord labels for '{d}' to align weights; using simple mean.",
                        )
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
    # Space dims excluding time and vertical
    return [d for d in da.dims if d not in ("time", "siglay")]


def _align_art1_to_da(
    ds: xr.Dataset, da: xr.DataArray, verbose: bool = False
) -> Optional[xr.DataArray]:
    """Return area weights aligned to the current data selection, or None if unsuitable."""
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
            print(
                f"{prefix}Error: Var '{var}' has no depth dimension; dims={dims}. "
                "Cannot generate depth-differentiated (surface/bottom/profile) plots."
            )
        return False
    return True


# ----------------- plotting functions -----------------


def domain_mean_timeseries(
    ds: xr.Dataset,
    variables: List[str],
    *,
    depth: Any = "surface",  # default to surface
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
    combine_by: Optional[str] = None,  # None | "var"
    show_std: bool = False,
    std_alpha: float = 0.25,
) -> None:
    """
    Plot domain-mean time series for one or more variables, with optional depth selection
    and time filtering. Writes one or multiple PNGs depending on `combine_by`.

    Notes
    -----
    - If show_std=True, plots mean +/- 1 std as a shaded band.
    - Robust to singleton dims returned by reductions (squeezes/reshapes to 1D).
    """
    if combine_by not in (None, "var"):
        raise ValueError("domain_mean_timeseries: combine_by must be None or 'var'.")

    tag = depth_tag(depth)
    label = build_time_window_label(months, years, start_date, end_date)
    prefix = file_prefix(base_dir)
    outdir = out_dir(base_dir, figures_root)

    _vprint(verbose, "[domain] Start domain mean time series")
    _vprint(verbose, f"[domain] Depth={depth} -> tag='{tag}' | Time window='{label}'")

    ds_t = filter_time(ds, months, years, start_date, end_date)

    # (var, tindex, mean_1d, std_1d)
    series: List[Tuple[str, pd.DatetimeIndex, np.ndarray, np.ndarray]] = []
    for var in variables:
        _vprint(verbose, f"[domain] Variable '{var}': resolving with depth handling...")
        try:
            da = resolve_da_with_depth(ds_t, var, depth=depth, groups=groups, verbose=verbose)
        except Exception as e:
            _vprint(verbose, f"[domain] Skipping '{var}': {e}")
            continue

        if "time" not in da.dims:
            _vprint(verbose, f"[domain] '{var}' has no 'time' dimension; skipping.")
            continue

        sdims = _space_dims(da)
        w = _align_art1_to_da(ds, da, verbose=verbose)
        mean, std = weighted_mean_std(da, sdims, w)

        if "time" not in mean.dims:
            _vprint(verbose, f"[domain] '{var}' has no 'time' after reduction; skipping.")
            continue

        tidx = _time_index(mean)

        y = np.asarray(mean.values).squeeze().reshape(-1)
        s = np.asarray(std.values).squeeze().reshape(-1)

        if y.shape[0] != len(tidx):
            raise ValueError(
                f"[domain] '{var}': time len={len(tidx)} but mean shape={y.shape} "
                "after squeeze/reshape. Check reduction dims."
            )
        if s.shape[0] != len(tidx):
            raise ValueError(
                f"[domain] '{var}': time len={len(tidx)} but std shape={s.shape} "
                "after squeeze/reshape. Check reduction dims."
            )

        series.append((var, tidx, y, s))

    if not series:
        _vprint(verbose, "[domain] nothing to plot.")
        return

    if combine_by == "var":
        fig, ax = plt.subplots(figsize=figsize)
        for var, t, y, s in series:
            color = style_get(var, styles, "line_color", None)
            (line,) = ax.plot(t, y, lw=linewidth, color=color, label=var, zorder=2)
            if show_std:
                c = line.get_color()
                ax.fill_between(t, y - s, y + s, alpha=std_alpha, color=c, zorder=1)
        ax.set_title(f"Domain - ({tag}, {label})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend(loc="best")
        fname = f"{prefix}__Domain__multi__{tag}__{label}__Timeseries__CombinedByVar.png"
        fig.savefig(os.path.join(outdir, fname), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        _vprint(verbose, f"[domain] Saved: {os.path.join(outdir, fname)}")
        return

    for var, t, y, s in series:
        fig, ax = plt.subplots(figsize=figsize)
        color = style_get(var, styles, "line_color", None)
        (line,) = ax.plot(t, y, lw=linewidth, color=color, zorder=2)
        if show_std:
            c = line.get_color()
            ax.fill_between(t, y - s, y + s, alpha=std_alpha, color=c, zorder=1)
        ax.set_title(f"{var} - Domain ({tag}, {label})")
        ax.set_xlabel("Time")
        ax.set_ylabel(var)
        fname = f"{prefix}__Domain__{var}__{tag}__{label}__Timeseries.png"
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        _vprint(verbose, f"[domain] Saved: {path}")


def station_timeseries(
    ds: xr.Dataset,
    variables: List[str],
    stations: List[Tuple[str, float, float]],  # (name, lat, lon)
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
    combine_by: Optional[str] = None,  # None | "var" | "station"
) -> None:
    """
    Plot station time series by sampling the nearest grid column at each station
    (node or element), with optional depth selection and time filtering.

    Notes
    -----
    - This function samples a single column (nearest node/element). Spatial std is not
      meaningful here unless you implement a neighborhood (k-nearest) option.
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
    for name, lat, lon in stations:
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
                # do depth selection on the dataset, THEN evaluate the expression
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

    # combine_by='station' -> per variable, lines = stations
    if combine_by == "station":
        for var in variables:
            plotted = []
            fig, ax = plt.subplots(figsize=figsize)
            for name, _lat, _lon in stations:
                s = one_series(var, name)
                if s is None:
                    continue
                t, y = s
                color = style_get(name, styles, "line_color", None)
                ax.plot(t, y, lw=linewidth, label=name, color=color)
                plotted.append(name)
            if not plotted:
                plt.close(fig)
                continue
            ax.set_title(f"{var} - Stations ({tag}, {label})")
            ax.set_xlabel("Time")
            ax.set_ylabel(var)
            ax.legend(loc="best")
            fname = (
                f"{prefix}__Station-All__{var}__{tag}__{label}__Timeseries__CombinedByStation.png"
            )
            fig.savefig(os.path.join(outdir, fname), dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            _vprint(verbose, f"[station] Saved: {os.path.join(outdir, fname)}")
        return

    # combine_by='var' -> per station, lines = variables
    if combine_by == "var":
        for name, _lat, _lon in stations:
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
                plt.close(fig)
                continue
            ax.set_title(f"Station {name} - ({tag}, {label})")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend(loc="best")
            fname = (
                f"{prefix}__Station-{name}__multi__{tag}__{label}__Timeseries__CombinedByVar.png"
            )
            fig.savefig(os.path.join(outdir, fname), dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            _vprint(verbose, f"[station] Saved: {os.path.join(outdir, fname)}")
        return

    # default: one per (station x variable)
    for name, _lat, _lon in stations:
        for var in variables:
            s = one_series(var, name)
            if s is None:
                continue
            t, y = s
            fig, ax = plt.subplots(figsize=figsize)
            color = style_get(var, styles, "line_color", None)
            ax.plot(t, y, lw=linewidth, color=color)
            ax.set_title(f"{var} - Station {name} ({tag}, {label})")
            ax.set_xlabel("Time")
            ax.set_ylabel(var)
            fname = f"{prefix}__Station-{name}__{var}__{tag}__{label}__Timeseries.png"
            path = os.path.join(outdir, fname)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            _vprint(verbose, f"[station] Saved: {path}")



def region_timeseries(
    ds: xr.Dataset,
    variables: List[str],
    regions: List[Tuple[str, Dict[str, Any]]],  # (region_name, spec)
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
    combine_by: Optional[str] = None,  # None | "var" | "region"
    show_std: bool = False,
    std_alpha: float = 0.25,
) -> None:
    """
    Region-mean time series using polygon masks (shapefile or CSV boundary).

    Notes
    -----
    - If show_std=True, plots mean +/- 1 std across space at each time step.
    - Robust to singleton dims returned by reductions (squeezes/reshapes to 1D).
    """
    if combine_by not in (None, "var", "region"):
        raise ValueError("region_timeseries: combine_by must be None, 'var', or 'region'.")

    tag = depth_tag(depth)
    label = build_time_window_label(months, years, start_date, end_date)
    prefix = file_prefix(base_dir)
    outdir = out_dir(base_dir, figures_root)

    if "lon" not in ds or "lat" not in ds:
        raise ValueError("Dataset must contain 'lon' and 'lat' for region masking.")

    _vprint(verbose, f"[region] Start region time series for {len(regions)} region(s)")
    _vprint(verbose, f"[region] Depth={depth} -> tag='{tag}' | Time window='{label}'")

    ds_t = filter_time(ds, months, years, start_date, end_date)

    def region_series(
        region_name: str, spec: Dict[str, Any], var: str
    ) -> Optional[Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]]:
        # --- Build node mask (and element mask if topology present) ---
        try:
            if "shapefile" in spec:
                mask_nodes = polygon_mask_from_shapefile(
                    ds,
                    spec["shapefile"],
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
            _vprint(verbose, f"[region:{region_name}] '{var}' has no 'siglay' - lifting to single layer.")
            sig = xr.DataArray([-0.5], dims=["siglay"], name="siglay")
            da = da.expand_dims(siglay=sig)
            da["siglay"] = sig

        # --- Apply region mask and build column-aligned dataset subset ---
        ds_sub = ds_t
        if "node" in da.dims:
            idx_nodes = np.where(mask_nodes)[0]
            da = da.isel(node=idx_nodes)
            ds_sub = ds_sub.isel(node=idx_nodes)
        elif "nele" in da.dims and mask_elems is not None:
            idx_elems = np.where(mask_elems)[0]
            da = da.isel(nele=idx_elems)
            ds_sub = ds_sub.isel(nele=idx_elems)

        # --- Vertical selection on the masked subset ---
        if "siglay" in da.dims:
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
                ds_tmp = da.to_dataset(name=da.name)
                ds_depth = select_depth(ds_tmp, depth, verbose=verbose)
                da = ds_depth[da.name]

        if "time" not in da.dims:
            _vprint(verbose, f"[region:{region_name}] '{var}' has no 'time'; skip.")
            return None

        sdims = _space_dims(da)
        w = _align_art1_to_da(ds, da, verbose=verbose)
        mean, std = weighted_mean_std(da, sdims, w)

        tidx = _time_index(mean)

        y = np.asarray(mean.values).squeeze().reshape(-1)
        s = np.asarray(std.values).squeeze().reshape(-1)

        if y.shape[0] != len(tidx):
            raise ValueError(
                f"[region:{region_name}] '{var}': time len={len(tidx)} but mean shape={y.shape} "
                "after squeeze/reshape. Check reduction dims."
            )
        if s.shape[0] != len(tidx):
            raise ValueError(
                f"[region:{region_name}] '{var}': time len={len(tidx)} but std shape={s.shape} "
                "after squeeze/reshape. Check reduction dims."
            )

        return tidx, y, s

    # combine_by='var' -> one plot per region, lines = variables
    if combine_by == "var":
        for region_name, spec in regions:
            plotted = []
            fig, ax = plt.subplots(figsize=figsize)
            for var in variables:
                s = region_series(region_name, spec, var)
                if s is None:
                    continue
                t, y, st = s
                color = style_get(var, styles, "line_color", None)
                (line,) = ax.plot(t, y, lw=linewidth, label=var, color=color, zorder=2)
                if show_std:
                    c = line.get_color()
                    ax.fill_between(t, y - st, y + st, alpha=std_alpha, color=c, zorder=1)
                plotted.append(var)
            if not plotted:
                plt.close(fig)
                continue
            ax.set_title(f"Region {region_name} - ({tag}, {label})")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend(loc="best")
            fname = f"{prefix}__Region-{region_name}__multi__{tag}__{label}__Timeseries__CombinedByVar.png"
            fig.savefig(os.path.join(outdir, fname), dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            _vprint(verbose, f"[region:{region_name}] Saved: {os.path.join(outdir, fname)}")
        return

    # combine_by='region' -> one plot per variable, lines = regions
    if combine_by == "region":
        for var in variables:
            series = []
            for region_name, spec in regions:
                s = region_series(region_name, spec, var)
                if s is not None:
                    t, y, st = s
                    series.append((region_name, t, y, st))
            if not series:
                continue

            fig, ax = plt.subplots(figsize=figsize)
            for rname, t, y, st in series:
                color = style_get(rname, styles, "line_color", None)
                (line,) = ax.plot(t, y, lw=linewidth, label=rname, color=color, zorder=2)
                if show_std:
                    c = line.get_color()
                    ax.fill_between(t, y - st, y + st, alpha=std_alpha, color=c, zorder=1)
            ax.set_title(f"{var} - Regions ({tag}, {label})")
            ax.set_xlabel("Time")
            ax.set_ylabel(var)
            ax.legend(loc="best")
            fname = f"{prefix}__Region-All__{var}__{tag}__{label}__Timeseries__CombinedByRegion.png"
            fig.savefig(os.path.join(outdir, fname), dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            _vprint(verbose, f"[region] Saved: {os.path.join(outdir, fname)}")
        return

    # default: one PNG per (region x variable)
    for region_name, spec in regions:
        for var in variables:
            s = region_series(region_name, spec, var)
            if s is None:
                continue
            t, y, st = s
            fig, ax = plt.subplots(figsize=figsize)
            color = style_get(var, styles, "line_color", None)
            (line,) = ax.plot(t, y, lw=linewidth, color=color, zorder=2)
            if show_std:
                c = line.get_color()
                ax.fill_between(t, y - st, y + st, alpha=std_alpha, color=c, zorder=1)
            ax.set_title(f"{var} - Region {region_name} ({tag}, {label})")
            ax.set_xlabel("Time")
            ax.set_ylabel(var)
            fname = f"{prefix}__Region-{region_name}__{var}__{tag}__{label}__Timeseries.png"
            path = os.path.join(outdir, fname)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            _vprint(verbose, f"[region:{region_name}] Saved: {path}")

def _plot_three_panel(
    *,
    t: np.ndarray,
    surf_mean: np.ndarray,
    surf_std: np.ndarray,
    bott_mean: np.ndarray,
    bott_std: np.ndarray,
    zcoord: np.ndarray,
    prof_mean: np.ndarray,
    prof_std: np.ndarray,
    title_prefix: str,
    var: str,
    label: str,
    outdir: str,
    prefix: str,
    styles: Optional[Dict[str, Dict[str, Any]]] = None,
    dpi: int = 150,
    figsize: tuple = (11, 9),
) -> None:
    """
    Internal helper to render a 3-panel figure for a single variable:

      Panel 1: Surface time series (mean +/- 1 std)
      Panel 2: Bottom  time series (mean +/- 1 std)
      Panel 3: Depth profile vs 'siglay' (mean +/- 1 std across time and space)
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

    # Resolve style for this var
    line_color = style_get(var, styles, "line_color", None)  # None -> mpl default cycle
    line_width = style_get(var, styles, "line_width", 1.6)
    shade_alpha = style_get(var, styles, "shade_alpha", 0.25)
    shade_color_pref = style_get(var, styles, "shade_color", None)
    shade_lighten = style_get(var, styles, "shade_lighten", 0.6)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize, constrained_layout=True)

    # Panel 1: surface
    ax = axes[0]
    (line_surf,) = ax.plot(t, surf_mean, lw=line_width, color=line_color, label="mean", zorder=2)
    actual_color = line_surf.get_color()
    shade_color = shade_color_pref or _lighter(actual_color, amount=shade_lighten)
    ax.fill_between(
        t,
        surf_mean - surf_std,
        surf_mean + surf_std,
        alpha=shade_alpha,
        color=shade_color,
        label="+/-1 std",
        zorder=1,
    )
    ax.set_title(f"{title_prefix} - {var} - Surface (+/-1 std)")
    ax.set_xlabel("Time")
    ax.set_ylabel(var)
    ax.legend(loc="best")

    # Panel 2: bottom
    ax = axes[1]
    ax.plot(t, bott_mean, lw=line_width, color=actual_color, label="mean", zorder=2)
    ax.fill_between(
        t,
        bott_mean - bott_std,
        bott_mean + bott_std,
        alpha=shade_alpha,
        color=shade_color,
        label="+/-1 std",
        zorder=1,
    )
    ax.set_title(f"{title_prefix} - {var} - Bottom (+/-1 std)")
    ax.set_xlabel("Time")
    ax.set_ylabel(var)
    ax.legend(loc="best")

    # Panel 3: profile vs siglay
    ax = axes[2]
    ax.plot(prof_mean, zcoord, lw=line_width, color=actual_color, label="mean", zorder=2)
    ax.fill_betweenx(
        zcoord,
        prof_mean - prof_std,
        prof_mean + prof_std,
        alpha=shade_alpha,
        color=shade_color,
        label="+/-1 std",
        zorder=1,
    )
    ax.set_title(f"{title_prefix} - {var} - Profile vs siglay (mean +/-1 std)")
    ax.set_xlabel(var)
    ax.set_ylabel("siglay")
    try:
        if np.nanmax(zcoord) > np.nanmin(zcoord) and zcoord[0] >= zcoord[-1]:
            ax.invert_yaxis()
    except Exception:
        pass
    ax.legend(loc="best")

    fname = f"{prefix}__{title_prefix.replace(' ', '-')}__{var}__3Panel__{label}__Timeseries.png"
    fig.savefig(os.path.join(outdir, fname), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def domain_three_panel(
    ds: xr.Dataset,
    variables: list[str],
    *,
    months=None,
    years=None,
    start_date=None,
    end_date=None,
    base_dir: str,
    figures_root: str,
    groups: Optional[Dict[str, Any]] = None,
    styles: Optional[Dict[str, Dict[str, Any]]] = None,
    dpi: int = 150,
    figsize: tuple = (11, 9),
    verbose: bool = False,
) -> None:
    """
    Render 3-panel summaries for each variable at the domain scale.

    Notes
    -----
    - Surface/bottom time series are computed after select_depth(ds_t, "surface"/"bottom").
    - Spatial mean/std per time step use area weights (art1) if available and alignable.
    """
    ds_t = filter_time(ds, months, years, start_date, end_date)
    label = build_time_window_label(months, years, start_date, end_date)
    outdir = out_dir(base_dir, figures_root)
    prefix = file_prefix(base_dir)

    if verbose:
        print(f"[3panel/domain] Time label={label}")

    for var in variables:
        try:
            da_raw = eval_group_or_var(ds_t, var, groups)
        except Exception as e:
            print(f"[3panel/domain] Skip '{var}': {e}")
            continue

        if not _require_vertical(da_raw, var, where="3panel/domain", verbose=verbose):
            continue

        try:
            da_surf = eval_group_or_var(select_depth(ds_t, "surface"), var, groups)
            da_bott = eval_group_or_var(select_depth(ds_t, "bottom"), var, groups)
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

        # Profile: mean +/- 1 std over (time + space) at each siglay
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
            if verbose:
                print(f"[3panel/domain] '{var}' has no 'siglay'; using surface-only profile.")
            zcoord = np.array([0.0])
            m, s = weighted_mean_std(da_surf, _space_dims(da_surf), w_s)
            prof_mean, prof_std = m.mean("time"), s.mean("time")

        _plot_three_panel(
            t=t,
            surf_mean=surf_mean.values,
            surf_std=surf_std.values,
            bott_mean=bott_mean.values,
            bott_std=bott_std.values,
            zcoord=zcoord,
            prof_mean=np.asarray(prof_mean),
            prof_std=np.asarray(prof_std),
            title_prefix="Domain",
            var=var,
            label=label,
            outdir=outdir,
            prefix=prefix,
            styles=styles,
            dpi=dpi,
            figsize=figsize,
        )


def station_three_panel(
    ds: xr.Dataset,
    variables: list[str],
    stations: List[Tuple[str, float, float]],
    *,
    months=None,
    years=None,
    start_date=None,
    end_date=None,
    base_dir: str,
    figures_root: str,
    groups: Optional[Dict[str, Any]] = None,
    styles: Optional[Dict[str, Dict[str, Any]]] = None,
    dpi: int = 150,
    figsize: tuple = (11, 9),
    verbose: bool = False,
) -> None:
    """
    Render 3-panel summaries for each (station x variable).

    Notes
    -----
    - At a single station column, spatial sigma does not exist, so the +/- band is zero.
      (If you want station neighborhood sigma, add a k-nearest option.)
    """
    if not stations:
        return

    ds_t = filter_time(ds, months, years, start_date, end_date)
    label = build_time_window_label(months, years, start_date, end_date)
    prefix = file_prefix(base_dir)
    outdir = out_dir(base_dir, figures_root)

    for name, lat, lon in stations:
        for var in variables:
            try:
                da_raw = eval_group_or_var(ds_t, var, groups)
            except Exception as e:
                print(f"[3panel/station {name}] Skip '{var}': {e}")
                continue

            if not _require_vertical(da_raw, var, where=f"3panel/station {name}", verbose=verbose):
                continue
            try:
                da_surf = eval_group_or_var(select_depth(ds_t, "surface"), var, groups)
                da_bott = eval_group_or_var(select_depth(ds_t, "bottom"), var, groups)
                da_prof = eval_group_or_var(ds_t, var, groups)
            except Exception as e:
                print(f"[3panel/station {name}] Skip '{var}': {e}")
                continue

            try:
                node_idx = nearest_index_for_dim(ds_t, lat, lon, "node")
            except Exception:
                node_idx = None
            try:
                nele_idx = nearest_index_for_dim(ds_t, lat, lon, "nele")
            except Exception:
                nele_idx = None

            if "node" in da_surf.dims and node_idx is not None:
                da_surf = da_surf.isel(node=node_idx)
            if "nele" in da_surf.dims and nele_idx is not None:
                da_surf = da_surf.isel(nele=nele_idx)
            if "node" in da_bott.dims and node_idx is not None:
                da_bott = da_bott.isel(node=node_idx)
            if "nele" in da_bott.dims and nele_idx is not None:
                da_bott = da_bott.isel(nele=nele_idx)
            if "node" in da_prof.dims and node_idx is not None:
                da_prof = da_prof.isel(node=node_idx)
            if "nele" in da_prof.dims and nele_idx is not None:
                da_prof = da_prof.isel(nele=nele_idx)

            if "time" not in da_surf.dims:
                print(f"[3panel/station {name}] '{var}' has no time dim; skipping.")
                continue

            t = pd.to_datetime(da_surf["time"].values)

            # Spatial sigma does not exist at a single node: provide zero for shading
            surf_mean, surf_std = da_surf, xr.zeros_like(da_surf)
            bott_mean, bott_std = da_bott, xr.zeros_like(da_bott)

            if "siglay" in da_prof.dims:
                prof_mean = da_prof.mean("time", skipna=True)
                prof_std = da_prof.std("time", skipna=True)
                zcoord = da_prof["siglay"].values
            else:
                zcoord = np.array([0.0])
                prof_mean = da_surf.mean("time")
                prof_std = xr.zeros_like(prof_mean)

            _plot_three_panel(
                t=t,
                surf_mean=surf_mean.values,
                surf_std=surf_std.values,
                bott_mean=bott_mean.values,
                bott_std=bott_std.values,
                zcoord=zcoord,
                prof_mean=np.asarray(prof_mean),
                prof_std=np.asarray(prof_std),
                title_prefix=f"Station {name}",
                var=var,
                label=label,
                outdir=outdir,
                prefix=prefix,
                styles=styles,
                dpi=dpi,
                figsize=figsize,
            )


def region_three_panel(
    ds: xr.Dataset,
    variables: List[str],
    regions: List[Tuple[str, Dict[str, Any]]],
    *,
    months=None,
    years=None,
    start_date=None,
    end_date=None,
    base_dir: str,
    figures_root: str,
    groups: Optional[Dict[str, Any]] = None,
    styles: Optional[Dict[str, Dict[str, Any]]] = None,
    dpi: int = 150,
    figsize: tuple = (11, 9),
    verbose: bool = False,
) -> None:
    """
    Render 3-panel summaries for each (region x variable).
    """
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
        try:
            if "shapefile" in spec:
                if verbose:
                    print(f"[3panel/region {region_name}] Using shapefile: {spec['shapefile']}")
                mask_nodes = polygon_mask_from_shapefile(
                    ds,
                    spec["shapefile"],
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

        mask_elems = element_mask_from_node_mask(ds, mask_nodes)
        if verbose and mask_elems is not None:
            print(f"[3panel/region {region_name}] Element mask size: {np.count_nonzero(mask_elems)}")

        for var in variables:
            try:
                da_raw = eval_group_or_var(ds_t, var, groups)
            except Exception as e:
                print(f"[3panel/region {region_name}] Skip '{var}': {e}")
                continue

            if not _require_vertical(
                da_raw, var, where=f"3panel/region {region_name}", verbose=verbose
            ):
                continue
            try:
                da_surf = eval_group_or_var(select_depth(ds_t, "surface"), var, groups)
                da_bott = eval_group_or_var(select_depth(ds_t, "bottom"), var, groups)
                da_prof = eval_group_or_var(ds_t, var, groups)
            except Exception as e:
                print(f"[3panel/region {region_name}] Skip '{var}': {e}")
                continue

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

            surf_mean, surf_std = weighted_mean_std(da_surf, space_dims(da_surf), w_s)
            bott_mean, bott_std = weighted_mean_std(da_bott, space_dims(da_bott), w_b)

            if "siglay" in da_prof.dims:
                prof_mean, prof_std = weighted_mean_std(da_prof, ["time"] + space_dims(da_prof), w_p)
                zcoord = da_prof["siglay"].values
            else:
                zcoord = np.array([0.0])
                prof_mean = da_surf.mean("time", skipna=True)
                prof_std = xr.zeros_like(prof_mean)

            if verbose:
                print(f"[3panel/region {region_name}] Saving 3-panel for '{var}' ({label})")

            _plot_three_panel(
                t=t,
                surf_mean=surf_mean.values,
                surf_std=surf_std.values,
                bott_mean=bott_mean.values,
                bott_std=bott_std.values,
                zcoord=zcoord,
                prof_mean=np.asarray(prof_mean),
                prof_std=np.asarray(prof_std),
                title_prefix=f"Region {region_name}",
                var=var,
                label=label,
                outdir=outdir,
                prefix=prefix,
                styles=styles,
                dpi=dpi,
                figsize=figsize,
            )

