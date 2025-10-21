from __future__ import annotations

# --- Safe, headless backend for batch animations (avoids Tk crashes) ---
import matplotlib as _mpl
# If a GUI backend is active (TkAgg/Qt/MacOSX/etc), switch to Agg for saving.
try:
    _backend = _mpl.get_backend().lower()
except Exception:
    _backend = ""  # be defensive
if any(k in _backend for k in ("tk", "qt", "macosx", "wx")):
    _mpl.use("Agg")
# -----------------------------------------------------------------------

from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

from ..io import filter_time, eval_group_or_var
from ..utils import (
    out_dir, file_prefix, depth_tag, build_time_window_label, style_get,
    select_depth, select_da_by_z,build_triangulation,
    robust_clims
)

from ..regions import build_region_masks, nearest_node_index




# --- helper: filename builder with explicit combined_by tag ---
def _fname(prefix: str, scope: str, region_name: Optional[str], station_name: Optional[str],
           var_or_multi: str, tag: str, label: str,
           combined_by: Optional[str]) -> str:
    scope_str = "Domain" if scope == "domain" else (
        f"Region_{region_name}" if scope == "region" else f"Station_{station_name}"
    )
    base = f"{prefix}__{scope_str}__{var_or_multi}__{tag}__{label}__TimeseriesAnim"
    if combined_by:
        base += f"__CombinedBy{combined_by.capitalize()}"
    return base + ".gif"

def _validate_combine_by(scope: str, combine_by: Optional[str]) -> Optional[str]:
    if combine_by is None:
        return None
    combine_by = combine_by.lower()
    if combine_by not in ("var", "region", "station"):
        raise ValueError("combine_by must be one of: 'var', 'region', 'station', or None")
    if scope == "domain" and combine_by in ("region", "station"):
        raise ValueError("combine_by='region' or 'station' only make sense with scope='region' or scope='station'.")
    if scope == "region" and combine_by == "station":
        raise ValueError("combine_by='station' not valid when scope='region'.")
    if scope == "station" and combine_by == "region":
        raise ValueError("combine_by='region' not valid when scope='station'.")
    return combine_by
    
    # -----------------------
# Local helpers (required)
# -----------------------

def _vprint(verbose: bool, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)

def _ensure_list_int(v: Optional[Union[int, Sequence[int]]]) -> Optional[List[int]]:
    if v is None:
        return None
    if isinstance(v, (int, np.integer)):
        return [int(v)]
    return list(v)

def _validate_scope(scope: str,
                    regions: Optional[Sequence[Tuple[str, Dict[str, Any]]]],
                    stations: Optional[Sequence[Tuple[str, float, float]]]) -> str:
    scope = scope.lower().strip()
    if scope not in ("domain", "region", "station"):
        raise ValueError("scope must be one of: 'domain', 'region', 'station'")
    if scope == "region" and (not regions or len(regions) == 0):
        raise ValueError("scope='region' requires a non-empty list of regions ([(name, spec), ...]).")
    if scope == "station" and (not stations or len(stations) == 0):
        raise ValueError("scope='station' requires a non-empty list of stations ([(name, lat, lon), ...]).")
    return scope

def _space_mean(da: xr.DataArray, ds_full: xr.Dataset, *, verbose: bool=False) -> xr.DataArray:
    """Weighted spatial mean using 'art1' when alignable; else simple mean over non-time dims."""
    space_dims = [d for d in da.dims if d != "time"]
    if not space_dims:
        return da
    if "art1" in ds_full and all(d in ds_full["art1"].dims for d in space_dims):
        w = ds_full["art1"]
        try:
            w_aligned = w
            # align along any shared 1D coords
            for d in w.dims:
                if d in da.coords:
                    w_aligned = w_aligned.sel({d: da[d]})
            return (da * w_aligned).sum(space_dims) / w_aligned.sum(space_dims)
        except Exception as e:
            _vprint(verbose, f"[space-mean] weight alignment failed ({e}); simple mean fallback.")
    return da.mean(space_dims)

def _apply_depth(da: xr.DataArray, ds_for_z: xr.Dataset, depth: Any, *, verbose: bool) -> xr.DataArray:
    """
    Depth modes:
      - 'surface' | 'bed' | 'bottom' | 'depth_avg'
      - sigma index: int or ('sigma', k)
      - absolute metres: float (e.g., -10.0), or {'z_m': -10}
    """
    from ..utils import select_depth, select_da_by_z
    if depth is None:
        return da
    if isinstance(depth, str) and depth.lower() == "bed":
        depth = "bottom"
    da2 = select_depth(da, depth, verbose=verbose)
    # absolute z slice (metres below surface)
    if isinstance(depth, (float, np.floating)) and not (-1.0 <= float(depth) <= 0.0):
        return select_da_by_z(da2, ds_for_z, float(depth), verbose=verbose)
    if isinstance(depth, tuple) and len(depth) > 0 and depth[0] == "z_m":
        return select_da_by_z(da2, ds_for_z, float(depth[1]), verbose=verbose)
    if isinstance(depth, dict) and "z_m" in depth:
        return select_da_by_z(da2, ds_for_z, float(depth["z_m"]), verbose=verbose)
    return da2

def _title_bits(scope: str,
                region_name: Optional[str],
                station_name: Optional[str],
                tag: str, label: str) -> str:
    if scope == "domain":
        return f"Domain ({tag}, {label})"
    if scope == "region":
        return f"Region {region_name} ({tag}, {label})"
    if scope == "station":
        return f"Station {station_name} ({tag}, {label})"
    return f"{tag}, {label}"

# --- Helpers for explicit instants & friendly frequencies ---

def _timepoints_to_list(at_time: Optional[Any], at_times: Optional[Sequence[Any]]) -> Optional[List[pd.Timestamp]]:
    if at_times is not None:
        return [pd.to_datetime(t) for t in at_times]
    if at_time is not None:
        return [pd.to_datetime(at_time)]
    return None

def _choose_instants(
    da: xr.DataArray,
    desired: List[pd.Timestamp],
    method: str = "nearest",
) -> List[Tuple[pd.Timestamp, xr.DataArray]]:  # (chosen_time, instantaneous-DA)
    if "time" not in da.dims:
        return [(pd.Timestamp("NaT"), da)]
    out = []
    for want in desired:
        _one = da.sel(time=want, method=method)
        chosen = pd.to_datetime(np.atleast_1d(_one["time"].values)[0])
        out.append((pd.Timestamp(chosen), _one))
    return out

def _normalize_frequency(freq: Optional[str]) -> Optional[str]:
    """
    Map friendly names to pandas aliases:
    'hourly' -> 'H', 'daily' -> 'D', 'monthly' -> 'MS' (month-start).
    """
    if not freq:
        return None
    f = str(freq).strip().lower()
    if f in ("h", "hour", "hourly", "1h"):
        return "H"
    if f in ("d", "day", "daily", "1d"):
        return "D"
    if f in ("m", "mon", "month", "monthly"):
        return "MS"
    raise ValueError("frequency must be one of: hourly, daily, monthly (or None)")


# --- main entry point  ---
def animate_timeseries(
    ds: xr.Dataset,
    *,
    vars: Sequence[str],
    groups: Optional[Dict[str, Any]],
    scope: str,                                  # 'domain' | 'region' | 'station'
    regions: Optional[Sequence[Tuple[str, Dict[str, Any]]]] = None,   # list of (name, spec)
    stations: Optional[Sequence[Tuple[str, float, float]]] = None,    # list of (name, lat, lon)
    # time filters (any combination)
    months: Optional[Union[int, Sequence[int]]] = None,
    years: Optional[Union[int, Sequence[int]]]  = None,
    start_date: Optional[str] = None,
    end_date: Optional[str]   = None,
    # NEW: timestep control
    at_time: Optional[Any] = None,
    at_times: Optional[Sequence[Any]] = None,
    time_method: str = "nearest",
    frequency: Optional[str] = None,                    # 'hourly' | 'daily' | 'monthly' | None
    # depth
    depth: Any = "surface",
    # output + styling
    base_dir: str = "",
    figures_root: str = "",
    combine_by: Optional[str] = None,                   # 'var' | 'region' | 'station' | None
    linewidth: float = 1.8,
    figsize: Tuple[int, int] = (10, 4),
    dpi: int = 150,
    styles: Optional[Dict[str, Dict[str, Any]]] = None,
    verbose: bool = True,
) -> List[str]:
    """
    Create **growing-line animations** of time series for variables, scoped to the
    domain, specific regions, or stations. Saves animated GIFs and returns their paths.

    Frame selection
    ---------------
    The sequence of animation frames is chosen in this priority:
    1) If `at_time`/`at_times` are provided ⇒ use those instants (matched via `time_method`).
    2) Else if `frequency` in {'hourly','daily','monthly'} ⇒ sample one representative
       timestep per period (nearest available).
    3) Else ⇒ use **every** timestep in the filtered window.

    Combination modes (`combine_by`)
    --------------------------------
    - ``None``        : one animation per (scope item × variable).
    - ``'var'``       : one animation per scope item; **multiple variables** as separate lines.
    - ``'region'``    : (scope='region') one animation per variable; **lines = regions**.
    - ``'station'``   : (scope='station') one animation per variable; **lines = stations**.

    Workflow
    --------
    1. Filter dataset by depth (`select_depth`) and time (`filter_time`).
    2. Resolve each variable using `eval_group_or_var`, also handling absolute-z depth via `_apply_depth`.
    3. For each scope item:
       - domain  → area-weighted mean via `_space_mean`.
       - region  → build masks with `build_region_masks`, then area-weighted mean.
       - station → nearest column (node/element) series.
    4. Choose frame indices from available times (explicit instants, cadence, or all).
    5. Build a Matplotlib `FuncAnimation` that progressively extends each line up to frame `k`.
    6. Save to GIF (Pillow writer) using a filename that encodes scope, depth tag, time label,
       frequency tag (if any), and combination mode.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset.
    vars : sequence of str
        Variable or expression names to animate (resolved via `groups` if provided).
    groups : dict, optional
        Global alias/composite expressions for variable resolution.
    scope : {"domain","region","station"}
        Which spatial scope to animate. For 'region'/'station', provide `regions`/`stations`.
    regions : sequence of (str, dict), optional
        Regions as `(name, spec)`; required when `scope="region"`.
    stations : sequence of (str, float, float), optional
        Stations as `(name, lat, lon)`; required when `scope="station"`.
    months, years : int or sequence of int, optional
        Calendar filters. Accept single ints or lists; internally normalized to lists.
    start_date, end_date : str, optional
        Inclusive date bounds "YYYY-MM-DD".
    at_time : any, optional
        A single desired timestamp (np.datetime64, pandas Timestamp, ISO string, etc.).
    at_times : sequence, optional
        Multiple desired timestamps.
    time_method : {"nearest","pad","backfill"}, default "nearest"
        Method for matching desired instants to available model times.
    frequency : {"hourly","daily","monthly"} or None, optional
        Periodic sampling for frames when explicit `at_time(s)` not given.
    depth : Any, default "surface"
        Depth selector passed to `select_depth`; absolute-z handled by `_apply_depth`.
    base_dir : str, default ""
        Run root used by `file_prefix(base_dir)` for filename stems.
    figures_root : str, default ""
        Output directory root; if empty, current working directory is used.
    combine_by : {None,"var","region","station"}, optional
        Controls how lines are grouped in a single animation (see above).
    linewidth : float, default 1.8
        Line width for plots.
    figsize : (int, int), default (10, 4)
        Figure size in inches.
    dpi : int, default 150
        Render resolution for saved GIFs.
    styles : dict, optional
        Per-series line style overrides (e.g., `styles["chl"]["line_color"] = "C3"`).
    verbose : bool, default True
        Print progress and skip reasons.

    Returns
    -------
    list of str
        Full filesystem paths to the saved GIFs.

    Notes
    -----
    - Area-weighted means use `'art1'` if available and alignable; otherwise simple means.
    - Y-limits are padded by 5% to reduce clipping when lines grow.
    - File names encode scope, depth tag, time window label and, when applicable,
      a frequency tag like `__FreqDaily`.
    """


    # -----------------------
    # Local helpers (new)
    # -----------------------
   

    def _frame_indices_for_time_axis(t_axis: np.ndarray) -> List[int]:
        """
        Return sorted, unique indices into t_axis per (at_times | frequency | all points).
        """
        if t_axis is None or len(t_axis) == 0:
            return [0]
        idx = pd.to_datetime(t_axis)
        desired_list = _timepoints_to_list(at_time, at_times)
        freq_alias   = _normalize_frequency(frequency)

        if desired_list:
            pos = idx.get_indexer(desired_list, method=time_method if time_method else "nearest")
            pos = [int(p) for p in pos if p >= 0]
        elif freq_alias is not None:
            buckets = (
                pd.Series(range(len(idx)), index=idx)
                .groupby(pd.Grouper(freq=freq_alias))
                .first()
                .dropna()
                .index
            )
            pos = idx.get_indexer(buckets, method="nearest")
            pos = [int(p) for p in pos if p >= 0]
        else:
            pos = list(range(len(idx)))

        pos = sorted(set(pos))
        if not pos:
            pos = [0]
        return pos

    # -----------------------
    # Original logic (with small tweaks)
    # -----------------------
    scope = _validate_scope(scope, regions, stations)

    # Back-compat: handle legacy 'combine' flag if present elsewhere
    if combine_by is None:
        try:
            if combine is True:  # type: ignore[name-defined]
                combine_by = "var"
        except NameError:
            pass
    combine_by = _validate_combine_by(scope, combine_by)

    months_l = _ensure_list_int(months)
    years_l  = _ensure_list_int(years)

    dtag   = depth_tag(depth)
    tlabel = build_time_window_label(months_l, years_l, start_date, end_date)
    prefix = file_prefix(base_dir) if base_dir else "dataset"
    outdir = out_dir(base_dir, figures_root) if figures_root else os.getcwd()

    # For filename tagging when a cadence is used (not for explicit instants)
    freq_alias_for_tag = _normalize_frequency(frequency)
    freq_tag = f"__Freq{str(frequency).capitalize()}" if (freq_alias_for_tag and not _timepoints_to_list(at_time, at_times)) else ""

    ds_win = filter_time(select_depth(ds, depth, verbose=verbose), months_l, years_l, start_date, end_date)

    # resolve variables (native or via GROUPS)
    resolved: List[Tuple[str, xr.DataArray]] = []
    for v in vars:
        try:
            da = eval_group_or_var(ds_win, v, groups)
            da = _apply_depth(da, ds_win, depth, verbose=verbose)  # absolute z if needed
            resolved.append((v, da))
        except Exception as e:
            _vprint(verbose, f"[animate] skipping '{v}': {e}")

    if not resolved:
        _vprint(verbose, "[animate] no variables resolved; nothing to do.")
        return []

    # scope iterators (keep your input formats)
    if scope == "domain":
        scopes = [("domain", None, None)]  # (scope, region_name, station_tuple)
    elif scope == "region":
        scopes = [("region", name, None) for (name, _spec) in regions]  # type: ignore[arg-type]
    else:
        scopes = [("station", None, st) for st in stations]  # type: ignore[arg-type]

    # helper: 1D series for a given var & scope item
    def series_for(scope_kind: str,
                   var_name: str,
                   da: xr.DataArray,
                   region_name: Optional[str],
                   station_tuple: Optional[Tuple[str, float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        target = da
        if scope_kind == "region":
            spec = None
            for (nm, sp) in regions:  # type: ignore[iteration-over-optional]
                if nm == region_name:
                    spec = sp
                    break
            if spec is None:
                raise KeyError(f"Region '{region_name}' not found in regions list.")
            mask_nodes, mask_elems = build_region_masks(ds_win, (region_name, spec), verbose=verbose)
            if mask_nodes is not None:
                for d in ("node", "nnode", "nodes"):
                    if d in target.dims and isinstance(mask_nodes, np.ndarray):
                        target = target.where(xr.DataArray(mask_nodes, dims=(d,)), drop=True)
                        break
            if mask_elems is not None:
                for d in ("nele", "element", "elem"):
                    if d in target.dims and isinstance(mask_elems, np.ndarray):
                        target = target.where(xr.DataArray(mask_elems, dims=(d,)), drop=True)
                        break
            m = _space_mean(target, ds_win, verbose=verbose)

        elif scope_kind == "station":
            name, lat, lon = station_tuple  # type: ignore
            idx = nearest_node_index(ds_win, lat, lon)
            picked = None
            for d in ("node", "nnode", "nodes"):
                if d in target.dims:
                    picked = target.isel({d: idx})
                    break
            m = picked if picked is not None else target
        else:
            m = _space_mean(target, ds_win, verbose=verbose)

        if "time" not in m.dims:
            raise ValueError(f"'{var_name}' has no 'time' dimension after selection.")
        return m["time"].values, m.values

    outputs: List[str] = []

    # ---------- MODE A: combine_by == 'var' (one animation per scope item; lines = variables) ----------
    if combine_by == "var":
        for scope_kind, region_name, station_tuple in scopes:
            # gather series across variables for this scope item
            series = []
            for (vname, da) in resolved:
                try:
                    t, y = series_for(scope_kind, vname, da, region_name, station_tuple)
                    series.append((vname, t, y))
                except Exception as e:
                    _vprint(verbose, f"[animate] {scope_kind}: '{vname}' failed -> {e}")
            if not series:
                continue

            t0 = series[0][1]
            frames_idx = _frame_indices_for_time_axis(t0)

            ymin = np.nanmin([np.nanmin(y) for (_, _, y) in series])
            ymax = np.nanmax([np.nanmax(y) for (_, _, y) in series])
            pad  = 0.05 * (ymax - ymin if ymax > ymin else (abs(ymax) + 1.0))

            fig, ax = plt.subplots(figsize=figsize)
            lines = {}
            for (vname, _t, _y) in series:
                color = style_get(vname, styles, "line_color", None)
                (ln,) = ax.plot([], [], lw=linewidth, label=vname, color=color)
                lines[vname] = ln
            ax.set_xlim(t0[0], t0[-1]); ax.set_ylim(ymin - pad, ymax + pad)
            ax.set_xlabel("Time"); ax.set_ylabel("Value")
            title = _title_bits(scope_kind, region_name, (station_tuple[0] if station_tuple else None), dtag, tlabel)
            ax.set_title(title); ax.legend(loc="best")

            def update(i):
                k = frames_idx[i]
                for (vname, t, y) in series:
                    lines[vname].set_data(t[:k+1], y[:k+1])
                return tuple(lines.values())

            ani = animation.FuncAnimation(fig, update, frames=len(frames_idx), interval=80, blit=False)
            var_label = "multi"
            fname = _fname(prefix, scope_kind, region_name, (station_tuple[0] if station_tuple else None),
                           var_label, dtag, f"{tlabel}{freq_tag}", combined_by="var")
            path = os.path.join(outdir, fname)
            ani.save(path, writer=animation.PillowWriter(fps=10), dpi=dpi)
            plt.close(fig)
            outputs.append(path)
            _vprint(verbose, f"[animate] saved: {path}")

        return outputs

    # ---------- MODE B: combine_by == 'region' (scope='region'; one animation per variable; lines = regions) ----------
    if combine_by == "region":
        for (vname, da) in resolved:
            series = []
            for (scope_kind, region_name, _st) in scopes:  # scopes are all regions here
                try:
                    t, y = series_for(scope_kind, vname, da, region_name, None)
                    series.append((region_name, t, y))
                except Exception as e:
                    _vprint(verbose, f"[animate] region '{region_name}' for '{vname}' failed -> {e}")
            if not series:
                continue

            t0 = series[0][1]
            frames_idx = _frame_indices_for_time_axis(t0)

            ymin = np.nanmin([np.nanmin(y) for (_, _, y) in series])
            ymax = np.nanmax([np.nanmax(y) for (_, _, y) in series])
            pad  = 0.05 * (ymax - ymin if ymax > ymin else (abs(ymax) + 1.0))

            fig, ax = plt.subplots(figsize=figsize)
            lines = {}
            for (rname, _t, _y) in series:
                color = style_get(rname, styles, "line_color", None)
                (ln,) = ax.plot([], [], lw=linewidth, label=rname, color=color)
                lines[rname] = ln
            ax.set_xlim(t0[0], t0[-1]); ax.set_ylim(ymin - pad, ymax + pad)
            ax.set_xlabel("Time"); ax.set_ylabel(vname)
            ax.set_title(f"{vname} — Regions ({dtag}, {tlabel})")
            ax.legend(loc="best")

            def update(i):
                k = frames_idx[i]
                for (rname, t, y) in series:
                    lines[rname].set_data(t[:k+1], y[:k+1])
                return tuple(lines.values())

            ani = animation.FuncAnimation(fig, update, frames=len(frames_idx), interval=80, blit=False)
            fname = _fname(prefix, "region", "All", None, vname, dtag, f"{tlabel}{freq_tag}", combined_by="region")
            path = os.path.join(outdir, fname)
            ani.save(path, writer=animation.PillowWriter(fps=10), dpi=dpi)
            plt.close(fig)
            outputs.append(path)
            _vprint(verbose, f"[animate] saved: {path}")

        return outputs

    # ---------- MODE C: combine_by == 'station' (scope='station'; one animation per variable; lines = stations) ----------
    if combine_by == "station":
        for (vname, da) in resolved:
            series = []
            for (scope_kind, _r, st) in scopes:  # scopes are all stations here
                try:
                    t, y = series_for(scope_kind, vname, da, None, st)
                    series.append((st[0], t, y))  # label by station name
                except Exception as e:
                    _vprint(verbose, f"[animate] station '{st[0]}' for '{vname}' failed -> {e}")
            if not series:
                continue

            t0 = series[0][1]
            frames_idx = _frame_indices_for_time_axis(t0)

            ymin = np.nanmin([np.nanmin(y) for (_, _, y) in series])
            ymax = np.nanmax([np.nanmax(y) for (_, _, y) in series])
            pad  = 0.05 * (ymax - ymin if ymax > ymin else (abs(ymax) + 1.0))

            fig, ax = plt.subplots(figsize=figsize)
            lines = {}
            for (stname, _t, _y) in series:
                color = style_get(stname, styles, "line_color", None)
                (ln,) = ax.plot([], [], lw=linewidth, label=stname, color=color)
                lines[stname] = ln
            ax.set_xlim(t0[0], t0[-1]); ax.set_ylim(ymin - pad, ymax + pad)
            ax.set_xlabel("Time"); ax.set_ylabel(vname)
            ax.set_title(f"{vname} — Stations ({dtag}, {tlabel})")
            ax.legend(loc="best")

            def update(i):
                k = frames_idx[i]
                for (stname, t, y) in series:
                    lines[stname].set_data(t[:k+1], y[:k+1])
                return tuple(lines.values())

            ani = animation.FuncAnimation(fig, update, frames=len(frames_idx), interval=80, blit=False)
            fname = _fname(prefix, "station", None, "All", vname, dtag, f"{tlabel}{freq_tag}", combined_by="station")
            path = os.path.join(outdir, fname)
            ani.save(path, writer=animation.PillowWriter(fps=10), dpi=dpi)
            plt.close(fig)
            outputs.append(path)
            _vprint(verbose, f"[animate] saved: {path}")

        return outputs

    # ---------- MODE D: combine_by is None (no combining; one per scope item × variable) ----------
    for scope_kind, region_name, station_tuple in scopes:
        for (vname, da) in resolved:
            try:
                t, y = series_for(scope_kind, vname, da, region_name, station_tuple)
            except Exception as e:
                _vprint(verbose, f"[animate] {scope_kind}: '{vname}' failed -> {e}")
                continue

            frames_idx = _frame_indices_for_time_axis(t)

            ymin, ymax = np.nanmin(y), np.nanmax(y)
            pad = 0.05 * (ymax - ymin if ymax > ymin else (abs(ymax) + 1.0))

            fig, ax = plt.subplots(figsize=figsize)
            color = style_get(vname, styles, "line_color", None)
            (line,) = ax.plot([], [], lw=linewidth, color=color)
            ax.set_xlim(t[0], t[-1]); ax.set_ylim(ymin - pad, ymax + pad)
            ax.set_xlabel("Time"); ax.set_ylabel(vname)
            ax.set_title(f"{vname} — {_title_bits(scope_kind, region_name, (station_tuple[0] if station_tuple else None), dtag, tlabel)}")

            def update(i):
                k = frames_idx[i]
                line.set_data(t[:k+1], y[:k+1])
                return (line,)

            ani = animation.FuncAnimation(fig, update, frames=len(frames_idx), interval=80, blit=True)
            fname = _fname(prefix, scope_kind, region_name, (station_tuple[0] if station_tuple else None),
                           vname, dtag, f"{tlabel}{freq_tag}", combined_by=None)
            path = os.path.join(outdir, fname)
            ani.save(path, writer=animation.PillowWriter(fps=10), dpi=dpi)
            plt.close(fig)
            outputs.append(path)
            _vprint(verbose, f"[animate] saved: {path}")

    return outputs
 
def animate_maps(
    ds: xr.Dataset,
    *,
    variables: Sequence[str],
    scope: str = "domain",                        # 'domain' | 'region'
    regions: Optional[Sequence[Tuple[str, Dict[str, Any]]]] = None,
    # time filters (window)
    months: Optional[Union[int, Sequence[int]]] = None,
    years: Optional[Union[int, Sequence[int]]]  = None,
    start_date: Optional[str] = None,
    end_date: Optional[str]   = None,
    # optional explicit instants
    at_time: Optional[Any] = None,
    at_times: Optional[Sequence[Any]] = None,
    time_method: str = "nearest",
    frequency: Optional[str] = None,              # 'hourly' | 'daily' | 'monthly' | None
    # depth selection
    depth: Any = "surface",
    # IO / styling
    base_dir: str = "",
    figures_root: str = "",
    groups: Optional[Dict[str, Any]] = None,
    cmap: str = "viridis",
    clim: Optional[Tuple[float, float]] = None,   # overrides robust if provided (unless vmin/vmax set via styles)
    robust_q: Tuple[float, float] = (5, 95),
    shading: str = "gouraud",                     # node-centered; element-centered will force 'flat'
    grid_on: bool = False,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 150,
    interval_ms: int = 100,
    fps: int = 10,
    styles: Optional[Dict[str, Dict[str, Any]]] = None,
    verbose: bool = True,
) -> List[str]:
    """
    Animate **maps over time** for one or more variables on a triangular mesh,
    at a selected depth and spatial scope (domain or regions). Saves animated GIFs
    and returns their paths.

    Frames
    ------
    Priority for frame times (like `animate_timeseries`):
      1) Explicit `at_time`/`at_times` matched via `time_method`.
      2) If `frequency` is given ('hourly'|'daily'|'monthly'): sample one frame per period.
      3) Otherwise: include every timestep in the time-filtered window.

    Depth modes
    -----------
    Accepts the same selectors as static maps, e.g.:
      "surface" | "bottom" | "depth_avg" | sigma index | float sigma |
      absolute meters below surface: -10.0 | {"z_m": -10} | ("z_m", -10).

    Colour scaling (per variable)
    -----------------------------
    Priority: `styles[var]["norm"]` > (`styles[var]["vmin"]`,`styles[var]["vmax"]`)
              > `clim` argument > **robust** percentiles (`robust_q`) computed
              across the **masked values in the selected frames**.

    Workflow
    --------
    1. Select depth (`select_depth`) and time window (`filter_time`) → `ds_t`.
    2. Build a triangulation from `ds` with `build_triangulation(ds)`.
    3. For each variable:
       a. Resolve via `eval_group_or_var(ds_t, var, groups)`, then apply absolute-z
          refinement if applicable.
       b. Build the list of frames (explicit instants, cadence, or all).
       c. For each scope item:
          - domain → no mask.
          - region → build masks via `build_region_masks`; mask values outside region.
          Compute colour limits according to the priority above.
       d. Create a `FuncAnimation` that re-draws `tripcolor` for each frame using
          node- or element-centered values (element-centered forces `shading='flat'`).
    4. Save GIF with a filename that encodes scope, var, depth tag, time label,
       and frequency tag (if any).

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset with lon/lat and connectivity (`nv`) for triangulation.
    variables : sequence of str
        Variable/expression names to animate (resolved via `groups` if provided).
    scope : {"domain","region"}, default "domain"
        Spatial scope. For `"region"`, supply `regions`.
    regions : sequence of (str, dict), optional
        Regions as `(name, spec)`; required when `scope="region"`.
    months, years : int or sequence of int, optional
        Calendar-based time filters (normalized to lists).
    start_date, end_date : str, optional
        Inclusive date bounds "YYYY-MM-DD".
    at_time, at_times : optional
        Explicit instants to show (see also `time_method`).
    time_method : {"nearest","pad","backfill"}, default "nearest"
        Method for matching requested instants to available model times.
    frequency : {"hourly","daily","monthly"} or None, optional
        If given and no explicit instants, sample one frame per period.
    depth : Any, default "surface"
        Depth selector for `select_depth`; absolute-z refinements are applied per variable.
    base_dir : str, default ""
        Run root used by `file_prefix(base_dir)` for filename stems.
    figures_root : str, default ""
        Output root directory; if empty, current working directory is used.
    groups : dict, optional
        Global alias/composite expressions for variable resolution.
    cmap : str, default "viridis"
        Default colormap (overridden by `styles[var]["cmap"]` if present).
    clim : (float, float), optional
        Global vmin/vmax (used if no per-variable vmin/vmax and no norm).
    robust_q : (float, float), default (5, 95)
        Percentiles for robust auto-scaling when limits are not explicit.
    shading : {"flat","gouraud"}, default "gouraud"
        Shading for node-centered `tripcolor`; element-centered values force `"flat"`.
    grid_on : bool, default False
        If True, overlay mesh edges for diagnostics.
    figsize : (float, float), default (8, 6)
        Figure size in inches.
    dpi : int, default 150
        Render resolution for saved GIFs.
    interval_ms : int, default 100
        Delay between frames used by `FuncAnimation` (milliseconds).
    fps : int, default 10
        Frame rate for writing GIFs with Pillow.
    styles : dict, optional
        Per-variable overrides: "cmap", "norm", "vmin", "vmax", "shading".
    verbose : bool, default True
        Print progress and skip reasons.

    Returns
    -------
    list of str
        Full filesystem paths to the saved GIFs.

    Notes
    -----
    - Plot center is inferred from data dims: 'node' or 'nele'.
    - Region masking uses node/element masks consistently with the data center.
    - When computing robust limits, only **in-scope** (masked) values are used.
    - File names encode scope, variable, depth tag, time label, and optional frequency tag.
    """
    # validate scope
    scope = scope.lower().strip()
    if scope not in ("domain", "region"):
        raise ValueError("scope must be 'domain' or 'region'")
    if scope == "region" and (not regions or len(regions) == 0):
        raise ValueError("scope='region' requires a non-empty list of regions ([(name, spec), ...]).")

    # normalize month/year lists
    months_l = _ensure_list_int(months)
    years_l  = _ensure_list_int(years)

    # labels / paths
    dtag   = depth_tag(depth)
    tlabel = build_time_window_label(months_l, years_l, start_date, end_date)
    prefix = file_prefix(base_dir) if base_dir else "dataset"
    outdir = out_dir(base_dir, figures_root) if figures_root else os.getcwd()

    # select depth + time window
    ds_depth = select_depth(ds, depth, verbose=verbose)
    ds_t     = filter_time(ds_depth, months_l, years_l, start_date, end_date)

    # triangulation from base ds (expects lon/lat + connectivity)
    tri = build_triangulation(ds)

    outputs: List[str] = []

    # region masks
    def _mask_for_region(scope_kind: str,
                         region_name: Optional[str],
                         spec: Optional[Dict[str, Any]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if scope_kind == "domain":
            return (None, None)
        mask_nodes, mask_elems = build_region_masks(ds_t, (region_name, spec), verbose=verbose)
        return (mask_nodes, mask_elems)

    # absolute z handling, if requested
    def _resolve_var(da: xr.DataArray) -> xr.DataArray:
        if isinstance(depth, (float, np.floating)) and not (-1.0 <= float(depth) <= 0.0):
            return select_da_by_z(da, ds_t, float(depth), verbose=verbose)
        if isinstance(depth, tuple) and len(depth) > 0 and depth[0] == "z_m":
            return select_da_by_z(da, ds_t, float(depth[1]), verbose=verbose)
        if isinstance(depth, dict) and "z_m" in depth:
            return select_da_by_z(da, ds_t, float(depth["z_m"]), verbose=verbose)
        return da

    # scope iterator
    scope_items: List[Tuple[str, Optional[str], Optional[Dict[str, Any]]]]
    if scope == "domain":
        scope_items = [("domain", None, None)]
    else:
        scope_items = [("region", nm, sp) for (nm, sp) in regions]  # type: ignore

    # desired explicit instants (if provided)
    desired_list = _timepoints_to_list(at_time, at_times)
    freq_alias   = _normalize_frequency(frequency)

    for var in variables:
        # eval + depth
        try:
            da0 = eval_group_or_var(ds_t, var, groups)
            da  = _resolve_var(da0)
        except Exception as e:
            _vprint(verbose, f"[animate/maps] Skip '{var}': {e}")
            continue

        # center
        center = "node" if "node" in da.dims else ("nele" if "nele" in da.dims else None)
        if center is None:
            _vprint(verbose, f"[animate/maps] '{var}' has no 'node' or 'nele' dim; skipping.")
            continue

        # Build frames: list of (timestamp, instantaneous DataArray)
        if desired_list:
            frames: List[Tuple[pd.Timestamp, xr.DataArray]] = _choose_instants(da, desired_list, method=time_method)

        elif freq_alias and "time" in da.dims:
            # sample period representatives from available timestamps, then nearest-select
            idx = pd.to_datetime(np.atleast_1d(da["time"].values))
            buckets = (
                pd.Series(np.arange(len(idx)), index=idx)
                .groupby(pd.Grouper(freq=freq_alias))
                .first()
                .dropna()
                .index
            )
            desired_list2 = [pd.Timestamp(t) for t in buckets]
            frames = _choose_instants(da, desired_list2, method=time_method)

        else:
            if "time" in da.dims:
                frames = []
                times = np.atleast_1d(da["time"].values)
                for i in range(len(times)):
                    sub = da.isel(time=i)
                    ts  = pd.to_datetime(np.atleast_1d(sub["time"].values)[0])
                    frames.append((pd.Timestamp(ts), sub))
            else:
                frames = [(pd.Timestamp("NaT"), da)]

        # per-var styling
        cmap_eff    = style_get(var, styles, "cmap", cmap)
        norm_eff    = style_get(var, styles, "norm", None)
        vmin_style  = style_get(var, styles, "vmin", None)
        vmax_style  = style_get(var, styles, "vmax", None)
        shading_eff = style_get(var, styles, "shading", shading)

        for (scope_kind, region_name, spec) in scope_items:
            mask_nodes, mask_elems = _mask_for_region(scope_kind, region_name, spec)

            # color limits: norm > (vmin/vmax via styles) > clim arg > robust across selected frames
            def _masked(arr: np.ndarray) -> np.ndarray:
                if center == "node":
                    return arr if mask_nodes is None else arr[mask_nodes]
                else:
                    return arr if mask_elems is None else arr[mask_elems]

            if norm_eff is not None:
                clim_eff = None
            else:
                if vmin_style is not None and vmax_style is not None:
                    clim_eff = (vmin_style, vmax_style)
                elif clim is not None:
                    clim_eff = clim
                else:
                    vals_accum: List[float] = []
                    for (_ts, sub) in frames:
                        arr = np.asarray(sub.values).ravel()
                        vin = _masked(arr)
                        vin = vin[np.isfinite(vin)]
                        if vin.size:
                            vals_accum.append(np.nanpercentile(vin, robust_q[0]))
                            vals_accum.append(np.nanpercentile(vin, robust_q[1]))
                    if len(vals_accum) >= 2:
                        clim_eff = (float(np.nanmin(vals_accum)), float(np.nanmax(vals_accum)))
                    else:
                        clim_eff = (0.0, 1.0)

            # figure + initial draw
            fig, ax = plt.subplots(figsize=figsize)
            mesh_handle = [None]
            if grid_on:
                ax.triplot(tri, color="k", lw=0.3, alpha=0.4, zorder=3)
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

            def _vals(sub: xr.DataArray) -> np.ndarray:
                arr = np.asarray(sub.values).ravel().astype(float)
                if center == "node":
                    if mask_nodes is not None:
                        arr = arr.copy(); arr[~mask_nodes] = np.nan
                else:
                    if mask_elems is not None:
                        arr = arr.copy(); arr[~mask_elems] = np.nan
                return arr

            def _draw(ts: pd.Timestamp, sub: xr.DataArray):
                vals = _vals(sub)
                if mesh_handle[0] is not None:
                    try:
                        mesh_handle[0].remove()
                    except Exception:
                        pass
                if center == "node":
                    tpc = ax.tripcolor(tri, np.ma.masked_invalid(vals),
                                       shading=shading_eff, cmap=cmap_eff, norm=norm_eff)
                else:
                    tpc = ax.tripcolor(tri, np.ma.masked_invalid(vals),
                                       shading="flat", cmap=cmap_eff, norm=norm_eff)
                if clim_eff is not None and norm_eff is None:
                    tpc.set_clim(*clim_eff)
                mesh_handle[0] = tpc

                scope_lbl = "Domain" if scope_kind == "domain" else f"Region {region_name}"
                if pd.notnull(ts):
                    ax.set_title(f"{scope_lbl} — {var} ({dtag}, {ts.strftime('%Y-%m-%d %H:%M')})")
                else:
                    ax.set_title(f"{scope_lbl} — {var} ({dtag}, {tlabel})")
                return tpc

            # initial + colorbar
            first_artist = _draw(frames[0][0], frames[0][1])
            cbar = fig.colorbar(first_artist, ax=ax, shrink=0.9, pad=0.02)
            cbar.set_label(var)

            # animator
            def update(i):
                ts, sub = frames[i]
                return _draw(ts, sub)

            ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=interval_ms, blit=False)

            # filename
            scope_str = "Domain" if scope_kind == "domain" else f"Region-{region_name}"
            freq_tag  = f"__Freq{frequency.capitalize()}" if _normalize_frequency(frequency) else ""
            fname = f"{prefix}__Map-{scope_str}__{var}__{dtag}__{tlabel}{freq_tag}__Anim.gif"
            path = os.path.join(outdir, fname)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            ani.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
            plt.close(fig)
            outputs.append(path)
            _vprint(verbose, f"[animate/maps] saved: {path}")

    return outputs

