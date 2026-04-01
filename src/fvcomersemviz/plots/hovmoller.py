# fvcomersemviz/plots/hovmoller.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from ..io import filter_time, eval_group_or_var, ensure_z_from_sigma
from ..utils import (
    out_dir,
    file_prefix,
    robust_clims,
    style_get,
    resolve_cmap,
    build_time_window_label,
    nearest_index_for_dim,
)


# ---------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------
def _vprint(verbose: bool, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


# vectorized depth interpolation helper
def _interp_profile_to_z(zvec: np.ndarray, avals: np.ndarray, z_levels: np.ndarray) -> np.ndarray:
    """
    Interpolate a single (siglay) profile to regular z_levels (ascending).
    zvec likely negative up to 0; ensure sorted for np.interp. NaNs handled by masking.
    """
    z = np.asarray(zvec, dtype=float)
    a = np.asarray(avals, dtype=float)
    m = np.isfinite(z) & np.isfinite(a)
    if m.sum() < 2:
        return np.full_like(z_levels, np.nan, dtype=float)
    z_m = z[m]
    a_m = a[m]
    order = np.argsort(z_m)  # ensure ascending z for np.interp
    z_s = z_m[order]
    a_s = a_m[order]
    zmin, zmax = z_s[0], z_s[-1]  # clamp interpolation range
    zq = np.clip(z_levels, zmin, zmax)
    return np.interp(zq, z_s, a_s)


def _siglev_z_from_siglay_z(z_siglay: xr.DataArray, ds: xr.Dataset) -> xr.DataArray:
    """
    Derive depth (m) at siglev interfaces by linear interpolation in sigma space.

    Since z is linear in sigma (z = zeta + (h + zeta) * sigma), we can map from
    siglay midpoints to siglev interfaces by interpolating along the sigma axis.

    Parameters
    ----------
    z_siglay : xr.DataArray
        Depth at siglay midpoints, shape (time, siglay).
    ds : xr.Dataset
        Source dataset containing 1-D or 2-D ``siglay`` and ``siglev`` coordinate arrays.

    Returns
    -------
    xr.DataArray
        Depth at siglev interfaces, shape (time, siglev).
    """
    # Representative 1-D sigma arrays (use first node column if 2-D)
    sig_lay = np.asarray(ds["siglay"]).reshape(ds.dims["siglay"], -1)[:, 0]
    sig_lev = np.asarray(ds["siglev"]).reshape(ds.dims["siglev"], -1)[:, 0]

    z_vals = z_siglay.values  # (time, siglay)
    # sigma goes surface (~0) → bottom (~-1); negate to get ascending for np.interp
    order = np.argsort(-sig_lay)

    def _interp_row(col: np.ndarray) -> np.ndarray:
        return np.interp(-sig_lev, -sig_lay[order], col[order])

    result = np.apply_along_axis(_interp_row, axis=1, arr=z_vals)
    return xr.DataArray(
        result,
        dims=["time", "siglev"],
        coords={"time": z_siglay["time"], "siglev": ("siglev", sig_lev)},
    )


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def station_hovmoller(
    ds: xr.Dataset,
    variables: List[str],
    stations: List[Tuple[str, float, float]],
    *,
    axis: str = "z",
    z_levels: Optional[np.ndarray] = None,
    months: Optional[List[int]] = None,
    years: Optional[List[int]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    base_dir: str,
    figures_root: str,
    groups: Optional[Dict[str, Any]] = None,
    cmap: str = "viridis",
    title: Optional[Dict[str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: int = 150,
    figsize: tuple = (9, 5),
    verbose: bool = True,
    styles: Optional[Dict[str, Dict[str, Any]]] = None,
    average_by: Optional[str] = None,
) -> None:
    """
    Plot Hovmöller diagrams (time vs. vertical coordinate) for one or more variables
    at one or more **stations** (nearest model column to each (lat, lon)).
    Saves one PNG per (station × variable × axis-mode).

    Two vertical axis modes are supported:

    - ``axis='sigma'``: y-axis is the model's sigma coordinate (``siglay``).
      Data are shown on native layers (no vertical interpolation).
    - ``axis='z'``: y-axis is **absolute depth** in meters (negative downward).
      Each station column is **interpolated** from sigma layers onto a shared set of
      absolute depth levels ``z_levels``. If ``z_levels`` is not provided, a default
      is auto-built from the column's minimum depth to 0 m (surface).

    Workflow
    --------
    1. Apply the time filter with ``filter_time`` (months/years/date bounds) to obtain
       a consistent dataset for station lookup and plotting.
    2. For each station, find the nearest index for ``'node'`` and/or ``'nele'`` using
       ``nearest_index_for_dim``. The data array is sliced on whichever of those dims it has.
    3. For each variable, resolve the DataArray via ``eval_group_or_var(ds_t, var, groups)``.
    4. Validate required dims: must have ``'time'`` and ``'siglay'``.
    5. Build the panel:
       - **sigma mode**: plot ``pcolormesh(time, siglay, values)``.
       - **z mode**: create/obtain per-station z profiles with ``ensure_z_from_sigma``,
         interpolate (time, siglay) → (time, z) using ``_interp_profile_to_z`` via
         ``xr.apply_ufunc``, then plot ``pcolormesh(time, z, values)``.
    6. Determine colour scaling with the following priority:
         ``styles[var]['norm']`` > (``styles[var]['vmin']``, ``styles[var]['vmax']``)
         > (``vmin``, ``vmax``) > robust percentiles of the plotted values.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset containing variables to plot and vertical sigma information.
    variables : list of str
        Variable or expression names. Each is resolved through ``groups`` if provided.
    stations : list of (str, float, float)
        Stations as ``(name, lat, lon)``. The nearest model column (node/element) is used.
    axis : {"sigma", "z"}, default "z"
        Vertical axis mode:
          - "sigma": native sigma layers (no interpolation).
          - "z": absolute depth (m, negative downward) with interpolation onto ``z_levels``.
    z_levels : array-like of float, optional
        Absolute depth levels (negative downward) used when ``axis="z"``.
        If ``None``, levels are auto-generated from the station column’s minimum depth to 0 m.
    months, years : list[int], optional
        Calendar-based time filters (1–12 months; integer years). Can be combined.
    start_date, end_date : str, optional
        Inclusive date bounds "YYYY-MM-DD".
    base_dir : str
        Run root; used by ``file_prefix(base_dir)`` to construct output filenames.
    figures_root : str
        Root directory for figure outputs (created if missing).
    groups : dict, optional
        Global alias/expression mapping for variable resolution by ``eval_group_or_var``.
    cmap : str, default "viridis"
        Colormap for the pcolormesh (can be overridden per-variable via ``styles``).
    vmin, vmax : float, optional
        Global colour limits used when a per-variable norm or explicit vmin/vmax are not supplied.
    dpi : int, default 150
        PNG resolution.
    figsize : tuple, default (9, 5)
        Figure size in inches.
    verbose : bool, default True
        Print progress and skip reasons.
    styles : dict, optional
        Per-variable style overrides. For key ``var``, recognized entries:
          - ``"cmap"``: str or Colormap
          - ``"norm"``: matplotlib.colors.Normalize (takes precedence over vmin/vmax)
          - ``"vmin"``: float
          - ``"vmax"``: float
    average_by : str, optional
        Temporal averaging period applied before plotting. Resamples the
        time-filtered dataset to period means via ``xr.Dataset.resample().mean()``.
        Accepted values: ``"hour"``, ``"day"``, ``"week"``, ``"month"``,
        ``"year"`` (and common variants such as ``"daily"``, ``"monthly"``).
        Default ``None`` (no averaging).

    Returns
    -------
    None
        Saves a PNG for each (station × variable × axis-mode). Filenames encode the
        scope (Station-<name>), variable, axis ("sigma" or "z"), and the time window label.

    Notes
    -----
    - If a variable lacks ``'time'`` or both ``'siglay'`` and ``'siglev'``, it is skipped.
      Variables on ``siglev`` (e.g. ``km``, ``kh``) are handled alongside ``siglay`` variables.
    - If neither ``'node'`` nor ``'nele'`` can be located for a station, that station-variable
      pair is skipped.
    - In ``axis='z'`` mode, vertical coordinates are obtained via ``ensure_z_from_sigma``:
      picks ``'z'`` (node-centered) or ``'z_nele'`` (element-centered), aligned to the same
      spatial index used for the variable slice, before interpolation. For ``siglev`` variables,
      the siglay-based z column is interpolated to siglev interfaces via ``_siglev_z_from_siglay_z``.
    - Robust colour limits are computed only when no norm or explicit vmin/vmax are present.
    """
    if axis not in ("sigma", "z"):
        raise ValueError("axis must be 'sigma' or 'z'.")

    outdir = out_dir(base_dir, figures_root)
    prefix = file_prefix(base_dir)
    label = build_time_window_label(months, years, start_date, end_date)

    # time filter first; use this dataset consistently for nearest-index & z
    ds_t = filter_time(ds, months=months, years=years, start_date=start_date, end_date=end_date, average_by=average_by)

    for name, lat, lon in stations:
        # compute nearest indices ONCE per station
        try:
            node_idx = nearest_index_for_dim(ds_t, lat, lon, "node")
        except Exception:
            node_idx = None
        try:
            nele_idx = nearest_index_for_dim(ds_t, lat, lon, "nele")
        except Exception:
            nele_idx = None

        # Pre-compute vertical coordinate columns once per station (not per variable).
        # For node vars: isel first so z is (time, siglay) not (time, siglay, all_nodes).
        z_node_col = None
        z_nele_col = None
        if axis == "z":
            if node_idx is not None:
                try:
                    _ds_node = ds_t.isel(node=node_idx)
                    _ds_zn = ensure_z_from_sigma(
                        _ds_node, compute_elements=False, verbose=verbose
                    )
                    z_node_col = _ds_zn.get("z")  # (time, siglay)
                except Exception as e:
                    _vprint(verbose, f"[hovmoller:{name}] node z pre-compute failed: {e}")
            if nele_idx is not None:
                try:
                    _ds_z_full = ensure_z_from_sigma(
                        ds_t, compute_elements=True, verbose=verbose
                    )
                    if "z_nele" in _ds_z_full:
                        z_nele_col = _ds_z_full["z_nele"].isel(nele=nele_idx)  # (time, siglay)
                    # fallback node col if the efficient path above failed
                    if z_node_col is None and "z" in _ds_z_full:
                        z_node_col = _ds_z_full["z"].isel(node=0)
                except Exception as e:
                    _vprint(verbose, f"[hovmoller:{name}] nele z pre-compute failed: {e}")

        for var in variables:
            _vprint(verbose, f"[hovmoller:{name}] {var} ({axis})")

            # resolve variable/group
            try:
                da = eval_group_or_var(ds_t, var, groups)
            except Exception as e:
                _vprint(verbose, f"[hovmoller:{name}] skip '{var}': {e}")
                continue

            # which spatial dim?
            pick_dim = "node" if "node" in da.dims else ("nele" if "nele" in da.dims else None)
            if pick_dim is None:
                _vprint(
                    verbose,
                    f"[hovmoller:{name}] '{var}' has no 'node'/'nele'; proceeding without spatial slice.",
                )
                idx = None
            else:
                idx = node_idx if pick_dim == "node" else nele_idx
                if idx is None:
                    _vprint(
                        verbose,
                        f"[hovmoller:{name}] no {pick_dim} coordinates available; skipping.",
                    )
                    continue
                da = da.isel({pick_dim: idx})

            if "time" not in da.dims:
                _vprint(verbose, f"[hovmoller:{name}] '{var}' has no time dim; skip.")
                continue
            if "siglay" in da.dims:
                vert_dim = "siglay"
            elif "siglev" in da.dims:
                vert_dim = "siglev"
            else:
                _vprint(verbose, f"[hovmoller:{name}] '{var}' has no 'siglay'/'siglev' dim; skip.")
                continue

            # (time, vert_dim)
            A = da.transpose("time", vert_dim)
            t = pd.to_datetime(A["time"].values)

            # per-variable style (fallbacks)
            cmap_eff = resolve_cmap(style_get(var, styles, "cmap", cmap))
            norm_eff = style_get(var, styles, "norm", None)
            vmin_eff = style_get(var, styles, "vmin", vmin)
            vmax_eff = style_get(var, styles, "vmax", vmax)

            if axis == "sigma":
                if vert_dim in A.coords and getattr(A[vert_dim], "ndim", 0) == 1:
                    y = A[vert_dim].values
                elif vert_dim in ds_t and getattr(ds_t[vert_dim], "ndim", 0) == 1:
                    y = ds_t[vert_dim].values
                else:
                    y = np.arange(A.sizes[vert_dim])

                # decide colour limits (norm wins; else vmin/vmax; else robust)
                vvmin, vvmax = vmin_eff, vmax_eff
                if norm_eff is None and (vvmin is None or vvmax is None):
                    lo, hi = robust_clims(A.values)
                    vvmin = lo if vvmin is None else vvmin
                    vvmax = hi if vvmax is None else vvmax

                fig, ax = plt.subplots(figsize=figsize)
                pc = ax.pcolormesh(
                    t,
                    y,
                    A.values.T,
                    shading="nearest",
                    cmap=cmap_eff,
                    norm=norm_eff,
                    vmin=None if norm_eff is not None else vvmin,
                    vmax=None if norm_eff is not None else vvmax,
                )
                try:
                    ax.set_title(f"{title[var]}") 
                except:        
                    ax.set_title(f"{var} — Hovmöller at {name} (sigma, {label})")
             #   ax.set_title(f"{var} — Hovmöller at {name} (sigma, {label})")
                ax.set_xlabel("Time")
                ax.set_ylabel("sigma")
                ax.set_ylim(np.nanmin(y), np.nanmax(y))
                cb = fig.colorbar(pc, ax=ax)
                cb.set_label(var)
                fname = f"{prefix}__Hovmoller-Station-{name}__{var}__sigma__{label}.png"
                fig.savefig(os.path.join(outdir, fname), dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                _vprint(verbose, f"[hovmoller:{name}] saved {fname}")

            else:  # axis == ‘z’
                # Use pre-computed z columns (built once per station, before this loop)
                if pick_dim == "node" and z_node_col is not None:
                    z_col = z_node_col
                elif pick_dim == "nele" and z_nele_col is not None:
                    z_col = z_nele_col
                elif z_node_col is not None:
                    z_col = z_node_col  # fallback (e.g. pick_dim is None)
                elif z_nele_col is not None:
                    z_col = z_nele_col  # fallback
                else:
                    _vprint(
                        verbose,
                        f"[hovmoller:{name}] cannot build vertical coords; skipping.",
                    )
                    continue

                # For siglev variables (e.g. km, kh), interpolate the siglay-based z
                # column to siglev interfaces so both arrays share the same vertical dim.
                if vert_dim == "siglev":
                    try:
                        z_col_v = _siglev_z_from_siglay_z(z_col, ds_t)
                    except Exception as e:
                        _vprint(
                            verbose,
                            f"[hovmoller:{name}] siglev z derivation failed for ‘{var}’: {e}; skip.",
                        )
                        continue
                else:
                    z_col_v = z_col  # already on siglay

                # auto z grid if not supplied
                if z_levels is None:
                    zmin = float(np.nanmin(z_col_v.values))
                    zlev = np.linspace(zmin, 0.0, 60)
                else:
                    zlev = np.asarray(z_levels, dtype=float)
                zlev = np.sort(zlev)

                # interpolate (time, vert_dim) -> (time, z)
                hov = xr.apply_ufunc(
                    _interp_profile_to_z,
                    z_col_v,
                    A,  # both (time, vert_dim)
                    input_core_dims=[[vert_dim], [vert_dim]],
                    output_core_dims=[["z"]],
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[float],
                    dask_gufunc_kwargs={"output_sizes": {"z": zlev.size}, "allow_rechunk": True},
                    kwargs={"z_levels": zlev},
                ).assign_coords(z=("z", zlev))

                # decide colour limits (norm wins; else vmin/vmax; else robust)
                vvmin, vvmax = vmin_eff, vmax_eff
                if norm_eff is None and (vvmin is None or vvmax is None):
                    lo, hi = robust_clims(hov.values)
                    vvmin = lo if vvmin is None else vvmin
                    vvmax = hi if vvmax is None else vvmax

                fig, ax = plt.subplots(figsize=figsize)
                pc = ax.pcolormesh(
                    t,
                    zlev,
                    hov.values.T,
                    shading="nearest",
                    cmap=cmap_eff,
                    norm=norm_eff,
                    vmin=None if norm_eff is not None else vvmin,
                    vmax=None if norm_eff is not None else vvmax,
                )
                try:
                    ax.set_title(f"{title[var]}") 
                except:        
                    ax.set_title(f"{var} — Hovmöller at {name} (z, {label})")
                #ax.set_title(f"{var} — Hovmöller at {name} (z, {label})")
                ax.set_xlabel("Time")
                ax.set_ylabel("Depth (m)")
                ax.set_ylim(np.nanmin(zlev), np.nanmax(zlev))  # negative at bottom, 0 at top
                cb = fig.colorbar(pc, ax=ax)
                cb.set_label(var)
                fname = f"{prefix}__Hovmoller-Station-{name}__{var}__z__{label}.png"
                fig.savefig(os.path.join(outdir, fname), dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                _vprint(verbose, f"[hovmoller:{name}] saved {fname}")
