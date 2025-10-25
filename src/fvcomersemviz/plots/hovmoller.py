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
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: int = 150,
    figsize: tuple = (9, 5),
    verbose: bool = True,
    styles: Optional[Dict[str, Dict[str, Any]]] = None,
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

    Returns
    -------
    None
        Saves a PNG for each (station × variable × axis-mode). Filenames encode the
        scope (Station-<name>), variable, axis ("sigma" or "z"), and the time window label.

    Notes
    -----
    - If a variable lacks either ``'time'`` or ``'siglay'``, it is skipped.
    - If neither ``'node'`` nor ``'nele'`` can be located for a station, that station-variable
      pair is skipped.
    - In ``axis='z'`` mode, vertical coordinates are obtained via ``ensure_z_from_sigma``:
      picks ``'z'`` (node-centered) or ``'z_nele'`` (element-centered), aligned to the same
      spatial index used for the variable slice, before interpolation.
    - Robust colour limits are computed only when no norm or explicit vmin/vmax are present.
    """
    if axis not in ("sigma", "z"):
        raise ValueError("axis must be 'sigma' or 'z'.")

    outdir = out_dir(base_dir, figures_root)
    prefix = file_prefix(base_dir)
    label = build_time_window_label(months, years, start_date, end_date)

    # time filter first; use this dataset consistently for nearest-index & z
    ds_t = filter_time(ds, months=months, years=years, start_date=start_date, end_date=end_date)

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
            if "siglay" not in da.dims:
                _vprint(verbose, f"[hovmoller:{name}] '{var}' has no 'siglay' dim; skip.")
                continue

            # (time, siglay)
            A = da.transpose("time", "siglay")
            t = pd.to_datetime(A["time"].values)

            # per-variable style (fallbacks)
            cmap_eff = style_get(var, styles, "cmap", cmap)
            norm_eff = style_get(var, styles, "norm", None)
            vmin_eff = style_get(var, styles, "vmin", vmin)
            vmax_eff = style_get(var, styles, "vmax", vmax)

            # --- SIGMA AXIS ---
            if axis == "sigma":
                if "siglay" in A.coords and getattr(A["siglay"], "ndim", 0) == 1:
                    y = A["siglay"].values
                elif "siglay" in ds_t and getattr(ds_t["siglay"], "ndim", 0) == 1:
                    y = ds_t["siglay"].values
                else:
                    y = np.arange(A.sizes["siglay"])

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
                ax.set_title(f"{var} — Hovmöller at {name} (sigma, {label})")
                ax.set_xlabel("Time")
                ax.set_ylabel("sigma")
                ax.set_ylim(np.nanmin(y), np.nanmax(y))
                cb = fig.colorbar(pc, ax=ax)
                cb.set_label(var)
                fname = f"{prefix}__Hovmoller-Station-{name}__{var}__sigma__{label}.png"
                fig.savefig(os.path.join(outdir, fname), dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                _vprint(verbose, f"[hovmoller:{name}] saved {fname}")

            # --- ABSOLUTE Z AXIS ---
            else:  # axis == 'z'
                ds_z = ensure_z_from_sigma(ds_t, verbose=verbose)
                if "z" not in ds_z and "z_nele" not in ds_z:
                    _vprint(
                        verbose,
                        f"[hovmoller:{name}] cannot build vertical coords; skipping.",
                    )
                    continue

                # choose column’s z profile aligned with the same index we sliced above
                if pick_dim == "node" and "z" in ds_z and idx is not None:
                    z_col = ds_z["z"].isel(node=idx)  # (time, siglay)
                elif pick_dim == "nele" and "z_nele" in ds_z and idx is not None:
                    z_col = ds_z["z_nele"].isel(nele=idx)  # (time, siglay)
                else:
                    # fallback: first node/element if dims are missing
                    z_col = ds_z["z"].isel(node=0) if "z" in ds_z else ds_z["z_nele"].isel(nele=0)

                # auto z grid if not supplied
                if z_levels is None:
                    zmin = float(np.nanmin(z_col.values))
                    zlev = np.linspace(zmin, 0.0, 60)
                else:
                    zlev = np.asarray(z_levels, dtype=float)
                zlev = np.sort(zlev)

                # interpolate (time, siglay) -> (time, z)
                hov = xr.apply_ufunc(
                    _interp_profile_to_z,
                    z_col,
                    A,  # both (time, siglay)
                    input_core_dims=[["siglay"], ["siglay"]],
                    output_core_dims=[["z"]],
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[float],
                    dask_gufunc_kwargs={"output_sizes": {"z": zlev.size}},
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
                ax.set_title(f"{var} — Hovmöller at {name} (z, {label})")
                ax.set_xlabel("Time")
                ax.set_ylabel("Depth (m)")
                ax.set_ylim(np.nanmin(zlev), np.nanmax(zlev))  # negative at bottom, 0 at top
                cb = fig.colorbar(pc, ax=ax)
                cb.set_label(var)
                fname = f"{prefix}__Hovmoller-Station-{name}__{var}__z__{label}.png"
                fig.savefig(os.path.join(outdir, fname), dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                _vprint(verbose, f"[hovmoller:{name}] saved {fname}")
