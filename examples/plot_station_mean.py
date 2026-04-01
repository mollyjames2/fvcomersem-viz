#!/usr/bin/env python3
"""
Examples: station-mean bar plots and composition timeseries
-----------------------------------------------------------

Tests the new station-grouping and station_mean scope features:

  1) plot_bars  — single station group, depth_avg, error over depth+time
  2) plot_bars  — multiple named groups (e.g. inner vs outer shelf), surface
  3) plot_bars  — station groups, depth_avg, faceted by variable
  4) composition_fraction_timeseries — scope='station_mean', surface
  5) composition_fraction_timeseries — scope='station_mean', depth_avg
     (std band reflects spread across stations AND depth layers)
  6) composition_fraction_timeseries — scope='station' (individual panels)
     vs scope='station_mean' (pooled panel) side-by-side for comparison

Edit BASE_DIR / FILE_PATTERN / FIG_DIR and the station coordinates below.

Run:
    pip install -e .
    python examples/plot_station_mean.py
"""

from __future__ import annotations
import os
import sys
import warnings
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from fvcomersemviz.io import load_from_base
from fvcomersemviz.plot import hr, info, kv, bullet, list_files, summarize_files, print_dataset_summary
from fvcomersemviz.utils import out_dir, file_prefix
from fvcomersemviz.plots.bars_box import plot_bars
from fvcomersemviz.plots.composition import composition_fraction_timeseries
from fvcomersemviz.regions import nearest_node_index

# -----------------------------------------------------------------------------
# Project paths  (EDIT THESE)
# -----------------------------------------------------------------------------
BASE_DIR = "/data/proteus1/scratch/yli/project/lake_erie/output_updated_river_var"
FILE_PATTERN = "erie_00??.nc"
FIG_DIR = "/data/proteus1/scratch/moja/projects/Lake_Erie/fviz-plots/"

# -----------------------------------------------------------------------------
# Station definitions  (name, lat, lon)
# -----------------------------------------------------------------------------
# Two groups representing different shelf zones.  Replace coordinates with
# real locations in your model domain.
INNER_SHELF = [
    ("IS1", 41.80, -82.50),
    ("IS2", 41.70, -82.80),
    ("IS3", 41.90, -82.20),
]
OUTER_SHELF = [
    ("OS1", 42.30, -81.50),
    ("OS2", 42.20, -81.80),
    ("OS3", 42.40, -81.20),
]

# Flat list used for scope='station' / scope='station_mean'
ALL_STATIONS = INNER_SHELF + OUTER_SHELF

# For single-group examples
CENTRAL_STATIONS = [
    ("C1", 42.00, -82.00),
    ("C2", 41.95, -82.30),
    ("C3", 42.05, -81.80),
]

# -----------------------------------------------------------------------------
# Variable / biology definitions
# -----------------------------------------------------------------------------
PHYTO_VARS = ["P1_c", "P2_c", "P4_c"]
ZOO_VARS   = ["Z4_c", "Z5_c", "Z6_c"]

COLORS = {
    "P1_c": "#1f77b4",
    "P2_c": "#2ca02c",
    "P4_c": "#9467bd",
    "Z4_c": "#ff7f0e",
    "Z5_c": "#d62728",
    "Z6_c": "#8c564b",
}

DOC_GROUPS = {"DOC": "R1_c + R2_c + R3_c + T1_30d_c + T2_30d_c"}

# -----------------------------------------------------------------------------
# Time windows
# -----------------------------------------------------------------------------
MONTHS_APR_OCT = [4, 5, 6, 7, 8, 9, 10]
MONTHS_JJA     = [6, 7, 8]
YEARS_2018     = [2018]


def plot_station_locations(
    groups: dict[str, list[tuple[str, float, float]]],
    base_dir: str,
    figures_root: str,
    *,
    padding: float = 0.5,
) -> str:
    """
    Plot station locations on a cartopy map and save to figures_root.

    Parameters
    ----------
    groups : dict
        Mapping of group name → list of (name, lat, lon) tuples.
    base_dir : str
        Used to resolve the output directory.
    figures_root : str
        Root figures directory.
    padding : float
        Degrees of lat/lon padding around the station extent.

    Returns
    -------
    str
        Path to the saved figure.
    """
    all_stations = [(n, lat, lon) for slist in groups.values() for n, lat, lon in slist]
    lats = [s[1] for s in all_stations]
    lons = [s[2] for s in all_stations]

    extent = [
        min(lons) - padding, max(lons) + padding,
        min(lats) - padding, max(lats) + padding,
    ]

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    proj = ccrs.PlateCarree()

    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": proj})
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
    ax.add_feature(cfeature.LAKES, facecolor="aliceblue", edgecolor="steelblue", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":", zorder=2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--", color="gray", alpha=0.6)
    gl.top_labels = False
    gl.right_labels = False

    for i, (group_name, slist) in enumerate(groups.items()):
        c = colours[i % len(colours)]
        for j, (name, lat, lon) in enumerate(slist):
            ax.plot(lon, lat, "o", color=c, markersize=8, transform=proj,
                    label=group_name if j == 0 else None, zorder=3)
            ax.text(lon + 0.05, lat + 0.05, name, fontsize=8, transform=proj,
                    color=c, zorder=4)

    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Station locations", fontsize=12)

    odir = out_dir(base_dir, figures_root)
    prefix = file_prefix(base_dir)
    out_path = os.path.join(odir, f"{prefix}_station_locations.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def sanity_check_stations(
    ds: object,
    groups: dict[str, list[tuple[str, float, float]]],
    base_dir: str,
    figures_root: str,
    *,
    warn_km: float = 10.0,
) -> str:
    """
    Print a snap-distance table and save a map showing requested vs snapped positions.

    For each station the nearest mesh node is found; the table reports the
    requested coordinates, the actual node coordinates, and the great-circle
    distance between them.  Rows where the snap distance exceeds ``warn_km``
    are flagged with '!!' — a large distance usually means a coord swap or an
    out-of-domain station.

    The saved map overlays requested positions (hollow circle) and snapped node
    positions (filled cross) connected by a thin line so mismatches are
    immediately visible.

    Parameters
    ----------
    ds : xr.Dataset
        Loaded dataset with 'lat' and 'lon' node coordinates.
    groups : dict
        Mapping of group name → list of (name, lat, lon) tuples.
    base_dir : str
        Used to resolve the output directory.
    figures_root : str
        Root figures directory.
    warn_km : float
        Snap distance (km) above which a row is flagged in the table.

    Returns
    -------
    str
        Path to the saved figure.
    """
    import numpy as np

    lat_arr = np.asarray(ds["lat"].values).ravel()
    lon_arr = np.asarray(ds["lon"].values).ravel()

    R_KM = 6371.0

    rows = []
    for group_name, slist in groups.items():
        for name, req_lat, req_lon in slist:
            idx = nearest_node_index(ds, req_lat, req_lon)
            if idx is None:
                rows.append((group_name, name, req_lat, req_lon, None, None, None))
                continue
            snap_lat = float(lat_arr[idx])
            snap_lon = float(lon_arr[idx])
            # haversine distance
            dlat = np.deg2rad(snap_lat - req_lat)
            dlon = np.deg2rad(snap_lon - req_lon)
            a = np.sin(dlat / 2) ** 2 + np.cos(np.deg2rad(req_lat)) * np.cos(np.deg2rad(snap_lat)) * np.sin(dlon / 2) ** 2
            dist_km = 2 * R_KM * np.arcsin(np.sqrt(a))
            rows.append((group_name, name, req_lat, req_lon, snap_lat, snap_lon, dist_km))

    # --- print table ---
    header = f"{'':2}  {'Group':<14}  {'Station':<8}  {'Req lat':>8}  {'Req lon':>9}  {'Node lat':>9}  {'Node lon':>10}  {'Dist km':>8}"
    print(hr("-"))
    print("Station snap-distance check")
    print(header)
    print("-" * len(header))
    any_warn = False
    for group_name, name, req_lat, req_lon, snap_lat, snap_lon, dist_km in rows:
        if dist_km is None:
            flag = "??"
            row = f"{flag}  {group_name:<14}  {name:<8}  {req_lat:>8.4f}  {req_lon:>9.4f}  {'N/A':>9}  {'N/A':>10}  {'N/A':>8}"
            any_warn = True
        else:
            flag = "!!" if dist_km > warn_km else "  "
            if dist_km > warn_km:
                any_warn = True
            row = f"{flag}  {group_name:<14}  {name:<8}  {req_lat:>8.4f}  {req_lon:>9.4f}  {snap_lat:>9.4f}  {snap_lon:>10.4f}  {dist_km:>8.3f}"
        print(row)
    print("-" * len(header))
    if any_warn:
        print("!! one or more stations have large snap distances — check coordinates or domain extent")
    print(hr("-"))

    # --- map: requested (hollow circle) vs snapped (filled x), connected by line ---
    all_req_lats = [r[2] for r in rows if r[6] is not None]
    all_req_lons = [r[3] for r in rows if r[6] is not None]
    padding = 0.5
    extent = [
        min(all_req_lons) - padding, max(all_req_lons) + padding,
        min(all_req_lats) - padding, max(all_req_lats) + padding,
    ]

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": proj})
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
    ax.add_feature(cfeature.LAKES, facecolor="aliceblue", edgecolor="steelblue", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--", color="gray", alpha=0.6)
    gl.top_labels = False
    gl.right_labels = False

    group_names = list(groups.keys())
    for row in rows:
        group_name, name, req_lat, req_lon, snap_lat, snap_lon, dist_km = row
        c = colours[group_names.index(group_name) % len(colours)]
        if snap_lat is None:
            continue
        # line connecting requested → snapped
        ax.plot([req_lon, snap_lon], [req_lat, snap_lat], "-", color=c, linewidth=0.8,
                alpha=0.6, transform=proj, zorder=3)
        # requested position: hollow circle
        ax.plot(req_lon, req_lat, "o", color=c, markersize=9, markerfacecolor="none",
                markeredgewidth=1.5, transform=proj, zorder=4)
        # snapped node: filled x
        ax.plot(snap_lon, snap_lat, "x", color=c, markersize=7, markeredgewidth=2,
                transform=proj, zorder=4)
        ax.text(req_lon + 0.04, req_lat + 0.04, name, fontsize=7, color=c,
                transform=proj, zorder=5)

    # legend proxies
    from matplotlib.lines import Line2D
    proxies = [
        Line2D([0], [0], marker="o", color="gray", markerfacecolor="none",
               markeredgewidth=1.5, markersize=8, linestyle="none", label="Requested"),
        Line2D([0], [0], marker="x", color="gray", markeredgewidth=2,
               markersize=7, linestyle="none", label="Snapped node"),
    ]
    for i, gname in enumerate(group_names):
        c = colours[i % len(colours)]
        proxies.append(Line2D([0], [0], color=c, linewidth=2, label=gname))
    ax.legend(handles=proxies, loc="lower right", fontsize=8)
    ax.set_title("Station snap check  (○ requested  ✕ nearest node)", fontsize=11)

    odir = out_dir(base_dir, figures_root)
    prefix = file_prefix(base_dir)
    out_path = os.path.join(odir, f"{prefix}_station_snap_check.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    if not os.environ.get("PYTHONWARNINGS"):
        warnings.filterwarnings("default")

    print(hr("="))
    print("Station-mean bar plots and composition timeseries")
    print(hr("="))

    info("Discovering files")
    files = list_files(BASE_DIR, FILE_PATTERN)
    summarize_files(files)
    if not files:
        print("No files found; abort.")
        return 1

    info("Loading dataset")
    ds = load_from_base(BASE_DIR, FILE_PATTERN)
    print_dataset_summary(ds)

    kv("Figure folder", out_dir(BASE_DIR, FIG_DIR))
    kv("Filename prefix", file_prefix(BASE_DIR))

    # =========================================================================
    # Station snap-distance check
    # =========================================================================
    info("Checking station snapping")
    station_groups_all = {
        "Inner shelf": INNER_SHELF,
        "Outer shelf": OUTER_SHELF,
        "Central": CENTRAL_STATIONS,
    }
    snap_path = sanity_check_stations(ds, station_groups_all, base_dir=BASE_DIR, figures_root=FIG_DIR)
    kv("Saved (snap check map)", snap_path)

    # =========================================================================
    # Station location map — quick sanity check
    # =========================================================================
    info("Plotting station locations")
    loc_path = plot_station_locations(
        {
            "Inner shelf": INNER_SHELF,
            "Outer shelf": OUTER_SHELF,
            "Central": CENTRAL_STATIONS,
        },
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
    )
    kv("Saved (station locations)", loc_path)

    # =========================================================================
    # Example 1: Single station group, depth_avg, error over depth + time
    # -------------------------------------------------------------------------
    # All stations in CENTRAL_STATIONS are pooled into one bar labelled
    # "Central stations".  Error bars = SD across all (time × depth) samples.
    # =========================================================================
    info("Example 1: Single group — depth_avg, error over depth + time")
    bullet("station_groups pools all stations into one mean bar per variable")
    plot_bars(
        ds,
        variables=PHYTO_VARS,
        station_groups={"Central stations": CENTRAL_STATIONS},
        depth="depth_avg",
        months=MONTHS_APR_OCT,
        years=YEARS_2018,
        x_by="station",
        hue_by="variable",
        error="sd",
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        title="Phyto: central-station mean (depth+time avg, ±1SD)",
        ylabel="Concentration",
        verbose=True,
    )

    # =========================================================================
    # Example 2: Two named groups — surface, error over time
    # -------------------------------------------------------------------------
    # One bar per group per variable; error reflects temporal spread at surface.
    # =========================================================================
    info("Example 2: Two shelf zones — surface, hue by variable")
    plot_bars(
        ds,
        variables=PHYTO_VARS + ZOO_VARS,
        station_groups={
            "Inner shelf": INNER_SHELF,
            "Outer shelf": OUTER_SHELF,
        },
        depth="surface",
        months=MONTHS_JJA,
        years=YEARS_2018,
        x_by="station",       # one bar-group per shelf zone
        hue_by="variable",
        error="ci95",
        figsize=(14, 6),
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        title="Phyto+Zoo surface: inner vs outer shelf (JJA, ±95% CI)",
        ylabel="Concentration",
    )

    # =========================================================================
    # Example 3: Station groups, depth_avg, faceted by variable
    # -------------------------------------------------------------------------
    # One panel per variable; x = shelf zone.  Error bars capture depth+time
    # variance (because depth='depth_avg' keeps siglay for variance).
    # =========================================================================
    info("Example 3: Two shelf zones — depth_avg, facet by variable")
    plot_bars(
        ds,
        variables=PHYTO_VARS,
        station_groups={
            "Inner shelf": INNER_SHELF,
            "Outer shelf": OUTER_SHELF,
        },
        depth="depth_avg",
        months=MONTHS_APR_OCT,
        years=YEARS_2018,
        facet_by="variable",
        x_by="station",
        hue_by=None,
        error="sd",
        figsize=(10, 10),
        ncols=1,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        title="Phyto depth-avg: inner vs outer shelf (±1SD over depth+time)",
        ylabel="Concentration",
    )

    # =========================================================================
    # Example 4: composition_fraction_timeseries — scope='station_mean', surface
    # -------------------------------------------------------------------------
    # Single panel showing the mean fraction timeseries across all stations;
    # ±1σ band = spread across the station ensemble at each timestep.
    # =========================================================================
    info("Example 4: composition TS — scope='station_mean', surface")
    bullet("Single panel; std band = spread across station ensemble")
    phy_path, zoo_path = composition_fraction_timeseries(
        ds,
        phyto_vars=PHYTO_VARS,
        zoo_vars=ZOO_VARS,
        scope="station_mean",
        stations=ALL_STATIONS,
        depth="surface",
        months=MONTHS_APR_OCT,
        years=YEARS_2018,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        colors=COLORS,
        show_std_band=True,
        linewidth=2.0,
        verbose=True,
    )
    if phy_path:
        kv("Saved (Phyto TS, station_mean, surface)", phy_path)
    if zoo_path:
        kv("Saved (Zoo TS, station_mean, surface)", zoo_path)

    # =========================================================================
    # Example 5: composition_fraction_timeseries — scope='station_mean', depth_avg
    # -------------------------------------------------------------------------
    # ±1σ band now reflects spread across BOTH stations AND depth layers.
    # =========================================================================
    info("Example 5: composition TS — scope='station_mean', depth_avg")
    bullet("std band = spread across station ensemble × depth layers")
    phy_path, zoo_path = composition_fraction_timeseries(
        ds,
        phyto_vars=PHYTO_VARS,
        zoo_vars=ZOO_VARS,
        scope="station_mean",
        stations=ALL_STATIONS,
        depth="depth_avg",
        months=MONTHS_APR_OCT,
        years=YEARS_2018,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        colors=COLORS,
        show_std_band=True,
        linewidth=2.0,
    )
    if phy_path:
        kv("Saved (Phyto TS, station_mean, depth_avg)", phy_path)
    if zoo_path:
        kv("Saved (Zoo TS, station_mean, depth_avg)", zoo_path)

    # =========================================================================
    # Example 6: scope='station' (individual panels) for comparison
    # -------------------------------------------------------------------------
    # Useful sanity check: run both scope='station' and scope='station_mean'
    # on the same station list and compare the individual vs pooled result.
    # =========================================================================
    info("Example 6: composition TS — scope='station' (individual panels, for comparison)")
    bullet("Run this alongside Example 5 to compare individual vs pooled")
    phy_path, zoo_path = composition_fraction_timeseries(
        ds,
        phyto_vars=PHYTO_VARS,
        zoo_vars=ZOO_VARS,
        scope="station",
        stations=ALL_STATIONS,
        depth="surface",
        months=MONTHS_APR_OCT,
        years=YEARS_2018,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        colors=COLORS,
        show_std_band=False,   # single point — band is zero anyway
        linewidth=1.5,
    )
    if phy_path:
        kv("Saved (Phyto TS, individual stations)", phy_path)
    if zoo_path:
        kv("Saved (Zoo TS, individual stations)", zoo_path)

    print(hr("="))
    print("Done")
    print(hr("="))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
