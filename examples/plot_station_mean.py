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

from fvcomersemviz.io import load_from_base
from fvcomersemviz.plot import hr, info, kv, bullet, list_files, summarize_files, print_dataset_summary
from fvcomersemviz.utils import out_dir, file_prefix
from fvcomersemviz.plots.bars_box import plot_bars
from fvcomersemviz.plots.composition import composition_fraction_timeseries

# -----------------------------------------------------------------------------
# Project paths  (EDIT THESE)
# -----------------------------------------------------------------------------
BASE_DIR = "/data/proteus1/scratch/yli/project/lake_erie/output_updated_river_var"
FILE_PATTERN = "erie_00??.nc"
FIG_DIR = "/data/proteus1/scratch/moja/projects/Lake_Erie/fviz-plots/"

# -----------------------------------------------------------------------------
# Station definitions  (name, lon, lat)
# -----------------------------------------------------------------------------
# Two groups representing different shelf zones.  Replace coordinates with
# real locations in your model domain.
INNER_SHELF = [
    ("IS1", -82.50, 41.80),
    ("IS2", -82.80, 41.70),
    ("IS3", -82.20, 41.90),
]
OUTER_SHELF = [
    ("OS1", -81.50, 42.30),
    ("OS2", -81.80, 42.20),
    ("OS3", -81.20, 42.40),
]

# Flat list used for scope='station' / scope='station_mean'
ALL_STATIONS = INNER_SHELF + OUTER_SHELF

# For single-group examples
CENTRAL_STATIONS = [
    ("C1", -82.00, 42.00),
    ("C2", -82.30, 41.95),
    ("C3", -81.80, 42.05),
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
