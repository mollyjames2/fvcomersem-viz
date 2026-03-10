#!/usr/bin/env python3
"""
timeseries_examples.py
===============================

A narrated demo of fvcomersemviz that:
  - Loads files via BASE_DIR + FILE_PATTERN
  - Explains where figures are saved: FIG_DIR/<basename(BASE_DIR)>/
  - Shows compound/group variables
  - Produces domain / station / region time series
  - Prints a lot so newcomers can follow

Run:
  pip install -e .
  python examples/timeseries_examples.py
"""

from __future__ import annotations
import os
import sys
import time
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib

matplotlib.use("Agg", force=True)  # must be before any pyplot import


# Optional version printouts only
try:
    import geopandas as gpd  # noqa: F401

    HAS_GPD = True
except Exception:
    HAS_GPD = False

try:
    import shapely  # noqa: F401

    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False

try:
    import dask  # noqa: F401

    HAS_DASK = True
except Exception:
    HAS_DASK = False

# Package imports
from fvcomersemviz.io import load_from_base
from fvcomersemviz.utils import out_dir, file_prefix
from fvcomersemviz.plots.timeseries import (
    domain_mean_timeseries,
    station_timeseries,
    region_timeseries,
    domain_three_panel,
    station_three_panel,
    region_three_panel,
)
from fvcomersemviz.plot import (
    hr,
    info,
    bullet,
    kv,
    try_register_progress_bar,
    list_files,
    summarize_files,
    plot_call,
    print_dataset_summary,
    ensure_paths_exist,
    sample_output_listing,
)

# xarray preferences
xr.set_options(use_new_combine_kwarg_defaults=True)

# -----------------------------------------------------------------------------
# User inputs (EDIT FOR YOUR PROJECT)
# -----------------------------------------------------------------------------
BASE_DIR = "/data/proteus1/scratch/yli/project/lake_erie/output_updated_river_var"
FILE_PATTERN = "erie_00??.nc"
FIG_DIR = "/data/proteus1/scratch/moja/projects/Lake_Erie/fviz-plots/"

# -----------------------------
# Variable groups / composites
# -----------------------------
GROUPS = {
    "DOC": "R1_c + R2_c + R3_c + T1_30d_c + T2_30d_c",
    "phyto": ["P1_c", "P2_c", "P4_c", "P5_c"],
    "zoo": ["Z4_c", "Z5_c", "Z6_c"],
    "chl": "P1_Chl + P2_Chl + P4_Chl + P5_Chl",
}

PLOT_STYLES = {
    "temp": {"line_color": "lightblue"},
    "DOC": {"line_color": "blue"},
    "chl": {"line_color": "lightgreen"},
    "phyto": {"line_color": "darkgreen"},
    "zoo": {"line_color": "purple"},
}

# -----------------------------
# Station list (nearest-node)
# -----------------------------
STATIONS = [
    ("WE12", 41.90, -83.10),
    ("WE13", 41.80, -83.20),
]

# -----------------------------
# Regions (polygon masks)
# -----------------------------
REGIONS = [
    ("Central", {"shapefile": "../data/shapefiles/central_basin_single.shp"}),
    ("East", {"shapefile": "../data/shapefiles/east_basin_single.shp"}),
    ("West", {"shapefile": "../data/shapefiles/west_basin_single.shp"}),
]

# -----------------------------
# Example time windows
# -----------------------------
MONTHS_EXAMPLE = [4, 5, 6, 7, 8, 9, 10]  # Apr-Oct
YEARS_EXAMPLE = [2018]
DATE_RANGE = ("2018-04-01", "2018-10-31")

# -----------------------------
# Dask progress bar toggle
# -----------------------------
SHOW_PROGRESS = False


def main():
    start_ts = time.time()
    print(hr("="))
    print("fvcomersemviz: Time Series Runner")
    print(hr("="))
    kv("Python", sys.version.split()[0])
    kv("xarray", xr.__version__)
    kv("numpy", np.__version__)
    kv("pandas", pd.__version__)
    kv("geopandas", gpd.__version__ if HAS_GPD else "not installed")
    kv("shapely", shapely.__version__ if HAS_SHAPELY else "not installed")
    kv("dask", dask.__version__ if HAS_DASK else "not installed")
    print()

    # 0) Inputs
    info("0) Inputs")
    kv("BASE_DIR", BASE_DIR)
    kv("FILE_PATTERN", FILE_PATTERN)
    kv("FIG_DIR", FIG_DIR)

    bullet("\nThis script will:")
    bullet("- List files matching BASE_DIR/FILE_PATTERN.")
    bullet("- Load them into a single Dataset (lazy if Dask is available).")
    bullet("- Summarize key coordinates (time, lon/lat, connectivity).")
    bullet("- Produce example plots (domain, station, region).")

    try_register_progress_bar(SHOW_PROGRESS)

    bullet("\nGroup definitions:")
    for k, v in GROUPS.items():
        bullet(f"• {k}: {v}")

    bullet("\nStations (name, lat, lon):")
    for s in STATIONS:
        bullet(f"• {s}")

    bullet("\nRegions provided:")
    for name, spec in REGIONS:
        bullet(f"• {name}: {spec}")
    ensure_paths_exist(REGIONS)

    # Discover files
    info(" Discovering files")
    files = list_files(BASE_DIR, FILE_PATTERN)
    summarize_files(files)
    if not files:
        print("\nNo files found. Exiting.")
        sys.exit(2)

    # Load dataset
    info(" Loading dataset (this may be lazy if Dask is available)")
    ds = load_from_base(BASE_DIR, FILE_PATTERN)
    bullet("Dataset loaded. Summary:")
    print_dataset_summary(ds)

    # Where figures will go / filename prefix
    out_folder = out_dir(BASE_DIR, FIG_DIR)
    prefix = file_prefix(BASE_DIR)
    kv("Figure folder", out_folder)
    kv("Filename prefix", prefix)

    # -------------------------------------------------------------------------
    # NEW: std shading support in domain_mean_timeseries and region_timeseries
    # -------------------------------------------------------------------------
    # Enable by passing show_std=True (and optional std_alpha).
    # Station timeseries still does not plot std by default because it samples
    # a single grid column; use station_three_panel for temporal sigma.
    # -------------------------------------------------------------------------

    # Domain mean time series
    info(" Example 1: Domain mean time series at surface (DOC and Chl) with ±1σ")
    bullet("Goal: area-weighted (if 'art1') mean over space, surface layer, July only.")
    bullet("Variables: 'DOC', 'chl' (groups) and 'temp'. Adds shaded ±1σ band.")

    plot_call(
        domain_mean_timeseries,
        ds=ds,
        variables=["DOC", "chl", "temp"],
        depth="surface",
        months=[7],
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        linewidth=1.8,
        figsize=(10, 4),
        styles=PLOT_STYLES,
        dpi=150,
        show_std=True,
        std_alpha=0.25,
    )

    # Station time series
    info(" Example 2: Station time series - Depth averaged (no σ band)")
    bullet("Goal: plot 'phyto' at nearest node to WE12, entire run, depth averaged.")
    bullet("Note: station_timeseries samples a single column; use station_three_panel for σ shading.")

    plot_call(
        station_timeseries,
        ds=ds,
        variables=["phyto"],
        stations=[STATIONS[0]],
        depth="depth_avg",
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        linewidth=1.8,
        figsize=(10, 4),
        styles=PLOT_STYLES,
        dpi=150,
    )

    # Region time series
    info(" Example 3: Seabed Regional time series (zooplankton) with ±1σ")
    bullet("Goal: mask nodes inside 'Central', compute mean at bottom, full span.")
    bullet("Adds shaded ±1σ band across the region.")

    plot_call(
        region_timeseries,
        ds=ds,
        variables=["zoo"],
        regions=[REGIONS[0]],
        depth="bottom",
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        linewidth=1.8,
        figsize=(10, 4),
        styles=PLOT_STYLES,
        dpi=150,
        show_std=True,
        std_alpha=0.25,
    )

    # 2D variables
    info(" Example 4: Dealing with 2D variables (aice) with ±1σ by region")
    bullet("We can plot vars without a depth dim (time,node) on region timeseries.")
    bullet("Here we compare regions (combine_by='region') and include ±1σ shading.")

    plot_call(
        region_timeseries,
        ds=ds,
        variables=["aice"],
        regions=REGIONS,  # multiple regions
        years=[2018],
        months=[4, 5, 6, 7, 8, 9, 10],
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        combine_by="region",
        verbose=False,
        show_std=True,
        std_alpha=0.25,
    )

    # --- 3-panel demos ---
    info("Example 5: Three-panel plots (already include ±1σ shading)")

    bullet("\n[3-panel / Domain] DOC, full run")
    plot_call(
        domain_three_panel,
        ds=ds,
        variables=["DOC", "aice"],
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        months=None,
        years=None,
        start_date=None,
        end_date=None,
        verbose=True,
    )

    bullet("\n[3-panel / Station WE12] Chl, full run")
    plot_call(
        station_three_panel,
        ds=ds,
        variables=["chl"],
        stations=[STATIONS[0]],
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
    )

    bullet("\n[3-panel / Region Central] DOC, Apr-Oct")
    plot_call(
        region_three_panel,
        ds=ds,
        variables=["DOC"],
        regions=[REGIONS[0]],
        months=[4, 5, 6, 7, 8, 9, 10],
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
    )

    # --- Specific-depth selections ---
    info("Example 6: Plotting at a specific depth")

    bullet("\n Full domain DOC at sigma layer index k=5, July (with ±1σ)")
    plot_call(
        domain_mean_timeseries,
        ds=ds,
        variables=["DOC"],
        depth=5,
        months=[7],
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        show_std=True,
        std_alpha=0.25,
    )

    bullet("\n Station chl at sigma value s=-0.7, full run")
    plot_call(
        station_timeseries,
        ds=ds,
        variables=["chl"],
        stations=[STATIONS[0]],
        depth=-0.7,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
    )

    bullet("\n Regional temp at depth z=-8m, Apr-Oct 2018 (with ±1σ)")
    plot_call(
        region_timeseries,
        ds=ds,
        variables=["temp"],
        regions=[REGIONS[0]],
        depth=-8.0,
        years=[2018],
        months=[4, 5, 6, 7, 8, 9, 10],
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        show_std=True,
        std_alpha=0.25,
    )

    # Multi-line static plots (combine_by options)
    info("Example 7: Combined-line plots (combine_by) with optional ±1σ")

    bullet("\n Domain - Surface, 2018: lines = temp, DOC, chl, phyto, zoo (with ±1σ)")
    plot_call(
        domain_mean_timeseries,
        ds=ds,
        variables=["temp", "DOC", "chl", "phyto", "zoo"],
        depth="surface",
        years=[2018],
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        combine_by="var",
        show_std=True,
        std_alpha=0.20,
    )

    bullet("\n Regions comparison - z=-10 m, 2018: lines=regions (per variable) (with ±1σ)")
    plot_call(
        region_timeseries,
        ds=ds,
        variables=["chl", "phyto"],
        regions=REGIONS,
        depth={"z_m": -10.0},
        years=[2018],
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        combine_by="region",
        show_std=True,
        std_alpha=0.20,
    )

    bullet("\n Regions (all) - JJA 2018 at surface: lines=variables (per region) (with ±1σ)")
    plot_call(
        region_timeseries,
        ds=ds,
        variables=["chl", "phyto", "zoo"],
        regions=REGIONS,
        depth="surface",
        years=[2018],
        months=[6, 7, 8],
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        combine_by="var",
        show_std=True,
        std_alpha=0.20,
    )

    bullet("\n Stations comparison - Apr-Oct 2018, depth-avg: lines=stations (per variable)")
    plot_call(
        station_timeseries,
        ds=ds,
        variables=["chl", "phyto"],
        stations=STATIONS,
        depth="depth_avg",
        start_date="2018-04-01",
        end_date="2018-10-31",
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        combine_by="station",
    )

    bullet("\n Station WE12 - z=-5 m, Apr-Oct 2018: lines=temp, DOC")
    plot_call(
        station_timeseries,
        ds=ds,
        variables=["temp", "DOC"],
        stations=[STATIONS[0]],
        depth=-5.0,
        start_date="2018-04-01",
        end_date="2018-10-31",
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        styles=PLOT_STYLES,
        combine_by="var",
    )

    info(" Output recap")
    bullet(
        "All figures are PNGs named like:\n"
        "  <basename(BASE_DIR)>__<Scope>__<Var>__<DepthTag>__<TimeLabel>__Timeseries.png\n"
        "Combined plots include suffixes like __CombinedByVar / __CombinedByRegion."
    )
    bullet(f"Listing a few outputs in: {out_folder}")
    sample_output_listing(out_folder, prefix)

    elapsed = time.time() - start_ts
    kv("Finished at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    kv("Elapsed (s)", f"{elapsed:0.1f}")
    print(hr("="))
    print("Done")
    print(hr("="))


if __name__ == "__main__":
    if not os.environ.get("PYTHONWARNINGS"):
        warnings.filterwarnings("default")

    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)

