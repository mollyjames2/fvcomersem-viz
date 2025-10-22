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
from matplotlib.colors import LogNorm

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
    region_three_panel
)
from fvcomersemviz.plot import (
    hr, info, bullet, kv,
    try_register_progress_bar,
    list_files, summarize_files,
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
BASE_DIR     = "/data/proteus1/scratch/yli/project/lake_erie/output_updated_river_var"
FILE_PATTERN = "erie_00??.nc"
FIG_DIR      = "/data/proteus1/scratch/moja/projects/Lake_Erie/fviz-plots/"


# By default, fvcomersem-viz auto-detects the plotting module and saves the plots to a corresponding subfolder e.g.:
#   FIG_DIR/<basename(BASE_DIR)>/timeseries/   (for timeseries plots)
#   FIG_DIR/<basename(BASE_DIR)>/maps/         (for maps plots)
# You can OVERRIDE this auto subfolder by setting FVCOM_PLOT_SUBDIR.

# Example: force all following plots to go under "project"
#os.environ["FVCOM_PLOT_SUBDIR"] = "project"

# To return to automatic per-module subfolders use:
#os.environ.pop("FVCOM_PLOT_SUBDIR", None)

# If you want to disable subfolders entirely (everything into the base folder "FIG_DIR"):
#os.environ["FVCOM_PLOT_SUBDIR"] = ""   # empty string disables subfoldering

#------------------------------

# -----------------------------
# Variable groups / composites
# -----------------------------
# You can pass either:
#   - a native model variable name present in the dataset, e.g. "P1_c"
#   - or a *group* defined here:
#       • list/tuple  -> members are summed elementwise
#       • string expr -> evaluated in the dataset namespace (you can do +, -, *, /, etc.)
# Notes:
#   - Make sure every referenced variable exists in the dataset.
#   - Expressions run in a safe namespace that only exposes dataset variables.
#   - Example of an average (uncomment to use):
#       "phyto_avg": "(P1_c + P2_c + P4_c + P5_c) / 4",
GROUPS = {
    "DOC":   "R1_c + R2_c + R3_c + T1_30d_c + T2_30d_c",  # dissolved organic carbon (sum of pools)
    "phyto": ["P1_c", "P2_c", "P4_c", "P5_c"],            # total phytoplankton carbon (sum)
    "zoo":   ["Z4_c", "Z5_c", "Z6_c"],                    # total zooplankton carbon (sum)
    "chl":   "P1_Chl + P2_Chl + P4_Chl + P5_Chl",         # total chlorophyll (sum)
}


#We can set different colourschemes for each of the variables/groups we plot.
#if we don't set a specific colour for a variable it will fall back to default
#If writing a script that produces multiple types of plots (line plots, pcolour plots etc) we can set the colour echeme for each type here as e.g:
 
#    "zoo":   {"line_color": "#9467bd", "cmap": "PuBu"}

PLOT_STYLES = {
    "temp":   {"line_color": "lightblue"},
    "DOC":   {"line_color": "blue"},
    "chl":   {"line_color": "lightgreen"},
    "phyto": {"line_color": "darkgreen"},
    "zoo":   {"line_color": "purple"},
    # Example with log scaling for maps/hov:
    # "nh4": {"line_color": "#ff7f0e", "cmap": "plasma", "norm": LogNorm(1e-3, 1e0)}
}
# -----------------------------
# Station list (nearest-node)
# -----------------------------
# List of (name, latitude, longitude) in decimal degrees (WGS84).
# The plotting code finds the nearest model *node* by great-circle distance (WGS84 ellipsoid).
# Tip: Longitudes west of Greenwich are negative (e.g., -83.10).
STATIONS = [
    ("WE12", 41.90, -83.10),
    ("WE13", 41.80, -83.20),
]

# -----------------------------
# Regions (polygon masks)
# -----------------------------
# Each entry is (region_name, spec_dict).
# Provide exactly ONE of:
#   • shapefile: path to a polygon shapefile (optionally filter with name_field/name_equals)
#   • csv_boundary: path to CSV with boundary coordinates (columns 'lon'/'lat' by default)
# Notes:
#   - For shapefiles, we read all features unless you pass:
#         "name_field": "FIELD_NAME", "name_equals": "Central"
#   - For CSV: if columns aren't named lon/lat, set 'lon_col' and 'lat_col' keys.
#   - The polygon should be ordered around the boundary; closing the ring is handled.


REGIONS = [
    ("Central", {
        "shapefile": "../data/shapefiles/central_basin_single.shp"
    }),
    ("East", {
        "shapefile": "../data/shapefiles/east_basin_single.shp"
    }),
    ("West", {
        "shapefile": "../data/shapefiles/west_basin_single.shp"
    }),
 
# csv example
#   ("West", {
#       "csv_boundary": "/data/proteus1/backup/rito/Models/FVCOM/fvcom-projects/erie/python/postprocessing/west_stations.csv",
#       "lon_col": "lon", 
#       "lat_col": "lat",
#       "convex_hull": True,   # <— wrap points
#       # "sort": "auto",      # (use this if your CSV is a boundary but unordered)
#   }),
]   

# -----------------------------
# Example time windows
# -----------------------------
# MONTHS_EXAMPLE: filter by calendar months (1–12). This applies across all years in the dataset.
MONTHS_EXAMPLE = [4, 5, 6, 7, 8, 9, 10]   # Apr–Oct

# YEARS_EXAMPLE: filter by calendar year(s). You can list multiple years, e.g., [2018, 2019].
YEARS_EXAMPLE  = [2018]

# DATE_RANGE: explicit inclusive date range. Used as (start_date, end_date) in "YYYY-MM-DD" format.
# Internally, the filter keeps timestamps >= start_date and <= end_date.
DATE_RANGE     = ("2018-04-01", "2018-10-31")

# -----------------------------
# Dask progress bar toggle
# -----------------------------
# If True AND dask is installed, the runner enables a simple textual progress bar
# during dask computations. Has no effect if dask is not installed.
SHOW_PROGRESS = False


#----------------------------------------------------------------------------------------


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
    bullet("- Produce three example plots (domain, station, region).")

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

    #  Discover files
    info(" Discovering files")
    files = list_files(BASE_DIR, FILE_PATTERN)
    summarize_files(files)
    if not files:
        print("\nNo files found. Exiting.")
        sys.exit(2)

    #  Load dataset
    info(" Loading dataset (this may be lazy if Dask is available)")
    ds = load_from_base(BASE_DIR, FILE_PATTERN)
    bullet("Dataset loaded. Summary:")
    print_dataset_summary(ds)

    # Where figures will go / filename prefix
    out_folder = out_dir(BASE_DIR, FIG_DIR)
    prefix = file_prefix(BASE_DIR)
    kv("Figure folder", out_folder)
    kv("Filename prefix", prefix)

    # Domain mean time series
    info(" Example 1: Domain mean time series at surface (DOC and Chl)")
    bullet("Goal: area-weighted (if 'art1') mean over space, 'surface' layer, July only.")
    bullet("Variables: 'DOC' and 'chl' (compound/group variables) and temp (FVCOM/ERSEM variable.")

    plot_call(
        domain_mean_timeseries,
        ds=ds,
        variables=["DOC", "chl","temp"],
        depth="surface",
        months=[7],
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        groups=GROUPS,
        linewidth=1.8,
        figsize=(10, 4),
        styles=PLOT_STYLES,
        dpi=150,
    )

    #  Station time series
    info(" Example 2: Station time series - Depth averaged")
    bullet("Goal: plot 'phyto' at the nearest model node to station WE12, entire run, depth averaged.")
    bullet("Nearest node is computed by great-circle distance (WGS84).")

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
    info(" Example 3 : Seabed Regional time series (zooplankton)")
    bullet("Goal: mask nodes inside 'Central' shapefile region, compute mean, bottom, full span.")
    bullet("If connectivity 'nv' exists, elements whose three nodes are inside are kept (strict).")
    bullet("If 'art1' exists, the spatial mean is area-weighted.")

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
    )
    
    # 2D variables
    info(" Example 4 : Dealing with 2d variables")
    bullet("We can also plot variables without a depth dimension i.e. aice (time,node) on domain, region and timeseries plots")
    bullet("just skip adding a depth field to the call")

    plot_call(
        region_timeseries,
        ds=ds,
        variables=["aice"],
        regions=REGIONS,                             # multiple regions
        years=[2018], months=[4,5,6,7,8,9,10],
        base_dir=BASE_DIR, figures_root=FIG_DIR,
        groups=GROUPS, styles=PLOT_STYLES,
        combine_by="region",
        verbose=False,
    )

    # --- 3-panel demos ---
    info("Example 4: Three-pane plots (Surface ±1σ, Bottom ±1σ, Profile mean  ±1σ): Domain/Station/Region")
    
    bullet("\n[3-panel / Domain] DOC, full run")
    plot_call(
        domain_three_panel,
        ds=ds,
        variables=["DOC","aice"],
        base_dir=BASE_DIR, figures_root=FIG_DIR, groups=GROUPS,styles=PLOT_STYLES,
        months=None, years=None, start_date=None, end_date=None, verbose=True  # full span
    )
    
    bullet("\n[3-panel / Station WE12] Chl, full run")
    plot_call(
        station_three_panel,
        ds=ds,
        variables=["chl"],
        stations=[STATIONS[0]],
        base_dir=BASE_DIR, figures_root=FIG_DIR, groups=GROUPS,styles=PLOT_STYLES,
    )
    
    bullet("\n[3-panel / Region Central] DOC, Apr–Oct")
    plot_call(
        region_three_panel,
        ds=ds,
        variables=["DOC"],
        regions=[REGIONS[0]],
        months=[4,5,6,7,8,9,10],  # Apr–Oct example window
        base_dir=BASE_DIR, figures_root=FIG_DIR, groups=GROUPS,styles=PLOT_STYLES,
    )

    # 
    # --- Specific-depth selections (shorthand + longhand)
    info("Example 6: Plotting at a specific depth")
    
    #   Shorthand:
    #     depth=5        -> select sigma layer index k=5              == ("siglay_index", 5)
    #     depth=-0.7     -> select nearest sigma value s in [-1, 0]   == ("sigma", -0.7)
    #     depth=-8.0     -> select absolute depth z=-8 m (down)       == ("z_m", -8.0)
    #
    #   Longhand (explicit tuples/dicts):
    #     depth=("siglay_index", 5)
    #     depth=("sigma", -0.7)
    #     depth=("z_m", -8.0)                 # optional extras: depth=("z_m", -8.0, {"zvar": "z"})
    #     depth={"z_m": -8.0, "zvar": "z"}    # dict form
    #
    # Notes:
    #   • Floats in [-1, 0] are interpreted as sigma; other floats are treated as meters (z, negative downward).
    #   • Absolute-depth selection requires a vertical coordinate with a 'siglay' dim (default variable name: 'z').

    
    #DOMAIN — DOC at sigma layer index k=5, July only
    bullet("\n Full domain DOC at sigma layer index k=5, July")
    plot_call(
        domain_mean_timeseries,
        ds=ds,
        variables=["DOC"],
        depth=5,                      # == ("siglay_index", 5)
        months=[7],                   # July across all years
        base_dir=BASE_DIR, figures_root=FIG_DIR, groups=GROUPS,styles=PLOT_STYLES,
    )

    # STATION (WE12) — chl at sigma value s=-0.7, full run
    bullet("\n Station chl at sigma value s=-0.7, full run")
    plot_call(
        station_timeseries,
        ds=ds,
        variables=["chl"],
        stations=[STATIONS[0]],       # e.g., ("WE12", 41.90, -83.10)
        depth=-0.7,                   # == ("sigma", -0.7)
        base_dir=BASE_DIR, figures_root=FIG_DIR, groups=GROUPS,styles=PLOT_STYLES,
    )

    # REGION (Central) — temperature at absolute depth z=-8 m, Apr–Oct 2018
    bullet("\n Regional temp at depth z=-8m, Apr–Oct 2018")
    plot_call(
        region_timeseries,
        ds=ds,
        variables=["temp"],
        regions=[REGIONS[0]],         # ("Central", {...})
        depth=-8.0,                   # == ("z_m", -8.0)  (meters; negative = below surface)
        years=[2018], months=[4,5,6,7,8,9,10],
        base_dir=BASE_DIR, figures_root=FIG_DIR, groups=GROUPS,styles=PLOT_STYLES,
    )
    
    # =============================================================================
    # Multi-line static plots (combine_by options)
    # =============================================================================
    # The static timeseries functions now support a `combine_by` keyword to group
    # multiple lines onto a single figure instead of making one plot per variable.
    #
    #   combine_by=None      → default (one PNG per variable / per region / per station)
    #   combine_by="var"     → one plot per domain/region/station, lines = variables
    #   combine_by="region"  → one plot per variable, lines = regions
    #   combine_by="station" → one plot per variable, lines = stations
    #
    # Example usage patterns:
    #   - domain_mean_timeseries(..., combine_by="var")      # domain-wide, lines = variables
    #   - region_timeseries(..., combine_by="region")        # compare regions for one variable
    #   - region_timeseries(..., combine_by="var")           # multiple variables in each region
    #   - station_timeseries(..., combine_by="station")      # compare stations for one variable
    #   - station_timeseries(..., combine_by="var")          # multiple variables per station
    #
    # All filenames automatically include a suffix like "__CombinedByVar" etc.
    # =============================================================================
    
    # DOMAIN — combine_by='var': one figure, lines = variables
    bullet("\n Domain — Surface, 2018: lines = temp, DOC, chl, phyto, zoo")
    plot_call(
        domain_mean_timeseries,
        ds=ds,
        variables=["temp", "DOC", "chl", "phyto", "zoo"],
        depth="surface",
        years=[2018],
        base_dir=BASE_DIR, figures_root=FIG_DIR, groups=GROUPS, styles=PLOT_STYLES,
        combine_by="var",
    )
    
    # REGIONS — combine_by='region': one figure per variable, lines = regions
    bullet("\n Regions comparison — z = -10 m, 2018: lines = regions (per variable)")
    plot_call(
        region_timeseries,
        ds=ds,
        variables=["chl", "phyto"],
        regions=REGIONS,
        depth={"z_m": -10.0},
        years=[2018],
        base_dir=BASE_DIR, figures_root=FIG_DIR, groups=GROUPS, styles=PLOT_STYLES,
        combine_by="region",
    )

    # REGIONS — combine_by='var': one figure per region, lines = variables
    bullet("\n Regions (all) — JJA 2018 at surface: lines = chl, phyto, zoo (per region)")
    plot_call(
        region_timeseries,
        ds=ds,
        variables=["chl", "phyto", "zoo"],
        regions=REGIONS,                 # list of (name, spec)
        depth="surface",
        years=[2018], months=[6, 7, 8],
        base_dir=BASE_DIR, figures_root=FIG_DIR, groups=GROUPS, styles=PLOT_STYLES,
        combine_by="var",
    )

    # STATIONS — combine_by='station': one figure per variable, lines = stations
    bullet("\n Stations comparison — Apr–Oct 2018, depth-avg: lines = stations (per variable)")
    plot_call(
        station_timeseries,
        ds=ds,
        variables=["chl", "phyto"],
        stations=STATIONS,               # list of (name, lat, lon)
        depth="depth_avg",
        start_date="2018-04-01", end_date="2018-10-31",
        base_dir=BASE_DIR, figures_root=FIG_DIR, groups=GROUPS, styles=PLOT_STYLES,
        combine_by="station",
    )
    
    # STATIONS — combine_by='var': one figure per station, lines = variables
    bullet("\n Station WE12 — z = -5 m, Apr–Oct 2018: lines = temp, DOC")
    plot_call(
        station_timeseries,
        ds=ds,
        variables=["temp", "DOC"],
        stations=[STATIONS[0]],          # just WE12
        depth=-5.0,                      # absolute metres below surface
        start_date="2018-04-01", end_date="2018-10-31",
        base_dir=BASE_DIR, figures_root=FIG_DIR, groups=GROUPS, styles=PLOT_STYLES,
        combine_by="var",
    )

    

    #  Output recap
    info(" Output recap")
    bullet("All figures are PNGs named like:\n"
           "  <basename(BASE_DIR)>__<Scope>__<Var>__<DepthTag>__<TimeLabel>__Timeseries.png")
    bullet(f"Listing a few outputs in: {out_folder}")
    sample_output_listing(out_folder, prefix)

    elapsed = time.time() - start_ts
    kv("Finished at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    kv("Elapsed (s)", f"{elapsed:0.1f}")
    print(hr("="))
    print("Done ")
    print(hr("="))


if __name__ == "__main__":
    
    if not os.environ.get("PYTHONWARNINGS"):
        warnings.filterwarnings("default")

    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
