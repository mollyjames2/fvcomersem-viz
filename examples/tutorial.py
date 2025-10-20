#!/usr/bin/env python
# coding: utf-8

# FVCOM–ERSEM Visualisation Tutorial (using `fvcomersem-viz`)
# 
# Welcome! This notebook demonstrates how to make clear, reproducible figures from FVCOM–ERSEM model output using the lightweight, function-first plotting toolkit`fvcomersem-viz`. The package keeps the API simple (plain Python functions, no GUI) and ships with focused plotting routines and helpers so you can go from raw NetCDF to publication-ready graphics quickly.
# 
# ---
# 
# What you’ll learn here
# 
#   How to point the toolkit at your FVCOM output** (single file or collections).
#   How to make:
# 
#      Map plots
#      Hovmöller diagrams 
#      Time series plots
#      KDE stoichiometry panels (e.g., N:C / P:C vs variables at surface/bottom).
#      How to use variable groups/composites (e.g., `chl`, `phyto`, `DOC`) via simple algebraic expressions.
#      Relationship curves
#      Commmunity compostion plots (i.e. phyto/zoo)
#      Time series and map animations
#      How to control time windows (months/years/date ranges) and depth slices (surface, bottom, fixed-z, depth-avg).
# 
# ---
# 
# What this script expects
# 
#  Model data: FVCOM–ERSEM NetCDF output (can take single or multiple files)
#  Optional region definitions: shapefiles or CSV polygons 
#  Optional station list: for point-based analysis
# 
# > If your paths differ from the examples, just edit the `BASE_DIR`, `FILE_PATTERN`, and any region/station paths in the “Setup” cells.
# 
# ---
# 
# Package at a glance
# 
# Name: fvcomersem-viz
#   Key modules:
# 
#    plots/maps.py – maps of scalar fields on the FVCOM grid
#    plots/hovmoller.py – along-time/along-depth sections at stations
#    plots/timeseries.py – single or multi-variable time series & composites
#    plots/kde_stoichiometry.py – 2×2 stoichiometry panels
#    plots/composition.py, - visualises the composition of grouped communities (e.g., phytoplankton or zooplankton) as relative shares across time/space.
#    plots/curves.py - builds reusable diagnostic curves 
#    plots/animate.py - creates animations of timeseries and map plots
#    io.py, regions.py, utils.py, plot.py – data discovery, time/depth filters, masks, labels, plotting functions
# 
# ---
# 
# Installing and Setting Up fvcomersem-viz
# ========================================
#
# Requirements (tested versions)
# ------------------------------
# Python ≥ 3.9  (3.11 recommended)
# Core stack: numpy, pandas, xarray, matplotlib, netCDF4, cftime, scipy
# Geospatial (for region masks/overlays): geopandas, shapely, pyproj, rtree (optional but recommended)
# Performance (optional): dask[array]
#
# ------------------------------------------------------------
# Installation Instructions
# ------------------------------------------------------------
#
# 1. Create a clean environment with FVCOM-compatible libraries:
#    conda create -n fviz python=3.11 geopandas shapely pyproj rtree -c conda-forge
#    conda activate fviz
#
# 2. Install fvcomersem-viz from GitHub:
#    pip install "git+https://github.com/mollyjames2/fvcomersem-viz.git"
#
#    # or, if you’ve cloned the repo locally (for development):
#    git clone https://github.com/mollyjames2/fvcomersem-viz.git
#    cd fvcomersem-viz
#    pip install -e .    # editable/development mode
#
# 3. Verify installation:
#    python -c "import fvcomersemviz; print(fvcomersemviz.__version__)"
#
# 4. (Optional) Run a basic functionality test:
#    python tests/check_install.py
#
#    # This script checks that core dependencies, plotting modules,
#    # and required imports load correctly in your environment.
#
# ------------------------------------------------------------
# Notes
# ------------------------------------------------------------
# - The package depends on a working FVCOM or FVCOM–ERSEM model output dataset.
# - If using regional masking or shapefile overlays, install the optional geospatial libs.
# - Always activate your environment before running analysis scripts:
#     conda activate fviz
#
#
# After installation, you can run the examples under:
#     https://github.com/mollyjames2/fvcomersem-viz/examples

#-------------------------------------

# 
# Typical workflow used in this notebook
# 
# 1.Setup paths & imports
# 
#    
#    from fvcomersemviz.plots import maps, hovmoller, timeseries
#    from fvcomersemviz.io import filter_time
#    
# 2. Load data (single file or pattern), select time window and depth slice.
# 3. Plot using a purpose-built function (e.g., maps.plot_surface_field(...)), tweak labels, save.
# 4. Repeat for stations/regions/variables as needed.
# 
# ---
# 
# Reproducibility & citations
# 
#   Please cite fvcomersem-viz alongside the relevant FVCOM/ERSEM model references when using figures generated from this toolkit. 
# ---
# 
# Core capabilities
# 
#   Map visualisation:
#     Create horizontal maps of surface or depth-averaged fields, optionally masked by regions or polygons.
#     Useful for showing spatial patterns (e.g., chlorophyll, nutrients, oxygen).
# 
#   Hovmöller diagrams:
#     Plot time-depth (sigma or fixed-z) sections at selected stations or regions to reveal seasonal and interannual variability.
# 
#   Time series and composites:
#     Produce single or multi-variable time series, monthly/seasonal climatologies, and box-region averages.
# 
#   Stoichiometry and diagnostics:
#     Generate 2×2 KDE panels or scatter plots to explore relationships between model tracers (e.g., N:C, P:C ratios, oxygen vs temperature).
# 
#   Variable groups and expressions:
#     Access groups like phyto, zoo, nutrients, or define on-the-fly algebraic combinations (e.g., total chlorophyll or N:P).
# 
# Typical use cases
# 
#   Generating consistent visual outputs across multiple FVCOM–ERSEM experiments.
# 
#   Producing diagnostics or summary figures for reports and publications.
# 
#   Quickly inspecting model fields without building a full analysis pipeline.
# 
#   Supporting automated post-processing workflows for long-term simulations.
# 
# 
# 
#---------------------------------- SETUP AND EXAMPLES -----------------------------------------------#

# Setting up your data paths
# 
#   In this section, we tell the notebook where to find the FVCOM–ERSEM model output and where to save plots.
# 
#   You can configure everything by editing a few key variables:
#             
# BASE_DIR → the folder where your FVCOM–ERSEM NetCDF files live.
# This should point to the root of your model output directory — the location of all NETCDF output files.
# 
# FILE_PATTERN → the naming pattern for your files. 
# This pattern uses wildcards (? or *) to match all relevant NetCDF files you want to load together.# 
# 
# FIG_DIR → the directory where all output plots will be saved.

# The package automatically creates subfolders inside FIG_DIR for different plot types, e.g.:
# 
# <FIG_DIR>/<basename(BASE_DIR)>/maps/
# <FIG_DIR>/<basename(BASE_DIR)>/timeseries/
# where basename(BASE_DIR) is the name of the folder holding the fbcom-ersem output files
# 
# You can override or disable that behaviour using the variable FVCOM_PLOT_SUBDIR:
#
# FVCOM_PLOT_SUBDIR = "project" # force all plots into a folder called “project”.
# FVCOM_PLOT_SUBDIR = "" # disable subfolders; save everything directly into FIG_DIR.
# 
# 
# 
# Tip: keeping these paths and patterns together makes it easy to reuse the same notebook for different model runs — just edit BASE_DIR, FILE_PATTERN, and (optionally) FIG_DIR at the top.
# 

#------ Set the filepaths here

BASE_DIR = "/data/proteus1/scratch/yli/project/lake_erie/output_updated_river_var"
FILE_PATTERN = "erie_00??.nc"
FIG_DIR      = "/data/proteus1/scratch/moja/projects/Lake_Erie/fvcomersem-viz/examples/plots/"
#FVCOM_PLOT_SUBDIR = "" # disable subfolders; save everything directly into FIG_DIR


# ---


# Subsampling for Regional and Station Plots
# 
#   To focus on specific areas or points within the FVCOM–ERSEM domain, we can subsample the dataset using simple metadata that defines stations (points) and regions (polygons).
# 

# 
# ####  Stations
# 
#  Defined as a list of tuples: `(name, latitude, longitude)` in decimal degrees (WGS84).
#  The code automatically finds the nearest model node for each station using great-circle distance.
#  Ideal for generating time series or Hovmöller plots at fixed locations.

#     Note: longitudes west of Greenwich are negative.

#------ Set stations here
STATIONS = [
    ("WE12", 41.90, -83.10),
    ("WE13", 41.80, -83.20),
]

# 
# ####  Regions
# 
#  Defined as a list of tuples: (region_name, spec_dict) where spec_dict describes a polygon source.
#  You can provide either:
# 
#    A shapefile path (optionally filtered by name_field / name_equals), or
#    A CSV boundary file with lon/lat columns (use convex_hull=True to wrap scattered points).
#  These polygons are converted into grid masks, allowing plots or averages to be limited to specific basins or zones.

#------ Set regions here
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
]
  
# Using these simple definitions, the plotting functions automatically extract the relevant subset of model data — either at the nearest node (for stations) or within a polygon mask (for regions) — before generating plots or summary statistics.


# ---

#### Groups and plot styles

# Variable Groups / Composites
#  You can pass either:
#    - a native model variable name present in the dataset, e.g. "P1_c"
#    - or a group defined here:
#        • list/tuple  -> members are summed elementwise
#        • string expr -> evaluated in the dataset namespace (you can do +, -, *, /, etc.)
#   Notes:
#    - Make sure every referenced variable exists in the dataset.
#    - Expressions run in a safe namespace that only exposes dataset variables.
#    - Example of an average:     "phyto_avg": "(P1_c + P2_c + P4_c + P5_c) / 4",


#------ Set groups here
GROUPS = {
    "DOC":   "R1_c + R2_c + R3_c + T1_30d_c + T2_30d_c",  # dissolved organic carbon (sum of pools)
    "phyto": ["P1_c", "P2_c", "P4_c", "P5_c"],            # total phytoplankton carbon (sum)
    "zoo":   ["Z4_c", "Z5_c", "Z6_c"],                    # total zooplankton carbon (sum)
    "chl":   "P1_Chl + P2_Chl + P4_Chl + P5_Chl",         # total chlorophyll (sum)
}
 
# ##### Plot styles
# We can set different colourschemes for each of the variables/groups we plot.
# if we don't set a specific colour for a variable it will fall back to default
# If writing a script that produces multiple types of plots (line plots, pcolour plots etc) we can set the colour scheme for each type


#------ Set plot styles here
PLOT_STYLES = {
    "temp":   {"line_color": "lightblue", "cmap": "coolwarm"},
    "DOC":   {"line_color": "blue", "cmap": "viridis"},
    "chl":   {"line_color": "lightgreen", "cmap": "Greens", "vmin": 0.0, "vmax": 5.0},
    "phyto": {"line_color": "darkgreen","cmap": "YlGn"},
    "zoo":   {"line_color": "purple","cmap": "PuBu"},
}

# When combining by "region" or "station" for multiline plots and animations, you can also key styles by the region or station name to set their line colors.



#---------------------------------------#### PLOTS ####---------------------------------------#

# Use the toggles here to switch exmple plots on and off in this script
plot_timeseries = True
plot_maps = True
plot_hovmoller = True
plot_kde = True
plot_composition = True
plot_curves = True
plot_animate = True

# ------Loading the data-------

# Imports
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np

  
# Package imports
from fvcomersemviz.io import load_from_base
from fvcomersemviz.utils import out_dir, file_prefix
from fvcomersemviz.plot import (
    hr, info, bullet, kv,
    try_register_progress_bar,
    list_files, summarize_files,
    plot_call,
    print_dataset_summary,
    ensure_paths_exist,
    sample_output_listing,
)

#read in stations
bullet("\nStations (name, lat, lon):")
for s in STATIONS:
    bullet(f"• {s}")

#read in regions
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


#             _______ _                               _           
#            |__   __(_)                             (_)          
#               | |   _ _ __ ___   ___  ___  ___ _ __ _  ___  ___ 
#               | |  | | '_ ` _ \ / _ \/ __|/ _ \ '__| |/ _ \/ __|
#               | |  | | | | | | |  __/\__ \  __/ |  | |  __/\__ \
#               |_|  |_|_| |_| |_|\___||___/\___|_|  |_|\___||___/
#                                                      
                                                                                                   

# Time-Series Overview
# --------------------
# Tools to extract and plot FVCOM–ERSEM time series at three scopes:
# - Domain: area-weighted mean across the full grid.
# - Region: mean over a polygon (from shapefile or CSV boundary).
# - Station: series at the nearest model node to each (lat, lon).
#
# You can choose variables or predefined groups (e.g., "chl", "phyto"),
# filter the time window (months, years, or explicit start/end dates),
# and select depth ("surface", "bottom", "depth_avg", sigma value/index, or fixed meters).
#
# Combine multiple series in a single plot with:
#   combine_by="var" | "region" | "station"
#
# Outputs are saved under:
#   FIG_DIR/<basename(BASE_DIR)>/timeseries/
# Filenames encode scope, variable, depth, and time span.

# The full function signatures for each of the timeseries plotting functions provided by this package are described in full in the examples

#Examples:
if plot_timeseries:

  from fvcomersemviz.plots.timeseries import (
      domain_mean_timeseries,
      station_timeseries,
      region_timeseries,
      domain_three_panel,
      station_three_panel,
      region_three_panel,
  )
  




  # --- Timeseries examples: domain, station, region  ---
  
  # 1) Domain mean timeseries 
  # Full argument reference for domain_mean_timeseries(...)
  # Each parameter below is annotated with what it does and accepted values.
  
  # def domain_mean_timeseries(
  #     ds: xr.Dataset,                      # The opened FVCOM–ERSEM dataset (e.g., from load_from_base()).
  #     variables: List[str],                # One or more variable names to plot. Each entry can be:
  #                                          #   • a native variable in ds (e.g., "temp", "P1_c"), or
  #                                          #   • a group name defined in `groups` (e.g., "chl", "DOC").
  #     *,
  #     depth: Any,                          # Vertical selection for all variables (dataset-level slice unless absolute z):
  #                                          #   "surface" | "bottom" | "depth_avg"
  #                                          #   int (sigma index, k)          -> e.g., 5
  #                                          #   float in [-1, 0] (sigma val)  -> e.g., -0.7
  #                                          #   other float (absolute depth m)-> e.g., -8.0 (8 m below surface)
  #                                          #   ("z_m", z) or {"z_m": z}      -> explicit absolute-depth form
  #     months: Optional[List[int]] = None,  # Filter to calendar months (1–12). Example: [7] for July, [4,5,6,7,8,9,10] for Apr–Oct.
  #     years: Optional[List[int]] = None,   # Filter to calendar years. Example: [2018] or [2019, 2020].
  #     start_date: Optional[str] = None,    # Start date (inclusive) "YYYY-MM-DD". Use together with end_date.
  #     end_date: Optional[str] = None,      # End date (inclusive) "YYYY-MM-DD". Use together with start_date.
  #     base_dir: str,                       # Model run directory; used for filename prefix and output folder structure.
  #     figures_root: str,                   # Root folder for figures. Module subfolder is auto-added (e.g., /timeseries/).
  #     groups: Optional[Dict[str, Any]] = None,  # Composite/group definitions so you can request semantic vars:
  #                                               #   "chl": "P1_Chl + P2_Chl + P4_Chl + P5_Chl"
  #                                               #   "phyto": ["P1_c","P2_c","P4_c","P5_c"]  (summed elementwise)
  #     linewidth: float = 1.5,              # Line thickness for plotted series.
  #     figsize: tuple = (10, 4),            # Figure size in inches (width, height).
  #     dpi: int = 150,                      # Output resolution for saved PNGs.
  #     styles: Optional[Dict[str, Dict[str, Any]]] = None,  # Optional per-variable styles, e.g.:
  #                                               #   {"temp": {"line_color": "lightblue"},
  #                                               #    "chl":  {"line_color": "lightgreen"}}
  #                                               # If a var has no style, Matplotlib defaults are used.
  #     verbose: bool = True,                # Print progress (selected depth, time window, saved path, etc.).
  #     combine_by: Optional[str] = None,    # Multi-line mode:
  #                                          #   None      -> one PNG per variable (default).
  #                                          #   "var"     -> one PNG total with multiple lines (one per variable).
  # ) -> None:
  #     """
  #     Plot domain-wide mean time series and save PNG(s) to disk.
  #
  #     File name pattern:
  #       <prefix>__Domain__<VarOrMulti>__<DepthTag>__<TimeLabel>__Timeseries[__CombinedByVar].png
  #
  # Examples:

  # Separate figures (one per variable)  
  domain_mean_timeseries(
      ds=ds,
      variables=["DOC", "chl", "temp"],
      depth="surface",
      months=[7],
      base_dir=BASE_DIR,
      figures_root=FIG_DIR,
      groups=GROUPS,
      styles=PLOT_STYLES,
      dpi=150,
      verbose=False,
  )
  
  # One multi-line figure (lines = variables)
  domain_mean_timeseries(
      ds=ds,
      variables=["DOC", "chl", "temp"],
      depth="surface",
      months=[7],
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      groups=GROUPS, styles=PLOT_STYLES,
      combine_by="var", # <- control for plotting multiple lines on one plot
      verbose=False,
  )

  # 2) Station timeseries
  # def station_timeseries(
  #     ds: xr.Dataset,                               # Opened FVCOM–ERSEM dataset (e.g., via load_from_base()).
  #     variables: List[str],                         # One or more series to plot. Each can be:
  #                                                   #   • a native variable in ds (e.g., "temp", "P1_c"), or
  #                                                   #   • a group name from `groups` (e.g., "chl", "DOC", "phyto").
  #     stations: List[Tuple[str, float, float]],     # Station list: (name, lat, lon) in WGS84.
  #                                                   #   - Longitude west of Greenwich should be negative (e.g., -83.10).
  #                                                   #   - The nearest model *node* is found by great-circle distance (WGS84).
  #     *,
  #     depth: Any,                                   # Vertical selection at each station:
  #                                                   #   "surface" | "bottom" | "depth_avg"
  #                                                   #   int (sigma index, k)           -> e.g., 5
  #                                                   #   float in [-1, 0] (sigma val)   -> e.g., -0.7
  #                                                   #   other float (absolute depth m) -> e.g., -8.0 (8 m below surface)
  #                                                   #   ("z_m", z) or {"z_m": z}       -> explicit absolute-depth form
  #     months: Optional[List[int]] = None,           # Optional month filter (1–12). Example: [4,5,6,7,8,9,10] for Apr–Oct.
  #     years: Optional[List[int]] = None,            # Optional year filter. Example: [2018] or [2019, 2020].
  #     start_date: Optional[str] = None,             # Optional start date "YYYY-MM-DD" (used with end_date).
  #     end_date: Optional[str] = None,               # Optional end date   "YYYY-MM-DD" (used with start_date).
  #     base_dir: str,                                # Model run directory; used for filename prefix and output folder structure.
  #     figures_root: str,                            # Root output folder. A module subfolder (e.g., /timeseries/) is added automatically.
  #     groups: Optional[Dict[str, Any]] = None,      # Composite definitions so you can request semantic variables:
  #                                                   #   "chl": "P1_Chl + P2_Chl + P4_Chl + P5_Chl"
  #                                                   #   "phyto": ["P1_c","P2_c","P4_c","P5_c"]   (elementwise sum)
  #     linewidth: float = 1.5,                       # Line thickness.
  #     figsize: tuple = (10, 4),                     # Figure size in inches (width, height).
  #     dpi: int = 150,                               # PNG resolution.
  #     styles: Optional[Dict[str, Dict[str, Any]]] = None,  # Optional per-variable styles, e.g.:
  #                                                   #   {"temp": {"line_color": "lightblue"},
  #                                                   #    "DOC":  {"line_color": "blue"}}
  #     verbose: bool = True,                         # Print progress (resolved node index, time window, saved path, etc.).
  #     combine_by: Optional[str] = None,             # Multi-line modes for convenience:
  #                                                   #   None       -> one PNG per (station × variable)  [default]
  #                                                   #   "var"      -> one PNG per station,  lines = variables
  #                                                   #   "station"  -> one PNG per variable, lines = stations
  # ) -> None:
  #     """

  # Notes:
  # - Nearest-node lookup uses great-circle distance in WGS84; ensure station lon/lat are WGS84 and lon west < 0.
  # - Composites in `groups` allow variables like "chl"/"phyto"/"zoo" without rewriting expressions each time.
  # - Works with Dask-chunked datasets; computation is triggered during reduction/plot.
  # - Returns None; to view in a notebook, display the saved PNGs afterwards (e.g., with a gallery cell).

  # Examples:
  # Depth-averaged phyto at first station in STATIONS
  station_timeseries(
      ds=ds,
      variables=["phyto"],
      stations=[STATIONS[0]],  # e.g., ("WE12", 41.90, -83.10)
      depth="depth_avg",
      base_dir=BASE_DIR,
      figures_root=FIG_DIR,
      groups=GROUPS,
      styles=PLOT_STYLES,
      dpi=150,
      verbose=False,
  )
  
  # Station WE12 — z = -5 m, Apr–Oct 2018: temp + DOC on one plot
  station_timeseries(
      ds=ds,
      variables=["temp", "DOC"],
      stations=[STATIONS[0]],                      # e.g., ("WE12", 41.90, -83.10)
      depth=-5.0,                                  # absolute metres below surface (requires vertical coords)
      start_date="2018-04-01", end_date="2018-10-31",
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      groups=GROUPS, styles=PLOT_STYLES,
      combine_by="var",
      verbose=False,
  )
  
  # All stations — surface temp, Apr–Oct 2018: one plot, one line per station
  station_timeseries(
      ds=ds,
      variables=["temp"],
      stations=STATIONS,                           # multiple stations
      depth="surface",
      start_date="2018-04-01", end_date="2018-10-31",
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      groups=GROUPS, styles=PLOT_STYLES,
      combine_by="station",
      verbose=False,
  )


  # 3) Region timeseries 
  # def region_timeseries(
  #     ds: xr.Dataset,                               # Opened FVCOM–ERSEM dataset (e.g., via load_from_base()).
  #     variables: List[str],                         # One or more series to plot. Each can be:
  #                                                   #   • a native variable in ds (e.g., "temp", "P1_c"), or
  #                                                   #   • a group name from `groups` (e.g., "chl", "DOC", "phyto").
  #     regions: List[Tuple[str, Dict[str, Any]]],    # Region list as (region_name, spec_dict). spec_dict provides EXACTLY ONE source:
  #                                                   #   {"shapefile": "/path/to/region.shp"}                      # optional: "name_field", "name_equals"
  #                                                   #   {"csv_boundary": "/path/to/boundary.csv"}                 # optional: "lon_col", "lat_col", "convex_hull"
  #     *,
  #     depth: Any,                                   # Vertical selection before spatial aggregation:
  #                                                   #   "surface" | "bottom" | "depth_avg"
  #                                                   #   int (sigma index, k)           -> e.g., 5
  #                                                   #   float in [-1, 0] (sigma val)   -> e.g., -0.7
  #                                                   #   other float (absolute depth m) -> e.g., -8.0 (8 m below surface)
  #                                                   #   ("z_m", z) or {"z_m": z}       -> explicit absolute-depth form
  #     months: Optional[List[int]] = None,           # Optional month filter (1–12). Example: [4,5,6,7,8,9,10] for Apr–Oct.
  #     years: Optional[List[int]] = None,            # Optional year filter. Example: [2018] or [2019, 2020].
  #     start_date: Optional[str] = None,             # Optional start date "YYYY-MM-DD" (used with end_date).
  #     end_date: Optional[str] = None,               # Optional end date   "YYYY-MM-DD" (used with start_date).
  #     base_dir: str,                                # Model run directory; used for filename prefix and output folder structure.
  #     figures_root: str,                            # Root output folder. A module subfolder (e.g., /timeseries/) is added automatically.
  #     groups: Optional[Dict[str, Any]] = None,      # Composite definitions so you can request semantic variables:
  #                                                   #   "chl": "P1_Chl + P2_Chl + P4_Chl + P5_Chl"
  #                                                   #   "phyto": ["P1_c","P2_c","P4_c","P5_c"]   (elementwise sum)
  #     linewidth: float = 1.5,                       # Line thickness.
  #     figsize: tuple = (10, 4),                     # Figure size in inches (width, height).
  #     dpi: int = 150,                               # PNG resolution.
  #     styles: Optional[Dict[str, Dict[str, Any]]] = None,  # Optional per-series styles, e.g. line colors:
  #                                                   #   {"chl": {"line_color": "lightgreen"}, "temp": {"line_color": "lightblue"}}
  #     verbose: bool = True,                         # Print progress (mask details, time window, saved path, etc.).
  #     combine_by: Optional[str] = None,             # Multi-line modes for convenience:
  #                                                   #   None       -> one PNG per (region × variable)  [default]
  #                                                   #   "var"      -> one PNG per region,   lines = variables
  #                                                   #   "region"   -> one PNG per variable, lines = regions
  # ) -> None:
  #     """
  #     Plot regional mean time series using polygon masks and save PNG(s).
  #
  #     How masking works:
  #       • A node mask is built from the shapefile/CSV polygon (nodes inside are kept).
  #       • If mesh connectivity `nv` exists, an element mask can be derived (keep elements whose 3 nodes are inside).
  #       • Area weighting is used automatically if `art1` is available; otherwise means are unweighted.
  #       • Absolute-depth requests (e.g., depth=-8.0) are applied AFTER masking to ensure the correct local water column.
  #
  #     Output name pattern:
  #       <prefix>__Region-<Name>__<VarOrMulti>__<DepthTag>__<TimeLabel>__Timeseries[__CombinedByVar|__CombinedByRegion].png
  #
  # Notes:
  # - Region masks are built on the FVCOM grid; elements/nodes strictly inside the polygon are included.
  # - If mesh connectivity `nv` is present, “strict” element-inclusion may be used (all three nodes inside).
  # - If an area field (e.g., 'art1') exists, regional means are area-weighted; otherwise unweighted.
  # - CSV boundaries should trace the polygon perimeter (or set convex_hull=True to wrap scattered points).
  # - Works with Dask-chunked datasets; computation is triggered during reduction/plot.
  # - Returns None; to view in a notebook, display the saved PNGs afterwards (e.g., with a gallery cell).
  
  # Examples:
  
  #Bottom zooplankton in first region (e.g., "Central"), full span
  region_timeseries(
      ds=ds,
      variables=["zoo"],
      regions=[REGIONS[0]],    # e.g., ("Central", {"shapefile": "...shp"})
      depth="bottom",
      base_dir=BASE_DIR,
      figures_root=FIG_DIR,
      groups=GROUPS,
      styles=PLOT_STYLES,
      dpi=150,
      verbose=False,
  )
  
  # Central region — surface, Jul 2018: chl + phyto + zoo on one plot
  region_timeseries(
      ds=ds,
      variables=["chl", "phyto", "zoo"],
      regions=[REGIONS[0]],                        
      depth="surface",
      months=[7], years=[2018],
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      groups=GROUPS, styles=PLOT_STYLES,
      combine_by="var", 
      verbose=False,
  )
  
  # Compare regions — bottom DOC, Apr–Oct 2018: one plot, one line per region
  region_timeseries(
      ds=ds,
      variables=["DOC"],
      regions=REGIONS,                             # multiple regions
      depth="bottom",
      years=[2018], months=[4,5,6,7,8,9,10],
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      groups=GROUPS, styles=PLOT_STYLES,
      combine_by="region",
      verbose=False,
  )

  # 4) Domain-wide Three-panel Figures (Surface ±1σ, Bottom ±1σ, Profile mean ±1σ) (1 figure with 3 subplots)
  # Full argument reference for domain_three_panel(...)
  # Each parameter below is annotated with what it does and accepted values.
  # Produces a 3×1 (or similar) figure with:
  #   • Surface time series ±1σ
  #   • Bottom  time series ±1σ
  #   • Vertical-profile mean (depth-avg) time series ±1σ
  # One figure per variable; saves to disk; returns None.
  
  # def domain_three_panel(
  #     ds: xr.Dataset,                              # Xarray Dataset with FVCOM–ERSEM output (already opened/combined)
  #     variables: list[str],                        # One or more names: native vars (e.g., "temp") or composites (e.g., "chl") if provided in `groups`
  #     *,                                           # Everything after this must be passed as keyword-only (safer, clearer)
  #     months=None,                                 # Calendar months to include (1–12) across all years; e.g., [7] or [4,5,6,7,8,9,10]; None = no month filter
  #     years=None,                                  # Calendar years to include; e.g., [2018] or [2018, 2019]; None = no year filter
  #     start_date=None,                             # Inclusive start date "YYYY-MM-DD"; used with end_date; None = no start bound
  #     end_date=None,                               # Inclusive end date   "YYYY-MM-DD"; used with start_date; None = no end bound
  #     base_dir: str,                               # Path to the model run folder; used for output subfolder and filename prefix
  #     figures_root: str,                           # Root directory where figures are saved (module subfolder is created under this)
  #     groups: Optional[Dict[str, Any]] = None,     # Composite definitions to allow semantic names in `variables`:
  #                                                  #   {"chl": "P1_Chl + P2_Chl + P4_Chl + P5_Chl"}    # string expression evaluated in ds namespace
  #                                                  #   {"phyto": ["P1_c", "P2_c", "P4_c", "P5_c"]}     # list/tuple summed elementwise
  #     styles: Optional[Dict[str, Dict[str, Any]]] = None,  # Optional per-variable style hints (e.g., line colors/labels used across panels)
  #     dpi: int = 150,                              # Output resolution (dots per inch) for the saved PNG
  #     figsize: tuple = (11, 9),                    # Figure size in inches (width, height)
  #     verbose: bool = False,                       # If True, print progress (time window, file path, etc.)
  # ) -> None:
  #     pass  # Function computes domain-wide surface/bottom/depth-avg series (+/- 1σ), plots 3 panels, SAVES a PNG; returns None
  
  # Output path pattern (per variable):
  #   <figures_root>/<basename(base_dir)>/timeseries/
  #     <prefix>__Domain__<VarOrGroup>__ThreePanel__<TimeLabel>__Timeseries.png
  #
  # where:
  #   <prefix>    = file_prefix(base_dir)
  #   <TimeLabel> = derived from months/years/start_date/end_date (AllTime, Jul, 2018, 2018-04–2018-10, ...)
  #
  # Notes:
  # - Each panel shows the mean line and a ±1 standard deviation envelope for the selected vertical slice
  #   (top panel = surface, middle = bottom, bottom = depth-averaged).
  # - Spatial mean is over the full domain; if an area field (e.g., 'art1') exists, area weighting is applied.
  # - Composites in `groups` let you pass semantic variables like "chl"/"phyto"/"zoo" without rewriting expressions.
  # - Works with Dask-chunked datasets; computation occurs during reductions/plotting.
  # - Returns None; to view in a notebook, display saved PNGs afterwards (e.g., using a small gallery cell).
  
  # Examples:
  
  # Domain three-panel — DOC (full run)
 domain_three_panel(
      ds=ds,
      variables=["DOC"],
      base_dir=BASE_DIR,
      figures_root=FIG_DIR,
      groups=GROUPS,
      styles=PLOT_STYLES,
      dpi=150,
      verbose=False,
  )
  
  #5) Station-specific Three-panel Figures (Surface ±1σ, Bottom ±1σ, Profile mean ±1σ) (1 figure with 3 subplots)
  # Full argument reference for station_three_panel(...)
  # Each parameter below is annotated with what it does and accepted values.
  # Produces a 3×1 figure per (station × variable) with:
  #   • Surface time series ±1σ (temporal σ at the station's nearest node)
  #   • Bottom  time series ±1σ (temporal σ at the station's nearest node)
  #   • Depth-averaged time series ±1σ (temporal σ at the station's nearest node)
  # Saves one PNG per (station × variable); returns None.
  
  # def station_three_panel(
  #     ds: xr.Dataset,                               # Xarray Dataset with FVCOM–ERSEM output (already opened/combined)
  #     variables: list[str],                         # One or more names: native vars (e.g., "temp") or composites (e.g., "chl") if provided in `groups`
  #     stations: List[Tuple[str, float, float]],     # Station metadata as (name, lat, lon) in WGS84 decimal degrees
  #                                                   #   - lon west of Greenwich is negative (e.g., -83.10)
  #                                                   #   - nearest model *node* is selected by great-circle distance (WGS84)
  #     *,                                            # Everything after this must be passed as keyword-only (safer, clearer)
  #     months=None,                                  # Calendar months to include (1–12) across all years; e.g., [7] or [4,5,6,7,8,9,10]; None = no month filter
  #     years=None,                                   # Calendar years to include; e.g., [2018] or [2018, 2019]; None = no year filter
  #     start_date=None,                              # Inclusive start date "YYYY-MM-DD"; used with end_date; None = no start bound
  #     end_date=None,                                # Inclusive end date   "YYYY-MM-DD"; used with start_date; None = no end bound
  #     base_dir: str,                                # Path to the model run folder; used for output subfolder and filename prefix
  #     figures_root: str,                            # Root directory where figures are saved (module subfolder is created under this)
  #     groups: Optional[Dict[str, Any]] = None,      # Composite definitions to allow semantic names in `variables`:
  #                                                   #   {"chl": "P1_Chl + P2_Chl + P4_Chl + P5_Chl"}    # string expression evaluated in ds namespace
  #                                                   #   {"phyto": ["P1_c", "P2_c", "P4_c", "P5_c"]}     # list/tuple summed elementwise
  #     styles: Optional[Dict[str, Dict[str, Any]]] = None,  # Optional per-variable style hints (e.g., line colors/labels used across panels)
  #     dpi: int = 150,                               # Output resolution (dots per inch) for the saved PNG
  #     figsize: tuple = (11, 9),                     # Figure size in inches (width, height)
  #     verbose: bool = False,                        # If True, print progress (resolved station index, time window, file path, etc.)
  # ) -> None:
  #     pass  # Function extracts nearest-node series per station, computes surface/bottom/depth-avg series, plots 3 panels with temporal ±1σ, SAVES PNG(s); returns None
  
  # Output path pattern (per station × variable):
  #   <figures_root>/<basename(base_dir)>/timeseries/
  #     <prefix>__Station-<Name>__<VarOrGroup>__ThreePanel__<TimeLabel>__Timeseries.png
  #
  # where:
  #   <prefix>    = file_prefix(base_dir)
  #   <Name>      = station name from `stations`
  #   <TimeLabel> = derived from months/years/start_date/end_date (AllTime, Jul, 2018, 2018-04–2018-10, ...)
  #
  # Notes:
  # - σ shading is *temporal* at stations (single grid node): the envelope reflects time-wise standard deviation around the mean line.
  # - Surface/bottom selections use the top/bottom sigma layers at the resolved nearest node; depth-avg is a vertical mean at that node.
  # - Composites in `groups` let you pass semantic variables like "chl"/"phyto"/"zoo" without rewriting expressions.
  # - Works with Dask-chunked datasets; computation occurs during reductions/plotting.
  # - Returns None; to view in a notebook, display the saved PNGs afterwards (e.g., using a small gallery cell).
  
  # Example: 
  
  # Station three-panel — chl at first station (full run)
  station_three_panel(
      ds=ds,
      variables=["chl"],
      stations=[STATIONS[0]],
      base_dir=BASE_DIR,
      figures_root=FIG_DIR,
      groups=GROUPS,
      styles=PLOT_STYLES,
      dpi=150,
      verbose=False,
  )
  
  
  # 6) Region three-panel 
  # Full argument reference for region_three_panel(...)
  # Each parameter below is annotated with what it does and accepted values.
  # Produces a 3×1 figure per (region × variable) with:
  #   • Surface time series ±1σ (SPATIAL σ across the region at each timestep)
  #   • Bottom  time series ±1σ (SPATIAL σ across the region at each timestep)
  #   • Depth-averaged time series ±1σ (SPATIAL σ across the region at each timestep)
  # Saves one PNG per (region × variable); returns None.
  
  # def region_three_panel(
  #     ds: xr.Dataset,                               # Xarray Dataset with FVCOM–ERSEM output (already opened/combined)
  #     variables: List[str],                         # One or more names: native vars (e.g., "temp") or composites (e.g., "chl") if provided in `groups`
  #     regions: List[Tuple[str, Dict[str, Any]]],    # List of region specs as (region_name, spec_dict).
  #                                                   #   spec_dict provides exactly ONE polygon source:
  #                                                   #     {"shapefile": "/path/to/region.shp"}                       # optional filtering:
  #                                                   #       + "name_field": "<FIELD>", "name_equals": "<VALUE>"
  #                                                   #     {"csv_boundary": "/path/to/boundary.csv"}                  # CSV boundary polygon
  #                                                   #       + "lon_col": "lon", "lat_col": "lat"                      # column names (defaults: lon/lat)
  #                                                   #       + "convex_hull": True|False                               # wrap scattered points
  #                                                   #       + "sort": "auto" | None                                   # attempt to order perimeter points
  #     *,                                            # Everything after this must be passed as keyword-only (safer, clearer)
  #     months=None,                                  # Calendar months to include (1–12) across all years; e.g., [7] or [4,5,6,7,8,9,10]; None = no month filter
  #     years=None,                                   # Calendar years to include; e.g., [2018] or [2018, 2019]; None = no year filter
  #     start_date=None,                              # Inclusive start date "YYYY-MM-DD"; used with end_date; None = no start bound
  #     end_date=None,                                # Inclusive end date   "YYYY-MM-DD"; used with start_date; None = no end bound
  #     base_dir: str,                                # Path to the model run folder; used for output subfolder and filename prefix
  #     figures_root: str,                            # Root directory where figures are saved (module subfolder is created under this)
  #     groups: Optional[Dict[str, Any]] = None,      # Composite definitions to allow semantic names in `variables`:
  #                                                   #   {"chl": "P1_Chl + P2_Chl + P4_Chl + P5_Chl"}    # string expression evaluated in ds namespace
  #                                                   #   {"phyto": ["P1_c", "P2_c", "P4_c", "P5_c"]}     # list/tuple summed elementwise
  #     styles: Optional[Dict[str, Dict[str, Any]]] = None,  # Optional per-variable style hints (e.g., line colors/labels used across panels)
  #     dpi: int = 150,                               # Output resolution (dots per inch) for the saved PNG
  #     figsize: tuple = (11, 9),                     # Figure size in inches (width, height)
  #     verbose: bool = False,                        # If True, print progress (masking details, time window, file path, etc.)
  # ) -> None:
  #     pass  # Function masks nodes/elements inside each region, computes regional surface/bottom/depth-avg means,
  #           # plots 3 panels with SPATIAL ±1σ envelopes per timestep, SAVES PNG(s); returns None
  
  # Output path pattern (per region × variable):
  #   <figures_root>/<basename(base_dir)>/timeseries/
  #     <prefix>__Region-<Name>__<VarOrGroup>__ThreePanel__<TimeLabel>__Timeseries.png
  #
  # where:
  #   <prefix>    = file_prefix(base_dir)
  #   <Name>      = region name from `regions`
  #   <TimeLabel> = derived from months/years/start_date/end_date (AllTime, Jul, 2018, 2018-04–2018-10, ...)
  #
  # Notes:
  # - Region mask is built on the FVCOM grid; only nodes/elements inside the polygon are included.
  # - If mesh connectivity `nv` is present, a strict element-inclusion rule (all three nodes inside) may be applied.
  # - If an area field (e.g., 'art1') exists, regional means are area-weighted; otherwise unweighted.
  # - The shaded ±1σ is spatial (spread across grid cells within the region at each time), unlike station_three_panel which uses temporal σ.
  # - Composites in `groups` let you pass semantic variables like "chl"/"phyto"/"zoo" without rewriting expressions.
  # - Works with Dask-chunked datasets; computation occurs during reductions/plotting.
  # - Returns None; to view in a notebook, display the saved PNGs afterwards (e.g., with a small gallery cell).
  
  # Example: 
  
  # Region three-panel — DOC in first region, Apr–Oct
  region_three_panel(
      ds=ds,
      variables=["DOC"],
      regions=[REGIONS[0]],
      months=[4,5,6,7,8,9,10],
      base_dir=BASE_DIR,
      figures_root=FIG_DIR,
      groups=GROUPS,
      styles=PLOT_STYLES,
      dpi=150,
      verbose=False,
  )
  
  
  # 7) Depth selection shorthand demos (sigma index, sigma value, absolute depth)
  
  # Examples: 
  
  # Domain — DOC at sigma layer index k=5, July
  domain_mean_timeseries(
      ds=ds,
      variables=["DOC"],
      depth=5,                  # == ("siglay_index", 5)
      months=[7],
      base_dir=BASE_DIR,
      figures_root=FIG_DIR,
      groups=GROUPS,
      styles=PLOT_STYLES,
      dpi=150,
      verbose=False,
  )
  
  
  # Example: Station — chl at sigma value s = -0.7 (in [-1, 0]), full run
  station_timeseries(
      ds=ds,
      variables=["chl"],
      stations=[STATIONS[0]],
      depth=-0.7,               # == ("sigma", -0.7)
      base_dir=BASE_DIR,
      figures_root=FIG_DIR,
      groups=GROUPS,
      styles=PLOT_STYLES,
      dpi=150,
      verbose=False,
  )
  
  
  # Example: Region — temperature at absolute depth z = -8 m, Apr–Oct 2018
  region_timeseries(
      ds=ds,
      variables=["temp"],
      regions=[REGIONS[0]],
      depth=-8.0,               # == ("z_m", -8.0)  (meters; negative = below surface)
      years=[2018],
      months=[4,5,6,7,8,9,10],
      base_dir=BASE_DIR,
      figures_root=FIG_DIR,
      groups=GROUPS,
      styles=PLOT_STYLES,
      dpi=150,
      verbose=False,
  )
  
  
  print(" Timeseries examples completed. Figures saved under:", FIG_DIR)
#
#------------------------------------------------------------------------------------
#            __  __                 
#           |  \/  |                
#           | \  / | __ _ _ __  ___ 
#           | |\/| |/ _` | '_ \/ __|
#           | |  | | (_| | |_) \__ \
#           |_|  |_|\__,_| .__/|___/
#                        | |        
#                        |_|        
#

# Maps Overview
# ---------------
# Two functions create plan-view maps from FVCOM–ERSEM output:
# - domain_map: full-model domain.
# - region_map: masked to polygons (shapefile or CSV).
#
# Both accept native or grouped variables, handle depth modes
# ("surface", "bottom", "depth_avg", sigma, or fixed z), and can
# plot time means or specific instants. Colour limits are chosen
# automatically (robust quantiles) or from your style settings.
#
# Key parameters:
# - variables: native names or GROUPS (e.g., "chl", "phyto").
# - time: window via months/years/date range or explicit instants.
# - depth: surface / bottom / depth_avg / sigma / fixed meters.
# - style: control cmap, norm, quantiles, overlays, figsize, dpi.
#
# Figures are saved under:
#   FIG_DIR/<basename(BASE_DIR)>/maps/
# Filenames encode scope, variable, depth, and time label.


if plot_maps:
  from fvcomersemviz.plots.maps import domain_map, region_map
  
  
  
  
  # 1) Domain maps
  
  # Full argument reference for domain_map(...)
  # Each parameter below is annotated with what it does and accepted values.
  # Renders plan-view maps over the FULL domain; saves PNGs; returns None.
  
  # def domain_map(
  #     ds: xr.Dataset,                              # Xarray Dataset with FVCOM–ERSEM output (already opened/combined)
  #     variables: List[str],                        # One or more names: native vars (e.g., "temp") or composites (e.g., "chl") if provided in `groups`
  #     *,                                           # Everything after this must be passed as keyword-only (safer, clearer)
  #     depth: Any,                                  # Vertical selection:
  #                                                  #   "surface" | "bottom" | "depth_avg"
  #                                                  #   int -> sigma layer index (e.g., 5 == k=5)
  #                                                  #   float in [-1, 0] -> sigma value (e.g., -0.7)
  #                                                  #   other float -> absolute depth z (meters, negative downward; e.g., -8.0 == 8 m below surface)
  #                                                  #   ("siglay_index", k) | ("sigma", s) | ("z_m", z)    # explicit tuple forms
  #                                                  #   {"z_m": z, "zvar": "z"}                            # dict form if vertical coord has non-default name
  #     months: Optional[List[int]] = None,          # Calendar months to include (1–12) across all years; e.g., [7] or [4,5,6,7,8,9,10]; None = no month filter
  #     years: Optional[List[int]] = None,           # Calendar years to include; e.g., [2018] or [2018, 2019]; None = no year filter
  #     start_date: Optional[str] = None,            # Inclusive start date "YYYY-MM-DD"; used with end_date; None = open start
  #     end_date: Optional[str] = None,              # Inclusive end date   "YYYY-MM-DD"; used with start_date; None = open end
  #     at_time: Optional[Any] = None,               # Single timestamp to render an instantaneous map; accepts str/np.datetime64/pd.Timestamp
  #     at_times: Optional[Sequence[Any]] = None,    # Multiple timestamps to render multiple instantaneous maps
  #     time_method: str = "nearest",                # Selection policy when matching requested instants to data: "nearest" (typical)
  #     base_dir: str,                               # Path to the model run folder; used for output subfolder and filename prefix
  #     figures_root: str,                           # Root directory where figures are saved (module subfolder "maps/" is created under this)
  #     groups: Optional[Dict[str, Any]] = None,     # Composite definitions enabling semantic names in `variables`:
  #                                                  #   {"chl": "P1_Chl + P2_Chl + P4_Chl + P5_Chl"}    # string expression evaluated in ds namespace
  #                                                  #   {"phyto": ["P1_c", "P2_c", "P4_c", "P5_c"]}     # list/tuple summed elementwise
  #     cmap: str = "viridis",                       # Default colormap (overridden per-variable by `styles`, if provided)
  #     clim: Optional[Tuple[float, float]] = None,  # Explicit (vmin, vmax). If None, uses `styles` vmin/vmax if set, else robust quantiles
  #     robust_q: Tuple[float, float] = (5, 95),     # Percentile limits (q_low, q_high) for robust autoscaling when no norm/vmin/vmax is set
  #     dpi: int = 150,                              # Output resolution (dots per inch) for saved PNG
  #     figsize: Tuple[float, float] = (8, 6),       # Figure size in inches (width, height)
  #     shading: str = "gouraud",                    # Tri shading mode: "gouraud" (node-centered) or "flat" (face-centered forced internally)
  #     grid_on: bool = False,                       # If True, overlay the triangular mesh lines on top of the map
  #     verbose: bool = False,                       # If True, print progress messages (selected times, paths, etc.)
  #     styles: Optional[Dict[str, Dict[str, Any]]] = None,  # Per-variable style overrides:
  #                                                  #   {"chl": {"cmap": "Greens", "vmin": 0, "vmax": 5},
  #                                                  #    "zoo": {"norm": LogNorm(1e-4, 1e0), "shading": "flat"}}
  # ) -> None:
  #     pass  # Function selects depth/time, evaluates variables/groups, chooses color limits, plots full-domain tri map(s), and SAVES PNG(s); returns None
  
  # Output path patterns:
  #   Mean over window:
  #     <figures_root>/<basename(base_dir)>/maps/
  #       <prefix>__Map-Domain__<VarOrGroup>__<DepthTag>__<TimeLabel>__Mean.png
  #   Instantaneous at time t:
  #       <prefix>__Map-Domain__<VarOrGroup>__<DepthTag>__<YYYY-MM-DDTHHMM>__Instant.png
  #
  # where:
  #   <prefix>    = file_prefix(base_dir)
  #   <DepthTag>  = derived from `depth` (Surface, Bottom, DepthAvg, SigmaK5, SigmaS0.7, Z8m, ...)
  #   <TimeLabel> = derived from months/years/start_date/end_date (AllTime, Jul, 2018, 2018-04–2018-10, ...)
  #
  # Notes:
  # - Node- vs element-centered variables are detected by presence of 'node' or 'nele' dims and plotted accordingly.
  # - If a normalization (`norm`, e.g., LogNorm) is provided via `styles`, it takes precedence over `clim`/robust quantiles.
  # - Absolute-depth selections use select_da_by_z(...) per variable; sigma selections use sigma coords.
  # - Returns None; to view in a notebook, display saved PNGs afterward (e.g., with the gallery cell).
  
  # Examples:
  # Domain mean maps at SURFACE — per-variable styles (DOC, chl, temp); July only
  domain_map(
      ds=ds,
      variables=["DOC", "chl", "temp"],
      depth="surface",
      months=[7],                         # July across all years
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      groups=GROUPS,
      styles=PLOT_STYLES,                 # per-var cmap/vmin/vmax/norm
      grid_on=True,                       # draw mesh overlay
      dpi=150, figsize=(8, 6),
      verbose=False,
  )
  
  # Domain instantaneous maps at BOTTOM (phyto) — two timestamps
  domain_map(
      ds=ds,
      variables=["phyto"],
      depth="bottom",
      at_times=["2018-06-15 00:00", "2018-09-01 12:00"],  # nearest match in data
      time_method="nearest",
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      groups=GROUPS,
      styles=PLOT_STYLES,
      grid_on=False,
      dpi=150, figsize=(8, 6),
      verbose=False,
  )
  
  # Domain mean at ABSOLUTE depth z = -8 m (phyto), Apr–Oct 2018
  domain_map(
      ds=ds,
      variables=["phyto"],
      depth=-8.0,                         # absolute depth in metres (negative downward)
      years=[2018],
      months=[4,5,6,7,8,9,10],
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      groups=GROUPS,
      styles=PLOT_STYLES,
      grid_on=True,
      dpi=150, figsize=(8, 6),
      verbose=False,
  )
  
  
  #2) Region maps
  
  # Full argument reference for region_map(...)
  # Each parameter below is annotated with what it does and accepted values.
  # Renders plan-view maps MASKED to polygon regions; saves PNGs; returns None.
  
  # def region_map(
  #     ds: xr.Dataset,                               # Xarray Dataset with FVCOM–ERSEM output (already opened/combined).
  #                                                  # Must include 'lon' and 'lat' for building region masks.
  #     variables: List[str],                         # One or more names: native vars (e.g., "temp") or composites (e.g., "chl") if provided in `groups`.
  #     regions: List[Tuple[str, Dict[str, Any]]],    # List of (region_name, spec_dict) entries. Each spec_dict provides EXACTLY ONE polygon source:
  #                                                   #   {"shapefile": "/path/to/region.shp"}                     # optional feature filter:
  #                                                   #       + "name_field": "<FIELD>", "name_equals": "<VALUE>"
  #                                                   #   {"csv_boundary": "/path/to/boundary.csv"}               # CSV boundary polygon
  #                                                   #       + "lon_col": "lon", "lat_col": "lat"                 # column names (defaults: lon/lat)
  #                                                   #       + "convex_hull": True|False                          # wrap scattered points into a hull
  #                                                   #       + "sort": "auto" | None                              # attempt to order perimeter points
  #     *,                                            # Everything after this must be passed as keyword-only (safer, clearer).
  #     depth: Any,                                   # Vertical selection:
  #                                                   #   "surface" | "bottom" | "depth_avg"
  #                                                   #   int -> sigma layer index (e.g., 5 == k=5)
  #                                                   #   float in [-1, 0] -> sigma value (e.g., -0.7)
  #                                                   #   other float -> absolute depth z (meters, negative downward; e.g., -8.0 == 8 m below surface)
  #                                                   #   ("siglay_index", k) | ("sigma", s) | ("z_m", z)          # explicit tuple forms
  #                                                   #   {"z_m": z, "zvar": "z"}                                  # dict form if vertical coord has non-default name
  #     months: Optional[List[int]] = None,           # Calendar months to include (1–12) across all years; e.g., [7] or [4,5,6,7,8,9,10]; None = no month filter.
  #     years: Optional[List[int]] = None,            # Calendar years to include; e.g., [2018] or [2018, 2019]; None = no year filter.
  #     start_date: Optional[str] = None,             # Inclusive start date "YYYY-MM-DD"; used with end_date; None = open start.
  #     end_date: Optional[str] = None,               # Inclusive end date   "YYYY-MM-DD"; used with start_date; None = open end.
  #     at_time: Optional[Any] = None,                # Single timestamp to render an instantaneous map; accepts str/np.datetime64/pd.Timestamp.
  #     at_times: Optional[Sequence[Any]] = None,     # Multiple timestamps to render multiple instantaneous maps.
  #     time_method: str = "nearest",                 # Selection policy when matching requested instants to data: "nearest" (typical).
  #     base_dir: str,                                # Path to the model run folder; used for output subfolder and filename prefix.
  #     figures_root: str,                            # Root directory where figures are saved (module subfolder "maps/" is created under this).
  #     groups: Optional[Dict[str, Any]] = None,      # Composite definitions enabling semantic names in `variables`:
  #                                                   #   {"chl": "P1_Chl + P2_Chl + P4_Chl + P5_Chl"}             # string expression evaluated in ds namespace
  #                                                   #   {"phyto": ["P1_c", "P2_c", "P4_c", "P5_c"]}              # list/tuple summed elementwise
  #     cmap: str = "viridis",                        # Default colormap (overridden per-variable by `styles`, if provided).
  #     clim: Optional[Tuple[float, float]] = None,   # Explicit (vmin, vmax). If None, uses `styles` vmin/vmax if set, else robust in-region quantiles.
  #     robust_q: Tuple[float, float]] = (5, 95),     # Percentile limits (q_low, q_high) for robust autoscaling when no norm/vmin/vmax is set.
  #     dpi: int = 150,                               # Output resolution (dots per inch) for saved PNG.
  #     figsize: Tuple[float, float]] = (8, 6),       # Figure size in inches (width, height).
  #     shading: str = "gouraud",                     # Tri shading mode: "gouraud" (node-centered) or "flat" (face-centered forced internally).
  #     grid_on: bool = False,                        # If True, overlay the triangular mesh lines on top of the map.
  #     verbose: bool = False,                        # If True, print progress messages (mask-building, selected times, paths, etc.).
  #     styles: Optional[Dict[str, Dict[str, Any]]] = None,  # Per-variable style overrides:
  #                                                   #   {"chl": {"cmap": "Greens", "vmin": 0, "vmax": 5},
  #                                                   #    "zoo": {"norm": LogNorm(1e-4, 1e0), "shading": "flat"}}
  # ) -> None:
  #     pass  # Function builds a region mask (nodes/elements), selects depth/time, evaluates variables/groups,
  #           # chooses color limits from in-region values, plots masked tri map(s), and SAVES PNG(s); returns None.
  
  # Output path patterns:
  #   Mean over window:
  #     <figures_root>/<basename(base_dir)>/maps/
  #       <prefix>__Map-Region-<Name>__<VarOrGroup>__<DepthTag>__<TimeLabel>__Mean.png
  #   Instantaneous at time t:
  #       <prefix>__Map-Region-<Name>__<VarOrGroup>__<DepthTag>__<YYYY-MM-DDTHHMM>__Instant.png
  #
  # where:
  #   <prefix>    = file_prefix(base_dir)
  #   <Name>      = region name from `regions`
  #   <DepthTag>  = derived from `depth` (Surface, Bottom, DepthAvg, SigmaK5, SigmaS0.7, Z8m, ...)
  #   <TimeLabel> = derived from months/years/start_date/end_date (AllTime, Jul, 2018, 2018-04–2018-10, ...)
  #
  # Notes:
  # - Node- vs element-centered variables are detected by presence of 'node' or 'nele' dims and plotted accordingly.
  # - Region masks: nodes are m
  
  # Examples:
  # Region=CENTRAL, depth-averaged mean (zoo with log norm), mesh overlay
  region_map(
      ds=ds,
      variables=["zoo"],
      regions=[REGIONS[0]],               # Central
      depth="depth_avg",
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      groups=GROUPS,
      styles=PLOT_STYLES,
      grid_on=True,
      dpi=150, figsize=(8, 6),
      verbose=False,
  )
  
  # Region=WEST, sigma selections: k=5 for DOC, s=-0.7 for chl
  region_map(
      ds=ds,
      variables=["DOC"],
      regions=[REGIONS[2]],               # West
      depth=5,                            # == ("siglay_index", 5)
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      groups=GROUPS,
      styles=PLOT_STYLES,
      grid_on=False,
      dpi=150, figsize=(8, 6),
      verbose=False,
  )
  region_map(
      ds=ds,
      variables=["chl"],
      regions=[REGIONS[2]],               # West
      depth=-0.7,                         # == ("sigma", -0.7)
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      groups=GROUPS,
      styles=PLOT_STYLES,
      grid_on=False,
      dpi=150, figsize=(8, 6),
      verbose=False,
  )
  
  # Region=EAST, ABSOLUTE z = -15 m, instantaneous (DOC)
  region_map(
      ds=ds,
      variables=["DOC"],
      regions=[REGIONS[1]],               # East
      depth=("z_m", -15.0),               # explicit absolute depth
      at_time="2018-08-15 00:00",
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      groups=GROUPS,
      styles=PLOT_STYLES,
      grid_on=True,
      dpi=150, figsize=(8, 6),
      verbose=False,
  )
  
  print(" Map examples completed. Figures saved under:", FIG_DIR)
#
#------------------------------------------------------------------------------------
# 
#            _    _ _   _                      _ _           
#           | |  | (_) (_)                    | | |          
#           | |__| | _____   ___ __ ___   ___ | | | ___ _ __ 
#           |  __  |/ _ \ \ / / '_ ` _ \ / _ \| | |/ _ \ '__|
#           | |  | | (_) \ V /| | | | | | (_) | | |  __/ |   
#           |_|  |_|\___/ \_/ |_| |_| |_|\___/|_|_|\___|_|   
#                                                 
                                                  
# Hovmöller Overview
# -------------------
# Show how variables evolve over time and depth at each station
# (nearest node/element to given lat/lon in STATIONS).
#
# Output path:
#   FIG_DIR/<basename(BASE_DIR)>/
# Filenames like:
#   <prefix>__Hovmoller-Station-<NAME>__<VAR>__sigma|z__<TimeLabel>.png
#
# Key options:
# - axis: "sigma" (native layers) or "z" (interpolated absolute depth, m)
# - variables: native or grouped (e.g., "temp", "chl", "phyto")
# - time window: months / years / start_date–end_date
# - styling: colormap, norm, vmin/vmax, robust limits, figsize, dpi
#
# Typical workflow:
# 1. Subset dataset in time.
# 2. Extract vertical column at each station.
# 3. Choose vertical axis ("sigma" or "z_levels").
# 4. Plot time × depth using pcolormesh and save figure.

if plot_hovmoller:
  from fvcomersemviz.plots.hovmoller import station_hovmoller
  
  # Station Hovmoller
  # Full argument reference for station_hovmoller(...)
  # Each parameter below is annotated with what it does and accepted values.
  # Produces time × depth (Hovmöller) plots at STATIONS; saves PNGs; returns None.
  
  # def station_hovmoller(
  #     ds: xr.Dataset,                               # Xarray Dataset with FVCOM–ERSEM output (already opened/combined)
  #     variables: List[str],                         # One or more names: native vars (e.g., "temp") or composites (e.g., "chl") if provided in `groups`
  #     stations: List[Tuple[str, float, float]],     # Station metadata as (name, lat, lon) in WGS84 decimal degrees
  #                                                   #   - lon west of Greenwich is negative (e.g., -83.10)
  #                                                   #   - nearest model node/element is selected by great-circle distance (WGS84)
  #     *,                                            # Everything after this must be passed as keyword-only (safer, clearer)
  #     axis: str = "z",                              # Vertical axis for the plot:
  #                                                   #   "sigma" -> y-axis is sigma layers (unitless), no interpolation
  #                                                   #   "z"     -> y-axis is absolute depth in meters (negative downward), σ-profiles interpolated to `z_levels`
  #     z_levels: Optional[np.ndarray] = None,        # Regular depth levels (ascending, e.g., np.linspace(-30, 0, 61)) used when axis="z".
  #                                                   # If None, levels are auto-built from the station column’s min depth to 0 m.
  #     months: Optional[List[int]] = None,           # Calendar months to include (1–12) across all years; e.g., [7] or [4,5,6,7,8,9,10]; None = no month filter
  #     years: Optional[List[int]] = None,            # Calendar years to include; e.g., [2018] or [2018, 2019]; None = no year filter
  #     start_date: Optional[str] = None,             # Inclusive start date "YYYY-MM-DD"; used with end_date; None = open start
  #     end_date: Optional[str] = None,               # Inclusive end date   "YYYY-MM-DD"; used with start_date; None = open end
  #     base_dir: str,                                # Path to the model run folder; used for output subfolder and filename prefix
  #     figures_root: str,                            # Root directory where figures are saved (module subfolder, e.g., "hovmoller/", is created under this)
  #     groups: Optional[Dict[str, Any]] = None,      # Composite definitions enabling semantic names in `variables`:
  #                                                   #   {"chl": "P1_Chl + P2_Chl + P4_Chl + P5_Chl"}    # string expression evaluated in ds namespace
  #                                                   #   {"phyto": ["P1_c", "P2_c", "P4_c", "P5_c"]}     # list/tuple summed elementwise
  #     cmap: str = "viridis",                        # Default colormap (overridden per-variable by `styles`, if provided)
  #     vmin: Optional[float] = None,                 # Explicit lower color limit (ignored if a normalization `norm` is provided via `styles`)
  #     vmax: Optional[float] = None,                 # Explicit upper color limit (ignored if a normalization `norm` is provided via `styles`)
  #     dpi: int = 150,                               # Output resolution (dots per inch) for saved PNG
  #     figsize: tuple = (9, 5),                      # Figure size in inches (width, height)
  #     verbose: bool = True,                         # If True, print progress (resolved station index, axis type, time window, file path, etc.)
  #     styles: Optional[Dict[str, Dict[str, Any]]] = None,  # Per-variable style overrides:
  #                                                   #   {"chl": {"cmap": "Greens", "vmin": 0, "vmax": 5}}
  #                                                   #   {"DOC": {"cmap": "viridis"}}
  #                                                   #   {"zoo": {"norm": LogNorm(1e-4, 1e0)}}  # norm takes precedence over vmin/vmax
  # ) -> None:
  #     pass  # Function time-filters ds, resolves nearest node/element per station, builds (time × sigma) or
  #           # interpolated (time × z) arrays, chooses color limits (norm -> explicit -> robust), plots pcolormesh,
  #           # and SAVES PNG(s); returns None.
  
  # Output path pattern (per station × variable × axis):
  #   <figures_root>/<basename(base_dir)>/hovmoller/
  #     <prefix>__Hovmoller-Station-<Name>__<VarOrGroup>__sigma|z__<TimeLabel>.png
  #
  # where:
  #   <prefix>    = file_prefix(base_dir)
  #   <Name>      = station name from `stations`
  #   <TimeLabel> = derived from months/years/start_date/end_date (AllTime, Jul, 2018, 2018-04–2018-10, ...)
  #
  # Notes:
  # - For axis="sigma": plots native σ layers; fastest, no vertical interpolation.
  # - For axis="z": vertical coordinates are built (ensure_z_from_sigma); σ-profiles are interpolated to `z_levels`.
  # - Station location: nearest grid node (or element) is chosen via great-circle distance in WGS84.
  # - If no explicit vmin/vmax/norm, limits are chosen robustly from the plotted data.
  # - Returns None; to view in a notebook, display saved PNGs afterward (e.g., using a small gallery cell).
  
  # Examples:
  
  # WE12 — chlorophyll on sigma layers (full run, robust colour limits)
  station_hovmoller(
      ds=ds,
      variables=["chl"],
      stations=[STATIONS[0]],               # e.g., ("WE12", 41.90, -83.10)
      axis="sigma",
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      groups=GROUPS,
      styles=PLOT_STYLES,                   # per-var cmap/norm/vmin/vmax if set
  )
  
  # WE12 — DOC on absolute depth z (Apr–Oct 2018), explicit z grid
  station_hovmoller(
      ds=ds,
      variables=["DOC"],
      stations=[STATIONS[0]],
      axis="z",
      z_levels=np.linspace(-20.0, 0.0, 60), # omit to auto-build from column depth
      months= [4, 5, 6, 7, 8, 9, 10],   # Apr–Oct
      years  = [2018],
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      groups=GROUPS,
      styles=PLOT_STYLES,
  )
  
  # WE13 — zooplankton on sigma layers (Apr–Oct 2018)
  station_hovmoller(
      ds=ds,
      variables=["zoo"],
      stations=[STATIONS[1]],
      axis="sigma",
      months= [4, 5, 6, 7, 8, 9, 10],   # Apr–Oct
      years  = [2018],
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      groups=GROUPS,
      styles=PLOT_STYLES,
  )
  
  print(" Hovmöller examples completed. Figures saved under:", FIG_DIR)
  
  
#
#------------------------------------------------------------------------------------
# 
#             _____ _        _      _     _                      _                _  _______  ______    _____     _____  
#            / ____| |      (_)    | |   (_)                    | |              | |/ /  __ \|  ____|  / /__ \   |__ \ \ 
#           | (___ | |_ ___  _  ___| |__  _  ___  _ __ ___   ___| |_ _ __ _   _  | ' /| |  | | |__    | |   ) /\/\  ) | |
#            \___ \| __/ _ \| |/ __| '_ \| |/ _ \| '_ ` _ \ / _ \ __| '__| | | | |  < | |  | |  __|   | |  / />  < / /| |
#            ____) | || (_) | | (__| | | | | (_) | | | | | |  __/ |_| |  | |_| | | . \| |__| | |____  | | / /_\/\// /_| |
#           |_____/ \__\___/|_|\___|_| |_|_|\___/|_| |_| |_|\___|\__|_|   \__, | |_|\_\_____/|______| | ||____|  |____| |
#                                                                          __/ |                       \_\           /_/ 
#                                                                         |___/                                          

# KDE Stoichiometry Plots
# ------------------------
# Creates a 2×2 density figure showing N:C and P:C ratios vs a chosen variable
# at surface and bottom depths:
#   [surface N:C vs var]  [surface P:C vs var]
#   [bottom  N:C vs var]  [bottom  P:C vs var]
#
# Samples are pooled over the selected time window and optional region.
# Output path:
#   FIG_DIR/<basename(BASE_DIR)>/
# Filenames like:
#   <prefix>__KDE-Stoich__<Group>__<Variable>__<RegionTag>__<TimeLabel>.png
#
# Main options:
# - group: biological group (e.g., "P5"); uses <group>_NC and <group>_PC.
# - variable: native or grouped variable (y-axis).
# - region: optional polygon mask from REGIONS.
# - time: months / years / start_date–end_date.
# - method: "kde" (accurate) or "hist" (fast); control smoothness via bw/hist_sigma.
# - style: cmap, vmin/vmax, scatter_underlay, figsize, dpi.
#
# Typical workflow:
# 1. Subset time and (optionally) region.
# 2. Extract surface and bottom slices for variable and stoichiometric ratios.
# 3. Build density grids (KDE or hist+blur) and plot 2×2 panels.
# 4. Save figure; empty panels are skipped automatically.



# --- Stoichiometry KDE (2×2) examples:  ---

if plot_kde:
  from fvcomersemviz.plots.kde_stoichiometry import kde_stoichiometry_2x2
  
  # kde_stoichiometry_2x2
  
  # Full argument reference for kde_stoichiometry_2x2(...)
  # Each parameter below is annotated with what it does and accepted values.
  # Builds a single 2×2 figure of density plots:
  #   [surface N:C vs <variable>]  [surface P:C vs <variable>]
  #   [bottom  N:C vs <variable>]  [bottom  P:C vs <variable>]
  # Samples are pooled over time×space within an optional region and time window; panels with too
  # few finite samples are skipped. The figure is SAVED to disk; function returns None.
  
  # def kde_stoichiometry_2x2(
  #     ds: xr.Dataset,                                 # Xarray Dataset with FVCOM–ERSEM output (opened/combined)
  #     *,                                              # Everything after this must be passed as keyword-only (safer, clearer)
  #     group: str,                                     # ERSEM functional group tag, e.g. "P5".
  #                                                     # The function expects native stoichiometry fields "<group>_NC" and "<group>_PC".
  #     variable: str,                                  # Y-axis variable: native name (e.g., "P5_c", "chl_a")
  #                                                     # or a composite defined in `groups` (e.g., "chl", "phyto", "DOC").
  #     region: Optional[Tuple[str, Dict[str, Any]]] = None,  # Optional spatial mask as (name, spec). If None → full domain.
  #                                                     # spec options (one of):
  #                                                     #   {"shapefile": "/path/to/region.shp"} [+ optional "name_field","name_equals"]
  #                                                     #   {"csv_boundary": "/path/to/boundary.csv"} [+ "lon_col","lat_col","convex_hull","sort"]
  #     months: Optional[List[int]] = None,             # Calendar months (1–12) to include; e.g., [6,7,8] for JJA; None = no month filter.
  #     years: Optional[List[int]] = None,              # Calendar years to include; e.g., [2018] or [2018,2019]; None = no year filter.
  #     start_date: Optional[str] = None,               # Inclusive start date "YYYY-MM-DD"; used with end_date; None = open start.
  #     end_date: Optional[str] = None,                 # Inclusive end date   "YYYY-MM-DD"; used with start_date; None = open end.
  #     base_dir: str,                                  # Path to model run folder; used for filename prefix generation.
  #     figures_root: str,                              # Root folder where figures are saved (subfolder "kde_stoichiometry/" is created).
  #     groups: Optional[Dict[str, Any]] = None,        # Composite definitions enabling semantic names in `variable`, e.g.:
  #                                                     #   {"chl": "P1_Chl + P2_Chl + P4_Chl + P5_Chl"}
  #                                                     #   {"phyto": ["P1_c","P2_c","P4_c","P5_c"]}   # list/tuple summed elementwise
  #     dpi: int = 150,                                 # Output resolution for the saved PNG.
  #     figsize: Tuple[float, float]] = (11, 9),        # Figure size (inches): width, height.
  #     cmap: str = "viridis",                          # Default colormap for density (can be overridden per variable via `styles`).
  #     grids: int = 100,                               # Grid resolution for density evaluation (higher = more detail, slower).
  #     bw_method: Optional[float | str] = "scott",     # KDE bandwidth ("scott", "silverman", or float scalar); ignored if method="hist".
  #     min_samples: int = 200,                         # Minimum number of finite (x,y) pairs required to render a panel; otherwise it’s skipped.
  #     scatter_underlay: int = 0,                      # If >0, plot up to N random raw points under the density for context (thin black dots).
  #     verbose: bool = False,                          # If True, print progress, panel skips, output path.
  #     styles: Optional[Dict[str, Dict[str, Any]]] = None,  # Per-variable style overrides, e.g.:
  #                                                     #   {"P5_c": {"cmap": "magma", "vmin": 0.0, "vmax": 100.0}}
  #                                                     # vmin/vmax (if given) are applied to the Y-axis (the chosen `variable`) for nicer limits.
  #     # --- performance/approximation controls ---
  #     method: Literal["kde", "hist"] = "kde",         # "kde" = Gaussian KDE (accurate), "hist" = 2D histogram + Gaussian blur (much faster).
  #     sample_max: Optional[int] = 200_000,            # Optional cap on pooled pairs before density; random subsample for speed on huge datasets.
  #     hist_sigma: float = 1.2,                        # Gaussian blur (in bins) when method="hist" (controls smoothness).
  #     random_seed: Optional[int] = 12345,             # RNG seed for reproducible subsampling/underlay selection.
  # ) -> None:
  #     pass  # Function filters by time, selects surface/bottom slices, builds optional region mask, pools samples,
  #           # computes 2D density for (NC:C vs variable) and (PC:C vs variable) at surface & bottom,
  #           # renders a 2×2 figure, and SAVES it; returns None.
  
  # Output path pattern:
  #   <figures_root>/<basename(base_dir)>/kde_stoichiometry/
  #     <prefix>__KDE-Stoich__<Group>__<Variable>__<RegionTag>__<TimeLabel>.png
  #
  # where:
  #   <prefix>    = file_prefix(base_dir)
  #   <RegionTag> = region name or "Domain" if region=None
  #   <TimeLabel> = derived from months/years/start_date/end_date (AllTime, JJA, 2018, 2018-04–2018-10, ...)
  #
  # Notes:
  # - Panels are computed from internally selected "surface" and "bottom" slices.
  # - Center-aware masking aligns node/element data before pooling; only in-region samples contribute when a region is set.
  # - When `styles[variable]["vmin"/"vmax"]` exist, they’re applied to the y-axis for consistent comparisons.
  # - If every panel has < min_samples (after filtering/masking), no file is saved and a verbose message is printed instead.
  
  # Fast/default options
  FAST = dict(
      method="kde",          # "kde" (accurate) or "hist" (very fast on huge datasets)
      sample_max=150_000,    # cap pooled pairs for speed
      hist_sigma=1.2,        # blur (bins) if method="hist"
      grids=100,             # density grid resolution
      bw_method="scott",     # KDE bandwidth (ignored if method="hist")
      verbose=False,
  )
  # Examples:
  
  # DOMAIN • JJA • group=P5 • variable=P5_c
  kde_stoichiometry_2x2(
      ds=ds,
      group="P5",
      variable="P5_c",
      region=None,                     # full domain
      months=[6,7,8], years=None,      # Jun–Aug across all years
      base_dir=BASE_DIR, figures_root=FIG_DIR, groups=GROUPS,
      min_samples=200, scatter_underlay=800,
      styles=PLOT_STYLES if "PLOT_STYLES" in globals() else None,
      FAST,
  )
  
  # REGION • Apr–Oct 2018 • group=P5 • variable=phyto (composite)
  if "REGIONS" in globals() and REGIONS:
      kde_stoichiometry_2x2(
          ds=ds,
          group="P5",
          variable="phyto",                 # composite from GROUPS
          region=REGIONS[0],                # e.g., ("Central", {...})
          months=[4,5,6,7,8,9,10], years=[2018],
          base_dir=BASE_DIR, figures_root=FIG_DIR, groups=GROUPS,
          min_samples=200, scatter_underlay=1200,
          styles=PLOT_STYLES if "PLOT_STYLES" in globals() else None,
          FAST,
      )
  
  # DOMAIN • full run • group=P5 • variable=chl (composite)
  kde_stoichiometry_2x2(
      ds=ds,
      group="P5",
      variable="chl",
      region=None,
      months=None, years=None,         # full time span
      base_dir=BASE_DIR, figures_root=FIG_DIR, groups=GROUPS,
      min_samples=300, scatter_underlay=1500,
      styles=PLOT_STYLES if "PLOT_STYLES" in globals() else None,
      FAST,
  )
  
  # REGION COMPARISON • JJA 2018 • group=P5 • variable=P5_c (first two regions if available)
  if "REGIONS" in globals() and len(REGIONS) >= 2:
      for reg in REGIONS[:2]:
          kde_stoichiometry_2x2(
              ds=ds,
              group="P5",
              variable="P5_c",
              region=reg,
              months=[6,7,8], years=[2018],
              base_dir=BASE_DIR, figures_root=FIG_DIR, groups=GROUPS,
              min_samples=180, scatter_underlay=800,
              styles=PLOT_STYLES if "PLOT_STYLES" in globals() else None,
              FAST,
          )
  
  print(" KDE stoichiometry examples completed. Figures saved under:", FIG_DIR)
  
#
#------------------------------------------------------------------------------------ 
#             _____                          
#            / ____|                         
#           | |    _   _ _ ____   _____  ___ 
#           | |   | | | | '__\ \ / / _ \/ __|
#           | |___| |_| | |   \ V /  __/\__ \
#            \_____\__,_|_|    \_/ \___||___/
#                                           
                                            

# Curves (x–y Diagnostics)
# ------------------------
# Visualize relationships between two model variables (Y vs X)
# after applying time filters, spatial scope (domain / region / station),
# and depth selection. Useful for exploring responses (e.g., chl vs PAR,
# phyto vs temp, DOC vs MLD) and comparing regions, depths, or seasons.
#
# Function: plot_curves(specs=[...], ds=..., groups=...)
# Each spec defines one curve with:
#   x, y: variable names or expressions (support GROUPS and algebra)
#   filters: months / years / dates / predicates (where)
#   depth: "surface", "bottom", "depth_avg", sigma, or fixed z
#   scope: domain | region | station
#   bin: binned median + IQR (robust trend)
#   scatter: raw sample cloud (context)
#   style: color, linewidth, etc.
#
# Output path:
#   FIG_DIR/<basename(BASE_DIR)>/curves/
# Filenames encode scope, depth, time, and variable pairing.
#
# Interpretation:
# - Rising/falling curves → limitation or inhibition
# - Plateau → saturation
# - Hump → optimum response
# - Wide IQR → variability or mixed regimes
#
# Tips:
# - Combine binned + scatter for clarity and context.
# - Keep consistent filters, depths, and styles across comparisons.
# - Log-scale axes for wide dynamic ranges.
# - Ensure bins have enough samples; adjust x_bins or min_count.



#Examples:

if plot_curves:
from fvcomersemviz.plots.curves import plot_curves

  # We use different groups / variables for this example - so we will need to update the GROUPS dictionary
  GROUPS = {
      # Aliases (nice short names you’ll use in specs)
      "PAR": "light_parEIR",
      "DIN": "N3_n + N4_n",
  
      # Composites (elementwise sums)
      "chl_total":     "P1_Chl + P2_Chl + P4_Chl + P5_Chl",
      "phyto_c_total": "P1_c  + P2_c  + P4_c  + P5_c",
  
      # Derived metrics (safe algebra; add epsilons to avoid divide-by-zero)
      "P5_spec_prod": "P5_Cfix / (P5_c + 1e-12)",
  
      # Predicates (boolean expressions you can reuse in `filters.where`)
      "PAR_pos": "light_parEIR > 0",
  }
  
  # plot_curves
  # Full argument reference for plot_curves(...)
  # Renders one figure containing one or more “curves” describing y vs x relationships,
  
  # def plot_curves(
  #     specs: Sequence[Dict[str, Any]],                # Plot_curves requires the user to build a specs dictionary (one dict per curve).
  #                                                     # Each spec supports:
  #                                                     #   {
  #                                                     #     "name": "label for legend",
  #                                                     #     "x": "<expr or alias>",                 # resolvable via `groups` and tolerant names
  #                                                     #     "y": "<expr or alias>",
  #                                                     #     "filters": {                           # optional time/predicate filters
  #                                                     #        "months": [6,7,8], "years": [2018],
  #                                                     #        "start": "YYYY-MM-DD", "end": "...",
  #                                                     #        "where": "<boolean expr>"            # e.g., "light_parEIR > 0"
  #                                                     #     },
  #                                                     #     "depth": "surface"|"bottom"|"depth_avg"|float sigma|{"z_m": -10},
  #                                                     #     "scope": {                              # choose exactly one or none:
  #                                                     #        "region": (name, spec)               # e.g., ("Central", {"shapefile": ".../central.shp"})
  #                                                     #        # or
  #                                                     #        "station": (name, lat, lon)          # nearest-node column extract
  #                                                     #     },
  #                                                     #     "bin": {                                # to draw a robust trend line
  #                                                     #        "x_bins": 40, "agg": "median"|"mean"|"pXX",
  #                                                     #        "min_count": 10, "iqr": True         # show IQR band if True
  #                                                     #     },
  #                                                     #     # If "bin" omitted → raw scatter
  #                                                     #     "scatter": {"alpha": 0.15, "s": 4},     # when plotting scatter
  #                                                     #     "style": {...},                         # Matplotlib kwargs (color, lw, etc.)
  #                                                     #     "aliases": {"PAR": "light_parEIR"}      # optional per-spec alias map
  #                                                     #   }
  #     *,
  #     ds: xr.Dataset,                                  # FVCOM–ERSEM dataset (already opened/combined).
  #     groups: Optional[Dict[str, Any]] = None,         # Global alias/composite/derived expressions used by specs, e.g.:
  #                                                       # {"chl_total": "P1_Chl + P2_Chl + P4_Chl",
  #                                                       #  "DIN": "N3_n + N4_n"}
  #     # --- axes labels / legend ---
  #     xlabel: Optional[str] = None,                    # Force x-axis label; default picks from first spec ("x_label" or "x").
  #     ylabel: Optional[str] = None,                    # Force y-axis label; default picks from first spec ("y_label" or "y").
  #     show_legend: bool = True,                        # Toggle legend.
  #     legend_outside: bool = True,                     # If True, place legend outside (right); else use "best".
  #     legend_fontsize: int = 8,                        # Legend font size.
  #     verbose: bool = False,                           # Print resolution/filters/where errors in a tolerant way.
  #     # --- figure creation + saving (ALWAYS saves) ---
  #     base_dir: str,                                   # Model run root; used by file_prefix() and output path builder.
  #     figures_root: str,                               # Root folder where figures are written (package will create subfolders).
  #     stem: Optional[str] = None,                      # Optional filename stem override. If None, a stem is auto-built from:
  #                                                       #   ScopeTag (Domain|Region-<N>|Station-<N>|MultiScope),
  #                                                       #   DepthTag (surface|bottom|depth_avg|z-XXm|AllDepth|MixedDepth),
  #                                                       #   TimeLabel (JJA|2018|YYYY-MM–YYYY-MM|AllTime|MixedTime),
  #                                                       #   Content ("<X>_vs_<Y>" and "Ncurves" if >1 spec).
  #     dpi: int = 150,                                  # Output PNG resolution.
  #     figsize: Tuple[float, float]] = (7.2, 4.6),      # Figure size in inches.
  #     constrained_layout: bool = True,                 # Use constrained layout when creating the figure.
  # ) -> str:
  #     pass  # Internals:
  #           # 1) For each spec: filter time → apply scope → select depth → resolve x/y (with tolerant names, groups, aliases)
  #           #    → optional 'where' predicate → align/flatten → produce binned stats (median+IQR) or scatter payload.
  #           # 2) Create a figure and axes; draw each curve with distinct colors from the rc cycle.
  #           # 3) Auto-label axes if not provided; place legend (outside by default).
  #           # 4) Build output folder via fvcomersemviz.utils.out_dir(). Subdir behavior:
  #           #      - If FVCOM_PLOT_SUBDIR is set (e.g., "project" or ""), it is respected.
  #           #      - Else defaults to "curves", so files go under .../<basename(BASE_DIR)>/curves/.
  #           # 5) Build filename:
  #           #      <prefix>__Curves__<ScopeTag>__<DepthTag>__<TimeLabel>__<Content>.png
  #           #    or, if `stem` provided: <prefix>__Curves__<stem>.png
  #           # 6) Save the PNG and return the full path.
  
  # Output path pattern:
  #   <figures_root>/<basename(base_dir)>/<subdir>/
  #     <prefix>__Curves__<ScopeTag>__<DepthTag>__<TimeLabel>__<Content>.png
  #
  # where:
  #   <prefix>     = file_prefix(base_dir)
  #   <subdir>     = env FVCOM_PLOT_SUBDIR if set; otherwise "curves"
  #   <ScopeTag>   = Domain | Region-<Name> | Station-<Name> | MultiScope
  #   <DepthTag>   = surface | bottom | depth_avg | z-10m | AllDepth | MixedDepth
  #   <TimeLabel>  = built from months/years/start/end (e.g., JJA, 2018, 2018-04–2018-10, AllTime, MixedTime)
  #   <Content>    = "<X>_vs_<Y>" from the first spec plus "Ncurves" if multiple curves are shown
  #
  # Notes:
  # - If a spec requests "bin", a robust median curve is drawn with optional IQR shading; otherwise raw scatter is used.
  # - Variable resolution is tolerant to case/underscores and can evaluate algebraic expressions via `groups`.
  # - A failing 'where' expression is safely ignored with a verbose note if `verbose=True`.
  # - The function always saves and returns the output PNG path; you do not need to manage axes or saving yourself.
  
  
  #    Key fields:
  #      - name: legend label for the curve
  #      - x, y: variables/expressions (can use GROUPS keys like "PAR", "chl_total", etc.)
  #      - filters: months/years/start/end + optional 'where' (can use GROUPS predicates like "PAR_pos")
  #      - depth: "surface" | "bottom" | "depth_avg" | float sigma | {"z_m": -10}
  #      - scope: {"region": (name, spec)} | {"station": (name, lat, lon)} | {}
  #      - Choose ONE of:  bin={...}  OR  scatter={...}
  #      - style: optional matplotlib kwargs (color/linestyle/marker/etc.)
  
  # Example 1 — Region vs Region (binned median + IQR), surface, JJA 2018, daylight only
  
  # Define the spec
  specs_light_chl = [
      {
          "name": "Central",
          "x": "PAR",               # alias -> light_parEIR
          "y": "chl_total",         # composite chlorophyll
          "filters": {"months": [6,7,8], "years": [2018], "where": "PAR_pos"},
          "depth": "surface",
          "scope": {"region": REGIONS[0]},
          "bin": {"x_bins": 40, "agg": "median", "min_count": 20, "iqr": True},
          "style": {"color": "C0"},
          "x_label": "PAR (EIR)",
          "y_label": "Total chlorophyll",
      },
      {
          "name": "East",
          "x": "PAR",
          "y": "chl_total",
          "filters": {"months": [6,7,8], "years": [2018], "where": "PAR_pos"},
          "depth": "surface",
          "scope": {"region": REGIONS[1]},
          "bin": {"x_bins": 40, "agg": "median", "min_count": 20, "iqr": True},
          "style": {"color": "C3"},
      },
  ]
  
  # Plot
  plot_curves(
      specs=specs_light_chl, ds=ds, groups=GROUPS,
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      dpi=150,
  )
  
  # Example 2 — Domain, depth-avg, Apr–Oct 2018 (binned)
  
  # Spec
  specs_temp_phyto = [{
      "name": "Domain",
      "x": "temp",
      "y": "phyto_c_total",
      "filters": {"months": [4,5,6,7,8,9,10], "years": [2018]},
      "depth": "depth_avg",
      "scope": {},  # domain
      "bin": {"x_bins": 32, "agg": "median", "min_count": 20, "iqr": True},
      "style": {"color": "C2"},
      "x_label": "Temperature (°C)",
      "y_label": "Total phytoplankton C",
  }]
  
  # Plot
  plot_curves(
      specs=specs_temp_phyto, ds=ds, groups=GROUPS,
      base_dir=BASE_DIR, figures_root=FIG_DIR, dpi=150,
  )
  
  # Example 3 — Domain, depth-avg, Apr–Oct 2018 (scatter cloud)
  
  # Spec
  specs_temp_prod_scatter = [{
      "name": "Domain",
      "x": "temp",
      "y": "P5_spec_prod",   # derived metric from GROUPS
      "filters": {"months": [4,5,6,7,8,9,10], "years": [2018]},
      "depth": "depth_avg",
      "scope": {},
      "scatter": {"s": 3, "alpha": 0.12},
      "style": {"marker": ".", "linewidths": 0},
      "x_label": "Temperature (°C)",
      "y_label": "P5 specific production (Cfix / C)",
  }]
  
  # Plot
  plot_curves(
      specs=specs_temp_prod_scatter, ds=ds, groups=GROUPS,
      base_dir=BASE_DIR, figures_root=FIG_DIR, dpi=150,
  )
  
  # In order to show both the binned median (backbone) and the scatter cloud on the same graph,
  # we include two specs in the same list: one with "scatter" for the raw points,
  # and one with "bin" for the aggregated median + IQR curve.
  
  # Example 4: Binned backbone + scatter context (same x/y, same filters/scope/depth) ---
  
  # Spec
  specs_par_chl_backbone = [
      # 1) Scatter cloud for context (drawn first → under the line)
      {
          "name": "All points",
          "x": "PAR",                       # alias from GROUPS → light_parEIR
          "y": "chl_total",                 # composite from GROUPS
          "filters": {"months": [6,7,8], "years": [2018], "where": "PAR_pos"},
          "depth": "surface",
          "scope": {},                      # domain
          "scatter": {"s": 6, "alpha": 0.08},
          "style": {"marker": ".", "linewidths": 0, "color": "red"},
          "x_label": "PAR (EIR)",
          "y_label": "Total chlorophyll",
      },
      # 2) Binned median + IQR “backbone” (drawn second → on top)
      {
          "name": "Median (IQR)",
          "x": "PAR",
          "y": "chl_total",
          "filters": {"months": [6,7,8], "years": [2018], "where": "PAR_pos"},
          "depth": "surface",
          "scope": {},
          "bin": {"x_bins": 40, "agg": "median", "min_count": 20, "iqr": True},
          "style": {"color": "blue", "lw": 2},
      },
  ]
  
  # Plot
  plot_curves(
      specs=specs_par_chl_backbone,
      ds=ds,
      groups=GROUPS,
      base_dir=BASE_DIR,
      figures_root=FIG_DIR,
      dpi=150,
      legend_outside=True,
  )
  
  
  print(" Curve examples completed. Figures saved under:", FIG_DIR)

#
#------------------------------------------------------------------------------------ 
#
#                           _                 _   _                 
#               /\         (_)               | | (_)                
#              /  \   _ __  _ _ __ ___   __ _| |_ _  ___  _ __  ___ 
#             / /\ \ | '_ \| | '_ ` _ \ / _` | __| |/ _ \| '_ \/ __|
#            / ____ \| | | | | | | | | | (_| | |_| | (_) | | | \__ \
#           /_/    \_\_| |_|_|_| |_| |_|\__,_|\__|_|\___/|_| |_|___/
#                                                                   
                                                                   
# Animations
# -----------
# The fvcomersemviz package can animate time series and maps
#
# - Timeseries animations:
#   Draw a growing line through the selected time window.
#   Supports single or multi-line plots (by variable, region, or station).
#   Depth options are the same as for static time-series.
#
# - Map animations:
#   Show how a variable or group evolves spatially over time.
#   Depth and time filters follow the same logic as static maps.

if plot_animate:
  from fvcomersemviz.plots.animate import animate_timeseries, animate_maps
  # Reset our groups and our plot styles
  
  GROUPS = {
      "DOC":   "R1_c + R2_c + R3_c + T1_30d_c + T2_30d_c",  # dissolved organic carbon (sum of pools)
      "phyto": ["P1_c", "P2_c", "P4_c", "P5_c"],            # total phytoplankton carbon (sum)
      "zoo":   ["Z4_c", "Z5_c", "Z6_c"],                    # total zooplankton carbon (sum)
      "chl":   "P1_Chl + P2_Chl + P4_Chl + P5_Chl",         # total chlorophyll (sum)
  }
  
  PLOT_STYLES = {
      "temp":   {"line_color": "lightblue", "cmap": "coolwarm"},
      "DOC":   {"line_color": "blue", "cmap": "viridis"},
      "chl":   {"line_color": "lightgreen", "cmap": "Greens", "vmin": 0.0, "vmax": 5.0},
      "phyto": {"line_color": "darkgreen","cmap": "YlGn"},
      "zoo":   {"line_color": "purple","cmap": "PuBu"},
  }
  
  
  
  
  # def animate_timeseries(
  #     ds: xr.Dataset,
  #     *,
  #     vars: Sequence[str],
  #     groups: Optional[Dict[str, Any]],
  #     scope: str,                                   # "domain" | "region" | "station"
  #     regions: Optional[Sequence[Tuple[str, Dict[str, Any]]]] = None,
  #                                                   # For scope="region": list of (region_name, spec_dict)
  #                                                   #   spec_dict: {"shapefile": "..."} OR {"csv_boundary": "...", "lon_col": "...", "lat_col": "..."}
  #     stations: Optional[Sequence[Tuple[str, float, float]]] = None,
  #                                                   # For scope="station": list of (name, lat, lon) in WGS84
  #     # --- time filters (any combination; applied before spatial ops) ---
  #     months: Optional[Union[int, Sequence[int]]] = None,  # e.g., 7 or [6,7,8]
  #     years: Optional[Union[int, Sequence[int]]]  = None,  # e.g., 2018 or [2018,2019]
  #     start_date: Optional[str] = None,                    # "YYYY-MM-DD"
  #     end_date: Optional[str]   = None,                    # "YYYY-MM-DD"
  #     at_time: Optional[Any] = None,                       # NEW: single explicit instant; any pandas-parsable timestamp
  #                                                          #      (e.g., "2018-06-10 12:00"); selects the nearest data time.
  #                                                          #      Produces a one-frame GIF unless combined with other series/scope lines.
  #     at_times: Optional[Sequence[Any]] = None,            # NEW: list of explicit instants; sequence of pandas-parsable timestamps
  #                                                          #      (e.g., ["2018-06-01 00:00","2018-06-10 12:00", ...]).
  #                                                          #      For each requested instant, the nearest dataset timestep is used.
  #                                                          #      Takes precedence over `frequency` when provided.
  #     time_method: str = "nearest",                        # NEW: method used when matching `at_time/at_times` to data times.
  #                                                          #      Typically "nearest". Pandas-style options like "pad"/"backfill"
  #                                                          #      are also accepted if your time index is monotonic.
  #     frequency: Optional[str] = None,                     # NEW: user-friendly sampling cadence for frames when `at_*` is not set.
  #                                                          #      One of: "hourly" | "daily" | "monthly".
  #                                                          #      Internally mapped to pandas offsets: H / D / MS (month-start).
  #                                                          #      Samples one representative (nearest) timestep per period bucket.
  #     # --- vertical selection (applied before series extraction) ---
  #     depth: Any = "surface",                              # "surface" | "bottom" | "depth_avg"
  #                                                          # int -> sigma index (k)
  #                                                          # float in [-1,0] -> sigma value (s)
  #                                                          # other float or {"z_m": z} or ("z_m", z) -> absolute depth (m, negative downward)
  #     # --- output + styling ---
  #     base_dir: str = "",                                  # Used to form filename prefix
  #     figures_root: str = "",                              # Root folder for saving GIFs (module subdir auto-added)
  #     combine_by: Optional[str] = None,                    # None | "var" | "region" | "station"
  #                                                          #   None      -> one GIF per (scope item × variable)
  #                                                          #   "var"     -> one GIF per scope item; lines = variables
  #                                                          #   "region"  -> scope="region": one GIF per variable; lines = regions
  #                                                          #   "station" -> scope="station": one GIF per variable; lines = stations
  #     linewidth: float = 1.8,                              # Line width in the animation
  #     figsize: Tuple[int, int] = (10, 4),                  # (width, height) inches
  #     dpi: int = 150,                                      # Render resolution for saved GIF
  #     styles: Optional[Dict[str, Dict[str, Any]]] = None,  # Optional style map; e.g., {"temp":{"line_color":"lightblue"}}
  #     verbose: bool = True,                                # Print progress / debug
  # ) -> List[str]:
  #     """
  #     Create growing-line time-series GIF animations from FVCOM–ERSEM datasets.
  #
  #     Parameters
  #     ----------
  #     ds : xarray.Dataset
  #         Model dataset already opened/combined across files.
  #     vars : sequence of str
  #         Names to plot. Each entry may be a native variable (e.g., "temp") or a
  #         composite/group key resolvable via `groups` (e.g., "chl", "DOC").
  #     groups : dict or None
  #         Composite definitions; expressions or lists summed elementwise.
  #     scope : {"domain","region","station"}
  #         What to animate:
  #           - "domain"  → spatial mean over entire mesh (area-weighted if `art1` present).
  #           - "region"  → mask to polygon per region (nodes/elements inside).
  #           - "station" → nearest *node* to each (name, lat, lon).
  #     regions, stations :
  #         Required only for their respective scopes (validated).
  #
  #     Time filters
  #     ------------
  #     `months`, `years`, and/or `start_date`–`end_date` may be combined. Omitted ⇒ full span.
  #
  #     Depth selection
  #     ---------------
  #     "surface" | "bottom" | "depth_avg" | sigma index/value | absolute z (meters; negative downward).
  #     Absolute-z slices are done *per variable* using vertical coordinates ("z"/"z_nele").
  #
  #     Combining (multi-line animations)
  #     ---------------------------------
  #     combine_by=None:
  #         One GIF per (scope item × variable).
  #     combine_by="var":
  #         One GIF per scope item, overlaying all variables as separate lines.
  #     combine_by="region":
  #         (scope="region") One GIF per variable, overlaying all regions as lines.
  #     combine_by="station":
  #         (scope="station") One GIF per variable, overlaying all stations as lines.
  #
  #     Styling
  #     -------
  #     `styles` can provide per-series hints (e.g., color) keyed by var/region/station label.
  #     Only `line_color` is used currently; others follow Matplotlib defaults.
  #
  #     Returns
  #     -------
  #     List[str]
  #         Full file paths to the saved GIF(s).
  #
  #     Output filenames
  #     ----------------
  #     <prefix>__<ScopeOrName>__<VarOrMulti>__<DepthTag>__<TimeLabel>__TimeseriesAnim[__CombinedByX].gif
  #       - prefix      = basename(base_dir)
  #       - ScopeOrName = Domain | Region_<name> | Station_<name> | "All" (for combined comparisons)
  #       - VarOrMulti  = variable name or "multi" when combining by var
  #       - DepthTag    = e.g., surface | bottom | zavg | sigma-0.7 | zm-10
  #       - TimeLabel   = built from months/years/range, e.g., "Jul__2018" or "2018-04-01 to 2018-10-31"
  #
  #     Notes
  #     -----
  #     - Spatial means use `art1` when available; otherwise simple means.
  #     - Region masks accept shapefile or CSV boundary; elements can be derived from node masks if `nv` exists.
  #     - Station selection uses great-circle distance on WGS84; longitudes west are negative.
  #     - Works with Dask-backed datasets; computation occurs during reduction and GIF encoding.
  #
  # Examples:
  

  # DOMAIN — combine_by='var': one GIF, multiple lines = variables

  info("[animate] Domain (one animation, lines = vars)…")
  animate_timeseries(
      ds,
      vars=["temp", "DOC", "chl", "phyto", "zoo"],
      groups=GROUPS,
      scope="domain",
      years=2018,
      depth="surface",
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      combine_by="var",            # one animation for the domain; lines are variables
      styles=PLOT_STYLES,
      verbose=True,
  )
  

  # DOMAIN — no combining: one GIF per variable (classic behaviour)

  info("[animate] Domain (separate per variable)…")
  animate_timeseries(
      ds,
      vars=["temp", "DOC", "chl", "phyto", "zoo"],
      groups=GROUPS,
      scope="domain",
      years=2018,
      depth="surface",
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      combine_by=None,             # one animation per variable
      styles=PLOT_STYLES,
      verbose=False,
  )
  

  # REGIONS — combine_by='var': one GIF per region, lines = variables

  info("[animate] Regions (per region, lines = vars)…")
  animate_timeseries(
      ds,
      vars=["chl", "phyto", "zoo"],
      groups=GROUPS,
      scope="region",
      regions=REGIONS,
      months=[6, 7, 8], years=2018,
      depth={"z_m": -10},          # 10 m below surface
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      combine_by="var",            # one animation per region; lines are variables
      styles=PLOT_STYLES,
      verbose=False,
  )
  

  # REGIONS — combine_by='region': one GIF per variable, lines = regions

  info("[animate] Regions (per var, lines = regions)…")
  animate_timeseries(
      ds,
      vars=["chl", "phyto"],
      groups=GROUPS,
      scope="region",
      regions=REGIONS,
      years=2018,
      depth="surface",
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      combine_by="region",         # one animation per variable; lines are regions
      styles=PLOT_STYLES,
      verbose=False,
  )
  

  # STATIONS — combine_by=None: one GIF per (station × variable)

  info("[animate] Stations (separate per station × variable)…")
  animate_timeseries(
      ds,
      vars=["chl", "phyto"],
      groups=GROUPS,
      scope="station",
      stations=STATIONS,
      start_date="2018-04-01", end_date="2018-10-31",
      depth="depth_avg",
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      combine_by=None,             # one per variable per station
      styles=PLOT_STYLES,
      verbose=False,
  )
  

  # STATIONS — combine_by='station': one GIF per variable, lines = stations

  info("[animate] Stations (per var, lines = stations)…")
  animate_timeseries(
      ds,
      vars=["chl", "phyto"],
      groups=GROUPS,
      scope="station",
      stations=STATIONS,
      start_date="2018-04-01", end_date="2018-10-31",
      depth="surface",
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      combine_by="station",        # one animation per variable; lines are stations
      styles=PLOT_STYLES,
      verbose=False,
  )
  
  print(" Timeseries animation examples completed. Animations saved under:", FIG_DIR)
  
  
  # def animate_maps(
  #     ds: xr.Dataset,
  #     *,
  #     variables: Sequence[str],                         # Names to plot as individual frames/series
  #     scope: str = "domain",                            # "domain" | "region"
  #     regions: Optional[Sequence[Tuple[str, Dict[str, Any]]]] = None,
  #                                                       # For scope="region": list of (region_name, spec_dict)
  #                                                       #   spec_dict: {"shapefile": "..."} OR {"csv_boundary": "...", "lon_col": "...", "lat_col": "..."}
  #     # --- time filters (any combination; applied before spatial ops) ---
  #     months: Optional[Union[int, Sequence[int]]] = None,# e.g., 7 or [6,7,8]
  #     years: Optional[Union[int, Sequence[int]]]  = None,# e.g., 2018 or [2018,2019]
  #     start_date: Optional[str] = None,                  # "YYYY-MM-DD"
  #     end_date: Optional[str]   = None,                  # "YYYY-MM-DD"
  #     # --- explicit instants (override cadence) ---
  #     at_time: Optional[Any] = None,                     # Single pandas-parsable instant → one-frame map (unless multiple vars/regions)
  #     at_times: Optional[Sequence[Any]] = None,          # List of instants; nearest dataset timestep used for each
  #     time_method: str = "nearest",                      # Matching method for at_time/at_times ("nearest", "pad", "backfill", ...)
  #     frequency: Optional[str] = None,                   # "hourly" | "daily" | "monthly" | None → sampled cadence when no at_* given
  #     # --- vertical selection (applied before mapping) ---
  #     depth: Any = "surface",                            # "surface" | "bottom" | "depth_avg"
  #                                                       # int -> sigma index (k)
  #                                                       # float in [-1,0] -> sigma value (s)
  #                                                       # other float or {"z_m": z} or ("z_m", z) -> absolute depth (m, negative)
  #     # --- output + styling ---
  #     base_dir: str = "",                                # Used to form filename prefix
  #     figures_root: str = "",                            # Root folder for saving GIFs/MP4s (module subdir auto-added)
  #     groups: Optional[Dict[str, Any]] = None,           # Composite variable definitions (e.g., {"chl": ["diatChl","flagChl",...]})
  #     cmap: str = "viridis",                             # Matplotlib colormap name
  #     clim: Optional[Tuple[float, float]] = None,        # (vmin, vmax); overrides robust quantiles if provided
  #     robust_q: Tuple[float, float] = (5, 95),           # Percentiles for robust limits when clim not set
  #     shading: str = "gouraud",                          # Node-centered default; element-centered forces "flat"
  #     grid_on: bool = False,                             # Overlay mesh edges/nodes
  #     figsize: Tuple[float, float]] = (8, 6),            # (width, height) inches
  #     dpi: int = 150,                                    # Render resolution
  #     interval_ms: int = 100,                            # Frame delay for GIF writer (ms)
  #     fps: int = 10,                                     # Frames per second for MP4
  #     styles: Optional[Dict[str, Dict[str, Any]]] = None,# Per-var/region style hints (e.g., {"temp": {"vmin":..,"vmax":..,"cmap":"..."}})
  #     verbose: bool = True,                              # Print progress / debug
  # ) -> List[str]:
  #     """
  #     Create animated maps (GIF/MP4) from FVCOM–ERSEM (or similar) datasets.
  #
  #     Parameters
  #     ----------
  #     ds : xarray.Dataset
  #         Model dataset already opened/combined across files.
  #     variables : sequence of str
  #         Variables to render. Each may be a native variable (e.g., "temp") or a composite
  #         resolvable via `groups` (e.g., "chl").
  #     scope : {"domain","region"}
  #         Spatial scope for each frame:
  #           - "domain" → full mesh domain.
  #           - "region" → mask/clip to polygon(s) provided in `regions`.
  #     regions :
  #         Required when `scope="region"`. Provide a list of tuples:
  #           (region_name, {"shapefile": "/path/to/shape.shp"})
  #         or
  #           (region_name, {"csv_boundary": "/path/to/pts.csv", "lon_col": "lon", "lat_col": "lat"}).
  #
  #     Time filters
  #     ------------
  #     `months`, `years`, and/or `start_date`–`end_date` may be combined. If all omitted,
  #     the full dataset span is used.
  #
  #     Explicit instants vs cadence
  #     ----------------------------
  #     If `at_time` or `at_times` is provided, the nearest dataset times are used per instant
  #     (with `time_method`). When absent, `frequency` ("hourly"/"daily"/"monthly") samples a
  #     representative timestep per period bucket.
  #
  #     Depth selection
  #     ---------------
  #     "surface" | "bottom" | "depth_avg" | sigma index/value | absolute z (meters; negative downward).
  #     Absolute-z slices are applied per variable using appropriate vertical coordinates.
  #
  #     Styling
  #     -------
  #     Use `cmap`/`clim` or `robust_q` to control color scaling. `styles` can override per
  #     variable/region (e.g., vmin, vmax, cmap). `shading="gouraud"` is suited to node-centered
  #     fields; element-centered data should use "flat". `grid_on=True` draws mesh overlays.
  #
  #     Output
  #     ------
  #     Files are written under `<figures_root>/<basename(base_dir)>/animate` if present,
  #     otherwise under the run root. Both GIF and/or MP4 may be produced depending on writer.
  #
  #     Returns
  #     -------
  #     List[str]
  #         Full file paths to the saved animation(s).
  #
  #     Output filenames
  #     ----------------
  #     <prefix>__<ScopeOrName>__<Var>__<DepthTag>__<TimeLabel>__MAP.gif
  #       - prefix      = basename(base_dir)
  #       - ScopeOrName = Domain | Region_<name>
  #       - Var         = variable name
  #       - DepthTag    = e.g., surface | bottom | zavg | sigma-0.7 | zm-10
  #       - TimeLabel   = built from months/years/range or explicit instants
  #
  #     Notes
  #     -----
  #     - Works with Dask-backed datasets; computation occurs during slicing/encoding.
  #     - Area-weighting (via `art1`) may be used internally when aggregating to elements.
  #     - MP4 output uses `fps`; GIF respects `interval_ms`.
  #     """
  
  # Examples:
  
  # DOMAIN MAPS - daily frames for June 2018 at the surface
  #    Uses robust color limits per time window unless overridden by MAP_STYLES or clim/norm.
  print("[animate] domain map animation (hourly, surface, June 2018)")
  animate_maps(
      ds,
      variables=["temp", "chl"],   # native variables or GROUPS keys both work
      scope="domain",
      months=6, years=2018,
      depth="surface",
      groups=GROUPS,
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      frequency="daily",          #  hourly | daily | monthly
      grid_on=True,                # draw mesh overlay
      styles=PLOT_STYLES,           # optional per-var map styling
      verbose=True,
  )
  
  # REGION MAPS  all avaiablle frames across JJA 2018 at 10 m below surface
  print("[animate] region map animation (daily, z=10 m below surface, JJA 2018)")
  animate_maps(
      ds,
      variables=["chl"],
      scope="region",
      regions=REGIONS,
      months=[6, 7, 8], years=2018,
      depth={"z_m": -10},          # absolute metres below surface (negative down)
      groups=GROUPS,
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      grid_on=False,
      styles=PLOT_STYLES,
      verbose=True,
  )
  
  
  # DOMAIN MAPS - monthly frames for 2018 bottom layer
  print("[animate] domain map animation (monthly, bottom, 2018)")
  animate_maps(
      ds,
      variables=["chl"],
      scope="domain",
      years=2018,
      depth="bottom",
      groups=GROUPS,
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      frequency="monthly",
      styles=PLOT_STYLES,
      verbose=True,
  )
  
  # DOMAIN MAPS - explicit instants
  print("[animate] domain map animation (explicit instants)")
  animate_maps(
      ds,
      variables=["temp"],
      scope="domain",
      depth="depth_avg",
      groups=GROUPS,
      base_dir=BASE_DIR, figures_root=FIG_DIR,
      at_times=["2018-06-01 00:00", "2018-06-10 12:00", "2018-06-20 00:00"],
      grid_on=True,
      styles=PLOT_STYLES,
      verbose=True,
  )
  
  print(" Maps animation examples completed. Animations saved under:", FIG_DIR)


