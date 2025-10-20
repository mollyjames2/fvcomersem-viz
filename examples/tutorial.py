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
#   How to point the toolkit at your **FVCOM output** (single file or collections).
#   How to make:
# 
#      Map plots
#      Hovmöller diagrams 
#      Time series plots
#      KDE stoichiometry** panels (e.g., N:C / P:C vs variables at surface/bottom).
#      How to use variable **groups/composites** (e.g., `chl`, `phyto`, `DOC`) via simple algebraic expressions.
#      Relationship curves
#      Commmunity compostion plots (i.e. phyto/zoo)
#      Time series and map animations
#      How to control time windows (months/years/date ranges) and depth slices (surface, bottom, fixed-z, depth-avg).
# 
# ---
# 
# ## What this notebook expects
# 
# * **Model data:** FVCOM–ERSEM NetCDF output (e.g., daily or hourly files).
# * **Optional region definitions:** shapefiles or CSV polygons for domain/central/boxes.
# * **Optional station list:** for point-based Hovmöller/series.
# 
# > If your paths differ from the examples, just edit the `BASE_DIR`, `FILE_PATTERN`, and any region/station paths in the “Setup” cells.
# 
# ---
# 
# ## Package at a glance
# 
# * **Name:** `fvcomersem-viz`
# * **Design:** function-first (import functions and call them), no CLI/GUI
# * **Key modules:**
# 
#   * `plots/maps.py` – maps of scalar fields on the FVCOM grid
#   * `plots/hovmoller.py` – along-time/along-depth sections at stations
#   * `plots/timeseries.py` – single or multi-variable time series & composites
#   * `plots/kde_stoichiometry.py` – 2×2 stoichiometry panels
#   * `plots/composition.py`, - visualises the composition of grouped communities (e.g., phytoplankton or zooplankton) as relative shares across time/space.
#   * `plots/curves.py` - builds reusable diagnostic curves (scatter/KDE/fits) and shared styling used across stoichiometry and time-series plot
#   * `io.py`, `regions.py`, `utils.py`, `plot.py` – data discovery, time/depth filters, masks, labels, plotting functions
# 
# ---
# 
# ## Requirements (tested versions)
# 
# * Python ≥ 3.9 (3.11 recommended)
# * Core stack: `numpy`, `pandas`, `xarray`, `matplotlib`, `netCDF4`, `cftime`, `scipy`
# * Geospatial (for regional masks/overlays): `geopandas`, `shapely`, `pyproj`, `rtree` (optional but recommended)
# * Performance (optional): `dask[array]`
# 
# 
# ##  Installation
# 
# Create a clean environment with FVCOM-compatible dependencies:
# 
# >```bash
# >conda create -n fviz python=3.11 geopandas shapely pyproj rtree -c conda-forge
# >conda activate fviz
# >```
# 
#  If working locally, you can install in editable mode:
# 
# > ```bash
# > pip install -e .
# > ```
# 
# ---
# 
# ## Typical workflow used in this notebook
# 
# 1. **Setup paths & imports**
# 
#    ```python
#    from fvcomersemviz.plots import maps, hovmoller, timeseries
#    from fvcomersemviz.io import filter_time
#    ```
# 2. **Load data** (single file or pattern), select **time window** and **depth slice**.
# 3. **Plot** using a purpose-built function (e.g., `maps.plot_surface_field(...)`), tweak labels, save.
# 4. **Repeat** for stations/regions/variables as needed.
# 
# ---
# 
# ## Reproducibility & citations
# 
# Please cite **`fvcomersem-viz`** alongside the relevant FVCOM/ERSEM model references when using figures generated from this toolkit. The notebook cells are structured so outputs are reproducible from a minimal set of inputs (file paths, variable names, time/depth selections, and optional region/station definitions).
# 
# ---
# 
# ## Core capabilities
# 
# Map visualisation:
# Create horizontal maps of surface or depth-averaged fields, optionally masked by regions or polygons.
# Useful for showing spatial patterns (e.g., chlorophyll, nutrients, oxygen).
# 
# Hovmöller diagrams:
# Plot time-depth (sigma or fixed-z) sections at selected stations or regions to reveal seasonal and interannual variability.
# 
# Time series and composites:
# Produce single or multi-variable time series, monthly/seasonal climatologies, and box-region averages.
# 
# Stoichiometry and diagnostics:
# Generate 2×2 KDE panels or scatter plots to explore relationships between model tracers (e.g., N:C, P:C ratios, oxygen vs temperature).
# 
# Variable groups and expressions:
# Access groups like phyto, zoo, nutrients, or define on-the-fly algebraic combinations (e.g., total chlorophyll or N:P).
# 
# ## Typical use cases
# 
# Generating consistent visual outputs across multiple FVCOM–ERSEM experiments.
# 
# Producing diagnostics or summary figures for reports and publications.
# 
# Quickly inspecting model fields without building a full analysis pipeline.
# 
# Supporting automated post-processing workflows for long-term simulations.
# 
# **Next:** run the setup cell below to configure your data paths and make your first map plot.
# 

# # Setup
# 
# This section builds a conda enmviuronment and installs the packpage with all its required dependencies. 
# 
# 

# ### (Optional) Install locally in editable mode 
# uncomment below to create a conda env and install the package in editable mode

# In[ ]:


## If you're working locally and have the repo checked out:
#!conda create -y -n fvcomersemviz python=3.11 geopandas shapely pyproj rtree -c conda-forge
#!conda run -n fvcomersemviz python3 -m pip install -e ..
#!conda run -n fvcomersemviz python3 -m pip install numpy pandas xarray matplotlib netCDF4 cftime scipy dask geopandas shapely pyproj rtree ipykernel notebook
#!conda run -n fvcomersemviz python3 -m ipykernel install --user --name fvcomersemviz --display-name "Python (fvcomersemviz)"

# Save kernelspec into this notebook so it opens with Python (fviz) next time
#from IPython.display import display, Javascript
#display(Javascript("""
#if (typeof Jupyter !== 'undefined' && Jupyter.notebook) {
#  Jupyter.notebook.metadata.kernelspec = {
#    "display_name": "Python (fvcomersemviz)",
#    "language": "python",
#    "name": "fvcomersemviz"
#  };
#  Jupyter.notebook.save_notebook();
#  alert("Saved. Now use: Kernel → Change kernel → Python (fviz) to continue in that env.");
#}
#"""))


# ### Switching to the new Conda environment
# 
# The environment fviz has now been created, dependencies installed, and a Jupyter kernel registered as “Python (fviz)”.
# To continue working inside that environment:
# 
# In the Jupyter menu bar, go to:
# Kernel → Change kernel → Python (fviz)
# 
# Wait a few seconds for the notebook to reconnect — the kernel name in the top-right corner should now read Python (fviz).
# 
# Once switched, all imports and plotting functions will run inside the new Conda environment you just created
# 
# We can check everything has installed correctly by running the cell below:

# In[1]:


get_ipython().system('conda run -n fvcomersemviz python3 -u ../tests/check_install.py')


# ## Setting up your data paths
# 
# In this section, we tell the notebook where to find the FVCOM–ERSEM model output and where to save plots.
# 
# You can configure everything by editing a few key variables:
# 
# BASE_DIR → the folder where your FVCOM–ERSEM NetCDF files live.
# Example:
# >```python
# >BASE_DIR = "/data/proteus1/scratch/yli/project/lake_erie/output_updated_river_var"
# >```
# 
# This should point to the root of your model output directory — the location of all NETCDF output files.
# 
# FILE_PATTERN → the naming pattern for your files.
# Example:
# >```python
# >FILE_PATTERN = "netcdf_00??.nc"
# >```
# 
# This pattern uses wildcards (? or *) to match all relevant NetCDF files you want to load together.
# For instance, this would match:
# 
# netcdf_0001.nc, netcdf_0002.nc, netcdf_0003.nc, ...
# 
# 
# FIG_DIR → the directory where all output plots will be saved.
# Example:
# >```python
# >FIG_DIR = "/data/proteus1/scratch/moja/projects/Lake_Erie/fvcomersem-viz/examples/plots/"
# >```
# 
# The package automatically creates subfolders inside FIG_DIR for different plot types, e.g.:
# >```python
# ><FIG_DIR>/<basename(BASE_DIR)>/maps/
# ><FIG_DIR>/<basename(BASE_DIR)>/timeseries/
# >```
# 
# You can override or disable that behaviour using the variable FVCOM_PLOT_SUBDIR:
# >```python
# >FVCOM_PLOT_SUBDIR = "project" # force all plots into a folder called “project”.
# >
# >FVCOM_PLOT_SUBDIR = "" # disable subfolders; save everything directly into FIG_DIR.
# >```
# 
# 
# *Tip: keeping these paths and patterns together makes it easy to reuse the same notebook for different model runs — just edit BASE_DIR, FILE_PATTERN, and (optionally) FIG_DIR at the top.*
# 
# **Run the cell below to set the datapaths for this notebook**:

# In[2]:


BASE_DIR = "/data/proteus1/scratch/yli/project/lake_erie/output_updated_river_var"
FILE_PATTERN = "erie_00??.nc"
FIG_DIR      = "/data/proteus1/scratch/moja/projects/Lake_Erie/fvcomersem-viz/examples/plots/"
#FVCOM_PLOT_SUBDIR = "" # disable subfolders; save everything directly into FIG_DIR


# ##  Subsampling for Regional and Station Plots
# 
# To focus on specific areas or points within the FVCOM–ERSEM domain, we can **subsample the dataset** using simple metadata that defines **stations** (points) and **regions** (polygons).
# 
# ---
# 
# ####  Stations
# 
# * Defined as a list of tuples: `(name, latitude, longitude)` in **decimal degrees (WGS84)**.
# * The code automatically finds the **nearest model node** for each station using great-circle distance.
# * Ideal for generating **time series** or **Hovmöller plots** at fixed locations.
# * Example format:
# 
# >  ```python
# >  STATIONS = [
# >      ("WE12", 41.90, -83.10),
# >      ("WE13", 41.80, -83.20),
# >  ]
# >  ```
# 
#   > Note: longitudes west of Greenwich are **negative**.
# 
# ---
# 
# ####  Regions
# 
# * Defined as a list of tuples: `(region_name, spec_dict)` where `spec_dict` describes a polygon source.
# * You can provide **either**:
# 
#   * A **shapefile** path (optionally filtered by `name_field` / `name_equals`), or
#   * A **CSV boundary file** with `lon`/`lat` columns (use `convex_hull=True` to wrap scattered points).
# * These polygons are converted into **grid masks**, allowing plots or averages to be limited to specific basins or zones.
# * Example format:
# 
#   ```python
#   REGIONS = [
#       ("Central", {"shapefile": "../data/shapefiles/central_basin_single.shp"}),
#       ("East",    {"shapefile": "../data/shapefiles/east_basin_single.shp"}),
#       ("West", {
#           "csv_boundary": "/data/proteus1/backup/rito/Models/FVCOM/fvcom-projects/erie/python/postprocessing/west_stations.csv",
#            "lon_col": "lon", 
#            "lat_col": "lat",
#            "convex_hull": True,   # <— wrap points
#            # "sort": "auto",      # (use this if your CSV is a boundary but unordered)
#        }),
#   ]
#   ```
# 
# ---
# 
# Using these simple definitions, the plotting functions automatically extract the relevant subset of model data — either at the **nearest node** (for stations) or **within a polygon mask** (for regions) — before generating plots or summary statistics.
# 
# **Run the cell below to set the stations and regions for this notebook**:

# In[3]:


STATIONS = [
    ("WE12", 41.90, -83.10),
    ("WE13", 41.80, -83.20),
]

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


# ### Groups and plot styles

# ##### Variable Groups / Composites
#  You can pass either:
#    - a native model variable name present in the dataset, e.g. "P1_c"
#    - or a *group* defined here:
#        • list/tuple  -> members are summed elementwise
#        • string expr -> evaluated in the dataset namespace (you can do +, -, *, /, etc.)
# Notes:
#    - Make sure every referenced variable exists in the dataset.
#    - Expressions run in a safe namespace that only exposes dataset variables.
#    - Example of an average (uncomment to use):
#        "phyto_avg": "(P1_c + P2_c + P4_c + P5_c) / 4",
# Example Groups:
# >```python
# >GROUPS = {
# >    "DOC":   "R1_c + R2_c + R3_c + T1_30d_c + T2_30d_c",  # dissolved organic carbon (sum of pools)
# >    "phyto": ["P1_c", "P2_c", "P4_c", "P5_c"],            # total phytoplankton carbon (sum)
# >    "zoo":   ["Z4_c", "Z5_c", "Z6_c"],                    # total zooplankton carbon (sum)
# >    "chl":   "P1_Chl + P2_Chl + P4_Chl + P5_Chl",         # total chlorophyll (sum)
# >}
# >```
# 
# ##### Plot styles
# We can set different colourschemes for each of the variables/groups we plot.
# if we don't set a specific colour for a variable it will fall back to default
# If writing a script that produces multiple types of plots (line plots, pcolour plots etc) we can set the colour scheme for each type here as e.g:
# >```python 
# >"zoo":   {"line_color": "#9467bd", "cmap": "PuBu"}
# >
# Example plot styles:
# >```python
# >PLOT_STYLES = {
# >    "temp":   {"line_color": "lightblue"},
# >    "DOC":   {"line_color": "blue"},
# >    "chl":   {"line_color": "lightgreen"},
# >    "phyto": {"line_color": "darkgreen"},
# >    "zoo":   {"line_color": "purple"},
# >    # Example with log scaling for maps/hov:
# >    # "nh4": {"line_color": "#ff7f0e", "cmap": "plasma", "norm": LogNorm(1e-3, 1e0)}
# >}
# >```
# 
# When combining by "region" or "station" for multiline plots and animations, you can also key styles by the region or station name to set their line colors.
# 
# **Run the cell below to load the groups and plot styles for this notebook**

# In[4]:


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




# # PLOTS

# # Timeseries
# 
# This section shows how to produce **time-series** from FVCOM–ERSEM output at three scopes:
# 
# * **Domain mean** — averages over the whole model domain (area-weighted if `art1` is present).
# * **Station** — extracts the nearest model node to each `(name, lat, lon)` and plots a series at that point.
# * **Region** — masks nodes **inside** a polygon (shapefile/CSV) and plots a regional aggregate.
# 
# Figures are written under:
# 
# ```
# FIG_DIR/<basename(BASE_DIR)>/timeseries/
# ```
# 
# …unless overridden via `FVCOM_PLOT_SUBDIR`.
# 
# ---
# 
# ###  Choosing variables, time windows, and depth
# 
# * **Variables**
#   You can pass either native tracers (e.g., `P1_c`, `temp`) or **group names** you define in `GROUPS` (e.g., `chl`, `phyto`, `zoo`, `DOC`...explined in more detail below). Styles (colours, norms) come from `PLOT_STYLES` when provided.
# 
# * **Time selection**
# 
#   * `months=[...]` — calendar months across all years (e.g., `[7]` for July or `[4,5,6,7,8,9,10]` for Apr–Oct).
#   * `years=[...]` — specific calendar years (e.g., `[2018]` or `[2019,2020]`).
#   * `start_date="YYYY-MM-DD", end_date="YYYY-MM-DD"` — explicit date range.
# 
# * **Depth selection** (shorthand accepted)
# 
#   * `"surface"` / `"bottom"` / `"depth_avg"`
#   * **Sigma layer index**: `depth=5` → layer `k=5`
#   * **Sigma value**: `depth=-0.7` → sigma `s=-0.7` (in `[-1, 0]`)
#   * **Absolute depth (m)**: `depth=-8.0` → `z = −8 m` (downward)
# 
#   > Notes: floats in `[-1, 0]` are treated as sigma; other floats are meters. Absolute-depth requires a vertical coordinate with `siglay` (default `z`).
# 
# ---
# ### Combining multiple variables, regions, or stations in one plot
# 
# You can make **multi-line plots** instead of one image per variable or location by using the keyword:
# 
# ```python
# combine_by="var"      # or "region" or "station"
# ```
# This option controls what is shown as separate lines within the same figure.
# 
# combine_by value	   One plot per...	   Lines represent...	    Typical use
#     "var"	         region / station	      variables	  Compare multiple tracers at one location
#   "region"	            variable	           regions	  Compare regions for a single tracer
#   "station"         	variable	          stations	  Compare stations for a single tracer
# 
# If you leave `combine_by=None` (the default) or omit it entirely, the script will produce one PNG per (variable × region/station) pair.
# 
# ###  Scopes & required metadata
# 
# * **Domain mean**
#   Uses the full mesh (area weighting if `art1` exists). No extra metadata needed.
# 
# * **Station**
#   Uses your `STATIONS` list of `(name, lat, lon)` in WGS84. The nearest model **node** is chosen by great-circle distance (WGS84 ellipsoid).
# 
#   > Reminder: longitudes west of Greenwich are **negative**.
# 
# * **Region**
#   Uses your `REGIONS` list of `(region_name, spec_dict)`, where `spec_dict` provides **one** of:
# 
#   * `{"shapefile": "/path/to/region.shp"}` (optionally with `name_field` / `name_equals`), or
#   * `{"csv_boundary": "/path/to/boundary.csv", "lon_col": "lon", "lat_col": "lat", "convex_hull": True}`.
#     The polygon is converted to a grid mask; nodes inside are included. If `nv` exists, “strict” element-inclusion may be used. If `art1` exists, regional means are area-weighted.
# 
# ---
# 
# ###  File naming & outputs
# 
# Output filenames follow a structured pattern so you can tell **scope**, **variable**, **depth**, and **time filter** at a glance, e.g.:
# 
# ```
# <basename(BASE_DIR)>__<Scope>__<VarOrGroup>__<DepthTag>__<TimeLabel>__Timeseries.png
# ```
# 
# * **Scope**: `Domain`, `Station_<NAME>`, or `Region_<NAME>`
# * **DepthTag**: `Surface`, `Bottom`, `DepthAvg`, `SigmaK5`, `SigmaS0.7`, `Z8m`, etc.
# * **TimeLabel**: derived from your `months`/`years`/`start_date`–`end_date`.
# 
# > With your paths, groups, styles, stations, and regions already set, you’re ready to run the time-series cells for domain, station, and region—just choose the variables, time window, and depth as described above.
# 
# 

# ### Time series example plots

# In[5]:


# --- Timeseries examples: domain, station, region  ---

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
from fvcomersemviz.plots.timeseries import (
    domain_mean_timeseries,
    station_timeseries,
    region_timeseries,
    domain_three_panel,
    station_three_panel,
    region_three_panel,
)

import matplotlib.pyplot as plt
from IPython.display import display

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
#     Examples:
#       # Separate figures (one per variable)
#       domain_mean_timeseries(ds, ["DOC", "chl", "temp"], depth="surface",
#                              months=[7], base_dir=BASE_DIR, figures_root=FIG_DIR,
#                              groups=GROUPS, styles=PLOT_STYLES)
#
#       # One multi-line figure (lines = variables)
#       domain_mean_timeseries(ds, ["DOC", "chl", "temp"], depth="surface",
#                              months=[7], base_dir=BASE_DIR, figures_root=FIG_DIR,
#                              groups=GROUPS, styles=PLOT_STYLES, combine_by="var")
#     """


#Example — surface DOC + chl + temp, July (months=[7]) - makes 3 figures (one for each variable)
fig = domain_mean_timeseries(
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

# Domain (surface, Jul), one plot with DOC + chl + temp
domain_mean_timeseries(
    ds=ds,
    variables=["DOC", "chl", "temp"],
    depth="surface",
    months=[7],
    base_dir=BASE_DIR, figures_root=FIG_DIR,
    groups=GROUPS, styles=PLOT_STYLES,
    combine_by="var",
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

#Example — depth-averaged phyto at first station in STATIONS
fig = station_timeseries(
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

# Example -Station WE12 — z = -5 m, Apr–Oct 2018: temp + DOC on one plot
fig = station_timeseries(
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

# Example - All stations — surface temp, Apr–Oct 2018: one plot, one line per station
fig = station_timeseries(
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


#Example — bottom zooplankton in first region (e.g., "Central"), full span
fig = region_timeseries(
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

# Example - Central region — surface, Jul 2018: chl + phyto + zoo on one plot
fig = region_timeseries(
    ds=ds,
    variables=["chl", "phyto", "zoo"],
    regions=[REGIONS[0]],                        # e.g., ("Central", {...})
    depth="surface",
    months=[7], years=[2018],
    base_dir=BASE_DIR, figures_root=FIG_DIR,
    groups=GROUPS, styles=PLOT_STYLES,
    combine_by="var",
    verbose=False,
)

# Example -  Compare regions — bottom DOC, Apr–Oct 2018: one plot, one line per region
fig = region_timeseries(
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

# Example:  Domain three-panel — DOC (full run)
fig = domain_three_panel(
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

# Example: Station three-panel — chl at first station (full run)
fig = station_three_panel(
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
# - The shaded ±1σ is **spatial** (spread across grid cells within the region at each time), unlike station_three_panel which uses **temporal** σ.
# - Composites in `groups` let you pass semantic variables like "chl"/"phyto"/"zoo" without rewriting expressions.
# - Works with Dask-chunked datasets; computation occurs during reductions/plotting.
# - Returns None; to view in a notebook, display the saved PNGs afterwards (e.g., with a small gallery cell).

#Example: Region three-panel — DOC in first region, Apr–Oct
fig = region_three_panel(
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

# Example: Domain — DOC at sigma layer index k=5, July
fig = domain_mean_timeseries(
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
fig = station_timeseries(
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
fig = region_timeseries(
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



# #### We can view these figures here by running the cell below:

# In[6]:


# Show saved figures for this run (timeseries)

from pathlib import Path
from IPython.display import display, Image, SVG

# Build the output root from your existing config
RUN_ROOT = Path(FIG_DIR) / Path(BASE_DIR).name     # e.g. <FIG_DIR>/<basename(BASE_DIR)>
OUT_ROOT = RUN_ROOT / "timeseries"                 

print("Looking under:", OUT_ROOT.resolve())

if not OUT_ROOT.exists():
    print(f" Folder does not exist: {OUT_ROOT}")
else:
    # grab newest first; include PNG and SVG
    files = sorted(
        list(OUT_ROOT.rglob("*.png")) + list(OUT_ROOT.rglob("*.svg")),
        key=lambda p: p.stat().st_mtime
    )
    if not files:
        print(f"No images found under {OUT_ROOT}")
    else:
        N = 20  # how many to show
        print(f"Found {len(files)} image(s). Showing the latest {min(N, len(files))}…")
        for p in files[-N:]:
            print("•", p.relative_to(RUN_ROOT))
            if p.suffix.lower() == ".svg":
                display(SVG(filename=str(p)))
            else:
                display(Image(filename=str(p)))


# ## Maps
# 
# This section covers the two mapping helpers used to make plan-view figures from FVCOM–ERSEM output:
# 
# * **`domain_map`** — plots variables over the **full model domain**.
# * **`region_map`** — same as above, but **masked to polygon regions** you define (shapefile or CSV boundary).
# 
# Both functions:
# 
# * accept native tracer names **or** composite/group names (from `GROUPS`),
# * handle **surface / bottom / depth-averaged / sigma / fixed-z** depth selections,
# * can render a **time mean** over a selected window **or** **specific instants**,
# * auto-pick colour limits robustly (or use your styles/limits),
# * save figures under:
#   `FIG_DIR/<basename(BASE_DIR)>/maps/`
# 
# ---
# 
# #### Choosing parameters
# 
# * **Variables**: any native variable in `ds` or a group from `GROUPS` (e.g., `chl`, `phyto`, `DOC`). Functions can accept multiple variables
# * **Time**: choose a **window** (months/years/date range) for a mean map, or specify **instant(s)** to plot.
# * **Depth**: `"surface"`, `"bottom"`, `"depth_avg"`, a **sigma index/value**, or a **fixed depth** (meters, negative downward).
# * **Styling**: set colormap, normalization (e.g., `LogNorm` for positive skew), robust quantile limits, mesh overlay, figure size/DPI via `PLOT_STYLES` or function kwargs.
# * **Output**: filenames encode the scope, variable, depth tag, and time label, e.g.
#   `<prefix>__Map-Domain__<Var>__<DepthTag>__<TimeLabel>__Mean.png` or `__Instant.png`.
# 
# ---
# 
# #### Domain vs Region maps
# 
# * **Domain maps**: plot the whole grid using the triangulation built from `lon`, `lat`, and connectivity.
# * **Region maps**: apply a polygon mask **before** plotting so colour limits and statistics reflect **only in-region** values.
# 
#   * Regions come from `(name, spec)` entries in `REGIONS`, using either a **shapefile** (optionally filtered by an attribute) or a **CSV boundary** (lon/lat columns, optional convex hull).
# 
# 
# 
# 
# 
# 
# 

# In[8]:


# --- Map examples: domain + region  ---

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

# Example 1: Domain mean maps at SURFACE — per-variable styles (DOC, chl, temp); July only
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

# Example 2: Domain instantaneous maps at BOTTOM (phyto) — two timestamps
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

# Example 3: Domain mean at ABSOLUTE depth z = -8 m (phyto), Apr–Oct 2018
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

# Example 1: Region=CENTRAL, depth-averaged mean (zoo with log norm), mesh overlay
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

# Example 2: Region=WEST, sigma selections: k=5 for DOC, s=-0.7 for chl
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

# Example 3: Region=EAST, ABSOLUTE z = -15 m, instantaneous (DOC)
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



# #### We can view these figures here by running the cell below:

# In[9]:


# Show saved figures for this run (maps)


# Build the output root from your existing config
RUN_ROOT = Path(FIG_DIR) / Path(BASE_DIR).name     # e.g. <FIG_DIR>/<basename(BASE_DIR)>
OUT_ROOT = RUN_ROOT / "maps"     

print("Looking under:", OUT_ROOT.resolve())

if not OUT_ROOT.exists():
    print(f" Folder does not exist: {OUT_ROOT}")
else:
    # grab newest first; include PNG and SVG
    files = sorted(
        list(OUT_ROOT.rglob("*.png")) + list(OUT_ROOT.rglob("*.svg")),
        key=lambda p: p.stat().st_mtime
    )
    if not files:
        print(f"No images found under {OUT_ROOT}")
    else:
        N = 12  # how many to show
        print(f"Found {len(files)} image(s). Showing the latest {min(N, len(files))}…")
        for p in files[-N:]:
            print("•", p.relative_to(RUN_ROOT))
            if p.suffix.lower() == ".svg":
                display(SVG(filename=str(p)))
            else:
                display(Image(filename=str(p)))


# ## Hovmoller
# 
#  A Hovmöller shows how a variable evolves **through time** and **down the water column** at the nearest model node/element to each `(name, lat, lon)` in `STATIONS`.
# 
# Figures are written under:
# 
# ```
# FIG_DIR/<basename(BASE_DIR)>/
# ```
# 
# …with filenames like:
# 
# ```
# <prefix>__Hovmoller-Station-<NAME>__<VAR>__sigma|z__<TimeLabel>.png
# ```
# 
# ---
# 
# ###  What you control (high level)
# 
# * **Stations**
# 
#   * The code resolves the **nearest grid node/element** by great-circle distance.
# 
# * **Axis (vertical)**
# 
#   * `axis="sigma"` — plot native σ layers (unitless). No interpolation.
#   * `axis="z"` — plot **absolute depth** (m, negative downward) by interpolating σ-profiles to regular `z_levels`.
# 
#     * If `z_levels` is **omitted**, a sensible set is auto-built from the column’s min depth to 0 m.
# 
# * **Variables**
# 
#   * Native fields (e.g., `temp`) or **groups** from `GROUPS` (e.g., `chl`, `phyto`, `DOC`).
# 
# * **Time window**
# 
#   * `months=[…]`, `years=[…]`, or `start_date="YYYY-MM-DD"`, `end_date="YYYY-MM-DD"`.
#   * A compact **time label** is appended to filenames.
# 
# * **Styling & scales**
# 
#   * Colormap via `cmap` or per-var in `PLOT_STYLES` (e.g., `{"chl": {"cmap": "Greens"}, "chl": {"norm": LogNorm(...)}}`).
#   * If no `norm` or `vmin/vmax`, limits are chosen **robustly** from the data.
#   * `dpi`, `figsize` control output look; `verbose` toggles printouts.
# 
# ---
# 
# ###  Usage
# 
# 1. Time-filter `ds` using your window.
# 2. For each station, pick nearest **node/element** and extract that **single column**.
# 3. Choose vertical axis:
# 
#    * **σ**: direct `time × siglay` pcolormesh.
#    * **z**: compute vertical coordinates (`ensure_z_from_sigma`), interpolate each profile to `z_levels`, then pcolormesh.
# 4. Compute colour limits (norm → explicit → robust) and **save** the figure.
# 
# 
# 

# In[10]:


# --- Hovmöller examples: station time × depth (save + inline preview) ---

from fvcomersemviz.plots.hovmoller import station_hovmoller
import numpy as np

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


# Example 1: WE12 — chlorophyll on sigma layers (full run, robust colour limits)
station_hovmoller(
    ds=ds,
    variables=["chl"],
    stations=[STATIONS[0]],               # e.g., ("WE12", 41.90, -83.10)
    axis="sigma",
    base_dir=BASE_DIR, figures_root=FIG_DIR,
    groups=GROUPS,
    styles=PLOT_STYLES,                   # per-var cmap/norm/vmin/vmax if set
)

# Example 2: WE12 — DOC on absolute depth z (Apr–Oct 2018), explicit z grid
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

# Example 3: WE13 — zooplankton on sigma layers (Apr–Oct 2018)
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




# In[11]:


# ---- Inline preview: -------
RUN_ROOT = Path(FIG_DIR) / Path(BASE_DIR).name        
HOV_DIR  = RUN_ROOT / "hovmoller"                      
search_root = HOV_DIR if HOV_DIR.exists() else RUN_ROOT

files = sorted(
    list(search_root.rglob("*.png")) + list(search_root.rglob("*.svg")),
    key=lambda p: p.stat().st_mtime
)

if not files:
    print(f"No Hovmöller images found under {search_root}")
else:
    N = 12
    print(f"Showing the latest {min(N, len(files))} Hovmöller plot(s) from {search_root}:")
    for p in files[-N:]:
        print("•", p.relative_to(RUN_ROOT))
        if p.suffix.lower() == ".svg":
            display(SVG(filename=str(p)))
        else:
            display(Image(filename=str(p)))


# ##  Stoichiometry KDE (2×2)
# 
# This plot builds a **2×2 density figure** to visualize stoichiometric ratios against a variable, at **surface** and **bottom**:
# 
# ```
# [ surface  N:C  vs <variable> ]   [ surface  P:C  vs <variable> ]
# [ bottom   N:C  vs <variable> ]   [ bottom   P:C  vs <variable> ]
# ```
# 
# Samples are pooled over the **selected time window** (and optional **region**). Each panel shows a **2D density** (Gaussian KDE or fast hist+blur).
# 
# Figures are written under:
# 
# ```
# FIG_DIR/<basename(BASE_DIR)>/
# ```
# 
# with filenames like:
# 
# ```
# <prefix>__KDE-Stoich__<Group>__<Variable>__<RegionTag>__<TimeLabel>.png
# ```
# 
# ---
# 
# ### What you control (high level)
# 
# * **Target group & variable**
# 
#   * `group`: which biological group to use for stoichiometry (e.g., `P5`). The code looks for **`<group>_NC`** and **`<group>_PC`**.
#   * `variable`: any **native** variable or a **group** from `GROUPS` (e.g., `chl`, `phyto`, `DOC`) to plot on the y-axis.
# 
# * **Region mask (optional)**
# 
#   * `region`: `(name, spec)` from your `REGIONS` list, or `None` for the whole domain.
#     Masks are applied on **nodes/elements** before sampling so density reflects **in-region** values only.
# 
# * **Time window**
# 
#   * `months=[…]`, `years=[…]`, or `start_date`/`end_date`.
#     A compact label is embedded in the filename.
# 
# * **Depth slices**
# 
#   * Panels are computed from **`surface`** and **`bottom`** depth selections internally.
# 
# * **Density method & speed**
# 
#   * `method="kde"`: classic Gaussian KDE (accurate).
#   * `method="hist"`: 2D histogram + Gaussian blur (very fast for large datasets).
#   * `sample_max`: optional random subsample cap for huge datasets.
#   * `grids`, `bw_method` (for KDE) or `hist_sigma` (for hist) control smoothness/detail.
# 
# * **Styling**
# 
#   * `cmap` or per-variable overrides via `PLOT_STYLES` (e.g., fixed `vmin/vmax` for the y-axis variable).
#   * `scatter_underlay`: draw a light random subset of raw points beneath the density for context.
#   * `dpi`, `figsize`, `verbose`.
# 
# ---
# 
# ###  Usage
# 
# 1. **Filter time** once, then create **surface** and **bottom** slices.
# 2. **Build region masks** (if any) and apply them to the three fields needed per panel:
#    `variable`, `<group>_NC`, `<group>_PC`.
# 3. **Align & flatten** time×space samples (node/element-aware), optionally **subsample**.
# 4. Compute **2D density** (KDE or hist+blur) and render each panel; optionally add sparse **scatter**.
# 5. Save a single **2×2** figure (skipping panels with too few samples).
# 
# ---
# 
# ###  Good practice
# 
# * Use `method="hist"` with a reasonable `sample_max` for **very large** runs.
# * Keep y-axis limits consistent across comparisons via `PLOT_STYLES[variable]["vmin"/"vmax"]`.
# * Region tags and time labels in filenames help you **batch** different scenarios cleanly.
# * If a panel is empty (too few finite pairs), it’s **skipped** rather than saved blank.
# 

# In[12]:


# --- Stoichiometry KDE (2×2) examples:  ---

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

# Example 1: DOMAIN • JJA • group=P5 • variable=P5_c
kde_stoichiometry_2x2(
    ds=ds,
    group="P5",
    variable="P5_c",
    region=None,                     # full domain
    months=[6,7,8], years=None,      # Jun–Aug across all years
    base_dir=BASE_DIR, figures_root=FIG_DIR, groups=GROUPS,
    min_samples=200, scatter_underlay=800,
    styles=PLOT_STYLES if "PLOT_STYLES" in globals() else None,
    **FAST,
)

# Exampl 2: REGION • Apr–Oct 2018 • group=P5 • variable=phyto (composite)
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
        **FAST,
    )

# Example 3: DOMAIN • full run • group=P5 • variable=chl (composite)
kde_stoichiometry_2x2(
    ds=ds,
    group="P5",
    variable="chl",
    region=None,
    months=None, years=None,         # full time span
    base_dir=BASE_DIR, figures_root=FIG_DIR, groups=GROUPS,
    min_samples=300, scatter_underlay=1500,
    styles=PLOT_STYLES if "PLOT_STYLES" in globals() else None,
    **FAST,
)

# Example 4: REGION COMPARISON • JJA 2018 • group=P5 • variable=P5_c (first two regions if available)
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
            **FAST,
        )

print(" KDE stoichiometry examples completed. Figures saved under:", FIG_DIR)



# In[13]:


# ---- Inline preview: show newest KDE images from this run ----
RUN_ROOT = Path(FIG_DIR) / Path(BASE_DIR).name                 # <FIG_DIR>/<basename(BASE_DIR)>
KDE_DIR  = RUN_ROOT / "kde_stoichiometry"                      # default subfolder (fallback to RUN_ROOT if absent)
search_root = KDE_DIR if KDE_DIR.exists() else RUN_ROOT

files = sorted(
    list(search_root.rglob("*.png")) + list(search_root.rglob("*.svg")),
    key=lambda p: p.stat().st_mtime
)

if not files:
    print(f"No KDE images found under {search_root}")
else:
    N = 8
    print(f"Showing the latest {min(N, len(files))} KDE plot(s) from {search_root}:")
    for p in files[-N:]:
        print("•", p.relative_to(RUN_ROOT))
        if p.suffix.lower() == ".svg":
            display(SVG(filename=str(p)))
        else:
            display(Image(filename=str(p)))


# ##  Curves (x–y diagnostics)
# 
# These “curves” visualize the **relationship between two model variables** (x on the horizontal axis, y on the vertical) after you’ve applied your **time filters**, **spatial scope** (domain / region / station), and **depth selection**. They answer questions like:
# 
# * **How does Y respond to X?** e.g., *chl vs PAR*, *phyto vs temp*, *DOC vs mixed-layer depth*.
# * **Is there a threshold, optimum, or saturation?** (e.g., light limitation at low PAR; nutrient saturation at high DIN.)
# * **How different are regions or stations?** (Plot multiple curves with the same x/y but different scopes.)
# * **How does the relationship change with depth or season?** (Use different depth selections or time windows.)
# 
# ---
# 
# #### Defining specs (what to plot)
# 
# `plot_curves(specs=[...], ds=..., groups=...)` takes a list of spec dictionaries—one per curve. Each spec tells the function what to compute and how to render it.
# 
# ##### Required
# ```python
# x: str # variable/expression for the X-axis
# 
# y: str # variable/expression for the Y-axis
# 
# #Common (optional but useful)
# 
# name: str # legend label for this curve
# 
# filters: dict # — time/predicate filters
#     months: [6,7,8] # Example JJA
#     years: [2018] # Example 1 year
#     start / end: "YYYY-MM-DD"
#     where: str # boolean expression mask (e.g., "PAR > 0", or a named expression in groups)
# 
# depth: one of
#     "surface", "bottom", "depth_avg"
#     a sigma level (float, e.g., 0.5)
#     absolute-z dict, e.g., {"z_m": -10} (10 m below surface)
# 
# scope: dict # where to sample
#     {"region": (name, spec)} # polygon from shapefile or boundary CSV
#     {"station": (name, lat, lon)} # nearest column
#     {} — domain # (default)
# 
# Choose one rendering mode
# 
# bin: {"x_bins": 40, "agg": "median"|"mean"|"p90", "min_count": 10, "iqr": True} # draws binned median with optional IQR band
# scatter: {"s": 4, "alpha": 0.15} # plots all samples as a semi-transparent cloud
# 
# style: matplotlib overrides (e.g., {"color": "C3"})
# 
# aliases (optional): per-spec name remaps, e.g. {"PAR": "light_parEIR"}
# ```
# ##### About names & expressions
# 
# x/y (and filters.where) support GROUPS and algebra (e.g., "P1_Chl + P2_Chl").
# 
# Variable lookup is tolerant to case/underscores (chl_total, ChlTotal, chl-total all match if present).
# 
# ---
# 
# #### Auto file naming
# 
# If you don’t pass stem=..., the function auto-builds a filename tag from:
# 
# Scope: Domain | Region-<Name> | Station-<Name> | MultiScope
# 
# Depth: e.g., surface, bottom, depth_avg, sigma0.5, z-10m, or MixedDepth
# 
# Time: e.g., JJA, 2018, 2018-04–2018-10, or MixedTime
# 
# Content: <X>_vs_<Y> (+ Ncurves if multiple specs)
# 
#     Figures are written under:
#     FIG_DIR/<basename(BASE_DIR)>/curves/
#     …unless overridden via FVCOM_PLOT_SUBDIR.
# 
# ---
# 
# #### Two ways of drawing the relationship
# 
# * **Binned median + IQR (robust trend):**
#   X is split into equal-width bins. For each bin we compute the **median** Y (the central tendency) and the **IQR band** (25–75th percentiles) to show spread.
# 
#   * **When to use:** you want a **clean functional shape** (less noise), and a sense of variability that’s robust to outliers.
#   * **What it means:** the line traces the *typical* Y for a given X; the shaded region is the central half of outcomes at that X.
# 
# * **Raw scatter (cloud of points):**
#   Every time×space sample (after filters/masks) is plotted.
# 
#   * **When to use:** you want to see **full dispersion**, multi-modal patterns, or rare extremes the median might hide.
#   * **What it means:** density and spread of points reflect how often combinations of (X,Y) occur in your filtered subset.
# 
# > Tip: Combine both—plot a **binned curve** for the backbone and **scatter** (semi-transparent) underneath for context.
# 
# ---
# 
# #### How to *read* the curves
# 
# * **Monotonic rise or fall:** suggests **limitation** or **inhibition** (e.g., phyto rising with PAR up to saturation).
# * **Plateau (saturation):** beyond a certain X, **another** factor likely limits Y.
# * **Hump-shaped (optimum):** indicates **peak performance** at intermediate X (temperature, for instance).
# * **Wide IQR band:** strong **spatial/temporal heterogeneity**, unresolved drivers, or mixed regimes within your filter.
# * **Split/looping clouds:** potential **regime shifts**, **seasonality**, or **hysteresis**—narrow your time window or add a `where` predicate.
# 
# 
# #### Good practice & caveats
# 
# * **Keep units consistent** and label axes clearly (use `x_label`/`y_label` if your expressions are verbose).
# * **Mind bin support:** if many bins have **low sample counts**, the curve can wiggle; increase `min_count` or reduce `x_bins`.
# * **Avoid mixing regimes** unintentionally—narrow time windows or add a `where` clause to focus the story.
# * **Compare like with like:** when contrasting regions/stations, use the **same filters/depth** so differences are interpretable.
# * **Consider log scales** (set in your plotting style) if X or Y spans orders of magnitude.
# * **Interpret IQR correctly:** it’s the **middle 50%**; large bands do not necessarily mean noise—they may indicate **real heterogeneity**.
# 
# 

# In[14]:


# --- Curves (x–y) ---


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


# --- Curves (x–y diagnostics) reference ---

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

# Example spec
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

# Example spec
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
out2 = plot_curves(
    specs=specs_temp_phyto, ds=ds, groups=GROUPS,
    base_dir=BASE_DIR, figures_root=FIG_DIR, dpi=150,
)

# Example 3 — Domain, depth-avg, Apr–Oct 2018 (scatter cloud)

# Example spec
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


# In[15]:


# ---- Inline preview: show curves images from this run ----
RUN_ROOT = Path(FIG_DIR) / Path(BASE_DIR).name                 # <FIG_DIR>/<basename(BASE_DIR)>
CURVES_DIR  = RUN_ROOT / "curves"                    
search_root = CURVES_DIR if CURVES_DIR.exists() else RUN_ROOT

files = sorted(
    list(search_root.rglob("*.png")) + list(search_root.rglob("*.svg")),
    key=lambda p: p.stat().st_mtime
)

if not files:
    print(f"No curve images found under {search_root}")
else:
    N = 8
    print(f"Showing the latest {min(N, len(files))} curve plot(s) from {search_root}:")
    for p in files[-N:]:
        print("•", p.relative_to(RUN_ROOT))
        if p.suffix.lower() == ".svg":
            display(SVG(filename=str(p)))
        else:
            display(Image(filename=str(p)))


# ## Animations
# 
# the `fvcomersemviz` package contains functionality to create animations from the timeseries, maps, and community composition plots. 
# 
# 
# ### Timeseries animations
# 
# plots a growing line over time within the selected time window. can be multi-line (`vars`,`regions`, `stations`) or a single line.
# Depth selection works the same as the static timeseries plots
# 
# 
# 

# In[16]:


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



from fvcomersemviz.plots.animate import animate_timeseries
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
#     Create growing-line time-series **GIF animations** from FVCOM–ERSEM datasets.
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
#     Examples

# ------------------------------------------------------------------
# 1) DOMAIN — combine_by='var': one GIF, multiple lines = variables
# ------------------------------------------------------------------
info("[animate] Domain (one animation, lines = vars)…")
anim = animate_timeseries(
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

# ------------------------------------------------------------------
# 1b) DOMAIN — no combining: one GIF per variable (classic behaviour)
# ------------------------------------------------------------------
info("[animate] Domain (separate per variable)…")
anim = animate_timeseries(
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

# ------------------------------------------------------------------
# 2) REGIONS — combine_by='var': one GIF per region, lines = variables
# ------------------------------------------------------------------
info("[animate] Regions (per region, lines = vars)…")
anim = animate_timeseries(
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

# ------------------------------------------------------------------
# 2b) REGIONS — combine_by='region': one GIF per variable, lines = regions
# ------------------------------------------------------------------
info("[animate] Regions (per var, lines = regions)…")
anim = animate_timeseries(
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

# ------------------------------------------------------------------
# 3) STATIONS — combine_by=None: one GIF per (station × variable)
# ------------------------------------------------------------------
info("[animate] Stations (separate per station × variable)…")
anim = animate_timeseries(
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

# ------------------------------------------------------------------
# 3b) STATIONS — combine_by='station': one GIF per variable, lines = stations
# ------------------------------------------------------------------
info("[animate] Stations (per var, lines = stations)…")
anim = animate_timeseries(
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


# In[17]:


from IPython.display import display, Image, Video, HTML

RUN_ROOT = Path(FIG_DIR) / Path(BASE_DIR).name          # <FIG_DIR>/<basename(BASE_DIR)>
ANIM_DIR = RUN_ROOT / "animate"                      # flat folder with animations

search_root = ANIM_DIR if ANIM_DIR.exists() else RUN_ROOT

# Non-recursive listing; only .gif and .mp4
anims = sorted(
    [p for p in search_root.iterdir() if p.suffix.lower() in {".gif", ".mp4"}],
    key=lambda p: p.stat().st_mtime
)

if not anims:
    print(f"No GIF/MP4 animations found under {search_root}")
else:
    print(f"Found {len(anims)} animation(s) under {search_root}\n")
    for p in anims:
        display(HTML(f"<div style='font-family:monospace; margin:0.25em 0;'>• {p.relative_to(RUN_ROOT)}</div>"))
        if p.suffix.lower() == ".gif":
            display(Image(filename=str(p), width=720))
        else:  # .mp4
            display(Video(filename=str(p), embed=True, width=720))


# In[ ]:


from fvcomersemviz.plots.animate import animate_timeseries
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
#     Create **animated maps** (GIF/MP4) from FVCOM–ERSEM (or similar) datasets.
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


# ==========================================================================
# === ANIMATION EXAMPLES =========================================
# ==========================================================================

# 4) DOMAIN MAPS - daily frames for June 2018 at the surface
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

# 4b) REGION MAPS  all avaiablle frames across JJA 2018 at 10 m below surface
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


# 4c) DOMAIN MAPS - monthly frames for 2018 bottom layer
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

# 4d) DOMAIN MAPS - explicit instants
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



# In[ ]:


from pathlib import Path
from IPython.display import display, Image, Video, HTML

RUN_ROOT = Path(FIG_DIR) / Path(BASE_DIR).name          # <FIG_DIR>/<basename(BASE_DIR)>
ANIM_DIR = RUN_ROOT / "animate"                         # flat folder with animations

search_root = ANIM_DIR if ANIM_DIR.exists() else RUN_ROOT

# Non-recursive listing; only .gif and .mp4, and filename contains "MAP" (case-insensitive)
anims = sorted(
    [
        p for p in search_root.iterdir()
        if p.suffix.lower() in {".gif", ".mp4"} and "map" in p.name.lower()
    ],
    key=lambda p: p.stat().st_mtime
)

if not anims:
    print(f"No GIF/MP4 animations with 'MAP' in the name found under {search_root}")
else:
    print(f"Found {len(anims)} animation(s) with 'MAP' in the name under {search_root}\n")
    for p in anims:
        display(HTML(f"<div style='font-family:monospace; margin:0.25em 0;'>• {p.relative_to(RUN_ROOT)}</div>"))
        if p.suffix.lower() == ".gif":
            display(Image(filename=str(p), width=720))
        else:  # .mp4
            display(Video(filename=str(p), embed=True, width=720))

