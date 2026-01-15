#!/usr/bin/env python3
"""
Replicate a "Minor & Brinkley style" seasonal boxplot figure using FVCOM-ERSEM outputs.

What this script does
---------------------
This script loads a FVCOM-ERSEM NetCDF time series (many files), then produces
box-and-whisker plots that compare *distributions over time*.

The "Minor & Brinkley style" figure you want has:
- 4 panels (2x2 grid), one panel per variable:
    temp, O3_pH, O3_TA, O3_pCO2
- x-axis categories are basins/regions (Central/East/West)
- two boxplots per basin:
    Spring (blue) and Summer (pink)

How this maps to plot_box(...)
------------------------------
You control the grouping with three "plot grammar" arguments:

- facet_by : what becomes subplots (panels)
- x_by     : what becomes the x-axis categories inside each panel
- hue_by   : what becomes side-by-side colored groups within each x category

For the target figure:
- facet_by="variable"  -> 4 panels (one per variable)
- x_by="region"        -> x-axis is Central/East/West
- hue_by="season"      -> two boxplots per basin: Spring vs Summer

IMPORTANT: Avoiding "Unknown" / extra white boxes
-------------------------------------------------
If you define only Spring and Summer, then months outside those seasons
(Jan/Feb/Sep/Oct/Nov/Dec) do not belong to any season. In the updated plot_box,
unmapped season samples are dropped. Still, the cleanest approach is to filter
to the months you care about:

    months=[3,4,5,6,7,8]

This ensures the dataset only contains Spring+Summer samples before binning.

Per-panel y-axis labels (new function capability)
-------------------------------------------------
When facet_by="variable", each panel represents a different variable and unit.
The updated plot_box supports:

    ylabels={ "temp": "...", "O3_pH": "...", ... }

If ylabel is None, then for facet_by="variable", the function uses ylabels to
label each panel correctly.

Memory safety for huge datasets
-------------------------------
Boxplots require a distribution, so the function keeps a bounded sample per bin
using reservoir sampling:

- max_samples_per_bin: maximum stored values per (facet, x, hue) bin
  Typical: 10000 to 50000 depending on available memory and desired accuracy.

Outputs
-------
Plots are saved into a figure folder computed from:
- base_dir      (your model output directory)
- figures_root  (your plots directory)

The package uses these to create a consistent file prefix and output path.
"""

import matplotlib

from fvcomersemviz.io import load_from_base
from fvcomersemviz.plots.bars_box import plot_box
from fvcomersemviz.plot import (
    info,
    kv,
    list_files,
    summarize_files,
    print_dataset_summary,
)
from fvcomersemviz.utils import out_dir, file_prefix

# Headless backend (good for clusters / no display)
matplotlib.use("Agg", force=True)

# ---------------------------------------------------------------------
# Project paths (edit these)
# ---------------------------------------------------------------------
# BASE_DIR: directory containing your NetCDF outputs
# FILE_PATTERN: pattern to match files (wildcards allowed)
# FIG_DIR: directory where plots should be written
BASE_DIR = "/data/proteus1/scratch/yli/project/lake_erie/output_updated_river_var"
FILE_PATTERN = "erie_00??.nc"
FIG_DIR = "/data/proteus1/scratch/moja/projects/Lake_Erie/fviz-plots/"

# ---------------------------------------------------------------------
# Region specs (these define the x-axis categories when x_by="region")
# ---------------------------------------------------------------------
# Each region entry is: (label, {"shapefile": path})
# The shapefile defines a polygon boundary used to mask the model grid.
REGIONS = [
    ("Central", {"shapefile": "../data/shapefiles/central_basin_single.shp"}),
    ("East", {"shapefile": "../data/shapefiles/east_basin_single.shp"}),
    ("West", {"shapefile": "../data/shapefiles/west_basin_single.shp"}),
]

# ---------------------------------------------------------------------
# Seasons (define at runtime; used when any of facet/x/hue == "season")
# ---------------------------------------------------------------------
# For the "paper style" figure, we only want Spring vs Summer.
SEASONS_2 = {
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
}

# ---------------------------------------------------------------------
# Colors for the hue groups (Spring vs Summer)
# ---------------------------------------------------------------------
# These colors are used only when hue_by is not None.
SEASON_COLORS = {
    "Spring": "#87CEEB",
    "Summer": "#FF69B4",
}

# ---------------------------------------------------------------------
# Per-panel y-axis labels when facet_by="variable"
# ---------------------------------------------------------------------
# The updated plot_box can use this mapping to label each panel correctly.
# (ASCII only)
Y_LABELS = {
    "temp": "Temperature (deg C)",
    "O3_pH": "pH (seawater scale)",
    "O3_TA": "Total alkalinity (mmol m^-3)",
    "O3_pCO2": "pCO2 (uatm)",
}

# ---------------------------------------------------------------------
# Variables to plot (must exist in ds, or be resolvable via groups=...)
# ---------------------------------------------------------------------
VARS_4 = ["temp", "O3_pH", "O3_TA", "O3_pCO2"]


def main() -> int:
    # -----------------------------------------------------------------
    # 1) Discover and load data
    # -----------------------------------------------------------------
    info("Discovering files")
    files = list_files(BASE_DIR, FILE_PATTERN)
    summarize_files(files)
    if not files:
        print("No files found; abort.")
        return 1

    info("Loading dataset")
    ds = load_from_base(BASE_DIR, FILE_PATTERN)
    print_dataset_summary(ds)

    # Show where figures will be saved and how they will be named
    out_folder = out_dir(BASE_DIR, FIG_DIR)
    prefix = file_prefix(BASE_DIR)
    kv("Figure folder", out_folder)
    kv("Filename prefix", prefix)

    # -----------------------------------------------------------------
    # EXAMPLE 1 (closest to your screenshot):
    # 4-panel figure:
    # - panel (facet) = variable
    # - x-axis = region
    # - hue = season (Spring vs Summer)
    #
    # Key things:
    # - months filter restricts data to Spring+Summer only, so no "Unknown"
    #   season bin can appear.
    # - ylabel=None + ylabels=Y_LABELS enables per-panel y-axis labels.
    # -----------------------------------------------------------------
    info("Example 1: 4-panel seasonal boxplots (paper-style)")
    plot_box(
        ds,
        VARS_4,
        regions=REGIONS,
        depth="surface",
        seasons=SEASONS_2,
        months=[3, 4, 5, 6, 7, 8],  # keep only Spring+Summer samples
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        facet_by="variable",
        x_by="region",
        hue_by="season",
        hue_colors=SEASON_COLORS,
        ylabel=None,        # IMPORTANT: let ylabels drive panel labels
        ylabels=Y_LABELS,   # NEW: per-variable axis labels in facets
        ncols=2,
        figsize=(14, 10),
        title="Seasonal differences by basin (Spring vs Summer)",
        max_samples_per_bin=20000,
        random_seed=0,
        verbose=True,
    )

    # -----------------------------------------------------------------
    # EXAMPLE 2:
    # One variable, monthly panels (12 panels):
    # - panel (facet) = month
    # - x-axis = region
    # - no hue (one box per region per month)
    #
    # This shows distributions through the year.
    # -----------------------------------------------------------------
    info("Example 2: Monthly panels for pCO2 by basin")
    plot_box(
        ds,
        ["O3_pCO2"],
        regions=REGIONS,
        depth="surface",
        seasons=SEASONS_2,  # not used unless facet/x/hue is "season"
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        facet_by="month",
        x_by="region",
        hue_by="region",
        ylabel=Y_LABELS["O3_pCO2"],
        ncols=4,
        figsize=(14, 10),
        title="pCO2: monthly distributions by basin",
        max_samples_per_bin=10000,
        random_seed=0,
    )

    # -----------------------------------------------------------------
    # EXAMPLE 3:
    # Same as Example 1, but restrict to a specific year window.
    # This is useful when reproducing a paper's time range.
    # -----------------------------------------------------------------
    info("Example 3: Paper-style figure but restricted to years 2018-2020")
    plot_box(
        ds,
        VARS_4,
        regions=REGIONS,
        depth="surface",
        seasons=SEASONS_2,
        months=[3, 4, 5, 6, 7, 8],  # keep only Spring+Summer
        years=[2018, 2019, 2020],
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        facet_by="variable",
        x_by="region",
        hue_by="season",
        hue_colors=SEASON_COLORS,
        ylabel=None,
        ylabels=Y_LABELS,
        ncols=2,
        figsize=(14, 10),
        title="Seasonal differences by basin (Spring vs Summer), 2018-2020",
        max_samples_per_bin=20000,
        random_seed=0,
    )

    info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
