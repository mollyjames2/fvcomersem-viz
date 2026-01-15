#!/usr/bin/env python3
"""
Bar-plot summaries for Lake Erie FVCOM-ERSEM outputs.

    Create bar-chart summaries (means with uncertainty) for FVCOM-ERSEM outputs.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing FVCOM/ERSEM variables and a 'time' coordinate.
        For region masking, the dataset must contain 'lon' and 'lat' (node coords).
        For element-centered station selection, the dataset should contain 'lonc'/'latc'.

    variables : List[str]
        List of variable names to plot. Items may be:
          - names of variables present in `ds`, OR
          - names of grouped variables defined in `groups` (e.g. "DOC"),
            which will be evaluated via the package's variable/group evaluation logic.

    regions : Optional[List[Tuple[str, Dict[str, Any]]]]
        Optional list of named region specifications used for spatial masking.
        Each entry is (region_name, spec_dict) where spec_dict is either:
          - {"shapefile": <path>, ...} for shapefile-based polygons, or
          - {"csv_boundary": <path>, ...} for CSV boundary polygons.
        If you use any of x_by/hue_by/facet_by == "region", you must provide regions.
        Region masking uses node masks for node-centered variables and element masks
        (derived from nodes) for element-centered variables (requires ds['nv']).

    stations : Optional[List[Tuple[str, float, float]]]
        Optional list of point stations for station-based grouping/selection.
        Format is [(station_name, lon, lat), ...].
        If you use any of x_by/hue_by/facet_by == "station", you must provide stations.
        For node-centered variables, the nearest node is used.
        For element-centered variables, the nearest element center is used if ds has
        'lonc' and 'latc'.

    depth : Any
        Depth selector(s) passed through to the package's depth-resolution helper.
        Can be:
          - a single selector (e.g. "surface", "bottom", numeric depth, etc.), OR
          - a list/tuple of selectors to enable depth as a grouping dimension.
        If you want to group/compare by depth (x_by/hue_by/facet_by == "depth"),
        pass multiple depths, e.g. depth=["surface", "bottom"].

    months : Optional[List[int]]
        Time filter applied before aggregation: keep only samples whose timestamp month
        is in this list (1..12). Example: months=[6,7,8] for summer-only.
        This filter is applied in addition to years/start_date/end_date.

    years : Optional[List[int]]
        Time filter applied before aggregation: keep only samples whose timestamp year
        is in this list (e.g. years=[2018,2019,2020]).
        This filter is applied in addition to months/start_date/end_date.

    start_date : Optional[str]
        Start date for time filtering (inclusive). Typically an ISO-like string
        (e.g. "2018-01-01"). Applied before aggregation.

    end_date : Optional[str]
        End date for time filtering (inclusive). Typically an ISO-like string
        (e.g. "2020-12-31"). Applied before aggregation.

    seasons : Optional[Dict[str, Sequence[int]]]
        Runtime season definition used when any of x_by/hue_by/facet_by == "season".
        Mapping: season_label -> months (1..12).
        Example:
            seasons = {"Spring":[3,4,5], "Summer":[6,7,8], "Autumn":[9,10,11], "Winter":[12,1,2]}
        If None, a default meteorological season mapping is used.

    groups : Optional[Dict[str, Any]]
        Optional variable group definitions used to compute grouped variables.
        Example:
            groups = {"DOC": "R1_c + R2_c + R3_c + T1_30d_c + T2_30d_c"}
        If provided, entries in `variables` can reference these group keys and the
        package will evaluate them into a DataArray.

    facet_by : Optional[str]
        Dimension used to create subplot panels (facets). If None, a single panel
        is produced. Supported values are:
            "variable", "region", "station", "depth", "day", "month", "year", "season"
        Example: facet_by="month" creates a 12-panel figure (Jan..Dec).

    x_by : str
        Dimension used for x-axis categories within each facet.
        Supported values are:
            "variable", "region", "station", "depth", "day", "month", "year", "season"
        Example: x_by="region" makes one category per region on the x-axis.

    hue_by : Optional[str]
        Dimension used for colored bar groups within each x category.
        If None, only one bar per x category is drawn.
        Supported values are:
            "variable", "region", "station", "depth", "day", "month", "year", "season"
        Example: hue_by="season" draws four bars (Spring/Summer/Autumn/Winter) per x.

    hue_colors : Optional[Any]
        Optional hue color specification. Can be:
          - None: use matplotlib default color cycle
          - list/tuple: colors used in order of hue levels
          - dict: mapping hue_level -> color
        Only used when hue_by is not None.

    error : str
        Uncertainty / error-bar method:
          - "sd"   : mean +/- 1 sample standard deviation across time samples
          - "ci95" : 95% confidence interval for the mean across time samples
        Error bars are computed across the time samples that fall into each bin.

    dpi : int
        Output figure resolution (dots per inch) used when saving the PNG.

    figsize : Tuple[float, float]
        Figure size in inches (width, height). For many facets, increase this.

    ncols : Optional[int]
        Number of columns in the facet grid (only relevant when facet_by is not None).
        If None, a reasonable default is chosen.

    title : Optional[str]
        Optional figure-level title. If None, a default title is generated from
        variables/depth/time window.

    ylabel : Optional[str]
        Optional y-axis label for all panels. If None, defaults to a generic label
        ("Mean") or whatever the implementation chooses.

    base_dir : str
        Base directory used for output naming (prefix) and for computing output paths
        via fvcomersemviz.utils.file_prefix and fvcomersemviz.utils.out_dir.

    figures_root : str
        Root directory where figures are written (combined with base_dir by out_dir()).

    verbose : bool
        If True, print progress/debug messages while computing masks/series and saving.
"""

import matplotlib
from fvcomersemviz.io import load_from_base
from fvcomersemviz.plots.bars_box import plot_bars
from fvcomersemviz.plot import (
    info,
    kv,
    list_files,
    summarize_files,
    print_dataset_summary,
)
from fvcomersemviz.utils import out_dir, file_prefix

# Use a headless backend (good for clusters / no display)
matplotlib.use("Agg", force=True)

# ---------------------------------------------------------------------
# Project paths (edit as needed)
# ---------------------------------------------------------------------
BASE_DIR = "/data/proteus1/scratch/yli/project/lake_erie/output_updated_river_var"
FILE_PATTERN = "erie_00??.nc"
FIG_DIR = "/data/proteus1/scratch/moja/projects/Lake_Erie/fviz-plots/"

# ---------------------------------------------------------------------
# Region specs
# ---------------------------------------------------------------------
REGIONS = [
    ("Central", {"shapefile": "../data/shapefiles/central_basin_single.shp"}),
    ("East", {"shapefile": "../data/shapefiles/east_basin_single.shp"}),
    ("West", {"shapefile": "../data/shapefiles/west_basin_single.shp"}),
]

# ---------------------------------------------------------------------
# Variable group definitions
# ---------------------------------------------------------------------
DOC_GROUPS = {
    "DOC": "R1_c + R2_c + R3_c + T1_30d_c + T2_30d_c",
}

# ---------------------------------------------------------------------
# Season definition 
# ---------------------------------------------------------------------
SEASONS = {
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10, 11],
    "Winter": [12, 1, 2],
}


def main() -> int:
    info("Discovering files")
    files = list_files(BASE_DIR, FILE_PATTERN)
    summarize_files(files)
    if not files:
        print("No files found; abort.")
        return 1

    info("Loading dataset")
    ds = load_from_base(BASE_DIR, FILE_PATTERN)
    print_dataset_summary(ds)

    out_folder = out_dir(BASE_DIR, FIG_DIR)
    prefix = file_prefix(BASE_DIR)
    kv("Figure folder", out_folder)
    kv("Filename prefix", prefix)

    # -----------------------------------------------------------------
    # Example 1: Annual (all-time-filtered) DOC by region, with SD
    # -----------------------------------------------------------------
    info("Example 1: DOC by region (SD)")
    plot_bars(
        ds,
        ["DOC"],
        regions=REGIONS,
        depth="surface",
        groups=DOC_GROUPS,
        seasons=SEASONS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        x_by="region",
        hue_by=None,
        facet_by=None,
        error="sd",
        title="DOC (surface): mean by region (SD)",
        ylabel="DOC",
        verbose=True,
    )

    # -----------------------------------------------------------------
    # Example 2: DOC by region, colored by season (CI95)
    #   - x: region
    #   - hue: season
    # -----------------------------------------------------------------
    info("Example 2: DOC by region, hue by season (CI95)")
    plot_bars(
        ds,
        ["DOC"],
        regions=REGIONS,
        depth="surface",
        groups=DOC_GROUPS,
        seasons=SEASONS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        x_by="region",
        hue_by="season",
        facet_by=None,
        error="ci95",
        title="DOC (surface): region x season (CI95)",
        ylabel="DOC",
    )

    # -----------------------------------------------------------------
    # Example 3: Facet by month (12 panels), DOC by region (CI95)
    #   - facet: month
    #   - x: region
    # -----------------------------------------------------------------
    info("Example 3: DOC by region faceted by month (CI95)")
    plot_bars(
        ds,
        ["DOC"],
        regions=REGIONS,
        depth="surface",
        groups=DOC_GROUPS,
        seasons=SEASONS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        facet_by="month",
        x_by="region",
        hue_by=None,
        error="ci95",
        figsize=(14, 10),
        ncols=4,
        title="DOC (surface): monthly panels, mean by region (CI95)",
        ylabel="DOC",
    )

    # -----------------------------------------------------------------
    # Example 4: Variable comparison on x-axis, faceted by region (SD)
    #   - facet: region
    #   - x: variable
    #
    # If you have other variables you want to compare, add them here.
    # This example shows DOC (group) vs a raw variable "TEMP" (if present).
    # -----------------------------------------------------------------
    info("Example 4: Compare variables, facet by region (SD)")
    plot_bars(
        ds,
        ["DOC", "temp"],
        regions=REGIONS,
        depth="surface",
        groups=DOC_GROUPS,
        seasons=SEASONS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        facet_by="region",
        x_by="variable",
        hue_by=None,
        error="sd",
        figsize=(14, 10),
        ncols=3,
        title="Surface: variable comparison by region (SD)",
        ylabel="Value",
    )

    # -----------------------------------------------------------------
    # Example 5: Multi-depth comparison (hue by depth)
    #   - Pass depth as a list to enable "depth" as a grouping dimension.
    #   - hue: depth
    #
    # Depth selectors depend on your resolve_da_with_depth implementation.
    #
    # -----------------------------------------------------------------
    info("Example 5: Multi-depth DOC by region, hue by depth (CI95)")
    plot_bars(
        ds,
        ["DOC"],
        regions=REGIONS,
        depth=["surface", "bottom"],
        groups=DOC_GROUPS,
        seasons=SEASONS,
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        x_by="region",
        hue_by="depth",
        facet_by=None,
        error="ci95",
        title="DOC: region x depth (CI95)",
        ylabel="DOC",
    )

    # -----------------------------------------------------------------
    # Example 6: Restrict time window (e.g. summer months only)
    #   - months filters the dataset first, then aggregation happens.
    # -----------------------------------------------------------------
    info("Example 6: Summer-only (months filter) DOC by region (SD)")
    plot_bars(
        ds,
        ["DOC"],
        regions=REGIONS,
        depth="surface",
        groups=DOC_GROUPS,
        seasons=SEASONS,
        months=[6, 7, 8],
        base_dir=BASE_DIR,
        figures_root=FIG_DIR,
        x_by="region",
        hue_by=None,
        facet_by=None,
        error="sd",
        title="DOC (surface): summer-only mean by region (SD)",
        ylabel="DOC",
    )

    info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
