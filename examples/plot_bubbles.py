# examples/run_bubbles_from_file.py
from __future__ import annotations
import os

from fvcomersemviz.plots.bubbles import plot_bubbles

# ---------------- user inputs ----------------
INFILE = "/users/modellers/moja/src/fvcomersem-viz/data/bubbles/1-plume-1b.dat"
FIG_DIR = "/data/proteus1/scratch/moja/projects/"                      # figures root
GASES   = ("CO2")                                       # which gases to render
TIME_WINDOW = None  # e.g. ("2020-06-01 12:00", "2020-06-01 18:00") or None

# visual/performance knobs (optional)
FPS = 12
DPI = 160
MAX_POINTS_PER_FRAME = None
JITTER_FRAC = 0.0
CMAP = "viridis"
FIGSIZE = (12, 9)
EDGE_COLOR = "k"
SIZE_SCALE = 0.6
RECENTER = True
CENTER_MODE = "global"   # or "per-gas"
VERBOSE = True
WRITER_ARGS = {"codec": "h264", "bitrate": 8000}
# ---------------------------------------------


def main() -> None:
    if not os.path.isfile(INFILE):
        print(f"[ubbles] Input file not found: {INFILE}")
        return

    # Derive:
    #   base_dir        = directory containing the input file
    #   run_name        = basename of that directory (used for output folder + filename prefix)
    base_dir = os.path.dirname(os.path.abspath(INFILE))
    run_name = os.path.basename(base_dir)

    print(f"[bubbles] Input file     : {INFILE}")
    print(f"[bubbles] Data directory : {base_dir}")
    print(f"[bubbles] Run name       : {run_name} (derived from parent folder)")
    print(f"[bubbles] FIG root       : {FIG_DIR}")
    print(f"[bubbles] Will save under: {FIG_DIR}/{run_name}/bubbles/")

    outputs = plot_bubbles(
        infile=INFILE,
        gases=GASES,
        time_window=TIME_WINDOW,
        fps=FPS,
        dpi=DPI,
        max_points_per_frame=MAX_POINTS_PER_FRAME,
        jitter_frac=JITTER_FRAC,
        cmap=CMAP,
        figsize=FIGSIZE,
        facecolor="white",
        edgecolor=EDGE_COLOR,
        output_root=FIG_DIR,          # let utils.out_dir(base_dir, FIG_DIR) build folders
        output_basename=run_name,     # filenames start with this
        writer_args=WRITER_ARGS,
        verbose=VERBOSE,
        recenter=RECENTER,
        center_mode=CENTER_MODE,
        size_scale=SIZE_SCALE,
        use_out_dir_helper=True,      # => FIG_DIR/<basename(base_dir)>/bubbles/
    )

    if outputs:
        print("[bubbles] Saved files:")
        for gas, path in outputs.items():
            print(f"  - {gas}: {path}")
    else:
        print("[bubbles] No outputs (no matching gas / empty frames?)")

    print("[bubbles] Done.")


if __name__ == "__main__":
    main()

