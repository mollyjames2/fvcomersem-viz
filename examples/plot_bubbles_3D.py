#!/usr/bin/env python3
"""
scripts/run_bubbles_3d.py

Simple runner for rendering 3D bubble plume animations.

This script:
1. Imports the main plotting function from plots/bubbles_3D.py.
2. Sets input/output parameters directly (no command-line interface).
3. Calls plot_bubbles_3D(), which reads the plume data, builds frames,
   renders 3D animations, and saves one MP4 per gas.

Output directory structure:
    /data/proteus1/scratch/moja/projects/bubbles/
        └── bubbles_3D/
            ├── <basename>__CO2__bubbles3D.mp4
            └── <basename>__CH4__bubbles3D.mp4
"""

from __future__ import annotations
import os
from fvcomersemviz.plots.bubbles_3D import plot_bubbles_3D


def main() -> None:
    """Run a 3D bubble animation job with fixed configuration."""

    # ------------------------------------------------------------------
    #  Configuration
    # ------------------------------------------------------------------

    # Path to your PPlume output text file
    infile = "/data/proteus1/scratch/moja/projects/bubbles/1-plume-1b.dat"

    # List of gases to render (must match gas names in the file)
    gases = ["CO2"]

   
    # Output directory (the script will make a subfolder "bubbles_3D/")
    output_root = "/data/proteus1/scratch/moja/projects/bubbles/"

   
    # ------------------------------------------------------------------
    #  Run the renderer
    # ------------------------------------------------------------------

    print("[runner] Starting 3D bubble animation render...")


    outputs = plot_bubbles_3D(
            infile,
            gases,
            collar_radius_frac=None,   # disable shaping
            collar_softness=0.0,
            relax_iters=0,
            max_push_frac=0.0,
            anchor_strength=0.0,
            size_scale=0.06,          # small bubbles, realistic spacing
            color_by="size", 
            size_color_scale="linear",     # "log", "gamma", "linear"
            size_gamma=0.5,             # only for gamma mode       
            output_root=output_root,
        )



    # ------------------------------------------------------------------
    #  Done: list results
    # ------------------------------------------------------------------
    print("\n[runner] Render complete. Output files:")
    for gas, path in outputs.items():
        print(f"  - {gas}: {path}")

    print("\nAll animations saved under:")
    print(f"  {os.path.join(output_root, 'bubbles_3D')}")


if __name__ == "__main__":
    main()

