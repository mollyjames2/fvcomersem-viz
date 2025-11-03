# fvcomersemviz/plots/bubbles.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

from ..utils import out_dir  # uses FIG_DIR + auto subfolder


# -----------------------------
# Public API (Matplotlib)
# -----------------------------
def plot_bubbles(
    infile: str,
    gases: Sequence[str] | str = ("CO2",),
    *,
    time_window: Tuple[str | pd.Timestamp | None, str | pd.Timestamp | None] | None = None,
    fps: int = 12,
    dpi: int = 160,
    max_points_per_frame: Optional[int] = None,
    jitter_frac: float = 0.0,
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (12, 9),
    facecolor: str | Tuple[float, float, float] = "white",
    edgecolor: str = "k",
    output_root: str = os.environ.get("FIG_DIR", "./figures"),
    output_basename: Optional[str] = None,
    writer_args: Optional[Dict] = None,
    verbose: bool = False,
    # controls
    recenter: bool = True,  # recentre XY so (0,0) sits at the plume source
    center_mode: str = "global",  # 'global' or 'per-gas'
    size_scale: float = 0.6,  # shrink/grow bubbles globally (1.0 ≈ baseline)
    use_out_dir_helper: bool = True,  # set False to use output_root verbatim
) -> Dict[str, str]:
    """
    Create per-gas 3D bubble animations from a PPlume text file (Matplotlib).

    Notes on depth:
      - Input column 5 is **positive downward** (e.g., ~70 m).
      - We plot Z = -depth (negative), matching the MATLAB script.
      - Z-limits are zoomed like MATLAB to the plume band (not up to 0).

    Returns
    -------
    dict : {gas: output_video_path}
    """
    # Normalize gases -> list
    gases = [gases] if isinstance(gases, str) else list(gases)

    # Parse file blocks fast
    blocks = _read_pplume_blocks(infile, verbose=verbose)
    if verbose:
        print(f"[bubbles] File: {infile}")
        print(f"[bubbles] Blocks parsed: {len(blocks)}")

    # Filter by time window if requested
    if time_window is not None:
        start, end = time_window
        start_ts = pd.to_datetime(start) if start is not None else None
        end_ts = pd.to_datetime(end) if end is not None else None
        before = len(blocks)
        blocks = [b for b in blocks if _in_window(b.time, start_ts, end_ts)]
        if verbose:
            print(
                f"[bubbles] Time window: {start_ts} → {end_ts} | kept {len(blocks)}/{before} blocks"
            )

    if not blocks:
        raise ValueError("No blocks found after parsing/time filtering.")

    # ---- Global stats/centre (robust) ----
    all_depth_pos = (
        np.concatenate([b.depth[np.isfinite(b.depth)] for b in blocks if b.depth.size])
        if any(b.depth.size for b in blocks)
        else np.array([0.0])
    )

    # MATLAB z-window logic:
    if np.isfinite(all_depth_pos).any():
        dmax = float(np.nanmax(all_depth_pos))
        dmin = float(np.nanmin(all_depth_pos))
        depth_min_neg = -dmax  # most negative (deeper)
        height_dif = dmax - dmin
        pad = 1.0  # +1 m like the MATLAB code
        zlim = (depth_min_neg, depth_min_neg + height_dif + pad)
    else:
        zlim = (-10.0, 1.0)

    # Robust XY centre (median) so (0,0) sits at the source-ish location
    all_x = (
        np.concatenate([b.x[np.isfinite(b.x)] for b in blocks if b.x.size])
        if any(b.x.size for b in blocks)
        else np.array([0.0])
    )
    all_y = (
        np.concatenate([b.y[np.isfinite(b.y)] for b in blocks if b.y.size])
        if any(b.y.size for b in blocks)
        else np.array([0.0])
    )
    global_cx = float(np.nanmedian(all_x)) if (recenter and all_x.size) else 0.0
    global_cy = float(np.nanmedian(all_y)) if (recenter and all_y.size) else 0.0

    # Global diameter max for consistent color scale
    diam_candidates = [
        np.nanmax(b.size) for b in blocks if b.size.size and np.isfinite(b.size).any()
    ]
    max_diam = float(np.nanmax(diam_candidates)) if diam_candidates else 1.0

    if verbose:
        print(
            f"[bubbles] XY centre (global): ({global_cx:.3f}, {global_cy:.3f})  | size max: {max_diam:0.4g}"
        )
        print(f"[bubbles] Z window (m): {zlim[0]:.3f} .. {zlim[1]:.3f}")

    # Output directory
    base_dir = os.path.dirname(os.path.abspath(infile)) or "."
    if use_out_dir_helper:
        figdir = out_dir(base_dir, output_root)
    else:
        figdir = os.path.abspath(output_root)
        os.makedirs(figdir, exist_ok=True)
    if verbose:
        print(f"[bubbles] Output directory: {figdir}")
    if output_basename is None:
        output_basename = os.path.splitext(os.path.basename(infile))[0]

    # Per-gas videos
    outputs: Dict[str, str] = {}
    for gas in gases:
        # Per-gas XY centre if requested
        if recenter and center_mode.lower() == "per-gas":
            gx_list, gy_list = [], []
            for b in blocks:
                m = b.gas == gas
                if np.any(m):
                    gx_list.append(b.x[m][np.isfinite(b.x[m])])
                    gy_list.append(b.y[m][np.isfinite(b.y[m])])
            gx = np.concatenate(gx_list) if gx_list else np.array([global_cx])
            gy = np.concatenate(gy_list) if gy_list else np.array([global_cy])
            cx, cy = float(np.nanmedian(gx)), float(np.nanmedian(gy))
        else:
            cx, cy = global_cx, global_cy

        # Build frames (vectorized expansion + optional subsample)
        frames = _build_frames(
            blocks,
            gas,
            jitter_frac=jitter_frac,
            max_points=max_points_per_frame,
            cx=cx,
            cy=cy,
        )

        if verbose:
            total_pts = int(sum(fr.x.size for fr in frames))
            nonempty = int(sum(1 for fr in frames if fr.x.size))
            print(
                f"[bubbles] Gas '{gas}': {len(frames)} frames ({nonempty} non-empty), ~{total_pts:,} points"
            )

        if not frames:
            if verbose:
                print(f"[bubbles] Gas '{gas}': no particles; skipping.")
            continue

        out_mp4 = os.path.join(figdir, f"{output_basename}__{gas}__bubbles.mp4")
        if verbose:
            print(f"[bubbles] Rendering {gas} → {out_mp4}")

        _animate_frames(
            frames=frames,
            out_path=out_mp4,
            fps=fps,
            dpi=dpi,
            figsize=figsize,
            facecolor=facecolor,
            cmap=cmap,
            edgecolor=edgecolor,
            clim=(0.0, float(max_diam if np.isfinite(max_diam) else 1.0)),
            zlim=zlim,  # MATLAB-style zoomed window in negative Z
            verbose=verbose,
            writer_args=writer_args or {"codec": "h264", "bitrate": 8000},
            gas_label=gas,
            size_scale=size_scale,
        )
        outputs[gas] = out_mp4

    if verbose:
        print(f"[bubbles] Done. Outputs: {outputs}")

    return outputs


# -----------------------------
# Internals
# -----------------------------
@dataclass
class Block:
    """One header + its detail rows, in numeric arrays (no per-particle expansion yet)."""

    time: pd.Timestamp
    gas: np.ndarray  # shape (n,) of strings like "CO2"
    x: np.ndarray  # east (m)        [col 3]
    y: np.ndarray  # north (m)       [col 4]
    depth: np.ndarray  # POSITIVE DOWN (m)   [col 5]; we plot Z = -depth
    count: np.ndarray  # particle multiplier (rounded) [col 7]
    size: np.ndarray  # diameter proxy for bubble size/color [col 8]


@dataclass
class Frame:
    """Per-frame fully expanded particle arrays (already filtered by gas)."""

    time: pd.Timestamp
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray  # negative (Z = -depth)
    size: np.ndarray  # marker size source values
    color: np.ndarray  # same as size (for c=)


def _in_window(t: pd.Timestamp, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> bool:
    if start is not None and t < start:
        return False
    if end is not None and t > end:
        return False
    return True


def _read_pplume_blocks(path: str, *, verbose: bool = False) -> List[Block]:
    """
    Parse the custom text format quickly with a single pass.

    Header line example:
        72,2020-06-01T12:59:59, 0.662467755E-05, 0.000000000E+00, ...

    After the header, 'count' detail rows follow. Columns used (1-based):
      3 → X (east, m)
      4 → Y (north, m)
      5 → Depth (POSITIVE DOWN)
      7 → Number of particles
      8 → Size (diameter proxy)
     11 → Gas string ("CO2","CH4","H2")
    """
    blocks: List[Block] = []
    parse_num = _fast_float

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    i = 0
    n = len(lines)
    block_index = 0
    while i < n:
        line = lines[i].strip()
        i += 1
        if not line:
            continue

        # Header line: integer count, then ISO8601 time
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue
        try:
            block_count = int(parts[0])
        except Exception:
            # Not a header; skip
            continue

        # Parse timestamp in second token
        if len(parts) < 2:
            continue
        try:
            t = pd.to_datetime(parts[1].replace(" ", "T"))
        except Exception:
            t = pd.NaT

        # Read next 'block_count' detail rows
        rows = lines[i : i + block_count]
        i += block_count
        if not rows:
            continue

        m = len(rows)
        x = np.empty(m, dtype=float)
        y = np.empty(m, dtype=float)
        depth = np.empty(m, dtype=float)
        count = np.empty(m, dtype=float)
        size = np.empty(m, dtype=float)
        gas = np.empty(m, dtype=object)

        for k, r in enumerate(rows):
            cols = [c.strip() for c in r.split(",")]

            def col(idx: int, default: float = np.nan) -> float:
                try:
                    return parse_num(cols[idx])
                except Exception:
                    return default

            x[k] = col(2)  # 3rd column (0-based index 2)
            y[k] = col(3)  # 4th
            depth[k] = col(4)  # 5th (POSITIVE DOWN)
            count[k] = np.rint(col(6, 0.0))  # 7th → round
            size[k] = col(7)  # 8th
            gas[k] = (cols[10] if len(cols) > 10 else "").strip()  # 11th token

        # Clean NaNs and negatives in count; sanitize size
        count = np.where(np.isfinite(count) & (count > 0), count, 0).astype(np.int32)
        size = np.where(np.isfinite(size) & (size > 0), size, np.nan)

        blocks.append(Block(time=t, gas=gas, x=x, y=y, depth=depth, count=count, size=size))
        block_index += 1
        if verbose and (block_index % 100 == 0):
            print(f"[bubbles] Parsed {block_index} blocks...")

    if verbose:
        print(f"[bubbles] Parsed total: {len(blocks)} blocks from '{path}'")
    return blocks


_num_re = re.compile(r"^[\s\+\-]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eEdD][\+\-]?\d+)?$")


def _fast_float(s: str) -> float:
    """Fast float that accepts Fortran 'D' exponents and blanks."""
    if not s:
        return np.nan
    s = s.strip()
    if not s:
        return np.nan
    s = s.replace("D", "E").replace("d", "e")  # FORTRAN exponent
    try:
        if _num_re.match(s) is None:
            return np.nan
        return float(s)
    except Exception:
        return np.nan


def _build_frames(
    blocks: List[Block],
    gas: str,
    *,
    jitter_frac: float = 0.0,
    max_points: Optional[int] = None,
    cx: float = 0.0,
    cy: float = 0.0,
) -> List[Frame]:
    """
    Vectorized per-block expansion (repeat by 'count'), optional jitter, optional subsampling.
    Recenter XY by subtracting (cx, cy). Plot Z = -depth (negative).
    """
    rng = np.random.default_rng(12345)
    frames: List[Frame] = []

    for b in blocks:
        m = b.gas == gas
        if not np.any(m):
            frames.append(
                Frame(b.time, np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0))
            )
            continue

        x, y, depth, count, size = b.x[m], b.y[m], b.depth[m], b.count[m], b.size[m]

        keep = (
            np.isfinite(size) & (count > 0) & np.isfinite(x) & np.isfinite(y) & np.isfinite(depth)
        )
        if not np.any(keep):
            frames.append(
                Frame(b.time, np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0))
            )
            continue

        x, y, depth, count, size = x[keep], y[keep], depth[keep], count[keep], size[keep]

        # Repeat each row by 'count' using np.repeat
        reps = count.astype(np.int64)
        idx = np.repeat(np.arange(reps.size, dtype=np.int64), reps)
        X = x[idx] - cx
        Y = y[idx] - cy
        Z = -depth[idx]  # MATLAB-style: negative Z (0 is near the surface)
        S = size[idx]

        # Optional jitter to de-overlap (scaled by median size)
        if jitter_frac > 0.0 and X.size:
            s_med = float(np.nanmedian(S)) if np.isfinite(S).any() else 1.0
            jitter = jitter_frac * s_med
            X = X + (rng.random(X.size) - 0.5) * 2.0 * jitter
            Y = Y + (rng.random(Y.size) - 0.5) * 2.0 * jitter

        # Optional random subsample for speed
        if max_points is not None and X.size > max_points:
            sel = rng.choice(X.size, size=max_points, replace=False)
            X, Y, Z, S = X[sel], Y[sel], Z[sel], S[sel]

        frames.append(Frame(time=b.time, x=X, y=Y, z=Z, size=S, color=S))

    return frames


def _animate_frames(
    *,
    frames: List[Frame],
    out_path: str,
    fps: int,
    dpi: int,
    figsize: Tuple[float, float],
    facecolor,
    cmap: str,
    edgecolor: str,
    clim: Tuple[float, float],
    zlim: Tuple[float, float],
    verbose: bool,
    writer_args: Dict,
    gas_label: str,
    size_scale: float,
) -> None:
    # --- Shared XY limits: identical numeric range for East/North, centered on 0 ---
    def shared_xy_limits(
        frames: List[Frame], pad: float = 0.05
    ) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        xs = (
            np.concatenate([fr.x[np.isfinite(fr.x)] for fr in frames if fr.x.size])
            if any(fr.x.size for fr in frames)
            else np.array([])
        )
        ys = (
            np.concatenate([fr.y[np.isfinite(fr.y)] for fr in frames if fr.y.size])
            if any(fr.y.size for fr in frames)
            else np.array([])
        )
        if xs.size + ys.size == 0:
            span = 1.0
        else:
            span = float(np.nanmax(np.abs(np.concatenate([xs, ys]))))
            span *= 1.0 + pad  # breathing room
        return (-span, span), (-span, span), span

    xlim, ylim, xy_span = shared_xy_limits(frames)

    # --- Figure & layout (manual margins beat constrained_layout for 3D) ---
    fig = plt.figure(figsize=figsize, facecolor=facecolor)  # use requested facecolor
    # Leave space on the right for colorbar and extra margin so 3D box never touches edges
    fig.subplots_adjust(left=0.12, right=0.86, bottom=0.12, top=0.92)

    ax = fig.add_subplot(111, projection="3d", proj_type="persp")
    ax.set_facecolor(facecolor)

    # Remove pane fills (no tinted planes); keep faint edges
    try:
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor((1.0, 1.0, 1.0, 0.0))  # transparent
            pane.set_edgecolor((0.80, 0.80, 0.80, 1.0))  # subtle borders
    except Exception:
        pass

    # Axis cosmetics
    ax.grid(False)
    ax.set_xlabel("East (m)", fontweight="bold", labelpad=12)
    ax.set_ylabel("North (m)", fontweight="bold", labelpad=12)
    ax.set_zlabel("Depth (m)", fontweight="bold", labelpad=26)  # push label away from markers
    ax.tick_params(pad=7)  # move tick labels off the axes a touch

    # Limits & aspect
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)  # negative values; surface near 0
    # Keep X:Y numeric scales identical; Z relative to XY span
    z_span = max(1e-6, (zlim[1] - zlim[0]))
    # 3D aspect uses relative data ranges; XY equal, Z scaled to its range vs XY
    ax.set_box_aspect((1.0, 1.0, z_span / (2.0 * xy_span) if xy_span > 0 else 1.0))
    # Lock limits to prevent autoscale drift during updates
    ax.set_autoscale_on(False)

    # Viewing angle
    ax.view_init(elev=20, azim=-30)

    # Scatter artist (start empty)
    scat = ax.scatter([], [], [], s=[], c=[], cmap=cmap, edgecolors=edgecolor, linewidths=0.2)

    # Dedicated colorbar axis (keeps box inside figure margins)
    cax = fig.add_axes([0.88, 0.18, 0.028, 0.64])  # [left, bottom, width, height]
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_clim(*clim)
    cb = plt.colorbar(mappable, cax=cax)
    cb.set_label("Bubble diameter (a.u.)")

    # Title
    title = ax.set_title(f"{gas_label}", pad=10)

    def init():
        scat._offsets3d = (np.empty(0), np.empty(0), np.empty(0))
        scat.set_sizes(np.empty(0))
        scat.set_array(np.empty(0))
        return scat, title

    def update(i):
        fr = frames[i]
        scat._offsets3d = (fr.x, fr.y, fr.z)
        scat.set_sizes(_diameter_to_ms(fr.size, scale=size_scale))
        scat.set_array(fr.color)  # for colormap
        if not pd.isna(fr.time):
            title.set_text(f"{gas_label} — {pd.to_datetime(fr.time).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            title.set_text(f"{gas_label}")
        if verbose and (i % 10 == 0 or i == len(frames) - 1):
            print(f"[bubbles]   frame {i + 1}/{len(frames)}")
        return scat, title

    anim = FuncAnimation(
        fig, update, init_func=init, frames=len(frames), interval=1000 / fps, blit=False
    )

    writer = FFMpegWriter(fps=fps, **writer_args)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # Save with requested facecolor
    anim.save(out_path, writer=writer, dpi=dpi, savefig_kwargs={"facecolor": facecolor})
    if verbose:
        print(f"[bubbles] Saved: {out_path}")
    plt.close(fig)


def _diameter_to_ms(d: np.ndarray, scale: float = 0.6) -> np.ndarray:
    """
    Map 'diameter-like' values to marker sizes (points^2).
    Gentler mapping with an explicit scale knob.
    """
    if d.size == 0:
        return d
    d = np.asarray(d, dtype=float)
    d = np.where(np.isfinite(d) & (d > 0), d, np.nan)

    med = np.nanmedian(d)
    if not np.isfinite(med) or med <= 0:
        med = 1.0

    # Normalise and soften extremes
    s = np.clip(d / med, 0.05, 6.0)

    # Base size ~ 8pt at median; 'scale' shrinks/grows globally
    pts2 = (8.0 * np.sqrt(s) * scale) ** 2  # points^2
    return pts2
