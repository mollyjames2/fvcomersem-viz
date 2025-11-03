#!/usr/bin/env python3
"""
bubbles_3D_orbit_realistic.py — Realistic 3D bubble plume renderer (PyVista)

Features:
- Keeps plume geometry as provided (no artificial shaping unless explicitly enabled)
- Physically plausible glossy spheres with smooth shading and perspective
- Realistic seabed (relief shading; optional texture/PBR)
- Flat east/north arrows at the source on the seabed
- Flat rulers with metre tick labels on seabed (E/N)
- Side Z-depth bar that sits away from the plume
- Bubble diameter colorbar (a.u.) with stable global range
- Camera orbit with framing controls (fov, elev, distance_scale, center_z_frac, cam_z_lift)

"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict
import numpy as np
import pandas as pd
import pyvista as pv


# =============================
# Data structures
# =============================
@dataclass
class Block:
    time: pd.Timestamp
    gas: np.ndarray
    x: np.ndarray
    y: np.ndarray
    depth: np.ndarray  # positive down (m)
    count: np.ndarray
    size: np.ndarray  # diameter-like (a.u.)


@dataclass
class Frame:
    time: pd.Timestamp
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray  # negative (Z = -depth)
    size: np.ndarray


# =============================
# Parsing
# =============================
_num_re = re.compile(r"^[\s\+\-]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eEdD][\+\-]?\d+)?$")


def _fast_float(s: str) -> float:
    if not s:
        return np.nan
    s = s.strip().replace("D", "E").replace("d", "e")
    try:
        if _num_re.match(s) is None:
            return np.nan
        return float(s)
    except Exception:
        return np.nan


def read_pplume_blocks(path: str, verbose: bool = False) -> List[Block]:
    blocks: List[Block] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        try:
            block_count = int(parts[0])
        except Exception:
            continue
        t = pd.to_datetime(parts[1].replace(" ", "T")) if len(parts) > 1 else pd.NaT
        rows = lines[i : i + block_count]
        i += block_count
        if not rows:
            continue

        m = len(rows)
        x = np.empty(m, float)
        y = np.empty(m, float)
        depth = np.empty(m, float)
        count = np.empty(m, float)
        size = np.empty(m, float)
        gas = np.empty(m, object)

        for k, r in enumerate(rows):
            cols = [c.strip() for c in r.split(",")]

            def col(idx: int, default=np.nan):
                try:
                    return _fast_float(cols[idx])
                except Exception:
                    return default

            x[k] = col(2)
            y[k] = col(3)
            depth[k] = col(4)
            count[k] = col(6, 0.0)
            size[k] = col(7)
            gas[k] = cols[10] if len(cols) > 10 else ""

        gas = np.char.upper(np.char.strip(gas.astype(str)))
        count = np.where(np.isfinite(count) & (count > 0), count, 0.0)
        size = np.where(np.isfinite(size) & (size > 0), size, np.nan)

        blocks.append(Block(time=t, gas=gas, x=x, y=y, depth=depth, count=count, size=size))

    if verbose:
        gases = sorted({g for b in blocks for g in set(b.gas.tolist()) if g})
        print(f"[bubbles_3D_orbit] Parsed {len(blocks)} blocks from {path}")
        print(f"[bubbles_3D_orbit] Gases detected: {gases}")
    return blocks


# =============================
# Frame building
# =============================
def build_frames(
    blocks: List[Block],
    gas: str,
    *,
    jitter_frac: float = 0.0,
    max_points: Optional[int] = None,
    cx: float = 0.0,
    cy: float = 0.0,
) -> List[Frame]:
    gas_norm = gas.strip().upper()
    rng = np.random.default_rng(12345)
    frames: List[Frame] = []

    for b in blocks:
        m = b.gas == gas_norm
        if not np.any(m):
            frames.append(Frame(b.time, *(np.empty(0),) * 3, np.empty(0)))
            continue

        x, y, depth, count, size = b.x[m], b.y[m], b.depth[m], b.count[m], b.size[m]
        keep = np.isfinite(size) & (size > 0) & np.isfinite(x) & np.isfinite(y) & np.isfinite(depth)
        if not np.any(keep):
            frames.append(Frame(b.time, *(np.empty(0),) * 3, np.empty(0)))
            continue
        x, y, depth, count, size = x[keep], y[keep], depth[keep], count[keep], size[keep]

        reps = np.maximum(1, np.round(count).astype(np.int64))
        idx = np.repeat(np.arange(reps.size, dtype=np.int64), reps)

        X = x[idx] - cx
        Y = y[idx] - cy
        Z = -depth[idx]
        S = size[idx]

        if jitter_frac > 0.0 and X.size:
            s_med = float(np.nanmedian(S)) if np.isfinite(S).any() else 1.0
            jitter = jitter_frac * s_med
            X += (rng.random(X.size) - 0.5) * 2.0 * jitter
            Y += (rng.random(Y.size) - 0.5) * 2.0 * jitter
            Z += (rng.random(Z.size) - 0.5) * 0.02 * jitter

        if max_points is not None and X.size > max_points:
            sel = rng.choice(X.size, size=max_points, replace=False)
            X, Y, Z, S = X[sel], Y[sel], Z[sel], S[sel]

        frames.append(Frame(time=b.time, x=X, y=Y, z=Z, size=S))
    return frames


# =============================
# Utilities
# =============================
def size_to_radius_mpl_like(d: np.ndarray, scale_world: float = 0.08) -> np.ndarray:
    """Radius ∝ sqrt(size/median)."""
    d = np.asarray(d, float)
    d = np.where(np.isfinite(d) & (d > 0), d, np.nan)
    med = np.nanmedian(d) if np.isfinite(d).any() else 1.0
    s = np.clip(d / med, 0.05, 6.0)
    r = np.sqrt(s) * scale_world
    return np.where(np.isfinite(r), r, 0.0)


def soften_radial_collar(P, *, cx=0.0, cy=0.0, r_cap=0.35, softness=0.15):
    """Softly clamp XY radius around a centerline (unused unless you enable)."""
    XY = P[:, :2] - np.array([cx, cy])
    r = np.linalg.norm(XY, axis=1)
    mask = r > r_cap
    if np.any(mask):
        r_old = r[mask]
        r_new = r_cap + softness * (r_old - r_cap)
        XY[mask] *= (r_new / np.maximum(r_old, 1e-12))[:, None]
    P[:, :2] = XY + np.array([cx, cy])
    return P


def _transform_sizes_for_color(
    s: np.ndarray, mode: str = "linear", gamma: float = 0.5
) -> np.ndarray:
    """Return a transformed copy of s for coloring only."""
    s = np.asarray(s, float)
    s = np.where(np.isfinite(s) & (s > 0), s, np.nan)
    if mode == "log":
        # avoid -inf: clip to a small positive floor based on finite data
        floor = np.nanmin(s[s > 0]) if np.isfinite(s).any() else 1e-12
        s = np.log10(np.clip(s, floor, None))
    elif mode == "gamma":
        # normalize by median then apply power <1 (boost small values)
        med = np.nanmedian(s) if np.isfinite(s).any() else 1.0
        s = np.power(np.clip(s / max(med, 1e-12), 1e-6, None), gamma)
    # linear: unchanged
    return s


# =============================
# Rendering (realistic seabed, guides, z-bar, colorbar)
# =============================
def render_orbit(
    frames: List[Frame],
    out_path: str,
    *,
    fps: int = 24,
    size_scale: float = 0.006,
    # optional shaping
    collar_radius_frac: Optional[float] = None,
    collar_softness: float = 0.0,
    relax_iters: int = 0,
    max_push_frac: float = 0.0,
    anchor_strength: float = 0.0,
    # coloring
    color_by: Optional[str] = "size",
    size_color_scale: str = "linear",  # "linear" | "log" | "gamma"
    size_gamma: float = 0.5,  # only used when size_color_scale == "gamma"
    show_colorbar: bool = True,
    # camera
    fov: float = 30.0,
    elev: float = 26.0,
    distance_scale: float = 2,
    center_z_frac: float = 0.55,
    cam_z_lift: float = 0.0,
    # seabed
    show_seabed: bool = True,
    seabed_color: Tuple[float, float, float] = (0.38, 0.26, 0.15),
    seabed_size_factor: float = 5,
    seabed_margin_z: float = 0.05,
    seabed_roughness: float = 0.06,
    seabed_res: int = 300,
    seabed_texture: Optional[str] = None,
    seabed_pbr: bool = True,
    # seabed guides (compatibility)
    show_axes_lines: bool = True,
    show_rulers: bool = True,
    show_zbar: bool = True,
    ruler_target_ticks: int = 8,
    zbar_corner: str = "NE",
    zbar_color: str = "white",
    zbar_label_color: str = "black",
    # on-screen HUD overlay
    show_hud: bool = True,
    hud_font_size: int = 15,
    hud_color: str = "white",
    hud_time_fmt: str = "%Y-%m-%d %H:%M:%S",
    label_gas: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Render cinematic 3D bubble plume orbit with seabed, rulers, depth bar, colorbar, and HUD timestamp."""

    if not frames:
        raise ValueError("No frames to render")

    # --- bounds ---
    xs = (
        np.concatenate([fr.x for fr in frames if fr.x.size])
        if any(fr.x.size for fr in frames)
        else np.array([0.0])
    )
    ys = (
        np.concatenate([fr.y for fr in frames if fr.y.size])
        if any(fr.y.size for fr in frames)
        else np.array([0.0])
    )
    zs = (
        np.concatenate([fr.z for fr in frames if fr.z.size])
        if any(fr.z.size for fr in frames)
        else np.array([0.0])
    )
    x_span = float(np.ptp(xs))
    y_span = float(np.ptp(ys))
    z_span = float(np.ptp(zs))
    full_span = max(x_span, y_span, z_span, 1.0)
    z_min, z_max = float(np.nanmin(zs)), float(np.nanmax(zs))
    z_span_nonzero = max(z_max - z_min, 1e-6)
    center = (
        float(np.nanmean(xs)),
        float(np.nanmean(ys)),
        z_min + np.clip(center_z_frac, 0, 1) * z_span_nonzero,
    )
    distance = distance_scale * full_span

    # --- plot setup ---
    pv.set_plot_theme("default")
    pl = pv.Plotter(off_screen=True, window_size=(1280, 720))
    pl.set_background((0.02, 0.08, 0.20), top=(0.05, 0.15, 0.35))
    pl.enable_anti_aliasing()
    pl.enable_depth_peeling()
    key = pv.Light(position=(0, -full_span, full_span), intensity=1.0, color="white")
    fill = pv.Light(
        position=(full_span, full_span, 0.8 * full_span), intensity=0.50, color=(0.6, 0.7, 1.0)
    )
    rim = pv.Light(
        position=(-full_span, 0, -0.4 * full_span), intensity=0.30, color=(0.5, 0.7, 1.0)
    )
    pl.add_light(key)
    pl.add_light(fill)
    pl.add_light(rim)

    # --- seabed ---
    seabed_z = None
    if show_seabed and xs.size:
        xy_span = max(x_span, y_span, 1.0)
        plane_size = seabed_size_factor * xy_span
        seabed_z = z_min - seabed_margin_z * z_span

        seabed = pv.Plane(
            center=(center[0], center[1], seabed_z),
            direction=(0, 0, 1),
            i_size=plane_size,
            j_size=plane_size,
            i_resolution=seabed_res,
            j_resolution=seabed_res,
        )

        # === roughness: small height variations (unitless -> metres via z_span) ===
        P = seabed.points.copy()
        # map to [0,1] across plane so the pattern is scale independent
        Xi = ((P[:, 0] - center[0]) / (plane_size / 2) + 1.0) / 2.0
        Yi = ((P[:, 1] - center[1]) / (plane_size / 2) + 1.0) / 2.0
        # multi-freq ripples + a bit of cross term; centered around 0
        noise = (
            np.sin(6 * np.pi * Xi)
            + np.sin(6 * np.pi * Yi)
            + 0.5 * np.sin(13 * np.pi * Xi + 0.3)
            + 0.5 * np.sin(11 * np.pi * Yi + 1.1)
            + 0.25 * np.sin(17 * np.pi * (Xi + Yi))
        )
        noise = (noise - noise.min()) / (noise.max() - noise.min()) - 0.5
        seabed["elev"] = noise
        seabed = seabed.warp_by_scalar("elev", factor=seabed_roughness * z_span)

        # brown material (looks like seafloor)
        seabed_actor = pl.add_mesh(
            seabed,
            color=seabed_color,  # e.g. (0.38, 0.26, 0.15)
            smooth_shading=True,
        )
        # slightly matte
        seabed_actor.prop.diffuse = 0.95
        seabed_actor.prop.specular = 0.02
        seabed_actor.prop.specular_power = 10

        # Flat seabed guides (E/N + rulers)
        if show_axes_lines or show_rulers:
            # --- guides sit just above the highest warped seabed triangle ---
            z_on_bed = seabed.bounds[5] + 0.003 * z_span
            txt_eps = 0.001 * z_span  # tiny lift to avoid z-fighting
            xy_span = max(x_span, y_span, 1.0)

            def add_tube_line(a_xy, b_xy, radius):
                tube = pv.Line((a_xy[0], a_xy[1], z_on_bed), (b_xy[0], b_xy[1], z_on_bed)).tube(
                    radius=radius, n_sides=18
                )
                pl.add_mesh(tube, color="white", lighting=False)

            def add_arrow_tip(end_xy, dir_xy, tip_len, wing_span):
                d = np.array(dir_xy, float)
                n = np.linalg.norm(d)
                d = d / n if n else d
                side = np.array([-d[1], d[0]])
                tip = np.array(end_xy)
                a = tip - d * tip_len
                p1 = tip
                p2 = a + side * wing_span
                p3 = a - side * wing_span
                tri = pv.PolyData(
                    np.array(
                        [
                            [p1[0], p1[1], z_on_bed],
                            [p2[0], p2[1], z_on_bed],
                            [p3[0], p3[1], z_on_bed],
                        ],
                        float,
                    )
                )
                tri.faces = np.hstack([[3, 0, 1, 2]])
                pl.add_mesh(tri, color="white", lighting=False)

            def add_flat_text(text, pos, height, *, yaw_deg=0.0, color="white"):
                t = pv.Text3D(text)
                if yaw_deg:
                    t.rotate_z(yaw_deg, point=(0, 0, 0), inplace=True)
                # give the text a hair of thickness so it never vanishes
                t.scale(
                    [height, height, height * 0.02], inplace=True, transform_all_input_vectors=False
                )
                t.translate([pos[0], pos[1], pos[2] + txt_eps], inplace=True)
                actor = pl.add_mesh(t, color=color, lighting=False)
                # ensure neither face is culled
                try:
                    actor.prop.culling = "none"
                except Exception:
                    pass

            def nice_step(span, target=8):
                raw = span / max(target, 1)
                p10 = 10 ** np.floor(np.log10(raw))
                for m in (1, 2, 5, 10):
                    if raw <= m * p10:
                        return m * p10
                return 10 * p10

            # geometry
            axis_len = 0.60 * xy_span
            axis_rad = 0.004 * xy_span
            tick_half = 0.018 * xy_span
            tick_rad = 0.0035 * xy_span
            tip_len = 0.035 * xy_span
            wing_span = 0.020 * xy_span
            label_h = 0.090 * xy_span  # "N" and "E" height (bigger)
            num_h = 0.045 * xy_span  # numeric label height (bigger)
            post_offset = 0.075 * xy_span  # "N/E" beyond arrow head
            num_offset = 0.040 * xy_span  # distance of numbers from axis

            # ticks every "nice" step, but guarantee at least ~2 ticks
            step_nice = nice_step(axis_len, target=ruler_target_ticks)
            min_ticks = 2
            fallback_step = axis_len / (min_ticks + 1)  # 2 ticks = 3 segments
            # never let the step be bigger than half the axis
            step = min(step_nice, axis_len / 2.0)
            if step <= 0 or step > axis_len - 1e-9:
                step = fallback_step

            tickvals = np.arange(step, axis_len + 1e-9, step)  # skip 0 at the origin

            # format tick labels with decimals if step < 1
            def tick_label(val, step):
                if step >= 1.0:
                    return f"{int(round(val))} m"
                elif step >= 0.1:
                    return f"{val:.1f} m"
                else:
                    return f"{val:.2f} m"

            # +E axis ------------------------------------------------------------
            p0 = np.array([center[0], center[1]])
            pE = p0 + np.array([axis_len, 0.0])
            add_tube_line(p0, pE, axis_rad)
            add_arrow_tip(pE, (1, 0), tip_len, wing_span)
            # "E" just beyond the arrowhead
            add_flat_text(
                "E",
                (pE[0] + post_offset, pE[1], z_on_bed),
                height=label_h,
                yaw_deg=0.0,
                color="white",
            )

            # ticks & numbers along +E
            for i, tval in enumerate(tickvals):
                x = p0[0] + tval
                # tick mark
                add_tube_line((x, p0[1] - tick_half), (x, p0[1] + tick_half), tick_rad)
                # only draw numeric label every second tick to reduce clutter
                if i % 2 == 0 or len(tickvals) <= 4:
                    add_flat_text(
                        tick_label(tval, step),
                        (x, p0[1] + 1.3 * num_offset, z_on_bed),
                        height=0.8 * num_h,
                        yaw_deg=0.0,
                        color="black",
                    )

            # +N axis ------------------------------------------------------------
            pN = p0 + np.array([0.0, axis_len])
            add_tube_line(p0, pN, axis_rad)
            add_arrow_tip(pN, (0, 1), tip_len, wing_span)
            # "N" just beyond the arrowhead
            add_flat_text(
                "N",
                (pN[0], pN[1] + post_offset, z_on_bed),
                height=label_h,
                yaw_deg=90.0,
                color="white",
            )

            # ticks & numbers along +N
            for i, tval in enumerate(tickvals):
                y = p0[1] + tval
                add_tube_line((p0[0] - tick_half, y), (p0[0] + tick_half, y), tick_rad)
                if i % 2 == 0 or len(tickvals) <= 4:
                    add_flat_text(
                        tick_label(tval, step),
                        (p0[0] + 1.3 * num_offset, y, z_on_bed),
                        height=0.8 * num_h,
                        yaw_deg=90.0,
                        color="black",
                    )

            # Origin "0 m" label exactly in the corner of both axes
            add_flat_text(
                "0 m",
                (p0[0] - 0.4 * num_offset, p0[1] - 0.4 * num_offset, z_on_bed),
                height=num_h,
                yaw_deg=0.0,
                color="black",
            )

    # --- Z depth bar: plain white with black labels ---
    if show_zbar and xs.size:
        xy_span = max(x_span, y_span, 1.0)
        corner = (zbar_corner or "NE").upper()

        # place the bar a bit away from the plume
        x_off = 0.58 * xy_span if "E" in corner else -0.58 * xy_span
        y_off = 0.58 * xy_span if "N" in corner else -0.58 * xy_span
        bx, by = center[0] + x_off, center[1] + y_off

        # main vertical bar (no scalars -> no colormap)
        bar_line = pv.Line((bx, by, z_min), (bx, by, z_max))
        bar = bar_line.tube(radius=0.008 * xy_span, n_sides=24)
        pl.add_mesh(bar, color="white", lighting=False)

        # ticks + labels (right side of bar)
        nticks = 6
        zs_ticks = np.linspace(z_min, z_max, nticks)
        tick_half = 0.018 * xy_span
        label_dx = 0.12 * xy_span  # how far labels sit from bar

        for zt in zs_ticks:
            tick = pv.Line((bx - tick_half, by, zt), (bx + tick_half, by, zt))
            pl.add_mesh(tick, color="white", line_width=3, lighting=False)
            pl.add_point_labels(
                [(bx + label_dx, by, zt)],  # put labels just to the right
                [f"{-zt:.0f} m"],  # positive down
                text_color=zbar_label_color,
                font_size=12,
                shape=None,
                always_visible=True,
            )

    # --- HUD overlay ---
    hud_name = "hud_top_left"
    if show_hud:
        pl.add_text(
            "", position="upper_left", font_size=hud_font_size, color=hud_color, name=hud_name
        )

    # --- bubble actor ---
    glyph_geom = pv.Sphere(theta_resolution=24, phi_resolution=24, radius=1.0)
    actor = None
    scalars_name = None
    if color_by:
        scalars_name = "size" if color_by.lower() == "size" else "z_for_color"

    # consistent color scale
    all_sizes = (
        np.concatenate([f.size[np.isfinite(f.size)] for f in frames if f.size.size])
        if any(f.size.size for f in frames)
        else np.array([1.0])
    )
    smin, smax = float(np.nanmin(all_sizes)), float(np.nanmax(all_sizes))
    if not np.isfinite(smin) or not np.isfinite(smax) or smin == smax:
        smin, smax = 0.1, 1.0

    all_sizes_t = _transform_sizes_for_color(all_sizes, mode=size_color_scale, gamma=size_gamma)
    cmin, cmax = float(np.nanmin(all_sizes_t)), float(np.nanmax(all_sizes_t))
    if not np.isfinite(cmin) or not np.isfinite(cmax) or cmin == cmax:
        cmin, cmax = 0.0, 1.0

    # choose a clear, perceptual map
    cmap_name = "viridis"

    # scalar bar title that reflects the transform
    if size_color_scale == "log":
        size_title = "Bubble diameter (a.u., log₁₀)"
    elif size_color_scale == "gamma":
        size_title = f"Bubble diameter (a.u., γ={size_gamma:g})"
    else:
        size_title = "Bubble diameter (a.u.)"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pl.open_movie(out_path, framerate=fps)
    total_frames = len(frames)

    for i, fr in enumerate(frames):
        if fr.x.size:
            P = np.column_stack([fr.x, fr.y, fr.z])
            R = size_to_radius_mpl_like(fr.size, scale_world=size_scale)
            pts = pv.PolyData(P)
            pts["radius"] = R

            if color_by:
                if color_by.lower() == "size":
                    size_t = _transform_sizes_for_color(
                        fr.size, mode=size_color_scale, gamma=size_gamma
                    )
                    pts["size_color"] = size_t
                    scalars_name = "size_color"
                elif color_by.lower() == "z_for_color":
                    pts["z_for_color"] = P[:, 2]
                    scalars_name = "z_for_color"
                else:
                    scalars_name = None
            else:
                scalars_name = None

            glyphs = pts.glyph(geom=glyph_geom, scale="radius", orient=False)
            if actor is None:
                if color_by is None or scalars_name is None:
                    actor = pl.add_mesh(
                        glyphs,
                        color=(0.96, 0.98, 1.0),
                        lighting=True,
                        smooth_shading=True,
                        opacity=0.25,
                    )
                    actor.mapper.ScalarVisibilityOff()
                else:
                    scalar_bar_args = dict(
                        title=size_title if scalars_name == "size_color" else "Depth (m)",
                        vertical=True,
                        title_font_size=14,
                        label_font_size=11,
                        n_labels=5,
                        fmt="%.2g",  # simple compact formatting
                        position_x=0.92,
                        position_y=0.25,
                        width=0.03,
                        height=0.5,
                        color="white",
                    )
                    actor = pl.add_mesh(
                        glyphs,
                        scalars=scalars_name,
                        cmap=cmap_name,
                        clim=(cmin, cmax) if scalars_name == "size_color" else (smin, smax),
                        scalar_bar_args=scalar_bar_args,
                        lighting=True,
                        opacity=1.0,
                        smooth_shading=True,
                    )

                    # remove the built-in title and add our own vertical one in pixel coords
                    # ---- custom vertical scalar-bar title (hide built-in title) ----
                    sbar_title = size_title if scalars_name == "size_color" else "Depth (m)"

                    # find the scalar bar actor we just added
                    sbar = None
                    try:
                        sbar = pl.scalar_bars[sbar_title]  # most PyVista versions
                    except Exception:
                        try:
                            sbar = next(iter(pl.scalar_bars.values()))  # fallback: first/only bar
                        except Exception:
                            sbar = None

                    if sbar is not None:
                        try:
                            sbar.SetTitle("")  # hide built-in title
                        except Exception:
                            pass

                        # add our own vertical title actor in pixel coords
                        try:
                            try:
                                from pyvista import _vtk
                            except Exception:
                                import vtk as _vtk

                            ta = _vtk.vtkTextActor()
                            ta.SetInput(sbar_title)

                            tp = ta.GetTextProperty()
                            tp.SetColor(1, 1, 1)
                            tp.SetBold(False)
                            tp.SetShadow(False)
                            tp.SetOrientation(90)
                            tp.SetJustificationToCentered()
                            tp.SetVerticalJustificationToCentered()

                            win_w, win_h = pl.window_size
                            pos_x = scalar_bar_args.get("position_x", 0.92)
                            pos_y = scalar_bar_args.get("position_y", 0.25)
                            width = scalar_bar_args.get("width", 0.03)
                            height = scalar_bar_args.get("height", 0.5)

                            # a little to the right of the bar, vertically centered
                            px = int((pos_x + width) * win_w + 12)  # "+ 12" nudges to the right
                            py = int((pos_y + 0.5 * height) * win_h)
                            ta.SetDisplayPosition(px, py)

                            ta.SetDisplayPosition(px, py)

                            pl.add_actor(ta)
                        except Exception:
                            pass

                actor.prop.specular = 0.15
                actor.prop.specular_power = 20
            else:
                actor.mapper.SetInputData(glyphs)
                if color_by and scalars_name:
                    actor.mapper.SetScalarModeToUsePointFieldData()
                    actor.mapper.SelectColorArray(scalars_name)
                    if scalars_name == "size_color":
                        actor.mapper.SetScalarRange(cmin, cmax)
                    else:
                        actor.mapper.SetScalarRange(smin, smax)
                    actor.mapper.Modified()

        # update HUD
        if show_hud:
            if fr.time is not None and pd.notna(fr.time):
                ts = pd.to_datetime(fr.time).strftime(hud_time_fmt)
            else:
                ts = "—"
            txt = f"{label_gas}: {ts}" if label_gas else ts
            pl.add_text(
                txt, position="upper_left", font_size=hud_font_size, color=hud_color, name=hud_name
            )

        # camera orbit
        az = 360.0 * (i / max(1, total_frames - 1))
        azr, elr = np.deg2rad(az), np.deg2rad(elev)
        dx = distance * np.cos(azr) * np.cos(elr)
        dy = distance * np.sin(azr) * np.cos(elr)
        dz = distance * np.sin(elr)
        pl.camera.position = (center[0] + dx, center[1] + dy, center[2] + dz + cam_z_lift)
        pl.camera.focal_point = center
        pl.camera.up = (0, 0, 1)
        pl.camera.view_angle = fov

        pl.write_frame()
        if verbose and (i % 50 == 0 or i == total_frames - 1):
            print(f"[bubbles_3D_orbit] frame {i + 1}/{total_frames} az={az:.1f}° npts={fr.x.size}")

    pl.close()
    if verbose:
        print(f"[bubbles_3D_orbit] Saved: {out_path}")


# =============================
# Public entry
# =============================
def plot_bubbles_3D(
    infile: str,
    gases: Sequence[str] | str = ("CO2",),
    *,
    fps: int = 24,
    max_points_per_frame: Optional[int] = None,
    jitter_frac: float = 0.0,
    size_scale: float = 0.006,
    # optional shaping (normally OFF)
    collar_radius_frac: Optional[float] = None,
    collar_softness: float = 0.0,
    relax_iters: int = 0,
    max_push_frac: float = 0.0,
    anchor_strength: float = 0.0,
    # color and bars
    color_by: Optional[str] = "size",
    size_color_scale: str,
    size_gamma: float = 0.5,
    show_colorbar: bool = True,
    # camera defaults
    fov: float = 30.0,
    elev: float = 26.0,
    distance_scale: float = 2.5,
    center_z_frac: float = 0.55,
    cam_z_lift: float = 0.0,
    # seabed/guides/zbar
    show_seabed: bool = True,
    seabed_color: Tuple[float, float, float] = (0.55, 0.35, 0.20),
    seabed_size_factor: float = 5.0,
    seabed_margin_z: float = 0.05,
    seabed_roughness: float = 0.02,
    seabed_res: int = 300,
    seabed_texture: Optional[str] = None,
    seabed_pbr: bool = True,
    show_axes_lines: bool = True,
    show_rulers: bool = True,
    ruler_target_ticks: int = 8,
    show_zbar: bool = True,
    zbar_corner: str = "NE",
    zbar_color: str = "white",
    zbar_label_color: str = "white",
    # io
    output_root: str = "./figures",
    output_basename: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    gases = [gases] if isinstance(gases, str) else list(gases)
    blocks = read_pplume_blocks(infile, verbose=verbose)

    # global recenter (median)
    all_x = (
        np.concatenate([b.x for b in blocks if b.x.size])
        if any(b.x.size for b in blocks)
        else np.array([0.0])
    )
    all_y = (
        np.concatenate([b.y for b in blocks if b.y.size])
        if any(b.y.size for b in blocks)
        else np.array([0.0])
    )
    cx = float(np.nanmedian(all_x)) if all_x.size else 0.0
    cy = float(np.nanmedian(all_y)) if all_y.size else 0.0

    if output_basename is None:
        output_basename = os.path.splitext(os.path.basename(infile))[0]
    os.makedirs(output_root, exist_ok=True)

    outputs: Dict[str, str] = {}
    for gas in gases:
        frames = build_frames(
            blocks, gas, jitter_frac=jitter_frac, max_points=max_points_per_frame, cx=cx, cy=cy
        )

        out_path = os.path.join(output_root, f"{output_basename}__{gas}__orbit3D.mp4")

        render_orbit(
            frames,
            out_path,
            fps=fps,
            size_scale=size_scale,
            color_by=color_by,
            size_color_scale=size_color_scale,
            size_gamma=size_gamma,
            show_colorbar=show_colorbar,
            fov=fov,
            elev=elev,
            distance_scale=distance_scale,
            center_z_frac=center_z_frac,
            cam_z_lift=cam_z_lift,
            show_seabed=show_seabed,
            seabed_color=seabed_color,
            seabed_size_factor=seabed_size_factor,
            seabed_margin_z=seabed_margin_z,
            seabed_roughness=seabed_roughness,
            seabed_res=seabed_res,
            seabed_texture=seabed_texture,
            seabed_pbr=seabed_pbr,
            show_axes_lines=show_axes_lines,
            show_rulers=show_rulers,
            ruler_target_ticks=ruler_target_ticks,
            show_zbar=show_zbar,
            zbar_corner=zbar_corner,
            zbar_color=zbar_color,
            zbar_label_color=zbar_label_color,
            verbose=verbose,
        )
        outputs[gas] = out_path

    if verbose:
        print(f"[bubbles_3D_orbit] Done. Outputs: {outputs}")
    return outputs


__all__ = [
    "Block",
    "Frame",
    "read_pplume_blocks",
    "build_frames",
    "size_to_radius_mpl_like",
    "soften_radial_collar",
    "render_orbit",
    "plot_bubbles_3D",
]
