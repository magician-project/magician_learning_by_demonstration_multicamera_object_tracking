#!/usr/bin/env python3
"""
Top-down camera layout planner (matplotlib) — one-side camera rig + optional mirrored side.

What it draws:
- Work cell boundary
- Conveyor lane (car moves +x)
- Keep-out zone (lane + margin), with margin distance shown in meters
- Car footprint
- ArUco markers: 18 cm on car, 6 cm on tool region
- Sync screen (15") near keep-out boundary (Ammar sync QR/pattern)
- Cameras with HFOV wedges + (optional) mirrored cameras across conveyor centerline

How to use:
1) Edit CAMERAS list (x,y, heading_deg, hfov_deg, range_m, height_m).
2) Edit dimensions (CELL_*, CONVEYOR_*, KEEP_OUT_MARGIN, CAR_*).
3) Run: python3 layout_plan.py
4) Outputs: camera_layout_one_side.png (+ optional .svg)

Dependencies:
  pip install matplotlib numpy
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge, Circle

# ----------------------------
# Scene / layout assumptions
# ----------------------------
# Work cell size (meters) - for drawing context
CELL_L = 10.0   # along conveyor direction (x)
CELL_W = 7.0    # across conveyor (y)

# Car footprint (meters)
CAR_L = 4.6
CAR_W = 1.9

# Conveyor: car moves along +x direction
CONVEYOR_CENTER_Y = CELL_W * 0.50
CONVEYOR_LANE_W = 2.6         # width of conveyor lane / moving envelope
KEEP_OUT_MARGIN = 0.6         # extra no-camera margin from lane edge (meters)

# Place car centered in conveyor lane
car_origin = np.array([
    (CELL_L - CAR_L) / 2.0,
    CONVEYOR_CENTER_Y - CAR_W / 2.0
])

# Cameras on one side of the conveyor (near side). Other side is mirror-symmetric.
NEAR_SIDE = "bottom"  # "bottom" (y < center) or "top" (y > center)

# Draw mirrored cameras on the opposite side as well?
DRAW_MIRRORED_SIDE = False

# Output files
OUT_PNG = "camera_layout_one_side.png"
OUT_SVG = "camera_layout_one_side.svg"  # optional vector output

# ----------------------------
# Declutter controls
# ----------------------------
SHOW_ALL_LABELS = False          # label every camera (can be busy)
SHOW_ONLY_PRIMARY = True         # label only primary cameras
PRIMARY_KEYWORDS = ["GoPro", "Nikon"]
FOV_ALPHA = 0.10
LABEL_FONT = 8

# ----------------------------
# Camera definitions (EDIT ME)
# ----------------------------
# Each camera: dict with name, position (x,y), heading_deg (0=+x), hfov_deg, range_m, height_m
# heading_deg convention: 0 points +x (right), 90 points +y (up)
CAMERAS = [
    dict(
        name="GoPro Hero 13 (4K60)\nOverhead wide",
        x=2.0, y=1.0, heading_deg=25, hfov_deg=90, range_m=6.0, height_m=3.0
    ),
    dict(
        name="GoPro Hero 4 (2.7K50)\nOverhead wide",
        x=6.5, y=1.1, heading_deg=160, hfov_deg=94, range_m=6.0, height_m=3.0
    ),
    dict(
        name="Nikon D7500 (1080p60)\nSide long-ish",
        x=3.5, y=0.8, heading_deg=55, hfov_deg=52, range_m=8.0, height_m=1.8
    ),
    dict(
        name="Nikon D7000 (1080p30)\nSide long-ish",
        x=5.0, y=0.8, heading_deg=120, hfov_deg=52, range_m=8.0, height_m=1.8
    ),
    dict(
        name="Small cam A\nClose-up",
        x=4.0, y=0.5, heading_deg=80, hfov_deg=80, range_m=3.0, height_m=1.3
    ),
    dict(
        name="Small cam B\nClose-up",
        x=2.8, y=0.6, heading_deg=40, hfov_deg=80, range_m=3.0, height_m=1.3
    ),
]

# ----------------------------
# Marker assumptions
# ----------------------------
# 18 cm marker on car (roof-ish reference)
ARUCO_CAR_XY = car_origin + np.array([CAR_L * 0.55, CAR_W * 0.55])

# 6 cm marker on tool, typical work area (adjust to where grinding occurs)
ARUCO_TOOL_XY = car_origin + np.array([CAR_L * 0.35, CAR_W * 0.15])

# Marker draw sizes (visual only; meters)
DRAW_RADIUS_CAR = 0.08
DRAW_RADIUS_TOOL = 0.06

# ----------------------------
# Sync screen (Ammar time QR/pattern) assumptions
# ----------------------------
# Approx dimensions for 15" 16:9 screen (meters)
SYNC_SCREEN_W_M = 0.33
SYNC_SCREEN_H_M = 0.19

# Place screen near the keep-out boundary on the camera side
lane_y0 = CONVEYOR_CENTER_Y - CONVEYOR_LANE_W / 2.0
keepout_y0 = lane_y0 - KEEP_OUT_MARGIN
keepout_y1 = lane_y0 + CONVEYOR_LANE_W + KEEP_OUT_MARGIN

if NEAR_SIDE == "bottom":
    # 25 cm outside keep-out zone, on the camera side
    SYNC_SCREEN_XY = (CELL_L * 0.55, keepout_y0 + 0.25)
    SYNC_SCREEN_HEADING_DEG = 90   # face upward (+y)
else:
    SYNC_SCREEN_XY = (CELL_L * 0.55, keepout_y1 - 0.25)
    SYNC_SCREEN_HEADING_DEG = 270  # face downward (-y)

# ----------------------------
# Helpers
# ----------------------------
def add_fov(ax, x, y, heading_deg, hfov_deg, r, alpha=FOV_ALPHA):
    """Draw an HFOV wedge from camera position."""
    theta1 = heading_deg - hfov_deg / 2.0
    theta2 = heading_deg + hfov_deg / 2.0
    ax.add_patch(Wedge((x, y), r, theta1, theta2, alpha=alpha))

def mirror_camera(cam: dict, mirror_y: float) -> dict:
    """Mirror a camera across a horizontal line y=mirror_y (top-down symmetry)."""
    mirrored = cam.copy()
    mirrored["y"] = 2 * mirror_y - cam["y"]
    # Mirroring across horizontal axis flips the y component => heading' = (-heading) mod 360
    mirrored["heading_deg"] = (-cam["heading_deg"]) % 360
    mirrored["name"] = cam["name"] + "\n(mirrored)"
    return mirrored

def draw_conveyor(ax):
    """Draw conveyor lane + keep-out zone."""
    lane_y0 = CONVEYOR_CENTER_Y - CONVEYOR_LANE_W / 2.0
    lane = Rectangle((0, lane_y0), CELL_L, CONVEYOR_LANE_W, fill=False, linewidth=2)
    ax.add_patch(lane)
    ax.text(CELL_L * 0.5, lane_y0 + CONVEYOR_LANE_W + 0.12,
            "Conveyor lane (car moves +x)", ha="center", va="bottom")

    # Keep-out zone: lane plus margin
    ko_y0 = lane_y0 - KEEP_OUT_MARGIN
    ko_h = CONVEYOR_LANE_W + 2 * KEEP_OUT_MARGIN
    keepout = Rectangle((0, ko_y0), CELL_L, ko_h, fill=False, linestyle="--", linewidth=1.8)
    ax.add_patch(keepout)

    ax.text(CELL_L * 0.5, ko_y0 - 0.12,
            f"Keep-out zone margin ({KEEP_OUT_MARGIN:.2f} m)",
            ha="center", va="top", fontsize=9)

def draw_car(ax):
    """Draw car footprint rectangle and simple orientation label."""
    ax.add_patch(Rectangle(tuple(car_origin), CAR_L, CAR_W, fill=False, linewidth=2))

    ax.text(car_origin[0] + CAR_L / 2.0,
            car_origin[1] + CAR_W + 0.10,
            f"Car footprint ~{CAR_L:.1f}m x {CAR_W:.1f}m",
            ha="center", va="bottom")

    # Indicate "front" (assume +x direction)
    ax.annotate("front →",
                xy=(car_origin[0] + CAR_L * 0.78, car_origin[1] + CAR_W + 0.02),
                xytext=(car_origin[0] + CAR_L * 0.55, car_origin[1] + CAR_W + 0.02),
                arrowprops=dict(arrowstyle="->", lw=1.5))

def draw_markers(ax):
    """Draw marker locations."""
    ax.add_patch(Circle(tuple(ARUCO_CAR_XY), DRAW_RADIUS_CAR, fill=False, linewidth=2))
    ax.text(ARUCO_CAR_XY[0] + 0.12, ARUCO_CAR_XY[1],
            "18 cm ArUco on car",
            fontsize=9, va="center")

    ax.add_patch(Circle(tuple(ARUCO_TOOL_XY), DRAW_RADIUS_TOOL, fill=False,
                        linewidth=2, linestyle="--"))
    ax.text(ARUCO_TOOL_XY[0] + 0.12, ARUCO_TOOL_XY[1],
            "6 cm ArUco on tool\n(typical work region)",
            fontsize=9, va="center")

def draw_sync_screen(ax):
    """Draw the 15\" sync QR screen near the keep-out boundary."""
    x, y = SYNC_SCREEN_XY

    # rectangle centered at (x,y)
    ax.add_patch(Rectangle(
        (x - SYNC_SCREEN_W_M / 2.0, y - SYNC_SCREEN_H_M / 2.0),
        SYNC_SCREEN_W_M, SYNC_SCREEN_H_M,
        fill=False, linewidth=2
    ))

    ax.text(x + 0.20, y,
            'Sync screen (15")\ntimestamp QR/pattern',
            fontsize=9, va="center")

    # Arrow showing screen facing direction
    dx = 0.40 * math.cos(math.radians(SYNC_SCREEN_HEADING_DEG))
    dy = 0.40 * math.sin(math.radians(SYNC_SCREEN_HEADING_DEG))
    ax.annotate("",
                xy=(x + dx, y + dy),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="->", lw=1.8))

def validate_cameras(cams: list[dict]):
    """Warn if cameras are inside keep-out zone or on wrong side (only checks non-mirrored list)."""
    lane_y0 = CONVEYOR_CENTER_Y - CONVEYOR_LANE_W / 2.0
    ko_y0 = lane_y0 - KEEP_OUT_MARGIN
    ko_y1 = lane_y0 + CONVEYOR_LANE_W + KEEP_OUT_MARGIN

    wrong_side = []
    in_keepout = []

    for cam in cams:
        y = cam["y"]
        if ko_y0 <= y <= ko_y1:
            in_keepout.append(cam["name"].split("\n")[0])

        if NEAR_SIDE == "bottom":
            if y >= CONVEYOR_CENTER_Y:
                wrong_side.append(cam["name"].split("\n")[0])
        else:
            if y <= CONVEYOR_CENTER_Y:
                wrong_side.append(cam["name"].split("\n")[0])

    return wrong_side, in_keepout

# ----------------------------
# Main
# ----------------------------
def main():
    # Build camera list (optionally mirrored)
    cams = CAMERAS.copy()
    if DRAW_MIRRORED_SIDE:
        cams += [mirror_camera(c, mirror_y=CONVEYOR_CENTER_Y) for c in CAMERAS]

    wrong_side, in_keepout = validate_cameras(CAMERAS)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Work cell boundary
    ax.add_patch(Rectangle((0, 0), CELL_L, CELL_W, fill=False, linewidth=2))
    ax.text(CELL_L / 2.0, CELL_W + 0.15,
            "Magician Dataset Recording Top-down camera plan ",
            ha="center", va="bottom", fontsize=12)

    draw_conveyor(ax)
    draw_car(ax)
    draw_markers(ax)
    draw_sync_screen(ax)

    # Cameras + FOV wedges
    for cam in cams:
        x, y = cam["x"], cam["y"]
        add_fov(ax, x, y, cam["heading_deg"], cam["hfov_deg"], cam["range_m"])
        ax.plot(x, y, marker="o", markersize=6)

        is_primary = any(k in cam["name"] for k in PRIMARY_KEYWORDS)
        if SHOW_ALL_LABELS or (SHOW_ONLY_PRIMARY and is_primary):
            ax.text(x + 0.10, y + 0.10,
                    f'{cam["name"]}\n'
                    f'h={cam["height_m"]:.1f}m, HFOV≈{cam["hfov_deg"]}°',
                    fontsize=LABEL_FONT, va="bottom")

    # Warnings (printed + drawn)
    warn_lines = []
    if wrong_side:
        warn_lines.append("Cameras not on NEAR_SIDE: " + ", ".join(wrong_side))
    if in_keepout:
        warn_lines.append("Cameras inside keep-out zone: " + ", ".join(in_keepout))
    if warn_lines:
        ax.text(0.2, CELL_W + 0.05, "\n".join(warn_lines),
                fontsize=9, va="bottom")

    # Axes styling
    ax.set_xlabel("x (m)  [conveyor direction]")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.2, CELL_L + 0.2)
    ax.set_ylim(-0.2, CELL_W + 0.8)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=220)
    fig.savefig(OUT_SVG)
    plt.show()

    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_SVG}")

if __name__ == "__main__":
    main()

