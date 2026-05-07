#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--axis_len", type=float, default=1.00)
    ap.add_argument("--trail", type=int, default=60)
    ap.add_argument("--save_mp4", type=str, default="")
    ap.add_argument("--save_gif", type=str, default="")
    ap.add_argument("--headless", action="store_true", help="Render without GUI")
    ap.add_argument("--width", type=int, default=1280, help="Output width in pixels")
    ap.add_argument("--height", type=int, default=720, help="Output height in pixels")
    ap.add_argument("--dpi", type=int, default=100)
    args = ap.parse_args()

    if args.headless:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    def rot_from_rpy_deg(roll, pitch, yaw):
        r, p, y = np.deg2rad([roll, pitch, yaw])
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)

        Rx = np.array([[1, 0, 0],
                       [0, cr, -sr],
                       [0, sr, cr]])
        Ry = np.array([[cp, 0, sp],
                       [0, 1, 0],
                       [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0],
                       [sy,  cy, 0],
                       [0,    0, 1]])
        return Rz @ Ry @ Rx

    def finite_pose(row, prefix):
        keys = [
            f"{prefix}_tvec_x_m",
            f"{prefix}_tvec_y_m",
            f"{prefix}_tvec_z_m",
            f"{prefix}_roll_deg",
            f"{prefix}_pitch_deg",
            f"{prefix}_yaw_deg",
        ]
        vals = np.array([row.get(k, np.nan) for k in keys], float)
        if not np.isfinite(vals).all():
            return None
        t = vals[:3]
        R = rot_from_rpy_deg(vals[3], vals[4], vals[5])
        return t, R

    df = pd.read_csv(args.csv_path)
    if args.every > 1:
        df = df.iloc[::args.every].reset_index(drop=True)

    # Compute bounds
    pts = []
    for p in ("aruco_0", "aruco_1"):
        cols = [f"{p}_tvec_x_m", f"{p}_tvec_y_m", f"{p}_tvec_z_m"]
        if all(c in df.columns for c in cols):
            A = df[cols].to_numpy(float)
            A = A[np.isfinite(A).all(axis=1)]
            if len(A):
                pts.append(A)

    if pts:
        P = np.vstack(pts)
        c = P.mean(axis=0)
        r = max(np.ptp(P, axis=0).max(), args.axis_len * 4) * 0.6
        lims = np.array([c - r, c + r])
    else:
        lims = np.array([[-1, -1, -1], [1, 1, 1]])

    # Figure size in inches
    fig_w = args.width / args.dpi
    fig_h = args.height / args.dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")

    def setup_axes():
        ax.set_xlim(lims[0, 0], lims[1, 0])
        ax.set_ylim(lims[0, 1], lims[1, 1])
        ax.set_zlim(lims[0, 2], lims[1, 2])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")

    def draw_triad(t, R, L, name):
        ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]
        ax.scatter(*t, s=20)
        ax.text(*t, f" {name}", fontsize=9)
        ax.quiver(*t, *ex, length=L, normalize=True)
        ax.quiver(*t, *ey, length=L, normalize=True)
        ax.quiver(*t, *ez, length=L, normalize=True)

    def draw_trail(prefix, i):
        if args.trail <= 0:
            return
        cols = [f"{prefix}_tvec_x_m", f"{prefix}_tvec_y_m", f"{prefix}_tvec_z_m"]
        j0 = max(0, i - args.trail)
        P = df.loc[j0:i, cols].to_numpy(float)
        P = P[np.isfinite(P).all(axis=1)]
        if len(P):
            ax.plot(P[:, 0], P[:, 1], P[:, 2], linewidth=1)

    def update(i):
        ax.cla()
        setup_axes()
        ax.set_title(f"Frame {i}/{len(df)-1}")
        print(f"\r  Frame {i}/{len(df)-1}                ",end="",flush=True)

        for p in ("aruco_0", "aruco_1"):
            draw_trail(p, i)
            pose = finite_pose(df.iloc[i], p)
            if pose:
                draw_triad(*pose, args.axis_len, p)

        return []

    anim = FuncAnimation(
        fig,
        update,
        frames=len(df),
        interval=1000 / max(args.fps, 1e-6),
    )

    if args.save_mp4:
        anim.save(args.save_mp4, fps=args.fps, dpi=args.dpi)
        print("Saved MP4:", args.save_mp4)

    if args.save_gif:
        anim.save(args.save_gif, fps=args.fps, dpi=args.dpi)
        print("Saved GIF:", args.save_gif)

    if not args.headless:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()

