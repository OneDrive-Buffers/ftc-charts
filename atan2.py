import numpy as np
import matplotlib.pyplot as plt

def arc(center, r, a0, a1, n=140):
    t = np.linspace(a0, a1, n)
    return center[0] + r*np.cos(t), center[1] + r*np.sin(t)

def diagram_aim_trig(
    robot=(1.0, 1.0),
    target=(4.0, 3.0),
    robotAngle_deg=20.0,
    out_png="diagram_1_aim_trig.png",
    out_svg="diagram_1_aim_trig.svg"
):
    rx, ry = robot
    tx, ty = target
    theta = np.deg2rad(robotAngle_deg)

    dx = tx - rx
    dy = ty - ry
    d  = np.hypot(dx, dy)

    theta_t = np.arctan2(dy, dx)
    delta_raw = theta_t - theta
    delta = np.arctan2(np.sin(delta_raw), np.cos(delta_raw))  # normalized

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Auto-aim trig: distance + atan2 target angle + normalized heading error", pad=12)

    # limits
    pad = 1.2
    ax.set_xlim(min(rx, tx) - pad, max(rx, tx) + pad)
    ax.set_ylim(min(ry, ty) - pad, max(ry, ty) + pad)

    # points
    ax.plot(rx, ry, marker="o", markersize=9)
    ax.plot(tx, ty, marker="o", markersize=9)
    # ax.text(rx, ry, "  Robot (r_x, r_y)", va="bottom")
    # ax.text(tx, ty, "  Target (t_x, t_y)", va="bottom")

    # right triangle legs
    ax.plot([rx, tx], [ry, ry], linewidth=2.0)  # dx leg
    ax.plot([tx, tx], [ry, ty], linewidth=2.0)  # dy leg

    # vector to target (distance)
    ax.annotate("", xy=(tx, ty), xytext=(rx, ry),
                arrowprops=dict(arrowstyle="->", linewidth=2.6))
    ax.text((rx+tx)/2, (ry+ty)/2, f"  d = {d:.2f}", va="bottom")

    ax.text((rx+tx)/2, ry, f"  d_x = t_x - r_x = {dx:.2f}", va="bottom")
    ax.text(tx, (ry+ty)/2, f"  d_y = t_y - r_y = {dy:.2f}", va="center", ha="left")

    # right angle marker at (tx, ry)
    s = 0.22
    ax.plot([tx, tx - s], [ry, ry], linewidth=2.0)
    ax.plot([tx, tx], [ry, ry + s], linewidth=2.0)
    ax.plot([tx - s, tx - s], [ry, ry + s], linewidth=2.0)
    ax.plot([tx - s, tx], [ry + s, ry + s], linewidth=2.0)

    # robot heading ray (θ)
    L = 1.3
    hx, hy = rx + L*np.cos(theta), ry + L*np.sin(theta)
    ax.annotate("", xy=(hx, hy), xytext=(rx, ry),
                arrowprops=dict(arrowstyle="->", linewidth=2.0))
    ax.text(hx, hy, "  heading θ", va="bottom")

    # target direction ray (θ_t)
    L2 = 1.6
    txray, tyray = rx + L2*np.cos(theta_t), ry + L2*np.sin(theta_t)
    ax.annotate("", xy=(txray, tyray), xytext=(rx, ry),
                arrowprops=dict(arrowstyle="->", linewidth=2.0))
    ax.text(txray, tyray, "  θ_t", va="bottom")

    # arc for θ_t (from +X axis)
    arc_r = 0.8
    axx, axy = arc((rx, ry), arc_r, 0, theta_t)
    ax.plot(axx, axy, linewidth=1.8)
    ax.text(rx + arc_r*0.95*np.cos(theta_t/2),
            ry + arc_r*0.95*np.sin(theta_t/2),
            r"$\theta_t$", ha="center", va="center")

    # arc for θ (from +X axis)
    arc_r2 = 0.55
    axx2, axy2 = arc((rx, ry), arc_r2, 0, theta)
    ax.plot(axx2, axy2, linewidth=1.5)
    ax.text(rx + arc_r2*0.95*np.cos(theta/2),
            ry + arc_r2*0.95*np.sin(theta/2),
            r"$\theta$", ha="center", va="center")

    # arc for Δ between heading and target direction (normalized)
    a0 = theta
    a1 = theta + delta
    arc_r3 = 1.05
    axd, ayd = arc((rx, ry), arc_r3, a0, a1)
    ax.plot(axd, ayd, linewidth=2.2)
    ax.text(rx + arc_r3*0.95*np.cos((a0+a1)/2),
            ry + arc_r3*0.95*np.sin((a0+a1)/2),
            r"$\Delta$", ha="center", va="center")

    # trig ratios
    cosv = dx/d if d != 0 else 0.0
    sinv = dy/d if d != 0 else 0.0
    ax.text(0.02, 0.96,
            (r"$\cos\theta_t=\frac{d_x}{d},\quad \sin\theta_t=\frac{d_y}{d}$"
             "\n"
             rf"$\cos\theta_t\approx {cosv:.3f},\ \sin\theta_t\approx {sinv:.3f}$"),
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", linewidth=1.0))

    # IMPORTANT FIX:
    # matplotlib's mathtext can choke on \operatorname{atan2}.
    # Use \mathrm{atan2} instead.
    box = (
        r"$d_x=t_x-r_x,\ \ d_y=t_y-r_y$" "\n"
        r"$d=\sqrt{d_x^2+d_y^2}$" "\n"
        r"$\theta_t=\mathrm{atan2}(d_y,d_x)$" "\n"
        r"$\Delta=\mathrm{atan2}\!\left(\sin(\theta_t-\theta),\ \cos(\theta_t-\theta)\right)$"
        "\n"
        rf"$\theta_t={np.rad2deg(theta_t):.1f}^\circ,\ \theta={robotAngle_deg:.1f}^\circ,\ \Delta={np.rad2deg(delta):.1f}^\circ$"
    )
    ax.text(0.02, 0.02, box, transform=ax.transAxes,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", linewidth=1.0))

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg)
    plt.close(fig)

if __name__ == "__main__":
    diagram_aim_trig()
    print("Saved:")
    print(" - diagram_1_aim_trig.png / .svg")