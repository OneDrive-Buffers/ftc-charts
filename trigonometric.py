import numpy as np
import matplotlib.pyplot as plt

def arc(center, r, a0, a1, n=220):
    t = np.linspace(a0, a1, n)
    return center[0] + r*np.cos(t), center[1] + r*np.sin(t)

def wrap_pi(a):
    # (-pi, pi]
    return (a + np.pi) % (2*np.pi) - np.pi

def diagram_atan2_circle_shortest_path(
    theta_deg=150.0,        # robot heading θ
    theta_t_deg=-150.0,     # target angle θ_t
    out_png="diagram_4_atan2_circle_shortest_path.png",
    out_svg="diagram_4_atan2_circle_shortest_path.svg"
):
    theta   = np.deg2rad(theta_deg)
    theta_t = np.deg2rad(theta_t_deg)

    raw = theta_t - theta
    norm = np.arctan2(np.sin(raw), np.cos(raw))  # shortest path

    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")
    ax.set_title("Cercul trigonometric: atan2 + normalizare pe drumul cel mai scurt", pad=16)

    # Unit circle
    circle = plt.Circle((0, 0), 1.0, fill=False, linewidth=2.0)
    ax.add_patch(circle)

    # Axes
    ax.annotate("", xy=(1.25, 0), xytext=(-1.25, 0), arrowprops=dict(arrowstyle="->", linewidth=1.5))
    ax.annotate("", xy=(0, 1.25), xytext=(0, -1.25), arrowprops=dict(arrowstyle="->", linewidth=1.5))
    ax.text(1.28, 0, "X", va="center")
    ax.text(0, 1.28, "Y", ha="center")

    # Quadrant labels
    ax.text( 0.65,  0.65, "QI",  ha="center", va="center")
    ax.text(-0.65,  0.65, "QII", ha="center", va="center")
    ax.text(-0.65, -0.65, "QIII",ha="center", va="center")
    ax.text( 0.65, -0.65, "QIV", ha="center", va="center")

    # Reference angles: 0, pi/2, pi, 3pi/2
    for a, lab in [(0, "0"), (np.pi/2, r"$\frac{\pi}{2}$"), (np.pi, r"$\pi$"), (3*np.pi/2, r"$\frac{3\pi}{2}$")]:
        x, y = np.cos(a), np.sin(a)
        ax.plot([x], [y], marker="o", markersize=4)
        ax.text(1.08*x, 1.08*y, lab, ha="center", va="center")

    # Helper to draw a ray
    def ray(angle, label, lw=2.6):
        x, y = np.cos(angle), np.sin(angle)
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", linewidth=lw))
        ax.text(1.12*x, 1.12*y, label, ha="center", va="center")

    # Draw θ and θ_t rays
    ray(theta,   r"$\theta$ (heading)")
    ray(theta_t, r"$\theta_t$ (target)")

    # Show raw difference arc (can be "long way")
    # raw might exceed pi in magnitude; draw in its direction for illustration
    # We'll draw both raw and normalized arcs to make the point.
    # Raw arc: from θ to θ_t using the direct subtraction direction
    a0 = theta
    a1 = theta + raw
    ax_raw_x, ax_raw_y = arc((0, 0), 0.72, a0, a1)
    ax.plot(ax_raw_x, ax_raw_y, linewidth=1.8, linestyle="--")
    mid_raw = (a0 + a1) / 2
    ax.text(0.78*np.cos(mid_raw), 0.78*np.sin(mid_raw), r"$\theta_t-\theta$", ha="center", va="center")

    # Normalized arc (shortest path)
    b0 = theta
    b1 = theta + norm
    ax_norm_x, ax_norm_y = arc((0, 0), 0.92, b0, b1)
    ax.plot(ax_norm_x, ax_norm_y, linewidth=3.0)
    mid_norm = (b0 + b1) / 2
    ax.text(0.98*np.cos(mid_norm), 0.98*np.sin(mid_norm), r"$\Delta$", ha="center", va="center")

    # Explain boxes (Romanian, concise, CS-ish)
    explain_1 = (
        "Quadrante (de ce atan2):\n"
        r"$\theta_t=\mathrm{atan2}(d_y,d_x)$ alege automat QI..QIV\n"
        "și întoarce unghiul corect pe cerc."
    )
    ax.text(-1.32, 1.22, explain_1,
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.4", linewidth=1.0))

    explain_2 = (
        "Drumul cel mai scurt (normalizare):\n"
        r"$\Delta=\mathrm{atan2}\!\left(\sin(\theta_t-\theta),\cos(\theta_t-\theta)\right)$"
        "\n"
        "→ rezultatul e mereu în (−π, π], deci robotul\n"
        "se rotește minim (nu face o tură întreagă)."
    )
    ax.text(-1.32, -1.25, explain_2,
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", linewidth=1.0))

    # Numeric readout
    readout = (
        rf"$\theta={theta_deg:.0f}^\circ,\ \theta_t={theta_t_deg:.0f}^\circ$" "\n"
        rf"$\theta_t-\theta={np.rad2deg(raw):.1f}^\circ$ (poate fi lung)" "\n"
        rf"$\Delta={np.rad2deg(norm):.1f}^\circ$ (scurt, folosit la PID)"
    )
    ax.text(0.15, 0.0, readout, ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.35", linewidth=1.0))

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg)
    plt.close(fig)

if __name__ == "__main__":
    diagram_atan2_circle_shortest_path()
    print("Saved:")
    print(" - diagram_4_atan2_circle_shortest_path.png / .svg")