import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Helpers
# ----------------------------
def setup_ax(ax, title):
    ax.set_title(title, pad=12)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', linewidth=0.6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

def draw_axes(ax, origin=(0, 0), length=1.0, labels=("X", "Y")):
    ox, oy = origin
    ax.annotate("", xy=(ox + length, oy), xytext=(ox, oy),
                arrowprops=dict(arrowstyle="->", linewidth=1.5))
    ax.annotate("", xy=(ox, oy + length), xytext=(ox, oy),
                arrowprops=dict(arrowstyle="->", linewidth=1.5))
    ax.text(ox + length * 1.03, oy, labels[0], va='center')
    ax.text(ox, oy + length * 1.03, labels[1], ha='center')

def arc_points(center, radius, a0, a1, n=80):
    t = np.linspace(a0, a1, n)
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    return x, y

# ----------------------------
# 1) Diagram: Euclidean distance
# ----------------------------
def diagram_euclidean_distance(
    robot=(1.0, 1.0),
    target=(4.0, 3.0),
    out_png="diagram_1_euclidean_distance.png",
    out_svg="diagram_1_euclidean_distance.svg"
):
    rx, ry = robot
    tx, ty = target

    dx = tx - rx
    dy = ty - ry
    d = np.sqrt(dx*dx + dy*dy)

    fig, ax = plt.subplots(figsize=(8, 6))
    setup_ax(ax, "Euclidean distance:  d = √((x_t − x)^2 + (y_t − y)^2)")

    # Set limits with padding
    xs = [rx, tx, rx + dx]
    ys = [ry, ty, ry]
    pad = 1.0
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    # Points
    ax.plot([rx], [ry], marker='o', markersize=8)
    ax.plot([tx], [ty], marker='o', markersize=8)
    ax.text(rx, ry, "  Robot (x, y)", va='bottom')
    ax.text(tx, ty, "  Target (x_t, y_t)", va='bottom')

    # Vector from robot to target
    ax.annotate("", xy=(tx, ty), xytext=(rx, ry),
                arrowprops=dict(arrowstyle="->", linewidth=2.0))
    ax.text((rx + tx)/2, (ry + ty)/2, f"  d = {d:.2f}", va='bottom')

    # Decompose into dx and dy using right triangle
    # Horizontal leg: from (rx, ry) to (tx, ry)
    ax.plot([rx, tx], [ry, ry], linewidth=2.0)
    ax.text((rx + tx)/2, ry, f"  Δx = x_t − x = {dx:.2f}", va='bottom')

    # Vertical leg: from (tx, ry) to (tx, ty)
    ax.plot([tx, tx], [ry, ty], linewidth=2.0)
    ax.text(tx, (ry + ty)/2, f"  Δy = y_t − y = {dy:.2f}", va='center', ha='left')

    # Right angle marker at (tx, ry)
    s = 0.25
    ax.plot([tx, tx - s], [ry, ry], linewidth=2.0)
    ax.plot([tx, tx], [ry, ry + s], linewidth=2.0)
    ax.plot([tx - s, tx - s], [ry, ry + s], linewidth=2.0)
    ax.plot([tx - s, tx], [ry + s, ry + s], linewidth=2.0)

    # Show formula box
    formula = (
        r"$\Delta x = x_t - x$" "\n"
        r"$\Delta y = y_t - y$" "\n"
        r"$d = \sqrt{(\Delta x)^2 + (\Delta y)^2}$"
    )
    ax.text(0.02, 0.98, formula, transform=ax.transAxes,
            va='top', ha='left',
            bbox=dict(boxstyle="round,pad=0.4", linewidth=1.0))

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg)
    plt.close(fig)

# ----------------------------
# 2) Diagram: 2D rotation (field → robot frame)
# ----------------------------
def diagram_rotation_field_to_robot(
    theta_deg=35,
    v_field=(4.0, 2.0),
    out_png="diagram_2_rotation_field_to_robot.png",
    out_svg="diagram_2_rotation_field_to_robot.svg"
):
    theta = np.deg2rad(theta_deg)
    xf, yf = v_field

    # Rotation matrix (field -> robot) as used in many robotics conventions:
    # [x_r] = [ cosθ  sinθ] [x_f]
    # [y_r]   [-sinθ  cosθ] [y_f]
    xr =  xf*np.cos(theta) + yf*np.sin(theta)
    yr = -xf*np.sin(theta) + yf*np.cos(theta)

    fig, ax = plt.subplots(figsize=(9, 6))
    setup_ax(ax, "2D rotation: field frame → robot frame")

    # Axes lengths
    L = 5.5
    ax.set_xlim(-1.2, L)
    ax.set_ylim(-1.2, L)

    # Field axes at origin
    draw_axes(ax, origin=(0, 0), length=5.0, labels=("X_field", "Y_field"))
    ax.text(0.02, 0.90, "Field frame", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", linewidth=1.0))

    # Robot axes (rotated by +theta relative to field axes)
    # X_robot unit vector in field coords
    xru = (np.cos(theta), np.sin(theta))
    yru = (-np.sin(theta), np.cos(theta))

    ax.annotate("", xy=(xru[0]*4.2, xru[1]*4.2), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", linewidth=1.8))
    ax.annotate("", xy=(yru[0]*4.2, yru[1]*4.2), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", linewidth=1.8))
    ax.text(xru[0]*4.35, xru[1]*4.35, "X_robot", va='center')
    ax.text(yru[0]*4.35, yru[1]*4.35, "Y_robot", va='center')

    # Angle arc between X_field and X_robot
    arc_r = 1.1
    ax_x, ax_y = arc_points((0, 0), arc_r, 0, theta)
    ax.plot(ax_x, ax_y, linewidth=1.6)
    ax.text(arc_r*0.9*np.cos(theta/2), arc_r*0.9*np.sin(theta/2),
            r"$\theta$", va='center', ha='center')

    # Vector in field frame
    ax.annotate("", xy=(xf, yf), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", linewidth=2.2))
    ax.text(xf, yf, "  v_field = (x_f, y_f)", va='bottom')

    # Projections of v onto robot axes (in field coordinates)
    # Projection onto X_robot: xr * X_robot_unit
    proj_x_end = (xr * xru[0], xr * xru[1])
    ax.plot([0, proj_x_end[0]], [0, proj_x_end[1]], linewidth=2.0)
    ax.text(proj_x_end[0], proj_x_end[1], "  x_r (along X_robot)", va='bottom')

    # From proj_x_end to v (perpendicular component along Y_robot)
    ax.plot([proj_x_end[0], xf], [proj_x_end[1], yf], linewidth=2.0)
    ax.text((proj_x_end[0]+xf)/2, (proj_x_end[1]+yf)/2,
            "  y_r component", va='bottom')

    # Formula box (matrix + numeric example)
    box = (
        r"$\begin{pmatrix}x_r\\y_r\end{pmatrix}="
        r"\begin{pmatrix}\cos\theta&\sin\theta\\-\sin\theta&\cos\theta\end{pmatrix}"
        r"\begin{pmatrix}x_f\\y_f\end{pmatrix}$" "\n"
        rf"$\theta={theta_deg}^\circ,\ (x_f,y_f)=({xf:.1f},{yf:.1f}) \Rightarrow (x_r,y_r)=({xr:.2f},{yr:.2f})$"
    )
    ax.text(0.02, 0.02, box, transform=ax.transAxes,
            va='bottom', ha='left',
            bbox=dict(boxstyle="round,pad=0.4", linewidth=1.0))

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg)
    plt.close(fig)

# ----------------------------
# Generate both diagrams
# ----------------------------
if __name__ == "__main__":
    diagram_euclidean_distance()
    diagram_rotation_field_to_robot()
    print("Saved:")
    print(" - diagram_1_euclidean_distance.png / .svg")
    print(" - diagram_2_rotation_field_to_robot.png / .svg")