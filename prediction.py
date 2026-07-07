import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Helpers
# -------------------------
def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

def save(fig, png, svg):
    # Avoid tight_layout mathtext edge cases; use bbox_inches instead
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

# -------------------------
# 1) Diagram: field velocity -> robot velocity (vF, vS)
# -------------------------
def diagram_glide_frames(
    x=2.0, y=1.2,
    h_deg=35.0,
    vx=1.1, vy=0.3,
    out_png="diagram_glide_1_frames.png",
    out_svg="diagram_glide_1_frames.svg"
):
    h = np.deg2rad(h_deg)

    v_field = np.array([vx, vy])
    v_robot = rot(-h) @ v_field
    vF, vS = v_robot

    fwd   = np.array([np.cos(h), np.sin(h)])
    strf  = np.array([-np.sin(h), np.cos(h)])

    vF_field = vF * fwd
    vS_field = vS * strf

    fig, ax = plt.subplots(figsize=(10, 6.6))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.6)
    ax.set_title("Predicția inerției — rotație viteză (field → robot frame)", pad=12)
    ax.set_xlabel("X (field)")
    ax.set_ylabel("Y (field)")

    pad = 1.4
    xs = [x, x+vx, x+vF_field[0], x+vS_field[0]]
    ys = [y, y+vy, y+vF_field[1], y+vS_field[1]]
    ax.set_xlim(min(xs)-pad, max(xs)+2.2)
    ax.set_ylim(min(ys)-pad, max(ys)+2.2)

    # Robot
    ax.plot(x, y, marker="o", markersize=9)
    ax.text(x, y, "  Robot (x, y)", va="bottom")

    # v(field)
    ax.annotate("", xy=(x+vx, y+vy), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", linewidth=2.8))
    ax.text(x+vx/2, y+vy/2, "  v(field)", va="bottom")

    # Robot axes
    L = 1.7
    ax.annotate("", xy=(x+L*fwd[0], y+L*fwd[1]), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", linewidth=2.0))
    ax.annotate("", xy=(x+L*strf[0], y+L*strf[1]), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", linewidth=2.0))
    ax.text(x+L*fwd[0],  y+L*fwd[1],  "  +F (forward)", va="bottom")
    ax.text(x+L*strf[0], y+L*strf[1], "  +S (strafe)",  va="bottom")

    # vF / vS projections (dashed)
    ax.annotate("", xy=(x+vF_field[0], y+vF_field[1]), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", linewidth=2.0, linestyle="--"))
    ax.annotate("", xy=(x+vS_field[0], y+vS_field[1]), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", linewidth=2.0, linestyle="--"))
    ax.text(x+vF_field[0]/2, y+vF_field[1]/2, "  vF", va="bottom")
    ax.text(x+vS_field[0]/2, y+vS_field[1]/2, "  vS", va="bottom")

    # Mathtext-friendly (NO \begin{aligned})
    box = (
        r"$v_F = v_x\cos(-\theta) - v_y\sin(-\theta)$" "\n"
        r"$v_S = v_x\sin(-\theta) + v_y\cos(-\theta)$" "\n"
        rf"$\theta={h_deg:.0f}^\circ,\ (v_x,v_y)=({vx:.2f},{vy:.2f})$" "\n"
        rf"$v_F={vF:.2f},\ v_S={vS:.2f}$"
    )
    ax.text(0.02, 0.02, box, transform=ax.transAxes,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", linewidth=1.0))

    save(fig, out_png, out_svg)

# -------------------------
# 2) Diagram: braking distance g = sign(v)*v^2/(2a)
# -------------------------
def diagram_glide_braking_curve(
    aF=0.8, aS=2.0,
    out_png="diagram_glide_2_braking.png",
    out_svg="diagram_glide_2_braking.svg"
):
    v = np.linspace(-2.5, 2.5, 700)
    gF = np.sign(v) * (v*v) / (2*aF)
    gS = np.sign(v) * (v*v) / (2*aS)

    fig, ax = plt.subplots(figsize=(10, 6.0))
    ax.grid(True, linestyle="--", linewidth=0.6)
    ax.set_title("Predicția inerției — distanța de frânare:  g = sign(v)·v²/(2a)", pad=12)
    ax.set_xlabel("v (viteza pe axă)")
    ax.set_ylabel("g (distanță de coastă / frânare)")

    ax.plot(v, gF, linewidth=2.6, label="gF (forward axis)")
    ax.plot(v, gS, linewidth=2.6, label="gS (strafe axis)")

    box = (
        r"$g_F=\mathrm{sign}(v_F)\dfrac{v_F^2}{2a_F}$" "\n"
        r"$g_S=\mathrm{sign}(v_S)\dfrac{v_S^2}{2a_S}$" "\n"
        rf"$a_F={aF},\ a_S={aS}$"
    )
    ax.text(0.02, 0.02, box, transform=ax.transAxes,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", linewidth=1.0))

    ax.legend(loc="upper left")
    save(fig, out_png, out_svg)

# -------------------------
# 3) Diagram: current pose -> predicted pose
# -------------------------
def diagram_glide_predicted_pose(
    x=2.0, y=1.2, h_deg=35.0,
    vx=1.1, vy=0.3,
    aF=0.8, aS=2.0,
    out_png="diagram_glide_3_predicted_pose.png",
    out_svg="diagram_glide_3_predicted_pose.svg"
):
    h = np.deg2rad(h_deg)

    v_field = np.array([vx, vy])
    v_robot = rot(-h) @ v_field
    vF, vS = v_robot

    gF = np.sign(vF) * (vF*vF) / (2*aF)
    gS = np.sign(vS) * (vS*vS) / (2*aS)

    glide_field = rot(h) @ np.array([gF, gS])
    x_g, y_g = glide_field

    x_pred = x + x_g
    y_pred = y + y_g

    fig, ax = plt.subplots(figsize=(10, 6.6))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.6)
    ax.set_title("Frânare anticipată — poziția prezisă (x_pred, y_pred)", pad=12)
    ax.set_xlabel("X (field)")
    ax.set_ylabel("Y (field)")

    pad = 1.6
    ax.set_xlim(min(x, x_pred)-pad, max(x, x_pred)+pad)
    ax.set_ylim(min(y, y_pred)-pad, max(y, y_pred)+pad)

    ax.plot(x, y, marker="o", markersize=10)
    ax.text(x, y, "  (x, y)", va="bottom")

    ax.plot(x_pred, y_pred, marker="o", markersize=10)
    ax.text(x_pred, y_pred, "  (x_pred, y_pred)", va="bottom")

    ax.annotate("", xy=(x_pred, y_pred), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", linewidth=3.0))
    ax.text((x+x_pred)/2, (y+y_pred)/2, "  glide (x_g, y_g)", va="bottom")

    # Heading arrow
    fwd = np.array([np.cos(h), np.sin(h)])
    L = 1.4
    ax.annotate("", xy=(x+L*fwd[0], y+L*fwd[1]), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", linewidth=2.0))
    ax.text(x+L*fwd[0], y+L*fwd[1], "  heading θ", va="bottom")

    box = (
        r"$v_F = v_x\cos(-\theta)-v_y\sin(-\theta)$" "\n"
        r"$v_S = v_x\sin(-\theta)+v_y\cos(-\theta)$" "\n"
        r"$g_F = \mathrm{sign}(v_F)\dfrac{v_F^2}{2a_F},\ \ g_S=\mathrm{sign}(v_S)\dfrac{v_S^2}{2a_S}$" "\n"
        r"$\begin{bmatrix}x_g\\y_g\end{bmatrix}=R(\theta)\begin{bmatrix}g_F\\g_S\end{bmatrix}$" "\n"
        r"$x_{\mathrm{pred}}=x+x_g,\ \ y_{\mathrm{pred}}=y+y_g$"
    )
    ax.text(0.02, 0.02, box, transform=ax.transAxes,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", linewidth=1.0))

    save(fig, out_png, out_svg)

if __name__ == "__main__":
    diagram_glide_frames()
    diagram_glide_braking_curve()
    diagram_glide_predicted_pose()
    print("Saved:")
    print(" - diagram_glide_1_frames.(png/svg)")
    print(" - diagram_glide_2_braking.(png/svg)")
    print(" - diagram_glide_3_predicted_pose.(png/svg)")