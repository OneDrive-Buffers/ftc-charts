import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Helpers
# -------------------------
def R(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def save(fig, name):
    fig.savefig(name + ".png", dpi=220, bbox_inches="tight")
    fig.savefig(name + ".svg", bbox_inches="tight")
    plt.close(fig)

def draw_arc(ax, center, radius, a0, a1, label=None):
    t = np.linspace(a0, a1, 120)
    ax.plot(center[0] + radius*np.cos(t),
            center[1] + radius*np.sin(t),
            linewidth=2.0)
    if label:
        mid = 0.5*(a0+a1)
        ax.text(center[0] + (radius+0.05)*np.cos(mid),
                center[1] + (radius+0.05)*np.sin(mid),
                label, va="bottom")

# -------------------------
# 1) Velocity decomposition (field -> robot axes)
# -------------------------
def diagram_1_velocity_projection(
    x=0.0, y=0.0,
    h_deg=35.0,
    vx=1.1, vy=0.3,
    out="D1_velocity_projection"
):
    h = np.deg2rad(h_deg)

    p = np.array([x, y])
    v_field = np.array([vx, vy])

    # Robot axes in field coords
    fwd = np.array([np.cos(h), np.sin(h)])        # +F
    strf = np.array([-np.sin(h), np.cos(h)])      # +S (90° left from forward)

    # Projections (dot products) = components in robot frame
    vF = float(np.dot(v_field, fwd))
    vS = float(np.dot(v_field, strf))

    vF_vec = vF * fwd
    vS_vec = vS * strf

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.6)
    ax.set_title("1) Descompunerea vitezei pe axele robotului (Forward / Strafe)", pad=12)
    ax.set_xlabel("X (field)")
    ax.set_ylabel("Y (field)")

    # Plot robot position
    ax.plot(p[0], p[1], marker="o", markersize=10)
    ax.text(p[0], p[1], "  Robot", va="bottom")

    # Draw robot axes
    L = 2.2
    ax.annotate("", xy=p + L*fwd,  xytext=p, arrowprops=dict(arrowstyle="->", linewidth=3))
    ax.annotate("", xy=p + L*strf, xytext=p, arrowprops=dict(arrowstyle="->", linewidth=3))
    ax.text(*(p + L*fwd),  "  +F", va="bottom")
    ax.text(*(p + L*strf), "  +S", va="bottom")

    # Draw heading arc (theta from +X axis)
    draw_arc(ax, p, radius=0.9, a0=0.0, a1=h, label="θ")

    # Draw field velocity
    ax.annotate("", xy=p + v_field, xytext=p, arrowprops=dict(arrowstyle="->", linewidth=4))
    ax.text(*(p + 0.55*v_field), "  v(field)", va="bottom")

    # Draw projection vectors (vF and vS)
    ax.annotate("", xy=p + vF_vec, xytext=p, arrowprops=dict(arrowstyle="->", linewidth=3, linestyle="--"))
    ax.annotate("", xy=p + vS_vec, xytext=p, arrowprops=dict(arrowstyle="->", linewidth=3, linestyle="--"))
    ax.text(*(p + 0.55*vF_vec), "  vF", va="bottom")
    ax.text(*(p + 0.55*vS_vec), "  vS", va="bottom")

    # Draw a right-angle “projection box” to show v = vF + vS
    corner = p + vF_vec
    ax.plot([corner[0], (corner+vS_vec)[0]],
            [corner[1], (corner+vS_vec)[1]],
            linewidth=2)
    ax.plot([p[0] + v_field[0], (corner+vS_vec)[0]],
            [p[1] + v_field[1], (corner+vS_vec)[1]],
            linewidth=2)

    # Text box (math in plain, no aligned)
    box = (
        "Ideea: proiectăm v(field) pe axele robotului.\n"
        r"$v_F = v_x\cos(-\theta) - v_y\sin(-\theta)$" "\n"
        r"$v_S = v_x\sin(-\theta) + v_y\cos(-\theta)$" "\n"
        f"Exemplu: θ={h_deg:.0f}°, v=({vx:.2f},{vy:.2f}) → vF={vF:.2f}, vS={vS:.2f}"
    )
    ax.text(0.02, 0.02, box, transform=ax.transAxes,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", linewidth=1.0))

    # Limits
    pts = np.vstack([p, p+v_field, p+L*fwd, p+L*strf, p+vF_vec, p+vS_vec, corner+vS_vec])
    ax.set_xlim(pts[:,0].min()-1.0, pts[:,0].max()+1.4)
    ax.set_ylim(pts[:,1].min()-1.0, pts[:,1].max()+1.4)

    save(fig, out)

# -------------------------
# 2) Braking / glide distance on each axis (signed v^2/(2a))
# -------------------------
def diagram_2_braking_distance(
    aF=0.8, aS=2.0,
    out="D2_braking_distance"
):
    v = np.linspace(-2.8, 2.8, 900)
    gF = np.sign(v) * (v*v) / (2*aF)
    gS = np.sign(v) * (v*v) / (2*aS)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.grid(True, linestyle="--", linewidth=0.6)
    ax.set_title("2) Din vF / vS estimăm distanța de coastă (frânare anticipată)", pad=12)
    ax.set_xlabel("v (viteza pe axă)")
    ax.set_ylabel("g (distanță prezisă)")

    ax.plot(v, gF, linewidth=3, label="Forward axis (gF)")
    ax.plot(v, gS, linewidth=3, label="Strafe axis (gS)")

    # Mark 3 reference points
    for vv in [-2.0, 1.5, 2.5]:
        ax.plot([vv], [np.sign(vv)*(vv*vv)/(2*aF)], marker="o", markersize=7)
        ax.plot([vv], [np.sign(vv)*(vv*vv)/(2*aS)], marker="o", markersize=7)

    box = (
        "Formulă (pe fiecare axă):\n"
        r"$g = \mathrm{sign}(v)\dfrac{v^2}{2a}$" "\n"
        "De ce e utilă: dacă robotul se mișcă mai tare (|v| mare), va aluneca mai mult.\n"
        f"Parametri exemplu: aF={aF}, aS={aS} (strafe frânează diferit față de forward)"
    )
    ax.text(0.02, 0.02, box, transform=ax.transAxes,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", linewidth=1.0))

    ax.legend(loc="upper left")
    save(fig, out)

# -------------------------
# 3) Current pose + glide vector -> predicted pose
# -------------------------
def diagram_3_predicted_pose(
    x=0.0, y=0.0,
    h_deg=35.0,
    vx=1.1, vy=0.3,
    aF=0.8, aS=2.0,
    out="D3_predicted_pose"
):
    h = np.deg2rad(h_deg)
    p = np.array([x, y])
    v_field = np.array([vx, vy])

    # Decompose velocity into robot axes
    fwd  = np.array([np.cos(h), np.sin(h)])
    strf = np.array([-np.sin(h), np.cos(h)])
    vF = float(np.dot(v_field, fwd))
    vS = float(np.dot(v_field, strf))

    # Glide distance along robot axes
    gF = np.sign(vF) * (vF*vF) / (2*aF)
    gS = np.sign(vS) * (vS*vS) / (2*aS)

    # Convert glide back to field frame
    glide_field = R(h) @ np.array([gF, gS])
    p_pred = p + glide_field

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.6)
    ax.set_title("3) Poziția prezisă:  (x_pred, y_pred) = (x, y) + (x_g, y_g)", pad=12)
    ax.set_xlabel("X (field)")
    ax.set_ylabel("Y (field)")

    # Draw current pose
    ax.plot(p[0], p[1], marker="o", markersize=10)
    ax.text(p[0], p[1], "  poziție curentă", va="bottom")

    # Draw predicted pose
    ax.plot(p_pred[0], p_pred[1], marker="o", markersize=10)
    ax.text(p_pred[0], p_pred[1], "  poziție prezisă (stop)", va="bottom")

    # Draw velocity arrow
    ax.annotate("", xy=p + v_field, xytext=p, arrowprops=dict(arrowstyle="->", linewidth=3))
    ax.text(*(p + 0.55*v_field), "  v(field)", va="bottom")

    # Draw glide arrow
    ax.annotate("", xy=p_pred, xytext=p, arrowprops=dict(arrowstyle="->", linewidth=4, linestyle="--"))
    ax.text(*((p + p_pred)/2), "  glide (x_g, y_g)", va="bottom")

    # Draw heading arrow
    L = 1.8
    ax.annotate("", xy=p + L*fwd, xytext=p, arrowprops=dict(arrowstyle="->", linewidth=2.5))
    ax.text(*(p + L*fwd), "  heading θ", va="bottom")

    box = (
        "Pașii vizuali:\n"
        "1) calculezi vF, vS (pe axele robotului)\n"
        "2) calculezi gF, gS cu v^2/(2a)\n"
        "3) rotești (gF,gS) înapoi în field → (x_g,y_g)\n"
        r"$x_{pred}=x+x_g,\ \ y_{pred}=y+y_g$" "\n"
        f"Exemplu: vF={vF:.2f}, vS={vS:.2f} → gF={gF:.2f}, gS={gS:.2f}"
    )
    ax.text(0.02, 0.02, box, transform=ax.transAxes,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", linewidth=1.0))

    pts = np.vstack([p, p+v_field, p_pred, p+L*fwd])
    ax.set_xlim(pts[:,0].min()-1.2, pts[:,0].max()+1.6)
    ax.set_ylim(pts[:,1].min()-1.2, pts[:,1].max()+1.6)

    save(fig, out)

if __name__ == "__main__":
    diagram_1_velocity_projection()
    diagram_2_braking_distance()
    diagram_3_predicted_pose()
    print("Saved:")
    print(" - D1_velocity_projection.(png/svg)")
    print(" - D2_braking_distance.(png/svg)")
    print(" - D3_predicted_pose.(png/svg)")