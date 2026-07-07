import numpy as np
import matplotlib.pyplot as plt

def diagram_pid_response(
    out_png="diagram_3_pid.png",
    out_svg="diagram_3_pid.svg",
    dt=0.02,
    T=6.0,
    # Plant: first-order system y' = (u - y)/tau
    tau=0.7,
    # PID gains (tune here)
    Kp=1.2,
    Ki=0.6,
    Kd=0.10,
    # Setpoint
    r=1.0,
    # Optional output clamp (motor power style)
    u_min=-1.0,
    u_max=1.0
):
    n = int(T / dt) + 1
    t = np.linspace(0, T, n)

    y = np.zeros(n)           # plant output
    u = np.zeros(n)           # controller output
    e = np.zeros(n)           # error
    I = np.zeros(n)           # integral term
    D = np.zeros(n)           # derivative term
    P = np.zeros(n)           # proportional term

    integral = 0.0
    last_error = 0.0

    for i in range(1, n):
        # --- error ---
        e[i] = r - y[i-1]

        # --- PID terms ---
        P[i] = Kp * e[i]

        integral += e[i] * dt
        I[i] = Ki * integral

        derivative = (e[i] - last_error) / dt
        D[i] = Kd * derivative

        # --- output ---
        u_raw = P[i] + I[i] + D[i]
        u[i] = np.clip(u_raw, u_min, u_max)

        # --- plant update (first-order) ---
        # y' = (u - y)/tau
        ydot = (u[i] - y[i-1]) / tau
        y[i] = y[i-1] + ydot * dt

        last_error = e[i]

    # ---- Plot 1: response (r vs y) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("PID response (annotated):  u = Kp·e + Ki∫e dt + Kd·de/dt", pad=12)
    ax.grid(True, linestyle="--", linewidth=0.6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("value")

    ax.plot(t, np.full_like(t, r), linewidth=2.2, label="setpoint r(t)")
    ax.plot(t, y, linewidth=2.2, label="output y(t)")

    # Mark typical metrics (overshoot, settle-ish)
    peak_i = int(np.argmax(y))
    ax.plot(t[peak_i], y[peak_i], marker="o")
    ax.text(t[peak_i], y[peak_i], f"  peak = {y[peak_i]:.2f}", va="bottom")

    # show final value
    ax.plot(t[-1], y[-1], marker="o")
    ax.text(t[-1], y[-1], f"  final = {y[-1]:.2f}", va="bottom", ha="right")

    # Formula + term meanings box
    box = (
        r"$e(t)=r(t)-y(t)$" "\n"
        r"$u(t)=K_p e(t)+K_i\int_0^t e(\tau)d\tau+K_d\frac{de(t)}{dt}$" "\n"
        rf"$K_p={Kp},\ K_i={Ki},\ K_d={Kd},\ \Delta t={dt}$" "\n"
        rf"clamp: $u\in[{u_min},{u_max}]$, plant: $y'=(u-y)/\tau,\ \tau={tau}$"
    )
    ax.text(0.02, 0.02, box, transform=ax.transAxes,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", linewidth=1.0))

    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_svg)
    plt.close(fig)

    # ---- Plot 2: term breakdown (P, I, D, u) ----
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.set_title("PID terms (annotated): P, I, D contributions", pad=12)
    ax2.grid(True, linestyle="--", linewidth=0.6)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("value")

    ax2.plot(t, P, linewidth=2.0, label="P = Kp·e")
    ax2.plot(t, I, linewidth=2.0, label="I = Ki·∫e dt")
    ax2.plot(t, D, linewidth=2.0, label="D = Kd·de/dt")
    ax2.plot(t, u, linewidth=2.4, label="u = P + I + D (clamped)")

    # Annotate early transient (D spike) and steady-state (I holds)
    i_spike = min(10, n-1)
    ax2.plot(t[i_spike], D[i_spike], marker="o")
    ax2.text(t[i_spike], D[i_spike], "  D spike (rapid change)", va="bottom")

    i_mid = int(0.65 * n)
    ax2.plot(t[i_mid], I[i_mid], marker="o")
    ax2.text(t[i_mid], I[i_mid], "  I accumulates (removes bias)", va="bottom")

    box2 = (
        "Interpretare:\n"
        "P = reacție imediată (pe eroare)\n"
        "I = corectează eroarea persistentă (acumulează)\n"
        "D = frânează oscilația (reacționează la schimbare)"
    )
    ax2.text(0.02, 0.98, box2, transform=ax2.transAxes,
             va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.4", linewidth=1.0))

    ax2.legend(loc="upper right")
    fig2.tight_layout()
    fig2.savefig(out_png.replace(".png", "_terms.png"), dpi=220)
    fig2.savefig(out_svg.replace(".svg", "_terms.svg"))
    plt.close(fig2)

if __name__ == "__main__":
    diagram_pid_response()
    print("Saved:")
    print(" - diagram_3_pid.png / .svg")
    print(" - diagram_3_pid_terms.png / .svg")