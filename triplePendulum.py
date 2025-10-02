"""
testing ability of chatgpt to write complex physics code, not currently my own code. going through
to verify if AI can reproduce a physically coherent and valid system.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Parameters (play with these) ---
g = 9.81
m1 = 1.0
m2 = 1.0
m3 = 1.0
L1 = 1.0
L2 = 1.0
L3 = 1.0

# Initial conditions: angles (rad) and angular velocities (rad/s)
# example: small offsets
theta1_0 = 1.2
theta2_0 = -0.5
theta3_0 = 0.5
dtheta1_0 = 0.0
dtheta2_0 = 0.0
dtheta3_0 = 0.0

t_max = 30.0
dt = 0.01

# --- Build EOMs using Lagrangian formalism with symbolic simplification ---
# We derive mass matrix and RHS via symbolic algebra once, then lambdify.

try:
    import sympy as sp

    # Symbols for generalized coords and their derivatives
    th1, th2, th3 = sp.symbols('th1 th2 th3')            # angles
    w1, w2, w3 = sp.symbols('w1 w2 w3')                 # angular velocities
    a1, a2, a3 = sp.symbols('a1 a2 a3')                 # angular accelerations

    # parameters
    G, M1, M2, M3, L_1, L_2, L_3 = sp.symbols('G M1 M2 M3 L_1 L_2 L_3')

    # positions
    x1 = L_1 * sp.sin(th1)
    y1 = -L_1 * sp.cos(th1)

    x2 = x1 + L_2 * sp.sin(th2)
    y2 = y1 - L_2 * sp.cos(th2)

    x3 = x2 + L_3 * sp.sin(th3)
    y3 = y2 - L_3 * sp.cos(th3)

    # velocities (using chain rule: dx/dt = dx/dth * dth)
    dx1 = sp.diff(x1, th1) * w1
    dy1 = sp.diff(y1, th1) * w1

    dx2 = sp.diff(x2, th1) * w1 + sp.diff(x2, th2) * w2
    dy2 = sp.diff(y2, th1) * w1 + sp.diff(y2, th2) * w2

    dx3 = (sp.diff(x3, th1) * w1 +
           sp.diff(x3, th2) * w2 +
           sp.diff(x3, th3) * w3)
    dy3 = (sp.diff(y3, th1) * w1 +
           sp.diff(y3, th2) * w2 +
           sp.diff(y3, th3) * w3)

    # kinetic energy
    T = (sp.Rational(1,2) * M1 * (dx1**2 + dy1**2) +
         sp.Rational(1,2) * M2 * (dx2**2 + dy2**2) +
         sp.Rational(1,2) * M3 * (dx3**2 + dy3**2))

    # potential energy (choose zero at pivot); using (1 - cos) form avoids constant offsets
    V = (M1 * G * L_1 * (1 - sp.cos(th1)) +
         M2 * G * (L_1 * (1 - sp.cos(th1)) + L_2 * (1 - sp.cos(th2))) +
         M3 * G * (L_1 * (1 - sp.cos(th1)) + L_2 * (1 - sp.cos(th2)) + L_3 * (1 - sp.cos(th3))))

    Lagr = T - V

    # generalized coordinates and velocities
    qs = (th1, th2, th3)
    ws = (w1, w2, w3)
    accs = (a1, a2, a3)

    # Build Euler-Lagrange equations symbolically.
    # For each coordinate q_i:
    # d/dt(∂L/∂w_i) - ∂L/∂q_i = 0
    # Because we treat w_i as independent symbols, d/dt(∂L/∂w_i) =
    # sum_j ( ∂(∂L/∂w_i)/∂q_j * w_j ) + sum_j ( ∂(∂L/∂w_i)/∂w_j * a_j )
    EL_eqs = []
    for i in range(3):
        dLdwi = sp.diff(Lagr, ws[i])
        # time derivative expansion:
        d_dt_dLdwi = sum(sp.diff(dLdwi, qs[j]) * ws[j] for j in range(3)) + \
                     sum(sp.diff(dLdwi, ws[j]) * accs[j] for j in range(3))
        dLdq = sp.diff(Lagr, qs[i])
        EL = sp.simplify(d_dt_dLdwi - dLdq)
        EL_eqs.append(sp.simplify(EL))

    # Now EL_eqs are linear in the accelerations a1,a2,a3.
    # Put them in matrix form M * a = RHS (where RHS collects remaining terms with velocities & gravity)
    M_mat, RHS = sp.linear_eq_to_matrix(EL_eqs, accs)
    M_mat = sp.simplify(M_mat)
    RHS = sp.simplify(-RHS)  # move to RHS as usual: M a = RHS

    # Convert to numeric function with lambdify.
    func_acc = sp.lambdify((th1, th2, th3, w1, w2, w3,
                            G, M1, M2, M3, L_1, L_2, L_3),
                           (M_mat, RHS), 'numpy')

    def accelerations(thetas, ws_vals, params):
        """Compute angular accelerations given angles and angular velocities.
        returns (a1,a2,a3).
        """
        th1v, th2v, th3v = thetas
        w1v, w2v, w3v = ws_vals
        Mnum, RHSnum = func_acc(th1v, th2v, th3v, w1v, w2v, w3v,
                                params['g'], params['m1'], params['m2'], params['m3'],
                                params['L1'], params['L2'], params['L3'])
        # Solve M a = RHS
        a = np.linalg.solve(np.array(Mnum, dtype=float), np.array(RHSnum, dtype=float)).flatten()
        return a

    # numeric wrapper for integrator
    params = {'g': g, 'm1': m1, 'm2': m2, 'm3': m3, 'L1': L1, 'L2': L2, 'L3': L3}

    def deriv(t, y):
        """state y = [th1, th2, th3, w1, w2, w3]"""
        ths = y[0:3]
        ws_now = y[3:6]
        a1v, a2v, a3v = accelerations(ths, ws_now, params)
        return np.array([ws_now[0], ws_now[1], ws_now[2], a1v, a2v, a3v], dtype=float)

except Exception as e:
    raise RuntimeError("Symbolic derivation with sympy failed. Make sure sympy is installed and working.") from e

# --- Integrate numerically ---
y0 = np.array([theta1_0, theta2_0, theta3_0, dtheta1_0, dtheta2_0, dtheta3_0], dtype=float)
t_eval = np.arange(0, t_max + dt, dt)

print("Integrating... (this may take a few seconds)")

sol = solve_ivp(deriv, [0, t_max], y0, t_eval=t_eval, rtol=1e-9, atol=1e-9, method='DOP853')
if not sol.success:
    print("Warning: integrator reported failure:", sol.message)

ths = sol.y[0:3]    # shape (3, N)
ws_hist = sol.y[3:6]

# --- compute positions for animation & energies ---
x1s = L1 * np.sin(ths[0, :])
y1s = -L1 * np.cos(ths[0, :])

x2s = x1s + L2 * np.sin(ths[1, :])
y2s = y1s - L2 * np.cos(ths[1, :])

x3s = x2s + L3 * np.sin(ths[2, :])
y3s = y2s - L3 * np.cos(ths[2, :])

# energies
def energy_from_state(th1v, th2v, th3v, w1v, w2v, w3v):
    # velocities
    dx1 = L1 * np.cos(th1v) * w1v
    dy1 = L1 * np.sin(th1v) * w1v

    dx2 = L1 * np.cos(th1v) * w1v + L2 * np.cos(th2v) * w2v
    dy2 = L1 * np.sin(th1v) * w1v + L2 * np.sin(th2v) * w2v

    dx3 = (L1 * np.cos(th1v) * w1v +
           L2 * np.cos(th2v) * w2v +
           L3 * np.cos(th3v) * w3v)
    dy3 = (L1 * np.sin(th1v) * w1v +
           L2 * np.sin(th2v) * w2v +
           L3 * np.sin(th3v) * w3v)

    Tkin = 0.5 * m1 * (dx1**2 + dy1**2) + 0.5 * m2 * (dx2**2 + dy2**2) + 0.5 * m3 * (dx3**2 + dy3**2)
    # potential: zero at pivot
    Vpot = (m1 * g * (L1 * (1 - np.cos(th1v))) +
            m2 * g * (L1 * (1 - np.cos(th1v)) + L2 * (1 - np.cos(th2v))) +
            m3 * g * (L1 * (1 - np.cos(th1v)) + L2 * (1 - np.cos(th2v)) + L3 * (1 - np.cos(th3v))))
    return Tkin + Vpot, Tkin, Vpot

E = np.zeros_like(t_eval)
T_kin = np.zeros_like(t_eval)
V_pot = np.zeros_like(t_eval)
for i in range(len(t_eval)):
    Et, Tt, Vt = energy_from_state(ths[0, i], ths[1, i], ths[2, i],
                                   ws_hist[0, i], ws_hist[1, i], ws_hist[2, i])
    E[i] = Et
    T_kin[i] = Tt
    V_pot[i] = Vt

# --- Animation ---
fig = plt.figure(figsize=(10, 6))
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.4)

ax_anim = fig.add_subplot(grid[:, :2])
ax_anim.set_aspect('equal')
ax_anim.set_xlim(- (L1 + L2 + L3) - 0.2, (L1 + L2 + L3) + 0.2)
ax_anim.set_ylim(- (L1 + L2 + L3) - 0.2, (L1 + L2 + L3) + 0.2)
ax_anim.set_title("Triple Pendulum Animation")

line, = ax_anim.plot([], [], '-o', lw=2)
trace1, = ax_anim.plot([], [], lw=1, alpha=0.6)  # path of mass 3

ax_a = fig.add_subplot(grid[0, 2])
ax_w = fig.add_subplot(grid[1, 2])

ax_a.set_title("Angles (rad)")
ax_a.plot(t_eval, ths[0, :], label='θ1')
ax_a.plot(t_eval, ths[1, :], label='θ2')
ax_a.plot(t_eval, ths[2, :], label='θ3')
ax_a.legend(loc='upper right')

ax_w.set_title("Energy")
ax_w.plot(t_eval, E, label='Total energy')
ax_w.plot(t_eval, T_kin, label='Kinetic', linestyle='--')
ax_w.plot(t_eval, V_pot, label='Potential', linestyle=':')
ax_w.legend(loc='upper right')

path_x = []
path_y = []

def init():
    line.set_data([], [])
    trace1.set_data([], [])
    return line, trace1

def update(i):
    xs = [0, x1s[i], x2s[i], x3s[i]]
    ys = [0, y1s[i], y2s[i], y3s[i]]
    line.set_data(xs, ys)
    # update trace of last mass
    path_x.append(x3s[i])
    path_y.append(y3s[i])
    if len(path_x) > 2000:
        path_x.pop(0)
        path_y.pop(0)
    trace1.set_data(path_x, path_y)
    return line, trace1

ani = animation.FuncAnimation(fig, update, frames=len(t_eval), init_func=init,
                              interval=dt*1000, blit=True)

# Show plots and animation
plt.show()

# Optionally save the animation:
# ani.save('triple_pendulum.mp4', fps=30, dpi=200)