import numpy as np
from matplotlib import pyplot as plt

from scipy.integrate import odeint


# Example one, malthusian growth model

# 1-Constants/Initial Conditions

N_0 = 10
r = 3
timepoints = np.linspace(0, 10, 1001)  # interval


# 2-Defining the function


def system_of_odes(
    y,
    t,
    r,
):
    N = y

    return r * N


# 3-Solving the system

solution = odeint(system_of_odes, y0=N_0, t=timepoints, args=(r,))

plt.plot(timepoints, solution)
plt.show()


# Higher Order Example

# 1-Constants/Initial Conditions

m = 1
b = 0.7
k = 1

init_conditions = [1.0, 0.0]
time_points = np.linspace(0, 100, 1001)

# 2-Defining System


def system_of_odes2(y, t, m, b, k):

    x, xdot = y

    xddot = -b / m * xdot - k / m * x

    return xdot, xddot


solution1 = odeint(system_of_odes2, y0=init_conditions, t=time_points, args=(m, b, k))
x_sol = solution1[:, 0]
x_dot_sol = solution1[:, 1]
plt.plot(time_points, solution1)
plt.show()
