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

plt.plot(solution)
plt.show()
