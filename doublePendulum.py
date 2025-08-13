import numpy as np
import sympy as smp
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from sympy import false


def doublePendulum(l1,l2,m1,m2,t_total,interval,i_th1,i_z1,i_th2,i_z2):

    th1, th2 = smp.symbols('th1 th2', cls=smp.Function)
    t = smp.symbols('t')
    th1 = th1(t)
    th2 = th2(t)

    th1dot = smp.diff(th1, t)
    th2dot = smp.diff(th2, t)
    th1dd = smp.diff(th1dot, t)
    th2dd = smp.diff(th2dot, t)

    x1 = l1*smp.sin(th1)
    y1 = -l1*smp.cos(th1)
    x2 = l1*smp.sin(th1) + l2*smp.sin(th2)
    y2 = -l1*smp.cos(th1) + l2*smp.cos(th2)

    x1dot = smp.diff(x1,t)
    y1dot = smp.diff(y1, t)
    x2dot = smp.diff(x2,t)
    y2dot = smp.diff(y2,t)

    T = smp.Rational(1,2)*m1*(y1dot**2+x1dot**2) + smp.Rational(1,2)*m2*(y2dot**2+x2dot**2)
    U = m1*9.81*y1 + m2*9.81*y2

    L = T - U

    eq1 = smp.diff(smp.diff(L,th1dot),t) - smp.diff(L,th1)
    eq2 = smp.diff(smp.diff(L,th2dot),t) - smp.diff(L,th2)

    sols = smp.solve([eq1,eq2], (th1dd,th2dd), rational=false, simplify=False)

    ##transition/conversion of symbolic expressions to numerical in order to solve the ODE

    dz1dt = smp.lambdify((t,th1,th1dot,th2,th2dot), sols[th1dd])
    dz2dt = smp.lambdify((t,th1,th1dot,th2,th2dot), sols[th2dd])
    dth1dt = smp.lambdify(th1dot, th1dot)
    dth2dt = smp.lambdify(th2dot, th2dot)

    ##then we solve the system of DE's above

    def dSdt(S,t):
        th1, z1, th2, z2 = S

        return [
            dth1dt(z1),
            dz1dt(t, th1,th2,z1,z2),
            dth2dt(z2),
            dz2dt(t,th1,th2,z1,z2),
        ]

    t = np.arange(0, t_total, interval)
    ans = odeint(dSdt, y0=[i_th1,i_z1,i_th2,i_z2], t = t)

    plt.plot(t,ans.T[2])
    plt.show()









doublePendulum(3,5,2,3,40,0.01,1, -3, -1,5)





