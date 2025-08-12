import numpy as np
import sympy as smp
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import Axes3D








def doublePendulum(l1,l2,m1,m2):

    th1, th2 = smp.symbols('th1 th2', cls=smp.Function)
    t = smp.symbols('t')
    th1 = th1(t)
    th2 = th2(t)

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

    eq1 = smp.diff(smp.diff(L,smp.diff(th1,t)),t) - smp.diff(L,th1)
    eq2 = smp.diff(smp.diff(L,smp.diff(th2,t)),t) - smp.diff(L,th2)






    print(L)





doublePendulum(3,4,5,5)





