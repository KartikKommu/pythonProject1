import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from sympy import smp
from matplotlib import animation
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import Axes3D









def doublePendulum1(m1, theta1in, length1, m2, theta2in, length2, time, increment):

    ##theta1 = np.linspace(0,2*np.pi,1000)
    ##theta2 = np.linspace(0,2*np.pi,1000)
    ##theta1[0] = theta1in
    ##theta2[0] = theta2in##

    tval = np.linspace(0, time, int(time/increment))
    x1val = np.linspace(0,)
    x2val = np.linspace(0,)
    y1val = np.linspace(0,)
    y2val = np.linspace(0,)



    x1 = length1 * np.sin(theta1in)
    y1 = -length1 * np.cos(theta1in)
    x2 = length1 * np.sin(theta1in) + length2 * np.sin(theta2in)
    y2 = -length1 * np.cos(theta1in) + length2 * np.cos(theta2in)

    vx1 = length1 * np.sin



    counter = 0

    while t<time:

        x1[counter] = length1 * np.sin(theta1)
        y1[counter] = length1 * np.cos(theta1)
        x2 = length1 * np.sin(theta1) + length2 * np.sin(theta2)
        y2 = length1 * np.cos(theta1) + length2 * np.cos(theta2)





        counter += increment


