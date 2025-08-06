import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt


def()



def doublePendulumTweak(m1, theta1in, length1, m2, theta2in, length2, time):

    theta1 = np.linspace(0,2*np.pi,1000)
    theta2 = np.linspace(0,2*np.pi,1000)
    theta1[0] = theta1in
    theta2[0] = theta2in







    counter = 0

    while t<time:

        x1[counter] = length1 * np.sin(theta1)
        y1[counter] = length1 * np.cos(theta1)
        x2 = length1 * np.sin(theta1) + length2 * np.sin(theta2)
        y2 = length1 * np.cos(theta1) + length2 * np.cos(theta2)


