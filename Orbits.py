from math import *
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint

# uses MKS Units

G = 6.67 * (10**-11)


def orbitEulerCromer(initVel, initRadius, satMass, ancMass, increment, length):

    t = 0.0
    velocity = [0.0, initVel]
    acceleration = [0.0, 0.0]

    r = [initRadius, 0]
    ValX = []
    ValY = []

    while t < length:

        acceleration[0] = -(G * ancMass) * r[0] / abs(r[0]) ** 3

        velocity[0] = velocity[0] + increment * acceleration[0]
        velocity[1] = velocity[1] + increment * acceleration[1]
        r[0] = r[0] + increment * velocity[0]
        r[1] = r[1] + increment * velocity[1]
        t += increment

        ValX.append(r[0])
        ValY.append(r[1])



    plt.show()


orbitEulerCromer(29806.07, 1.496 * (10**11), 5 * (10**14), 1.9891 * (10**30), 0.02, 40)
