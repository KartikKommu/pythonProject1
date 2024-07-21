from math import *
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint

# uses MKS Units

G = 6.67 * (10**-11)


def orbitEulerCromer(initVel, initRadius, satMass, ancMass, increment, length):

    t = 0.0
    velocity = [0, initVel]

    r = [initRadius, 0]
    thetaVal = []

    while t < length:

        t += increment

        acceleration = (-G * ancMass) * r / abs(r) ** 3

        velocity = velocity + increment * acceleration
        r = r + increment * velocity

        thetaVal.append(r)

    ax = plt.subplot(111, projection="polar")
    ax.plot(thetaVal)

    plt.show()


orbitEulerCromer(29806.07, 1.496 * (10**11), 5 * (10**14), 1.9891 * (10**30), 0.02, 40)
