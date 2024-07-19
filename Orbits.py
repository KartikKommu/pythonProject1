from math import *
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint

# uses MKS Units

G = 6.67 * (10 ^ -11)


def orbitEulerCromer(
    initVel, initTheta, radius, acceleration, satMass, ancMass, increment, length
):
    t = 0.0
    velocity = initVel

    theta = initTheta
    thetaVal = []
    acceleration = (
        (-G * satMass * ancMass * radius) / abs(radius * radius * radius) * satMass
    )

    while t < length:

        t += increment

        velocity = velocity + increment * acceleration * (theta)
        theta = theta + increment * velocity

        thetaVal.append(theta)

    ax = plt.subplot(111, projection="polar")
    ax.plot(thetaVal)

    plt.show()


orbitEulerCromer(
    12700, 0, 8.28784 * (10 ^ 11), 0.0, 5 * (10 ^ 14), 1.9891 * (10 ^ 30), 0.02, 1000
)
