import math

# testing
from math import *
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def pendulumGraphEuler(angle, initialV, stringLen, time, increment):
    t = 0.0
    timeVal = []
    theta = angle
    velocityVal = []
    thetaVal = []
    velocity = initialV
    accelerationAng = 0
    while t < time:
        t += increment
        timeVal.append(t)
        accelerationAng = -(9.81 / stringLen) * math.sin(theta)
        theta = theta + increment * velocity
        thetaVal.append(theta)
        velocity = velocity + increment * accelerationAng
        velocityVal.append(velocity)

    plt.scatter(timeVal, thetaVal)
    plt.show()

    return thetaVal[0]


# regardless of lowering increment, use of Euler method seems very innacurate, growing in energy, which is physically
# impossible
# passing around 15 seconds, the model completely implodes. Could be optimized, but likely a foundational issue with
# Euler method in this instance
pendulumGraphEuler(pi / 12, 0, 4, 30.0, 0.1)


def pendulumGraphVerlet(angle, stringLen, time, increment):
    t = 0.0
    timeVal = [0.0]
    theta = angle
    thetaVal = []
    accelerationAng = 0
    counter = 0
    # Due to neccesary back stepping and lack of starting position in verlet formula, Another method must be used to obtain
    # first value. In this case, Euler-Cromer
    thetaVal.append(pendulumGraphEuler(angle, 0, stringLen, increment * 2, increment))

    while t < time:
        t += increment
        timeVal.append(t)
        counter += 1
        accelerationAng = -(9.81 / stringLen) * math.sin(theta)
        theta = (
            2 * theta
            - thetaVal[counter - 2]
            + (increment * increment) * accelerationAng
        )
        thetaVal.append(theta)

    plt.plot(timeVal, thetaVal)
    plt.show()


# Verlet much more accurate, even at double the increment size
pendulumGraphVerlet(pi / 12, 6, 30.0, 0.1)
