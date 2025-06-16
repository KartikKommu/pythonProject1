from math import *
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint

# uses AU Units

G = 4 * pi**2


# using with orbits that have any appreciable eccentricity leads to failure of method
def orbitEulerCromer(initVel, initRadius, ancMass, increment, length):

    t = 0.0
    velocity = [0.0, initVel]
    acceleration = [0.0, 0.0]

    r = [initRadius, 0]
    ValX = []
    ValY = []
    rVal = []
    thetaVal = []

    while t < length:

        magR = sqrt((r[0] ** 2) + (r[1] ** 2))

        # fails if y starts at 0, must be a solution(FIXED)
        acceleration[0] = -(G * ancMass) * r[0] / magR**3
        acceleration[1] = -(G * ancMass) * r[1] / magR**3

        velocity[0] = velocity[0] + increment * acceleration[0]
        velocity[1] = velocity[1] + increment * acceleration[1]
        r[0] = r[0] + increment * velocity[0]
        r[1] = r[1] + increment * velocity[1]
        t += increment

        ValX.append(r[0])
        ValY.append(r[1])
        rVal.append(magR)
        thetaVal.append(atan2(r[1], r[0]))

    # currently unable to properly plot Orbital data.(FIXED)
    ax = plt.subplot(111, projection="polar")

    ax.plot(thetaVal, rVal)
    plt.show()


orbitEulerCromer(2 * pi, 1, 1, 0.02, 4)


# where f(x,t) is the right hand side of a DE dx/dt = f(x,t)


def rk4Method(function, increment, tMax, initT):
    xData = []
    tData = []
    # only works if x is not dependent on t AND x. unsure how to fix
    x = function(0, initT)
    t = initT
    while t < tMax:
        F1 = function(x, t)
        F2 = function(x + ((increment / 2) * F1), t + (increment / 2))
        F3 = function(x + ((increment / 2) * F2), t + (increment / 2))
        F4 = function(x + increment * F3, t + increment)
        t = t + increment

        x = x + ((1 / 6) * increment * (F1 + 2 * F2 + 2 * F3 + F4))
        xData.append(x)
        tData.append(t)
    return xData, tData


def fun(x, t):

    fun = 3 * t**2
    return fun


x, t = rk4Method(fun, 0.1, 2, -2)
plt.plot(t, x)
plt.show()


def orbitRK4(initVel, initRadius, satMass, ancMass, increment, length):

    t = 0.0
    t = 0.0
    velocity = [0.0, initVel]
    acceleration = [0.0, 0.0]

    def accel0(v, r):
        accel0 = -(G * ancMass) * r[0] / magR**3
        return accel0

    def accel1(v, r):
        accel1 = -(G * ancMass) * r[0] / magR**3
        return accel1

    r = [initRadius, 0]
    ValX = []
    ValY = []
    rVal
    thetaVal

    while t < length:
        magR = sqrt((r[0] ** 2) + (r[1] ** 2))

        velocity
