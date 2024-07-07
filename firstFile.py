from math import *
from matplotlib import pyplot as plt

"""
for index in range(4,8):
    print(index)
"""


def projectileMotion(
    angle,
    increment,
    initialVel,
    initialH,
    initialAcc,
    flightT,
    dragCoeff,
    csa,
    airDensity,
    mass,
    airResistance,
):
    time = 0
    velocity = [0, 0]
    position = [0, 0]
    velocity[0] = initialVel * cos(angle)
    velocity[1] = initialVel * sin(angle)
    position[1] = initialH
    acceleration = initialAcc
    posValX = []
    posValY = []
    while time < flightT:
        # potential issues here with assuming that dragCoeff is constant when it isnt
        # also unsure if acceleration is zeroing out like it should with drag(fixed)
        if airResistance:
            absVel = [abs(velocity[0]), abs(velocity[1])]
            acceleration[0] = (
                -((0.5) * dragCoeff * airDensity * csa * absVel[0] * velocity[0]) / mass
            )
            acceleration[1] = (
                -9.81
                - ((0.5) * dragCoeff * airDensity * csa * absVel[1] * velocity[1])
                / mass
            )

        else:
            acceleration[0] = 0
            acceleration[1] = -9.81

        time += increment

        velocity[0] = velocity[0] + increment * acceleration[0]
        velocity[1] = velocity[1] + increment * acceleration[1]
        position[0] = position[0] + velocity[0] * increment
        position[1] = position[1] + increment * velocity[1]
        print(position)
        posValX.append(position[0])
        posValY.append(position[1])
        print(velocity)
        print(time)
        print("-----")
    plt.plot(posValX, posValY)
    if airResistance:
        plt.title("projectile motion(with drag)")
    else:
        plt.title("projectile motion")
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.show()


projectileMotion(pi / 4, 0.1, 50, 30, [0, -9.81], 7.98, 0, 0, 0, 0.8, False)
# modeling values after avg tennis ball
projectileMotion(pi / 4, 0.1, 50, 30, [0, -9.81], 7.98, 0.53, 0.0154, 1.225, 0.8, True)


# testing
