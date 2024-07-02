from math import *
from matplotlib import pyplot as plt

'''
for index in range(4,8):
    print(index)
'''




def projectileMotion(angle,increment,initialVel, initialH, initialAcc,flightT, dragCoeff, csa, airDensity, airResistance):
    time = 0
    velocity = [0,0]
    position = [0,0]
    velocity[0] = initialVel*cos(angle)
    velocity[1] = initialVel*sin(angle)
    position[1] = initialH
    acceleration = initialAcc
    posValX = []
    posValY = []
    while time<flightT:
        if airResistance:
            absVel = [abs(velocity[0]),abs(velocity[1])]
            acceleration[0] = acceleration[0] -  (1/2*dragCoeff*airDensity*csa*absVel[0]*velocity[0])
            acceleration[1] = acceleration[1] - (1 / 2 * dragCoeff * airDensity * csa * absVel[1] * velocity[1])
            velocity[0] = velocity[0] - increment * acceleration[0]

        else:
            acceleration[0] = 0
            acceleration[1] = 9.81

        time += increment

        velocity[1] = velocity[1] - increment*acceleration[1]
        position[0] = position[0] + velocity[0]*increment
        position[1] = position[1] + increment*velocity[1]
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


projectileMotion(pi/4,0.1,50,30,[0,9.81],7.98, 0,0,0,False)
# modeling values after avg tennis ball
projectileMotion(pi/4,0.1,50,30,[0,9.81],7.98, 0.53,0.0154,0.1225,True)


# testing





