from math import *
from matplotlib import pyplot as plt

'''
for index in range(4,8):
    print(index)
'''



def projectileMotion(angle,increment,initialVel, initialH, initialAcc,flightT, airResistance):
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
            print("Placeholder")
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
    plt.title("projectile motion")
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.show()


projectileMotion(pi/4,0.1,50,30,[0,9.81],7.98,False)








