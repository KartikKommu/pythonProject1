from math import *
from matplotlib import pyplot as plt
import numpy as np


# uses MKS Units
def orbitEulerCromer(radius, acceleration, satMass, ancMass, increment, length):
    t = 0
    velocity = [0,0]
    position = [,]
    positionVal=[]
    acceleration = (-G*satMass*ancMass*radius)/abs(radius*radius*radius)*satMass

    while t < length:

        velocity = sqrt(G*ancMass*((2/radius) - (1/acceleration)))



