import numpy as np
from matplotlib import pyplot as plt

from scipy.integrate import odeint

##types of arrays##

arr = np.array([3, 4, 5, 1])
a2 = np.zeros(12)
a3 = np.random.random(12)
a4 = np.random.randn(12)  ##normal/guassian dist with mean of 0, SD of 1##
a5 = np.linspace(0, 5, 200)  ##give the number of divisions, does the interval for you
a6 = np.arange(0, 5, 0.01)  ##give the interval, does number of divisions


print(a6)

# array operations are performed on each element
