import numpy as np
from matplotlib import pyplot as plt

from scipy.integrate import odeint

##types of arrays##

arr = np.array([3, 4, 5, 1])
a2 = np.zeros(12)
a3 = np.random.random(12)
a4 = np.random.randn(12)  ##normal/guassian dist with mean of 0, SD of 1
a5 = np.linspace(0, 5, 200)  ##give the number of divisions, does the interval for you
a6 = np.arange(0, 5, 0.01)  ##give the interval, does number of divisions


# array operations are performed on each element#

x = np.arange(0, 10, 0.01)
y = x**2 + 3
z = 1 / (x + 1)


a7 = np.array([1, 2, 3, 5, 7, 9])

# indexing

a7[2:]
a7[:-2]
a7[1:4]
print(a7 > 4)
print(a7[a7 > 4])

# returns desired element of given string, first element in this case
f = lambda s: s[0]

names = np.array(["kart", "blud", "kkringle"])

# vectorize applies a given function to an array
letterK = np.vectorize(lambda s: s[0])(names) == "k"

print(names[letterK])


# calc/stats

x = np.linspace(1, 10, 100)
y = 1 / x**2 * np.sin(x)

dydx = np.gradient(y, x)
y_int = np.cumsum(y) * (x[1] - x[0])


# examples

x = np.linspace(0, 10, 10000)
y = np.exp(-(x / 10)) * np.sin(x)

dydx = np.gradient(y, x)

plt.plot(x, y)
plt.plot(x, dydx)
print(np.mean(y[(x >= 4) * (7 >= x)]))
print(x[1:][0 > ((dydx[1:]) * (dydx[:-1]))])

plt.show()
