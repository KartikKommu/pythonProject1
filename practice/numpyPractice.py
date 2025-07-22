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

# multiD-arrays

a1 = np.array([[1, 4, 5], [4, 5, 6], [7, 1, 9]])

print(a1 > 3)
a1[0]
print(a1[:, 1])
print(a1[1:, 2])
print(a1[1:, :2])

x = np.linspace(0, 10, 1000)
y = np.linspace(0, 10, 1000)
xv, yv = np.meshgrid(
    x, y
)  ##create a 2d array(xv and yv) for x and y for multi-variable fxn

zv = xv**2 + yv**2

# plt.contourf(xv, yv, zv, levels=50)
# plt.colorbar()

# plt.show()

# basic lin-Alg
A = np.array([[3, 2, 1], [5, -5, 4], [6, 0, 1]])
b1 = np.array([1, 2, 3])
b2 = np.array([-1, 2, -5])
c = np.array([4, 0, 3])

print(np.linalg.solve(A, c))


print(A @ b1)
A.T

np.dot(b1, b2)
np.cross(b1, b2)

D = np.array([[4, 2, 2], [2, 4, 2], [2, 2, 4]])
print(np.linalg.eig(D))  ##eigenvectors are the columns of the 2d array, not rows

# examples

x = np.linspace(0, 10, 10000)
y = np.exp(-(x / 10)) * np.sin(x)

dydx = np.gradient(y, x)

# plt.plot(x, y)
# plt.plot(x, dydx)
print(np.mean(y[(x >= 4) * (7 >= x)]))
print(x[1:][0 > ((dydx[1:]) * (dydx[:-1]))])


# plt.show()

x = np.arange(0, 10001, 1)

sum = np.sum(x[((x % 4) != 0) * ((x % 7) != 0)])
print(sum)


theta = np.linspace(0, 2 * np.pi, 1000)
radius = 1 + ((3 / 4) * np.sin(3 * theta))
drdt = np.gradient(radius, theta)

x = radius * np.cos(theta)
y = radius * np.sin(theta)

area = np.sum(radius) * (theta[1] - theta[0])
arclength = np.sum(np.sqrt((radius**2) + (drdt**2))) * (theta[1] - theta[0])

print(arclength)

##plt.plot(theta, area)
# plt.plot(theta, arclength)
# plt.plot(x, y)
# plt.show()


x = np.linspace(-2, 2, 1000)
y = np.linspace(-2, 2, 1000)

xv, yv = np.meshgrid(x, y)
zv = np.exp(-((xv**2) + (yv**2))) * np.sin(xv)

print(np.sum(np.abs(zv)) * (x[1] - x[0]) * (y[1] - y[0]))
print(
    np.sum(np.abs(zv[np.sqrt((xv**2 + yv**2)) > 0.5])) * (x[1] - x[0]) * (y[1] - y[0])
)

# plt.contourf(xv, yv, zv, levels=40)
# plt.colorbar()
# plt.show()
