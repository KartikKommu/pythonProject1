import sympy as smp
import numpy as np
import matplotlib.pyplot as plt

x = smp.symbols('x')
print(x**2)

y = smp.sin(x)
y = x**2 + 4*x + 3
print(y)

z = y**2
print(z)
print(z.factor())
print(z.expand())
print(smp.solve(z,x))


##if you know the nature of some variable x, like it being real or rational, etc, specify that in the variable statement
x = smp.symbols('x')
print(smp.solve(x**2+1,x))
#vs#
x = smp.symbols('x', real = True, positive = True)
print(smp.solve(x**2+1,x))


x, y, z = smp.symbols('x y z')
F = x**2 + smp.sin(z)*y
print(F)

z_sols = smp.solve(F,z)
print(z_sols)

#turning sympy functions and expressions into numerical expressions through "lambdify"

expr = z_sols[0]
print(expr)

expr_f = smp.lambdify([x,y],expr)
print(expr_f(1,2))

x_num = np.linspace(0,1,100)
y_num =2
plt.plot(x_num, expr_f(x_num,y_num))
plt.show()

#you can also use expr.subs([]) to sub 1 or 2 out of 3 variables in an expression.






