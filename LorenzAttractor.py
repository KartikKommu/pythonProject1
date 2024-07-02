import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# testing
from scipy.integrate import odeint

sigma = 10
beta = 8/3
rho = 28


def system_of_odes(vector,t,sigma,beta,rho):
    x, y, z = vector

    d_vector = [
        sigma*(y-x),
        x*(rho-z)-y,
        x*y-beta*z
    ]
    return d_vector


position_0_1 = [2.5,2.5,2.5]
position_0_2 = [1.0,1.0,1.0]
time_points = np.linspace(0,80,2001)


positions_1 = odeint(system_of_odes,position_0_1,time_points,args=(sigma,beta,rho))
x_sol_1, y_sol_1, z_sol_1 = positions_1[:,0], positions_1[:,1], positions_1[:,2]

positions_2 = odeint(system_of_odes,position_0_2,time_points,args=(sigma,beta,rho))
x_sol_2, y_sol_2, z_sol_2 = positions_2[:,0], positions_2[:,1], positions_2[:,2]

fig, ax = plt.subplots(subplot_kw={'projection':'3d'})

lorenz_plt_1, = ax.plot(x_sol_1, y_sol_1, z_sol_1, 'red', label=f'Position 0 of 1st: {position_0_1}')
lorenz_plt_2, = ax.plot(x_sol_2, y_sol_2, z_sol_2, 'blue', label=f'Position 0 of 2nd: {position_0_2}')

#-----animation segment-----#
def update(frame):

    lower_lim = max(0, frame-100)

    x_current_1 = x_sol_1[lower_lim:frame+1]
    y_current_1 = y_sol_1[lower_lim:frame+1]
    z_current_1 = z_sol_1[lower_lim:frame+1]

    x_current_2 = x_sol_2[lower_lim:frame + 1]
    y_current_2 = y_sol_2[lower_lim:frame + 1]
    z_current_2 = z_sol_2[lower_lim:frame + 1]

    '''
    x_current = x_sol[0:frame + 1]
    y_current = y_sol[0:frame + 1]
    z_current = z_sol[0:frame + 1]
    '''


    lorenz_plt_1.set_data(x_current_1,y_current_1)
    lorenz_plt_1.set_3d_properties(z_current_1)

    lorenz_plt_2.set_data(x_current_2, y_current_2)
    lorenz_plt_2.set_3d_properties(z_current_2)

    return lorenz_plt_1,lorenz_plt_2


animation = FuncAnimation(fig, update, frames=len(time_points), interval=25, blit=False)
plt.show()