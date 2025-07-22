import numpy as np
from matplotlib import pyplot as plt

##1D heat transfer


def heatTransferOne(length, nodes, time, initialT, boundaryT, alpha):

    dx = length / nodes
    dt = (0.5 * (dx**2)) / alpha
    # t_nodes = int(time / dt)#
    u = np.zeros(nodes) + initialT
    u[0] = boundaryT
    u[-1] = boundaryT

    counter = 0

    # visualizing
    fig, axis = plt.subplots()
    pcm = axis.pcolormesh(
        [u],
        cmap=plt.cm.jet,
        vmin=min(initialT, boundaryT, 0),
        vmax=max(initialT, boundaryT),
    )
    plt.colorbar(pcm, ax=axis)
    axis.set_ylim(-2, 3)

    while counter < time:

        w = u.copy()

        for i in range(1, nodes - 1):

            u[i] = ((alpha * dt * (w[i - 1] - (2 * w[i]) + w[i + 1])) / (dx**2)) + w[i]

        counter += dt
        pcm.set_array([u])
        axis.set_title("1-D Heat transfer, time: {:.3f}[s]".format(counter))
        plt.pause(0.01)

    plt.show()


def heatTransferTwo(length, nodes, time, initialT, boundaryT, alpha):

    dx = length / nodes
    dy = length / nodes
    dt = min(dx**2 / (4 * alpha), dy**2 / (4 * alpha))
    t_nodes = int(time / dt)
    u = np.zeros((nodes, nodes)) + initialT
    u[0, :] = boundaryT
    u[-1, :] = boundaryT

    # visualizing
    fig, axis = plt.subplots()
    pcm = axis.pcolormesh(
        u, cmap=plt.cm.jet, vmin=min(initialT, boundaryT), vmax=max(initialT, boundaryT)
    )
    plt.colorbar(pcm, ax=axis)

    counter = 0

    while counter < time:

        w = u.copy()

        for i in range(1, nodes - 1):

            for j in range(1, nodes - 1):

                dd_ux = (w[i + 1, j] - 2 * w[i, j] + w[i - 1, j]) / (dx**2)
                dd_uy = (w[i, j + 1] - 2 * w[i, j] + w[i, j - 1]) / (dy**2)

                u[i, j] = alpha * dt * (dd_ux + dd_uy) + w[i, j]

        counter += dt
        pcm.set_array(u)
        axis.set_title("2-D Heat transfer, time: {:.3f}[s]".format(counter))
        plt.pause(0.01)

    plt.show()


heatTransferOne(50, 20, 4, 20, 100, 110)
heatTransferTwo(50, 60, 4, 20, 100, 110)
