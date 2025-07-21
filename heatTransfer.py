import numpy as np
from matplotlib import pyplot as plt

##1D heat transfer


def heatTransferOne(length, nodes, time, initialT, boundaryT, alpha):

    dx = length / nodes
    dt = (0.5 * (dx**2)) / alpha
    t_nodes = int(time / dt)
    u = np.zeros(nodes) + initialT
    u[0] = boundaryT
    u[-1] = boundaryT

    counter = 0

    # visualizing
    fig, axis = plt.subplots()
    pcm = axis.pcolormesh([u], cmap=plt.cm.jet, vmin=0, vmax=100)
    plt.colorbar(pcm, ax=axis)
    axis.set_ylim(-2, 3)

    while counter < time:

        w = u.copy()

        for i in range(1, nodes - 1):

            u[i] = ((alpha * dt * (w[i - 1] - (2 * w[i]) + w[i + 1])) / (dx**2)) + w[i]

            counter += dt
            pcm.set_array([u])
            axis.set_title("Heat transfer, time: {:.3f}[s]".format(counter))
            plt.pause(0.001)

    plt.show()


heatTransferOne(50, 20, 4, 20, 100, 110)
