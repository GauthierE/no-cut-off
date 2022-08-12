import numpy as np
import matplotlib.pyplot as plt


# compute the total variation (TV) distance of two probability measures p and q defined on a finite group
def TV_distance(p, q):
    return 1/2*sum(abs(p-q))


# compute the TV distances between Q^*k: the distribution of the k-th state (k = 0, ..., n-1) of the RW, and U: the uniform distribution on the cyclic group of order d
def simulation(n, d):

    # initialize: Q^*0
    distribution = np.zeros(d)
    distribution[0] = 1

    # uniform distribution
    uniform = np.full((d), 1/d)

    TV_distances = [] # TV distances between Q^*k and U (k = 0, ..., n-1)
    TV_distances.append(TV_distance(distribution, uniform))

    # compute TV distances
    for k in range(1, n):

        # update Q^*k
        distribution = np.array([1/2*distribution[(i-1)%d] + 1/2*distribution[(i+1)%d] for i in range(d)])

        # add TV(Q^*k, U)
        TV_distances.append(TV_distance(distribution, uniform))

    return TV_distances


# plot the evolution of the TV distance between Q^*k and U (k = 0, ..., n-1) on the cyclic group of order d
def TV_plot(n, d):

    plt.figure(figsize=(10, 7))
    plt.ylim(0, 1)

    plt.xlabel(r'$k$')
    plt.ylabel(r'$|Q^{*k}-U|_{TV}$')
    plt.title(r'Simple RW on $\mathrm{\mathbb{Z}}/n\mathrm{\mathbb{Z}}$')

    label = r'$\mathrm{\mathbb{Z}}$/' + str(d) + r'$\mathrm{\mathbb{Z}}$'
    plt.plot(np.arange(n), simulation(n, d), label=label)

    plt.legend()
    plt.show()