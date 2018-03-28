import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sklearn.manifold as mani

zLen = 3
iterRange = range(zLen)


def main():
    X = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
         [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]]

    # plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
    # plt.show()


    x = []
    y = []
    z = []
    Z = []

    for i in iterRange:

        tsne = mani.TSNE(perplexity=i*2+1)
        #Z.append(tsne.fit_transform(X))
        Z.extend(tsne.fit_transform(X))
        #plt.scatter(Z[:,0], Z[:,1], s=100, c=Y, alpha=0.5)
        #plt.show()
        z = z + [i]*len(X)

    Y = np.array(Z)

    for i in range(len(Y)):
        x.append(Y[i][0])
        y.append(Y[i][1])

    plot = plt.figure()

    ax =Axes3D(plot)

    ax.scatter(x,y,z)
    plt.show()

if __name__ == '__main__':
    main()