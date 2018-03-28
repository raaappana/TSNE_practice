# This code takes the sklearn breastcancer data, performs t-SNE on it, and then plots it on a 3D plot.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

perp = 290

def processData():
    data = pd.read_excel("wdbc.xlsx")
    return data.as_matrix()



def main():

    dat = processData()
    X = dat[:,2:]
    Y = dat[:,1]

    X,Y = shuffle(X,Y)

    X = X[:150,:]
    c = Y[:150]


    for j in range(10):

        tsne = TSNE(n_components=3,perplexity= perp)

        Z = []
        Z.extend(tsne.fit_transform(X))

        x = []
        y = []
        z = []
        for i in range(len(Z)):
            x.append(Z[i][0])
            y.append(Z[i][1])
            z.append(Z[i][2])

        plot = plt.figure()

        ax =Axes3D(plot)

        ax.scatter(x,y,z,c=c)
        plt.show()


if __name__ == '__main__':
    main()
