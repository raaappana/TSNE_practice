import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import shuffle

perp = 30

iter = 15

def processData():
    data = pd.read_excel("wdbc.xlsx")
    return data.as_matrix()

def main():

    dat = processData()


    dat = processData()
    X = dat[:,2:]
    Y = dat[:,1]

    X,Y = shuffle(X,Y)

    X = X[:150,:]
    c = Y[:150]

    for j in range(iter):

        tsne = TSNE(perplexity= perp)

        Z = []
        Z.extend(tsne.fit_transform(X))

        x = []
        y = []
        #z = []
        for i in range(len(Z)):
            x.append(Z[i][0])
            y.append(Z[i][1])

        plot = plt.figure()

        plt.scatter(x,y,c=c)
        plt.show()


if __name__ == '__main__':
    main()