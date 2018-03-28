# Make a hypercube, perform t-SNE on it, and then plots it on a 2D plot.

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


X = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
         [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]]

model = TSNE(n_components=2, random_state=0)

Y = model.fit_transform(X)
Y = list(Y)

x = []
y = []
for i in range(len(Y)):
    x.append(Y[i][0])
    y.append(Y[i][1])

plt.scatter(x,y)
plt.show()
