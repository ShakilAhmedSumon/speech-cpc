import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data #numpy array
y = iris.target #numpy array

perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

print(X.shape)
print(y[perm])
print(y)