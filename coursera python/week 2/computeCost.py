import numpy as np

def computeCost (X, y, theta):
    m = len(y)
    J = 0
    Y = np.array(y).reshape(m, 1)
    predictions = np.matmul(X, theta)
    errorDifference = np.subtract(predictions, Y)
    errorSquared = np.power(errorDifference, 2)

    J = (1/(2 * m)) * sum(errorSquared)
    return J
pass