import numpy as np

def gradientDescent(X, y, theta, alpha, n):
    m = len(y)
    x = X[:,1]
    for _ in range(n):
        h = theta[0] + (theta[1] * x)
        thetaZero = theta[0] - (alpha * (1/m) * sum(h - y))
        thetaOne = theta[1] - (alpha * (1/m) * sum(np.multiply(h - y, x)))

        theta = np.array([thetaZero, thetaOne])
    pass
    return theta
pass