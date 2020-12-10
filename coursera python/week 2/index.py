import numpy as np
import matplotlib.pyplot as plt

from gradientDescent import gradientDescent
from computeCost import computeCost

# load data into x and y
data = np.loadtxt('./coursera python/week 2/ex1data1.txt', delimiter=',')
x, y = zip(*data)
m = len(y)

# plotting x and y
# plt.scatter(x, y)
# plt.show()

X = np.column_stack((np.ones((m, 1)), np.array(x).reshape((m, 1))))

# configuration for gradient descent
theta = np.zeros((2, 1))
iterations = 1500
alpha = 0.01


theta = gradientDescent(X, y, theta, alpha, iterations)
J = computeCost(X, y, theta)
print(J)