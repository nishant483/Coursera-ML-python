import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize

data = pd.read_csv('data/ex2data1.txt', header=None)
data.columns = ['exam1', 'exam2', 'admitted']


def plot_function(px, py, nx, ny, c1, c2, xmin, xmax, ymin, ymax, label1, label2):
    plt.scatter(px, py, c='k', s=50, label=c1)
    plt.scatter(nx, ny, c='y', s=50, label=c2)

    plt.gca().set_xlim([xmin, xmax])
    plt.gca().set_ylim([ymin, ymax])
    plt.gca().set_xlabel(label1)
    plt.gca().set_ylabel(label2)
    plt.gca().legend(bbox_to_anchor=(1.0, 1.0), fancybox=True)


plot_function(data[data.admitted == 0]['exam1'], data[data.admitted == 0]['exam2'], data[data.admitted == 1]['exam1'],
              data[data.admitted == 1]['exam2'], "Admitted", "Not-Admitted", data['exam1'].min() - 10,
              data['exam1'].max() + 10, data['exam2'].min() - 10, data['exam2'].max() + 10, "Exam 1 Score",
              "Exam 2 Score")


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def costFunction(theta, x, y):
    m = x.shape[0]
    predicted_value = sigmoid(np.dot(x, theta))
    cost = (-1 / m) * (np.dot(y.T, np.log(predicted_value)) + np.dot((1 - y).T, np.log(1 - predicted_value)))

    if np.isnan(cost[0]):
        return (np.inf)

    return cost[0]


def gradientDescent(theta, x, y):
    m = x.shape[0]
    n = x.shape[1]
    theta = np.array(theta).reshape(n, 1)
    predicted_value = sigmoid(np.dot(x, theta))
    diffrence = (predicted_value - y)
    theta = ((1 / m) * np.dot(diffrence.T, x)).T.flatten()
    return theta


def predict(theta, x, threshold=0.5):
    return (sigmoid(np.dot(x, theta)) >= 0.5).astype('int')


def poly_features(X, deg):
    x1 = X[:, 0]
    x2 = X[:, 1]
    features = X
    count = 0
    for i in range(2, deg + 1):
        for j in range(0, i + 1):
            count += 1
            column = np.array(x1 ** (i - j) * x2 ** j).reshape(X.shape[0], 1)
            features = np.hstack((features, column))
    features = np.insert(features, 0, 1, axis=1)
    return features

def reg_cost_function ( theta, x, y, lmda ):
    theta = theta.reshape(theta.shape[0],1)
    m = x.shape[0]
    predicted_value = sigmoid(np.dot(x, theta))
    reg_theta = np.delete(theta,0,axis=0)
    cost = (-1 / m) * (np.dot(y.T, np.log(predicted_value)) + np.dot((1 - y).T, np.log(1 - predicted_value))) +(lmda / (2 * m)) * np.sum(np.square(reg_theta))
    if np.isnan(cost[0]):
        return (np.inf)

    return cost[0]


def reg_gradient_function ( theta, x, y, lmda ):
    m = x.shape[0]
    n = x.shape[1]
    theta = np.array(theta).reshape(n, 1)
    predicted_value = sigmoid(np.dot(x, theta))
    diffrence = (predicted_value - y)
    reg_theta = np.insert(np.delete((lmda / m) * theta, 0, axis=0),0,0,axis=0)
    optimized_theta = ((1 / m) * np.dot(diffrence.T, x).T + reg_theta).flatten()
    return optimized_theta


x = np.array(data[['exam1', 'exam2']])
m = x.shape[0]
n = x.shape[1]
y = np.array(data['admitted']).reshape(m, 1)
theta = np.zeros([3, 1])
x = np.insert(x, 0, 1, axis=1)

cost = costFunction(theta, x, y)
theta = gradientDescent(theta, x, y)
print(theta)

logistic_regression = minimize(fun=costFunction, x0=np.zeros([3, 1]),
                               args=(x, y), jac=gradientDescent, options={'maxiter': 400})
print(logistic_regression)

theta = logistic_regression.x.reshape(n + 1, 1)

# example with marks - exam1 - 45,exam2 - 85
probability = sigmoid(np.dot(np.array([1, 45, 85]), theta))

p = predict(theta, x)
print('Training Score : {0}%'.format((y[y == p].shape[0] * 100) / y.shape[0]))

# Plot the decision boundary
plt.scatter(45, 85, s=50, c='r', marker='s', label='"Test" data')

xx1, xx2 = np.meshgrid(np.linspace(0, 100), np.linspace(0, 100))
# Grid of all points (2500 by default) in the mesh, with entry of 1 in front ( for theta0 )
X_grid = np.c_[np.ones((np.ravel(xx1).shape[0], 1)), np.ravel(xx1), np.ravel(xx2)]
h = sigmoid(X_grid.dot(logistic_regression.x))
h = h.reshape(xx1.shape)

plt.contour(xx1, xx2, h, [0.5], linewidths=0.5, colors='b')
# plt.show()


plt.clf()
data = pd.read_csv('data/ex2data2.txt', header=None)
data.columns = ['chip1', 'chip2', 'y']

plot_function(data[data.y == 0]['chip1'], data[data.y == 0]['chip2'], data[data.y == 1]['chip1'],
              data[data.y == 1]['chip2'], "y=0", "y=1", data['chip1'].min() - 0.5, data['chip1'].max() + 0.5,
              data['chip2'].min() - 0.5, data['chip2'].max() + 0.5, "Microchip Test1", "Microchip Test2")
# plt.show()

X = np.array(data[['chip1', 'chip2']])
y = np.array(data['y'])

features = poly_features(X, 6)

m = features.shape[0]
n = features.shape[1]

y = y.reshape(m,1)


theta = np.zeros([n, 1])

cost = reg_cost_function(theta, features, y,1)
theta = reg_gradient_function(theta, features, y,1)
print(theta)

reg_logistic_regression = minimize(fun=reg_cost_function, x0=np.zeros([n, 1]),
                               args=(features, y,1), jac=reg_gradient_function, options={'maxiter': 400})
print(reg_logistic_regression)

xx1, xx2 = np.meshgrid(np.linspace(data['chip1'].min(), data['chip1'].max()), np.linspace(data['chip2'].min(), data['chip2'].max()))

# Grid of all points (2500 by default) in the mesh, with entry of 1 in front ( for theta0 )
X_grid = np.c_[np.ravel(xx1), np.ravel(xx2) ]
X_grid = poly_features(X_grid,6)
h = sigmoid( X_grid.dot(reg_logistic_regression.x) )
h = h.reshape(xx1.shape)

plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='chartreuse')

plt.show()
