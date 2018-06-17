import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize

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


def onevsall(X,y,num_labels,lmbda):
    initialize_theta = np.zeros((n,1))
    all_theta = np.zeros([num_labels,X.shape[1]])
    for i in range(1,num_labels+1):
        reg_logistic_regression = minimize(fun = reg_cost_function,x0=initialize_theta,args=(X,(y == i)*1,lmbda),jac=reg_gradient_function,options = {'maxiter' : 100})
        all_theta[i - 1] = reg_logistic_regression.x
    return all_theta


def predict_onevsall(X,all_theta):
    predict = sigmoid(np.dot(X,all_theta.T))
    return (np.argmax(predict,axis=1)+1)

def predict_nn(X,theta1,theta2):
    a2 = np.c_[np.ones(m),sigmoid(np.dot(X,theta1.T))]
    predictions = predict_onevsall(a2,theta2)
    return predictions

data = loadmat('data/ex3data1.mat')


X = np.c_[np.ones( data['X'].shape[0] ), data['X']]
y = data['y']

m = X.shape[0] # number of training examples
n = X.shape[1] # number of features (with bias unit)

all_theta = onevsall(X,y,10,0.1)
predictions = predict_onevsall(X,all_theta)
y = y.ravel()
print((y[predictions == y].shape[0]*100)/m)


data2 = loadmat('data/ex3weights.mat')
theta1, theta2 = data2['Theta1'], data2['Theta2']

predictions = predict_nn(X,theta1,theta2)
y = y.ravel()
print((y[predictions == y].shape[0]*100)/m)