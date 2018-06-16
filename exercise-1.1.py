import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import inv

data = pd.read_csv('data/ex1data2.txt', sep=',',header=None)
data.columns = ["x1","x2","y"]

def compute_cost(x,theta,y):
    predicted_cost = np.dot(x,theta)
    diffrence = (predicted_cost - y)
    cost = np.sum(np.power(diffrence,2))
    return cost

def gradientDescent(x,y,theta,alpha,num_iterations):
    xx = []
    yy = []
    i = 0
    while i < num_iterations:
        predicted_cost = np.dot(x,theta)
        diffrence = (predicted_cost - y)
        denominator = x.shape[0]
        theta = theta - (alpha/denominator) * np.dot(x.T,diffrence)
        i = i+1
        if i%100 == 0:
            print("Cost for"+str(i)+"="+str(compute_cost(x,theta,y)))
        xx.append(i)
        yy.append(compute_cost(x,theta,y))
    plt.plot(xx,yy)
    plt.show()
    return theta


x = data[['x1','x2']].values
y = data[['y']].values

x = np.insert(x,0,1,axis=1)
mean_x = x.mean(axis=0)
std_x = x.std(axis=0)

x = (x-mean_x)/(std_x) #normalization
x[np.isnan(x)] = 1 #converting nan to 1's

theta = gradientDescent(x,y,np.zeros([3,1]),0.01,500)

scaled_x = (np.array([1,1650,3]) - mean_x)/(std_x)
scaled_x[np.isnan(scaled_x)] = 1
print(scaled_x)

predict_y = np.dot(scaled_x,theta)
print(predict_y)


theta = np.dot(np.dot(inv(np.dot(x.T,x)),x.T),y)
predict_y = np.dot(scaled_x,theta)
print(predict_y)