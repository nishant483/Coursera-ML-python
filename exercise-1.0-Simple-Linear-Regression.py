import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



data = pd.read_csv('data/ex1data1.txt', sep=',',header=None)  #loading data in dataframe
# #data can be directly loaded into numpy arrays but i am putting into dataframe for my love to Pandas


data.columns = ["x","y"] #Assigning columns
data['ones'] = 1 #adding an extra column 1 to the data frame


x = np.array(data['x']) #loading x from the dataframe
y = np.array(data['y']) #loading y from the dataframe

plt.scatter(x,y,c='r',marker = 'x') #plotting data on the graph

plt.xlabel('Population of City in 10,000s') #setting the label
plt.ylabel('Profit in $10,000s') #setting the label
plt.show() #show


x_array = data[['ones','x']].values #loading data into numpy array from pandas dataframe

y_array = data[['y']]


def compute_cost(x,theta,y):
    predicted_value = np.dot(x,theta) #computing predicted
    diffrence = (predicted_value - y)
    cost = np.sum(np.power(diffrence,2))
    return cost

def gradientDescent(x,y,theta,alpha,num_iterations):
    i = 0
    while i < num_iterations:
        i = i+1
        predicted_cost = np.dot(x,theta)
        diffrence = (predicted_cost - y)
        denominator = x.shape[0]
        theta = theta - (alpha/denominator) * np.dot(x.T,diffrence)
        if i%100 == 0:
            print("Cost for"+str(i)+"="+str(compute_cost(x,theta,y)))
    return theta


# theta = np.zeros([2,1])

theta = gradientDescent(x_array,y_array,np.zeros([2,1]),0.01,1500)

#to plot the most accurate line through data
xx = np.arange(0,data['x'].max())
z = np.ones((xx.shape[0],1))
yy = np.dot(np.column_stack((z,xx)),theta)

plt.plot(xx,yy,label='Linear regression (Gradient Descent)')

regr = LinearRegression()
regr.fit(data[['x']].values.reshape(-1,1), data[['y']].values)

xx = xx.reshape(xx.shape[0],-1)
# print((regr.coef_*xx).shape)
plt.plot(xx, regr.intercept_+regr.coef_*xx, label='Linear regression (Scikit-learn GLM)')

plt.show()

print(gradientDescent(x_array,y_array,np.zeros([2,1]),0.01,1500))
compute_cost(x_array,np.zeros([2,1]),y_array)





