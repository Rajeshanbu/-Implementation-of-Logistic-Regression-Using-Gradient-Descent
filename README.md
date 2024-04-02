# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report b
6. importing the required modules from sklearn.
Obtain the graph. .

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: RAJESH A
RegisterNumber: 212222100042 
*/
```
```py
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:, [0, 1]]
y=data[:, 2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
  
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

x_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j

def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)

np.mean(predict(res.x,X)==y)
```

## Output:
### Array value of X:
![5 1](https://github.com/Rajeshanbu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118924713/f7798562-f12d-4e4b-b876-77f7cda0e583)

### Array value of Y:
![5 2](https://github.com/Rajeshanbu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118924713/39fad63f-9cbc-4e6f-b469-951f36f097ca)

### Exam 1-Score graph:
![5 3](https://github.com/Rajeshanbu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118924713/2f4f7c2d-e5cd-43a7-a4fb-f6365d1f2418)

### Sigmoid function graph:
![5 4](https://github.com/Rajeshanbu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118924713/8ab30e63-56f1-4b5f-a210-2f384fedfad5)
### X_Train_grad value:
![5 5](https://github.com/Rajeshanbu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118924713/704875e9-dc97-4574-9c8a-b4ff7696572a)

### Y_Train_grad value:
![5 6](https://github.com/Rajeshanbu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118924713/ead08da5-f5bd-4519-9cd8-eefbf0d2ee2a)

### Print res.X:
![5 7](https://github.com/Rajeshanbu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118924713/e2997d4d-7dfe-4e85-9822-8525e188eef5)

### Decision boundary-gragh for exam score:
![5 8](https://github.com/Rajeshanbu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118924713/c257de90-310e-4f48-9f73-c772caa38673)

### Probability value:
![5 9](https://github.com/Rajeshanbu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118924713/8e17cb7b-e803-41cb-ad81-abb2d8ac27b7)

### Prediction value of mean:
![5 10](https://github.com/Rajeshanbu/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118924713/a5466606-6bb9-49e3-9734-518652684ff2)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

