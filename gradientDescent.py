#!/bin/env python
from __future__ import absolute_import
from __future__ import print_function
import six
import numpy as np
import sys
import matplotlib.pyplot as plt

def computeCost(X, y, theta):
    m = len(y)
    J = 0;
    J = np.sum(np.square(np.dot(X,theta) - y),axis=0) / (2 * m)
    return J


#function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
def gradientDescent(X, y, theta, alpha, num_iters):
#%GRADIENTDESCENT Performs gradient descent to learn theta
#%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
#%   taking num_iters gradient steps with learning rate alpha

    #% Initialize some useful values
    #m = length(y); % number of training examples
    m = len(y) #; % number of training examples
    #J_history = zeros(num_iters, 1);
    J_history = np.zeros((num_iters, 1))

    # for iter = 1:num_iters
    for iter in range(0,num_iters):
    #% ====================== YOUR CODE HERE ======================
    #% Instructions: Perform a single gradient step on the parameter vector
    #%               theta. 
    #%
    #% Hint: While debugging, it can be useful to print out the values
    #%       of the cost function (computeCost) and gradient here.
    #%
        #theta = theta - alpha/m * (X' * (X * theta - y));
        theta += - alpha/m * np.dot(np.transpose(X),(np.dot(X,theta)-y))
        #P = np.dot(X,theta) - y
        #Z = np.dot(np.transpose(X),P)
        #theta += - alpha/m * Z
        #theta = theta - alpha/m * np.dot(X,(X * theta -y))

    #% ============================================================

    #% Save the cost J in every iteration    
        #J_history[iter] = computeCost(X, y, theta);
        J_history[iter] = computeCost(X,Y,theta)
    #end
    #end
    return theta, J_history

if __name__ == '__main__':
    with open('ex1data1.txt') as f:
        data = f.readlines()
        f.close()

    xlist, ylist = [],[]
    for str in data:
        x, y = str.strip().split(',')
        xlist.append(x)
        ylist.append(y)

    X = np.ndarray((len(xlist),2))
    Y = np.ndarray((len(ylist),))
    for i in xrange(len(xlist)):
        X[i,0] = 1.
        X[i,1], Y[i]  = xlist[i], ylist[i]

    init = np.ndarray((2,), buffer=np.array([-0.5,0.5]))
    alpha, maxIter = 0.01, 10
    theta, J_history = gradientDescent(X, Y, init, alpha, maxIter)
    print('theta:', theta)
    plt.figure()
    plt.title('J as a function of iteration')
    plt.plot(range(len(J_history)), J_history)
    plt.show()
