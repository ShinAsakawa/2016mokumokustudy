#!/bin/env python
from __future__ import absolute_import
from __future__ import print_function
import six
import numpy as np
import sys


def computeCost(X, y, theta):
#function J = computeCost(X, y, theta)
#%COMPUTECOST Compute cost for linear regression
#%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
#%   parameter for linear regression to fit the data points in X and y

#% Initialize some useful values
#m = length(y)# ; % number of training examples
    m = len(y)

# % You need to return the following variables correctly 
    J = 0;
#J = np.zeros(len(y[0]))

#% ====================== YOUR CODE HERE ======================
#% Instructions: Compute the cost of a particular choice of theta
#%               You should set J to the cost.

#J = sum((X*theta - y) .^ 2)/(2*m);
    J = np.sum(np.square(np.dot(X,theta) - y),axis=0) / (2 * m)

#% =========================================================================
#end
    return J



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
    # one_col_vec = np.ndarray(len(X),dtype=float)
    # X = np.concatenate(([one_col_vec, X]), axis=0)
    # print(X[:10,])
    # sys.exit(-1)
    for i in xrange(len(xlist)):
        X[i,0] = 1.
        X[i,1], Y[i]  = xlist[i], ylist[i]

    theta = np.zeros(2,)
    print(computeCost(X, Y, theta))
