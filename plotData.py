#!/bin/env python
from __future__ import absolute_import
from __future__ import print_function
import six
import matplotlib.pyplot as plt

with open('ex1data1.txt') as f:
    data = f.readlines()
    f.close()

X, Y =  [], []
for str in data:
    x, y = str.strip().split(',')
    X.append(x)
    Y.append(y)
m = len(y)

print(len(X), len(Y))

plt.scatter(X, Y, c='r', marker='x')
plt.show()
