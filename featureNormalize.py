import numpy as np
import sys
"""
function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
"""
def featureNormalize(X):
    #X_norm = X;
    X_norm = X
    #mu = zeros(1, size(X, 2));
    mu = np.zeros((1, X.shape[1]), dtype=float)
    #sigma = zeros(1, size(X, 2));
    sigma = np.zeros((1, X.shape[1]), dtype=float)

    # % ====================== YOUR CODE HERE ======================
    # % Instructions: First, for each feature dimension, compute the mean
    # %               of the feature and subtract it from the dataset,
    # %               storing the mean value in mu. Next, compute the 
    # %               standard deviation of each feature and divide
    # %               each feature by it's standard deviation, storing
    # %               the standard deviation in sigma. 
    # %
    # %               Note that X is a matrix where each column is a 
    # %               feature and each row is an example. You need 
    # %               to perform the normalization separately for 
    # %               each feature. 
    # %
    # % Hint: You might find the 'mean' and 'std' functions useful.
    # %
    # mu = mu + mean(X);
    mu += np.average(X)
    # sigma = sigma + std(X_norm);
    sigma += np.std(X_norm)

    # X_norm(:,:)
    # for i=1:size(X_norm,2)
    #     X_norm(:,i) = X_norm(:,i) - mu(:,i);
    #     X_norm(:,i) = X_norm(:,i) / sigma(:,i);
    # end
    for i in xrange(X.shape[1]):
        X_norm[:,i] -= mu[:,i]
        X_norm[:,i] /= sigma[:,i]

    print(X_norm)
    return X_norm, mu, sigma




#% ============================================================

#end
