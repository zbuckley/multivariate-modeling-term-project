# This file the collection of optimization algorithms
#   and tools used throughout the course

import numpy as np

def memoize(func):
    cache = {}
    def cache_func(x):
        if not x in cache:
            cache[x] = func(x)
        return cache[x]
    return cache_func

# Perform Least Square Esimator Batch Compute
#   X Matrix of independent data, features in columns, obs in rows.
#   Y vector of depenedent data, (column vector)
def LSE(X, Y):
    return np.dot(
        np.dot(
            np.linalg.inv(
                np.dot(
                    np.transpose(X),
                    X
                )
            ),
            np.transpose(X)
        ), 
        Y
    )