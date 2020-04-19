# This file defines functions related to generating GPAC tables
#  for ARMA Model Identification

# Core Python Dependencies

# Import Third-part Libraries
import numpy as np

# Import things developed by me either
#   throughout the coursework, or 
#   specifically for the term project
#   (provided in additional python files)
from .stats import autocorrelation_estimation
from .optimization import memoize

# function for defining an Ry function as described in class
def _R(y):  
    y_mean = np.mean(y)

    def Ry(k): 
        return autocorrelation_estimation(y, k, y_mean=y_mean)  
    
    # add memoization, and vectorization to autocorrelation function for y
    return np.vectorize(memoize(Ry))

# B, denominator of cramer rule fraction
#   Eq. 2.3 from paper, uisng Ry instead of p
def _B(s, t, Ry): 
    retVal = np.zeros((s, s))
    for i in range(s):
        retVal[:, i] = Ry(np.array(range(t - i, t + s - i)))
    return retVal

# A, numerator of cramer rule fraction
#   paragraph after Eq. 2.3 in paper, using Ry instead of p
def _A(s, t, Ry):
    retVal = _B(s, t, Ry)
    retVal[:, s - 1] = Ry(np.array(range(t + 1, t + s + 1)))
    return retVal

# kth autoregressive coefficient of ARMA(k, j)
#   partially based on paper:
#     https://www.smu.edu/-/media/Site/Dedman/Departments/Statistics/TechReports/TR-222.pdf?la=en
#   k is autoregression coefficient number, and autoregression order of ARMA model
#   j is moving average order of ARMA model
#   Ry is autocorrelation function for y
#       expected to be vectorized (see numpy vectorize)
#       performance improves when also memoized (see function above)
def _phi(k, j, Ry):
    # sanity check
    assert(k > 0)

    # equation 2.2 from paper, using Ry instead of p
    if k is 1:
        return Ry(j + 1)/Ry(j)
    else:
        return np.linalg.det(_A(k, j, Ry))/np.linalg.det(_B(k, j, Ry))

def gpac(sig, ar_order_max, ma_order_max):
    Ry = _R(sig)
    
    retVal = np.zeros((ma_order_max, ar_order_max))

    # let's go row in outer loop, that seems to be how 
    #   numpy organizes the ndarray types, so should 
    #   go a bit faster *shrug*
    for i in range(ma_order_max):
        for j in range(ar_order_max):
            retVal[i, j] = _phi(j + 1, i, Ry)
    return retVal
