# This file provides a collection of routines and utilities used 
#   throughout the other python programs. 
# For the most part, These are slightly modified versions of the 
#   utilities we've developed throughout the course. 
import numpy as np

# Pearson's correlation coefficient estimation as developed 
#   in lab 2. Renamed, and converted to a numpy ufunc
# def correlation_coefficient_cal(x, y):
def __corr(x, y):
    # assumes that datasets are of the same length
    assert(len(x) == len(y))

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum(
        np.multiply(
            np.subtract(x, x_mean),
            np.subtract(y, y_mean)
        )
    )

    denominator = np.sqrt(np.sum(np.power(
        np.subtract(x, x_mean),
        2
    ))) * np.sqrt(np.sum(np.power(
        np.subtract(y, y_mean),
        2
    )))

    return numerator/denominator
corr = np.frompyfunc(__corr)
