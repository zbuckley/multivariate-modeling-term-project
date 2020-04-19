# This file defines various statistics related functions

# Third Party Imports
import numpy as np
from scipy.stats import distributions, chi2
import matplotlib.pyplot as plt # supports acf_plot function

# assumes 2 sided t-test
#   uses the survival function of t distribution to 
#   figure out the p-value given a t-statistic
#   and the degrees of freedom.
def ttest_pvalue(t, df, sides=2):
    # sides should be 1 or 2
    assert(sides == 1 or sides == 2)

    # adapted from scipy stats code
    #   https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/stats/stats.py#L4988
    return distributions.t.sf(np.abs(t), df) * sides

# t-Test for computing the statistical significance of a given correlation r, and 
#   dataset size n.
#   https://stats.stackexchange.com/questions/344006/understanding-t-test-for-linear-regression
#   https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
def ttest(r, n):
    t = r * np.sqrt(
        (n-2)/(1 - r**2)
    )
    
    prob = ttest_pvalue(t, n-2)

    return (t, prob)

# differs from lab 8/lab 9
#   but based on that work, and wikipedia
#   https://en.wikipedia.org/wiki/Standard_error
def ttest_1sample(sample_mean, std_err, target_mean=0):
    return (sample_mean - target_mean)/std_err

# Pearson's correlation coefficient estimation as developed 
#   in lab 2. Renamed
# def correlation_coefficient_cal(x, y):
def corr(x, y):
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


# 11 - Autocorrelation Function (ACF)
# python code for estimating AutoCorrelation Function
#   copied from lab 3
def autocorrelation_estimation(y, k, y_mean = None):
    if y_mean == None:
        y_mean = np.mean(y)
    
    # absolute value, as negative k values are symmetric with positive ones
    #  this makes it much easier to deal with below.
    k = np.abs(k)

    # length of y
    T = len(y)

    # initial t of 1st part of numerator summation
    t0 = k

    # final t of 2nd part of numerator summation
    t1 = T - k
    
    return np.sum (
        np.multiply(
            (y[t0:] - y_mean),
            (y[:t1] - y_mean)
        )
    ) / np.sum (
        np.power(
            y - y_mean,
            2
        )
    )

# copied and adapted from lab 3
#  updated to allow caller to manage saving plot
def acf_plot(residuals, label, k_max):
    residuals_mean = np.mean(residuals)
    acf = [(k, autocorrelation_estimation(residuals, k, y_mean=residuals_mean))
        for k in range(-k_max, k_max + 1)]

    # setting use_line_collection to true hides a deprecation warning
    plt.stem([a for a, b in acf],
        [b for a, b in acf])
    plt.title(f'ACF Plot of Generated Signal\n{label}')


# compute r_ab, t-test, and p-value
#   r_ab is Pearson's correlation coefficient between
#      a and b. 
#   t-test is the t-statistic value computed based on 
#      r_ab and the length of a and b. 
#   p-value is a measure of the statistical significance
#      of the r_ab estimation.
def cor_and_ttest(a, b):
    n = len(a)

    # verify sanity
    assert(len(a) == len(b))
    assert(n > 2)
    r_ab = corr(a, b)
    t, prob = ttest(r_ab, n)
    
    return r_ab, t, prob

# assuming h based on this guidance:
#  https://robjhyndman.com/hyndsight/ljung-box-test/
def _h(T):
    # assumes that the timeseries data is not seasonal
    #   seems appropriate as we really hope our residuals don't 
    #   show seasonal bahavior
    return min(10, int(T/5))


# based on reviewing lecture slides/notes
# and referring to scipy source code
def q_value(residuals, T, h=None):
    if h is None:
        h = _h(T)
    res_mean = np.mean(residuals)
    acfs = [autocorrelation_estimation(residuals, k, res_mean) for k in range(1, h + 1)]
    return T * np.sum(np.power(acfs, 2)[:h])

# Ljung-Box Q* from lectures
def q_value2(residuals, T, h=None):
    if h is None:
        h = _h(T)
    res_mean = np.mean(residuals)
    ks = range(1, h + 1)
    sum_tmp = sum([((T - k)**(-1)) * (autocorrelation_estimation(residuals, k, res_mean)**2) for k in ks])
    return T * (T + 2) * sum_tmp

# convert q to a p-value based on the survival function of the chi2 distribution
def q_to_pvalue(q, T, h=None):
    if h is None:
        h = _h(T)
    # adapted from https://www.statsmodels.org/stable/_modules/statsmodels/tsa/stattools.html#q_stat
    #   and 
    return chi2.sf(q, h)

# adjusted r2 given
#   r2, pearson r squared
#   T, number of observations
def adj_r2(r2, T):
    return 1 - (1 - r2) * (T - 1)/(T - 1)

# sse given
#  residuals
def sse(residuals):
    return np.sum(residuals**2)

# rmse given 
#   residuals
def rmse(residuals):
    return np.sqrt(np.mean(residuals**2))

# AIC based on lecture notes
#  T is number of observations
#  SSE is sum of squared error
#  k is number of parameters
#  TODO: uncertain if hardcoded 2,2 values need
#    adjustment per degrees of freedom or something?
def AIC(T, residuals, k):
    return T*np.log(sse(residuals)/T) + 2*(k + 2)

# AICc based on lecture notes
#  T is number of observations
#  SSE is sum of squared error
#  k is number of parameters
#  TODO: uncertain if hardcoded 2,2,3,3 values need 
#    adjustment per degrees of freedom or something?
def AICc(T, residuals, k):
    return AIC(T, residuals, k) + (2 * (k + 2) * (k + 3)) / (T - k - 3)

# BIC based on lecture notes
#  T is number of observations
#  SSE is sum of squared error
#  k is number of parameters
#  TODO: uncertain if hardcoded 2 needs
#     adjustment per degrees of freedom or something?
def BIC(T, residuals, k):
    return T * np.log(sse(residuals) / T) + (k + 2) * np.log(T)