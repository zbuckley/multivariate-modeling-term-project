# This file contains code spcifically related to the holt-winter
#   and simpler related models types. Largely in support of the 
#   holt-winter.py file for building holt-winter models of the 
#   dataset. The contents of this file are based on code 
#   developed for lab-4.

# The functions implemented here take a dataframe/series as input
#   and then return a forecasting function for that dataset.
 
# Core Python Dependencies

# Third-Party Dependencies
import numpy as np
import statsmodels.tsa.holtwinters as ets

# average method
#  train function returns the forecast function.
def avg_trainer(df):
    y_data = np.asarray(df)
    mean = np.mean(y_data)
    def avg_forecast(h): 
        return np.repeat(mean, h)
    return avg_forecast

# naive method
#   train function returns the forecast function.
def naive_trainer(df):
    y_data = np.asarray(df)
    y_T = y_data[len(y_data)-1]
    def naive_forecast(h):
        return np.repeat(y_T, h)
    return naive_forecast

# drift method
#   train function returns the forecast function.
def drift_trainer(df):
    y_data = np.asarray(df)
    y_T = y_data[len(y_data)-1]
    y_0 = y_data[0]
    T = len(y_data)
    slope = (y_T - y_0)/(T - 1)
    def drift_forecast(h):
        hs = np.array(range(1, h + 1))
        return y_T + hs*slope
    return drift_forecast

# Simple Exponential Smoothing
#   train function returns the forecast function
#   we're assuming alpha is 0.5, and initial condition is 0 as instructed.
def ses_trainer(
    df,
    alpha=0.5, 
    l_0=0):

    def l(t):
        _l = alpha * y_data[t] + (1 - alpha) * l_0
        for i in range(t):
            _l = alpha * y_data[i] + (1 - alpha) * _l
        return _l
    
    y_data = np.asarray(df)
    l_T = l(len(y_data)-1)

    def ses_forecast(h):
        return np.repeat(l_T, h)
    return ses_forecast

# Holt Linear Train
#   train function returns the predict function
#   we're assuming alpha is 0.5, and initial condition is 0 as instructed.
# Parameters below were selected by hand after numerous attempts.
def holt_linear_trainer(
    df,
    alpha = 0.5,
    beta = 0.5,
    l_0 = 0.0,
    b_0 = 0.0
):

    # Below is a for loop which performs the recursive operation iteratively
    def l_b(t):
        l = l_0
        b = b_0
        for i in range(t):
            l_new = np.nan_to_num(alpha * y_data[i] + (1 - alpha) * (l * b))
            b = np.nan_to_num(beta * (l_new - l) + (1 - beta) * b)
            l = l_new
            #   print(i, l, b)
        return l, b
    
    y_data = np.asarray(df)
    l_T, b_T = l_b  (len(y_data) - 1)

    def holt_linear_predict(h):
        hs = np.array(range(1, h + 1))
        return l_T + hs * b_T
    return holt_linear_predict

# proxying the statsmodel api to the api i came up with.
#   Ouch... This takes a long time to fit
def holt_winters_trainer_full(
    df,
    seasonal='additive',
    trend='additive', 
    seasonal_periods=None):
    
    # optimized ExponentialSmoothing on y_data
    print('WARNING: This takes a long time. Recommend a coffee break.')
    model = ets.ExponentialSmoothing(
        np.asarray(df),
        dates=np.asarray(df.index),
        seasonal=seasonal,
        trend=trend,
        seasonal_periods=seasonal_periods
    ).fit()
    def holt_winters_predict(h):
        return model.forecast(h)
    return holt_winters_predict
