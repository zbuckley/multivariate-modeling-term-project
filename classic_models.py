# This file contains code spcifically related to the holt-winter
#   and simpler models types. Largely in support of the 
#   holt-winter.py file for building holt-winter models of the 
#   dataset. The contents of this file are based on code 
#   developed for lab-4.
# For the most part, These are slightly modified versions of the 
#   utilities we've developed throughout the course.
# The models implemented here take a dataframe/series as input
#   and then return forecasting function for that dataset.
# Update: modified the returned forecast functions to forecast 
#   1 to h values, more in line with existing statsmodel api
#   functionality.
 
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
def ses_trainer(df):
    alpha = 0.5
    initial_condition = 0

    # using component form, and defining l(t) recursively
    def l(t):
        if t == 0:
            return alpha * y_data[t] + (1 - alpha) * initial_condition
        else:
            return alpha * y_data[t] + (1 - alpha) * l(t - 1)
    
    
    y_data = np.asarray(df)
    l_T = l(len(y_data)-1)

    def ses_forecast(h):
        return np.repeat(l_T, h)
    return ses_forecast

# Holt Linear Train
#   train function returns the predict function
#   we're assuming alpha is 0.5, and initial condition is 0 as instructed.
# Parameters below were selected by hand after numerous attempts.
def holt_linear_trainer(df):
    alpha = 0.5
    beta = 0.5
    l_0 = 0.0
    b_0 = 0.0

    # using component form, and defining l(t) recursively
    # this gets really slow... let's speed it up by memoizing
    l_res_map = {}
    def l(t):
        if not t in l_res_map:
            if t == 0:
                l_res_map[t] = l_0
            else:
                l_res_map[t] = alpha * y_data[t] + (1 - alpha) * (l(t - 1) + b(t-1))
        return l_res_map[t]

    b_res_map = {}
    def b(t):
        if not t in b_res_map:
            if t == 0:
                b_res_map[t] = b_0
            else:
                b_res_map[t] = beta * (l(t) - l(t-1)) + (1 - beta) * b(t-1)
        return b_res_map[t]

    
    y_data = np.asarray(df)
    l_T = l(len(y_data)-1)
    b_T = b(len(y_data)-1)

    def holt_linear_predict(h):
        hs = np.array(range(1, h + 1))
        return l_T + hs * b_T
    return holt_linear_predict

# proxying the statsmodel api to the api i came up with.
def holt_winters_trainer_full(df):
    # optimized ExponentialSmoothing on y_data
    model = ets.ExponentialSmoothing(np.asarray(df), dates=np.asarray(df.index), freq=df.index.freq).fit()
    def holt_winters_predict(h):
        return model.forecast(h)
    return holt_winters_predict

