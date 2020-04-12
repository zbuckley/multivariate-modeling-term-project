# This file is responsibly for applying
#   the Holt-Winters Method to the dataset
# Core Python Dependencies
from os.path import sep # attempts to maintain OS portability

# Import Third-part Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Import things developed by me either
#   throughout the coursework, or 
#   specifically for the term project
#   (provided in additional python files)
import utils
import classic_models as mods

# Let's load y_train and y_test. 
#   attempt to fit and predict using a holtwinter model
y_train = utils.load_y_train()
y_test = utils.load_y_test()

# Average
forecast = mods.avg_trainer(y_train)
y_forecasted = forecast(len(y_test))
y_test['avg_forecast'] = pd.Series(y_forecasted, index=y_test.index)

# Naive
forecast = mods.naive_trainer(y_train)
y_forecasted = forecast(len(y_test))
y_test['naive_forecast'] = pd.Series(y_forecasted, index=y_test.index)

# Drift
forecast = mods.drift_trainer(y_train)
y_forecasted = forecast(len(y_test))
y_test['drift_forecast'] = pd.Series(y_forecasted, index=y_test.index)

# Simple Exponential Smoothing
#  TODO: Can we use LMA to optimize alpha, a_0
forecast = mods.ses_trainer(y_train)
y_forecasted = forecast(len(y_test))
y_test['ses_forecast'] = pd.Series(y_forecasted, index=y_test.index)

# Holt Linear Method
#  TODO: Can we use LMA to optimize alpha, beta, l_0, b_0
forecast = mods.holt_linear_trainer(y_train)
y_forecasted = forecast(len(y_test))
y_test['holt_linear_forecast'] = pd.Series(y_forecasted, index=y_test.index)

# Hold-Winter
#  This takes a very long time... wow.
#  TODO: Can we implement our own version of this
#  TODO: Can we use LMA to optimize for the parameters.
forecast = mods.holt_winters_trainer_full(y_train, trend=None, seasonal_periods=utils.daily_freq)
y_forecasted = forecast(len(y_test))
y_test['hold_winter_forecast'] = pd.Series(y_forecasted, index=y_test.index)


# plot all the forecasts
legend = []
for col in y_test.columns:
    legend.append(col)
    y_test[col].plot()

plt.legend(legend)
plt.savefig('wat')
plt.figure()
