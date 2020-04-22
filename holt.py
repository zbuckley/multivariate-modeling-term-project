# This file is responsibly for applying
#   the Holt-Winters Method to the dataset
# Core Python Dependencies
from os.path import sep # attempts to maintain OS portability

# Import Third-part Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import things developed by me either
#   throughout the coursework, or 
#   specifically for the term project
#   (provided in additional python files)
from utils.data import load_y_train, load_y_test, weekly_freq, daily_freq, hourly_freq
from utils.conf import tmp_graphics_folder
import utils.models as mods
import utils.stats as stats
from utils.arma import compose_transform, logarithmic_transform, normalization_transform

# Let's load y_train and y_test. 
#   attempt to fit and predict using a holtwinter model
y_train = load_y_train()
y_test = load_y_test()

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
forecast = mods.holt_linear_trainer(y_train,)
y_forecasted = forecast(len(y_test))
y_test['holt_linear_forecast'] = pd.Series(y_forecasted, index=y_test.index)

# Holt-Winter
#  This takes a very long time... wow.
#  TODO: Can we implement our own version of this?
#  TODO: Can we use LMA to optimize for the parameters? Maybe faster??? hmmm..

y_train_np = y_train.to_numpy().reshape(-1, 1)
y_train_scaled, inverter = compose_transform(
    logarithmic_transform,
    normalization_transform
)(y_train_np)

forecast = mods.holt_winters_trainer_full(y_train_scaled, trend=None, seasonal_periods=int(daily_freq))
y_forecasted = forecast(len(y_test))
y_test['holt_winter_forecast'] = pd.Series(inverter(y_forecasted), index=y_test.index)


# plot all the forecasts
cols = list(y_test.columns)

y_tmp = y_test['Appliances'].to_numpy().reshape(-1, 1)
y_train = y_train.to_numpy().reshape(-1, 1)
y_actual = np.append(y_train, y_tmp)
xs = list(range(y_train.shape[0], y_actual.shape[0]))

cols.remove('Appliances')

# assuming this should be the number of parameters
#  applying to the forecast
#  so i'm ignoring parameters that are 
#  only used in the fit phase of the model. 
num_params = {
    'avg_forecast': 0,
    'naive_forecast': 0,
    'drift_forecast': 0,
    'ses_forecast': 2,
    'holt_linear_forecast': 4,
    'holt_winter_forecast': 6,
    'holt_winter_forecast_normalization': 6
}

for col in cols:
    legend = ['Actual']
    legend.append(col)
    plt.plot(y_actual)
    y_tmp2 = y_test[col].to_numpy().reshape(-1, 1)
    plt.plot(xs, y_tmp2)
    plt.title(f'Actual and Pred vs Time\n{col}')
    plt.savefig(f'{tmp_graphics_folder}{sep}{col}-actual-pred-vs-time')
    plt.figure()
    print('Model Metrics for', col)
    stats.print_metrics(y_tmp, y_tmp2, num_params[col], y_train.shape[0])
