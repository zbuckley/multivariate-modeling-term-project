# This file will find a decomposition that is reasonable
#   for the dataset.
from os.path import sep # attempts to maintain OS portability

# Import Third-part Libraries
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import things developed by me either
#   throughout the coursework, or 
#   specifically for the term project
#   (provided in additional python files)
from utils.data import load_y_train, weekly_freq, daily_freq, hourly_freq
from utils.conf import tmp_graphics_folder

# Load data and convert to numpy array
y_train = load_y_train()
data = y_train.to_numpy().reshape(-1)

# really doesn't seem to be a need to look at a multiplicative model...
#   found a better way to think of the freq parameter... based on 
#   ExponentialSmoothing model's documentation. 
#   https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
#   "The number of periods in a complete seasonal cycle, e.g., 4 for quarterly data or 7 for daily data with a weekly cycle."
res = seasonal_decompose(data, model='additive', freq=weekly_freq)
res.plot()
plt.savefig(f'{tmp_graphics_folder}{sep}dep-decompose-additive-weekly')
plt.figure()

# let's try daily
#   number of minutes in a day
mins_per_day = 60*24
res = seasonal_decompose(data, model='additive', freq=daily_freq)
res.plot()
plt.savefig(f'{tmp_graphics_folder}{sep}dep-decompose-additive-daily')
plt.figure()

# let's try hourly
#   number of minutes in a hour
mins_per_hour = 60
res = seasonal_decompose(data, model='additive', freq=hourly_freq)
res.plot()
plt.savefig(f'{tmp_graphics_folder}{sep}dep-decompose-additive-hourly')
plt.figure()
