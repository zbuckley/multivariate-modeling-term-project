# This file will find a decomposition that is reasonable
#   for the dataset. We know from stationary_testing.py
#   that the dataset is stationary per our threshold using
#   the ADFTest, but it does seem to have some seasonality 
#   to it. Based on the associated paper, I suspect that 
#   will prove to be weekly.  
# Core Python Dependencies
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
import utils

y_train = utils.load_y_train()

print(pd.infer_freq(y_train.index))

data = y_train.to_numpy().reshape(-1)

# really doesn't seem to be a need to look at a multiplicative model...
#   found a better way to think of the freq parameter... based on 
#   ExponentialSmoothing model's documentation. 
#   https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
#   "The number of periods in a complete seasonal cycle, e.g., 4 for quarterly data or 7 for daily data with a weekly cycle."
res = seasonal_decompose(data, model='additive', freq=utils.weekly_freq)
res.plot()
plt.savefig(f'{utils.tmp_graphics_folder}{sep}dep-decompose-additive-weekly')
plt.figure()

# let's try daily
#   number of minutes in a day
mins_per_day = 60*24
res = seasonal_decompose(data, model='additive', freq=utils.daily_freq)
res.plot()
plt.savefig(f'{utils.tmp_graphics_folder}{sep}dep-decompose-additive-daily')
plt.figure()

# let's try hourly
#   number of minutes in a hour
mins_per_hour = 60
res = seasonal_decompose(data, model='additive', freq=utils.hourly_freq)
res.plot()
plt.savefig(f'{utils.tmp_graphics_folder}{sep}dep-decompose-additive-hourly')
plt.figure()

# out of curriosity... let's do a log transform and check for weekly, daily, and hourly again
#   Commented out as this led to little or no change :)
# data = np.log(data)

# res = seasonal_decompose(data, model='additive', freq=int(mins_per_week/10))
# res.plot()
# plt.savefig(f'{utils.tmp_graphics_folder}{sep}log-dep-decompose-additive-weekly')
# plt.figure()

# res = seasonal_decompose(data, model='additive', freq=int(mins_per_day/10))
# res.plot()
# plt.savefig(f'{utils.tmp_graphics_folder}{sep}log-dep-decompose-additive-daily')
# plt.figure()

# res = seasonal_decompose(data, model='additive', freq=int(mins_per_hour/10))
# res.plot()
# plt.savefig(f'{utils.tmp_graphics_folder}{sep}log-dep-decompose-additive-hourly')
# plt.figure()