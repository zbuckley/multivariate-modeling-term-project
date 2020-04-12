# This file dedicated to performing stationarity testing
#   and if necessary developing transforms needed for 
#   getting stationary behaviour from the dataset
# For consistency, we'll perform this testing against the 
#   training set build in dataset_info_split.py
# Core Python Dependencies
from os.path import sep # attempts to maintain OS portability

# Import Third-part Libraries 
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Import things developed by me either
#   throughout the coursework, or 
#   specifically for the term project
#   (provided in additional python files)
import utils
import corr_utils

# Testing for stationary with the dependent variable
#   training data only
y_train = utils.load_y_train()

# plt vs time (just training set)
y_train.plot()
plt.savefig(f'{utils.tmp_graphics_folder}{sep}dep-train-vs-time')
plt.figure()

# histogram
y_train.hist(bins=100)
plt.savefig(f'{utils.tmp_graphics_folder}{sep}dep-train-hist')
plt.figure()

# acf plot
corr_utils.acf_plot(y_train.to_numpy(), 'dep-train', 50)
plt.savefig(f'{utils.tmp_graphics_folder}{sep}dep-train-acf-50')
plt.figure()

corr_utils.acf_plot(y_train.to_numpy(), 'dep-train', 200)
plt.savefig(f'{utils.tmp_graphics_folder}{sep}dep-train-acf-200')
plt.figure()

# adf test
print('ADF Test p-value for dependent training data:', adfuller(y_train.to_numpy().reshape(-1))[1])