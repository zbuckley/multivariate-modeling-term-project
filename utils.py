# This file provides a collection of routines and utilities used 
#   throughout the other python programs. I created it in an attempt
#   to avoid a rediculout amount of copy-pasted code in code 
#   for the term project.
# For the most part, These are slightly modified versions of the 
#   utilities we've developed throughout the course. 
# Core Python Dependencies
from pathlib import Path # used to setup folders
from os.path import sep

# Third-Party Dependencies
import numpy as np
import matplotlib.pyplot as plt

# folder config/setup
# This file is provided, but can be downloaded from 
#   the github project associated with the original 
#   research paper.
#   https://github.com/LuisM78/Appliances-energy-prediction-data/blob/master/energydata_complete.csv
data_source_file=f'data{sep}energydata_complete.csv'
tmp_graphics_folder='tmp_graphics'
tmp_data_folder='tmp_data'

# function to initialize folders as needed
def init_tmp_folders():    
    Path(tmp_graphics_folder).mkdir(parents=True, exist_ok=True)

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