# let's evaluate the chosen arma model on the test set

# This file generates the content for the Introduction
#   and performs the train-test-split
# Core Python Dependencies
from os.path import sep # attempts to maintain OS portability

# Import Third-part Libraries
import matplotlib.pyplot as plt
import numpy as np

# Import things developed by me either
#   throughout the coursework, or 
#   specifically for the term project
#   (provided in additional python files)
from utils.data import load_y_train, load_y_test
from utils.conf import tmp_graphics_folder
from utils import arma
from utils import stats

# load the data
y_test = load_y_test()
y_train = load_y_train()

# convert to column vectors
y_test = y_test.to_numpy().reshape(-1, 1)
y_train = y_train.to_numpy().reshape(-1, 1)

# setup transform
transform = arma.compose_transform(
    arma.logarithmic_transform,
    arma.normalization_transform
)

# transform data
#  NOTE: The transform is being created based on the training
#  set, as the nomalization transform saves mean and var
#  for use in the inverter.
y_train_trans, y_train_inverter = transform(y_train)

# train ARMA(4,3) model
model = arma.lm(y_train_trans, 4, 3)

# predict using last 4 values of y_train_trans as initial
#   values, let's see how close we get to the y_test data
_, y_test_pred = model.dlsim(
    num_samples=y_test.shape[0] + 4, # as first 4 values are coming from y_train
    #  provide first (ar_order + 1) values of data to base
    #    prediction on, as this is a recursive prediction 
    #    mechanism, it does need a starting point
    #    (last 4 values in training set)
    y0=y_train_trans[y_train_trans.shape[0] - 4 - 1:, 0] 
) 
# reshape, and drop vals from y_train

y_test_pred = y_test_pred.reshape(-1, 1)[4:, 0]

# invert transform
#   NOTE: again... so much faster using column vectors
y_test_pred = y_train_inverter(y_test_pred).reshape(-1, 1)

# plot results
print('INFO: y_test.shape:', y_test.shape)
y_actual = np.append(y_train, y_test)
plt.plot(y_actual)
xs = list(range(y_train.shape[0], y_actual.shape[0]))
plt.plot(xs, y_test_pred)
plt.savefig(f'{tmp_graphics_folder}{sep}final-arma-4-3-test')
plt.figure()

residuals = stats.print_metrics(y_test_pred, y_test, 7, y_train.shape[0])
stats.acf_plot(residuals, 'ARMA Model Residuals', 20)
plt.savefig(f'{tmp_graphics_folder}{sep}final-arma-4-3-test-residuals-acf')
plt.figure()

# train ARMA(1, 0) model
model = arma.lm(y_train_trans, 1, 0)

# predict using last 4 values of y_train_trans as initial
#   values, let's see how close we get to the y_test data
_, y_test_pred = model.dlsim(
    num_samples=y_test.shape[0] + 1, # as first values are coming from y_train
    #  provide first (ar_order + 1) values of data to base
    #    prediction on, as this is a recursive prediction 
    #    mechanism, it does need a starting point
    #    (last 4 values in training set)
    y0=y_train_trans[y_train_trans.shape[0] - 1:, 0] 
) 
# reshape, and drop vals from y_train
y_test_pred = y_test_pred.reshape(-1, 1)[1:, 0]

# invert transform
#   NOTE: again... so much faster using column vectors
y_test_pred = y_train_inverter(y_test_pred).reshape(-1, 1)

# plot results
print('INFO: y_test.shape:', y_test.shape)
y_actual = np.append(y_train, y_test)
plt.plot(y_actual)
xs = list(range(y_train.shape[0], y_actual.shape[0]))
plt.plot(xs, y_test_pred)
plt.savefig(f'{tmp_graphics_folder}{sep}final-arma-1-0-test')
plt.figure()

residuals = stats.print_metrics(y_test_pred, y_test, 1, y_train.shape[0])
stats.acf_plot(residuals, 'ARMA Model Residuals', 20)
plt.savefig(f'{tmp_graphics_folder}{sep}final-arma-1-0-test-residuals-acf')
plt.figure()

