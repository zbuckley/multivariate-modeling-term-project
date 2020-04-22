# This file will recreate the final models built from 
#   feature_selection_regression, and evaluate them on
#   the test dataset. 

# Core Python Dependencies
from os.path import sep # attempts to maintain OS portability

# Import Third-part Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# Import things developed by me either
#   throughout the coursework, or 
#   specifically for the term project
#   (provided in additional python files)
from utils.data import load_y_train, load_x_train, load_y_test, load_x_test
from utils.regression import LinRegModel
import utils.stats as stats
from utils.conf import tmp_graphics_folder

# Model based on filtering by F-test, then incrementally removing
#  insignificant features based on the t-tests
features1 = ['lights', 'T2', 'RH_8', 'T3', 'RH_1', 'T5']
# Model based on the papers choices (uses all the features)
features2 = [
    'NSM', 'lights', 'Press_mm_hg', 'RH_5', 'T3', 'RH_3'
]

# Let's setup StandardScaler Transforms for normalizing the data
#   we'll use the same one transform for the dependent variables
#   but we'll need two different scalers for the inputs, as
#   we have two different feature lists.
x_scaler1 = StandardScaler()
x_scaler2 = StandardScaler()
y_scaler = StandardScaler()

# Let's get the data setup
x_train = load_x_train()
y_train = load_y_train()
x_test = load_x_test()
y_test = load_y_test()

# apply feature selection
x_train1 = x_train[features1]
x_test1 = x_test[features1]
x_train2 = x_train[features2]
x_test2 = x_test[features2]

# fit scalers using only the test data
x_scaler1.fit(x_train1)
x_scaler2.fit(x_train2)
y_scaler.fit(y_train)

# apply scalers
#   decided to setup a little helper function for this
def _apply_scaler(scaler, df):
    return pd.DataFrame(
        scaler.transform(df), 
        index=df.index, 
        columns=df.columns
    )

x_train1 = _apply_scaler(
    x_scaler1,
    x_train1
)
x_train2 = _apply_scaler(
    x_scaler2,
    x_train2
)
y_train_scaled = _apply_scaler(
    y_scaler,
    y_train
)

# Setup and fit Linear Regression Models
#  in keeping with our finding during feature selection
#  we'll have no interecept values.
model1 = LinRegModel(intercept=False)
model2 = LinRegModel(intercept=False)
model1.fit(x_train1, y_train_scaled)
model2.fit(x_train2, y_train_scaled)

# Let's print the model summary results
print('MODEL 1 RESULTS (on transformed data):')
model1.print_summary()

print('MODEL 2 RESULTS (on transformed data):')
model2.print_summary()

# OK, let's transform the test data
x_test1 = _apply_scaler(
    x_scaler1,
    x_test1
)

x_test2 = _apply_scaler(
    x_scaler2,
    x_test2
)

# and get our predicted y data
#  y_pred1 and y_pred2 are numpy arrays,
#  note dataframes
y_pred1 = model1.predict(x_test1)
y_pred2 = model2.predict(x_test2)

# now we need to invert the scaler transform
y_pred1 = y_scaler.inverse_transform(y_pred1)
y_pred2 = y_scaler.inverse_transform(y_pred2)

# let's convert the y_test actual data to numpy arrays
#   column vectors for everything!!
y_test = y_test.to_numpy().reshape(-1, 1)

# evaluate models
#  another helper function
print()
print('Model 1 Test Set Regression Analysis:')
residuals1 = stats.print_metrics(y_test, y_pred1, x_test1.shape[1], x_train1.shape[0])
stats.acf_plot(residuals1, 'Model 1 Residuals', 20)
plt.savefig(f'{tmp_graphics_folder}{sep}regression-model-1-residuals-acf')
plt.figure()

print()
print('Model 2 Test Set Regression Analysis:')
residuals2 = stats.print_metrics(y_test, y_pred2, x_test2.shape[1], x_train2.shape[0])
stats.acf_plot(residuals2, 'Model 2 Residuals', 20)
plt.savefig(f'{tmp_graphics_folder}{sep}regression-model-2-residuals-acf')
plt.figure()

y_train = y_train.to_numpy().reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_actuals = np.append(y_train, y_test)
plt.plot(y_actuals)
xs = list(range(y_train.shape[0], y_actuals.shape[0]))
plt.plot(xs, y_pred1)
plt.title('Regression Model 1 Prediction vs Time')
plt.legend(['Actual', 'Predicted'])
plt.savefig(f'{tmp_graphics_folder}{sep}final-regression-model-1-actual-pred-vs-time')
plt.figure()

plt.plot(y_actuals)
plt.plot(xs, y_pred2)
plt.title('Regression Model 2 Prediction vs Time')
plt.legend(['Actual', 'Predicted'])
plt.savefig(f'{tmp_graphics_folder}{sep}final-regression-model-2-actual-pred-vs-time')
plt.figure()