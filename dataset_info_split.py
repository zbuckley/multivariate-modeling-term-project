# This file generates the content for the Introduction
#   and performs the train-test-split
# Core Python Dependencies
from os.path import sep # attempts to maintain OS portability
from datetime import datetime

# Import Third-part Libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import things developed by me either
#   throughout the coursework, or 
#   specifically for the term project
#   (provided in additional python files)
from utils import conf, data, stats, visualizations as viz

# init_folders takes care of setting up folders for organizing outputs
conf.init_tmp_folders()

# Load dataset, and deal with parsing data information
df = data.load_original_data()

# Basic Info 
print('Shape of Dataset:', df.shape)

# Adding NSM Feature as described in the paper
#  Number of seconds since midnight
#  https://stackoverflow.com/questions/15971308/get-seconds-since-midnight-in-python
#  https://www.w3resource.com/python-exercises/date-time-exercise/python-date-time-exercise-8.php
#  https://stackoverflow.com/questions/3743222/how-do-i-convert-datetime-to-date-in-python
def NSM(dt):
    return (dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()

# vectorize the function
# NSM = np.vectorize(NSM_simple)
dts = pd.to_datetime(df.index)
print(type(dts[0]), dts[0])
df['NSM'] = [NSM(dt) for dt in pd.to_datetime(df.index).tolist()]
print(df['NSM'].head())


# This takes a long time to generate, so we'll comment it out for submission
#   but it does give a very useful/pretty graphic for analyzing relationships
#   between variables.
sns.pairplot(df, height=3)
plt.savefig(f'{conf.tmp_graphics_folder}{sep}large-sns-pairplot-all-data')
plt.figure()

# Let's drop things that really shouldn't be used to predict
#  or forecast Appliance energy use.
df = df.drop(['rv1', 'rv2'], axis=1)

# a - plot the dependent variable vs. time
df['Appliances'].plot()
plt.savefig(f'{conf.tmp_graphics_folder}{sep}dep-vs-time')
plt.figure()

# b - ACF
viz.acf_plot(df['Appliances'].to_numpy(), 'Appliances', 50)
plt.savefig(f'{conf.tmp_graphics_folder}{sep}dep-acf-50-lag')
plt.figure()

viz.acf_plot(df['Appliances'].to_numpy(), 'Appliances', 2000)
plt.savefig(f'{conf.tmp_graphics_folder}{sep}def-acf-2000-lag')
plt.figure()

# c - correlation matrix
# df.corr takes an optional callable, so let's use 
#   the implementation of pearson's r that we developed
#   previously in the course
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
corr_table = df.corr(
    method=stats.corr
)

# locked color scale bounds, based on theoretical min and max
# https://seaborn.pydata.org/generated/seaborn.heatmap.html
sns.heatmap(
    corr_table,
    vmin=-1, vmax=1
)
plt.tight_layout() # fixes issue with image bounds cutting off labels
plt.savefig(f'{conf.tmp_graphics_folder}{sep}corrplot-all-beforesplit')
plt.figure()


# d - cleaning procedures
print('Info:', df.info())

# based on this info, there are no missing values to account for

# e - split training and testing sets
y = df['Appliances']
x = df.drop(['Appliances'], axis=1)
split_labels = ['x_train', 'x_test', 'y_train', 'y_test']
splits = dict(zip(split_labels, train_test_split(
    x, 
    y, 
    shuffle=False, 
    test_size=0.33
)))

# f - makes sense... let's check sizes of train and test split
# Let's save it all to the data folder
for label, split in splits.items():
    print(f'{label}:', split.shape, type(split))
    split.to_csv(f'{conf.tmp_data_folder}{sep}{label}.csv', header=True)

