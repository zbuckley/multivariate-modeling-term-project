# This file generates the content for the Introduction
#   and performs the train-test-split
# Core Python Dependencies
from os.path import sep # attempts to maintain OS portability

# Import Third-part Libraries 
import pandas as pd
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


# Let's drop things that really shouldn't be used to predict
#  or forecase Application energy use. 
#    lights, is another variable we could predict
df = df.drop(['lights', 'rv1', 'rv2'], axis=1)

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

# This takes a long time to generate, so we'll comment it out for submission
#   but it does give a very useful/pretty graphic for analyzing relationships
#   between variables.
# sns.pairplot(df, height=10)
# plt.savefig(f'{conf.tmp_graphics_folder}{sep}large-sns-pairplot-all-data')
# plt.figure()

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

