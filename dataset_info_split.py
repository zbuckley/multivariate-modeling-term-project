# Exploratory Data Analysis
#   This file generates the content for the Introduction
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
import utils

# init_folders takes care of setting up folders for organizing outputs
utils.init_tmp_folders()

# Need a custom time parser, as the format isn't well-behaved by default.
mydateparser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
df = pd.read_csv(utils.data_source_file, parse_dates=['date'], date_parser=mydateparser)
df = df.set_index('date')

# Basic Info 
print('Shape of Dataset:', df.shape)


# Let's drop things that really shouldn't be used to predict
#  or forecase Application energy use. 
#    lights, is another variable we could predict
df = df.drop(['lights', 'rv1', 'rv2'], axis=1)

# a - plot the dependent variable vs. time
df['Appliances'].plot()
plt.savefig(f'{utils.tmp_graphics_folder}{sep}dep-vs-time')
plt.figure()

# b - ACF
utils.acf_plot(df['Appliances'].to_numpy(), 'Appliances', 50)
plt.savefig(f'{utils.tmp_graphics_folder}{sep}dep-acf-50-lag')
plt.figure()

utils.acf_plot(df['Appliances'].to_numpy(), 'Appliances', 2000)
plt.savefig(f'{utils.tmp_graphics_folder}{sep}def-acf-2000-lag')
plt.figure()

# c - correlation matrix
# df.corr takes an optional callable, so let's use 
#   the implementation of pearson's r that we developed
#   previously in the course
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
corr_table = df.corr(
    method=utils.corr
)

# locked color scale bounds, based on theoretical min and max
# https://seaborn.pydata.org/generated/seaborn.heatmap.html
sns.heatmap(
    corr_table,
    vmin=-1, vmax=1
)
plt.tight_layout() # fixes issue with image bounds cutting off labels
plt.savefig(f'{utils.tmp_graphics_folder}{sep}corrplot-all-beforesplit')
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
    split.to_csv(f'{utils.tmp_data_folder}{sep}{label}.csv', header=True)

