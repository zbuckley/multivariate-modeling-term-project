# This file is responsible for performing feature selection

# Core Python Dependencies
from os.path import sep # attempts to maintain OS portability

# Import Third-part Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler

# Import things developed by me either
#   throughout the coursework, or 
#   specifically for the term project
#   (provided in additional python files)
from utils.data import load_y_train, load_x_train
from utils.conf import tmp_graphics_folder
from utils.stats import ttest, corr, cor_and_ttest
from utils.regression import LinRegModel

# function for filtering through through best ftest scored features
#   based on a maximum correlation threshold against features 
#   that have already been accepted 
def filter_ordered_features(x, ordered_features, corr_threshold):
    # we'll need a correlation table for look-up purposes
    feature_corr_table = x.corr().abs()
    #print(feature_corr_table)

    selected_features = []
    deselected_features = []
    for i in range(len(ordered_features)):
        # select feature[i], if not in deselected list
        if i not in deselected_features:
            selected_features.append(i)
        
        if i < len(ordered_features) - 1:
            for j in range(i + 1, len(ordered_features)):
                if feature_corr_table[ordered_features[j]][ordered_features[i]] > corr_threshold and j not in deselected_features:
                    deselected_features.append(j)

    # print(selected_features)
    selected_features = [ordered_features[k] for k in selected_features]
    # print(selected_features)

    return selected_features

# Let's start with a corrplot, similar to that which we 
#   generated before, but only on the training datasets now. 
x_train = load_x_train()
y_train = load_y_train()

# let's get a combined train dataframe
train = x_train.copy()
train['Appliances'] = y_train['Appliances']

# df.corr takes an optional callable, so let's use 
#   the implementation of pearson's r that we developed
#   previously in the course
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
corr_table = train.corr(
    method=corr
)

# locked color scale bounds, based on theoretical min and max
# https://seaborn.pydata.org/generated/seaborn.heatmap.html
sns.heatmap(
    corr_table,
    vmin=-1, vmax=1
)
plt.tight_layout() # fixes issue with image bounds cutting off labels
plt.savefig(f'{tmp_graphics_folder}{sep}corrplot-train')
plt.figure()

# let's take the abs of the corr_table and do it again
#   This should make it more obvious which features are 
#   correlated with the target variable
sns.heatmap(
    np.abs(corr_table),
    vmin=0, vmax=1
)
plt.tight_layout()
plt.savefig(f'{tmp_graphics_folder}{sep}corrplot-abs-train')
plt.figure()

# Let's look at the F-Test scores
#   using f_regression function provided by sklearn to 
#   run the F-Test on each of features.
f_res = f_regression(x_train, y_train)
ftests = zip(x_train.columns, f_res[0], f_res[1])

# feature selection, sort by p-value
ftests_sorted = sorted(ftests, key=lambda x: x[2])

# F-test null hypothesis states that the model with no 
#   independent variables fits the data as well as your model
# Alternative Hypothesis, is that your model fit's the data 
#   better than a model with no independent variables.
print('features to drop:')
drop_list = []
for (col, ftest, pvalue) in ftests_sorted:
    if (pvalue > 0.01): # 99% confidence
        print(f'\t{ftest}\t{pvalue}\t{col}')
        drop_list.append(col)

# Whats left are features which give us a 99% 
#  confidence of them improving on the fit from 
#  model with no independent variables
x_train = x_train.drop(drop_list, axis=1)

# Now, let's use corrplot, and the t-test to 
#   ensure the we have no multicollinearity.
corr_table = x_train.corr(
    method=corr
)

sns.heatmap(
    corr_table,
    vmin=-1, vmax=1
)
plt.tight_layout()
plt.savefig(f'{tmp_graphics_folder}{sep}corrplot-ftest-filtered')
plt.figure()

# Let's use the t-test and correlation to determine
#   which regressands are significantly related
ttest_table = x_train.corr(
    method= lambda x, y: cor_and_ttest(x, y)[2]
)

# So... this actually works... which is really cool :)
#   Let's tailor the heatmap color scale a bit
#   to help us figure out which features can't co-exist
#   in our feature-set.
sns.heatmap(
    ttest_table,
    vmin=0.05, vmax=1
)
plt.tight_layout()
plt.savefig('ttestplot-ftest-filtered')
plt.figure()

# Based on that it seems reasonable to say that pretty much
#   all the correlation coefficients being generated
#   are statistically significant. The only exceptions
#   are (T1, RH_9), (T2, RH_2), and (RH_6, Press_mm_hg)
#   In hindsight... based on what the ttest of a correlation
#   coefficient should actually be asking... this totally
#   makes sense with the number of samples involved.
# So, going back to the original corrplot, let's worry 
#   more about the correlation magnitude
# Let's look at seaborn's pairplot
#   Cool, but not very useful... and fairly expensive
#   to run. (took a while...)
# sns.pairplot(x_train)
# plt.savefig('pairplot-ftest-filtered')

# Redo F-test for new ordered list of most predictive variables
f_res = f_regression(x_train, y_train)
ftests = zip(x_train.columns, f_res[0], f_res[1])

# feature selection, sort by p-value
ftests_sorted = sorted(ftests, key=lambda x: x[2])

# only need the features in order
ftests_sorted = [col for col, _, _ in ftests_sorted]

# let's decide on a correlation threshold for these... 
#   to do that I'd like a plot of threhold vs number of features in the model. 
# we'll need a function for grabbing the features based on the threshold... 
thresholds = np.linspace(0, 1.0, num=100)
feature_counts = [len(filter_ordered_features(x_train, ftests_sorted, threshold)) for threshold in thresholds]
plt.plot(thresholds, feature_counts)
plt.title('Feature Count vs. Threshold')
plt.xlabel('Threshold')
plt.ylabel('Feature Count')
plt.savefig(f'{tmp_graphics_folder}{sep}feature-count-vs-threshold')
plt.figure()

# Based on the chart, I'm thinking we start with a threshold around .6 or .7
#   and incrementally pair down the number of features by removing the 
#   features whose coefficients are least significant. If all of the features
#   are significant, we may up the threshold again.
features1 = filter_ordered_features(x_train, ftests_sorted, 0.65)
print('attempting feature-set:', features1)
model = LinRegModel()
x_train1 = x_train[features1]
print('DEBUG: x_train1.shape', x_train1.shape)

model.fit(x_train1, y_train)
# print coefficients and ttest results
model.print_summary()

# OK, so, let's begin attempting to improve the model. 
#  Let's remove, all the parameters which aren't in the ball-park. 
#  specfically the RH_out, T6, and Press_mm-hg as they have
#  p-values that are greater than 0.1. 
features2 = ['T2', 'RH_8', 'T3', 'Windspeed']
x_train2 = x_train[features2]
print('DEBUG: x_train2.shape', x_train2.shape)
# model = LinRegModel(intercept=False)
model.fit(x_train2, y_train)
model.print_summary()

# Let's attempt normalizing the input and target data to see if that helps
#   i could see that helping the intercept term, so let's give it a go.
# NOTE: cannot compare RMSE/SSE directly with models above from here out
#   as adding the transform adjusts the 'baselines' for these values
#   R2 and Adjusted R2, should still be valid comparitors.
x_scaler3 = StandardScaler()
x_scaler3.fit(x_train2)
x_train3 = x_scaler3.transform(x_train2)
print('DEBUG: x_train3', type(x_train3), x_train3.shape)
x_train3 = pd.DataFrame(data=x_train3, index=x_train2.index, columns=x_train2.columns)

y_scaler = StandardScaler()
y_scaler.fit(y_train)
y_train3 = y_scaler.transform(y_train)
print('DEBUG: y_train3', type(y_train3), y_train3.shape)
y_train3 = pd.DataFrame(data=y_train3, index=y_train.index, columns=y_train.columns)

model = LinRegModel()
model.fit(x_train3, y_train3)
model.print_summary()

# not too much change, very certain we don't need intercept at this point. 
model3 = LinRegModel(intercept=False)
model3.fit(x_train3, y_train3)
model3.print_summary()

# so removing intercept in the normalized model based on 
#  T2, RH_8, T3, and Windspeed lead gave a VERY small 
#  increase in Adjusted R2. 
# As T3 is only significant with a t-test p-value of 0.02 (98% confidence)
#  let's see if the model improves any removing that feature.
x_train4 = x_train[['T2', 'RH_8', 'Windspeed']]
x_scaler4 = StandardScaler()
x_scaler4.fit(x_train4)
tmp = x_scaler4.transform(x_train4)
print('DEBUG: x_train4', type(x_train4), x_train4.shape)
x_train4 = pd.DataFrame(data=tmp, index=x_train4.index, columns=x_train4.columns)

# # this is actually redundant... so we'll just use y_train3
# y_scaler = StandardScaler()
# y_scaler.fit(y_train)
# y_train4 = y_scaler.transform(y_train)
# print('DEBUG: y_train4', type(y_train4), y_train4.shape)
# y_train4 = pd.DataFrame(data=y_train4, index=y_train.index, columns=y_train.columns)

model4 = LinRegModel()
model.fit(x_train4, y_train3)
model.print_summary()

# alas, that yielded a small decrease in R2 (explained variance)
#   but we do have statistically significant variables across the board
#   this may be as far as we can go using a normal regression model on this dataset

# Let's apply the last 2 models, to the test dataset, we'll do that in
#   a sepparate python file for clarity.