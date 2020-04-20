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
from utils.stats import ttest, corr, cor_and_ttest, print_metrics
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

def fit_and_eval(model, x_scaled, y_scaled, y, scaler):
    # fit model
    model.fit(x_scaled, y_scaled)
    
    # get y_pred for residual computes
    y_pred_scaled = model.predict(x_scaled)
    y_pred = scaler.inverse_transform(y_pred_scaled)

    # need y_np for print_metrics helper
    #   reshape into column vector
    y_np = y.to_numpy().reshape(-1, 1)

    # determine number of params
    num_params = x_scaled.shape[1]
    if model.intercept:
        num_params += 1

    # print coefficients
    print()
    print(model.coeffTable())
    
    # print metrics
    print()
    print_metrics(y_pred, y_np, num_params, x_scaled.shape[0])

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

# Let's perform column-wise normalization on the features and y. 
x_scaler = StandardScaler()
x_scaler.fit(x_train)
x_train_scaled = x_scaler.transform(x_train)
print('DEBUG: x_train_scaled', type(x_train_scaled), x_train_scaled.shape)
x_train_scaled = pd.DataFrame(data=x_train_scaled, index=x_train.index, columns=x_train.columns)

y_scaler = StandardScaler()
y_scaler.fit(y_train)
y_train_scaled = y_scaler.transform(y_train)
print('DEBUG: y_train_scaled', type(y_train_scaled), y_train_scaled.shape)
y_train_scaled = pd.DataFrame(data=y_train_scaled, index=y_train.index, columns=y_train.columns)

# Let's look at the F-Test scores
#   using f_regression function provided by sklearn to 
#   run the F-Test on each of features.
f_res = f_regression(x_train_scaled, y_train_scaled)
ftests = zip(x_train.columns, f_res[0], f_res[1])

# feature selection, sort by p-value
ftests_sorted = sorted(ftests, key=lambda x: x[2])

print('features ordered by ftest significance')
for (col, ftest, pvalue) in ftests_sorted:
    print(f'\t{ftest}\t{pvalue}\t{col}')

# F-test null hypothesis states that the model with no 
#   independent variables fits the data as well as your model
# Alternative Hypothesis, is that your model fit's the data 
#   better than a model with no independent variables.
print('features to drop:')
drop_list = []
for (col, ftest, pvalue) in ftests_sorted:
    if (pvalue > 0.05): # 95% confidence
        print(f'\t{ftest}\t{pvalue}\t{col}')
        drop_list.append(col)

# Whats left are features which give us a 95% 
#  confidence of them improving on the fit from 
#  model with no independent variables
x_train_scaled = x_train_scaled.drop(drop_list, axis=1)
print('remaining features:', x_train_scaled.columns)

# Now, let's use corrplot, and the t-test to 
#   ensure the we have no multicollinearity.
corr_table = x_train_scaled.corr(
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
ttest_table = x_train_scaled.corr(
    method= lambda x, y: cor_and_ttest(x, y)[2]
)

# So... this actually works... which is really cool :)
#   Let's tailor the heatmap color scale a bit
sns.heatmap(
    ttest_table,
    vmin=0.05, vmax=1
)
plt.tight_layout()
plt.savefig(f'{tmp_graphics_folder}{sep}ttestplot-ftest-filtered')
plt.figure()

# Based on that it seems reasonable to say that pretty much
#   all the correlation coefficients being generated
#   are statistically significant.  
# In hindsight... based on what the ttest of a correlation
#   coefficient should actually be asking... this totally
#   makes sense given the number of samples we have.
# So, going back to the original corrplot, let's worry 
#   more about the correlation magnitude

# Redo F-test for new ordered list of remaining features
f_res = f_regression(x_train_scaled, y_train_scaled)
ftests = zip(x_train_scaled.columns, f_res[0], f_res[1])

# feature selection, sort by p-value
ftests_sorted = sorted(ftests, key=lambda x: x[2])

# only need the features in order
ftests_sorted = [col for col, _, _ in ftests_sorted]

print('remaining features in order:', ftests_sorted)

# let's decide on a correlation threshold for these... 
#   to do that I'd like a plot of threhold vs number of features in the model. 
# we'll need a function for grabbing the features based on the threshold... 
thresholds = np.linspace(0, 1.0, num=100)
feature_counts = [len(filter_ordered_features(x_train_scaled, ftests_sorted, threshold)) for threshold in thresholds]
plt.plot(thresholds, feature_counts)
plt.title('Feature Count vs. Threshold')
plt.xlabel('Threshold')
plt.ylabel('Feature Count')
plt.savefig(f'{tmp_graphics_folder}{sep}feature-count-vs-threshold')
plt.figure()

# Based on the chart, I'm thinking we start with a threshold around .8
#   correlation between features, to avoid the worst multicollinearity
#   offenders amongst the features. We can then incrementally drop
#   features that aren't highly signficant.
features1 = filter_ordered_features(x_train_scaled, ftests_sorted, 0.8)
print('attempting feature-set:', features1)
model = LinRegModel()
x_train1 = x_train_scaled[features1]
print('DEBUG: x_train1.shape', x_train1.shape)

fit_and_eval(model, x_train1, y_train_scaled, y_train, y_scaler)

# Let's remove the intercept as, it's very insignificant.
model = LinRegModel(intercept=False)
print('removing intercept:')

fit_and_eval(model, x_train1, y_train_scaled, y_train, y_scaler)

# remove T6
x_train1 = x_train1.drop('T6', axis=1)
print('removed T6:')

fit_and_eval(model, x_train1, y_train_scaled, y_train, y_scaler)

# remove Windspeed
x_train1 = x_train1.drop('Windspeed', axis=1)
print('removed Windspeed')

fit_and_eval(model, x_train1, y_train_scaled, y_train, y_scaler)

# remove Press_mm_hg
x_train1 = x_train1.drop('Press_mm_hg', axis=1)
print('removed Press_mm_hg')

fit_and_eval(model, x_train1, y_train_scaled, y_train, y_scaler)

# remove T8
x_train1 = x_train1.drop('T8', axis=1)

fit_and_eval(model, x_train1, y_train_scaled, y_train, y_scaler)

# remove RH_6
x_train1 = x_train1.drop('RH_6', axis=1)

fit_and_eval(model, x_train1, y_train_scaled, y_train, y_scaler)

# remove RH_out
x_train1 = x_train1.drop('RH_out', axis=1)

fit_and_eval(model, x_train1, y_train_scaled, y_train, y_scaler)

# remove NSM
x_train1 = x_train1.drop('NSM', axis=1)

fit_and_eval(model, x_train1, y_train_scaled, y_train, y_scaler)

# remove T4
x_train1 = x_train1.drop('T4', axis=1)

fit_and_eval(model, x_train1, y_train_scaled, y_train, y_scaler)

# We'll utilize this model, but out of curiosity, the published 
#  article/paper associated with the dataset, used a recursive
#  feature elimination (RFE) algorithm provided by the caret
#  package in R. In python, A similar algorithm, is available 
#  from the sklearn package. 
# Let's use the features identified in the paper, as those to use
#  for a linear regression model, and see if our outcome matches.
# NOTE: I am missing 2 engineered features
#   'WeekStatus', and 'Day of Week' as those are both categorical.
#   And my linear regression model, isn't setup to handle it.
# NOTE: This list is all the features, except rv1, and rv2
features_paper=[
    'lights', 'T1', 'RH_1', 'T2', 'RH_2',
    'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5',
    'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8',
    'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out',
    'Windspeed', 'Tdewpoint', 'NSM'
]

x_train_paper = x_train[features_paper]

x_scaler_paper = StandardScaler()
x_scaler_paper.fit(x_train_paper)
x_tmp = x_scaler_paper.transform(x_train_paper)
print('DEBUG: x_train_scaled', type(x_train_paper), x_train_paper.shape)
x_train_paper = pd.DataFrame(data=x_tmp, index=x_train_paper.index, columns=x_train_paper.columns)

model_paper = LinRegModel(intercept=False)

fit_and_eval(model_paper, x_train_paper, y_train_scaled, y_train, y_scaler)

# This behaves similarly to to the way described in paper
#   I intend to evaluate both models against the test set.
