# This file responsible for generating GPAC code
#   and exploring various ARMA models for consideration

# This file generates the content for the Introduction
#   and performs the train-test-split
# Core Python Dependencies
from os.path import sep # attempts to maintain OS portability

# Import Third-part Libraries
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Import things developed by me either
#   throughout the coursework, or 
#   specifically for the term project
#   (provided in additional python files)
from utils.data import load_y_train
from utils.gpac import gpac
import utils.visualizations as viz
from utils.conf import tmp_graphics_folder
from utils import arma
from utils import stats

# load x and y train datasets
y_train = load_y_train()

# setup our numpy array/matrix content
#  need dependent variable as column vector
y = y_train.to_numpy().reshape(-1, 1)

# we'll arbitrarily start with 8 for now.
table = gpac(
    y,
    8, 8
)

# and plot the table
viz.plot_GPAC(table, 'y-train')
plt.savefig(f'{tmp_graphics_folder}{sep}gpac-y-train')
plt.figure()

# acf of training sig
viz.acf_plot(y, 'y-train', 20)
plt.savefig(f'{tmp_graphics_folder}{sep}acf-20-y-train')

# I have my doubts, but the most obvious ARMA Identification from
#   the GPAC appears to be an ARMA(1, 0) model.
#   Let's give that a try.
# Failed to converge on initial attempt, let's bump µ_max by 5
#   orders of magnitude and try again.
# Still failing to convert, let's attempt shrining the gradient/hessian
#   approximation step size
# WOW.. ok... uping µ_max by another 5 orders of magnitude.
# Still not converging.. let's increase convergence threshold
# We've gone to far... Let's increase the number of iterations
#   and reset to all the defaults on everything else.
# Luckily the iterations are happening rather quickly.
#   but i'm betting the signal is simply NOT a well identified ARMA model,
#   if we can't achieve convergence within 200 iterations.
# Actually looking at model results, and continuing to experiment... I 
#   believe our criteria for convergence is simply to small
# Yep, LMA converges in 1 iteration if ε is 1, and doesn't otherwise. 
#   I suspect that the mean/variance of the errors is high enough it 
#   can't converg to the same delta of SSE we'd expected for prev. 
#   datasets covered in class.
print()
print('ARMA(1,0):')
res = arma.lm(y, 1, 0, ε = 1, verbose=False)
res.summary()

# Since it'll converge with less restrictions... 
#   is it useful. Actuall... not bad based on this graph.
res.plot_pred_vs_true('arma-1-0-train', y)
plt.savefig(f'{tmp_graphics_folder}{sep}arma-1-0-train')
plt.figure()

# So, despite it failing to converge, we did have an interesting outcome. 
#  a_1 has p-value of 0.. i'm thinking that may have something to do with
#  the large number of samples.
# Let's go back to the GPAC table, and see if a more complicated model
#  can be identified if we lower our expectation a bit. 
# From the GPAC, I'm interesting in trying ARMA(1, 1), ARMA(1, 2), ARMA(2, 1), and an ARMA(2, 2)
print()
print('ARMA(1,1):')

res = arma.lm(y, 1, 1, verbose=False)
res.summary()
res.plot_pred_vs_true('arma-1-1-train', y)
plt.savefig(f'{tmp_graphics_folder}{sep}arma-1-1-train')
plt.figure()

print()
print('ARMA(1,2):')

res = arma.lm(y, 1, 2, verbose=False)
res.summary()
res.plot_pred_vs_true('arma-1-2-train', y)
plt.savefig(f'{tmp_graphics_folder}{sep}arma-1-2-train')
plt.figure()

print()
print('ARMA(2,1):')

res = arma.lm(y, 2, 1, verbose=False)
res.summary()
res.plot_pred_vs_true('arma-2-1-train', y)
plt.savefig(f'{tmp_graphics_folder}{sep}arma-2-1-train')
plt.figure()

print()
print('ARMA(2,2):')

res = arma.lm(y, 2, 2, verbose=False)
res.summary()
res.plot_pred_vs_true('arma-2-2-train', y)
plt.savefig(f'{tmp_graphics_folder}{sep}arma-2-2-train')
plt.figure()

# OK.. so highly significant parameters for every model.. 
#  I'm thinking we'd get massively better results with something 
#  similar to the normalization or logarithm transform we've used in the 
#  labs and homeworks before. 
# Higher MA orders appears to add cyclic behavior we don't want. 
# AR seems more useful, BUT, isn't doing a good job of capturing 
#  the skewed nature of the signal. I'm thinking to attempt 
#  some transforms, as we did with homework9.
# log and norm seem fairly resonably behaved
# I attempted to take advantage of seasonality (daily) that
#   we identified in the decomp using the lagged difference transform
#   but did not have much luck. issue is the lagged difference inverse
#   manages to 
y_trans, inverter = arma.compose_transform(
    arma.logarithmic_transform,
    arma.normalization_transform,
)(y)
print('DEBUG: y_trans.shape', y_trans.shape)

# But.. What did that do, let's do some quick plots
#   and check we're still stationary
plt.plot(y_trans)
plt.title('y training after Transform')
plt.xlabel('Time Steps')
plt.ylabel('Tranformed y')
plt.savefig(f'{tmp_graphics_folder}{sep}y-training-trans-vs-time')
plt.figure()

plt.hist(y_trans)
plt.title('y training hist after Transform')
plt.savefig(f'{tmp_graphics_folder}{sep}y-training-trans-hist')
plt.figure()

print('ADFTest for y transform:', adfuller(y_trans.reshape(-1)))

viz.acf_plot(y_trans, 'y-train-trans', 20)
plt.savefig(f'{tmp_graphics_folder}{sep}y-train-trans-acf')
plt.figure()

trans_table = gpac(
    y_trans, 8, 8
)

viz.plot_GPAC(trans_table, 'y-train-trans')
plt.savefig(f'{tmp_graphics_folder}{sep}y-train-trans-gpac')
plt.figure()

# Cool, based on the new GPAC
print()
ar_order = 4
ma_order = 3
print(f'ARMA({ar_order},{ma_order}):')

res = arma.lm(y_trans, ar_order, ma_order, verbose=False)
res.summary()
res.plot_pred_vs_true(f'arma-{ar_order}-{ma_order}-train-trans', y_trans)
plt.savefig(f'{tmp_graphics_folder}{sep}arma-{ar_order}-{ma_order}-train-trans')
plt.figure()

# undo transform and check RMSE (against train data)
_, y_pred = res.dlsim(
    num_samples=y_trans.shape[0],
    #  provide first (ar_order + 1) values of data to base
    #    prediction on, as this is a recursive prediction 
    #    mechanism, it does need a starting point
    y0=y_trans[:ar_order+1] 
)
# making this a column vector makes print_metrics A LOT faster
y_pred = y_pred.reshape(-1, 1)
y_pred = inverter(y_pred)

plt.plot(y)
plt.plot(y_pred)
plt.legend(['Actual', 'Predicted'])
plt.savefig(f'{tmp_graphics_folder}{sep}arma-{ar_order}-{ma_order}-actual-pred-vs-time')
plt.figure()

stats.print_metrics(y_pred, y, ar_order+ma_order, y.shape[0])

