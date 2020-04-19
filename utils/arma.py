# This file defines our arma models and utility functions

# Core Python Imports
from os.path import sep # attempts to maintain OS portability

# Import Third-part Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import distributions

# Import things developed by me either
#   throughout the coursework, or 
#   specifically for the term project
#   (provided in additional python files)
from . import stats

# dlsim function mimicing api for statsmodel dlsim, 
#   except that we always assume 1
#   TODO: make this more column vector based?
def dlsim(sys, e, y0=None):
    # drop first entry from sys[1] to get a, give it dimensions
    a = np.array(sys[1])[1:].reshape(1, -1)
    # sys[0] is b
    b = np.array(sys[0]).reshape(1, -1)

    # wrapper for getting initialization right with coef_vec usage
    def coef_vec(v, i):
        if i < 0:
            return np.empty((0, v.shape[1]))
        if  i < v.shape[1]:
            return v[:, :i + 1]
        else:
            return v

    # wrapper for getting initialization right with e and y "vals" vectors
    def vals_vec(vals, i, order):
        if i < 0:
            return np.empty((0, 1))
        if i < order:
            return vals[i::-1]
        else:
            return vals[i:i - order:-1]

    e = np.array(e).reshape(-1, 1)

    # normal case
    if y0 is None:
        y = np.zeros(e.shape)
        i_start = 0
    else:
        # when starting y values have been provided.
        y0 = y0.reshape(-1, 1)
        y = np.append(y0, np.zeros((e.shape[0] - y0.shape[0], 1)), axis=0)
        print('DEBUG: y shape', y.shape)
        i_start = y0.shape[0]     

    for i in range(i_start, e.shape[0]):
        # get a and b, as function of i
        b_temp = coef_vec(b, i)
        a_temp = coef_vec(a, i - 1)

        # get e and y, as function of i
        e_temp = vals_vec(e, i, b.shape[1])
        y_temp = vals_vec(y, i - 1, a.shape[1])

        assert(e_temp.shape[0] > 0)

        if y_temp.shape[0] > 0:
            ar_comp = np.dot(
                a_temp,
                y_temp
            )
            ma_comp = np.dot(
                b_temp,
                e_temp
            )

            y[i, 0] = ma_comp - ar_comp
        else:
            y[i, 0] = np.dot(
                b_temp,
                e_temp
            )

    return np.array(range(1, e.shape[0])), y.reshape(1, -1)

# generate normally distributed white noise
#   mu is desired mean
#   sigma2 is desired variance
#   samples is number of samples
#   random_seed (optional) is the random seed to use. 
def generate_e(mu, sigma2, samples, random_seed=42):
    np.random.seed(random_seed)
    return np.random.randn(samples)*np.sqrt(sigma2) + mu


# Added an LMAResult Class (inspired by statsmodel/scipy/sklearn apis)
#  to simplify model result aggregation/interpretation
class LMAResult:
    def __init__(self, ar_coefficients, ma_coefficients, 
                 cov_θ, var_e, sse_accum, num_samples, e):
        self.ar_coefficients = ar_coefficients
        self.ma_coefficients = ma_coefficients
        self.cov_θ = cov_θ
        self.var_e = var_e
        self.sse_accum = sse_accum
        self.num_samples = num_samples
        self.e = e
        self.ar_order = len(self.ar_coefficients[1:])
        self.ma_order = len(self.ma_coefficients[1:])

    # calulcated ci, based on slides, and web resources
    #   https://kite.com/python/examples/702/scipy-compute-a-confidence-interval-from-a-dataset
    #   https://stats.stackexchange.com/questions/241449/matrix-and-regression-model
    def _ci_calc(self, val, se, confidence):
        h = distributions.norm.ppf((1 - (1-confidence)/2))
        low = val - h*se
        high = val + h*se
        return low, high

    # prints header for Table
    def _printCIHeader(self):
        print('\tlabel\tcoeff\tlower\tupper\tt-test\tp-value')

    # prints coefficients and their low/high intervals
    def _printCIs(self, label, coefs, ses, confidence):
        # verify sanity
        assert(len(coefs) == len(ses))
        
        # build labels list
        labels = [f'{label}_{i}' for i in range(1, len(coefs) + 1)]
        
        # print line per coef/CI
        for label, coef, se in zip(labels, coefs, ses):
            low, high = self._ci_calc(coef, se, confidence=confidence)
            t = stats.ttest_1sample(coef, se * np.sqrt(self.num_samples), self.num_samples)
            p = stats.ttest_pvalue(t, self.num_samples - self.cov_θ.shape[0])
            print('\t%s\t%.4g\t%.4g\t%.4g\t%.4g\t%.4g' % (label, coef, low, high, t, p))
    
    # prints table of coefficients with Confidence Intervals
    def printCoeffsTable(self, confidence=0.95):
        # all standard errors
        ses = np.sqrt(
            self.cov_θ[np.diag_indices_from(self.cov_θ)]
        )

        self._printCIHeader()
        
        # print('ar_coefs and confidence intervals:')
        self._printCIs(
            label='a', 
            coefs=self.ar_coefficients[1:], 
            ses=ses[:self.ar_order],
            confidence=confidence
        )
        
        # print('ma_coefs and confidence intervals:')
        self._printCIs(
            label='b',
            coefs=self.ma_coefficients[1:],
            ses=ses[self.ar_order:],
            confidence=confidence
        )

    # returns estimated ARMA in transform function form 
    #   can be directly passed into dlsim as system tuple
    def _system(self):
        def fix_coeffs(coeffs, order):
            return np.append(coeffs, np.zeros(order + 1 - len(coeffs)))
        
        max_order = max(self.ar_order, self.ma_order)

        return (
            fix_coeffs(self.ma_coefficients, max_order),
            fix_coeffs(self.ar_coefficients, max_order), 
            1
        )

    # returns generated noise based on summary stats from error
    #   collected by LMAResult on creation
    def _generate_e(self, num_samples):
        return generate_e(
            np.mean(self.e),
            np.var(self.e),
            num_samples
        )
    
    # returns simulated ARMA based on result coefficients
    #   and error characterists
    def dlsim(self, num_samples, y0=None):
        return dlsim(
            self._system(),
            self._generate_e(num_samples),
            y0=y0
        )

    # helper for taking care of 2 through 4
    def summary(self):
        # 1-ish
        # print('predicted transfer function (as dlsim system):')
        # print(self.system())

        # 2 - confidence intervals
        print('ARMA coefficients:')
        self.printCoeffsTable()

        # 3 - cov matrix
        # updated to pretty-print using numpy
        coef_labels = [f'a_{i}' for i in range(1, self.ar_order + 1)]
        coef_labels.extend([f'b_{i}' for i in range(1, self.ma_order + 1)])
        cov_df = pd.DataFrame(self.cov_θ, index=coef_labels, columns=coef_labels)
        print('covariance matrix:', cov_df, sep='\n')

        # 4 - variance of error
        # unwapping dimensions around this... it's always scalar though
        print('estimated variance of error', self.var_e[0][0])
        print('estimated mean of error', np.mean(self.e))

    # sse vs iterations
    def plot_sse_vs_iter(self):
        plt.plot(range(1, len(self.sse_accum) + 1), self.sse_accum)
        plt.title('sse vs iteration')
        plt.xlabel('iterations')
        plt.ylabel('sse')

    # helper for taking care of true vs pred plot
    def plot_pred_vs_true(self, label, original_data):
        _, pred = self.dlsim(
            num_samples=len(original_data),
            y0=original_data[:self.ar_order + 1])
        pred = pred.reshape(-1)

        # pyplot stuffs
        plt.plot(original_data)
        plt.plot(pred)
        plt.title(f'True and Pred. vs Time Step\n{label}')
        plt.xlabel('Time Step')
        plt.ylabel(label)
        plt.legend(['True', 'Pred.'])

# Welp... this got pretty long
#  returns ar_coefs, ma_coefs 
#  Note: heavy use of optional params for easy tuning/adjustments
#  Note: generally see convergence well within 50 iterations
#  TODO: Generalize/sepparate from ARMA application
#    sounds like an interesting thing to do, but i'm not 
#    sure I'll have time.
def lm(sig, ar_order, ma_order,
    δ = 1e-6, # jacobian matrix approximation (dθ) 
    µ = 0.01, # Gaus-Newton vs Gradient Descent weighting
    µ_max = 1e10, # maximum
    ε = 1e-4, # convergence threshold
    max_iterations = 50,
    verbose=True):
    # hey! python supports symbols... let's use them... it'll match the slides...
    # N is number of samples
    N = sig.shape[0]

    # confirm sig is a column vector
    assert(sig.shape[0] == N and sig.shape[1] == 1)

    # Step 0
    #   setup θ and dimension properly as column vector
    θ = np.zeros((ar_order + ma_order, 1)).reshape(-1, 1)
    n = θ.shape[0]

    # this turned out to be very similar to dlsim... but reversed... 
    #   probably could use common code in some way... mey... maybe another day
    #   returns column vector of Nx1 for [e(1), e(2), ... e(N-1)].T
    def _e(θ):
        # Note: uses sig, ar_order, and ma_order 'implicitly'
        # Note: unlike dlsim, this time decided to use column vectors consistently
        # unless i'm way mistaken... this should pretty much be the reverse of dlsim process... 
        #   pretty much working backways to get the 'estimated' noise array, and then minimize
        #   sum-squared noise array... kinda makes sense.
        # so... 
        #   y(t) + a_1y(t - 1) + ... + a_na y(t - na) = e(t) + b_1 e(t - 1) + ... + b_na y(t - na)
        #   a.T * y = e(t) + b.T * e'  
        #   e(t) = a.T * y - b.T * e'
        # where:
        #   a = [1, a_1, ..., a_na].T
        #   b = [b_1, ..., b_nb].T
        #   e' = [e(t-1), e(t-2), ... , e(t-nb)].T
        #   y = [y(t), y[t-1], ... , y[t-na]].T
        # ar and ma coefficient column vectors
        ar_coefficients = np.append(np.array([1]), θ[:ar_order]).reshape(-1, 1)
        ma_coefficients =  θ[ar_order:].reshape(-1, 1)

        # copied from dlsim above
        #   return column vec of correct size/order
        def coef_vec(v, i):
            if i < 0:
                return np.empty((0, 1))
            if  i < v.shape[0]:
                return v[:i + 1, :]
            else:
                return v

        # copied from dlsim above
        #   return column vec of correct size/order
        def vals_vec(vals, i, order):
            # need reverse order, line up a1y(t-1), a2 y(t-2), etc..
            if i < 0:
                return np.empty((0, 1))
            if i < order:
                return vals[i::-1, :]
            else:
                return vals[i:i - order:-1, :]

        e = np.zeros((N, 1))
        for i in range(N):
            a = coef_vec(
                ar_coefficients,
                i,
            )

            y_temp = vals_vec(sig, i, ar_order + 1)

            # always looking one time-delay back noise wise, as we're 
            #  returning e(t) per iteration... hence i-1
            b = coef_vec(
                ma_coefficients,
                i - 1
            )

            e_temp = vals_vec(e, i - 1, ma_order)

            assert(y_temp.shape[0] > 0)

            if (e_temp.shape[0] > 0):
                e[i, 0] = np.dot(
                    a.T, 
                    y_temp
                ) - np.dot(
                    b.T,
                    e_temp
                )
            else:
                e[i, 0] = np.dot(
                    a.T,
                    y_temp
                )
        return np.nan_to_num(e)
    
    def _sse(e):
        return np.dot(e.T, e)

    # appears to approximate jacobian? really not sure
    #   Note: uses δ
    #   returns matrix with each column x_i as described in lecture slides
    def _X(e, θ):
        # return column vector with ith θ updated
        #   to θ + δ
        def θ_new(i):
            tmp = θ.copy()
            tmp[i, 0] = tmp[i, 0] + δ
            return tmp
        
        X = np.zeros((N, n))
        for i in range(n):
            X[:, i] = ((e - _e(θ_new(i)))/δ).reshape(-1)
        return X

    def step1(θ, e=None, sse=None):
        if e is None:
            e = _e(θ)
        if sse is None:
            sse = _sse(e)
        X = _X(e, θ)
        A = np.dot(X.T, X)
        g = np.dot(X.T, e)
        return e, sse, A, g
    
    def step2(A, g, θ): 
        delta_θ = np.dot(np.linalg.inv(
                A + µ*np.identity(n)
            ),
            g
        )

        θ_new = θ + delta_θ
        e_new = _e(θ_new)
        sse_new = _sse(e_new)
        if np.isnan(sse_new):
            sse_new[0, 0] = 1e8
        return delta_θ, θ_new, e_new, sse_new

    # running initial step1 operation outside loop
    e, sse, A, g = step1(θ)
    num_iterations = None
    cov_θ = None
    var_e = None

    sse_accum = []
    # assumes initial θ is set          
    for i in range(max_iterations):
        sse_accum.append(sse[0,0])
        # Step 1 - after initialization ran at bottom of loop
        # Step 2
        #   Update step math and results       
        #   aww... delta isn't valid variable name symbol... oh well
        delta_θ, θ_new, e_new, sse_new = step2(A, g, θ)

        # Step 3 
        #   Update Logic
        if sse_new < sse:
            # indicates that we've converged... set vars and break from loop
            if np.linalg.norm(delta_θ) < ε:
                θ = θ_new
                e = e_new
                num_iterations = i + 1
                var_e = sse_new/(N-n)
                cov_θ = var_e*np.linalg.inv(A)
                break
            else:
                θ = θ_new
                µ = µ/10

        while sse_new >= sse:
            µ = µ*10
            if µ > µ_max:
                if verbose: 
                    print(f'INFO: µ has exceeded µ_max in iteration {i + 1}')
                # initially (for lab 9) i was off by literally 10 orders of 
                #  magnitude for what an appropriate µ_max value was. 
                #  with it set more reasonably, happiness abounds
                break
            _, θ_new, e_new, sse_new = step2(A, g, θ)
        
        # allows us to skip an e calculation
        θ = θ_new
        e, sse, A, g = step1(θ, e=e_new, sse=sse_new)

    ar_coefficients = np.append([1], θ[:ar_order]).reshape(-1)
    ma_coefficients =  np.append([1], θ[ar_order:]).reshape(-1)

    # check for convergence, and compute esimated standard_error and cov_θ
    if num_iterations is None or cov_θ is None or var_e is None:
        print('WARNING: LMA failed to converge')
        var_e = sse_new/(N-n)
        cov_θ = var_e*np.linalg.inv(A)
    else:
        print('INFO: lma converged in', num_iterations, 'iterations.')

    return LMAResult(
        ar_coefficients, 
        ma_coefficients, 
        cov_θ, 
        var_e, 
        sse_accum,
        sig.shape[0], # num samples,
        e
    )

# let's pull in all the transforms from hw 9
# identity transform
#   haha! the inverse of the identity transform... IS the identify transform
#   doh! not quite. the inverse isn't supposed to return a tuple
#   returns transformed data, and inverse transform function
def identity_transform(data):
    def _identity_inverse(pred):
        return pred
    return data, _identity_inverse

# difference transform
#   return transformed data, and inverse transform function
def lagged_difference_transform(lag):
    _lag = lag
    def difference_transform(data):
        c = data.tolist()[:_lag]

        # define difference inverse in scope where it will 
        #   have a copy of c
        def difference_inverse(pred):
            print('inverting lagdif', _lag)
            inv_accum = c.copy()
            for i in range(len(pred)):
                inv_accum.append(inv_accum[i] + pred[i])
            return np.array(inv_accum)
        
        # perform transform
        accum = []
        for i in range(_lag, len(data)):
            accum.append(data[i] - data[i-_lag])
        return np.array(accum), difference_inverse
    return difference_transform

# difference transform
#   return transformed data, and inverse transform function
def difference_transform(data):
    c = data[0]

    # define difference inverse in scope where it will 
    #   have a copy of c
    def difference_inverse(pred):
        inv_accum = [c]
        for i in range(len(pred)):
            inv_accum.append(inv_accum[i] + pred[i])
        return np.array(inv_accum)
    
    # perform transform
    accum = []
    for i in range(1, len(data)):
        accum.append(data[i] - data[i-1])
    return np.array(accum), difference_inverse

def logarithmic_transform(data):
    def logarithmic_inverse(pred):
        return np.exp(pred)
    
    return np.log(data), logarithmic_inverse


def normalization_transform(data):
    variance = np.var(data)
    mean = np.mean(data)

    def nomalization_inverse(pred):
        return pred * variance + mean

    return (data - mean)/variance, nomalization_inverse

# function for contructing multi-layer transform
#   welp... that's an eyesore.. but it works :)
#   returns transform function
def compose_transform(*transforms):
    inverse_accum = []
    def _compose_transform(data):
        def _compose_inverse(pred):
            for i in inverse_accum[::-1]:
                pred = i(pred)
            return pred
        
        for t in transforms:
            data, i = t(data)
            inverse_accum.append(i)
        return data, _compose_inverse
    return _compose_transform

# got some inspiration from this
#   https://fmwww.bc.edu/repec/bocode/t/transint.html
def reciprocal_transform(data):
    def reciprocal_inverse(pred):
        return np.reciprocal(pred)

    # inverse of the reciprocal tranform, is itself :shrug:
    return np.reciprocal(data), reciprocal_inverse

# why not...
#  https://en.wikipedia.org/wiki/Hyperbolic_functions
def htan_transform(data):
    def htan_inverse(pred):
        return np.arctanh(np.maximum(np.minimum(pred, 1-1e-100), 1e-100-1))
    
    return np.tanh(data), htan_inverse