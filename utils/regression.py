# This file is a landing place for various optimization algorithms
#   Specifically LSE and LMA as developed in class. 

# Core Python Dependencies
from os.path import sep # attempts to maintain OS portability

# Import Third-part Libraries
import numpy as np # used lots
import pandas as pd # used for pretty print, and input structuring
from scipy.stats import distributions # used for p-values, and confidence intervals

# Import things developed by me either
#   throughout the coursework, or 
#   specifically for the term project
#   (provided in additional python files)
from .optimization import LSE
from . import stats

def print_metrics(y_pred, y_actual, num_params, sample_size):
    r2 = stats.corr(y_actual, y_pred)**2
    residuals = y_actual - y_pred
    print('\tR2:', r2)
    print('\tAdj R2:', stats.adj_r2(r2, y_pred.shape[0]))
    # Had trouble figuring out how to compute this or associated p-value without 
    # just building an OLS model from statsmodels library and getting it from the summary
    # TODO: fix this if time allows.
    # print('\tF-Statistic:', )
    # print('\tF-Statistic (probability):', )
    print('\tAIC:', stats.AIC(sample_size, residuals, num_params))
    print('\tAICc:', stats.AICc(sample_size, residuals, num_params))
    print('\tBIC:', stats.BIC(sample_size, residuals, num_params))
    q = stats.q_value(residuals, sample_size)
    print('\tQ:', q)
    print('\tQ (p-value):', stats.q_to_pvalue(q, sample_size))
    q2 = stats.q_value2(residuals, sample_size)
    print('\tQ*:', q2)
    print('\tQ* (p-value):', stats.q_to_pvalue(q, sample_size))
    print('\tSSE:', stats.sse(residuals))
    print('\tRMSE:', stats.rmse(residuals))
    print('\tResidual Mean:', np.mean(residuals))
    print('\tResidual Var:', np.var(residuals))
    return residuals

def _convertToNumpy(input):
    retVal = input.to_numpy()
    if len(retVal.shape) == 1:
        # force single variable input to be
        #   column vector
        retVal = retVal.reshape(-1, 1)
    return retVal

def _prepXInputForIntercept(x_input):
        tmp = _convertToNumpy(x_input)
        x = np.zeros((tmp.shape[0], tmp.shape[1] + 1))
        x[:, 0] = 1
        x[:, 1:] = tmp
        return x

def predict(X, B):
    return np.dot(X, B)

# calulcated ci, based on slides, and web resources
#   https://kite.com/python/examples/702/scipy-compute-a-confidence-interval-from-a-dataset
#   https://stats.stackexchange.com/questions/241449/matrix-and-regression-model
def _ci_calc(val, se, confidence):
    h = distributions.norm.ppf((1 - (1 - confidence)/2))
    low = val - h*se
    high = val + h*se
    return low, high


class LinRegModel(): 
    def __init__(self, intercept=True):
        self.intercept=intercept
        self._reset_state()


    def _reset_state(self):
        # parameters set in fit routine
        self.X = None
        self.Y = None
        self.B = None
        self._p = None
        self._n = None
        self._df = None
        self.x_cols = None

        # cache params that are set elsewhere
        #  and need be reset when fit is called
        self._residuals = None
        self._cov = None
        self._coeffTable = None
        self._r2 = None
        self._adj_r2 = None

    def _preprocessX(self, x_input):
        if self.intercept is True:
            return _prepXInputForIntercept(x_input)
        else:
            return _convertToNumpy(x_input)

    def fit(self, x_input, y_input):
        self._reset_state()

        self.X = self._preprocessX(x_input)

        self.Y = _convertToNumpy(y_input)
        self.B = LSE(self.X, self.Y)
        self._p = self.B.shape[0] # num parameters
        self._n = self.Y.shape[0] # num samples
        self._df = self._n - self._p # degrees of freedom

        if isinstance(x_input, pd.DataFrame):
            self.x_cols = x_input.columns
        else:
            self.x_cols = [x_input.name]
    
    def predict(self, x_input):
        x_in = self._preprocessX(x_input)

        y_pred = predict(x_in, self.B)
        
        return predict(x_in, self.B)

    # predict residuals given same X used to fit
    def residuals(self):
        if self._residuals is None:
            self._residuals = self.Y - predict(self.X, self.B)
        return self._residuals

    # give covariance estimate per fitted regression problem.
    #  https://stats.stackexchange.com/questions/68151/how-to-derive-variance-covariance-matrix-of-coefficients-in-linear-regression
    #  
    def cov(self):
        if self._cov is None:
            # MLE estimate for variance given in wikipedia article
            #   https://en.wikipedia.org/wiki/Ordinary_least_squares#Estimation
            s2 = np.dot(self.residuals().T, self.residuals()) / self._df
            est_var = s2 * (self._df/self._n)

            # Simple covariance matrix
            #   https://en.wikipedia.org/wiki/Ordinary_least_squares#Covariance_matrix
            self._cov = est_var * np.linalg.inv(np.dot(self.X.T, self.X))
        return self._cov

    def covTable(self):
        labels = self._labelList()

        return pd.DataFrame(self.cov(), columns=labels, index=labels)

    # prints coefficients and their low/high intervals
    def _compute_CITableRow(self, labels, coefs, ses, confidence):
        # verify sanity
        assert(len(coefs) == len(ses))
           
        # print line per coef/CI
        row_accum = []
        for label, coef, se in zip(labels, coefs, ses):
            low, high = _ci_calc(coef, se, confidence=confidence)
            t = stats.ttest_1sample(coef, se)
            p = stats.ttest_pvalue(t, self._df)
            row_accum.append((label, coef, se, low, high, t, p))
        return row_accum

    def _labelList(self):
        if self.intercept is True:
            labels = ['intercept']
            labels.extend(self.x_cols)
        else:
            labels = list(self.x_cols)
        return labels

    # prints table of coefficients with Confidence Intervals
    def coeffTable(self, confidence=0.95):
        # all standard errors
        if self._coeffTable is None:
            ses = np.sqrt(
                self.cov()[np.diag_indices_from(self.cov())]
            )

            labels = self._labelList()

            columns = ['label','coeff', 'std err', 'lower', 'upper','t-test','p-value']
            rows = self._compute_CITableRow(
                labels= labels,
                coefs=self.B.reshape(-1), 
                ses=ses,
                confidence=confidence
            )

            self._coeffTable = pd.DataFrame(rows, columns=columns)
        return self._coeffTable

        
    def print_summary(self):
        # print coefficient, low, high, t-test, p-value for coefficients involved
        print()
        print('##################################################################')
        print('## Linear Regression Model Summary                              ##')
        print('##################################################################')
        print('Coefficients Table:')
        print(self.coeffTable())
        print()
        print('Model Metrics:')
        print_metrics(
            y_pred = predict(self.X, self.B).reshape(-1),
            y_actual = self.Y.reshape(-1),
            num_params = self._p,
            sample_size = self._n
        )