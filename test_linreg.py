# used this to debug some issues with linreg model
#  relating to Confidence interval/covariance matrix compute
import pandas as pd
import numpy as np
import statsmodels.api as sm
from utils.regression import LinRegModel

# c is target variable
#  1a + 1b + 1 = c
num_samples = 1000
a = np.random.normal(size=1000)
b = np.random.normal(size=1000)
B = np.array([2, .5, .1]).reshape(-1, 1)
# noise = np.random.normal(scale=0, size=1000).reshape(-1, 1)

X = np.zeros((1000, 3))
X[:, 0] = 1
X[:, 1] = a
X[:, 2] = b
Y = np.dot(X, B) # + noise

data = pd.DataFrame({
    'a': a,
    'b': b,
    'c': Y.reshape(-1)
})

model = LinRegModel(intercept=True)
model.fit(data[['a', 'b']], data['c'])
model.print_summary()
print(model.covTable())

# ok.. now I'm wondering if my regression model is broke, let's confirm i get similar
#  behaviour from sklearn's linearregression setup the same way. 
model = sm.OLS(data['c'], sm.add_constant(data[['a', 'b']]))
res = model.fit()
# print(res.cov_params())
print(res.summary())
