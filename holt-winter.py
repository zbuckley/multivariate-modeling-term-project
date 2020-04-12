# This file is responsibly for applying
#   the Holt-Winters Method to the dataset
# Core Python Dependencies
from os.path import sep # attempts to maintain OS portability

# Import Third-part Libraries


# Import things developed by me either
#   throughout the coursework, or 
#   specifically for the term project
#   (provided in additional python files)
import utils

# Let's load y_train and y_test. 
#   attempt to fit and predict using a holtwinter model
y_train = utils.load_y_train()
y_test = utils.load_y_test()

model = ets.ExponentialSmoothing(y_train, seasonal='additive', freq=utils.daily_freq)
