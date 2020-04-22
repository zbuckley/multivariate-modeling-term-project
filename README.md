# Term Project
Term Project for Multivariate Modeling

## Getting the Data

The data is available from the github repository associated with the paper. 
https://github.com/LuisM78/Appliances-energy-prediction-data

That being said. It's also included in the submission package for convenience.

Getting access to the paper does seem to require paying sciencedirect, despite digging for quite a while. 
As such I'll simply include it in my submission. Along with, for convenience, 2 other papers I reference in the report, that didn't require payment but took quite a bit of digging to locate. 

## Running the Code

The software is organized into drivers/scripts (all the .py files in the root directory of the project), and the utils package (located in the utils directory). The software in the utils directory can only be used through the scripts, as internally the utils software uses relative references, which don't allow for direct execution. 

The order in which I ran the driver scripts. 
NOTE: In theory, running `data_info_split.py` first, is the only dependency the other scripts have, but I did not test that extensively. 

1. `data_info_split.py`
2. `stationary_testing.py`
3. `decomposition.py`
4. `holt.py` NOTE: This file takes about 10 minutes to run.
5. `feature_selection_regression.py`
6. `regression_model_evaluation.py`
7. `arma_model_identification.py`
8. `arma_model_evaluation.py`

I've also included the `test_linreg.py` file which I used for validating reasonable behavior while developing the LinReg model class located in `utils/regression.py`. 



