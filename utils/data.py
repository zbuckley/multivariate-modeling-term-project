
# This file is the landing place for a number of data
#   specific settings and utilities used through
#   the term-project code.

# Core Python Dependencies
from os.path import sep

# Third Party Imports
import pandas as pd

# Import things developed by me either
#   throughout the coursework, or 
#   specifically for the term project
#   (provided in additional python files)
from .conf import tmp_data_folder, data_source_file

# custom dateparser
dateparser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

# assumes dataset_info_split.py has been ran
def __common_load_df(filename):
    df = pd.read_csv(filename, parse_dates=['date'], date_parser=dateparser)
    df = df.set_index('date')
    return df

def load_original_data():
    return __common_load_df(data_source_file)

def load_y_test():
    return __common_load_df(f'{tmp_data_folder}{sep}y_test.csv')

def load_y_train():
    return __common_load_df(f'{tmp_data_folder}{sep}y_train.csv')

def load_x_test():
    return __common_load_df(f'{tmp_data_folder}{sep}x_test.csv')

def load_x_train():
    return __common_load_df(f'{tmp_data_folder}{sep}x_train.csv')

# used in decomposition.py and holt-winter.py
#   seemed a reasonable place for it
__data_freq = 10 # minutes

# minutes per week over data freq in minutes
weekly_freq = int(60*24*7/__data_freq)

# minutes per day over data freq in minutes
daily_freq = int(60*24/__data_freq)

# minutes per hour over data freq in minutes
hourly_freq = int(60/__data_freq)