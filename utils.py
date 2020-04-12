# This file provides a collection of routines and utilities used 
#   throughout the other python programs. I created it in an attempt
#   to avoid a rediculout amount of copy-pasted code in code 
#   for the term project.
# For the most part, These are slightly modified versions of the 
#   utilities we've developed throughout the course. 
# Core Python Dependencies
from pathlib import Path # used to setup folders
from os.path import sep

# Third-Party Dependencies
import pandas as pd

# folder config/setup
# This file is provided, but can be downloaded from 
#   the github project associated with the original 
#   research paper.
#   https://github.com/LuisM78/Appliances-energy-prediction-data/blob/master/energydata_complete.csv
data_source_file=f'data{sep}energydata_complete.csv'
tmp_graphics_folder='tmp_graphics'
tmp_data_folder='tmp_data'

# function to initialize folders as needed
def init_tmp_folders():    
    Path(tmp_graphics_folder).mkdir(parents=True, exist_ok=True)

# custom dateparser
dateparser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

# assumes dataset_info_split.py has been ran
def __common_load_df(filename):
    df = pd.read_csv(f'{tmp_data_folder}{sep}{filename}', parse_dates=['date'], date_parser=dateparser)
    df = df.set_index('date')
    return df

def load_y_test():
    return __common_load_df('y_test.csv')

def load_y_train():
    return __common_load_df('y_train.csv')

def load_x_test():
    return __common_load_df('x_test.csv')

def load_x_train():
    return __common_load_df('x_train.csv')
