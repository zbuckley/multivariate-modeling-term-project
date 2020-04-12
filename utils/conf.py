# This file to provide general configuration information and helper
#   routines, used throughout the term-project code.

# Core Python Dependencies
from pathlib import Path # used to setup folders
from os.path import sep

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



