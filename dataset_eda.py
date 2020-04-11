# Exploratory Data Analysis
#   This file generates the content for the Introduction

# Import Third-part Libraries 
import pandas as pd

# Import Functions developed through course
from utils import corr

# Need a custom time parser, as the format isn't well-behaved by default.
mydateparser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
df = pd.read_csv('energydata_complete.csv', parse_dates=['date'], date_parser=mydateparser)
df = df.set_index('date')

# Basic Info 
print('Shape of Dataset:', df.shape)
print('Info:', df.info())

# corrplot