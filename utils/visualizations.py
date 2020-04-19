# this file defines several graphics utilities
#  for assisting with plotting ACF, GPAC, etc..

# Core Python Dependencies

# Import Third-part Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import things developed by me either
#   throughout the coursework, or 
#   specifically for the term project
#   (provided in additional python files)
from . import stats


def acf_plot(residuals, label, k_max):
    residuals_mean = np.mean(residuals)
    acf = [(k, stats.autocorrelation_estimation(residuals, k, y_mean=residuals_mean))
        for k in range(-k_max, k_max + 1)]

    # setting use_line_collection to true hides a deprecation warning
    plt.stem([a for a, b in acf],
        [b for a, b in acf])
    plt.title(f'ACF Plot of Generated Signal\n{label}')
    plt.tight_layout()

def plot_GPAC(table, label):
    sns.heatmap(table, robust=True, annot=True)
    plt.title("GPAC Table\n" + label)
    plt.xscale
    plt.xlabel('Autoregressive Order (na-1)')
    plt.ylabel('Moving Avg Order (nb)')
    plt.tight_layout()