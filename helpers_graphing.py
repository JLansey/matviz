'''
This file is to make your working environment pretty similar to the matlab working environment, with lots of useful
functions imported so you don't need to worry about typing 'plt.' beforehand. It also imports several custom functions
'''

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import collections
import datetime
from datetime import datetime as dt

import pickle


# # use this in notebooks
# %load_ext autoreload
# %autoreload 2

# %matplotlib


# should we filter warnings by default? probably not
# import warnings
# warnings.filterwarnings('ignore')


import time # used in tic() toc()

from matplotlib.pyplot import plot, hist, figure, clf, cla, xlabel, ylabel, xlim, ylim, \
                              gcf, gca, close, title, legend, grid, bar, suptitle, show,\
                              xticks, yticks, hist2d, pcolor, yscale, xscale, axis

from numpy import mean, log10, log, sqrt, power

from .viz import *
from .etl import *
from .histogram_utils import nhist, ndhist


# import etl as etl
# from etl import flatten_list,recurse_func,list_depth
# from toolbox.interactive_computing import *

import matplotlib.patches as patches


def fig_sizer(a, b):
    plt.rcParams["figure.figsize"] = [a, b]

def display_pnct(cnt,N):
    print("%"+ str(round(100*cnt/N)) + ", " + str(cnt) + "/" + str(N))

def return_pnct(cnt,N):
    return ("%"+ str(round(100*cnt/N)) + ", " + str(cnt) + "/" + str(N))


# display the status every so often
def count_helper(cnt, S, freq=10):
    if cnt % freq == 0:
        print(str(cnt) + " / " + str(S))


def tic():
    """
    Homemade version of matlab tic function
    modified by Lansey for python3 from->
    http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python

    """
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    """
    Homemade version of matlab toc function
    modified by Lansey for python3 from->
    http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python

    """
    if not startTime_for_tictoc:
        print('hey you never hit start')
    else:
        if 'startTime_for_tictoc' in globals():
            print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
        else:
            print("Toc: start time not set")
        return time.time() - startTime_for_tictoc


def silent_toc():
    """
    Homemade version of matlab toc function
    modified by Lansey for python3 from->
    http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
    The silent version does not print any statements, just returns the value

    """
    if not startTime_for_tictoc:
        return 0
    else:
        import time
        if 'startTime_for_tictoc' in globals():
            return time.time() - startTime_for_tictoc
        else:
            return None



# set some nice defaults for plotting
plt.rcParams["figure.figsize"] = [12, 9]
plt.rcParams['image.cmap'] = 'viridis'
