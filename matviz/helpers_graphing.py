'''
This file is to make your working environment pretty similar to the matlab working environment, with lots of useful
functions imported so you don't need to worry about typing 'plt.' beforehand. It also imports several custom functions
'''

import matplotlib
# matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections
import datetime
from datetime import datetime as dt
import pickle
import glob
import mpld3
import time # used in tic() toc()
import copy


# # use these in notebooks
# %load_ext autoreload
# %autoreload 2
# %matplotlib

# should we filter warnings by default? probably not
# import warnings
# warnings.filterwarnings('ignore')

# import directly a bunch of useful functions from matplotlib and numpy
import matplotlib.patches as patches
from matplotlib.pyplot import plot, hist, figure, clf, cla, xlabel, ylabel, xlim, ylim, \
                              gcf, gca, sca, close, title, legend, grid, bar, suptitle, show,\
                              xticks, yticks, hist2d, pcolor, pcolormesh, yscale, xscale, axis,\
                              contour, colorbar, scatter, boxplot, savefig, tight_layout,\
                              text

from numpy import mean, log10, log, sqrt, power, linspace, sin, cos, tan,\
                        arcsin, arccos, arctan, inf, nan

from math import pi
from random import random as rand

# load in all the specialized functions.
from .viz import *
from .etl import *
from .histogram_utils import nhist, ndhist
from . import cbrt_scale

# If you get JSON serilization errors because of zoomplot, then you might need this
# python -m pip install --user "git+https://github.com/javadba/mpld3@display_fix"
def zoom_plot(enable = True):
    if enable:
        mpld3.enable_notebook()
    else:
        mpld3.disable_notebook()


def fig_sizer(a='none', b='none'):
    if a == 'none':
        a = 10
        b = 15
    elif b == 'none':
        b = a

    plt.rcParams["figure.figsize"] = [b, a]

def display_pnct(cnt,N):
    print("%"+ str(round(100*cnt/N)) + ", " + str(cnt) + "/" + str(N))

def return_pnct(cnt,N):
    return ("%"+ str(round(100*cnt/N)) + ", " + str(cnt) + "/" + str(N))


# display the status every so often
def count_helper(cnt, S=1, freq=10, pcnt=False):
    if pcnt:
        freq = np.round(S * freq / 100)

    if cnt % freq == 0:
        if pcnt:
            print(str(100.0 * cnt / S) + "% complete")
        else:
            print(str(cnt) + " / " + str(S))

def tic():
    """
    Homemade version of matlab tic function
    modified by Lansey for python3 from->
    http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python

    """
    global startTime_for_tictoc
    startTime_for_tictoc = time.perf_counter()

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
            print("Elapsed time is " + str(time.perf_counter() - startTime_for_tictoc) + " seconds.")
        else:
            print("Toc: start time not set")
        return time.perf_counter() - startTime_for_tictoc


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


def xticklabels(all_lbl):
    gca().set_xticklabels(all_lbl)


def nhist_multi(cur, **varargs):
    n = int(np.ceil(np.sqrt(len(cur))))
    for cnt, k in enumerate(cur.keys()):
        subplotter(n, n, cnt)
        nhist(cur[k], **varargs)
        title(k)
    tight_layout()

# set some nice defaults for plotting
plt.rcParams["figure.figsize"] = [12, 9]
plt.rcParams['image.cmap'] = 'viridis'
