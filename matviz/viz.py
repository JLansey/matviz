''' random functions that are helpful
'''

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.pyplot import plot, scatter, hist, figure, clf, cla, xlabel, ylabel, xlim, ylim,\
                              gcf, gca, close, title, legend, grid, bar, suptitle, show,\
                              xticks, yticks, axis, boxplot

from sklearn.metrics import roc_curve, auc
import numpy as np
import datetime
from datetime import datetime as dt

# std library
from itertools import chain

import math
import sys
from queue import PriorityQueue
import pandas as pd

# typing
from typing import List, Dict, Union, Any, Iterable

from scipy import stats
import seaborn as sns
from .etl import nan_smooth
from .etl import round_time
from .histogram_utils import nhist
from scipy import interpolate


# mini function to change a [a, b] into [[a,b]] for use in the for loop later
def list_ize(w):
    if len(w) > 0:
        if hasattr(w[0], "__len__"):
            return w
        else:
            return [w]
    else:
        return []




def plot_range(events, color='#0093e7', y_offset='none', height='none', zorder=None, alpha=0.5, **varargs):
    """
    Shade vertical regions on the current axes.

    Parameters
    ----------
    events : list of [start, end] pairs
        X positions defining the regions to shade.
    color : str, optional
        Fill color. Default is ``'#0093e7'``.
    y_offset : float, optional
        Bottom of the shaded region. Default uses current y-axis lower limit.
    height : float, optional
        Height of the shaded region. Default spans the full y-axis.
    zorder : int, optional
        Drawing order.
    alpha : float, optional
        Transparency. Default is 0.5.
    """
    events = list_ize(events)
    yy = ylim()
    if y_offset == 'none':
        y_offset = yy[0]
    if height == 'none':
        height = yy[1] - yy[0]

    to_label = 'none'
    # Fill registered cur_event times
    for cur_event in events:
        plt.fill_between([cur_event[0], cur_event[1]],
                         [height + y_offset, height + y_offset],
                         [y_offset, y_offset],
                         color=color, alpha=alpha, zorder=zorder, label=to_label, **varargs)
        # make sure only one legend item appears for this event series
        to_label = '_nolegend_'




def plot_range_idx(t, events, **varargs):
    """
    Shade regions by index into a time series.

    Converts index pairs into time-domain ranges and calls
    `plot_range`.

    Parameters
    ----------
    t : array-like
        Time series (x-axis values).
    events : list of [start_idx, end_idx] pairs
        Index pairs into *t* defining the regions.
    **varargs
        Passed to `plot_range`.
    """
    # if it is a series, get the values
    if str(type(t)) == "<class 'pandas.core.series.Series'>":
        t = t.values

    events = list_ize(events)

    #                 last element + ds
    t = np.append(t, [t[-1] + (t[-1] - t[-2])])

    event_times = [[t[a], t[b]] for a, b in events]

    plot_range(event_times, **varargs)



def plot_cdf(data, *args, **kargs):
    """
    Plot the empirical cumulative distribution function (CDF).

    Parameters
    ----------
    data : array-like
        Input data. NaN values are removed.
    *args, **kargs
        Passed to ``plt.plot``.
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    x = np.sort(data)
    y = 100 * np.arange(len(x)) / len(x)
    plt.plot(x, y, *args, **kargs)
    plt.ylim([0,100])


def set_fontsize(f_size=15):
    """
    Set the font size of the current axes' title, labels, and tick labels.

    Parameters
    ----------
    f_size : int, optional
        Font size in points. Default is 15.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The current axes.
    """
    ax = plt.gca()
    [w.set_fontsize(f_size) for w in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                                    ax.get_xticklabels() +
                                                    ax.get_yticklabels())]
    return ax


def print_fig_fancy(fpathh, dpi=300, **kwargs):
    plt.savefig(fpathh, dpi=dpi, facecolor=[0, 0, 0, 0], **kwargs)


def make_title(title_str,format=False):

    if title_str:

        lower_case_words = {'in','the','a','and','of'}
        upper_case_words = {}

        # title_str = 'a_time_in_bed and fish'
    #     split by
        tmp = title_str.replace("_"," ").split()
        if format:
            upper_words = [w[0].upper()+w[1:] if len(w)>1 else w[0].upper() for w in tmp]
            lower_words = [w[0].lower()+w[1:] if w.lower() in lower_case_words else w for w in upper_words]
        else :
            lower_words = [w.lower() for w in tmp]
            # lower_words[0][0]=lower_words[0][0].upper()
            lower_words[0] = lower_words[0][0].upper() + lower_words[0][1:]

        upper_words =  [w.upper() if w.upper() in upper_case_words else w for w in lower_words]


        combined_words = ' '.join(upper_words)
    else:
        combined_words = ''

    return combined_words

def large_fig(fig_num=1):
    """
    Create a large figure (15 x 8 inches).

    Parameters
    ----------
    fig_num : int, optional
        Figure number. Default is 1.
    """
    plt.figure(fig_num,figsize=(15,8))


def fancy_plotter(x,y,marker_style='o',line_styles=None):
    """
    Plot x vs y with a linear trend line overlay.

    Parameters
    ----------
    x : array-like
        X data.
    y : array-like
        Y data.
    marker_style : str, optional
        Marker format string. Default is ``'o'``.
    line_styles : dict, optional
        Keyword arguments for the trend line. Default is
        ``{'color': '0.4', 'lw': 3}``.
    """

    if line_styles is None:
        line_styles = {'color':'0.4','lw':3}

    # check that these are not regular lists, so that we can use logical indexing later
    if type(y)!=np.ndarray:
        y=np.array(y)
    if type(x)!=np.ndarray:
        x = np.array(x)

    plt.plot(x,y,marker_style)
    # if
    I1 = np.logical_not(np.isnan(y))
    I2 = np.logical_not(np.isnan(x))
    I = np.logical_and(I1,I2)
    if sum(I) > 3:
        m, b = np.polyfit(x[I], y[I], 1)
        plt.plot(x, m*np.array(x) + b,**line_styles)


def cplot(z, *args, **kargs):
    """
    Plot complex numbers on the real/imaginary plane.

    Parameters
    ----------
    z : complex array-like
        Complex data to plot (real -> x, imag -> y).
    *args, **kargs
        Passed to ``plt.plot``.
    """
    plot(np.real(z), np.imag(z), *args, **kargs)


def cplot_circle(z_center, r):
    t = np.linspace(0, 2 * np.pi, 100)
    z = r * np.exp(1j * t) + z_center
    cplot(z)

def ctext(z, *args, **kargs):
    """
    Place text at a position given by a complex number.

    Parameters
    ----------
    z : complex
        Position (real -> x, imag -> y).
    *args, **kargs
        Passed to ``plt.text``.
    """
    plt.text(np.real(z), np.imag(z), *args, **kargs)

def cscatter(z, *args, **kargs):
    """
    Scatter plot of complex numbers on the real/imaginary plane.

    Parameters
    ----------
    z : complex array-like
        Complex data (real -> x, imag -> y).
    *args, **kargs
        Passed to ``plt.scatter``.
    """
    scatter(np.real(z), np.imag(z), *args, **kargs)

def polar_grid(lw=1, r=False, linecolor='.3', style=':', nrings=2, nrays = 6):
    """
    Overlay a polar grid (rings and rays) on the current axes.

    Useful for complex-plane plots.

    Parameters
    ----------
    lw : float, optional
        Line width. Default is 1.
    r : float, optional
        Radius of the grid. Default is auto-detected from axis limits.
    linecolor : str, optional
        Color of grid lines. Default is ``'.3'``.
    style : str, optional
        Line style. Default is ``':'``.
    nrings : int, optional
        Number of concentric rings. Default is 2.
    nrays : int, optional
        Number of radial rays. Default is 6.

    Returns
    -------
    line : list of Line2D
        The plotted grid lines.
    """
    ax = plt.gca()
    axis('equal')
    ex = ax.get_xlim()
    yy = ax.get_ylim()

    to_plot = []
    if not r:
        r = np.min(np.abs(np.concatenate([ex, yy])))
    t = np.linspace(0, 2 * np.pi, nrays +1)[:-1]
    z = 100 * r * np.exp(1j * t)
    for cur in z:
        to_plot.append(0)
        to_plot.append(cur)
        to_plot.append(np.nan)

    t = np.linspace(0, 2 * np.pi, 200)
    for cur_r in np.linspace(r/nrings, r, nrings):
        circz = cur_r * np.exp(1j * t)
        to_plot = to_plot + [np.nan] + list(circz)

    cur_plt = cplot(np.array(to_plot), style, color=linecolor, lw=lw)
    xlim(ex)
    ylim(yy)
    return cur_plt


def plot_diag(lw=1, color='.5', reverse=False):
    """
    Plot a diagonal x=y reference line on the current axes.

    Parameters
    ----------
    lw : float, optional
        Line width. Default is 1.
    color : str, optional
        Line color. Default is ``'.5'``.
    reverse : bool, optional
        If True, plot x = -y instead. Default is False.
    """
    ax = plt.gca()
    ex = ax.get_xlim()
    yy = ax.get_ylim()
    if np.diff(ex)[0] > np.diff(yy)[0]:
        y = yy
        x = yy
    else:
        y = ex
        x = ex

    if reverse:
        x = np.flip(x)
    plt.plot(x, y, '--', color=color, lw=lw, label='_nolegend_')

def plot_zero(lineheight=0, axx='x', **kwargs):
    """
    Plot a horizontal or vertical reference line.

    Parameters
    ----------
    lineheight : float, optional
        Position of the line. Default is 0.
    axx : {'x', 'y'}, optional
        ``'x'`` for a horizontal line, ``'y'`` for vertical. Default is ``'x'``.
    **kwargs
        Passed to ``plt.plot``. Defaults to a gray dashed line.
    """

    if len(kwargs) == 0:
        kwargs = {'color' : '.5',
                    'linestyle' : '--',
                    'lw' : 1,
                    'label': '_nolegend_'
                    }

    ax = plt.gca()
    if axx == 'x':
        ex = ax.get_xlim()
        x = ex
        y = [lineheight, lineheight]
    elif axx == 'y':
        yy = ax.get_ylim()
        x = [lineheight, lineheight]
        y = yy
    else:
        raise Exception("you can't plot zero on no axis!")

    return plt.plot(x,y, **kwargs)
    # return plt.plot(x,y,style, lw=lw, **kwargs)

def plot_axes(color='.5'):
    """
    Plot both horizontal and vertical reference lines at zero.

    Parameters
    ----------
    color : str, optional
        Line color. Default is ``'.5'``.
    """
    plot_zero(axx='x', color=color)
    plot_zero(axx='y', color=color)


def plot_pin(x, y, color='k'):
    """
    Plot a pin (vertical line with dot) at a specific x position.

    Parameters
    ----------
    x : float
        X position of the pin.
    y : float
        Height of the pin.
    color : str, optional
        Color of the line and marker. Default is ``'k'``.
    """
    plot([x, x], [0, y], linewidth=3, color=color)
    plot([x], [y], 'o', color=color, markersize=10)


def bar_centered(y, **kwargs):
    """
    Bar plot centered on integers 1 through N.

    Parameters
    ----------
    y : array-like
        Bar heights.
    **kwargs
        Passed to ``plt.bar``.

    Returns
    -------
    container : BarContainer
        The bar container.
    """
    x = np.arange(len(y))+1
    h = plt.bar(x, y, align='center', **kwargs)
    plt.xticks(x)
    return h


def errorb(cur_series, serror=True):
    """
    Bar plot with error bars from a pandas Series of arrays.

    Parameters
    ----------
    cur_series : pandas.Series
        Each element is an array-like of values. The index provides
        x-tick labels.
    serror : bool, optional
        If True (default), use standard error. If False, use standard
        deviation.
    """
    means = cur_series.apply(np.nanmean)
    errors = cur_series.apply(np.nanstd)
    if serror:
        errors = errors / np.sqrt(len(errors))

    x_pos = np.arange(len(cur_series)) + 1
    bar(x_pos, means,
        yerr=errors,
        align='center',
        alpha=0.5,
        color='gray',
        capsize=4)

    ax = gca()
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cur_series.index)
    format_axis_date()

def bplot(X):
    if type(X) == dict:
        keys = X.keys()
        S = len(keys)
        for idx, k in enumerate(keys):
            x = X[k]
            x = x.astype(float)
            x = x[~np.isnan(x)]
            boxplot(x, positions=[idx], patch_artist=True, boxprops=dict(facecolor='lightblue', lw=1, color='k'),
             widths=0.6, whiskerprops=dict(linewidth=2, color='k', linestyle='-'),
             medianprops=dict(linewidth=2))

        xticks(range(S))
        xticklabels(keys)

        if type(keys) == str:
            format_axis_date()

        nicefy()


def subplotter_auto(n, ii, **kwargs):
    """
    Create a subplot with automatically chosen grid dimensions.

    Computes a near-square grid that fits *n* subplots and activates
    the *ii*-th one.

    Parameters
    ----------
    n : int
        Total number of subplots.
    ii : int
        Index of the subplot to activate (0-based).
    **kwargs
        Passed to `subplotter`.
    """
    y = int(np.ceil(np.sqrt(n)))
    x = int(np.ceil(n / y))
    subplotter(x, y, ii, **kwargs)


def xticklabels(all_lbl):
    gca().set_xticklabels(all_lbl)
    
def yticklabels(all_lbl):
    gca().set_yticklabels(all_lbl)


def subplotter(x, y=None, nth=None, xlbl=None, ylbl=None, y_ticks=None):
    """
    MATLAB-style subplot with 0-based indexing and spanning support.

    Parameters
    ----------
    x : int
        Number of rows, or a 3-digit integer like ``220`` meaning
        2 rows, 2 columns, 0th subplot.
    y : int, optional
        Number of columns.
    nth : int or list of int, optional
        Subplot index (0-based). Pass a list to span multiple cells.
    xlbl : str, optional
        X-axis label, shown only on the bottom row.
    ylbl : str, optional
        Y-axis label, shown only on the left column.
    y_ticks : list or False, optional
        Y-tick labels for non-left columns. False hides them.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The created subplot axes.
    """

    # allow you to enter input like (220) instead of x=2, y=2, nth =0
    if x > 99:
        nth = int(x % 10)
        y = int((x - nth) % 100)
        x = int((x - y) / 100)
        y = int(y / 10)

    kwargs = {}
    if type(nth) != int:
        # note special case y == 1, where rowspan should be used
        if len(nth) > y:
            kwargs = {'colspan': y,
                      'rowspan': len(nth) / y}
            if int(kwargs['rowspan']) != kwargs['rowspan']:
                raise Exception("this isn't supported yet")
            else:
                kwargs['rowspan'] = int(kwargs['rowspan'])

        else:
            if nth[1] == nth[0] + 1 and y > 1:
                kwargs = {'colspan': len(nth)}
            else:
                kwargs = {'rowspan': len(nth)}

        nth = nth[0]
    tupp = (x, y)
    cnt = 0
    for ii in range(tupp[0]):
        for jj in range(tupp[1]):
            if cnt==nth:
                ax = plt.subplot2grid(tupp, (ii, jj), **kwargs)
                if jj == 0:
                    if ylbl:
                        ylabel(ylbl)
                    
                else:
                    if y_ticks is not None:
                        if y_ticks == False:
                            yticklabels([])
                        else:
                            yticklabels(y_ticks)

                if ii + 1 == x:
                    if xlbl is not None:
                        xlabel(xlbl)


                return ax
            cnt+=1



    raise Exception("You have only " + str(x*y) + " subplots, but you asked for the " + str(nth) + "'th")


def pop_all():
    """
    Bring all matplotlib figures to the foreground.

    Useful when using IPython in the terminal.

    Returns
    -------
    int
        Number of figures shown.
    """
    all_figures=[manager.canvas.figure for manager in matplotlib.\
        _pylab_helpers.Gcf.get_all_fig_managers()]
    [fig.canvas.manager.show() for fig in all_figures]
    return len(all_figures)

def suplabel(axis,label,label_prop=None,
             labelpad=5,
             ha='center',va='center'):
    """
    Add a shared xlabel or ylabel to the figure, similar to ``suptitle``.

    Parameters
    ----------
    axis : {'x', 'y'}
        Which axis to label.
    label : str
        The label text.
    label_prop : dict, optional
        Keyword arguments for ``plt.text``.
    labelpad : float, optional
        Padding from the axis in points. Default is 5.
    ha : str, optional
        Horizontal alignment. Default is ``'center'``.
    va : str, optional
        Vertical alignment. Default is ``'center'``.
    """
    fig = plt.gcf()
    xmin = []
    ymin = []
    for ax in fig.axes:
        xmin.append(ax.get_position().xmin)
        ymin.append(ax.get_position().ymin)
    xmin,ymin = min(xmin),min(ymin)
    dpi = fig.dpi
    if axis.lower() == "y":
        rotation=90.
        x = xmin-float(labelpad)/dpi
        y = 0.5
    elif axis.lower() == 'x':
        rotation = 0.
        x = 0.5
        y = ymin - float(labelpad)/dpi
    else:
        raise Exception("Unexpected axis: x or y")
    if label_prop is None:
        label_prop = dict()
    plt.text(x,y,label,rotation=rotation,
               transform=fig.transFigure,
               ha=ha,va=va,
               **label_prop)


# Note: this is a half-complete port of my former matlab code
# https://www.mathworks.com/matlabcentral/fileexchange/29545-power-law-exponential-and-logarithmic-fit?s_tid=prof_contriblnk
def logfit(x, y=None, graph_type='linear', ftir=.05, marker_style='.k', line_style='--g',
           skip_begin = 0, skip_end = 0):
    """
    Fit and plot a line through data on linear, semi-log, or log-log axes.

    Parameters
    ----------
    x : array-like
        X data (or y data if *y* is None, or complex with real=x, imag=y).
    y : array-like, optional
        Y data.
    graph_type : {'linear', 'logy', 'loglog'}, optional
        Fit type. Default is ``'linear'``.
    ftir : float, optional
        Fraction to extend the fit line beyond the data. Default is 0.05.
    marker_style : str or dict, optional
        Marker style for data points. Default is ``'.k'``.
    line_style : str or dict, optional
        Line style for the fit. Default is ``'--g'``.
    skip_begin, skip_end : int, optional
        Number of points to skip at the beginning/end of the fit.

    Returns
    -------
    slope : float
        Slope of the fit.
    intercept : float
        Intercept of the fit.
    """

    # check if you only passes one var in
    if y is  None:
        if np.iscomplex(x[0]):
            y = np.imag(x)
            x = np.real(x)
        else:
            y = x
            x = np.array(range(len(y)))

    # convert to floats
    x = np.array(x).astype(float)
    y = np.array(y).astype(float)

    # remove any nans
    I = np.logical_not(np.isnan(x)) & np.logical_not(np.isnan(y))
    x = x[I]
    y = y[I]

    def linearfit(x2fit, y2fit):

        cur_range = np.array([x2fit.min(), x2fit.max()])
        tot_range = cur_range[1] - cur_range[0]
        exp_range = cur_range + ftir * tot_range * np.array([- 1, 1])

        ex = np.linspace(exp_range[0], exp_range[1], 100)

        slope, intercept, r_value, p_value, std_err = stats.linregress(x2fit, y2fit)
        yy = slope * ex + intercept
        return slope, intercept, ex, yy

    def logyfit(x2fit, y2fit):
        gca().set_yscale('log')
        y2fit = np.log10(y2fit)
        idxs = np.isfinite(y2fit)
        y2fit = y2fit[idxs]
        x2fit = x2fit[idxs]

        slope, intercept, ex, yy = linearfit(x2fit, y2fit)
        return slope, intercept, ex, np.power(10,yy)

    def loglogfit(x2fit, y2fit):
        gca().set_xscale('log')
        gca().set_yscale('log')
        x2fit = np.log10(x2fit)
        y2fit = np.log10(y2fit)

        idxs = np.isfinite(y2fit) & np.isfinite(x2fit)
        y2fit = y2fit[idxs]
        x2fit = x2fit[idxs]

        slope, intercept, ex, yy = linearfit(x2fit, y2fit)
        return slope, intercept, np.power(10,ex), np.power(10,yy)


    if len(x) < 3:
        return np.nan, np.nan

    graph_calc = {'linear': linearfit,
                  'logy': logyfit,
                  'loglog': loglogfit}
    if graph_type in graph_calc:
        if skip_end > 0:
            slope, intercept, ex, yy = graph_calc[graph_type](x[skip_begin:-skip_end], y[skip_begin:-skip_end])
        else:
            slope, intercept, ex, yy = graph_calc[graph_type](x[skip_begin:], y[skip_begin:])


    else:
        raise Exception("we can't do that graph type yet")

    if marker_style:
        if type(marker_style) == str:
            plot(x, y, marker_style)
        else:
            plot(x, y, **marker_style)
    if line_style:
        if type(line_style) == str:
            plot(ex, yy, line_style, linewidth=3)
        else:
            plot(ex, yy, '--g', **line_style)

    return slope, intercept

def test_logfit():
    logfit(np.arange(10), 2 * np.random.randn(10) + np.arange(10),
           marker_style={'color': 'r', 'lw':0,  'marker': 'o'},
           line_style={'color': 'g', 'lw': 3, 'linestyle':'--'})


def streamgraph(df, smooth=None, normalize=None,
                wiggle=None, label_dict=None, color=None,
                order=True, linewidth=0.5, round_time=False, legend_flag=True):
    """
    Create a streamgraph (stacked area chart with wiggle baseline).

    Parameters
    ----------
    df : DataFrame
        Two-column DataFrame: first column is the time/x axis,
        second column is the categorical variable to stack.
    smooth : int, optional
        Smoothing window width. Default is None (no smoothing).
    normalize : bool, optional
        If True, normalize to 100%. Default is None.
    wiggle : bool or str, optional
        Baseline mode. True for wiggle, False for zero baseline,
        or ``'stream'``/``'river'`` for weighted wiggle. Default is auto.
    color : str, dict, or palette, optional
        Seaborn palette name, dict of label->color, or palette object.
    order : bool or list, optional
        True to sort by peak height, or a list of labels. Default is True.
    linewidth : float, optional
        Edge line width. Default is 0.5.
    round_time : str, optional
        Pandas frequency string to round the time column. Default is False.
    legend_flag : bool, optional
        Show legend. Default is True.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the streamgraph.
    """
    # reorder a list from inside out, for use with streamgraph
    def reorder_idx(idxs):
        idxs = list(np.flip(idxs))
        for ii in np.flip(np.arange(1, len(idxs), 2)):
            idxs.append(idxs.pop(ii))
        return np.array(idxs)

    #   normalize for use with streamgraph so everything sums to 1
    def normalize_y(y):
        return 100 * y / y.sum(axis=0)

    #     interpret wiggle and normalize params for cool defaults
    #     if not wiggle:
    #         if normalize:

    #     curate the data:
    #     add stupid col for who knows why - maybe can leave it out ..

    cur_cols = df.columns
    if len(cur_cols) == 1:
        df['fake_col'] = 'blank'
        legend_flag = False

    if round_time:
        df[cur_cols[0]] = df[cur_cols[0]].dt.round(round_time)


    df['dumb12345'] = 'stoopid'
    cur_cols = df.columns
    df = df[[cur_cols[2], cur_cols[0], cur_cols[1]]]
    cur_datas_grouped = df.groupby([cur_cols[0], cur_cols[1]], sort=False).count().unstack()
    cur_clean = cur_datas_grouped.fillna(0).transpose().sort_index(axis=1)

    cur_labels = np.array([w[1] for w in list(cur_clean.index)])

    if len(cur_labels) > 200:
        raise Exception("Whoa dude - you are stacking too many things (" + str(len(cur_labels)) + ")")


    # prepare that sweet sweet data
    x = list(cur_clean.columns)
    y = np.array(cur_clean)

    #     smooth it so it isn't so choppy
    if smooth:
        y = np.array([nan_smooth(w, smooth) for w in y])

    #     normalize it so it takes up the whole graph
    #     remember to set the y'lims below - or unset the autoscale y
    if normalize:
        y = normalize_y(y)
        if wiggle is None:
            wiggle = False

    # prepare the colors for plotting
    if not color:
        #   Maybe do something depending on "N here?"
        color = 'muted'
    if type(color) == str:
        pallet_list = sns.color_palette(color, n_colors=len(y))
        pallet_dict = {k: pallet_list[ii] for ii, k in enumerate(cur_labels)}
    elif type(color) == sns.palettes._ColorPalette:
        pallet_list = color
        pallet_dict = {k: pallet_list[ii] for ii, k in enumerate(cur_labels)}
    elif type(color) == dict:
        pallet_dict = color

    if wiggle is None:
        wiggle = True

    #   prepare the idxs (if you'll wiggle)
    if order:
        if type(order) == list:
            idxs = [cur_labels.tolist().index(w) for w in order]
        #             print(idxs)
        else:
            idxs = np.argsort(np.amax(y, axis=1))
            if wiggle in [True, 'wiggle', 'streamgraph', 'stream', 'themeriver', 'river']:
                idxs = reorder_idx(idxs)[::-1]
    else:
        idxs = np.arange(len(y))

    labels_reord = cur_labels[idxs]
    cur_palette = [pallet_dict[w] for w in labels_reord]

    if wiggle == True:
        cur_base = 'wiggle'
    elif wiggle == False:
        cur_base = 'zero'
    elif wiggle in ['streamgraph', 'stream', 'themeriver', 'river']:
        cur_base = 'weighted_wiggle'
    else:
        cur_base = wiggle

    #   do the plot!
    ax = plt.stackplot(x, y[idxs], labels=labels_reord,
                       baseline=cur_base, colors=cur_palette, lw=linewidth, edgecolor='k')


    gca().set_yticklabels(np.abs(gca().get_yticks()).astype(int))

    if legend_flag:
        lgnd = legend(labels_reord)
        gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.0)

    # nicefy()
    plt.style.use('classic')

    return gca()

## example data for use in streamgraph:
# index   date column, grouped column
# 11623   2011-06-01  United States
# 13117   2011-07-01         Europe
# 13118   2011-06-01  United States
# 13120   2011-07-01        Oceania
# 13121   2011-08-01         Canada
# 13122   2011-12-01  United States
# 13123   2011-08-01  United States
# 13125   2011-10-01         Canada
# 65169   2011-08-01         Canada
# 13126   2011-09-01  United States
# 65170   2011-09-01         Canada








# Does this work? can it be used inside nicefy?
# def set_fontsize(f_size=15, ax='none'):
#     if ax == 'none':
#         ax = plt.gca()
#
#     [w.set_fontsize(f_size) for w in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
#                                                     ax.get_xticklabels() +
#                                                     ax.get_yticklabels())]
#     return ax



def nicefy(fsize=15, f_size=False, clean_legend=False, cur_fig=None, background = 'white', resize=False, legend_outside=False,
           expand_y=False, expand_x=False, touch_limits=False, touch_text=False):
    """
    Make the current figure publication-ready.

    Adjusts font sizes, removes top/right spines, applies tight layout,
    and optionally sets background color and cleans up legends.

    Parameters
    ----------
    fsize : int, optional
        Font size for all text elements. Default is 15.
    f_size : int, optional
        Deprecated alias for *fsize*.
    clean_legend : bool, optional
        If True, make the legend background transparent. Default is False.
    cur_fig : Figure, optional
        Figure to nicefy. Default is the current figure.
    background : str, optional
        ``'white'``, ``'black'``, or a matplotlib style name.
        Default is ``'white'``.
    touch_limits : bool, optional
        If True, enable tight autoscaling. Default is False.
    expand_y, expand_x : bool or str, optional
        Expand axis limits slightly. ``True`` expands both directions,
        ``'top'`` expands only the upper end. Default is False.
    touch_text : bool, optional
        If True, auto-format axis label and title text. Default is False.
    """

    # backwards compatability for this change
    if f_size != False:
        fsize = f_size

    #todo: check if you are log scale, and do expandx or expandy to be top

    if cur_fig:
        fig = cur_fig
    else:
        fig = plt.gcf()

    # fig.set_tight_layout(True)
    if background == 'black':
        fig.set_facecolor('black')
        plt.style.use('dark_background')
    elif background  == 'white':
        fig.set_facecolor('white')
        plt.style.use('classic')
    else:
        plt.style.use(background)


    # ['seaborn-darkgrid', 'Solarize_Light2', 'seaborn-notebook', 'classic', 'seaborn-ticks', 'grayscale', 'bmh',
    #  'seaborn-talk', 'dark_background', 'ggplot', 'fivethirtyeight', '_classic_test', 'seaborn-colorblind',
    #  'seaborn-deep', 'seaborn-whitegrid', 'seaborn-bright', 'seaborn-poster', 'seaborn-muted', 'seaborn-paper',
    #  'seaborn-white', 'fast', 'seaborn-pastel', 'seaborn-dark', 'tableau-colorblind10', 'seaborn',
    #  'seaborn-dark-palette']

    axes = fig.get_axes()
    # remove the grids
    # [ax.grid(False) for ax in axes]

    for ax in axes:
        if touch_text:
            ax.xaxis.set_label_text(make_title(ax.xaxis.label.get_text()))
            ax.yaxis.set_label_text(make_title(ax.yaxis.label.get_text()))
            ax.set_title(make_title(ax.title.get_text()))

        [w.set_fontsize(fsize) for w in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                          ax.get_xticklabels() +
                                          ax.get_yticklabels())]

    [plt.setp(ax.get_legend().get_texts(), fontsize=fsize) for ax in axes if ax.get_legend()]

    ax = gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


    def expand_the_bounds(cur_bounds):
        cur_range = np.diff(cur_bounds)
        return cur_bounds + cur_range * 0.02 * np.array([-1, 1])

    def expand_the_top(cur_bounds):
        cur_range = np.diff(cur_bounds)
        return cur_bounds + cur_range * 0.02 * np.array([0, 1])

    if touch_limits:
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.autoscale(enable=True, axis='y', tight=True)
        if expand_y == True:
            ylim(expand_the_bounds(ylim()))
        elif str(expand_y) == 'top':
            ylim(expand_the_top(ylim()))
        if expand_x == True:
            xlim(expand_the_bounds(xlim()))
        elif str(expand_x) == 'top':
            xlim(expand_the_top(xlim()))
    # thanks
    # https://stackoverflow.com/questions/925024/how-can-i-remove-the-top-and-right-axis-in-matplotlib

    if resize and (not touch_limits):
        plt.rcParams["figure.figsize"] = [7, 5]
    plt.rcParams['image.cmap'] = 'viridis'

    if clean_legend:
        plt.legend(framealpha=0.0)

    # fix titles so they all appear
    plt.tight_layout()
    # if legend_outside:
    #     gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.0)

def xylim(w):
    """
    Set both x and y axis limits to the same range.

    Parameters
    ----------
    w : float or array-like
        If scalar, sets limits to ``[-w, w]``. If a pair, sets limits
        to ``[w[0], w[1]]``.
    """
    if not hasattr(w, '__len__'):
        w = [-w, w]
    xlim(w)
    ylim(w)

def xyscale(w):
    plt.xscale(w)
    plt.yscale(w)

def axis_robust(AX):
    """
    Set axis limits selectively, leaving ``None`` entries unchanged.

    Parameters
    ----------
    AX : list of 4 floats or None
        ``[xmin, xmax, ymin, ymax]``. Use ``None`` for any limit you
        want to keep at its current value.

    Examples
    --------
    >>> axis_robust([0, None, None, None])  # only set xmin to 0
    """
    AX_orig = axis()
    for ii in range(4):
        if AX[ii] is None:
            AX[ii] = AX_orig[ii]
    axis(AX)

def xlim_robust(AX):
    axis_robust([AX, *ylim()])

def ylim_robust(AX):
    axis_robust([*xlim(), AX])


def set_axis_ticks_pctn(cur_axis = 'x'):
    fmt = '%.0f%%'  # Format you want the ticks, e.g. '40%'
    ticker_obj = matplotlib.ticker.FormatStrFormatter(fmt)
    if cur_axis.lower() == 'x':
        cur_axis_h = gca().xaxis
    elif cur_axis.lower() == 'y':
        cur_axis_h = gca().yaxis
    else:
        raise Exception("You must pass either x or y, you passed: " + str(cur_axis))
    cur_axis_h.set_major_formatter(ticker_obj)


def plot_endpoints(endpoints, color='#0093e7'):

        x_starts = [w[0] for w in endpoints]
        x_ends = [w[1] for w in endpoints]

        # Set figure size
        plt.figure(figsize=(16, 8))

        # Make a subplot for each day highlighting registered, filtered and
        # Get timestamp for star and end of each day
        x_start = min(x_starts)
        x_end = max(x_ends)

        # Make a subplot for each day
        # plt.ylabel(day.strftime('%Y-%m-%d'), rotation=0, labelpad= 30)

        # Get event for each date
        for idx, event in enumerate(endpoints):
            plt.fill_between(
                [event[0], event[1]],
                [1, 1],
                color=color, label='Event')

        plt.ylim((0, 1))
        plt.xlim((x_start, x_end))

        # Eliminate spaces between subplots
        # plt.subplots_adjust(hspace=0)

        # Create color patches for the legend
        # Is this right???
        # red_patch = plt.patches.Patch(color='#0093e7', label='Detected')

        # Plot legend on the bottom subplot
        # plt.legend(loc='lower left', borderaxespad=0., prop={'size': 12},
        #            ncol=3, fancybox=True, shadow=True, handles=[red_patch])

def linspecer(n, color='muted'):
    """
    Generate *n* distinguishable colors from a seaborn palette.

    Parameters
    ----------
    n : int
        Number of colors.
    color : str, optional
        Seaborn palette name. Default is ``'muted'``.

    Returns
    -------
    colors : ndarray of shape (n, 3)
        RGB color values.
    """
    return np.array(sns.color_palette(color, n_colors=n))


def format_axis_date(rot=77):
    """
    Rotate x-axis tick labels for date readability.

    Parameters
    ----------
    rot : float, optional
        Rotation angle in degrees. Default is 77.
    """
    plt.xticks(rotation=rot, rotation_mode="anchor", ha='right')
    # ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    # plt.xticks(x,tick_labels)
    # plt.xticks(rotation=70)

# def x
# gca().set_xscale('log')
# gca().set_yscale('log')


# pcolor on a log scale, with zero values marked
def logpcolor(x, y, C):
    C = np.log10(C)
    C[C == -np.inf] = -np.max(C)
    return plt.pcolor(x, y, C)


def brighten(c, frac = .5):
    """
    Brighten an RGB color by blending it toward white.

    Parameters
    ----------
    c : array-like
        RGB color (values 0-1).
    frac : float, optional
        Blend fraction. Smaller values produce brighter colors.
        Default is 0.5.

    Returns
    -------
    color : ndarray
        Brightened RGB color.
    """
    return np.array(c) * frac + 1 - frac



# manually add a colorbar, definitely not the best way to go about this
# but was really tricky to do otherwise
def example_add_colobar():
    Y = np.linspace(0, 10.5, 100)
    scale_func = lambda w: (w * 100 / 10.5).astype(int)
    C = linspecer(101, "coolwarm")
    add_colorbar(Y, C, scale_func)


def add_colorbar(Y, C, scale_func):
#     Y = np.linspace(0, .8, 100)
    c = [C[w] for w in scale_func(Y)]
    for ii, y in enumerate(Y):
        plot([0, 1], [y, y], lw=4, color=c[ii])
    xticks([])
    gca().yaxis.tick_right()



def legend_helper(fig: Union[plt.Figure, plt.Axes],
                  *args: Iterable[plt.Axes]) -> Dict[str, Any]:
    """
    Collect legend handles and labels from multiple axes.

    Parameters
    ----------
    fig : Figure or Axes
        A matplotlib Figure (collects from all its axes) or a single Axes.
    *args : Axes
        Additional axes to collect from (when *fig* is an Axes).

    Returns
    -------
    dict
        Dict with ``'handles'`` and ``'labels'`` lists.
    """
    if isinstance(fig, plt.Figure):
        handles, labels = [list(chain.from_iterable(seq)) for seq in zip(*(
            ax.get_legend_handles_labels() for ax in fig.axes
        ))]

    else:
        handles, labels = [list(chain.from_iterable(seq)) for seq in zip(*(
            ax.get_legend_handles_labels() for ax in chain([fig], args)
        ))]

    return {
        'handles': handles,
        'labels': labels,
    }



def calc_plot_ROC(y1, y2):
    """
    Plot an ROC curve from two distributions used as a binary classifier.

    Parameters
    ----------
    y1 : array-like
        Scores for the negative class.
    y2 : array-like
        Scores for the positive class.

    Returns
    -------
    auc : float
        Area under the ROC curve.
    """

    y_score = np.concatenate([y1, y2])
    y_true = np.concatenate([np.zeros(len(y1)), np.ones(len(y2))])

    return plot_ROC(y_true, y_score)

def plot_ROC_hist(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    return nhist({'true': y_score[y_true], 'false': y_score[~y_true]}, normalize='number')

def plot_ROC(y_true, y_score, c='k'):
    """
    Plot an ROC curve from true labels and predicted scores.

    Parameters
    ----------
    y_true : array-like
        Binary ground-truth labels.
    y_score : array-like
        Predicted scores (higher = more likely positive).
    c : str, optional
        Line color. Default is ``'k'``.

    Returns
    -------
    auc : float
        Area under the ROC curve.
    """

    I = np.logical_not(np.isnan(y_score))
    y_true = y_true[I]
    y_score = y_score[I]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    cur_auc = auc(fpr, tpr)
    plot(fpr, tpr, c)
    plot_diag()
    xylim([-.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve, AUC = ' + '{:.2f}'.format(cur_auc))
    nicefy()
    return cur_auc


def jitter(xx, yy, maxn=4, xscale=None):
    """
    Horizontally jitter overlapping points so they are visible.

    Parameters
    ----------
    xx : array-like
        X positions.
    yy : array-like
        Y positions (used to detect overlap).
    maxn : int, optional
        Maximum number of dots wide per bin. Default is 4.
    xscale : float, optional
        Horizontal spacing between jittered points. Default is auto.

    Returns
    -------
    xx : ndarray
        Jittered x positions.
    """

    if xscale is None:
        # split 50 percent of a bar width 1 by the max number of elements to fit
        xscale = 0.5 / maxn


    def get_bins(yy, bin_width):
        """
        Get the bins if you know the bin_width
        Then center it
        :param yy:
        :param bin_width:
        :return:
        """
        bins = np.arange(min(yy), max(yy) + bin_width, bin_width)
        # print(bins)
        # bins = bins - (max(bins) - max(yy)) / 2
        return bins


    def get_bin_width(yy, maxn, n_iter=5):
        """
        Get the right bin width, do a binary search on histogram bin width
        checking various bin widths until you've found one where at most 'n' items fall in one bin
        if everything is equal then the maxn parameter doesn't matter
        todo: handle the case where y values are equal more elegantly
        :param yy:
        :return:
        """

        # if you know the bin width - what is the most that fall in one bin
        get_max_per_bin = lambda yy, bin_width: np.max(np.histogram(yy, bins=get_bins(yy, bin_width))[0])

        # initialize bin width options to be the most and least it could be
        if np.std(yy) == 0:
            bin_width = 1
        else:
            yy_sort = sorted(yy)
            bin_width_min = np.min(np.diff(yy_sort))
            bin_width_max = yy_sort[-1] - yy_sort[0]
            for cnt in range(n_iter):
                bin_width = (bin_width_max + bin_width_min) / 2
                max_per_bin = get_max_per_bin(yy, bin_width)
                if max_per_bin >= maxn:
                    bin_width_max = bin_width
                else:
                    bin_width_min = bin_width

        return bin_width

    if len(yy) < 2:
        # if there is only one point, then you don't need to jitter it
        return xx
    else:
        xx = np.array(xx).astype(float)

        bin_width = get_bin_width(yy, maxn, n_iter=5)
        bins = get_bins(yy, bin_width)

        idxs = np.digitize(yy, bins, right=True)
        for bin_idx in range(len(bins)):
            I = idxs == bin_idx
            n = sum(I)
            push = -(n - 1) / 2 # so it will be centered if there is only one.
            xx[I] = xx[I] + xscale * (push + np.arange(n))

    return xx

def interp_plot(x, y, *args, **kargs):
    """
    Plot with PCHIP interpolation for smooth curves from sparse data.

    Parameters
    ----------
    x : array-like
        X data (supports pandas Timestamps).
    y : array-like
        Y data. NaN values are removed before interpolation.
    *args, **kargs
        Passed to ``plt.plot``.

    Returns
    -------
    x_i : ndarray
        Interpolated x values.
    y_i : ndarray
        Interpolated y values.
    """

    y = np.array(y)
    x = np.array(x)

    if len(x) == 0:
        return

    date_flag = type(x[0]) == pd._libs.tslibs.timestamps.Timestamp
    if date_flag:
        x = np.array([w.value for w in x])

    n = 400
    I = np.logical_not(pd.isnull(y))

    x = x[I]
    y = y[I]

    x_i = np.linspace(np.min(x), np.max(x), n)

    f = interpolate.PchipInterpolator(x, y)
    y_i = f(x_i)

    if date_flag:
        x_i = np.array([pd.Timestamp(w) for w in x_i])

    plot(x_i, y_i, *args, **kargs)

    return x_i, y_i





