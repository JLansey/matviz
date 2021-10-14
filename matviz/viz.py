''' random functions that are helpful
'''

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.pyplot import plot, scatter, hist, figure, clf, cla, xlabel, ylabel, xlim, ylim,\
                              gcf, gca, close, title, legend, grid, bar, suptitle, show,\
                              xticks, yticks, axis

from sklearn.metrics import roc_curve, auc
import numpy as np
import datetime
from datetime import datetime as dt

# std library
from itertools import chain

import math
import sys
from queue import PriorityQueue

# typing
from typing import List, Dict, Union, Any, Iterable

from scipy import stats
import seaborn as sns
from .etl import nan_smooth
from .etl import round_time


# mini function to change a [a, b] into [[a,b]] for use in the for loop later
def list_ize(w):
    if hasattr(w[0], "__len__"):
        return w
    else:
        return [w]




def plot_range(events, color='#0093e7', y_offset='none', height='none', zorder=None, **varargs):
    """

    :param events: x positions where the range should be plotted
    :param color:
    :param y_offset:
    :param height:
    :return:
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
                         color=color, alpha=0.5, zorder=zorder, label=to_label, **varargs)
        # make sure only one legend item appears for this event series
        to_label = '_nolegend_'




def plot_range_idx(t, events, **varargs):
    """
    Plot range - for timeseries t, and events=indexed points in t

    :param t: timeseries
    :param events: indexes in that series
    :return:
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
    x = sorted(data)
    y = 100 * np.arange(len(data)) / len(data)
    plt.plot(x, y, *args, **kargs)
    plt.ylim([0,100])




def set_fontsize(f_size=15):
    ax = plt.gca()
    [w.set_fontsize(f_size) for w in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                                    ax.get_xticklabels() +
                                                    ax.get_yticklabels())]
    return ax


def print_fig_fancy(fpathh, dpi=300):
    plt.savefig(fpathh, dpi=dpi, facecolor=[0, 0, 0, 0])


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
    plt.figure(fig_num,figsize=(15,8))


def fancy_plotter(x,y,marker_style='o',line_styles=None):
    '''
    will plot x and y along with a fancy trend line

    '''

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


# plot complex numbers in an arg and diagram
def cplot(z, *args, **kargs):
    plot(np.real(z), np.imag(z), *args, **kargs)

def cscatter(z, *args, **kargs):
    scatter(np.real(z), np.imag(z), *args, **kargs)

# add a bulls-eye to the graph, polar grid lines
def polar_grid(lw=1, r=False, linecolor='.3', style=':', nrings=2, nrays = 6):
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


# plot a diagonal line with x=y to see if your predictions are biased
def plot_diag(lw=1, color='.5'):
    ax = plt.gca()
    ex = ax.get_xlim()
    yy = ax.get_ylim()
    if np.diff(ex)[0] > np.diff(yy)[0]:
        y = yy
        x = yy
    else:
        y = ex
        x = ex
    plt.plot(x, y, '--', color=color, lw=lw)

# plot a horizontal line, or a vertical line
def plot_zero(lineheight=0, axx='x', **kwargs):

    if len(kwargs) == 0:
        kwargs = {'color' : '.5',
                    'linestyle' : '--',
                    'lw' : 1
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
    plot_zero(axx='x', color=color)
    plot_zero(axx='y', color=color)


def plot_pin(x, y, color='k'):
    """
    Plot a pin at a specific point on the x axis
    :param x: position of the pin on the x axis
    :param y: height of the pin
    :param color: color of the line and marker
    :return:
    """
    plot([x, x], [0, y], linewidth=3, color=color)
    plot([x], [y], 'o', color=color, markersize=10)


def bar_centered(y, **kwargs):
#     its like a regular bar plot, except that it is centered on 1:N integers
    x = np.arange(len(y))+1
    h = plt.bar(x,y,align='center',**kwargs)
    plt.xticks(x)
    return h

def subplotter_auto(n, ii, **kwargs):
    #     automatically select the right number of subplots for n items
    x = int(np.ceil(np.sqrt(n)))
    y = x
    subplotter(x, y, ii, **kwargs)


def subplotter(x, y, nth, xlbl=None, ylbl=None):
    """
    :param x: number of rows
    :param y: number of columns
    :param nth: order, if you pass a list then it spans multiple rows or columns
    :param xlbl: xlabel, if you want it to appear way on the bottom
    :param ylbl: ylabel, if you want it to appear way on the left only
    :return:
    """
    # a subplotter function that works like the matlab one does but with index starting at 0
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
                if ii + 1 == x:
                    if xlbl is not None:
                        xlabel(xlbl)


                return ax
            cnt+=1



    raise Exception("You have only " + str(x*y) + " subplots, but you asked for the " + str(nth) + "'th")


def pop_all():
    '''
    bring all the figures hiding in the background to the foreground
    useful when using ipython in the terminal
    '''
    all_figures=[manager.canvas.figure for manager in matplotlib.\
        _pylab_helpers.Gcf.get_all_fig_managers()]
    [fig.canvas.manager.show() for fig in all_figures]
    return len(all_figures)

def suplabel(axis,label,label_prop=None,
             labelpad=5,
             ha='center',va='center'):
    ''' Add super ylabel or xlabel to the figure
    Similar to matplotlib.suptitle
    axis       - string: "x" or "y"
    label      - string
    label_prop - keyword dictionary for Text
    labelpad   - padding from the axis (default: 5)
    ha         - horizontal alignment (default: "center")
    va         - vertical alignment (default: "center")
    '''
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


# stream_graph
def streamgraph(df, smooth=None, normalize=None,
                wiggle=None, label_dict=None, color=None,
                order=True, linewidth=0.5, round_time=False, legend_flag=True):
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
    '''
    make the figure nicer in general, like ready to be printed etc.
    '''

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
    Sets x and y limits to be w[0], w[1]
    if w is a single number then axes are set to -w, w
    :param w: 
    :return:
    """
    if not hasattr(w, '__len__'):
        w = [-w, w]
    xlim(w)
    ylim(w)

def xyscale(w):
    plt.xscale(w)
    plt.yscale(w)


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

# fix it to depend on N like this:
# https://www.mathworks.com/matlabcentral/fileexchange/42673-beautiful-and-distinguishable-line-colors-colormap
def linspecer(n, color='muted'):
    return np.array(sns.color_palette(color, n_colors=n))


def format_axis_date(rot=77):
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


# brighten a color c
# the smaller the frac, the brighter it will be
def brighten(c, frac = .5):
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
    Provides handles and labels of all provided axes.

    David S. Fulford

    https://towardsdatascience.com/easy-matplotlib-legends-with-functional-programming-64615b529118
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
    Take two distributions and plot the ROC curve if you used the difference
    in those distributions as a binary classifier.
    :param y1:
    :param y2:
    :return:
    """

    y_score = np.concatenate([y1, y2])
    y_true = np.concatenate([np.zeros(len(y1)), np.ones(len(y2))])

    return plot_ROC(y_true, y_score)


def plot_ROC(y_true, y_score):

    I = np.logical_not(np.isnan(y_score))
    y_true = y_true[I]
    y_score = y_score[I]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    cur_auc = auc(fpr, tpr)
    plot(fpr, tpr, 'k')
    plot_diag()
    xylim([-.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve, AUC = ' + '{:.2f}'.format(cur_auc))
    nicefy()
    return cur_auc


def jitter(xx, yy, maxn=4, xscale=None):
    """
    in case two point appear at the same value, the jitter function will make
    them appear slightly separated from each other so you can see the real
    number of points at a given location.
    
    :param xx: 
    :param yy:
    :param maxn: the maximum number of dots wide
    :param exact: if you want to jitter exactly
    :return:
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
        :param yy:
        :return:
        """

        # if you know the bin width - what is the most that fall in one bin
        get_max_per_bin = lambda yy, bin_width: np.max(np.histogram(yy, bins=get_bins(yy, bin_width))[0])

        # initialize bin width options to be the most and least it could be
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

