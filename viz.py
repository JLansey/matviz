''' random functions that are helpful
'''

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.pyplot import plot,hist,figure,clf,cla,xlabel,ylabel,xlim,ylim,\
                              gcf,gca,close,title,legend,grid,bar,suptitle,show,\
                              xticks,yticks

import numpy as np
import datetime
from datetime import datetime as dt

from scipy import stats
import seaborn as sns
from .etl import nan_smooth
from .etl import round_time


def plot_range(events, color='#0093e7',offset=0, height=1):
    if not hasattr(events[0], "__len__"):
        events = [events]
    # Fill registered cur_event times
    for cur_event in events:
        plt.fill_between([cur_event[0], cur_event[1]],
                         [height + offset, height + offset],
                         [offset,offset],
                         color=color, label='Event',alpha=0.5)


def set_fontsize(f_size=15):
    ax = plt.gca()
    [w.set_fontsize(f_size) for w in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                                                    ax.get_xticklabels() +
                                                    ax.get_yticklabels())]
    return ax


def print_fig_fancy(fpathh,dpi=300):
    plt.savefig(fpathh, dpi=dpi)


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

    if line_styles==None:
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
    if sum(I)>3:
        m, b = np.polyfit(x[I], y[I], 1)
        plt.plot(x, m*np.array(x) + b,**line_styles)


# plot a diagonal line with x=y to see if your predictions are biased
def plot_diag(lw=1):
    ax = plt.gca()
    ex = ax.get_xlim()
    yy = ax.get_ylim()
    if np.diff(ex)[0]>np.diff(yy)[0]:
        y = yy
        x = yy
    else:
        y=ex
        x=ex
    plt.plot(x,y,'--',color='.5', lw=lw)


def plot_zero(lw=1, lineheight=0, linecolor='.5', style='--', axx='x'):
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

    return plt.plot(x,y,style, color=linecolor, lw=lw)


def bar_centered(y,**kwargs):
#     its like a regular bar plot, except that it is centered on 1:N integers
    x = np.arange(len(y))+1
    h = plt.bar(x,y,align='center',**kwargs)
    plt.xticks(x)
    return h



def subplotter(x,y,n, xlbl=None, ylbl=None):
    # a subplotter function that works like the matlab one does but with index starting at 0
    tupp = (x,y)
    cnt = 0
    for ii in range(tupp[0]):
        for jj in range(tupp[1]):
            if cnt==n:
                ax = plt.subplot2grid(tupp, (ii, jj))
                if jj == 0:
                    if ylbl:
                        ylabel(ylbl)
                if ii - 1 == y:
                    if xlbl:
                        xlabel(xlbl)

                return ax
            cnt+=1



    raise Exception("You have only " + str(x*y) + " subplots, but you asked for the " + str(n) + "'th")


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
    if y is  None:
        y = x
        x = np.array(range(len(y)))

    x = np.array(x).astype(float)
    y = np.array(y).astype(float)

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
            plot(ex, yy, '--g', linewidth=3)
        else:
            plot(ex, yy, '--g', **line_style)

    return slope, intercept

def test_logfit():
    logfit(np.arange(10), 2 * np.random.randn(10) + np.arange(10),
           marker_style={'color': 'r', 'lw':0,  'marker': 'o'},
           line_style={'color': 'g', 'lw': 3})


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
        if wiggle == None:
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

    if wiggle == None:
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



def nicefy(fsize=15, f_size=False, clean_legend=False, cur_fig=None, background = 'white', resize=True, legend_outside=False,
           expand_y=False, expand_x=False, touch_limits=True):
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

    # if legend_outside:
    #     gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.0)





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
    return sns.color_palette(color, n_colors=n)

def format_axis_date(rot=77):
    plt.xticks(rotation=rot)
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