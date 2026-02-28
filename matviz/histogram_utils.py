

import matplotlib.pyplot as plt
import copy
import numpy as np
import matplotlib.dates as md
import datetime
from numpy import log10
import seaborn as sns
from .etl import handle_dates
from .etl import eps
from .etl import flatten, unflatten
from scipy.ndimage import gaussian_filter

import logging




# exclude extremes will ignore the things outside the bounds,
def nhist(X,f=1.2,title=None,xlabel=None,ylabel=None,labels=None,legend=None,noerror=False,
          max_bins=175,std_times=4,color=None,normalize=False,same_bins_flag=False,int_bins_flag=None,
          maxx=None, minx=None, exclude_extremes=False, alpha = .4):
    """
    Plot one or more histograms with automatic binning.

    Automatically sets the number and range of bins using Scott's normal
    reference rule. Compares multiple datasets on one plot with legend,
    mean, and standard deviation markers.

    Parameters
    ----------
    X : list, dict, or DataFrame
        The data to histogram. Can be a single list/array, a list of
        lists, a dictionary of arrays, or a pandas DataFrame.
    f : float, optional
        Factor applied to Scott's normal reference rule. Higher values
        produce more bins. Default is 1.2.
    title : str, optional
        Title for the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis. Default is inferred from other settings.
    labels : list of str, optional
        Legend labels for each dataset.
    legend : list of str, optional
        Alias for *labels*.
    noerror : bool, optional
        If True, suppress the mean/std error bars. Default is False.
    max_bins : int, optional
        Maximum number of bins allowed. Default is 175.
    std_times : float, optional
        Number of standard deviations to show on each side.
        Data beyond this is bunched into edge bars. Default is 4.
    color : list or str, optional
        List of RGB colors, a seaborn palette, or a palette name string.
    normalize : bool or str, optional
        Controls the y-axis. ``False`` uses counts for single plots and
        PDF for multiple. Options: ``'frac'``, ``'proportion'``,
        ``'percent'``, ``'number'``, ``'none'``, or ``True`` for PDF.
    same_bins_flag : bool, optional
        Force all datasets to use identical bin edges. Default is False.
    int_bins_flag : bool, optional
        Force bin edges onto integers. Default is None (auto-detect).
    maxx : float, optional
        Maximum x-axis limit.
    minx : float, optional
        Minimum x-axis limit.
    exclude_extremes : bool, optional
        If True, exclude the edge bars that collect data beyond
        *std_times*. Default is False.
    alpha : float, optional
        Transparency of the histogram bars. Default is 0.4.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The current figure. Histogram data is attached as ``fig.nhist``,
        a dict with keys ``'N'`` (counts), ``'bins'`` (bin edges), and
        ``'rawN'`` (raw counts before normalization).

    Examples
    --------
    >>> A = {'mu=0': np.random.randn(10**5), 'mu=2': np.random.randn(10**3) + 2}
    >>> fig = nhist(A)
    >>> fig = nhist(A, color='viridis')
    >>> fig = nhist(A, same_bins_flag=True)
    """

    # do we really want to have redundant labels?
    if legend:
        labels = legend

    # if you passed a pandas DataFrame:
    if str(type(X)) == "<class 'pandas.core.frame.DataFrame'>":
        X = X.to_dict('list')

    # if you passed a dictionary, turn it into a list
    if isinstance(X,dict):
        if legend:
            logging.warning('passed in legend is being ignored. Since you passed a dict there ' + \
                  'would be no way for us to enforce that the order be the same')
        labels = list(X.keys())
        X = list(X.values())

    # in case we only have one item total to graph
    if not hasattr(X[0], "__len__"):
        X = [X]

    X = [list(x) for x in X]


    X, datetime_flag, date_formatting = handle_dates(X)

    # remove nones or nans
    X = [[x1 for x1 in x if x1 is not None and not np.isnan(x1)] for x in X]

    # count and remove -infs and infs
    inf_counts = [ [sum(np.array(x) == -np.inf), sum(np.array(x) == np.inf)] for x in X]
    X = [[x1 for x1 in x if np.isfinite(x1)] for x in X]



    S = len(X)

    bin_factor = f


    multiple_flag = len(X)>1 # normal is separate hists for each guy you plot instead of them all on the same bit.
    fracylabel_flag = False
    if normalize:
        # correct for shorthand input
        if normalize == 'pcnt':
            normalize = 'percent'

        if normalize in ['frac','proportion','prop','percent']:
            normalize_flag = True
            # if normalize in ['frac']:
            fracylabel_flag = True
            # else:
            #     fracylabel_flag = False
        elif normalize in {'number', 'none'}:
            normalize_flag = False
        else:
            normalize_flag = normalize


    elif normalize == False:
        normalize_flag = multiple_flag
    else: # must be set to none specifically
        normalize_flag = False

    num_points = [len(x) for x in X]
    X_std = [np.std(x) for x in X]
    X_mean = [np.mean(x) for x in X]


    # n_bins = np.min([n_bins, 1000])
    # n_bins = np.max([n_bins,10])
    bins_in, bin_widths = choose_bins(X,bin_factor=bin_factor,max_bins=max_bins,std_times=std_times,
                                      sameBinsFlag=same_bins_flag,maxx=maxx,minx=minx,int_bins_flag=int_bins_flag,
                                      exclude_extremes=exclude_extremes)

    # todo: change the error here to a warning, and set labels to the default
    if labels:
        if len(labels) != S:
            logging.error("Your labels list length " + str(len(labels)) + " doesn't equal the number of elements in your data: " + str(S))
    else:
        labels = [str(w) for w in range(S)]


    # calculate the histogram:
    N = [np.histogram(x, bins=bin)[0] for x,bin in zip(X,bins_in)]

    if not exclude_extremes:
        # add in those infinity's that we removed earlier
        for ii, cur_counts in enumerate(inf_counts):
            N[ii][0] += cur_counts[0]
            N[ii][-1] += cur_counts[1]



    def adjust_bins_to_plot(bins, bin_width):
        bins[0] = bins[1] - bin_width
        bins[-1] = bins[-2] + bin_width
        return bins

    if exclude_extremes:
        bins_to_plot = bins_in
    else:
        bins_to_plot = [adjust_bins_to_plot(cur_bins, cur_width) for cur_bins, cur_width in zip(bins_in, bin_widths)]

    rawN=copy.deepcopy(N) # save the rawN before normalization

    if normalize_flag:
        #     n   = (each value)/(width of a bin * total number of points)
        #     n   =  n /(Total area under histogram);
        if fracylabel_flag:
            for k in range(S):
                N[k]=100.0*N[k]/(num_points[k])
        else:
            for k in range(S):
                N[k]=N[k]/(bin_widths[k]*num_points[k])

    # current_palette = sns.color_palette("Set2",5)
    if color is not None:
        if type(color) == str:
            current_palette = sns.color_palette(color, S)
        else:
            current_palette= color
    else:
        current_palette = sns.color_palette("Set1",S)
        current_palette = current_palette[::-1]
        # switch red and blue if you want ...
        # if len(current_palette)>1:
        #     tmp = current_palette[0]
        #     current_palette[0]=current_palette[1]
        #     current_palette[1]=tmp
        # make it so that blue is the color if only one color is used.
        if len(current_palette)==1:
            current_palette = sns.color_palette("Set1",2)
            current_palette = [current_palette[1]]

    # sns.palplot(current_palette)

    N_max = max([max(n) for n in N])

    for k in range(S):
        # add a zero to the end here for that last non-bin
        plt.bar(bins_to_plot[k], np.append(N[k],0), width=bin_widths[k], \
                label=labels[k], alpha=alpha, \
                lw=0, color=current_palette[k], \
                align='edge')
        # plt.bar(np.append(0,bins_in[k]),np.append(np.append(N[k],0),0), width=bin_widths[k], \
        #         label=labels[k],alpha=.5, \
        #         lw=0,color=current_palette[k], \
        #         align='edge')

        # add a zero to the beginning here so that the line completes down to the bottom
        # and on the other end too.
        plt.step(np.append(bins_to_plot[k],bins_in[k][-1]), np.append(np.append(0, N[k]), 0), \
                 lw=2, color=current_palette[k], \
                 )


        # plt.bar(bins_in[k][:-1], N[k], width=bin_widths[k], alpha=alpha, \
        #         fc=current_palette[k],label=labels[k], \
        #         lw=0,\
        #         align='center')
        #
    # plot the errorbars
    if not noerror:
        for k in range(S):
            plt.errorbar(X_mean[k],N_max*(1+.1*(S-k+1)),xerr=X_std[k],fmt='o',color=current_palette[k],alpha=1,lw=3)


    if not ylabel:
        if normalize_flag:
            if fracylabel_flag:
                ylabel = '% of points'
            else:
                ylabel = 'pdf'
        else:
            ylabel = 'number'
    plt.ylabel(ylabel)

    if multiple_flag:
        plt.legend(loc='upper right')
        plt.legend(framealpha=0.0)

    # plt.autoscale(enable=True, axis='y', tight=True)

    # plot those ***s
    # get those legend colors - and plot them only if
    # for each thingy the N[0] is not zero, on the left, or if the N[-1] is not zero on the right


    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)

    ax = plt.gca()
    if datetime_flag:
        xfmt = md.DateFormatter(date_formatting)
        ax.xaxis.set_major_formatter(xfmt)
        [tick.set_rotation(70) for tick in ax.get_xticklabels()]
        plt.subplots_adjust(bottom=0.3)



    # clean up things a bit
    plt.grid(False)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    fig = plt.gcf()
    fig.nhist = {'N': N, 'bins': bins_in, 'rawN': rawN}
    return fig



def ndhist(x, y=None, log_colorbar_flag=False, maxx=None, maxy=None, minx=None, miny=None,
                                    int_bins_flag=False, int_bins_flagx=False, int_bins_flagy=False,
                                    exclude_extremes=False,
                                    normy=False, normx = False,
                                    fx=1.5, fy=1.5, std_times=4, f=None,
                                    smooth = False,
                                    markertype=None,
                                    normr = False,
                                    colors = 'none',
                                    levels=False,
                                    level_color=None):
    """
    Plot a 2D histogram (heat map) with automatic binning.

    Accepts two arrays, a single array (treated as a time series), or a
    complex-valued array (real/imag mapped to x/y).

    Parameters
    ----------
    x : array-like or complex array
        X values, or y values if *y* is not given, or complex numbers
        where real part is x and imaginary part is y.
    y : array-like, optional
        Y values. Leave blank for time-series mode or complex input.
    log_colorbar_flag : bool, optional
        Use a log scale for the color bar. Default is False.
    maxx, maxy, minx, miny : float, optional
        Axis limits. Default is None (auto).
    int_bins_flag : bool, optional
        Force both x and y bin edges onto integers. Default is False.
    int_bins_flagx, int_bins_flagy : bool, optional
        Force only x or y bins onto integers. Default is False.
    exclude_extremes : bool, optional
        Exclude edge bins that collect data beyond *std_times*.
        Default is False.
    normy : bool, optional
        Normalize colors per y-slice. Default is False.
    normx : bool, optional
        Normalize colors per x-slice. Default is False.
    fx : float, optional
        Bin factor for x-axis. Default is 1.5.
    fy : float, optional
        Bin factor for y-axis. Default is 1.5.
    std_times : float, optional
        Number of standard deviations to display. Default is 4.
    f : float, optional
        Bin factor applied to both axes. Overrides *fx* and *fy*.
    smooth : float, optional
        Gaussian filter sigma in pixels. 0 or False for no smoothing.
        Default is False.
    markertype : str, optional
        Marker style to overlay data points (e.g. ``'.'``).
    levels : bool or list, optional
        If True, draw filled contours. If a list of values, draw contour
        lines at those percentile levels. Default is False.
    level_color : dict, optional
        Dict with ``'level'`` and ``'cmap'`` keys for filled contour
        coloring.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The current figure. Data is attached as ``fig.ndhist``, a dict
        with keys ``'counts'``, ``'bins_x'``, and ``'bins_y'``.

    Examples
    --------
    >>> x = np.random.randn(10000)
    >>> y = x + np.random.randn(10000)
    >>> fig = ndhist(x, y)

    >>> z = (5 + np.random.randn(1000)) * np.exp(1j * np.random.randn(1000))
    >>> fig = ndhist(z, smooth=1)
    """

    # convertes the counts into percentages
    def counts_to_pcnts(counts):
        flat_counts = flatten(counts)
        flat_norms = np.zeros(flat_counts.shape)
        cum_summ = 0
        for w in sorted(np.unique(flat_counts), reverse=True):
            I = (w == flat_counts)
            flat_norms[I] = cum_summ
            cum_summ += w * sum(I)
        flat_norms = 100 * flat_norms / cum_summ
        norms = unflatten(flat_norms, counts)

        return norms

    class nf(float):
        def __repr__(self):
            s = f'{self:.1f}'
            return f'{self:.0f}' if s[-1] == '0' else s


    if colors == 'none':
        if levels:
            if type(levels) == bool:
                colors = plt.cm.Greens
            else:
                colors = 'gray'

        else:
            colors = plt.cm.Greens_r

    if np.array(x).dtype == 'complex128':
        y = x.imag
        x = x.real

    # if you just passed a timeseries, then use the x as an index
    if not hasattr(y, '__len__'):
        y = copy.deepcopy(x)
        x = range(len(y))

    # check for incompatable user params
    if normy & normx:
        raise Warning("you can't normalize on both x and y at the same time, so we just normalized by X")
        normy = False

    if f:
        fx = f
        fy = f

    x = np.array(x)
    y = np.array(y)

    mask = np.isfinite(y + x)
    x = x[mask]
    y = y[mask]

    # if your limits are infinitiy, then choose the max/min of the data
    if minx == -np.inf:
        minx = np.min(x)
    if miny == -np.inf:
        miny = np.min(y)
    if maxx == np.inf:
        maxx = np.max(x)
    if maxy == np.inf:
        maxy = np.max(y)

    # if ylims == 'auto':
    #     cur_std = np.std(y)
    #     cur_mean = np.mean(y)
    #     mask = (y > cur_mean - std_times * cur_std) & (y < cur_mean + std_times * cur_std)
    #     x = x[mask]
    #     y = y[mask]
    #

    #  choose the first result, then the first answer in it


    if int_bins_flag:
        int_bins_flagx = True
        int_bins_flagy = True

    # Better to implement choosebins base to work with one rather than an array
    binsx = choose_bins([x], minx=minx, maxx=maxx, int_bins_flag=int_bins_flagx, bin_factor=fx, exclude_extremes=exclude_extremes, std_times=std_times)[0][0]
    binsy = choose_bins([y], minx=miny, maxx=maxy, int_bins_flag=int_bins_flagy, bin_factor=fy, exclude_extremes=exclude_extremes, std_times=std_times)[0][0]

    counts, bins_x, bins_y = np.histogram2d(x, y, (binsx, binsy))

    if smooth:
        counts = gaussian_filter(counts, smooth)

    to_plot = counts.transpose()
    if log_colorbar_flag:
        # todo: change this to be actually adjusting the colorbar, not the counts
        to_plot = log10(1 + to_plot)
        # maybe this isn't needed? Counts cant be less than 0, so the log10(1) must be zero
        to_plot[to_plot == -np.inf] = -np.max(counts) / len(x)
        if levels:
            #ToDo: Make % bins work with log colorbar, switching back to regular plot
            logging.warning("Your chosen levels are being ignored because of the log_colorbar_flag")
            levels = 'none'

    #     else:
    #         to_plot[to_plot==-np.inf] = -np.max(counts) / len(x)
    # normalize by x or y
    if normx:
        to_plot = to_plot / (eps + np.max(to_plot, axis=0))
    elif normy:
        to_plot = (to_plot.T/(np.max(to_plot, axis=1) + eps)).T

    if normr:
        # R2 = X. ^ 2 + Y. ^ 2; % R. ^ 2
        X, Y = np.meshgrid(binsx, binsy)
        R2 = np.power(X, 2) + np.power(Y, 2)
        R = np.sqrt(R2)# R^2
        to_plot = to_plot * R


    # eventually have this be the default if it can be sped up
    if levels:
        to_plot = counts_to_pcnts_fast(to_plot)
        dsx = np.diff(bins_x)[1]
        dsy = np.diff(bins_y)[1]

        bx = np.append(bins_x[1:-1] - dsx / 2, np.array([bins_x[-2] + dsx / 2]))
        by = np.append(bins_y[1:-1] - dsy / 2, np.array([bins_y[-2] + dsy / 2]))

        if isinstance(levels, bool):

            plt.contourf(bx, by, to_plot, cmap=colors, levels=100)
        else:
            # fig, ax = plt.subplots()
            ax = plt.gca()
            CS = plt.contour(bx, by, to_plot, colors=colors, levels=levels)
            CS.levels = [nf(val) for val in CS.levels]
            ax.clabel(CS, CS.levels, inline=True, fmt='%r %%', fontsize=10, colors = 'k')
            if level_color is not None:
                plt.contourf(bx, by, to_plot, levels=[0, level_color['level']], cmap=level_color['cmap'])

        # fig, ax = plt.subplots()

    else:
        plt.pcolor(bins_x, bins_y, to_plot, cmap=colors)

    plt.xlim(np.array(bins_x)[[1, -2]])
    plt.ylim(np.array(bins_y)[[1, -2]])

    if markertype != None:
        plt.plot(x, y, markertype)

    # plt.axis('tight')
    fig = plt.gcf()
    fig.ndhist = {'counts': counts, 'bins_x': bins_x, 'bins_y': bins_y}
    return fig


# this version of the algorithm is very fast - but will artificially include differences in colors
# could rewrite it somehow to use this and be faster:
# etl.start_and_ends(diff(counts_ordered) < 0)
def counts_to_pcnts_fast(counts):
    flat_counts = flatten(counts)
    # sort the counts,            [in reverese]
    idxs = np.argsort(flat_counts)[::-1]
    idxs_undo = np.argsort(idxs)
    # flip from greatest to smallest
    counts_ordered = flat_counts[idxs]
    # this is counting the contents of the bins in order from fullest to least full
    counts_summed = np.cumsum(counts_ordered)
    # convert the counts to a percent of the total N items histogrammed
    flat_norms = 100 * counts_summed / sum(counts_ordered)

    # get unique values of number of elements in each bin
    unique_values = np.unique(counts_ordered)

    # for each unique bin value, avg the pcnts and assign that avg for every index with that value
    for value in unique_values:
        indices_to_avg = np.where(counts_ordered == value)
        avg_pcnt = np.mean(flat_norms[indices_to_avg])
        flat_norms[indices_to_avg] = avg_pcnt

    # reverse the operations to get back to our original orders and shape
    norms_unsorted = flat_norms[idxs_undo]
    norms = unflatten(norms_unsorted, counts)
    return norms

def choose_bins(X, min_bins=10, max_bins=175, bin_factor=1.5, sameBinsFlag=False, std_times=4, minx=None, maxx=None,
                int_bins_flag=None, exclude_extremes=True):

    # Uses Scott's normal reference rule to select the number of bins
    # https://en.wikipedia.org/wiki/Histogram#Scott's_normal_reference_rule
    # If multiple sets of data are passed then it makes sensible choices

    S = len(X)

    minS = np.zeros(S)
    maxS = np.zeros(S)

    stdV = [np.std(w) for w in X]
    meanV = [np.mean(w) for w in X]
    for k in range(S):
        Values = X[k]
        # set x MIN values
        very_small_num = 0.000000000000001
        if minx is None: # user did not specify - then we need to find the minimum x value to use
            if stdV[k]>0.000000000000001: # just checking there are more than two different points to the data, checking for rounding errors.
                leftEdge = meanV[k]-stdV[k]*std_times
                if leftEdge<np.min(Values): # if the std is larger than the largest value
                    minS[k]=np.min(Values)
                else: # cropp it now on the bottom.
    #             cropped!
                    minS[k] = leftEdge
            else: # stdV==0, wow, all your data points are equal
                minS[k]=np.min(Values)-0.000000000000001 # padd it by 100, seems reasonable
        else: # minX is specified so minS is just set stupidly here
            if minx < np.max(Values):
                minS[k] = minx
            else: # ooh man, even your biggest value is smaller than your min
                logging.warning(['user parameter minx=' + str(minx) + ' is being overriden since it put all your data out of bounds'])
                minS[k] = np.min(Values)

    # set x MAX values
        if maxx is None:
            if stdV[k]>very_small_num: # just checking there are more than two different points to the data
                rightEdge = meanV[k]+stdV[k]*std_times
                if rightEdge>np.max(Values): # if the suggested border is larger than the largest value
                    maxS[k]=np.max(Values)
                else: # crop the graph to cutoff values
                    maxS[k]=rightEdge
            else: # stdV==0, wow,
    #           Note that minX no longer works in this case.
                maxS[k]=np.max(Values)+very_small_num # padd it by 100, seems reasonable
        else: # maxX is specified so minS is just set here
            maxS[k]=maxx;
            if maxx>np.min(Values):
                maxS[k]=maxx
            else: # ooh man, even your smallest value is bigger than your max
                maxS[k]=np.max(Values)
                # warning(['user parameter maxx=' num2str(maxX) ' override since it put all your data out of bounds']);
    #

    # if intbins(k)
    #     maxS(k)=round(maxS(k))+.5;
    #     minS(k)=round(minS(k))-.5; # subtract 1/2 to make the bin peaks appear on the numbers.
    # end

    # need the isData boolean to so you are only looking at the ones that actually have some data
    x_min = np.min(minS)
    x_max = np.max(maxS)

# % note that later there will be a bit added to maxS of SXRange
# % This below is to get estimates for appropriate binsizes
# totalRange=diff(SXRange); % if the range is zero, then make it eps instead.


    total_range = x_max - x_min

    if int_bins_flag is not None:
        int_bins = [int_bins_flag for x in X]
    else:
        int_bins = [isdiscrete(x) for x in X]

    # scotts choice bin width
    bin_widths = [3.5 * np.std(x) / (bin_factor * np.power(len(x), 1.0 / 3)) for x in X]

  # Instate a mininum and maximum number of bins
    num_bins = [1.0 * total_range/bin_width for bin_width in bin_widths] # Approx number of bins
    num_bins = np.round(num_bins)
    for k in range(S):
        if num_bins[k]<min_bins: # if this will imply less than 10 bins
            num_bins[k] = min_bins
            bin_widths[k] = total_range / (min_bins) # set so there are ten bins

        if num_bins[k]>max_bins: # if there would be more than 175 bins (way too many)
            num_bins[k] = max_bins
            bin_widths[k]=total_range / max_bins

        # This will make the span of bins have edges that go up to and include the extreme-most data
        bin_widths[k] = total_range / num_bins[k] + eps * 100

#   Check if it is intbins, because then:
        if int_bins[k]: # binwidth must be an integer, and it must be at least 1
            bin_widths[k] = np.max([np.round(bin_widths[k]), 1])
            x_min = np.floor(x_min)


##   if there is enough space to plot them, then plot vertical lines.
## 30 bins is arbitrarily chosen to be the number after which there are
## vertical lines plotted by default
    # if n_bins(k)<30:
    #     vertLinesArray(k)=1;
    # else
    #     vertLinesArray(k)=0;
## fix the automatic decision if vertical lines were specified by the user
# if vertLinesForcedFlag
#     vertLinesArray=vertLinesArray*0+vertLinesFlag;

# only plot lines if they all can be plotted.
# also creates one flag, so the 'array' does not need to be used.
# vertLinesFlag=prod(vertLinesArray); # since they are zeros and 1's, this is an "and" statement

# find the maximum bin width
    big_bin_width=np.max(bin_widths)

##  resize bins to be multiples of each other - or equal
# sameBinsFlag will make all bin sizes the same.

    # Note that in all these conditions the largest histogram bin width
    # divides evenly into the smaller bins. This way the data will line up and
    # you can easily visually compare the values in different bins
    if sameBinsFlag: # if 'same' is passed then make them all equal to the average recommended size
        only_bin_width = np.mean(bin_widths)
        if int_bins_flag:
            only_bin_width = np.round(only_bin_width)
        bin_widths=[only_bin_width] * len(bin_widths)
    else: # the bins will be different if neccesary
        for k in range(len(X)):
    #       The 'ceil' rather than 'round' is supposed to make sure that the
    #       ratio is at lease 1 (divisor at least 2).
            bin_widths[k]=big_bin_width/np.ceil(big_bin_width/bin_widths[k])

    # SXRange(2) = SXRange(2)+max(binWidth)
    if exclude_extremes:
        bins = [np.arange(x_min, x_max + big_bin_width, w) for w in bin_widths]
    else:
        bins = [np.concatenate([[-np.inf], np.arange(x_min, x_max + big_bin_width, w), [np.inf]]) for w in bin_widths]

    return bins, bin_widths


def dictify_cols2(df):
    return df.groupby(df.columns[0])[df.columns[1]].apply(lambda w: w.values).to_dict()


# This will tell if the data is an integer or not.
# first it will check if the variable is integer, but even so, they
# might be integers stored as floats!
def isdiscrete(x,min_error='default'): # L stands for logical
    if min_error=='default':
        min_error=np.spacing(1)*100 # the minimum average difference from integers they can be.

    if isinstance(x,int):# your done, duh, its an int.
        return True
    else:
        return np.sum(np.abs(x-np.round(x)))/len(x)<min_error


def test_nhist_list():
    # import numpy as np
    # from viz.histogram_utils import nhist
    y = np.random.randn(100) + 2
    nhist(y)

    A = [np.random.randn(10 ** 5), np.random.randn(10 ** 3) + 1]
    nhist(A, legend = [r'$\mu$=0', r'$\mu$=1'])

def test_nhist_dict():
    A = {'mu_is_Zero': np.random.randn(10 ** 5), 'mu_is_Two': np.random.randn(10 ** 3) + 2}
    _ = nhist(A)
    _ = nhist(A, color='viridis')
    _ = nhist(A, f=4)
    _ = nhist(A, same_bins_flag=True)
    _ = nhist(A, noerror=True)

# Use ndhist without specifying an x axis.
def test_ndhist_timeseries():
    n = 10000
    y = np.cumsum(np.random.randn(n)) + 15 * np.random.randn(n)
    _ = ndhist(y, fx=5)
    plt.xlabel('sample number')

# This one looks kind of like a black hole
def test_ndhist_complex():
    n = 10000
    z = (5 + np.random.randn(n)) * np.exp(1j * (np.random.randn(n) + np.pi / 4))
    _ = ndhist(z, smooth=1)
    plt.colorbar()

# use the log colorbar to manage varying
def test_ndhist():
    x = np.concatenate([np.ones(500), np.random.randn(100000), 4 + np.random.randn(1000) / 2])
    y = np.concatenate([np.ones(500), np.random.randn(100000), 3 + np.random.randn(1000) / 2])
    counts, bins_x, bins_y = ndhist(x, y, log_colorbar_flag=True)



if __name__=='main':
    pass

