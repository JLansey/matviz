
import os,sys,csv


# regular pythong stuff
# redundant form interactive computing
import collections
from collections import defaultdict
from collections import Sequence
from itertools import chain, count
from operator import itemgetter
import glob
import json


# datetime related things
from dateutil.parser import parse as dateutil_parse
import matplotlib.dates as md # for handledates
import datetime
from pytz import timezone


# regular anaconda stuff
import numpy as np
from scipy import signal
import pandas as pd


# useful stuffs:
from numpy import diff


data_dir = 'data'
fig_dir = 'figs'

eps = np.spacing(1)



def time_delta_to_days(w):
    return w / np.timedelta64(1, 'D')



def get_object_size(obj):
    the_size = sys.getsizeof(obj)*1e-6
    print("Object size in MB: {0:.2f}".format(the_size))
    # return the_size


def pprint_entire_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)



def sql(query, db):
    """
    Parameters:
        query: A sql query written in normal sql language inside quotes,
            type = string
        db: from_db.sql_connection()

    Returns:
        a dictof sql data
    """
    db.execute(query)
    results = db.fetchall()

    return results

def sql_redshift(query,db,array):
    db.execute(query,array)
    results = db.fetchall()
    db.close()
    return results


def subsetter(results,vars):
    # in case you only want a single variable returned
    # is there a check for just a scalar??
    # Note that this subsetter only works for resultsets

    if type(vars) not in [list,set]:
        vars = [vars]

    if len(results)>0:
        return [{k:v for k,v in dict(cur_row).items() if k in vars} for cur_row in results]
    else:
        return []



def remove_tz(cur_datetime):
    if hasattr(cur_datetime,"__len__"):
        naive_datetime = [w.replace(tzinfo=None) for w in cur_datetime]
    else:
        naive_datetime = cur_datetime.replace(tzinfo=None)
    return naive_datetime

def tz_to_utc(cur_datetime,local_tz='US/Eastern',native=True):
    local_datetime = timezone(local_tz).localize(cur_datetime)
    utc_datetime = local_datetime.astimezone(timezone('UTC'))

    if native:
        return utc_datetime.replace(tzinfo=None)
    else:
        return utc_datetime

def utc_to_tz(cur_utc,local_tz='US/Eastern'):
    # this is a UTC time but without timezone information
    utc_datetime = timezone('UTC').localize(cur_utc)
    local_datetime = utc_datetime.astimezone(timezone(local_tz))
    naive_datetime = local_datetime.replace(tzinfo=None)
    return naive_datetime

def to_tz(cur_tz,local_tz='US/Eastern'):
    # if there is already time zone information associated here, then just go with it
    local_datetime = cur_tz.astimezone(timezone(local_tz))
    naive_datetime = local_datetime.replace(tzinfo=None)
    return naive_datetime

def round_time(ts, round_by='H'):
    ts_no_tz = remove_tz(ts)
    return pd.Series(ts_no_tz).dt.round(round_by)



def average_times(time_1,time_2):
    dummy_time = datetime.datetime(2000, 1, 1, 0, 0)
    return dummy_time + datetime.timedelta(seconds=((time_1-dummy_time).total_seconds()+(time_2-dummy_time).total_seconds())/2)




def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]


def parse_min_sec(time_str):
    '''
    convert normal times into seconds
    gosh, surprising that there wasn't already some way to do this robustly
    in python. Note that this does not work if you've got hours
    '''
    min_sec = time_str.split(":")
    if len(min_sec)>2:
        raise Exception("we don't do hours yet folks")
    elif len(min_sec)>1:
        minn = 60*float(min_sec[0])
        secc = min_sec[1]
    else:
        minn = 0
        secc = min_sec[0]
    return minn + float(secc)



def clean_whitespace(my_str):

    my_str = my_str.replace('\n','')
    my_str = my_str.replace('\r','')
    my_str = my_str.replace('\t',' ')
    for ii in range(10):
        my_str = my_str.replace('  ',' ')

    if my_str:
        if my_str[0] == ' ':
            my_str = my_str[1:]
        if my_str[-1] == ' ':
            my_str = my_str[:-1]

    return my_str

# This essentially splits a giant list by
# some function.
def full_group_by(l, key=lambda x: x):
    d = defaultdict(list)
    for item in l:
        d[key(item)].append(item)
    return d


def read_csv(name,qt = 1):
  return list(csv.reader(open(name),quoting = qt))

def write_csv(name,array,param='w'):
    with open(name, param) as f:
        writer = csv.writer(f,quoting =1)
        writer.writerows(array)
    return True


def write_csv2(name,array,param='w'): # a for append
    with open(name, param) as f:
        writer = csv.writer(f,quoting =1)
        writer.writerows(array)
    return True


# accept a data frame with only two columns, and make first the key and second the value
# Unsure when one or both of these functions work ..
def dictify_cols(df):
    return df.set_index(df.columns[0])[df.columns[1]].to_dict()

def dictify_cols2(df):
    return df.groupby(df.columns[0])[df.columns[1]].apply(lambda w: w.values).to_dict()


def dictify_csv(weird_array,headers = None):
    # turns a csv into a dictionary with long columns
    '''
    Args:
        weird_array: just an array with items in the rows

    Returns:
        a dictionary where the header row becomes the keys for the dict
    '''
    if headers:
        dicto = dictify_csv([headers]+weird_array)
    else:
        dicto =  {w[0]:w[1:] for w in np.transpose(weird_array)}

    return dicto


def csvify_dict(weird_dict):
    '''
    # turn a dictionary with long columns into a thing you can print
    Args:
        weird_dict:
         just a dictionary

    Returns:
        a list of lists ready to be turned into a csv

    '''
    # initialize it to be on more
    headers = []
    csvo = [[] for w in next(iter(weird_dict.values()))] # [[]]+
    for k,v in weird_dict.items():
        headers.append(k)
        for ii in range(len(v)):
            csvo[ii].append(str(v[ii]))

    return [headers]+csvo



def nan_smooth(y,n=5,ens=[],ignore_nans=True):
    """
    Args:
        y: your timeseries (don't need x)
        n: int-> window is hanning length n
           list-> window is exactly the list you pass,[1,1,1,1]/4 for moving average
        ens: weighting of the points in 'y' in
        ignore_nans: if you want to ignore nans, or exclude data that has nans in it

    Returns:
        Smoothed values, centered, and same dimension as input

    """

    # just in case you didn't pass in a numpy datatype
    y = np.array(y)
    ens = np.array(ens)

    # handle weird cases
    if len(y)<3:
        raise Warning('Sorry bud, you can''t smooth less than 3 points, thats silly')
        return y

    if len(ens)==0:
        ens = np.ones(len(y))
        ens[np.isnan(y)]=0

    elif len(ens)!=len(y):
        raise Exception('ens must be the same length as y')

    # user did not pass a window function in
    if np.isscalar(n):
        if n<2:
            # windows size is 1, that is silly, no smoothing is happenening
            return y
        # create a nice smoothing window
        # normal distribution with 95% confidence bounds
        # tmp = np.linspace(-1.96,1.96,n)
        # window = np.exp(-np.power(tmp,2))/sum(np.exp(-np.power(tmp,2)))
        # so that n=1 represents window [0 1 0]
        window = signal.hann(n+2)
    else:
        window = n
        if round(sum(window)*100000)/100000 != 1:
            raise Warning('the sum of your window does not equal to one. Your smoother will be biased')
        n = len(window)

    if len(y)<=n:
        raise Warning('Sorry bud, you can''t smooth that, pick a smaller n')
        return y

    # ignore those nans
    if ignore_nans:
        ens[np.isnan(y)]=0
        y[ens==0]=0
    else:
        y[ens==0]=np.nan

    # yup, these three lines do all the real work ... VERY FAST with FFT
    ys = signal.convolve(y*ens, window, mode='same')
    enss = signal.convolve(ens, window, mode='same')
    # normalize things.
    outt = ys/enss


    #  defining a subfunction in case you want to test it out
    def test_nan_smooth():

        N2=50
        x2 = np.linspace(0,8*np.pi,N2)
        ens2 = np.random.randint(0,30,N2)+20
        noise2 = [np.mean(np.random.normal(0,1,w)) for w in ens2]
        sig2 = np.sin(x2)
        y2 = sig2 + np.array(noise2)

        #####
        win_width = 18
        # plot.clf()
        # plot(x2,y2,'o-',color=".65",lw=4)
        # plot(x2,sig2,'r',lw=3)
        # plot(x2,nan_smooth(y2,[],np.ones(win_width)/win_width),'g',lw=3)
        # plot(x2,nan_smooth(y2,ens2,win_width),'b',lw=3)
        # grid(False)
        # legend(['noisy signal','true signal','moving average','nan_smoothed'])
        # nicefy()

    return outt



# normalized version of cross correlation - mimiking matlabs'
def xcorr(a, b, ds):
    S = len(a)
    a_norm = (a - np.mean(a)) / np.std(a)
    b_norm = (b - np.mean(b)) / np.std(b)
    corrs = np.correlate(a_norm, b_norm / S, 'full')

    lags_half = np.arange(0, ds * S, ds)
    lags = np.concatenate([-np.flip(lags_half[1:]), lags_half])

    return corrs, lags



def reverse_dict(tmp_dict):
    return {v: k for k, v in tmp_dict.items()}


def recurse_func(my_list,my_func,stop_level=False):
    """
    Will compute some function, at some level down.
    Args:
        my_list:
        my_func:
        stop_level:

    Returns:

    """
    if list_depth(my_list) == stop_level:
        return my_func(my_list)
    elif hasattr(my_list,"__len__"):
        return [recurse_func(w,my_func,stop_level) for w in my_list]
    else:
        return my_func(my_list)
# http://stackoverflow.com/questions/6039103/counting-deepness-or-the-deepest-level-a-nested-list-goes-to

def list_depth(seq):
    seq = iter(seq)
    try:
        for level in count():
            seq = chain([next(seq)], seq)
            seq = chain.from_iterable(s for s in seq if isinstance(s, Sequence))
    except StopIteration:
        return level

def flatten_list(my_list):
    return list(flatten(my_list))

def flatten(my_list):
    for el in my_list:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            for sub in flatten(el):
                yield sub
        else:
            yield el
# http://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists-in-python/2158532#2158532


# convert dates to numbers, probably needs improvement or a complete overhaul, maybe moved to etl

def handle_dates(X):
    # in case you passed a date object better convert that
    if isinstance(X[0][0],datetime.time):
        start_day = datetime.date(2000, 1, 1)
        # HOURS_PER_DAY = 24
        # MINUTES_PER_DAY = 24*60
        # X = [[(w.hour / HOURS_PER_DAY + w.minute / MINUTES_PER_DAY) for w in x] for x in X]
        X = [[md.date2num(datetime.datetime.combine(start_day,w)) for w in x] for x in X]
        datetime_flag = True
        date_formatting = '%H:%M:%S'
    else:
        if isinstance(X[0][0],datetime.datetime) or isinstance(X[0][0],datetime.date):
            X = [md.date2num(x) for x in X]
            datetime_flag = True

            global_min = min(np.concatenate(X))
            global_max = max(np.concatenate(X))
            global_range = global_max - global_min
#           if the range is greater than 3 days
            if global_range > 3:
                date_formatting = '%Y-%m-%d'
            else:
                date_formatting = '%Y-%m-%d %H:%M:%S'

        else:
            datetime_flag = False
            date_formatting = 'N/A'
    return X, datetime_flag, date_formatting



def start_and_ends(logical_array):
    """
     Return the starts and end times for when the logical
     array True
    :param logical_array:
    :return:
    list of (start,end) tuples of the indexes

    Note: if the array starts with a [True, False], you completely
          miss it because it technically *ended at that point
          and started before the logical array began
          If the array starts with [True,True, False]
          then you get [(0,1),...]
    #

    """

    # Padd the array with Falses to get the ends
    padded_array = np.concatenate(([False],logical_array,[False]))

    #
    idxs = np.array(range(len(padded_array)-1))
    differences = np.diff([np.int(w) for w in padded_array])
    starts = idxs[differences>0]
    ends   = idxs[differences<0]

    # we added an element, now we take it away
    starts_shift = np.maximum(starts-1,0)
    # easier than doing a check if its empty
    ends_shift   = np.maximum(ends-1,0)

    return list(zip(starts_shift,ends_shift))


def chop(seq, size):
    """Chop a sequence into chunks of the given size."""
    chunk = lambda ii: seq[ii:ii+size]
    return map(chunk,range(0,len(seq),size))

def chopn(seq, n):
    """Chop a sequence into chunks of the given size."""
    size = int(len(seq) / n)
    chunks = chop(seq, size)
    return [w for w in chunks if len(w) == size]



def form_day(key):
    return str(key.month) + "/" + str(key.day)

def form_year(key):
    return str(key.year) + "-" + str(key.month)


def rolling_diff(w, n=1):
    return w - np.roll(w,n)



def most_common(cur_list):
    return collections.Counter(cur_list).most_common()


def find_dom_freq(x, ds, window = 'hann'):
    freq, power = signal.periodogram(x, 1 / ds, window=window)
    peak_freq = freq[power == np.max(power)].mean()
    return peak_freq


def sort_dict_list(dict_list, k, reverse_param = True):
    return sorted(dict_list, key=itemgetter(k), reverse=reverse_param)

def robust_mkdir(cur_dir):
    if not os.path.exists(cur_dir):
        os.mkdir(cur_dir)
