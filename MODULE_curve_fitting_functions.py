# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:56:16 2022

@author: kevin

A module consisting some relevant functions to help with the curve fitting of LHCb data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

##Changing fonts
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"


params = {
        'axes.labelsize':12,
        'axes.titlesize':12,
        'font.size':12,
        'figure.figsize': [7,4],
        'mathtext.fontset': 'stix',
        }
plt.rcParams.update(params)

"""
The binning scheme that is used, based on LHCb convention
"""
_q2_bin_scheme = [[0.1, 0.98], [1.1, 2.5], [2.5, 4.0], [4.0, 6.0], [6.0, 8.0], 
                 [15.0, 17.0], [17.0, 19.0], [11.0, 12.5], [1.0, 6.0], [15.0, 17.9]]

def sort_dataframe(dataframe, _var = 'q2', _min = 0, _max = 1):
    """
    Parameters
    ----------
    dataframe : Pandas DataFrame
        The pandas DataFrame that is being sliced and sorted according to the column heading '_var', 
        between _min and _max values in _var.
    _var : Str
        The column variable name as a string. The default is 'q2'.
    _min : Float
        The minimum value of _var being sliced. The default is 0.
    _max : Float
        The maximum value of _var being sliced. The default is 1.

    Returns
    -------
    new_dataframe : Pandas DataFrame
        Returns a new pandas DataFrame that is sliced, only containing entries with _var values between
        _min and _max. This DataFrame will be (usually) be smaller than the original DataFrame

    """
    new_dataframe = dataframe[(dataframe[_var] >= _min) & (dataframe[_var] <= _max)]
    return new_dataframe.sort_values(by = _var)




def create_sorted_q2_bins(dataframe):
    """
    Parameters
    ----------
    dataframe : Pandas DataFrame
        The pandas DataFrame that is being sorted in q2 bins according to _q2_binning_scheme.

    Returns
    -------
    _bins : List 
        A list of pandas DataFrame sorted according to _q2_binning_scheme.

    """
    _bins = []
    for i in _q2_bin_scheme:
        _bins.append(sort_dataframe(dataframe, _var = 'q2', _min = i[0], _max = i[1]))
    return _bins

def polynomial(x,a,b,c,d,f,g,h):
    """
    6th order polynomial equation
    """
    y = a*x**6 + b*x**5 + c*x**4 + d*x**3 + f*x**2 + g*x + h
    return y

def center_bins(bin_edge):
    """
    center the bin_edges 
    """
    bin_center = bin_edge[:-1] + (bin_edge[1] - bin_edge[0])/2
    return bin_center

def accept_fit(n, bins):
    """
    Parameters
    ----------
    n : float
        Bin heights from the histogram.
    bins : float
        Bin edges from the histogram. len(bins) should be len(n) + 1.
    order : int
        Order of polynomial being fitted to the datapoints.

    Returns
    -------
    Coefficients of polynomial of order = order.

    """
    x_data = center_bins(bins)
    params, cov = op.curve_fit(polynomial, x_data, n)
    
    x = np.linspace(x_data[0], x_data[-1], 100000)
    y = polynomial(x, *params)
    
    return x, y, params, cov
    
def integrate(x,y):
    """
    Integrate ydx, setting negative values of ydx to 0
    """
    da_list = []
    x = x.to_numpy()
    y = y.to_numpy()
    for i in range(len(x)-1):
        dx = x[i+1] - x[i]
        avg_y = (y[i+1] + y[i])/2
        da = avg_y*dx
        if da < 0:
            da = 0
        da_list.append(da)
    A = np.sum(da_list)
    return abs(A)

def integrate2(x,y):
    da_list = []
    for i in range(len(x)-1):
        dx = x[i+1] - x[i]
        avg_y = (y[i+1] + y[i])/2
        da = avg_y*dx
        if da < 0:
            da = 0
        da_list.append(da)
    A = np.sum(da_list)
    return abs(A)

def Gauss(x, A, mu, sigma):
    """
    Gaussian with height A, mean mu and standard deviation sigma
    """
    return A * np.exp((-0.5 * ((x - mu) / sigma)**2))

def exp_decay(x, B, lamda):
    """
    Exponential decay with initial value B and decay exponent lambda
    """
    exponent = np.longdouble(-lamda * x)
    return B * np.exp(exponent)

def gaussian_exponent_fit(x, A, mu, sigma, B, lamda):
    """
    A gaussian with exponential decay curve
    """
    return Gauss(x, A, mu, sigma) + exp_decay(x, B, lamda)

def read_data_into_bins(filtered_data, acceptance_data, file_type=0):
    """
    Read the filtered data and acceptance data in pkl = 0 or csv = 1, and returns
    filtered_dataframe, filtered_dataframe_in_q2_bins, acceptance_dataframe, acceptance_dataframe_in_q2_bins
    """
    if file_type == 0:
        file_type = 'pkl'
        df = pd.read_pickle(f'{filtered_data}.{file_type}')
        bins = create_sorted_q2_bins(df)
        df_mc = pd.read_pickle(f'{acceptance_data}.{file_type}')
        bins_mc = create_sorted_q2_bins(df_mc)
    elif file_type == 1:
        file_type = 'csv'
        df = pd.read_csv(f'{filtered_data}.{file_type}')
        bins = create_sorted_q2_bins(df)
        df_mc = pd.read_csv(f'{acceptance_data}.{file_type}')
        bins_mc = create_sorted_q2_bins(df_mc)
    else: 
        raise Exception('Wrong file type')
    return df, bins, df_mc, bins_mc

"""
Some example code below used below. It is currently commented out, and should be commented out 
before using the Module to prevent running the code below uneccessarily. Requires user to have
filt_frst_3101.pkl downloaded in the same working folder.
"""
# df = pd.read_pickle('filt_frst_3101.pkl')
# _bins = create_sorted_q2_bins(df)
# print('bin 0')
# print(_bins[0].q2.head())
