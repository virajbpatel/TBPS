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
    
def d2gamma_p_d2q2_dcostheta(fl, afb, cos_theta_l):
    """
    Returns the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param cos_theta_l: cos(theta_l)
    :return:
    """
    ctl = cos_theta_l
    c2tl = 2 * ctl ** 2 - 1
    scalar_array = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) + 8/3 * afb * ctl) * acceptance_func(ctl)
    return scalar_array

def norm_d2gamma_p_d2q2_dcostheta(fl, afb, cos_theta_l):
    """
    Returns the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param cos_theta_l: cos(theta_l)
    :return:
    """
    scalar_array = d2gamma_p_d2q2_dcostheta(fl, afb, cos_theta_l)
    norm_factor = integrate(cos_theta_l, d2gamma_p_d2q2_dcostheta(fl=fl, afb=afb, cos_theta_l=cos_theta_l))
    normalised_scalar_array = scalar_array /norm_factor  # normalising scalar array to account for the non-unity acceptance function
    return normalised_scalar_array

def log_likelihood(fl, afb, _bin):
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    :return:
    """
    _bin = bins[int(_bin)]
    ctl = _bin['costhetal']
    normalised_scalar_array = norm_d2gamma_p_d2q2_dcostheta(fl=fl, afb=afb, cos_theta_l= ctl)
    return - np.sum(np.log(normalised_scalar_array))

"""
Some example code below used below. It is currently commented out, and should be commented out 
before using the Module to prevent running the code below uneccessarily. Requires user to have
filt_frst_3101.pkl downloaded in the same working folder.
"""
# df = pd.read_pickle('filt_frst_3101.pkl')
# _bins = create_sorted_q2_bins(df)
# print('bin 0')
# print(_bins[0].q2.head())
