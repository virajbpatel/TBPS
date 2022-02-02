# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 02:15:16 2022

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

def polynomial(x,a,b,c,d,f,g,h):
    y = a*x**6 + b*x**5 + c*x**4 + d*x**3 + f*x**2 + g*x + h
    return y

def center_bins(bin_edge):
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