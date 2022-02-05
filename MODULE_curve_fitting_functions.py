# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:56:16 2022

@author: kevin

A module consisting some relevant functions to help with the curve fitting of LHCb data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

"""
Some example code below used below. It is currently commented out, and should be commented out 
before using the Module to prevent running the code below uneccessarily. Requires user to have
filt_frst_3101.pkl downloaded in the same working folder.
"""
# df = pd.read_pickle('filt_frst_3101.pkl')
# _bins = create_sorted_q2_bins(df)
# print('bin 0')
# print(_bins[0].q2.head())
