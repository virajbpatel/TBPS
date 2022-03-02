# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 16:18:32 2022

@author: user
"""
 #%%
import pandas as pd
import matplotlib.pyplot as pl
import numpy as np
from scipy.optimize import curve_fit
df = pd.read_pickle('0222_forest_total.pkl')["B0_MM"]
bins, bin_edges, _ = pl.hist(df, bins = 30)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
bin_width = bin_edges[2] - bin_edges[1]
pl.errorbar(bin_centres, bins, xerr = bin_width, yerr = None, fmt = ".", mew=2, color = "black")


def Gauss(x, A, mu, sigma):
    return A * np.exp((-0.5 * ((x - mu) / sigma)**2))

def exp_decay(x, B, lamda):
    return B * np.exp(-lamda * x)

def fitting_Func(x, A, mu, sigma, B, lamda):
    return Gauss(x, A, mu, sigma) + exp_decay(x, B, lamda)

#%%
x = np.arange(5100, 5700, 0.2)
popt, pcov = curve_fit(fitting_Func, bin_centres[3:], bins[3:], p0 = [800, 5280, 200, 800, 0.0001])
pl.errorbar(bin_centres, bins, xerr = bin_width, yerr = None, fmt = ".", mew=2, color = "black", label = "Data")
pl.plot(x, fitting_Func(x, *popt), color = "black", linewidth = 2.3, label = "Total dataset fit")
pl.fill_between(x, 0, exp_decay(x, popt[3], popt[4]), color='blue', alpha = 0.8, label = "Background")
pl.legend()

#%%
pl.plot(x, exp_decay(x, popt[3], popt[4]), color='blue', alpha = 0.8, label = "Background")
pl.plot(x, Gauss(x, popt[0], popt[1], popt[2]))
Signal = 0
Background = 0
for i in bin_centres:
    Signal += Gauss(i, popt[0], popt[1], popt[2])
    Background += exp_decay(i, popt[3], popt[4])
print(f"Total number of signal events: {Signal}")
print(f"Total number of Background events: {Background}")
print(len(df))




























