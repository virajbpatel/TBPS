# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 13:16:47 2022

@author: kevin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit as minuit
import module_fit_functions as mfit
import MODULE_curve_fitting_functions as mcf

def read_data_files(*data_files):
    return data_files

data_files = read_data_files('total_filtered', 'acceptance_filtered')
var_to_read = 2

var = ['phi', 'costhetal', 'costhetak', 'q2']   
# df = pd.read_pickle(f'{data_files[0]}.pkl')
# bins = mcf.create_sorted_q2_bins(df)
# df_mc = pd.read_pickle(f'{data_files[1]}.pkl')
# bins_mc = mcf.create_sorted_q2_bins(df_mc)

df = pd.read_csv(f'{data_files[0]}.csv')
bins = mcf.create_sorted_q2_bins(df)
df_mc = pd.read_csv(f'{data_files[1]}.csv')
bins_mc = mcf.create_sorted_q2_bins(df_mc)

i = var[1]

def acceptance_func(x):
    y = mfit.polynomial(x, *params_acc)
    return y

def integrate(x,y):
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

def d2gamma_p_d2q2_dcosthetal(fl, afb, cos_theta_l):
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

def norm_d2gamma_p_d2q2_dcosthetal(fl, afb, cos_theta_l):
    """
    Returns the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param cos_theta_l: cos(theta_l)
    :return:
    """
    scalar_array = d2gamma_p_d2q2_dcosthetal(fl, afb, cos_theta_l)
    cos_theta_l_acc = np.linspace(-1,1,10000)
    norm_factor = integrate(cos_theta_l_acc, d2gamma_p_d2q2_dcosthetal(fl=fl, afb=afb, cos_theta_l=cos_theta_l_acc))
    normalised_scalar_array = scalar_array /norm_factor  
    return normalised_scalar_array

def log_likelihood_ctl(fl, afb, _bin):
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    :return:
    """
    _bin = bins[int(_bin)]
    ctl = _bin['costhetal']
    normalised_scalar_array = norm_d2gamma_p_d2q2_dcosthetal(fl=fl, afb=afb, cos_theta_l= ctl)
    return - np.sum(np.log(normalised_scalar_array))


#%%
bin_width = 25
fl_array = []
afb_array = []
for b in range(10):
    n, edge, patch = plt.hist(bins[b][i], bins = bin_width, density = True, histtype = 'step', label = 'total data')
    n_mc, edge_mc, patch_mc = plt.hist(bins_mc[b][i], bins = 10, density = True, histtype = 'step', label = 'acceptance data')

    x_acc, n_acc, params_acc, cov_acc = mfit.accept_fit(n_mc, edge_mc)

    plt.title(f'{data_files[0]} bin {b}')
    plt.plot(x_acc, n_acc, label = 'fitted acceptance function')
    plt.xlabel(i)
    plt.ylabel('Prob density')
    plt.legend()
    plt.clf()
    plt.show()
        
    log_likelihood_ctl.errordef = minuit.LIKELIHOOD
    decimal_places = 3
    starting_point = [0,0]
    fls, fl_errs = [], []
    afbs, afb_errs = [], []
    m = minuit(log_likelihood_ctl, fl=starting_point[0], afb=starting_point[1], _bin = b)
    m.fixed['_bin'] = True  # fixing the bin number as we don't want to optimize it
    m.limits=((-1.0, 1.0), (-1.0, 1.0), None)
    m.migrad()
    m.hesse()
    bin_results_to_check = m
    fls.append(m.values[0])
    afbs.append(m.values[1])
    fl_errs.append(m.errors[0])
    afb_errs.append(m.errors[1])
    print(f"Bin {b}: Fl = {np.round(fls, decimal_places)} pm {np.round(fl_errs, decimal_places)},", 
          f" Afb = {np.round(afbs, decimal_places)} pm {np.round(afb_errs, decimal_places)}. Function minimum considered valid: {m.fmin.is_valid}")
    
    h, e, ppp = plt.hist(bins[b][i].sort_values(), bins = bin_width, density = True, histtype = 'step', color = 'blue')
    plt.errorbar((e[:-1] + 0.5*e[1] - 0.5*e[0]), h, yerr = np.sqrt(h * len(bins[b][i]))/len(bins[b][i]), fmt = '.', color = 'blue')
    plt.plot(bins[b][i].sort_values(), norm_d2gamma_p_d2q2_dcosthetal(fls[0], afbs[0], bins[b][i].sort_values()), 
             label = f'fl =  {np.round(fls[0], decimal_places)} pm {np.round(fl_errs[0], decimal_places)} Afb = {np.round(afbs[0], decimal_places)} pm {np.round(afb_errs[0], decimal_places)}')
    plt.title(f'{data_files[0]} bin {b}')
    plt.legend()
    plt.xlabel(i)
    plt.ylabel('P')
    plt.ylim(0,1.5)
    plt.show()
    
    fl_array.append([fls[0], fl_errs[0]])
    afb_array.append([afbs[0], afb_errs[0]])

#%%