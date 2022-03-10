# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 13:16:47 2022

@author: kevin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit as minuit
import MODULE_curve_fitting_functions as mcf
from numpy.polynomial import legendre as lg
import matplotlib.patches as patches


def read_data_files(*data_files):
    return data_files

data_files = read_data_files('total_dataset_binary3', 'acceptance_mc_binary3')
var_to_read = 2

#standard model values for params
sm_fl = [0.296448, 0.760396, 0.796265, 0.711290, 0.606965, 0.348441, 0.328081, 
         0.435190, 0.747644, 0.340156]
var = ['phi', 'costhetal', 'costhetak', 'q2']   
df = pd.read_pickle(f'{data_files[0]}.pkl')
bins = mcf.create_sorted_q2_bins(df)
df_mc = pd.read_pickle(f'{data_files[1]}.pkl')
bins_mc = mcf.create_sorted_q2_bins(df_mc)

# df = pd.read_csv(f'{data_files[0]}.csv')
# bins = mcf.create_sorted_q2_bins(df)
# df_mc = pd.read_csv(f'{data_files[1]}.csv')
# bins_mc = mcf.create_sorted_q2_bins(df_mc)

i = var[2]

def acceptance_func(x):
    y = lg.legval(x, params_acc)
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

def d2gamma_p_d2q2_dcosthetak(fl, cos_theta_k):
    """
    Returns the pdf for theta_K

    Parameters
    ----------
    fl : float
        F_L observable.
    afb : float
        A_FB observable
    cos_theta_k : float
        angular variable.
    """
    ctk = cos_theta_k
    s2tk = 1 - ctk**2 # sine-squared term
    scalar_array = (3/2) * (fl * ctk**2) + (3/4) * (1 - fl) * s2tk
    return scalar_array

def norm_d2gamma_p_d2q2_dcosthetak(fl, cos_theta_k):
    """
    theta_K pdf but normalised to 1

    Parameters
    ----------
    see above

    """
    scalar_array = d2gamma_p_d2q2_dcosthetak(fl, cos_theta_k)
    cos_theta_k_acc = np.linspace(-1, 1, 10000)
    norm_factor= integrate(cos_theta_k_acc, d2gamma_p_d2q2_dcosthetak(fl=fl, 
                                                cos_theta_k=cos_theta_k_acc))
    normalised_scalar_array = scalar_array / norm_factor
    return normalised_scalar_array

def log_likelihood_ctk(fl, _bin):
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param _bin: number of the bin to fit
    :return:
    """
    _bin = bins[int(_bin)]
    ctk = _bin['costhetak']
    normalised_scalar_array = norm_d2gamma_p_d2q2_dcosthetak(fl=fl, cos_theta_k= ctk)
    return - np.sum(np.log(normalised_scalar_array))
#%%
bin_width = 25
fl_array = []
afb_array = []
for b in range(10):
    n, edge, patch = plt.hist(bins[b][i], bins = bin_width, density = True, histtype = 'step', label = 'total data')
    n_mc, edge_mc, patch_mc = plt.hist(bins_mc[b][i], bins = 10, density = True, histtype = 'step', label = 'acceptance data')

    x_acc, n_acc, params_acc, cov_acc = mcf.accept_fit(n_mc, edge_mc)

    plt.title(f'{data_files[0]} bin {b}')
    plt.plot(x_acc, n_acc, label = 'fitted acceptance function')
    plt.xlabel(i)
    plt.ylabel('Prob density')
    plt.legend()
    plt.clf()
    plt.show()
        
    log_likelihood_ctk.errordef = minuit.LIKELIHOOD
    decimal_places = 3
    starting_point = [0,0]
    fls, fl_errs = [], []
    afbs, afb_errs = [], []
    m = minuit(log_likelihood_ctk, fl=starting_point[0], _bin = b)
    m.fixed['_bin'] = True  # fixing the bin number as we don't want to optimize it
    m.limits=((-1.0, 1.0), None)
    m.migrad()
    m.hesse()
    bin_results_to_check = m
    fls.append(m.values[0])
    fl_errs.append(m.errors[0])
    print(f"Bin {b}: Fl = {np.round(fls, decimal_places)} pm {np.round(fl_errs, decimal_places)},", 
          f"Function minimum considered valid: {m.fmin.is_valid}")
    
    h, e, ppp = plt.hist(bins[b][i].sort_values(), bins = bin_width, density = True, histtype = 'step', color = 'blue')
    plt.errorbar((e[:-1] + 0.5*e[1] - 0.5*e[0]), h, yerr = np.sqrt(h * len(bins[b][i]))/len(bins[b][i]), fmt = '.', color = 'blue')
    plt.plot(bins[b][i].sort_values(), norm_d2gamma_p_d2q2_dcosthetak(fls[0], bins[b][i].sort_values()), 
             label = f'fl =  {np.round(fls[0], decimal_places)} pm {np.round(fl_errs[0], decimal_places)}')
    plt.title(f'{data_files[0]} bin {b}')
    plt.plot(bins[b][i].sort_values(), norm_d2gamma_p_d2q2_dcosthetak(sm_fl[b], bins[b][i].sort_values()))
    plt.legend(['Acceptance curve fit', 'SM prediction', 'Filtered data'])
    plt.xlabel(i)
    plt.ylabel('P')
    plt.ylim(0,1.5)
    plt.show()
    
    fl_array.append([fls[0], fl_errs[0]])

#%%
FL = [fl_array[i][0] for i in range(10)]
FL_err = [fl_array[i][1] for i in range(10)]

def fl_obs_plot(FL, FL_err):
    q2_bins = [[0.1, 0.98], [1.1, 2.5], [2.5, 4.0], [4.0, 6.0], [6.0, 8.0], [15.0, 17.0],
           [17.0, 19.0], [11.0, 12.5], [1.0, 6.0], [15.0, 17.9]]
    
    ind = [0, 1, 2, 3, 4, 5, 6, 7]
    over = [8, 9]
    
    q2_centre = []
    q2_width = []
    for i in range(len(q2_bins)):
        q2_centre += [(q2_bins[i][1]+q2_bins[i][0])/2]
        q2_width += [(q2_bins[i][1]-q2_bins[i][0])/2]
    
    FL_SM = np.array([0.296448, 0.760396, 0.796265, 0.711290, 0.606965, 
                  0.348441, 0.328081, 0.435190, 0.747644, 0.340156])
    FL_SM_err = np.array([0.050642, 0.043174, 0.033640, 0.049096, 0.050785, 
                          0.035540, 0.027990, 0.036637, 0.039890, 0.024826])
    
    plt.clf()
    # Different colours used to distinguish overlapping bins
    for k in range(len(ind)):
        i = ind[k]
        x, y, width, height = (q2_centre[i]-q2_width[i], FL_SM[i]-FL_SM_err[i], q2_width[i]*2, FL_SM_err[i]*2)
        rect = patches.Rectangle((x, y), width, height, alpha=0.4, color='m')
        plt.gca().add_patch(rect)
    for j in range(len(over)):
        i = over[j]
        x, y, width, height = (q2_centre[i]-q2_width[i], FL_SM[i]-FL_SM_err[i], q2_width[i]*2, FL_SM_err[i]*2)
        rect = patches.Rectangle((x, y), width, height, alpha=0.4, color='c')
        plt.gca().add_patch(rect)
    plt.errorbar(q2_centre, FL, yerr=FL_err, xerr=q2_width, fmt='ko', markersize=5, capsize=0, label='Filtered Data')
    plt.xlabel(r'$q^2$')
    plt.ylabel(r'$F_{L}$')
    plt.grid('both')
    plt.legend()
    plt.show()

fl_obs_plot(FL, FL_err)
#%%