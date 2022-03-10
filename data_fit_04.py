# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 20:08:03 2022

@author: kevin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import MODULE_curve_fitting_functions as mcf
from iminuit import Minuit as minuit

files = ['total_dataset_binary3', 'acceptance_mc_binary3']
DF, BINS, DF_MC, BINS_MC = mcf.read_data_into_bins(files[0], files[1], 0)
variables = ['phi', 'costhetal', 'costhetak', 'q2']


v= variables[1] # variable we are inspecting
B = 10 # number of bins
NO_OF_BINS = 25
SM_FL = [0.296, 0.760, 0.796, 0.711, 0.607, 0.348, 0.328, 0.435, 0.748, 0.340]
SM_AFB = [-0.097, -0.138, -0.017, 0.122, 0.240, 0.402, 0.318, 0.391, 0.005, 0.368]

#%% FUNCTIONS HERE
from MODULE_curve_fitting_functions import integrate 
from MODULE_curve_fitting_functions import integrate2

def chebyshev(x, a0, a1, a2):
    return a0 + a1 * x + a2 * (2 * x ** 2 -1)

def norm_chebyshev(x, a0, a1, a2):
    y = chebyshev(x, a0, a1, a2)
    x = np.linspace(-1,1,10000)
    norm_factor = integrate2(x, chebyshev(x, a0, a1, a2))
    return y/norm_factor

def pdf(fl, afb, _bin, cos_theta_l):
    ctl = cos_theta_l
    c2tl = 2 * ctl ** 2 - 1
    scalar_array = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) + 8/3 * afb * ctl) 
    return scalar_array * mcf.polynomial(ctl, *X_PARAMS[_bin])

def norm_pdf_without_bg(fl, afb, _bin, cos_theta_l):
    cos_theta_l_acc = np.linspace(-1,1,10000)
    norm_factor = integrate2(cos_theta_l_acc, pdf(fl=fl, afb=afb, _bin = _bin, cos_theta_l=cos_theta_l_acc))
    return pdf(fl, afb, _bin, cos_theta_l)/norm_factor

def pdf_with_bg(fl, afb, a0, a1, a2, _bin, cos_theta_l):
    R = BG_RATIO[_bin]/(1-BG_RATIO[_bin])
    scalar_array = pdf(fl, afb, _bin, cos_theta_l)
    cos_theta_l_acc = np.linspace(-1,1,10000)
    norm_factor = integrate2(cos_theta_l_acc, pdf(fl=fl, afb=afb, _bin = _bin, cos_theta_l=cos_theta_l_acc))
    return scalar_array/norm_factor +  R*norm_chebyshev(cos_theta_l, a0, a1, a2)

def norm_pdf(fl, afb, a0, a1, a2, _bin, cos_theta_l):
    scalar_array = pdf_with_bg(fl, afb, a0, a1, a2, _bin, cos_theta_l)
    cos_theta_l_acc = np.linspace(-1,1,10000)
    norm_factor = integrate2(cos_theta_l_acc, pdf_with_bg(fl, afb, a0, a1, a2, _bin, cos_theta_l_acc))
    normalised_scalar_array = scalar_array /norm_factor  
    return normalised_scalar_array


def log_likelihood_ctl(fl, afb, a0, a1, a2, _bin):
    _BIN = BINS[int(_bin)]
    ctl = _BIN['costhetal']
    normalised_scalar_array = norm_pdf(fl=fl, afb=afb, a0=a0, a1=a1, a2=a2, _bin = int(_bin), cos_theta_l= ctl)
    return - np.sum(np.log(normalised_scalar_array))

def norm_factor_signal_bg(fl, afb, a0, a1, a2, _bin, cos_theta_l):
    cos_theta_l_acc = np.linspace(-1,1,10000)
    Y = pdf_with_bg(fl, afb, a0, a1, a2, _bin, cos_theta_l_acc)
    norm_factor = integrate2(cos_theta_l_acc, Y)
    return norm_factor

def chi_sq(x, data, fls, afbs, a0, a1, a2, b):
    X = 0
    N = sum(data)
    dx = x[1] - x[0]
    for i in range(len(data)):
        X += ((N*norm_pdf(fls, afbs, a0, a1, a2, b, cos_theta_l = x[i])*dx - data[i]) ** 2) / data[i]
    return X/(len(data) + 5)
#%%
X_PARAMS, X_COV = [], []
CHEBY_PARAMS, CHEBY_COV, BG_RATIO = [], [], []

exp_guess = [[100, 200, 100, 100, 200, 150, 100, 150, 300, 150], ### for binary3 dataset
             [0.00001, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]]

for i in range(B):
    ### Extract acceptance params
    n, e, _ = plt.hist(BINS_MC[i][v], bins = NO_OF_BINS)
    _x, _y, x_params, x_cov = mcf.accept_fit(n, e)
    X_PARAMS.append(x_params)
    X_COV.append(x_cov)
    plt.plot(_x, _y)
    plt.show()
    
for i in range(B):
    ### Extract chebyshev params
    n, e, _ = plt.hist(BINS[i].B0_MM, bins = NO_OF_BINS)
    ec = mcf.center_bins(e)
    ge_params, ge_cov = curve_fit(mcf.gaussian_exponent_fit, ec[3:], n[3:], 
                                  p0 = [200, 5280, 180, exp_guess[0][i], exp_guess[1][i]])
    plt.plot(np.arange(5100, 5700), mcf.gaussian_exponent_fit(np.arange(5100, 5700), *ge_params))
    plt.show()
    # print(ge_params)
    SIGNAL = 0
    BG = 0
    for j in ec[3:]:
        SIGNAL += mcf.Gauss(j, *ge_params[:3])
        BG += mcf.exp_decay(j, *ge_params[3:])
    BG_RATIO.append(BG/(SIGNAL + BG))
    print(BG/(SIGNAL + BG))    
    
    BINS_Q2_HM = BINS[i][(BINS[i]['B0_MM'] >= 5355) & (BINS[i]['B0_MM'] <= 5700)]
    n_hm, e_hm, _ = plt.hist(BINS_Q2_HM[v], bins = NO_OF_BINS)
    ec_hm = mcf.center_bins(e_hm)
    ec_width_hm = ec_hm[1] - ec_hm[0]
    cheby_params, cheby_cov = curve_fit(chebyshev, ec_hm, n_hm)
    plt.plot(np.linspace(-1,1,20), chebyshev(np.linspace(-1,1,20), *cheby_params))
    plt.show()
    
    CHEBY_PARAMS.append(cheby_params)
    CHEBY_COV.append(cheby_cov)

#%%
fl_array = []
afb_array = []

for b in range(10):    
    # n, edge, patch = plt.hist(BINS[b][v], bins = NO_OF_BINS, density = True, histtype = 'step', label = 'total data')
    # plt.title(f'{data_files[0]} bin {b}')
    # plt.plot(x_acc, n_acc, label = 'fitted acceptance function')
    # plt.xlabel(i)
    # plt.ylabel('Prob density')
    # plt.legend()
    # plt.clf()
    # plt.show()
        
    log_likelihood_ctl.errordef = minuit.LIKELIHOOD
    decimal_places = 3
    ST = [SM_FL[b],SM_AFB[b], CHEBY_PARAMS[b][0], CHEBY_PARAMS[b][1], CHEBY_PARAMS[b][2]]
    SD = [0,0, np.sqrt(CHEBY_COV[b][0][0]), np.sqrt(CHEBY_COV[b][1][1]), np.sqrt(CHEBY_COV[b][2][2])]
    fls, fl_errs = [], []
    afbs, afb_errs = [], []
    a0s, a0_errs = [], []
    a1s, a1_errs = [], []
    a2s, a2_errs = [], []

    m = minuit(log_likelihood_ctl, fl=ST[0], afb=ST[1], a0=ST[2], a1=ST[3], a2=ST[4], _bin = b)
    m.fixed['_bin'] = True  # fixing the bin number as we don't want to optimize it
    # m.fixed['a_0'] = True
    """
    Change limits here to allow a_0, a_1 and a_2 to freely vary. At the moment
    dont seems to have found good limits which allows the fit to converge. 
    """
    m.limits=((-1.0, 1.0), (-1.0, 1.0), 
              (ST[2]-SD[2], ST[2]), (ST[3], ST[3]), (ST[4], ST[4]), None) 
    m.migrad()
    m.hesse()
    bin_results_to_check = m
    fls.append(m.values[0])
    afbs.append(m.values[1])
    a0s.append(m.values[2])
    a1s.append(m.values[3])
    a2s.append(m.values[4])
    fl_errs.append(m.errors[0])
    afb_errs.append(m.errors[1])
    a0_errs.append(m.errors[2])
    a1_errs.append(m.errors[3])
    a2_errs.append(m.errors[4])
    print(f"Bin {b}: Fl = {np.round(fls, decimal_places)} pm {np.round(fl_errs, decimal_places)},", 
          f" Afb = {np.round(afbs, decimal_places)} pm {np.round(afb_errs, decimal_places)}. Function minimum considered valid: {m.fmin.is_valid}")
    
    X = BINS[b][v].sort_values()
    signal = norm_pdf_without_bg(fls[0], afbs[0], b, X)                                                                        
    background = BG_RATIO[b]/(1-BG_RATIO[b]) * norm_chebyshev(X, a0s[0], a1s[0], a2s[0])
    norm_factor = norm_factor_signal_bg(fls[0], afbs[0], a0s[0], a1s[0], a2s[0], b, X)
    h, e, _ = plt.hist(X, bins = NO_OF_BINS, density = True, histtype = 'step', color = 'tab:blue')
    plt.errorbar(mcf.center_bins(e), h, yerr = np.sqrt(h * len(X))/len(X), fmt = '.', color = 'tab:blue')
    plt.plot(X, norm_pdf(fls[0], afbs[0], a0s[0], a1s[0], a2s[0], b, BINS[b][v].sort_values()),
             label = f"Total Fit: fl = {np.round(fls, decimal_places)} pm {np.round(fl_errs, decimal_places)}, afb = {np.round(afbs, decimal_places)} pm {np.round(afb_errs, decimal_places)}")
    plt.plot(X, signal/norm_factor, label = 'Signal Fit', color = 'black')
    plt.plot(X, background/norm_factor, ':', label = "Background fit", color = 'red')
    plt.plot(0, '.', label = f'SM vals: fl = {SM_FL[b]}, afb = {SM_AFB[b]}', color = 'grey')
    plt.title(f'{files[0]} bin {b}')
    plt.legend()
    plt.xlabel(v)
    plt.ylabel('PDF')
    plt.ylim(0,1.5)
    plt.show()
    
    red_chi = chi_sq(mcf.center_bins(e), h, fls[0], afbs[0], a0s[0], a1s[0], a2s[0], b)

    fl_array.append([fls[0], fl_errs[0]])
    afb_array.append([afbs[0], afb_errs[0]])











































