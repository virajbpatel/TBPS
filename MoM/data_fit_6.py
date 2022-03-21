# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 20:08:03 2022

@author: kevin, the legend
READ ME
4th version of the data-fitting code
Fitting over costhetal
Lines of codes or functions which need special attention when adapting this for 
other variables will be indicated with #!! 
    example:
        def pdf(fl, afb, costhetal): #!!
            (...)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import MODULE_curve_fitting_functions as mcf
from iminuit import Minuit as minuit
#%%
params = {'axes.labelsize': 21,
          'legend.fontsize': 12,
          'xtick.labelsize': 18,
          'ytick.labelsize': 18,
          'figure.figsize': [8.8, 8.8/1.618]}
font = {'family' : 'Times New Roman',
        'size'   : 14}
plt.rc('font', **font)
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update(params)
#%%
"""
Input the file names here.
mcf.read_data_into_bins() returns the dataframe AND the binned dataframe in q2 bins for the data and acceptance files.
    argument: file_type = 0 or 1 for pkl or csv file 
"""
files = ['datasets/total_dataset_binary3', 'datasets/acceptance_mc_binary3', 'datasets/part_filt']
DF, BINS, DF_MC, BINS_MC, DF_UF, BINS_UF = mcf.read_data_into_bins2(files[0], files[1], files[2], 0)
variables = ['phi', 'costhetal', 'costhetak', 'q2']

#%%
# import pickle5 as p
# import pickle
# path_to_protocol5 = 'datasets/acceptance_mc_binary3.pkl'

# with open(path_to_protocol5, "rb") as fh:
#     data = p.load(fh)

# with open(path_to_protocol5, "wb") as f:
#     # Pickle the 'labeled-data' dictionary using the highest protocol available.
#     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

#%%

"""
choose v as as the variable we are fitting over.
0 = phi
1 = costhetal
2 = costhetak
3 = q2

also define various SM predictions for the free parameters. Will be good as initial guesses and for final checks
"""
v= variables[1] #!!
B = 10 # = number of q2 bins to iterate over
NO_OF_BINS = 20 # number of bins to 
SM_FL = [0.296, 0.760, 0.796, 0.711, 0.607, 0.348, 0.328, 0.435, 0.748, 0.340] 
SM_AFB = [-0.097, -0.138, -0.017, 0.122, 0.240, 0.402, 0.318, 0.391, 0.005, 0.368] 

#%%
"""
All the funtions are defined here:
    chebyshev() = un-normalised chebyshev polynomial. which is an approximation for the background 
    norm_chebyshev() = normalised chebyshev polynomial
    pdf() = the un-normalised d2gamma_p/d_(...)_ang_var (ang_var, free_params) WITH acceptance
    norm_pdf_without_bg() = normalised pdf() above. No background included
    pdf_with_bg() = norm_pdf_without_bg() + ratio * norm_chebyshev. This is NOT normalised
    norm_pdf() = normalised pdf_with_bg(). Includes acceptance and background, and is normalised
    log_likelihood_xxx() = negative log likelihood of norm_pdf
    norm_factor_signal_bg() = normalisation factor for plotting norm_pdf_without_bg() and norm_chebyshev()
    chi_sq() = reduced chi-squared test
    
THINGS TO CHANGE FOR DIFFERENT ANGULAR VARIABLES:
    norm_chebyshev() might need different limits in intergral based on what range the angular variable can take
    pdf() and the parameters
    norm_pdf_without_bg()
    pdf_with_bg()
    norm_pdf()
    log_likelihood_xxx(): paramters in function needs to be changed
    
"""
from MODULE_curve_fitting_functions import integrate2

from numpy.polynomial import chebyshev as chb
from numpy.polynomial import legendre as lg

def chebyshev(x, a0, a1, a2):
    return a0 + a1 * x + a2 * (2 * x ** 2 -1)

def norm_chebyshev(x, a0, a1, a2): #!!
    y = chb.chebval(x, [a0, a1, a2])
    # x = np.linspace(-1,1,10000) #!!
    norm_factor = chb.chebval(1, chb.chebint([a0, a1, a2], lbnd=-1, k=0))
    return y/norm_factor


def pdf(fl, afb, _bin, cos_theta_l): #!!
    ctl = cos_theta_l 
    c2tl = 2 * ctl ** 2 - 1 
    scalar_array = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) + 8/3 * afb * ctl)  
    return scalar_array * lg.legval(ctl, X_PARAMS[_bin]) # Here it is multiplied by acceptance

def norm_pdf_without_bg(fl, afb, _bin, cos_theta_l): #!!
    cos_theta_l_acc = np.linspace(-1,1,10000) #!!
    norm_factor = integrate2(cos_theta_l_acc, pdf(fl=fl, afb=afb, _bin = _bin, cos_theta_l=cos_theta_l_acc))
    return pdf(fl, afb, _bin, cos_theta_l)/norm_factor

def pdf_with_bg(fl, afb, a0, a1, a2, R, _bin, cos_theta_l): #!!
    # R = BG_RATIO[_bin]/(1-BG_RATIO[_bin]) # BG_RATIO is defined in the next cell
    norm_scalar_array = norm_pdf_without_bg(fl, afb, _bin, cos_theta_l) 
    return norm_scalar_array +  R*norm_chebyshev(cos_theta_l, a0, a1, a2)

def norm_pdf(fl, afb, a0, a1, a2, R, _bin, cos_theta_l): #!! 
    scalar_array = pdf_with_bg(fl, afb, a0, a1, a2, R, _bin, cos_theta_l)
    cos_theta_l_acc = np.linspace(-1,1,10000)
    norm_factor = integrate2(cos_theta_l_acc, pdf_with_bg(fl, afb, a0, a1, a2, R, _bin, cos_theta_l_acc))
    normalised_scalar_array = scalar_array/norm_factor  
    return normalised_scalar_array

def log_likelihood_ctl(fl, afb, a0, a1, a2, R, _bin): #!!
    # _BIN = BINS[int(_bin)] # 1st option: all data in q2 bins
    _BIN = BINS[int(_bin)][(BINS[int(_bin)].B0_MM >= 5240)  & (BINS[int(_bin)].B0_MM <=5320)] # 2nd option: B0_MM cuts in q2 bins
    ctl = _BIN['costhetal']
    normalised_scalar_array = norm_pdf(fl=fl, afb=afb, a0=a0, a1=a1, a2=a2, R=R, _bin = int(_bin), cos_theta_l= ctl)
    dummy_x = np.linspace(-1,1,1000)
    check = pdf(fl, afb, int(_bin), dummy_x)
    physicalness = 1
    if np.sum([check < 0]):
        physicalness = 1 + np.sum([check < 0])/10
    
    return - np.sum(np.log(normalised_scalar_array)) * physicalness

def norm_factor_signal_bg(fl, afb, a0, a1, a2, R, _bin): #!!
    cos_theta_l = np.linspace(-1,1,10000) #!!
    Y = pdf_with_bg(fl, afb, a0, a1, a2, R, _bin, cos_theta_l)
    norm_factor = integrate2(cos_theta_l, Y)
    return norm_factor

def chi_sq(x, data, fls, afbs, a0, a1, a2, b):
    X = 0
    N = sum(data)
    dx = x[1] - x[0]
    for i in range(len(data)):
        X += ((N*norm_pdf(fls, afbs, a0, a1, a2, b, cos_theta_l = x[i])*dx - data[i]) ** 2) / data[i]
    return X/(len(data) + 5)
#%%
"""
Obtaining the chebyshev polynomials for each q2 bin
Obtaining acceptance coefficients for each q2 bin
"""
X_PARAMS, X_COV = [], [] # acceptance
CHEBY_PARAMS, CHEBY_COV, BG_RATIO, R_VALS = [], [], [], [] #chebyshev 

### initial guesses for the exponent parameter B and lamda. The fitting is sensitive so must be fine-tuned
### below is tuned for binary3 dataset
exp_guess = [[100, 200, 100, 100, 200, 150, 100, 150, 300, 150], 
             [0.00001, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]]

import MoM_bin_bs as mom

c = np.load('accept_arrays/c_ijmn_NN_4or.npy')

for i in range(B):
    ### Extract acceptance params
    ### Use plt.hist() to show plots. Requires one more output argument: n, e, _= plt.hist()
    lg_acc = mom.acceptance_ctl_bin(DF_MC, c, i)
    
    X_PARAMS += [lg_acc]

#%%
for i in range(B):
    ### Extract chebyshev params
    n, e, _ = plt.hist(BINS[i][BINS[i].B0_MM.between(5170, 5700)].B0_MM, bins = 50, range=[5170, 5700], alpha=0.3)
    ec = mcf.center_bins(e)
    ge_params, ge_cov = curve_fit(mcf.gaussian_exponent_fit, ec, n, 
                                  p0 = [200, 5280, 180, exp_guess[0][i], exp_guess[1][i]], maxfev = 30000)
    
    n_p, b_p = np.histogram(BINS[i][BINS[i].B0_MM.between(5170, 5700)].B0_MM, bins = 50, range=[5170, 5700])
    b_c = (b_p[1:] + b_p[:-1]) / 2
    n_p_err = np.sqrt(n_p)
    plt.errorbar(b_c, n_p, yerr=n_p_err, xerr = (b_c[1]-b_c[0])/2, color='k', fmt='o',
                 markersize=3)
    
    ge_params, ge_cov = curve_fit(mcf.gaussian_exponent_fit, ec, n, 
                                  p0 = [200, 5280, 180, exp_guess[0][i], exp_guess[1][i]], maxfev = 30000)
    plt.plot(np.arange(5170, 5700), mcf.gaussian_exponent_fit(np.arange(5170, 5700), *ge_params))
    
    
    
    plt.xlim((5170, 5700))
    plt.xlabel(r'$m(K\pi\mu\mu)$')
    ylabel = 'Candidate Count / %.1f MeV' % (b_c[1]-b_c[0])
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()
    
    SIGNAL = 0
    BG = 0
    
    sig_int = mcf.Gauss_int(5240, 5320, *ge_params[:3])
    bg_int = mcf.exp_int(5240, 5320, *ge_params[3:])
    
    # print(mcf.Gauss_int(5240, 5320, *ge_params[:3])/mcf.Gauss_int(5200, 5355, *ge_params[:3]))
    
    SIGNAL += sig_int
    BG += bg_int
    RATIO = BG/(SIGNAL + BG)
    BG_RATIO.append(RATIO)
    R_VALS.append(RATIO/(1-RATIO))
    print(BG/(SIGNAL + BG))

    
    BINS_Q2_HM = BINS[i][(BINS[i]['B0_MM'] >= 5355) & (BINS[i]['B0_MM'] <= 5700)]
    n_hm, e_hm, _ = plt.hist(BINS_Q2_HM[v], bins = 10) 
    ec_hm = mcf.center_bins(e_hm)
    ec_width_hm = ec_hm[1] - ec_hm[0]
    cheby_params, cheby_cov = curve_fit(chebyshev, ec_hm, n_hm)
    plt.plot(np.linspace(-1,1,20), chebyshev(np.linspace(-1,1,20), *cheby_params))
    plt.show()
    
    CHEBY_PARAMS.append(cheby_params)
    CHEBY_COV.append(cheby_cov)

#%%
fl_array = [] #!!
afb_array = [] #!!

for b in range(10):            
    log_likelihood_ctl.errordef = minuit.LIKELIHOOD
    decimal_places = 3
    ST = [SM_FL[b],SM_AFB[b], CHEBY_PARAMS[b][0], CHEBY_PARAMS[b][1], CHEBY_PARAMS[b][2], R_VALS[b]] #!!
    SD = [0.0, 0.0, np.sqrt(CHEBY_COV[b][0][0]), np.sqrt(CHEBY_COV[b][1][1]), np.sqrt(CHEBY_COV[b][2][2]), 0.1*R_VALS[b]] 
    fls, fl_errs = [], [] #!!
    afbs, afb_errs = [], [] #!!
    Rs, Rs_errs = [], []
    a0s, a0_errs = [], []
    a1s, a1_errs = [], []
    a2s, a2_errs = [], []

    m = minuit(log_likelihood_ctl, fl=ST[0], afb=ST[1], a0=ST[2], a1=ST[3], a2=ST[4], R=ST[5], _bin = b)
    m.fixed['_bin'] = True  # fixing the bin number as we don't want to optimize it
    # m.fixed['a_0'] = True
    """
    Change limits here to allow a_0, a_1 and a_2 to freely vary. At the moment
    dont seems to have found good limits which allows the fit to converge. 
    """
    # Expect of abs(Afb)<=0.75*(1-FL), 0<FL<1 from 2012 paper
    # Can use this to check for outside SM
    m.limits=((-1.0, 1.0), (-1.0, 1.0),  #!!
              (ST[2], ST[2]), (ST[3], ST[3]), (ST[4], ST[4]), (ST[5]-SD[5], ST[5]+SD[5]), None)
    m.migrad()
    m.hesse()
    bin_results_to_check = m
    fls.append(m.values[0]) #!!
    afbs.append(m.values[1]) #!!
    a0s.append(m.values[2])
    a1s.append(m.values[3])
    a2s.append(m.values[4])
    Rs.append(m.values[5])
    fl_errs.append(m.errors[0]) #!!
    afb_errs.append(m.errors[1]) #!!
    a0_errs.append(m.errors[2])
    a1_errs.append(m.errors[3])
    a2_errs.append(m.errors[4])
    Rs_errs.append(m.values[5])
    print(f"Bin {b}: Fl = {np.round(fls, decimal_places)} pm {np.round(fl_errs, decimal_places)},", 
          f" Afb = {np.round(afbs, decimal_places)} pm {np.round(afb_errs, decimal_places)}. Function minimum considered valid: {m.fmin.is_valid}")
    
    plt.clf()
    
    X = BINS[b][v].sort_values()
    vals = np.linspace(-1, 1, 100)
    signal = norm_pdf_without_bg(fls[0], afbs[0], b, vals)                                                                        
    background = Rs[0] * norm_chebyshev(vals, a0s[0], a1s[0], a2s[0])
    norm_factor = norm_factor_signal_bg(fls[0], afbs[0], a0s[0], a1s[0], a2s[0], Rs[0], b)
    
    # h, e, _ = plt.hist(X, bins = NO_OF_BINS, range=[-1, 1], density = True, histtype = 'step', color = 'tab:blue')
    
    n, b_n = np.histogram(X, bins = NO_OF_BINS, range= [-1, 1])
    n_w, b_w = np.histogram(X, bins = NO_OF_BINS, range= [-1, 1], density=True)
    
    b_c = (b_w[1:] + b_w[:-1]) / 2
    n_w_err = np.sqrt(n)*(n_w[5]/n[5])
    n_err = np.sqrt(n)
    plt.errorbar(b_c, n, yerr = n_err, xerr = (b_w[1]-b_w[0])/2, fmt = 'o', color = 'k', markersize=4)
    
    
    # plt.errorbar(mcf.center_bins(e), h, yerr = np.sqrt(h * len(X))/len(X), fmt = '.', color = 'tab:blue')
    plt.plot(vals, (n[5]/n_w[5])*norm_pdf(fls[0], afbs[0], a0s[0], a1s[0], a2s[0], Rs[0], b, vals), 'r--',
             label = f"Total Fit: fl = {np.round(fls, decimal_places)} pm {np.round(fl_errs, decimal_places)}, afb = {np.round(afbs, decimal_places)} pm {np.round(afb_errs, decimal_places)}")
    # plt.plot(vals, signal/norm_factor, label = 'Signal Fit', color = 'black')
    
    plt.plot(vals, (n[5]/n_w[5])*background/norm_factor, '-', color = 'grey')
    plt.fill_between(vals, (n[5]/n_w[5])*background/norm_factor, 0, color='c', alpha=0.2, label='Background')
    plt.plot(0, '.', label = f'SM vals: fl = {SM_FL[b]}, afb = {SM_AFB[b]}', color = 'grey') 
    plt.title(f'{files[0]} bin {b}')
    plt.legend()
    plt.xlabel(r'$\cos\theta_l$')
    plt.grid()
    ylabel = 'Candidate Count / %.1f' % ((b_w[1]-b_w[0]))
    plt.ylabel(str(ylabel))
    plt.xlim(-1, 1)
    plt.ylim(bottom=0)
    # plt.ylim(0,1.5)
    plt.show()
    
    # print(chi_sq(mcf.center_bins(e), h, fls[0], afbs[0], a0s[0], a1s[0], a2s[0], b))

    fl_array.append([fls[0], fl_errs[0]])
    afb_array.append([afbs[0], afb_errs[0]])
#%%
from time import gmtime, strftime
actual_time = strftime("%Y-%m-%d %H-%M-%S", gmtime())
fl_save = 'values/fl_' + str(actual_time) + '.npy'
afb_save = 'values/afb_' + str(actual_time) + '.npy'
np.save(str(fl_save) , np.array(fl_array))
np.save(str(afb_save) , np.array(afb_array))
#%%
import matplotlib.patches as patches

def fl_obs_plot(FL, FL_err): #!!
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

def afb_obs_plot(AFB, AFB_err): #!!
    q2_bins = [[0.1, 0.98], [1.1, 2.5], [2.5, 4.0], [4.0, 6.0], [6.0, 8.0], [15.0, 17.0],
           [17.0, 19.0], [11.0, 12.5], [1.0, 6.0], [15.0, 17.9]]
    ind = [0, 1, 2, 3, 4, 5, 6, 7]
    over = [8, 9]
    q2_centre = []
    q2_width = []
    for i in range(len(q2_bins)):
        q2_centre += [(q2_bins[i][1]+q2_bins[i][0])/2]
        q2_width += [(q2_bins[i][1]-q2_bins[i][0])/2]
    
    AFB_SM = np.array([-0.097052, -0.137987, -0.017385, 0.122155, 0.239939, 
                       0.401914, 0.318391, 0.391390, 0.004929, 0.367672])
    AFB_SM_err = np.array([0.008421, 0.031958, 0.029395, 0.039619, 0.047281, 
                           0.030141, 0.033931, 0.023627, 0.028058, 0.032230])
    plt.clf()
    for k in range(len(ind)):
        i = ind[k]
        x, y, width, height = (q2_centre[i]-q2_width[i], AFB_SM[i]-AFB_SM_err[i], q2_width[i]*2, AFB_SM_err[i]*2)
        rect = patches.Rectangle((x, y), width, height, alpha=0.4, color='m')
        plt.gca().add_patch(rect)
    for j in range(len(over)):
        i = over[j]
        x, y, width, height = (q2_centre[i]-q2_width[i], AFB_SM[i]-AFB_SM_err[i], q2_width[i]*2, AFB_SM_err[i]*2)
        rect = patches.Rectangle((x, y), width, height, alpha=0.4, color='c')
        plt.gca().add_patch(rect)
    
    plt.errorbar(q2_centre, AFB, yerr=AFB_err, xerr=q2_width, fmt='ko', markersize=5, capsize=0, label='Filtered Data')
    plt.xlabel(r'$q^2$')
    plt.ylabel(r'$A_{FB}$')
    plt.grid('both')
    plt.legend()
    plt.show()



AFB = np.array(afb_array)[:,0] #!!
AFB_err = np.array(afb_array)[:,1] #!!
FL = np.array(fl_array)[:,0] #!!
FL_err = np.array(fl_array)[:,1] #!!


# FL = np.load('values/fl_2022-03-16 19-54-52.npy')[:,0]
# FL_err = np.load('values/fl_2022-03-16 19-54-52.npy')[:,1]
# AFB = np.load('values/afb_2022-03-16 19-54-52.npy')[:,0]
# AFB_err = np.load('values/afb_2022-03-16 19-54-52.npy')[:,1]

fl_obs_plot(FL, FL_err)
afb_obs_plot(AFB, AFB_err)

#%%
def fl_obs_plot_loop(files): #!!
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
        
    color_cycle = ['k', 'b', 'r', 'g', 'c', 'm']
    for l in range(len(files)):
        FL = np.load(files[l])[:,0]
        FL_err = np.load(files[l])[:,1]
        plt.errorbar(q2_centre, FL, yerr=FL_err, xerr=q2_width, fmt='o', color=color_cycle[l],
                     markersize=5, capsize=0, label='Filtered Data')
    
    plt.xlabel(r'$q^2$')
    plt.ylabel(r'$F_{L}$')
    plt.grid('both')
    plt.legend()
    plt.show()

def afb_obs_plot_loop(files): #!!
    q2_bins = [[0.1, 0.98], [1.1, 2.5], [2.5, 4.0], [4.0, 6.0], [6.0, 8.0], [15.0, 17.0],
           [17.0, 19.0], [11.0, 12.5], [1.0, 6.0], [15.0, 17.9]]
    ind = [0, 1, 2, 3, 4, 5, 6, 7]
    over = [8, 9]
    q2_centre = []
    q2_width = []
    for i in range(len(q2_bins)):
        q2_centre += [(q2_bins[i][1]+q2_bins[i][0])/2]
        q2_width += [(q2_bins[i][1]-q2_bins[i][0])/2]
    
    AFB_SM = np.array([-0.097052, -0.137987, -0.017385, 0.122155, 0.239939, 
                       0.401914, 0.318391, 0.391390, 0.004929, 0.367672])
    AFB_SM_err = np.array([0.008421, 0.031958, 0.029395, 0.039619, 0.047281, 
                           0.030141, 0.033931, 0.023627, 0.028058, 0.032230])
    plt.clf()
    for k in range(len(ind)):
        i = ind[k]
        x, y, width, height = (q2_centre[i]-q2_width[i], AFB_SM[i]-AFB_SM_err[i], q2_width[i]*2, AFB_SM_err[i]*2)
        rect = patches.Rectangle((x, y), width, height, alpha=0.4, color='m')
        plt.gca().add_patch(rect)
    for j in range(len(over)):
        i = over[j]
        x, y, width, height = (q2_centre[i]-q2_width[i], AFB_SM[i]-AFB_SM_err[i], q2_width[i]*2, AFB_SM_err[i]*2)
        rect = patches.Rectangle((x, y), width, height, alpha=0.4, color='c')
        plt.gca().add_patch(rect)
    
    color_cycle = ['k', 'b', 'r', 'g', 'c', 'm']
    for l in range(len(files)):
        AFB = np.load(files[l])[:,0]
        AFB_err = np.load(files[l])[:,1]
        plt.errorbar(q2_centre, AFB, yerr=AFB_err, xerr=q2_width, fmt='o', color=color_cycle[l],
                     markersize=5, capsize=0, label='Filtered Data')
    
    plt.xlabel(r'$q^2$')
    plt.ylabel(r'$A_{FB}$')
    plt.grid('both')
    plt.legend()
    plt.show()


fl_obs_plot_loop(['values/fl_2022-03-16 19-54-52.npy', 'values/fl_2022-03-16 20-11-48.npy'])

afb_obs_plot_loop(['values/afb_2022-03-16 19-54-52.npy', 'values/afb_2022-03-16 20-11-48.npy'])
































