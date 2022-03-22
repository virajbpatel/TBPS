# -*- coding: utf-8 -*-
"""
Created on Mon  21st 20:08:03 2022
@author: kevin
READ ME
4th version of the data-fitting code
Fitting over costhetal
Lines of codes or functions which need special attention when adapting this for 
other variables will be indicated with #!! 
    example:
        def pdf(fl, afb, costhetal): #!!
            (...)
Modules needed:
    MODULE_curve_fitting_functions.py
    MOM_bins.py
    
Also, create a folder named 'accept_arrays', which will store the data for the acceptance function (see line 39 & 40)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import MODULE_curve_fitting_functions as mcf
from iminuit import Minuit as minuit
import MoM_bins as momb


"""
Input the file names here.
mcf.read_data_into_bins() returns the dataframe AND the binned dataframe in q2 bins for the data and acceptance files.
    argument: file_type = 0 or 1 for pkl or csv file 
"""
files = ['0321_forest_total80', '0321_forest_acc80', '0321_forest_accnoq280']
DF, BINS, DF_MC, BINS_MC, acc_filt_noq2, _ = mcf.read_data_into_bins2(files[0], files[1], files[2], 0)
variables = ['phi', 'costhetal', 'costhetak', 'q2']

# acc_filt_noq2 = pd.read_pickle('acceptance_mc_binary3_no_cuts.pkl')
path = 'accept_arrays/c_ijmn_CF_6or.npy'
c = np.load(path)


"""
choose v as as the variable we are fitting over.
0 = phi
1 = costhetal
2 = costhetak
3 = q2
also define various SM predictions for the free parameters. Will be good as initial guesses and for final checks
"""
v= variables[3] #!!
B = 10 # = number of q2 bins to iterate over
NO_OF_BINS = 20 # number of bins to 
SM_FL = [0.296, 0.760, 0.796, 0.711, 0.607, 0.348, 0.328, 0.435, 0.748, 0.340] 
SM_AFB = [-0.097, -0.138, -0.017, 0.122, 0.240, 0.402, 0.318, 0.391, 0.005, 0.368] 
SM_S3 = [0.010876,0.002373,-0.010864,-0.024751,-0.039754,-0.173464,-0.251488,-0.085975,-0.012641,-0.204963]
SM_S9 = [-0.000701,-0.000786,-0.000735,-0.000706,-0.000715,0.000292, 0.000169, 0.000449	,-0.000739,0.000242]

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
    y = a0 + a1 * x + a2 * (2 * x ** 2 - 1)
    # test_x = np.linspace(-1, 1, 20)
    # test_y = a0 + a1 * test_x + a2 * (2 * test_x ** 2 - 1)
    # print(test_y)
    # if any(test_y<0) == True:
    #     y = y*10
    # if a0 - a1 + a2 * (2 -1) < 0:
    #     y = y*0
    return y

def norm_chebyshev(x, a0, a1, a2): #!!
    y = chebyshev(x, a0, a1, a2)
    # x = np.linspace(-np.pi,np.pi,5000) #!!
    # norm_factor = integrate2(x, chebyshev(x, a0, a1, a2))
    norm_factor = chb.chebval(1, chb.chebint([a0, a1, a2], lbnd = -np.pi, k=0))
    return y/norm_factor

def pdf(s3, s9, _bin, phi): #!!
    scalar_array = 1/(2*np.pi) * (s3*np.cos(2*phi)+s9*np.sin(2*phi)+1)
    scalar_array[scalar_array<0] = 0
    return scalar_array * lg.legval(phi, X_PARAMS[_bin])

def norm_pdf_without_bg(s3, s9, _bin, phi): #!!
    phi_acc = np.linspace(-np.pi,np.pi,5000) #!!
    norm_factor = integrate2(phi_acc, pdf(s3=s3, s9=s9, _bin = _bin, phi=phi_acc))
    return pdf(s3, s9, _bin, phi)/norm_factor

def pdf_with_bg(s3, s9, a0, a1, a2, _bin, phi): #!!
    R = BG_RATIO[_bin]/(1-BG_RATIO[_bin]) # BG_RATIO is defined in the next cell
    norm_scalar_array = norm_pdf_without_bg(s3, s9, _bin, phi) 
    return norm_scalar_array +  R*norm_chebyshev(phi, a0, a1, a2)

def norm_pdf(s3, s9, a0, a1, a2, _bin, phi): #!! 
    scalar_array = pdf_with_bg(s3, s9, a0, a1, a2, _bin, phi)
    phi_acc = np.linspace(-np.pi,np.pi,5000)
    norm_factor = integrate2(phi_acc, pdf_with_bg(s3, s9, a0, a1, a2, _bin, phi_acc))
    normalised_scalar_array = scalar_array /norm_factor  
    return normalised_scalar_array


def log_likelihood_ctl(s3, s9, a0, a1, a2, _bin): #!!
    # _BIN = BINS[int(_bin)] # 1st option: all data in q2 bins
    _BIN = BINS[int(_bin)][(BINS[int(_bin)].B0_MM >= 5200)  & (BINS[int(_bin)].B0_MM <=5355)] # 2nd option: B0_MM cuts in q2 bins
    phi = _BIN['phi']
    normalised_scalar_array = norm_pdf(s3=s3, s9=s9, a0=a0, a1=a1, a2=a2, _bin = int(_bin), phi= phi)
    dummy_x = np.linspace(-np.pi,np.pi,5000)
    
    physicalness = 1 # physicalness raises the -log_likelihood if the signal or background has negative probability
    # check_signal = pdf(fl, afb, int(_bin), dummy_x)
    # if np.sum([check_signal < 0]):
    #     physicalness += (np.sum([check_signal < 0])/500)
    
    check_background = chebyshev(dummy_x, a0, a1, a2)
    if np.sum([check_background < 0]):
        physicalness += (np.sum([check_background])/500)**2

    return - np.sum(np.log(normalised_scalar_array)) * physicalness

def norm_factor_signal_bg(s3, s9, a0, a1, a2, _bin): #!!
    phi = np.linspace(-np.pi,np.pi,5000) #!!
    Y = pdf_with_bg(s3, s9, a0, a1, a2, _bin, phi)
    norm_factor = integrate2(phi, Y)
    return norm_factor

def chi_sq(x, data, s3s, s9s, a0, a1, a2, b):
    X = 0
    N = sum(data)
    dx = x[1] - x[0]
    for i in range(len(data)):
        X += ((N*norm_pdf(s3s, s9s, a0, a1, a2, b, phi = x[i])*dx - data[i]) ** 2) / data[i]
    return X/(len(data) + 5)

#%%
"""
Obtaining the chebyshev polynomials for each q2 bin
Obtaining acceptance coefficients for each q2 bin
"""
X_PARAMS, X_COV = [], [] # acceptance
CHEBY_PARAMS, CHEBY_COV, BG_RATIO = [], [], [] #chebyshev 

### initial guesses for the exponent parameter B and lamda. The fitting is sensitive so must be fine-tuned
### below is tuned for binary3 dataset
exp_guess = [[100, 200, 100, 100, 200, 150, 100, 150, 300, 150], 
             [0.00001, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]]

path = 'accept_arrays/c_ijmn_CF_6or.npy'
# momb.acceptance_c(acc_filt_noq2, [0.1, 19.0], path)
c = np.load(path)

for i in range(B):
    X_PARAMS.append(momb.acceptance_phi_bin(acc_filt_noq2, c, i))

#%%
for i in range(B):
    ## Extract chebyshev params
    # n, e, _ = plt.hist(BINS[i][BINS[i].B0_MM.between(5170, 5700)].B0_MM, bins = 50, range=[5170, 5700], alpha=0.3)
    # ec = mcf.center_bins(e)
    # ge_params, ge_cov = curve_fit(mcf.gaussian_exponent_fit, ec, n, 
    #                               p0 = [200, 5280, 180, exp_guess[0][i], exp_guess[1][i]], maxfev = 30000)
    
    n_p, b_p, _ = plt.hist(BINS[i][BINS[i].B0_MM.between(5170, 5700)].B0_MM, bins = 50, range=[5170, 5700], alpha = 0.3)
    b_c = (b_p[1:] + b_p[:-1]) / 2
    n_p_err = np.sqrt(n_p)
    plt.errorbar(b_c, n_p, yerr=n_p_err, xerr = (b_c[1]-b_c[0])/2, color='k', fmt='o',
                 markersize=3)
    
    ge_params, ge_cov = curve_fit(mcf.gaussian_exponent_fit, b_c, n_p, 
                                  p0 = [200, 5280, 180, exp_guess[0][i], exp_guess[1][i]], maxfev = 30000)
    plt.plot(np.arange(5170, 5700), mcf.gaussian_exponent_fit(np.arange(5170, 5700), *ge_params))
    
    
    
    plt.xlim((5170, 5700))
    plt.xlabel(r'$m(K\pi\mu\mu)$')
    ylabel = 'Candidate Count / %.1f MeV' % (b_c[1]-b_c[0])
    plt.ylabel(ylabel)
    plt.grid()
    plt.clf()
    plt.show()
   
    SIGNAL = 0
    BG = 0
    
    sig_int = mcf.Gauss_int(5200, 5360, *ge_params[:3])
    # print(mcf.Gauss_int(5240, 5320, *ge_params[:3])/mcf.Gauss_int(5200, 5355, *ge_params[:3]))
    bg_int = mcf.exp_int(5200, 5360, *ge_params[3:])
    
    SIGNAL += sig_int
    BG += bg_int
    BG_RATIO.append(BG/(SIGNAL + BG))
    print(BG/(SIGNAL + BG))
    
    BINS_Q2_HM = BINS[i][(BINS[i]['B0_MM'] >= 5355) & (BINS[i]['B0_MM'] <= 5700)]
    n_hm, e_hm, _ = plt.hist(BINS_Q2_HM[v], bins = 6, range = [-np.pi,np.pi])
    ec_hm = mcf.center_bins(e_hm)
    
    zeros_list = []
    for j in range(len(n_hm)):
        if n_hm[j] == 0:
            zeros_list.append(j)
    if len(zeros_list) > 0:
        n_hm = np.delete(n_hm, zeros_list)
        ec_hm = np.delete(ec_hm, zeros_list)
        
    ec_width_hm = ec_hm[1] - ec_hm[0]
    cheby_params, cheby_cov = curve_fit(chebyshev, ec_hm, n_hm)
    plt.plot(np.linspace(-np.pi,np.pi,20), chebyshev(np.linspace(-np.pi,np.pi,20), *cheby_params))
    plt.title(f'Bin {i}')
    plt.show()
    
    CHEBY_PARAMS.append(cheby_params)
    CHEBY_COV.append(cheby_cov)

#%%
s3_array = [] #!!
s9_array = [] #!!

for b in range(10):            
    log_likelihood_ctl.errordef = minuit.LIKELIHOOD
    decimal_places = 3
    ST = [SM_S3[b],SM_S9[b], CHEBY_PARAMS[b][0], CHEBY_PARAMS[b][1], CHEBY_PARAMS[b][2]] #!!
    SD = [0,0, np.sqrt(CHEBY_COV[b][0][0]), np.sqrt(CHEBY_COV[b][1][1]), np.sqrt(CHEBY_COV[b][2][2])] 
    s3s, s3_errs = [], [] #!!
    s9s, s9_errs = [], [] #!!
    a0s, a0_errs = [], []
    a1s, a1_errs = [], []
    a2s, a2_errs = [], []

    m = minuit(log_likelihood_ctl, s3=ST[0], s9=ST[1], a0=ST[2], a1=ST[3], a2=ST[4], _bin = b)
    m.fixed['_bin'] = True  # fixing the bin number as we don't want to optimize it
    # m.fixed['a_0'] = True
    """
    Change limits here to allow a_0, a_1 and a_2 to freely vary. At the moment
    dont seems to have found good limits which allows the fit to converge. 
    """
    m.limits=((-np.pi,np.pi), (-np.pi,np.pi), #!!
              (ST[2]-0*SD[2], ST[2]+SD[2]), (ST[3]-SD[3], ST[3]+SD[3]), (ST[4]-SD[3], ST[4]+SD[3]), None) 
    m.migrad()
    m.hesse()
    bin_results_to_check = m
    s3s.append(m.values[0]) #!!
    s9s.append(m.values[1]) #!!
    a0s.append(m.values[2])
    a1s.append(m.values[3])
    a2s.append(m.values[4])
    s3_errs.append(m.errors[0]) #!!
    s9_errs.append(m.errors[1]) #!!
    a0_errs.append(m.errors[2])
    a1_errs.append(m.errors[3])
    a2_errs.append(m.errors[4])
    print(f"Bin {b}: S3 = {np.round(s3s, decimal_places)} pm {np.round(s3_errs, decimal_places)},", 
          f" S9 = {np.round(s9s, decimal_places)} pm {np.round(s9_errs, decimal_places)}. Function minimum considered valid: {m.fmin.is_valid}")
    
    
    x = np.linspace(-np.pi,np.pi, 20)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.plot(x, [log_likelihood_ctl(s3=i, s9=s9s[0], a0 = a0s[0], a1 = a1s[0], a2 = a2s[0], _bin = b) for i in x])
    ax1.set_title(r'$S9$ = ' + str(s9s[0]))
    ax1.set_xlabel(r'$S3$')
    ax1.set_ylabel(r'$-\mathcal{L}$')
    ax1.grid()
    ax2.plot(x, [log_likelihood_ctl(s3=s3s[0], s9=i, a0 = a0s[0], a1 = a1s[0], a2 = a2s[0], _bin = b) for i in x])
    ax2.set_title(r'$S3$ = ' + str(s3s[0]))
    ax2.set_xlabel(r'$S9$')
    ax2.set_ylabel(r'$-\mathcal{L}$')
    ax2.grid()
    plt.tight_layout()
    plt.show()
    
    X = BINS[b][v].sort_values()
    vals = np.linspace(-np.pi,np.pi, 100)
    signal = norm_pdf_without_bg(s3s[0], s9s[0], b, vals)                                                                        
    background = BG_RATIO[b]/(1-BG_RATIO[b]) * norm_chebyshev(vals, a0s[0], a1s[0], a2s[0])
    norm_factor = norm_factor_signal_bg(s3s[0], s9s[0], a0s[0], a1s[0], a2s[0], b)
    
    n, b_n = np.histogram(X, bins = NO_OF_BINS, range= [-np.pi,np.pi])
    n_w, b_w = np.histogram(X, bins = NO_OF_BINS, range= [-np.pi,np.pi], density=True)
    
    b_c = (b_w[1:] + b_w[:-1]) / 2
    n_w_err = np.sqrt(n)*(n_w[5]/n[5])
    n_err = np.sqrt(n)
    plt.errorbar(b_c, n, yerr = n_err, xerr = (b_w[1]-b_w[0])/2, fmt = 'o', color = 'k', markersize=4)
    
    
    plt.plot(vals, (n[5]/n_w[5])*norm_pdf(s3s[0], s9s[0], a0s[0], a1s[0], a2s[0], b, vals), 'r--',
             label = f"Total Fit: s3 = {np.round(s3s, decimal_places)} pm {np.round(s3_errs, decimal_places)}, s9 = {np.round(s9s, decimal_places)} pm {np.round(s9_errs, decimal_places)}")
    
    plt.plot(vals, (n[5]/n_w[5])*background/norm_factor, '-', color = 'grey')
    plt.fill_between(vals, (n[5]/n_w[5])*background/norm_factor, 0, color='c', alpha=0.2, label='Background')
    plt.plot(0, '.', label = f'SM vals: s3 = {SM_S3[b]}, s9 = {SM_S9[b]}', color = 'grey') 
    plt.title(f'{files[0]} bin {b}')
    plt.legend()
    plt.xlabel(r'$phi$')
    plt.grid()
    ylabel = 'Candidate Count / %.1f' % ((b_w[1]-b_w[0]))
    plt.ylabel(str(ylabel))
    plt.xlim(-np.pi,np.pi)
    plt.ylim(bottom=0)
    # plt.ylim(0,1.5)
    plt.show()
    
    s3_array.append([s3s[0], s3_errs[0]])
    s9_array.append([s9s[0], s9_errs[0]])

#%%
import matplotlib.patches as patches
"""
From David's code, see that for reference
"""
def s3_obs_plot(S3, S3_err): #!!
    q2_bins = [[0.1, 0.98], [1.1, 2.5], [2.5, 4.0], [4.0, 6.0], [6.0, 8.0], [15.0, 17.0],
           [17.0, 19.0], [11.0, 12.5], [1.0, 6.0], [15.0, 17.9]]
    
    ind = [0, 1, 2, 3, 4, 5, 6, 7]
    over = [8, 9]
    
    q2_centre = []
    q2_width = []
    for i in range(len(q2_bins)):
        q2_centre += [(q2_bins[i][1]+q2_bins[i][0])/2]
        q2_width += [(q2_bins[i][1]-q2_bins[i][0])/2]
    
    S3_SM = np.array([0.010876,0.002373,-0.010864,-0.024751,-0.039754,-0.173464,-0.251488,-0.085975,-0.012641,-0.204963])
    S3_SM_err = np.array([0.006028,0.005363,0.004088,0.008459,0.013499,0.019065,0.015277,0.014357,0.004989,0.018582])
    
    plt.clf()
    # Different colours used to distinguish overlapping bins
    for k in range(len(ind)):
        i = ind[k]
        x, y, width, height = (q2_centre[i]-q2_width[i], S3_SM[i]-S3_SM_err[i], q2_width[i]*2, S3_SM_err[i]*2)
        rect = patches.Rectangle((x, y), width, height, alpha=0.4, color='m')
        plt.gca().add_patch(rect)
    for j in range(len(over)):
        i = over[j]
        x, y, width, height = (q2_centre[i]-q2_width[i], S3_SM[i]-S3_SM_err[i], q2_width[i]*2, S3_SM_err[i]*2)
        rect = patches.Rectangle((x, y), width, height, alpha=0.4, color='c')
        plt.gca().add_patch(rect)
    plt.errorbar(q2_centre, S3, yerr=S3_err, xerr=q2_width, fmt='ko', markersize=5, capsize=0, label='Filtered Data')
    plt.xlabel(r'$q^2$')
    plt.ylabel(r'$S3$')
    plt.grid('both')
    plt.legend()
    plt.show()

S3 = np.array(s3_array)[:,0] #!!
S3_err = np.array(s3_array)[:,1] #!!
s3_obs_plot(S3, S3_err)

def s9_obs_plot(S9, S9_err): #!!
    q2_bins = [[0.1, 0.98], [1.1, 2.5], [2.5, 4.0], [4.0, 6.0], [6.0, 8.0], [15.0, 17.0],
           [17.0, 19.0], [11.0, 12.5], [1.0, 6.0], [15.0, 17.9]]
    ind = [0, 1, 2, 3, 4, 5, 6, 7]
    over = [8, 9]
    q2_centre = []
    q2_width = []
    for i in range(len(q2_bins)):
        q2_centre += [(q2_bins[i][1]+q2_bins[i][0])/2]
        q2_width += [(q2_bins[i][1]-q2_bins[i][0])/2]
    
    S9_SM = np.array([-0.000701,-0.000786,-0.000735,-0.000706,-0.000715,0.000292, 0.000169, 0.000449	,-0.000739,0.000242])
    S9_SM_err = np.array([0.007420,0.005376,0.002259,0.004633,0.007751,0.008629,0.013439,0.000414,0.002428,0.009785])
    plt.clf()
    for k in range(len(ind)):
        i = ind[k]
        x, y, width, height = (q2_centre[i]-q2_width[i], S9_SM[i]-S9_SM_err[i], q2_width[i]*2, S9_SM_err[i]*2)
        rect = patches.Rectangle((x, y), width, height, alpha=0.4, color='m')
        plt.gca().add_patch(rect)
    for j in range(len(over)):
        i = over[j]
        x, y, width, height = (q2_centre[i]-q2_width[i], S9_SM[i]-S9_SM_err[i], q2_width[i]*2, S9_SM_err[i]*2)
        rect = patches.Rectangle((x, y), width, height, alpha=0.4, color='c')
        plt.gca().add_patch(rect)
    
    plt.errorbar(q2_centre, S9, yerr=S9_err, xerr=q2_width, fmt='ko', markersize=5, capsize=0, label='Filtered Data')
    plt.xlabel(r'$q^2$')
    plt.ylabel(r'$S9$')
    plt.grid('both')
    plt.legend()
    plt.show()

S9 = np.array(s9_array)[:,0] #!!
S9_err = np.array(s9_array)[:,1] #!!
s9_obs_plot(S9, S9_err)