# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:06:06 2022

@author: dd719

this file straps boots

  #%%%%#%%%%%%#(#&%%%                     #%%%%#%%%%%%#(#&%%%                   
  %     %,   ,%     %                     %.    %,   ,%     %                   
  %*    %.%(%,%    .%                     %*    %.%(%,%    .%                   
  #%    #%( ,%%    *%                     #%    #%( ,%%    ,%                   
  *%               (#                     *%               (%                   
   %               %(                      %               %(                   
   %(              %/                      %(              %/                   
   (%              %,                      (%              %,                   
    %           ## %.                       %           ## %.                   
    %%%#,     ,%#  %*                       %%%#,     ,%#  %*                   
   %(               %                      %(               %                   
   %                .%.                    %                .%.                 
  (&                  %%                  (&                  %%                
  %*                    .%%%%%%%%%%%%%*   %*                    .%%%%%%%%%%%%%* 
  %                                   ,%  %.                                  ,%
  %(      .                          /%   %(      .                          /% 
   %(     %(%%*             ,(%%%%#*       %(     %(%%*             ,(%%%%#*    
      ,*///      .,*/*,.                      ,*///      .,*/*,.                


"""
#%%
#import the relevant modules
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from scipy.optimize import curve_fit
import MODULE_curve_fitting_functions as mcf
from iminuit import Minuit as minuit
#%%
original_dataset = pd.read_pickle('total_dataset_binary3.pkl')
original_dataset.reset_index(drop=True, inplace=True)
sample_proportion = 0.98
repetitions = 20

#%%
for i in range(1,repetitions+1):
    indices = random.sample(range(0, len(original_dataset)), int(np.round(len(original_dataset))*sample_proportion))
    columns = list(np.arange(0, 82, 1))
    
    reduced_dataset = original_dataset.iloc[indices,columns]
    reduced_dataset.reset_index(drop=True, inplace=True)
    
    reduced_dataset.to_pickle("total_dataset_binary3_{}%_retained_trial_{}.pkl".format(int(sample_proportion*100),i))

#%%
"""
following is all the contents of kevin's code but iterated over:
"""

#this resets the txt files storing outputs, only re-run once you have renamed them
#(or comment the following 4 lines out to append to current files)

open('afb.txt', 'w').close()
open('afb_err.txt', 'w').close()
open('fl.txt', 'w').close()
open('fl_err.txt', 'w').close()

all_afb_arrays = []
all_afb_err_arrays = []

all_fl_arrays = []
all_fl_err_arrays = []

for i in range(1,repetitions+1):
    
    files = ["total_dataset_binary3_{}%_retained_trial_{}".format(int(sample_proportion*100),i), 'acceptance_mc_binary3']
    
    #files = ['total_dataset_binary3', 'acceptance_mc_binary3']
    DF, BINS, DF_MC, BINS_MC = mcf.read_data_into_bins(files[0], files[1], 0)
    variables = ['phi', 'costhetal', 'costhetak', 'q2']
    
    
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
    
    def chebyshev(x, a0, a1, a2):
        return a0 + a1 * x + a2 * (2 * x ** 2 -1)
    
    def norm_chebyshev(x, a0, a1, a2): #!!
        y = chebyshev(x, a0, a1, a2)
        x = np.linspace(-1,1,5000) #!!
        norm_factor = integrate2(x, chebyshev(x, a0, a1, a2))
        return y/norm_factor
    
    def pdf(fl, afb, _bin, cos_theta_l): #!!
        ctl = cos_theta_l 
        c2tl = 2 * ctl ** 2 - 1 
        scalar_array = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) + 8/3 * afb * ctl)  
        return scalar_array * mcf.polynomial(ctl, *X_PARAMS[_bin])
    
    def norm_pdf_without_bg(fl, afb, _bin, cos_theta_l): #!!
        cos_theta_l_acc = np.linspace(-1,1,5000) #!!
        norm_factor = integrate2(cos_theta_l_acc, pdf(fl=fl, afb=afb, _bin = _bin, cos_theta_l=cos_theta_l_acc))
        return pdf(fl, afb, _bin, cos_theta_l)/norm_factor
    
    def pdf_with_bg(fl, afb, a0, a1, a2, _bin, cos_theta_l): #!!!
        R = BG_RATIO[_bin]/(1-BG_RATIO[_bin]) # BG_RATIO is defined in the next cell
        norm_scalar_array = norm_pdf_without_bg(fl, afb, _bin, cos_theta_l) 
        return norm_scalar_array +  R*norm_chebyshev(cos_theta_l, a0, a1, a2)
    
    def norm_pdf(fl, afb, a0, a1, a2, _bin, cos_theta_l): #!! 
        scalar_array = pdf_with_bg(fl, afb, a0, a1, a2, _bin, cos_theta_l)
        cos_theta_l_acc = np.linspace(-1,1,5000)
        norm_factor = integrate2(cos_theta_l_acc, pdf_with_bg(fl, afb, a0, a1, a2, _bin, cos_theta_l_acc))
        normalised_scalar_array = scalar_array /norm_factor  
        return normalised_scalar_array
    
    
    def log_likelihood_ctl(fl, afb, a0, a1, a2, _bin): #!!
        # _BIN = BINS[int(_bin)] # 1st option: all data in q2 bins
        _BIN = BINS[int(_bin)][(BINS[int(_bin)].B0_MM >= 5240)  & (BINS[int(_bin)].B0_MM <=5355)] # 2nd option: B0_MM cuts in q2 bins
        ctl = _BIN['costhetal']
        normalised_scalar_array = norm_pdf(fl=fl, afb=afb, a0=a0, a1=a1, a2=a2, _bin = int(_bin), cos_theta_l= ctl)
        dummy_x = np.linspace(-1,1,1000)
        check = pdf(fl, afb, int(_bin), dummy_x)
        physicalness = 1
        if np.sum([check < 0]):
            physicalness = 1 + np.sum([check < 0])/10
        
        return - np.sum(np.log(normalised_scalar_array)) * physicalness
    
    def norm_factor_signal_bg(fl, afb, a0, a1, a2, _bin): #!!
        cos_theta_l = np.linspace(-1,1,5000) #!!
        Y = pdf_with_bg(fl, afb, a0, a1, a2, _bin, cos_theta_l)
        norm_factor = integrate2(cos_theta_l, Y)
        return norm_factor
    
    def chi_sq(x, data, fls, afbs, a0, a1, a2, b):
        X = 0
        N = sum(data)
        dx = x[1] - x[0]
        for i in range(len(data)):
            X += ((N*norm_pdf(fls, afbs, a0, a1, a2, b, cos_theta_l = x[i])*dx - data[i]) ** 2) / data[i]
        return X/(len(data) + 5)
    
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
    
    for i in range(B):
        ### Extract acceptance params
        ### Use plt.hist() to show plots. Requires one more output argument: n, e, _= plt.hist()
        n, e = np.histogram(BINS_MC[i][v], bins = NO_OF_BINS) # plt.hist(BINS_MC[i][v], bins = NO_OF_BINS)
        _x, _y, x_params, x_cov = mcf.accept_fit(n, e)
        X_PARAMS.append(x_params)
        X_COV.append(x_cov)
        # plt.plot(_x, _y)
        # plt.show()
    
    
    for i in range(B):
        ### Extract chebyshev params
        n, e, _ = plt.hist(BINS[i].B0_MM, bins = 25)
        ec = mcf.center_bins(e)
        ge_params, ge_cov = curve_fit(mcf.gaussian_exponent_fit, ec[3:], n[3:], 
                                      p0 = [200, 5280, 180, exp_guess[0][i], exp_guess[1][i]])
        plt.plot(np.arange(5100, 5700), mcf.gaussian_exponent_fit(np.arange(5100, 5700), *ge_params))
        plt.show()
        SIGNAL = 0
        BG = 0
        for j in [i for i in ec if i>5200 and i <5355]:
            SIGNAL += mcf.Gauss(j, *ge_params[:3])
            BG += mcf.exp_decay(j, *ge_params[3:])
        BG_RATIO.append(BG/(SIGNAL + BG))
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
    
    
    fl_array = [] #!!
    afb_array = [] #!!
    
    for b in range(10):            
        log_likelihood_ctl.errordef = minuit.LIKELIHOOD
        decimal_places = 3
        ST = [SM_FL[b],SM_AFB[b], CHEBY_PARAMS[b][0], CHEBY_PARAMS[b][1], CHEBY_PARAMS[b][2]] #!!
        SD = [0,0, np.sqrt(CHEBY_COV[b][0][0]), np.sqrt(CHEBY_COV[b][1][1]), np.sqrt(CHEBY_COV[b][2][2])] 
        fls, fl_errs = [], [] #!!
        afbs, afb_errs = [], [] #!!
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
        m.limits=((-1.0, 1.0), (-1.0, 1.0), #!!
                  (ST[2], ST[2]), (ST[3], ST[3]), (ST[4], ST[4]), None) 
        m.migrad()
        m.hesse()
        bin_results_to_check = m
        fls.append(m.values[0]) #!!
        afbs.append(m.values[1]) #!!
        a0s.append(m.values[2])
        a1s.append(m.values[3])
        a2s.append(m.values[4])
        fl_errs.append(m.errors[0]) #!!
        afb_errs.append(m.errors[1]) #!!
        a0_errs.append(m.errors[2])
        a1_errs.append(m.errors[3])
        a2_errs.append(m.errors[4])
        print(f"Bin {b}: Fl = {np.round(fls, decimal_places)} pm {np.round(fl_errs, decimal_places)},", 
              f" Afb = {np.round(afbs, decimal_places)} pm {np.round(afb_errs, decimal_places)}. Function minimum considered valid: {m.fmin.is_valid}")
        
        X = BINS[b][v].sort_values()
        signal = norm_pdf_without_bg(fls[0], afbs[0], b, X)                                                                        
        background = BG_RATIO[b]/(1-BG_RATIO[b]) * norm_chebyshev(X, a0s[0], a1s[0], a2s[0])
        norm_factor = norm_factor_signal_bg(fls[0], afbs[0], a0s[0], a1s[0], a2s[0], b)
        h, e, _ = plt.hist(X, bins = NO_OF_BINS, density = True, histtype = 'step', color = 'tab:blue', range = [-1,1])
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
        
        # print(chi_sq(mcf.center_bins(e), h, fls[0], afbs[0], a0s[0], a1s[0], a2s[0], b))
    
        fl_array.append([fls[0], fl_errs[0]])
        afb_array.append([afbs[0], afb_errs[0]])
    
    
    import matplotlib.patches as patches
    """
    From David's code, see that for reference
    """
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
    
    FL = np.array(fl_array)[:,0] #!!
    FL_err = np.array(fl_array)[:,1] #!!
    fl_obs_plot(FL, FL_err)
    
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
    afb_obs_plot(AFB, AFB_err)
    
    
    all_afb_arrays.append(AFB)
    all_afb_err_arrays.append(AFB_err)

    all_fl_arrays.append(FL)
    all_fl_err_arrays.append(FL_err)
    

    with open('afb.txt', 'a') as f:
        f.write('{}\n'.format(AFB))
        
    with open('afb_err.txt', 'a') as f:
        f.write('{}\n'.format(AFB_err))
        
    with open('fl.txt', 'a') as f:
        f.write('{}\n'.format(FL))
        
    with open('fl_err.txt', 'a') as f:
        f.write('{}\n'.format(FL_err))


