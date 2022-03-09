# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 09:37:49 2022

@author: David Vico Benet
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from scipy.optimize import curve_fit

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
tot_filt = pd.read_csv('datasets/total_filtered.csv')
tot_filt.name = 'CF Filtered'

tot_unfilt = pd.read_pickle('data/total_dataset.pkl')
tot_unfilt.name = 'Dataset'
#%%
def masshist(data_frame, label, bin_range, bin_N):
    # bins = np.linspace(bin_range[0], bin_range[1], bin_N)
    plt.clf()
    
    x = data_frame[str(label)].to_numpy()
    N, b = np.histogram(x, bin_N, range=bin_range)
    N_w, b_w = np.histogram(x, bin_N, range=bin_range, density=True)
    err = np.sqrt(N)/(N[0]/N_w[0])
    
    b_vals = (b[1:] + b[:-1]) / 2
    width = b[1]-b[0]
    plt.errorbar(b_vals, N_w, xerr=width/2, yerr=err, fmt='ko', capsize=0, markersize=3, label=data_frame.name)
    
    # occur1 = x > bin_range[1]
    # occur2 = x < bin_range[0]
    # left_out = occur1.sum() + occur2.sum()
    # print('Data set %s: %d/%d points left out for given bins, %f' %(data_frame.name, left_out,len(x), left_out/len(x)))
    plt.legend()
    title = str(label) + ' Nomralised Histogram'
    plt.title(title)
    plt.xlabel(r'Reconstructed $B^0$ mass (MeV/c$^2$)')
    ylabel = r'Candidate Density / %.1f MeV/c$^2$' % (width)
    plt.ylabel(ylabel)
    plt.grid('both')
    plt.show()

def masshist_bin(data_frame, bin_N, bin_n):
    q2_bins = [[0.1, 0.98], [1.1, 2.5], [2.5, 4.0], [4.0, 6.0], [6.0, 8.0], [15.0, 17.0],
           [17.0, 19.0], [11.0, 12.5], [1.0, 6.0], [15.0, 17.9]]
    r = q2_bins[bin_n]
    q2bin = data_frame[data_frame.q2.between(r[0], r[1])]
    q2bin.name = r'$q^2 \in [%.2f, %.2f]$' % (r[0], r[1])
    masshist(q2bin, 'B0_MM', [5170, 5700], bin_N)

def masshist_allbins(dataframe, bin_N):
    for i in range(10):
        masshist_bin(dataframe, bin_N, i)

# masshist_allbins(tot_filt, 50)
#%%
# Fitting mass spetrum with a gaussian + linear backg fit

def gauss(x, A, mu, sig):
    g = A / (np.sqrt(2*np.pi) * sig) * np.exp( -(x-mu)**2/(2*sig**2))
    return g

def gauss_backg(x, A, mu, sig, m, b):
    g = A * np.exp( -(x-mu)**2/(2*sig**2)) + m*x + b
    return g

def fittotnormalhistogram(data_frames, label, bin_range, bin_N):
    # bins = np.linspace(bin_range[0], bin_range[1], bin_N)
    plt.clf()
    for i in range(len(data_frames)):
        x = data_frames[i][str(label)].to_numpy()
        N, b = np.histogram(x, bin_N, range=bin_range)
        N_w, b_w = np.histogram(x, bin_N, range=bin_range, density=True)
        err = np.sqrt(N)/(N[0]/N_w[0])
        b_vals = (b[1:] + b[:-1]) / 2
        plt.errorbar(b_vals, N_w, yerr=err, fmt='o', capsize=3, markersize=2, label=data_frames[i].name)
        width = b[1]-b[0] 
        # plt.bar(b_vals, N_w, width=width, alpha=0.2, linewidth=0)
        params, cov = curve_fit(gauss, b_vals, N_w, p0 = [0.1, 5280, 30])
        print(params)
        # print(cov)
        print(np.sqrt(abs(cov)))
        # plt.plot(b_vals, gauss(b_vals, params[0], params[1], params[2]), 'r--')
        
        params2, cov2 = curve_fit(gauss_backg, b_vals, N_w, p0 = [0.1, 5280, 30, -4e-5, 0.2])
        print(params2)
        print(np.sqrt(abs(cov2)))
        line = params2[3]*np.array(b_vals) + params2[4]
        plt.plot(b_vals, line, 'k--')
        plt.plot(b_vals, gauss_backg(b_vals, params2[0], params2[1], params2[2], params2[3], params2[4]), 'r--')
        
        occur1 = x > bin_range[1]
        occur2 = x < bin_range[0]
        left_out = occur1.sum() + occur2.sum()
        print('Data set %s: %d/%d points left out for given bins, %f' %(data_frames[i].name, left_out,len(x), left_out/len(x)))
    plt.legend()
    title = str(label) + ' Nomralised Histogram'
    plt.title(title)
    plt.xlabel(r'Reconstructed $B^0$ mass (MeV/c$^2$)')
    ylabel = r'Frac Candidate Count / %.1f MeV/c$^2$' % (width)
    plt.ylabel(ylabel)
    plt.grid('both')
    plt.show()
#%%
# Fitting mass spetrum with a gaussian + exp decay backg fit

def gauss_exp(x, A, mu, sig, B, m, b, c):
    g = A * np.exp( -(x-mu)**2/(2*sig**2)) + B*np.exp(-m*(x-b)) + c
    return g

def fittotnormalhistogram_exp(data_frame, label, bin_range, bin_N):
    # bins = np.linspace(bin_range[0], bin_range[1], bin_N)
    plt.clf()
    
    x = data_frame[str(label)].to_numpy()
    N, b = np.histogram(x, bin_N, range=bin_range)
    N_w, b_w = np.histogram(x, bin_N, range=bin_range, density=True)
    err = np.sqrt(N)/(N[0]/N_w[0])
    
    b_vals = (b[1:] + b[:-1]) / 2
    width = b[1]-b[0]
    plt.errorbar(b_vals, N_w, xerr=width/2, yerr=err, fmt='ko', capsize=0, markersize=3, label=data_frame.name)
    
    # plt.bar(b_vals, N_w, width=width, alpha=0.2, linewidth=0)
    
    params, cov = curve_fit(gauss_exp, b_vals, N_w, p0 = [0.018, 5280, 20, 0.06, 0.02, 5010, 0.002])
    print(params)
    for i in range(len(params)):
        print(np.sqrt(abs(cov[i][i])))
    
    exp = params[3]*np.exp(-params[4]*(b_vals-params[5])) + params[6]
    plt.plot(b_vals, exp, 'k--')
    plt.fill_between(b_vals, 0, exp, color = 'cyan', alpha=0.2)
    
    plt.plot(b_vals, gauss_exp(b_vals, params[0], params[1], params[2], params[3], params[4], params[5], params[6]), 'r--')
    
    
    
    occur1 = x > bin_range[1]
    occur2 = x < bin_range[0]
    left_out = occur1.sum() + occur2.sum()
    print('Data set %s: %d/%d points left out for given bins, %f' %(data_frame.name, left_out,len(x), left_out/len(x)))
    plt.legend()
    title = str(label) + ' Nomralised Histogram'
    plt.title(title)
    plt.xlabel(r'Reconstructed $B^0$ mass (MeV/c$^2$)')
    ylabel = r'Candidate Density / %.1f MeV/c$^2$' % (width)
    plt.ylabel(ylabel)
    plt.grid('both')
    plt.show()

# fittotnormalhistogram_exp(tot_filt, 'B0_MM', [5170, 5700], 50)
#%%
# Plotting the costhetal distribution for different q^2 bins
def cosl_histogram(data_frame, label, bin_range, bin_N, bin_name):
    # bins = np.linspace(bin_range[0], bin_range[1], bin_N)
    plt.clf()
    for i in range(len(data_frame)):
        q2bin = data_frame[i][data_frame[i].q2.between(bin_range[0], bin_range[1])]
        q2bin.name = r'$q^2 \in [%.2f, %.2f]$' % (bin_range[0], bin_range[1])
        
        x = q2bin[str(label)].to_numpy()
        N, b = np.histogram(x, bin_N, range=[-1, 1])
        err = np.sqrt(N)
        b_vals = (b[1:] + b[:-1]) / 2
        width = b[1]-b[0]
        plt.errorbar(b_vals, N, xerr = width/2, yerr=err, fmt='ko', capsize=0, markersize=5, label=data_frame[i].name)
    plt.legend()
    title = str(bin_name) +' ' + str(q2bin.name) + r' $\cos (\theta_l)$ Normalised Histogram'
    plt.title(title)
    ylabel = 'Fractional Candidates / %.1f' % (width)
    plt.ylabel(ylabel)
    plt.xlabel(r'\cos (\theta_l)')
    plt.grid('both')
    plt.show()
    
def cosl_histogram_normal(data_frame, label, bin_range, bin_N, bin_name):
    # bins = np.linspace(bin_range[0], bin_range[1], bin_N)
    plt.clf()
    for i in range(len(data_frame)):
        q2bin = data_frame[i][data_frame[i].q2.between(bin_range[0], bin_range[1])]
        q2bin.name = r'$q^2 \in [%.2f, %.2f]$' % (bin_range[0], bin_range[1])
        
        x = q2bin[str(label)].to_numpy()
        N, b = np.histogram(x, bin_N, range=[-1, 1])
        N_w, b_w = np.histogram(x, bin_N, range=[-1, 1], density=True)
        err = np.sqrt(N)/(N[0]/N_w[0])
        b_vals = (b[1:] + b[:-1]) / 2
        width = b[1]-b[0]
        plt.errorbar(b_vals, N_w, xerr = width/2, yerr=err, fmt='o', capsize=0, markersize=5, label=data_frame[i].name)
    plt.legend()
    title = str(bin_name) +' ' + str(q2bin.name) + r' $\cos (\theta_l)$ Normalised Histogram'
    plt.title(title)
    ylabel = 'Candidate Density / %.1f' % (width)
    plt.ylabel(ylabel)
    plt.xlabel(r'$\cos (\theta_l)$')
    plt.grid('both')
    plt.show()
    
def cosl_histall(data_frame, norm = True):
    q2_bins = [[0.1, 0.98], [1.1, 2.5], [2.5, 4.0], [4.0, 6.0], [6.0, 8.0], [15.0, 17.0],
           [17.0, 19.0], [11.0, 12.5], [1.0, 6.0], [15.0, 17.9]]
    if norm == True:
        for i in range(len(q2_bins)):
            b = 'Bin %d' % (i)
            cosl_histogram_normal(data_frame, 'costhetal', q2_bins[i], 20, b)
    else:
        for i in range(len(q2_bins)):
            b = 'Bin %d' % (i)
            cosl_histogram(data_frame, 'costhetal', q2_bins[i], 20, b)

def cosl_bin(data_frame, bin_N, bin_n):
    q2_bins = [[0.1, 0.98], [1.1, 2.5], [2.5, 4.0], [4.0, 6.0], [6.0, 8.0], [15.0, 17.0],
           [17.0, 19.0], [11.0, 12.5], [1.0, 6.0], [15.0, 17.9]]
    r = q2_bins[bin_n]
    q2bin = data_frame[data_frame.q2.between(r[0], r[1])]
    q2bin.name = r'$q^2 \in [%.2f, %.2f]$' % (r[0], r[1])
    
    x = q2bin['costhetal'].to_numpy()
    N, b = np.histogram(x, bin_N, range=[-1, 1])
    N_w, b_w = np.histogram(x, bin_N, range=[-1, 1], density=True)
    err = np.sqrt(N)/(N[5]/N_w[5])
    b_vals = (b[1:] + b[:-1]) / 2
    width = b[1]-b[0]
    plt.errorbar(b_vals, N_w, xerr = width/2, yerr=err, fmt='ko', capsize=0, markersize=5, label=data_frame.name)
    plt.legend()
    bin_name = 'Bin %d' % (bin_n)
    title = str(bin_name) +' ' + str(q2bin.name) + r' $\cos (\theta_l)$ Normalised Histogram'
    plt.title(title)
    ylabel = 'Candidate Density / %.1f' % (width)
    plt.ylabel(ylabel)
    plt.xlabel(r'$\cos (\theta_l)$')
    plt.grid('both')
    plt.show()

# cosl_bin(tot_filt, 20, 1)
#%%
high_mass = tot_unfilt[tot_unfilt['B0_MM'] > 5355]
high_mass.name = r'$m>5355$'

cosl_histall([high_mass], norm=False)

#%%
# Standard model predictions
# FL_SM = np.array([0.296448, 0.760396, 0.796265, 0.711290, 0.606965, 
#                   0.348441, 0.328081, 0.435190, 0.747644, 0.340156])
# FL_SM_err = np.array([0.050642, 0.043174, 0.033640, 0.049096, 0.050785, 
#                       0.035540, 0.027990, 0.036637, 0.039890, 0.024826])

# AFB_SM = np.array([-0.097052, -0.137987, -0.017385, 0.122155, 0.239939, 
#                    0.401914, 0.318391, 0.391390, 0.004929, 0.367672])
# AFB_SM_err = np.array([0.008421, 0.031958, 0.029395, 0.039619, 0.047281, 
#                        0.030141, 0.033931, 0.023627, 0.028058, 0.032230])
#%%
# Values obtained from CF hard cuts
FL = np.array([0.081, 0.608, 0.704, 0.648, 0.568, 
               0.38, 0.399, 0.539, 0.635, 0.384])
FL_err = np.array([0.116, 0.18, 0.132, 0.086, 0.07, 
                   0.067, 0.086, 0.066, 0.067, 0.057])

AFB = np.array([-0.052, -0.27, 0.03, 0.058, 0.237, 
                0.344, 0.268, 0.316, -0.011, 0.342])
AFB_err = np.array([0.066, 0.083, 0.071, 0.05, 0.041, 
                    0.039, 0.053, 0.036, 0.038, 0.034])
#%%
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
def afb_obs_plot(AFB, AFB_err):
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

afb_obs_plot(AFB, AFB_err)