# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 17:34:10 2022

@author: David Vico Benet
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.polynomial import legendre as lg


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
acc_filt = pd.read_csv('datasets/acceptance_filtered.csv')
acc_filt.name = 'Accept Filtered'
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

def cosl_bin(data_frame, bin_n, bin_N):
    q2_bins = [[0.1, 0.98], [1.1, 2.5], [2.5, 4.0], [4.0, 6.0], [6.0, 8.0], [15.0, 17.0],
           [17.0, 19.0], [11.0, 12.5], [1.0, 6.0], [15.0, 17.9]]
    r = q2_bins[bin_n]
    q2bin = data_frame[data_frame.q2.between(r[0], r[1])]
    q2bin.name = r'$q^2 \in [%.2f, %.2f]$' % (r[0], r[1])
    
    x = q2bin['costhetal'].to_numpy()
    N, b = np.histogram(x, bin_N, range=[-1, 1])
    N_w, b_w = np.histogram(x, bin_N, range=[-1, 1], density=True)
    err = np.sqrt(N)/(N[0]/N_w[0])
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

# cosl_bin(acc_filt, 0, 20)
#%%
def MoM_acc(data_frame, bin_range, bin_num):
    q2bin = data_frame[data_frame.q2.between(bin_range[0], bin_range[1])]
    q2bin.name = r'$q^2 \in [%.2f, %.2f]$' % (bin_range[0], bin_range[1])
    
    cosl = q2bin['costhetal'].to_numpy()
    moments = [1]
    for i in range(6):
        vals = cosl**(i+1)
        mu = np.average(vals)
        moments += [mu]
    moments = np.array(moments)
    
    p = np.zeros((7, 7))
    np.fill_diagonal(p, 1.0)
    
    M = np.zeros((7, 7))
    for j in range(len(p)):
        if j == 0:
            for i in range(len(p)):
                p_int = lg.legint(p[i], lbnd=-1)
                coef = lg.legval(1, p_int)
                M[j][i] = coef
        else:
            for l in range(len(p)):
                p_v = p[l]
                for k in range(j):
                    p_v = lg.legmulx(p_v)
                p_int = lg.legint(p_v, lbnd=-1)
                coef = lg.legval(1, p_int)
                M[j][l] = coef
                
    alph = np.linalg.solve(M, moments)
    # print(alph)
    # This checks that our acceptance fit is normalised, ie integrates to 1
    print(lg.legval(1, lg.legint(alph, lbnd=-1, k=0)))
    
    N, b = np.histogram(cosl, 20, range=[-1, 1])
    N_w, b_w = np.histogram(cosl, 20, range=[-1, 1], density=True)
    err = np.sqrt(N)/(N[0]/N_w[0])
    b_vals = (b[1:] + b[:-1]) / 2
    width = b[1]-b[0]
    plt.errorbar(b_vals, N_w, xerr = width/2, yerr=err, fmt='ko', capsize=0, markersize=5, label='Filtered Acc')
    
    leg_MoM = lg.legval(b_vals, alph)
    plt.plot(b_vals, leg_MoM, 'r--', label='MoM Legendre')
    
    # Chi^2 tests for the legendre MoM acceptance.
    chi2_uncert = sum((N_w - leg_MoM)**2/(err**2))
    print(chi2_uncert)
    chi2_pearson = sum((N_w - leg_MoM)**2/(leg_MoM))
    print(chi2_pearson)
    chi2_neyman = sum((N_w - leg_MoM)**2/(N_w))
    print(chi2_neyman)
    # Combined chi2_pearson and neyman
    print(1/3 * (chi2_neyman + 2*chi2_pearson))
    
    plt.legend()
    title = str(bin_num) + str(q2bin.name) + r' $\cos\theta_l$ Normalised Histogram'
    plt.title(title)
    ylabel = 'Candidate density / %.1f' % (width)
    plt.ylabel(ylabel)
    plt.xlabel(r'$\cos\theta_l$')
    plt.grid('both')
    plt.show()

def MoM_bin(data_frame, bin_n):
    q2_bins = [[0.1, 0.98], [1.1, 2.5], [2.5, 4.0], [4.0, 6.0], [6.0, 8.0], [15.0, 17.0],
           [17.0, 19.0], [11.0, 12.5], [1.0, 6.0], [15.0, 17.9]]
    bin_num = 'Bin %d ' % (bin_n)
    MoM_acc(data_frame, q2_bins[bin_n], bin_num)
        
def MoM_all(data_frame):
    for i in range(10):
        MoM_bin(data_frame, i)
MoM_all(acc_filt)