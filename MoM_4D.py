# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:09:21 2022

@author: David Vico Benet
"""
#%%
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
# acc_filt = pd.read_csv('datasets/acceptance_filtered.csv')
# acc_filt.name = 'Accept Filtered'

acc_filt_noq2 = pd.read_pickle('Datasets/acc_filt_noq2.pkl')
acc_filt_noq2.name = 'Acceptance 2'

#%%
# 4D Acceptance
def acceptance_c(acc_filt_noq2, r):
    q2 = (acc_filt_noq2[acc_filt_noq2.q2.between(r[0], r[1])]['q2'].to_numpy() - 0.5*(r[1]+r[0])) / (0.5*(r[1]-r[0]))
    cosl = acc_filt_noq2[acc_filt_noq2.q2.between(r[0], r[1])]['costhetal'].to_numpy()
    cosk = acc_filt_noq2[acc_filt_noq2.q2.between(r[0], r[1])]['costhetak'].to_numpy()
    phi = acc_filt_noq2[acc_filt_noq2.q2.between(r[0], r[1])]['phi'].to_numpy() / np.pi
    N = len(q2)
    # Order has +1 added to account for 0th order
    cosl_order = 5
    cosk_order = 6
    phi_order = 7
    q2_order = 6
    c = np.zeros((cosl_order, cosk_order, phi_order, q2_order))
    
    # Legendre polynomials as arrays for np.polynomial interface
    p = np.zeros((7, 7))
    np.fill_diagonal(p, 1.0)
    
    for i in range(cosl_order):
        for j in range(cosk_order):
            for m in range(phi_order):
                for n in range(q2_order):
                    L_i = lg.legval(cosl, p[i])
                    L_j = lg.legval(cosk, p[j])
                    L_m = lg.legval(phi, p[m])
                    L_n = lg.legval(q2, p[n])
                    c_ijmn = sum(L_i * L_j * L_m * L_n)
                    c_ijmn *= ((2*i + 1) * (2*j + 1) * (2*m + 1) * (2*n + 1))
                    c_ijmn *= 1/(16*N)
                    c[i][j][m][n] = c_ijmn
                    # print(c_ijmn, i, j, m, n)
    
    return c

def acceptance_q2(acc_filt_noq2, c, r):
    ''' c is calculated with r=[0.0, 19.0] by default '''
    cosl_order = 5
    cosk_order = 6
    phi_order = 7
    q2_order = 6
    
    p = np.zeros((7, 7))
    np.fill_diagonal(p, 1.0)
    
    c_q2 = np.zeros((q2_order))
    for n in range(q2_order):
        c_n = 0
        for j in range(cosk_order):
            for m in range(phi_order):
                for i in range(cosl_order):
                    L_j = lg.legval(1, lg.legint(p[j], lbnd=-1, k=0))
                    L_m = lg.legval(1, lg.legint(p[m], lbnd=-1, k=0))
                    L_i = lg.legval(1, lg.legint(p[i], lbnd=-1, k=0))
                    prod = L_j * L_m * L_i * c[i][j][m][n]
                    c_n += prod
        c_q2[n] = c_n
    
    # Acceptance is normalised in the rescaled range -1 to 1
    q_vals = np.linspace(r[0], r[1], 100)
    q_vals_shift = (q_vals - (0.5*(r[1]+r[0])))/(0.5*(r[1]-r[0]))
    leg_MoM_q2 = lg.legval(q_vals_shift, c_q2)
    plt.plot(q_vals, leg_MoM_q2, 'r--', label='MoM Legendre')
    
    q2_data = acc_filt_noq2[acc_filt_noq2.q2.between(r[0], r[1])]['q2'].to_numpy()

    q2_data_shift = (q2_data - (0.5*(r[1]+r[0])))/(0.5*(r[1]-r[0]))
    n_q2, b_q2 = np.histogram(q2_data_shift, 50, range=[-1, 1])
    n_q2_w, b_q2_w = np.histogram(q2_data_shift, 50, range=[-1, 1], density=True)
    b_q2_vals_shift = (b_q2[1:] + b_q2[:-1]) / 2
    b_q2_vals = (0.5*(r[1]-r[0])) * b_q2_vals_shift + (0.5*(r[1]+r[0]))
    n_q2_w_err = np.sqrt(n_q2)/(n_q2[0]/n_q2_w[0])
    plt.errorbar(b_q2_vals, n_q2_w, xerr = (b_q2_vals[1]-b_q2_vals[0])/2, yerr=n_q2_w_err, 
                 fmt='ko', capsize=0, markersize=4, label = 'Unfiltered Acceptance')
        

    title = r' $q^2$ Histogram (Normalised in the rescaled range -1 to 1)'
    plt.title(title)
    ylabel = 'Efficency'
    plt.ylabel(ylabel)
    plt.xlabel(r'$q^2$')
    plt.grid('both')
    plt.legend()
    plt.show()   
    
    
    return c_q2

def acceptance_cosl(acc_filt_noq2, c, r):
    ''' Assuming r=[0.0, 19.0] '''
    cosl_order = 5
    cosk_order = 6
    phi_order = 7
    q2_order = 6
    
    p = np.zeros((7, 7))
    np.fill_diagonal(p, 1.0)
    
    c_cosl = np.zeros((cosl_order))
    for i in range(cosl_order):
        c_i = 0
        for j in range(cosk_order):
            for m in range(phi_order):
                for n in range(q2_order):
                    L_j = lg.legval(1, lg.legint(p[j], lbnd=-1, k=0))
                    L_m = lg.legval(1, lg.legint(p[m], lbnd=-1, k=0))
                    L_n = lg.legval(1, lg.legint(p[n], lbnd=-1, k=0))
                    prod = L_j * L_m * L_n * c[i][j][m][n]
                    c_i += prod
        c_cosl[i] = c_i
    
    # Acceptance is normalised in the rescaled range -1 to 1
    cosl_vals = np.linspace(-1, 1, 100)
    leg_MoM_cosl = lg.legval(cosl_vals, c_cosl)
    plt.plot(cosl_vals, leg_MoM_cosl, 'r--', label='MoM Legendre')
    
    cosl_data = acc_filt_noq2[acc_filt_noq2.q2.between(r[0], r[1])]['costhetal'].to_numpy()

    n_cosl_o, b_cosl_o = np.histogram(cosl_data, 50, range=[-1, 1]) # unnormalised
    n_cosl, b_cosl = np.histogram(cosl_data, 50, range=[-1, 1],density=True)
    b_cosl_vals = (b_cosl[1:] + b_cosl[:-1]) / 2
    n_cosl_err = np.sqrt(n_cosl)/(n_cosl_o[0]/n_cosl[0])
    
    plt.errorbar(b_cosl_vals, n_cosl, xerr = (b_cosl[1]-b_cosl[0])/2, yerr=n_cosl_err, fmt='ko', capsize=0, markersize=4, label = 'Unfiltered Acceptance')
        

    if bin_num == 'all':
        title = r' Costhetal Normalised Histogram (All bins)'
    else: 
        title = f'Costhetal Normalised Histogram (bin {bin_num})'
    
    plt.title(title)
    ylabel = 'Efficency'
    plt.ylabel(ylabel)
    plt.xlabel(r'$costhetal$')
    plt.grid('both')
    plt.legend()
    plt.show()   
    
    return c_cosl

def acceptance_cosk(acc_filt_noq2, c, r):
    ''' Assuming r=[0.0, 19.0] '''
    cosl_order = 5
    cosk_order = 6
    phi_order = 7
    q2_order = 6
    
    p = np.zeros((7, 7))
    np.fill_diagonal(p, 1.0)
    
    c_cosk = np.zeros((cosk_order))
    for j in range(cosk_order):
        c_j = 0
        for i in range(cosl_order):
            for m in range(phi_order):
                for n in range(q2_order):
                    L_i = lg.legval(1, lg.legint(p[i], lbnd=-1, k=0))
                    L_m = lg.legval(1, lg.legint(p[m], lbnd=-1, k=0))
                    L_n = lg.legval(1, lg.legint(p[n], lbnd=-1, k=0))
                    prod = L_i * L_m * L_n * c[i][j][m][n]
                    c_j += prod
        c_cosk[j] = c_j
    
    # Acceptance is normalised in the rescaled range -1 to 1
    cosk_vals = np.linspace(-1, 1, 100)
    leg_MoM_cosk = lg.legval(cosk_vals, c_cosk)
    plt.plot(cosk_vals, leg_MoM_cosk, 'r--', label='MoM Legendre')
    
    cosk_data = acc_filt_noq2[acc_filt_noq2.q2.between(r[0], r[1])]['costhetak'].to_numpy()
    n_cosk_o, b_cosk_o = np.histogram(cosk_data, 50, range=[-1, 1]) # unnormalised
    n_cosk, b_cosk = np.histogram(cosk_data, 50, range=[-1, 1],density=True)
    b_cosk_vals = (b_cosk[1:] + b_cosk[:-1]) / 2
    n_cosk_err = np.sqrt(n_cosk)/(n_cosk_o[0]/n_cosk[0])
    
    plt.errorbar(b_cosk_vals, n_cosk, xerr = (b_cosk[1]-b_cosk[0])/2, yerr=n_cosk_err, fmt='ko', capsize=0, markersize=4, label = 'Unfiltered Acceptance')
        
    if bin_num == 'all':
        title = r' Costhetak Normalised Histogram (All bins)'
    else: 
        title = f'Costhetak Normalised Histogram (bin {bin_num})'
        
    plt.title(title)
    ylabel = 'Efficency'
    plt.ylabel(ylabel)
    plt.xlabel(r'costhetak')
    plt.grid('both')
    plt.legend()
    plt.show()   
    
    return c_cosk

def acceptance_phi(acc_filt_noq2, c, r):
    ''' Assuming r=[0.0, 19.0] '''
    cosl_order = 5
    cosk_order = 6
    phi_order = 7
    q2_order = 6
    
    p = np.zeros((7, 7))
    np.fill_diagonal(p, 1.0)
    
    c_phi = np.zeros((phi_order))
    for m in range(phi_order):
        c_m = 0
        for i in range(cosl_order):
            for j in range(cosk_order):
                for n in range(q2_order):
                    L_i = lg.legval(1, lg.legint(p[i], lbnd=-1, k=0))
                    L_j = lg.legval(1, lg.legint(p[j], lbnd=-1, k=0))
                    L_n = lg.legval(1, lg.legint(p[n], lbnd=-1, k=0))
                    prod = L_i * L_j * L_n * c[i][j][m][n]
                    c_m += prod
        c_phi[m] = c_m
    
    # Acceptance is normalised in the rescaled range -1 to 1
    phi_vals = np.linspace(-np.pi, np.pi, 100)
    phi_vals_shift = phi_vals/np.pi
    leg_MoM_phi = lg.legval(phi_vals_shift, c_phi)
    plt.plot(phi_vals, leg_MoM_phi, 'r--', label='MoM Legendre')
    
    phi_data =  acc_filt_noq2[acc_filt_noq2.q2.between(r[0], r[1])]['phi'].to_numpy()

    phi_data_shift = phi_data/np.pi
    n_phi, b_phi = np.histogram(phi_data_shift, 50, range=[-1, 1])
    n_phi_w, b_phi_w = np.histogram(phi_data_shift, 50, range=[-1, 1], density=True)
    b_phi_vals_shift = (b_phi[1:] + b_phi[:-1]) / 2
    b_phi_vals = np.pi * b_phi_vals_shift
    n_q2_w_err = np.sqrt(n_phi)/(n_phi[0]/n_phi_w[0])
    plt.errorbar(b_phi_vals, n_phi_w, xerr = (b_phi_vals[1]-b_phi_vals[0])/2, yerr=n_q2_w_err, 
                 fmt='ko', capsize=0, markersize=4, label = 'Unfiltered Acceptance')
        

    if bin_num == 'all':
        title = r' Phi Histogram (All bins; Normalised in the rescaled range -1 to 1)'
    else: 
        title = f' Phi Histogram (bin {bin_num}; Normalised in the rescaled range -1 to 1)'
        
    plt.title(title)
    ylabel = 'Efficency'
    plt.ylabel(ylabel)
    plt.xlabel(r'phi')
    plt.grid('both')
    plt.legend()
    plt.show()   
    
    return c_phi
#%%
# Calculate c for all bins
c = acceptance_c(acc_filt_noq2, r=[0.0, 19])
np.save('Datasets/c_ijmn_CF.npy', c)
#%%
# Plot acceptance
r = [0.0, 19.0]
bin_num = 'all'
c = np.load('Datasets/c_ijmn_CF.npy')
c_q2 = acceptance_q2(acc_filt_noq2, c, r)
# plt.savefig('Figures/leg_MoM_q2_all_bin.png', doi = 640)
c_cosl = acceptance_cosl(acc_filt_noq2, c, r)
# plt.savefig('Figures/leg_MoM_cosl_all_bin.png', doi = 640)
c_cosk = acceptance_cosk(acc_filt_noq2, c, r)
# plt.savefig('Figures/leg_MoM_cosk_all_bin.png', doi = 640)
c_phi = acceptance_phi(acc_filt_noq2, c, r)
# plt.savefig('Figures/leg_MoM_phi_all_bin.png', doi = 640)

# %%
#q2_bins = [[0.1, 0.98], [1.1, 2.5], [2.5, 4.0], [4.0, 6.0], [6.0, 8.0], [15.0, 17.0],[17.0, 19.0], [11.0, 12.5], [1.0, 6.0], [15.0, 17.9]]

#%%
# Calculate c for each bin
q2_bins = [[0.1, 0.98], [1.1, 2.5], [2.5, 4.0], [4.0, 6.0], [6.0, 8.0], [15.0, 17.0],[17.0, 19.0], [11.0, 12.5], [1.0, 6.0], [15.0, 17.9]]

for bin_num in range(len(q2_bins)): 
    r = q2_bins[bin_num]
    c = acceptance_c(acc_filt_noq2, r)
    np.save(f'Datasets/c_ijmn_CF_{r[0]}_{r[1]}.npy', c)
    
#%%
# Plot acceptance
for bin_num in range(len(q2_bins)): 
    r = q2_bins[bin_num]
    c = np.load(f'Datasets/c_ijmn_CF_{r[0]}_{r[1]}.npy')
    c_q2 = acceptance_q2(acc_filt_noq2, c, r)
    c_cosl = acceptance_cosl(acc_filt_noq2, c, r)
    c_cosk = acceptance_cosk(acc_filt_noq2, c, r)
    c_phi = acceptance_phi(acc_filt_noq2, c, r)
    
    
# %%
