# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:09:21 2022

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
# acc_filt = pd.read_csv('datasets/acceptance_filtered.csv')
# acc_filt.name = 'Accept Filtered'

acc_filt_noq2 = pd.read_pickle('datasets/acc_filt_noq2.pkl')
acc_filt_noq2.name = 'Acceptance 2'
#%%
def MoM_coef(data_frame):
    cosl = data_frame['costhetal'].to_numpy()
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
    print(chi2_uncert) # This is not reduced
    # chi2_pearson = sum((N_w - leg_MoM)**2/(leg_MoM))
    # print(chi2_pearson)
    # chi2_neyman = sum((N_w - leg_MoM)**2/(N_w))
    # print(chi2_neyman)
    # # Combined chi2_pearson and neyman
    # print(1/3 * (chi2_neyman + 2*chi2_pearson))
    
    plt.legend()
    title = r' $\cos\theta_l$ Normalised Histogram'
    plt.title(title)
    ylabel = 'Candidate density / %.1f' % (width)
    plt.ylabel(ylabel)
    plt.xlabel(r'$\cos\theta_l$')
    plt.grid('both')
    plt.show()
    
    return alph

#%%
def MoM_coefq2(data_frame, r = [0.0, 19.0], bin_N = 30):
    q2 = data_frame[data_frame.q2.between(r[0], r[1])]['q2'].to_numpy()
    moments = [1]
    for i in range(6):
        vals = q2**(i+1)
        mu = np.average(vals)
        moments += [mu]
    moments = np.array(moments)
    
    p = np.zeros((7, 7))
    np.fill_diagonal(p, 1.0)
    
    M = np.zeros((7, 7))
    for j in range(len(p)):
        if j == 0:
            for i in range(len(p)):
                p_int = lg.legint(p[i], lbnd=r[0])
                coef = lg.legval(r[1], p_int)
                M[j][i] = coef
        else:
            for l in range(len(p)):
                p_v = p[l]
                for k in range(j):
                    p_v = lg.legmulx(p_v)
                p_int = lg.legint(p_v, lbnd=r[0])
                coef = lg.legval(r[1], p_int)
                M[j][l] = coef
                
    alph = np.linalg.solve(M, moments)
    
    print(lg.legval(r[1], lg.legint(alph, lbnd=r[0], k=0)))
    
    plt.clf()
    N, b = np.histogram(q2, bin_N, range=r)
    N_w, b_w = np.histogram(q2, bin_N, range=r, density=True)
    err = np.sqrt(N)/(N[0]/N_w[0])
    b_vals = (b[1:] + b[:-1]) / 2
    width = b[1]-b[0]
    plt.errorbar(b_vals, N_w, xerr = width/2, yerr=err, fmt='ko', capsize=0, markersize=5, label='Filtered Acc')
    
    leg_MoM = lg.legval(b_vals, alph)
    plt.plot(b_vals, leg_MoM, 'r--', label='MoM Legendre')
    
    # Chi^2 tests for the legendre MoM acceptance.
    chi2_uncert = sum((N_w - leg_MoM)**2/(err**2))
    print(chi2_uncert) # This is not reduced
    
    plt.legend()
    title = r' $q^2$ Normalised Histogram'
    plt.title(title)
    ylabel = 'Candidate density / %.1f' % (width)
    plt.ylabel(ylabel)
    plt.xlabel(r'$q^2$')
    plt.grid('both')
    plt.show()
    
    return alph

MoM_coefq2(acc_filt_noq2, [0.1, 19.0], 50)
#%%
def acceptance_c(acc_filt_noq2, r=[0.0, 19]):
    q2 = (acc_filt_noq2[acc_filt_noq2.q2.between(r[0], r[1])]['q2'].to_numpy() - 9.5) / 9.5
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
                    print(c_ijmn, i, j, m, n)
    
    np.save('accept_arrays/c_ijmn_CF.npy', c)

def acceptance_q2(acc_filt_noq2, c):
    # Assuming r=[0.0, 19.0]
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
    q_vals = np.linspace(0, 19, 100)
    q_vals_shift = (q_vals - 9.5)/9.5
    leg_MoM_q2 = lg.legval(q_vals_shift, c_q2)
    plt.plot(q_vals, leg_MoM_q2, 'r--', label='MoM Legendre')
    
    q2_data = acc_filt_noq2[acc_filt_noq2.q2.between(0.0, 19.0)]['q2'].to_numpy()
    q2_data_shift = (q2_data - 9.5)/9.5
    n_q2, b_q2 = np.histogram(q2_data_shift, 50, range=[-1, 1])
    n_q2_w, b_q2_w = np.histogram(q2_data_shift, 50, range=[-1, 1], density=True)
    b_q2_vals_shift = (b_q2[1:] + b_q2[:-1]) / 2
    b_q2_vals = 9.5 * b_q2_vals_shift + 9.5
    n_q2_w_err = np.sqrt(n_q2)/(n_q2[0]/n_q2_w[0])
    
    plt.errorbar(b_q2_vals, n_q2_w, xerr = (b_q2_vals[1]-b_q2_vals[0])/2, yerr=n_q2_w_err, 
                 fmt='ko', capsize=0, markersize=4)
        
    # plt.plot(b_q2_vals, n_q2_w, 'ko', markersize = 3)
    
    title = r' $q^2$ Normalised Histogram'
    plt.title(title)
    ylabel = 'Efficency'
    plt.ylabel(ylabel)
    plt.xlabel(r'$q^2$')
    plt.grid('both')
    plt.show()    
    
    return c_q2
#%%
c = np.load('accept_arrays/c_ijmn_CF.npy')
c_q2 = acceptance_q2(acc_filt_noq2, c)