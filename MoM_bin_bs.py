# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 20:25:54 2022

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
# Import the file of the acceptance that has undergone all filters EXCEPT q2,
# acceptance that has undergone q2 cuts will be render anomalous results.
# Replace file path with whatever it is in you pc.
acc_filt_noq2 = pd.read_pickle('accept_arrays/acceptance_mc_binary3_no_cuts.pkl')
acc_filt_noq2.name = 'Acceptance no q2'

#%%
def acceptance_c(acc_filt_noq2, r=[0.1, 19], path = 'accept_arrays/c_ijmn_CF.npy', 
                 cosl_order = 7,
                 cosk_order = 6,
                 phi_order = 7,
                 q2_order = 6):
    """
    Calculates the coefficentes c_ijmn of the acceptance function

    Parameters
    ----------
    acc_filt_noq2 : pandas dataframe
        The acceptance file with all cuts except q2.
    r : list, optional
        The complete range of all q2 over which acceptance is calculated.
        The default is [0.1, 19]. This shouldn't need adjustement.
    path : string, optional
        The file path in which the array of c_ijmn will be saved.
        Save it as a .npy file. This is so this function can be run once, and then
        commented out, as this way it avoids accidentally rerunning it.
        The default is 'accept_arrays/c_ijmn_CF.npy'.
    cosl_order : int, optional
        Order of Legendre polynomials in costhetal. This includes zeroth order.
        The default is 7 (6th order). Max. of 7
    cosk_order : TYPE, optional
        Order of Legendre polynomials in costhetak. This includes zeroth order.
        The default is 6 (5th order). Max. of 7
    phi_order : TYPE, optional
        Order of Legendre polynomials in phi. This includes zeroth order.
        The default is 7 (6th order). Max. of 7
    q2_order : TYPE, optional
        Order of Legendre polynomials in q2. This includes zeroth order.
        The default is 6 (5th order). Max. of 7

    Returns
    -------
    None. Again, this function is made to be run once and then it shouldn't be called
    again. Whetever array of c_ijmn it generates is saved as a .npy file to then
    be imported again.

    """
    q2 = (acc_filt_noq2[acc_filt_noq2.q2.between(r[0], r[1])]['q2'].to_numpy() - r[0]) * 2 / (r[1]-r[0]) - 1
    cosl = acc_filt_noq2[acc_filt_noq2.q2.between(r[0], r[1])]['costhetal'].to_numpy()
    cosk = acc_filt_noq2[acc_filt_noq2.q2.between(r[0], r[1])]['costhetak'].to_numpy()
    phi = acc_filt_noq2[acc_filt_noq2.q2.between(r[0], r[1])]['phi'].to_numpy() / np.pi
    N = len(q2)
    # Order has +1 added to account for 0th order

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
                    # Print is unneccesary but serves to show progress
                    print(c_ijmn, i, j, m, n)
    
    np.save(path, c)

# # Set a path and file name and run acceptance_c by uncommenting the line below:
path = 'accept_arrays/c_ijmn_NN_6or.npy'
acceptance_c(acc_filt_noq2, [0.1, 19.0], path)

# # Then comment it again to avoid running it and just load it as the variable c
# # to use in the other functions
c = np.load(path)
#%%
def acceptance_ctl_bin(acc_filt_noq2, c, bin_n, r=[0.1, 19.0]):
    """
    Calculates and plots the acceptance in costhetal for an individual q2 bin

    Parameters
    ----------
    acc_filt_noq2 : pandas dataframe
        The acceptance file with all cuts except q2.
    c : array
        Array of c_ijmn coefficients.
    bin_n : int
        Bin number from the binning scheme.
    r : list, optional
        The complete range of all q2 over which c_ijmn were calculated.
        The default is [0.1, 19.0]. Should not need adjustement unless this 
        was adjuested for acceptance_c, in which case it should match that.

    Returns
    -------
    c_ctl_norm : array
        Coefficients for each term in the legendre series. The acceptance function
        at arbitrary x = costhetal in the bin is then:
            lg.legval(x, c_ctl_norm)
        provided:
            from numpy.polynomial import legendre as lg
    """
    
    q2_bins = [[0.1, 0.98], [1.1, 2.5], [2.5, 4.0], [4.0, 6.0], [6.0, 8.0], [15.0, 17.0],
           [17.0, 19.0], [11.0, 12.5], [1.0, 6.0], [15.0, 17.9]]
    r_bin = np.array(q2_bins[bin_n])

    p = np.zeros((7, 7))
    np.fill_diagonal(p, 1.0)
    
    cosl_order = np.shape(c)[0]
    cosk_order = np.shape(c)[1]
    phi_order = np.shape(c)[2]
    q2_order = np.shape(c)[3]
    
    r_bin_shift = (r_bin - r[0]) * 2 / (r[1]-r[0]) - 1
    
    c_ctl = np.zeros((cosl_order))
    for i in range(cosl_order):
        c_i = 0
        for j in range(cosk_order):
            for m in range(phi_order):
                for n in range(q2_order):
                    L_j = lg.legval(1, lg.legint(p[j], lbnd=-1, k=0))
                    L_m = lg.legval(1, lg.legint(p[m], lbnd=-1, k=0))
                    L_n = lg.legval(r_bin_shift[1], lg.legint(p[n], lbnd=r_bin_shift[0], k=0))
                    prod = L_j * L_m * L_n * c[i][j][m][n]
                    c_i += prod
        c_ctl[i] = c_i
    
    plt.clf()
    
    ctl_data = acc_filt_noq2[acc_filt_noq2.q2.between(r_bin[0], r_bin[1])]['costhetal'].to_numpy()

    n, b = np.histogram(ctl_data, 50, range=[-1, 1])
    n_w, b_w = np.histogram(ctl_data, 50, range=[-1, 1], density=True)
    b_c = (b[1:] + b[:-1]) / 2
    n_w_err = np.sqrt(n)*(n_w[5]/n[5])
    
    plt.errorbar(b_c, n_w, xerr = (b[1]-b[0])/2, yerr=n_w_err, 
                 fmt='ko', capsize=0, markersize=4)
    
    
    norm = lg.legval(1, lg.legint(c_ctl, lbnd=-1, k=0))
    leg_MoM_ctl = lg.legval(b_c, c_ctl)/norm
    plt.plot(b_c, leg_MoM_ctl, 'r--', label='MoM Legendre')
    
    # Calculates and prints chi2 values for the fit
    chi2 = sum((np.array(n_w) - np.array(leg_MoM_ctl))**2 / (np.array(n_w_err)**2))
    print('Chi2: ' + str(chi2))
    print('Reduced Chi2: ' + str(chi2/len(n)) + ' (assuming no constraints)')
    
    # Checks that the function will be normalised
    c_ctl_norm = c_ctl/norm
    print(lg.legval(1, lg.legint(c_ctl_norm, lbnd=-1, k=0)))
    
    title = r'$\cos\theta_l$, $q^2\in$ %s Normalised Histogram' % str(q2_bins[bin_n])
    plt.title(title)
    ylabel = 'Normalised Acceptance'
    plt.ylabel(ylabel)
    plt.xlabel(r'$\cos\theta_l$')
    plt.grid('both')
    plt.show()
    
    return c_ctl_norm

def acceptance_ctk_bin(acc_filt_noq2, c, bin_n, r=[0.1, 19.0]):
    """
    Calculates and plots the acceptance in costhetak for an individual q2 bin

    Parameters
    ----------
    acc_filt_noq2 : pandas dataframe
        The acceptance file with all cuts except q2.
    c : array
        Array of c_ijmn coefficients.
    bin_n : int
        Bin number from the binning scheme.
    r : list, optional
        The complete range of all q2 over which c_ijmn were calculated.
        The default is [0.1, 19.0]. Should not need adjustement unless this 
        was adjuested for acceptance_c, in which case it should match that.

    Returns
    -------
    c_ctk_norm : array
        Coefficients for each term in the legendre series. The acceptance function
        at arbitrary x = costhetak in the bin is then:
            lg.legval(x, c_ctk_norm)
        provided:
            from numpy.polynomial import legendre as lg
    """
    # Calculates and plots acceptance for an individual q2 bin
    q2_bins = [[0.1, 0.98], [1.1, 2.5], [2.5, 4.0], [4.0, 6.0], [6.0, 8.0], [15.0, 17.0],
           [17.0, 19.0], [11.0, 12.5], [1.0, 6.0], [15.0, 17.9]]
    r_bin = np.array(q2_bins[bin_n])

    p = np.zeros((7, 7))
    np.fill_diagonal(p, 1.0)
    
    cosl_order = np.shape(c)[0]
    cosk_order = np.shape(c)[1]
    phi_order = np.shape(c)[2]
    q2_order = np.shape(c)[3]
    
    r_bin_shift = (r_bin - r[0]) * 2 / (r[1]-r[0]) - 1
    
    c_ctk = np.zeros((cosk_order))
    for j in range(cosk_order):
        c_j = 0
        for i in range(cosl_order):
            for m in range(phi_order):
                for n in range(q2_order):
                    L_i = lg.legval(1, lg.legint(p[i], lbnd=-1, k=0))
                    L_m = lg.legval(1, lg.legint(p[m], lbnd=-1, k=0))
                    L_n = lg.legval(r_bin_shift[1], lg.legint(p[n], lbnd=r_bin_shift[0], k=0))
                    prod = L_i * L_m * L_n * c[i][j][m][n]
                    c_j += prod
        c_ctk[j] = c_j
    
    plt.clf()
    
    ctk_data = acc_filt_noq2[acc_filt_noq2.q2.between(r_bin[0], r_bin[1])]['costhetak'].to_numpy()

    n, b = np.histogram(ctk_data, 50, range=[-1, 1])
    n_w, b_w = np.histogram(ctk_data, 50, range=[-1, 1], density=True)
    b_c = (b[1:] + b[:-1]) / 2
    n_w_err = np.sqrt(n)/(n[5]/n_w[5])
    
    plt.errorbar(b_c, n_w, xerr = (b[1]-b[0])/2, yerr=n_w_err, 
                 fmt='ko', capsize=0, markersize=4)
    
    
    norm = lg.legval(1, lg.legint(c_ctk, lbnd=-1, k=0))
    leg_MoM_ctk = lg.legval(b_c, c_ctk)/norm
    # print(norm)
    plt.plot(b_c, leg_MoM_ctk, 'r--', label='MoM Legendre')
    
    chi2 = sum((np.array(n_w) - np.array(leg_MoM_ctk))**2 / (np.array(n_w_err)**2))
    print('Chi2: ' + str(chi2))
    print('Reduced Chi2: ' + str(chi2/len(n)) + ' (assuming no constraints)')
    
    c_ctk_norm = c_ctk/norm
    print(lg.legval(1, lg.legint(c_ctk_norm, lbnd=-1, k=0)))
    
    title = r'$\cos\theta_K$, $q^2\in$ %s Normalised Histogram' % str(q2_bins[bin_n])
    plt.title(title)
    ylabel = 'Normalised Acceptance'
    plt.ylabel(ylabel)
    plt.xlabel(r'$\cos\theta_K$')
    plt.grid('both')
    plt.show()
    
    return c_ctk_norm

def acceptance_phi_bin(acc_filt_noq2, c, bin_n, r=[0.1, 19.0]):
    """
    Calculates and plots the acceptance in phi for an individual q2 bin
    IMPORTANT: The acceptance is calculated with phi rescaled
    to the range -1 to 1 instead of -pi to pi. Therefore, when inputing values
    to the resulting acceptance function, it has to be phi/np.pi instead of phi
    directly.

    Parameters
    ----------
    acc_filt_noq2 : pandas dataframe
        The acceptance file with all cuts except q2.
    c : array
        Array of c_ijmn coefficients.
    bin_n : int
        Bin number from the binning scheme.
    r : list, optional
        The complete range of all q2 over which c_ijmn were calculated.
        The default is [0.1, 19.0]. Should not need adjustement unless this 
        was adjuested for acceptance_c, in which case it should match that.

    Returns
    -------
    c_phi_norm : array
        Coefficients for each term in the legendre series. The acceptance function
        at arbitrary x = phi in [-pi, pi] is then:
            lg.legval(x/np.pi, c_phi_norm)
        provided:
            from numpy.polynomial import legendre as lg
    """
    # Calculates and plots acceptance for an individual q2 bin
    q2_bins = [[0.1, 0.98], [1.1, 2.5], [2.5, 4.0], [4.0, 6.0], [6.0, 8.0], [15.0, 17.0],
           [17.0, 19.0], [11.0, 12.5], [1.0, 6.0], [15.0, 17.9]]
    r_bin = np.array(q2_bins[bin_n])

    p = np.zeros((7, 7))
    np.fill_diagonal(p, 1.0)
    
    cosl_order = np.shape(c)[0]
    cosk_order = np.shape(c)[1]
    phi_order = np.shape(c)[2]
    q2_order = np.shape(c)[3]
    
    r_bin_shift = (r_bin - r[0]) * 2 / (r[1]-r[0]) - 1
    
    c_phi = np.zeros((phi_order))
    for m in range(phi_order):
        c_m = 0
        for i in range(cosl_order):
            for j in range(cosk_order):
                for n in range(q2_order):
                    L_i = lg.legval(1, lg.legint(p[i], lbnd=-1, k=0))
                    L_j = lg.legval(1, lg.legint(p[j], lbnd=-1, k=0))
                    L_n = lg.legval(r_bin_shift[1], lg.legint(p[n], lbnd=r_bin_shift[0], k=0))
                    prod = L_i * L_j * L_n * c[i][j][m][n]
                    c_m += prod
        c_phi[m] = c_m
    
    plt.clf()
    
    phi_data = acc_filt_noq2[acc_filt_noq2.q2.between(r_bin[0], r_bin[1])]['phi'].to_numpy()

    n, b = np.histogram(phi_data, 50, range=[-np.pi, np.pi])
    n_w, b_w = np.histogram(phi_data, 50, range=[-np.pi, np.pi], density=True)
    b_c = (b[1:] + b[:-1]) / 2
    n_w_err = np.sqrt(n)/(n[5]/n_w[5])
    
    plt.errorbar(b_c, n_w, xerr = (b_c[1]-b_c[0])/2, yerr=n_w_err, 
                 fmt='ko', capsize=0, markersize=4)
    
    norm = lg.legval(1, lg.legint(c_phi, lbnd=-1, k=0))
    leg_MoM_phi = lg.legval(b_c/np.pi, c_phi)/ (norm*np.pi)
    plt.plot(b_c, leg_MoM_phi, 'r--', label='MoM Legendre')
    
    chi2 = sum((np.array(n_w) - np.array(leg_MoM_phi))**2 / (np.array(n_w_err)**2))
    print('Chi2: ' + str(chi2))
    print('Reduced Chi2: ' + str(chi2/len(n)) + ' (assuming no constraints)')
    
    c_phi_norm = c_phi/norm
    print(lg.legval(1, lg.legint(c_phi_norm, lbnd=-1, k=0)))
    
    title = r'$\phi$, $q^2\in$ %s Normalised Histogram' % str(q2_bins[bin_n])
    plt.title(title)
    ylabel = 'Normalised Acceptance'
    plt.ylabel(ylabel)
    plt.xlabel(r'$\phi$')
    plt.grid('both')
    plt.show()
    
    return c_phi_norm/np.pi

