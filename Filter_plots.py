# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:34:37 2022
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
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
# This section just imports the pkl files
# You will need to adjust this to wherever the data is locally in your computer
# Also, this has never been an issue for me but I've been told that pkl imports
# Can be an issue in some pytho versions

total = pd.read_pickle('data/total_dataset.pkl')
accept = pd.read_pickle('data/acceptance_mc.pkl')
# The signal decay, simulated as per the Standard Model
sig = pd.read_pickle('data/signal.pkl')
#Signal decay but with kaon reconstructed as pion and pion as kaon
k_pi_swap = pd.read_pickle('data/k_pi_swap.pkl')
# B_0 to J/psi
jpsi = pd.read_pickle('data/jpsi.pkl')
# J/psi with muon reconstructed as kaon and kaon reconstructed as muon
jpsi_mu_k_swap = pd.read_pickle('data/jpsi_mu_k_swap.pkl')
# J/psi with muon reconstructed as pion and pion reconstructed as muon
jpsi_mu_pi_swap = pd.read_pickle('data/jpsi_mu_pi_swap.pkl')
# B_0 to psi2S
psi2S = pd.read_pickle('data/psi2S.pkl')

# B_0 to (phi to KK)\mu\mu 
phimumu = pd.read_pickle('data/phimumu.pkl')
# Lam decays
pKmumu_piTok_kTop = pd.read_pickle('data/pKmumu_piTok_kTop.pkl')
pKmumu_piTop = pd.read_pickle('data/pKmumu_piTop.pkl')

#%%
# This names each dataframe so they are printed correctly in the plot labels
# The plotting functions below will break if the dataframes are not named
total.name = 'total_dataset'
accept.name = 'Acceptance'
sig.name = 'Simulated SM Signal'
k_pi_swap.name = 'Signal w/ $K\pi\leftrightarrow \pi K$'
jpsi.name = r'$B^0\rightarrow (J/\psi\rightarrow \mu\mu)K^{*0}$'
jpsi_mu_k_swap.name = r'$J/\psi$ w/ $\mu K\leftrightarrow K\mu$'
jpsi_mu_pi_swap.name = r'$J/\psi$ w/ $\mu \pi\leftrightarrow \pi\mu$'

psi2S.name = r'$B^0\rightarrow (\psi (2S)\rightarrow\mu\mu)K^{*0}$'
phimumu.name = r'$B^0 _s\rightarrow (\phi\rightarrow KK)\mu\mu$, w/ $KK\leftrightarrow K\pi$'
pKmumu_piTok_kTop.name = r'$\Lambda^0 _b \rightarrow pK\mu\mu$ w/ $pK\leftrightarrow K\pi$'
pKmumu_piTop.name = r'$\Lambda^0 _b \rightarrow pK\mu\mu$ w/ $p\leftrightarrow \pi$'
#%%
def IP_check(df, B0_Impact_parameter_chi2_max = 16,
             B0_flight_disyance_chi2_min = 64,
             cos_DIRA_min = 0.9999,
             B0_endvertex_chi2_ndf_max = 8,
             J_psi_endvertex_chi2_max = 9,
             Kstar_endvertex_chi2_max = 9,
             Kstar_flight_distance_chi2_min = 9,
             mu_plus_impact_parameter_chi2_min = 9,
             mu_minus_impact_parameter_chi2_min = 9,
             K_impact_parameter_chi2_min = 9,
             pi_impact_parameter_chi2_min = 9):

    df1 = df[df["B0_IPCHI2_OWNPV"] < B0_Impact_parameter_chi2_max]
    df2 = df1[df1["B0_FDCHI2_OWNPV"] > B0_flight_disyance_chi2_min]
    df3 = df2[df2["B0_DIRA_OWNPV"] > cos_DIRA_min]
    df4 = df3[df3["B0_ENDVERTEX_CHI2"]/df3["B0_ENDVERTEX_NDOF"] < B0_endvertex_chi2_ndf_max]
    df5 = df4[df4["J_psi_ENDVERTEX_CHI2"]/df4["J_psi_ENDVERTEX_NDOF"] < J_psi_endvertex_chi2_max]
    df6 = df5[df5["Kstar_ENDVERTEX_CHI2"]/df5["Kstar_ENDVERTEX_NDOF"] < Kstar_endvertex_chi2_max]
    df7 = df6[df6["Kstar_FDCHI2_OWNPV"] > Kstar_flight_distance_chi2_min]

    df8 = df7[df7["mu_plus_IPCHI2_OWNPV"] > mu_plus_impact_parameter_chi2_min]
    df9 = df8[df8["mu_minus_IPCHI2_OWNPV"] > mu_minus_impact_parameter_chi2_min]
    df10 = df9[df9["K_IPCHI2_OWNPV"] > K_impact_parameter_chi2_min]
    df11 = df10[df10["Pi_IPCHI2_OWNPV"] > pi_impact_parameter_chi2_min]
    
    print("percentage dataframe removed =",(len(df)-len(df11))/len(df))
    
    return(df11)
    
def mom_check(df, 
              K_PT_min = 250,
              Pi_PT_min = 250):
    
    df1 = df[df["K_PT"] > K_PT_min]
    df2 = df1[df1["Pi_PT"] > Pi_PT_min]
    
    print("percentage dataframe removed =",(len(df)-len(df2))/len(df))

    return(df2)

def muon_PT_check(df,
                  min_mu_PT=800):
    
    df1 = df[df['mu_plus_PT'] > min_mu_PT]
    df2 = df1[df1['mu_minus_PT'] > min_mu_PT]
    
    print("percentage dataframe removed =",(len(df)-len(df2))/len(df))
    return (df2)

def invariant_mass_check(df,                #maybe add k star and j psi invarient
                         min_B0_MM=5170,    #mass checks later if they don't destroy
                         max_B0_MM=5700,
                         min_Kstar_MM = 790,
                         max_Kstar_MM = 1000):   #the signal too much
    
    df1 = df[df["B0_MM"] > min_B0_MM]
    df2 = df1[df1["B0_MM"] < max_B0_MM]
    
    df3 = df2[df2['Kstar_MM'] > min_Kstar_MM]
    df4 = df3[df3['Kstar_MM'] < max_Kstar_MM]
    
    print("percentage dataframe removed =",(len(df)-len(df2))/len(df))
    
    return(df4)
    
def particle_ID_check(df,                           #add cross probs if need be
                      mu_plus_mu_min_prob = 0.5,
                      mu_minus_mu_min_prob = 0.5,
                      K_K_min_prob = 0.5,
                      pi_pi_min_prob = 0.5):
    
    df1 = df[df["mu_plus_MC15TuneV1_ProbNNmu"] > mu_plus_mu_min_prob]
    df2 = df1[df1["mu_minus_MC15TuneV1_ProbNNmu"] > mu_minus_mu_min_prob]
    df3 = df2[df2["K_MC15TuneV1_ProbNNk"] > K_K_min_prob]
    df4 = df3[df3["Pi_MC15TuneV1_ProbNNpi"] > pi_pi_min_prob]
    
    print("percentage dataframe removed =",(len(df)-len(df4))/len(df))
    
    return(df4)

def q2_cuts(df):
    
    df1 = df[~df.q2.between(8.0, 11.0)]
    df2 = df1[~df1.q2.between(12.5, 15.0)]
    df3 = df2[df2['q2'] < 19.0]
    df4 = df3[df3['q2'] > 0.1]
    
    print("percentage dataframe removed =",(len(df)-len(df4))/len(df))
    
    return(df4)
#%%
# This compliles all the checks into a single function, it is made to 
# print the percentage data removed at each step, mainly to monitor
# how much data is cut
def check_all(df):
    # For some reason the muon check gets an error if it is not run first
    df0 = q2_cuts(df)
    print('q2 check:')
    per_rem0 = (len(df)-len(df0))/len(df)
    print("percentage total dataframe removed =",(len(df)-len(df0))/len(df))
    print(len(df0))
    
    df1 = muon_PT_check(df0)
    print('Muon PT check:')
    per_rem1 = (len(df)-len(df1))/len(df)
    diff_rem1 = per_rem1 - per_rem0
    print("percentage total dataframe removed =",(len(df)-len(df1))/len(df))
    print(diff_rem1)
    print(len(df1))
    
    df2 = invariant_mass_check(df1)
    print('Mass check:')
    per_rem2 = (len(df)-len(df2))/len(df)
    diff_rem2 = per_rem2 - per_rem1
    print("percentage total dataframe removed =",(len(df)-len(df2))/len(df))
    print(diff_rem2)
    print(len(df2))
    
    df3 = particle_ID_check(df2)
    print('PID check:')
    per_rem3 = (len(df)-len(df3))/len(df)
    diff_rem3 = per_rem3 - per_rem2
    print("percentage total dataframe removed =",(len(df)-len(df3))/len(df))
    print(diff_rem3)
    print(len(df3))
    
    df4 = IP_check(df3)
    print('IP, vertex check:')
    per_rem4 = (len(df)-len(df4))/len(df)
    diff_rem4 = per_rem4 - per_rem3
    print("percentage total dataframe removed =",(len(df)-len(df4))/len(df))
    print(diff_rem4)
    print(len(df4))
    
    df5 = mom_check(df4)
    print('Momentum check:')
    per_rem5 = (len(df)-len(df5))/len(df)
    diff_rem5 = per_rem5 - per_rem4
    print("percentage total dataframe removed =",(len(df)-len(df5))/len(df))
    print(diff_rem5)
    print(len(df5))
    print('_______________________________')
    return df5
#%%
tot_filt = check_all(total)
# sig_filt = check_all(sig)
# acc_filt = check_all(accept)

tot_filt.name = 'Total filtered'
# sig_filt.name = 'SM filtered'
# acc_filt.name = 'Acceptance Filtered'
#%%
def histogram(data_frames, label, bin_range, bin_N):
    bins = np.linspace(bin_range[0], bin_range[1], bin_N)
    plt.clf()
    for i in range(len(data_frames)):
        x = data_frames[i][str(label)].to_numpy()
        plt.hist(x, bins, label=data_frames[i].name, histtype=u'step')
        occur1 = x > bin_range[1]
        occur2 = x < bin_range[0]
        left_out = occur1.sum() + occur2.sum()
        print('Data set %s: %d/%d points left out for given bins' %(data_frames[i].name, left_out,len(x)))
    plt.legend()
    title = str(label) + ' Histogram'
    plt.title(title)
    plt.grid('both')
    plt.show()

def totnormalhistogram(data_frames, label, bin_range, bin_N):
    # bins = np.linspace(bin_range[0], bin_range[1], bin_N)
    plt.clf()
    for i in range(len(data_frames)):
        x = data_frames[i][str(label)].to_numpy()
        N, b = np.histogram(x, bin_N, range=bin_range)
        N_w = N/len(x)
        err = np.sqrt(N)/len(x)
        b_vals = (b[1:] + b[:-1]) / 2
        plt.errorbar(b_vals, N_w, yerr=err, fmt='o', capsize=3, markersize=2, label=data_frames[i].name)
        width = b[1]-b[0] 
        # plt.bar(b_vals, N_w, width=width, alpha=0.2, linewidth=0)
        occur1 = x > bin_range[1]
        occur2 = x < bin_range[0]
        left_out = occur1.sum() + occur2.sum()
        print('Data set %s: %d/%d points left out for given bins, %f' %(data_frames[i].name, left_out,len(x), left_out/len(x)))
    plt.legend()
    title = str(label) + ' Normalised Histogram'
    plt.title(title)
    plt.grid('both')
    plt.show()

def normalhistogram(data_frames, label, bin_range, bin_N):
    bins = np.linspace(bin_range[0], bin_range[1], bin_N)
    plt.clf()
    for i in range(len(data_frames)):
        x = data_frames[i][str(label)].to_numpy()
        plt.hist(x, bins, label=data_frames[i].name, density=True, histtype=u'step')
        occur1 = x > bin_range[1]
        occur2 = x < bin_range[0]
        left_out = occur1.sum() + occur2.sum()
        print('Data set %s: %d/%d points left out for given bins, %f' %(data_frames[i].name, left_out,len(x), left_out/len(x)))
    plt.legend()
    title = 'Normalised ' + str(label) + ' Histogram'
    plt.title(title)
    plt.grid('both')
    plt.show()

# Plot whatever column for the list of dataframes and bin range you want
# The code should print out how many values are outside bin range so it can be 
# adjusted accordingly
datasets = [total, sig, tot_filt]
# datasets = [total, sig, jpsi, psi2S, phimumu, jpsi_mu_k_swap, jpsi_mu_pi_swap, 
#             k_pi_swap, pKmumu_piTok_kTop, pKmumu_piTop]
bin_range = [-1, 1]
bin_number = 50
value = 'costhetal'
# histogram(datasets, value, bin_range, bin_number)
# normalhistogram(datasets, value, bin_range, bin_number)
# totnormalhistogram(datasets, value, bin_range, bin_number)
#%%
histogram([total], 'B0_MM', [5100, 5700], 50)
normalhistogram([total, sig, tot_filt], 'B0_MM', [5100, 5700], 50)
totnormalhistogram([total, sig, tot_filt], 'B0_MM', [5100, 5700], 50)
#%%
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
        N_w = N/len(x)
        err = np.sqrt(N)/len(x)
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

fittotnormalhistogram([tot_filt], 'B0_MM', [5170, 5700], 50)
#%%
# This plots a 2d histogram of the distribution of events at different B0 mass and 
# q^2 values
def qsquared_inv_m(data, den=False):
    b0_m = data['B0_MM'].to_numpy()
    q2 = data['q2'].to_numpy()
    x_edges = np.linspace(5100, 5600, 50)
    y_edges = np.linspace(0, 19, 50)
    
    plt.clf()
    plt.hist2d(b0_m, q2, bins=[x_edges, y_edges], density = den)
    plt.title(data.name)
    plt.xlabel(r'Reconstructed $B^0$ mass (MeV/c$^2$)')
    plt.ylabel(r'$q^2$ (GeV$^2$/c$^4$)')
    plt.colorbar()
    plt.show()
    
qsquared_inv_m(tot_filt, False)
