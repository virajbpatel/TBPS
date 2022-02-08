# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 11:06:55 2022

@author: Dariusz

Description: package containing a couple of manual data cutting filters
"""
#%%
#import relevant packages
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%

def IP_check(df,B0_Impact_parameter_chi2_max = 16,
             B0_flight_disyance_chi2_min = 121,
             cos_DIRA_min = 0.9999,
             B0_endvertex_chi2_ndf_max = 8,
             J_psi_emdvertex_chi2_max = 9,
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
    df5 = df4[df4["J_psi_ENDVERTEX_CHI2"]/df4["J_psi_ENDVERTEX_NDOF"] < J_psi_emdvertex_chi2_max]
    df6 = df5[df5["Kstar_ENDVERTEX_CHI2"]/df5["Kstar_ENDVERTEX_NDOF"] < Kstar_endvertex_chi2_max]
    df7 = df6[df6["Kstar_FDCHI2_OWNPV"] > Kstar_flight_distance_chi2_min]

    df8 = df7[df7["mu_plus_IPCHI2_OWNPV"] > mu_plus_impact_parameter_chi2_min]
    df9 = df8[df8["mu_minus_IPCHI2_OWNPV"] > mu_minus_impact_parameter_chi2_min]
    df10 = df9[df9["K_IPCHI2_OWNPV"] > K_impact_parameter_chi2_min]
    df11 = df10[df10["Pi_IPCHI2_OWNPV"] > pi_impact_parameter_chi2_min]
    
    print("percentage dataframe removed =",(len(df)-len(df11))/len(df))
    
    return(df11)
    
def mom_check(df,
              K_PT_max = 1600,
              Pi_PT_max = 1600):
    
    df1 = df[df["K_PT"] > K_PT_max]
    df2 = df1[df1["Pi_PT"] > Pi_PT_max]
    
    print("percentage dataframe removed =",(len(df)-len(df2))/len(df))

    return(df2)

def muon_PT_check(df,
                  min_mu_PT=1760):
    
    drops = []
    for i in range(len(df['mu_plus_PT'])):
        if df['mu_plus_PT'][i] < min_mu_PT and df['mu_minus_PT'][i] < min_mu_PT:
            drops += [i]
        else:
            continue
    df1 = df.drop(labels=drops, axis=0)
    
    print("percentage dataframe removed =",(len(df)-len(df1))/len(df))
    
    return (df1)

def invariant_mass_check(df,                #maybe add k star and j psi invarient
                         min_B0_MM=5180,    #mass checks later if they don't destroy
                         max_B0_MM=5380):   #the signal too much
    
    df1 = df[df["B0_MM"] > min_B0_MM]
    df2 = df1[df1["B0_MM"] < max_B0_MM]
    
    print("percentage dataframe removed =",(len(df)-len(df2))/len(df))
    
    return(df2)
    
def particle_ID_check(df,                           #add cross probs if need be
                      mu_plus_mu_min_prob = 0.6,
                      mu_minus_mu_min_prob = 0.6,
                      K_K_min_prob = 0.6,
                      pi_pi_min_prob = 0.6):
    
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
    
    
    
    
    
    
    
