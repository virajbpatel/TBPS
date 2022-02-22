# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 11:06:55 2022

@author: Dariusz

Description: package containing a couple of manual data cutting filters
"""
# %%
# import relevant packages
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %%


def IP_check(df, B0_Impact_parameter_chi2_max=16,  # all vals are < 16.1
             B0_flight_disyance_chi2_min=64,  # All values are greater than 62, barely cuts
             cos_DIRA_min=0.9999,
             B0_endvertex_chi2_ndf_max=8,  # No impact
             J_psi_endvertex_chi2_max=9,  # all vals are < 12
             Kstar_endvertex_chi2_max=9,  # all vals are < 12
             Kstar_flight_distance_chi2_min=9,  # No impact all vals >= 10
             mu_plus_impact_parameter_chi2_min=9,  # all vals are > 6
             mu_minus_impact_parameter_chi2_min=9,  # all vals are > 6
             K_impact_parameter_chi2_min=9,  # all vals are > 6
             pi_impact_parameter_chi2_min=9):  # all vals are > 6

    df1 = df[df["B0_IPCHI2_OWNPV"] < B0_Impact_parameter_chi2_max]
    df2 = df1[df1["B0_FDCHI2_OWNPV"] > B0_flight_disyance_chi2_min]
    df3 = df2[df2["B0_DIRA_OWNPV"] > cos_DIRA_min]
    df4 = df3[df3["B0_ENDVERTEX_CHI2"] /
              df3["B0_ENDVERTEX_NDOF"] < B0_endvertex_chi2_ndf_max]
    df5 = df4[df4["J_psi_ENDVERTEX_CHI2"] /
              df4["J_psi_ENDVERTEX_NDOF"] < J_psi_endvertex_chi2_max]
    df6 = df5[df5["Kstar_ENDVERTEX_CHI2"] /
              df5["Kstar_ENDVERTEX_NDOF"] < Kstar_endvertex_chi2_max]
    df7 = df6[df6["Kstar_FDCHI2_OWNPV"] > Kstar_flight_distance_chi2_min]

    df8 = df7[df7["mu_plus_IPCHI2_OWNPV"] > mu_plus_impact_parameter_chi2_min]
    df9 = df8[df8["mu_minus_IPCHI2_OWNPV"] >
              mu_minus_impact_parameter_chi2_min]
    df10 = df9[df9["K_IPCHI2_OWNPV"] > K_impact_parameter_chi2_min]
    df11 = df10[df10["Pi_IPCHI2_OWNPV"] > pi_impact_parameter_chi2_min]

    print("percentage dataframe removed =", (len(df)-len(df11))/len(df))

    return(df11)


def mom_check(df,
              K_PT_min=250,
              Pi_PT_min=250):

    df1 = df[df["K_PT"] > K_PT_min]
    df2 = df1[df1["Pi_PT"] > Pi_PT_min]

    print("percentage dataframe removed =", (len(df)-len(df2))/len(df))

    return(df2)


def muon_PT_check(df,
                  min_mu_PT=800):  # All mu_PT > 1000, so this has no effect

    df1 = df[df['mu_plus_PT'] > min_mu_PT]
    df2 = df1[df1['mu_minus_PT'] > min_mu_PT]

    print("percentage dataframe removed =", (len(df)-len(df2))/len(df))
    return (df2)


def invariant_mass_check(df,
                         min_B0_MM=5170,  # Very few below this
                         max_B0_MM=5700,  # All values are below his threshold
                         min_Kstar_MM=790,  # Kstar check a lot more stringent - could widen
                         max_Kstar_MM=1000):

    df1 = df[df["B0_MM"] > min_B0_MM]
    df2 = df1[df1["B0_MM"] < max_B0_MM]

    df3 = df2[df2['Kstar_MM'] > min_Kstar_MM]
    df4 = df3[df3['Kstar_MM'] < max_Kstar_MM]

    print("percentage dataframe removed =", (len(df)-len(df2))/len(df))

    return(df4)


def particle_ID_check(df,  # add cross probs if need be
                      mu_plus_mu_min_prob=0.5,
                      mu_minus_mu_min_prob=0.5,
                      K_K_min_prob=0.5,
                      pi_pi_min_prob=0.5):

    df1 = df[df["mu_plus_MC15TuneV1_ProbNNmu"] > mu_plus_mu_min_prob]
    df2 = df1[df1["mu_minus_MC15TuneV1_ProbNNmu"] > mu_minus_mu_min_prob]
    df3 = df2[df2["K_MC15TuneV1_ProbNNk"] > K_K_min_prob]
    df4 = df3[df3["Pi_MC15TuneV1_ProbNNpi"] > pi_pi_min_prob]

    print("percentage dataframe removed =", (len(df)-len(df4))/len(df))

    return(df4)


def q2_cuts(df):

    df1 = df[~df.q2.between(8.0, 11.0)]
    df2 = df1[~df1.q2.between(12.5, 15.0)]
    df3 = df2[df2['q2'] < 19.0]
    df4 = df3[df3['q2'] > 0.1]

    print("percentage dataframe removed =", (len(df)-len(df4))/len(df))

    return(df4)
# %%
# This compliles all the checks into a single function, it is made to
# print the percentage data removed at each step, mainly to monitor
# how much data is cut


def check_all(df):
    # For some reason the muon check gets an error if it is not run first
    df0 = q2_cuts(df)
    print('q2 check:')
    per_rem0 = (len(df)-len(df0))/len(df)
    print("percentage total dataframe removed =", (len(df)-len(df0))/len(df))
    print(len(df0))

    df1 = muon_PT_check(df0)
    print('Muon PT check:')
    per_rem1 = (len(df)-len(df1))/len(df)
    diff_rem1 = per_rem1 - per_rem0
    print("percentage total dataframe removed =", (len(df)-len(df1))/len(df))
    print(diff_rem1)
    print(len(df1))

    df2 = invariant_mass_check(df1)
    print('Mass check:')
    per_rem2 = (len(df)-len(df2))/len(df)
    diff_rem2 = per_rem2 - per_rem1
    print("percentage total dataframe removed =", (len(df)-len(df2))/len(df))
    print(diff_rem2)
    print(len(df2))

    df3 = particle_ID_check(df2)
    print('PID check:')
    per_rem3 = (len(df)-len(df3))/len(df)
    diff_rem3 = per_rem3 - per_rem2
    print("percentage total dataframe removed =", (len(df)-len(df3))/len(df))
    print(diff_rem3)
    print(len(df3))

    df4 = IP_check(df3)
    print('IP, vertex check:')
    per_rem4 = (len(df)-len(df4))/len(df)
    diff_rem4 = per_rem4 - per_rem3
    print("percentage total dataframe removed =", (len(df)-len(df4))/len(df))
    print(diff_rem4)
    print(len(df4))

    df5 = mom_check(df4)
    print('Momentum check:')
    per_rem5 = (len(df)-len(df5))/len(df)
    diff_rem5 = per_rem5 - per_rem4
    print("percentage total dataframe removed =", (len(df)-len(df5))/len(df))
    print(diff_rem5)
    print(len(df5))
    print('_______________________________')
    return df5
