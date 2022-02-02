# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 20:48:28 2022

@author: kevin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit as minuit
import module_fit_functions as mfit


# Import datasets and plot the histograms
df = pd.read_pickle('filt_frst_3101.pkl')

var = ['phi', 'costhetal', 'costhetak', 'q2']

for i in var:    
    plt.hist(df[i], bins = 25, density = True, histtype = 'step')
    plt.title('fit_first_3101')
    plt.xlabel(i)
    plt.ylabel('P')
    plt.show()

df_mc = pd.read_pickle('acceptance_mc.pkl')

for i in var:
    plt.hist(df_mc[i], bins = 25, density = True, histtype = 'step')
    plt.title('acceptance_mc')
    plt.xlabel(i)
    plt.ylabel('P')
    plt.show()

#%% overlay the two histograms

i = var[1]
n, bins, patch = plt.hist(df[i], bins = 25, density = True, histtype = 'step')
n_mc, bins_mc, patch_mc = plt.hist(df_mc[i], bins = 25, density = True, histtype = 'step')

x_acc, n_acc, params_acc, cov_acc = mfit.accept_fit(n_mc, bins_mc)

plt.plot(x_acc, n_acc)
plt.xlabel(i)
plt.ylabel('P')
plt.show()

def acceptance_func(x):
    y = mfit.polynomial(x, *params_acc)
    return y

def integrate(x,y):
    da_list = []
    for i in range(len(x)-1):
        dx = x[i+1] - x[i]
        avg_y = (y[i+1] + y[i])/2
        da = avg_y*dx
        da_list.append(da)
    A = np.sum(da_list)
    return A

def d2gamma_p_d2q2_dcostheta(fl, afb, cos_theta_l):
    """
    Returns the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param cos_theta_l: cos(theta_l)
    :return:
    """
    ctl = cos_theta_l
    c2tl = 2 * ctl ** 2 - 1
    scalar_array = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) + 8/3 * afb * ctl) * acceptance_func(ctl)
    return scalar_array

def norm_d2gamma_p_d2q2_dcostheta(fl, afb, cos_theta_l):
    """
    Returns the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param cos_theta_l: cos(theta_l)
    :return:
    """
    ctl = cos_theta_l
    c2tl = 2 * ctl ** 2 - 1
    scalar_array = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) + 8/3 * afb * ctl) * acceptance_func(ctl)
    norm_factor = integrate(ctl, d2gamma_p_d2q2_dcostheta(fl=fl, afb=afb, cos_theta_l=ctl))
    normalised_scalar_array = scalar_array /norm_factor  # normalising scalar array to account for the non-unity acceptance function
    return normalised_scalar_array
#%%

df_ctl = df['costhetal'].sort_values().to_numpy()
print(df['costhetal'].sort_values())

y_test = d2gamma_p_d2q2_dcostheta(0, 0.07, df_ctl)
y_test2 = norm_d2gamma_p_d2q2_dcostheta(0, 0.07, df_ctl)

plt.plot(df_ctl, y_test)
plt.show()
plt.plot(df_ctl, y_test2)

print(integrate(df_ctl, y_test))
print(integrate(df_ctl, y_test2))

# showed that norm_d2gamma_p_d2q2_dcostheta is a normalised distributiom in the integral over costhetal


#%%
def log_likelihood(fl, afb):
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    :return:
    """
    normalised_scalar_array = norm_d2gamma_p_d2q2_dcostheta(fl=fl, afb=afb, cos_theta_l= df_ctl)
    return - np.sum(np.log(normalised_scalar_array))

x = np.linspace(-1,1,1000)
_test_afb = 0.7
_test_fl = 0.0

# plt.plot(x, [norm_d2gamma_p_d2q2_dcostheta(fl=i, afb, cos_theta_l)for i in x])
# plt.show()
# plt.plot(x, [log_likelihood(fl=i, afb = _test_afb, ctl=df_ctl) for i in x])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
ax1.plot(x, [log_likelihood(fl=i, afb=_test_afb) for i in x])
ax1.set_title(r'$A_{FB}$ = ' + str(_test_afb))
ax1.set_xlabel(r'$F_L$')
ax1.set_ylabel(r'$-\mathcal{L}$')
ax1.grid()
ax2.plot(x, [log_likelihood(fl=_test_fl, afb=i) for i in x])
ax2.set_title(r'$F_{L}$ = ' + str(_test_fl))
ax2.set_xlabel(r'$A_{FB}$')
ax2.set_ylabel(r'$-\mathcal{L}$')
ax2.grid()
plt.tight_layout()
plt.show()

#%%

# bin_number_to_check = 0  # bin that we want to check in more details in the next cell
bin_results_to_check = None

log_likelihood.errordef = minuit.LIKELIHOOD
decimal_places = 3
starting_point = [-0.1,0.0]
fls, fl_errs = [], []
afbs, afb_errs = [], []
m = minuit(log_likelihood, fl=starting_point[0], afb=starting_point[1])
# m.fixed['ctl'] = True  # fixing the bin number as we don't want to optimize it
m.limits=((-1.0, 1.0), (-1.0, 1.0))
m.migrad()
m.hesse()
bin_results_to_check = m
fls.append(m.values[0])
afbs.append(m.values[1])
fl_errs.append(m.errors[0])
afb_errs.append(m.errors[1])
print(f"Bin: {np.round(fls, decimal_places)} pm {np.round(fl_errs, decimal_places)},", 
      f"{np.round(afbs, decimal_places)} pm {np.round(afb_errs, decimal_places)}. Function minimum considered valid: {m.fmin.is_valid}")

#%%

plt.figure(figsize=(8, 5))
plt.ticklabel_format(style = 'plain')
plt.subplot(221)
bin_results_to_check.draw_mnprofile('afb', bound=3)
plt.subplot(222)
bin_results_to_check.draw_mnprofile('fl', bound=3)
plt.tight_layout()
plt.ticklabel_format(style = 'plain')
plt.show()

#%%

plt.hist(df[var[1]], bins = 25, density = True, histtype = 'step')
plt.plot(df_ctl, norm_d2gamma_p_d2q2_dcostheta(fls[0], afbs[0], df_ctl))
plt.title('fit_first_3101')
plt.xlabel(var[1])
plt.ylabel('P')
plt.show()
