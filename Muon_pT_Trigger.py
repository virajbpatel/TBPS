# -*- coding: utf-8 -*-
"""
Code to filter out events where no muon has transverse momentum >1.76GeV.
Corresponds to L0muon trigger in OneNote.
"""

#%%
from IPython.display import display
import pandas as pd
import numpy as np

#Considering mu_Plus and mu_Minus transverse momenta and check that at least
#one has transverse momentum above threshold value
mu_plus_PT,mu_minus_PT = np.genfromtxt('total_dataset.csv', delimiter=',',skip_header=1,usecols=(7,21),unpack= True)

max_row = 500000 # number of events to read

mu_plus_PT,mu_minus_PT = mu_plus_PT[:max_row],mu_minus_PT[:max_row]

thre = 1760 #MeV, threshold (value used in literature dependendt on LHCb version)
loc_mu_plus = (mu_plus_PT > thre) 
loc_mu_minus = (mu_minus_PT > thre)

#%%
#check that at least one muon has momentum above threshold
failed_events = []
accepted_events = []

 
for i in range (len(loc_mu_plus)):
    if loc_mu_plus[i]==False and loc_mu_minus[i]==False:
        failed_events.append(i)
    else:
        accepted_events.append(i)
        
                
#%%
# displaying the disregarded events
print(f"Transverse muon momenta check; diregarded events; threshold = {thre} MeV")
dict = {'mu_plus_PT' : mu_plus_PT[failed_events],
        'mu_minus_PT' : mu_minus_PT[failed_events]}
df = pd.DataFrame(dict)
display(df)

#%%

# displaying the accepted events
print(f"Transverse muon momenta check; accepted events; threshold = {thre} MeV")

dict = {'mu_plus_PT' : mu_plus_PT[accepted_events],
        'mu_minus_PT' : mu_minus_PT[accepted_events]}
df = pd.DataFrame(dict)
display(df)

