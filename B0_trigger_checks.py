# -*- coding: utf-8 -*-
"""
1. vertex fit chi2 prob. > 10 %
     (56) B0_ENDVERTEX_CHI2 > 10 %
2. cos(αxy) > 0.9994, where αxy is the angle, in the transverse plane, between the B0 momentum vector and the line-of-flight between the beamspot and the B0 vertex.
     (68) B0_DIRA_OWNPV > 0.9994
3. Invariant mass must be within 280 MeV of the accepted B0 mass mB0 for both hypotheses
     (55) 5000 < B0_MM = 5280 MeV/c^2 < 5560

@author: Alexander
"""
from IPython.display import display
import pandas as pd
import numpy as np

B_chi2, B_alpha, B_MM = np.genfromtxt('total_dataset.csv', delimiter=',',skip_header=1,usecols=(58,70,57),unpack= True)

#%% 1.
"""Vertex fit of B0 chi2 check"""

accepted_B_chi2 = []

thrsh = 10 #% (as given in literature review)

for i in range(0,len(B_chi2)):
    
    if B_chi2[i] > thrsh:
        accepted_B_chi2.append(i)

#%%
print(f"Chi2 of B0 vertex fit; accepted events; threshold = {thrsh}")
dict = {"B0_chi2" : B_chi2[accepted_B_chi2]}
df = pd.DataFrame(dict)
display(df)

#%% 2.
"""Implementing selection criterion on angle"""

accepted_B_alpha = []

thrsh = 0.9994 # (as given in literature review)

for i in range(0,len(B_alpha)):
    
    if B_alpha[i] > thrsh:
        accepted_B_alpha.append(i)

#%%
print(f"Angle alpha; accepted events; threshold = {thrsh}")
dict = {"B_alpha" : B_alpha[accepted_B_alpha]}
df = pd.DataFrame(dict)
display(df)

#%% 3.
"""Checking range of invariant mass"""

accepted_B_MM = []

thrsh1 = 5000
thrsh2 = 5560

for i in range(0,len(B_MM)):
    
    if B_MM[i] > thrsh1 or B_MM[i] < thrsh2:
        accepted_B_MM.append(i)

#%%
print(f"Invariant mass; accepted events; threshold = {thrsh1, thrsh2}")
dict = {"B_MM" : B_MM[accepted_B_MM]}
df = pd.DataFrame(dict)
display(df)

#%%
"""
The very small percentages of rejected events for cases 2. and 3. a sign
of pre-filtering data set. Chi2 test only is significant"""