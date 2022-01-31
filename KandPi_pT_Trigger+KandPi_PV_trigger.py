# -*- coding: utf-8 -*-
"""
K and Pi displacement from primary vertex (PV) P_Z/P_XY < 40 accepted
K and Pi tranverse momentum >1.6 GeV accepted

@author: Maria
"""
from IPython.display import display
import pandas as pd
import numpy as np

K_PT,K_PX,K_PY,K_PZ,Pi_PT,Pi_PX,Pi_PY,Pi_PZ = np.genfromtxt('total_dataset.csv', delimiter=',',skip_header=1,usecols=(35,39,40,41,49,53,54,55),unpack= True)


#%%

'''Displacement from Primary Vertex check'''
# Displacement from PV checked using ratio of z-plane momentum and xy-plane momentum

# modulus of momenta in XY plane
K_PXY = np.sqrt(K_PX*K_PX+K_PY*K_PY)
Pi_PXY = np.sqrt(Pi_PX*Pi_PX+Pi_PY*Pi_PY)

accepted_K_PV = []
accepted_Pi_PV = []

#threshold for ratio of z-momentum/xy-momentum for K and pi particles
thre_VPdspl_K = 40 #value to be adjusted
thre_VPdspl_pi = 40 #value to be adjusted

for i in range(0,len(K_PXY)):
    
    if K_PZ[i]/K_PXY[i] < thre_VPdspl_K: #kaons z/xy momentum check
        accepted_K_PV.append(i)
        
    if Pi_PZ[i]/Pi_PXY[i] < thre_VPdspl_pi: #pions z/xy momentum check
        accepted_Pi_PV.append(i)

#%% 
        
print(f"Displacement from PV check, Kaon; accepted events; threshold = {thre_VPdspl_K}")
dict = {'K_PX' : K_PX[accepted_K_PV],
        'K_PY' : K_PY[accepted_K_PV],
        'K_PZ' : K_PY[accepted_K_PV]}
df = pd.DataFrame(dict)
display(df)

print(f"Displacement from PV check, Pion; accepted events; threshold = {thre_VPdspl_pi}")
dict = {'Pi_PX' : Pi_PX[accepted_Pi_PV],
        'Pi_PY' : Pi_PY[accepted_Pi_PV],
        'Pi_PZ' : Pi_PZ[accepted_Pi_PV]}
df = pd.DataFrame(dict)
display(df)




#%%

'''Transverse momenta of Pion and Kaon check'''

accepted_K_PT = []
accepted_Pi_PT = []

thre = 1600 #MeV, threshold used in literature on LHCb

for j in range(0,len(K_PT)):
    
    if K_PT[j] > thre:
        accepted_K_PT.append(j)
        
    if Pi_PT[j] > thre:
        accepted_Pi_PT.append(j)


#%%
print(f"Tranverse momenta of Kaon check; accepted events; threshold = {thre}")
dict = {'K_PT' : K_PT[accepted_K_PT]}
df = pd.DataFrame(dict)
display(df)

print(f"Tranverse momenta of Pion check; accepted events; threshold = {thre}")
dict = {'Pi_PT' : Pi_PT[accepted_Pi_PT]}
df = pd.DataFrame(dict)
display(df)

