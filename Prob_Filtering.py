#%%
from IPython.display import display
import pandas as pd
import numpy as np

#Considering mu_Plus track
ProbNNk,ProbNNpi,ProbNNmu,ProbNNe, ProbNNp = np.genfromtxt('total_dataset.csv', delimiter=',',skip_header=1,usecols=(1,2,3,4,5),unpack= True)

max_row = 100 # number of events to read

ProbNNk,ProbNNpi,ProbNNmu,ProbNNe, ProbNNp = ProbNNk[:max_row],ProbNNpi[:max_row],ProbNNmu[:max_row],ProbNNe[:max_row], ProbNNp[:max_row]

thre = 0.6 #threshold
loc_k = (ProbNNk > thre) 
loc_pi = (ProbNNpi > thre)
loc_mu = (ProbNNmu > thre)
loc_e = (ProbNNe > thre)
loc_p = (ProbNNp > thre)

loc_list = [loc_k, loc_pi, loc_e, loc_p] # locs except loc_mu

#%%
#check that only ProbNNmu is greater than the threshold
event_unclear = []

for loc in loc_list: 
    for i in range (len(loc_mu)):
        if loc_mu[i]: #if ProbNNmu[loc_mu[i]]>threshold
            if (loc_mu == loc)[i]: # if ProbNNmu[loc_mu[i]] & ProbNNX[loc_X[i]]>threshold at the same time
                event_unclear.append(i)
                
#%%
# displaying the unclear events
print(f"mu_Plus track (the first 100 events); unclear events; threshold = {thre}")
dict = {'ProbNNk' : ProbNNk[event_unclear],
        'ProbNNpi' : ProbNNpi[event_unclear],
        'ProbNNmu' : ProbNNmu[event_unclear],
        'ProbNNe' : ProbNNe[event_unclear],
        'ProbNNp' : ProbNNp[event_unclear]}
df = pd.DataFrame(dict)
display(df)

#%%
# Find the locations of clear events, i.e., only ProbNNmu > threshold
loc_mu[event_unclear]=False

#%%
# displaying the clear events
print(f"mu_Plus track (the first 100 events); clear events; threshold = {thre}")

dict = {'ProbNNk' : ProbNNk[loc_mu],
        'ProbNNpi' : ProbNNpi[loc_mu],
        'ProbNNmu' : ProbNNmu[loc_mu],
        'ProbNNe' : ProbNNe[loc_mu],
        'ProbNNp' : ProbNNp[loc_mu]}
df = pd.DataFrame(dict)
display(df)

#%%