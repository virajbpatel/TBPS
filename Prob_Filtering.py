#%%
from IPython.display import display
import pandas as pd
import numpy as np
import time

start_time = time.time() # start timer

##### Input #####
max_rows = 498244 # number of events to read
#max_rows = 100
#################

#Considering mu_plus track
# print('Reading mu_Plus track Data...')
# ProbNNk,ProbNNpi,ProbNNmu,ProbNNe, ProbNNp = np.genfromtxt('total_dataset.csv', delimiter=',',skip_header=1,usecols=(1,2,3,4,5),unpack= True, max_rows = max_rows)

#Considering pi track usecols= 14*2 + (1,2,3,4,5)
print('Reading pi track Data...')
ProbNNk,ProbNNpi,ProbNNmu,ProbNNe, ProbNNp = np.genfromtxt('total_dataset.csv', delimiter=',',skip_header=1,usecols=(29,30,31,32,33),unpack= True, max_rows = max_rows)

#Considering mu_minus track usecols=14 + (1,2,3,4,5)
##### Input #####
thre = 0.6 #threshold
#################

loc_k = (ProbNNk > thre) 
loc_pi = (ProbNNpi > thre)
loc_mu = (ProbNNmu > thre)
loc_e = (ProbNNe > thre)
loc_p = (ProbNNp > thre)

##### Input #####
loc_need = loc_pi
loc_list = [loc_k, loc_mu, loc_e, loc_p] # locs except loc_pi
#################
#%%
#check that only ProbNNmu is greater than the threshold
print('Now check that only ProbNNmu is greater than the threshold')
event_unclear = []

for loc in loc_list: 
    print(loc)
    for i in range (len(loc_need)):
        print(i)
        if loc_mu[i]: #if ProbNNmu[loc_mu[i]]>threshold
            if (loc_need == loc)[i]: # if ProbNNmu[loc_mu[i]] & ProbNNX[loc_X[i]]>threshold at the same time
                event_unclear.append(i)
           
#%%
# Find the locations of clear events, i.e., only ProbNNmu > threshold
loc_need[event_unclear] = False

#%% 
print("Running time = %s s" % (time.time() - start_time)) # print out running time
     
#%%
# displaying the unclear events
print(f"pi track (the first 100 events); unclear events; threshold = {thre}")
dict = {'ProbNNk' : ProbNNk[event_unclear],
        'ProbNNpi' : ProbNNpi[event_unclear],
        'ProbNNmu' : ProbNNmu[event_unclear],
        'ProbNNe' : ProbNNe[event_unclear],
        'ProbNNp' : ProbNNp[event_unclear]}
df = pd.DataFrame(dict)
display(df)
##### Input #####
df.to_csv('pi track unclear events.csv')
#################

#%%
# displaying the clear events
print(f"pi track (the first 100 events); clear events; threshold = {thre}")

dict = {'ProbNNk' : ProbNNk[loc_need],
        'ProbNNpi' : ProbNNpi[loc_need],
        'ProbNNmu' : ProbNNmu[loc_need],
        'ProbNNe' : ProbNNe[loc_need],
        'ProbNNp' : ProbNNp[loc_need]}
df = pd.DataFrame(dict)

display(df)
##### Input #####
df.to_csv('pi track clear events.csv')
#################

#%%