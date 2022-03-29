# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 13:51:06 2022

@author: ariel
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import xgboost as xgb
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tqdm import tqdm
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
#%%

df = pd.DataFrame()
f = open("ref.txt", "w")
simulations = ['jpsi_mu_k_swap','psi2S','jpsi_mu_pi_swap', 'phimumu', 'pKmumu_piTop','signal','pKmumu_piTok_kTop','k_pi_swap','jpsi']
for file in simulations:
    temp_df = pd.read_pickle('data/' + file + '.pkl')
    temp_df["class"] = file
    df = df.append(temp_df, ignore_index=True)

#%%
classlabels = []
for file in simulations:
    label = 'class__' + file
    classlabels.append(label)
#%%
variables = ['mu_plus_MC15TuneV1_ProbNNk', 'mu_plus_MC15TuneV1_ProbNNpi',
       'mu_plus_MC15TuneV1_ProbNNmu', 'mu_plus_MC15TuneV1_ProbNNe',
       'mu_plus_MC15TuneV1_ProbNNp', 'mu_plus_P', 'mu_plus_PT', 'mu_plus_ETA',
       'mu_plus_PHI', 'mu_plus_PE', 'mu_plus_PX', 'mu_plus_PY', 'mu_plus_PZ',
       'mu_plus_IPCHI2_OWNPV', 'mu_minus_MC15TuneV1_ProbNNk',
       'mu_minus_MC15TuneV1_ProbNNpi', 'mu_minus_MC15TuneV1_ProbNNmu',
       'mu_minus_MC15TuneV1_ProbNNe', 'mu_minus_MC15TuneV1_ProbNNp',
       'mu_minus_P', 'mu_minus_PT', 'mu_minus_ETA', 'mu_minus_PHI',
       'mu_minus_PE', 'mu_minus_PX', 'mu_minus_PY', 'mu_minus_PZ',
       'mu_minus_IPCHI2_OWNPV', 'K_MC15TuneV1_ProbNNk',
       'K_MC15TuneV1_ProbNNpi', 'K_MC15TuneV1_ProbNNmu',
       'K_MC15TuneV1_ProbNNe', 'K_MC15TuneV1_ProbNNp', 'K_P', 'K_PT', 'K_ETA',
       'K_PHI', 'K_PE', 'K_PX', 'K_PY', 'K_PZ', 'K_IPCHI2_OWNPV',
       'Pi_MC15TuneV1_ProbNNk', 'Pi_MC15TuneV1_ProbNNpi',
       'Pi_MC15TuneV1_ProbNNmu', 'Pi_MC15TuneV1_ProbNNe',
       'Pi_MC15TuneV1_ProbNNp', 'Pi_P', 'Pi_PT', 'Pi_ETA', 'Pi_PHI', 'Pi_PE',
       'Pi_PX', 'Pi_PY', 'Pi_PZ', 'Pi_IPCHI2_OWNPV', 'B0_MM',
       'B0_ENDVERTEX_CHI2', 'B0_ENDVERTEX_NDOF', 'B0_FDCHI2_OWNPV', 'Kstar_MM',
       'Kstar_ENDVERTEX_CHI2', 'Kstar_ENDVERTEX_NDOF', 'Kstar_FDCHI2_OWNPV',
       'J_psi_MM', 'J_psi_ENDVERTEX_CHI2', 'J_psi_ENDVERTEX_NDOF',
       'J_psi_FDCHI2_OWNPV', 'B0_IPCHI2_OWNPV', 'B0_DIRA_OWNPV', 'B0_OWNPV_X',
       'B0_OWNPV_Y', 'B0_OWNPV_Z', 'B0_FD_OWNPV', 'B0_ID', 'q2', 'phi',
       'costhetal', 'costhetak', 'polarity', 'class'] #+ classlabels  

#use classlabels in [variables] when using manual OHC


#%%

"Crate Feature and Target"

# Labels/ class dataset
Y = df.iloc[:, -1]
#label encoder
#%%
label_encoder = LabelEncoder()
#label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.fit_transform(Y)



#%%

#attempt to make onehotencoder manually
# ONE_HOT_COLS = ['class']

# for col in ONE_HOT_COLS:
#     s = df[col].unique()

#     # Create a One Hot Dataframe with 1 row for each unique value
#     one_hot_df = pd.get_dummies(s, prefix='%s_' % col)
#     one_hot_df[col] = s

#     print("Adding One Hot values for %s (the column has %d unique values)" % (col, len(s)))
#     pre_len = len(df)

#     # Merge the one hot columns
#     df = df.merge(one_hot_df, on=[col], how="left")
#     assert len(df) == pre_len
#     print(df.shape)
# #%%
# classtest = df[classlabels]
# column = []

# for i in tqdm(range(len(df['class']))):
#     arr = np.zeros(len(classlabels))
#     for j in range(len(classlabels)):
#         arr[j] = classtest.iloc[i, j]
#     column.append(arr)

#%%
X1 = df[variables] #loads df without year (and without class, unless it's part of variables)
X2 = df[(df["class"] == "signal")] #create signal only file
#X1["class"] = column #for manual onehotencoding method
#X1["class"] = label_encoded_y #for straight y encoding

#X = X1.values

#%%
#this method successfully spits out a column. I'm not sure if it's actually what we want.
# one_hot_enc = OneHotEncoder()

# arr =  one_hot_enc.fit_transform(X1[['class']])
# new_X = pd.DataFrame(arr)

#%%
#One Hot Encoding using SKLEARN package

#Encode Country Column
labelencoder_X = LabelEncoder()
X1.iloc[:, 80] = labelencoder_X.fit_transform(X1.iloc[:, 80])

#%%
ct = ColumnTransformer([("class", OneHotEncoder(), [80])], remainder = 'passthrough')
X1 = ct.fit_transform(X1)
#the first 9 lines are the encoded part


#%%
# Use sklearn and default parameters for xgboost
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, label_encoded_y, test_size=test_size, random_state=seed)
# fit model no training data
model = xgb.XGBClassifier(objective = "multi:softprob")
model.fit(X_train, y_train)
print(model)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
#%%
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#%%

#fine tuning for XGBOOST

# split X1, X2, and y into train and validation dataset 

X1_train,X1_test, y1_train, y1_test  = train_test_split(
    X1,
    label_encoded_y,
    test_size=0.2,
    random_state=123456,
    stratify=label_encoded_y,
)

# X2_train,X2_test, y2_train, y2_test  = train_test_split(
#     X2,
#     y,
#     test_size=0.2,
#     random_state=123456,
#     stratify=y.values,
# )

# In[38]:


# define some XGBoost parameters, unspecified will be default
# https://xgboost.readthedocs.io/en/latest////index.html
# not optimised at all, just playing by ear

xgb_params = {
    "objective": "multi:softprob",
    "max_depth": 5,
    "learning_rate": 0.02,
    "silent": 1,
    "n_estimators": 1000, #1000?
    "subsample": 0.7,
    "seed": 123451,
}


# In[39]:


# first run the training for simple case with just 1 variable

xgb_clf = xgb.XGBClassifier(**xgb_params)
xgb_clf.fit(
    X1_train,
    y1_train,
    early_stopping_rounds=200, # stops the training if doesn't improve after 200 iterations
    eval_set=[(X1_train, y1_train), (X1_test, y1_test)],
    eval_metric = "auc", # can use others
    verbose=True,
)

#%%

# plot ROC curve for training
y1_proba = xgb_clf.predict_proba(X1_test) # outputs probabilities of event being in certain class
y_pred = xgb_clf.predict(X1_test)

# evaluate predictions
accuracy = accuracy_score(y1_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#%%

# #  define a function to plot the ROC curves - just makes the roc_curve look nicer than the default
# def plot_roc_curve(fpr, tpr, auc):
#     fig, ax = plt.subplots()
#     ax.plot(fpr, tpr)
#     ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
#     ax.grid()
#     ax.text(0.6, 0.3, 'ROC AUC Score: {:.3f}'.format(auc),
#             bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='k'))
#     lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
#     ax.plot(lims, lims, 'k--')
#     ax.set_xlim(lims)
#     ax.set_ylim(lims)
#     plt.savefig('roc_rho_rho')
# # look at feature importance
# # can use different metrics (weight or gain), look up online
# xgb.plot_importance(xgb_clf, importance_type='weight')
# xgb.plot_importance(xgb_clf, importance_type='gain')
# auc = roc_auc_score(y1_test, y1_proba[:,1], multi_class = "ovr")
# fpr, tpr, _ = roc_curve(y1_test, y1_proba[:,1])
# plot_roc_curve(fpr, tpr, auc)
# plt.show()