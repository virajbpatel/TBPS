# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 13:51:06 2022

@author: ariel
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, classification_report, roc_curve, roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import xgboost as xgb
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from tqdm import tqdm
from yellowbrick.classifier import ClassBalance, ROCAUC, ClassificationReport, ClassPredictionError


dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
#%%

df = pd.DataFrame()
f = open("ref.txt", "w")
simulations = ['jpsi_mu_k_swap','psi2S','jpsi_mu_pi_swap', 'phimumu', 'pKmumu_piTop','signal','pKmumu_piTok_kTop','k_pi_swap','jpsi']
frames = []
for file in simulations:
    temp_df = pd.read_pickle('data/' + file + '.pkl')
    temp_df["class"] = file
    df = df.append(temp_df, ignore_index=True)
    frames.append(temp_df)

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
       'costhetal', 'costhetak', 'polarity'] #+ classlabels



#%%

"Crate Feature and Target"

# Labels/ class dataset
Y = df.iloc[:, -1]
#label encoder
#%%
label_encoder = LabelEncoder()
label_encoded_y = label_encoder.fit_transform(Y)


#%%
X1 = df[variables] #loads df without year (and without class, unless it's part of variables)
#X2 = df[(df["class"] == "signal")] #create signal only file
#X1["class"] = column #for manual onehotencoding method
#X1["class"] = label_encoded_y #for straight y encoding
X = X1.values
#%%

y = label_binarize(label_encoded_y, classes = list(range(len(simulations))))
y_model = pd.DataFrame()
y_model['class'] = label_encoded_y
#%%

# define function to plot 'signal' vs 'background' for a specified variables
# useful to check whether a variable gives some separation between
# signal and background states
# def plot_signal_background(data1, data2, column,
#                         bins=100, x_uplim=0, **kwargs):

#     # if "alpha" not in kwargs:
#     #     kwargs["alpha"] = 0.5

#     df1 = data1[column]
#     df2 = data2[column]

#     fig, ax = plt.subplots()
#     df1 = df1.sample(3000, random_state=1234)
#     df2 = df2.sample(3000, random_state=1234)
#     low = max(min(df1.min(), df2.min()),-5)
#     high = max(df1.max(), df2.max())
#     if x_uplim != 0: high = x_uplim

#     ax.hist(df1, bins=bins, range=(low,high), histtype=u'step', label = "background", **kwargs)
#     ax.hist(df2, bins=bins, range=(low,high), histtype=u'step', label = "signal", **kwargs)
#     plt.legend()
#     plt.title("%s" % (column))

#     if x_uplim != 0:
#         ax.set_xlim(0,x_uplim)
    
#     #plt.show()
#     # ax.set_yscale('log')

# # make plots of all variables

# for key, values in X2.iteritems():
#     print(key)
#     if key == 'year':
#         continue
#     plot_signal_background(X1, X2, key, bins=100)

#%%
# split X1, X2, and y into train and validation dataset 
X1_train,X1_test, y1_train, y1_test  = train_test_split(
    X1,
    y_model,
    test_size=0.2,
    random_state=123456,
    stratify=label_encoded_y,
)


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
    early_stopping_rounds=1000, # stops the training if doesn't improve after 200 iterations
    eval_set=[(X1_train, y1_train), (X1_test, y1_test)],
    eval_metric = "mlogloss", # can use others
    verbose=True,
)


# save the model
xgb_clf.get_booster().save_model('TBPS_xgboost.model')
#%%
# plot ROC curve for training
y1_proba = xgb_clf.predict_proba(X1_test) # outputs probabilities of event being in certain class

y_pred = xgb_clf.predict(X1_test)
#%%
# evaluate predictions
accuracy = accuracy_score(y1_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#%%
"create confusion matrix"



ConfusionMatrixDisplay.from_predictions(y1_test, y_pred, display_labels = simulations, normalize = 'true')
#plt.colorbar()
plt.title("confusion matrix for original discriminant BDT, normalised over true")
plt.savefig("confusion matrix OG normover true.png")
plt.show()

ConfusionMatrixDisplay.from_predictions(y1_test, y_pred, display_labels = simulations, normalize = 'pred')
#plt.colorbar()
plt.title("confusion matrix for original discriminant BDT, normalised over prediction")
plt.savefig("Confusion matrix OG norm over pred.png")
plt.show()

#%%
results = xgb_clf.evals_result()
#epochs = len(results['validation_0']['error'])
#x_axis = range(0, epochs)
# plot log loss
plt.plot(results['validation_0']['mlogloss'], label='Train')
plt.plot(results['validation_1']['mlogloss'], label='Test')
plt.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.grid()
plt.show()
plt.savefig("XGBOOST logloss new discriminant.png")

#%%
"create ROCAUC curve"
classes= [i for i in range(len(simulations))]
rocauc = ROCAUC(xgb_clf, size=(1080, 720), classes = classes)

rocauc.score(X1_test, y1_test)  
r = rocauc.poof()

#%%
"classification report"
report = ClassificationReport(xgb_clf, size=(1080, 720), classes=classes)

report.score(X1_test, y1_test)
c = report.poof()
