import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import os

# Prep data: this is where I combined all the separale .pkl files into one
'''
df = pd.DataFrame()
path = "C:/Viraj/University/Year 3/Comprehensives/TBPS/samples/"
for i,file in enumerate(os.listdir(path)):
    if file.endswith(".pkl"):
        temp_df = pd.read_pickle(path+file)
        temp_df["class"] = i+1
        df = df.append(temp_df, ignore_index=True)

print(df.head())
print(df.info())
df.to_pickle("all_samples_df.pkl")
'''
# The df has an extra column called class which is an int64 based on file order in the path
# Import massive dataframe
df = pd.read_pickle("all_samples_df.pkl")
# Only using values in df
data = df.values
print("Data loaded")
# Final column has labels 
X,Y = data[:,:-1], data[:,-1]
Y = Y.astype('int') # Gives ValueError without this line
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)
print(y_train)
print("Data split")

# Logistic Regressor
# Feel free to play with the arguments in the LR model
LR = LogisticRegression(random_state = 0, solver = 'saga', multi_class = 'multinomial', verbose = True)
print("LR initialised")
# Train LR model
LR = LR.fit(x_train,y_train)
print("LR fitted")
# Predict values of y based on x_test
LR.predict(x_test)
print("LR predicted")
# Compare predicted values with actual values
print(round(LR.score(x_test,y_test), 4)) # Accuracy: 38.75%

# Need to try other classifers: svm.SVC, MLPClassifier (could make our own NN instead), or RandomForestClassifier