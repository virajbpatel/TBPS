# TBPS Neural Network attempt 1
# Accuracy: 98.09%
# Loss: 6.81%
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import seaborn as sns

# Function to preprocess raw dataframe
def prep_data(df):
    data = df.values
    # Classes (labels) are in the final column, the rest are features
    X,Y = data[:,:-1], data[:,-1]
    Y = Y.astype('int') # Gives ValueError without this line
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42, stratify = Y)
    scaler = StandardScaler()
    x_train_normalised = scaler.fit_transform(x_train)
    x_test_normalised = scaler.transform(x_test)
    return x_train_normalised, x_test_normalised, y_train, y_test

# Builds classifier model from number of features
def build_model(num_features):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape = (num_features,)))
    model.add(tf.keras.layers.Dense(32, activation = 'relu', kernel_regularizer = 'l2'))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    model.compile(
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = ['accuracy']
    )
    return model

def plot_history(history, parameter):
    if parameter == "acc":
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(['train', 'val'], loc = 'upper left')
        plt.savefig('nn_binary_accuracy.png')
        plt.show()
    elif parameter == "loss":
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(['train', 'val'], loc = 'upper right')
        plt.savefig('nn_binary_loss.png')
        plt.show()

# Plots heatmap based on confusion matrix
def plot_heatmap(class_names, y_pred, y_test):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(15, 15))
    heatmap = sns.heatmap(cm, fmt='g', cmap='Blues', annot=True, ax=ax)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(class_names, rotation = 90)
    ax.yaxis.set_ticklabels(class_names, rotation = 0)
    # Save the heatmap to file
    heatmapfig = heatmap.get_figure()
    heatmapfig.savefig('confusion_matrix.png')

def main():
    # Import data file
    df = pd.read_pickle("all_samples_df.pkl")
    df = df.drop(columns = ['year'])
    # Reformat data for binary classification
    df.loc[df['class'] != 8, 'class'] = 0 # 0 for not a signal
    df.loc[df['class'] == 8, 'class'] = 1 # 1 for signal
    cols = df.columns.tolist()
    print(df.head())
    # Record features and labels
    features, labels = cols[:-1], cols[-1]
    # Preprocess data
    x_train, x_test, y_train, y_test = prep_data(df)
    # Build learning model
    model = build_model(len(features))
    print("Summary of Classifier: ")
    model.summary()
    # Trains model
    epochs = 100
    batch_size = 1024
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', min_delta = 0.0001, patience = 3)
    history = model.fit(
        x_train,
        y_train,
        epochs = epochs,
        batch_size = batch_size,
        callbacks = [earlystop_callback],
        validation_split = 0.1,
        verbose = 1
    )
    # Plots history of model
    plot_history(history, 'acc')
    plot_history(history, 'loss')
    # Tests model
    score = model.evaluate(x_test, y_test, verbose = 0)
    print(f'Loss: {score[0]}')
    print(f'Accuracy: {score[1]}')
    # Predicts classes based on test data
    y_pred = model.predict(x_test)
    y_pred = np.round(y_pred)
    class_names = ["no signal", "signal"]
    print(classification_report(y_test, y_pred, target_names = class_names))
    plot_heatmap(class_names, y_pred, y_test)
    plt.show()
    model.save('/hep_models/kstar_nn_binary')
    
    

if __name__ == "__main__":
    main()