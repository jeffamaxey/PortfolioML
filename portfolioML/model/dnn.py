"""DNN model"""
import numpy as np
import pandas as pd
import logging
import argparse
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath("..")))
from split import split_Tperiod, get_train_set
from portfolioML.data.data_returns import read_filepath

import matplotlib.pyplot as plt

def all_data_LSTM(df_returns, df_binary, period, len_train=981, len_test=327):
    """
    Function that create the right input for the LSTM algorithm.
    X_train and X_test are normalized. X_train is reshaped.

    Parameters
    ----------
    df_returns : pandas dataframe
        Pandas dataframe of returns.
    df_binary : pandas dataframe
        Pandas dataframe of returns..
    period : int
        Period over which you wanto to create the input for the LSTM.
    len_train : int, optional
        Lenght of the training set. The default is 981.
    len_test : int, optional
        Lenght of the trading set. The default is 327.

    Returns
    -------
    X_train : numpy array

    y_train : numpy array

    X_test : numpy array

    y_test : numpy array

    """
    scaler = StandardScaler()

    periods_returns, periods_binary = split_Tperiod(df_returns, df_binary)

    T1_input = periods_returns[period]
    T1_target = periods_binary[period]

    T1_input[:len_train] = scaler.fit_transform(T1_input[:len_train])

    X_input_train, y_input_train = T1_input[:len_train], T1_target[:len_train]

    T1_input[len_train:] = scaler.fit_transform(T1_input[len_train:])
    X_test, y_test = T1_input[len_train:], T1_target[len_train:]

    X_train, y_train = get_train_set(X_input_train, y_input_train)
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_test, y_test = get_train_set(X_test, y_test)
    X_test, y_test = np.array(X_test), np.array(y_test)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_train, y_train, X_test, y_test

def all_data_DNN(df_returns, df_binary, period, len_train=981, len_test=327):
    """
    Create a right input data for DNN starting from the right data of LSTM.
    Indeed the parameters are the same of the all_data_LSTM, these are changed select
    anly m values (features) exctrated from the 240 values in the LSTM input data.
    """
    X_train, y_train, X_test, y_test = all_data_LSTM(df_returns, df_binary, period)

    m = list(range(0,240,20))+list(range(221,240))
    X_train = X_train[:,m,:]
    X_train = np.reshape(X_train, (X_train.shape[0], 31))

    X_test = X_test[:,m,:]
    X_test = np.reshape(X_test, (X_test.shape[0], 31))

    return X_train, y_train, X_test, y_test

def DNN_model(*nodes_args, hidden=None , activation='tanh', loss='binary_crossentropy', optimizer='adam'):
    """
    DNN model with selected number of hidden layer for classification task.
    For more details about the model see the reference
    The model is maded by:

    - Input: shape = (feature), features are the numer of values taken from the past,
    follow the leterature the default is 31.

    - Hidden Layers: Dense(feature, activation=activation), sequential hidden layers full-connected 
    with different nodes. If hiddin is an integer the number of nodes for each layer follow a descrescent
    way from 31 to 5, note that the actual values of the nodes is determine by np.linspace(feature,5,hidden).
    
    - Output: Dense(1, activation='sigmoid'), the output is interpretated as the probability that 
    the input is grater than the cross-section median

    Reference: "doi:10.1016/j.ejor.2016.10.031"

    Parameters
    ----------

    *nodes_args: integer 
        Number of nodes for each layers.    

    hidden: integer(optional), default = None
        Number of hidden layers, the actual values of the nodes are fixed in descrescent way 
        from 3 to 5 through the np.linspace function (np.linspace(31,5,hidden)). 
        Follow some useful example:
        - 3: [31,18,5]
        - 4: [31,22,13,5]
        - 5: [31,24,18,11,5]
        - 6: [31,25,20,15,10,5]

    activation: string(optional)
        Activation function to use of hidden layers, default='tanh'
        Reference: https://keras.io/api/layers/core_layers/dense/

    loss: String (name of objective function), objective function or tf.keras.losses.Loss instance. See tf.keras.losses.
        Loss fuction, it must be a loss compatible with classification problem, defaul=binary_crossentropy'
        Reference: https://www.tensorflow.org/api_docs/python/tf/keras/Model

    optimater: string(optional)
        String (name of optimizer) or optimizer instance. See tf.keras.optimizers., default='adam'
        Reference: https://www.tensorflow.org/api_docs/python/tf/keras/Model


    Returns
    -------
    model: tensorflow.python.keras.engine.sequential.Sequential
        tensorflow model with selected hidden layers

    """
    model = Sequential()

    model.add(Input(shape=(31)))
    model.add(Dropout(0.1))

    if hidden is not None:
        logging.info("Nember of layers is determined by 'hidden',numebrs of neurons descrescent from 31 to 5")
        nodes = [int(i) for i in np.linspace(31,5,hidden)]
    else:
        nodes = nodes_args

    for nod in nodes:
        model.add(Dense(nod, activation=activation))
        model.add(Dropout(0.5))
    
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    logging.info(model.summary()) 
    return model

def training(model, model_feature, periods=10, validation_split=0.2, batch_size=1024, epochs=400):
    """
    Training of a selected model over several study period. Plot for each periods
    loss trand and accuracy trand.
    Models are saved in format "h5" for future development

    Paremeters
    ----------
    model: tensorflow.python.keras.engine
        Tensorflow model to training

    model_feature: bool
         

    periods: integer(optional)
        Study periods over wich the model are traning. Default = 10

    validation_split: float between 0 and 1
        Part of training set dedicated for validation part. Default = 0.2

    batch_size: Integer or None
        Number of samples per gradient update.
        Do not specify the batch_size if your data is in the form of datasets,
        generators, or keras.utils.Sequence instances (since they generate batches). Default = 1024.
        References: https://www.tensorflow.org/api_docs/python/tf/keras/Model

    epochs: Integer
        Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
        Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". 
        The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
        References: https://www.tensorflow.org/api_docs/python/tf/keras/Model
    """
    for per in range(0,periods):
        #Splitting data for each period
        if model_feature:
            X_train, y_train, X_test, y_test = all_data_DNN(df_returns, df_binary, per)
        else:
            X_train, y_train, X_test, y_test = all_data_LSTM(df_returns, df_binary, per)
        
        es = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        mc = ModelCheckpoint(f'DNN_mymod3_adadelta_period{per}.h5', monitor='val_loss', mode='min', verbose=0)
        history = model.fit(X_train ,y_train, callbacks=[es,mc],
                            validation_split=validation_split, batch_size=batch_size, epochs=epochs, verbose=1)

        #Elbow curve
        plt.figure(f'Loss and Accuracy period {per}')
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], label='train_loss') 
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epochs')
        plt.title('Training and Validation Loss vs Epochs')
        plt.grid()
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epochs')
        plt.title('Training and Validation Accuracy vs Epochs')
        plt.grid()
        plt.legend()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make DNN for classification task to predicti class label 0 or 1')
    parser.add_argument('returns_file', type=str, help='Path to the returns input data')
    parser.add_argument('binary_file', type=str, help='Path to the binary target data')
    parser.add_argument("-log", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))

    args = parser.parse_args()

    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}

    logging.basicConfig(level= levels[args.log])

    #Read the data
    df_returns = read_filepath(args.returns_file)
    df_binary = read_filepath(args.binary_file)

    for per in range(5,10):
        model = DNN_model(150,80,15,5, optimizer='adam')
        #Splitting data for each period
        X_train, y_train, X_test, y_test = all_data_DNN(df_returns, df_binary, per)
        #Trainng
        es = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
        mc = ModelCheckpoint(f'DNN_mymod4_period{per}.h5', monitor='val_loss', mode='min', verbose=0)
        history = model.fit(X_train ,y_train, callbacks=[es,mc],validation_split=0.2, batch_size=256, epochs=400, verbose=1)

        #Elbow curve
        plt.figure(f'Loss and Accuracy period {per}')
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], label='train_loss') 
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epochs')
        plt.title('Training and Validation Loss vs Epochs')
        plt.grid()
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epochs')
        plt.title('Training and Validation Accuracy vs Epochs')
        plt.grid()
        plt.legend()

    plt.show()

    
    plt.show()

    
