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

def DNN_model(hidden, activation='tanh', loss='binary_crossentropy', optimizer='adam'):
    """
    DNN model with 2+i hidden layer for classification task.
    For more details about the model see the reference
    The model is maded by:

    - Input: shape = (feature), features are the numer of values taken from the past,
    follow the leterature the default is 31.

    - First Hidden: Dense(feature, activation='tanh'), number of nodes is equal to 
    the number of input shape

    - Inner Hiddens: Dense(feature - c*i, activation='tanh'), this is the depp part of the model
    composed by i hidden layers whose number of nodes descresing until reach the numerbs of 
    last hidden layer. The costant c modulates the discending number for nodes, actually the number of nodes
    is determin by an array of type np.linspace(feature,5,hidden).

    - Last hidden: Dense(5, activation='tanh'), numer of nodes are setting to 5

    - Output: Dense(1, activation='sigmoid'), the output is interpretated as the probability that 
    the input is grater than the cross-section median

    Reference: "doi:10.1016/j.ejor.2016.10.031"

    Parameters
    ----------
    hidden: integer
        Number of hidden layers, the actual values of the nodes are fixed 
        - 3: [31,18,5]
        - 4: [31,22,13,5]
        - 5: [31,24,18,11,5]
        - 6: [31,25,20,15,10,5]

    activation: string(optional)
        Activation faction of hidden nodes, default='tanh'

    loss: string(optional)
        Loss fuction, it must be a loss compatible with classification problem, defaul=binary_crossentropy'

    optimater: string(optional)
        Optimazer of the model, default='adam'


    Returns
    -------
    model: tensorflow.python.keras.engine.sequential.Sequential
        tensorflow model with selected hidden layers

    """
    model = Sequential()

    model.add(Input(shape=(31)))
    model.add(Dropout(0.1))

    nodes = [int(i) for i in np.linspace(31,5,hidden)]
    for nod in nodes:
        model.add(Dense(nod, activation=activation))
        model.add(Dropout(0.5))
    
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    print(model.summary()) 
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply DNN for classification problem')
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

    for hid in [3,4,5,6]:
        #DNN model
        model = DNN_model(hidden=hid, optimizer='adam')

        for per in range(0,10):
            #Splitting data for each period
            X_train, y_train, X_test, y_test = all_data_DNN(df_returns, df_binary, per)
            #Trainng
            es = EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True)
            mc = ModelCheckpoint(f'DNN_hidden{hid}_adam_period{per}y.h5', monitor='val_loss', mode='min', verbose=0)
            history = model.fit(X_train ,y_train, callbacks=[es,mc],validation_split=0.2, batch_size=256, epochs=200, verbose=1)
            # model.save(f'DNN_hidden3_adadelta_period{per}.h5')

    #Prediction
    y_pred = model.predict(X_test)
    for i,j in zip(y_test[:365], y_pred[:365]):
        print(i,j)

    #Elbow curve
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss') 
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.grid()
    plt.legend()

    #ROC curve
    # model1 = load_model("DNN_second_period0.h5")
    # model2 = load_model("DNN_second_period1.h5")
    
    # probas_1 = model1.predict(X_test)
    # probas_2 = model2.predict(X_test)

    # fpr1, tpr1, thresholds = roc_curve(y_test, probas_1[:, 0])
    # fpr2, tpr2, thresholds2 = roc_curve(y_test, probas_2[:, 0])

    # roc_auc1 = roc_auc_score(y_test, probas_1[:,0], average=None)
    # roc_auc2 = roc_auc_score(y_test, probas_2[:,0], average=None)

    # plt.figure('ROC CURVE')
    # plt.plot(fpr1, tpr1, label='Model 1 (area = %0.4f)' % (roc_auc1))
    # plt.plot(fpr2, tpr2, label='Model 2 (area = %0.4f)' % (roc_auc2))
    # plt.plot([0, 1], [0, 1], 'k--')

    # plt.xlabel('False Positive Rate',)
    # plt.ylabel('True Positive Rate')
    # plt.legend(loc="lower right", fontsize=12, frameon=False)

    # plt.show()
