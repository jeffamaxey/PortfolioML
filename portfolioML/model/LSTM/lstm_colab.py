#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LSTM model suited for running easily on Colab"""
import numpy as np
import pandas as pd
import logging
import argparse
import tensorflow as tf
from keras.layers import Input, Dense, LSTM, Dropout
from keras.models import Model, Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop, Adam
from sklearn.metrics import roc_curve, auc, roc_auc_score
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath("..")))
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, LSTM, RepeatVector, TimeDistributed
from keras.layers.convolutional import Conv1D, MaxPooling1D
def read_filepath(file_path):
    """
    Read and compute basic informations about a data set in csv.

    Parameters
    ----------
    file_path: str
        Path to the csv file

    Returns
    -------
    df: pandas dataframe
        Pandas Dataframe of the read file
    """
    name_file = file_path.split('/')[-1]
    extention = name_file.split('.')[-1]

    try:
        if extention != 'csv':
            raise NameError('The file is not a csv one')
    except NameError as ne:
        logging.error(ne)
        exit()

    df = pd.read_csv(file_path, encoding='latin-1')
    df = df.drop(['Days'], axis=1)

    # logging.info('DATA INFO, attributes: %s', df.columns)
    # logging.info('DATA INFO, shape: %s', df.shape)

    # total_data = df.shape[0]*df.shape[1]
    # total_missing = pd.DataFrame.isnull(df).sum().sum()

    # logging.info(f'DATA QUALITY, missing values: {total_missing/total_data:.2%}')

    return df

def split_Tperiod(df_returns, df_binary, len_period=1308, len_test=327):
    """
    Split the entire dataframe in study period T, each of them having len_period
    elements of which len_test account for the trading set. To generate all the
    periods, a rolling window of len_period lenght is moved along the entire
    dataset in len_test steps.


    Parameters
    ----------
    df_returns: pandas dataframe
        Pandas dataframe of returns.

    df_binary: pandas dataframe
        Pandas dataframe of binary targets.

    len_period: integer(optional)
        Lenght of the study period

    len_test: integer(optional)
        Lenght of the trading set.
    Results
    -------
    periods: list of pandas dataframe
        List of pandas dataframe of all periods of lenght len_period.
    """

    len_total_leave = len(df_returns)-len_period #ho solo chiamato come unica variabile quella cosa che c'era nel for, il nome è da rivedere
    periods_ret = [(df_returns[i:len_period+i]) for i in range(0, len_total_leave+1, len_test)]
    periods_bin = [(df_binary[i:len_period+i]) for i in range(0, len_total_leave+1, len_test)] # questa mancava

    return periods_ret, periods_bin


def get_sequences(returns, targets, n_steps=240):
    """
    Returns the  sequences of inputs and targets for classification task (not already
    ready for the LSTM).


    Parameters
    ----------
    returns: pandas dataframe
        pandas dataframe of time-series data of returns to split.

    targets: pandas dataframe
        pandas dataframe of time-series data of target to split. It must have the same length of returns

    n_steps: integer(optional)
        number of time steps for each istance. Default = 100

    Results
    -------
    X: list
        Array of the input set, its shape is (len(sequences)-n_steps, n_steps)

    y: list
        Array of the input target, its shape is (len(sequences)-n_steps, 1)
    """
    try:
        returns = returns.to_numpy()
        targets = targets.to_numpy()
    except AttributeError:
        pass

    X = [returns[i:i+n_steps] for i in range(len(returns)-n_steps)]
    y = [targets[i+n_steps] for i in range(len(targets)-n_steps)]

    return X, y


def get_train_set(df_returns, df_binary):
    """
    Return the train set for the LSTM.
    The argumets are the returns dataframe and the binary dataframe. The function compute respectively
    the X_train and the y_train for classification task, stacking sequences of different companies
    one over another.

    Parameters
    ----------
    df_returns: pandas dataframe, numpy array
        Dataframe of returns

    df_binary: pandas dataframe, numpy array
        Datframe of binary target associated to data returns. It has the same shape of df_returns

    Returns
    -------
    list_tot_X: numpy array
        Array of input data for LSTM

    list_tot_y: numpy array
        Array of input target class for LSTM
    """

    list_tot_X = []
    list_tot_y = []

    if (str(type(df_returns)) != "pandas.core.frame.DataFrame"):
        df_returns = pd.DataFrame(df_returns)
    if (str(type(df_binary)) != "pandas.core.frame.DataFrame"):
        df_binary = pd.DataFrame(df_binary)

    for comp in df_returns.columns:
        X_train, y_train = get_sequences(df_returns[comp], df_binary[comp])
        list_tot_X.append(X_train)
        list_tot_y.append(y_train)

    list_tot_X = np.array(list_tot_X)
    list_tot_y = np.array(list_tot_y)

    list_tot_X = np.reshape(list_tot_X,(list_tot_X.shape[0]*list_tot_X.shape[1],list_tot_X.shape[2]))
    list_tot_y = np.reshape(list_tot_y,(list_tot_y.shape[0]*list_tot_y.shape[1]))

    return list_tot_X, list_tot_y

def all_data_LSTM(df_returns, df_binary, period, len_train=981):
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

    return X_train, y_train, X_test, y_test

def LSTM_model(num_units=25, drop_out=0.1):
    inputs = Input(shape= (240, 1))
    drop = Dropout(drop_out)(inputs)
    hidden = LSTM(num_units, return_sequences=False)(drop)
    drop = Dropout(drop_out)(hidden)
    outputs = Dense(1, activation='sigmoid')(drop)

    model = Model(inputs=inputs, outputs=outputs)

    # Two optimiziers used during training
    rms_prop = RMSprop(learning_rate=0.005, momentum=0.5, clipvalue=0.5)
    adam = Adam(learning_rate=0.005)

    model.compile(loss='binary_crossentropy', optimizer=rms_prop, metrics=['accuracy'])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creation of input and output data for lstm classification problem')
    parser.add_argument('returns_file', type=str, help='Path to the returns input data')
    parser.add_argument('binary_file', type=str, help='Path to the binary target data')
    parser.add_argument("-log", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument('num_units', type=int, help='Number of units in the LSTM layer')
    parser.add_argument('num_periods', type=int, help='Number of periods you want to train')
    parser.add_argument('num_epochs', type=int, help='Number of epochs you want to train')
    parser.add_argument('batch_size', type=int, help='Batch_size')
    parser.add_argument('drop_out', type=float, help='Value of the dropout')

    args = parser.parse_args()

    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}

    logging.basicConfig(level= levels[args.log])
    # tf.get_logger().setLevel('CRITICAL')
    pd.options.mode.chained_assignment = None # Mute some warnings of Pandas

    #Read the data
    df_returns = read_filepath(args.returns_file)
    df_binary = read_filepath(args.binary_file)
    # Pass or not the weights from one period to another
    recursive = True


    for i in range(args.num_periods):
        logging.info(f'============ Start Period {i}th ===========')
        if (i!=0) and (recursive):
            logging.info('LOADING PREVIOUS MODEL')
            model = load_model(f"LSTM_{i-1}_period.h5")
        else:
            logging.info('CREATING NEW MODEL')
            model = LSTM_model(args.num_units, args.drop_out)
        logging.info(model.summary())
        X_train, y_train, X_test, y_test = all_data_LSTM(df_returns, df_binary, i)
        es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        mc = ModelCheckpoint(f'LSTM_{i}_period.h5', monitor='val_loss', mode='min', verbose=0)
        history = model.fit(X_train, y_train, epochs=args.num_epochs, batch_size=args.batch_size,
                            callbacks=[es,mc], validation_split=0.2, shuffle=False, verbose=1)

        plt.figure(f'Period {i} Losses')
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'losses_{i}.png')

        plt.figure(f'Period {i} Accuracies')
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'accuracies_{i}.png')

        y_pred = model.predict(X_test)
        y_pred_companies = [y_pred[i:87+i] for i in range(0,len(y_pred)-87+1,87)]
        dict_comp = {df_returns.columns[i]: y_pred_companies[i] for i in range(0,365)}
        df_predictions = pd.DataFrame()
        for tick in df_returns.columns:
            df_predictions[tick] = dict_comp[tick][:,0]
        df_predictions.to_csv(f'Predictions_{i}th_Period.csv')


        logging.info(f'============ End Period {i}th ===========')



