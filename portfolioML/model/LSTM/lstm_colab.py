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
from sklearn.decomposition import PCA
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

def pca(df_returns_path, n_components):

    df_returns = read_filepath(df_returns_path)
    pca = PCA(n_components=n_components)
    pca.fit(df_returns.values)
    logging.info(f"Fraction of variance preserved: {pca.explained_variance_ratio_.sum():.2f}")

    n_pca= pca.n_components_ # get number of components
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pca)] # get the index of the most important feature on EACH component
    initial_feature_names = df_returns.columns
    most_important_companies = [initial_feature_names[most_important[i]] for i in range(n_pca)] # get the most important feature names
    most_important_companies = list(set(most_important_companies))

    return most_important_companies

def LSTM_model(nodes,optimizer, drop_out=0.2):
    '''
    Architeture for the LSTM algorithm

    Parameters
    ----------
    nodes : list
        List that contains one number of nodes for each layer the user want to use.
    optimizer : str
        Optimizier between RMS_prop or Adam.
    drop_out : float, optional
        Value of the dropout in all the dropout layers. The default is 0.2.

    Returns
    -------
    model : tensorflow.python.keras.engine.sequential.Sequential
        Model.

    '''

    model = Sequential()
    model.add(Input(shape= (240, 1)))
    model.add(Dropout(drop_out))

    if len(nodes) > 1:
        ret_seq = True
    else:
        ret_seq = False

    for nod in nodes:
        model.add(LSTM(nod, return_sequences=ret_seq))
        model.add(Dropout(drop_out))

    model.add(Dense(1, activation='sigmoid'))

    # Two optimiziers used during training
    if optimizer == 'RMS_prop':
        opt = RMSprop(learning_rate=0.005, momentum=0.5, clipvalue=0.5)
    if optimizer == 'Adam':
        opt = Adam(learning_rate=0.005)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creation of input and output data for lstm classification problem')
    parser.add_argument('returns_file', type=str, help='Path of returns data')
    parser.add_argument('binary_file', type=str, help='Path of target data')
    parser.add_argument("-log", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument('num_periods', type=int, help='Number of periods you want to train')
    parser.add_argument('nodes',type=int, nargs='+', help='Choose the number of nodes in LSTM+Dropout layers')
    parser.add_argument('model_name', type=str, help='Choose the name of the model')
    parser.add_argument('-prin_comp_anal', type=bool, default=False, help='Use the most important companies obtained by a PCA decomposition on the first 250 PCs')
    parser.add_argument('-recursive', type=bool, default=True, help='Choose whether or not to pass parameters from one period to another during training')
    parser.add_argument('-optimizer', type=str, default='RMS_prop', help='Choose RMS_prop or Adam')

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
    if args.prin_comp_anal:
        logging.info("Using the most important companies obtained from a PCA decomposition")
        most_imp_comp = pca(args.returns_file, n_components=250)
        df_returns = df_returns[most_imp_comp]
        df_binary = df_binary[most_imp_comp]
    else:
        pass


    for i in range(args.num_periods):
        logging.info(f'============ Start Period {i}th ===========')
        if (i!=0) and (args.recursive):
            logging.info('LOADING PREVIOUS MODEL')
            model = load_model(f"LSTM_{i-1}_period.h5")
        else:
            logging.info('CREATING NEW MODEL')
            model = LSTM_model(args.nodes, args.optimizer)
        logging.info(model.summary())
        X_train, y_train, X_test, y_test = all_data_LSTM(df_returns, df_binary, i)
        es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        mc = ModelCheckpoint(f'{args.model_name}_period{i}.h5', monitor='val_loss', mode='min', verbose=0)
        history = model.fit(X_train, y_train, epochs=args.num_epochs, batch_size=args.batch_size,
                            callbacks=[es,mc], validation_split=0.2, shuffle=False, verbose=1)

        plt.figure(f'Loss and Accuracy Period {i}', figsize=[20.0,10.0])
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epochs')
        plt.title('Training and Validation Losses vs Epochs')
        plt.grid()
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epochs')
        plt.title('Training and Validation Accuracies vs Epochs')
        plt.grid()
        plt.legend()
        plt.savefig(f'accuracies_{i}.png')

        y_pred = model.predict(X_test)
        y_pred_companies = [y_pred[i:87+i] for i in range(0,len(y_pred)-87+1,87)]
        dict_comp = {df_returns.columns[i]: y_pred_companies[i] for i in range(0,365)}
        df_predictions = pd.DataFrame()
        for tick in df_returns.columns:
            df_predictions[tick] = dict_comp[tick][:,0]
        df_predictions.to_csv(f'Predictions_{i}th_Period.csv')


        logging.info(f'============ End Period {i}th ===========')



