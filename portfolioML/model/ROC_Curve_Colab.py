#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.models import load_model
from sklearn.metrics import roc_curve, auc, roc_auc_score
import sys
import os
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Creation of input and output data for lstm classification problem')
    parser.add_argument("-log", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument('num_periods', type=int, help='Number of periods you want to train')
    args = parser.parse_args()

    df_returns = read_filepath('portfolioML/data/ReturnsData.csv')
    df_binary = read_filepath('portfolioML/data/ReturnsBinary.csv')

    for i in range(0,args.num_periods):
        X_train, y_train, X_test, y_test = all_data_LSTM(df_returns, df_binary, i)
        model1 = load_model(f'portfolioML/model/Trained_Models/Modello1/1/LSTM_{i}_period.h5')
        model2 = load_model(f'portfolioML/model/Trained_Models/Modello2/1/LSTM_{i}_period.h5')
        model3 = load_model(f'portfolioML/model/Trained_Models/Modello3/1/LSTM_{i}_period.h5')
        model4 = load_model(f'portfolioML/model/Trained_Models/Modello4/1/LSTM_{i}_period.h5')
        model5 = load_model(f'portfolioML/model/Trained_Models/Modello5/1/LSTM_{i}_period.h5')
        model6 = load_model(f'portfolioML/model/Trained_Models/Modello6/1/LSTM_{i}_period.h5')
        model7 = load_model(f'portfolioML/model/Trained_Models/Modello7/1/LSTM_{i}_period.h5')

        probas_1 = model1.predict(X_test)
        probas_2 = model2.predict(X_test)
        probas_3 = model3.predict(X_test)
        probas_4 = model4.predict(X_test)
        probas_5 = model5.predict(X_test)
        probas_6 = model6.predict(X_test)
        probas_7 = model7.predict(X_test)

        fpr1, tpr1, thresholds1 = roc_curve(y_test, probas_1[:, 0])
        fpr2, tpr2, thresholds2 = roc_curve(y_test, probas_2[:, 0])
        fpr3, tpr3, thresholds3 = roc_curve(y_test, probas_3[:, 0])
        fpr4, tpr4, thresholds4 = roc_curve(y_test, probas_4[:, 0])
        fpr5, tpr5, thresholds5 = roc_curve(y_test, probas_5[:, 0])
        fpr6, tpr6, thresholds6 = roc_curve(y_test, probas_6[:, 0])
        fpr7, tpr7, thresholds7 = roc_curve(y_test, probas_7[:, 0])

        roc_auc1 = roc_auc_score(y_test, probas_1[:,0], average=None)
        roc_auc2 = roc_auc_score(y_test, probas_2[:,0], average=None)
        roc_auc3 = roc_auc_score(y_test, probas_3[:,0], average=None)
        roc_auc4 = roc_auc_score(y_test, probas_4[:,0], average=None)
        roc_auc5 = roc_auc_score(y_test, probas_5[:,0], average=None)
        roc_auc6 = roc_auc_score(y_test, probas_6[:,0], average=None)
        roc_auc7 = roc_auc_score(y_test, probas_7[:,0], average=None)

        plt.figure(f'ROC CURVE PERIOD {i}')
        plt.plot(fpr1, tpr1, label='Model 1 (area = %0.4f)' % (roc_auc1))
        plt.plot(fpr2, tpr2, label='Model 2 (area = %0.4f)' % (roc_auc2))
        plt.plot(fpr3, tpr3, label='Model 3 (area = %0.4f)' % (roc_auc3))
        plt.plot(fpr4, tpr4, label='Model 4 (area = %0.4f)' % (roc_auc4))
        plt.plot(fpr5, tpr5, label='Model 5 (area = %0.4f)' % (roc_auc5))
        plt.plot(fpr6, tpr6, label='Model 6 (area = %0.4f)' % (roc_auc6))
        plt.plot(fpr7, tpr7, label='Model 7 (area = %0.4f)' % (roc_auc7))
        plt.plot([0, 1], [0, 1], 'k--')

        plt.xlabel('False Positive Rate',)
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right", fontsize=12, frameon=False)

        plt.savefig(f'ROC_Curve_Period_{i}.png')
