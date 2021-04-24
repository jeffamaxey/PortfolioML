"""Split time-series in traning and test(trading) for classification """
import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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

    # ho solo chiamato come unica variabile quella cosa che c'era nel for, il nome è da rivedere
    len_total_leave = len(df_returns) - len_period
    periods_ret = [(df_returns[i:len_period + i])
                   for i in range(0, len_total_leave + 1, len_test)]
    periods_bin = [(df_binary[i:len_period + i])
                   for i in range(0, len_total_leave + 1, len_test)]  # questa mancava

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

    X = [returns[i:i + n_steps] for i in range(len(returns) - n_steps)]
    y = [targets[i + n_steps] for i in range(len(targets) - n_steps)]

    return X, y


def get_train_set(df_returns1, df_binary1):
    """
    Return the train set for the LSTM.
    The argumets are the returns dataframe and the binary dataframe. The function compute respectively
    the X_train and the y_train for classification task, stacking sequences of different companies
    one over another.

    Parameters
    ----------
    df_returns1: pandas dataframe, numpy array
        Dataframe of returns

    df_binary1: pandas dataframe, numpy array
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

    if (str(type(df_returns1)) != "pandas.core.frame.DataFrame"):
        df_returns1 = pd.DataFrame(df_returns1)
    if (str(type(df_binary1)) != "pandas.core.frame.DataFrame"):
        df_binary1 = pd.DataFrame(df_binary1)

    for comp in df_returns1.columns:
        X_train, y_train = get_sequences(df_returns1[comp], df_binary1[comp])
        list_tot_X.append(X_train)
        list_tot_y.append(y_train)

    list_tot_X = np.array(list_tot_X)
    list_tot_y = np.array(list_tot_y)

    list_tot_X = np.reshape(
        list_tot_X, (list_tot_X.shape[0] * list_tot_X.shape[1], list_tot_X.shape[2]))
    list_tot_y = np.reshape(
        list_tot_y, (list_tot_y.shape[0] * list_tot_y.shape[1]))

    return list_tot_X, list_tot_y


def all_data_LSTM(df_returns, df_binary, period, len_train=981):
    """
    Function that create the right input for the LSTM algorithm.
    X_train and X_test are normalized and reshaped.

    Parameters
    ----------
    df_returns : pandas dataframe
        Pandas dataframe of returns.
    df_binary : pandas dataframe
        Pandas dataframe of returns.
    period : int
        Period over which you want to create the input for the LSTM.
    len_train : int, optional
        Lenght of the training set. The default is 981.

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


def all_multidata_LSTM(df_multidim_list, df_binary, period):
    '''
    When computing the DWT you get different approximations of the signal. This function
    returns the multidimensional data that will be used for LSTM and CNN.

    Parameters
    ----------
    df_multidim_list : list
        List of the multidimensional dataframe. Note, they are 3 and only 3.
    df_binary : pandas dataframe
        Pandas dataframe of returns.
    period : int
        Period over which you want to generate the data

    Returns
    ---------
    X_train, y_train, X_test, y_test : numpy array
        Data for LSTM or CNN models
    '''

    df_multireturns1, df_multireturns2, df_multireturns3 = df_multidim_list[
        0], df_multidim_list[1], df_multidim_list[2]

    X_train1, y_train, X_test1, y_test = all_data_LSTM(
        df_multireturns1, df_binary, period)
    X_train2, y_train, X_test2, y_test = all_data_LSTM(
        df_multireturns1, df_binary, period)
    X_train3, y_train, X_test3, y_test = all_data_LSTM(
        df_multireturns1, df_binary, period)
    X_train = np.stack((X_train1, X_train2, X_train3),
                       axis=-1).reshape(X_train1.shape[0], 240, 3)
    X_test = np.stack((X_test1, X_test2, X_test3), axis=-
                      1).reshape(X_test1.shape[0], 240, 3)
    return X_train, y_train, X_test, y_test


def all_data_DNN(df_returns, df_binary, period, len_train=981):
    """
    Create a right input data for DNN starting from the right data of LSTM.
    This function selects only m values (features) exctrated from the 240 values
    in the LSTM input data.

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

    Returns
    -------
    X_train : numpy array

    y_train : numpy array

    X_test : numpy array

    y_test : numpy array

    """

    X_train, y_train, X_test, y_test = all_data_LSTM(
        df_returns, df_binary, period)

    m = list(range(0, 240, 20)) + list(range(221, 240))
    X_train = X_train[:, m, :]
    X_train = np.reshape(X_train, (X_train.shape[0], 31))

    X_test = X_test[:, m, :]
    X_test = np.reshape(X_test, (X_test.shape[0], 31))

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Creation of input and output data for lstm classification problem')
    parser.add_argument('returns_file', type=str,
                        help='Path to the returns input data')
    parser.add_argument('binary_file', type=str,
                        help='Path to the binary target data')
    parser.add_argument("-log", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))

    args = parser.parse_args()

    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}

    logging.basicConfig(level=levels[args.log])

    df_returns = pd.read_csv(args.returns_file)
    df_binary = pd.read_csv(args.binary_file)

    X_train, y_train = get_sequences(df_returns, df_binary)
