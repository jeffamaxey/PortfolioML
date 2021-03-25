"""Split time-series in traning and test(trading) for classification """
import logging
import argparse
import numpy as np
import pandas as pd
import tqdm


def split_sequences(returns, targets, n_steps=240):
    """
    Returns the input sequences and target label for classification task.


    Parameters
    ----------
    returns: list, numpy array
        time-series data of returns to split.

    targets: list, numpy array
        time-series data of target to split. It musta have the same length of returns

    n_steps: integer(optional)
        number of time steps for each istance. Default = 100

    Results
    -------
    X: list
        Arrey of the input set, its shape is (len(sequences)-n_steps, n_steps)

    y: list
        Arrey of the input target, its shape is (len(sequences)-n_steps, 1)
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
    Return the train set for LSTM. 
    The argumets are the returns dataframe and the binary data frame for copute respectvely
    the X_train and the y_train for classification task

    Parameters
    ----------
    df_returns: pandas dataframe
        Dataframe of returns

    df_binary:
        Datframe of binary target associated to data returns. It has the same shape of df_returns

    Returns
    -------
    list_tot_X: numpy arrey
        Arrey of input data for LSTM

    list_tot_y: numpy arrey
        Arrey of input target class for LSTM
    """

    list_tot_X = []
    list_tot_y = []
    for comp in df_returns.columns[1:]:
        X_train, y_train = split_sequences(df_returns[comp], df_binary[comp])
        list_tot_X.append(X_train)
        list_tot_y.append(y_train)

    list_tot_X = np.array(list_tot_X)
    list_tot_y = np.array(list_tot_y)

    list_tot_X = np.reshape(list_tot_X,(list_tot_X.shape[0]*list_tot_X.shape[1],list_tot_X.shape[2]))
    list_tot_y = np.reshape(list_tot_y,(list_tot_y.shape[0]*list_tot_y.shape[1]))

    return list_tot_X, list_tot_y



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creation of input and output data for lstm classification problem')
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

    df_returns = pd.read_csv(args.returns_file)
    df_binary = pd.read_csv(args.binary_file)

    data = np.linspace(10,900,90)
    X_train, y_train = split_sequences(df_returns.AEP, df_binary.AEP)

    # for i,j in zip(X_train, y_train):
    #     print(i,j)

    list_totX, list_toty = list(), list()
    for comp in df_returns.columns[1:6]:
        X_train, y_train = split_sequences(df_returns[comp], df_binary[comp])
        list_totX.append(X_train)
        list_toty.append(y_train)

    list_toty = np.array(list_toty)
    print(list_toty.shape)
    
    # utile per fare il test della funzione get_train_set
    # a = list((list_tot[i] for i in range(list_tot.shape[0])))
    # a_list = np.vstack(a)
    # print(a_list.shape)
    
    # print(list_toty.shape)
    # list_tot = np.reshape(list_toty,(list_toty.shape[0]*list_toty.shape[1]))
    # print(list_tot.shape)
    # print(a_list == list_tot)
    