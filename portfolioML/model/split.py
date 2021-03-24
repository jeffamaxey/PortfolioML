"""Split time-series in traning and test(trading) for classification """
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input, Dense 
from keras.models import Model


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
    X: numpy array
        Arrey of the input set, its shape is (len(sequences)-n_steps, n_steps) 

    y: numpy array
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

    print(len(X_train))
    