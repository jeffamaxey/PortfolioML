""" Autoencoder CNN"""
import argparse
import logging
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from portfolioML.model.split import all_data_LSTM
from portfolioML.makedir import smart_makedir
from portfolioML.data.preprocessing import pca

def CNN_autoencoders(df_returns, df_binary, period):
    """
    """
    x_train, y_train, x_test, y_test = all_data_LSTM(df_returns, df_binary, period)
    print(x_train.shape)
    #Model
    inputs = Input(shape=(x_train.shape))
    conv = Conv2D(12, kernel_size=(1,5), strides=(1,1), padding="same", activation="relu")(inputs)
    maxpool = MaxPooling2D((1,3), padding="same")(conv)
    conv = Conv2D(1, kernel_size=(1,5), strides=(1,1), padding="same", activation="relu")(maxpool)
    encoded  = MaxPooling2D((1,2), padding="same")(conv)
    encoder = Model(inputs, encoded)

    conv = Conv2D(1, kernel_size=(1,5), strides=(1,1), padding="same", activation="relu")(encoded)
    upsamp =  UpSampling2D((1,2))(conv)
    conv = Conv2D(12, kernel_size=(1,5), strides=(1,1), padding="same", activation="relu")(upsamp)
    upsamp = UpSampling2D((1,3))(conv)
    decoded = Conv2D(1, kernel_size=(1,2), strides=1, padding="same", activation="relu")(upsamp)


    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    print(autoencoder.summary())
    history = autoencoder.fit(x_train, x_train,
                             epochs=10,
                             batch_size=1024,
                             validation_split=0.2)

    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])



    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Creation of encoder for data reduction')
    requiredNamed = parser.add_argument_group('Required named arguments')
    requiredNamed.add_argument(
        '-r', '--returns_file', type=str, help='Path to the returns data')
    requiredNamed.add_argument(
        '-b', '--binary_file', type=str, help='Path to the binary target data')
    requiredNamed.add_argument('-p', '--period', type=int, help='Study period')
    # parser.add_argument('-bn', '--botneck', type=int, default=31,
    #                     help='Number of nodes in the middle layer (default=31)')
    parser.add_argument("-log", "--log", default="error",
                        help=("Provide logging level. Example --log debug, default=info"))
    args = parser.parse_args()

    levels = {'critical': logging.CRITICAL,
            'error': logging.ERROR,
            'warning': logging.WARNING,
            'info': logging.INFO,
            'debug': logging.DEBUG}

    logging.basicConfig(level=levels[args.log])
    pd.options.mode.chained_assignment = None

    df_return = pd.read_csv(args.returns_file)
    df_binary = pd.read_csv(args.binary_file)

    CNN_autoencoders(df_return, df_binary, args.period)
    plt.show()