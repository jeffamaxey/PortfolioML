'''
Autoencoder
'''
import argparse
import logging
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from portfolioML.model.split import all_data_LSTM
from portfolioML.makedir import smart_makedir
from portfolioML.data.preprocessing import pca

def autoencoder(df_returns, df_binary, period, bottneck):
    """
    Creation of a csv file with features selected from autoencoder.

    Parameters
    ----------
    df_returns: TYPE = pandas dataframe.
        Input data frame.

    df_binary: TYPE = pandas dataframe.
        Input data frame of binary target.

    period: TYPE = integer.
        Study period.

    bottneck: TYPE = integer.
        Number of nodes in the middle layer of the autoencoder.

    Return
    ----------
    after: TYPE = pandas dataframe
        Csv file with selected features extract from encoder.
    """

    x_train, y_train, x_test, y_test = all_data_LSTM(
        df_returns, df_binary, period)
    x_train = np.reshape(x_train, (len(x_train), 1, 240))
    print(len(x_train))

    input_img = Input(shape=(240,))
    encoded = Dense(150, activation='relu',
                    activity_regularizer=regularizers.l1(10e-5))(input_img)
    encoded = Dense(80, activation='relu',
                    activity_regularizer=regularizers.l1(10e-5))(encoded)
    bottleneck = Dense(bottneck)(encoded)
    decoder = Dense(80, activation='relu')(bottleneck)
    decoded = Dense(150, activation='relu')(decoder)
    decoded = Dense(240, activation='linear')(decoded)

    start = time.time()

    lenght_x_train = len(x_train)
    after = []
    for i in range(lenght_x_train):
        encoder = Model(input_img, bottleneck)
        encoder.compile(optimizer='adam', loss='mse')
        autoencod = Model(input_img, decoded)
        autoencod.compile(optimizer='adam', loss='mse')
        history = autoencod.fit(
            x_train[i], x_train[i], epochs=40, batch_size=1, shuffle=False)
        # autoencod.summary()
        reconstructed_data = autoencod.predict(x_train[i])
        afterbot = encoder.predict(x_train[i])
        after.append(afterbot[0])

        difference = x_train[i] - reconstructed_data

        # plt.figure(f'selected feature {i}')
        # plt.subplot(131)
        # plt.bar(list(range(len(x_train[0][0]))), x_train[0][0], color='b', alpha=0.5)
        # plt.bar(list(range(len(reconstructed_data[0]))), reconstructed_data[0], color='r', alpha=0.5)
        # plt.subplot(132)
        # plt.bar(list(range(len(difference[0]))),difference[0], color='orange', alpha=0.5)
        # plt.subplot(133)
        # plt.bar(list(range(len(afterbot[0]))), afterbot[0], color='green')

    after = np.array(after)
    df = pd.DataFrame(
        after, columns=[f'selected feature_{i}' for i in range(len(after[0]))])
    df.to_csv('encoder_train.csv', index=False)

    end = time.time()
    print(f'Total time: {end-start} seconds')
    # dot_img_autoencod = f'Autoencoder_for_period_{period}/autoencoder_nocompress_for_period_{period}.png'
    # plot_model(autoencod, to_file=dot_img_autoencod, show_shapes=True)
    # dot_img_encoder = f'Autoencoder_for_period_{period}/encoder_nocompress_for_period_{period}.png'
    # plot_model(encoder, to_file=dot_img_encoder, show_shapes=True)
    # encoder.save(f'Autoencoder_for_period_{period}/encoder_{period}.h5')
    # encoder = load_model('Autoencoder/encoder.h5')

    # plt.figure()
    # plt.bar(list(range(len(difference[0]))), difference[0])
    # plt.figure()
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend()
    # plt.show()

    return after


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Creation of encoder for data reduction and features selection.')
    requiredNamed = parser.add_argument_group('Required named arguments')
    requiredNamed.add_argument(
        '-r', '--returns_file', type=str, help='Path to the returns data.')
    requiredNamed.add_argument(
        '-b', '--binary_file', type=str, help='Path to the binary target data.')
    requiredNamed.add_argument('-p', '--period', type=int, help='Study period.')
    parser.add_argument('-bn', '--botneck', type=int, default=31,
                        help='Number of nodes in the middle layer (default=31).')
    parser.add_argument("-log", "--log", default="error",
                        help=("Provide logging level. Example --log debug, default=info."))
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
    per = args.period
    botneck = args.botneck

    autoencoder(df_return, df_binary, per, botneck)
