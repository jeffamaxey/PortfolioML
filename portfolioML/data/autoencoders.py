'''
Autoencoder for features extraction
N.B. this algorithm takes a lot of time (20 hours) just only for one period.
It was not worth it to implement for every periods of study (17 periods).

'''
import argparse
import logging
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.utils import plot_model
from portfolioML.model.split import all_data_LSTM
from portfolioML.makedir import smart_makedir

def autoencoder(df_returns, df_binary, period, bottneck, save=True, plot=False):
    """
    Creation of a csv file with features selected from autoencoder.

    An autoencoder is a neural network that is trained to attempt to copy its input to its output.
    Usually they are restricted in a ways that allow them to copy only approximately
    and so the model is forced to prioritize which aspects of the input should be copied.
    One way to obtain useful features from the autoencoder is to constrain the middle layer
    to have smaller dimension than the input layer.
    An autoencoder whit this property is called undercomplete.
    The learning process is performed minimizing a loss function (like mean squared error).

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

    smart_makedir(f'Autoencoder/autoencoder_period_{period}/')

    x_train, y_train, x_test, y_test = all_data_LSTM(
        df_returns, df_binary, period)
    x_train = np.reshape(x_train, (len(x_train), 1, 240))
    print(len(x_train))

    input_img = Input(shape=(240,), name='Input')
    encoded = Dense(150, activation='relu',
                    activity_regularizer=regularizers.l1(10e-5), name='dense1')(input_img)
    encoded = Dense(80, activation='relu',
                    activity_regularizer=regularizers.l1(10e-5), name='dense2')(encoded)
    bottleneck = Dense(bottneck, name='MiddleLayer')(encoded)
    decoder = Dense(80, activation='relu', name='dense3')(bottleneck)
    decoded = Dense(150, activation='relu', name='dense4')(decoder)
    decoded = Dense(240, activation='linear', name='dense5')(decoded)

    start = time.time()
    after = []
    for i in range(5,6):
        encoder = Model(input_img, bottleneck)
        encoder.compile(optimizer='adam', loss='mse')
        autoencod = Model(input_img, decoded)
        autoencod.compile(optimizer='adam', loss='mse')

        history = autoencod.fit(x_train[i], x_train[i], epochs=40, batch_size=1, shuffle=False)
        autoencod.summary()
        reconstructed_data = autoencod.predict(x_train[i])
        afterbot = encoder.predict(x_train[i])
        after.append(afterbot[0])

        difference = x_train[i] - reconstructed_data

        if plot:
            plt.figure(f'selected feature {i}', figsize=[18, 5])
            plt.subplot(131)
            plt.bar(list(range(len(x_train[0][0]))), x_train[0][0], color='darkgreen')
            plt.xlabel('Features')
            plt.title('Original sequence')
            plt.subplot(132)
            plt.bar(list(range(len(afterbot[0]))), afterbot[0], color='cornflowerblue', alpha=0.8)
            plt.title('Encoded sequence')
            plt.xlabel('Features')
            plt.subplot(133)
            plt.bar(list(range(len(reconstructed_data[0]))), reconstructed_data[0],\
                                    color='forestgreen', label='reconstructed')
            plt.bar(list(range(len(difference[0]))),difference[0], color='crimson', label='differences')
            plt.title('Reconstructed sequence')
            plt.xlabel('Features')
            plt.legend()
            plt.savefig('autoencoder.png')
    
    after = np.array(after)
    df = pd.DataFrame(
        after, columns=[f'selected feature_{i}' for i in range(len(after[0]))])
    df.to_csv(f'Autoencoder/autoencoder_period_{period}/encoder_train.csv', index=False)

    end = time.time()
    print(f'Total time: {end-start} seconds')

    dot_img_auto = f'Autoencoder/autoencoder_period_{period}/autoencoder_model_train.png'
    plot_model(autoencod, to_file=dot_img_auto, show_shapes=True)
    dot_img_enco = f'Autoencoder/autoencoder_period_{period}/encoder_model_train.png'
    plot_model(encoder, to_file=dot_img_enco, show_shapes=True)

    if save:
        autoencod.save(f'Autoencoder/autoencoder_period_{period}/autoencoder_train.h5')
        encoder.save(f'Autoencoder/autoencoder_period_{period}/encoder_train.h5')
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    description='Creation of encoder for data reduction and features selection.')
    requiredNamed = parser.add_argument_group('Required named arguments')
    requiredNamed.add_argument('-p', '--period', type=int,
                        help='Study period.')
    requiredNamed.add_argument('-l', '--load', action='store_true',
                        help='Type -l to load the encoder.')
    requiredNamed.add_argument('-c', '--create', action='store_true',
                        help='Type -c to create the encoder.')
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

    df_ret = pd.read_csv('ReturnsDataPCA.csv')
    df_bin = pd.read_csv('ReturnsBinaryPCA.csv')
    per = args.period
    botneck = args.botneck

    loaded = args.load
    created = args.create

    if loaded:
        encoder = load_model(f'Autoencoder/autoencoder_period_{per}/encoder_train.h5')
        x_train, y_train, x_test, y_test = all_data_LSTM(df_ret, df_bin, per)
        x_train = np.reshape(x_train, (len(x_train), 1, 240))
        after = []
        for i in range(len(x_train)):
            afterbot = encoder.predict(x_train[i])
            after.append(afterbot[0])
        after = np.array(after)
        df = pd.DataFrame(
        after, columns=[f'selected feature_{i}' for i in range(len(after[0]))])
        df.to_csv('encoder_train123.csv', index=False)

    if created:
        autoencoder(df_ret, df_bin, per, botneck)
