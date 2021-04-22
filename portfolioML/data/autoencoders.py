'''
mll
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
from portfolioML.data.data_returns import read_filepath
from portfolioML.model.split import all_data_LSTM
from portfolioML.makedir import smart_makedir
from portfolioML.data.preprocessing import pca

def autoencoder(df_returns, df_binary, period, bottneck):
    '''
    lll
    '''
    smart_makedir(f'Autoencoder_for_period_{period}')

    x_train, y_train, x_test, y_test = all_data_LSTM(df_returns, df_binary, period)
    x_train = np.reshape(x_train, (len(x_train),1,240))

    # Autoencoder
    input_img = Input(shape=(240,))
    encoded = Dense(150, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_img)
    encoded = Dense(80, activation='relu', activity_regularizer=regularizers.l1(10e-5))(encoded)
    bottleneck = Dense(bottneck)(encoded)
    decoder =  Dense(80, activation='relu')(bottleneck)
    decoded = Dense(150, activation='relu')(decoder)
    decoded = Dense(240, activation='linear')(decoded)

    start = time.time()

    lenght_x_train = len(x_train)
    after = []
    for i in range(len(x_train)):
        # Encoder
        encoder = Model(input_img, bottleneck)
        encoder.compile(optimizer='adam', loss='mse')
        # Autoencoder
        autoencod = Model(input_img, decoded)
        autoencod.compile(optimizer='adam', loss='mse')
        # Fit autoencoder
        history = autoencod.fit(x_train[i], x_train[i], epochs=40, batch_size=1, shuffle=False)
        # autoencod.summary()
        # Results autoencoder
        reconstructed_data = autoencod.predict(x_train[i])
        # Encoder information for feature extraction after dimensionality reduction
        afterbot = encoder.predict(x_train[i])
        after.append(afterbot[0])

        # Difference between original data and data from autoencoder
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
    df = pd.DataFrame(after, columns = [f'selected feature_{i}' for i in range(len(after[0]))])
    df.to_csv('after.csv')

    end = time.time()
    print(f'Total time: {end-start} seconds')
    # Save model structure of Autoencoder end encoder
    dot_img_autoencod = f'Autoencoder_for_period_{period}/autoencoder_nocompress_for_period_{period}.png'
    plot_model(autoencod, to_file=dot_img_autoencod, show_shapes=True)
    dot_img_encoder = f'Autoencoder_for_period_{period}/encoder_nocompress_for_period_{period}.png'
    plot_model(encoder, to_file=dot_img_encoder, show_shapes=True)
    encoder.save(f'Autoencoder_for_period_{period}/encoder_{period}.h5')
    # encoder = load_model('Autoencoder/encoder.h5')

    plt.figure()
    plt.bar(list(range(len(difference[0]))), difference[0])
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    return after


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creation of encoder for data reduction')
    requiredNamed = parser.add_argument_group('Required named arguments')
    requiredNamed.add_argument('-r', '--returns_file', type=str, help='Path to the returns data')
    requiredNamed.add_argument('-b', '--binary_file', type=str, help='Path to the binary target data')
    requiredNamed.add_argument('-p', '--period', type=int, help='Study period')
    parser.add_argument('-nc', '--n_components', type=int, default=250, help='Space dimension of the projected vector (default=250)')
    parser.add_argument('-bn', '--botneck', type=int, default=31, help='Number of nodes in the middle layer (default=31)')
    parser.add_argument("-log", "--log", default="error",
                        help=("Provide logging level. Example --log debug, default=info"))
    args = parser.parse_args()

    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}

    logging.basicConfig(level= levels[args.log])
    pd.options.mode.chained_assignment = None


    df_ret = args.returns_file
    df_bin = read_filepath(args.binary_file)
    per = args.period
    n_components = args.n_components
    botneck = args.botneck

    # Principal component analysis for new DataFrame with less companies
    tick = pca(df_ret, n_components)
    df_ret = pd.read_csv(df_ret, index_col=0)
    df_return = df_ret[tick]
    df_binar = df_bin[tick]

    autoencoder(df_return, df_binar, per, botneck)
    plt.show()
