'''
mll
'''
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from portfolioML.data.data_returns import read_filepath
from portfolioML.model.split import all_data_DNN, all_data_LSTM
from portfolioML.makedir import smart_makedir

def autoencoder(df_returns, df_binary, period):
    '''
    lll
    '''
    smart_makedir('Autoencoder')

    x_train, y_train, x_test, y_test = all_data_LSTM(df_returns, df_binary, period)

    n_inputs = x_train[0,:].shape
    n_inputs = n_inputs[0]
    x_train = np.reshape(x_train[1], (1, len(x_train[0])))

    n_bottleneck_down = round(float(n_inputs) / 4.0)
    n_bottleneck_up = round(float(n_inputs) * 4.0)

    input_img = Input(shape=(n_inputs,))
    encoded = Dense(150, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_img)
    bottleneck = Dense(31)(encoded)
    decoded = Dense(150, activation='relu')(bottleneck)
    decoded = Dense(n_inputs, activation='linear')(decoded)

    encoder = Model(input_img, bottleneck)
    encoder.compile(optimizer='adam', loss='mse')

    autoencod = Model(input_img, decoded)
    autoencod.compile(optimizer='adam', loss='mse')

    history = autoencod.fit(x_train, x_train, epochs=40, batch_size=1, shuffle=False)
    autoencod.summary()

    afterbot = encoder.predict(x_train)
    reconstructed = autoencod.predict(x_train)
    diff = x_train[0] - reconstructed[0]

    plt.figure()
    plt.bar(list(range(len(diff))), diff)

    plt.figure()
    plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    
    dot_img_autoencod = 'Autoencoder/autoencoder_no_compress.png'
    plot_model(autoencod, to_file=dot_img_autoencod, show_shapes=True)
    dot_img_encoder = 'Autoencoder/encoder_no_compress.png'
    plot_model(encoder, to_file=dot_img_encoder, show_shapes=True)
    encoder.save('Autoencoder/encoder.h5')
    # encoder = load_model('encoder.h5')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creation of input and output data')
    parser.add_argument('returns_file', type=str, help='Path to the returns input data')
    parser.add_argument('binary_file', type=str, help='Path to the binary target data')
    parser.add_argument('period', type=int, help='periodo di studio' )
    parser.add_argument("-log", "--log", default="error",
                        help=("Provide logging level. Example --log debug', default='info"))

    args = parser.parse_args()

    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}

    logging.basicConfig(level= levels[args.log])
    pd.options.mode.chained_assignment = None


    df_return = read_filepath(args.returns_file)
    df_binar = read_filepath(args.binary_file)
    perio = args.period

    autoencoder(df_return, df_binar, perio)
