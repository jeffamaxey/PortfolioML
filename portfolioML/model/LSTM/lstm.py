"""LSTM model"""
import numpy as np
import pandas as pd
import logging
import argparse
from keras.layers import Input, Dense, LSTM, Dropout
from keras.models import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop, Adam
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath("..")))
from model.split import split_Tperiod, get_train_set
from data.data_returns import read_filepath
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from makedir import smart_makedir, go_up


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

def LSTM_model(num_units=1, drop_out=0.2):
    inputs = Input(shape= (240, 1))
    drop = Dropout(drop_out)(inputs)
    hidden = LSTM(num_units, return_sequences=False)(drop)
    drop = Dropout(drop_out)(hidden)
    outputs = Dense(1, activation='sigmoid')(drop)

    model = Model(inputs=inputs, outputs=outputs)

    # Two optimiziers used during training
    rms_prop = RMSprop(learning_rate=0.005, momentum=0.5, clipvalue=0.5)
    adam = Adam(learning_rate=0.005)

    model.compile(loss='binary_crossentropy', optimizer=rms_prop, metrics=['accuracy'])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creation of input and output data for lstm classification problem')
    parser.add_argument("-log", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument('num_periods', type=int, help='Number of periods you want to train')
    parser.add_argument('model_name', type=str, help='Choose the name of the model')

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
    df_returns_path = go_up(2) + "/data/ReturnsData.csv"
    df_binary_path = go_up(2) + "/data/ReturnsBinary.csv"
    df_returns = read_filepath(df_returns_path)
    df_binary = read_filepath(df_binary_path)
    # Pass or not the weights from one period to another
    recursive = True

    smart_makedir(args.model_name)
    losses = smart_makedir(args.model_name + "/losses")
    accuracies = smart_makedir(args.model_name + "/accuracies")

    for i in range(args.num_periods):
        logging.info(f'============ Start Period {i}th ===========')
        if (i!=0) and (recursive):
            logging.info('LOADING PREVIOUS MODEL')
            model = load_model(f"{args.model_name}/{args.model_name}_period{i-1}.h5")
        else:
            logging.info('CREATING NEW MODEL')
            model = LSTM_model()
        logging.info(model.summary())
        X_train, y_train, X_test, y_test = all_data_LSTM(df_returns, df_binary, i)
        es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        mc = ModelCheckpoint(f'{args.model_name}/{args.model_name}_period{i}.h5', monitor='val_loss', mode='min', verbose=0)
        history = model.fit(X_train, y_train, epochs=1, batch_size=896,
                            callbacks=[es,mc], validation_split=0.2, shuffle=False, verbose=1)

        plt.figure(f'Period {i} Losses')
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.getcwd() + f'/{args.model_name}/losses/losses_{i}.png')

        plt.figure(f'Period {i} Accuracies')
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.getcwd() + f'/{args.model_name}/accuracies/accuracies_{i}.png')


        logging.info(f'============ End Period {i}th ===========')



