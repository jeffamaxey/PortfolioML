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
from split import split_Tperiod, get_train_set
from portfolioML.data.data_returns import read_filepath
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


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

def LSTM_model(num_units=25):
    inputs = Input(shape= (240, 1))
    drop = Dropout(0.2)(inputs)
    hidden = LSTM(num_units, return_sequences=False)(drop)
    drop = Dropout(0.2)(hidden)
    outputs = Dense(1, activation='sigmoid')(drop)

    model = Model(inputs=inputs, outputs=outputs)
    # RMSprop is an adaptive learning rate algorithm
    rms_prop = RMSprop(learning_rate=0.005, momentum=0.5, clipvalue=0.5)
    # Adam derives from "adaptive momentum". It can be seen as a variant of RMSprop
    adam = Adam(learning_rate=0.005)
    model.compile(loss='binary_crossentropy', optimizer=rms_prop, metrics=['accuracy'])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creation of input and output data for lstm classification problem')
    parser.add_argument('returns_file', type=str, help='Path to the returns input data')
    parser.add_argument('binary_file', type=str, help='Path to the binary target data')
    parser.add_argument("-log", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument('num_units', type=int, help='Number of units in the LSTM layer')
    parser.add_argument('num_periods', type=int, help='Number of periods you want to train')
    parser.add_argument('num_epochs', type=int, help='Number of epochs you want to train')
    parser.add_argument('batch_size', type=int, help='Batch_size')
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
    df_returns = read_filepath(args.returns_file)
    df_binary = read_filepath(args.binary_file)
    recursive = True



    for i in range(0,args.num_periods):
        logging.info(f'============ Start Period {i}th ===========')
        if (i!=0) and recursive:
            model = load_model(f"LSTM_{i-1}_period.h5")
        else:
            model = LSTM_model(args.num_units)
        logging.info(model.summary())
        X_train, y_train, X_test, y_test = all_data_LSTM(df_returns, df_binary, i)
        es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        mc = ModelCheckpoint(f'LSTM_{i}_period.h5', monitor='val_loss', mode='min', verbose=0)
        history = model.fit(X_train, y_train, epochs=args.num_epochs, batch_size=args.batch_size,
                            callbacks=[es,mc], validation_split=0.2, shuffle=False, verbose=1)
        # model.save(f"LSTM_{i}_period.h5")

        plt.figure(f'Period {i} Losses')
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'losses_{i}.png')

        plt.figure(f'Period {i} Accuracies')
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'accuracies_{i}.png')

        y_pred = model.predict(X_test)
        y_pred_companies = [y_pred[i:87+i] for i in range(0,len(y_pred)-87+1,87)]
        dict_comp = {df_returns.columns[i]: y_pred_companies[i] for i in range(0,365)}
        df_predictions = pd.DataFrame()
        for tick in df_returns.columns:
            df_predictions[tick] = dict_comp[tick][:,0]
        df_predictions.to_csv(f'Predictions_{i}th_Period.csv')

        logging.info(f'============ End Period {i}th ===========')



