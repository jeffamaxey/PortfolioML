"""LSTM model"""
import numpy as np
import pandas as pd
import logging
import argparse
from keras.layers import Input, Dense, LSTM, Dropout
from keras.models import Model, Sequential
from keras.models import load_model
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath("..")))
from split import split_sequences, split_Tperiod, get_train_set
from portfolioML.data.data_returns import read_filepath
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

    #Read the data
    df_returns = read_filepath(args.returns_file)
    df_binary = read_filepath(args.binary_file)

    def all_data_LSTM(df_returns, df_binary, period, len_train=981, len_test=327):
        scaler = StandardScaler()
        periods_returns, periods_binary = split_Tperiod(df_returns, df_binary)

        T1_input = periods_returns[period]
        T1_target = periods_binary[period]

        X_input_train, y_input_train = T1_input[:len_train], T1_target[:len_train]

        withoutDays = X_input_train.drop(['Days'], axis=1)
        x_tran = withoutDays.values
        scaled = scaler.fit_transform(x_tran)
        X_input_tr = pd.DataFrame(scaled, index=None)
        d = withoutDays.columns
        X_input_tr.columns = d


        X_test, y_test = T1_input[len_test:], T1_target[len_test:]

        withoutDay = X_test.drop(['Days'], axis=1)
        x_t = withoutDay.values
        scal = scaler.fit_transform(x_t)
        X_tests = pd.DataFrame(scal, index=None)
        f = withoutDay.columns
        X_tests.columns = d

        X_train, y_train = get_train_set(X_input_tr, y_input_train)
        X_train, y_train = np.array(X_train), np.array(y_train)

        X_test, y_test = get_train_set(X_tests, y_test)
        X_test, y_test = np.array(X_test), np.array(y_test)

        # scaler = StandardScaler()
        # X_train_norm = scaler.fit_transform(X_train)
        # X_train = np.reshape(X_train_norm, (X_train_norm.shape[0], X_train_norm.shape[1], 1))

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        return X_train, y_train, X_test, y_test

    def LSTM_model():
        inputs = Input(shape= (240, 1))
        hidden = LSTM(25)(inputs)
        drop = Dropout(0.1)(hidden)
        outputs = Dense(2, activation='softmax')(drop)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    model = LSTM_model()
    print(model.summary())

    #Modello per il primo periodo
    X_train, y_train, X_test, y_test = all_data_LSTM(df_returns, df_binary, 0)
    history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, verbose=1)

    # for i in range(0,3):
    #     model = load_model(f"LSTM_{i}_period.h5")
    #     X_train, y_train, X_test, y_test = all_data_LSTM(df_returns, df_binary, i+1)
    #     history = model.fit(X_train, y_train, epochs=1, batch_size=128, validation_split=0.2, verbose=1)
    #     plt.figure()
    #     plt.plot(history.history['loss'], label='loss')
    #     plt.plot(history.history['val_loss'], label='val_loss')
    #     plt.xlabel('Epochs')
    #     plt.legend()
    #     plt.title(f'Period {i}')
    #     plt.grid(True)
    #     # #Prediction
    #     # y_pred = model.predict(X_test)
    #     # for i,j in zip(y_test,y_pred):
    #     #     logging.info(i,j)
    #     model.save(f"LSTM_{i+1}_period.h5")



