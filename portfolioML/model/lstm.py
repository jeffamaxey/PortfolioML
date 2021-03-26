"""LSTM model"""
import numpy as np
import pandas as pd
import logging
import argparse
from keras.layers import Input, Dense, LSTM, Dropout
from keras.models import Model, Sequential
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath("..")))
from split import split_sequences, split_Tperiod
from portfolioML.data.data_returns import read_filepath
from sklearn.preprocessing import RobustScaler, StandardScaler
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

    #Split data. We choose 2/3 of the period for train and 1/3 for test
    len_train = 981
    len_test = 327

    periods_returns, periods_binary = split_Tperiod(df_returns, df_binary)

    T1_input = periods_returns[0]
    T1_target = periods_binary[0]

    X_input_train = T1_input[:len_train]
    y_input_train = T1_target[:len_train]

    X_test = T1_input[len_test:]
    y_test = T1_target[len_test:]

    X_train, y_train = split_sequences(X_input_train, y_input_train)

    #Normalization and reshaping
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_train = np.reshape(X_train_norm, (X_train_norm.shape[0], X_train_norm.shape[1], 1))

    #Model1
    inputs = Input(shape= (240, 1))
    hidden = LSTM(100)(inputs)
    drop = Dropout(0.1)(hidden)
    outputs = Dense(2, activation='softmax')(drop)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    #Fit
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

    #Prediction
    y_pred = model.predict(X_test)
    for i,j in zip(y_test,y_pred):
        print(i,j)


    #Loss and Val-loss
    print('loss sul traing in funzione delle epoche')
    print(history.history['loss'])
    print('\t')
    print('loss sul validation in funzione delle epoche')
    print(history.history['val_loss'])
    print('\t')

    plt.plot(history.history['loss']) #funzione a gomito dalla quale si deve scegliere il valore ottimale delle epoche
    plt.plot(history.history['val_loss'])
    plt.show()



