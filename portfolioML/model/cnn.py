""" CNN model """
import numpy as np
import pandas as pd
import logging
import argparse
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Dropout, Conv1D, MaxPool1D, Flatten
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath("..")))
from split import split_Tperiod, get_train_set
from portfolioML.data.data_returns import read_filepath
from dnn import all_data_LSTM

def CNN_model():
    """
    CNN model for classification task
    """
    model = Sequential()
    model.add(Input(shape=(240,1)))
    model.add(Dropout(0.1))
    model.add(Conv1D(15, kernel_size=(20), strides=5, activation='tanh'))
    model.add(MaxPool1D(pool_size=5, strides=1, padding='valid'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(25, activation= 'tanh' ))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation= 'sigmoid' ))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make CNN model for classification task to predict class label 0 or 1')
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


    for per in range(0,10):
        model = CNN_model()
        #Training
        X_train, y_train, X_test, y_test = all_data_LSTM(df_returns, df_binary, per)
        #Trainng
        es = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
        mc = ModelCheckpoint(f'CNN_dense+_period{per}.h5', monitor='val_loss', mode='min', verbose=0)
        history = model.fit(X_train ,y_train, callbacks=[es,mc],validation_split=0.2, batch_size=512, epochs=200, verbose=1)

        #Elbow curve
        plt.figure(f'Loss and Accuracy period {per}')
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], label='train_loss') 
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epochs')
        plt.title('Training and Validation Loss vs Epochs')
        plt.grid()
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epochs')
        plt.title('Training and Validation Accuracy vs Epochs')
        plt.grid()
        plt.legend()

    plt.show()

    

