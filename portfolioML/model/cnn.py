""" CNN model """
import numpy as np
import pandas as pd
import logging
import argparse
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Concatenate
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.vis_utils import plot_model
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath("..")))
from split import split_Tperiod, get_train_set
from portfolioML.data.data_returns import read_filepath
from dnn import all_data_LSTM

class MinPooling1D(MaxPooling1D):
    """
    Class for minimun pooling
    """

    def __init__(self, pool_size=2, strides=None, padding='valid', data_format='channels_last', **kwargs):
        super(MinPooling1D, self).__init__(pool_size, strides, padding, data_format, **kwargs)

    def pooling(self, x):
        """
        Compute minimun pooling
        """
        return -MaxPooling1D(self.pool_size, self.strides, self.padding, self.data_format)(-x)


def CNN_model():
    """
    CNN model for classification task
    """
    # model = Sequential()
    # model.add(Input(shape=(240,1)))
    # model.add(Dropout(0.1))
    # model.add(Conv1D(15, kernel_size=(20), strides=5, activation='tanh'))
    # model.add(MaxPooling1D(pool_size=5, strides=1, padding='valid'))
    # model.add(Dropout(0.4))
    # model.add(Flatten())
    # model.add(Dropout(0.4))
    # model.add(Dense(25, activation= 'tanh' ))
    # model.add(Dropout(0.4))
    # model.add(Dense(1, activation= 'sigmoid' ))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    inputs = Input(shape=(240,1))
    drop = Dropout(0.1)(inputs)
    conv = Conv1D(8, kernel_size=(20), strides=5, activation='tanh')(drop)
    min_pool = MinPooling1D(pool_size=5, strides=1, padding='valid').pooling(conv)
    max_pool = MaxPooling1D(pool_size=5, strides=1, padding='valid')(conv)
    merge = Concatenate()([min_pool, max_pool])
    drop = Dropout(0.4)(merge)
    flatten = Flatten()(drop)
    dense = Dense(25, activation= 'tanh')(flatten)
    outputs = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    plot_model(model, to_file='model_multihead.png', show_shapes=True, show_layer_names=True)
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
        mc = ModelCheckpoint(f'CNN_minpool_period{per}.h5', monitor='val_loss', mode='min', verbose=0)
        history = model.fit(X_train ,y_train, callbacks=[es,mc],validation_split=0.2, batch_size=512, epochs=400, verbose=1)

        #Elbow curve
        plt.figure(f'CNN_minpool_Loss and Accuracy period {per}')
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

    

