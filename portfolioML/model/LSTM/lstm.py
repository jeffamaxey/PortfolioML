"""LSTM model"""
import numpy as np
import pandas as pd
import logging
import argparse
from keras.layers import Input, Dense, LSTM, Dropout
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop, Adam
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath("..")))
from model.split import split_Tperiod, get_train_set, all_data_LSTM
from data.data_returns import read_filepath
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from makedir import smart_makedir, go_up

def LSTM_model(nodes,optimizer, drop_out=0.2):
  
    model = Sequential()
    model.add(Input(shape= (240, 1)))
    model.add(Dropout(drop_out))

    if len(nodes) > 1:
        ret_seq = True
    else:
        ret_seq = False

    for nod in nodes:
        model.add(LSTM(nod, return_sequences=ret_seq))
        model.add(Dropout(drop_out))

    model.add(Dense(1, activation='sigmoid'))

    # Two optimiziers used during training
    if optimizer == 'RMS_prop':
        opt = RMSprop(learning_rate=0.005, momentum=0.5, clipvalue=0.5)
    if optimizer == 'Adam':
        opt = Adam(learning_rate=0.005)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creation of input and output data for lstm classification problem')
    parser.add_argument("-log", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument('num_periods', type=int, help='Number of periods you want to train')
    parser.add_argument('model_name', type=str, help='Choose the name of the model')
    parser.add_argument('nodes',type=int, nargs='+', help='Choose the number of LSTM+Dropout layers')
    parser.add_argument('-optimizier', type=str, default='RMS_prop', help='Choose RMS_prop or Adam')


    args = parser.parse_args()

    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}

    logging.basicConfig(level= levels[args.log])
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
            model = LSTM_model(args.nodes, args.optimizier)
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



