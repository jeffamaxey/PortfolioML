"""LSTM model"""
import numpy as np
import pandas as pd 
import logging 
import argparse
from keras.layers import Input, Dense, LSTM
from keras.models import Model
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath("..")))
from split import split_sequences
from portfolioML.data.data_returns import read_filepath

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

    #Split data, only one company for the moment. We choose 2/3 of the period for train and 1/3 for test
    T1_input = df_returns.AEP[:1308]
    T1_target = df_binary.AEP[:1308]

    X_input_train = T1_input[:981]
    y_input_train = T1_target[:981]

    X_test = T1_input[982:]
    y_test = T1_target[982:]

    X_train, y_train = split_sequences(X_input_train, y_input_train)
    print(X_train.shape)
    # for i,j in zip(X_train,y_train):
    #     print(i,j)


