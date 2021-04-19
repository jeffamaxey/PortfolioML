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
from portfolioML.data.data_returns import read_filepath
import matplotlib.pyplot as plt
import seaborn as sns
from portfolioML.makedir import smart_makedir, go_up

def stock_features_df(dataframe, time_feature_steps=5):
    """
    Given a dataset of returns(prices) it returns the feature each time_feature_steps.
    the featare are now only the mean values of returns(prices) each time_feature_steps

    Parameters
    ----------
    dataframe: pandas dataframe
        Input dtaframe

    time_feature_steps: integer (optional)
        Time window over which compute the feature. Default = 10

    Returns
    -------
    data_feature: pandas dataframe
        Output dataframe. It has gat the companies as istances(records) and the feature
        as attributes
    """

    df_feature = [dataframe[i:i+time_feature_steps].mean() for i in range(0, len(dataframe), time_feature_steps)]
    data_feature = pd.DataFrame(df_feature[:-1])
    data_feature = data_feature.transpose()

    new_attributes = {i: f'{i+1}-Period' for i in data_feature.columns}
    data_feature = data_feature.rename(columns=new_attributes)

    return data_feature

def correlated_stock(data_feature, anti_value = -0.5, positive_value=0.85):
    """
    Returns a list of anticorrelated and positive correlated stocks w.r.t a thrashold level setting by users
    the output is a complete list of both type of companies.

    Parameters
    ----------
    data_feature: pandas dataframe
        Input dataframe, it must hìbe a dataframe with features for each companies, see stock_features_df.
    """
    cormatrix = data_feature.transpose().corr()
    cor_np = np.array(cormatrix)
    cor_unique = np.triu(cor_np)

    name = data_feature.transpose().columns

    #anti correlated
    anti_cor = cor_unique[cor_unique < anti_value]
    print(len(anti_cor))
    anticor_list = []
    for val in anti_cor:
        position = np.where(cor_unique == val)
        print(f'value:{val}, position: {position} ... {name[position[0]].values[0]}-{name[position[1]].values}')
        anticor_list.append(name[position[0]].values[0])
        anticor_list.append(name[position[1]].values[0])

    anticor_list = list(set(anticor_list))
    print(f'anti {(anticor_list)}')

    #positive correlated
    positive_cor = cor_unique[cor_unique > positive_value]
    positive_cor = positive_cor[positive_cor != 1]
    positcor_list = []
    for val in positive_cor:
        position = np.where(cor_unique == val)
        # print(f'value:{val}, position: {position} ... {name[position[0]].values[0]}-{name[position[1]].values}')
        positcor_list.append(name[position[0]].values[0])
        positcor_list.append(name[position[1]].values[0])

    positcor_list = list(set(positcor_list))
    print(f'posit {len(positcor_list)}')

    stocks = list(set(positcor_list + anticor_list))

    return stocks, cor_np, cormatrix, anticor_list, positcor_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creation of input and output data for lstm classification problem')
    parser.add_argument("-time_feature_steps", type=int, default=5, help="Window over which you compute the mean")
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
    df_price = pd.read_csv(go_up(1) + "/data/PriceData.csv")
    df_price = df_price.dropna(axis=1)

    #Create feature
    data_feature = stock_features_df(df_price[1:], time_feature_steps=args.time_feature_steps)
    print(data_feature.head())

    #Correlazioni
    stocks, cor_np, cormatrix, a, p = correlated_stock(data_feature, anti_value = -0.7, positive_value=0.95)
    print(f'selected stocks:{len(stocks)}')

    plt.figure()
    plt.imshow(np.triu(cor_np))

    ax = plt.figure()
    ax = sns.heatmap(cormatrix, cmap=sns.diverging_palette(20, 220, n=200))

    #Sub matrix
    stocks.sort()
    cormatrix_sub = cormatrix[stocks].loc[stocks] 
    positive = cormatrix[cormatrix > 0.9][p].loc[p]
    negative = cormatrix[cormatrix < -0.7][a].loc[a]    

    ax = plt.figure()
    ax = sns.heatmap(cormatrix_sub, cmap=sns.diverging_palette(20, 220, n=200))

    plt.figure()
    plt.imshow(positive)
    plt.imshow(negative)
    plt.colorbar()

    plt.show()
    


