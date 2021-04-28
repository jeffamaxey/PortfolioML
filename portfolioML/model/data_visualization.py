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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creation of input and output data for lstm classification problem')
    parser.add_argument("-time_feature_steps", type=int, default=240, help="Window over which you compute the mean")
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
    
    #Data Visualization
    days = df_price['Date']
    x_label_position = np.arange(0, len(days), 150)
    x_label_day = [days[i] for i in x_label_position]

    plt.figure(figsize=[13,15])
    for i in np.random.randint(1,365,5):
        comp=df_price.columns[i]
        plt.plot(df_price[comp], lw=0.8, label=f"{comp}")
    plt.xticks(x_label_position, x_label_day, rotation=60)
    plt.legend()
    plt.grid()
    plt.title("Close price of several Companies")
    plt.ylabel("Close price")
    plt.xlabel("Days")
    

    #Create feature
    data_feature = stock_features_df(df_price[1:], time_feature_steps=args.time_feature_steps)
    print(data_feature.head())

    #Correlazioni semplici
    corr = data_feature.transpose().corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    ax.set_title("Correlation Matrix")

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 15, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap,vmax=0.5, center=0,
                square=True, linewidths=0.00005)

    plt.show()
    


