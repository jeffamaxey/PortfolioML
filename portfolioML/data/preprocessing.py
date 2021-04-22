import argparse
import logging
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from portfolioML.data.data_returns import read_filepath
from portfolioML.data.wavelet import approx_details_scale
from portfolioML.model.split import all_data_LSTM


def pca(df_returns_path, n_components):
    '''
    Compute the PCA decomposition of a dataset

    Parameters
    ----------
    df_returns_price : str
        Path to the csv dataframe file.
    n_components : int
        Number of components you want your dataframe be projected to.

    Returns
    -------
    most_important_features : list
        List of the most important features in the dataset.

    '''

    df_returns = read_filepath(df_returns_path)
    pca = PCA(n_components=n_components)
    pca.fit(df_returns.values)
    logging.info(f"Fraction of variance preserved: {pca.explained_variance_ratio_.sum():.2f}")

    n_pca= pca.n_components_ # get number of components
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pca)] # get the index of the most important feature on EACH component
    initial_feature_names = df_returns.columns
    most_important_features = [initial_feature_names[most_important[i]] for i in range(n_pca)] # get the most important feature names
    most_important_features = list(set(most_important_features))

    return most_important_features

def wavelet_dataframe(df_returns_path, wavelet):
    '''
    Compute the DWT (Discrete Wavelet Tranform) of a dataset composed by multiple time signals

    Parameters
    ----------
    df_price_path : str
        Path of the pandas dataframe of time signals
    wavelet : str
        Which wavelet is used during the decomposition.

    Returns
    -------
        dataframe : pandas.DataFrame
            Pandas dataframe in which each element is composed by the three timestamp of the first 3 approximations
    '''

    df_returns = read_filepath(df_returns_path)
    dic1, dic2, dic3 = {}, {}, {}
    for tick in df_returns.columns:
        a1,d1 = approx_details_scale(df_returns[tick], wavelet, 1)
        a2,d2 = approx_details_scale(df_returns[tick], wavelet, 2)
        a3,d3 = approx_details_scale(df_returns[tick], wavelet, 3)
        dic1[tick], dic2[tick], dic3[tick] = a1, a2, a3
    dataframe1 = pd.DataFrame(dic1)
    dataframe2 = pd.DataFrame(dic2)
    dataframe3 = pd.DataFrame(dic3)
    dataframe1.to_csv("MultidimReturnsData1.csv")
    dataframe2.to_csv("MultidimReturnsData2.csv")
    dataframe3.to_csv("MultidimReturnsData3.csv")

    return dataframe1, dataframe2, dataframe3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Determine the most important companies on each components obtained from PCA')
    parser.add_argument("-log", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    args = parser.parse_args()

    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}

    logging.basicConfig(level= levels[args.log])


    df_returns_path = os.getcwd() + "/ReturnsData.csv"
    df_binary = read_filepath("ReturnsBinary.csv")
    most_important_companies = pca(df_returns_path, n_components=250)
    print(most_important_companies)
    # wavelet_data1, wavelet_data2, wavelet_data3 = wavelet_dataframe(df_returns_path, 'haar')
    # X_train1, y_train, X_test1, y_test = all_data_LSTM(wavelet_data1, df_binary, 1)
    # X_train2, y_train, X_test2, y_test = all_data_LSTM(wavelet_data2, df_binary, 1)
    # X_train3, y_train, X_test3, y_test = all_data_LSTM(wavelet_data3, df_binary, 1)
    # X_train = np.stack((X_train1, X_train2, X_train3), axis=-1).reshape(X_train1.shape[0],240,3)
    # X_test = np.stack((X_test1, X_test2, X_test3), axis=-1).reshape(X_test1.shape[0],240,3)
