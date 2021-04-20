import argparse
import logging
import os
import numpy as np
from sklearn.decomposition import PCA
from portfolioML.data.data_returns import read_filepath
from portfolioML.data.wavelet import approx_details_scale


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
    most_important_features = list(set(most_important_companies))

    return most_important_features

    def wavelet_dataframe(df_price_path, wavelet):
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
            d : dictionary
        '''

        df_price = read_filepath(df_price_path)
        df_price = df_price.dropna(axis=1) # Drop NaN
        df_price = df_price.drop(labels='Date', axis=1)
        d = {}
        for tick in df_price.columns[:5]:
            a1,d1 = approx_details_scale(df_price[tick], wavelet, 1)
            a2,d2 = approx_details_scale(df_price[tick], wavelet, 2)
            a3,d3 = approx_details_scale(df_price[tick], wavelet, 3)
            d[tick] = [[a1[i], a2[i], a3[i]] for i in range(len(a1))]

        return d


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
    df_price_path = od.getcwd() + "/PriceData.csv"
    most_important_companies = pca(df_returns_path)
    wavelet_data = wavelet_dataframe(df_price_path, 'haar')
