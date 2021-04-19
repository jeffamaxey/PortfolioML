import argparse
import logging
import os
import numpy as np
from sklearn.decomposition import PCA
from portfolioML.data.data_returns import read_filepath


def pca(df_returns_path, n_components):
    '''


    Parameters
    ----------
    df_returns_price : TYPE
        DESCRIPTION.
    n_components : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    df_returns = read_filepath(df_returns_path)
    pca = PCA(n_components=n_components)
    pca.fit(df_returns.values)
    logging.info(f"Fraction of variance preserved: {pca.explained_variance_ratio_.sum():.2f}")

    n_pca= pca.n_components_ # get number of components
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pca)] # get the index of the most important feature on EACH component
    initial_feature_names = df_returns.columns
    most_important_companies = [initial_feature_names[most_important[i]] for i in range(n_pca)] # get the most important feature names
    most_important_companies = list(set(most_important_companies))

    return most_important_companies


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
    most_important_companies = pca(df_returns_path)
