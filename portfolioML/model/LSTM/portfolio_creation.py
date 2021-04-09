import pandas as pd
import numpy as np
import logging
import argparse
import sys
import os
import shutil
# sys.path.append(os.path.dirname(os.path.abspath("..")))
# from split import split_Tperiod

def get_trading_values(df_price, predictions_folder, len_period=1308, len_train = 981, len_test=327):
    '''
    Generate a pandas dataframe composed by all the days of which lstm.py forecasts
    the prices. This is due to the fact that lstm.py doesn't track the dates'


    Parameters
    ----------
    df_price : str
        Path of the csv file of prices.
    predictions_folder : str
        Path of the folder in which the predictions made by lstm.py are.
    len_period : int, optional
        Lenght of each study period. The default is 1308.
    len_train : int, optional
        Lenght of the training set. The default is 981.
    len_test : int, optional
        Lenght of the trading set. The default is 327.

    Raises
    ------
    OSError
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    len_total_leave = len(df_price)-len_period
    # Divide in periods the price dataframe
    periods_price = [(df_price[i:len_period+i]) for i in range(0, len_total_leave+1, len_test)]
    # Select only the test sets
    tests = [periods_price[i][len_train:] for i in range(len(periods_price))]
    # Select then only days in test sets of which forecasts are made
    trading_values = [tests[i][240:] for i in range(len(tests))]

    path = os.getcwd() + '/Predictions'
    if os.path.exists(path):
        logging.info(f"Path '{path}' already exists, it will be overwrited")
        shutil.rmtree(path)
    os.mkdir(path)
    logging.info(f"Successfully created the directory '{path}' \n")

    # Insert the Date column in the forecasts made by lstm.py
    for i in range(10):
        ith_predictions = pd.read_csv(f"{predictions_folder}/Predictions_{i}th_Period.csv",
                                      index_col=0)
        ith_predictions.insert(0,'Date',trading_values[i]['Date'].values)
        ith_predictions.to_csv(f"{path}/Trading_days_period{i}.csv")

def portfolio(forecasts, k=10):
    '''
    This function creates a list composed by list of pandas dataframe each of which contains
    the top k and flop k forecasts made by the LSTM algorithm in lstm.py

    Parameters
    ----------
    forecasts : str
        Path to the predictions csv file.
    k : int, optional
        Number of top and flop forecasts that will be considered. The default is 10.

    Returns
    -------
    portfolio : list
        List of lists of pandas dataframe.
    '''
    trading_days = pd.read_csv(forecasts, index_col=0)
    portfolio = []
    for i in range(trading_days.shape[0]):
        df_portfolio = pd.DataFrame()
        day = trading_days['Date'][i]
        values = trading_days.iloc[i][1:]
        values = values.sort_values()
        first_k_val, first_k_comp = values[:k], values[:k].index
        last_k_val, last_k_comp = values[-k:], values[-k:].index
        values_traded = list(first_k_val.values) + list(last_k_val.values)
        companies_traded = list(first_k_comp) + list(last_k_comp)
        df_portfolio[day] = values_traded
        df_portfolio['Company'] = companies_traded
        portfolio.append(df_portfolio)
    return portfolio

def forecast_returns(num_period=10):
    '''
    The following is aimed to calculate the daily returns. We set a long position for the
    first k companies at each day and a short position for the last k ones. So, we
    calculate the return using prices ad t ad t+1 for the formers and t-1 ad t for the
    latters

    Parameters
    ----------
    num_period : int, optional
        Number of period over which forecast returns have to be calculated. The default is 10

    Returns
    -------
    returns : numpy array
        Numpy array of all returns for all periods

    '''
    returns = []
    for period in range(num_period):
        for i in range(len(portfolio[0])):
            # Select one trading day
            trading_day = portfolio[period][i].columns[0]
            # List all companies in that day
            companies = portfolio[period][i].Company.tolist()
            # Select the corresponding index in df_price
            index = df_price.Date[df_price.Date == trading_day].index.tolist()
            index = index[0]
            for i,comp in enumerate(companies):
                if i <=9:
                    returns.append(df_price[comp][index]/df_price[comp][index+1] - 1)
                else:
                    returns.append(df_price[comp][index+1]/df_price[comp][index] - 1)
    returns = np.array(returns)
    logging.info('Average daily returns %2f', returns.mean())
    logging.info('Standard deviation %2f', returns.std())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creation of portfolios based on LSTM predictions')
    parser.add_argument("-log", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("df_prices", type=str, help="Path of the csv file of prices")
    parser.add_argument("predictions_folder", type=str, help="Path of the folder in which the predictions made by lstm.py are ")
    parser.add_argument("-num_period", default=10, help="Number of period over which returns have to be calculated ")
    args = parser.parse_args()

    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}

    logging.basicConfig(level= levels[args.log])

    df_price = pd.read_csv(args.df_prices)
    df_price = df_price.dropna(axis=1)
    predictions_folder = args.predictions_folder
    num_period = args.num_period

    trading_values = get_trading_values(df_price,predictions_folder)
    path = os.getcwd()
    portfolio = [portfolio(path + f"/Predictions/Trading_days_period{i}.csv") for i in range(10)]
    returns = forecast_returns(args.num_period)




    # sum_returns = []
    # tmp = 0
    # for i in range(len(returns)-1):
    # sum_returns.append(returns[i]+tmp)
    # tmp = sum_returns[i]


