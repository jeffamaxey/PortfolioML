"""Pre-processing dataframe"""
import pandas as pd
import numpy as np
import argparse
import logging

def read_filepath(file_path):
    """
    Read and compute basic informations about a data set in csv.

    Parameters
    ----------
    file_path: str
        Path to the csv file

    Returns
    -------
    df: pandas dataframe
        Pandas Dataframe of the read file
    """
    name_file = file_path.split('/')[-1]
    extention = name_file.split('.')[-1]

    try:
        if extention != 'csv':
            raise NameError('The file is not a csv one')
    except NameError as ne:
        logging.error(ne)
        exit()

    df = pd.read_csv(file_path, encoding='latin-1')

    logging.info('DATA INFO, attributes: %s', df.columns)
    logging.info('DATA INFO, shape: %s', df.shape)

    total_data = df.shape[0]*df.shape[1]
    total_missing = pd.DataFrame.isnull(df).sum().sum()

    logging.info(f'DATA QUALITY, missing values: {total_missing/total_data:.2%}')

    return df


def get_returns(dataframe, m, export_returns_csv, no_missing=True):
    """
    Get the day-by-day returns value of a company. The dataframe has got companies as attributes
    and days as rows, the values are close prices of each days

    Parameters
    ----------
    dataframe: pandas dataframe
        Input data frame

    col: string
        Name of company, for possible values check dataframe.columns

    m: int
        m-period return

    export_csv: bool(optional)
        Export dataframe in csv. default=False

    no_missing: bool(optional)
        drop companies with missing values. default=True

    Returns
    -------
    df: pandas dataframe
        Dataframe with m-returns
    """
    try:
        if m < 0: raise ValueError("Ops! Invalid input, you can't go backward in time. m must be positive.")
    except ValueError as ve:
        print(ve)
        exit()

    df = pd.DataFrame()
    for col in dataframe.columns[1:]:
        today = dataframe[col]
        tomorrow = today[m:]
        df[col] = (np.array(tomorrow)/np.array(today)[:-m])-1

    if no_missing: df = df.dropna(axis=1)

    if export_returns_csv:
        df.to_csv('ReturnsData.csv')

    return df

def binary_targets(dataframe, export_binary_csv):
    """
    Returns binary value of returns for classification task, binary values are 0 and 1.
    To define the two classes, we order all m-period returns of all stocks 's'
    in period t + m in ascending order and cut them into two equally sized classes.
    Class 0 is realized if the m-period return of stock is smaller than the cross-sectional median return
    of all stocks in period t + 1 .
    Similarly, class 1 is realized if the m-period return of 's' is larger than or equal to the cross-sectional
    median.
    For more details see: https://doi.org/10.1016/j.ejor.2017.11.054

    Parameters
    ----------
    dataframe: pandas dataframe
        Inpunt dataframe of returns

    export_binary_csv: bool
        Export dataframe in csv. default=False

    Returns
    -------
    df: pandas dataframe
        Output dataframe of binary returns
    """
    df = dataframe
    for time_idx in range(dataframe.shape[0]):
        compare_list = list(dataframe.iloc[time_idx].values)
        compare_list.sort()
        compare_value = compare_list[int(len(compare_list)/2)]

        df.iloc[time_idx] = dataframe.iloc[time_idx].apply(lambda x:  0 if x<=compare_value else 1)

    if export_binary_csv: df.to_csv('ReturnsBinary.csv')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process price data and get the dataframe of m period returns')
    parser.add_argument('input_file', type=str, help='Path to the input file')
    parser.add_argument('m_period_return', type=int, help='m period return')
    parser.add_argument("-export_returns_csv", default= False, help='Export to csv the dataframe of m-period price returns')
    parser.add_argument("-export_binary_csv", default= False, help='Export to csv the dataframe for the classification task')
    parser.add_argument("-log", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))

    args = parser.parse_args()

    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}

    logging.basicConfig(level= levels[args.log])

    df = read_filepath(args.input_file)
    dataframe_ritorni = get_returns(df,args.m_period_return, args.export_returns_csv)
    print(dataframe_ritorni.isnull().sum())

    dataframe_binary = binary_targets(dataframe_ritorni, args.export_binary_csv)

