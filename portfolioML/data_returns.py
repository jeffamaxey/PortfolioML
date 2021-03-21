"""Pre-processing dataframe"""
import pandas as pd
import numpy as np
import argparse
import logging

def read_filepath(file_path):
    """
    Read and compute basic informations about a data set in csv.

    Return name and extention file in a tuple.
    """
    name_file = file_path.split('/')[-1]
    extention = name_file.split('.')[-1]

    if extention != 'csv':
        logging.error('The input file is not a csv file')

    df = pd.read_csv(file_path, encoding='latin-1')

    logging.info('DATA INFO, attributes: %s', df.columns)
    logging.info('DATA INFO, shape: %s', df.shape)

    total_data = df.shape[0]*df.shape[1]
    total_missing = pd.DataFrame.isnull(df).sum().sum()

    logging.info(f'DATA QUALITY, missing values: {total_missing/total_data:.2%}')

    return df
  

def get_returns(dataframe, m, export_csv = False):
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

    Returns
    -------
    df: pandas dataframe
        Dataframe with m-returns
    """
    try:    
        if m < 0: raise ValueError("Ops! Invalid input, you can't go backward in time. m must be positive.")
    except ValueError as ve:
        print(ve)
        # logging.error("Ops! Invalid input, you can't go backward in time. m must be positive.")
        # quit()

    df = pd.DataFrame()
    for col in dataframe.columns[1:]:
        today = dataframe[col]
        tomorrow = today[m:] 
        df[col] = (np.array(tomorrow)/np.array(today)[:-m])-1

    if export_csv: df.to_csv('ReturnsData.csv')
    return df
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing dataframe')
    parser.add_argument('input_file', type=str, help='Path to the input file')
    parser.add_argument('m_period_return', type=int, help='m period return')
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
    print((df.ALL[1]/df.ALL[0])-1)
    dataframe_ritorni = get_returns(df,args.m_period_return)
    print(dataframe_ritorni.ALL[0])