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

def check_missing_values(dataframe,step=250):
    '''Print some info about missing values in a dataframe. 
    
    Parameters
    ----------
    dataframe: pandas dataframe
        It must have as rows the dates and as columns the companies
    
    step (optional): integer
        Step size relative to the rows used for the checking. Defalut = 250
    '''
    totale = dataframe.shape[0]*dataframe.shape[1]
    for i in range(0,len(dataframe),step):
        null = dataframe[i:].isnull().sum()
        companies = {index: element for index,element in enumerate(null) if element != 0}
        percentage = null.sum()/totale
        tickers = [dataframe.columns[comp] for comp in companies]
        
        print(f'{i}) %: {percentage*100} \n Symbols: {len(tickers)} \n Residuals: {len(dataframe)-i}, Date: {dataframe.Date[i]} \n')


def ritorni(dataframe, export_csv = False):
    """
    Get the day-by-day returns value of a company. The dataframe has got companies as attributes
    and days as rows, the values are close prices of each days

    Parameters
    ----------
    dataframe: pandas dataframe
        Input data frame

    col: string
        Name of company, for possible values check dataframe.columns

    Returns
    -------
    """
    for col in dataframe.columns[1:]:
        today = df[col]
        tomorrow = today[1:]
        tomorrow[-1] = 0  
        dataframe[col] = (np.array(tomorrow)/np.array(today))-1  

    if export_csv: dataframe.to_csv('PriceData.csv')
    return dataframe
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing dataframe')
    parser.add_argument('input_file', type=str, help='Path to the input file')
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
    a = ritorni(df)

    print(a)
    # for col in dataframe.columns[1:]:
    #     dataframe[col] = ritorni(dataframe,col)

