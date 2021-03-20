import pandas as pd
import numpy as np
import argparse
import logging

def check_missing_values(dataframe,step=250):
    '''Print some info about missing values in the dataframe of returns.  
    
    Parameters
    ----------
    dataframe: str
        Path to the dataframe that must have as rows the dates and as columns the companies
    
    step (optional): integer
        Step size relative to the rows used for the checking. Defalut = 250
    '''
    dataframe = pd.read_csv(dataframe)
    totale = dataframe.shape[0]*dataframe.shape[1]
    for i in range(0,len(dataframe),step):
        null = dataframe[i:].isnull().sum()
        companies = {index: element for index,element in enumerate(null) if element != 0}
        percentage = null.sum()/totale
        tickers = [dataframe.columns[comp] for comp in companies]
        
        print(f'{i}) %: {percentage*100} \n Symbols: {len(tickers)} \n Residuals: {len(dataframe)-i}, Date: {dataframe.Date[i]} \n')
        

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Get some statistics about returns')
    parser.add_argument('dataframe', type=str, help="Path to the dataframe of returns")
    parser.add_argument('-step', default=250, type=int, help='Choose the step size for the checking')
    args = parser.parse_args()
    
    
    check_missing_values(args.dataframe, args.step)