import pandas as pd
import numpy as np

dataframe = pd.read_csv('/hdd/Università/Magistrale/Computing methods/PortfolioML/PriceData.csv')

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


def ritorni(dataframe, col):
    oggi = dataframe[col].values
    domani = oggi[1:]
    domani = np.append(domani,0)
    ritorni = domani/oggi -1
    return ritorni

for col in dataframe.columns[1:]:
    dataframe[col] = ritorni(dataframe,col)
        

    
