""" Generate csv file """
import pandas as pd 
import requests 
from bs4 import BeautifulSoup 
import pandas_datareader as web
import logging 
import argparse
                    
def get_ticker():
    """
    Get tickers of companies in S&P500 over all time from Wikipedia page
    url = https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#S&P_500_component_stocks
    
    Return
    ------
    ticker: list
        list of tickers
    """
    website_url = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#S&P_500_component_stocks').text
    soup = BeautifulSoup(website_url,'lxml')


    My_table = soup.find('table',{'class':'wikitable sortable', 'id':"constituents"})
    df = pd.read_html(str(My_table))
    df = pd.DataFrame(df[0])
    
    ticker = list(df.Symbol)

    return ticker

def data_generator(start, end, data_source = 'yahoo', export_csv = True):
    '''
    Generate a pandas dataframe of historical close daily price.
    
    Parameters
    ----------
    start: str
        Start time. Its format must be yyyy-mm-dd
    
    end: str
        End time. Its format must be yyyy-mm-dd
        
    data_source: str(optional)
        The data source ("iex", "fred", "ff"). Default = 'yahoo'
        
    export_csv: bool(optional)
        Choose whether to export to csv. Default = True
        
    
    Returns
    -------
    data: pandas dataframe
        Pandas dataframe of historical close daily price
    '''
        
    tickers = get_ticker()
    data_list = {}
    for ticks in tickers[:3]:
        try:
            data_list[ticks] = web.DataReader(ticks, data_source = data_source, start = start, end = end).Close
            logging.info(f'Downloading data of {ticks}')
        except:
            logging.info(f"There's no data for {ticks}" )
    data = pd.DataFrame(data_list)
    if export_csv: data.to_csv('PriceData.csv')
    
    
    return data

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generator of historical price data') # questo prende l'input da terminale
    parser.add_argument('start', type=str, help="Start time")
    parser.add_argument('end', type=str, help="End time")
    parser.add_argument("-l", "--log", default="info", help=("Provide logging level. Example --log debug', default='info"))
    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    
    logging.basicConfig(level= levels[args.log])
    data = data_generator(args.start, args.end)
   
    
  
        
        
    

    


