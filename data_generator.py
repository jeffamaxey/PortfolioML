"""Export wititable to pandas data frame"""
import pandas as pd 
import requests 
from bs4 import BeautifulSoup 

def GetTicker():
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
    
    stocks = list(df.Symbol)

    return stocks


if __name__=='__main__':

    tickers = GetTicker()
    print(len(tickers))

    


