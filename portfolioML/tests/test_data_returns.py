"""Unit test data_returns"""
import os
import unittest
import sys 
import pandas as pd
from portfolioML.data_returns import get_returns

def _full_path(file_name):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), file_name)

dataframe = pd.read_csv(_full_path('PriceData.csv'))
class TestDataRaturns(unittest.TestCase):
    """Class test for data_returns.py"""

    def test_m(self):
        """test correst result with some value m"""
        m_list = range(1,10)
        for m in m_list:
            self.assertAlmostEqual(get_returns(dataframe,m).ALL[0],(dataframe.ALL[m]/dataframe.ALL[0])-1)

if __name__ == '__main__':
    unittest.main()