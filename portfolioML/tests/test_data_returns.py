"""Unit test data_returns"""
import os
import unittest
import sys
import pandas as pd
from portfolioML.data.data_returns import get_returns

def _full_path(file_name):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), file_name)

dataframe = pd.read_csv(_full_path('PriceData.csv'))
class TestDataReturns(unittest.TestCase):
    """Class test for data_returns.py"""

    def test_m(self):
        """
        Test correct result with some value m
        """
        m_list = range(1,10)
        for m in m_list:
            self.assertAlmostEqual(get_returns(dataframe,m,export_csv=False).ALL[0],(dataframe.ALL[m]/dataframe.ALL[0])-1)

    def test_binary_targets(self):
        """
        Test if the dataframe created by get_returns is without missing values
        """
        dataframe_no_nan = get_returns(dataframe, 1, export_csv=False)
        bool_list = dataframe_no_nan.isnull().any()
        self.assertTrue(bool_list.any() == False)

if __name__ == '__main__':
    unittest.main()
