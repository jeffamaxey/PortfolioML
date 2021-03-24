"""Unit test split.py"""
import os
import unittest
import sys
import pandas as pd
import random
from portfolioML.data.split import split_sequences

def _full_path(file_name):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), file_name)

df_return = pd.read_csv(_full_path('ReturnsData.csv'))
df_binary = pd.read_csv(_full_path('ReturnsBinary.csv'))

class TestDataReturns(unittest.TestCase):
    def test_split_sequences(self):
        self.assertEqual(len(df_return), len(df_binary))
        column_idx = random.randint(1, df_return.shape[1])
        random_tick = df_binary.columns[column_idx]
        X, y = split_sequences(df_return[random_tick], df_binary[random_tick])
        for i in range(1,10):
            self.assertAlmostEqual(X[i,0], X[i-1,1])

