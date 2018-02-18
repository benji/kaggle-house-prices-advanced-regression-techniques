import unittest
import math
import json
import sys
import os
import yaml
import pandas as pd
import numpy as np
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils import *
from Training import *

pd.options.mode.chained_assignment = None  # default='warn'


class TestTraining(unittest.TestCase):
    def setUp(self):
        df_train = pd.DataFrame(data={
            'id': [0, 1, 2, 3, 4],
            'sex': ['M', 'F', 'M', 'M', 'F'],
            'age': [12, 36, 32, 16, 21],
            'kids': [0, 2, 3, 0, 1]})
        df_test = pd.DataFrame(data={
            'id': [0, 1, 2],
            'sex': ['F', 'F', 'M'],
            'age': [27, 26, 41]})
        schema = {
            'target': 'kids',
            'id': 'id',
            'columns': {
                'kids': {'type': 'NUMERIC'},
                'age': {'type': 'NUMERIC'},
                'sex': {'type': 'CATEGORICAL', 'categories': ['F', 'M']},
            }}
        self.t = Training(df_train, df_test, schema)

    def test_dummies(self):
        self.t.dummify_at_init = True
        self.t.prepare()

        self.assertEqual(3, len(self.t.df_train.columns))

        values = self.t.df_train['sex_M'].unique()
        self.assertEqual(2, len(values))
        self.assertTrue(0 in values)
        self.assertTrue(1 in values)

    def test_label_encoding(self):
        self.assertEqual('M', self.t.df_train.iloc[0]['sex'])

        self.t.use_label_encoding = True
        self.t.prepare()

        # test categories order from schema is respected
        self.assertEqual(1, self.t.df_train.iloc[0]['sex'])
        self.assertEqual(0, self.t.df_train.iloc[1]['sex'])


if __name__ == '__main__':
    unittest.main()
