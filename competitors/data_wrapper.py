from carla.data.catalog import DataCatalog

from recourse_fare.utils.preprocess.fast_preprocessor import FastPreprocessor

import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

class DataWrapper(DataCatalog):

    def __init__(self, train_data: str, test_data:str, preprocessor, immutables: list):
        
        self._preprocessor = preprocessor

        self._continuous = []
        self._categorical = []
        self._immutables = immutables

        test_data = pd.read_csv(test_data)
        self._df_test = preprocessor.transform(test_data.copy(), type="dataframe")

        train_data = pd.read_csv(train_data)
        self._df_train = preprocessor.transform(train_data.copy(), type="dataframe")

        self.original_feature_order = test_data.columns
    
    @property
    def encoder(self):
        return self._preprocessor

    @property
    def df(self) -> pd.DataFrame:
        return self._df_train.copy()

    @property
    def df_train(self) -> pd.DataFrame:
        return self._df_train._df.copy()

    @property
    def df_test(self) -> pd.DataFrame:
        return self._df_test.copy()

    @property
    def categorical(self):
        return self._preprocessor.categorical.copy()

    @property
    def continuous(self):
        return self._preprocessor.continuous.copy()

    @property
    def immutables(self):
       return self._immutables.copy()

    @property
    def target(self):
        return ""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        output = df.copy()
        return self._preprocessor.transform(output, type="dataframe")

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        output = df.copy()
        return self._preprocessor.inverse_transform(output, type="dataframe")

       
