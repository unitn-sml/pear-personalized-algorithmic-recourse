from carla.data.api.data import Data
from recourse_fare.utils.preprocess.fast_preprocessor import FastPreprocessor

import pandas as pd
from sklearn.model_selection import train_test_split

class Adult(Data):

    def __init__(self, data_file, test_data_file) -> None:

        self._df = pd.read_csv(data_file)
        self._df_test = pd.read_csv(test_data_file)
        self.columns_order = self._df.columns

        self.preprocessor = FastPreprocessor()
        self.preprocessor.fit(self._df)

        self._categorical = []
        self._continuous = self.columns_order
        self._immutables = ["native_country", "race", "relationship", "marital_status"]
        self._target = "income_target"

        self._identity_encoding = (
            "Identity"
        )

        self._df = self.transform(self._df)
        self._df_test = self.transform(self._df_test)

        self.name = "Adult"

        super().__init__()
    
    def transform(self, df):
        return pd.DataFrame(self.preprocessor.transform(df), columns=self.columns_order)

    def inverse_transform(self, df, type="value"):
        return self.preprocessor.inverse_transform(df, type)

    @property
    def categorical(self):
        return self._categorical

    @property
    def continuous(self):
        return self._continuous
    
    @property
    def df(self):
        return self._df.copy()
    
    @property
    def df_test(self):
        return self._df_test.copy()
    
    @property
    def df_train(self):
        return self._df_train.copy()

    @property
    def immutables(self):
        return self._immutables
    
    @property
    def target(self):
        return self._target
    