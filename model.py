from carla.models.api import MLModel
from carla.data.api import Data

import pandas as pd
import numpy as np

import torch

from typing import Union

class BlackBox(MLModel):

    def __init__(self, raw_model, data: Data) -> None:
        super().__init__(data)

        self._feature_input_order = list(data.df.columns)
        self._raw_model = raw_model
    
    @property
    def feature_input_order(self):
        return self._feature_input_order
    
    @property
    def backend(self):
        return "sklearn"
    
    @property
    def raw_model(self):
        return self._raw_model
    
    def predict(self, x: Union[np.ndarray, pd.DataFrame]):
        return self._raw_model.predict(x)
    
    def predict_proba(self, x: Union[np.ndarray, pd.DataFrame]):
        return self._raw_model.predict_proba(x)

class BlackBoxTorch(MLModel):

    def __init__(self, raw_model, data: Data) -> None:
        super().__init__(data)

        self._feature_input_order = list(data.df.columns)
        self._raw_model = raw_model
    
    @property
    def feature_input_order(self):
        return self._feature_input_order
    
    @property
    def backend(self):
        return "pytorch"
    
    @property
    def raw_model(self):
        return self._raw_model
    
    def predict(self, x: Union[np.ndarray, pd.DataFrame]):
        return torch.argmax(self._raw_model(x), dim=1)
    
    def predict_proba(self, x: Union[np.ndarray, pd.DataFrame]):
        with torch.no_grad():
            return self._raw_model(torch.FloatTensor(x.values))