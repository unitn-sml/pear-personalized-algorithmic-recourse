import warnings
from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np
import pandas as pd

import torch

from carla.models.api import MLModel
from carla.data.api import Data

from recourse_fare.utils.preprocess.fast_preprocessor import StandardPreprocessor

class BlackBoxWrapper(MLModel):

    def __init__(self, model: Any, preprocessor: Any, data: Data) -> None:

        self._model = model
        self._preprocessor = preprocessor
        super().__init__(data)

    @property
    def feature_input_order(self):
        return self.data._preprocessor.feature_names_ordering

    @property
    def backend(self):
        return "pytorch"
    
    @property
    def model_type(self):
        return "ann"

    @property
    def raw_model(self):
        return self._model

    def predict(self, x: Union[np.ndarray, pd.DataFrame]):
        
        if isinstance(self._model, torch.nn.Module):
            self._model.eval()
            with torch.no_grad():
                return np.round(
                    self.predict_proba(
                    x
                    )
                )
        else:
            return np.round(self.predict_proba(x))

    def predict_proba(self, x: Union[np.ndarray, pd.DataFrame]) -> np.array:
        
        if not isinstance(x, np.ndarray):
            x_pred = x.values
        else:
            x_pred = x.copy()

        if isinstance(self._model, torch.nn.Module):
            with torch.no_grad():
                output = self._model(torch.FloatTensor(
                        x_pred
                    ))
            return output.numpy()
        else:
            return self._model.predict_proba(x_pred)[:, 1]
    
    def predict_proba_double(self, x: Union[np.ndarray, pd.DataFrame]) -> np.array:
        
        if isinstance(x, np.ndarray):
            x_pred = x.copy()
            x_pred = pd.DataFrame(x_pred, columns=self.feature_input_order)
            x_pred = self.data._preprocessor.inverse_transform(x_pred, type="dataframe")
            x_pred = self._preprocessor.transform(x_pred)
        else:
            x_pred = self.data._preprocessor.inverse_transform(x, type="dataframe")
            x_pred = self._preprocessor.transform(x_pred)

        with torch.no_grad():
            output = self._model(torch.FloatTensor(
                    x_pred
                ))
        return torch.cat([1-output, output], -1).numpy()
    
    def get_mutable_mask(self):
        """
        Get mask of mutable features.

        For example with mutable feature "income" and immutable features "age", the
        mask would be [True, False] for feature_input_order ["income", "age"].

        This mask can then be used to index data to only get the columns that are (im)mutable.

        Returns
        -------
        mutable_mask: np.array(bool)
        """
        # find the index of the immutables in the feature input order
        immutable = [self.feature_input_order.index(col) for col in self.data.immutables]
        # make a mask
        mutable_mask = np.ones(len(self.feature_input_order), dtype=bool)
        # set the immutables to False
        mutable_mask[immutable] = False
        return mutable_mask