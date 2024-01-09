from typing import Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
from competitors.face.face_method import graph_search
from carla.recourse_methods.processing import (
    encode_feature_names,
    merge_default_parameters,
)

def check_counterfactuals(
    mlmodel: MLModel,
    counterfactuals,
    factuals_index: pd.Index,
    negative_label: int = 0,
) -> pd.DataFrame:
    """
    Takes the generated list of counterfactuals from recourse methods and checks if these samples are able
    to flip the label from 0 to 1. Every counterfactual which still has a negative label, will be replaced with an
    empty row.

    Parameters
    ----------
    mlmodel:
        Black-box-model we want to discover.
    counterfactuals:
        List or DataFrame of generated samples from recourse method.
    factuals_index:
        Index of the original factuals DataFrame.
    negative_label:
        Defines the negative label.

    Returns
    -------
    pd.DataFrame
    """

    if isinstance(counterfactuals, list):
        df_cfs = pd.DataFrame(
            np.array(counterfactuals),
            columns=mlmodel.feature_input_order,
            index=factuals_index.copy(),
        )
    else:
        df_cfs = counterfactuals.copy()

    return df_cfs, np.round(mlmodel.predict_proba(df_cfs)).flatten()


class Face(RecourseMethod):
    """
    Implementation of FACE from Poyiadzi et.al. [1]_.

    Parameters
    ----------
    mlmodel : carla.model.MLModel
        Black-Box-Model
    hyperparams : dict
        Dictionary containing hyperparameters. See notes below for its contents.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "mode": {"knn", "epsilon"},
            Decides over type of FACE
        * "fraction": float [0 < x < 1]
            determines fraction of data set to be used to construct neighbourhood graph

    .. [1] Rafael Poyiadzi, Kacper Sokol, Raul Santos-Rodriguez, Tijl De Bie, and Peter Flach. 2020. In Proceedings
            of the AAAI/ACM Conference on AI, Ethics, and Society (AIES)
    """

    _DEFAULT_HYPERPARAMS = {
        "mode": "knn",
        "fraction": 0.05,
        "radius": 0.25,
        "n_neighbours": 50,
        "weight_function": "_optional_",
        "epsilon_constraints": 0.5,
    }

    def __init__(self, mlmodel: MLModel, hyperparams: Optional[Dict] = None) -> None:

        supported_backends = ["tensorflow", "pytorch", "sklearn"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )

        super().__init__(mlmodel)

        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )
        self.mode = checked_hyperparams["mode"]
        self.fraction = checked_hyperparams["fraction"]
        self.radius = checked_hyperparams["radius"]
        self.weight_function = checked_hyperparams["weight_function"]
        self.epsilon_constraints = checked_hyperparams["epsilon_constraints"]
        self.knn_neighbours = checked_hyperparams["n_neighbours"]

        self._immutables = encode_feature_names(
            self._mlmodel.data.immutables, self._mlmodel.feature_input_order
        )

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, x: float) -> float:
        self._radius = x

    @property
    def fraction(self) -> float:
        """
        Controls the fraction of the used dataset to build graph on.

        Returns
        -------
        float
        """
        return self._fraction

    @fraction.setter
    def fraction(self, x: float) -> None:
        if 0 < x < 1:
            self._fraction = x
        else:
            raise ValueError("Fraction has to be between 0 and 1")

    @property
    def mode(self) -> str:
        """
        Sets and changes the type of FACE. {"knn", "epsilon"}

        Returns
        -------
        str
        """
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> None:
        if mode in ["knn", "epsilon"]:
            self._mode = mode
        else:
            raise ValueError("Mode has to be either knn or epsilon")

    def get_counterfactuals(
        self, factuals: pd.DataFrame, W, G=None
    ) -> pd.DataFrame:
        # >drop< factuals from dataset to prevent duplicates,
        # >reorder< and >add< factuals to top; necessary in order to use the index
        df = self._mlmodel.data.df.copy()
        cond = df.isin(factuals).values
        df = df.drop(df[cond].index)
        # df = pd.concat([factuals, df], ignore_index=True)

        df = self._mlmodel.get_ordered_features(df)
        factuals = self._mlmodel.get_ordered_features(factuals)

        list_cfs = []
        for i in tqdm(range(factuals.shape[0])):
            cf = graph_search(
                df,
                factuals,
                W,
                G,
                self._mlmodel._preprocessor,
                i,
                self._immutables,
                self._mlmodel,
                n_neighbors=self.knn_neighbours,
                mode=self._mode,
                frac=self._fraction,
                radius=self._radius,
                weight_function=self.weight_function,
                epsilon_constraints=self.epsilon_constraints,
            )
            list_cfs.append(cf if not any(np.isnan(cf)) else factuals.values[i])
        df_cfs, Y = check_counterfactuals(self._mlmodel, list_cfs, factuals.index)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        return df_cfs, Y
