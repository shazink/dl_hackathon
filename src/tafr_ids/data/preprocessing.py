"""Fit-once E1 development transformer; later data can only transform."""

import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tafr_ids.data.inspection import CATEGORICAL, FEATURES


class FrozenPreprocessor:
    def __init__(self):
        self._transformer = None

    @staticmethod
    def _check(features):
        if list(features.columns) != FEATURES:
            raise ValueError("Unexpected feature schema or leakage columns")
        numeric = [name for name in FEATURES if is_numeric_dtype(features[name])]
        categorical = [name for name in FEATURES if name not in numeric]
        if set(categorical) != CATEGORICAL:
            raise ValueError("Unexpected feature types")
        if np.isinf(features[numeric].to_numpy(dtype=np.float64)).any():
            raise ValueError("Infinite numeric input")
        return numeric, categorical

    def fit_e1(self, features, *, partition, experience):
        if self._transformer is not None:
            raise ValueError("Preprocessor is frozen; refitting is prohibited")
        if partition != "development" or experience != 1:
            raise ValueError("Fit requires E1 development features")
        numeric, categorical = self._check(features)
        if features.empty or features.isna().all().any():
            raise ValueError("E1 must supply observations for every imputation statistic")
        transformer = ColumnTransformer([
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32))]), categorical),
        ], remainder="drop", sparse_threshold=0)
        transformer.fit(features)
        self._transformer = transformer
        self.feature_names_ = tuple(transformer.get_feature_names_out())
        self.fit_rows_ = len(features)
        return self

    def transform(self, features):
        if self._transformer is None:
            raise ValueError("Preprocessor has not been fitted on E1")
        self._check(features)
        with np.errstate(over="ignore", invalid="ignore"):
            result = self._transformer.transform(features).astype(np.float32)
        if not np.isfinite(result).all() or result.shape[1] != len(self.feature_names_):
            raise ValueError("Nonfinite output or changed dimensionality")
        return result
