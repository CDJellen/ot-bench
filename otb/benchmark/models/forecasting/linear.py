from typing import Union

import numpy as np

from otb.benchmark.models.forecasting.base_model import BaseForecastingModel

class LinearForecastingModel(BaseForecastingModel):

    def __init__(self, name: str, target_name: str, window_size: int, forecast_horizon: int, **kwargs):
        super().__init__(name, target_name, window_size, forecast_horizon, **kwargs)

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        # maintain the same interface as the other models
        pass

    def predict(self, X: 'pd.DataFrame'):
        """Forecast the cn2 by fitting a line using the lagged values."""
        # obtain the lagged values of the target variable
        X = X[[c for c in X.columns if c.startswith(self.target_name)]]

        # develop a prediction for each row in X
        preds = []
        for i in range(len(X)):
            # fit a line to the lagged values
            lagged_values = X.iloc[i, :].values
            A = np.vstack([np.arange(len(lagged_values)), np.ones(len(lagged_values))]).T
            m, b = np.linalg.lstsq(A, lagged_values, rcond=None)[0]

            # predict the next value in the sequence
            pred = m * len(lagged_values) + b
            preds.append(pred)

        return np.array(preds)