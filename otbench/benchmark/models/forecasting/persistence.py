from typing import Union

import numpy as np

from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel


class PersistenceForecastingModel(BaseForecastingModel):
    """A model which predicts the most recent value of the target variable."""

    def __init__(self, name: str, target_name: str, window_size: int, forecast_horizon: int, **kwargs):
        super().__init__(name, target_name, window_size, forecast_horizon, **kwargs)

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Maintain the same interface as the other models."""
        pass

    def predict(self, X: 'pd.DataFrame'):
        # predict the most recent (past) observed target for each entry in X
        # X contains some number of lagged values of the target variable
        # we will use the mean of these lagged values as our prediction
        most_recent_observation = X[f"{self.target_name} (t-1)"].values

        return np.array(most_recent_observation)
