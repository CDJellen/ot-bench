from typing import Union

import numpy as np

from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel


class LinearForecastingModel(BaseForecastingModel):
    """A model that fits a line to the lagged values of the target variable."""

    def __init__(self, name: str, target_name: str, window_size: int, forecast_horizon: int, **kwargs):
        super().__init__(name, target_name, window_size, forecast_horizon, **kwargs)

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Maintain the same interface as the other models."""
        pass

    def predict(self, X: 'pd.DataFrame'):
        """Forecast the cn2 by fitting a line using the lagged values."""
        # obtain the lagged values of the target variable
        X = X[[c for c in X.columns if c.startswith(self.target_name)]]

        # develop a prediction for each row in X
        preds = []
        for i in range(len(X)):
            lagged_values = X.iloc[i, :]
            data = lagged_values.to_numpy()
            timestamps = np.arange(len(data))
            
            _, (m, b) = self._interpolate_and_fit(data=data, timestamps=timestamps)
            
            # predict the next value at the forecast horizon
            pred = m * (len(lagged_values) + self.forecast_horizon) + b
            preds.append(pred)

        return np.array(preds)

    def _interpolate_and_fit(self, data: 'np.ndarray', timestamps: 'np.ndarray'):
        """Interpolates missing values and fits a line to non-null data points.

        Args:
            data (numpy.ndarray): The data array with potential missing values.
            timestamps (numpy.ndarray): The corresponding timestamps for the data points.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The interpolated data array and coefficients of the line of best fit.
        """
        valid_indices = ~np.isnan(data)  
        valid_data = data[valid_indices]
        valid_timestamps = timestamps[valid_indices]

        if np.any(~valid_indices):  # Check if there are missing values
            interpolated_data = np.interp(timestamps, valid_timestamps, valid_data)
        else:
            interpolated_data = data 
        coefficients = np.polyfit(valid_timestamps, valid_data, 1)
        
        return interpolated_data, coefficients
