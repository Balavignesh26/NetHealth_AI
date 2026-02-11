import numpy as np
from sklearn.linear_model import LinearRegression
from typing import List, Dict, Tuple, Any
import pandas as pd

class Predictor:
    def __init__(self):
        self.model = LinearRegression()

    def predict_next(self, history: List[float], lookahead: int = 1) -> Tuple[float, str]:
        """
        Predicts the next value based on the provided history.
        Returns: (predicted_value, trend_description)
        """
        if len(history) < 2:
            return (history[-1] if history else 0.0, "Insufficient Data")

        # Prepare X (time steps) and y (values)
        X = np.array(range(len(history))).reshape(-1, 1)
        y = np.array(history)

        self.model.fit(X, y)

        # Predict next step
        next_step = np.array([[len(history) + lookahead - 1]])
        prediction = self.model.predict(next_step)[0]

        # Determine Trend
        slope = self.model.coef_[0]
        if slope > 0.5:
            trend = "Increasing Rapidly 📈"
        elif slope > 0.1:
            trend = "Increasing ↗️"
        elif slope < -0.5:
            trend = "Decreasing Rapidly 📉"
        elif slope < -0.1:
            trend = "Decreasing ↘️"
        else:
            trend = "Stable ➡️"

        return (round(prediction, 2), trend)

    def forecast_asset_metrics(self, df: pd.DataFrame, asset_id: str, metric_name: str) -> Dict[str, Any]:
        """
        Extracts history for a specific asset/metric and forecasts the next value.
        """
        # Filter for asset and metric
        mask = (df['asset_id'] == asset_id) & (df['metric_name'] == metric_name)
        series = df[mask].sort_values('timestamp')['value'].values

        if len(series) == 0:
            return {}

        # Use last 10 points for recent trend
        history = series[-10:] if len(series) > 10 else series
        
        pred, trend = self.predict_next(history)
        
        return {
            "current": series[-1],
            "prediction": pred,
            "trend": trend,
            "metric": metric_name
        }
