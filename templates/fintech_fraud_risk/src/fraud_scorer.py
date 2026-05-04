"""
XGBoost fraud scoring class for Ray Data map_batches.
"""
import os

import numpy as np
import xgboost as xgb

from src.feature_engineering import FEATURE_COLUMNS
from src.paths import get_demo_base_dir


class FraudScorer:
    """Stateful batch scorer for the fraud detection pipeline.

    Loaded once per worker via Ray Data's actor pool. Scores transaction
    batches and assigns risk tiers.
    """

    def __init__(self, model_path: str | None = None):
        path = model_path or os.path.join(get_demo_base_dir(), "model", "fraud_model.json")
        self.model = xgb.Booster()
        self.model.load_model(path)
        self.feature_columns = FEATURE_COLUMNS

    def __call__(self, batch: dict) -> dict:
        """Score a batch of feature-engineered transactions.

        Adds:
          - fraud_probability: float [0, 1]
          - risk_tier: 'low' / 'medium' / 'high' / 'critical'
          - should_alert: bool (critical tier)
        """
        features = np.column_stack(
            [batch[col].astype(np.float64) for col in self.feature_columns]
        )
        dmatrix = xgb.DMatrix(features, feature_names=self.feature_columns)
        probabilities = self.model.predict(dmatrix)

        batch["fraud_probability"] = probabilities
        batch["risk_tier"] = np.where(
            probabilities > 0.8, "critical",
            np.where(probabilities > 0.5, "high",
            np.where(probabilities > 0.1, "medium", "low")),
        )
        batch["should_alert"] = probabilities > 0.8
        return batch
