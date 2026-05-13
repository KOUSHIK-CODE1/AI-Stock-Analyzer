"""Machine-learning models for next-day stock direction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None


@dataclass
class MLResult:
    """ML training and prediction outputs."""

    model_name: str
    accuracy: float
    confusion_matrix: pd.DataFrame
    prediction_frame: pd.DataFrame
    feature_importance: pd.DataFrame
    latest_prediction: str
    latest_probability: float
    trained_model: Any


class StockMovementPredictor:
    """Train models that predict whether tomorrow closes higher than today."""

    feature_columns = [
        "RSI",
        "MACD",
        "Signal_Line",
        "MACD_Histogram",
        "Volume",
        "EMA_Diff",
        "Momentum_5",
        "Momentum_10",
        "Daily_Return",
        "SMA_20",
        "SMA_50",
        "ATR",
        "StochRSI",
        "Volume_Ratio",
        "Volatility_20",
        "Close_to_SMA20",
        "Close_to_SMA50",
    ]

    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        self.test_size = test_size
        self.random_state = random_state

    def train(self, data: pd.DataFrame) -> MLResult:
        dataset = self._build_dataset(data)
        if len(dataset) < 80:
            raise ValueError(
                "Not enough clean rows for ML training. Try a longer period such as 2y or 5y."
            )

        features = [column for column in self.feature_columns if column in dataset.columns]
        X = dataset[features]
        y = dataset["Target"]

        split_index = int(len(dataset) * (1 - self.test_size))
        split_index = min(max(split_index, 40), len(dataset) - 10)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        models = self._build_models()
        trained: list[tuple[str, Any, float, np.ndarray, np.ndarray]] = []
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                probabilities = self._bullish_probabilities(model, X_test)
                accuracy = accuracy_score(y_test, predictions)
                trained.append((name, model, accuracy, predictions, probabilities))
            except Exception:
                continue

        if not trained:
            raise ValueError("All ML models failed to train on the current data.")

        best_name, best_model, best_accuracy, best_predictions, best_probabilities = max(
            trained, key=lambda item: item[2]
        )

        prediction_frame = pd.DataFrame(
            {
                "Actual": y_test,
                "Predicted": best_predictions,
                "Bullish_Probability": best_probabilities,
            },
            index=X_test.index,
        )
        prediction_frame["Prediction_Label"] = np.where(
            prediction_frame["Predicted"].eq(1), "Bullish", "Bearish"
        )

        latest_features = data[features].replace([np.inf, -np.inf], np.nan).dropna().tail(1)
        if latest_features.empty:
            latest_prediction = "Unknown"
            latest_probability = 0.0
        else:
            latest_raw = int(best_model.predict(latest_features)[0])
            latest_probability = float(
                self._bullish_probabilities(best_model, latest_features)[0]
            )
            latest_prediction = "Bullish" if latest_raw == 1 else "Bearish"

        matrix = confusion_matrix(y_test, best_predictions, labels=[0, 1])
        matrix_frame = pd.DataFrame(
            matrix,
            index=["Actual Bearish", "Actual Bullish"],
            columns=["Predicted Bearish", "Predicted Bullish"],
        )

        return MLResult(
            model_name=best_name,
            accuracy=float(best_accuracy),
            confusion_matrix=matrix_frame,
            prediction_frame=prediction_frame,
            feature_importance=self._feature_importance(best_model, features),
            latest_prediction=latest_prediction,
            latest_probability=latest_probability,
            trained_model=best_model,
        )

    def save_model(self, result: MLResult, path: str | bytes | Any) -> None:
        """Persist the trained model with joblib."""
        joblib.dump(result.trained_model, path)

    def _build_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = data.copy()
        frame["Target"] = (frame["Close"].shift(-1) > frame["Close"]).astype(int)
        frame["Trend_Target_5D"] = (frame["Close"].shift(-5) > frame["Close"]).astype(int)

        features = [column for column in self.feature_columns if column in frame.columns]
        dataset = frame[features + ["Target", "Trend_Target_5D"]].replace(
            [np.inf, -np.inf], np.nan
        )
        dataset = dataset.dropna()
        return dataset.iloc[:-1] if len(dataset) else dataset

    def _build_models(self) -> dict[str, Any]:
        models: dict[str, Any] = {
            "Logistic Regression": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=self.random_state,
                        ),
                    ),
                ]
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=250,
                max_depth=8,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=self.random_state,
                n_jobs=-1,
            ),
        }

        if XGBClassifier is not None:
            models["XGBoost"] = XGBClassifier(
                n_estimators=250,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=self.random_state,
            )

        return models

    @staticmethod
    def _bullish_probabilities(model: Any, features: pd.DataFrame) -> np.ndarray:
        if not hasattr(model, "predict_proba"):
            return model.predict(features).astype(float)

        probabilities = model.predict_proba(features)
        classes = list(getattr(model, "classes_", []))
        if not classes and hasattr(model, "named_steps"):
            classes = list(model.named_steps["model"].classes_)

        if 1 in classes:
            class_index = classes.index(1)
            return probabilities[:, class_index]

        only_class = classes[0] if classes else 0
        return np.ones(len(features)) if only_class == 1 else np.zeros(len(features))

    @staticmethod
    def _feature_importance(model: Any, features: list[str]) -> pd.DataFrame:
        raw_model = model.named_steps["model"] if hasattr(model, "named_steps") else model

        if hasattr(raw_model, "feature_importances_"):
            importance = raw_model.feature_importances_
        elif hasattr(raw_model, "coef_"):
            importance = np.abs(raw_model.coef_[0])
        else:
            importance = np.zeros(len(features))

        frame = pd.DataFrame({"Feature": features, "Importance": importance})
        return frame.sort_values("Importance", ascending=False).reset_index(drop=True)
