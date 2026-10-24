"""Pace model: train on the user's hikes, predict hours for a route."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

# One definition of names + order, used for BOTH training and prediction.
FEATURES = ["distance_mi", "ascent_ft"]
TARGET = "time_hr"

MIN_HIKES_FOR_REGRESSION = 5


def naismith_hours(distance_mi, ascent_ft):
    """Classic rule of thumb: 1 h per 3 miles + 1 h per 2,000 ft climbed."""
    return np.asarray(distance_mi) / 3.0 + np.asarray(ascent_ft) / 2000.0


def fit_regression(df: pd.DataFrame, features: list[str], target: str) -> dict:
    data = df[features + [target]].apply(pd.to_numeric, errors="coerce").dropna()
    X, y = data[features], data[target]
    model = LinearRegression().fit(X, y)
    resid = y - model.predict(X)
    dof = max(len(y) - len(features) - 1, 1)
    out = {
        "model": model, "features": features, "target": target, "n": len(y),
        "r2": model.score(X, y), "rmse": float(np.sqrt((resid ** 2).sum() / dof)),
        "loo_mae": None,
    }
    if len(y) >= 6:
        pred = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
        out["loo_mae"] = float(np.mean(np.abs(y - pred)))
    return out


def predict(fit: dict, values: dict) -> float:
    """values = {feature_name: number}. Column names/order are rebuilt from fit['features']."""
    row = pd.DataFrame([[values[f] for f in fit["features"]]], columns=fit["features"])
    return float(fit["model"].predict(row)[0])


class PaceModel:
    """Regression when there's enough data, else Naismith scaled to the user's pace."""

    def __init__(self, hikes: pd.DataFrame):
        d = hikes[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
        d = d[(d[TARGET] > 0) & (d["distance_mi"] > 0)]
        self.n = len(d)
        if self.n >= MIN_HIKES_FOR_REGRESSION:
            self.kind = "regression"
            self.fit = fit_regression(d, FEATURES, TARGET)
            self.spread = self.fit["rmse"]
        else:
            self.kind = "naismith"
            self.fit = None
            base = naismith_hours(d["distance_mi"], d["ascent_ft"]) if self.n else np.array([])
            self.factor = float(np.mean(d[TARGET] / base)) if self.n else 1.0
            self.spread = 0.25 * max(self.factor, 1.0)  # rough, flagged as such in the UI

    def hours(self, distance_mi: float, ascent_ft: float) -> float:
        if self.kind == "regression":
            return max(predict(self.fit, {"distance_mi": distance_mi, "ascent_ft": ascent_ft}), 0.25)
        return float(naismith_hours(distance_mi, ascent_ft) * self.factor)
