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


# ---------------------------------------------------------------------------------------------
# Moving + stops model: uses the per-hike track data from .fit files.
#   moving time  = regression on distance + climb (cleaner than total time: breaks don't add noise)
#   stopped time = your typical stopped-to-moving ratio (rests, summit time, photos, pauses)
# Total = moving * (1 + ratio). Compared against the simple model with leave-one-out error.
# ---------------------------------------------------------------------------------------------
MIN_HIKES_FOR_MOVING_MODEL = 5
MAX_STOP_RATIO = 1.5  # ignore absurd ratios (e.g. a watch left running at a trailhead)


def _stop_ratio(d: pd.DataFrame) -> float:
    r = (d["stopped_hr"] / d["moving_hr"]).clip(0, MAX_STOP_RATIO)
    return float(r.median())


class MovingStopModel:
    kind = "moving+stops"

    def __init__(self, d: pd.DataFrame):
        self.n = len(d)
        self.fit = fit_regression(d, FEATURES, "moving_hr")
        self.stop_ratio = _stop_ratio(d)
        # leave-one-out error on TOTAL time, so it is comparable with the simple model
        errs = []
        if self.n >= MIN_HIKES_FOR_MOVING_MODEL + 1:
            for i in range(self.n):
                rest = d.drop(d.index[i])
                lr = LinearRegression().fit(rest[FEATURES], rest["moving_hr"])
                row = pd.DataFrame([d.iloc[i][FEATURES].to_list()], columns=FEATURES)
                pred = max(float(lr.predict(row)[0]), 0.1) * (1 + _stop_ratio(rest))
                errs.append(float(d.iloc[i][TARGET]) - pred)
        self.loo_errors = np.array(errs)
        self.loo_mae = float(np.mean(np.abs(errs))) if errs else None
        resid = d[TARGET] - np.array([self.hours(r.distance_mi, r.ascent_ft) for r in d.itertuples()])
        in_sample = float(np.sqrt((resid ** 2).sum() / max(self.n - 3, 1)))
        self.spread = float(np.sqrt(np.mean(self.loo_errors ** 2))) if errs else in_sample

    @classmethod
    def try_build(cls, hikes: pd.DataFrame):
        """-> (model, rows_used) or (None, rows_used) when there aren't enough hikes with track data."""
        need = FEATURES + [TARGET, "moving_hr", "stopped_hr"]
        if not set(need) <= set(hikes.columns):
            return None, 0
        d = hikes[need].apply(pd.to_numeric, errors="coerce").dropna()
        d = d[(d[TARGET] > 0) & (d["distance_mi"] > 0) & (d["moving_hr"] > 0)].reset_index(drop=True)
        return (cls(d) if len(d) >= MIN_HIKES_FOR_MOVING_MODEL else None), len(d)

    def parts(self, distance_mi: float, ascent_ft: float) -> tuple[float, float]:
        moving = max(predict(self.fit, {"distance_mi": distance_mi, "ascent_ft": ascent_ft}), 0.1)
        return moving, moving * self.stop_ratio

    def hours(self, distance_mi: float, ascent_ft: float) -> float:
        m, s = self.parts(distance_mi, ascent_ft)
        return max(m + s, 0.25)


def simple_loo_mae(d: pd.DataFrame) -> float | None:
    """Leave-one-out miss of the simple distance+climb -> total-time regression, on the same rows."""
    d = d[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < MIN_HIKES_FOR_MOVING_MODEL + 1:
        return None
    pred = cross_val_predict(LinearRegression(), d[FEATURES], d[TARGET], cv=LeaveOneOut())
    return float(np.mean(np.abs(d[TARGET] - pred)))
