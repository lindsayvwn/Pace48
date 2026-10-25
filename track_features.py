"""Turn one hike's second-by-second records into a few pace features and a small elevation/heart-rate profile.

Privacy: this module only ever receives time, distance, altitude, speed and heart rate. GPS coordinates are never
passed in, and nothing here stores a track: the profile is ~80 points of (distance, altitude, heart rate, stop minutes).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

M_PER_MILE = 1609.344
FT_PER_M = 3.28084

MOVING_MPS = 0.3          # below this speed you're counted as stopped (about 0.7 mph)
MAX_STEP_S = 5            # a gap between records longer than this is a pause, not movement
SMOOTH_SAMPLES = 15       # altitude smoothing (barometric altitude is noisy second to second)
GRADE_HALF_WINDOW_M = 50  # grade is measured over ~100 m of trail
MODERATE, STEEP = 5.0, 15.0   # grade thresholds, percent
PROFILE_POINTS = 80
MIN_FLAT_MI = 0.25        # need at least this much level walking for a flat pace
MIN_UP_S, MIN_STEEP_S, MIN_DOWN_S = 600, 300, 600  # minimum moving time in a bin for its rate to be reported

TRACK_NUMERIC = [
    "moving_hr", "stopped_hr", "flat_pace_min_mi", "climb_rate_fthr",
    "steep_climb_rate_fthr", "descent_rate_fthr", "steep_ascent_pct",
]
TRACK_COLUMNS = TRACK_NUMERIC + ["profile"]
_NAN = float("nan")


def empty_features() -> dict:
    return {**{k: _NAN for k in TRACK_NUMERIC}, "profile": ""}


def extract(ts, dist_m, alt_m, speed_mps, hr, elapsed_s: float) -> dict:
    """ts: datetimes/seconds; the rest are equal-length sequences (None/NaN allowed). Returns a dict of features."""
    out = empty_features()
    df = pd.DataFrame({"t": ts, "d": dist_m, "a": alt_m, "v": speed_mps, "h": hr})
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    for c in ("d", "a", "v", "h"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["t", "d", "a"]).reset_index(drop=True)
    if len(df) < 60 or df["d"].max() < 200:
        return out

    dt_raw = df["t"].diff().dt.total_seconds().fillna(0).clip(lower=0).to_numpy()
    dt = np.minimum(dt_raw, MAX_STEP_S)
    v = df["v"].fillna(0).to_numpy()
    moving = (v >= MOVING_MPS) & (dt_raw <= MAX_STEP_S)

    dist = np.maximum.accumulate(df["d"].to_numpy())
    alt = df["a"].rolling(SMOOTH_SAMPLES, center=True, min_periods=1).mean().to_numpy()

    # grade at each record: altitude change over +/-50 m of distance
    g = pd.DataFrame({"d": dist, "a": alt}).groupby("d", as_index=False).mean()
    xs, ys = g["d"].to_numpy(), g["a"].to_numpy()
    lo = np.clip(dist - GRADE_HALF_WINDOW_M, xs[0], xs[-1])
    hi = np.clip(dist + GRADE_HALF_WINDOW_M, xs[0], xs[-1])
    span = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        grade = np.where(span >= 20, (np.interp(hi, xs, ys) - np.interp(lo, xs, ys)) / span * 100, np.nan)

    ddist = np.diff(dist, prepend=dist[0])
    dalt = np.diff(alt, prepend=alt[0])
    bins = {
        "steep_up": grade >= STEEP,
        "up": grade >= MODERATE,                      # includes steep
        "flat": np.abs(grade) < MODERATE,
        "down": grade <= -MODERATE,
    }

    def tot(mask, arr):
        return float(np.sum(arr[mask & moving]))

    moving_s = float(np.sum(dt[moving]))
    out["moving_hr"] = moving_s / 3600
    out["stopped_hr"] = max(float(elapsed_s) - moving_s, 0.0) / 3600 if elapsed_s else _NAN

    flat_mi = tot(bins["flat"], ddist) / M_PER_MILE
    if flat_mi >= MIN_FLAT_MI:
        out["flat_pace_min_mi"] = tot(bins["flat"], dt) / 60 / flat_mi

    up_s = tot(bins["up"], dt)
    up_ft = tot(bins["up"], np.clip(dalt, 0, None)) * FT_PER_M
    if up_s >= MIN_UP_S and up_ft > 0:
        out["climb_rate_fthr"] = up_ft / (up_s / 3600)
    st_s = tot(bins["steep_up"], dt)
    st_ft = tot(bins["steep_up"], np.clip(dalt, 0, None)) * FT_PER_M
    if st_s >= MIN_STEEP_S and st_ft > 0:
        out["steep_climb_rate_fthr"] = st_ft / (st_s / 3600)
    dn_s = tot(bins["down"], dt)
    dn_ft = -tot(bins["down"], np.clip(dalt, None, 0)) * FT_PER_M
    if dn_s >= MIN_DOWN_S and dn_ft > 0:
        out["descent_rate_fthr"] = dn_ft / (dn_s / 3600)
    all_up_m = float(np.sum(np.clip(dalt, 0, None)[moving]))
    if all_up_m > 0:
        out["steep_ascent_pct"] = 100 * tot(bins["steep_up"], np.clip(dalt, 0, None)) / all_up_m

    out["profile"] = _profile(df, dist, alt, dt_raw, moving)
    return out


def _profile(df, dist, alt, dt_raw, moving) -> str:
    """~80 evenly spaced points along the route: miles, altitude (ft), heart rate (bpm), minutes stopped there."""
    total = dist[-1]
    edges = np.linspace(0, total, PROFILE_POINTS + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    g = pd.DataFrame({"d": dist, "a": alt}).groupby("d", as_index=False).mean()
    alt_c = np.interp(centers, g["d"], g["a"]) * FT_PER_M
    hr = df["h"]
    if hr.notna().sum() > 10:
        hs = hr.rolling(31, center=True, min_periods=1).mean()
        gh = pd.DataFrame({"d": dist, "h": hs.to_numpy()}).dropna().groupby("d", as_index=False).mean()
        hr_c = np.interp(centers, gh["d"], gh["h"])
    else:
        hr_c = np.full(PROFILE_POINTS, np.nan)
    idx = np.clip(np.digitize(dist, edges) - 1, 0, PROFILE_POINTS - 1)
    stop_min = np.bincount(idx, weights=np.where(moving, 0.0, dt_raw), minlength=PROFILE_POINTS) / 60
    rnd = lambda arr, n: [None if not np.isfinite(x) else round(float(x), n) for x in arr]  # noqa: E731
    return json.dumps({"d": rnd(centers / M_PER_MILE, 2), "a": rnd(alt_c, 0), "h": rnd(hr_c, 0), "s": rnd(stop_min, 1)},
                      separators=(",", ":"))


def profile_frame(profile_json: str) -> pd.DataFrame | None:
    """Saved profile string -> DataFrame(miles, altitude_ft, heart_rate, stop_min), or None."""
    if not isinstance(profile_json, str) or not profile_json.strip():
        return None
    try:
        p = json.loads(profile_json)
        return pd.DataFrame({"miles": p["d"], "altitude_ft": p["a"], "heart_rate": p["h"], "stop_min": p["s"]})
    except Exception:
        return None
