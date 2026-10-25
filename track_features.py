"""Turn one hike's second-by-second records into a few pace features and a small elevation/heart-rate profile.

Privacy: this module only ever receives time, distance, altitude and heart rate. GPS coordinates are never
passed in, and nothing here stores a track: the profile is ~80 points of (distance, altitude, heart rate, stop minutes).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

M_PER_MILE = 1609.344
FT_PER_M = 3.28084

# "Stopped" is judged from how far you actually travelled, NOT from the watch's speed field: that field reads exactly 0
# for stretches of a hike while distance keeps advancing, which made a hike look 37% stopped when it was ~13%.
MOVING_MPS = 0.3          # covering less than this (about 0.7 mph) over WINDOW_S counts as stopped
WINDOW_S = 30             # distance covered in a 30 s window centred on each record
MAX_STEP_S = 5            # a gap between records longer than this is a pause (timer stopped), not movement
MIN_STOP_S = 180          # a "stop" is a break of at least 3 minutes; shorter slowdowns count as part of moving
MERGE_GAP_S = 30          # a rest interrupted by under 30 s of movement (GPS wobble) is still one stop
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


def extract(ts, dist_m, alt_m, hr, elapsed_s: float) -> dict:
    """ts: datetimes; the rest are equal-length sequences (None/NaN allowed). Returns a dict of features."""
    out = empty_features()
    df = pd.DataFrame({"t": ts, "d": dist_m, "a": alt_m, "h": hr})
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    for c in ("d", "a", "h"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["t", "d", "a"]).drop_duplicates(subset="t").sort_values("t").reset_index(drop=True)
    if len(df) < 60 or df["d"].max() < 200:
        return out

    dt_raw = df["t"].diff().dt.total_seconds().fillna(0).clip(lower=0).to_numpy()
    dt = np.minimum(dt_raw, MAX_STEP_S)
    dist = np.maximum.accumulate(df["d"].to_numpy())
    secs = (df["t"] - df["t"].iloc[0]).dt.total_seconds().to_numpy()
    travelled = np.interp(secs + WINDOW_S / 2, secs, dist) - np.interp(secs - WINDOW_S / 2, secs, dist)
    moving = (travelled / WINDOW_S >= MOVING_MPS) & (dt_raw <= MAX_STEP_S)

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

    stops = find_stops(secs, moving, dt_raw)
    stopped_s = sum(d for _, _, d in stops)
    if elapsed_s:
        stopped_s = min(stopped_s, float(elapsed_s))
        out["stopped_hr"] = stopped_s / 3600
        out["moving_hr"] = (float(elapsed_s) - stopped_s) / 3600   # everything that isn't a 3+ minute stop
    else:
        out["moving_hr"] = float(np.sum(dt[moving])) / 3600

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

    out["profile"] = _profile(df, dist, alt, stops)
    return out


def find_stops(secs, moving, dt_raw) -> list[tuple[int, int, float]]:
    """Breaks of at least MIN_STOP_S: [(first_record, last_record, seconds)]. Paused-timer gaps count as stopped time."""
    runs, i, n = [], 0, len(secs)
    while i < n:
        if moving[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and not moving[j + 1]:
            j += 1
        runs.append([i, j, secs[i] - dt_raw[i], secs[j]])   # [first, last, start_s, end_s]
        i = j + 1
    merged = []
    for r in runs:
        if merged and r[2] - merged[-1][3] <= MERGE_GAP_S:
            merged[-1][1], merged[-1][3] = r[1], r[3]
        else:
            merged.append(r)
    return [(a, b, e - s) for a, b, s, e in merged if e - s >= MIN_STOP_S]


def _profile(df, dist, alt, stops) -> str:
    """~80 evenly spaced points along the route: miles, altitude (ft), heart rate (bpm), minutes of 3+ minute stops there."""
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
    stop_min = np.zeros(PROFILE_POINTS)
    for first, _last, secs_stopped in stops:   # each stop is credited, whole, to where it began
        k = int(np.clip(np.digitize(dist[first], edges) - 1, 0, PROFILE_POINTS - 1))
        stop_min[k] += secs_stopped / 60
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
