"""Turn one hike's second-by-second records into a few pace features and a small elevation/heart-rate profile.

Privacy: this module only ever receives time, distance, altitude and heart rate. GPS coordinates are never
passed in, and nothing here stores a track: the profile is ~80 points of (distance, altitude, heart rate, stop minutes).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

# Bump this whenever the definitions below change (what counts as moving/stopped, how rates are computed). Saved hikes
# and cached results made with an older version are then recognised as stale instead of being shown as if current.
TRACK_VERSION = 7

M_PER_MILE = 1609.344
FT_PER_M = 3.28084

# "Stopped" is judged from how far you actually travelled, NOT from the watch's speed field: that field reads exactly 0
# for stretches of a hike while distance keeps advancing, which made a hike look 37% stopped when it was ~13%.
MOVING_MPS = 0.5          # slow = covering less than this (about 1.1 mph) over WINDOW_S; never higher (see ADAPT_FRAC below). Deliberately
                          # generous: standing around at a summit wanders at 0.1-0.5 m/s of GPS drift. Heart rate decides if it was a rest.
                          # Progress counts height gained too (Naismith-style), so a slow scramble straight up rock counts as moving.
WINDOW_S = 45             # distance covered in a 45 s window centred on each record
MAX_STEP_S = 30           # a gap between records longer than this is a pause (timer stopped), not movement. Watches on 'smart
                          # recording' write a point only every 5-10 s, so a gap of a few seconds is normal, NOT a pause.
MIN_STOP_S = 180          # a "stop" is a break of at least 3 minutes; shorter slowdowns count as part of moving
MERGE_GAP_S = 120         # a rest interrupted by under a minute of apparent movement (GPS wobble, shuffling about) is still one stop
# Statistical approach, all relative to the hike itself rather than fixed numbers:
#  * "slow" = under ADAPT_FRAC of your own typical moving speed on that hike (capped at MOVING_MPS, floored at MIN_MOVING_MPS),
#    so a hike done at a crawl on ice isn't judged by a fast hike's yardstick.
#  * A slow stretch of 3+ minutes is a BREAK only if you actually rested: the watch timer paused, or your heart rate fell at
#    least DEFAULT_HR_DROP_SD standard deviations below where it was while moving (SD = how much your heart rate normally varies
#    on that hike, robustly measured). Otherwise (ice, scrambling) it is "slow technical": still effort, counted as moving.
# The app lets the viewer change the length and the SD number, so the raw candidates are saved with each hike.
ADAPT_FRAC, MIN_MOVING_MPS = 0.4, 0.2
CLIMB_WEIGHT, DESCENT_WEIGHT = 8.0, 3.0
DEFAULT_MIN_STOP_S = MIN_STOP_S
DEFAULT_HR_DROP_SD = 2.0
MIN_CANDIDATE_S = 60      # slow stretches at least this long are saved, so the min-stop slider can go down to 1 minute
PAUSED_BREAK_FRAC = 0.6   # a slow stretch that is mostly timer-paused is always a break
HR_BASELINE_S = 300       # "where your heart rate was while moving" = the 5 minutes before the slow stretch
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
    alt = df["a"].rolling(SMOOTH_SAMPLES, center=True, min_periods=1).mean().to_numpy()
    half = WINDOW_S / 2
    flat_m = np.interp(secs + half, secs, dist) - np.interp(secs - half, secs, dist)
    up_m = np.interp(secs + half, secs, alt) - np.interp(secs - half, secs, alt)
    # progress over the window in 'flat-ground equivalent' metres (Naismith's rule: 1 m climbed ~ 8 m walked; steep descent ~ 3 m).
    # A slow scramble up a steep slope covers little ground but a lot of climb: that is progress, not standing still.
    travelled = flat_m + CLIMB_WEIGHT * np.clip(up_m, 0, None) + DESCENT_WEIGHT * np.clip(-up_m, 0, None)
    speed = travelled / WINDOW_S
    typical = speed[speed >= MOVING_MPS]
    thresh = float(np.clip(ADAPT_FRAC * np.median(typical), MIN_MOVING_MPS, MOVING_MPS)) if len(typical) >= 30 else MOVING_MPS
    moving = (speed >= thresh) & (dt_raw <= MAX_STEP_S)

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

    cands, hr_sd = find_candidates(secs, moving, dt_raw, df["h"].to_numpy(), dist)
    # slow technical stretches (effort without rest) are still hiking: count them as moving for the pace numbers too
    moving = moving.copy()
    for c in cands:
        if not is_break(c, DEFAULT_MIN_STOP_S, DEFAULT_HR_DROP_SD):
            moving[c["first"]:c["last"] + 1] = True

    def tot(mask, arr):
        return float(np.sum(arr[mask & moving]))

    stopped_s = sum(c["secs"] for c in cands if c["secs"] >= DEFAULT_MIN_STOP_S and is_break(c, DEFAULT_MIN_STOP_S, DEFAULT_HR_DROP_SD))
    if elapsed_s:
        stopped_s = min(stopped_s, float(elapsed_s))
        out["stopped_hr"] = stopped_s / 3600
        out["moving_hr"] = (float(elapsed_s) - stopped_s) / 3600   # everything that isn't a break
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

    out["profile"] = _profile(df, dist, alt, cands, hr_sd, dt, moving, secs)
    return out


def find_candidates(secs, moving, dt_raw, hr, dist) -> tuple[list[dict], float | None]:
    """Every slow/still stretch of at least MIN_CANDIDATE_S, with what is needed to tell a rest from slow going:
    seconds, how much of it was timer-paused, and how far heart rate fell below its level while moving, in standard
    deviations of this hike's own moving heart rate. Returns (candidates, that SD in bpm)."""
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
    hr = np.asarray(hr, dtype=float)
    moving_hr_all = hr[moving & np.isfinite(hr)]
    overall = float(np.median(moving_hr_all)) if len(moving_hr_all) >= 30 else np.nan
    # typical beat-to-beat-minute variation of YOUR heart rate while moving: robust SD (1.4826 x MAD) of a 30 s smoothed series
    # How much YOUR heart rate wobbles minute to minute while moving: robust SD (1.4826 x MAD) of the 30 s average minus the
    # trailing 5 minute average, measured on the actual timeline (records are 1-8 s apart). NOT the SD of the whole hike, which
    # would mostly measure climbing vs descending and be far too large to ever see a rest in.
    sd = np.nan
    if len(moving_hr_all) >= 60:
        ser = pd.Series(np.where(np.isfinite(hr), hr, np.nan), index=pd.to_timedelta(secs, unit="s"))
        resid = (ser.rolling("30s", min_periods=3).mean() - ser.rolling("300s", min_periods=20).mean()).to_numpy()
        v = resid[moving & np.isfinite(resid)]
        sd = float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) >= 60 else np.nan
        sd = max(sd, 3.0) if np.isfinite(sd) else sd   # a very steady heart rate must not turn a 2 bpm dip into "many SD"
    out = []
    for first, last, s0, e0 in merged:
        dur = e0 - s0
        if dur < MIN_CANDIDATE_S:
            continue
        seg = slice(first, last + 1)
        gaps = dt_raw[seg]
        paused = float(np.sum(gaps[gaps > MAX_STEP_S]))
        base_mask = moving & (secs >= s0 - HR_BASELINE_S) & (secs < s0) & np.isfinite(hr)
        base = float(np.mean(hr[base_mask])) if base_mask.sum() >= 20 else np.nan   # no effort before it (start of the hike): nothing to compare, so it counts as a rest
        in_run = (secs >= s0 + dur / 2) & (secs <= e0) & np.isfinite(hr)   # second half: heart rate lags behind a rest
        if in_run.sum() < 3:
            in_run = (secs >= s0) & (secs <= e0) & np.isfinite(hr)
        drop = (base - float(np.mean(hr[in_run]))) / sd if (np.isfinite(base) and np.isfinite(sd) and in_run.sum() >= 3) else None
        out.append({"first": first, "last": last, "secs": float(dur), "paused_frac": min(paused / dur, 1.0) if dur else 0.0,
                    "hr_drop": drop, "mile": float(dist[first]) / M_PER_MILE})
    return out, (None if not np.isfinite(sd) else sd)


def is_break(c: dict, min_stop_s: float, hr_drop_sd: float) -> bool:
    """A slow stretch that is long enough AND a real rest: timer paused, no heart-rate data to judge by, or heart rate fell."""
    if c["secs"] < min_stop_s:
        return False
    if c["paused_frac"] >= PAUSED_BREAK_FRAC or c["hr_drop"] is None or hr_drop_sd <= 0:
        return True
    return c["hr_drop"] >= hr_drop_sd


def _profile(df, dist, alt, cands, hr_sd=None, dt=None, moving=None, secs=None) -> str:
    """~80 evenly spaced points along the route: miles, altitude (ft), heart rate (bpm); plus the slow stretches ("c"):
    [mile, seconds, share of it timer-paused, heart-rate drop in SDs or null]; "sd" = that SD in bpm, so stop rules can be re-applied without the .fit file."""
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
    pace = np.full(PROFILE_POINTS, np.nan)   # min/mi of MOVING time in each slice (breaks excluded), lightly smoothed
    if dt is not None and moving is not None:
        k = np.clip(np.digitize(dist, edges) - 1, 0, PROFILE_POINTS - 1)
        sec = np.bincount(k, weights=np.where(moving, dt, 0.0), minlength=PROFILE_POINTS)
        mi = np.diff(edges) / M_PER_MILE
        raw = np.where(sec >= 20, sec / 60 / mi, np.nan)
        pace = pd.Series(np.clip(raw, 0, 60)).rolling(3, center=True, min_periods=1).mean().to_numpy()
    # speed over ELAPSED time in each slice (breaks included, not smoothed): a rest drops it to ~0 where it happened
    speed = np.full(PROFILE_POINTS, np.nan)
    if secs is not None:
        ud, first = np.unique(dist, return_index=True)
        if len(ud) > 2:
            t_edge = np.interp(edges, ud, secs[first])
            speed = np.diff(edges) / M_PER_MILE / (np.maximum(np.diff(t_edge), 1.0) / 3600)
            speed = np.clip(speed, 0, 8)
    rnd = lambda arr, n: [None if not np.isfinite(x) else round(float(x), n) for x in arr]  # noqa: E731
    c = [[round(x["mile"], 2), int(round(x["secs"])), round(x["paused_frac"], 2),
          None if x["hr_drop"] is None else round(x["hr_drop"], 2)] for x in cands]
    return json.dumps({"v": TRACK_VERSION, "d": rnd(centers / M_PER_MILE, 2), "a": rnd(alt_c, 0), "h": rnd(hr_c, 0), "p": rnd(pace, 1), "v": rnd(speed, 2), "c": c,
                       "sd": None if hr_sd is None else round(hr_sd, 1)},
                      separators=(",", ":"))


def has_pace(profile_json) -> bool:
    """True if a saved profile carries the per-slice pace AND speed series (profiles made before they were added don't)."""
    try:
        p = json.loads(profile_json)
        return "p" in p and "v" in p
    except Exception:
        return False


def profile_version(profile_json) -> int:
    """Version of the definitions a saved profile was made with (0 = unknown/older, or no profile)."""
    try:
        return int(json.loads(profile_json).get("v", 0))
    except Exception:
        return 0


def _cands_from_profile(p: dict) -> list[dict]:
    return [{"mile": m, "secs": float(sec), "paused_frac": pf, "hr_drop": hd} for m, sec, pf, hd in p.get("c", [])]


def stop_totals(profile_json, min_stop_s: float = DEFAULT_MIN_STOP_S, hr_drop_sd: float = DEFAULT_HR_DROP_SD):
    """(break seconds, slow-technical seconds) for one hike under the given rules, or None if it has no saved candidates."""
    try:
        p = json.loads(profile_json)
        if "c" not in p:
            return None
    except Exception:
        return None
    cs = [c for c in _cands_from_profile(p) if c["secs"] >= min_stop_s]
    br = sum(c["secs"] for c in cs if is_break(c, min_stop_s, hr_drop_sd))
    return br, sum(c["secs"] for c in cs) - br


def apply_stop_rules(df: pd.DataFrame, min_stop_s: float, hr_drop_sd: float) -> pd.DataFrame:
    """Recompute stopped/moving hours (and add slow_technical_hr) from each hike's saved slow stretches."""
    df = df.copy()
    df["slow_technical_hr"] = float("nan")
    for i, prof in df["profile"].items():
        tt = stop_totals(prof, min_stop_s, hr_drop_sd) if isinstance(prof, str) and len(prof) > 2 else None
        if tt is None:
            continue
        total_s = float(df.at[i, "time_hr"]) * 3600
        if not np.isfinite(total_s):
            continue
        br = min(tt[0], total_s)
        df.at[i, "stopped_hr"], df.at[i, "moving_hr"] = br / 3600, (total_s - br) / 3600
        df.at[i, "slow_technical_hr"] = tt[1] / 3600
    return df


def profile_frame(profile_json: str, min_stop_s: float = DEFAULT_MIN_STOP_S, hr_drop_sd: float = DEFAULT_HR_DROP_SD) -> pd.DataFrame | None:
    """Saved profile string -> DataFrame(miles, altitude_ft, heart_rate, pace_min_mi, speed_mph, stop_min, slow_min), or None.
    stop_min = minutes of real breaks starting in that slice; slow_min = minutes of slow technical going."""
    if not isinstance(profile_json, str) or not profile_json.strip():
        return None
    try:
        p = json.loads(profile_json)
        df = pd.DataFrame({"miles": p["d"], "altitude_ft": p["a"], "heart_rate": p["h"],
                           "pace_min_mi": p.get("p") or [None] * len(p["d"])})
        df["pace_min_mi"] = pd.to_numeric(df["pace_min_mi"], errors="coerce")
        df["speed_mph"] = pd.to_numeric(pd.Series(p.get("v") or [None] * len(p["d"])), errors="coerce")
        stop, slow = np.zeros(len(df)), np.zeros(len(df))
        miles = df["miles"].to_numpy(dtype=float)
        for c in _cands_from_profile(p):
            if c["secs"] < min_stop_s:
                continue
            k = int(np.argmin(np.abs(miles - c["mile"])))
            (stop if is_break(c, min_stop_s, hr_drop_sd) else slow)[k] += c["secs"] / 60
        df["stop_min"], df["slow_min"] = stop, slow
        return df
    except Exception:
        return None
