"""Save / restore the user's peak tags and include/exclude choices as a small CSV."""
from __future__ import annotations

import io

import pandas as pd

SEP = "|"  # peak names can contain commas ("Wildcat, A peak"), so don't split on commas
COLUMNS = ["key", "date", "title", "distance_mi", "nh4000_peaks", "include"]


def tags_to_csv(tags: dict) -> bytes:
    rows = [{
        "key": k,
        "date": v.get("date", ""),
        "title": v.get("title", ""),
        "distance_mi": v.get("distance_mi", ""),
        "nh4000_peaks": SEP.join(v.get("peaks", [])),
        "include": bool(v.get("include", True)),
    } for k, v in sorted(tags.items())]
    return pd.DataFrame(rows, columns=COLUMNS).to_csv(index=False).encode("utf-8")


def tags_from_csv(data: bytes) -> dict:
    df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    if "key" not in df.columns:
        raise ValueError("This doesn't look like a saved-tags file (no 'key' column).")
    out = {}
    for _, r in df.iterrows():
        peaks = [p for p in r.get("nh4000_peaks", "").split(SEP) if p]
        out[r["key"]] = {
            "peaks": peaks,
            "include": str(r.get("include", "True")).strip().lower() not in ("false", "0", "no"),
            "date": r.get("date", ""), "title": r.get("title", ""), "distance_mi": r.get("distance_mi", ""),
        }
    return out


# ------------------------------------------------------------ saved hikes file
# One condensed CSV holding every activity's summary numbers PLUS the user's tags and include/exclude choices.
# Upload it next visit together with only the NEW files. No GPS, name or weight, same as the rest of the app.
HIKE_FILE_COLUMNS = [
    "start_local", "date", "title", "activity_type", "distance_mi", "ascent_ft", "descent_ft",
    "time_hr", "avg_hr", "max_hr", "calories", "nh4000_peaks", "include",
]
_NUM = ["distance_mi", "ascent_ft", "descent_ft", "time_hr", "avg_hr", "max_hr", "calories"]


def hikes_to_csv(activities: pd.DataFrame, tags: dict, key_fn) -> bytes:
    """activities = the full table (garmin_io.HIKE_COLUMNS); tags = the app's tag store; key_fn = activity_key."""
    out = activities.copy()
    keys = [key_fn(a, d, m) for a, d, m in zip(out["start_local"], out["date"], out["distance_mi"])]
    out["nh4000_peaks"] = [SEP.join(tags.get(k, {}).get("peaks", [])) for k in keys]
    out["include"] = [bool(tags.get(k, {}).get("include", True)) for k in keys]
    return out[HIKE_FILE_COLUMNS].to_csv(index=False).encode("utf-8")


def hikes_from_csv(data: bytes) -> pd.DataFrame:
    """Read a saved hikes file -> table with garmin_io columns plus 'peaks' (list) and 'include' (bool)."""
    df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    missing = [c for c in ("start_local", "distance_mi", "time_hr") if c not in df.columns]
    if missing:
        raise ValueError("This doesn't look like a saved hikes file (missing: " + ", ".join(missing) + ").")
    for c in HIKE_FILE_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    out = pd.DataFrame({
        "start_local": pd.to_datetime(df["start_local"], errors="coerce"),
        "date": pd.to_datetime(df["date"], errors="coerce"),
        "title": df["title"], "match": "saved", "activity_type": df["activity_type"].replace("", "unknown"),
    })
    for c in _NUM:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    out["date"] = out["date"].fillna(out["start_local"].dt.normalize())
    out["peaks"] = [[p for p in s.split(SEP) if p] for s in df["nh4000_peaks"]]
    out["include"] = ~df["include"].str.strip().str.lower().isin(["false", "0", "no"])
    return out.dropna(subset=["start_local"]).reset_index(drop=True)
