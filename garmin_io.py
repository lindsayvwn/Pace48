"""Read Garmin exports (.fit, .zip of .fit files, or Garmin Connect Activities.csv)
into ONE tidy table with fixed column names, in a fixed order.

Privacy: only the handful of numeric summary fields below are kept. GPS tracks,
your name, weight, device serial etc. inside a .fit file are never read into the table.
"""
from __future__ import annotations

import io
import re
import zipfile

import pandas as pd

M_PER_MILE = 1609.344
FT_PER_M = 3.28084

# Same columns, same order, every time (also the order used for model training).
HIKE_COLUMNS = [
    "date", "title", "source", "activity_type",
    "distance_mi", "ascent_ft", "descent_ft", "time_hr",
    "avg_hr", "max_hr", "calories",
]

MAX_ZIP_MEMBERS = 2000
MAX_MEMBER_BYTES = 80 * 1024 * 1024


# ---------------------------------------------------------------- .fit files
def parse_fit(data: bytes, name: str = "activity.fit") -> dict | None:
    """Return one row (dict) from the 'session' summary of a .fit file, or None."""
    import fitdecode

    try:
        with fitdecode.FitReader(io.BytesIO(data)) as reader:
            for frame in reader:
                if frame.frame_type != fitdecode.FIT_FRAME_DATA or frame.name != "session":
                    continue
                g = lambda k: frame.get_value(k, fallback=None)  # noqa: E731
                dist_m, asc_m, desc_m = g("total_distance"), g("total_ascent"), g("total_descent")
                secs = g("total_timer_time") or g("total_elapsed_time")
                start = g("start_time")
                if dist_m is None or secs is None:
                    return None
                sport = " ".join(str(x) for x in (g("sport"), g("sub_sport")) if x and x != "generic")
                return {
                    "date": pd.Timestamp(start).tz_convert("America/New_York").tz_localize(None).normalize()
                    if start is not None else pd.NaT,
                    "title": name,
                    "source": "fit",
                    "activity_type": sport or "unknown",
                    "distance_mi": dist_m / M_PER_MILE,
                    "ascent_ft": (asc_m or 0) * FT_PER_M,
                    "descent_ft": (desc_m or 0) * FT_PER_M,
                    "time_hr": secs / 3600,
                    "avg_hr": g("avg_heart_rate"),
                    "max_hr": g("max_heart_rate"),
                    "calories": g("total_calories"),
                }
    except Exception:
        return None
    return None


# ----------------------------------------------------------- Activities.csv
_ALIASES = {
    "activity_type": ["activity type"],
    "date": ["date"],
    "title": ["title"],
    "distance": ["distance"],
    "ascent": ["total ascent"],
    "descent": ["total descent"],
    "time": ["time", "moving time"],
    "elapsed": ["elapsed time"],
    "avg_hr": ["avg hr"],
    "max_hr": ["max hr"],
    "calories": ["calories"],
}


def _to_num(s: pd.Series) -> pd.Series:
    """'1,234' -> 1234 ; '--' / '' -> NaN."""
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def _to_hours(s: pd.Series) -> pd.Series:
    """'02:13:45' or '1:05:30.5' -> hours (float). '--' -> NaN."""
    clean = s.astype(str).str.strip().replace({"--": None, "": None, "nan": None})
    # Garmin gives H:MM:SS ; pandas wants HH:MM:SS
    clean = clean.map(lambda x: x if x is None or len(x.split(":")[0]) > 1 else "0" + x)
    return pd.to_timedelta(clean, errors="coerce").dt.total_seconds() / 3600


def parse_activities_csv(data: bytes, distance_unit: str = "mi", elevation_unit: str = "ft") -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(data), dtype=str, encoding="utf-8-sig")
    lookup = {c.strip().lower(): c for c in raw.columns}
    col = {k: next((lookup[a] for a in v if a in lookup), None) for k, v in _ALIASES.items()}
    if col["distance"] is None or (col["time"] is None and col["elapsed"] is None):
        raise ValueError(
            "This CSV doesn't look like a Garmin Activities export. "
            "I need at least 'Distance' and 'Time' (or 'Elapsed Time') columns."
        )
    n = len(raw)
    empty = pd.Series([pd.NA] * n)
    pick = lambda k: raw[col[k]] if col[k] else empty  # noqa: E731

    dist = _to_num(pick("distance"))
    asc, desc = _to_num(pick("ascent")), _to_num(pick("descent"))
    if distance_unit == "km":
        dist = dist / 1.609344
    if elevation_unit == "m":
        asc, desc = asc * FT_PER_M, desc * FT_PER_M

    # Prefer total (elapsed) time: it includes breaks, which is what matters for planning a day.
    hours = _to_hours(pick("elapsed")) if col["elapsed"] else _to_hours(pick("time"))
    if col["elapsed"] and hours.isna().all():
        hours = _to_hours(pick("time"))

    out = pd.DataFrame({
        "date": pd.to_datetime(pick("date"), errors="coerce").dt.normalize(),
        "title": pick("title").fillna("").astype(str),
        "source": "csv",
        "activity_type": pick("activity_type").fillna("unknown").astype(str),
        "distance_mi": dist,
        "ascent_ft": asc,
        "descent_ft": desc,
        "time_hr": hours,
        "avg_hr": _to_num(pick("avg_hr")),
        "max_hr": _to_num(pick("max_hr")),
        "calories": _to_num(pick("calories")),
    })
    return out[HIKE_COLUMNS]


# -------------------------------------------------------------- dispatcher
def _walk_zip(data: bytes, rows: list, notes: list, depth: int = 0) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if len(members) > MAX_ZIP_MEMBERS:
            notes.append(f"Zip has {len(members)} files; only the first {MAX_ZIP_MEMBERS} were read.")
            members = members[:MAX_ZIP_MEMBERS]
        for m in members:
            low = m.filename.lower()
            if m.file_size > MAX_MEMBER_BYTES:
                notes.append(f"Skipped {m.filename} (too large).")
                continue
            if low.endswith(".fit"):
                row = parse_fit(zf.read(m), re.split(r"[\\/]", m.filename)[-1])
                if row:
                    rows.append(row)
            elif low.endswith(".zip") and depth < 2:
                _walk_zip(zf.read(m), rows, notes, depth + 1)


def load_uploads(files: list[tuple[str, bytes]], distance_unit="mi", elevation_unit="ft"):
    """files = [(filename, bytes), ...]  ->  (DataFrame[HIKE_COLUMNS], [notes])"""
    rows: list[dict] = []
    frames: list[pd.DataFrame] = []
    notes: list[str] = []
    for fname, data in files:
        low = fname.lower()
        try:
            if low.endswith(".fit"):
                row = parse_fit(data, fname)
                rows.append(row) if row else notes.append(f"{fname}: no activity summary found.")
            elif low.endswith(".zip"):
                _walk_zip(data, rows, notes)
            elif low.endswith(".csv"):
                frames.append(parse_activities_csv(data, distance_unit, elevation_unit))
            else:
                notes.append(f"{fname}: unsupported file type.")
        except Exception as e:  # keep the app alive on a bad file
            notes.append(f"{fname}: could not be read ({e}).")
    if rows:
        frames.append(pd.DataFrame(rows)[HIKE_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=HIKE_COLUMNS), notes
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["date", "distance_mi", "time_hr"]).sort_values("date").reset_index(drop=True)
    return df, notes


def hikes_only(df: pd.DataFrame, types: list[str] | None = None) -> pd.DataFrame:
    """Keep hiking activities (default) or the activity types the user picked."""
    if types is None:
        mask = df["activity_type"].str.contains("hik", case=False, na=False)
    else:
        mask = df["activity_type"].isin(types)
    return df[mask].reset_index(drop=True)
