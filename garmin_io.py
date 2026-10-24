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
    "start_local", "date", "title", "match", "activity_type",
    "distance_mi", "ascent_ft", "descent_ft", "time_hr",
    "avg_hr", "max_hr", "calories",
]
NUMERIC = ["distance_mi", "ascent_ft", "descent_ft", "time_hr", "avg_hr", "max_hr", "calories"]
MATCH_TOLERANCE_S = 180  # CSV and .fit start times may differ by a few seconds

MAX_ZIP_MEMBERS = 2000
MAX_MEMBER_BYTES = 80 * 1024 * 1024


# ---------------------------------------------------------------- .fit files
def parse_fit(data: bytes, name: str = "activity.fit") -> dict | None:
    """Return one row (dict) from the 'session' summary of a .fit file, or None."""
    import fitdecode

    session = None
    local_ts = None
    try:
        with fitdecode.FitReader(io.BytesIO(data)) as reader:
            for frame in reader:
                if frame.frame_type != fitdecode.FIT_FRAME_DATA:
                    continue
                if frame.name == "session" and session is None:
                    session = {k: frame.get_value(k, fallback=None) for k in (
                        "sport", "sub_sport", "start_time", "total_distance", "total_ascent", "total_descent",
                        "total_timer_time", "total_elapsed_time", "avg_heart_rate", "max_heart_rate",
                        "total_calories")}
                elif frame.name == "activity" and local_ts is None:
                    local_ts = frame.get_value("local_timestamp", fallback=None)
    except Exception:
        return None
    if not session or session["total_distance"] is None:
        return None
    secs = session["total_timer_time"] or session["total_elapsed_time"]
    if secs is None:
        return None
    start = session["start_time"]
    if local_ts is not None:  # watch's local clock = what Garmin Connect shows in the CSV
        start_local = pd.Timestamp(local_ts)
        if start_local.tzinfo is not None:  # wall-clock value stamped as UTC; keep the clock reading
            start_local = start_local.tz_localize(None)
    elif start is not None:
        start_local = pd.Timestamp(start).tz_convert("America/New_York").tz_localize(None)
    else:
        start_local = pd.NaT
    sport = " ".join(str(x) for x in (session["sport"], session["sub_sport"]) if x and x != "generic")
    return {
        "start_local": start_local,
        "date": start_local.normalize() if pd.notna(start_local) else pd.NaT,
        "title": name,
        "match": "fit only",
        "activity_type": sport or "unknown",
        "distance_mi": session["total_distance"] / M_PER_MILE,
        "ascent_ft": (session["total_ascent"] or 0) * FT_PER_M,
        "descent_ft": (session["total_descent"] or 0) * FT_PER_M,
        "time_hr": secs / 3600,
        "avg_hr": session["avg_heart_rate"],
        "max_hr": session["max_heart_rate"],
        "calories": session["total_calories"],
    }


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

    start_local = pd.to_datetime(pick("date"), errors="coerce")
    out = pd.DataFrame({
        "start_local": start_local,
        "date": start_local.dt.normalize(),
        "title": pick("title").fillna("").astype(str),
        "match": "csv only",
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


def read_fit_uploads(files: list[tuple[str, bytes]]) -> tuple[pd.DataFrame, list[str]]:
    """files = [(filename, bytes)] of .fit or .zip  ->  (one row per activity, notes)"""
    rows: list[dict] = []
    notes: list[str] = []
    for fname, data in files:
        low = fname.lower()
        try:
            if low.endswith(".fit"):
                row = parse_fit(data, fname)
                rows.append(row) if row else notes.append(f"{fname}: no activity summary found.")
            elif low.endswith(".zip"):
                _walk_zip(data, rows, notes)
            else:
                notes.append(f"{fname}: unsupported file type.")
        except Exception as e:  # keep the app alive on a bad file
            notes.append(f"{fname}: could not be read ({e}).")
    if not rows:
        return pd.DataFrame(columns=HIKE_COLUMNS), notes
    df = pd.DataFrame(rows)[HIKE_COLUMNS].drop_duplicates(subset=["start_local", "distance_mi"])
    return df, notes


def merge_sources(csv_df: pd.DataFrame, fit_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-reference Activities.csv with .fit files.

    * Match on start time (CSV 'Date' vs the .fit's local start), within a few minutes.
      If the CSV has dates only (no clock times), match on same day + closest distance.
    * Matched rows: numbers come from the .fit (the CSV often has no climb); title/type come from the CSV.
    * Unmatched rows are kept and labelled 'csv only' / 'fit only'.
    """
    csv_df = csv_df.copy().reset_index(drop=True)
    if fit_df.empty:
        return csv_df.sort_values("start_local").reset_index(drop=True)
    t = csv_df["start_local"]
    has_clock = bool(((t.dt.hour + t.dt.minute + t.dt.second) > 0).any())
    used: set[int] = set()
    rows: list[dict] = []
    for _, f in fit_df.sort_values("start_local").iterrows():
        free = csv_df[~csv_df.index.isin(used)]
        hit = None
        if has_clock and pd.notna(f["start_local"]):
            diff = (free["start_local"] - f["start_local"]).abs().dt.total_seconds()
            if len(diff) and diff.min() <= MATCH_TOLERANCE_S:
                hit = diff.idxmin()
        elif pd.notna(f["date"]):
            same_day = free[free["date"] == f["date"]]
            if len(same_day):
                rel = (same_day["distance_mi"] - f["distance_mi"]).abs() / max(f["distance_mi"], 0.1)
                if rel.min() <= 0.15:
                    hit = rel.idxmin()
        row = f.to_dict()
        if hit is not None:
            used.add(hit)
            c = csv_df.loc[hit]
            row["match"] = "matched"
            row["title"] = c["title"] if str(c["title"]).strip() else row["title"]
            row["activity_type"] = c["activity_type"]
            for col in NUMERIC:  # .fit wins; CSV only fills gaps
                if pd.isna(row[col]) and pd.notna(c[col]):
                    row[col] = c[col]
        rows.append(row)
    leftovers = csv_df[~csv_df.index.isin(used)]
    out = pd.concat([pd.DataFrame(rows), leftovers], ignore_index=True)[HIKE_COLUMNS]
    return out.sort_values("start_local").reset_index(drop=True)


def load_activities(csv_bytes: bytes, fit_files: list[tuple[str, bytes]], distance_unit="mi", elevation_unit="ft"):
    """Activities.csv (required) + .fit/.zip files (optional but recommended) -> (table, notes)."""
    notes: list[str] = []
    csv_df = parse_activities_csv(csv_bytes, distance_unit, elevation_unit)
    fit_df, fit_notes = read_fit_uploads(fit_files) if fit_files else (pd.DataFrame(columns=HIKE_COLUMNS), [])
    notes += fit_notes
    return merge_sources(csv_df, fit_df), notes


def activity_key(start_local, date, distance_mi) -> str:
    """Stable ID for an activity: its start time to the minute (falls back to date + distance)."""
    if pd.notna(start_local) and (start_local.hour or start_local.minute or start_local.second):
        return pd.Timestamp(start_local).strftime("%Y-%m-%d %H:%M")
    return f"{pd.Timestamp(date):%Y-%m-%d} {distance_mi:.1f}mi" if pd.notna(date) else f"unknown {distance_mi}"


def hikes_only(df: pd.DataFrame, types: list[str] | None = None) -> pd.DataFrame:
    """Keep hiking activities (default) or the activity types the user picked."""
    if types is None:
        mask = df["activity_type"].str.contains("hik", case=False, na=False)
    else:
        mask = df["activity_type"].isin(types)
    return df[mask].reset_index(drop=True)
