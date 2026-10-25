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

from track_features import TRACK_COLUMNS, TRACK_NUMERIC, empty_features, extract

M_PER_MILE = 1609.344
FT_PER_M = 3.28084

# Same columns, same order, every time (also the order used for model training).
HIKE_COLUMNS = [
    "start_local", "date", "title", "match", "source_file", "activity_type",
    "distance_mi", "ascent_ft", "descent_ft", "time_hr",
    "avg_hr", "max_hr", "calories",
] + TRACK_COLUMNS  # moving/stopped time, pace by grade, elevation+HR profile (only from .fit files)
NUMERIC = ["distance_mi", "ascent_ft", "descent_ft", "time_hr", "avg_hr", "max_hr", "calories"] + TRACK_NUMERIC
MATCH_TOLERANCE_S = 180  # CSV and .fit start times may differ by a few seconds

MAX_ZIP_MEMBERS = 2000
MAX_MEMBER_BYTES = 80 * 1024 * 1024


# ---------------------------------------------------------------- .fit files
def parse_fit(data: bytes, name: str = "activity.fit") -> dict | None:
    """Return one row (dict) from the 'session' summary of a .fit file, or None."""
    import fitdecode

    session = None
    local_ts = None
    rec: dict[str, list] = {"t": [], "d": [], "a": [], "h": []}  # time, distance, altitude, heart rate
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
                elif frame.name == "record":  # no GPS fields are read
                    rec["t"].append(frame.get_value("timestamp", fallback=None))
                    rec["d"].append(frame.get_value("distance", fallback=None))
                    rec["a"].append(frame.get_value("enhanced_altitude", fallback=frame.get_value("altitude", fallback=None)))
                    rec["h"].append(frame.get_value("heart_rate", fallback=None))
                elif frame.name == "activity" and local_ts is None:
                    local_ts = frame.get_value("local_timestamp", fallback=None)
    except Exception:
        return None
    if not session or session["total_distance"] is None:
        return None
    # Total (elapsed) time includes breaks and pauses, which is what matters for planning a day, and it matches
    # how the Activities.csv path measures time. Timer time (clock stopped on pauses) is the fallback.
    secs = session["total_elapsed_time"] or session["total_timer_time"]
    if secs is None:
        return None
    try:
        track = extract(rec["t"], rec["d"], rec["a"], rec["h"], secs)
    except Exception:  # a odd track must never stop the hike from loading
        track = empty_features()
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
        "source_file": name,  # the file this hike came from (zip name is swapped in by read_fit_uploads)
        "activity_type": sport or "unknown",
        "distance_mi": session["total_distance"] / M_PER_MILE,
        "ascent_ft": (session["total_ascent"] or 0) * FT_PER_M,
        "descent_ft": (session["total_descent"] or 0) * FT_PER_M,
        "time_hr": secs / 3600,
        "avg_hr": session["avg_heart_rate"],
        "max_hr": session["max_heart_rate"],
        "calories": session["total_calories"],
        **track,
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
        "source_file": "",
        "activity_type": pick("activity_type").fillna("unknown").astype(str),
        "distance_mi": dist,
        "ascent_ft": asc,
        "descent_ft": desc,
        "time_hr": hours,
        "avg_hr": _to_num(pick("avg_hr")),
        "max_hr": _to_num(pick("max_hr")),
        "calories": _to_num(pick("calories")),
    })
    for c in TRACK_NUMERIC:
        out[c] = float("nan")
    out["profile"] = ""
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
                before = len(rows)
                _walk_zip(data, rows, notes)
                added = len(rows) - before
                for r in rows[before:]:
                    # a single-hike zip (Garmin "Export Original", e.g. 19909908649.zip) is named by the zip itself;
                    # a big multi-hike zip is named "bigzip.zip › member.fit" so you can still find the hike
                    r["source_file"] = fname if added == 1 else f"{fname} › {r['source_file']}"
            else:
                notes.append(f"{fname}: unsupported file type.")
        except Exception as e:  # keep the app alive on a bad file
            notes.append(f"{fname}: could not be read ({e}).")
    if not rows:
        return pd.DataFrame(columns=HIKE_COLUMNS), notes
    df = pd.DataFrame(rows)[HIKE_COLUMNS].drop_duplicates(subset=["start_local", "distance_mi"])
    if len(df) < len(rows):  # the same activity uploaded more than once (same start time and distance)
        k = len(rows) - len(df)
        notes.append(f"ℹ️ Ignored {k} duplicate file{'s' if k != 1 else ''}: the same hike was uploaded more than once. "
                     "Hikes on different days are all kept.")
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
    out = pd.concat([pd.DataFrame(rows), leftovers], ignore_index=True)
    out = out.reindex(columns=HIKE_COLUMNS)
    out["source_file"] = out["source_file"].fillna("")
    out["profile"] = out["profile"].fillna("")
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


def combine_saved(prior: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Merge hikes from a saved file with freshly uploaded ones.

    * Same activity = starts within MATCH_TOLERANCE_S of each other.
    * For a duplicate, keep the row that has climb data (a bare CSV row won't overwrite a .fit-derived one);
      if both do, the newly uploaded row wins. The saved start time and title are kept so tags stay attached.
    """
    prior = prior.copy()
    if new.empty:
        return prior.sort_values("start_local").reset_index(drop=True)
    if prior.empty:
        return new.sort_values("start_local").reset_index(drop=True)
    extra = [c for c in prior.columns if c not in HIKE_COLUMNS]  # 'peaks', 'include' ride along
    for c in extra:
        new = new.assign(**{c: pd.Series([None] * len(new), dtype=object)})
    rows = prior.to_dict("records")
    starts = pd.to_datetime(prior["start_local"]).reset_index(drop=True)
    for _, n in new.sort_values("start_local").iterrows():
        diffs = (starts - n["start_local"]).abs().dt.total_seconds() if len(starts) else pd.Series(dtype=float)
        i = int(diffs.idxmin()) if len(diffs) and diffs.min() <= MATCH_TOLERANCE_S else None
        if i is None:
            rows.append(n.to_dict())
            starts = pd.concat([starts, pd.Series([n["start_local"]])], ignore_index=True)
            continue
        old = rows[i]
        if pd.notna(n["ascent_ft"]) or pd.isna(old["ascent_ft"]):  # new has climb, or neither does
            merged = n.to_dict()
            for c in ("start_local", "date") + tuple(extra):
                merged[c] = old[c]
            if str(old.get("title", "")).strip() and merged.get("match") in ("fit only", None):
                merged["title"] = old["title"]
            if not str(merged.get("source_file") or "").strip():
                merged["source_file"] = old.get("source_file", "")
            for c in TRACK_COLUMNS:  # a bare CSV row must not wipe saved track features
                v = merged.get(c)
                if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
                    merged[c] = old.get(c, v)
            rows[i] = merged
    out = pd.DataFrame(rows)
    for c in extra:
        out[c] = out[c].map(lambda v: v if isinstance(v, (list, bool)) else None)
    return out.sort_values("start_local").reset_index(drop=True)
