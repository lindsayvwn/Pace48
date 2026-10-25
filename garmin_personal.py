"""Garmin *Personal* API: pull YOUR OWN hikes with a personal API key, no uploads needed.

This is for the person who runs the app (or anyone who clones the repo and adds their own key). Visitors never
see or use the key. Endpoints and the token flow follow Garmin's official Python client
(github.com/garmin/connect-personal-api-python, Apache-2.0):

  1. POST {base}/personalapitoken-service/personal-api/personal-api-key/token    Authorization: ApiKey <key>   -> JWT
  2. GET  {base}/activitylist-service/activities/search/activities?start=&limit=&startDate=&activityType=   Bearer <jwt>
  3. GET  {base}/download-service/files/activity/{id}                                                    -> ZIP with the .fit

Each activity's ZIP goes through the same reader as uploaded files (garmin_io / track_features), so the owner's data
has exactly the same columns, moving/stopped time, paces and profiles as an upload. Parsed hikes are cached in
``.cache/`` so only NEW activities are downloaded on later loads.

Create a key (shown once): https://connect.garmin.com/app/settings/personalApiToken
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

from garmin_io import HIKE_COLUMNS, FT_PER_M, M_PER_MILE, NUMERIC, merge_sources, read_fit_uploads
from tags_io import hikes_from_csv, hikes_to_csv
from track_features import TRACK_NUMERIC, TRACK_VERSION, has_pace, profile_version

DEFAULT_BASE_URL = "https://personal.connectapi.garmin.com"
BASE_URL_ENV = "GARMIN_PERSONAL_API_BASE_URL"
KEY_ENV = "GARMIN_PERSONAL_API_KEY"
TOKEN_PATH = "/personalapitoken-service/personal-api/personal-api-key/token"
LIST_PATH = "/activitylist-service/activities/search/activities"
DOWNLOAD_PATH = "/download-service/files/activity/{id}"
KEY_PAGE = "https://connect.garmin.com/app/settings/personalApiToken"
TIMEOUT_S = 30
PAGE_SIZE = 100
MAX_PAGES = 60
DEFAULT_TYPES = ("hiking",)
CACHE_FILE = Path(__file__).parent / ".cache" / "owner_hikes.csv"

_TOKENS: dict[str, tuple[str, float]] = {}   # key fingerprint -> (token, expiry): memory only, never written to disk


class PersonalApiError(Exception):
    """A problem talking to Garmin. The message is safe to show: it never contains the key or a token."""


def _base_url() -> str:
    return os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL).rstrip("/")


def _fingerprint(key: str) -> str:
    return hashlib.sha256(f"{_base_url()}\0{key}".encode()).hexdigest()[:16]


def _friendly(status: int) -> str:
    if status in (401, 403):
        return f"Garmin rejected the personal API key (HTTP {status}). It may have expired or been revoked: create a new one at {KEY_PAGE}."
    if status == 429:
        return "Garmin is rate-limiting requests (HTTP 429). Try again in a few minutes."
    return f"Garmin returned HTTP {status}."


def _jwt_exp(token: str) -> float:
    try:
        payload = token.split(".")[1]
        return float(json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))["exp"])
    except Exception:
        return time.time() + 55 * 60   # can't read the expiry: assume just under an hour


def get_token(api_key: str, *, force: bool = False) -> str:
    """Exchange the personal API key for a bearer token (cached in memory until an hour before it expires)."""
    fp = _fingerprint(api_key)
    cached = _TOKENS.get(fp)
    if cached and not force and cached[1] - time.time() > 3600:
        return cached[0]
    try:
        r = requests.post(f"{_base_url()}{TOKEN_PATH}", headers={"Accept": "*/*", "Authorization": f"ApiKey {api_key.strip()}"},
                          timeout=TIMEOUT_S)
    except requests.RequestException as e:
        raise PersonalApiError(f"Couldn't reach Garmin ({type(e).__name__}).") from None
    if not r.ok:
        raise PersonalApiError(_friendly(r.status_code))
    token = r.text.strip()
    if not token:
        raise PersonalApiError("Garmin returned an empty token.")
    _TOKENS[fp] = (token, _jwt_exp(token))
    return token


def _get(path: str, token: str, *, params: dict | None = None, accept: str = "application/json") -> requests.Response:
    try:
        r = requests.get(f"{_base_url()}{path}", params=params, timeout=TIMEOUT_S,
                         headers={"Authorization": f"Bearer {token}", "Accept": accept})
    except requests.RequestException as e:
        raise PersonalApiError(f"Couldn't reach Garmin ({type(e).__name__}).") from None
    if not r.ok:
        raise PersonalApiError(_friendly(r.status_code))
    return r


def _type_key(item: dict) -> str:
    t = item.get("activityType")
    if isinstance(t, dict):
        t = t.get("typeKey") or t.get("typeId")
    return str(t or "unknown").lower()


def list_activities(token: str, *, start_date: str | None = None, types: tuple[str, ...] = DEFAULT_TYPES) -> list[dict]:
    """All activity summaries since start_date whose type matches one of `types` (substring match, e.g. 'hik')."""
    out: list[dict] = []
    for page in range(MAX_PAGES):
        params = {"start": page * PAGE_SIZE, "limit": PAGE_SIZE}
        if start_date:
            params["startDate"] = start_date
        data = _get(LIST_PATH, token, params=params).json()
        items = data if isinstance(data, list) else (data or {}).get("activities") or (data or {}).get("activityList") or []
        out.extend(i for i in items if isinstance(i, dict))
        if len(items) < PAGE_SIZE:
            break
    want = tuple(t.lower() for t in types)
    return [i for i in out if i.get("activityId") and (not want or any(w in _type_key(i) for w in want))]


def download_zip(token: str, activity_id) -> bytes:
    r = _get(DOWNLOAD_PATH.format(id=int(activity_id)), token,
             accept="application/zip, application/x-zip-compressed, application/octet-stream")
    if not r.content:
        raise PersonalApiError(f"Activity {activity_id}: Garmin sent an empty file.")
    return r.content


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def summaries_to_df(items: list[dict]) -> pd.DataFrame:
    """Garmin activity-list summaries -> the app's hike table (used for titles/types and as the CSV stand-in)."""
    rows = []
    for i in items:
        start = pd.to_datetime(i.get("startTimeLocal"), errors="coerce")
        if pd.isna(start):
            continue
        secs = _num(i.get("elapsedDuration")) if pd.notna(_num(i.get("elapsedDuration"))) else _num(i.get("duration"))
        rows.append({
            "start_local": start, "date": start.normalize(),
            "title": i.get("activityName") or f"Garmin {_type_key(i)}",
            "match": "garmin list", "source_file": "", "activity_type": _type_key(i),
            "distance_mi": _num(i.get("distance")) / M_PER_MILE,
            "ascent_ft": _num(i.get("elevationGain")) * FT_PER_M,
            "descent_ft": _num(i.get("elevationLoss")) * FT_PER_M,
            "time_hr": secs / 3600,
            "avg_hr": _num(i.get("averageHR")), "max_hr": _num(i.get("maxHR")), "calories": _num(i.get("calories")),
            **{c: float("nan") for c in TRACK_NUMERIC}, "profile": "",
            "_id": str(i["activityId"]),
        })
    if not rows:
        return pd.DataFrame(columns=HIKE_COLUMNS + ["_id"])
    return pd.DataFrame(rows)


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    """Real number/date dtypes. Concatenating with an empty frame leaves columns as generic objects, which charts reject."""
    df = df.copy()
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("start_local", "date"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ("title", "match", "source_file", "activity_type", "profile"):
        df[c] = df[c].fillna("").astype(str)
    return df


def _read_cache() -> pd.DataFrame:
    try:
        df = hikes_from_csv(CACHE_FILE.read_bytes())
    except Exception:
        return pd.DataFrame(columns=HIKE_COLUMNS)
    # drop rows computed by an older version of the stop/pace definitions so they are re-downloaded and recomputed
    stale = df["moving_hr"].notna() & ((df["profile"].map(profile_version) != TRACK_VERSION) | ~df["profile"].map(has_pace))
    return df[~stale][HIKE_COLUMNS].reset_index(drop=True)


def _write_cache(df: pd.DataFrame) -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_bytes(hikes_to_csv(df, {}, lambda *_: ""))
    except OSError:
        pass   # read-only or ephemeral disk: just means the next load re-downloads


def sync(api_key: str, *, types=DEFAULT_TYPES, start_date: str | None = None, max_new: int = 40,
         progress=None) -> tuple[pd.DataFrame, list[str]]:
    """Return (all hikes, notes). Downloads only activities that aren't cached yet, at most `max_new` per call."""
    notes: list[str] = []
    token = get_token(api_key)
    try:
        items = list_activities(token, start_date=start_date, types=tuple(types))
    except PersonalApiError as e:
        if "rejected" not in str(e):
            raise
        items = list_activities(get_token(api_key, force=True), start_date=start_date, types=tuple(types))
    cached = _read_cache()
    done = {str(s).removesuffix(".zip") for s in cached["source_file"] if str(s).endswith(".zip")}
    todo = [i for i in items if str(i["activityId"]) not in done]
    batch, left = todo[:max_new], max(len(todo) - max_new, 0)

    new_rows = pd.DataFrame(columns=HIKE_COLUMNS)
    if batch:
        listed = summaries_to_df(batch)
        fits = []
        for n, item in enumerate(batch, 1):
            if progress:
                progress(n, len(batch))
            try:
                zbytes = download_zip(token, item["activityId"])
            except PersonalApiError as e:
                notes.append(f"Activity {item['activityId']}: {e}")
                continue
            df, fit_notes = read_fit_uploads([(f"{item['activityId']}.zip", zbytes)])
            notes += [n_ for n_ in fit_notes if not n_.startswith("ℹ️")]
            if not df.empty:
                fits.append(df)
        fit_df = pd.concat(fits, ignore_index=True) if fits else pd.DataFrame(columns=HIKE_COLUMNS)
        merged = merge_sources(listed[HIKE_COLUMNS], fit_df)   # same matching as an Activities.csv + .fit upload
        new_rows = merged.drop_duplicates(subset=["start_local", "distance_mi"])
    if not new_rows.empty:
        allrows = pd.concat([cached, new_rows[HIKE_COLUMNS]], ignore_index=True)
        allrows = allrows.drop_duplicates(subset=["start_local", "distance_mi"], keep="last")
        _write_cache(allrows)
    else:
        allrows = cached
    if left:
        notes.append(f"ℹ️ Loaded {len(batch)} new hikes from Garmin; {left} more will load the next time the page refreshes.")
    return _typed(allrows).sort_values("start_local").reset_index(drop=True), notes
