"""NH48 Pace Predictor — a Streamlit app.

Upload your own Garmin hikes -> tag the ones that were NH 4000-footers ->
the app learns YOUR pace and predicts how long any of the 48 peaks will take you.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import garmin_personal as gp
from garmin_io import HIKE_COLUMNS, NUMERIC, activity_key, combine_saved, load_activities
from tags_io import hikes_from_csv, hikes_to_csv, tags_from_csv, tags_to_csv
from model import (FEATURES, MIN_HIKES_FOR_MOVING_MODEL, MIN_HIKES_FOR_REGRESSION, TARGET, MovingStopModel, PaceModel,
                   fit_regression, predict, simple_loo_mae)
from track_features import (DEFAULT_HR_DROP_SD, DEFAULT_MIN_STOP_S, TRACK_NUMERIC, TRACK_VERSION, apply_stop_rules,
                            profile_frame, profile_version)

HERE = Path(__file__).parent
PEAKS = pd.read_csv(HERE / "data" / "nh4000_peaks.csv")
DEMO_CSV = HERE / "data" / "demo_activities_SYNTHETIC.csv"
ACCENT = "#2E6F57"

st.set_page_config(page_title="NH48 Pace", page_icon="🏔️", layout="wide")

LABELS = {
    "distance_mi": "Distance (mi)", "ascent_ft": "Elevation gain (ft)", "descent_ft": "Elevation loss (ft)",
    "time_hr": "Total time (hr)", "avg_hr": "Avg heart rate (bpm)", "max_hr": "Max heart rate (bpm)",
    "calories": "Calories",
    "moving_hr": "Moving time (hr)", "stopped_hr": "Stopped time (hr)", "stopped_pct": "Stopped (% of total time)",
    "flat_pace_min_mi": "Flat-ground pace (min/mi)", "climb_rate_fthr": "Climb rate on uphill ≥5% (ft/hr)",
    "steep_climb_rate_fthr": "Climb rate on steep ≥15% (ft/hr)", "descent_rate_fthr": "Descent rate on downhill ≥5% (ft/hr)",
    "steep_ascent_pct": "Share of climb on steep ≥15% (%)",
}


def fmt_hours(h: float) -> str:
    """3.04 -> '3 h 02 min' (used for anything the viewer reads as a time)."""
    m = int(round(h * 60))
    return f"{m // 60} h {m % 60:02d} min"


def fmt_delta(h: float) -> str:
    """Short durations: 0.45 -> '27 min'; 1.6 -> '1 h 36 min'."""
    m = int(round(h * 60))
    return f"{m} min" if m < 60 else fmt_hours(h)


WINTER_MONTHS = (12, 1, 2, 3)   # December to March
SEASON_COLORS = {"Winter (Dec-Mar)": "#3B6FB6", "Rest of year": ACCENT}


def season_of(ts) -> str:
    return "Winter (Dec-Mar)" if pd.notna(ts) and ts.month in WINTER_MONTHS else "Rest of year"


def uphill_hr(profile_json) -> float:
    """Average heart rate over the uphill (>=5% grade) slices of a hike's saved profile; NaN if it can't be told."""
    pf = profile_frame(profile_json) if isinstance(profile_json, str) else None
    if pf is None or pf["heart_rate"].notna().sum() < 10:
        return float("nan")
    dm = pf["miles"].diff() * 5280
    grade = pf["altitude_ft"].diff() / dm.where(dm > 0)
    hr = pf["heart_rate"][grade >= 0.05]
    return float(hr.mean()) if hr.notna().sum() >= 5 else float("nan")


# Animated loader: running, mountains, climbing, hiking, birds. Replaces Streamlit's built-in status figure.
HIDE_DEFAULT_STATUS = "<style>[data-testid='stStatusWidget']{display:none !important;}</style>"
LOADER_HTML = """
<style>
.trail-loader{display:flex;gap:.6rem;align-items:center;justify-content:center;padding:1.2rem 0 .2rem}
.trail-loader span{display:inline-block;font-size:2.2rem;animation:trail-hop 1.3s ease-in-out infinite;
  animation-delay:calc(var(--i) * 0.16s)}
@keyframes trail-hop{0%,55%,100%{transform:translateY(0)}25%{transform:translateY(-16px)}}
.trail-msg{text-align:center;opacity:.7;font-size:.95rem;padding-bottom:1rem}
@media (prefers-reduced-motion: reduce){.trail-loader span{animation:none}}
</style>
<div class="trail-loader">
  <span style="--i:0">🏃</span><span style="--i:1">🏔️</span><span style="--i:2">🧗</span>
  <span style="--i:3">🥾</span><span style="--i:4">🐦</span>
</div>
<div class="trail-msg">Reading your Garmin activities…</div>
"""


def suggest_peaks(title: str) -> list[str]:
    """Pre-fill peak tags when the activity title mentions a peak (user can edit)."""
    t = " " + " ".join(re.sub(r"[^a-z0-9 ]", " ", title.lower()).split()) + " "
    found = []
    for p in PEAKS["peak"]:
        key = " " + " ".join(re.sub(r"[^a-z0-9 ]", " ", p.lower()).split()) + " "
        if key in t:
            found.append(p)
    return found


# ------------------------------------------------------- owner's default data (Garmin Personal API)
def _owner_cfg() -> dict | None:
    """The app owner's personal-API settings from secrets.toml ([garmin_personal]) or the GARMIN_PERSONAL_API_KEY env var.
    None = not configured, so the app behaves exactly as a plain upload app. The key is never shown or sent to visitors."""
    try:
        g = dict(st.secrets.get("garmin_personal", {}))
    except Exception:
        g = {}
    key = str(g.get("api_key") or os.environ.get(gp.KEY_ENV, "")).strip()
    if not key or str(g.get("enabled", True)).strip().lower() in ("false", "0", "no"):
        return None
    types = g.get("activity_types", gp.DEFAULT_TYPES)
    return {"key": key, "types": tuple([types] if isinstance(types, str) else types),
            "start_date": str(g["start_date"]) if g.get("start_date") else None,
            "max_new": int(g.get("max_new_per_load", 40)),
            "hidden_dates": tuple(_as_list(g.get("hidden_dates"))),
            "hidden_ids": tuple(str(i).strip() for i in _as_list(g.get("hidden_activity_ids"))),
            "show_hidden": str(g.get("show_hidden", False)).strip().lower() in ("true", "1", "yes")}


def _as_list(v) -> list:
    return [] if v is None else ([v] if isinstance(v, (str, int)) else list(v))


REFRESH_COOLDOWN_S = 120


@st.cache_resource
def _refresh_state() -> dict:
    """Server-wide (shared by every visitor), so the 2-minute cooldown can't be dodged by opening a new tab."""
    return {"last": 0.0}


def _hide(df: pd.DataFrame, hidden_dates, hidden_ids) -> pd.DataFrame:
    """Drop hikes the owner doesn't want shown, by calendar day of the start time or by Garmin activity id."""
    if df.empty:
        return df
    day = pd.to_datetime(df["start_local"], errors="coerce").dt.strftime("%Y-%m-%d")
    ids = df["source_file"].astype(str).str.replace(r"\.zip$", "", regex=True)
    drop = day.isin({str(d)[:10] for d in hidden_dates}) | ids.isin(set(hidden_ids))
    return df[~drop].reset_index(drop=True)


@st.cache_data(ttl=900, show_spinner=False)
def _owner_hikes(_key, key_id, types, start_date, max_new, schema, hidden_dates, hidden_ids):
    """Cached for 15 minutes and shared by all visitors. `_key` (underscore) is left out of the cache key on purpose.
    Hidden hikes are removed HERE, so they never reach a visitor's page, table, charts or 'Save my hikes' download."""
    df, notes = gp.sync(_key, types=types, start_date=start_date, max_new=max_new)
    return _hide(df, hidden_dates, hidden_ids), notes, time.time()


# ------------------------------------------------------------------ header
st.markdown(HIDE_DEFAULT_STATUS, unsafe_allow_html=True)
st.title("🥾 NH 4000-Footer Pace Predictor")

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Your data")
    saved_up = st.file_uploader(
        "⓪ Saved hikes file (if you've been here before)", type=["csv"],
        help="The 'Save my hikes' file you downloaded last time. It holds your hikes AND your tags, so you only "
             "need to upload files for NEW hikes.",
    )
    csv_up = st.file_uploader(
        "① Activities.csv (required for new hikes)", type=["csv"],
        help="Garmin Connect → Activities → Export CSV. It supplies your activity titles.",
    )
    fit_ups = st.file_uploader(
        "② Activity .zip / .fit files (recommended)", type=["zip", "fit"], accept_multiple_files=True,
        help="Garmin's CSV often leaves climb blank. The .fit files carry the real distance, climb and time, "
             "and are matched to the CSV by start time.",
    )
    tags_up = st.file_uploader(
        "③ Older tags-only file (optional)", type=["csv"],
        help="The file you downloaded from this app last time. It restores your NH 4000 tags and exclusions.",
    )
    use_demo = st.checkbox("No Garmin data? Use synthetic demo data", value=False)
    with st.expander("Stop detection (advanced)"):
        stop_min_min = st.slider("Shortest break to count (minutes)", 1, 10, DEFAULT_MIN_STOP_S // 60,
                                 help="A slow or still stretch shorter than this counts as moving.")
        hr_drop = st.slider("Heart-rate drop that proves a rest (standard deviations)", 0.0, 6.0, float(DEFAULT_HR_DROP_SD), 0.25,
                            help="Each hike is judged against itself: the app measures how much YOUR heart rate normally varies "
                                 "while you move on that hike (one standard deviation, often 10-15 bpm). On ice or scrambling the watch "
                                 "can look 'stopped' while you are working hard. A slow stretch only counts as a break if your heart "
                                 "rate fell at least this many standard deviations below where it was while moving, or the watch "
                                 "timer paused. Other slow stretches are shown as 'slow technical' and counted as moving. "
                                 "Higher = fewer breaks. 0 = ignore heart rate.")
    c1, c2 = st.columns(2)
    dist_unit = c1.radio("CSV distance", ["mi", "km"], horizontal=True)
    elev_unit = c2.radio("CSV elevation", ["ft", "m"], horizontal=True)
    with st.expander("How do I get my files from Garmin?"):
        st.markdown(
            "**Activities.csv:** Garmin Connect → Activities → scroll to load the activities you want → *Export CSV*.\n\n"
            "**Each hike (.zip):** open the activity → ⚙️ → *Export Original*. Upload the .zip as-is; you can select many at once.\n\n"
            "**Everything at once:** Garmin account → *Data Management* → *Export Your Data* (a big zip of .fit files)."
        )
    st.caption(
        "🔒 Files are read in memory for this session only. From .fit files the app keeps just distance, "
        "climb, time, date, heart-rate summaries and a small elevation/heart-rate profile: **no GPS, name or weight**. Nothing is saved on a server."
    )
    st.caption(
        "Why no Garmin login? Garmin only allows account connections for approved companies, and asking for "
        "your password in a public app would be unsafe. Uploading gives you full control over which activities "
        "the app sees."
    )

# ------------------------------------------------------------- load data
csv_bytes = csv_up.getvalue() if csv_up else (DEMO_CSV.read_bytes() if use_demo and not saved_up else None)
fit_files = [(f.name, f.getvalue()) for f in (fit_ups or [])]

# Default: the owner's hikes (if a personal API key is configured). ANY upload by the visitor replaces them.
visitor_uploaded = bool(csv_up or saved_up or fit_ups or use_demo)
owner_cfg = _owner_cfg()
owner_df, owner_notes, owner_error, owner_synced = None, [], None, None
if owner_cfg and not visitor_uploaded:
    try:
        with st.spinner("Loading hikes from Garmin (the first load can take a minute)…"):
            _hd = () if owner_cfg["show_hidden"] else owner_cfg["hidden_dates"]
            _hi = () if owner_cfg["show_hidden"] else owner_cfg["hidden_ids"]
            owner_df, owner_notes, owner_synced = _owner_hikes(
                owner_cfg["key"], gp._fingerprint(owner_cfg["key"]), owner_cfg["types"], owner_cfg["start_date"],
                owner_cfg["max_new"], (tuple(HIKE_COLUMNS), TRACK_VERSION), _hd, _hi)
        if owner_df.empty:
            owner_df, owner_error = None, "Connected to Garmin, but no matching activities were found."
    except gp.PersonalApiError as e:
        owner_error = str(e)

if csv_bytes is None and saved_up is None and owner_df is None:
    if owner_error:
        st.error(f"Couldn't load the default hikes. {owner_error}")
    st.markdown("### 👈 Start by uploading your Garmin Activities.csv")
    st.markdown(
        "1. Upload **Activities.csv** (your titles) and the **.zip** files for the hikes you want (real climb and time)  \n"
        "2. Tag which ones were NH 4000-footers, and untick any you want to leave out  \n"
        "3. Explore your pace, then get a predicted time for any peak  \n"
        "4. **Download your tags** so next time you can upload them instead of starting over"
    )
    with st.expander("Preview the 48 NH 4000-footers reference list"):
        st.dataframe(PEAKS, hide_index=True, width="stretch")
    st.stop()


@st.cache_data(show_spinner=False)
def _load(csv_b, fit_tuple, du, eu, schema):
    # `schema` is only part of the cache key: when the table's columns change, old cached tables are not reused
    return load_activities(csv_b, list(fit_tuple), du, eu)


loader = st.empty()
loader.markdown(LOADER_HTML, unsafe_allow_html=True)
try:
    if owner_df is not None:
        activities, notes = owner_df.copy(), list(owner_notes)
    elif csv_bytes is not None:
        activities, notes = _load(csv_bytes, tuple(fit_files), dist_unit, elev_unit, (tuple(HIKE_COLUMNS), TRACK_VERSION))
    else:
        activities, notes = pd.DataFrame(columns=HIKE_COLUMNS), []
        if fit_files:
            notes.append("Upload the matching Activities.csv too, so new .fit files get their titles.")
    saved_df = hikes_from_csv(saved_up.getvalue()) if saved_up is not None else None
    if saved_df is not None:
        # track data (stops, paces, profile) made by an older version used different definitions: clear it, don't mix
        stale = saved_df["moving_hr"].notna() & (saved_df["profile"].map(profile_version) != TRACK_VERSION)
        if stale.any():
            saved_df.loc[stale, TRACK_NUMERIC] = float("nan")
            saved_df.loc[stale, "profile"] = ""
            notes.append(f"{int(stale.sum())} saved hike(s) had stop and pace data from an older version of the app, so I "
                         "cleared it. Re-upload those hikes' .zip files to refresh it. Total times saved by an older version "
                         "may also be shorter, because they left out paused time.")
except ValueError as e:
    loader.empty()
    st.error(str(e))
    st.stop()
loader.empty()
if saved_df is not None:
    n_before, n_uploaded = len(saved_df), len(activities)
    activities = combine_saved(saved_df[HIKE_COLUMNS], activities)
    n_new = len(activities) - n_before
    st.caption(f"Loaded **{n_before} saved hikes**; {n_new} new from this visit's uploads"
               + (f" ({n_uploaded - n_new} already saved, not added again)." if n_uploaded - n_new > 0 else "."))
for _c in HIKE_COLUMNS:  # belt and braces: never crash on a table that lacks a column
    if _c not in activities.columns:
        activities[_c] = float("nan") if _c in NUMERIC else ""
if owner_df is not None:
    _bc, _rc = st.columns([4, 1], vertical_alignment="center")
    _ago = max(int(time.time() - owner_synced), 0) if owner_synced else 0
    _bc.info("**Showing selected hikes from the app owner**, pulled from their Garmin account"
             + (f" (last synced {_ago // 60} min ago)" if _ago >= 60 else " (synced just now)")
             + ". Upload your own files in the sidebar and they replace these. Alternatively, you can clone the repository and run it locally with your personal API.")
    if _rc.button("🔄 Refresh Garmin data", help="Re-check Garmin for new hikes. Limited to once every 2 minutes."):
        _st = _refresh_state()
        _wait = REFRESH_COOLDOWN_S - (time.time() - _st["last"])
        if _wait > 0:
            _rc.warning(f"Refreshed recently. Try again in {int(_wait) + 1} s.")
        else:
            _st["last"] = time.time()
            _owner_hikes.clear()
            st.rerun()
for _c in NUMERIC:  # whatever the source (upload, saved file, Garmin sync), numbers must be real numbers for charts
    activities[_c] = pd.to_numeric(activities[_c], errors="coerce")
for _c in ("start_local", "date"):
    activities[_c] = pd.to_datetime(activities[_c], errors="coerce")
for n in notes:
    (st.info if n.startswith("ℹ️") else st.warning)(n)
if activities.empty:
    st.error("I couldn't find any activities in those files.")
    st.stop()

# match summary
counts = activities["match"].value_counts()
if fit_files:
    st.caption(
        f"Cross-referenced with your .fit files: **{counts.get('matched', 0)} matched**"
        + (f" · {counts.get('fit only', 0)} .fit file(s) with no CSV row (named by filename)" if counts.get("fit only", 0) else "")
        + (f" · {counts.get('csv only', 0)} CSV rows with no .fit file (climb may be missing)" if counts.get("csv only", 0) else "")
    )
elif saved_df is None and owner_df is None:
    st.warning("No .fit/.zip files uploaded: using the CSV's own numbers, and Garmin often leaves climb blank there. "
               "Upload the .zip files for best results.")

types = sorted(activities["activity_type"].unique())
default_types = [t for t in types if "hik" in t.lower()] or types
chosen_types = st.sidebar.multiselect(
    "Activity types to show", types, default=default_types,
    help="Hiking is the default. Running pace is usually much faster than hiking pace, "
         "so mixing types can make predictions too optimistic.",
)
all_rows = activities[activities["activity_type"].isin(chosen_types)].reset_index(drop=True)
all_rows = apply_stop_rules(all_rows, stop_min_min * 60, hr_drop)   # stopped/moving time under the sidebar's stop rules
all_rows["stopped_pct"] = 100 * all_rows["stopped_hr"] / all_rows["time_hr"].where(all_rows["time_hr"] > 0)
if all_rows.empty:
    st.error("No activities of the selected types.")
    st.stop()
if not all("hik" in t.lower() for t in chosen_types):
    st.sidebar.warning("Non-hiking activities selected: predictions may be too optimistic.")

# --------------------------------------------------------- tags that survive new uploads
# st.session_state["tags"] maps activity key -> {"peaks": [...], "include": bool, ...}.
# The key is the activity's start time, so it stays valid when you add more files later.
tags: dict = st.session_state.setdefault("tags", {})
if saved_df is not None:
    sig = (saved_up.name, saved_up.size)
    if st.session_state.get("saved_sig") != sig:  # apply each saved file once, not on every rerun
        for _, r in saved_df.iterrows():
            tags[activity_key(r["start_local"], r["date"], r["distance_mi"])] = {
                "peaks": r["peaks"], "include": bool(r["include"]),
                "date": f"{r['date']:%Y-%m-%d}" if pd.notna(r["date"]) else "", "title": r["title"],
                "distance_mi": round(float(r["distance_mi"]), 2) if pd.notna(r["distance_mi"]) else "",
            }
        st.session_state["saved_sig"] = sig
        st.session_state["tags_version"] = st.session_state.get("tags_version", 0) + 1
if tags_up is not None:
    sig = (tags_up.name, tags_up.size)
    if st.session_state.get("tags_sig") != sig:  # apply each saved file once, not on every rerun
        try:
            tags.update(tags_from_csv(tags_up.getvalue()))
            st.session_state["tags_sig"] = sig
            st.session_state["tags_version"] = st.session_state.get("tags_version", 0) + 1
        except Exception as e:
            st.sidebar.error(f"Couldn't read the saved tags file: {e}")
            st.session_state["tags_sig"] = sig

all_rows["key"] = [activity_key(a, d, m) for a, d, m in zip(all_rows["start_local"], all_rows["date"], all_rows["distance_mi"])]
for _, r in all_rows.iterrows():
    entry = tags.setdefault(r["key"], {"peaks": suggest_peaks(r["title"]), "include": True})
    entry.update(date=f"{r['date']:%Y-%m-%d}" if pd.notna(r["date"]) else "", title=r["title"],
                 distance_mi=round(float(r["distance_mi"]), 2) if pd.notna(r["distance_mi"]) else "")

tab1, tab2, tab3, tab4, tab_prof, tab5 = st.tabs(
    ["① Preview, tag & exclude", "② Explore", "③ Build a model", "④ Predict an NH 4000-footer",
     "Your pace profile", "Limitations & roadmap"]
)

# ============================================================ TAB 5 (drawn first so it shows even if a later step stops)
with tab5:
    st.subheader("What happens when it's wrong")
    st.info(
        "This model estimates how long a New Hampshire 4,000-footer will take *you*, based on the pace in your own "
        "past hikes. If it's wrong (especially too low), you could start too late and get caught by darkness or "
        "weather above treeline. Use it as a planning aid, add a safety buffer, and always carry a headlamp."
    )
    st.subheader("Current limitations")
    st.markdown(
        "- **It only knows your past hikes.** It can't see weather, ice, stream crossings, group size or trail "
        "difficulty. Steep, rocky terrain is slower than your average hike.\n"
        "- **Route numbers are approximate.** Each peak's round-trip miles and climb come from public hiking sites "
        "and are shared by peaks on the same route. Check a current guidebook and edit them if you take a different route.\n"
        f"- **It needs data.** With fewer than {MIN_HIKES_FOR_REGRESSION} usable hikes it falls back to the Naismith rule "
        "scaled to your pace, which is rougher.\n"
        "- **The time range is rough.** The 80% range is based on how far off the model is on your own hikes, so a "
        "small or very varied history makes it less trustworthy.\n"
        "- **Stops and steep-section pace need .fit files.** Activities.csv alone has no second-by-second data, so those "
        "features (and the moving + stops model) only use hikes whose .zip/.fit you uploaded.\n"
        "- **Total time includes breaks.** Time is measured start to finish, including rests and pauses, because that is "
        "what you need for planning a day.\n"
        "- **Garmin data only.** Visitors upload files. The hikes shown by default come from the app owner's own Garmin "
        "account, and only the owner can connect one."
    )
    st.subheader("Roadmap")
    st.markdown(
        "- Sign in with Garmin instead of uploading files (depends on getting Garmin API access)\n"
        "- More inputs for the model (for example season or trail difficulty)\n"
        "- Better route data for each peak"
    )

# ============================================================ TAB 1
with tab1:
    # Base table is rebuilt only when the data/tags change, so edits made in the grid aren't reset.
    editor_key = f"editor-{hash(tuple(all_rows['key']))}-{st.session_state.get('tags_version', 0)}-{stop_min_min}-{hr_drop}"
    base_key = f"base-{editor_key}"
    if base_key not in st.session_state:
        st.session_state[base_key] = pd.DataFrame({
            "include": [tags[k]["include"] for k in all_rows["key"]],
            "date": all_rows["date"], "title": all_rows["title"], "source_file": all_rows["source_file"],
            "match": all_rows["match"], "activity_type": all_rows["activity_type"],
            "distance_mi": all_rows["distance_mi"], "ascent_ft": all_rows["ascent_ft"],
            "time_hr": all_rows["time_hr"], "moving_hr": all_rows["moving_hr"], "stopped_hr": all_rows["stopped_hr"],
            "avg_hr": all_rows["avg_hr"],
            "nh4000": [tags[k]["peaks"] for k in all_rows["key"]],
        })
    st.markdown(
        "**Tick the activities to use, and tag any NH 4000-footers.** Untick *Use* to leave an activity out of every "
        "chart and model. Pick peak(s) in the *NH 4000 peaks* column (some are pre-filled from the title)."
        " Why might you want to *untick* an activity? Depending on your goals, you might want to exclude certain activities from analysis or visualization."
        "For example, if you went on a hike with a group and your pace was significantly slower than usual, you might want to untick that activity to avoid skewing the results."
    )
    edited = st.data_editor(
        st.session_state[base_key],
        key=editor_key, hide_index=True, width="stretch",
        disabled=["date", "title", "source_file", "match", "activity_type", "distance_mi", "ascent_ft", "time_hr",
                  "moving_hr", "stopped_hr", "avg_hr"],
        column_config={
            "include": st.column_config.CheckboxColumn("Use", default=True),
            "date": st.column_config.DateColumn("Date", format="MMM D, YYYY"),
            "title": "Title", "activity_type": "Type",
            "source_file": st.column_config.TextColumn("Source file", width="medium", help="The .zip/.fit you uploaded that this hike's "
                "numbers came from. Blank = only in Activities.csv (no file matched)."),
            "match": st.column_config.TextColumn("Match", help="matched = CSV row + .fit file; csv only; fit only (no CSV "
                "row, so named by its filename); saved = restored from your saved hikes file."),
            "distance_mi": st.column_config.NumberColumn("Miles", format="%.1f"),
            "ascent_ft": st.column_config.NumberColumn("Climb (ft)", format="%.0f"),
            "time_hr": st.column_config.NumberColumn("Hours", format="%.2f", help="Total time, start to finish, including stops."),
            "moving_hr": st.column_config.NumberColumn("Moving", format="%.2f", help="Hours spent actually moving (.fit files only)."),
            "stopped_hr": st.column_config.NumberColumn("Stopped", format="%.2f", help="Hours stopped: rests, summit time, pauses (.fit files only)."),
            "avg_hr": st.column_config.NumberColumn("Avg HR", format="%.0f"),
            "nh4000": st.column_config.MultiselectColumn("NH 4000 peaks", options=list(PEAKS["peak"])),
        },
    )
    # write edits back into the persistent store
    clean_peaks = edited["nh4000"].apply(lambda v: list(v) if isinstance(v, (list, tuple, np.ndarray)) else [])
    for k, inc, pk in zip(all_rows["key"], edited["include"].fillna(True), clean_peaks):
        tags[k]["include"], tags[k]["peaks"] = bool(inc), pk
    all_rows["peaks"] = clean_peaks.values
    all_rows["include"] = edited["include"].fillna(True).astype(bool).values
    hikes = all_rows[all_rows["include"]].reset_index(drop=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Activities used", f"{len(hikes)} of {len(all_rows)}")
    m2.metric("Total miles", f"{hikes['distance_mi'].sum():,.0f}")
    m3.metric("Total climb (ft)", f"{hikes['ascent_ft'].sum():,.0f}")
    m4.metric("Total hours", f"{hikes['time_hr'].sum():,.0f}")
    # Which used activities have real data, and which still need a matching .fit/.zip?
    def _summary(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "Date": df["date"], "Activity": df["title"], "Source file": df["source_file"],
            "NH 4000 peaks": df["peaks"].map(lambda ps: ", ".join(ps)),
            "Miles": df["distance_mi"],
        })

    has_climb = hikes["ascent_ft"].notna()
    ok_col, missing_col = st.columns(2)
    with ok_col:
        st.markdown(f"**✅ With climb data ({int(has_climb.sum())})**")
        st.caption("Used by the model.")
        st.dataframe(_summary(hikes[has_climb]), hide_index=True, width="stretch", height=260,
                     column_config={"Date": st.column_config.DateColumn("Date", format="MMM D, YYYY"),
                                    "Miles": st.column_config.NumberColumn("Miles", format="%.1f")})
    with missing_col:
        st.markdown(f"**⚠️ No match / no climb data ({int((~has_climb).sum())})**")
        st.caption("Left out of the model. Upload each one's .zip (step ②) to fill it in, or untick *Use* above.")
        if (~has_climb).any():
            st.dataframe(_summary(hikes[~has_climb]), hide_index=True, width="stretch", height=260,
                         column_config={"Date": st.column_config.DateColumn("Date", format="MMM D, YYYY"),
                                        "Miles": st.column_config.NumberColumn("Miles", format="%.1f")})
        else:
            st.success("Every activity you're using has climb data.")

    st.download_button(
        "💾 Save my hikes (upload this next time)", type="primary",
        data=hikes_to_csv(activities, tags, activity_key), file_name="my_pace48_hikes.csv", mime="text/csv",
        help="One condensed file with your hike numbers, peak tags and exclusions. Next visit, upload it in the "
             "sidebar (step ⓪) plus only your NEW hikes, and everything comes back. No GPS data is included.",
    )
    st.download_button(
        "Save tags only (older format)", data=tags_to_csv(tags), file_name="my_nh4000_tags.csv", mime="text/csv",
        help="Tags and exclusions only, without hike numbers. Prefer the hikes file above.",
    )
    st.caption("Tip: this app can't remember you between visits, so download your hikes file when you're done.")

if hikes.empty:
    st.info("Every activity is unticked. Tick at least one in the table above.")
    st.stop()

# usable rows for modelling (same rule everywhere)
model_df = hikes.dropna(subset=FEATURES + [TARGET])
model_df = model_df[(model_df[TARGET] > 0) & (model_df["distance_mi"] > 0)].reset_index(drop=True)
numeric_cols = [c for c in LABELS if c in model_df and model_df[c].notna().sum() >= 2]

# ============================================================ TAB 2
with tab2:
    st.subheader("Explore your hikes")
    if len(model_df) < 2 or len(numeric_cols) < 2:
        st.info("Charts need at least 2 hikes with complete numbers. Upload more files (or try the demo data).")
    else:
        a, b, c = st.columns(3)
        chart = a.selectbox("Chart type", ["Scatter", "Line", "Histogram"])
        ycol = b.selectbox("Column", numeric_cols, format_func=LABELS.get,
                           index=numeric_cols.index(TARGET) if TARGET in numeric_cols else 0)
        if chart == "Scatter":
            others = [x for x in numeric_cols if x != ycol]
            xcol = c.selectbox("Compare against", others, format_func=LABELS.get,
                               index=others.index("distance_mi") if "distance_mi" in others else 0)
            fig = px.scatter(model_df if ycol in model_df else hikes, x=xcol, y=ycol, hover_name="title",
                             trendline="ols" if len(hikes) >= 3 else None,
                             labels=LABELS, color_discrete_sequence=[ACCENT])
        elif chart == "Line":
            fig = px.line(hikes.sort_values("date"), x="date", y=ycol, markers=True, hover_name="title",
                          labels={**LABELS, "date": "Date"}, color_discrete_sequence=[ACCENT])
        else:
            bins = c.slider("Bins", 5, 40, 12)
            fig = px.histogram(hikes, x=ycol, nbins=bins, labels=LABELS, color_discrete_sequence=[ACCENT])
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=430)
        st.plotly_chart(fig, width="stretch")


# ============================================================ PACE PROFILE TAB
def _num(v, unit="", dec=0):
    return "n/a" if pd.isna(v) else f"{v:,.{dec}f}{unit}"


with tab_prof:
    st.subheader("Your pace profile")
    trk = hikes.dropna(subset=["moving_hr"]).reset_index(drop=True)
    if trk.empty:
        st.info("This tab needs the .zip/.fit files for your hikes, not just Activities.csv. They carry the second-by-second "
                "data behind moving vs stopped time, pace on steep sections and the elevation and heart-rate profiles.")
    else:
        st.caption(f"Based on {len(trk)} hike(s) with track data. Medians across hikes:")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Stopped", _num(trk["stopped_pct"].median(), "%"), help="Share of your total time spent stopped.")
        k2.metric("Flat-ground pace", _num(trk["flat_pace_min_mi"].median(), " min/mi", 1))
        k3.metric("Climb rate (uphill ≥5%)", _num(trk["climb_rate_fthr"].median(), " ft/hr"),
                  help="Vertical feet gained per hour of moving time on uphill sections.")
        k4.metric("Climb rate (steep ≥15%)", _num(trk["steep_climb_rate_fthr"].median(), " ft/hr"))
        k5.metric("Descent rate (≥5%)", _num(trk["descent_rate_fthr"].median(), " ft/hr"))

        st.markdown("**Each hike**")
        st.dataframe(
            trk[["date", "title", "distance_mi", "time_hr", "moving_hr", "stopped_pct", "flat_pace_min_mi",
                 "climb_rate_fthr", "steep_climb_rate_fthr", "descent_rate_fthr", "steep_ascent_pct"]],
            hide_index=True, width="stretch", height=260,
            column_config={
                "date": st.column_config.DateColumn("Date", format="MMM D, YYYY"), "title": "Activity",
                "distance_mi": st.column_config.NumberColumn("Miles", format="%.1f"),
                "time_hr": st.column_config.NumberColumn("Total hr", format="%.2f"),
                "moving_hr": st.column_config.NumberColumn("Moving hr", format="%.2f"),
                "stopped_pct": st.column_config.NumberColumn("Stopped %", format="%.0f"),
                "flat_pace_min_mi": st.column_config.NumberColumn("Flat pace (min/mi)", format="%.1f"),
                "climb_rate_fthr": st.column_config.NumberColumn("Climb ft/hr (≥5%)", format="%.0f"),
                "steep_climb_rate_fthr": st.column_config.NumberColumn("Steep ft/hr (≥15%)", format="%.0f"),
                "descent_rate_fthr": st.column_config.NumberColumn("Descent ft/hr", format="%.0f"),
                "steep_ascent_pct": st.column_config.NumberColumn("Climb on steep %", format="%.0f"),
            },
        )

        st.markdown("**Elevation, heart rate and stops along one hike**")
        withp = trk[trk["profile"].astype(str).str.len() > 2].reset_index(drop=True)
        if withp.empty:
            st.info("No elevation profiles in these hikes.")
        else:
            labels = [f"{r['date']:%b %d, %Y} · {r['title']} · {r['distance_mi']:.1f} mi" for _, r in withp.iterrows()]
            pick = st.selectbox("Hike", range(len(withp)), index=len(withp) - 1, format_func=lambda i: labels[i])
            pf = profile_frame(withp.loc[pick, "profile"], stop_min_min * 60, hr_drop)
            if pf is None:
                st.warning("Couldn't read this hike's profile.")
            else:
                col_of = {"Pace": "pace_min_mi", "Speed": "speed_mph", "Heart rate": "heart_rate"}
                metric = st.radio("Show on the chart", list(col_of), horizontal=True, key="profile_metric")
                if not pf[col_of[metric]].notna().any():
                    st.info(f"This hike's saved profile has no {metric.lower()} data (saved by an older version). Re-upload its .zip "
                            "to add it. Showing heart rate instead.")
                    metric = "Heart rate"
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=pf["miles"], y=pf["altitude_ft"], name="Elevation (ft)", fill="tozeroy",
                                         line=dict(color=ACCENT), fillcolor="rgba(46,111,87,0.25)"), secondary_y=False)
                if metric == "Pace":
                    fig.add_trace(go.Scatter(x=pf["miles"], y=pf["pace_min_mi"], name="Pace (min/mi)",
                                             line=dict(color="#2C5AA0", width=1.5), connectgaps=False,
                                             hovertemplate="%{y:.1f} min/mi<extra></extra>"), secondary_y=True)
                elif metric == "Speed":
                    fig.add_trace(go.Scatter(x=pf["miles"], y=pf["speed_mph"], name="Speed (mph)",
                                             line=dict(color="#2C5AA0", width=1.5),
                                             hovertemplate="%{y:.2f} mph<extra></extra>"), secondary_y=True)
                else:
                    fig.add_trace(go.Scatter(x=pf["miles"], y=pf["heart_rate"], name="Heart rate (bpm)",
                                             line=dict(color="#C0392B", width=1.5)), secondary_y=True)
                stops = pf[pf["stop_min"] > 0]
                slow = pf[pf["slow_min"] > 0]
                if len(slow):
                    fig.add_trace(go.Scatter(
                        x=slow["miles"], y=slow["altitude_ft"], mode="markers", name=f"Slow technical ({stop_min_min}+ min)",
                        marker=dict(size=(8 + slow["slow_min"] * 1.2).clip(upper=40), color="#5B7DB1", symbol="diamond",
                                    line=dict(color="white", width=1)),
                        text=[f"slow going for about {m:.0f} min (heart rate stayed up, so not counted as a rest)" for m in slow["slow_min"]],
                        hoverinfo="text+x"), secondary_y=False)
                if len(stops):
                    fig.add_trace(go.Scatter(
                        x=stops["miles"], y=stops["altitude_ft"], mode="markers", name=f"Stops ({stop_min_min}+ min)",
                        marker=dict(size=(8 + stops["stop_min"] * 1.2).clip(upper=40), color="#E6A100",
                                    line=dict(color="white", width=1)),
                        text=[f"stopped about {m:.0f} min" for m in stops["stop_min"]], hoverinfo="text+x"), secondary_y=False)
                lo_a, hi_a = float(pf["altitude_ft"].min()), float(pf["altitude_ft"].max())
                fig.update_yaxes(title_text="Elevation (ft)", range=[lo_a - 150, hi_a + 400], secondary_y=False)
                if metric == "Pace":
                    fig.update_yaxes(title_text="Moving pace (min/mi, faster is up)", autorange="reversed", secondary_y=True)
                elif metric == "Speed":
                    fig.update_yaxes(title_text="Speed over elapsed time (mph, rests included)", rangemode="tozero", secondary_y=True)
                else:
                    fig.update_yaxes(title_text="Heart rate (bpm)", secondary_y=True)
                fig.update_xaxes(title_text="Miles from start")
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=400, legend=dict(orientation="h", y=1.12))
                st.plotly_chart(fig, width="stretch")
                r = withp.loc[pick]
                biggest = pf.loc[pf["stop_min"].idxmax()]
                slow_hr = r.get("slow_technical_hr")
                st.caption(
                    f"Moving {fmt_hours(r['moving_hr'])}, stopped {fmt_hours(r['stopped_hr'])} "
                    f"({_num(r['stopped_pct'], '%')} of {fmt_hours(r['time_hr'])})"
                    + (f" · biggest stop about {biggest['stop_min']:.0f} min near mile {biggest['miles']:.1f}."
                       if biggest["stop_min"] > 0 else ".")
                    + (f" Slow technical going (counted as moving): {fmt_delta(slow_hr)}." if pd.notna(slow_hr) and slow_hr > 0 else "")
                )

        if len(trk) >= 2:
            trk = trk.copy()
            trk["season"] = trk["date"].map(season_of)
            trk["uphill_hr"] = trk["profile"].map(uphill_hr)
            trk["efficiency"] = trk["climb_rate_fthr"] / trk["uphill_hr"]   # feet climbed per hour, per heartbeat-per-minute of effort
            trk["when"] = trk["date"].dt.strftime("%b %d, %Y")
            eff = trk.dropna(subset=["efficiency"]).sort_values("date")
            st.markdown("#### Climbing efficiency: how much climbing do you get for your effort?")
            if len(eff) < 2:
                st.info("This needs at least 2 hikes with heart-rate data and a climb rate (.fit/.zip files).")
            else:
                c1, c2 = st.columns(2)
                f1 = px.scatter(eff, x="uphill_hr", y="climb_rate_fthr", color="season", color_discrete_map=SEASON_COLORS,
                                hover_name="title", hover_data={"when": True, "uphill_hr": ":.0f", "climb_rate_fthr": ":.0f",
                                                                 "efficiency": ":.1f", "season": False},
                                labels={"uphill_hr": "Average heart rate while climbing (bpm)",
                                        "climb_rate_fthr": "Climb rate (ft/hr)", "when": "Date", "efficiency": "ft/hr per bpm"})
                f1.update_traces(marker=dict(size=12, line=dict(color="white", width=1)))
                f1.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=340, legend_title_text="",
                                 legend=dict(orientation="h", y=-0.28), title="Higher and to the left = fitter")
                c1.plotly_chart(f1, width="stretch")
                f1b = go.Figure()
                f1b.add_trace(go.Scatter(x=eff["date"], y=eff["efficiency"], mode="lines", hoverinfo="skip", showlegend=False,
                                         line=dict(color="#B8C2BD", width=1.5)))
                for sname, g in eff.groupby("season"):
                    f1b.add_trace(go.Scatter(x=g["date"], y=g["efficiency"], mode="markers", name=sname, text=g["title"],
                                             marker=dict(size=12, color=SEASON_COLORS[sname], line=dict(color="white", width=1)),
                                             hovertemplate="<b>%{text}</b><br>%{x|%b %d, %Y}<br>%{y:.1f} ft/hr per bpm<extra></extra>"))
                f1b.update_xaxes(title_text="Date")
                f1b.update_yaxes(title_text="Climb rate per bpm (ft/hr per bpm)")
                f1b.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=340, legend_title_text="",
                                  legend=dict(orientation="h", y=-0.28), title="Is it improving over time?")
                c2.plotly_chart(f1b, width="stretch")
                st.caption("Efficiency = your climb rate divided by your average heart rate on the uphill parts. If it rises, you are "
                           "getting more climbing out of the same effort. Heart rate also moves with heat, sleep and caffeine, and a "
                           "winter climb is slower for reasons that have nothing to do with fitness, so compare like with like "
                           "(the colours separate winter from the rest of the year).")

            st.markdown("#### Winter vs the rest of the year")
            n_w = int((trk["season"] == "Winter (Dec-Mar)").sum())
            n_o = len(trk) - n_w
            trk["slow_pct"] = 100 * trk.get("slow_technical_hr", np.nan) / trk["time_hr"].where(trk["time_hr"] > 0)
            season_metrics = {
                "climb_rate_fthr": "Climb rate (ft/hr)", "steep_climb_rate_fthr": "Steep climb rate (ft/hr)",
                "flat_pace_min_mi": "Flat-ground pace (min/mi)", "descent_rate_fthr": "Descent rate (ft/hr)",
                "stopped_pct": "Stopped (% of time)", "slow_pct": "Slow technical going (% of time)",
                "efficiency": "Climb rate per bpm", "avg_hr": "Average heart rate (bpm)",
            }
            if n_w < 2 or n_o < 2:
                st.info(f"Only {n_w} winter and {n_o} other hike(s) with track data, so there isn't enough to compare yet. "
                        "Winter here means December to March.")
            else:
                cols = [c for c in season_metrics if c in trk and trk[c].notna().any()]
                summ = trk.groupby("season")[cols].median().T
                summ.index = [season_metrics[c] for c in summ.index]
                summ = summ.reindex(columns=list(SEASON_COLORS))
                summ["Difference"] = summ["Winter (Dec-Mar)"] - summ["Rest of year"]
                st.caption(f"Medians: {n_w} winter hike(s), {n_o} others. Winter = December to March.")
                st.dataframe(summ.round(1), width="stretch")
                pick_m = st.selectbox("Compare", cols, format_func=season_metrics.get, key="season_metric")
                fb = px.box(trk.dropna(subset=[pick_m]), x="season", y=pick_m, color="season", color_discrete_map=SEASON_COLORS,
                            points="all", hover_name="title", category_orders={"season": list(SEASON_COLORS)},
                            labels={pick_m: season_metrics[pick_m], "season": ""})
                fb.update_traces(marker=dict(size=9), boxmean=False)
                fb.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=340, showlegend=False)
                st.plotly_chart(fb, width="stretch")
                if min(n_w, n_o) < 5:
                    st.caption("⚠️ Fewer than 5 hikes in one group: treat differences as hints, not findings.")

            f2 = px.scatter(trk, x="distance_mi", y="stopped_pct", hover_name="title", size="time_hr", color="season",
                            color_discrete_map=SEASON_COLORS,
                            labels={"distance_mi": "Distance (mi)", "stopped_pct": "Stopped (% of time)"})
            f2.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=330, title="Do you stop more on longer days?",
                             legend_title_text="", legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(f2, width="stretch")
        st.caption("A stop is a break: a stretch at or above the shortest-break setting (sidebar, default 3 minutes) where you made "
                   "almost no progress (under about 40% of your typical moving speed on that hike, over 45 seconds; climbing counts "
                   "as progress, 1 m up is worth about 8 m of flat ground, so a slow scramble isn't mistaken for standing still; the "
                   "watch's own speed reading drops to zero too often to trust) AND either the "
                   "watch's timer paused or your heart rate fell by the sidebar's number of standard deviations (measured on that hike's own "
                   "minute-to-minute heart-rate variation). Slow stretches where it stayed up, like ice or rock, are 'slow technical' "
                   "time and count as moving. Grade is measured over about 100 m of trail from smoothed barometric "
                   "altitude, so short scrambles are averaged out. Peak predictions use distance and total climb because "
                   "the reference routes don't include steepness data yet.")

# ============================================================ TAB 3
with tab3:
    st.subheader("Fit a linear regression")
    if len(model_df) < 3:
        st.warning("Need at least 3 activities with complete numbers to fit a model.")
    else:
        a, b, c = st.columns(3)
        in1 = a.selectbox("Input 1", numeric_cols, format_func=LABELS.get, index=numeric_cols.index(FEATURES[0]))
        in2_opts = [x for x in numeric_cols if x != in1]
        in2 = b.selectbox("Input 2", in2_opts, format_func=LABELS.get,
                          index=in2_opts.index(FEATURES[1]) if FEATURES[1] in in2_opts else 0)
        tg_opts = [x for x in numeric_cols if x not in (in1, in2)]
        target = c.selectbox("Target to predict", tg_opts, format_func=LABELS.get,
                             index=tg_opts.index(TARGET) if TARGET in tg_opts else 0)
        feats = [in1, in2]  # same names + order used to train AND predict
        usable = hikes[feats + [target]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(usable) < 3:
            st.warning(f"Only {len(usable)} hike(s) have all three of these values (the stops and steep-section numbers "
                       "need .fit files). Pick other columns or upload more .fit/.zip files.")
        else:
            fit = fit_regression(hikes, feats, target)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Activities used", fit["n"])
            k2.metric("R²", f"{fit['r2']:.2f}", help="1.0 = perfect fit on your own data.")
            is_time = target == TARGET
            err = (lambda v: fmt_delta(v)) if is_time else (lambda v: f"{v:.2f}")
            k3.metric("Typical error (RMSE)", err(fit["rmse"]))
            k4.metric("Leave-one-out error", "n/a (need 6+)" if fit["loo_mae"] is None else err(fit["loo_mae"]),
                      help="Average miss when each activity is predicted from all the others — more honest than R².")
            coefs = fit["model"].coef_
            st.caption(
                f"**{LABELS[target]}** ≈ {fit['model'].intercept_:.2f} + {coefs[0]:.4g} × {LABELS[in1]} "
                f"+ {coefs[1]:.4g} × {LABELS[in2]}"
            )
            st.markdown("#### Test predictions")
            d1, d2 = st.columns(2)
            v1 = d1.number_input(LABELS[in1], value=float(round(model_df[in1].median(), 1)), min_value=0.0, key=f"v1-{in1}")
            v2 = d2.number_input(LABELS[in2], value=float(round(model_df[in2].median(), 1)), min_value=0.0, key=f"v2-{in2}")
            pred = predict(fit, {in1: v1, in2: v2})
            if target == TARGET:
                st.success(f"Predicted **{LABELS[target].replace(' (hr)', '')}: {fmt_hours(pred)}**")
            else:
                st.success(f"Predicted **{LABELS[target]}: {pred:.2f}**")
            tmp = model_df[["title", "date"] + feats + [target]].dropna(subset=feats + [target]).copy()
            tmp["predicted"] = fit["model"].predict(tmp[feats])
            if is_time:   # show hours as h:mm (4.5 -> 4:30) on the axes and in the hover
                hm = lambda h: f"{int(round(h * 60)) // 60}:{int(round(h * 60)) % 60:02d}"   # noqa: E731
                tmp["pred_txt"], tmp["act_txt"] = tmp["predicted"].map(hm), tmp[target].map(hm)
            else:
                tmp["pred_txt"], tmp["act_txt"] = tmp["predicted"].map("{:.2f}".format), tmp[target].map("{:.2f}".format)
            tmp["when"] = pd.to_datetime(tmp["date"], errors="coerce").dt.strftime("%b %d, %Y").fillna("")
            ylab = LABELS[target].replace(" (hr)", "") if is_time else LABELS[target]
            f2 = go.Figure()
            lo, hi = float(tmp[[target, "predicted"]].min().min()), float(tmp[[target, "predicted"]].max().max())
            pad = (hi - lo) * 0.05 or 0.5
            f2.add_trace(go.Scatter(x=[lo - pad, hi + pad], y=[lo - pad, hi + pad], mode="lines", hoverinfo="skip", showlegend=False,
                                    line=dict(dash="dot", color="gray")))
            f2.add_trace(go.Scatter(
                x=tmp["predicted"], y=tmp[target], mode="markers", showlegend=False,
                marker=dict(size=13, color=ACCENT, line=dict(color="white", width=1)),
                customdata=np.stack([tmp["title"].astype(str), tmp["when"], tmp["pred_txt"], tmp["act_txt"]], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Predicted: %{customdata[2]}<br>"
                              "Actual: %{customdata[3]}<extra></extra>"))
            axis = dict(title_font=dict(size=18), tickfont=dict(size=14), showgrid=False, zeroline=False)
            xax, yax = dict(axis), dict(axis)
            if is_time:   # whole-hour ticks labelled 4:00, 5:00 ...
                step = 1 if (hi - lo) <= 10 else 2
                ticks = list(np.arange(np.floor(lo - pad), np.ceil(hi + pad) + 1, step))
                for ax in (xax, yax):
                    ax.update(tickmode="array", tickvals=ticks, ticktext=[f"{int(t)}:00" for t in ticks])
            xax["title_text"] = f"Predicted {ylab}"
            yax["title_text"] = f"Actual {ylab}"
            f2.update_xaxes(**xax)
            f2.update_yaxes(**yax)
            f2.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=380, hoverlabel=dict(font_size=14))
            st.plotly_chart(f2, width="stretch")
            st.caption("Dots on the dotted line = perfect predictions. Hover a dot to see which hike it is.")
            st.caption(
                f"**Limitations:** this model is fit on {fit['n']} activit{'y' if fit['n'] == 1 else 'ies'}, a small sample, so a single unusual day "
                "(an icy hike, a long summit rest, a slow group) can shift it noticeably, and the error figures above are rough. "
                "It only sees distance and climb, not weather, trail roughness or how tired you were, and it is least reliable for "
                "trips longer or steeper than anything you've logged. More activities, ideally with a mix of lengths and "
                "climbs, will make the predictions steadier and better."
            )

# ============================================================ TAB 4
with tab4:
    st.subheader("How long will an NH 4000-footer take you?")
    pm_simple = PaceModel(model_df)
    pm_ms, n_track = MovingStopModel.try_build(hikes)
    pm = pm_simple
    if pm_ms is None:
        st.caption(f"The moving + stops model needs {MIN_HIKES_FOR_MOVING_MODEL}+ hikes with .fit track data "
                   f"(you have {n_track}). Until then the simple model is used.")
    else:
        simple_mae = simple_loo_mae(hikes.dropna(subset=["moving_hr", "stopped_hr"]))
        ms_better = pm_ms.loo_mae is not None and simple_mae is not None and pm_ms.loo_mae <= simple_mae
        choice = st.radio(
            "Model", ["Simple: distance + climb → total time", "Moving + stops: uses your track data"],
            index=1 if ms_better else 0, horizontal=True,
            help="Moving + stops predicts the time you'll be moving, then adds your typical stopped time.",
        )
        if choice.startswith("Moving"):
            pm = pm_ms
        if pm_ms.loo_mae is not None and simple_mae is not None:
            st.caption(
                f"On your {pm_ms.n} hikes with track data, a leave-one-out test misses by "
                f"**{fmt_delta(simple_mae)}** (simple) vs **{fmt_delta(pm_ms.loo_mae)}** (moving + stops)"
                + (" · moving + stops is selected because it misses by less or the same." if ms_better and pm is pm_ms else
                   " · the simple model misses by less, so it's the default." if not ms_better else ".")
            )
        else:
            st.caption("Not enough hikes with track data yet to compare the two models fairly (needs 6+).")
    if pm.kind == "moving+stops":
        st.caption(
            f"Moving time from a regression on **{pm.n} hikes** (R² {pm.fit['r2']:.2f}) plus your typical stopped time "
            f"(**{pm.stop_ratio:.0%}** of moving time) · typical error ±{fmt_delta(pm.spread)}."
        )
    elif pm.n == 0:
        st.warning("No complete hikes yet, so I'm using the standard Naismith rule (3 mph + 1 h per 2,000 ft).")
    elif pm.kind == "naismith":
        st.warning(
            f"You have {pm.n} usable hike(s); a regression needs {MIN_HIKES_FOR_REGRESSION}+. For now I'm using the "
            f"Naismith rule scaled to your pace (you've been {pm.factor:.2f}× the rule). Add more hikes for a real model."
        )
    else:
        st.caption(
            f"Linear regression trained on **{pm.n} hikes** · R² {pm.fit['r2']:.2f} · typical error ±{fmt_delta(pm.fit['rmse'])}."
        )

    peak = st.selectbox("Pick a peak", list(PEAKS["peak"]), index=list(PEAKS["peak"]).index("Tecumseh"))
    row = PEAKS[PEAKS["peak"] == peak].iloc[0]
    st.caption(f"Reference route: {row['route_note']} — {row['route_miles']} mi round trip, {row['route_gain_ft']:,} ft of climb. "
               "Peaks that share a route share the same numbers. Edit them below if you take a different route.")
    c1, c2, c3 = st.columns(3)
    miles = c1.number_input("Round-trip miles", value=float(row["route_miles"]), min_value=0.5, step=0.5, key=f"mi-{peak}")
    gain = c2.number_input("Elevation gain (ft)", value=float(row["route_gain_ft"]), min_value=0.0, step=100.0, key=f"g-{peak}")
    start = c3.time_input("Start time", value=pd.Timestamp("07:00").time())

    hrs = pm.hours(miles, gain)
    lo, hi = max(hrs - 1.28 * pm.spread, 0.25), hrs + 1.28 * pm.spread
    finish = (pd.Timestamp.combine(pd.Timestamp.today().date(), start) + pd.Timedelta(hours=hrs)).strftime("%-I:%M %p")
    st.markdown(f"## Hiking **{peak}** will take you about **{fmt_hours(hrs)}**")
    if pm.kind == "moving+stops":
        mv, sp = pm.parts(miles, gain)
        st.caption(f"≈ {fmt_hours(mv)} moving + {fmt_hours(sp)} stopped (rests, summit, photos).")
    st.markdown(
        f"Likely range **{fmt_hours(lo)} – {fmt_hours(hi)}** (a rough 80% range based on how far off the model is on your own hikes) · "
        f"starting at {start.strftime('%-I:%M %p')} you'd finish around **{finish}**. "
        f"Plan for the high end."
    )

    past = hikes[hikes["peaks"].apply(lambda ps: peak in ps)]
    if len(past):
        r = past.iloc[-1]
        st.info(f"You've hiked {peak} before ({r['date']:%b %d, %Y}): it took **{fmt_hours(r['time_hr'])}**.")

    with st.expander("All 48 peaks, ranked by predicted time"):
        tbl = PEAKS[["peak", "elevation_ft", "route_miles", "route_gain_ft"]].copy()
        tbl["predicted_hours"] = [pm.hours(m, g) for m, g in zip(tbl["route_miles"], tbl["route_gain_ft"])]
        done = {p for ps in hikes["peaks"] for p in ps}
        tbl["hiked"] = tbl["peak"].isin(done)
        tbl["predicted_time"] = tbl["predicted_hours"].map(fmt_hours)
        tbl = tbl[["peak", "elevation_ft", "route_miles", "route_gain_ft", "predicted_time", "predicted_hours", "hiked"]]
        st.dataframe(tbl.sort_values("predicted_hours"), hide_index=True, width="stretch",
                     column_config={"predicted_time": "Predicted time",
                                    "predicted_hours": st.column_config.NumberColumn("Hours (sortable)", format="%.1f"),
                                    "hiked": st.column_config.CheckboxColumn("Hiked?"),
                                    "route_miles": "Miles", "route_gain_ft": "Gain (ft)", "elevation_ft": "Summit (ft)"})
    st.caption(
        "Limits: route miles/gain are approximate single-route figures from public hiking sites — check a current "
        "guide. The model can't see weather, ice, stream crossings, group size or trail difficulty; steep, rocky "
        "terrain is slower than your average hike."
    )
