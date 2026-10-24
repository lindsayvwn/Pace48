"""NH 4000-Footer Pace Predictor — a Streamlit app.

Upload your own Garmin hikes -> tag the ones that were NH 4000-footers ->
the app learns YOUR pace and predicts how long any of the 48 peaks will take you.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from garmin_io import activity_key, load_activities
from tags_io import tags_from_csv, tags_to_csv
from model import FEATURES, MIN_HIKES_FOR_REGRESSION, TARGET, PaceModel, fit_regression, predict

HERE = Path(__file__).parent
PEAKS = pd.read_csv(HERE / "data" / "nh4000_peaks.csv")
DEMO_CSV = HERE / "data" / "demo_activities_SYNTHETIC.csv"
ACCENT = "#2E6F57"

st.set_page_config(page_title="NH 4000-Footer Pace Predictor", page_icon="🏔️", layout="wide")

LABELS = {
    "distance_mi": "Distance (mi)", "ascent_ft": "Elevation gain (ft)", "descent_ft": "Elevation loss (ft)",
    "time_hr": "Total time (hr)", "avg_hr": "Avg heart rate (bpm)", "max_hr": "Max heart rate (bpm)",
    "calories": "Calories",
}


def fmt_hours(h: float) -> str:
    """3.04 -> '3 h 02 min' (used for anything the viewer reads as a time)."""
    m = int(round(h * 60))
    return f"{m // 60} h {m % 60:02d} min"


def fmt_delta(h: float) -> str:
    """Short durations: 0.45 -> '27 min'; 1.6 -> '1 h 36 min'."""
    m = int(round(h * 60))
    return f"{m} min" if m < 60 else fmt_hours(h)


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


# ------------------------------------------------------------------ header
st.markdown(HIDE_DEFAULT_STATUS, unsafe_allow_html=True)
st.title("🥾 NH 4000-Footer Pace Predictor")
st.info(
    "**What it's for and what happens when it's wrong:** this model estimates how long a New Hampshire "
    "4,000-footer will take *you*, based on the pace in your own past hikes — and if it's wrong (especially "
    "too low), you could start too late and get caught by darkness or weather above treeline, so use it as a "
    "planning aid, add a safety buffer, and always carry a headlamp."
)

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Your data")
    csv_up = st.file_uploader(
        "① Activities.csv (required)", type=["csv"],
        help="Garmin Connect → Activities → Export CSV. It supplies your activity titles.",
    )
    fit_ups = st.file_uploader(
        "② Activity .zip / .fit files (recommended)", type=["zip", "fit"], accept_multiple_files=True,
        help="Garmin's CSV often leaves climb blank. The .fit files carry the real distance, climb and time, "
             "and are matched to the CSV by start time.",
    )
    tags_up = st.file_uploader(
        "③ Saved peak tags (optional)", type=["csv"],
        help="The file you downloaded from this app last time. It restores your NH 4000 tags and exclusions.",
    )
    use_demo = st.checkbox("No Garmin data? Use synthetic demo data", value=False)
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
        "climb, time, date and heart-rate summaries: **no GPS, name or weight**. Nothing is saved on a server."
    )
    st.caption(
        "Why no Garmin login? Garmin only allows account connections for approved companies, and asking for "
        "your password in a public app would be unsafe. Uploading gives you full control over which activities "
        "the app sees."
    )

# ------------------------------------------------------------- load data
csv_bytes = csv_up.getvalue() if csv_up else (DEMO_CSV.read_bytes() if use_demo else None)
fit_files = [(f.name, f.getvalue()) for f in (fit_ups or [])]

if csv_bytes is None:
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
def _load(csv_b, fit_tuple, du, eu):
    return load_activities(csv_b, list(fit_tuple), du, eu)


loader = st.empty()
loader.markdown(LOADER_HTML, unsafe_allow_html=True)
try:
    activities, notes = _load(csv_bytes, tuple(fit_files), dist_unit, elev_unit)
except ValueError as e:
    loader.empty()
    st.error(str(e))
    st.stop()
loader.empty()
for n in notes:
    st.warning(n)
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
else:
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
if all_rows.empty:
    st.error("No activities of the selected types.")
    st.stop()
if not all("hik" in t.lower() for t in chosen_types):
    st.sidebar.warning("Non-hiking activities selected: predictions may be too optimistic.")

# --------------------------------------------------------- tags that survive new uploads
# st.session_state["tags"] maps activity key -> {"peaks": [...], "include": bool, ...}.
# The key is the activity's start time, so it stays valid when you add more files later.
tags: dict = st.session_state.setdefault("tags", {})
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

tab1, tab2, tab3, tab4 = st.tabs(
    ["① Preview, tag & exclude", "② Explore", "③ Build a model", "④ Predict an NH 4000-footer"]
)

# ============================================================ TAB 1
with tab1:
    # Base table is rebuilt only when the data/tags change, so edits made in the grid aren't reset.
    editor_key = f"editor-{hash(tuple(all_rows['key']))}-{st.session_state.get('tags_version', 0)}"
    base_key = f"base-{editor_key}"
    if base_key not in st.session_state:
        st.session_state[base_key] = pd.DataFrame({
            "include": [tags[k]["include"] for k in all_rows["key"]],
            "date": all_rows["date"], "title": all_rows["title"], "activity_type": all_rows["activity_type"],
            "distance_mi": all_rows["distance_mi"], "ascent_ft": all_rows["ascent_ft"],
            "time_hr": all_rows["time_hr"], "avg_hr": all_rows["avg_hr"],
            "nh4000": [tags[k]["peaks"] for k in all_rows["key"]],
        })
    st.markdown(
        "**Tick the activities to use, and tag any NH 4000-footers.** Untick *Use* to leave an activity out of every "
        "chart and model. Pick peak(s) in the *NH 4000 peaks* column (some are pre-filled from the title)."
    )
    edited = st.data_editor(
        st.session_state[base_key],
        key=editor_key, hide_index=True, width="stretch",
        disabled=["date", "title", "activity_type", "distance_mi", "ascent_ft", "time_hr", "avg_hr"],
        column_config={
            "include": st.column_config.CheckboxColumn("Use", default=True),
            "date": st.column_config.DateColumn("Date", format="MMM D, YYYY"),
            "title": "Title", "activity_type": "Type",
            "distance_mi": st.column_config.NumberColumn("Miles", format="%.1f"),
            "ascent_ft": st.column_config.NumberColumn("Climb (ft)", format="%.0f"),
            "time_hr": st.column_config.NumberColumn("Hours", format="%.2f"),
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
            "Date": df["date"], "Activity": df["title"],
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
        "💾 Save my peak tags & exclusions", data=tags_to_csv(tags), file_name="my_nh4000_tags.csv", mime="text/csv",
        help="Next visit, upload this file in the sidebar (step ③) and your tags come right back.",
    )
    st.caption("Tip: this app can't remember you between visits, so download this file when you're done tagging.")

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

# ============================================================ TAB 3
with tab3:
    st.subheader("Fit a linear regression: two inputs → one target")
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
        st.markdown("**Try it — type in values:**")
        d1, d2 = st.columns(2)
        v1 = d1.number_input(LABELS[in1], value=float(round(model_df[in1].median(), 1)), min_value=0.0, key=f"v1-{in1}")
        v2 = d2.number_input(LABELS[in2], value=float(round(model_df[in2].median(), 1)), min_value=0.0, key=f"v2-{in2}")
        pred = predict(fit, {in1: v1, in2: v2})
        if target == TARGET:
            st.success(f"Predicted **{LABELS[target].replace(' (hr)', '')}: {fmt_hours(pred)}**")
        else:
            st.success(f"Predicted **{LABELS[target]}: {pred:.2f}**")
        tmp = model_df[feats + [target]].dropna().copy()
        tmp["predicted"] = fit["model"].predict(tmp[feats])
        f2 = px.scatter(tmp, x="predicted", y=target, labels={"predicted": f"Predicted {LABELS[target]}", target: f"Actual {LABELS[target]}"},
                        color_discrete_sequence=[ACCENT])
        lo, hi = float(tmp[[target, "predicted"]].min().min()), float(tmp[[target, "predicted"]].max().max())
        f2.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dot", color="gray"))
        f2.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=360)
        st.plotly_chart(f2, width="stretch")
        st.caption("Dots on the dotted line = perfect predictions.")

# ============================================================ TAB 4
with tab4:
    st.subheader("How long will an NH 4000-footer take you?")
    pm = PaceModel(model_df)
    if pm.n == 0:
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
