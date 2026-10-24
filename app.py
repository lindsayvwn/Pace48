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

from garmin_io import hikes_only, load_uploads
from model import FEATURES, MIN_HIKES_FOR_REGRESSION, TARGET, PaceModel, fit_regression, predict

HERE = Path(__file__).parent
PEAKS = pd.read_csv(HERE / "data" / "nh4000_peaks.csv")
DEMO_CSV = HERE / "data" / "demo_activities_SYNTHETIC.csv"
ACCENT = "#2E6F57"

st.set_page_config(page_title="NH 4000-Footer Pace Predictor", page_icon="⛰️", layout="wide")

LABELS = {
    "distance_mi": "Distance (mi)", "ascent_ft": "Elevation gain (ft)", "descent_ft": "Elevation loss (ft)",
    "time_hr": "Total time (hr)", "avg_hr": "Avg heart rate (bpm)", "max_hr": "Max heart rate (bpm)",
    "calories": "Calories",
}


def fmt_hours(h: float) -> str:
    m = int(round(h * 60))
    return f"{m // 60} h {m % 60:02d} min"


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
st.title("⛰️ NH 4000-Footer Pace Predictor")
st.info(
    "**What it's for and what happens when it's wrong:** this model estimates how long a New Hampshire "
    "4,000-footer will take *you*, based on the pace in your own past hikes — and if it's wrong (especially "
    "too low), you could start too late and get caught by darkness or weather above treeline, so use it as a "
    "planning aid, add a safety buffer, and always carry a headlamp."
)

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1 · Your data")
    uploads = st.file_uploader(
        "Upload Garmin files", type=["fit", "zip", "csv"], accept_multiple_files=True,
        help="Any mix of: .fit files (one per activity), a .zip of .fit files, "
             "or the Activities.csv from Garmin Connect.",
    )
    use_demo = st.checkbox("No Garmin data? Use synthetic demo data", value=False)
    c1, c2 = st.columns(2)
    dist_unit = c1.radio("CSV distance", ["mi", "km"], horizontal=True)
    elev_unit = c2.radio("CSV elevation", ["ft", "m"], horizontal=True)
    with st.expander("How do I get my files from Garmin?"):
        st.markdown(
            "**One hike (.fit):** garminconnect.com → open the activity → ⚙️ → *Export Original*. "
            "You get a .zip containing a .fit — upload the zip as-is.\n\n"
            "**Many activities (CSV):** Garmin Connect → Activities → *Export CSV*.\n\n"
            "**Everything:** Garmin account → *Data Management* → *Export Your Data* (a big zip)."
        )
    st.caption(
        "🔒 Files are read in memory for this session only. From .fit files the app keeps just distance, "
        "climb, time, date and heart-rate summaries — **no GPS, name or weight**. Nothing is saved on a server."
    )
    st.caption(
        "Why no Garmin login? Garmin only allows account connections for approved companies, and asking for "
        "your password in a public app would be unsafe. Uploading gives you full control over which activities "
        "the app sees."
    )

# ------------------------------------------------------------- load data
files = [(f.name, f.getvalue()) for f in (uploads or [])]
if use_demo:
    files.append((DEMO_CSV.name, DEMO_CSV.read_bytes()))


@st.cache_data(show_spinner="Reading Garmin files…")
def _load(file_tuple, du, eu):
    return load_uploads(list(file_tuple), du, eu)


if not files:
    st.markdown("### 👈 Start by uploading a Garmin file (or tick the demo-data box)")
    st.markdown(
        "1. Upload hikes from Garmin  \n2. Tag which ones were NH 4000-footers  \n"
        "3. Explore your pace, then get a predicted time for any peak"
    )
    with st.expander("Preview the 48 NH 4000-footers reference list"):
        st.dataframe(PEAKS, hide_index=True, width="stretch")
    st.stop()

activities, notes = _load(tuple(files), dist_unit, elev_unit)
for n in notes:
    st.warning(n)
if activities.empty:
    st.error("I couldn't find any activities in those files.")
    st.stop()

types = sorted(activities["activity_type"].unique())
default_types = [t for t in types if "hik" in t.lower()] or types
chosen_types = st.sidebar.multiselect(
    "Activity types to use", types, default=default_types,
    help="Hiking is the default. Running pace is usually much faster than hiking pace, "
         "so mixing types can make predictions too optimistic.",
)
hikes = activities[activities["activity_type"].isin(chosen_types)].reset_index(drop=True)
if hikes.empty:
    st.error("No activities of the selected types.")
    st.stop()
if not all("hik" in t.lower() for t in chosen_types):
    st.sidebar.warning("Non-hiking activities selected: predictions may be too optimistic.")

# --------------------------------------------------------- peak tagging
data_key = f"{len(hikes)}-{hikes['distance_mi'].round(2).sum():.2f}"
hikes = hikes.assign(peaks=[suggest_peaks(t) for t in hikes["title"]])

tab1, tab2, tab3, tab4 = st.tabs(
    ["① Preview & tag peaks", "② Explore", "③ Build a model", "④ Predict an NH 4000-footer"]
)

# ============================================================ TAB 1
with tab1:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Activities", len(hikes))
    m2.metric("Total miles", f"{hikes['distance_mi'].sum():,.0f}")
    m3.metric("Total climb (ft)", f"{hikes['ascent_ft'].sum():,.0f}")
    m4.metric("Total hours", f"{hikes['time_hr'].sum():,.0f}")
    st.markdown(
        "**Was any of these a NH 4000-footer?** Pick the peak(s) in the *NH 4000 peaks* column "
        "(some are pre-filled from the title). It's optional for the model, but lets the predictor compare "
        "against your real times."
    )
    show_cols = ["date", "title", "activity_type", "distance_mi", "ascent_ft", "time_hr", "avg_hr", "peaks"]
    edited = st.data_editor(
        hikes[show_cols].rename(columns={"peaks": "nh4000"}),
        key=f"editor-{data_key}",
        hide_index=True, width="stretch",
        disabled=[c for c in show_cols if c != "peaks"],
        column_config={
            "date": st.column_config.DateColumn("Date", format="MMM D, YYYY"),
            "title": "Title", "activity_type": "Type",
            "distance_mi": st.column_config.NumberColumn("Miles", format="%.1f"),
            "ascent_ft": st.column_config.NumberColumn("Climb (ft)", format="%.0f"),
            "time_hr": st.column_config.NumberColumn("Hours", format="%.2f"),
            "avg_hr": st.column_config.NumberColumn("Avg HR", format="%.0f"),
            "nh4000": st.column_config.MultiselectColumn("NH 4000 peaks", options=list(PEAKS["peak"])),
        },
    )
    hikes["peaks"] = edited["nh4000"].apply(lambda v: list(v) if isinstance(v, (list, tuple, np.ndarray)) else []).values

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
        k3.metric("Typical error (RMSE)", f"{fit['rmse']:.2f}")
        k4.metric("Leave-one-out error", "n/a (need 6+)" if fit["loo_mae"] is None else f"{fit['loo_mae']:.2f}",
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
            f"Linear regression trained on **{pm.n} hikes** · R² {pm.fit['r2']:.2f} · typical error ±{pm.fit['rmse']:.2f} h."
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
    st.markdown(f"## Hiking **{peak}** will take you about **{hrs:.1f} hours**")
    st.markdown(
        f"Likely range **{lo:.1f}–{hi:.1f} h** (a rough 80% range based on how far off the model is on your own hikes) · "
        f"starting at {start.strftime('%-I:%M %p')} you'd finish around **{finish}**. "
        f"Plan for the high end."
    )

    past = hikes[hikes["peaks"].apply(lambda ps: peak in ps)]
    if len(past):
        r = past.iloc[-1]
        st.info(f"You've hiked {peak} before ({r['date']:%b %d, %Y}): it took **{r['time_hr']:.1f} h**.")

    with st.expander("All 48 peaks, ranked by predicted time"):
        tbl = PEAKS[["peak", "elevation_ft", "route_miles", "route_gain_ft"]].copy()
        tbl["predicted_hours"] = [pm.hours(m, g) for m, g in zip(tbl["route_miles"], tbl["route_gain_ft"])]
        done = {p for ps in hikes["peaks"] for p in ps}
        tbl["hiked"] = tbl["peak"].isin(done)
        st.dataframe(tbl.sort_values("predicted_hours"), hide_index=True, width="stretch",
                     column_config={"predicted_hours": st.column_config.NumberColumn("Predicted hours", format="%.1f"),
                                    "hiked": st.column_config.CheckboxColumn("Hiked?"),
                                    "route_miles": "Miles", "route_gain_ft": "Gain (ft)", "elevation_ft": "Summit (ft)"})
    st.caption(
        "Limits: route miles/gain are approximate single-route figures from public hiking sites — check a current "
        "guide. The model can't see weather, ice, stream crossings, group size or trail difficulty; steep, rocky "
        "terrain is slower than your average hike."
    )
