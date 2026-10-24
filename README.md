# ⛰️ NH 4000-Footer Pace Predictor

A Streamlit app that learns **your** hiking pace from Garmin files and predicts how long each of New Hampshire's
48 four-thousand-footers will take you.

**Live app:** _(paste your Streamlit Community Cloud URL here)_

## How it works
1. **Upload** Garmin data: `.fit` files, a `.zip` of them, or Garmin Connect's `Activities.csv`.
2. **Tag** which hikes were NH 4000-footers (auto-suggested from the activity title).
3. **Explore** columns with line / scatter / histogram charts.
4. **Model**: pick two inputs + a target, fit a linear regression, type values, get a prediction.
5. **Predict a peak**: the default model is `distance_mi + ascent_ft → time_hr`, applied to each peak's reference route.

With fewer than 5 hikes the app falls back to the Naismith rule scaled to your pace and says so on screen.

## Privacy
- Your Garmin data is **never committed** (`.gitignore` blocks `.fit` and `Activities.csv`) and is processed in memory only.
- From `.fit` files only distance, climb, time, date and heart-rate summaries are read: no GPS track, name or weight.
- `data/demo_activities_SYNTHETIC.csv` is **fake** data for trying the app.

## Data
- `data/nh4000_peaks.csv`: peak, elevation and prominence (from my NH 4000 checklist) plus an *approximate* single-route
  round-trip distance and elevation gain gathered from public hiking sites (Trailspotting / AllTrails). Peaks that share a
  route share the same numbers. Double-check against a current guidebook.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
Push this folder to a **public** GitHub repo, go to https://share.streamlit.io, choose *Create app*, pick the repo,
branch `main`, main file `app.py`, then Deploy.

## Files
`app.py` UI · `garmin_io.py` .fit/CSV parsing + cleaning · `model.py` features, regression, fallback
